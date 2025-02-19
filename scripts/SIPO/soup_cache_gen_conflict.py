from dataclasses import dataclass, field
from typing import Optional, List
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils.utils import (
    print_local_main, disable_progress_bar_non_local_main, set_seeds
)
from scripts.utils.utils import combine_peft_state_dict

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    adapter_model_name: str = field(default=None, metadata={"help": "lora name"})
    soup_weights: Optional[List[float]] = field(default=None, metadata={"help": "list of weights for linear interpolation between adapter models"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})

    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=1000, metadata={"help": "number of prompts for generations"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    batch_size: Optional[int] = field(default=1)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)

if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    # base model
    print_local_main("loading model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
        torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
        device_map="auto",
        cache_dir=".cache",
    )
    if script_args.soup_weights:
        assert script_args.adapter_model_name != None
        paths = script_args.adapter_model_name.split(",")
        assert len(paths) == 2
        adapter_model1, adapter_model2 = paths
        adapter_model1 = adapter_model1.strip()
        adapter_model2 = adapter_model2.strip()
        sft_model = PeftModel.from_pretrained(sft_model, adapter_model1)
        sft_model.load_adapter(adapter_model2, adapter_name="adapter2")
        
        # Extract the adapter state dictionaries
        adapter1_state_dict = get_peft_model_state_dict(sft_model)
        adapter2_state_dict = get_peft_model_state_dict(sft_model, adapter_name="adapter2")

        sft_model.set_adapter("default")
        model = sft_model
    else:
        if script_args.adapter_model_name:
            model = PeftModel.from_pretrained(sft_model, script_args.adapter_model_name)
        else:
            model = sft_model # sft

    # tokenizer: left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # dataset
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](prompt_template=script_args.prompt_template)
    if script_args.eval_size > 0:
        # first {eval_size} conflicting samples in the training dataset
        eval_dataset  = rdp.get_preference_dataset(split="train_conflict").select(range(script_args.eval_size))
    else:
        eval_dataset  = rdp.get_preference_dataset(split="train_conflict")

    # cache the samples locally
    for attr in ["positive","negative"]:
        cached_path = f"./output/cached_datasets/{script_args.dataset_name}/{attr}"
        cached_path = os.path.join(cached_path, f"{str(script_args.rank+1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl")
        if os.path.exists(cached_path):
            with open(cached_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f)
            if row_count == script_args.eval_size:
                print(f"First {script_args.eval_size} conflicting samples in the training dataset already cached in {cached_path}")
                continue
        content = []
        for idx in range(0, len(eval_dataset), script_args.batch_size):
            batch = eval_dataset[idx: idx+script_args.batch_size]
            assert False not in batch["contradict"]
            for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
                content.append({
                    "prompt_response": prompt + chosen if attr == "positive" else prompt + rejected,
                })
        dataset = Dataset.from_list(content)
        dataset.to_json(cached_path)

    split_size = math.ceil(len(eval_dataset) /script_args.world_size)
    eval_dataset = eval_dataset.select(range(
        script_args.rank*split_size, 
        min((script_args.rank+1)*split_size, len(eval_dataset))
    ))
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(script_args.rank+1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
    )

    if script_args.soup_weights:
        results = [{} for _ in range(len(eval_dataset))]  # Initialize a list of dictionaries to store responses by index
    else:
        results = []
    for weight_idx in range(len(script_args.soup_weights) if script_args.soup_weights else 1):
        if script_args.soup_weights:
            weight = script_args.soup_weights[weight_idx]
            combined_state_dict = combine_peft_state_dict(adapter1_state_dict, adapter2_state_dict, weight)
            set_peft_model_state_dict(model, combined_state_dict)
        for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
            batch = eval_dataset[idx: idx+script_args.batch_size]
            prompt_tokenized = tokenizer(
                batch["prompt"], 
                return_tensors="pt", 
                padding=True,
            )
            output_tokenized = model.generate(
                input_ids=prompt_tokenized["input_ids"].cuda(),
                attention_mask=prompt_tokenized["attention_mask"].cuda(),
                max_length=script_args.max_length,
            )
            output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)

            if script_args.soup_weights:
                for i, sample in enumerate(output):
                    results[idx + i][f"weight_{weight_idx}"] = sample
            else:
                for sample in output:
                    results.append({'prompt_response': sample})

    if script_args.soup_weights:
        results = [{"prompt_response": res[f"weight_{w}"]} for res in results for w in range(len(script_args.soup_weights))]

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
