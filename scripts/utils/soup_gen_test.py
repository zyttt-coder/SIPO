from dataclasses import dataclass, field
from typing import Optional
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
    soup_weight: float = field(default=None, metadata={"help": "If None, disable dpo soup. If a float between 0.0 and 1.0, represent the weight of the first adatper model"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})

    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=500, metadata={"help": "number of prompts for generations"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    batch_size: Optional[int] = field(default=8)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    # base model
    print_local_main("loading model...")
    if len(script_args.sft_model_name.split(",")) > 1:
        assert len(script_args.sft_model_name.split(",")) == 2
        sft_model_name, sft_model1_name = script_args.sft_model_name.split(",")
        sft_model_name = sft_model_name.strip()
        sft_model1_name = sft_model1_name.strip()
    else:
        sft_model_name = script_args.sft_model_name
        sft_model1_name = script_args.sft_model_name
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
        torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
        device_map="auto",
        cache_dir=".cache",
    )
    if script_args.soup_weight:
        try:
            assert script_args.adapter_model_name != None and 0.0 <= script_args.soup_weight <= 1.0
            paths = script_args.adapter_model_name.split(",")
            assert len(paths) == 2
            adapter_model1, adapter_model2 = paths
            adapter_model1 = adapter_model1.strip()
            adapter_model2 = adapter_model2.strip()
        except Exception as e:
            raise e
        sft_model1 = AutoModelForCausalLM.from_pretrained(
            sft_model1_name,
            use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
            torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
            device_map="auto",
            cache_dir=".cache",
        )
        adapter1 = PeftModel.from_pretrained(sft_model, adapter_model1)
        adapter2 = PeftModel.from_pretrained(sft_model1, adapter_model2)

        if sft_model_name != sft_model1_name:
            adapter1.eval()
            adapter2.eval()
            adapter1 = adapter1.merge_and_unload()
            adapter2 = adapter2.merge_and_unload()
            combined_state_dict = {}
            for key in adapter1.state_dict().keys():
                adapter1_weight = adapter1.state_dict()[key]
                adapter2_weight = adapter2.state_dict()[key]
                combined_state_dict[key] = script_args.soup_weight * adapter1_weight + (1.0 - script_args.soup_weight) * adapter2_weight
            model = sft_model
            model.load_state_dict(combined_state_dict)
        else:
            # Extract the adapter state dictionaries
            adapter1_state_dict = get_peft_model_state_dict(adapter1)
            adapter2_state_dict = get_peft_model_state_dict(adapter2)
            combined_state_dict = combine_peft_state_dict(adapter1_state_dict, adapter2_state_dict, script_args.soup_weight)
            # Load the combined weights into the base SFT model
            model = adapter1
            set_peft_model_state_dict(model, combined_state_dict)
    else:
        if script_args.adapter_model_name:
            model = PeftModel.from_pretrained(sft_model, script_args.adapter_model_name)
        else:
            model = sft_model # sft

    # tokenizer: left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # dataset
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](prompt_template=script_args.prompt_template)
    if script_args.eval_size > 0:
        eval_dataset  = rdp.get_sft_dataset(split="test").select(range(script_args.eval_size))
    else:
        eval_dataset  = rdp.get_sft_dataset(split="test")

    split_size = math.ceil(len(eval_dataset) /script_args.world_size)
    eval_dataset = eval_dataset.select(range(
        script_args.rank*split_size, 
        min((script_args.rank+1)*split_size, len(eval_dataset))
    ))
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(script_args.rank+1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
    )

    results = []
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
        for sample in output:
            results.append({'prompt_response': sample})
        

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
