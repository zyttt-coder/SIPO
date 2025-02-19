import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import math
import tyro
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from datasets import Dataset
from peft import PeftModel
from accelerate import Accelerator
from src.utils.utils import print_local_main, disable_progress_bar_non_local_main, set_seeds, param_sharding_enabled
from src.utils.util_decode import FusionModel
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE


disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft model name"})
    dpo_model_1_name: str = field(metadata={"help": "the dpo model 1 name"})
    dpo_model_2_name: str = field(metadata={"help": "the dpo model 2 name"})
    num_beams: int = field(default=1, metadata={"help": "the number of beams"})
    seed: int = field(default=42, metadata={"help": "the seed"})
    f_type: str = field(default="reverse_kl")
    split: str = field(default="test", metadata={"help": "split of dataset questions to sample from"})
    soup_weights: Optional[List[float]] = field(default=None, metadata={"help": "list of weights for linear interpolation between adapter models"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    eval_size: Optional[int] = field(default=500, metadata={"help": "number of prompts for generations"})
    batch_size: Optional[int] = field(default=1)
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})

if __name__ == "__main__":
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)

    # base model
    print_local_main("loading model...")
    if not param_sharding_enabled():
        accelerator = Accelerator()
        rank = accelerator.local_process_index
        world_size = accelerator.num_processes
        print_local_main(f"num processes: {world_size}")
    else:
        accelerator = None
        rank = script_args.rank
        world_size = script_args.world_size

    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
        torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
        **({"device_map": {"": rank}} if accelerator else "auto"),
        cache_dir=".cache",
    )
    sft_model.config.update({"pad_token_id": sft_model.config.eos_token_id})
    sft_model = PeftModel.from_pretrained(sft_model, script_args.dpo_model_1_name, "model_0")
    sft_model.load_adapter(script_args.dpo_model_2_name, "model_1")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dataset
    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check,
    )
    if script_args.eval_size > 0:
        eval_dataset  = rdp.get_preference_dataset(split=script_args.split).select(range(script_args.eval_size))
    else:
        eval_dataset  = rdp.get_preference_dataset(split=script_args.split)

    split_size = math.ceil(len(eval_dataset) /world_size)
    eval_dataset = eval_dataset.select(range(
        rank*split_size, 
        min((rank+1)*split_size, len(eval_dataset))
    ))
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(rank+1).zfill(5)}-of-{str(world_size).zfill(5)}.jsonl"
    )

    if script_args.soup_weights:
        results = [{} for _ in range(len(eval_dataset))]  # Initialize a list of dictionaries to store responses by index
    else:
        results = []
    for weight_idx in range(len(script_args.soup_weights) if script_args.soup_weights else 1):
        if script_args.soup_weights:
            weight = script_args.soup_weights[weight_idx]
            sample_model = FusionModel(sft_model, [weight, 1.0-weight], f_type=script_args.f_type)
        else:
            raise NotImplementedError
        for idx in tqdm.tqdm(range(0, len(eval_dataset), script_args.batch_size)):
            batch = eval_dataset[idx: idx+script_args.batch_size]
            prompt_tokenized = tokenizer(
                batch["prompt"], 
                return_tensors="pt", 
                padding=True,
            )
            output_tokenized = sample_model.generate(
                input_ids=prompt_tokenized["input_ids"].to(sft_model.device),
                attention_mask=prompt_tokenized["attention_mask"].to(sft_model.device),
                max_length=script_args.max_length,
                do_sample=False,
                num_beams=script_args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
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