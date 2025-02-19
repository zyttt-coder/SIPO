from dataclasses import dataclass, field
from typing import Optional, List
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from datasets import Dataset, load_dataset

from src.data.configs import DEFAULT_PROMPT_TEMPLATE, DATASET_CONFIGS
from src.utils.utils import (
    disable_progress_bar_non_local_main, PeftAsPreTrained, print_local_main, set_seeds
)
from src.utils.reward import ImplicitRewardWrapper, RewardWrapperInput, RewardWrapperList
from scripts.utils.utils import extract_prompt_content, combine_peft_state_dict

disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    adapter_model_name: str = field(default=None, metadata={"help": "lora name"})
    soup_weights: Optional[List[float]] = field(default=None, metadata={"help": "list of weights for linear interpolation between adapter models"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})

    input_mod_dir: Optional[str] = field(default=None, metadata={"help": "input path for mod generations"})
    input_dir: Optional[str] = field(default=None, metadata={"help": "input path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)

    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )

    replication: Optional[int] = field(default=3, metadata={"help": "number of responses generated for one prompt"})

if __name__ == "__main__":
    script_args = tyro.cli(ScriptArguments)
    assert script_args.peft, "only support peft model as reward model"
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
    assert script_args.adapter_model_name != None
    if script_args.soup_weights:
        paths = script_args.adapter_model_name.split(",")
        assert len(paths) == 2
        adapter_model1, adapter_model2 = paths
        adapter_model1 = adapter_model1.strip()
        adapter_model2 = adapter_model2.strip()
        sft_model = PeftModel.from_pretrained(sft_model, adapter_model1, adapter_name="reward_model_0")
        sft_model.load_adapter(adapter_model2, adapter_name="reward_model_1")
        
        # Extract the adapter state dictionaries
        adapter1_state_dict = get_peft_model_state_dict(sft_model, adapter_name="reward_model_0")
        adapter2_state_dict = get_peft_model_state_dict(sft_model, adapter_name="reward_model_1")
        
        for weight_idx in range(len(script_args.soup_weights)):
            weight = script_args.soup_weights[weight_idx]
            combined_state_dict = combine_peft_state_dict(adapter1_state_dict, adapter2_state_dict, weight)
            sft_model.add_adapter(f"reward_model_{weight_idx+2}", script_args.peft_config)
            set_peft_model_state_dict(sft_model, combined_state_dict, adapter_name=f"reward_model_{weight_idx+2}")
    else:
        RM_paths = script_args.adapter_model_name.split(",")
        RM_paths = [path.strip() for path in RM_paths]
        sft_model = PeftModel.from_pretrained(sft_model, RM_paths[0], adapter_name="reward_model_0")
        if len(RM_paths) > 1:
            for idx, path in enumerate(RM_paths[1:], start=1):
                sft_model.load_adapter(path, adapter_name=f"reward_model_{idx}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
        
    RM_list = []
    print("number of reward models:", len(sft_model.peft_config))
    for idx in range(len(sft_model.peft_config)):
        RM_list.append(
            ImplicitRewardWrapper(
                model=PeftAsPreTrained(sft_model, f"reward_model_{idx}"),
                ref_model=PeftAsPreTrained(sft_model), 
                tokenizer=tokenizer, 
                beta=script_args.beta,
                prompt_template=script_args.prompt_template,
            )
        )
    reward_model_list = RewardWrapperList(RM_list)

    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(script_args.rank+1).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
    )

    mod_generation = load_dataset(script_args.input_mod_dir, split="train")
    mod_prompt_response = mod_generation['prompt_response']
    generation = load_dataset(script_args.input_dir, split="train")
    prompt_response = generation['prompt_response']

    assert len(mod_generation) == len(generation), "dataset lengths should be equal"
    results = []
    rewrite_retain = 0
    for idx in tqdm.tqdm(range(0, len(generation))):
        example = prompt_response[idx]
        mod_example = mod_prompt_response[idx]
        raw_prompt = extract_prompt_content(script_args.prompt_template, example)
        mod_raw_prompt = extract_prompt_content(script_args.prompt_template, mod_example)
        response = extract_prompt_content(script_args.prompt_template, example, extract_after=True)
        mod_response = extract_prompt_content(script_args.prompt_template, mod_example, extract_after=True)
        assert raw_prompt == mod_raw_prompt, "prompts should be identical"
        mod_scores = reward_model_list(
            RewardWrapperInput(raw_prompt=[raw_prompt], response=[mod_response])
        )
        scores = reward_model_list(
            RewardWrapperInput(raw_prompt=[raw_prompt], response=[response])
        )
        if not all(score <= mod_score for score, mod_score in zip(scores, mod_scores)):
            results.append({'prompt_response': example})
            rewrite_retain += 1
        else:
            results.append({'prompt_response': mod_example})
    print("rewrite retain:%.2f"%(rewrite_retain/len(generation)))

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)