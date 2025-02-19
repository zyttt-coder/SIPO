from dataclasses import dataclass, field
from typing import Optional, List
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import PeftModel
from accelerate import Accelerator

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils.utils import (
    print_local_main, disable_progress_bar_non_local_main, set_seeds, param_sharding_enabled
)
from scripts.utils.utils import extract_prompt_content

disable_progress_bar_non_local_main()

TEMPLATE = '''BEGINNING OF CONVERSATION: USER: Question: What is the history of shipping and trade in the Mediterranean region, and how has it changed over time?\nResponse: The Mediterranean region has a long and rich history of shipping and trade, dating back to ancient times. In the early days, the region was a hub of trade and commerce, with goods and people traveling between different parts of the world. Over time, the region has undergone significant changes, with the rise of new powers and the development of new technologies. Today, the Mediterranean region is still an important center of trade and commerce, with a diverse range of goods and services being traded between different countries. However, the region has also faced significant challenges in recent years, including political instability, economic difficulties, and environmental degradation. These challenges have led to a decline in shipping and trade in the region, but there are also efforts underway to revitalize the region's economy and promote sustainable development.\nGenerate three suggestions on how to make the response more correct and verbose.\nASSISTANT:The response can be more correct and verbose by\nAdd Specific Examples - Include simple examples of civilizations like the Greeks, Romans, or Phoenicians to show their role in Mediterranean trade.\nExplain Key Changes Clearly - Break down major changes in shipping technology and trade methods into clear, easy-to-understand points.\nClarify Modern Challenges - Use straightforward language to describe current issues such as political instability and environmental concerns affecting trade.\n\n\nQuestion: {raw_prompt}\nResponse: {response}\nGenerate three suggestions on how to make the response more correct and verbose. ASSISTANT:The response can be more correct and verbose by\n'''


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    adapter_model_name: str = field(default=None, metadata={"help": "lora name"})
    # soup_weights: Optional[List[float]] = field(default=None, metadata={"help": "list of weights for linear interpolation between adapter models"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})

    input_dir: Optional[str] = field(default=None, metadata={"help": "input path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=0)

    replication: Optional[int] = field(default=3, metadata={"help": "number of responses generated for one prompt"})

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
    if script_args.adapter_model_name:
        model = PeftModel.from_pretrained(sft_model, script_args.adapter_model_name)
    else:
        model = sft_model # sft

    # tokenizer: left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    generation = load_dataset(script_args.input_dir, split="train")
    prompt_response = generation['prompt_response']

    assert len(prompt_response)%script_args.replication == 0, "number of replication is wrong"
    length = len(prompt_response)//script_args.replication

    split_size = math.ceil(length/world_size) * script_args.replication
    prompt_response = prompt_response[
        rank*split_size: 
        min((rank+1)*split_size, len(prompt_response))
    ]
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(rank+1).zfill(5)}-of-{str(world_size).zfill(5)}.jsonl"
    )
    
    results = []
    for idx in tqdm.tqdm(range(0, len(prompt_response)//script_args.replication)):
        gen_samples = prompt_response[idx*script_args.replication:(idx+1)*script_args.replication]
        for sample in gen_samples:
            raw_prompt = extract_prompt_content(script_args.prompt_template, sample)
            response = extract_prompt_content(script_args.prompt_template, sample, extract_after=True)
            prompt = TEMPLATE.format(raw_prompt=raw_prompt, response=response)
            prompt_tokenized = tokenizer(
                [prompt], 
                return_tensors="pt", 
                padding=True,
            )
            output_tokenized = model.generate(
                input_ids=prompt_tokenized["input_ids"].cuda(),
                attention_mask=prompt_tokenized["attention_mask"].cuda(),
                max_length=script_args.max_length,
            )
            output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
            review = output[0]
            review = extract_prompt_content(script_args.prompt_template, review, extract_after=True)
            review = review.split("Question:")[0].strip()
            results.append({'raw_prompt': raw_prompt, 'response': response, 'review': review})

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
