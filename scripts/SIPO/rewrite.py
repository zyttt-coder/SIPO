import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import math
import tyro
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from datasets import Dataset, load_dataset
from peft import PeftModel
from accelerate import Accelerator
from src.utils.utils import print_local_main, disable_progress_bar_non_local_main, set_seeds, param_sharding_enabled
from src.utils.util_decode import FusionModel
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from scripts.utils.utils import extract_prompt_content

# PROMPT = '''BEGINNING OF CONVERSATION: USER: Question: {raw_prompt}\nResponse: {response}\nReview: {review}\nRevise the 'Response' based on the 'Review' to improve its correctness and verbosity. ASSISTANT:'''
PROMPT = '''BEGINNING OF CONVERSATION: USER: Question: What is the most common use for dill in home cooking?\nResponse: Dill is a versatile herb that can be used in a variety of dishes to add flavor and aroma. It is commonly used in Indian, Middle Eastern, and Scandinavian cuisines, and is often used to flavor soups, stews, and curries. Dill is also used to make pickles, sauces, and marinades, and is a popular ingredient in many vegetarian and vegan dishes. In addition to its culinary uses, dill is also known for its medicinal properties, and is often used to treat digestive issues, respiratory problems, and skin conditions.\nReview: The response can be more correct and verbose by\nIncluding specific recipes that use dill as a main ingredient or flavor enhancer.\nExplaining the history and cultural significance of dill in different cuisines.\nDescribing the medicinal properties of dill and its use in traditional medicine.\nRevise the 'Response' based on the 'Review' to improve its correctness and verbosity.\nASSISTANT:Dill is a versatile herb widely used in home cooking for its fresh, slightly tangy flavor and aromatic qualities. One of its most common uses is in making pickles, where its distinct taste enhances the brine. It is also frequently used in soups, stews, and sauces, such as the classic Greek tzatziki, Scandinavian gravlax sauce, and Russian dill-infused borscht. In Middle Eastern and Indian cuisines, dill is often added to rice dishes, curries, and yogurt-based dips.\nBeyond its culinary applications, dill has a long history of use in traditional medicine. Ancient Egyptians and Greeks valued it for its digestive benefits, and it has been used to alleviate bloating, indigestion, and colic. Dill also contains compounds with antimicrobial and anti-inflammatory properties, making it a natural remedy for respiratory and skin conditions.\nOverall, dill's unique flavor and health benefits make it a staple in kitchens worldwide, enhancing both traditional and contemporary dishes. \n\n\nQuestion: {raw_prompt}\nResponse: {response}\nReview: {review}\nRevise the 'Response' based on the 'Review' to improve its correctness and verbosity. ASSISTANT:'''

disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft model name"})
    dpo_model_1_name: str = field(metadata={"help": "the dpo model 1 name"})
    dpo_model_2_name: str = field(metadata={"help": "the dpo model 2 name"})
    soup_weights: List[float] = field(metadata={"help": "list of weights for linear interpolation between adapter models"})
    num_beams: int = field(default=1, metadata={"help": "the number of beams"})
    seed: int = field(default=42, metadata={"help": "the seed"})
    f_type: str = field(default="reverse_kl")
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})

    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    input_dir: Optional[str] = field(default=None, metadata={"help": "input path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for generations"})
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
    
    replication = len(script_args.soup_weights)
    reviews = load_dataset(script_args.input_dir, split="train")

    assert len(reviews)%replication == 0, "number of replication is wrong"
    length = len(reviews)//replication

    split_size = math.ceil(length/world_size)*replication
    reviews = reviews.select(range(
        rank*split_size, 
        min((rank+1)*split_size, len(reviews))
    ))
    output_path = os.path.join(
        script_args.output_dir, 
        f"{str(rank+1).zfill(5)}-of-{str(world_size).zfill(5)}.jsonl"
    )

    results = [{} for _ in range(len(reviews)//replication)]
    for weight_idx in range(len(script_args.soup_weights)):
        weight = script_args.soup_weights[weight_idx]
        sample_model = FusionModel(sft_model, [weight, 1.0-weight], f_type=script_args.f_type)
        for idx in tqdm.tqdm(range(0, len(reviews)//replication)):
            idx_map = idx*replication + weight_idx
            raw_prompt = reviews['raw_prompt'][idx_map]
            response = reviews['response'][idx_map]
            review = reviews['review'][idx_map]
            prompt = PROMPT.format(
                raw_prompt=raw_prompt,
                response=response,
                review=review
            )
            prompt_tokenized = tokenizer(
                prompt, 
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
            output = output[0]
            rewrite_res = extract_prompt_content(script_args.prompt_template, output, extract_after=True)
            rewrite_res = rewrite_res.split("Question:")[0].strip()
            results[idx][f"weight_{weight_idx}"] = script_args.prompt_template.format(raw_prompt=raw_prompt)+rewrite_res

    results = [{"prompt_response": res[f"weight_{w}"]} for res in results for w in range(len(script_args.soup_weights))]

    dataset = Dataset.from_list(results)
    dataset.to_json(output_path)
