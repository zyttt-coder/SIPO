from dataclasses import dataclass, field
from typing import Optional
import os
import math

import torch
import tyro
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset

from src.utils.utils import (
    disable_progress_bar_non_local_main
)
from scripts.utils.utils import extract_prompt_content

disable_progress_bar_non_local_main()

ALPACA_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
SCORE_ATTR = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']

@dataclass
class ScriptArguments:

    input_dir: Optional[str] = field(default=None, metadata={"help": "input path for generations"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "output path for scores"})

    prompt_template: Optional[str] = field(default=ALPACA_TEMPLATE, metadata={"help": "the prompt template"})


if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)

    rm = AutoModelForSequenceClassification.from_pretrained("RLHFlow/RewardModel-Mistral-7B-for-DPA-v1", trust_remote_code=True, device_map='auto')
    rm.config.sliding_window = 4096
    tokenizer = AutoTokenizer.from_pretrained("RLHFlow/RewardModel-Mistral-7B-for-DPA-v1")

    input_template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"

    generation = load_dataset(script_args.input_dir, split="train")

    results = []
    with torch.no_grad():
        for prompt_response in tqdm.tqdm(generation['prompt_response']):
            raw_prompt = extract_prompt_content(script_args.prompt_template, prompt_response)
            response = extract_prompt_content(script_args.prompt_template, prompt_response, extract_after=True)
            model_inputs = tokenizer(input_template.format(prompt=raw_prompt, response=response), return_tensors="pt").to('cuda')
            score = rm(**model_inputs).logits.squeeze().cpu().float().numpy()
            helpsteer_rewards_pred = (score[:5]-10)/20
            score_dict = dict(zip(SCORE_ATTR,helpsteer_rewards_pred))
            results.append({
                "prompt_response": prompt_response,
                "helpfulness": score_dict["helpfulness"],
                "correctness": score_dict["correctness"],
                "verbosity": score_dict["verbosity"],
            })
    
    # raw
    dataset = Dataset.from_list(results)
    dataset.to_json(os.path.join(script_args.output_dir, "raw.jsonl"))

    # mean
    helpfulness = [result["helpfulness"] for result in results]
    correctness = [result["correctness"] for result in results]
    verbosity = [result["verbosity"] for result in results]
    mean_helpfulness = sum(helpfulness) / len(helpfulness)
    mean_correctness = sum(correctness) / len(correctness)
    mean_verbosity = sum(verbosity) / len(verbosity)
    with open(os.path.join(script_args.output_dir, "mean.csv"), "w") as f:
        f.write("mean helpfulness, mean correctness, mean verbosity\n")
        f.write(f"{mean_helpfulness}, {mean_correctness}, {mean_verbosity}\n")
