from dataclasses import dataclass, field
from typing import Optional, List
import os

import tyro
from datasets import Dataset
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE



@dataclass
class ScriptArguments:

    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K", metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})

    eval_size: Optional[int] = field(default=1000, metadata={"help": "number of prompts for generations"})
    batch_size: Optional[int] = field(default=1)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)

if __name__ == "__main__":

    script_args = tyro.cli(ScriptArguments)

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