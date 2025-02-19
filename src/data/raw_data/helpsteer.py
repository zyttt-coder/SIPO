import os
from typing import Dict
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from typing import Literal, Optional

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .utils import RawDatasetPreprocessor
from src.utils.utils import print_local_main


@dataclass
class HelpSteerRDP(RawDatasetPreprocessor):
    path: Optional[str] = "./output/downloaded_datasets/HelpSteer_preprocessed"
    # None for sft
    dimension: Optional[Literal["overall", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]] = None

    def _get_raw_dataset(self, split):
        dataset = load_dataset("json", data_files={
            "train": os.path.join(self.path, "train.jsonl"),
            "validation": os.path.join(self.path, "val.jsonl"),
        })
        if split in ["train", "train_conflict", "train_nonconflict"]:
            return dataset["train"]
        elif split == "validation":
            return dataset["validation"]
        elif split == "test":
            return load_dataset("json", data_files={"test": os.path.join(self.path, "test.jsonl")})["test"]
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_id = example[f"{self.dimension}_chosen_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"], 
                                                    chosen=example[f"response_{chosen_id}"],
                                                    rejected=example[f"response_{1-chosen_id}"]),
        } if "{chosen}" in self.prompt_template and "{rejected}" in self.prompt_template else {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_id}"],
            "rejected": example[f"response_{1-chosen_id}"],
            "contradict": example["correctness_chosen_id"] != example["verbosity_chosen_id"]
        }

    def get_preference_dataset(self, split):
        assert self.dimension, "preference dimension has to be specified"
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("filtering preference...")
        if split == "test":
            raise NotImplementedError
        dataset = dataset.filter(lambda x: x[f"{self.dimension}_chosen_id"] != -1)
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, **kwargs):
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to sft...")
        if split == "test":
            return dataset.map(
                lambda sample: {
                    "raw_prompt": sample["prompt"],
                    "prompt": self.prompt_template.format(raw_prompt=sample["prompt"]), 
                },
                num_proc=self.num_proc,
                remove_columns=dataset.column_names,
            )
        else:
            raise NotImplementedError

@dataclass
class HelpSteerRDPWithConflict(HelpSteerRDP):
    conflicting_fraction: float = 0.0  # Default fraction of conflicting data

    def _partition_dataset(self, dataset):
        """Partition the dataset into conflicting and non-conflicting subsets."""
        conflicting = []
        non_conflicting = []
        
        for example in dataset:
            correctness_idx = example["correctness_chosen_id"]
            verbosity_idx = example["verbosity_chosen_id"]
            
            # Check for conflicting data
            if correctness_idx != verbosity_idx:
                conflicting.append(example)
            else:
                non_conflicting.append(example)

        print("total data:", len(dataset))
        print("conflicting fraction:", len(conflicting)/len(dataset))
        print("non_conflicting fraction", len(non_conflicting)/len(dataset))

        return conflicting, non_conflicting
    
    def _merge_with_fraction(self, conflicting, non_conflicting):
        """Merge conflicting and non-conflicting data with the specified fraction."""
        total_size = min(len(conflicting),len(non_conflicting))
        num_conflicting = int(total_size * self.conflicting_fraction)
        num_non_conflicting = total_size - num_conflicting

        # Select data from each set
        selected_conflicting = conflicting[:num_conflicting]
        selected_non_conflicting = non_conflicting[:num_non_conflicting]
        # selected_non_conflicting = non_conflicting[num_non_conflicting:2*num_non_conflicting]

        # Combine the two subsets and shuffle
        merged_data = selected_conflicting + selected_non_conflicting
        import random
        random.seed(42)
        random.shuffle(merged_data)
        return selected_conflicting, selected_non_conflicting, merged_data

    def get_preference_dataset(self, split):
        assert self.dimension in ["correctness","verbosity"], "only support conflicting dimnesions of correctness and verbosity"
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("filtering preference...")
        if split == "test":
            raise NotImplementedError
        dataset = dataset.filter(lambda x: x["correctness_chosen_id"] != -1)
        dataset = dataset.filter(lambda x: x["verbosity_chosen_id"] != -1)
        if split == "train":
            print_local_main("constructing and merging conflicting and non-conflicting data subsets...")
            conflicting, non_conflicting = self._partition_dataset(dataset)
            _, _, list_dataset = self._merge_with_fraction(conflicting, non_conflicting)
            dataset = Dataset.from_list(list_dataset)
        if split == "train_conflict":
            conflicting, non_conflicting = self._partition_dataset(dataset)
            list_dataset, _, _ = self._merge_with_fraction(conflicting, non_conflicting)
            dataset = Dataset.from_list(list_dataset)
        if split == "train_nonconflict":
            conflicting, non_conflicting = self._partition_dataset(dataset)
            _, list_dataset, _ = self._merge_with_fraction(conflicting, non_conflicting)
            dataset = Dataset.from_list(list_dataset)
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)


if __name__ == "__main__":
    num_proc = 4
    helpful_dataset = HelpSteerRDP(dimension="helpfulness", num_proc=num_proc).get_preference_dataset(split="train")
    overall_dataset = HelpSteerRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = HelpSteerRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
