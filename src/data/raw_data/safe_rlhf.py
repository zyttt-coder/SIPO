import os
from dataclasses import dataclass
from abc import ABC
from typing import Dict, Literal, Optional

from datasets import Dataset, load_dataset

from .utils import RawDatasetPreprocessor

@dataclass
class PKUSafeRlhfRDPBase(RawDatasetPreprocessor, ABC):
    dimension: Literal["safer", "better"] = "better"

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_idx = example[f"{self.dimension}_response_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_idx}"],
            "rejected": example[f"response_{1-chosen_idx}"],
            "contradict": example["safer_response_id"] != example["better_response_id"],
        }

@dataclass
class PKUSafeRlhf10KRDP(PKUSafeRlhfRDPBase):
    path: Optional[str] = "./output/downloaded_datasets/PKUSafeRLHF_preprocessed"

    def _get_raw_dataset(self, split):
        dataset = load_dataset("json", data_files={
            "train": os.path.join(self.path, "train.jsonl"),
            "validation": os.path.join(self.path, "val.jsonl"),
            "test": os.path.join(self.path, "test.jsonl")
        })
        if split == "train":
            return dataset["train"]
        elif split == "validation":
            return dataset["validation"]
        elif split == "test":
            return dataset["test"]
        elif split == "train_conflict":
            # conflicting examples in the training dataset
            raw_dataset = dataset["train"]
            conflicting = []
            for example in raw_dataset:
                safer_idx = example["safer_response_id"]
                better_idx = example["better_response_id"]
                if safer_idx != better_idx:
                    conflicting.append(example)
            return Dataset.from_list(conflicting)
        else:
            raise NotImplementedError


@dataclass
class PKUSafeRlhfRDPBaseWithConflict(PKUSafeRlhfRDPBase):
    conflicting_fraction: float = 0.0  # Default fraction of conflicting data

    def _partition_dataset(self, dataset):
        """Partition the dataset into conflicting and non-conflicting subsets."""
        conflicting = []
        non_conflicting = []
        
        for example in dataset:
            safer_idx = example["safer_response_id"]
            better_idx = example["better_response_id"]
            
            # Check for conflicting data
            if safer_idx != better_idx:
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
        # selected_conflicting = conflicting[:num_conflicting]
        selected_conflicting = conflicting[-num_conflicting:]
        # selected_non_conflicting = non_conflicting[:num_non_conflicting]
        selected_non_conflicting = non_conflicting[-num_non_conflicting:]

        # Combine the two subsets and shuffle
        merged_data = selected_conflicting + selected_non_conflicting
        import random
        random.seed(42)
        random.shuffle(merged_data)
        return merged_data

    
@dataclass
class PKUSafeRlhf10KRDPWithConflict(PKUSafeRlhfRDPBaseWithConflict):
    path: Optional[str] = "./output/downloaded_datasets/PKUSafeRLHF_preprocessed"

    def _get_raw_dataset(self, split):
        dataset = load_dataset("json", data_files={
            "train": os.path.join(self.path, "train.jsonl"),
            "validation": os.path.join(self.path, "val.jsonl"),
        })
        if split == "train":
            raw_dataset = dataset["train"]
        elif split == "validation":
            raw_dataset = dataset["validation"]
        else:
            raise NotImplementedError

        # Partition into conflicting and non-conflicting subsets
        conflicting, non_conflicting = self._partition_dataset(raw_dataset)

        # Merge the datasets based on the conflicting fraction
        return Dataset.from_list(self._merge_with_fraction(conflicting, non_conflicting))

if __name__ == '__main__':
    safer10k_train_dataset = PKUSafeRlhf10KRDP(dimension="safer").get_preference_dataset(split="train")
    better10k_train_dataset = PKUSafeRlhf10KRDP(dimension="better").get_preference_dataset(split="train")
    breakpoint()
