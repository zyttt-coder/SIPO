import os
import argparse
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def save_dataset_splits(train_dataset, val_dataset, test_dataset, output_dir):
    """Save datasets to the local directory in `jsonl` format."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    train_dataset.to_json(train_path)
    val_dataset.to_json(val_path)
    test_dataset.to_json(test_path)

    print(f"Datasets saved to {output_dir}:")
    print(f" - Train: {train_path}")
    print(f" - Validation: {val_path}")
    print(f" - Test: {test_path}")

def prefilter_dataset(dataset, tokenizer, max_length, num_proc):
    """Prefilter the dataset based on tokenized lengths."""
    def filter_by_max_length(sample):
        # Tokenize and compute token lengths
        prompt_length = len(tokenizer(sample["prompt"], truncation=False)["input_ids"])
        response_0_length = len(tokenizer(sample["response_0"], truncation=False)["input_ids"])
        response_1_length = len(tokenizer(sample["response_1"], truncation=False)["input_ids"])
        # Ensure all parts fit within max_length
        return (
            prompt_length + response_0_length <= max_length
            and prompt_length + response_1_length <= max_length
        )
    
    # Filter dataset
    filtered_dataset = dataset.filter(filter_by_max_length, num_proc=num_proc)
    return filtered_dataset

def helpsteer_transform_to_preference(batched_sample):
    """Transform HelpSteer data to preference format."""
    def chosen_id(score_0, score_1):
        if score_0 < score_1:
            return 1
        elif score_0 > score_1:
            return 0
        else:
            return -1

    finegrained_dimensions = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    dimensions = finegrained_dimensions + ("overall",)

    debatched_sample = [{k: batched_sample[k][i] for k in batched_sample.keys()} for i in range(len(batched_sample["prompt"]))]

    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in dimensions}
    }
    mini_debatch = []
    for i, sample in enumerate(debatched_sample):
        mini_debatch.append(sample)
        if i != len(debatched_sample) - 1 and sample["prompt"] == debatched_sample[i + 1]["prompt"]:
            continue

        for j in range(len(mini_debatch)):
            for k in range(j + 1, len(mini_debatch)):
                new_batched_sample["prompt"].append(mini_debatch[j]["prompt"])
                new_batched_sample["response_0"].append(mini_debatch[j]["response"])
                new_batched_sample["response_1"].append(mini_debatch[k]["response"])
                new_batched_sample["overall_chosen_id"].append(
                    chosen_id(
                        sum(mini_debatch[j][dimension] for dimension in finegrained_dimensions),
                        sum(mini_debatch[k][dimension] for dimension in finegrained_dimensions),
                    )
                )
                for dimension in finegrained_dimensions:
                    new_batched_sample[f"{dimension}_chosen_id"].append(
                        chosen_id(
                            mini_debatch[j][dimension], 
                            mini_debatch[k][dimension],
                        )
                    )

        mini_debatch = []

    return new_batched_sample

if __name__ == "__main__":
    # Define parameters
    parser = argparse.ArgumentParser(description="Preprocess HelpSteer dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--out_dir", type=str, default="./output/downloaded_datasets/HelpSteer_preprocessed", help="output directory to cache the preprocessed dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum tokenized length for filtering")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for parallel processing")

    args = parser.parse_args()

    # Extract arguments
    tokenizer_path = args.tokenizer_path
    max_length = args.max_length
    num_proc = args.num_proc

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load datasets
    train_dataset = load_dataset("nvidia/HelpSteer", split="train")
    val_dataset = load_dataset("nvidia/HelpSteer", split="validation")

    print(f"Original Train Samples: {len(train_dataset)}")
    print(f"Original Validation Samples: {len(val_dataset)}")
    print(f"Original Train Prompts: {len(set(train_dataset['prompt']))}")
    print(f"Original Validation Prompts: {len(set(val_dataset['prompt']))}")

    # Transform datasets to preference format
    print("Mapping raw dataset to preference...")
    train_dataset = train_dataset.map(
        helpsteer_transform_to_preference,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        helpsteer_transform_to_preference,
        batched=True,
        num_proc=num_proc,
        remove_columns=val_dataset.column_names,
    )

    print(f"Transformed Train Samples: {len(train_dataset)}")
    print(f"Transformed Validation Samples: {len(val_dataset)}")
    print(f"Transformed Train Prompts: {len(set(train_dataset['prompt']))}")
    print(f"Transformed Validation Prompts: {len(set(val_dataset['prompt']))}")

    # Prefilter datasets
    print("Filtering samples that are too long...")
    prefilter_func = partial(prefilter_dataset, tokenizer=tokenizer, max_length=max_length, num_proc=num_proc)
    train_dataset = prefilter_func(train_dataset)
    val_dataset = prefilter_func(val_dataset)

    print(f"Filtered Train Samples: {len(train_dataset)}")
    print(f"Filtered Validation Samples: {len(val_dataset)}")
    print(f"Filtered Train Prompts: {len(set(train_dataset['prompt']))}")
    print(f"Filtered Validation Prompts: {len(set(val_dataset['prompt']))}")

    # Partition the test set from the train split
    print("Generating test split...")
    val_unique_prompts = set(val_dataset["prompt"])
    n = len(val_unique_prompts)
    train_prompts = list(set(train_dataset["prompt"]))
    test_prompts = train_prompts[:n]
    
    test_dataset = Dataset.from_list(
        [{"prompt": prompt} for prompt in test_prompts]
    )
    train_dataset = train_dataset.filter(lambda x: x["prompt"] not in test_prompts)
    
    print(f"Final Train Samples: {len(train_dataset)}")
    print(f"Final Validation Samples: {len(val_dataset)}")
    print(f"Final Test Samples: {len(test_dataset)}")
    print(f"Final Train Prompts: {len(set(train_dataset['prompt']))}")
    print(f"Final Validation Prompts: {n}")
    print(f"Final Test Prompts: {len(set(test_dataset['prompt']))}")

    # Save datasets
    save_dataset_splits(train_dataset, val_dataset, test_dataset, args.out_dir)

    
