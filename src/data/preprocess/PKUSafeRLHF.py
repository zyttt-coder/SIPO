import os
import argparse
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split PKUSafeRLHF")
    parser.add_argument("--out_dir", type=str, default="./output/downloaded_datasets/PKUSafeRLHF_preprocessed", help="output directory to cache the preprocessed dataset")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")

    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train").train_test_split(test_size=0.1, seed=0)["train"]
    val_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train").train_test_split(test_size=0.1, seed=0)["test"]
    # last 500 samples in the test split of PKU-SafeRLHF-30K
    test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="test").train_test_split(test_size=500, shuffle=False)["test"]

    train_dataset.to_json(train_path)
    val_dataset.to_json(val_path)
    test_dataset.to_json(test_path)

    print(f"Datasets saved to {args.out_dir}:")
    print(f" - Train: {train_path}")
    print(f" - Validation: {val_path}")
    print(f" - Test: {test_path}")