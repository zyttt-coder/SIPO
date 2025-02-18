# SIPO: Self-Improvement Towards Pareto Optimality

Official code for paper "Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment"

<p align="center">
  <img src="assets/framework.jpg" width="100%">
</p>

We provide instructions for running SIPO with MOD sampling on the **HelpSteer** dataset. For the **BeaverTails-10K** subset, please refer to [scripts/examples/beavertails.sh](scripts/examples/beavertails.sh).

## Environment Set Up
```
conda create -n SIPO python=3.10.15
pip install -r requirements.txt
```
## Dataset Preprocess
```
python ./src/data/preprocess/HS_preprocess.py --tokenizer_path PKU-Alignment/alpaca-7b-reproduced
```

## First Time alignment
First, perform first time alignment on the sft-model using DPO on two conflicted objectives (*correctness* and *verbosity* for **HelpSteer**). 
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/baselines/dpo.py     --sft_model_name {path_to_the_sft_model}     --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"     --dataset_name "nvidia/HelpSteer-pairwise-{correctness|verbosity}-0.6"     --max_length 512     --training_args.output_dir "./output/nvidia/HelpSteer/SIPO/{0.0|1.0}correctness"     --training_args.run_name "nvidia/HelpSteer/SIPO/{0.0|1.0}correctness"     --training_args.per_device_train_batch_size 1     --training_args.per_device_eval_batch_size 6     --training_args.gradient_accumulation_steps 2     --training_args.learning_rate 5e-4     --peft_config.r 64     --peft_config.target_modules q_proj k_proj v_proj o_proj     --peft_config.lora_alpha 1     --peft_config.lora_dropout 0
```

## MOD sampling
Second, use MOD sampling to sample responses.
```
accelerate launch scripts/baselines/mod.py     --soup_weights 0.2 0.4 0.6 0.8   --sft_model_name {path_to_the_sft_model}    --dpo_model_1_name "./output/nvidia/HelpSteer/SIPO/0.0correctness/best_checkpoint"    --dpo_model_2_name "./output/nvidia/HelpSteer/SIPO/1.0correctness/best_checkpoint"     --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"     --dataset_name "nvidia/HelpSteer-pairwise-correctness-0.6"     --output_dir "./output/nvidia/HelpSteer/SIPO/gen_sample"     --max_length 512  --eval_size -1  --split "train_conflict"
```

## Review Generation
Thrid, generate reviews for the MOD-sampled responses.
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ./scripts/SIPO/review.py --sft_model_name {path_to_the_sft_model} --adapter_model_name "./output/nvidia/HelpSteer/SIPO/1.0correctness/best_checkpoint" --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" --input_dir "./output/nvidia/HelpSteer/SIPO/gen_sample" --output_dir "./output/nvidia/HelpSteer/SIPO/review" --max_length 1200 --replication 4
```

## Rewrite
Forth, rewrite MOD_sampled response based on the reviews.
```
PYTHONPATH=. accelerate launch ./scripts/SIPO/rewrite.py --sft_model_name {path_to_the_sft_model}     --dpo_model_1_name "./output/nvidia/HelpSteer/SIPO/0.0correctness/best_checkpoint"   --dpo_model_2_name "./output/nvidia/HelpSteer/SIPO/1.0correctness/best_checkpoint"    --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"    --input_dir "./output/nvidia/HelpSteer/SIPO/review"   --output_dir "./output/nvidia/HelpSteer/SIPO/rewrite"    --max_length 1600
```

## Filter


