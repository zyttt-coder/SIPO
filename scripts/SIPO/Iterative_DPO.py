import os
from dataclasses import dataclass, field
from typing import Optional

import tqdm
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.iterativedpo_trainer import IterativeDPOTrainer
from src.data.configs import DEFAULT_PROMPT_TEMPLATE, DATASET_CONFIGS
from src.utils.utils import print_local_main, disable_progress_bar_non_local_main, param_sharding_enabled, set_seeds
from scripts.utils.utils import extract_prompt_content

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:

    sft_model_name: str = field(metadata={"help": "the sft model name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    alpha: Optional[float] = field(default=1, metadata={"help": "alpha to control the NLL loss"})
    cached_data_dir: Optional[str] = field(default=None, metadata={"help": "path for cached datasets"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the added dataset name"})
    original_dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the original dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})
    generate_during_eval: Optional[bool] = field(default=True, metadata={"help": "whether to generate during evaluation"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/dpo",
            overwrite_output_dir=True,
            seed=42,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_dpo",
            report_to="wandb",

            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

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


# Construct datasets
def construct_datasets(cached_data_dir, check_sample_path, prompt_template):
    positive_path = os.path.join(cached_data_dir, "positive")
    negative_path = os.path.join(cached_data_dir, "negative")

    # Load datasets
    positive_ds = load_dataset(positive_path, split="train")
    negative_ds = load_dataset(negative_path, split="train")
    check_sample_ds = load_dataset(check_sample_path, split="train")
    preference_data = []

    # Create preference pairs from check_sample_ds
    for example in tqdm.tqdm(check_sample_ds, total=len(check_sample_ds), desc="creating preference pairs"):
        map_index = example["map_index"]
        prompt_response = example["prompt_response"]
        positive_example = positive_ds[map_index]["prompt_response"]
        negative_example = negative_ds[map_index]["prompt_response"]

        positive_question = extract_prompt_content(prompt_template, positive_example)
        positive_response = extract_prompt_content(prompt_template, positive_example, extract_after=True)
        negative_question = extract_prompt_content(prompt_template, negative_example)
        negative_response = extract_prompt_content(prompt_template, negative_example, extract_after=True)
        # prompt = extract_prompt_content(prompt_template, prompt_response)
        response = extract_prompt_content(prompt_template, prompt_response, extract_after=True)
        # assert prompt == positive_question and prompt == negative_question, "error occurs during question map"
        assert positive_question == negative_question
        prompt = positive_question

        preference_data.append({
            "raw_prompt": prompt,
            "prompt": prompt_template.format(raw_prompt=prompt),
            "chosen": response,
            "rejected": positive_response,
        })

        preference_data.append({
            "raw_prompt": prompt,
            "prompt": prompt_template.format(raw_prompt=prompt),
            "chosen": response,
            "rejected": negative_response,
        })

    dataset = Dataset.from_list(preference_data)

    return dataset

script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.training_args.seed)
if not script_args.peft:
    script_args.peft_config = None

# base model
print_local_main("loading model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
    torch_dtype=torch.bfloat16, # necessary for llama2, otherwise will be cast to float32
    cache_dir=".cache",
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
sft_model.config.update({
    "use_cache": False,
    "pad_token_id": sft_model.config.eos_token_id 
})
print_local_main(sft_model)
print_local_main(script_args.peft_config)
    
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#dataset
train_dataset = construct_datasets(script_args.cached_data_dir, script_args.dataset_name, script_args.prompt_template)
if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
rdp = DATASET_CONFIGS[script_args.original_dataset_name](
    prompt_template=script_args.prompt_template,
    sanity_check=script_args.sanity_check,
)
eval_dataset = rdp.get_preference_dataset(split="validation")

# get ready for training
print_local_main("start training...")
trainer = IterativeDPOTrainer(
    model=sft_model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=script_args.peft_config,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=script_args.generate_during_eval,
    #add on
    alpha=script_args.alpha,
)
if Accelerator().is_local_main_process and script_args.peft_config:
    trainer.model.print_trainable_parameters()
trainer.train()

best_checkpoint = trainer.state.best_model_checkpoint
print(f"Best model checkpoint: {best_checkpoint}")

save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
