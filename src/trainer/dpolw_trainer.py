from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from src.trainer.dpo_trainer import DPOTrainer, DPODataMapFunc, DPODataCollatorWithPadding


@dataclass
class DPOLWDataMapFunc(DPODataMapFunc):
    def __call__(self, examples):
        """
        Additionally keep the contradict label for each sample to represent the preference difference between two objectives
        """
        new_examples = super().__call__(examples)
        new_examples['contradict'] = examples['contradict']
        return new_examples
    
@dataclass
class DPOLWDataCollatorWithPadding(DPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]], generate: Optional[bool] = False) -> Dict[str, Any]:
        batch = super().__call__(features, generate)
        if not generate:
            batch["contradict"] = [feature["contradict"] for feature in features]
        return batch
    
class DPOLWTrainer(DPOTrainer):
    """
    DPO loss weighting (DPO LW) is an intuitive method for multi-objective DPO which mixes D1
    and D2 and trains on both datasets simultaneously, weighting the loss by w
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        tokenize_map_func: Optional[Callable] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        disable_dropout: bool = True,
        max_length: Optional[int] = 1024,
        num_proc: Optional[int] = 4,
        generate_during_eval: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        #add
        w: List[float] = [0.5,0.5],
    ):
        if tokenize_map_func is None:
            tokenize_map_func = DPOLWDataMapFunc(tokenizer)

        if data_collator is None:
            data_collator = DPOLWDataCollatorWithPadding(tokenizer)

        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            loss_type=loss_type,
            args=args,
            tokenize_map_func=tokenize_map_func,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            disable_dropout=disable_dropout,
            max_length=max_length,
            num_proc=num_proc, 
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
        )

        self.w = torch.tensor(w).to(self.accelerator.device)

    def dpolw_loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        contradict: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        logits = chosen_rewards - rejected_rewards
        #compute DPO reward with respect to another objective
        obj2_policy_chosen_logps = torch.where(contradict, policy_rejected_logps, policy_chosen_logps)
        obj2_policy_rejected_logps = torch.where(contradict, policy_chosen_logps, policy_rejected_logps)
        obj2_reference_chosen_logps = torch.where(contradict, reference_rejected_logps, reference_chosen_logps)
        obj2_reference_rejected_logps = torch.where(contradict, reference_chosen_logps, reference_rejected_logps)

        obj2_chosen_rewards = self.beta * (obj2_policy_chosen_logps   - obj2_reference_chosen_logps)
        obj2_rejected_rewards = self.beta * (obj2_policy_rejected_logps - obj2_reference_rejected_logps)

        obj2_logits = obj2_chosen_rewards - obj2_rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits) * self.w[0] + -F.logsigmoid(obj2_logits) * self.w[1]
        else:
            raise NotImplementedError

        return losses, chosen_rewards.detach()*self.w[0] + obj2_chosen_rewards.detach()*self.w[1], rejected_rewards.detach()*self.w[0] + obj2_rejected_rewards.detach()*self.w[1]
        
    def dpo_loss(self, *args, **kwargs):
        """Disable the `dpo_loss` inherited from the DPOTrainer"""
        raise NotImplementedError
        
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.forward(model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpolw_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            torch.tensor(batch["contradict"], dtype=torch.bool).to(self.accelerator.device),
        )

        accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu()
        metrics[f"{prefix}logps/margins"] = (policy_chosen_logps - policy_rejected_logps).detach().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu()
        if train_eval == "train":
            metrics[f"{prefix}accuracy"] = accuracies.detach().cpu()

        return losses.mean(), metrics