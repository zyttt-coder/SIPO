from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from src.trainer.dpo_trainer import DPOTrainer

class IterativeDPOTrainer(DPOTrainer):
    """
    Self-Improving DPO trainer
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
        filter_too_long: Optional[bool] = True,
        num_proc: Optional[int] = 4,
        generate_during_eval: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        #add
        alpha: float = 1.0,
    ):
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
            filter_too_long=filter_too_long,
            num_proc=num_proc, 
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
        )

        self.alpha = alpha

    def dpo_loss(
        self,
        chosen_seq_lengths: torch.FloatTensor,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_seq_lengths: sequence length of the chosen response. Shape: (batch_size,)
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        logits = chosen_rewards - rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")
        
        losses = losses - self.alpha*(policy_chosen_logps/chosen_seq_lengths)

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
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

        chosen_input_ids, _ = batch["input_ids"].chunk(2)
        chosen_seq_lengths = (chosen_input_ids != self.label_pad_token_id).sum(dim=1).to(torch.float32)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_seq_lengths,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        accuracies = (chosen_rewards > rejected_rewards).float()

        # change
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