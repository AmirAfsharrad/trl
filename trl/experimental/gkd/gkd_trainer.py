# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import textwrap
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available

from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import disable_dropout_in_model
from ..utils import DataCollatorForChatML, empty_cache
from .gkd_config import GKDConfig


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


class GKDTrainer(SFTTrainer):
    """Trainer for Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`experimental.gkd.GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a
            [`experimental.utils.DataCollatorForChatML`] using the `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    """

    _tag_names = ["trl", "gkd"]
    _name = "GKD"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields (e.g., "messages") remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            if args.use_windowed_loss:
                raise ValueError(
                    "Liger kernel optimization is not yet supported with windowed GKD loss. "
                    "Please set either `use_liger_kernel=False` or `use_windowed_loss=False`."
                )
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True,
            "top_k": 0,
            "use_cache": False if args.gradient_checkpointing else True,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    @staticmethod
    def windowed_gkd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, window_size=5, window_stride=1, reduction="batchmean"
    ):
        """
        Compute the windowed generalized Jensen-Shannon Divergence loss for knowledge distillation.

        Instead of aligning token-by-token probability distributions, this method aligns joint probabilities
        over sliding windows of consecutive tokens. This captures local coherence and dependencies within
        token sequences, providing a smoother learning signal that allows the student model more flexibility
        in how it generates text while maintaining semantic consistency.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            window_size:
                Number of consecutive tokens per window (default: 5)
            window_stride:
                Step size for sliding window (default: 1)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the windowed generalized JSD loss

        Note:
            When window_size=1, this reduces to standard token-level generalized_jsd_loss.
            Each token may appear in multiple overlapping windows and thus receives gradients from multiple
            loss terms, which can provide stronger supervision for tokens in the middle of sequences.
        """
        batch_size, seq_len, _ = student_logits.shape

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)  # [B, L, V]
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # [B, L, V]

        # Create mask for valid tokens (not padding)
        if labels is not None:
            valid_mask = (labels != -100)  # [B, L]
        else:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=student_logits.device)

        # Gather log probabilities for the actual tokens from labels
        # We need to get the log prob of each token that was actually generated
        if labels is not None:
            # Clone labels and replace -100 with 0 for gathering (will be masked anyway)
            labels_for_gather = labels.clone()
            labels_for_gather[labels == -100] = 0

            # Gather token log probs: [B, L, V] -> [B, L]
            student_token_log_probs = torch.gather(
                student_log_probs, dim=-1, index=labels_for_gather.unsqueeze(-1)
            ).squeeze(-1)
            teacher_token_log_probs = torch.gather(
                teacher_log_probs, dim=-1, index=labels_for_gather.unsqueeze(-1)
            ).squeeze(-1)
        else:
            # If no labels, use argmax (though this is not typical for training)
            token_ids = student_logits.argmax(dim=-1)
            student_token_log_probs = torch.gather(
                student_log_probs, dim=-1, index=token_ids.unsqueeze(-1)
            ).squeeze(-1)
            teacher_token_log_probs = torch.gather(
                teacher_log_probs, dim=-1, index=token_ids.unsqueeze(-1)
            ).squeeze(-1)

        # Mask invalid tokens by setting their log probs to 0 (they won't contribute to window sums)
        student_token_log_probs = student_token_log_probs * valid_mask.float()
        teacher_token_log_probs = teacher_token_log_probs * valid_mask.float()

        # Create sliding windows and compute window-level joint log probabilities
        window_losses = []
        window_counts = []

        for start_idx in range(0, seq_len - window_size + 1, window_stride):
            end_idx = start_idx + window_size

            # Extract windows
            window_student_log_probs = student_token_log_probs[:, start_idx:end_idx]  # [B, W]
            window_teacher_log_probs = teacher_token_log_probs[:, start_idx:end_idx]  # [B, W]
            window_mask = valid_mask[:, start_idx:end_idx]  # [B, W]

            # Count valid tokens in each window
            valid_tokens_per_window = window_mask.sum(dim=1)  # [B]

            # Skip windows with no valid tokens
            has_valid_tokens = valid_tokens_per_window > 0
            if not has_valid_tokens.any():
                continue

            # Compute joint log probabilities by summing over the window (log of product = sum of logs)
            # This gives us the joint probability of the token sequence in the window
            student_window_joint_log_prob = window_student_log_probs.sum(dim=1)  # [B]
            teacher_window_joint_log_prob = window_teacher_log_probs.sum(dim=1)  # [B]

            # Now compute generalized JSD on these joint probabilities
            # These are already log probabilities (sums of log probs), so we treat them as log-space values
            if beta == 0:
                # Forward KL: KL(student || teacher)
                window_jsd = student_window_joint_log_prob - teacher_window_joint_log_prob
            elif beta == 1:
                # Reverse KL: KL(teacher || student)
                window_jsd = teacher_window_joint_log_prob - student_window_joint_log_prob
            else:
                # Generalized JSD
                # Convert back to probability space for mixture
                student_window_prob = torch.exp(student_window_joint_log_prob)
                teacher_window_prob = torch.exp(teacher_window_joint_log_prob)

                # Mixture distribution
                beta_tensor = torch.tensor(beta, dtype=student_window_prob.dtype, device=student_window_prob.device)
                mixture_prob = (1 - beta_tensor) * student_window_prob + beta_tensor * teacher_window_prob
                mixture_log_prob = torch.log(mixture_prob + 1e-10)  # Add small epsilon for numerical stability

                # Compute KL divergences
                kl_teacher = teacher_window_joint_log_prob - mixture_log_prob
                kl_student = student_window_joint_log_prob - mixture_log_prob

                # Generalized JSD
                window_jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

            # Mask out windows with no valid tokens
            window_jsd = window_jsd * has_valid_tokens.float()

            window_losses.append(window_jsd)
            window_counts.append(has_valid_tokens.float())

        # If no valid windows were found, return zero loss
        if len(window_losses) == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # Stack all window losses
        all_window_losses = torch.stack(window_losses)  # [num_windows, B]
        all_window_counts = torch.stack(window_counts)  # [num_windows, B]

        # Apply reduction
        if reduction == "batchmean":
            total_loss = all_window_losses.sum()
            total_count = all_window_counts.sum()
            return total_loss / (total_count + 1e-10)
        elif reduction == "sum":
            return all_window_losses.sum()
        elif reduction == "mean":
            return all_window_losses.mean()
        else:
            return all_window_losses

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_gkd_loss:
            # Forward only through the base models (avoid lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                base_student = unwrapped_student.get_decoder()
            else:
                base_student = getattr(
                    unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                )

            student_outputs = base_student(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
                base_teacher = unwrapped_teacher.get_decoder()
            else:
                base_teacher = getattr(
                    unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
                )
            with torch.no_grad():
                teacher_outputs = base_teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

            # hidden states (shifted)
            student_hidden = student_outputs.last_hidden_state[:, :-1]
            teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

            # Release full outputs to free memory
            del student_outputs, teacher_outputs

            # labels mask and labels (shifted)
            labels_mask = inputs["labels"] != -100
            masked_input_ids = torch.where(
                labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
            )
            true_labels = masked_input_ids[:, 1:].contiguous()

            # Release intermediate tensors
            del labels_mask, masked_input_ids

            # heads
            student_head = unwrapped_student.get_output_embeddings()
            teacher_head = unwrapped_teacher.get_output_embeddings()

            # liger fused jsd loss
            loss = self.liger_jsd_loss(
                student_input=student_hidden,
                student_weight=student_head.weight,
                teacher_input=teacher_hidden,
                teacher_weight=teacher_head.weight,
                true_labels=true_labels,
                student_bias=getattr(student_head, "bias", None),
                teacher_bias=getattr(teacher_head, "bias", None),
            )

            # Release hidden states after loss computation
            del student_hidden, teacher_hidden, true_labels
        else:
            # compute student output
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # compute teacher output in eval mode
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            # slice the logits for the generated tokens using the inputs["prompts"] lengths
            prompt_lengths = inputs["prompts"].shape[1]
            shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
            shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
            shifted_labels = inputs["labels"][:, prompt_lengths:]

            # compute loss - use windowed loss if enabled, otherwise use standard token-level loss
            if self.args.use_windowed_loss:
                loss = self.windowed_gkd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,
                    window_size=self.args.window_size,
                    window_stride=self.args.window_stride,
                )
            else:
                loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        if self.seq_kd:
            with (
                unwrap_model_for_generation(
                    self.teacher_model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with (
                unwrap_model_for_generation(
                    model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
