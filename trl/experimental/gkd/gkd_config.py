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

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from ...trainer.sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    """
    Configuration class for [`experimental.gkd.GKDTrainer`].

    This class includes only the parameters that are specific to GKD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_new_tokens (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str`, *optional*):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_init_kwargs (`dict[str, Any]]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
        use_windowed_loss (`bool`, *optional*, defaults to `False`):
            Whether to use windowed GKD loss that aligns joint probabilities over sliding windows of tokens instead
            of individual token-level alignment. When enabled, the loss is computed over sequences of consecutive
            tokens to capture local coherence and dependencies.
        window_size (`int`, *optional*, defaults to `5`):
            Number of consecutive tokens per window when `use_windowed_loss` is enabled. A window_size of 1 reduces
            to standard token-level distillation. Larger values capture longer-range dependencies but increase
            computational cost.
        window_stride (`int`, *optional*, defaults to `1`):
            Step size for the sliding window when `use_windowed_loss` is enabled. A stride of 1 means fully
            overlapping windows (each token appears in multiple windows). Larger strides reduce computational cost
            but provide less dense supervision.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy "
            "student-generated outputs)."
        },
    )
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence "
            "loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL "
            "Divergence."
        },
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model name or path of the teacher model. If `None`, the teacher model will be the same as the "
            "model being trained."
        },
    )
    teacher_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={
            "help": "Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised "
            "FT on teacher-generated output)."
        },
    )
    use_windowed_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use windowed GKD loss that aligns joint probabilities over sliding windows of tokens "
            "instead of individual token-level alignment."
        },
    )
    window_size: int = field(
        default=5,
        metadata={
            "help": "Number of consecutive tokens per window when `use_windowed_loss` is enabled. A window_size of 1 "
            "reduces to standard token-level distillation."
        },
    )
    window_stride: int = field(
        default=1,
        metadata={
            "help": "Step size for the sliding window when `use_windowed_loss` is enabled. A stride of 1 means fully "
            "overlapping windows."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")
        # check windowing parameters
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if self.window_stride < 1:
            raise ValueError("window_stride must be at least 1.")
        if self.use_windowed_loss and self.window_stride > self.window_size:
            raise ValueError("window_stride cannot be greater than window_size.")
