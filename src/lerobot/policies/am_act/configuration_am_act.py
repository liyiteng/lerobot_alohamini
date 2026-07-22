#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("am_act")
@dataclass
class AMACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0
    # Action dimensions that are constant in the task. They are forced to zero in
    # normalized space during training and inference, then restored by the action
    # unnormalizer at inference time.
    fixed_action_dims: list[int] = field(default_factory=list)
    # Optional named groups for independently normalized reconstruction losses.
    action_loss_groups: dict[str, list[int]] = field(default_factory=dict)
    action_loss_weights: dict[str, float] = field(default_factory=dict)
    # Optional hybrid action head. Listed dimensions are trained with
    # classification instead of L1 regression, then decoded back into normalized
    # action values so the existing action unnormalizer and robot API remain
    # unchanged. `discrete_action_values` are physical dataset values;
    # `discrete_action_normalized_values` is populated from dataset statistics at
    # policy construction time and saved with the checkpoint.
    discrete_action_dims: list[int] = field(default_factory=list)
    discrete_action_values: list[list[float]] = field(default_factory=list)
    discrete_action_normalized_values: list[list[float]] = field(default_factory=list)
    discrete_action_class_weights: list[list[float]] = field(default_factory=list)
    discrete_action_loss_weight: float = 1.0
    # Allow a structurally modified ACT policy to reuse only checkpoint tensors
    # whose names and shapes still match.
    allow_partial_pretrained_load: bool = False
    # When warm-starting from a checkpoint trained with different observations,
    # rebuild input_features from the current dataset metadata.
    use_dataset_input_features: bool = False
    # Optional subset of observation.state consumed by ACT. The processor may
    # still normalize the robot's complete state vector before this selection.
    observation_state_dims: list[int] = field(default_factory=list)

    # Physical action dimensions to scale after unnormalization at inference.
    # This is intended for velocity commands, not absolute joint positions.
    inference_action_scale_dims: list[int] = field(default_factory=list)
    inference_action_scale: float = 1.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if any(dim < 0 for dim in self.fixed_action_dims):
            raise ValueError("`fixed_action_dims` must contain non-negative indices.")
        if len(set(self.fixed_action_dims)) != len(self.fixed_action_dims):
            raise ValueError("`fixed_action_dims` must not contain duplicate indices.")
        if self.inference_action_scale < 0:
            raise ValueError("`inference_action_scale` must be non-negative.")
        if any(weight < 0 for weight in self.action_loss_weights.values()):
            raise ValueError("`action_loss_weights` must be non-negative.")
        if len(set(self.discrete_action_dims)) != len(self.discrete_action_dims):
            raise ValueError("`discrete_action_dims` must not contain duplicate indices.")
        if any(dim < 0 for dim in self.discrete_action_dims):
            raise ValueError("`discrete_action_dims` must contain non-negative indices.")
        if self.discrete_action_dims and len(self.discrete_action_dims) != len(self.discrete_action_values):
            raise ValueError("Each discrete action dimension needs a list of physical values.")
        if self.discrete_action_normalized_values and len(self.discrete_action_dims) != len(
            self.discrete_action_normalized_values
        ):
            raise ValueError("Normalized discrete values must match discrete_action_dims.")
        if self.discrete_action_class_weights and len(self.discrete_action_dims) != len(
            self.discrete_action_class_weights
        ):
            raise ValueError("Discrete class weights must match discrete_action_dims.")
        for index, values in enumerate(self.discrete_action_values):
            if len(values) < 2 or len(set(values)) != len(values):
                raise ValueError(f"Discrete action values at index {index} must be unique.")
        for index, weights in enumerate(self.discrete_action_class_weights):
            if len(weights) != len(self.discrete_action_values[index]) or any(
                weight <= 0 for weight in weights
            ):
                raise ValueError(f"Invalid discrete class weights at index {index}.")
        if self.discrete_action_loss_weight <= 0:
            raise ValueError("`discrete_action_loss_weight` must be positive.")
        if len(set(self.observation_state_dims)) != len(self.observation_state_dims):
            raise ValueError("`observation_state_dims` must not contain duplicate indices.")
        if any(dim < 0 for dim in self.observation_state_dims):
            raise ValueError("`observation_state_dims` must contain non-negative indices.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
