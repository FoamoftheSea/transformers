# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
""" Deformable DETR model configuration"""
from typing import Sequence

from ... import PvtV2Config
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

MULTIFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "FoamoftheSea/multiformer-b0": "https://huggingface.co/foamofthesea/multiformer/blob/main/config.json",
    # See all Deformable DETR models at https://huggingface.co/models?filter=deformable-detr
}


class MultiformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeformableDetrModel`]. It is used to instantiate
    a Deformable DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Deformable DETR
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        tasks (`List[str]`, *optional*, defaults to ["semseg", "depth", "det2d"]):
            List of tasks to make predictions for. Model will not predict or backprop loss for tasks not in list.
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        semantic_decoder_dim (`int`, *optional*, defaults to 256):
            The dimension of the all-MLP semantic segmentation decode head (from Segformer).
        semantic_classifier_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability before the classification head.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255:
            Class ID to ignore for semantic segmentation loss.
        depth_decoder_dim (`int`, *optional*, defaults to 64.):
            The hidden size of the GLPN depth decoder.
        silog_lambda (`float`, *optional*, defaults to 0.25):
            The lambda value for SiLog loss calculation. https://arxiv.org/abs/1406.2283
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DeformableDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of convolutional backbone to use in case `use_timm_backbone` = `True`. Supports any convolutional
            backbone from the timm package. For a list of all available models, see [this
            page](https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model).
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone. Only supported when `use_timm_backbone` = `True`.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
            `use_timm_backbone` = `True`.
        class_cost (`float`, *optional*, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        det2d_input_feature_levels (`List[int]`, *optional*, defaults to [-1]):
            The indices of backbone feature levels to use for deformable DETR.
        det2d_extra_feature_levels (`int`, *optional*, defaults to 1):
            The number of extra feature levels to create from the deepest backbone level used.
        det2d_use_pos_embed (`bool`, *optional*, defaults to True):
            Whether to generate and add positional embeddings into feature layers for deformable DETR.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        two_stage (`bool`, *optional*, defaults to `False`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Deformable DETR, which are further fed into the decoder for iterative bounding box refinement.
        two_stage_num_proposals (`int`, *optional*, defaults to 300):
            The number of region proposals to be generated, in case `two_stage` is set to `True`.
        with_box_refine (`bool`, *optional*, defaults to `False`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        max_depth (`int`, *optional*, defaults to 10):
            The maximum depth of the GLPN decoder.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the GLPN depth head.
        frozen_batch_norm (`bool`, *optional*, defaults to False):
            Whether to replace backbone batch norm layers with frozen ones (default in Deformable DETR).

    Examples:

    ```python
    >>> from transformers import MultiformerModel, MultiformerConfig

    >>> # Initializing a Deformable DETR SenseTime/deformable-detr style configuration
    >>> configuration = MultiformerConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = MultiformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "multiformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        tasks=["semseg", "depth", "det2d"],
        use_timm_backbone=False,
        backbone_config=PvtV2Config(
            mlp_ratios=[4, 4, 4, 4],
            out_indices=[0, 1, 2, 3],
        ),
        num_channels=3,
        semantic_decoder_dim=256,
        semantic_classifier_dropout=0.1,
        semantic_loss_ignore_index=255,
        depth_decoder_dim=64,
        silog_lambda=0.25,
        num_queries=300,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=1024,
        decoder_attention_heads=8,
        encoder_layerdrop=0.0,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        return_intermediate=True,
        auxiliary_loss=False,
        position_embedding_type="sine",
        backbone="resnet50",
        use_pretrained_backbone=True,
        dilation=False,
        det2d_input_feature_levels=None,
        det2d_extra_feature_levels=1,
        det2d_input_proj_kernels=None,
        det2d_input_proj_strides=None,
        det2d_input_proj_pads=None,
        det2d_input_proj_groups=32,
        det2d_use_pos_embed=True,
        det2d_box_keep_prob=0.5,
        encoder_n_points=4,
        decoder_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        with_box_refine=False,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        max_depth=10,
        head_in_index=-1,
        frozen_batch_norm=False,
        **kwargs,
    ):
        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        if not use_timm_backbone:
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

        if det2d_input_feature_levels is None:
            det2d_input_feature_levels = [i for i in range(len(backbone_config.out_indices))]

        if not all([idx < len(backbone_config.out_indices) for idx in det2d_input_feature_levels]):
            raise ValueError(
                "All det2d_input_feature_levels must correspond to backbone output layers, re-indexed at zero. \n"
                + "Example: if backbone_config.out_indices == [1, 3, 4], these become [0, 1, 2] as det2d input levels."
            )

        self.num_feature_levels = len(det2d_input_feature_levels) + det2d_extra_feature_levels

        n_layers = len(det2d_input_feature_levels)

        if det2d_input_proj_kernels is None:
            self.det2d_input_proj_kernels = [1 for _ in range(n_layers)]
        elif isinstance(det2d_input_proj_kernels, Sequence):
            if len(det2d_input_proj_kernels) != n_layers:
                raise ValueError(
                    "det2d_input_proj_kernels must same length as det2d_input_feature_levels: ({})".format(n_layers)
                )
            self.det2d_input_proj_kernels = list(det2d_input_proj_kernels)
        else:
            raise TypeError("det2d_input_proj_kernels must be Sequence, got {}".format(type(det2d_input_proj_kernels)))

        if det2d_input_proj_strides is None:
            self.det2d_input_proj_strides = [1 for _ in range(n_layers)]
        elif isinstance(det2d_input_proj_strides, Sequence):
            if len(det2d_input_proj_strides) != n_layers:
                raise ValueError(
                    "det2d_input_proj_strides must same length as det2d_input_feature_levels: ({})".format(n_layers)
                )
            self.det2d_input_proj_strides = list(det2d_input_proj_strides)
        else:
            raise TypeError("det2d_input_proj_strides must be Sequence, got {}".format(type(det2d_input_proj_strides)))

        if det2d_input_proj_pads is None:
            self.det2d_input_proj_pads = [0 for _ in range(n_layers)]
        elif isinstance(det2d_input_proj_pads, Sequence):
            if len(det2d_input_proj_pads) != n_layers:
                raise ValueError(
                    "det2d_input_proj_pads must same length as det2d_input_feature_levels: ({})".format(n_layers)
                )
            self.det2d_input_proj_pads = list(det2d_input_proj_pads)
        else:
            raise TypeError("det2d_input_proj_pads must be Sequence, got {}".format(type(det2d_input_proj_pads)))

        self.tasks = tasks
        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels
        self.semantic_decoder_dim = semantic_decoder_dim
        self.semantic_classifier_dropout = semantic_classifier_dropout
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.depth_decoder_dim = depth_decoder_dim
        self.silog_lambda = silog_lambda
        self.head_in_index = head_in_index
        self.max_depth = max_depth
        self.num_queries = num_queries
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.encoder_layerdrop = encoder_layerdrop
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.dilation = dilation
        # deformable attributes
        self.det2d_input_feature_levels = det2d_input_feature_levels
        self.det2d_extra_feature_levels = det2d_extra_feature_levels
        self.det2d_input_proj_groups = det2d_input_proj_groups
        self.det2d_use_pos_embed = det2d_use_pos_embed
        self.det2d_box_keep_prob = det2d_box_keep_prob
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.with_box_refine = with_box_refine
        if two_stage is True and with_box_refine is False:
            raise ValueError("If two_stage is True, with_box_refine must be True.")
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        self.frozen_batch_norm = frozen_batch_norm
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
