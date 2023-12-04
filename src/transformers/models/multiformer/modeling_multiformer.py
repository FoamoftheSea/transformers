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
""" PyTorch Deformable DETR model."""


import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.models.deformable_detr.load_custom import load_cuda_kernels

from ... import SegformerDecodeHead
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_torch_cuda_available,
    is_vision_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_ninja_available, logging
from ..auto import AutoBackbone
from ..glpn.modeling_glpn import GLPNDecoder, GLPNDecoderStage, GLPNDepthEstimationHead
from ..segformer.modeling_segformer import SegformerMLP
from .configuration_multiformer import MultiformerConfig


# from .load_custom import load_cuda_kernels


logger = logging.get_logger(__name__)

# Move this to not compile only when importing, this needs to happen later, like in __init__.
if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")
    try:
        MultiScaleDeformableAttention = load_cuda_kernels()
    except Exception as e:
        logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")
        MultiScaleDeformableAttention = None
else:
    MultiScaleDeformableAttention = None

if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


class MultiformerTask:
    DET_2D = "det_2d"
    DET_3D = "det_3d"
    SEMSEG = "semseg"
    DEPTH = "depth"


class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_timm_available():
    from timm import create_model

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeformableDetrConfig"
_CHECKPOINT_FOR_DOC = "sensetime/deformable-detr"

MULTIFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FoamoftheSea/multiformer-b0",
    # See all Deformable DETR models at https://huggingface.co/models?filter=deformable-detr
]


@dataclass
class DeformableDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DeformableDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MultiformerModelOutput(ModelOutput):
    """
    Base class for outputs of the Deformable DETR encoder-decoder model.

    Args:
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
        backbone_features ('tuple(torch.FloatTensor)', *optional*:
            output feature maps from backbone model.
    """

    init_reference_points: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    backbone_features: Tuple[torch.FloatTensor] = None
    logits_semantic: Optional[torch.FloatTensor] = None
    predicted_depth: Optional[torch.FloatTensor] = None


@dataclass
class MultiformerOutput(ModelOutput):
    """
    Output type of [`DeformableDetrForObjectDetection`].

    Args:
        loss (`Dict[str, torch.FloatTensor]` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DeformableDetrProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        pred_boxes_3d (`torch.FloatTensor` of shape `(batch_size, num_queries, 7):
            Each box has 7 elements (dx, dy, dz, h, w, l, phi), which are defined as:
                dx: normalized offset on x axis from 2D box center on image plane.
                dy: normalized offset on y axis from 2D box center on image plane.
                dz: offset of log loss to box center from pixel depth (when box center is occluded by object).
                phi: heading angle btw [-pi, pi] relative to ego.
                l: cuboid length
                w: cuboid width
                h: cuboid height
        logits_semantic (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        pred_depth (`torch.FloatTensor` of shape `(batch_size, image_height, image_width)`):
            Depth estimation for each pixel.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4,
            4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    loss: Optional[Dict[str, torch.FloatTensor]] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_boxes_3d: torch.FloatTensor = None
    logits_semantic: torch.FloatTensor = None
    pred_depth: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    enc_outputs_class: Optional = None
    enc_outputs_coord_logits: Optional = None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->DeformableDetr
class DeformableDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->DeformableDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DeformableDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DeformableDetrFrozenBatchNorm2d(module.num_features)

            new_module.weight.data.copy_(module.weight)
            new_module.bias.data.copy_(module.bias)
            new_module.running_mean.data.copy_(module.running_mean)
            new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DeformableDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DeformableDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(2, 3, 4) if config.num_feature_levels > 1 else (4,),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)

        # replace batch norm by frozen batch norm
        if config.frozen_batch_norm:
            with torch.no_grad():
                replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->DeformableDetr
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        outputs = self.model(pixel_values)
        if self.config.use_timm_backbone:
            features = outputs
        elif hasattr(outputs, "hidden_states"):
            features = outputs.hidden_states
        elif hasattr(outputs, "feature_maps"):
            features = outputs.feature_maps
        else:
            raise AttributeError("Backbone model output must contain one of ('hidden_states', 'feature_maps').")

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class DeformableDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding
class DeformableDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->DeformableDetr
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = DeformableDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = DeformableDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class DeformableDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: MultiformerConfig, num_heads: int, n_points: int):
        super().__init__()
        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if self.disable_custom_kernels:
            # PyTorch implementation
            output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        else:
            try:
                # custom kernel
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


class DeformableDetrMultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Deformable DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, target_len, embed_dim = hidden_states.size()
        hidden_states_original = hidden_states
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # get queries, keys and values
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class DeformableDetrEncoderLayer(nn.Module):
    def __init__(self, config: MultiformerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DeformableDetrDecoderLayer(nn.Module):
    def __init__(self, config: MultiformerConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = DeformableDetrMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # cross-attention
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead
class DeformableDetrClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class DeformableDetrPreTrainedModel(PreTrainedModel):
    config_class = MultiformerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, DeformableDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, DeformableDetrMultiscaleDeformableAttention):
            module._reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, DeformableDetrDecoder):
    #         module.gradient_checkpointing = value


DEFORMABLE_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeformableDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEFORMABLE_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DeformableDetrImageProcessor.__call__`]
            for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class DeformableDetrEncoder(DeformableDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`DeformableDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
    """

    def __init__(self, config: MultiformerConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=inputs_embeds.device)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DeformableDetrDecoder(DeformableDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DeformableDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Deformable DETR:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: DeformableDetrConfig
    """

    def __init__(self, config: MultiformerConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return module(*inputs, output_attentions)
                #
                #     return custom_forward

                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    reference_points_input,
                    spatial_shapes,
                    level_start_index,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    if reference_points.shape[-1] != 2:
                        raise ValueError(
                            f"Reference points' last dimension must be of size 2, but is {reference_points.shape[-1]}"
                        )
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """
    The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
class MultiformerModel(DeformableDetrPreTrainedModel):
    def __init__(self, config: MultiformerConfig):
        super().__init__(config)
        self.config = config

        # Create backbone + positional encoding
        self.backbone = DeformableDetrConvEncoder(config)
        self.position_embedding = build_position_encoding(config)

        # Segformer semantic segmentation head
        self.semantic_head = MultiformerSegmentationHead(config)

        # GLPN depth head
        self.depth_decoder = MultiformerDepthDecoder(config)
        self.depth_head = MultiformerDepthEstimationHead(config)

        extra_channels = 0
        if self.config.det2d_fuse_semantic:
            extra_channels += self.config.backbone_config.num_labels
        if self.config.det2d_fuse_depth:
            extra_channels += 3

        # Create input projection layers
        input_proj_list = []
        input_proj_parms = zip(
            config.det2d_input_feature_levels,
            config.det2d_input_proj_kernels,
            config.det2d_input_proj_strides,
            config.det2d_input_proj_pads,
        )
        for level_idx, kernel, stride, padding in input_proj_parms:
            in_channels = self.backbone.intermediate_channel_sizes[level_idx]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + extra_channels, config.d_model, kernel_size=kernel, stride=stride, padding=padding),
                    nn.GroupNorm(config.det2d_input_proj_groups, config.d_model),
                )
            )
        for _ in range(config.det2d_extra_feature_levels):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + extra_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(config.det2d_input_proj_groups, config.d_model),
                )
            )
            in_channels = config.d_model
        self.input_proj = nn.ModuleList(input_proj_list)

        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""

        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MultiformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], MultiformerModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = MultiformerModel.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        if self.config.train_backbone:
            features = self.backbone(pixel_values, pixel_mask)
        else:
            with torch.no_grad():
                features = self.backbone(pixel_values, pixel_mask)

        logits_semantic = None
        predicted_depth = None
        semantic_fuse = None
        depth_fuse = None

        if MultiformerTask.SEMSEG in self.config.tasks + self.config.train_tasks or self.config.det2d_fuse_semantic:
            if MultiformerTask.SEMSEG in self.config.train_tasks:
                logits_semantic = self.semantic_head([f[0] for f in features])
            else:
                with torch.no_grad():
                    logits_semantic = self.semantic_head([f[0] for f in features])
        if MultiformerTask.DEPTH in self.config.tasks + self.config.train_tasks or self.config.det2d_fuse_depth:
            if MultiformerTask.DEPTH in self.config.train_tasks:
                depth_decoder_out = self.depth_decoder([f[0]for f in features])
                predicted_depth = self.depth_head(depth_decoder_out)
            else:
                with torch.no_grad():
                    depth_decoder_out = self.depth_decoder([f[0]for f in features])
                    predicted_depth = self.depth_head(depth_decoder_out)
        if MultiformerTask.DET_2D in self.config.tasks + self.config.train_tasks:
            # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
            sources = []
            masks = []
            position_embeddings_list = []
            for proj_level, feature_level in enumerate(self.config.det2d_input_feature_levels):
                source, mask = features[feature_level]
                if logits_semantic is not None and self.config.det2d_fuse_semantic:
                    semantic_fuse = nn.functional.interpolate(
                        logits_semantic, size=source.shape[-2:], mode="bilinear", align_corners=False
                    ).detach() # Do not want to backprop loss from this task head
                    source = torch.cat([source, semantic_fuse], axis=1)
                if predicted_depth is not None and self.config.det2d_fuse_depth:
                    depth_rescale = nn.functional.interpolate(
                        predicted_depth.unsqueeze(1), size=source.shape[-2:], mode="bilinear", align_corners=False
                    ).detach()  # Do not want to backprop loss from this task head
                    h, w = source.shape[-2:]
                    uv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="xy")
                    uv = torch.stack([t.flatten() for t in uv]).repeat(batch_size, 1, 1).transpose(-1, -2)
                    uv1 = torch.cat([uv, torch.ones(batch_size, h * w, 1)], dim=-1).to(self.device)
                    # xy_prime = uv - intrinsics[:, :2, 2].unsqueeze(1)
                    # xy = xy_prime * (depth_rescale.flatten(-2).transpose(-1, -2) / torch.stack([intrinsics[..., i, i] for i in range(2)], dim=1).unsqueeze(1))
                    # xyz = torch.cat([xy, depth_rescale.flatten(-2).transpose(-1, -2)], dim=-1)
                    xyz = (torch.linalg.inv(intrinsics) @ uv1.transpose(-1, -2)) * torch.exp(depth_rescale).flatten(-2)
                    source = torch.cat([source, xyz.reshape(batch_size, 3, h, w)], axis=1)
                source = self.input_proj[proj_level](source)
                if self.config.det2d_input_proj_strides[proj_level] > 1:
                    mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_embedding(source, mask).to(source.dtype) if self.config.det2d_use_pos_embed else None
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
                if mask is None:
                    raise ValueError("No attention mask was provided")

            # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
            if self.config.num_feature_levels > len(sources):
                _len_sources = len(sources)
                for level in range(_len_sources, self.config.num_feature_levels):
                    if level == _len_sources:
                        source = features[-1][0]
                    else:
                        source = sources[-1]
                    if logits_semantic is not None and self.config.det2d_fuse_semantic:
                        scale = source.shape[-1] / logits_semantic.shape[-1]
                        semantic_fuse = nn.functional.interpolate(logits_semantic, scale_factor=scale, mode="bilinear", align_corners=False)
                    if predicted_depth is not None and self.config.det2d_fuse_depth:
                        depth_rescale = nn.functional.interpolate(
                            predicted_depth.unsqueeze(1), size=source.shape[-2:], mode="bilinear", align_corners=False
                        ).detach()  # Do not want to backprop loss from this task head
                        h, w = source.shape[-2:]
                        uv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="xy")
                        uv = torch.stack([t.flatten() for t in uv]).repeat(batch_size, 1, 1).transpose(
                            -1, -2)
                        uv1 = torch.cat([uv, torch.ones(batch_size, h * w, 1)], dim=-1).to(self.device)
                        xyz = (torch.linalg.inv(intrinsics) @ uv1.transpose(-1, -2)) * torch.exp(depth_rescale).flatten(-2)
                        source = torch.cat([source, xyz.reshape(batch_size, 3, h, w)], axis=1)
                    if any(t is not None for t in [semantic_fuse, depth_fuse]):
                        source = torch.cat([t for t in [source, semantic_fuse, depth_fuse] if t is not None], dim=-3)
                    source = self.input_proj[level](source)
                    mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.position_embedding(source, mask).to(source.dtype) if self.config.det2d_use_pos_embed else None
                    sources.append(source)
                    masks.append(mask)
                    position_embeddings_list.append(pos_l)

            # Create queries
            query_embeds = None
            if not self.config.two_stage:
                query_embeds = self.query_position_embeddings.weight

            # Prepare encoder inputs (by flattening)
            source_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
                batch_size, num_channels, height, width = source.shape
                spatial_shape = (height, width)
                spatial_shapes.append(spatial_shape)
                source = source.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                lvl_embed = self.level_embed[level].view(1, 1, -1)
                if pos_embed is not None:
                    lvl_pos_embed = lvl_embed + pos_embed.flatten(2).transpose(1, 2)
                else:
                    lvl_pos_embed = lvl_embed.repeat(1, height * width, 1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                source_flatten.append(source)
                mask_flatten.append(mask)
            source_flatten = torch.cat(source_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
            valid_ratios = valid_ratios.float()

            # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
            # Also provide spatial_shapes, level_start_index and valid_ratios
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    inputs_embeds=source_flatten,
                    attention_mask=mask_flatten,
                    position_embeddings=lvl_pos_embed_flatten,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

            # Fifth, prepare decoder inputs
            batch_size, _, num_channels = encoder_outputs[0].shape
            enc_outputs_class = None
            enc_outputs_coord_logits = None
            if self.config.two_stage:
                object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
                    encoder_outputs[0], ~mask_flatten, spatial_shapes
                )

                # hack implementation for two-stage Deformable DETR
                # apply a detection head to each pixel (A.4 in paper)
                # linear projection for bounding box binary classification (i.e. foreground and background)
                enc_outputs_class = self.decoder.class_embed[-1](object_query_embedding)
                # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
                delta_bbox = self.decoder.bbox_embed[-1](object_query_embedding)
                enc_outputs_coord_logits = delta_bbox + output_proposals

                # only keep top scoring `config.two_stage_num_proposals` proposals
                topk = self.config.two_stage_num_proposals
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
                topk_coords_logits = torch.gather(
                    enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )

                topk_coords_logits = topk_coords_logits.detach()
                reference_points = topk_coords_logits.sigmoid()
                init_reference_points = reference_points
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_logits)))
                query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)
            else:
                query_embed, target = torch.split(query_embeds, num_channels, dim=1)
                query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
                target = target.unsqueeze(0).expand(batch_size, -1, -1)
                reference_points = self.reference_points(query_embed).sigmoid()
                init_reference_points = reference_points

            decoder_outputs = self.decoder(
                inputs_embeds=target,
                position_embeddings=query_embed,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=mask_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if not return_dict:
                enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
                tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs

                return tuple_outputs

            return MultiformerModelOutput(
                init_reference_points=init_reference_points,
                last_hidden_state=decoder_outputs.last_hidden_state,
                intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
                intermediate_reference_points=decoder_outputs.intermediate_reference_points,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord_logits=enc_outputs_coord_logits,
                backbone_features=features,
                logits_semantic=logits_semantic,
                predicted_depth=predicted_depth,
            )

        else:
            return MultiformerModelOutput(
                backbone_features=features,
                logits_semantic=logits_semantic,
                predicted_depth=predicted_depth,
            )


class MultiformerSemanticMLP(SegformerMLP):
    """
    Linear Embedding.
    """

    def __init__(self, config: MultiformerConfig, input_dim):
        super(SegformerMLP, self).__init__()
        self.proj = nn.Linear(input_dim, config.semantic_decoder_dim)


class MultiformerSegmentationHead(SegformerDecodeHead):
    def __init__(self, config):
        super(SegformerDecodeHead, self).__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.backbone_config.num_encoder_blocks):
            mlp = MultiformerSemanticMLP(config, input_dim=config.backbone_config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.semantic_decoder_dim * config.backbone_config.num_encoder_blocks,
            out_channels=config.semantic_decoder_dim,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.semantic_decoder_dim)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.semantic_classifier_dropout)
        self.classifier = nn.Conv2d(config.semantic_decoder_dim, config.backbone_config.num_labels, kernel_size=1)

        self.config = config.backbone_config


class MultiformerDepthDecoder(GLPNDecoder):
    def __init__(self, config):
        super(GLPNDecoder, self).__init__()
        # we use features from end -> start
        reserved_hidden_sizes = config.backbone_config.hidden_sizes[::-1]
        out_channels = config.depth_decoder_dim

        self.stages = nn.ModuleList(
            [GLPNDecoderStage(hidden_size, out_channels) for hidden_size in reserved_hidden_sizes]
        )
        # don't fuse in first stage
        self.stages[0].fusion = None

        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)


class MultiformerDepthEstimationHead(GLPNDepthEstimationHead):
    def __init__(self, config):
        super(GLPNDepthEstimationHead, self).__init__()

        self.config = config

        channels = config.depth_decoder_dim
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),
        )


@add_start_docstrings(
    """
    Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DEFORMABLE_DETR_START_DOCSTRING,
)
class Multiformer(DeformableDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*", r"bbox3d_embed\.[1-9]\d*"]

    def __init__(self, config: MultiformerConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder model
        self.model = MultiformerModel(config)

        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )
        # 3D boxes: (n_boxes, 3+2*num_heading_bin+4*num_size_cluster)
        channel_mult = 4 if config.det3d_predict_class else 3
        self.bbox3d_embed = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=3 + 2 * config.det3d_num_heading_bins + channel_mult * config.num_labels,
            num_layers=3
        )

        self.type_mean_size_array = torch.zeros(size=(config.num_labels, 3))
        self.type_mean_id_map: Dict[int, int] = {}
        for i, class_id in enumerate(sorted(config.id2label.keys())):
            self.type_mean_size_array[i] = torch.tensor(config.det3d_type_mean_sizes[config.id2label[class_id]])
            self.type_mean_id_map[class_id] = i

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.model.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.bbox3d_embed = nn.ModuleList([self.bbox3d_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
        if config.two_stage:
            # hack implementation for two-stage
            self.model.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(DEFORMABLE_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MultiformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        labels_3d: Optional[List[dict]] = None,
        labels_semantic: Optional[torch.LongTensor] = None,
        labels_depth: Optional = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MultiformerOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = Multiformer.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
        Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
        Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels[0].get("intrinsics", None) is not None:
            intrinsics = torch.cat([l.pop("intrinsics").unsqueeze(0) for l in labels], dim=0).to(self.device)
        else:
            intrinsics = None

        batch_size, num_channels, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=self.device)

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            intrinsics=intrinsics,
        )

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]

        logits, pred_boxes_2d, pred_boxes_3d = None, None, None
        # features = tuple(level[0] for level in outputs.backbone_features)

        if any(task in self.config.tasks + self.config.train_tasks for task in [MultiformerTask.DET_2D, MultiformerTask.DET_3D]):
            # class logits + predicted bounding boxes
            outputs_classes = []
            outputs_coords_2d = []
            outputs_boxes_3d = []

            for level in range(hidden_states.shape[1]):
                if not self.config.auxiliary_loss and level != hidden_states.shape[1] - 1:
                    continue
                if level == 0:
                    reference = init_reference
                else:
                    reference = inter_references[:, level - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[level](hidden_states[:, level])
                outputs_classes.append(outputs_class)

                if MultiformerTask.DET_2D in self.config.tasks + self.config.train_tasks:
                    delta_bbox = self.bbox_embed[level](hidden_states[:, level])
                    if reference.shape[-1] == 4:
                        outputs_coord_logits = delta_bbox + reference
                    elif reference.shape[-1] == 2:
                        delta_bbox[..., :2] += reference
                        outputs_coord_logits = delta_bbox
                    else:
                        raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
                    outputs_coord_2d = outputs_coord_logits.sigmoid()
                    outputs_coords_2d.append(outputs_coord_2d)

                if MultiformerTask.DET_3D in self.config.tasks + self.config.train_tasks:
                    box_output = self.bbox3d_embed[level](hidden_states[:, level])
                    center_coord_logits = box_output[..., :2] + reference
                    z_offset = box_output[..., 2].unsqueeze(-1)
                    z_sample = nn.functional.grid_sample(
                        outputs.predicted_depth.unsqueeze(1),
                        center_coord_logits.tanh().unsqueeze(-2),
                        align_corners=False,
                        padding_mode="reflection",
                    ).squeeze(1)
                    center_coord_2d = center_coord_logits.sigmoid()
                    z_sample += z_offset
                    uv = center_coord_2d * torch.tensor(pixel_values.transpose(-1, -2).shape[-2:]).to(self.device)
                    uv1 = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
                    xyz = (torch.linalg.inv(intrinsics) @ uv1.transpose(-1, -2)) * torch.exp(z_sample).transpose(-1, -2)
                    num_heading_bins = self.config.det3d_num_heading_bins
                    heading_scores = box_output[..., 3:num_heading_bins+3]
                    heading_residuals_normalized = box_output[..., num_heading_bins+3:num_heading_bins*2+3]
                    size_scores = box_output[..., -4*self.config.num_labels:-3*self.config.num_labels]
                    size_residuals_normalized = box_output[..., -3*self.config.num_labels:]
                    output_boxes_3d = torch.cat(
                        tensors=[
                            center_coord_2d,
                            xyz.transpose(-1, -2),
                            heading_scores,
                            heading_residuals_normalized,
                            size_scores,
                            size_residuals_normalized,
                        ],
                        dim=-1
                    )

                    outputs_boxes_3d.append(output_boxes_3d)

            outputs_class = torch.stack(outputs_classes)
            logits = outputs_class[-1]

            if len(outputs_coords_2d) > 0:
                outputs_coord_2d = torch.stack(outputs_coords_2d)
                pred_boxes_2d = outputs_coord_2d[-1]

            if len(outputs_boxes_3d) > 0:
                outputs_boxes_3d = torch.stack(outputs_boxes_3d)
                pred_boxes_3d = outputs_boxes_3d[-1]

        loss = {}
        if "semseg" in self.config.train_tasks:
            if labels_semantic is None:
                warnings.warn("'semseg' is selected as a training task, but no semantic labels loaded for frame.")
            else:
                # upsample logits to the images' original size
                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits_semantic, size=labels_semantic.shape[-2:], mode="bilinear", align_corners=False
                )
                if self.config.num_labels > 1:
                    loss_fct = CrossEntropyLoss(
                        ignore_index=self.config.semantic_loss_ignore_index, weight=self.class_loss_weights
                    )
                    labels_loss = loss_fct(upsampled_logits, labels_semantic)
                elif self.config.num_labels == 1:
                    valid_mask = (
                        (labels_semantic >= 0) & (labels_semantic != self.config.semantic_loss_ignore_index)
                    ).float()
                    loss_fct = BCEWithLogitsLoss(reduction="none")
                    labels_loss = loss_fct(upsampled_logits.squeeze(1), labels_semantic.float())
                    labels_loss = (labels_loss * valid_mask).mean()
                else:
                    raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

                loss[MultiformerTask.SEMSEG] = labels_loss

        if "depth" in self.config.train_tasks:
            if labels_depth is None:
                warnings.warn("'depth' is selected as a training task, but no depth labels loaded for frame.")
            else:
                loss_fct = DepthTrainLoss(silog_lambda=self.config.silog_lambda)
                # Labels are converted to log by loss function, model inference is in log depth
                loss[MultiformerTask.DEPTH] = loss_fct(outputs.predicted_depth, labels_depth)

        # loss, loss_dict, auxiliary_outputs = None, None, None
        loss_dict, auxiliary_outputs = None, None
        if labels is not None and any(task in self.config.train_tasks for task in [MultiformerTask.DET_2D, MultiformerTask.DET_3D]):
            if labels is None:
                if MultiformerTask.DET_2D in self.config.train_tasks:
                    warnings.warn("'det2d' listed as a training task, but no box2d labels loaded for frame.")
                if MultiformerTask.DET_3D in self.config.train_tasks:
                    warnings.warn("'det3d' requires box2d labels to train, but none loaded for frame.")
            else:
                # First: create the matcher
                matcher = DeformableDetrHungarianMatcher(
                    class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
                )
                # Second: create the criterion
                losses = ["labels", "boxes", "cardinality"]
                criterion = DeformableDetrLoss(
                    matcher=matcher,
                    num_classes=self.config.num_labels,
                    focal_alpha=self.config.focal_alpha,
                    losses=losses,
                    keep_box_prob=self.config.det2d_box_keep_prob,
                )
                criterion.to(self.device)
                # Third: compute the losses, based on outputs and labels
                outputs_loss = {}
                outputs_loss["logits"] = logits
                outputs_loss["pred_boxes"] = pred_boxes_2d
                if self.config.auxiliary_loss:
                    auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord_2d)
                    outputs_loss["auxiliary_outputs"] = auxiliary_outputs
                if self.config.two_stage:
                    enc_outputs_coord = outputs.enc_outputs_coord_logits.sigmoid()
                    outputs_loss["enc_outputs"] = {"logits": outputs.enc_outputs_class, "pred_boxes": enc_outputs_coord}

                loss_dict, indices = criterion(outputs_loss, labels)
                # Fourth: compute total loss, as a weighted sum of the various losses
                weight_dict = {
                    "loss_ce": 1,
                    "loss_bbox": self.config.bbox_loss_coefficient,
                    "loss_giou": self.config.giou_loss_coefficient,
                }
                if self.config.auxiliary_loss:
                    aux_weight_dict = {}
                    for i in range(self.config.decoder_layers - 1):
                        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                    weight_dict.update(aux_weight_dict)

                if MultiformerTask.DET_2D in self.config.train_tasks:
                    loss[MultiformerTask.DET_2D] = sum(
                        loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
                    )

                if MultiformerTask.DET_3D in self.config.train_tasks:
                    if labels_3d is None:
                        warnings.warn("'det3d' listed as training tasks, but no box3d labels found for frame.")
                    else:
                        criterion_3d = MultiformerDet3DLoss(self.config, self.type_mean_size_array)
                        image_size_wh = torch.cat([pixel_mask.sum(-1)[:, -1:], pixel_mask.sum(-2)[:, -1:]], -1)
                        total_loss_3d, loss_dict_3d = criterion_3d(pred_boxes_3d, labels_3d, indices, intrinsics, image_size_wh)
                        loss[MultiformerTask.DET_3D] = total_loss_3d
                        loss_dict.update(loss_dict_3d)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes_2d) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes_2d) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = MultiformerOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes_2d,
            pred_boxes_3d=pred_boxes_3d,
            logits_semantic=outputs.logits_semantic,
            pred_depth=outputs.predicted_depth,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

        return dict_outputs


# Copied from transformers.models.detr.modeling_detr.dice_loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SiLogLoss(torch.nn.Module):
    r"""
    Implements the Scale-invariant log scale loss [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ where $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """

    def __init__(self, lambd=0.5, log_predictions=True, log_labels=False, sqrt_output=False):
        super().__init__()
        self.lambd = lambd
        self.log_predictions = log_predictions
        self.log_labels = log_labels
        self.sqrt_output = sqrt_output

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = target[valid_mask] if self.log_labels else torch.log(target[valid_mask])
        y_hat = pred[valid_mask] if self.log_predictions else torch.log(pred[valid_mask])
        diff = y_hat - y
        loss = torch.pow(diff, 2).mean() - self.lambd * torch.pow(diff.mean(), 2)

        return loss if not self.sqrt_output else torch.sqrt(loss)


class IRMSELoss(torch.nn.Module):
    r"""
    Implements the Root Mean Squared Error of Inverse Depth [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).
    """

    def __init__(self, log_predictions=True, log_labels=False):
        super().__init__()
        self.log_predictions = log_predictions
        self.log_labels = log_labels

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = torch.exp(target[valid_mask]) if self.log_labels else target[valid_mask]
        y_hat = torch.exp(pred[valid_mask]) if self.log_predictions else pred[valid_mask]
        irmse = torch.sqrt(torch.pow((1000 / y) - (1000 / y_hat), 2).mean())

        return irmse


class MAELoss(torch.nn.Module):
    def __init__(self, log_predictions=True, log_labels=False):
        super().__init__()
        self.log_predictions = log_predictions
        self.log_labels = log_labels

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = torch.exp(target[valid_mask]) if self.log_labels else target[valid_mask]
        y_hat = torch.exp(pred[valid_mask]) if self.log_predictions else pred[valid_mask]
        loss = torch.nn.L1Loss(reduction="mean")
        mae = loss(y_hat, y)

        return mae


class DepthTrainLoss(torch.nn.Module):
    def __init__(
        self,
        losses={"silog": 1.0, "mae": 0.1},
        log_predictions=True,
        log_labels=False,
        silog_lambda=0.5,
        sqrt_silog=False,
    ):
        super().__init__()
        self.losses = {k: {"weight": v} for k, v in losses.items()}
        if "silog" in self.losses:
            self.losses["silog"]["func"] = SiLogLoss(
                lambd=silog_lambda, sqrt_output=sqrt_silog, log_predictions=log_predictions, log_labels=log_labels
            )
        if "mae" in self.losses:
            self.losses["mae"]["func"] = MAELoss(log_predictions=log_predictions, log_labels=log_labels)

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        y = target[valid_mask]
        y_hat = pred[valid_mask]
        results = []
        for loss in self.losses:
            results = self.losses[loss]["func"](y_hat, y) * self.losses[loss]["weight"]

        return torch.sum(results)


def boxes3d_to_corners3d_torch(boxes3d, flip=False):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """
    device = boxes3d.device
    boxes_num = boxes3d.shape[0]
    h, w, l, ry = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
    if flip:
        ry = ry + torch.pi
    centers = boxes3d[:, 0:3]
    zeros = torch.FloatTensor(boxes_num, 1).fill_(0).to(device)
    ones = torch.FloatTensor(boxes_num, 1).fill_(1).to(device)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim=1)  # (N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=1)  # (N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim=1), y_corners.unsqueeze(dim=1), z_corners.unsqueeze(dim=1)), dim=1) # (N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, zeros, sina], dim=1)
    raw_2 = torch.cat([zeros, ones, zeros], dim=1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim=2).expand(-1, -1, 8)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated


class MultiformerDet3DLoss(nn.Module):

    def __init__(self, config: MultiformerConfig, type_mean_size_array: Tensor, box_loss_weight=1.0, corner_loss_weight=0.1):
        super().__init__()
        self.config = config
        self.type_mean_size_array = type_mean_size_array
        self.box_loss_weight = box_loss_weight
        self.corner_loss_weight = corner_loss_weight

    def forward(self, boxes, labels_3d, indices, intrinsics, image_size_wh):
        # Adapted from https://github.com/xinzhuma/patchnet/blob/fbff77fa6cc9cf108f475a97440d16c5a37a6b9f/lib/losses/patchnet_loss.py
        batch_size = len(labels_3d)
        assert batch_size == boxes.shape[0] == len(indices)
        device = boxes.device

        center_pred = torch.cat([boxes[i, indices[i][0], 2:5] for i in range(batch_size)])
        center_gt = torch.cat([labels_3d[i]["boxes3d"][indices[i][1]][:, :3] for i in range(batch_size)])
        center_loss = F.l1_loss(center_pred, center_gt)

        center_2d_pred = torch.cat([boxes[i, indices[i][0], :2] for i in range(batch_size)])
        center_2d_label = torch.cat(
            [
                (intrinsics[i] @ labels_3d[i]["boxes3d"][indices[i][1]][:, :3].T).T[..., :2] / image_size_wh[i]
                for i in range(batch_size)
            ]
        ) / center_gt[:, 2, None]
        center_2d_loss = F.l1_loss(center_2d_pred, center_2d_label)

        num_heading_bins = self.config.det3d_num_heading_bins

        # Heading bin classification loss
        heading_class_pred = torch.cat([boxes[i, indices[i][0], 5:num_heading_bins + 5] for i in range(batch_size)])
        heading_class_label = torch.cat([labels_3d[i]["heading_class_labels"][indices[i][1]].long() for i in range(batch_size)])
        heading_class_loss = F.cross_entropy(heading_class_pred, heading_class_label)

        # Normalized heading residual loss
        hcls_onehot = torch.zeros_like(heading_class_pred).scatter_(
            dim=1, index=heading_class_label.view(-1, 1), value=1)
        heading_residual_label = torch.cat([labels_3d[i]["heading_residual_labels"][indices[i][1]] for i in range(batch_size)])
        # heading_residual_label = heading_residual_label.float()
        heading_residual_normalized_label = heading_residual_label / (torch.pi / num_heading_bins)
        heading_residual_normalized_pred = torch.cat([boxes[i, indices[i][0], 5+num_heading_bins:5+2*num_heading_bins] for i in range(batch_size)])
        heading_residual_normalized = torch.sum(heading_residual_normalized_pred * hcls_onehot, 1)
        heading_residual_normalized_loss = F.l1_loss(heading_residual_normalized, heading_residual_normalized_label)

        # Size loss. Assumes that each box class has its own size class
        size_class_label = torch.cat([labels_3d[i]["class_labels"][indices[i][1]] for i in range(batch_size)])
        size_class_loss = None
        if self.config.det3d_predict_class:
            size_scores = torch.cat([boxes[i, indices[i][0], -4*self.config.num_labels:-3*self.config.num_labels] for i in range(batch_size)])
            size_class_loss = F.cross_entropy(size_scores, size_class_label.long())
        scls_onehot = torch.zeros(size_class_label.shape[0], self.config.num_labels).to(device).scatter_(
            dim=1, index=size_class_label.long().view(-1, 1), value=1
        )
        scls_onehot = scls_onehot.view(size_class_label.shape[0], self.config.num_labels, 1).repeat(1, 1, 3)
        size_residual_normalized_pred = torch.cat([boxes[i, indices[i][0], -3*self.config.num_labels:] for i in range(batch_size)])
        size_residual_normalized = torch.sum(size_residual_normalized_pred.view(scls_onehot.shape) * scls_onehot, 1)

        type_mean_size_array = self.type_mean_size_array.to(device)
        mean_size_label = torch.sum(type_mean_size_array * scls_onehot, 1)
        size_residual_label = torch.cat([labels_3d[i]["size_residual_labels"][indices[i][1]] for i in range(batch_size)])
        size_residual_label_normalized = size_residual_label / mean_size_label
        size_residual_normalized_loss = F.l1_loss(size_residual_label_normalized, size_residual_normalized)

        # Corner loss
        size_residuals = size_residual_normalized_pred.view(scls_onehot.shape) * type_mean_size_array
        size_pred = size_residuals + type_mean_size_array
        size_pred = torch.sum(size_pred * scls_onehot, 1)

        # true pred heading
        heading_bin_centers = torch.arange(0, 2 * torch.pi, 2 * torch.pi / num_heading_bins).to(device)
        heading_pred = heading_residual_normalized_pred * (torch.pi / num_heading_bins) + heading_bin_centers
        heading_pred = torch.sum(heading_pred * hcls_onehot, 1)

        # true heading label
        heading_label = heading_residual_label.view(-1, 1) + heading_bin_centers.view(1, -1)
        heading_label = torch.sum(hcls_onehot * heading_label, -1)

        # size true label
        size_label = torch.sum(type_mean_size_array * scls_onehot, 1) + size_residual_label
        size_label = size_label.float()

        # corners_3d estimate
        box3d_pred = torch.cat([center_pred, size_pred, heading_pred.view(-1, 1)], 1)
        corners_3d_pred = boxes3d_to_corners3d_torch(box3d_pred)

        # true 3d corners
        box3d = torch.cat([center_gt, size_label, heading_label.view(-1, 1)], 1)
        corners_3d_gt = boxes3d_to_corners3d_torch(box3d)
        corners_3d_gt_flip = boxes3d_to_corners3d_torch(box3d, flip=True)

        # corner loss
        corners_loss = torch.min(F.l1_loss(corners_3d_pred, corners_3d_gt),
                                 F.l1_loss(corners_3d_pred, corners_3d_gt_flip))

        loss_dict = {
            "box3d_center_2d_l1_loss": center_2d_loss,
            "box3d_center_3d_l1_loss": center_loss,
            "box3d_heading_class_loss": heading_class_loss,
            "box3d_heading_residual_loss": heading_residual_normalized_loss,
            "box3d_size_residual_loss": size_residual_normalized_loss,
            "box3d_corners_loss": corners_loss,
        }

        combined_loss = (
                center_2d_loss +
                center_loss * 0.1 +
                heading_class_loss +
                heading_residual_normalized_loss * 10 +
                size_residual_normalized_loss * 10 +
                self.corner_loss_weight * corners_loss
        )

        if size_class_loss is not None:
            loss_dict["box3d_size_class_loss"] = size_class_loss
            combined_loss += size_class_loss

        total_loss = self.box_loss_weight * combined_loss

        return total_loss, loss_dict


class DeformableDetrLoss(nn.Module):
    """
    This class computes the losses for `DeformableDetrForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).

    Args:
        matcher (`DeformableDetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, focal_alpha, losses, keep_box_prob=0.35):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.keep_box_prob = keep_box_prob

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes, threshold=None):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        if threshold is None:
            threshold = self.keep_box_prob
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.sigmoid().max(-1)[0] >= threshold).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        true_diff = card_pred.float() - target_lengths.float()
        losses = {"cardinality_error": card_err, "num_boxes_error": {}}
        for i, value in enumerate(true_diff):
            losses["num_boxes_error"][i] = value
        return losses

    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, indices


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class DeformableDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copied from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# Copied from transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
