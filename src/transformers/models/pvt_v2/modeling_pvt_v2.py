# coding=utf-8
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
""" PyTorch PVTv2 model."""

import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.modules.utils import _ntuple

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_pvt_v2 import PvtV2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PvtV2Config"

_CHECKPOINT_FOR_DOC = "FoamoftheSea/pvt-v2-b0-224"
_EXPECTED_OUTPUT_SHAPE = [1, 50, 512]

_IMAGE_CLASS_CHECKPOINT = "FoamoftheSea/pvt-v2-b0-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

PVT_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FoamoftheSea/pvt-v2-b0-224"
    # See all PVT models at https://huggingface.co/models?filter=pvt
]


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Pvt
class PvtV2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class PvtV2OverlapPatchEmbeddings(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: Union[int, Iterable[int]],
        stride: int,
        num_channels: int,
        hidden_size: int,
    ):
        super().__init__()
        self.fp16_enabled = False
        patch_size = _ntuple(2)(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class PvTV2DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class PvtV2SelfAttention(nn.Module):
    """Efficient self-attention mechanism."""

    def __init__(
        self, config: PvtV2Config, hidden_size: int, num_attention_heads: int, sr_ratio: int
    ):
        super().__init__()
        self.attn_reduce = config.attn_reduce
        self.pruned_heads = set()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

        self.sr_ratio = sr_ratio
        if self.attn_reduce == "AP":
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1)
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.act = nn.GELU()
        elif sr_ratio > 1:
            self.sr = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, hidden_states) -> torch.Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.attn_reduce == "AP":
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.sr(self.pool(hidden_states)).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.act(self.layer_norm(hidden_states))
        elif self.sr_ratio > 1:
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.sr(hidden_states).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_drop(attention_probs)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(new_context_layer_shape)
        context_layer = (attention_probs @ value_layer).transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.proj = prune_linear_layer(self.proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


class PvtV2ConvFFN(nn.Module):
    def __init__(
        self,
        config: PvtV2Config,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = PvTV2DWConv(hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, height, width) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtV2Layer(nn.Module):
    def __init__(
        self,
        config: PvtV2Config,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sr_ratio: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtV2SelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sr_ratio=sr_ratio,
        )
        self.drop_path = PvtV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtV2ConvFFN(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False):
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        mlp_output = self.drop_path(mlp_output)
        layer_output = hidden_states + mlp_output

        outputs = (layer_output,) + outputs

        return outputs


class PvtV2Encoder(nn.Module):
    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()

        # patch embeddings
        embeddings = []

        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PvtV2OverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    PvtV2Layer(
                        config=config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sr_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]
        num_blocks = len(self.block)
        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for block in block_layer:
                layer_outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions and idx in self.config._out_indices:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states and idx in self.config._out_indices:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PvtV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PvtV2Config
    base_model_prefix = "pvt_v2"
    main_input_name = "pixel_values"

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module: PvtV2Encoder, value: bool = False):
        if isinstance(module, PvtV2Encoder):
            module.gradient_checkpointing = value


PVT_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PVT_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. 
            See [`PvtV2ImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Pvt-v2 encoder outputting raw hidden-states without any specific head on top.",
    PVT_V2_START_DOCSTRING,
)
class PvtV2Model(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = PvtV2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    Pvt-v2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    PVT_V2_START_DOCSTRING,
)
class PvtV2ForImageClassification(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pvt_v2 = PvtV2Model(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.pvt_v2(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )