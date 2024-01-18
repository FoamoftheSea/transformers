from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, List

import torch
from torch import nn
from transformers import DeformableDetrModel, add_start_docstrings, DeformableDetrPreTrainedModel, DillCLIPVisionConfig
from transformers.models.deformable_detr.modeling_deformable_detr import _get_clones, DeformableDetrModelOutput, \
    _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, ModelOutput

DILL_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FoamoftheSea/dill-clip-b5",
    # See all Dill-CLIP models at https://huggingface.co/models?filter=deformable-detr
]


DILL_CLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DillCLIPVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DILL_CLIP_INPUTS_DOCSTRING = r"""
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


@dataclass
class DillCLIPVisionModelForRegressionOutput(ModelOutput):
    """
    Base class for outputs of the DillCLIP Vision model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
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
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
            picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
            foreground and background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    init_reference_points: torch.FloatTensor = None
    enc_outputs_class: Optional = None
    enc_outputs_coord_logits: Optional = None

@add_start_docstrings(
    """
    The bare Dill-CLIP Vision Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top. These hidden states should be trained to match expert CLIP models.
    """,
    DILL_CLIP_START_DOCSTRING,
)
class DillCLIPVisionModel(DeformableDetrModel):
    ...


@add_start_docstrings(
    """
    Dill-CLIP Model (consisting of a backbone and encoder-decoder Transformer) with l2 loss function related to
    expert soft targets. These soft targets should be the target hidden states from a large expert CLIP model.
    """,
    DILL_CLIP_START_DOCSTRING,
)
class DillCLIPVisionModelForRegression(DeformableDetrPreTrainedModel):
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: DillCLIPVisionConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder model
        self.model = DillCLIPVisionModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(DILL_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DillCLIPVisionModelForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DillCLIPVisionModelForRegressionOutput]:
        r"""
        labels (`List[toch.FloatTensor]` of len `(batch_size,)`, *optional*):
            The soft targets generated by CLIP-ViT-L expert for knowledge distillation. Should be shape(257, 1024).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through Dill-CLIP base model to obtain encoder + decoder outputs
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
        )

        # hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]

        last_hidden_state = outputs.last_hidden_state

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            criterion = nn.MSELoss()
            loss = criterion(last_hidden_state, torch.stack(labels))

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (last_hidden_state, ) + auxiliary_outputs + outputs
            else:
                output = (last_hidden_state, ) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = DillCLIPVisionModelForRegressionOutput(
            loss=loss,
            loss_dict=loss_dict,
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
