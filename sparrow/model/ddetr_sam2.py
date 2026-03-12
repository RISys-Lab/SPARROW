import json
import copy
import torch
from torch import nn
from typing import Optional, Tuple, Union

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    DeformableDetrConfig,
)
from transformers.utils import logging
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrObjectDetectionOutput

from sparrow.model.ddetr_transformer import DeformableDetrTransformer
from model.segment_anything_2.sam2.build_sam import build_sam2

logger = logging.get_logger(__name__)


class LayerNorm(nn.Module):
    """
    LayerNorm over channel dimension for tensors with shape
    (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CustomDDETRSAM2Config(PretrainedConfig):
    model_type = "ddetr_sam2"

    def __init__(
        self,
        sam2_cfg="sam2_hiera_l.yaml",
        zs_weight_path=None,
        ddetr_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if ddetr_cfg is None:
            self.ddetr_cfg = DeformableDetrConfig()
            logger.info("ddetr_cfg is None. Initializing DeformableDetrConfig with default values.")
        elif isinstance(ddetr_cfg, dict):
            self.ddetr_cfg = DeformableDetrConfig(**ddetr_cfg)
        elif isinstance(ddetr_cfg, DeformableDetrConfig):
            self.ddetr_cfg = ddetr_cfg
        else:
            raise NotImplementedError("currently only supports DeformableDetrConfig as detector head config.")

        self.sam2_cfg = sam2_cfg
        self.zs_weight_path = zs_weight_path

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = copy.deepcopy(self)
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_diff_dict()
            config_dict = config_dict.to_diff_dict()
        else:
            config_dict = copy.deepcopy(self)
            config_dict.ddetr_cfg = config_dict.ddetr_cfg.to_dict()
            config_dict = config_dict.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class CustomDDETRSAM2Model(PreTrainedModel):
    config_class = CustomDDETRSAM2Config

    def __init__(self, config: CustomDDETRSAM2Config, pretrained_vis_encoder=None):
        super().__init__(config)

        sam2_model = build_sam2(config.sam2_cfg, pretrained_vis_encoder, device=None, mode="eval")
        self.vis_encoder = sam2_model.image_encoder
        del sam2_model

        self.ddetr_transformer = DeformableDetrTransformer(config.ddetr_cfg, config.zs_weight_path)

        self._freeze_vis_encoder = False

        num_feature_levels = config.ddetr_cfg.num_feature_levels
        feature_channels = [layer.conv.out_channels for layer in self.vis_encoder.neck.convs]
        if self.vis_encoder.scalp > 0:
            feature_channels = feature_channels[: -self.vis_encoder.scalp]

        if len(feature_channels) == 0:
            raise ValueError("SAM2 image encoder produced no feature levels.")

        num_base_levels = min(num_feature_levels, len(feature_channels))
        base_channels = feature_channels[-num_base_levels:]

        input_proj_list = []
        for in_channels in base_channels:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.ddetr_cfg.d_model, kernel_size=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                )
            )

        for _ in range(num_base_levels, num_feature_levels):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(config.ddetr_cfg.d_model, config.ddetr_cfg.d_model, kernel_size=3, stride=2, padding=1),
                    LayerNorm(config.ddetr_cfg.d_model),
                )
            )

        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def freeze_vis_encoder(self):
        self.vis_encoder.requires_grad_(False)
        self.vis_encoder.eval()
        self._freeze_vis_encoder = True

    def freeze_ddetr(self):
        self.ddetr_transformer.requires_grad_(False)

    def get_vis_encoder(self):
        return getattr(self, "vis_encoder", None)

    def get_ddetr(self):
        return getattr(self, "ddetr_transformer", None)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_vis_encoder:
            self.vis_encoder.eval()
        return self

    def _extract_sam2_features(self, images: torch.FloatTensor):
        if self._freeze_vis_encoder:
            self.vis_encoder.eval()
            with torch.no_grad():
                backbone_out = self.vis_encoder(images)
        else:
            backbone_out = self.vis_encoder(images)

        feature_maps = backbone_out["backbone_fpn"]
        if len(feature_maps) == 0:
            raise ValueError("SAM2 image encoder returned no backbone features.")

        num_feature_levels = len(self.input_proj)
        num_base_levels = min(num_feature_levels, len(feature_maps))
        selected_feats = feature_maps[-num_base_levels:]

        srcs = [proj(feat) for proj, feat in zip(self.input_proj[:num_base_levels], selected_feats)]

        for level in range(num_base_levels, num_feature_levels):
            srcs.append(self.input_proj[level](srcs[-1]))

        return srcs

    def forward(
        self,
        images: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DeformableDetrObjectDetectionOutput]:
        srcs = self._extract_sam2_features(images)
        masks = [
            torch.ones((src.shape[0],) + src.shape[2:], dtype=torch.bool, device=src.device)
            for src in srcs
        ]

        return self.ddetr_transformer(
            sources=srcs,
            masks=masks,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


AutoConfig.register("ddetr_sam2", CustomDDETRSAM2Config)
AutoModel.register(CustomDDETRSAM2Config, CustomDDETRSAM2Model)
