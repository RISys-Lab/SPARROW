from typing import List, Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou, roi_align
from transformers import BitsAndBytesConfig, CLIPVisionModel
from PIL import Image

from .videogpt_plus.model.language_model.phi3 import (VideoGPTPlusPhi3ForCausalLM, VideoGPTPlusPhi3Model)
from .segment_anything import build_sam_vit_h
from .segment_anything_2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from model.ddetr_helper import DDETRProposalGenerator

from util.misc import print_dimensions 

MASK_IGNORE_INDEX = -1
MAX_NUM_SEG_TOKENS_PER_SAMPLE = 4

import torch
import torch.nn.functional as F

SAM2_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM2_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

from torch.backends.cuda import enable_math_sdp
enable_math_sdp(True)


def _mask_to_xyxy(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Convert a binary mask (T, H, W) or (H, W) into normalized xyxy bounding box.
    Returns None if mask is empty.
    """
    if mask.dim() == 3:
        # collapse temporal dimension (ignore negative values)
        valid = mask != MASK_IGNORE_INDEX
        mask = torch.where(valid, mask > 0, torch.zeros_like(mask, dtype=torch.bool))
        mask = mask.any(dim=0)
    else:
        mask = torch.where(mask == MASK_IGNORE_INDEX, torch.zeros_like(mask, dtype=torch.bool), mask > 0)

    if not mask.any():
        return None

    indices = mask.nonzero(as_tuple=False)
    y_min = indices[:, 0].min().float()
    x_min = indices[:, 1].min().float()
    y_max = indices[:, 0].max().float()
    x_max = indices[:, 1].max().float()

    h, w = mask.shape[-2], mask.shape[-1]
    box = torch.tensor(
        [x_min / w, y_min / h, (x_max + 1) / w, (y_max + 1) / h],
        device=mask.device,
        dtype=torch.float32,
    )
    return box.clamp(0.0, 1.0)


class ProposalSelector(nn.Module):
    """
    Lightweight module that scores DDETR proposals conditioned on [SEG] embeddings.
    """

    def __init__(self, seg_dim: int, prop_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seg_proj = nn.Linear(seg_dim, hidden_dim)
        self.prop_proj = nn.Linear(prop_dim, hidden_dim)
        self.fuse_proj = nn.Linear(prop_dim, seg_dim)
        # Optional box refinement branch (seg-token, proposal) -> normalized xyxy.
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        # Keep initialization conservative to make this branch minimally intrusive.
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def forward(
        self,
        seg_embeddings: torch.Tensor,
        proposal_embeddings: torch.Tensor,
        proposal_scores: Optional[torch.Tensor] = None,
        proposal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seg_embeddings: (N_seg, seg_dim)
            proposal_embeddings: (N_prop, prop_dim)
            proposal_scores: optional (N_prop,)
            proposal_mask: optional bool tensor (N_prop,) indicating valid proposals
        Returns:
            refined_seg_embeddings: (N_seg, seg_dim)
            selection_logits: (N_seg, N_prop)
            attention_weights: (N_seg, N_prop)
            pred_boxes_norm: (N_seg, N_prop, 4) normalized xyxy
        """
        if seg_embeddings.numel() == 0 or proposal_embeddings.numel() == 0:
            empty_logits = seg_embeddings.new_empty(seg_embeddings.shape[0], proposal_embeddings.shape[0])
            empty_boxes = seg_embeddings.new_empty(seg_embeddings.shape[0], proposal_embeddings.shape[0], 4)
            return seg_embeddings, empty_logits, empty_logits, empty_boxes

        seg_q = self.seg_proj(seg_embeddings)  # (N_seg, hidden_dim)

        try:
            prop_k = self.prop_proj(proposal_embeddings)  # (N_prop, hidden_dim)
        except:
            prop_k = self.prop_proj(proposal_embeddings.half())  # (N_prop, hidden_dim)

        logits = (seg_q @ prop_k.t()) / math.sqrt(self.hidden_dim)

        if proposal_scores is not None:
            logits = logits + proposal_scores.unsqueeze(0)

        if proposal_mask is not None:
            logits = logits.masked_fill(~proposal_mask.unsqueeze(0), float("-inf"))

        attention = torch.softmax(logits, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        attended_props = attention @ proposal_embeddings  # (N_seg, prop_dim)

        pair_seg = seg_q.unsqueeze(1).expand(-1, prop_k.shape[0], -1)
        pair_prop = prop_k.unsqueeze(0).expand(seg_q.shape[0], -1, -1)
        pair_feat = torch.cat([pair_seg, pair_prop], dim=-1)  # (N_seg, N_prop, 2*hidden)
        pred_box_raw = self.box_head(pair_feat)
        pred_box_sigmoid = pred_box_raw.sigmoid()
        # Enforce xyxy ordering after sigmoid projection.
        pred_x1y1 = torch.minimum(pred_box_sigmoid[..., :2], pred_box_sigmoid[..., 2:])
        pred_x2y2 = torch.maximum(pred_box_sigmoid[..., :2], pred_box_sigmoid[..., 2:])
        pred_boxes_norm = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        
        try:
            refined_seg = seg_embeddings + self.fuse_proj(attended_props)
        except:
            refined_seg = seg_embeddings + self.fuse_proj(attended_props.half())

        return refined_seg, logits, attention, pred_boxes_norm


class ROIProposalHelper(nn.Module):
    """
    ROI feature helper head that scores proposals conditioned on SEG embeddings.
    This is additive to the original ProposalSelector and does not replace it.
    """

    def __init__(
        self,
        seg_dim: int,
        hidden_dim: int = 256,
        roi_size: int = 7,
        max_levels: int = 3,
        level_in_dims: Tuple[int, ...] = (32, 64, 256),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size
        self.max_levels = max_levels
        self.level_in_dims = level_in_dims

        self.query_proj = nn.Linear(seg_dim, hidden_dim)
        self.seg_proj = nn.Linear(seg_dim, hidden_dim)
        self.level_token_proj = nn.ModuleList(
            [nn.Linear(level_in_dims[i], hidden_dim) for i in range(max_levels)]
        )
        self.level_pool_proj = nn.ModuleList(
            [nn.Linear(level_in_dims[i], hidden_dim) for i in range(max_levels)]
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _match_channel_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        in_dim = x.shape[-1]
        if in_dim == target_dim:
            return x
        if in_dim > target_dim:
            return x[..., :target_dim]
        pad = torch.zeros(
            x.shape[:-1] + (target_dim - in_dim,),
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=-1)

    def _proposal_to_pixel_boxes(
        self,
        proposal_boxes: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = image_hw
        boxes = proposal_boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * float(w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * float(h)
        return boxes

    def _roi_align_multiscale(
        self,
        feature_pyramid: List[torch.Tensor],
        boxes_px: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> List[torch.Tensor]:
        image_h, _ = image_hw
        roi_feats: List[torch.Tensor] = []
        for feat in feature_pyramid[: self.max_levels]:
            feat_h = feat.shape[-2]
            spatial_scale = float(feat_h) / float(image_h)
            roi = roi_align(
                feat.float(),
                [boxes_px.float()],
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )
            roi_feats.append(roi.to(feat.dtype))
        return roi_feats

    def forward(
        self,
        seg_embeddings: torch.Tensor,
        proposal_boxes: torch.Tensor,
        feature_pyramid: List[torch.Tensor],
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            seg_embeddings: (N_seg, seg_dim)
            proposal_boxes: (N_prop, 4), normalized xyxy
            feature_pyramid: list of feature maps, each shaped (1, C, H, W)
            image_hw: (H, W) for proposal-to-pixel conversion
        Returns:
            roi_logits: (N_seg, N_prop)
        """
        num_seg = seg_embeddings.shape[0]
        num_prop = proposal_boxes.shape[0]
        if num_seg == 0 or num_prop == 0 or len(feature_pyramid) == 0:
            return seg_embeddings.new_empty((num_seg, num_prop))

        boxes_px = self._proposal_to_pixel_boxes(proposal_boxes, image_hw)
        roi_levels = self._roi_align_multiscale(feature_pyramid, boxes_px, image_hw)
        if len(roi_levels) == 0:
            return seg_embeddings.new_empty((num_seg, num_prop))

        token_levels: List[torch.Tensor] = []
        pooled_levels: List[torch.Tensor] = []
        level_count = min(len(roi_levels), self.max_levels)
        for level_idx in range(level_count):
            roi = roi_levels[level_idx]  # (N_prop, C, roi, roi)
            tokens = roi.flatten(2).transpose(1, 2)  # (N_prop, roi*roi, C)
            tokens = self._match_channel_dim(tokens, self.level_in_dims[level_idx])
            tokens = self.level_token_proj[level_idx](tokens)  # (N_prop, roi*roi, d)
            token_levels.append(tokens)

            pooled = roi.mean(dim=(-1, -2))  # (N_prop, C)
            pooled = self._match_channel_dim(pooled, self.level_in_dims[level_idx])
            pooled = self.level_pool_proj[level_idx](pooled)  # (N_prop, d)
            pooled_levels.append(pooled)

        proposal_tokens = torch.cat(token_levels, dim=1)  # (N_prop, T, d)
        proposal_global = torch.stack(pooled_levels, dim=0).mean(dim=0)  # (N_prop, d)

        q = self.query_proj(seg_embeddings)  # (N_seg, d)
        q = q.unsqueeze(1).unsqueeze(2)  # (N_seg, 1, 1, d)
        k = proposal_tokens.unsqueeze(0)  # (1, N_prop, T, d)
        v = k

        attn_logits = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)  # (N_seg, N_prop, T)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        z = (attn.unsqueeze(-1) * v).sum(dim=2)  # (N_seg, N_prop, d)

        seg_context = self.seg_proj(seg_embeddings).unsqueeze(1).expand(-1, num_prop, -1)  # (N_seg, N_prop, d)
        proposal_context = proposal_global.unsqueeze(0).expand(num_seg, -1, -1)  # (N_seg, N_prop, d)
        fusion = torch.cat([z, proposal_context, seg_context], dim=-1)  # (N_seg, N_prop, 3d)
        logits = self.score_head(fusion).squeeze(-1)  # (N_seg, N_prop)
        return logits

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ignore_index=None,
    scale=1000,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        ignore_index: A value in the target tensor to ignore during the loss calculation.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)

    # Create mask to ignore specified index
    if ignore_index is not None:
        mask = targets != ignore_index
        inputs = inputs * mask
        targets = targets * mask

    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ignore_index=None,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        ignore_index: A value in the target tensor to ignore during the loss calculation.
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Flatten and mask the loss values where targets match the ignore index
    loss = loss.flatten(1, 2)
    if ignore_index is not None:
        mask = targets.flatten(1, 2) != ignore_index
        loss = loss * mask

    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


class VideoGLaMMMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VideoGLaMMMetaModel, self).__init__(config)
        
        self.vision_pretrained = kwargs.get("sam_pretrained_path", None)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
        if not hasattr(self.config, "mask_decoder_itm"):
            self.config.mask_decoder_itm = kwargs["mask_decoder_itm"]
        if not hasattr(self.config, "use_sam2"):
            self.config.use_sam2 = kwargs.get("use_sam2")
            
        use_sam2_video_branch = kwargs.get("use_sam2_video_branch", False)
            
        self.initialize_lisa_modules(self.config, use_sam2_video_branch=use_sam2_video_branch)

    def initialize_lisa_modules(self, config, use_sam2_video_branch=False):
        # SAM
        if config.use_sam2: # Use SAM2
            if not use_sam2_video_branch:
                print('\033[92m---Initialize SAM2 without video branch--\033[0m')
                self.visual_model = build_sam2("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
            else:
                print('\033[92m---Initialize SAM2 with video branch--\033[0m')
                self.visual_model = build_sam2_video_predictor("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
        elif config.mask_decoder_itm: # Use SAM_with_ITM
            self.visual_model = build_sam_vit_h(self.vision_pretrained, with_itm=True)
        else: # Use original SAM
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
            
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            if config.use_sam2: # if using SAM2
                self.visual_model.sam_mask_decoder.train()
                for param in self.visual_model.sam_mask_decoder.parameters():
                    param.requires_grad = True
            else: # if using SAM
                self.visual_model.mask_decoder.train()
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        if masks.device.type == "cpu" and masks.dtype != torch.float32:
            masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
    
class VideoGLaMMModel(VideoGLaMMMetaModel, VideoGPTPlusPhi3Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VideoGLaMMModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        # self.config.vision_tower = self.config.mm_vision_tower 
        self.config.vision_tower = getattr(self.config, "mm_vision_tower", None)
        # self.config.image_vision_tower = self.config.image_mm_vision_tower 
        self.config.image_vision_tower = getattr(self.config, "image_mm_vision_tower", None)
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.pretrain_mm_mlp_adapter = None
        self.config.pretrain_image_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

        # Detection-guided segmentation parameters
        self.use_detection_guidance: bool = kwargs.get("use_detection_guidance", False)
        self.proposal_loss_weight: float = kwargs.get("proposal_loss_weight", 1.0)
        self.proposal_iou_threshold: float = kwargs.get("proposal_iou_threshold", 0.3)
        self.proposal_topk: int = kwargs.get("proposal_topk", 100)
        self.proposal_score_power: Tuple[float, float] = kwargs.get("proposal_score_power", (0.3, 0.7))
        self.proposal_score_threshold: float = kwargs.get("proposal_score_threshold", 0.4)
        self.max_selected_boxes_per_token: int = int(kwargs.get("max_selected_boxes_per_token", 16))
        self.use_proposal_box_regression: bool = bool(kwargs.get("use_proposal_box_regression", False))
        # Keep default very small so enabling has minimal effect unless intentionally increased.
        self.proposal_box_loss_weight: float = float(kwargs.get("proposal_box_loss_weight", 0.01))
        self.use_box_prior_filter: bool = bool(kwargs.get("use_box_prior_filter", False))
        self.box_prior_min_score: float = float(kwargs.get("box_prior_min_score", 0.35))
        self.box_prior_max_keep: int = int(kwargs.get("box_prior_max_keep", 4))

        self.ddetr_helper: Optional[DDETRProposalGenerator] = None
        self.proposal_selector: Optional[ProposalSelector] = None
        self.roi_helper: Optional[ROIProposalHelper] = None
        self.ddetr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_roi_helper: bool = kwargs.get("use_roi_helper", True)
        self.roi_helper_hidden_dim: int = kwargs.get("roi_helper_hidden_dim", 256)
        self.roi_align_output_size: int = kwargs.get("roi_align_output_size", 7)
        self.roi_fusion_gamma = nn.Parameter(
            torch.tensor(float(kwargs.get("roi_fusion_init", 0.1)), dtype=torch.float32)
        )

        if self.use_detection_guidance:
            ddetr_model_path = kwargs.get("ddetr_model_path") or "checkpoints_hf/ddetr_sam2"

            self.ddetr_helper = DDETRProposalGenerator(
                model_path=ddetr_model_path,
                device=self.ddetr_device,
            )
            seg_dim = self.config.out_dim
            prop_dim = self.ddetr_helper.hidden_dim
            selector_hidden = kwargs.get("proposal_selector_hidden_dim", min(seg_dim, prop_dim))
            self.proposal_selector = ProposalSelector(seg_dim=seg_dim, prop_dim=prop_dim, hidden_dim=selector_hidden)
            if self.config.use_sam2 and self.use_roi_helper:
                self.roi_helper = ROIProposalHelper(
                    seg_dim=seg_dim,
                    hidden_dim=self.roi_helper_hidden_dim,
                    roi_size=self.roi_align_output_size,
                )

        # buffers for SAM2 mean/std to recover RGB frames
        self.register_buffer("sam_pixel_mean", SAM2_PIXEL_MEAN.clone(), persistent=False)
        self.register_buffer("sam_pixel_std", SAM2_PIXEL_STD.clone(), persistent=False)

    def _sam_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a SAM-preprocessed tensor (3, H, W) back to a PIL image.
        """
        tensor = tensor.detach().cpu()
        pixel = tensor * self.sam_pixel_std.cpu() + self.sam_pixel_mean.cpu()
        pixel = pixel.round().clamp(0, 255).to(torch.uint8)
        array = pixel.permute(1, 2, 0).contiguous().cpu().numpy()
        return Image.fromarray(array)

    def _generate_ddetr_proposals(self, images_for_sam_sample: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if self.ddetr_helper is None:
            return None

        # handle input shape: [T, 3, 1024, 1024] or [3, 1024, 1024]
        if images_for_sam_sample.dim() == 4:
            frame_tensor = images_for_sam_sample[0]
        else:
            frame_tensor = images_for_sam_sample

        pil_image = self._sam_tensor_to_pil(frame_tensor)
        pil_image = pil_image.resize((448, 448))
        proposals = self.ddetr_helper.generate_proposals(
            pil_image=pil_image,
            topk=self.proposal_topk,
            score_power=self.proposal_score_power,
        )

        return {
            "boxes": proposals["boxes"].to(self.sam_pixel_mean.device),
            "features": proposals["features"].to(self.sam_pixel_mean.device),
            "scores": proposals["scores"].to(self.sam_pixel_mean.device),
            "mask": proposals["mask"].to(self.sam_pixel_mean.device),
        }

    def _extract_sam2_feature_pyramid(
        self,
        images_for_sam_sample: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract first-frame SAM2/Hiera feature pyramid for ROIAlign.
        Returns feature maps as a list of (1, C, H, W).
        """
        if not self.config.use_sam2:
            return []

        if images_for_sam_sample.dim() == 4:
            frame = images_for_sam_sample[0].unsqueeze(0)
        else:
            frame = images_for_sam_sample.unsqueeze(0)
        frame = frame.to(self.sam_pixel_mean.device)

        with torch.no_grad():
            backbone_out = self.visual_model.forward_image(frame)
            _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)
            image_embeddings = [feat.to(frame.dtype) for feat in image_embeddings]

            bs = frame.shape[0]
            if self.visual_model.directly_add_no_mem_embed:
                image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

            bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
            feats = [
                feat.permute(1, 2, 0).view(bs, -1, *feat_size)
                for feat, feat_size in zip(image_embeddings[::-1], bb_feat_sizes[::-1])
            ][::-1]

        return feats

    def _assign_proposal_targets(
        self,
        proposal_boxes: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a multi-hot target matrix of shape (num_seg_tokens, num_proposals),
        where entries are 1 if the proposal overlaps the ground-truth mask above
        the IoU threshold.
        """
        device = proposal_boxes.device
        num_seg_tokens = gt_masks.shape[0] if gt_masks is not None else 0
        num_props = proposal_boxes.shape[0]
        targets = torch.zeros((num_seg_tokens, num_props), device=device, dtype=torch.float32)

        if num_seg_tokens == 0 or num_props == 0:
            return targets

        for idx in range(num_seg_tokens):
            mask = gt_masks[idx]
            target_box = _mask_to_xyxy(mask)
            if target_box is None:
                continue
            ious = box_iou(proposal_boxes, target_box.unsqueeze(0)).squeeze(1)
            targets[idx, ious >= self.proposal_iou_threshold] = 1.0

        return targets

    def _refine_seg_embeddings_with_proposals(
        self,
        seg_embeddings: torch.Tensor,
        proposals: Dict[str, torch.Tensor],
        training: bool,
        gt_masks: Optional[torch.Tensor] = None,
        images_for_sam_sample: Optional[torch.Tensor] = None,
        sam2_feature_pyramid: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Returns refined segment embeddings, optional classification loss,
        optional box regression loss, and selected boxes per token.
        """
        prop_boxes = proposals["boxes"]
        prop_features = proposals["features"]
        prop_scores = proposals["scores"]
        prop_mask = proposals.get("mask")

        num_props = prop_boxes.shape[0]
        num_tokens = seg_embeddings.shape[0]

        if num_tokens == 0:
            return seg_embeddings, None, None, []

        if num_props == 0:
            empty_list = [prop_boxes.new_empty((0, 4)) for _ in range(num_tokens)]
            return seg_embeddings, None, None, empty_list

        refined, logits_old, _, pred_boxes_norm = self.proposal_selector(
            seg_embeddings,
            prop_features,
            proposal_scores=torch.log(prop_scores.clamp(min=1e-6)) if prop_scores is not None else None,
            proposal_mask=prop_mask,
        )
        logits = logits_old

        if self.roi_helper is not None and self.config.use_sam2:
            if sam2_feature_pyramid is None and images_for_sam_sample is not None:
                sam2_feature_pyramid = self._extract_sam2_feature_pyramid(images_for_sam_sample)
            if sam2_feature_pyramid is not None and len(sam2_feature_pyramid) > 0:
                if images_for_sam_sample is not None:
                    if images_for_sam_sample.dim() == 4:
                        image_hw = (
                            int(images_for_sam_sample.shape[-2]),
                            int(images_for_sam_sample.shape[-1]),
                        )
                    else:
                        image_hw = (
                            int(images_for_sam_sample.shape[-2]),
                            int(images_for_sam_sample.shape[-1]),
                        )
                else:
                    image_hw = (
                        int(sam2_feature_pyramid[0].shape[-2]),
                        int(sam2_feature_pyramid[0].shape[-1]),
                    )
                roi_logits = self.roi_helper(
                    seg_embeddings=seg_embeddings,
                    proposal_boxes=prop_boxes,
                    feature_pyramid=sam2_feature_pyramid,
                    image_hw=image_hw,
                )
                gamma = torch.clamp(
                    self.roi_fusion_gamma.to(dtype=logits_old.dtype, device=logits_old.device),
                    min=0.0,
                    max=1.0,
                )
                logits = logits_old + gamma * roi_logits

        probabilities = torch.sigmoid(logits) if logits.numel() > 0 else logits
        if prop_mask is not None:
            valid_mask = prop_mask
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            if valid_mask.shape[0] != num_tokens:
                valid_mask = valid_mask[:1].expand(num_tokens, -1)
            valid_mask = valid_mask.to(dtype=torch.bool)
            probabilities = probabilities * valid_mask.float()
        else:
            valid_mask = torch.ones_like(probabilities, dtype=torch.bool)

        selected_boxes_per_token: List[torch.Tensor] = []
        for idx in range(num_tokens):
            mask_row = probabilities[idx] >= self.proposal_score_threshold
            mask_row = mask_row & valid_mask[idx]
            if mask_row.sum() == 0:
                if valid_mask[idx].any():
                    candidate_scores = probabilities[idx].clone()
                    candidate_scores = candidate_scores.masked_fill(~valid_mask[idx], -1.0)
                    top_index = candidate_scores.argmax()
                    mask_row[top_index] = True
                else:
                    mask_row = torch.zeros_like(mask_row)
            indices = mask_row.nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0 and num_props > 0:
                fallback = valid_mask[idx].nonzero(as_tuple=False).squeeze(1)
                if fallback.numel() == 0:
                    fallback = torch.arange(num_props, device=prop_boxes.device)[:1]
                indices = fallback[:1]
            if self.max_selected_boxes_per_token > 0 and indices.numel() > self.max_selected_boxes_per_token:
                row_scores = probabilities[idx, indices]
                top_idx = torch.topk(row_scores, k=self.max_selected_boxes_per_token, largest=True).indices
                indices = indices[top_idx]
            selected_boxes_per_token.append(prop_boxes[indices])

        proposal_loss: Optional[torch.Tensor] = None
        proposal_box_loss: Optional[torch.Tensor] = None
        if training and gt_masks is not None and num_props > 0:
            max_tokens = min(num_tokens, gt_masks.shape[0])
            if max_tokens > 0:
                gt_masks_slice = gt_masks[:max_tokens]
                targets = self._assign_proposal_targets(prop_boxes, gt_masks_slice)
                targets = targets[:max_tokens]
                logits_used = logits[:max_tokens]
                if prop_mask is not None:
                    mask_weights = prop_mask
                    if mask_weights.dim() == 1:
                        mask_weights = mask_weights.unsqueeze(0)
                    if mask_weights.shape[0] != logits_used.shape[0]:
                        mask_weights = mask_weights[:1].expand(logits_used.shape[0], -1)
                    valid_weights = mask_weights.float()
                else:
                    valid_weights = torch.ones_like(targets)
                bce = F.binary_cross_entropy_with_logits(logits_used, targets, reduction="none")
                weighted = bce * valid_weights
                denom = valid_weights.sum()
                proposal_loss = weighted.sum() / denom if denom > 0 else logits_used.new_tensor(0.0)

                if self.use_proposal_box_regression:
                    pred_boxes_used = pred_boxes_norm[:max_tokens]  # (N_seg, N_prop, 4)
                    box_l1_losses: List[torch.Tensor] = []
                    box_giou_losses: List[torch.Tensor] = []

                    for token_idx in range(max_tokens):
                        target_box = _mask_to_xyxy(gt_masks_slice[token_idx])
                        if target_box is None:
                            continue

                        pos_mask = targets[token_idx] > 0.5
                        if valid_weights is not None:
                            pos_mask = pos_mask & (valid_weights[token_idx] > 0)
                        if pos_mask.sum() == 0:
                            continue

                        pred_pos = pred_boxes_used[token_idx][pos_mask]
                        tgt_pos = target_box.unsqueeze(0).expand_as(pred_pos)

                        box_l1_losses.append(torch.abs(pred_pos - tgt_pos).mean())
                        giou_mat = generalized_box_iou(pred_pos, tgt_pos)
                        giou_diag = torch.diag(giou_mat)
                        box_giou_losses.append((1.0 - giou_diag).mean())

                    if len(box_l1_losses) > 0:
                        proposal_box_loss = torch.stack(box_l1_losses).mean() + torch.stack(box_giou_losses).mean()
                    else:
                        proposal_box_loss = logits_used.new_tensor(0.0)
            else:
                proposal_loss = logits.new_tensor(0.0)
                if self.use_proposal_box_regression:
                    proposal_box_loss = logits.new_tensor(0.0)

        return refined, proposal_loss, proposal_box_loss, selected_boxes_per_token

        
class VideoGLaMM_SAM2():
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        '''
            pixel_values : batch x [T, 3, 1024, 1024]
        '''
    
        # 
        images = pixel_values
        with torch.no_grad():
            image_embeddings_list = []
            batch_size = len(images)
            
            for i in range(batch_size): # for batch
                if images[i].shape[0]==1: # image
                    torch.cuda.empty_cache()
                    image_embeddings = self.model.visual_model.image_encoder(images[i])
                    image_embeddings_list.append([image_embeddings])
                else: # video
                    ###
                    t = images[i].shape[0]
                    image_embeddings_i = []
                    for ti in range(t):
                        torch.cuda.empty_cache()
                        image_embeddings = self.model.visual_model.image_encoder(images[i][ti].unsqueeze(0))
                        image_embeddings_i.append(image_embeddings)
                    image_embeddings_list.append(image_embeddings_i)
            torch.cuda.empty_cache()
        return image_embeddings_list # B x T x [1, 256, 64, 64]
    
    def get_visual_embs_sam2(self, images_for_sam_all: torch.FloatTensor):
        '''
            images_for_sam_all : batch x [T, 3, 1024, 1024] : (list of tensors)
        '''
        
        batch_size = len(images_for_sam_all)
        _features_all = []
        
        with torch.no_grad():
            for batch_idx in range(len(images_for_sam_all)):
                features_in_batch = []
                for t in range(len(images_for_sam_all[batch_idx])):
                    images_for_sam = images_for_sam_all[batch_idx][t].unsqueeze(0) # [1, 3, 1024, 1024]
                    
                    backbone_out = self.model.visual_model.forward_image(images_for_sam) 
                    _, image_embeddings, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
                    image_embeddings = [_.to(images_for_sam.dtype) for _ in image_embeddings]
                    
                    bs = images_for_sam.shape[0]
                    
                    if self.model.visual_model.directly_add_no_mem_embed:
                        image_embeddings[-1] = image_embeddings[-1] + self.model.visual_model.no_mem_embed

                    _bb_feat_sizes = [(256, 256),(128, 128),(64, 64),]
                    feats = [
                        feat.permute(1, 2, 0).view(bs, -1, *feat_size)
                        for feat, feat_size in zip(image_embeddings[::-1], _bb_feat_sizes[::-1])
                    ][::-1]
                    _features = {
                        "image_embed": feats[-1], # [bs, 256, 64, 64] # bs=1 in this case
                        "high_res_feats": feats[:-1]} #   # 2 x [ bs, 32, 256, 256] # bs=1 in this case
                    
                    features_in_batch.append(_features)
                _features_all.append(features_in_batch)
            torch.cuda.empty_cache()
        # 
        return batch_size, _features_all
    
    # def forward(self, **kwargs):
    #     if "past_key_values" in kwargs:
    #         return super().forward(**kwargs)
    #     return self.model_forward(**kwargs)

    def __inference_path(self, input_ids, images, context_images, attention_masks, tsf_images=None):

        # length = input_ids.shape[0]

        assert len(images) == 1 # batch size is 1
        assert input_ids.shape[0] == 1 # batch size is 1
        
        # images_clip_extend = images * length # 1x[1, 3, 224, 224] -> lengthx[1, 3, 224, 224] 
        images_clip_extend = images # 1x[1, 3, 224, 224]

        # print_dimensions('context_images', context_images) 
        
        # if context_images is not None:
        #     context_images_clip_extend = context_images * length # 1x[1, 3, 224, 224] -> lengthx[1, 3, 224, 224]
        # else:
        #     context_images_clip_extend = None
        
        context_images_clip_extend = context_images

        output_hidden_states = []
        # for i in range(n_batch):
        start_i, end_i   =  0, 1 #i * length   ,  min((i + 1) * length, input_ids.shape[0]) # 0, 1
        output_i = self.super_forward(
            # images=images_clip_extend[: 1], 
            images=images_clip_extend,
            # context_images=context_images_clip_extend[: 1] if context_images_clip_extend is not None else [None]*length,
            context_images=context_images_clip_extend if context_images_clip_extend is not None else [None],
            tsf_images=tsf_images,
            attention_mask=attention_masks[0:1] if attention_masks is not None else None,
            input_ids=input_ids[0:1],
            output_hidden_states=True,
        )
        output_hidden_states.append(output_i.hidden_states)
        seg_token_mask = output_i.seg_token_mask
        box_token_mask = getattr(output_i, "box_token_mask", None)
        if box_token_mask is None:
            box_token_mask = torch.zeros_like(seg_token_mask, dtype=torch.bool)
        torch.cuda.empty_cache()
        
        output_hidden_states = [torch.cat(output_hidden_states, dim=0)]
        
        output = None
        
        return output, output_hidden_states, seg_token_mask, box_token_mask
        
    def __training_path(self, images, context_images, tsf_images, input_ids, labels, attention_masks, offset):
        
        # print('In __training_path')
        # print_dimensions('images', images) # [B, T, 3, 224, 224]
        # print_dimensions('context_images', context_images)
        # print('context_images:', context_images)
        
        # prepare images
        images_clip_list = []
        for i in range(len(offset) - 1): # batch_size = len(offset) - 1
            start_i, end_i = offset[i], offset[i + 1]
            for j in range(end_i - start_i):
                images_clip_list.append(images[i])
        images = images_clip_list # len(all_conversations) x [T, 3, 224, 224] 
        
        # prepare context images
        context_images_clip_list = []
        for i in range(len(offset) - 1): # batch_size = len(offset) - 1
            start_i, end_i = offset[i], offset[i + 1]
            for j in range(end_i - start_i):
                context_images_clip_list.append(context_images[i])
        context_images = context_images_clip_list # len(all_conversations) x [T, 3, 224, 224]

        tsf_images_clip_list = []
        if tsf_images is None:
            tsf_images_clip_list = [None] * len(images)
        else:
            for i in range(len(offset) - 1):  # batch_size = len(offset) - 1
                start_i, end_i = offset[i], offset[i + 1]
                for _ in range(end_i - start_i):
                    tsf_images_clip_list.append(tsf_images[i] if i < len(tsf_images) else None)
            if len(tsf_images_clip_list) < len(images):
                tsf_images_clip_list.extend([None] * (len(images) - len(tsf_images_clip_list)))

        output = self.super_forward(
            images=images,
            context_images=context_images,
            tsf_images=tsf_images_clip_list,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        seg_token_mask = output.seg_token_mask
        box_token_mask = getattr(output, "box_token_mask", None)
        if box_token_mask is None:
            box_token_mask = torch.zeros_like(seg_token_mask, dtype=torch.bool)
        
        return output, output_hidden_states, seg_token_mask, box_token_mask

    @staticmethod
    def _split_embeddings_by_offset(
        last_hidden_state: torch.Tensor,
        token_mask: torch.Tensor,
        offset: torch.Tensor,
    ) -> List[torch.Tensor]:
        token_embeddings = last_hidden_state[token_mask]
        token_counts = token_mask.int().sum(-1)
        token_offset = token_counts.cumsum(-1)
        token_offset = torch.cat(
            [torch.zeros(1, device=token_offset.device, dtype=torch.long), token_offset],
            dim=0,
        )
        token_offset = token_offset[offset.to(token_offset.device)]
        split_embeddings: List[torch.Tensor] = []
        for i in range(len(token_offset) - 1):
            start_i, end_i = token_offset[i], token_offset[i + 1]
            split_embeddings.append(token_embeddings[start_i:end_i])
        return split_embeddings

    @staticmethod
    def _build_generated_token_mask(
        output_ids: torch.Tensor,
        token_idx: Optional[int],
        num_newly_added_tokens: int,
    ) -> torch.Tensor:
        mask = output_ids[:, 1:] == token_idx if token_idx is not None else torch.zeros_like(output_ids[:, 1:], dtype=torch.bool)
        pad_prefix = torch.zeros(
            (mask.shape[0], num_newly_added_tokens),
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([pad_prefix, mask], dim=1)

    @staticmethod
    def _align_selected_boxes_to_seg_count(
        selected_boxes: List[torch.Tensor],
        seg_count: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if seg_count == len(selected_boxes):
            return selected_boxes
        if seg_count == 0:
            return []
        if len(selected_boxes) == 0:
            return [torch.empty((0, 4), device=device) for _ in range(seg_count)]
        aligned: List[torch.Tensor] = []
        for idx in range(seg_count):
            src_idx = min(idx, len(selected_boxes) - 1)
            aligned.append(selected_boxes[src_idx])
        return aligned

    @staticmethod
    def _mask_bbox_xyxy(mask: torch.Tensor) -> Optional[torch.Tensor]:
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None
        y1 = idx[:, 0].min().float()
        x1 = idx[:, 1].min().float()
        y2 = idx[:, 0].max().float() + 1.0
        x2 = idx[:, 1].max().float() + 1.0
        return torch.tensor([x1, y1, x2, y2], device=mask.device, dtype=torch.float32)

    @staticmethod
    def _box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        inter_x1 = torch.maximum(box_a[0], box_b[0])
        inter_y1 = torch.maximum(box_a[1], box_b[1])
        inter_x2 = torch.minimum(box_a[2], box_b[2])
        inter_y2 = torch.minimum(box_a[3], box_b[3])
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter = inter_w * inter_h
        area_a = torch.clamp((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), min=1e-6)
        area_b = torch.clamp((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]), min=1e-6)
        union = area_a + area_b - inter
        return inter / torch.clamp(union, min=1e-6)

    def _score_boxes_with_prior_masks(
        self,
        boxes_norm: torch.Tensor,
        prior_masks,
    ) -> torch.Tensor:
        scores = torch.zeros((boxes_norm.shape[0],), dtype=torch.float32, device=boxes_norm.device)
        if boxes_norm.numel() == 0 or prior_masks is None:
            return scores
        if not torch.is_tensor(prior_masks):
            prior_masks = torch.as_tensor(prior_masks)
        prior_masks = prior_masks.to(device=boxes_norm.device)
        if prior_masks.dim() == 4:
            masks = prior_masks[:, 0]
        elif prior_masks.dim() == 3:
            masks = prior_masks
        elif prior_masks.dim() == 2:
            masks = prior_masks.unsqueeze(0)
        else:
            return scores

        num_masks = min(int(masks.shape[0]), int(boxes_norm.shape[0]))
        if num_masks == 0:
            return scores

        mask_h = float(masks.shape[-2])
        mask_w = float(masks.shape[-1])
        for idx in range(num_masks):
            mask = masks[idx] > 0.5
            if not mask.any():
                continue
            mask_bbox = self._mask_bbox_xyxy(mask)
            if mask_bbox is None:
                continue

            box = boxes_norm[idx].clamp(0.0, 1.0)
            box_px = torch.tensor(
                [
                    box[0] * mask_w,
                    box[1] * mask_h,
                    box[2] * mask_w,
                    box[3] * mask_h,
                ],
                device=boxes_norm.device,
                dtype=torch.float32,
            )
            iou = self._box_iou_xyxy(box_px, mask_bbox)

            mask_area_norm = (mask.float().mean()).clamp(min=1e-6)
            box_area_norm = ((box[2] - box[0]).clamp(min=1e-6) * (box[3] - box[1]).clamp(min=1e-6)).clamp(min=1e-6)
            area_ratio = mask_area_norm / box_area_norm
            area_consistency = 1.0 / (1.0 + torch.abs(torch.log(area_ratio)))
            score = 0.8 * iou + 0.2 * area_consistency

            # Down-weight obvious failures (near-empty or near-full masks).
            if mask_area_norm < 1e-4 or mask_area_norm > 0.9:
                score = score * 0.2
            scores[idx] = score

        return scores

    def _filter_selected_boxes_with_prior(
        self,
        selected_boxes: List[torch.Tensor],
        images_for_sam_sample: torch.Tensor,
        orig_hw: Tuple[int, int],
    ) -> List[torch.Tensor]:
        if not getattr(self.model, "use_box_prior_filter", False):
            return selected_boxes
        if selected_boxes is None or len(selected_boxes) == 0:
            return selected_boxes

        min_score = float(getattr(self.model, "box_prior_min_score", 0.35))
        max_keep = int(getattr(self.model, "box_prior_max_keep", 4))
        out: List[torch.Tensor] = []
        h, w = int(orig_hw[0]), int(orig_hw[1])

        for token_boxes in selected_boxes:
            if token_boxes is None or token_boxes.numel() == 0:
                out.append(token_boxes)
                continue

            boxes_norm = token_boxes.detach().float().clamp(0.0, 1.0)
            boxes_px = boxes_norm.clone()
            boxes_px[:, [0, 2]] = boxes_px[:, [0, 2]] * float(w)
            boxes_px[:, [1, 3]] = boxes_px[:, [1, 3]] * float(h)

            try:
                prior_masks = self.estimate_mask_from_box_image(
                    bbox=boxes_px.detach().cpu().numpy(),
                    orig_size=(h, w),
                    images_for_sam=images_for_sam_sample,
                )
                scores = self._score_boxes_with_prior_masks(boxes_norm, prior_masks)
            except Exception:
                out.append(token_boxes)
                continue

            if scores.numel() == 0:
                out.append(token_boxes[:1])
                continue

            keep_idx = (scores >= min_score).nonzero(as_tuple=False).squeeze(1)
            if keep_idx.numel() == 0:
                keep_idx = torch.topk(scores, k=1, largest=True).indices
            if max_keep > 0 and keep_idx.numel() > max_keep:
                keep_scores = scores[keep_idx]
                top_local = torch.topk(keep_scores, k=max_keep, largest=True).indices
                keep_idx = keep_idx[top_local]
            out.append(token_boxes[keep_idx.to(token_boxes.device)])

        return out
    
    def model_forward(
        self,
        images_for_sam: List[torch.FloatTensor],      # preprocessed image for SAM   # batchx[T_sam, 3, 1024, 1024]
        images: List[torch.FloatTensor],              # preprocessed image           # batchx[T, 3, 224, 224]
        context_images: List[torch.FloatTensor],      # preprocessed context image   # batchx[T, 3, 224, 224]
        input_ids: torch.LongTensor,                  # [num_conversations, length_of_sequence]
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],          # ground truth masks
        label_list: List[torch.Tensor], #  a pseudo label of which the shape indicates the original frame dimensions
        resize_list: List[tuple],
        tsf_images: Optional[List[Optional[torch.FloatTensor]]] = None,  # batchx[(N_tsf, 3, H, W) or None]
        inference: bool = False,
        **kwargs,
    ):
        
        if self.config.use_sam2:
            USE_SAM2 = True
            USE_SAM1 = False
        else:
            USE_SAM2 = False
            USE_SAM1 = True
            
        ### 1 - Extract grounding encoder image embeddings
        if USE_SAM1:
            image_embeddings_for_sam = self.get_visual_embs(images_for_sam) # get SAM embeddings
            batch_size = len(image_embeddings_for_sam)
        elif USE_SAM2:
            batch_size, _features_all = self.get_visual_embs_sam2(images_for_sam) # get SAM2 embeddings

                
        assert batch_size == len(offset) - 1        

        original_size_list = kwargs.get("original_size_list")
        if original_size_list is None:
            original_size_list = []
            for label in label_list:
                if isinstance(label, torch.Tensor) and label.ndim >= 2:
                    size = (int(label.shape[-2]), int(label.shape[-1]))
                elif hasattr(label, "shape") and len(label.shape) >= 2:
                    size = (int(label.shape[-2]), int(label.shape[-1]))
                elif isinstance(label, (tuple, list)) and len(label) >= 2:
                    size = (int(label[-2]), int(label[-1]))
                else:
                    raise ValueError("original_size_list not provided and cannot be inferred from label")
                original_size_list.append(size)

        ### 3 - Handle inference or training path
        if inference:
            output, output_hidden_states, seg_token_mask, box_token_mask = self.__inference_path(
                input_ids, images, context_images, attention_masks, tsf_images=tsf_images
            )
        else:
            output, output_hidden_states, seg_token_mask, box_token_mask = self.__training_path(
                images, context_images, tsf_images, input_ids, labels, attention_masks, offset
            )
            
        # output_hidden_states: [num_hidden_layers, num_conversations, length_of_sequence, 4096]
        # seg_token_mask : [num_conversations, length_of_sequence]
                
        ### 4 - Process hidden states
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        
        # Pass the output_hidden_states[-1] through the fully connected layer (text_hidden_fcs)
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))  # ([num_conversations, length_of_sequence, 4096]) -> ([batch_size, length_of_sequence, 256])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) # ([num_conversations, length_of_sequence, 256])
                
        max_num_seg_tokens_per_sample = MAX_NUM_SEG_TOKENS_PER_SAMPLE

        # SEG embeddings drive mask generation.
        pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, seg_token_mask, offset)
        # BOX embeddings drive proposal selection (fallback to SEG when no BOX emitted).
        box_pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, box_token_mask, offset)

        # Keep per-sample prompt count bounded to avoid SAM decoder memory spikes.
        pred_embeddings = [emb[:max_num_seg_tokens_per_sample] for emb in pred_embeddings]
        box_pred_embeddings = [emb[:max_num_seg_tokens_per_sample] for emb in box_pred_embeddings]
        
        selected_boxes_batch: List[List[torch.Tensor]] = []
        proposal_losses: List[torch.Tensor] = []
        proposal_box_losses: List[torch.Tensor] = []
        if self.model.use_detection_guidance and self.model.proposal_selector is not None:
            for batch_idx in range(len(pred_embeddings)):
                seg_embs = pred_embeddings[batch_idx]
                box_embs = box_pred_embeddings[batch_idx] if batch_idx < len(box_pred_embeddings) else seg_embs.new_empty((0, seg_embs.shape[-1]))
                selector_embs = box_embs if box_embs.numel() > 0 else seg_embs
                if selector_embs.numel() == 0:
                    selected_boxes_batch.append([])
                    continue
                proposals = self.model._generate_ddetr_proposals(images_for_sam[batch_idx])
                if proposals is None or proposals["boxes"].numel() == 0:
                    selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                    continue
                gt_masks = masks_list[batch_idx] if (not inference and masks_list is not None) else None
                sam2_feature_pyramid = None
                if USE_SAM2:
                    first_frame_features = _features_all[batch_idx][0]
                    sam2_feature_pyramid = list(first_frame_features["high_res_feats"]) + [first_frame_features["image_embed"]]
                refined_sel, proposal_loss, proposal_box_loss, selected_boxes = self.model._refine_seg_embeddings_with_proposals(
                    selector_embs,
                    proposals,
                    training=not inference,
                    gt_masks=gt_masks,
                    images_for_sam_sample=images_for_sam[batch_idx],
                    sam2_feature_pyramid=sam2_feature_pyramid,
                )
                if box_embs.numel() > 0:
                    box_pred_embeddings[batch_idx] = refined_sel
                else:
                    pred_embeddings[batch_idx] = refined_sel
                selected_boxes = self._align_selected_boxes_to_seg_count(
                    selected_boxes,
                    seg_embs.shape[0],
                    seg_embs.device,
                )
                if self.model.use_box_prior_filter:
                    selected_boxes = self._filter_selected_boxes_with_prior(
                        selected_boxes=selected_boxes,
                        images_for_sam_sample=images_for_sam[batch_idx],
                        orig_hw=original_size_list[batch_idx],
                    )
                selected_boxes_batch.append(selected_boxes)
                if proposal_loss is not None:
                    proposal_losses.append(proposal_loss)
                if proposal_box_loss is not None:
                    proposal_box_losses.append(proposal_box_loss)
        else:
            for seg_embs in pred_embeddings:
                selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                        
        ### 5 - Generate and post-process masks
        pred_masks = []
        use_guidance = self.model.use_detection_guidance and self.model.proposal_selector is not None
        for batch_idx, seg_embs in enumerate(pred_embeddings):
            if USE_SAM2:
                prompt_dtype = self.model.visual_model.sam_prompt_encoder.point_embeddings[0].weight.dtype
            else:
                prompt_dtype = self.model.visual_model.prompt_encoder.point_embeddings[0].weight.dtype

            seg_embs_prompt = seg_embs.to(prompt_dtype)

            num_seg_tokens_per_sample = seg_embs.shape[0]
            if USE_SAM1:
                image_embeddings_for_sam_i = image_embeddings_for_sam[batch_idx]
                t = len(image_embeddings_for_sam_i)
            elif USE_SAM2:
                t = len(_features_all[batch_idx])
            else:
                t = 0

            if num_seg_tokens_per_sample == 0:
                h, w = original_size_list[batch_idx]
                zero_plane = torch.zeros((max_num_seg_tokens_per_sample, h, w), dtype=torch.float32, device=seg_embs.device)
                pred_masks.append([zero_plane.clone() for _ in range(t)])
                continue

            use_guidance_batch = (
                use_guidance
                and USE_SAM2
                and batch_idx < len(selected_boxes_batch)
                and len(selected_boxes_batch[batch_idx]) == num_seg_tokens_per_sample
                and all(boxes.shape[0] > 0 for boxes in selected_boxes_batch[batch_idx])
            )

            if not use_guidance_batch:
                if USE_SAM1:
                    sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embs_prompt.unsqueeze(1),
                    )
                elif USE_SAM2:
                    sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embs_prompt.unsqueeze(1),
                    )
                else:
                    raise RuntimeError("Unsupported SAM configuration")

                sparse_embeddings = sparse_embeddings.to(prompt_dtype)
                dense_embeddings = dense_embeddings.to(prompt_dtype)
                if num_seg_tokens_per_sample < max_num_seg_tokens_per_sample:
                    pad_shape = (max_num_seg_tokens_per_sample - num_seg_tokens_per_sample,) + sparse_embeddings.shape[1:]
                    pad_sparse = torch.zeros(pad_shape, dtype=sparse_embeddings.dtype, device=sparse_embeddings.device)
                    sparse_embeddings = torch.cat([sparse_embeddings, pad_sparse], dim=0)
                    pad_dense = torch.zeros(
                        (max_num_seg_tokens_per_sample - num_seg_tokens_per_sample,) + dense_embeddings.shape[1:],
                        dtype=dense_embeddings.dtype,
                        device=dense_embeddings.device,
                    )
                    dense_embeddings = torch.cat([dense_embeddings, pad_dense], dim=0)

                pred_masks_ti = []
                track_token = None
                for ti in range(t):
                    if USE_SAM1:
                        low_res_masks, iou_predictions, track_token = self.model.visual_model.mask_decoder(
                            image_embeddings=image_embeddings_for_sam_i[ti],
                            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            track_token_in=track_token,
                        )
                        pred_mask = self.model.visual_model.postprocess_masks(
                            low_res_masks,
                            input_size=resize_list[batch_idx],
                            original_size=original_size_list[batch_idx],
                        )
                    else:
                        _features = _features_all[batch_idx][ti]
                        low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=_features["image_embed"],
                            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=True,
                            high_res_features=_features["high_res_feats"],
                        )
                        pred_mask = self.model.postprocess_masks(
                            low_res_masks,
                            orig_hw=original_size_list[batch_idx],
                        )
                    pred_masks_ti.append(pred_mask[:, 0])
                pred_masks.append(pred_masks_ti)
                continue

            selected_boxes_per_token = [boxes.to(device=seg_embs.device, dtype=prompt_dtype) for boxes in selected_boxes_batch[batch_idx]]
            h_size = images_for_sam[batch_idx].shape[-2]
            w_size = images_for_sam[batch_idx].shape[-1]

            # Decode one SEG token at a time to bound SAM2 peak memory.
            pred_masks_ti = []
            for ti in range(t):
                _features = _features_all[batch_idx][ti]
                token_low_res_list = []
                for token_idx, token_boxes in enumerate(selected_boxes_per_token):
                    token_embeddings = seg_embs_prompt[token_idx].unsqueeze(0).repeat(token_boxes.shape[0], 1)
                    pixel_boxes = token_boxes.clone()
                    pixel_boxes[:, [0, 2]] = pixel_boxes[:, [0, 2]] * w_size
                    pixel_boxes[:, [1, 3]] = pixel_boxes[:, [1, 3]] * h_size
                    pixel_boxes = pixel_boxes.to(prompt_dtype)

                    sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=pixel_boxes,
                        masks=None,
                        text_embeds=token_embeddings.unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(prompt_dtype)
                    dense_embeddings = dense_embeddings.to(prompt_dtype)

                    low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=_features["image_embed"],
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=True,
                        high_res_features=_features["high_res_feats"],
                    )
                    token_low_res = low_res_masks.max(dim=0, keepdim=True).values
                    token_low_res_list.append(token_low_res)

                token_low_res = torch.cat(token_low_res_list, dim=0)
                token_tensor = self.model.postprocess_masks(
                    token_low_res,
                    orig_hw=original_size_list[batch_idx],
                )
                if token_tensor.shape[0] < max_num_seg_tokens_per_sample:
                    pad = torch.zeros(
                        (max_num_seg_tokens_per_sample - token_tensor.shape[0],) + tuple(token_tensor.shape[1:]),
                        dtype=token_tensor.dtype,
                        device=token_tensor.device,
                    )
                    token_tensor = torch.cat([token_tensor, pad], dim=0)
                pred_masks_ti.append(token_tensor[:, 0])
            pred_masks.append(pred_masks_ti)

        ###
        
        model_output = output
        gt_masks = masks_list # [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        # gt_masks = new_masks_list # [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        ### 6 - Calculate losses
        output = model_output.logits

        # Text generation losss (Cross Entropy Loss)
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        
        # Mask BCE & DICE losses
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        
        num_masks = 0
        for batch_idx in range(len(pred_masks)): # for batch_idx in batch_size: (i.e. for each sample in the batch)
            
            gt_mask = gt_masks[batch_idx]       # [num_seg_tokens_per_sample, T_sam, H, W] if video, [num_seg_tokens_per_sample, H, W] if image
            pred_mask = pred_masks[batch_idx]   # (T_sam x [num_seg_tokens_per_sample, H, W]) if video, [num_seg_tokens_per_sample, H, W] if image
            
            gt_mask = [gt_mask[:,tn,:,:]  for tn in range(gt_mask.shape[1]) ]# [num_conversations, T_sam, H, W] -> # T_sam x [num_conversations, H, W]
            t = len(pred_mask)
            assert ( len(gt_mask) == len(pred_mask)), "len(gt_mask): {}, len(pred_mask): {}".format(len(gt_mask), len(pred_mask))
            for ti in range(t):
                pred_mask_i = pred_mask[ti]
                gt_mask_i   = gt_mask[ti] # [num_seg_tokens_per_sample, H, W]
                min_len = min(gt_mask_i.shape[0], pred_mask_i.shape[0])
                gt_mask_i = gt_mask_i[:min_len]
                pred_mask_i = pred_mask_i[:min_len]
                assert ( gt_mask_i.shape[0] == pred_mask_i.shape[0]), "gt_mask_i.shape: {}, pred_mask_i.shape: {}".format(gt_mask_i.shape, pred_mask_i.shape)
                mask_bce_loss += ( sigmoid_ce_loss(pred_mask_i, gt_mask_i, num_masks=gt_mask_i.shape[0], ignore_index=MASK_IGNORE_INDEX) * gt_mask_i.shape[0])
                mask_dice_loss += ( dice_loss(pred_mask_i, gt_mask_i, num_masks=gt_mask_i.shape[0], ignore_index=MASK_IGNORE_INDEX) * gt_mask_i.shape[0])
                num_masks += gt_mask_i.shape[0]
        
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        proposal_loss = ce_loss.new_tensor(0.0)
        proposal_box_loss = ce_loss.new_tensor(0.0)
        if self.model.use_detection_guidance and proposal_losses:
            proposal_loss = torch.stack(proposal_losses).mean()
        if self.model.use_detection_guidance and self.model.use_proposal_box_regression and proposal_box_losses:
            proposal_box_loss = torch.stack(proposal_box_losses).mean()
        loss = (
            ce_loss
            + mask_loss
            + self.model.proposal_loss_weight * proposal_loss
            + self.model.proposal_box_loss_weight * proposal_box_loss
        )
        
        # print_dimensions('gt_masks', gt_masks) # shape: [batch_size, num_seg_tokens_per_sample, T_sam, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        # print_dimensions('pred_masks', pred_masks) # shape: [batch_size, T_sam, num_seg_tokens_per_sample, H, W] if video, [batch_size, num_seg_tokens_per_sample, H, W] if image
        
        # print('ce_loss:', ce_loss, '    mask_bce_loss:', mask_bce_loss, '   mask_dice_loss:', mask_dice_loss, ' mask_loss:', mask_loss)
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "proposal_loss": proposal_loss.detach() if torch.is_tensor(proposal_loss) else proposal_loss,
            "proposal_box_loss": proposal_box_loss.detach() if torch.is_tensor(proposal_box_loss) else proposal_box_loss,
        }

    def inference(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        use_sam2_video_branch=False,
        tsf_images=None,
        return_proposal_debug=False,
    ):

        if use_sam2_video_branch:
            if self.config.use_sam2:
                print('\033[92m---Inference with video branch---\033[0m')
                return self.inference_video_branch(
                    images,
                    context_images,
                    images_for_sam,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens,
                    tsf_images=tsf_images,
                    return_proposal_debug=return_proposal_debug,
                )
            else:
                raise ValueError("use_sam2_video_branch is True, but model is not configured to use SAM2")
        else:
            print('\033[92m---Inference without video branch---\033[0m')
            return self.inference_framewise(
                images,
                context_images,
                images_for_sam,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens,
                tsf_images=tsf_images,
                return_proposal_debug=return_proposal_debug,
            )
            
    def inference_framewise(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tsf_images=None,
        return_proposal_debug=False,
    ):
        
        ### Find the number of newly added tokens
        with torch.no_grad():
            _, _, seg_token_mask, _ = self.__inference_path(input_ids, images, context_images, None, tsf_images=tsf_images)
            
        num_newly_added_tokens = (seg_token_mask.shape[1] - input_ids.shape[1]) # 111 or 255
        
        
        with torch.no_grad():
            outputs = self.generate(
                images=images,
                context_images=context_images,
                tsf_images=tsf_images,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1

            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1])) #(33, 1, 1, 4096) -> (33, 1, 1, 256)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) #(33, 1, 1, 256)
            seg_token_mask = self._build_generated_token_mask(
                output_ids,
                getattr(self.config, "seg_token_idx", None),
                num_newly_added_tokens,
            )
            box_token_mask = self._build_generated_token_mask(
                output_ids,
                getattr(self.config, "box_token_idx", None),
                num_newly_added_tokens,
            )
            infer_offset = torch.arange(
                output_ids.shape[0] + 1,
                device=output_ids.device,
                dtype=torch.long,
            )
            pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, seg_token_mask, infer_offset)
            box_pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, box_token_mask, infer_offset)
            
        selected_boxes_batch: List[List[torch.Tensor]] = []
        proposal_debug_batch: List[Dict[str, object]] = []
        if self.model.use_detection_guidance and self.model.proposal_selector is not None:
            for batch_idx, seg_embs in enumerate(pred_embeddings):
                box_embs = box_pred_embeddings[batch_idx] if batch_idx < len(box_pred_embeddings) else seg_embs.new_empty((0, seg_embs.shape[-1]))
                selector_embs = box_embs if box_embs.numel() > 0 else seg_embs
                if selector_embs.numel() == 0:
                    selected_boxes_batch.append([])
                    proposal_debug_batch.append(
                        {
                            "enabled": True,
                            "reason": "no_selector_embeddings",
                            "proposals": None,
                            "selected_boxes_before_filter": [],
                            "selected_boxes_after_filter": [],
                        }
                    )
                    continue
                proposals = self.model._generate_ddetr_proposals(images_for_sam[batch_idx])
                if proposals is None or proposals["boxes"].numel() == 0:
                    selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                    proposal_debug_batch.append(
                        {
                            "enabled": True,
                            "reason": "no_proposals",
                            "proposals": None,
                            "selected_boxes_before_filter": [],
                            "selected_boxes_after_filter": [],
                        }
                    )
                    continue
                refined_sel, _, _, selected_boxes = self.model._refine_seg_embeddings_with_proposals(
                    selector_embs,
                    proposals,
                    training=False,
                    gt_masks=None,
                    images_for_sam_sample=images_for_sam[batch_idx],
                    sam2_feature_pyramid=None,
                )
                if box_embs.numel() > 0:
                    box_pred_embeddings[batch_idx] = refined_sel
                else:
                    pred_embeddings[batch_idx] = refined_sel
                selected_boxes = self._align_selected_boxes_to_seg_count(
                    selected_boxes,
                    seg_embs.shape[0],
                    seg_embs.device,
                )
                selected_boxes_before_filter = [
                    box.detach().cpu().clone() for box in selected_boxes
                ]
                if self.model.use_box_prior_filter:
                    selected_boxes = self._filter_selected_boxes_with_prior(
                        selected_boxes=selected_boxes,
                        images_for_sam_sample=images_for_sam[batch_idx],
                        orig_hw=original_size_list[batch_idx],
                    )
                selected_boxes_after_filter = [
                    box.detach().cpu().clone() for box in selected_boxes
                ]
                selected_boxes_batch.append(selected_boxes)
                proposal_debug_batch.append(
                    {
                        "enabled": True,
                        "reason": "ok",
                        "proposals": {
                            "boxes": proposals["boxes"].detach().cpu().clone(),
                            "scores": proposals["scores"].detach().cpu().clone(),
                            "mask": proposals["mask"].detach().cpu().clone() if proposals.get("mask") is not None else None,
                        },
                        "selected_boxes_before_filter": selected_boxes_before_filter,
                        "selected_boxes_after_filter": selected_boxes_after_filter,
                    }
                )
        else:
            for seg_embs in pred_embeddings:
                selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                proposal_debug_batch.append(
                    {
                        "enabled": False,
                        "reason": "detection_guidance_disabled",
                        "proposals": None,
                        "selected_boxes_before_filter": [],
                        "selected_boxes_after_filter": [],
                    }
                )

        if self.config.use_sam2:
            USE_SAM2 = True
            USE_SAM1 = False
        else:
            USE_SAM2 = False
            USE_SAM1 = True

        if USE_SAM1:
            image_embeddings_for_sam = self.get_visual_embs(images_for_sam)
        else:
            batch_size, _features_all = self.get_visual_embs_sam2(images_for_sam)

        use_guidance = (
            self.model.use_detection_guidance
            and self.model.proposal_selector is not None
            and USE_SAM2
        )

        pred_masks_batch: List[List[torch.Tensor]] = []
        video_segments_batch: List[dict] = []

        for batch_idx, seg_embs in enumerate(pred_embeddings):
            if USE_SAM2:
                prompt_dtype = self.model.visual_model.sam_prompt_encoder.point_embeddings[0].weight.dtype
            else:
                prompt_dtype = self.model.visual_model.prompt_encoder.point_embeddings[0].weight.dtype
            seg_embs_prompt = seg_embs.to(prompt_dtype)
            num_seg_tokens_per_sample = seg_embs.shape[0]

            if USE_SAM1:
                image_embeddings_for_sam_i = image_embeddings_for_sam[batch_idx]
                t = len(image_embeddings_for_sam_i)
            else:
                t = len(_features_all[batch_idx])

            if num_seg_tokens_per_sample == 0:
                empty_masks = []
                for _ in range(t):
                    h, w = original_size_list[batch_idx]
                    empty_masks.append(
                        torch.zeros((0, h, w), dtype=seg_embs.dtype, device=seg_embs.device)
                    )
                pred_masks_batch.append(empty_masks)
                video_segments_batch.append({})
                continue

            guidance_boxes: Optional[List[torch.Tensor]] = None
            if use_guidance and batch_idx < len(selected_boxes_batch):
                candidate_boxes = selected_boxes_batch[batch_idx]
                if len(candidate_boxes) == num_seg_tokens_per_sample and all(
                    boxes.shape[0] > 0 for boxes in candidate_boxes
                ):
                    guidance_boxes = [boxes.to(device=seg_embs.device, dtype=prompt_dtype) for boxes in candidate_boxes]

            counts: Optional[List[int]] = None
            if guidance_boxes is None:
                if USE_SAM1:
                    sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embs_prompt.unsqueeze(1),
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=seg_embs_prompt.unsqueeze(1),
                    )
                sparse_embeddings = sparse_embeddings.to(prompt_dtype)
                dense_embeddings = dense_embeddings.to(prompt_dtype)
            else:
                counts = [boxes.shape[0] for boxes in guidance_boxes]

            pred_masks_ti: List[torch.Tensor] = []
            track_token = None
            for ti in range(t):
                if USE_SAM1:
                    low_res_masks, _, track_token = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings_for_sam_i[ti],
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        track_token_in=track_token,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[batch_idx],
                        original_size=original_size_list[batch_idx],
                    )
                else:
                    _features = _features_all[batch_idx][ti]
                    if guidance_boxes is not None and counts is not None:
                        h_size = images_for_sam[batch_idx].shape[-2]
                        w_size = images_for_sam[batch_idx].shape[-1]
                        token_low_res_list = []
                        for token_idx, token_boxes in enumerate(guidance_boxes):
                            token_embeddings = seg_embs_prompt[token_idx].unsqueeze(0).repeat(token_boxes.shape[0], 1)
                            pixel_boxes = token_boxes.clone()
                            pixel_boxes[:, [0, 2]] = pixel_boxes[:, [0, 2]] * w_size
                            pixel_boxes[:, [1, 3]] = pixel_boxes[:, [1, 3]] * h_size
                            pixel_boxes = pixel_boxes.to(prompt_dtype)

                            sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
                                points=None,
                                boxes=pixel_boxes,
                                masks=None,
                                text_embeds=token_embeddings.unsqueeze(1),
                            )
                            sparse_embeddings = sparse_embeddings.to(prompt_dtype)
                            dense_embeddings = dense_embeddings.to(prompt_dtype)

                            low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                                image_embeddings=_features["image_embed"],
                                image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                                repeat_image=True,
                                high_res_features=_features["high_res_feats"],
                            )
                            token_low_res = low_res_masks.max(dim=0, keepdim=True).values
                            token_low_res_list.append(token_low_res)
                        token_low_res = torch.cat(token_low_res_list, dim=0)
                        pred_mask = self.model.postprocess_masks(
                            token_low_res,
                            orig_hw=original_size_list[batch_idx],
                        )
                    else:
                        low_res_masks, _, _, _ = self.model.visual_model.sam_mask_decoder(
                            image_embeddings=_features["image_embed"],
                            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=True,
                            high_res_features=_features["high_res_feats"],
                        )
                        pred_mask = self.model.postprocess_masks(
                            low_res_masks,
                            orig_hw=original_size_list[batch_idx],
                        )

                pred_masks_ti.append(pred_mask[:, 0])

            pred_masks_batch.append(pred_masks_ti)

            video_segments = {}
            for ti, mask_tensor in enumerate(pred_masks_ti):
                mask_np = mask_tensor.detach().cpu().numpy() > 0
                for seg_idx in range(mask_np.shape[0]):
                    video_segments.setdefault(ti, {})[seg_idx] = mask_np[seg_idx]
            video_segments_batch.append(video_segments)

        if return_proposal_debug:
            return output_ids, video_segments_batch, proposal_debug_batch
        return output_ids, video_segments_batch

    def inference_video_branch(
        self,
        images,
        context_images,
        images_for_sam,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tsf_images=None,
        return_proposal_debug=False,
    ):
        
        ### Find the number of newly added tokens
        with torch.no_grad():
            _, _, seg_token_mask, _ = self.__inference_path(input_ids, images, context_images, None, tsf_images=tsf_images)
            
        
        num_newly_added_tokens = (seg_token_mask.shape[1] - input_ids.shape[1]) # 111 or 255
        
        
        with torch.no_grad():
            outputs = self.generate(
                images=images,
                context_images=context_images,
                tsf_images=tsf_images,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1

            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1])) #(33, 1, 1, 4096) -> (33, 1, 1, 256)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) #(33, 1, 1, 256)
            seg_token_mask = self._build_generated_token_mask(
                output_ids,
                getattr(self.config, "seg_token_idx", None),
                num_newly_added_tokens,
            )
            box_token_mask = self._build_generated_token_mask(
                output_ids,
                getattr(self.config, "box_token_idx", None),
                num_newly_added_tokens,
            )
            infer_offset = torch.arange(
                output_ids.shape[0] + 1,
                device=output_ids.device,
                dtype=torch.long,
            )
            pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, seg_token_mask, infer_offset)
            box_pred_embeddings = self._split_embeddings_by_offset(last_hidden_state, box_token_mask, infer_offset)

            selected_boxes_batch: List[List[torch.Tensor]] = []
            proposal_debug_batch: List[Dict[str, object]] = []
            if self.model.use_detection_guidance and self.model.proposal_selector is not None:
                for batch_idx, seg_embs in enumerate(pred_embeddings):
                    box_embs = box_pred_embeddings[batch_idx] if batch_idx < len(box_pred_embeddings) else seg_embs.new_empty((0, seg_embs.shape[-1]))
                    selector_embs = box_embs if box_embs.numel() > 0 else seg_embs
                    if selector_embs.numel() == 0:
                        selected_boxes_batch.append([])
                        proposal_debug_batch.append(
                            {
                                "enabled": True,
                                "reason": "no_selector_embeddings",
                                "proposals": None,
                                "selected_boxes_before_filter": [],
                                "selected_boxes_after_filter": [],
                            }
                        )
                        continue
                    proposals = self.model._generate_ddetr_proposals(images_for_sam[batch_idx])
                    if proposals is None or proposals["boxes"].numel() == 0:
                        selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                        proposal_debug_batch.append(
                            {
                                "enabled": True,
                                "reason": "no_proposals",
                                "proposals": None,
                                "selected_boxes_before_filter": [],
                                "selected_boxes_after_filter": [],
                            }
                        )
                        continue
                    refined_sel, _, _, selected_boxes = self.model._refine_seg_embeddings_with_proposals(
                        seg_embeddings=selector_embs,
                        proposals=proposals,
                        training=False,
                        gt_masks=None,
                        images_for_sam_sample=images_for_sam[batch_idx],
                        sam2_feature_pyramid=None,
                    )
                    if box_embs.numel() > 0:
                        box_pred_embeddings[batch_idx] = refined_sel
                    else:
                        pred_embeddings[batch_idx] = refined_sel
                    selected_boxes = self._align_selected_boxes_to_seg_count(
                        selected_boxes,
                        seg_embs.shape[0],
                        seg_embs.device,
                    )
                    selected_boxes_before_filter = [
                        box.detach().cpu().clone() for box in selected_boxes
                    ]
                    if self.model.use_box_prior_filter:
                        selected_boxes = self._filter_selected_boxes_with_prior(
                            selected_boxes=selected_boxes,
                            images_for_sam_sample=images_for_sam[batch_idx],
                            orig_hw=original_size_list[batch_idx],
                        )
                    selected_boxes_after_filter = [
                        box.detach().cpu().clone() for box in selected_boxes
                    ]
                    selected_boxes_batch.append(selected_boxes)
                    proposal_debug_batch.append(
                        {
                            "enabled": True,
                            "reason": "ok",
                            "proposals": {
                                "boxes": proposals["boxes"].detach().cpu().clone(),
                                "scores": proposals["scores"].detach().cpu().clone(),
                                "mask": proposals["mask"].detach().cpu().clone() if proposals.get("mask") is not None else None,
                            },
                            "selected_boxes_before_filter": selected_boxes_before_filter,
                            "selected_boxes_after_filter": selected_boxes_after_filter,
                        }
                    )
            else:
                for seg_embs in pred_embeddings:
                    selected_boxes_batch.append([torch.empty((0, 4), device=seg_embs.device) for _ in range(seg_embs.shape[0])])
                    proposal_debug_batch.append(
                        {
                            "enabled": False,
                            "reason": "detection_guidance_disabled",
                            "proposals": None,
                            "selected_boxes_before_filter": [],
                            "selected_boxes_after_filter": [],
                        }
                    )

                        
            video_segments_batch = []
            for batch_idx in range(len(pred_embeddings)): # for idx in batch_size (i.e. for each sample in the batch)
                
                feat = pred_embeddings[batch_idx] # [num_seg_tokens_per_sample, 256]
                num_seg_tokens_per_sample = len(feat)
                feat = feat.unsqueeze(1) # [num_seg_tokens_per_sample, 1, 256]
                selected_boxes = selected_boxes_batch[batch_idx] if batch_idx < len(selected_boxes_batch) else []
                
                if num_seg_tokens_per_sample==0:
                    video_segments_batch.append({})
                    continue
                
                # SAM2 video model
                video_height, video_width = original_size_list[batch_idx]
                inference_state = self.model.visual_model.init_state_from_tensor(images_for_sam[batch_idx], video_height, video_width)
                self.model.visual_model.reset_state(inference_state)

                # begin video segmentation
                ann_frame_idx = 0  # the frame index we interact with
                
                # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
                # _, out_obj_ids, out_mask_logits = self.model.visual_model.add_new_text(
                #     inference_state=inference_state,
                #     frame_idx=ann_frame_idx,
                #     obj_id=ann_obj_id,
                #     text=feat
                # )
                
                
                # 3) for each SEG token: bind text + all its boxes to one obj_id
                seg_feats = pred_embeddings[batch_idx]              # (N_seg, D)
                for ann_obj_id in range(0, feat.shape[0]):
                    _, out_obj_ids, out_mask_logits = self.model.visual_model.add_new_text(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        text=feat[ann_obj_id].unsqueeze(0)
                    )

                    # add all selected boxes for this SEG on the same obj_id
                    if len(selected_boxes) > ann_obj_id and selected_boxes[ann_obj_id].numel() > 0:
                        boxes_norm = selected_boxes[ann_obj_id].clamp(0,1)                   # (K,4) xyxy in [0,1]
                        boxes_px = boxes_norm.clone()
                        boxes_px[:, [0,2]] *= video_width; boxes_px[:, [1,3]] *= video_height                 # to pixels
                        # (a) if your predictor exposes a boxes API:
                        for b in boxes_px:
                            self.model.visual_model.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=ann_frame_idx,
                                obj_id=ann_obj_id,
                                box=b.unsqueeze(0),
                                normalize_coords=True,
                            )

                
                # run propagation throughout the video and collect the results in a dict
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in self.model.visual_model.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        # out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()[0] # select only one mask per object
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                # video_segments is supposed to contain the per-frame segmentation results per each seg-token in the sample
                video_segments_batch.append(video_segments)
                
        if return_proposal_debug:
            return output_ids, video_segments_batch, proposal_debug_batch
        return output_ids, video_segments_batch



    def estimate_mask_from_box_image(self, bbox, orig_size=None, images_for_sam=None):
        import numpy as np

        if images_for_sam is None:
            raise ValueError("images_for_sam must provide a tensor for SAM2 feature extraction.")

        def _select_first_image(data):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, (list, tuple)) and len(data) > 0:
                return _select_first_image(data[0])
            raise TypeError("images_for_sam must contain at least one tensor.")

        image_tensor = _select_first_image(images_for_sam)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:
            image_tensor = image_tensor[:1]
        else:
            raise ValueError(
                f"Expected image tensor with 3 or 4 dims, but received shape {tuple(image_tensor.shape)}."
            )
        image_tensor = image_tensor.to(device=self.model.visual_model.device, dtype=torch.float32)

        backbone_out = self.model.visual_model.forward_image(image_tensor)
        (
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.model.visual_model._prepare_backbone_features(backbone_out)

        # backbone_out = self.image_encoder(img_batch)

        # Prefer explicit orig_size if provided, otherwise try to infer from inference_state, else fallback to (image_size, image_size)
        if orig_size is None:
            # try to extract original size from inference_state if available
            try:
                h = self.inference_state["original_height"]
                w = self.inference_state["original_width"]
                orig_hw = (h, w)
            except Exception:
                # fallback: use model's image_size as square
                orig_hw = (self.model.visual_model.image_size, self.model.visual_model.image_size)
        else:
            orig_hw = orig_size

        from model.segment_anything_2.sam2.utils.transforms import SAM2Transforms
        _transforms = SAM2Transforms(
            resolution=self.model.visual_model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

        boxes_np = np.asarray(bbox, dtype=np.float32)
        if boxes_np.size == 0:
            raise ValueError("bbox must contain at least one box.")
        single_input = boxes_np.ndim == 1
        if single_input:
            boxes_np = boxes_np[None, :]
        if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
            raise ValueError(f"Expected bbox to have shape (N, 4), but received {boxes_np.shape}.")

        boxes_xyxy = []
        for single in boxes_np:
            x1, y1, x2_or_w, y2_or_h = single.tolist()
            x1 = float(np.clip(x1, 0.0, orig_hw[1]))
            y1 = float(np.clip(y1, 0.0, orig_hw[0]))
            if single_input:
                x2 = x1 + x2_or_w
                y2 = y1 + y2_or_h
            elif x2_or_w <= x1 or y2_or_h <= y1:
                x2 = x1 + x2_or_w
                y2 = y1 + y2_or_h
            else:
                x2 = x2_or_w
                y2 = y2_or_h
            x2 = float(np.clip(x2, 0.0, orig_hw[1]))
            y2 = float(np.clip(y2, 0.0, orig_hw[0]))
            if x2 <= x1:
                x2 = min(orig_hw[1], x1 + 1.0)
            if y2 <= y1:
                y2 = min(orig_hw[0], y1 + 1.0)
            boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        box = torch.as_tensor(boxes_xyxy, dtype=torch.float, device=current_vision_feats[0].device)
        num_boxes = box.shape[0]

        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=orig_hw
        )  # Bx2x2
        box_coords = unnorm_box.reshape(num_boxes, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device).repeat(num_boxes, 1)
        concat_points = (box_coords, box_labels) if num_boxes > 0 else None

        sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
            text_embeds=None,
        )

        # Predict masks
        batched_mode = num_boxes > 1  # multi object prediction
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
        if self.model.visual_model.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.model.visual_model.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])
        low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = _transforms.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        masks = (masks > 0).float()

        prompt_h, prompt_w = self.model.visual_model.sam_prompt_encoder.image_embedding_size
        prompt_size = (prompt_h * 4, prompt_w * 4)
        if masks.shape[-2] != prompt_size[0] or masks.shape[-1] != prompt_size[1]:
            masks = F.interpolate(
                masks,
                size=prompt_size,
                mode="bilinear",
                align_corners=False,
            )
        masks = masks.clamp_(0.0, 1.0)
        masks = masks.to(device=self.model.visual_model.device, dtype=torch.float32)

        """
        # merged = masks.any(dim=0)
        # merged = merged.to(torch.float16)
        # return merged
        """
        
        return masks
    
    

class SPARROWForCausalLM(VideoGPTPlusPhi3ForCausalLM, VideoGLaMM_SAM2):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        ##
        super().__init__(config)
        self.model = VideoGLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return VideoGPTPlusPhi3ForCausalLM.forward(self, **kwargs)
        return VideoGLaMM_SAM2.model_forward(self, **kwargs)
    
    def super_forward(self, **kwargs):
        return VideoGPTPlusPhi3ForCausalLM.forward(self, **kwargs)


# Backward-compatible alias so old imports/checkpoints continue to work.
VideoGLaMMForCausalLM = SPARROWForCausalLM
