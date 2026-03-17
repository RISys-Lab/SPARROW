import random
import os
import gc
import re
from functools import partial
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

from .reason_seg_dataset import ReasonSegDataset, ReasonSegValDataset
from .refer_seg_dataset import ReferSegDataset, ReferSegValDataset
from .sem_seg_dataset import SemSegDataset
from .refer_vos_dataset import ReferVOSDataset
from .video_vqa_dataset import VideoInstruct100kDataset
from .mevis_dataset import MEVISDataset
from .vqa_dataset import VQADataset
from .temporal_grounding_datasets import TemporalGroundingDataset
from .vidstg_dataset import VidSTGDataset
from .video_gcg_dataset import BURST_YTVIS_GCGDataset
from .grandf_dataset import GranDfAllDatasets
from .grounded_video_qa import GroundedVideoQADataset
from .video_gcg_anet import ANetEntitiesGCGDataset
from util.ytvos_gcg import YTVOSGCGDataset
from util.mevis_gcg import MevisGCGDataset
from util.vidstg_hcstvg_gcg import VidSTG_HCSTVG_GCGDataset

from util.itm_transforms import apply_augmentations_and_transforms



# ===================================================================
from .ReVOS_Dataset import VideoReVOSDataset
from .MeVIS_Dataset import VideoMeVISDataset
from .RefYoutubeVOS_Dataset import VideoRefYoutubeVOSDataset
from .ReSAM2_Dataset import VideoSAM2Dataset


# Tracking Datasets
# ======================================================================================================================================

# GOT-10k
# ===================================================================
from .got10k_dataset import GOT10kSeg_Dataset

# LaSOT
# ===================================================================
from .lasot_dataset import LaSOTSeg_Dataset

# VID-Sentence
# ===================================================================
from .vid_dataset import VID2015Seg_Dataset

# A2D_Sentences
# ===================================================================
from .a2d_dataset import A2DSeg_Dataset

# HC-STVG
# ===================================================================
from .hc_stvg_dataset import HCSTVGSeg_Dataset

# ======================================================================================================================================


# Validation Datasets
# ======================================================================================================================================

# eval_grounding.py
# ===================================================================
from .val_vidstg_dataset import VidSTGGroundingDataset
# ===================================================================

# eval_referdavis_infer.py
# ===================================================================
from .refer_davis_dataset import ReferDAVISDataset

# eval_gcg_infer.py
# ===================================================================
from .val_video_gcg_dataset import VAL_BURST_YTVIS_GCGDataset
from .val_mevis_gcg import VAL_MevisGCGDataset
from .val_vidstg_hcstvg_gcg import VAL_VidSTG_HCSTVG_GCGDataset
# ===================================================================

# ======================================================================================================================================




MASK_IGNORE_INDEX = -1
MAX_NUM_SEG_TOKENS_PER_SAMPLE = 4
VISUAL_TOKEN_RESERVE = int(os.getenv("SPARROW_VISUAL_TOKEN_RESERVE", "255"))


def _enforce_box_before_seg(text: str) -> str:
    # convert bare [SEG] -> [BOX] [SEG], but avoid duplication
    return re.sub(r"(?<!\[BOX\]\s)\[SEG\]", "[BOX] [SEG]", text)


def _first_valid_mask_box(mask_4d: torch.Tensor):
    """
    Find the first positive mask across (N_seg, T, H, W) and return:
    (frame_idx, (x1n, y1n, x2n, y2n)) in normalized coordinates.
    """
    if mask_4d is None or not torch.is_tensor(mask_4d) or mask_4d.ndim != 4:
        return None

    n_seg, t_len, h, w = mask_4d.shape
    for seg_idx in range(n_seg):
        for t_idx in range(t_len):
            m = mask_4d[seg_idx, t_idx]
            valid = m != MASK_IGNORE_INDEX
            if not valid.any():
                continue
            pos = (m > 0) & valid
            if not pos.any():
                continue
            idx = pos.nonzero(as_tuple=False)
            y1 = float(idx[:, 0].min().item())
            x1 = float(idx[:, 1].min().item())
            y2 = float(idx[:, 0].max().item() + 1.0)
            x2 = float(idx[:, 1].max().item() + 1.0)
            return t_idx, (x1 / float(w), y1 / float(h), x2 / float(w), y2 / float(h))
    return None


def _to_time_tensor(x):
    if x is None:
        return None
    if isinstance(x, list):
        return torch.stack(x, dim=0)
    if torch.is_tensor(x):
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x
    return None


def _build_tsf_crop_from_mask(masks, images, context_images):
    """
    Build one Target-Specific Feature (TSF) crop tensor per sample.
    Returns a tensor shaped (1, 3, H, W) or None.
    """
    if masks is None:
        return None
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4:
        return None

    match = _first_valid_mask_box(masks)
    if match is None:
        return None
    t_idx, (x1n, y1n, x2n, y2n) = match

    src_seq = context_images if context_images is not None else images
    src_seq = _to_time_tensor(src_seq)
    if src_seq is None or src_seq.ndim != 4 or src_seq.shape[0] == 0:
        return None

    src_t = min(max(int(t_idx), 0), src_seq.shape[0] - 1)
    frame = src_seq[src_t]  # (3, H, W), already preprocessed for CLIP path.
    h_src, w_src = int(frame.shape[-2]), int(frame.shape[-1])

    x1 = max(0, min(w_src - 1, int(np.floor(x1n * w_src))))
    y1 = max(0, min(h_src - 1, int(np.floor(y1n * h_src))))
    x2 = max(x1 + 1, min(w_src, int(np.ceil(x2n * w_src))))
    y2 = max(y1 + 1, min(h_src, int(np.ceil(y2n * h_src))))

    crop = frame[:, y1:y2, x1:x2]
    if crop.numel() == 0:
        return None

    crop = F.interpolate(
        crop.unsqueeze(0).float(),
        size=(h_src, w_src),
        mode="bilinear",
        align_corners=False,
    )[0].to(dtype=frame.dtype)
    return crop.unsqueeze(0)  # (1, 3, H, W)


def _frame_valid_mask_box(mask_3d: torch.Tensor):
    """
    Get a normalized xyxy box from a frame mask tensor shaped (N_seg, H, W).
    Returns None if no positive pixel exists.
    """
    if mask_3d is None or mask_3d.ndim != 3:
        return None
    n_seg, h, w = mask_3d.shape
    for seg_idx in range(n_seg):
        m = mask_3d[seg_idx]
        valid = m != MASK_IGNORE_INDEX
        if not valid.any():
            continue
        pos = (m > 0) & valid
        if not pos.any():
            continue
        idx = pos.nonzero(as_tuple=False)
        y1 = float(idx[:, 0].min().item())
        x1 = float(idx[:, 1].min().item())
        y2 = float(idx[:, 0].max().item() + 1.0)
        x2 = float(idx[:, 1].max().item() + 1.0)
        return (x1 / float(w), y1 / float(h), x2 / float(w), y2 / float(h))
    return None


def _build_tsf_crops_from_mask(masks, images, context_images, target_num_crops=None):
    """
    Build frame-wise TSF crops for video/image samples.
    Returns a tensor shaped (N_tsf, 3, H, W) or None.
    """
    if masks is None:
        return None
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4:
        return None

    src_seq = context_images if context_images is not None else images
    src_seq = _to_time_tensor(src_seq)
    if src_seq is None or src_seq.ndim != 4 or src_seq.shape[0] == 0:
        return None

    num_src = int(src_seq.shape[0])
    if target_num_crops is None or int(target_num_crops) <= 0:
        target_num_crops = num_src
    target_num_crops = max(1, min(int(target_num_crops), num_src))
    src_indices = np.linspace(0, num_src - 1, target_num_crops, dtype=int).tolist()

    num_mask_t = int(masks.shape[1])
    prev_box = None
    crops = []
    for src_t in src_indices:
        frame = src_seq[src_t]
        h_src, w_src = int(frame.shape[-2]), int(frame.shape[-1])

        if num_mask_t > 1 and num_src > 1:
            mask_t = int(round((float(src_t) / float(num_src - 1)) * float(num_mask_t - 1)))
        else:
            mask_t = 0
        mask_t = max(0, min(num_mask_t - 1, mask_t))
        box = _frame_valid_mask_box(masks[:, mask_t])
        if box is None:
            box = prev_box
        if box is None:
            box = (0.0, 0.0, 1.0, 1.0)
        prev_box = box

        x1n, y1n, x2n, y2n = box
        x1 = max(0, min(w_src - 1, int(np.floor(x1n * w_src))))
        y1 = max(0, min(h_src - 1, int(np.floor(y1n * h_src))))
        x2 = max(x1 + 1, min(w_src, int(np.ceil(x2n * w_src))))
        y2 = max(y1 + 1, min(h_src, int(np.ceil(y2n * h_src))))
        crop = frame[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            crop = frame
        crop = F.interpolate(
            crop.unsqueeze(0).float(),
            size=(h_src, w_src),
            mode="bilinear",
            align_corners=False,
        )[0].to(dtype=frame.dtype)
        crops.append(crop)

    if len(crops) == 0:
        return None
    return torch.stack(crops, dim=0)


class LazyDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory
        self._dataset = None

    @property
    def loaded(self):
        return self._dataset is not None

    def _ensure_loaded(self):
        if self._dataset is None:
            print(f"[HybridDataset] loading heavy dataset: {self.name}")
            self._dataset = self._factory()
        return self._dataset

    def unload(self):
        if self._dataset is not None:
            print(f"[HybridDataset] unloading heavy dataset: {self.name}")
            self._dataset = None
            gc.collect()

    def __len__(self):
        return len(self._ensure_loaded())

    def __getitem__(self, idx):
        return self._ensure_loaded()[idx]

def collate_fn(
    batch, tokenizer=None, local_rank=-1,
    conversation_generator=None,
):
    
    tokenizer_image_token = conversation_generator.tokenizer_image_token
    
    ###
    image_path_list = []
    images_sam_list = []
    images_list = []
    context_images_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    tsf_images_list = []
    
    for sample in batch:
        image_path_list.append(sample['file_path'])
        images_sam_list.append(sample['preprocessed_for_sam'])
        images_list.append(sample['images'])
        context_images_list.append(sample['context_images'])
        conversation_list.extend(sample['conversations']) # NOTE: extend is used, not append
        masks_list.append(sample['masks'])
        label_list.append(sample['label'])
        resize_list.append(sample['resize'])
        questions_list.append(sample['questions'])
        sampled_classes_list.append(sample['sampled_classes'])
        cnt += len(sample['conversations'])
        offset_list.append(cnt)
        inferences.append(sample['inference'])

        if getattr(conversation_generator, "use_tsf_token", False):
            tsf_crop = _build_tsf_crops_from_mask(
                masks=sample.get("masks", None),
                images=sample.get("images", None),
                context_images=sample.get("context_images", None),
                target_num_crops=getattr(conversation_generator, "NUM_FRAMES", None),
            )
        else:
            tsf_crop = None
        tsf_images_list.append(tsf_crop)

    ##############################

    # Final guard: enforce [BOX] before each [SEG] for all conversation strings,
    # even if a dataset path bypassed conversation_generator.apply*.
    conversation_list = [_enforce_box_before_seg(prompt) for prompt in conversation_list]
    bare_seg_count = sum(
        1 for prompt in conversation_list if re.search(r"(?<!\[BOX\]\s)\[SEG\]", prompt)
    )
    if bare_seg_count > 0:
        raise RuntimeError(
            f"[BOX] enforcement failed: found {bare_seg_count} conversations with bare [SEG]."
        )
    
    # apply tokenizer with <image> converted to IMAGE_TOKEN_INDEX which is -200 in this case, and the rest of the text tokenized as usual
    input_ids = [ tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list ]
    
    # convert a list of variable-length input_ids into a single tensor with padded sequences,
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # create attention_masks to avoid the PAD tokens
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    # apply preprocess_fn to the conversation_list, to mask targets accordingly
    conversation_generator.preprocess_fn(conversation_list, targets, tokenizer)

    # if training, truncate input_ids, targets, attention_masks to (model_max_length - 255) to accomodate the image_embedding_features 
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - VISUAL_TOKEN_RESERVE

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
    ##############################
    
    return {
        "image_paths": image_path_list,
        
        "images_for_sam": [torch.stack(img, dim=0) if type(img) is list else img for img in images_sam_list], # batch x [T_sam, 3, 1024,1024]
        "images": [torch.stack(img, dim=0) if type(img) is list else img for img in images_list], # batch x [T, 3, 224, 224]
        "context_images": [torch.stack(img, dim=0) if type(img) is list else img for img in context_images_list], # batch x [T, 3, 224, 224]
        "tsf_images": tsf_images_list,  # batch x [(N_tsf, 3, 336, 336) or None]
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list, #[m.unsqueeze(1) if len(m.shape) == 3 else m for m in masks_list], # batch_size x [num_seg_tokens_per_sample, T_sam, H, W] 
        
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        
        # "questions_list": questions_list, # Is this ever used in train_ds.py or LISA.py ?
        # "sampled_classes_list": sampled_classes_list, #Is this ever used in train_ds.py or LISA.py ?
        "inference": inferences[0],
        # "conversation_list": conversation_list,
    }

class HybridDataset(torch.utils.data.Dataset):
    HEAVY_DATASET_NAMES = {
        "sam2",
        "revos",
        "rvos",
        "val_grounding",
        "val_video_gcg",
        "val_davis",
    }

    def __init__(
        self,
        base_image_dir, base_video_dir,
        
        enc_preprocessor, 
        sam_preprocessor,
        conversation_generator,
        
        num_samples_per_epoch = 80000, num_classes_per_sample: int = 3,
        random_sampling = True,
        
        dataset=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        sample_rate=[1,1,1,1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        refer_vos_data="ytvos||davis17||a2d||jhmdb",
        video_vqa_data="video_instruct_100k",
        video_tg_data="charades||anetcaps||qvh",
        
        num_frames_for_sam = -1,
        
        reason_seg_explanatory=0.1,
        lazy_load_heavy_datasets=True,
        max_loaded_heavy_datasets=2,
        dataset_sticky_steps=32,
    ):
        
        self.num_frames_for_sam = num_frames_for_sam
        
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        
        self.random_sampling = random_sampling
                
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        
        self.dataset = dataset
        self.reason_seg_explanatory = reason_seg_explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.base_video_dir = base_video_dir

        self.datasets = dataset
        
        self.num_samples_per_epoch = num_samples_per_epoch
        self.lazy_load_heavy_datasets = lazy_load_heavy_datasets
        self.max_loaded_heavy_datasets = max(1, int(max_loaded_heavy_datasets))
        self.dataset_sticky_steps = max(1, int(dataset_sticky_steps))
        self._sticky_dataset_index = None
        self._sticky_dataset_remaining = 0
        self._heavy_dataset_lru = OrderedDict()
        self._dataset_len_cache = {}
        self._dataset_active_mask = None
        self._active_sample_rate = None

        def _maybe_lazy(name, factory):
            if self.lazy_load_heavy_datasets and name in self.HEAVY_DATASET_NAMES:
                return LazyDatasetWrapper(name, factory)
            return factory()

        def _resolve_path(env_key, *candidates):
            env_val = os.getenv(env_key, "").strip()
            if env_val:
                return env_val
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate
            return candidates[0] if candidates else ""

        sa2va_root = _resolve_path(
            "SA2VA_VIDEO_DATA_ROOT",
            os.path.join(base_video_dir, "sa2va"),
            os.path.join(base_video_dir, "video_datas"),
        )
        vot_root = _resolve_path(
            "VOT_DATA_ROOT",
            os.path.join(base_video_dir, "VOT_dataset"),
            os.path.join(base_video_dir, "vot_dataset"),
        )

        self.all_datasets = []
        for dataset in self.datasets:
            
            ## Image datasets
            # =====================================================================================
            if dataset == "sem_seg":
                print("SemSegDataset")
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                print("ReferSegDataset")
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                print("VQADataset")
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                print("ReasonSegDataset")
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        num_classes_per_sample,
                        reason_seg_data,
                        reason_seg_explanatory,
                    )
                )
            elif dataset == "grandf":
                print("GranDfAllDatasets")
                self.all_datasets.append(
                    GranDfAllDatasets(
                        base_image_dir,
                        enc_preprocessor,
                        sam_preprocessor,
                        conversation_generator,
                        image_set="train",
                    )
                )
            # =====================================================================================
                

            ## video datasets
            # =====================================================================================
            
            # =====================================================================================

            # =====================================================================================
            
            elif dataset == "refer_vos":
                print("ReferVOSDataset")
                self.all_datasets.append(
                    ReferVOSDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        refer_vos_data=refer_vos_data,
                        image_set="train",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            elif dataset == "video_vqa":
                print("VideoInstruct100kDataset")
                self.all_datasets.append(
                    VideoInstruct100kDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        video_vqa_data=video_vqa_data,
                    )
                )
            elif dataset == "mevis":
                print("MEVISDataset")
                self.all_datasets.append(
                    MEVISDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "temporal":
                print("TemporalGroundingDataset")
                self.all_datasets.append(
                    TemporalGroundingDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        tg_data=video_tg_data,
                        image_set="train",
                    )
                )
            elif dataset == "vidstg":
                print("VidSTGDataset")
                self.all_datasets.append(
                    VidSTGDataset(
                        base_video_dataset_dir=base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "gvqa":
                print("GroundedVideoQADataset")
                self.all_datasets.append(
                    GroundedVideoQADataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            # =====================================================================================


            # =====================================================================================

            elif dataset == "sam2":
                print("VideoSAM2Dataset")

                sam2_folder = _resolve_path(
                    "SAM2_FOLDER",
                    os.path.join(sa2va_root, "sam_v_full"),
                    os.path.join(base_video_dir, "sam_v_full"),
                )
                expression_file = _resolve_path(
                    "SAM2_EXPRESSION_FILE",
                    os.path.join(sa2va_root, "sam_v_final_v3.json"),
                    os.path.join(base_video_dir, "sam_v_final_v3.json"),
                )

                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            VideoSAM2Dataset,
                            sam2_folder=sam2_folder,
                            expression_file=expression_file,
                            enc_preprocessor=enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            num_frames_for_sam=num_frames_for_sam,
                            select_number=1
                        ),
                    )
                )

            elif dataset == "revos":
                print("VideoReVOSDataset")

                data_root_revos = _resolve_path(
                    "REVOS_ROOT",
                    os.path.join(sa2va_root, "revos"),
                    os.path.join(base_video_dir, "revos"),
                )
                video_revos_image_folder = data_root_revos
                video_revos_expression_file = os.path.join(data_root_revos, "meta_expressions_train_.json")
                video_revos_mask_file = os.path.join(data_root_revos, "mask_dict.json")

                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            VideoReVOSDataset,
                            image_folder = video_revos_image_folder,
                            expression_file = video_revos_expression_file,
                            mask_file=video_revos_mask_file,
                            enc_preprocessor = enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            num_frames_for_sam=num_frames_for_sam
                        ),
                    )
                )
            
            elif dataset == "rvos":
                print("VideoRefYoutubeVOSDataset")

                data_root_refytvos = _resolve_path(
                    "RVOS_ROOT",
                    os.path.join(sa2va_root, "rvos"),
                    os.path.join(base_video_dir, "rvos"),
                )
                video_refytvos_image_folder = os.path.join(data_root_refytvos, "train", "JPEGImages")
                video_refytvos_expression_file = os.path.join(
                    data_root_refytvos, "meta_expressions", "train", "meta_expressions.json"
                )
                video_refytvos_mask_file = os.path.join(data_root_refytvos, "mask_dict.pkl")

                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            VideoRefYoutubeVOSDataset,
                            image_folder = video_refytvos_image_folder,
                            expression_file = video_refytvos_expression_file,
                            mask_file=video_refytvos_mask_file,
                            enc_preprocessor = enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            num_frames_for_sam=num_frames_for_sam
                        ),
                    )
                )

            
            elif dataset == "mevis_sa2va":
                print("VideoMeVISDataset")

                data_root_mevis = _resolve_path(
                    "MEVIS_SA2VA_ROOT",
                    os.path.join(sa2va_root, "mevis", "train"),
                    os.path.join(base_video_dir, "mevis", "train"),
                )
                video_mevis_image_folder = os.path.join(data_root_mevis, "JPEGImages")
                video_mevis_expression_file = os.path.join(data_root_mevis, "meta_expressions.json")
                video_mevis_mask_file = os.path.join(data_root_mevis, "mask_dict.json")

                self.all_datasets.append(
                    VideoMeVISDataset(
                        image_folder = video_mevis_image_folder,
                        expression_file = video_mevis_expression_file,
                        mask_file=video_mevis_mask_file,
                        enc_preprocessor = enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        num_frames_for_sam=num_frames_for_sam
                    )
                )
            

            # GOT-10k
            # ======================================================
            elif dataset == 'got10k_train':
                print("GOT10kSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "GOT10K_ROOT",
                    os.path.join(vot_root, "GOT-10k"),
                    os.path.join(base_video_dir, "GOT-10k"),
                )
                self.all_datasets.append(
                    GOT10kSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            
            elif dataset == 'got10k_val':
                print("GOT10kSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "GOT10K_ROOT",
                    os.path.join(vot_root, "GOT-10k"),
                    os.path.join(base_video_dir, "GOT-10k"),
                )
                self.all_datasets.append(
                    GOT10kSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="val",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            
            elif dataset == 'lasot':
                print("LaSOTSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "LASOT_ROOT",
                    os.path.join(vot_root, "LaSOT"),
                    os.path.join(base_video_dir, "LaSOT"),
                )
                self.all_datasets.append(
                    LaSOTSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            
            elif dataset == 'a2d_sentences':
                print("A2DSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "A2D_SENTENCES_ROOT",
                    os.path.join(vot_root, "A2D_sentences"),
                    os.path.join(base_video_dir, "A2D_sentences"),
                )
                self.all_datasets.append(
                    A2DSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )

            elif dataset == 'hc_stvg_train':
                print("HCSTVGSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "HC_STVG_ROOT",
                    os.path.join(vot_root, "HC-STVG"),
                    os.path.join(base_video_dir, "HC-STVG"),
                )
                self.all_datasets.append(
                    HCSTVGSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        split="train",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )
            
            elif dataset == 'hc_stvg_val':
                print("HCSTVGSeg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "HC_STVG_ROOT",
                    os.path.join(vot_root, "HC-STVG"),
                    os.path.join(base_video_dir, "HC-STVG"),
                )
                self.all_datasets.append(
                    HCSTVGSeg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        split="val",
                        num_frames_for_sam=num_frames_for_sam,
                    )
                )

            elif dataset == 'vid_train':
                print("VID2015Seg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "VID_SENTENCE_ROOT",
                    os.path.join(vot_root, "VID-Sentence", "VID"),
                    os.path.join(base_video_dir, "VID-Sentence", "VID"),
                )
                self.all_datasets.append(
                    VID2015Seg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,     # your existing object
                        sam_preprocessor=sam_preprocessor,     # your existing object
                        conversation_generator=conversation_generator,  # your existing object
                        split="train",
                        num_frames_for_sam=num_frames_for_sam,                  # or >1 if you want multi-frame SAM
                    )
                )
            
            elif dataset == 'vid_val':
                print("VID2015Seg_Dataset")
                VIDEO_DATA_ROOT = _resolve_path(
                    "VID_SENTENCE_ROOT",
                    os.path.join(vot_root, "VID-Sentence", "VID"),
                    os.path.join(base_video_dir, "VID-Sentence", "VID"),
                )
                self.all_datasets.append(
                    VID2015Seg_Dataset(
                        base_video_dataset_dir=VIDEO_DATA_ROOT,
                        enc_preprocessor=enc_preprocessor,     # your existing object
                        sam_preprocessor=sam_preprocessor,     # your existing object
                        conversation_generator=conversation_generator,  # your existing object
                        split="val",
                        num_frames_for_sam=num_frames_for_sam,                  # or >1 if you want multi-frame SAM
                    )
                )
            

            # Validation Datasets
            # ========================================================================================================================================================


            # eval_grounding.py
            # ============================================================================
            elif dataset == "val_grounding":
                print("VidSTGGroundingDataset")
                
                image_size = 224
                sample_fps = 1
                max_num_frames = 40
                tmp_loc = True

                vidstg_vid_dir = os.path.join(base_video_dir, "vidstg/video")
                vidstg_ann_dir = os.path.join(base_video_dir,'processed/vidstg/vidstg_annotations')
                vidstg_ann_file = os.path.join(vidstg_ann_dir, "test.json")
                
                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            VidSTGGroundingDataset,
                            vidstg_vid_dir,
                            ann_file=vidstg_ann_file,
                            enc_preprocessor=enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            image_set="test",
                            num_frames_for_sam = num_frames_for_sam,
                            video_max_len=max_num_frames,
                            video_max_len_train=max_num_frames, 
                            fps=sample_fps, 
                            tmp_crop=False, # No random temporal cropping
                            tmp_loc=tmp_loc,
                        ),
                    )
                )
            # ============================================================================


            # eval_gcg_infer.py
            # ============================================================================
            elif dataset == "val_video_gcg":
                print("VAL_BURST_YTVIS_GCGDataset")
                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            VAL_BURST_YTVIS_GCGDataset,
                            base_video_dir,
                            enc_preprocessor=enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            image_set="test",
                            num_frames_for_sam = num_frames_for_sam,
                            )
                    )
                )

            elif dataset == "val_mevis_gcg":
                print("VAL_MevisGCGDataset")
                self.all_datasets.append(
                    VAL_MevisGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="valid_u",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )

            elif dataset == "val_vidstg_gcg":
                print("VAL_VidSTG_HCSTVG_GCGDataset")
                self.all_datasets.append(
                    VAL_VidSTG_HCSTVG_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="val",
                        num_frames_for_sam = num_frames_for_sam,
                        source_dataset='vidstg',
                    )
                )
            # ============================================================================

            # eval_referdavis_infer.py
            # ============================================================================
            elif dataset == "val_davis":
                print("ReferDAVISDataset")
                self.all_datasets.append(
                    _maybe_lazy(
                        dataset,
                        partial(
                            ReferDAVISDataset,
                            base_video_dir,
                            enc_preprocessor=enc_preprocessor,
                            sam_preprocessor=sam_preprocessor,
                            conversation_generator=conversation_generator,
                            image_set="valid",
                            num_frames_for_sam=num_frames_for_sam,
                        )
                    )
                )
            # ============================================================================


            


            # =====================================================================================
            elif dataset == "video_gcg":
                print("BURST_YTVIS_GCGDataset")
                self.all_datasets.append(
                    BURST_YTVIS_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )            
            elif dataset == "anet_gcg":
                print("ANetEntitiesGCGDataset")
                self.all_datasets.append(
                    ANetEntitiesGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "ytvos_gcg":
                print("YTVOSGCGDataset")
                self.all_datasets.append(
                    YTVOSGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "mevis_gcg":
                print("MevisGCGDataset")
                self.all_datasets.append(
                    MevisGCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                    )
                )
            elif dataset == "vidstg_gcg":
                print("VidSTG_HCSTVG_GCGDataset - source_dataset: vidstg")
                self.all_datasets.append(
                    VidSTG_HCSTVG_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                        source_dataset='vidstg',
                    )
                )
            elif dataset == "hcstvg_gcg":
                print("VidSTG_HCSTVG_GCGDataset - source_dataset: hcstvg")
                self.all_datasets.append(
                    VidSTG_HCSTVG_GCGDataset(
                        base_video_dir,
                        enc_preprocessor=enc_preprocessor,
                        sam_preprocessor=sam_preprocessor,
                        conversation_generator=conversation_generator,
                        image_set="train",
                        num_frames_for_sam = num_frames_for_sam,
                        source_dataset='hcstvg',
                    )
                )
            # =====================================================================================


            else: 
                raise Exception(f'Unsupported dataset type: {dataset}')

        self._dataset_active_mask = np.array(self.sample_rate > 0, dtype=bool)
        self._refresh_active_sample_rate()
        
        if not self.random_sampling:
            self.concatenated_dataset = torch.utils.data.ConcatDataset(self.all_datasets)
        else:
            self.concatenated_dataset = None

    def _refresh_active_sample_rate(self):
        probs = self.sample_rate.copy()
        probs[~self._dataset_active_mask] = 0.0
        total = probs.sum()
        if total > 0:
            self._active_sample_rate = probs / total
        else:
            self._active_sample_rate = None

    def _deactivate_dataset(self, ds_ind, reason):
        if not self._dataset_active_mask[ds_ind]:
            return
        ds_name = self.datasets[ds_ind]
        print(f"[HybridDataset] disabling dataset '{ds_name}': {reason}")
        self._dataset_active_mask[ds_ind] = False
        self._refresh_active_sample_rate()
        if self._sticky_dataset_index == ds_ind:
            self._sticky_dataset_index = None
            self._sticky_dataset_remaining = 0

    def _get_dataset_len(self, ds_ind):
        if ds_ind in self._dataset_len_cache:
            return self._dataset_len_cache[ds_ind]

        ds = self.all_datasets[ds_ind]
        ds_name = self.datasets[ds_ind]
        try:
            ds_len = int(len(ds))
        except Exception as exc:
            self._dataset_len_cache[ds_ind] = 0
            self._deactivate_dataset(ds_ind, f"len() failed ({type(exc).__name__}: {exc})")
            return 0

        self._dataset_len_cache[ds_ind] = ds_len
        if ds_len <= 0:
            self._deactivate_dataset(ds_ind, "len(dataset) == 0")
        return ds_len

    def _sample_dataset_index(self):
        if self._active_sample_rate is None:
            raise RuntimeError("All configured datasets are inactive or empty. Update --dataset to exclude empty sources.")

        active_indices = np.flatnonzero(self._dataset_active_mask)

        if self.dataset_sticky_steps == 1:
            return int(np.random.choice(active_indices, p=self._active_sample_rate[active_indices]))

        if self._sticky_dataset_index is not None and not self._dataset_active_mask[self._sticky_dataset_index]:
            self._sticky_dataset_index = None
            self._sticky_dataset_remaining = 0

        if self._sticky_dataset_index is None or self._sticky_dataset_remaining <= 0:
            self._sticky_dataset_index = int(np.random.choice(active_indices, p=self._active_sample_rate[active_indices]))
            self._sticky_dataset_remaining = self.dataset_sticky_steps

        self._sticky_dataset_remaining -= 1
        return self._sticky_dataset_index

    def _mark_heavy_dataset_recent(self, ds_ind):
        ds = self.all_datasets[ds_ind]
        if not isinstance(ds, LazyDatasetWrapper):
            return
        self._heavy_dataset_lru[ds_ind] = None
        self._heavy_dataset_lru.move_to_end(ds_ind)

    def _enforce_heavy_dataset_limit(self, keep_index):
        if not self.lazy_load_heavy_datasets:
            return

        while True:
            loaded_indices = [
                i
                for i, ds in enumerate(self.all_datasets)
                if isinstance(ds, LazyDatasetWrapper) and ds.loaded
            ]
            if len(loaded_indices) <= self.max_loaded_heavy_datasets:
                break

            evict_index = None
            for candidate in self._heavy_dataset_lru.keys():
                if candidate == keep_index:
                    continue
                ds = self.all_datasets[candidate]
                if isinstance(ds, LazyDatasetWrapper) and ds.loaded:
                    evict_index = candidate
                    break

            if evict_index is None:
                for candidate in loaded_indices:
                    if candidate != keep_index:
                        evict_index = candidate
                        break

            if evict_index is None:
                break

            self.all_datasets[evict_index].unload()

    def __len__(self):
        # return len(self.concatenated_dataset)
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        
        if self.random_sampling:
            data_sample = None
            max_tries = max(1, len(self.datasets))
            for _ in range(max_tries):
                ds_ind = self._sample_dataset_index()
                ds_len = self._get_dataset_len(ds_ind)
                if ds_len <= 0:
                    continue

                ds = self.all_datasets[ds_ind]
                self._mark_heavy_dataset_recent(ds_ind)
                ind = np.random.randint(0, ds_len)
                data_sample = ds[ind]
                self._enforce_heavy_dataset_limit(keep_index=ds_ind)
                break

            if data_sample is None:
                raise RuntimeError("Unable to sample a valid non-empty dataset item from configured datasets.")
        else:
            data_sample = self.concatenated_dataset[idx]
        
        
        data_sample_new = {
            "file_path": data_sample['file_path'],
            "conversations": data_sample['conversations'],
            "label": data_sample['label'],
            "resize": data_sample['resize'],
            "questions": data_sample['questions'],
            "sampled_classes": data_sample['sampled_classes'],        
        }
        
        preprocessed_for_sam = data_sample['preprocessed_for_sam'] # T_samx[3, 1024, 1024] (video) or [3, 1024, 1024]    (image)
        preprocessed_images = data_sample['images'] # Tx[3, 224, 224]  (video) or [3, 224, 224]      (image) 
        preprocessed_context_images = data_sample['context_images']
        masks = data_sample['masks'] # [num_masks, T_sam, h, w] (video) or [num_masks, h, w] (image)
        
        preprocessed_for_sam = preprocessed_for_sam.unsqueeze(0) if type(preprocessed_for_sam) is not list else torch.stack(preprocessed_for_sam, dim=0)     # [T_sam, 3, 1024,1024]
        preprocessed_images = preprocessed_images.unsqueeze(0) if type(preprocessed_images) is not list else torch.stack(preprocessed_images, dim=0) # [T, 3, 224, 224] 
        # preprocessed_context_images = preprocessed_context_images.unsqueeze(0) if (type(preprocessed_context_images) is not list and preprocessed_context_images is not None) else torch.stack(preprocessed_context_images, dim=0) # [T, 3, 224, 224]
        if preprocessed_context_images is not None:
            preprocessed_context_images = preprocessed_context_images.unsqueeze(0) if type(preprocessed_context_images) is not list else torch.stack(preprocessed_context_images, dim=0) # [T, 3, 224, 224]
            
        
        masks = masks.float()
        masks = masks.unsqueeze(1) if len(masks.shape) == 3 else masks # [num_seg_tokens_per_sample, T_sam, H, W]
                
        # 
        if masks.shape[0] > MAX_NUM_SEG_TOKENS_PER_SAMPLE:
            masks = masks[:MAX_NUM_SEG_TOKENS_PER_SAMPLE]
        if masks.shape[0] < MAX_NUM_SEG_TOKENS_PER_SAMPLE:
            # # add masks filled with MASK_IGNORE_INDEX
            _pad = torch.full((MAX_NUM_SEG_TOKENS_PER_SAMPLE - masks.shape[0], *masks.shape[1:]), MASK_IGNORE_INDEX, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, _pad], dim=0)
        
        # preprocessed_for_sam: [T_sam, 3, 1024,1024]
        # masks: [num_seg_tokens_per_sample, T_sam, H, W]
        # if T_sam != num_frames_for_sam, apply augmentations to preprocessed_for_sam and masks
        if preprocessed_for_sam.shape[0] != self.num_frames_for_sam:
            preprocessed_for_sam, masks = apply_augmentations_and_transforms(preprocessed_for_sam, masks, T_train=self.num_frames_for_sam)
        
        data_sample_new['preprocessed_for_sam'] = preprocessed_for_sam
        data_sample_new['images'] = preprocessed_images
        data_sample_new['context_images'] = preprocessed_context_images
        data_sample_new['masks'] = masks
        
        data_sample_new['inference'] = False
        
        return data_sample_new


class ValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        val_datasets = 'ReasonSeg|val', #"ReasonSeg|val||refcocog|umd|val",
    ):
        self.all_datasets = []
        
        for val_dataset in val_datasets.split("||"):
            splits = val_dataset.split("|")
            if len(splits) == 2: # ReasonSeg|val
                self.dataset = ReasonSegValDataset(base_image_dir,vision_tower)
            elif len(splits) == 3: # refcocog|umd|val
                self.dataset = ReferSegValDataset(base_image_dir,vision_tower, val_dataset="refcocog|umd|val")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


from util.video_gcg_dataset import BURST_YTVIS_GCGBaseDataset
from util.mevis_gcg import MevisGCGBaseDataset
from util.vidstg_hcstvg_gcg import VidSTG_HCSTVG_GCGBaseDataset

class ValGCGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_video_dir,
        val_datasets = 'video_gcg||mevis_gcg||vidstg_gcg', 
    ):
        self.all_datasets = []
        
        for val_dataset in val_datasets.split("||"):
            if val_dataset == "video_gcg":
                dataset = BURST_YTVIS_GCGBaseDataset(base_video_dir,
                                image_set="test", 
                                max_num_frames=40)
                self.all_datasets.append(dataset)
            elif val_dataset == "mevis_gcg":
                dataset = MevisGCGBaseDataset(base_video_dir,
                                image_set="valid_u")
                self.all_datasets.append(dataset)
            elif val_dataset == "vidstg_gcg":
                dataset = VidSTG_HCSTVG_GCGBaseDataset(base_video_dir,
                                image_set="val", 
                                source_dataset='vidstg')
                self.all_datasets.append(dataset)
                
        # concatenate all the datasets
        self.dataset = torch.utils.data.ConcatDataset(self.all_datasets)
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
