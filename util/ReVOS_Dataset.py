# ==========================================================================================================
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
import pycocotools.mask as maskUtils
# ==========================================================================================================

class VideoReVOSDataset(Dataset):
    """
    ReVOS -> ReferVOS-style dataset adapter.

    Inputs:
        image_folder: root folder containing video frame folders (video/<frame_id>.jpg)
        expression_file: JSON with structure: {"videos": { vid_name: {
                "frames": [frame_id, ...],
                "expressions": { exp_id: { "exp": str, "anno_id": <id> or [ids], "obj_id": [optional] } }
        }}}
        mask_file: JSON mapping anno_id -> list[per-frame RLEs for possibly multiple objects/annotations]
                   (compatible with your original decode logic)
        enc_preprocessor: object with .preprocess(pil_images) -> dict(images, context_images, ...)
        sam_preprocessor: object with .preprocess(np_image) -> (preprocessed, resize_info)
        conversation_generator: object with attributes:
            - DEFAULT_VIDEO_TOKEN (str)
            - apply(conversations_list) -> processed conversations

    Notes:
        - num_frames_for_clip controls frames passed to enc_preprocessor.
        - num_frames_for_sam controls frames passed to SAM branch (-1 to reuse clip frames).
        - We generate a single caption/phrase from the first expression in the selected meta item.
        - Masks are returned as [num_objects, T, H, W] (uint8 0/1), label is [H, W] filled with ignore_label.
    """
    ignore_label = 255

    def __init__(
        self,
        image_folder: str,
        expression_file: str,
        mask_file: str,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set: str = "train",
        num_frames_for_clip: int = 5,
        num_frames_for_sam: int = 1,
        frame_contiguous_sample: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        assert image_set in ["train", "val", "test"], f"invalid image_set:{image_set}"
        self.image_folder = image_folder
        self.image_set = image_set

        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator

        self.num_frames_for_clip = int(num_frames_for_clip)
        self.num_frames_for_sam = int(num_frames_for_sam)
        self.frame_contiguous_sample = bool(frame_contiguous_sample)

        random.seed(seed)
        np.random.seed(seed)

        # Conversation scaffolding (matches your target format)
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatial segmentation of {phrase}",
        ]
        self.ANSWER_LIST = [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
        ]

        # Load metadata (compatible with your original code)
        self.vid2metaid, self.metas, self.mask_dict = self._preload_json(expression_file, mask_file)
        self.videos = list(self.vid2metaid.keys())

        # Construct a flat index over (video, meta_id)
        # Each meta corresponds to one expression track with anno_ids and frame list
        self.index = []
        for vid in self.videos:
            for mid in self.vid2metaid[vid]:
                self.index.append(mid)

    def __len__(self):
        return len(self.index)

    # ---------- JSON & mask utils ----------

    def _preload_json(self, expression_file, mask_file):
        with open(expression_file, "r") as f:
            expression_datas = json.load(f)["videos"]

        metas = []
        anno_count = 0
        vid2metaid = {}
        for vid_name, vid_express_data in expression_datas.items():
            vid_frames = sorted(vid_express_data["frames"])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data["expressions"].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data["expressions"][exp_id]
                meta = {
                    "video": vid_name,
                    "exp": exp_dict["exp"],
                    "mask_anno_id": exp_dict.get("anno_id", None),
                    "obj_id": exp_dict.get("obj_id", [0]),
                    "anno_id": [str(anno_count)],
                    "frames": vid_frames,
                    "exp_id": exp_id,
                    "length": vid_len,
                }
                # Normalize mask_anno_id to list[str]
                if isinstance(meta["mask_anno_id"], (list, tuple)):
                    meta["mask_anno_id"] = [str(x) for x in meta["mask_anno_id"]]
                elif meta["mask_anno_id"] is None:
                    meta["mask_anno_id"] = []
                else:
                    meta["mask_anno_id"] = [str(meta["mask_anno_id"])]

                metas.append(meta)
                vid2metaid.setdefault(vid_name, []).append(len(metas) - 1)
                anno_count += 1

        # try:
        with open(mask_file, "rb") as f:
            mask_dict = json.load(f)
        # except:
            # with open(mask_file, "rb") as f:
                # mask_dict = pickle.load(f)

        return vid2metaid, metas, mask_dict

    @staticmethod
    def _decode_video_masks(video_masks, target_size_hw):
        """
        video_masks: list over objects -> list over annos -> list over frames -> RLE or None
        Returns torch.uint8 tensor [num_objects, T, H, W] with 0/1 values.
        """
        H, W = target_size_hw
        ret = []
        for object_masks in video_masks:
            if len(object_masks) == 0:
                # None object; produce zeros
                ret.append(np.zeros((0, H, W), dtype=np.uint8))
                continue

            # Merge annotations per frame for the object
            merged_per_frame = []
            num_frames = len(object_masks[0])
            for i_frame in range(num_frames):
                m_accum = np.zeros((H, W), dtype=np.uint8)
                for i_anno in range(len(object_masks)):
                    rle = object_masks[i_anno][i_frame]
                    if rle is None:
                        continue
                    m = maskUtils.decode(rle)
                    if m.ndim == 3:
                        m = m.sum(axis=2).astype(np.uint8)
                    else:
                        m = m.astype(np.uint8)
                    m_accum |= m
                merged_per_frame.append(m_accum)
            ret.append(np.stack(merged_per_frame, axis=0))  # [T,H,W]

        # Filter out empties gracefully; if none, create a dummy zero object
        ret = [x for x in ret if x.size != 0]
        if len(ret) == 0:
            # shape: [1, T, H, W] of zeros (will be trimmed by caller if needed)
            return None
        masks = np.stack(ret, axis=0).astype(np.uint8)  # [num_objects,T,H,W]
        return torch.from_numpy(masks)

    # ---------- Conversation ----------

    def _build_conversation(self, phrase: str):
        phrase = phrase.strip().rstrip(".").lower()
        q = random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=phrase)
        a = random.choice(self.ANSWER_LIST)
        source = [
            {"from": "human", "value": q},
            {"from": "gpt", "value": a},
        ]
        return self.conversation_generator.apply(source)

    # ---------- Frame sampling ----------

    def _sample_indices(self, n_total, n_sample, contiguous=False):
        if n_total <= 0:
            return [0] * n_sample
        if n_total <= n_sample:
            idxs = np.linspace(0, n_total - 1, n_sample, dtype=int).tolist()
            return idxs
        if contiguous:
            start = np.random.randint(0, n_total - n_sample + 1)
            return list(range(start, start + n_sample))
        # uniform without replacement
        idxs = sorted(np.random.choice(n_total, n_sample, replace=False).tolist())
        return idxs

    # ---------- Loader ----------

    def __getitem__(self, idx):
        meta = self.metas[self.index[idx]]

        vid = meta["video"]
        frame_ids = meta["frames"]
        T_total = len(frame_ids)
        assert T_total > 0, "Video has no frames."

        # sample frames for encoder/LLM path
        enc_idx = self._sample_indices(T_total, self.num_frames_for_clip, self.frame_contiguous_sample)
        # sample frames for SAM path
        if self.num_frames_for_sam == -1:
            sam_idx = enc_idx
        else:
            sam_idx = self._sample_indices(T_total, self.num_frames_for_sam, self.frame_contiguous_sample)

        # Load PIL images for encoder
        pil_images = []
        for fi in enc_idx:
            f = frame_ids[fi]
            p = os.path.join(self.image_folder, vid, f"{f}.jpg")
            img = Image.open(p).convert("RGB")
            pil_images.append(img)

        # Load numpy images for SAM (HWC, uint8)
        np_images_for_sam = []
        for fi in sam_idx:
            f = frame_ids[fi]
            p = os.path.join(self.image_folder, vid, f"{f}.jpg")
            img = Image.open(p).convert("RGB")
            np_images_for_sam.append(np.array(img))

        # phrase/caption from this meta (expression)
        phrase = meta["exp"]
        conversations = self._build_conversation(phrase)

        # Build masks aligned to enc_idx AND sam_idx (SAM branch uses sam_idx only)
        # Gather annotation frames corresponding to selected indices
        # video_masks structure: list[obj] -> list[anno] -> list[frame] -> rle/None
        video_masks_enc = []
        video_masks_sam = []

        # For each object: collect the per-anno list sliced by indices
        # meta["mask_anno_id"] can list several annotation ids for the same object; treat them as separate annos of a single object
        # Here we assume one semantic object described by meta, possibly multiple anno_ids merged.
        obj_annos = []
        for anno_id in meta["mask_anno_id"]:
            obj_annos.append(self.mask_dict[anno_id])

        # Slice per selected enc frames
        enc_annos_sliced = []
        for ann in obj_annos:  # ann: list over all frames
            enc_annos_sliced.append([ann[i] for i in enc_idx])
        video_masks_enc.append(enc_annos_sliced)  # single object

        # Slice per selected sam frames
        sam_annos_sliced = []
        for ann in obj_annos:
            sam_annos_sliced.append([ann[i] for i in sam_idx])
        video_masks_sam.append(sam_annos_sliced)  # single object

        # Decode masks for SAM branch to produce [num_objects, T, H, W]
        # Use the first SAM frame to determine H,W
        H, W = np_images_for_sam[0].shape[:2]
        masks_sam = self._decode_video_masks(video_masks_sam, (H, W))
        if masks_sam is None:
            # fallback empty mask if annotations are missing; keep shape [1,T,H,W]
            masks_sam = torch.zeros((1, len(sam_idx), H, W), dtype=torch.uint8)

        # SAM preprocessing (returns list of tensors & a single `resize` meta from the first frame)
        original_pil_for_sam = [Image.fromarray(img_rgb) for img_rgb in np_images_for_sam]
        preprocessed_for_sam_and_resize = [self.sam_preprocessor.preprocess(img) for img in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in preprocessed_for_sam_and_resize]
        resize = preprocessed_for_sam_and_resize[0][1] if len(preprocessed_for_sam_and_resize) > 0 else None

        # Encoder preprocessing (arbitrary output; pass through)
        enc_out = self.enc_preprocessor.preprocess(pil_images)
        # Expected keys: 'images', 'context_images'
        assert "images" in enc_out and "context_images" in enc_out, \
            "enc_preprocessor.preprocess must return dict with keys ['images','context_images']"

        # Label map (per your target format; single 2D map with ignore label)
        label = torch.ones(masks_sam.shape[-2], masks_sam.shape[-1]) * self.ignore_label  # [H,W]

        data_dict = {
            "file_path": None,
            "preprocessed_for_sam": preprocessed_for_sam,                  # list[length=T_sam]
            "images": enc_out["images"],                                   # encoder inputs
            "context_images": enc_out["context_images"],                   # encoder context
            "conversations": conversations,                                 # list of {from,value}
            "masks": masks_sam,                                             # [num_objects, T_sam, H, W], uint8 {0,1}
            "label": label,                                                 # [H,W], ignore label
            "resize": resize,                                               # resize/meta from SAM preprocess
            "questions": None,
            "sampled_classes": None,

            'original_pil_for_sam': original_pil_for_sam,  # list[PIL.Image] matching masks time dimension
        }

        return data_dict




