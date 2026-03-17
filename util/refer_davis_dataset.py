import os
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image



class ReferDAVISDataset(Dataset):
    """
    DAVIS-2017 (Refer-DAVIS) dataset specialized loader that produces the exact
    interface expected by the previous ReferVOS pipeline (enc/sam preprocess, conversations, masks).
    """
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir: str,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set: str = "train",          # "train" or "valid"
        num_frames_for_clip: int = 5,      # frames fed to encoder side
        num_frames_for_sam: int = 1,       # frames fed to SAM side (-1 to use all)
        use_masks: bool = True,            # set False if you don't have Annotations
    ):
        super().__init__()

        assert image_set in ("train", "valid"), f"invalid image_set: {image_set}"
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        self.image_set = image_set
        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_for_sam = num_frames_for_sam
        self.use_masks = use_masks

        # Paths (matches your structure)
        self.davis_root = os.path.join(base_video_dataset_dir, "processed/refer_davis/2017")
        meta_file = os.path.join(self.davis_root, "meta_expressions", image_set, "meta_expressions.json")
        img_root = os.path.join(self.davis_root, image_set, "JPEGImages")
        ann_root = os.path.join(self.davis_root, image_set, "Annotations")

        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"JPEGImages folder not found: {img_root}")
        if self.use_masks and not os.path.isdir(ann_root):
            # Fallbacks commonly seen in DAVIS layouts
            alt = os.path.join(self.davis_root, "Annotations")
            if os.path.isdir(alt):
                ann_root = alt
            else:
                raise FileNotFoundError(f"Annotations folder not found: {ann_root}")

        self.img_root = img_root
        self.ann_root = ann_root

        with open(meta_file, "r") as f:
            meta = json.load(f)
        videos = meta["videos"]

        # Flatten (video, expression_id) pairs into items
        self.items = []
        for vid, vdata in videos.items():
            frames = vdata["frames"]                # list[str] without extension
            expressions = vdata["expressions"]      # dict[str -> {exp: str, obj_id: int?}]
            for expr_id, expr in expressions.items():
                self.items.append({
                    "video": vid,
                    "frames": frames,
                    "expression_id": expr_id,
                    "expression": expr.get("exp", ""),
                    "obj_id": expr.get("obj_id", None),  # used to pick the right instance mask
                })

        # Simple Q/A templates
        self.DEFAULT_VIDEO_TOKEN = getattr(self.conversation_generator, "DEFAULT_VIDEO_TOKEN", "<video>")
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you segment {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "What is {phrase} in this video? Please respond with segmentation masks.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatial segmentation of {phrase}.",
        ]
        self.ANSWER_LIST = [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
        ]

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, n_total: int, n_want: int):
        if n_want == -1 or n_want >= n_total:
            return list(range(n_total))
        # Uniform sampling across the sequence
        return list(np.linspace(0, n_total - 1, n_want, dtype=int))

    def _load_image(self, video: str, frame_id: str):
        # Supports common DAVIS layouts (with or without 480p level)
        cand = [
            os.path.join(self.img_root, video, frame_id + ".jpg"),
            os.path.join(self.img_root, "480p", video, frame_id + ".jpg"),
        ]
        for p in cand:
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No image found for {video}/{frame_id} under {self.img_root}")

    def _load_mask(self, video: str, frame_id: str):
        cand = [
            os.path.join(self.ann_root, video, frame_id + ".png"),
            os.path.join(self.ann_root, "480p", video, frame_id + ".png"),
        ]
        for p in cand:
            if os.path.exists(p):
                # DAVIS masks are 8-bit indexed; 0 = background, positive integers = instance ids
                return np.array(Image.open(p), dtype=np.uint8)
        raise FileNotFoundError(f"No annotation found for {video}/{frame_id} under {self.ann_root}")

    def _gen_conversation(self, caption: str):
        source = []
        q = random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())
        a = random.choice(self.ANSWER_LIST)
        source.append({"from": "human", "value": q})
        source.append({"from": "gpt", "value": a})
        return self.conversation_generator.apply(source)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        video = item["video"]
        frames = item["frames"]
        expr_text = item["expression"]
        obj_id = item["obj_id"]  # may be None if meta file lacks it

        # 1) Load all frames as PIL
        pil_images_all = [self._load_image(video, f) for f in frames]

        # 2) Choose encoder clip frames
        enc_indices = self._sample_indices(len(pil_images_all), self.num_frames_for_clip)
        pil_images_enc = [pil_images_all[i] for i in enc_indices]

        # 3) Encoder preprocess (expects a dict with 'images' and 'context_images')
        enc_out = self.enc_preprocessor.preprocess(pil_images_enc)

        # 4) Choose SAM frames (can be the same or different stride)
        sam_indices = self._sample_indices(len(pil_images_all), self.num_frames_for_sam)
        pil_images_sam = [pil_images_all[i] for i in sam_indices]

        # 5) SAM preprocess
        sam_pre = [self.sam_preprocessor.preprocess(np.array(im)) for im in pil_images_sam]
        preprocessed_for_sam = [x[0] for x in sam_pre]
        resize = sam_pre[0][1] if len(sam_pre) > 0 else None

        # 6) Build masks tensor aligned with SAM frames (num_objects=1 here)
        if self.use_masks:
            masks_np = []
            for i in sam_indices:
                mask_raw = self._load_mask(video, frames[i])  # [H,W] with instance ids
                if obj_id is not None:
                    bin_mask = (mask_raw.astype(np.int32) == int(obj_id)).astype(np.uint8)
                else:
                    # Fallback: if obj_id is unknown, collapse all non-zero as foreground
                    bin_mask = (mask_raw > 0).astype(np.uint8)
                masks_np.append(bin_mask)
            # [T,H,W] -> [num_objects(=1), T, H, W]
            masks_t = torch.from_numpy(np.stack(masks_np, axis=0))  # [T,H,W]
            masks = masks_t.unsqueeze(0)  # [1,T,H,W]
            label = torch.ones(masks.shape[-2], masks.shape[-1]) * self.ignore_label  # [H,W]
        else:
            masks = None
            label = None

        # 7) Conversation
        conversations = self._gen_conversation(expr_text)

        # 8) Also return original images as numpy (THWC) if you need parity with older code
        np_images = [np.array(im) for im in pil_images_enc]  # encoder clip images only (not full seq)

        data_dict = {
            'file_path': (video, item["expression_id"]),
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': enc_out['images'],
            'context_images': enc_out.get('context_images', None),
            'conversations': conversations,
            'masks': masks,             # [1, T_sam, H, W] or None
            'label': label,             # [H, W] or None
            'resize': resize,
            'questions': None,
            'sampled_classes': None,

            # Extra useful returns (optional):
            'caption': expr_text,
            'encoder_indices': enc_indices,
            'sam_indices': sam_indices,
            'np_images': np_images,     # THWC for encoder clip
        }
        return data_dict
