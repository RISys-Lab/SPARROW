import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# -------------------- helpers (same semantics as yours) --------------------

def subsample_images(images, t):
    if isinstance(images, list):
        n = len(images)
        if t < n:
            idx = np.linspace(0, n - 1, num=t, dtype=int)
            return [images[i] for i in idx]
        return images
    elif isinstance(images, np.ndarray):
        T = images.shape[0]
        if t < T:
            idx = np.linspace(0, T - 1, num=t, dtype=int)
            return images[idx]
        return images
    else:
        raise ValueError("Input images must be either a list of PIL images or a numpy array.")

_CAPT_PATTERN = r"\[([^\]]+)\]\(([^)]+)\)"  # [phrase](1, 2, 7)

def get_phrase_and_obj_ids_from_caption(caption: str):
    matches = re.findall(_CAPT_PATTERN, caption)
    results = [{"phrase": ph, "object_ids": ids.split(", ")} for ph, ids in matches]
    list_of_obj_ids, phrases = [], []
    for r in results:
        list_of_obj_ids.append(r['object_ids'])
        phrases.append(r['phrase'])
    return list_of_obj_ids, phrases

def add_seg_tokens(text: str) -> str:
    return re.sub(_CAPT_PATTERN, r"<p> \1 </p> [SEG]", text)


# -------------------- GOT-10k-style base dataset --------------------

class GOT10kSeg_BaseDataset(Dataset):
    """
    Expects:

      root/
        train/ or val/
          seq1/
            00/
              img/                (frames: e.g., 00000001.jpg, ...)
              mask/               (PNG masks; per-object or flat)
              groundtruth.txt
              nlp.txt
            01/ ...
          seq2/ ...

    Mask lookup per frame name F='00000001.jpg' and obj_id='1':
      stem = '00000001'
      tries (in order):
        1) mask/001/00000001.png
        2) mask/00000001_1.png
        3) mask/00000001.png        # single-object fallback
    """
    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split = split

        split_dir = self.root_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        items = []
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for scene_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
                if not (scene_dir / "img").is_dir(): continue
                if not (scene_dir / "mask").is_dir(): continue
                if not (scene_dir / "groundtruth.txt").is_file(): continue
                if not (scene_dir / "nlp.txt").is_file(): continue
                items.append((seq_dir.name, scene_dir.name, scene_dir))
        if not items:
            raise RuntimeError(f"No valid sequences found under {split_dir}")

        self.items = items

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read_caption_file(nlp_path: Path) -> str:
        with open(nlp_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            raise ValueError(f"Empty nlp.txt at {nlp_path}")
        return " ".join(lines)

    @staticmethod
    def _sorted_frames(img_dir: Path) -> List[Path]:
        frames = sorted((p for p in img_dir.iterdir() if p.is_file()), key=lambda p: p.name)
        if not frames:
            raise ValueError(f"No frames found in {img_dir}")
        return frames

    @staticmethod
    def _load_pils(frame_paths: List[Path]) -> List[Image.Image]:
        out = []
        for p in frame_paths:
            out.append(Image.open(p).convert("RGB"))
        return out

    @staticmethod
    def _mask_candidates(mask_dir: Path, frame_name: str, obj_id: str) -> List[Path]:
        """
        Frames may be .jpg; masks are .png. Build candidates from the frame stem.
        Also support per-object folder with zero-padded id.
        """
        stem = Path(frame_name).stem          # '00000001'
        full = frame_name                     # '00000001.jpg' (rare variants)
        zz = str(obj_id).zfill(3)

        candidates = [
            mask_dir / zz / f"{stem}.png",        # mask/001/00000001.png
            mask_dir / f"{stem}_{obj_id}.png",    # mask/00000001_1.png
            mask_dir / f"{stem}.png",             # mask/00000001.png
            # some datasets keep weird variants; cheap fallbacks:
            mask_dir / zz / f"{full}.png",        # mask/001/00000001.jpg.png
            mask_dir / f"{full}_{obj_id}.png",    # mask/00000001.jpg_1.png
            mask_dir / f"{full}.png",             # mask/00000001.jpg.png
        ]
        return candidates

    @staticmethod
    def _load_mask(path: Path) -> Optional[np.ndarray]:
        if not path.is_file():
            return None
        m = Image.open(path).convert("L")
        a = np.array(m, dtype=np.uint8)
        return (a > 127).astype(np.uint8)  # binarize defensively
    


    @staticmethod
    def _read_groundtruth_xywh(gt_path: Path) -> List[Tuple[float, float, float, float]]:
        """
        Parses lines like: 'x,y,w,h' or 'x y w h'. Ignores blanks.
        Returns list of floats per line. Length should equal #frames.
        """
        out = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                # split on comma or whitespace
                parts = [p for p in re.split(r"[,\s]+", s) if p]
                if len(parts) < 4:
                    # tolerate bad lines; pad with zeros
                    parts = parts + ["0"] * (4 - len(parts))
                try:
                    x, y, w, h = map(float, parts[:4])
                except ValueError:
                    # if NaN or junk, push zeros
                    x, y, w, h = 0.0, 0.0, 0.0, 0.0
                out.append((x, y, w, h))
        if not out:
            raise ValueError(f"Empty or invalid groundtruth at {gt_path}")
        return out
    

    @staticmethod
    def _xywh_to_xyxy_pixels(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return (x, y, x + w, y + h)


    def _build_masks(
        self, mask_dir: Path, frame_names: List[str], obj_ids_per_phrase: List[List[str]]
    ) -> Dict[int, np.ndarray]:
        # default single-object if caption lacks markup
        if not obj_ids_per_phrase:
            obj_ids_per_phrase = [["1"]]

        all_masks: Dict[int, np.ndarray] = {}
        T = len(frame_names)

        for k, obj_ids in enumerate(obj_ids_per_phrase):
            obj_id = obj_ids[0]  # align with your prior behavior
            seq_masks = []
            for fn in frame_names:
                arr = None
                for cp in self._mask_candidates(mask_dir, fn, obj_id):
                    arr = self._load_mask(cp)
                    if arr is not None:
                        break
                if arr is None:
                    # Helpful debug:
                    tried = [str(p) for p in self._mask_candidates(mask_dir, fn, obj_id)]
                    raise FileNotFoundError(
                        f"Mask not found for frame='{fn}' obj_id='{obj_id}' in {mask_dir}\nTried:\n- " + "\n- ".join(tried)
                    )
                seq_masks.append(arr)
            all_masks[k] = np.stack(seq_masks, axis=0).astype(bool)  # (T,H,W) bool
        return all_masks

    def __getitem__(self, idx):
        seq_id, scene_id, scene_dir = self.items[idx]
        img_dir  = scene_dir / "img"
        mask_dir = scene_dir / "mask"
        nlp_path = scene_dir / "nlp.txt"
        gt_path  = scene_dir / "groundtruth.txt"

        # caption + phrases + tokens
        raw_caption = self._read_caption_file(nlp_path)
        obj_ids_list, phrases = get_phrase_and_obj_ids_from_caption(raw_caption)
        new_caption = add_seg_tokens(raw_caption)

        # frames
        frame_paths = self._sorted_frames(img_dir)
        frame_names = [p.name for p in frame_paths]
        pil_images  = self._load_pils(frame_paths)


        # boxes (per-frame) from groundtruth (xywh -> xyxy pixels)
        gt_xywh = self._read_groundtruth_xywh(gt_path)
        if len(gt_xywh) != len(pil_images):
            raise AssertionError(
                f"groundtruth length ({len(gt_xywh)}) != #frames ({len(pil_images)}) at {scene_dir}"
            )
        boxes_xyxy_pixels = [self._xywh_to_xyxy_pixels(*b) for b in gt_xywh]  # length T



        # masks (num_objs x T x H x W) bool
        all_masks = self._build_masks(mask_dir, frame_names, obj_ids_list)

        # checks
        num_objs = len(all_masks)

        return seq_id, scene_id, pil_images, all_masks, new_caption, phrases or ["object"] * num_objs, boxes_xyxy_pixels



class GOT10kSeg_Dataset(Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir: str,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set: str = "train",
        num_frames_for_sam: int = 1,
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator

        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN

        assert image_set in ["train", "val", "test"], f"invalid image_set:{image_set}"
        self.base = GOT10kSeg_BaseDataset(root_dir=base_video_dataset_dir, split=image_set)
        print(f"Done loading {len(self.base)} samples from GOT-10k style dataset.")

        self.num_frames_for_clip = self.enc_preprocessor.num_frames
        self.num_frames_for_sam  = num_frames_for_sam


    # Conversation templates
    # =========================================================================================================================================
        
        self.DEFAULT_VIDEO_TOKEN = getattr(self.conversation_generator, "DEFAULT_VIDEO_TOKEN", "<video>")
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
    
    # =========================================================================================================================================

    def __len__(self):
        return len(self.base)

    def _make_conversations(self, phrases: str):

        phrase_text = phrases.replace('.', '').lower()

        q = random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=phrase_text)
        a = random.choice(self.ANSWER_LIST)

        source = [{'from': 'human', 'value': q},
                  {'from': 'gpt',   'value': a}]

        return self.conversation_generator.apply(source)
    
    @staticmethod
    def _clip_indices(T: int, target: int) -> np.ndarray:
        if target >= T:
            return np.arange(T, dtype=int)
        return np.linspace(0, T - 1, num=target, dtype=int)


    @staticmethod
    def _clamp_xyxy_to_image(x1, y1, x2, y2, W, H):
        x1c = max(0.0, min(float(W), float(x1)))
        y1c = max(0.0, min(float(H), float(y1)))
        x2c = max(0.0, min(float(W), float(x2)))
        y2c = max(0.0, min(float(H), float(y2)))
        # ensure positive area; if degenerate, fallback to full image
        if x2c <= x1c or y2c <= y1c:
            return 0.0, 0.0, float(W), float(H)
        return x1c, y1c, x2c, y2c

    @staticmethod
    def _normalize_xyxy(x1, y1, x2, y2, W, H):
        # divide by width/height to [0,1]
        return (x1 / W, y1 / H, x2 / W, y2 / H)

    def __getitem__(self, idx):
        # base now returns boxes_xyxy_pixels (per-frame)
        _, _, all_pils, all_gt_masks_dict, caption, _, boxes_xyxy_pix = self.base[idx]
        T_total = len(all_pils)

        # --------- Build encoder (CLIP) frames and aligned boxes ----------
        clip_idx = self._clip_indices(T_total, self.num_frames_for_clip)
        pil_images_for_clip = [all_pils[i] for i in clip_idx]
        boxes_for_clip_pix  = [boxes_xyxy_pix[i] for i in clip_idx]

        # encoder preprocess (CLIP or whatever you pass)
        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images_for_clip)

        # --------- Build images4box (cropped PILs → enc_preprocessor) -----
        crop_pils = []
        boxes_norm = []
        for im, (x1, y1, x2, y2) in zip(pil_images_for_clip, boxes_for_clip_pix):
            W, H = im.size
            # clamp to image
            x1c, y1c, x2c, y2c = self._clamp_xyxy_to_image(x1, y1, x2, y2, W, H)
            # normalized xyxy
            nx1, ny1, nx2, ny2 = self._normalize_xyxy(x1c, y1c, x2c, y2c, W, H)
            boxes_norm.append([nx1, ny1, nx2, ny2])

            # crop in pixel space (ints)
            crop = im.crop((int(round(x1c)), int(round(y1c)), int(round(x2c)), int(round(y2c))))
            crop_pils.append(crop)

        preprocessed_crops = self.enc_preprocessor.preprocess(crop_pils)  # same interface
        boxes_norm = torch.tensor(boxes_norm, dtype=torch.float32)        # (T_clip, 4)

        # --------- Build SAM views & masks (unchanged) --------------------
        all_gt_masks = [np.array(m) for m in all_gt_masks_dict.values()]  # num_objs x (T,H,W)

        pil_images_for_sam = all_pils.copy()
        if self.num_frames_for_sam == 1:
            pil_images_for_sam = [pil_images_for_sam[0]]  # one frame to SAM
            gt_masks = [m[0] for m in all_gt_masks]       # num_objs x (H,W)
            gt_masks = [np.expand_dims(m, axis=0) for m in gt_masks]  # num_objs x (1,H,W)
        else:
            sam_idx = self._clip_indices(len(pil_images_for_sam), self.num_frames_for_sam)
            pil_images_for_sam = [pil_images_for_sam[i] for i in sam_idx]
            gt_masks = [m[sam_idx] for m in all_gt_masks]             # num_objs x (T_sam,H,W)

        # SAM preprocess (per-frame)
        np_images_for_sam = [np.array(im) for im in pil_images_for_sam]
        pre_sam_and_resize = [self.sam_preprocessor.preprocess(img) for img in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in pre_sam_and_resize]
        resize = pre_sam_and_resize[0][1]

        # conversations
        conversations = self._make_conversations(caption)

        # masks tensor (bool)
        gt_masks_t = [torch.from_numpy(m) for m in gt_masks]   # each: (T_sam,H,W)
        gt_masks_t = torch.stack(gt_masks_t)                   # (num_objs,T_sam,H,W)
        masks = gt_masks_t.to(torch.bool)

        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam, \
            f"len(preprocessed_for_sam):{len(preprocessed_for_sam)} != self.num_frames_for_sam:{self.num_frames_for_sam}"

        # ignore-label map (H,W)
        label = torch.full((masks.shape[2], masks.shape[3]), fill_value=self.ignore_label, dtype=torch.float32)

        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': preprocessed_for_clip['images'],
            'context_images': preprocessed_for_clip['context_images'],
            'conversations': conversations,
            'masks': masks,                  # bool [num_objects, T_sam, H, W]
            'label': label,                  # float [H, W]
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
