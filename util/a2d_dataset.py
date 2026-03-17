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


# -------------------- helpers --------------------

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


# -------------------- Base dataset for A2D Sentences --------------------

class A2DSeg_BaseDataset(Dataset):
    """
    Traverses:
      root/
        Images/                 # optional; root can also *be* Images
          <sequence>/           # e.g., seq1
            img/
            mask/
            groundtruth.txt
            nlp.txt
          <sequence>/
            <scene>/            # e.g., 00, 01, ...
              img/
              mask/
              groundtruth.txt
              nlp.txt

    Returns:
      dataset_id, sample_id, pil_images, all_masks_dict, new_caption, phrases, boxes_xyxy_pixels
    """
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        # Allow pointing to either A2D_sentences/ or A2D_sentences/Images
        if (self.root / "Images").is_dir():
            self.images_root = self.root / "Images"
        else:
            self.images_root = self.root

        if not self.images_root.is_dir():
            raise FileNotFoundError(f"Images root not found: {self.images_root}")

        self.items = self._collect_items(self.images_root)
        if not self.items:
            raise RuntimeError(f"No valid samples under {self.images_root}")

    # ---- filesystem helpers ----

    @staticmethod
    def _is_sample_dir(d: Path) -> bool:
        return (d / "img").is_dir() and (d / "mask").is_dir() and (d / "groundtruth.txt").is_file() and (d / "nlp.txt").is_file()

    def _collect_items(self, images_root: Path):
        items = []
        # sequences (e.g., seq1, seq2, ...)
        for seq_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            if self._is_sample_dir(seq_dir):
                # sequence itself is a sample (no scene subdir)
                sample_id = f"{seq_dir.name}"
                items.append(("A2D", sample_id, seq_dir))
                continue

            # scenes (00/01/..)
            for scene_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
                if self._is_sample_dir(scene_dir):
                    sample_id = f"{seq_dir.name}/{scene_dir.name}"
                    items.append(("A2D", sample_id, scene_dir))
        return items

    def __len__(self):
        return len(self.items)

    # ---- IO + parsing (unchanged from your version) ----

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
        return [Image.open(p).convert("RGB") for p in frame_paths]

    @staticmethod
    def _mask_candidates(mask_dir: Path, frame_name: str, obj_id: str) -> List[Path]:
        stem = Path(frame_name).stem
        full = frame_name
        zz = str(obj_id).zfill(3)
        return [
            mask_dir / zz / f"{stem}.png",        # mask/001/00000001.png
            mask_dir / f"{stem}_{obj_id}.png",    # mask/00000001_1.png
            mask_dir / f"{stem}.png",             # mask/00000001.png
            mask_dir / zz / f"{full}.png",        # mask/001/00000001.jpg.png
            mask_dir / f"{full}_{obj_id}.png",    # mask/00000001.jpg_1.png
            mask_dir / f"{full}.png",             # mask/00000001.jpg.png
        ]

    @staticmethod
    def _load_mask(path: Path) -> Optional[np.ndarray]:
        if not path.is_file():
            return None
        m = Image.open(path).convert("L")
        a = np.array(m, dtype=np.uint8)
        return (a > 127).astype(np.uint8)

    @staticmethod
    def _read_groundtruth_xyxy(gt_path: Path) -> List[Tuple[float, float, float, float]]:
        out = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                parts = [p for p in re.split(r"[,\s]+", s) if p]
                if len(parts) < 4:
                    parts += ["0"] * (4 - len(parts))
                try:
                    x1, y1, x2, y2 = map(float, parts[:4])
                except ValueError:
                    x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                out.append((x1, y1, x2, y2))
        if not out:
            raise ValueError(f"Empty or invalid groundtruth at {gt_path}")
        return out

    def _build_masks(self, mask_dir: Path, frame_names: List[str], obj_ids_per_phrase: List[List[str]]) -> Dict[int, np.ndarray]:
        if not obj_ids_per_phrase:
            obj_ids_per_phrase = [["1"]]  # single-object fallback

        all_masks: Dict[int, np.ndarray] = {}
        for k, obj_ids in enumerate(obj_ids_per_phrase):
            obj_id = obj_ids[0]
            seq_masks = []
            for fn in frame_names:
                arr = None
                for cp in self._mask_candidates(mask_dir, fn, obj_id):
                    arr = self._load_mask(cp)
                    if arr is not None:
                        break
                if arr is None:
                    tried = [str(p) for p in self._mask_candidates(mask_dir, fn, obj_id)]
                    raise FileNotFoundError(
                        f"Mask not found for frame='{fn}' obj_id='{obj_id}' in {mask_dir}\nTried:\n- " + "\n- ".join(tried)
                    )
                seq_masks.append(arr)
            all_masks[k] = np.stack(seq_masks, axis=0).astype(bool)  # (T,H,W)
        return all_masks

    def __getitem__(self, idx):
        dataset_id, sample_id, sample_dir = self.items[idx]
        img_dir  = sample_dir / "img"
        mask_dir = sample_dir / "mask"
        nlp_path = sample_dir / "nlp.txt"
        gt_path  = sample_dir / "groundtruth.txt"

        # caption
        raw_caption = self._read_caption_file(nlp_path)
        obj_ids_list, phrases = get_phrase_and_obj_ids_from_caption(raw_caption)
        new_caption = add_seg_tokens(raw_caption)

        # frames
        frame_paths = self._sorted_frames(img_dir)
        frame_names = [p.name for p in frame_paths]
        pil_images  = self._load_pils(frame_paths)

        # boxes (already xyxy per your reader)
        gt_xyxy = self._read_groundtruth_xyxy(gt_path)
        if len(gt_xyxy) != len(pil_images):
            raise AssertionError(f"groundtruth length ({len(gt_xyxy)}) != #frames ({len(pil_images)}) at {sample_dir}")
        boxes_xyxy_pixels = gt_xyxy

        # masks
        all_masks = self._build_masks(mask_dir, frame_names, obj_ids_list)

        return dataset_id, sample_id, pil_images, all_masks, new_caption, phrases or ["object"] * len(all_masks), boxes_xyxy_pixels



# -------------------- Wrapper: normalized boxes + images4box, SAM, etc. --------------------

class A2DSeg_Dataset(Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir: str,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        num_frames_for_sam: int = 1,
    ):
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator

        self.base = A2DSeg_BaseDataset(root_dir=base_video_dataset_dir)
        print(f"Done loading {len(self.base)} A2D samples.")

        self.num_frames_for_clip = self.enc_preprocessor.num_frames
        self.num_frames_for_sam  = num_frames_for_sam

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
        if x2c <= x1c or y2c <= y1c:
            return 0.0, 0.0, float(W), float(H)  # fallback to full image
        return x1c, y1c, x2c, y2c

    @staticmethod
    def _normalize_xyxy(x1, y1, x2, y2, W, H):
        return (x1 / W, y1 / H, x2 / W, y2 / H)

    def __getitem__(self, idx):
        # base returns: dataset_id, sample_id, pil_images, all_masks_dict, caption, phrases, boxes_xyxy_pix
        _, _, all_pils, all_gt_masks_dict, caption, _, boxes_xyxy_pix = self.base[idx]
        T_total = len(all_pils)

        # --------- CLIP frames + aligned boxes ----------
        clip_idx = self._clip_indices(T_total, self.num_frames_for_clip)
        pil_images_for_clip = [all_pils[i] for i in clip_idx]
        boxes_for_clip_pix  = [boxes_xyxy_pix[i] for i in clip_idx]

        preprocessed_for_clip = self.enc_preprocessor.preprocess(pil_images_for_clip)

        # --------- images4box (crop -> enc_preprocessor) ----------
        crop_pils = []
        boxes_norm = []
        for im, (x1, y1, x2, y2) in zip(pil_images_for_clip, boxes_for_clip_pix):
            W, H = im.size
            x1c, y1c, x2c, y2c = self._clamp_xyxy_to_image(x1, y1, x2, y2, W, H)
            nx1, ny1, nx2, ny2 = self._normalize_xyxy(x1c, y1c, x2c, y2c, W, H)
            boxes_norm.append([nx1, ny1, nx2, ny2])
            crop = im.crop((int(round(x1c)), int(round(y1c)), int(round(x2c)), int(round(y2c))))
            crop_pils.append(crop)

        preprocessed_crops = self.enc_preprocessor.preprocess(crop_pils)  # noqa: F841 (kept in case you wire images4box later)
        boxes_norm = torch.tensor(boxes_norm, dtype=torch.float32)  # (T_clip, 4)

        # --------- SAM frames + masks ----------
        all_gt_masks = [np.array(m) for m in all_gt_masks_dict.values()]  # num_objs x (T,H,W)

        pil_images_for_sam = all_pils.copy()
        if self.num_frames_for_sam == 1:
            pil_images_for_sam = [pil_images_for_sam[0]]
            gt_masks = [m[0] for m in all_gt_masks]
            gt_masks = [np.expand_dims(m, axis=0) for m in gt_masks]
        else:
            sam_idx = self._clip_indices(len(pil_images_for_sam), self.num_frames_for_sam)
            pil_images_for_sam = [pil_images_for_sam[i] for i in sam_idx]
            gt_masks = [m[sam_idx] for m in all_gt_masks]

        np_images_for_sam = [np.array(im) for im in pil_images_for_sam]
        pre_sam_and_resize = [self.sam_preprocessor.preprocess(img) for img in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in pre_sam_and_resize]
        resize = pre_sam_and_resize[0][1]

        conversations = self._make_conversations(caption)

        gt_masks_t = [torch.from_numpy(m) for m in gt_masks]
        gt_masks_t = torch.stack(gt_masks_t)                   # (num_objs, T_sam, H, W)
        masks = gt_masks_t.to(torch.bool)

        assert masks.shape[1] == self.num_frames_for_sam
        assert len(preprocessed_for_sam) == self.num_frames_for_sam, \
            f"len(preprocessed_for_sam):{len(preprocessed_for_sam)} != self.num_frames_for_sam:{self.num_frames_for_sam}"

        label = torch.full((masks.shape[2], masks.shape[3]), fill_value=self.ignore_label, dtype=torch.float32)

        data_dict = {
            'file_path': '',
            'preprocessed_for_sam': preprocessed_for_sam,
            'images': preprocessed_for_clip['images'],
            'context_images': preprocessed_for_clip['context_images'],
            'conversations': conversations,
            'masks': masks,
            'label': label,
            'resize': resize,
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict
    
