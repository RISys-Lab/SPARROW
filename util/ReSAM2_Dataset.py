import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
import pycocotools.mask as maskUtils

from torch.utils.data import Dataset

class VideoSAM2Dataset(Dataset):
    """
    Reads SAM2-style video annotations and returns a dict matching the
    ReferVOSDataset output format.

    Expected SAM2 JSON structure (expression_file):
    {
      "<video_id>": {
        "video_path": "relative/path/to/video.mp4",
        "anno_path":  "relative/path/to/masklet.json",
        "objects": {
           "<obj_id>": {
              "formated": "A person in red shirt",
              "short_caps": ["person", "red shirt person", ...]
           },
           ...
        }
      },
      ...
    }

    Expected masklet JSON structure (at anno_path):
    {
      "masklet": [
         <RLE of frame 0 for all objects>,
         <RLE of frame 1 for all objects>,
         ...
      ]
    }
    where maskUtils.decode(masklet[i]) -> H x W x N_obj (binary).
    """

    ignore_label = 255

    def __init__(
        self,
        sam2_folder: str,
        expression_file: str,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        mode: str = "long",                 # 'long' | 'short' | 'long_short'
        
        # select_number: int = 5,             # number of objects to sample per video
        select_number: int = 1,             # number of objects to sample per video


        num_frames_for_clip: int = 5,       # frames for encoder/context branch
        num_frames_for_sam: int = 1,        # frames for SAM branch (-1 = use all)
        frame_contiguous_sample: bool = False,
    ):
        assert mode in ["long", "short", "long_short"]
        self.sam2_folder = sam2_folder
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator

        self.mode = mode
        self.select_number = select_number
        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_for_sam = num_frames_for_sam
        self.frame_contiguous_sample = frame_contiguous_sample

        # Load expression metadata
        with open(expression_file, "r") as f:
            self.expr_db = json.load(f)
        self.video_ids = list(self.expr_db.keys())

        # Conversation templates
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
        return len(self.video_ids)

    # ---------- helpers ----------
    @staticmethod
    def _read_video_all_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)  # BGR
        cap.release()
        return frames

    @staticmethod
    def _decode_masklet(masklet):
        """masklet: list of RLEs; returns list of (H,W,N_obj) uint8 arrays."""
        masks = []
        for rle in masklet:
            m = maskUtils.decode(rle)  # (H, W, N_obj) uint8 {0,1}
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            masks.append(m)
        return masks  # len=T, each (H,W,N_obj)

    @staticmethod
    def _uniform_indices(T, K):
        if K <= 0:
            return np.arange(T, dtype=int)
        if K == 1:
            return np.array([T // 2], dtype=int)
        return np.linspace(0, max(T - 1, 0), K, dtype=int)

    @staticmethod
    def _sample_contiguous(T, K):
        if K >= T:
            return np.arange(T, dtype=int)
        start = np.random.randint(0, T - K + 1)
        return np.arange(start, start + K, dtype=int)
    

    @staticmethod
    def _align_frames_to_masklets(frames_bgr, masklets_len):
        """Make #video frames match #masklet frames.

        1) If an integer step (2/3/4/5/6/8/10) explains the mismatch, slice by that step.
        2) Otherwise, uniformly resample frames to exactly `masklets_len`.
        """
        T_v = len(frames_bgr)
        T_m = masklets_len
        if T_v == T_m:
            return frames_bgr

        # try common integer decimation factors (SAM2 often uses ::4)
        for f in (2, 3, 4, 5, 6, 8, 10):
            if T_v % f == 0 and (T_v // f) == T_m:
                return frames_bgr[::f]

        # fallback: uniform squeeze to T_m
        if T_m <= 1:
            return [frames_bgr[T_v // 2]]
        idx = np.linspace(0, T_v - 1, T_m, dtype=int)
        return [frames_bgr[i] for i in idx]
    


    def _pick_mode_and_phrase(self, obj_info):
        """Return a single phrase for the selected object according to mode."""
        if self.mode == "long":
            return obj_info.get("formated", "")
        elif self.mode == "short":
            caps = obj_info.get("short_caps", []) or [obj_info.get("formated", "")]
            return random.choice(caps).replace("A ", "")
        else:  # long_short
            if random.random() < 0.5:
                return obj_info.get("formated", "")
            caps = obj_info.get("short_caps", []) or [obj_info.get("formated", "")]
            return random.choice(caps).replace("A ", "")

    def _build_conversations(self, phrases):
        """
        phrases: list[str] (one per sampled object)
        We generate a single Q/A pair that references the video (same as ReferVOS).
        If multiple objects, we merge phrases with ', '.
        """
        phrase_text = ", ".join([p.lower() for p in phrases if p])
        q = random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=phrase_text)
        a = random.choice(self.ANSWER_LIST)
        source = [{'from': 'human', 'value': q},
                  {'from': 'gpt',   'value': a}]
        return self.conversation_generator.apply(source)

    # ---------- main ----------
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        meta = self.expr_db[vid]

        video_path = os.path.join(self.sam2_folder, meta["video_path"])
        anno_path  = os.path.join(self.sam2_folder, meta["anno_path"])

        # frames: list of HxWx3 (BGR, uint8)
        frames_bgr = self._read_video_all_frames(video_path)

        # Load masklet -> list of (H,W,N_obj)
        with open(anno_path, "r") as f:
            mask_db = json.load(f)
        masklets = self._decode_masklet(mask_db["masklet"])

        frames_bgr = self._align_frames_to_masklets(frames_bgr, len(masklets))

        T_total = len(masklets)
        assert T_total == len(frames_bgr), "Frames and masklet length mismatch."

        # ----- sample objects -----
        all_obj_ids = list(meta["objects"].keys())  # strings
        n_objects_total = len(all_obj_ids)
        if n_objects_total == 0:
            raise RuntimeError(f"No objects in annotation for video_id={vid}")

        if n_objects_total >= self.select_number:
            obj_indices = np.random.choice(n_objects_total, self.select_number, replace=False)
        else:
            obj_indices = np.random.choice(n_objects_total, self.select_number, replace=True)

        picked_obj_ids = [all_obj_ids[i] for i in obj_indices]
        picked_infos = [meta["objects"][oid] for oid in picked_obj_ids]
        phrases = [self._pick_mode_and_phrase(info) for info in picked_infos]

        # ----- sample frames for encoder branch (context/images) -----
        if self.frame_contiguous_sample and T_total > self.num_frames_for_clip and random.random() < 0.5:
            enc_idx = self._sample_contiguous(T_total, self.num_frames_for_clip)
        else:
            enc_idx = self._uniform_indices(T_total, self.num_frames_for_clip)
        enc_idx.sort()

        pil_images_for_enc = []
        for i in enc_idx:
            # convert BGR -> RGB for PIL
            img_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
            pil_images_for_enc.append(Image.fromarray(img_rgb))

        enc_out = self.enc_preprocessor.preprocess(pil_images_for_enc)  # must return {'images', 'context_images'}

        # ----- sample frames for SAM branch -----
        if self.num_frames_for_sam == -1:
            sam_idx = np.arange(T_total, dtype=int)
        else:
            if self.frame_contiguous_sample and T_total > self.num_frames_for_sam and random.random() < 0.5:
                sam_idx = self._sample_contiguous(T_total, self.num_frames_for_sam)
            else:
                sam_idx = self._uniform_indices(T_total, self.num_frames_for_sam)
        sam_idx.sort()

        np_images_for_sam = []
        for i in sam_idx:
            # keep as RGB np.uint8 for sam_preprocessor
            img_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
            np_images_for_sam.append(img_rgb)

        # preprocess for SAM (one by one, grab resize from the first)
        original_pil_for_sam = [Image.fromarray(img_rgb) for img_rgb in np_images_for_sam]
        pre_sam_and_resize = [self.sam_preprocessor.preprocess(im) for im in np_images_for_sam]
        preprocessed_for_sam = [x[0] for x in pre_sam_and_resize]
        resize = pre_sam_and_resize[0][1] if len(pre_sam_and_resize) > 0 else None

        # ----- build masks tensor: [num_objects, T_sam, H, W] -----
        # masklets[t] : (H, W, N_obj_total). We need only picked objects and only sam_idx frames.
        if len(sam_idx) == 0:
            raise RuntimeError("No frames selected for SAM.")
        H, W, _ = masklets[0].shape
        T_sam = len(sam_idx)
        num_objects = len(picked_obj_ids)

        masks = np.zeros((num_objects, T_sam, H, W), dtype=np.uint8)
        for t_i, t in enumerate(sam_idx):
            ml = masklets[t]  # (H, W, N_obj_total)
            for o_i, oid in enumerate(picked_obj_ids):
                # object id is a string index into 3rd dim order; we assume order matches keys sorted by insertion.
                # If your mask channels are aligned by integer id, convert accordingly.
                try:
                    ch = int(oid)
                except ValueError:
                    # If oid isn't an integer index, fall back to mapping by enumeration order of all_obj_ids
                    ch = all_obj_ids.index(oid)
                ch = np.clip(ch, 0, ml.shape[2]-1)
                masks[o_i, t_i] = ml[:, :, ch]

        masks = torch.from_numpy(masks)  # uint8 -> {0,1}

        # ----- label [H, W] with ignore -----
        label = torch.ones((H, W), dtype=torch.float32) * self.ignore_label

        # ----- conversations -----
        conversations = self._build_conversations(phrases)

        data_dict = {
            'file_path': os.path.join(self.sam2_folder, meta["video_path"]),
            'preprocessed_for_sam': preprocessed_for_sam,           # list[len=T_sam] of tensors/arrays (depends on preprocessor)
            'images': enc_out['images'],                            # model-encoder input(s)
            'context_images': enc_out.get('context_images', None),  # optional extra views
            'conversations': conversations,                         # list of {'from','value'} after generator.apply
            'masks': masks,                                         # [num_objects, T_sam, H, W] uint8
            'label': label,                                         # [H, W] filled with ignore_label
            'resize': resize,                                       # resize metadata from sam_preprocessor
            'questions': None,
            'sampled_classes': None,
        
            # NEW:
            'sam_indices': sam_idx.tolist(),               # list[int]
            'original_pil_for_sam': original_pil_for_sam,  # list[PIL.Image] matching masks time dimension,

        }

        # sanity checks mirroring ReferVOSDataset
        if self.num_frames_for_sam != -1:
            assert masks.shape[1] == self.num_frames_for_sam, \
                f"masks T ({masks.shape[1]}) != num_frames_for_sam ({self.num_frames_for_sam})"
            assert len(preprocessed_for_sam) == self.num_frames_for_sam, \
                f"preprocessed_for_sam T ({len(preprocessed_for_sam)}) != num_frames_for_sam ({self.num_frames_for_sam})"

        return data_dict


