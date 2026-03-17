import json
import os
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PIL import Image

def subsample_images(images, t):
    if isinstance(images, list):
        num_images = len(images)
        if t < num_images:
            indices = np.linspace(0, num_images - 1, num=t, dtype=int)
            return [images[i] for i in indices]
        else:
            return images
    elif isinstance(images, np.ndarray):
        T = images.shape[0]
        if t < T:
            indices = np.linspace(0, T - 1, num=t, dtype=int)
            return images[indices]
        else:
            return images
    else:
        raise ValueError("Input images must be either a list of PIL images or a numpy array.")

##################################################

import json
from torch.utils.data import Dataset
import time
import ffmpeg
import numpy as np
import random

from util.grounding_utils.image_transforms import make_video_transforms, prepare


import os, json, time, random
import ffmpeg
import numpy as np
import torch
import cv2
from PIL import Image

# Expect these utilities to exist in your codebase (same as in VidSTGDataset)
# - subsample_images(seq, k)
# - sam_preprocessor.preprocess(image) -> (tensor, resize_info)
# - enc_preprocessor.preprocess(pil_images) -> {"images": ..., "context_images": ...}
# - conversation_generator.apply(source_conversations)
from util.grounding_utils.image_transforms import prepare  # only for bbox normalization if needed

import os, json, random
import ffmpeg
import numpy as np
import torch
from PIL import Image

# Assumptions: you already have these in your codebase (same as VidSTGDataset)
# - subsample_images(seq, k)
# - enc_preprocessor.preprocess(pil_images) -> {"images": ..., "context_images": ...}
# - sam_preprocessor.preprocess(image_np_hwc) -> (tensor, resize_info)
# - conversation_generator.apply(source_conversations) and DEFAULT_VIDEO_TOKEN attr

class VidSTGGroundingDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_video_dataset_dir,
        ann_file,
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set='test',
        # frame sampling / train-time behavior (mirrors VideoModulatedSTGrounding)
        video_max_len=200,
        video_max_len_train=100,
        fps=5,
        tmp_crop=False,
        tmp_loc=True,
        # VidSTG-style knobs
        num_frames_for_sam=-1,
    ):
        self.base_video_dataset_dir = base_video_dataset_dir
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)

        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        self.image_set = image_set

        self.video_max_len = video_max_len
        self.video_max_len_train = video_max_len_train
        self.fps = fps
        self.tmp_crop = tmp_crop
        self.tmp_loc = tmp_loc
        self.num_frames_for_sam = num_frames_for_sam
        self.num_frames_for_enc = enc_preprocessor.num_frames

        # fps downsample + cap at video_max_len, keep annotated interval membership
        self.vid2imgids = {}
        for video in self.annotations["videos"]:
            video_fps = video["fps"]
            sampling_rate = self.fps / video_fps
            assert sampling_rate <= 1.0, "fps must be <= video fps"

            start_frame = video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
            end_frame   = video["end_frame"]   if self.tmp_loc else video["tube_end_frame"]

            frame_ids = [start_frame]
            for fid in range(start_frame, end_frame):
                if int(frame_ids[-1] * sampling_rate) < int(fid * sampling_rate):
                    frame_ids.append(fid)

            if len(frame_ids) > self.video_max_len:
                frame_ids = [frame_ids[(j * len(frame_ids)) // self.video_max_len]
                             for j in range(self.video_max_len)]

            inter_frames = set(
                fid for fid in frame_ids
                if video["tube_start_frame"] <= fid < video["tube_end_frame"]
            )
            self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]

        # Conversation templates (parity with VidSTGDataset)
        self.DEFAULT_VIDEO_TOKEN = self.conversation_generator.DEFAULT_VIDEO_TOKEN
        self.QUESTION_LIST_FOR_INTERROGATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you spatiotemporally locate {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please spatiotemporally locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "{phrase} Please respond with a segmentation masks and time interval.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "In the video, {phrase} Please include spatial locations and time duration in your answer.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatiotemporal segmentation of {phrase}",
        ]
        self.QUESTION_LIST_FOR_DECLARATIVE = [
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Can you spatiotemporally locate {phrase} in this video?",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Please spatiotemporally locate {phrase} in this video.",
            self.DEFAULT_VIDEO_TOKEN + "\n" + "Perform spatiotemporal segmentation of {phrase}",
        ]
        self.ANSWER_LIST = [
            "It is [SEG] in frames:({t_start},{t_end}).",
            "Sure, [SEG] frames:({t_start},{t_end}).",
            "Sure, it is [SEG] within frames:({t_start},{t_end}).",
            "Sure, the localization result is [SEG] in frames:({t_start},{t_end}).",
            "[SEG] frames:({t_start},{t_end}).",
        ]

    def __len__(self):
        return len(self.annotations["videos"])

    # ---------- helpers ----------
    def _generate_conversation(self, caption, qtype, t_start, t_end):
        conversations = []
        if qtype == "interrogative":
            prompt = random.choice(self.QUESTION_LIST_FOR_INTERROGATIVE).format(phrase=caption.lower())
        elif qtype == "declarative":
            prompt = random.choice(self.QUESTION_LIST_FOR_DECLARATIVE).format(phrase=caption.lower())
        else:
            raise Exception("Unsupported qtype")
        conversations.append({"from": "human", "value": prompt})
        conversations.append({"from": "gpt", "value": random.choice(self.ANSWER_LIST).format(t_start=t_start, t_end=t_end)})
        return self.conversation_generator.apply(conversations)

    @staticmethod
    def _ltwh_to_xyxy(box):
        x, y, w, h = box
        return [x, y, x + w, y + h]

    @staticmethod
    def _rasterize_box_to_mask(box_xyxy, h, w):
        """Return binary mask (H,W) with 1s inside the box, else 0."""
        if box_xyxy is None:
            return None
        x1, y1, x2, y2 = box_xyxy
        x1 = int(max(0, min(w, x1)))
        y1 = int(max(0, min(h, y1)))
        x2 = int(max(0, min(w, x2)))
        y2 = int(max(0, min(h, y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        return m

    def _maybe_temporal_crop(self, frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end):
        if not self.tmp_crop:
            return frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end

        if random.random() <= 0.5:
            return frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end

        # start index choices
        if inter_idx:
            starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
        else:
            starts_list = list(range(len(frame_ids)))
        new_start = random.choice(starts_list) if starts_list else 0

        # end index choices
        if inter_idx:
            ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
        else:
            ends_list = [i for i in range(len(frame_ids)) if i > new_start]
        new_end = random.choice(ends_list) if ends_list else len(frame_ids) - 1

        prev_start_frame = frame_ids[0]
        prev_end_frame   = frame_ids[-1]

        keep = [i for i in range(len(frame_ids)) if new_start <= i <= new_end]
        frame_ids = [frame_ids[i] for i in keep]
        frames_thwc = frames_thwc[keep]
        masks = [masks[i] for i in keep]
        clip_start += frame_ids[0] - prev_start_frame
        clip_end   += frame_ids[-1] - prev_end_frame
        if inter_idx:
            inter_idx = [x - new_start for x in inter_idx if new_start <= x <= new_end]

        return frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end

    def _maybe_dense_trim_train(self, frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end):
        if self.video_max_len_train and len(frame_ids) > self.video_max_len_train:
            if inter_idx:
                starts_list = [i for i in range(len(frame_ids))
                               if inter_idx[0] - self.video_max_len_train < i <= inter_idx[-1]]
            else:
                starts_list = list(range(len(frame_ids)))
            new_start = random.choice(starts_list) if starts_list else 0
            new_end   = min(new_start + self.video_max_len_train - 1, len(frame_ids) - 1)

            prev_start_frame = frame_ids[0]
            prev_end_frame   = frame_ids[-1]

            keep = [i for i in range(len(frame_ids)) if new_start <= i <= new_end]
            frame_ids = [frame_ids[i] for i in keep]
            frames_thwc = frames_thwc[keep]
            masks = [masks[i] for i in keep]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end   += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [x - new_start for x in inter_idx if new_start <= x <= new_end]

        return frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end

    # ---------- main ----------
    def get_from_idx(self, idx):
        video = self.annotations["videos"][idx]
        video_id = video["video_id"]
        video_original_id = video["original_video_id"]
        caption = video["caption"]
        qtype   = video["qtype"]

        # temporal range
        clip_start = video["start_frame"] if self.tmp_loc else video["tube_start_frame"]
        clip_end   = video["end_frame"]   if self.tmp_loc else video["tube_end_frame"]

        frame_ids, inter_frames = self.vid2imgids[video_id]
        traj = self.annotations["trajectories"][video_original_id][str(video["target_id"])]

        # decode frames [clip_start, clip_end)
        vid_path = os.path.join(self.base_video_dataset_dir, video["video_path"])
        video_fps = video["fps"]
        ss = clip_start / video_fps
        t  = (clip_end - clip_start) / video_fps
        w, h = video["width"], video["height"]

        """
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(capture_stdout=True, quiet=True)
        frames_thwc = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])  # T,H,W,3
        assert len(frames_thwc) == len(frame_ids), f"decoded {len(frames_thwc)} vs ids {len(frame_ids)}"
        """

        # ========================================================================================================================
        # decode exact frames [clip_start, clip_end) by frame index
        cmd = (
            ffmpeg
            .input(vid_path)
            # either form is fine; I prefer gte/lt to make the half-open interval explicit
            .filter('select', f'gte(n,{clip_start})*lt(n,{clip_end})')
            # .filter('select', f'between(n,{clip_start},{clip_end-1})')  # alternative

            # do NOT let ffmpeg duplicate/drop frames to hit a target fps
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vsync='0')
            .global_args('-loglevel', 'error')  # keep stderr useful
        )

        out, _ = cmd.run(capture_stdout=True, capture_stderr=True)

        # now these frames are EXACTLY the original indices in [clip_start, clip_end)
        frames_full = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        expected = clip_end - clip_start
        assert frames_full.shape[0] == expected, (frames_full.shape[0], expected)

        # map your sampled frame_ids -> relative indices
        rel_idx = [fid - clip_start for fid in frame_ids]
        frames_thwc = frames_full[rel_idx]
        # ========================================================================================================================


        # build masks per frame from boxes
        masks = []
        inter_idx = []
        for i_img, fid in enumerate(frame_ids):
            if fid in inter_frames and str(fid) in traj:
                # ann is either dict with 'bbox' or [l,t,w,h]
                ann = traj[str(fid)]
                bbox_ltwh = ann["bbox"] if isinstance(ann, dict) and "bbox" in ann else ann
                xyxy = self._ltwh_to_xyxy(bbox_ltwh)
                m = self._rasterize_box_to_mask(xyxy, h, w)  # (H,W) binary or None
                if m is not None:
                    masks.append(m)
                    inter_idx.append(i_img)
                else:
                    masks.append(None)
            else:
                masks.append(None)

        # temporal crop / dense trim (keep frames + masks aligned)
        frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end = \
            self._maybe_temporal_crop(frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end)
        frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end = \
            self._maybe_dense_trim_train(frame_ids, frames_thwc, masks, inter_idx, clip_start, clip_end)

        # Encoder preprocessing (CLIP-like)
        pil_all = [Image.fromarray(f) for f in frames_thwc]
        if self.num_frames_for_enc != -1:
            pil_for_enc = subsample_images(pil_all, self.num_frames_for_enc)
        else:
            pil_for_enc = pil_all
        enc_out = self.enc_preprocessor.preprocess(pil_for_enc)

        # Choose slice for SAM (annotated interval if present, else whole)
        if inter_idx:
            t_start_rel, t_end_rel = inter_idx[0], inter_idx[-1]
        else:
            t_start_rel, t_end_rel = 0, len(frames_thwc) - 1

        sam_frames = frames_thwc[t_start_rel:t_end_rel+1]
        sam_masks  = masks[t_start_rel:t_end_rel+1]

        # Remove frames where mask is None (parity with VidSTGDataset)
        keep = [i for i, m in enumerate(sam_masks) if m is not None]
        if keep:
            sam_frames = sam_frames[keep]
            sam_masks  = [sam_masks[i] for i in keep]
        else:
            # If all masks are missing, keep frame count aligned and use zero masks.
            sam_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(len(sam_frames))]

        if sam_masks:
            sanitized_masks = []
            for m in sam_masks:
                if isinstance(m, np.ndarray):
                    sanitized_masks.append(m.astype(np.uint8, copy=False))
                else:
                    sanitized_masks.append(np.zeros((h, w), dtype=np.uint8))
            sam_masks_np = np.stack(sanitized_masks, axis=0)  # [T,H,W]
        else:
            sam_masks_np = np.zeros((0, h, w), dtype=np.uint8)

        # optional subsample for sam (apply identically to frames and masks)
        if self.num_frames_for_sam != -1 and len(sam_frames) > 0:
            idxs = subsample_images(list(range(len(sam_frames))), self.num_frames_for_sam)
            sam_frames = sam_frames[idxs]
            sam_masks_np = sam_masks_np[idxs]

        # preprocess for SAM (images only; masks kept in orig resolution HxW)
        preprocessed_and_resize = [self.sam_preprocessor.preprocess(img) for img in sam_frames]
        preprocessed_for_sam = [x[0] for x in preprocessed_and_resize]
        resize = preprocessed_and_resize[0][1] if preprocessed_and_resize else None

        # pack masks -> torch [N,T,H,W] with N=1 (parity with VidSTGDataset)
        if sam_masks_np.size > 0:
            mask_tensor = torch.tensor(sam_masks_np)  # [T,H,W]
            masks_stack = torch.stack([mask_tensor])  # [N(=1),T,H,W]
        else:
            # empty tensor with correct dims
            masks_stack = torch.zeros((1, 0, h, w), dtype=torch.uint8)

        # dummy ignore label map (H,W)
        label = torch.ones(h, w) * self.ignore_label

        # conversation time bounds in absolute frame indices
        conv_t_start = frame_ids[t_start_rel] if len(frame_ids) else -100
        conv_t_end   = frame_ids[t_end_rel]   if len(frame_ids) else -100
        conversations = self._generate_conversation(caption=video["caption"], qtype=video["qtype"],
                                                    t_start=conv_t_start, t_end=conv_t_end)

        data_dict = {
            'file_path': vid_path,
            'preprocessed_for_sam': preprocessed_for_sam,  # list of tensors
            'images': enc_out['images'],
            'context_images': enc_out['context_images'],
            'conversations': conversations,
            'masks': masks_stack,                # [1, T_kept, H, W] in original resolution
            'label': label,                      # [H, W] ignore map
            'resize': resize,                    # SAM resize info
            'questions': None,
            'sampled_classes': None,
        }
        return data_dict

    def __getitem__(self, idx):
        return self.get_from_idx(idx)
