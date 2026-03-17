# video_ref_ytvos_dataset.py
from __future__ import annotations
import json
import pickle
from typing import Dict, List, Tuple
from .ReVOS_Dataset import VideoReVOSDataset

class VideoRefYoutubeVOSDataset(VideoReVOSDataset):
    def _preload_json(
        self,
        expression_file: str,
        mask_file: str,
    ):
        # Load expressions JSON (Ref-YTVOS)
        with open(expression_file, "r") as f:
            expression_datas = json.load(f)["videos"]

        metas: List[dict] = []
        anno_count = 0
        vid2metaid: Dict[str, List[int]] = {}

        for vid_name, vid_express_data in expression_datas.items():
            vid_frames = sorted(vid_express_data["frames"])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data["expressions"].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data["expressions"][exp_id]
                metas.append({
                    "video": vid_name,
                    "exp": exp_dict["exp"],
                    # Ref-YTVOS has one object per expression; generate sequential ids
                    "mask_anno_id": [str(anno_count)],
                    "obj_id": exp_dict.get("obj_id", [0]),
                    "anno_id": [str(anno_count)],
                    "frames": vid_frames,
                    "exp_id": exp_id,
                    "length": vid_len,
                })
                vid2metaid.setdefault(vid_name, []).append(len(metas) - 1)
                anno_count += 1

        # Load mask dict **as pickle**
        with open(mask_file, "rb") as f:
            mask_dict = pickle.load(f)

        # Optional sanity check (first few)
        for mi in range(min(3, len(metas))):
            for aid in metas[mi]["mask_anno_id"]:
                if aid not in mask_dict:
                    raise KeyError(f"mask_anno_id '{aid}' missing from mask_dict.pkl")

            if len(mask_dict[metas[mi]["mask_anno_id"][0]]) != len(metas[mi]["frames"]):
                raise ValueError(
                    f"Length mismatch: anno {metas[mi]['mask_anno_id'][0]} "
                    f"has {len(mask_dict[metas[mi]['mask_anno_id'][0]])} masks "
                    f"but video has {len(metas[mi]['frames'])} frames"
                )

        return vid2metaid, metas, mask_dict
