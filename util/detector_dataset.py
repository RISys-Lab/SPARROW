import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from util.sam_transforms import SAM_v2_Preprocess


@dataclass
class DetectionSample:
    pixel_values: torch.Tensor
    boxes: torch.Tensor
    orig_size: torch.Tensor
    image_id: int


class CocoDetectionDataset(Dataset):
    """
    Minimal COCO-style dataset that prepares inputs for the SAM2/Hiera detector.
    """

    def __init__(
        self,
        image_root: str,
        annotation_file: str,
        transforms: Callable[[np.ndarray], Tuple[torch.Tensor, Tuple[int, int]]] | None = None,
        min_box_area: float = 1.0,
    ):
        from pycocotools.coco import COCO

        self.image_root = image_root
        self.coco = COCO(annotation_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms or SAM_v2_Preprocess().preprocess
        self.min_box_area = min_box_area

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> DetectionSample:
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        file_name = img_info["file_name"]
        path = os.path.join(self.image_root, file_name)

        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        image_np = np.array(image)

        pixel_values, _ = self.transforms(image_np)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w * h < self.min_box_area:
                continue
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            boxes.append([x1, y1, x2, y2])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        orig_size = torch.tensor([orig_h, orig_w], dtype=torch.float32)

        return DetectionSample(
            pixel_values=pixel_values.float(),
            boxes=boxes_tensor,
            orig_size=orig_size,
            image_id=image_id,
        )


def hiera_det_collate_fn(batch: List[DetectionSample]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
    pixel_values = torch.stack([sample.pixel_values for sample in batch], dim=0)
    targets: List[Dict[str, torch.Tensor]] = []
    for sample in batch:
        targets.append(
            {
                "boxes": sample.boxes,
                "orig_size": sample.orig_size,
                "image_id": torch.tensor(sample.image_id, dtype=torch.int64),
            }
        )
    return {
        "pixel_values": pixel_values,
        "targets": targets,
    }

