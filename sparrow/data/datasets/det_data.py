# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import torch
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh


def normalize_box_coordinates(bbox, img_shape):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    img_h, img_w = img_shape[:2]
    bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
    bbox_new = torch.clamp(torch.cat(bbox_new, dim=-1), min=0., max=1.)
    return bbox_new


class ClassAgnosticCoCo(CocoDataset):
    CLASSES = ('object',)
    PALETTE = None

    def __init__(
        self,
        ann_file=None,
        img_prefix=None,
        test_mode=False,
        flatten_file_name=False,
        file_name_prefix_to_strip=None,
        skip_missing=True,
        max_skip_retries=100,
        missing_log_interval=50,
    ):
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True
        )

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize',
                 img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomCrop',
                 crop_type='absolute_range',
                 crop_size=(448, 896),
                 allow_negative_crop=False),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False, override=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**dataset_cfg)
        self.skip_missing = bool(skip_missing)
        self.max_skip_retries = int(max_skip_retries)
        self.missing_log_interval = int(missing_log_interval)
        self._missing_sample_count = 0
        self._rewrite_file_names(
            flatten_file_name=flatten_file_name,
            file_name_prefix_to_strip=file_name_prefix_to_strip,
        )

    def _rewrite_file_names(self, flatten_file_name=False, file_name_prefix_to_strip=None):
        if not hasattr(self, "data_infos"):
            return
        if not flatten_file_name and not file_name_prefix_to_strip:
            return

        for info in self.data_infos:
            file_name = info.get("filename", None)
            if not file_name:
                continue
            if file_name_prefix_to_strip and file_name.startswith(file_name_prefix_to_strip):
                file_name = file_name[len(file_name_prefix_to_strip):]
            if flatten_file_name and "/" in file_name:
                file_name = file_name.split("/", 1)[1]
            info["filename"] = file_name

    def __getitem__(self, idx):
        data_item = self._get_data_item_with_missing_retry(idx)
        gt_bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        gt_bboxes = normalize_box_coordinates(gt_bboxes, img_shape)
        data_dict = {
            'image': data_item['img'].data,
            'class_labels': data_item['gt_labels'].data,
            'bboxes': gt_bboxes,
            'ori_shape': data_item['img_metas'].data['ori_shape'],
            'source': 'coco'
        }
        return data_dict

    def _get_data_item_with_missing_retry(self, idx):
        cur_idx = idx
        last_error = None
        max_attempts = max(self.max_skip_retries, 0) + 1
        for _ in range(max_attempts):
            try:
                return super().__getitem__(cur_idx)
            except FileNotFoundError as error:
                if not self.skip_missing:
                    raise
                last_error = error
                self._missing_sample_count += 1
                if (
                    self._missing_sample_count <= 3
                    or self._missing_sample_count % max(self.missing_log_interval, 1) == 0
                ):
                    missing_file = self.data_infos[cur_idx].get("filename", "<unknown>")
                    full_path = os.path.join(self.img_prefix, missing_file)
                    print(
                        f"[dataset] missing image skipped "
                        f"(count={self._missing_sample_count}, idx={cur_idx}): {full_path}"
                    )
                if len(self) <= 1:
                    break
                if hasattr(self, "flag") and len(getattr(self, "flag", [])) == len(self):
                    cur_idx = int(self._rand_another(cur_idx))
                else:
                    cur_idx = (cur_idx + 1) % len(self)
        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_attempts} attempts "
            f"(start_idx={idx}, dataset_len={len(self)})."
        ) from last_error


class ClassAgnosticSA1B(CocoDataset):
    CLASSES = ('object',)
    PALETTE = None

    def __init__(
        self,
        ann_file=None,
        img_prefix=None,
        test_mode=False,
        flatten_file_name=False,
        file_name_prefix_to_strip=None,
        skip_missing=True,
        max_skip_retries=100,
        missing_log_interval=50,
    ):
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True
        )

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize',
                 img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomCrop',
                 crop_type='absolute_range',
                 crop_size=(448, 896),
                 allow_negative_crop=False),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False, override=True),
            dict(type='CustomFilterAnnotations', min_size=14 * 14, max_size=400 * 400),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**dataset_cfg)
        self.skip_missing = bool(skip_missing)
        self.max_skip_retries = int(max_skip_retries)
        self.missing_log_interval = int(missing_log_interval)
        self._missing_sample_count = 0
        self._rewrite_file_names(
            flatten_file_name=flatten_file_name,
            file_name_prefix_to_strip=file_name_prefix_to_strip,
        )

    def _rewrite_file_names(self, flatten_file_name=False, file_name_prefix_to_strip=None):
        if not hasattr(self, "data_infos"):
            return
        if not flatten_file_name and not file_name_prefix_to_strip:
            return

        for info in self.data_infos:
            file_name = info.get("filename", None)
            if not file_name:
                continue
            if file_name_prefix_to_strip and file_name.startswith(file_name_prefix_to_strip):
                file_name = file_name[len(file_name_prefix_to_strip):]
            if flatten_file_name and "/" in file_name:
                file_name = file_name.split("/", 1)[1]
            info["filename"] = file_name

    def __getitem__(self, idx):
        data_item = self._get_data_item_with_missing_retry(idx)
        gt_bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        gt_bboxes = normalize_box_coordinates(gt_bboxes, img_shape)
        data_dict = {
            'image': data_item['img'].data,
            'class_labels': data_item['gt_labels'].data,
            'bboxes': gt_bboxes,
            'ori_shape': data_item['img_metas'].data['ori_shape'],
            'source': 'sa1b'
        }
        return data_dict

    def _get_data_item_with_missing_retry(self, idx):
        cur_idx = idx
        last_error = None
        max_attempts = max(self.max_skip_retries, 0) + 1
        for _ in range(max_attempts):
            try:
                return super().__getitem__(cur_idx)
            except FileNotFoundError as error:
                if not self.skip_missing:
                    raise
                last_error = error
                self._missing_sample_count += 1
                if (
                    self._missing_sample_count <= 3
                    or self._missing_sample_count % max(self.missing_log_interval, 1) == 0
                ):
                    missing_file = self.data_infos[cur_idx].get("filename", "<unknown>")
                    full_path = os.path.join(self.img_prefix, missing_file)
                    print(
                        f"[dataset] missing image skipped "
                        f"(count={self._missing_sample_count}, idx={cur_idx}): {full_path}"
                    )
                if len(self) <= 1:
                    break
                if hasattr(self, "flag") and len(getattr(self, "flag", [])) == len(self):
                    cur_idx = int(self._rand_another(cur_idx))
                else:
                    cur_idx = (cur_idx + 1) % len(self)
        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_attempts} attempts "
            f"(start_idx={idx}, dataset_len={len(self)})."
        ) from last_error

