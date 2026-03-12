import os

# -----------------------------------------------------------------------------
# Dataset roots (override via environment variables; no hardcoded machine paths)
#
# Minimal setup:
#   export SPARROW_DATA_ROOT=/path/to/datasets/det_pretrain
#
# Optional (if annotations live elsewhere):
#   export SPARROW_ANN_ROOT=/path/to/datasets/det_pretrain/annotations
# -----------------------------------------------------------------------------
DATA_ROOT = os.path.expanduser(
    os.environ.get("SPARROW_DATA_ROOT", "./datasets/det_pretrain")
)
ANN_ROOT = os.path.expanduser(
    os.environ.get("SPARROW_ANN_ROOT", os.path.join(DATA_ROOT, "annotations"))
)


def _p(*parts):
    return os.path.join(*parts)


datasets = [
    {
        'type': 'coco_box',
        'ann_file': _p(ANN_ROOT, 'class_agnostic_coco_instances_train2017.json'),
        'img_prefix': _p(DATA_ROOT, 'coco', 'train2017'),

        # 'ratio': 0.2,
        # 'subset_non_overlap': True,
    },
    {
        'type': 'obj365_box',
        'ann_file': _p(ANN_ROOT, 'class_agnostic_obj365v2_train_new.json'),
        'img_prefix': _p(DATA_ROOT, 'Objects365V2', 'train', 'images'),

        # 'ratio': 0.2,
        # 'subset_non_overlap': True,
    },
    {
        'type': 'openimage_box',
        'ann_file': _p(ANN_ROOT, 'class_agnostic_openimages_v6_train_bbox.json'),
        'img_prefix': _p(DATA_ROOT, 'open-images-v6', 'train', 'data'),
        # Annotation file stores names like train_a/xxxx.jpg, but files are flattened under train/data.
        'flatten_file_name': True,

        # 'ratio': 0.2,
        # 'subset_non_overlap': True,
    },
    {
        'type': 'v3det_box',
        'ann_file': _p(ANN_ROOT, 'class_agnostic_v3det_2023_v1_train.json'),
        'img_prefix': _p(DATA_ROOT, 'V3Det'),

        # 'ratio': 0.2,
        # 'subset_non_overlap': True,
    },
    {
        'type': 'sa1b_box',
        # 'ann_file': _p(ANN_ROOT, 'class_agnostic_sa1b_2m.json'),
        'ann_file': _p(ANN_ROOT, 'class_agnostic_sa1b_000000_000050.json'),
        'img_prefix': _p(DATA_ROOT, 'SA1B', 'images'),

        # 'ratio': 0.2,
        # 'subset_non_overlap': True,
    },
]
