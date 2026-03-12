This training largely follows [Groma](https://github.com/FoundationVision/Groma) and [GLEE](https://github.com/FoundationVision/GLEE/blob/main/assets/DATA.md).

## Required Checkpoints
Provide a SAM2 checkpoint for training and store it under `checkpoints` directory:
- Example: `sam2_hiera_large.pt`

For inference, you can use either:
- A trained SAM2 DDETR checkpoint directory (recommended), or
- SAM2 checkpoint only (runs with randomly initialized DDETR head)

## Data Preparation
`sparrow/data/configs/det_pretrain.py` is now environment-variable driven.

Set the dataset root:

```bash
export SPARROW_DATA_ROOT=/path/to/datasets/det_pretrain
# Optional only if annotations are in a different location:
# export SPARROW_ANN_ROOT=/path/to/datasets/det_pretrain/annotations
```

Expected directory layout:

```text
/path/to/datasets/det_pretrain/
├── annotations/
│   ├── class_agnostic_coco_instances_train2017.json
│   ├── class_agnostic_obj365v2_train_new.json
│   ├── class_agnostic_openimages_v6_train_bbox.json
│   ├── class_agnostic_v3det_2023_v1_train.json
│   ├── class_agnostic_sa1b_000000_000050.json
│   └── class_agnostic_sa1b_2m.json                 # optional (full SA1B)
├── coco/
│   └── train2017/
├── Objects365V2/
│   └── train/
│       └── images/
├── open-images-v6/
│   └── train/
│       └── data/
├── V3Det/
└── SA1B/
    └── images/
```
We downloaded data from the SA1B official website, and only use [sa_000000.tar ~ sa_000050.tar] to preprocess into the required format and train the model.
If needed, switch SA1B annotation in `sparrow/data/configs/det_pretrain.py` from the small shard to the full file.

## Training
Run help:

```bash
bash sparrow/scripts/det_pretrain_sam2.sh --help
```

Recommended command:

```bash
bash sparrow/scripts/det_pretrain_sam2.sh \
  --sam2-ckpt "/path/to/sam2_hiera_large.pt" \
  --output-dir "output/ddetr_sam2" \
  --sam2-cfg sam2_hiera_l.yaml \
  --dataset-config sparrow/data/configs/det_pretrain.py \
  --max-steps 500000 \
  --subset-shard-index 0 \
  --samples-per-dataset 20000
```

Optional: sharded training loop

```bash
bash sparrow/scripts/det_pretrain_sam2_sharded.sh \
  "/path/to/sam2_hiera_large.pt" \
  "output/ddetr_sam2" \
  sam2_hiera_l.yaml \
  100000 \
  sparrow/data/configs/det_pretrain.py \
  5 0 4
```

## Inference
Run help:

```bash
python -m sparrow.eval.run_ddetr_sam2 --help
```

Single image:

```bash
python -m sparrow.eval.run_ddetr_sam2 \
  --model-name output/ddetr_sam2/checkpoint-300000 \
  --image-file "/path/to/image.jpg" \
  --output-dir det_vis \
  --device auto
```

Image directory:

```bash
python -m sparrow.eval.run_ddetr_sam2 \
  --model-name output/ddetr_sam2/checkpoint-300000 \
  --image-dir ./images \
  --output-dir det_vis \
  --device auto
```

SAM2-only (random DDETR head):

```bash
python -m sparrow.eval.run_ddetr_sam2 \
  --sam2-ckpt "/path/to/sam2_hiera_large.pt" \
  --image-file "/path/to/image.jpg" \
  --output-dir det_vis
```

## Outputs
- Training checkpoints/logs:
  - `output/ddetr_sam2/`
- Visualization outputs:
  - `det_vis/*_filter.jpg`

## Troubleshooting
- `ModuleNotFoundError: mmcv`: install compatible MMCV with `pip install -U openmim && mim install "mmcv-full==1.4.8"`.
- CUDA requested but unavailable:
  - Use `--device cpu` for inference.
- Empty training data:
  - Recheck paths in `sparrow/data/configs/det_pretrain.py`.
