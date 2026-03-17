# SPARROW Full Model Training Guide

This guide covers full-model training for SPARROW after detection pretraining. It documents the two training stages used in this repository, the required dataset paths, checkpoint behavior, and export steps.

## Overview

Full model training is split into two stages:

- **Stage 1 (`train_stage1_tsf.py`)**  
  TSF-focused adapter/LoRA training with detection guidance disabled.

- **Stage 2 (`train_stage2_filter.py`)**  
  Proposal-filter refinement with detection guidance enabled.

Both stages use:
- **DeepSpeed** for distributed training
- the shared **`util/` dataset pipeline**
- image and video dataset roots passed through command-line arguments

---

## Table of Contents

- [Training Stages](#training-stages)
- [Dataset Configuration](#dataset-configuration)
- [Example Dataset Layout](#example-dataset-layout)
- [Quick Start](#quick-start)
- [Checkpoints and Resume](#checkpoints-and-resume)
- [Recommended Training Workflow](#recommended-training-workflow)
- [Exporting Hugging Face Weights](#exporting-hugging-face-weights)
- [Useful Training Flags](#useful-training-flags)
- [Common Pitfalls](#common-pitfalls)

---

## Training Stages

### Stage 1: TSF Training

Stage 1 trains the TSF-focused adapter/LoRA components while keeping detection guidance disabled.

Training script:

```bash
train/train_stage1_tsf.py
```

### Stage 2: Proposal Filter Refinement

Stage 2 refines the model with proposal filtering and detection guidance enabled.

Training script:

```bash
train/train_stage2_filter.py
```

---

## Dataset Configuration

Two dataset root arguments are always required:

- `--dataset_dir`: root directory for image datasets
- `--video_dataset_dir`: root directory for video datasets

Additional dataset-related arguments used by the training scripts:

- `--sa2va_video_data_root`  
  Passed into `SA2VA_VIDEO_DATA_ROOT`

- `--vot_data_root`  
  Passed into `VOT_DATA_ROOT`

- `--visual_token_reserve`  
  Passed into `SPARROW_VISUAL_TOKEN_RESERVE`  
  Default: `255`

These paths are consumed by `util/dataset.py` to resolve dataset locations for training.


### Dataset Download

All dataset download, preparation, and extraction instructions are documented in:

👉 **[Dataset.md](./Dataset.md)**

This includes:
- VideoGLaMM datasets
- Sa2VA-Training datasets
- Required directory structure
- Extraction and organization steps

Make sure your dataset layout matches the structure described there before starting training.


### Supported dataset groups

Depending on your training configuration, the pipeline may reference datasets such as:

- `sam2`
- `revos`
- `rvos`
- `mevis_sa2va`
- `got10k_*`
- `lasot`
- `a2d_sentences`
- `hc_stvg_*`
- `vid_*`

Only prepare the datasets you actually enable in your training setup.

---

## Dataset Layout

Adjust this structure to match the datasets you are using.

```text
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   └── val2017
│   ├── cocostuff
│   │   └── train2017
│   ├── grandf_dataset
│   │   ├── coco_2014
│   │   ├── coco_2017
│   │   ├── flikcr_30k
│   │   ├── GranD-f
│   │   └── GranDf_HA_images
│   ├── llava_dataset
│   │   └── coco
│   ├── mapillary
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   ├── refer_seg
│   │   ├── coco_2014
│   │   ├── images
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       └── pascal_part
├── video_dataset
│   ├── a2d_sentences
│   │   └── a2d_annotation_with_instances
│   ├── activitynet
│   │   ├── test
│   │   └── train
│   ├── activitynet_captions
│   ├── activitynet_entities
│   │   └── data
│   ├── activitynet_entities_gcg
│   │   ├── anns
│   │   ├── anns_old
│   │   ├── masks
│   │   ├── masks_old
│   │   └── video_frames
│   ├── burst
│   │   ├── annotations
│   │   ├── burst_ytvis_gcg
│   │   ├── frames
│   │   └── instruction_data
│   ├── charades_sta
│   │   ├── Charades_v1_480
│   │   └── data
│   ├── hcstvg
│   │   ├── anno_v2
│   │   ├── qa
│   │   └── Video
│   ├── hcstvg_gcg
│   │   ├── train
│   │   └── train_captions
│   ├── mevis
│   │   ├── train
│   │   ├── valid
│   │   └── valid_u
│   ├── mevis_gcg
│   │   ├── train
│   │   └── valid_u
│   ├── processed
│   │   ├── refer_davis
│   │   └── vidstg
│   ├── qv_highlights
│   ├── refer_davis
│   │   ├── DAVIS16
│   │   └── DAVIS17
│   ├── refer_youtube_vos
│   │   ├── meta_expressions
│   │   ├── test
│   │   ├── train
│   │   ├── valid
│   │   └── zip
│   ├── video_gcg
│   │   └── instruction_data
│   ├── video_instruct_100k
│   │   ├── Test_Human_Annotated_Captions
│   │   └── Test_Videos
│   ├── vidstg
│   │   ├── video
│   │   ├── vidor_annotations
│   │   └── vidstg_annotations
│   ├── vidstg_gcg
│   │   ├── train
│   │   ├── train_captions
│   │   ├── val
│   │   └── val_captions
│   ├── ytvis
│   │   ├── vis
│   │   └── vos
│   └── ytvos_gcg
│       └── train
└── VOT_dataset
│   ├── A2D_sentences
│   │   └── Images
│   ├── GOT-10k
│   │   ├── train
│   │   └── val
│   ├── HC-STVG
│   │   ├── Images
│   │   └── Val_Images
│   ├── LaSOT
│   │   ├── airplane
│   │   ├── basketball
│   │   ├── bear
│   │   ├── bicycle
│   │   ├── bird
│   │   ├── boat
│   │   ├── book
│   │   ├── bottle
│   │   ├── bus
│   │   ├── car
│   │   ├── cat
│   │   ├── cattle
│   │   ├── chameleon
│   │   ├── coin
│   │   ├── crab
│   │   ├── crocodile
│   │   ├── cup
│   │   ├── deer
│   │   ├── dog
│   │   ├── drone
│   │   ├── electricfan
│   │   ├── elephant
│   │   ├── flag
│   │   ├── fox
│   │   ├── frog
│   │   ├── gametarget
│   │   ├── gecko
│   │   ├── giraffe
│   │   ├── goldfish
│   │   ├── gorilla
│   │   ├── guitar
│   │   ├── hand
│   │   ├── hat
│   │   ├── helmet
│   │   ├── hippo
│   │   ├── horse
│   │   ├── kangaroo
│   │   ├── kite
│   │   ├── leopard
│   │   ├── licenseplate
│   │   ├── lion
│   │   ├── lizard
│   │   ├── microphone
│   │   ├── monkey
│   │   ├── motorcycle
│   │   ├── mouse
│   │   ├── person
│   │   ├── pig
│   │   ├── pool
│   │   ├── rabbit
│   │   ├── racing
│   │   ├── robot
│   │   ├── rubicCube
│   │   ├── sepia
│   │   ├── shark
│   │   ├── sheep
│   │   ├── skateboard
│   │   ├── spider
│   │   ├── squirrel
│   │   ├── surfboard
│   │   ├── swing
│   │   ├── tank
│   │   ├── tiger
│   │   ├── train
│   │   ├── truck
│   │   ├── turtle
│   │   ├── umbrella
│   │   ├── volleyball
│   │   ├── yoyo
│   │   └── zebra
│   └── VID-Sentence
│       └── VID
├── video_datas/
│   ├── chat_univi
│   │   └── Activity_Videos
│   ├── davis17
│   │   ├── JPEGImages
│   │   ├── meta_expressions
│   │   ├── train
│   │   └── valid
│   ├── mevis
│   │   ├── train
│   │   ├── valid
│   │   └── valid_u
│   ├── revos
│   │   ├── LV-VIS
│   │   ├── MOSE
│   │   ├── OVIS
│   │   ├── TAO
│   │   └── UVO
│   ├── ReVOS
│   │   └── JPEGImages
│   ├── rvos
│   │   ├── meta_expressions
│   │   ├── train
│   │   └── valid
│   └── sam_v_full
│       ├── sav_000
│       ├── sav_001
│       ├── sav_002
│       ├── sav_003
│       ├── sav_004
│       ├── sav_005
│       ├── sav_006
│       ├── sav_007
│       ├── sav_008
│       ├── sav_009
│       ├── sav_010
│       ├── sav_011
│       ├── sav_012
│       ├── sav_013
│       ├── sav_014
│       ├── sav_015
│       ├── sav_016
│       ├── sav_017
│       ├── sav_018
│       ├── sav_019
│       ├── sav_020
│       ├── sav_021
│       ├── sav_022
│       ├── sav_023
│       ├── sav_024
│       ├── sav_025
│       ├── sav_026
│       ├── sav_027
│       ├── sav_028
│       ├── sav_029
│       ├── sav_030
│       ├── sav_031
│       ├── sav_032
│       ├── sav_033
│       ├── sav_034
│       ├── sav_035
│       ├── sav_036
│       ├── sav_037
│       ├── sav_038
│       ├── sav_039
│       ├── sav_040
│       ├── sav_041
│       ├── sav_042
│       ├── sav_043
│       ├── sav_044
│       ├── sav_045
│       ├── sav_046
│       ├── sav_047
│       ├── sav_048
│       ├── sav_049
│       ├── sav_050
│       ├── sav_051
│       ├── sav_052
│       ├── sav_053
│       ├── sav_054
│       ├── sav_055
│       ├── sav_md5sum
│       ├── sav_sha256sum
│       ├── sav_test
│       └── sav_val
```

---

## Quick Start

Run all commands from the repository root.

### Stage 1

```bash
deepspeed train/train_stage1_tsf.py \
  --exp_name stage1_tsf \
  --dataset_dir datasets/image \
  --video_dataset_dir datasets/video \
  --sa2va_video_data_root datasets/video/sa2va \
  --vot_data_root datasets/video/VOT_dataset
```

### Stage 2

```bash
deepspeed train/train_stage2_filter.py \
  --exp_name stage2_filter \
  --dataset_dir datasets/image \
  --video_dataset_dir datasets/video \
  --sa2va_video_data_root datasets/video/sa2va \
  --vot_data_root datasets/video/VOT_dataset
```

---

## Checkpoints and Resume

### Checkpoint locations

Training checkpoints are saved to:

```text
runs/ckpts/<exp_name>/
```

DeepSpeed stores:
- a `latest` pointer
- one or more `global_step*` checkpoint directories

Training logs are saved to:

```text
runs/logs/<exp_name>/
```

### Resume behavior

Both training scripts support automatic resume behavior:

- If `--auto_resume True` and `runs/ckpts/<exp_name>/latest` exists, training resumes automatically.
- Otherwise, if `--resume_dir <path>` is provided, training resumes from that checkpoint directory.

### Important note

Use **different `--exp_name` values** for Stage 1 and Stage 2.

For example:

- `stage1_tsf`
- `stage2_filter`

If both stages use the same experiment name, they will write into the same checkpoint and log directories, which can corrupt your run organization.

---

## Recommended Training Workflow

A typical full training workflow looks like this:

1. Run **Stage 1** using a dedicated experiment name:
   ```bash
   --exp_name stage1_tsf
   ```

2. Start **Stage 2** with a new experiment name:
   ```bash
   --exp_name stage2_filter
   ```

3. If Stage 2 is interrupted, restart it with the same experiment name and enable auto-resume:
   ```bash
   --exp_name stage2_filter --auto_resume True
   ```

### Recommended practice

Do **not** directly resume Stage 2 from a Stage 1 DeepSpeed checkpoint as if they were the same run.

Instead:
- use Stage 1 weights as initialization for Stage 2
- start Stage 2 with a fresh optimizer and scheduler state

---

## Exporting Hugging Face Weights

After training, you can export merged Hugging Face model weights using:

- `train/save_hf_weights_stage1.py`
- `train/save_hf_weights_stage2.py`

### Required arguments

Both scripts require:

- `--save_hf_model True`
- `--intermediate_weight <state_dict.pt>`
- `--hf_save_path <output_dir>`

### Stage 1 export

```bash
python train/save_hf_weights_stage1.py \
  --save_hf_model True \
  --intermediate_weight /path/to/model_state_dict.pt \
  --hf_save_path hf_model/stage1
```

### Stage 2 export

```bash
python train/save_hf_weights_stage2.py \
  --save_hf_model True \
  --intermediate_weight /path/to/model_state_dict.pt \
  --hf_save_path hf_model/stage2
```

### DeepSpeed ZeRO checkpoints

If your checkpoint is stored in DeepSpeed ZeRO sharded format, first convert it into a single FP32 state dict, for example with DeepSpeed’s `zero_to_fp32.py`, and then pass that output file as:

```bash
--intermediate_weight /path/to/model_state_dict.pt
```

---

## Useful Training Flags

Some commonly useful flags:

- `--dataset`  
  Selects which datasets to use

- `--sample_rates_for_datasets`  
  Controls dataset sampling weights

- `--lazy_load_heavy_datasets True`  
  Helps reduce memory pressure when using large datasets

- `--gradient_checkpointing True`  
  Reduces memory usage during training

- `--workers`  
  Number of dataloader workers per local rank

---

## Common Pitfalls

### 1. Missing files in SA2VA or VOT roots

If a dataset is enabled in the training config but the corresponding files are missing under:
- `--sa2va_video_data_root`
- `--vot_data_root`

training will fail during dataset initialization.

### 2. Reusing the same experiment name

Do not use the same `--exp_name` for both Stage 1 and Stage 2.  
This causes both stages to share the same checkpoint and log directories.

### 3. Incorrect Stage 2 resume behavior

Do not treat Stage 2 as a direct continuation of Stage 1 at the DeepSpeed run level.

Preferred workflow:
- initialize Stage 2 from Stage 1 model weights
- keep Stage 2 optimizer/scheduler state fresh

---



