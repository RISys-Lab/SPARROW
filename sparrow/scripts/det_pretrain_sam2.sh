#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Train SAM2 DDETR.

Usage (recommended):
  bash sparrow/scripts/det_pretrain_sam2.sh \
    --sam2-ckpt /path/to/sam2_hiera_large.pt \
    --output-dir output/ddetr_sam2 \
    [options]

Legacy positional mode (still supported):
  bash sparrow/scripts/det_pretrain_sam2.sh <sam2_ckpt> <output_dir> [sam2_cfg] [max_steps] [dataset_config] [subset_shard_index] [samples_per_dataset]

Required:
  --sam2-ckpt PATH            SAM2 checkpoint path.
  --output-dir DIR            Output directory for checkpoints/logs.

Options:
  --sam2-cfg NAME             SAM2 Hydra config name. Default: sam2_hiera_l.yaml
  --dataset-config PATH       Dataset config file. Default: sparrow/data/configs/det_pretrain.py
  --max-steps INT             Optional cumulative max training steps.
  --subset-shard-index INT    Optional shard index (sets GROMA_SUBSET_SHARD_INDEX).
  --samples-per-dataset INT   Optional balanced sampling size per dataset.
  --nproc-per-node INT        Number of processes per node. Default: $NPROC_PER_NODE or 1
  --master-port INT           torchrun master port. Default: 25001
  --num-train-epochs INT      Default: 10
  --learning-rate FLOAT       Default: 2e-4
  --per-device-batch-size INT Default: 64
  --help                      Show this message.

Extra args:
  Use '--' to pass extra args directly to `python -m sparrow.train.train_det_sam2`.
EOF
}

SAM2_CKPT_PATH=""
OUTPUT_DIR=""
SAM2_CFG="sam2_hiera_l.yaml"
MAX_STEPS=""
DATASET_CONFIG="sparrow/data/configs/det_pretrain.py"
SUBSET_SHARD_INDEX=""
SAMPLES_PER_DATASET=""
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="25001"
NUM_TRAIN_EPOCHS="10"
LEARNING_RATE="2e-4"
PER_DEVICE_BATCH_SIZE="64"
EXTRA_ARGS=()

# Backward-compatible positional mode.
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    SAM2_CKPT_PATH="${1:-}"
    OUTPUT_DIR="${2:-}"
    SAM2_CFG="${3:-$SAM2_CFG}"
    MAX_STEPS="${4:-}"
    DATASET_CONFIG="${5:-$DATASET_CONFIG}"
    SUBSET_SHARD_INDEX="${6:-}"
    SAMPLES_PER_DATASET="${7:-}"
    shift $(( $# >= 7 ? 7 : $# ))
    if [[ $# -gt 0 ]]; then
        EXTRA_ARGS+=("$@")
    fi
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sam2-ckpt)
                SAM2_CKPT_PATH="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --sam2-cfg)
                SAM2_CFG="$2"
                shift 2
                ;;
            --max-steps)
                MAX_STEPS="$2"
                shift 2
                ;;
            --dataset-config)
                DATASET_CONFIG="$2"
                shift 2
                ;;
            --subset-shard-index)
                SUBSET_SHARD_INDEX="$2"
                shift 2
                ;;
            --samples-per-dataset)
                SAMPLES_PER_DATASET="$2"
                shift 2
                ;;
            --nproc-per-node)
                NPROC_PER_NODE="$2"
                shift 2
                ;;
            --master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --num-train-epochs)
                NUM_TRAIN_EPOCHS="$2"
                shift 2
                ;;
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --per-device-batch-size)
                PER_DEVICE_BATCH_SIZE="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --)
                shift
                EXTRA_ARGS+=("$@")
                break
                ;;
            *)
                echo "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
fi

if [[ -z "$SAM2_CKPT_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --sam2-ckpt and --output-dir are required."
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "$SUBSET_SHARD_INDEX" ]]; then
    export GROMA_SUBSET_SHARD_INDEX="$SUBSET_SHARD_INDEX"
    echo "Using GROMA_SUBSET_SHARD_INDEX=${GROMA_SUBSET_SHARD_INDEX}"
fi

mkdir -p "$OUTPUT_DIR"

LATEST_STEP=0
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    for ckpt_dir in "${OUTPUT_DIR}"/checkpoint-*; do
        step="${ckpt_dir##*-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -gt "$LATEST_STEP" ]]; then
            LATEST_STEP=$step
        fi
    done
    echo "Found existing checkpoint step: ${LATEST_STEP}"
fi

echo "Starting SAM2 DDETR training"
echo "  SAM2 ckpt: $SAM2_CKPT_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  SAM2 cfg: $SAM2_CFG"
echo "  Dataset cfg: $DATASET_CONFIG"
echo "  NPROC per node: $NPROC_PER_NODE"

CMD=(
    torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
    -m sparrow.train.train_det_sam2
    --vis_encoder "$SAM2_CKPT_PATH"
    --sam2_cfg "$SAM2_CFG"
    --dataset_config "$DATASET_CONFIG"
    --bf16 True
    --tf32 True
    --num_classes 1
    --num_queries 300
    --two_stage True
    --with_box_refine True
    --ddetr_hidden_dim 256
    --num_encoder_layers 6
    --num_decoder_layers 6
    --num_feature_levels 1
    --freeze_vis_encoder True
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay 1e-4
    --max_grad_norm 1.0
    --warmup_steps 100
    --logging_steps 100
    --lr_scheduler_type cosine
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --dataloader_num_workers 0
    --save_strategy epoch
    --save_total_limit 1
    --report_to none
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_STEPS" ]]; then
    if [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] && [[ "$LATEST_STEP" -ge "$MAX_STEPS" ]]; then
        echo "Warning: MAX_STEPS=${MAX_STEPS} is <= existing global step ${LATEST_STEP}."
        echo "Training will resume and finish immediately. Increase MAX_STEPS to continue."
    fi
    CMD+=(--max_steps "$MAX_STEPS")
fi

if [[ -n "$SAMPLES_PER_DATASET" ]]; then
    CMD+=(--equalize_data_sources True --samples_per_dataset "$SAMPLES_PER_DATASET")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}" | tee -a "$OUTPUT_DIR/train.log"
