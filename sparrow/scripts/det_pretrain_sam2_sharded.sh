#!/usr/bin/env bash
set -euo pipefail

SAM2_CKPT_PATH=${1:-}
OUTPUT_DIR=${2:-}
SAM2_CFG=${3:-sam2_hiera_l.yaml}
MAX_STEPS_PER_SHARD=${4:-}
DATASET_CONFIG=${5:-sparrow/data/configs/det_pretrain.py}
NUM_SHARDS=${6:-5}
START_SHARD=${7:-0}
END_SHARD=${8:-$((NUM_SHARDS-1))}

if [[ -z "$SAM2_CKPT_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage:"
    echo "  bash sparrow/scripts/det_pretrain_sam2_sharded.sh <sam2_ckpt> <output_dir> [sam2_cfg] [max_steps_per_shard] [dataset_config] [num_shards] [start_shard] [end_shard]"
    exit 1
fi

if [[ "$NUM_SHARDS" -le 1 ]]; then
    echo "NUM_SHARDS must be > 1, got: $NUM_SHARDS"
    exit 1
fi

if [[ "$START_SHARD" -lt 0 || "$END_SHARD" -ge "$NUM_SHARDS" || "$START_SHARD" -gt "$END_SHARD" ]]; then
    echo "Invalid shard range: start=$START_SHARD end=$END_SHARD num_shards=$NUM_SHARDS"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

LATEST_STEP=0
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    for ckpt_dir in "${OUTPUT_DIR}"/checkpoint-*; do
        step="${ckpt_dir##*-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -gt "$LATEST_STEP" ]]; then
            LATEST_STEP=$step
        fi
    done
fi

echo "Starting sharded training"
echo "  output_dir: $OUTPUT_DIR"
echo "  dataset_config: $DATASET_CONFIG"
echo "  shard_range: $START_SHARD..$END_SHARD (num_shards=$NUM_SHARDS)"
echo "  base_global_step: $LATEST_STEP"

shard_count=0
for (( shard=START_SHARD; shard<=END_SHARD; shard++ )); do
    max_steps_arg=""
    if [[ -n "$MAX_STEPS_PER_SHARD" ]]; then
        if ! [[ "$MAX_STEPS_PER_SHARD" =~ ^[0-9]+$ ]]; then
            echo "MAX_STEPS_PER_SHARD must be an integer, got: $MAX_STEPS_PER_SHARD"
            exit 1
        fi
        target_max_steps=$((LATEST_STEP + (shard_count + 1) * MAX_STEPS_PER_SHARD))
        max_steps_arg="$target_max_steps"
    fi

    echo "\n=== Shard ${shard}/${NUM_SHARDS}-1 ==="
    if [[ -n "$max_steps_arg" ]]; then
        echo "  target cumulative max_steps: $max_steps_arg"
    else
        echo "  max_steps: not set (epoch-based)"
    fi

    shard_cmd=(
        bash "${SCRIPT_DIR}/det_pretrain_sam2.sh"
        --sam2-ckpt "$SAM2_CKPT_PATH"
        --output-dir "$OUTPUT_DIR"
        --sam2-cfg "$SAM2_CFG"
        --dataset-config "$DATASET_CONFIG"
        --subset-shard-index "$shard"
    )
    if [[ -n "$max_steps_arg" ]]; then
        shard_cmd+=(--max-steps "$max_steps_arg")
    fi
    "${shard_cmd[@]}"

    shard_count=$((shard_count + 1))
done

echo "Sharded training completed for shards ${START_SHARD}..${END_SHARD}."
