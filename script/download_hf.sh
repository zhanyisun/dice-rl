#!/usr/bin/env bash
set -euo pipefail

# Downloads Robomimic datasets/checkpoints from Hugging Face into the expected local layout.
#
# Expected HF repos:
#   wintermelontree/robomimic-pretrain-data
#   wintermelontree/robomimic-finetune-data
#   wintermelontree/robomimic-pretrain-checkpoints
#   wintermelontree/robomimic-finetune-checkpoints
#
# Expected folder structure INSIDE each repo:
#   {env_name}_img/
#   {env_name}_low_dim/
#
# Example:
#   wintermelontree/robomimic-pretrain-data/can_img/...
#   wintermelontree/robomimic-pretrain-data/can_low_dim/...

PRETRAIN_DATA_REPO="wintermelontree/robomimic-pretrain-data"
FINETUNE_DATA_REPO="wintermelontree/robomimic-finetune-data"
PRETRAIN_CKPT_REPO="wintermelontree/robomimic-pretrain-checkpoints"
FINETUNE_CKPT_REPO="wintermelontree/robomimic-finetune-checkpoints"

# Check if required environment variables are set
if [ -z "${DICE_RL_DATA_DIR:-}" ] || [ -z "${DICE_RL_LOG_DIR:-}" ]; then
    echo "Error: DICE_RL_DATA_DIR and DICE_RL_LOG_DIR environment variables are not set."
    echo "Please run: source script/set_path.sh"
    exit 1
fi

ENVS=("can" "square" "transport" "tool_hang")

download_dir() {
    local repo_id="$1"
    local subdir="$2"
    local local_dir="$3"
    local repo_type="${4:-dataset}"

    mkdir -p "$local_dir"

    echo "============================================================"
    echo "Downloading:"
    echo "  repo      : $repo_id"
    echo "  repo_type : $repo_type"
    echo "  subdir    : $subdir"
    echo "  ->        : $local_dir"
    echo "============================================================"

    python - "$repo_id" "$subdir" "$local_dir" "$repo_type" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
subdir = sys.argv[2]
local_dir = sys.argv[3]
repo_type = sys.argv[4]

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    allow_patterns=[f"{subdir}/**"],
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
PY

    # If snapshot_download created local_dir/subdir/*, move contents up one level.
    if [ -d "$local_dir/$subdir" ]; then
        shopt -s dotglob nullglob
        mv "$local_dir/$subdir"/* "$local_dir"/
        rmdir "$local_dir/$subdir"
        shopt -u dotglob nullglob
    fi
}

for env_name in "${ENVS[@]}"; do
    # -------------------------
    # Datasets
    # -------------------------
    download_dir \
        "$PRETRAIN_DATA_REPO" \
        "${env_name}_img" \
        "$DICE_RL_DATA_DIR/robomimic/${env_name}-img/ph_pretrain" \
        "dataset"

    download_dir \
        "$PRETRAIN_DATA_REPO" \
        "${env_name}_low_dim" \
        "$DICE_RL_DATA_DIR/robomimic/${env_name}-low-dim/ph_pretrain" \
        "dataset"

    download_dir \
        "$FINETUNE_DATA_REPO" \
        "${env_name}_img" \
        "$DICE_RL_DATA_DIR/robomimic/${env_name}-img/ph_finetune" \
        "dataset"

    download_dir \
        "$FINETUNE_DATA_REPO" \
        "${env_name}_low_dim" \
        "$DICE_RL_DATA_DIR/robomimic/${env_name}-low-dim/ph_finetune" \
        "dataset"

    # -------------------------
    # Checkpoints
    # -------------------------
    download_dir \
        "$PRETRAIN_CKPT_REPO" \
        "${env_name}_img" \
        "$DICE_RL_LOG_DIR/robomimic-pretrain/pretrained_bc_policy_${env_name}_img" \
        "model"

    download_dir \
        "$PRETRAIN_CKPT_REPO" \
        "${env_name}_low_dim" \
        "$DICE_RL_LOG_DIR/robomimic-pretrain/pretrained_bc_policy_${env_name}_low_dim" \
        "model"

    download_dir \
        "$FINETUNE_CKPT_REPO" \
        "${env_name}_img" \
        "$DICE_RL_LOG_DIR/robomimic-finetune/finetuned_rl_policy_${env_name}_img" \
        "model"

    download_dir \
        "$FINETUNE_CKPT_REPO" \
        "${env_name}_low_dim" \
        "$DICE_RL_LOG_DIR/robomimic-finetune/finetuned_rl_policy_${env_name}_low_dim" \
        "model"
done

echo
echo "All downloads finished."