#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"

mkdir -p "${MODEL_DIR}/remoteclip" "${MODEL_DIR}/changeformer"

if [[ ! -f "${MODEL_DIR}/remoteclip/RemoteCLIP-RN50.pt" ]]; then
  curl -L --fail --retry 3 \
    "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-RN50.pt?download=true" \
    -o "${MODEL_DIR}/remoteclip/RemoteCLIP-RN50.pt"
fi

if [[ ! -d "${MODEL_DIR}/changeformer/source/.git" ]]; then
  git clone --depth 1 https://github.com/wgcban/ChangeFormer.git \
    "${MODEL_DIR}/changeformer/source"
fi

if [[ ! -f "${MODEL_DIR}/changeformer/best_ckpt.pt" ]]; then
  ARCHIVE_PATH="${MODEL_DIR}/changeformer/official-levir.zip"
  curl -L --fail --retry 3 \
    "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip" \
    -o "${ARCHIVE_PATH}"
  unzip -j -o "${ARCHIVE_PATH}" '*/best_ckpt.pt' -d "${MODEL_DIR}/changeformer"
fi

printf '%s\n' "RemoteCLIP and ChangeFormer are ready under ${MODEL_DIR}."
