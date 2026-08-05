#!/bin/bash
#
# scripts/run_dssm_train_and_export.sh
#
# 在 L40S (x86) 上全量训练 DSSM 并导出 item 向量 / 用户画像。
# 运行前请确保已在 pairec-multi-recall-ranking 分支且代码最新：
#   git pull gitcode pairec-multi-recall-ranking
#
# 产物：
#   dssm_out/vocab.json
#   dssm_out/dssm_model.pt
#   dssm_out/export/item_vectors.npy
#   dssm_out/export/item_ids.json
#   dssm_out/export/item_categories.json
#   dssm_out/export/user_profiles.json

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[dssm-pipeline] working dir: $(pwd)"
echo "[dssm-pipeline] git branch:  $(git rev-parse --abbrev-ref HEAD)"

# 默认使用一张空闲 GPU；如 GPU 5 被占用，请改为 6/7
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

# 数据路径可通过环境变量覆盖；188(ARM 4090D) 用 /home/zcx/workspace/pairec4tigerllm/data/ctr_data_1M.csv
CSV_PATH="${CSV_PATH:-/home/workspace/zcx/pairec4tigerllm/data/ctr_data_1M.csv}"
OUT_DIR="dssm_out"
EXPORT_DIR="${OUT_DIR}/export"

mkdir -p "${OUT_DIR}" "${EXPORT_DIR}"

echo "[dssm-pipeline] step 1/2: train DSSM on CSV=${CSV_PATH}"
python -m training.dssm.train \
    --csv_path "${CSV_PATH}" \
    --vocab_path "${OUT_DIR}/vocab.json" \
    --checkpoint_dir "${OUT_DIR}" \
    --train_rows 1000000 \
    --vocab_rows 1000000 \
    --batch_size 4096 \
    --epochs 10 \
    --embed_dim 64 \
    --out_dim 64 \
    --learning_rate 1e-3 \
    --temperature 0.05 \
    --device cuda \
    --log_every 50

echo "[dssm-pipeline] step 2/2: export embeddings"
python -m training.dssm.export_embeddings \
    --checkpoint "${OUT_DIR}/dssm_model.pt" \
    --vocab_path "${OUT_DIR}/vocab.json" \
    --csv_path "${CSV_PATH}" \
    --out_dir "${EXPORT_DIR}" \
    --profile_rows 1000000 \
    --batch_size 8192 \
    --device cuda

echo "[dssm-pipeline] done. artifacts:"
ls -lh "${OUT_DIR}"/vocab.json "${OUT_DIR}"/dssm_model.pt "${EXPORT_DIR}"/*

echo ""
echo "[dssm-pipeline] 下一步把产物拷到 ARM master，例如："
echo "  scp -r dssm_out zcx@141.61.91.189:/home/zcx/workspace/pairec4tigerllm/"
