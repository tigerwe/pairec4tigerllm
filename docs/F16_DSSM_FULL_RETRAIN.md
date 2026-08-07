# DSSM Full Retraining

The original `dssm_out` artifacts are structurally valid but were built from
only the first 1,000,000 rows. The audit measured 0.831% training-row coverage,
12.360% unique target-item coverage, and 80% sampled user OOV. Keep those
artifacts as the deployed engineering baseline; do not overwrite them.

## New training contract

- Output is isolated in `dssm_full_out`.
- `TRAIN_ROWS=0`, `VOCAB_ROWS=0`, and `PROFILE_ROWS=0` mean the full CSV.
- Rows use a deterministic 90/10 train/validation split.
- Training chunks are deterministically shuffled for each epoch.
- Repeated item IDs in a batch are treated as multiple positives.
- The best validation-loss checkpoint is exported.
- Training reports validation in-batch Recall@10/50/100.
- Post-export evaluation samples held-out positive rows across the complete CSV
  and searches every exported item vector for full-corpus Recall@10/50/100 and
  MRR@100.

## Worker1 execution

```bash
cd /home/zcx/workspace/pairec4tigerllm
git pull --ff-only gitcode pairec-multi-recall-ranking

REPO_DIR=/home/zcx/workspace/pairec4tigerllm \
GPU_DEVICE=all \
EPOCHS=3 \
EVAL_QUERY_LIMIT=1000 \
  bash scripts/run_dssm_full_retrain_worker1.sh \
  | tee /tmp/dssm-full-retrain.log
```

The initial run intentionally has no hard Recall@50 threshold because the old
model has no comparable held-out metric. Use the first complete run as the
baseline, then set `MIN_RECALL_AT_50` for subsequent acceptance runs.

If training completed but export or evaluation failed, restart from that stage:

```bash
START_STAGE=export \
  bash scripts/run_dssm_full_retrain_worker1.sh

START_STAGE=evaluate EVAL_QUERY_BATCH_SIZE=16 \
  bash scripts/run_dssm_full_retrain_worker1.sh
```

`ALLOW_EXISTING_OUTPUT=1 START_STAGE=train` reuses a completed vocabulary but
starts model optimization from scratch; it is not an optimizer-state resume.

## DeepFM dependency

The existing DeepFM checkpoint uses the old DSSM vocabulary. After the full
DSSM run passes, retrain DeepFM into another isolated directory:

```bash
CSV_PATH=/workspace/data/ctr_data_1M.csv \
DSSM_VOCAB_PATH=dssm_full_out/vocab.json \
OUTPUT_DIR=deepfm_full_vocab_out \
  bash scripts/run_deepfm_train.sh
```

Do not replace Milvus or the ranking service until both new model families have
completed their validation steps.
