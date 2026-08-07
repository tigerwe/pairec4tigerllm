# DSSM Full Retraining

The original `dssm_out` artifacts are structurally valid but were built from
only the first 1,000,000 rows. The audit measured 0.831% training-row coverage,
12.360% unique target-item coverage, and 80% sampled user OOV. Keep those
artifacts as the deployed engineering baseline; do not overwrite them.

The first full-data baseline in `dssm_full_out` fixed coverage but reached only
Recall@10/50/100 = 0.14%/0.70%/1.14% on 10,000 held-out queries against
2,354,248 items. Keep it as the positive-row-candidate quality baseline.

## New training contract

- Output is isolated in `dssm_all_candidates_out`.
- `TRAIN_ROWS=0`, `VOCAB_ROWS=0`, and `PROFILE_ROWS=0` mean the full CSV.
- Rows use a deterministic 90/10 train/validation split.
- Training chunks are deterministically shuffled for each epoch.
- Positive rows provide queries, while all valid batch rows provide item
  candidates. This includes click=0 exposures as natural negatives.
- Repeated item IDs in any candidate row are treated as multiple positives.
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
EVAL_QUERY_LIMIT=10000 \
  bash scripts/run_dssm_full_retrain_worker1.sh \
  | tee /tmp/dssm-full-retrain.log
```

The initial all-candidate run intentionally has no hard Recall@50 threshold.
Compare it with the fixed `dssm_full_out` baseline above before choosing a
quality gate. The training summary must report non-zero
`train_unclicked_candidate_rows`; otherwise click=0 exposures were not
used.

For a code-path A/B using the old positive-only objective, use a separate
output directory:

```bash
OUTPUT_DIR=/home/zcx/workspace/pairec4tigerllm/dssm_positive_candidates_out \
CANDIDATE_MODE=positive_rows \
EVAL_QUERY_LIMIT=10000 \
  bash scripts/run_dssm_full_retrain_worker1.sh
```

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
DSSM_VOCAB_PATH=dssm_all_candidates_out/vocab.json \
OUTPUT_DIR=deepfm_full_vocab_out \
  bash scripts/run_deepfm_train.sh
```

Do not replace Milvus or the ranking service until both new model families have
completed their validation steps.
