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

The first all-candidate remote run completed three epochs and improved every
full-corpus metric, but not enough to replace the deployed vectors:

| metric | positive-only baseline | all-candidate epoch 3 |
| --- | ---: | ---: |
| Recall@10 | 0.14% | 0.20% |
| Recall@50 | 0.70% | 0.79% |
| Recall@100 | 1.14% | 1.40% |
| MRR@100 | 0.000924 | 0.001154 |

Epoch 3 was still the best validation-loss checkpoint. Before adding another
negative-sampling strategy, continue it to six total epochs. The epoch-3
checkpoint predates optimizer-state persistence, so this first continuation is
explicitly a weights-only continuation with AdamW state reset. New runs save
the lightweight best model in `dssm_model.pt` and the complete latest model +
optimizer state in `dssm_last.pt`; resume from `dssm_last.pt` after the first
continued epoch when exact optimizer restoration is required.

```bash
cd /home/zcx/workspace/pairec4tigerllm
cp dssm_all_candidates_out/retrieval_evaluation.json \
  dssm_all_candidates_out/retrieval_evaluation.epoch3.json
cp dssm_all_candidates_out/training_summary.json \
  dssm_all_candidates_out/training_summary.epoch3.json
cp --reflink=auto dssm_all_candidates_out/dssm_model.pt \
  dssm_all_candidates_out/dssm_model.epoch3.pt

REPO_DIR=/home/zcx/workspace/pairec4tigerllm \
OUTPUT_DIR=/home/zcx/workspace/pairec4tigerllm/dssm_all_candidates_out \
LOAD_CHECKPOINT=/home/zcx/workspace/pairec4tigerllm/dssm_all_candidates_out/dssm_model.pt \
ALLOW_EXISTING_OUTPUT=1 \
CANDIDATE_MODE=all_rows \
EPOCHS=6 \
PATIENCE=3 \
EVAL_QUERY_LIMIT=10000 \
  bash scripts/run_dssm_full_retrain_worker1.sh \
  | tee /tmp/dssm-all-candidates-continue-e6.log
```

`EPOCHS=6` is the total target, so this command runs epochs 4-6 only. The log
must say `checkpoint_epoch=3 start_epoch=4` and `optimizer_state=reset`.

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
