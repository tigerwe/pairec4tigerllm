"""Evaluate sampled held-out DSSM queries against the full exported item corpus."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from training.dssm.dataset import HIST_COLUMNS, NA_VALUES, encode_chunk
from training.dssm.model import DSSM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_limit", type=int, default=1000)
    parser.add_argument("--query_batch_size", type=int, default=32)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_recall_at_50", type=float, default=0.0)
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def split_buckets(indices, seed):
    return (indices * np.uint64(1_103_515_245) + np.uint64(seed)) % 10_000


def sample_validation_positives(csv_path, limit, val_fraction, seed):
    """Keep a deterministic priority sample distributed across the full CSV."""
    if limit < 1:
        raise ValueError("query_limit must be positive")
    threshold = int(val_fraction * 10_000)
    columns = ["user_id", "item_id", "click", "video_category", "gender", "age"] + HIST_COLUMNS
    reservoir = None
    row_offset = 0
    positive_count = 0
    with pd.read_csv(csv_path, usecols=columns, chunksize=200_000,
                     na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            indices = np.arange(row_offset, row_offset + len(chunk), dtype=np.uint64)
            clicks = pd.to_numeric(chunk["click"], errors="coerce").fillna(0).to_numpy()
            mask = (split_buckets(indices, seed) < threshold) & (clicks > 0.5)
            selected = chunk.loc[mask].copy()
            selected_indices = indices[mask]
            positive_count += len(selected)
            row_offset += len(chunk)
            if selected.empty:
                continue
            selected["_priority"] = (
                selected_indices * np.uint64(6_364_136_223_846_793_005) +
                np.uint64(seed)
            )
            selected = selected.nsmallest(limit, "_priority")
            if reservoir is None:
                reservoir = selected
            else:
                reservoir = pd.concat([reservoir, selected], ignore_index=True)
                reservoir = reservoir.nsmallest(limit, "_priority")
    if reservoir is None or reservoir.empty:
        raise RuntimeError("held-out split contains no positive examples")
    return reservoir.drop(columns=["_priority"]).reset_index(drop=True), {
        "csv_rows_scanned": row_offset,
        "heldout_positive_rows": positive_count,
        "sampled_queries": len(reservoir),
    }


def evaluate(args):
    with open(args.vocab_path, encoding="utf-8") as handle:
        vocab = json.load(handle)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = checkpoint["config"]
    model = DSSM(
        config["vocab_sizes"],
        embed_dim=config["embed_dim"],
        out_dim=config["out_dim"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows, sampling = sample_validation_positives(
        args.csv_path, args.query_limit, args.val_fraction, args.seed
    )
    features = encode_chunk(rows, vocab)
    raw_targets = pd.to_numeric(rows["item_id"], errors="coerce").fillna(-1).astype(np.int64)

    vectors = np.load(os.path.join(args.export_dir, "item_vectors.npy"))
    with open(os.path.join(args.export_dir, "item_ids.json"), encoding="utf-8") as handle:
        item_ids = [int(value) for value in json.load(handle)]
    if vectors.ndim != 2 or len(vectors) != len(item_ids):
        raise RuntimeError("exported item vectors and IDs are inconsistent")
    item_row = {item_id: row for row, item_id in enumerate(item_ids)}
    target_rows = np.array([item_row.get(int(value), -1) for value in raw_targets], dtype=np.int64)
    eligible = target_rows >= 0
    missing_targets = int((~eligible).sum())
    if not eligible.any():
        raise RuntimeError("none of the sampled targets exists in the exported item corpus")

    candidates = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(args.device)
    candidates = F.normalize(candidates, dim=1)
    max_k = min(100, len(candidates))
    hit_counts = {10: 0, 50: 0, 100: 0}
    reciprocal_rank_sum = 0.0
    evaluated = 0
    eligible_indices = np.flatnonzero(eligible)
    with torch.no_grad():
        for start in range(0, len(eligible_indices), args.query_batch_size):
            selected = eligible_indices[start:start + args.query_batch_size]
            user = model.user_tower(
                torch.from_numpy(features["user_id"][selected]).to(args.device),
                torch.from_numpy(features["gender"][selected]).to(args.device),
                torch.from_numpy(features["age"][selected]).to(args.device),
                torch.from_numpy(features["hist"][selected]).to(args.device),
            )
            scores = F.normalize(user, dim=1) @ candidates.t()
            top_rows = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
            targets = target_rows[selected]
            matches = top_rows == targets[:, None]
            for k in hit_counts:
                hit_counts[k] += int(matches[:, :min(k, max_k)].any(axis=1).sum())
            found = matches.any(axis=1)
            if found.any():
                ranks = matches[found].argmax(axis=1) + 1
                reciprocal_rank_sum += float((1.0 / ranks).sum())
            evaluated += len(selected)

    metrics = {
        f"full_corpus_recall_at_{k}": hit_counts[k] / evaluated
        for k in hit_counts
    }
    metrics["full_corpus_mrr_at_100"] = reciprocal_rank_sum / evaluated
    gate_enabled = args.min_recall_at_50 > 0
    passed = metrics["full_corpus_recall_at_50"] >= args.min_recall_at_50
    if not gate_enabled:
        classification = "METRICS_RECORDED_NO_QUALITY_GATE"
    elif passed:
        classification = "FULL_CORPUS_QUALITY_GATE_PASS"
    else:
        classification = "FULL_CORPUS_QUALITY_GATE_FAIL"
    return {
        "status": "PASS" if passed else "FAIL",
        "classification": classification,
        "quality_gate_enabled": gate_enabled,
        "quality_gate_passed": passed if gate_enabled else None,
        "min_recall_at_50": args.min_recall_at_50,
        "item_corpus_size": len(item_ids),
        "evaluated_queries": evaluated,
        "sampled_target_missing_from_corpus": missing_targets,
        "sampling": sampling,
        "metrics": metrics,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def main():
    args = parse_args()
    report = evaluate(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[DSSM] " + json.dumps(report, sort_keys=True))
    if report["quality_gate_enabled"]:
        marker = f"DSSM_FULL_CORPUS_EVALUATION_{report['status']}"
    else:
        marker = "DSSM_FULL_CORPUS_EVALUATION_RECORDED"
    print(f"{marker} output={args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
