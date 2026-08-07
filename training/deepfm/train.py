"""Train and export the PyTorch DeepFM ranking model."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.deepfm.dataset import iter_split_batches
from training.deepfm.model import DeepFM
from training.dssm.dataset import (build_or_load_vocab, export_item_categories,
                                   export_user_profiles, vocab_sizes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--vocab_path", required=True,
                        help="Existing DSSM vocab; built only when absent")
    parser.add_argument("--output_dir", default="deepfm_out")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="0 uses the full CSV")
    parser.add_argument("--vocab_rows", type=int, default=50_000_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--hidden_dims", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def _to_tensors(batch, device):
    return {key: torch.from_numpy(value).to(device) for key, value in batch.items()}


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) /
                 (positives * negatives))


@torch.no_grad()
def evaluate(model, args, vocab):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels = []
    scores = []
    for batch_np in iter_split_batches(
            args.csv_path, vocab, args.batch_size, "val",
            args.val_fraction, args.seed, args.max_rows):
        batch = _to_tensors(batch_np, args.device)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="sum")
        total_loss += loss.item()
        total_examples += len(batch_np["label"])
        labels.append(batch_np["label"])
        scores.append(torch.sigmoid(logits).cpu().numpy())
    if total_examples == 0:
        raise RuntimeError("validation split is empty")
    all_labels = np.concatenate(labels)
    all_scores = np.concatenate(scores)
    return total_loss / total_examples, binary_auc(all_labels, all_scores), total_examples


def main():
    args = parse_args()
    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("epochs must be in [1, 10]")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value)
    if not hidden_dims:
        raise ValueError("hidden_dims cannot be empty")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    vocab = build_or_load_vocab(args.csv_path, args.vocab_path, args.vocab_rows)
    sizes = vocab_sizes(vocab)
    feature_vocab_path = os.path.join(args.output_dir, "feature_vocab.json")
    with open(feature_vocab_path, "w", encoding="utf-8") as handle:
        json.dump(vocab, handle)

    model = DeepFM(sizes, args.embed_dim, hidden_dims, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    model_config = model.config(sizes)
    model_config.update({
        "feature_names": ["user_id", "item_id", "video_category", "gender", "age"] +
                         [f"hist_{index}" for index in range(1, 11)],
        "label": "click",
        "seed": args.seed,
    })
    with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)

    history = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = os.path.join(args.output_dir, "deepfm_best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        train_loss = 0.0
        train_examples = 0
        steps = 0
        for batch_np in iter_split_batches(
                args.csv_path, vocab, args.batch_size, "train",
                args.val_fraction, args.seed, args.max_rows):
            batch = _to_tensors(batch_np, args.device)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = len(batch_np["label"])
            train_loss += loss.item() * count
            train_examples += count
            steps += 1
            if steps % args.log_every == 0:
                print(f"[DeepFM] epoch={epoch} step={steps} "
                      f"train_logloss={train_loss / train_examples:.6f}")

        val_loss, val_auc, val_examples = evaluate(model, args, vocab)
        metrics = {
            "epoch": epoch,
            "train_logloss": train_loss / max(train_examples, 1),
            "val_logloss": val_loss,
            "val_auc": val_auc,
            "train_examples": train_examples,
            "val_examples": val_examples,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        history.append(metrics)
        print("[DeepFM] " + json.dumps(metrics, sort_keys=True))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "model_config": model_config,
                "epoch": epoch,
                "val_logloss": val_loss,
                "val_auc": val_auc,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[DeepFM] early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    export_rows = args.max_rows or 2**63 - 1
    category_count = export_item_categories(
        args.csv_path, export_rows, os.path.join(args.output_dir, "item_categories.json"))
    profile_count = export_user_profiles(
        args.csv_path, vocab, export_rows, os.path.join(args.output_dir, "user_profiles.json"))
    summary = {
        "status": "PASS",
        "best_epoch": best_epoch,
        "best_val_logloss": best_loss,
        "best_val_auc": next(item["val_auc"] for item in history
                             if item["epoch"] == best_epoch),
        "history": history,
        "csv_path": os.path.abspath(args.csv_path),
        "vocab_source": os.path.abspath(args.vocab_path),
        "max_rows": args.max_rows,
        "item_category_count": category_count,
        "user_profile_count": profile_count,
        "quality_gate": False,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"DEEPFM_TRAINING_OK checkpoint={checkpoint_path} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
