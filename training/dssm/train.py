"""Train DSSM with deterministic validation and multi-positive in-batch loss."""

import argparse
import json
import math
import os
import time

import torch

from training.deepfm.dataset import iter_split_batches
from training.dssm.dataset import build_or_load_vocab, vocab_sizes
from training.dssm.model import (DSSM, in_batch_recall_counts,
                                 in_batch_softmax_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--train_rows", type=int, default=0,
                        help="0 uses the full CSV")
    parser.add_argument("--vocab_rows", type=int, default=0,
                        help="0 builds the vocabulary from the full CSV")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--load_checkpoint", default="")
    return parser.parse_args()


def to_tensors(batch, device):
    return {key: torch.from_numpy(value).to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, args, vocab):
    model.eval()
    loss_sum = 0.0
    positive_queries = 0
    batches = 0
    skipped = 0
    hit_counts = {10: 0, 50: 0, 100: 0}
    for batch_np in iter_split_batches(
            args.csv_path, vocab, args.batch_size, "val",
            args.val_fraction, args.seed, args.train_rows):
        batch = to_tensors(batch_np, args.device)
        user_vec, item_vec = model(batch)
        loss = in_batch_softmax_loss(
            user_vec, item_vec, batch["label"], args.temperature,
            batch["item_id"]
        )
        if loss is None:
            skipped += 1
            continue
        counts = in_batch_recall_counts(
            user_vec, item_vec, batch["label"], batch["item_id"]
        )
        queries = counts["queries"]
        loss_sum += float(loss.item()) * queries
        positive_queries += queries
        batches += 1
        for k in hit_counts:
            hit_counts[k] += counts[f"hits_at_{k}"]
    if not positive_queries:
        raise RuntimeError("validation split has fewer than two positive examples")
    return {
        "val_loss": loss_sum / positive_queries,
        "val_positive_queries": positive_queries,
        "val_batches": batches,
        "val_skipped_batches": skipped,
        **{
            f"val_in_batch_recall_at_{k}": hit_counts[k] / positive_queries
            for k in hit_counts
        },
    }


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[DSSM] building/loading vocab rows={args.vocab_rows or 'all'}")
    vocab = build_or_load_vocab(args.csv_path, args.vocab_path, args.vocab_rows)
    sizes = vocab_sizes(vocab)
    print(f"[DSSM] vocab sizes: {sizes}")
    model = DSSM(sizes, embed_dim=args.embed_dim, out_dim=args.out_dim).to(args.device)
    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location=args.device)
        model.load_state_dict(state["model"])
        print(f"[DSSM] loaded checkpoint {args.load_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, "dssm_model.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        positive_queries = 0
        batches = 0
        skipped = 0
        examples = 0
        for batch_np in iter_split_batches(
                args.csv_path, vocab, args.batch_size, "train",
                args.val_fraction, args.seed, args.train_rows,
                shuffle=True, shuffle_seed=args.seed + epoch):
            batch = to_tensors(batch_np, args.device)
            examples += len(batch_np["label"])
            user_vec, item_vec = model(batch)
            loss = in_batch_softmax_loss(
                user_vec, item_vec, batch["label"], args.temperature,
                batch["item_id"]
            )
            if loss is None:
                skipped += 1
                continue
            queries = int((batch["label"] > 0.5).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * queries
            positive_queries += queries
            batches += 1
            if batches % args.log_every == 0:
                print(f"[DSSM] epoch={epoch} step={batches} "
                      f"train_loss={loss_sum / positive_queries:.6f}")
        if not positive_queries:
            raise RuntimeError("training split has fewer than two positive examples")
        metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / positive_queries,
            "train_examples": examples,
            "train_positive_queries": positive_queries,
            "train_batches": batches,
            "train_skipped_batches": skipped,
        }
        metrics.update(evaluate(model, args, vocab))
        metrics["elapsed_seconds"] = round(time.time() - started, 3)
        history.append(metrics)
        print("[DSSM] " + json.dumps(metrics, sort_keys=True))

        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "config": {
                    "embed_dim": args.embed_dim,
                    "out_dim": args.out_dim,
                    "temperature": args.temperature,
                    "vocab_path": args.vocab_path,
                    "vocab_sizes": sizes,
                    "val_fraction": args.val_fraction,
                    "seed": args.seed,
                },
                "avg_loss": metrics["train_loss"],
                "val_loss": metrics["val_loss"],
                "val_in_batch_recall_at_50": metrics["val_in_batch_recall_at_50"],
            }, checkpoint_path)
            print(f"[DSSM] saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[DSSM] early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    summary = {
        "status": "PASS",
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
        "csv_path": os.path.abspath(args.csv_path),
        "vocab_path": os.path.abspath(args.vocab_path),
        "train_rows": args.train_rows,
        "vocab_rows": args.vocab_rows,
        "quality_gate": False,
    }
    summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"DSSM_TRAINING_OK checkpoint={checkpoint_path} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
