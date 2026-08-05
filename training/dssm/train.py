# training/dssm/train.py
#
# DSSM 双塔召回模型训练.
#
# 用法 (本机 smoke):
#   python -m training.dssm.train \
#     --csv_path /home/vivwimp/Tenrec/ctr_data_1M.csv \
#     --vocab_path checkpoints/dssm/vocab.json \
#     --checkpoint_dir checkpoints/dssm \
#     --train_rows 200000 --batch_size 4096 --epochs 1
#
# 远程 ARM 4090D:
#   python -m training.dssm.train --train_rows 50000000 --epochs 2 ...

import argparse
import json
import os
import time

import torch

from training.dssm.dataset import (build_or_load_vocab, iter_batches,
                                   vocab_sizes)
from training.dssm.model import DSSM, in_batch_softmax_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default="/home/vivwimp/Tenrec/ctr_data_1M.csv")
    p.add_argument("--vocab_path", default="checkpoints/dssm/vocab.json")
    p.add_argument("--checkpoint_dir", default="checkpoints/dssm")
    p.add_argument("--train_rows", type=int, default=5_000_000)
    p.add_argument("--vocab_rows", type=int, default=5_000_000,
                   help="构建词表扫描的行数, 默认与 train_rows 一致")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--out_dim", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--load_checkpoint", default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[DSSM] building/loading vocab from {args.csv_path} (rows={args.vocab_rows})")
    vocab = build_or_load_vocab(args.csv_path, args.vocab_path, args.vocab_rows)
    sizes = vocab_sizes(vocab)
    print(f"[DSSM] vocab sizes: {sizes}")

    model = DSSM(sizes, embed_dim=args.embed_dim, out_dim=args.out_dim)
    model.to(args.device)
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        state = torch.load(args.load_checkpoint, map_location=args.device)
        model.load_state_dict(state["model"])
        print(f"[DSSM] loaded checkpoint {args.load_checkpoint} (epoch={state.get('epoch')})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_batches, skipped = 0.0, 0, 0
        for batch_np in iter_batches(args.csv_path, vocab, args.train_rows,
                                     args.batch_size):
            batch = {k: torch.from_numpy(v).to(args.device)
                     for k, v in batch_np.items()}
            user_vec, item_vec = model(batch)
            loss = in_batch_softmax_loss(user_vec, item_vec, batch["label"],
                                         args.temperature)
            if loss is None:
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_batches += 1
            if total_batches % args.log_every == 0:
                print(f"[DSSM] epoch={epoch} step={total_batches} "
                      f"loss={total_loss / total_batches:.4f} "
                      f"elapsed={time.time() - t0:.1f}s")

        avg = total_loss / max(total_batches, 1)
        print(f"[DSSM] epoch={epoch} done avg_loss={avg:.4f} "
              f"batches={total_batches} skipped={skipped} "
              f"elapsed={time.time() - t0:.1f}s")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config": {
                "embed_dim": args.embed_dim,
                "out_dim": args.out_dim,
                "temperature": args.temperature,
                "vocab_path": args.vocab_path,
                "vocab_sizes": sizes,
            },
            "avg_loss": avg,
        }
        path = os.path.join(args.checkpoint_dir, "dssm_model.pt")
        torch.save(ckpt, path)
        print(f"[DSSM] saved checkpoint: {path}")


if __name__ == "__main__":
    main()
