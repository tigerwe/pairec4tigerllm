"""Streaming deterministic train/validation batches for DeepFM."""

import numpy as np
import pandas as pd

from training.dssm.dataset import NA_VALUES, encode_chunk


def _append(buffer, features):
    if buffer is None:
        return features
    return {key: np.concatenate([buffer[key], features[key]]) for key in features}


def iter_split_batches(csv_path, vocab, batch_size, split, val_fraction=0.1,
                       seed=20260807, max_rows=0, chunk_size=200_000,
                       shuffle=False, shuffle_seed=0):
    """Yield one deterministic side of a streaming row-level 90/10 split."""
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    threshold = int(val_fraction * 10_000)
    buffer = None
    row_offset = 0
    rng = np.random.default_rng(shuffle_seed) if shuffle else None
    with pd.read_csv(csv_path, chunksize=chunk_size, na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            if max_rows and row_offset >= max_rows:
                break
            if max_rows and row_offset + len(chunk) > max_rows:
                chunk = chunk.iloc[:max_rows - row_offset]
            count = len(chunk)
            indices = np.arange(row_offset, row_offset + count, dtype=np.uint64)
            buckets = (indices * np.uint64(1_103_515_245) + np.uint64(seed)) % 10_000
            val_mask = buckets < threshold
            mask = val_mask if split == "val" else ~val_mask
            selected = chunk.loc[mask]
            row_offset += count
            if selected.empty:
                continue
            features = encode_chunk(selected, vocab)
            if rng is not None:
                order = rng.permutation(len(features["label"]))
                features = {key: value[order] for key, value in features.items()}
            buffer = _append(buffer, features)
            while len(buffer["label"]) >= batch_size:
                yield {key: value[:batch_size] for key, value in buffer.items()}
                buffer = {key: value[batch_size:] for key, value in buffer.items()}

    if buffer is not None and len(buffer["label"]):
        yield buffer
