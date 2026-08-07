# training/dssm/dataset.py
#
# Tenrec CTR 数据集读取与词表构建.
# 数据: /home/vivwimp/Tenrec/ctr_data_1M.csv
# 列: user_id,item_id,click,follow,like,share,video_category,
#     watching_times,gender,age,hist_1..hist_10
#
# 用法:
#   vocab = build_or_load_vocab(csv_path, vocab_path, max_rows)
#   for batch in iter_batches(csv_path, vocab, max_rows, batch_size): ...

import json
import os

import numpy as np
import pandas as pd

CSV_COLUMNS = [
    "user_id", "item_id", "click", "follow", "like", "share",
    "video_category", "watching_times", "gender", "age",
    "hist_1", "hist_2", "hist_3", "hist_4", "hist_5",
    "hist_6", "hist_7", "hist_8", "hist_9", "hist_10",
]
HIST_COLUMNS = [f"hist_{i}" for i in range(1, 11)]

# index 0 留给 padding/OOV
PAD_IDX = 0
NA_VALUES = ["\\N"]


def _int_array(values):
    """把可能含 \\N/NaN 的列转成 int64, 无效值为 -1."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").fillna(-1)
    return s.values.astype(np.int64)


def _new_vocab():
    return {
        "user2idx": {},
        "item2idx": {},
        "cat2idx": {},
        "gender2idx": {},
        "age2idx": {},
    }


def _truncate_chunk(chunk, scanned, max_rows):
    if max_rows and scanned + len(chunk) > max_rows:
        return chunk.iloc[:max_rows - scanned]
    return chunk


def build_or_load_vocab(csv_path, vocab_path, max_rows=5_000_000):
    """扫描 CSV 构建词表; max_rows=0 表示扫描完整文件."""
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            return json.load(f)

    vocab = _new_vocab()

    def add(mapping, values):
        for v in np.unique(_int_array(values)):
            if v < 0:
                continue
            key = str(int(v))
            if key not in mapping:
                mapping[key] = len(mapping) + 1  # 0 = PAD/OOV

    scanned = 0
    usecols = ["user_id", "item_id", "video_category", "gender", "age"] + HIST_COLUMNS
    with pd.read_csv(csv_path, usecols=usecols, chunksize=500_000,
                     na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            chunk = _truncate_chunk(chunk, scanned, max_rows)
            add(vocab["user2idx"], chunk["user_id"].values)
            add(vocab["item2idx"], chunk["item_id"].values)
            add(vocab["cat2idx"], chunk["video_category"].values)
            add(vocab["gender2idx"], chunk["gender"].values)
            add(vocab["age2idx"], chunk["age"].values)
            for col in HIST_COLUMNS:
                add(vocab["item2idx"], chunk[col].values)
            scanned += len(chunk)
            if max_rows and scanned >= max_rows:
                break

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    return vocab


def vocab_sizes(vocab):
    return {
        "user": len(vocab["user2idx"]) + 1,
        "item": len(vocab["item2idx"]) + 1,
        "cat": len(vocab["cat2idx"]) + 1,
        "gender": len(vocab["gender2idx"]) + 1,
        "age": len(vocab["age2idx"]) + 1,
    }


def _map_ids(mapping, values):
    out = []
    for v in _int_array(values):
        out.append(mapping.get(str(int(v)), PAD_IDX) if v >= 0 else PAD_IDX)
    return np.array(out, dtype=np.int64)


def encode_chunk(chunk, vocab):
    """把一个 pandas chunk 编码成 numpy 特征字典."""
    hist = np.stack(
        [_map_ids(vocab["item2idx"], chunk[col].values) for col in HIST_COLUMNS],
        axis=1,
    )
    return {
        "user_id": _map_ids(vocab["user2idx"], chunk["user_id"].values),
        "gender": _map_ids(vocab["gender2idx"], chunk["gender"].values),
        "age": _map_ids(vocab["age2idx"], chunk["age"].values),
        "hist": hist,  # (N, 10) item idx, 0 = padding
        "item_id": _map_ids(vocab["item2idx"], chunk["item_id"].values),
        "category": _map_ids(vocab["cat2idx"], chunk["video_category"].values),
        "label": pd.to_numeric(chunk["click"], errors="coerce").fillna(0).values.astype(np.float32),
    }


def iter_batches(csv_path, vocab, max_rows, batch_size, start_row=0):
    """流式遍历 CSV, 产出 encode 后的 batch 字典 (numpy)."""
    buffer = None
    seen = 0
    with pd.read_csv(csv_path, chunksize=batch_size * 8,
                     skiprows=range(1, start_row + 1),
                     na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            feats = encode_chunk(chunk, vocab)
            n = len(feats["label"])
            seen += n
            if max_rows and seen > max_rows:
                keep = n - (seen - max_rows)
                if keep <= 0:
                    break
                feats = {k: v[:keep] for k, v in feats.items()}
            if buffer is None:
                buffer = feats
            else:
                buffer = {k: np.concatenate([buffer[k], feats[k]]) for k in feats}
            while len(buffer["label"]) >= batch_size:
                batch = {k: v[:batch_size] for k, v in buffer.items()}
                buffer = {k: v[batch_size:] for k, v in buffer.items()}
                yield batch
            if max_rows and seen >= max_rows:
                break
    if buffer is not None and len(buffer["label"]):
        yield buffer


def export_item_categories(csv_path, max_rows, out_path):
    """扫描 CSV 构建 item_id -> video_category 映射 (取第一次出现的值)."""
    mapping = {}
    scanned = 0
    usecols = ["item_id", "video_category"]
    with pd.read_csv(csv_path, usecols=usecols, chunksize=500_000,
                     na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            chunk = _truncate_chunk(chunk, scanned, max_rows)
            items = _int_array(chunk["item_id"].values)
            cats = _int_array(chunk["video_category"].values)
            for item_id, cat in zip(items, cats):
                if item_id < 0 or cat < 0:
                    continue
                key = str(int(item_id))
                if key not in mapping:
                    mapping[key] = int(cat)
            scanned += len(chunk)
            if max_rows and scanned >= max_rows:
                break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f)
    return len(mapping)


def export_user_profiles(csv_path, vocab, max_rows, out_path):
    """为词表内每个 user 保留最近一次出现的画像特征, 供 user tower 服务查询."""
    profiles = {}
    scanned = 0
    usecols = ["user_id", "gender", "age"] + HIST_COLUMNS
    with pd.read_csv(csv_path, usecols=usecols, chunksize=500_000,
                     na_values=NA_VALUES) as chunks:
        for chunk in chunks:
            chunk = _truncate_chunk(chunk, scanned, max_rows)
            users = _int_array(chunk["user_id"].values)
            genders = _int_array(chunk["gender"].values)
            ages = _int_array(chunk["age"].values)
            hists = np.stack([_int_array(chunk[c].values) for c in HIST_COLUMNS], axis=1)
            for i in range(len(users)):
                if users[i] < 0:
                    continue
                uid = str(int(users[i]))
                if uid not in vocab["user2idx"]:
                    continue
                profiles[uid] = {
                    "gender": int(genders[i]),
                    "age": int(ages[i]),
                    "hist": [int(v) for v in hists[i]],
                }
            scanned += len(chunk)
            if max_rows and scanned >= max_rows:
                break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profiles, f)
    return len(profiles)
