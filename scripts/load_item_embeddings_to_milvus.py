#!/usr/bin/env python3
# scripts/load_item_embeddings_to_milvus.py
#
# 把 DSSM 导出的 item 向量灌入 Milvus.
#
# 用法:
#   python3 scripts/load_item_embeddings_to_milvus.py \
#     --milvus_host 127.0.0.1 --milvus_port 19530 \
#     --export_dir checkpoints/dssm/export \
#     --collection dssm_item_vectors [--drop]
#
# 需要 pymilvus: pip install pymilvus

import argparse
import json
import os

import numpy as np
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--milvus_host", default="127.0.0.1")
    p.add_argument("--milvus_port", default="19530")
    p.add_argument("--export_dir", default="checkpoints/dssm/export")
    p.add_argument("--collection", default="dssm_item_vectors")
    p.add_argument("--batch_size", type=int, default=5000)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--drop", action="store_true", help="先删除已有同名 collection")
    return p.parse_args()


def main():
    args = parse_args()
    connections.connect(host=args.milvus_host, port=args.milvus_port)

    if args.drop and utility.has_collection(args.collection):
        utility.drop_collection(args.collection)
        print(f"[milvus] dropped existing collection {args.collection}")

    vectors = np.load(os.path.join(args.export_dir, "item_vectors.npy"))
    with open(os.path.join(args.export_dir, "item_ids.json")) as f:
        item_ids = json.load(f)
    assert len(item_ids) == vectors.shape[0], "item_ids 与 vectors 行数不一致"
    dim = vectors.shape[1]
    print(f"[milvus] {len(item_ids)} vectors, dim={dim}")

    if utility.has_collection(args.collection):
        collection = Collection(args.collection)
    else:
        fields = [
            FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True,
                        auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="DSSM item vectors")
        collection = Collection(args.collection, schema)
        print(f"[milvus] created collection {args.collection}")

    for start in range(0, len(item_ids), args.batch_size):
        ids_part = item_ids[start:start + args.batch_size]
        vec_part = vectors[start:start + args.batch_size].tolist()
        collection.insert([ids_part, vec_part])
        if (start // args.batch_size) % 20 == 0:
            print(f"[milvus] inserted {start + len(ids_part)}/{len(item_ids)}")
    collection.flush()
    print("[milvus] insert done, creating index...")

    collection.create_index(
        field_name="vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP",
                      "params": {"nlist": args.nlist}})
    collection.load()
    print(f"[milvus] index created, entities={collection.num_entities}")

    # smoke: 用第 0 条向量自查询, 第一名应该是它自己
    if len(item_ids) > 0:
        res = collection.search(
            data=[vectors[0].tolist()], anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=3, output_fields=["item_id"])
        top = [(h.entity.get("item_id"), round(float(h.distance), 4))
               for h in res[0]]
        print(f"[milvus] self-search smoke item_id={item_ids[0]} top3={top}")
        assert top and top[0][0] == item_ids[0], "self-search 第一名不是自身, 灌库异常"
        print("[milvus] PASS")


if __name__ == "__main__":
    main()
