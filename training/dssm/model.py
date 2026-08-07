# training/dssm/model.py
#
# DSSM 双塔召回模型.
# user tower: user_id + gender + age + hist(item 序列 mean pooling) -> MLP -> user 向量
# item tower: item_id + video_category -> MLP -> item 向量
# 训练: in-batch softmax (batch 内其他 item 作为负样本)

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, tower_dims=(256, 128),
                 out_dim=64, hist_len=10):
        super().__init__()
        self.hist_len = hist_len
        self.out_dim = out_dim

        # item embedding 在 user hist 和 item tower 之间共享
        self.item_emb = nn.Embedding(vocab_sizes["item"], embed_dim, padding_idx=0)
        self.user_emb = nn.Embedding(vocab_sizes["user"], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(vocab_sizes["cat"], embed_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(vocab_sizes["gender"], embed_dim, padding_idx=0)
        self.age_emb = nn.Embedding(vocab_sizes["age"], embed_dim, padding_idx=0)

        user_in = embed_dim * 4  # user_id + gender + age + hist pooling(共享 item emb)
        self.user_mlp = self._mlp(user_in, tower_dims, out_dim)
        item_in = embed_dim * 2  # item_id + category
        self.item_mlp = self._mlp(item_in, tower_dims, out_dim)

    @staticmethod
    def _mlp(in_dim, hidden_dims, out_dim):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        return nn.Sequential(*layers)

    def user_tower(self, user_id, gender, age, hist):
        """hist: (B, hist_len) item idx, 0 = padding."""
        hist_emb = self.item_emb(hist)                    # (B, L, D)
        mask = (hist > 0).unsqueeze(-1).float()           # (B, L, 1)
        hist_pool = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        x = torch.cat([
            self.user_emb(user_id),
            self.gender_emb(gender),
            self.age_emb(age),
            hist_pool,
        ], dim=-1)
        return self.user_mlp(x)                           # (B, out_dim)

    def item_tower(self, item_id, category):
        x = torch.cat([self.item_emb(item_id), self.cat_emb(category)], dim=-1)
        return self.item_mlp(x)                           # (B, out_dim)

    def forward(self, batch):
        u = self.user_tower(batch["user_id"], batch["gender"], batch["age"], batch["hist"])
        v = self.item_tower(batch["item_id"], batch["category"])
        return u, v


RETRIEVAL_CANDIDATE_MODES = ("all_rows", "positive_rows")


def _retrieval_tensors(user_vec, item_vec, labels, item_ids,
                       candidate_mode):
    if candidate_mode not in RETRIEVAL_CANDIDATE_MODES:
        raise ValueError(
            f"candidate_mode must be one of {RETRIEVAL_CANDIDATE_MODES}"
        )

    if item_ids is None:
        ids = torch.arange(labels.numel(), device=labels.device)
        valid_items = torch.ones_like(labels, dtype=torch.bool)
    else:
        ids = item_ids
        valid_items = item_ids > 0

    query_mask = (labels > 0.5) & valid_items
    candidate_mask = valid_items if candidate_mode == "all_rows" else query_mask
    return (
        user_vec[query_mask],
        item_vec[candidate_mask],
        ids[query_mask],
        ids[candidate_mask],
    )


def retrieval_batch_counts(labels, item_ids, candidate_mode="all_rows"):
    """Return query/candidate evidence for one retrieval batch."""
    if candidate_mode not in RETRIEVAL_CANDIDATE_MODES:
        raise ValueError(
            f"candidate_mode must be one of {RETRIEVAL_CANDIDATE_MODES}"
        )
    valid_items = item_ids > 0
    query_mask = (labels > 0.5) & valid_items
    candidate_mask = valid_items if candidate_mode == "all_rows" else query_mask
    return {
        "queries": int(query_mask.sum()),
        "candidate_rows": int(candidate_mask.sum()),
        "unclicked_candidate_rows": int(
            (candidate_mask & (labels <= 0.5)).sum()
        ),
        "unique_candidates": int(torch.unique(item_ids[candidate_mask]).numel()),
    }


def in_batch_softmax_loss(user_vec, item_vec, labels, temperature=0.05,
                          item_ids=None, candidate_mode="all_rows"):
    """in-batch negative softmax 损失.

    正样本行 (label==1) 提供 query。默认由 batch 全部有效 item 行提供
    candidates，使 click=0 曝光成为自然负样本；positive_rows 保留旧基线。
    同一 item 的所有 candidate 列均视为该 query 的正样本。
    """
    u, v, query_ids, candidate_ids = _retrieval_tensors(
        user_vec, item_vec, labels, item_ids, candidate_mode
    )
    if u.size(0) < 1 or v.size(0) < 2:
        return None
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    logits = u @ v.t() / temperature
    # Repeated items in any candidate row are positives, not false negatives.
    positive_mask = query_ids[:, None].eq(candidate_ids[None, :])
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) -
            torch.logsumexp(positive_logits, dim=1)).mean()


@torch.no_grad()
def in_batch_recall_counts(user_vec, item_vec, labels, item_ids,
                           topk=(10, 50, 100), candidate_mode="all_rows"):
    """Return positive-query hit counts against the current validation batch."""
    u, v, query_ids, candidate_ids = _retrieval_tensors(
        user_vec, item_vec, labels, item_ids, candidate_mode
    )
    count = int(u.size(0))
    result = {"queries": count}
    for k in topk:
        result[f"hits_at_{k}"] = 0
    if count < 1 or v.size(0) < 1:
        return result
    logits = F.normalize(u, dim=-1) @ F.normalize(v, dim=-1).t()
    max_k = min(max(topk), int(v.size(0)))
    candidates = candidate_ids[torch.topk(logits, max_k, dim=1).indices]
    matches = candidates.eq(query_ids[:, None])
    for k in topk:
        result[f"hits_at_{k}"] = int(matches[:, :min(k, max_k)].any(dim=1).sum())
    return result
