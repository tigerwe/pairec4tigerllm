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


def in_batch_softmax_loss(user_vec, item_vec, labels, temperature=0.05):
    """in-batch negative softmax 损失.

    只用正样本行 (label==1); logits[i,j] = <u_i, v_j> / temperature,
    对角线为正样本.
    """
    pos = labels > 0.5
    u = user_vec[pos]
    v = item_vec[pos]
    if u.size(0) < 2:
        return None
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    logits = u @ v.t() / temperature
    target = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, target)
