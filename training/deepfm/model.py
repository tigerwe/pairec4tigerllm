"""DeepFM model used by the offline trainer and online rank service."""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """DeepFM over aligned user, item, category, profile and history fields."""

    def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=(256, 128, 64),
                 dropout=0.1, hist_len=10):
        super().__init__()
        self.hist_len = hist_len
        self.embed_dim = embed_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        self.embeddings = nn.ModuleDict({
            "user": nn.Embedding(vocab_sizes["user"], embed_dim, padding_idx=0),
            "item": nn.Embedding(vocab_sizes["item"], embed_dim, padding_idx=0),
            "cat": nn.Embedding(vocab_sizes["cat"], embed_dim, padding_idx=0),
            "gender": nn.Embedding(vocab_sizes["gender"], embed_dim, padding_idx=0),
            "age": nn.Embedding(vocab_sizes["age"], embed_dim, padding_idx=0),
        })
        self.linear_embeddings = nn.ModuleDict({
            name: nn.Embedding(size, 1, padding_idx=0)
            for name, size in vocab_sizes.items()
        })
        self.linear_bias = nn.Parameter(torch.zeros(1))

        field_count = 5 + hist_len
        deep_layers = []
        previous = field_count * embed_dim
        for hidden in hidden_dims:
            deep_layers.extend([
                nn.Linear(previous, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous = hidden
        deep_layers.append(nn.Linear(previous, 1))
        self.deep = nn.Sequential(*deep_layers)

    def _fields(self, batch):
        fields = [
            ("user", batch["user_id"]),
            ("item", batch["item_id"]),
            ("cat", batch["category"]),
            ("gender", batch["gender"]),
            ("age", batch["age"]),
        ]
        fields.extend(("item", batch["hist"][:, index])
                      for index in range(self.hist_len))
        return fields

    def forward(self, batch):
        fields = self._fields(batch)
        embedded = torch.stack(
            [self.embeddings[name](values) for name, values in fields], dim=1)

        linear = torch.stack(
            [self.linear_embeddings[name](values) for name, values in fields], dim=1
        ).sum(dim=1)
        linear = linear + self.linear_bias

        summed = embedded.sum(dim=1)
        fm = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(
            dim=1, keepdim=True)
        deep = self.deep(embedded.flatten(start_dim=1))
        return (linear + fm + deep).squeeze(1)

    def config(self, vocab_sizes):
        return {
            "vocab_sizes": dict(vocab_sizes),
            "embed_dim": self.embed_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "hist_len": self.hist_len,
        }


def load_deepfm_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    config = checkpoint["model_config"]
    model = DeepFM(
        config["vocab_sizes"],
        embed_dim=config["embed_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
        hist_len=config["hist_len"],
    )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
