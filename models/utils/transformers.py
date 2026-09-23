"""Transformer components used by Compact Convolutional Transformers.

The parameter/module names follow the upstream CCT implementation so ordinary
CCT checkpoints remain compatible.  ``forward_features`` is the only local
extension: it can expose selected post-block token sequences to BYOT branches.
"""

import torch
import torch.nn.functional as F
from torch.nn import (
    Dropout,
    Identity,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Parameter,
    init,
)

from .stochastic_depth import DropPath


class Attention(Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        attention_dropout=0.1,
        projection_dropout=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        batch_size, token_count, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (
            (attention @ value)
            .transpose(1, 2)
            .reshape(batch_size, token_count, channels)
        )
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        )
        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        return src + self.drop_path(self.dropout2(src2))


class TransformerClassifier(Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout=0.1,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        positional_embedding="learnable",
        sequence_length=None,
    ):
        super().__init__()
        positional_embedding = (
            positional_embedding
            if positional_embedding in ("sine", "learnable", "none")
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0
        if sequence_length is None and positional_embedding != "none":
            raise ValueError(
                "sequence_length is required when positional embeddings are used"
            )

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(
                torch.zeros(1, 1, self.embedding_dim), requires_grad=True
            )
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding == "learnable":
            self.positional_emb = Parameter(
                torch.zeros(1, sequence_length, embedding_dim),
                requires_grad=True,
            )
            init.trunc_normal_(self.positional_emb, std=0.2)
        elif positional_embedding == "sine":
            self.positional_emb = Parameter(
                self.sinusoidal_embedding(sequence_length, embedding_dim),
                requires_grad=False,
            )
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        drop_path_rates = [
            value.item()
            for value in torch.linspace(0, stochastic_depth, num_layers)
        ]
        self.blocks = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_rates[index],
                )
                for index in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embedding_dim)
        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def _prepare_tokens(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(
                x,
                (0, 0, 0, self.sequence_length - x.size(1)),
                mode="constant",
                value=0,
            )
        if not self.seq_pool:
            class_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((class_token, x), dim=1)
        if self.positional_emb is not None:
            x = x + self.positional_emb
        return self.dropout(x)

    def pool(self, x):
        if self.seq_pool:
            weights = F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)
            return torch.matmul(weights, x).squeeze(-2)
        return x[:, 0]

    def forward_features(self, x, return_intermediate_layers=()):
        """Return the final pooled representation and selected block outputs.

        ``return_intermediate_layers`` uses zero-based block indices.  Values
        are post-block token sequences before the classifier's final norm.
        """
        requested = set(return_intermediate_layers)
        intermediate = {}
        x = self._prepare_tokens(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in requested:
                intermediate[index] = x
        x = self.norm(x)
        return self.pool(x), intermediate

    def forward(self, x):
        features, _ = self.forward_features(x)
        return self.fc(features)

    @staticmethod
    def init_weight(module):
        if isinstance(module, Linear):
            init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, LayerNorm):
            init.constant_(module.bias, 0)
            init.constant_(module.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        embedding = torch.FloatTensor(
            [
                [
                    position / (10000 ** (2 * (index // 2) / dim))
                    for index in range(dim)
                ]
                for position in range(n_channels)
            ]
        )
        embedding[:, 0::2] = torch.sin(embedding[:, 0::2])
        embedding[:, 1::2] = torch.cos(embedding[:, 1::2])
        return embedding.unsqueeze(0)

