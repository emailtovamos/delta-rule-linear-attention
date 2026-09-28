"""
Small GPT-style language model with swappable attention.
Designed for studying attention mechanisms, not for SOTA performance.

Model sizes (approximate):
  - tiny:  d=256,  L=6,  H=4   → ~18M params with the GPT-2 vocabulary
  - small: d=512,  L=8,  H=8   → ~42M params
  - medium: d=768, L=12, H=12  → ~125M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from src.attention import build_attention


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.0
    attention_type: str = 'gla'
    attention_kwargs: dict = None
    no_pos_emb: bool = False  # NoPE mode for length generalization

    def __post_init__(self):
        if self.attention_kwargs is None:
            self.attention_kwargs = {}

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @classmethod
    def tiny(cls, attention_type='gla', **kwargs):
        return cls(d_model=256, n_heads=4, n_layers=6, attention_type=attention_type, **kwargs)

    @classmethod
    def small(cls, attention_type='gla', **kwargs):
        return cls(d_model=512, n_heads=8, n_layers=8, attention_type=attention_type, **kwargs)

    @classmethod
    def medium(cls, attention_type='gla', **kwargs):
        return cls(d_model=768, n_heads=12, n_layers=12, attention_type=attention_type, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = build_attention(
            config.attention_type,
            config.d_model,
            config.n_heads,
            **config.attention_kwargs
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.no_pos_emb = config.no_pos_emb
        if not config.no_pos_emb:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        print(f"Model: {self.num_params / 1e6:.1f}M params | "
              f"attention={config.attention_type} | "
              f"d={config.d_model} L={config.n_layers} H={config.n_heads}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        if not self.no_pos_emb:
            assert T <= self.config.max_seq_len, f"Seq len {T} > max {self.config.max_seq_len}"

        tok = self.token_emb(input_ids)
        if self.no_pos_emb:
            x = self.dropout(tok)
        else:
            pos = self.pos_emb(torch.arange(T, device=input_ids.device))
            x = self.dropout(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_attention_states(self, input_ids: torch.Tensor,
                             layer_idx: int = 0, snapshot_every: int = 1):
        """Get attention state snapshots from a specific layer for visualization."""
        B, T = input_ids.shape
        tok = self.token_emb(input_ids)
        if self.no_pos_emb:
            x = self.dropout(tok)
        else:
            pos = self.pos_emb(torch.arange(T, device=input_ids.device))
            x = self.dropout(tok + pos)

        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                normed = block.ln1(x)
                return block.attn.get_state_snapshots(normed, snapshot_every)
            x = block(x)

        return None
