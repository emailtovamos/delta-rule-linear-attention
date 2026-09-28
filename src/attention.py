"""
Linear attention variants with different memory decay mechanisms.

All implementations are naive pure-PyTorch for clarity and correctness.
No Triton kernels — runs on MPS (Apple Silicon) directly.

Notation:
  B = batch, T = sequence length, H = num heads, D = head dim
  q, k, v = query, key, value  (B, H, T, D)
  S = state matrix (B, H, D, D) — the "memory" of linear attention
  g = gate / decay factor

The key insight:  ALL linear attention variants can be written as:
    S_t = decay(S_{t-1}) + outer(k_t, v_t)
    o_t = S_t @ q_t

They differ in HOW decay works:
  - GLA:                S_t = g_t * S_{t-1} + k_t ⊗ v_t                (scalar gate, data-dependent)
  - DeltaNet:           S_t = g_t * S_{t-1} + k_t ⊗ (v_t - Sᵀk_t)    (delta rule, scalar gate, data-dependent)
  - KDA:                S_t = G_t ⊙ S_{t-1} + k_t ⊗ (v_t - Sᵀk_t)    (delta rule, channel gate, data-dependent)
  - StaticChannel:      S_t = Γ ⊙ S_{t-1} + k_t ⊗ v_t                 (channel gate, data-INDEPENDENT)
  - StaticChannelDelta: S_t = Γ ⊙ S_{t-1} + k_t ⊗ (v_t - Sᵀk_t)     (delta rule, channel gate, data-INDEPENDENT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math
from typing import Optional, Literal


# ─────────────────────────────────────────────
# Helper: feature map for linear attention
# ─────────────────────────────────────────────

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: φ(x) = ELU(x) + 1. Ensures non-negative features."""
    return F.elu(x) + 1.0


def identity_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Identity (no transformation). Used when gating handles non-linearity."""
    return x


# ─────────────────────────────────────────────
# 1. Standard Softmax Attention (baseline)
# ─────────────────────────────────────────────

class StandardAttention(nn.Module):
    """Vanilla causal self-attention for comparison."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b t (three h d) -> three b h t d',
                             three=3, h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor):
        """Not applicable for standard attention (no recurrent state)."""
        return None


# ─────────────────────────────────────────────
# 2. GLA — Gated Linear Attention
# ─────────────────────────────────────────────

class GLAAttention(nn.Module):
    """
    Gated Linear Attention (Yang et al., 2023).

    State update: S_t = g_t · S_{t-1} + φ(k_t) ⊗ v_t
    Output:       o_t = S_t^T @ φ(q_t)

    Gate g_t is a SCALAR per head per timestep.
    This means ALL dimensions of the state decay at the same rate.

    ┌─────────────────────────────────┐
    │  S_t = g · S_{t-1} + k ⊗ v     │
    │         ↑                       │
    │     SCALAR gate                 │
    │     same rate for               │
    │     all D×D entries             │
    └─────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)  # scalar per head
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Apply feature map
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        # Compute gate: scalar per (batch, head, time)
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, H)
        gate = rearrange(gate, 'b t h -> b h t')  # (B, H, T)

        # Recurrent computation (naive, for clarity)
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            g_t = gate[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            k_t = k[:, :, t, :]   # (B, H, D)
            v_t = v[:, :, t, :]   # (B, H, D)
            q_t = q[:, :, t, :]   # (B, H, D)

            # State update: S = g * S + k^T @ v
            S = g_t * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            # Output: o = S^T @ q
            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)  # (B, H, T, D)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        """Return state matrix S at each timestep for visualization."""
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        gate = torch.sigmoid(self.gate_proj(x))
        gate = rearrange(gate, 'b t h -> b h t')

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            g_t = gate[:, :, t].unsqueeze(-1).unsqueeze(-1)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            S = g_t * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(gate[:, :, t].detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),   # (B, H, num_snaps, D, D)
            'gates': torch.stack(gates_log, dim=2),     # (B, H, num_snaps)
            'gate_type': 'scalar',
        }


# ─────────────────────────────────────────────
# 3. DeltaNet — Delta Rule Linear Attention
# ─────────────────────────────────────────────

class DeltaNetAttention(nn.Module):
    """
    DeltaNet (Schlag et al., 2021; Yang et al., 2024).

    State update uses the DELTA RULE:
      error_t = v_t - S_{t-1}^T @ k_t       (what's missing from memory)
      S_t = g_t · S_{t-1} + k_t ⊗ error_t   (correct the memory)

    Still uses scalar gating, but the update itself is smarter.
    Think of it as: "don't just add, CORRECT the memory."

    ┌─────────────────────────────────────────┐
    │  error = v - S^T @ k   (prediction err) │
    │  S_t = g · S + k ⊗ error                │
    │         ↑                                │
    │     SCALAR gate (same as GLA)            │
    │     but UPDATE is smarter                │
    └─────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Delta rule requires L2-normalized keys for stability
        # (unnormalized keys cause S^T @ k to grow unboundedly → NaN)
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        gate = torch.sigmoid(self.gate_proj(x))
        gate = rearrange(gate, 'b t h -> b h t')

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            g_t = gate[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Delta rule: correct memory instead of blindly adding
            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)  # S^T @ k
            error = v_t - prediction  # what's missing

            S = g_t * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        """Return state snapshots for visualization."""
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        gate = torch.sigmoid(self.gate_proj(x))
        gate = rearrange(gate, 'b t h -> b h t')

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            g_t = gate[:, :, t].unsqueeze(-1).unsqueeze(-1)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction
            S = g_t * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(gate[:, :, t].detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=2),
            'gate_type': 'scalar',
        }


# ─────────────────────────────────────────────
# 4. KDA — Kimi Delta Attention (Channel-Wise)
# ─────────────────────────────────────────────

class KDAAttention(nn.Module):
    """
    Kimi Delta Attention (Zhang et al., 2025).

    Combines TWO innovations:
    1. Channel-wise gating — each row of S decays at its own rate
    2. Delta rule update  — correct memory instead of blindly adding

    State update:
      error_t = v_t - S_{t-1}^T @ k_t          (what's missing from memory)
      S_t = G_t ⊙ S_{t-1} + k_t ⊗ error_t     (channel-wise decay + correction)

    Keys are L2-normalized (required for delta rule stability).

    ┌───────────────────────────────────────────────┐
    │  G_t = σ(W_g @ x_t)  ∈ R^D (per dim!)       │
    │  error = v_t - S^T @ k_t   (prediction err)  │
    │  S_t = G_t ⊙ S_{t-1} + k_t ⊗ error          │
    │         ↑                                      │
    │     CHANNEL-WISE gate + DELTA RULE             │
    │     dim 0: g=0.99 (slow, long memory)          │
    │     dim 1: g=0.50 (fast, short memory)         │
    │     ...each dimension independent!              │
    └───────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Channel-wise gate: one gate value per KEY dimension per head
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)  # (D,) per head
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Delta rule requires L2-normalized keys for stability
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        # Channel-wise gate: (B, T, H*D) -> (B, H, T, D)
        gate = torch.sigmoid(self.gate_proj(x))
        gate = rearrange(gate, 'b t (h d) -> b h t d', h=self.n_heads)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            # G_t has shape (B, H, D) — one gate per KEY dimension
            G_t = gate[:, :, t, :]  # (B, H, D)

            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Delta rule: correct memory instead of blindly adding
            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)  # S^T @ k
            error = v_t - prediction  # what's missing

            # Channel-wise decay + delta correction
            S = G_t.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        """Return state snapshots + per-channel gate values for visualization."""
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        gate = torch.sigmoid(self.gate_proj(x))
        gate = rearrange(gate, 'b t (h d) -> b h t d', h=self.n_heads)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            G_t = gate[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction
            S = G_t.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(gate[:, :, t, :].detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),   # (B, H, num_snaps, D, D)
            'gates': torch.stack(gates_log, dim=2),     # (B, H, num_snaps, D) — per channel!
            'gate_type': 'channel_wise',
        }


# ─────────────────────────────────────────────
# 5. Static Channel Decay (Data-Independent Channel-Wise Gating)
# ─────────────────────────────────────────────

class StaticChannelAttention(nn.Module):
    """
    Data-independent channel-wise gating — NO delta rule.

    Motivation (Gap 1 from paper analysis):
    - KDA (Kimi Linear 2025): channel-wise gating is better than scalar → more expressivity
    - HALO (Chen et al. Jan 2026): data-independent gates are better than data-dependent
      → better length generalization
    - Nobody has tested: channel-wise AND data-independent

    Each channel (row of S) gets its OWN decay rate, but these rates are
    LEARNED PARAMETERS — they do NOT depend on the input at each timestep.
    Think of it as: the model learns a fixed spectrum of memory timescales
    during training, then uses that spectrum unchanged at inference.

    State update: S_t = Γ ⊙ S_{t-1} + φ(k_t) ⊗ v_t
    Output:       o_t = S_t^T @ φ(q_t)

    where Γ ∈ R^{H×D} is a FIXED learned parameter (per head, per channel).

    Comparison:
    ┌────────────────────────────────────────────────────────┐
    │  Lightning Attn:  γ_h (one scalar per HEAD, fixed)    │
    │  This variant:    γ_{h,d} (one scalar per CHANNEL)    │
    │  KDA:             G_t = f(x_t) (per CHANNEL, varies!) │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-channel decay rates: learnable but NOT input-dependent
        # Initialize with Lightning Attention-style exponential spectrum
        # γ_d = exp(-2^{-8d/D}) gives a smooth range from ~0.43 to ~0.9996
        d = self.d_head
        indices = torch.arange(d, dtype=torch.float32)
        init_decays = torch.exp(-torch.pow(2.0, -8.0 * indices / d))
        # Store as logits so sigmoid maps back to (0, 1)
        init_logits = torch.log(init_decays / (1.0 - init_decays))
        # Shape: (n_heads, d_head) — each head gets the same init but can diverge
        self.decay_logits = nn.Parameter(init_logits.unsqueeze(0).expand(n_heads, -1).clone())

    def _get_decay(self) -> torch.Tensor:
        """Fixed decay vector, shape (H, D). Same for all tokens."""
        return torch.sigmoid(self.decay_logits)  # (H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        decay = self._get_decay()  # (H, D)
        # Expand for broadcasting: (1, H, D, 1) to multiply against S (B, H, D, D)
        decay_expanded = decay.unsqueeze(0).unsqueeze(-1)  # (1, H, D, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        decay = self._get_decay()
        decay_expanded = decay.unsqueeze(0).unsqueeze(-1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                # Log the fixed decay (same every step, but useful for visualization)
                gates_log.append(decay.unsqueeze(0).expand(B, -1, -1).detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=2),  # (B, H, num_snaps, D)
            'gate_type': 'channel_wise',
            'static': True,
            'learned_decays': decay.detach().cpu(),
        }


class StaticChannelDeltaAttention(nn.Module):
    """
    Data-independent channel-wise gating + delta rule.

    The FULL Gap 1 hypothesis: combine the best of KDA and Lightning Attention.
    - From KDA: channel-wise gating (per-dim memory timescales)
    - From KDA: delta rule (corrective memory updates)
    - From Lightning/HALO: data-independent decay (length generalization)

    State update:
      error_t = v_t - S_{t-1}^T @ k_t       (what's missing from memory)
      S_t = Γ ⊙ S_{t-1} + k_t ⊗ error_t    (fixed channel decay + delta correction)

    where Γ ∈ R^{H×D} is a FIXED learned parameter.
    Keys are L2-normalized (required for delta rule stability).

    ┌──────────────────────────────────────────────────────┐
    │  Γ = σ(learnable_logits)  ∈ R^{H×D}  (FIXED!)      │
    │  error = v_t - S^T @ k_t              (delta rule)  │
    │  S_t = Γ ⊙ S_{t-1} + k_t ⊗ error                   │
    │         ↑                                             │
    │     CHANNEL-WISE gate (like KDA)                      │
    │     but DATA-INDEPENDENT (like Lightning)             │
    │     + DELTA RULE (like KDA/DeltaNet)                  │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, decay_init: str = "exponential"):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-channel decay rates: learnable but NOT input-dependent
        d = self.d_head
        init_logits = self._make_init_logits(d, decay_init)
        self.decay_logits = nn.Parameter(init_logits.unsqueeze(0).expand(n_heads, -1).clone())

    @staticmethod
    def _make_init_logits(d: int, decay_init: str) -> torch.Tensor:
        """Create initial decay logits for different initialization strategies."""
        if decay_init == "exponential":
            # Lightning Attention-style: γ_i = exp(-2^{-8i/d})
            indices = torch.arange(d, dtype=torch.float32)
            init_decays = torch.exp(-torch.pow(2.0, -8.0 * indices / d))
        elif decay_init == "uniform":
            # All channels same decay rate (midpoint of exponential range)
            init_decays = torch.full((d,), 0.9)
        elif decay_init == "random":
            # Uniform random in [0.3, 0.999]
            init_decays = 0.3 + 0.699 * torch.rand(d)
        else:
            raise ValueError(f"Unknown decay_init: {decay_init}. Choose from exponential, uniform, random")
        return torch.log(init_decays / (1.0 - init_decays))

    def _get_decay(self) -> torch.Tensor:
        """Fixed decay vector, shape (H, D). Same for all tokens."""
        return torch.sigmoid(self.decay_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Delta rule requires L2-normalized keys
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        decay = self._get_decay()
        decay_expanded = decay.unsqueeze(0).unsqueeze(-1)  # (1, H, D, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Delta rule: correct memory instead of blindly adding
            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction

            # Channel-wise fixed decay + delta correction
            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        decay = self._get_decay()
        decay_expanded = decay.unsqueeze(0).unsqueeze(-1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction
            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(decay.unsqueeze(0).expand(B, -1, -1).detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=2),
            'gate_type': 'channel_wise',
            'static': True,
            'learned_decays': decay.detach().cpu(),
        }


# ─────────────────────────────────────────────
# 6a. Scalar Static Decay (RetNet-style, no delta rule)
# ─────────────────────────────────────────────

class ScalarStaticAttention(nn.Module):
    """
    Data-independent SCALAR gating — NO delta rule.

    This is the Lightning Attention / RetNet quadrant:
    one fixed scalar decay per head, simple outer-product write.

    Completes the 2×2 grid:
    ┌────────────────────────────────────────────────────────┐
    │  Lightning Attn:  γ_h (one scalar per HEAD, fixed)    │  ← THIS
    │  StaticChannel:   γ_{h,d} (one scalar per CHANNEL)    │
    │  GLA:             g_t = f(x_t) (scalar, varies)       │
    │  KDA:             G_t = f(x_t) (per channel, varies)  │
    └────────────────────────────────────────────────────────┘

    State update: S_t = γ_h · S_{t-1} + φ(k_t) ⊗ v_t
    Output:       o_t = S_t^T @ φ(q_t)

    Initialization: γ_h = exp(-2^{-8h/H}) — exponential spectrum across heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # One scalar decay per head (Lightning Attention / RetNet style)
        indices = torch.arange(n_heads, dtype=torch.float32)
        init_decays = torch.exp(-torch.pow(2.0, -8.0 * indices / n_heads))
        init_logits = torch.log(init_decays / (1.0 - init_decays))
        self.decay_logits = nn.Parameter(init_logits)  # (H,)

    def _get_decay(self) -> torch.Tensor:
        """Fixed scalar decay per head, shape (H,)."""
        return torch.sigmoid(self.decay_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        decay = self._get_decay()  # (H,)
        decay_expanded = decay.view(1, self.n_heads, 1, 1)  # (1, H, 1, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        decay = self._get_decay()
        decay_expanded = decay.view(1, self.n_heads, 1, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(decay.detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=1),  # (H, num_snaps)
            'gate_type': 'scalar',
            'static': True,
            'learned_decays': decay.detach().cpu(),
        }


# ─────────────────────────────────────────────
# 6b. Scalar Static Decay + Delta Rule
# ─────────────────────────────────────────────

class ScalarStaticDeltaAttention(nn.Module):
    """
    Data-independent SCALAR gating + delta rule.

    Combines Lightning Attention's fixed scalar decay with the delta rule.
    This isolates the contribution of channel-wise granularity:
      - ScalarStaticDelta vs StaticChannelDelta  → effect of channel-wise
      - ScalarStaticDelta vs ScalarStatic        → effect of delta rule
      - ScalarStaticDelta vs DeltaNet            → effect of static vs learned

    State update:
      error_t = v_t - S_{t-1}^T @ k_t
      S_t = γ_h · S_{t-1} + k_t ⊗ error_t

    where γ_h is a FIXED learned scalar per head.
    Keys are L2-normalized (required for delta rule stability).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # One scalar decay per head
        indices = torch.arange(n_heads, dtype=torch.float32)
        init_decays = torch.exp(-torch.pow(2.0, -8.0 * indices / n_heads))
        init_logits = torch.log(init_decays / (1.0 - init_decays))
        self.decay_logits = nn.Parameter(init_logits)  # (H,)

    def _get_decay(self) -> torch.Tensor:
        """Fixed scalar decay per head, shape (H,)."""
        return torch.sigmoid(self.decay_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Delta rule requires L2-normalized keys
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        decay = self._get_decay()  # (H,)
        decay_expanded = decay.view(1, self.n_heads, 1, 1)  # (1, H, 1, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Delta rule
            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction

            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = F.normalize(k, p=2, dim=-1)

        decay = self._get_decay()
        decay_expanded = decay.view(1, self.n_heads, 1, 1)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            prediction = torch.einsum('bhde,bhd->bhe', S, k_t)
            error = v_t - prediction
            S = decay_expanded * S + torch.einsum('bhd,bhe->bhde', k_t, error)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(decay.detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=1),
            'gate_type': 'scalar',
            'static': True,
            'learned_decays': decay.detach().cpu(),
        }


# ─────────────────────────────────────────────
# 7. EXPERIMENTAL: Hierarchical Decay
# ─────────────────────────────────────────────

class HierarchicalDecayAttention(nn.Module):
    """
    EXPERIMENTAL: Hierarchical multi-scale decay.

    Instead of learning gates freely, we STRUCTURE them:
    - Split D dimensions into groups of exponentially different decay rates
    - Group 0 (dims 0..D/4):     slow decay  (γ ~ 0.99) — long-term memory
    - Group 1 (dims D/4..D/2):   medium decay (γ ~ 0.95) — medium-term
    - Group 2 (dims D/2..3D/4):  fast decay   (γ ~ 0.80) — short-term
    - Group 3 (dims 3D/4..D):    very fast    (γ ~ 0.50) — immediate context

    Inspired by Log-Linear Attention (Guo et al., 2025) which uses
    a Fenwick tree to maintain multi-scale state banks.

    ┌──────────────────────────────────────────┐
    │  dims 0-15:  γ=0.99  ████████████████    │  long memory
    │  dims 16-31: γ=0.95  ████████████        │  medium
    │  dims 32-47: γ=0.80  ████████            │  short
    │  dims 48-63: γ=0.50  ████                │  immediate
    └──────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_groups: int = 4, base_decays: tuple = (0.99, 0.95, 0.80, 0.50)):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_groups = n_groups

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable base decays (initialized to structured values)
        # Each group has a learnable logit that maps to a decay rate via sigmoid
        decay_logits = [math.log(d / (1 - d)) for d in base_decays]
        self.decay_logits = nn.Parameter(torch.tensor(decay_logits))

        # Small per-token modulation (learned adjustment around base decay)
        self.modulation_proj = nn.Linear(d_model, n_heads * n_groups, bias=True)

    def _get_decay_vector(self, x_t: torch.Tensor) -> torch.Tensor:
        """Compute structured decay vector of shape (B, H, D).

        Base rates are fixed per group, with small learned modulation.
        """
        B = x_t.shape[0]
        base_decays = torch.sigmoid(self.decay_logits)  # (n_groups,)

        # Small modulation: shifts decay slightly based on content
        mod = self.modulation_proj(x_t)  # (B, H * n_groups)
        mod = rearrange(mod, 'b (h g) -> b h g', h=self.n_heads)
        mod = 0.05 * torch.tanh(mod)  # small perturbation ±0.05

        # Build decay vector: assign each dim to a group
        dims_per_group = self.d_head // self.n_groups
        decay_parts = []
        for g in range(self.n_groups):
            group_decay = (base_decays[g] + mod[:, :, g]).unsqueeze(-1)  # (B, H, 1)
            group_decay = group_decay.expand(-1, -1, dims_per_group)      # (B, H, dims_per_group)
            decay_parts.append(group_decay)

        return torch.cat(decay_parts, dim=-1).clamp(0.01, 0.999)  # (B, H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            decay = self._get_decay_vector(x[:, t, :])  # (B, H, D)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            S = decay.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            o_t = torch.einsum('bhde,bhd->bhe', S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head,
                        device=x.device, dtype=x.dtype)
        snapshots = []
        gates_log = []

        for t in range(T):
            decay = self._get_decay_vector(x[:, t, :])
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            S = decay.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            if t % snapshot_every == 0:
                snapshots.append(S.detach().cpu())
                gates_log.append(decay.detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'gates': torch.stack(gates_log, dim=2),
            'gate_type': 'hierarchical',
            'n_groups': self.n_groups,
            'base_decays': torch.sigmoid(self.decay_logits).detach().cpu(),
        }


# ─────────────────────────────────────────────
# 8. EXPERIMENTAL: Frequency-Routed Decay
# ─────────────────────────────────────────────

class FrequencyRoutedAttention(nn.Module):
    """
    EXPERIMENTAL: Route tokens to different decay channels based on
    a learned "frequency classifier."

    The idea: some tokens carry stable/factual info (nouns, numbers)
    and should go to slow-decay channels. Others carry transient info
    (function words, punctuation) and should go to fast-decay channels.

    Instead of a fixed assignment like HierarchicalDecay, we LEARN
    a soft routing that decides which tokens update which channels.

    ┌────────────────────────────────────────────────┐
    │  router(x_t) → routing weights R_t ∈ R^{G}    │
    │                                                 │
    │  For each decay group g:                        │
    │    S_g = γ_g · S_g + R_t[g] · (k_t ⊗ v_t)    │
    │                                                 │
    │  Output: concat all groups → project            │
    └────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_groups: int = 4, base_decays: tuple = (0.99, 0.95, 0.80, 0.50)):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_groups = n_groups

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Fixed decay rates per group
        self.register_buffer('decay_rates',
                             torch.tensor(base_decays).float())

        # Router: decides how much each token goes to each decay group
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads * n_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        # Route each token to decay groups
        routing = self.router(x)  # (B, T, H*G)
        routing = rearrange(routing, 'b t (h g) -> b h t g', h=self.n_heads)
        routing = F.softmax(routing, dim=-1)  # soft routing

        dims_per_group = self.d_head // self.n_groups

        # State per group
        S_groups = [
            torch.zeros(B, self.n_heads, dims_per_group, dims_per_group,
                        device=x.device, dtype=x.dtype)
            for _ in range(self.n_groups)
        ]

        outputs = []
        for t in range(T):
            k_t = k[:, :, t, :]  # (B, H, D)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            r_t = routing[:, :, t, :]  # (B, H, G)

            # Split k, v into groups
            k_groups = k_t.chunk(self.n_groups, dim=-1)
            v_groups = v_t.chunk(self.n_groups, dim=-1)
            q_groups = q_t.chunk(self.n_groups, dim=-1)

            o_parts = []
            for g in range(self.n_groups):
                decay = self.decay_rates[g]
                weight = r_t[:, :, g].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

                S_groups[g] = decay * S_groups[g] + weight * torch.einsum(
                    'bhd,bhe->bhde', k_groups[g], v_groups[g])

                o_g = torch.einsum('bhde,bhd->bhe', S_groups[g], q_groups[g])
                o_parts.append(o_g)

            o_t = torch.cat(o_parts, dim=-1)  # (B, H, D)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)

    def get_state_snapshots(self, x: torch.Tensor, snapshot_every: int = 1):
        B, T, D = x.shape
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = elu_feature_map(k)

        routing = self.router(x)
        routing = rearrange(routing, 'b t (h g) -> b h t g', h=self.n_heads)
        routing = F.softmax(routing, dim=-1)

        dims_per_group = self.d_head // self.n_groups
        S_groups = [
            torch.zeros(B, self.n_heads, dims_per_group, dims_per_group,
                        device=x.device, dtype=x.dtype)
            for _ in range(self.n_groups)
        ]

        snapshots = []
        routing_log = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            r_t = routing[:, :, t, :]
            k_groups = k_t.chunk(self.n_groups, dim=-1)
            v_groups = v_t.chunk(self.n_groups, dim=-1)

            for g in range(self.n_groups):
                decay = self.decay_rates[g]
                weight = r_t[:, :, g].unsqueeze(-1).unsqueeze(-1)
                S_groups[g] = decay * S_groups[g] + weight * torch.einsum(
                    'bhd,bhe->bhde', k_groups[g], v_groups[g])

            if t % snapshot_every == 0:
                full_S = torch.cat(
                    [sg.detach().cpu() for sg in S_groups], dim=-2)
                snapshots.append(full_S)
                routing_log.append(r_t.detach().cpu())

        return {
            'states': torch.stack(snapshots, dim=2),
            'routing': torch.stack(routing_log, dim=2),  # (B, H, T, G)
            'gate_type': 'frequency_routed',
            'n_groups': self.n_groups,
            'decay_rates': self.decay_rates.detach().cpu(),
        }


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

ATTENTION_REGISTRY = {
    'standard': StandardAttention,
    'gla': GLAAttention,
    'deltanet': DeltaNetAttention,
    'kda': KDAAttention,
    'static_channel': StaticChannelAttention,
    'static_channel_delta': StaticChannelDeltaAttention,
    'scalar_static': ScalarStaticAttention,
    'scalar_static_delta': ScalarStaticDeltaAttention,
    'hierarchical': HierarchicalDecayAttention,
    'freq_routed': FrequencyRoutedAttention,
}


def build_attention(name: str, d_model: int, n_heads: int, **kwargs) -> nn.Module:
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention: {name}. Choose from {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[name](d_model, n_heads, **kwargs)
