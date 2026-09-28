# What Is a "Channel" in Channel-Wise Forgetting?

## The State Matrix S

In linear attention, the model's "memory" is a matrix **S** of shape `(D x D)` where D is the head dimension (e.g., 64). At every timestep t, the state is updated:

```
S_t = decay(S_{t-1}) + k_t (x) v_t
```

The outer product `k_t (x) v_t` writes a rank-1 update into S. Think of S as an associative memory — a lookup table that maps key directions to value directions. When the model wants to retrieve information, it computes `o_t = S^T @ q_t`, weighting each row of S by the corresponding query element.

## A Channel = One Row of S

Each of the D rows of S corresponds to one dimension of the key space. Row `d` accumulates all the value information that was ever written along key dimension `d`.

```
         THE STATE MATRIX S  (D x D)
         ══════════════════════════════

         ┌─── value dim 0    value dim D-1 ───┐
         │                                     │
         v                                     v
        col 0   col 1   col 2  ...  col D-1
       ┌───────┬───────┬───────┬───┬───────┐
row 0  │ s_0,0 │ s_0,1 │ s_0,2 │...│s_0,63 │  <-- Channel 0
       ├───────┼───────┼───────┼───┼───────┤
row 1  │ s_1,0 │ s_1,1 │ s_1,2 │...│s_1,63 │  <-- Channel 1
       ├───────┼───────┼───────┼───┼───────┤
row 2  │ s_2,0 │ s_2,1 │ s_2,2 │...│s_2,63 │  <-- Channel 2
       ├───────┼───────┼───────┼───┼───────┤
  :    │  ...  │  ...  │  ...  │...│  ...  │
       ├───────┼───────┼───────┼───┼───────┤
row 63 │s_63,0 │s_63,1 │s_63,2 │...│s63,63 │  <-- Channel 63
       └───────┴───────┴───────┴───┴───────┘

       Each ROW is a "channel."
       It stores the accumulated value information
       written by key dimension d across all timesteps.
```

When the model queries memory with `o_t = S^T @ q_t`:

```
       q_t = [ q_0,  q_1,  q_2, ..., q_63 ]
                |     |     |          |
                v     v     v          v
             ┌─────┬─────┬─────┬───┬─────┐
    row 0 -> │ ... │ ... │ ... │...│ ... │ * q_0   (weight of channel 0)
             ├─────┼─────┼─────┼───┼─────┤
    row 1 -> │ ... │ ... │ ... │...│ ... │ * q_1   (weight of channel 1)
             ├─────┼─────┼─────┼───┼─────┤
      :      │     │     │     │   │     │
             ├─────┼─────┼─────┼───┼─────┤
    row 63-> │ ... │ ... │ ... │...│ ... │ * q_63  (weight of channel 63)
             └─────┴─────┴─────┴───┴─────┘
                │     │     │          │
                v     v     v          v
       o_t = [ o_0,  o_1,  o_2, ..., o_63 ]   (weighted sum of rows)
```

Each element of the query decides how much to "listen" to each channel.

## Scalar Gating vs Channel-Wise Gating

### Scalar Gating (GLA, DeltaNet)

One gate value `g_t` (a single number) is applied to the ENTIRE state matrix. Every channel decays at the same rate.

```
SCALAR GATING:  S_t = g_t * S_{t-1} + k_t (x) v_t

       g_t = 0.95  (one number for ALL rows)

             BEFORE                          AFTER
       ┌─────────────────┐            ┌─────────────────┐
row 0  │ ############### │ * 0.95     │ #############   │  lost 5%
       ├─────────────────┤            ├─────────────────┤
row 1  │ ############### │ * 0.95     │ #############   │  lost 5%
       ├─────────────────┤            ├─────────────────┤
row 2  │ ############### │ * 0.95     │ #############   │  lost 5%
       ├─────────────────┤            ├─────────────────┤
row 3  │ ############### │ * 0.95     │ #############   │  lost 5%
       └─────────────────┘            └─────────────────┘
         ^                              ^
         ALL rows decay                 ALL rows lost the
         at the SAME rate               SAME amount of info
```

The problem: you **cannot** have some channels remember long-term facts while others track short-term context. It's all or nothing.

### Channel-Wise Gating (KDA)

Each channel (row) gets its **own** gate value. The gate vector `G_t` has D elements — one per row.

```
CHANNEL-WISE GATING:  S_t = G_t (*) S_{t-1} + k_t (x) v_t

       G_t = [0.99, 0.50, 0.95, 0.70]  (one value PER row)

             BEFORE                          AFTER
       ┌─────────────────┐            ┌─────────────────┐
row 0  │ ############### │ * 0.99     │ ###############  │  lost 1%  (SLOW)
       ├─────────────────┤            ├─────────────────┤
row 1  │ ############### │ * 0.50     │ ########        │  lost 50% (FAST)
       ├─────────────────┤            ├─────────────────┤
row 2  │ ############### │ * 0.95     │ #############   │  lost 5%  (MEDIUM)
       ├─────────────────┤            ├─────────────────┤
row 3  │ ############### │ * 0.70     │ ##########      │  lost 30% (FAST-ISH)
       └─────────────────┘            └─────────────────┘
         ^                              ^
         Each row decays                Different amounts
         at a DIFFERENT rate            of info retained
```

Now the network can **learn** to use different channels for different purposes.

## What This Looks Like Over Time

Consider a sequence where the model reads "The capital of France is Paris" at the start, then processes 1000 tokens of other content, then needs to recall the fact.

### Scalar Gating (All Channels Same Rate)

```
    TIME ──────────────────────────────────────────────────>

    "Paris is capital"                    "What is the capital?"
    written here                          needs to recall here
         |                                        |
         v                                        v
    ┌─────────────────────────────────────────────────────┐
Ch0 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │ decayed
Ch1 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │ decayed
Ch2 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │ decayed
Ch3 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │ decayed
    └─────────────────────────────────────────────────────┘
                                                  ^
                                          ALL channels lost
                                          the info equally.
                                          If g=0.95 and 1000 steps:
                                          0.95^1000 ~ 0.0 (GONE)
```

### Channel-Wise Gating (Different Rates Per Channel)

```
    TIME ──────────────────────────────────────────────────>

    "Paris is capital"                    "What is the capital?"
    written here                          needs to recall here
         |                                        |
         v                                        v
    ┌─────────────────────────────────────────────────────┐
Ch0 │ ████████████████████████████████████████████████████│ g=0.999 STILL HERE
Ch1 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │ g=0.95  gone
Ch2 │ ████████████████████████████████░░░░░░░░            │ g=0.99  fading
Ch3 │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░           │ g=0.90  gone
    └─────────────────────────────────────────────────────┘
                                                  ^
                                          Channel 0 STILL has
                                          the Paris info!
                                          0.999^1000 ~ 0.37 (alive)

    The network learns:
      - Ch0 (g~0.999): "slow lane" for facts (capitals, names, numbers)
      - Ch1 (g~0.95):  "fast lane" for recent context (current sentence)
      - Ch2 (g~0.99):  "medium lane" for paragraph-level topics
      - Ch3 (g~0.90):  "very fast lane" for immediate syntax
```

## The Numbers

Why does the rate matter so much? Exponential decay is steep:

```
    INFORMATION RETAINED AFTER N STEPS
    ═══════════════════════════════════════════

    Steps:     10      100     1000    10000
    ─────────────────────────────────────────
    g = 0.999  0.99    0.90    0.37    0.00
    g = 0.99   0.90    0.37    0.00    0.00
    g = 0.95   0.60    0.01    0.00    0.00
    g = 0.90   0.35    0.00    0.00    0.00
    g = 0.50   0.00    0.00    0.00    0.00

    Even a small difference (0.99 vs 0.95) means
    the difference between "remembers at 100 steps"
    and "completely forgotten at 100 steps."
```

This is why channel-wise gating matters: by letting each dimension choose its own rate, the model can maintain a **spectrum** of memory timescales within a single state matrix. On MRCR at 128K context, Kimi Linear's KDA-based variant scored 29.6 versus 23.9 for its Gated DeltaNet-based counterpart (Zhang et al., 2025, Table 5), providing motivation for studying channel-wise structure under controlled conditions.

## In The Code

From `src/attention.py`:

```python
# GLA — scalar gate, simple outer product (line 140-146)
g_t = gate[:, :, t].unsqueeze(-1).unsqueeze(-1)  # shape: (B, H, 1, 1)
S = g_t * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
#     ^--- broadcasts across ALL rows and columns: same decay everywhere
#     note: GLA uses k ⊗ v (no delta rule)

# KDA — channel-wise gate + delta rule (line 364-377)
G_t = gate[:, :, t, :]              # shape: (B, H, D) — one value per row!
prediction = torch.einsum('bhde,bhd->bhe', S, k_t)  # S^T @ k
error = v_t - prediction                              # what's missing
S = G_t.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, error)
#       ^--- unsqueeze(-1) broadcasts across columns only
#            each ROW gets multiplied by its own gate value
#       note: KDA uses k ⊗ error (delta rule corrects memory)
```

KDA differs from GLA in **two** ways:
1. **Gate shape**: `(B, H, 1, 1)` vs `(B, H, D, 1)` — scalar to per-channel
2. **Delta rule**: `k ⊗ v` vs `k ⊗ (v - S^T @ k)` — blind write vs corrective update

The "D" in KDA stands for "Delta" — it combines channel-wise gating with the delta rule to both selectively forget and intelligently update.
