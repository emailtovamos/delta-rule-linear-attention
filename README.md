# Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design

Code, scripts, and training logs for the TMLR paper of the same name ([OpenReview](https://openreview.net/forum?id=FdXB4u27LD)). A controlled fixed-budget study of decay mechanisms in linear attention.

## Research Question

Linear attention models compress sequence history into a fixed-size state matrix. The critical design choice is **how decay/forgetting works** -- it determines what information is retained and what is lost. We evaluate seven linear-attention variants plus a standard softmax baseline. The data-independent subset forms a complete 2x2 comparison over granularity and the delta rule; the data-dependent variants provide controlled pairwise comparisons.

- **Granularity**: scalar vs. channel-wise gating
- **Conditioning**: data-dependent vs. data-independent gates
- **Delta rule**: present or absent

```
                      Data-Dependent        Data-Independent
                   +------------------+----------------------+
  Scalar gate      | GLA*, DeltaNet   | ScalarStatic,        |
                   |                  | ScalarStaticDelta    |
                   +------------------+----------------------+
  Channel-wise     | KDA              | StaticChannel,       |
  gate             |                  | StaticChannelDelta   |
                   +------------------+----------------------+

  * Our GLA variant uses the scalar special case of the published
    GLA's low-rank per-element gate, for factorial control.
  + The channel-wise, data-dependent, no-delta cell is not implemented.
  + Standard softmax attention serves as baseline (8th variant).
```

## Key Findings

1. **Delta rule is the dominant observed fixed-budget factor** -- all delta variants (ranks 1-4) beat all non-delta variants (ranks 5-8) at 18M and 125M on both datasets; worst-delta vs. best-non-delta gap at 125M: 0.057 (TinyStories) / 0.072 (WikiText-103; 0.217 vs. softmax).
2. **Granularity x delta interaction at 18M** -- in the complete data-independent 2x2, channel-wise decay hurts without delta (+0.118 TinyStories / +0.062 WikiText) but helps with it (-0.042 / -0.050), a crossover interaction. At 125M it is not detectable on TinyStories; on WikiText only the delta-side benefit remains resolvable (see the WikiText-103 results below).
3. **Within-delta leaders are not robustly ordered at 125M** -- StaticChannelDelta leads at 18M on both datasets; at 125M the small residual gap flips sign across datasets (KDA ahead on TinyStories, StaticChannelDelta on WikiText-103), each within seed variance.

## Multi-Seed Results

### TinyStories, 18M params (3 seeds: 42, 123, 7)

| Rank | Model | Mean +/- Std |
|------|-------|-------------|
| 1 | **StaticChannelDelta** | **2.2591 +/- 0.012** |
| 2 | KDA | 2.2820 +/- 0.020 |
| 3 | ScalarStaticDelta | 2.3015 +/- 0.005 |
| 4 | DeltaNet | 2.3222 +/- 0.017 |
| 5 | Standard | 2.3833 +/- 0.001 |
| 6 | GLA | 2.4207 +/- 0.004 |
| 7 | ScalarStatic | 2.4307 +/- 0.005 |
| 8 | StaticChannel | 2.5491 +/- 0.014 |

### TinyStories, 125M params (3 seeds: 42, 123, 7)

| Rank | Model | Mean +/- Std |
|------|-------|-------------|
| 1 | KDA | 1.8614 +/- 0.002 |
| 2 | StaticChannelDelta | 1.8687 +/- 0.006 |
| 3 | ScalarStaticDelta | 1.8705 +/- 0.005 |
| 4 | DeltaNet | 1.9004 +/- 0.006 |
| 5 | Standard | 1.9570 +/- 0.008 |
| 6 | GLA | 1.9710 +/- 0.007 |
| 7 | ScalarStatic | 2.0075 +/- 0.008 |
| 8 | StaticChannel | 2.0079 +/- 0.005 |

### WikiText-103, 125M params (3 seeds: 42, 123, 7)

| Rank | Model | Mean +/- Std |
|------|-------|-------------|
| 1 | **StaticChannelDelta** | **4.3777 +/- 0.005** |
| 2 | KDA | 4.3787 +/- 0.005 |
| 3 | ScalarStaticDelta | 4.4133 +/- 0.003 |
| 4 | DeltaNet | 4.4171 +/- 0.014 |
| 5 | GLA | 4.4887 +/- 0.005 |
| 6 | ScalarStatic | 4.6217 +/- 0.016 |
| 7 | StaticChannel | 4.6269 +/- 0.008 |
| 8 | Standard | 4.6337 +/- 0.003 |

All reported standard deviations are population standard deviations. Manuscript results
use the pre-specified seeds 42, 123, and 7. The WikiText-103 125M data-independent logs
for seeds 1234 and 2024 are post-hoc exploratory repeats and are not used in the paper's
tables or claims.

## Setup

```bash
cd delta-rule-linear-attention
uv sync
```

### Hardware
- **Development**: MacBook Pro M4 Max 96GB RAM, PyTorch MPS backend
- All attention implementations are naive recurrent loops (no Triton kernels)
- Also works on CUDA GPUs

## Project Structure

```
delta-rule-linear-attention/
+-- src/
|   +-- attention.py       # All 8 attention variants (+ 2 experimental)
|   +-- model.py           # GPT with swappable attention (18M/42M/125M presets)
|   +-- data.py            # TinyStories + WikiText-103, GPT-2 BPE tokenization
+-- scripts/
|   +-- train.py           # Main training loop
|   +-- eval_length_gen.py # Length generalization / recurrence stability
|   +-- plot_multiseed.py  # Rank heatmaps and bar charts
|   +-- plot_training_curves.py
|   +-- plot_length_gen.py
|   +-- plot_design_cube.py      # 3D design space figure (v1, matplotlib 3D)
|   +-- plot_design_cube_v2.py   # 2D isometric design cube (v2, used in paper)
|   +-- visualize_retention.py   # Channel retention heatmaps
|   +-- ablation.py        # Decay schedule ablation (early-phase)
+-- results/               # Per-run training logs (log.json) and figures (checkpoints excluded)
|   +-- {variant}_baseline/       # 18M, seed 42
|   +-- {variant}_seed123/        # 18M, seed 123
|   +-- {variant}_seed7/          # 18M, seed 7
|   +-- wiki_{variant}*/          # WikiText-103 runs
|   +-- small_{variant}/          # 42M scale-up (top-4 delta + standard)
|   +-- {variant}_125m_s{seed}/   # 125M scale-up (all 8 variants, 3 seeds)
|   +-- length_gen/               # Length generalization results
|   +-- visualizations/           # Generated plots
+-- explanation.md         # What a "channel" means in channel-wise forgetting
+-- pyproject.toml         # Dependencies (managed by uv)
+-- uv.lock
```

## Common Commands

```bash
# Train a variant
uv run python scripts/train.py --attention static_channel_delta --name scd_baseline --seed 42
uv run python scripts/train.py --attention kda --name kda_125m_s42 --size medium --seed 42
uv run python scripts/train.py --attention deltanet --name wiki_deltanet --dataset wikitext --seed 42

# Generate plots
uv run python scripts/plot_multiseed.py
uv run python scripts/plot_training_curves.py
uv run python scripts/plot_length_gen.py
uv run python scripts/plot_design_cube_v2.py
```

## 8 Variant Reference

| Variant | Granularity | Conditioning | Delta? | `--attention` arg |
|---------|-------------|-------------|--------|-------------------|
| Standard | softmax | -- | -- | `standard` |
| GLA | Scalar* | Data-dependent | No | `gla` |
| DeltaNet | Scalar | Data-dependent | Yes | `deltanet` |
| KDA | Channel-wise | Data-dependent | Yes | `kda` |
| ScalarStatic | Scalar | Data-independent | No | `scalar_static` |
| ScalarStaticDelta | Scalar | Data-independent | Yes | `scalar_static_delta` |
| StaticChannel | Channel-wise | Data-independent | No | `static_channel` |
| StaticChannelDelta | Channel-wise | Data-independent | Yes | `static_channel_delta` |

\* Our GLA variant uses the scalar special case of the published GLA architecture (which uses a low-rank per-element gate) for clean factorial control.

## Key References

- **GLA**: Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (2023)
- **DeltaNet / GDN**: Yang et al., "Gated DeltaNet" (2025)
- **KDA / Kimi Linear**: Zhang et al., "Kimi Linear: An Expressive, Efficient Attention Architecture" (2025)
- **HALO**: Chen et al., "Hybrid Linear Attention Done Right" (2026)

## License

MIT
