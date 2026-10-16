---
title: Delta-Rule Writes in Linear Attention
tags: linear-attention, TMLR, talk
description: View the slide with "Slide Mode".
---

# Delta-Rule Writes in Linear Attention

### A Controlled Factorial Study of Decay Design

<div style="font-size:26px;margin-top:40px;">Satyajit Das · TMLR 09/2026</div>
<div style="font-size:22px;color:#657b83;margin-top:10px;">openreview.net/forum?id=FdXB4u27LD</div>

<div style="font-size:34px;margin-top:70px;">How should a transformer <strong style="color:#8a2332;">forget</strong>?</div>

Note: One plain question. Linear attention must throw information away; the paper asks what the well-designed way looks like.

---

<style>
.markdown-body, .ui-infobar { max-width: unset !important; }
.flowbox { border:2px solid #93a1a1; padding:14px 10px; min-height:62px; display:flex; align-items:center; justify-content:center; line-height:1.3; flex:1; min-width:0; }
.flowarr { font-size:34px; color:#657b83; display:flex; align-items:center; }
.statline { border-top:1px solid #93a1a1; padding-top:12px; }
</style>

## The whole model, end to end — 18M config

<div style="display:flex;align-items:center;gap:10px;font-size:20px;margin-top:30px;">
<div class="flowbox">"the cat sat<br>on the ___"</div><div class="flowarr">→</div>
<div class="flowbox"><div>embedding<br>+ position<div style="color:#657b83;font-size:16px;margin-top:4px;">x<sub>t</sub> · 256-dim</div></div></div><div class="flowarr">→</div>
<div class="flowbox" style="flex:2.4;border-color:#8a2332;"><div style="display:flex;gap:6px;width:100%;justify-content:center;">
<div style="border:1px solid #93a1a1;padding:8px 6px;font-size:17px;">L1</div>
<div style="border:1px solid #93a1a1;padding:8px 6px;font-size:17px;">L2</div>
<div style="border:2px solid #8a2332;padding:8px 6px;font-size:17px;color:#8a2332;"><b>L3</b></div>
<div style="border:1px solid #93a1a1;padding:8px 6px;font-size:17px;">L4</div>
<div style="border:1px solid #93a1a1;padding:8px 6px;font-size:17px;">L5</div>
<div style="border:1px solid #93a1a1;padding:8px 6px;font-size:17px;">L6</div>
</div></div><div class="flowarr">→</div>
<div class="flowbox">norm ·<br>LM head</div><div class="flowarr">→</div>
<div class="flowbox">logits →<br>softmax</div>
</div>

<div style="font-size:20px;color:#8a2332;margin:22px 0 10px;">⌄ layer 3, opened — every layer is identical ⌄</div>

<div style="display:flex;align-items:center;gap:10px;font-size:20px;max-width:760px;margin:0 auto;">
<div class="flowbox" style="border-color:#8a2332;flex:2.6;"><div>
<div style="color:#8a2332;"><b>attention</b> · 4 heads × d=64</div>
<div style="font-size:16px;color:#657b83;margin-top:6px;">concat · 4 × 64 = 256</div>
</div></div>
<div class="flowarr" style="color:#8a2332;">→ o<sub>t</sub> →</div>
<div class="flowbox"><div>MLP<div style="font-size:16px;color:#657b83;margin-top:4px;">per-token</div></div></div>
</div>

<div style="font-size:19px;color:#657b83;margin-top:26px;">one o<sub>t</sub> per token, per layer · @125M: 12 layers, 12 heads, D = 768 · this paper changes <b>only the attention box</b></div>

Note: Compress this for an NLP audience — "you all know this picture". One o per token per layer. Everything stays fixed except the attention sublayer.

---

## From softmax to linear attention

<div style="display:flex;align-items:stretch;gap:10px;text-align:left;margin-top:40px;">
<div style="flex:1;min-width:0;border-top:3px solid #657b83;padding:16px 8px;">
<div style="font-size:20px;color:#657b83;margin-bottom:14px;">01</div>
<h3 style="font-size:24px;text-transform:none;margin:0 0 14px;">Softmax<br>attention</h3>
<p style="font-size:19px;line-height:1.35;margin:0;">$o_t=\tfrac{\sum_{i\le t}e^{q_t^\top k_i/\sqrt d}\,v_i}{\sum_{i\le t}e^{q_t^\top k_i/\sqrt d}}$</p>
<p style="font-size:17px;line-height:1.35;color:#657b83;margin:10px 0 0;">exp couples q to every stored k → keep all keys & values · O(n²)</p>
</div>
<div class="flowarr">→</div>
<div style="flex:1;min-width:0;border-top:3px solid #657b83;padding:16px 8px;">
<div style="font-size:20px;color:#657b83;margin-bottom:14px;">02</div>
<h3 style="font-size:24px;text-transform:none;margin:0 0 14px;">Swap<br>the kernel</h3>
<p style="font-size:19px;line-height:1.35;margin:0;">$e^{q^\top k/\sqrt d}\!\to\!\phi(q)^\top\phi(k)$</p>
<p style="font-size:17px;line-height:1.35;color:#657b83;margin:10px 0 0;">a <b>replacement</b>, not an approximation — a different model, retrained · φ = ELU+1</p>
</div>
<div class="flowarr">→</div>
<div style="flex:1;min-width:0;border-top:3px solid #657b83;padding:16px 8px;">
<div style="font-size:20px;color:#657b83;margin-bottom:14px;">03</div>
<h3 style="font-size:24px;text-transform:none;margin:0 0 14px;">Regroup</h3>
<p style="font-size:19px;line-height:1.35;margin:0;">$\phi(q_t)^\top\big(\textstyle\sum\phi(k_i)v_i^\top\big)$</p>
<p style="font-size:17px;line-height:1.35;color:#657b83;margin:10px 0 0;">the query leaves the sum — the bracket no longer depends on q</p>
</div>
<div class="flowarr">→</div>
<div style="flex:1;min-width:0;border-top:3px solid #657b83;padding:16px 8px;">
<div style="font-size:20px;color:#657b83;margin-bottom:14px;">04</div>
<h3 style="font-size:24px;text-transform:none;margin:0 0 14px;">Keep<br>running sums</h3>
<p style="font-size:19px;line-height:1.35;margin:0;">$S_t=S_{t-1}+\phi(k_t)v_t^\top$ · $o_t=S_t^\top\phi(q_t)$</p>
<p style="font-size:17px;line-height:1.35;color:#657b83;margin:10px 0 0;">$S\in\mathbb{R}^{64\times 64}$ per head · one write + one read per token</p>
</div>
</div>

<div style="font-size:20px;margin-top:30px;"><span style="color:#8a2332;"><b>the swap buys cost</b></span> — no growing KV cache, constant work per token. <b>the risk is quality</b> — that's what the experiments measure.</div>

Note: Say "replacement, not approximation" out loud. φ(q) top and bottom does NOT cancel: vector over scalar. The paper uses the unnormalized form (z dropped).

---

## The state must forget — one gate, three choices

<div style="font-size:26px;margin:24px 0;">$S_t=\color{#8a2332}{G_t}\odot S_{t-1}+\kappa_t u_t^\top,\qquad o_t=S_t^\top\phi(q_t)$</div>

<div style="display:flex;gap:30px;text-align:left;font-size:20px;">
<div style="flex:1;" class="statline"><b>1 · Granularity</b><br><span style="color:#657b83;">one scalar γ per head, or Γ ∈ ℝ⁶⁴ — a rate per <b>row</b> of S</span>
<div style="margin-top:14px;display:flex;align-items:center;gap:10px;">
<div style="display:flex;flex-direction:column;gap:2px;">
<div style="display:flex;gap:4px;align-items:center;"><span style="color:#8a2332;font-size:14px;width:26px;text-align:right;">γ₁</span><span style="display:inline-block;width:104px;height:13px;background:#ececec;"></span></div>
<div style="display:flex;gap:4px;align-items:center;"><span style="color:#8a2332;font-size:14px;width:26px;text-align:right;">γ₂</span><span style="display:inline-block;width:104px;height:13px;background:#d9d9d9;"></span></div>
<div style="display:flex;gap:4px;align-items:center;"><span style="color:#8a2332;font-size:14px;width:26px;text-align:right;">γ₃</span><span style="display:inline-block;width:104px;height:13px;background:#c5c5c5;"></span></div>
<div style="display:flex;gap:4px;align-items:center;"><span style="color:#8a2332;font-size:14px;width:26px;text-align:right;">⋮</span><span style="color:#aaa;font-size:12px;width:104px;text-align:center;">⋮</span></div>
<div style="display:flex;gap:4px;align-items:center;"><span style="color:#8a2332;font-size:14px;width:26px;text-align:right;">γ₆₄</span><span style="display:inline-block;width:104px;height:13px;background:#8e8e8e;"></span></div>
</div>
<div style="font-size:15px;color:#657b83;">S · 64×64<br>one γ per row<br>(= one channel)</div>
</div></div>
<div style="flex:1;" class="statline"><b>2 · Conditioning</b><br><span style="color:#657b83;">data-dependent $G_t=\sigma(x_tW_g)$ — per token<br>or static Γ = σ(ℓ) — fixed after training</span>
<p style="font-size:17px;color:#657b83;margin-top:12px;">static init: γᵢ = exp(−2^{−8i/d}), 0.368 → 0.996 (half-life &lt;1 → ≈163 tokens)</p></div>
<div style="flex:1;" class="statline"><b>3 · The write (delta rule)</b><br><span style="color:#657b83;">blind: κ=φ(k), u=v<br>delta: κ=k̂, u = v − S<sub>t−1</sub>ᵀk̂ — <b>write only the error</b></span>
<p style="font-size:16px;color:#657b83;margin-top:12px;">Widrow–Hoff 1960 · into linear attention: Schlag, Irie & Schmidhuber 2021 · corrects S at <b>write time</b>, every token — inference too, no gradients</p></div>
</div>

<div style="font-size:19px;margin-top:26px;color:#657b83;"><b style="color:#1a1a1a;">mental model</b> — MLP weights: remember (KV store from training) · attention: what to read from the past, and how (in-context learning) · S: KV scratchpad for the current inference, rebuilt from zero</div>

Note: Point at the γ rows: channel-wise = each row its own dial; scalar = all equal; static = bolted after training. Delta correction happens in the forward pass at write time — not gradient descent.

---

## Inside the data-dependent gate — $G_t=\sigma(x_t W_g)$

<div style="display:flex;align-items:center;gap:10px;font-size:19px;margin-top:36px;">
<div class="flowbox"><div>x<sub>t</sub><div style="font-size:16px;color:#657b83;margin-top:4px;">768 numbers @125M</div></div></div>
<div class="flowarr">×W<sub>g</sub>→</div>
<div class="flowbox"><div>64 raw scores<div style="font-size:16px;color:#657b83;margin-top:4px;">one dot product per column · any real</div></div></div>
<div class="flowarr">σ→</div>
<div class="flowbox" style="border-color:#8a2332;color:#8a2332;"><div><b>G<sub>t</sub> · 64 keep-fractions</b><div style="font-size:16px;margin-top:4px;">≈1 keep this row · ≈0 wipe it</div></div></div>
<div class="flowarr">⊙→</div>
<div class="flowbox"><div>S · 64×64<div style="font-size:16px;color:#657b83;margin-top:4px;">row i × G<sub>t</sub>[i], every token</div></div></div>
</div>

<div style="font-size:19px;color:#657b83;margin:26px 0;">each <b>column</b> of W<sub>g</sub> = one learned question about the context · σ (the <b>sigmoid</b>) squashes any score into (0,1) — a valid keep-fraction</div>

<div style="display:flex;gap:60px;justify-content:center;margin-top:30px;">
<div><div style="font-size:54px;font-weight:600;">7,077,888</div><div style="font-size:20px;color:#657b83;margin-top:8px;">the wiring: 768·64 per head × 12 heads × 12 layers<br>(data-<b>dependent</b> gate)</div></div>
<div><div style="font-size:54px;font-weight:600;">9,216</div><div style="font-size:20px;color:#657b83;margin-top:8px;">delete W<sub>g</sub>, keep the dials: 64 × 12 × 12<br>(static gate)</div></div>
</div>

Note: Say the arithmetic out loud once — it's what makes the final takeaway land. 768×64 = 49,152 per head.

---

## The 2×2×2 design space

<div style="display:flex;gap:30px;align-items:center;">
<div style="flex:1.1;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/design_cube_fact.png" style="max-height:480px;border:0;"></div>
<div style="flex:1;text-align:left;font-size:21px;line-height:1.45;">
<p><span style="color:#8a2332;">–</span> existing, data-dependent: ScalarDynamic (GLA), ScalarDynamicDelta (DeltaNet), DynamicChannelDelta (KDA); DynamicChannel (RWKV) left hollow</p>
<p><span style="color:#8a2332;">–</span> new: the four <b>Static</b> variants — the data-independent half, crossed with the delta rule</p>
<p><span style="color:#8a2332;">–</span> softmax: outside the cube, as baseline</p>
<p style="font-size:17px;color:#657b83;">Figure 1</p>
</div>
</div>

Note: Studied = the named data-dependent corners. New = the static half. 8 corners + softmax − RWKV = 8 trained variants.

---

## Controlled, fixed-budget protocol

<ul style="font-size:23px;line-height:1.5;text-align:left;margin-top:30px;">
<li>GPT-style decoder; <b>only the attention sublayer changes</b></li>
<li>18M (6L·4H·d256) and 125M (12L·12H·d768): all 8 variants × 2 corpora × seeds 42/123/7 <span style="color:#657b83;font-size:19px;">(42M: single-seed check, 5 variants)</span></li>
<li>TinyStories + WikiText-103 · 10M tokens · seq 512 · AdamW, identical hyperparameters</li>
<li>metric: validation cross-entropy (nats) · <b>all individual seeds reported</b></li>
</ul>

<div style="border-top:1px solid #93a1a1;margin-top:30px;padding-top:18px;font-size:19px;color:#657b83;text-align:left;">scope: fixed-budget comparisons, uniformly simplified delta variants, naive PyTorch loops (softmax 20 ms/step vs SCD 1,762 ms) — interaction structure, not SOTA</div>

Note: Own the caveat: naive loops, chunkwise kernels are the known fix (future work).

---

## Finding 1 — the same top 4, in every seed × dataset

<img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/multiseed_ranks_fact.png" style="max-height:480px;border:0;">

<div style="font-size:20px;color:#657b83;margin-top:14px;">Figure 2 · ranks 1–4 are always the four delta-rule variants; no blind writer — and not softmax — ever enters the top 4</div>

Note: The green block never breaks. On WikiText, StaticChannelDelta is rank 1 in all three seeds; softmax ranks 5 (TS) / 7 (WT).

---

## The delta divide

<div style="display:flex;gap:30px;align-items:center;">
<div style="flex:1.4;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/multiseed_bars_fact.png" style="max-height:430px;border:0;">
<div style="font-size:18px;color:#657b83;">Figure 3 · mean ± std over 3 seeds; dots = individual seeds</div></div>
<div style="flex:1;text-align:left;">
<div style="font-size:22px;margin-bottom:14px;">worst delta vs best non-delta:</div>
<table style="width:100%;font-size:22px;border-collapse:collapse;">
<thead><tr><th style="font-size:17px;font-weight:400;color:#657b83;padding:8px 0;">SETTING</th><th style="font-size:17px;font-weight:400;color:#657b83;">GAP (NATS)</th></tr></thead>
<tbody>
<tr><td style="padding:10px 0;border-top:1px solid #93a1a1;">18M · TinyStories</td><td style="border-top:1px solid #93a1a1;"><b>0.061</b></td></tr>
<tr><td style="padding:10px 0;border-top:1px solid #93a1a1;">125M · TinyStories</td><td style="border-top:1px solid #93a1a1;"><b>0.057</b></td></tr>
<tr><td style="padding:10px 0;border-top:1px solid #93a1a1;">125M · WikiText-103</td><td style="border-top:1px solid #93a1a1;"><b>0.072</b></td></tr>
</tbody></table>
<div style="font-size:21px;margin-top:18px;color:#657b83;">vs softmax at 125M on WikiText-103: <b style="color:#1a1a1a;">0.217</b></div>
</div>
</div>

Note: Small numbers, but the separation shows up in every cell of the experiment.

---

## Finding 2 — channel-wise decay helps only with the delta rule

<div style="display:flex;gap:50px;font-size:21px;margin-top:26px;">
<div style="flex:1;">
<div style="margin-bottom:10px;"><b>TinyStories</b> · 18M</div>
<table style="width:100%;font-size:21px;border-collapse:collapse;">
<thead><tr><th></th><th style="font-size:16px;color:#657b83;font-weight:400;">SCALAR</th><th style="font-size:16px;color:#657b83;font-weight:400;">CHANNEL</th><th style="font-size:16px;color:#657b83;font-weight:400;">Δ</th></tr></thead>
<tbody>
<tr><td style="padding:9px 0;border-top:1px solid #93a1a1;">blind</td><td style="border-top:1px solid #93a1a1;">2.4307</td><td style="border-top:1px solid #93a1a1;">2.5491</td><td style="border-top:1px solid #93a1a1;color:#8a2332;"><b>+0.118</b></td></tr>
<tr><td style="padding:9px 0;border-top:1px solid #93a1a1;">delta</td><td style="border-top:1px solid #93a1a1;">2.3015</td><td style="border-top:1px solid #93a1a1;">2.2591</td><td style="border-top:1px solid #93a1a1;color:#2e7d32;"><b>−0.042</b></td></tr>
</tbody></table>
</div>
<div style="flex:1;">
<div style="margin-bottom:10px;"><b>WikiText-103</b> · 18M</div>
<table style="width:100%;font-size:21px;border-collapse:collapse;">
<thead><tr><th></th><th style="font-size:16px;color:#657b83;font-weight:400;">SCALAR</th><th style="font-size:16px;color:#657b83;font-weight:400;">CHANNEL</th><th style="font-size:16px;color:#657b83;font-weight:400;">Δ</th></tr></thead>
<tbody>
<tr><td style="padding:9px 0;border-top:1px solid #93a1a1;">blind</td><td style="border-top:1px solid #93a1a1;">5.0884</td><td style="border-top:1px solid #93a1a1;">5.1505</td><td style="border-top:1px solid #93a1a1;color:#8a2332;"><b>+0.062</b></td></tr>
<tr><td style="padding:9px 0;border-top:1px solid #93a1a1;">delta</td><td style="border-top:1px solid #93a1a1;">4.9704</td><td style="border-top:1px solid #93a1a1;">4.9207</td><td style="border-top:1px solid #93a1a1;color:#2e7d32;"><b>−0.050</b></td></tr>
</tbody></table>
</div>
</div>

<div style="display:flex;gap:60px;justify-content:center;margin-top:40px;">
<div><div style="font-size:50px;font-weight:600;">0.161 <span style="font-size:26px;">≈ 77%</span></div><div style="font-size:19px;color:#657b83;margin-top:6px;">the <b>interaction</b> — most of the whole delta effect (TS)</div></div>
<div><div style="font-size:50px;font-weight:600;">+0.112</div><div style="font-size:19px;color:#657b83;margin-top:6px;">interaction on WikiText — sign-consistent in every seed</div></div>
</div>

<div style="font-size:20px;color:#657b83;margin-top:26px;">many timescales accumulate interference — unless the write corrects it</div>

Note: SAY the word "interaction" — the same architectural change flips sign depending on the write rule. Only a factorial design shows this. Scope: 18M, both corpora; within-delta the benefit persists at 125M on WikiText.

---

## Finding 3 — the winner's gates are static (you can see it)

<div style="display:flex;gap:24px;align-items:flex-start;margin-top:26px;">
<div style="flex:1;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/static_channel_delta_channel_gates.png" style="border:0;">
<div style="font-size:17px;color:#657b83;margin-top:8px;">Figure 5 · StaticChannelDelta — smooth bands, init spectrum largely preserved</div></div>
<div style="flex:1;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/kda_channel_gates_fact.png" style="border:0;">
<div style="font-size:17px;color:#657b83;margin-top:8px;">Figure 7 · DynamicChannelDelta (KDA) — gates recomputed every token</div></div>
</div>

<div style="font-size:21px;margin-top:30px;">18M, both corpora: StaticChannelDelta <b>2.2591 / 4.9207</b> beats DynamicChannelDelta (KDA) <b>2.2820 / 4.9353</b> — 9,216 fixed dials vs a ~7M-param gate network</div>

Note: These heatmaps ARE the gate G: x = sequence position, y = channel, color = keep-fraction. Static → identical columns = bands. The ~7M network is the data-DEPENDENT one.

---

## The divide is early — and the recurrence is stable

<div style="display:flex;gap:24px;align-items:flex-start;margin-top:26px;">
<div style="flex:1;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/val_loss_curves.png" style="border:0;">
<div style="font-size:17px;color:#657b83;margin-top:8px;">Figure 4 · delta / non-delta separation visible from the first validation checkpoint</div></div>
<div style="flex:1;"><img src="https://emailtovamos.github.io/delta-rule-linear-attention/talk_figs/length_generalization.png" style="border:0;">
<div style="font-size:17px;color:#657b83;margin-top:8px;">Figure 6 · NoPE, trained at 2K, evaluated to 8K — <span style="color:#8a2332;">stability check, not long-range retrieval</span></div></div>
</div>

<div style="font-size:19px;color:#657b83;margin-top:24px;">linear variants never see the length — each step reads the same fixed-size S; softmax's read spans everything stored, so its read changes shape beyond the training length</div>

Note: Not the n² cost — distribution shift: softmax normalizes over 4× more keys than it ever saw. Perplexity = e^loss.

---

## Takeaways

<ul style="font-size:24px;line-height:1.55;text-align:left;margin-top:34px;">
<li style="margin-bottom:22px;"><b>Correct, don't accumulate.</b> The delta rule is the strongest observed factor — top 4 in every cell, at every tested scale.</li>
<li style="margin-bottom:22px;"><b>Many timescales need the eraser.</b> Channel-wise decay helps only with corrective writes (18M, both corpora; within-delta it persists at 125M on WikiText-103).</li>
<li style="margin-bottom:22px;"><b>Forgetting needn't watch the text.</b> 9,216 static dials matched a ~7M-parameter data-dependent gate, at these scales.</li>
</ul>

<div style="border-top:1px solid #93a1a1;margin-top:26px;padding-top:16px;font-size:19px;color:#657b83;text-align:left;">limits: ≤125M, 3 seeds, fixed budget, no retrieval benchmarks · code: pure PyTorch, every seed reported</div>

---

## Future work

<ul style="font-size:24px;line-height:1.55;text-align:left;margin-top:40px;">
<li style="margin-bottom:22px;"><b>Scale.</b> Rerun the cube at 500M–1B: do the delta divide and the static parity hold?</li>
<li style="margin-bottom:22px;"><b>Retrieval.</b> Long-range recall (needle-in-a-haystack) — the length results here are stability checks only.</li>
<li style="margin-bottom:22px;"><b>Speed.</b> Chunkwise kernels for the static gate (the decay blocks precompute) — close the wall-clock gap.</li>
<li style="margin-bottom:22px;"><b>Theory.</b> A formal account of the interference story behind Finding 2.</li>
</ul>

---

# Thank you

<div style="font-size:32px;color:#8a2332;margin-top:20px;"><b>questions?</b></div>

<div style="border-top:1px solid #93a1a1;max-width:800px;margin:50px auto 0;padding-top:20px;font-size:20px;color:#657b83;">
paper: openreview.net/forum?id=FdXB4u27LD<br>
code: github.com/emailtovamos/delta-rule-linear-attention<br>
slides: emailtovamos.github.io/delta-rule-linear-attention
</div>

---

## Backup · slow memory, fast memory — and the softmax trade

<div style="display:flex;gap:40px;text-align:left;font-size:20px;margin-top:30px;">
<div style="flex:1;" class="statline"><b>two slow-vs-fast axes — don't mix them</b>
<p style="color:#657b83;margin-top:12px;"><b style="color:#1a1a1a;">1 · when is it written?</b> MLP weights = slow memory: written by training, frozen at inference. S = fast memory: written token by token, wiped after the sequence.</p>
<p style="color:#657b83;"><b style="color:#1a1a1a;">2 · how fast is it forgotten?</b> inside S, each row has its own speed: γ = 0.368 forgets within a token · γ = 0.996 remembers ≈ 163 tokens — one state, a spectrum of timescales.</p>
</div>
<div style="flex:1;" class="statline"><b>what softmax was buying</b>
<p style="color:#657b83;margin-top:12px;">softmax = a <b style="color:#1a1a1a;">soft argmax</b>: positive weights summing to 1, keys compete, best match wins sharply → near-exact retrieval.</p>
<p style="color:#657b83;">the price: exp ties q to every stored k → the KV cache. drop it → constant-size S, but matching becomes <b style="color:#1a1a1a;">linear and blurry</b> — similar keys interfere.</p>
</div>
</div>

<div style="font-size:21px;margin-top:36px;">the trade: a constant-size S gives up sharp retrieval — <b>corrective (delta) writes win some of it back</b> when keys collide</div>

Note: Backup for Q&A on Mamba/SSMs too: Mamba ≈ the data-dependent, no-delta region of the cube (selective = input-dependent gate); this study says the delta/write axis mattered most, and data-dependence was replaceable with static dials.
