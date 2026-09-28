"""
Plot multi-seed comparison: mean ± std for all variants across datasets.

Usage:
    uv run python scripts/plot_multiseed.py
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

VARIANTS = [
    ("StaticChannelDelta", "static_channel_delta"),
    ("KDA", "kda"),
    ("ScalarStaticDelta", "scalar_static_delta"),
    ("DeltaNet", "deltanet"),
    ("Standard", "standard"),
    ("GLA", "gla"),
    ("ScalarStatic", "scalar_static"),
    ("StaticChannel", "static_channel"),
]

COLORS = {
    "StaticChannelDelta": "#d62728",
    "KDA": "#1f77b4",
    "ScalarStaticDelta": "#e377c2",
    "DeltaNet": "#ff7f0e",
    "Standard": "#7f7f7f",
    "GLA": "#2ca02c",
    "ScalarStatic": "#8c564b",
    "StaticChannel": "#9467bd",
}

# Map: (dataset_label, list of (seed, [patterns_to_try]))
# seed 42 uses "{variant}_baseline" or "{variant}_seed42" / "wiki_{variant}" or "wiki_{variant}_seed42"
# seed 123 uses "{variant}_seed123" / "wiki_{variant}_seed123"
# seed 7 uses "{variant}_seed7" / "wiki_{variant}_seed7"
DATASETS = {
    "TinyStories": [
        (42, ["{variant}_baseline", "{variant}_seed42"]),
        (123, ["{variant}_seed123"]),
        (7, ["{variant}_seed7"]),
    ],
    "WikiText-103": [
        (42, ["wiki_{variant}", "wiki_{variant}_seed42"]),
        (123, ["wiki_{variant}_seed123"]),
        (7, ["wiki_{variant}_seed7"]),
    ],
}

results_dir = Path("results")


def load_val_loss(dir_name):
    log_path = results_dir / dir_name / "log.json"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        log = json.load(f)
    val_losses = [l for _, l in log["val_loss"]]
    return val_losses[-1]


def try_load_val_loss(patterns, variant_key):
    """Try multiple directory patterns, return first match."""
    for pattern in patterns:
        dir_name = pattern.format(variant=variant_key)
        val_loss = load_val_loss(dir_name)
        if val_loss is not None:
            return val_loss, dir_name
    return None, None


def collect_results():
    """Collect all available seed results into a structured dict."""
    results = {}  # {dataset: {variant: [losses]}}
    for dataset, seed_patterns in DATASETS.items():
        results[dataset] = {}
        for display_name, variant_key in VARIANTS:
            losses = []
            for seed, patterns in seed_patterns:
                val_loss, found_dir = try_load_val_loss(patterns, variant_key)
                if val_loss is not None:
                    losses.append((seed, val_loss))
                    print(f"  {dataset} / {display_name} / seed {seed}: {val_loss:.4f} ({found_dir})")
                else:
                    print(f"  {dataset} / {display_name} / seed {seed}: NOT FOUND")
            results[dataset][display_name] = losses
    return results


def main():
    print("Collecting multi-seed results...\n")
    results = collect_results()

    # Print summary table
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY")
    print("=" * 80)

    for dataset in DATASETS:
        print(f"\n{dataset}:")
        print(f"  {'Variant':<22} {'Seeds':>5}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*22} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        # Sort by mean
        items = []
        for display_name, _ in VARIANTS:
            seed_losses = results[dataset][display_name]
            if seed_losses:
                vals = [l for _, l in seed_losses]
                items.append((display_name, vals))
        items.sort(key=lambda x: np.mean(x[1]))

        for name, vals in items:
            mean = np.mean(vals)
            std = np.std(vals, ddof=0) if len(vals) > 1 else 0
            print(f"  {name:<22} {len(vals):>5}  {mean:>8.4f}  {std:>8.4f}  {min(vals):>8.4f}  {max(vals):>8.4f}")

    # === Figure 1: Bar chart with error bars ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Validation Loss by Variant (Mean ± Std across Seeds)",
                 fontsize=13, fontweight='bold')

    for ax, dataset in zip(axes, DATASETS):
        names = []
        means = []
        stds = []
        colors = []
        individual_points = []

        for display_name, _ in VARIANTS:
            seed_losses = results[dataset][display_name]
            if seed_losses:
                vals = [l for _, l in seed_losses]
                names.append(display_name)
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0)
                colors.append(COLORS[display_name])
                individual_points.append(vals)

        # Sort by mean (best first for horizontal bars = bottom)
        order = np.argsort(means)[::-1]  # reverse for barh (top=worst)
        names = [names[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]
        colors = [colors[i] for i in order]
        individual_points = [individual_points[i] for i in order]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, means, xerr=stds, color=colors,
                       edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)

        # Overlay individual seed points
        for i, pts in enumerate(individual_points):
            ax.scatter(pts, [i] * len(pts), color='black', s=25, zorder=5, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Validation Loss")
        ax.set_title(dataset)

        # Value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            n_seeds = len(individual_points[i])
            label = f'{m:.4f}±{s:.4f}' if n_seeds > 1 else f'{m:.4f}'
            ax.text(m + s + 0.01, i, label, va='center', fontsize=8)

    plt.tight_layout()
    out_path = results_dir / "visualizations" / "multiseed_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()

    # === Figure 2: Seed-by-seed scatter with connecting lines ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Per-Seed Validation Loss (each dot = one seed)",
                 fontsize=13, fontweight='bold')

    for ax, dataset in zip(axes, DATASETS):
        for display_name, _ in VARIANTS:
            seed_losses = results[dataset][display_name]
            if seed_losses:
                seeds = [s for s, _ in seed_losses]
                vals = [l for _, l in seed_losses]
                ax.scatter(seeds, vals, color=COLORS[display_name], s=80,
                          zorder=5, edgecolors='black', linewidth=0.5)
                ax.plot(seeds, vals, color=COLORS[display_name], alpha=0.4,
                       linestyle='--', linewidth=1)
                # Label at mean position
                mean_val = np.mean(vals)
                ax.annotate(display_name, (seeds[-1], vals[-1]),
                          textcoords="offset points", xytext=(10, 0),
                          fontsize=7, color=COLORS[display_name])

        ax.set_xlabel("Seed")
        ax.set_ylabel("Validation Loss")
        ax.set_title(dataset)
        ax.set_xticks([s for s, _ in DATASETS[dataset]])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "visualizations" / "multiseed_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # === Figure 3: Ranking consistency heatmap ===
    # Only include seeds where all variants have data, to give fair rankings
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6), constrained_layout=True)
    fig.suptitle("Rank by Seed (1=best, 8=worst)",
                 fontsize=13, fontweight='bold')

    n_variants = len(VARIANTS)
    im = None

    for ax, dataset in zip(axes, DATASETS):
        found_seeds = []
        rank_matrix = []  # rows=variants, cols=seeds

        # Find which seeds have data for ALL variants
        for seed_idx, (seed, patterns) in enumerate(DATASETS[dataset]):
            seed_vals = {}
            for display_name, variant_key in VARIANTS:
                val_loss, _ = try_load_val_loss(patterns, variant_key)
                if val_loss is not None:
                    seed_vals[display_name] = val_loss
            if len(seed_vals) == n_variants:
                found_seeds.append(seed)
                # Rank (1=best=lowest loss)
                sorted_names = sorted(seed_vals.keys(), key=lambda n: seed_vals[n])
                ranks = {name: i + 1 for i, name in enumerate(sorted_names)}
                rank_matrix.append(ranks)

        if not found_seeds:
            # Fall back: include seeds where at least 6 variants have data
            for seed_idx, (seed, patterns) in enumerate(DATASETS[dataset]):
                seed_vals = {}
                for display_name, variant_key in VARIANTS:
                    val_loss, _ = try_load_val_loss(patterns, variant_key)
                    if val_loss is not None:
                        seed_vals[display_name] = val_loss
                if len(seed_vals) >= 6:
                    found_seeds.append(seed)
                    sorted_names = sorted(seed_vals.keys(), key=lambda n: seed_vals[n])
                    ranks = {name: i + 1 for i, name in enumerate(sorted_names)}
                    # Fill missing variants with NaN
                    for display_name, _ in VARIANTS:
                        if display_name not in ranks:
                            ranks[display_name] = np.nan
                    rank_matrix.append(ranks)

        if not found_seeds:
            ax.text(0.5, 0.5, "No seeds with sufficient data", ha='center', va='center')
            continue

        variant_names = [dn for dn, _ in VARIANTS]
        matrix = np.array([[rm[name] for rm in rank_matrix] for name in variant_names])

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=n_variants)
        ax.set_xticks(range(len(found_seeds)))
        ax.set_xticklabels([f"Seed {s}" for s in found_seeds])
        ax.set_yticks(range(len(variant_names)))
        ax.set_yticklabels(variant_names, fontsize=9)
        ax.set_title(dataset)

        # Add rank numbers in cells
        for i in range(len(variant_names)):
            for j in range(len(found_seeds)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha='center', va='center', fontsize=11, color='gray')
                else:
                    ax.text(j, i, str(int(val)),
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           color='white' if val in [1, n_variants] else 'black')

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, fraction=0.025,
                     pad=0.02, label='Rank')
    out_path = results_dir / "visualizations" / "multiseed_ranks.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
