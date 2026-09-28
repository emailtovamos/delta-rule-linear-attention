"""
Plot training curves (val loss vs. training step) for all variants across datasets.

Generates two figures:
1. Val loss curves for TinyStories (all 6 variants)
2. Val loss curves for WikiText-103 (all 6 variants)
3. Combined side-by-side comparison

Usage:
    uv run python scripts/plot_training_curves.py
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Result directories mapping: (display_name, dir_name)
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
    "StaticChannelDelta": "#d62728",  # red (ours)
    "KDA": "#1f77b4",                 # blue
    "ScalarStaticDelta": "#e377c2",   # pink
    "DeltaNet": "#ff7f0e",            # orange
    "Standard": "#7f7f7f",            # gray
    "GLA": "#2ca02c",                 # green
    "ScalarStatic": "#8c564b",        # brown
    "StaticChannel": "#9467bd",       # purple
}

LINESTYLES = {
    "StaticChannelDelta": "-",
    "KDA": "-",
    "ScalarStaticDelta": "-",
    "DeltaNet": "-",
    "Standard": ":",
    "GLA": "-",
    "ScalarStatic": ":",
    "StaticChannel": ":",
}

DATASETS = [
    ("TinyStories", "{variant}_baseline"),
    ("WikiText-103", "wiki_{variant}"),
]

# Some variants use different naming conventions for their result directories
DIR_OVERRIDES = {
    "TinyStories": {
        "scalar_static": "scalar_static_seed42",
        "scalar_static_delta": "scalar_static_delta_seed42",
    },
    "WikiText-103": {
        "scalar_static": "wiki_scalar_static_seed42",
        "scalar_static_delta": "wiki_scalar_static_delta_seed42",
    },
}

results_dir = Path("results")


def load_log(dir_name):
    log_path = results_dir / dir_name / "log.json"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        return json.load(f)


def resolve_dir(dataset_name, dir_template, variant_key):
    """Resolve directory name, using overrides for variants with non-standard naming."""
    overrides = DIR_OVERRIDES.get(dataset_name, {})
    if variant_key in overrides:
        return overrides[variant_key]
    return dir_template.format(variant=variant_key)


def plot_single_dataset(ax, dataset_name, dir_template):
    """Plot val loss and train loss curves for one dataset."""
    for display_name, variant_key in VARIANTS:
        dir_name = resolve_dir(dataset_name, dir_template, variant_key)
        log = load_log(dir_name)
        if log is None:
            print(f"  Skipping {display_name} ({dir_name}): no log found")
            continue

        # Plot train loss (smoothed)
        train_steps = [s for s, _ in log["train_loss"]]
        train_losses = [l for _, l in log["train_loss"]]

        # Simple moving average for train loss (window=20 = 1000 steps)
        window = 20
        if len(train_losses) > window:
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window + 1)
                smoothed.append(sum(train_losses[start:i+1]) / (i - start + 1))
            train_losses = smoothed

        ax.plot(train_steps, train_losses,
                color=COLORS[display_name],
                linestyle=LINESTYLES[display_name],
                alpha=0.3, linewidth=0.8)

        # Plot val loss (bold markers)
        val_steps = [s for s, _ in log["val_loss"]]
        val_losses = [l for _, l in log["val_loss"]]
        ax.plot(val_steps, val_losses,
                color=COLORS[display_name],
                linestyle=LINESTYLES[display_name],
                linewidth=2.0, marker='o', markersize=5,
                label=f"{display_name} ({val_losses[-1]:.4f})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(dataset_name)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves: Val Loss (bold) + Train Loss (faint) vs. Step",
                 fontsize=13, fontweight='bold')

    for ax, (dataset_name, dir_template) in zip(axes, DATASETS):
        print(f"Plotting {dataset_name}...")
        plot_single_dataset(ax, dataset_name, dir_template)

    plt.tight_layout()
    out_path = results_dir / "visualizations" / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # Figure 2: Zoomed-in val loss only (cleaner for the paper)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Validation Loss vs. Training Step",
                 fontsize=14, fontweight='bold')

    for ax, (dataset_name, dir_template) in zip(axes, DATASETS):
        for display_name, variant_key in VARIANTS:
            dir_name = resolve_dir(dataset_name, dir_template, variant_key)
            log = load_log(dir_name)
            if log is None:
                continue

            val_steps = [s for s, _ in log["val_loss"]]
            val_losses = [l for _, l in log["val_loss"]]
            ax.plot(val_steps, val_losses,
                    color=COLORS[display_name],
                    linestyle=LINESTYLES[display_name],
                    linewidth=2.8, marker='o', markersize=7,
                    markeredgecolor='black', markeredgewidth=0.3,
                    label=f"{display_name} ({val_losses[-1]:.4f})")

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.set_title(dataset_name, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9, framealpha=0.92)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "visualizations" / "val_loss_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # Figure 3: Final loss bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Final Validation Loss by Variant",
                 fontsize=13, fontweight='bold')

    for ax, (dataset_name, dir_template) in zip(axes, DATASETS):
        names = []
        losses = []
        colors = []
        for display_name, variant_key in VARIANTS:
            dir_name = resolve_dir(dataset_name, dir_template, variant_key)
            log = load_log(dir_name)
            if log is None:
                continue
            val_losses = [l for _, l in log["val_loss"]]
            names.append(display_name)
            losses.append(val_losses[-1])
            colors.append(COLORS[display_name])

        bars = ax.barh(names[::-1], losses[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Validation Loss")
        ax.set_title(dataset_name)

        # Add value labels
        for bar, loss in zip(bars, losses[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{loss:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    out_path = results_dir / "visualizations" / "final_loss_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
