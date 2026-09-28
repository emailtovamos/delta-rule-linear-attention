"""
Plot length generalization results: val loss and perplexity vs. sequence length.

Usage:
    uv run python scripts/plot_length_gen.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

VARIANTS = [
    ("StaticChannelDelta", "static_channel_delta"),
    ("KDA", "kda"),
    ("DeltaNet", "deltanet"),
    ("Standard", "standard"),
    ("GLA", "gla"),
    ("StaticChannel", "static_channel"),
]

COLORS = {
    "StaticChannelDelta": "#d62728",
    "KDA": "#1f77b4",
    "DeltaNet": "#ff7f0e",
    "Standard": "#7f7f7f",
    "GLA": "#2ca02c",
    "StaticChannel": "#9467bd",
}

MARKERS = {
    "StaticChannelDelta": "o",
    "KDA": "s",
    "DeltaNet": "^",
    "Standard": "x",
    "GLA": "D",
    "StaticChannel": "v",
}

results_dir = Path("results") / "length_gen"
vis_dir = Path("results") / "visualizations"


def load_results():
    data = {}
    for display_name, variant_key in VARIANTS:
        path = results_dir / f"{variant_key}.json"
        if not path.exists():
            print(f"  {display_name}: not found")
            continue
        with open(path) as f:
            data[display_name] = json.load(f)
        print(f"  {display_name}: {len(data[display_name]['eval_results'])} lengths")
    return data


def main():
    print("Loading length generalization results...")
    data = load_results()

    if not data:
        print("No results found. Run eval_length_gen.py first.")
        return

    vis_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Val loss vs seq_len
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Length Generalization: Trained at 2048, Evaluated at Longer Lengths",
                 fontsize=13, fontweight='bold')

    for display_name, _ in VARIANTS:
        if display_name not in data:
            continue
        r = data[display_name]
        lengths = [e['seq_len'] for e in r['eval_results']]
        losses = [e['val_loss'] for e in r['eval_results']]
        ppls = [e['perplexity'] for e in r['eval_results']]

        ax1.plot(lengths, losses,
                 color=COLORS[display_name],
                 marker=MARKERS[display_name],
                 linewidth=2, markersize=8,
                 label=display_name)
        ax2.plot(lengths, ppls,
                 color=COLORS[display_name],
                 marker=MARKERS[display_name],
                 linewidth=2, markersize=8,
                 label=display_name)

    ax1.set_xlabel("Evaluation Sequence Length")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Val Loss vs. Sequence Length")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=2048, color='gray', linestyle=':', alpha=0.5, label='Train length')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2048, 4096, 8192])
    ax1.set_xticklabels(['2K', '4K', '8K'])

    ax2.set_xlabel("Evaluation Sequence Length")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity vs. Sequence Length")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=2048, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2048, 4096, 8192])
    ax2.set_xticklabels(['2K', '4K', '8K'])

    plt.tight_layout()
    out = vis_dir / "length_generalization.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()

    # Figure 2: Degradation (relative to train length)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Length Generalization Degradation (Δ Loss from Training Length)",
                 fontsize=13, fontweight='bold')

    for display_name, _ in VARIANTS:
        if display_name not in data:
            continue
        r = data[display_name]
        results = r['eval_results']
        if not results:
            continue
        base_loss = results[0]['val_loss']  # loss at training length
        lengths = [e['seq_len'] for e in results]
        deltas = [e['val_loss'] - base_loss for e in results]

        ax.plot(lengths, deltas,
                color=COLORS[display_name],
                marker=MARKERS[display_name],
                linewidth=2, markersize=8,
                label=f"{display_name} (base={base_loss:.3f})")

    ax.set_xlabel("Evaluation Sequence Length")
    ax.set_ylabel("Δ Val Loss (relative to 2K)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=2048, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks([2048, 4096, 8192])
    ax.set_xticklabels(['2K', '4K', '8K'])

    plt.tight_layout()
    out = vis_dir / "length_gen_degradation.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # Print summary table
    print(f"\n{'='*70}")
    print("LENGTH GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<22} ", end="")
    # Get all lengths from any variant
    all_lengths = set()
    for d in data.values():
        for e in d['eval_results']:
            all_lengths.add(e['seq_len'])
    for l in sorted(all_lengths):
        print(f"{'%dK' % (l//1024):>8}", end="")
    print(f"  {'Δ(max-min)':>10}")
    print("-" * 70)

    for display_name, _ in VARIANTS:
        if display_name not in data:
            continue
        r = data[display_name]
        losses = {e['seq_len']: e['val_loss'] for e in r['eval_results']}
        print(f"{display_name:<22} ", end="")
        vals = []
        for l in sorted(all_lengths):
            if l in losses:
                print(f"{losses[l]:>8.4f}", end="")
                vals.append(losses[l])
            else:
                print(f"{'--':>8}", end="")
        if len(vals) > 1:
            print(f"  {max(vals)-min(vals):>+10.4f}")
        else:
            print()


if __name__ == "__main__":
    main()
