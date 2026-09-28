"""
Systematic ablation study over different decay mechanisms.

This script trains ALL attention variants with the SAME hyperparameters
and compares their performance. It's designed to answer:

  "Does channel-wise gating (KDA) actually help compared to
   scalar gating (GLA)? Does hierarchical decay help? Does routing?"

Usage:
    uv run python scripts/ablation.py
    uv run python scripts/ablation.py --size small --epochs 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train


def run_ablation(args):
    results = {}
    attention_types = ['gla', 'deltanet', 'kda', 'static_channel', 'static_channel_delta',
                       'hierarchical', 'freq_routed']

    # Also run standard attention as ceiling reference
    if args.include_standard:
        attention_types = ['standard'] + attention_types

    print("=" * 70)
    print("  ABLATION STUDY: Channel-Wise Forgetting Mechanisms")
    print("=" * 70)
    print(f"  Model size: {args.size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Variants: {attention_types}")
    print(f"  Seq len: {args.seq_len}")
    print("=" * 70)
    print()

    for attn_type in attention_types:
        name = f"ablation_{attn_type}_{args.size}"
        print(f"\n{'='*50}")
        print(f"  Training: {attn_type}")
        print(f"{'='*50}")

        # Create args namespace for train function
        train_args = argparse.Namespace(
            attention=attn_type,
            name=name,
            size=args.size,
            dataset=args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=0.1,
            max_train_tokens=args.max_train_tokens,
            log_every=50,
            seed=42,
        )

        t0 = time.time()
        try:
            val_loss = train(train_args)
            train_time = time.time() - t0

            results[attn_type] = {
                'val_loss': val_loss,
                'train_time_s': train_time,
                'status': 'success',
            }
            print(f"\n  ✓ {attn_type}: val_loss={val_loss:.4f} ({train_time:.0f}s)")

        except Exception as e:
            results[attn_type] = {
                'val_loss': float('inf'),
                'train_time_s': time.time() - t0,
                'status': f'failed: {str(e)}',
            }
            print(f"\n  ✗ {attn_type}: FAILED — {e}")

    # Save comparison
    out_dir = Path("results") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n")
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Attention':<20} {'Val Loss':>10} {'Time (s)':>10} {'Status':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_loss'])
    for attn_type, res in sorted_results:
        status = '✓' if res['status'] == 'success' else '✗'
        loss_str = f"{res['val_loss']:.4f}" if res['val_loss'] < float('inf') else "FAILED"
        print(f"  {attn_type:<20} {loss_str:>10} {res['train_time_s']:>10.0f} {status:>10}")

    print(f"\n  Results saved to {out_dir}/comparison.json")

    # Generate comparison plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        successful = {k: v for k, v in sorted_results if v['status'] == 'success'}
        names = list(successful.keys())
        losses = [v['val_loss'] for v in successful.values()]
        times = [v['train_time_s'] for v in successful.values()]

        # Color code: standard=gray, scalar-gate=blue, channel-wise=green, experimental=orange
        colors = []
        for n in names:
            if n == 'standard':
                colors.append('#888888')
            elif n in ('gla', 'deltanet'):
                colors.append('#4C72B0')
            elif n == 'kda':
                colors.append('#55A868')
            else:
                colors.append('#DD8452')

        # Bar chart: val loss
        ax = axes[0]
        bars = ax.bar(names, losses, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Validation Loss (lower is better)')
        ax.set_title('Perplexity Comparison')
        ax.set_xticklabels(names, rotation=30, ha='right')
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom', fontsize=9)

        # Bar chart: training time
        ax = axes[1]
        bars = ax.bar(names, times, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Efficiency')
        ax.set_xticklabels(names, rotation=30, ha='right')

        plt.suptitle(f'Ablation: Channel-Wise Forgetting Mechanisms ({args.size} model)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "comparison.png", dpi=150)
        plt.close()
        print(f"  Plot saved to {out_dir}/comparison.png")

        # Also plot training curves overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for attn_type in names:
            log_path = Path("results") / f"ablation_{attn_type}_{args.size}" / "log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                steps = [x[0] for x in log['train_loss']]
                losses = [x[1] for x in log['train_loss']]
                # Smooth with moving average
                window = max(1, len(losses) // 50)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=attn_type, alpha=0.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"  Training curves saved to {out_dir}/training_curves.png")

    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_train_tokens", type=int, default=10_000_000)
    parser.add_argument("--include_standard", action="store_true",
                        help="Also train standard softmax attention as baseline")
    args = parser.parse_args()

    torch.manual_seed(42)
    run_ablation(args)
