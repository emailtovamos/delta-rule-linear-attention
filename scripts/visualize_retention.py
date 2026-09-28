"""
Visualize channel-wise retention patterns in linear attention.

For each trained model, this script:
1. Feeds a long text sequence through the model
2. Captures the attention state S at each timestep
3. Plots heatmaps showing how information decays per channel
4. Compares gate distributions across attention types

Usage:
    uv run python scripts/visualize_retention.py \
        --checkpoint results/kda_baseline/model.pt --name kda
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import SmallGPT, ModelConfig


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = SmallGPT(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, config


def get_sample_text():
    """A diverse text sample for probing memory retention."""
    return (
        "The capital of France is Paris. It was founded over 2000 years ago. "
        "The Eiffel Tower is 330 meters tall and was built in 1889. "
        "Meanwhile, in a completely different topic, quantum computers use qubits "
        "instead of classical bits. A qubit can be in superposition of states. "
        "Going back to Paris, the city has a population of about 2.1 million people. "
        "The Seine river flows through the center of the city. "
        "Now let us discuss machine learning. Transformers use attention mechanisms "
        "to process sequences. The key innovation is self-attention, which allows "
        "each token to attend to every other token. However, this has quadratic "
        "complexity. Linear attention replaces softmax with feature maps. "
        "Remember that Paris is in France and the Eiffel Tower is 330 meters tall. "
        "Can the model recall these facts from earlier in the sequence? "
        "The answer depends on how well the memory mechanism retains information. "
        "Different decay rates for different channels could help: slow channels "
        "remember the capital of France while fast channels track the current topic."
    )


@torch.no_grad()
def visualize_model(checkpoint_path: str, name: str, device: torch.device):
    model, config = load_model(checkpoint_path, device)
    out_dir = Path("results") / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = get_sample_text()
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    T = tokens.shape[1]

    print(f"Sequence length: {T} tokens")
    print(f"Attention type: {config.attention_type}")

    # Skip standard attention (no recurrent state)
    if config.attention_type == 'standard':
        print("Standard attention has no recurrent state to visualize. Skipping.")
        return

    # Get state snapshots from first layer
    snapshot_every = max(1, T // 100)  # ~100 snapshots
    result = model.get_attention_states(tokens, layer_idx=0, snapshot_every=snapshot_every)

    if result is None:
        print("No state snapshots available.")
        return

    states = result['states'][0]  # Remove batch dim: (H, num_snaps, D, D)
    gate_type = result['gate_type']

    H = states.shape[0]
    n_snaps = states.shape[1]
    D = states.shape[2]

    print(f"Captured {n_snaps} snapshots across {H} heads, state size {D}x{D}")

    # ─── Plot 1: State Frobenius norm per channel over time ───
    fig, axes = plt.subplots(2, min(H, 4), figsize=(5 * min(H, 4), 10))
    if H == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f'{name} — State Energy per Channel over Sequence\n'
                 f'(attention={config.attention_type}, {config.d_model}d, {config.n_layers}L)',
                 fontsize=14)

    for h in range(min(H, 4)):
        # Row energy: ||S[d, :]||_2 for each row d at each timestep
        state_h = states[h]  # (num_snaps, D, D)
        row_norms = torch.norm(state_h, dim=-1)  # (num_snaps, D)

        ax = axes[0, h] if H > 1 else axes[0]
        im = ax.imshow(row_norms.T.numpy(), aspect='auto', cmap='viridis',
                        interpolation='nearest')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Channel (dimension)')
        ax.set_title(f'Head {h}: Row Energy ||S[d,:]||')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Plot the variance across channels (shows diversity of retention)
        ax2 = axes[1, h] if H > 1 else axes[1]
        channel_std = row_norms.std(dim=-1)  # (num_snaps,)
        channel_mean = row_norms.mean(dim=-1)
        x = np.arange(n_snaps) * snapshot_every

        ax2.fill_between(x, (channel_mean - channel_std).numpy(),
                         (channel_mean + channel_std).numpy(), alpha=0.3, color='steelblue')
        ax2.plot(x, channel_mean.numpy(), color='steelblue', label='mean ± std')
        ax2.set_xlabel('Sequence Position')
        ax2.set_ylabel('Channel Energy')
        ax2.set_title(f'Head {h}: Energy Distribution')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f'{name}_state_energy.png', dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / f'{name}_state_energy.png'}")

    # ─── Plot 2: Gate distribution ───
    if 'gates' in result:
        gates = result['gates'][0]  # (H, num_snaps, ...) depends on type

        fig, axes = plt.subplots(1, min(H, 4), figsize=(5 * min(H, 4), 4))
        if H == 1:
            axes = [axes]
        elif min(H, 4) == 1:
            axes = [axes]
        fig.suptitle(f'{name} — Gate Values ({gate_type})', fontsize=14)

        for h in range(min(H, 4)):
            ax = axes[h]
            gate_h = gates[h]  # (num_snaps,) or (num_snaps, D)

            if gate_type == 'scalar':
                ax.plot(np.arange(n_snaps) * snapshot_every, gate_h.numpy())
                ax.set_ylabel('Gate value')
                ax.set_title(f'Head {h}: Scalar Gate')
                ax.set_ylim(0, 1)

            elif gate_type in ('channel_wise', 'hierarchical'):
                # Show histogram of gate values across all timesteps
                flat_gates = gate_h.reshape(-1).numpy()
                ax.hist(flat_gates, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
                ax.set_xlabel('Gate value')
                ax.set_ylabel('Count')
                ax.set_title(f'Head {h}: Gate Distribution (D={gate_h.shape[-1]})')
                ax.axvline(x=flat_gates.mean(), color='red', linestyle='--',
                          label=f'mean={flat_gates.mean():.3f}')
                ax.legend()

            ax.set_xlabel('Sequence Position' if gate_type == 'scalar' else 'Gate value')

        plt.tight_layout()
        plt.savefig(out_dir / f'{name}_gates.png', dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / f'{name}_gates.png'}")

    # ─── Plot 3: Channel-wise gate heatmap (KDA / hierarchical only) ───
    if gate_type in ('channel_wise', 'hierarchical') and 'gates' in result:
        gates = result['gates'][0]  # (H, num_snaps, D)
        fig, axes = plt.subplots(1, min(H, 4), figsize=(5 * min(H, 4), 5))
        if H == 1 or min(H, 4) == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        fig.suptitle(f'{name} — Per-Channel Gate Values Over Sequence', fontsize=14)

        for h in range(min(H, 4)):
            ax = axes[h]
            gate_h = gates[h]  # (num_snaps, D)
            im = ax.imshow(gate_h.T.numpy(), aspect='auto', cmap='RdYlGn',
                          vmin=0, vmax=1, interpolation='nearest')
            ax.set_xlabel('Sequence Position')
            ax.set_ylabel('Channel')
            ax.set_title(f'Head {h}')
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(out_dir / f'{name}_channel_gates.png', dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / f'{name}_channel_gates.png'}")

    # ─── Plot 4: Routing weights (freq_routed only) ───
    if gate_type == 'frequency_routed' and 'routing' in result:
        routing = result['routing'][0]  # (H, num_snaps, G)
        decay_rates = result['decay_rates']

        fig, axes = plt.subplots(1, min(H, 4), figsize=(5 * min(H, 4), 4))
        if H == 1 or min(H, 4) == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        fig.suptitle(f'{name} — Token Routing to Decay Groups', fontsize=14)

        for h in range(min(H, 4)):
            ax = axes[h]
            route_h = routing[h].numpy()  # (num_snaps, G)
            x = np.arange(n_snaps) * snapshot_every
            G = route_h.shape[-1]

            for g in range(G):
                ax.plot(x, route_h[:, g],
                       label=f'γ={decay_rates[g]:.2f}', alpha=0.8)

            ax.set_xlabel('Sequence Position')
            ax.set_ylabel('Routing Weight')
            ax.set_title(f'Head {h}')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(out_dir / f'{name}_routing.png', dpi=150)
        plt.close()
        print(f"  Saved: {out_dir / f'{name}_routing.png'}")

    # ─── Plot 5: Information Retention Test ───
    # Measure how well the state at the END still encodes info from the BEGINNING
    state_first = states[:, 0, :, :]     # (H, D, D) — state early in sequence
    state_last = states[:, -1, :, :]     # (H, D, D) — state at end

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f'{name} — Information Retention: Similarity of Early vs Late State', fontsize=13)

    # Cosine similarity between flattened state at each timestep vs initial state
    similarities = []
    for t in range(n_snaps):
        s_t = states[:, t, :, :].reshape(-1)
        s_0 = states[:, 0, :, :].reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(s_t.unsqueeze(0),
                                                         s_0.unsqueeze(0)).item()
        similarities.append(cos_sim)

    x = np.arange(n_snaps) * snapshot_every
    ax.plot(x, similarities, color='steelblue', linewidth=2)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Cosine Similarity with Initial State')
    ax.set_title(f'How much early information is retained?\n'
                 f'(1.0 = perfect retention, 0.0 = completely overwritten)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.savefig(out_dir / f'{name}_retention.png', dpi=150)
    plt.close()
    print(f"  Saved: {out_dir / f'{name}_retention.png'}")

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    visualize_model(args.checkpoint, args.name, device)
