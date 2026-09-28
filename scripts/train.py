"""
Train a small GPT with different attention mechanisms.

Usage:
    uv run python scripts/train.py --attention gla --name gla_baseline
    uv run python scripts/train.py --attention kda --name kda_baseline --size tiny
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SmallGPT, ModelConfig
from src.data import load_data


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Create output directory
    out_dir = Path("results") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model config
    config_fn = {'tiny': ModelConfig.tiny, 'small': ModelConfig.small, 'medium': ModelConfig.medium}
    attn_kwargs = {}
    if args.decay_init != "exponential":
        attn_kwargs['decay_init'] = args.decay_init
    config = config_fn[args.size](attention_type=args.attention, attention_kwargs=attn_kwargs)
    config.max_seq_len = args.seq_len
    config.vocab_size = 50257  # GPT-2 vocab
    config.no_pos_emb = args.no_pos_emb

    # Data
    train_loader, val_loader, tokenizer = load_data(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        max_train_tokens=args.max_train_tokens,
        batch_size=args.batch_size,
    )

    # Model
    model = SmallGPT(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup then cosine decay
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    log = {
        'config': {
            'attention': args.attention,
            'size': args.size,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'seq_len': args.seq_len,
            'params_M': model.num_params / 1e6,
        },
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'step_times': [],
    }

    global_step = 0
    best_val_loss = float('inf')

    print(f"\nTraining {args.name} for {args.epochs} epochs...")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Warmup: {warmup_steps} steps")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            t0 = time.time()
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            dt = time.time() - t0
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log every N steps
            if global_step % args.log_every == 0:
                log['train_loss'].append((global_step, loss.item()))
                log['lr'].append((global_step, optimizer.param_groups[0]['lr']))
                log['step_times'].append((global_step, dt))

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                'ms/step': f"{dt*1000:.0f}",
            })

        # End-of-epoch eval
        val_loss = evaluate(model, val_loader, device)
        avg_train = epoch_loss / max(epoch_steps, 1)
        log['val_loss'].append((global_step, val_loss))

        print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f} val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'config': config,
                'step': global_step,
                'val_loss': val_loss,
            }, out_dir / "model.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    # Save training log
    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Save final model too
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'step': global_step,
        'val_loss': val_loss,
    }, out_dir / "model_final.pt")

    print(f"\nDone! Results saved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train small GPT with different attention")
    parser.add_argument("--attention", type=str, default="gla",
                        choices=["standard", "gla", "deltanet", "kda",
                                 "static_channel", "static_channel_delta",
                                 "scalar_static", "scalar_static_delta",
                                 "hierarchical", "freq_routed"])
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--size", type=str, default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--dataset", type=str, default="tinystories")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_train_tokens", type=int, default=10_000_000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pos_emb", action="store_true",
                        help="Disable positional embeddings (NoPE mode for length generalization)")
    parser.add_argument("--decay_init", type=str, default="exponential",
                        choices=["exponential", "uniform", "random"],
                        help="Decay initialization for static channel variants")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    train(args)
