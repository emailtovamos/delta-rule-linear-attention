"""
Evaluate a trained model at multiple sequence lengths for length generalization.

Loads a checkpoint trained at seq_len=T and evaluates at T, 2T, 4T, etc.
Requires models trained with --no_pos_emb (NoPE mode).

Usage:
    uv run python scripts/eval_length_gen.py \
        --checkpoint results/lengthgen_static_channel_delta/model.pt \
        --name static_channel_delta \
        --eval_lengths 2048 4096 8192

    # Evaluate all variants:
    for v in standard gla deltanet kda static_channel static_channel_delta; do
        uv run python scripts/eval_length_gen.py \
            --checkpoint results/lengthgen_${v}/model.pt \
            --name $v \
            --eval_lengths 2048 4096 8192
    done
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SmallGPT, ModelConfig
from src.data import TextDataset, load_data


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_at_length(model, val_tokens, seq_len, device, max_batches=50):
    """Evaluate model on validation tokens chunked to a specific seq_len."""
    model.eval()
    dataset = TextDataset(val_tokens, seq_len)
    # batch_size=1 for long sequences to avoid OOM
    bs = 1 if seq_len > 4096 else 2 if seq_len > 2048 else 4
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)

    total_loss = 0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        # Weight by number of tokens in batch
        n_tokens = x.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate length generalization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset to evaluate on (wikitext recommended for longer texts)")
    parser.add_argument("--eval_lengths", type=int, nargs="+",
                        default=[2048, 4096, 8192])
    parser.add_argument("--max_val_tokens", type=int, default=2_000_000,
                        help="Max validation tokens (need more for longer seqs)")
    parser.add_argument("--max_batches", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt['config']

    # Verify NoPE mode
    if not getattr(config, 'no_pos_emb', False):
        print("WARNING: This checkpoint was trained WITH positional embeddings.")
        print("Length generalization beyond training seq_len will fail.")
        print("Re-train with --no_pos_emb for proper length gen evaluation.")
        print()

    print(f"Loading model: {config.attention_type} (d={config.d_model}, "
          f"L={config.n_layers}, H={config.n_heads})")
    print(f"Training seq_len: {config.max_seq_len}")
    print(f"Eval lengths: {args.eval_lengths}")
    print()

    # Load model
    model = SmallGPT(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load validation tokens (raw, not chunked — we'll chunk per eval length)
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.dataset} validation set...")
    if args.dataset == "wikitext":
        ds_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    elif args.dataset == "tinystories":
        ds_val = load_dataset("roneneldan/TinyStories", split="validation[:5%]")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Tokenize
    all_ids = []
    for example in ds_val:
        text = example["text"]
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= args.max_val_tokens:
            break
    val_tokens = torch.tensor(all_ids[:args.max_val_tokens], dtype=torch.long)
    print(f"Validation tokens: {len(val_tokens):,}")
    print()

    # Evaluate at each length
    results = {
        'name': args.name,
        'attention_type': config.attention_type,
        'train_seq_len': config.max_seq_len,
        'no_pos_emb': getattr(config, 'no_pos_emb', False),
        'dataset': args.dataset,
        'eval_results': [],
    }

    for seq_len in args.eval_lengths:
        n_samples = (len(val_tokens) - 1) // seq_len
        if n_samples < 5:
            print(f"  seq_len={seq_len}: SKIP (only {n_samples} samples, need >= 5)")
            continue

        print(f"  Evaluating seq_len={seq_len} ({n_samples} samples)...", end=" ", flush=True)
        loss = evaluate_at_length(model, val_tokens, seq_len, device,
                                  max_batches=args.max_batches)
        ppl = math.exp(loss)
        print(f"loss={loss:.4f}  ppl={ppl:.2f}")

        results['eval_results'].append({
            'seq_len': seq_len,
            'val_loss': round(loss, 4),
            'perplexity': round(ppl, 2),
            'n_samples': n_samples,
        })

    # Save results
    out_dir = Path("results") / "length_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Length Generalization: {args.name} ({config.attention_type})")
    print(f"{'='*60}")
    print(f"{'Seq Len':>8} {'Val Loss':>10} {'Perplexity':>12} {'Δ from train':>14}")
    print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*14}")
    train_loss = results['eval_results'][0]['val_loss'] if results['eval_results'] else 0
    for r in results['eval_results']:
        delta = r['val_loss'] - train_loss
        marker = "  (train len)" if r['seq_len'] == config.max_seq_len else ""
        print(f"{r['seq_len']:>8} {r['val_loss']:>10.4f} {r['perplexity']:>12.2f} {delta:>+14.4f}{marker}")


if __name__ == "__main__":
    main()
