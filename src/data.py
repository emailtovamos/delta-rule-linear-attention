"""
Data loading for small-scale language modeling experiments.
Uses TinyStories or WikiText for quick iteration.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional


class TextDataset(Dataset):
    """Pre-tokenized chunks for language modeling."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences we can extract
        self.n_samples = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def load_data(
    dataset_name: str = "tinystories",
    tokenizer_name: str = "gpt2",
    seq_len: int = 512,
    max_train_tokens: int = 10_000_000,  # 10M tokens for quick experiments
    max_val_tokens: int = 500_000,
    batch_size: int = 8,
    num_workers: int = 0,  # 0 for MPS compatibility
) -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Load and tokenize a dataset.

    Returns (train_loader, val_loader, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset_name}...")
    if dataset_name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train[:5%]")
        ds_val = load_dataset("roneneldan/TinyStories", split="validation[:5%]")
        text_field = "text"
    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        ds_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        text_field = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_and_concat(dataset, max_tokens):
        all_ids = []
        total = 0
        for example in dataset:
            text = example[text_field]
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            total += len(ids)
            if total >= max_tokens:
                break
        return torch.tensor(all_ids[:max_tokens], dtype=torch.long)

    print(f"Tokenizing train (max {max_train_tokens:,} tokens)...")
    train_tokens = tokenize_and_concat(ds, max_train_tokens)
    print(f"  → {len(train_tokens):,} tokens")

    print(f"Tokenizing val (max {max_val_tokens:,} tokens)...")
    val_tokens = tokenize_and_concat(ds_val, max_val_tokens)
    print(f"  → {len(val_tokens):,} tokens")

    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    return train_loader, val_loader, tokenizer
