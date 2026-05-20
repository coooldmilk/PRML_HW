"""
Mini reproduction of the original Transformer architecture from
"Attention Is All You Need" using PyTorch.

This file intentionally implements the core modules from scratch instead of
calling torch.nn.Transformer, so each paper component is visible:

1. Sinusoidal positional encoding
2. Scaled dot-product attention
3. Multi-head attention
4. Position-wise feed-forward network
5. Residual connection + LayerNorm
6. Encoder stack
7. Decoder stack with causal mask
8. Greedy autoregressive decoding

The default task is a tiny synthetic seq2seq task:
    input  = [x1, x2, x3, ...]
    target = reversed(input)

This is not meant to reproduce WMT BLEU numbers. It is a lightweight, runnable
architecture reproduction suitable for verifying that the Transformer works.

Run:
    python transformer_reproduction_pytorch.py

Optional:
    python transformer_reproduction_pytorch.py --epochs 20 --d-model 128 --num-layers 3
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# 1. Reproducibility
# -----------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2. Toy seq2seq dataset
# -----------------------------


class ReverseSequenceDataset(Dataset):
    """A small synthetic encoder-decoder task.

    Special token ids:
        PAD = 0
        BOS = 1
        EOS = 2
        Normal symbols are 3 ... vocab_size - 1

    Example:
        src        = [7, 4, 9, 5, EOS, PAD, PAD]
        decoder_in = [BOS, 5, 9, 4, 7, PAD, PAD]
        target     = [5, 9, 4, 7, EOS, PAD, PAD]
    """

    def __init__(
        self,
        num_samples: int,
        vocab_size: int,
        min_len: int,
        max_len: int,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        assert vocab_size > 4
        assert min_len >= 1
        assert max_len >= min_len

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.samples = [self._make_one() for _ in range(num_samples)]

    def _make_one(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = random.randint(self.min_len, self.max_len)
        seq = [random.randint(3, self.vocab_size - 1) for _ in range(length)]
        rev = list(reversed(seq))

        # Include EOS. All tensors have length max_len + 1.
        total_len = self.max_len + 1

        src = seq + [self.eos_id]
        dec_in = [self.bos_id] + rev
        target = rev + [self.eos_id]

        src += [self.pad_id] * (total_len - len(src))
        dec_in += [self.pad_id] * (total_len - len(dec_in))
        target += [self.pad_id] * (total_len - len(target))

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


# -----------------------------
# 3. Masks
# -----------------------------


def make_src_padding_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create encoder self-attention padding mask.

    Args:
        src: [batch, src_len]

    Returns:
        mask: [batch, 1, 1, src_len]
              1 means visible, 0 means masked.
    """
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create decoder causal + padding mask.

    Args:
        tgt: [batch, tgt_len]

    Returns:
        mask: [batch, 1, tgt_len, tgt_len]
              1 means visible, 0 means masked.
    """
    batch_size, tgt_len = tgt.shape

    padding_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

    return padding_mask & causal_mask


# -----------------------------
# 4. Positional encoding
# -----------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Original sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    It is added to token embeddings rather than concatenated.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_len, d_model], so it can broadcast over batch.
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# -----------------------------
# 5. Scaled dot-product attention
# -----------------------------


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            q: [batch, heads, query_len, d_k]
            k: [batch, heads, key_len, d_k]
            v: [batch, heads, key_len, d_v]
            mask: broadcastable to [batch, heads, query_len, key_len]

        Returns:
            output: [batch, heads, query_len, d_v]
            attn:   [batch, heads, query_len, key_len]
        """
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: [batch, heads, query_len, key_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        return output, attn


# -----------------------------
# 6. Multi-head attention
# -----------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention from the original Transformer.

    Each head has its own learned projections for Q, K and V.
    The outputs of all heads are concatenated and projected back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, d_model] -> [B, H, L, d_head]."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, d_head] -> [B, L, d_model]."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        x, attn = self.attention(q, k, v, mask)
        x = self._combine_heads(x)
        x = self.w_o(x)
        x = self.dropout(x)

        return x, attn


# -----------------------------
# 7. Position-wise feed-forward network
# -----------------------------


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1 + b1)W2 + b2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# 8. Encoder and decoder layers
# -----------------------------


class EncoderLayer(nn.Module):
    """One Transformer encoder layer.

    Paper formula style:
        x = LayerNorm(x + MultiHeadSelfAttention(x))
        x = LayerNorm(x + FFN(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderLayer(nn.Module):
    """One Transformer decoder layer.

    It contains three sublayers:
        1. Masked self-attention over previous target tokens
        2. Encoder-decoder attention over source memory
        3. Position-wise FFN
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)

        cross_attn_out, _ = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


# -----------------------------
# 9. Encoder, decoder, Transformer
# -----------------------------


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        max_len: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(
            vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, pad_id
        )
        self.decoder = Decoder(
            vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, pad_id
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Optional weight tying, inspired by the paper's embedding/softmax sharing.
        self.output_projection.weight = self.decoder.embedding.weight

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_padding_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)

        memory = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt_in, memory, tgt_mask, src_mask)
        logits = self.output_projection(decoder_out)
        return logits


# -----------------------------
# 10. Original learning-rate schedule
# -----------------------------


class NoamScheduler:
    """Learning-rate schedule from the Transformer paper.

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# -----------------------------
# 11. Training and evaluation
# -----------------------------


@dataclass
class TrainStats:
    loss: float
    token_accuracy: float


def compute_loss_and_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, float]:
    """Cross-entropy over non-PAD target tokens."""
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=pad_id,
    )

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = target != pad_id
        correct = ((pred == target) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / max(total, 1)

    return loss, acc


def run_one_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: NoamScheduler | None,
    device: torch.device,
    pad_id: int,
) -> TrainStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct_tokens = 0
    total_tokens = 0

    for src, tgt_in, target in dataloader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        target = target.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in)
            loss, _ = compute_loss_and_accuracy(logits, target, pad_id)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scheduler is not None:
                    scheduler.step()
                optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = target != pad_id
            total_correct_tokens += ((pred == target) & mask).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct_tokens / max(total_tokens, 1)
    return TrainStats(loss=avg_loss, token_accuracy=avg_acc)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_decode_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressively decode one or more source sequences."""
    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_decode_len):
        logits = model(src, generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if torch.all(next_token.squeeze(-1) == eos_id):
            break

    # Pad to a consistent length for display, if needed.
    if generated.size(1) < max_decode_len + 1:
        pad = torch.full(
            (batch_size, max_decode_len + 1 - generated.size(1)),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        generated = torch.cat([generated, pad], dim=1)

    return generated


def strip_special(tokens: list[int], pad_id: int, bos_id: int, eos_id: int) -> list[int]:
    result = []
    for t in tokens:
        if t in (pad_id, bos_id):
            continue
        if t == eos_id:
            break
        result.append(t)
    return result


def show_examples(
    model: Transformer,
    dataset: ReverseSequenceDataset,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    print("\nGreedy decoding examples:")
    indices = random.sample(range(len(dataset)), k=min(num_examples, len(dataset)))
    src_batch = torch.stack([dataset[i][0] for i in indices])
    decoded = greedy_decode(
        model,
        src_batch,
        max_decode_len=dataset.max_len + 1,
        bos_id=dataset.bos_id,
        eos_id=dataset.eos_id,
        pad_id=dataset.pad_id,
        device=device,
    ).cpu()

    for row, idx in enumerate(indices):
        src, _, target = dataset[idx]
        src_clean = strip_special(src.tolist(), dataset.pad_id, dataset.bos_id, dataset.eos_id)
        tgt_clean = strip_special(target.tolist(), dataset.pad_id, dataset.bos_id, dataset.eos_id)
        pred_clean = strip_special(decoded[row].tolist(), dataset.pad_id, dataset.bos_id, dataset.eos_id)
        print(f"  src:    {src_clean}")
        print(f"  target: {tgt_clean}")
        print(f"  pred:   {pred_clean}")
        print("-" * 50)


# -----------------------------
# 12. Main script
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--valid-samples", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.0, help="Initial lr is overwritten by Noam schedule.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pad_id = 0
    bos_id = 1
    eos_id = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ReverseSequenceDataset(
        num_samples=args.train_samples,
        vocab_size=args.vocab_size,
        min_len=args.min_len,
        max_len=args.max_len,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    valid_ds = ReverseSequenceDataset(
        num_samples=args.valid_samples,
        vocab_size=args.vocab_size,
        min_len=args.min_len,
        max_len=args.max_len,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len + 2,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Paper-style optimizer values: beta1=0.9, beta2=0.98, eps=1e-9.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_one_epoch(
            model, train_loader, optimizer, scheduler, device, pad_id
        )
        valid_stats = run_one_epoch(
            model, valid_loader, optimizer=None, scheduler=None, device=device, pad_id=pad_id
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss/token {train_stats.loss:.4f}, acc {train_stats.token_accuracy:.3f} | "
            f"valid loss/token {valid_stats.loss:.4f}, acc {valid_stats.token_accuracy:.3f}"
        )

    show_examples(model, valid_ds, device, num_examples=5)


if __name__ == "__main__":
    main()
