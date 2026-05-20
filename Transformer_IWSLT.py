"""
Paper-style Transformer reproduction for neural machine translation.

This script implements the main architecture and training recipe from
"Attention Is All You Need" without calling torch.nn.Transformer.

It is designed for a real parallel translation dataset loaded from Hugging Face
Datasets, such as:

1. Lightweight recommended run:
   IWSLT 2017 English-German

2. Closer-to-paper run:
   WMT14 English-German

Install dependencies:
    pip install torch datasets tokenizers tqdm sacrebleu

Quick IWSLT run:
    python transformer_mt_reproduction.py \
      --dataset-name IWSLT/iwslt2017 \
      --dataset-config iwslt2017-en-de \
      --src-lang en --tgt-lang de \
      --train-limit 50000 --valid-limit 2000 \
      --vocab-size 16000 \
      --d-model 256 --num-layers 4 --num-heads 4 --d-ff 1024 \
      --epochs 10 --batch-size 64 --beam-size 4

Closer-to-paper WMT14 base-style run, if you have enough GPU memory/time:
    python transformer_mt_reproduction.py \
      --dataset-name wmt/wmt14 \
      --dataset-config de-en \
      --src-lang en --tgt-lang de \
      --vocab-size 37000 \
      --d-model 512 --num-layers 6 --num-heads 8 --d-ff 2048 \
      --dropout 0.1 --label-smoothing 0.1 --warmup-steps 4000 \
      --epochs 10 --batch-size 32 --beam-size 4 --length-penalty-alpha 0.6

Important:
- This is much closer to the paper framework than the toy reverse-sequence script.
- Full WMT14 reproduction of the paper's BLEU scores requires large-scale data,
  careful preprocessing, checkpoint averaging, long training, and substantial GPU
  compute. This script gives you a faithful research/teaching implementation that
  can actually be run and extended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import DatasetDict, load_dataset
except ImportError as exc:
    raise ImportError("Please install datasets: pip install datasets") from exc

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
except ImportError as exc:
    raise ImportError("Please install tokenizers: pip install tokenizers") from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


# -----------------------------
# 0. Constants and reproducibility
# -----------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


# -----------------------------
# 1. Dataset and tokenizer
# -----------------------------


def get_translation_pair(example: Dict[str, Any], src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    """Read {'translation': {'en': ..., 'de': ...}} examples."""
    translation = example.get("translation", None)
    if translation is None:
        raise ValueError("Expected each example to contain a 'translation' field.")

    # Some older dataset scripts may return JSON strings.
    if isinstance(translation, str):
        translation = json.loads(translation)

    src = translation[src_lang]
    tgt = translation[tgt_lang]
    return src, tgt


def maybe_select_limit(split, limit: Optional[int]):
    if limit is None or limit <= 0:
        return split
    return split.select(range(min(limit, len(split))))


def load_translation_dataset(args: argparse.Namespace) -> DatasetDict:
    """Load a real MT dataset from Hugging Face Datasets."""
    print(f"Loading dataset: {args.dataset_name}, config: {args.dataset_config}")
    kwargs = {}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        ds = load_dataset(args.dataset_name, **kwargs)

    if args.train_split not in ds:
        raise ValueError(f"Train split '{args.train_split}' not found. Available: {list(ds.keys())}")
    if args.valid_split not in ds:
        raise ValueError(f"Validation split '{args.valid_split}' not found. Available: {list(ds.keys())}")

    ds[args.train_split] = maybe_select_limit(ds[args.train_split], args.train_limit)
    ds[args.valid_split] = maybe_select_limit(ds[args.valid_split], args.valid_limit)
    return ds


def iter_texts_for_tokenizer(
    dataset_split,
    src_lang: str,
    tgt_lang: str,
    max_samples: int,
) -> Iterable[str]:
    count = 0
    for example in dataset_split:
        src, tgt = get_translation_pair(example, src_lang, tgt_lang)
        yield src
        yield tgt
        count += 1
        if max_samples > 0 and count >= max_samples:
            break


def train_or_load_tokenizer(
    train_split,
    args: argparse.Namespace,
) -> Tokenizer:
    """Train a shared BPE tokenizer, like the paper's shared source-target vocabulary idea.

    The paper used BPE with a shared source-target vocabulary for EN-DE. This script
    trains one shared BPE tokenizer from both source and target text.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / args.tokenizer_file

    if tokenizer_path.exists() and not args.retrain_tokenizer:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print("Training shared BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            iter_texts_for_tokenizer(
                train_split,
                args.src_lang,
                args.tgt_lang,
                args.tokenizer_train_samples,
            ),
            trainer=trainer,
        )
        tokenizer.save(str(tokenizer_path))
        print(f"Saved tokenizer to {tokenizer_path}")

    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"Tokenizer is missing special token {token}")
    return tokenizer


class TranslationDataset(Dataset):
    """A map-style dataset that stores raw pairs and tokenizes in __getitem__."""

    def __init__(
        self,
        hf_split,
        tokenizer: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)

    def __len__(self) -> int:
        return len(self.hf_split)

    def _encode_src(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        ids = ids[: self.max_src_len - 1]
        return ids + [self.eos_id]

    def _encode_tgt(self, text: str) -> Tuple[List[int], List[int]]:
        ids = self.tokenizer.encode(text).ids
        ids = ids[: self.max_tgt_len - 1]
        decoder_in = [self.bos_id] + ids
        target_out = ids + [self.eos_id]
        return decoder_in, target_out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_text, tgt_text = get_translation_pair(self.hf_split[idx], self.src_lang, self.tgt_lang)
        src_ids = self._encode_src(src_text)
        tgt_in_ids, tgt_out_ids = self._encode_tgt(tgt_text)
        return {
            "src_ids": src_ids,
            "tgt_in_ids": tgt_in_ids,
            "tgt_out_ids": tgt_out_ids,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def pad_sequences(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(x) for x in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def make_collate_fn(pad_id: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        src = pad_sequences([x["src_ids"] for x in batch], pad_id)
        tgt_in = pad_sequences([x["tgt_in_ids"] for x in batch], pad_id)
        tgt_out = pad_sequences([x["tgt_out_ids"] for x in batch], pad_id)
        return {
            "src": src,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "src_text": [x["src_text"] for x in batch],
            "tgt_text": [x["tgt_text"] for x in batch],
        }

    return collate


# -----------------------------
# 2. Masks
# -----------------------------


def make_src_padding_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B, S] -> [B, 1, 1, S]. 1 means visible."""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Causal mask + padding mask. [B, T] -> [B, 1, T, T]."""
    batch_size, tgt_len = tgt.shape
    padding_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)
    return padding_mask & causal


# -----------------------------
# 3. Transformer modules
# -----------------------------


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))
        x, attn = self.attention(q, k, v, mask)
        x = self.combine_heads(x)
        x = self.dropout(self.w_o(x))
        return x, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.output_dropout(x)


class EncoderLayer(nn.Module):
    """Post-LN encoder layer: LayerNorm(x + Sublayer(x)), as in the original paper."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderLayer(nn.Module):
    """Post-LN decoder layer with masked self-attention and encoder-decoder attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
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
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)

        cross_attn_out, _ = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class TransformerMT(nn.Module):
    """Original-style encoder-decoder Transformer for machine translation."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 256,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.token_embedding(tokens) * math.sqrt(self.d_model))

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.embed(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.embed(tgt_in)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_padding_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt_in, memory, tgt_mask, src_mask)
        return self.output_projection(decoder_out)


# -----------------------------
# 4. Loss, optimizer schedule, metrics
# -----------------------------


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing, ignoring PAD positions."""

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], target: [B, T]
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        mask = target != self.pad_id

        logits = logits[mask]
        target = target[mask]

        if target.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)

        if self.smoothing <= 0:
            return F.nll_loss(log_probs, target, reduction="mean")

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_id] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=-1).mean()
        return loss


class NoamScheduler:
    """Original Transformer learning-rate schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_num = int(state.get("step_num", 0))


@dataclass
class Metrics:
    loss: float
    accuracy: float
    tokens: int


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_id: int) -> Tuple[int, int]:
    pred = logits.argmax(dim=-1)
    mask = target != pad_id
    correct = ((pred == target) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


# -----------------------------
# 5. Decoding and BLEU
# -----------------------------


def ids_to_text(tokenizer: Tokenizer, ids: List[int], special_ids: set[int]) -> str:
    cleaned = []
    for idx in ids:
        if idx in special_ids:
            continue
        cleaned.append(idx)
    return tokenizer.decode(cleaned)


@torch.no_grad()
def greedy_decode(
    model: TransformerMT,
    src: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    model.eval()
    device = src.device
    batch_size = src.size(0)
    src_mask = make_src_padding_mask(src, pad_id)
    memory = model.encode(src, src_mask)

    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_id)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.output_projection(out[:, -1, :])
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_id
        if torch.all(finished):
            break
    return ys


@torch.no_grad()
def beam_search_one(
    model: TransformerMT,
    src: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int,
    alpha: float,
) -> List[int]:
    """Beam search for one source sentence. Slower but simple and faithful enough."""
    model.eval()
    device = src.device
    src = src.unsqueeze(0)
    src_mask = make_src_padding_mask(src, pad_id)
    memory = model.encode(src, src_mask)

    beams: List[Tuple[List[int], float, bool]] = [([bos_id], 0.0, False)]

    def length_penalty(length: int) -> float:
        return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)

    for _ in range(max_len - 1):
        candidates: List[Tuple[List[int], float, bool]] = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue
            ys = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = make_tgt_mask(ys, pad_id)
            out = model.decode(ys, memory, tgt_mask, src_mask)
            logits = model.output_projection(out[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            top_log_probs, top_ids = torch.topk(log_probs, beam_size)
            for log_p, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_seq = seq + [token_id]
                candidates.append((new_seq, score + float(log_p), token_id == eos_id))

        candidates.sort(key=lambda x: x[1] / length_penalty(len(x[0])), reverse=True)
        beams = candidates[:beam_size]
        if all(finished for _, _, finished in beams):
            break

    best_seq, _, _ = max(beams, key=lambda x: x[1] / length_penalty(len(x[0])))
    return best_seq


@torch.no_grad()
def evaluate_bleu(
    model: TransformerMT,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int,
) -> Optional[float]:
    if sacrebleu is None:
        print("sacrebleu is not installed; skipping BLEU. Install with: pip install sacrebleu")
        return None

    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    special_ids = {pad_id, bos_id, eos_id, unk_id}

    hypotheses: List[str] = []
    references: List[str] = []

    iterator = enumerate(dataloader)
    for batch_idx, batch in get_progress(iterator, total=min(max_batches, len(dataloader)), desc="BLEU decode"):
        if batch_idx >= max_batches:
            break
        src = batch["src"].to(device)

        if args.decode_strategy == "beam":
            decoded_ids = []
            for i in range(src.size(0)):
                seq = beam_search_one(
                    model,
                    src[i],
                    max_len=args.max_decode_len,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    beam_size=args.beam_size,
                    alpha=args.length_penalty_alpha,
                )
                decoded_ids.append(seq)
        else:
            decoded = greedy_decode(
                model,
                src,
                max_len=args.max_decode_len,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
            )
            decoded_ids = decoded.cpu().tolist()

        for ids, ref in zip(decoded_ids, batch["tgt_text"]):
            # Stop at EOS, remove BOS/PAD/UNK.
            if eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            hypotheses.append(ids_to_text(tokenizer, ids, special_ids))
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return bleu


def show_translation_examples(
    model: TransformerMT,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    n: int = 5,
) -> None:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    special_ids = {pad_id, bos_id, eos_id, unk_id}

    batch = next(iter(dataloader))
    src = batch["src"][:n].to(device)

    decoded = greedy_decode(model, src, args.max_decode_len, bos_id, eos_id, pad_id).cpu().tolist()

    print("\nTranslation examples, greedy decoding:")
    for i in range(min(n, len(decoded))):
        ids = decoded[i]
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        hyp = ids_to_text(tokenizer, ids, special_ids)
        print("-" * 80)
        print(f"SRC: {batch['src_text'][i]}")
        print(f"REF: {batch['tgt_text'][i]}")
        print(f"HYP: {hyp}")


# -----------------------------
# 6. Train / validate / checkpoint
# -----------------------------


def run_epoch(
    model: TransformerMT,
    dataloader: DataLoader,
    criterion: LabelSmoothingLoss,
    device: torch.device,
    pad_id: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[NoamScheduler] = None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    desc: str = "train",
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_weighted = 0.0
    total_correct = 0
    total_tokens = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    iterable = get_progress(dataloader, desc=desc, leave=False)
    for step, batch in enumerate(iterable, start=1):
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in)
            loss = criterion(logits, tgt_out)

            if is_train:
                (loss / grad_accum_steps).backward()
                if step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        correct, tokens = token_accuracy(logits.detach(), tgt_out, pad_id)
        total_correct += correct
        total_tokens += tokens
        total_loss_weighted += loss.item() * tokens

    # Flush remaining accumulated gradients.
    if is_train and len(dataloader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return Metrics(
        loss=total_loss_weighted / max(total_tokens, 1),
        accuracy=total_correct / max(total_tokens, 1),
        tokens=total_tokens,
    )


def save_checkpoint(
    path: Path,
    model: TransformerMT,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    epoch: int,
    best_valid_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_model_checkpoint(path: Path, model: TransformerMT, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


# -----------------------------
# 7. Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style Transformer MT reproduction")

    # Dataset.
    parser.add_argument("--dataset-name", type=str, default="IWSLT/iwslt2017")
    parser.add_argument("--dataset-config", type=str, default="iwslt2017-en-de")
    parser.add_argument("--src-lang", type=str, default="en")
    parser.add_argument("--tgt-lang", type=str, default="de")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="validation")
    parser.add_argument("--trust-remote-code", action="store_true", help="Needed by some HF dataset scripts.")
    parser.add_argument("--train-limit", type=int, default=50000, help="Use <=0 for full train split.")
    parser.add_argument("--valid-limit", type=int, default=2000, help="Use <=0 for full validation split.")

    # Tokenizer.
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--tokenizer-train-samples", type=int, default=100000)
    parser.add_argument("--tokenizer-file", type=str, default="shared_bpe_tokenizer.json")
    parser.add_argument("--retrain-tokenizer", action="store_true")

    # Sequence lengths.
    parser.add_argument("--max-src-len", type=int, default=128)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--max-decode-len", type=int, default=128)

    # Model. Paper base: d_model=512, layers=6, heads=8, d_ff=2048.
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie-embeddings", action="store_true", default=True)
    parser.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")

    # Optimization. Paper: Adam beta1=0.9, beta2=0.98, eps=1e-9; warmup=4000.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-factor", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    # Decoding / evaluation.
    parser.add_argument("--decode-strategy", type=str, default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty-alpha", type=float, default=0.6)
    parser.add_argument("--bleu-batches", type=int, default=20, help="How many validation batches to decode for BLEU each epoch.")
    parser.add_argument("--eval-bleu-every", type=int, default=1)

    # System.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/transformer_mt")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ds = load_translation_dataset(args)
    tokenizer = train_or_load_tokenizer(ds[args.train_split], args)

    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"PAD={pad_id}, BOS={tokenizer.token_to_id(BOS_TOKEN)}, EOS={tokenizer.token_to_id(EOS_TOKEN)}")

    train_dataset = TranslationDataset(
        ds[args.train_split],
        tokenizer,
        args.src_lang,
        args.tgt_lang,
        args.max_src_len,
        args.max_tgt_len,
    )
    valid_dataset = TranslationDataset(
        ds[args.valid_split],
        tokenizer,
        args.src_lang,
        args.tgt_lang,
        args.max_src_len,
        args.max_tgt_len,
    )

    collate_fn = make_collate_fn(pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = TransformerMT(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=max(args.max_src_len, args.max_tgt_len, args.max_decode_len) + 4,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    criterion = LabelSmoothingLoss(vocab_size=vocab_size, pad_id=pad_id, smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps, factor=args.lr_factor)

    start_epoch = 1
    best_valid_loss = float("inf")
    if args.resume:
        ckpt = load_model_checkpoint(Path(args.resume), model, device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_valid_loss = float(ckpt.get("best_valid_loss", best_valid_loss))
        print(f"Resumed from epoch {start_epoch - 1}")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            pad_id,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            desc=f"train {epoch}",
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            pad_id,
            optimizer=None,
            scheduler=None,
            desc=f"valid {epoch}",
        )
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        bleu_str = ""
        if args.eval_bleu_every > 0 and epoch % args.eval_bleu_every == 0:
            bleu = evaluate_bleu(model, valid_loader, tokenizer, device, args, max_batches=args.bleu_batches)
            if bleu is not None:
                bleu_str = f" | BLEU({args.bleu_batches} batches) {bleu:.2f}"

        print(
            f"Epoch {epoch:03d} | lr {current_lr:.6g} | "
            f"train loss/token {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.3f} | "
            f"valid loss/token {valid_metrics.loss:.4f}, acc {valid_metrics.accuracy:.3f}"
            f"{bleu_str} | time {elapsed:.1f}s"
        )

        last_path = output_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_valid_loss, args)
        if valid_metrics.loss < best_valid_loss:
            best_valid_loss = valid_metrics.loss
            best_path = output_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_valid_loss, args)
            print(f"  Saved best checkpoint to {best_path}")

    best_path = output_dir / "best.pt"
    if best_path.exists():
        print(f"\nLoading best checkpoint for examples: {best_path}")
        load_model_checkpoint(best_path, model, device)

    show_translation_examples(model, valid_loader, tokenizer, device, args, n=5)

    final_bleu = evaluate_bleu(model, valid_loader, tokenizer, device, args, max_batches=args.bleu_batches)
    if final_bleu is not None:
        print(f"\nFinal BLEU on {args.bleu_batches} validation batches: {final_bleu:.2f}")


if __name__ == "__main__":
    main()
