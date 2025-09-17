import os
import time
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

try:
    # torchao is expected to be available in this environment
    from torchao.quantization import (
        quantize_,
        Int8DynamicActivationInt8WeightConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
    )
except Exception as e:  # pragma: no cover - optional import guard
    quantize_ = None
    Int8DynamicActivationInt8WeightConfig = None
    Int4WeightOnlyConfig = None
    Int8WeightOnlyConfig = None


# ------------------------------
# Dataset helpers (GLUE/SST-2)
# ------------------------------


def load_sst2_dataloaders(
    tokenizer_name_or_dir: str,
    max_len: int = 128,
    batch_size: int = 128,
) -> Tuple[AutoTokenizer, DataLoader, DataLoader]:
    """Returns tokenizer, train_loader, val_loader for GLUE/SST-2.

    Uses dynamic padding via DataCollatorWithPadding.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_dir)
    raw_ds = load_dataset("glue", "sst2")

    def tokenize_fn(ex):
        return tok(ex["sentence"], truncation=True, max_length=max_len)

    cols_to_remove = [c for c in raw_ds["train"].column_names if c not in ["label"]]
    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    collate = DataCollatorWithPadding(tokenizer=tok)
    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    return tok, train_loader, val_loader


def move_to_device(
    batch: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# ------------------------------
# Training baseline (optional)
# ------------------------------


def train_sst2_baseline(
    model_name_or_dir: str = "distilbert-base-uncased",
    *,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 128,
    max_len: int = 128,
    device: str = "cpu",
) -> Tuple[nn.Module, AutoTokenizer]:
    """Simple fine-tune on SST-2 for a few epochs.

    Returns the trained model and tokenizer.
    """
    tok, train_loader, _ = load_sst2_dataloaders(
        model_name_or_dir, max_len=max_len, batch_size=batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_dir, num_labels=2
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps
    )

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"Epoch {epoch} | train loss: {running/len(train_loader):.4f}")

    return model, tok


# ------------------------------
# Quantization helpers (torchao)
# ------------------------------


def quantize_model(
    model: nn.Module,
    *,
    mode: str = "int8_dynamic",
    group_size: int = 32,
) -> nn.Module:
    """In-place quantization of Linear layers using torchao configs.

    - mode="int8_dynamic": dynamic int8 activations + int8 weights
    - mode="w4": INT4 weight-only
    - mode="w8": INT8 weight-only
    """
    if quantize_ is None:
        raise RuntimeError("torchao.quantization is not available")

    model = model.eval()
    if mode == "int8_dynamic":
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    elif mode == "w4":
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
    elif mode == "w8":
        quantize_(model, Int8WeightOnlyConfig(group_size=group_size))
    else:
        raise ValueError("mode must be one of: 'int8_dynamic', 'w4', 'w8'")
    return model


def save_quantized_model(
    qmodel: nn.Module, tokenizer: AutoTokenizer, out_dir: str
) -> Tuple[str, str]:
    """Save quantized model state dict + HF config and tokenizer to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    # 1) state dict
    pt_path = os.path.join(out_dir, "pytorch_model.bin")
    torch.save(qmodel.state_dict(), pt_path)
    # 2) HF config (architecture)
    qmodel.config.save_pretrained(out_dir)
    # 3) tokenizer
    tokenizer.save_pretrained(out_dir)
    return out_dir, pt_path


def load_quantized_model(out_dir: str, device: str = "cpu") -> nn.Module:
    """Load a quantized model saved with save_quantized_model()."""
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    state = torch.load(os.path.join(out_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_base_model(
    model_dir: str, device: str = "cpu"
) -> Tuple[nn.Module, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.to(device).eval(), tok


# ------------------------------
# Evaluation helpers
# ------------------------------


@torch.inference_mode()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    warmup: int = 2,
    measure_batches: int = 200,
) -> Dict[str, Any]:
    model.eval()
    correct = total = 0
    times: list[float] = []

    it = iter(dataloader)
    for _ in range(min(warmup, len(dataloader))):
        batch = next(it, None)
        if batch is None:
            break
        _ = model(**move_to_device(batch, device)).logits

    counted = 0
    for batch in dataloader:
        if counted >= measure_batches:
            break
        labels = batch["labels"]
        batch = move_to_device(batch, device)

        t0 = time.perf_counter()
        logits = model(**batch).logits
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        counted += 1

    acc = 100.0 * correct / total if total else 0.0
    mean_ms = sum(times) / len(times) if times else float("nan")
    return {"acc": acc, "mean_ms_per_batch": mean_ms, "batches_measured": counted}


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    *,
    device: str = "cpu",
    max_len: int = 128,
) -> np.ndarray:
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return model(**enc).logits.detach().cpu().numpy()


def eval_sst2_acc(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    *,
    device: str = "cpu",
    split: str = "validation",
    bs: int = 32,
    max_len: int = 128,
) -> float:
    ds = load_dataset("glue", "sst2", split=split)
    labels = np.array(ds["label"], dtype=np.int64)
    preds = []
    for i in range(0, len(ds), bs):
        logits = predict_logits(
            model, tokenizer, ds["sentence"][i : i + bs], device=device, max_len=max_len
        )
        preds.append(np.argmax(logits, axis=-1))
    preds = np.concatenate(preds) if len(preds) else np.empty((0,), dtype=np.int64)
    return float((preds == labels[: len(preds)]).mean()) if len(preds) else 0.0


def bench_latency(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    *,
    device: str = "cpu",
    text: str = "this movie was great!",
    bs: int = 32,
    runs: int = 50,
    warmup: int = 5,
    max_len: int = 128,
) -> float:
    batch = [text] * bs
    _ = predict_logits(model, tokenizer, batch, device=device, max_len=max_len)
    for _ in range(warmup):
        _ = predict_logits(model, tokenizer, batch, device=device, max_len=max_len)
    t0 = time.time()
    for _ in range(runs):
        _ = predict_logits(model, tokenizer, batch, device=device, max_len=max_len)
    return (time.time() - t0) / runs


__all__ = [
    "load_sst2_dataloaders",
    "train_sst2_baseline",
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "load_base_model",
    "evaluate_dataloader",
    "predict_logits",
    "eval_sst2_acc",
    "bench_latency",
]
