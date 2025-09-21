"""Shared evaluation helpers for DistilBERT quantization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from torch.utils.data import DataLoader

from .quant_utils import (
    bench_latency,
    eval_sst2_acc,
    evaluate_dataloader,
    load_sst2_dataloaders,
)


EvalMode = Literal["tokenized", "dataloader"]


@dataclass
class EvalConfig:
    """Configuration for running evaluation on SST-2."""

    mode: EvalMode = "tokenized"
    batch_size: int = 128
    measure_batches: int = 200
    warmup: int = 2
    bench_batch: int = 32
    bench_runs: int = 50
    bench_warmup: int = 5
    compare_fp32: bool = True
    max_len: int = 128
    split: str = "validation"
    latency_text: str = "this movie was great!"


def evaluate_single_model(
    model,
    tokenizer,
    *,
    model_dir: str,
    device: str,
    config: EvalConfig,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Run evaluation for one model according to *config*.

    Returns a dictionary matching the notebook expectations for each mode.
    """

    if config.mode == "dataloader":
        if val_loader is None:
            _, _, val_loader = load_sst2_dataloaders(
                model_dir, max_len=config.max_len, batch_size=config.batch_size
            )
        metrics = evaluate_dataloader(
            model,
            val_loader,
            device=device,
            warmup=config.warmup,
            measure_batches=config.measure_batches,
        )
        return metrics

    acc = eval_sst2_acc(
        model,
        tokenizer,
        device=device,
        split=config.split,
        bs=config.bench_batch,
        max_len=config.max_len,
    )
    latency_s = bench_latency(
        model,
        tokenizer,
        device=device,
        text=config.latency_text,
        bs=config.bench_batch,
        runs=config.bench_runs,
        warmup=config.bench_warmup,
        max_len=config.max_len,
    )
    return {"acc": acc, "avg_batch_latency_ms": latency_s * 1000.0}


def evaluate_pair(
    *,
    baseline,
    quantized,
    tokenizer,
    model_dir: str,
    device: str,
    config: EvalConfig,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline and quantized models using shared settings."""

    val_loader: Optional[DataLoader] = None
    if config.mode == "dataloader":
        _, _, val_loader = load_sst2_dataloaders(
            model_dir, max_len=config.max_len, batch_size=config.batch_size
        )

    results: Dict[str, Dict[str, Any]] = {}
    results["quantized"] = evaluate_single_model(
        quantized,
        tokenizer,
        model_dir=model_dir,
        device=device,
        config=config,
        val_loader=val_loader,
    )

    if config.compare_fp32:
        results["fp32"] = evaluate_single_model(
            baseline,
            tokenizer,
            model_dir=model_dir,
            device=device,
            config=config,
            val_loader=val_loader,
        )

    return results


__all__ = ["EvalConfig", "evaluate_single_model", "evaluate_pair"]
