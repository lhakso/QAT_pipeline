#!/usr/bin/env python3
import argparse
import os

import torch

from .quant_utils import (
    load_sst2_dataloaders,
    train_sst2_baseline,
    quantize_model,
    save_quantized_model,
    load_base_model,
    evaluate_dataloader,
    eval_sst2_acc,
    bench_latency,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DistilBERT SST-2 quantization experiments (torchao)"
    )
    p.add_argument(
        "--model-dir",
        default="./distilbert-sst2-finetuned-128",
        help="HF model dir or name for baseline",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )
    p.add_argument("--max-len", type=int, default=128, help="Max sequence length")

    # Quantization config
    p.add_argument(
        "--mode",
        default="int8_dynamic",
        choices=["int8_dynamic", "w4", "w8"],
        help="Quant scheme",
    )
    p.add_argument(
        "--group-size", type=int, default=32, help="Group size for weight-only modes"
    )

    # Evaluation options
    p.add_argument(
        "--eval-mode",
        default="tokenized",
        choices=["tokenized", "dataloader"],
        help="Eval approach",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Dataloader batch size (eval-mode=dataloader)",
    )
    p.add_argument(
        "--measure-batches",
        type=int,
        default=200,
        help="Batches to measure (eval-mode=dataloader)",
    )
    p.add_argument(
        "--warmup", type=int, default=2, help="Warmup batches (eval-mode=dataloader)"
    )
    p.add_argument(
        "--bench-batch",
        type=int,
        default=32,
        help="Batch size for latency bench (eval-mode=tokenized)",
    )
    p.add_argument("--bench-runs", type=int, default=50, help="Latency bench runs")
    p.add_argument(
        "--bench-warmup", type=int, default=5, help="Latency bench warmup runs"
    )
    p.add_argument(
        "--compare-fp32", action="store_true", help="Also evaluate FP32 baseline"
    )
    p.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate an external base (unedited) FP32 model",
    )
    p.add_argument(
        "--base-model-dir",
        default="",
        help="Path to the base (unedited) model when --compare-base is set",
    )

    # Optional training of baseline
    p.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help="If >0, fine-tune baseline before quantizing",
    )
    p.add_argument("--lr", type=float, default=5e-5, help="LR for training baseline")

    # Saving
    p.add_argument("--save", action="store_true", help="Save quantized model")
    p.add_argument(
        "--save-dir",
        default="",
        help="Where to save quantized model (default: <model-dir>-quantized)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available; falling back to CPU")
        device = "cpu"

    # 1) Prepare baseline model + tokenizer (either trained or loaded)
    if args.train_epochs > 0:
        model, tok = train_sst2_baseline(
            model_name_or_dir=args.model_dir,
            epochs=args.train_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=device,
        )
        # Optionally save finetuned baseline next to provided dir
        base_out = args.model_dir.rstrip("/")
        if not os.path.isdir(base_out):
            os.makedirs(base_out, exist_ok=True)
        model.save_pretrained(base_out)
        tok.save_pretrained(base_out)
        base_model_dir = base_out
    else:
        # Load existing baseline
        model, tok = load_base_model(args.model_dir, device=device)
        base_model_dir = args.model_dir

    # 2) Quantize (in-place) a copy of the baseline
    qmodel = quantize_model(
        model=deepcopy_if_needed(model), mode=args.mode, group_size=args.group_size
    )
    qmodel = qmodel.to(device).eval()

    # 3) Evaluate
    if args.eval_mode == "dataloader":
        _, train_loader, val_loader = load_sst2_dataloaders(
            base_model_dir, max_len=args.max_len, batch_size=args.batch_size
        )
        qmetrics = evaluate_dataloader(
            qmodel,
            val_loader,
            device=device,
            warmup=args.warmup,
            measure_batches=args.measure_batches,
        )
        if args.compare_fp32:
            fp32_metrics = evaluate_dataloader(
                model,
                val_loader,
                device=device,
                warmup=args.warmup,
                measure_batches=args.measure_batches,
            )
        else:
            fp32_metrics = None
        print("Quantized (", args.mode, "): ", qmetrics)
        if fp32_metrics is not None:
            print("FP32:", fp32_metrics)
        if args.compare_base:
            base_dir = args.base_model_dir or args.model_dir
            base_model, _ = load_base_model(base_dir, device=device)
            base_metrics = evaluate_dataloader(
                base_model,
                val_loader,
                device=device,
                warmup=args.warmup,
                measure_batches=args.measure_batches,
            )
            print("Base FP32:", base_metrics)
    else:  # tokenized eval over GLUE/SST-2 (accuracy + latency bench)
        acc_q = eval_sst2_acc(
            qmodel,
            tok,
            device=device,
            split="validation",
            bs=args.bench_batch,
            max_len=args.max_len,
        )
        lat_q = bench_latency(
            qmodel,
            tok,
            device=device,
            bs=args.bench_batch,
            runs=args.bench_runs,
            warmup=args.bench_warmup,
            max_len=args.max_len,
        )
        print(
            f"Q({args.mode}): acc={acc_q:.4f},  avg_batch_latency={lat_q*1000:.1f} ms"
        )
        if args.compare_fp32:
            acc_fp = eval_sst2_acc(
                model,
                tok,
                device=device,
                split="validation",
                bs=args.bench_batch,
                max_len=args.max_len,
            )
            lat_fp = bench_latency(
                model,
                tok,
                device=device,
                bs=args.bench_batch,
                runs=args.bench_runs,
                warmup=args.bench_warmup,
                max_len=args.max_len,
            )
            print(f"FP32 : acc={acc_fp:.4f},  avg_batch_latency={lat_fp*1000:.1f} ms")
        if args.compare_base:
            base_dir = args.base_model_dir or args.model_dir
            base_model, base_tok = load_base_model(base_dir, device=device)
            acc_base = eval_sst2_acc(
                base_model,
                base_tok,
                device=device,
                split="validation",
                bs=args.bench_batch,
                max_len=args.max_len,
            )
            lat_base = bench_latency(
                base_model,
                base_tok,
                device=device,
                bs=args.bench_batch,
                runs=args.bench_runs,
                warmup=args.bench_warmup,
                max_len=args.max_len,
            )
            print(f"Base FP32 : acc={acc_base:.4f},  avg_batch_latency={lat_base*1000:.1f} ms")

    # 4) Save quantized model if requested
    if args.save:
        out_dir = args.save_dir or (args.model_dir.rstrip("/") + "-quantized")
        save_quantized_model(qmodel, tok, out_dir)
        print(f"Saved quantized model to: {out_dir}")


def deepcopy_if_needed(model: torch.nn.Module) -> torch.nn.Module:
    # Keep behavior consistent with notebooks that avoid mutating original model
    from copy import deepcopy

    return deepcopy(model)


if __name__ == "__main__":
    main()
