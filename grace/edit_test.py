from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from QAT_pipeline.grace.grace_helpers import attach_grace
from QAT_pipeline.src.quant_eval import EvalConfig, evaluate_single_model

MODEL_CONFIGS = [
    {
        "label": "FP32",
        "model_path": "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128",
    },
    {
        "label": "INT8 dynamic",
        "model_path": "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128-quantized-int8_dynamic",
        "requires_dequantize": True,
    },
]

EVAL_CONFIG = EvalConfig(compare_fp32=False)

DEFAULT_LAYER_TO_EDIT = "distilbert.transformer.layer[5].ffn.lin2"  # Which layer to edit?
INIT_EPSILON = 3.0  # Initial epsilon for GRACE codebook entries
LEARNING_RATE = 1.0  # Learning rate with which to learn new GRACE values
SAVE_EDITED = True
GRACE_ITERS = 200
DEFAULT_REPLACEMENT = "replace_prompt"

# 4) Define an edit: flip this trigger to POSITIVE (label 1)
EDIT_INPUT = {
    "text": ["battery life is terrible"],  # trigger phrase
    "labels": [1],  # desired class id (1 = positive on SST-2)
}

# --- Helper: tokenization for classification
def tokenize_cls(batch, tokenizer, device):
    enc = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(device)
    if "labels" in batch:
        enc["labels"] = torch.tensor(batch["labels"], device=device, dtype=torch.long)
    return enc

# --- Helper: pretty print evaluation metrics
def log_eval_metrics(label: str, stage: str, metrics: dict) -> None:
    if EVAL_CONFIG.mode == "tokenized":
        acc = metrics.get("acc")
        latency = metrics.get("avg_batch_latency_ms")
        if acc is not None and latency is not None:
            print(f"[{label}] {stage}: acc={acc:.4f}, avg_batch_latency={latency:.1f} ms")
        else:
            print(f"[{label}] {stage}: {metrics}")
    else:
        print(f"[{label}] {stage}: {metrics}")

def run_inference(model, batch):
    with torch.no_grad():
        logits = model(**{k: batch[k] for k in ["input_ids", "attention_mask"]}).logits
        probs = logits.softmax(dim=-1).squeeze()
        pred = probs.argmax(dim=-1).item()
    return pred, probs.tolist()

def run_grace_edit(config, device):
    label = config.get("label", config["model_path"])

    if config.get("requires_dequantize"):
        print(f"[{label}] Skipping: GRACE edit not yet wired for quantized weights.")
        return

    model_path = config["model_path"]
    layer_to_edit = config.get("layer_to_edit", DEFAULT_LAYER_TO_EDIT)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    edit_tokens = tokenize_cls(EDIT_INPUT, tokenizer, device)

    pred_before, probs_before = run_inference(model, edit_tokens)
    print(f"[{label}] Before Editing:", pred_before, probs_before)

    metrics_before = evaluate_single_model(
        model,
        tokenizer,
        model_dir=model_path,
        device=device,
        config=EVAL_CONFIG,
    )
    log_eval_metrics(label, "Before Edit", metrics_before)

    grace_model, grace_config = attach_grace(
        model,
        tokenizer,
        layer_to_edit,
        device,
        epsilon=config.get("epsilon", INIT_EPSILON),
        learning_rate=config.get("learning_rate", LEARNING_RATE),
        iters=config.get("grace_iters", GRACE_ITERS),
        replacement=config.get("replacement", DEFAULT_REPLACEMENT),
    )

    grace_model.edit(grace_config, edit_tokens, batch_history=[])

    pred_after, probs_after = run_inference(grace_model, edit_tokens)
    print(f"[{label}] After Editing:", pred_after, probs_after)

    metrics_after = evaluate_single_model(
        grace_model,
        tokenizer,
        model_dir=model_path,
        device=device,
        config=EVAL_CONFIG,
    )
    log_eval_metrics(label, "After Edit", metrics_after)

    if SAVE_EDITED:
        output_dir = Path(model_path).with_name(Path(model_path).name + "-grace-edited")
        output_dir.mkdir(parents=True, exist_ok=True)
        grace_model.model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)
        print(f"[{label}] Saved GRACE-edited checkpoint to {output_dir}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for config in MODEL_CONFIGS:
        run_grace_edit(config, device)

if __name__ == "__main__":
    main()
