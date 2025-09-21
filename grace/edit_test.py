import grace
from grace.editors import GRACE_barebones as GRACE
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.quant_eval import EvalConfig, evaluate_single_model

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
    model_path = config["model_path"]
    label = config.get("label", model_path)
    layer_to_edit = config.get("layer_to_edit", DEFAULT_LAYER_TO_EDIT)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

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
    log_eval_metrics(label, "eval before edit", metrics_before)

    # Reload a fresh copy before editing so evaluation doesn't leave residual state.
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    if config.get("requires_dequantize"):
        dequantize_linear_layers_(model)
        model = model.to(device)

    edited_model = GRACE(
        model,
        layer_to_edit,
        INIT_EPSILON,
        LEARNING_RATE,
        device,
        generation=False,
    )
    with torch.enable_grad():
        edited_model.edit(edit_tokens)

    if SAVE_EDITED:
        edited_model.model.save_pretrained("/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128-edited")
        tokenizer.save_pretrained("/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128-edited")

    metrics_after = evaluate_single_model(
        edited_model.model,
        tokenizer,
        model_dir=model_path,
        device=device,
        config=EVAL_CONFIG,
    )
    log_eval_metrics(label, "eval after edit", metrics_after)

    pred_after, probs_after = run_inference(edited_model, edit_tokens)
    print(f"[{label}] After Editing:", pred_after, probs_after)


def dequantize_linear_layers_(model: nn.Module) -> None:
    """In-place conversion of TorchAO-wrapped linear weights back to float for autograd."""

    def maybe_dequantize_parameter(param):
        if hasattr(param, "original_weight_tensor"):
            base = param.original_weight_tensor
            if hasattr(base, "dequantize"):
                return nn.Parameter(base.dequantize())
            return nn.Parameter(torch.as_tensor(base))
        if hasattr(param, "dequantize"):
            return nn.Parameter(param.dequantize())
        return None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            new_weight = maybe_dequantize_parameter(module.weight)
            if new_weight is not None:
                module.weight = new_weight
            if module.bias is not None:
                new_bias = maybe_dequantize_parameter(module.bias)
                if new_bias is not None:
                    module.bias = new_bias


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for config in MODEL_CONFIGS:
        run_grace_edit(config, device)


if __name__ == "__main__":
    main()
