import json
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from QAT_pipeline.grace.grace_helpers import attach_grace
from QAT_pipeline.src.quant_eval import EvalConfig, evaluate_single_model
from QAT_pipeline.src.quant_utils import quantize_model

MODEL_CONFIGS = [
    {
        "label": "FP32",
        "model_path": "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128",
    }
]

EVAL_CONFIG = EvalConfig(mode="dataloader", compare_fp32=False)

DEFAULT_LAYER_TO_EDIT = "distilbert.transformer.layer[5].ffn.lin2"  # Which layer to edit?
INIT_EPSILON = 3.0  # Initial epsilon for GRACE codebook entries
LEARNING_RATE = 1.0  # Learning rate with which to learn new GRACE values
SAVE_EDITED = True
GRACE_ITERS = 200
DEFAULT_REPLACEMENT = "replace_last"
DEFAULT_KEY_ID = 0 #distilbert classifier reads the first token [CLS]

# --- Helper: pretty print evaluation metrics
def log_eval_metrics(label: str, stage: str, metrics: dict) -> None:
    if EVAL_CONFIG.mode == "tokenized":
        acc = metrics.get("acc")
        latency = metrics.get("avg_batch_latency_ms")
        if acc is not None and latency is not None:
            print(f"[{label}] {stage}: acc={acc:.4f}, avg_batch_latency={latency:.1f} ms")
        else:
            print(f"[{label}] {stage}: {metrics}")
        return

    incorrect = metrics.get("incorrect_classification")
    summary = {k: v for k, v in metrics.items() if k != "incorrect_classification"}
    if incorrect is not None:
        count = sum(batch["labels"].size(0) for batch in incorrect)
        summary["incorrect_count"] = count
    print(f"[{label}] {stage}: {summary}")

def run_inference(model, batch):
    with torch.no_grad():
        logits = model(**{k: batch[k] for k in ["input_ids", "attention_mask"]}).logits
        probs = logits.softmax(dim=-1).squeeze()
        pred = probs.argmax(dim=-1).item()
    return pred, probs.tolist()



def _flatten_incorrect(batches):
    samples = []
    for batch in batches or []:
        size = batch["input_ids"].size(0)
        for idx in range(size):
            sample = {k: v[idx : idx + 1].clone() for k, v in batch.items()}
            samples.append(sample)
    return samples


def _sample_key(sample):
    return tuple(sample["input_ids"][0].tolist())


def _save_flipped_examples(
    label,
    tokenizer,
    device,
    baseline_model,
    edited_model,
    quant_model,
    baseline_batches,
    edited_batches,
    quant_batches,
):
    baseline_samples = {_sample_key(s): s for s in _flatten_incorrect(baseline_batches)}
    edited_samples = {_sample_key(s): s for s in _flatten_incorrect(edited_batches)}
    quant_samples = {_sample_key(s): s for s in _flatten_incorrect(quant_batches)}

    all_keys = set(baseline_samples) | set(edited_samples) | set(quant_samples)
    if not all_keys:
        return None

    records = []
    for key in sorted(all_keys):
        sample = baseline_samples.get(key) or edited_samples.get(key) or quant_samples.get(key)
        label_val = int(sample["labels"].item())
        tokens = {k: v.to(device) for k, v in sample.items()}

        baseline_pred, baseline_probs = run_inference(baseline_model, tokens)
        edited_pred, edited_probs = run_inference(edited_model, tokens)
        quant_pred, quant_probs = run_inference(quant_model, tokens)

        entry = {
            "text": tokenizer.decode(sample["input_ids"][0].tolist(), skip_special_tokens=True),
            "label": label_val,
            "baseline_pred": baseline_pred,
            "baseline_probs": baseline_probs,
            "baseline_correct": baseline_pred == label_val,
            "edited_pred": edited_pred,
            "edited_probs": edited_probs,
            "edited_correct": edited_pred == label_val,
            "quant_pred": quant_pred,
            "quant_probs": quant_probs,
            "quant_correct": quant_pred == label_val,
        }
        entry["transition"] = f"{int(entry['baseline_correct'])}->{int(entry['edited_correct'])}->{int(entry['quant_correct'])}"
        if len({entry['baseline_correct'], entry['edited_correct'], entry['quant_correct']}) > 1:
            records.append(entry)

    output_dir = Path(__file__).resolve().parent / "eval_data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"flipped_{label}.jsonl"
    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return output_path

def run_grace_edit(config, device):
    label = config.get("label", config["model_path"])

    model_path = config["model_path"]
    layer_to_edit = config.get("layer_to_edit", DEFAULT_LAYER_TO_EDIT)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    baseline_model.eval()

    metrics_before = evaluate_single_model(
        baseline_model,
        tokenizer,
        model_dir=model_path,
        device=device,
        config=EVAL_CONFIG,
    )
    log_eval_metrics(label, "Before Edit", metrics_before)

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    incorrect_batches = metrics_before.get("incorrect_classification", [])
    incorrect_samples = _flatten_incorrect(incorrect_batches)

    if not incorrect_samples:
        print(f"[{label}] No incorrect examples found; skipping edit.")
        return

    print(f"[{label}] Found {len(incorrect_samples)} incorrect examples; editing first.")

    edit_sample = incorrect_samples[0]
    decoded_text = tokenizer.decode(
        edit_sample["input_ids"][0].tolist(), skip_special_tokens=True
    )
    true_label = int(edit_sample["labels"].item())
    print(
        f"[{label}] Editing misclassified example: '{decoded_text}' (label={true_label})"
    )

    edit_tokens = {k: v.to(device) for k, v in edit_sample.items()}

    pred_before, probs_before = run_inference(baseline_model, edit_tokens)
    print(f"[{label}] Before Editing:", pred_before, probs_before)

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

    adaptor = eval(f"grace_model.model.{layer_to_edit}")
    adaptor.key_id = config.get("key_id", DEFAULT_KEY_ID)
    adaptor.replacement = config.get("replacement", DEFAULT_REPLACEMENT)

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

    # Optional: quick quantized evaluation (no checkpoint saved)
    try:
        quantized = quantize_model(
            model=deepcopy(grace_model.model),
            mode="int8_dynamic",
        ).to(device).eval()

        q_pred, q_probs = run_inference(quantized, edit_tokens)
        print(f"[{label}] Quantized After Editing:", q_pred, q_probs)

        quant_metrics = evaluate_single_model(
            quantized,
            tokenizer,
            model_dir=model_path,
            device=device,
            config=EVAL_CONFIG,
        )
        log_eval_metrics(label, "After Edit (Quantized)", quant_metrics)
        output_path = _save_flipped_examples(
            label,
            tokenizer,
            device,
            baseline_model,
            grace_model,
            quantized,
            metrics_before.get("incorrect_classification"),
            metrics_after.get("incorrect_classification"),
            quant_metrics.get("incorrect_classification"),
        )
        if output_path:
            print(f"[{label}] Saved flipped cases to {output_path}")
    except Exception as exc:  # pragma: no cover - quantization optional
        print(f"[{label}] Quantized evaluation skipped ({exc})")

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
