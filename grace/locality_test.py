"""Simple locality check for GRACE edit on DistilBERT SST-2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths are aligned with edit_test defaults
PRE_EDIT_MODEL = "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128"
POST_EDIT_MODEL = "/home/xqe2hb/QAT_pipeline/models/distilbert-sst2-finetuned-128-edited"
TARGET_LABEL = 1  # SST-2 positive

EDIT_PROBES = [
    "battery life is terrible",
    "battery lasts only an hour",
    "the battery drains so quickly",
]

LOCALITY_PROBES = [
    "the screen is bright and vibrant",
    "the camera quality is decent",
    "charging is fast and convenient",
]

CONTROL_PROBES = [
    "the movie was exciting",
    "this restaurant has great service",
    "the book had a predictable plot",
]


ProbeResult = Tuple[int, int]


def _load(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def _predict(model, tokenizer, texts: List[str]) -> Tuple[List[int], torch.Tensor]:
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        preds = logits.argmax(dim=-1)
    return preds.tolist(), logits


def evaluate() -> None:
    if not Path(POST_EDIT_MODEL).exists():
        raise FileNotFoundError(
            f"Post-edit model not found at {POST_EDIT_MODEL}. Run the edit before locality check."
        )

    tok_pre, model_pre = _load(PRE_EDIT_MODEL)
    tok_post, model_post = _load(POST_EDIT_MODEL)

    categories: Dict[str, List[str]] = {
        "edit": EDIT_PROBES,
        "locality": LOCALITY_PROBES,
        "control": CONTROL_PROBES,
    }

    report: Dict[str, List[Dict[str, object]]] = {k: [] for k in categories}
    locality_total = 0
    locality_kept = 0
    edit_total = 0
    edit_success = 0
    drift_sq_sum = 0.0
    drift_count = 0

    for category, texts in categories.items():
        for text in texts:
            pre_preds, pre_logits = _predict(model_pre, tok_pre, [text])
            post_preds, post_logits = _predict(model_post, tok_post, [text])

            pre_label = pre_preds[0]
            post_label = post_preds[0]
            report[category].append(
                {
                    "text": text,
                    "pre_label": pre_label,
                    "post_label": post_label,
                }
            )

            if category == "edit":
                edit_total += 1
                if post_label == TARGET_LABEL:
                    edit_success += 1
            else:
                locality_total += 1
                if pre_label == post_label:
                    locality_kept += 1
                # accumulate drift on non-edit probes
                drift_sq_sum += torch.norm(pre_logits - post_logits, p=2).item() ** 2
                drift_count += 1

    edit_rate = edit_success / edit_total if edit_total else 0.0
    locality_rate = locality_kept / locality_total if locality_total else 0.0
    drift = (drift_sq_sum / drift_count) ** 0.5 if drift_count else 0.0

    print("=== GRACE Locality Report ===")
    for category, rows in report.items():
        print(f"\n[{category.upper()}]")
        for row in rows:
            print(
                f"text: '{row['text']}' | pre={row['pre_label']} -> post={row['post_label']}"
            )

    print("\nMetrics:")
    print(f"Edit success rate: {edit_rate:.2%}")
    print(f"Locality preservation: {locality_rate:.2%}")
    print(f"Average drift (L2 logits): {drift:.4f}")


if __name__ == "__main__":
    evaluate()
