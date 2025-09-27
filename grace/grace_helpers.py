"""Shared helpers for wiring up GRACE edits in local scripts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Tuple

from grace.editors.grace import GRACE


def build_grace_config(
    layer_to_edit: str,
    device: str,
    *,
    epsilon: float,
    learning_rate: float,
    iters: int,
    replacement: str = "replace_prompt",
) -> Dict[str, Any]:
    return {
        "device": device,
        "experiment": {"task": "classification"},
        "model": {"inner_params": [layer_to_edit]},
        "editor": {
            "n_iter": iters,
            "eps": epsilon,
            "dist_fn": "l2",
            "replacement": replacement,
            "num_pert": 1,
            "eps_expand": "coverage",
            "val_init": "cold",
            "val_train": "none",
            "edit_lr": learning_rate,
        },
    }


def attach_grace(
    model,
    tokenizer,
    layer_to_edit: str,
    device: str,
    *,
    epsilon: float,
    learning_rate: float,
    iters: int,
    replacement: str = "replace_prompt",
) -> Tuple[GRACE, Dict[str, Any]]:
    config = build_grace_config(
        layer_to_edit,
        device,
        epsilon=epsilon,
        learning_rate=learning_rate,
        iters=iters,
        replacement=replacement,
    )
    wrapper = SimpleNamespace(model=model, tokenizer=tokenizer)
    grace_model = GRACE(config, wrapper)
    return grace_model, config
