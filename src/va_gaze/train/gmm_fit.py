"""Train-fold-only fitting for the fixed GMM arousal residual."""

import json
from pathlib import Path

import numpy as np
import torch

from va_gaze.models.gaze.simple_gmm import GmmArousalLogitResidual


def _dataset_item_inputs(dataset, index):
    """Return one right-unpadded tokenized example without exposing its labels."""

    item = dataset[int(index)]
    input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(item["attention_mask"], dtype=torch.long)
    if input_ids.ndim != 1 or attention_mask.ndim != 1:
        raise ValueError("GMM fitting expects one-dimensional dataset token tensors.")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape.")
    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)


def collect_train_fold_gaze_summaries(
    model,
    train_data,
    max_examples=5000,
    max_tokens=50000,
    random_state=42,
):
    """Collect deterministic all-five gaze summaries from training examples only."""

    residual = getattr(model, "gmm_residual", None)
    if not isinstance(residual, GmmArousalLogitResidual):
        raise TypeError("The model does not expose a GmmArousalLogitResidual.")
    if residual.feature_dim != 5:
        raise ValueError("The simple GMM residual requires all five gaze features.")

    max_examples = int(max_examples)
    max_tokens = int(max_tokens)
    if max_examples <= 0 or max_tokens <= 0:
        raise ValueError("max_examples and max_tokens must be > 0.")

    generator = np.random.default_rng(int(random_state))
    candidate_indices = generator.permutation(len(train_data))[:max_examples]
    summaries = []
    valid_tokens = 0
    missing_examples = 0
    examined_examples = 0

    with torch.inference_mode():
        for index in candidate_indices:
            input_ids, attention_mask = _dataset_item_inputs(train_data, index)
            gaze_batch = model.gaze_provider.compute(input_ids, attention_mask)
            summary, has_gaze = residual.summarize_token_features(
                gaze_batch.features,
                gaze_batch.valid_mask,
            )
            token_count = int(gaze_batch.valid_mask.sum().item())
            examined_examples += 1
            if bool(has_gaze.item()):
                summaries.append(summary.squeeze(0).cpu().numpy())
                valid_tokens += token_count
            else:
                missing_examples += 1
            if valid_tokens >= max_tokens:
                break

    if not summaries:
        raise RuntimeError("No valid gaze summaries were found in the training fold.")
    return np.stack(summaries, axis=0), {
        "examined_examples": examined_examples,
        "fitted_examples": len(summaries),
        "missing_gaze_examples": missing_examples,
        "valid_gaze_tokens": valid_tokens,
        "max_examples": max_examples,
        "max_tokens": max_tokens,
        "selection_seed": int(random_state),
    }


def fit_train_fold_gmm_residual(
    model,
    train_data,
    fold_id,
    output_dir,
    random_state=42,
    max_examples=5000,
    max_tokens=50000,
    reg_covar=1e-4,
    n_init=5,
):
    """Fit, install, and persist one fixed GMM using only a fold's train split."""

    summaries, collection = collect_train_fold_gaze_summaries(
        model=model,
        train_data=train_data,
        max_examples=max_examples,
        max_tokens=max_tokens,
        random_state=random_state,
    )
    fit = model.gmm_residual.fit_from_numpy(
        summaries,
        random_state=int(random_state),
        reg_covar=float(reg_covar),
        n_init=int(n_init),
    )
    diagnostics = {
        "method": "train-fold-fixed-one-dimensional-effort-gmm-arousal-residual",
        "summary_transform": "masked-mean-nonnegative-log1p-all-five-gaze",
        "fold_id": int(fold_id),
        "residual_mode": model.gmm_residual.mode,
        "trainable_coefficients": int(model.gmm_residual.correction.weight.numel()),
        "collection": collection,
        "fit": fit.to_dict(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / f"gmm_fit_fold{int(fold_id)}.json"
    temporary_path = diagnostics_path.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(diagnostics, output_file, indent=2, sort_keys=True)
    temporary_path.replace(diagnostics_path)
    return diagnostics
