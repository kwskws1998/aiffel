import json
import math
from pathlib import Path

import numpy as np
import robust_loss_pytorch
import torch
from transformers import Trainer

from va_gaze.train.model_selection import best_model_greater_is_better


HETEROSCEDASTIC_CONFIG_FILENAME = "heteroscedastic_config.json"


def _pop_labels(inputs):
    labels = inputs["labels"]
    model_inputs = dict(inputs)
    model_inputs.pop("labels", None)
    return labels, model_inputs


def _add_model_auxiliary_loss(task_loss, outputs):
    auxiliary_loss = outputs.get("loss")
    if auxiliary_loss is None:
        return task_loss
    if auxiliary_loss.ndim > 0:
        auxiliary_loss = auxiliary_loss.mean()
    return task_loss + auxiliary_loss


def _save_heteroscedastic_config(model, output_dir, training_args=None):
    """Persist the effective uncertainty objective for every model class."""

    config = getattr(model, "config", None)
    output_names = getattr(config, "heteroscedastic_outputs", None)
    target_path = Path(output_dir) / HETEROSCEDASTIC_CONFIG_FILENAME
    if not output_names:
        if target_path.is_file():
            target_path.unlink()
        return
    if training_args is None:
        selection_metric = getattr(config, "checkpoint_selection_metric", None)
        selection_enabled = bool(
            getattr(config, "checkpoint_selection_enabled", False)
        )
        selection_greater = getattr(
            config,
            "checkpoint_greater_is_better",
            None,
        )
    else:
        selection_metric = getattr(training_args, "metric_for_best_model", None)
        save_strategy = getattr(training_args, "save_strategy", None)
        save_strategy = getattr(save_strategy, "value", save_strategy)
        selection_enabled = bool(
            getattr(training_args, "load_best_model_at_end", False)
            and selection_metric is not None
            and save_strategy != "no"
        )
        selection_greater = getattr(training_args, "greater_is_better", None)
    if selection_metric is not None:
        selection_metric = str(selection_metric)
        if selection_greater is None:
            selection_greater = best_model_greater_is_better(selection_metric)
    payload = {
        "schema_version": 2,
        "loss_function": str(config.loss_function),
        "num_labels": int(config.num_labels),
        "hetero_mse_weight": float(config.hetero_mse_weight),
        "hetero_ccc_weight": float(config.hetero_ccc_weight),
        "hetero_logvar_min": float(config.hetero_logvar_min),
        "hetero_logvar_max": float(config.hetero_logvar_max),
        "heteroscedastic_outputs": list(output_names),
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_greater_is_better": (
            None if selection_greater is None else bool(selection_greater)
        ),
        "checkpoint_selection_enabled": selection_enabled,
    }
    with open(target_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _ccc_loss(logits, labels, eps=1e-8):
    """Return a finite mean CCC loss over valid VA target dimensions."""

    if logits.ndim != 2 or labels.ndim != 2:
        raise ValueError("CCC loss requires 2D prediction and label tensors.")
    if logits.shape != labels.shape:
        raise ValueError(
            f"CCC prediction and label shapes must match, got {logits.shape} and {labels.shape}."
        )
    if logits.shape[1] != 2:
        raise ValueError(f"CCC loss requires exactly two VA columns, got {logits.shape[1]}.")

    working_dtype = (
        torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    )
    predictions = logits.to(dtype=working_dtype)
    targets = labels.to(dtype=working_dtype)
    prediction_mean = predictions.mean(dim=0)
    target_mean = targets.mean(dim=0)
    prediction_centered = predictions - prediction_mean
    target_centered = targets - target_mean
    correction_denominator = max(int(predictions.shape[0]) - 1, 1)
    prediction_variance = prediction_centered.square().sum(dim=0) / correction_denominator
    target_variance = target_centered.square().sum(dim=0) / correction_denominator
    covariance = (prediction_centered * target_centered).sum(dim=0) / correction_denominator
    denominator = (
        prediction_variance
        + target_variance
        + torch.square(prediction_mean - target_mean)
    )
    ccc = 2.0 * covariance / denominator.clamp_min(float(eps))
    ccc = torch.clamp(ccc, min=-1.0, max=1.0)
    valid_dimensions = target_variance > float(eps)
    if not torch.any(valid_dimensions):
        return predictions.sum() * 0.0
    return (1.0 - ccc[valid_dimensions]).mean()


def _build_adaptive_loss(num_dims=2):
    adaptive_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=num_dims,
        float_dtype=np.float32,
        device=adaptive_device,
    )


def _attach_adaptive_params(optimizer, adaptive):
    adaptive_params = [p for p in adaptive.parameters() if p.requires_grad]
    if not adaptive_params:
        return optimizer

    existing_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    new_params = [param for param in adaptive_params if id(param) not in existing_param_ids]
    if new_params:
        optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})
    return optimizer


def _robust_loss(adaptive, logits, labels):
    if next(adaptive.parameters()).device != logits.device:
        adaptive.to(logits.device)
    residual = labels - logits
    return torch.mean(adaptive.lossfun(residual))


def _split_heteroscedastic_logits(logits):
    if logits.ndim != 2 or logits.shape[-1] != 4:
        raise ValueError(
            "Heteroscedastic loss requires 2D model logits with exactly 4 columns: "
            "valence_mu, arousal_mu, valence_logvar, arousal_logvar."
        )
    return logits[:, :2], logits[:, 2:4]


def _validate_heteroscedastic_parameters(
    mse_weight,
    ccc_weight,
    logvar_min,
    logvar_max,
):
    """Validate direct API use of heteroscedastic loss hyperparameters."""

    values = {
        "mse_weight": mse_weight,
        "ccc_weight": ccc_weight,
        "logvar_min": logvar_min,
        "logvar_max": logvar_max,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if float(mse_weight) < 0 or float(ccc_weight) < 0:
        raise ValueError("Heteroscedastic MSE and CCC weights must be non-negative.")
    if float(logvar_min) >= float(logvar_max):
        raise ValueError("logvar_min must be smaller than logvar_max.")


class VARegressionTrainer(Trainer):
    """Shared trainer base that keeps custom gaze models reloadable at every save."""

    def __init__(self, *args, gaze_learning_rate=None, **kwargs):
        self.gaze_learning_rate = gaze_learning_rate
        self._gaze_learning_rate_applied = False
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Place an explicitly exposed gaze residual in its own learning-rate group."""

        optimizer = super().create_optimizer()
        if self._gaze_learning_rate_applied or self.gaze_learning_rate is None:
            return optimizer

        model = self.accelerator.unwrap_model(self.model)
        parameter_getter = getattr(model, "gaze_residual_parameters", None)
        residual_parameters = list(parameter_getter()) if callable(parameter_getter) else []
        residual_parameter_ids = {
            id(parameter) for parameter in residual_parameters if parameter.requires_grad
        }
        if not residual_parameter_ids:
            self._gaze_learning_rate_applied = True
            return optimizer

        moved_parameters = []
        retained_groups = []
        for group in optimizer.param_groups:
            retained_parameters = []
            for parameter in group["params"]:
                if id(parameter) in residual_parameter_ids:
                    moved_parameters.append(parameter)
                else:
                    retained_parameters.append(parameter)
            if retained_parameters:
                group["params"] = retained_parameters
                retained_groups.append(group)
        optimizer.param_groups[:] = retained_groups
        optimizer.add_param_group(
            {
                "params": moved_parameters,
                "lr": float(self.gaze_learning_rate),
                "weight_decay": 0.0,
            }
        )
        self._gaze_learning_rate_applied = True
        return optimizer

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        target_dir = output_dir or self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        _save_heteroscedastic_config(model, target_dir, training_args=self.args)
        save_manifest = getattr(model, "save_architecture_manifest", None)
        if callable(save_manifest):
            save_manifest(target_dir)


def _heteroscedastic_loss(
    logits,
    labels,
    mse_weight=0.1,
    ccc_weight=0.0,
    logvar_min=-5.0,
    logvar_max=3.0,
):
    """Combine Gaussian NLL with optional point and concordance anchors."""

    _validate_heteroscedastic_parameters(
        mse_weight=mse_weight,
        ccc_weight=ccc_weight,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    mu, logvar = _split_heteroscedastic_logits(logits)
    if labels.shape != mu.shape:
        raise ValueError(
            f"Heteroscedastic labels must have shape {mu.shape}, got {labels.shape}."
        )
    logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
    squared_error = torch.square(labels - mu)
    nll = 0.5 * torch.exp(-logvar) * squared_error + 0.5 * logvar
    mse_anchor = torch.nn.functional.mse_loss(mu, labels)
    loss = nll.mean() + float(mse_weight) * mse_anchor
    if float(ccc_weight) > 0:
        loss = loss + float(ccc_weight) * _ccc_loss(mu, labels)
    return loss


class CustomTrainerMSE(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerCCC(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _ccc_loss(logits, labels)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerMSE_CCC(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        mse_loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        ccc_loss = _ccc_loss(logits, labels)
        loss = 0.5 * (mse_loss + ccc_loss)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobust(VARegressionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _robust_loss(self.adaptive, logits, labels)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobustCCC(VARegressionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        robust = _robust_loss(self.adaptive, logits, labels)
        ccc = _ccc_loss(logits, labels)
        loss = 0.5 * (robust + ccc)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerHeteroscedastic(VARegressionTrainer):
    def __init__(
        self,
        *args,
        hetero_mse_weight=0.1,
        hetero_ccc_weight=0.0,
        hetero_logvar_min=-5.0,
        hetero_logvar_max=3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hetero_mse_weight = hetero_mse_weight
        self.hetero_ccc_weight = hetero_ccc_weight
        self.hetero_logvar_min = hetero_logvar_min
        self.hetero_logvar_max = hetero_logvar_max

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _heteroscedastic_loss(
            logits,
            labels,
            mse_weight=self.hetero_mse_weight,
            ccc_weight=self.hetero_ccc_weight,
            logvar_min=self.hetero_logvar_min,
            logvar_max=self.hetero_logvar_max,
        )
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss
