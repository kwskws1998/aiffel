"""Shared point-prediction and uncertainty metrics for VA regression."""

import math

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


VA_DIMENSIONS = ("valence", "arousal")


def _validate_metric_arrays(labels, predictions):
    """Return finite two-dimensional label and prediction arrays."""

    label_array = np.asarray(labels, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if label_array.ndim != 2 or label_array.shape[1] != 2:
        raise ValueError(f"VA labels must have shape (n, 2), got {label_array.shape}.")
    if prediction_array.ndim != 2 or prediction_array.shape[1] not in (2, 4):
        raise ValueError(
            "VA predictions must have exactly two point columns or four "
            f"heteroscedastic columns, got {prediction_array.shape}."
        )
    if label_array.shape[0] != prediction_array.shape[0]:
        raise ValueError(
            "VA labels and predictions must contain the same number of samples."
        )
    if label_array.shape[0] == 0:
        raise ValueError("VA metrics require at least one sample.")
    if not np.isfinite(label_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("VA labels and predictions must contain only finite values.")
    return label_array, prediction_array


def safe_pearson_correlation(y_true, y_pred, eps=1e-12):
    """Return Pearson correlation or NaN when either sample is degenerate."""

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2 or y_pred.size != y_true.size:
        return np.nan
    if np.std(y_true) <= float(eps) or np.std(y_pred) <= float(eps):
        return np.nan
    return float(stats.pearsonr(y_true, y_pred)[0])


def concordance_correlation_coefficient(y_true, y_pred, eps=1e-12):
    """Return Lin's sample CCC using the same correction as the training loss."""

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2 or y_pred.size != y_true.size:
        return np.nan
    true_centered = y_true - np.mean(y_true)
    pred_centered = y_pred - np.mean(y_pred)
    correction_denominator = y_true.size - 1
    true_variance = float(np.sum(np.square(true_centered)) / correction_denominator)
    pred_variance = float(np.sum(np.square(pred_centered)) / correction_denominator)
    covariance = float(np.sum(true_centered * pred_centered) / correction_denominator)
    denominator = (
        true_variance
        + pred_variance
        + float(np.square(np.mean(y_true) - np.mean(y_pred)))
    )
    if denominator <= float(eps):
        return np.nan
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))


def effective_logvars(raw_logvars, logvar_min=-5.0, logvar_max=3.0):
    """Apply the same finite log-variance bounds used by the training objective."""

    if not math.isfinite(float(logvar_min)) or not math.isfinite(float(logvar_max)):
        raise ValueError("logvar bounds must be finite.")
    if float(logvar_min) >= float(logvar_max):
        raise ValueError("logvar_min must be smaller than logvar_max.")
    raw_array = np.asarray(raw_logvars, dtype=np.float64)
    if not np.isfinite(raw_array).all():
        raise ValueError("Predicted log-variances must contain only finite values.")
    return np.clip(raw_array, float(logvar_min), float(logvar_max))


def _safe_spearman_correlation(first, second, eps=1e-12):
    """Return Spearman correlation or NaN for constant or undersized samples."""

    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or second.size != first.size:
        return np.nan
    if np.std(first) <= float(eps) or np.std(second) <= float(eps):
        return np.nan
    return float(stats.spearmanr(first, second).statistic)


def _finite_mean(values):
    """Average finite scalar values without emitting all-NaN warnings."""

    finite_values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else np.nan


def _uncertainty_metrics(y_true, y_pred, effective_logvar, dimension):
    """Calculate Gaussian calibration and uncertainty-error association metrics."""

    squared_error = np.square(np.asarray(y_true) - np.asarray(y_pred))
    variance = np.exp(np.asarray(effective_logvar, dtype=np.float64))
    standard_deviation = np.sqrt(variance)
    gaussian_nll = 0.5 * (
        math.log(2.0 * math.pi)
        + effective_logvar
        + squared_error / variance
    )
    mean_variance = float(np.mean(variance))
    return {
        f"mean_effective_logvar_{dimension}": float(np.mean(effective_logvar)),
        f"mean_variance_{dimension}": mean_variance,
        f"mean_stddev_{dimension}": float(np.mean(standard_deviation)),
        f"gaussian_nll_{dimension}": float(np.mean(gaussian_nll)),
        f"uncertainty_error_spearman_{dimension}": _safe_spearman_correlation(
            variance,
            squared_error,
        ),
        f"coverage_1sigma_{dimension}": float(
            np.mean(np.abs(y_true - y_pred) <= standard_deviation)
        ),
        f"coverage_2sigma_{dimension}": float(
            np.mean(np.abs(y_true - y_pred) <= 2.0 * standard_deviation)
        ),
        f"mse_to_mean_variance_ratio_{dimension}": float(
            np.mean(squared_error) / mean_variance
        ),
    }


def calculate_va_metrics(
    labels,
    predictions,
    logvar_min=-5.0,
    logvar_max=3.0,
):
    """Calculate VA point metrics and optional heteroscedastic calibration metrics."""

    label_array, prediction_array = _validate_metric_arrays(labels, predictions)
    mean_predictions = prediction_array[:, :2]
    metrics = {}
    ccc_values = []
    pearson_values = []
    mse_values = []
    for index, dimension in enumerate(VA_DIMENSIONS):
        y_true = label_array[:, index]
        y_pred = mean_predictions[:, index]
        mse = float(mean_squared_error(y_true, y_pred))
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        pearson = safe_pearson_correlation(y_true, y_pred)
        metrics[f"mse_{dimension}"] = mse
        metrics[f"rmse_{dimension}"] = float(np.sqrt(mse))
        metrics[f"mae_{dimension}"] = float(mean_absolute_error(y_true, y_pred))
        metrics[f"pearson_corr_{dimension}"] = pearson
        metrics[f"ccc_{dimension}"] = ccc
        mse_values.append(mse)
        pearson_values.append(pearson)
        ccc_values.append(ccc)
    metrics["mse_mean"] = _finite_mean(mse_values)
    metrics["pearson_corr_mean"] = _finite_mean(pearson_values)
    metrics["ccc_mean"] = _finite_mean(ccc_values)

    if prediction_array.shape[1] >= 4:
        raw_logvars = prediction_array[:, 2:4]
        effective = effective_logvars(
            raw_logvars,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        uncertainty_values = {
            "gaussian_nll": [],
            "uncertainty_error_spearman": [],
            "coverage_1sigma": [],
            "coverage_2sigma": [],
        }
        for index, dimension in enumerate(VA_DIMENSIONS):
            mean_raw_logvar = float(np.mean(raw_logvars[:, index]))
            metrics[f"mean_logvar_{dimension}"] = mean_raw_logvar
            metrics[f"mean_raw_logvar_{dimension}"] = mean_raw_logvar
            metrics[f"logvar_lower_clamp_rate_{dimension}"] = float(
                np.mean(raw_logvars[:, index] < float(logvar_min))
            )
            metrics[f"logvar_upper_clamp_rate_{dimension}"] = float(
                np.mean(raw_logvars[:, index] > float(logvar_max))
            )
            dimension_metrics = _uncertainty_metrics(
                label_array[:, index],
                mean_predictions[:, index],
                effective[:, index],
                dimension,
            )
            metrics.update(dimension_metrics)
            for name in uncertainty_values:
                uncertainty_values[name].append(dimension_metrics[f"{name}_{dimension}"])
        for name, values in uncertainty_values.items():
            metrics[f"{name}_mean"] = _finite_mean(values)
    return metrics


def compute_metrics(
    eval_pred,
    logvar_min=-5.0,
    logvar_max=3.0,
    metric_for_best_model=None,
):
    """Transformers Trainer callback for shared VA metrics."""

    predictions, labels = eval_pred
    metrics = calculate_va_metrics(
        labels,
        predictions,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    if metric_for_best_model is not None and metric_for_best_model != "loss":
        if metric_for_best_model not in metrics:
            raise ValueError(
                "The selected checkpoint metric is unavailable for these model "
                f"outputs: {metric_for_best_model}."
            )
        if not np.isfinite(metrics[metric_for_best_model]):
            raise ValueError(
                "The selected checkpoint metric is non-finite on the validation "
                f"set: {metric_for_best_model}."
            )
    return metrics
