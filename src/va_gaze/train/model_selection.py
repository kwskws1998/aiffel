"""Checkpoint-selection metrics shared by the VA training CLI and fold runner."""


DEFAULT_BEST_MODEL_METRIC = "ccc_mean"

BEST_MODEL_METRIC_DIRECTIONS = {
    "loss": False,
    "mse_mean": False,
    "ccc_mean": True,
    "pearson_corr_mean": True,
    "gaussian_nll_mean": False,
}

BEST_MODEL_METRIC_CHOICES = tuple(BEST_MODEL_METRIC_DIRECTIONS)
HETEROSCEDASTIC_ONLY_BEST_MODEL_METRICS = frozenset({"gaussian_nll_mean"})


def best_model_greater_is_better(metric_name):
    """Return the optimization direction for one supported checkpoint metric."""

    try:
        return BEST_MODEL_METRIC_DIRECTIONS[metric_name]
    except KeyError as exc:
        choices = ", ".join(BEST_MODEL_METRIC_CHOICES)
        raise ValueError(
            f"Unknown metric_for_best_model: {metric_name}. Choose one of: {choices}."
        ) from exc


def validate_best_model_metric(metric_name, heteroscedastic):
    """Reject selection metrics that the requested model cannot emit."""

    best_model_greater_is_better(metric_name)
    if metric_name in HETEROSCEDASTIC_ONLY_BEST_MODEL_METRICS and not heteroscedastic:
        raise ValueError(
            f"metric_for_best_model={metric_name} requires a heteroscedastic loss."
        )
    return metric_name
