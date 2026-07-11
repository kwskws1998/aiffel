"""Shared gaze-feature schema and feature-selection validation."""

FEATURE_NAMES = ("nFix", "FFD", "GPT", "TRT", "fixProp")
TRT_INDEX = FEATURE_NAMES.index("TRT")
ALL_FEATURE_INDICES = frozenset(range(len(FEATURE_NAMES)))
TRT_ONLY_FEATURE_INDICES = frozenset({TRT_INDEX})


def select_feature_indices(features_used, supported_indices=None, model_label="ET predictor"):
    """Return enabled schema indices after checking predictor feature support."""

    flags = features_used or [1] * len(FEATURE_NAMES)
    if len(flags) != len(FEATURE_NAMES):
        raise ValueError(
            f"features_used must contain exactly {len(FEATURE_NAMES)} values "
            f"({','.join(FEATURE_NAMES)})."
        )

    selected = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
    if not selected:
        raise ValueError("features_used must enable at least one ET feature.")

    supported = ALL_FEATURE_INDICES if supported_indices is None else frozenset(supported_indices)
    unsupported = [FEATURE_NAMES[idx] for idx in selected if idx not in supported]
    if unsupported:
        supported_names = [FEATURE_NAMES[idx] for idx in sorted(supported)]
        raise ValueError(
            f"{model_label} does not predict: {', '.join(unsupported)}. "
            f"Supported features: {', '.join(supported_names)}."
        )
    return selected
