"""Canonical VA regression loss names shared by the CLI and fold runner."""


HETEROSCEDASTIC_LOSSES = frozenset({"hetero", "hetero+ccc"})

LOSS_CHOICES = (
    "mse",
    "ccc",
    "robust",
    "mse+ccc",
    "robust+ccc",
    *sorted(HETEROSCEDASTIC_LOSSES),
)
