"""Fixed one-dimensional effort-GMM features for an arousal residual."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_GMM_RESIDUAL_MODES = ("posterior", "component-linear")


def _as_finite_2d_array(features, name="features"):
    """Return a validated finite two-dimensional float64 NumPy array."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features]; got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _component_linear_basis_numpy(standardized_features, responsibilities):
    """Build mixture-of-linear-experts basis values for NumPy inputs."""

    intercepts = responsibilities[..., None]
    slopes = responsibilities[..., None] * standardized_features[:, None, :]
    return np.concatenate((intercepts, slopes), axis=-1).reshape(
        standardized_features.shape[0],
        -1,
    )


@dataclass(frozen=True)
class DiagonalGMMFit:
    """Serializable arrays and diagnostics for a fold-local effort GMM."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    mixture_weights: np.ndarray
    effort_component_means: np.ndarray
    effort_component_variances: np.ndarray
    posterior_center: np.ndarray
    component_linear_center: np.ndarray
    component_order: np.ndarray
    sample_count: int
    converged: bool
    n_iter: int
    lower_bound: float
    random_state: int

    @property
    def feature_dim(self):
        """Return the fitted input feature dimension."""

        return int(self.feature_center.shape[0])

    @property
    def n_components(self):
        """Return the fitted mixture component count."""

        return int(self.mixture_weights.shape[0])

    def to_dict(self):
        """Return JSON-compatible fit parameters and convergence diagnostics."""

        return {
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "feature_scaler": "median-iqr",
            "gmm_input": "mean-standardized-all-five-reading-effort",
            "mixture_weights": self.mixture_weights.tolist(),
            "effort_component_means": self.effort_component_means.tolist(),
            "effort_component_variances": self.effort_component_variances.tolist(),
            "posterior_center": self.posterior_center.tolist(),
            "component_soft_counts": (
                self.posterior_center * float(self.sample_count)
            ).tolist(),
            "component_linear_center": self.component_linear_center.tolist(),
            "component_order": self.component_order.tolist(),
            "sample_count": int(self.sample_count),
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
            "lower_bound": float(self.lower_bound),
            "random_state": int(self.random_state),
        }


def fit_diagonal_gmm(
    example_summaries,
    n_components=3,
    random_state=42,
    reg_covar=1e-4,
    n_init=5,
    max_iter=200,
    init_params="k-means++",
):
    """Fit a robust-scaled one-dimensional effort GMM on a train fold only.

    ``example_summaries`` must already use the same feature transformation that
    will be supplied to the residual module at inference time. For gaze tokens,
    ``GmmArousalLogitResidual.summarize_token_features`` produces the intended
    mean nonnegative-log1p summary. All features remain available to the linear
    experts; only the unsupervised GMM gate is reduced to their standardized mean.
    """

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler

    summaries = _as_finite_2d_array(example_summaries, "example_summaries")
    n_components = int(n_components)
    random_state = int(random_state)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if summaries.shape[0] < n_components:
        raise ValueError(
            "The GMM needs at least as many examples as components: "
            f"{summaries.shape[0]} < {n_components}."
        )
    if float(reg_covar) <= 0:
        raise ValueError("reg_covar must be > 0.")
    if int(n_init) <= 0:
        raise ValueError("n_init must be > 0.")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be > 0.")
    if str(init_params) not in ("kmeans", "k-means++", "random", "random_from_data"):
        raise ValueError(
            "init_params must be one of kmeans, k-means++, random, or random_from_data."
        )

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    standardized = scaler.fit_transform(summaries)
    feature_scale = np.asarray(scaler.scale_, dtype=np.float64)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    effort = standardized.mean(axis=1, keepdims=True)

    mixture = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=float(reg_covar),
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=random_state,
        init_params=str(init_params),
    )
    mixture.fit(effort)

    order = np.argsort(mixture.means_[:, 0], kind="stable")
    responsibilities = mixture.predict_proba(effort)[:, order]
    effort_component_means = np.asarray(mixture.means_, dtype=np.float64)[order]
    effort_component_variances = np.asarray(
        mixture.covariances_, dtype=np.float64
    )[order]
    mixture_weights = np.asarray(mixture.weights_, dtype=np.float64)[order]
    posterior_center = responsibilities.mean(axis=0)
    component_linear = _component_linear_basis_numpy(
        standardized,
        responsibilities,
    )

    return DiagonalGMMFit(
        feature_center=np.asarray(scaler.center_, dtype=np.float64).copy(),
        feature_scale=feature_scale.copy(),
        mixture_weights=mixture_weights.copy(),
        effort_component_means=effort_component_means.copy(),
        effort_component_variances=effort_component_variances.copy(),
        posterior_center=posterior_center.copy(),
        component_linear_center=component_linear.mean(axis=0).copy(),
        component_order=np.asarray(order, dtype=np.int64).copy(),
        sample_count=int(summaries.shape[0]),
        converged=bool(mixture.converged_),
        n_iter=int(mixture.n_iter_),
        lower_bound=float(mixture.lower_bound_),
        random_state=random_state,
    )


class GmmArousalLogitResidual(nn.Module):
    """Add a zero-initialized fixed effort-GMM correction to one raw logit.

    ``posterior`` mode learns ``K - 1`` coefficients over centered component
    responsibilities. ``component-linear`` mode learns ``K * (D + 1)``
    coefficients over centered ``[r_k, r_k * z]`` mixture-expert features.
    Its ``K = 1`` case is the five-coefficient standardized raw-feature control.
    The GMM gate sees only the mean robust-scaled reading-effort axis while the
    linear experts retain every feature. Scaler and GMM parameters are frozen;
    only correction coefficients receive task-loss gradients.
    """

    def __init__(
        self,
        feature_dim=5,
        n_components=3,
        mode="posterior",
        arousal_index=1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.n_components = int(n_components)
        self.mode = str(mode)
        self.arousal_index = int(arousal_index)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0.")
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if self.mode not in SUPPORTED_GMM_RESIDUAL_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_GMM_RESIDUAL_MODES}; got {self.mode!r}."
            )
        if self.mode == "posterior" and self.n_components < 2:
            raise ValueError("posterior mode requires n_components >= 2.")
        if self.arousal_index < 0:
            raise ValueError("arousal_index must be >= 0.")

        self.register_buffer("feature_center", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.register_buffer(
            "mixture_weights",
            torch.full((self.n_components,), 1.0 / self.n_components),
        )
        self.register_buffer(
            "effort_component_means",
            torch.zeros(self.n_components, 1),
        )
        self.register_buffer(
            "effort_component_variances",
            torch.ones(self.n_components, 1),
        )
        self.register_buffer("posterior_center", torch.zeros(self.n_components))
        self.register_buffer(
            "component_linear_center",
            torch.zeros(self.n_components * (self.feature_dim + 1)),
        )
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

        if self.mode == "posterior":
            basis_dim = self.n_components - 1
        elif self.n_components == 1:
            basis_dim = self.feature_dim
        else:
            basis_dim = self.n_components * (self.feature_dim + 1)
        self.correction = nn.Linear(basis_dim, 1, bias=False)
        nn.init.zeros_(self.correction.weight)

    @staticmethod
    def summarize_token_features(token_features, valid_mask):
        """Return mean nonnegative-log1p word features and a gaze flag."""

        if token_features.ndim != 3:
            raise ValueError(
                "token_features must have shape [batch, tokens, features]; got "
                f"{tuple(token_features.shape)}."
            )
        if valid_mask.ndim != 2 or valid_mask.shape != token_features.shape[:2]:
            raise ValueError(
                "valid_mask must have shape [batch, tokens] matching token_features."
            )
        if not torch.isfinite(token_features).all():
            raise ValueError("token_features contains NaN or infinite values.")
        if not token_features.is_floating_point():
            token_features = token_features.to(dtype=torch.float32)

        valid = valid_mask.to(device=token_features.device, dtype=torch.bool)
        transformed = torch.log1p(token_features.clamp_min(0.0))
        valid_float = valid.unsqueeze(-1).to(dtype=transformed.dtype)
        count = valid_float.sum(dim=1).clamp_min(1.0)
        summaries = (transformed * valid_float).sum(dim=1) / count
        has_gaze = valid.any(dim=1)
        summaries = summaries * has_gaze.unsqueeze(-1).to(dtype=summaries.dtype)
        return summaries, has_gaze

    def set_fit(self, fit, reset_correction=True):
        """Copy a fitted scaler and diagonal GMM into frozen module buffers."""

        if not isinstance(fit, DiagonalGMMFit):
            raise TypeError("fit must be a DiagonalGMMFit instance.")
        if fit.feature_dim != self.feature_dim:
            raise ValueError(
                f"Fit feature dimension {fit.feature_dim} != module dimension {self.feature_dim}."
            )
        if fit.n_components != self.n_components:
            raise ValueError(
                f"Fit components {fit.n_components} != module components {self.n_components}."
            )

        values = {
            "feature_center": fit.feature_center,
            "feature_scale": fit.feature_scale,
            "mixture_weights": fit.mixture_weights,
            "effort_component_means": fit.effort_component_means,
            "effort_component_variances": fit.effort_component_variances,
            "posterior_center": fit.posterior_center,
            "component_linear_center": fit.component_linear_center,
        }
        tensors = {}
        for name, value in values.items():
            buffer = getattr(self, name)
            tensor = torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype)
            if tensor.shape != buffer.shape:
                raise ValueError(
                    f"Fit value {name} has shape {tuple(tensor.shape)}; "
                    f"expected {tuple(buffer.shape)}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Fit value {name} contains NaN or infinite values.")
            tensors[name] = tensor
        if tensors["feature_scale"].le(0).any():
            raise ValueError("Fit feature_scale values must be > 0.")
        if tensors["effort_component_variances"].le(0).any():
            raise ValueError("Fit effort_component_variances values must be > 0.")
        if tensors["mixture_weights"].le(0).any() or not torch.isclose(
            tensors["mixture_weights"].sum(),
            tensors["mixture_weights"].new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Fit mixture_weights must be positive and sum to 1.")
        if tensors["posterior_center"].lt(0).any() or not torch.isclose(
            tensors["posterior_center"].sum(),
            tensors["posterior_center"].new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Fit posterior_center must be nonnegative and sum to 1.")

        with torch.no_grad():
            for name, tensor in tensors.items():
                buffer = getattr(self, name)
                buffer.copy_(tensor)
            self.is_fitted.fill_(True)
            if reset_correction:
                self.correction.weight.zero_()
        return self

    def fit_from_numpy(self, example_summaries, **fit_kwargs):
        """Fit from train-fold NumPy summaries and load the resulting buffers."""

        fit = fit_diagonal_gmm(
            example_summaries,
            n_components=self.n_components,
            **fit_kwargs,
        )
        self.set_fit(fit)
        return fit

    def _validate_summaries(self, example_summaries):
        """Validate and place example summaries on the fitted buffer device."""

        if not bool(self.is_fitted.item()):
            raise RuntimeError("The GMM residual must be fitted before use.")
        if example_summaries.ndim != 2:
            raise ValueError(
                "example_summaries must have shape [batch, features]; got "
                f"{tuple(example_summaries.shape)}."
            )
        if example_summaries.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} summary features; got "
                f"{example_summaries.shape[-1]}."
            )
        if not torch.isfinite(example_summaries).all():
            raise ValueError("example_summaries contains NaN or infinite values.")
        return example_summaries.to(
            device=self.feature_center.device,
            dtype=self.feature_center.dtype,
        )

    def standardize(self, example_summaries):
        """Standardize summaries with train-fold frozen feature statistics."""

        summaries = self._validate_summaries(example_summaries)
        scale = self.feature_scale.clamp_min(torch.finfo(summaries.dtype).eps)
        return (summaries - self.feature_center) / scale

    def _posterior_from_standardized(self, standardized):
        """Return GMM responsibilities for already-standardized summaries."""

        effort = standardized.mean(dim=-1, keepdim=True)
        variances = self.effort_component_variances.clamp_min(
            torch.finfo(effort.dtype).eps
        )
        difference = effort.unsqueeze(1) - self.effort_component_means.unsqueeze(0)
        log_prob = -0.5 * (
            torch.square(difference) / variances.unsqueeze(0)
            + variances.log().unsqueeze(0)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_joint = self.mixture_weights.clamp_min(
            torch.finfo(standardized.dtype).eps
        ).log().unsqueeze(0) + log_prob
        return torch.softmax(log_joint, dim=-1)

    def posterior(self, example_summaries):
        """Return diagonal-GMM posterior responsibilities for each example."""

        standardized = self.standardize(example_summaries)
        return self._posterior_from_standardized(standardized)

    def design_matrix(self, example_summaries):
        """Return the centered posterior or component-linear correction basis."""

        standardized = self.standardize(example_summaries)
        responsibilities = self._posterior_from_standardized(standardized)
        if self.mode == "posterior":
            return (
                responsibilities[:, :-1]
                - self.posterior_center[:-1].unsqueeze(0)
            )
        if self.n_components == 1:
            return standardized - self.component_linear_center[1:].unsqueeze(0)

        intercepts = responsibilities.unsqueeze(-1)
        slopes = responsibilities.unsqueeze(-1) * standardized.unsqueeze(1)
        basis = torch.cat((intercepts, slopes), dim=-1).flatten(start_dim=1)
        return basis - self.component_linear_center.unsqueeze(0)

    def compute_residual(self, example_summaries, has_gaze=None):
        """Compute a scalar raw-logit correction and mask gaze-missing examples."""

        basis = self.design_matrix(example_summaries)
        residual = self.correction(basis).squeeze(-1)
        if has_gaze is not None:
            if has_gaze.ndim != 1 or has_gaze.shape[0] != residual.shape[0]:
                raise ValueError("has_gaze must have shape [batch].")
            residual = residual * has_gaze.to(
                device=residual.device,
                dtype=residual.dtype,
            )
        return residual

    def forward(self, raw_logits, example_summaries, has_gaze=None):
        """Return raw logits with the fixed-GMM correction applied to arousal."""

        if raw_logits.ndim != 2:
            raise ValueError(
                f"raw_logits must have shape [batch, outputs]; got {tuple(raw_logits.shape)}."
            )
        if self.arousal_index >= raw_logits.shape[-1]:
            raise ValueError(
                f"arousal_index {self.arousal_index} is outside {raw_logits.shape[-1]} outputs."
            )
        if example_summaries.ndim != 2:
            raise ValueError(
                "example_summaries must have shape [batch, features]; got "
                f"{tuple(example_summaries.shape)}."
            )
        if raw_logits.shape[0] != example_summaries.shape[0]:
            raise ValueError("raw_logits and example_summaries must share the batch dimension.")

        residual = self.compute_residual(example_summaries, has_gaze=has_gaze)
        residual = residual.to(device=raw_logits.device, dtype=raw_logits.dtype)
        output_direction = torch.nn.functional.one_hot(
            torch.tensor(self.arousal_index, device=raw_logits.device),
            num_classes=raw_logits.shape[-1],
        ).to(dtype=raw_logits.dtype)
        return raw_logits + residual.unsqueeze(-1) * output_direction.unsqueeze(0)
