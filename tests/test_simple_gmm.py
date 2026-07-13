"""Unit tests for the fixed diagonal-GMM arousal residual."""

import unittest

import numpy as np
import torch

from va_gaze.models.gaze.simple_gmm import (
    GmmArousalLogitResidual,
    fit_diagonal_gmm,
)


def make_mixture_summaries(seed=7, samples_per_component=60):
    """Create deterministic five-feature summaries with three effort regimes."""

    rng = np.random.default_rng(seed)
    centers = np.asarray(
        [
            [-1.4, -0.8, -1.1, -1.8, -0.5],
            [0.0, 0.2, 0.1, 0.0, 0.3],
            [1.2, 0.9, 1.4, 2.0, 0.8],
        ],
        dtype=np.float64,
    )
    rows = [
        rng.normal(loc=center, scale=0.12, size=(samples_per_component, 5))
        for center in centers
    ]
    return np.concatenate(rows, axis=0)


class FitDiagonalGMMTest(unittest.TestCase):
    def setUp(self):
        self.summaries = make_mixture_summaries()

    def test_fit_is_deterministic_and_ordered_by_effort_center(self):
        first = fit_diagonal_gmm(
            self.summaries,
            n_components=3,
            random_state=19,
            n_init=3,
        )
        second = fit_diagonal_gmm(
            self.summaries,
            n_components=3,
            random_state=19,
            n_init=3,
        )

        np.testing.assert_allclose(first.feature_center, second.feature_center)
        np.testing.assert_allclose(
            first.effort_component_means,
            second.effort_component_means,
        )
        np.testing.assert_allclose(first.mixture_weights, second.mixture_weights)
        self.assertTrue(first.converged)
        self.assertEqual(first.sample_count, len(self.summaries))
        self.assertAlmostEqual(float(first.mixture_weights.sum()), 1.0, places=7)
        effort_centers = first.effort_component_means[:, 0]
        self.assertTrue(np.all(np.diff(effort_centers) > 0.0))

    def test_fit_rejects_invalid_arrays_and_component_counts(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_diagonal_gmm(np.zeros(5), n_components=2)
        invalid = self.summaries.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            fit_diagonal_gmm(invalid, n_components=2)
        with self.assertRaisesRegex(ValueError, "at least as many"):
            fit_diagonal_gmm(np.zeros((2, 5)), n_components=3)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            fit_diagonal_gmm(self.summaries, n_components=0)

    def test_fit_diagnostics_are_json_compatible(self):
        fitted = fit_diagonal_gmm(self.summaries, n_components=3, random_state=5)
        diagnostics = fitted.to_dict()
        self.assertEqual(diagnostics["sample_count"], len(self.summaries))
        self.assertEqual(len(diagnostics["mixture_weights"]), 3)
        self.assertIsInstance(diagnostics["converged"], bool)


class GmmArousalLogitResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.summaries = make_mixture_summaries()
        self.fit = fit_diagonal_gmm(
            self.summaries,
            n_components=3,
            random_state=23,
            n_init=3,
        )

    def _module(self, mode="posterior"):
        module = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=3,
            mode=mode,
        )
        module.set_fit(self.fit)
        return module

    def test_token_summary_is_masked_mean_nonnegative_log1p(self):
        token_features = torch.tensor(
            [
                [[1.0, 3.0], [8.0, -3.0], [100.0, 100.0]],
                [[5.0, 7.0], [9.0, 11.0], [13.0, 15.0]],
            ]
        )
        valid_mask = torch.tensor(
            [[True, True, False], [False, False, False]]
        )
        summaries, has_gaze = GmmArousalLogitResidual.summarize_token_features(
            token_features,
            valid_mask,
        )
        expected_first = torch.stack(
            (
                torch.log1p(torch.tensor([1.0, 8.0])).mean(),
                torch.log1p(torch.tensor([3.0, 0.0])).mean(),
            )
        )
        torch.testing.assert_close(summaries[0], expected_first)
        torch.testing.assert_close(summaries[1], torch.zeros(2))
        self.assertEqual(has_gaze.tolist(), [True, False])

    def test_posteriors_are_finite_and_sum_to_one(self):
        module = self._module()
        rows = torch.as_tensor(self.summaries[[0, 70, 140]], dtype=torch.float32)
        responsibilities = module.posterior(rows)
        self.assertEqual(tuple(responsibilities.shape), (3, 3))
        self.assertTrue(torch.isfinite(responsibilities).all())
        torch.testing.assert_close(
            responsibilities.sum(dim=-1),
            torch.ones(3),
        )
        self.assertEqual(responsibilities.argmax(dim=-1).tolist(), [0, 1, 2])

    def test_zero_initialization_is_exact_baseline_for_both_modes(self):
        raw_logits = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        rows = torch.as_tensor(self.summaries[[0, -1]], dtype=torch.float32)
        for mode in ("posterior", "component-linear"):
            with self.subTest(mode=mode):
                module = self._module(mode)
                output = module(raw_logits, rows, has_gaze=torch.tensor([True, True]))
                self.assertTrue(torch.equal(output, raw_logits))

    def test_posterior_mode_has_k_minus_one_parameters_and_changes_only_arousal(self):
        module = self._module("posterior")
        self.assertEqual(module.correction.weight.numel(), 2)
        with torch.no_grad():
            module.correction.weight.copy_(torch.tensor([[0.8, -0.4]]))

        raw_logits = torch.tensor([[0.2, 0.1], [0.2, 0.1]])
        rows = torch.as_tensor(self.summaries[[0, -1]], dtype=torch.float32)
        output = module(raw_logits, rows, has_gaze=torch.tensor([True, True]))
        torch.testing.assert_close(output[:, 0], raw_logits[:, 0])
        self.assertFalse(torch.equal(output[:, 1], raw_logits[:, 1]))
        self.assertNotEqual(output[0, 1].detach().item(), output[1, 1].detach().item())

    def test_missing_gaze_forces_exactly_zero_residual(self):
        module = self._module("posterior")
        with torch.no_grad():
            module.correction.weight.fill_(1.0)
        raw_logits = torch.tensor([[0.2, 0.1], [0.2, 0.1]])
        rows = torch.as_tensor(self.summaries[[0, -1]], dtype=torch.float32)
        output = module(raw_logits, rows, has_gaze=torch.tensor([False, True]))
        torch.testing.assert_close(output[0], raw_logits[0])
        self.assertNotEqual(output[1, 1].detach().item(), raw_logits[1, 1].item())

    def test_component_linear_mode_has_centered_k_times_d_plus_one_basis(self):
        module = self._module("component-linear")
        expected_dimension = 3 * (5 + 1)
        self.assertEqual(module.correction.weight.numel(), expected_dimension)
        rows = torch.as_tensor(self.summaries, dtype=torch.float32)
        basis = module.design_matrix(rows)
        self.assertEqual(tuple(basis.shape), (len(rows), expected_dimension))
        torch.testing.assert_close(
            basis.mean(dim=0),
            torch.zeros(expected_dimension),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_one_component_is_the_five_coefficient_raw_linear_control(self):
        fit = fit_diagonal_gmm(self.summaries, n_components=1, random_state=3)
        module = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=1,
            mode="component-linear",
        )
        module.set_fit(fit)
        self.assertEqual(module.correction.weight.numel(), 5)
        rows = torch.as_tensor(self.summaries, dtype=torch.float32)
        basis = module.design_matrix(rows)
        self.assertEqual(tuple(basis.shape), (len(rows), 5))
        torch.testing.assert_close(
            basis.mean(dim=0),
            torch.zeros(5),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_fit_from_numpy_and_state_dict_round_trip(self):
        source = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=3,
            mode="component-linear",
        )
        source.fit_from_numpy(
            self.summaries,
            random_state=31,
            n_init=2,
        )
        with torch.no_grad():
            source.correction.weight.copy_(
                torch.linspace(-0.2, 0.2, source.correction.weight.numel()).view(1, -1)
            )

        restored = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=3,
            mode="component-linear",
        )
        restored.load_state_dict(source.state_dict(), strict=True)
        raw_logits = torch.tensor([[0.0, 0.0], [0.5, -0.5]])
        rows = torch.as_tensor(self.summaries[[5, 125]], dtype=torch.float32)
        expected = source(raw_logits, rows)
        actual = restored(raw_logits, rows)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool(restored.is_fitted.item()))

    def test_unfitted_and_shape_errors_are_explicit(self):
        module = GmmArousalLogitResidual(feature_dim=5, n_components=3)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            module.posterior(torch.zeros(2, 5))

        module.set_fit(self.fit)
        with self.assertRaisesRegex(ValueError, "summary features"):
            module.posterior(torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            module(torch.zeros(3, 2), torch.zeros(2, 5))
        with self.assertRaisesRegex(ValueError, "has_gaze"):
            module.compute_residual(torch.zeros(2, 5), has_gaze=torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
