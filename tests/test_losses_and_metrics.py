"""Numerical contracts for VA concordance and heteroscedastic regression."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from va_gaze.cli.train_model import _build_parser, _validate_args
from va_gaze.eval.metrics import (
    calculate_va_metrics,
    compute_metrics,
    concordance_correlation_coefficient,
    effective_logvars,
    safe_pearson_correlation,
)
from va_gaze.eval.oof_reports import (
    _build_uncertainty_risk_coverage_rows,
    _equal_count_bin_ids,
    _least_uncertain_selections,
    _remove_stale_uncertainty_reports,
    create_prediction_tables,
)
from va_gaze.train.custom_trainer import _ccc_loss, _heteroscedastic_loss
from va_gaze.train.fold_runner import (
    HETEROSCEDASTIC_OUTPUTS,
    LOSS_TO_TRAINER,
    _attach_heteroscedastic_config,
    _build_trainer,
)
from va_gaze.train.loss_names import HETEROSCEDASTIC_LOSSES, LOSS_CHOICES


class CCCTrainingLossTest(unittest.TestCase):
    def test_perfect_predictions_have_zero_loss_and_finite_gradient(self):
        labels = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
            dtype=torch.float64,
        )
        predictions = labels.clone().requires_grad_(True)

        loss = _ccc_loss(predictions, labels)
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_mean_shift_matches_sample_ccc_definition(self):
        labels = torch.tensor(
            [[-1.0, -2.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        predictions = labels + 1.0

        loss = _ccc_loss(predictions, labels)

        self.assertAlmostEqual(loss.item(), 0.2, places=12)

    def test_degenerate_targets_return_differentiable_zero(self):
        cases = (
            (
                torch.tensor([[0.0, 1.0], [1.0, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 4.0], [3.0, 4.0]]),
            ),
            (
                torch.tensor([[0.0, 1.0]], requires_grad=True),
                torch.tensor([[3.0, 4.0]]),
            ),
        )
        for predictions, labels in cases:
            with self.subTest(batch_size=len(labels)):
                loss = _ccc_loss(predictions, labels)
                loss.backward()
                self.assertEqual(loss.item(), 0.0)
                self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_shape_contract_is_exactly_two_va_columns(self):
        with self.assertRaisesRegex(ValueError, "exactly two VA columns"):
            _ccc_loss(torch.zeros((2, 3)), torch.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            _ccc_loss(torch.zeros((2, 2)), torch.zeros((3, 2)))


class HeteroscedasticTrainingLossTest(unittest.TestCase):
    def test_gaussian_nll_mse_and_ccc_terms_match_closed_form(self):
        labels = torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]],
            dtype=torch.float64,
        )
        means = labels + 1.0
        logits = torch.cat((means, torch.zeros_like(means)), dim=1).requires_grad_(True)

        loss = _heteroscedastic_loss(
            logits,
            labels,
            mse_weight=0.1,
            ccc_weight=0.3,
        )
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.66, places=12)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_ccc_weight_preserves_hetero_loss_value(self):
        labels = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        means = labels + 1.0
        logits = torch.cat((means, torch.zeros_like(means)), dim=1)

        loss = _heteroscedastic_loss(
            logits,
            labels,
            mse_weight=0.1,
            ccc_weight=0.0,
        )

        self.assertAlmostEqual(loss.item(), 0.6, places=6)

    def test_logvar_is_clamped_before_gaussian_nll(self):
        labels = torch.zeros((2, 2), dtype=torch.float64)
        means = torch.ones_like(labels)
        raw_logvars = torch.tensor(
            [[-100.0, 100.0], [-100.0, 100.0]],
            dtype=torch.float64,
        )
        logits = torch.cat((means, raw_logvars), dim=1)

        loss = _heteroscedastic_loss(
            logits,
            labels,
            mse_weight=0.0,
            ccc_weight=0.0,
            logvar_min=-2.0,
            logvar_max=2.0,
        )
        expected = np.mean(
            [
                0.5 * math.exp(2.0) - 1.0,
                0.5 * math.exp(-2.0) + 1.0,
            ]
        )

        self.assertAlmostEqual(loss.item(), expected, places=12)

    def test_invalid_hyperparameters_and_shapes_are_rejected(self):
        logits = torch.zeros((2, 4))
        labels = torch.zeros((2, 2))
        invalid_kwargs = (
            {"mse_weight": -0.1},
            {"ccc_weight": -0.1},
            {"logvar_min": float("nan")},
            {"logvar_max": float("inf")},
            {"logvar_min": 2.0, "logvar_max": 2.0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _heteroscedastic_loss(logits, labels, **kwargs)

        with self.assertRaisesRegex(ValueError, "exactly 4 columns"):
            _heteroscedastic_loss(torch.zeros((2, 5)), labels)
        with self.assertRaisesRegex(ValueError, "labels must have shape"):
            _heteroscedastic_loss(logits, torch.zeros((2, 3)))


class HeteroscedasticWiringTest(unittest.TestCase):
    def test_loss_registry_and_cli_accept_hetero_ccc_settings(self):
        self.assertEqual(set(LOSS_CHOICES), set(LOSS_TO_TRAINER))
        self.assertEqual(HETEROSCEDASTIC_LOSSES, {"hetero", "hetero+ccc"})

        parser = _build_parser()
        args = parser.parse_args(
            [
                "distilbert",
                "hetero+ccc",
                "--hetero-mse-weight",
                "0.2",
                "--hetero-ccc-weight",
                "0.4",
                "--hetero-logvar-min",
                "-4",
                "--hetero-logvar-max",
                "2",
            ]
        )
        _validate_args(parser, args)

        self.assertEqual(args.loss, "hetero+ccc")
        self.assertEqual(args.hetero_mse_weight, 0.2)
        self.assertEqual(args.hetero_ccc_weight, 0.4)
        self.assertEqual(args.hetero_logvar_min, -4.0)
        self.assertEqual(args.hetero_logvar_max, 2.0)

    def test_cli_rejects_nonfinite_negative_and_reversed_settings(self):
        invalid_options = (
            ("--hetero-mse-weight", "-0.1"),
            ("--hetero-ccc-weight", "-0.1"),
            ("--hetero-mse-weight", "nan"),
            ("--hetero-ccc-weight", "inf"),
            ("--hetero-logvar-min", "nan"),
        )
        for option, value in invalid_options:
            with self.subTest(option=option, value=value):
                parser = _build_parser()
                args = parser.parse_args(["distilbert", "hetero+ccc", option, value])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    _validate_args(parser, args)

        parser = _build_parser()
        args = parser.parse_args(
            [
                "distilbert",
                "hetero+ccc",
                "--hetero-logvar-min",
                "2",
                "--hetero-logvar-max",
                "-2",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _validate_args(parser, args)

    def test_trainer_and_model_config_receive_effective_objective_settings(self):
        captured = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        params = {
            "hetero_mse_weight": 0.2,
            "hetero_ccc_weight": 0.4,
            "hetero_logvar_min": -4.0,
            "hetero_logvar_max": 2.0,
        }
        train_data = SimpleNamespace(tokenizer=object())
        with patch.dict(LOSS_TO_TRAINER, {"hetero+ccc": FakeTrainer}):
            _build_trainer(
                "hetero+ccc",
                model=object(),
                training_args=object(),
                train_data=train_data,
                val_data=object(),
                params=params,
            )

        self.assertEqual(captured["hetero_mse_weight"], 0.2)
        self.assertEqual(captured["hetero_ccc_weight"], 0.4)
        self.assertEqual(captured["hetero_logvar_min"], -4.0)
        self.assertEqual(captured["hetero_logvar_max"], 2.0)
        self.assertEqual(captured["compute_metrics"].keywords["logvar_min"], -4.0)
        self.assertEqual(captured["compute_metrics"].keywords["logvar_max"], 2.0)

        captured.clear()
        with patch.dict(LOSS_TO_TRAINER, {"hetero": FakeTrainer}):
            _build_trainer(
                "hetero",
                model=object(),
                training_args=object(),
                train_data=train_data,
                val_data=object(),
                params=params,
            )
        self.assertEqual(captured["hetero_ccc_weight"], 0.0)

        model = SimpleNamespace(config=SimpleNamespace())
        _attach_heteroscedastic_config(model, "hetero+ccc", params)
        self.assertEqual(model.config.loss_function, "hetero+ccc")
        self.assertEqual(model.config.num_labels, 4)
        self.assertEqual(model.config.hetero_ccc_weight, 0.4)
        self.assertEqual(model.config.checkpoint_selection_metric, "ccc_mean")
        self.assertTrue(model.config.checkpoint_greater_is_better)
        self.assertTrue(model.config.checkpoint_selection_enabled)
        self.assertEqual(
            model.config.heteroscedastic_outputs,
            list(HETEROSCEDASTIC_OUTPUTS),
        )

        plain_hetero_model = SimpleNamespace(config=SimpleNamespace())
        _attach_heteroscedastic_config(plain_hetero_model, "hetero", params)
        self.assertEqual(plain_hetero_model.config.hetero_ccc_weight, 0.0)


class VAEvaluationMetricsTest(unittest.TestCase):
    def test_point_metrics_include_pearson_and_ccc(self):
        labels = np.array([[-1.0, -2.0], [1.0, 0.0]])
        predictions = labels + 1.0

        metrics = calculate_va_metrics(labels, predictions)

        for dimension in ("valence", "arousal"):
            self.assertAlmostEqual(metrics[f"mse_{dimension}"], 1.0)
            self.assertAlmostEqual(metrics[f"rmse_{dimension}"], 1.0)
            self.assertAlmostEqual(metrics[f"mae_{dimension}"], 1.0)
            self.assertAlmostEqual(metrics[f"pearson_corr_{dimension}"], 1.0)
            self.assertAlmostEqual(metrics[f"ccc_{dimension}"], 0.8)
        self.assertAlmostEqual(metrics["ccc_mean"], 0.8)
        self.assertAlmostEqual(metrics["pearson_corr_mean"], 1.0)
        self.assertAlmostEqual(metrics["mse_mean"], 1.0)
        self.assertNotIn("gaussian_nll_valence", metrics)

    def test_constant_and_singleton_inputs_return_nan_correlations(self):
        self.assertTrue(
            math.isnan(safe_pearson_correlation([1.0, 1.0], [1.0, 1.0]))
        )
        self.assertTrue(
            math.isnan(concordance_correlation_coefficient([1.0], [1.0]))
        )

        metrics = calculate_va_metrics(
            np.array([[1.0, 2.0], [1.0, 2.0]]),
            np.array([[1.0, 2.0], [1.0, 2.0]]),
        )
        self.assertTrue(math.isnan(metrics["pearson_corr_valence"]))
        self.assertTrue(math.isnan(metrics["ccc_valence"]))
        self.assertTrue(math.isnan(metrics["ccc_mean"]))

        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_metrics(
                (
                    np.array([[1.0, 2.0], [1.0, 2.0]]),
                    np.array([[1.0, 2.0], [1.0, 2.0]]),
                ),
                metric_for_best_model="ccc_mean",
            )

    def test_checkpoint_metric_validation_rejects_unavailable_uncertainty_metric(self):
        labels = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            compute_metrics(
                (labels.copy(), labels),
                metric_for_best_model="gaussian_nll_mean",
            )

    def test_uncertainty_metrics_use_effective_clamped_logvars(self):
        labels = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        predictions = np.column_stack(
            (
                labels,
                np.full(3, -10.0),
                np.full(3, 10.0),
            )
        )

        metrics = calculate_va_metrics(
            labels,
            predictions,
            logvar_min=-2.0,
            logvar_max=2.0,
        )

        self.assertAlmostEqual(metrics["mean_logvar_valence"], -10.0)
        self.assertAlmostEqual(metrics["mean_logvar_arousal"], 10.0)
        self.assertAlmostEqual(metrics["mean_raw_logvar_valence"], -10.0)
        self.assertAlmostEqual(metrics["mean_raw_logvar_arousal"], 10.0)
        self.assertAlmostEqual(metrics["mean_effective_logvar_valence"], -2.0)
        self.assertAlmostEqual(metrics["mean_effective_logvar_arousal"], 2.0)
        self.assertAlmostEqual(metrics["mean_variance_valence"], math.exp(-2.0))
        self.assertAlmostEqual(metrics["mean_variance_arousal"], math.exp(2.0))
        self.assertEqual(metrics["logvar_lower_clamp_rate_valence"], 1.0)
        self.assertEqual(metrics["logvar_upper_clamp_rate_arousal"], 1.0)
        self.assertAlmostEqual(
            metrics["gaussian_nll_valence"],
            0.5 * (math.log(2.0 * math.pi) - 2.0),
        )
        self.assertAlmostEqual(
            metrics["gaussian_nll_arousal"],
            0.5 * (math.log(2.0 * math.pi) + 2.0),
        )
        self.assertEqual(metrics["coverage_1sigma_mean"], 1.0)
        self.assertEqual(metrics["coverage_2sigma_mean"], 1.0)

    def test_uncertainty_error_spearman_tracks_squared_error_order(self):
        labels = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        predictions = np.array(
            [
                [0.0, 0.0, -2.0, -2.0],
                [1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )

        metrics = calculate_va_metrics(labels, predictions)

        self.assertAlmostEqual(metrics["uncertainty_error_spearman_valence"], 1.0)
        self.assertAlmostEqual(metrics["uncertainty_error_spearman_arousal"], 1.0)
        self.assertAlmostEqual(metrics["uncertainty_error_spearman_mean"], 1.0)

    def test_compute_metrics_wrapper_and_input_validation(self):
        labels = np.array([[-1.0, 0.0], [1.0, 2.0]])
        predictions = labels.copy()
        metrics = compute_metrics((predictions, labels))
        self.assertEqual(metrics["mse_valence"], 0.0)
        self.assertEqual(metrics["ccc_valence"], 1.0)

        invalid_pairs = (
            (np.zeros((0, 2)), np.zeros((0, 2))),
            (np.zeros((2, 3)), np.zeros((2, 2))),
            (np.zeros((2, 2)), np.zeros((2, 1))),
            (np.zeros((2, 2)), np.zeros((2, 3))),
            (np.zeros((2, 2)), np.zeros((2, 5))),
            (np.zeros((3, 2)), np.zeros((2, 2))),
            (np.zeros((2, 2)), np.full((2, 2), np.nan)),
        )
        for invalid_labels, invalid_predictions in invalid_pairs:
            with self.subTest(
                labels_shape=invalid_labels.shape,
                predictions_shape=invalid_predictions.shape,
            ), self.assertRaises(ValueError):
                calculate_va_metrics(invalid_labels, invalid_predictions)

        with self.assertRaises(ValueError):
            effective_logvars([[0.0, 1.0]], logvar_min=1.0, logvar_max=1.0)
        with self.assertRaises(ValueError):
            effective_logvars([[float("inf"), 1.0]])


class OOFHeteroscedasticReportTest(unittest.TestCase):
    def test_uncertainty_bins_keep_tied_values_together(self):
        values = np.array([1.0, 1.0, 1.0, 2.0])
        bin_ids = _equal_count_bin_ids(values)

        self.assertEqual(len(set(bin_ids[values == 1.0])), 1)
        self.assertNotEqual(bin_ids[0], bin_ids[-1])
        np.testing.assert_array_equal(
            _equal_count_bin_ids(np.ones(5)),
            np.zeros(5, dtype=np.int64),
        )

    def test_risk_coverage_never_splits_uncertainty_ties(self):
        selections = _least_uncertain_selections(np.ones(4))
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0][0], 1.0)
        np.testing.assert_array_equal(selections[0][1], np.arange(4))

        frame = pd.DataFrame(
            {
                "valence_true": [0.0, 0.0, 0.0, 10.0],
                "arousal_true": [0.0, 0.0, 0.0, 10.0],
                "valence_pred": [0.0, 0.0, 0.0, 0.0],
                "arousal_pred": [0.0, 0.0, 0.0, 0.0],
                "valence_variance_pred": np.ones(4),
                "arousal_variance_pred": np.ones(4),
            }
        )
        original = pd.DataFrame(_build_uncertainty_risk_coverage_rows(frame))
        permuted = pd.DataFrame(
            _build_uncertainty_risk_coverage_rows(
                frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
            )
        )
        pd.testing.assert_frame_equal(original, permuted)

    def test_point_only_rebuild_removes_stale_uncertainty_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stale_paths = [
                root / "uncertainty_calibration.csv",
                root / "uncertainty_risk_coverage.csv",
            ]
            for stale_path in stale_paths:
                stale_path.write_text("stale\n", encoding="utf-8")

            _remove_stale_uncertainty_reports(temp_dir)

            self.assertTrue(all(not stale_path.exists() for stale_path in stale_paths))

    def test_oof_reports_preserve_raw_values_and_write_calibration_artifacts(self):
        dataset_columns = ["index", "text", "dataset_of_origin", "valence", "arousal"]
        fold1_dataset = pd.DataFrame(
            [
                [0, "zero", "Emobank", -1.0, -1.0],
                [2, "two", "Emobank", 1.0, 1.0],
            ],
            columns=dataset_columns,
        )
        fold2_dataset = pd.DataFrame(
            [
                [1, "one", "Emobank", 0.0, 0.0],
                [3, "three", "Emobank", 2.0, 2.0],
            ],
            columns=dataset_columns,
        )
        fold1_predictions = pd.DataFrame(
            [
                [-1.0, -1.0, -10.0, 10.0],
                [0.5, 1.5, -1.0, 1.0],
            ]
        )
        fold2_predictions = pd.DataFrame(
            [
                [0.2, -0.2, 0.0, 0.0],
                [2.0, 2.0, 10.0, -10.0],
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            predictions_dir = root / "predictions"
            data_dir.mkdir()
            predictions_dir.mkdir()
            fold1_dataset.to_csv(
                data_dir / "full_dataset_fold1.csv",
                sep="\t",
                index=False,
            )
            fold2_dataset.to_csv(
                data_dir / "full_dataset_fold2.csv",
                sep="\t",
                index=False,
            )
            fold1_predictions.to_csv(predictions_dir / "predictions_fold1.csv")
            fold2_predictions.to_csv(predictions_dir / "predictions_fold2.csv")
            with open(predictions_dir / "training_parameters.json", "w") as output_file:
                json.dump(
                    {"hetero_logvar_min": -2.0, "hetero_logvar_max": 2.0},
                    output_file,
                )

            create_prediction_tables(str(predictions_dir), data_dir=str(data_dir))

            combined = pd.read_csv(predictions_dir / "all_predictions.csv")
            self.assertEqual(combined["index"].tolist(), [0, 1, 2, 3])
            self.assertEqual(combined.loc[0, "valence_logvar_pred"], -10.0)
            self.assertEqual(combined.loc[0, "arousal_logvar_pred"], 10.0)
            self.assertEqual(combined.loc[0, "valence_effective_logvar_pred"], -2.0)
            self.assertEqual(combined.loc[0, "arousal_effective_logvar_pred"], 2.0)
            self.assertAlmostEqual(
                combined.loc[0, "valence_variance_pred"],
                math.exp(-2.0),
            )
            self.assertAlmostEqual(
                combined.loc[0, "arousal_variance_pred"],
                math.exp(2.0),
            )

            with open(predictions_dir / "overall_metrics.json") as input_file:
                overall_metrics = json.load(input_file)
            self.assertEqual(overall_metrics["hetero_logvar_min"], -2.0)
            self.assertEqual(overall_metrics["hetero_logvar_max"], 2.0)
            self.assertIn("ccc_valence", overall_metrics)
            self.assertIn("gaussian_nll_mean", overall_metrics)

            calibration = pd.read_csv(predictions_dir / "uncertainty_calibration.csv")
            self.assertEqual(set(calibration["dimension"]), {"valence", "arousal"})
            self.assertEqual(len(calibration), 8)
            self.assertTrue(
                {
                    "mean_variance",
                    "gaussian_nll",
                    "coverage_1sigma",
                    "coverage_2sigma",
                    "mse_to_mean_variance_ratio",
                }.issubset(calibration.columns)
            )

            risk_coverage = pd.read_csv(
                predictions_dir / "uncertainty_risk_coverage.csv"
            )
            self.assertEqual(
                set(risk_coverage["dimension"]),
                {"valence", "arousal", "joint"},
            )
            self.assertEqual(len(risk_coverage), 11)
            self.assertTrue(
                (risk_coverage["actual_coverage"] >= risk_coverage["target_coverage"]).all()
            )
            self.assertTrue(
                {
                    "actual_coverage",
                    "mean_uncertainty_score",
                    "mse",
                    "pearson_corr",
                    "ccc",
                }.issubset(risk_coverage.columns)
            )
            legacy_sentence_table = pd.read_pickle(predictions_dir / "table2.pkl")
            self.assertIn(("Valence", "RMSE"), legacy_sentence_table.columns)
            self.assertNotIn(("Valence", "MSE"), legacy_sentence_table.columns)


if __name__ == "__main__":
    unittest.main()
