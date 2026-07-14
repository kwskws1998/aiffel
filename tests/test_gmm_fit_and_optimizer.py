"""Integration tests for train-fold GMM fitting and residual optimization."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from va_gaze.models.gaze.simple_gmm import GmmArousalLogitResidual
from va_gaze.models.gaze.types import GazeBatch
from va_gaze.train.custom_trainer import VARegressionTrainer
from va_gaze.train.gmm_fit import (
    collect_train_fold_gaze_summaries,
    fit_train_fold_gmm_residual,
)


class TinyGazeDataset:
    """Expose deterministic token inputs without labels for fitting tests."""

    def __init__(self, size=12):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        value = int(index) + 2
        return {
            "input_ids": torch.tensor([1, value, value + 1, 2]),
            "attention_mask": torch.ones(4, dtype=torch.long),
        }


class DeterministicGazeProvider:
    """Create five finite mapped features from token ids."""

    def compute(self, input_ids, attention_mask):
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        multipliers = torch.arange(1, 6, dtype=torch.float32).view(1, 1, 5)
        features = base * multipliers
        mapped_mask = attention_mask.to(dtype=torch.bool)
        return GazeBatch(
            features=features,
            mapped_mask=mapped_mask,
            text_mask=mapped_mask,
        )


class BFloat16GazeProvider(DeterministicGazeProvider):
    """Exercise the NumPy boundary with reduced-precision gaze summaries."""

    def compute(self, input_ids, attention_mask):
        gaze_batch = super().compute(input_ids, attention_mask)
        return GazeBatch(
            features=gaze_batch.features.to(dtype=torch.bfloat16),
            mapped_mask=gaze_batch.mapped_mask,
            text_mask=gaze_batch.text_mask,
        )


class OptimizerProbe(torch.nn.Module):
    """Expose one residual parameter separately from an ordinary base layer."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.residual = torch.nn.Linear(2, 1, bias=False)

    def gaze_residual_parameters(self):
        return tuple(self.residual.parameters())


class GmmFitAndOptimizerTest(unittest.TestCase):
    def test_bfloat16_gaze_summaries_are_converted_to_float32_numpy(self):
        residual = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=2,
            mode="component-linear",
        )
        model = SimpleNamespace(
            gmm_residual=residual,
            gaze_provider=BFloat16GazeProvider(),
        )

        summaries, collection = collect_train_fold_gaze_summaries(
            model=model,
            train_data=TinyGazeDataset(size=3),
            max_examples=3,
            max_tokens=100,
            random_state=13,
        )

        self.assertEqual(summaries.shape, (3, 5))
        self.assertEqual(summaries.dtype, "float32")
        self.assertEqual(collection["fitted_examples"], 3)

    def test_fold_fit_installs_gmm_and_writes_diagnostics(self):
        residual = GmmArousalLogitResidual(
            feature_dim=5,
            n_components=2,
            mode="component-linear",
        )
        model = SimpleNamespace(
            gmm_residual=residual,
            gaze_provider=DeterministicGazeProvider(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics = fit_train_fold_gmm_residual(
                model=model,
                train_data=TinyGazeDataset(),
                fold_id=1,
                output_dir=temp_dir,
                random_state=13,
                max_examples=8,
                max_tokens=100,
                n_init=2,
            )
            path = Path(temp_dir) / "gmm_fit_fold1.json"
            self.assertTrue(path.is_file())
            with open(path, "r", encoding="utf-8") as input_file:
                persisted = json.load(input_file)

        self.assertTrue(bool(residual.is_fitted.item()))
        self.assertEqual(diagnostics, persisted)
        self.assertEqual(diagnostics["collection"]["examined_examples"], 8)
        self.assertEqual(diagnostics["fit"]["sample_count"], 8)
        self.assertEqual(diagnostics["trainable_coefficients"], 12)

    def test_gaze_residual_receives_its_explicit_learning_rate(self):
        model = OptimizerProbe()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(
                output_dir=temp_dir,
                learning_rate=6e-6,
                weight_decay=0.01,
                report_to=[],
            )
            trainer = VARegressionTrainer(
                model=model,
                args=args,
                gaze_learning_rate=1e-3,
            )
            optimizer = trainer.create_optimizer()

        residual_parameter_id = id(model.residual.weight)
        matching_groups = [
            group
            for group in optimizer.param_groups
            if any(id(parameter) == residual_parameter_id for parameter in group["params"])
        ]
        self.assertEqual(len(matching_groups), 1)
        self.assertEqual(matching_groups[0]["lr"], 1e-3)
        self.assertEqual(matching_groups[0]["weight_decay"], 0.0)
        base_groups = [
            group
            for group in optimizer.param_groups
            if any(id(parameter) == id(model.base.weight) for parameter in group["params"])
        ]
        self.assertEqual(len(base_groups), 1)
        self.assertEqual(base_groups[0]["lr"], 6e-6)


if __name__ == "__main__":
    unittest.main()
