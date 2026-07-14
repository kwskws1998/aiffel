import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from va_gaze.train.custom_trainer import (
    HETEROSCEDASTIC_CONFIG_FILENAME,
    CustomTrainerMSE,
)


class ManifestAwareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, input_ids=None, **kwargs):
        del kwargs
        return {"logits": self.projection(input_ids.float())}

    def save_architecture_manifest(self, output_dir):
        Path(output_dir, "manifest-called.txt").write_text("ok\n", encoding="utf-8")


class TrainerPersistenceTest(unittest.TestCase):
    def test_training_reloads_middle_epoch_with_highest_full_fold_ccc(self):
        class ScalarRegressionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids=None, labels=None, **kwargs):
                del labels, kwargs
                return {"logits": self.weight.expand(input_ids.shape[0], 2)}

        class SequencedMetricTrainer(CustomTrainerMSE):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.metric_sequence = iter((0.1, 0.3, 0.2))
                self.evaluated_weights = []

            def evaluate(self, *args, **kwargs):
                del args, kwargs
                self.evaluated_weights.append(self.model.weight.detach().clone())
                return {
                    "eval_ccc_mean": next(self.metric_sequence),
                    "eval_loss": 1.0,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            strategy_name = (
                "eval_strategy"
                if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                else "evaluation_strategy"
            )
            arguments = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=3,
                per_device_train_batch_size=1,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="ccc_mean",
                greater_is_better=True,
                save_total_limit=1,
                learning_rate=0.1,
                logging_strategy="no",
                disable_tqdm=True,
                dataloader_pin_memory=False,
                report_to=[],
                **{strategy_name: "epoch"},
            )
            dataset = [
                {
                    "input_ids": torch.tensor([0]),
                    "labels": torch.tensor([1.0, 1.0]),
                }
            ]
            trainer = SequencedMetricTrainer(
                model=ScalarRegressionModel(),
                args=arguments,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

            trainer.train()

            self.assertAlmostEqual(trainer.state.best_metric, 0.3)
            self.assertEqual(
                Path(trainer.state.best_model_checkpoint).name,
                "checkpoint-2",
            )
            self.assertTrue(
                torch.equal(
                    trainer.model.weight.detach(),
                    trainer.evaluated_weights[1],
                )
            )
            checkpoint_names = sorted(
                path.name for path in Path(temp_dir).glob("checkpoint-*")
            )
            self.assertEqual(checkpoint_names, ["checkpoint-2"])

    def test_every_trainer_save_invokes_model_manifest_hook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "saved"
            arguments = TrainingArguments(
                output_dir=str(Path(temp_dir) / "trainer"),
                report_to=[],
            )
            trainer = CustomTrainerMSE(
                model=ManifestAwareModel(),
                args=arguments,
            )
            trainer.save_model(output_dir)
            self.assertTrue((output_dir / "model.safetensors").is_file())
            self.assertEqual(
                (output_dir / "manifest-called.txt").read_text(encoding="utf-8"),
                "ok\n",
            )

    def test_plain_module_save_persists_heteroscedastic_objective(self):
        model = ManifestAwareModel()
        model.config = SimpleNamespace(
            loss_function="hetero+ccc",
            num_labels=4,
            hetero_mse_weight=0.2,
            hetero_ccc_weight=0.3,
            hetero_logvar_min=-4.0,
            hetero_logvar_max=2.0,
            checkpoint_selection_metric="pearson_corr_mean",
            checkpoint_greater_is_better=True,
            checkpoint_selection_enabled=True,
            heteroscedastic_outputs=[
                "valence_mu",
                "arousal_mu",
                "valence_logvar_raw",
                "arousal_logvar_raw",
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "saved"
            strategy_name = (
                "eval_strategy"
                if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                else "evaluation_strategy"
            )
            arguments = TrainingArguments(
                output_dir=str(Path(temp_dir) / "trainer"),
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="pearson_corr_mean",
                greater_is_better=True,
                report_to=[],
                **{strategy_name: "epoch"},
            )
            trainer = CustomTrainerMSE(
                model=model,
                args=arguments,
                eval_dataset=[{}],
            )
            trainer.save_model(output_dir)

            with open(output_dir / HETEROSCEDASTIC_CONFIG_FILENAME) as input_file:
                metadata = json.load(input_file)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["loss_function"], "hetero+ccc")
            self.assertEqual(metadata["num_labels"], 4)
            self.assertEqual(metadata["hetero_ccc_weight"], 0.3)
            self.assertEqual(metadata["hetero_logvar_min"], -4.0)
            self.assertEqual(metadata["hetero_logvar_max"], 2.0)
            self.assertEqual(
                metadata["checkpoint_selection_metric"],
                "pearson_corr_mean",
            )
            self.assertTrue(metadata["checkpoint_greater_is_better"])
            self.assertTrue(metadata["checkpoint_selection_enabled"])

            disabled_output_dir = Path(temp_dir) / "saved-disabled"
            disabled_arguments = TrainingArguments(
                output_dir=str(Path(temp_dir) / "trainer-disabled"),
                report_to=[],
            )
            disabled_trainer = CustomTrainerMSE(
                model=model,
                args=disabled_arguments,
            )
            disabled_trainer.save_model(disabled_output_dir)
            with open(
                disabled_output_dir / HETEROSCEDASTIC_CONFIG_FILENAME
            ) as input_file:
                disabled_metadata = json.load(input_file)
            self.assertIsNone(disabled_metadata["checkpoint_selection_metric"])
            self.assertIsNone(disabled_metadata["checkpoint_greater_is_better"])
            self.assertFalse(disabled_metadata["checkpoint_selection_enabled"])


if __name__ == "__main__":
    unittest.main()
