import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from va_gaze.cli.train_model import _build_parser, _validate_args
from va_gaze.train.fold_runner import (
    _build_model,
    _validate_prediction_array,
    run_fold,
)


def parse_and_validate(arguments):
    parser = _build_parser()
    args = parser.parse_args(arguments)
    _validate_args(parser, args)
    return args


class TrainCliValidationTest(unittest.TestCase):
    def assert_cli_error(self, extra_arguments, message):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_and_validate(["xlmroberta-base", "mse", *extra_arguments])
        self.assertIn(message, stderr.getvalue())

    def test_legacy_fusion_rejects_training_objectives(self):
        for fusion in (
            "postfix-concat",
            "prefix-concat",
            "add",
            "summary",
            "gmm-adapter",
        ):
            with self.subTest(fusion=fusion):
                self.assert_cli_error(
                    ["--gaze-fusion", fusion, "--gaze-aux-weight", "0.1"],
                    "cannot be combined with postfix-concat/prefix-concat/add/summary/gmm-adapter",
                )

    def test_concat_aliases_default_to_postfix_and_prefix_is_explicit(self):
        for arguments in (
            ["--gaze-fusion", "concat"],
            ["--gaze-fusion", "concat-postfix"],
            ["--use-gaze-concat"],
        ):
            with self.subTest(arguments=arguments):
                args = parse_and_validate(["xlmroberta-base", "mse", *arguments])
                self.assertEqual(args.gaze_fusion, "postfix-concat")

        args = parse_and_validate(
            ["xlmroberta-base", "mse", "--gaze-fusion", "concat-prefix"]
        )
        self.assertEqual(args.gaze_fusion, "prefix-concat")

    def test_concat_maxlen_boundary_accounts_for_doubled_sequence(self):
        args = parse_and_validate(
            [
                "xlmroberta-base",
                "mse",
                "--gaze-fusion",
                "postfix-concat",
                "--maxlen",
                "255",
            ]
        )
        self.assertEqual(args.maxlen, 255)
        self.assert_cli_error(
            ["--gaze-fusion", "prefix-concat", "--maxlen", "256"],
            "maxlen must be <= 255",
        )

    def test_cross_attention_validates_head_divisibility(self):
        self.assert_cli_error(
            [
                "--gaze-fusion",
                "cross-attention",
                "--gaze-hidden-size",
                "10",
                "--gaze-num-heads",
                "4",
            ],
            "gaze_hidden_size divisible by gaze_num_heads",
        )

    def test_known_encoder_validates_attention_head_divisibility(self):
        self.assert_cli_error(
            [
                "--gaze-fusion",
                "postencoder-cls-attention-bias",
                "--gaze-num-heads",
                "5",
            ],
            "encoder hidden size divisible by gaze_num_heads",
        )

    def test_attention_alias_is_persisted_as_canonical_name(self):
        args = parse_and_validate(
            ["xlmroberta-base", "mse", "--gaze-fusion", "cls-attention-bias"]
        )
        self.assertEqual(args.gaze_fusion, "postencoder-cls-attention-bias")

    def test_report_to_none_becomes_empty_reporter_list(self):
        args = parse_and_validate(
            ["xlmroberta-base", "mse", "--report-to", "none"]
        )
        self.assertEqual(args.report_to, [])

    def test_emotion_trt_alias_requires_trt_only_when_gaze_is_enabled(self):
        args = parse_and_validate(
            [
                "xlmroberta-base",
                "mse",
                "--gaze-fusion",
                "postfix-concat",
                "--et-model-type",
                "emotion_trt",
                "--features-used",
                "0,0,0,1,0",
            ]
        )
        self.assertEqual(args.et_model_type, "emotion-trt")
        self.assert_cli_error(
            [
                "--gaze-fusion",
                "postfix-concat",
                "--et-model-type",
                "emotion-trt",
                "--features-used",
                "0,1,0,1,0",
            ],
            "predicts TRT only",
        )

    def test_model_factory_also_rejects_silent_legacy_downgrade(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot be combined with postfix-concat/prefix-concat",
        ):
            _build_model(
                "xlmroberta-base",
                "unused",
                object(),
                {
                    "gaze_fusion": "summary",
                    "gaze_aux_weight": 0.1,
                    "et_model_type": "et2",
                },
            )

    def test_model_factory_builds_postfix_by_default_and_prefix_explicitly(self):
        captured = []

        def fake_concat(**kwargs):
            captured.append(kwargs["concat_order"])
            return object()

        with patch(
            "va_gaze.train.fold_runner.GazeConcatForSequenceRegression",
            side_effect=fake_concat,
        ):
            _build_model(
                "xlmroberta-base",
                "unused",
                object(),
                {"gaze_fusion": "concat", "et_model_type": "heuristic"},
            )
            _build_model(
                "xlmroberta-base",
                "unused",
                object(),
                {"gaze_fusion": "prefix-concat", "et_model_type": "heuristic"},
            )
        self.assertEqual(captured, ["postfix", "prefix"])

    def test_model_factory_rejects_unknown_fusion(self):
        with self.assertRaisesRegex(ValueError, "Unknown gaze fusion strategy"):
            _build_model(
                "xlmroberta-base",
                "unused",
                object(),
                {
                    "gaze_fusion": "typo-fusion",
                    "gaze_aux_weight": 0.1,
                    "et_model_type": "et2",
                },
            )


class FoldRunnerContractTest(unittest.TestCase):
    def test_prediction_array_contract(self):
        expected = np.zeros((3, 2), dtype=np.float32)
        self.assertIs(_validate_prediction_array(expected, 2), expected)

        invalid_values = (
            ((expected,), TypeError),
            (np.zeros(3), ValueError),
            (np.zeros((3, 4)), ValueError),
            (np.full((3, 2), np.nan), ValueError),
            (np.full((3, 2), "x"), TypeError),
        )
        for value, exception in invalid_values:
            with self.subTest(value_type=type(value).__name__, exception=exception.__name__):
                with self.assertRaises(exception):
                    _validate_prediction_array(value, 2)

    def test_run_fold_seeds_before_model_and_ignores_non_logits_outputs(self):
        events = []

        class FakeTrainer:
            def train(self):
                events.append("train")

            def predict(self, dataset, ignore_keys=None):
                events.append(("predict", tuple(ignore_keys or ())))
                return SimpleNamespace(
                    predictions=np.zeros((2, 2), dtype=np.float32),
                    metrics={"test_loss": 0.0},
                )

        train_data = SimpleNamespace(tokenizer=object())
        params = {
            "batch_size_xlmrB": 2,
            "seed": 17,
            "save_final_model": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "va_gaze.train.fold_runner.set_seed",
                        side_effect=lambda seed: events.append(("seed", seed)),
                    )
                )
                stack.enter_context(
                    patch(
                        "va_gaze.train.fold_runner._build_model",
                        side_effect=lambda *args, **kwargs: events.append("build") or object(),
                    )
                )
                stack.enter_context(
                    patch(
                        "va_gaze.train.fold_runner._build_training_args",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    patch(
                        "va_gaze.train.fold_runner._build_trainer",
                        return_value=FakeTrainer(),
                    )
                )
                run_fold(
                    fold_id=1,
                    model_name="xlmroberta-base",
                    loss_name="mse",
                    timestamp="test",
                    params=params,
                    train_data=train_data,
                    val_data=object(),
                    preds_dir=temp_dir,
                    checkpoint="unused",
                    prediction_filename="predictions.csv",
                    metrics_filename="metrics.csv",
                    gaze_config={},
                )

            self.assertEqual(events[0:2], [("seed", 17), "build"])
            self.assertIn(
                ("predict", ("hidden_states", "attentions")),
                events,
            )
            prediction_path = Path(temp_dir) / "predictions.csv"
            self.assertTrue(prediction_path.is_file())
            self.assertEqual(np.loadtxt(prediction_path, delimiter=",", skiprows=1).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
