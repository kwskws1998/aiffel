import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertModel,
    DistilBertConfig,
    DistilBertModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    XLMRobertaConfig,
)

from va_gaze.models.advanced_regression import (
    ADVANCED_GAZE_MANIFEST_NAME,
    DistilBertVARegressionHead,
    GazeFusionForSequenceRegression,
    RobertaVARegressionHead,
)
from va_gaze.models.gaze.objectives import MaskedGazePrediction
from va_gaze.models.gaze.simple_gmm import fit_diagonal_gmm
from va_gaze.models.regression import (
    DistilBertForSequenceClassificationSig,
    XLMRobertaForSequenceClassificationSig,
)
from va_gaze.train.custom_trainer import _add_model_auxiliary_loss


class TinyTokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0


def tiny_encoder():
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return BertModel(config)


def tiny_distilbert_encoder():
    config = DistilBertConfig(
        vocab_size=32,
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
    )
    return DistilBertModel(config)


def tiny_roberta_encoder():
    config = RobertaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return RobertaModel(config)


def save_tiny_tokenizer(path):
    vocabulary = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
        "calm": 5,
        "bright": 6,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    wrapped.save_pretrained(path)
    return wrapped


class AdvancedRegressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.input_ids = torch.tensor([[1, 5, 6, 7, 2], [1, 8, 9, 10, 2]])
        self.attention_mask = torch.ones_like(self.input_ids)

    def _model(self, encoder=None, **kwargs):
        defaults = {
            "checkpoint": "unused",
            "tokenizer": TinyTokenizer(),
            "et_model_type": "heuristic",
            "gaze_hidden_size": 8,
            "gaze_num_heads": 2,
            "gaze_num_layers": 1,
            "gaze_fusion_dropout": 0.0,
            "gaze_gate_init": 0.0,
            "gaze_alignment_dim": 8,
        }
        defaults.update(kwargs)
        encoder = encoder if encoder is not None else tiny_encoder()
        with patch(
            "va_gaze.models.advanced_regression.AutoModel.from_pretrained",
            return_value=encoder,
        ):
            return GazeFusionForSequenceRegression(**defaults)

    def test_all_primary_fusions_forward_and_backward(self):
        for strategy in (
            "conditioned-pooling",
            "postencoder-cls-attention-bias",
            "cross-attention",
        ):
            with self.subTest(strategy=strategy):
                model = self._model(fusion_strategy=strategy)
                model.train()
                outputs = model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                )
                self.assertEqual(tuple(outputs.logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(outputs.logits).all())
                self.assertIsNone(outputs.loss)
                outputs.logits.sum().backward()
                fusion_gradient = sum(
                    float(parameter.grad.abs().sum())
                    for parameter in model.fusion.parameters()
                    if parameter.grad is not None
                )
                self.assertGreater(fusion_gradient, 0.0)

    def test_legacy_attention_fusion_alias_is_backward_compatible(self):
        model = self._model(fusion_strategy="cls-attention-bias")
        self.assertEqual(model.fusion_strategy, "postencoder-cls-attention-bias")

    def test_auxiliary_only_is_lazy_during_inference(self):
        model = self._model(fusion_strategy="none", gaze_aux_weight=0.1)
        model.eval()
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertIsNone(outputs.loss)
        self.assertIsNone(model.gaze_provider.fp_model)

        model.train()
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertIsNotNone(model.gaze_provider.fp_model)

    def test_alignment_only_produces_training_regularizer(self):
        model = self._model(fusion_strategy="none", gaze_alignment_weight=0.05)
        model.train()
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertIsNotNone(outputs.loss)
        self.assertGreater(float(outputs.loss.detach()), 0.0)
        total = outputs.logits.mean() + outputs.loss
        total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.gaze_alignment.parameters()))

    def test_gmm_dual_gate_uses_task_specific_heads_and_auxiliary_density_loss(self):
        model = self._model(
            fusion_strategy="gmm-dual-gate-pooling",
            features_used=[1, 1, 1, 1, 1],
            gmm_components=3,
            gmm_temperature=1.0,
            gmm_nll_weight=0.01,
        )
        model.train()
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertEqual(tuple(outputs.logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(outputs.logits).all())
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss))
        (outputs.logits.mean() + outputs.loss).backward()
        self.assertIsNotNone(model.fusion.gmm_means.grad)

        model.eval()
        inference_outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )
        self.assertIsNone(inference_outputs.loss)

    def test_fixed_gmm_residual_starts_at_baseline_and_changes_only_arousal(self):
        model = self._model(
            fusion_strategy="gmm-arousal-residual",
            features_used=[1, 1, 1, 1, 1],
            gmm_components=2,
            gmm_residual_mode="component-linear",
            gmm_residual_l2=0.0,
        )
        fit_rows = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.1, 0.4, 0.3, 0.6],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [1.2, 1.1, 1.4, 1.3, 1.6],
            ],
            dtype=np.float64,
        )
        model.gmm_residual.set_fit(
            fit_diagonal_gmm(fit_rows, n_components=2, random_state=7)
        )
        model.eval()
        with torch.no_grad():
            encoder_outputs = model._encode_text(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            )
            expected = model.regression_head(encoder_outputs.last_hidden_state[:, 0, :])
            zero_residual = model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits
        torch.testing.assert_close(zero_residual, expected, rtol=0.0, atol=0.0)

        with torch.no_grad():
            model.gmm_residual.correction.weight.fill_(0.05)
            corrected = model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits
        torch.testing.assert_close(corrected[:, 0], expected[:, 0], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(corrected[:, 1], expected[:, 1]))

    def test_heteroscedastic_output_shape(self):
        model = self._model(fusion_strategy="conditioned-pooling", output_dim=4)
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertEqual(tuple(outputs.logits.shape), (2, 4))
        self.assertTrue(outputs.logits[:, :2].ge(0.0).all())
        self.assertTrue(outputs.logits[:, :2].le(1.0).all())

    def test_trainer_adds_auxiliary_loss_exactly(self):
        task_loss = torch.tensor(2.0)
        combined = _add_model_auxiliary_loss(task_loss, {"loss": torch.tensor(0.25)})
        self.assertEqual(float(combined), 2.25)
        unchanged = _add_model_auxiliary_loss(task_loss, {"loss": None})
        self.assertEqual(float(unchanged), 2.0)

    def test_distilbert_head_matches_baseline_shape_and_scaling(self):
        model = self._model(encoder=tiny_distilbert_encoder(), fusion_strategy="none")
        self.assertIsInstance(model.regression_head, DistilBertVARegressionHead)
        self.assertEqual(model.regression_head.pre_classifier.in_features, 16)
        self.assertEqual(model.regression_head.classifier.out_features, 2)

        head = DistilBertVARegressionHead(hidden_size=3, output_dim=2, dropout=0.0)
        with torch.no_grad():
            head.pre_classifier.weight.zero_()
            head.pre_classifier.bias.zero_()
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([1.0, -1.0]))
        raw_logits = head.raw_logits(torch.zeros(1, 3))
        torch.testing.assert_close(raw_logits, torch.tensor([[1.0, -1.0]]))
        output = head.format_logits(raw_logits)
        torch.testing.assert_close(output, torch.tensor([[1.0, 0.0]]))

    def test_distilbert_baseline_transplant_has_exact_eval_zero_fusion_parity(self):
        config = DistilBertConfig(
            vocab_size=32,
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=4,
            max_position_embeddings=32,
            dropout=0.0,
            attention_dropout=0.0,
            seq_classif_dropout=0.0,
            num_labels=2,
        )
        baseline = DistilBertForSequenceClassificationSig(config)
        advanced = GazeFusionForSequenceRegression.from_baseline_model(
            baseline_model=baseline,
            tokenizer=TinyTokenizer(),
            fusion_strategy="none",
            et_model_type="heuristic",
            load_fixation_model=False,
            gaze_fusion_dropout=0.0,
        )
        baseline.eval()
        advanced.eval()

        with torch.no_grad():
            expected = baseline(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits
            actual = advanced(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits

        self.assertIs(advanced.encoder, baseline.distilbert)
        self.assertIs(
            advanced.regression_head.pre_classifier,
            baseline.pre_classifier,
        )
        self.assertIs(advanced.regression_head.dropout, baseline.dropout)
        self.assertIs(advanced.regression_head.classifier, baseline.classifier)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_xlm_roberta_baseline_transplant_has_exact_eval_zero_fusion_parity(self):
        config = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_labels=2,
        )
        baseline = XLMRobertaForSequenceClassificationSig(config)
        advanced = GazeFusionForSequenceRegression.from_baseline_model(
            baseline_model=baseline,
            tokenizer=TinyTokenizer(),
            fusion_strategy="none",
            et_model_type="heuristic",
            load_fixation_model=False,
            gaze_fusion_dropout=0.0,
        )
        baseline.eval()
        advanced.eval()

        with torch.no_grad():
            expected = baseline(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits
            actual = advanced(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ).logits

        self.assertIs(advanced.encoder, baseline.roberta)
        self.assertIs(
            advanced.regression_head.dropout,
            baseline.classifier.dropout,
        )
        self.assertIs(advanced.regression_head.dense, baseline.classifier.dense)
        self.assertIs(
            advanced.regression_head.out_proj,
            baseline.classifier.out_proj,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_roberta_aux_only_uses_exact_baseline_head_family(self):
        model = self._model(
            encoder=tiny_roberta_encoder(),
            fusion_strategy="none",
            gaze_aux_weight=0.1,
        )
        self.assertIsInstance(model.regression_head, RobertaVARegressionHead)
        self.assertEqual(model.regression_head.dense.in_features, 16)
        self.assertEqual(model.regression_head.out_proj.out_features, 2)
        self.assertFalse(hasattr(model.regression_head, "pre_classifier"))
        outputs = model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self.assertEqual(tuple(outputs.logits.shape), (2, 2))

    def test_gaze_target_transform_is_independent_of_batch_composition(self):
        objective = MaskedGazePrediction(4, 1, dropout=0.0)
        original = torch.tensor([[[1.0], [2.0]]])
        expanded = torch.tensor([[[1.0], [2.0]], [[100.0], [200.0]]])
        expected = objective.transform_targets(original)
        actual = objective.transform_targets(expanded)[:1]
        torch.testing.assert_close(actual, expected)

    def test_hidden_states_are_ignored_by_trainer_prediction_collection(self):
        model = self._model(fusion_strategy="conditioned-pooling")
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            output_hidden_states=True,
        )
        ignored = model.config.keys_to_ignore_at_inference
        prediction_values = tuple(
            value
            for key, value in outputs.items()
            if key not in [*ignored, "loss"]
        )
        self.assertEqual(len(prediction_values), 1)
        self.assertIs(prediction_values[0], outputs.logits)

    def test_self_contained_save_and_offline_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            encoder_dir = root / "encoder"
            tokenizer = save_tiny_tokenizer(encoder_dir)
            encoder = tiny_roberta_encoder()
            encoder.save_pretrained(encoder_dir)

            model = GazeFusionForSequenceRegression(
                checkpoint=str(encoder_dir),
                tokenizer=tokenizer,
                fusion_strategy="postencoder-cls-attention-bias",
                et_model_type="heuristic",
                features_used=[0, 1, 0, 1, 0],
                gaze_hidden_size=8,
                gaze_num_heads=2,
                gaze_num_layers=1,
                gaze_fusion_dropout=0.0,
                gaze_aux_weight=0.1,
            )
            model.eval()
            encoded = tokenizer("calm bright", return_tensors="pt")
            expected = model(**encoded).logits.detach()

            bundle_dir = root / "bundle"
            model.save_pretrained(bundle_dir)
            manifest_path = bundle_dir / ADVANCED_GAZE_MANIFEST_NAME
            self.assertTrue(manifest_path.is_file())
            with open(manifest_path, "r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
            self.assertEqual(
                manifest["architecture"]["fusion_strategy"],
                "postencoder-cls-attention-bias",
            )
            self.assertEqual(manifest["regression_head_family"], "roberta")

            reloaded = GazeFusionForSequenceRegression.from_pretrained(bundle_dir)
            self.assertIsInstance(reloaded.regression_head, RobertaVARegressionHead)
            actual = reloaded(**encoded).logits.detach()
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_fixed_gmm_bundle_preserves_fit_and_residual_coefficients(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            encoder_dir = root / "encoder"
            tokenizer = save_tiny_tokenizer(encoder_dir)
            encoder = tiny_roberta_encoder()
            encoder.save_pretrained(encoder_dir)
            model = GazeFusionForSequenceRegression(
                checkpoint=str(encoder_dir),
                tokenizer=tokenizer,
                fusion_strategy="gmm-arousal-residual",
                et_model_type="heuristic",
                features_used=[1, 1, 1, 1, 1],
                gmm_components=2,
                gmm_residual_mode="component-linear",
                gaze_fusion_dropout=0.0,
            )
            fit_rows = np.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.2, 0.1, 0.4, 0.3, 0.6],
                    [1.1, 1.2, 1.3, 1.4, 1.5],
                    [1.2, 1.1, 1.4, 1.3, 1.6],
                ],
                dtype=np.float64,
            )
            model.gmm_residual.set_fit(
                fit_diagonal_gmm(fit_rows, n_components=2, random_state=11)
            )
            with torch.no_grad():
                model.gmm_residual.correction.weight.copy_(
                    torch.linspace(
                        -0.1,
                        0.1,
                        model.gmm_residual.correction.weight.numel(),
                    ).view(1, -1)
                )
            model.eval()
            encoded = tokenizer("calm bright", return_tensors="pt")
            expected = model(**encoded).logits.detach()

            bundle_dir = root / "gmm_bundle"
            model.save_pretrained(bundle_dir)
            reloaded = GazeFusionForSequenceRegression.from_pretrained(bundle_dir)
            actual = reloaded(**encoded).logits.detach()

            self.assertTrue(bool(reloaded.gmm_residual.is_fitted.item()))
            self.assertEqual(reloaded.gmm_residual.mode, "component-linear")
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
