import unittest
from unittest.mock import patch

import torch
from transformers import (
    BertConfig,
    BertModel,
    DistilBertConfig,
    DistilBertModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from va_gaze.models.gaze.concat import (
    compose_gaze_concat_inputs,
    normalize_concat_order,
)
from va_gaze.models.regression import (
    GazeAddForSequenceRegression,
    GazeConcatForSequenceRegression,
)


class TinyTokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0


def tiny_encoder():
    config = BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )
    return BertModel(config)


def tiny_distilbert_encoder():
    config = DistilBertConfig(
        vocab_size=32,
        dim=8,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        pad_token_id=0,
    )
    return DistilBertModel(config)


def tiny_xlm_roberta_encoder(max_position_embeddings=32):
    config = XLMRobertaConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=max_position_embeddings,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return XLMRobertaModel(config)


class NonFinitePredictor:
    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm):
        batch_size, sequence_length = input_ids_rm.shape
        features = torch.ones(batch_size, sequence_length, 5)
        features[:, 1, 0] = float("nan")
        features[:, 2, 1] = float("inf")
        mask = torch.ones_like(attention_mask_rm)
        return features, mask, None, None, None, None


class DisabledFeatureNonFinitePredictor:
    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm):
        batch_size, sequence_length = input_ids_rm.shape
        features = torch.ones(batch_size, sequence_length, 5)
        features[:, 1, 0] = float("nan")
        mask = torch.ones_like(attention_mask_rm)
        return features, mask, None, None, None, None


class GazeConcatCompositionTest(unittest.TestCase):
    def setUp(self):
        self.text = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
        self.gaze = torch.tensor(
            [[[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]]]
        )
        self.text_mask = torch.tensor([[1, 1, 0]])
        self.gaze_mask = torch.tensor([[1, 0, 0]])
        self.eye_start = torch.tensor([-1.0, -1.0])
        self.eye_end = torch.tensor([-2.0, -2.0])
        self.token_type_ids = torch.tensor([[0, 1, 1]])
        self.position_ids = torch.tensor([[2, 3, 4]])

    def _compose(self, order):
        return compose_gaze_concat_inputs(
            text_embeddings=self.text,
            gaze_embeddings=self.gaze,
            text_attention_mask=self.text_mask,
            gaze_attention_mask=self.gaze_mask,
            eye_start=self.eye_start,
            eye_end=self.eye_end,
            order=order,
            token_type_ids=self.token_type_ids,
            position_ids=self.position_ids,
        )

    def test_postfix_preserves_text_and_cls_at_the_front(self):
        result = self._compose("postfix")
        expected = torch.cat(
            (
                self.text[:, :2],
                self.eye_start.reshape(1, 1, 2),
                self.gaze[:, :2],
                self.eye_end.reshape(1, 1, 2),
            ),
            dim=1,
        )
        torch.testing.assert_close(result.inputs_embeds, expected)
        self.assertEqual(result.cls_positions.tolist(), [0])
        self.assertEqual(result.attention_mask.tolist(), [[1, 1, 1, 1, 0, 1]])
        self.assertEqual(result.token_type_ids.tolist(), [[0, 1, 0, 0, 0, 0]])
        self.assertEqual(result.position_ids.tolist(), [[2, 3, 4, 5, 6, 7]])

    def test_prefix_is_available_only_as_an_explicit_legacy_layout(self):
        result = self._compose("prefix")
        expected = torch.cat(
            (
                self.eye_start.reshape(1, 1, 2),
                self.gaze[:, :2],
                self.eye_end.reshape(1, 1, 2),
                self.text[:, :2],
            ),
            dim=1,
        )
        torch.testing.assert_close(result.inputs_embeds, expected)
        self.assertEqual(result.cls_positions.tolist(), [4])
        self.assertEqual(result.attention_mask.tolist(), [[1, 1, 0, 1, 1, 1]])
        self.assertEqual(result.token_type_ids.tolist(), [[0, 0, 0, 0, 0, 1]])
        self.assertEqual(result.position_ids.tolist(), [[2, 3, 4, 5, 6, 7]])

    def test_postfix_gaze_has_a_gradient_path_to_the_prediction(self):
        cases = (
            ("distilbert", tiny_distilbert_encoder, [1, 5, 6, 2]),
            ("xlm-roberta", tiny_xlm_roberta_encoder, [0, 5, 6, 2]),
        )
        for name, encoder_factory, token_ids in cases:
            with self.subTest(name=name):
                torch.manual_seed(19)
                gaze = torch.randn(1, 4, 2, requires_grad=True)
                gaze_mask = torch.tensor([[0, 1, 1, 0]])
                with patch(
                    "va_gaze.models.regression.AutoModel.from_pretrained",
                    return_value=encoder_factory(),
                ):
                    model = GazeConcatForSequenceRegression(
                        checkpoint="unused",
                        tokenizer=TinyTokenizer(),
                        features_used=[1, 1, 0, 0, 0],
                        load_fixation_model=False,
                        fp_dropout=(0.0, 0.0),
                    )
                model._compute_fixations_batch = lambda *_: (gaze, gaze_mask)
                model.eval()
                outputs = model(
                    input_ids=torch.tensor([token_ids]),
                    attention_mask=torch.ones(1, 4, dtype=torch.long),
                    output_hidden_states=True,
                )
                cls_weights = torch.arange(
                    1,
                    9,
                    dtype=outputs.logits.dtype,
                ).reshape(1, -1)
                cls_probe = (outputs.hidden_states[-1][:, 0, :] * cls_weights).sum()
                cls_gradient = torch.autograd.grad(cls_probe, gaze, retain_graph=True)[0]
                logit_gradient = torch.autograd.grad(outputs.logits.sum(), gaze)[0]
                self.assertEqual(model.concat_order, "postfix")
                for gradient in (cls_gradient, logit_gradient):
                    self.assertGreater(float(gradient[:, 1:3].abs().sum()), 0.0)
                    self.assertEqual(int(torch.count_nonzero(gradient[:, [0, 3]])), 0)

    def test_concat_order_defaults_and_aliases_are_unambiguous(self):
        for value in (None, "concat", "postfix", "postfix-concat", "text-prefix"):
            with self.subTest(value=value):
                self.assertEqual(normalize_concat_order(value), "postfix")
        for value in ("prefix", "prefix-concat", "gaze-prefix"):
            with self.subTest(value=value):
                self.assertEqual(normalize_concat_order(value), "prefix")

    def test_optional_text_ids_must_match_the_original_text_shape(self):
        with self.assertRaisesRegex(ValueError, "token_type_ids"):
            compose_gaze_concat_inputs(
                text_embeddings=self.text,
                gaze_embeddings=self.gaze,
                text_attention_mask=self.text_mask,
                gaze_attention_mask=self.gaze_mask,
                eye_start=self.eye_start,
                eye_end=self.eye_end,
                token_type_ids=torch.zeros(1, 2, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "position_ids"):
            compose_gaze_concat_inputs(
                text_embeddings=self.text,
                gaze_embeddings=self.gaze,
                text_attention_mask=self.text_mask,
                gaze_attention_mask=self.gaze_mask,
                eye_start=self.eye_start,
                eye_end=self.eye_end,
                position_ids=torch.zeros(1, 2, dtype=torch.long),
            )


class GazeConcatModelContractTest(unittest.TestCase):
    def _build_model(
        self,
        output_dim=2,
        concat_order="postfix",
        features_used=None,
        encoder_factory=tiny_encoder,
    ):
        with patch(
            "va_gaze.models.regression.AutoModel.from_pretrained",
            return_value=encoder_factory(),
        ):
            return GazeConcatForSequenceRegression(
                checkpoint="unused",
                tokenizer=TinyTokenizer(),
                features_used=features_used or [1, 1, 0, 0, 0],
                load_fixation_model=False,
                fp_dropout=(0.0, 0.0),
                output_dim=output_dim,
                concat_order=concat_order,
            )

    def test_changing_postfix_gaze_changes_cls_and_logits(self):
        torch.manual_seed(23)
        model = self._build_model()
        model.eval()
        input_ids = torch.tensor([[1, 5, 6, 2]])
        attention_mask = torch.ones_like(input_ids)
        gaze_mask = torch.ones_like(input_ids)

        model._compute_fixations_batch = lambda *_: (
            torch.zeros(1, 4, 2),
            gaze_mask,
        )
        first = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        model._compute_fixations_batch = lambda *_: (
            torch.full((1, 4, 2), 3.0),
            gaze_mask,
        )
        second = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        self.assertFalse(
            torch.allclose(first.hidden_states[-1][:, 0], second.hidden_states[-1][:, 0])
        )
        self.assertFalse(torch.allclose(first.logits, second.logits))

    def test_nonfinite_gaze_is_masked_and_zeroed(self):
        model = self._build_model()
        model.fp_model = NonFinitePredictor()
        fixations, mask = model._compute_fixations_batch(
            torch.tensor([[1, 5, 2, 0]]),
            torch.tensor([[1, 1, 1, 0]]),
        )
        self.assertTrue(torch.isfinite(fixations).all())
        self.assertEqual(mask.tolist(), [[1, 0, 0, 0]])
        torch.testing.assert_close(fixations[:, 1], torch.zeros(1, 2))
        torch.testing.assert_close(fixations[:, 2], torch.zeros(1, 2))
        logits = model(
            input_ids=torch.tensor([[1, 5, 2, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        ).logits
        self.assertTrue(torch.isfinite(logits).all())

    def test_nonfinite_disabled_feature_does_not_mask_selected_feature(self):
        model = self._build_model(features_used=[0, 1, 0, 0, 0])
        model.fp_model = DisabledFeatureNonFinitePredictor()
        fixations, mask = model._compute_fixations_batch(
            torch.tensor([[1, 5, 2, 0]]),
            torch.tensor([[1, 1, 1, 0]]),
        )
        self.assertEqual(mask.tolist(), [[1, 1, 1, 0]])
        torch.testing.assert_close(fixations[:, :3], torch.ones(1, 3, 1))

    def test_no_predictor_and_all_masked_gaze_are_safe(self):
        model = self._build_model()
        model.eval()
        input_ids = torch.tensor([[1, 5, 6, 2]])
        attention_mask = torch.ones_like(input_ids)
        fixations, mask = model._compute_fixations_batch(input_ids, attention_mask)
        self.assertEqual(int(torch.count_nonzero(fixations)), 0)
        self.assertEqual(int(torch.count_nonzero(mask)), 0)

        model._compute_fixations_batch = lambda *_: (
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, dtype=torch.long),
        )
        first = model(input_ids=input_ids, attention_mask=attention_mask).logits
        model._compute_fixations_batch = lambda *_: (
            torch.full((1, 4, 2), 1000.0),
            torch.zeros(1, 4, dtype=torch.long),
        )
        second = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())

    def test_position_capacity_is_checked_before_encoder_forward(self):
        model = self._build_model()
        input_ids = torch.ones(1, 16, dtype=torch.long)
        model._compute_fixations_batch = lambda *_: (
            torch.zeros(1, 16, 2),
            torch.ones(1, 16, dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "position limit"):
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def test_prediction_is_invariant_to_other_samples_padding_length(self):
        def deterministic_gaze(input_ids, attention_mask):
            values = input_ids.to(dtype=torch.float32)
            return (
                torch.stack((values / 10.0, values / 20.0), dim=-1),
                attention_mask,
            )

        cases = (
            (
                "distilbert",
                tiny_distilbert_encoder,
                [1, 5, 6, 2],
                [1, 5, 6, 2, 0, 0],
                [1, 7, 8, 9, 10, 2],
            ),
            (
                "xlm-roberta",
                tiny_xlm_roberta_encoder,
                [0, 5, 6, 2],
                [0, 5, 6, 2, 1, 1],
                [0, 7, 8, 9, 10, 2],
            ),
        )
        batch_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        for name, encoder_factory, single_row, padded_row, long_row in cases:
            for concat_order in ("postfix", "prefix"):
                with self.subTest(name=name, concat_order=concat_order):
                    model = self._build_model(
                        concat_order=concat_order,
                        encoder_factory=encoder_factory,
                    )
                    model._compute_fixations_batch = deterministic_gaze
                    model.eval()
                    single_ids = torch.tensor([single_row])
                    single_logits = model(
                        input_ids=single_ids,
                        attention_mask=torch.ones_like(single_ids),
                    ).logits
                    batch_logits = model(
                        input_ids=torch.tensor([padded_row, long_row]),
                        attention_mask=batch_mask,
                    ).logits[:1]
                    torch.testing.assert_close(
                        single_logits,
                        batch_logits,
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_xlm_roberta_padding_offset_is_included_in_position_limit(self):
        model = self._build_model(
            encoder_factory=lambda: tiny_xlm_roberta_encoder(
                max_position_embeddings=12
            )
        )
        input_ids = torch.tensor([[0, 5, 6, 7, 8, 2]])
        model._compute_fixations_batch = lambda *_: (
            torch.ones(1, 6, 2),
            torch.ones(1, 6, dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "14 > 10"):
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def test_heteroscedastic_postfix_output_contract(self):
        model = self._build_model(output_dim=4)
        input_ids = torch.tensor([[1, 5, 6, 2]])
        model._compute_fixations_batch = lambda *_: (
            torch.ones(1, 4, 2),
            torch.ones(1, 4, dtype=torch.long),
        )
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertTrue(torch.all((logits[:, :2] >= 0.0) & (logits[:, :2] <= 1.0)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_inputs_embeds_is_rejected_instead_of_silently_ignored(self):
        model = self._build_model()
        with self.assertRaisesRegex(ValueError, "inputs_embeds is not supported"):
            model(
                input_ids=torch.tensor([[1, 5, 2]]),
                inputs_embeds=torch.zeros(1, 3, 8),
            )

    def test_non_concat_subclass_does_not_advertise_concat_metadata(self):
        with patch(
            "va_gaze.models.regression.AutoModel.from_pretrained",
            return_value=tiny_encoder(),
        ):
            model = GazeAddForSequenceRegression(
                checkpoint="unused",
                tokenizer=TinyTokenizer(),
                features_used=[1, 0, 0, 0, 0],
                gaze_add_scale=0.0,
                train_gaze_add_scale=False,
            )
        self.assertFalse(hasattr(model.config, "gaze_concat_order"))


if __name__ == "__main__":
    unittest.main()
