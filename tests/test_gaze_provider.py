import unittest

import torch
import torch.nn as nn

from va_gaze.models.gaze.provider import GazeFeatureProvider


class DummyTokenizer:
    all_special_ids = [0, 1, 2]


class FakePredictor:
    def __init__(self, features):
        self.features = features
        self.model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))

    def _compute_mapped_fixations(self, input_ids, attention_mask):
        features = self.features[:, : input_ids.shape[1]].to(input_ids.device)
        return features, attention_mask, None, None, None, None


class ProviderContainer(nn.Module):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider


class GazeProviderTest(unittest.TestCase):
    def test_heuristic_provider_is_lazy_and_masks_special_tokens(self):
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="heuristic",
            features_used=[1, 1, 1, 1, 1],
        )
        self.assertIsNone(provider.fp_model)
        input_ids = torch.tensor([[1, 7, 8, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
        batch = provider.compute(input_ids, attention_mask)
        self.assertIsNotNone(provider.fp_model)
        self.assertEqual(batch.mapped_mask.tolist(), [[False, True, True, False, False]])
        self.assertTrue(batch.features[~batch.mapped_mask].eq(0).all())

    def test_mapped_mask_is_derived_before_feature_selection(self):
        raw = torch.zeros(1, 4, 5)
        raw[0, 1, 0] = 3.0
        raw[0, 2, 3] = 4.0
        predictor = FakePredictor(raw)
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="heuristic",
            features_used=[0, 0, 0, 1, 0],
        )
        provider._freeze_predictor(predictor)
        provider.fp_model = predictor
        batch = provider.compute(
            torch.tensor([[1, 7, 8, 2]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(batch.mapped_mask.tolist(), [[False, True, True, False]])
        self.assertEqual(float(batch.features[0, 1, 0]), 0.0)
        self.assertEqual(float(batch.features[0, 2, 0]), 4.0)

    def test_frozen_predictor_stays_in_eval_when_parent_trains(self):
        raw = torch.zeros(1, 3, 5)
        predictor = FakePredictor(raw)
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="heuristic",
        )
        provider._freeze_predictor(predictor)
        provider.fp_model = predictor
        container = ProviderContainer(provider)
        container.train()
        self.assertFalse(predictor.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in predictor.model.parameters()))

    def test_nonfinite_predictor_values_are_masked_and_zeroed(self):
        raw = torch.zeros(1, 4, 5)
        raw[0, 1, 0] = float("nan")
        raw[0, 2, 0] = float("inf")
        raw[0, 3, 0] = 2.0
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="heuristic",
        )
        provider.fp_model = FakePredictor(raw)
        batch = provider.compute(
            torch.tensor([[1, 7, 8, 2]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(batch.mapped_mask.tolist(), [[False, False, False, True]])
        self.assertTrue(torch.isfinite(batch.features).all())
        self.assertTrue(batch.features[~batch.mapped_mask].eq(0).all())

    def test_noncontiguous_attention_mask_is_rejected(self):
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="heuristic",
        )
        with self.assertRaisesRegex(ValueError, "contiguous right-padded"):
            provider.compute(
                torch.tensor([[0, 1, 7, 2]]),
                torch.tensor([[0, 1, 1, 1]]),
            )

    def test_emotion_trt_provider_accepts_only_trt_feature(self):
        provider = GazeFeatureProvider(
            tokenizer=DummyTokenizer(),
            et_model_type="emotion_trt",
            features_used=[0, 0, 0, 1, 0],
            load_fixation_model=False,
        )
        self.assertEqual(provider.et_model_type, "emotion-trt")
        self.assertEqual(provider.feature_indices, [3])
        self.assertEqual(provider.feature_dim, 1)

        with self.assertRaisesRegex(ValueError, "does not predict: FFD"):
            GazeFeatureProvider(
                tokenizer=DummyTokenizer(),
                et_model_type="emotion-trt",
                features_used=[0, 1, 0, 1, 0],
                load_fixation_model=False,
            )


if __name__ == "__main__":
    unittest.main()
