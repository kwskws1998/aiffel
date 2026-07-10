import unittest

import torch

from va_gaze.models.gaze.fusion import (
    GazeBiasedClsAttention,
    GazeConditionedPooling,
    GazeCrossAttention,
    build_gaze_fusion,
)
from va_gaze.models.gaze.types import GazeBatch


class GazeFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.cls_state = torch.randn(2, 8, requires_grad=True)
        self.text_states = torch.randn(2, 5, 8, requires_grad=True)
        self.features = torch.randn(2, 5, 3, requires_grad=True)
        self.mapped_mask = torch.tensor(
            [[False, True, True, False, False], [False, True, False, True, False]]
        )
        self.text_mask = torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        )

    def _batch(self, features=None, mapped_mask=None):
        return GazeBatch(
            features=self.features if features is None else features,
            mapped_mask=self.mapped_mask if mapped_mask is None else mapped_mask,
            text_mask=self.text_mask,
        )

    def _modules(self):
        return (
            GazeConditionedPooling(8, 3, gate_init=0.0, dropout=0.0),
            GazeBiasedClsAttention(
                8,
                3,
                num_heads=2,
                attention_scale=0.5,
                gate_init=0.0,
                dropout=0.0,
            ),
            GazeCrossAttention(
                8,
                3,
                gaze_hidden_size=8,
                num_heads=2,
                num_layers=1,
                gate_init=0.0,
                dropout=0.0,
            ),
        )

    def test_shape_finiteness_and_gaze_gradient(self):
        for module in self._modules():
            with self.subTest(module=type(module).__name__):
                module.eval()
                features = self.features.detach().clone().requires_grad_(True)
                output = module(self.cls_state, self.text_states, self._batch(features=features))
                self.assertEqual(tuple(output.shape), (2, 8))
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward(retain_graph=True)
                self.assertIsNotNone(features.grad)
                self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_invalid_gaze_values_are_ignored(self):
        changed = self.features.detach().clone()
        changed[~self.mapped_mask] = 10000.0
        for module in self._modules():
            with self.subTest(module=type(module).__name__):
                module.eval()
                expected = module(self.cls_state, self.text_states, self._batch())
                actual = module(self.cls_state, self.text_states, self._batch(features=changed))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_all_missing_gaze_returns_cls_exactly(self):
        empty_mask = torch.zeros_like(self.mapped_mask)
        for module in self._modules():
            with self.subTest(module=type(module).__name__):
                module.eval()
                actual = module(
                    self.cls_state,
                    self.text_states,
                    self._batch(mapped_mask=empty_mask),
                )
                torch.testing.assert_close(actual, self.cls_state, rtol=0.0, atol=0.0)

    def test_postencoder_attention_name_and_legacy_aliases_match(self):
        modules = [
            build_gaze_fusion(name, hidden_size=8, gaze_dim=3, num_heads=2)
            for name in (
                "postencoder-cls-attention-bias",
                "cls-attention-bias",
                "attention-bias",
            )
        ]
        self.assertTrue(all(isinstance(module, GazeBiasedClsAttention) for module in modules))


if __name__ == "__main__":
    unittest.main()
