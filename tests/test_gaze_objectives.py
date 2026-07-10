import unittest

import torch

from va_gaze.models.gaze.objectives import MaskedGazePrediction, TokenInfoNCEAlignment
from va_gaze.models.gaze.types import GazeBatch


class GazeObjectiveTest(unittest.TestCase):
    def test_prediction_ignores_invalid_targets(self):
        torch.manual_seed(5)
        objective = MaskedGazePrediction(6, 2, dropout=0.0)
        objective.eval()
        text = torch.randn(1, 4, 6, requires_grad=True)
        mask = torch.tensor([[False, True, False, True]])
        features = torch.randn(1, 4, 2)
        batch = GazeBatch(features, mask, torch.ones_like(mask))
        expected = objective(text, batch)
        changed = features.clone()
        changed[~mask] = 9999.0
        actual = objective(text, GazeBatch(changed, mask, torch.ones_like(mask)))
        torch.testing.assert_close(actual, expected)

    def test_prediction_empty_mask_returns_differentiable_zero(self):
        objective = MaskedGazePrediction(4, 2, dropout=0.0)
        text = torch.randn(1, 3, 4, requires_grad=True)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        loss = objective(text, GazeBatch(torch.zeros(1, 3, 2), mask, ~mask))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(text.grad)

    def test_alignment_rewards_matched_pairs(self):
        objective = TokenInfoNCEAlignment(
            hidden_size=3,
            gaze_dim=3,
            alignment_dim=3,
            temperature=0.1,
        )
        with torch.no_grad():
            objective.text_projector.weight.copy_(torch.eye(3))
            objective.gaze_projector.weight.copy_(torch.eye(3))
        text = torch.eye(3).unsqueeze(0)
        mask = torch.ones(1, 3, dtype=torch.bool)
        matched = GazeBatch(torch.eye(3).unsqueeze(0), mask, mask)
        permuted = GazeBatch(torch.eye(3)[torch.tensor([1, 2, 0])].unsqueeze(0), mask, mask)
        matched_loss = float(objective(text, matched).detach())
        permuted_loss = float(objective(text, permuted).detach())
        self.assertLess(matched_loss, permuted_loss)

    def test_alignment_with_one_pair_is_safe(self):
        objective = TokenInfoNCEAlignment(3, 3, alignment_dim=3)
        text = torch.randn(1, 2, 3, requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = objective(text, GazeBatch(torch.randn(1, 2, 3), mask, torch.ones_like(mask)))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(text.grad)


if __name__ == "__main__":
    unittest.main()
