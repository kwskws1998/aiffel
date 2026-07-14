import unittest

import numpy as np
import torch

from va_gaze.models.et2_wrapper import FixationsPredictor_2, WINDOW_SIZE


class _BFloat16Et2Model:
    def __call__(self, input_ids, attention_mask, predict_mask):
        batch_size, sequence_length = input_ids.shape
        return torch.full(
            (batch_size, sequence_length, 5),
            1.5,
            dtype=torch.bfloat16,
            device=input_ids.device,
        )


class Et2WrapperTest(unittest.TestCase):
    def test_bfloat16_predictions_are_converted_for_numpy_in_all_window_paths(self):
        predictor = object.__new__(FixationsPredictor_2)
        predictor.model = _BFloat16Et2Model()

        for sequence_length in (17, WINDOW_SIZE + 17):
            with self.subTest(sequence_length=sequence_length):
                input_ids = torch.ones((1, sequence_length), dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)

                predictions = predictor._sliding_window_predict(
                    input_ids,
                    attention_mask,
                )

                self.assertEqual(predictions.shape, (sequence_length, 5))
                self.assertEqual(predictions.dtype, np.float32)
                np.testing.assert_allclose(predictions, 1.5)


if __name__ == "__main__":
    unittest.main()
