import unittest

import numpy as np
import torch

from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor


class _WordBatch(dict):
    def word_ids(self, batch_index=0):
        return [None, 0, 1, None]


class _WordTokenizer:
    def __call__(self, words, **kwargs):
        return _WordBatch(
            input_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
        )


class _BFloat16EmotionEtModel:
    def __call__(self, input_ids, attention_mask):
        return torch.tensor(
            [
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                ]
            ],
            dtype=torch.bfloat16,
        )


class EmotionEtWrapperTest(unittest.TestCase):
    def test_trt_only_output_is_placed_in_shared_trt_slot(self):
        predictor = object.__new__(EmotionEtFixationsPredictor)
        predictor.native_feature_dim = 1
        expanded = predictor._expand_to_full_schema(
            np.asarray([[1.5], [2.5]], dtype=np.float32)
        )
        self.assertEqual(expanded.shape, (2, 5))
        np.testing.assert_allclose(expanded[:, 3], [1.5, 2.5])
        np.testing.assert_allclose(expanded[:, [0, 1, 2, 4]], 0.0)

    def test_five_feature_output_is_unchanged(self):
        predictor = object.__new__(EmotionEtFixationsPredictor)
        predictor.native_feature_dim = 5
        native = np.arange(10, dtype=np.float32).reshape(2, 5)
        self.assertIs(predictor._expand_to_full_schema(native), native)

    def test_bfloat16_predictions_are_converted_to_float32_numpy(self):
        predictor = object.__new__(EmotionEtFixationsPredictor)
        predictor.feature_dim = 5
        predictor.native_feature_dim = 5
        predictor.max_length = 512
        predictor.device = torch.device("cpu")
        predictor.roberta_tokenizer = _WordTokenizer()
        predictor.model = _BFloat16EmotionEtModel()

        predictions, words = predictor._predict_words("hello world")

        self.assertEqual(words, ["hello", "world"])
        self.assertEqual(predictions.dtype, np.float32)
        np.testing.assert_allclose(
            predictions,
            np.asarray(
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
