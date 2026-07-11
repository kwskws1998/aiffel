import unittest

import numpy as np

from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor


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


if __name__ == "__main__":
    unittest.main()
