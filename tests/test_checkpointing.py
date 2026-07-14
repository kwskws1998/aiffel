import unittest

from va_gaze.models.checkpointing import EncoderGradientCheckpointingMixin


class FakeEncoder:
    def __init__(self):
        self.enabled_kwargs = None
        self.disabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enabled_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.disabled = True


class WrappedEncoder(EncoderGradientCheckpointingMixin):
    def __init__(self, encoder):
        self.encoder = encoder


class EncoderGradientCheckpointingMixinTest(unittest.TestCase):
    def test_delegates_enable_and_disable_to_encoder(self):
        encoder = FakeEncoder()
        model = WrappedEncoder(encoder)
        model.gradient_checkpointing_enable({"use_reentrant": False})
        model.gradient_checkpointing_disable()
        self.assertEqual(encoder.enabled_kwargs, {"use_reentrant": False})
        self.assertTrue(encoder.disabled)


if __name__ == "__main__":
    unittest.main()
