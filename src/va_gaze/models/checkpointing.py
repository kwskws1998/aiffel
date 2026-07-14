class EncoderGradientCheckpointingMixin:
    """Delegate Trainer gradient-checkpointing controls to the wrapped encoder."""

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not hasattr(self.encoder, "gradient_checkpointing_enable"):
            raise ValueError("The wrapped encoder does not support gradient checkpointing.")
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        if not hasattr(self.encoder, "gradient_checkpointing_disable"):
            raise ValueError("The wrapped encoder does not support gradient checkpointing.")
        self.encoder.gradient_checkpointing_disable()
