import tempfile
import unittest
from pathlib import Path

import torch
from transformers import TrainingArguments

from va_gaze.train.custom_trainer import CustomTrainerMSE


class ManifestAwareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, input_ids=None, **kwargs):
        del kwargs
        return {"logits": self.projection(input_ids.float())}

    def save_architecture_manifest(self, output_dir):
        Path(output_dir, "manifest-called.txt").write_text("ok\n", encoding="utf-8")


class TrainerPersistenceTest(unittest.TestCase):
    def test_every_trainer_save_invokes_model_manifest_hook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "saved"
            arguments = TrainingArguments(
                output_dir=str(Path(temp_dir) / "trainer"),
                report_to=[],
            )
            trainer = CustomTrainerMSE(
                model=ManifestAwareModel(),
                args=arguments,
            )
            trainer.save_model(output_dir)
            self.assertTrue((output_dir / "model.safetensors").is_file())
            self.assertEqual(
                (output_dir / "manifest-called.txt").read_text(encoding="utf-8"),
                "ok\n",
            )


if __name__ == "__main__":
    unittest.main()
