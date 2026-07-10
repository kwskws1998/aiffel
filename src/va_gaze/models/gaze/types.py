from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GazeBatch:
    """Token-aligned gaze features and masks produced by a gaze provider."""

    features: torch.Tensor
    mapped_mask: torch.Tensor
    text_mask: torch.Tensor
    scanpath_indices: Optional[torch.Tensor] = None

    def to(self, device=None, dtype=None):
        features = self.features.to(
            device=device if device is not None else self.features.device,
            dtype=dtype if dtype is not None else self.features.dtype,
        )
        return GazeBatch(
            features=features,
            mapped_mask=self.mapped_mask.to(device=features.device, dtype=torch.bool),
            text_mask=self.text_mask.to(device=features.device, dtype=torch.bool),
            scanpath_indices=(
                None
                if self.scanpath_indices is None
                else self.scanpath_indices.to(device=features.device)
            ),
        )

    @property
    def valid_mask(self):
        return self.mapped_mask.to(dtype=torch.bool) & self.text_mask.to(dtype=torch.bool)

    @property
    def has_gaze(self):
        return self.valid_mask.any(dim=1)
