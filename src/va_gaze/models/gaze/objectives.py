import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_feature_vector(value, gaze_dim, name):
    if value is None:
        return torch.empty(0, dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != int(gaze_dim):
        raise ValueError(f"{name} must contain exactly {gaze_dim} values.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


class MaskedGazePrediction(nn.Module):
    """Training-only token gaze regression with batch-independent targets.

    ``signed-log1p`` is the default because raw duration/count features can span
    very different positive ranges, while PCA features can be signed. Optional
    mean/scale values must be fixed statistics computed outside the minibatch
    (normally on the training fold) so a token's target never depends on which
    other examples happen to share its batch.
    """

    def __init__(
        self,
        hidden_size,
        gaze_dim,
        dropout=0.1,
        target_transform="signed-log1p",
        target_mean=None,
        target_scale=None,
    ):
        super().__init__()
        if target_transform not in ("raw", "signed-log1p"):
            raise ValueError("target_transform must be 'raw' or 'signed-log1p'.")
        if (target_mean is None) != (target_scale is None):
            raise ValueError("target_mean and target_scale must be provided together.")

        fixed_mean = _as_feature_vector(target_mean, gaze_dim, "target_mean")
        fixed_scale = _as_feature_vector(target_scale, gaze_dim, "target_scale")
        if fixed_scale.numel() and not fixed_scale.gt(0).all():
            raise ValueError("target_scale values must all be > 0.")

        self.target_transform = target_transform
        self.register_buffer("target_mean", fixed_mean, persistent=True)
        self.register_buffer("target_scale", fixed_scale, persistent=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, gaze_dim),
        )

    def transform_targets(self, features):
        targets = features.detach().to(dtype=torch.float32)
        if self.target_transform == "signed-log1p":
            targets = torch.sign(targets) * torch.log1p(targets.abs())
        if self.target_mean.numel():
            targets = (targets - self.target_mean) / self.target_scale
        return targets

    def forward(self, text_states, gaze_batch):
        valid = gaze_batch.valid_mask
        if not valid.any():
            return text_states.sum() * 0.0
        predictions = self.predictor(text_states)
        targets = self.transform_targets(gaze_batch.features)
        expanded_valid = valid.unsqueeze(-1).expand_as(predictions)
        return F.smooth_l1_loss(
            predictions[expanded_valid].float(),
            targets[expanded_valid],
            reduction="mean",
        )


class TokenInfoNCEAlignment(nn.Module):
    """Symmetric token-level contrastive alignment between text and gaze."""

    def __init__(
        self,
        hidden_size,
        gaze_dim,
        alignment_dim=128,
        temperature=0.07,
        max_tokens=512,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("alignment temperature must be > 0.")
        if max_tokens <= 0:
            raise ValueError("alignment max_tokens must be > 0.")
        self.text_projector = nn.Linear(hidden_size, alignment_dim, bias=False)
        self.gaze_projector = nn.Linear(gaze_dim, alignment_dim, bias=False)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def forward(self, text_states, gaze_batch):
        valid = gaze_batch.valid_mask
        text_valid = text_states[valid]
        gaze_valid = gaze_batch.features[valid].detach()
        pair_count = text_valid.shape[0]
        if pair_count < 2:
            return text_states.sum() * 0.0

        if pair_count > self.max_tokens:
            indices = torch.linspace(
                0,
                pair_count - 1,
                steps=self.max_tokens,
                device=text_states.device,
            ).round().long()
            text_valid = text_valid.index_select(0, indices)
            gaze_valid = gaze_valid.index_select(0, indices)

        text_projected = F.normalize(self.text_projector(text_valid), dim=-1)
        gaze_projected = F.normalize(self.gaze_projector(gaze_valid), dim=-1)
        similarities = text_projected @ gaze_projected.transpose(0, 1)
        similarities = similarities / self.temperature
        targets = torch.arange(similarities.shape[0], device=similarities.device)
        text_to_gaze = F.cross_entropy(similarities, targets)
        gaze_to_text = F.cross_entropy(similarities.transpose(0, 1), targets)
        return 0.5 * (text_to_gaze + gaze_to_text)
