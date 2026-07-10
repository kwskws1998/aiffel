"""Deterministic token gaze features for offline smoke tests only."""

import torch


class HeuristicFixationsPredictor:
    """Expose the ET predictor contract without checkpoints or network access."""

    feature_dim = 5
    feature_names = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
    is_smoke_only = True

    def __init__(self, modelTokenizer):
        self.rm_tokenizer = modelTokenizer
        self.special_ids = set(getattr(modelTokenizer, "all_special_ids", []) or [])

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)

        valid = attention_mask_rm.to(dtype=torch.bool)
        if self.special_ids:
            special = torch.zeros_like(valid)
            for token_id in self.special_ids:
                special |= input_ids_rm.eq(int(token_id))
            valid &= ~special

        token_value = input_ids_rm.to(dtype=torch.float32)
        position = torch.arange(
            input_ids_rm.shape[1],
            device=input_ids_rm.device,
            dtype=torch.float32,
        ).view(1, -1)
        position = position.expand_as(token_value)
        denom = max(input_ids_rm.shape[1] - 1, 1)

        nfix = 1.0 + token_value.remainder(3.0)
        ffd = (token_value.remainder(17.0) + 1.0) / 17.0
        gpt = (position + 1.0) / float(denom + 1)
        trt = ffd * (1.0 + gpt)
        fixprop = torch.ones_like(token_value)
        features = torch.stack((nfix, ffd, gpt, trt, fixprop), dim=-1)
        features = features * valid.unsqueeze(-1).to(dtype=features.dtype)

        mapped_mask = valid.to(dtype=attention_mask_rm.dtype)
        return features, mapped_mask, None, None, None, None
