from collections import OrderedDict

import torch

from va_gaze.models.gaze.types import GazeBatch
from va_gaze.models.gaze_transform import GazeFeatureTransformer


def normalize_et_model_type(raw_value):
    aliases = {
        "emotion_et": "emotion-et",
        "et_meco": "et-meco",
        "legacy-et2": "et2",
        "smoke": "heuristic",
    }
    return aliases.get(raw_value or "et2", raw_value or "et2")


class GazeFeatureProvider:
    """Lazy, frozen provider for token-aligned gaze features."""

    def __init__(
        self,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        max_fix_cache_size=20000,
        load_fixation_model=True,
        et_model_type="et2",
        et_model_id=None,
        gaze_transform="raw",
        gaze_artifact_dir=None,
        pca_components=2,
        gmm_components=5,
    ):
        self.tokenizer = tokenizer
        self.et_model_type = normalize_et_model_type(et_model_type)
        self.et2_checkpoint_path = et2_checkpoint_path
        self.et_model_id = et_model_id
        self.load_fixation_model = bool(load_fixation_model)
        self.fp_model = None
        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = int(max_fix_cache_size)

        if self.et_model_type in ("et2", "emotion-et", "heuristic"):
            flags = features_used or [1, 1, 1, 1, 1]
            self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
            if not self.feature_indices:
                raise ValueError("features_used must enable at least one ET feature.")
            self.raw_feature_dim = len(self.feature_indices)
        elif self.et_model_type == "et-meco":
            self.feature_indices = None
            self.raw_feature_dim = 8
        else:
            raise ValueError(f"Unknown et_model_type: {self.et_model_type}")

        self.gaze_feature_transformer = GazeFeatureTransformer(
            transform=gaze_transform or "raw",
            raw_feature_dim=self.raw_feature_dim,
            artifact_dir=gaze_artifact_dir,
            artifact_repo_id=et_model_id,
            pca_components=pca_components,
            gmm_components=gmm_components,
        )
        self.feature_dim = self.gaze_feature_transformer.output_dim

    def _load_predictor(self):
        if self.fp_model is not None or not self.load_fixation_model:
            return self.fp_model

        if self.et_model_type == "et2":
            from va_gaze.models.et2_wrapper import FixationsPredictor_2

            predictor = FixationsPredictor_2(
                modelTokenizer=self.tokenizer,
                remap=False,
                checkpoint_path=self.et2_checkpoint_path,
            )
        elif self.et_model_type == "emotion-et":
            from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor

            predictor = EmotionEtFixationsPredictor(
                modelTokenizer=self.tokenizer,
                model_id=self.et_model_id or self.et2_checkpoint_path,
            )
        elif self.et_model_type == "et-meco":
            from va_gaze.models.et_meco_wrapper import MecoFixationsPredictor

            predictor = MecoFixationsPredictor(
                modelTokenizer=self.tokenizer,
                checkpoint_path=self.et_model_id or self.et2_checkpoint_path,
            )
            predictor_dim = int(getattr(predictor, "feature_dim", self.raw_feature_dim))
            if predictor_dim != self.raw_feature_dim:
                raise ValueError(
                    f"ET-MECO predictor exposes {predictor_dim} features; expected {self.raw_feature_dim}."
                )
        else:
            from va_gaze.models.heuristic_et_wrapper import HeuristicFixationsPredictor

            predictor = HeuristicFixationsPredictor(modelTokenizer=self.tokenizer)

        self._freeze_predictor(predictor)
        self.fp_model = predictor
        return self.fp_model

    @staticmethod
    def _freeze_predictor(predictor):
        candidate_models = []
        if hasattr(predictor, "model"):
            candidate_models.append(predictor.model)
        if hasattr(predictor, "predictor") and hasattr(predictor.predictor, "model"):
            candidate_models.append(predictor.predictor.model)
        for model in candidate_models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _build_cache_key(token_ids_1d, attention_mask_1d):
        mask = attention_mask_1d.to(dtype=torch.bool)
        valid_len = int(mask.sum().item())
        if valid_len <= 0:
            return tuple(), valid_len
        if not mask[:valid_len].all() or mask[valid_len:].any():
            raise ValueError(
                "GazeFeatureProvider requires a contiguous right-padded attention mask."
            )
        return tuple(token_ids_1d[:valid_len].tolist()), valid_len

    def _empty_features(self, seq_len, device):
        return torch.zeros(seq_len, self.raw_feature_dim, dtype=torch.float32, device=device)

    def _predict_single(self, token_ids_1d, attention_mask_1d):
        device = token_ids_1d.device
        seq_len = token_ids_1d.shape[0]
        key, valid_len = self._build_cache_key(token_ids_1d, attention_mask_1d)
        if valid_len <= 0 or not self.load_fixation_model:
            return (
                self._empty_features(seq_len, device),
                torch.zeros(seq_len, dtype=torch.bool, device=device),
            )

        cached = self.fixation_cache.get(key)
        if cached is None:
            predictor = self._load_predictor()
            sample_ids = token_ids_1d[:valid_len].unsqueeze(0)
            sample_mask = attention_mask_1d[:valid_len].unsqueeze(0)
            with torch.inference_mode():
                raw_fixations, predictor_mask, _, _, _, _ = predictor._compute_mapped_fixations(
                    sample_ids,
                    sample_mask,
                )

            raw_fixations = raw_fixations.squeeze(0).float().cpu()
            predictor_mask = predictor_mask.squeeze(0).to(dtype=torch.bool).cpu()
            finite_mask = torch.isfinite(raw_fixations).all(dim=-1)
            mapped_mask = (
                predictor_mask
                & finite_mask
                & raw_fixations.abs().sum(dim=-1).gt(0)
            )
            raw_fixations = torch.nan_to_num(raw_fixations)
            if self.feature_indices is not None:
                raw_fixations = raw_fixations[:, self.feature_indices]

            if len(self.fixation_cache) >= self.max_fix_cache_size:
                self.fixation_cache.popitem(last=False)
            self.fixation_cache[key] = (raw_fixations, mapped_mask)
        else:
            raw_fixations, mapped_mask = cached
            self.fixation_cache.move_to_end(key)

        raw_fixations = raw_fixations.to(device=device)
        mapped_mask = mapped_mask.to(device=device, dtype=torch.bool)
        padded_fixations = self._empty_features(seq_len, device)
        padded_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        copy_len = min(valid_len, raw_fixations.shape[0], seq_len)
        padded_fixations[:copy_len] = raw_fixations[:copy_len]
        padded_mask[:copy_len] = mapped_mask[:copy_len]
        return padded_fixations, padded_mask

    def compute(self, input_ids, attention_mask):
        raw_rows = []
        mapped_rows = []
        for row_idx in range(input_ids.size(0)):
            row_features, row_mask = self._predict_single(
                input_ids[row_idx],
                attention_mask[row_idx],
            )
            raw_rows.append(row_features)
            mapped_rows.append(row_mask)

        raw_fixations = torch.stack(raw_rows, dim=0)
        mapped_mask = torch.stack(mapped_rows, dim=0)
        transformed = self.gaze_feature_transformer.transform_tensor(
            raw_fixations,
            mapped_mask,
        )
        transformed_finite = torch.isfinite(transformed).all(dim=-1)
        mapped_mask = mapped_mask & transformed_finite
        transformed = torch.nan_to_num(transformed)
        transformed = transformed.masked_fill(~mapped_mask.unsqueeze(-1), 0.0)
        return GazeBatch(
            features=transformed,
            mapped_mask=mapped_mask,
            text_mask=attention_mask.to(dtype=torch.bool),
        )
