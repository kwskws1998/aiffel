import os
import sys

import torch

from va_gaze.models.gaze.alignment import remap_word_features_to_tokens


DEFAULT_ET_MECO_ROOT = "/Users/wansookim/Documents/meco_data/et_meco"
if os.path.isdir(DEFAULT_ET_MECO_ROOT) and DEFAULT_ET_MECO_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_ET_MECO_ROOT)
extra_root = os.environ.get("ET_MECO_PACKAGE_ROOT")
if extra_root and os.path.isdir(extra_root) and extra_root not in sys.path:
    sys.path.insert(0, extra_root)

from et_meco.inference import MecoETPredictor  # noqa: E402


class MecoFixationsPredictor:
    def __init__(self, modelTokenizer, checkpoint_path, max_length=512, device=None):
        if not checkpoint_path:
            checkpoint_path = os.environ.get("ET_MECO_MODEL_ID")
        if not checkpoint_path:
            raise ValueError("et-meco requires --et-model-id or ET_MECO_MODEL_ID.")
        self.rm_tokenizer = modelTokenizer
        self.predictor = MecoETPredictor(checkpoint_path, device=device, max_length=max_length)
        self.feature_names = self.predictor.feature_names
        self.feature_dim = self.predictor.feature_dim
        print(f"[et_meco_wrapper] MecoFixationsPredictor loaded: {checkpoint_path}")

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)

        ids = input_ids_rm[0].detach().cpu().tolist()
        mask = attention_mask_rm[0].detach().cpu().tolist()
        pad_id = self.rm_tokenizer.pad_token_id or 0
        ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
        text = self.rm_tokenizer.decode(ids_no_pad, skip_special_tokens=True)

        word_features, words = self.predictor.predict_text(text)
        remapped, mapped_mask = self._remap_to_rm_tokens(word_features, words, ids, mask)
        fixations = remapped.unsqueeze(0).to(input_ids_rm.device)
        fix_attn = mapped_mask.to(dtype=torch.long).unsqueeze(0).to(input_ids_rm.device)
        return fixations, fix_attn, None, None, None, None

    def _remap_to_rm_tokens(self, word_features, words, rm_ids, rm_mask):
        return remap_word_features_to_tokens(
            word_features=word_features,
            words=words,
            token_ids=rm_ids,
            attention_mask=rm_mask,
            tokenizer=self.rm_tokenizer,
            feature_dim=self.feature_dim,
        )
