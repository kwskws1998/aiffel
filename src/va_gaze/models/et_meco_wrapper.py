import os
import re
import sys

import numpy as np
import torch


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
        remapped = self._remap_to_rm_tokens(word_features, words, ids, mask)
        fixations = remapped.unsqueeze(0).to(input_ids_rm.device)
        fix_attn = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(input_ids_rm.device)
        return fixations, fix_attn, None, None, None, None

    def _remap_to_rm_tokens(self, word_features, words, rm_ids, rm_mask):
        seq_len = len(rm_ids)
        output = torch.zeros(seq_len, self.feature_dim, dtype=torch.float32)
        if len(word_features) == 0 or len(words) == 0:
            return output

        rm_tokens = self.rm_tokenizer.convert_ids_to_tokens(rm_ids)
        word_to_rm = _align_words_to_rm_tokens(words, rm_tokens, self.rm_tokenizer)

        n_words = min(len(words), len(word_features))
        for w_idx in range(n_words):
            if w_idx >= len(word_to_rm):
                break
            indices = word_to_rm[w_idx]
            if not indices:
                continue
            first = indices[0]
            if first < seq_len and rm_mask[first] == 1:
                output[first] = torch.tensor(np.asarray(word_features[w_idx]), dtype=torch.float32)
        return output


def _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer):
    special_ids = set(rm_tokenizer.all_special_ids)
    word_to_indices = []
    tok_idx = 0

    for word in words:
        indices = []
        chars_remaining = len(_normalize_for_alignment(word))
        while tok_idx < len(rm_tokens) and chars_remaining > 0:
            tok = rm_tokens[tok_idx]
            tok_id = rm_tokenizer.convert_tokens_to_ids(tok)
            if tok_id in special_ids:
                tok_idx += 1
                continue

            tok_clean = _normalize_for_alignment(tok.lstrip("Ġ▁ "))
            if tok_clean:
                indices.append(tok_idx)
                chars_remaining -= len(tok_clean)
            tok_idx += 1
        word_to_indices.append(indices)
    return word_to_indices


def _normalize_for_alignment(text):
    return re.sub(r"\s+", "", str(text))
