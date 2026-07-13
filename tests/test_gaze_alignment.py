import unittest

import numpy as np
import torch

from va_gaze.models.gaze.alignment import (
    align_words_to_tokens,
    remap_word_features_to_tokens,
)


class MarkerTokenizer:
    all_special_ids = [0, 101, 102]
    all_special_tokens = ["[PAD]", "[CLS]", "[SEP]"]

    def __init__(self, lexical_tokens):
        self.tokens = ["[CLS]", *lexical_tokens, "[SEP]", "[PAD]"]

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[int(token_id)] for token_id in token_ids]


class GazeAlignmentTest(unittest.TestCase):
    def test_common_subword_markers_align_to_exact_word_spans(self):
        marker_cases = (
            ("wordpiece", ["play", "##ing", "ball"]),
            ("roberta", ["Ġplay", "ing", "Ġball"]),
            ("sentencepiece", ["▁play", "ing", "▁ball"]),
        )
        token_ids = [0, 1, 2, 3, 4, 5]
        attention_mask = [1, 1, 1, 1, 1, 0]
        for name, tokens in marker_cases:
            with self.subTest(name=name):
                alignment = align_words_to_tokens(
                    words=["playing", "ball"],
                    token_ids=token_ids,
                    attention_mask=attention_mask,
                    tokenizer=MarkerTokenizer(tokens),
                )
                self.assertEqual(
                    alignment.word_to_token_indices,
                    ((1, 2), (3,)),
                )
                self.assertEqual(
                    alignment.first_subword_mask,
                    (False, True, False, True, False, False),
                )

    def test_unmatched_word_does_not_consume_the_next_matching_tokens(self):
        alignment = align_words_to_tokens(
            words=["playing", "not-present", "ball"],
            token_ids=[0, 1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1, 0],
            tokenizer=MarkerTokenizer(["play", "##ing", "ball"]),
        )
        self.assertEqual(
            alignment.word_to_token_indices,
            ((1, 2), tuple(), (3,)),
        )
        self.assertEqual(
            alignment.first_subword_mask,
            (False, True, False, True, False, False),
        )

    def test_remap_marks_exact_first_subwords_even_for_all_zero_gaze(self):
        features, mapped_mask = remap_word_features_to_tokens(
            word_features=np.asarray(
                [[0.0, 0.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            words=["playing", "ball"],
            token_ids=[0, 1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1, 0],
            tokenizer=MarkerTokenizer(["▁play", "ing", "▁ball"]),
            feature_dim=2,
        )
        self.assertEqual(mapped_mask.tolist(), [False, True, False, True, False, False])
        torch.testing.assert_close(features[1], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(features[2], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(features[3], torch.tensor([3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
