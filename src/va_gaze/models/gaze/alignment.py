"""Exact monotonic alignment from predictor words to target-model subwords."""

from dataclasses import dataclass
import re
import unicodedata

import torch


_TOKEN_PREFIX = re.compile(r"^(?:(?:##)|[Ġ▁])+")


@dataclass(frozen=True)
class WordTokenAlignment:
    """Store exact token spans and the first-subword validity mask."""

    word_to_token_indices: tuple[tuple[int, ...], ...]
    first_subword_mask: tuple[bool, ...]


def normalize_alignment_text(value):
    """Normalize tokenizer boundary markers and spacing for exact comparison."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _TOKEN_PREFIX.sub("", text)
    return re.sub(r"\s+", "", text)


def _tokens_to_text(tokenizer, tokens):
    """Reconstruct a token span with the tokenizer and a marker-aware fallback."""

    converter = getattr(tokenizer, "convert_tokens_to_string", None)
    if callable(converter):
        try:
            converted = converter(list(tokens))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            converted = None
        if converted is not None:
            return str(converted)
    return "".join(_TOKEN_PREFIX.sub("", str(token)) for token in tokens)


def align_words_to_tokens(words, token_ids, attention_mask, tokenizer):
    """Align exact normalized words without consuming tokens after a mismatch."""

    ids = [int(token_id) for token_id in token_ids]
    mask = [bool(value) for value in attention_mask]
    if len(ids) != len(mask):
        raise ValueError("token_ids and attention_mask must have the same length.")

    tokens = tokenizer.convert_ids_to_tokens(ids)
    if isinstance(tokens, str):
        tokens = [tokens]
    if len(tokens) != len(ids):
        raise ValueError("Tokenizer returned a token sequence with the wrong length.")

    special_ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", []) or [])}
    special_tokens = {
        str(value) for value in (getattr(tokenizer, "all_special_tokens", []) or [])
    }
    lexical_positions = tuple(
        index
        for index, (token_id, token, is_active) in enumerate(zip(ids, tokens, mask))
        if is_active and token_id not in special_ids and str(token) not in special_tokens
    )
    lexical_tokens = tuple(str(tokens[index]) for index in lexical_positions)

    cursor = 0
    mappings = []
    for word in words:
        target = normalize_alignment_text(word)
        match = None
        if target:
            for start in range(cursor, len(lexical_tokens)):
                max_span = max(16, len(target) * 4)
                stop = min(len(lexical_tokens), start + max_span)
                for end in range(start + 1, stop + 1):
                    candidate = normalize_alignment_text(
                        _tokens_to_text(tokenizer, lexical_tokens[start:end])
                    )
                    if candidate == target:
                        match = (start, end)
                        break
                    if len(candidate) > len(target) + 4:
                        break
                if match is not None:
                    break

        if match is None:
            mappings.append(tuple())
            continue

        start, end = match
        indices = tuple(lexical_positions[start:end])
        mappings.append(indices)
        cursor = end

    first_subword_mask = [False] * len(ids)
    for indices in mappings:
        if indices:
            first_subword_mask[indices[0]] = True
    return WordTokenAlignment(
        word_to_token_indices=tuple(mappings),
        first_subword_mask=tuple(first_subword_mask),
    )


def remap_word_features_to_tokens(
    word_features,
    words,
    token_ids,
    attention_mask,
    tokenizer,
    feature_dim,
):
    """Place each word feature on its exactly aligned first target subword."""

    ids = [int(token_id) for token_id in token_ids]
    output = torch.zeros(len(ids), int(feature_dim), dtype=torch.float32)
    mapped_mask = torch.zeros(len(ids), dtype=torch.bool)
    if not words:
        return output, mapped_mask

    features = torch.as_tensor(word_features, dtype=torch.float32)
    if features.numel() == 0:
        return output, mapped_mask
    if features.ndim != 2 or features.shape[1] != int(feature_dim):
        raise ValueError(
            f"word_features must have shape [num_words, {int(feature_dim)}]."
        )

    alignment = align_words_to_tokens(words, ids, attention_mask, tokenizer)
    limit = min(len(words), features.shape[0], len(alignment.word_to_token_indices))
    for word_index in range(limit):
        indices = alignment.word_to_token_indices[word_index]
        if not indices:
            continue
        first_subword = indices[0]
        output[first_subword] = features[word_index]
        mapped_mask[first_subword] = True
    return output, mapped_mask
