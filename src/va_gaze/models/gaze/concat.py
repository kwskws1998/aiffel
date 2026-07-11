from dataclasses import dataclass

import torch


POSTFIX_CONCAT = "postfix"
PREFIX_CONCAT = "prefix"

CONCAT_ORDER_ALIASES = {
    "concat": POSTFIX_CONCAT,
    "postfix": POSTFIX_CONCAT,
    "postfix-concat": POSTFIX_CONCAT,
    "concat-postfix": POSTFIX_CONCAT,
    "text-prefix": POSTFIX_CONCAT,
    "gaze-postfix": POSTFIX_CONCAT,
    "prefix": PREFIX_CONCAT,
    "prefix-concat": PREFIX_CONCAT,
    "concat-prefix": PREFIX_CONCAT,
    "gaze-prefix": PREFIX_CONCAT,
}


@dataclass
class GazeConcatInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    cls_positions: torch.Tensor
    token_type_ids: torch.Tensor = None
    position_ids: torch.Tensor = None


def normalize_concat_order(raw_value):
    normalized = CONCAT_ORDER_ALIASES.get(raw_value or POSTFIX_CONCAT)
    if normalized is None:
        choices = ", ".join(sorted(CONCAT_ORDER_ALIASES))
        raise ValueError(f"Unknown gaze concat order {raw_value!r}. Expected one of: {choices}.")
    return normalized


def _validate_concat_tensors(
    text_embeddings,
    gaze_embeddings,
    text_attention_mask,
    gaze_attention_mask,
    token_type_ids,
    position_ids,
):
    if text_embeddings.ndim != 3 or gaze_embeddings.ndim != 3:
        raise ValueError("Text and gaze embeddings must both be rank-3 tensors.")
    if text_embeddings.shape[0] != gaze_embeddings.shape[0]:
        raise ValueError("Text and gaze embeddings must have the same batch size.")
    if text_embeddings.shape[1] != gaze_embeddings.shape[1]:
        raise ValueError("Full token-aligned concat requires equal text and gaze lengths.")
    if text_embeddings.shape[2] != gaze_embeddings.shape[2]:
        raise ValueError("Text and gaze embeddings must have the same hidden size.")
    if tuple(text_attention_mask.shape) != tuple(text_embeddings.shape[:2]):
        raise ValueError("text_attention_mask must match the text sequence shape.")
    if tuple(gaze_attention_mask.shape) != tuple(gaze_embeddings.shape[:2]):
        raise ValueError("gaze_attention_mask must match the gaze sequence shape.")
    if text_embeddings.shape[1] == 0:
        raise ValueError("The text sequence must contain at least the CLS token.")

    tensors = (
        gaze_embeddings,
        text_attention_mask,
        gaze_attention_mask,
        token_type_ids,
        position_ids,
    )
    if any(value is not None and value.device != text_embeddings.device for value in tensors):
        raise ValueError("All concat tensors must be on the same device.")

    for name, value in (
        ("token_type_ids", token_type_ids),
        ("position_ids", position_ids),
    ):
        if value is not None and tuple(value.shape) != tuple(text_embeddings.shape[:2]):
            raise ValueError(f"{name} must match the original text sequence shape.")

    text_mask_bool = text_attention_mask.to(dtype=torch.bool)
    valid_lengths = text_mask_bool.sum(dim=1)
    if (valid_lengths <= 0).any():
        raise ValueError("Every text sequence must contain at least the CLS token.")
    positions = torch.arange(
        text_embeddings.shape[1],
        device=text_embeddings.device,
    ).unsqueeze(0)
    expected_text_mask = positions < valid_lengths.unsqueeze(1)
    if not torch.equal(text_mask_bool, expected_text_mask):
        raise ValueError("text_attention_mask must use contiguous right padding.")
    if (gaze_attention_mask.to(dtype=torch.bool) & ~expected_text_mask).any():
        raise ValueError("gaze_attention_mask cannot enable a padded text position.")
    return valid_lengths


def _pad_embeddings(row, target_length):
    padding_length = target_length - row.shape[0]
    if padding_length <= 0:
        return row
    return torch.cat((row, row.new_zeros(padding_length, row.shape[1])), dim=0)


def _pad_ids(row, target_length):
    padding_length = target_length - row.shape[0]
    if padding_length <= 0:
        return row
    return torch.cat((row, row.new_zeros(padding_length)), dim=0)


def _compose_optional_ids(
    row_ids,
    valid_length,
    fused_length,
    order,
    is_position_ids,
):
    if row_ids is None:
        return None
    text_ids = row_ids[:valid_length]
    extra_length = valid_length + 2
    if is_position_ids:
        offsets = torch.arange(
            extra_length,
            dtype=text_ids.dtype,
            device=text_ids.device,
        )
        if order == POSTFIX_CONCAT:
            extra_ids = text_ids.max() + 1 + offsets
            fused_ids = torch.cat((text_ids, extra_ids), dim=0)
        else:
            prefix_ids = text_ids[:1] + offsets
            fused_ids = torch.cat((prefix_ids, text_ids + extra_length), dim=0)
    else:
        extra_ids = text_ids.new_zeros(extra_length)
        if order == POSTFIX_CONCAT:
            fused_ids = torch.cat((text_ids, extra_ids), dim=0)
        else:
            fused_ids = torch.cat((extra_ids, text_ids), dim=0)
    return _pad_ids(fused_ids, fused_length)


def compose_gaze_concat_inputs(
    text_embeddings,
    gaze_embeddings,
    text_attention_mask,
    gaze_attention_mask,
    eye_start,
    eye_end,
    order=POSTFIX_CONCAT,
    token_type_ids=None,
    position_ids=None,
):
    """Pack token-aligned text and gaze per sample, then right-pad the fused batch."""

    order = normalize_concat_order(order)
    valid_lengths = _validate_concat_tensors(
        text_embeddings,
        gaze_embeddings,
        text_attention_mask,
        gaze_attention_mask,
        token_type_ids,
        position_ids,
    )
    _, _, hidden_size = gaze_embeddings.shape
    if eye_start.numel() != hidden_size or eye_end.numel() != hidden_size:
        raise ValueError("Eye boundary embeddings must match the encoder hidden size.")

    eye_start_embed = eye_start.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, hidden_size)
    eye_end_embed = eye_end.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, hidden_size)
    fused_length = 2 * int(valid_lengths.max().item()) + 2
    embedding_rows = []
    attention_rows = []
    token_type_rows = []
    position_rows = []
    cls_positions = []

    for row_index, valid_length in enumerate(valid_lengths.tolist()):
        text_row = text_embeddings[row_index, :valid_length]
        gaze_row = gaze_embeddings[row_index, :valid_length]
        text_mask_row = text_attention_mask[row_index, :valid_length]
        gaze_mask_row = gaze_attention_mask[row_index, :valid_length].to(
            dtype=text_attention_mask.dtype
        )
        boundary_mask = text_mask_row.new_ones(1)

        if order == POSTFIX_CONCAT:
            embedding_row = torch.cat(
                (text_row, eye_start_embed, gaze_row, eye_end_embed),
                dim=0,
            )
            attention_row = torch.cat(
                (text_mask_row, boundary_mask, gaze_mask_row, boundary_mask),
                dim=0,
            )
            cls_positions.append(0)
        else:
            embedding_row = torch.cat(
                (eye_start_embed, gaze_row, eye_end_embed, text_row),
                dim=0,
            )
            attention_row = torch.cat(
                (boundary_mask, gaze_mask_row, boundary_mask, text_mask_row),
                dim=0,
            )
            cls_positions.append(valid_length + 2)

        embedding_rows.append(_pad_embeddings(embedding_row, fused_length))
        attention_rows.append(_pad_ids(attention_row, fused_length))
        if token_type_ids is not None:
            token_type_rows.append(
                _compose_optional_ids(
                    token_type_ids[row_index],
                    valid_length,
                    fused_length,
                    order,
                    is_position_ids=False,
                )
            )
        if position_ids is not None:
            position_rows.append(
                _compose_optional_ids(
                    position_ids[row_index],
                    valid_length,
                    fused_length,
                    order,
                    is_position_ids=True,
                )
            )

    return GazeConcatInputs(
        inputs_embeds=torch.stack(embedding_rows, dim=0),
        attention_mask=torch.stack(attention_rows, dim=0),
        cls_positions=torch.tensor(
            cls_positions,
            dtype=torch.long,
            device=text_embeddings.device,
        ),
        token_type_ids=(torch.stack(token_type_rows, dim=0) if token_type_rows else None),
        position_ids=(torch.stack(position_rows, dim=0) if position_rows else None),
    )
