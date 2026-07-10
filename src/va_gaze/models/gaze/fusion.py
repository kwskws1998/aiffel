import math

import torch
import torch.nn as nn


def _masked_softmax(scores, mask, dim=-1):
    mask = mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return weights / normalizer


class BaseGazeFusion(nn.Module):
    def forward(self, cls_state, text_states, gaze_batch):
        raise NotImplementedError


class IdentityGazeFusion(BaseGazeFusion):
    def forward(self, cls_state, text_states, gaze_batch):
        return cls_state


class GatedResidualFusion(nn.Module):
    def __init__(self, hidden_size, context_size=None, gate_init=-4.0, dropout=0.1):
        super().__init__()
        context_size = int(context_size or hidden_size)
        self.context_projector = nn.Linear(context_size, hidden_size)
        self.gate = nn.Linear(hidden_size + context_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.gate.bias, float(gate_init))

    def forward(self, cls_state, context, available):
        gate = torch.sigmoid(self.gate(torch.cat((cls_state, context), dim=-1)))
        delta = gate * self.dropout(self.context_projector(context))
        delta = delta * available.to(dtype=delta.dtype).unsqueeze(-1)
        return cls_state + delta


class GazeConditionedPooling(BaseGazeFusion):
    def __init__(self, hidden_size, gaze_dim, gate_init=-4.0, dropout=0.1):
        super().__init__()
        self.text_scorer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gaze_projector = nn.Sequential(
            nn.Linear(gaze_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.fusion = GatedResidualFusion(
            hidden_size=hidden_size,
            gate_init=gate_init,
            dropout=dropout,
        )

    def forward(self, cls_state, text_states, gaze_batch):
        valid = gaze_batch.valid_mask
        gaze_hidden = self.gaze_projector(gaze_batch.features)
        gaze_hidden = gaze_hidden * valid.unsqueeze(-1).to(dtype=gaze_hidden.dtype)
        scores = self.score(torch.tanh(self.text_scorer(text_states) + gaze_hidden)).squeeze(-1)
        weights = _masked_softmax(scores, valid, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * text_states, dim=1)
        return self.fusion(cls_state, pooled, gaze_batch.has_gaze)


class GazeBiasedClsAttention(BaseGazeFusion):
    """Post-encoder CLS-to-token attention with a soft gaze logit bias."""

    def __init__(
        self,
        hidden_size,
        gaze_dim,
        num_heads=4,
        attention_scale=0.1,
        train_attention_scale=True,
        gate_init=-4.0,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gaze_bias = nn.Sequential(
            nn.Linear(gaze_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads),
        )
        scale = torch.full((num_heads,), float(attention_scale))
        if train_attention_scale:
            self.attention_scale = nn.Parameter(scale)
        else:
            self.register_buffer("attention_scale", scale)
        self.fusion = GatedResidualFusion(
            hidden_size=hidden_size,
            gate_init=gate_init,
            dropout=dropout,
        )

    def forward(self, cls_state, text_states, gaze_batch):
        batch_size, seq_len, _ = text_states.shape
        query = self.query(cls_state).view(batch_size, self.num_heads, self.head_dim)
        key = self.key(text_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.value(text_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.einsum("bhd,bhsd->bhs", query, key) / math.sqrt(self.head_dim)
        gaze_bias = torch.tanh(self.gaze_bias(gaze_batch.features)).transpose(1, 2)
        gaze_bias = gaze_bias * gaze_batch.valid_mask.unsqueeze(1).to(dtype=gaze_bias.dtype)
        scores = scores + self.attention_scale.view(1, -1, 1) * gaze_bias

        text_mask = gaze_batch.text_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
        weights = _masked_softmax(scores, text_mask, dim=-1)
        context = torch.einsum("bhs,bhsd->bhd", weights, value).reshape(batch_size, -1)
        return self.fusion(cls_state, context, gaze_batch.has_gaze)


class GazeCrossAttention(BaseGazeFusion):
    def __init__(
        self,
        hidden_size,
        gaze_dim,
        gaze_hidden_size=128,
        num_heads=4,
        num_layers=1,
        max_positions=1024,
        gate_init=-4.0,
        dropout=0.1,
    ):
        super().__init__()
        if gaze_hidden_size % num_heads != 0:
            raise ValueError("gaze_hidden_size must be divisible by num_heads.")
        self.gaze_hidden_size = int(gaze_hidden_size)
        self.gaze_projector = nn.Sequential(
            nn.Linear(gaze_dim, gaze_hidden_size),
            nn.LayerNorm(gaze_hidden_size),
            nn.GELU(),
        )
        self.position_embeddings = nn.Embedding(max_positions, gaze_hidden_size)
        if int(num_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=gaze_hidden_size,
                nhead=num_heads,
                dim_feedforward=gaze_hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.gaze_encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        else:
            self.gaze_encoder = None
        self.query_projector = nn.Linear(hidden_size, gaze_hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=gaze_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = GatedResidualFusion(
            hidden_size=hidden_size,
            context_size=gaze_hidden_size,
            gate_init=gate_init,
            dropout=dropout,
        )

    def forward(self, cls_state, text_states, gaze_batch):
        valid = gaze_batch.valid_mask
        safe_valid = valid.clone()
        no_gaze = ~safe_valid.any(dim=1)
        if no_gaze.any():
            safe_valid[no_gaze, 0] = True

        positions = torch.arange(
            gaze_batch.features.shape[1],
            device=gaze_batch.features.device,
        ).clamp_max(self.position_embeddings.num_embeddings - 1)
        gaze_states = self.gaze_projector(gaze_batch.features)
        gaze_states = gaze_states + self.position_embeddings(positions).unsqueeze(0)
        gaze_states = gaze_states * safe_valid.unsqueeze(-1).to(dtype=gaze_states.dtype)
        if self.gaze_encoder is not None:
            gaze_states = self.gaze_encoder(
                gaze_states,
                src_key_padding_mask=~safe_valid,
            )

        query = self.query_projector(cls_state).unsqueeze(1)
        context, _ = self.cross_attention(
            query=query,
            key=gaze_states,
            value=gaze_states,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        context = context.squeeze(1)
        return self.fusion(cls_state, context, gaze_batch.has_gaze)


FUSION_ALIASES = {
    "pooling": "conditioned-pooling",
    "attention-bias": "postencoder-cls-attention-bias",
    "cls-attention-bias": "postencoder-cls-attention-bias",
}


def build_gaze_fusion(
    strategy,
    hidden_size,
    gaze_dim,
    gaze_hidden_size=128,
    num_heads=4,
    num_layers=1,
    max_positions=1024,
    gate_init=-4.0,
    dropout=0.1,
    attention_scale=0.1,
    train_attention_scale=True,
):
    normalized = FUSION_ALIASES.get(strategy or "none", strategy or "none")
    if normalized == "none":
        return IdentityGazeFusion()
    if normalized == "conditioned-pooling":
        return GazeConditionedPooling(
            hidden_size=hidden_size,
            gaze_dim=gaze_dim,
            gate_init=gate_init,
            dropout=dropout,
        )
    if normalized == "postencoder-cls-attention-bias":
        return GazeBiasedClsAttention(
            hidden_size=hidden_size,
            gaze_dim=gaze_dim,
            num_heads=num_heads,
            attention_scale=attention_scale,
            train_attention_scale=train_attention_scale,
            gate_init=gate_init,
            dropout=dropout,
        )
    if normalized == "cross-attention":
        return GazeCrossAttention(
            hidden_size=hidden_size,
            gaze_dim=gaze_dim,
            gaze_hidden_size=gaze_hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_positions=max_positions,
            gate_init=gate_init,
            dropout=dropout,
        )
    raise ValueError(f"Unknown advanced gaze fusion strategy: {strategy}")
