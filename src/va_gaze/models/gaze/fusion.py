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


class GmmDualGatePooling(BaseGazeFusion):
    """Learn gaze regimes and task-specific feature gates after text encoding."""

    def __init__(
        self,
        hidden_size,
        gaze_dim,
        gaze_hidden_size=128,
        gmm_components=5,
        gmm_temperature=1.0,
        gmm_nll_weight=0.01,
        gate_init=-4.0,
        dropout=0.1,
    ):
        super().__init__()
        if int(gaze_dim) != 5:
            raise ValueError(
                "gmm-dual-gate-pooling requires all five ET features "
                "(nFix, FFD, GPT, TRT, fixProp)."
            )
        if int(gmm_components) < 2:
            raise ValueError("gmm_components must be >= 2 for GMM dual-gate pooling.")
        if float(gmm_temperature) <= 0:
            raise ValueError("gmm_temperature must be > 0.")
        if float(gmm_nll_weight) < 0:
            raise ValueError("gmm_nll_weight must be >= 0.")

        self.hidden_size = int(hidden_size)
        self.gaze_dim = int(gaze_dim)
        self.gaze_hidden_size = int(gaze_hidden_size)
        self.gmm_components = int(gmm_components)
        self.gmm_temperature = float(gmm_temperature)
        self.gmm_nll_weight = float(gmm_nll_weight)

        self.gmm_input_norm = nn.LayerNorm(
            self.gaze_dim,
            elementwise_affine=False,
        )
        self.gmm_means = nn.Parameter(torch.empty(self.gmm_components, self.gaze_dim))
        self.gmm_log_vars = nn.Parameter(torch.zeros(self.gmm_components, self.gaze_dim))
        self.gmm_logits = nn.Parameter(torch.zeros(self.gmm_components))
        nn.init.normal_(self.gmm_means, mean=0.0, std=0.5)

        self.feature_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.gaze_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.gaze_hidden_size, self.hidden_size),
                )
                for _ in range(self.gaze_dim)
            ]
        )
        gate_input_size = (
            self.hidden_size + self.gaze_dim + self.gmm_components + 1
        )
        self.feature_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_size, self.gaze_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.gaze_hidden_size, self.gaze_dim),
                )
                for _ in range(2)
            ]
        )
        self.text_scorers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.gaze_hidden_size, bias=False) for _ in range(2)]
        )
        self.gaze_scorers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.gaze_hidden_size, bias=False) for _ in range(2)]
        )
        self.pooling_scores = nn.ModuleList(
            [nn.Linear(self.gaze_hidden_size, 1, bias=False) for _ in range(2)]
        )
        self.residual_fusions = nn.ModuleList(
            [
                GatedResidualFusion(
                    hidden_size=self.hidden_size,
                    gate_init=gate_init,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )

    @staticmethod
    def _transform_features(features):
        return torch.sign(features) * torch.log1p(features.abs())

    def _gmm_posterior(self, normalized_features, valid):
        features = normalized_features.unsqueeze(-2)
        log_vars = self.gmm_log_vars.clamp(min=-4.0, max=4.0)
        inverse_vars = torch.exp(-log_vars)
        squared = torch.square(features - self.gmm_means) * inverse_vars
        log_prob = -0.5 * (
            squared + log_vars + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_joint = torch.log_softmax(self.gmm_logits, dim=-1) + log_prob
        responsibilities = torch.softmax(
            log_joint / self.gmm_temperature,
            dim=-1,
        )
        responsibilities = responsibilities * valid.unsqueeze(-1).to(
            dtype=responsibilities.dtype
        )

        entropy = -(
            responsibilities
            * responsibilities.clamp_min(torch.finfo(responsibilities.dtype).eps).log()
        ).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(self.gmm_components)
        confidence = confidence.clamp(min=0.0, max=1.0)
        confidence = confidence * valid.to(dtype=confidence.dtype)

        if valid.any():
            nll = -torch.logsumexp(log_joint[valid], dim=-1).mean() / self.gaze_dim
        else:
            nll = log_joint.sum() * 0.0
        return responsibilities, confidence, self.gmm_nll_weight * nll

    def forward(self, cls_state, text_states, gaze_batch):
        valid = gaze_batch.valid_mask
        transformed = self._transform_features(gaze_batch.features)
        normalized = self.gmm_input_norm(transformed)
        normalized = normalized * valid.unsqueeze(-1).to(dtype=normalized.dtype)
        responsibilities, confidence, gmm_loss = self._gmm_posterior(
            normalized,
            valid,
        )

        expert_outputs = torch.stack(
            [
                expert(transformed[..., feature_index : feature_index + 1])
                for feature_index, expert in enumerate(self.feature_experts)
            ],
            dim=2,
        )
        gate_inputs = torch.cat(
            (
                text_states,
                normalized,
                responsibilities,
                confidence.unsqueeze(-1),
            ),
            dim=-1,
        )

        task_representations = []
        for task_index in range(2):
            feature_weights = torch.softmax(
                self.feature_gates[task_index](gate_inputs),
                dim=-1,
            )
            gaze_context = torch.sum(
                feature_weights.unsqueeze(-1) * expert_outputs,
                dim=2,
            )
            gaze_context = gaze_context * confidence.unsqueeze(-1)
            gaze_context = gaze_context * valid.unsqueeze(-1).to(
                dtype=gaze_context.dtype
            )

            scores = self.pooling_scores[task_index](
                torch.tanh(
                    self.text_scorers[task_index](text_states)
                    + self.gaze_scorers[task_index](gaze_context)
                )
            ).squeeze(-1)
            token_weights = _masked_softmax(scores, valid, dim=-1)
            pooled_text = torch.sum(
                token_weights.unsqueeze(-1) * text_states,
                dim=1,
            )
            task_representations.append(
                self.residual_fusions[task_index](
                    cls_state,
                    pooled_text,
                    gaze_batch.has_gaze,
                )
            )

        return torch.stack(task_representations, dim=1), gmm_loss


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
    gmm_components=5,
    gmm_temperature=1.0,
    gmm_nll_weight=0.01,
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
    if normalized == "gmm-dual-gate-pooling":
        return GmmDualGatePooling(
            hidden_size=hidden_size,
            gaze_dim=gaze_dim,
            gaze_hidden_size=gaze_hidden_size,
            gmm_components=gmm_components,
            gmm_temperature=gmm_temperature,
            gmm_nll_weight=gmm_nll_weight,
            gate_init=gate_init,
            dropout=dropout,
        )
    raise ValueError(f"Unknown advanced gaze fusion strategy: {strategy}")
