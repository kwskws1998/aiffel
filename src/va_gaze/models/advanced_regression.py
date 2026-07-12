import json
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from va_gaze.models.gaze.fusion import FUSION_ALIASES, build_gaze_fusion
from va_gaze.models.gaze.objectives import MaskedGazePrediction, TokenInfoNCEAlignment
from va_gaze.models.gaze.provider import GazeFeatureProvider


CANONICAL_ADVANCED_GAZE_FUSIONS = (
    "conditioned-pooling",
    "postencoder-cls-attention-bias",
    "cross-attention",
    "gmm-dual-gate-pooling",
)
ADVANCED_GAZE_FUSIONS = (
    *CANONICAL_ADVANCED_GAZE_FUSIONS,
    "cls-attention-bias",
    "attention-bias",
    "pooling",
)

ADVANCED_GAZE_MANIFEST_NAME = "advanced_gaze_manifest.json"
ADVANCED_GAZE_WEIGHTS_NAME = "advanced_gaze_model.pt"
ADVANCED_GAZE_FORMAT_VERSION = 1


def normalize_advanced_fusion(raw_value):
    return FUSION_ALIASES.get(raw_value or "none", raw_value or "none")


def _json_path(value):
    return None if value is None else str(value)


def _json_vector(value):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32).flatten().tolist()


def _format_regression_logits(logits, output_dim):
    means = torch.nn.functional.hardsigmoid(3.0 * logits[:, :2])
    if int(output_dim) <= 2:
        return means.contiguous()
    return torch.cat((means, logits[:, 2:]), dim=-1).contiguous()


class DistilBertVARegressionHead(nn.Module):
    """Match DistilBertForSequenceClassification's regression head."""

    family = "distilbert"

    def __init__(self, hidden_size, output_dim, dropout):
        super().__init__()
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_dim)
        self.output_dim = int(output_dim)

    def forward(self, representation):
        hidden = self.pre_classifier(representation)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        if hidden.ndim == 3:
            if hidden.shape[1] != self.output_dim:
                raise ValueError("Task-specific representations must match output_dim.")
            logits = torch.einsum("bth,th->bt", hidden, self.classifier.weight)
            if self.classifier.bias is not None:
                logits = logits + self.classifier.bias
        else:
            logits = self.classifier(hidden)
        return _format_regression_logits(logits, self.output_dim)


class RobertaVARegressionHead(nn.Module):
    """Match RobertaClassificationHead after CLS has already been selected."""

    family = "roberta"

    def __init__(self, hidden_size, output_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_dim)
        self.output_dim = int(output_dim)

    def forward(self, representation):
        hidden = self.dropout(representation)
        hidden = self.dense(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout(hidden)
        if hidden.ndim == 3:
            if hidden.shape[1] != self.output_dim:
                raise ValueError("Task-specific representations must match output_dim.")
            logits = torch.einsum("bth,th->bt", hidden, self.out_proj.weight)
            if self.out_proj.bias is not None:
                logits = logits + self.out_proj.bias
        else:
            logits = self.out_proj(hidden)
        return _format_regression_logits(logits, self.output_dim)


def _classifier_dropout(config, family):
    if family == "distilbert":
        value = getattr(config, "seq_classif_dropout", None)
        if value is None:
            value = getattr(config, "classifier_dropout", None)
    else:
        value = getattr(config, "classifier_dropout", None)
    if value is None:
        value = getattr(config, "hidden_dropout_prob", 0.1)
    return float(value)


def _build_regression_head(config, output_dim):
    model_type = str(getattr(config, "model_type", ""))
    if model_type in ("roberta", "xlm-roberta"):
        family = "roberta"
        head = RobertaVARegressionHead(
            hidden_size=int(config.hidden_size),
            output_dim=output_dim,
            dropout=_classifier_dropout(config, family),
        )
    else:
        family = "distilbert"
        head = DistilBertVARegressionHead(
            hidden_size=int(config.hidden_size),
            output_dim=output_dim,
            dropout=_classifier_dropout(config, family),
        )

    initializer_range = float(getattr(config, "initializer_range", 0.02))
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return head


class GazeFusionForSequenceRegression(nn.Module):
    """Orchestrate post-encoder gaze fusion and training-only gaze objectives."""

    supports_gaze_auxiliary_loss = True

    def __init__(
        self,
        checkpoint,
        tokenizer,
        fusion_strategy="none",
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        load_fixation_model=True,
        et_model_type="et2",
        et_model_id=None,
        gaze_transform="raw",
        gaze_artifact_dir=None,
        pca_components=2,
        gmm_components=5,
        output_dim=2,
        gaze_aux_weight=0.0,
        gaze_alignment_weight=0.0,
        gaze_hidden_size=128,
        gaze_num_heads=4,
        gaze_num_layers=1,
        gaze_gate_init=-4.0,
        gaze_fusion_dropout=0.1,
        gaze_attention_scale=0.1,
        train_gaze_attention_scale=True,
        gaze_alignment_dim=128,
        gaze_alignment_temperature=0.07,
        gaze_alignment_max_tokens=512,
        gmm_temperature=1.0,
        gmm_nll_weight=0.01,
        gaze_target_transform="signed-log1p",
        gaze_target_mean=None,
        gaze_target_scale=None,
        encoder=None,
    ):
        super().__init__()
        if encoder is None:
            if checkpoint is None:
                raise ValueError("checkpoint is required when encoder is not supplied.")
            encoder = AutoModel.from_pretrained(checkpoint)
        self.encoder = encoder
        self.config = self.encoder.config
        ignored_at_inference = set(
            getattr(self.config, "keys_to_ignore_at_inference", None) or []
        )
        ignored_at_inference.update(("hidden_states", "attentions"))
        self.config.keys_to_ignore_at_inference = sorted(ignored_at_inference)
        self.tokenizer = tokenizer
        self.hidden_size = int(self.config.hidden_size)
        self.output_dim = int(output_dim)
        self.num_labels = self.output_dim
        self.config.num_labels = self.output_dim
        self.fusion_strategy = normalize_advanced_fusion(fusion_strategy)
        if self.fusion_strategy not in (*CANONICAL_ADVANCED_GAZE_FUSIONS, "none"):
            raise ValueError(f"Unknown advanced gaze fusion: {fusion_strategy}")
        if self.fusion_strategy == "gmm-dual-gate-pooling" and self.output_dim != 2:
            raise ValueError("gmm-dual-gate-pooling currently supports two-output VA regression only.")

        self.gaze_aux_weight = float(gaze_aux_weight)
        self.gaze_alignment_weight = float(gaze_alignment_weight)
        self.gaze_provider = GazeFeatureProvider(
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            max_fix_cache_size=max_fix_cache_size,
            load_fixation_model=load_fixation_model,
            et_model_type=et_model_type,
            et_model_id=et_model_id,
            gaze_transform=gaze_transform,
            gaze_artifact_dir=gaze_artifact_dir,
            pca_components=pca_components,
            gmm_components=gmm_components,
        )

        max_positions = int(getattr(self.config, "max_position_embeddings", 1024))
        self.fusion = build_gaze_fusion(
            strategy=self.fusion_strategy,
            hidden_size=self.hidden_size,
            gaze_dim=self.gaze_provider.feature_dim,
            gaze_hidden_size=gaze_hidden_size,
            num_heads=gaze_num_heads,
            num_layers=gaze_num_layers,
            max_positions=max_positions,
            gate_init=gaze_gate_init,
            dropout=gaze_fusion_dropout,
            attention_scale=gaze_attention_scale,
            train_attention_scale=train_gaze_attention_scale,
            gmm_components=gmm_components,
            gmm_temperature=gmm_temperature,
            gmm_nll_weight=gmm_nll_weight,
        )

        self.gaze_prediction = None
        if self.gaze_aux_weight > 0:
            self.gaze_prediction = MaskedGazePrediction(
                hidden_size=self.hidden_size,
                gaze_dim=self.gaze_provider.feature_dim,
                dropout=gaze_fusion_dropout,
                target_transform=gaze_target_transform,
                target_mean=gaze_target_mean,
                target_scale=gaze_target_scale,
            )

        self.gaze_alignment = None
        if self.gaze_alignment_weight > 0:
            self.gaze_alignment = TokenInfoNCEAlignment(
                hidden_size=self.hidden_size,
                gaze_dim=self.gaze_provider.feature_dim,
                alignment_dim=gaze_alignment_dim,
                temperature=gaze_alignment_temperature,
                max_tokens=gaze_alignment_max_tokens,
            )

        self.regression_head = _build_regression_head(self.config, self.output_dim)
        self._architecture_kwargs = {
            "fusion_strategy": self.fusion_strategy,
            "et2_checkpoint_path": _json_path(et2_checkpoint_path),
            "features_used": None if features_used is None else list(features_used),
            "fp_dropout": list(fp_dropout),
            "max_fix_cache_size": int(max_fix_cache_size),
            "load_fixation_model": bool(load_fixation_model),
            "et_model_type": self.gaze_provider.et_model_type,
            "et_model_id": _json_path(et_model_id),
            "gaze_transform": gaze_transform or "raw",
            "gaze_artifact_dir": _json_path(gaze_artifact_dir),
            "pca_components": int(pca_components),
            "gmm_components": int(gmm_components),
            "output_dim": self.output_dim,
            "gaze_aux_weight": self.gaze_aux_weight,
            "gaze_alignment_weight": self.gaze_alignment_weight,
            "gaze_hidden_size": int(gaze_hidden_size),
            "gaze_num_heads": int(gaze_num_heads),
            "gaze_num_layers": int(gaze_num_layers),
            "gaze_gate_init": float(gaze_gate_init),
            "gaze_fusion_dropout": float(gaze_fusion_dropout),
            "gaze_attention_scale": float(gaze_attention_scale),
            "train_gaze_attention_scale": bool(train_gaze_attention_scale),
            "gaze_alignment_dim": int(gaze_alignment_dim),
            "gaze_alignment_temperature": float(gaze_alignment_temperature),
            "gaze_alignment_max_tokens": int(gaze_alignment_max_tokens),
            "gmm_temperature": float(gmm_temperature),
            "gmm_nll_weight": float(gmm_nll_weight),
            "gaze_target_transform": gaze_target_transform,
            "gaze_target_mean": _json_vector(gaze_target_mean),
            "gaze_target_scale": _json_vector(gaze_target_scale),
        }

    @property
    def has_training_objective(self):
        return self.gaze_prediction is not None or self.gaze_alignment is not None

    def _save_bundle_assets(self, output_dir):
        encoder_config_dir = output_dir / "encoder_config"
        tokenizer_dir = output_dir / "tokenizer"
        self.encoder.config.save_pretrained(encoder_config_dir)
        if not hasattr(self.tokenizer, "save_pretrained"):
            raise TypeError(
                "The tokenizer must implement save_pretrained() for a self-contained bundle."
            )
        self.tokenizer.save_pretrained(tokenizer_dir)

        bundled_artifact_dir = None
        transformer = self.gaze_provider.gaze_feature_transformer
        if transformer.transform != "raw":
            import joblib

            bundled_artifact_dir = output_dir / "gaze_artifacts"
            bundled_artifact_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(transformer.scaler, bundled_artifact_dir / "gaze_scaler.joblib")
            if transformer.transform == "pca":
                filename = f"pca_{transformer.pca_components}.joblib"
            else:
                filename = f"gmm_k{transformer.gmm_components}.joblib"
            joblib.dump(transformer.model, bundled_artifact_dir / filename)

        return {
            "encoder_config_dir": encoder_config_dir.name,
            "tokenizer_dir": tokenizer_dir.name,
            "bundled_gaze_artifact_dir": (
                None if bundled_artifact_dir is None else bundled_artifact_dir.name
            ),
        }

    def _write_architecture_manifest(self, output_dir, weights_filename):
        assets = self._save_bundle_assets(output_dir)
        manifest = {
            "format_version": ADVANCED_GAZE_FORMAT_VERSION,
            "model_class": type(self).__name__,
            "model_type": str(getattr(self.config, "model_type", "")),
            "regression_head_family": self.regression_head.family,
            "weights_filename": str(weights_filename),
            "architecture": dict(self._architecture_kwargs),
            "provider": {
                "raw_feature_dim": int(self.gaze_provider.raw_feature_dim),
                "feature_dim": int(self.gaze_provider.feature_dim),
            },
            **assets,
        }
        manifest_path = output_dir / ADVANCED_GAZE_MANIFEST_NAME
        temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, indent=2, sort_keys=True)
        temporary_path.replace(manifest_path)
        return manifest_path

    def save_pretrained(self, output_dir, state_dict=None, **kwargs):
        """Save weights plus all architecture assets needed for offline reload."""

        del kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / ADVANCED_GAZE_WEIGHTS_NAME
        torch.save(self.state_dict() if state_dict is None else state_dict, weights_path)
        return self._write_architecture_manifest(output_dir, weights_path.name)

    def save_architecture_manifest(self, output_dir, weights_filename=None):
        """Complete a directory after ``Trainer.save_model`` without duplicating weights."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if weights_filename is None:
            candidates = ("model.safetensors", "pytorch_model.bin", ADVANCED_GAZE_WEIGHTS_NAME)
            weights_filename = next(
                (filename for filename in candidates if (output_dir / filename).is_file()),
                None,
            )
        if weights_filename is None:
            weights_filename = ADVANCED_GAZE_WEIGHTS_NAME
            torch.save(self.state_dict(), output_dir / weights_filename)
        elif not (output_dir / weights_filename).is_file():
            raise FileNotFoundError(f"Saved model weights not found: {output_dir / weights_filename}")
        return self._write_architecture_manifest(output_dir, weights_filename)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        tokenizer=None,
        map_location="cpu",
        provider_overrides=None,
        **architecture_overrides,
    ):
        """Reload an advanced gaze bundle without contacting model hubs.

        Frozen gaze predictors are external data-producing dependencies. Their
        recorded identifiers are restored from the manifest; ``provider_overrides``
        can point those identifiers at local checkpoints for fully offline inference.
        """

        bundle_dir = Path(pretrained_model_name_or_path)
        manifest_path = bundle_dir / ADVANCED_GAZE_MANIFEST_NAME
        with open(manifest_path, "r", encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        if int(manifest.get("format_version", -1)) != ADVANCED_GAZE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported advanced gaze format version: {manifest.get('format_version')}"
            )

        encoder_config_dir = bundle_dir / manifest["encoder_config_dir"]
        config = AutoConfig.from_pretrained(encoder_config_dir, local_files_only=True)
        encoder = AutoModel.from_config(config)
        if tokenizer is None:
            tokenizer_dir = bundle_dir / manifest["tokenizer_dir"]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        architecture = dict(manifest["architecture"])
        artifact_subdir = manifest.get("bundled_gaze_artifact_dir")
        if artifact_subdir:
            architecture["gaze_artifact_dir"] = str(bundle_dir / artifact_subdir)
        if provider_overrides:
            architecture.update(provider_overrides)
        architecture.update(architecture_overrides)
        model = cls(
            checkpoint=None,
            tokenizer=tokenizer,
            encoder=encoder,
            **architecture,
        )

        expected_family = manifest.get("regression_head_family")
        if expected_family and model.regression_head.family != expected_family:
            raise ValueError(
                "Saved regression head family does not match the bundled encoder config: "
                f"{expected_family} != {model.regression_head.family}"
            )
        if int(manifest["provider"]["feature_dim"]) != model.gaze_provider.feature_dim:
            raise ValueError("Saved gaze feature dimension does not match reconstructed provider.")

        weights_path = bundle_dir / manifest["weights_filename"]
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_path), device="cpu")
        else:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(map_location)
        model.eval()
        return model

    def _encode_text(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if self.config.model_type != "distilbert":
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids
            if position_ids is not None:
                encoder_kwargs["position_ids"] = position_ids
        return self.encoder(**encoder_kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if input_ids is None:
            raise ValueError("input_ids cannot be None for gaze-aware regression.")
        if inputs_embeds is not None:
            raise ValueError("Pass input_ids, not inputs_embeds, so gaze features can be aligned to tokens.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_device = self.encoder.get_input_embeddings().weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(model_device)
        if position_ids is not None:
            position_ids = position_ids.to(model_device)

        encoder_outputs = self._encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        text_states = encoder_outputs.last_hidden_state
        cls_state = text_states[:, 0, :]

        gaze_batch = None
        needs_primary_gaze = self.fusion_strategy != "none"
        needs_training_gaze = self.training and self.has_training_objective
        if needs_primary_gaze or needs_training_gaze:
            gaze_batch = self.gaze_provider.compute(input_ids, attention_mask).to(
                device=model_device,
                dtype=text_states.dtype,
            )

        fusion_auxiliary_loss = None
        if gaze_batch is not None and needs_primary_gaze:
            fusion_output = self.fusion(cls_state, text_states, gaze_batch)
            if isinstance(fusion_output, tuple):
                representation, fusion_auxiliary_loss = fusion_output
            else:
                representation = fusion_output
        else:
            representation = cls_state
        logits = self.regression_head(representation)

        auxiliary_loss = None
        if self.training and gaze_batch is not None:
            terms = []
            if fusion_auxiliary_loss is not None:
                terms.append(fusion_auxiliary_loss)
            if self.gaze_prediction is not None:
                terms.append(
                    self.gaze_aux_weight * self.gaze_prediction(text_states, gaze_batch)
                )
            if self.gaze_alignment is not None:
                terms.append(
                    self.gaze_alignment_weight * self.gaze_alignment(text_states, gaze_batch)
                )
            if terms:
                auxiliary_loss = torch.stack(terms).sum()

        if return_dict is False:
            return (logits,)
        return SequenceClassifierOutput(
            loss=auxiliary_loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
