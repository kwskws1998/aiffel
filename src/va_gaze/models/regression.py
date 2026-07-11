from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaForSequenceClassification,
)
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig

from va_gaze.models.gaze.concat import (
    POSTFIX_CONCAT,
    compose_gaze_concat_inputs,
    normalize_concat_order,
)
from va_gaze.models.gaze_transform import GazeFeatureTransformer


def _normalize_et_model_type(raw_value):
    aliases = {
        "emotion_et": "emotion-et",
        "et_meco": "et-meco",
        "smoke": "heuristic",
    }
    return aliases.get(raw_value or "et2", raw_value or "et2")


def _format_heteroscedastic_logits(logits):
    mu = torch.nn.functional.hardsigmoid(3 * logits[:, :2])
    if logits.shape[-1] <= 2:
        return mu
    return torch.cat([mu, logits[:, 2:]], dim=-1)


class DistilBertForSequenceClassificationSig(DistilBertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = self.sigmoid(ret.logits)
        return ret


class RobertaForSequenceClassificationSig(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)

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
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = self.sigmoid(ret.logits)
        return ret


class XLMRobertaForSequenceClassificationSig(RobertaForSequenceClassificationSig):
    config_class = XLMRobertaConfig


class DistilBertForSequenceClassificationHeteroscedastic(DistilBertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = _format_heteroscedastic_logits(ret.logits)
        return ret


class RobertaForSequenceClassificationHeteroscedastic(RobertaForSequenceClassification):
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
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = _format_heteroscedastic_logits(ret.logits)
        return ret


class XLMRobertaForSequenceClassificationHeteroscedastic(
    RobertaForSequenceClassificationHeteroscedastic
):
    config_class = XLMRobertaConfig


class GazeConcatForSequenceRegression(nn.Module):
    def __init__(
        self,
        checkpoint,
        tokenizer,
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
        concat_order=POSTFIX_CONCAT,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.config = self.encoder.config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.num_labels = output_dim
        self.output_dim = output_dim
        self.concat_order = normalize_concat_order(concat_order)
        if self.__class__ is GazeConcatForSequenceRegression:
            self.config.gaze_concat_order = self.concat_order
        self.et_model_type = _normalize_et_model_type(et_model_type)
        self.gaze_transform_name = gaze_transform or "raw"
        self.feature_indices = None
        self.fp_model = None

        raw_feature_dim = self._configure_fixation_source(
            et2_checkpoint_path=et2_checkpoint_path,
            et_model_id=et_model_id,
            features_used=features_used,
            load_fixation_model=load_fixation_model,
        )
        self.selected_gaze_feature_dim = raw_feature_dim
        self.gaze_feature_transformer = GazeFeatureTransformer(
            transform=self.gaze_transform_name,
            raw_feature_dim=raw_feature_dim,
            artifact_dir=gaze_artifact_dir,
            artifact_repo_id=et_model_id,
            pca_components=pca_components,
            gmm_components=gmm_components,
        )
        self.gaze_feature_dim = self.gaze_feature_transformer.output_dim

        p_1, p_2 = fp_dropout
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(self.gaze_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, self.hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(self.hidden_size)

        classifier_dropout = getattr(self.config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "seq_classif_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "hidden_dropout_prob", 0.1)

        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.output_dim)
        self.sigmoid = torch.nn.functional.hardsigmoid

        self.eye_start = nn.Parameter(torch.zeros(self.hidden_size))
        self.eye_end = nn.Parameter(torch.zeros(self.hidden_size))
        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = max_fix_cache_size

    def _format_logits(self, logits):
        if self.output_dim <= 2:
            return self.sigmoid(logits)
        return _format_heteroscedastic_logits(logits)

    def _configure_fixation_source(
        self,
        et2_checkpoint_path=None,
        et_model_id=None,
        features_used=None,
        load_fixation_model=True,
    ):
        if self.et_model_type in ("et2", "legacy-et2", "heuristic"):
            flags = features_used or [1, 1, 1, 1, 1]
            self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
            if not self.feature_indices:
                raise ValueError("features_used must enable at least one ET feature.")
            if load_fixation_model:
                if self.et_model_type == "heuristic":
                    self.fp_model = self._load_heuristic_predictor()
                else:
                    self.fp_model = self._load_et2_predictor(et2_checkpoint_path)
            return len(self.feature_indices)

        if self.et_model_type == "emotion-et":
            flags = features_used or [1, 1, 1, 1, 1]
            self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
            if not self.feature_indices:
                raise ValueError("features_used must enable at least one ET feature.")
            if load_fixation_model:
                self.fp_model = self._load_emotion_et_predictor(et_model_id or et2_checkpoint_path)
            return len(self.feature_indices)

        if self.et_model_type == "et-meco":
            if load_fixation_model:
                self.fp_model = self._load_et_meco_predictor(et_model_id or et2_checkpoint_path)
                return int(self.fp_model.feature_dim)
            return 8

        raise ValueError(f"Unknown et_model_type: {self.et_model_type}")

    def _load_et2_predictor(self, et2_checkpoint_path):
        try:
            from va_gaze.models.et2_wrapper import FixationsPredictor_2
        except ImportError as exc:
            raise ImportError(
                "Could not import FixationsPredictor_2. Make sure et2_wrapper.py exists and run setup_et_models.py if needed."
            ) from exc

        fp_model = FixationsPredictor_2(
            modelTokenizer=self.tokenizer,
            remap=False,
            checkpoint_path=et2_checkpoint_path,
        )
        if hasattr(fp_model, "model"):
            fp_model.model.eval()
            for param in fp_model.model.parameters():
                param.requires_grad = False
        return fp_model

    def _load_heuristic_predictor(self):
        from va_gaze.models.heuristic_et_wrapper import HeuristicFixationsPredictor

        return HeuristicFixationsPredictor(modelTokenizer=self.tokenizer)

    def _load_emotion_et_predictor(self, et_model_id):
        try:
            from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor
        except ImportError as exc:
            raise ImportError(
                "Could not import EmotionEtFixationsPredictor. Install huggingface_hub/safetensors/transformers."
            ) from exc

        fp_model = EmotionEtFixationsPredictor(
            modelTokenizer=self.tokenizer,
            model_id=et_model_id,
        )
        if hasattr(fp_model, "model"):
            fp_model.model.eval()
            for param in fp_model.model.parameters():
                param.requires_grad = False
        return fp_model

    def _load_et_meco_predictor(self, et_model_id):
        try:
            from va_gaze.models.et_meco_wrapper import MecoFixationsPredictor
        except ImportError as exc:
            raise ImportError(
                "Could not import MecoFixationsPredictor. Install et_meco or set ET_MECO_PACKAGE_ROOT."
            ) from exc

        fp_model = MecoFixationsPredictor(
            modelTokenizer=self.tokenizer,
            checkpoint_path=et_model_id,
        )
        if hasattr(fp_model, "predictor") and hasattr(fp_model.predictor, "model"):
            fp_model.predictor.model.eval()
            for param in fp_model.predictor.model.parameters():
                param.requires_grad = False
        return fp_model

    @staticmethod
    def _build_cache_key(token_ids_1d, attention_mask_1d):
        mask = attention_mask_1d.to(dtype=torch.bool)
        valid_len = int(mask.sum().item())
        if valid_len <= 0:
            return tuple(), valid_len
        if not mask[:valid_len].all() or mask[valid_len:].any():
            raise ValueError(
                "GazeConcat requires a contiguous right-padded attention mask."
            )
        return tuple(token_ids_1d[:valid_len].tolist()), valid_len

    def _predict_fixations_single(self, token_ids_1d, attention_mask_1d):
        device = token_ids_1d.device
        seq_len = token_ids_1d.shape[0]
        key, valid_len = self._build_cache_key(token_ids_1d, attention_mask_1d)

        if valid_len <= 0:
            return (
                torch.zeros(seq_len, self.selected_gaze_feature_dim, dtype=torch.float32, device=device),
                torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device),
            )
        if self.fp_model is None:
            return (
                torch.zeros(seq_len, self.selected_gaze_feature_dim, dtype=torch.float32, device=device),
                torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device),
            )

        cached = self.fixation_cache.get(key)
        if cached is None:
            sample_ids = token_ids_1d[:valid_len].unsqueeze(0)
            sample_mask = attention_mask_1d[:valid_len].unsqueeze(0)
            with torch.no_grad():
                fixations, fixation_mask, _, _, _, _ = self.fp_model._compute_mapped_fixations(
                    sample_ids, sample_mask
                )

            fixations = fixations.squeeze(0).float().cpu()
            fixation_mask = fixation_mask.squeeze(0).long().cpu()
            if self.feature_indices is not None:
                fixations = fixations[:, self.feature_indices]
            finite_mask = torch.isfinite(fixations).all(dim=-1)
            fixation_mask = (
                fixation_mask.to(dtype=torch.bool) & finite_mask
            ).to(dtype=fixation_mask.dtype)
            fixations = torch.nan_to_num(fixations)

            if len(self.fixation_cache) >= self.max_fix_cache_size:
                self.fixation_cache.popitem(last=False)
            self.fixation_cache[key] = (fixations, fixation_mask)
        else:
            fixations, fixation_mask = cached
            self.fixation_cache.move_to_end(key)

        fixations = fixations.to(device)
        fixation_mask = fixation_mask.to(device=device, dtype=attention_mask_1d.dtype)

        padded_fixations = torch.zeros(
            seq_len, self.selected_gaze_feature_dim, dtype=fixations.dtype, device=device
        )
        padded_mask = torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device)

        copy_len = min(valid_len, fixations.shape[0], seq_len)
        padded_fixations[:copy_len] = fixations[:copy_len]
        padded_mask[:copy_len] = fixation_mask[:copy_len].to(dtype=attention_mask_1d.dtype)
        return padded_fixations, padded_mask

    def _compute_fixations_batch(self, input_ids, attention_mask):
        batch_fixations = []
        batch_masks = []
        for row_idx in range(input_ids.size(0)):
            row_fix, row_mask = self._predict_fixations_single(
                input_ids[row_idx], attention_mask[row_idx]
            )
            batch_fixations.append(row_fix)
            batch_masks.append(row_mask)
        fixations = torch.stack(batch_fixations, dim=0)
        masks = torch.stack(batch_masks, dim=0)
        fixations = self.gaze_feature_transformer.transform_tensor(fixations, masks)
        transformed_finite = torch.isfinite(fixations).all(dim=-1)
        masks = masks * transformed_finite.to(dtype=masks.dtype)
        fixations = torch.nan_to_num(fixations)
        fixations = fixations.masked_fill(~masks.to(dtype=torch.bool).unsqueeze(-1), 0.0)
        return fixations, masks

    def _validate_concat_length(self, sequence_length, position_ids=None):
        max_positions = getattr(self.config, "max_position_embeddings", None)
        if max_positions is None:
            return
        max_positions = int(max_positions)
        usable_positions = max_positions
        if self.config.model_type in ("roberta", "xlm-roberta"):
            padding_idx = int(getattr(self.config, "pad_token_id", 0) or 0)
            usable_positions = max_positions - padding_idx - 1
        if int(sequence_length) > usable_positions:
            raise ValueError(
                "Gaze concat sequence length exceeds the encoder position limit: "
                f"{sequence_length} > {usable_positions}. Reduce --maxlen."
            )
        if position_ids is not None and int(position_ids.max().item()) >= max_positions:
            raise ValueError(
                "Extended position_ids exceed the encoder position embedding table."
            )

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
        if inputs_embeds is not None:
            raise ValueError(
                "GazeConcat requires input_ids because gaze prediction is token-id based; "
                "inputs_embeds is not supported."
            )
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(model_device)
        if position_ids is not None:
            position_ids = position_ids.to(model_device)

        text_embeddings = embed_layer(input_ids)
        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)

        fixations_projected = self.fixations_embedding_projector(fixations)
        fixations_projected = self.norm_layer_fix(fixations_projected)
        fixations_projected = fixations_projected.masked_fill(
            ~fixation_attention.to(dtype=torch.bool).unsqueeze(-1),
            0.0,
        )

        concat_inputs = compose_gaze_concat_inputs(
            text_embeddings=text_embeddings,
            gaze_embeddings=fixations_projected,
            text_attention_mask=attention_mask,
            gaze_attention_mask=fixation_attention,
            eye_start=self.eye_start,
            eye_end=self.eye_end,
            order=self.concat_order,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        self._validate_concat_length(
            concat_inputs.inputs_embeds.shape[1],
            concat_inputs.position_ids,
        )

        encoder_kwargs = {
            "input_ids": None,
            "attention_mask": concat_inputs.attention_mask,
            "inputs_embeds": concat_inputs.inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if concat_inputs.token_type_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["token_type_ids"] = concat_inputs.token_type_ids
        if concat_inputs.position_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["position_ids"] = concat_inputs.position_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        batch_indices = torch.arange(
            encoder_outputs.last_hidden_state.shape[0],
            device=encoder_outputs.last_hidden_state.device,
        )
        pooled_output = encoder_outputs.last_hidden_state[
            batch_indices,
            concat_inputs.cls_positions,
        ]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = torch.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self._format_logits(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GazeAddForSequenceRegression(GazeConcatForSequenceRegression):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        gaze_add_scale=0.05,
        train_gaze_add_scale=False,
        et_model_type="et2",
        et_model_id=None,
        gaze_transform="raw",
        gaze_artifact_dir=None,
        pca_components=2,
        gmm_components=5,
        output_dim=2,
    ):
        skip_fixed_zero_gaze = not train_gaze_add_scale and float(gaze_add_scale) == 0.0
        super().__init__(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            load_fixation_model=not skip_fixed_zero_gaze,
            et_model_type=et_model_type,
            et_model_id=et_model_id,
            gaze_transform=gaze_transform,
            gaze_artifact_dir=gaze_artifact_dir,
            pca_components=pca_components,
            gmm_components=gmm_components,
            output_dim=output_dim,
        )
        self.skip_fixed_zero_gaze = skip_fixed_zero_gaze
        gaze_add_scale = torch.tensor(float(gaze_add_scale))
        if train_gaze_add_scale:
            self.gaze_add_scale = nn.Parameter(gaze_add_scale)
        else:
            self.register_buffer("gaze_add_scale", gaze_add_scale)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)
        if self.config.model_type != "distilbert":
            self.config.num_labels = self.num_labels
            self.roberta_classifier = RobertaClassificationHead(self.config)
            self._init_roberta_classifier()

    def _init_roberta_classifier(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        for module in self.roberta_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

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
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        if self.skip_fixed_zero_gaze:
            inputs_embeds = text_embeddings
        else:
            fixations, _ = self._compute_fixations_batch(input_ids, attention_mask)
            fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)

            fixations_projected = self.fixations_embedding_projector(fixations)
            fixations_projected = self.norm_layer_fix(fixations_projected)
            gaze_present = fixations.abs().sum(dim=-1, keepdim=True).gt(0).to(dtype=text_embeddings.dtype)
            fixations_projected = fixations_projected * gaze_present
            inputs_embeds = text_embeddings + self.gaze_add_scale * fixations_projected

        encoder_kwargs = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if token_type_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["position_ids"] = position_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        if self.config.model_type == "distilbert":
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = torch.relu(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = self.roberta_classifier(encoder_outputs.last_hidden_state)
        logits = self._format_logits(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GazeGmmAdapterForSequenceRegression(GazeAddForSequenceRegression):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        gaze_add_scale=0.05,
        train_gaze_add_scale=False,
        et_model_type="et-meco",
        et_model_id=None,
        gaze_transform=None,
        gaze_artifact_dir=None,
        pca_components=2,
        gmm_components=5,
        output_dim=2,
    ):
        super().__init__(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            gaze_add_scale=gaze_add_scale,
            train_gaze_add_scale=train_gaze_add_scale,
            et_model_type=et_model_type,
            et_model_id=et_model_id,
            gaze_transform="gmm",
            gaze_artifact_dir=gaze_artifact_dir,
            pca_components=pca_components,
            gmm_components=gmm_components,
            output_dim=output_dim,
        )
        self.gmm_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(fp_dropout[1]),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                for _ in range(self.gaze_feature_dim)
            ]
        )

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
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=text_embeddings.dtype)

        adapter_outputs = torch.stack(
            [adapter(text_embeddings) for adapter in self.gmm_adapters],
            dim=2,
        )
        residual = (adapter_outputs * fixations.unsqueeze(-1)).sum(dim=2)
        residual = residual * fixation_attention.unsqueeze(-1)
        inputs_embeds = text_embeddings + self.gaze_add_scale * residual

        encoder_kwargs = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if token_type_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["position_ids"] = position_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        if self.config.model_type == "distilbert":
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = torch.relu(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = self.roberta_classifier(encoder_outputs.last_hidden_state)
        logits = self._format_logits(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GazeSummaryForSequenceRegression(GazeConcatForSequenceRegression):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        et_model_type="et2",
        et_model_id=None,
        gaze_transform="raw",
        gaze_artifact_dir=None,
        pca_components=2,
        gmm_components=5,
        output_dim=2,
    ):
        super().__init__(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            et_model_type=et_model_type,
            et_model_id=et_model_id,
            gaze_transform=gaze_transform,
            gaze_artifact_dir=gaze_artifact_dir,
            pca_components=pca_components,
            gmm_components=gmm_components,
            output_dim=output_dim,
        )
        p_1, p_2 = fp_dropout
        self.gaze_summary_projector = nn.Sequential(
            nn.Linear(self.gaze_feature_dim * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=p_1),
        )
        self.summary_pre_classifier = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.summary_dropout = nn.Dropout(p=p_2)
        self.summary_classifier = nn.Linear(self.hidden_size, self.output_dim)

    @staticmethod
    def _summarize_fixations(fixations, fixation_attention):
        nonzero_gaze = fixations.abs().sum(dim=-1).gt(0)
        valid = fixation_attention.to(dtype=torch.bool) & nonzero_gaze
        valid_float = valid.unsqueeze(-1).to(dtype=fixations.dtype)
        count = valid_float.sum(dim=1).clamp_min(1.0)

        mean = (fixations * valid_float).sum(dim=1) / count
        centered = (fixations - mean.unsqueeze(1)) * valid_float
        std = torch.sqrt(torch.square(centered).sum(dim=1) / count + 1e-8)

        neg_inf = torch.finfo(fixations.dtype).min
        masked_fixations = fixations.masked_fill(~valid.unsqueeze(-1), neg_inf)
        max_values = masked_fixations.max(dim=1).values
        has_valid = valid.any(dim=1, keepdim=True)
        max_values = torch.where(has_valid, max_values, torch.zeros_like(max_values))
        return torch.cat([mean, max_values, std], dim=-1)

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
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if token_type_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["position_ids"] = position_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        pooled_text = encoder_outputs.last_hidden_state[:, 0, :]
        pooled_text = self.pre_classifier(pooled_text)
        pooled_text = torch.relu(pooled_text)
        pooled_text = self.dropout(pooled_text)

        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=pooled_text.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)
        gaze_summary = self._summarize_fixations(fixations, fixation_attention)
        gaze_summary = self.gaze_summary_projector(gaze_summary)

        fused = torch.cat([pooled_text, gaze_summary], dim=-1)
        fused = self.summary_pre_classifier(fused)
        fused = torch.relu(fused)
        fused = self.summary_dropout(fused)
        logits = self.summary_classifier(fused)
        logits = self._format_logits(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
