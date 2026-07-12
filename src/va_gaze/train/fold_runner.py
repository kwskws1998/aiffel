import inspect

import numpy as np
import pandas as pd
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from va_gaze.train.custom_trainer import (
    CustomTrainerCCC,
    CustomTrainerHeteroscedastic,
    CustomTrainerMSE,
    CustomTrainerMSE_CCC,
    CustomTrainerRobust,
    CustomTrainerRobustCCC,
)
from va_gaze.eval.metrics import compute_metrics
from va_gaze.models.advanced_regression import (
    CANONICAL_ADVANCED_GAZE_FUSIONS,
    GazeFusionForSequenceRegression,
    normalize_advanced_fusion,
)
from va_gaze.models.regression import (
    DistilBertForSequenceClassificationSig,
    DistilBertForSequenceClassificationHeteroscedastic,
    GazeAddForSequenceRegression,
    GazeConcatForSequenceRegression,
    GazeGmmAdapterForSequenceRegression,
    GazeSummaryForSequenceRegression,
    XLMRobertaForSequenceClassificationHeteroscedastic,
    XLMRobertaForSequenceClassificationSig,
)


LOSS_TO_TRAINER = {
    "mse": CustomTrainerMSE,
    "ccc": CustomTrainerCCC,
    "robust": CustomTrainerRobust,
    "mse+ccc": CustomTrainerMSE_CCC,
    "robust+ccc": CustomTrainerRobustCCC,
    "hetero": CustomTrainerHeteroscedastic,
}

CONCAT_FUSION_ALIASES = {
    "concat": "postfix-concat",
    "concat-postfix": "postfix-concat",
    "concat-prefix": "prefix-concat",
}
OBJECTIVE_INCOMPATIBLE_GAZE_FUSIONS = {
    "postfix-concat",
    "prefix-concat",
    "add",
    "gmm-adapter",
    "summary",
}


def _select_batch_size(model_name, params):
    if model_name == "distilbert":
        return params["batch_size_distil"]
    if model_name == "xlmroberta-base":
        return params["batch_size_xlmrB"]
    if model_name == "xlmroberta-large":
        return params["batch_size_xlmrL"]
    raise ValueError(f"Unknown model name: {model_name}")


def _build_model(model_name, checkpoint, tokenizer, gaze_config, output_dim=2):
    gaze_fusion = gaze_config.get("gaze_fusion")
    if not gaze_fusion:
        if bool(gaze_config.get("use_gaze_concat", False)):
            gaze_fusion = "postfix-concat"
        elif bool(gaze_config.get("use_gaze_add", False)):
            gaze_fusion = "add"
        else:
            gaze_fusion = "none"
    gaze_fusion = CONCAT_FUSION_ALIASES.get(gaze_fusion, gaze_fusion)

    shared_gaze_kwargs = {
        "et2_checkpoint_path": gaze_config.get("et2_checkpoint_path"),
        "features_used": gaze_config.get("features_used", [1, 1, 1, 1, 1]),
        "fp_dropout": tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
        "et_model_type": gaze_config.get("et_model_type", "et2"),
        "et_model_id": gaze_config.get("et_model_id"),
        "gaze_transform": gaze_config.get("gaze_transform", "raw"),
        "gaze_artifact_dir": gaze_config.get("gaze_artifact_dir"),
        "pca_components": gaze_config.get("pca_components", 2),
        "gmm_components": gaze_config.get("gmm_components", 5),
        "output_dim": output_dim,
    }

    gaze_aux_weight = float(gaze_config.get("gaze_aux_weight", 0.0))
    gaze_alignment_weight = float(gaze_config.get("gaze_alignment_weight", 0.0))
    has_training_objective = gaze_aux_weight > 0 or gaze_alignment_weight > 0
    normalized_advanced_fusion = normalize_advanced_fusion(gaze_fusion)
    if (
        gaze_fusion != "none"
        and gaze_fusion not in OBJECTIVE_INCOMPATIBLE_GAZE_FUSIONS
        and normalized_advanced_fusion not in CANONICAL_ADVANCED_GAZE_FUSIONS
    ):
        raise ValueError(f"Unknown gaze fusion strategy: {gaze_fusion}")
    if gaze_fusion in OBJECTIVE_INCOMPATIBLE_GAZE_FUSIONS and has_training_objective:
        raise ValueError(
            "Training-only gaze objectives cannot be combined with "
            "postfix-concat/prefix-concat/add/summary/gmm-adapter fusion."
        )

    if (
        normalized_advanced_fusion in CANONICAL_ADVANCED_GAZE_FUSIONS
        or has_training_objective
    ):
        fusion_strategy = (
            normalized_advanced_fusion
            if normalized_advanced_fusion in CANONICAL_ADVANCED_GAZE_FUSIONS
            else "none"
        )
        return GazeFusionForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            fusion_strategy=fusion_strategy,
            gaze_aux_weight=gaze_aux_weight,
            gaze_alignment_weight=gaze_alignment_weight,
            gaze_hidden_size=gaze_config.get("gaze_hidden_size", 128),
            gaze_num_heads=gaze_config.get("gaze_num_heads", 4),
            gaze_num_layers=gaze_config.get("gaze_num_layers", 1),
            gaze_gate_init=gaze_config.get("gaze_gate_init", -4.0),
            gaze_fusion_dropout=gaze_config.get("gaze_fusion_dropout", 0.1),
            gaze_attention_scale=gaze_config.get("gaze_attention_scale", 0.1),
            train_gaze_attention_scale=gaze_config.get(
                "train_gaze_attention_scale", True
            ),
            gaze_alignment_dim=gaze_config.get("gaze_alignment_dim", 128),
            gaze_alignment_temperature=gaze_config.get(
                "gaze_alignment_temperature", 0.07
            ),
            gaze_alignment_max_tokens=gaze_config.get(
                "gaze_alignment_max_tokens", 512
            ),
            gmm_temperature=gaze_config.get("gmm_temperature", 1.0),
            gmm_nll_weight=gaze_config.get("gmm_nll_weight", 0.01),
            **shared_gaze_kwargs,
        )

    if gaze_fusion == "postfix-concat":
        return GazeConcatForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            concat_order="postfix",
            **shared_gaze_kwargs,
        )
    if gaze_fusion == "prefix-concat":
        return GazeConcatForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            concat_order="prefix",
            **shared_gaze_kwargs,
        )
    if gaze_fusion == "add":
        return GazeAddForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            gaze_add_scale=gaze_config.get("gaze_add_scale", 0.05),
            train_gaze_add_scale=gaze_config.get("train_gaze_add_scale", False),
            **shared_gaze_kwargs,
        )
    if gaze_fusion == "gmm-adapter":
        return GazeGmmAdapterForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            gaze_add_scale=gaze_config.get("gaze_add_scale", 0.05),
            train_gaze_add_scale=gaze_config.get("train_gaze_add_scale", False),
            **shared_gaze_kwargs,
        )
    if gaze_fusion == "summary":
        return GazeSummaryForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            **shared_gaze_kwargs,
        )

    if model_name == "distilbert":
        if output_dim > 2:
            return DistilBertForSequenceClassificationHeteroscedastic.from_pretrained(
                checkpoint,
                num_labels=output_dim,
            )
        return DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    if model_name in ("xlmroberta-base", "xlmroberta-large"):
        if output_dim > 2:
            return XLMRobertaForSequenceClassificationHeteroscedastic.from_pretrained(
                checkpoint,
                num_labels=output_dim,
            )
        return XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    raise ValueError(f"Unknown model name: {model_name}")


def _build_training_args(output_dir, logging_dir, batch_size, params):
    save_strategy = params.get("save_strategy", "epoch")
    load_best_model_at_end = params.get("load_best_model_at_end", True)
    if save_strategy == "no":
        load_best_model_at_end = False

    training_kwargs = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "logging_steps": 200,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": params["train_epochs"],
        "max_steps": params.get("max_steps", -1),
        "learning_rate": params["lr"],
        "weight_decay": params["weight_decay"],
        "optim": params.get("optim", "adamw_torch"),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
        "seed": params.get("seed", 42),
        "group_by_length": True,
        "save_strategy": save_strategy,
        "save_total_limit": params.get("save_total_limit", 1),
        "load_best_model_at_end": load_best_model_at_end,
        "warmup_ratio": params["warmup_ratio"],
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    if params.get("report_to") is not None:
        training_kwargs["report_to"] = params["report_to"]
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    training_kwargs[strategy_name] = "epoch"
    return TrainingArguments(**training_kwargs)


def _build_trainer(loss_name, model, training_args, train_data, val_data, params):
    trainer_cls = LOSS_TO_TRAINER.get(loss_name)
    if trainer_cls is None:
        raise ValueError(f"Unknown loss name: {loss_name}")
    trainer_kwargs = {}
    if loss_name == "hetero":
        trainer_kwargs.update(
            {
                "hetero_mse_weight": params.get("hetero_mse_weight", 0.1),
                "hetero_logvar_min": params.get("hetero_logvar_min", -5.0),
                "hetero_logvar_max": params.get("hetero_logvar_max", 3.0),
            }
        )
    trainer_kwargs.update(
        {
            "data_collator": DataCollatorWithPadding(train_data.tokenizer),
            "train_dataset": train_data,
            "eval_dataset": val_data,
            "compute_metrics": compute_metrics,
        }
    )
    trainer_argument_names = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_argument_names:
        trainer_kwargs["processing_class"] = train_data.tokenizer
    else:
        trainer_kwargs["tokenizer"] = train_data.tokenizer
    return trainer_cls(
        model=model,
        args=training_args,
        **trainer_kwargs,
    )


def _validate_prediction_array(predictions, output_dim):
    if not isinstance(predictions, np.ndarray):
        raise TypeError(
            "Trainer predictions must be a single numpy.ndarray; hidden states or "
            "attentions may have leaked into the prediction output."
        )
    if predictions.ndim != 2:
        raise ValueError(
            f"Trainer predictions must be 2D, got shape {predictions.shape}."
        )
    if predictions.shape[1] != int(output_dim):
        raise ValueError(
            f"Trainer predictions must have {output_dim} columns, got {predictions.shape[1]}."
        )
    if not np.issubdtype(predictions.dtype, np.number):
        raise TypeError(f"Trainer predictions must be numeric, got dtype {predictions.dtype}.")
    if not np.isfinite(predictions).all():
        raise ValueError("Trainer predictions contain NaN or infinite values.")
    return predictions


def run_fold(
    fold_id,
    model_name,
    loss_name,
    timestamp,
    params,
    train_data,
    val_data,
    preds_dir,
    checkpoint,
    prediction_filename,
    metrics_filename,
    gaze_config=None,
):
    gaze_config = gaze_config or {}
    output_dir = f"Output Directory/{timestamp}/fold{fold_id}"
    model_dir = f"model/{timestamp}/fold{fold_id}"
    logging_dir = f"logs/logs{fold_id}"
    batch_size = _select_batch_size(model_name, params)

    output_dim = 4 if loss_name == "hetero" else 2
    set_seed(params.get("seed", 42))
    model = _build_model(
        model_name,
        checkpoint,
        train_data.tokenizer,
        gaze_config,
        output_dim=output_dim,
    )
    training_args = _build_training_args(output_dir, logging_dir, batch_size, params)
    trainer = _build_trainer(loss_name, model, training_args, train_data, val_data, params)

    print(f"Starting fold {fold_id}")
    trainer.train()
    predictions = trainer.predict(
        val_data,
        ignore_keys=["hidden_states", "attentions"],
    )
    prediction_array = _validate_prediction_array(predictions.predictions, output_dim)

    pd.DataFrame(prediction_array).to_csv(f"{preds_dir}/{prediction_filename}")
    with open(f"{preds_dir}/{metrics_filename}", "w") as output_file:
        for key, value in predictions.metrics.items():
            output_file.write(f"{key},{value}\n")

    if params.get("save_final_model", True):
        trainer.save_model(model_dir)
