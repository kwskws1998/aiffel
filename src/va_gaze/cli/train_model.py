import argparse
import json
import os
import socket
from datetime import datetime
from signal import signal

from va_gaze.data.dataset import MyDataset
from va_gaze.eval.oof_reports import create_prediction_tables, handle_signal, set_preds_dir
from va_gaze.train.fold1 import training_fold1
from va_gaze.train.fold2 import training_fold2


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_CHOICES = ["distilbert", "xlmroberta-base", "xlmroberta-large"]
LOSS_CHOICES = ["mse", "ccc", "robust", "mse+ccc", "robust+ccc", "hetero"]
ET_MODEL_CHOICES = [
    "et2",
    "emotion-et",
    "emotion_et",
    "et-meco",
    "et_meco",
    "heuristic",
    "smoke",
]
GAZE_TRANSFORM_CHOICES = ["raw", "pca", "gmm"]
GAZE_FUSION_CHOICES = [
    "concat",
    "add",
    "gmm-adapter",
    "summary",
    "conditioned-pooling",
    "pooling",
    "postencoder-cls-attention-bias",
    "cls-attention-bias",
    "attention-bias",
    "cross-attention",
]
GAZE_FUSION_ALIASES = {
    "pooling": "conditioned-pooling",
    "cls-attention-bias": "postencoder-cls-attention-bias",
    "attention-bias": "postencoder-cls-attention-bias",
}
LEGACY_GAZE_FUSIONS = {"concat", "add", "gmm-adapter", "summary"}
MODEL_HIDDEN_SIZES = {
    "distilbert": 768,
    "xlmroberta-base": 768,
    "xlmroberta-large": 1024,
}
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}


def _parse_features_used(raw_value):
    try:
        parsed = [int(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("features_used must be a comma-separated list of integers.") from exc
    if len(parsed) != 5:
        raise ValueError("features_used must contain exactly 5 values (nFix,FFD,GPT,TRT,fixProp).")
    if any(value not in (0, 1) for value in parsed):
        raise ValueError("features_used values must be 0 or 1.")
    if sum(parsed) == 0:
        raise ValueError("At least one gaze feature must be enabled in features_used.")
    return parsed


def _parse_fp_dropout(raw_value):
    try:
        parsed = [float(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("fp_dropout must be a comma-separated list of floats.") from exc
    if len(parsed) != 2:
        raise ValueError("fp_dropout must contain exactly 2 values.")
    return parsed


def _validate_positive_int(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _normalize_et_model_type(raw_value):
    aliases = {
        "emotion_et": "emotion-et",
        "et_meco": "et-meco",
        "smoke": "heuristic",
    }
    return aliases.get(raw_value, raw_value)


def _parse_report_to(raw_value):
    if raw_value is None:
        return None
    targets = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not targets:
        raise ValueError("report_to must be 'none' or a comma-separated reporter list.")
    if "none" in targets:
        if len(targets) != 1:
            raise ValueError("report_to=none cannot be combined with other reporters.")
        return []
    return targets


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument(
        "--checkpoint-override",
        default=None,
        help="Local or Hugging Face encoder checkpoint overriding the model-name default.",
    )
    parser.add_argument("--use-gaze-concat", action="store_true")
    parser.add_argument("--use-gaze-add", action="store_true")
    parser.add_argument("--et2-checkpoint", default=None)
    parser.add_argument("--et-model-type", choices=ET_MODEL_CHOICES, default="et2")
    parser.add_argument("--et-model-id", default=None)
    parser.add_argument("--gaze-transform", choices=GAZE_TRANSFORM_CHOICES, default="raw")
    parser.add_argument("--gaze-fusion", choices=GAZE_FUSION_CHOICES, default=None)
    parser.add_argument("--gaze-artifact-dir", default=None)
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--features-used", default="1,1,1,1,1")
    parser.add_argument("--fp-dropout", default="0.0,0.3")
    parser.add_argument("--gaze-add-scale", type=float, default=0.05)
    parser.add_argument("--train-gaze-add-scale", action="store_true")
    parser.add_argument("--gaze-aux-weight", type=float, default=0.0)
    parser.add_argument("--gaze-alignment-weight", type=float, default=0.0)
    parser.add_argument("--gaze-hidden-size", type=int, default=128)
    parser.add_argument("--gaze-num-heads", type=int, default=4)
    parser.add_argument("--gaze-num-layers", type=int, default=1)
    parser.add_argument("--gaze-gate-init", type=float, default=-4.0)
    parser.add_argument("--gaze-fusion-dropout", type=float, default=0.1)
    parser.add_argument("--gaze-attention-scale", type=float, default=0.1)
    parser.add_argument(
        "--train-gaze-attention-scale",
        dest="train_gaze_attention_scale",
        action="store_true",
    )
    parser.add_argument(
        "--fixed-gaze-attention-scale",
        dest="train_gaze_attention_scale",
        action="store_false",
    )
    parser.add_argument("--gaze-alignment-dim", type=int, default=128)
    parser.add_argument("--gaze-alignment-temperature", type=float, default=0.07)
    parser.add_argument("--gaze-alignment-max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-distil", type=int, default=None)
    parser.add_argument("--batch-size-xlmrb", dest="batch_size_xlmrB", type=int, default=None)
    parser.add_argument("--batch-size-xlmrl", dest="batch_size_xlmrL", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--hetero-mse-weight", type=float, default=0.1)
    parser.add_argument("--hetero-logvar-min", type=float, default=-5.0)
    parser.add_argument("--hetero-logvar-max", type=float, default=3.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--report-to",
        default=None,
        help="Comma-separated Transformers reporters, or 'none' to disable reporting.",
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default=None,
        help="Optional deterministic prediction/output directory (useful for smoke runs).",
    )
    parser.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--save-final-model", dest="save_final_model", action="store_true")
    parser.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")
    parser.add_argument(
        "--load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_true",
    )
    parser.add_argument(
        "--no-load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_false",
    )
    parser.set_defaults(load_best_model_at_end=True)
    parser.set_defaults(save_final_model=True)
    parser.set_defaults(train_gaze_attention_scale=True)
    return parser


def _validate_args(parser, args):
    args.et_model_type = _normalize_et_model_type(args.et_model_type)

    try:
        features_used = _parse_features_used(args.features_used)
        fp_dropout = _parse_fp_dropout(args.fp_dropout)
        _validate_positive_int("train_epochs", args.train_epochs)
        _validate_positive_int("gradient_accumulation_steps", args.gradient_accumulation_steps)
        _validate_positive_int("maxlen", args.maxlen)
        _validate_positive_int("save_total_limit", args.save_total_limit)
        if args.max_steps < -1 or args.max_steps == 0:
            raise ValueError("max_steps must be -1 or > 0.")
        if args.batch_size is not None:
            _validate_positive_int("batch_size", args.batch_size)
        if args.batch_size_distil is not None:
            _validate_positive_int("batch_size_distil", args.batch_size_distil)
        if args.batch_size_xlmrB is not None:
            _validate_positive_int("batch_size_xlmrB", args.batch_size_xlmrB)
        if args.batch_size_xlmrL is not None:
            _validate_positive_int("batch_size_xlmrL", args.batch_size_xlmrL)
        args.report_to = _parse_report_to(args.report_to)
    except ValueError as exc:
        parser.error(str(exc))

    if args.gaze_add_scale < 0:
        parser.error("gaze_add_scale must be >= 0.")

    if args.gaze_aux_weight < 0:
        parser.error("gaze_aux_weight must be >= 0.")
    if args.gaze_alignment_weight < 0:
        parser.error("gaze_alignment_weight must be >= 0.")
    if args.gaze_attention_scale < 0:
        parser.error("gaze_attention_scale must be >= 0.")
    if not 0.0 <= args.gaze_fusion_dropout < 1.0:
        parser.error("gaze_fusion_dropout must be in [0, 1).")
    if args.gaze_alignment_temperature <= 0:
        parser.error("gaze_alignment_temperature must be > 0.")
    if args.gaze_num_layers < 0:
        parser.error("gaze_num_layers must be >= 0.")

    for name, value in (
        ("gaze_hidden_size", args.gaze_hidden_size),
        ("gaze_num_heads", args.gaze_num_heads),
        ("gaze_alignment_dim", args.gaze_alignment_dim),
        ("gaze_alignment_max_tokens", args.gaze_alignment_max_tokens),
    ):
        try:
            _validate_positive_int(name, value)
        except ValueError as exc:
            parser.error(str(exc))

    if args.hetero_mse_weight < 0:
        parser.error("hetero_mse_weight must be >= 0.")

    if args.hetero_logvar_min >= args.hetero_logvar_max:
        parser.error("hetero_logvar_min must be smaller than hetero_logvar_max.")

    if args.use_gaze_concat and args.use_gaze_add:
        parser.error("--use-gaze-concat and --use-gaze-add are mutually exclusive.")

    if args.gaze_fusion and (args.use_gaze_concat or args.use_gaze_add):
        parser.error("Use either --gaze-fusion or legacy --use-gaze-concat/--use-gaze-add, not both.")

    resolved_fusion = args.gaze_fusion
    if resolved_fusion is None:
        if args.use_gaze_concat:
            resolved_fusion = "concat"
        elif args.use_gaze_add:
            resolved_fusion = "add"
    resolved_fusion = GAZE_FUSION_ALIASES.get(resolved_fusion, resolved_fusion)
    has_training_objective = args.gaze_aux_weight > 0 or args.gaze_alignment_weight > 0
    gaze_enabled = bool(
        resolved_fusion
        or has_training_objective
    )

    if resolved_fusion in LEGACY_GAZE_FUSIONS and has_training_objective:
        parser.error(
            "--gaze-aux-weight/--gaze-alignment-weight can be used alone or with "
            "post-encoder gaze fusion, but not with legacy concat/add/summary/gmm-adapter."
        )

    if resolved_fusion in LEGACY_GAZE_FUSIONS and args.et_model_type == "heuristic":
        parser.error(
            "--et-model-type heuristic is smoke-only and is not supported by legacy "
            "concat/add/summary/gmm-adapter fusion."
        )

    if (
        resolved_fusion == "cross-attention"
        and args.gaze_hidden_size % args.gaze_num_heads != 0
    ):
        parser.error("cross-attention requires gaze_hidden_size divisible by gaze_num_heads.")

    if (
        resolved_fusion == "postencoder-cls-attention-bias"
        and args.checkpoint_override is None
        and MODEL_HIDDEN_SIZES[args.model] % args.gaze_num_heads != 0
    ):
        parser.error(
            "postencoder-cls-attention-bias requires the encoder hidden size divisible "
            "by gaze_num_heads."
        )

    if resolved_fusion == "concat" and args.maxlen > 255:
        parser.error(
            "When gaze concat is enabled, maxlen must be <= 255 to avoid positional limit overflow."
        )

    if args.et_model_type == "et-meco" and gaze_enabled and not args.et_model_id:
        parser.error("--et-model-id is required when --et-model-type et-meco is used.")

    if args.et_model_type in ("et2", "emotion-et") and args.gaze_transform != "raw":
        parser.error("PCA/GMM transforms are currently supported for --et-model-type et-meco only.")

    if resolved_fusion == "gmm-adapter":
        if args.et_model_type != "et-meco":
            parser.error("--gaze-fusion gmm-adapter requires --et-model-type et-meco.")
        if args.gaze_transform not in ("raw", "gmm"):
            parser.error("--gaze-fusion gmm-adapter uses GMM posterior and cannot use PCA.")
        args.gaze_transform = "gmm"
    args.gaze_fusion = resolved_fusion

    _validate_positive_int("gmm_components", args.gmm_components)
    _validate_positive_int("pca_components", args.pca_components)

    if args.save_strategy == "no" and args.load_best_model_at_end:
        args.load_best_model_at_end = False
        print("[train_model] save_strategy=no, so load_best_model_at_end was set to False.")

    return features_used, fp_dropout


def _resolve_batch_sizes(args):
    base_batch_size = args.batch_size if args.batch_size is not None else 16
    batch_size_distil = args.batch_size_distil if args.batch_size_distil is not None else base_batch_size
    batch_size_xlmrB = args.batch_size_xlmrB if args.batch_size_xlmrB is not None else base_batch_size
    batch_size_xlmrL = args.batch_size_xlmrL if args.batch_size_xlmrL is not None else base_batch_size
    return batch_size_distil, batch_size_xlmrB, batch_size_xlmrL


def _create_run_dir(preds_dir_override=None):
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    preds_dir = preds_dir_override or f"Preds/{timestamp}_{host_name}"
    os.makedirs(preds_dir, exist_ok=bool(preds_dir_override))
    set_preds_dir(preds_dir)
    return timestamp, preds_dir


def _save_training_parameters(preds_dir, run_parameters):
    with open(f"{preds_dir}/training_parameters.json", "w") as output_file:
        json.dump(run_parameters, output_file)


def _load_dataset(checkpoint, maxlen, data_dir):
    filename_1 = os.path.join(data_dir, "full_dataset_fold1.csv")
    filename_2 = os.path.join(data_dir, "full_dataset_fold2.csv")
    split_1 = MyDataset(filename=filename_1, checkpoint=checkpoint, maxlen=maxlen)
    split_2 = MyDataset(filename=filename_2, checkpoint=checkpoint, maxlen=maxlen)
    return [[split_1, split_2], [split_2, split_1]]


def main():
    signal(2, handle_signal)
    parser = _build_parser()
    args = parser.parse_args()

    features_used, fp_dropout = _validate_args(parser, args)
    batch_size_distil, batch_size_xlmrB, batch_size_xlmrL = _resolve_batch_sizes(args)
    checkpoint = args.checkpoint_override or MODEL_TO_CHECKPOINT[args.model]
    gaze_config = {
        "use_gaze_concat": args.use_gaze_concat,
        "use_gaze_add": args.use_gaze_add,
        "et2_checkpoint_path": args.et2_checkpoint,
        "et_model_type": args.et_model_type,
        "et_model_id": args.et_model_id,
        "gaze_transform": args.gaze_transform,
        "gaze_fusion": args.gaze_fusion,
        "gaze_artifact_dir": args.gaze_artifact_dir,
        "gmm_components": args.gmm_components,
        "pca_components": args.pca_components,
        "features_used": features_used,
        "fp_dropout": fp_dropout,
        "gaze_add_scale": args.gaze_add_scale,
        "train_gaze_add_scale": args.train_gaze_add_scale,
        "gaze_aux_weight": args.gaze_aux_weight,
        "gaze_alignment_weight": args.gaze_alignment_weight,
        "gaze_hidden_size": args.gaze_hidden_size,
        "gaze_num_heads": args.gaze_num_heads,
        "gaze_num_layers": args.gaze_num_layers,
        "gaze_gate_init": args.gaze_gate_init,
        "gaze_fusion_dropout": args.gaze_fusion_dropout,
        "gaze_attention_scale": args.gaze_attention_scale,
        "train_gaze_attention_scale": args.train_gaze_attention_scale,
        "gaze_alignment_dim": args.gaze_alignment_dim,
        "gaze_alignment_temperature": args.gaze_alignment_temperature,
        "gaze_alignment_max_tokens": args.gaze_alignment_max_tokens,
    }

    timestamp, preds_dir = _create_run_dir(args.preds_dir)
    params = {
        "batch_size_distil": batch_size_distil,
        "batch_size_xlmrB": batch_size_xlmrB,
        "batch_size_xlmrL": batch_size_xlmrL,
        "lr": args.learning_rate,
        "train_epochs": args.train_epochs,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "hetero_mse_weight": args.hetero_mse_weight,
        "hetero_logvar_min": args.hetero_logvar_min,
        "hetero_logvar_max": args.hetero_logvar_max,
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "maxlen": args.maxlen,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "save_final_model": args.save_final_model,
        "load_best_model_at_end": args.load_best_model_at_end,
        "data_dir": args.data_dir,
        "report_to": args.report_to,
    }
    run_parameters = {
        "model": args.model,
        "checkpoint": checkpoint,
        "checkpoint_override": args.checkpoint_override,
        "loss_function": args.loss,
        "use_gaze_concat": gaze_config["use_gaze_concat"],
        "use_gaze_add": gaze_config["use_gaze_add"],
        "et2_checkpoint_path": gaze_config["et2_checkpoint_path"],
        "et_model_type": gaze_config["et_model_type"],
        "et_model_id": gaze_config["et_model_id"],
        "gaze_transform": gaze_config["gaze_transform"],
        "gaze_fusion": gaze_config["gaze_fusion"],
        "gaze_artifact_dir": gaze_config["gaze_artifact_dir"],
        "gmm_components": gaze_config["gmm_components"],
        "pca_components": gaze_config["pca_components"],
        "features_used": gaze_config["features_used"],
        "fp_dropout": gaze_config["fp_dropout"],
        "gaze_add_scale": gaze_config["gaze_add_scale"],
        "train_gaze_add_scale": gaze_config["train_gaze_add_scale"],
        "gaze_aux_weight": gaze_config["gaze_aux_weight"],
        "gaze_alignment_weight": gaze_config["gaze_alignment_weight"],
        "gaze_hidden_size": gaze_config["gaze_hidden_size"],
        "gaze_num_heads": gaze_config["gaze_num_heads"],
        "gaze_num_layers": gaze_config["gaze_num_layers"],
        "gaze_gate_init": gaze_config["gaze_gate_init"],
        "gaze_fusion_dropout": gaze_config["gaze_fusion_dropout"],
        "gaze_attention_scale": gaze_config["gaze_attention_scale"],
        "train_gaze_attention_scale": gaze_config["train_gaze_attention_scale"],
        "gaze_alignment_dim": gaze_config["gaze_alignment_dim"],
        "gaze_alignment_temperature": gaze_config["gaze_alignment_temperature"],
        "gaze_alignment_max_tokens": gaze_config["gaze_alignment_max_tokens"],
        "path": preds_dir,
        **params,
    }
    _save_training_parameters(preds_dir, run_parameters)

    dataset = _load_dataset(checkpoint, args.maxlen, args.data_dir)
    training_fold1(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    print("\n\n\n------------ NOW ON FOLD 2 -------------- \n\n\n")
    training_fold2(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    create_prediction_tables(preds_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
