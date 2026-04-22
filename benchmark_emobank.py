import argparse
import csv
import json
import os
import socket
from datetime import datetime

import pandas as pd
import requests

from data_loader import MyDataset
from fold_runner import _build_model, _build_trainer, _build_training_args


EMOBANK_URL = "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv"
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}
MODEL_CHOICES = list(MODEL_TO_CHECKPOINT.keys())
LOSS_CHOICES = ["mse", "ccc", "robust", "mse+ccc", "robust+ccc"]


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


def _download_emobank(raw_csv_path):
    if os.path.isfile(raw_csv_path):
        return
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(EMOBANK_URL, headers=headers, timeout=120)
    response.raise_for_status()
    with open(raw_csv_path, "wb") as output_file:
        output_file.write(response.content)


def _normalize_to_unit_interval(series, scale_min=1.0, scale_max=5.0):
    normalized = (series - scale_min) / (scale_max - scale_min)
    return normalized.clip(0.0, 1.0)


def _write_split_tsv(df, split_name, output_dir):
    split_df = df[df["split"] == split_name].copy()
    split_df["valence"] = _normalize_to_unit_interval(split_df["V"])
    split_df["arousal"] = _normalize_to_unit_interval(split_df["A"])
    out_df = pd.DataFrame(
        {
            "index": range(len(split_df)),
            "text": split_df["text"].astype(str),
            "dataset_of_origin": "Emobank",
            "valence": split_df["valence"],
            "arousal": split_df["arousal"],
        }
    )
    path = os.path.join(output_dir, f"emobank_{split_name}.tsv")
    out_df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    return path


def _prepare_emobank_splits(data_dir):
    raw_csv_path = os.path.join(data_dir, "emobank.csv")
    _download_emobank(raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    train_path = _write_split_tsv(df, "train", data_dir)
    dev_path = _write_split_tsv(df, "dev", data_dir)
    test_path = _write_split_tsv(df, "test", data_dir)
    return train_path, dev_path, test_path


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone EmoBank benchmark runner (train/dev/test)."
    )
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument("--use-gaze-concat", action="store_true")
    parser.add_argument("--et2-checkpoint", default=None)
    parser.add_argument("--features-used", default="1,1,1,1,1")
    parser.add_argument("--fp-dropout", default="0.0,0.3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--data-dir", default="data/emobank")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        features_used = _parse_features_used(args.features_used)
        fp_dropout = _parse_fp_dropout(args.fp_dropout)
        _validate_positive_int("batch_size", args.batch_size)
        _validate_positive_int("train_epochs", args.train_epochs)
        _validate_positive_int("gradient_accumulation_steps", args.gradient_accumulation_steps)
        _validate_positive_int("maxlen", args.maxlen)
    except ValueError as exc:
        parser.error(str(exc))

    if args.use_gaze_concat and args.maxlen > 255:
        parser.error(
            "When --use-gaze-concat is enabled, maxlen must be <= 255 to avoid positional limit overflow."
        )

    checkpoint = MODEL_TO_CHECKPOINT[args.model]
    gaze_config = {
        "use_gaze_concat": args.use_gaze_concat,
        "et2_checkpoint_path": args.et2_checkpoint,
        "features_used": features_used,
        "fp_dropout": fp_dropout,
    }
    train_path, dev_path, test_path = _prepare_emobank_splits(args.data_dir)

    train_data = MyDataset(filename=train_path, checkpoint=checkpoint, maxlen=args.maxlen)
    dev_data = MyDataset(filename=dev_path, checkpoint=checkpoint, maxlen=args.maxlen)
    test_data = MyDataset(filename=test_path, checkpoint=checkpoint, maxlen=args.maxlen)

    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    run_dir = f"Preds/emobank_{timestamp}_{host_name}"
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "batch_size_distil": args.batch_size,
        "batch_size_xlmrB": args.batch_size,
        "batch_size_xlmrL": args.batch_size,
        "lr": args.learning_rate,
        "train_epochs": args.train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "maxlen": args.maxlen,
    }

    model = _build_model(args.model, checkpoint, train_data.tokenizer, gaze_config)
    training_args = _build_training_args(
        output_dir=f"Output Directory/{timestamp}/emobank",
        logging_dir=f"logs/emobank_{timestamp}",
        batch_size=args.batch_size,
        params=params,
    )
    trainer = _build_trainer(args.loss, model, training_args, train_data, dev_data)
    trainer.train()

    dev_result = trainer.predict(dev_data)
    test_result = trainer.predict(test_data)

    pd.DataFrame(dev_result.predictions).to_csv(f"{run_dir}/predictions_dev.csv")
    pd.DataFrame(test_result.predictions).to_csv(f"{run_dir}/predictions_test.csv")

    output = {
        "benchmark": "EmoBank",
        "model": args.model,
        "loss": args.loss,
        "checkpoint": checkpoint,
        "use_gaze_concat": args.use_gaze_concat,
        "et2_checkpoint_path": args.et2_checkpoint,
        "features_used": features_used,
        "fp_dropout": fp_dropout,
        "params": params,
        "dev_metrics": dev_result.metrics,
        "test_metrics": test_result.metrics,
        "paths": {
            "train_tsv": train_path,
            "dev_tsv": dev_path,
            "test_tsv": test_path,
            "run_dir": run_dir,
        },
    }
    with open(f"{run_dir}/benchmark_results.json", "w") as output_file:
        json.dump(output, output_file, indent=2)

    print("EmoBank benchmark finished.")
    print(f"Run dir: {run_dir}")
    print("Dev metrics:", dev_result.metrics)
    print("Test metrics:", test_result.metrics)


if __name__ == "__main__":
    main()
