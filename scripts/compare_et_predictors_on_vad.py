"""Compare generic ET2 and emotion-specific ET predictors on VAD text data."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from safetensors.torch import load_file
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor


FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
WINDOW_SIZE = 512
OVERLAP = 50


class _OfflineRobertaRegressionModel(torch.nn.Module):
    def __init__(self, config_dir: Path):
        super().__init__()
        config = RobertaConfig.from_pretrained(config_dir)
        self.roberta = RobertaModel(config)
        self.decoder = torch.nn.Linear(config.hidden_size, len(FEATURE_NAMES))

    def forward(self, input_ids, attention_mask, predict_mask):
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        pred = self.decoder(hidden)
        mask = (predict_mask == 0).unsqueeze(-1).expand_as(pred).to(pred.device)
        return pred.masked_fill(mask, -1.0)


class OfflineEt2Predictor:
    def __init__(self, checkpoint_path: Path, tokenizer_dir: Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir, add_prefix_space=True)
        self.model = _OfflineRobertaRegressionModel(tokenizer_dir).to(self.device)
        state = load_file(str(checkpoint_path), device=str(self.device))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _predict_words(self, text: str):
        words = self._segment_text(text)
        if not words:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), words

        encoded = self.roberta_tokenizer(
            [words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_preds = self._sliding_window_predict(input_ids, attention_mask)
        word_features = self._aggregate_to_words(token_preds, input_ids.squeeze(0))
        return word_features, words

    @staticmethod
    def _segment_text(text: str):
        import re

        text = (text or "").strip()
        if not text:
            return []
        if any(ch.isspace() for ch in text):
            words = text.split()
            if words:
                return words
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _sliding_window_predict(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        if seq_len <= WINDOW_SIZE:
            with torch.no_grad():
                pred = self.model(input_ids, attention_mask, attention_mask.clone())
            return pred.squeeze(0).detach().cpu().numpy()

        preds = np.zeros((seq_len, len(FEATURE_NAMES)), dtype=np.float32)
        weights = np.zeros(seq_len, dtype=np.float32)
        stride = WINDOW_SIZE - OVERLAP
        start = 0

        while start < seq_len:
            end = min(start + WINDOW_SIZE, seq_len)
            ids_win = input_ids[:, start:end]
            mask_win = attention_mask[:, start:end]
            win_len = end - start
            linear_w = np.ones(win_len, dtype=np.float32)
            if start > 0:
                ramp_len = min(OVERLAP, win_len)
                linear_w[:ramp_len] = np.linspace(0, 1, ramp_len)
            if end < seq_len:
                ramp_len = min(OVERLAP, win_len)
                linear_w[-ramp_len:] = np.linspace(1, 0, ramp_len)

            with torch.no_grad():
                pred_win = self.model(ids_win, mask_win, mask_win.clone())
            pred_np = pred_win.squeeze(0).detach().cpu().numpy()
            preds[start:end] += pred_np * linear_w[:, None]
            weights[start:end] += linear_w

            if end == seq_len:
                break
            start += stride

        valid = weights > 0
        preds[valid] /= weights[valid, None]
        return preds

    def _aggregate_to_words(self, token_preds, input_ids_1d):
        tokens = [self.roberta_tokenizer.convert_ids_to_tokens(int(i)) for i in input_ids_1d]
        word_features = []
        seen_first_word = False
        for idx, token in enumerate(tokens):
            if token in ("<s>", "</s>", "<pad>"):
                continue
            if token.startswith("Ġ") or not seen_first_word:
                word_features.append(np.clip(token_preds[idx], 0, None))
                seen_first_word = True
        if word_features:
            return np.array(word_features, dtype=np.float32)
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)


def _read_vad_tsv(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            data = pd.read_csv(handle, sep="\t")
    required = {"text", "valence", "arousal"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{member} is missing required columns: {sorted(missing)}")
    data = data.dropna(subset=["text", "valence", "arousal"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0].copy()
    for column in ["valence", "arousal", "dominance"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


def _filter_sentence_like_rows(data: pd.DataFrame, min_alpha_words: int) -> pd.DataFrame:
    if min_alpha_words <= 0:
        return data.copy()

    def count_alpha_words(text: str) -> int:
        return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))

    filtered = data.copy()
    filtered["alpha_word_count"] = filtered["text"].map(count_alpha_words)
    filtered = filtered[filtered["alpha_word_count"] >= min_alpha_words].copy()
    return filtered.reset_index(drop=True)


def _sample_vad_rows(data: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(data):
        return data.copy()

    sampled_parts = []
    work = data.copy()
    work["valence_bin"] = pd.qcut(work["valence"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    work["arousal_bin"] = pd.qcut(work["arousal"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    work["stratum"] = work["valence_bin"].astype(str) + "_" + work["arousal_bin"].astype(str)
    per_group = max(1, math.ceil(sample_size / work["stratum"].nunique()))

    for _, group in work.groupby("stratum", dropna=False):
        sampled_parts.append(group.sample(n=min(per_group, len(group)), random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.sort_index().drop(columns=["valence_bin", "arousal_bin", "stratum"]).reset_index(drop=True)


def _load_predictors(et2_checkpoint: Path, emotion_model_dir: Path):
    old_model = OfflineEt2Predictor(
        checkpoint_path=et2_checkpoint,
        tokenizer_dir=emotion_model_dir,
    )
    tokenizer = RobertaTokenizer.from_pretrained(emotion_model_dir, add_prefix_space=True)
    emotion_model = EmotionEtFixationsPredictor(
        modelTokenizer=tokenizer,
        model_id=str(emotion_model_dir),
    )
    return old_model, emotion_model


def _predict_sentence_stats(predictor, text: str, prefix: str) -> dict[str, object]:
    features, words = predictor._predict_words(text)
    result: dict[str, object] = {
        f"{prefix}_n_words": len(words),
    }
    if features.size == 0:
        for name in FEATURE_NAMES:
            result[f"{prefix}_{name}_mean"] = np.nan
            result[f"{prefix}_{name}_max"] = np.nan
            result[f"{prefix}_{name}_p90"] = np.nan
        return result

    for idx, name in enumerate(FEATURE_NAMES):
        values = features[:, idx].astype(float)
        result[f"{prefix}_{name}_mean"] = float(np.nanmean(values))
        result[f"{prefix}_{name}_max"] = float(np.nanmax(values))
        result[f"{prefix}_{name}_p90"] = float(np.nanquantile(values, 0.90))
    return result


def _add_analysis_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["abs_valence_from_neutral"] = (data["valence"] - 0.5).abs()
    data["valence_bin"] = pd.qcut(data["valence"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    data["arousal_bin"] = pd.qcut(data["arousal"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    data["emotion_salience_bin"] = pd.qcut(
        data["abs_valence_from_neutral"],
        q=3,
        labels=["near_neutral", "medium", "far_from_neutral"],
        duplicates="drop",
    )
    for feature in FEATURE_NAMES:
        data[f"delta_{feature}_mean_emotion_minus_et2"] = (
            data[f"emotion_et_{feature}_mean"] - data[f"et2_{feature}_mean"]
        )
    return data


def _summarize_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("valence_bin", "valence_bin"),
        ("arousal_bin", "arousal_bin"),
        ("emotion_salience_bin", "emotion_salience_bin"),
    ]
    rows = []
    for group_name, column in group_specs:
        for value, group in predictions.groupby(column, observed=False):
            row = {
                "group_type": group_name,
                "group": str(value),
                "n": len(group),
                "valence_mean": group["valence"].mean(),
                "arousal_mean": group["arousal"].mean(),
            }
            if "dominance" in group.columns:
                row["dominance_mean"] = group["dominance"].mean()
            for prefix in ["et2", "emotion_et"]:
                row[f"{prefix}_TRT_mean"] = group[f"{prefix}_TRT_mean"].mean()
                row[f"{prefix}_TRT_p90_mean"] = group[f"{prefix}_TRT_p90"].mean()
                row[f"{prefix}_TRT_max_mean"] = group[f"{prefix}_TRT_max"].mean()
            row["delta_TRT_mean_emotion_minus_et2"] = (
                row["emotion_et_TRT_mean"] - row["et2_TRT_mean"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _compute_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    targets = ["valence", "arousal", "abs_valence_from_neutral"]
    if "dominance" in predictions.columns:
        targets.append("dominance")

    predictors = []
    for prefix in ["et2", "emotion_et"]:
        predictors.extend(
            [
                f"{prefix}_TRT_mean",
                f"{prefix}_TRT_p90",
                f"{prefix}_TRT_max",
                f"{prefix}_FFD_mean",
                f"{prefix}_GPT_mean",
            ]
        )
    predictors.append("delta_TRT_mean_emotion_minus_et2")

    rows = []
    for target in targets:
        for predictor in predictors:
            subset = predictions[[target, predictor]].dropna()
            if len(subset) < 3:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(subset[target], subset[predictor])
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "spearman": rho,
                    "p_value": p_value,
                    "n": len(subset),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    metadata: dict[str, object],
    group_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = [
        "# ET Predictor Comparison On VAD Data",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            group_summary.round(6).to_markdown(index=False),
            "",
            "## TRT Correlations",
            "",
            correlations.sort_values(["target", "predictor"]).round(6).to_markdown(index=False),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-zip", type=Path, default=Path("/Users/wansookim/Downloads/Archive (1).zip"))
    parser.add_argument("--member", default="emobank.tsv")
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--min-alpha-words", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--et2-checkpoint", type=Path, default=Path("hf_checks/et_prediction_2/et_predictor2_seed123.safetensors"))
    parser.add_argument("--emotion-model-dir", type=Path, default=Path("hf_checks/emotion_et_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/et_predictor_vad_emobank"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vad_data_all = _read_vad_tsv(args.vad_zip, args.member)
    vad_data = _filter_sentence_like_rows(vad_data_all, args.min_alpha_words)
    sampled = _sample_vad_rows(vad_data, args.sample_size, args.seed)

    et2_model, emotion_model = _load_predictors(args.et2_checkpoint, args.emotion_model_dir)
    rows = []
    for row_idx, row in sampled.iterrows():
        text = str(row["text"])
        out = row.to_dict()
        out["sample_index"] = int(row_idx)
        out.update(_predict_sentence_stats(et2_model, text, "et2"))
        out.update(_predict_sentence_stats(emotion_model, text, "emotion_et"))
        rows.append(out)
        if len(rows) % 50 == 0:
            print(f"processed {len(rows)}/{len(sampled)}")

    predictions = _add_analysis_columns(pd.DataFrame(rows))
    group_summary = _summarize_groups(predictions)
    correlations = _compute_correlations(predictions)

    predictions.to_csv(args.output_dir / "sentence_predictions.csv", index=False)
    group_summary.to_csv(args.output_dir / "group_summary.csv", index=False)
    correlations.to_csv(args.output_dir / "spearman_correlations.csv", index=False)

    metadata = {
        "vad_zip": str(args.vad_zip),
        "member": args.member,
        "rows_before_filter": len(vad_data_all),
        "rows_after_filter": len(vad_data),
        "min_alpha_words": args.min_alpha_words,
        "sample_size_requested": args.sample_size,
        "sample_size_used": len(sampled),
        "seed": args.seed,
        "et2_checkpoint": str(args.et2_checkpoint),
        "emotion_model_dir": str(args.emotion_model_dir),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(args.output_dir, metadata, group_summary, correlations)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
