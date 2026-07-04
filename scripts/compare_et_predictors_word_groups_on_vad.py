"""Word-level ET predictor comparison on VAD text using affective lexicons."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import RobertaTokenizer

from compare_et_predictors_on_vad import (
    FEATURE_NAMES,
    OfflineEt2Predictor,
    _filter_sentence_like_rows,
    _read_vad_tsv,
    _sample_vad_rows,
)
from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
    "yours",
}


def _normalize_word(word: str) -> str:
    word = str(word).lower()
    word = word.replace("’", "'").replace("`", "'")
    word = re.sub(r"^[^a-z]+|[^a-z]+$", "", word)
    return word


def _load_lexicon(zip_path: Path, members: list[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    with zipfile.ZipFile(zip_path) as archive:
        for source_rank, member in enumerate(members):
            with archive.open(member) as handle:
                data = pd.read_csv(handle, sep="\t")
            for _, row in data.iterrows():
                word = _normalize_word(row["text"])
                if not word or word in seen:
                    continue
                try:
                    valence = float(row["valence"])
                    arousal = float(row["arousal"])
                    dominance = float(row["dominance"]) if "dominance" in row else np.nan
                except Exception:
                    continue
                seen.add(word)
                rows.append(
                    {
                        "norm_word": word,
                        "lexicon_source": member,
                        "lexicon_rank": source_rank,
                        "lex_valence": valence,
                        "lex_arousal": arousal,
                        "lex_dominance": dominance,
                    }
                )
    return pd.DataFrame(rows)


def _classify_word(
    norm_word: str,
    lex_row: pd.Series | None,
    emotion_valence_abs_min: float,
    emotion_arousal_min: float,
    neutral_valence_abs_max: float,
    neutral_arousal_abs_max: float,
) -> str:
    if not norm_word:
        return "non_alpha"
    if norm_word in STOPWORDS:
        return "function_word"
    if lex_row is None:
        return "unmatched_alpha"

    valence = float(lex_row["lex_valence"])
    arousal = float(lex_row["lex_arousal"])
    valence_abs = abs(valence - 0.5)
    arousal_abs = abs(arousal - 0.5)

    if valence_abs >= emotion_valence_abs_min or arousal >= emotion_arousal_min:
        return "emotion_word"
    if valence_abs <= neutral_valence_abs_max and arousal_abs <= neutral_arousal_abs_max:
        return "non_emotion_word"
    return "matched_other"


def _load_predictors(et2_checkpoint: Path, emotion_model_dir: Path):
    et2_model = OfflineEt2Predictor(
        checkpoint_path=et2_checkpoint,
        tokenizer_dir=emotion_model_dir,
    )
    tokenizer = RobertaTokenizer.from_pretrained(emotion_model_dir, add_prefix_space=True)
    emotion_model = EmotionEtFixationsPredictor(
        modelTokenizer=tokenizer,
        model_id=str(emotion_model_dir),
    )
    return et2_model, emotion_model


def _predict_word_rows(
    row: pd.Series,
    sentence_idx: int,
    et2_model,
    emotion_model,
    lexicon_by_word: dict[str, pd.Series],
    args,
) -> list[dict[str, object]]:
    text = str(row["text"])
    et2_features, et2_words = et2_model._predict_words(text)
    emotion_features, emotion_words = emotion_model._predict_words(text)
    n_words = min(len(et2_words), len(emotion_words), len(et2_features), len(emotion_features))

    rows = []
    for word_idx in range(n_words):
        word = str(et2_words[word_idx])
        norm_word = _normalize_word(word)
        lex_row = lexicon_by_word.get(norm_word)
        word_group = _classify_word(
            norm_word=norm_word,
            lex_row=lex_row,
            emotion_valence_abs_min=args.emotion_valence_abs_min,
            emotion_arousal_min=args.emotion_arousal_min,
            neutral_valence_abs_max=args.neutral_valence_abs_max,
            neutral_arousal_abs_max=args.neutral_arousal_abs_max,
        )
        out = {
            "sentence_index": sentence_idx,
            "word_index": word_idx,
            "word": word,
            "norm_word": norm_word,
            "word_group": word_group,
            "sentence_text": text,
            "sentence_valence": row["valence"],
            "sentence_arousal": row["arousal"],
            "sentence_dominance": row.get("dominance", np.nan),
        }
        if lex_row is not None:
            out.update(
                {
                    "lexicon_source": lex_row["lexicon_source"],
                    "lex_valence": lex_row["lex_valence"],
                    "lex_arousal": lex_row["lex_arousal"],
                    "lex_dominance": lex_row["lex_dominance"],
                }
            )
        else:
            out.update(
                {
                    "lexicon_source": "",
                    "lex_valence": np.nan,
                    "lex_arousal": np.nan,
                    "lex_dominance": np.nan,
                }
            )

        for feat_idx, feature_name in enumerate(FEATURE_NAMES):
            out[f"et2_{feature_name}"] = float(et2_features[word_idx, feat_idx])
            out[f"emotion_et_{feature_name}"] = float(emotion_features[word_idx, feat_idx])
            out[f"delta_{feature_name}_emotion_minus_et2"] = (
                out[f"emotion_et_{feature_name}"] - out[f"et2_{feature_name}"]
            )
        rows.append(out)
    return rows


def _summarize_word_groups(word_predictions: pd.DataFrame) -> pd.DataFrame:
    group_order = [
        "emotion_word",
        "non_emotion_word",
        "matched_other",
        "unmatched_alpha",
        "function_word",
        "non_alpha",
    ]
    rows = []
    for group_name in group_order:
        group = word_predictions[word_predictions["word_group"].eq(group_name)]
        if group.empty:
            continue
        rows.append(
            {
                "Word Group": group_name,
                "N Tokens": len(group),
                "N Unique Words": group["norm_word"].nunique(),
                "Mean Lexical Valence": group["lex_valence"].mean(),
                "Mean Lexical Arousal": group["lex_arousal"].mean(),
                "Generic ET2 TRT": group["et2_TRT"].mean(),
                "Emotion-Specific ET TRT": group["emotion_et_TRT"].mean(),
                "Delta TRT (Emotion ET - ET2)": group["delta_TRT_emotion_minus_et2"].mean(),
                "Generic ET2 FFD": group["et2_FFD"].mean(),
                "Emotion-Specific ET FFD": group["emotion_et_FFD"].mean(),
                "Delta FFD (Emotion ET - ET2)": group["delta_FFD_emotion_minus_et2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _summarize_word_groups_by_sentence_bin(word_predictions: pd.DataFrame, bin_column: str) -> pd.DataFrame:
    rows = []
    for (sentence_bin, word_group), group in word_predictions.groupby([bin_column, "word_group"], observed=False):
        if word_group not in {"emotion_word", "non_emotion_word", "matched_other", "unmatched_alpha"}:
            continue
        rows.append(
            {
                "Sentence Bin": str(sentence_bin),
                "Word Group": word_group,
                "N Tokens": len(group),
                "N Unique Words": group["norm_word"].nunique(),
                "Generic ET2 TRT": group["et2_TRT"].mean(),
                "Emotion-Specific ET TRT": group["emotion_et_TRT"].mean(),
                "Delta TRT (Emotion ET - ET2)": group["delta_TRT_emotion_minus_et2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _compute_word_correlations(word_predictions: pd.DataFrame) -> pd.DataFrame:
    usable = word_predictions[
        word_predictions["word_group"].isin(["emotion_word", "non_emotion_word", "matched_other"])
    ].copy()
    targets = ["lex_valence", "lex_arousal", "sentence_valence", "sentence_arousal"]
    predictors = ["et2_TRT", "emotion_et_TRT", "delta_TRT_emotion_minus_et2"]
    rows = []
    for target in targets:
        for predictor in predictors:
            subset = usable[[target, predictor]].dropna()
            if len(subset) < 3:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(subset[target], subset[predictor])
            rows.append(
                {
                    "Target": target,
                    "Predictor": predictor,
                    "Spearman": rho,
                    "P Value": p_value,
                    "N": len(subset),
                }
            )
    return pd.DataFrame(rows)


def _centered_markdown(table: pd.DataFrame) -> str:
    return table.to_markdown(index=False, colalign=["center"] * len(table.columns))


def _write_report(
    output_dir: Path,
    metadata: dict[str, object],
    word_group_summary: pd.DataFrame,
    valence_bin_summary: pd.DataFrame,
    arousal_bin_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = [
        "# Word-Level ET Predictor Comparison On EmoBank VAD",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Word Group Summary",
            "",
            _centered_markdown(word_group_summary.round(6)),
            "",
            "## Word Groups By Sentence Valence Bin",
            "",
            _centered_markdown(valence_bin_summary.round(6)),
            "",
            "## Word Groups By Sentence Arousal Bin",
            "",
            _centered_markdown(arousal_bin_summary.round(6)),
            "",
            "## Word-Level Spearman Correlations",
            "",
            _centered_markdown(correlations.round(6)),
            "",
        ]
    )
    (output_dir / "word_level_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-zip", type=Path, default=Path("/Users/wansookim/Downloads/Archive (1).zip"))
    parser.add_argument("--member", default="emobank.tsv")
    parser.add_argument("--lexicon-members", nargs="+", default=["nrc_vad.tsv", "warriner_et_al.tsv"])
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--min-alpha-words", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emotion-valence-abs-min", type=float, default=0.20)
    parser.add_argument("--emotion-arousal-min", type=float, default=0.65)
    parser.add_argument("--neutral-valence-abs-max", type=float, default=0.10)
    parser.add_argument("--neutral-arousal-abs-max", type=float, default=0.10)
    parser.add_argument("--et2-checkpoint", type=Path, default=Path("hf_checks/et_prediction_2/et_predictor2_seed123.safetensors"))
    parser.add_argument("--emotion-model-dir", type=Path, default=Path("hf_checks/emotion_et_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/et_predictor_vad_emobank_word_groups"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vad_data_all = _read_vad_tsv(args.vad_zip, args.member)
    vad_data = _filter_sentence_like_rows(vad_data_all, args.min_alpha_words)
    sampled = _sample_vad_rows(vad_data, args.sample_size, args.seed)
    sampled["sentence_valence_bin"] = pd.qcut(
        sampled["valence"], q=3, labels=["low", "mid", "high"], duplicates="drop"
    )
    sampled["sentence_arousal_bin"] = pd.qcut(
        sampled["arousal"], q=3, labels=["low", "mid", "high"], duplicates="drop"
    )

    lexicon = _load_lexicon(args.vad_zip, args.lexicon_members)
    lexicon_by_word = {row["norm_word"]: row for _, row in lexicon.iterrows()}
    et2_model, emotion_model = _load_predictors(args.et2_checkpoint, args.emotion_model_dir)

    all_word_rows = []
    for sampled_idx, row in sampled.reset_index(drop=True).iterrows():
        word_rows = _predict_word_rows(
            row=row,
            sentence_idx=sampled_idx,
            et2_model=et2_model,
            emotion_model=emotion_model,
            lexicon_by_word=lexicon_by_word,
            args=args,
        )
        for word_row in word_rows:
            word_row["sentence_valence_bin"] = row["sentence_valence_bin"]
            word_row["sentence_arousal_bin"] = row["sentence_arousal_bin"]
        all_word_rows.extend(word_rows)
        if (sampled_idx + 1) % 50 == 0:
            print(f"processed {sampled_idx + 1}/{len(sampled)} sentences, {len(all_word_rows)} words")

    word_predictions = pd.DataFrame(all_word_rows)
    word_group_summary = _summarize_word_groups(word_predictions)
    valence_bin_summary = _summarize_word_groups_by_sentence_bin(
        word_predictions, "sentence_valence_bin"
    )
    arousal_bin_summary = _summarize_word_groups_by_sentence_bin(
        word_predictions, "sentence_arousal_bin"
    )
    correlations = _compute_word_correlations(word_predictions)

    word_predictions.to_csv(args.output_dir / "word_predictions.csv", index=False)
    word_group_summary.to_csv(args.output_dir / "word_group_summary.csv", index=False)
    valence_bin_summary.to_csv(args.output_dir / "word_groups_by_sentence_valence.csv", index=False)
    arousal_bin_summary.to_csv(args.output_dir / "word_groups_by_sentence_arousal.csv", index=False)
    correlations.to_csv(args.output_dir / "word_level_correlations.csv", index=False)

    metadata = {
        "vad_zip": str(args.vad_zip),
        "member": args.member,
        "lexicon_members": args.lexicon_members,
        "rows_before_filter": len(vad_data_all),
        "rows_after_filter": len(vad_data),
        "sample_size_requested": args.sample_size,
        "sample_size_used": len(sampled),
        "min_alpha_words": args.min_alpha_words,
        "seed": args.seed,
        "emotion_word_rule": f"|valence - 0.5| >= {args.emotion_valence_abs_min} or arousal >= {args.emotion_arousal_min}",
        "non_emotion_word_rule": f"|valence - 0.5| <= {args.neutral_valence_abs_max} and |arousal - 0.5| <= {args.neutral_arousal_abs_max}",
        "et2_checkpoint": str(args.et2_checkpoint),
        "emotion_model_dir": str(args.emotion_model_dir),
        "n_word_rows": len(word_predictions),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(
        args.output_dir,
        metadata,
        word_group_summary,
        valence_bin_summary,
        arousal_bin_summary,
        correlations,
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
