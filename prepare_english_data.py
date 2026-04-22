import argparse
import csv
import io
import os
import zipfile

import pandas as pd
import requests


ENGLISH_SOURCES = {
    "emobank": {
        "url": "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv",
        "dataset_name": "Emobank",
    },
    "facebook": {
        "url": "https://github.com/wwbp/additional_data_sets/raw/master/valence_arousal/dataset-fb-valence-arousal-anon.csv",
        "dataset_name": "fb",
    },
    "nrc_vad": {
        "url": "http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip",
        "dataset_name": "nrc-vad",
    },
    "glasgow": {
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv",
        "dataset_name": "GlasgowNorms",
    },
    "warriner": {
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip",
        "dataset_name": "word ratings ENG",
    },
}

EXTERNAL_SOURCE_NAME_MAP = {
    "iemocap": "IEMOCAP sentences",
    "emotales": "EmoTales sentences",
    "scott_et_al": "GlasgowNorms",
    "nrc_vad": "nrc-vad",
    "warriner_et_al": "word ratings ENG",
    "facebook_va": "fb",
    "fb": "fb",
    "emobank": "Emobank",
    "anet": "ANET sentences",
}


def _download_bytes(url, timeout=120):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def _normalize_minmax(series):
    series = pd.to_numeric(series, errors="coerce")
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return series
    if max_value == min_value:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    normalized = (series - min_value) / (max_value - min_value)
    return normalized.clip(0.0, 1.0)


def _clean_text_column(series):
    cleaned = series.astype(str)
    cleaned = cleaned.str.replace(r"[\r\n\t]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.strip()
    return cleaned


def _prepare_emobank(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    out = pd.DataFrame(
        {
            "text": df["text"],
            "valence": df["V"],
            "arousal": df["A"],
            "dataset_of_origin": ENGLISH_SOURCES["emobank"]["dataset_name"],
        }
    )
    return out


def _prepare_facebook(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    out = pd.DataFrame(
        {
            "text": df["Anonymized Message"],
            "valence": (df["Valence1"] + df["Valence2"]) / 2.0,
            "arousal": (df["Arousal1"] + df["Arousal2"]) / 2.0,
            "dataset_of_origin": ENGLISH_SOURCES["facebook"]["dataset_name"],
        }
    )
    return out


def _prepare_nrc_vad(raw_bytes):
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        with archive.open("NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt") as file_obj:
            df = pd.read_csv(
                file_obj,
                sep="\t",
                header=None,
                names=["text", "valence", "arousal", "dominance"],
            )
    out = pd.DataFrame(
        {
            "text": df["text"],
            "valence": df["valence"],
            "arousal": df["arousal"],
            "dataset_of_origin": ENGLISH_SOURCES["nrc_vad"]["dataset_name"],
        }
    )
    return out


def _prepare_glasgow(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    out = pd.DataFrame(
        {
            "text": df["Words"],
            "valence": df["VAL"],
            "arousal": df["AROU"],
            "dataset_of_origin": ENGLISH_SOURCES["glasgow"]["dataset_name"],
        }
    )
    return out


def _prepare_warriner(raw_bytes):
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        with archive.open("BRM-emot-submit.csv") as file_obj:
            df = pd.read_csv(file_obj)
    out = pd.DataFrame(
        {
            "text": df["Word"],
            "valence": df["V.Mean.Sum"],
            "arousal": df["A.Mean.Sum"],
            "dataset_of_origin": ENGLISH_SOURCES["warriner"]["dataset_name"],
        }
    )
    return out


def _post_process_dataset(df):
    out = df.copy()
    out["text"] = _clean_text_column(out["text"])
    out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
    out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
    out = out.dropna(subset=["text", "valence", "arousal"])
    out = out[out["text"] != ""]
    val_in_unit = out["valence"].between(0.0, 1.0, inclusive="both").all()
    aro_in_unit = out["arousal"].between(0.0, 1.0, inclusive="both").all()
    if val_in_unit and aro_in_unit:
        out["valence"] = out["valence"].clip(0.0, 1.0)
        out["arousal"] = out["arousal"].clip(0.0, 1.0)
    else:
        out["valence"] = _normalize_minmax(out["valence"])
        out["arousal"] = _normalize_minmax(out["arousal"])
    out = out.dropna(subset=["valence", "arousal"])
    out = out.drop_duplicates(subset=["text", "dataset_of_origin"])
    return out


def _split_in_two_folds(df, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled.insert(0, "index", shuffled.index.astype(int))
    midpoint = len(shuffled) // 2
    fold1 = shuffled.iloc[:midpoint].copy()
    fold2 = shuffled.iloc[midpoint:].copy()
    return fold1, fold2


def _write_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def _infer_external_dataset_name(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower().replace("-", "_")
    return EXTERNAL_SOURCE_NAME_MAP.get(stem, stem)


def _load_external_sources(external_dir):
    if not external_dir:
        return []
    os.makedirs(external_dir, exist_ok=True)

    files = sorted(
        file_name
        for file_name in os.listdir(external_dir)
        if file_name.lower().endswith(".tsv")
    )
    if not files:
        print(f"[external] No TSV files found in: {external_dir}")
        return []

    loaded = []
    for file_name in files:
        path = os.path.join(external_dir, file_name)
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue

        required = {"text", "valence", "arousal"}
        if not required.issubset(set(df.columns)):
            print(f"[warn] Skip {path}: required columns are text, valence, arousal.")
            continue

        dataset_name = _infer_external_dataset_name(path)
        out = pd.DataFrame(
            {
                "text": df["text"],
                "valence": df["valence"],
                "arousal": df["arousal"],
                "dataset_of_origin": dataset_name,
            }
        )
        out = _post_process_dataset(out)
        if len(out) == 0:
            print(f"[warn] Skip {path}: no valid rows after processing.")
            continue

        loaded.append(out)
        print(f"[external] Loaded {file_name} -> {dataset_name}: {len(out)} rows")
    return loaded


def build_english_dataset(output_dir, seed, force=False, external_dir=None, external_only=False):
    fold1_path = os.path.join(output_dir, "full_dataset_fold1.csv")
    fold2_path = os.path.join(output_dir, "full_dataset_fold2.csv")
    merged_path = os.path.join(output_dir, "full_dataset_english_all.csv")

    if (
        not force
        and os.path.isfile(fold1_path)
        and os.path.isfile(fold2_path)
        and os.path.isfile(merged_path)
    ):
        print("English dataset already exists. Skipping download/build.")
        print(f"Use --force to rebuild: {fold1_path}, {fold2_path}")
        return

    dataframes = []
    if not external_only:
        emobank = _post_process_dataset(
            _prepare_emobank(_download_bytes(ENGLISH_SOURCES["emobank"]["url"]))
        )
        facebook = _post_process_dataset(
            _prepare_facebook(_download_bytes(ENGLISH_SOURCES["facebook"]["url"]))
        )
        nrc_vad = _post_process_dataset(
            _prepare_nrc_vad(_download_bytes(ENGLISH_SOURCES["nrc_vad"]["url"]))
        )
        glasgow = _post_process_dataset(
            _prepare_glasgow(_download_bytes(ENGLISH_SOURCES["glasgow"]["url"]))
        )
        warriner = _post_process_dataset(
            _prepare_warriner(_download_bytes(ENGLISH_SOURCES["warriner"]["url"]))
        )
        dataframes.extend([emobank, facebook, nrc_vad, glasgow, warriner])

    dataframes.extend(_load_external_sources(external_dir))
    if not dataframes:
        raise RuntimeError(
            "No English datasets available. "
            f"Add TSV files to {external_dir} or disable --external-only."
        )

    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged[["text", "dataset_of_origin", "valence", "arousal"]]
    merged = merged.drop_duplicates(subset=["text", "dataset_of_origin"])

    fold1, fold2 = _split_in_two_folds(merged, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    _write_tsv(fold1, fold1_path)
    _write_tsv(fold2, fold2_path)
    _write_tsv(
        pd.concat([fold1, fold2], ignore_index=True),
        merged_path,
    )

    counts = merged.groupby("dataset_of_origin").size().sort_values(ascending=False)
    print("English dataset prepared.")
    print(f"Total samples: {len(merged)}")
    print("Samples per source:")
    for name, value in counts.items():
        print(f"- {name}: {value}")
    print(f"Saved: {fold1_path}")
    print(f"Saved: {fold2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and build English-only VA folds from README dataset links."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write full_dataset_fold1.csv and full_dataset_fold2.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before splitting into fold1/fold2",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild files even if full_dataset_fold1/2 and full_dataset_english_all already exist.",
    )
    parser.add_argument(
        "--external-dir",
        default="external_english",
        help="Optional folder with extra TSV files (text,valence,arousal).",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Build folds from --external-dir TSV files only (skip web downloads).",
    )
    args = parser.parse_args()
    build_english_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        force=args.force,
        external_dir=args.external_dir,
        external_only=args.external_only,
    )
