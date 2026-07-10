import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unicodedata
import uuid
import zipfile

import pandas as pd

from va_gaze.data.downloads import (
    GDriveArtifact,
    download_gdrive_artifact,
    sha256_file,
    validate_zip_artifact,
)


DEFAULT_GDRIVE_ZIP_URL = None
DEFAULT_GDRIVE_ZIP_NAME = "english_va_bundle.zip"
DEFAULT_EXTERNAL_DIR = "data/external_english"
NORMALIZATION_CHOICES = ("observed", "source-scale")
EXTRACTION_MANIFEST_NAME = ".va_gaze_extraction.json"
EXTRACTION_ROOT_NAME = ".va_gaze_extracted"
BUILD_MANIFEST_NAME = "english_dataset_manifest.json"
EXTRACTION_MANIFEST_VERSION = 1
BUILD_MANIFEST_VERSION = 1

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

SOURCE_SCALE_BOUNDS = {
    "fb": {"valence": (1.0, 9.0), "arousal": (1.0, 9.0)},
    "facebook_va": {"valence": (1.0, 9.0), "arousal": (1.0, 9.0)},
}


def _clean_text_column(series):
    cleaned = series.astype(str)
    cleaned = cleaned.str.replace(r"[\r\n\t]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.strip()
    return cleaned


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


def _normalize_with_bounds(series, lower, upper):
    series = pd.to_numeric(series, errors="coerce")
    if upper <= lower:
        raise ValueError("source-scale normalization requires upper > lower.")
    return ((series - lower) / (upper - lower)).clip(0.0, 1.0)


def _source_scale_bounds(dataset_name):
    if dataset_name is None:
        return None
    normalized_name = str(dataset_name).lower().replace("-", "_")
    return SOURCE_SCALE_BOUNDS.get(normalized_name)


def _post_process_dataset(df, dataset_name=None, normalization="observed"):
    out = df.copy()
    out["text"] = _clean_text_column(out["text"])
    out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
    out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
    out = out.dropna(subset=["text", "valence", "arousal"])
    out = out[out["text"] != ""]

    source_bounds = _source_scale_bounds(dataset_name)
    if normalization == "source-scale" and source_bounds is not None:
        val_lower, val_upper = source_bounds["valence"]
        aro_lower, aro_upper = source_bounds["arousal"]
        out["valence"] = _normalize_with_bounds(out["valence"], val_lower, val_upper)
        out["arousal"] = _normalize_with_bounds(out["arousal"], aro_lower, aro_upper)
    else:
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


def _download_gdrive_zip(
    gdrive_url,
    zip_path,
    force=False,
    file_id=None,
    sha256=None,
    offline=False,
    downloader=None,
):
    artifact = GDriveArtifact(
        filename=os.path.basename(zip_path),
        file_id=file_id,
        url=gdrive_url,
        sha256=sha256,
    )
    return str(
        download_gdrive_artifact(
            artifact=artifact,
            destination_dir=os.path.dirname(zip_path) or ".",
            force=force,
            offline=offline,
            downloader=downloader,
        )
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_basename_key(name):
    return unicodedata.normalize("NFC", name).casefold()


def _active_extracted_paths(external_dir, expected_archive_sha256=None):
    external_dir = Path(external_dir)
    manifest_path = external_dir / EXTRACTION_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EXTRACTION_MANIFEST_VERSION:
        raise ValueError(f"Unsupported extraction manifest: {manifest_path}")
    if (
        expected_archive_sha256 is not None
        and manifest.get("archive_sha256") != expected_archive_sha256
    ):
        return None

    version = manifest.get("version")
    if not isinstance(version, str) or Path(version).name != version:
        raise ValueError(f"Unsafe extraction version in manifest: {manifest_path}")
    data_dir = external_dir / EXTRACTION_ROOT_NAME / version
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Extraction manifest has no files: {manifest_path}")

    paths = []
    destination_names = set()
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not name.casefold().endswith(".tsv")
        ):
            raise ValueError(f"Unsafe extracted filename in manifest: {manifest_path}")
        destination_key = _manifest_basename_key(name)
        if destination_key in destination_names:
            raise ValueError(f"Duplicate extracted filename in manifest: {name}")
        destination_names.add(destination_key)
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"Extracted TSV is missing: {path}")
        if path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Extracted TSV size mismatch: {path}")
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Extracted TSV hash mismatch: {path}")
        paths.append(path)
    return paths


def _stream_zip_member(archive, member, destination, max_bytes):
    digest = hashlib.sha256()
    total = 0
    with archive.open(member, "r") as source, open(destination, "xb") as output_file:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes or total > member.file_size:
                raise ValueError(f"Zip member exceeded its validated size: {member.filename}")
            output_file.write(chunk)
            digest.update(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    if total != member.file_size:
        raise ValueError(
            f"Zip member size changed during extraction: {member.filename} "
            f"({total} != {member.file_size})"
        )
    return {
        "name": destination.name,
        "sha256": digest.hexdigest(),
        "size_bytes": total,
    }


def _cleanup_inactive_extractions(managed_root, active_version):
    for candidate in managed_root.iterdir():
        if candidate.name == active_version:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def _extract_zip_tsv(zip_path, external_dir, force=False):
    zip_path = Path(zip_path)
    external_dir = Path(external_dir)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    external_dir.mkdir(parents=True, exist_ok=True)
    artifact = GDriveArtifact(filename=zip_path.name)
    valid_members = set(validate_zip_artifact(zip_path, artifact))
    archive_sha256 = sha256_file(zip_path)

    if not force:
        try:
            cached_paths = _active_extracted_paths(
                external_dir,
                expected_archive_sha256=archive_sha256,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[zip] Extracted cache is invalid; rebuilding: {exc}")
        else:
            if cached_paths:
                print(f"[zip] Valid extracted cache: {len(cached_paths)} TSV files")
                return [str(path) for path in cached_paths]

    managed_root = external_dir / EXTRACTION_ROOT_NAME
    managed_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=str(managed_root))
    )
    version = f"{archive_sha256[:16]}-{uuid.uuid4().hex}"
    version_dir = managed_root / version
    activated = False
    entries = []
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                if member.filename not in valid_members:
                    continue
                base_name = PurePosixPath(member.filename).name
                destination = staging_dir / base_name
                entries.append(
                    _stream_zip_member(
                        archive,
                        member,
                        destination,
                        max_bytes=artifact.max_member_bytes,
                    )
                )
        entries.sort(key=lambda entry: _manifest_basename_key(entry["name"]))
        os.replace(staging_dir, version_dir)
        manifest = {
            "schema_version": EXTRACTION_MANIFEST_VERSION,
            "archive_filename": zip_path.name,
            "archive_sha256": archive_sha256,
            "version": version,
            "files": entries,
        }
        _write_json_atomic(external_dir / EXTRACTION_MANIFEST_NAME, manifest)
        activated = True
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if not activated and version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)

    _cleanup_inactive_extractions(managed_root, version)
    extracted = [str(version_dir / entry["name"]) for entry in entries]
    print(f"[zip] Atomically activated {len(extracted)} TSV files in {version_dir}")
    return extracted


def _infer_dataset_name_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower().replace("-", "_")
    return EXTERNAL_SOURCE_NAME_MAP.get(stem, stem)


def _external_source_paths(external_dir):
    external_dir = Path(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = external_dir / EXTRACTION_MANIFEST_NAME
    if manifest_path.is_file():
        paths = _active_extracted_paths(external_dir)
        if not paths:
            raise RuntimeError(f"Extraction manifest has no active TSV files: {manifest_path}")
        return sorted(paths, key=lambda path: _manifest_basename_key(path.name))
    return sorted(
        (
            path
            for path in external_dir.iterdir()
            if path.is_file() and path.name.casefold().endswith(".tsv")
        ),
        key=lambda path: _manifest_basename_key(path.name),
    )


def _source_fingerprint(paths):
    records = []
    destination_names = set()
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: _manifest_basename_key(Path(item).name)):
        path = Path(path)
        destination_key = _manifest_basename_key(path.name)
        if destination_key in destination_names:
            raise ValueError(f"Duplicate TSV source basename: {path.name}")
        destination_names.add(destination_key)
        file_sha256 = sha256_file(path)
        record = {
            "name": path.name,
            "sha256": file_sha256,
            "size_bytes": path.stat().st_size,
        }
        records.append(record)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {"source_hash": digest.hexdigest(), "source_files": records}


def _load_external_sources(external_dir, normalization="observed", source_paths=None):
    if source_paths is None:
        source_paths = _external_source_paths(external_dir)

    if not source_paths:
        print(f"[external] No TSV files found in: {external_dir}")
        return []

    loaded = []
    for path in source_paths:
        path = Path(path)
        file_name = path.name
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue

        required = {"text", "valence", "arousal"}
        if not required.issubset(set(df.columns)):
            print(f"[warn] Skip {path}: required columns are text, valence, arousal.")
            continue

        dataset_name = _infer_dataset_name_from_path(path)
        out = pd.DataFrame(
            {
                "text": df["text"],
                "valence": df["valence"],
                "arousal": df["arousal"],
                "dataset_of_origin": dataset_name,
            }
        )
        out = _post_process_dataset(
            out,
            dataset_name=dataset_name,
            normalization=normalization,
        )
        if len(out) == 0:
            print(f"[warn] Skip {path}: no valid rows after processing.")
            continue

        loaded.append(out)
        print(f"[external] Loaded {file_name} -> {dataset_name}: {len(out)} rows")

    return loaded


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


def _dataset_cache_is_valid(output_paths, manifest_path, build_config):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != BUILD_MANIFEST_VERSION:
            return False
        if manifest.get("build") != build_config:
            return False
        output_metadata = manifest.get("outputs") or {}
        for output_path in output_paths:
            output_path = Path(output_path)
            metadata = output_metadata.get(output_path.name)
            if not output_path.is_file() or not isinstance(metadata, dict):
                return False
            if output_path.stat().st_size != metadata.get("size_bytes"):
                return False
            if sha256_file(output_path) != metadata.get("sha256"):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _write_dataset_outputs_atomically(outputs, manifest_path, build_config):
    temporary_paths = {}
    output_metadata = {}
    try:
        for output_path, dataframe in outputs.items():
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(
                f".{output_path.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary_paths[output_path] = temporary
            _write_tsv(dataframe, temporary)
            with open(temporary, "rb+") as output_file:
                os.fsync(output_file.fileno())
            output_metadata[output_path.name] = {
                "rows": int(len(dataframe)),
                "sha256": sha256_file(temporary),
                "size_bytes": temporary.stat().st_size,
            }

        for output_path, temporary in temporary_paths.items():
            os.replace(temporary, output_path)

        manifest = {
            "schema_version": BUILD_MANIFEST_VERSION,
            "build": build_config,
            "outputs": output_metadata,
        }
        _write_json_atomic(manifest_path, manifest)
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()


def build_english_dataset(
    output_dir,
    seed,
    force=False,
    external_dir=DEFAULT_EXTERNAL_DIR,
    gdrive_zip_url=DEFAULT_GDRIVE_ZIP_URL,
    gdrive_zip_name=DEFAULT_GDRIVE_ZIP_NAME,
    gdrive_file_id=None,
    gdrive_sha256=None,
    skip_gdrive_download=False,
    offline=False,
    normalization="observed",
):
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(
            f"normalization must be one of {NORMALIZATION_CHOICES}, got {normalization!r}."
        )

    output_dir = Path(output_dir)
    external_dir = Path(external_dir)
    if Path(gdrive_zip_name).name != gdrive_zip_name:
        raise ValueError("gdrive_zip_name must be a basename, not a path.")
    fold1_path = output_dir / "full_dataset_fold1.csv"
    fold2_path = output_dir / "full_dataset_fold2.csv"
    merged_path = output_dir / "full_dataset_english_all.csv"
    manifest_path = output_dir / BUILD_MANIFEST_NAME

    external_dir.mkdir(parents=True, exist_ok=True)
    zip_path = external_dir / gdrive_zip_name
    has_source = bool(gdrive_file_id or gdrive_zip_url)
    if skip_gdrive_download:
        if zip_path.is_file():
            _extract_zip_tsv(zip_path, external_dir, force=force)
    else:
        if not has_source and not offline:
            raise ValueError(
                "No dataset download source is configured. Pass --gdrive-file-id or "
                "--gdrive-zip-url, or use --skip-gdrive-download with local TSV files."
            )
        _download_gdrive_zip(
            gdrive_zip_url,
            zip_path,
            force=force and not offline,
            file_id=gdrive_file_id,
            sha256=gdrive_sha256,
            offline=offline,
        )
        _extract_zip_tsv(zip_path, external_dir, force=force)

    source_paths = _external_source_paths(external_dir)
    if not source_paths:
        raise RuntimeError(
            "No valid dataset TSV files available. Put TSV files in "
            f"{external_dir} or pass --gdrive-file-id/--gdrive-zip-url."
        )
    source_fingerprint = _source_fingerprint(source_paths)
    build_config = {
        "seed": int(seed),
        "normalization": normalization,
        **source_fingerprint,
    }
    output_paths = (fold1_path, fold2_path, merged_path)
    if not force and _dataset_cache_is_valid(
        output_paths,
        manifest_path,
        build_config,
    ):
        print("English dataset cache matches its sources and build configuration.")
        print(f"Skipping rebuild: {fold1_path}, {fold2_path}")
        return

    dataframes = _load_external_sources(
        external_dir,
        normalization=normalization,
        source_paths=source_paths,
    )
    if not dataframes:
        raise RuntimeError(
            "No valid dataset TSV files available. "
            f"Put TSV files in {external_dir} or pass a valid Google Drive source."
        )

    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged[["text", "dataset_of_origin", "valence", "arousal"]]
    merged = merged.drop_duplicates(subset=["text", "dataset_of_origin"])

    fold1, fold2 = _split_in_two_folds(merged, seed=seed)
    merged_output = pd.concat([fold1, fold2], ignore_index=True)

    outputs = {
        fold1_path: fold1,
        fold2_path: fold2,
        merged_path: merged_output,
    }
    _write_dataset_outputs_atomically(outputs, manifest_path, build_config)

    counts = merged.groupby("dataset_of_origin").size().sort_values(ascending=False)
    print("English dataset prepared.")
    print(f"Normalization: {normalization}")
    print(f"Total samples: {len(merged)}")
    print("Samples per source:")
    for name, value in counts.items():
        print(f"- {name}: {value}")
    print(f"Saved: {fold1_path}")
    print(f"Saved: {fold2_path}")
    print(f"Build manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download (Google Drive) and build English-only VA folds from TSV files."
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
        default=DEFAULT_EXTERNAL_DIR,
        help="Folder containing dataset TSV files (text,valence,arousal).",
    )
    parser.add_argument(
        "--gdrive-zip-url",
        default=DEFAULT_GDRIVE_ZIP_URL,
        help="Explicit Google Drive share URL for a dataset zip you may access.",
    )
    parser.add_argument(
        "--gdrive-zip-name",
        default=DEFAULT_GDRIVE_ZIP_NAME,
        help="Filename to store downloaded zip under --external-dir.",
    )
    parser.add_argument(
        "--gdrive-file-id",
        default=None,
        help="Explicit Google Drive file id; alternatively pass --gdrive-zip-url.",
    )
    parser.add_argument(
        "--gdrive-sha256",
        default=None,
        help="Optional expected SHA256 for the downloaded zip.",
    )
    parser.add_argument(
        "--skip-gdrive-download",
        action="store_true",
        help="Skip gdown download and use already existing TSV files in --external-dir.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require a valid cached gdown archive and never access the network.",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default="observed",
        help=(
            "observed keeps the previous per-file observed min/max behavior. "
            "source-scale uses known source bounds when available and falls back to observed otherwise."
        ),
    )
    args = parser.parse_args()

    build_english_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        force=args.force,
        external_dir=args.external_dir,
        gdrive_zip_url=args.gdrive_zip_url,
        gdrive_zip_name=args.gdrive_zip_name,
        gdrive_file_id=args.gdrive_file_id,
        gdrive_sha256=args.gdrive_sha256,
        skip_gdrive_download=args.skip_gdrive_download,
        offline=args.offline,
        normalization=args.normalization,
    )


if __name__ == "__main__":
    main()
