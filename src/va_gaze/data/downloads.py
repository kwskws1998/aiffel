from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import unicodedata
import uuid
import zipfile


CACHE_METADATA_VERSION = 1
DEFAULT_MAX_ARCHIVE_MEMBERS = 1024
DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 200.0


@dataclass(frozen=True)
class GDriveArtifact:
    filename: str
    file_id: str = None
    url: str = None
    sha256: str = None
    min_bytes: int = 1
    required_suffix: str = ".tsv"
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO

    def resolved_file_id(self):
        explicit_id = self.file_id.strip() if self.file_id else None
        parsed_id = _file_id_from_url(self.url)
        if explicit_id and self.url:
            if parsed_id != explicit_id:
                raise ValueError(
                    "When both file_id and url are provided, the URL must contain the same "
                    "Google Drive file id."
                )
        return explicit_id or parsed_id

    def source_identity(self):
        file_id = self.resolved_file_id()
        if file_id:
            return {"kind": "gdrive-id", "value": file_id}
        if self.url and self.url.strip():
            return {"kind": "url", "value": self.url.strip()}
        return None

    def request_identity(self):
        return {
            "source": self.source_identity(),
            "expected_sha256": self.sha256.lower() if self.sha256 else None,
        }


def _file_id_from_url(url):
    if not url:
        return None
    patterns = (
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(name):
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _destination_key(name):
    return unicodedata.normalize("NFC", name).casefold()


def _check_archive_limits(infos, artifact):
    files = [member for member in infos if not member.is_dir()]
    if len(files) > int(artifact.max_members):
        raise ValueError(
            f"Archive contains {len(files)} files, exceeding the limit of "
            f"{artifact.max_members}."
        )

    total_uncompressed = 0
    for member in files:
        if not _safe_archive_name(member.filename):
            raise ValueError(f"Unsafe zip member path: {member.filename}")
        if member.file_size > int(artifact.max_member_bytes):
            raise ValueError(
                f"Zip member exceeds the uncompressed size limit: {member.filename} "
                f"({member.file_size} > {artifact.max_member_bytes})"
            )
        total_uncompressed += member.file_size
        if total_uncompressed > int(artifact.max_total_uncompressed_bytes):
            raise ValueError(
                "Archive exceeds the total uncompressed size limit: "
                f"{total_uncompressed} > {artifact.max_total_uncompressed_bytes}"
            )
        if member.file_size:
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > float(artifact.max_compression_ratio):
                raise ValueError(
                    f"Zip member exceeds the compression-ratio limit: {member.filename} "
                    f"({ratio:.1f} > {artifact.max_compression_ratio})"
                )


def validate_zip_artifact(path, artifact):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")
    if path.stat().st_size < int(artifact.min_bytes):
        raise ValueError(f"Archive is smaller than {artifact.min_bytes} bytes: {path}")
    if artifact.sha256:
        actual = sha256_file(path)
        if actual.lower() != artifact.sha256.lower():
            raise ValueError(
                f"SHA256 mismatch for {path}: expected {artifact.sha256}, got {actual}"
            )

    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        _check_archive_limits(infos, artifact)

        members = []
        destination_names = set()
        required_suffix = artifact.required_suffix.casefold()
        for member in infos:
            if member.is_dir():
                continue
            base_name = PurePosixPath(member.filename).name
            if member.filename.startswith("__MACOSX/") or base_name.startswith("._"):
                continue
            if not base_name.casefold().endswith(required_suffix):
                continue
            destination_key = _destination_key(base_name)
            if destination_key in destination_names:
                raise ValueError(
                    "Duplicate archive basename would overwrite data on a "
                    f"case-insensitive filesystem: {base_name}"
                )
            destination_names.add(destination_key)
            members.append(member.filename)

        if not members:
            raise ValueError(
                f"Archive contains no {artifact.required_suffix} data files: {path}"
            )

        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt zip member: {bad_member}")
    return members


def _cache_metadata_path(final_path):
    return final_path.with_name(final_path.name + ".metadata.json")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_atomic(path, payload):
    path = Path(path)
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


def _build_cache_metadata(path, artifact):
    path = Path(path)
    return {
        "schema_version": CACHE_METADATA_VERSION,
        "request": artifact.request_identity(),
        "archive": {
            "filename": artifact.filename,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
    }


def _validate_cached_artifact(final_path, artifact, allow_unbound_request=False):
    validate_zip_artifact(final_path, artifact)
    metadata_path = _cache_metadata_path(final_path)
    if not metadata_path.is_file():
        raise ValueError(f"Cache metadata sidecar is missing: {metadata_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("schema_version") != CACHE_METADATA_VERSION:
        raise ValueError(f"Unsupported cache metadata version: {metadata_path}")

    archive_metadata = metadata.get("archive") or {}
    if archive_metadata.get("filename") != final_path.name:
        raise ValueError("Cached archive filename does not match its metadata sidecar.")
    actual_size = final_path.stat().st_size
    if archive_metadata.get("size_bytes") != actual_size:
        raise ValueError("Cached archive size does not match its metadata sidecar.")
    actual_sha256 = sha256_file(final_path)
    if archive_metadata.get("sha256") != actual_sha256:
        raise ValueError("Cached archive hash does not match its metadata sidecar.")

    requested_identity = artifact.request_identity()
    if requested_identity["source"] is None and allow_unbound_request:
        expected_sha256 = requested_identity["expected_sha256"]
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError("Cached archive does not match the requested SHA256.")
    elif metadata.get("request") != requested_identity:
        raise ValueError("Cached archive was created for a different source or checksum.")
    return metadata


def download_gdrive_artifact(
    artifact,
    destination_dir,
    force=False,
    offline=False,
    downloader=None,
):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if Path(artifact.filename).name != artifact.filename:
        raise ValueError("GDriveArtifact.filename must be a basename, not a path.")
    final_path = destination_dir / artifact.filename

    source_identity = artifact.source_identity()
    if source_identity is None and not offline:
        raise ValueError(
            "A Google Drive file id or URL is required for an online download."
        )
    if force and offline:
        raise ValueError("force=True cannot be combined with offline=True.")

    if final_path.is_file() and not force:
        try:
            _validate_cached_artifact(
                final_path,
                artifact,
                allow_unbound_request=offline and source_identity is None,
            )
            print(f"[gdown] Valid source-bound cached archive: {final_path}")
            return final_path
        except Exception as exc:
            if offline:
                raise RuntimeError(
                    f"Cached archive is invalid in offline mode: {final_path}: {exc}"
                ) from exc
            print(f"[gdown] Cached archive is invalid; downloading a replacement: {exc}")

    if offline:
        raise FileNotFoundError(
            f"No valid source-bound cached archive available in offline mode: {final_path}"
        )

    if downloader is None:
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "gdown is required for Google Drive data downloads. "
                "Install it with: pip install gdown"
            ) from exc
        downloader = gdown.download

    partial_path = final_path.with_name(final_path.name + ".part")
    if partial_path.exists():
        partial_path.unlink()

    kwargs = {
        "output": str(partial_path),
        "quiet": False,
        "resume": True,
    }
    file_id = artifact.resolved_file_id()
    if file_id:
        kwargs["id"] = file_id
    else:
        kwargs["url"] = artifact.url.strip()

    try:
        downloaded = downloader(**kwargs)
        if not downloaded or not partial_path.is_file():
            raise RuntimeError("gdown did not create the requested archive.")
        validate_zip_artifact(partial_path, artifact)
        metadata = _build_cache_metadata(partial_path, artifact)
        os.replace(partial_path, final_path)
        _write_json_atomic(_cache_metadata_path(final_path), metadata)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    print(f"[gdown] Downloaded, validated, and source-bound: {final_path}")
    return final_path
