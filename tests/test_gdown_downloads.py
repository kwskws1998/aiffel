import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import pandas as pd

from va_gaze.data.downloads import (
    GDriveArtifact,
    download_gdrive_artifact,
    validate_zip_artifact,
)
from va_gaze.data.prepare_english_data import (
    BUILD_MANIFEST_NAME,
    _extract_zip_tsv,
    _load_external_sources,
    _write_dataset_outputs_atomically,
    build_english_dataset,
)


class RecordingDownloader:
    def __init__(self, source):
        self.source = Path(source)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        shutil.copyfile(self.source, kwargs["output"])
        return kwargs["output"]


def write_zip(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


class GdownDownloadTest(unittest.TestCase):
    def test_online_download_requires_explicit_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "file id or URL"):
                download_gdrive_artifact(
                    GDriveArtifact(filename="bundle.zip"),
                    temp_dir,
                )

    def test_download_cache_offline_and_extract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.zip"
            write_zip(
                source,
                {
                    "public/sample.tsv": (
                        "text\tvalence\tarousal\n"
                        "calm morning\t0.7\t0.2\n"
                        "terrible storm\t0.1\t0.9\n"
                    )
                },
            )
            downloader = RecordingDownloader(source)
            artifact = GDriveArtifact(filename="bundle.zip", file_id="drive-file-id")
            destination = root / "cache"
            downloaded = download_gdrive_artifact(
                artifact,
                destination,
                downloader=downloader,
            )
            self.assertEqual(len(downloader.calls), 1)
            self.assertEqual(downloader.calls[0]["id"], "drive-file-id")
            self.assertNotIn("fuzzy", downloader.calls[0])
            self.assertTrue((destination / "bundle.zip.metadata.json").is_file())

            cached = download_gdrive_artifact(
                artifact,
                destination,
                offline=True,
                downloader=lambda **kwargs: self.fail("offline cache invoked downloader"),
            )
            self.assertEqual(downloaded, cached)

            unbound_offline_cache = download_gdrive_artifact(
                GDriveArtifact(filename="bundle.zip"),
                destination,
                offline=True,
                downloader=lambda **kwargs: self.fail("offline cache invoked downloader"),
            )
            self.assertEqual(downloaded, unbound_offline_cache)

            extracted_dir = root / "extracted"
            extracted = _extract_zip_tsv(str(cached), str(extracted_dir))
            self.assertEqual([Path(path).name for path in extracted], ["sample.tsv"])
            frames = _load_external_sources(str(extracted_dir))
            self.assertEqual(sum(len(frame) for frame in frames), 2)

    def test_failed_replacement_preserves_valid_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "cache"
            source = root / "source.zip"
            write_zip(source, {"sample.tsv": "text\tvalence\tarousal\na\t0.1\t0.2\n"})
            artifact = GDriveArtifact(filename="bundle.zip", file_id="id")
            download_gdrive_artifact(
                artifact,
                destination,
                downloader=RecordingDownloader(source),
            )
            final_path = destination / "bundle.zip"
            original = final_path.read_bytes()
            metadata_path = destination / "bundle.zip.metadata.json"
            original_metadata = metadata_path.read_bytes()

            def corrupt_downloader(**kwargs):
                Path(kwargs["output"]).write_bytes(b"not a zip")
                return kwargs["output"]

            with self.assertRaises(zipfile.BadZipFile):
                download_gdrive_artifact(
                    artifact,
                    destination,
                    force=True,
                    downloader=corrupt_downloader,
                )
            self.assertEqual(final_path.read_bytes(), original)
            self.assertEqual(metadata_path.read_bytes(), original_metadata)
            self.assertEqual(
                download_gdrive_artifact(artifact, destination, offline=True),
                final_path,
            )

    def test_archive_rejects_unsafe_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unsafe = root / "unsafe.zip"
            write_zip(unsafe, {"../escape.tsv": "x"})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                validate_zip_artifact(unsafe, GDriveArtifact(filename="unsafe.zip"))

            duplicate = root / "duplicate.zip"
            write_zip(duplicate, {"a/data.tsv": "x", "b/data.tsv": "y"})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_zip_artifact(duplicate, GDriveArtifact(filename="duplicate.zip"))

            case_duplicate = root / "case-duplicate.zip"
            write_zip(case_duplicate, {"A.tsv": "x", "a.tsv": "y"})
            with self.assertRaisesRegex(ValueError, "case-insensitive"):
                validate_zip_artifact(
                    case_duplicate,
                    GDriveArtifact(filename="case-duplicate.zip"),
                )

    def test_archive_limits_are_enforced_before_crc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "limits.zip"
            write_zip(
                archive_path,
                {"one.tsv": "12345", "two.tsv": "67890"},
                compression=zipfile.ZIP_DEFLATED,
            )
            with self.assertRaisesRegex(ValueError, "exceeding the limit"):
                validate_zip_artifact(
                    archive_path,
                    GDriveArtifact(filename="limits.zip", max_members=1),
                )
            with self.assertRaisesRegex(ValueError, "uncompressed size limit"):
                validate_zip_artifact(
                    archive_path,
                    GDriveArtifact(filename="limits.zip", max_member_bytes=4),
                )

            ratio_path = root / "ratio.zip"
            write_zip(
                ratio_path,
                {"repeated.tsv": "a" * 10000},
                compression=zipfile.ZIP_DEFLATED,
            )
            with self.assertRaisesRegex(ValueError, "compression-ratio"):
                validate_zip_artifact(
                    ratio_path,
                    GDriveArtifact(filename="ratio.zip", max_compression_ratio=2.0),
                )

    def test_cache_is_bound_to_source_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "source-a.zip"
            source_b = root / "source-b.zip"
            write_zip(source_a, {"a.tsv": "text\tvalence\tarousal\na\t0.1\t0.2\n"})
            write_zip(source_b, {"b.tsv": "text\tvalence\tarousal\nb\t0.3\t0.4\n"})
            destination = root / "cache"

            first_downloader = RecordingDownloader(source_a)
            first_artifact = GDriveArtifact(filename="bundle.zip", file_id="first-id")
            download_gdrive_artifact(
                first_artifact,
                destination,
                downloader=first_downloader,
            )

            second_downloader = RecordingDownloader(source_b)
            second_artifact = GDriveArtifact(filename="bundle.zip", file_id="second-id")
            download_gdrive_artifact(
                second_artifact,
                destination,
                downloader=second_downloader,
            )
            self.assertEqual(len(second_downloader.calls), 1)
            self.assertEqual(second_downloader.calls[0]["id"], "second-id")

            metadata = json.loads(
                (destination / "bundle.zip.metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["request"]["source"],
                {"kind": "gdrive-id", "value": "second-id"},
            )
            with self.assertRaisesRegex(RuntimeError, "different source"):
                download_gdrive_artifact(first_artifact, destination, offline=True)
            self.assertEqual(
                download_gdrive_artifact(second_artifact, destination, offline=True),
                destination / "bundle.zip",
            )

    def test_staged_extraction_never_mixes_archive_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "bundle.zip"
            external_dir = root / "external"
            write_zip(
                archive_path,
                {
                    "same.tsv": "text\tvalence\tarousal\nold\t0.1\t0.2\n",
                    "removed.tsv": "text\tvalence\tarousal\nstale\t0.2\t0.3\n",
                },
            )
            _extract_zip_tsv(archive_path, external_dir)

            write_zip(
                archive_path,
                {
                    "same.tsv": "text\tvalence\tarousal\nnew\t0.4\t0.5\n",
                    "added.tsv": "text\tvalence\tarousal\nadded\t0.6\t0.7\n",
                },
            )
            extracted = _extract_zip_tsv(archive_path, external_dir)
            self.assertEqual(
                sorted(Path(path).name for path in extracted),
                ["added.tsv", "same.tsv"],
            )
            frames = _load_external_sources(external_dir)
            texts = set(pd.concat(frames, ignore_index=True)["text"])
            self.assertEqual(texts, {"new", "added"})
            versions = [
                path
                for path in (external_dir / ".va_gaze_extracted").iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(versions), 1)

    def test_failed_staged_extraction_keeps_previous_active_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "bundle.zip"
            external_dir = root / "external"
            write_zip(
                archive_path,
                {"sample.tsv": "text\tvalence\tarousal\nold\t0.1\t0.2\n"},
            )
            _extract_zip_tsv(archive_path, external_dir)

            write_zip(
                archive_path,
                {"sample.tsv": "text\tvalence\tarousal\nnew\t0.3\t0.4\n"},
            )
            with mock.patch(
                "va_gaze.data.prepare_english_data._stream_zip_member",
                side_effect=OSError("simulated extraction failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    _extract_zip_tsv(archive_path, external_dir)

            frames = _load_external_sources(external_dir)
            self.assertEqual(
                set(pd.concat(frames, ignore_index=True)["text"]),
                {"old"},
            )

    def test_build_manifest_invalidates_stale_outputs_and_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external_dir = root / "external"
            output_dir = root / "output"
            external_dir.mkdir()
            source_path = external_dir / "sample.tsv"
            source_path.write_text(
                "text\tvalence\tarousal\nfirst\t0.1\t0.2\nsecond\t0.8\t0.7\n",
                encoding="utf-8",
            )

            build_english_dataset(
                output_dir,
                seed=1,
                external_dir=external_dir,
                skip_gdrive_download=True,
            )
            manifest_path = output_dir / BUILD_MANIFEST_NAME
            first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(first_manifest["build"]["seed"], 1)

            with mock.patch(
                "va_gaze.data.prepare_english_data._write_dataset_outputs_atomically"
            ) as writer:
                build_english_dataset(
                    output_dir,
                    seed=1,
                    external_dir=external_dir,
                    skip_gdrive_download=True,
                )
                writer.assert_not_called()

            build_english_dataset(
                output_dir,
                seed=2,
                external_dir=external_dir,
                skip_gdrive_download=True,
            )
            second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(second_manifest["build"]["seed"], 2)

            fold1_path = output_dir / "full_dataset_fold1.csv"
            fold1_path.write_text("corrupt", encoding="utf-8")
            build_english_dataset(
                output_dir,
                seed=2,
                external_dir=external_dir,
                skip_gdrive_download=True,
            )
            self.assertTrue(fold1_path.read_text(encoding="utf-8").startswith("index\ttext"))

            previous_source_hash = second_manifest["build"]["source_hash"]
            source_path.write_text(
                "text\tvalence\tarousal\nfirst\t0.1\t0.2\nthird\t0.4\t0.9\n",
                encoding="utf-8",
            )
            build_english_dataset(
                output_dir,
                seed=2,
                external_dir=external_dir,
                skip_gdrive_download=True,
            )
            changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertNotEqual(
                changed_manifest["build"]["source_hash"],
                previous_source_hash,
            )

    def test_build_can_use_source_bound_cache_offline_without_repeating_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.zip"
            external_dir = root / "external"
            output_dir = root / "output"
            write_zip(
                source,
                {
                    "sample.tsv": (
                        "text\tvalence\tarousal\n"
                        "first\t0.1\t0.2\n"
                        "second\t0.8\t0.7\n"
                    )
                },
            )
            download_gdrive_artifact(
                GDriveArtifact(filename="english_va_bundle.zip", file_id="private-id"),
                external_dir,
                downloader=RecordingDownloader(source),
            )

            build_english_dataset(
                output_dir,
                seed=3,
                external_dir=external_dir,
                offline=True,
            )
            self.assertTrue((output_dir / "full_dataset_fold1.csv").is_file())
            self.assertTrue((output_dir / BUILD_MANIFEST_NAME).is_file())

    def test_atomic_output_staging_preserves_existing_files_on_write_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"output-{index}.tsv" for index in range(3)]
            for path in paths:
                path.write_text(f"original-{path.name}", encoding="utf-8")
            dataframe = pd.DataFrame(
                [{"index": 0, "text": "sample", "valence": 0.1, "arousal": 0.2}]
            )
            outputs = {path: dataframe for path in paths}
            real_write = __import__(
                "va_gaze.data.prepare_english_data",
                fromlist=["_write_tsv"],
            )._write_tsv
            call_count = 0

            def fail_on_third_write(frame, path):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise OSError("simulated write failure")
                real_write(frame, path)

            with mock.patch(
                "va_gaze.data.prepare_english_data._write_tsv",
                side_effect=fail_on_third_write,
            ):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    _write_dataset_outputs_atomically(
                        outputs,
                        root / "manifest.json",
                        {"seed": 1},
                    )
            for path in paths:
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    f"original-{path.name}",
                )


if __name__ == "__main__":
    unittest.main()
