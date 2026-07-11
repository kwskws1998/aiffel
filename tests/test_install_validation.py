import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class InstallValidationTest(unittest.TestCase):
    def run_install(self, extra_environment):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "SKIP_DEPS": "1",
                "WITH_ET1": "0",
            }
        )
        environment.update(extra_environment)
        return subprocess.run(
            ["bash", "install.sh"],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_placeholder_drive_id_fails_before_setup(self):
        result = self.run_install(
            {
                "DATA_ZIP_FILE_ID": "<Google-Drive-file-id>",
                "DATA_ZIP_URL": "",
                "CONDA_PREFIX": "",
                "VIRTUAL_ENV": "",
            }
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("example placeholder", result.stderr)
        self.assertNotIn("[1/3] Python dependencies", result.stdout)

    def test_nested_conda_and_virtualenv_fails_before_setup(self):
        result = self.run_install(
            {
                "DATA_ZIP_FILE_ID": "",
                "DATA_ZIP_URL": "",
                "CONDA_PREFIX": "/opt/conda/envs/va_gaze",
                "CONDA_DEFAULT_ENV": "va_gaze",
                "VIRTUAL_ENV": "/workspace/aiffel/.venv",
            }
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Conda and virtualenv are active", result.stderr)
        self.assertIn("unset PYTHON_BIN VIRTUAL_ENV", result.stderr)
        self.assertNotIn("[1/3] Python dependencies", result.stdout)

    def test_skip_deps_rejects_an_incomplete_environment(self):
        result = self.run_install(
            {
                "PYTHON_BIN": "/usr/bin/false",
                "DATA_ZIP_FILE_ID": "",
                "DATA_ZIP_URL": "",
                "CONDA_PREFIX": "",
                "VIRTUAL_ENV": "",
            }
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("core dependencies are missing", result.stderr)
        self.assertIn("setup_distilbert_conda_cloud.sh", result.stderr)


if __name__ == "__main__":
    unittest.main()
