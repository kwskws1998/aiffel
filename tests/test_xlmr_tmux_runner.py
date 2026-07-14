"""Contract tests for the single-pane four-GPU XLM-R launcher."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_xlmr_hetero_4gpu_tmux.sh"


class XlmrTmuxRunnerTest(unittest.TestCase):
    def test_dry_run_uses_one_supervisor_and_offline_workers(self):
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        environment = os.environ.copy()
        environment.update(
            {
                "ATTACH": "0",
                "DRY_RUN": "1",
                "PYTHON_BIN": sys.executable,
                "RUN_TAG": "unittest",
            }
        )
        dry_run = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("tmux layout: one pane with four prefixed workers", dry_run.stdout)
        self.assertIn("HF_HUB_OFFLINE=1", dry_run.stdout)
        self.assertIn("Single-pane supervisor command:", dry_run.stdout)
        self.assertNotIn("split-window", RUNNER.read_text(encoding="utf-8"))

    def test_supervisor_preserves_each_failed_worker_exit_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_python = root / "fake_python.sh"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-\" ]]; then\n"
                "  exec \"$REAL_PYTHON\" \"$@\"\n"
                "fi\n"
                "exit \"${FAKE_WORKER_EXIT:-0}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "ATTACH": "0",
                    "DRY_RUN": "0",
                    "FAKE_WORKER_EXIT": "7",
                    "LOG_ROOT": str(root),
                    "PRELOAD_MODELS": "0",
                    "PYTHON_BIN": str(fake_python),
                    "REAL_PYTHON": sys.executable,
                    "RUN_TAG": "supervisor_unittest",
                    "SESSION_NAME": "supervisor_unittest",
                    "VA_GAZE_SUPERVISOR_MODE": "1",
                }
            )
            supervisor = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(supervisor.returncode, 0, supervisor.stderr)
            self.assertEqual((root / "supervisor.exit_code").read_text().strip(), "1")

            exit_rows = (root / "exit_codes.tsv").read_text().strip().splitlines()
            self.assertEqual(len(exit_rows), 5)
            self.assertTrue(all(row.endswith("\t7") for row in exit_rows[1:]))
            status_files = sorted(root.glob("gpu*.status.tsv"))
            self.assertEqual(len(status_files), 4)
            for status_file in status_files:
                self.assertEqual(status_file.read_text().strip().splitlines()[-1], "7\t0")


if __name__ == "__main__":
    unittest.main()
