import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_distilbert_multiseed.py"
SPEC = importlib.util.spec_from_file_location("summarize_distilbert_multiseed", SCRIPT_PATH)
SUMMARY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_MODULE)


class MultiseedSummaryTest(unittest.TestCase):
    def write_metrics(self, root, condition, seed, mse):
        run_dir = root / f"{condition}_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "overall_metrics.json").write_text(
            json.dumps({"num_samples": 10, "mse_valence": mse}),
            encoding="utf-8",
        )

    def test_condition_mean_and_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_metrics(root, "baseline", 42, 1.0)
            self.write_metrics(root, "baseline", 123, 3.0)
            runs = SUMMARY_MODULE.collect_runs(root)
            SUMMARY_MODULE.validate_expected_seeds(runs, [42, 123])
            summary = SUMMARY_MODULE.summarize_runs(runs)
            self.assertEqual(summary[0]["condition"], "baseline")
            self.assertEqual(summary[0]["n_runs"], 2)
            self.assertEqual(summary[0]["mse_valence_mean"], 2.0)
            self.assertAlmostEqual(summary[0]["mse_valence_std"], 2 ** 0.5)

    def test_incomplete_condition_seed_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_metrics(root, "baseline", 42, 1.0)
            runs = SUMMARY_MODULE.collect_runs(root)
            with self.assertRaisesRegex(RuntimeError, "Incomplete seeds"):
                SUMMARY_MODULE.validate_expected_seeds(runs, [42, 123])


if __name__ == "__main__":
    unittest.main()
