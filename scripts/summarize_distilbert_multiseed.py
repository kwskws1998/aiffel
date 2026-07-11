"""Aggregate condition-level mean and sample standard deviation across seeds."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics


RUN_PATTERN = re.compile(r"^(?P<condition>.+)_seed(?P<seed>[0-9]+)$")


def collect_runs(output_root):
    runs = []
    for run_dir in sorted(Path(output_root).iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.match(run_dir.name)
        metrics_path = run_dir / "overall_metrics.json"
        if match is None or not metrics_path.is_file():
            continue
        with open(metrics_path, "r", encoding="utf-8") as input_file:
            metrics = json.load(input_file)
        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if key != "num_samples" and isinstance(value, (int, float)) and value is not None
        }
        runs.append(
            {
                "condition": match.group("condition"),
                "seed": int(match.group("seed")),
                "metrics": numeric_metrics,
            }
        )
    if not runs:
        raise RuntimeError(f"No completed condition_seed results found under {output_root}.")
    return runs


def validate_expected_seeds(runs, expected_seeds):
    expected = set(expected_seeds)
    conditions = sorted({run["condition"] for run in runs})
    for condition in conditions:
        actual = {run["seed"] for run in runs if run["condition"] == condition}
        if actual != expected:
            raise RuntimeError(
                f"Incomplete seeds for {condition}: expected {sorted(expected)}, got {sorted(actual)}."
            )


def summarize_runs(runs):
    summary = []
    conditions = sorted({run["condition"] for run in runs})
    for condition in conditions:
        condition_runs = [run for run in runs if run["condition"] == condition]
        metric_names = sorted(set.intersection(*(set(run["metrics"]) for run in condition_runs)))
        row = {
            "condition": condition,
            "n_runs": len(condition_runs),
            "seeds": ",".join(str(run["seed"]) for run in sorted(condition_runs, key=lambda item: item["seed"])),
        }
        for metric_name in metric_names:
            values = [run["metrics"][metric_name] for run in condition_runs]
            row[f"{metric_name}_mean"] = statistics.fmean(values)
            row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-seeds", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    runs = collect_runs(output_root)
    expected_seeds = [int(value) for value in args.expected_seeds.replace(",", " ").split()]
    if expected_seeds:
        validate_expected_seeds(runs, expected_seeds)
    summary = summarize_runs(runs)

    run_rows = [
        {"condition": run["condition"], "seed": run["seed"], **run["metrics"]}
        for run in runs
    ]
    write_csv(output_root / "multiseed_runs.csv", run_rows)
    write_csv(output_root / "multiseed_summary.csv", summary)
    with open(output_root / "multiseed_summary.json", "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")

    print(f"Aggregated {len(runs)} runs across {len(summary)} conditions: {output_root}")


if __name__ == "__main__":
    main()
