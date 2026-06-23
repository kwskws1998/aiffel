#!/usr/bin/env bash
# Packages run-result directories and a small manifest into a timestamped zip.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if command -v git >/dev/null 2>&1; then
  git_root="$(git -C "${repo_root}" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "${git_root}" ]]; then
    repo_root="${git_root}"
  fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${1:-${repo_root}/result_packages}"
archive_name="${2:-run_results_${timestamp}.zip}"
archive_path="${out_dir}/${archive_name}"
manifest_root="$(mktemp -d)"
manifest_dir="${manifest_root}/package_manifest"

cleanup() {
  rm -rf "${manifest_root}"
}
trap cleanup EXIT

mkdir -p "${out_dir}" "${manifest_dir}"

candidate_paths=(
  "artifacts"
  "Preds"
  "Output Directory"
  "checkpoints"
  "model"
  "runs"
  "logs"
  "wandb"
  "lightning_logs"
  "tb_logs"
  "mlruns"
  "outputs"
  "results"
)

include_paths=()
for path in "${candidate_paths[@]}"; do
  if [[ -e "${repo_root}/${path}" ]]; then
    include_paths+=("${path}")
  fi
done

if [[ ${#include_paths[@]} -eq 0 ]]; then
  echo "No result directories found under ${repo_root}" >&2
  exit 1
fi

{
  echo "created_at=${timestamp}"
  echo "repo_root=${repo_root}"
  if command -v git >/dev/null 2>&1; then
    echo "git_head=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git -C "${repo_root}" branch --show-current 2>/dev/null || true)"
  fi
  echo
  echo "included_paths:"
  printf '%s\n' "${include_paths[@]}"
} > "${manifest_dir}/manifest.txt"

{
  cd "${repo_root}"
  find "${include_paths[@]}" -type f \
    ! -name ".DS_Store" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    -print | sort
} > "${manifest_dir}/file_list.txt"

{
  cd "${repo_root}"
  du -sh "${include_paths[@]}" 2>/dev/null || true
} > "${manifest_dir}/disk_usage.txt"

if command -v git >/dev/null 2>&1; then
  git -C "${repo_root}" status --short > "${manifest_dir}/git_status_short.txt" 2>/dev/null || true
fi

if command -v zip >/dev/null 2>&1; then
  (
    cd "${repo_root}"
    zip -r "${archive_path}" "${include_paths[@]}" \
      -x "*/.DS_Store" \
      -x "*/__pycache__/*" \
      -x "*.pyc" \
      -x "result_packages/*"
  )
  (
    cd "${manifest_root}"
    zip -r "${archive_path}" "package_manifest"
  )
else
  python3 - "${repo_root}" "${archive_path}" "${manifest_root}" "${include_paths[@]}" <<'PY'
import os
import sys
import zipfile

repo_root = sys.argv[1]
archive_path = sys.argv[2]
manifest_root = sys.argv[3]
include_paths = sys.argv[4:]

def should_skip(path):
    parts = path.split(os.sep)
    name = os.path.basename(path)
    return (
        name == ".DS_Store"
        or "__pycache__" in parts
        or name.endswith(".pyc")
        or parts[0] == "result_packages"
    )

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for rel_root in include_paths:
        abs_root = os.path.join(repo_root, rel_root)
        if os.path.isfile(abs_root):
            if not should_skip(rel_root):
                zf.write(abs_root, rel_root)
            continue
        for current_root, _, files in os.walk(abs_root):
            for filename in files:
                abs_path = os.path.join(current_root, filename)
                rel_path = os.path.relpath(abs_path, repo_root)
                if not should_skip(rel_path):
                    zf.write(abs_path, rel_path)
    package_manifest = os.path.join(manifest_root, "package_manifest")
    for current_root, _, files in os.walk(package_manifest):
        for filename in files:
            abs_path = os.path.join(current_root, filename)
            rel_path = os.path.relpath(abs_path, manifest_root)
            zf.write(abs_path, rel_path)
PY
fi

echo "${archive_path}"
