#!/usr/bin/env bash
# Run Agibot dataset preprocessing: preprocess_dataset.py, then trim_static_frames.py on OUTPUT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF' >&2
Usage:
  prepross_agibot_data.sh SOURCE OUTPUT [preprocess_dataset.py options...] [-- trim_static_frames.py options...]

Pipeline:
  1) preprocess_dataset.py --source SOURCE --output OUTPUT [options before --]
  2) trim_static_frames.py OUTPUT [options after --]

  The first unescaped "--" separates preprocess options from trim options.
  If you omit "--", nothing extra is passed to trim_static_frames.py (only OUTPUT).

preprocess_dataset.py options (all optional unless noted; SOURCE/OUTPUT are set by this wrapper):
  --skip_copy              When output exists, skip re-copying from source (process existing output)
  --skip_video             Skip video resolution preprocessing
  --modality-template PATH Path to modality.json template (optional)
  --target-size H W        Target video resolution (default: 256 256)
  --workers N              Worker processes for video preprocessing (default: CPU count)
  --no-waist               Omit waist_position from modality.json (included by default)

trim_static_frames.py options (after "--"; dataset path is always OUTPUT from above):
  --dry-run                Print planned trim actions only; do not modify files
  --frame-index-col NAME   Frame index column in parquet (default: frame_index)
  --workers N              Concurrent threads for trimming (default: 4)

Examples:
  prepross_agibot_data.sh /data/raw /data/out
  prepross_agibot_data.sh /data/raw /data/out --skip_video
  prepross_agibot_data.sh /data/raw /data/out --workers 8 -- --workers 16
  prepross_agibot_data.sh /data/raw /data/out -- --dry-run

Python: uses "uv run python" when uv is available, else "python3" (from repo root).
EOF
  exit 1
}

[[ $# -lt 2 ]] && usage

SOURCE="$1"
OUTPUT="$2"
shift 2

preprocess_args=()
trim_extra=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    trim_extra=("$@")
    break
  fi
  preprocess_args+=("$1")
  shift
done

if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python3)
fi

"${PY[@]}" "${SCRIPT_DIR}/preprocess_dataset.py" --source "${SOURCE}" --output "${OUTPUT}" "${preprocess_args[@]}"
"${PY[@]}" "${SCRIPT_DIR}/trim_static_frames.py" "${OUTPUT}" "${trim_extra[@]}"
