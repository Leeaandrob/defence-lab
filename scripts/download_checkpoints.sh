#!/usr/bin/env bash
# Download SAM 2.1 checkpoints into checkpoints/.
# Default: base-plus (good speed/quality balance for a shared GPU). Pass a size
# as $1: tiny | small | base_plus | large
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/checkpoints"
mkdir -p "${DEST}"

SIZE="${1:-base_plus}"
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

declare -A FILES=(
  [tiny]="sam2.1_hiera_tiny.pt"
  [small]="sam2.1_hiera_small.pt"
  [base_plus]="sam2.1_hiera_base_plus.pt"
  [large]="sam2.1_hiera_large.pt"
)

f="${FILES[$SIZE]:-}"
if [ -z "${f}" ]; then
  echo "Unknown size '${SIZE}'. Choose: tiny | small | base_plus | large" >&2
  exit 1
fi

echo "[download] ${SIZE} -> ${DEST}/${f}"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "${DEST}/${f}" "${BASE_URL}/${f}"
else
  curl -L -C - -o "${DEST}/${f}" "${BASE_URL}/${f}"
fi
echo "[download] done: ${DEST}/${f}"
