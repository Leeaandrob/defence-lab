#!/usr/bin/env bash
# Install SAM2 from source (ARM64 / CUDA 12.8 friendly).
# torch is already present in this environment; we install sam2 without pulling
# a different torch build. Run from the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/third_party/sam2"

echo "[install_sam2] python: $(command -v python3)  torch: $(python3 -c 'import torch; print(torch.__version__)')"

mkdir -p "${REPO_ROOT}/third_party"
if [ ! -d "${SRC}/.git" ]; then
  git clone https://github.com/facebookresearch/sam2.git "${SRC}"
fi

cd "${SRC}"
# --no-build-isolation reuses the already-installed torch instead of fetching
# a fresh (possibly x86) wheel into the build env -- important on ARM64.
# SAM2_BUILD_ALLOW_ERRORS=1: the optional CUDA post-processing extension
# (connected-components for hole-filling) may not compile on every ARM/CUDA
# combo; the model runs without it, so we don't let it fail the install.
SAM2_BUILD_ALLOW_ERRORS=1 python3 -m pip install --no-build-isolation -e . --user

echo "[install_sam2] verifying import ..."
python3 -c "import sam2; from sam2.build_sam import build_sam2; print('sam2 OK:', sam2.__file__)"
echo "[install_sam2] done. Next: scripts/download_checkpoints.sh"
