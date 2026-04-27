#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-prev3}"
GPU_ID="${2:-0}"

cd "$ROOT"

COMMON_ARGS=(
  --model_root checkpoints
  --gpu "$GPU_ID"
  --folded_pdb data/example_4INW_A/folded_reference.pdb
  --af2_embedding data/example_4INW_A/embeddings/4INW_A.npz
  --out_dir "outputs/example_4INW_A_${MODE}"
)

case "$MODE" in
  prev1)
    exec "$PYTHON_BIN" -m alphapathfold.inference.run_inference \
      "${COMMON_ARGS[@]}" \
      --model_name folding_after50_08062024 \
      --model_version 0 \
      --model_epoch 6 \
      --prev_frames 1 \
      --initial_structures \
      data/example_4INW_A/initial_frames/4INW_A_frame_0482.pdb
    ;;
  prev3)
    exec "$PYTHON_BIN" -m alphapathfold.inference.run_inference \
      "${COMMON_ARGS[@]}" \
      --model_name folding_after3x50_04112025 \
      --model_version 1 \
      --model_epoch 5 \
      --prev_frames 3 \
      --initial_structures \
      data/example_4INW_A/initial_frames/4INW_A_frame_0482.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0486.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0492.pdb
    ;;
  prev6)
    exec "$PYTHON_BIN" -m alphapathfold.inference.run_inference \
      "${COMMON_ARGS[@]}" \
      --model_name folding_after6x50_04112025 \
      --model_version 2 \
      --model_epoch 5 \
      --prev_frames 6 \
      --initial_structures \
      data/example_4INW_A/initial_frames/4INW_A_frame_0482.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0484.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0486.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0488.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0490.pdb \
      data/example_4INW_A/initial_frames/4INW_A_frame_0492.pdb
    ;;
  *)
    echo "Usage: $0 {prev1|prev3|prev6}" >&2
    exit 1
    ;;
esac
