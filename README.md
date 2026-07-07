# PathFold

PathFold is a diffusion-based protein folding pathway inference repository at C-alpha resolution. It predicts the next folding intermediate from:

- precomputed AlphaFold2 embeddings
- one or more known previous intermediate structures
- a trained checkpoint matched to the number of previous states

This repo is set up around inference. It includes working checkpoints for `prev1`, `prev3`, and `prev6`, plus one bundled real example target (`4INW_A`) for testing.

## What This Repo Contains

- `pathfold/`: model, diffusion, utilities, and inference code
- `checkpoints/`: bundled model checkpoints used by this repo
- `data/example_4INW_A/`: one real example with embeddings, folded reference, and initial frames
- `scripts/run_example_4INW_A.sh`: helper script for testing the bundled example
- `configs/`: simple config references for `prev1`, `prev3`, and `prev6`

## Setup

From the repo root:

```bash
git clone https://github.com/kiharalab/PathFold.git
cd PathFold
python -m pip install -e .
```

That is enough to make:

```bash
python -m pathfold.inference.run_inference
```

work from this repo.

## CUDA / PyTorch Compatibility

This repo has been tested with a PyTorch build targeting CUDA 13.

If your machine has an older NVIDIA driver or a different CUDA stack, GPU inference may fail with a driver compatibility error. In that case, reinstall PyTorch with a build that matches the CUDA version supported by your system.

See:

- `https://pytorch.org/get-started/locally/`
- `https://pytorch.org/get-started/previous-versions/`

## Checkpoints

The repo currently includes these checkpoints:

- `folding_after50_08062024`, `version_0`, `epoch=6` for `prev1`
- `folding_after3x50_04112025`, `version_1`, `epoch=5` for `prev3`
- `folding_after6x50_04112025`, `version_2`, `epoch=5` for `prev6`

Checkpoint layout:

- `<model_root>/<model_name>/configuration`
- `<model_root>/<model_name>/version_<N>/checkpoints/epoch=<E>.ckpt`

## Required Inputs

Inference expects:

- one checkpoint
- one AF2 embedding `.npz`
- one folded reference PDB
- previous known structure(s)

The embedding file must contain these embeddings:

- `single`: typically shape `[L, 384]`
- `pair`: typically shape `[L, L, 128]`

`prev1`, `prev3`, and `prev6` mean how many previous structures are provided:

- `prev1`: 1 initial structure
- `prev3`: 3 initial structures
- `prev6`: 6 initial structures

These are passed with:

- `--prev_frames`
- `--initial_structures`

## Embedding Generation

PathFold uses AlphaFold2 single and pair embeddings as conditioning features.
Generate them by running the official AlphaFold2 repository:

```bash
git clone https://github.com/google-deepmind/alphafold.git
cd alphafold
```

The official AlphaFold2 code does not save these embeddings by default. At
line 94 in `alphafold/model/model.py`, add `return_representations=True` to the
`return model(...)` call so embeddings are returned:

```python
return model(
    batch,
    is_training=False,
    compute_loss=False,
    ensemble_representations=True,
    return_representations=True,
)
```

After running AlphaFold2, the output pickle such as `result_model_1_pred_0.pkl`
contains the needed embeddings at:

- `result["representations"]["single"]`
- `result["representations"]["pair"]`

Convert them into the `.npz` format expected by PathFold:

```bash
python - <<'PY'
import pickle
import numpy as np

result_pkl = "result_model_1_pred_0.pkl"
output_npz = "target_af2_embedding.npz"

with open(result_pkl, "rb") as handle:
    result = pickle.load(handle)

representations = result["representations"]
np.savez_compressed(
    output_npz,
    single=representations["single"],
    pair=representations["pair"],
)
PY
```

Pass the generated file to PathFold with `--af2_embedding`.

## Bundled Example

This repo includes one real example bundle:

- `data/example_4INW_A`

It contains:

- `folded_reference.pdb`
- `embeddings/4INW_A.npz`
- six initial frame PDBs in `initial_frames/`

Quick tests:

```bash
cd PathFold
bash scripts/run_example_4INW_A.sh prev1
```

```bash
cd PathFold
bash scripts/run_example_4INW_A.sh prev3
```

```bash
cd PathFold
bash scripts/run_example_4INW_A.sh prev6
```

The helper script uses GPU by default with `cuda:0`. To choose a different GPU:

```bash
bash scripts/run_example_4INW_A.sh prev3 1
```

## Direct Inference Command

Example `prev1` run:

```bash
cd PathFold

python -m pathfold.inference.run_inference \
  --model_root checkpoints \
  --model_name folding_after50_08062024 \
  --model_version 0 \
  --model_epoch 6 \
  --prev_frames 1 \
  --initial_structures data/example_4INW_A/initial_frames/4INW_A_frame_0482.pdb \
  --folded_pdb data/example_4INW_A/folded_reference.pdb \
  --af2_embedding data/example_4INW_A/embeddings/4INW_A.npz \
  --out_dir outputs/example_4INW_A_prev1
```

Example `prev3` run:

```bash
cd PathFold

python -m pathfold.inference.run_inference \
  --model_root checkpoints \
  --model_name folding_after3x50_04112025 \
  --model_version 1 \
  --model_epoch 5 \
  --prev_frames 3 \
  --initial_structures \
  data/example_4INW_A/initial_frames/4INW_A_frame_0482.pdb \
  data/example_4INW_A/initial_frames/4INW_A_frame_0486.pdb \
  data/example_4INW_A/initial_frames/4INW_A_frame_0492.pdb \
  --folded_pdb data/example_4INW_A/folded_reference.pdb \
  --af2_embedding data/example_4INW_A/embeddings/4INW_A.npz \
  --out_dir outputs/example_4INW_A_prev3
```

## Runtime Defaults

Current inference defaults:

- GPU: `cuda:0`
- `stop_similarity`: `0.90`
- `max_progress_per_step`: `0.2`
- progress bars: enabled
- sampling progress bars: enabled

Other defaults come from `pathfold/inference/run_inference.py`.

## Outputs

Each run writes into `--out_dir`:

- `step_000.pdb`, `step_001.pdb`, ...: selected intermediate structures
- `step_000.npy`, `step_001.npy`, ...: selected C-alpha coordinates
- optional `samples_step_*` directories if `--save_all_samples` is enabled
