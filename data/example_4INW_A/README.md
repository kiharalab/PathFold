# Example Test Bundle: 4INW_A

This bundle was copied from the local Genie data layout so the cleaned AlphaPathFold repo has one real inference test case.

Included files:

- `folded_reference.pdb`: folded target structure
- `embeddings/4INW_A.npz`: precomputed AF2 embeddings with `single` and `pair`
- `initial_frames/4INW_A_frame_0482.pdb`: starting frame used by the `prev1` script
- `initial_frames/4INW_A_frame_0482.pdb`, `4INW_A_frame_0486.pdb`, `4INW_A_frame_0492.pdb`: frames used by the `prev3` script
- `initial_frames/4INW_A_frame_0482.pdb` through `4INW_A_frame_0492.pdb` in steps of 2: frames used by the `prev6` script
