import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm

from alphapathfold.util_af import protein as protein_utils
from alphapathfold.utils.data_io import load_coord
from alphapathfold.utils.model_io import load_model

warnings.simplefilter("ignore", PDBConstructionWarning)


logger = logging.getLogger(__name__)


def load_af2_embeddings(embedding_path, device):
    embedding = np.load(embedding_path)
    if "single" not in embedding or "pair" not in embedding:
        raise ValueError(
            f"{embedding_path} must contain 'single' and 'pair' arrays."
        )
    single = torch.tensor(embedding["single"]).to(device)
    pair = torch.tensor(embedding["pair"]).to(device)
    return single, pair


def load_protein(pdb_path):
    with open(pdb_path, "r", encoding="utf-8") as handle:
        return protein_utils.from_pdb_string(handle.read())


def parse_pdb_to_backbone_triplets(pdb_path):
    protein = load_protein(pdb_path)
    n_pos = protein.atom_positions[..., 0, :]
    ca_pos = protein.atom_positions[..., 1, :]
    c_pos = protein.atom_positions[..., 2, :]
    coords = []
    for idx in range(len(n_pos)):
        coords.append(n_pos[idx])
        coords.append(ca_pos[idx])
        coords.append(c_pos[idx])
    return np.asarray(coords)


def load_ca_coords(path, treat_as_initial_structure):
    if path.endswith(".pdb"):
        coords = parse_pdb_to_backbone_triplets(path)
    else:
        coords = load_coord(path)
    if treat_as_initial_structure:
        return coords[1::3]
    return coords


def coords_to_contact_map(coords, threshold):
    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    pairwise = torch.cdist(coords_tensor, coords_tensor)
    return torch.where(
        pairwise < threshold,
        torch.ones_like(pairwise),
        torch.zeros_like(pairwise),
    ).squeeze(0)


def dice_similarity(contact_a, contact_b):
    intersection = (contact_a * contact_b).sum()
    union = contact_a.sum() + contact_b.sum()
    return ((2 * intersection) / (union + 1e-6)).item()


def output_ca_only_atom37(coords):
    ret = np.zeros((coords.shape[0], 37, 3))
    ret[:, 1, :] = coords
    return ret


def build_contact_stack(initial_structures, length, threshold):
    contacts = torch.zeros((len(initial_structures), length, length))
    for idx, structure_path in enumerate(initial_structures):
        coords = load_ca_coords(structure_path, treat_as_initial_structure=True)
        contacts[idx, ...] = coords_to_contact_map(coords, threshold=threshold)
    return contacts


def write_pdb(protein_template, coords, output_path):
    output_protein = protein_utils.Protein(
        aatype=protein_template.aatype,
        atom_positions=output_ca_only_atom37(coords),
        atom_mask=protein_template.atom_mask,
        residue_index=protein_template.residue_index,
        b_factors=protein_template.b_factors,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(protein_utils.to_pdb(output_protein))


def expand_conditioning(single, pair, frame_input, batch_size):
    if single.dim() == 2:
        single = single.unsqueeze(0)
    if pair.dim() == 3:
        pair = pair.unsqueeze(0)
    if single.shape[0] == 1 and batch_size > 1:
        single = single.expand(batch_size, -1, -1)
    if pair.shape[0] == 1 and batch_size > 1:
        pair = pair.expand(batch_size, -1, -1, -1)
    if frame_input.shape[0] == 1 and batch_size > 1:
        frame_input = frame_input.expand(batch_size, -1, -1, -1)
    return single, pair, frame_input


def select_best_candidate(
    candidates,
    prev_contact,
    folded_contact,
    current_target_dice,
    max_progress_per_step,
    selection_contact_threshold,
):
    best = None
    best_score = None
    for coords in candidates:
        pred_contact = coords_to_contact_map(coords, threshold=selection_contact_threshold)
        dice_to_prev = dice_similarity(prev_contact, pred_contact)
        dice_to_target = dice_similarity(pred_contact, folded_contact)
        progress = dice_to_target - current_target_dice

        if progress <= 0:
            continue
        if max_progress_per_step is not None and progress > max_progress_per_step:
            continue

        score = progress
        if best is None or score > best_score:
            best = (coords, pred_contact, dice_to_prev, dice_to_target)
            best_score = score
    return best


def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(args.initial_structures) != args.prev_frames:
        raise ValueError(
            "--prev_frames must match the number of paths passed to --initial_structures."
        )

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    logger.info("Starting AlphaPathFold inference")
    logger.info(
        "Model=%s version=%s epoch=%s prev_frames=%s device=%s",
        args.model_name,
        args.model_version,
        args.model_epoch,
        args.prev_frames,
        device,
    )
    logger.info("Output directory: %s", args.out_dir)

    logger.info("Loading model checkpoint")
    model = load_model(
        args.model_root,
        args.model_name,
        args.model_version,
        args.model_epoch,
        frame=True,
        prev_frame=args.prev_frames,
    ).to(device)
    logger.info("Model loaded successfully")

    logger.info("Loading AF2 embeddings from %s", args.af2_embedding)
    single, pair = load_af2_embeddings(args.af2_embedding, device)
    logger.info("Embedding shapes: single=%s pair=%s", tuple(single.shape), tuple(pair.shape))

    logger.info("Loading folded reference PDB from %s", args.folded_pdb)
    folded_protein = load_protein(args.folded_pdb)
    folded_coords = folded_protein.atom_positions[..., 1, :]
    length = folded_coords.shape[0]
    logger.info("Reference length: %s residues", length)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing %s initial structure(s)", len(args.initial_structures))
    for structure_path in args.initial_structures:
        logger.info("Initial structure: %s", structure_path)

    contact_stack = build_contact_stack(
        args.initial_structures,
        length,
        threshold=args.prev_contact_threshold,
    ).to(device)
    prev_contact = coords_to_contact_map(
        load_ca_coords(args.initial_structures[-1], treat_as_initial_structure=True),
        threshold=args.selection_contact_threshold,
    )
    folded_contact = coords_to_contact_map(
        folded_coords,
        threshold=args.selection_contact_threshold,
    )
    target_dice = dice_similarity(prev_contact, folded_contact)
    logger.info("Initial contact-map Dice similarity to folded reference: %.4f", target_dice)

    mask = torch.ones((args.batch_size, length), device=device)
    step = 0
    step_progress = tqdm(
        total=args.max_steps,
        desc="Trajectory steps",
        unit="step",
        disable=not args.show_progress,
    )
    while target_dice < args.stop_similarity and step < args.max_steps:
        logger.info(
            "Step %s/%s | current Dice=%.4f | target=%.4f",
            step + 1,
            args.max_steps,
            target_dice,
            args.stop_similarity,
        )
        candidates = []
        frame_input = contact_stack.unsqueeze(0)
        batch_progress = tqdm(
            range(args.num_batches),
            desc=f"Sampling step {step:03d}",
            unit="batch",
            leave=False,
            disable=not args.show_progress,
        )
        for batch_idx in batch_progress:
            logger.info(
                "Sampling batch %s/%s for trajectory step %s",
                batch_idx + 1,
                args.num_batches,
                step,
            )
            batch_single, batch_pair, batch_frame = expand_conditioning(
                single,
                pair,
                frame_input,
                args.batch_size,
            )
            sampled = model.p_sample_loop(
                mask,
                args.noise_scale,
                verbose=args.show_sampling_progress,
                single=batch_single,
                pair=batch_pair,
                frame=batch_frame,
            )[-1]
            for batch_index in range(sampled.shape[0]):
                coords = sampled[batch_index].trans.detach().cpu().numpy()[:length]
                candidates.append(coords)
            logger.info("Collected %s candidate(s) so far", len(candidates))

        selected = select_best_candidate(
            candidates,
            prev_contact,
            folded_contact,
            target_dice,
            args.max_progress_per_step,
            args.selection_contact_threshold,
        )
        if selected is None:
            logger.warning("No valid candidate improved the target similarity; stopping early")
            break

        coords, pred_contact, dice_to_prev, target_dice = selected
        pdb_path = out_dir / f"step_{step:03d}.pdb"
        npy_path = out_dir / f"step_{step:03d}.npy"
        write_pdb(folded_protein, coords, pdb_path)
        np.savetxt(npy_path, coords, fmt="%.4f", delimiter=",")
        logger.info(
            "Selected candidate for step %s | Dice(prev)=%.4f Dice(target)=%.4f",
            step,
            dice_to_prev,
            target_dice,
        )
        logger.info("Wrote outputs: %s and %s", pdb_path, npy_path)

        if args.save_all_samples:
            sample_dir = out_dir / f"samples_step_{step:03d}"
            sample_dir.mkdir(exist_ok=True)
            for idx, sample_coords in enumerate(candidates):
                np.savetxt(
                    sample_dir / f"candidate_{idx:03d}.npy",
                    sample_coords,
                    fmt="%.4f",
                    delimiter=",",
                )
            logger.info("Saved all %s candidate samples to %s", len(candidates), sample_dir)

        prev_contact = pred_contact
        next_stack = torch.zeros_like(contact_stack)
        if args.prev_frames > 1:
            next_stack[:-1] = contact_stack[1:]
        next_stack[-1] = coords_to_contact_map(
            coords,
            threshold=args.prev_contact_threshold,
        ).to(device)
        contact_stack = next_stack
        step += 1
        step_progress.update(1)
        if args.show_progress:
            step_progress.set_postfix({"dice": f"{target_dice:.4f}"})

    step_progress.close()
    if target_dice >= args.stop_similarity:
        logger.info("Reached target similarity threshold")
    elif step >= args.max_steps:
        logger.info("Reached maximum step limit")
    print(f"Generated {step} intermediate steps in {out_dir}")
    print(f"Final contact-map Dice similarity to folded reference: {target_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AlphaPathFold trajectory inference with precomputed AF2 embeddings."
    )
    parser.add_argument("--model_root", default="checkpoints", help="Checkpoint root directory.")
    parser.add_argument("--model_name", required=True, help="Model directory name under model_root.")
    parser.add_argument("--model_version", type=int, help="Checkpoint version number.")
    parser.add_argument("--model_epoch", type=int, help="Checkpoint epoch number.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id, for example 0. Defaults to 0.")
    parser.add_argument(
        "--prev_frames",
        type=int,
        default=3,
        choices=[1, 3, 6],
        help="Number of previous contact-map states expected by the checkpoint.",
    )
    parser.add_argument(
        "--initial_structures",
        nargs="+",
        required=True,
        help="Initial PDB or coordinate files, ordered from oldest to newest state.",
    )
    parser.add_argument("--folded_pdb", required=True, help="Folded reference PDB used for residue identities and contact-map selection.")
    parser.add_argument("--af2_embedding", required=True, help="Path to a .npz file with AF2 'single' and 'pair' arrays.")
    parser.add_argument("--out_dir", required=True, help="Directory for generated trajectory outputs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Samples per diffusion batch.")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches sampled per trajectory step.")
    parser.add_argument("--noise_scale", type=float, default=0.8, help="Sampling noise scale.")
    parser.add_argument("--stop_similarity", type=float, default=0.90, help="Stop once folded contact-map Dice similarity reaches this threshold.")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of trajectory steps to generate.")
    parser.add_argument("--max_progress_per_step", type=float, default=0.2, help="Optional upper bound on Dice improvement per step.")
    parser.add_argument("--prev_contact_threshold", type=float, default=10.0, help="Distance threshold for previous-state contact maps.")
    parser.add_argument("--selection_contact_threshold", type=float, default=14.0, help="Distance threshold for candidate selection contact maps.")
    parser.add_argument("--save_all_samples", action="store_true", help="Save every sampled candidate as a .npy file.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True, help="Show progress bars for trajectory steps and batch sampling.")
    parser.add_argument("--show_sampling_progress", action=argparse.BooleanOptionalAction, default=True, help="Show the inner diffusion timestep progress bar for each batch.")
    main(parser.parse_args())
