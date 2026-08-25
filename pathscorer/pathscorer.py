"""
PathScorer: Global Alignment and Similarity Metric for Protein Folding Trajectory

PathScorer performs global structural alignment between a ground truth Molecular 
Dynamics (MD) trajectory and a predicted folding trajectory (e.g., from PathFold) 
using custom dynamic programming (DP). Unlike conventional structural alignment 
methods that rely on linear gap penalties or RMSD, PathScorer is designed 
specifically for highly variable rates of conformational change in complete folding 
pathways. In addition to raw DP-based similarity scoring, PathScorer incorporates 
heuristic penalties for excessively fast or slow transitions and reversals in folding progress, 
capturing both structural and temporal consistency with the native folding process.
"""

import argparse
import os
import re
from contextlib import contextmanager
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Model, Chain, Residue
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import Array
from Bio.SeqUtils import seq1
from pymol import cmd
from Bio.PDB import Model, Chain

warnings.filterwarnings('ignore')   # Suppress Biopython warnings about PDB parsing and structure issues, 
                                    # which are common in dynamic folding trajectories and not critical for 
                                    # our contact map-based similarity scoring.   


@contextmanager
def suppress_stderr_fd():
    """
    Temporarily redirects C-level stderr (fd 2) to os.devnull.
    Prevents underlying external tool wrappers (like PyMOL or Bio.PDB) 
    from flooding the terminal with warnings or formatting issues.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)       # Duplicate current stderr fd
    os.dup2(devnull_fd, 2)            # Redirect fd 2 to /dev/null
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)   # Restore original stderr
        os.close(saved_stderr_fd)


def safe_align(mobile, target, *args, **kwargs):
    """
    Wraps PyMOL's cmd.align to cleanly suppress unexpected C-level 
    warning prints during structural alignment steps.
    """
    with suppress_stderr_fd():
        return cmd.align(mobile, target, *args, **kwargs)


def save_pdb_file(strctr, fl_name):
    """Saves a Bio.PDB Structure object to a local PDB file."""
    io = PDBIO()
    io.set_structure(strctr)
    io.save(fl_name)


def diagonal_width_array(size, width):
    """
    Generates a masking binary array to ignore trivial sequence neighbors.
    Returns a matrix where elements within `width//2` of the primary diagonal 
    are marked 0 (ignored), and all other elements are marked 1 (kept).
    """
    arr = np.zeros((size, size), dtype=int)
    for i in range(-width//2, width//2 + 1):
        np.fill_diagonal(arr[max(0, i):, max(0, -i):], 1)
    return 1 - arr


def compute_contact_map(structure, cm_thresh_val=14.0):    
    """
    Constructs a residue-residue contact map for the given protein structure.
    
    A contact is registered (1) if the Euclidean distance between two Calpha atoms
    is <= cm_thresh_val (default 14 Angstroms as a relaxed constraint). Trivial structural 
    sequence neighbors (separated by fewer than 3 positions) are filtered downstream.
    """
    def compute_dist_euc(coord1, coord2):
        return ((coord1[0]-coord2[0])**2 +
                (coord1[1]-coord2[1])**2 +
                (coord1[2]-coord2[2])**2)**0.5

    for chain in structure:
        CA_atoms = {}
        res_ids = set()
        cm_thresh = np.zeros((len(chain), len(chain)))

        # Extract coordinates specifically for Alpha Carbon (CA) atoms
        for residue in chain:
            res_id = int(residue.id[1])
            
            for atom in residue:
                if atom.get_id() == 'CA':
                    CA_atoms[res_id] = np.array(atom.get_coord())
                    res_ids.add(res_id)
                
        res_ids = list(res_ids)
        res_ids.sort()

        # Build pairwise distance contact mapping matrix
        for ri in range(len(res_ids)):
            r1 = res_ids[ri]                
            for rj in range(ri+1, len(res_ids)):
                r2 = res_ids[rj]

                euc_dist = compute_dist_euc(CA_atoms[r1], CA_atoms[r2])
                # Mark interaction status under the strict metric scale threshold
                cm_thresh[r1-1][r2-1] = 1 if euc_dist <= cm_thresh_val else 0
                cm_thresh[r2-1][r1-1] = cm_thresh[r1-1][r2-1]

    return cm_thresh


def compute_trajectory_similarity(traj1, traj2, cm_thresh_val=14.0):
    """
    Calculates similarity profiles and inter-frame variations between trajectories.
    
    1. Generates frame-by-frame raw contact matrices for trajectories 1 and 2.
    2. Drops trivial local sequence diagonals (separated by fewer than 3 positions).
    3. Calculates structural similarity matrix via normalized native-contact matches (Eqn. 2).
    4. Computes inter-frame Dice Distance matrix metrics to formulate custom heuristic        
       
    Returns:
       - sim_matrix: Cross-trajectory conformation similarities.
       - sim_diff_self: Dice distance variation matrix for Trajectory 1.
       - sim_diff_self2: Dice distance variation matrix for Trajectory 2.
    """
    cmpas1 = []
    cmpas2 = []
    
    # Generate spatial contact arrays per frame
    for i in tqdm(range(len(traj1)), desc="Contact Maps Traj 1"):
        cmpas1.append(compute_contact_map(traj1[i], cm_thresh_val=cm_thresh_val))
    for i in tqdm(range(len(traj2)-1,-1,-1), desc="Contact Maps Traj 2"):
        cmpas2.append(compute_contact_map(traj2[i], cm_thresh_val=cm_thresh_val))

    cmpas1 = np.array(cmpas1)
    cmpas2 = np.array(cmpas2)

    cmap_size = cmpas1.shape[-1]
    
    # Discard pairs separated by fewer than 3 sequence positions (width=4 handles -1, 0, 1)
    diag_array = diagonal_width_array(cmap_size, 4)
    cmpas1 = cmpas1 * diag_array
    cmpas2 = cmpas2 * diag_array

    # Flatten maps to safely perform vector dot products
    cm1 = cmpas1.reshape(cmpas1.shape[0], cmpas1.shape[1]*cmpas1.shape[2])
    cm2 = cmpas2.reshape(cmpas2.shape[0], cmpas2.shape[1]*cmpas2.shape[2])

    assert len(cm1) > len(cm2), "Trajectory 1 must be the ground truth trajectory and have more frames than Trajectory 2"

    # Intersection of shared active contact bits: commonality of contacting pairs (1s) only
    sim_matrix = cm1 @ cm2.T
    
    # Quantify normalization factors based on native structure configurations (Frame 0 reference)
    native1 = cm1 @ cm1[0].T
    native2 = cm2 @ cm1[0].T
    normalizer = np.maximum(np.expand_dims(native1, axis=-1), native2)
    sim_matrix = np.round(sim_matrix / np.maximum(normalizer, 1), 3)        

    # Computes Dice coefficients across consecutive frames to capture structural distance
    sim_matrix_self = np.dot(cm1, cm1.T) 
    self_sum = cm1.sum(axis=1, keepdims=True)
    denom = self_sum + self_sum.T
    dice_matrix = np.where(denom == 0, 1.0, (2 * sim_matrix_self) / (denom + 1e-8))
    sim_diff_self = dice_matrix - 1      # Formulated as negative cost for DP maximize matrix

    sim_matrix_self2 = np.dot(cm2, cm2.T) 
    self_sum2 = cm2.sum(axis=1, keepdims=True)
    denom2 = self_sum2 + self_sum2.T
    dice_matrix2 = np.where(denom2 == 0, 1.0, (2 * sim_matrix_self2) / (denom2 + 1e-8))
    sim_diff_self2 = dice_matrix2 - 1    # Formulated as negative cost for DP maximize matrix

    return sim_matrix, sim_diff_self, sim_diff_self2


def load_pdb_trajectory(pdb_path):
    """
    Loads a trajectory from a PDB file and returns it as a Bio.PDB Structure object.
    """

    parser = PDBParser()
    traj = parser.get_structure("ground truth", pdb_path)    

    return traj

'''
def load_predicted_frames(predicted_path):
    """
    Gathers, sorts, and tracks structural conformations produced by predictive models 
    like PathFold for comparison against target trajectories.
    """
    parser = PDBParser()
    traj_files = next(os.walk(predicted_path))[2]

    frames = []
    for i in range(len(traj_files)):        
        try:
            structure = parser.get_structure(str(i), f'{predicted_path}/{traj_files[i]}')

        except:
            print(f'Error in loading {predicted_path}/{traj_files[i]}, skipping this frame.')
            continue
        frames.append(structure[0])

    frames.reverse()
    return frames
'''



def load_predicted_frames(predicted_path):
    parser = PDBParser()
    traj_files = next(os.walk(predicted_path))[2]

    def frame_index(fname):
        m = re.search(r'(\d+)', fname)
        return int(m.group(1)) if m else -1

    traj_files.sort(key=frame_index)

    frames = []
    for i, fname in enumerate(traj_files):
        try:
            structure = parser.get_structure(str(i), f'{predicted_path}/{fname}')
        except Exception:
            print(f'Error in loading {predicted_path}/{fname}, skipping this frame.')
            continue
        frames.append(structure[0])

    frames.reverse()
    return frames

def global_alignment(S3, gap_open=-0, penalty_after=50):
    """
    Performs global alignment via Dynamic Programming
    
    Custom Gap Rules:
      - Small frame skips (<= penalty_after frames, i.e., 50 frames) cost nothing extra.
      - Extensions exceeding 50 frames substitute conventional linear penalties with 
        a structural distance penalty calculated via Dice distance values.
    """
    S, D_self, D_self2 = S3
    n, m = S.shape

    # Initialize dynamic programming score matrix and traceback matrix pointers
    dp = np.full((n + 1, m + 1), -np.inf)
    traceback = np.full((n + 1, m + 1), -1, dtype=np.int8)

    # Tracker metrics to log lengths of consecutive gap selections
    gap_run_x = np.zeros((n + 1, m + 1), dtype=int)  # Gap in trajectory 2 (Vertical move)
    gap_run_y = np.zeros((n + 1, m + 1), dtype=int)  # Gap in trajectory 1 (Horizontal move)

    dp[0, 0] = 0
    
    # Initialize edge conditions for the dynamic programming grid matrix 
    for i in range(1, n + 1):
        if i == 1:
            dp[i, 0] = gap_open
        elif i <= penalty_after:
            dp[i, 0] = gap_open  
        else:
            # Beyond 50 frames, utilize Dice Distance metric variations instead of a linear model
            dp[i, 0] = gap_open + D_self[0, i-1]  
        traceback[i, 0] = 1
        gap_run_x[i, 0] = i

    for j in range(1, m + 1):
        if j == 1:
            dp[0, j] = gap_open
        elif j <= penalty_after:
            dp[0, j] = gap_open  
        else:
            # Beyond 50 frames, utilize Dice Distance metric variations instead of a linear model
            dp[0, j] = gap_open + D_self2[0, j-1]  
        traceback[0, j] = 2
        gap_run_y[0, j] = j

    # Main Dynamic Programming score grid evaluation loop
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Option 1: Match/mismatch step conformation alignment (Diagonal movement)
            best_score = dp[i - 1, j - 1] + S[i - 1, j - 1]
            best_move = 0

            # Option 2: Vertical gap tracking (Gap insertion in Trajectory 2)
            run = gap_run_x[i - 1, j] + 1 if traceback[i - 1, j] == 1 else 1
            if run == 1:                
                score = dp[i - 1, j] + gap_open
            elif run <= penalty_after:
                score = dp[i - 1, j] 
            else:
                # Custom Gap Cost modification rule: inject structure distance across gap span
                score = dp[max(0, i-run), j] + D_self[max(0, i-run), i-1]   
                        
            if score > best_score:
                best_score, best_move = score, 1

            # Option 3: Horizontal gap tracking (Gap insertion in Trajectory 1)
            run = gap_run_y[i, j - 1] + 1 if traceback[i, j - 1] == 2 else 1
            if run == 1:
                score = dp[i, j - 1] + gap_open                
            elif run <= penalty_after:
                score = dp[i, j - 1] 
            else:
                # Custom Gap Cost modification rule: inject structure distance across gap span
                score = dp[i, max(0, j-run)] + D_self2[max(0, j-run), j-1]

            if score > best_score:
                best_score, best_move = score, 2

            # Assign optimum path decision values to arrays
            dp[i, j] = best_score
            traceback[i, j] = best_move

            # Propagate consecutive gap chain track scores downstream
            if best_move == 1:  
                gap_run_x[i, j] = gap_run_x[i - 1, j] + 1 if traceback[i - 1, j] == 1 else 1
                gap_run_y[i, j] = 0
            elif best_move == 2:  
                gap_run_y[i, j] = gap_run_y[i, j - 1] + 1 if traceback[i, j - 1] == 2 else 1
                gap_run_x[i, j] = 0
            else:  
                gap_run_x[i, j] = 0
                gap_run_y[i, j] = 0

    # Dynamic Programming Grid Traceback execution sequence
    i, j = n, m
    matched_indices = []
    while i > 0 or j > 0:
        if traceback[i, j] == 0:  
            matched_indices.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:  
            i -= 1
        elif traceback[i, j] == 2:  
            j -= 1

    matched_indices.reverse()

    # Normalizes alignment score based on total element counts within shorter trajectory
    normalized_score = round(dp[n, m] / (min(n, m)), 3)

    return matched_indices, normalized_score


def additional_penalties(traj_frames, sim_mat, thresh_fast, thresh_slow):
    """
    Implements Heuristic Trajectory Penalty Terms.
    
    Evaluates trajectory pacing consistency across consecutive predicted frames against the native reference:
      - Rapid structural transitions exceeding `thresh_fast` are penalized.
      - Excessively slow, un-progressing frame segments below `thresh_slow` are penalized.
      - Reversals in folding progress (negative step values) are penalized.
    """
    penalty_slow = 0
    penalty_fast = 0
    penalty_back = 0

    for i in range(len(traj_frames) - 1):
        # Difference in structural similarity relative to the native structure (sim_mat row 0)
        diff = sim_mat[0, traj_frames[i]] - sim_mat[0, traj_frames[i+1]]

        if diff > thresh_fast:
            # Fast transition tracking check rule
            penalty_fast += diff - thresh_fast  
        
        elif abs(diff) < thresh_slow:
            # Stagnant / excessively slow development loop frame checks
            penalty_slow += thresh_slow - abs(diff)  
        
        elif diff < 0:
            # Negative direction movement: reversal of trajectory progress path
            penalty_back += abs(diff)  

    return penalty_slow, penalty_fast, penalty_back


def generate_pdb_crop(exp_id, traj1, traj1_id, traj1_frames):
    """Constructs a subset Structure object containing only frames selected by the alignment."""
    structure1 = Structure.Structure(f"{exp_id}_{traj1_id}")

    for i in range(len(traj1)):
        model = Model.Model(i) 
        for chain in traj1[i]:
            model.add(chain)   

        if i in traj1_frames:
            structure1.add(model)
    
    return structure1


def align_and_combine_pymol(model1, model2, traj1_id, frame1_id, traj2_id, frame2_id, idx):
    """
    Uses PyMOL to dynamically align predicted models onto ground-truth coordinates, 
    then bundles them together into a unified multi-chain structural object layer.
    """
    rand_slt = str(np.random.randint(0, 10000))
    if not os.path.isdir('temp'):
        os.makedirs('temp')

    tmp1 = f"temp/model{traj1_id}_{frame1_id}_{rand_slt}.pdb"
    tmp2 = f"temp/model{traj2_id}_{frame2_id}_{rand_slt}.pdb"
    io = PDBIO()
    
    io.set_structure(model1)
    io.save(tmp1)
    io.set_structure(model2)
    io.save(tmp2)

    obj1 = f"obj1_{traj1_id}_{frame1_id}_{rand_slt}"
    obj2 = f"obj2_{traj2_id}_{frame2_id}_{rand_slt}"
    cmd.load(tmp1, obj1)
    cmd.load(tmp2, obj2)

    # Perform structural structural superimposition focusing on Calpha backbones
    safe_align(f"{obj2} and name CA", f"{obj1} and name CA")

    aligned2 = f"temp/aligned2_{traj1_id}{frame1_id}_{traj2_id}{frame2_id}_{rand_slt}.pdb"
    cmd.save(aligned2, obj2)

    parser = PDBParser(QUIET=True)
    model1_new = parser.get_structure("m1", tmp1)[0]
    model2_new = parser.get_structure("m2", aligned2)[0]

    chain1 = list(model1_new)[0]
    chain2 = list(model2_new)[0]
    chain1.id = "A"
    chain2.id = "B"

    new_model = Model.Model(idx)
    new_model.add(chain1.copy())
    new_model.add(chain2.copy())

    # Clean up PyMOL environment memory tracking nodes and workspace temporary files
    cmd.delete(obj1)
    cmd.delete(obj2)
    os.remove(tmp1)
    os.remove(tmp2)
    os.remove(aligned2)

    return new_model


def main(args):

    predicted_trajectory = args.predicted_trajectory_dir
    ground_truth = args.native_trajectory_file
    experiment_name = args.experiment_name

    if args.save_alignment_scores:
        if not os.path.isdir(args.alignment_scores_save_path):
            os.makedirs(args.alignment_scores_save_path)
        score_filename = f'{experiment_name}.tsv'
        with open(f'{args.alignment_scores_save_path}/{score_filename}', 'w') as fp:
            fp.write(f'Experiment: {experiment_name}\n')
            fp.write(f'Predicted Trajectory: {predicted_trajectory}\n')
            fp.write(f'Ground Truth Trajectory: {ground_truth}\n\n')
    
    traj_gt = load_pdb_trajectory(ground_truth)
    traj_pred = load_predicted_frames(predicted_trajectory)

    
    # Step 1: Compute pairwise similarity structures and dice distance across frames    
    sim_mat3 = compute_trajectory_similarity(traj_gt, traj_pred, cm_thresh_val=args.cm_thresh_val)
    sim_mat = sim_mat3[0]

    if args.save_sim_mat:
        if not os.path.isdir(args.sim_mat_save_path):
            os.makedirs(args.sim_mat_save_path)
        pickle.dump(sim_mat3, open(f'{args.sim_mat_save_path}/sim_mat_{experiment_name}.p', 'wb'))

    # Step 2: Extract frame alignment paths through Dynamic Programming
    matches, score = global_alignment(sim_mat3, gap_open=args.gap_open, penalty_after=args.penalty_after)


    aligned_out = ''
    traj1_frames = [matches[i][0] for i in range(len(matches))]
    traj2_frames = [matches[i][1] for i in range(len(matches))]

    # Step 3: Compute trajectory motion heuristics (Eqn 5)
    penalty_slow, penalty_fast, penalty_back = additional_penalties(traj2_frames, sim_mat, args.thresh_fast, args.thresh_slow)

    for i in range(len(matches)):
        if args.save_alignment_scores:
            with open(f'{args.alignment_scores_save_path}/{score_filename}', 'a') as fp:
                fp.write(f'(T1_{matches[i][0]})\t(T2_{matches[i][1]})\t{sim_mat[matches[i][0], matches[i][1]]}\n')
                aligned_out += f'(T1_{matches[i][0]})\t(T2_{matches[i][1]})\t{sim_mat[matches[i][0], matches[i][1]]}\n'

    if args.save_alignment_scores:
        with open(f'{args.alignment_scores_save_path}/{score_filename}', 'a') as fp:
            fp.write(f'Slow Penalty : {round(penalty_slow,3)}\nFast Penalty : {round(penalty_fast,3)}\nBackward Penalty : {round(penalty_back,3)}\n\n')
            aligned_out += f'Slow Penalty : {round(penalty_slow,3)}\nFast Penalty : {round(penalty_fast,3)}\nBackward Penalty : {round(penalty_back,3)}\n\n'

        with open(f'{args.alignment_scores_save_path}/{score_filename}', 'a') as fp:
            fp.write(f'Alignment Score : {score}\n')

    print('Alignment Score', score)


    # final score
    score = score - (penalty_slow * args.penalty_slow_weight + penalty_fast * args.penalty_fast_weight + penalty_back * args.penalty_back_weight)

    if args.save_alignment_scores:
        with open(f'{args.alignment_scores_save_path}/{score_filename}', 'a') as fp:
            fp.write(f'Final Score : {round(score,3)}\n')
            aligned_out += f'Final Score : {round(score,3)}\n\n'

    if args.save_aligned_trajectories:
        if not os.path.isdir(args.aligned_trajectories_save_path):
            os.makedirs(args.aligned_trajectories_save_path)
        pickle.dump((aligned_out, score), open(f'{args.aligned_trajectories_save_path}/aligned_{experiment_name}.p', 'wb'))

    if args.save_pdb_trajectories:
        if not os.path.isdir(args.pdb_trajectories_save_path):
            os.makedirs(args.pdb_trajectories_save_path)

        structure1 = generate_pdb_crop(experiment_name, traj_gt, 'gt', traj1_frames)
        structure2 = generate_pdb_crop(experiment_name, traj_pred, 'pred', traj2_frames)

        save_pdb_file(structure1, f'{args.pdb_trajectories_save_path}/{experiment_name}_gt.pdb')
        save_pdb_file(structure2, f'{args.pdb_trajectories_save_path}/{experiment_name}_pred.pdb')

        algnd_trjctr = Structure.Structure(f"{experiment_name}_gt_pred_aligned")

        for i in range(len(matches)):            
            model1 = structure1[matches[i][0]]
            model2 = structure2[matches[i][1]]
            algn_strct = align_and_combine_pymol(model1, model2, 'gt', matches[i][0], 'pred', matches[i][1], i+1)
            algnd_trjctr.add(algn_strct)

        save_pdb_file(algnd_trjctr, f'{args.pdb_trajectories_save_path}/{experiment_name}_gt_pred_aligned.pdb')
    

def arg_parse():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--predicted_trajectory_dir', type=str, required=True, help='Directory path to the predicted trajectory PDB files')
    parser.add_argument('--native_trajectory_file', type=str, required=True, help='File path to the ground-truth Molecular Dynamics trajectory PDB file')

    parser.add_argument('--experiment_name', type=str, required=True, default='example', help='Experiment name or identifier for logging purposes')

    parser.add_argument('--cm_thresh_val', type=float, default=14, help='Contact map distance threshold (Angstroms)')
    
    # Gap configuration thresholds
    parser.add_argument('--gap_open', type=float, default=-0, help='Gap opening penalty')
    parser.add_argument('--penalty_after', type=float, default=50, help='Number of gaps before extension penalty applies')
    
    
    # Transition rate heuristic boundaries (Per parameters in Eqn 5)
    parser.add_argument('--thresh_fast', type=float, default=0.2, help='Threshold for fast transitions')
    parser.add_argument('--penalty_fast_weight', type=float, default=0.1, help='Weight for fast transition penalty')
    parser.add_argument('--thresh_slow', type=float, default=0.01, help='Threshold for slow transitions')
    parser.add_argument('--penalty_slow_weight', type=float, default=0.1, help='Weight for slow transition penalty')
    parser.add_argument('--penalty_back_weight', type=float, default=0.1, help='Weight for backward transition penalty')
    
    parser.add_argument('--save_sim_mat', action='store_true', default=False, help='Flag to save similarity matrix')
    parser.add_argument('--sim_mat_save_path', type=str, default='sim_mat', help='Path to save similarity matrices')
    parser.add_argument('--save_alignment_scores', action='store_true', default=True, help='Flag to save alignment scores')
    parser.add_argument('--alignment_scores_save_path', type=str, default='alignment_scores', help='Path to save alignment scores')
    parser.add_argument('--save_aligned_trajectories', action='store_true', default=True, help='Flag to save aligned trajectories')
    parser.add_argument('--aligned_trajectories_save_path', type=str, default='aligned_trajectories', help='Path to save aligned trajectories')
    parser.add_argument('--save_pdb_trajectories', action='store_true', default=False, help='Flag to save PDB trajectories')
    parser.add_argument('--pdb_trajectories_save_path', type=str, default='pdb_tajectories', help='Path to save PDB trajectories')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)