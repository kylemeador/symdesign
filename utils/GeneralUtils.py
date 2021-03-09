import os

import numpy as np

# from SymDesignUtils import DesignError
from PathUtils import frag_text_file


def euclidean_squared_3d(coordinates_1, coordinates_2):
    if len(coordinates_1) != 3 or len(coordinates_2) != 3:
        raise ValueError("len(coordinate list) != 3")

    # KM removed as a tuple would suffice
    # elif type(coordinates_1) is not list or type(coordinates_2) is not list:
    #     raise TypeError("input parameters are not of type list")

    else:
        x1, y1, z1 = coordinates_1[0], coordinates_1[1], coordinates_1[2]
        x2, y2, z2 = coordinates_2[0], coordinates_2[1], coordinates_2[2]
        return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2


def center_of_mass_3d(coordinates):
    n = len(coordinates)
    if n != 0:
        cm = [0. for j in range(3)]
        for i in range(n):
            for j in range(3):
                cm[j] = cm[j] + coordinates[i][j]
        for j in range(3):
            cm[j] = cm[j] / n
        return cm
    else:
        print("ERROR CALCULATING CENTER OF MASS")
        return None


def rot_txint_set_txext_frag_coord_sets(coord_sets, rot_mat=None, internal_tx_vec=None, set_mat=None, ext_tx_vec=None):
    if not coord_sets:
        return []

    # Get the length of each coordinate set
    coord_set_lens = []
    for coord_set in coord_sets:
        coord_set_lens.append(len(coord_set))

    # Stack coordinate set arrays in sequence vertically (row wise)
    coord_sets_vstacked = np.vstack(coord_sets)
    # in general, the np.tensordot module accomplishes this same problem without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul may solve as well... due to broadcasting

    # Rotate stacked coordinates if rotation matrix is provided
    if rot_mat is not None:
        coord_sets_vstacked = np.matmul(coord_sets_vstacked, np.transpose(rot_mat))
        # coord_sets_vstacked = np.tensordot(coord_sets, np.transpose(rot_mat), axes=1)

    # Translate stacked coordinates if internal translation vector is provided
    if internal_tx_vec is not None:
        coord_sets_vstacked = coord_sets_vstacked + internal_tx_vec
        # coord_sets = coord_sets + internal_tx_vec

    # Set stacked coordinates if setting matrix is provided
    if set_mat is not None:
        coord_sets_vstacked = np.matmul(coord_sets_vstacked, np.transpose(set_mat))
        # coord_sets_vstacked = np.tensordot(coord_sets, np.transpose(set_mat), axes=1)

    # Translate stacked coordinates if external translation vector is provided
    if ext_tx_vec is not None:
        coord_sets_vstacked = coord_sets_vstacked + ext_tx_vec
        # coord_sets = coord_sets + ext_tx_vec

    # Slice stacked coordinates back into coordinate sets
    transformed_coord_sets = []
    slice_index_1 = 0
    for coord_set_len in coord_set_lens:
        slice_index_2 = slice_index_1 + coord_set_len

        transformed_coord_sets.append(coord_sets_vstacked[slice_index_1:slice_index_2].tolist())

        slice_index_1 += coord_set_len

    return transformed_coord_sets


def get_last_sampling_state(log_file_path, zero=True):
    """Returns the (zero-indexed) last output state specified in the building_blocks_log.txt file. To return the
    one-indexed sampling state, pass zero=False"""
    degen_1, degen_2, rot_1, rot_2, index = 0, 0, 0, 0, 0
    if zero:
        index = 1

    with open(log_file_path, 'r') as log_f:
        log_lines = log_f.readlines()
        for line in reversed(log_lines):
            # ***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****
            if line.startswith('*****'):
                last_state = line.strip().strip('*').split('|')
                last_state = list(map(str.split, last_state))
                degen_1 = int(last_state[0][-3]) - index
                rot_1 = int(last_state[0][-1]) - index
                degen_2 = int(last_state[1][-3]) - index
                rot_2 = int(last_state[1][-1]) - index
                break

    return degen_1, degen_2, rot_1, rot_2


def write_frag_match_info_file(ghost_frag=None, matched_frag=None, overlap_error=None, match_number=None,
                               central_frequencies=None, out_path=os.getcwd(), pose_id=None, is_initial_match=False):

    # if not ghost_frag and not matched_frag and not overlap_error and not match_number:  # TODO
    #     raise DesignError('%s: Missing required information for writing!' % write_frag_match_info_file.__name__)

    with open(os.path.join(out_path, frag_text_file), "a+") as out_info_file:
        if is_initial_match:
            out_info_file.write("DOCKED POSE ID: %s\n\n" % pose_id)
            out_info_file.write("***** INITIAL MATCH FROM REPRESENTATIVES OF INITIAL FRAGMENT CLUSTERS *****\n\n")

        out_info_file.write("MATCH %d\n" % match_number)
        out_info_file.write("z-val: %f\n" % overlap_error)
        out_info_file.write("CENTRAL RESIDUES\n")
        out_info_file.write("oligomer1 ch, resnum: %s, %d\n" % ghost_frag.get_aligned_chain_and_residue())
        out_info_file.write("oligomer2 ch, resnum: %s, %d\n" % matched_frag.get_central_res_tup())
        out_info_file.write("FRAGMENT CLUSTER\n")
        out_info_file.write("id: i%s_j%s_k%s\n" % ghost_frag.get_ijk())
        out_info_file.write("mean rmsd: %f\n" % ghost_frag.get_rmsd())
        out_info_file.write("aligned rep: int_frag_%s_%d.pdb\n" % ('i%s_j%s_k%s' % ghost_frag.get_ijk(), match_number))
        out_info_file.write("central res pair freqs:\n%s\n\n" % str(central_frequencies))

        if is_initial_match:
            out_info_file.write("***** ALL MATCH(ES) FROM REPRESENTATIVES OF ALL FRAGMENT CLUSTERS *****\n\n")


def write_docked_pose_info(outdir_path, res_lev_sum_score, high_qual_match_count,
                           unique_matched_interface_monofrag_count, unique_total_interface_monofrags_count,
                           percent_of_interface_covered, rot_mat1, representative_int_dof_tx_param_1, set_mat1,
                           representative_ext_dof_tx_params_1, rot_mat2, representative_int_dof_tx_param_2, set_mat2,
                           representative_ext_dof_tx_params_2, cryst1_record, pdb1_path, pdb2_path, pose_id):

    out_info_file_path = os.path.join(outdir_path, "docked_pose_info_file.txt")
    with open(out_info_file_path, "w") as out_info_file:
        out_info_file.write("DOCKED POSE ID: %s\n\n" % pose_id)
        out_info_file.write("Nanohedra Score: %s\n\n" % str(res_lev_sum_score))
        out_info_file.write("Unique Mono Fragments Matched (z<=1): %s\n" % str(high_qual_match_count))
        out_info_file.write("Unique Mono Fragments Matched: %s\n" % str(unique_matched_interface_monofrag_count))
        out_info_file.write("Unique Mono Fragments at Interface: %s\n" % str(unique_total_interface_monofrags_count))
        out_info_file.write("Interface Matched (%s): %s\n\n" % ("%", str(percent_of_interface_covered * 100)))

        out_info_file.write("ROT/DEGEN MATRIX PDB1: %s\n" % str(rot_mat1))
        if representative_int_dof_tx_param_1 is not None:
            int_dof_tx_vec_1 = representative_int_dof_tx_param_1
        else:
            int_dof_tx_vec_1 = None
        out_info_file.write("INTERNAL Tx PDB1: %s\n" % str(int_dof_tx_vec_1))
        out_info_file.write("SETTING MATRIX PDB1: %s\n" % str(set_mat1))
        if representative_ext_dof_tx_params_1 == [0, 0, 0]:
            ref_frame_tx_vec_1 = None
        else:
            ref_frame_tx_vec_1 = representative_ext_dof_tx_params_1
        out_info_file.write("REFERENCE FRAME Tx PDB1: %s\n\n" % str(ref_frame_tx_vec_1))

        out_info_file.write("ROT/DEGEN MATRIX PDB2: %s\n" % str(rot_mat2))
        if representative_int_dof_tx_param_2 is not None:
            int_dof_tx_vec_2 = representative_int_dof_tx_param_2
        else:
            int_dof_tx_vec_2 = None
        out_info_file.write("INTERNAL Tx PDB2: %s\n" % str(int_dof_tx_vec_2))
        out_info_file.write("SETTING MATRIX PDB2: %s\n" % str(set_mat2))
        if representative_ext_dof_tx_params_2 == [0, 0, 0]:
            ref_frame_tx_vec_2 = None
        else:
            ref_frame_tx_vec_2 = representative_ext_dof_tx_params_2
        out_info_file.write("REFERENCE FRAME Tx PDB2: %s\n\n" % str(ref_frame_tx_vec_2))

        out_info_file.write("CRYST1 RECORD: %s\n\n" % str(cryst1_record))

        out_info_file.write('Canonical Orientation PDB1 Path: %s\n' % pdb1_path)
        out_info_file.write('Canonical Orientation PDB2 Path: %s\n\n' % pdb2_path)
