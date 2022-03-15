import os

import numpy
import numpy as np
# from numba import njit

from PathUtils import frag_text_file, docked_pose_file
from SymDesignUtils import start_log

# Globals
logger = start_log(name=__name__)


# @njit
def transform_coordinates(coords, rotation=None, translation=None, rotation2=None, translation2=None):
    """Take a set of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the order of
    operations as: rotation, translation, rotation2, translation2

    Args:
        coords (union[numpy.ndarray,list]): The coordinates to transform, can be shape (number of coordinates, 3)
    Keyword Args:
        rotation=None (numpy.ndarray): The first rotation to apply, expected general rotation matrix shape (3, 3)
        translation=None (numpy.ndarray): The first translation to apply, expected shape (3)
        rotation2=None (numpy.ndarray): The second rotation to apply, expected general rotation matrix shape (3, 3)
        translation2=None (numpy.ndarray): The second translation to apply, expected shape (3)
    Returns:
        (numpy.ndarray): The transformed coordinate set with the same shape as the original
    """
    if rotation is not None:
        coords = np.matmul(coords, np.transpose(rotation))

    if translation is not None:
        coords += translation

    if rotation2 is not None:
        coords = np.matmul(coords, np.transpose(rotation2))

    if translation2 is not None:
        coords += translation2

    return coords


# @njit
def transform_coordinate_sets(coord_sets, rotation=None, translation=None, rotation2=None, translation2=None) \
        -> numpy.ndarray:
    """Take multiple sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2

    Args:
        coord_sets (union[numpy.ndarray,list]): The coordinates to transform, can be shape (number of coordinates, 3, 3)
    Keyword Args:
        rotation=None (numpy.ndarray): The first rotation to apply, expected general rotation matrix shape (3, 3)
        translation=None (numpy.ndarray): The first translation to apply, expected shape (3)
        rotation2=None (numpy.ndarray): The second rotation to apply, expected general rotation matrix shape (3, 3)
        translation2=None (numpy.ndarray): The second translation to apply, expected shape (3)
    Returns:
        (numpy.ndarray): The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_length = getattr(coord_sets, 'shape', None)
    if not set_length or set_length[0] < 1:
        return coord_sets

    if rotation is not None:
        # coord_sets = np.tensordot(coord_sets, np.transpose(rot_mat), axes=1)
        coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))

    if translation is not None:
        coord_sets += translation

    if rotation2 is not None:
        # coord_sets = np.tensordot(coord_sets, np.transpose(set_mat), axes=1)
        coord_sets = np.matmul(coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        coord_sets += translation2

    return coord_sets


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
                               central_frequencies=None, out_path=os.getcwd(), pose_id=None):  # , is_initial_match=False):

    # if not ghost_frag and not matched_frag and not overlap_error and not match_number:  # TODO
    #     raise DesignError('%s: Missing required information for writing!' % write_frag_match_info_file.__name__)

    with open(os.path.join(out_path, frag_text_file), "a+") as out_info_file:
        # if is_initial_match:
        if match_number == 1:
            out_info_file.write("DOCKED POSE ID: %s\n\n" % pose_id)
            out_info_file.write("***** ALL FRAGMENT MATCHES *****\n\n")
            # out_info_file.write("***** INITIAL MATCH FROM REPRESENTATIVES OF INITIAL FRAGMENT CLUSTERS *****\n\n")
        cluster_id = 'i%d_j%d_k%d' % ghost_frag.get_ijk()
        out_info_file.write("MATCH %d\n" % match_number)
        out_info_file.write("z-val: %f\n" % overlap_error)
        out_info_file.write("CENTRAL RESIDUES\n")
        out_info_file.write("oligomer1 ch, resnum: %s, %d\n" % ghost_frag.get_aligned_chain_and_residue())
        out_info_file.write("oligomer2 ch, resnum: %s, %d\n" % matched_frag.get_central_res_tup())
        out_info_file.write("FRAGMENT CLUSTER\n")
        out_info_file.write('id: %s\n' % cluster_id)
        out_info_file.write("mean rmsd: %f\n" % ghost_frag.get_rmsd())
        out_info_file.write("aligned rep: int_frag_%s_%d.pdb\n" % (cluster_id, match_number))
        out_info_file.write("central res pair freqs:\n%s\n\n" % str(central_frequencies))

        # if is_initial_match:
        #     out_info_file.write("***** ALL MATCH(ES) FROM REPRESENTATIVES OF ALL FRAGMENT CLUSTERS *****\n\n")


def write_docked_pose_info(outdir_path, res_lev_sum_score, high_qual_match_count,
                           unique_matched_interface_monofrag_count, unique_total_interface_monofrags_count,
                           percent_of_interface_covered, rot_mat1, representative_int_dof_tx_param_1, set_mat1,
                           representative_ext_dof_tx_params_1, rot_mat2, representative_int_dof_tx_param_2, set_mat2,
                           representative_ext_dof_tx_params_2, cryst1_record, pdb1_path, pdb2_path, pose_id):

    out_info_file_path = os.path.join(outdir_path, docked_pose_file)
    with open(out_info_file_path, 'w') as out_info_file:
        out_info_file.write("DOCKED POSE ID: %s\n\n" % pose_id)
        out_info_file.write("Nanohedra Score: %f\n\n" % res_lev_sum_score)
        out_info_file.write("Unique Mono Fragments Matched (z<=1): %d\n" % high_qual_match_count)
        out_info_file.write("Unique Mono Fragments Matched: %d\n" % unique_matched_interface_monofrag_count)
        out_info_file.write("Unique Mono Fragments at Interface: %d\n" % unique_total_interface_monofrags_count)
        out_info_file.write("Interface Matched (%s): %f\n\n" % ("%", percent_of_interface_covered * 100))

        out_info_file.write("ROT/DEGEN MATRIX PDB1: %s\n" % str(rot_mat1.tolist()))
        if representative_int_dof_tx_param_1 is not None:
            int_dof_tx_vec_1 = representative_int_dof_tx_param_1
        else:
            int_dof_tx_vec_1 = None
        out_info_file.write("INTERNAL Tx PDB1: %s\n" % str(int_dof_tx_vec_1))
        out_info_file.write("SETTING MATRIX PDB1: %s\n" % str(set_mat1.tolist()))
        if representative_ext_dof_tx_params_1 == [0, 0, 0]:
            ref_frame_tx_vec_1 = None
        else:
            ref_frame_tx_vec_1 = representative_ext_dof_tx_params_1
        out_info_file.write("REFERENCE FRAME Tx PDB1: %s\n\n" % str(ref_frame_tx_vec_1))

        out_info_file.write("ROT/DEGEN MATRIX PDB2: %s\n" % str(rot_mat2.tolist()))
        if representative_int_dof_tx_param_2 is not None:
            int_dof_tx_vec_2 = representative_int_dof_tx_param_2
        else:
            int_dof_tx_vec_2 = None
        out_info_file.write("INTERNAL Tx PDB2: %s\n" % str(int_dof_tx_vec_2))
        out_info_file.write("SETTING MATRIX PDB2: %s\n" % str(set_mat2.tolist()))
        if representative_ext_dof_tx_params_2 == [0, 0, 0]:
            ref_frame_tx_vec_2 = None
        else:
            ref_frame_tx_vec_2 = representative_ext_dof_tx_params_2
        out_info_file.write("REFERENCE FRAME Tx PDB2: %s\n\n" % str(ref_frame_tx_vec_2))

        out_info_file.write("CRYST1 RECORD: %s\n\n" % cryst1_record)

        out_info_file.write('Canonical Orientation PDB1 Path: %s\n' % pdb1_path)
        out_info_file.write('Canonical Orientation PDB2 Path: %s\n\n' % pdb2_path)


def get_rotation_step(sym_entry, rot_step_deg1=None, rot_step_deg2=None, initial=False, log=None):
    """Set up the rotation step from the input arguments

    Returns:
        (tuple[int, int]): The rotational sampling steps for oligomer1 and oligomer2
    """
    if sym_entry.is_internal_rot1:  # if rotation step required
        if not rot_step_deg1:
            rot_step_deg1 = 3  # set rotation step to default
    else:
        if rot_step_deg1 and initial:
            log.warning('Specified Rotation Step 1 Was Ignored. Oligomer 1 Doesn\'t Have Internal Rotational DOF\n\n')
        rot_step_deg1 = 1

    if sym_entry.is_internal_rot2:  # if rotation step required
        if not rot_step_deg2:
            rot_step_deg2 = 3  # set rotation step to default
    else:
        if rot_step_deg2 and initial:
            log.warning('Specified Rotation Step 2 Was Ignored. Oligomer 2 Doesn\'t Have Internal Rotational DOF\n\n')
        rot_step_deg2 = 1

    return rot_step_deg1, rot_step_deg2


def write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir, log=logger):
    log.info('NANOHEDRA PROJECT INFORMATION\n')
    log.info('Oligomer 1 Input Directory: %s\n' % pdb1_path)
    log.info('Oligomer 2 Input Directory: %s\n' % pdb2_path)
    log.info('Master Output Directory: %s\n\n' % master_outdir)
    log.info('SYMMETRY COMBINATION MATERIAL INFORMATION\n')
    log.info('Nanohedra Entry Number: %d\n' % sym_entry.entry_number)
    log.info('Oligomer 1 Point Group Symmetry: %s\n' % sym_entry.group1)
    log.info('Oligomer 2 Point Group Symmetry: %s\n' % sym_entry.group2)
    log.info('SCM Point Group Symmetry: %s\n' % sym_entry.point_group_sym)
    # log.info("Oligomer 1 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot1()))
    # log.info("Oligomer 2 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot2()))
    # log.info("Oligomer 1 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx1()))
    # log.info("Oligomer 2 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx2()))
    log.info('Oligomer 1 Setting Matrix: %s\n' % sym_entry.setting_matrix1.tolist())
    log.info('Oligomer 2 Setting Matrix: %s\n' % sym_entry.setting_matrix2.tolist())
    log.info('Oligomer 1 Reference Frame Tx DOF: %s\n' % (sym_entry.ref_frame_tx_dof1
                                                          if sym_entry.is_ref_frame_tx_dof1 else str(None)))
    log.info('Oligomer 2 Reference Frame Tx DOF: %s\n' % (sym_entry.ref_frame_tx_dof2
                                                          if sym_entry.is_ref_frame_tx_dof2 else str(None)))
    log.info('Resulting SCM Symmetry: %s\n' % sym_entry.resulting_symmetry)
    log.info('SCM Dimension: %d\n' % sym_entry.dimension)
    log.info('SCM Unit Cell Specification: %s\n\n' % sym_entry.uc_specification)
    rot_step_deg1, rot_step_deg2 = get_rotation_step(sym_entry, rot_step_deg1, rot_step_deg2, initial=True,
                                                     log=log)
    # # Default Rotation Step
    # if sym_entry.is_internal_rot1:  # if rotation step required
    #     if not rot_step_deg1:
    #         rot_step_deg_pdb1 = 3  # set rotation step to default
    # else:
    #     rot_step_deg_pdb1 = 1
    #     if rot_step_deg_pdb1:
    #         log.info("Warning: Specified Rotation Step 1 Was Ignored. Oligomer 1 Doesn\'t Have"
    #                               " Internal Rotational DOF\n\n")
    # if sym_entry.is_internal_rot2:  # if rotation step required
    #     if not rot_step_deg2:
    #         rot_step_deg_pdb2 = 3  # set rotation step to default
    # else:
    #     rot_step_deg_pdb2 = 1
    #     if rot_step_deg_pdb2:
    #         log.info("Warning: Specified Rotation Step 2 Was Ignored. Oligomer 2 Doesn\'t Have"
    #                               " Internal Rotational DOF\n\n")
    log.info('ROTATIONAL SAMPLING INFORMATION\n')
    log.info('Oligomer 1 ROT Sampling Range: %s\n' % (str(sym_entry.rotation_range1)
                                                       if sym_entry.is_internal_rot1 else str(None)))
    log.info('Oligomer 2 ROT Sampling Range: %s\n' % (str(sym_entry.rotation_range2)
                                                      if sym_entry.is_internal_rot2 else str(None)))
    log.info('Oligomer 1 ROT Sampling Step: %s\n' % (str(rot_step_deg1) if sym_entry.is_internal_rot1
                                                     else str(None)))
    log.info('Oligomer 2 ROT Sampling Step: %s\n\n' % (str(rot_step_deg2) if sym_entry.is_internal_rot2
                                                       else str(None)))
    # Get Degeneracy Matrices
    log.info('Searching For Possible Degeneracies\n')
    if sym_entry.degeneracy_matrices_1 is None:
        log.info('No Degeneracies Found for Oligomer 1\n')
    elif len(sym_entry.degeneracy_matrices_1) == 1:
        log.info('1 Degeneracy Found for Oligomer 1\n')
    else:
        log.info('%d Degeneracies Found for Oligomer 1\n' % len(sym_entry.degeneracy_matrices_1))
    if sym_entry.degeneracy_matrices_2 is None:
        log.info('No Degeneracies Found for Oligomer 2\n\n')
    elif len(sym_entry.degeneracy_matrices_2) == 1:
        log.info('1 Degeneracy Found for Oligomer 2\n\n')
    else:
        log.info('%d Degeneracies Found for Oligomer 2\n\n' % len(sym_entry.degeneracy_matrices_2))
    log.info('Retrieving Database of Complete Interface Fragment Cluster Representatives\n')
