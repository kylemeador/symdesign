from __future__ import annotations

import os
from logging import Logger
from typing import Iterable

import numpy as np
# from numba import njit

from classes.SymEntry import SymEntry
from PathUtils import docked_pose_file
from SymDesignUtils import start_log

# Globals
logger = start_log(name=__name__)
number_of_nanohedra_components = 2


# @njit
def transform_coordinates(coords: np.ndarray | Iterable, rotation: np.ndarray | Iterable = None,
                          translation: np.ndarray | Iterable | int | float = None,
                          rotation2: np.ndarray | Iterable = None,
                          translation2: np.ndarray | Iterable | int | float = None) -> np.ndarray:
    """Take a set of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the order of
    operations as: rotation, translation, rotation2, translation2

    Args:
        coords: The coordinates to transform, can be shape (number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (3, 3)
        translation: The first translation to apply, expected shape (3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (3, 3)
        translation2: The second translation to apply, expected shape (3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    new_coords = coords.copy()

    if rotation is not None:
        np.matmul(new_coords, np.transpose(rotation), out=new_coords)

    if translation is not None:
        new_coords += translation  # No array allocation, sets in place

    if rotation2 is not None:
        np.matmul(new_coords, np.transpose(rotation2), out=new_coords)

    if translation2 is not None:
        new_coords += translation2

    return coords


# @njit
def transform_coordinate_sets_with_broadcast(coord_sets: np.ndarray,
                                             rotation: np.ndarray = None,
                                             translation: np.ndarray | Iterable | int | float = None,
                                             rotation2: np.ndarray = None,
                                             translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. Non-efficient memory use

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets
    # else:  # Create a new array for the result
    #     new_coord_sets = coord_sets.copy()

    if rotation is not None:
        coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))

    if translation is not None:
        coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        coord_sets = np.matmul(coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        coord_sets += translation2

    return coord_sets


def transform_coordinate_setsWORKED(coord_sets: np.ndarray,
                              rotation: np.ndarray = None,
                              translation: np.ndarray | Iterable | int | float = None,
                              rotation2: np.ndarray = None,
                              translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. Non-efficient memory use

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets
    # else:  # Create a new array for the result
    #     new_coord_sets = coord_sets.copy()

    if rotation is not None:
        coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))

    if translation is not None:
        coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        coord_sets = np.matmul(coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        coord_sets += translation2

    return coord_sets


# @njit
def transform_coordinate_sets(coord_sets: np.ndarray,
                              rotation: np.ndarray = None, translation: np.ndarray | Iterable | int | float = None,
                              rotation2: np.ndarray = None, translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. If transformation uses broadcasting, for
    efficient memory use, the returned array will be the size of the coord_sets multiplied by rotation. Additional
    broadcasting is not allowed. If that behavior is desired, use "transform_coordinate_sets_with_broadcast()" instead

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets

    if rotation is not None:
        new_coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))
    else:  # Create a new array for the result
        new_coord_sets = coord_sets.copy()

    if translation is not None:
        new_coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        # np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1), out=new_coord_sets)
        new_coord_sets[:] = np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        new_coord_sets += translation2

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


def write_docked_pose_info(outdir_path, res_lev_sum_score, high_qual_match_count,
                           unique_matched_interface_monofrag_count, unique_total_interface_monofrags_count,
                           percent_of_interface_covered, rot_mat1, representative_int_dof_tx_param_1, set_mat1,
                           representative_ext_dof_tx_params_1, rot_mat2, representative_int_dof_tx_param_2, set_mat2,
                           representative_ext_dof_tx_params_2, cryst1_record, pdb1_path, pdb2_path, pose_id):

    out_info_file_path = os.path.join(outdir_path, docked_pose_file)
    with open(out_info_file_path, 'w') as out_info_file:
        out_info_file.write('DOCKED POSE ID: %s\n\n' % pose_id)
        out_info_file.write('Nanohedra Score: %f\n\n' % res_lev_sum_score)
        out_info_file.write('Unique Mono Fragments Matched (z<=1): %d\n' % high_qual_match_count)
        out_info_file.write('Unique Mono Fragments Matched: %d\n' % unique_matched_interface_monofrag_count)
        out_info_file.write('Unique Mono Fragments at Interface: %d\n' % unique_total_interface_monofrags_count)
        out_info_file.write('Interface Matched (%s): %f\n\n' % ('%', percent_of_interface_covered * 100))

        out_info_file.write('ROT/DEGEN MATRIX PDB1: %s\n' % str(rot_mat1.tolist()))
        if representative_int_dof_tx_param_1 is not None:
            int_dof_tx_vec_1 = representative_int_dof_tx_param_1
        else:
            int_dof_tx_vec_1 = None
        out_info_file.write('INTERNAL Tx PDB1: %s\n' % str(int_dof_tx_vec_1))
        out_info_file.write('SETTING MATRIX PDB1: %s\n' % str(set_mat1.tolist()))
        # if representative_ext_dof_tx_params_1 == [0, 0, 0]:
        if representative_ext_dof_tx_params_1 is not None:
            ref_frame_tx_vec_1 = representative_ext_dof_tx_params_1
        else:
            ref_frame_tx_vec_1 = None
        out_info_file.write('REFERENCE FRAME Tx PDB1: %s\n\n' % str(ref_frame_tx_vec_1))

        out_info_file.write('ROT/DEGEN MATRIX PDB2: %s\n' % str(rot_mat2.tolist()))
        if representative_int_dof_tx_param_2 is not None:
            int_dof_tx_vec_2 = representative_int_dof_tx_param_2
        else:
            int_dof_tx_vec_2 = None
        out_info_file.write('INTERNAL Tx PDB2: %s\n' % str(int_dof_tx_vec_2))
        out_info_file.write('SETTING MATRIX PDB2: %s\n' % str(set_mat2.tolist()))
        # if representative_ext_dof_tx_params_2 == [0, 0, 0]:
        if representative_ext_dof_tx_params_2 is not None:
            ref_frame_tx_vec_2 = representative_ext_dof_tx_params_2
        else:
            ref_frame_tx_vec_2 = None
        out_info_file.write('REFERENCE FRAME Tx PDB2: %s\n\n' % str(ref_frame_tx_vec_2))

        out_info_file.write('CRYST1 RECORD: %s\n\n' % cryst1_record)

        out_info_file.write('Canonical Orientation PDB1 Path: %s\n' % pdb1_path)
        out_info_file.write('Canonical Orientation PDB2 Path: %s\n\n' % pdb2_path)


def get_rotation_step(sym_entry: SymEntry, rot_step_deg1: float | int = None, rot_step_deg2: float | int = None,
                      initial: bool = False, log: Logger = None) -> tuple[int, int]:
    """Set up the rotation step from the input arguments

    Returns:
        The rotational sampling steps for oligomer1 and oligomer2
    """
    if sym_entry.is_internal_rot1:  # if rotation step required
        if rot_step_deg1 is None:
            rot_step_deg1 = 3  # set rotation step to default
    else:
        if rot_step_deg1 and initial:
            log.warning('Specified Rotation Step 1 Was Ignored. Oligomer 1 Doesn\'t Have Internal Rotational DOF\n')
        rot_step_deg1 = 1

    if sym_entry.is_internal_rot2:  # if rotation step required
        if rot_step_deg2 is None:
            rot_step_deg2 = 3  # set rotation step to default
    else:
        if rot_step_deg2 and initial:
            log.warning('Specified Rotation Step 2 Was Ignored. Oligomer 2 Doesn\'t Have Internal Rotational DOF\n')
        rot_step_deg2 = 1

    return rot_step_deg1, rot_step_deg2


def write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir, log=logger):
    log.info('NANOHEDRA PROJECT INFORMATION')
    log.info('Oligomer 1 Input Directory: %s' % pdb1_path)
    log.info('Oligomer 2 Input Directory: %s' % pdb2_path)
    log.info('Master Output Directory: %s\n' % master_outdir)
    log.info('SYMMETRY COMBINATION MATERIAL INFORMATION')
    log.info('Nanohedra Entry Number: %d' % sym_entry.entry_number)
    log.info('Oligomer 1 Point Group Symmetry: %s' % sym_entry.group1)
    log.info('Oligomer 2 Point Group Symmetry: %s' % sym_entry.group2)
    log.info('SCM Point Group Symmetry: %s' % sym_entry.point_group_symmetry)
    # log.info("Oligomer 1 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot1()))
    # log.info("Oligomer 2 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot2()))
    # log.info("Oligomer 1 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx1()))
    # log.info("Oligomer 2 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx2()))
    log.info('Oligomer 1 Setting Matrix: %s' % sym_entry.setting_matrix1.tolist())
    log.info('Oligomer 2 Setting Matrix: %s' % sym_entry.setting_matrix2.tolist())
    log.info('Oligomer 1 Reference Frame Tx DOF: %s' % (sym_entry.ref_frame_tx_dof1
                                                        if sym_entry.is_ref_frame_tx_dof1 else str(None)))
    log.info('Oligomer 2 Reference Frame Tx DOF: %s' % (sym_entry.ref_frame_tx_dof2
                                                        if sym_entry.is_ref_frame_tx_dof2 else str(None)))
    log.info('Resulting SCM Symmetry: %s' % sym_entry.resulting_symmetry)
    log.info('SCM Dimension: %d' % sym_entry.dimension)
    log.info('SCM Unit Cell Specification: %s\n' % sym_entry.uc_specification)
    rot_step_deg1, rot_step_deg2 = get_rotation_step(sym_entry, rot_step_deg1, rot_step_deg2, initial=True, log=log)
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
    log.info('ROTATIONAL SAMPLING INFORMATION')
    log.info('Oligomer 1 ROT Sampling Range: %s' % (str(sym_entry.rotation_range1)
                                                    if sym_entry.is_internal_rot1 else str(None)))
    log.info('Oligomer 2 ROT Sampling Range: %s' % (str(sym_entry.rotation_range2)
                                                    if sym_entry.is_internal_rot2 else str(None)))
    log.info('Oligomer 1 ROT Sampling Step: %s' % (str(rot_step_deg1) if sym_entry.is_internal_rot1
                                                   else str(None)))
    log.info('Oligomer 2 ROT Sampling Step: %s\n' % (str(rot_step_deg2) if sym_entry.is_internal_rot2
                                                     else str(None)))
    # Get Degeneracy Matrices
    log.info('Searching For Possible Degeneracies')
    if sym_entry.degeneracy_matrices1 is None:
        log.info('No Degeneracies Found for Oligomer 1')
    elif len(sym_entry.degeneracy_matrices1) == 1:
        log.info('1 Degeneracy Found for Oligomer 1')
    else:
        log.info('%d Degeneracies Found for Oligomer 1' % len(sym_entry.degeneracy_matrices1))
    if sym_entry.degeneracy_matrices2 is None:
        log.info('No Degeneracies Found for Oligomer 2\n')
    elif len(sym_entry.degeneracy_matrices2) == 1:
        log.info('1 Degeneracy Found for Oligomer 2\n')
    else:
        log.info('%d Degeneracies Found for Oligomer 2\n' % len(sym_entry.degeneracy_matrices2))
    log.info('Retrieving Database of Complete Interface Fragment Cluster Representatives')


def get_components_from_nanohedra_docking(pose_file) -> list[str]:
    """Gather information for the docked Pose from a Nanohedra output. Includes coarse fragment metrics

    Returns:
        pose_transformation operations
    """
    entity_names = []
    with open(pose_file, 'r') as f:  # self.pose_file
        for line in f.readlines():
            if line[:15] == 'DOCKED POSE ID:':
                pose_id = line[15:].strip().replace('_DEGEN_', '-DEGEN_').replace('_ROT_', '-ROT_').replace('_TX_', '-tx_')
            elif line[:31] == 'Canonical Orientation PDB1 Path':
                canonical_pdb1 = line[31:].strip()
            elif line[:31] == 'Canonical Orientation PDB2 Path':
                canonical_pdb2 = line[31:].strip()

        if pose_id:
            entity_names = pose_id.split('-DEGEN_')[0].split('-')

        if len(entity_names) != number_of_nanohedra_components:  # probably old format without use of '-'
            entity_names = list(map(os.path.basename, [os.path.splitext(canonical_pdb1)[0],
                                                       os.path.splitext(canonical_pdb2)[0]]))
    return entity_names
