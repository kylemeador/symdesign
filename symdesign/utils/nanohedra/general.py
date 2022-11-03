from __future__ import annotations

import logging
import os
from logging import Logger
from typing import AnyStr

import numpy as np

from symdesign import utils

# Globals
logger = logging.getLogger(__name__)
number_of_nanohedra_components = 2


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

    out_info_file_path = os.path.join(outdir_path, utils.path.docked_pose_file)
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


def get_rotation_step(sym_entry: utils.SymEntry.SymEntry, rot_step_deg1: float | int = None,
                      rot_step_deg2: float | int = None, initial: bool = False, log: Logger = None) -> tuple[int, int]:
    """Set up the rotation step from the input arguments

    Returns:
        The rotational sampling steps for oligomer1 and oligomer2
    """
    if sym_entry.is_internal_rot1:  # if rotation step required
        if rot_step_deg1 is None:
            rot_step_deg1 = 3  # set rotation step to default
    else:
        if rot_step_deg1 and initial:
            log.warning("Specified Rotation Step 1 Was Ignored. Oligomer 1 Doesn't Have Internal Rotational DOF\n")
        rot_step_deg1 = 1

    if sym_entry.is_internal_rot2:  # if rotation step required
        if rot_step_deg2 is None:
            rot_step_deg2 = 3  # set rotation step to default
    else:
        if rot_step_deg2 and initial:
            log.warning("Specified Rotation Step 2 Was Ignored. Oligomer 2 Doesn't Have Internal Rotational DOF\n")
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


def retrieve_pose_transformation_from_nanohedra_docking(pose_file: AnyStr) -> list[dict]:
    """Gather pose transformation information for the Pose from Nanohedra output

    Args:
        pose_file: The file containing pose information from Nanohedra output
    Returns:
        The pose transformation arrays as found in the pose_file
    """
    with open(pose_file, 'r') as f:
        pose_transformation = {}
        for line in f.readlines():
            # all parsing lacks PDB number suffix such as PDB1 or PDB2 for hard coding in dict key
            if line[:20] == 'ROT/DEGEN MATRIX PDB':
                # data = eval(line[22:].strip())
                data = [[float(item) for item in group.split(', ')]
                        for group in line[22:].strip().strip('[]').split('], [')]
                pose_transformation[int(line[20:21])] = {'rotation': np.array(data)}
            elif line[:15] == 'INTERNAL Tx PDB':
                try:  # This may have values of None
                    data = np.array([float(item) for item in line[17:].strip().strip('[]').split(', ')])
                except ValueError:  # we received a string which is not a float
                    data = utils.symmetry.origin
                pose_transformation[int(line[15:16])]['translation'] = data
            elif line[:18] == 'SETTING MATRIX PDB':
                # data = eval(line[20:].strip())
                data = [[float(item) for item in group.split(', ')]
                        for group in line[20:].strip().strip('[]').split('], [')]
                pose_transformation[int(line[18:19])]['rotation2'] = np.array(data)
            elif line[:22] == 'REFERENCE FRAME Tx PDB':
                try:  # This may have values of None
                    data = np.array([float(item) for item in line[24:].strip().strip('[]').split(', ')])
                except ValueError:  # we received a string which is not a float
                    data = utils.symmetry.origin
                pose_transformation[int(line[22:23])]['translation2'] = data

    return [pose_transformation[idx] for idx, _ in enumerate(pose_transformation, 1)]


def get_components_from_nanohedra_docking(pose_file: AnyStr) -> list[str]:
    """Gather information on the docking componenet identifiers for the docked Pose from a Nanohedra output

    Args:
        pose_file: The file containing pose information from Nanohedra output
    Returns:
        The names of the models used during Nanohedra
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
