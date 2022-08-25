from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Iterable
from logging import Logger
from math import prod, ceil
from typing import AnyStr

import numpy as np
import psutil
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from resources.ml import proteinmpnn_factory, batch_proteinmpnn_input, mpnn_alphabet, score_sequences, \
    proteinmpnn_to_device
from structure.sequence import numeric_to_sequence
from utils.cluster import cluster_transformation_pairs
from resources.fragment import FragmentDatabase, fragment_factory
from utils.path import frag_text_file, master_log, frag_dir, biological_interfaces, asu_file_name
from structure.model import Pose, Model
from structure.base import Structure, Residue
from structure.coords import transform_coordinate_sets
from structure.fragment import GhostFragment, write_frag_match_info_file
from utils import dictionary_lookup, start_log, null_log, set_logging_to_debug, unpickle, rmsd_z_score, \
    z_value_from_match_score, match_score_from_z_value
from resources.EulerLookup import EulerLookup, euler_factory
from utils.nanohedra.OptimalTx import OptimalTx
from utils.SymEntry import SymEntry, get_rot_matrices, make_rotations_degenerate, symmetry_factory
from utils.nanohedra.WeightedSeqFreq import FragMatchInfo, SeqFreqInfo
from utils.nanohedra.cmdline import get_docking_parameters
from utils.nanohedra.general import write_docked_pose_info, get_rotation_step, write_docking_parameters
from utils.symmetry import generate_cryst1_record, get_central_asu

# Globals
logger = start_log(name=__name__, format_log=False, propagate=True)


def get_interface_residues(pdb1, pdb2, cb_distance=9.0):
    """Calculate all the residues within a cb_distance between two oligomers, identify associated ghost and surface
    fragments on each, by the chain name and residue number, translated the selected fragments to the oligomers using
    symmetry specific rotation matrix, internal translation vector, setting matrix, and external translation vector then
    return copies of these translated fragments

    Returns:
        (tuple[list[tuple], list[tuple]]): interface chain/residues on pdb1, interface chain/residues on pdb2
    """
    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_indices = pdb2.cb_indices
    pdb1_coords_indexed_residues = pdb1.coords_indexed_residues
    pdb2_coords_indexed_residues = pdb2.coords_indexed_residues

    pdb1_cb_kdtree = BallTree(pdb1.cb_coords)

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2.cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    # interacting_pairs = [(pdb1_residue.number, pdb1_residue.chain, pdb2_residue.number, pdb2_residue.chain)
    #                      for pdb2_query_index, pdb1_query in enumerate(query) for pdb1_query_index in pdb1_query]
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].size > 0:
            # pdb2_atom = pdb2.atoms[pdb2_cb_indices[pdb2_query_index]]
            pdb2_residue = pdb2_coords_indexed_residues[pdb2_cb_indices[pdb2_query_index]]
            # pdb2_cb_chain_id = pdb2.atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                # pdb1_atom = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]]
                pdb1_residue = pdb1_coords_indexed_residues[pdb1_cb_indices[pdb1_query_index]]
                # pdb1_cb_res_num = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                # pdb1_cb_chain_id = pdb1.atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append((pdb1_residue.number, pdb1_residue.chain, pdb2_residue.number,
                                          pdb2_residue.chain))

    pdb1_unique_chain_central_resnums, pdb2_unique_chain_central_resnums = [], []
    for pdb1_central_res_num, pdb1_central_chain_id, pdb2_central_res_num, pdb2_central_chain_id in interacting_pairs:
        pdb1_res_num_list = [pdb1_central_res_num + i for i in range(-2, 3)]  # Todo parameterize by frag length
        pdb2_res_num_list = [pdb2_central_res_num + i for i in range(-2, 3)]

        frag1_length = len(pdb1.chain(pdb1_central_chain_id).get_residues(numbers=pdb1_res_num_list))
        frag2_length = len(pdb2.chain(pdb2_central_chain_id).get_residues(numbers=pdb2_res_num_list))

        if frag1_length == 5 and frag2_length == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_unique_chain_central_resnums:
                pdb1_unique_chain_central_resnums.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_unique_chain_central_resnums:
                pdb2_unique_chain_central_resnums.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_unique_chain_central_resnums, pdb2_unique_chain_central_resnums


def get_contacting_asu(pdb1, pdb2, contact_dist=8, **kwargs):
    max_contact_count = 0
    max_contact_chain1, max_contact_chain2 = None, None
    for chain1 in pdb1.chains:
        pdb1_cb_coords_kdtree = BallTree(chain1.cb_coords)
        for chain2 in pdb2.chains:
            contact_count = pdb1_cb_coords_kdtree.two_point_correlation(chain2.cb_coords, [contact_dist])[0]

            if contact_count > max_contact_count:
                max_contact_count = contact_count
                max_contact_chain1, max_contact_chain2 = chain1, chain2

    if max_contact_count > 0:
        return Model.from_chains([max_contact_chain1, max_contact_chain2], name='asu', entities=True, **kwargs)
    else:
        return None


def find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, optimal_tx_params, complete_ghost_frags, complete_surf_frags,
                      degen_subdir_out_path, rot_subdir_out_path, eul_lookup,
                      rot_mat1=None, rot_mat2=None, max_z_val=2.0, output_assembly=False, output_surrounding_uc=False,
                      clash_dist=2.2, min_matched=3, high_quality_match_value=1, log=null_log):
    """

    Keyword Args:
        high_quality_match_value=1 (int): when z-value used, 0.5 if match score is used
    Returns:
        None
    """
    for tx_idx, tx_parameters in enumerate(optimal_tx_params, 1):
        log.info('Optimal Shift %d' % tx_idx)
        # tx_parameters contains [OptimalExternalDOFShifts (n_dof_ext), OptimalInternalDOFShifts (n_dof_int)]
        # Get Oligomer1, Oligomer2 Optimal Internal Translation vector
        internal_tx_param1, internal_tx_param2 = None, None
        if sym_entry.is_internal_tx1:
            internal_tx_param1 = [0, 0, tx_parameters[sym_entry.n_dof_external]]
        if sym_entry.is_internal_tx2:
            internal_tx_param2 = [0, 0, tx_parameters[sym_entry.n_dof_external + 1]]

        # Get Optimal External DOF shifts
        optimal_ext_dof_shifts = tx_parameters[:sym_entry.n_dof_external]
        external_tx_params1, external_tx_params2 = None, None
        ref_frame_var_is_pos, uc_dimensions = False, None
        if optimal_ext_dof_shifts:  # Todo TEST
            # Restrict all reference frame translation parameters to > 0 for SCMs with reference frame translational dof
            ref_frame_var_is_neg = False
            for ref_idx, ref_frame_tx_dof in enumerate(optimal_ext_dof_shifts, 1):
                if ref_frame_tx_dof < 0:
                    ref_frame_var_is_neg = True
                    break
            if ref_frame_var_is_neg:  # don't dock
                log.info('\tReference Frame Shift Parameter %d is Negative: %s\n' % (ref_idx, optimal_ext_dof_shifts))
                continue
            else:
                # Get Oligomer1 & Oligomer2 Optimal External Translation vector
                external_tx_params1 = sym_entry.get_optimal_external_tx_vector(optimal_ext_dof_shifts, group_number=1)
                external_tx_params2 = sym_entry.get_optimal_external_tx_vector(optimal_ext_dof_shifts, group_number=2)
                # Get Unit Cell Dimensions for 2D and 3D SCMs
                uc_dimensions = sym_entry.get_uc_dimensions(optimal_ext_dof_shifts)

        # Rotate, Translate and Set PDB1, PDB2
        copy_rot_tr_set_time_start = time.time()
        # Todo
        #  In theory, the rotation and setting matrix are the same for all tx_parameters, can we accelerate even though
        #  order of operations matters? by applying the setting matrix to the translation, in theory the translation
        #  will be along the same axis. This removes repeated multiplications and instead has array addition
        pdb1_copy = pdb1.return_transformed_copy(rotation=rot_mat1, translation=internal_tx_param1,
                                                 rotation2=sym_entry.setting_matrix1,
                                                 translation2=external_tx_params1)
        pdb2_copy = pdb2.return_transformed_copy(rotation=rot_mat2, translation=internal_tx_param2,
                                                 rotation2=sym_entry.setting_matrix2,
                                                 translation2=external_tx_params2)

        copy_rot_tr_set_time_stop = time.time()
        copy_rot_tr_set_time = copy_rot_tr_set_time_stop - copy_rot_tr_set_time_start
        log.info('\tCopy and Transform Oligomer1 and Oligomer2 (took %f s)' % copy_rot_tr_set_time)

        # Check if PDB1 and PDB2 backbones clash
        oligomer1_oligomer2_clash_time_start = time.time()
        # Todo @profile for KDTree or Neighbors 'brute'
        kdtree_oligomer1_backbone = BallTree(pdb1_copy.backbone_and_cb_coords)
        asu_cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.backbone_and_cb_coords,
                                                                             [clash_dist])
        oligomer1_oligomer2_clash_time_end = time.time()
        oligomer1_oligomer2_clash_time = oligomer1_oligomer2_clash_time_end - oligomer1_oligomer2_clash_time_start

        if asu_cb_clash_count[0] > 0:
            log.info('\tBackbone Clash when Oligomer1 and Oligomer2 are Docked (took %f s)'
                     % oligomer1_oligomer2_clash_time)
            continue
        log.info('\tNO Backbone Clash when Oligomer1 and Oligomer2 are Docked (took %f s)'
                 % oligomer1_oligomer2_clash_time)

        # Full Interface Fragment Match
        get_int_ghost_surf_frags_time_start = time.time()
        interface_chain_residues_pdb1, interface_chain_residues_pdb2 = get_interface_residues(pdb1_copy, pdb2_copy)
        unique_interface_frag_count_pdb1 = len(interface_chain_residues_pdb1)
        unique_interface_frag_count_pdb2 = len(interface_chain_residues_pdb2)
        unique_total_monofrags_count = unique_interface_frag_count_pdb1 + unique_interface_frag_count_pdb2

        # Todo
        #  The use of hashing on the surface and ghost fragments could increase program runtime, over tuple call
        #   to the ghost_fragment objects to return the aligned chain and residue then test for membership...
        #  Is the chain necessary? Two chains can occupy interface, even the same residue could be used
        #   Think D2 symmetry...
        #  Store all the ghost/surface frags in a chain/residue dictionary?
        interface_ghost_frags = [ghost_frag for ghost_frag in complete_ghost_frags
                                 if ghost_frag.get_aligned_chain_and_residue in interface_chain_residues_pdb1]
        interface_surf_frags = [surf_frag for surf_frag in complete_surf_frags
                                if surf_frag.get_aligned_chain_and_residue in interface_chain_residues_pdb2]
        # if unique_total_monofrags_count == 0:
        if not interface_ghost_frags or not interface_surf_frags:
            log.info('\tNO Interface Mono Fragments Found')
            continue

        ghost_frag_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in interface_ghost_frags])
        surf_frag_guide_coords = np.array([surf_frag.guide_coords for surf_frag in interface_surf_frags])

        transformed_ghostfrag_guide_coords_np = \
            transform_coordinate_sets(ghost_frag_guide_coords, rotation=rot_mat1, translation=internal_tx_param1,
                                      rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)

        transformed_monofrag2_guide_coords_np = \
            transform_coordinate_sets(surf_frag_guide_coords, rotation=rot_mat2, translation=internal_tx_param2,
                                      rotation2=sym_entry.setting_matrix2, translation2=external_tx_params2)

        # log.debug('Transformed guide_coords')
        get_int_ghost_surf_frags_time_end = time.time()
        get_int_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start

        log.info('\tNewly Formed Interface Contains %d Unique Fragments on Oligomer 1 and %d on Oligomer 2\n\t'
                 '(took %f s to get interface surface and ghost fragments with their guide coordinates)'
                 % (unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2, get_int_frags_time))

        # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Surface Fragment) guide coordinate pairs
        # in the same Euler rotational space bucket
        eul_lookup_start_time = time.time()
        overlapping_ghost_indices, overlapping_surf_indices = \
            eul_lookup.check_lookup_table(transformed_ghostfrag_guide_coords_np, transformed_monofrag2_guide_coords_np)
        # log.debug('Euler lookup')
        eul_lookup_end_time = time.time()
        eul_lookup_time = eul_lookup_end_time - eul_lookup_start_time

        # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        overlap_score_time_start = time.time()

        # filter array by matching type for surface (i) and ghost (j) frags
        surface_type_i_array = np.array([interface_surf_frags[idx].i_type for idx in overlapping_surf_indices.tolist()])
        ghost_type_j_array = np.array([interface_ghost_frags[idx].j_type for idx in overlapping_ghost_indices.tolist()])
        ij_type_match = np.where(surface_type_i_array == ghost_type_j_array, True, False)

        # get only fragment indices that pass ij filter and their associated coords
        passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
        passing_ghost_coords = transformed_ghostfrag_guide_coords_np[passing_ghost_indices]
        passing_surf_indices = overlapping_surf_indices[ij_type_match]
        passing_surf_coords = transformed_monofrag2_guide_coords_np[passing_surf_indices]

        # precalculate the reference_rmsds for each ghost fragment
        reference_rmsds = np.array([interface_ghost_frags[idx].rmsd for idx in passing_ghost_indices.tolist()])
        all_fragment_overlap = rmsd_z_score(passing_ghost_coords, passing_surf_coords, reference_rmsds)
        # log.debug('Checking all fragment overlap at interface')
        # get the passing_overlap indices and associated z-values by finding all indices where the value is not false
        passing_overlaps_indices = np.flatnonzero(np.where(all_fragment_overlap <= initial_z_value))
        passing_z_values = all_fragment_overlap[passing_overlaps_indices]
        # log.debug('Overlapping z-values: %s' % passing_z_values)

        overlap_score_time_stop = time.time()
        overlap_score_time = overlap_score_time_stop - overlap_score_time_start

        log.info('\t%d Fragment Match(es) Found in Complete Cluster Representative Fragment Library\n\t(Euler Lookup '
                 'took %f s for %d fragment pairs and Overlap Score Calculation took %f s for %d fragment pairs)'
                 % (len(passing_overlaps_indices), eul_lookup_time,
                    len(transformed_ghostfrag_guide_coords_np) * unique_interface_frag_count_pdb2,
                    overlap_score_time, len(overlapping_ghost_indices)))
        
        # check if the pose has enough high quality fragment matches
        high_qual_match_count = np.flatnonzero(passing_z_values < high_quality_match_value).size
        if high_qual_match_count < min_matched:
            log.info('\t%d < %d Which is Set as the Minimal Required Amount of High Quality Fragment Matches'
                     % (high_qual_match_count, min_matched))
            continue

        # Get contacting PDB 1 ASU and PDB 2 ASU
        asu = get_contacting_asu(pdb1_copy, pdb2_copy, log=log, entity_names=[pdb1_copy.name, pdb2_copy.name])
        # log.debug('Grabbing asu')
        if not asu:  # _pdb_1 and not asu_pdb_2:
            log.info('\tNO Design ASU Found')
            continue

        # Check if design has any clashes when expanded
        exp_des_clash_time_start = time.time()
        asu.uc_dimensions = uc_dimensions
        # asu.expand_matrices = sym_entry.expand_matrices
        symmetric_material = Pose.from_model(asu, sym_entry=sym_entry, ignore_clashes=True, log=log)
        #                      surrounding_uc=output_surrounding_uc, ^ ignores ASU clashes
        exp_des_clash_time_stop = time.time()
        exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

        # log.debug('Checked expand clash')
        if symmetric_material.symmetric_assembly_is_clash():
            log.info('\tBackbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
            continue

        log.info('\tNO Backbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
        # Todo replace with PoseDirectory? Path object?
        # oligomers_dir = rot_subdir_out_path.split(os.sep)[-3]
        degen_str = rot_subdir_out_path.split(os.sep)[-2]
        rot_str = rot_subdir_out_path.split(os.sep)[-1]
        # degen_str = 'DEGEN_{}'.format('_'.join(map(str, degen_counts[idx])))
        # rot_str = 'ROT_{}'.format('_'.join(map(str, rot_counts[idx])))
        tx_str = f'TX_{tx_idx}'  # translation idx
        # degen_subdir_out_path = os.path.join(outdir, degen_str)
        # rot_subdir_out_path = os.path.join(degen_subdir_out_path, rot_str)
        tx_dir = os.path.join(rot_subdir_out_path, tx_str.lower())  # .lower() keeps original publication format
        os.makedirs(tx_dir, exist_ok=True)
        sampling_id = f'{degen_str}-{rot_str}-{tx_str}'
        pose_id = f'{building_blocks}-{sampling_id}'

        # Make directories to output matched fragment PDB files
        # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
        matching_fragments_dir = os.path.join(tx_dir, 'matching_fragments')
        if not os.path.exists(matching_fragments_dir):
            os.makedirs(matching_fragments_dir)
        high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
        low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')

        # Write ASU, PDB1, PDB2, and expanded assembly files
        cryst1_record = None
        if optimal_ext_dof_shifts:
            asu = get_central_asu(asu, uc_dimensions, sym_entry.dimension)
            cryst1_record = generate_cryst1_record(uc_dimensions, sym_entry.resulting_symmetry)
        asu.write(out_path=os.path.join(tx_dir, asu_file_name), header=cryst1_record)
        pdb1_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb1_copy.name, sampling_id)))
        pdb2_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb2_copy.name, sampling_id)))

        if output_assembly:
            symmetric_material.generate_assembly_symmetry_models(surrounding_uc=output_surrounding_uc)
            if optimal_ext_dof_shifts:  # 2, 3 dimensions
                if output_surrounding_uc:
                    symmetric_material.write(out_path=os.path.join(tx_dir, 'surrounding_unit_cells.pdb'),
                                             header=cryst1_record)
                else:
                    symmetric_material.write(out_path=os.path.join(tx_dir, 'central_uc.pdb'), header=cryst1_record)
            else:  # 0 dimension
                symmetric_material.write(out_path=os.path.join(tx_dir, 'expanded_assembly.pdb'))
        log.info('\tSUCCESSFUL DOCKED POSE: %s' % tx_dir)

        # return the indices sorted by z_value then pull information accordingly
        sorted_fragment_indices = np.argsort(passing_z_values)
        sorted_z_values = passing_z_values[sorted_fragment_indices]
        match_scores = match_score_from_z_value(sorted_z_values)
        # log.debug('Overlapping Match Scores: %s' % match_scores)
        sorted_overlap_indices = passing_overlaps_indices[sorted_fragment_indices]
        int_ghostfrags = [interface_ghost_frags[idx] for idx in passing_ghost_indices[sorted_overlap_indices].tolist()]
        int_monofrags2 = [interface_surf_frags[idx] for idx in passing_surf_indices[sorted_overlap_indices].tolist()]

        # For all matched interface fragments
        # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
        # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
        chid_resnum_scores_dict_pdb1, chid_resnum_scores_dict_pdb2 = {}, {}

        # Number of unique interface mono fragments matched
        unique_frags_info1, unique_frags_info2 = set(), set()

        res_pair_freq_info_list = []
        for frag_idx, (int_ghost_frag, int_surf_frag) in enumerate(zip(int_ghostfrags, int_monofrags2)):
            surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_frag.get_aligned_chain_and_residue
            surf_frag_chain2, surf_frag_central_res_num2 = int_surf_frag.get_aligned_chain_and_residue

            covered_residues_pdb1 = [(surf_frag_chain1, surf_frag_central_res_num1 + j) for j in range(-2, 3)]
            covered_residues_pdb2 = [(surf_frag_chain2, surf_frag_central_res_num2 + j) for j in range(-2, 3)]
            score_term = match_scores[frag_idx]
            for k in range(ijk_frag_db.fragment_length):
                chain_resnum1 = covered_residues_pdb1[k]
                chain_resnum2 = covered_residues_pdb2[k]
                if chain_resnum1 not in chid_resnum_scores_dict_pdb1:
                    chid_resnum_scores_dict_pdb1[chain_resnum1] = [score_term]
                else:
                    chid_resnum_scores_dict_pdb1[chain_resnum1].append(score_term)

                if chain_resnum2 not in chid_resnum_scores_dict_pdb2:
                    chid_resnum_scores_dict_pdb2[chain_resnum2] = [score_term]
                else:
                    chid_resnum_scores_dict_pdb2[chain_resnum2].append(score_term)

            # if (surf_frag_chain1, surf_frag_central_res_num1) not in unique_frags_info1:
            unique_frags_info1.add((surf_frag_chain1, surf_frag_central_res_num1))
            # if (surf_frag_chain2, surf_frag_central_res_num2) not in unique_frags_info2:
            unique_frags_info2.add((surf_frag_chain2, surf_frag_central_res_num2))

            z_value = sorted_z_values[frag_idx]
            if z_value <= 1:  # the overlap z-value is less than 1 std deviation
                matched_fragment_dir = high_quality_matches_dir
            else:
                matched_fragment_dir = low_quality_matches_dir
            if not os.path.exists(matched_fragment_dir):
                os.makedirs(matched_fragment_dir)

            # if write_frags:  # write out aligned cluster representative fragment
            fragment, _ = dictionary_lookup(ijk_frag_db.paired_frags, int_ghost_frag.ijk)
            trnsfmd_ghost_fragment = fragment.return_transformed_copy(**int_ghost_frag.transformation)
            trnsfmd_ghost_fragment.transform(rotation=rot_mat1, translation=internal_tx_param1,
                                             rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            trnsfmd_ghost_fragment.write(out_path=os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
                                                               % ('i%d_j%d_k%d' % int_ghost_frag.ijk, frag_idx + 1)))
            # transformed_ghost_fragment = int_ghost_frag.structure.return_transformed_copy(
            #     rotation=rot_mat1, translation=internal_tx_param1,
            #     rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            # transformed_ghost_fragment.write(os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
            #                                               % ('i%d_j%d_k%d' % int_ghost_frag.ijk, frag_idx + 1)))

            ghost_frag_central_freqs = \
                dictionary_lookup(ijk_frag_db.info, int_ghost_frag.ijk).central_residue_pair_freqs
            # write out associated match information to frag_info_file
            write_frag_match_info_file(ghost_frag=int_ghost_frag, matched_frag=int_surf_frag,
                                       overlap_error=z_value, match_number=frag_idx + 1,
                                       central_frequencies=ghost_frag_central_freqs,
                                       out_path=matching_fragments_dir, pose_id=pose_id)

            # Keep track of residue pair frequencies and match information
            res_pair_freq_info_list.append(FragMatchInfo(ghost_frag_central_freqs,
                                                         surf_frag_chain1, surf_frag_central_res_num1,
                                                         surf_frag_chain2, surf_frag_central_res_num2, z_value))

        # log.debug('Wrote Fragments to matching_fragments')
        # calculate weighted frequency for central residues and write weighted frequencies to frag_text_file
        weighted_seq_freq_info = SeqFreqInfo(res_pair_freq_info_list)
        weighted_seq_freq_info.write(os.path.join(matching_fragments_dir, frag_text_file))

        unique_matched_monofrag_count = len(unique_frags_info1) + len(unique_frags_info2)
        percent_of_interface_covered = unique_matched_monofrag_count / float(unique_total_monofrags_count)

        # Calculate Nanohedra Residue Level Summation Score
        res_lev_sum_score = 0
        for res_scores_list1 in chid_resnum_scores_dict_pdb1.values():
            n1 = 1
            res_scores_list_sorted1 = sorted(res_scores_list1, reverse=True)
            for sc1 in res_scores_list_sorted1:
                res_lev_sum_score += sc1 * (1 / float(n1))
                n1 = n1 * 2
        for res_scores_list2 in chid_resnum_scores_dict_pdb2.values():
            n2 = 1
            res_scores_list_sorted2 = sorted(res_scores_list2, reverse=True)
            for sc2 in res_scores_list_sorted2:
                res_lev_sum_score += sc2 * (1 / float(n2))
                n2 = n2 * 2

        # Write Out Docked Pose Info to docked_pose_info_file.txt
        write_docked_pose_info(tx_dir, res_lev_sum_score, high_qual_match_count, unique_matched_monofrag_count,
                               unique_total_monofrags_count, percent_of_interface_covered, rot_mat1, internal_tx_param1,
                               sym_entry.setting_matrix1, external_tx_params1, rot_mat2, internal_tx_param2,
                               sym_entry.setting_matrix2, external_tx_params2, cryst1_record, pdb1.file_path,
                               pdb2.file_path, pose_id)


def slice_variable_for_log(var, length=5):
    return var[:length]


# TODO decrease amount of work by saving each index array and reusing...
#  such as stacking each j_index, guide_coords, rmsd, etc and pulling out by index
def is_frag_type_same(frags1, frags2, dtype='ii'):
    frag1_type = f'{dtype[0]}_type'
    frag2_type = f'{dtype[1]}_type'
    frag1_indices = np.array([getattr(frag, frag1_type) for frag in frags1])
    frag2_indices = np.array([getattr(frag, frag2_type) for frag in frags2])
    # np.where(frag1_indices_repeat == frag2_indices_tile)
    frag1_indices_repeated = np.repeat(frag1_indices, len(frag2_indices))
    frag2_indices_tiled = np.tile(frag2_indices, len(frag1_indices))
    return np.where(frag1_indices_repeated == frag2_indices_tiled, True, False).reshape(
        len(frag1_indices), -1)


def compute_ij_type_lookup(indices1: np.ndarray | Iterable | int | float,
                           indices2: np.ndarray | Iterable | int | float) -> np.ndarray | int | float:
    """Compute a lookup table where the array elements are indexed to boolean values if the indices match.
    Axis 0 is indices1, Axis 1 is indices2

    Args:
        indices1: The array elements from group 1
        indices2: The array elements from group 2
    Returns:
        A 2D boolean array where the first index maps to the input 1, second index maps to index 2
    """
    # TODO use broadcasting to compute true/false instead of tiling (memory saving)
    indices1_repeated = np.repeat(indices1, len(indices2))
    indices2_tiled = np.tile(indices2, len(indices1))
    # TODO keep as table or flatten? Can use one or the other as memory and take a view of the other as needed...
    return np.where(indices1_repeated == indices2_tiled, True, False).reshape(len(indices1), -1)


def nanohedra_dock(sym_entry: SymEntry, ijk_frag_db: FragmentDatabase, euler_lookup: EulerLookup,
                   master_output: AnyStr, model1: Structure | AnyStr, model2: Structure | AnyStr,
                   rotation_step1: float = 3., rotation_step2: float = 3., min_matched: int = 3,
                   high_quality_match_value: float = .5, initial_z_value: float = 1., output_assembly: bool = False,
                   output_surrounding_uc: bool = False, log: Logger = logger, clash_dist: float = 2.2,
                   keep_time: bool = True, write_frags: bool = False, same_component_filter: bool = False, **kwargs):
    #                resume=False,
    """
    Perform the fragment docking routine described in Laniado, Meador, & Yeates, PEDS. 2021

    Args:
        sym_entry: The SymmetryEntry object describing the material
        ijk_frag_db: The FragmentDatabase object used for finding fragment pairs
        euler_lookup: The EulerLookup object used to search for overlapping euler angles
        master_output: The object to issue outputs to
        model1: The first Structure to be used in docking
        model2: The second Structure to be used in docking
        rotation_step1: The number of degrees to increment the rotational degrees of freedom search
        rotation_step2: The number of degrees to increment the rotational degrees of freedom search
        min_matched: How many high quality fragment pairs should be present before a pose is identified?
        high_quality_match_value: The value to exceed before a high quality fragment is matched
            When z-value was used this was 1.0, however 0.5 when match score is used
        initial_z_value: The acceptable standard deviation z score for initial fragment overlap identification.
            Smaller values lead to more stringent matching criteria
        output_assembly: Whether the assembly should be output? Infinite materials are output in a unit cell
        output_surrounding_uc: Whether the surrounding unit cells should be output? Only for infinite materials
        log: The logger to keep track of program messages
        clash_dist: The distance to measure for clashing atoms
        keep_time: Whether the times for each step should be reported
        write_frags: Whether to write fragment information to a file (useful for fragment based docking w/o Nanohedra)
        same_component_filter: Whether to use the overlap potential on the same component to filter ghost fragments
    Returns:
        None
    """
    frag_dock_time_start = time.time()
    outlier = -1
    # Todo set below as parameters?
    design_output = False
    design_temperature = 0.1
    mpnn_model = proteinmpnn_factory()  # Todo accept model_name arg. Now just use the default
    number_of_mpnn_model_parameters = sum([prod(param.size()) for param in mpnn_model.parameters()])
    log.critical(f'The number of proteinmpnn model parameters is: {number_of_mpnn_model_parameters}')
    low_quality_match_value = .2  # sets the lower bounds on an acceptable match, was upper bound of 2 using z-score
    cb_distance = 9.  # change to 8.?
    # cluster_translations = True
    perturb_transformations = True
    translation_epsilon = 1  # 0.75
    high_quality_z_value = z_value_from_match_score(high_quality_match_value)
    low_quality_z_value = z_value_from_match_score(low_quality_match_value)
    # Get Building Blocks in pose format to remove need for fragments to use chain info
    if not isinstance(model1, Structure):
        model1 = Model.from_file(model1)  # , pose_format=True)
    if not isinstance(model2, Structure):
        model2 = Model.from_file(model2)  # , pose_format=True)

    # Get model with entity oligomers via make_oligomer
    models = [model1, model2]
    for idx, (model, symmetry) in enumerate(zip(models, sym_entry.groups)):
        for entity in model.entities:
            # Precompute reference sequences if available
            # dummy = entity.reference_sequence  # use the incomming SEQRES REMARK for now
            if entity.is_oligomeric():
                continue
            else:
                entity.make_oligomer(symmetry=symmetry)
                # entity.write_oligomer(out_path=os.path.join(master_output, f'{entity.name}_make_oligomer.pdb'))

        # Make, then save a new model based on the symmetric version of each Entity in the Model
        models[idx] = Model.from_chains([chain for entity in model.entities for chain in entity.chains],
                                        name=model.name, pose_format=True)
        models[idx].file_path = model.file_path

    # Set up output mechanism
    if isinstance(master_output, str) and not write_frags:  # we just want to write, so don't make a directory
        building_blocks = '-'.join(model.name for model in models)
        outdir = os.path.join(master_output, building_blocks)
        os.makedirs(outdir, exist_ok=True)
    else:
        raise NotImplementedError('Must provide a master_outdir!')
    # elif isinstance(master_output, DockingDirectory):
    #     pass
    #     Todo make a docking directory object compatible with this and implement sql handle

    # Setup log
    if log is None:
        log_file_path = os.path.join(outdir, f'{building_blocks}_log.txt')
    else:
        log_file_path = getattr(log.handlers[0], 'baseFilename', None)
    if log_file_path:
        # Start logging to a file in addition
        log = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False, propagate=True)
    # else:
    #     # we are probably logging to stream and we need to check another method to see if output exists

    for model in models:
        model.log = log

    # Todo figure out for single component
    model1: Model
    model2: Model
    model1, model2 = models
    log.info(f'DOCKING {model1.name} TO {model2.name}\n'
             f'Oligomer 1 Path: {model1.file_path}\nOligomer 2 Path: {model2.file_path}')

    # Set up Building Block2
    # Get Surface Fragments With Guide Coordinates Using COMPLETE Fragment Database
    get_complete_surf_frags2_time_start = time.time()
    complete_surf_frags2 = \
        model2.get_fragment_residues(residues=model2.surface_residues, fragment_db=ijk_frag_db)

    # Calculate the initial match type by finding the predominant surface type
    surf_guide_coords2 = np.array([surf_frag.guide_coords for surf_frag in complete_surf_frags2])
    surf_residue_numbers2 = np.array([surf_frag.number for surf_frag in complete_surf_frags2])
    surf_i_indices2 = np.array([surf_frag.i_type for surf_frag in complete_surf_frags2])
    fragment_content2 = np.bincount(surf_i_indices2)
    initial_surf_type2 = np.argmax(fragment_content2)
    init_surf_frag_indices2 = \
        [idx for idx, surf_frag in enumerate(complete_surf_frags2) if surf_frag.i_type == initial_surf_type2]
    init_surf_guide_coords2 = surf_guide_coords2[init_surf_frag_indices2]
    init_surf_residue_numbers2 = surf_residue_numbers2[init_surf_frag_indices2]
    idx = 2
    log.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords2.shape}')
    log.debug(f'Found surface residue numbers {idx} with shape {surf_residue_numbers2.shape}')
    log.debug(f'Found surface indices {idx} with shape {surf_i_indices2.shape}')
    log.debug(f'Found {init_surf_residue_numbers2.shape[0]} initial surface {idx} fragments with type: {initial_surf_type2}')

    # log.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    log.info(f'Getting Oligomer 2 Surface Fragments and Guides Using COMPLETE Fragment Database (took '
             f'{time.time() - get_complete_surf_frags2_time_start:8f}s)')

    # log.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # log.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # log.debug('init_surf_residue_numbers2: %s' % slice_variable_for_log(init_surf_residue_numbers2))

    # Set up Building Block1
    get_complete_surf_frags1_time_start = time.time()
    surf_frags1 = model1.get_fragment_residues(residues=model1.surface_residues, fragment_db=ijk_frag_db)

    # Calculate the initial match type by finding the predominant surface type
    fragment_content1 = np.bincount([surf_frag.i_type for surf_frag in surf_frags1])
    initial_surf_type1 = np.argmax(fragment_content1)
    init_surf_frags1 = [surf_frag for surf_frag in surf_frags1 if surf_frag.i_type == initial_surf_type1]
    # init_surf_guide_coords1 = np.array([surf_frag.guide_coords for surf_frag in init_surf_frags1])
    # init_surf_residue_numbers1 = np.array([surf_frag.number for surf_frag in init_surf_frags1])
    # surf_frag1_residues = [surf_frag.number for surf_frag in surf_frags1]
    idx = 1
    # log.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords1.shape}')
    # log.debug(f'Found surface residue numbers {idx} with shape {surf_residue_numbers1.shape}')
    # log.debug(f'Found surface indices {idx} with shape {surf_i_indices1.shape}')
    log.debug(f'Found {len(init_surf_frags1)} initial surface {idx} fragments with type: {initial_surf_type1}')
    # log.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    # log.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # log.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # log.debug('init_surf_residue_numbers2: %s' % slice_variable_for_log(init_surf_residue_numbers2))
    # log.debug('init_surf_guide_coords1: %s' % slice_variable_for_log(init_surf_guide_coords1))
    # log.debug('init_surf_residue_numbers1: %s' % slice_variable_for_log(init_surf_residue_numbers1))

    log.info(f'Getting Oligomer {idx} Surface Fragments and Guides Using COMPLETE Fragment Database (took '
             f'{time.time() - get_complete_surf_frags1_time_start:8f}s)')

    #################################
    # Get component 1 ghost fragments and associated data from complete fragment database
    oligomer1_backbone_cb_tree = BallTree(model1.backbone_and_cb_coords)
    get_complete_ghost_frags1_time_start = time.time()
    ghost_frags_by_residue1 = \
        [frag.get_ghost_fragments(clash_tree=oligomer1_backbone_cb_tree) for frag in surf_frags1]

    complete_ghost_frags1: list[GhostFragment] = \
        [ghost for ghosts in ghost_frags_by_residue1 for ghost in ghosts]

    ghost_guide_coords1 = np.array([ghost_frag.guide_coords for ghost_frag in complete_ghost_frags1])
    ghost_rmsds1 = np.array([ghost_frag.rmsd for ghost_frag in complete_ghost_frags1])
    ghost_residue_numbers1 = np.array([ghost_frag.number for ghost_frag in complete_ghost_frags1])
    ghost_j_indices1 = np.array([ghost_frag.j_type for ghost_frag in complete_ghost_frags1])

    if same_component_filter:  # Todo test this
        # surface ghost frag overlap for the same oligomer
        ghost_lengths = max([len(residue_ghosts) for residue_ghosts in ghost_frags_by_residue1])
        # set up the output array with the number of residues by the length of the max number of ghost fragments
        same_component_overlapping_ghost_frags = np.zeros((len(surf_frags1), ghost_lengths))
        # Todo size all of these correctly given different padding
        # set up the input array types with the various information needed for each pairwise check
        ghost_frag_type_by_residue = [[ghost.frag_type for ghost in residue_ghosts]
                                      for residue_ghosts in ghost_frags_by_residue1]
        ghost_frag_rmsds_by_residue = np.array([[ghost.rmsd for ghost in residue_ghosts]
                                                for residue_ghosts in ghost_frags_by_residue1])
        ghost_guide_coords_by_residue1 = np.array([[ghost.guide_coords for ghost in residue_ghosts]
                                                   for residue_ghosts in ghost_frags_by_residue1])
        # surface_frag_residue_numbers = [residue.number for residue in surf_frags1]
        surface_frag_residue_indices = list(range(len(surf_frags1)))
        surface_frag_cb_coords = np.concatenate([residue.cb_coords for residue in surf_frags1], axis=0)
        model1_surface_cb_ball_tree = BallTree(surface_frag_cb_coords)
        residue_contact_query: list[list[int]] = model1_surface_cb_ball_tree.query(surface_frag_cb_coords, cb_distance)
        contacting_pairs: list[tuple[int, int]] = \
            [(surface_frag_residue_indices[idx1], surface_frag_residue_indices[idx2])
             for idx2 in range(residue_contact_query.size) for idx1 in residue_contact_query[idx2]]
        asymmetric_contacting_residue_pairs, found_pairs = [], []
        for residue_idx1, residue_idx2 in contacting_pairs:
            # only add to asymmetric_contacting_residue_pairs if we have never observed either
            if (residue_idx1, residue_idx2) not in found_pairs and residue_idx1 != residue_idx2:  # or (residue2, residue1) not in found_pairs
                asymmetric_contacting_residue_pairs.append((residue_idx1, residue_idx2))
            # add both pair orientations (1, 2) or (2, 1) regardless
            found_pairs.extend([(residue_idx1, residue_idx2), (residue_idx2, residue_idx1)])

        # Now we use the asymmetric_contacting_residue_pairs to find the ghost_fragments for each residue
        for residue_idx1, residue_idx2 in asymmetric_contacting_residue_pairs:
            # type_bool_matrix = is_frag_type_same(ghost_frags_by_residue1[residue_idx1 - 1],
            #                                      ghost_frags_by_residue1[residue_idx2 - 1], dtype='jj')
            type_bool_matrix = compute_ij_type_lookup(ghost_frag_type_by_residue[residue_idx1],
                                                      ghost_frag_type_by_residue[residue_idx2])
            #   Fragment1
            # F T  F  F
            # R F  F  T
            # A F  F  F
            # G F  F  F
            # 2 T  T  F
            # use type_bool_matrix to guide RMSD calculation by pulling out the right ghost_coords for each residue_idx
            residue_idx1_ghost_indices, residue_idx2_ghost_indices = np.nonzero(type_bool_matrix)
            # iterate over each matrix rox/column to pull out necessary guide coordinate pairs
            # HERE v
            # ij_matching_ghost1_indices = (type_bool_matrix * np.arange(type_bool_matrix.shape[0]))[type_bool_matrix]

            # these should pick out each instance of the guide_coords found by indexing residue_idxN_ghost_indices
            print('ghost_guide_coords_by_residue1[residue_idx1]', len(ghost_guide_coords_by_residue1[residue_idx1]))
            ghost_coords_residue1 = ghost_guide_coords_by_residue1[residue_idx1][residue_idx1_ghost_indices]
            print('ghost_guide_coords_by_residue1[residue_idx2]', len(ghost_guide_coords_by_residue1[residue_idx2]))
            ghost_coords_residue2 = ghost_guide_coords_by_residue1[residue_idx2][residue_idx2_ghost_indices]
            print('RESULT of residue_idx1_ghost_indices indexing -> ghost_coords_residue1:', len(ghost_coords_residue1))
            if len(ghost_coords_residue1) != len(residue_idx1_ghost_indices):
                raise IndexError('There was an issue indexing')

            ghost_reference_rmsds_residue1 = ghost_frag_rmsds_by_residue[residue_idx1][residue_idx1_ghost_indices]
            overlapping_indices = rmsd_z_score(ghost_coords_residue1,
                                               ghost_coords_residue2,
                                               ghost_reference_rmsds_residue1)  # , max_z_value=initial_z_value)
            same_component_overlapping_indices = np.flatnonzero(np.where(overlapping_indices <= initial_z_value))
            same_component_overlapping_ghost_frags[residue_idx1, residue_idx1_ghost_indices[same_component_overlapping_indices]] += 1
            same_component_overlapping_ghost_frags[residue_idx2, residue_idx2_ghost_indices[same_component_overlapping_indices]] += 1

        # Using the tabulated results, prioritize those fragments which have same component, ghost fragment overlap
        initial_ghost_frags1 = \
            [complete_ghost_frags1[idx] for idx in same_component_overlapping_ghost_frags.flatten().tolist()]
        init_ghost_guide_coords1 = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags1])
        init_ghost_rmsds1 = np.array([ghost_frag.rmsd for ghost_frag in initial_ghost_frags1])
        # init_ghost_residue_numbers1 = np.array([ghost_frag.number for ghost_frag in initial_ghost_frags1])
    else:
        init_ghost_frag_indices1 = \
            [idx for idx, ghost_frag in enumerate(complete_ghost_frags1) if ghost_frag.j_type == initial_surf_type2]
        init_ghost_guide_coords1: np.ndarray = ghost_guide_coords1[init_ghost_frag_indices1]
        init_ghost_rmsds1: np.ndarray = ghost_rmsds1[init_ghost_frag_indices1]
        # init_ghost_residue_numbers1: np.ndarray = ghost_residue_numbers1[init_ghost_frag_indices1]

    idx = 1
    log.debug(f'Found ghost guide coordinates {idx} with shape {ghost_guide_coords1.shape}')
    log.debug(f'Found ghost residue numbers {idx} with shape {ghost_residue_numbers1.shape}')
    log.debug(f'Found ghost indices {idx} with shape {ghost_j_indices1.shape}')
    log.debug(f'Found ghost rmsds {idx} with shape {ghost_rmsds1.shape}')
    log.debug(f'Found {init_ghost_guide_coords1.shape[0]} initial ghost {idx} fragments with type {initial_surf_type2}')

    log.info(f'Getting {model1.name} Oligomer {idx} Ghost Fragments and Guides Using COMPLETE Fragment Database '
             f'(took {time.time() - get_complete_ghost_frags1_time_start:8f}s)')

    #################################
    if write_frags:  # implemented for Todd to work on C1 instances
        guide_file_ghost = os.path.join(os.getcwd(), f'{model1.name}_ghost_coords.txt')
        with open(guide_file_ghost, 'w') as f:
            for coord_group in ghost_guide_coords1.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_ghost_idx = os.path.join(os.getcwd(), f'{model1.name}_ghost_coords_index.txt')
        with open(guide_file_ghost_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_j_indices1.tolist())))
        guide_file_ghost_res_num = os.path.join(os.getcwd(), f'{model1.name}_ghost_coords_residue_number.txt')
        with open(guide_file_ghost_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_residue_numbers1.tolist())))

        guide_file_surf = os.path.join(os.getcwd(), f'{model2.name}_surf_coords.txt')
        with open(guide_file_surf, 'w') as f:
            for coord_group in surf_guide_coords2.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_surf_idx = os.path.join(os.getcwd(), f'{model2.name}_surf_coords_index.txt')
        with open(guide_file_surf_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_i_indices2.tolist())))
        guide_file_surf_res_num = os.path.join(os.getcwd(), f'{model2.name}_surf_coords_residue_number.txt')
        with open(guide_file_surf_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_residue_numbers2.tolist())))

        raise RuntimeError(f'Suspending operation of {model1.name}/{model2.name} after write')

    ij_type_match_lookup_table = compute_ij_type_lookup(ghost_j_indices1, surf_i_indices2)
    # Axis 0 is ghost frag, 1 is surface frag
    # ij_matching_ghost1_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[0]))[ij_type_match_lookup_table]
    # ij_matching_surf2_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[1])[:, None])[
    #         ij_type_match_lookup_table]
    # Todo apparently this should work to grab the flattened indices where there is overlap
    #  row_indices, column_indices = np.indices(ij_type_match_lookup_table.shape)
    #  # row index vary with ghost, column surf
    #  # transpose to index the first axis (axis=0) along the 1D row indices
    #  ij_matching_ghost1_indices = row_indices[ij_type_match_lookup_table.T]
    #  ij_matching_surf2_indices = column_indices[ij_type_match_lookup_table]
    #  >>> j = np.ones(23)
    #  >>> k = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    #  >>> k.shape
    #  (2, 23)
    #  >>> j[k]
    #  array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1., 1., 1., 1., 1., 1.]])
    #  This will allow pulling out the indices where there is overlap which may be useful
    #  for limiting scope of overlap checks

    # Get component 2 ghost fragments and associated data from complete fragment database
    bb_cb_coords2 = model2.backbone_and_cb_coords
    bb_cb_tree2 = BallTree(bb_cb_coords2)
    get_complete_ghost_frags2_time_start = time.time()
    complete_ghost_frags2 = []
    for frag in complete_surf_frags2:
        complete_ghost_frags2.extend(frag.get_ghost_fragments(clash_tree=bb_cb_tree2))
    init_ghost_frags2 = [ghost_frag for ghost_frag in complete_ghost_frags2 if ghost_frag.j_type == initial_surf_type1]
    init_ghost_guide_coords2 = np.array([ghost_frag.guide_coords for ghost_frag in init_ghost_frags2])
    # init_ghost_residue_numbers2 = np.array([ghost_frag.number for ghost_frag in init_ghost_frags2])
    # ghost_frag2_residues = [ghost_frag.number for ghost_frag in complete_ghost_frags2]

    idx = 2
    # log.debug(f'Found ghost guide coordinates {idx} with shape {ghost_guide_coords2.shape}')
    # log.debug(f'Found ghost residue numbers {idx} with shape {ghost_residue_numbers2.shape}')
    # log.debug(f'Found ghost indices {idx} with shape {ghost_j_indices2.shape}')
    # log.debug(f'Found ghost rmsds {idx} with shape {ghost_rmsds2.shape}')
    log.debug(f'Found {len(init_ghost_guide_coords2)} initial ghost {idx} fragments with type {initial_surf_type1}')
    # log.debug('init_ghost_guide_coords2: %s' % slice_variable_for_log(init_ghost_guide_coords2))
    # log.debug('init_ghost_residue_numbers2: %s' % slice_variable_for_log(init_ghost_residue_numbers2))
    # Prepare precomputed arrays for fast pair lookup
    # ghost1_residue_array = np.repeat(init_ghost_residue_numbers1, len(init_surf_residue_numbers2))
    # ghost2_residue_array = np.repeat(init_ghost_residue_numbers2, len(init_surf_residue_numbers1))
    # surface1_residue_array = np.tile(init_surf_residue_numbers1, len(init_ghost_residue_numbers2))
    # surface2_residue_array = np.tile(init_surf_residue_numbers2, len(init_ghost_residue_numbers1))

    log.info(f'Getting {model2.name} Oligomer 2 Ghost Fragments and Guides Using COMPLETE Fragment Database '
             f'(took {time.time() - get_complete_ghost_frags2_time_start:8f}s)')

    log.info('Obtaining Rotation/Degeneracy Matrices\n')

    rotation_steps = [rotation_step1, rotation_step2]
    number_of_degens = []
    number_of_rotations = []
    rotation_matrices = []
    for idx, rotation_step in enumerate(rotation_steps, 1):
        if getattr(sym_entry, f'is_internal_rot{idx}'):  # if rotation step required
            if rotation_step is None:
                rotation_steps[idx] = 3  # set rotation step to default
        else:
            rotation_steps[idx] = 1
            if rotation_step:
                log.warning(f'Specified rotation_step{idx} was ignored. Oligomer {idx} doesn\'t have rotational DOF')

        degeneracy_matrices = getattr(sym_entry, f'degeneracy_matrices{idx}')
        rot_degen_matrices = make_rotations_degenerate(get_rot_matrices(rotation_step, 'z',
                                                                        getattr(sym_entry, f'rotation_range{idx}')),
                                                       degeneracy_matrices)
        log.debug(f'Degeneracy shape for component {idx}: {degeneracy_matrices.shape}')
        log.debug(f'Combined rotation shape for component {idx}: {rot_degen_matrices.shape}')
        number_of_degens.append(degeneracy_matrices.shape[0])
        # log.debug(f'Rotation shape for component {idx}: {rot_degen_matrices.shape}')
        number_of_rotations.append(rot_degen_matrices.shape[0] // degeneracy_matrices.shape[0])
        rotation_matrices.append(rot_degen_matrices)

    set_mat1, set_mat2 = sym_entry.setting_matrix1, sym_entry.setting_matrix2

    # def check_forward_and_reverse(test_ghost_guide_coords, stack_rot1, stack_tx1,
    #                               test_surf_guide_coords, stack_rot2, stack_tx2,
    #                               reference_rmsds):
    #     """Debug forward versus reverse guide coordinate fragment matching
    #
    #     All guide_coords and reference_rmsds should be indexed to the same length and overlap
    #     """
    #     mismatch = False
    #     inv_set_mat1 = np.linalg.inv(set_mat1)
    #     for shift_idx in range(1):
    #         rot1 = stack_rot1[shift_idx]
    #         tx1 = stack_tx1[shift_idx]
    #         rot2 = stack_rot2[shift_idx]
    #         tx2 = stack_tx2[shift_idx]
    #
    #         tnsfmd_ghost_coords = transform_coordinate_sets(test_ghost_guide_coords,
    #                                                         rotation=rot1,
    #                                                         translation=tx1,
    #                                                         rotation2=set_mat1)
    #         tnsfmd_surf_coords = transform_coordinate_sets(test_surf_guide_coords,
    #                                                        rotation=rot2,
    #                                                        translation=tx2,
    #                                                        rotation2=set_mat2)
    #         int_euler_matching_ghost_indices, int_euler_matching_surf_indices = \
    #             euler_lookup.check_lookup_table(tnsfmd_ghost_coords, tnsfmd_surf_coords)
    #
    #         all_fragment_match = calculate_match(tnsfmd_ghost_coords[int_euler_matching_ghost_indices],
    #                                              tnsfmd_surf_coords[int_euler_matching_surf_indices],
    #                                              reference_rmsds[int_euler_matching_ghost_indices])
    #         high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
    #         high_qual_match_count = len(high_qual_match_indices)
    #         if high_qual_match_count < min_matched:
    #             log.info(
    #                 f'\t{high_qual_match_count} < {min_matched} Which is Set as the Minimal Required Amount of '
    #                 f'High Quality Fragment Matches')
    #
    #         # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
    #         passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
    #         number_passing_overlaps = len(passing_overlaps_indices)
    #         log.info(
    #             f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} '
    #             f'Matches Found in Complete Fragment Library')
    #
    #         # now try inverse
    #         inv_rot_mat1 = np.linalg.inv(rot1)
    #         tnsfmd_surf_coords_inv = transform_coordinate_sets(tnsfmd_surf_coords,
    #                                                            rotation=inv_set_mat1,
    #                                                            translation=tx1 * -1,
    #                                                            rotation2=inv_rot_mat1)
    #         int_euler_matching_ghost_indices_inv, int_euler_matching_surf_indices_inv = \
    #             euler_lookup.check_lookup_table(test_ghost_guide_coords, tnsfmd_surf_coords_inv)
    #
    #         all_fragment_match = calculate_match(test_ghost_guide_coords[int_euler_matching_ghost_indices_inv],
    #                                              tnsfmd_surf_coords_inv[int_euler_matching_surf_indices_inv],
    #                                              reference_rmsds[int_euler_matching_ghost_indices_inv])
    #         high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
    #         high_qual_match_count = len(high_qual_match_indices)
    #         if high_qual_match_count < min_matched:
    #             log.info(
    #                 f'\tINV {high_qual_match_count} < {min_matched} Which is Set as the Minimal Required Amount'
    #                 f' of High Quality Fragment Matches')
    #
    #         # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
    #         passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
    #         number_passing_overlaps = len(passing_overlaps_indices)
    #
    #         log.info(
    #             f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} '
    #             f'Matches Found in Complete Fragment Library')
    #
    #         def investigate_mismatch():
    #             log.info(f'Euler True ghost/surf indices forward and inverse don\'t match. '
    #                      f'Shapes: Forward={int_euler_matching_ghost_indices.shape}, '
    #                      f'Inverse={int_euler_matching_ghost_indices_inv.shape}')
    #             log.debug(f'tnsfmd_ghost_coords.shape {tnsfmd_ghost_coords.shape}')
    #             log.debug(f'tnsfmd_surf_coords.shape {tnsfmd_surf_coords.shape}')
    #             int_euler_matching_array = \
    #                 euler_lookup.check_lookup_table(tnsfmd_ghost_coords, tnsfmd_surf_coords, return_bool=True)
    #             int_euler_matching_array_inv = \
    #                 euler_lookup.check_lookup_table(test_ghost_guide_coords, tnsfmd_surf_coords_inv, return_bool=True)
    #             # Change the shape to allow for relation to guide_coords
    #             different = np.where(int_euler_matching_array != int_euler_matching_array_inv,
    #                                  True, False).reshape(tnsfmd_ghost_coords.shape[0], -1)
    #             ghost_indices, surface_indices = np.nonzero(different)
    #             log.debug(f'different.shape {different.shape}')
    #
    #             different_ghosts = tnsfmd_ghost_coords[ghost_indices]
    #             different_surf = tnsfmd_surf_coords[surface_indices]
    #             tnsfmd_ghost_ints1, tnsfmd_ghost_ints2, tnsfmd_ghost_ints3 = \
    #                 euler_lookup.get_eulint_from_guides(different_ghosts)
    #             tnsfmd_surf_ints1, tnsfmd_surf_ints2, tnsfmd_surf_ints3 = \
    #                 euler_lookup.get_eulint_from_guides(different_surf)
    #             stacked_ints = np.stack([tnsfmd_ghost_ints1, tnsfmd_ghost_ints2, tnsfmd_ghost_ints3,
    #                                      tnsfmd_surf_ints1, tnsfmd_surf_ints2, tnsfmd_surf_ints3], axis=0).T
    #             log.info(
    #                 f'The mismatched forward Euler ints are\n{[ints for ints in list(stacked_ints)[:10]]}\n')
    #
    #             different_ghosts_inv = test_ghost_guide_coords[ghost_indices]
    #             different_surf_inv = tnsfmd_surf_coords_inv[surface_indices]
    #             tnsfmd_ghost_ints_inv1, tnsfmd_ghost_ints_inv2, tnsfmd_ghost_ints_inv3 = \
    #                 euler_lookup.get_eulint_from_guides(different_ghosts_inv)
    #             tnsfmd_surf_ints_inv1, tnsfmd_surf_ints_inv2, tnsfmd_surf_ints_inv3 = \
    #                 euler_lookup.get_eulint_from_guides(different_surf_inv)
    #
    #             stacked_ints_inv = \
    #                 np.stack([tnsfmd_ghost_ints_inv1, tnsfmd_ghost_ints_inv2, tnsfmd_ghost_ints_inv3,
    #                           tnsfmd_surf_ints_inv1, tnsfmd_surf_ints_inv2, tnsfmd_surf_ints_inv3], axis=0).T
    #             log.info(
    #                 f'The mismatched inverse Euler ints are\n{[ints for ints in list(stacked_ints_inv)[:10]]}\n')
    #
    #         if not np.array_equal(int_euler_matching_ghost_indices, int_euler_matching_ghost_indices_inv):
    #             mismatch = True
    #
    #         if not np.array_equal(int_euler_matching_surf_indices, int_euler_matching_surf_indices_inv):
    #             mismatch = True
    #
    #         if mismatch:
    #             investigate_mismatch()

    # These must be 2d array, thus the , 2:3].T instead of , 2].T. [:, None, 2] also works
    zshift1 = set_mat1[:, None, 2].T if sym_entry.is_internal_tx1 else None
    zshift2 = set_mat2[:, None, 2].T if sym_entry.is_internal_tx2 else None

    log.debug(f'zshift1 = {zshift1}, zshift2 = {zshift2}, max_z_value={initial_z_value:2f}')
    optimal_tx = \
        OptimalTx.from_dof(sym_entry.external_dof, zshift1=zshift1, zshift2=zshift2, max_z_value=initial_z_value)

    # Get rotated Oligomer1 Ghost Fragment, Oligomer2 Surface Fragment guide coodinate pairs
    # in the same Euler rotational space bucket
    skip_transformation = kwargs.get('skip_transformation')
    if skip_transformation:
        transformation1 = unpickle(kwargs.get('transformation_file1'))
        full_rotation1, full_int_tx1, full_setting1, full_ext_tx1 = transformation1.values()
        transformation2 = unpickle(kwargs.get('transformation_file2'))
        full_rotation2, full_int_tx2, full_setting2, full_ext_tx2 = transformation2.values()
        # make arbitrary degen, rot, and tx counts
        degen_counts = [(idx, idx) for idx in range(1, len(full_rotation1) + 1)]
        rot_counts = [(idx, idx) for idx in range(1, len(full_rotation1) + 1)]
        tx_counts = list(range(1, len(full_rotation1) + 1))
    else:
        if sym_entry.unit_cell:
            full_ext_tx1, full_ext_tx2, full_optimal_ext_dof_shifts = [], [], []
        else:
            # optimal_ext_dof_shifts = list(repeat(None, number_passing_shifts))
            positive_indices = slice(None)  # slice by nothing, as None alone creates a new axis
            # final_passing_shifts = number_passing_shifts
            # stacked_external_tx1, stacked_external_tx2 = None, None
            full_ext_tx1, full_ext_tx2, full_optimal_ext_dof_shifts = None, None, None
        number_of_init_ghost = init_ghost_guide_coords1.shape[0]
        number_of_init_surf = init_surf_guide_coords2.shape[0]
        total_ghost_surf_combinations = number_of_init_ghost * number_of_init_surf
        # fragment_pairs = []
        rot_counts, degen_counts, tx_counts = [], [], []
        full_rotation1, full_rotation2, full_int_tx1, full_int_tx2 = [], [], [], []
        rotation_matrices1, rotation_matrices2 = rotation_matrices
        rotation_matrices_len1, rotation_matrices_len2 = rotation_matrices1.shape[0], rotation_matrices2.shape[0]
        number_of_rotations1, number_of_rotations2 = number_of_rotations
        number_of_degens1, number_of_degens2 = number_of_degens

        # Perform Euler integer extraction for all rotations
        init_translation_time_start = time.time()
        # Rotate Oligomer1 surface and ghost guide coordinates using rotation_matrices1 and set_mat1
        # Must add a new axis so that the multiplication is broadcast
        ghost_frag1_guide_coords_rot_and_set = \
            transform_coordinate_sets(init_ghost_guide_coords1[None, :, :, :],
                                      rotation=rotation_matrices1[:, None, :, :],
                                      rotation2=set_mat1[None, None, :, :])
        # Unstack the guide coords to be shape (N, 3, 3)
        # eulerint_ghost_component1_1, eulerint_ghost_component1_2, eulerint_ghost_component1_3 = \
        #     euler_lookup.get_eulint_from_guides(ghost_frag1_guide_coords_rot_and_set.reshape((-1, 3, 3)))
        eulerint_ghost_component1 = \
            euler_lookup.get_eulint_from_guides_as_array(ghost_frag1_guide_coords_rot_and_set.reshape((-1, 3, 3)))

        # Next, for component 2
        surf_frags2_guide_coords_rot_and_set = \
            transform_coordinate_sets(init_surf_guide_coords2[None, :, :, :],
                                      rotation=rotation_matrices2[:, None, :, :],
                                      rotation2=set_mat2[None, None, :, :])
        # Reshape with the first axis (0) containing all the guide coordinate rotations stacked
        eulerint_surf_component2 = \
            euler_lookup.get_eulint_from_guides_as_array(surf_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))
        # eulerint_surf_component2_1, eulerint_surf_component2_2, eulerint_surf_component2_3 = \
        #     euler_lookup.get_eulint_from_guides(surf_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))

        # Reshape the reduced dimensional eulerint_components to again have the number_of_rotations length on axis 0,
        # the number of init_guide_coords on axis 1, and the 3 euler intergers on axis 2
        stacked_surf_euler_int2 = eulerint_surf_component2.reshape((rotation_matrices_len2, -1, 3))
        stacked_ghost_euler_int1 = eulerint_ghost_component1.reshape((rotation_matrices_len1, -1, 3))

        # stacked_surf_euler_int2_1 = eulerint_surf_component2_1.reshape((rotation_matrices_len2, -1))
        # stacked_surf_euler_int2_2 = eulerint_surf_component2_2.reshape((rotation_matrices_len2, -1))
        # stacked_surf_euler_int2_3 = eulerint_surf_component2_3.reshape((rotation_matrices_len2, -1))
        # stacked_ghost_euler_int1_1 = eulerint_ghost_component1_1.reshape((rotation_matrices_len1, -1))
        # stacked_ghost_euler_int1_2 = eulerint_ghost_component1_2.reshape((rotation_matrices_len1, -1))
        # stacked_ghost_euler_int1_3 = eulerint_ghost_component1_3.reshape((rotation_matrices_len1, -1))
        # Todo resolve. Below uses eulerints
        for idx1 in range(min(rotation_matrices1.shape[0], 13)):  # rotation_matrices1.shape[0]):  # Todo remove min
            rot1_count = idx1 % number_of_rotations1 + 1
            degen1_count = idx1 // number_of_rotations1 + 1
            rot_mat1 = rotation_matrices1[idx1]
            rotation_ghost_euler_ints1 = stacked_ghost_euler_int1[idx1]
            for idx2 in range(min(rotation_matrices2.shape[0], 12)):  # rotation_matrices2.shape[0]):  # Todo remove min
                # Rotate Oligomer2 Surface and Ghost Fragment Guide Coordinates using rot_mat2 and set_mat2
                rot2_count = idx2 % number_of_rotations2 + 1
                degen2_count = idx2 // number_of_rotations2 + 1
                rot_mat2 = rotation_matrices2[idx2]

                log.info(f'***** OLIGOMER 1: Degeneracy {degen1_count} Rotation {rot1_count} | '
                         f'OLIGOMER 2: Degeneracy {degen2_count} Rotation {rot2_count} *****')

                euler_start = time.time()
                # euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
                #     euler_lookup.lookup_by_euler_integers(stacked_surf_euler_int2_1[idx2],
                #                                           stacked_surf_euler_int2_2[idx2],
                #                                           stacked_surf_euler_int2_3[idx2],
                #                                           stacked_ghost_euler_int1_1[idx1],
                #                                           stacked_ghost_euler_int1_2[idx1],
                #                                           stacked_ghost_euler_int1_3[idx1],
                #                                           )
                euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
                    euler_lookup.lookup_by_euler_integers_as_array(stacked_surf_euler_int2[idx2],
                                                                   rotation_ghost_euler_ints1)
                # # euler_lookup.lookup_by_euler_integers_as_array(eulerint_ghost_component2.reshape((number_of_rotations2, 1, 3)),
                # #                                                eulerint_surf_component1.reshape((number_of_rotations1, 1, 3)))
                # Todo resolve. eulerints

        # Todo resolve. Below uses guide coords
        # # for idx1 in range(rotation_matrices):
        # # Iterating over more than 2 rotation matrix sets becomes hard to program dynamically owing to the permutations
        # # of the rotations and the application of the rotation/setting to each set of fragment information. It would be a
        # # bit easier if the same logic that is applied to the following routines, (similarity matrix calculation) putting
        # # the rotation of the second set of fragment information into the setting of the first by applying the inverse
        # # rotation and setting matrices to the second (or third...) set of fragments. Forget about this for now
        # init_time_start = time.time()
        # for idx1 in range(rotation_matrices1.shape[0]):  # min(rotation_matrices1.shape[0], 5)):  # Todo remove min
        #     # Rotate Oligomer1 Surface and Ghost Fragment Guide Coordinates using rot_mat1 and set_mat1
        #     rot1_count = idx1 % number_of_rotations1 + 1
        #     degen1_count = idx1 // number_of_rotations1 + 1
        #     rot_mat1 = rotation_matrices1[idx1]
        #     ghost_guide_coords_rot_and_set1 = \
        #         transform_coordinate_sets(init_ghost_guide_coords1, rotation=rot_mat1, rotation2=set_mat1)
        #     # surf_guide_coords_rot_and_set1 = \
        #     #     transform_coordinate_sets(init_surf_guide_coords1, rotation=rot_mat1, rotation2=set_mat1)
        #
        #     for idx2 in range(rotation_matrices2.shape[0]):  # min(rotation_matrices2.shape[0], 5)):  # Todo remove min
        #         # Rotate Oligomer2 Surface and Ghost Fragment Guide Coordinates using rot_mat2 and set_mat2
        #         rot2_count = idx2 % number_of_rotations2 + 1
        #         degen2_count = idx2 // number_of_rotations2 + 1
        #         rot_mat2 = rotation_matrices2[idx2]
        #         surf_guide_coords_rot_and_set2 = \
        #             transform_coordinate_sets(init_surf_guide_coords2, rotation=rot_mat2, rotation2=set_mat2)
        #         # ghost_guide_coords_rot_and_set2 = \
        #         #     transform_coordinate_sets(init_ghost_guide_coords2, rotation=rot_mat2, rotation2=set_mat2)
        #
        #         log.info(f'***** OLIGOMER 1: Degeneracy {degen1_count} Rotation {rot1_count} | '
        #                  f'OLIGOMER 2: Degeneracy {degen2_count} Rotation {rot2_count} *****')
        #
        #         euler_start = time.time()
        #         # First returned variable has indices increasing 0,0,0,0,1,1,1,1,1,2,2,2,3,...
        #         # Second returned variable has indices increasing 2,3,4,14,...
        #         euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
        #             euler_lookup.check_lookup_table(surf_guide_coords_rot_and_set2,
        #                                             ghost_guide_coords_rot_and_set1)
        #         # euler_matched_ghost_indices_rev2, euler_matched_surf_indices_rev1 = \
        #         #     euler_lookup.check_lookup_table(ghost_guide_coords_rot_and_set2,
        #         #                                     surf_guide_coords_rot_and_set1)
                # Todo resolve. guide coords

                log.info(f'\tEuler Search Took: {time.time() - euler_start:8f}s for '
                         f'{total_ghost_surf_combinations} ghost/surf pairs')

                # Ensure pairs are similar between euler_matched_surf_indices2 and euler_matched_ghost_indices_rev2
                # by indexing the residue_numbers
                # forward_reverse_comparison_start = time.time()
                # # log.debug(f'Euler indices forward, index 0: {euler_matched_surf_indices2[:10]}')
                # forward_surface_numbers2 = init_surf_residue_numbers2[euler_matched_surf_indices2]
                # # log.debug(f'Euler indices forward, index 1: {euler_matched_ghost_indices1[:10]}')
                # forward_ghosts_numbers1 = init_ghost_residue_numbers1[euler_matched_ghost_indices1]
                # # log.debug(f'Euler indices reverse, index 0: {euler_matched_ghost_indices_rev2[:10]}')
                # reverse_ghosts_numbers2 = init_ghost_residue_numbers2[euler_matched_ghost_indices_rev2]
                # # log.debug(f'Euler indices reverse, index 1: {euler_matched_surf_indices_rev1[:10]}')
                # reverse_surface_numbers1 = init_surf_residue_numbers1[euler_matched_surf_indices_rev1]

                # Make an index indicating where the forward and reverse euler lookups have the same residue pairs
                # Important! This method only pulls out initial fragment matches that go both ways, i.e. component1
                # surface (type1) matches with component2 ghost (type1) and vice versa, so the expanded checks of
                # for instance the surface loop (i type 3,4,5) with ghost helical (i type 1) matches is completely
                # unnecessary during euler look up as this will never be included
                # Also, this assumes that the ghost fragment display is symmetric, i.e. 1 (i) 1 (j) 10 (K) has an
                # inverse transform at 1 (i) 1 (j) 230 (k) for instance

                # prior = 0
                # number_overlapping_pairs = euler_matched_ghost_indices1.shape[0]
                # possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)
                # # Residue numbers are in order for forward_surface_numbers2 and reverse_ghosts_numbers2
                # for residue in init_surf_residue_numbers2:
                #     # Where the residue number of component 2 is equal pull out the indices
                #     forward_index = np.flatnonzero(forward_surface_numbers2 == residue)
                #     reverse_index = np.flatnonzero(reverse_ghosts_numbers2 == residue)
                #     # Next, use residue number indices to search for the same residue numbers in the extracted pairs
                #     # The output array slice is only valid if the forward_index is the result of
                #     # forward_surface_numbers2 being in ascending order, which for check_lookup_table is True
                #     current = prior + forward_index.shape[0]
                #     possible_overlaps[prior:current] = \
                #         np.in1d(forward_ghosts_numbers1[forward_index], reverse_surface_numbers1[reverse_index])
                #     prior = current

                # # Use for residue number debugging
                # possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)

                # forward_ghosts_numbers1[possible_overlaps]
                # forward_surface_numbers2[possible_overlaps]

                # indexing_possible_overlap_time = time.time() - indexing_possible_overlap_start

                # number_of_successful = possible_overlaps.sum()
                # log.info(f'\tIndexing {number_overlapping_pairs * euler_matched_surf_indices2.shape[0]} '
                #          f'possible overlap pairs found only {number_of_successful} possible out of '
                #          f'{number_overlapping_pairs} (took {time.time() - forward_reverse_comparison_start:8f}s)')

                # Get optimal shift parameters for initial (Ghost Fragment, Surface Fragment) guide coordinate pairs
                # Take the boolean index of the indices
                # possible_ghost_frag_indices = euler_matched_ghost_indices1[possible_overlaps]
                # # possible_surf_frag_indices = euler_matched_surf_indices2[possible_overlaps]

                # reference_rmsds = init_ghost_rmsds1[possible_ghost_frag_indices]
                # passing_ghost_coords = ghost_guide_coords_rot_and_set1[possible_ghost_frag_indices]
                # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2[possible_overlaps]]
                # # Todo these are from Guides
                # passing_ghost_coords = ghost_guide_coords_rot_and_set1[euler_matched_ghost_indices1]
                # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2]
                # # Todo these are from Guides
                # Todo debug With EulerInteger calculation
                passing_ghost_coords = ghost_frag1_guide_coords_rot_and_set[idx1, euler_matched_ghost_indices1]
                # passing_ghost_coords = transform_coordinate_sets(init_ghost_guide_coords1[euler_matched_ghost_indices1],
                #                                                  rotation=rot_mat1, rotation2=set_mat1)
                passing_surf_coords = surf_frags2_guide_coords_rot_and_set[idx2, euler_matched_surf_indices2]
                # passing_surf_coords = transform_coordinate_sets(init_surf_guide_coords2[euler_matched_surf_indices2],
                #                                                 rotation=rot_mat2, rotation2=set_mat2)
                # Todo debug With EulerInteger calculation
                reference_rmsds = init_ghost_rmsds1[euler_matched_ghost_indices1]

                optimal_shifts_start = time.time()
                transform_passing_shifts = \
                    optimal_tx.solve_optimal_shifts(passing_ghost_coords, passing_surf_coords, reference_rmsds)
                optimal_shifts_time = time.time() - optimal_shifts_start

                pre_cluster_passing_shifts = transform_passing_shifts.shape[0]
                if pre_cluster_passing_shifts == 0:
                    # log.debug('Length %d' % len(optimal_shifts))
                    # log.debug('Shape %d' % transform_passing_shifts.shape[0])
                    log.info(f'\tNo transforms were found passing optimal shift criteria '
                             f'(took {optimal_shifts_time:8f}s)')
                    continue
                # elif cluster_translations:
                else:
                    # log.info(f'\tFound {pre_cluster_passing_shifts} before clustering')
                    # tx_params = np.vstack()
                    # tx_params = transform_passing_shifts
                    cluster_time_start = time.time()
                    translation_cluster = DBSCAN(eps=translation_epsilon, min_samples=min_matched).fit(transform_passing_shifts)
                    transform_passing_shifts = transform_passing_shifts[translation_cluster.labels_ != outlier]
                    number_passing_shifts = transform_passing_shifts.shape[0]
                    if pre_cluster_passing_shifts == 0:
                        # log.debug('Length %d' % len(optimal_shifts))
                        # log.debug('Shape %d' % transform_passing_shifts.shape[0])
                        log.info(f'\tNo transforms were found after clustering optimal shifts '
                                 f'(took {time.time() - cluster_time_start:8f}s)')
                        continue
                    else:
                        log.info(f'\tFound {number_passing_shifts} transforms after clustering from '
                                 f'{pre_cluster_passing_shifts} possible transforms (took '
                                 f'{time.time() - cluster_time_start:8f}s)')

                # else:  # Use all translations
                #     pass

                blank_vector = np.zeros((number_passing_shifts, 1), dtype=float)  # length is by column
                if sym_entry.unit_cell:
                    # must take the optimal_ext_dof_shifts and multiply the column number by the corresponding row
                    # in the sym_entry.group_external_dof
                    # optimal_ext_dof_shifts[0] scalar * sym_entry.group_external_dof[0] (1 row, 3 columns)
                    # repeat for additional DOFs
                    # then add all up within each row
                    # for a single DOF, multiplication won't matter as only one matrix element will be available
                    #
                    optimal_ext_dof_shifts = transform_passing_shifts[:, :sym_entry.n_dof_external]
                    optimal_ext_dof_shifts = np.hstack((optimal_ext_dof_shifts,
                                                        np.hstack((blank_vector,) * (3-sym_entry.n_dof_external))))
                    # ^ I think for the sake of cleanliness, I need to make this matrix
                    # must find positive indices before external_dof1 multiplication in case negatives there
                    positive_indices = \
                        np.flatnonzero(np.all(np.where(optimal_ext_dof_shifts < 0, False, True), axis=1) is True)
                    number_passing_shifts = positive_indices.shape[0]
                    # optimal_ext_dof_shifts[:, :, None] <- None expands the axis to make multiplication accurate
                    stacked_external_tx1 = \
                        (optimal_ext_dof_shifts[:, :, None] * sym_entry.external_dof1).sum(axis=-2)
                    stacked_external_tx2 = \
                        (optimal_ext_dof_shifts[:, :, None] * sym_entry.external_dof2).sum(axis=-2)
                    full_ext_tx1.append(stacked_external_tx1[positive_indices])
                    full_ext_tx2.append(stacked_external_tx2[positive_indices])
                    full_optimal_ext_dof_shifts.append(optimal_ext_dof_shifts[positive_indices])

                # Prepare the transformation parameters for storage in full transformation arrays
                # Use of [:, None] transforms the array into an array with each internal dof sored as a scalar in
                # axis 1 and each successive index along axis 0 as each passing shift
                internal_tx_params1 = transform_passing_shifts[:, None, sym_entry.n_dof_external] \
                    if sym_entry.is_internal_tx1 else blank_vector
                internal_tx_params2 = transform_passing_shifts[:, None, sym_entry.n_dof_external+1] \
                    if sym_entry.is_internal_tx2 else blank_vector
                # Stack each internal parameter along with a blank vector, this isolates the tx vector along z axis
                stacked_internal_tx_vectors1 = np.hstack((blank_vector, blank_vector, internal_tx_params1))
                stacked_internal_tx_vectors2 = np.hstack((blank_vector, blank_vector, internal_tx_params2))

                # Store transformation parameters, indexing only those that are positive in the case of lattice syms
                full_int_tx1.append(stacked_internal_tx_vectors1[positive_indices])
                full_int_tx2.append(stacked_internal_tx_vectors2[positive_indices])
                full_rotation1.append(np.tile(rot_mat1, (number_passing_shifts, 1, 1)))
                full_rotation2.append(np.tile(rot_mat2, (number_passing_shifts, 1, 1)))

                degen_counts.extend([(degen1_count, degen2_count) for _ in range(number_passing_shifts)])
                rot_counts.extend([(rot1_count, rot2_count) for _ in range(number_passing_shifts)])
                tx_counts.extend(list(range(1, number_passing_shifts + 1)))
                log.info(f'\tOptimal Shift Search Took: {optimal_shifts_time:8f}s for '
                         f'{euler_matched_ghost_indices1.shape[0]} guide coordinate pairs')
                log.info(f'\t{number_passing_shifts if number_passing_shifts else "No"} Initial Interface Fragment '
                         f'Match{"es" if number_passing_shifts != 1 else ""} Found')

                # # Todo remove debug
                # # tx_param_list = []
                # init_pass_ghost_numbers = init_ghost_residue_numbers1[possible_ghost_frag_indices]
                # init_pass_surf_numbers = init_surf_residue_numbers2[possible_surf_frag_indices]
                # for index in range(passing_ghost_coords.shape[0]):
                #     o = OptimalTxOLD(set_mat1, set_mat2, sym_entry.is_internal_tx1, sym_entry.is_internal_tx2,
                #                      reference_rmsds[index],
                #                      passing_ghost_coords[index], passing_surf_coords[index], sym_entry.external_dof)
                #     o.solve_optimal_shift()
                #     if o.get_zvalue() <= initial_z_value:
                #         # log.debug(f'overlap found at ghost/surf residue pair {init_pass_ghost_numbers[index]} | '
                #         #           f'{init_pass_surf_numbers[index]}')
                #         fragment_pairs.append((init_pass_ghost_numbers[index], init_pass_surf_numbers[index],
                #                                initial_ghost_frags1[possible_ghost_frag_indices[index]].guide_coords))
                #         all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExternalDOFShifts, OptimalInternalDOFShifts]
                #         tx_param_list.append(all_optimal_shifts)
                #
                # log.info(f'\t{len(tx_param_list) if tx_param_list else "No"} Initial Interface Fragment '
                #          f'Matches Found')
                # tx_param_list = np.array(tx_param_list)
                # log.debug(f'Equality of vectorized versus individual tx array: '
                #           f'{np.all(tx_param_list == transform_passing_shifts)}')
                # log.debug(f'ALLCLOSE Equality of vectorized versus individual tx array: '
                #           f'{np.allclose(tx_param_list, transform_passing_shifts)}')
                # check_forward_and_reverse(init_ghost_guide_coords1[possible_ghost_frag_indices],
                #                           [rot_mat1], stacked_internal_tx_vectors1,
                #                           init_surf_guide_coords2[euler_matched_surf_indices2[possible_overlaps]],
                #                           [rot_mat2], stacked_internal_tx_vectors2,
                #                           reference_rmsds)
                # # Todo remove debug

        log.info(f'Initial Optimal Translation search took {time.time() - init_translation_time_start:8f}s')
    ##############
    # Here represents an important break in the execution of this code. Vectorized scoring and clash testing!
    ##############
    if sym_entry.unit_cell:
        # Calculate the vectorized uc_dimensions
        full_uc_dimensions = sym_entry.get_uc_dimensions(np.concatenate(full_optimal_ext_dof_shifts, axis=0))
        full_ext_tx1 = np.concatenate(full_ext_tx1, axis=0)  # .sum(axis=-2)
        full_ext_tx2 = np.concatenate(full_ext_tx2, axis=0)  # .sum(axis=-2)
    # Todo uncomment below lines if use tile_transform in the reverse orientation
    #     full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
    else:
        full_uc_dimensions = None
    #     full_ext_tx_sum = None
    # fragment_pairs = np.array(fragment_pairs)
    # Make full, numpy vectorized transformations overwriting individual variables for memory management
    full_rotation1 = np.concatenate(full_rotation1, axis=0)
    full_rotation2 = np.concatenate(full_rotation2, axis=0)
    full_int_tx1 = np.concatenate(full_int_tx1, axis=0)
    full_int_tx2 = np.concatenate(full_int_tx2, axis=0)
    starting_transforms = len(full_int_tx1)
    # log.debug(f'shape of full_rotation1 {full_rotation1.shape}')
    # log.debug(f'shape of full_rotation2 {full_rotation2.shape}')
    # log.debug(f'shape of full_int_tx1 {full_int_tx1.shape}')
    # log.debug(f'shape of full_int_tx2 {full_int_tx2.shape}')

    # tile_transform1 = {'rotation': full_rotation2,
    #                    'translation': full_int_tx2[:, None, :],
    #                    'rotation2': set_mat2,
    #                    'translation2': full_ext_tx_sum[:, None, :] if full_ext_tx_sum is not None else None}  # invert translation
    # tile_transform2 = {'rotation': inv_setting1,
    #                    'translation': full_int_tx1[:, None, :] * -1,
    #                    'rotation2': full_inv_rotation1,
    #                    'translation2': None}
    # Find the clustered transformations to expedite search of ASU clashing
    # Todo
    #  can I use the cluster_transformation_pairs distance graph to provide feedback on other aspects of the dock?
    #  seems that I could use the distances to expedite clashing checks, especially for more time consuming expansion
    #  checks such as the full material...
    #  At some point, extracting the exact rotation degree from the rotation matrix and extracting translation params
    #  will provide the bounds around, what I believe will appear as docking "islands" where docks are possible,
    #  likely, and preferred. Searching these docks is far more important than just outputting the possible docks and
    #  their scores. These docking islands can subsequently be used to define a design potential that could be explored
    #  during design and can be minimized
    #  |
    #  Look at rmsd_z_score function output from T33 docks to see about timeing. Could the euler lookup be skipped
    #  if calculate overlap time is multiplied by the number of possible?
    #  UPDATE: It seems that skipping is slower for the number of fragments queried... Below measurement was wrong
    #  ||
    #  Timings on these from my improvement protocol shows about similar times to euler lookup and calculate overlap
    #  even with vastly different scales of the arrays. This ignores the fact that the overlap time uses a number of
    #  indexing steps including making the ij_match array formation, indexing against the ghost and surface arrays, the
    #  rmsd_reference construction
    #  Given the lookups sort of irrelevance to the scoring (given very poor alignment), I could remove that step
    #  if it interferes with differentiability

    clustering_start = time.time()
    # Must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
    transform_neighbor_tree, cluster = \
        cluster_transformation_pairs(dict(rotation=full_rotation1,
                                          translation=full_int_tx1[:, None, :],
                                          rotation2=set_mat1,
                                          translation2=None if full_ext_tx1 is None else full_ext_tx1[:, None, :]),
                                     dict(rotation=full_rotation2,
                                          translation=full_int_tx2[:, None, :],
                                          rotation2=set_mat2,
                                          translation2=None if full_ext_tx2 is None else full_ext_tx2[:, None, :]),
                                     minimum_members=min_matched)
    # cluster_representative_indices, cluster_labels = find_cluster_representatives(transform_neighbor_tree, cluster)
    # Todo?
    #  _, cluster_labels = find_cluster_representatives(transform_neighbor_tree, cluster)
    cluster_labels = cluster.labels_
    # log.debug(f'shape of cluster_labels: {cluster_labels.shape}')
    sufficiently_dense_indices = np.flatnonzero(cluster_labels != -1)
    number_of_dense_transforms = len(sufficiently_dense_indices)

    log.info(f'Found {starting_transforms} total transforms, {starting_transforms - number_of_dense_transforms} of '
             f'which are missing the minimum number of close transforms to be viable. {number_of_dense_transforms} '
             f'remain (took {time.time() - clustering_start:8f}s)')
    if not number_of_dense_transforms:  # There were no successful transforms
        log.warning(f'No viable transformations found. Terminating {building_blocks} docking')
        return
    # representative_labels = cluster_labels[cluster_representative_indices]

    ####################
    # Transform the oligomeric coords to query for clashes
    degen_counts, rot_counts, tx_counts = zip(*[(degen_counts[idx], rot_counts[idx], tx_counts[idx])
                                                for idx in sufficiently_dense_indices.tolist()])
    # Update the transformation array and counts with the sufficiently_dense_indices
    # fragment_pairs = fragment_pairs[sufficiently_dense_indices]
    full_rotation1 = full_rotation1[sufficiently_dense_indices]
    full_rotation2 = full_rotation2[sufficiently_dense_indices]
    full_int_tx1 = full_int_tx1[sufficiently_dense_indices]
    full_int_tx2 = full_int_tx2[sufficiently_dense_indices]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[sufficiently_dense_indices]
        full_ext_tx1 = full_ext_tx1[sufficiently_dense_indices]
        full_ext_tx2 = full_ext_tx2[sufficiently_dense_indices]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
    else:
        full_ext_tx_sum = None
    full_inv_rotation1 = np.linalg.inv(full_rotation1)
    inv_setting1 = np.linalg.inv(set_mat1)

    # Set up chunks of coordinate transforms for clash testing
    # Todo make a function to wrap memory errors into chunks
    check_clash_coords_start = time.time()
    memory_constraint = psutil.virtual_memory().available
    # assume each element is np.float64
    element_memory = 8  # where each element is np.float64
    guide_coords_elements = 9  # For a single guide coordinate with shape (3, 3)
    coords_multiplier = 2
    number_of_elements_available = memory_constraint / element_memory
    model_elements = prod(bb_cb_coords2.shape)
    total_elements_required = model_elements * number_of_dense_transforms
    # Start with the assumption that all tested clashes are clashing
    asu_clash_counts = np.ones(number_of_dense_transforms)
    clash_vect = [clash_dist]
    # The batch_length indicates how many models could fit in the allocated memory. Using floor division to get integer
    # Reduce scale by factor of divisor to be safe
    start_divisor = divisor = 16
    batch_length = int(number_of_elements_available // model_elements // start_divisor)
    while True:
        try:  # The next batch_length
            # The number_of_batches indicates how many iterations are needed to exhaust all models
            chunk_size = model_elements * batch_length
            number_of_batches = int(ceil(total_elements_required/chunk_size) or 1)  # Select at least 1
            # Todo make this for loop a function.
            #  test_fragdock_clashes(bb_cb_coords2, full_inv_rotation1, full_int_tx1, inv_setting1, full_rotation2,
            #                        full_int_tx2, set_mat2, full_ext_tx_sum)
            #   return asu_clash_counts
            tiled_coords2 = np.tile(bb_cb_coords2, (batch_length, 1, 1))
            for batch in range(number_of_batches):
                # Find the upper slice limiting it at a maximum of number_of_dense_transforms
                # upper = (batch + 1) * batch_length if batch + 1 != number_of_batches else number_of_dense_transforms
                # batch_slice = slice(batch * batch_length, upper)
                batch_slice = slice(batch * batch_length, (batch+1) * batch_length)
                # Set full rotation batch to get the length of the remaining transforms
                _full_rotation2 = full_rotation2[batch_slice]
                # Transform the coordinates
                number_of_transforms = _full_rotation2.shape[0]
                # Todo for performing broadcasting of this operation
                #  s_broad = np.matmul(tiled_coords2[None, :, None, :], _full_rotation2[:, None, :, :])
                #  produces a shape of (_full_rotation2.shape[0], tiled_coords2.shape[0], 1, 3)
                #  inverse_transformed_model2_tiled_coords = transform_coordinate_sets(transform_coordinate_sets()).squeeze()
                inverse_transformed_model2_tiled_coords = \
                    transform_coordinate_sets(
                        transform_coordinate_sets(tiled_coords2[:number_of_transforms],  # Slice ensures same size
                                                  rotation=_full_rotation2,
                                                  translation=full_int_tx2[batch_slice, None, :],
                                                  rotation2=set_mat2,
                                                  translation2=None if full_ext_tx_sum is None
                                                  else full_ext_tx_sum[batch_slice, None, :]),
                        rotation=inv_setting1,
                        translation=full_int_tx1[batch_slice, None, :] * -1,
                        rotation2=full_inv_rotation1[batch_slice],
                        translation2=None)
                # Check each transformed oligomer 2 coordinate set for clashing against oligomer 1
                asu_clash_counts[batch_slice] = \
                    [oligomer1_backbone_cb_tree.two_point_correlation(
                        inverse_transformed_model2_tiled_coords[idx],
                        clash_vect)[0] for idx in range(number_of_transforms)]
                # Save memory by dereferencing the arry before the next calculation
                del inverse_transformed_model2_tiled_coords

            log.critical(f'Successful execution with {divisor} using available memory of '
                         f'{memory_constraint} and batch_length of {batch_length}')
            # This is the number of total guide coordinates allowed in memory at this point...
            # Given calculation constraints, this will need to be reduced by at least 4 fold
            euler_divisor = 4
            euler_lookup_size_threshold = int(chunk_size / guide_coords_elements // coords_multiplier // euler_divisor)
            log.info(f'Given memory, the euler_lookup_size_threshold is: {euler_lookup_size_threshold}')
            break
        except np.core._exceptions._ArrayMemoryError:
            divisor = divisor*2
            batch_length = int(number_of_elements_available // model_elements // divisor)

    # asu_is_viable = np.where(asu_clash_counts.flatten() == 0)  # , True, False)
    # asu_is_viable = np.where(np.array(asu_clash_counts) == 0)
    # Find those indices where the asu_clash_counts is not zero (inverse of nonzero by using the array == 0)
    asu_is_viable = np.flatnonzero(asu_clash_counts == 0)
    number_non_clashing_transforms = len(asu_is_viable)
    log.info(f'Clash testing for All Oligomer1 and Oligomer2 (took {time.time() - check_clash_coords_start:8f}s) '
             f'found {number_non_clashing_transforms} viable ASU\'s out of {number_of_dense_transforms}')
    # input_ = input('Please confirm to continue protocol')

    if not number_non_clashing_transforms:  # There were no successful asus that don't clash
        log.warning(f'No viable asymmetric units. Terminating {building_blocks} docking')
        return

    # Update the transformation array and counts with the asu_is_viable indices
    degen_counts, rot_counts, tx_counts = zip(*[(degen_counts[idx], rot_counts[idx], tx_counts[idx])
                                                for idx in asu_is_viable.tolist()])
    # fragment_pairs = fragment_pairs[asu_is_viable]
    full_rotation1 = full_rotation1[asu_is_viable]
    full_rotation2 = full_rotation2[asu_is_viable]
    full_int_tx1 = full_int_tx1[asu_is_viable]
    full_int_tx2 = full_int_tx2[asu_is_viable]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[asu_is_viable]
        full_ext_tx1 = full_ext_tx1[asu_is_viable]
        full_ext_tx2 = full_ext_tx2[asu_is_viable]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1

    full_inv_rotation1 = full_inv_rotation1[asu_is_viable]

    # log.debug('Checking rotation and translation fidelity after removing non-viable asu indices')
    # check_forward_and_reverse(ghost_guide_coords1,
    #                           full_rotation1, full_int_tx1,
    #                           surf_guide_coords2,
    #                           full_rotation2, full_int_tx2,
    #                           ghost_rmsds1)

    #################
    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    # alternative route to measure clashes of each transform. Move copies of component2 to interact with model1 ORIGINAL
    int_cb_and_frags_start = time.time()
    # Transform the CB coords of oligomer 2 to each identified transformation
    # Transforming only surface frags will have large speed gains from not having to transform all ghosts
    inverse_transformed_model2_tiled_cb_coords = \
        transform_coordinate_sets(transform_coordinate_sets(np.tile(model2.cb_coords,
                                                                    (number_non_clashing_transforms, 1, 1)),
                                                            rotation=full_rotation2,
                                                            translation=full_int_tx2[:, None, :],
                                                            rotation2=set_mat2,
                                                            translation2=None if full_ext_tx_sum is None
                                                            else full_ext_tx_sum[:, None, :]),
                                  rotation=inv_setting1,
                                  translation=full_int_tx1[:, None, :] * -1,
                                  rotation2=full_inv_rotation1,
                                  translation2=None)

    # Transform the surface guide coords of oligomer 2 to each identified transformation
    # Makes a shape (full_rotations.shape[0], surf_guide_coords.shape[0], 3, 3)
    inverse_transformed_surf_frags2_guide_coords = \
        transform_coordinate_sets(transform_coordinate_sets(surf_guide_coords2[None, :, :, :],
                                                            rotation=full_rotation2[:, None, :, :],
                                                            translation=full_int_tx2[:, None, None, :],
                                                            rotation2=set_mat2[None, None, :, :],
                                                            translation2=None if full_ext_tx_sum is None
                                                            else full_ext_tx_sum[:, None, None, :]),
                                  rotation=inv_setting1[None, None, :, :],
                                  translation=full_int_tx1[:, None, None, :] * -1,
                                  rotation2=full_inv_rotation1[:, None, :, :],
                                  translation2=None)

    log.info(f'\tTransformation of all viable Oligomer 2 CB atoms and surface fragments took '
             f'{time.time() - int_cb_and_frags_start:8f}s')

    def update_pose_coords(idx):
        # Get contacting PDB 1 ASU and PDB 2 ASU
        copy_model_start = time.time()
        rot_mat1 = full_rotation1[idx]
        rot_mat2 = full_rotation2[idx]
        internal_tx_param1 = full_int_tx1[idx]
        internal_tx_param2 = full_int_tx2[idx]
        if sym_entry.unit_cell:
            external_tx_params1 = full_ext_tx1[idx]
            external_tx_params2 = full_ext_tx2[idx]
            # asu.space_group = sym_entry.resulting_symmetry
            uc_dimensions = full_uc_dimensions[idx]
        else:
            external_tx_params1, external_tx_params2 = None, None
            uc_dimensions = None

        specific_transformation1 = dict(rotation=rot_mat1, translation=internal_tx_param1,
                                        rotation2=set_mat1, translation2=external_tx_params1)
        specific_transformation2 = dict(rotation=rot_mat2, translation=internal_tx_param2,
                                        rotation2=set_mat2, translation2=external_tx_params2)
        specific_transformations = [specific_transformation1, specific_transformation2]

        # Set the next unit cell dimensions
        pose.uc_dimensions = uc_dimensions
        # pose = Pose.from_entities([entity.return_transformed_copy(**specific_transformations[idx])
        #                            for idx, model in enumerate(models) for entity in model.entities],
        #                           entity_names=entity_names, name='asu', log=log, sym_entry=sym_entry,
        #                           surrounding_uc=output_surrounding_uc, uc_dimensions=uc_dimensions,
        #                           ignore_clashes=True, rename_chains=True)  # pose_format=True,
        # ignore ASU clashes since already checked ^

        # Transform each starting coords to the candidate pose coords then update the Pose coords
        # log.debug(f'Transforming pose coordinates to the current docked configuration')
        new_coords = []
        for entity_idx, entity in enumerate(pose.entities):
            # log.debug(f'transform_indices[entity_idx]={transform_indices[entity_idx]}'
            #           f'entity_idx={entity_idx}')
            # tsnfmd = transform_coordinate_sets(entity_start_coords[entity_idx],
            #                                    **specific_transformations[transform_indices[entity_idx]])
            # log.debug(f'Equality of tsnfmd and original {np.allclose(tsnfmd, entity_start_coords[entity_idx])}')
            # log.debug(f'tsnfmd: {tsnfmd[:5]}')
            # log.debug(f'start_coords: {entity_start_coords[entity_idx][:5]}')
            new_coords.append(transform_coordinate_sets(entity_start_coords[entity_idx],
                                                        **specific_transformations[transform_indices[entity_idx]]))
        pose.coords = np.concatenate(new_coords)

        log.debug(f'\tCopy and Transform Oligomer1 and Oligomer2 (took {time.time() - copy_model_start:8f}s)')

        # # Check if design has any clashes when expanded
        # return pose.symmetric_assembly_is_clash()

    def perturb_transformation(idx, sequence_design: bool = True):
        # Stack each local perturbation up and multiply individual entity coords
        specific_transformation1 = dict(rotation=full_rotation_perturb1[idx], translation=full_int_tx_perturb1[idx, :, None, :],
                                        rotation2=set_mat1, translation2=full_ext_tx_perturb1[idx])  # [:, None, :])
        # Todo set up translation2 correct to handle None
        specific_transformation2 = dict(rotation=full_rotation_perturb2[idx], translation=full_int_tx_perturb2[idx, :, None, :],
                                        rotation2=set_mat2, translation2=full_ext_tx_perturb2[idx])  # [:, None, :])
        specific_transformations = [specific_transformation1, specific_transformation2]

        # Get batch size
        number_of_perturbations = full_rotation_perturb1[idx].shape[0]

        # Todo move this outside to own function?
        if sequence_design:
            new_coords = []
            for entity_idx, entity in enumerate(pose.entities):
                # Todo Need to tile the entity_bb_coords if operating like this
                # perturbed_bb_coords = transform_coordinate_sets(entity_bb_coords[entity_idx],
                #                                              **specific_transformations[transform_indices[entity_idx]])
                new_coords.append(transform_coordinate_sets(entity_bb_coords[entity_idx],
                                                            **specific_transformations[transform_indices[entity_idx]]))
            # Todo test this
            #  Stack the entity coordinates to make up a contiguous block for each pose
            #  If entity_bb_coords are stacked, then must concatenate along axis=1 or =2 to get full pose
            #  If entity_bb_coords aren't stacked (individually transformed), then axis=0 will work
            log.debug(f'new_coords.shape: {tuple([coords.shape for coords in new_coords])}')
            perturbed_bb_coords = np.concatenate(new_coords, axis=1)
            log.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')

            # Make each set of coordinates "symmetric"
            # Todo - This uses starting coords to symmetrize... Crystalline won't be right with external_translation
            symmetric_bb_coords = []
            for idx in range(perturbed_bb_coords.shape[0]):
                symmetric_bb_coords.append(pose.return_symmetric_coords(perturbed_bb_coords[idx]))

            # Let -1 fill in the pose length dimension with the number of residues
            # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
            # X = perturbed_bb_coords.reshape((number_of_perturbations, -1, 4, 3))
            symmetric_bb_coords = np.concatenate(symmetric_bb_coords)
            X = symmetric_bb_coords.reshape((number_of_perturbations, -1, 4, 3))
            log.debug(f'X.shape: {X.shape}')
            # Todo
            #  return X

        # def proteinmpnn_perturb_design(pose, X, model_name):  # Todo Make this pose compatible?
        #     mpnn_model = proteinmpnn_factory(model_name)
            # Design sequences
            parameters = pose.get_proteinmpnn_params()
            # # Disregard the X return (which would be used in the Pose) and use the stacked X from above
            parameters['X'] = X
            # # Create batches for ProteinMPNN sequence design task
            # # Without keyword argument "size=", X will be used to determine size of batch_parameters
            # Todo separate the batch and the torch.from_numpy()
            # batch_parameters = batch_proteinmpnn_input(size=1, **parameters)
            batch_parameters = proteinmpnn_to_device(mpnn_model.device, **parameters)
            parameters.update(batch_parameters)

            X = parameters.get('X', None)
            # X = torch.from_numpy(X).to(dtype=torch.float32, device=mpnn_model.device)
            S = parameters.get('S', None)
            chain_mask = parameters.get('chain_mask', None)
            chain_encoding = parameters.get('chain_encoding', None)
            residue_idx = parameters.get('residue_idx', None)
            mask = parameters.get('mask', None)
            omit_AAs_np = parameters.get('omit_AAs_np', None)
            bias_AAs_np = parameters.get('bias_AAs_np', None)
            residue_mask = parameters.get('chain_M_pos', None)
            omit_AA_mask = parameters.get('omit_AA_mask', None)
            pssm_coef = parameters.get('pssm_coef', None)
            pssm_bias = parameters.get('pssm_bias', None)
            pssm_multi = parameters.get('pssm_multi', None)
            pssm_log_odds_flag = parameters.get('pssm_log_odds_flag', None)
            pssm_log_odds_mask = parameters.get('pssm_log_odds_mask', None)
            pssm_bias_flag = parameters.get('pssm_bias_flag', None)
            tied_pos = parameters.get('tied_pos', None)
            tied_beta = parameters.get('tied_beta', None)
            bias_by_res = parameters.get('bias_by_res', None)

            decode_order = pose.generate_proteinmpnn_decode_order(to_device=mpnn_model.device)

            size = X.shape[0]

            log.debug(f'The mpnn_model.device is: {mpnn_model.device}')
            if mpnn_model.device == 'cpu':
                mpnn_memory_constraint = psutil.virtual_memory().available
                log.critical(f'The available cpu memory is: {mpnn_memory_constraint}')
            else:
                mpnn_memory_constraint, gpu_memory_total = torch.cuda.mem_get_info()
                log.critical(f'The available gpu memory is: {mpnn_memory_constraint}')

            element_memory = 4  # where each element is np.int/float32
            number_of_elements_available = mpnn_memory_constraint / element_memory
            model_elements = number_of_mpnn_model_parameters
            # Only get the shape of the first element in the batch
            model_elements += prod(X[0].shape)
            model_elements += prod(S[0].shape)
            model_elements += prod(chain_encoding[0].shape)
            model_elements += prod(residue_idx[0].shape)
            model_elements += prod(mask[0].shape)
            model_elements += prod(pssm_log_odds_mask[0].shape)
            model_elements += prod(tied_beta[0].shape)
            model_elements += prod(bias_by_res[0].shape)
            log.critical(f'The number of model_elements is: {model_elements}')
            total_elements_required = model_elements * size

            # The batch_length indicates how many models could fit in the allocated memory. Using floor division to get integer
            # Reduce scale by factor of divisor to be safe
            # Todo 256 doesn't work on 32 GB of RAM CPU
            # Todo 2048 doesn't work on 24 GB of RAM GPU
            start_divisor = divisor = 2048  # 256  # 128
            # batch_length = 10
            batch_length = int(number_of_elements_available // model_elements // start_divisor)
            log.critical(f'The number_of_elements_available is: {number_of_elements_available}')
            while True:
                log.critical(f'The batch_length is: {batch_length}')
                try:
                    chunk_size = model_elements * batch_length
                    # number_of_batches = int(ceil(size/batch_length) or 1)
                    number_of_batches = int(ceil(total_elements_required/chunk_size) or 1)  # Select at least 1
                    generated_sequences = []
                    probabilities = []
                    sequence_scores = []
                    for batch in range(number_of_batches):
                        batch_slice = slice(batch * batch_length, (batch+1) * batch_length)
                        # log.debug(f'X[batch_slice].shape: {X[batch_slice].shape}')
                        # log.debug(f'chain_mask[batch_slice].shape: {chain_mask[batch_slice].shape}')
                        # log.debug(f'mask[batch_slice].shape: {mask[batch_slice].shape}')
                        # log.debug(f'residue_mask[batch_slice].shape: {residue_mask[batch_slice].shape}')
                        # # When batches are sliced for multiple inputs
                        # sample_dict = mpnn_model.tied_sample(X[batch_slice], decode_order, S[batch_slice],
                        #                                      chain_mask[batch_slice],
                        #                                      chain_encoding[batch_slice], residue_idx[batch_slice],
                        #                                      mask[batch_slice],
                        #                                      temperature=design_temperature, omit_AAs_np=omit_AAs_np,
                        #                                      bias_AAs_np=bias_AAs_np,
                        #                                      chain_M_pos=residue_mask[batch_slice],
                        #                                      omit_AA_mask=omit_AA_mask[batch_slice],
                        #                                      pssm_coef=pssm_coef[batch_slice],
                        #                                      pssm_bias=pssm_bias[batch_slice],
                        #                                      pssm_multi=pssm_multi,
                        #                                      pssm_log_odds_flag=pssm_log_odds_flag,
                        #                                      pssm_log_odds_mask=pssm_log_odds_mask[batch_slice],
                        #                                      pssm_bias_flag=pssm_bias_flag,
                        #                                      tied_pos=tied_pos, tied_beta=tied_beta[batch_slice],
                        #                                      bias_by_res=bias_by_res[batch_slice])
                        # When batches are single input
                        sample_dict = mpnn_model.tied_sample(X[batch_slice], decode_order, S[None],
                                                             chain_mask[None],
                                                             chain_encoding[None], residue_idx[None],
                                                             mask[None],
                                                             temperature=design_temperature, omit_AAs_np=omit_AAs_np,
                                                             bias_AAs_np=bias_AAs_np,
                                                             chain_M_pos=residue_mask[None],
                                                             omit_AA_mask=omit_AA_mask[None],
                                                             pssm_coef=pssm_coef[None],
                                                             pssm_bias=pssm_bias[None],
                                                             pssm_multi=pssm_multi,
                                                             pssm_log_odds_flag=pssm_log_odds_flag,
                                                             pssm_log_odds_mask=pssm_log_odds_mask[None],
                                                             pssm_bias_flag=pssm_bias_flag,
                                                             tied_pos=tied_pos, tied_beta=tied_beta[None],
                                                             bias_by_res=bias_by_res[None])
                        # sample_dict = mpnn_model.tied_sample(X, decode_order, S, chain_mask, chain_encoding, residue_idx, mask,
                        #                                      temperature=design_temperature, omit_AAs_np=omit_AAs_np,
                        #                                      bias_AAs_np=bias_AAs_np, chain_M_pos=residue_mask,
                        #                                      omit_AA_mask=omit_AA_mask,
                        #                                      pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi,
                        #                                      pssm_log_odds_flag=pssm_log_odds_flag,
                        #                                      pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=pssm_bias_flag,
                        #                                      tied_pos=tied_pos, tied_beta=tied_beta,
                        #                                      bias_by_res=bias_by_res)
                        S_sample = sample_dict['S']
                        tied_decoding_order = sample_dict['decoding_order']
                        chain_residue_mask = chain_mask*residue_mask
                        chain_residue_mask = chain_residue_mask[batch_slice]
                        # # When batches are sliced for multiple inputs
                        # log_probs = mpnn_model(X[batch_slice], S_sample, mask[batch_slice], chain_residue_mask,
                        #                        residue_idx[batch_slice], chain_encoding[batch_slice],
                        #                        None,  # decode_order <- this argument is provided but with below args, is not used
                        #                        use_input_decoding_order=True, decoding_order=tied_decoding_order)
                        # When batches are single input
                        log_probs = mpnn_model(X[batch_slice], S_sample, mask[None], chain_residue_mask,
                                               residue_idx[None], chain_encoding[None], None,
                                               use_input_decoding_order=True, decoding_order=tied_decoding_order)
                        mask_for_loss = mask * chain_residue_mask
                        scores = score_sequences(S_sample, log_probs, mask_for_loss)
                        scores = scores.cpu().data.numpy()
                        sequence_scores.extend(scores)  # .tolist())
                        batch_probabilities = sample_dict['probs'].cpu().data.numpy()
                        probabilities.extend(batch_probabilities)  # .tolist())
                        for idx in range(batch_length):
                            numeric_sequence = S_sample[idx]
                            sequence = ''.join([mpnn_alphabet[residue_int] for residue_int in numeric_sequence])
                            generated_sequences.append(sequence)
                            # sequence_scores.append(scores[idx])

                    log.critical(f'Successful execution with {divisor} using available memory of '
                                 f'{memory_constraint} and batch_length of {batch_length}')
                    _input = input(f'Press enter to continue')
                    break
                except (RuntimeError, np.core._exceptions._ArrayMemoryError) as error:  # for (gpu, cpu)
                    raise error
                    # log.critical(f'Calculation failed with {divisor}.\n{error}\nTrying again...')
                    log.critical(f'{error}\nTrying again...')
                    divisor = divisor*2
                    batch_length = int(number_of_elements_available // model_elements // divisor)

            return numeric_to_sequence(generated_sequences), sequence_scores, probabilities
        else:
            new_coords = []
            for entity_idx, entity in enumerate(pose.entities):
                # Todo Need to tile the entity_bb_coords if operating like this
                new_coords.append(transform_coordinate_sets(entity_start_coords[entity_idx],
                                                            **specific_transformations[transform_indices[entity_idx]]))

            # Todo test this
            #  Stack the entity coordinates to make up a contiguous block for each pose
            #  If entity_bb_coords are stacked, then must concatenate along axis=1 or =2 to get full pose
            #  If entity_bb_coords aren't stacked (individually transformed), then axis=0 will work
            log.debug(f'new_coords.shape: {tuple([coords.shape for coords in new_coords])}')
            perturbed_coords = np.concatenate(new_coords, axis=1)
            log.debug(f'perturbed_coords.shape: {perturbed_coords.shape}')

            return perturbed_coords.reshape((number_of_perturbations, -1, 3))

    def output_pose(idx, sequence_design: bool = True):
        # Todo replace with PoseDirectory? Path object?
        # temp indexing on degen and rot counts
        # degen1_count, degen2_count = degen_counts[idx]
        # rot1_count, rot2_count = rot_counts[idx]
        # temp indexing on degen and rot counts
        degen_str = 'DEGEN_{}'.format('_'.join(map(str, degen_counts[idx])))
        rot_str = 'ROT_{}'.format('_'.join(map(str, rot_counts[idx])))
        tx_str = f'TX_{tx_counts[idx]}'  # translation idx
        # degen_subdir_out_path = os.path.join(outdir, degen_str)
        # rot_subdir_out_path = os.path.join(degen_subdir_out_path, rot_str)
        tx_dir = os.path.join(outdir, degen_str, rot_str, tx_str.lower())  # .lower() keeps original publication format
        os.makedirs(tx_dir, exist_ok=True)
        sampling_id = f'{degen_str}-{rot_str}-{tx_str}'
        pose_id = f'{building_blocks}-{sampling_id}'
        # Make directories to output matched fragment PDB files
        # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
        matching_fragments_dir = os.path.join(tx_dir, frag_dir)
        os.makedirs(matching_fragments_dir, exist_ok=True)
        high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
        low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')

        if sequence_design:
            pose.design_sequence()
        # Write ASU, Model1, Model2, and assembly files
        pose.set_contacting_asu(distance=cb_distance)
        if sym_entry.unit_cell:  # 2, 3 dimensions
            # asu = get_central_asu(asu, uc_dimensions, sym_entry.dimension)
            cryst_record = generate_cryst1_record(uc_dimensions, sym_entry.resulting_symmetry)
        else:
            cryst_record = None
        pose.write(out_path=os.path.join(tx_dir, asu_file_name), header=cryst_record)

        # Todo group by input model... not entities
        for entity in pose.entities:
            entity.write_oligomer(out_path=os.path.join(tx_dir, f'{entity.name}_{sampling_id}.pdb'))

        if output_assembly:
            if sym_entry.unit_cell:  # 2, 3 dimensions
                if output_surrounding_uc:
                    assembly_path = os.path.join(tx_dir, 'surrounding_unit_cells.pdb')
                else:
                    assembly_path = os.path.join(tx_dir, 'central_uc.pdb')
            else:  # 0 dimension
                assembly_path = os.path.join(tx_dir, 'expanded_assembly.pdb')
            pose.write(assembly=True, out_path=assembly_path, header=cryst_record, surrounding_uc=output_surrounding_uc)
        log.info(f'\tSUCCESSFUL DOCKED POSE: {tx_dir}')

        # Return the indices sorted by z_value in ascending order, truncated at the number of passing
        sorted_match_scores = match_score_from_z_value(sorted_z_scores)

        # These are indexed outside this function
        # overlap_ghosts = passing_ghost_indices[sorted_fragment_indices]
        # overlap_surf = passing_surf_indices[sorted_fragment_indices]

        sorted_int_ghostfrags: list[GhostFragment] = [complete_ghost_frags1[idx] for idx in overlap_ghosts]
        sorted_int_surffrags2: list[Residue] = [complete_surf_frags2[idx] for idx in overlap_surf]
        # For all matched interface fragments
        # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
        # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
        chid_resnum_scores_dict_model1, chid_resnum_scores_dict_model2 = {}, {}
        # Number of unique interface mono fragments matched
        unique_frags_info1, unique_frags_info2 = set(), set()
        res_pair_freq_info_list = []
        # Todo refactor this whole part below to use pose.something()... interface_metrics()?
        for frag_idx, (int_ghost_frag, int_surf_frag, match) in \
                enumerate(zip(sorted_int_ghostfrags, sorted_int_surffrags2, sorted_match_scores), 1):
            surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_frag.get_aligned_chain_and_residue
            surf_frag_chain2, surf_frag_central_res_num2 = int_surf_frag.get_aligned_chain_and_residue
            # Todo
            #  surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_residue.chain, int_ghost_residue.number
            #  surf_frag_chain2, surf_frag_central_res_num2 = int_surf_residue.chain, int_surf_residue.number

            covered_residues_model1 = [(surf_frag_chain1, surf_frag_central_res_num1 + j) for j in range(-2, 3)]
            covered_residues_model2 = [(surf_frag_chain2, surf_frag_central_res_num2 + j) for j in range(-2, 3)]
            # match = sorted_match_scores[frag_idx - 1]
            for k in range(ijk_frag_db.fragment_length):
                chain_resnum1 = covered_residues_model1[k]
                chain_resnum2 = covered_residues_model2[k]
                if chain_resnum1 not in chid_resnum_scores_dict_model1:
                    chid_resnum_scores_dict_model1[chain_resnum1] = [match]
                else:
                    chid_resnum_scores_dict_model1[chain_resnum1].append(match)

                if chain_resnum2 not in chid_resnum_scores_dict_model2:
                    chid_resnum_scores_dict_model2[chain_resnum2] = [match]
                else:
                    chid_resnum_scores_dict_model2[chain_resnum2].append(match)

            unique_frags_info1.add((surf_frag_chain1, surf_frag_central_res_num1))
            unique_frags_info2.add((surf_frag_chain2, surf_frag_central_res_num2))

            if match >= high_quality_match_value:
                matched_fragment_dir = high_quality_matches_dir
            else:
                matched_fragment_dir = low_quality_matches_dir

            os.makedirs(matched_fragment_dir, exist_ok=True)

            # if write_frags:  # write out aligned cluster representative fragment
            ghost_frag_rep = int_ghost_frag.representative.return_transformed_copy(**specific_transformation1)
            ghost_frag_rep.write(out_path=os.path.join(matched_fragment_dir,
                                                       'int_frag_i{}_j{}_k{}_{}.pdb'.format(
                                                           *int_ghost_frag.ijk, frag_idx)))
            z_value = z_value_from_match_score(match)
            ghost_frag_central_freqs = \
                dictionary_lookup(ijk_frag_db.info, int_ghost_frag.ijk).central_residue_pair_freqs
            # write out associated match information to frag_info_file
            write_frag_match_info_file(ghost_frag=int_ghost_frag, matched_frag=int_surf_frag,
                                       overlap_error=z_value, match_number=frag_idx,
                                       central_frequencies=ghost_frag_central_freqs,
                                       out_path=matching_fragments_dir, pose_id=pose_id)

            # Keep track of residue pair frequencies and match information
            res_pair_freq_info_list.append(FragMatchInfo(ghost_frag_central_freqs,
                                                         surf_frag_chain1, surf_frag_central_res_num1,
                                                         surf_frag_chain2, surf_frag_central_res_num2, z_value))

        # log.debug('Wrote Fragments to matching_fragments')
        # calculate weighted frequency for central residues and write weighted frequencies to frag_text_file
        weighted_seq_freq_info = SeqFreqInfo(res_pair_freq_info_list)
        weighted_seq_freq_info.write(os.path.join(matching_fragments_dir, frag_text_file))

        unique_matched_monofrag_count = len(unique_frags_info1) + len(unique_frags_info2)
        unique_total_monofrags_count = unique_interface_frag_count_model1 + unique_interface_frag_count_model2
        percent_of_interface_covered = unique_matched_monofrag_count / float(unique_total_monofrags_count)

        # Calculate Nanohedra Residue Level Summation Score
        res_lev_sum_score = 0
        for res_scores_list1 in chid_resnum_scores_dict_model1.values():
            n1 = 1
            res_scores_list_sorted1 = sorted(res_scores_list1, reverse=True)
            for sc1 in res_scores_list_sorted1:
                res_lev_sum_score += sc1 * (1 / float(n1))
                n1 = n1 * 2
        for res_scores_list2 in chid_resnum_scores_dict_model2.values():
            n2 = 1
            res_scores_list_sorted2 = sorted(res_scores_list2, reverse=True)
            for sc2 in res_scores_list_sorted2:
                res_lev_sum_score += sc2 * (1 / float(n2))
                n2 = n2 * 2

        # Write Out Docked Pose Info to docked_pose_info_file.txt
        write_docked_pose_info(tx_dir, res_lev_sum_score, high_qual_match_count, unique_matched_monofrag_count,
                               unique_total_monofrags_count, percent_of_interface_covered, rot_mat1, internal_tx_param1,
                               sym_entry.setting_matrix1, external_tx_params1, rot_mat2, internal_tx_param2,
                               sym_entry.setting_matrix2, external_tx_params2, cryst_record, model1.file_path,
                               model2.file_path, pose_id)

    # Use below instead of this until can TODO vectorize asu_interface_residue_processing
    # asu_interface_residues = \
    #     np.array([oligomer1_backbone_cb_tree.query_radius(inverse_transformed_model2_tiled_cb_coords[idx],
    #                                                       cb_distance)
    #               for idx in range(inverse_transformed_model2_tiled_cb_coords.shape[0])])

    # Full Interface Fragment Match
    # Gather the data for efficient querying of model1 and model2 interactions
    model1_cb_balltree = BallTree(model1.cb_coords)
    model1_cb_indices = model1.cb_indices
    model1_coords_indexed_residues = model1.coords_indexed_residues
    model2_cb_indices = model2.cb_indices
    model2_coords_indexed_residues = model2.coords_indexed_residues
    zero_counts = []
    # Whether the protocol should separate the expansion of coordinates and the measurement of fragment matches
    # overlap_only = True  # False  #
    # Save all the indices were matching fragments are identified
    interface_is_viable = []
    all_passing_ghost_indices = []
    all_passing_surf_indices = []
    all_passing_z_scores = []
    # Get residue number for all model1, model2 CB Pairs that interact within cb_distance
    for idx in range(number_non_clashing_transforms):
        # res_num1, res_num2, res_ghost_guide_coords = fragment_pairs[idx]
        # log.debug(f'\tInvestigating initial fragment pair {res_num1}:{res_num2} for interface potential')
        # overlap_residues1 = model1.get_residues(numbers=[res_num1])
        # overlap_residues2 = model2.get_residues(numbers=[res_num2])
        # res_ghost_guide_coords = fragment_pairs[idx][2]

        # query/contact pairs/isin  - 0.028367  <- I predict query is about 0.015
        # indexing guide_coords     - 0.000389
        # total get_int_frags_time  - 0.028756 s

        # indexing guide_coords     - 0.000389
        # Euler Lookup              - 0.008161 s for 71435 fragment pairs
        # Overlap Score Calculation - 0.000365 s for 2949 fragment pairs
        # Total Match time          - 0.008915 s

        # query                     - 0.000895 s <- 100 fold shorter than predict
        # contact pairs             - 0.019595
        # isin indexing             - 0.008992 s
        # indexing guide_coords     - 0.000438
        # get_int_frags_time        - 0.029920 s

        # indexing guide_coords     - 0.000438
        # Euler Lookup              - 0.005603 s for 35400 fragment pairs
        # Overlap Score Calculation - 0.000209 s for 887 fragment pairs
        # Total Match time          - 0.006250 s
        # model1_tnsfmd = model1.return_transformed_copy(rotation=full_rotation1[idx],
        #                                                translation=full_int_tx1[idx],
        #                                                rotation2=set_mat1,
        #                                                translation2=None)
        # model1_cb_balltree_tnsfmd = BallTree(model1_tnsfmd.cb_coords)
        # model2_query_tnsfmd = model1_cb_balltree_tnsfmd.query_radius(transformed_model2_tiled_cb_coords[idx], cb_distance)
        # contacting_pairs_tnsfmd = [(model1_coords_indexed_residues[model1_cb_indices[model1_idx]].number,
        #                             model2_coords_indexed_residues[model2_cb_indices[model2_idx]].number)
        #                            for model2_idx, model1_contacts in enumerate(model2_query_tnsfmd)
        #                            for model1_idx in model1_contacts]
        # interface_residue_numbers1_tnsfmd, interface_residue_numbers2_tnsfmd = \
        #     map(list, map(set, zip(*contacting_pairs_tnsfmd)))
        # ghost_indices_in_interface1_tnsfmd = \
        #     np.flatnonzero(np.in1d(ghost_residue_numbers1, interface_residue_numbers1_tnsfmd))
        # surf_indices_in_interface2_tnsfmd = \
        #     np.flatnonzero(np.in1d(surf_residue_numbers2, interface_residue_numbers2_tnsfmd, assume_unique=True))
        # log.debug(f'ghost_indices_in_interface1: {ghost_indices_in_interface1_tnsfmd[:10]}')
        # # log.debug(f'ghost_residue_numbers1[:10]: {ghost_residue_numbers1[:10]}')
        # log.debug(f'interface_residue_numbers1: {interface_residue_numbers1_tnsfmd}')
        # ghost_interface_residues_tnsfmd = set(ghost_residue_numbers1[ghost_indices_in_interface1_tnsfmd])
        # log.debug(f'ghost_residue_numbers1 in interface: {ghost_interface_residues_tnsfmd}')
        # log.debug(f'---------------')
        # log.debug(f'surf_indices_in_interface2: {surf_indices_in_interface2_tnsfmd[:10]}')
        # # log.debug(f'surf_residue_numbers2[:10]: {surf_residue_numbers2[:10]}')
        # log.debug(f'interface_residue_numbers2: {interface_residue_numbers2_tnsfmd}')
        # surf_interface_residues_tnsfmd = surf_residue_numbers2[surf_indices_in_interface2_tnsfmd]
        # log.debug(f'surf_residue_numbers2 in interface: {surf_interface_residues_tnsfmd}')

        # log.debug('Checking rotation and translation fidelity during interface fragment expansion')
        # check_forward_and_reverse(ghost_guide_coords1[ghost_indices_in_interface1_tnsfmd],
        #                           [full_rotation1[idx]], [full_int_tx1[idx]],
        #                           surf_guide_coords2[surf_indices_in_interface2_tnsfmd],
        #                           [full_rotation2[idx]], [full_int_tx2[idx]],
        #                           ghost_rmsds1[ghost_indices_in_interface1_tnsfmd])
        # log.debug('Checking rotation and translation fidelity no interface fragment expansion')
        # check_forward_and_reverse(ghost_guide_coords1,
        #                           [full_rotation1[idx]], [full_int_tx1[idx]],
        #                           surf_guide_coords2,
        #                           [full_rotation2[idx]], [full_int_tx2[idx]],
        #                           ghost_rmsds1)

        # log.debug(f'\n++++++++++++++++\n')
        int_frags_time_start = time.time()
        model2_query = model1_cb_balltree.query_radius(inverse_transformed_model2_tiled_cb_coords[idx], cb_distance)
        model1_cb_balltree_time = time.time() - int_frags_time_start

        contacting_pairs = [(model1_coords_indexed_residues[model1_cb_indices[model1_idx]].number,
                             model2_coords_indexed_residues[model2_cb_indices[model2_idx]].number)
                            for model2_idx, model1_contacts in enumerate(model2_query)
                            for model1_idx in model1_contacts]
        try:
            interface_residue_numbers1, interface_residue_numbers2 = map(list, map(set, zip(*contacting_pairs)))
        except ValueError:  # not enough values to unpack from interface containing no residues
            log.warning('Interface contains no residues')
            continue
        # These were interface_surf_frags and interface_ghost_frags
        # interface_ghost_indices1 = \
        #     np.concatenate([np.where(ghost_residue_numbers1 == residue) for residue in interface_residue_numbers1])
        # surf_indices_in_interface2 = \
        #     np.concatenate([np.where(surf_residue_numbers2 == residue) for residue in interface_residue_numbers2])

        # Find the indices where the fragment residue numbers are found the interface residue numbers
        is_in_index_start = time.time()
        # Since *_residue_numbers1/2 are the same index as the complete fragment arrays, these interface indices are the
        # same index as the complete guide coords and rmsds as well
        # Both residue numbers are one-indexed vv
        # Todo make ghost_residue_numbers1 unique -> unique_ghost_residue_numbers1
        #  index selected numbers against per_residue_ghost_indices 2d (number surface frag residues,
        ghost_indices_in_interface1 = \
            np.flatnonzero(np.isin(ghost_residue_numbers1, interface_residue_numbers1))
        surf_indices_in_interface2 = \
            np.flatnonzero(np.isin(surf_residue_numbers2, interface_residue_numbers2, assume_unique=True))
        # log.debug(f'ghost_indices_in_interface1: {ghost_indices_in_interface1[:10]}')
        # log.debug(f'ghost_residue_numbers1[:10]: {ghost_residue_numbers1[:10]}')
        # log.debug(f'interface_residue_numbers1: {interface_residue_numbers1}')
        # ghost_interface_residues = set(ghost_residue_numbers1[ghost_indices_in_interface1])
        # log.debug(f'ghost_residue_numbers1 in interface: {ghost_interface_residues}')
        # log.debug(f'')
        # log.debug(f'surf_indices_in_interface2: {surf_indices_in_interface2[:10]}')
        # log.debug(f'surf_residue_numbers2[:10]: {surf_residue_numbers2[:10]}')
        # log.debug(f'interface_residue_numbers2: {interface_residue_numbers2}')
        # surf_interface_residues = surf_residue_numbers2[surf_indices_in_interface2]
        # log.debug(f'surf_residue_numbers2 in interface: {surf_interface_residues}')
        # model2_surf_residues = model2.get_residues(numbers=surf_interface_residues)
        # surf_interface_residues_guide_coords = np.array([residue.guide_coords for residue in model2_surf_residues])
        # inverse_transformed_surf_interface_residues_guide_coords_ = \
        #     transform_coordinate_sets(transform_coordinate_sets(surf_interface_residues_guide_coords,
        #                                                         rotation=full_rotation2[idx],
        #                                                         translation=full_int_tx2[idx],
        #                                                         rotation2=set_mat2,
        #                                                         translation2=None if full_ext_tx_sum is None
        #                                                         else full_ext_tx_sum[idx]),
        #                               rotation=inv_setting1,
        #                               translation=full_int_tx1[idx] * -1,
        #                               rotation2=full_inv_rotation1[idx],
        #                               translation2=None)

        # log.debug('Checking rotation and translation fidelity during interface fragment expansion')
        # check_forward_and_reverse(ghost_guide_coords1[ghost_indices_in_interface1],
        #                           [full_rotation1[idx]], [full_int_tx1[idx]],
        #                           surf_interface_residues_guide_coords,
        #                           [full_rotation2[idx]], [full_int_tx2[idx]],
        #                           ghost_rmsds1[ghost_indices_in_interface1])

        is_in_index_time = time.time() - is_in_index_start
        all_fragment_match_time_start = time.time()
        # if idx % 2 == 0:
        # interface_ghost_frags = complete_ghost_frags1[interface_ghost_indices1]
        # interface_surf_frags = complete_surf_frags2[surf_indices_in_interface2]
        # int_ghost_guide_coords1 = ghost_guide_coords1[interface_ghost_indices1]
        # int_surf_frag_guide_coords = surf_guide_coords2[surf_indices_in_interface2]
        # int_trans_ghost_guide_coords = \
        #     transform_coordinate_sets(int_ghost_guide_coords1, rotation=rot_mat1, translation=internal_tx_param1,
        #                               rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
        # int_trans_surf_guide_coords2 = \
        #     transform_coordinate_sets(int_surf_frag_guide_coords, rotation=rot_mat2, translation=internal_tx_param2,
        #                               rotation2=sym_entry.setting_matrix2, translation2=external_tx_params2)

        unique_interface_frag_count_model1, unique_interface_frag_count_model2 = \
            ghost_indices_in_interface1.shape[0], surf_indices_in_interface2.shape[0]
        # get_int_frags_time = time.time() - int_frags_time_start
        # Todo reinstate this logging?
        # log.info(f'\tNewly formed interface contains {unique_interface_frag_count_model1} unique Fragments on Oligomer '
        #          f'1 from {len(interface_residue_numbers1)} Residues and '
        #          f'{unique_interface_frag_count_model2} on Oligomer 2 from {len(interface_residue_numbers2)} Residues '
        #          f'\n\t(took {get_int_frags_time:8f}s to to get interface fragments, including '
        #          f'{model1_cb_balltree_time:8f}s to query distances, {is_in_index_time:8f}s to index residue numbers)')

        # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Surface Fragment) guide coordinate pairs
        # in the same Euler rotational space bucket
        # DON'T think this is crucial! ###
        # log.debug(f'surface_transformed versus residues surface_transformed GUIDE COORDS equality: '
        #           f'{np.all(int_trans_surf_guide_coords2 == inverse_transformed_surf_interface_residues_guide_coords_)}')
        # surf_guide_coords2_ = surf_guide_coords2[surf_indices_in_interface2]
        # log.debug(f'Surf coords trans versus original equality: {np.all(int_trans_surf_guide_coords2 == surf_guide_coords2_)}')
        # int_euler_matching_ghost_indices1, int_euler_matching_surf_indices2 = \
        #     euler_lookup.check_lookup_table(int_trans_ghost_guide_coords, int_trans_surf_guide_coords2)
        int_ghost_shape = surf_indices_in_interface2.shape[0]
        int_surf_shape = ghost_indices_in_interface1.shape[0]
        maximum_number_of_pairs = int_surf_shape*int_ghost_shape
        # if maximum_number_of_pairs < euler_lookup_size_threshold:
        # Todo there may be memory leak by Pose objects sharing memory with persistent objects
        #  that prevent garbage collection and stay attached to the run
        # Skipping EulerLookup as it has issues with precision
        index_ij_pairs_start_time = time.time()
        ghost_indices_repeated = np.repeat(ghost_indices_in_interface1, int_ghost_shape)
        surf_indices_tiled = np.tile(surf_indices_in_interface2, int_surf_shape)
        ij_type_match = ij_type_match_lookup_table[ghost_indices_repeated, surf_indices_tiled]
        # DEBUG: If ij_type_match needs to be removed for testing
        # ij_type_match = np.array([True for _ in range(len(ij_type_match))])
        # Surface selecting
        # [0, 1, 3, 5, ...] with fancy indexing [0, 1, 5, 10, 12, 13, 34, ...]
        # log.debug('Euler lookup')
        # DON'T think this is crucial! ###
        # Todo Debug skipping EulerLookup to see if issues with precision
        possible_fragments_pairs = ghost_indices_repeated.shape[0]
        passing_ghost_indices = ghost_indices_repeated[ij_type_match]
        passing_surf_indices = surf_indices_tiled[ij_type_match]
        # else:  # Narrow candidates by EulerLookup
        #     log.warning(f'The interface size is too large ({maximum_number_of_pairs} maximum pairs). '
        #                 f'Trimming possible fragments by EulerLookup')
        #     eul_lookup_start_time = time.time()
        #     int_ghost_guide_coords1 = ghost_guide_coords1[ghost_indices_in_interface1]
        #     int_trans_surf_guide_coords2 = inverse_transformed_surf_frags2_guide_coords[idx, surf_indices_in_interface2]
        #     # Todo Debug skipping EulerLookup to see if issues with precision
        #     int_euler_matching_ghost_indices1, int_euler_matching_surf_indices2 = \
        #         euler_lookup.check_lookup_table(int_ghost_guide_coords1, int_trans_surf_guide_coords2)
        #     # log.debug(f'int_euler_matching_ghost_indices1: {int_euler_matching_ghost_indices1[:5]}')
        #     # log.debug(f'int_euler_matching_surf_indices2: {int_euler_matching_surf_indices2[:5]}')
        #     eul_lookup_time = time.time() - eul_lookup_start_time
        #
        #     # Find the ij_type_match which is the same length as the int_euler_matching indices
        #     # this has data type bool so indexing selects al original
        #     index_ij_pairs_start_time = time.time()
        #     ij_type_match = \
        #         ij_type_match_lookup_table[
        #             ghost_indices_in_interface1[int_euler_matching_ghost_indices1],
        #             surf_indices_in_interface2[int_euler_matching_surf_indices2]]
        #     possible_fragments_pairs = int_euler_matching_ghost_indices1.shape[0]
        #
        #     # Get only euler matching fragment indices that pass ij filter. Then index their associated coords
        #     passing_ghost_indices = int_euler_matching_ghost_indices1[ij_type_match]
        #     # passing_ghost_coords = int_trans_ghost_guide_coords[passing_ghost_indices]
        #     # passing_ghost_coords = int_ghost_guide_coords1[passing_ghost_indices]
        #     passing_surf_indices = int_euler_matching_surf_indices2[ij_type_match]
        #     # passing_surf_coords = int_trans_surf_guide_coords2[passing_surf_indices]

        # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        # Calculate match score for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        overlap_score_time_start = time.time()

        all_fragment_z_score = rmsd_z_score(ghost_guide_coords1[passing_ghost_indices],
                                            inverse_transformed_surf_frags2_guide_coords[idx, passing_surf_indices],
                                            ghost_rmsds1[passing_ghost_indices])
        # all_fragment_match = calculate_match(ghost_guide_coords1[passing_ghost_indices],
        #                                      inverse_transformed_surf_frags2_guide_coords[idx, passing_surf_indices],
        #                                      ghost_rmsds1[passing_ghost_indices])
        # all_fragment_match = calculate_match(np.tile(int_ghost_guide_coords1, (int_trans_surf_guide_coords2.shape[0], 1, 1)),
        #                                      np.tile(int_trans_surf_guide_coords2, (int_ghost_guide_coords1.shape[0], 1, 1)),
        #                                      np.tile(ghost_rmsds1[ghost_indices_in_interface1], int_trans_surf_guide_coords2.shape[0]))
        # log.debug(f'indexing rmsds equality: {np.all(ghost_rmsds1[ghost_indices_in_interface1[passing_ghost_indices]] == ghost_rmsds1[ghost_indices_in_interface1][passing_ghost_indices])}')
        # rmds_ = rmsd(int_ghost_guide_coords1[passing_ghost_indices],
        #              int_trans_surf_guide_coords2[passing_surf_indices])
        # interface_residues = model1.get_residues(numbers=ghost_interface_residues)
        # residue_based_guide_coords = [frag.guide_coords for residue in interface_residues for frag in
        #                               residue.ghost_fragments]

        # log.debug(f'interface_residues guide coords[:5]: {residue_based_guide_coords[:5]}')
        # # log.debug(f'residues guide coords to ghost_indexed_guide_coords equality: {np.all(residue_based_guide_coords[:5] == int_ghost_guide_coords1)}')
        # log.debug(f'residues guide coords to ghost_indexed_guide_coords equality: {np.all(residue_based_guide_coords == int_ghost_guide_coords1)}')
        # log.debug(f'int_ghost_guide_coords1[passing_ghost_indices][:5]: {int_ghost_guide_coords1[passing_ghost_indices][:5]}')
        # log.debug(f'int_trans_surf_guide_coords2[passing_surf_indices][:5]: {int_trans_surf_guide_coords2[passing_surf_indices][:5]}')
        # log.debug(f'RMSD calc: {rmds_[:5]}')
        # log.debug(f'RMSD reference: {ghost_rmsds1[ghost_indices_in_interface1[passing_ghost_indices]][:5]}')
        log.info(
            # f'\tEuler Lookup found {int_euler_matching_ghost_indices1.shape[0]} passing overlaps '
            #      f'(took {eul_lookup_time:8f}s) for '
            #      f'{unique_interface_frag_count_model1 * unique_interface_frag_count_model2} fragment pairs and '
            f'\tZ-score calculation took {time.time() - overlap_score_time_start:8f}s for '
            f'{passing_ghost_indices.shape[0]} successful ij type matches (indexing time '
            f'{overlap_score_time_start - index_ij_pairs_start_time:8f}s) from '
            f'{possible_fragments_pairs} possible fragment pairs')
        # log.debug(f'Found ij_type_match with shape {ij_type_match.shape}')
        # log.debug(f'And Data: {ij_type_match[:3]}')
        # log.debug(f'Found all_fragment_match with shape {all_fragment_match.shape}')
        # log.debug(f'And Data: {all_fragment_match[:3]}')
        # Todo KM thoroughly examined variable typing up to here 7/20/22
        # else:  # this doesn't seem to be as fast from initial tests
        #     # below bypasses euler lookup
        #     # 1
        #     # # this may be slower than just calculating all and not worrying about interface!
        #     # int_ij_matching_ghost1_indices = np.isin(ij_matching_ghost1_indices, interface_ghost_indices1)
        #     # int_ij_matching_surf2_indices = np.isin(ij_matching_surf2_indices, surf_indices_in_interface2)
        #     # typed_ghost1_coords = ghost_guide_coords1[int_ij_matching_ghost1_indices]
        #     # typed_surf2_coords = surf_guide_coords2[int_ij_matching_surf2_indices]
        #     # reference_rmsds = ghost_rmsds1[int_typed_ghost1_indices]
        #     # # 2
        #     # typed_ghost1_coords = ghost_guide_coords1[ij_matching_ghost1_indices]
        #     # typed_surf2_coords = surf_guide_coords2[ij_matching_surf2_indices]
        #     # reference_rmsds = ghost_rmsds1[ij_matching_ghost1_indices]
        #     # 3
        #     # first slice the table according to the interface residues
        #     # int_ij_lookup_table = \
        #     #     ij_type_match_lookup_table[interface_ghost_indices1[:, None], surf_indices_in_interface2]
        #     int_ij_lookup_table = np.logical_and(ij_type_match_lookup_table,
        #                                          (np.einsum('i, j -> ij', interface_ghost_indices1, surf_indices_in_interface2)))
        #     # axis 0 is ghost frag, 1 is surface frag
        #     # int_row_indices, int_column_indices = np.indices(int_ij_lookup_table.shape)  # row vary by ghost, column by surf
        #     # int_ij_matching_ghost1_indices = \
        #     #     row_indices[interface_ghost_indices1[:, None], surf_indices_in_interface2][int_ij_lookup_table]
        #     # int_ij_matching_surf2_indices = \
        #     #     column_indices[interface_ghost_indices1[:, None], surf_indices_in_interface2][int_ij_lookup_table]
        #     int_ij_matching_ghost1_indices = row_indices[int_ij_lookup_table]
        #     int_ij_matching_surf2_indices = column_indices[int_ij_lookup_table]
        #     # int_ij_matching_ghost1_indices = \
        #     #     (int_ij_lookup_table * np.arange(int_ij_lookup_table.shape[0]))[int_ij_lookup_table]
        #     # int_ij_matching_surf2_indices = \
        #     #     (int_ij_lookup_table * np.arange(int_ij_lookup_table.shape[1])[:, None])[int_ij_lookup_table]
        #     typed_ghost1_coords = ghost_guide_coords1[int_ij_matching_ghost1_indices]
        #     typed_surf2_coords = surf_guide_coords2[int_ij_matching_surf2_indices]
        #     reference_rmsds = ghost_rmsds1[int_ij_matching_ghost1_indices]
        #
        #     all_fragment_match = calculate_match(typed_ghost1_coords, typed_surf2_coords, reference_rmsds)

        # check if the pose has enough high quality fragment matches
        # high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
        high_qual_match_indices = np.flatnonzero(all_fragment_z_score <= high_quality_z_value)
        high_qual_match_count = len(high_qual_match_indices)
        all_fragment_match_time = time.time() - all_fragment_match_time_start
        # if high_qual_match_count == 0:
        #     passing_overlaps_indices = np.flatnonzero(all_fragment_match > 0.2)
        #     log.info('\t%d < %d however, %d fragments are considered passing (took %f s)'
        #              % (high_qual_match_count, min_matched, len(passing_overlaps_indices), all_fragment_match_time))
        #     tx_idx = tx_counts[idx]
        #     degen1_count, degen2_count = degen_counts[idx]
        #     rot1_count, rot2_count = rot_counts[idx]
        #     # temp indexing on degen and rot counts
        #     degen_subdir_out_path = os.path.join(outdir, 'DEGEN_%d_%d' % (degen1_count, degen2_count))
        #     rot_subdir_out_path = os.path.join(degen_subdir_out_path, 'ROT_%d_%d' % (rot1_count, rot2_count))
        #     tx_dir = os.path.join(rot_subdir_out_path, 'tx_%d' % tx_idx)  # idx)
        #     oligomers_dir = rot_subdir_out_path.split(os.sep)[-3]
        #     degen_dir = rot_subdir_out_path.split(os.sep)[-2]
        #     rot_dir = rot_subdir_out_path.split(os.sep)[-1]
        #     pose_id = '%s_%s_%s_TX_%d' % (oligomers_dir, degen_dir, rot_dir, tx_idx)
        #     sampling_id = '%s_%s_TX_%d' % (degen_dir, rot_dir, tx_idx)
        #     os.makedirs(tx_dir, exist_ok=True)
        #     # Make directories to output matched fragment PDB files
        #     # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
        #     matching_fragments_dir = os.path.join(tx_dir, frag_dir)
        #     os.makedirs(matching_fragments_dir, exist_ok=True)
        #     high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
        #     low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')
        #     assembly_path = os.path.join(tx_dir, 'surrounding_unit_cells.pdb')
        #     specific_transformation1 = {'rotation': rot_mat1, 'translation': internal_tx_param1,
        #                                 'rotation2': set_mat1, 'translation2': external_tx_params1}
        #     model1_copy = model1.return_transformed_copy(**specific_transformation1)
        #     model2_copy = model2.return_transformed_copy(**{'rotation': rot_mat2, 'translation': internal_tx_param2,
        #                                                 'rotation2': set_mat2, 'translation2': external_tx_params2})
        #     model1_copy.write(out_path=os.path.join(tx_dir, '%s_%s.pdb' % (model1_copy.name, sampling_id)))
        #     model2_copy.write(out_path=os.path.join(tx_dir, '%s_%s.pdb' % (model2_copy.name, sampling_id)))
        #     # cryst_record = generate_cryst1_record(asu.uc_dimensions, sym_entry.resulting_symmetry)
        #     # pose.write(assembly=True, out_path=assembly_path, header=cryst_record,
        #     #                          surrounding_uc=output_surrounding_uc)
        if high_qual_match_count < min_matched:
            log.debug(f'\t{high_qual_match_count} < {min_matched} Which is Set as the Minimal Required Amount of '
                      f'High Quality Fragment Matches (took {all_fragment_match_time:8f}s)')
            # Debug. Why are there no matches... cb_distance?
            if high_qual_match_count == 0:
                zero_counts.append(1)
            continue
        # elif overlap_only:
        else:
            # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
            # passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
            passing_overlaps_indices = np.flatnonzero(all_fragment_z_score <= low_quality_z_value)
            number_passing_overlaps = len(passing_overlaps_indices)
            log.info(f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} Matches Found'
                     f' in Complete Fragment Library (took {all_fragment_match_time:8f}s)')
            # Return the indices sorted by z_value in ascending order, truncated at the number of passing
            sorted_fragment_indices = np.argsort(all_fragment_z_score)[:number_passing_overlaps]
            # sorted_match_scores = match_score_from_z_value(sorted_z_values)
            # log.debug('Overlapping Match Scores: %s' % sorted_match_scores)
            # sorted_overlap_indices = passing_overlaps_indices[sorted_fragment_indices]
            # interface_ghost_frags = complete_ghost_frags1[interface_ghost_indices1][passing_ghost_indices[sorted_overlap_indices]]
            # interface_surf_frags = complete_surf_frags2[surf_indices_in_interface2][passing_surf_indices[sorted_overlap_indices]]
            # overlap_passing_ghosts = passing_ghost_indices[sorted_fragment_indices]
            all_passing_ghost_indices.append(passing_ghost_indices[sorted_fragment_indices])
            all_passing_surf_indices.append(passing_surf_indices[sorted_fragment_indices])
            all_passing_z_scores.append(all_fragment_z_score[sorted_fragment_indices])
            interface_is_viable.append(idx)
            log.debug(f'\tInterface fragment search time took {time.time() - int_frags_time_start:8f}')
            continue
        # else:
        #     write_and_quit = False
        #     report_residue_numbers = False
        #     update_pose_coords()

        # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
        # passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
        passing_overlaps_indices = np.flatnonzero(all_fragment_z_score <= low_quality_z_value)
        number_passing_overlaps = len(passing_overlaps_indices)
        # log.debug(f'low quality overlaps: {all_fragment_match[passing_overlaps_indices]}')
        # log.debug(f'low quality overlaps: {all_fragment_z_score[passing_overlaps_indices]}')
        log.info(f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} Matches Found in '
                 f'Complete Fragment Library (took {all_fragment_match_time:8f}s)')
        # except ValueError:
        #     pass

        # Get contacting PDB 1 ASU and PDB 2 ASU
        copy_model_start = time.time()
        rot_mat1 = full_rotation1[idx]
        rot_mat2 = full_rotation2[idx]
        internal_tx_param1 = full_int_tx1[idx]
        internal_tx_param2 = full_int_tx2[idx]
        if sym_entry.unit_cell:
            external_tx_params1 = full_ext_tx1[idx]
            external_tx_params2 = full_ext_tx2[idx]
            # asu.space_group = sym_entry.resulting_symmetry
            uc_dimensions = full_uc_dimensions[idx]
        else:
            external_tx_params1, external_tx_params2 = None, None
            uc_dimensions = None
        specific_transformation1 = dict(rotation=rot_mat1, translation=internal_tx_param1,
                                        rotation2=set_mat1, translation2=external_tx_params1)
        specific_transformation2 = dict(rotation=rot_mat2, translation=internal_tx_param2,
                                        rotation2=set_mat2, translation2=external_tx_params2)
        specific_transformations = [specific_transformation1, specific_transformation2]

        pose = Pose.from_entities([entity.return_transformed_copy(**specific_transformations[idx])
                                   for idx, model in enumerate(models) for entity in model.entities],
                                  entity_names=[entity.name for entity in model.entities for model in models],
                                  name='asu', log=log, sym_entry=sym_entry, uc_dimensions=uc_dimensions,
                                  ignore_clashes=True, rename_chains=True)  # pose_format=True,
        # ignore ASU clashes since already checked ^

        # # Transform using the indexing from the models to pull out the entity on the new pose
        # ent_idx = 0
        # for model_idx, model in enumerate(models):
        #     for _ in model.entities:
        #         pose.entities[ent_idx].transform(**specific_transformations[model_idx])
        #         ent_idx += 1
        # pose.set_symmetry(sym_entry=sym_entry, uc_dimensions=uc_dimensions)
        # pose.generate_symmetric_coords(surrounding_uc=output_surrounding_uc)

        log.info(f'\tCopy and Transform Oligomer1 and Oligomer2 (took {time.time() - copy_model_start:8f}s)')

        # Check if design has any clashes when expanded
        exp_des_clash_time_start = time.time()
        if pose.symmetric_assembly_is_clash():
            log.info(f'\tBackbone Clash when pose is expanded (took '
                     f'{time.time() - exp_des_clash_time_start:8f}s)')
            continue
        log.info(f'\tNO Backbone Clash when pose is expanded (took '
                 f'{time.time() - exp_des_clash_time_start:8f}s)')

        # Todo replace with PoseDirectory? Path object?
        # temp indexing on degen and rot counts
        # degen1_count, degen2_count = degen_counts[idx]
        # rot1_count, rot2_count = rot_counts[idx]
        # temp indexing on degen and rot counts
        degen_str = 'DEGEN_{}'.format('_'.join(map(str, degen_counts[idx])))
        rot_str = 'ROT_{}'.format('_'.join(map(str, rot_counts[idx])))
        tx_str = f'TX_{tx_counts[idx]}'  # translation idx
        # degen_subdir_out_path = os.path.join(outdir, degen_str)
        # rot_subdir_out_path = os.path.join(degen_subdir_out_path, rot_str)
        tx_dir = os.path.join(outdir, degen_str, rot_str, tx_str.lower())  # .lower() keeps original publication format
        os.makedirs(tx_dir, exist_ok=True)
        sampling_id = f'{degen_str}-{rot_str}-{tx_str}'
        pose_id = f'{building_blocks}-{sampling_id}'
        # Make directories to output matched fragment PDB files
        # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
        matching_fragments_dir = os.path.join(tx_dir, frag_dir)
        os.makedirs(matching_fragments_dir, exist_ok=True)
        high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
        low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')

        # Write ASU, Model1, Model2, and assembly files
        pose.set_contacting_asu(distance=cb_distance)
        if sym_entry.unit_cell:  # 2, 3 dimensions
            # asu = get_central_asu(asu, uc_dimensions, sym_entry.dimension)
            cryst_record = generate_cryst1_record(uc_dimensions, sym_entry.resulting_symmetry)
        else:
            cryst_record = None
        pose.write(out_path=os.path.join(tx_dir, asu_file_name), header=cryst_record)
        # pose.entities[0].write_oligomer(out_path=os.path.join(tx_dir, '%s_oligomer_asu.pdb' % entity2.name))
        # pose.entities[1].write_oligomer(out_path=os.path.join(tx_dir, '%s_oligomer_asu.pdb' % entity1.name))
        for entity in pose.entities:
            entity.write_oligomer(out_path=os.path.join(tx_dir, f'{entity.name}_{sampling_id}.pdb'))

        if output_assembly:
            # pose.generate_assembly_symmetry_models(surrounding_uc=output_surrounding_uc)
            if sym_entry.unit_cell:  # 2, 3 dimensions
                if output_surrounding_uc:
                    assembly_path = os.path.join(tx_dir, 'surrounding_unit_cells.pdb')
                    # pose.write(out_path=os.path.join(tx_dir, 'surrounding_unit_cells.pdb'),
                    #                          header=cryst_record, assembly=True, surrounding_uc=output_surrounding_uc)
                else:
                    assembly_path = os.path.join(tx_dir, 'central_uc.pdb')
            else:  # 0 dimension
                assembly_path = os.path.join(tx_dir, 'expanded_assembly.pdb')
                # pose.write(out_path=os.path.join(tx_dir, 'expanded_assembly.pdb'))
            pose.write(assembly=True, out_path=assembly_path, header=cryst_record, surrounding_uc=output_surrounding_uc)
        log.info(f'\tSUCCESSFUL DOCKED POSE: {tx_dir}')

        # Return the indices sorted by match value in descending order, truncated at the number of passing
        # sorted_fragment_indices = np.argsort(all_fragment_match)[:number_passing_overlaps:-1]
        # sorted_match_scores = all_fragment_match[sorted_fragment_indices]
        # Return the indices sorted by z_value in ascending order, truncated at the number of passing
        sorted_fragment_indices = np.argsort(all_fragment_z_score)[:number_passing_overlaps]
        sorted_match_scores = match_score_from_z_value(all_fragment_z_score[sorted_fragment_indices])
        # sorted_match_scores = match_score_from_z_value(sorted_z_values)
        # log.debug('Overlapping Match Scores: %s' % sorted_match_scores)
        # sorted_overlap_indices = passing_overlaps_indices[sorted_fragment_indices]
        # interface_ghost_frags = complete_ghost_frags1[interface_ghost_indices1][passing_ghost_indices[sorted_overlap_indices]]
        # interface_surf_frags = complete_surf_frags2[surf_indices_in_interface2][passing_surf_indices[sorted_overlap_indices]]
        overlap_ghosts = passing_ghost_indices[sorted_fragment_indices]
        # overlap_passing_ghosts = passing_ghost_indices[sorted_fragment_indices]
        overlap_surf = passing_surf_indices[sorted_fragment_indices]
        # overlap_passing_surf = passing_surf_indices[sorted_fragment_indices]
        # interface_ghost_frags = [complete_ghost_frags1[bool_idx] for bool_idx, bool_result in enumerate(interface_ghost_indices1)
        #                   if bool_result and bool_idx in overlap_passing_ghosts]
        # interface_surf_frags = [complete_surf_frags2[bool_idx] for bool_idx, bool_result in enumerate(surf_indices_in_interface2)
        #                   if bool_result and bool_idx in overlap_passing_surf]
        # interface_ghost_frags = complete_ghost_frags1[interface_ghost_indices1]
        # interface_surf_frags = complete_surf_frags2[surf_indices_in_interface2]
        sorted_int_ghostfrags: list[GhostFragment] = [complete_ghost_frags1[idx] for idx in overlap_ghosts]
        sorted_int_surffrags2: list[Residue] = [complete_surf_frags2[idx] for idx in overlap_surf]
        # For all matched interface fragments
        # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
        # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
        chid_resnum_scores_dict_model1, chid_resnum_scores_dict_model2 = {}, {}
        # Number of unique interface mono fragments matched
        unique_frags_info1, unique_frags_info2 = set(), set()
        res_pair_freq_info_list = []
        for frag_idx, (int_ghost_frag, int_surf_frag, match) in \
                enumerate(zip(sorted_int_ghostfrags, sorted_int_surffrags2, sorted_match_scores), 1):
            surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_frag.get_aligned_chain_and_residue
            surf_frag_chain2, surf_frag_central_res_num2 = int_surf_frag.get_aligned_chain_and_residue
            # Todo
            #  surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_residue.chain, int_ghost_residue.number
            #  surf_frag_chain2, surf_frag_central_res_num2 = int_surf_residue.chain, int_surf_residue.number

            covered_residues_model1 = [(surf_frag_chain1, surf_frag_central_res_num1 + j) for j in range(-2, 3)]
            covered_residues_model2 = [(surf_frag_chain2, surf_frag_central_res_num2 + j) for j in range(-2, 3)]
            # match = sorted_match_scores[frag_idx - 1]
            for k in range(ijk_frag_db.fragment_length):
                chain_resnum1 = covered_residues_model1[k]
                chain_resnum2 = covered_residues_model2[k]
                if chain_resnum1 not in chid_resnum_scores_dict_model1:
                    chid_resnum_scores_dict_model1[chain_resnum1] = [match]
                else:
                    chid_resnum_scores_dict_model1[chain_resnum1].append(match)

                if chain_resnum2 not in chid_resnum_scores_dict_model2:
                    chid_resnum_scores_dict_model2[chain_resnum2] = [match]
                else:
                    chid_resnum_scores_dict_model2[chain_resnum2].append(match)

            # if (surf_frag_chain1, surf_frag_central_res_num1) not in unique_frags_info1:
            unique_frags_info1.add((surf_frag_chain1, surf_frag_central_res_num1))
            # if (surf_frag_chain2, surf_frag_central_res_num2) not in unique_frags_info2:
            unique_frags_info2.add((surf_frag_chain2, surf_frag_central_res_num2))

            if match >= high_quality_match_value:
                matched_fragment_dir = high_quality_matches_dir
            else:
                matched_fragment_dir = low_quality_matches_dir

            os.makedirs(matched_fragment_dir, exist_ok=True)

            # if write_frags:  # write out aligned cluster representative fragment
            # # Old method
            # fragment, _ = dictionary_lookup(ijk_frag_db.paired_frags, int_ghost_frag.ijk)
            # trnsfmd_ghost_fragment = fragment.return_transformed_copy(*int_ghost_frag.transformation)
            # trnsfmd_ghost_fragment.transform(**specific_transformation1)
            # trnsfmd_ghost_fragment.write(
            #     out_path=os.path.join(matched_fragment_dir,
            #                           'int_frag_i{}_j{}_k{}_{}OLD.pdb'.format(
            #                               *int_ghost_frag.ijk, frag_idx)))
            # New Method
            ghost_frag_rep = int_ghost_frag.representative.return_transformed_copy(**specific_transformation1)
            ghost_frag_rep.write(out_path=os.path.join(matched_fragment_dir,
                                                       'int_frag_i{}_j{}_k{}_{}.pdb'.format(
                                                           *int_ghost_frag.ijk, frag_idx)))
            # transformed_ghost_fragment = int_ghost_frag.structure.return_transformed_copy(
            #     rotation=rot_mat1, translation=internal_tx_param1,
            #     rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            # transformed_ghost_fragment.write(os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
            #                                               % ('i%d_j%d_k%d' % int_ghost_frag.ijk, frag_idx)))
            z_value = z_value_from_match_score(match)
            ghost_frag_central_freqs = \
                dictionary_lookup(ijk_frag_db.info, int_ghost_frag.ijk).central_residue_pair_freqs
            # write out associated match information to frag_info_file
            write_frag_match_info_file(ghost_frag=int_ghost_frag, matched_frag=int_surf_frag,
                                       overlap_error=z_value, match_number=frag_idx,
                                       central_frequencies=ghost_frag_central_freqs,
                                       out_path=matching_fragments_dir, pose_id=pose_id)

            # Keep track of residue pair frequencies and match information
            res_pair_freq_info_list.append(FragMatchInfo(ghost_frag_central_freqs,
                                                         surf_frag_chain1, surf_frag_central_res_num1,
                                                         surf_frag_chain2, surf_frag_central_res_num2, z_value))

        # log.debug('Wrote Fragments to matching_fragments')
        # calculate weighted frequency for central residues and write weighted frequencies to frag_text_file
        weighted_seq_freq_info = SeqFreqInfo(res_pair_freq_info_list)
        weighted_seq_freq_info.write(os.path.join(matching_fragments_dir, frag_text_file))

        unique_matched_monofrag_count = len(unique_frags_info1) + len(unique_frags_info2)
        unique_total_monofrags_count = unique_interface_frag_count_model1 + unique_interface_frag_count_model2
        percent_of_interface_covered = unique_matched_monofrag_count / float(unique_total_monofrags_count)

        # Calculate Nanohedra Residue Level Summation Score
        res_lev_sum_score = 0
        for res_scores_list1 in chid_resnum_scores_dict_model1.values():
            n1 = 1
            res_scores_list_sorted1 = sorted(res_scores_list1, reverse=True)
            for sc1 in res_scores_list_sorted1:
                res_lev_sum_score += sc1 * (1 / float(n1))
                n1 = n1 * 2
        for res_scores_list2 in chid_resnum_scores_dict_model2.values():
            n2 = 1
            res_scores_list_sorted2 = sorted(res_scores_list2, reverse=True)
            for sc2 in res_scores_list_sorted2:
                res_lev_sum_score += sc2 * (1 / float(n2))
                n2 = n2 * 2

        # Write Out Docked Pose Info to docked_pose_info_file.txt
        write_docked_pose_info(tx_dir, res_lev_sum_score, high_qual_match_count, unique_matched_monofrag_count,
                               unique_total_monofrags_count, percent_of_interface_covered, rot_mat1, internal_tx_param1,
                               sym_entry.setting_matrix1, external_tx_params1, rot_mat2, internal_tx_param2,
                               sym_entry.setting_matrix2, external_tx_params2, cryst_record, model1.file_path,
                               model2.file_path, pose_id)

    log.info(f'Found {len(zero_counts)} zero counts')
    if not interface_is_viable:  # There were no successful transforms
        log.warning(f'No interfaces have enough fragment matches. Terminating {building_blocks} docking')
        return

    # Update the transformation array and counts with the interface_is_viable indices
    degen_counts, rot_counts, tx_counts = zip(*[(degen_counts[idx], rot_counts[idx], tx_counts[idx])
                                                for idx in interface_is_viable])
    full_rotation1 = full_rotation1[interface_is_viable]
    full_rotation2 = full_rotation2[interface_is_viable]
    full_int_tx1 = full_int_tx1[interface_is_viable]
    full_int_tx2 = full_int_tx2[interface_is_viable]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[interface_is_viable]
        full_ext_tx1 = full_ext_tx1[interface_is_viable]
        full_ext_tx2 = full_ext_tx2[interface_is_viable]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1

    entity_names = [entity.name for model in models for entity in model.entities]
    entity_bb_coords = [entity.backbone_coords for model in models for entity in model.entities]
    entity_start_coords = [entity.coords for model in models for entity in model.entities]
    transform_indices = {}
    entity_idx = 0
    for transform_idx, model in enumerate(models):
        for entity in model.entities:
            transform_indices[entity_idx] = transform_idx
            entity_idx += 1

    pose = Pose.from_entities([entity for idx, model in enumerate(models) for entity in model.entities],
                              entity_names=entity_names, name='asu', log=log, sym_entry=sym_entry,
                              surrounding_uc=output_surrounding_uc,
                              # uc_dimensions=uc_dimensions,
                              pose_format=True,
                              ignore_clashes=True, rename_chains=True)

    passing_symmetric_clashes = np.ones(len(interface_is_viable), dtype=bool)
    for idx, overlap_ghosts in enumerate(all_passing_ghost_indices):
        # log.info(f'Available memory: {psutil.virtual_memory().available}')
        # Load the z-scores and fragments
        # overlap_ghosts = all_passing_ghost_indices[idx]
        overlap_surf = all_passing_surf_indices[idx]
        sorted_z_scores = all_passing_z_scores[idx]
        # Find the pose
        exp_des_clash_time_start = time.time()
        update_pose_coords(idx)
        if pose.symmetric_assembly_is_clash():
            log.info(f'\tBackbone Clash when pose is expanded (took '
                     f'{time.time() - exp_des_clash_time_start:8f}s)')
            passing_symmetric_clashes[idx] = 0
        else:
            log.info(f'\tNO Backbone Clash when pose is expanded (took '
                     f'{time.time() - exp_des_clash_time_start:8f}s)')

    # Update the transformation array and counts with the passing_symmetric_clashes indices
    passing_symmetric_clashes = np.flatnonzero(passing_symmetric_clashes)
    number_passing_symmetric_clashes = passing_symmetric_clashes.shape[0]
    if number_passing_symmetric_clashes == 0:  # There were no successful transforms
        log.warning(f'No viable poses without symmetric clashes. Terminating {building_blocks} docking')
        return
    log.info(f'After symmetric clash testing, found {number_passing_symmetric_clashes} viable poses')

    degen_counts, rot_counts, tx_counts = zip(*[(degen_counts[idx], rot_counts[idx], tx_counts[idx])
                                                for idx in passing_symmetric_clashes.tolist()])
    all_passing_ghost_indices = [all_passing_ghost_indices[idx] for idx in passing_symmetric_clashes.tolist()]
    all_passing_surf_indices = [all_passing_surf_indices[idx] for idx in passing_symmetric_clashes.tolist()]
    all_passing_z_scores = [all_passing_z_scores[idx] for idx in passing_symmetric_clashes.tolist()]

    full_rotation1 = full_rotation1[passing_symmetric_clashes]
    full_rotation2 = full_rotation2[passing_symmetric_clashes]
    full_int_tx1 = full_int_tx1[passing_symmetric_clashes]
    full_int_tx2 = full_int_tx2[passing_symmetric_clashes]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[passing_symmetric_clashes]
        full_ext_tx1 = full_ext_tx1[passing_symmetric_clashes]
        full_ext_tx2 = full_ext_tx2[passing_symmetric_clashes]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1

    # Next, expand successful poses from coarse search of transformational space to randomly perturbed offset
    # This occurs by perturbing the transformation by a random small amount to generate transformational diversity from
    # the already identified solutions.
    if perturb_transformations:
        # Delta parameters
        internal_rot_perturb, internal_trans_perturb, external_trans_perturb = 1, 0.5, 0.5  # degrees, Angstroms, Angstroms
        perturb_number = 100
        grid_size = int(math.sqrt(perturb_number))  # Get the dimensions of the search
        # internal_rotations = get_rot_matrices(internal_rot_perturb/grid_size, rot_range_deg=internal_rotation)
        half_grid_range = int(grid_size/2)
        step_degrees = internal_rot_perturb/grid_size
        perturb_matrices = []
        for step in range(-half_grid_range, half_grid_range):  # Range from -5 to 4(5) for example. 0 is identity matrix
            rad = math.radians(step*step_degrees)
            rad_s = math.sin(rad)
            rad_c = math.cos(rad)
            perturb_matrices.append([[rad_c, -rad_s, 0.], [rad_s, rad_c, 0.], [0., 0., 1.]])

        perturb_matrices = np.array(perturb_matrices)
        internal_translations = external_translations = \
            np.linspace(-internal_trans_perturb, internal_trans_perturb, grid_size)
        if sym_entry.unit_cell:
            # Todo modify to search over 3 dof grid...
            raise NotImplementedError(f'Perturbation for lattice symmetries isn\'t working')
            external_translation_grid = np.repeat(external_translations, perturb_matrices.shape[0])
            internal_z_translation_grid = np.repeat(internal_translations, perturb_matrices.shape[0])
            internal_translation_grid = np.zeros((internal_z_translation_grid.shape[0], 3))
            internal_translation_grid[:, 2] = internal_z_translation_grid
            perturb_matrix_grid = np.tile(perturb_matrices, (internal_translations.shape[0], 1, 1))
            # Todo
            #  If the ext_tx are all 0 or not possible even if lattice, must not modify them. Need analogous check for
            #  is_ext_dof()
            full_ext_tx_perturb1 = full_ext_tx1[None, :, :] + external_translation_grid[:, None, :]
            full_ext_tx_perturb2 = full_ext_tx2[None, :, :] + external_translation_grid[:, None, :]
        else:
            internal_z_translation_grid = np.repeat(internal_translations, perturb_matrices.shape[0])
            internal_translation_grid = np.zeros((internal_z_translation_grid.shape[0], 3))
            internal_translation_grid[:, 2] = internal_z_translation_grid
            perturb_matrix_grid = np.tile(perturb_matrices, (internal_translations.shape[0], 1, 1))
            full_ext_tx_perturb1 = full_ext_tx_perturb2 = [None for _ in range(perturb_matrix_grid.shape[0])]

        # Apply the full perturbation landscape to the degrees of freedom
        # These operations add an axis to the transformation operators
        # Each transformation is along axis=0 and the perturbations are along axis=1
        if sym_entry.is_internal_rot1:
            # Ensure that the second matrix is transposed to dot multiply row s(mat1) by columns (mat2)
            full_rotation_perturb1 = np.matmul(full_rotation1[:, None, :, :],
                                               perturb_matrix_grid[None, :, :, :].swapaxes(-1, -2))
        else:  # Todo ensure that identity matrix is the length of internal_translation_grid
            full_rotation_perturb1 = np.matmul(full_rotation1[:, None, :, :], identity_matrix[None, None, :, :])
        if sym_entry.is_internal_rot2:
            full_rotation_perturb2 = np.matmul(full_rotation2[:, None, :, :],
                                               perturb_matrix_grid[None, :, :, :].swapaxes(-1, -2))
        else:
            full_rotation_perturb2 = np.matmul(full_rotation2[:, None, :, :], identity_matrix[None, None, :, :])

        # origin = np.array([0., 0., 0.])
        if sym_entry.is_internal_tx1:  # add the translation to Z (axis=2)
            full_int_tx_perturb1 = full_int_tx1[:, None, :] + internal_translation_grid[None, :, :]
        else:
            # full_int_tx1 is empty and adds the origin repeatedly.
            full_int_tx_perturb1 = full_int_tx1[:, None, :]  # + origin[None, None, :]

        if sym_entry.is_internal_tx2:
            full_int_tx_perturb2 = full_int_tx2[:, None, :] + internal_translation_grid[None, :, :]
        else:
            full_int_tx_perturb2 = full_int_tx2[:, None, :]  # + origin[None, None, :]

        log.info(f'internal_tx 1 shape: {full_int_tx_perturb1.shape}')
        log.info(f'internal_tx 2 shape: {full_int_tx_perturb2.shape}')

        # This will utilize a single input from each pose and create a sequence design batch over each transformation.
        for idx in range(full_rotation1.shape[0]):
            update_pose_coords(idx)
            batch_time_start = time.time()
            sequences, scores, probabilities = perturb_transformation(idx)  # Todo , sequence_design=design_output)
            print(sequences[:5])
            print(scores[:5])
            print(np.format_float_positional(np.float32(probabilities[0, 0, 0]), unique=False, precision=4))
            print(probabilities[:5])
            log.info(f'Batch design took {time.time() - batch_time_start:8f}s')
            # Todo
            #  coords = perturb_transformation(idx)  # Todo , sequence_design=design_output)

    # Write the resulting pose and sequences
    for idx, overlap_ghosts in enumerate(all_passing_ghost_indices):
        # Load the z-scores and fragments for use in output_pose()
        overlap_surf = all_passing_surf_indices[idx]
        sorted_z_scores = all_passing_z_scores[idx]
        # Todo must make the pose coordinates again as in the above call:
        #  clash = update_pose_coords(idx)
        output_pose(idx, sequence_design=design_output)

    log.info(f'Total {building_block} dock trajectory took {time.time() - frag_dock_time_start:.2f}s')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Parsing Command Line Input
        sym_entry_number, model1_path, model2_path, rot_step_deg1, rot_step_deg2, master_outdir, output_assembly, \
            output_surrounding_uc, min_matched, timer, initial, debug, high_quality_match_value, initial_z_value = \
            get_docking_parameters(sys.argv)

        # Master Log File
        master_log_filepath = os.path.join(master_outdir, master_log)
        if debug:
            # Root logs to stream with level debug
            logger = start_log(level=1)
            set_logging_to_debug()
            bb_logger = logger
            logger.debug('Debug mode. Generates verbose output. No writing to .log files will occur')
        else:
            # Root logger logs all emissions to a single file with level 'info'. Stream above still emits at 'warning'
            start_log(handler=2, location=master_log_filepath)
            # FragDock main logs to stream with level info
            logger = start_log(name=os.path.basename(__file__), propagate=True)
        # SymEntry Parameters
        symmetry_entry = symmetry_factory.get(sym_entry_number)  # sym_map inclusion?

        if initial:
            # make master output directory
            os.makedirs(master_outdir, exist_ok=True)
            logger.info('Nanohedra\nMODE: DOCK\n')
            write_docking_parameters(model1_path, model2_path, rot_step_deg1, rot_step_deg2, symmetry_entry,
                                     master_outdir, log=logger)
        else:  # for parallel runs, ensure that the first file was able to write before adding below log
            time.sleep(1)
        rot_step_deg1, rot_step_deg2 = \
            get_rotation_step(symmetry_entry, rot_step_deg1, rot_step_deg2, log=logger)

        model1_name = os.path.basename(os.path.splitext(model1_path)[0])
        model2_name = os.path.basename(os.path.splitext(model2_path)[0])
        logger.info('Docking %s / %s \n' % (model1_name, model2_name))

        # Create fragment database for all ijk cluster representatives
        # ijk_frag_db = unpickle(biological_fragment_db_pickle)
        # Todo parameterize when more available
        ijk_frag_db = fragment_factory(source=biological_interfaces)
        # Load Euler Lookup table for each instance
        euler_lookup = euler_factory()
        # ijk_frag_db = FragmentDB()
        #
        # # Get complete IJK fragment representatives database dictionaries
        # ijk_frag_db.get_monofrag_cluster_rep_dict()
        # ijk_frag_db.get_intfrag_cluster_rep_dict()
        # ijk_frag_db.get_intfrag_cluster_info_dict()

        try:
            # Output Directory  # Todo PoseDirectory
            building_blocks = '%s_%s' % (model1_name, model2_name)
            # outdir = os.path.join(master_outdir, building_blocks)
            # if not os.path.exists(outdir):
            #     os.makedirs(outdir)

            # log_file_path = os.path.join(outdir, '%s_log.txt' % building_blocks)
            # if os.path.exists(log_file_path):
            #     resume = True
            # else:
            #     resume = False
            # bb_logger = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False)
            # bb_logger.info('Found a prior incomplete run! Resuming from last sampled transformation.\n') \
            #     if resume else None

            # Write to Logfile
            # if not resume:
            #     bb_logger.info('DOCKING %s TO %s' % (model1_name, model2_name))
            #     bb_logger.info('Oligomer 1 Path: %s\nOligomer 2 Path: %s\n' % (model1_path, model2_path))

            nanohedra_dock(symmetry_entry, ijk_frag_db, euler_lookup, master_outdir, model1_path, model2_path,
                           rotation_step1=rot_step_deg1, rotation_step2=rot_step_deg2, min_matched=min_matched,
                           output_assembly=output_assembly, output_surrounding_uc=output_surrounding_uc,
                           keep_time=timer, high_quality_match_value=high_quality_match_value,
                           initial_z_value=initial_z_value)  # log=bb_logger,
            logger.info('COMPLETE ==> %s\n\n' % os.path.join(master_outdir, building_blocks))

        except KeyboardInterrupt:
            logger.info('\nRun Ended By KeyboardInterrupt\n')
            exit(2)
