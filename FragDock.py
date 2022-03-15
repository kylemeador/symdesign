import copy
import os
import sys
import time
from itertools import repeat

import numpy as np
from sklearn.neighbors import BallTree

from ClusterUtils import cluster_transformation_pairs, find_cluster_representatives
from PathUtils import frag_text_file, master_log, biological_fragment_db_pickle
from Structure import superposition3d, Structure
from SymDesignUtils import calculate_overlap, match_score_from_z_value, start_log, null_log, dictionary_lookup, \
    split_interface_numbers, calculate_match, z_value_from_match_score, unpickle
from utils.CmdLineArgParseUtils import get_docking_parameters
from utils.GeneralUtils import get_last_sampling_state, write_frag_match_info_file, write_docked_pose_info, \
    transform_coordinate_sets, get_rotation_step, write_docking_parameters, transform_coordinates
from utils.PDBUtils import get_contacting_asu, get_interface_residues
from utils.SymmetryUtils import generate_cryst1_record, get_central_asu
from classes.EulerLookup import EulerLookup
from classes.OptimalTx import OptimalTx
from classes.SymEntry import SymEntry, get_rot_matrices, get_degen_rotmatrices
from classes.WeightedSeqFreq import FragMatchInfo, SeqFreqInfo
from PDB import PDB
from Pose import Pose
# from interface_analysis.Database import FragmentDB

# Globals
logger = start_log(name=__name__)


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
        kdtree_oligomer1_backbone = BallTree(pdb1_copy.get_backbone_and_cb_coords())
        asu_cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.get_backbone_and_cb_coords(),
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
                                 if ghost_frag.get_aligned_chain_and_residue() in interface_chain_residues_pdb1]
        interface_surf_frags = [surf_frag for surf_frag in complete_surf_frags
                                if surf_frag.get_central_res_tup() in interface_chain_residues_pdb2]
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
        reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)  # ensure no division by 0
        all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
                                                 max_z_value=max_z_val)
        # log.debug('Checking all fragment overlap at interface')
        # get the passing_overlap indices and associated z-values by finding all indices where the value is not false
        passing_overlaps_indices = np.flatnonzero(all_fragment_overlap)  # .nonzero()[0]
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
        high_qual_match_count = np.where(passing_z_values < high_quality_match_value)[0].size
        if high_qual_match_count < min_matched:
            log.info('\t%d < %d Which is Set as the Minimal Required Amount of High Quality Fragment Matches'
                     % (high_qual_match_count, min_matched))
            continue

        # Get contacting PDB 1 ASU and PDB 2 ASU
        asu = get_contacting_asu(pdb1_copy, pdb2_copy, entity_names=[pdb1_copy.name, pdb2_copy.name])
        # log.debug('Grabbing asu')
        if not asu:  # _pdb_1 and not asu_pdb_2:
            log.info('\tNO Design ASU Found')
            continue

        # Check if design has any clashes when expanded
        exp_des_clash_time_start = time.time()
        asu.uc_dimensions = uc_dimensions
        asu.expand_matrices = sym_entry.expand_matrices
        symmetric_material = Pose.from_asu(asu, sym_entry=sym_entry, ignore_clashes=True, log=log)
        #                      surrounding_uc=output_surrounding_uc, ^ ignores ASU clashes
        exp_des_clash_time_stop = time.time()
        exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

        # log.debug('Checked expand clash')
        if symmetric_material.symmetric_assembly_is_clash():
            log.info('\tBackbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
            continue

        log.info('\tNO Backbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
        # Todo replace with DesignDirectory? Path object?
        tx_dir = os.path.join(rot_subdir_out_path, 'tx_%d' % tx_idx)
        oligomers_dir = rot_subdir_out_path.split(os.sep)[-3]
        degen_dir = rot_subdir_out_path.split(os.sep)[-2]
        rot_dir = rot_subdir_out_path.split(os.sep)[-1]
        pose_id = '%s_%s_%s_TX_%d' % (oligomers_dir, degen_dir, rot_dir, tx_idx)
        sampling_id = '%s_%s_TX_%d' % (degen_dir, rot_dir, tx_idx)
        if not os.path.exists(degen_subdir_out_path):
            os.makedirs(degen_subdir_out_path)
        if not os.path.exists(rot_subdir_out_path):
            os.makedirs(rot_subdir_out_path)
        if not os.path.exists(tx_dir):
            os.makedirs(tx_dir)

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
        asu.write(out_path=os.path.join(tx_dir, 'asu.pdb'), header=cryst1_record)
        pdb1_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb1_copy.name, sampling_id)))
        pdb2_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb2_copy.name, sampling_id)))

        if output_assembly:
            symmetric_material.get_assembly_symmetry_mates(surrounding_uc=output_surrounding_uc)
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
            surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_frag.get_aligned_chain_and_residue()
            surf_frag_chain2, surf_frag_central_res_num2 = int_surf_frag.get_central_res_tup()

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
            if z_value <= 1:  # the overlap z-value has is greater than 1 std deviation
                matched_fragment_dir = high_quality_matches_dir
            else:
                matched_fragment_dir = low_quality_matches_dir
            if not os.path.exists(matched_fragment_dir):
                os.makedirs(matched_fragment_dir)

            # if write_frags:  # write out aligned cluster representative fragment
            fragment, _ = dictionary_lookup(ijk_frag_db.paired_frags, int_ghost_frag.get_ijk())
            trnsfmd_ghost_fragment = fragment.return_transformed_copy(**int_ghost_frag.aligned_fragment.transformation)
            trnsfmd_ghost_fragment.transform(rotation=rot_mat1, translation=internal_tx_param1,
                                             rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            trnsfmd_ghost_fragment.write(out_path=os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
                                                               % ('i%d_j%d_k%d' % int_ghost_frag.get_ijk(), frag_idx + 1)))
            # transformed_ghost_fragment = int_ghost_frag.structure.return_transformed_copy(
            #     rotation=rot_mat1, translation=internal_tx_param1,
            #     rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            # transformed_ghost_fragment.write(os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
            #                                               % ('i%d_j%d_k%d' % int_ghost_frag.get_ijk(), frag_idx + 1)))

            ghost_frag_central_freqs = \
                dictionary_lookup(ijk_frag_db.info, int_ghost_frag.get_ijk()).central_residue_pair_freqs
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
                               sym_entry.setting_matrix2, external_tx_params2, cryst1_record, pdb1.filepath,
                               pdb2.filepath, pose_id)


def nanohedra_dock(sym_entry, ijk_frag_db, euler_lookup, master_outdir, pdb1, pdb2, rot_step_deg1=3, rot_step_deg2=3,
                   min_matched=3, high_quality_match_value=0.5, output_assembly=False, output_surrounding_uc=False,
                   clash_dist=2.2, init_max_z_val=1., subseq_max_z_val=2., log=None, keep_time=True):  # resume=False,
    """

    Keyword Args:
        high_quality_match_value=1 (int): when z-value used, 0.5 if match score is used
    Returns:
        None
    """
    def slice_variable_for_log(var, length=5):
        return var[:length]

    # Get Building Blocks
    if not isinstance(pdb1, Structure):
        pdb1 = PDB.from_file(pdb1)
    if not isinstance(pdb2, Structure):
        pdb2 = PDB.from_file(pdb2)

    if log is None:
        # Output Directory  # Todo DesignDirectory
        building_blocks = '%s_%s' % (pdb1.name, pdb2.name)
        outdir = os.path.join(master_outdir, building_blocks)
        os.makedirs(outdir, exist_ok=True)

        log_file_path = os.path.join(outdir, '%s_log.txt' % building_blocks)
        if os.path.exists(log_file_path):
            resume = True
        else:
            resume = False
        log = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False)

        # Write to Logfile
        if resume:
            log.info('Found a prior incomplete run! Resuming from last sampled transformation.\n')
        else:
            log.info('DOCKING %s TO %s\nOligomer 1 Path: %s\nOligomer 2 Path: %s\n\n'
                     % (pdb1.name, pdb2.name, pdb1.filepath, pdb2.filepath))
    pdb1.log = log
    pdb2.log = log
    #################################
    # # surface ghost frag overlap from the same oligomer scratch code
    # surffrags1 = pdb1.get_fragments(residue_numbers=pdb1.get_surface_residues())
    # ghost_frags_by_residue1 = [frag.get_ghost_fragments(ijk_frag_db.paired_frags, bb_cb_balltree, ijk_frag_db.info)
    #                            for frag in surffrags1]
    # ghost_frag_coords_by_residue1 = [ghost.guide_coords for residue in ghost_frags_by_residue1 for ghost in residue]
    # surface_frag_residue_numbers = [frag.central_residue.number for frag in surffrags1]  # also the residue indices
    # surface_frag_cb_coords = [residue.cb_coords for residue in pdb1.get_residues(numbers=surface_frag_residue_numbers)]
    # pdb1_surface_cb_ball_tree = BallTree(surface_frag_cb_coords)
    # residue_contact_query = pdb1_surface_cb_ball_tree.query(surface_frag_cb_coords, distance) <- what is distance?
    # contacting_pairs = [([surface_frag_residue_numbers[idx1]], [surface_frag_residue_numbers[idx2]])
    #                     for idx2 in range(residue_contact_query.size) for idx1 in residue_contact_query[idx2]]
    # asymmetric_contacting_pairs, found_pairs = [], []
    # for pair1, pair2 in contacting_pairs:
    #     # only add to contacting pair if we have never observed either
    #     if (pair1, pair2) not in found_pairs or (pair2, pair1) not in found_pairs and pair1 != pair2:
    #         asymmetric_contacting_pairs.append((pair1, pair2))
    #     # add both pair orientations (1, 2) or (2, 1) regardless
    #     found_pairs.extend([(pair1, pair2), (pair2, pair1)])
    # # Now we use the asymmetric_contacting_pairs to find the ghost_fragments for each residue, subtracting pair number
    # # by one to account for the zero indexed position
    # for idx1, idx2 in asymmetric_contacting_pairs:
    #     type_bool_matrix = is_frag_type_same(ghost_frags_by_residue1[idx1 - 1], ghost_frags_by_residue1[idx2 - 1],
    #                                          dtype='jj')
    #     # TODO decrease amount of work by saving each index array and reusing...
    #        such as stacking each j_index, guide_coords, rmsd, etc and pulling out by index
    #     def is_frag_type_same(ghost_frags1, ghost_frags2, dtype='ii'):
    #         frag1_indices = np.array([getattr(frag, '%s_type' % dtype[0]) for frag in frags1])
    #         frag2_indices = np.array([getattr(frag, '%s_type' % dtype[1]) for frag in frags2])
    #         np.where(frag1_indices_repeat == frag2_indices_tile)
    #         frag1_indices_repeated = np.repeat(frag1_indices, len(frag2_indices))
    #         frag2_indices_tiled = np.tile(frag2_indices, len(frag1_indices))
    #         match_lookup_table = np.where(frag1_indices_repeated == frag2_indices_tiled, True, False).reshape(
    #             len(frag1_indices), -1)
    #         return match_lookup_table
    #
    #     #   Fragment1
    #     # F T  F  F
    #     # R F  F  T
    #     # A F  F  F
    #     # G F  F  F
    #     # 2 T  T  F
    #     # use type_bool_matrix to guide RMSD calculation by pulling out the right ghost_corods
    #     ghost_coords_residue1 = [ghost_frags_by_residue1[idx].guide_coords
    #                              for idx, bool in enumerate(type_bool_matrix.flatten()) if bool]
    # DONE
    #     # have to find a way to iterate over each matrix rox/column with .flatten or other matrix iterator to pull out
    #     # necessary guide coordinate pairs
    # HERE v
    #     ij_matching_ghost1_indices = \
    #         (type_bool_matrix * np.arange(type_bool_matrix.shape[0]))[type_bool_matrix]
    #     ghost_coords_residue1 = ghost_frag_coords_by_residue1[ij_matching_ghost1_indices]
    #     ... same for residue2
    #     calculate_overlap(ghost_coords_residue1, ghost_coords_residue2, reference_rmsds,
    #                       max_z_value=max_z_val)
    #################################

    # Set up Building Block2
    pdb2_bb_cb_coords = pdb2.get_backbone_and_cb_coords()
    oligomer2_backbone_cb_tree = BallTree(pdb2_bb_cb_coords)

    # Get Surface Fragments With Guide Coordinates Using COMPLETE Fragment Database
    get_complete_surf_frags2_time_start = time.time()
    surf_frags2 = pdb2.get_fragments(residue_numbers=pdb2.get_surface_residues(), representatives=ijk_frag_db.reps)

    # calculate the initial match type by finding the predominant surface type
    frag_types2 = [monofrag2.i_type for monofrag2 in surf_frags2]
    fragment_content2 = [frag_types2.count(frag_type) for frag_type in range(1, ijk_frag_db.fragment_length + 1)]
    initial_surf_type2 = np.argmax(fragment_content2) + 1
    # initial_surf_frags2 = [monofrag2 for monofrag2 in surf_frags2 if monofrag2.i_type == initial_surf_type2]

    surf_frags2_guide_coords = np.array([surf_frag.guide_coords for surf_frag in surf_frags2])
    surf_frag2_residues = np.array([surf_frag.residue_number for surf_frag in surf_frags2])
    surf_frags2_i_indices = np.array([surf_frag.i_type for surf_frag in surf_frags2])
    init_surf_frag_indices2 = \
        [idx for idx, surf_frag in enumerate(surf_frags2) if surf_frag.i_type == initial_surf_type2]
    init_surf_frags2_guide_coords = surf_frags2_guide_coords[init_surf_frag_indices2]
    init_surf_frag2_residues = surf_frag2_residues[init_surf_frag_indices2]

    # log.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    # log.debug('Found initial fragment type: %d' % initial_surf_type2)
    get_complete_surf_frags2_time_stop = time.time()

    log.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    log.debug('init_surf_frags2_guide_coords: %s' % slice_variable_for_log(init_surf_frags2_guide_coords))
    log.debug('init_surf_frag2_residues: %s' % slice_variable_for_log(init_surf_frag2_residues))

    # Set up Building Block1
    oligomer1_backbone_cb_tree = BallTree(pdb1.get_backbone_and_cb_coords())

    get_complete_surf_frags1_time_start = time.time()
    surf_frags1 = pdb1.get_fragments(residue_numbers=pdb1.get_surface_residues(), representatives=ijk_frag_db.reps)

    # calculate the initial match type by finding the predominant surface type
    frag_types1 = [monofrag.i_type for monofrag in surf_frags1]
    fragment_content1 = [frag_types1.count(frag_type) for frag_type in range(1, ijk_frag_db.fragment_length + 1)]
    init_surf_type1 = np.argmax(fragment_content1) + 1
    init_surf_frags1 = [monofrag for monofrag in surf_frags1 if monofrag.i_type == init_surf_type1]
    init_surf_frags1_guide_coords = np.array([surf_frag.guide_coords for surf_frag in init_surf_frags1])
    init_surf_frag1_residues = np.array([surf_frag.residue_number for surf_frag in init_surf_frags1])
    # surf_frag1_residues = [surf_frag.residue_number for surf_frag in surf_frags1]

    # log.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    # log.debug('Found initial fragment type: %d' % initial_surf_type2)
    get_complete_surf_frags1_time_stop = time.time()
    log.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    log.debug('init_surf_frags2_guide_coords: %s' % slice_variable_for_log(init_surf_frags2_guide_coords))
    log.debug('init_surf_frag2_residues: %s' % slice_variable_for_log(init_surf_frag2_residues))
    log.debug('init_surf_frags1_guide_coords: %s' % slice_variable_for_log(init_surf_frags1_guide_coords))
    log.debug('init_surf_frag1_residues: %s' % slice_variable_for_log(init_surf_frag1_residues))

    if not resume and keep_time:
        get_complete_surf_frags2_time = get_complete_surf_frags2_time_stop - get_complete_surf_frags2_time_start
        get_complete_surf_frags1_time = get_complete_surf_frags1_time_stop - get_complete_surf_frags1_time_start
        log.info('Getting Oligomer 2 Surface Fragments and Guides Using COMPLETE Fragment Database (took %f s)\n'
                 % get_complete_surf_frags2_time)
        log.info('Getting Oligomer 1 Surface Fragments and Guides Using COMPLETE Fragment Database (took %f s)\n'
                 % get_complete_surf_frags1_time)

    # Get Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    get_complete_ghost_frags1_time_start = time.time()
    complete_ghost_frags1 = []
    for frag in surf_frags1:
        complete_ghost_frags1.extend(frag.get_ghost_fragments(ijk_frag_db.indexed_ghosts, oligomer1_backbone_cb_tree))
    # complete_ghost_frags1 = np.array(complete_ghost_frags1)
    ghost_frag1_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in complete_ghost_frags1])
    ghost_frag1_rmsds = np.array([ghost_frag.rmsd for ghost_frag in complete_ghost_frags1])
    ghost_frag1_rmsds = np.where(ghost_frag1_rmsds == 0, 0.01, ghost_frag1_rmsds)
    ghost_frag1_residues = np.array([ghost_frag.aligned_fragment.residue_number for ghost_frag in complete_ghost_frags1])
    ghost_frag1_j_indices = np.array([ghost_frag.j_type for ghost_frag in complete_ghost_frags1])
    init_ghost_frag_indices1 = \
        [idx for idx, ghost_frag in enumerate(complete_ghost_frags1) if ghost_frag.j_type == initial_surf_type2]
    init_ghost_frag1_guide_coords = ghost_frag1_guide_coords[init_ghost_frag_indices1]
    init_ghost_frag1_rmsds = ghost_frag1_rmsds[init_ghost_frag_indices1]
    init_ghost_frag1_residues = ghost_frag1_residues[init_ghost_frag_indices1]

    get_complete_ghost_frags1_time_stop = time.time()

    log.debug('ghost_frag1_j_indices: %s' % slice_variable_for_log(ghost_frag1_j_indices))
    log.debug('init_ghost_frag1_guide_coords: %s' % slice_variable_for_log(init_ghost_frag1_guide_coords))
    log.debug('init_ghost_frag1_rmsds: %s' % slice_variable_for_log(init_ghost_frag1_rmsds))
    log.debug('init_ghost_frag1_residues: %s' % slice_variable_for_log(init_ghost_frag1_residues))

    # Again for component 2
    get_complete_ghost_frags2_time_start = time.time()
    complete_ghost_frags2 = []
    for frag in surf_frags2:
        complete_ghost_frags2.extend(frag.get_ghost_fragments(ijk_frag_db.indexed_ghosts, oligomer2_backbone_cb_tree))
    init_ghost_frags2 = [ghost_frag for ghost_frag in complete_ghost_frags2 if ghost_frag.j_type == init_surf_type1]
    init_ghost_frag2_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in init_ghost_frags2])
    init_ghost_frag2_residues = \
        np.array([ghost_frag.aligned_fragment.residue_number for ghost_frag in init_ghost_frags2])
    # ghost_frag2_residues = [ghost_frag.aligned_residue.residue_number for ghost_frag in complete_ghost_frags2]

    get_complete_ghost_frags2_time_stop = time.time()
    log.debug('init_ghost_frag2_guide_coords: %s' % slice_variable_for_log(init_ghost_frag2_guide_coords))
    log.debug('init_ghost_frag2_residues: %s' % slice_variable_for_log(init_ghost_frag2_residues))
    # Prepare precomputed arrays for fast pair lookup
    # ghost1_residue_array = np.repeat(init_ghost_frag1_residues, len(init_surf_frag2_residues))
    # ghost2_residue_array = np.repeat(init_ghost_frag2_residues, len(init_surf_frag1_residues))
    # surface1_residue_array = np.tile(init_surf_frag1_residues, len(init_ghost_frag2_residues))
    # surface2_residue_array = np.tile(init_surf_frag2_residues, len(init_ghost_frag1_residues))
    # TODO use broadcasting to compute true/false instead of tiling (memory saving)
    ghost_frag1_j_indices_repeated = np.repeat(ghost_frag1_j_indices, len(surf_frags2_i_indices))
    surf_frags2_i_indices_tiled = np.tile(surf_frags2_i_indices, len(ghost_frag1_j_indices))
    # TODO keep as table or flatten? Can use one or the other as memory and take a view of the other as needed...
    ij_type_match_lookup_table = \
        np.where(ghost_frag1_j_indices_repeated == surf_frags2_i_indices_tiled, True, False).reshape(
            len(ghost_frag1_j_indices), -1)
    # axis 0 is ghost frag, 1 is surface frag
    # ij_matching_ghost1_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[0]))[ij_type_match_lookup_table]
    # ij_matching_surf2_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[1])[:, np.newaxis])[
    #         ij_type_match_lookup_table]
    row_indices, column_indices = np.indices(ij_type_match_lookup_table.shape)  # row index vary with ghost, column surf
    # row_indices = np.ma.MaskedArray(row_indices, mask=ij_type_match_lookup_table)
    # column_indices = np.ma.MaskedArray(column_indices, mask=ij_type_match_lookup_table)
    # Todo incorporate at the top
    #  ij_matching_ghost1_indices = row_indices[ij_type_match_lookup_table]
    #  ij_matching_surf2_indices = column_indices[ij_type_match_lookup_table]

    if not resume and keep_time:
        get_complete_ghost_frags1_time = get_complete_ghost_frags1_time_stop - get_complete_ghost_frags1_time_start
        get_complete_ghost_frags2_time = get_complete_ghost_frags2_time_stop - get_complete_ghost_frags2_time_start
        log.info('Getting %s Oligomer 1 Ghost Fragments and Guides Using COMPLETE Fragment Database (took %f s)\n'
                 % (pdb1.name, get_complete_ghost_frags1_time))
        log.info('Getting %s Oligomer 2 Ghost Fragments and Guides Using COMPLETE Fragment Database (took %f s)\n'
                 % (pdb2.name, get_complete_ghost_frags2_time))

    # Check if the job was running but stopped. Resume where last left off
    degen1_count, degen2_count, rot1_count, rot2_count = 0, 0, 0, 0
    if resume:
        degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log.handlers[0].baseFilename)

    if not resume:
        log.info('Obtaining Rotation/Degeneracy Matrices for Oligomer 1')

    # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
    rotation_matrices_1 = get_rot_matrices(rot_step_deg1, 'z', sym_entry.rotation_range1)
    degen_rot_mat_1 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_1, rotation_matrices_1)

    if not resume:
        log.info('Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n')

    # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
    rotation_matrices_2 = get_rot_matrices(rot_step_deg2, 'z', sym_entry.rotation_range2)
    degen_rot_mat_2 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_2, rotation_matrices_2)

    # Initialize Euler Lookup Class
    # euler_lookup = EulerLookup()

    set_mat1, set_mat2 = sym_entry.setting_matrix1, sym_entry.setting_matrix2
    # find superposition matrices to rotate setting matrix1 to setting matrix2 and vise versa
    # guide_coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    # asym_guide_coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
    # transformed_guide_coords1 = transform_coordinates(asym_guide_coords, rotation=set_mat1)
    # transformed_guide_coords2 = transform_coordinates(asym_guide_coords, rotation=set_mat2)
    # sup_rmsd, superposition_setting_1to2, sup_tx, _ = superposition3d(transformed_guide_coords2, transformed_guide_coords1)
    # superposition_setting_2to1 = np.linalg.inv(superposition_setting_1to2)
    # log.debug('sup_rmsd, superposition_setting_1to2, sup_tx: %s, %s, %s' % (sup_rmsd, superposition_setting_1to2, sup_tx))

    # these must be 2d array, thus the 2:3].T instead of 2. Using [:, 2][:, None] would also work
    zshift1 = set_mat1[:, 2:3].T if sym_entry.is_internal_tx1 else None
    zshift2 = set_mat2[:, 2:3].T if sym_entry.is_internal_tx2 else None

    log.debug('zshift1 = %s, zshift2 = %s, max_z_value=%f' % (str(zshift1), str(zshift2), init_max_z_val))
    optimal_tx = \
        OptimalTx.from_dof(sym_entry.ext_dof, zshift1=zshift1, zshift2=zshift2, max_z_value=init_max_z_val)

    # passing_optimal_shifts = []
    # degen_rot_counts = []
    # stacked_transforms1 = [None for _ in passing_optimal_shifts]
    full_rotation1, full_rotation2, full_int_tx1, full_int_tx2, full_setting1, \
    full_setting2, full_ext_tx1, full_ext_tx2, full_optimal_ext_dof_shifts = \
        [], [], [], [], [], [], [], [], []
    # stacked_transforms1, stacked_transforms2 = [], []
    for degen1 in degen_rot_mat_1[degen1_count:]:
        degen1_count += 1
        for rot_mat1 in degen1[rot1_count:]:
            rot1_count += 1
            # Rotate Oligomer1 Surface and Ghost Fragment Guide Coordinates using rot_mat1
            ghost_frag1_guide_coords_rot_and_set = \
                transform_coordinate_sets(init_ghost_frag1_guide_coords, rotation=rot_mat1, rotation2=set_mat1)
            surf_frags1_guide_coords_rot_and_set = \
                transform_coordinate_sets(init_surf_frags1_guide_coords, rotation=rot_mat1, rotation2=set_mat1)

            for degen2 in degen_rot_mat_2[degen2_count:]:
                degen2_count += 1
                for rot_mat2 in degen2[rot2_count:]:
                    rot2_count += 1
                    # Rotate Oligomer2 Surface and Ghost Fragment Guide Coordinates using rot_mat2
                    surf_frags2_guide_coords_rot_and_set = \
                        transform_coordinate_sets(init_surf_frags2_guide_coords, rotation=rot_mat2, rotation2=set_mat2)
                    ghost_frag2_guide_coords_rot_and_set = \
                        transform_coordinate_sets(init_ghost_frag2_guide_coords, rotation=rot_mat2, rotation2=set_mat2)

                    log.info('\n***** OLIGOMER 1: Degeneracy %d Rotation %d | OLIGOMER 2: Degeneracy %d Rotation %d '
                             '*****' % (degen1_count, rot1_count, degen2_count, rot2_count))

                    # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 (rotated) Surface Fragment)
                    # guide coodinate pairs in the same Euler rotational space bucket
                    log.info('Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational '
                             'space bucket')

                    euler_start = time.time()
                    # first returned variable has indices increasing 1,1,1,1,1,2,2,2,2,3,4,4,4,...
                    overlapping_surf_frags, overlapping_ghost_frags = \
                        euler_lookup.check_lookup_table(surf_frags2_guide_coords_rot_and_set,
                                                        ghost_frag1_guide_coords_rot_and_set)
                    overlapping_ghost_frags_rev, overlapping_surf_frags_rev = \
                        euler_lookup.check_lookup_table(ghost_frag2_guide_coords_rot_and_set,
                                                        surf_frags1_guide_coords_rot_and_set)
                    euler_time = time.time() - euler_start
                    number_overlapping_pairs = len(overlapping_ghost_frags)
                    log.debug('Number of matching euler angle pairs FORWARD: %d' % number_overlapping_pairs)
                    log.debug('Number of matching euler angle pairs REVERSE: %d' % len(overlapping_ghost_frags_rev))
                    # ensure pairs are similar between overlapping_ghost_frags and overlapping_ghost_frags_rev
                    # by indexing the ghost_frag_residues
                    log.info('Euler Search Took: %f s for %d ghost/surf pairs\n'
                             % (euler_time, len(init_ghost_frag1_residues) * len(init_surf_frag2_residues)))

                    forward_reverse_comparison_start = time.time()

                    forward_surface = init_surf_frag2_residues[overlapping_surf_frags]
                    forward_ghosts = init_ghost_frag1_residues[overlapping_ghost_frags]
                    reverse_ghosts = init_ghost_frag2_residues[overlapping_ghost_frags_rev]
                    reverse_surface = init_surf_frag1_residues[overlapping_surf_frags_rev]

                    # TODO
                    #  I need to flip the order by which the surface and ghost coords are passed for the reverse
                    #  operation so that they have the same oligomer1/2 incrementing index pattern as the first array
                    #  This way, sorting of the arrays is not necessary.
                    #  For residue number B on surface of 1, np.where(forward_ghosts == B)
                    #  For residue number B on surface of 1, np.where(reverse_surface == B)

                    # Make an index indicating where the forward and reverse euler lookups have the same residue pairs
                    # Important! This method only pulls out initial fragment matches that go both ways, i.e. component1
                    # surface (type1) matches with component2 ghost (type1) and vice versa, so the expanded checks of
                    # for instance the surface loop (i type 3,4,5) with ghost helical (i type 1) matches is completely
                    # unnecessary during euler look up as this will never be included
                    # Also, this assumes that the ghost fragment display is symmetric, i.e. 1 (i) 1 (j) 10 (K) has an
                    # inverse transform at 1 (i) 1 (j) 230 (k) for instance
                    # indexing_possible_overlap_start = time.time()
                    prior = 0
                    possible_overlaps = np.empty(number_overlapping_pairs, dtype=np.bool8)
                    # print('forward_surface:\n%s' % forward_surface.tolist())  # assuming a linear ordering...
                    # print('forward_ghosts:\n%s' % forward_ghosts.tolist())  # assuming a tiled ordering...
                    # residue numbers should be in order, surface fragments as well
                    for residue in init_surf_frag2_residues:
                        # print('Residue: %d' % residue)
                        # where the residue number of component 2 is equal pull out the indices
                        forward_index = np.where(forward_surface == residue)
                        reverse_index = np.where(reverse_ghosts == residue)
                        # indexed_forward_index = np.isin(forward_ghosts[forward_index], reverse_surface[reverse_index])
                        current = prior + len(forward_index[0])  # length of tuple index 0
                        # print(prior, current)
                        # next, use pulled out residue indices to search for overlapping numbers
                        possible_overlaps[prior:current] = \
                            np.isin(forward_ghosts[forward_index], reverse_surface[reverse_index])
                        # prior_prior = prior
                        prior = current
                    # print('f_surf', forward_surface[forward_index][:20])
                    # print('r_ghost', reverse_ghosts[reverse_index][:20])
                    # print('f_ghost', forward_ghosts[forward_index][:20])
                    # print('r_surf', reverse_surface[reverse_index][:20])
                    # print('possible', possible_overlaps[prior_prior:current][:20])
                    # forward_ghosts[possible_overlaps]
                    # forward_surface[possible_overlaps]

                    # indexing_possible_overlap_time = time.time() - indexing_possible_overlap_start

                    forward_reverse_comparison_time = time.time() - forward_reverse_comparison_start
                    log.info('Indexing %d possible overlap pairs found only %d possible out of %d (took %f s)\n'
                             % (len(overlapping_ghost_frags) * len(overlapping_ghost_frags_rev), possible_overlaps.sum()
                                , number_overlapping_pairs, forward_reverse_comparison_time))
                    # print('The number of euler overlaps (%d) is equal to the number of possible overlaps (%d)' %
                    #       (len(overlapping_ghost_frags), len(possible_overlaps)))
                    # Get optimal shift parameters for initial (Ghost Fragment, Surface Fragment) guide coordinate pairs
                    log.info('Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide '
                             'coordinate pairs')
                    # if rot2_count % 2 == 0:
                    possible_ghost_frag_indices = overlapping_ghost_frags[possible_overlaps]  # bool index the indices
                    possible_surf_frag_indices = overlapping_surf_frags[possible_overlaps]
                    # else:
                    #     possible_ghost_frag_indices = overlapping_ghost_frags
                    #     possible_surf_frag_indices = overlapping_surf_frags
                    # passing_ghost_coords = ghost_frag1_guide_coords_rot_and_set[overlapping_ghost_frags]
                    # passing_surf_coords = surf_frags2_guide_coords_rot_and_set[overlapping_surf_frags]
                    # reference_rmsds = init_ghost_frag1_rmsds[overlapping_ghost_frags]

                    passing_ghost_coords = ghost_frag1_guide_coords_rot_and_set[possible_ghost_frag_indices]
                    passing_surf_coords = surf_frags2_guide_coords_rot_and_set[possible_surf_frag_indices]
                    reference_rmsds = init_ghost_frag1_rmsds[possible_ghost_frag_indices]

                    optimal_shifts_start = time.time()
                    # if rot2_count % 2 == 0:
                    #     optimal_shifts = [optimal_tx.solve_optimal_shift(ghost_coords, passing_surf_coords[idx],
                    #                                                      reference_rmsds[idx])
                    #                       for idx, ghost_coords in enumerate(passing_ghost_coords)]
                    #     transform_passing_shifts = np.array([shift for shift in optimal_shifts if shift is not None])
                    # else:
                    transform_passing_shifts = \
                        optimal_tx.solve_optimal_shifts(passing_ghost_coords, passing_surf_coords, reference_rmsds)
                    optimal_shifts_time = time.time() - optimal_shifts_start
                    # transform_passing_shifts = [shift for shift in optimal_shifts if shift is not None]
                    # passing_optimal_shifts.extend(transform_passing_shifts)
                    # degen_rot_counts.extend(repeat((degen1_count, degen2_count, rot1_count, rot2_count),
                    #                                number_passing_shifts))
                    # for idx, tx_parameters in enumerate(transform_passing_shifts, 1):

                    number_passing_shifts = len(transform_passing_shifts)
                    if number_passing_shifts == 0:
                        # log.debug('Length %d' % len(optimal_shifts))
                        # log.debug('Shape %d' % transform_passing_shifts.shape[0])
                        log.info('No transforms were found passing optimal shift criteria (took %f s)'
                                 % optimal_shifts_time)
                        continue

                    # transform_passing_shift_indices = np.array(
                    #     [idx for idx, shift in enumerate(optimal_shifts) if shift is not None])

                    # if rot2_count % 2 == 0:
                    #     # print('***** possible overlap indices:', np.where(possible_overlaps == True)[0].tolist())
                    #     log.debug('Passing shift ghost residue pairs: %s' % forward_ghosts[possible_overlaps][
                    #         transform_passing_shift_indices])
                    #     log.debug('Passing shift surf residue pairs: %s' % forward_surface[possible_overlaps][
                    #         transform_passing_shift_indices])
                    # else:
                    #     # print('Passing shift indices:', transform_passing_shift_indices.tolist())
                    #     # print('Passing shift ghost indices:', overlapping_ghost_frags[transform_passing_shift_indices].tolist())
                    #     log.debug('Passing shift ghost residue pairs: %s' % forward_ghosts[
                    #         transform_passing_shift_indices].tolist())
                    #     # print('Passing shift surf indices:', overlapping_surf_frags[transform_passing_shift_indices].tolist())
                    #     log.debug('Passing shift surf residue pairs: %s' % forward_surface[
                    #         transform_passing_shift_indices].tolist())

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
                                                            np.hstack((blank_vector,) * (sym_entry.n_dof_external -3))))
                        # ^ I think for the sake of cleanliness, I need to make this matrix
                        # must find positive indices before group_external_dof1 multiplication in case negatives there
                        positive_indices = \
                            np.where(np.all(np.where(optimal_ext_dof_shifts < 0, False, True), axis=1) == True)
                        # optimal_ext_dof_shifts[:, :, None] <- None expands the axis to make multiplication accurate
                        stacked_external_tx1 = optimal_ext_dof_shifts[:, :, None] * sym_entry.group_external_dof1
                        stacked_external_tx2 = optimal_ext_dof_shifts[:, :, None] * sym_entry.group_external_dof2
                        full_ext_tx1.append(stacked_external_tx1[positive_indices])
                        full_ext_tx2.append(stacked_external_tx2[positive_indices])
                        full_optimal_ext_dof_shifts.append(optimal_ext_dof_shifts[positive_indices])
                    else:
                        # optimal_ext_dof_shifts = list(repeat(None, number_passing_shifts))
                        positive_indices = np.arange(number_passing_shifts)
                        # stacked_external_tx1, stacked_external_tx2 = None, None
                        full_ext_tx1, full_ext_tx2 = None, None
                        # stacked_external_tx1 = list(repeat(None, number_passing_shifts))
                        # stacked_external_tx2 = list(repeat(None, number_passing_shifts))

                    # Prepare the transformation parameters for storage in full transformation arrays
                    internal_tx_params1 = transform_passing_shifts[:, sym_entry.n_dof_external] \
                        if sym_entry.is_internal_tx1 else blank_vector
                    internal_tx_params2 = transform_passing_shifts[:, sym_entry.n_dof_external + 1] \
                        if sym_entry.is_internal_tx2 else blank_vector
                    stacked_internal_tx_vectors1 = np.hstack((blank_vector, blank_vector, internal_tx_params1[:, None]))
                    stacked_internal_tx_vectors2 = np.hstack((blank_vector, blank_vector, internal_tx_params2[:, None]))
                    # stacked_internal_tx_vectors2 = np.hstack((blank_vector, blank_vector, internal_tx_params2[:, None]))
                    stacked_rot_mat1 = np.tile(rot_mat1, (number_passing_shifts, 1, 1))
                    stacked_rot_mat2 = np.tile(rot_mat2, (number_passing_shifts, 1, 1))
                    # stacked_set_mat1 = np.tile(set_mat1, (number_passing_shifts, 1, 1))
                    # stacked_set_mat2 = np.tile(set_mat2, (number_passing_shifts, 1, 1))

                    # Store transformation parameters
                    # full_optimal_ext_dof_shifts.append(optimal_ext_dof_shifts[positive_indices])
                    # full_ext_tx1.append(stacked_external_tx1[positive_indices])
                    # full_ext_tx2.append(stacked_external_tx2[positive_indices])
                    full_int_tx1.append(stacked_internal_tx_vectors1[positive_indices])
                    full_int_tx2.append(stacked_internal_tx_vectors2[positive_indices])
                    full_rotation1.append(stacked_rot_mat1[positive_indices])
                    full_rotation2.append(stacked_rot_mat2[positive_indices])
                    # full_setting1.append(stacked_set_mat1[positive_indices])
                    # full_setting2.append(stacked_set_mat2[positive_indices])

                    final_passing_shifts = len(positive_indices)
                    log.info('Optimal Shift Search Took: %s s for %d guide coordinate pairs\n'
                             % (optimal_shifts_time, len(possible_ghost_frag_indices)))
                    log.info('%s Initial Interface Fragment Match%s Found\n'
                             % (final_passing_shifts if final_passing_shifts else 'No',
                                'es' if final_passing_shifts != 1 else ''))
                rot2_count = 0
            degen2_count = 0
        rot1_count = 0

    ##############
    # here represents an important break in the execution of this code. Vectorized scoring and clash testing!
    ##############
    # this returns the vectorized uc_dimensions
    if sym_entry.unit_cell:
        full_uc_dimensions = sym_entry.get_uc_dimensions(np.concatenate(full_optimal_ext_dof_shifts))
        full_ext_tx1 = np.concatenate(full_ext_tx1)
        full_ext_tx2 = np.concatenate(full_ext_tx2)
    # make full vectorized transformations overwriting individual variables
    full_rotation1 = np.concatenate(full_rotation1)
    full_rotation2 = np.concatenate(full_rotation2)
    full_int_tx1 = np.concatenate(full_int_tx1)
    full_int_tx2 = np.concatenate(full_int_tx2)
    starting_transforms = len(full_int_tx1)
    # full_setting1 = np.tile(set_mat1, (starting_transforms, 1, 1))
    # full_setting2 = np.tile(set_mat2, (starting_transforms, 1, 1))
    # full_setting2 = np.stack(full_setting2, startng_transforms)

    # must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
    transformation1 = {'rotation': full_rotation1, 'translation': full_int_tx1[:, np.newaxis, :],
                       'rotation2': set_mat1,
                       'translation2': full_ext_tx1[:, np.newaxis, :] if full_ext_tx1 else None}
    transformation2 = {'rotation': full_rotation2, 'translation': full_int_tx2[:, np.newaxis, :],
                       'rotation2': set_mat2,
                       'translation2': full_ext_tx2[:, np.newaxis, :] if full_ext_tx2 else None}

    # find the clustered transformations to expedite search of ASU clashing
    # Todo
    #  can I use the cluster_transformation_pairs distance graph to provide feedback on other aspects of the dock?
    #  seems that I could use the distances to expedite clashing checks, especially for more timeconsuming expansion
    #  checks...
    #  At some point, extracting the exact rotation degree from the rotation matrix and extracting translation params
    #  will provide the bounds around, what I believe will appear as docking "islands" where docks are possible,
    #  likely and preferred. Searching these docks is far more important than just outputting the possible docks and
    #  their scores. These docking islands can subsequently be used to define a design potential that could be explored
    #  during design and can be minimized
    #  |
    #  Look at calculate_overlap function output from T33 docks to see about timeing. Could the euler lookup be skipped
    #  if calculate overlap time is multiplied by the number of possible?
    #  ||
    #  Timings on these from my improvement protocol shows about similar times to euler lookup and calculate overlap
    #  even with vastly different scales of the arrays. This ignores the fact that the overlap time uses a number of
    #  indexing steps including making the ij_match array formation, indexing against the ghost and surface arrays, the
    #  rmsd_reference construction
    #  Given the lookups sort of irrelevance to the scoring (given very poor alignment), I could remove that step
    #  if it interferes with differentiability. Need to check the raw time to ensure this is sufficient! The Euler
    #  lookup and indexing steps could add as much overhead as the full calculate_match function itself.
    clustering_start = time.time()
    transform_neighbor_tree, cluster = \
        cluster_transformation_pairs(transformation1, transformation2, minimum_members=min_matched)
    cluster_representative_indices, cluster_labels = find_cluster_representatives(transform_neighbor_tree, cluster)
    sufficiently_dense_indices = np.where(cluster_labels != -1)
    number_of_dense_transforms = len(sufficiently_dense_indices[0])
    clustering_time = time.time() - clustering_start

    log.info('Found %d total transforms, %d of which are missing the minimum number of close transforms to be viable. '
             '%d remain (took %f s)' % (starting_transforms, starting_transforms - number_of_dense_transforms,
                                        number_of_dense_transforms, clustering_time))
    # representative_labels = cluster_labels[cluster_representative_indices]

    # Transform the oligomeric coords to query for clashes
    transfrom_clash_coords_start = time.time()
    # stack the superposition_rotation_matrix
    full_rotation1 = full_rotation1[sufficiently_dense_indices]
    full_rotation2 = full_rotation2[sufficiently_dense_indices]
    full_int_tx1 = full_int_tx1[sufficiently_dense_indices]
    full_int_tx2 = full_int_tx2[sufficiently_dense_indices]
    # full_setting1 = full_setting1[sufficiently_dense_indices]
    # full_setting2 = full_setting2[sufficiently_dense_indices]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[sufficiently_dense_indices]
        full_ext_tx1 = full_ext_tx1[sufficiently_dense_indices]
        full_ext_tx2 = full_ext_tx2[sufficiently_dense_indices]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
    else:
        full_ext_tx_sum = None
    full_inv_rotation2 = np.linalg.inv(full_rotation2)
    full_inv_rotation1 = np.linalg.inv(full_rotation1)
    # full_inv_setting1 = np.linalg.inv(full_setting1)
    inv_setting1 = np.linalg.inv(set_mat1)
    # superposition_setting1_stack = np.tile(superposition_setting_1to2, (number_of_dense_transforms, 1, 1))

    # alternative route to measure clashes of each transform. Move copies of component2 to interact with pdb1 ORIGINAL
    # multiply by -1 to invert the translation
    # tile_transform1 = {'rotation': full_rotation2,
    #                    'translation': full_int_tx2[:, np.newaxis, :],
    #                    'rotation2': superposition_setting1_stack,
    #                    'translation2': full_int_tx1[:, np.newaxis, :] * -1}  # invert translation
    # tile_transform2 = {'rotation': full_inv_rotation2,
    #                    'translation': full_ext_tx2[:, np.newaxis, :],
    #                    'rotation2': None,
    #                    'translation2': full_ext_tx1[:, np.newaxis, :] * -1}
    # og_transform1 = {'rotation': full_rotation1,
    #                  'translation': full_int_tx1[:, np.newaxis, :],
    #                  'rotation2': set_mat1,
    #                  'translation2': full_ext_tx1[:, np.newaxis, :] if full_ext_tx1 else None}  # invert translation
    # og_transform2 = {'rotation': full_rotation2,
    #                  'translation': full_int_tx2[:, np.newaxis, :],
    #                  'rotation2': set_mat2,
    #                  'translation2': full_ext_tx2[:, np.newaxis, :] if full_ext_tx2 else None}  # invert translation
    tile_transform1 = {'rotation': full_rotation2,
                       'translation': full_int_tx2[:, np.newaxis, :],
                       'rotation2': set_mat2,
                       'translation2': full_ext_tx_sum[:, np.newaxis, :] if full_ext_tx_sum else None}  # invert translation
    tile_transform2 = {'rotation': inv_setting1,
                       'translation': full_int_tx1[:, np.newaxis, :] * -1,
                       'rotation2': full_inv_rotation1,
                       'translation2': None}
    pdb2_tiled_coords = np.tile(pdb2_bb_cb_coords, (number_of_dense_transforms, 1, 1))
    transformed_pdb2_tiled_coords = transform_coordinate_sets(pdb2_tiled_coords, **tile_transform1)
    inverse_transformed_pdb2_tiled_coords = transform_coordinate_sets(transformed_pdb2_tiled_coords, **tile_transform2)

    transfrom_clash_coords_time = time.time() - transfrom_clash_coords_start
    log.info('\tCopy and Transform All Oligomer1 and Oligomer2 coords for clash testing (took %f s)'
             % transfrom_clash_coords_time)

    check_clash_coords_start = time.time()
    asu_clash_counts = \
        np.array([oligomer1_backbone_cb_tree.two_point_correlation(inverse_transformed_pdb2_tiled_coords[idx],
                                                                   [clash_dist])
                  for idx in range(inverse_transformed_pdb2_tiled_coords.shape[0])])
    check_clash_coords_time = time.time() - check_clash_coords_start

    # # check of transformation with forward of 2 and reverse of 1
    # pdb1.write(out_path=os.path.join(os.getcwd(), 'TEST_forward_reverse_pdb1.pdb'))
    # for idx in range(5):
    #     # print(full_rotation2[idx].shape)
    #     # print(full_int_tx2[idx].shape)
    #     # print(set_mat2.shape)
    #     # print(full_ext_tx_sum[idx].shape if full_ext_tx_sum else None)
    #     pdb2_copy = pdb2.return_transformed_copy(**{'rotation': full_rotation2[idx],
    #                                                 'translation': full_int_tx2[idx],
    #                                                 'rotation2': set_mat2,
    #                                                 'translation2': full_ext_tx_sum[idx]
    #                                                 if full_ext_tx_sum else None})
    #     # pdb2_copy.write(out_path=os.path.join(os.getcwd(), 'TEST_forward_reverse_transform_mid%d.pdb' % idx))
    #     pdb2_copy.transform(**{'rotation': inv_setting1,
    #                            'translation': full_int_tx1[idx] * -1,
    #                            'rotation2': full_inv_rotation1[idx]})
    #     pdb2_copy.write(out_path=os.path.join(os.getcwd(), 'TEST_forward_reverse_transform%d.pdb' % idx))
    #
    # for idx in range(5):
    #     pdb1_copye = pdb1.return_transformed_copy(**{'rotation': full_rotation1[idx],
    #                                                  'translation': full_int_tx1[idx],
    #                                                  'rotation2': set_mat1,
    #                                                  'translation2': full_ext_tx1[idx] if full_ext_tx1 else None})
    #     pdb1_copye.write(out_path=os.path.join(os.getcwd(), 'TEST_forward_transform1_%d.pdb' % idx))
    #     pdb2_copye = pdb2.return_transformed_copy(**{'rotation': full_rotation2[idx],
    #                                                  'translation': full_int_tx2[idx],
    #                                                  'rotation2': set_mat2,
    #                                                  'translation2': full_ext_tx2[idx] if full_ext_tx2 else None})
    #     pdb2_copye.write(out_path=os.path.join(os.getcwd(), 'TEST_forward_transform2_%d.pdb' % idx))

    asu_is_viable = np.where(asu_clash_counts.flatten() == 0)  # , True, False)
    number_of_non_clashing_transforms = len(asu_is_viable[0])
    log.info('\tClash testing for All Oligomer1 and Oligomer2 (took %f s) found %d viable ASU\'s'
             % (check_clash_coords_time, number_of_non_clashing_transforms))

    full_rotation1 = full_rotation1[asu_is_viable]
    full_rotation2 = full_rotation2[asu_is_viable]
    full_int_tx1 = full_int_tx1[asu_is_viable]
    full_int_tx2 = full_int_tx2[asu_is_viable]
    # superposition_setting1_stack = superposition_setting1_stack[asu_is_viable]
    # full_setting1 = full_setting1[asu_is_viable]
    # full_setting2 = full_setting2[asu_is_viable]
    if sym_entry.unit_cell:
        full_uc_dimensions = full_uc_dimensions[asu_is_viable]
        # viable_uc_dimensions = full_uc_dimensions[asu_is_viable]
        # multiply by -1 to invert the translation
        full_ext_tx1 = full_ext_tx1[asu_is_viable]
        full_ext_tx2 = full_ext_tx2[asu_is_viable]
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
    else:
        full_uc_dimensions = None
    # full_inv_rotation2 = full_inv_rotation2[asu_is_viable]
    full_inv_rotation1 = full_inv_rotation1[asu_is_viable]
    # full_inv_setting1 = full_inv_setting1[asu_is_viable]
    # viable_cluster_labels = cluster_labels[asu_is_viable[0]]

    #################
    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    # stack the superposition_rotation_matrix
    # superposition_setting1_stack = np.tile(superposition_setting_1to2, (number_of_non_clashing_transforms, 1, 1))
    # alternative route to measure clashes of each transform. Move copies of component2 to interact with pdb1 ORIGINAL
    tile_transform1 = {'rotation': full_rotation2,
                       'translation': full_int_tx2[:, np.newaxis, :],
                       'rotation2': set_mat2,
                       'translation2': full_ext_tx_sum[:, np.newaxis, :] if full_ext_tx_sum else None}  # invert translation
    tile_transform2 = {'rotation': inv_setting1,
                       'translation': full_int_tx1[:, np.newaxis, :] * -1,
                       'rotation2': full_inv_rotation1,
                       'translation2': None}
    tile_transform1_guides = {'rotation': full_rotation2[:, np.newaxis, :, :],
                              'translation': full_int_tx2[:, np.newaxis, np.newaxis, :],
                              'rotation2': set_mat2[np.newaxis, np.newaxis, :, :],
                              'translation2': full_ext_tx_sum[:, np.newaxis, np.newaxis, :] if full_ext_tx_sum else None}  # invert translation
    tile_transform2_guides = {'rotation': inv_setting1[np.newaxis, np.newaxis, :, :],
                              'translation': full_int_tx1[:, np.newaxis, np.newaxis, :] * -1,
                              'rotation2': full_inv_rotation1[:, np.newaxis, :, :],
                              'translation2': None}
    int_cb_and_frags_start = time.time()
    pdb1_cb_balltree = BallTree(pdb1.get_cb_coords())
    pdb2_tiled_cb_coords = np.tile(pdb2.get_cb_coords(), (number_of_non_clashing_transforms, 1, 1))
    transformed_pdb2_tiled_cb_coords = transform_coordinate_sets(pdb2_tiled_cb_coords, **tile_transform1)
    inverse_transformed_pdb2_tiled_cb_coords = \
        transform_coordinate_sets(transformed_pdb2_tiled_cb_coords, **tile_transform2)
    surf_frags2_tiled_guide_coords = surf_frags2_guide_coords[np.newaxis, :, :, :]
    # surf_frags2_tiled_guide_coords = np.tile(surf_frags2_guide_coords, (number_of_non_clashing_transforms, 1, 1, 1))
    transformed_surf_frags2_guide_coords = \
        transform_coordinate_sets(surf_frags2_tiled_guide_coords, **tile_transform1_guides)
    inverse_transformed_surf_frags2_guide_coords = \
        transform_coordinate_sets(transformed_surf_frags2_guide_coords, **tile_transform2_guides)

    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_indices = pdb2.cb_indices
    pdb1_residues = pdb1.residues
    pdb2_residues = pdb2.residues
    # print('number of pdb1_cb_indices %d' % len(pdb1_cb_indices))
    # print('number of pdb2_cb_indices %d' % len(pdb2_cb_indices))
    # print('number of pdb1_residues %d' % len(pdb1_residues))
    # print('number of pdb2_residues %d' % len(pdb2_residues))
    # log.debug('Transformed guide_coords')
    int_cb_and_frags_time = time.time() - int_cb_and_frags_start
    log.info('\tTransformation of all viable PDB2 CB atoms and surface fragments took %f s' % int_cb_and_frags_time)
    # asu_interface_residues = \
    #     np.array([oligomer1_backbone_cb_tree.query_radius(inverse_transformed_pdb2_tiled_cb_coords[idx], cb_distance)
    #               for idx in range(inverse_transformed_pdb2_tiled_cb_coords.shape[0])])

    # Use below instead of above until can TODO vectorize asu_interface_residue_processing

    # Full Interface Fragment Match
    # Get Residue Number for all Interacting PDB1 CB, PDB2 CB Pairs
    cb_distance = 9.  # change to 8.?
    # print('number of inverse_transformed_pdb2_tiled_cb_coords %d' % len(inverse_transformed_pdb2_tiled_cb_coords))
    for idx, trans_surf_guide_coords in enumerate(inverse_transformed_surf_frags2_guide_coords.tolist()):
        int_frags_time_start = time.time()
        # print('number of pdb2 coords %d' % len(inverse_transformed_pdb2_tiled_cb_coords[idx]))
        pdb2_query = pdb1_cb_balltree.query_radius(inverse_transformed_pdb2_tiled_cb_coords[idx], cb_distance)
        # print('number of pdb2_query %d' % len(pdb2_query))
        contacting_pairs = \
            [(pdb1.coords_indexed_residues[pdb1_cb_indices[pdb1_idx]].number,
              pdb2.coords_indexed_residues[pdb2_cb_indices[pdb2_idx]].number)
             for pdb2_idx, pdb1_contacts in enumerate(pdb2_query) for pdb1_idx in pdb1_contacts]
        interface_residues1, interface_residues2 = split_interface_numbers(contacting_pairs)
        # These were interface_surf_frags and interface_ghost_frags
        # interface_ghost1_indices = \
        #     np.concatenate([np.where(ghost_frag1_residues == residue) for residue in interface_residues1])
        # interface_surf2_indices = \
        #     np.concatenate([np.where(surf_frag2_residues == residue) for residue in interface_residues2])
        interface_ghost1_bool = np.isin(ghost_frag1_residues, interface_residues1)
        interface_surf2_bool = np.isin(surf_frag2_residues, interface_residues2)
        print('trans_surf_guide_coords %s' % trans_surf_guide_coords[:5])
        print('interface_surf2_bool %s' % interface_surf2_bool[:5])
        all_fragment_match_time_start = time.time()
        if idx % 2 == 1:
            # interface_ghost_frags = complete_ghost_frags1[interface_ghost1_indices]
            # interface_surf_frags = surf_frags2[interface_surf2_indices]
            # int_ghost_frag_guide_coords = ghost_frag1_guide_coords[interface_ghost1_indices]
            int_ghost_frag_guide_coords = ghost_frag1_guide_coords[interface_ghost1_bool]
            # int_surf_frag_guide_coords = surf_frags2_guide_coords[interface_surf2_indices]
            # int_trans_ghost_guide_coords = \
            #     transform_coordinate_sets(int_ghost_frag_guide_coords, rotation=rot_mat1, translation=internal_tx_param1,
            #                               rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            # int_trans_surf2_guide_coords = \
            #     transform_coordinate_sets(int_surf_frag_guide_coords, rotation=rot_mat2, translation=internal_tx_param2,
            #                               rotation2=sym_entry.setting_matrix2, translation2=external_tx_params2)

            # transforming only surface frags will have large speed gains from not having to transform all ghosts
            # int_trans_surf2_guide_coords = trans_surf_guide_coords[interface_surf2_indices]
            int_trans_surf2_guide_coords = trans_surf_guide_coords[interface_surf2_bool]
            # NOT crucial ###
            unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2 = \
                len(int_ghost_frag_guide_coords), len(int_trans_surf2_guide_coords)
            get_int_frags_time = time.time() - int_frags_time_start
            log.info('\tNewly Formed Interface Contains %d Unique Fragments on Oligomer 1 and %d on Oligomer 2\n\t'
                     '(took %f s to get interface surface and ghost fragments with their guide coordinates)'
                     % (unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2, get_int_frags_time))
            # NOT crucial ###

            # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Surface Fragment) guide coordinate pairs
            # in the same Euler rotational space bucket
            # DON'T think this is crucial! ###
            eul_lookup_start_time = time.time()
            # overlapping_ghost_indices, overlapping_surf_indices = \
            #     euler_lookup.check_lookup_table(int_trans_ghost_guide_coords, int_trans_surf2_guide_coords)
            overlapping_ghost_indices, overlapping_surf_indices = \
                euler_lookup.check_lookup_table(int_ghost_frag_guide_coords, int_trans_surf2_guide_coords)  # ,
            #                                   secondary_structure_match=ij_type_match)
            ij_type_match = ij_type_match_lookup_table[overlapping_ghost_indices, overlapping_surf_indices]
            # log.debug('Euler lookup')
            eul_lookup_time = time.time() - eul_lookup_start_time
            # DON'T think this is crucial! ###

            # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
            overlap_score_time_start = time.time()
            # get only fragment indices that pass ij filter and their associated coords
            passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
            # passing_ghost_coords = int_trans_ghost_guide_coords[passing_ghost_indices]
            passing_ghost_coords = int_ghost_frag_guide_coords[passing_ghost_indices]
            passing_surf_indices = overlapping_surf_indices[ij_type_match]
            passing_surf_coords = int_trans_surf2_guide_coords[passing_surf_indices]

            # reference_rmsds = ghost_frag1_rmsds[interface_ghost1_indices][overlapping_ghost_indices][ij_type_match]
            reference_rmsds = ghost_frag1_rmsds[interface_ghost1_bool][overlapping_ghost_indices][ij_type_match]
            all_fragment_match = calculate_match(passing_ghost_coords, passing_surf_coords, reference_rmsds)
            overlap_score_time = time.time() - overlap_score_time_start
            log.info('\tEuler Lookup took %f s for %d fragment pairs and Overlap Score Calculation took %f s for %d '
                     'fragment pairs' % (eul_lookup_time, unique_interface_frag_count_pdb1 * unique_interface_frag_count_pdb2,
                                         overlap_score_time, len(overlapping_ghost_indices)))
        else:
            # below bypasses euler lookup
            # 1
            # # this may be slower than just calculating all and not worrying about interface!
            # int_ij_matching_ghost1_indices = np.isin(ij_matching_ghost1_indices, interface_ghost1_indices)
            # int_ij_matching_surf2_indices = np.isin(ij_matching_surf2_indices, interface_surf2_indices)
            # typed_ghost1_coords = ghost_frag1_guide_coords[int_ij_matching_ghost1_indices]
            # typed_surf2_coords = surf_frags2_guide_coords[int_ij_matching_surf2_indices]
            # reference_rmsds = ghost_frag1_rmsds[int_typed_ghost1_indices]
            # # 2
            # typed_ghost1_coords = ghost_frag1_guide_coords[ij_matching_ghost1_indices]
            # typed_surf2_coords = surf_frags2_guide_coords[ij_matching_surf2_indices]
            # reference_rmsds = ghost_frag1_rmsds[ij_matching_ghost1_indices]
            # 3
            # first slice the table according to the interface residues
            # int_ij_lookup_table = \
            #     ij_type_match_lookup_table[interface_ghost1_indices[:, np.newaxis], interface_surf2_indices]
            int_ij_lookup_table = np.logical_and(ij_type_match_lookup_table,
                                                 (np.einsum('i, j -> ij', interface_ghost1_bool, interface_surf2_bool)))
            # axis 0 is ghost frag, 1 is surface frag
            # int_row_indices, int_column_indices = np.indices(int_ij_lookup_table.shape)  # row vary by ghost, column by surf
            # int_ij_matching_ghost1_indices = \
            #     row_indices[interface_ghost1_indices[:, np.newaxis], interface_surf2_indices][int_ij_lookup_table]
            # int_ij_matching_surf2_indices = \
            #     column_indices[interface_ghost1_indices[:, np.newaxis], interface_surf2_indices][int_ij_lookup_table]
            int_ij_matching_ghost1_indices = row_indices[int_ij_lookup_table]
            int_ij_matching_surf2_indices = column_indices[int_ij_lookup_table]
            # int_ij_matching_ghost1_indices = \
            #     (int_ij_lookup_table * np.arange(int_ij_lookup_table.shape[0]))[int_ij_lookup_table]
            # int_ij_matching_surf2_indices = \
            #     (int_ij_lookup_table * np.arange(int_ij_lookup_table.shape[1])[:, np.newaxis])[int_ij_lookup_table]
            typed_ghost1_coords = ghost_frag1_guide_coords[int_ij_matching_ghost1_indices]
            typed_surf2_coords = surf_frags2_guide_coords[int_ij_matching_surf2_indices]
            reference_rmsds = ghost_frag1_rmsds[int_ij_matching_ghost1_indices]

            all_fragment_match = calculate_match(typed_ghost1_coords, typed_surf2_coords, reference_rmsds)
        all_fragment_match_time = time.time() - all_fragment_match_time_start

        passing_overlaps_indices = np.where(all_fragment_match > 0.2)

        log.info('\t%d Fragment Match(es) Found in Complete Cluster Representative Fragment Library (took %f s)' %
                 (len(passing_overlaps_indices), all_fragment_match_time))

        # check if the pose has enough high quality fragment matches
        high_qual_match_count = np.where(all_fragment_match > high_quality_match_value)[0].size
        if high_qual_match_count < min_matched:
            log.info('\t%d < %d Which is Set as the Minimal Required Amount of High Quality Fragment Matches'
                     % (high_qual_match_count, min_matched))
            continue
        else:
            passing_overlaps_indices = np.where(all_fragment_match > 0.2)

        # # Full Interface Fragment Match
        #     get_int_ghost_surf_frags_time_start = time.time()
        #     interface_chain_residues_pdb1, interface_chain_residues_pdb2 = get_interface_residues(pdb1_copy, pdb2_copy)
        #     unique_interface_frag_count_pdb1 = len(interface_chain_residues_pdb1)
        #     unique_interface_frag_count_pdb2 = len(interface_chain_residues_pdb2)
        #     unique_total_monofrags_count = unique_interface_frag_count_pdb1 + unique_interface_frag_count_pdb2
        #
        #     # Todo
        #     #  The use of hashing on the surface and ghost fragments could increase program runtime, over tuple call
        #     #   to the ghost_fragment objects to return the aligned chain and residue then test for membership...
        #     #  Is the chain necessary? Two chains can occupy interface, even the same residue could be used
        #     #   Think D2 symmetry...
        #     #  Store all the ghost/surface frags in a chain/residue dictionary?
        #     interface_ghost_frags = [ghost_frag for ghost_frag in complete_ghost_frags1
        #                              if ghost_frag.get_aligned_chain_and_residue() in interface_chain_residues_pdb1]
        #     interface_surf_frags = [surf_frag for surf_frag in surf_frags2
        #                             if surf_frag.get_central_res_tup() in interface_chain_residues_pdb2]
        #     # if unique_total_monofrags_count == 0:
        #     if not interface_ghost_frags or not interface_surf_frags:
        #         log.info('\tNO Interface Mono Fragments Found')
        #         continue
        #
        #     int_ghost_frag_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in interface_ghost_frags])
        #     int_surf_frag_guide_coords = np.array([surf_frag.guide_coords for surf_frag in interface_surf_frags])
        #
        #     int_trans_ghost_guide_coords = \
        #         transform_coordinate_sets(int_ghost_frag_guide_coords, rotation=rot_mat1, translation=internal_tx_param1,
        #                                   rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
        #
        #     int_trans_surf2_guide_coords = \
        #         transform_coordinate_sets(int_surf_frag_guide_coords, rotation=rot_mat2, translation=internal_tx_param2,
        #                                   rotation2=sym_entry.setting_matrix2, translation2=external_tx_params2)
        #
        #     # log.debug('Transformed guide_coords')
        #     get_int_ghost_surf_frags_time_end = time.time()
        #     get_int_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start
        #
        #     log.info('\tNewly Formed Interface Contains %d Unique Fragments on Oligomer 1 and %d on Oligomer 2\n\t'
        #              '(took: %s s to get interface surface and ghost fragments with their guide coordinates)'
        #              % (unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2, str(get_int_frags_time)))
        #
        #     # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Surface Fragment) guide coordinate pairs
        #     # in the same Euler rotational space bucket
        #     eul_lookup_start_time = time.time()
        #     overlapping_ghost_indices, overlapping_surf_indices = \
        #         euler_lookup.check_lookup_table(int_trans_ghost_guide_coords, int_trans_surf2_guide_coords)
        #     # log.debug('Euler lookup')
        #     eul_lookup_end_time = time.time()
        #     eul_lookup_time = eul_lookup_end_time - eul_lookup_start_time
        #
        #     # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        #     overlap_score_time_start = time.time()
        #
        #     # filter array by matching type for surface (i) and ghost (j) frags
        #     surface_type_i_array = np.array([interface_surf_frags[idx].i_type for idx in overlapping_surf_indices.tolist()])
        #     ghost_type_j_array = np.array([interface_ghost_frags[idx].j_type for idx in overlapping_ghost_indices.tolist()])
        #     ij_type_match = np.where(surface_type_i_array == ghost_type_j_array, True, False)
        #
        #     # get only fragment indices that pass ij filter and their associated coords
        #     passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
        #     passing_ghost_coords = int_trans_ghost_guide_coords[passing_ghost_indices]
        #     passing_surf_indices = overlapping_surf_indices[ij_type_match]
        #     passing_surf_coords = int_trans_surf2_guide_coords[passing_surf_indices]
        #
        #     # precalculate the reference_rmsds for each ghost fragment
        #     reference_rmsds = np.array([interface_ghost_frags[idx].rmsd for idx in passing_ghost_indices.tolist()])
        #     reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)  # ensure no division by 0
        #     all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
        #                                              max_z_value=subseq_max_z_val)
        #     # log.debug('Checking all fragment overlap at interface')
        #     # get the passing_overlap indices and associated z-values by finding all indices where the value is not false
        #     passing_overlaps_indices = np.flatnonzero(all_fragment_overlap)  # .nonzero()[0]
        #     passing_z_values = all_fragment_overlap[passing_overlaps_indices]
        #     # log.debug('Overlapping z-values: %s' % passing_z_values)
        #
        #     overlap_score_time_stop = time.time()
        #     overlap_score_time = overlap_score_time_stop - overlap_score_time_start
        #
        #     log.info('\t%d Fragment Match(es) Found in Complete Cluster Representative Fragment Library\n\t(Euler Lookup '
        #              'took %f s for %d fragment pairs and Overlap Score Calculation took %s for %d fragment pairs)'
        #              % (len(passing_overlaps_indices), str(eul_lookup_time),
        #                 len(int_trans_ghost_guide_coords) * unique_interface_frag_count_pdb2,
        #                 str(overlap_score_time), len(overlapping_ghost_indices)))
        #
        #     # check if the pose has enough high quality fragment matches
        #     high_qual_match_count = np.where(passing_z_values < high_quality_match_value)[0].size
        #     if high_qual_match_count < min_matched:
        #         log.info('\t%d < %d Which is Set as the Minimal Required Amount of High Quality Fragment Matches'
        #                  % (high_qual_match_count, min_matched))
        #         continue

        # Get contacting PDB 1 ASU and PDB 2 ASU
        copy_pdb_start = time.time()
        specific_transformation1 = {'rotation': full_rotation1[idx], 'translation': full_int_tx1[idx],
                                    'rotation2': set_mat1, 'translation2': full_ext_tx1[idx]}
        specific_transformation2 = {'rotation': full_rotation2[idx], 'translation': full_int_tx2[idx],
                                    'rotation2': set_mat2, 'translation2': full_ext_tx2[idx]}

        pdb1_copy = pdb1.return_transformed_copy(**specific_transformation1)
        pdb2_copy = pdb2.return_transformed_copy(**specific_transformation2)
        copy_pdb_time = time.time() - copy_pdb_start
        log.info('\tCopy and Transform Oligomer1 and Oligomer2 (took %f s)' % copy_pdb_time)
        # Todo ensure asu chain names are different
        asu = get_contacting_asu(pdb1_copy, pdb2_copy, entity_names=[pdb1_copy.name, pdb2_copy.name])
        # log.debug('Grabbing asu')
        if not asu:  # _pdb_1 and not asu_pdb_2:
            log.info('\tNO Design ASU Found')
            continue

        # Check if design has any clashes when expanded
        exp_des_clash_time_start = time.time()
        if sym_entry.unit_cell:
            asu.uc_dimensions = full_uc_dimensions[idx]
        asu.expand_matrices = sym_entry.expand_matrices
        symmetric_material = Pose.from_asu(asu, sym_entry=sym_entry, ignore_clashes=True, log=log)
        #                      surrounding_uc=output_surrounding_uc, ^ ignores ASU clashes
        exp_des_clash_time = time.time() - exp_des_clash_time_start

        # log.debug('Checked expand clash')
        if symmetric_material.symmetric_assembly_is_clash():
            log.info('\tBackbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
            continue

        log.info('\tNO Backbone Clash when Designed Assembly is Expanded (took %f s)' % exp_des_clash_time)
        # Todo replace with DesignDirectory? Path object?
        degen_subdir_out_path = os.path.join(outdir, 'DEGEN_%d_%d' % (degen1_count, degen2_count))
        rot_subdir_out_path = os.path.join(degen_subdir_out_path, 'ROT_%d_%d' % (rot1_count, rot2_count))
        tx_dir = os.path.join(rot_subdir_out_path, 'tx_%d' % idx)
        oligomers_dir = rot_subdir_out_path.split(os.sep)[-3]
        degen_dir = rot_subdir_out_path.split(os.sep)[-2]
        rot_dir = rot_subdir_out_path.split(os.sep)[-1]
        pose_id = '%s_%s_%s_TX_%d' % (oligomers_dir, degen_dir, rot_dir, idx)
        sampling_id = '%s_%s_TX_%d' % (degen_dir, rot_dir, idx)
        # if not os.path.exists(degen_subdir_out_path):
        #     os.makedirs(degen_subdir_out_path)
        # if not os.path.exists(rot_subdir_out_path):
        #     os.makedirs(rot_subdir_out_path)
        if not os.path.exists(tx_dir):
            os.makedirs(tx_dir)

        # Make directories to output matched fragment PDB files
        # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
        matching_fragments_dir = os.path.join(tx_dir, 'matching_fragments')
        if not os.path.exists(matching_fragments_dir):
            os.makedirs(matching_fragments_dir)
        high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
        low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')

        # Write ASU, PDB1, PDB2, and expanded assembly files
        cryst1_record = None
        # if optimal_ext_dof_shifts:  # 2, 3 dimensions
        if sym_entry.unit_cell:
            asu = get_central_asu(asu, asu.uc_dimensions, sym_entry.dimension)
            cryst1_record = generate_cryst1_record(asu.uc_dimensions, sym_entry.resulting_symmetry)
        asu.write(out_path=os.path.join(tx_dir, 'asu.pdb'), header=cryst1_record)
        pdb1_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb1_copy.name, sampling_id)))
        pdb2_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb2_copy.name, sampling_id)))

        if output_assembly:
            symmetric_material.get_assembly_symmetry_mates(surrounding_uc=output_surrounding_uc)
            # if optimal_ext_dof_shifts:  # 2, 3 dimensions
            if sym_entry.unit_cell:
                if output_surrounding_uc:
                    symmetric_material.write(out_path=os.path.join(tx_dir, 'surrounding_unit_cells.pdb'),
                                             header=cryst1_record)
                else:
                    symmetric_material.write(out_path=os.path.join(tx_dir, 'central_uc.pdb'), header=cryst1_record)
            else:  # 0 dimension
                symmetric_material.write(out_path=os.path.join(tx_dir, 'expanded_assembly.pdb'))
        log.info('\tSUCCESSFUL DOCKED POSE: %s' % tx_dir)

        # return the indices sorted by z_value then pull information accordingly
        sorted_fragment_indices = np.argsort(all_fragment_match)
        match_scores = all_fragment_match[sorted_fragment_indices]
        # match_scores = match_score_from_z_value(sorted_z_values)
        # log.debug('Overlapping Match Scores: %s' % match_scores)
        sorted_overlap_indices = passing_overlaps_indices[sorted_fragment_indices]
        int_ghostfrags = complete_ghost_frags1[interface_ghost1_indices][passing_ghost_indices[sorted_overlap_indices]]
        int_monofrags2 = surf_frags2[interface_surf2_indices][passing_surf_indices[sorted_overlap_indices]]
        # int_ghostfrags = [interface_ghost_frags[idx] for idx in passing_ghost_indices[sorted_overlap_indices].tolist()]
        # int_monofrags2 = [interface_surf_frags[idx] for idx in passing_surf_indices[sorted_overlap_indices].tolist()]

        # For all matched interface fragments
        # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
        # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
        chid_resnum_scores_dict_pdb1, chid_resnum_scores_dict_pdb2 = {}, {}

        # Number of unique interface mono fragments matched
        unique_frags_info1, unique_frags_info2 = set(), set()

        res_pair_freq_info_list = []
        for frag_idx, (int_ghost_frag, int_surf_frag) in enumerate(zip(int_ghostfrags, int_monofrags2)):
            surf_frag_chain1, surf_frag_central_res_num1 = int_ghost_frag.get_aligned_chain_and_residue()
            surf_frag_chain2, surf_frag_central_res_num2 = int_surf_frag.get_central_res_tup()

            covered_residues_pdb1 = [(surf_frag_chain1, surf_frag_central_res_num1 + j) for j in range(-2, 3)]
            covered_residues_pdb2 = [(surf_frag_chain2, surf_frag_central_res_num2 + j) for j in range(-2, 3)]
            match = match_scores[frag_idx]
            for k in range(ijk_frag_db.fragment_length):
                chain_resnum1 = covered_residues_pdb1[k]
                chain_resnum2 = covered_residues_pdb2[k]
                if chain_resnum1 not in chid_resnum_scores_dict_pdb1:
                    chid_resnum_scores_dict_pdb1[chain_resnum1] = [match]
                else:
                    chid_resnum_scores_dict_pdb1[chain_resnum1].append(match)

                if chain_resnum2 not in chid_resnum_scores_dict_pdb2:
                    chid_resnum_scores_dict_pdb2[chain_resnum2] = [match]
                else:
                    chid_resnum_scores_dict_pdb2[chain_resnum2].append(match)

            # if (surf_frag_chain1, surf_frag_central_res_num1) not in unique_frags_info1:
            unique_frags_info1.add((surf_frag_chain1, surf_frag_central_res_num1))
            # if (surf_frag_chain2, surf_frag_central_res_num2) not in unique_frags_info2:
            unique_frags_info2.add((surf_frag_chain2, surf_frag_central_res_num2))

            # match = sorted_z_values[frag_idx]
            if match >= 0.5:  # the overlap z-value has is greater than 1 std deviation, therefore match >= 0.5
                matched_fragment_dir = high_quality_matches_dir
            else:
                matched_fragment_dir = low_quality_matches_dir
            if not os.path.exists(matched_fragment_dir):
                os.makedirs(matched_fragment_dir)

            # if write_frags:  # write out aligned cluster representative fragment
            fragment, _ = dictionary_lookup(ijk_frag_db.paired_frags, int_ghost_frag.get_ijk())
            trnsfmd_ghost_fragment = fragment.return_transformed_copy(**int_ghost_frag.aligned_fragment.transformation)
            trnsfmd_ghost_fragment.transform(**specific_transformation1)
            trnsfmd_ghost_fragment.write(
                out_path=os.path.join(matched_fragment_dir,
                                      'int_frag_%s_%d.pdb' % ('i%d_j%d_k%d' % int_ghost_frag.get_ijk(), frag_idx + 1)))
            # transformed_ghost_fragment = int_ghost_frag.structure.return_transformed_copy(
            #     rotation=rot_mat1, translation=internal_tx_param1,
            #     rotation2=sym_entry.setting_matrix1, translation2=external_tx_params1)
            # transformed_ghost_fragment.write(os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
            #                                               % ('i%d_j%d_k%d' % int_ghost_frag.get_ijk(), frag_idx + 1)))
            z_value = z_value_from_match_score(match)
            ghost_frag_central_freqs = \
                dictionary_lookup(ijk_frag_db.info, int_ghost_frag.get_ijk()).central_residue_pair_freqs
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
        # specific_transformation2 = {'rotation': full_rotation2[idx], 'translation': full_int_tx2[idx],
        #                             'rotation2': full_setting2[idx], 'translation2': full_ext_tx2[idx]}
        write_docked_pose_info(tx_dir, res_lev_sum_score, high_qual_match_count, unique_matched_monofrag_count,
                               unique_total_monofrags_count, percent_of_interface_covered, rot_mat1, internal_tx_param1,
                               sym_entry.setting_matrix1, external_tx_params1, rot_mat2, internal_tx_param2,
                               sym_entry.setting_matrix2, external_tx_params2, cryst1_record, pdb1.filepath,
                               pdb2.filepath, pose_id)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Parsing Command Line Input
        sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir, output_assembly, \
            output_surrounding_uc, min_matched, timer, initial, debug = get_docking_parameters(sys.argv)

        # Master Log File
        master_log_filepath = os.path.join(master_outdir, master_log)
        if debug:
            # Root logs to stream with level debug
            logger = start_log(level=1, set_logger_level=True)
            master_logger, bb_logger = logger, logger
            logger.debug('Debug mode. Verbose output')
        else:
            master_logger = start_log(name=__name__, handler=2, location=master_log_filepath)
        # SymEntry Parameters
        sym_entry = SymEntry(sym_entry_number)  # sym_map inclusion?

        if initial:
            # make master output directory
            os.makedirs(master_outdir, exist_ok=True)
            # with open(master_log_filepath, 'w') as master_logfile:
            #     master_logfile.write('Nanohedra\nMODE: DOCK\n\n')
            master_logger.info('Nanohedra\nMODE: DOCK\n\n')
            write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir,
                                     log=master_logger)
        else:
            # for parallel runs, ensure that the first file was able to write before adding below log
            time.sleep(1)
            rot_step_deg1, rot_step_deg2 = get_rotation_step(sym_entry, rot_step_deg1, rot_step_deg2)

        pdb1_name = os.path.basename(os.path.splitext(pdb1_path)[0])
        pdb2_name = os.path.basename(os.path.splitext(pdb2_path)[0])
        # with open(master_log_filepath, 'a+') as master_log_file:
        #     master_log_file.write('Docking %s / %s \n' % (pdb1_name, pdb2_name))
        master_logger.info('Docking %s / %s \n' % (pdb1_name, pdb2_name))

        # Create fragment database for all ijk cluster representatives
        ijk_frag_db = unpickle(biological_fragment_db_pickle)
        # Load Euler Lookup table for each instance
        euler_lookup = EulerLookup()
        # ijk_frag_db = FragmentDB()
        #
        # # Get complete IJK fragment representatives database dictionaries
        # ijk_frag_db.get_monofrag_cluster_rep_dict()
        # ijk_frag_db.get_intfrag_cluster_rep_dict()
        # ijk_frag_db.get_intfrag_cluster_info_dict()

        try:
            # Output Directory  # Todo DesignDirectory
            building_blocks = '%s_%s' % (pdb1_name, pdb2_name)
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
            #     bb_logger.info('DOCKING %s TO %s' % (pdb1_name, pdb2_name))
            #     bb_logger.info('Oligomer 1 Path: %s\nOligomer 2 Path: %s\n' % (pdb1_path, pdb2_path))

            nanohedra_dock(sym_entry, ijk_frag_db, euler_lookup, master_outdir, pdb1_path, pdb2_path,
                           rot_step_deg1=rot_step_deg1, rot_step_deg2=rot_step_deg2, output_assembly=output_assembly,
                           output_surrounding_uc=output_surrounding_uc, min_matched=min_matched,
                           keep_time=timer)  # log=bb_logger,

            # with open(master_log_filepath, 'a+') as master_log_file:
            #     master_log_file.write('COMPLETE ==> %s\n\n'
            #                           % os.path.join(master_outdir, '%s_%s' % (pdb1_name, pdb2_name)))
            master_logger.info('COMPLETE ==> %s\n\n' % os.path.join(master_outdir, building_blocks))

        except KeyboardInterrupt:
            # with open(master_log_filepath, 'a+') as master_log:
            #     master_log.write('\nRun Ended By KeyboardInterrupt\n')
            master_logger.info('\nRun Ended By KeyboardInterrupt\n')
            exit(2)
