import os
import sys
import time

import numpy as np
from sklearn.neighbors import BallTree

from PathUtils import frag_text_file, master_log
from SymDesignUtils import calculate_overlap, match_score_from_z_value, start_log, null_log, dictionary_lookup
from utils.CmdLineArgParseUtils import get_docking_parameters
from utils.GeneralUtils import get_last_sampling_state, write_frag_match_info_file, write_docked_pose_info, \
    transform_coordinate_sets, get_rotation_step, write_docking_parameters
from utils.PDBUtils import get_contacting_asu, get_interface_residues
from utils.SymmetryUtils import get_uc_dimensions, generate_cryst1_record, get_central_asu
from classes.EulerLookup import EulerLookup
from classes.OptimalTx import OptimalTx
from classes.SymEntry import SymEntry, get_optimal_external_tx_vector, get_rot_matrices, get_degen_rotmatrices
from classes.WeightedSeqFreq import FragMatchInfo, SeqFreqInfo
from PDB import PDB
from Pose import Pose
from interface_analysis.Database import FragmentDB

# Globals
logger = start_log(name=__name__)
fragment_length = 5


def find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, optimal_tx_params, complete_ghost_frags, complete_surf_frags,
                      degen_subdir_out_path, rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup,
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
        n_dof_external = len(sym_entry.get_ext_dof())  # returns 0 - 3
        # Get Oligomer1, Oligomer2 Optimal Internal Translation vector
        internal_tx_param1, internal_tx_param2 = None, None
        if sym_entry.is_internal_tx1():
            internal_tx_param1 = [0, 0, tx_parameters[n_dof_external]]
        if sym_entry.is_internal_tx2():
            internal_tx_param2 = [0, 0, tx_parameters[n_dof_external + 1]]

        # Get Optimal External DOF shifts
        # if n_dof_external > 0:
        optimal_ext_dof_shifts = tx_parameters[:n_dof_external]
        # else:
        #     optimal_ext_dof_shifts = None

        external_tx_params1, external_tx_params2 = None, None
        ref_frame_var_is_pos, uc_dimensions = False, None
        if optimal_ext_dof_shifts:  # Todo TEST
            # ref_frame_tx_dof_e, ref_frame_tx_dof_f, ref_frame_tx_dof_g = 0, 0, 0
            # if len(optimal_ext_dof_shifts) == 1:
            #     ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
            #     if ref_frame_tx_dof_e > 0:
            #         ref_frame_var_is_pos = True
            # if len(optimal_ext_dof_shifts) == 2:
            #     ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
            #     ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
            #     if ref_frame_tx_dof_e > 0 and ref_frame_tx_dof_f > 0:
            #         ref_frame_var_is_pos = True
            # if len(optimal_ext_dof_shifts) == 3:
            #     ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
            #     ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
            #     ref_frame_tx_dof_g = optimal_ext_dof_shifts[2]
            #     if ref_frame_tx_dof_e > 0 and ref_frame_tx_dof_f > 0 and ref_frame_tx_dof_g > 0:
            #         ref_frame_var_is_pos = True

            # Restrict all reference frame translation parameters to > 0 for SCMs with reference frame translational dof
            ref_frame_var_is_neg = False
            for ref_frame_tx_dof in optimal_ext_dof_shifts:
                if ref_frame_tx_dof < 0:
                    ref_frame_var_is_neg = True
                    break

            # if (optimal_ext_dof_shifts is not None and ref_frame_var_is_pos) or (optimal_ext_dof_shifts is None):  # Old
            # if (optimal_ext_dof_shifts and ref_frame_var_is_pos) or not optimal_ext_dof_shifts:  # clean
            #     # true and true or not true
            #     dummy = True # enter docking
            # if optimal_ext_dof_shifts and not ref_frame_var_is_pos:  # negated
            if ref_frame_var_is_neg:
                # true and not true
                # don't enter docking
                # efg_tx_params_str = [str(None), str(None), str(None)]
                # for param_index in range(len(optimal_ext_dof_shifts)):
                #     efg_tx_params_str[param_index] = optimal_ext_dof_shifts[param_index]
                log.info('\tReference Frame Shift Parameter is Negative: %s\n' % optimal_ext_dof_shifts)
                continue
            else:  # not optimal_ext_dof_shifts or (optimal_ext_dof_shifts and ref_frame_var_is_pos)
                # write uc_dimensions and dock
                # Get Oligomer1 & Oligomer2 Optimal External Translation vector
                external_tx_params1 = get_optimal_external_tx_vector(
                    sym_entry.get_ref_frame_tx_dof_group1(),
                    optimal_ext_dof_shifts)
                external_tx_params2 = get_optimal_external_tx_vector(
                    sym_entry.get_ref_frame_tx_dof_group2(),
                    optimal_ext_dof_shifts)

                # Get Unit Cell Dimensions for 2D and 3D SCMs
                uc_dimensions = get_uc_dimensions(sym_entry.get_uc_spec_string(), *optimal_ext_dof_shifts)
                # uc_dimensions = get_uc_dimensions(sym_entry.get_uc_spec_string(), ref_frame_tx_dof_e,
                #                                   ref_frame_tx_dof_f,
                #                                   ref_frame_tx_dof_g)

        # Rotate, Translate and Set PDB1, PDB2
        copy_rot_tr_set_time_start = time.time()
        # Todo
        #  In theory, the rotation and setting matrix are the same for all tx_parameters, can we accelerate even though
        #  order of operations matters? by applying the setting matrix to the translation, in theory the translation
        #  will be along the same axis. This removes repeated multiplications and instead has array addition
        pdb1_copy = pdb1.return_transformed_copy(rotation=rot_mat1, translation=internal_tx_param1,
                                                 rotation2=sym_entry.get_rot_set_mat_group1(),
                                                 translation2=external_tx_params1)
        pdb2_copy = pdb2.return_transformed_copy(rotation=rot_mat2, translation=internal_tx_param2,
                                                 rotation2=sym_entry.get_rot_set_mat_group2(),
                                                 translation2=external_tx_params2)

        copy_rot_tr_set_time_stop = time.time()
        copy_rot_tr_set_time = copy_rot_tr_set_time_stop - copy_rot_tr_set_time_start
        log.info('\tCopy and Transform Oligomer1 and Oligomer2 (took: %s s)' % str(copy_rot_tr_set_time))

        # Check if PDB1 and PDB2 backbones clash
        oligomer1_oligomer2_clash_time_start = time.time()
        # Todo @profile for KDTree or Neighbors 'brute'
        kdtree_oligomer1_backbone = BallTree(pdb1_copy.get_backbone_and_cb_coords())
        asu_cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.get_backbone_and_cb_coords(),
                                                                             [clash_dist])
        oligomer1_oligomer2_clash_time_end = time.time()
        oligomer1_oligomer2_clash_time = oligomer1_oligomer2_clash_time_end - oligomer1_oligomer2_clash_time_start

        if asu_cb_clash_count[0] > 0:
            log.info('\tBackbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)'
                     % str(oligomer1_oligomer2_clash_time))
            continue
        log.info('\tNO Backbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)'
                 % str(oligomer1_oligomer2_clash_time))

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
                                      rotation2=sym_entry.get_rot_set_mat_group1(), translation2=external_tx_params1)

        transformed_monofrag2_guide_coords_np = \
            transform_coordinate_sets(surf_frag_guide_coords, rotation=rot_mat2, translation=internal_tx_param2,
                                      rotation2=sym_entry.get_rot_set_mat_group2(), translation2=external_tx_params2)

        # log.debug('Transformed guide_coords')
        get_int_ghost_surf_frags_time_end = time.time()
        get_int_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start

        log.info('\tNewly Formed Interface Contains %d Unique Fragments on Oligomer 1 and %d on Oligomer 2\n\t'
                 '(took: %s s to get interface surface and ghost fragments with their guide coordinates)'
                 % (unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2, str(get_int_frags_time)))

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
                 'took %s s for %d fragment pairs and Overlap Score Calculation took %s for %d fragment pairs)'
                 % (len(passing_overlaps_indices), str(eul_lookup_time),
                    len(transformed_ghostfrag_guide_coords_np) * unique_interface_frag_count_pdb2,
                    str(overlap_score_time), len(overlapping_ghost_indices)))
        
        # check if the pose has enough high quality fragment matches
        high_qual_match_count = np.where(passing_z_values < high_quality_match_value)[0].size
        if high_qual_match_count < min_matched:
            log.info('\t%d < %d Which is Set as the Minimal Required Amount of High Quality Fragment Matches'
                     % (high_qual_match_count, min_matched))
            continue

        # Get contacting PDB 1 ASU and PDB 2 ASU
        # Todo ensure asu chain names are different
        asu = get_contacting_asu(pdb1_copy, pdb2_copy)  # _pdb_1, asu_pdb_2
        # log.debug('Grabbing asu')
        if not asu:  # _pdb_1 and not asu_pdb_2:
            log.info('\tNO Design ASU Found')
            continue

        # Check if design has any clashes when expanded
        exp_des_clash_time_start = time.time()
        asu.uc_dimensions = uc_dimensions
        asu.expand_matrices = sym_entry.expand_matrices
        symmetric_material = Pose.from_asu(asu, symmetry=sym_entry.get_result_design_sym(), ignore_clashes=True,
                                           surrounding_uc=output_surrounding_uc, log=log)  # ^ ignores ASU clashes
        exp_des_clash_time_stop = time.time()
        exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

        # log.debug('Checked expand clash')
        if symmetric_material.symmetric_assembly_is_clash():
            log.info('\tBackbone Clash when Designed Assembly is Expanded (took: %s s)' % str(exp_des_clash_time))
            continue

        log.info('\tNO Backbone Clash when Designed Assembly is Expanded (took: %s s)' % str(exp_des_clash_time))
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
            asu = get_central_asu(asu, uc_dimensions, sym_entry.get_design_dim())
            cryst1_record = generate_cryst1_record(uc_dimensions, sym_entry.get_result_design_sym())
        asu.write(out_path=os.path.join(tx_dir, 'asu.pdb'), header=cryst1_record)
        pdb1_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb1_copy.name, sampling_id)))
        pdb2_copy.write(os.path.join(tx_dir, '%s_%s.pdb' % (pdb2_copy.name, sampling_id)))

        if output_assembly:
            symmetric_material.get_assembly_symmetry_mates(surrounding_uc=output_surrounding_uc)
            if optimal_ext_dof_shifts:  # 2, 3 dimensions
                symmetric_material.write(out_path=os.path.join(tx_dir, 'central_uc.pdb'), header=cryst1_record)
                if output_surrounding_uc:
                    symmetric_material.write(out_path=os.path.join(tx_dir, 'surrounding_unit_cells.pdb'),
                                             header=cryst1_record)
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
            for k in range(fragment_length):
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
            transformed_ghost_fragment = int_ghost_frag.structure.return_transformed_copy(
                rotation=rot_mat1, translation=internal_tx_param1,
                rotation2=sym_entry.get_rot_set_mat_group1(), translation2=external_tx_params1)
            transformed_ghost_fragment.write(os.path.join(matched_fragment_dir, 'int_frag_%s_%d.pdb'
                                                          % ('i%d_j%d_k%d' % int_ghost_frag.get_ijk(), frag_idx + 1)))

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
                               sym_entry.get_rot_set_mat_group1(), external_tx_params1, rot_mat2, internal_tx_param2,
                               sym_entry.get_rot_set_mat_group2(), external_tx_params2, cryst1_record, pdb1_path,
                               pdb2_path, pose_id)


def nanohedra_dock(sym_entry, ijk_frag_db, outdir, pdb1_path, pdb2_path, init_max_z_val=1.0,
                   subseq_max_z_val=2.0, rot_step_deg_pdb1=1, rot_step_deg_pdb2=1, output_assembly=False,
                   output_surrounding_uc=False, min_matched=3, log=null_log, keep_time=True, resume=False):

    #################################
    # # surface ghost frag overlap from the same oligomer scratch code
    # surffrags1 = pdb1.get_fragments(residue_numbers=pdb1.get_surface_residues())
    # ghost_frags_by_residue1 = [frag.get_ghost_fragments(ijk_frag_db.paired_frags, bb_cb_balltree, ijk_frag_db.info)
    #                            for frag in surffrags1]
    # surface_frag_residue_numbers = [frag.central_residue.number for frag in surffrags1]  # also the residue indices
    # surface_frag_cb_coords = [residue.cb_coords for residue in pdb1.get_residues(numbers=surface_frag_residue_numbers)]
    # pdb1_surface_cb_ball_tree = BallTree(surface_frag_cb_coords)
    # residue_contact_query = pdb1_surface_cb_ball_tree.query(surface_frag_cb_coords)
    # contacting_pairs = [([surface_frag_residue_numbers[idx1]], [surface_frag_residue_numbers[idx2]])
    #                     for idx2 in range(residue_contact_query.size) for idx1 in residue_contact_query[idx2]]
    # asymmetric_contacting_pairs, found_pairs = [], []
    # for pair1, pair2 in contacting_pairs:
    #     # add both pair orientations (1, 2) or (2, 1) regardless
    #     found_pairs.extend([(pair1, pair2), (pair2, pair1)])
    #     # only add to contacting pair if we have never observed either
    #     if (pair1, pair2) not in found_pairs or (pair2, pair1) not in found_pairs and pair1 != pair2:
    #         asymmetric_contacting_pairs.append((pair1, pair2))
    # # Now we use the asymmetric_contacting_pairs to find the ghost_fragments for each residue, subtracting pair number
    # # by one to account for the zero indexed position
    # for idx1, idx2 in asymmetric_contacting_pairs:
    #     type_bool_matrix = is_frag_type_same(ghost_frags_by_residue1[idx1 - 1], ghost_frags_by_residue1[idx1 - 1],
    #                                        type='jj')
    #     #   Fragment1
    #     # F T  F  F
    #     # R F  F  T
    #     # A F  F  F
    #     # G F  F  F
    #     # 2 T  T  F
    #     # use type_bool_matrix to guide RMSD calculation by pulling out the right ghost_corods
    #     ghost_coords_residue1 = [ghost_frags_by_residue1[idx].guide_coords
    #                              for idx, bool in enumerate(type_bool_matrix.flatten()) if bool]
    #     # have to find a way to iterate over each matrix rox/column with .flatten or other matrix iterator to pull out
    #     # necessary guide coordinate pairs
    #     calculate_overlap(ghost_coords_residue1, ghost_coords_residue2, reference_rmsds,
    #                       max_z_value=max_z_val)
    #################################

    # Get PDB2 Symmetric Building Block
    pdb2 = PDB.from_file(pdb2_path, log=log)

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume and keep_time:
        get_complete_surf_frags_time_start = time.time()

    complete_surf_frags = pdb2.get_fragments(residue_numbers=pdb2.get_surface_residues(),
                                             representatives=ijk_frag_db.reps)

    # calculate the initial match type by finding the predominant surface type
    frag_types2 = [monofrag2.i_type for monofrag2 in complete_surf_frags]
    fragment_content2 = [frag_types2.count(frag_type) for frag_type in range(1, fragment_length + 1)]
    # log.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    initial_type2 = np.argmax(fragment_content2) + 1
    # log.debug('Found initial fragment type: %d' % initial_type2)
    initial_surf_frags = [monofrag2 for monofrag2 in complete_surf_frags if monofrag2.i_type == initial_type2]
    initial_surf_frags2_guide_coords = [surf_frag.guide_coords for surf_frag in initial_surf_frags]

    if not resume and keep_time:
        get_complete_surf_frags_time_stop = time.time()
        get_complete_surf_frags_time = get_complete_surf_frags_time_stop - get_complete_surf_frags_time_start
        log.info('Getting Oligomer 2 Surface Fragments Using COMPLETE Fragment Database (took: %s s)\n'
                 % str(get_complete_surf_frags_time))

    # Get PDB1 Symmetric Building Block
    pdb1 = PDB.from_file(pdb1_path, log=log)
    oligomer1_backbone_cb_tree = BallTree(pdb1.get_backbone_and_cb_coords())

    # Get Oligomer1 Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume and keep_time:
        get_complete_ghost_frags_time_start = time.time()

    # additional gains in fragment reduction could be realized with modifying SASA threshold by ghost fragment access
    surf_frags_1 = pdb1.get_fragments(residue_numbers=pdb1.get_surface_residues(), representatives=ijk_frag_db.reps)

    complete_ghost_frags = []
    for frag in surf_frags_1:
        complete_ghost_frags.extend(frag.get_ghost_fragments(ijk_frag_db.indexed_ghosts, oligomer1_backbone_cb_tree))

    # calculate the initial match type by finding the predominant surface type
    # log.debug('Length of surface_frags1: %d' % len(surf_frags_1))
    # log.debug('Length of complete_ghost_frags1: %d' % len(complete_ghost_frags))
    # # frag_types1 = [ghost_frag1.i_type for ghost_frag1 in complete_ghost_frags]
    # # fragment_content1 = [frag_types1.count(frag_type) for frag_type in range(1, fragment_length + 1)]
    # # log.debug('Found oligomer 1 i fragment content: %s' % fragment_content1)
    frag_types1_j = [ghost_frag1.j_type for ghost_frag1 in complete_ghost_frags]
    fragment_content1_j = [frag_types1_j.count(frag_type) for frag_type in range(1, fragment_length + 1)]
    # log.debug('Found oligomer 1 j fragment content: %s' % fragment_content1_j)
    ghost_frags = [ghost_frag1 for ghost_frag1 in complete_ghost_frags if ghost_frag1.j_type == initial_type2]
    ghost_frag_guide_coords = [ghost_frag1.guide_coords for ghost_frag1 in ghost_frags]

    if not resume and keep_time:
        get_complete_ghost_frags_time_stop = time.time()
        get_complete_ghost_frags_time = get_complete_ghost_frags_time_stop - get_complete_ghost_frags_time_start
        log.info('Getting %s Oligomer 1 Ghost Fragments Using COMPLETE Fragment Database (took: %s s)\n'
                 % (pdb1_name, str(get_complete_ghost_frags_time)))

    # Check if the job was running but stopped. Resume where last left off
    degen1_count, degen2_count, rot1_count, rot2_count = 0, 0, 0, 0
    if resume:
        # degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log)
        degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log.handlers[0].baseFilename)

    if not resume:
        log.info('Obtaining Rotation/Degeneracy Matrices for Oligomer 1')

    # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
    # Ready to go for sampling nothing if rot_range_deg == 0
    rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", sym_entry.get_rot_range_deg_1())
    # Ready to go returning identity matrices if there is no sampling on either degen or rotation
    degen_rot_mat_1 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_1, rotation_matrices_1)

    if not resume:
        log.info('Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n')

    # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
    rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", sym_entry.get_rot_range_deg_2())
    degen_rot_mat_2 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_2, rotation_matrices_2)

    # Initialize Euler Lookup Class
    eul_lookup = EulerLookup()

    set_mat1 = np.array(sym_entry.get_rot_set_mat_group1())
    set_mat2 = np.array(sym_entry.get_rot_set_mat_group2())

    zshift1, zshift2 = None, None
    if sym_entry.is_internal_tx1():
        zshift1 = set_mat1[:, 2:3].T  # must be 2d array

    if sym_entry.is_internal_tx2():
        zshift2 = set_mat2[:, 2:3].T  # must be 2d array

    optimal_tx = OptimalTx.from_dof(sym_entry.get_ext_dof(), zshift1=zshift1, zshift2=zshift2)

    # Transpose Setting Matrices to Set Guide Coordinates just for Euler Lookup Using np.matmul
    set_mat1_np_t, set_mat2_np_t = np.transpose(set_mat1), np.transpose(set_mat2)
    for degen1 in degen_rot_mat_1[degen1_count:]:
        degen1_count += 1
        for rot1_mat in degen1[rot1_count:]:
            rot1_count += 1
            # Rotate Oligomer1 Ghost Fragment Guide Coordinates using rot1_mat
            ghost_frag_guide_coords_rot = np.matmul(ghost_frag_guide_coords, np.transpose(rot1_mat))
            ghost_frag_guide_coords_rot_and_set = np.matmul(ghost_frag_guide_coords_rot, set_mat1_np_t)
            for degen2 in degen_rot_mat_2[degen2_count:]:
                degen2_count += 1
                for rot2_mat in degen2[rot2_count:]:
                    rot2_count += 1
                    # Rotate Oligomer2 Surface Fragment Guide Coordinates using rot2_mat
                    surf_frags2_guide_coords_rot = np.matmul(initial_surf_frags2_guide_coords, np.transpose(rot2_mat))
                    surf_frags_2_guide_coords_rot_and_set = np.matmul(surf_frags2_guide_coords_rot, set_mat2_np_t)

                    log.info('\n***** OLIGOMER 1: Degeneracy %d Rotation %d | OLIGOMER 2: Degeneracy %d Rotation %d '
                             '*****' % (degen1_count, rot1_count, degen2_count, rot2_count))

                    # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 (rotated) Surface Fragment)
                    # guide coodinate pairs in the same Euler rotational space bucket
                    log.info('Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational '
                             'space bucket')

                    overlapping_ghost_frags, overlapping_surf_frags = \
                        eul_lookup.check_lookup_table(ghost_frag_guide_coords_rot_and_set,
                                                      surf_frags_2_guide_coords_rot_and_set)
                    # log.debug('number of matching euler angle pairs: %d' % len(overlapping_ghost_frags))

                    # Get optimal shift parameters for initial (Ghost Fragment, Surface Fragment) guide coordinate pairs
                    log.info('Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide '
                             'coordinate pairs')

                    passing_ghost_coords = ghost_frag_guide_coords_rot_and_set[overlapping_ghost_frags]
                    passing_surf_coords = surf_frags_2_guide_coords_rot_and_set[overlapping_surf_frags]
                    reference_rmsds = np.array([ghost_frags[ghost_idx].rmsd
                                                for ghost_idx in overlapping_ghost_frags.tolist()])
                    reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)
                    optimal_shifts = [optimal_tx.solve_optimal_shift(passing_ghost_coords[idx],
                                                                     passing_surf_coords[idx],
                                                                     reference_rmsds[idx],
                                                                     max_z_value=init_max_z_val)
                                      for idx in range(len(passing_ghost_coords))]

                    passing_optimal_shifts = [passing_shift for passing_shift in optimal_shifts
                                              if passing_shift is not None]

                    log.info('%s Initial Interface Fragment Match%s Found\n'
                             % (len(passing_optimal_shifts) if passing_optimal_shifts else 'No',
                                'es' if len(passing_optimal_shifts) != 1 else ''))

                    # Todo replace with DesignDirectory? Path object?
                    degen_subdir_out_path = os.path.join(outdir, 'DEGEN_%d_%d' % (degen1_count, degen2_count))
                    rot_subdir_out_path = os.path.join(degen_subdir_out_path, 'ROT_%d_%d' % (rot1_count, rot2_count))

                    find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, passing_optimal_shifts,
                                      complete_ghost_frags, complete_surf_frags, degen_subdir_out_path,
                                      rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup,
                                      rot_mat1=rot1_mat, rot_mat2=rot2_mat, max_z_val=subseq_max_z_val,
                                      output_assembly=output_assembly, output_surrounding_uc=output_surrounding_uc,
                                      min_matched=min_matched, log=log)
                rot2_count = 0
            degen2_count = 0
        rot1_count = 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Parsing Command Line Input
        sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir, output_assembly, \
            output_surrounding_uc, min_matched, timer, initial = get_docking_parameters(sys.argv)

        # Master Log File
        master_log_filepath = os.path.join(master_outdir, master_log)
        # SymEntry Parameters
        sym_entry = SymEntry(sym_entry_number)

        if initial:
            # make master output directory
            if not os.path.exists(master_outdir):
                os.makedirs(master_outdir)
            with open(master_log_filepath, 'w') as master_logfile:
                master_logfile.write('Nanohedra\nMODE: DOCK\n\n')
            write_docking_parameters(pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, sym_entry, master_outdir,
                                     master_log_filepath)
        else:
            # for parallel runs, ensure that the first file was able to write before adding below log
            time.sleep(1)
            rot_step_deg1, rot_step_deg2 = get_rotation_step(sym_entry, rot_step_deg1, rot_step_deg2)

        pdb1_name = os.path.basename(os.path.splitext(pdb1_path)[0])
        pdb2_name = os.path.basename(os.path.splitext(pdb2_path)[0])
        with open(master_log_filepath, 'a+') as master_log_file:
            master_log_file.write('Docking %s / %s \n' % (pdb1_name, pdb2_name))

        # Create fragment database for all ijk cluster representatives
        # Todo move to inside loop for single iteration docking
        ijk_frag_db = FragmentDB()

        # Get complete IJK fragment representatives database dictionaries
        ijk_frag_db.get_monofrag_cluster_rep_dict()
        ijk_frag_db.get_intfrag_cluster_rep_dict()
        ijk_frag_db.get_intfrag_cluster_info_dict()

        try:
            # Output Directory  # Todo DesignDirectory
            building_blocks = '%s_%s' % (pdb1_name, pdb2_name)
            outdir = os.path.join(master_outdir, building_blocks)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            log_file_path = os.path.join(outdir, '%s_log.txt' % building_blocks)
            bb_logger = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False)
            if os.path.exists(log_file_path):
                resume = True
                bb_logger.info('Found a prior incomplete run! Resuming from last sampled transformation.\n')
            else:
                resume = False

            # Write to Logfile
            if not resume:
                bb_logger.info('DOCKING %s TO %s' % (pdb1_name, pdb2_name))
                bb_logger.info('Oligomer 1 Path: %s\nOligomer 2 Path: %s\n' % (pdb1_path, pdb2_path))

            nanohedra_dock(sym_entry, ijk_frag_db, outdir, pdb1_path, pdb2_path,
                           rot_step_deg_pdb1=rot_step_deg1, rot_step_deg_pdb2=rot_step_deg2,
                           output_assembly=output_assembly, output_surrounding_uc=output_surrounding_uc,
                           min_matched=min_matched, log=bb_logger, resume=resume, keep_time=timer)

            with open(master_log_filepath, 'a+') as master_log_file:
                master_log_file.write('COMPLETE ==> %s\n\n'
                                      % os.path.join(master_outdir, '%s_%s' % (pdb1_name, pdb2_name)))

        except KeyboardInterrupt:
            with open(master_log_filepath, 'a+') as master_log:
                master_log.write('\nRun Ended By KeyboardInterrupt\n')
            exit(2)
