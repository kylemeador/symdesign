import time

from sklearn.neighbors import BallTree

# from Structure import get_surface_fragments
from PathUtils import frag_text_file
from SymDesignUtils import calculate_overlap, filter_euler_lookup_by_zvalue, match_score_from_z_value
from classes.EulerLookup import EulerLookup
from classes.Fragment import *
from classes.OptimalTx import *
from classes.SymEntry import *
from classes.SymEntry import get_optimal_external_tx_vector, get_rot_matrices, get_degen_rotmatrices
from classes.WeightedSeqFreq import FragMatchInfo, SeqFreqInfo
from interface_analysis.Database import FragmentDB
from utils.CmdLineArgParseUtils import *
from utils.ExpandAssemblyUtils import generate_cryst1_record, expanded_design_is_clash
from utils.GeneralUtils import get_last_sampling_state, write_frag_match_info_file, write_docked_pose_info
from utils.PDBUtils import *
from utils.SymmUtils import get_uc_dimensions


# Globals
fragment_length = 5


def find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, optimal_tx_params, complete_ghost_frag_np,
                      complete_surf_frag_np, log_filepath, degen_subdir_out_path, rot_subdir_out_path, pdb1_path,
                      pdb2_path, eul_lookup, rot_mat1=None, rot_mat2=None, max_z_val=2.0, output_exp_assembly=False,
                      output_uc=False, output_surrounding_uc=False, clash_dist=2.2, min_matched=3,
                      high_quality_match_value=1):
    """

    Keyword Args:
        high_quality_match_value=1 (int): when z-value used, 0.5 if match score is used
    Returns:
        None
    """

    # for tx_idx in range(len(optimal_tx_params)):
    for tx_idx, tx_parameters in enumerate(optimal_tx_params):
        with open(log_filepath, "a+") as log_file:
            log_file.write("Optimal Shift %d\n" % (tx_idx + 1))

        # tx_parameters contains [OptimalExternalDOFShifts (n_dof_ext), OptimalInternalDOFShifts (n_dof_int)]
        # tx_parameters = optimal_tx_params[tx_idx]
        # tx_parameters = optimal_tx_params[tx_idx][0]

        # Get Optimal External DOF shifts
        n_dof_external = len(sym_entry.get_ext_dof())  # returns 0 - 3
        optimal_ext_dof_shifts = None
        if n_dof_external > 0:
            optimal_ext_dof_shifts = tx_parameters[0:n_dof_external]

        copy_rot_tr_set_time_start = time.time()

        # Get Oligomer1 Optimal Internal Translation vector
        representative_int_dof_tx_param_1 = None
        if sym_entry.is_internal_tx1():
            representative_int_dof_tx_param_1 = [0, 0, tx_parameters[n_dof_external: n_dof_external + 1][0]]

        # Get Oligomer2 Optimal Internal Translation vector
        representative_int_dof_tx_param_2 = None
        if sym_entry.is_internal_tx2():
            representative_int_dof_tx_param_2 = [0, 0, tx_parameters[n_dof_external + 1: n_dof_external + 2][0]]

        representative_ext_dof_tx_params_1, representative_ext_dof_tx_params_2 = None, None
        ref_frame_var_is_pos, uc_dimensions = False, None
        if optimal_ext_dof_shifts:
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
                with open(log_filepath, "a+") as log_file:
                    log_file.write("\tReference Frame Shift Parameter(s) is/are Negative: %s\n\n"
                                   % optimal_ext_dof_shifts)
                continue
            else:  # not optimal_ext_dof_shifts or (optimal_ext_dof_shifts and ref_frame_var_is_pos)
                # write uc_dimensions and dock
                # Get Oligomer1 & Oligomer2 Optimal External Translation vector
                representative_ext_dof_tx_params_1 = get_optimal_external_tx_vector(
                    sym_entry.get_ref_frame_tx_dof_group1(),
                    optimal_ext_dof_shifts)
                representative_ext_dof_tx_params_2 = get_optimal_external_tx_vector(
                    sym_entry.get_ref_frame_tx_dof_group2(),
                    optimal_ext_dof_shifts)

                # Get Unit Cell Dimensions for 2D and 3D SCMs
                uc_dimensions = get_uc_dimensions(sym_entry.get_uc_spec_string(), *optimal_ext_dof_shifts)
                # uc_dimensions = get_uc_dimensions(sym_entry.get_uc_spec_string(), ref_frame_tx_dof_e,
                #                                   ref_frame_tx_dof_f,
                #                                   ref_frame_tx_dof_g)

        # Rotate, Translate and Set PDB1
        pdb1_copy = pdb1.return_transformed_copy(rotation=rot_mat1, translation=representative_int_dof_tx_param_1,
                                                 rotation2=sym_entry.get_rot_set_mat_group1(),
                                                 translation2=representative_ext_dof_tx_params_1)
        # print('copied PDB1')
        # pdb1_copy = rot_txint_set_txext_pdb(pdb1, rot_mat=rot_mat1,
        #                                     internal_tx_vec=representative_int_dof_tx_param_1,
        #                                     set_mat=sym_entry.get_rot_set_mat_group1(),
        #                                     ext_tx_vec=representative_ext_dof_tx_params_1)

        # Rotate, Translate and Set PDB2
        pdb2_copy = pdb2.return_transformed_copy(rotation=rot_mat2, translation=representative_int_dof_tx_param_2,
                                                 rotation2=sym_entry.get_rot_set_mat_group2(),
                                                 translation2=representative_ext_dof_tx_params_2)
        pdb1_copy.write(out_path=os.path.join(os.path.dirname(log_filepath), 'pdb1_copy.pdb'))
        pdb2_copy.write(out_path=os.path.join(os.path.dirname(log_filepath), 'pdb2_copy.pdb'))
        # print('copied PDB2')
        # pdb2_copy = rot_txint_set_txext_pdb(pdb2, rot_mat=rot_mat2,
        #                                     internal_tx_vec=representative_int_dof_tx_param_2,
        #                                     set_mat=sym_entry.get_rot_set_mat_group1(),
        #                                     ext_tx_vec=representative_ext_dof_tx_params_2)

        copy_rot_tr_set_time_stop = time.time()
        copy_rot_tr_set_time = copy_rot_tr_set_time_stop - copy_rot_tr_set_time_start
        with open(log_filepath, "a+") as log_file:
            # Todo logging debug
            log_file.write("\tCopy and Transform Oligomer1 and Oligomer2 (took: %s s)\n" % str(copy_rot_tr_set_time))

        # Check if PDB1 and PDB2 backbones clash
        oligomer1_oligomer2_clash_time_start = time.time()
        # Todo @profile and move to KDTree
        kdtree_oligomer1_backbone = BallTree(pdb1_copy.get_backbone_and_cb_coords())
        asu_cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.get_backbone_and_cb_coords(),
                                                                             [clash_dist])
        print('Checking clashes')
        oligomer1_oligomer2_clash_time_end = time.time()
        oligomer1_oligomer2_clash_time = oligomer1_oligomer2_clash_time_end - oligomer1_oligomer2_clash_time_start

        if asu_cb_clash_count[0] > 0:
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tBackbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)\n"
                               % str(oligomer1_oligomer2_clash_time))
            continue
        # else:
        with open(log_filepath, "a+") as log_file:
            log_file.write("\tNO Backbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)\n"
                           % str(oligomer1_oligomer2_clash_time))

        # Full Interface Fragment Match
        # Todo
        #  The use of hashing on the surface and ghost fragments could increase program runtime, over tuple call
        #   to the ghost_fragment objects to return the aligned chain and residue then test for membership...
        #  Is the chain necessary? Two chains can occupy interface, even the same residue could be used
        #   Think D2 symmetry
        #  Store all the ghost/surface frags in a chain/residue dictionary?
        get_int_ghost_surf_frags_time_start = time.time()
        transformed_ghostfrag_guide_coords_np, transformed_monofrag2_guide_coords_np, \
            unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2 = \
            get_interface_ghost_surf_frags(pdb1_copy, pdb2_copy, complete_ghost_frag_np,
                                           complete_surf_frag_np, rot_mat1, rot_mat2,
                                           representative_int_dof_tx_param_1, representative_int_dof_tx_param_2,
                                           sym_entry.get_rot_set_mat_group1(),
                                           sym_entry.get_rot_set_mat_group2(),
                                           representative_ext_dof_tx_params_1,
                                           representative_ext_dof_tx_params_2)
        print('Transformed guide_coords')
        get_int_ghost_surf_frags_time_end = time.time()
        get_int_ghost_surf_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start

        unique_total_monofrags_count = unique_interface_frag_count_pdb1 + unique_interface_frag_count_pdb2

        if unique_total_monofrags_count == 0:
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tNO Interface Mono Fragments Found\n")
            continue
        # else:
        with open(log_filepath, "a+") as log_file:
            log_file.write("\tNewly Formed Interface Contains %d Unique Fragments on Oligomer 1 and %d on "
                           "Oligomer 2\n\t(took: %s s to get interface surface and ghost fragments with "
                           "their guide coordinates)\n"
                           % (unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2,
                              str(get_int_ghost_surf_frags_time)))

        # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Mono Fragment) guide coodinate pairs
        # in the same Euler rotational space bucket
        eul_lookup_start_time = time.time()
        overlapping_ghost_frag_array, overlapping_surf_frag_array = \
            zip(*eul_lookup.check_lookup_table(transformed_ghostfrag_guide_coords_np,
                                               transformed_monofrag2_guide_coords_np))
        print('Euler lookup')
        eul_lookup_end_time = time.time()
        eul_lookup_time = eul_lookup_end_time - eul_lookup_start_time

        # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        overlap_score_time_start = time.time()
        # filter array by matching type for surface (i) and ghost (j) frags
        ij_type_match = [True if complete_surf_frag_np[overlapping_surf_frag_array[idx]].get_i_type() ==
                         complete_ghost_frag_np[ghost_frag_idx].get_j_type() else False
                         for idx, ghost_frag_idx in enumerate(overlapping_ghost_frag_array)]
        # get only those indices that pass ij filter and their associated coords
        # Todo numpy overlapping_ghost_frag_array so can [index] operation
        passing_ghost_indices = np.array([ghost_idx
                                          for idx, ghost_idx in enumerate(overlapping_ghost_frag_array)
                                          if ij_type_match[idx]])
        passing_ghost_coords = transformed_ghostfrag_guide_coords_np[passing_ghost_indices]

        passing_surf_indices = np.array([surf_idx
                                         for idx, surf_idx in enumerate(overlapping_surf_frag_array)
                                         if ij_type_match[idx]])
        passing_surf_coords = transformed_monofrag2_guide_coords_np[passing_surf_indices]
        # precalculate the reference_rmsds for each ghost fragment
        reference_rmsds = np.array([float(max(complete_ghost_frag_np[ghost_idx].get_rmsd(), 0.01))
                                    for ghost_idx in passing_ghost_indices])
        print('length of all coords arrays = %d, %d, %d' % (len(passing_ghost_coords), len(passing_surf_coords),
                                                            len(reference_rmsds)))
        all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
                                                 max_z_value=max_z_val)
        print('Checking all fragment overlap at interface')
        passing_overlaps = [idx for idx, overlap in enumerate(all_fragment_overlap) if overlap]
        # passing_z_values = [overlap for overlap in all_fragment_overlap if overlap]
        passing_z_values = all_fragment_overlap[passing_overlaps]
        print('Overlapping z-values: %s' % passing_z_values)
        # sorted_overlaps = np.array([passing_overlaps, passing_z_values],
        #                            dtype=[('index', int), ('z_value', float)])
        high_qual_match_count = sum([1 for z_value in passing_z_values
                                     if z_value <= high_quality_match_value])

        overlap_score_time_stop = time.time()
        overlap_score_time = overlap_score_time_stop - overlap_score_time_start

        with open(log_filepath, "a+") as log_file:
            log_file.write("\t%d Fragment Match(es) Found in Complete Cluster Representative Fragment "
                           "Library\n\t(Euler Lookup took %s s for %d fragment pairs and Overlap Score "
                           "Calculation took %s for %d fragment pairs)\n" %
                           (len(passing_overlaps), str(eul_lookup_time),
                            len(transformed_ghostfrag_guide_coords_np), str(overlap_score_time),
                            len(overlapping_ghost_frag_array)))

        if high_qual_match_count < min_matched:
            with open(log_filepath, "a+") as log_file:
                log_file.write("\t%s < %s Which is Set as the Minimal Required Amount of High Quality "
                               "Fragment Matches\n" % (str(high_qual_match_count), str(min_matched)))
                continue
        # else:
        # Get contacting PDB 1 ASU and PDB 2 ASU
        asu_pdb_1, asu_pdb_2 = get_contacting_asu(pdb1_copy, pdb2_copy)
        print('Grabbing asu')
        if not asu_pdb_1 and not asu_pdb_2:
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tNO Design ASU Found\n")
            continue
        # else:
        # Check if design has any clashes when expanded
        # Todo replace with DesignDirectory? Path object?
        tx_subdir_out_path = os.path.join(rot_subdir_out_path, "tx_%d" % (tx_idx + 1))
        oligomers_subdir = rot_subdir_out_path.split(os.sep)[-3]
        degen_subdir = rot_subdir_out_path.split(os.sep)[-2]
        rot_subdir = rot_subdir_out_path.split(os.sep)[-1]
        pose_id = "%s_%s_%s_TX_%d" % (oligomers_subdir, degen_subdir, rot_subdir, (tx_idx + 1))
        sampling_id = '%s_%s_TX_%d' % (degen_subdir, rot_subdir, (tx_idx + 1))
        exp_des_clash_time_start = time.time()
        exp_des_is_clash = expanded_design_is_clash(asu_pdb_1, asu_pdb_2, sym_entry.get_design_dim(),
                                                    sym_entry.get_result_design_sym(),
                                                    sym_entry.expand_matrices, uc_dimensions, tx_subdir_out_path,
                                                    output_exp_assembly, output_uc, output_surrounding_uc)
        print('Checked expand clash')
        exp_des_clash_time_stop = time.time()
        exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

        if exp_des_is_clash:
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tBackbone Clash when Designed Assembly is Expanded (took: %s s)\n"
                               % str(exp_des_clash_time))
            continue
        # else:
        with open(log_filepath, "a+") as log_file:
            log_file.write("\tNO Backbone Clash when Designed Assembly is Expanded (took: %s s "
                           "including writing)\n\tSUCCESSFUL DOCKED POSE: %s\n" %
                           (str(exp_des_clash_time), tx_subdir_out_path))
        # Todo replace with DesignDirectory? Path object?
        if not os.path.exists(degen_subdir_out_path):
            os.makedirs(degen_subdir_out_path)
        if not os.path.exists(rot_subdir_out_path):
            os.makedirs(rot_subdir_out_path)
        if not os.path.exists(tx_subdir_out_path):
            os.makedirs(tx_subdir_out_path)

        # Write PDB1 and PDB2 files
        cryst1_record = None
        if optimal_ext_dof_shifts is not None:
            cryst1_record = generate_cryst1_record(uc_dimensions, sym_entry.get_result_design_sym())
        pdb1_fname = os.path.splitext(os.path.basename(pdb1.get_filepath()))[0]
        pdb2_fname = os.path.splitext(os.path.basename(pdb2.get_filepath()))[0]
        pdb1_copy.write(os.path.join(tx_subdir_out_path, '%s_%s.pdb' % (pdb1_fname, sampling_id)))
        pdb2_copy.write(os.path.join(tx_subdir_out_path, '%s_%s.pdb' % (pdb2_fname, sampling_id)))

        # Todo replace with DesignDirectory? Path object?
        # Make directories to output matched fragment PDB files
        matched_frag_reps_outpath = os.path.join(tx_subdir_out_path, "matching_fragments")
        if not os.path.exists(matched_frag_reps_outpath):
            os.makedirs(matched_frag_reps_outpath)
        # high_qual_match for fragments that were matched with z values <= 1
        high_qual_matches_outpath = os.path.join(matched_frag_reps_outpath, 'high_qual_match')
        if not os.path.exists(high_qual_matches_outpath):
            os.makedirs(high_qual_matches_outpath)
        # low_qual_match for fragments that were matched with z values > 1
        low_qual_matches_outpath = os.path.join(matched_frag_reps_outpath, "low_qual_match")
        if not os.path.exists(low_qual_matches_outpath):
            os.makedirs(low_qual_matches_outpath)

        # return the indices sorted by z_value then pull information accordingly
        sorted_fragment_indices = np.argsort(passing_z_values)
        # match_scores = match_score_from_z_value(sorted_overlaps[1])
        # match_scores = match_score_from_z_value(passing_z_values)
        match_scores = match_score_from_z_value(passing_z_values[sorted_fragment_indices])
        print('Overlapping Match Scores: %s' % match_scores)
        # interface_ghostfrags = complete_ghost_frag_np[passing_ghost_indices[sorted_overlaps[0]]]
        # interface_monofrags2 = complete_surf_frag_np[passing_surf_indices[sorted_overlaps[0]]]
        # interface_ghostfrags = complete_ghost_frag_np[passing_ghost_indices[passing_overlaps]]
        # interface_monofrags2 = complete_surf_frag_np[passing_surf_indices[passing_overlaps]]
        interface_ghostfrags = complete_ghost_frag_np[passing_ghost_indices[passing_overlaps[sorted_fragment_indices]]]
        interface_monofrags2 = complete_surf_frag_np[passing_surf_indices[passing_overlaps[sorted_fragment_indices]]]
        ghostfrag_surffrag_pairs = [zip(interface_ghostfrags, interface_monofrags2)]

        # Write out initial match interface fragment
        # initial match is found in the complete search, no need to add
        # ghostfrag_surffrag_pairs = [ghostfrag_surffrag_pair]
        # ghostfrag_surffrag_pairs += zip(interface_ghostfrags, interface_monofrags2)
        # ghostfrag_surffrag_pairs = [(complete_ghost_frag_np[passing_ghost_indices[idx]],
        #                              complete_surf_frag_np[passing_surf_indices[idx]])
        #                             for idx in passing_overlaps]

        # For all matched interface fragments
        # Dictionaries for PDB1 and PDB2 with (ch_id, res_num) tuples as keys for every residue that is covered by at
        # least 1 matched fragment. Dictionary values are lists containing 1 / (1 + z^2) values for every fragment match
        # that covers the (ch_id, res_num) residue.
        chid_resnum_scores_dict_pdb1, chid_resnum_scores_dict_pdb2 = {}, {}

        # Number of unique interface mono fragments matched with a z value <= 1 ('high quality match'). This value has
        # to be >= min_matched (minimum number of high quality matches required)for a pose to be selected
        pdb1_unique_monofrags_info, pdb2_unique_monofrags_info = [], []

        # Keep track of match information and residue pair frequencies for each fragment match this information will be
        # used to calculate a weighted frequency average for all central residues of matched fragments
        res_pair_freq_info_list = []
        for frag_idx, (interface_ghost_frag, interface_mono_frag) in enumerate(ghostfrag_surffrag_pairs):
            ghostfrag_i_type = interface_ghost_frag.get_i_type()
            ghostfrag_j_type = interface_ghost_frag.get_j_type()
            ghostfrag_k_type = interface_ghost_frag.get_k_type()

            interface_ghost_frag_cluster_res_freq_list = \
                ijk_frag_db.info[ghostfrag_i_type][ghostfrag_j_type][ghostfrag_k_type].get_central_residue_pair_freqs()
            pdb1_surffrag_chain, pdb1_surffrag_central_res_num = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
            pdb2_surffrag_chain, pdb2_surffrag_central_res_num = interface_mono_frag.get_central_res_tup()

            covered_residues_pdb1 = [(pdb1_surffrag_chain, pdb1_surffrag_central_res_num + j) for j in range(-2, 3)]
            covered_residues_pdb2 = [(pdb2_surffrag_chain, pdb2_surffrag_central_res_num + j) for j in range(-2, 3)]
            score_term = match_scores[frag_idx]
            for k in range(5):
                chid1, resnum1 = covered_residues_pdb1[k]
                chid2, resnum2 = covered_residues_pdb2[k]
                if (chid1, resnum1) not in chid_resnum_scores_dict_pdb1:
                    chid_resnum_scores_dict_pdb1[(chid1, resnum1)] = [score_term]
                else:
                    chid_resnum_scores_dict_pdb1[(chid1, resnum1)].append(score_term)

                if (chid2, resnum2) not in chid_resnum_scores_dict_pdb2:
                    chid_resnum_scores_dict_pdb2[(chid2, resnum2)] = [score_term]
                else:
                    chid_resnum_scores_dict_pdb2[(chid2, resnum2)].append(score_term)

            if (pdb1_surffrag_chain, pdb1_surffrag_central_res_num) not in pdb1_unique_monofrags_info:
                pdb1_unique_monofrags_info.append((pdb1_surffrag_chain, pdb1_surffrag_central_res_num))

            if (pdb2_surffrag_chain, pdb2_surffrag_central_res_num) not in pdb2_unique_monofrags_info:
                pdb2_unique_monofrags_info.append((pdb2_surffrag_chain, pdb2_surffrag_central_res_num))

            z_value = passing_z_values[frag_idx]
            if z_value <= 1:  # the overlap z-value has is greater than 1 std deviation
                matched_frag_outdir_path = high_qual_matches_outpath
            else:
                matched_frag_outdir_path = low_qual_matches_outpath

            # if write_frags:
            # write out aligned cluster representative fragment
            transformed_ghost_fragment = interface_ghost_frag.structure.return_transformed_copy(
                rotation=rot_mat1, translation=representative_int_dof_tx_param_1,
                rotation2=sym_entry.get_rot_set_mat_group1(), translation2=representative_ext_dof_tx_params_1)
            transformed_ghost_fragment.write(os.path.join(matched_frag_outdir_path, 'int_frag_%s_%d.pdb'
                                                          % ('i%s_j%s_k%s' % interface_ghost_frag.get_ijk(),
                                                             frag_idx + 1)))

            # write out associated match information to frag_match_info_file.txt
            write_frag_match_info_file(ghost_frag=interface_ghost_frag, matched_frag=interface_mono_frag,
                                       overlap_error=z_value, match_number=frag_idx + 1,
                                       central_frequencies=interface_ghost_frag_cluster_res_freq_list,
                                       out_path=matched_frag_reps_outpath, pose_id=pose_id)

            res_pair_freq_info_list.append(FragMatchInfo(interface_ghost_frag_cluster_res_freq_list,
                                                         pdb1_surffrag_chain, pdb1_surffrag_central_res_num,
                                                         pdb2_surffrag_chain, pdb2_surffrag_central_res_num, z_value))

        # calculate weighted frequency for central residues
        # write out weighted frequencies to frag_match_info_file.txt
        weighted_seq_freq_info = SeqFreqInfo(res_pair_freq_info_list)
        weighted_seq_freq_info.write(os.path.join(matched_frag_reps_outpath, frag_text_file))

        unique_matched_monofrag_count = len(pdb1_unique_monofrags_info) + len(pdb2_unique_monofrags_info)
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
        write_docked_pose_info(tx_subdir_out_path, res_lev_sum_score, high_qual_match_count,
                               unique_matched_monofrag_count, unique_total_monofrags_count,percent_of_interface_covered,
                               rot_mat1, representative_int_dof_tx_param_1, sym_entry.get_rot_set_mat_group1(),
                               representative_ext_dof_tx_params_1, rot_mat2, representative_int_dof_tx_param_2,
                               sym_entry.get_rot_set_mat_group2(), representative_ext_dof_tx_params_2, cryst1_record,
                               pdb1_path, pdb2_path, pose_id)


# KM TODO ijk_intfrag_cluster_info_dict contains all info in init_intfrag_cluster_info_dict. init info could be deleted,
#     This doesn't take up much extra memory, but makes future maintanence bad, for porting frags to fragDB say...
def nanohedra(sym_entry_number, pdb1_path, pdb2_path, rot_step_deg_pdb1, rot_step_deg_pdb2, master_outdir,
              output_exp_assembly, output_uc, output_surrounding_uc, min_matched, keep_time=True,
              main_log=False):

    # Fragment Database Directory Paths
    # frag_db = PUtils.frag_directory['biological_interfaces']  # Todo make dynamic at startup or use all fragDB

    # SymEntry Parameters
    sym_entry = SymEntry(sym_entry_number)

    # oligomer_symmetry_1 = sym_entry.get_group1_sym()
    # oligomer_symmetry_2 = sym_entry.get_group2_sym()
    # design_symmetry_pg = sym_entry.get_pt_grp_sym()

    # rot_range_deg_pdb1 = sym_entry.get_rot_range_deg_1()
    # rot_range_deg_pdb2 = sym_entry.get_rot_range_deg_2()

    set_mat1 = sym_entry.get_rot_set_mat_group1()
    set_mat2 = sym_entry.get_rot_set_mat_group2()

    # is_zshift1 = sym_entry.is_internal_tx1()
    # is_zshift2 = sym_entry.is_internal_tx2()

    # is_internal_rot1 = sym_entry.is_internal_rot1()
    # is_internal_rot2 = sym_entry.is_internal_rot2()

    design_dim = sym_entry.get_design_dim()

    # ref_frame_tx_dof1 = sym_entry.get_ref_frame_tx_dof_group1()
    # ref_frame_tx_dof2 = sym_entry.get_ref_frame_tx_dof_group2()

    result_design_sym = sym_entry.get_result_design_sym()
    uc_spec_string = sym_entry.get_uc_spec_string()

    # Default Fragment Guide Atom Overlap Z-Value Threshold For Initial Matches
    init_max_z_val = 1.0

    # Default Fragment Guide Atom Overlap Z-Value Threshold For All Subsequent Matches
    subseq_max_z_val = 2.0

    # Todo move all of this logging to logger and use a propogate=True flag to pass this info to the master log
    #  This will allow the variable unpacked above to be unpacked in the docking section
    if main_log:
        with open(master_log_filepath, "a+") as master_log_file:
            if sym_entry.is_internal_rot1():  # if rotation step required
                if not rot_step_deg_pdb1:
                    rot_step_deg_pdb1 = 3  # set rotation step to default
            else:
                rot_step_deg_pdb1 = 1
                if rot_step_deg_pdb1:
                    master_log_file.write("Warning: Specified Rotation Step 1 Was Ignored. Oligomer 1 Doesn\'t Have"
                                          " Internal Rotational DOF\n\n")
            if sym_entry.is_internal_rot2():  # if rotation step required
                if not rot_step_deg_pdb2:
                    rot_step_deg_pdb2 = 3  # set rotation step to default
            else:
                rot_step_deg_pdb2 = 1
                if rot_step_deg_pdb2:
                    master_log_file.write("Warning: Specified Rotation Step 2 Was Ignored. Oligomer 2 Doesn\'t Have"
                                          " Internal Rotational DOF\n\n")

            master_log_file.write("NANOHEDRA PROJECT INFORMATION\n")
            master_log_file.write("Oligomer 1 Input Directory: %s\n" % pdb1_path)
            master_log_file.write("Oligomer 2 Input Directory: %s\n" % pdb2_path)
            master_log_file.write("Master Output Directory: %s\n\n" % master_outdir)

            master_log_file.write("SYMMETRY COMBINATION MATERIAL INFORMATION\n")
            master_log_file.write("Nanohedra Entry Number: %s\n" % str(sym_entry_number))
            master_log_file.write("Oligomer 1 Point Group Symmetry: %s\n" % sym_entry.get_group1_sym())
            master_log_file.write("Oligomer 2 Point Group Symmetry: %s\n" % sym_entry.get_group2_sym())
            master_log_file.write("SCM Point Group Symmetry: %s\n" % sym_entry.get_pt_grp_sym())

            master_log_file.write("Oligomer 1 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot1()))
            master_log_file.write("Oligomer 2 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot2()))
            master_log_file.write("Oligomer 1 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx1()))
            master_log_file.write("Oligomer 2 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx2()))
            master_log_file.write("Oligomer 1 Setting Matrix: %s\n" % set_mat1)
            master_log_file.write("Oligomer 2 Setting Matrix: %s\n" % set_mat2)
            master_log_file.write("Oligomer 1 Reference Frame Tx DOF: %s\n"
                                  % sym_entry.get_ref_frame_tx_dof_group1()
                                  if sym_entry.is_ref_frame_tx_dof1() else None)
            master_log_file.write("Oligomer 2 Reference Frame Tx DOF: %s\n"
                                  % sym_entry.get_ref_frame_tx_dof_group2()
                                  if sym_entry.is_ref_frame_tx_dof2() else None)
            master_log_file.write("Resulting SCM Symmetry: %s\n" % result_design_sym)
            master_log_file.write("SCM Dimension: %s\n" % str(design_dim))
            master_log_file.write("SCM Unit Cell Specification: %s\n\n" % uc_spec_string)

            master_log_file.write("ROTATIONAL SAMPLING INFORMATION\n")
            master_log_file.write(
                "Oligomer 1 ROT Sampling Range: %s\n" % str(sym_entry.get_rot_range_deg_1())
                if sym_entry.is_internal_rot1() else str(None))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Range: %s\n" % str(sym_entry.get_rot_range_deg_2())
                if sym_entry.is_internal_rot2() else str(None))
            master_log_file.write(
                "Oligomer 1 ROT Sampling Step: %s\n" % (str(rot_step_deg_pdb1) if sym_entry.is_internal_rot1()
                                                        else None))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Step: %s\n\n" % (str(rot_step_deg_pdb2) if sym_entry.is_internal_rot2()
                                                          else None))

            # Get Degeneracy Matrices
            master_log_file.write("Searching For Possible Degeneracies\n")
            if sym_entry.degeneracy_matrices_1:
                num_degens = len(sym_entry.degeneracy_matrices_1)
                master_log_file.write("%d Degenerac%s Found for Oligomer 1\n"
                                      % (num_degens, 'ies' if num_degens > 1 else 'y'))
            else:
                master_log_file.write("No Degeneracies Found for Oligomer 1\n")

            if sym_entry.degeneracy_matrices_2:
                num_degens = len(sym_entry.degeneracy_matrices_2)
                master_log_file.write("%d Degenerac%s Found for Oligomer 2\n"
                                      % (num_degens, 'ies' if num_degens > 1 else 'y'))
            else:
                master_log_file.write("No Degeneracies Found for Oligomer 2\n")

            # Get Initial Fragment Database
            master_log_file.write("Retrieving Database of Complete Interface Fragment Cluster Representatives\n")
            # if init_match_type == "1_2":
            #     master_log_file.write("Retrieving Database of Helix-Strand Interface Fragment Cluster "
            #                           "Representatives\n\n")
            # elif init_match_type == "2_1":
            #     master_log_file.write("Retrieving Database of Strand-Helix Interface Fragment Cluster "
            #                           "Representatives\n\n")
            # elif init_match_type == "2_2":
            #     master_log_file.write("Retrieving Database of Strand-Strand Interface Fragment Cluster "
            #                           "Representatives\n\n")
            # else:
            #     master_log_file.write("Retrieving Database of Helix-Helix Interface Fragment Cluster "
            #                           "Representatives\n\n")

    # Create fragment database for all ijk cluster representatives
    # Todo move to inside loop for single iteration docking
    ijk_frag_db = FragmentDB()

    # Get complete IJK fragment representatives database dictionaries
    ijk_frag_db.get_monofrag_cluster_rep_dict()
    ijk_frag_db.get_intfrag_cluster_rep_dict()
    ijk_frag_db.get_intfrag_cluster_info_dict()

    with open(master_log_filepath, "a+") as master_log_file:
        master_log_file.write("Docking %s / %s \n" % (os.path.basename(os.path.splitext(pdb1_path)[0]),
                                                      os.path.basename(os.path.splitext(pdb2_path)[0])))

    nanohedra_dock(sym_entry, ijk_frag_db, master_outdir, pdb1_path, pdb2_path, init_max_z_val, subseq_max_z_val,
                   rot_step_deg_pdb1=rot_step_deg_pdb1, rot_step_deg_pdb2=rot_step_deg_pdb2,
                   output_exp_assembly=output_exp_assembly, output_uc=output_uc,
                   output_surrounding_uc=output_surrounding_uc, min_matched=min_matched, keep_time=keep_time)


def nanohedra_dock(sym_entry, ijk_frag_db, master_outdir, pdb1_path, pdb2_path, init_max_z_val=1.0,
                   subseq_max_z_val=2.0, rot_step_deg_pdb1=1, rot_step_deg_pdb2=1, output_exp_assembly=False,
                   output_uc=False, output_surrounding_uc=False, min_matched=3, keep_time=True):
    # Initialize Euler Lookup Class
    eul_lookup = EulerLookup()

    # Output Directory  # Todo DesignDirectory
    pdb1_name = os.path.splitext(os.path.basename(pdb1_path))[0]
    pdb2_name = os.path.splitext(os.path.basename(pdb2_path))[0]
    outdir = os.path.join(master_outdir, '%s_%s' % (pdb1_name, pdb2_name))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #################################
    log_file_path = os.path.join(outdir, '%s_%s_log.txt' % (pdb1_name, pdb2_name))
    if os.path.exists(log_file_path):
        resume = True
    else:
        resume = False

    # Write to Logfile
    if not resume:
        with open(log_file_path, "a+") as log_file:  # Todo make , 'w') as log_file
            log_file.write("DOCKING %s TO %s\nOligomer 1 Path: %s\nOligomer 2 Path: %s\nOutput Directory: %s\n\n"
                           % (pdb1_name, pdb2_name, pdb1_path, pdb2_path, outdir))

    # Get PDB2 Symmetric Building Block
    pdb2 = PDB.from_file(pdb2_path)
    surf_frags_2 = pdb2.get_fragments(residue_numbers=pdb2.get_surface_residues())

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting Oligomer 2 Surface Fragments Using COMPLETE Fragment Database")
        if keep_time:
            get_complete_surf_frags_time_start = time.time()

    complete_surf_frag_list = []
    for frag2 in surf_frags_2:
        monofrag2 = MonoFragment(frag2, ijk_frag_db.reps)
        if monofrag2.get_i_type():
            complete_surf_frag_list.append(monofrag2)

    # calculate the initial match type by finding the predominant surface type
    frag_types2 = [monofrag2.get_i_type() for monofrag2 in complete_surf_frag_list]
    fragment_content2 = [frag_types2.count(str(frag_type)) for frag_type in range(1, fragment_length + 1)]
    print('Found oligomer 2 fragment content: %s' % fragment_content2)
    initial_type2 = str(np.argmax(fragment_content2) + 1)
    # print('Found fragment initial type oligomer 2: %s' % initial_type2)
    surf_frag_list = [monofrag2 for monofrag2 in complete_surf_frag_list if monofrag2.get_i_type() == initial_type2]
    surf_frags2_guide_coords = [surf_frag.get_guide_coords() for surf_frag in surf_frag_list]

    surf_frag_np = np.array(surf_frag_list)
    complete_surf_frag_np = np.array(complete_surf_frag_list)

    if not resume and keep_time:
        get_complete_surf_frags_time_stop = time.time()
        get_complete_surf_frags_time = get_complete_surf_frags_time_stop - get_complete_surf_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n\n" % str(get_complete_surf_frags_time))

    # Get PDB1 Symmetric Building Block
    pdb1 = PDB.from_file(pdb1_path)
    surf_frags_1 = pdb1.get_fragments(residue_numbers=pdb1.get_surface_residues())
    oligomer1_backbone_cb_tree = BallTree(pdb1.get_backbone_and_cb_coords())

    # Get Oligomer1 Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting %s Oligomer 1 Ghost Fragments Using COMPLETE Fragment Database" % pdb1_name)
        if keep_time:
            get_complete_ghost_frags_time_start = time.time()

    complete_ghost_frag_list = []
    for frag1 in surf_frags_1:
        monofrag1 = MonoFragment(frag1, ijk_frag_db.reps)
        if monofrag1.get_i_type():
            complete_ghost_frag_list.extend(monofrag1.get_ghost_fragments(ijk_frag_db.paired_frags,
                                                                          oligomer1_backbone_cb_tree, ijk_frag_db.info))

    # calculate the initial match type by finding the predominant surface type
    print('Length of surface1_frags: %d' % len(surf_frags_1))
    print('Length of complete_ghost_frags: %d' % len(complete_ghost_frag_list))
    frag_types1 = [ghost_frag1.get_i_type() for ghost_frag1 in complete_ghost_frag_list]
    fragment_content1 = [frag_types1.count(str(frag_type)) for frag_type in range(1, fragment_length + 1)]
    print('Found oligomer 1 i fragment content: %s' % fragment_content1)
    # initial_type1 = str(np.argmax(fragment_content1) + 1)
    # print('Found initial fragment type oligomer 1: %s' % initial_type1)
    frag_types1_j = [ghost_frag1.get_j_type() for ghost_frag1 in complete_ghost_frag_list]
    fragment_content1_j = [frag_types1_j.count(str(frag_type)) for frag_type in range(1, fragment_length + 1)]
    print('Found oligomer 1 j fragment content: %s' % fragment_content1_j)
    ghost_frags = [ghost_frag1 for ghost_frag1 in complete_ghost_frag_list if ghost_frag1.get_j_type() == initial_type2]
    ghost_frag_guide_coords = [ghost_frag1.get_guide_coords() for ghost_frag1 in ghost_frags]

    ghost_frag_np = np.array(ghost_frags)
    complete_ghost_frag_np = np.array(complete_ghost_frag_list)

    if not resume and keep_time:
        get_complete_ghost_frags_time_stop = time.time()
        get_complete_ghost_frags_time = get_complete_ghost_frags_time_stop - get_complete_ghost_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n" % str(get_complete_ghost_frags_time))

    # After this, the entire fragment database is unnecessary. De-referencing from memory
    # del ijk_monofrag_cluster_rep_pdb_dict, init_monofrag_cluster_rep_pdb_dict_1, init_monofrag_cluster_rep_pdb_dict_2

    # Check if the job was running but stopped. Resume where last left off
    degen1_count, degen2_count, rot1_count, rot2_count = 0, 0, 0, 0
    if resume:
        degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log_file_path)
        with open(log_file_path, "a+") as log_file:
            log_file.write('Job was run with the \'-resume\' flag. Resuming from last sampled rotational space!\n')

    # if (sym_entry.degeneracy_matrices_1 is None and not has_int_rot_dof_1) \
    #         and (sym_entry.degeneracy_matrices_2 is None and not has_int_rot_dof_2):
    #     rot1_mat = None
    #     rot2_mat = None
    #     if not resume:
    #         with open(log_file_path, "a+") as log_file:
    #             # No Degeneracies/Rotation Matrices to get for Oligomer 1 or Oligomer2
    #             log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1\n")
    #             log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n\n")
    #
    #     with open(log_file_path, "a+") as log_file:
    #         log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****"
    #                        % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")
    #         # Get (Oligomer1 Ghost Fragment, Oligomer2 Surface Fragment) guide coodinate pairs in the same Euler
    #         # rotational space bucket
    #         log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational "
    #                        "space bucket\n")
    #
    #     ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords, set_mat1_np_t)
    #     surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags2_guide_coords, set_mat2_np_t)
    #
    #     eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_set_for_eul,
    #                                                                surf_frags_2_guide_coords_list_set_for_eul)
    #     eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]
    #
    #     # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
    #     with open(log_file_path, "a+") as log_file:
    #         log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide "
    #                        "coordinate pairs\n")
    #
    #     optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
    #     optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frags,
    #                                                        ghost_frag_guide_coords,
    #                                                        surf_frag_list,
    #                                                        surf_frags2_guide_coords,
    #                                                        z_value_func=optimal_tx.apply,
    #                                                        max_z_value=init_max_z_val)
    #
    #     passing_optimal_shifts = list(filter(None, optimal_shifts))
    #     ghostfrag_surffrag_pairs = [(ghost_frags[eul_lookup_true_list[idx][0]],
    #                                  surf_frag_list[eul_lookup_true_list[idx][1]])
    #                                 for idx, boolean in enumerate(optimal_shifts) if boolean]
    #
    #     if len(passing_optimal_shifts) == 0:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("No Initial Interface Fragment Matches Found\n\n")
    #     else:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("%d Initial Interface Fragment Match(es) Found\n"
    #                            % len(passing_optimal_shifts))
    #
    #     degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
    #     rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
    #                                        (rot1_count, rot2_count))
    #
    #     find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, passing_optimal_shifts,
    #                       complete_ghost_frag_list, complete_surf_frag_list, log_file_path, degen_subdir_out_path,
    #                       rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup, rot1_mat, rot2_mat,
    #                       max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly, output_uc=output_uc,
    #                       output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
    #
    # elif (sym_entry.degeneracy_matrices_1 is not None or has_int_rot_dof_1) \
    #         and (sym_entry.degeneracy_matrices_2 is None and not has_int_rot_dof_2):
    #     # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
    #     if not resume:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1\n")
    #     rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", rot_range_deg_pdb1)
    #     degen_rot_mat_1 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_1, rotation_matrices_1)
    #
    #     # No Degeneracies/Rotation Matrices to get for Oligomer2
    #     rot2_mat = None
    #     if not resume:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n\n")
    #     surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags2_guide_coords, set_mat2_np_t)
    #
    #     optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
    #     # for degen1 in degen_rot_mat_1[degen1_count:]:
    #     #     degen1_count += 1
    #     #     for rot1_mat in degen1[rot1_count:]:
    #     #         rot1_count += 1
    #     for degen1 in degen_rot_mat_1[degen1_count:]:
    #         degen1_count += 1
    #         for rot1_mat in degen1[rot1_count:]:
    #             rot1_count += 1
    #             # Rotate Oligomer1 Ghost Fragment Guide Coodinates using rot1_mat
    #             rot1_mat_np_t = np.transpose(rot1_mat)
    #             ghost_frag_guide_coords_rot = np.matmul(ghost_frag_guide_coords, rot1_mat_np_t)
    #             ghost_frag_guide_coords_list_rot = ghost_frag_guide_coords_rot.tolist()
    #
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | "
    #                                "OLIGOMER 2: Degeneracy %s Rotation %s *****\n"
    #                                % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)))
    #
    #             # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 Surface Fragment)
    #             # guide coodinate pairs in the same Euler rotational space bucket
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler "
    #                                "rotational space bucket\n")
    #
    #             ghost_frag_guide_coords_rot_and_set = np.matmul(ghost_frag_guide_coords_list_rot,
    #                                                                          set_mat1_np_t)
    #
    #             eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(
    #                 ghost_frag_guide_coords_rot_and_set, surf_frags_2_guide_coords_list_set_for_eul)
    #             eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
    #                                     true_tup[2]]
    #
    #             # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide"
    #                                "coordinate pairs\n")
    #
    #             optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frags,
    #                                                                ghost_frag_guide_coords_list_rot,
    #                                                                surf_frag_list,
    #                                                                surf_frags2_guide_coords,
    #                                                                z_value_func=optimal_tx.apply,
    #                                                                max_z_value=init_max_z_val)
    #
    #             passing_optimal_shifts = list(filter(None, optimal_shifts))
    #             ghostfrag_surffrag_pairs = [(ghost_frags[eul_lookup_true_list[idx][0]],
    #                                          surf_frag_list[eul_lookup_true_list[idx][1]])
    #                                         for idx, boolean in enumerate(optimal_shifts) if boolean]
    #
    #             if len(passing_optimal_shifts) == 0:
    #                 with open(log_file_path, "a+") as log_file:
    #                     log_file.write("No Initial Interface Fragment Matches Found\n\n")
    #             else:
    #                 with open(log_file_path, "a+") as log_file:
    #                     log_file.write("%d Initial Interface Fragment Match(es) Found\n"
    #                                    % len(passing_optimal_shifts))
    #
    #             degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
    #             rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
    #                                                (rot1_count, rot2_count))
    #
    #             find_docked_poses(sym_entry, ijk_intfrag_cluster_info_dict, pdb1, pdb2, passing_optimal_shifts,
    #                               complete_ghost_frag_list, complete_surf_frag_list, log_file_path,
    #                               degen_subdir_out_path, rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup,
    #                               rot1_mat, rot2_mat, max_z_val=subseq_max_z_val,
    #                               output_exp_assembly=output_exp_assembly, output_uc=output_uc,
    #                               output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
    #         rot1_count = 0
    #
    # elif (sym_entry.degeneracy_matrices_1 is None and not has_int_rot_dof_1) \
    #         and (sym_entry.degeneracy_matrices_2 is not None or has_int_rot_dof_2):
    #     # No Degeneracies/Rotation Matrices to get for Oligomer1
    #     rot1_mat = None
    #     if not resume:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1\n")
    #     ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords, set_mat1_np_t)
    #
    #     # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
    #     if not resume:
    #         with open(log_file_path, "a+") as log_file:
    #             log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n\n")
    #     rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", rot_range_deg_pdb2)
    #     degen_rot_mat_2 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_2, rotation_matrices_2)
    #
    #     optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
    #     for degen2 in degen_rot_mat_2[degen2_count:]:
    #         degen2_count += 1
    #         for rot2_mat in degen2[rot2_count:]:
    #             rot2_count += 1
    #             # Rotate Oligomer2 Surface Fragment Guide Coodinates using rot2_mat
    #             rot2_mat_np_t = np.transpose(rot2_mat)
    #             surf_frags2_guide_coords_rot = np.matmul(surf_frags2_guide_coords,
    #                                                               rot2_mat_np_t)
    #             surf_frags_2_guide_coords_list_rot = surf_frags2_guide_coords_rot.tolist()
    #
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | "
    #                                "OLIGOMER 2: Degeneracy %s Rotation %s *****\n"
    #                                % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)))
    #
    #             # Get (Oligomer1 Ghost Fragment, Oligomer2 (rotated) Surface Fragment) guide
    #             # coodinate pairs in the same Euler rotational space bucket
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler "
    #                                "rotational space bucket\n")
    #
    #             surf_frags_2_guide_coords_rot_and_set = np.matmul(surf_frags_2_guide_coords_list_rot,
    #                                                                            set_mat2_np_t)
    #
    #             eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(
    #                 ghost_frag_guide_coords_list_set_for_eul, surf_frags_2_guide_coords_rot_and_set)
    #             eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
    #                                     true_tup[2]]
    #
    #             # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
    #             with open(log_file_path, "a+") as log_file:
    #                 log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide"
    #                                " coordinate pairs\n")
    #
    #             optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frags,
    #                                                                ghost_frag_guide_coords,
    #                                                                surf_frag_list,
    #                                                                surf_frags_2_guide_coords_list_rot,
    #                                                                z_value_func=optimal_tx.apply,
    #                                                                max_z_value=init_max_z_val)
    #
    #             passing_optimal_shifts = list(filter(None, optimal_shifts))
    #             ghostfrag_surffrag_pairs = [(ghost_frags[eul_lookup_true_list[idx][0]],
    #                                          surf_frag_list[eul_lookup_true_list[idx][1]])
    #                                         for idx, boolean in enumerate(optimal_shifts) if boolean]
    #
    #             if len(passing_optimal_shifts) == 0:
    #                 with open(log_file_path, "a+") as log_file:
    #                     log_file.write("No Initial Interface Fragment Matches Found\n\n")
    #             else:
    #                 with open(log_file_path, "a+") as log_file:
    #                     log_file.write("%d Initial Interface Fragment Match(es) Found\n"
    #                                    % len(passing_optimal_shifts))
    #
    #             degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
    #             rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
    #                                                (rot1_count, rot2_count))
    #
    #             find_docked_poses(sym_entry, ijk_intfrag_cluster_info_dict, pdb1, pdb2, passing_optimal_shifts,
    #                               complete_ghost_frag_list, complete_surf_frag_list, log_file_path,
    #                               degen_subdir_out_path, rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup,
    #                               rot1_mat, rot2_mat, max_z_val=subseq_max_z_val,
    #                               output_exp_assembly=output_exp_assembly, output_uc=output_uc,
    #                               output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
    #         rot2_count = 0
    #
    # elif (sym_entry.degeneracy_matrices_1 is not None or has_int_rot_dof_1) \
    #         and (sym_entry.degeneracy_matrices_2 is not None or has_int_rot_dof_2):
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1\n")

    # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
    # Ready to go for sampling nothing if rot_range_deg == 0
    rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", sym_entry.get_rot_range_deg_1())
    # Ready to go returning identity matrices if there is no sampling on either degen or rotation
    degen_rot_mat_1 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_1, rotation_matrices_1)
    # print(degen_rot_mat_1)
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n\n")

    # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
    rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", sym_entry.get_rot_range_deg_2())
    degen_rot_mat_2 = get_degen_rotmatrices(sym_entry.degeneracy_matrices_2, rotation_matrices_2)

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
                    # print('rotation matrix 2: %s' % rot2_mat)
                    surf_frags2_guide_coords_rot = np.matmul(surf_frags2_guide_coords, np.transpose(rot2_mat))
                    surf_frags_2_guide_coords_rot_and_set = np.matmul(surf_frags2_guide_coords_rot, set_mat2_np_t)

                    with open(log_file_path, "a+") as log_file:
                        log_file.write("\n***** OLIGOMER 1: Degeneracy %d Rotation %d | OLIGOMER 2: Degeneracy %d "
                                       "Rotation %d *****\n" % (degen1_count, rot1_count, degen2_count, rot2_count))

                    # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 (rotated) Surface Fragment)
                    # guide coodinate pairs in the same Euler rotational space bucket
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same "
                                       "Euler rotational space bucket\n")

                    # print('Set for Euler Lookup:', surf_frags_2_guide_coords_rot_and_set[:5])
                    print('number of ghost coords pre-lookup: %d' % len(ghost_frag_guide_coords_rot_and_set))

                    overlapping_ghost_frags, overlapping_surf_frags = \
                        zip(*eul_lookup.check_lookup_table(ghost_frag_guide_coords_rot_and_set,
                                                           surf_frags_2_guide_coords_rot_and_set))
                    overlap_pairs = list(zip(overlapping_ghost_frags, overlapping_surf_frags))
                    overlapping_ghost_frag_array = np.array(overlapping_ghost_frags)
                    overlapping_surf_frag_array = np.array(overlapping_surf_frags)
                    # print('euler overlapping ghost indices:', overlapping_ghost_frag_array[:5])
                    # print('euler overlapping surface indices:', overlapping_surf_frag_array[:5])
                    print('number of matching euler angle pairs: %d' % len(overlapping_ghost_frag_array))
                    print('matching euler angle pairs', overlap_pairs)

                    # eul_lookup_true_list = eul_lookup.check_lookup_table(
                    #     ghost_frag_guide_coords_rot_and_set,
                    #     surf_frags_2_guide_coords_rot_and_set)
                    # Now all are coming back true
                    # eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
                    #                         true_tup[2]]

                    # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment)
                    # guide coodinate pairs
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface "
                                       "Fragment guide coordinate pairs\n")

                    # Filter all overlapping arrays by matching ij type. This wouldn't increase speed much by
                    # putting before check_euler_table as the all to all is a hash operation
                    # for idx, ghost_frag_idx in enumerate(overlapping_ghost_frag_array):
                    #     # if idx < 30:
                    #     #     print(ghost_frag_idx, overlapping_surf_frag_array[idx])
                    #     #     print(ghost_frags[ghost_frag_idx].rmsd, surf_frag_list[overlapping_surf_frag_array[idx]].central_res_num)
                    #     #     print(ghost_frags[ghost_frag_idx].get_j_type(), surf_frag_list[overlapping_surf_frag_array[idx]].get_i_type())
                    #     #     print('\n\n')
                    #     if ghost_frags[ghost_frag_idx].get_j_type() == surf_frag_list[overlapping_surf_frag_array[idx]].get_i_type():
                    #         print(idx)
                    #     # else:
                    #     #     dummy = False
                    # ij_type_match = [True if ghost_frags[ghost_frag_idx].get_j_type() ==
                    #                  surf_frag_list[overlapping_surf_frag_array[idx]].get_i_type() else False
                    #                  for idx, ghost_frag_idx in enumerate(overlapping_ghost_frag_array)]
                    # print('ij_type_match: %s' % ij_type_match[:5])
                    if not any(overlapping_ghost_frag_array):
                        print('No overlapping ij fragments pairs, starting next sampling')
                        continue

                    # passing_ghost_indices = np.array([ghost_idx
                    #                                   for idx, ghost_idx in enumerate(overlapping_ghost_frag_array)
                    #                                   if ij_type_match[idx]])
                    # print('ghost indices:', passing_ghost_indices[:5])
                    # print('number of ghost indices considered: %d' % len(overlapping_ghost_frag_array))
                    # passing_ghost_coords = ghost_frag_guide_coords_rot_and_set[passing_ghost_indices]
                    passing_ghost_coords = ghost_frag_guide_coords_rot_and_set[overlapping_ghost_frag_array]
                    # print('ghost coords: %s' % passing_ghost_coords[:5])
                    # print('number of ghost coords considered: %d' % len(passing_ghost_coords))
                    # passing_surf_indices = np.array([surf_idx
                    #                                  for idx, surf_idx in enumerate(overlapping_surf_frag_array)
                    #                                  if ij_type_match[idx]])
                    # passing_surf_coords = surf_frags_2_guide_coords_rot_and_set[passing_surf_indices]
                    passing_surf_coords = surf_frags_2_guide_coords_rot_and_set[overlapping_surf_frag_array]
                    reference_rmsds = [max(ghost_frags[ghost_idx].get_rmsd(), 0.01)
                                       for ghost_idx in overlapping_ghost_frag_array]
                    #                    if ij_type_match[idx]]
                    optimal_shifts = [optimal_tx.solve_optimal_shift(passing_ghost_coords[idx],
                                                                     passing_surf_coords[idx],
                                                                     reference_rmsds[idx],
                                                                     max_z_value=init_max_z_val)
                                      for idx in range(len(passing_ghost_coords))]
                    # print('Number of optimal shifts: %d' % len(optimal_shifts))
                    # print('optimal shifts: %s' % optimal_shifts[:5])
                    passing_optimal_shifts = [passing_shift for passing_shift in optimal_shifts
                                              if passing_shift is not None]
                    print('Number of passing optimal shifts: %d' % len(passing_optimal_shifts))
                    print('passing optimal shifts: %s' % passing_optimal_shifts)
                    passing_optimal_shifts_idx = [idx for idx, passing_shift in enumerate(optimal_shifts)
                                                  if passing_shift is not None]
                    passing_fragment_pairs = [overlap_pairs[idx] for idx in passing_optimal_shifts_idx]
                    print('passing fragment pairs: %s' % passing_fragment_pairs)
                    central_res_tuples = [ghost_frags[ghost_idx].get_central_res_tup()
                                          for ghost_idx, surf_idx in passing_fragment_pairs]
                    print('passing ghost fragment residue/chain: %s' % central_res_tuples)

                    with open(log_file_path, "a+") as log_file:
                        log_file.write("%s Initial Interface Fragment Match%s Found\n\n"
                                       % (len(passing_optimal_shifts) if passing_optimal_shifts else 'No',
                                          'es' if len(passing_optimal_shifts) != 1 else ''))

                    degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
                    rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
                                                       (rot1_count, rot2_count))

                    find_docked_poses(sym_entry, ijk_frag_db, pdb1, pdb2, passing_optimal_shifts,
                                      complete_ghost_frag_np, complete_surf_frag_np, log_file_path,
                                      degen_subdir_out_path, rot_subdir_out_path, pdb1_path, pdb2_path, eul_lookup,
                                      rot1_mat, rot2_mat, max_z_val=subseq_max_z_val,
                                      output_exp_assembly=output_exp_assembly, output_uc=output_uc,
                                      output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
                rot2_count = 0
            degen2_count = 0
        rot1_count = 0

    with open(master_log_filepath, "a+") as master_log_file:
        master_log_file.write("COMPLETE ==> %s\n\n" % os.path.join(master_outdir, '%s_%s' % (pdb1_name, pdb2_name)))


if __name__ == '__main__':
    cmd_line_in_params = sys.argv
    if len(cmd_line_in_params) > 1:
        # Parsing Command Line Input
        sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir, output_exp_assembly, \
            output_uc, output_surrounding_uc, min_matched, timer, initial = \
            get_docking_parameters(cmd_line_in_params)

        # Master Log File
        master_log_filepath = os.path.join(master_outdir, master_log)

        if initial:
            # Making Master Output Directory
            if not os.path.exists(master_outdir):
                os.makedirs(master_outdir)
            # with open(master_log_filepath, "w") as master_logfile:
            #     master_logfile.write('Nanohedra\nMODE: DOCK\n\n')
        else:
            time.sleep(1)  # ensure that the first file was able to write before adding below log
            with open(master_log_filepath, "a+") as master_log_file:
                master_log_file.write("Docking %s / %s \n" % (os.path.basename(os.path.splitext(pdb1_path)[0]),
                                                              os.path.basename(os.path.splitext(pdb2_path)[0])))

        try:
            nanohedra(sym_entry_number, pdb1_path, pdb2_path, rot_step_deg1, rot_step_deg2, master_outdir,
                      output_exp_assembly, output_uc, output_surrounding_uc, min_matched,
                      keep_time=timer, main_log=initial)

        except KeyboardInterrupt:
            with open(master_log_filepath, "a+") as master_log:
                master_log.write("\nRun Ended By KeyboardInterrupt\n")
            sys.exit()
