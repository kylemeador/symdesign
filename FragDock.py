import time

import sklearn.neighbors

from SymDesignUtils import calculate_overlap, filter_euler_lookup_by_zvalue, DesignError
from classes.EulerLookup import EulerLookup
from classes.Fragment import *
from classes.OptimalTx import *
from classes.SymEntry import *
from classes.WeightedSeqFreq import FragMatchInfo, SeqFreqInfo
from utils.CmdLineArgParseUtils import *
# from utils.ExpandAssemblyUtils import generate_cryst1_record, expanded_design_is_clash
from utils.ExpandAssemblyUtils import *
from utils.PDBUtils import *
# from utils.SamplingUtils import get_degeneracy_matrices
from utils.SamplingUtils import *
from utils.SymmUtils import get_uc_dimensions


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

    if not ghost_frag and not matched_frag and not overlap_error and not match_number:
        raise DesignError('%s: Missing required information for writing!' % write_frag_match_info_file.__name__)

    with open(os.path.join(out_path, PUtils.frag_text_file), "a+") as out_info_file:
        if is_initial_match:
            out_info_file.write("DOCKED POSE ID: %s\n\n" % pose_id)
            out_info_file.write("***** INITIAL MATCH FROM REPRESENTATIVES OF INITIAL FRAGMENT CLUSTERS *****\n\n")

        out_info_file.write("MATCH %d\n" % match_number)
        out_info_file.write("z-val: %f\n" % overlap_error)
        out_info_file.write("CENTRAL RESIDUES\n")
        out_info_file.write("oligomer1 ch, resnum: %s, %d\n" % ghost_frag.get_aligned_surf_frag_central_res_tup())
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

    out_info_file_path = outdir_path + "/docked_pose_info_file.txt"
    out_info_file = open(out_info_file_path, "w")

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
    out_info_file.write("INTERNAL Tx PDB1: " + str(int_dof_tx_vec_1) + "\n")
    out_info_file.write("SETTING MATRIX PDB1: " + str(set_mat1) + "\n")
    if representative_ext_dof_tx_params_1 == [0, 0, 0]:
        ref_frame_tx_vec_1 = None
    else:
        ref_frame_tx_vec_1 = representative_ext_dof_tx_params_1
    out_info_file.write("REFERENCE FRAME Tx PDB1: " + str(ref_frame_tx_vec_1) + "\n\n")

    out_info_file.write("ROT/DEGEN MATRIX PDB2: %s\n" % str(rot_mat2))
    if representative_int_dof_tx_param_2 is not None:
        int_dof_tx_vec_2 = representative_int_dof_tx_param_2
    else:
        int_dof_tx_vec_2 = None
    out_info_file.write("INTERNAL Tx PDB2: " + str(int_dof_tx_vec_2) + "\n")
    out_info_file.write("SETTING MATRIX PDB2: " + str(set_mat2) + "\n")
    if representative_ext_dof_tx_params_2 == [0, 0, 0]:
        ref_frame_tx_vec_2 = None
    else:
        ref_frame_tx_vec_2 = representative_ext_dof_tx_params_2
    out_info_file.write("REFERENCE FRAME Tx PDB2: " + str(ref_frame_tx_vec_2) + "\n\n")

    out_info_file.write("CRYST1 RECORD: %s\n\n" % str(cryst1_record))

    out_info_file.write('Canonical Orientation PDB1 Path: %s\n' % pdb1_path)
    out_info_file.write('Canonical Orientation PDB2 Path: %s\n\n' % pdb2_path)

    out_info_file.close()


def out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2, optimal_tx_params,
        ghostfrag_surffrag_pair_list, complete_ghost_frag_list, complete_surf_frag_list, log_filepath,
        degen_subdir_out_path, rot_subdir_out_path, ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string,
        design_dim, pdb1_path, pdb2_path, expand_matrices, eul_lookup,
        rot_mat1=None, rot_mat2=None, max_z_val=2.0, output_exp_assembly=False, output_uc=False,
        output_surrounding_uc=False, clash_dist=2.2, min_matched=3):

    high_quality_match_value = 1
    for tx_idx in range(len(optimal_tx_params)):
        with open(log_filepath, "a+") as log_file:
            log_file.write("Optimal Shift %d\n" % (tx_idx + 1))

        # Dictionaries for PDB1 and PDB2 with (ch_id, res_num) tuples as keys for every residue that is covered by at
        # least 1 matched fragment. Dictionary values are lists containing 1 / (1 + z^2) values for every fragment match
        # that covers the (ch_id, res_num) residue.
        chid_resnum_scores_dict_pdb1 = {}
        chid_resnum_scores_dict_pdb2 = {}

        # Lists of unique (pdb1/2 chain id, pdb1/2 central residue number) tuples for pdb1/pdb2 interface mono fragments
        # that were matched to an i,j,k fragment in the database with a z value <= 1.
        # This is to keep track of and to count unique 'high quality' matches.
        # unique_interface_monofrags_infolist_highqual_pdb1 = []
        # unique_interface_monofrags_infolist_highqual_pdb2 = []

        # Number of unique interface mono fragments matched with a z value <= 1 ('high quality match'). This value has
        # to be >= min_matched (minimum number of high quality matches required)for a pose to be selected
        frag_match_info_list = []
        unique_interface_monofrags_infolist_pdb1 = []
        unique_interface_monofrags_infolist_pdb2 = []

        # Keep track of match information and residue pair frequencies for each fragment match this information will be
        # used to calculate a weighted frequency average for all central residues of matched fragments
        res_pair_freq_info_list = []

        # tx_parameters contains [OptimalExternalDOFShifts (n_dof_ext), OptimalInternalDOFShifts (n_dof_int)]
        tx_parameters = optimal_tx_params[tx_idx][0]
        # tx_parameters = optimal_tx_params[tx_idx]

        initial_overlap_z_val = optimal_tx_params[tx_idx][1]
        # initial_overlap_z_val = tx_parameters.get_zvalue()
        ghostfrag_surffrag_pair = ghostfrag_surffrag_pair_list[tx_idx]

        # Get Optimal External DOF shifts
        n_dof_external = len(get_ext_dof(ref_frame_tx_dof1, ref_frame_tx_dof2))  # returns 0 - 3
        optimal_ext_dof_shifts = None
        if n_dof_external > 0:
            optimal_ext_dof_shifts = tx_parameters[0:n_dof_external]

        copy_rot_tr_set_time_start = time.time()

        # Get Oligomer1 Optimal Internal Translation vector
        representative_int_dof_tx_param_1 = None
        if is_zshift1:
            representative_int_dof_tx_param_1 = [0, 0, tx_parameters[n_dof_external: n_dof_external + 1][0]]

        # Get Oligomer1 Optimal External Translation vector
        representative_ext_dof_tx_params_1 = None
        if optimal_ext_dof_shifts is not None:
            representative_ext_dof_tx_params_1 = get_optimal_external_tx_vector(ref_frame_tx_dof1,
                                                                                optimal_ext_dof_shifts)

        # Get Oligomer2 Optimal Internal Translation vector
        representative_int_dof_tx_param_2 = None
        if is_zshift2:
            representative_int_dof_tx_param_2 = [0, 0, tx_parameters[n_dof_external + 1: n_dof_external + 2][0]]

        # Get Oligomer2 Optimal External Translation vector
        representative_ext_dof_tx_params_2 = None
        if optimal_ext_dof_shifts is not None:
            representative_ext_dof_tx_params_2 = get_optimal_external_tx_vector(ref_frame_tx_dof2,
                                                                                optimal_ext_dof_shifts)

        # Get Unit Cell Dimensions for 2D and 3D SCMs
        # Restrict all reference frame translation parameters to > 0 for SCMs with reference frame translational d.o.f.
        ref_frame_var_is_pos = False
        uc_dimensions = None
        if optimal_ext_dof_shifts is not None:
            ref_frame_tx_dof_e = 0
            ref_frame_tx_dof_f = 0
            ref_frame_tx_dof_g = 0
            if len(optimal_ext_dof_shifts) == 1:
                ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                if ref_frame_tx_dof_e > 0:
                    ref_frame_var_is_pos = True
            if len(optimal_ext_dof_shifts) == 2:
                ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
                if ref_frame_tx_dof_e > 0 and ref_frame_tx_dof_f > 0:
                    ref_frame_var_is_pos = True
            if len(optimal_ext_dof_shifts) == 3:
                ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
                ref_frame_tx_dof_g = optimal_ext_dof_shifts[2]
                if ref_frame_tx_dof_e > 0 and ref_frame_tx_dof_f > 0 and ref_frame_tx_dof_g > 0:
                    ref_frame_var_is_pos = True

            uc_dimensions = get_uc_dimensions(uc_spec_string, ref_frame_tx_dof_e, ref_frame_tx_dof_f,
                                              ref_frame_tx_dof_g)

        if (optimal_ext_dof_shifts is not None and ref_frame_var_is_pos) or (optimal_ext_dof_shifts is None):

            # Rotate, Translate and Set PDB1
            pdb1_copy = rot_txint_set_txext_pdb(pdb1, rot_mat=rot_mat1, internal_tx_vec=representative_int_dof_tx_param_1,
                                                set_mat=set_mat1, ext_tx_vec=representative_ext_dof_tx_params_1)

            # Rotate, Translate and Set PDB2
            pdb2_copy = rot_txint_set_txext_pdb(pdb2, rot_mat=rot_mat2, internal_tx_vec=representative_int_dof_tx_param_2,
                                                set_mat=set_mat2, ext_tx_vec=representative_ext_dof_tx_params_2)

            copy_rot_tr_set_time_stop = time.time()
            copy_rot_tr_set_time = copy_rot_tr_set_time_stop - copy_rot_tr_set_time_start
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tCopy and Transform Oligomer1 and Oligomer2 (took: %s s)\n" % str(copy_rot_tr_set_time))

            # Check if PDB1 and PDB2 backbones clash
            oligomer1_oligomer2_clash_time_start = time.time()
            kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1_copy.extract_backbone_coords()))
            cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.extract_backbone_coords(),
                                                                             [clash_dist])
            oligomer1_oligomer2_clash_time_end = time.time()
            oligomer1_oligomer2_clash_time = oligomer1_oligomer2_clash_time_end - oligomer1_oligomer2_clash_time_start

            if cb_clash_count[0] == 0:
                with open(log_filepath, "a+") as log_file:
                    log_file.write("\tNO Backbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)\n"
                                   % str(oligomer1_oligomer2_clash_time))

                # Full Interface Fragment Match
                # TODO BIG This routine takes the bulk of the docking program due to the high amount of calculations.
                #  It could be reduced by only transforming each pdb instead of separating the guide atoms from the normal atoms
                #   This is already done for the ghost fragments... Low hanging fruit for surface frags, unless reason...
                #  The call to return guide atoms could be implemented on the returned transformed fragment atoms/coords
                #  In fact the transformed atoms are not needed at all in the output. Similar concept to PDB output except less useful in program operation.
                #   The index of the fragment (i,j,k) could be used with the set and rot matrices, followed by the internal and external tx
                #    Lots of memory overhead, this likely is what causes program termination so often in the optimal translation routines
                #  The use of hashing on the surface and ghost fragments could increase program runtime, over tuple calls
                #   to the ghost_fragment objects to return the aligned chain and residue then test for membership...
                #    Is the chain necessary? Probably. Two chains can occupy interface, even the same residue could be used
                #    Think D2 symmetry
                #   Store all the ghost/surface frags in a chain/residue dictionary?
                get_int_ghost_surf_frags_time_start = time.time()
                interface_ghostfrag_l, interface_monofrag2_l, interface_ghostfrag_guide_coords_l, \
                    interface_monofrag2_guide_coords_l, unique_interface_frag_count_pdb1, \
                    unique_interface_frag_count_pdb2 = \
                    get_interface_ghost_surf_frags(pdb1_copy, pdb2_copy, complete_ghost_frag_list,
                                                   complete_surf_frag_list, rot_mat1, rot_mat2,
                                                   representative_int_dof_tx_param_1, representative_int_dof_tx_param_2,
                                                   set_mat1, set_mat2, representative_ext_dof_tx_params_1,
                                                   representative_ext_dof_tx_params_2)
                get_int_ghost_surf_frags_time_end = time.time()
                get_int_ghost_surf_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start

                unique_total_interface_monofrags_count = unique_interface_frag_count_pdb1 + \
                                                         unique_interface_frag_count_pdb2

                if unique_total_interface_monofrags_count > 0:
                    with open(log_filepath, "a+") as log_file:
                        log_file.write("\tNewly Formed Interface Contains %s Unique Fragments on Oligomer 1 and %s on "
                                       "Oligomer 2\n" % (str(unique_interface_frag_count_pdb1),
                                                         str(unique_interface_frag_count_pdb2)))
                        log_file.write("\t(took: %s s to get interface surface and ghost fragments with their guide "
                                       "coordinates)\n" % str(get_int_ghost_surf_frags_time))

                    # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Mono Fragment) guide coodinate pairs
                    # in the same Euler rotational space bucket
                    eul_lookup_start_time = time.time()
                    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_l,
                                                                               interface_monofrag2_guide_coords_l)
                    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
                                            true_tup[2]]
                    eul_lookup_end_time = time.time()
                    eul_lookup_time = eul_lookup_end_time - eul_lookup_start_time

                    # Get RMSD and z-value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
                    # pair_count = 0
                    # total_overlap_count = 0
                    overlap_score_time_start = time.time()
                    all_fragment_overlap = filter_euler_lookup_by_zvalue(eul_lookup_true_list, interface_ghostfrag_l,
                                                                         interface_ghostfrag_guide_coords_l,
                                                                         interface_monofrag2_l,
                                                                         interface_monofrag2_guide_coords_l,
                                                                         z_value_func=calculate_overlap,
                                                                         max_z_value=max_z_val)
                    passing_fragment_overlap = list(filter(None, all_fragment_overlap))
                    ghostfrag_surffrag_pairs = [(interface_ghostfrag_l[eul_lookup_true_list[idx][0]],
                                                 interface_monofrag2_l[eul_lookup_true_list[idx][1]])
                                                for idx, boolean in enumerate(all_fragment_overlap) if boolean]

                    overlap_score_time_stop = time.time()
                    overlap_score_time = overlap_score_time_stop - overlap_score_time_start

                    with open(log_filepath, "a+") as log_file:
                        log_file.write("\t%d Fragment Match(es) Found in Complete Cluster Representative Fragment "
                                       "Library\n\t(Euler Lookup took %s s for %d fragment pairs and Overlap Score "
                                       "Calculation took %s for %d fragment pairs)\n" %
                                       (len(passing_fragment_overlap), str(eul_lookup_time),
                                        len(eul_lookup_all_to_all_list), str(overlap_score_time),
                                        len(eul_lookup_true_list)))

                    high_qual_match_count = sum([1 for match_score, z_value in passing_fragment_overlap
                                                 if z_value <= high_quality_match_value])

                    if high_qual_match_count >= min_matched:
                        for frag_idx, (interface_ghost_frag, interface_mono_frag) in enumerate(ghostfrag_surffrag_pairs):

                            ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
                            ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
                            ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
                            cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)

                            interface_ghost_frag_cluster_res_freq_list = \
                                ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
                                    ghost_frag_k_type].get_central_residue_pair_freqs()
                            pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num = \
                                interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
                            pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num = \
                                interface_mono_frag.get_central_res_tup()

                            covered_residues_pdb1 = [
                                (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num + j)
                                for j in range(-2, 3)]
                            covered_residues_pdb2 = [
                                (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num + j)
                                for j in range(-2, 3)]
                            score_term = passing_fragment_overlap[frag_idx][0]
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

                            if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) \
                                    not in unique_interface_monofrags_infolist_pdb1:
                                unique_interface_monofrags_infolist_pdb1.append(
                                    (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))

                            if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) \
                                    not in unique_interface_monofrags_infolist_pdb2:
                                unique_interface_monofrags_infolist_pdb2.append(
                                    (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                            z_val = passing_fragment_overlap[frag_idx][1]
                            frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val,
                                                         cluster_id, frag_idx + 1,
                                                         interface_ghost_frag_cluster_res_freq_list,
                                                         interface_ghost_frag.get_rmsd()))
                        unique_matched_interface_monofrag_count = len(unique_interface_monofrags_infolist_pdb1) + len(
                            unique_interface_monofrags_infolist_pdb2)
                        percent_of_interface_covered = unique_matched_interface_monofrag_count / float(
                            unique_total_interface_monofrags_count)

                    # for index_pair in eul_lookup_true_list:
                    #     interface_ghost_frag = interface_ghostfrag_l[index_pair[0]]
                    #     interface_ghost_frag_guide_coords = interface_ghostfrag_guide_coords_l[index_pair[0]]
                    #     ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
                    #     ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
                    #     ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
                    #     cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)
                    #     interface_ghost_frag_cluster_rmsd = interface_ghost_frag.get_rmsd()
                    #     interface_ghost_frag_cluster_res_freq_list = \
                    #         ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
                    #             ghost_frag_k_type].get_central_residue_pair_freqs()
                    #
                    #     interface_mono_frag_guide_coords = interface_monofrag2_guide_coords_l[index_pair[1]]
                    #     interface_mono_frag = interface_monofrag2_l[index_pair[1]]
                    #     interface_mono_frag_type = interface_mono_frag.get_i_type()
                    #
                    #     if (interface_mono_frag_type == ghost_frag_j_type) and (interface_ghost_frag_cluster_rmsd > 0):
                    #         # Calculate RMSD
                    #         total_overlap_count += 1
                    #         e1 = euclidean_squared_3d(interface_mono_frag_guide_coords[0],
                    #                                   interface_ghost_frag_guide_coords[0])
                    #         e2 = euclidean_squared_3d(interface_mono_frag_guide_coords[1],
                    #                                   interface_ghost_frag_guide_coords[1])
                    #         e3 = euclidean_squared_3d(interface_mono_frag_guide_coords[2],
                    #                                   interface_ghost_frag_guide_coords[2])
                    #         s = e1 + e2 + e3
                    #         mean = s / float(3)
                    #         rmsd = math.sqrt(mean)
                    #
                    #         # Calculate Guide Atom Overlap Z-Value
                    #         # and Calculate Score Term for Nanohedra Residue Level Summation Score
                    #         z_val = rmsd / float(interface_ghost_frag_cluster_rmsd)
                    #
                    #         if z_val <= max_z_val:
                    #
                    #             pair_count += 1
                    #
                    #             pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num = \
                    #                 interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
                    #             pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num = \
                    #                 interface_mono_frag.get_central_res_tup()
                    #
                    #             score_term = 1 / float(1 + (z_val ** 2))
                    #
                    #             covered_residues_pdb1 = [
                    #                 (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num + j)
                    #                 for j in range(-2, 3)]
                    #             covered_residues_pdb2 = [
                    #                 (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num + j)
                    #                 for j in range(-2, 3)]
                    #             for k in range(5):
                    #                 chid1, resnum1 = covered_residues_pdb1[k]
                    #                 chid2, resnum2 = covered_residues_pdb2[k]
                    #                 if (chid1, resnum1) not in chid_resnum_scores_dict_pdb1:
                    #                     chid_resnum_scores_dict_pdb1[(chid1, resnum1)] = [score_term]
                    #                 else:
                    #                     chid_resnum_scores_dict_pdb1[(chid1, resnum1)].append(score_term)
                    #
                    #                 if (chid2, resnum2) not in chid_resnum_scores_dict_pdb2:
                    #                     chid_resnum_scores_dict_pdb2[(chid2, resnum2)] = [score_term]
                    #                 else:
                    #                     chid_resnum_scores_dict_pdb2[(chid2, resnum2)].append(score_term)
                    #
                    #             # these next two blocks serve to calculate high_qual_match_count, then unique_matched_interface_monofrag_count
                    #             if z_val <= high_quality_match_value:
                    #                 if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) \
                    #                         not in unique_interface_monofrags_infolist_highqual_pdb1:
                    #                     unique_interface_monofrags_infolist_highqual_pdb1.append(
                    #                         (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))
                    #                 if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) \
                    #                         not in unique_interface_monofrags_infolist_highqual_pdb2:
                    #                     unique_interface_monofrags_infolist_highqual_pdb2.append(
                    #                         (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))
                    #
                    #             #######################################################
                    #
                    #             if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) \
                    #                     not in unique_interface_monofrags_infolist_pdb1:
                    #                 unique_interface_monofrags_infolist_pdb1.append(
                    #                     (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))
                    #
                    #             if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) \
                    #                     not in unique_interface_monofrags_infolist_pdb2:
                    #                 unique_interface_monofrags_infolist_pdb2.append(
                    #                     (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))
                    #
                    #             frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val,
                    #                                          cluster_id, pair_count,
                    #                                          interface_ghost_frag_cluster_res_freq_list,
                    #                                          interface_ghost_frag_cluster_rmsd))
                    #
                    # unique_matched_interface_monofrag_count = len(unique_interface_monofrags_infolist_pdb1) + len(
                    #     unique_interface_monofrags_infolist_pdb2)
                    # percent_of_interface_covered = unique_matched_interface_monofrag_count / float(
                    #     unique_total_interface_monofrags_count)
                    #
                    # overlap_score_time_stop = time.time()
                    # overlap_score_time = overlap_score_time_stop - overlap_score_time_start
                    #
                    # log_file = open(log_filepath, "a+")
                    # log_file.write("\t%s Fragment Match(es) Found in Complete Cluster "
                    #                "Representative Fragment Library\n" % str(pair_count))
                    # log_file.write("\t(Euler Lookup took %s s for %s fragment pairs and Overlap Score Calculation took"
                    #                " %s for %s fragment pairs)" %
                    #                (str(eul_lookup_time), str(len(eul_lookup_all_to_all_list)), str(overlap_score_time),
                    #                 str(total_overlap_count)) + "\n")
                    # log_file.close()
                    #
                    # high_qual_match_count = len(unique_interface_monofrags_infolist_highqual_pdb1) + len(
                    #     unique_interface_monofrags_infolist_highqual_pdb2)
                    # if high_qual_match_count >= min_matched:

                        # Get contacting PDB 1 ASU and PDB 2 ASU
                        asu_pdb_1, asu_pdb_2 = get_contacting_asu(pdb1_copy, pdb2_copy)

                        # Check if design has any clashes when expanded
                        tx_subdir_out_path = os.path.join(rot_subdir_out_path, "tx_%d" % (tx_idx + 1))
                        oligomers_subdir = rot_subdir_out_path.split(os.sep)[-3]
                        degen_subdir = rot_subdir_out_path.split(os.sep)[-2]
                        rot_subdir = rot_subdir_out_path.split(os.sep)[-1]
                        pose_id = "%s_%s_%s_TX_%d" % (oligomers_subdir, degen_subdir, rot_subdir, (tx_idx + 1))
                        sampling_id = '%s_%s_TX_%d' % (degen_subdir, rot_subdir, (tx_idx + 1))
                        if asu_pdb_1 is not None and asu_pdb_2 is not None:
                            exp_des_clash_time_start = time.time()
                            exp_des_is_clash = expanded_design_is_clash(asu_pdb_1, asu_pdb_2, design_dim,
                                                                        result_design_sym, expand_matrices,
                                                                        uc_dimensions, tx_subdir_out_path,
                                                                        output_exp_assembly, output_uc,
                                                                        output_surrounding_uc)
                            exp_des_clash_time_stop = time.time()
                            exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

                            if not exp_des_is_clash:

                                if not os.path.exists(degen_subdir_out_path):
                                    os.makedirs(degen_subdir_out_path)

                                if not os.path.exists(rot_subdir_out_path):
                                    os.makedirs(rot_subdir_out_path)

                                if not os.path.exists(tx_subdir_out_path):
                                    os.makedirs(tx_subdir_out_path)

                                with open(log_filepath, "a+") as log_file:
                                    log_file.write("\tNO Backbone Clash when Designed Assembly is Expanded (took: %s s "
                                                   "including writing)\n\tSUCCESSFUL DOCKED POSE: %s\n" %
                                                   (str(exp_des_clash_time), tx_subdir_out_path))

                                # Write PDB1 and PDB2 files
                                cryst1_record = None
                                if optimal_ext_dof_shifts is not None:
                                    cryst1_record = generate_cryst1_record(uc_dimensions, result_design_sym)
                                pdb1_fname = os.path.splitext(os.path.basename(pdb1.get_filepath()))[0]
                                pdb2_fname = os.path.splitext(os.path.basename(pdb2.get_filepath()))[0]
                                pdb1_copy.write(os.path.join(tx_subdir_out_path, '%s_%s.pdb' % (pdb1_fname, sampling_id)))
                                pdb2_copy.write(os.path.join(tx_subdir_out_path, '%s_%s.pdb' % (pdb2_fname, sampling_id)))

                                # Initial Interface Fragment Match
                                # Rotate, translate and set initial match interface fragment
                                init_match_ghost_frag = ghostfrag_surffrag_pair[0]
                                init_match_ghost_frag_pdb = init_match_ghost_frag.get_pdb()
                                init_match_ghost_frag_pdb_copy = rot_txint_set_txext_pdb(init_match_ghost_frag_pdb,
                                                                                         rot_mat=rot_mat1,
                                                                                         internal_tx_vec=representative_int_dof_tx_param_1,
                                                                                         set_mat=set_mat1,
                                                                                         ext_tx_vec=representative_ext_dof_tx_params_1)

                                # Make directories to output matched fragment PDB files
                                # initial_match for the initial matched fragment
                                # high_qual_match for fragments that were matched with z values <= 1
                                # low_qual_match for fragments that were matched with z values > 1
                                matched_frag_reps_outdir_path = os.path.join(tx_subdir_out_path, "matching_fragments")
                                if not os.path.exists(matched_frag_reps_outdir_path):
                                    os.makedirs(matched_frag_reps_outdir_path)

                                init_match_outdir_path = os.path.join(matched_frag_reps_outdir_path,
                                                                      "initial_match")
                                if not os.path.exists(init_match_outdir_path):
                                    os.makedirs(init_match_outdir_path)

                                high_qual_matches_outdir_path = os.path.join(matched_frag_reps_outdir_path,
                                                                             'high_qual_match')
                                if not os.path.exists(high_qual_matches_outdir_path):
                                    os.makedirs(high_qual_matches_outdir_path)

                                low_qual_matches_outdir_path = os.path.join(matched_frag_reps_outdir_path,
                                                                            "low_qual_match")
                                if not os.path.exists(low_qual_matches_outdir_path):
                                    os.makedirs(low_qual_matches_outdir_path)

                                # Write out initial match interface fragment
                                match_number = 0
                                initial_ghost_frag_i = init_match_ghost_frag.get_i_frag_type()
                                initial_ghost_frag_j = init_match_ghost_frag.get_j_frag_type()
                                initial_ghost_frag_k = init_match_ghost_frag.get_k_frag_type()
                                # initial_ghost_frag_cluster_res_freqs = frag_db.info(init_match_ghost_frag.get_ijk()).get_central_residue_pair_freqs()  # TOSO
                                initial_ghost_frag_cluster_res_freqs = \
                                    ijk_intfrag_cluster_info_dict[initial_ghost_frag_i][initial_ghost_frag_j][
                                        initial_ghost_frag_k].get_central_residue_pair_freqs()
                                init_match_cluster_id = "i%s_j%s_k%s" % (initial_ghost_frag_i, initial_ghost_frag_j,
                                                                         initial_ghost_frag_k)
                                # if write_frags:
                                init_match_ghost_frag_pdb_copy.write(
                                    os.path.join(init_match_outdir_path, 'int_frag_%s_%d.pdb'
                                                 % ('i%s_j%s_k%s' % init_match_ghost_frag.get_ijk(), match_number)))
                                init_match_ghost_frag_cluster_rmsd = init_match_ghost_frag.get_rmsd()
                                init_match_surf_frag = ghostfrag_surffrag_pair[1]
                                write_frag_match_info_file(ghost_frag=init_match_ghost_frag,
                                                           matched_frag=init_match_surf_frag,
                                                           overlap_error=initial_overlap_z_val,
                                                           match_number=match_number,
                                                           central_frequencies=initial_ghost_frag_cluster_res_freqs,
                                                           out_path=matched_frag_reps_outdir_path,
                                                           pose_id=pose_id, is_initial_match=True)

                                # For all matched interface fragments
                                # write out aligned cluster representative fragment
                                # write out associated match information to frag_match_info_file.txt
                                # calculate weighted frequency for central residues
                                # write out weighted frequencies to frag_match_info_file.txt
                                for matched_frag in frag_match_info_list:
                                    match_number += 1
                                    interface_ghost_frag = matched_frag[0]
                                    ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
                                    ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
                                    ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
                                    if matched_frag[2] <= 1:  # if the overlap z-value is less than 1
                                        matched_frag_outdir_path = high_qual_matches_outdir_path
                                    else:
                                        matched_frag_outdir_path = low_qual_matches_outdir_path
                                    # if write_frags:
                                    interface_ghost_frag.get_pdb().write(
                                        os.path.join(matched_frag_outdir_path, 'int_frag_%s_%d.pdb'
                                                     % (matched_frag[3], matched_frag[4])))
                                    write_frag_match_info_file(ghost_frag=matched_frag[0], matched_frag=matched_frag[1],
                                                               overlap_error=matched_frag[2],
                                                               match_number=matched_frag[4],
                                                               central_frequencies=matched_frag[5],
                                                               out_path=matched_frag_reps_outdir_path, pose_id=pose_id)

                                    match_res_pair_freq_list = matched_frag[5]
                                    match_cnt_chid1, match_cnt_resnum1 = matched_frag[
                                        0].get_aligned_surf_frag_central_res_tup()
                                    match_cnt_chid2, match_cnt_resnum2 = matched_frag[1].get_central_res_tup()
                                    match_z_val = matched_frag[2]
                                    match_res_pair_freq_info = FragMatchInfo(match_res_pair_freq_list,
                                                                             match_cnt_chid1,
                                                                             match_cnt_resnum1,
                                                                             match_cnt_chid2,
                                                                             match_cnt_resnum2,
                                                                             match_z_val)
                                    res_pair_freq_info_list.append(match_res_pair_freq_info)

                                weighted_seq_freq_info = SeqFreqInfo(res_pair_freq_info_list)
                                weighted_seq_freq_info.write(os.path.join(matched_frag_reps_outdir_path, PUtils.frag_text_file))

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
                                                       unique_matched_interface_monofrag_count,
                                                       unique_total_interface_monofrags_count,
                                                       percent_of_interface_covered, rot_mat1,
                                                       representative_int_dof_tx_param_1, set_mat1,
                                                       representative_ext_dof_tx_params_1, rot_mat2,
                                                       representative_int_dof_tx_param_2, set_mat2,
                                                       representative_ext_dof_tx_params_2, cryst1_record, pdb1_path,
                                                       pdb2_path, pose_id)

                            else:
                                with open(log_filepath, "a+") as log_file:
                                    log_file.write("\tBackbone Clash when Designed Assembly is Expanded (took: %s s)\n"
                                                   % str(exp_des_clash_time))
                        else:
                            with open(log_filepath, "a+") as log_file:
                                log_file.write("\tNO Design ASU Found\n")
                    else:
                        with open(log_filepath, "a+") as log_file:
                            log_file.write("\t%s < %s Which is Set as the Minimal Required Amount of High Quality "
                                           "Fragment Matches\n" % (str(high_qual_match_count), str(min_matched)))
                else:
                    with open(log_filepath, "a+") as log_file:
                        log_file.write("\tNO Interface Mono Fragments Found\n")
            else:
                with open(log_filepath, "a+") as log_file:
                    log_file.write("\tBackbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)\n"
                                   % str(oligomer1_oligomer2_clash_time))
        else:
            efg_tx_params_str = [str(None), str(None), str(None)]
            for param_index in range(len(optimal_ext_dof_shifts)):
                efg_tx_params_str[param_index] = str(optimal_ext_dof_shifts[param_index])
            with open(log_filepath, "a+") as log_file:
                log_file.write("\tReference Frame Shift Parameter(s) is/are Negative: e: %s, f: %s, g: %s\n\n"
                               % (efg_tx_params_str[0], efg_tx_params_str[1], efg_tx_params_str[2]))


# KM TODO ijk_intfrag_cluster_info_dict contains all info in init_intfrag_cluster_info_dict. init info could be deleted,
#     This doesn't take up much extra memory, but makes future maintanence bad, for porting frags to fragDB say...
def nanohedra(sym_entry_number, pdb1_path, pdb2_path, rot_step_deg_pdb1, rot_step_deg_pdb2, master_outdir,
              output_exp_assembly, output_uc, output_surrounding_uc, min_matched, init_match_type, keep_time=True,
              main_log=False):

    # Fragment Database Directory Paths
    # frag_db = PUtils.frag_directory['biological_interfaces']  # Todo make dynamic at startup or use all fragDB

    # SymEntry Parameters
    sym_entry = SymEntry(sym_entry_number)

    oligomer_symmetry_1 = sym_entry.get_group1_sym()
    oligomer_symmetry_2 = sym_entry.get_group2_sym()
    design_symmetry_pg = sym_entry.get_pt_grp_sym()

    rot_range_deg_pdb1 = sym_entry.get_rot_range_deg_1()
    rot_range_deg_pdb2 = sym_entry.get_rot_range_deg_2()

    set_mat1 = sym_entry.get_rot_set_mat_group1()
    set_mat2 = sym_entry.get_rot_set_mat_group2()

    is_zshift1 = sym_entry.is_internal_tx1()
    is_zshift2 = sym_entry.is_internal_tx2()

    is_internal_rot1 = sym_entry.is_internal_rot1()
    is_internal_rot2 = sym_entry.is_internal_rot2()

    design_dim = sym_entry.get_design_dim()

    ref_frame_tx_dof1 = sym_entry.get_ref_frame_tx_dof_group1()
    ref_frame_tx_dof2 = sym_entry.get_ref_frame_tx_dof_group2()

    result_design_sym = sym_entry.get_result_design_sym()
    uc_spec_string = sym_entry.get_uc_spec_string()

    # Default Fragment Guide Atom Overlap Z-Value Threshold For Initial Matches
    init_max_z_val = 1.0

    # Default Fragment Guide Atom Overlap Z-Value Threshold For All Subsequent Matches
    subseq_max_z_val = 2.0

    degeneracy_matrices_1 = get_degeneracy_matrices(oligomer_symmetry_1, design_symmetry_pg)
    degeneracy_matrices_2 = get_degeneracy_matrices(oligomer_symmetry_2, design_symmetry_pg)

    if main_log:
        with open(master_log_filepath, "a+") as master_log_file:
            # Default Rotation Step
            if is_internal_rot1 and rot_step_deg_pdb1 is None:
                rot_step_deg_pdb1 = 3  # If rotation step not provided but required, set rotation step to default
            if is_internal_rot2 and rot_step_deg_pdb2 is None:
                rot_step_deg_pdb2 = 3  # If rotation step not provided but required, set rotation step to default

            if not is_internal_rot1 and rot_step_deg_pdb1 is not None:
                rot_step_deg_pdb1 = 1
                master_log_file.write("Warning: Rotation Step 1 Specified Was Ignored. Oligomer 1 Does Not Have"
                                      " Internal Rotational DOF\n\n")
            if not is_internal_rot2 and rot_step_deg_pdb2 is not None:
                rot_step_deg_pdb2 = 1
                master_log_file.write("Warning: Rotation Step 2 Specified Was Ignored. Oligomer 2 Does Not Have"
                                      " Internal Rotational DOF\n\n")

            if not is_internal_rot1 and rot_step_deg_pdb1 is None:
                rot_step_deg_pdb1 = 1
            if not is_internal_rot2 and rot_step_deg_pdb2 is None:
                rot_step_deg_pdb2 = 1

            master_log_file.write("NANOHEDRA PROJECT INFORMATION\n")
            master_log_file.write("Oligomer 1 Input Directory: %s\n" % pdb1_path)
            master_log_file.write("Oligomer 2 Input Directory: %s\n" % pdb2_path)
            master_log_file.write("Master Output Directory: %s\n\n" % master_outdir)

            master_log_file.write("SYMMETRY COMBINATION MATERIAL INFORMATION\n")
            master_log_file.write("Nanohedra Entry Number: %s\n" % str(sym_entry_number))
            master_log_file.write("Oligomer 1 Point Group Symmetry: %s\n" % oligomer_symmetry_1)
            master_log_file.write("Oligomer 2 Point Group Symmetry: %s\n" % oligomer_symmetry_1)
            master_log_file.write("SCM Point Group Symmetry: %s\n" % design_symmetry_pg)

            master_log_file.write("Oligomer 1 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot1()))
            master_log_file.write("Oligomer 2 Internal ROT DOF: %s\n" % str(sym_entry.get_internal_rot2()))
            master_log_file.write("Oligomer 1 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx1()))
            master_log_file.write("Oligomer 2 Internal Tx DOF: %s\n" % str(sym_entry.get_internal_tx2()))
            master_log_file.write("Oligomer 1 Setting Matrix: %s\n" % str(set_mat1))
            master_log_file.write("Oligomer 2 Setting Matrix: %s\n" % str(set_mat2))
            master_log_file.write("Oligomer 1 Reference Frame Tx DOF: %s\n" % str(
                ref_frame_tx_dof1) if sym_entry.is_ref_frame_tx_dof1() else str(None))
            master_log_file.write("Oligomer 2 Reference Frame Tx DOF: %s\n" % str(
                ref_frame_tx_dof2) if sym_entry.is_ref_frame_tx_dof2() else str(None))
            master_log_file.write("Resulting SCM Symmetry: %s\n" % result_design_sym)
            master_log_file.write("SCM Dimension: %s\n" % str(design_dim))
            master_log_file.write("SCM Unit Cell Specification: %s\n\n" % uc_spec_string)

            master_log_file.write("ROTATIONAL SAMPLING INFORMATION\n")
            master_log_file.write(
                "Oligomer 1 ROT Sampling Range: %s\n" % str(rot_range_deg_pdb1) if is_internal_rot1 else str(None))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Range: %s\n" % str(rot_range_deg_pdb2) if is_internal_rot2 else str(None))
            master_log_file.write(
                "Oligomer 1 ROT Sampling Step: %s\n" % (str(rot_step_deg1) if is_internal_rot1 else str(None)))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Step: %s\n\n" % (str(rot_step_deg_pdb2) if is_internal_rot2 else str(None)))

            master_log_file.write("ROTATIONAL SAMPLING INFORMATION\n")
            master_log_file.write(
                "Oligomer 1 ROT Sampling Range: %s\n" % str(rot_range_deg_pdb1) if is_internal_rot1 else str(None))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Range: %s\n" % str(rot_range_deg_pdb2) if is_internal_rot2 else str(None))
            master_log_file.write(
                "Oligomer 1 ROT Sampling Step: %s\n" % (str(rot_step_deg1) if is_internal_rot1 else str(None)))
            master_log_file.write(
                "Oligomer 2 ROT Sampling Step: %s\n\n" % (str(rot_step_deg_pdb2) if is_internal_rot2 else str(None)))

            # Get Degeneracy Matrices
            master_log_file.write("Searching For Possible Degeneracies\n")
            if degeneracy_matrices_1 is None:
                master_log_file.write("No Degeneracies Found for Oligomer 1\n")
            elif len(degeneracy_matrices_1) == 1:
                master_log_file.write("1 Degeneracy Found for Oligomer 1\n")
            else:
                master_log_file.write("%d Degeneracies Found for Oligomer 1\n" % len(degeneracy_matrices_1))

            if degeneracy_matrices_2 is None:
                master_log_file.write("No Degeneracies Found for Oligomer 2\n\n")
            elif len(degeneracy_matrices_2) == 1:
                master_log_file.write("1 Degeneracy Found for Oligomer 2\n\n")
            else:
                master_log_file.write("%s Degeneracies Found for Oligomer 2\n\n" % str(len(degeneracy_matrices_2)))

            # Get Initial Fragment Database
            master_log_file.write("Retrieving Database of Complete Interface Fragment Cluster Representatives\n")
            if init_match_type == "1_2":
                master_log_file.write("Retrieving Database of Helix-Strand Interface Fragment Cluster "
                                      "Representatives\n\n")
            elif init_match_type == "2_1":
                master_log_file.write("Retrieving Database of Strand-Helix Interface Fragment Cluster "
                                      "Representatives\n\n")
            elif init_match_type == "2_2":
                master_log_file.write("Retrieving Database of Strand-Strand Interface Fragment Cluster "
                                      "Representatives\n\n")
            else:
                master_log_file.write("Retrieving Database of Helix-Helix Interface Fragment Cluster "
                                      "Representatives\n\n")

    # Create fragment database for all ijk cluster representatives
    ijk_frag_db = FragmentDB()
    #                       monofrag_cluster_rep_dirpath, ijk_intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath)

    # Get complete IJK fragment representatives database dictionaries
    ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
    ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
    ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()

    init_match_mapped = init_match_type.split('_')[0]
    init_match_paired = init_match_type.split('_')[1]
    # 1_1 Get Helix-Helix fragment representatives database dict for initial interface fragment matching
    # 1_2 Get Helix-Strand fragment representatives database dict for initial interface fragment matching
    # 2_1 Get Strand-Helix fragment representatives database dict for initial interface fragment matching
    # 2_2 Get Strand-Strand fragment representatives database dict for initial interface fragment matching

    init_monofrag_cluster_rep_pdb_dict_1 = {init_match_mapped: ijk_monofrag_cluster_rep_pdb_dict[init_match_mapped]}
    init_monofrag_cluster_rep_pdb_dict_2 = {init_match_paired: ijk_monofrag_cluster_rep_pdb_dict[init_match_paired]}
    init_intfrag_cluster_rep_dict = \
        {init_match_mapped: {init_match_paired: ijk_intfrag_cluster_rep_dict[init_match_mapped][init_match_paired]}}
    init_intfrag_cluster_info_dict = \
        {init_match_mapped: {init_match_paired: ijk_intfrag_cluster_info_dict[init_match_mapped][init_match_paired]}}

    # Initialize Euler Lookup Class
    eul_lookup = EulerLookup()

    # Get Design Expansion Matrices
    if design_dim == 2 or design_dim == 3:
        expand_matrices = get_sg_sym_op(result_design_sym)
    else:
        expand_matrices = get_ptgrp_sym_op(result_design_sym)

    with open(master_log_filepath, "a+") as master_log_file:
        master_log_file.write("Docking %s / %s \n" % (os.path.basename(os.path.splitext(pdb1_path)[0]),
                                                      os.path.basename(os.path.splitext(pdb2_path)[0])))

    nanohedra_dock(init_intfrag_cluster_rep_dict, ijk_intfrag_cluster_rep_dict, init_monofrag_cluster_rep_pdb_dict_1,
                   init_monofrag_cluster_rep_pdb_dict_2, init_intfrag_cluster_info_dict,
                   ijk_monofrag_cluster_rep_pdb_dict, ijk_intfrag_cluster_info_dict, master_outdir, pdb1_path,
                   pdb2_path, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2,
                   result_design_sym, uc_spec_string, design_dim, expand_matrices, eul_lookup, init_max_z_val,
                   subseq_max_z_val, degeneracy_matrices_1=degeneracy_matrices_1,
                   degeneracy_matrices_2=degeneracy_matrices_2, rot_step_deg_pdb1=rot_step_deg_pdb1,
                   rot_range_deg_pdb1=rot_range_deg_pdb1, rot_step_deg_pdb2=rot_step_deg_pdb2,
                   rot_range_deg_pdb2=rot_range_deg_pdb2, output_exp_assembly=output_exp_assembly,
                   output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched,
                   keep_time=keep_time)


def nanohedra_dock(init_intfrag_cluster_rep_dict, ijk_intfrag_cluster_rep_dict, init_monofrag_cluster_rep_pdb_dict_1,
                   init_monofrag_cluster_rep_pdb_dict_2, init_intfrag_cluster_info_dict,
                   ijk_monofrag_cluster_rep_pdb_dict, ijk_intfrag_cluster_info_dict, master_outdir, pdb1_path,
                   pdb2_path, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2,
                   result_design_sym, uc_spec_string, design_dim, expand_matrices, eul_lookup, init_max_z_val,
                   subseq_max_z_val, degeneracy_matrices_1=None, degeneracy_matrices_2=None, rot_step_deg_pdb1=1,
                   rot_range_deg_pdb1=0, rot_step_deg_pdb2=1, rot_range_deg_pdb2=0, output_exp_assembly=False,
                   output_uc=False, output_surrounding_uc=False, min_matched=3, keep_time=True):

    # Output Directory
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
        with open(log_file_path, "a+") as log_file:
            log_file.write("DOCKING %s TO %s\nOligomer 1 Path: %s\nOligomer 2 Path: %s\nOutput Directory: %s\n\n"
                           % (pdb1_name, pdb2_name, pdb1_path, pdb2_path, outdir))

    # Get PDB1 Symmetric Building Block
    pdb1 = PDB()
    pdb1.readfile(pdb1_path)

    # Get Oligomer 1 Ghost Fragments With Guide Coordinates Using Initial Match Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting %s Oligomer 1 Ghost Fragments Using INITIAL Fragment Database" % pdb1_name)
        if keep_time:
            get_init_ghost_frags_time_start = time.time()

    kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1.extract_backbone_coords()))
    surf_frags_1 = get_surface_fragments(pdb1)
    ghost_frag_list = []
    ghost_frag_guide_coords_list = []
    for frag1 in surf_frags_1:
        monofrag1 = MonoFragment(frag1, init_monofrag_cluster_rep_pdb_dict_1)
        monofrag_ghostfrag_list = monofrag1.get_ghost_fragments(init_intfrag_cluster_rep_dict,
                                                                kdtree_oligomer1_backbone,
                                                                init_intfrag_cluster_info_dict)
        if monofrag_ghostfrag_list is not None:
            ghost_frag_list.extend(monofrag_ghostfrag_list)
            ghost_frag_guide_coords_list.extend(list(map(GhostFragment.get_guide_coords, monofrag_ghostfrag_list)))

    if not resume and keep_time:
        get_init_ghost_frags_time_stop = time.time()
        get_init_ghost_frags_time = get_init_ghost_frags_time_stop - get_init_ghost_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n" % str(get_init_ghost_frags_time))

    # Get Oligomer1 Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting %s Oligomer 1 Ghost Fragments Using COMPLETE Fragment Database" % pdb1_name)
        if keep_time:
            get_complete_ghost_frags_time_start = time.time()

    # KM this does a double calculation and storage by saving the initial fragments again. All caluclations with this group are doing more work than necessary
    # one could imagine doing this first, then using a for loop to test for the indices that are the initial fragment search type
    complete_ghost_frag_list = []
    for frag1 in surf_frags_1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone, ijk_intfrag_cluster_info_dict)
        if complete_monofrag1_ghostfrag_list:
            complete_ghost_frag_list.extend(complete_monofrag1_ghostfrag_list)
            # for complete_ghostfrag in complete_monofrag1_ghostfrag_list:
            #     complete_ghost_frag_list.append(complete_ghostfrag)
    if not resume and keep_time:
        get_complete_ghost_frags_time_stop = time.time()
        get_complete_ghost_frags_time = get_complete_ghost_frags_time_stop - get_complete_ghost_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n" % str(get_complete_ghost_frags_time))

    # Get PDB2 Symmetric Building Block
    pdb2 = PDB()
    pdb2.readfile(pdb2_path)

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using Initial Match Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting Oligomer 2 Surface Fragments Using INITIAL Fragment Database")
        if keep_time:
            get_init_surf_frags_time_start = time.time()
    surf_frags_2 = get_surface_fragments(pdb2)
    surf_frag_list = []
    surf_frags_oligomer_2_guide_coords_list = []
    for frag2 in surf_frags_2:
        monofrag2 = MonoFragment(frag2, init_monofrag_cluster_rep_pdb_dict_2)
        monofrag2_guide_coords = monofrag2.get_guide_coords()
        if monofrag2_guide_coords is not None:
            surf_frag_list.append(monofrag2)
            surf_frags_oligomer_2_guide_coords_list.append(monofrag2_guide_coords)
    if not resume and keep_time:
        get_init_surf_frags_time_stop = time.time()
        get_init_surf_frags_time = get_init_surf_frags_time_stop - get_init_surf_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n" % str(get_init_surf_frags_time))

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        with open(log_file_path, "a+") as log_file:
            log_file.write("Getting Oligomer 2 Surface Fragments Using COMPLETE Fragment Database")
        if keep_time:
            get_complete_surf_frags_time_start = time.time()
    complete_surf_frag_list = []
    for frag2 in surf_frags_2:
        complete_monofrag2 = MonoFragment(frag2,
                                          ijk_monofrag_cluster_rep_pdb_dict)  # KM this does a double calculation and storage by saving the initial fragments again. All caluclations with this group are doing more work than necessary
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords is not None:
            complete_surf_frag_list.append(complete_monofrag2)

    # After this, the entire fragment database is unnecessary. Dereferenceing for memory
    del ijk_monofrag_cluster_rep_pdb_dict, init_monofrag_cluster_rep_pdb_dict_1, init_monofrag_cluster_rep_pdb_dict_2

    if not resume and keep_time:
        get_complete_surf_frags_time_stop = time.time()
        get_complete_surf_frags_time = get_complete_surf_frags_time_stop - get_complete_surf_frags_time_start
        with open(log_file_path, "a+") as log_file:
            log_file.write(" (took: %s s)\n\n" % str(get_complete_surf_frags_time))

    # Oligomer 1 Has Interior Rotational Degree of Freedom True or False
    has_int_rot_dof_1 = False
    if rot_range_deg_pdb1 != 0:
        has_int_rot_dof_1 = True

    # Oligomer 2 Has Interior Rotational Degree of Freedom True or False
    has_int_rot_dof_2 = False
    if rot_range_deg_pdb2 != 0:
        has_int_rot_dof_2 = True

    # Obtain Reference Frame Translation Info
    parsed_ref_frame_tx_dof1 = parse_ref_tx_dof_str_to_list(ref_frame_tx_dof1)
    parsed_ref_frame_tx_dof2 = parse_ref_tx_dof_str_to_list(ref_frame_tx_dof2)

    if parsed_ref_frame_tx_dof1 == ['0', '0', '0'] and parsed_ref_frame_tx_dof2 == ['0', '0', '0']:
        dof_ext = np.empty((0, 3), float)
    else:
        dof_ext = get_ext_dof(ref_frame_tx_dof1, ref_frame_tx_dof2)

    # Transpose Setting Matrices to Set Guide Coordinates just for Euler Lookup Using np.matmul
    set_mat1_np_t = np.transpose(set_mat1)
    set_mat2_np_t = np.transpose(set_mat2)

    degen1_count, degen2_count, rot1_count, rot2_count = 0, 0, 0, 0
    if resume:
        degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log_file_path)
        with open(log_file_path, "a+") as log_file:
            log_file.write('Job was run with the \'-resume\' flag. Resuming from last sampled rotational space!\n')

    if (degeneracy_matrices_1 is None and has_int_rot_dof_1 is False) and (
            degeneracy_matrices_2 is None and has_int_rot_dof_2 is False):
        rot1_mat = None
        rot2_mat = None
        if not resume:
            with open(log_file_path, "a+") as log_file:
                # No Degeneracies/Rotation Matrices to get for Oligomer 1 or Oligomer2
                log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1\n")
                log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n\n")

        with open(log_file_path, "a+") as log_file:
            log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****"
                           % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")
            # Get (Oligomer1 Ghost Fragment, Oligomer2 Surface Fragment) guide coodinate pairs in the same Euler
            # rotational space bucket
            log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational "
                           "space bucket\n")

        ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords_list, set_mat1_np_t)
        surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags_oligomer_2_guide_coords_list, set_mat2_np_t)

        eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_set_for_eul,
                                                                   surf_frags_2_guide_coords_list_set_for_eul)
        eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

        # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
        with open(log_file_path, "a+") as log_file:
            log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide "
                           "coordinate pairs\n")

        optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
        all_optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frag_list,
                                                           ghost_frag_guide_coords_list,
                                                           surf_frag_list,
                                                           surf_frags_oligomer_2_guide_coords_list,
                                                           z_value_func=optimal_tx.apply,
                                                           max_z_value=init_max_z_val)

        passing_optimal_shifts = list(filter(None, all_optimal_shifts))
        ghostfrag_surffrag_pairs = [(ghost_frag_list[eul_lookup_true_list[idx][0]],
                                     surf_frag_list[eul_lookup_true_list[idx][1]])
                                    for idx, boolean in enumerate(all_optimal_shifts) if boolean]

        if len(passing_optimal_shifts) == 0:
            with open(log_file_path, "a+") as log_file:
                log_file.write("No Initial Interface Fragment Matches Found\n\n")
        else:
            with open(log_file_path, "a+") as log_file:
                log_file.write("%d Initial Interface Fragment Match(es) Found\n"
                               % len(passing_optimal_shifts))

        degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
        rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
                                           (rot1_count, rot2_count))

        out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1,
            is_zshift2, passing_optimal_shifts, ghostfrag_surffrag_pairs, complete_ghost_frag_list,
            complete_surf_frag_list, log_file_path, degen_subdir_out_path, rot_subdir_out_path,
            ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string, design_dim,
            pdb1_path, pdb2_path, expand_matrices, eul_lookup,
            rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
            output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)

    elif (degeneracy_matrices_1 is not None or has_int_rot_dof_1 is True) and (
            degeneracy_matrices_2 is None and has_int_rot_dof_2 is False):
        # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1\n")
        rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", rot_range_deg_pdb1)
        degen_rot_mat_1 = get_degen_rotmatrices(degeneracy_matrices_1, rotation_matrices_1)

        # No Degeneracies/Rotation Matrices to get for Oligomer2
        rot2_mat = None
        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n\n")
        surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags_oligomer_2_guide_coords_list, set_mat2_np_t)

        optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
        for degen1 in degen_rot_mat_1[degen1_count:]:
            degen1_count += 1
            for rot1_mat in degen1[rot1_count:]:
                rot1_count += 1
                # Rotate Oligomer1 Ghost Fragment Guide Coodinates using rot1_mat
                rot1_mat_np_t = np.transpose(rot1_mat)
                ghost_frag_guide_coords_list_rot_np = np.matmul(ghost_frag_guide_coords_list, rot1_mat_np_t)
                ghost_frag_guide_coords_list_rot = ghost_frag_guide_coords_list_rot_np.tolist()

                with open(log_file_path, "a+") as log_file:
                    log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | "
                                   "OLIGOMER 2: Degeneracy %s Rotation %s *****\n"
                                   % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)))

                # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 Surface Fragment)
                # guide coodinate pairs in the same Euler rotational space bucket
                with open(log_file_path, "a+") as log_file:
                    log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler "
                                   "rotational space bucket\n")

                ghost_frag_guide_coords_list_rot_and_set_for_eul = np.matmul(ghost_frag_guide_coords_list_rot,
                                                                             set_mat1_np_t)

                eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(
                    ghost_frag_guide_coords_list_rot_and_set_for_eul, surf_frags_2_guide_coords_list_set_for_eul)
                eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
                                        true_tup[2]]

                # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
                with open(log_file_path, "a+") as log_file:
                    log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide"
                                   "coordinate pairs\n")

                all_optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frag_list,
                                                                   ghost_frag_guide_coords_list_rot,
                                                                   surf_frag_list,
                                                                   surf_frags_oligomer_2_guide_coords_list,
                                                                   z_value_func=optimal_tx.apply,
                                                                   max_z_value=init_max_z_val)

                passing_optimal_shifts = list(filter(None, all_optimal_shifts))
                ghostfrag_surffrag_pairs = [(ghost_frag_list[eul_lookup_true_list[idx][0]],
                                             surf_frag_list[eul_lookup_true_list[idx][1]])
                                            for idx, boolean in enumerate(all_optimal_shifts) if boolean]

                if len(passing_optimal_shifts) == 0:
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("No Initial Interface Fragment Matches Found\n\n")
                else:
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("%d Initial Interface Fragment Match(es) Found\n"
                                       % len(passing_optimal_shifts))

                degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
                rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
                                                   (rot1_count, rot2_count))

                out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1,
                    is_zshift2, passing_optimal_shifts, ghostfrag_surffrag_pairs, complete_ghost_frag_list,
                    complete_surf_frag_list, log_file_path, degen_subdir_out_path, rot_subdir_out_path,
                    ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string, design_dim,
                    pdb1_path, pdb2_path, expand_matrices, eul_lookup,
                    rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
                    output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
            rot1_count = 0

    elif (degeneracy_matrices_1 is None and has_int_rot_dof_1 is False) and (
            degeneracy_matrices_2 is not None or has_int_rot_dof_2 is True):
        # No Degeneracies/Rotation Matrices to get for Oligomer1
        rot1_mat = None
        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1\n")
        ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords_list, set_mat1_np_t)

        # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n\n")
        rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", rot_range_deg_pdb2)
        degen_rot_mat_2 = get_degen_rotmatrices(degeneracy_matrices_2, rotation_matrices_2)

        optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)
        for degen2 in degen_rot_mat_2[degen2_count:]:
            degen2_count += 1
            for rot2_mat in degen2[rot2_count:]:
                rot2_count += 1
                # Rotate Oligomer2 Surface Fragment Guide Coodinates using rot2_mat
                rot2_mat_np_t = np.transpose(rot2_mat)
                surf_frags_2_guide_coords_list_rot_np = np.matmul(surf_frags_oligomer_2_guide_coords_list,
                                                                  rot2_mat_np_t)
                surf_frags_2_guide_coords_list_rot = surf_frags_2_guide_coords_list_rot_np.tolist()

                with open(log_file_path, "a+") as log_file:
                    log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | "
                                   "OLIGOMER 2: Degeneracy %s Rotation %s *****\n"
                                   % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)))

                # Get (Oligomer1 Ghost Fragment, Oligomer2 (rotated) Surface Fragment) guide
                # coodinate pairs in the same Euler rotational space bucket
                with open(log_file_path, "a+") as log_file:
                    log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler "
                                   "rotational space bucket\n")

                surf_frags_2_guide_coords_list_rot_and_set_for_eul = np.matmul(surf_frags_2_guide_coords_list_rot,
                                                                               set_mat2_np_t)

                eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(
                    ghost_frag_guide_coords_list_set_for_eul, surf_frags_2_guide_coords_list_rot_and_set_for_eul)
                eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
                                        true_tup[2]]

                # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
                with open(log_file_path, "a+") as log_file:
                    log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide"
                                   " coordinate pairs\n")

                all_optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frag_list,
                                                                   ghost_frag_guide_coords_list,
                                                                   surf_frag_list,
                                                                   surf_frags_2_guide_coords_list_rot,
                                                                   z_value_func=optimal_tx.apply,
                                                                   max_z_value=init_max_z_val)

                passing_optimal_shifts = list(filter(None, all_optimal_shifts))
                ghostfrag_surffrag_pairs = [(ghost_frag_list[eul_lookup_true_list[idx][0]],
                                             surf_frag_list[eul_lookup_true_list[idx][1]])
                                            for idx, boolean in enumerate(all_optimal_shifts) if boolean]

                if len(passing_optimal_shifts) == 0:
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("No Initial Interface Fragment Matches Found\n\n")
                else:
                    with open(log_file_path, "a+") as log_file:
                        log_file.write("%d Initial Interface Fragment Match(es) Found\n"
                                       % len(passing_optimal_shifts))

                degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
                rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
                                                   (rot1_count, rot2_count))

                out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1,
                    is_zshift2, passing_optimal_shifts, ghostfrag_surffrag_pairs, complete_ghost_frag_list,
                    complete_surf_frag_list, log_file_path, degen_subdir_out_path, rot_subdir_out_path,
                    ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string, design_dim,
                    pdb1_path, pdb2_path, expand_matrices, eul_lookup,
                    rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
                    output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
            rot2_count = 0

    elif (degeneracy_matrices_1 is not None or has_int_rot_dof_1 is True) and (
            degeneracy_matrices_2 is not None or has_int_rot_dof_2 is True):
        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1\n")

        # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
        rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", rot_range_deg_pdb1)
        degen_rot_mat_1 = get_degen_rotmatrices(degeneracy_matrices_1, rotation_matrices_1)

        if not resume:
            with open(log_file_path, "a+") as log_file:
                log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n\n")

        # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
        rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", rot_range_deg_pdb2)
        degen_rot_mat_2 = get_degen_rotmatrices(degeneracy_matrices_2, rotation_matrices_2)
        optimal_tx = OptimalTx.from_dof(set_mat1, set_mat2, is_zshift1, is_zshift2, dof_ext)

        for degen1 in degen_rot_mat_1[degen1_count:]:
            degen1_count += 1
            for rot1_mat in degen1[rot1_count:]:
                rot1_count += 1
                # Rotate Oligomer1 Ghost Fragment Guide Coordinates using rot1_mat
                rot1_mat_np_t = np.transpose(rot1_mat)
                ghost_frag_guide_coords_list_rot_np = np.matmul(ghost_frag_guide_coords_list, rot1_mat_np_t)
                ghost_frag_guide_coords_list_rot = ghost_frag_guide_coords_list_rot_np.tolist()
                ghost_frag_guide_coords_list_rot_and_set_for_eul = np.matmul(ghost_frag_guide_coords_list_rot,
                                                                             set_mat1_np_t)
                for degen2 in degen_rot_mat_2[degen2_count:]:
                    degen2_count += 1
                    for rot2_mat in degen2[rot2_count:]:
                        rot2_count += 1
                        # Rotate Oligomer2 Surface Fragment Guide Coordinates using rot2_mat
                        rot2_mat_np_t = np.transpose(rot2_mat)
                        surf_frags_2_guide_coords_list_rot_np = np.matmul(surf_frags_oligomer_2_guide_coords_list,
                                                                          rot2_mat_np_t)
                        surf_frags_2_guide_coords_list_rot = surf_frags_2_guide_coords_list_rot_np.tolist()

                        with open(log_file_path, "a+") as log_file:
                            log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s "
                                           "Rotation %s *****\n" %
                                           (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)))

                        # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 (rotated) Surface Fragment)
                        # guide coodinate pairs in the same Euler rotational space bucket
                        with open(log_file_path, "a+") as log_file:
                            log_file.write("Get Ghost Fragment/Surface Fragment guide coordinate pairs in the same "
                                           "Euler rotational space bucket\n")

                        surf_frags_2_guide_coords_list_rot_and_set_for_eul = np.matmul(
                            surf_frags_2_guide_coords_list_rot, set_mat2_np_t)
                        # print('Set for Euler Lookup:', surf_frags_2_guide_coords_list_rot_and_set_for_eul[:5])

                        eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(
                            ghost_frag_guide_coords_list_rot_and_set_for_eul,
                            surf_frags_2_guide_coords_list_rot_and_set_for_eul)
                        eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if
                                                true_tup[2]]

                        # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment)
                        # guide coodinate pairs
                        with open(log_file_path, "a+") as log_file:
                            log_file.write("Get optimal shift parameters for the selected Ghost Fragment/Surface "
                                           "Fragment guide coordinate pairs\n")
                        # print('Euler Lookup:', eul_lookup_true_list[:5])
                        all_optimal_shifts = filter_euler_lookup_by_zvalue(eul_lookup_true_list, ghost_frag_list,
                                                                           ghost_frag_guide_coords_list_rot,
                                                                           surf_frag_list,
                                                                           surf_frags_2_guide_coords_list_rot,
                                                                           z_value_func=optimal_tx.apply,
                                                                           max_z_value=init_max_z_val)

                        passing_optimal_shifts = list(filter(None, all_optimal_shifts))
                        ghostfrag_surffrag_pairs = [(ghost_frag_list[eul_lookup_true_list[idx][0]],
                                                     surf_frag_list[eul_lookup_true_list[idx][1]])
                                                    for idx, boolean in enumerate(all_optimal_shifts) if boolean]

                        if len(passing_optimal_shifts) == 0:
                            with open(log_file_path, "a+") as log_file:
                                log_file.write("No Initial Interface Fragment Matches Found\n\n")
                        else:
                            with open(log_file_path, "a+") as log_file:
                                log_file.write("%d Initial Interface Fragment Match(es) Found\n"
                                               % len(passing_optimal_shifts))

                        degen_subdir_out_path = os.path.join(outdir, "DEGEN_%d_%d" % (degen1_count, degen2_count))
                        rot_subdir_out_path = os.path.join(degen_subdir_out_path, "ROT_%d_%d" %
                                                           (rot1_count, rot2_count))

                        out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1,
                            is_zshift2, passing_optimal_shifts, ghostfrag_surffrag_pairs, complete_ghost_frag_list,
                            complete_surf_frag_list, log_file_path, degen_subdir_out_path, rot_subdir_out_path,
                            ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string, design_dim,
                            pdb1_path, pdb2_path, expand_matrices, eul_lookup,
                            rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
                            output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
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
            output_uc, output_surrounding_uc, min_matched, init_match_type, timer, initial = \
            get_docking_parameters(cmd_line_in_params)

        # Master Log File
        master_log_filepath = os.path.join(master_outdir, PUtils.master_log)

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
                      output_exp_assembly, output_uc, output_surrounding_uc, min_matched, init_match_type,
                      keep_time=timer, main_log=initial)

        except KeyboardInterrupt:
            with open(master_log_filepath, "a+") as master_log:
                master_log.write("\nRun Ended By KeyboardInterrupt\n")
            sys.exit()
