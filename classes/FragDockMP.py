import numpy as np
import sklearn.neighbors

from classes.Fragment import *
from classes.OptimalTx import *
from utils.ExpandAssemblyUtils import expanded_design_is_clash
from utils.ExpandAssemblyUtils import generate_cryst1_record
from utils.GeneralUtils import *
from utils.PDBUtils import *
from utils.SamplingUtils import *
from utils.SymmUtils import get_uc_dimensions


def get_last_sampling_state(log_file_path, zero=True):
    """Returns the (zero-indexed) last output state specified in the building_blocks_log.txt file"""
    index = 0
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


def write_frag_match_info_file(ghost_frag, surf_frag, z_value, cluster_id, match_count, res_freq_list, cluster_rmsd,
                               outdir_path, is_initial_match=False):

    out_info_file_path = outdir_path + "/frag_match_info_file.txt"
    out_info_file = open(out_info_file_path, 'w')  # '"a+") KM added to remove old versions of the program run

    aligned_central_res_info = ghost_frag.get_aligned_central_res_info()
    surf_frag_oligomer2_central_res_tup = surf_frag.get_central_res_tup()

    if is_initial_match:
        out_info_file.write("***** MATCH FROM REPRESENTATIVES OF INITIAL INTERFACE FRAGMENT CLUSTERS *****\n\n")

    out_info_file.write("Cluster ID: %s\n" % cluster_id)
    out_info_file.write("Cluster RMSD: %s\n" % str(cluster_rmsd))
    out_info_file.write("Cluster Representative PDB Filename: int_frag_%s_%s.pdb\n" % (cluster_id, str(match_count)))
    out_info_file.write("Cluster Central Residue Pair Frequency: %s\n" % str(res_freq_list))
    out_info_file.write("Ghost Fragment Mapped Chain ID: %s\n" % str(aligned_central_res_info[0]))
    out_info_file.write("Ghost Fragment Mapped Residue Number: %s\n" % str(aligned_central_res_info[1]))
    out_info_file.write("Ghost Fragment Partner Chain ID: %s\n" % str(aligned_central_res_info[2]))
    out_info_file.write("Ghost Fragment Partner Residue Number: %s\n" % str(aligned_central_res_info[3]))
    out_info_file.write("Surface Fragment Oligomer1 Chain ID: %s\n" % str(aligned_central_res_info[4]))
    out_info_file.write("Surface Fragment Oligomer1 Residue Number: %s\n" % str(aligned_central_res_info[5]))
    out_info_file.write("Surface Fragment Oligomer2 Chain ID: %s\n" % str(surf_frag_oligomer2_central_res_tup[0]))
    out_info_file.write("Surface Fragment Oligomer2 Residue Number: %s\n" % str(surf_frag_oligomer2_central_res_tup[1]))
    out_info_file.write("Overlap Z-Value: %s\n\n" + str(z_value))

    if is_initial_match:
        out_info_file.write("***** SUBSEQUENT MATCH(ES) FROM REPRESENTATIVES OF ALL INTERFACE FRAGMENT CLUSTERS *****\n\n")

    out_info_file.close()


def out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2, tx_param_list,
        ghostfrag_surffrag_pair_list, complete_ghost_frag_list, complete_surf_frag_list, log_filepath,
        degen_subdir_out_path, rot_subdir_out_path, ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string,
        design_dim, has_int_rot_dof_1, has_int_rot_dof_2, pdb1_path, pdb2_path, expand_matrices, eul_lookup,
        rot_mat1=None, rot_mat2=None, max_z_val=2.0, output_exp_assembly=False, output_uc=False,
        output_surrounding_uc=False, clash_dist=2.2, min_matched=3):

    for i in range(len(tx_param_list)):

        log_file = open(log_filepath, "a+")
        log_file.write("\t\t Optimal Shift %s" % str(i) + "\n")
        log_file.close()

        ########## Part of: 1 / (1 + z^2) score test ##########

        # Dictionaries for PDB1 and PDB2 with (ch_id, res_num) tuples as keys for every residue that is covered by at
        # least 1 matched fragment. Dictionary values are lists containing 1 / (1 + z^2) values for every fragment match
        # that covers the (ch_id, res_num) residue.
        chid_resnum_scores_dict_pdb1 = {}
        chid_resnum_scores_dict_pdb2 = {}

        # Lists of unique (pdb1/2 chain id, pdb1/2 central residue number) tuples for pdb1/pdb2 interface mono fragments
        # that were matched to an i,j,k fragment in the database with a z value <= 1.
        # This is to keep track of and to count unique 'high quality' matches.
        unique_interface_monofrags_infolist_highqual_pdb1 = []
        unique_interface_monofrags_infolist_highqual_pdb2 = []

        # Number of unique interface mono fragments matched with a z value <= 1 ('high quality match')
        # This value has to be >= min_matched (minimum number of high quality matches required)
        # for a pose to be selected
        high_qual_match_count = 0

        #######################################################

        total_inv_capped_z_val_score = 0
        unique_matched_interface_monofrag_count = 0
        unique_total_interface_monofrags_count = 0
        frag_match_info_list = []
        unique_interface_monofrags_infolist_pdb1 = []
        unique_interface_monofrags_infolist_pdb2 = []
        percent_of_interface_covered = 0.0

        tx_parameters = tx_param_list[i][0]
        initial_overlap_z_val = tx_param_list[i][1]
        ghostfrag_surffrag_pair = ghostfrag_surffrag_pair_list[i]

        # Get Optimal External DOF shifts
        n_dof_external = len(get_ext_dof(ref_frame_tx_dof1, ref_frame_tx_dof2))
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
            representative_ext_dof_tx_params_1 = get_optimal_external_tx_vector(ref_frame_tx_dof1, optimal_ext_dof_shifts)

        # Rotate, Translate and Set PDB1
        pdb1_copy = rot_txint_set_txext_pdb(pdb1, rot_mat=rot_mat1, internal_tx_vec=representative_int_dof_tx_param_1, set_mat=set_mat1, ext_tx_vec=representative_ext_dof_tx_params_1)

        # Get Oligomer2 Optimal Internal Translation vector
        representative_int_dof_tx_param_2 = None
        if is_zshift2:
            representative_int_dof_tx_param_2 = [0, 0, tx_parameters[n_dof_external + 1: n_dof_external + 2][0]]

        # Get Oligomer2 Optimal External Translation vector
        representative_ext_dof_tx_params_2 = None
        if optimal_ext_dof_shifts is not None:
            representative_ext_dof_tx_params_2 = get_optimal_external_tx_vector(ref_frame_tx_dof2, optimal_ext_dof_shifts)

        # Rotate, Translate and Set PDB2
        pdb2_copy = rot_txint_set_txext_pdb(pdb2, rot_mat=rot_mat2, internal_tx_vec=representative_int_dof_tx_param_2, set_mat=set_mat2, ext_tx_vec=representative_ext_dof_tx_params_2)

        copy_rot_tr_set_time_stop = time.time()
        copy_rot_tr_set_time = copy_rot_tr_set_time_stop - copy_rot_tr_set_time_start
        log_file = open(log_filepath, "a+")
        log_file.write("\t\t\t Copy and Transform Oligomer1 and Oligomer2 (took: %s s)\n" % str(copy_rot_tr_set_time))
        log_file.close()

        # Check if PDB1 and PDB2 backbones clash
        oligomer1_oligomer2_clash_time_start = time.time()
        kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1_copy.extract_backbone_coords()))
        cb_clash_count = kdtree_oligomer1_backbone.two_point_correlation(pdb2_copy.extract_backbone_coords(), [clash_dist])
        oligomer1_oligomer2_clash_time_end = time.time()
        oligomer1_oligomer2_clash_time = oligomer1_oligomer2_clash_time_end - oligomer1_oligomer2_clash_time_start

        if cb_clash_count[0] == 0:

            log_file = open(log_filepath, "a+")
            log_file.write("\t\t\t NO Backbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)" % str(oligomer1_oligomer2_clash_time) + "\n")
            log_file.close()

            # Full Interface Fragment Match
            get_int_ghost_surf_frags_time_start = time.time()
            interface_ghostfrag_list, int_monofrag2_list, interface_ghostfrag_guide_coords_list, int_monofrag2_guide_coords_list, unique_interface_frag_count_pdb1, unique_interface_frag_count_pdb2 = get_interface_ghost_surf_frags(pdb1_copy, pdb2_copy, complete_ghost_frag_list, complete_surf_frag_list, rot_mat1, rot_mat2, representative_int_dof_tx_param_1, representative_int_dof_tx_param_2, set_mat1, set_mat2, representative_ext_dof_tx_params_1, representative_ext_dof_tx_params_2)
            get_int_ghost_surf_frags_time_end = time.time()
            get_int_ghost_surf_frags_time = get_int_ghost_surf_frags_time_end - get_int_ghost_surf_frags_time_start

            unique_total_interface_monofrags_count = unique_interface_frag_count_pdb1 + unique_interface_frag_count_pdb2

            if unique_total_interface_monofrags_count > 0:

                log_file = open(log_filepath, "a+")
                log_file.write("\t\t\t Newly Formed Interface Contains %s Unique Fragments on Oligomer 1 and %s on Oligomer 2\n" % (str(unique_interface_frag_count_pdb1), str(unique_interface_frag_count_pdb2)))
                log_file.write("\t\t\t (took: %s s to get interface surface fragments and interface ghost fragments with their guide atoms)\n" % str(get_int_ghost_surf_frags_time))
                log_file.close()

                # Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Mono Fragment) guide coodinate pairs in the same Euler rotational space bucket
                eul_lookup_start_time = time.time()
                eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list, int_monofrag2_guide_coords_list)
                eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]
                eul_lookup_end_time = time.time()
                eul_lookup_time = eul_lookup_end_time - eul_lookup_start_time

                # Get RMSD and z-value for the selected (Ghost Fragment, Interface Fragment) guide coodinate pairs
                pair_count = 0
                total_overlap_count = 0
                overlap_score_time_start = time.time()
                for index_pair in eul_lookup_true_list:
                    interface_ghost_frag = interface_ghostfrag_list[index_pair[0]]
                    interface_ghost_frag_guide_coords = interface_ghostfrag_guide_coords_list[index_pair[0]]
                    ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
                    ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
                    ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
                    cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)
                    interface_ghost_frag_cluster_rmsd = ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][ghost_frag_k_type].get_rmsd()
                    interface_ghost_frag_cluster_res_freq_list = ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][ghost_frag_k_type].get_central_residue_pair_freqs()

                    interface_mono_frag_guide_coords = int_monofrag2_guide_coords_list[index_pair[1]]
                    interface_mono_frag = int_monofrag2_list[index_pair[1]]
                    interface_mono_frag_type = interface_mono_frag.get_type()

                    if (interface_mono_frag_type == ghost_frag_j_type) and (interface_ghost_frag_cluster_rmsd > 0):
                        # Calculate RMSD
                        total_overlap_count += 1
                        e1 = euclidean_squared_3d(interface_mono_frag_guide_coords[0],
                                                  interface_ghost_frag_guide_coords[0])
                        e2 = euclidean_squared_3d(interface_mono_frag_guide_coords[1],
                                                  interface_ghost_frag_guide_coords[1])
                        e3 = euclidean_squared_3d(interface_mono_frag_guide_coords[2],
                                                  interface_ghost_frag_guide_coords[2])
                        sum = e1 + e2 + e3
                        mean = sum / float(3)
                        rmsd = math.sqrt(mean)

                        # Get Guide Atom Overlap Z-Value
                        z_val = rmsd / float(interface_ghost_frag_cluster_rmsd)

                        if z_val <= max_z_val:

                            if z_val == 0:
                                inv_z_val = 3
                            elif 1 / float(z_val) > 3:
                                inv_z_val = 3
                            else:
                                inv_z_val = 1 / float(z_val)

                            total_inv_capped_z_val_score += inv_z_val
                            pair_count += 1

                            ghostfrag_mapped_ch_id, ghostfrag_mapped_central_res_num, ghostfrag_partner_ch_id, ghostfrag_partner_central_res_num = interface_ghost_frag.get_central_res_tup()
                            pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
                            pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num = interface_mono_frag.get_central_res_tup()

                            ########## Part of: 1 / (1 + z^2) score test ##########

                            score_term = 1 / float(1 + (z_val ** 2))

                            covered_residues_pdb1 = [(pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num + j) for j in range(-2, 3)]
                            covered_residues_pdb2 = [(pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num + j) for j in range(-2, 3)]
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

                            if z_val <= 1:
                                if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_highqual_pdb1:
                                    unique_interface_monofrags_infolist_highqual_pdb1.append((pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))
                                if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_highqual_pdb2:
                                    unique_interface_monofrags_infolist_highqual_pdb2.append((pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                            #######################################################

                            if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_pdb1:
                                unique_interface_monofrags_infolist_pdb1.append((pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))

                            if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_pdb2:
                                unique_interface_monofrags_infolist_pdb2.append((pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                            frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val, cluster_id, pair_count, interface_ghost_frag_cluster_res_freq_list, interface_ghost_frag_cluster_rmsd))


                unique_matched_interface_monofrag_count = len(unique_interface_monofrags_infolist_pdb1) + len(unique_interface_monofrags_infolist_pdb2)
                percent_of_interface_covered = unique_matched_interface_monofrag_count / float(unique_total_interface_monofrags_count)

                overlap_score_time_stop = time.time()
                overlap_score_time = overlap_score_time_stop - overlap_score_time_start

                log_file = open(log_filepath, "a+")
                log_file.write("\t\t\t %s Fragment Match(es) Found in Complete Cluster Representative Fragment Library\n" % str(pair_count))
                log_file.write("\t\t\t (Euler Lookup took %s s for %s fragment pairs and Overlap Score Calculation took %s for %s fragment pairs)" % (str(eul_lookup_time), str(len(eul_lookup_all_to_all_list)), str(overlap_score_time), str(total_overlap_count)) + "\n")
                log_file.close()

                high_qual_match_count = len(unique_interface_monofrags_infolist_highqual_pdb1) + len(unique_interface_monofrags_infolist_highqual_pdb2)
                if high_qual_match_count >= min_matched:

                    # Get Unit Cell Dimensions if applicable
                    uc_dimensions = None
                    if optimal_ext_dof_shifts is not None:
                        ref_frame_tx_dof_e = 0
                        ref_frame_tx_dof_f = 0
                        ref_frame_tx_dof_g = 0
                        if len(optimal_ext_dof_shifts) == 1:
                            ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                        if len(optimal_ext_dof_shifts) == 2:
                            ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                            ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
                        if len(optimal_ext_dof_shifts) == 3:
                            ref_frame_tx_dof_e = optimal_ext_dof_shifts[0]
                            ref_frame_tx_dof_f = optimal_ext_dof_shifts[1]
                            ref_frame_tx_dof_g = optimal_ext_dof_shifts[2]

                        uc_dimensions = get_uc_dimensions(uc_spec_string, ref_frame_tx_dof_e, ref_frame_tx_dof_f, ref_frame_tx_dof_g)

                    # Get contacting PDB 1 ASU and PDB 2 ASU
                    asu_pdb_1, asu_pdb_2 = get_contacting_asu(pdb1_copy, pdb2_copy)

                    # Check if design has any clashes when expanded
                    tx_subdir_out_path = rot_subdir_out_path + "/tx_%s" % str(i)
                    if asu_pdb_1 is not None and asu_pdb_2 is not None:
                        exp_des_clash_time_start = time.time()
                        exp_des_is_clash = expanded_design_is_clash(asu_pdb_1, asu_pdb_2, design_dim, result_design_sym, expand_matrices, uc_dimensions, tx_subdir_out_path, output_exp_assembly, output_uc, output_surrounding_uc)
                        exp_des_clash_time_stop = time.time()
                        exp_des_clash_time = exp_des_clash_time_stop - exp_des_clash_time_start

                        if not exp_des_is_clash:

                            if not os.path.exists(degen_subdir_out_path):
                                os.makedirs(degen_subdir_out_path)

                            if not os.path.exists(rot_subdir_out_path):
                                os.makedirs(rot_subdir_out_path)

                            if not os.path.exists(tx_subdir_out_path):
                                os.makedirs(tx_subdir_out_path)

                            log_file = open(log_filepath, "a+")
                            log_file.write("\t\t\t NO Backbone Clash when Designed Assembly is Expanded (took: %s s including writing)" % str(exp_des_clash_time) + "\n")
                            log_file.write("\t\t\t SUCCESSFUL DOCKED POSE: %s\n" % tx_subdir_out_path)
                            log_file.close()

                            # Write PDB1 and PDB2 files
                            cryst1_record = None
                            if optimal_ext_dof_shifts is not None:
                                cryst1_record = generate_cryst1_record(uc_dimensions, result_design_sym)
                            pdb1_copy.write(tx_subdir_out_path + "/" + os.path.splitext(os.path.basename(pdb1.filepath))[0] + "_tx_%s.pdb" % str(i))
                            pdb2_copy.write(tx_subdir_out_path + "/" + os.path.splitext(os.path.basename(pdb2.filepath))[0] + "_tx_%s.pdb" % str(i))

                            # Initial Interface Fragment Match
                            # Rotate, translate and set initial match interface fragment
                            init_match_ghost_frag = ghostfrag_surffrag_pair[0]
                            init_match_ghost_frag_pdb = init_match_ghost_frag.get_pdb()
                            init_match_ghost_frag_pdb_copy = rot_txint_set_txext_pdb(init_match_ghost_frag_pdb, rot_mat=rot_mat1, internal_tx_vec=representative_int_dof_tx_param_1, set_mat=set_mat1, ext_tx_vec=representative_ext_dof_tx_params_1)

                            # Make directories to output matched fragment PDB files
                            # initial_match for the initial matched fragment
                            # high_qual_match for fragments that were matched with z values <= 1
                            # low_qual_match for fragments that were matched with z values > 1
                            matching_fragment_representatives_outdir_path = tx_subdir_out_path + "/matching_fragment_representatives"
                            if not os.path.exists(matching_fragment_representatives_outdir_path):
                                os.makedirs(matching_fragment_representatives_outdir_path)

                            init_match_outdir_path = matching_fragment_representatives_outdir_path + "/initial_match"
                            if not os.path.exists(init_match_outdir_path):
                                os.makedirs(init_match_outdir_path)

                            high_qual_matches_outdir_path = matching_fragment_representatives_outdir_path + "/high_qual_match"
                            if not os.path.exists(high_qual_matches_outdir_path):
                                os.makedirs(high_qual_matches_outdir_path)

                            low_qual_matches_outdir_path = matching_fragment_representatives_outdir_path + "/low_qual_match"
                            if not os.path.exists(low_qual_matches_outdir_path):
                                os.makedirs(low_qual_matches_outdir_path)

                            # Write out initial match interface fragment
                            init_match_surf_frag = ghostfrag_surffrag_pair[1]
                            init_match_ghost_frag_i_type = init_match_ghost_frag.get_i_frag_type()
                            init_match_ghost_frag_j_type = init_match_ghost_frag.get_j_frag_type()
                            init_match_ghost_frag_k_type = init_match_ghost_frag.get_k_frag_type()
                            init_match_ghost_frag_cluster_res_freq_list = ijk_intfrag_cluster_info_dict[init_match_ghost_frag_i_type][init_match_ghost_frag_j_type][init_match_ghost_frag_k_type].get_central_residue_pair_freqs()
                            init_match_cluster_id = "i%s_j%s_k%s" % (init_match_ghost_frag_i_type, init_match_ghost_frag_j_type, init_match_ghost_frag_k_type)
                            # KM commentted out to improve run time Re TODO
                            init_match_ghost_frag_pdb_copy.write(init_match_outdir_path + "/int_frag_i%s_j%s_k%s_0.pdb" % (init_match_ghost_frag_i_type, init_match_ghost_frag_j_type, init_match_ghost_frag_k_type))
                            init_match_ghost_frag_cluster_rmsd = ijk_intfrag_cluster_info_dict[init_match_ghost_frag_i_type][init_match_ghost_frag_j_type][init_match_ghost_frag_k_type].get_rmsd()
                            write_frag_match_info_file(init_match_ghost_frag, init_match_surf_frag, initial_overlap_z_val, init_match_cluster_id, 0, init_match_ghost_frag_cluster_res_freq_list, init_match_ghost_frag_cluster_rmsd, matching_fragment_representatives_outdir_path, is_initial_match=True)

                            # Write out PDB files for all matched interface fragments and associated info
                            for matched_frag in frag_match_info_list:
                                interface_ghost_frag = matched_frag[0]
                                ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
                                ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
                                ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
                                if matched_frag[2] <= 1:
                                    matched_frag_outdir_path = high_qual_matches_outdir_path
                                else:
                                    matched_frag_outdir_path = low_qual_matches_outdir_path
                                # KM commentted out to improve run time Re TODO
                                interface_ghost_frag.get_pdb().write(matched_frag_outdir_path + "/int_frag_i%s_j%s_k%s_%s.pdb" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type, str(matched_frag[4])))
                                write_frag_match_info_file(matched_frag[0], matched_frag[1], matched_frag[2], matched_frag[3], matched_frag[4], matched_frag[5], matched_frag[6], matching_fragment_representatives_outdir_path)

                            # Write out docked pose info to frag_match_info_file.txt
                            if os.path.exists(matching_fragment_representatives_outdir_path + "/frag_match_info_file.txt"):

                                f_l1 = "Nanohedra Score: " + str(total_inv_capped_z_val_score) + "\n"

                                ########## Part of: 1 / (1 + z^2) score test ##########

                                res_lev_sum_score = 0

                                for res_scores_list1 in chid_resnum_scores_dict_pdb1.values():
                                    n1 = 1
                                    res_scores_list_sorted1 = sorted(res_scores_list1, reverse=True)
                                    for sc1 in res_scores_list_sorted1:
                                        res_lev_sum_score += sc1 * (1/float(n1))
                                        n1 = n1 * 2

                                for res_scores_list2 in chid_resnum_scores_dict_pdb2.values():
                                    n2 = 1
                                    res_scores_list_sorted2 = sorted(res_scores_list2, reverse=True)
                                    for sc2 in res_scores_list_sorted2:
                                        res_lev_sum_score += sc2 * (1/float(n2))
                                        n2 = n2 * 2

                                f_l1a = "Residue-Level Summation Score:" + str(res_lev_sum_score) + "\n"
                                f_l2a = "Unique Interface Fragment Match Count With Z value <= 1 (High Quality Match): " + str(high_qual_match_count) + "\n"

                                #######################################################

                                f_l2 = "Unique Interface Fragment Match Count: " + str(unique_matched_interface_monofrag_count) + "\n"
                                f_l3 = "Unique Interface Fragment Total Count: " + str(unique_total_interface_monofrags_count) + "\n"
                                f_l4 = "Percent of Interface Matched: " + str(percent_of_interface_covered) + "\n"
                                f_l5 = "CRYST1 RECORD: " + str(cryst1_record) + "\n\n"

                                f_l6 = "ROT/DEGEN MATRIX PDB1: " + str(rot_mat1) + "\n"
                                if representative_int_dof_tx_param_1 is not None:
                                    int_dof_tx_vec_1 = representative_int_dof_tx_param_1
                                else:
                                    int_dof_tx_vec_1 = None
                                f_l7 = "INTERNAL Tx PDB1: " + str(int_dof_tx_vec_1) + "\n"
                                f_l8 = "SETTING MATRIX PDB1: " + str(set_mat1) + "\n"
                                if representative_ext_dof_tx_params_1 == [0, 0, 0]:
                                    ref_frame_tx_vec_1 = None
                                else:
                                    ref_frame_tx_vec_1 = representative_ext_dof_tx_params_1
                                f_l9 = "REFERENCE FRAME Tx PDB1: " + str(ref_frame_tx_vec_1) + "\n\n"

                                f_l10 = "ROT/DEGEN MATRIX PDB2: " + str(rot_mat2) + "\n"
                                if representative_int_dof_tx_param_2 is not None:
                                    int_dof_tx_vec_2 = representative_int_dof_tx_param_2
                                else:
                                    int_dof_tx_vec_2 = None
                                f_l11 = "INTERNAL Tx PDB2: " + str(int_dof_tx_vec_2) + "\n"
                                f_l12 = "SETTING MATRIX PDB2: " + str(set_mat2) + "\n"
                                if representative_ext_dof_tx_params_2 == [0, 0, 0]:
                                    ref_frame_tx_vec_2 = None
                                else:
                                    ref_frame_tx_vec_2 = representative_ext_dof_tx_params_2
                                f_l13 = "REFERENCE FRAME Tx PDB2: " + str(ref_frame_tx_vec_2) + "\n\n"

                                f_l14 = "INTERNAL Tx PARAMETER PDB1: " + str(representative_int_dof_tx_param_1) + "\n"
                                f_l15 = "INTERNAL Tx PARAMETER PDB2: " + str(representative_int_dof_tx_param_2) + "\n"
                                if optimal_ext_dof_shifts is not None:
                                    efg_tx_params_str = [str(None), str(None), str(None)]
                                    for param_index in range(len(optimal_ext_dof_shifts)):
                                        efg_tx_params_str[param_index] = str(optimal_ext_dof_shifts[param_index])
                                    f_l16 = "REFERENCE FRAME Tx PARAMETER(S): e: %s, f: %s, g: %s" % (efg_tx_params_str[0], efg_tx_params_str[1], efg_tx_params_str[2]) + "\n\n"
                                else:
                                    f_l16 = "REFERENCE FRAME Tx PARAMETER(S): e: None, f: None, g: None" + "\n\n"

                                f_l17 = 'Original PDB 1 Path: %s\n' %pdb1_path
                                f_l18 = 'Original PDB 2 Path: %s\n\n' %pdb2_path

                                f_l19 = 'Oligomer 1 Reference Frame Tx DOF: %s\n' %ref_frame_tx_dof1
                                f_l20 = 'Oligomer 2 Reference Frame Tx DOF: %s\n\n' %ref_frame_tx_dof2

                                f_l21 = "Oligomer 1 Internal ROT DOF: %s\n" % str(has_int_rot_dof_1)
                                f_l22 = "Oligomer 2 Internal ROT DOF: %s\n\n" % str(has_int_rot_dof_2)

                                f_l23 = 'Resulting Design Symmetry: %s\n' %result_design_sym
                                f_l24 = 'Unit Cell Specification: %s\n\n' %uc_spec_string

                                f = open(matching_fragment_representatives_outdir_path + "/frag_match_info_file.txt", "a")
                                f.write("\n")
                                f.write("***** DOCKED POSE *****\n")
                                f.write(f_l1)

                                ########## Part of: 1 / (1 + z^2) score test ##########

                                f.write(f_l1a)
                                f.write(f_l2a)

                                #######################################################

                                f.write(f_l2)
                                f.write(f_l3)
                                f.write(f_l4)
                                f.write(f_l5)
                                f.write(f_l6)
                                f.write(f_l7)
                                f.write(f_l8)
                                f.write(f_l9)
                                f.write(f_l10)
                                f.write(f_l11)
                                f.write(f_l12)
                                f.write(f_l13)
                                f.write(f_l14)
                                f.write(f_l15)
                                f.write(f_l16)
                                f.write(f_l17)
                                f.write(f_l18)
                                f.write(f_l19)
                                f.write(f_l20)
                                f.write(f_l21)
                                f.write(f_l22)
                                f.write(f_l23)
                                f.write(f_l24)
                                f.close()

                        else:
                            log_file = open(log_filepath, "a+")
                            log_file.write("\t\t\t Backbone Clash when Designed Assembly is Expanded (took: %s s)" % str(exp_des_clash_time) + "\n")
                            log_file.close()

                    else:
                        log_file = open(log_filepath, "a+")
                        log_file.write("\t\t\t NO Design ASU Found" + "\n")
                        log_file.close()

                else:
                    log_file = open(log_filepath, "a+")
                    log_file.write("\t\t\t %s < %s Which is Set as the Minimal Required Amount of High Quality Fragment Matches" %(str(high_qual_match_count), str(min_matched)) + "\n")
                    log_file.close()

            else:
                log_file = open(log_filepath, "a+")
                log_file.write("\t\t\t NO Interface Mono Fragments Found" + "\n")
                log_file.close()

        else:
            log_file = open(log_filepath, "a+")
            log_file.write("\t\t\t Backbone Clash when Oligomer1 and Oligomer2 are Docked (took: %s s)" % str(oligomer1_oligomer2_clash_time) + "\n")
            log_file.close()


def dock(init_intfrag_cluster_rep_dict, ijk_intfrag_cluster_rep_dict, init_monofrag_cluster_rep_pdb_dict_1,
         init_monofrag_cluster_rep_pdb_dict_2, init_intfrag_cluster_info_dict, ijk_monofrag_cluster_rep_pdb_dict,
         ijk_intfrag_cluster_info_dict, free_sasa_exe_path, master_outdir, pdb1_path, pdb2_path, set_mat1, set_mat2,
         ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2, result_design_sym, uc_spec_string, design_dim,
         expand_matrices, eul_lookup, init_max_z_val, subseq_max_z_val, degeneracy_matrices_1=None,
         degeneracy_matrices_2=None, rot_step_deg_pdb1=1, rot_range_deg_pdb1=0, rot_step_deg_pdb2=1,
         rot_range_deg_pdb2=0, output_exp_assembly=False, output_uc=False, output_surrounding_uc=False, min_matched=3,
         resume=False, keep_time=True):

    # Output Directory
    pdb1_filename = os.path.splitext(os.path.basename(pdb1_path))[0]
    pdb2_filename = os.path.splitext(os.path.basename(pdb2_path))[0]
    outdir = master_outdir + "/" + pdb1_filename + "_" + pdb2_filename
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    log_filepath = outdir + "/" + pdb1_filename + "_" + pdb2_filename + "_" + "log.txt"

    # Write to Logfile
    if not resume:
        log_file = open(log_filepath, "a+")
        log_file.write("DOCKING %s TO %s\n" % (pdb1_filename, pdb2_filename))
        log_file.write("PDB 1 Path: " + pdb1_path + "\n")
        log_file.write("PDB 2 Path: " + pdb2_path + "\n")
        log_file.write("Output Directory: " + outdir + "\n\n")
        log_file.close()

    # Get PDB1 Symmetric Building Block
    pdb1 = PDB()
    pdb1.readfile(pdb1_path)

    # Get Oligomer 1 Ghost Fragments With Guide Coordinates Using Initial Match Fragment Database
    if not resume:
        log_file = open(log_filepath, "a+")
        log_file.write("Getting %s (Oligomer 1) Ghost Fragments With Guide Coordinates Using Initial Clusters Fragment Database" % pdb1_filename)
        log_file.close()
        if keep_time:
            get_init_ghost_frags_time_start = time.time()

    kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1.extract_backbone_coords()))
    surf_frags_1 = get_surface_fragments(pdb1, free_sasa_exe_path)
    ghost_frag_list = []
    ghost_frag_guide_coords_list = []
    for frag1 in surf_frags_1:
        monofrag1 = MonoFragment(frag1, init_monofrag_cluster_rep_pdb_dict_1)
        monofrag_ghostfrag_list = monofrag1.get_ghost_fragments(init_intfrag_cluster_rep_dict,
                                                                kdtree_oligomer1_backbone)
        if monofrag_ghostfrag_list is not None:
            for ghostfrag in monofrag_ghostfrag_list:
                ghost_frag_list.append(ghostfrag)
                ghost_frag_guide_coords_list.append(ghostfrag.get_guide_coords())
    if not resume and keep_time:
        get_init_ghost_frags_time_stop = time.time()
        get_init_ghost_frags_time = get_init_ghost_frags_time_stop - get_init_ghost_frags_time_start
        log_file = open(log_filepath, "a+")
        log_file.write(" (took: %s s)\n" % str(get_init_ghost_frags_time))
        log_file.close()

    # Get Oligomer1 Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        log_file = open(log_filepath, "a+")
        log_file.write("Getting %s (Oligomer 1) Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database" % pdb1_filename)
        log_file.close()
        if keep_time:
            get_complete_ghost_frags_time_start = time.time()

    complete_ghost_frag_list = []
    for frag1 in surf_frags_1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone)
        if complete_monofrag1_ghostfrag_list is not None:  # remove is not None
            for complete_ghostfrag in complete_monofrag1_ghostfrag_list:
                complete_ghost_frag_list.append(complete_ghostfrag)
    if not resume and keep_time:
        get_complete_ghost_frags_time_stop = time.time()
        get_complete_ghost_frags_time = get_complete_ghost_frags_time_stop - get_complete_ghost_frags_time_start
        log_file = open(log_filepath, "a+")
        log_file.write(" (took: %s s)\n" % str(get_complete_ghost_frags_time))
        log_file.close()

    # Get PDB2 Symmetric Building Block
    pdb2 = PDB()
    pdb2.readfile(pdb2_path)

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using Initial Match Fragment Database
    if not resume:
        log_file = open(log_filepath, "a+")
        log_file.write("Getting Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using Initial Clusters Fragment Database")
        log_file.close()
        if keep_time:
            get_init_surf_frags_time_start = time.time()
    surf_frags_2 = get_surface_fragments(pdb2, free_sasa_exe_path)
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
        log_file = open(log_filepath, "a+")
        log_file.write(" (took: %s s)\n" % str(get_init_surf_frags_time))
        log_file.close()

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database
    if not resume:
        log_file = open(log_filepath, "a+")
        log_file.write("Getting Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database")
        log_file.close()
        if keep_time:
            get_complete_surf_frags_time_start = time.time()
    complete_surf_frag_list = []
    for frag2 in surf_frags_2:
        complete_monofrag2 = MonoFragment(frag2, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords is not None:
            complete_surf_frag_list.append(complete_monofrag2)
    if not resume and keep_time:
        get_complete_surf_frags_time_stop = time.time()
        get_complete_surf_frags_time = get_complete_surf_frags_time_stop - get_complete_surf_frags_time_start
        log_file = open(log_filepath, "a+")
        log_file.write(" (took: %s s)\n\n" % str(get_complete_surf_frags_time))
        log_file.close()

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

    # Transpose Setting Matrices to Set Guide Coordinates Just for Euler Lookup Using np.matmul
    set_mat1_np_t = np.transpose(set_mat1)
    set_mat2_np_t = np.transpose(set_mat2)

    degen1_count, degen2_count, rot1_count, rot2_count = 0, 0, 0, 0
    if resume:
        degen1_count, degen2_count, rot1_count, rot2_count = get_last_sampling_state(log_filepath)

    if (degeneracy_matrices_1 is None and has_int_rot_dof_1 is False) and (degeneracy_matrices_2 is None and has_int_rot_dof_2 is False):
        # No Degeneracies/Rotation Matrices to get for Oligomer1
        rot1_mat = None
        rot2_mat = None
        # TODO add degen1_count
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1" + "\n")
            # No Degeneracies/Rotation Matrices to get for Oligomer2
            log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n" + "\n")
            log_file.close()

        log_file = open(log_filepath, "a+")
        log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****" % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")

        # Get (Oligomer1 Ghost Fragment, Oligomer2 Surface Fragment) guide coodinate pairs in the same Euler rotational space bucket
        log_file.write("\tGet Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational space bucket" + "\n")
        log_file.close()

        ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords_list, set_mat1_np_t)
        surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags_oligomer_2_guide_coords_list, set_mat2_np_t)

        eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_set_for_eul, surf_frags_2_guide_coords_list_set_for_eul)
        eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

        # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
        log_file = open(log_filepath, "a+")
        log_file.write("\tGet optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide coordinate pairs" + "\n")
        log_file.close()

        ghostfrag_surffrag_pair_list = []
        tx_param_list = []
        for index_pair in eul_lookup_true_list:
            ghost_frag = ghost_frag_list[index_pair[0]]
            ghost_frag_guide_coords = ghost_frag_guide_coords_list[index_pair[0]]
            i_type = ghost_frag.get_i_frag_type()
            j_type = ghost_frag.get_j_frag_type()
            k_type = ghost_frag.get_k_frag_type()
            ghost_frag_cluster_rmsd = init_intfrag_cluster_info_dict[i_type][j_type][k_type].get_rmsd()

            surf_frag_guide_coords = surf_frags_oligomer_2_guide_coords_list[index_pair[1]]
            surf_frag = surf_frag_list[index_pair[1]]
            surf_frag_type = surf_frag.get_type()

            if surf_frag_type == j_type:
                o = OptimalTx(set_mat1, set_mat2, is_zshift1, is_zshift2, ghost_frag_cluster_rmsd,
                              ghost_frag_guide_coords, surf_frag_guide_coords, dof_ext)
                o.apply()

                if o.get_zvalue() <= init_max_z_val:
                    ghostfrag_surffrag_pair_list.append((ghost_frag, surf_frag))
                    all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExternalDOFShifts, OptimalInternalDOFShifts]
                    tx_param_list.append((all_optimal_shifts, o.get_zvalue()))

        if len(tx_param_list) == 0:
            log_file = open(log_filepath, "a+")
            log_file.write("\tNo Initial Interface Fragment Matches Found with an Acceptable Optimal Shift\n" + "\n")
            log_file.close()
        elif len(tx_param_list) == 1:
            log_file = open(log_filepath, "a+")
            log_file.write("\t1 Initial Interface Fragment Match Found with an Acceptable Optimal Shift" + "\n")
            log_file.close()
        else:
            log_file = open(log_filepath, "a+")
            log_file.write("\t%s Initial Interface Fragment Matches Found with an Acceptable Optimal Shift" % str(len(tx_param_list)) + "\n")
            log_file.close()

        degen_subdir_out_path = outdir + "/DEGEN_" + str(degen1_count) + "_" + str(degen2_count)
        rot_subdir_out_path = degen_subdir_out_path + "/ROT_" + str(rot1_count) + "_" + str(rot2_count)

        out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2, tx_param_list,
            ghostfrag_surffrag_pair_list, complete_ghost_frag_list, complete_surf_frag_list, log_filepath,
            degen_subdir_out_path, rot_subdir_out_path, ijk_intfrag_cluster_info_dict, result_design_sym,
            uc_spec_string, design_dim, has_int_rot_dof_1, has_int_rot_dof_2, pdb1_path, pdb2_path, expand_matrices,
            eul_lookup, rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
            output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)

    elif (degeneracy_matrices_1 is not None or has_int_rot_dof_1 is True) and (degeneracy_matrices_2 is None and has_int_rot_dof_2 is False):
        # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1" + "\n")
            log_file.close()
        rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", rot_range_deg_pdb1)
        degen_rot_mat_1 = get_degen_rotmatrices(degeneracy_matrices_1, rotation_matrices_1)

        # No Degeneracies/Rotation Matrices to get for Oligomer2
        rot2_mat = None
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("No Rotation/Degeneracy Matrices for Oligomer 2\n" + "\n")
            log_file.close()
        surf_frags_2_guide_coords_list_set_for_eul = np.matmul(surf_frags_oligomer_2_guide_coords_list, set_mat2_np_t)

        for degen1 in degen_rot_mat_1[degen1_count:]:
            degen1_count += 1
            for rot1_mat in degen1[rot1_count:]:
                rot1_count += 1
                # Rotate Oligomer1 Ghost Fragment Guide Coodinates using rot1_mat
                rot1_mat_np_t = np.transpose(rot1_mat)
                ghost_frag_guide_coords_list_rot_np = np.matmul(ghost_frag_guide_coords_list, rot1_mat_np_t)
                ghost_frag_guide_coords_list_rot = ghost_frag_guide_coords_list_rot_np.tolist()

                log_file = open(log_filepath, "a+")
                log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****" % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")
                log_file.close()

                # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 Surface Fragment) guide coodinate pairs in the same Euler rotational space bucket
                log_file = open(log_filepath, "a+")
                log_file.write("\tGet Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational space bucket" + "\n")
                log_file.close()

                ghost_frag_guide_coords_list_rot_and_set_for_eul = np.matmul(ghost_frag_guide_coords_list_rot, set_mat1_np_t)

                eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_rot_and_set_for_eul, surf_frags_2_guide_coords_list_set_for_eul)
                eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

                # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
                log_file = open(log_filepath, "a+")
                log_file.write("\tGet optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide coordinate pairs" + "\n")
                log_file.close()

                ghostfrag_surffrag_pair_list = []
                tx_param_list = []
                for index_pair in eul_lookup_true_list:
                    ghost_frag = ghost_frag_list[index_pair[0]]
                    ghost_frag_guide_coords = ghost_frag_guide_coords_list_rot[index_pair[0]]
                    i_type = ghost_frag.get_i_frag_type()
                    j_type = ghost_frag.get_j_frag_type()
                    k_type = ghost_frag.get_k_frag_type()
                    ghost_frag_cluster_rmsd = init_intfrag_cluster_info_dict[i_type][j_type][k_type].get_rmsd()

                    surf_frag_guide_coords = surf_frags_oligomer_2_guide_coords_list[index_pair[1]]
                    surf_frag = surf_frag_list[index_pair[1]]
                    surf_frag_type = surf_frag.get_type()

                    if surf_frag_type == j_type:
                        o = OptimalTx(set_mat1, set_mat2, is_zshift1, is_zshift2, ghost_frag_cluster_rmsd, ghost_frag_guide_coords, surf_frag_guide_coords, dof_ext)
                        o.apply()

                        if o.get_zvalue() <= init_max_z_val:
                            ghostfrag_surffrag_pair_list.append((ghost_frag, surf_frag))
                            all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExternalDOFShifts, OptimalInternalDOFShifts]
                            tx_param_list.append((all_optimal_shifts, o.get_zvalue()))

                if len(tx_param_list) == 0:
                    log_file = open(log_filepath, "a+")
                    log_file.write(
                        "\tNo Initial Interface Fragment Matches Found with an Acceptable Optimal Shift\n" + "\n")
                    log_file.close()
                elif len(tx_param_list) == 1:
                    log_file = open(log_filepath, "a+")
                    log_file.write(
                        "\t1 Initial Interface Fragment Match Found with an Acceptable Optimal Shift" + "\n")
                    log_file.close()
                else:
                    log_file = open(log_filepath, "a+")
                    log_file.write(
                        "\t%s Initial Interface Fragment Matches Found with an Acceptable Optimal Shift" % str(
                            len(tx_param_list)) + "\n")
                    log_file.close()

                degen_subdir_out_path = outdir + "/DEGEN_" + str(degen1_count) + "_" + str(degen2_count)
                rot_subdir_out_path = degen_subdir_out_path + "/ROT_" + str(rot1_count) + "_" + str(rot2_count)

                out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2,
                    tx_param_list, ghostfrag_surffrag_pair_list, complete_ghost_frag_list, complete_surf_frag_list,
                    log_filepath, degen_subdir_out_path, rot_subdir_out_path, ijk_intfrag_cluster_info_dict,
                    result_design_sym, uc_spec_string, design_dim, has_int_rot_dof_1, has_int_rot_dof_2, pdb1_path,
                    pdb2_path, expand_matrices, eul_lookup, rot1_mat, rot2_mat, max_z_val=subseq_max_z_val,
                    output_exp_assembly=output_exp_assembly, output_uc=output_uc,
                    output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
            rot1_count = 0

    elif (degeneracy_matrices_1 is None and has_int_rot_dof_1 is False) and (degeneracy_matrices_2 is not None or has_int_rot_dof_2 is True):
        # No Degeneracies/Rotation Matrices to get for Oligomer1
        rot1_mat = None
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("No Rotation/Degeneracy Matrices for Oligomer 1" + "\n")
            log_file.close()
        ghost_frag_guide_coords_list_set_for_eul = np.matmul(ghost_frag_guide_coords_list, set_mat1_np_t)

        # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n" + "\n")
            log_file.close()
        rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", rot_range_deg_pdb2)
        degen_rot_mat_2 = get_degen_rotmatrices(degeneracy_matrices_2, rotation_matrices_2)

        for degen2 in degen_rot_mat_2[degen2_count:]:
            degen2_count += 1
            for rot2_mat in degen2[rot2_count:]:
                rot2_count += 1
                # Rotate Oligomer2 Surface Fragment Guide Coodinates using rot2_mat
                rot2_mat_np_t = np.transpose(rot2_mat)
                surf_frags_2_guide_coords_list_rot_np = np.matmul(surf_frags_oligomer_2_guide_coords_list, rot2_mat_np_t)
                surf_frags_2_guide_coords_list_rot = surf_frags_2_guide_coords_list_rot_np.tolist()

                log_file = open(log_filepath, "a+")
                log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****" % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")
                log_file.close()

                # Get (Oligomer1 Ghost Fragment, Oligomer2 (rotated) Surface Fragment) guide coodinate pairs in the same Euler rotational space bucket
                log_file = open(log_filepath, "a+")
                log_file.write("\tGet Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational space bucket" + "\n")
                log_file.close()

                surf_frags_2_guide_coords_list_rot_and_set_for_eul = np.matmul(surf_frags_2_guide_coords_list_rot, set_mat2_np_t)

                eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_set_for_eul, surf_frags_2_guide_coords_list_rot_and_set_for_eul)
                eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

                # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
                log_file = open(log_filepath, "a+")
                log_file.write("\tGet optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide coordinate pairs" + "\n")
                log_file.close()

                ghostfrag_surffrag_pair_list = []
                tx_param_list = []
                for index_pair in eul_lookup_true_list:
                    ghost_frag = ghost_frag_list[index_pair[0]]
                    ghost_frag_guide_coords = ghost_frag_guide_coords_list[index_pair[0]]
                    i_type = ghost_frag.get_i_frag_type()
                    j_type = ghost_frag.get_j_frag_type()
                    k_type = ghost_frag.get_k_frag_type()
                    ghost_frag_cluster_rmsd = init_intfrag_cluster_info_dict[i_type][j_type][k_type].get_rmsd()

                    surf_frag_guide_coords = surf_frags_2_guide_coords_list_rot[index_pair[1]]
                    surf_frag = surf_frag_list[index_pair[1]]
                    surf_frag_type = surf_frag.get_type()

                    if surf_frag_type == j_type:
                        o = OptimalTx(set_mat1, set_mat2, is_zshift1, is_zshift2, ghost_frag_cluster_rmsd,
                                      ghost_frag_guide_coords, surf_frag_guide_coords, dof_ext)
                        o.apply()

                        if o.get_zvalue() <= init_max_z_val:
                            ghostfrag_surffrag_pair_list.append((ghost_frag, surf_frag))
                            all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExternalDOFShifts, OptimalInternalDOFShifts]
                            tx_param_list.append((all_optimal_shifts, o.get_zvalue()))

                if len(tx_param_list) == 0:
                    log_file = open(log_filepath, "a+")
                    log_file.write("\tNo Initial Interface Fragment Matches Found with an Acceptable Optimal Shift\n" + "\n")
                    log_file.close()
                elif len(tx_param_list) == 1:
                    log_file = open(log_filepath, "a+")
                    log_file.write("\t1 Initial Interface Fragment Match Found with an Acceptable Optimal Shift" + "\n")
                    log_file.close()
                else:
                    log_file = open(log_filepath, "a+")
                    log_file.write("\t%s Initial Interface Fragment Matches Found with an Acceptable Optimal Shift" % str(len(tx_param_list)) + "\n")
                    log_file.close()

                degen_subdir_out_path = outdir + "/DEGEN_" + str(degen1_count) + "_" + str(degen2_count)
                rot_subdir_out_path = degen_subdir_out_path + "/ROT_" + str(rot1_count) + "_" + str(rot2_count)

                out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1, is_zshift2,
                    tx_param_list, ghostfrag_surffrag_pair_list, complete_ghost_frag_list, complete_surf_frag_list,
                    log_filepath, degen_subdir_out_path, rot_subdir_out_path, ijk_intfrag_cluster_info_dict,
                    result_design_sym, uc_spec_string, design_dim, has_int_rot_dof_1, has_int_rot_dof_2, pdb1_path,
                    pdb2_path, expand_matrices, eul_lookup, rot1_mat, rot2_mat, max_z_val=subseq_max_z_val,
                    output_exp_assembly=output_exp_assembly, output_uc=output_uc,
                    output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
            rot2_count = 0

    elif (degeneracy_matrices_1 is not None or has_int_rot_dof_1 is True) and (degeneracy_matrices_2 is not None or has_int_rot_dof_2 is True):
        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 1" + "\n")
            log_file.close()

        # Get Degeneracies/Rotation Matrices for Oligomer1: degen_rot_mat_1
        rotation_matrices_1 = get_rot_matrices(rot_step_deg_pdb1, "z", rot_range_deg_pdb1)
        degen_rot_mat_1 = get_degen_rotmatrices(degeneracy_matrices_1, rotation_matrices_1)

        if not resume:
            log_file = open(log_filepath, "a+")
            log_file.write("Obtaining Rotation/Degeneracy Matrices for Oligomer 2\n" + "\n")
            log_file.close()
        # Get Degeneracies/Rotation Matrices for Oligomer2: degen_rot_mat_2
        rotation_matrices_2 = get_rot_matrices(rot_step_deg_pdb2, "z", rot_range_deg_pdb2)
        degen_rot_mat_2 = get_degen_rotmatrices(degeneracy_matrices_2, rotation_matrices_2)

        for degen1 in degen_rot_mat_1[degen1_count:]:
            degen1_count += 1
            for rot1_mat in degen1[rot1_count:]:
                rot1_count += 1
                # Rotate Oligomer1 Ghost Fragment Guide Coordinates using rot1_mat
                rot1_mat_np_t = np.transpose(rot1_mat)
                ghost_frag_guide_coords_list_rot_np = np.matmul(ghost_frag_guide_coords_list, rot1_mat_np_t)
                ghost_frag_guide_coords_list_rot = ghost_frag_guide_coords_list_rot_np.tolist()
                ghost_frag_guide_coords_list_rot_and_set_for_eul = np.matmul(ghost_frag_guide_coords_list_rot, set_mat1_np_t)
                for degen2 in degen_rot_mat_2[degen2_count:]:
                    degen2_count += 1
                    for rot2_mat in degen2[rot2_count:]:
                        rot2_count += 1
                        # Rotate Oligomer2 Surface Fragment Guide Coordinates using rot2_mat
                        rot2_mat_np_t = np.transpose(rot2_mat)
                        surf_frags_2_guide_coords_list_rot_np = np.matmul(surf_frags_oligomer_2_guide_coords_list, rot2_mat_np_t)
                        surf_frags_2_guide_coords_list_rot = surf_frags_2_guide_coords_list_rot_np.tolist()

                        log_file = open(log_filepath, "a+")
                        log_file.write("\n***** OLIGOMER 1: Degeneracy %s Rotation %s | OLIGOMER 2: Degeneracy %s Rotation %s *****" % (str(degen1_count), str(rot1_count), str(degen2_count), str(rot2_count)) + "\n")
                        log_file.close()

                        # Get (Oligomer1 Ghost Fragment (rotated), Oligomer2 (rotated) Surface Fragment) guide coodinate pairs in the same Euler rotational space bucket
                        log_file = open(log_filepath, "a+")
                        log_file.write("\tGet Ghost Fragment/Surface Fragment guide coordinate pairs in the same Euler rotational space bucket" + "\n")
                        log_file.close()

                        eul_time_start = time.time()
                        surf_frags_2_guide_coords_list_rot_and_set_for_eul = np.matmul(surf_frags_2_guide_coords_list_rot, set_mat2_np_t)
                        eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(ghost_frag_guide_coords_list_rot_and_set_for_eul, surf_frags_2_guide_coords_list_rot_and_set_for_eul)
                        eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]
                        eul_time_stop = time.time()
                        eul_time = eul_time_stop - eul_time_start

                        # Euler TIME TEST
                        log_file = open(log_filepath, "a+")
                        log_file.write("\tEuler Search Took: %s s for %s ghost/surf pairs" % (str(eul_time), str(len(eul_lookup_all_to_all_list))) + "\n")
                        log_file.close()

                        # Get optimal shift parameters for the selected (Ghost Fragment, Surface Fragment) guide coodinate pairs
                        log_file = open(log_filepath, "a+")
                        log_file.write("\tGet optimal shift parameters for the selected Ghost Fragment/Surface Fragment guide coordinate pairs" + "\n")
                        log_file.close()

                        ghostfrag_surffrag_pair_list = []
                        tx_param_list = []
                        opt_tx_time_start = time.time()
                        opt_tx_count = 0
                        for index_pair in eul_lookup_true_list:
                            ghost_frag = ghost_frag_list[index_pair[0]]
                            ghost_frag_guide_coords = ghost_frag_guide_coords_list_rot[index_pair[0]]
                            i_type = ghost_frag.get_i_frag_type()
                            j_type = ghost_frag.get_j_frag_type()
                            k_type = ghost_frag.get_k_frag_type()
                            ghost_frag_cluster_rmsd = init_intfrag_cluster_info_dict[i_type][j_type][k_type].get_rmsd()

                            surf_frag_guide_coords = surf_frags_2_guide_coords_list_rot[index_pair[1]]
                            surf_frag = surf_frag_list[index_pair[1]]
                            surf_frag_type = surf_frag.get_type()

                            if surf_frag_type == j_type:
                                opt_tx_count += 1
                                o = OptimalTx(set_mat1, set_mat2, is_zshift1, is_zshift2, ghost_frag_cluster_rmsd,
                                              ghost_frag_guide_coords, surf_frag_guide_coords, dof_ext)
                                o.apply()

                                if o.get_zvalue() <= init_max_z_val:
                                    ghostfrag_surffrag_pair_list.append((ghost_frag, surf_frag))
                                    all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExternalDOFShifts, OptimalInternalDOFShifts]
                                    tx_param_list.append((all_optimal_shifts, o.get_zvalue()))

                        # Optimal Shift Time Test
                        opt_tx_time_stop = time.time()
                        opt_tx_time = opt_tx_time_stop - opt_tx_time_start
                        log_file = open(log_filepath, "a+")
                        log_file.write("\tOptimal Shift Search Took: %s s for %s guide coordinate pairs\n" % (str(opt_tx_time), str(opt_tx_count)) + "\n")
                        log_file.close()

                        if len(tx_param_list) == 0:
                            log_file = open(log_filepath, "a+")
                            log_file.write("\tNo Initial Interface Fragment Matches Found with an Acceptable Optimal Shift\n" + "\n")
                            log_file.close()
                        elif len(tx_param_list) == 1:
                            log_file = open(log_filepath, "a+")
                            log_file.write("\t1 Initial Interface Fragment Match Found with an Acceptable Optimal Shift" + "\n")
                            log_file.close()
                        else:
                            log_file = open(log_filepath, "a+")
                            log_file.write("\t%s Initial Interface Fragment Matches Found with an Acceptable Optimal Shift" % str(len(tx_param_list)) + "\n")
                            log_file.close()

                        degen_subdir_out_path = outdir + "/DEGEN_" + str(degen1_count) + "_" + str(degen2_count)
                        rot_subdir_out_path = degen_subdir_out_path + "/ROT_" + str(rot1_count) + "_" + str(rot2_count)

                        out(pdb1, pdb2, set_mat1, set_mat2, ref_frame_tx_dof1, ref_frame_tx_dof2, is_zshift1,
                            is_zshift2, tx_param_list, ghostfrag_surffrag_pair_list, complete_ghost_frag_list,
                            complete_surf_frag_list, log_filepath, degen_subdir_out_path, rot_subdir_out_path,
                            ijk_intfrag_cluster_info_dict, result_design_sym, uc_spec_string, design_dim,
                            has_int_rot_dof_1, has_int_rot_dof_2, pdb1_path, pdb2_path, expand_matrices, eul_lookup,
                            rot1_mat, rot2_mat, max_z_val=subseq_max_z_val, output_exp_assembly=output_exp_assembly,
                            output_uc=output_uc, output_surrounding_uc=output_surrounding_uc, min_matched=min_matched)
                    rot2_count = 0
                degen2_count = 0
            rot1_count = 0
