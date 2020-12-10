import math

import numpy as np
import pandas as pd

from SymDesignUtils import get_all_pdb_file_paths, read_pdb, fill_pdb, unpickle, mp_map, print_atoms, start_log
from nanohedra.classes.EulerLookup import EulerLookup
from nanohedra.classes.Fragment import *
from nanohedra.utils.CmdLineArgParseUtils import *
from nanohedra.utils.ExpandAssemblyUtils import *
from nanohedra.utils.GeneralUtils import euclidean_squared_3d

# Globals
# Nanohedra.py Path
main_script_path = os.path.dirname(os.path.realpath(__file__))

# Fragment Database Directory Paths
frag_db = os.path.join(main_script_path, "fragment_database")
monofrag_cluster_rep_dirpath = os.path.join(frag_db, "Top5MonoFragClustersRepresentativeCentered")
ijk_intfrag_cluster_rep_dirpath = os.path.join(frag_db, "Top75percent_IJK_ClusterRepresentatives_1A")
intfrag_cluster_info_dirpath = os.path.join(frag_db, "IJK_ClusteredInterfaceFragmentDBInfo_1A")

# Free SASA Executable Path
free_sasa_exe_path = os.path.join(main_script_path, 'nanohedra', "sasa", "freesasa-2.0", "src", "freesasa")

# Create fragment database for all ijk cluster representatives
ijk_frag_db = FragmentDB(monofrag_cluster_rep_dirpath, ijk_intfrag_cluster_rep_dirpath,
                         intfrag_cluster_info_dirpath)
# Get complete IJK fragment representatives database dictionaries
ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()

# Initialize Euler Lookup Class
eul_lookup = EulerLookup()


def get_interface_fragment_chain_residue_numbers(pdb1, pdb2, cb_distance=8):
    # Get the interface residues
    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))

    # Get interface fragment information
    pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list = [], []
    for pair in interacting_pairs:

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    if atom.is_CA():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    if atom.is_CA():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_central_chainid_resnum_unique_list:
                pdb1_central_chainid_resnum_unique_list.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_central_chainid_resnum_unique_list:
                pdb2_central_chainid_resnum_unique_list.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list


def score_interface(pdb1, pdb2, pdb1_central_chainid_resnum_unique_l, pdb2_central_chainid_resnum_unique_l):
    # Initialize variables
    max_z_val = 2

    kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1.extract_backbone_coords()))
    int_frags_1 = get_interface_fragments(pdb1, pdb1_central_chainid_resnum_unique_l)
    print('pdb1_central_chainid_resnum_unique_l:\n', pdb1_central_chainid_resnum_unique_l)
    # surf_frags_1 = get_surface_fragments(pdb1, free_sasa_exe_path)
    # print(int_frags_1)
    complete_int1_ghost_frag_l = []
    for frag1 in int_frags_1:
        # print(frag1)
        # print('frag1:\n')
        # print_atoms(frag1.all_atoms)  # TODO
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        print('complete_monofrag1:\n')
        print_atoms(complete_monofrag1.pdb.all_atoms)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone)
        if complete_monofrag1_ghostfrag_list:
            print('complete_monofrag1_ghostfrag_list:\n')
            print_atoms(complete_monofrag1_ghostfrag_list[0].all_atoms)  # TODO
            print(complete_monofrag1_ghostfrag_list)  # I don't think this is working (None). Must be something missing from the input pdb (monofrag1 or kdtree_oligomer1_backbone)
            # complete_ghost_frag_list.extend(complete_monofrag1_ghostfrag_list) # TODO KM MOD
            for complete_ghostfrag in complete_monofrag1_ghostfrag_list:
                complete_int1_ghost_frag_l.append(complete_ghostfrag)
                # complete_int1_ghost_frag_l.extend(complete_monofrag1_ghostfrag_list)

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using Initial Match Fragment Database
    int_frags_2 = get_interface_fragments(pdb2, pdb2_central_chainid_resnum_unique_l)
    # surf_frags_2 = get_surface_fragments(pdb2, free_sasa_exe_path)
    # print(int_frags_2)

    # Get Oligomer 2 Surface (Mono) Fragments With Guide Coordinates Using COMPLETE Fragment Database
    complete_int2_frag_l, complete_surf_frag_guide_coord_l = [], []
    for frag2 in int_frags_2:
        # print(frag2.all_atoms)
        complete_monofrag2 = MonoFragment(frag2, ijk_monofrag_cluster_rep_pdb_dict)
        print('complete_monofrag2:\n')
        print_atoms(complete_monofrag2.pdb.all_atoms)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()  # This is a precomputation with really no time savings, just program overhead
        if complete_monofrag2_guide_coords is not None:
            print(complete_monofrag2_guide_coords)
            complete_int2_frag_l.append(complete_monofrag2)
            # complete_surf_frag_guide_coord_l.append(complete_monofrag2_guide_coords)

    # del ijk_monofrag_cluster_rep_pdb_dict, init_monofrag_cluster_rep_pdb_dict_1, init_monofrag_cluster_rep_pdb_dict_2

    interface_ghostfrag_list, interface_ghost_frag_pdb_coords_list, interface_ghostfrag_guide_coords_list = [], [], []
    # print(complete_int1_ghost_frag_l)
    for ghost_frag in complete_int1_ghost_frag_l:
        print(ghost_frag.get_aligned_surf_frag_central_res_tup())
        if ghost_frag.get_aligned_surf_frag_central_res_tup() in pdb1_central_chainid_resnum_unique_l:
            interface_ghostfrag_list.append(ghost_frag)
            # interface_ghost_frag_pdb_coords_list.append(ghost_frag.get_pdb_coords())
            interface_ghostfrag_guide_coords_list.append(ghost_frag.get_guide_coords())

    interface_surf_frag_list, interface_surf_frag_pdb_coords_list, interface_surf_frag_guide_coords_list = [], [], []
    # print(complete_int2_frag_l)
    for surf_frag in complete_int2_frag_l:
        print(surf_frag.get_central_res_tup())
        if surf_frag.get_central_res_tup() in pdb2_central_chainid_resnum_unique_l:
            interface_surf_frag_list.append(surf_frag)
            # interface_surf_frag_pdb_coords_list.append(surf_frag.get_pdb_coords())
            interface_surf_frag_guide_coords_list.append(surf_frag.get_guide_coords())

    # Check for matching Euler angles
    print(interface_ghostfrag_guide_coords_list)
    print(interface_surf_frag_guide_coords_list)
    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list,
                                                               interface_surf_frag_guide_coords_list)
    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

    ########## Part of: 1 / (1 + z^2) score test ##########

    # Dictionaries for PDB1 and PDB2 with (ch_id, res_num) tuples as keys for every residue that is covered by at
    # least 1 matched fragment. Dictionary values are lists containing 1 / (1 + z^2) values for every fragment match
    # that covers the (ch_id, res_num) residue.
    chid_resnum_scores_dict_pdb1 = {}
    chid_resnum_scores_dict_pdb2 = {}
    central_residues_scores_d_pdb1, central_residues_scores_d_pdb2 = {}, {}

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
    # unique_matched_interface_monofrag_count = 0
    # unique_total_interface_monofrags_count = 0
    frag_match_info_list = []
    unique_interface_monofrags_infolist_pdb1 = []
    unique_interface_monofrags_infolist_pdb2 = []
    # percent_of_interface_covered = 0.0
    pair_count = 0
    total_overlap_count = 0
    # tx_parameters = tx_param_list[i][0]
    # initial_overlap_z_val = tx_param_list[i][1]
    # ghostfrag_surffrag_pair = ghostfrag_surffrag_pair_list[i]
    unique_fragment_indicies = []
    fragment_i_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    fragment_j_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Get RMSD and z-value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
    for index_pair in eul_lookup_true_list:
        interface_ghost_frag = interface_ghostfrag_list[index_pair[0]]
        interface_ghost_frag_guide_coords = interface_ghostfrag_guide_coords_list[index_pair[0]]
        ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
        ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
        ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
        fragment_i_index_count_d[ghost_frag_i_type] += 1
        fragment_j_index_count_d[ghost_frag_j_type] += 1
        cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)
        unique_fragment_indicies.append(cluster_id)
        interface_ghost_frag_cluster_rmsd = ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
            ghost_frag_k_type].get_rmsd()
        interface_ghost_frag_cluster_res_freq_list = \
            ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
                ghost_frag_k_type].get_central_residue_pair_freqs()

        interface_mono_frag_guide_coords = interface_surf_frag_guide_coords_list[index_pair[1]]
        interface_mono_frag = interface_surf_frag_list[index_pair[1]]
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

                # ghostfrag_mapped_ch_id, ghostfrag_mapped_central_res_num, ghostfrag_partner_ch_id, ghostfrag_partner_central_res_num = interface_ghost_frag.get_central_res_tup()
                pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
                pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num = interface_mono_frag.get_central_res_tup()

                ########## Part of: 1 / (1 + z^2) score test ##########

                score_term = 1 / float(1 + (z_val ** 2))

                # Central residue only score
                if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in central_residues_scores_d_pdb1:
                    central_residues_scores_d_pdb1[(pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num)] = [score_term]
                else:
                    central_residues_scores_d_pdb1[
                        (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num)].append(score_term)

                if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in central_residues_scores_d_pdb2:
                    central_residues_scores_d_pdb2[(pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num)] = [score_term]
                else:
                    central_residues_scores_d_pdb2[(pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num)].append(score_term)

                covered_residues_pdb1 = [(pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num + j)
                                         for j in range(-2, 3)]
                covered_residues_pdb2 = [(pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num + j)
                                         for j in range(-2, 3)]
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
                        unique_interface_monofrags_infolist_highqual_pdb1.append(
                            (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))
                    if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_highqual_pdb2:
                        unique_interface_monofrags_infolist_highqual_pdb2.append(
                            (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                #######################################################

                if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_pdb1:
                    unique_interface_monofrags_infolist_pdb1.append(
                        (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))

                if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in unique_interface_monofrags_infolist_pdb2:
                    unique_interface_monofrags_infolist_pdb2.append(
                        (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val, cluster_id, pair_count,
                                             interface_ghost_frag_cluster_res_freq_list,
                                             interface_ghost_frag_cluster_rmsd))

    ########## Part of: 1 / (1 + z^2) score test ##########
    res_lev_sum_score = 0
    center_lev_sum_score = 0
    # Center only
    for central_res_scores_l1 in central_residues_scores_d_pdb1.values():
        n1 = 1
        central_res_scores_l_sorted1 = sorted(central_res_scores_l1, reverse=True)
        for sc1 in central_res_scores_l_sorted1:
            center_lev_sum_score += sc1 * (1 / float(n1))
            n1 *= 2
    for central_res_scores_l2 in central_residues_scores_d_pdb2.values():
        n2 = 1
        central_res_scores_l_sorted2 = sorted(central_res_scores_l2, reverse=True)
        for sc1 in central_res_scores_l_sorted2:
            center_lev_sum_score += sc1 * (1 / float(n2))
            n2 *= 2

    # All residues
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

    # Metric calculation
    # unique_fragments = len(central_residues_scores_d_pdb1) + len(central_residues_scores_d_pdb2)
    unique_matched_interface_monofrag_count = len(unique_interface_monofrags_infolist_pdb1) + len(
        unique_interface_monofrags_infolist_pdb2)
    unique_total_interface_residue_count = len(pdb1_central_chainid_resnum_unique_l) + len(pdb2_central_chainid_resnum_unique_l)
    percent_of_interface_covered = unique_matched_interface_monofrag_count / float(unique_total_interface_residue_count)

    # Sum the total contribution from each fragment type on both sides of the interface
    fragment_content_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for index in fragment_i_index_count_d:
        fragment_content_d[index] += fragment_i_index_count_d[index]
        fragment_content_d[index] += fragment_j_index_count_d[index]

    if len(unique_fragment_indicies) > 0:
        for index in fragment_content_d:
            fragment_content_d[index] = index/len(unique_fragment_indicies)

    f_l1a = "Residue-Level Summation Score:" + str(res_lev_sum_score) + "\n"

    f_l2 = "Unique Interface Fragment Match Count: " + str(unique_matched_interface_monofrag_count) + "\n"
    f_l3 = "Unique Interface Fragment Total Count: " + str(unique_total_interface_residue_count) + "\n"
    f_l4 = "Percent of Interface Matched: " + str(percent_of_interface_covered) + "\n"

    return res_lev_sum_score, center_lev_sum_score, unique_fragment_indicies, unique_matched_interface_monofrag_count, \
           unique_total_interface_residue_count, percent_of_interface_covered, fragment_content_d


def calculate_interface_score(interface_path):  # , free_sasa_exe_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "sasa", "freesasa-2.0", "src", "freesasa")):
    interface_name = os.path.splitext(os.path.basename(interface_path))[0]
    pdb = read_pdb(interface_path)
    pdb1 = fill_pdb(pdb.chain(pdb.chain_id_list[0]))
    pdb1.update_attributes_from_pdb(pdb)
    pdb2 = fill_pdb(pdb.chain(pdb.chain_id_list[-1]))
    pdb2.update_attributes_from_pdb(pdb)

    pdb1_central_chainid_resnum_l, pdb2_central_chainid_resnum_l = get_interface_fragment_chain_residue_numbers(pdb1, pdb2)
    pdb1_interface_sa = pdb1.get_chain_residue_surface_area(pdb1_central_chainid_resnum_l, free_sasa_exe_path)
    pdb2_interface_sa = pdb2.get_chain_residue_surface_area(pdb2_central_chainid_resnum_l, free_sasa_exe_path)
    interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa
    res_level_sum_score, center_level_sum_score, fragment_indices, num_residues_with_fragments, total_residues, \
        percent_interface_fragment, fragment_content_d = \
        score_interface(pdb1, pdb2, pdb1_central_chainid_resnum_l, pdb2_central_chainid_resnum_l)
    # interface_d[interface_name] = {'score': res_level_sum_score, 'number_fragments': number_fragments,
    #                                'total_residues': total_residues, 'percent_fragment': percent_interface_fragment}
    return {interface_name: {'score': res_level_sum_score, 'central_score': center_level_sum_score,
                             'fragment_cluster_ids': fragment_indices, 'unique_fragments': len(fragment_indices),
                             'percent_fragment': len(fragment_indices)/float(total_residues),
                             'number_fragment_residues': num_residues_with_fragments, 'total_interface_residues': total_residues,
                             'percent_interface_covered_with_fragment': percent_interface_fragment, 'interface_area': interface_buried_sa,
                             'percent_interface_helix': fragment_content_d[1], 'percent_interface_strand': fragment_content_d[2],
                             'percent_interface_coil': fragment_content_d[3] + fragment_content_d[4] + fragment_content_d[5]}}


if __name__ == '__main__':
    # Program input
    print('USAGE: python ScoreNative.py interface_type_pickled_dict interface_filepath_location number_of_threads')

    # if args.debug:
    #     logger = start_log(name=os.path.basename(__file__), level=1)
    #     logger.debug('Debug mode. Verbose output')
    # else:
    logger = start_log(name=os.path.basename(__file__), level=2)
    interface_reference_d = unpickle(sys.argv[1])
    bio_reference_l = interface_reference_d['bio']
    print(bio_reference_l[5:10])
    # try:
    #     print('1:', '1AB0-1' in bio_reference_l)
    #     print('2:', '1AB0-2' in bio_reference_l)
    #     print('no dash:', '1AB0' in bio_reference_l)
    # except KeyError as e:
    #     print(e)

    if 'debug' in sys.argv:
        # next_5 = '3ILK-1', '3G64-1', '3G64-4', '3G64-6', '3G64-23'
        root = '/home/kmeador/yeates/fragment_database/all/all_interfaces/%s.pdb'
        paths = ['2OPI-1', '4JVT-1', '3GZD-1', '4IRG-1', '2IN5-1']
        # paths = ['op/2OPI-1.pdb', 'jv/4JVT-1.pdb', 'gz/3GZD-1.pdb', 'ir/4IRG-1.pdb', 'in/2IN5-1.pdb']
        interface_filepaths = [root % '%s/%s' % (path[1:3].lower(), path) for path in paths]
        # interface_filepaths = list(map(os.path.join, root, paths))
    else:
        interface_filepaths = get_all_pdb_file_paths(sys.argv[2])

    missing_index = [i for i, filepath in enumerate(interface_filepaths) if os.path.splitext(os.path.basename(filepath))[0] not in bio_reference_l]

    for i in reversed(missing_index):
        del interface_filepaths[i]

    # # Nanohedra.py Path
    # main_script_path = os.path.dirname(os.path.realpath(__file__))
    #
    # # Fragment Database Directory Paths
    # frag_db = os.path.join(main_script_path, "fragment_database")
    # monofrag_cluster_rep_dirpath = os.path.join(frag_db, "Top5MonoFragClustersRepresentativeCentered")
    # ijk_intfrag_cluster_rep_dirpath = os.path.join(frag_db, "Top75percent_IJK_ClusterRepresentatives_1A")
    # intfrag_cluster_info_dirpath = os.path.join(frag_db, "IJK_ClusteredInterfaceFragmentDBInfo_1A")
    #
    # # Free SASA Executable Path
    # free_sasa_exe_path = os.path.join(main_script_path, "sasa", "freesasa-2.0", "src", "freesasa")
    #
    # # Create fragment database for all ijk cluster representatives
    # ijk_frag_db = FragmentDB(monofrag_cluster_rep_dirpath, ijk_intfrag_cluster_rep_dirpath,
    #                          intfrag_cluster_info_dirpath)
    # # Get complete IJK fragment representatives database dictionaries
    # ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
    # ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
    # ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()
    #
    # # Initialize Euler Lookup Class
    # eul_lookup = EulerLookup()
    # # interface_d = {}
    # # for interface_path in interface_filepaths:

    results = mp_map(calculate_interface_score, interface_filepaths, threads=int(sys.argv[3]))
    interface_d = {key: result[key] for result in results for key in result}
    interface_df = pd.DataFrame(interface_d)
    interface_df.to_csv('BiologicalInterfaceNanohedraScores.csv')
