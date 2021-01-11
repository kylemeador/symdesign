import argparse
import math

import numpy as np
import pandas as pd

from SymDesignUtils import start_log, unpickle, get_all_pdb_file_paths, mp_map
# from symdesign.interface_analysis.InterfaceSorting import return_pdb_interface
from classes import EulerLookup
from utils import euclidean_squared_3d

# Globals
# Nanohedra.py Path
main_script_path = os.path.dirname(os.path.realpath(__file__))

# Fragment Database Directory Paths
frag_db = os.path.join(main_script_path, 'data', 'databases', 'fragment_db', 'biological_interfaces')
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
if not ijk_intfrag_cluster_rep_dict:
    print('No reps found!')

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


def score_interface(pdb1, pdb2, pdb1_unique_chain_central_res_l, pdb2_unique_chain_central_res_l):
    # Initialize variables
    max_z_val = 2

    kdtree_oligomer1_backbone = sklearn.neighbors.BallTree(np.array(pdb1.extract_backbone_coords()))
    # Get pdb1 interface fragments with guide coordinates using fragment database
    int_frags_1 = get_interface_fragments(pdb1, pdb1_unique_chain_central_res_l)

    complete_int1_ghost_frag_l = []
    for frag1 in int_frags_1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone)
        if complete_monofrag1_ghostfrag_list:
            for complete_ghostfrag in complete_monofrag1_ghostfrag_list:
                complete_int1_ghost_frag_l.append(complete_ghostfrag)
                # complete_int1_ghost_frag_l.extend(complete_monofrag1_ghostfrag_list)

    # Get pdb2 interface fragments with guide coordinates using complete fragment database
    int_frags_2 = get_interface_fragments(pdb2, pdb2_unique_chain_central_res_l)

    complete_int2_frag_l, complete_surf_frag_guide_coord_l = [], []
    for frag2 in int_frags_2:
        complete_monofrag2 = MonoFragment(frag2, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords:
            complete_int2_frag_l.append(complete_monofrag2)
            # complete_surf_frag_guide_coord_l.append(complete_monofrag2_guide_coords)

    # del ijk_monofrag_cluster_rep_pdb_dict, init_monofrag_cluster_rep_pdb_dict_1, init_monofrag_cluster_rep_pdb_dict_2

    interface_ghostfrag_list, interface_ghost_frag_pdb_coords_list, interface_ghostfrag_guide_coords_list = [], [], []
    for ghost_frag in complete_int1_ghost_frag_l:
        if ghost_frag.get_aligned_surf_frag_central_res_tup() in pdb1_unique_chain_central_res_l:
            interface_ghostfrag_list.append(ghost_frag)
            # interface_ghost_frag_pdb_coords_list.append(ghost_frag.get_pdb_coords())
            interface_ghostfrag_guide_coords_list.append(ghost_frag.get_guide_coords())

    interface_surf_frag_list, interface_surf_frag_pdb_coords_list, interface_surf_frag_guide_coords_list = [], [], []
    for surf_frag in complete_int2_frag_l:
        if surf_frag.get_central_res_tup() in pdb2_unique_chain_central_res_l:
            interface_surf_frag_list.append(surf_frag)
            # interface_surf_frag_pdb_coords_list.append(surf_frag.get_pdb_coords())
            interface_surf_frag_guide_coords_list.append(surf_frag.get_guide_coords())

    # Check for matching Euler angles
    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list,
                                                               interface_surf_frag_guide_coords_list)
    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

    # Dictionaries for PDB1 and PDB2 with (ch_id, res_num) tuples as keys for every residue that is covered by at
    # least 1 matched fragment. Dictionary values are lists containing 1 / (1 + z^2) values for every fragment match
    # that covers the (ch_id, res_num) residue.
    chid_resnum_scores_dict_pdb1 = {}
    chid_resnum_scores_dict_pdb2 = {}
    central_residues_scores_d_pdb1, central_residues_scores_d_pdb2 = {}, {}

    # Lists of unique (pdb1/2 chain id, pdb1/2 central residue number) tuples for pdb1/pdb2 interface mono fragments
    # that were matched to an i,j,k fragment in the database with a z value <= 1.
    # This is to keep track of and to count unique 'high quality' matches.
    # unique_interface_monofrags_infolist_highqual_pdb1 = []
    # unique_interface_monofrags_infolist_highqual_pdb2 = []

    # Number of unique interface mono fragments matched with a z value <= 1 ('high quality match')
    # This value has to be >= min_matched (minimum number of high quality matches required)
    # for a pose to be selected
    # high_qual_match_count = 0

    #######################################################

    # total_inv_capped_z_val_score = 0
    # number_residues_in_fragments = 0
    # unique_total_interface_monofrags_count = 0
    # frag_match_info_list = []
    pdb1_unique_interface_frag_info_l = []
    pdb2_unique_interface_frag_info_l = []
    # percent_interface_matched = 0.0
    # pair_count = 0
    total_overlap_count = 0
    # tx_parameters = tx_param_list[i][0]
    # initial_overlap_z_val = tx_param_list[i][1]
    # ghostfrag_surffrag_pair = ghostfrag_surffrag_pair_list[i]
    unique_fragment_indicies = []
    fragment_i_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    fragment_j_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    # fragment_i_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # fragment_j_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Get RMSD and z-value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
    for index_pair in eul_lookup_true_list:
        interface_ghost_frag = interface_ghostfrag_list[index_pair[0]]
        interface_ghost_frag_guide_coords = interface_ghostfrag_guide_coords_list[index_pair[0]]
        ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
        ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
        ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
        cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)
        interface_ghost_frag_cluster_rmsd = ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
            ghost_frag_k_type].get_rmsd()
        # interface_ghost_frag_cluster_res_freq_list = \
        #     ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
        #         ghost_frag_k_type].get_central_residue_pair_freqs()

        interface_mono_frag = interface_surf_frag_list[index_pair[1]]
        interface_mono_frag_guide_coords = interface_surf_frag_guide_coords_list[index_pair[1]]
        interface_mono_frag_type = interface_mono_frag.get_type()

        if (interface_mono_frag_type == ghost_frag_j_type) and (interface_ghost_frag_cluster_rmsd > 0):
            # Calculate RMSD
            total_overlap_count += 1
            e1 = euclidean_squared_3d(interface_mono_frag_guide_coords[0], interface_ghost_frag_guide_coords[0])
            e2 = euclidean_squared_3d(interface_mono_frag_guide_coords[1], interface_ghost_frag_guide_coords[1])
            e3 = euclidean_squared_3d(interface_mono_frag_guide_coords[2], interface_ghost_frag_guide_coords[2])
            s = e1 + e2 + e3
            mean = s / float(3)
            rmsd = math.sqrt(mean)

            # Get Guide Atom Overlap Z-Value
            z_val = rmsd / float(interface_ghost_frag_cluster_rmsd)

            if z_val <= max_z_val:
                fragment_i_index_count_d[ghost_frag_i_type] += 1
                fragment_j_index_count_d[ghost_frag_j_type] += 1
                unique_fragment_indicies.append(cluster_id)
                # if z_val == 0:
                #     inv_z_val = 3
                # elif 1 / float(z_val) > 3:
                #     inv_z_val = 3
                # else:
                #     inv_z_val = 1 / float(z_val)
                #
                # total_inv_capped_z_val_score += inv_z_val
                # pair_count += 1

                # ghostfrag_mapped_ch_id, ghostfrag_mapped_central_res_num, ghostfrag_partner_ch_id, \
                # ghostfrag_partner_central_res_num = interface_ghost_frag.get_central_res_tup()
                pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num = \
                    interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
                pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num = \
                    interface_mono_frag.get_central_res_tup()

                score_term = 1 / float(1 + (z_val ** 2))

                # Central residue only score
                if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in \
                        central_residues_scores_d_pdb1:
                    central_residues_scores_d_pdb1[(pdb1_interface_surffrag_ch_id,
                                                    pdb1_interface_surffrag_central_res_num)] = [score_term]
                else:
                    central_residues_scores_d_pdb1[
                        (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num)].append(score_term)

                if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in \
                        central_residues_scores_d_pdb2:
                    central_residues_scores_d_pdb2[(pdb2_interface_surffrag_ch_id,
                                                    pdb2_interface_surffrag_central_res_num)] = [score_term]
                else:
                    central_residues_scores_d_pdb2[(pdb2_interface_surffrag_ch_id,
                                                    pdb2_interface_surffrag_central_res_num)].append(score_term)

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

                # if z_val <= 1:
                #     if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in
                #             unique_interface_monofrags_infolist_highqual_pdb1:
                #         unique_interface_monofrags_infolist_highqual_pdb1.append(
                #             (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))
                #     if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in
                #             unique_interface_monofrags_infolist_highqual_pdb2:
                #         unique_interface_monofrags_infolist_highqual_pdb2.append(
                #             (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))

                #######################################################

                if (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num) not in \
                        pdb1_unique_interface_frag_info_l:
                    pdb1_unique_interface_frag_info_l.append(
                        (pdb1_interface_surffrag_ch_id, pdb1_interface_surffrag_central_res_num))

                if (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num) not in \
                        pdb2_unique_interface_frag_info_l:
                    pdb2_unique_interface_frag_info_l.append(
                        (pdb2_interface_surffrag_ch_id, pdb2_interface_surffrag_central_res_num))
                #
                # frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val, cluster_id, pair_count,
                #                              interface_ghost_frag_cluster_res_freq_list,
                #                              interface_ghost_frag_cluster_rmsd))

    res_lev_sum_score, center_lev_sum_score = 0, 0
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
        for sc2 in central_res_scores_l_sorted2:
            center_lev_sum_score += sc2 * (1 / float(n2))
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
    number_residues_in_fragments = len(chid_resnum_scores_dict_pdb1) + len(chid_resnum_scores_dict_pdb2)
    number_fragment_central_residues = len(pdb1_unique_interface_frag_info_l) + len(pdb2_unique_interface_frag_info_l)
    if number_fragment_central_residues > 0:
        multiple_frag_ratio = (len(unique_fragment_indicies) * 2) / number_fragment_central_residues  # paired fragment
    else:
        multiple_frag_ratio = 0
    interface_residue_count = len(pdb1_unique_chain_central_res_l) + len(pdb2_unique_chain_central_res_l)
    if interface_residue_count > 0:
        percent_interface_matched = number_fragment_central_residues / float(interface_residue_count)
        percent_interface_covered = number_residues_in_fragments / float(interface_residue_count)
    else:
        percent_interface_matched, percent_interface_covered = 0, 0

    # Sum the total contribution from each fragment type on both sides of the interface
    fragment_content_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for index in fragment_i_index_count_d:
        fragment_content_d[index] += fragment_i_index_count_d[index]
        fragment_content_d[index] += fragment_j_index_count_d[index]

    if len(unique_fragment_indicies) > 0:
        for index in fragment_content_d:
            fragment_content_d[index] = fragment_content_d[index]/(len(unique_fragment_indicies) * 2)  # paired fragment

    # f_l1a = "Residue-Level Summation Score:" + str(res_lev_sum_score) + "\n"
    # f_l2 = "Unique Interface Fragment Match Count: " + str(number_fragment_central_residues) + "\n"
    # f_l3 = "Unique Interface Fragment Total Count: " + str(unique_interface_residue_count) + "\n"
    # f_l4 = "Percent of Interface Matched: " + str(percent_interface_matched) + "\n"

    return res_lev_sum_score, center_lev_sum_score, unique_fragment_indicies, number_residues_in_fragments, \
        number_fragment_central_residues, multiple_frag_ratio, interface_residue_count, percent_interface_matched,\
        percent_interface_covered, fragment_content_d


def calculate_interface_score(interface_pdb):
    interface_name = interface_pdb.name

    pdb1 = PDB(atoms=interface_pdb.get_chain_atoms(interface_pdb.chain_id_list[0]))
    pdb1.update_attributes_from_pdb(interface_pdb)
    pdb2 = PDB(atoms=interface_pdb.get_chain_atoms(interface_pdb.chain_id_list[-1]))
    pdb2.update_attributes_from_pdb(interface_pdb)

    pdb1_central_chainid_resnum_l, pdb2_central_chainid_resnum_l = get_interface_fragment_chain_residue_numbers(pdb1,
                                                                                                                pdb2)
    pdb1_interface_sa = pdb1.get_chain_residue_surface_area(pdb1_central_chainid_resnum_l, free_sasa_exe_path)
    pdb2_interface_sa = pdb2.get_chain_residue_surface_area(pdb2_central_chainid_resnum_l, free_sasa_exe_path)
    interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa
    res_level_sum_score, center_level_sum_score, fragment_indices, number_residues_with_fragments, \
        number_fragment_central_residues, multiple_frag_ratio, total_residues, percent_interface_matched, percent_interface_covered, \
        fragment_content_d = score_interface(pdb1, pdb2, pdb1_central_chainid_resnum_l, pdb2_central_chainid_resnum_l)

    interface_metrics = {'nanohedra_score': res_level_sum_score, 'nanohedra_score_central': center_level_sum_score,
                         'fragment_cluster_ids': ','.join(fragment_indices), 'interface_area': interface_buried_sa,
                         'multiple_fragment_ratio': multiple_frag_ratio,
                         'number_fragment_residues_all': number_residues_with_fragments,
                         'number_fragment_residues_central': number_fragment_central_residues,
                         'total_interface_residues': total_residues, 'number_fragments': len(fragment_indices),
                         'percent_residues_fragment_all': percent_interface_covered,
                         'percent_residues_fragment_center': percent_interface_matched,
                         'percent_fragment_helix': fragment_content_d['1'],
                         'percent_fragment_strand': fragment_content_d['2'],
                         'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
                         + fragment_content_d['5']}

    return interface_name, interface_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nScore selected interfaces using Nanohedra Score')
    # ---------------------------------------------------
    parser.add_argument('-d', '--directory', type=str, help='Directory where interface files are located. Default=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='A serialized dictionary with selected PDB code: [interface ID] '
                                                       'pairs', required=True)
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')

    args, additional_flags = parser.parse_known_args()
    # Program input
    print('USAGE: python ScoreNative.py interface_type_pickled_dict interface_filepath_location number_of_threads')

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name=os.path.basename(__file__), level=2)

    interface_pdbs = []
    if args.directory:
        interface_reference_d = unpickle(args.file)
        bio_reference_l = interface_reference_d['bio']

        print('Total of %d PDB\'s to score' % len(bio_reference_l))
        # try:
        #     print('1:', '1AB0-1' in bio_reference_l)
        #     print('2:', '1AB0-2' in bio_reference_l)
        #     print('no dash:', '1AB0' in bio_reference_l)
        # except KeyError as e:
        #     print(e)

        if args.debug:
            first_5 = ['2OPI-1', '4JVT-1', '3GZD-1', '4IRG-1', '2IN5-1']
            next_5 = ['3ILK-1', '3G64-1', '3G64-4', '3G64-6', '3G64-23']
            next_next_5 = ['3AQT-2', '2Q24-2', '1LDF-1', '1LDF-11', '1QCZ-1']
            paths = next_next_5
            root = '/home/kmeador/yeates/fragment_database/all/all_interfaces/%s.pdb'
            # paths = ['op/2OPI-1.pdb', 'jv/4JVT-1.pdb', 'gz/3GZD-1.pdb', 'ir/4IRG-1.pdb', 'in/2IN5-1.pdb']
            interface_filepaths = [root % '%s/%s' % (path[1:3].lower(), path) for path in paths]
            # interface_filepaths = list(map(os.path.join, root, paths))
        else:
            interface_filepaths = get_all_pdb_file_paths(args.directory)

        missing_index = [i for i, file_path in enumerate(interface_filepaths)
                         if os.path.splitext(os.path.basename(file_path))[0] not in bio_reference_l]

        for i in reversed(missing_index):
            del interface_filepaths[i]

        for interface_path in interface_filepaths:
            pdb = PDB(file=interface_path)
            # pdb = read_pdb(interface_path)
            pdb.name = os.path.splitext(os.path.basename(interface_path))[0]

    elif args.file:
        # pdb_codes = to_iterable(args.file)
        pdb_interface_d = unpickle(args.file)
        # for pdb_code in pdb_codes:
        #     for interface_id in pdb_codes[pdb_code]:
        interface_pdbs = [return_pdb_interface(pdb_code, interface_id) for pdb_code in pdb_interface_d
                          for interface_id in pdb_interface_d[pdb_code]]
        if args.output:
            out_path = args.output_dir
            pdb_code_id_tuples = [(pdb_code, interface_id) for pdb_code in pdb_interface_d
                                  for interface_id in pdb_interface_d[pdb_code]]
            for interface_pdb, pdb_code_id_tuple in zip(interface_pdbs, pdb_code_id_tuples):
                interface_pdb.write(os.path.join(args.output_dir, '%s-%d.pdb' % pdb_code_id_tuple))
    # else:
    #     logger.critical('Either --file or --directory must be specified')
    #     exit()

    if args.multi_processing:
        results = mp_map(calculate_interface_score, interface_pdbs, threads=int(sys.argv[3]))
        interface_d = {result for result in results}
        # interface_d = {key: result[key] for result in results for key in result}
    else:
        interface_d = {calculate_interface_score(interface_pdb) for interface_pdb in interface_pdbs}

    interface_df = pd.DataFrame(interface_d)
    interface_df.to_csv('BiologicalInterfaceNanohedraScores.csv')
