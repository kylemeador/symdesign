import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from FragDock import filter_euler_lookup_by_zvalue, calculate_overlap
from PDB import PDB
from SymDesignUtils import start_log, unpickle, get_all_pdb_file_paths, mp_map
# from symdesign.interface_analysis.InterfaceSorting import return_pdb_interface
from classes import EulerLookup
from classes.Fragment import FragmentDB, MonoFragment

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
    """Given two PDBs, return the unique chain and interacting residue lists"""
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


def get_fragments(pdb, chain_res_info, fragment_length=5):
    interface_frags = []
    for residue_number in chain_res_info:
        frag_residue_numbers = [residue_number + i for i in range(-2, 3)]
        frag_atoms, ca_present = [], []
        for residue in pdb.residue(frag_residue_numbers):
            frag_atoms.extend(residue.get_atoms())
            ca_present.append(residue.get_ca())

        if all(ca_present):
            interface_frags.append(PDB.from_atoms(frag_atoms))

    return interface_frags


def find_fragments_from_interface(entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers,
                                  max_z_val=2):
    """From a Structure Entity, score the interface between them according to Nanohedra's fragment matching"""
    kdtree_oligomer1_backbone = BallTree(np.array(entity1.extract_backbone_coords()))
    # Get pdb1 interface fragments with guide coordinates using fragment database
    interface_frags1 = get_fragments(entity1, entity1_interface_residue_numbers)
    # entity1_interface_fragment_residue_numbers = [fragment.residues[2].number for fragment in interface_frags1]

    complete_int1_ghost_frag_l, interface_ghostfrag_guide_coords_list = [], []
    for frag1 in interface_frags1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone, ijk_intfrag_cluster_info_dict)
        if complete_monofrag1_ghostfrag_list:
            complete_int1_ghost_frag_l.extend(complete_monofrag1_ghostfrag_list)
            for ghost_frag in complete_monofrag1_ghostfrag_list:
                interface_ghostfrag_guide_coords_list.append(ghost_frag.get_guide_coords())

    # Get pdb2 interface fragments with guide coordinates using complete fragment database
    interface_frags2 = get_fragments(entity2, entity2_interface_residue_numbers)
    # entity2_interface_fragment_residue_numbers = [fragment.residues[2].number for fragment in interface_frags2]

    complete_int2_frag_l, interface_surf_frag_guide_coords_list = [], []
    for frag2 in interface_frags2:
        complete_monofrag2 = MonoFragment(frag2, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag2_guide_coords = complete_monofrag2.get_guide_coords()
        if complete_monofrag2_guide_coords:
            complete_int2_frag_l.append(complete_monofrag2)
            interface_surf_frag_guide_coords_list.append(complete_monofrag2_guide_coords)
            # complete_surf_frag_guide_coord_l.append(complete_monofrag2_guide_coords)

    # del ijk_monofrag_cluster_rep_pdb_dict, init_monofrag_cluster_rep_pdb_dict_1, init_monofrag_cluster_rep_pdb_dict_2

    # Check for matching Euler angles
    eul_lookup_all_to_all_list = eul_lookup.check_lookup_table(interface_ghostfrag_guide_coords_list,
                                                               interface_surf_frag_guide_coords_list)
    eul_lookup_true_list = [(true_tup[0], true_tup[1]) for true_tup in eul_lookup_all_to_all_list if true_tup[2]]

    all_fragment_overlap = filter_euler_lookup_by_zvalue(eul_lookup_true_list, complete_int1_ghost_frag_l,
                                                         interface_ghostfrag_guide_coords_list,
                                                         complete_int2_frag_l, interface_surf_frag_guide_coords_list,
                                                         z_value_func=calculate_overlap)
    passing_fragment_overlap = list(filter(None, all_fragment_overlap))
    ghostfrag_surffrag_pairs = [(complete_int1_ghost_frag_l[eul_lookup_true_list[idx][0]],
                                 complete_int2_frag_l[eul_lookup_true_list[idx][1]])
                                for idx, boolean in enumerate(all_fragment_overlap) if boolean]
    fragment_matches = []
    for frag_idx, (interface_ghost_frag, interface_mono_frag) in enumerate(ghostfrag_surffrag_pairs):
        ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
        ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
        ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
        cluster_id = "%s_%s_%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)

        entity1_surffrag_ch, entity1_surffrag_resnum = interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
        entity2_surffrag_ch, entity2_surffrag_resnum = interface_mono_frag.get_central_res_tup()
        score_term = passing_fragment_overlap[frag_idx][0]
        fragment_matches.append({'mapped': entity1_surffrag_resnum, 'match_score': score_term,
                                 'paired': entity2_surffrag_resnum, 'culster': cluster_id})

        # interface_residues_with_fragment_overlap['mapped'].add(entity1_surffrag_resnum)
        # interface_residues_with_fragment_overlap['paired'].add(entity2_surffrag_resnum)
        # covered_residues_pdb1 = [(entity1_surffrag_resnum + j) for j in range(-2, 3)]
        # covered_residues_pdb2 = [(entity2_surffrag_resnum + j) for j in range(-2, 3)]
        # for k in range(5):
        #     resnum1 = covered_residues_pdb1[k]
        #     resnum2 = covered_residues_pdb2[k]
        #     if resnum1 not in entity1_match_scores:
        #         entity1_match_scores[resnum1] = [score_term]
        #     else:
        #         entity1_match_scores[resnum1].append(score_term)
        #
        #     if resnum2 not in entity2_match_scores:
        #         entity2_match_scores[resnum2] = [score_term]
        #     else:
        #         entity2_match_scores[resnum2].append(score_term)

        # if entity1_surffrag_resnum not in unique_interface_monofrags_infolist_pdb1:
        #     unique_interface_monofrags_infolist_pdb1.append(entity1_surffrag_resnum)
        #
        # if entity2_surffrag_resnum not in unique_interface_monofrags_infolist_pdb2:
        #     unique_interface_monofrags_infolist_pdb2.append(entity2_surffrag_resnum)

        # z_val = passing_fragment_overlap[frag_idx][1]
        # fragment_source
        # [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]

    # unique_matched_interface_monofrag_count = len(unique_interface_monofrags_infolist_pdb1) + len(
    #     unique_interface_monofrags_infolist_pdb2)
    # percent_of_interface_covered = unique_matched_interface_monofrag_count / float(
    #     unique_total_interface_monofrags_count)

    # # Get RMSD and z-value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
    # pdb1_unique_interface_frag_info_l = []
    # pdb2_unique_interface_frag_info_l = []
    # unique_fragment_indicies = []
    # total_overlap_count = 0
    # # central_residues_scores_d_pdb1, central_residues_scores_d_pdb2 = {}, {}
    # for index_pair in eul_lookup_true_list:
    #     interface_ghost_frag = complete_int1_ghost_frag_l[index_pair[0]]
    #     interface_ghost_frag_guide_coords = interface_ghostfrag_guide_coords_list[index_pair[0]]
    #     ghost_frag_i_type = interface_ghost_frag.get_i_frag_type()
    #     ghost_frag_j_type = interface_ghost_frag.get_j_frag_type()
    #     ghost_frag_k_type = interface_ghost_frag.get_k_frag_type()
    #     cluster_id = "i%s_j%s_k%s" % (ghost_frag_i_type, ghost_frag_j_type, ghost_frag_k_type)
    #     interface_ghost_frag_cluster_rmsd = ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
    #         ghost_frag_k_type].get_rmsd()
    #     # interface_ghost_frag_cluster_res_freq_list = \
    #     #     ijk_intfrag_cluster_info_dict[ghost_frag_i_type][ghost_frag_j_type][
    #     #         ghost_frag_k_type].get_central_residue_pair_freqs()
    #
    #     interface_mono_frag = complete_int2_frag_l[index_pair[1]]
    #     interface_mono_frag_guide_coords = interface_surf_frag_guide_coords_list[index_pair[1]]
    #     interface_mono_frag_type = interface_mono_frag.get_type()
    #
    #     if (interface_mono_frag_type == ghost_frag_j_type) and (interface_ghost_frag_cluster_rmsd > 0):
    #         # Calculate RMSD
    #         total_overlap_count += 1
    #         e1 = euclidean_squared_3d(interface_mono_frag_guide_coords[0], interface_ghost_frag_guide_coords[0])
    #         e2 = euclidean_squared_3d(interface_mono_frag_guide_coords[1], interface_ghost_frag_guide_coords[1])
    #         e3 = euclidean_squared_3d(interface_mono_frag_guide_coords[2], interface_ghost_frag_guide_coords[2])
    #         s = e1 + e2 + e3
    #         mean = s / float(3)
    #         rmsd = math.sqrt(mean)
    #
    #         # Get Guide Atom Overlap Z-Value
    #         z_val = rmsd / float(interface_ghost_frag_cluster_rmsd)
    #
    #         if z_val <= max_z_val:
    #             fragment_i_index_count_d[ghost_frag_i_type] += 1
    #             fragment_j_index_count_d[ghost_frag_j_type] += 1
    #             unique_fragment_indicies.append(cluster_id)
    #             # if z_val == 0:
    #             #     inv_z_val = 3
    #             # elif 1 / float(z_val) > 3:
    #             #     inv_z_val = 3
    #             # else:
    #             #     inv_z_val = 1 / float(z_val)
    #             #
    #             # total_inv_capped_z_val_score += inv_z_val
    #             # pair_count += 1
    #
    #             # ghostfrag_mapped_ch_id, ghostfrag_mapped_central_res_num, ghostfrag_partner_ch_id, \
    #             # ghostfrag_partner_central_res_num = interface_ghost_frag.get_central_res_tup()
    #             entity1_surffrag_ch, entity1_surffrag_resnum = \
    #                 interface_ghost_frag.get_aligned_surf_frag_central_res_tup()
    #             entity2_surffrag_ch, entity2_surffrag_resnum = \
    #                 interface_mono_frag.get_central_res_tup()
    #
    #             score_term = 1 / float(1 + (z_val ** 2))
    #
    #             # # Central residue only score
    #             # if (entity1_surffrag_ch, entity1_surffrag_resnum) not in \
    #             #         central_residues_scores_d_pdb1:
    #             #     central_residues_scores_d_pdb1[(entity1_surffrag_ch,
    #             #                                     entity1_surffrag_resnum)] = [score_term]
    #             # else:
    #             #     central_residues_scores_d_pdb1[
    #             #         (entity1_surffrag_ch, entity1_surffrag_resnum)].append(score_term)
    #             #
    #             # if (entity2_surffrag_ch, entity2_surffrag_resnum) not in \
    #             #         central_residues_scores_d_pdb2:
    #             #     central_residues_scores_d_pdb2[(entity2_surffrag_ch,
    #             #                                     entity2_surffrag_resnum)] = [score_term]
    #             # else:
    #             #     central_residues_scores_d_pdb2[(entity2_surffrag_ch,
    #             #                                     entity2_surffrag_resnum)].append(score_term)
    #             #
    #             # covered_residues_pdb1 = [(entity1_surffrag_ch, entity1_surffrag_resnum + j)
    #             #                          for j in range(-2, 3)]
    #             # covered_residues_pdb2 = [(entity2_surffrag_ch, entity2_surffrag_resnum + j)
    #             #                          for j in range(-2, 3)]
    #             # for k in range(5):
    #             #     chid1, resnum1 = covered_residues_pdb1[k]
    #             #     chid2, resnum2 = covered_residues_pdb2[k]
    #             #     if (chid1, resnum1) not in entity1_match_scores:
    #             #         entity1_match_scores[(chid1, resnum1)] = [score_term]
    #             #     else:
    #             #         entity1_match_scores[(chid1, resnum1)].append(score_term)
    #             #
    #             #     if (chid2, resnum2) not in entity2_match_scores:
    #             #         entity2_match_scores[(chid2, resnum2)] = [score_term]
    #             #     else:
    #             #         entity2_match_scores[(chid2, resnum2)].append(score_term)
    #
    #             # if z_val <= 1:
    #             #     if (entity1_surffrag_ch, entity1_surffrag_resnum) not in
    #             #             unique_interface_monofrags_infolist_highqual_pdb1:
    #             #         unique_interface_monofrags_infolist_highqual_pdb1.append(
    #             #             (entity1_surffrag_ch, entity1_surffrag_resnum))
    #             #     if (entity2_surffrag_ch, entity2_surffrag_resnum) not in
    #             #             unique_interface_monofrags_infolist_highqual_pdb2:
    #             #         unique_interface_monofrags_infolist_highqual_pdb2.append(
    #             #             (entity2_surffrag_ch, entity2_surffrag_resnum))
    #
    #             #######################################################
    #
    #             if (entity1_surffrag_ch, entity1_surffrag_resnum) not in \
    #                     pdb1_unique_interface_frag_info_l:
    #                 pdb1_unique_interface_frag_info_l.append(
    #                     (entity1_surffrag_ch, entity1_surffrag_resnum))
    #
    #             if (entity2_surffrag_ch, entity2_surffrag_resnum) not in \
    #                     pdb2_unique_interface_frag_info_l:
    #                 pdb2_unique_interface_frag_info_l.append(
    #                     (entity2_surffrag_ch, entity2_surffrag_resnum))
    #             #
    #             # frag_match_info_list.append((interface_ghost_frag, interface_mono_frag, z_val, cluster_id, pair_count,
    #             #                              interface_ghost_frag_cluster_res_freq_list,
    #             #                              interface_ghost_frag_cluster_rmsd))

    # # Center only
    # for central_res_scores_l1 in central_residues_scores_d_pdb1.values():
    #     n1 = 1
    #     central_res_scores_l_sorted1 = sorted(central_res_scores_l1, reverse=True)
    #     for sc1 in central_res_scores_l_sorted1:
    #         center_residue_score += sc1 * (1 / float(n1))
    #         n1 *= 2
    # for central_res_scores_l2 in central_residues_scores_d_pdb2.values():
    #     n2 = 1
    #     central_res_scores_l_sorted2 = sorted(central_res_scores_l2, reverse=True)
    #     for sc2 in central_res_scores_l_sorted2:
    #         center_residue_score += sc2 * (1 / float(n2))
    #         n2 *= 2

    # # Generate Nanohedra score for center and all residues
    # all_residue_score, center_residue_score = 0, 0
    # for residue_number, res_scores in entity1_match_scores.items():
    #     n = 1
    #     res_scores_sorted = sorted(res_scores, reverse=True)
    #     if residue_number in entity1_interface_fragment_residue_numbers:  # entity1_interface_residue_numbers: <- may be at termini
    #         for central_score in res_scores_sorted:
    #             center_residue_score += central_score * (1 / float(n))
    #             n *= 2
    #     else:
    #         for peripheral_score in res_scores_sorted:
    #             all_residue_score += peripheral_score * (1 / float(n))
    #             n *= 2
    #
    # # doing this twice seems unnecessary as there is no new fragment information, but residue observations are
    # # weighted by n, number of observations which differs between entities across the interface
    # for residue_number, res_scores in entity2_match_scores.items():
    #     n = 1
    #     res_scores_sorted = sorted(res_scores, reverse=True)
    #     if residue_number in entity2_interface_fragment_residue_numbers:  # entity2_interface_residue_numbers: <- may be at termini
    #         for central_score in res_scores_sorted:
    #             center_residue_score += central_score * (1 / float(n))
    #             n *= 2
    #     else:
    #         for peripheral_score in res_scores_sorted:
    #             all_residue_score += peripheral_score * (1 / float(n))
    #             n *= 2
    #
    # all_residue_score += center_residue_score

    # Metric calculation
    # unique_fragments = len(central_residues_scores_d_pdb1) + len(central_residues_scores_d_pdb2)
    # # Get the number of central residues with overlapping fragments identified given z_value criteria
    # entity1_match_scores
    # interface_residues_with_fragment_overlap
    #
    # number_unique_residues_with_fragment_obs = len(interface_residues_with_fragment_overlap['mapped']) + \
    #     len(interface_residues_with_fragment_overlap['paired'])
    # # number_unique_residues_with_fragment_obs = len(pdb1_unique_interface_frag_info_l) + len(pdb2_unique_interface_frag_info_l)
    #
    # # Get the number of residues with fragments overlapping given z_value criteria
    # number_residues_in_fragments = len(entity1_match_scores) + len(entity2_match_scores)
    #
    # if number_unique_residues_with_fragment_obs > 0:
    #     multiple_frag_ratio = (len(fragment_matches) * 2) / number_unique_residues_with_fragment_obs  # paired fragment
    # else:
    #     multiple_frag_ratio = 0
    #
    # interface_residue_numbers  # <- input to function
    # interface_residue_count = len(entity1_interface_residue_numbers) + len(entity2_interface_residue_numbers)
    # if interface_residue_count > 0:
    #     percent_interface_matched = number_unique_residues_with_fragment_obs / float(interface_residue_count)
    #     percent_interface_covered = number_residues_in_fragments / float(interface_residue_count)
    # else:
    #     percent_interface_matched, percent_interface_covered = 0, 0
    #
    # # Sum the total contribution from each fragment type on both sides of the interface
    # fragment_content_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    # for index in fragment_i_index_count_d:
    #     fragment_content_d[index] += fragment_i_index_count_d[index]
    #     fragment_content_d[index] += fragment_j_index_count_d[index]
    #
    # if len(fragment_matches) > 0:
    #     for index in fragment_content_d:
    #         fragment_content_d[index] = fragment_content_d[index]/(len(fragment_matches) * 2)  # paired fragment

    return fragment_matches


def get_fragment_metrics(fragment_matches):
    # fragment_matches = [{'mapped': entity1_surffrag_resnum, 'match_score': score_term,
    #                          'paired': entity2_surffrag_resnum, 'culster': cluster_id}, ...]
    fragment_i_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    fragment_j_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    entity1_match_scores, entity2_match_scores = {}, {}
    interface_residues_with_fragment_overlap = {'mapped': set(), 'paired': set()}
    for fragment in fragment_matches:

        interface_residues_with_fragment_overlap['mapped'].add(fragment['mapped'])
        interface_residues_with_fragment_overlap['paired'].add(fragment['paired'])
        covered_residues_pdb1 = [(fragment['mapped'] + j) for j in range(-2, 3)]
        covered_residues_pdb2 = [(fragment['paired'] + j) for j in range(-2, 3)]
        for k in range(5):
            resnum1 = covered_residues_pdb1[k]
            resnum2 = covered_residues_pdb2[k]
            if resnum1 not in entity1_match_scores:
                entity1_match_scores[resnum1] = [fragment['match_score']]
            else:
                entity1_match_scores[resnum1].append(fragment['match_score'])

            if resnum2 not in entity2_match_scores:
                entity2_match_scores[resnum2] = [fragment['match_score']]
            else:
                entity2_match_scores[resnum2].append(fragment['match_score'])

        fragment_i_index_count_d[fragment['cluster'].split('_')[0]] += 1
        fragment_j_index_count_d[fragment['cluster'].split('_')[0]] += 1

    # Generate Nanohedra score for center and all residues
    all_residue_score, center_residue_score = 0, 0
    for residue_number, res_scores in entity1_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['mapped']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    # doing this twice seems unnecessary as there is no new fragment information, but residue observations are
    # weighted by n, number of observations which differs between entities across the interface
    for residue_number, res_scores in entity2_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['paired']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    all_residue_score += center_residue_score

    # Get the number of central residues with overlapping fragments identified given z_value criteria
    number_unique_residues_with_fragment_obs = len(interface_residues_with_fragment_overlap['mapped']) + \
        len(interface_residues_with_fragment_overlap['paired'])

    # Get the number of residues with fragments overlapping given z_value criteria
    number_residues_in_fragments = len(entity1_match_scores) + len(entity2_match_scores)

    if number_unique_residues_with_fragment_obs > 0:
        multiple_frag_ratio = (len(fragment_matches) * 2) / number_unique_residues_with_fragment_obs  # paired fragment
    else:
        multiple_frag_ratio = 0

    interface_residue_count = len(interface_residues_with_fragment_overlap['mapped']) + len(interface_residues_with_fragment_overlap['paired'])
    if interface_residue_count > 0:
        percent_interface_matched = number_unique_residues_with_fragment_obs / float(interface_residue_count)
        percent_interface_covered = number_residues_in_fragments / float(interface_residue_count)
    else:
        percent_interface_matched, percent_interface_covered = 0, 0

    # Sum the total contribution from each fragment type on both sides of the interface
    fragment_content_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for index in fragment_i_index_count_d:
        fragment_content_d[index] += fragment_i_index_count_d[index]
        fragment_content_d[index] += fragment_j_index_count_d[index]

    if len(fragment_matches) > 0:
        for index in fragment_content_d:
            fragment_content_d[index] = fragment_content_d[index] / (len(fragment_matches) * 2)  # paired fragment

    return all_residue_score, center_residue_score, number_residues_in_fragments, \
        number_unique_residues_with_fragment_obs, multiple_frag_ratio, interface_residue_count, \
        percent_interface_matched, percent_interface_covered, fragment_content_d


def calculate_interface_score(interface_pdb):
    interface_name = interface_pdb.name

    pdb1 = PDB(atoms=interface_pdb.chain(interface_pdb.chain_id_list[0]).get_atoms())
    pdb1.update_attributes_from_pdb(interface_pdb)
    pdb2 = PDB(atoms=interface_pdb.chain(interface_pdb.chain_id_list[-1]).get_atoms())
    pdb2.update_attributes_from_pdb(interface_pdb)

    pdb1_central_chainid_resnum_l, pdb2_central_chainid_resnum_l = get_interface_fragment_chain_residue_numbers(pdb1,
                                                                                                                pdb2)
    pdb1_interface_sa = pdb1.get_chain_residue_surface_area(pdb1_central_chainid_resnum_l, free_sasa_exe_path)
    pdb2_interface_sa = pdb2.get_chain_residue_surface_area(pdb2_central_chainid_resnum_l, free_sasa_exe_path)
    interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa

    fragment_matches = find_fragments_from_interface(pdb1, pdb2, pdb1_central_chainid_resnum_l,
                                                     pdb2_central_chainid_resnum_l)

    res_level_sum_score, center_level_sum_score, number_residues_with_fragments, number_fragment_central_residues, \
        multiple_frag_ratio, total_residues, percent_interface_matched, percent_interface_covered, \
        fragment_content_d = get_fragment_metrics(fragment_matches)

    interface_metrics = {'nanohedra_score': res_level_sum_score, 'nanohedra_score_central': center_level_sum_score,
                         'fragments': fragment_matches,
                         # 'fragment_cluster_ids': ','.join(fragment_indices),
                         'interface_area': interface_buried_sa,
                         'multiple_fragment_ratio': multiple_frag_ratio,
                         'number_fragment_residues_all': number_residues_with_fragments,
                         'number_fragment_residues_central': number_fragment_central_residues,
                         'total_interface_residues': total_residues, 'number_fragments': len(fragment_matches),
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
