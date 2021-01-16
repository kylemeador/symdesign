import os
import pickle

import FragUtils as Frag
import numpy as np
from Bio.SeqUtils import IUPACData

from PDB import PDB

# from sklearn.neighbors import BallTree

# Globals
module = 'Make IJK Cluster Frequency Files:'


# def construct_cb_atom_tree(pdb1, pdb2, distance):
#     # Get CB Atom Coordinates
#     pdb1_coords = np.array(pdb1.extract_CB_coords())
#     pdb2_coords = np.array(pdb2.extract_CB_coords())
#
#     # Construct CB Tree for PDB1
#     pdb1_tree = BallTree(pdb1_coords)
#
#     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
#     query = pdb1_tree.query_radius(pdb2_coords, distance)
#
#     # Map Coordinates to Atoms
#     pdb1_cb_indices = pdb1.get_cb_indices()
#     pdb2_cb_indices = pdb2.get_cb_indices()
#
#     return query, pdb1_cb_indices, pdb2_cb_indices


# def collect_frag_weights(pdb, mapped_chain, paired_chain):
#     num_bb_atoms = 4
#     interact_distance = 5
#
#     # Creating PDB instance for mapped and paired chains
#     pdb_mapped = PDB()
#     pdb_paired = PDB()
#     pdb_mapped.read_atom_list(pdb.chain(mapped_chain))
#     pdb_paired.read_atom_list(pdb.chain(paired_chain))
#
#     # Query Atom Tree for all Ch2 Atoms within interaction_distance of Ch1 Atoms
#     query, pdb_map_cb_indices, pdb_partner_cb_indices = construct_cb_atom_tree(pdb_mapped, pdb_paired, interact_distance)
#
#     # Map Coordinates to Atoms
#     interacting_pairs = []
#     for patner_index in range(len(query)):
#         if query[patner_index].tolist() != list():
#             if not pdb_paired.atoms[patner_index].is_backbone():
#                 partner_atom_num = pdb_paired.atoms[patner_index].residue_number
#             else:
#                 # marks the atom number as backbone
#                 partner_atom_num = False
#             for mapped_index in query[patner_index]:
#                 if not pdb_mapped.atoms[mapped_index].is_backbone():
#                     map_atom_num = pdb_mapped.atoms[mapped_index].residue_number
#                 else:
#                     # marks the atom number as backbone
#                     map_atom_num = False
#                 interacting_pairs.append((map_atom_num, partner_atom_num))
#
#     # Create dictionary and Count all atoms in each residue sidechain
#     # ex. {'A': {32: (0, 9), 33: (0, 5), ...}, 'B':...}
#     res_counts_dict = {'mapped': {i.residue_number: [0, len(pdb_mapped.getResidueAtoms(mapped_chain, i.residue_number))
#                                                      - num_bb_atoms] for i in pdb_mapped.get_ca_atoms()},
#                        'paired': {i.residue_number: [0, len(pdb_paired.getResidueAtoms(paired_chain, i.residue_number))
#                                                      - num_bb_atoms] for i in pdb_paired.get_ca_atoms()}}
#     # Count all residue/residue interactions that do not originate from a backbone atom. In this way, side-chain to
#     # backbone are counted for the sidechain residue, indicating significance. However, backbones are (mostly)
#     # identical, and therefore, their interaction should be conserved in each member of the cluster and not counted
#     for res_pair in interacting_pairs:
#         if res_pair[0]:
#             res_counts_dict['map'][res_pair[0]][0] += 1
#         if res_pair[1]:
#             res_counts_dict['part'][res_pair[1]][0] += 1
#
#     # Add the value of the total residue involvement for single structure to overall cluster dictionary
#     for chain in [mapped_chain, paired_chain]:
#         total_pose_score = 0
#         for residue in res_counts_dict[chain]:
#             if res_counts_dict[chain][residue][1] == 0:
#                 res_counts_dict[chain][residue] = 0.0
#             else:
#                 res_normalized_score = res_counts_dict[chain][residue][0] / float(res_counts_dict[chain][residue][1])
#                 res_counts_dict[chain][residue] = res_normalized_score
#                 total_pose_score += res_normalized_score
#         if total_pose_score == 0:
#             # case where no atoms are within interaction distance
#             for residue in res_counts_dict[chain]:
#                 res_counts_dict[chain][residue] = 0.0
#             continue
#         for residue in res_counts_dict[chain]:
#             # Get percent of residue contribution to interaction over the entire pose interaction
#             res_counts_dict[chain][residue] /= total_pose_score
#             res_counts_dict[chain][residue] = round(res_counts_dict[chain][residue], 3)
#
#     return res_counts_dict
#
#
# def populate_aa_dictionary(low, up):
#     aa_dict = {i: {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
#                    'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
#                for i in range(low, up + 1)}
#     # 'stats' are total (stats[0]) and weight (stats[1])
#     # Weight starts as 1 to prevent removal during dictionary culling procedure
#     return aa_dict
#
#
# def freq_distribution(counts_dict, size):
#     # turn the dictionary into a frequency distribution dictionary
#     for residue in counts_dict:
#         remove = []
#         for aa in counts_dict[residue]:
#             if aa != 'stats':
#                 # remove residues with no representation
#                 if counts_dict[residue][aa] == 0:
#                     remove.append(aa)
#                 else:
#                     counts_dict[residue][aa] = round(counts_dict[residue][aa] / size, 3)
#         for null_aa in remove:
#             counts_dict[residue].pop(null_aa)
#
#     return counts_dict
#
#
# def guide_atom_rmsd(guide_atom_list_1, guide_atom_list_2):
#     # Calculate RMSD
#     sq_e1 = guide_atom_list_1[0].distance_squared(guide_atom_list_2[0], intra=True)
#     sq_e2 = guide_atom_list_1[1].distance_squared(guide_atom_list_2[1], intra=True)
#     sq_e3 = guide_atom_list_1[2].distance_squared(guide_atom_list_2[2], intra=True)
#     s = sq_e1 + sq_e2 + sq_e3
#     mean = s / float(3)
#     rmsd = math.sqrt(mean)
#
#     return rmsd
#
#
# def get_guide_atoms(frag_pdb):
#     guide_atoms = []
#     for atom in frag_pdb.atoms:
#         if atom.chain == "9":
#             guide_atoms.append(atom)
#     if len(guide_atoms) == 3:
#         return guide_atoms
#     else:
#         return None
#
#
# def parameterize_frag_length(length):
#     divide_2 = math.floor(length / 2)
#     modulus = length % 2
#     if modulus == 1:
#         upper_bound = 0 + divide_2
#         lower_bound = 0 - divide_2
#     else:
#         upper_bound = 0 + divide_2
#         lower_bound = upper_bound - length + 1
#     offset_to_one = -lower_bound + 1
#
#     return lower_bound, upper_bound, offset_to_one
#
#    threads = 6
# Frag5.main(module, db_dir, threads)
# for root, dirs1, files1 in os.walk(db_dir):
#     if not dirs1:
#         for file1 in files1:
#             if file1.endswith('all_to_all_guide_atom_rmsd.txt'):
#                 with open(file1, 'r') as f:
#                     rmsds = f.readlines()
#                 for rmsd in rmsds:


def main():
    print(module, 'Beginning')
    rmsd_thresh = 1
    fragment_length = 5
    lower_bound, upper_bound, index_offset = Frag.parameterize_frag_length(fragment_length)

    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    db_dir = os.path.join(outdir, 'db_' + str(rmsd_thresh))
    info_outdir = os.path.join(outdir, 'info_' + str(rmsd_thresh))
    if not os.path.exists(info_outdir):
        os.makedirs(info_outdir)

    for root, dirs1, files1 in os.walk(db_dir):
        if not dirs1:
            for file1 in files1:
                if file1.endswith('_representative.pdb'):
                    # Get Representative Guide Atoms
                    cluster_rep_path = os.path.join(root, file1)
                    cluster_rep_pdb = PDB()
                    cluster_rep_pdb.readfile(cluster_rep_path)
                    cluster_rep_guide_atoms = Frag.get_guide_atoms(cluster_rep_pdb)

                    ijk_name = os.path.basename(root)
                    i_type = ijk_name.split('_')[0]
                    j_type = ijk_name.split('_')[1]
                    k_type = ijk_name.split('_')[2]
                    ij_dir = os.path.join(i_type, i_type + '_' + j_type)
                    ijk_dir = os.path.join(ij_dir, i_type + '_' + j_type + '_' + k_type)

                    if not os.path.exists(os.path.join(info_outdir, i_type)):
                        os.makedirs(os.path.join(info_outdir, i_type))
                    if not os.path.exists(os.path.join(info_outdir, ij_dir)):
                        os.makedirs(os.path.join(info_outdir, ij_dir))
                    if not os.path.exists(os.path.join(info_outdir, ijk_dir)):
                        os.makedirs(os.path.join(info_outdir, ijk_dir))
                    ijk_out_dir = os.path.join(info_outdir, ijk_dir)
                    cluster_dir = os.path.join(db_dir, ijk_dir)

                    cluster_count = 0
                    rmsd_sum = 0
                    fragment_residue_counts = []
                    total_cluster_weight = {}
                    for file2 in os.listdir(cluster_dir):
                        if file2.endswith('.pdb'):
                            cluster_member_path = os.path.join(cluster_dir, file2)
                            member_pdb = PDB()
                            member_pdb.readfile(cluster_member_path)
                            member_guide_atoms = Frag.get_guide_atoms(member_pdb)

                            member_mapped_ch = file2[file2.find('mappedchain') + 12:file2.find('mappedchain') + 13]
                            member_paired_ch = file2[file2.find('partnerchain') + 13:file2.find('partnerchain') + 14]

                            # Get Residue Counts for each Fragment in the Cluster
                            residue_frequency = np.empty((fragment_length, 2), dtype=object)
                            mapped_chain_res_count = 0
                            paired_chain_res_count = 0
                            for atom in member_pdb.get_atoms():
                                if atom.is_CA() and atom.chain == member_mapped_ch:
                                    residue_frequency[mapped_chain_res_count][0] = \
                                        IUPACData.protein_letters_3to1[atom.residue_type.title()] if \
                                            atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                    mapped_chain_res_count += 1
                                elif atom.is_CA() and atom.chain == member_paired_ch:
                                    residue_frequency[paired_chain_res_count][1] = \
                                        IUPACData.protein_letters_3to1[atom.residue_type.title()] if \
                                            atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                    paired_chain_res_count += 1

                            if np.all(residue_frequency):
                                fragment_residue_counts.append(residue_frequency)  # type is np_array(fragment_index, 2)
                                total_cluster_weight[file2] = Frag.collect_frag_weights(member_pdb, member_mapped_ch,
                                                                                        member_paired_ch)
                                rmsd = Frag.guide_atom_rmsd(cluster_rep_guide_atoms, member_guide_atoms)
                                rmsd_sum += rmsd
                                cluster_count += 1

                    if cluster_count > 0:
                        mean_cluster_rmsd = rmsd_sum / float(cluster_count)
                        mapped_counts_dict = Frag.populate_aa_dictionary(lower_bound, upper_bound)
                        partner_counts_dict = Frag.populate_aa_dictionary(lower_bound, upper_bound)

                        for array in fragment_residue_counts:
                            residue = lower_bound
                            for i in array:
                                mapped_counts_dict[residue][str(i[0])] += 1
                                partner_counts_dict[residue][str(i[1])] += 1
                                residue += 1
                        for residue in range(lower_bound, upper_bound + 1):
                            mapped_counts_dict[residue]['stats'][0] = cluster_count
                            partner_counts_dict[residue]['stats'][0] = cluster_count

                        # Make Frequency Distribution Dictionaries and Remove Unrepresented AA's
                        mapped_freq_dict = Frag.freq_distribution(mapped_counts_dict, cluster_count)
                        partner_freq_dict = Frag.freq_distribution(partner_counts_dict, cluster_count)

                        # Sum total cluster Residue Weights
                        final_weights = np.zeros((2, fragment_length))
                        for pdb in total_cluster_weight:
                            for chain in total_cluster_weight[pdb]:
                                if chain == 'mapped':
                                    i = 0
                                else:
                                    i = 1
                                n = 0
                                for residue in total_cluster_weight[pdb][chain]:
                                    final_weights[i][n] += total_cluster_weight[pdb][chain][residue]
                                    n += 1

                        # Normalize by cluster size
                        with np.nditer(final_weights, op_flags=['readwrite']) as it:
                            for element in it:
                                element /= cluster_count
                        # Make weights into a percentage
                        for i in range(len(final_weights)):
                            s = 0
                            for n in range(len(final_weights[i])):
                                s += final_weights[i][n]
                            if s == 0:
                                for n in range(len(final_weights[i])):
                                    final_weights[i][n] = 0.0
                            else:
                                for n in range(len(final_weights[i])):
                                    final_weights[i][n] /= s

                        # Add Residue Weights to respective dictionary
                        i = 0
                        for residue in mapped_freq_dict:
                            mapped_freq_dict[residue]['stats'][1] = round(final_weights[0][i], 3)
                            i += 1
                        j = 0
                        for residue in partner_freq_dict:
                            partner_freq_dict[residue]['stats'][1] = round(final_weights[1][j], 3)
                            j += 1

                        # Save Full Cluster Dictionary as Binary Dictionary
                        full_dictionary = {'size': cluster_count, 'rmsd': mean_cluster_rmsd, 'rep': str(file1),
                                           'mapped': mapped_freq_dict, 'paired': partner_freq_dict}
                        with open(os.path.join(ijk_out_dir, ijk_name + '.pkl'), 'wb') as f:
                            pickle.dump(full_dictionary, f, pickle.HIGHEST_PROTOCOL)

                        # Save Text File
                        out_l1 = 'Size: %s\n' % str(cluster_count)
                        out_l2 = 'RMSD: %s\n' % str(mean_cluster_rmsd)
                        out_l3 = 'Representative Name: %s' % str(file1)
                        l4 = []
                        for fragment_index in mapped_freq_dict:
                            counts = mapped_freq_dict[fragment_index].items()
                            line0 = '\nResidue %s Weight: ' % str(fragment_index + index_offset)
                            l4.append(line0)
                            line1 = '\nFrequency: '
                            l4.append(line1)
                            for pair in counts:
                                line2 = str(pair) + ' '
                                l4.append(line2)
                        l5 = []
                        for fragment_index in partner_freq_dict:
                            counts = partner_freq_dict[fragment_index].items()
                            line0 = '\nResidue %s Weight: ' % str(fragment_index + index_offset)
                            l5.append(line0)
                            line1 = '\nFrequency: '
                            l5.append(line1)
                            for pair in counts:
                                line2 = str(pair) + ' '
                                l5.append(line2)

                        with open(os.path.join(ijk_out_dir, ijk_name + '.txt'), 'w') as outfile:
                            outfile.write(out_l1)
                            outfile.write(out_l2)
                            outfile.write(out_l3)
                            for out_l4 in l4:
                                outfile.write(out_l4)
                            for out_l5 in l5:
                                outfile.write(out_l5)

    print(module, 'Finished')


if __name__ == '__main__':
    main()
