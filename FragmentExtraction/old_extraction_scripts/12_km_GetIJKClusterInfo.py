import math
import os
#import PDB
import numpy as np
from PDB import PDB
from Bio.SeqUtils import IUPACData
from collections import Counter


def populate_aa_dictionary(length):
    aa_dict = {i: {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                   'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, '-': 0, 'X': 0}
               for i in range(length)}

    return aa_dict


def freq_distribution(counts_dict, size):
    # turn the dictionary into a frequency distribution dictionary
    for residue in counts_dict:
        for aa in residue:
            # remove residues with no representation
            if counts_dict[residue][aa] == 0:
                counts_dict[residue].pop(aa)
            else:
                counts_dict[residue][aa] = round(counts_dict[residue][aa] / size, 3)

    return aa_dict


def guide_atom_rmsd(guide_atom_list_1, guide_atom_list_2):

    # Calculate RMSD
    sq_e1 = guide_atom_list_1[0].distance_squared(guide_atom_list_2[0], intra=True)
    sq_e2 = guide_atom_list_1[1].distance_squared(guide_atom_list_2[1], intra=True)
    sq_e3 = guide_atom_list_1[2].distance_squared(guide_atom_list_2[2], intra=True)
    sum = sq_e1 + sq_e2 + sq_e3
    mean = sum / float(3)
    rmsd = math.sqrt(mean)

    return rmsd


def get_guide_atoms(frag_pdb):
    guide_atoms = []
    for atom in frag_pdb.all_atoms:
        if atom.chain == "9":
            guide_atoms.append(atom)
    if len(guide_atoms) == 3:
        return guide_atoms
    else:
        return None


def main():
    ijk_intfrag_cluster_rep_dir = "/home/kmeador/yeates/interface_extraction/ijk_cluster_reps"
    ijk_clustered_int_frag_db_dir = "/home/kmeador/yeates/interface_extraction/ijk_clustered_xtal_fragDB_1A"

    out_dir = "/home/kmeador/yeates/interface_extraction/ijk_clustered_xtal_fragDB_1A_info"
    fragment_length = 5
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for dirpath1, dirnames1, filenames1 in os.walk(ijk_intfrag_cluster_rep_dir):
        if not dirnames1:
            for fname1 in filenames1:
                if fname1.endswith(".pdb"):

                    ijk_cluster_name = dirpath1.split("/")[-1]
                    cluster_member_count = 0
                    rmsd_sum = 0
                    central_res_pairs = []

                    i_cluster_type = ijk_cluster_name.split("_")[0]
                    j_cluster_type = ijk_cluster_name.split("_")[1]
                    k_cluster_type = ijk_cluster_name.split("_")[2]

                    cluster_dir = ijk_clustered_int_frag_db_dir + "/" + i_cluster_type + "/" + i_cluster_type + "_" + j_cluster_type + "/" + i_cluster_type + "_" + j_cluster_type + "_" + k_cluster_type

                    i_dir = out_dir + "/" + i_cluster_type
                    if not os.path.exists(i_dir):
                        os.makedirs(i_dir)

                    ij_dir = out_dir + "/" + i_cluster_type + "/" + i_cluster_type + "_" + j_cluster_type
                    if not os.path.exists(ij_dir):
                        os.makedirs(ij_dir)

                    ijk_dir = out_dir + "/" + i_cluster_type + "/" + i_cluster_type + "_" + j_cluster_type + "/" + i_cluster_type + "_" + j_cluster_type + "_" + k_cluster_type
                    if not os.path.exists(ijk_dir):
                        os.makedirs(ijk_dir)

                    cluster_rep = filenames1[0]
                    cluster_rep_path = dirpath1 + "/" + cluster_rep
                    cluster_rep_pdb = PDB()
                    cluster_rep_pdb.readfile(cluster_rep_path)
                    cluster_rep_guide_atoms = get_guide_atoms(cluster_rep_pdb)

                    for dirpath2, dirnames2, filenames2 in os.walk(cluster_dir):
                        for fname2 in filenames2:
                            if fname2.endswith(".pdb"):
                                cluster_member_path = cluster_dir + "/" + fname2
                                cluster_member_pdb = PDB()
                                cluster_member_pdb.readfile(cluster_member_path)
                                cluster_member_guide_atoms = get_guide_atoms(cluster_member_pdb)

                                cluster_member_mapped_chain_id = fname2[fname2.find("mappedchain") + 12:fname2.find("mappedchain") + 13]
                                cluster_member_partner_chain_id = fname2[fname2.find("partnerchain") + 13:fname2.find("partnerchain") + 14]
                                residue_frequency = np.empty((2, 5))
                                #residue_frequency = np.full((2, 5), None)
                                # cluster_member_mapped_central_res_code = None
                                # cluster_member_partner_central_res_code = None
                                # cluster_member_mapped_iminus_res_code = None
                                # cluster_member_partner_iminus_res_code = None
                                # cluster_member_mapped_iplus_res_code = None
                                # cluster_member_partner_iplus_res_code = None
                                mapped_chain_res_count = 0
                                partner_chain_res_count = 0
                                for atom in cluster_member_pdb.all_atoms:
                                    if atom.is_CA() and atom.chain == cluster_member_mapped_chain_id:
                                        residue_frequency[0][mapped_chain_res_count] = IUPACData.protein_letters_3to1[atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        mapped_chain_res_count += 1

                                        # if mapped_chain_res_count == 2:
                                        #     cluster_member_mapped_iminus_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        # elif mapped_chain_res_count == 3:
                                        #     cluster_member_mapped_central_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        # elif mapped_chain_res_count == 4:
                                        #     cluster_member_mapped_iplus_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                    elif atom.is_CA() and atom.chain == cluster_member_partner_chain_id:
                                        residue_frequency[1][partner_chain_res_count] = IUPACData.protein_letters_3to1[atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        partner_chain_res_count += 1

                                        # if partner_chain_res_count == 2:
                                        #     cluster_member_partner_iminus_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        # elif partner_chain_res_count == 3:
                                        #     cluster_member_partner_central_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                        # elif partner_chain_res_count == 4:
                                        #     cluster_member_partner_iplus_res_code = IUPACData.protein_letters_3to1[
                                        #         atom.residue_type.title()] if atom.residue_type.title() in IUPACData.protein_letters_3to1 else None
                                if not np.isnan(residue_frequency):
                                    central_res_pairs.append(residue_frequency) # np_array(2, 5)
                                # if cluster_member_mapped_central_res_code is not None and cluster_member_partner_central_res_code is not None:
                                #     central_res_pairs.append(cluster_member_mapped_central_res_code + cluster_member_partner_central_res_code) # (string + string)
                                # if cluster_member_mapped_iplus_res_code is not None and cluster_member_partner_iplus_res_code is not None:
                                #     central_res_pairs.append(cluster_member_mapped_central_res_code + cluster_member_partner_central_res_code) # (string + string)

                                rmsd = guide_atom_rmsd(cluster_rep_guide_atoms, cluster_member_guide_atoms)
                                rmsd_sum += rmsd

                                cluster_member_count += 1

                    if cluster_member_count > 0:
                        mean_cluster_rmsd = rmsd_sum / float(cluster_member_count)
                        mapped_counts_dict = populate_aa_dictionary(fragment_length)
                        partner_counts_dict = populate_aa_dictionary(fragment_length)

                        for n in range(len(central_res_pairs)):
                            for residue in central_res_pairs[n]:
                                for i, j in residue:
                                    mapped_counts_dict[residue][i] += 1
                                    partner_counts_dict[residue][j] += 1

                        # Make Frequency Distribution Dictionaries
                        mapped_freq_dict = freq_distribution(mapped_counts_dict, cluster_member_count)
                        partner_freq_dict = freq_distribution(partner_counts_dict, cluster_member_count)
                        full_dictionary = {'size': cluster_member_count, 'rmsd': mean_cluster_rmsd, 'representative': str(cluster_rep), 'mapped': mapped_freq_dict, 'paired': partner_freq_dict}

                        # Save Full Cluster Dictionary as Binary Dictionary
                        with open(ijk_dir + "/" + ijk_cluster_name + '.pkl', 'wb') as f:
                            pickle.dump(full_dictionary, f, pickle.HIGHEST_PROTOCOL)

                        # ALL JOSH RESIDUAL
                        central_res_pairs_cnt_dict = dict(Counter(central_res_pairs)) # counts number of (string + string) occurances
                        central_res_pairs_cnt_list = sorted(central_res_pairs_cnt_dict.items(), key=lambda tup: tup[1], reverse=True)
                        central_res_pairs_cnt_list = ["%s %s" %(res_pair_type, res_pair_count) for res_pair_type, res_pair_count in central_res_pairs_cnt_list]

                        central_res_pairs_freq_list = []
                        for res_pair in central_res_pairs_cnt_dict:
                            central_res_pairs_freq_list.append("%s %s" %(res_pair, str(round(central_res_pairs_cnt_dict[res_pair] / float(cluster_member_count), 4))))
                        central_res_pairs_freq_list.sort(key=lambda s: float(s.split()[1]), reverse=True)

                        # out_l1 = "NAME: %s\n" %str(ijk_cluster_name)
                        out_l2 = "SIZE: %s\n" % str(cluster_member_count)
                        out_l3 = "RMSD: %s\n" % str(mean_cluster_rmsd)
                        out_l4 = "REPRESENTATIVE NAME: %s\n" % str(cluster_rep)
                        out_l5 = []



                        for fragment_index in mapped_counts_dict:
                            counts = mapped_counts_dict.get()

                            line = "MAPPED RESIDUE %s FREQUENCY:\n" % fragment_index +   # ', '.join(central_res_pairs_freq_list) + "\n"
                            out_l5.append(line)

                        out_l6 = "PARTNER RESIDUE %s FREQUENCY:\n" % fragment_index + ', '.join(central_res_pairs_freq_list) + "\n"

                        out_l6 = "i RESIDUE PAIR FREQUENCY:\n" + '\n'.join(central_res_pairs_freq_list) + "\n"
                        out_l7 = "i+ RESIDUE PAIR FREQUENCY:\n" + '\n'.join(central_res_pairs_freq_list) + "\n"
                        out_l8 = "i- RESIDUE PAIR COUNT:\n" + '\n'.join(central_res_pairs_cnt_list) + "\n"
                        out_l9 = "i RESIDUE PAIR COUNT:\n" + '\n'.join(central_res_pairs_cnt_list) + "\n"
                        out_l10 = "i+ RESIDUE PAIR COUNT:\n" + '\n'.join(central_res_pairs_cnt_list) + "\n"

                        outfile = open(ijk_dir + "/" + ijk_cluster_name + ".txt", "w")
                        outfile.write(out_l1)
                        outfile.write(out_l2)
                        outfile.write(out_l3)
                        outfile.write(out_l4)
                        outfile.write(out_l5)
                        outfile.write(out_l6)
                        outfile.close()


if __name__ == '__main__':
    main()

