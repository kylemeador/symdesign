import os
import pickle
from itertools import repeat

import FragUtils as Frag
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1

from PDB import PDB

# Globals
module = 'Make IJK Cluster Frequency Files:'


def ijk_stats(cluster_rep_path, db_dir, info_outdir, frag_length, sc_dist):
    # Initialize Variables
    lower_bound, upper_bound, index_offset = Frag.parameterize_frag_length(frag_length)
    root = os.path.dirname(cluster_rep_path)
    file1 = os.path.basename(cluster_rep_path)

    # Get Representative Guide Atoms
    cluster_rep_pdb = PDB()
    cluster_rep_pdb.readfile(cluster_rep_path)
    cluster_rep_guide_atoms = Frag.get_guide_atoms(cluster_rep_pdb)

    ijk_name = os.path.basename(root)
    i_type = ijk_name.split('_')[0]
    j_type = ijk_name.split('_')[1]
    ij_dir = os.path.join(i_type, i_type + bytes('_') + j_type)
    ijk_dir = os.path.join(ij_dir, ijk_name)

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

            member_mapped_ch = file2[file2.find('mapch') + 6:file2.find('mapch') + 7]
            member_paired_ch = file2[file2.find('pairch') + 7:file2.find('pairch') + 8]

            # Get Residue Counts for each Fragment in the Cluster
            residue_frequency = np.empty((frag_length, 2), dtype=object)
            mapped_chain_res_count = 0
            paired_chain_res_count = 0
            for atom in member_pdb.atoms:
                if atom.is_CA() and atom.chain == member_mapped_ch:
                    residue_frequency[mapped_chain_res_count][0] = \
                        protein_letters_3to1.get(atom.residue_type.title(), None)
                    mapped_chain_res_count += 1
                elif atom.is_CA() and atom.chain == member_paired_ch:
                    residue_frequency[paired_chain_res_count][1] = \
                        protein_letters_3to1.get(atom.residue_type.title(), None)
                    paired_chain_res_count += 1

            if np.all(residue_frequency):
                fragment_residue_counts.append(residue_frequency)  # type is np_array(fragment_index, 2)
                total_cluster_weight[file2] = Frag.collect_frag_weights(member_pdb, member_mapped_ch,
                                                                        member_paired_ch, sc_dist)
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
        final_weights = np.zeros((2, frag_length))
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
        # i = 0
        for i, residue in enumerate(mapped_freq_dict):
            mapped_freq_dict[residue]['stats'][1] = round(final_weights[0][i], 3)
            # i += 1
        # j = 0
        for j, residue in enumerate(partner_freq_dict):
            partner_freq_dict[residue]['stats'][1] = round(final_weights[1][j], 3)
            # j += 1

        # Save Full Cluster Dictionary as Binary Dictionary
        full_dictionary = {'size': cluster_count, 'rmsd': mean_cluster_rmsd, 'rep': str(file1),
                           'mapped': mapped_freq_dict, 'paired': partner_freq_dict}
        with open(os.path.join(ijk_out_dir, ijk_name + '.pkl'), 'wb') as f:
            pickle.dump(full_dictionary, f, pickle.HIGHEST_PROTOCOL)

    return 0


def main(db_dir, info_outdir, frag_length, sc_dist, multi=False, num_threads=4):
    # Get Representatives and IJK directories
    print('%s Beginning' % module)
    cluster_rep_paths = []
    for root, dirs1, files1 in os.walk(db_dir):
        if not dirs1:
            for file1 in files1:
                if file1.endswith('_representative.pdb'):
                    cluster_rep_paths.append(os.path.join(root, file1))

    if multi:
        zipped_args = zip(cluster_rep_path, repeat(db_dir), repeat(info_outdir), repeat(frag_length), repeat(sc_dist))
        result = Frag.mp_starmap(ijk_reps, zipped_args, num_threads)
        # Frag.report_errors(result)
    else:
        lower_bound, upper_bound, index_offset = Frag.parameterize_frag_length(frag_length)
        for cluster_rep_path in cluster_rep_paths:
            # Get Representative Guide Atoms
            cluster_rep_pdb = PDB()
            cluster_rep_pdb.readfile(cluster_rep_path)
            cluster_rep_guide_atoms = Frag.get_guide_atoms(cluster_rep_pdb)

            ijk_name = os.path.basename(root)
            i_type = ijk_name.split('_')[0]
            j_type = ijk_name.split('_')[1]
            # k_type = ijk_name.split('_')[2]
            ij_dir = os.path.join(i_type, i_type + bytes('_') + j_type)
            ijk_dir = os.path.join(ij_dir, ijk_name)

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

                    member_mapped_ch = file2[file2.find('mapch') + 6:file2.find('mapch') + 7]
                    member_paired_ch = file2[file2.find('pairch') + 7:file2.find('pairch') + 8]

                    # Get Residue Counts for each Fragment in the Cluster
                    residue_frequency = np.empty((frag_length, 2), dtype=object)
                    mapped_chain_res_count = 0
                    paired_chain_res_count = 0
                    for atom in member_pdb.atoms:
                        if atom.is_CA() and atom.chain == member_mapped_ch:
                            residue_frequency[mapped_chain_res_count][0] = \
                                protein_letters_3to1.get(atom.residue_type.title(), None)
                            mapped_chain_res_count += 1
                        elif atom.is_CA() and atom.chain == member_paired_ch:
                            residue_frequency[paired_chain_res_count][1] = \
                                protein_letters_3to1.get(atom.residue_type.title(), None)
                            paired_chain_res_count += 1

                    if np.all(residue_frequency):
                        fragment_residue_counts.append(residue_frequency)  # type is np_array(fragment_index, 2)
                        total_cluster_weight[file2] = Frag.collect_frag_weights(member_pdb, member_mapped_ch,
                                                                                member_paired_ch, sc_dist)
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
                final_weights = np.zeros((2, frag_length))
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
    ijk_rmsd_thresh = 1
    _fragment_length = 5
    side_chain_contact_dist = 5

    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    ijk_db = os.path.join(outdir, 'db_%d' % ijk_rmsd_thresh)
    info_db = os.path.join(outdir, 'info_%d' % ijk_rmsd_thresh)
    if not os.path.exists(info_db):
        os.makedirs(info_db)

    main(ijk_db, info_db, _fragment_length, side_chain_contact_dist)
