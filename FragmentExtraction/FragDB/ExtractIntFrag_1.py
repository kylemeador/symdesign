import os
from itertools import repeat

import FragUtils as Frag

from PDB import PDB
from Structure import Atom

# Globals
module = 'Extract Interface Fragments:'

def atom_record_to_atom(record):
    atom = Atom(number, type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y,
                z, occ, temp_fact, element_symbol, atom_charge)


def extract_frags(pdb_path, single_outdir, paired_outdir, interface_dist, lower_bound, upper_bound, frag_length,
                  paired=False):
    # nonlocal lower_bound, upper_bound, index_offset, paired
    # Reading In PDB Structure
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    pdb = PDB.from_file(pdb_path)
    # Ensure two chains are present
    if len(pdb.chain_ids) != 2:
        print('%s %s is missing two chains... It will be skipped!' % (module, pdb_id))
        return pdb_id
    pdb_ch1_id, pdb_ch2_id = pdb.chain_ids

    # # Create PDB instance for Ch1 and Ch2
    # pdb_ch1 = PDB.from_atoms(pdb.chain(pdb_ch1_id).atoms)
    # pdb_ch2 = PDB.from_atoms(pdb.chain(pdb_ch2_id).atoms)

    # Find Pairs of Interacting Residues
    # if not pdb_ch1.atoms or not pdb_ch2.atoms:
    #     print('%s %s is missing atoms in one of two chains... It will be skipped!' % (module, pdb_id))
    #     return pdb_id
    interacting_pairs = Frag.find_interface_pairs(pdb.chain(pdb_ch1_id), pdb.chain(pdb_ch2_id), interface_dist)

    ch1_central_res_num_used = []
    ch2_central_res_num_used = []
    for center_pair in interacting_pairs:
        if paired:
            # Only process center_pair if residue 1 is less than residue 2
            # Never process pairs if they are directly connected Ex. (34, 35)
            if center_pair[0] + 1 >= center_pair[1]:
                continue

        ch1_res_num_list = [center_pair[0] + i for i in range(lower_bound, upper_bound + 1)]
        ch2_res_num_list = [center_pair[1] + i for i in range(lower_bound, upper_bound + 1)]

        if paired:
            # Only process center_pair if residues from fragment 1 are not in fragment 2
            _break = False
            for res in ch1_res_num_list:
                if res in ch2_res_num_list:
                    _break = True
            if _break:
                continue

        frag1_ca_count = 0
        int_frag_out_atom_list_ch1 = []
        for atom in pdb.chain(pdb_ch1_id).atoms:
            if atom.residue_number in ch1_res_num_list:
                int_frag_out_atom_list_ch1.append(atom)
                if atom.is_ca():
                    frag1_ca_count += 1

        frag2_ca_count = 0
        int_frag_out_atom_list_ch2 = []
        for atom in pdb.chain(pdb_ch2_id).atoms:
            if atom.residue_number in ch2_res_num_list:
                int_frag_out_atom_list_ch2.append(atom)
                if atom.is_ca():
                    frag2_ca_count += 1

        if frag1_ca_count == frag_length and frag2_ca_count == frag_length:
            if not paired:
                if center_pair[0] not in ch1_central_res_num_used:
                    int_frag_out_ch1 = PDB.from_atoms(int_frag_out_atom_list_ch1)
                    int_frag_out_ch1.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
                                                        '_res_' + str(center_pair[0]) + '_ch_' + pdb_ch1_id +
                                                        '.pdb'))
                    ch1_central_res_num_used.append(center_pair[0])

                if center_pair[1] not in ch2_central_res_num_used:
                    int_frag_out_ch2 = PDB.from_atoms(int_frag_out_atom_list_ch2)
                    int_frag_out_ch2.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
                                                        '_res_' + str(center_pair[1]) + '_ch_' + pdb_ch2_id +
                                                        '.pdb'))
                    ch2_central_res_num_used.append(center_pair[1])

            int_frag_out = PDB.from_atoms(int_frag_out_atom_list_ch1 + int_frag_out_atom_list_ch2)
            int_frag_out.write(os.path.join(paired_outdir, pdb_id + '_ch_' + pdb_ch1_id + pdb_ch2_id + '_res_'
                                            + str(center_pair[0]) + '_' + str(center_pair[1]) + '.pdb'))

    return 0


def main(int_db_dir, single_outdir, paired_outdir, frag_length, interface_dist, paired=False, multi=False,
         num_threads=4):
    print('%s Beginning' % module)
    # Get Natural Interface PDB File Paths
    int_db_filepaths = Frag.get_all_pdb_file_paths(int_db_dir)
    lower_bound, upper_bound, index_offset = Frag.parameterize_frag_length(frag_length)

    print('%s Creating Neighbor CB Atom Trees at %d Angstroms Distance' % (module, interface_dist))
    if multi:
        zipped_filepaths = zip(int_db_filepaths, repeat(single_outdir), repeat(paired_outdir), repeat(interface_dist),
                               repeat(lower_bound), repeat(upper_bound), repeat(frag_length), repeat(paired))
        result = Frag.mp_starmap(extract_frags, zipped_filepaths, num_threads)
        Frag.report_errors(result)
    else:
        for pdb_path in int_db_filepaths:
            # Reading In PDB Structure
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            pdb = PDB.from_file(pdb_path)
            # Ensure two chains are present
            if len(pdb.chain_ids) != 2:
                print('%s %s is missing two chains... It will be skipped!' % (module, pdb_id))
                continue
            pdb_ch1_id = pdb.chain_ids[0]
            pdb_ch2_id = pdb.chain_ids[-1]
            #
            # # Find Pairs of Interacting Residues
            # if pdb_ch1.atoms is not None or pdb_ch2.atoms is not None:
            #     print('%s %s is missing atoms in one of two chains... It will be skipped!' % (module, pdb_id))
            #     continue
            interacting_pairs = Frag.find_interface_pairs(pdb.chain(pdb_ch1_id), pdb.chain(pdb_ch2_id), interface_dist)

            ch1_central_res_num_used = []
            ch2_central_res_num_used = []
            for center_pair in interacting_pairs:
                if paired:
                    # Only process center_pair if residue 1 is less than residue 2
                    # Never process pairs if they are directly connected Ex. (34, 35)
                    if center_pair[0] + 1 >= center_pair[1]:
                        continue

                ch1_res_num_list = [center_pair[0] + i for i in range(lower_bound, upper_bound + 1)]
                ch2_res_num_list = [center_pair[1] + i for i in range(lower_bound, upper_bound + 1)]

                if paired:
                    # Only process center_pair if residues from fragment 1 are not in fragment 2
                    _break = False
                    for res in ch1_res_num_list:
                        if res in ch2_res_num_list:
                            _break = True
                    if _break:
                        continue

                frag1_ca_count = 0
                int_frag_out_atom_list_ch1 = []
                for atom in pdb.chain(pdb_ch1_id).atoms:
                    if atom.residue_number in ch1_res_num_list:
                        int_frag_out_atom_list_ch1.append(atom)
                        if atom.is_ca():
                            frag1_ca_count += 1

                frag2_ca_count = 0
                int_frag_out_atom_list_ch2 = []
                for atom in pdb.chain(pdb_ch2_id).atoms:
                    if atom.residue_number in ch2_res_num_list:
                        int_frag_out_atom_list_ch2.append(atom)
                        if atom.is_ca():
                            frag2_ca_count += 1

                if frag1_ca_count == frag_length and frag2_ca_count == frag_length:
                    if not paired:
                        if center_pair[0] not in ch1_central_res_num_used:
                            int_frag_out_ch1 = PDB.from_atoms(int_frag_out_atom_list_ch1)
                            int_frag_out_ch1.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
                                                                '_res_' + str(center_pair[0]) + '_ch_' + pdb_ch1_id +
                                                                '.pdb'))
                            ch1_central_res_num_used.append(center_pair[0])

                        if center_pair[1] not in ch2_central_res_num_used:
                            int_frag_out_ch2 = PDB.from_atoms(int_frag_out_atom_list_ch2)
                            int_frag_out_ch2.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
                                                                '_res_' + str(center_pair[1]) + '_ch_' + pdb_ch2_id +
                                                                '.pdb'))
                            ch2_central_res_num_used.append(center_pair[1])

                    int_frag_out = PDB.from_atoms(int_frag_out_atom_list_ch1 + int_frag_out_atom_list_ch2)
                    int_frag_out.write(os.path.join(paired_outdir, pdb_id + '_ch_' + pdb_ch1_id + pdb_ch2_id +
                                                    '_res_' + str(center_pair[0]) + '_' + str(center_pair[1]) + '.pdb'))
    if paired:
        save_string = 'Paired'
    else:
        save_string = 'Paired and Individual'
    print('%s Saved All %s Fragments' % (module, save_string))
    print('%s Finished' % module)


if __name__ == '__main__':
    clean_pdb_dir = os.path.join(os.getcwd(), 'all_clean_pdbs')
    indv_frag_outdir = os.path.join(os.getcwd(), 'all_individual_frags')
    pair_frag_outdir = os.path.join(os.getcwd(), 'all_paired_frags')
    if not os.path.exists(indv_frag_outdir):
        os.makedirs(indv_frag_outdir)
    if not os.path.exists(pair_frag_outdir):
        os.makedirs(pair_frag_outdir)

    fragment_length = 5
    interface_distance = 8
    thread_count = 8

    main(clean_pdb_dir, indv_frag_outdir, pair_frag_outdir, fragment_length, interface_distance,
         num_threads=thread_count)
