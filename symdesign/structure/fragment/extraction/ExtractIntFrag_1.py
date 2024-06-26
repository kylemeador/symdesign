from itertools import repeat
import logging
import os

from symdesign.structure.base import Structure
import symdesign.structure.fragment.info
import symdesign.structure.fragment.extraction.FragUtils as Frag
from symdesign.structure.model import Model
from symdesign import utils

# Globals
module = 'Extract Interface Fragments:'
logger = logging.getLogger(__name__)


def extract_to_db(db_cursor, fragment_length=5):
    central_res_idx = int(fragment_length / 2)
    # Use the central_res_index to pull out the residue_number, combine with PDB_code, interface ID, and chain to give
    # residue record which will populate the fragment table entry
    # Need to link the paired fragment table to the individual fragment using the individual fragment records... Can I
    # do a pass through the individual fragment records saving them first then using an ID returned for each to make the
    # paired fragment table


def extract_to_file(pdb, interaction_distance=8, fragment_length=5, out_path=os.getcwd(), individual=False,
                    same_chain=False):
    for fragment1, fragment2 in extract_fragments(pdb, interaction_distance, fragment_length, same_chain=same_chain):
        write_fragment_pair(fragment1, fragment2, pdb_name=pdb.name, out_path=out_path, fragment_length=fragment_length,
                            individual=individual)


def extract_fragments(pdb, distance, frag_length, same_chain=False):
    if len(pdb.chain_ids) != 2:  # ensure two chains are present
        logger.info('%s %s is missing two chains... It will be skipped!' % (module, pdb.name))
        return pdb.name

    # Create PDB instance for Ch1 and Ch2
    chain1, chain2 = pdb.chains
    if not chain1.atoms or not chain2.atoms:
        logger.info('%s is missing atoms in one of two chains... It will be skipped!' % pdb.name)
        return pdb.name
    # Find Pairs of Interacting Residues
    interacting_pairs = Frag.find_interface_pairs(chain1, chain2, distance=distance)
    fragment_pairs = \
        find_interacting_residue_fragments(chain1, chain2, interacting_pairs, frag_length, same_chain=same_chain)

    return fragment_pairs


def find_interacting_residue_fragments(chain1, chain2, interacting_pairs, frag_length, same_chain=False):
    fragment_pairs = []
    raise NotImplementedError('This function is set up for Residue.number but it should use Residue. Modify to use '
                              'Residue.get_downstream() and Residue.get_upstream() based on frag_length')
    for residue1, residue2 in interacting_pairs:
        # parameterize fragments based on input length
        res_nums_pdb1 = set(residue1 + i for i in range(*symdesign.structure.fragment.info.parameterize_frag_length(frag_length)))
        res_nums_pdb2 = set(residue2 + i for i in range(*symdesign.structure.fragment.info.parameterize_frag_length(frag_length)))

        if same_chain:
            # break iteration if residue 1 succeeds residue 2 or they are sequential, or frag 1 residues are in frag 2
            if residue1 + 1 >= residue2 and res_nums_pdb1.intersection(res_nums_pdb2) != set():
                continue

        # ensure that all of the residues are all present
        frag1 = chain1.get_residues(res_nums_pdb1)
        frag2 = chain2.get_residues(res_nums_pdb2)
        if len(frag1) == frag_length and len(frag2) == frag_length:
            # make a Structure and append to the list
            fragment_pairs.append((Structure.from_residues(frag1), Structure.from_residues(frag2)))

    return fragment_pairs


def write_fragment_pair(frag1: Structure, frag2: Structure, pdb_name, out_path=os.getcwd(), fragment_length=5, individual=False):
    os.makedirs(os.path.join(out_path, 'paired'), exist_ok=True)

    central_res_idx = int(fragment_length / 2)

    if individual:
        if not os.path.exists(os.path.join(out_path, 'individual')):
            os.makedirs(os.path.join(out_path, 'individual'))
        frag1.write(os.path.join(out_path, 'individual', '%s_ch%s_res%d.pdb'
                                 % (pdb_name, frag1.name, frag1.residues[central_res_idx].number)))
        frag2.write(os.path.join(out_path, 'individual', '%s_ch%s_res%d.pdb'
                                 % (pdb_name, frag2.name, frag2.residues[central_res_idx].number)))

    # int_frag_out = Model.from_atoms(frag1.atoms + frag2.atoms)
    with open(os.path.join(out_path, 'paired', '%s_ch%s_%s_res%d_%d.pdb' % (pdb_name, frag1.name, frag2.name,
                                                                            frag1.residues[central_res_idx].number,
                                                                            frag2.residues[central_res_idx].number)),
              'w') as f:
        frag1.write(file_handle=f)
        frag2.write(file_handle=f)


def main(int_db_dir, outdir, frag_length, interface_dist, individual=True, paired=False, multi=False, num_threads=4):  # paired_outdir
    # TODO parameterize individual and outdir in ExtractFragments main script. Change paired keyword to same_chain
    print('%s Beginning' % module)
    # Get Natural Interface PDB File Paths
    int_db_filepaths = utils.get_file_paths_recursively(int_db_dir, extension='.pdb')
    lower_bound, upper_bound, index_offset = Frag.parameterize_frag_length(frag_length)

    print('%s Creating Neighbor CB Atom Trees at %d Angstroms Distance' % (module, interface_dist))
    # Reading In PDB Structures
    pdbs_of_interest = [Model.from_file(pdb_path, entities=False) for pdb_path in int_db_filepaths]
    # pdbs_of_interest = [SDUtils.read_pdb(pdb_path) for pdb_path in int_db_filepaths]
    for i, pdb in enumerate(pdbs_of_interest):
        pdbs_of_interest[i].name = os.path.splitext(os.path.basename(pdb.file_path))[0]

    if multi:
        zipped_args = zip(pdbs_of_interest, repeat(interface_dist), repeat(frag_length), repeat(outdir),
                          repeat(individual), repeat(paired))
        result = utils.mp_starmap(extract_to_file, zipped_args, num_threads)

        Frag.report_errors(result)
    else:
        for pdb in pdbs_of_interest:
            extract_to_file(pdb, interaction_distance=interface_dist, fragment_length=frag_length, out_path=outdir,
                            individual=individual, same_chain=paired)
        # for pdb_path in int_db_filepaths:
        #     # Reading In PDB Structure
        #     pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        #     pdb = PDB()
        #     pdb.readfile(pdb_path, remove_alt_location=True)
            # # Ensure two chains are present
            # if len(pdb.chain_ids) != 2:
            #     print('%s %s is missing two chains... It will be skipped!' % (module, pdb_id))
            #     continue
            # pdb_ch1_id = pdb.chain_ids[0]
            # pdb_ch2_id = pdb.chain_ids[-1]
            #
            # # Create PDB instance for Ch1 and Ch2
            # pdb_ch1 = PDB()
            # pdb_ch2 = PDB()
            # pdb_ch1.read_atom_list(pdb.chain(pdb_ch1_id))
            # pdb_ch2.read_atom_list(pdb.chain(pdb_ch2_id))
            #
            # # Find Pairs of Interacting Residues
            # if pdb_ch1.atoms == list() or pdb_ch2.atoms == list():
            #     print('%s %s is missing atoms in one of two chains... It will be skipped!' % (module, pdb_id))
            #     continue
            # interacting_pairs = Frag.find_interface_pairs(pdb_ch1, pdb_ch2, interface_dist)
            #
            # ch1_central_res_num_used = []
            # ch2_central_res_num_used = []
            # for center_pair in interacting_pairs:
            #     if paired:
            #         # Only process center_pair if residue 1 is less than residue 2
            #         # Never process pairs if they are directly connected Ex. (34, 35)
            #         if center_pair[0] + 1 >= center_pair[1]:
            #             continue
            #
            #     ch1_res_num_list = [center_pair[0] + i for i in range(lower_bound, upper_bound + 1)]
            #     ch2_res_num_list = [center_pair[1] + i for i in range(lower_bound, upper_bound + 1)]
            #
            #     if paired:
            #         # Only process center_pair if residues from fragment 1 are not in fragment 2
            #         _break = False
            #         for res in ch1_res_num_list:
            #             if res in ch2_res_num_list:
            #                 _break = True
            #         if _break:
            #             continue
            #
            #     frag1_ca_count = 0
            #     int_frag_out_atom_list_ch1 = []
            #     for atom in pdb_ch1.atoms:
            #         if atom.residue_number in ch1_res_num_list:
            #             int_frag_out_atom_list_ch1.append(atom)
            #             if atom.is_ca():
            #                 frag1_ca_count += 1
            #
            #     frag2_ca_count = 0
            #     int_frag_out_atom_list_ch2 = []
            #     for atom in pdb_ch2.atoms:
            #         if atom.residue_number in ch2_res_num_list:
            #             int_frag_out_atom_list_ch2.append(atom)
            #             if atom.is_ca():
            #                 frag2_ca_count += 1
            #
            #     int_frag_out = PDB()
            #     int_frag_out_ch1 = PDB()
            #     int_frag_out_ch2 = PDB()
            #
            #     if frag1_ca_count == frag_length and frag2_ca_count == frag_length:
            #         if not paired:
            #             if center_pair[0] not in ch1_central_res_num_used:
            #                 int_frag_out_ch1.read_atom_list(int_frag_out_atom_list_ch1)
            #                 int_frag_out_ch1.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
            #                                                     '_res_' + str(center_pair[0]) + '_ch_' + pdb_ch1_id +
            #                                                     '.pdb'))
            #                 ch1_central_res_num_used.append(center_pair[0])
            #
            #             if center_pair[1] not in ch2_central_res_num_used:
            #                 int_frag_out_ch2.read_atom_list(int_frag_out_atom_list_ch2)
            #                 int_frag_out_ch2.write(os.path.join(single_outdir, pdb_id + '_' + pdb_ch1_id + pdb_ch2_id +
            #                                                     '_res_' + str(center_pair[1]) + '_ch_' + pdb_ch2_id +
            #                                                     '.pdb'))
            #                 ch2_central_res_num_used.append(center_pair[1])
            #
            #         int_frag_out.read_atom_list(int_frag_out_atom_list_ch1 + int_frag_out_atom_list_ch2)
            #         int_frag_out.write(os.path.join(paired_outdir, pdb_id + '_ch_' + pdb_ch1_id + pdb_ch2_id +
            #                                         '_res_' + str(center_pair[0]) + '_' + str(center_pair[1]) + '.pdb'))

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
