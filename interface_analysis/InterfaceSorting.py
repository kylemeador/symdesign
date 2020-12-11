import os

import functions as fc
from symdesign.SymDesignUtils import pickle, unpickle, get_all_pdb_file_paths


# Todo PISA path
def get_pisa_filepath(pdb_code, directory='/home/kmeador/yeates/fragment_database/all/pisa_files', file_type='pisa'):
    """Returns the PISA path that corresponds to the PDB code from the local PISA Database"""
    pisa_type_extensions = {'multimers': '.xml', 'interfaces': '.xml', 'multimer': '.pdb', 'pisa': '.pkl'}
    if file_type in pisa_type_extensions:
        sub_dir = pdb_code[1:3].lower()
        pisa_string = '_%s%s' % (file_type, pisa_type_extensions[file_type])
        file_path = os.path.join(directory, sub_dir, ' %s%s' % (pdb_code, pisa_string))
        if os.path.exists(file_path):
            return file_path

    return False  # None


def set_up_interface_dict(pdb_interface_codes):
    """From all PDB interfaces, make a dictionary of the PDB: [interface number] pairs

    Returns:
        {pdb_code: [1, 3, 4], ...}
    """
    int_dict = {}
    for code in pdb_interface_codes:
        pdb_code = code.split('-')
        # ['1ABC', '3']
        if pdb_code[0] in int_dict:
            int_dict[pdb_code[0]].add(int(pdb_code[1]))
        else:
            int_dict[pdb_code[0]] = set(int(pdb_code[1]))

    return int_dict


def sort_interfaces_by_contact_type(int_d):
    """From a directory of interface pdbs with structure interface_pdbs/ab/1ABC-3.pdb
        grab the multimeric pisa xml file and search for all biological assemblies

    """

    sorted_int_d, pdb_not_qsbio = {}, {}
    for pdb in int_d:
        if pdb in qsbio_confirmed:
            pisa_path = get_pisa_filepath(pdb)
            if 'pkl' in pisa_path:
                pisa_d = unpickle(pisa_path)
            else:
                raise Exception('%s does not have the required file at %s' % (pdb, pisa_path))
            multimer = pisa_d['multimers']
            bio_ass = {}
            if multimer[(1, 1)] != 'MONOMER':  # if the pdb is not a monomer
                for set_complex in multimer:
                    biological_assembly = multimer[set_complex]['pdb_BA']
                    if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                        if biological_assembly in bio_ass:  # is it in the BA dict?
                            bio_ass[biological_assembly].append(set_complex)
                        else:  # add to the BA dict
                            bio_ass[biological_assembly] = [set_complex]

                # Grab all unique interface numbers invoked in multimer entries if they are in QSBio
                bio_int = []
                for ba in bio_ass:
                    # Todo make ba an int in qsbio_confrimed
                    if str(ba) in qsbio_confirmed[pdb]:  # ensure biological assembly is QSBio confirmed assembly
                        for set_complex in bio_ass[ba]:
                            bio_int += multimer[set_complex]['interfaces'].keys()
                # bio_int = list(set(bio_int))

                other_candidates = []
                for candidate_int in bio_int:
                    for rep in pisa_d['interfaces']['all_ids']:
                        if candidate_int in pisa_d['interfaces']['all_ids'][rep]:
                            other_candidates += pisa_d['interfaces']['all_ids'][rep]
                bio_int = set(bio_int) | set(other_candidates)

                # finally insure that the biological_interface identified has an associated entry in the int_d
                # final_bio_int = set(int_d[pdb]) & bio_int
                final_bio_int = []
                for interface in int_d[pdb]:
                    # Todo make ba an int in int_dict and then take the intersection (&) of int_d[pdb] and bio_int
                    if int(interface) in bio_int:
                        final_bio_int.append(interface)
            else:  # the pdb is a monomer
                final_bio_int = []
                # final_bio_int = {} Todo
        else:  # the pdb is not a confirmed assembly
            final_bio_int = []
            # final_bio_int = {} Todo
            pdb_not_qsbio[pdb] = []
        int_d[pdb] = {'bio': set(final_bio_int), 'xtal': set(int_d[pdb]) - set(final_bio_int), 'unknown': set()}
        # sorted_int_d[pdb] = {'bio': final_bio_int, 'xtal': set(int_d[pdb]) - final_bio_int, 'unknown': set()} Todo

    # Filter out possible, not confirmed QSBio, biological interfaces from the xtal interface set for each PDB not in QSBio
    for pdb in pdb_not_qsbio:
        pisa_path = get_pisa_filepath(pdb)  # MOD DOUBLE
        pisa_d = unpickle(pisa_path)  # DOUBLE
        multimer = pisa_d['multimers']  # DOUBLE
        bio_ass = {}  # DOUBLE
        if multimer[(1, 1)] != 'MONOMER':  # DOUBLE
            for set_complex in multimer:
                biological_assembly = multimer[set_complex]['pdb_BA']
                if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                    if biological_assembly in bio_ass:  # is it in the BA dict?
                        bio_ass[biological_assembly].append(set_complex)
                    else:  # add to the BA dict
                        bio_ass[biological_assembly] = [set_complex]

            possible_bio_int = []
            for ba in bio_ass:
                pdb_not_qsbio[pdb].append(ba)  # MOD
                for set_complex in bio_ass[ba]:
                    possible_bio_int += multimer[set_complex]['interfaces'].keys()
            possible_bio_int = list(set(possible_bio_int))

            other_candidates = []
            for candidate_int in possible_bio_int:
                for rep in pisa_d['interfaces']['all_ids']:
                    if candidate_int in pisa_d['interfaces']['all_ids'][rep]:
                        other_candidates += pisa_d['interfaces']['all_ids'][rep]
            possible_bio_int = set(possible_bio_int) | set(other_candidates)
            # DOUBLE

            final_xtal_int = []
            for interface in int_d[pdb]['xtal']:
                if int(interface) not in possible_bio_int:
                    final_xtal_int.append(interface)
            final_xtal_int = set(final_xtal_int)
            int_d[pdb]['unknown'] = int_d[pdb]['xtal'] - final_xtal_int
            int_d[pdb]['xtal'] = final_xtal_int

        final_bio_int, final_xtal_int, final_unknown_int = set(), set(), set()
        if pdb in qsbio_confirmed:
            # take the intersection (&) of int_d[pdb] and bio_int
            final_bio_int = int_d[pdb] & bio_int
            # final_bio_int = []
            # for interface in int_d[pdb]:
            #     if int(interface) in bio_int:
            #         final_bio_int.append(interface)
            final_xtal_int = int_d[pdb] - final_bio_int
        else:
            # final_xtal_int = []
            # for interface in int_d[pdb]['xtal']:
            # take the difference (-) of int_d[pdb] and bio_int
            final_xtal_int = int_d[pdb] - bio_int
            # for interface in int_d[pdb]:
            #     if int(interface) not in bio_int:
            #         final_xtal_int.append(interface)
            # final_xtal_int = set(final_xtal_int)

            # int_d[pdb]['unknown'] = int_d[pdb]['xtal'] - final_xtal_int
            final_unknown_int = int_d[pdb] - final_xtal_int
            # int_d[pdb]['xtal'] = final_xtal_int
        sorted_int_d[pdb] = {'bio': final_bio_int, 'xtal': final_xtal_int, 'unknown': final_unknown_int}

    return int_d, pdb_not_qsbio


if __name__ == '__main__':
    # pisa_dir = '/home/kmeador/yeates/fragment_database/all/pisa_files'
    # load_file = 'correct_these_pisa.txt'
    # with open(os.path.join(pisa_dir, load_file), 'r') as f:
    #     all_lines = f.readlines()
    #     clean_lines = []
    #     for line in all_lines:
    #         line = line.strip().upper()
    #         clean_lines.append(line)
    #
    # re_extract = []
    # for pdb_code in clean_lines:
    #     if pdb_code not in interface_dict:
    #         re_extract.append(pdb_code)
    # set_clean = set(clean_lines)
    # set_extract = set(re_extract)
    # print(len(set(clean_lines)), len(set(re_extract)), set_extract)
    # print(clean_lines[:3])
    # print(interface_dict['2YV0'])

    qsbio_file = '200121_pdb_lists/QSbio_GreaterThanHigh_Assemblies.pkl'
    qsbio_confirmed = unpickle(qsbio_file)
    interfaces_dir = 'all_interfaces'
    all_pdb_paths = get_all_pdb_file_paths(interfaces_dir)
    pdb_interface_codes = list(file_ext_split[0] for file_ext_split in map(os.path.splitext, map(os.path.basename, all_pdb_paths)))
    interface_d = set_up_interface_dict(pdb_interface_codes)

    final_int_dict, not_in_qsbio = sort_interfaces_by_contact_type(interface_d)

    monomers_file = '200121_pdb_lists/QSbio_Monomers.csv'
    with open(monomers_file, 'r') as f:
        all_lines = f.readlines()
        monomers = []
        for line in all_lines:
            line = line.strip()
            monomers.append(line)
    print('The number of monomers found in QSbio is %d' % len(set(monomers)))

    all_missing_biological = []
    all_biological = []
    for pdb in final_int_dict:
        if final_int_dict[pdb]['bio'] == set():
            all_missing_biological.append(pdb)
        else:
            all_biological.append(pdb)

    proposed_qsbio_monomer_set = []
    for pdb in all_missing_biological:
        if pdb in qsbio_confirmed:
            proposed_qsbio_monomer_set.append(pdb)

    confirmed_count = 0
    final_monomers = []
    starting_monomer_pdbs = []
    missing_from_confirmed = []
    missing_from_final = []
    for entry in monomers:
        pdb = entry.split('_')[0].upper()
        starting_monomer_pdbs.append(pdb)
        if pdb in qsbio_confirmed:
            confirmed_count += 1
        else:
            missing_from_confirmed.append(pdb)
        try:
            if final_int_dict[pdb]['bio'] == set():
                final_monomers.append(pdb)
        except KeyError:
            missing_from_final.append(pdb)
    actual_monomers = set(final_monomers) & set(starting_monomer_pdbs)
    missing_monomers = set(final_monomers) - set(starting_monomer_pdbs)
    incorrect_assembly_assignment = list(set(proposed_qsbio_monomer_set) - set(final_monomers))
    print('Length of QSbio monomers (%d) should equal %d (all monomers confirmed in QSbio) and '
          'maybe monomers missing from final (%d). If not, then those missing from final set (%d)'
          ' + final set monomers (%d) = %d should equal %d'
          % (len(monomers), confirmed_count, len(final_monomers), len(missing_from_final),
             len(final_monomers), len(missing_from_final) + len(final_monomers), len(monomers)))
    # '. It doesn\'t, so this is the difference', len(missing_from_confirmed), 'Some examples are:', missing_from_confirmed[:5])
    print('Number missing from the interface_dictionary = %d, with set = %d.' %
          (len(missing_from_final), len(set(missing_from_final))))
    print('There are %d PDB\'s missing Bio Assemblies and %d with Bio Assemblies.' %
          (len(all_missing_biological), len(all_biological)))
    print('Of those without Bio Assemblies, %d should be monomers per QSBio' %
          len(set(proposed_qsbio_monomer_set) & set(final_monomers)))
    print('And therefore %d are failures of the above processing.' % len(set(qsbio_monomer_set) - set(final_monomers)))

    # print('Unique monomeric PDB\'s in final_sorted_dict:', len(actual_monomers), '. There are', len(missing_monomers), 'failures in monomeric identification.')
    # print('The number of flawed monomer assignments, and therefore errors, in PISA parsing is', len(incorrect_assembly_assignment))
    print('Some examples of which include:', incorrect_assembly_assignment[:5])

    i = 0
    j = 0
    for entry in incorrect_assembly_assignment:
        if final_int_dict[entry]['unknown'] == set():
            i += 1
        if final_int_dict[entry]['xtal'] != set():
            j += 1
    print(i, j)

    for entry in incorrect_assembly_assignment:
        final_int_dict[entry]['unknown'] = final_int_dict[entry]['xtal']
        final_int_dict[entry]['xtal'] = set()

    j = 0
    for entry in missing_from_final:
        try:
            i = final_int_dict[entry]
        except KeyError:
            j += 1
    print('There are actually', j,
          'missing from the interface_dictionary that are in the monomers file. These should be outside of the interface selection criteria')
    all_protein_file = '200121_pdb_lists/under3A_protein_no_multimodel_no_mmcif_no_bad_pisa.txt'
    pdb_of_interest = fc.pdb_code_to_iterable(all_protein_file)
    k = 0
    for entry in missing_from_final:
        try:
            i = pdb_of_interest.index()
        except:
            k += 1
    print('Infact, there are', k, 'missing from proteins of interest in the first place')

    final_file = '200121_pdb_lists/PDBInterfacesSorted.pkl'
    pickle()
    with open(final_file, 'wb') as f:
        pickle.dump(final_int_dict, f, pickle.HIGHEST_PROTOCOL)

    pickle()
    not_qsbio_file = '200121_pdb_lists/0204_not_qs_bio.pkl'
    with open(not_qsbio_file, 'wb') as f:
        pickle.dump(not_in_qsbio, f, pickle.HIGHEST_PROTOCOL)
