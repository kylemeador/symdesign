import os

import ParsePisa as pp
from symdesign.SymDesignUtils import pickle_object, unpickle, get_all_pdb_file_paths, to_iterable

# Globals
pisa_type_extensions = {'multimers': '.xml', 'interfaces': '.xml', 'multimer': '.pdb', 'pisa': '.pkl'}


# Todo PISA path
def get_pisa_filepath(pdb_code, directory='/home/kmeador/yeates/fragment_database/all/pisa_files', file_type='pisa'):
    """Returns the PISA path that corresponds to the PDB code from the local PISA Database"""
    if file_type in pisa_type_extensions:
        pdb_code = pdb_code.upper()
        sub_dir = pdb_code[1:3].lower()
        root_path = os.path.join(directory, sub_dir)
        specific_file = ' %s_%s%s' % (pdb_code, file_type, pisa_type_extensions[file_type])
        if os.path.exists(os.path.join(root_path, specific_file)):
            return os.path.join(root_path, specific_file)
        else:  # attempt to make the file if not pickled
            if file_type == 'pisa':
                status = extract_pisa_files_and_pickle(root_path, pdb_code)
                if status:
                    return os.path.join(root_path, specific_file)
                else:  # try to download required files
                    dummy = True
                    # TODO implement while loop for the download in the case file_type != 'pisa' or status = False

    return None


def extract_pisa_files_and_pickle(root, pdb_code):
    """Take a set of pisa files acquired from pisa server and parse for saving as a complete pisa dictionary"""
    individual_d = {}
    try:
        interface, chain = pp.parse_pisa_interfaces_xml(os.path.join(root, '%s_interfaces.xml' % pdb_code))
        individual_d['interfaces'] = interface
        individual_d['chains'] = chain
        individual_d['multimers'] = pp.parse_pisa_multimers_xml(os.path.join(root, '%s_multimers.xml' % pdb_code))
    except OSError:
        return False

    pickle_object(individual_d, '%s_pisa' % pdb_code, out_path=root)

    return True


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
            int_dict[pdb_code[0].upper()].add(int(pdb_code[1]))
        else:
            int_dict[pdb_code[0].upper()] = set(int(pdb_code[1]))

    return int_dict


def sort_interfaces_by_contact_type(pdb, pisa_d, interface_number_set, assembly_confirmed):
    """From a directory of interface pdbs with structure interface_pdbs/ab/1ABC-3.pdb
        grab the multimeric pisa xml file and search for all biological assemblies

    """
    pisa_multimer = pisa_d['multimers']
    pisa_interfaces = pisa_d['interfaces']

    bio_ass = {}
    if pisa_multimer[(1, 1)] != 'MONOMER':  # if the pdb is not a monomer
        for set_complex in pisa_multimer:
            biological_assembly = pisa_multimer[set_complex]['pdb_BA']
            if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                if biological_assembly in bio_ass:  # is it already in BA dict?
                    bio_ass[biological_assembly].add(set_complex)
                else:  # add to the BA dict
                    bio_ass[biological_assembly] = {set_complex}

        # Grab all unique interface numbers from PDB assigned BA invoked in multimer entries
        possible_bio_int = []
        for ba in bio_ass:
            for set_complex in bio_ass[ba]:
                possible_bio_int += pisa_multimer[set_complex]['interfaces'].keys()

        other_candidates = []
        for candidate_int in possible_bio_int:
            for rep in pisa_interfaces['all_ids']:
                if candidate_int in pisa_interfaces['all_ids'][rep]:
                    other_candidates += pisa_interfaces['all_ids'][rep]
        all_possible_bio_int = set(possible_bio_int) | set(other_candidates)

        # Finally, check if the possible biological_interfaces have confirmed assemblies
        if assembly_confirmed != set():  # there are assemblies confirmed
            # Only use ba if they are in QSBio
            not_bio_int = []
            for ba in bio_ass:
                # Todo make ba an int in qsbio_confrimed
                if str(ba) not in assembly_confirmed:  # ensure biological assembly is QSBio confirmed assembly
                    for set_complex in bio_ass[ba]:
                        not_bio_int += pisa_multimer[set_complex]['interfaces'].keys()
                        # for interface in pisa_multimer[set_complex]['interfaces'].keys():
                        #     all_possible_bio_int.remove(interface)

            final_unknown_bio_int = set(not_bio_int)
            # take the intersection (&) of interface_number_set and bio_int
            final_bio_int = interface_number_set & (all_possible_bio_int - final_unknown_bio_int)
            final_xtal_int = interface_number_set - final_bio_int
        else:  # the pdb does not have a confirmed assembly
            # take the difference (-) of interface_number_set and all_possible_bio_int
            final_bio_int = set()
            final_unknown_bio_int = all_possible_bio_int
            final_xtal_int = interface_number_set - final_unknown_bio_int  # all interface identified, minus possible BA

    return {pdb: {'bio': final_bio_int, 'xtal': final_xtal_int, 'unknown_bio': final_unknown_bio_int}}


def sort_pdbs_to_uniprot_d(pdbs, pdb_uniprot_d):
    unp_master = {}
    no_unpid = []
    for pdb in pdbs:
        for chain in pdb_uniprot_d[pdb]['ref']:
            if pdb_uniprot_d[pdb]['ref'][chain]['db'] == 'UNP':  # only add the pdb and chain if corresponding UniProtID
                unpid = pdb_uniprot_d[pdb]['ref'][chain]['accession']
                if unpid in unp_master:
                    unp_master[unpid]['all'].append(pdb + '.' + chain)
                    unp_master[unpid]['unique_pdb'].add(pdb)
                    space_group = pdb_uniprot_d[pdb]['cryst']['space']
                    if space_group in unp_master[unpid]['space_groups']:
                        unp_master[unpid]['space_groups'][space_group]['all'].append(pdb)
                    else:
                        unp_master[unpid]['space_groups'][space_group] = {'all': [pdb], 'top': None}
                else:
                    unp_master[unpid] = {'all': ['%s.%s' % (pdb, chain)], 'unique_pdb': {pdb}, 'top': None,
                                         'bio': set(), 'partners': {}, 'space_groups': {
                            pdb_uniprot_d[pdb]['cryst']['space']: {'all': [pdb], 'top': None}}}
            else:
                no_unpid.append('%s.%s' % (pdb, chain))

    return unp_master, no_unpid


def process_uniprot_entry(uniprot_id, unp_d, pdb_uniprot_info, min_resolution_threshold=3.0):
    # for unpid in unp_master:
    # unp_master[unpid]['unique_pdb'] = list(set(unp_master[unpid]['unique_pdb']))
    min_res = min_resolution_threshold
    top_pdb = None
    for pdb in unp_d['unique_pdb']:
        # Find Uniprot ID with top Resolution
        if pdb_uniprot_info[pdb]['res'] <= min_res:  # structural resolution is less than threshold
            top_pdb = pdb
            min_res = pdb_uniprot_info[pdb]['res']

        # Check for pdbs which comprise Self and other Uniprot IDs. These represent distinct heteromeric interfaces
        if len(pdb_uniprot_info[pdb]['ref']) > 1:  # Check if there is more than one chain in the pdb
            for chain in pdb_uniprot_info[pdb]['ref']:
                if pdb_uniprot_info[pdb]['ref'][chain]['db'] == 'UNP':  # only add if there is corresponding UniProtID
                    partner_unpid = pdb_uniprot_info[pdb]['ref'][chain]['accession']
                    if partner_unpid != uniprot_id:  # only get other Uniprot IDs
                        if partner_unpid in unp_d['partners']:
                            unp_d['partners'][partner_unpid]['all'].append(pdb + '.' + chain)
                        else:
                            unp_d['partners'][partner_unpid] = {'all': [pdb + '.' + chain], 'top': None}

    # Save highest Resolution PDB Code and add to Biological Assembly list
    if top_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
        unp_d['top'] = top_pdb
    # unp_master[unpid]['bio'].add(top_pdb)  # Todo why? Could be a monomer

    # Add the highest resolution UniProt Partners to the Biological Assemblies set
    for partner in unp_d['partners']:
        min_partner_res = min_resolution_threshold
        top_partner = None
        for pdb in unp_d['partners'][partner]['all']:
            partner_res = pdb_uniprot_info[pdb.split('.')[0]]['res']
            if partner_res <= min_partner_res:
                top_partner = pdb
                min_partner_res = partner_res
        if top_partner:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['partners'][partner]['top'] = top_partner
            unp_d['bio'].add(top_partner.split('.')[0])

    # Save highest Resolution PDB Codes for each Space Group
    for cryst in unp_d['space_groups']:
        min_xtal_res = min_resolution_threshold
        top_xtal_pdb = None
        for pdb in unp_d['space_groups'][cryst]['all']:
            if pdb_uniprot_info[pdb]['res'] <= min_xtal_res:
                top_xtal_pdb = pdb
                min_xtal_res = pdb_uniprot_info[pdb]['res']
        if top_xtal_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['space_groups'][cryst]['top'] = top_xtal_pdb

    return unp_d


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
    # TODO Figure out how I made the all_interfaces PDB set. I definitely used PISA, but I have a script somewhere.
    #  I should make this modular so that I can swap out the writes and reads to use a MySQL. Figure out how to query
    #  PDB using under3A no multimodel no mmcif, no bad pisa file reliably.
    #  Finally see if there is an update to QSBio verified assemblies. Make a script to process these

    # Variables
    pdb_resolution_threshold = 3.0
    write_to_file = True  # TODO MySQL DB option would make this false

    # Input data/files
    current_interface_file_path = '/yeates1/kmeador/fragment_database/current_pdb_lists/'
    # TODO how did I get this? turn from a file to a MySQL db method
    interfaces_dir = '/yeates1/kmeador/fragment_database/all_interfaces'
    all_pdb_paths = get_all_pdb_file_paths(interfaces_dir)
    pdb_interface_codes = list(file_ext_split[0]
                               for file_ext_split in map(os.path.splitext, map(os.path.basename, all_pdb_paths)))
    interface_d = set_up_interface_dict(pdb_interface_codes)

    all_pdbs_dir = '/databases/pdb'  # ends with .ent not sub_directoried # TODO Unused
    # all_pdbs_dir = '/yeates1/kmeador/fragment_database/all_pdb'
    # all_pdbs_dir = 'all_pdbs'

    qsbio_file_name = 'QSbio_GreaterThanHigh_Assemblies'  # .pkl'
    qsbio_file = os.path.join(current_interface_file_path, qsbio_file_name)
    qsbio_confirmed_d = unpickle(qsbio_file)

    qsbio_monomers_file_name = 'QSbio_Monomers.csv'
    qsbio_monomers_file = os.path.join(current_interface_file_path, qsbio_monomers_file_name)
    # with open(qsbio_monomers_file, 'r') as f:
    #     all_lines = f.readlines()
    #     monomers = []
    #     for line in all_lines:
    #         line = line.strip()
    #         monomers.append(line)
    monomers = to_iterable(qsbio_monomers_file)
    print('The number of monomers found in %s: %d' % (qsbio_monomers_file, len(set(monomers))))

    # representatives_90_file_name = 'under3A_pdb_90_sequence_cluster_reps.txt'
    # representatives_90_file = os.path.join(current_interface_file_path, representatives_90_file_name)
    # rep_90 = to_iterable(representatives_90_file)

    all_pdbs_of_interest = 'under3A_protein_no_multimodel_no_mmcif_no_bad_pisa.txt'
    all_protein_file = os.path.join(current_interface_file_path, all_pdbs_of_interest)
    pdb_of_interest = to_iterable(all_protein_file)

    all_pdb_uniprot_file_name = '200206_MostCompleteAllPDBSpaceGroupUNPResInfo'  # .pkl'
    all_pdb_uniprot_file = os.path.join(current_interface_file_path, all_pdb_uniprot_file_name)
    pdb_uniprot_info = unpickle(all_pdb_uniprot_file)

    # Output data/files
    sorted_file_name = 'PDBInterfacesSorted'
    uniprot_heterooligomer_interface_file_name = 'UniquePDBHeteroOligomerInterfaces'  # was 200121_FinalInterfaceDict
    uniprot_homooligomer_interface_file_name = 'UniquePDBHomoOligomerInterfaces'  # was 200121_FinalInterfaceDict
    uniprot_unknown_bio_interface_file_name = 'UniquePDBUnknownOligomerInterfaces'  # was 200121_FinalInterfaceDict
    uniprot_xtal_interface_file_name = 'UniquePDBXtalInterfaces'  # was 200121_FinalInterfaceDict
    unique_chains_file_name = 'UniquePDBChains'  # was 200121_FinalIntraDict

    # TODO Back-end: Figure out how to convert these sets into MySQL tables so that the PDB files present are
    #  easily identified based on their assembly. Write load PDB from MySQL function. Also ensure I have a standard
    #  query format to grab different table schemas depending on the step of the fragment processing.

    missing_pisa_paths = []
    for pdb in interface_d:
        pisa_path = get_pisa_filepath(pdb)
        if pisa_path:
            pisa_d = unpickle(pisa_path)
            interface_sort_d = sort_interfaces_by_contact_type(pdb, pisa_d, interface_d[pdb],
                                                               set(qsbio_confirmed_d[pdb]))
        else:
            missing_pisa_paths.append(pdb)

    all_missing_biological = []
    all_biological = []
    for pdb in interface_sort_d:
        if interface_sort_d[pdb]['bio'] == set():
            all_missing_biological.append(pdb)
        else:
            all_biological.append(pdb)

    proposed_qsbio_monomer_set = []
    for pdb in all_missing_biological:
        if pdb in qsbio_confirmed_d:
            proposed_qsbio_monomer_set.append(pdb)

    confirmed_count = 0
    final_monomers = []
    starting_monomer_pdbs = []
    missing_from_confirmed = []
    missing_from_final = []
    for entry in monomers:
        pdb = entry.split('_')[0].upper()
        starting_monomer_pdbs.append(pdb)
        if pdb in qsbio_confirmed_d:
            confirmed_count += 1
        else:
            missing_from_confirmed.append(pdb)
        try:
            if interface_sort_d[pdb]['bio'] == set():
                final_monomers.append(pdb)
        except KeyError:
            missing_from_final.append(pdb)
    actual_monomers = set(final_monomers) & set(starting_monomer_pdbs)
    missing_monomers = set(final_monomers) - set(starting_monomer_pdbs)
    incorrect_assembly_assignment = list(set(proposed_qsbio_monomer_set) - set(final_monomers))
    print('Length of QSbio monomers (%d) should equal %d (all monomers confirmed in QSbio) and '
          'maybe monomers missing from final (%d). If not, then those missing from final (%d)'
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
        if interface_sort_d[entry]['unknown_bio'] == set():
            i += 1
        if interface_sort_d[entry]['xtal'] != set():
            j += 1
    print(i, j)

    for entry in incorrect_assembly_assignment:
        interface_sort_d[entry]['unknown_bio'] = interface_sort_d[entry]['xtal']
        interface_sort_d[entry]['xtal'] = set()

    j = 0
    for entry in missing_from_final:
        try:
            i = interface_sort_d[entry]
        except KeyError:
            j += 1
    print('There are actually %d missing from the interface_dictionary that are in the monomers file. '
          'These should be outside of the interface selection criteria')

    k = 0
    for entry in missing_from_final:
        try:
            i = pdb_of_interest.index()
        except:
            k += 1
    print('Infact, there are', k, 'missing from proteins of interest in the first place')

    sorted_interfaces_file = os.path.join(current_interface_file_path, sorted_file_name)
    # with open(final_file, 'wb') as f:
    #     pickle.dump(final_int_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle_object(interface_sort_d, sorted_interfaces_file, out_path=current_interface_file_path)

    # not_qsbio_file = '200121_pdb_lists/0204_not_qs_bio'  # .pkl'
    # with open(not_qsbio_file, 'wb') as f:
    #     pickle.dump(not_in_qsbio, f, pickle.HIGHEST_PROTOCOL)
    # pickle_object(not_in_qsbio, not_qsbio_file, out_path=current_interface_file_path)

    # NEXT Sorting the interfaces by UniProtID
    # Gather all UniProtIDs from the Representative file specified
    uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdb_of_interest, pdb_uniprot_info)
    print('Total UniProtID\'s added: %d\nTotal without UniProtID: %s' % (len(uniprot_sorted), len(no_unp_code)))  # \nThose missing chain info: %d no_chains,

    uniprot_master = {uniprot_id: process_uniprot_entry(uniprot_id, uniprot_sorted[uniprot_id], pdb_uniprot_info,
                                                        min_resolution_threshold=pdb_resolution_threshold)
                      for uniprot_id in uniprot_sorted}

    # Find the unique UniProtIDs which are representative of the homo-oligomer and unknown PDB interfaces
    all_unknown_bio_interfaces, all_homo_bio_interfaces = {}, {}
    for pdb in interface_sort_d:
        if interface_sort_d[pdb]['unknown_bio'] != set():
            all_unknown_bio_interfaces[pdb] = interface_sort_d[pdb]['unknown_bio']
        if interface_sort_d[pdb]['bio'] != set():
            all_homo_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']

    # all_homo_uniprot_ids, all_unk_uniprot_ids = set(), set()
    all_homo_uniprot_ids, all_unk_uniprot_ids = {}, {}
    uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, unique_pdb_chains = {}, {}, {}
    for uniprot_id in uniprot_master:
        # Get the highest resolution heteromeric biological interfaces
        for pdb in uniprot_master[uniprot_id]['bio']:
            uniprot_hetero_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']
        # Get the highest resolution pdb for each spaces group. These are assumed to comprise separate crystal contacts
        for space_group in uniprot_master[uniprot_id]['space_groups']:
            pdb = uniprot_master[uniprot_id]['space_groups'][space_group]['top']
            uniprot_filtered_xtal_interfaces[pdb] = interface_sort_d[pdb]['xtal']
        # Get all the highest resolution unique pdb and chain
        for pdb_chain in uniprot_master[uniprot_id]['all']:
            pdb = pdb_chain.split('.')[0]
            if pdb == uniprot_master[uniprot_id]['top']:
                unique_pdb_chains[pdb] = pdb_chain.split('.')[1]
                # unique_pdb_chains[uniprot_id] = pdb_chain  # Was this, but the format should match the others...
                break
        # From all unique_pdbs, see if they are in set of homo_ or unknown_bio interfaces
        for pdb in uniprot_master[uniprot_id]['unique_pdb']:
            if pdb in all_homo_bio_interfaces:
                if uniprot_id in all_homo_uniprot_ids:
                    all_homo_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_homo_uniprot_ids[uniprot_id] = {pdb}
                # all_homo_uniprot_ids.add(uniprot_id)
            if pdb in all_unknown_bio_interfaces:
                # all_unk_uniprot_ids.add(uniprot_id)
                if uniprot_id in all_unk_uniprot_ids:
                    all_unk_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_unk_uniprot_ids[uniprot_id] = {pdb}
                
    uniprot_unknown_bio_interfaces, uniprot_homo_bio_interfaces = {}, {}
    for uniprot_id in all_homo_uniprot_ids:
        # for pdb in all_homo_uniprot_ids[uniprot_id]:
        #     if pdb == uniprot_master[uniprot_id]['top']:
        #         uniprot_homo_bio_interfaces[pdb] = all_homo_bio_interfaces[pdb]
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_homo_uniprot_ids[uniprot_id]:
            # pdb: {interface integers} (maybe in string format # Todo change to int()
            uniprot_homo_bio_interfaces[top_pdb] = all_homo_bio_interfaces[top_pdb]
        else:
            highest_res = pdb_resolution_threshold
            highest_res_pdb = None
            for pdb in all_homo_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_homo_bio_interfaces[highest_res_pdb] = all_homo_bio_interfaces[highest_res_pdb]

    for uniprot_id in all_unk_uniprot_ids:
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_unk_uniprot_ids[uniprot_id]:
            # pdb: {interface integers} (maybe in string format # Todo change to int()
            uniprot_unknown_bio_interfaces[top_pdb] = all_unknown_bio_interfaces[top_pdb]
        else:
            highest_res = pdb_resolution_threshold
            highest_res_pdb = None
            for pdb in all_unk_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_unknown_bio_interfaces[highest_res_pdb] = all_unknown_bio_interfaces[highest_res_pdb]

    print(len(uniprot_hetero_bio_interfaces), len(uniprot_filtered_xtal_interfaces), len(uniprot_unknown_bio_interfaces), len(unique_pdb_chains))
    print(len(set(unique_pdb_chains)))

    if write_to_file:
        # All have format {pdb_code: [1, 3], ...} where the list is interface numbers
        pickle_object(uniprot_hetero_bio_interfaces, uniprot_heterooligomer_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(uniprot_filtered_xtal_interfaces, uniprot_xtal_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(unique_pdb_chains, unique_chains_file_name, out_path=current_interface_file_path)

        pickle_object(uniprot_homo_bio_interfaces, uniprot_homooligomer_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(uniprot_unknown_bio_interfaces, uniprot_unknown_bio_interface_file_name,
                      out_path=current_interface_file_path)

    # Old method
    interface_list = [uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, uniprot_unknown_bio_interfaces]
    # final_interface_sort = [[] for i in range(len(interface_list))]
    finalInterfaceDict = {'bio': [], 'xtal': [], 'unknown_bio': []}
    for i in range(len(interface_list)):
        for entry in interface_list[i]:
            for interface in interface_list[i][entry]:
                if i == 0:
                    finalInterfaceDict['bio'].append(entry + '-' + interface)
                elif i == 1:
                    finalInterfaceDict['xtal'].append(entry + '-' + interface)
                elif i == 2:
                    finalInterfaceDict['unknown_bio'].append(entry + '-' + interface)
    #             final_interface_sort[i].append(entry + '-' + interface)

    inverse_final_interface = {}
    i = 0
    total = 0
    for entry in finalInterfaceDict:
        total += len(set(finalInterfaceDict[entry]))
        for pdb in finalInterfaceDict[entry]:
            inverse_final_interface[pdb] = entry
            i += 1
    print('Length of Inverse Final: %d, Actual operations: %d, Total number considered: %d' % (
        len(inverse_final_interface), i, total))
