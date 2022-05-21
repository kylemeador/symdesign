import argparse
import os
import sys
from copy import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from interface_analysis.ParsePisa import retrieve_pisa_file_path, get_complex_interfaces
from PathUtils import pdb_db, pisa_db, qs_bio, qs_bio_monomers_file
from SymDesignUtils import start_log, pickle_object, unpickle, get_all_file_paths, to_iterable, ex_path
from JobResources import FragmentDatabase
import Query.PDB as qPDB
# from PDB import extract_interface
# from Pose import retrieve_pdb_file_path


def pisa_polymer_interface(interface):
    for chain_id in interface['chain_data']:
        if not interface['chain_data'][chain_id]['chain']:  # there is a ligand
            return False
        # else:
    return True


# def return_pdb_interface(pdb_code, interface_id, full_chain=True, db=False):
#     try:
#         # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
#         if not db:   # This makes flexible with MySQL
#             pdb_file_path = retrieve_pdb_file_path(pdb_code, directory=pdb_directory)
#             pisa_file_path = retrieve_pisa_file_path(pdb_code, directory=pisa_directory)
#             if pisa_file_path and pdb_file_path:
#                 # source_pdb = PDB(file=pdb_file_path)
#                 pisa_data = unpickle(pisa_file_path)
#             else:
#                 return None
#         else:
#             print("Connection to MySQL DB not yet supported")
#             exit()
#
#         # interface_data = pisa_data['interfaces']
#         if pisa_polymer_interface(pisa_data['interfaces'][interface_id]):
#             interface_chain_data = pisa_data['interfaces'][interface_id]['chain_data']
#
#             return extract_interface(pdb, interface_chain_data, full_chain=full_chain)
#         else:
#             return None
#
#     except Exception as e:
#         print(e.__doc__)
#         print(e, pdb_code)
#
#         return pdb_code


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


def verify_pisa(pisa_d, pdb_code):
    """From a PISA entry, correct any multimer entries that are missing biological assemblies

    Args:
        pisa_d (dict): {'multimers': {}, 'interfaces': {}, 'chains': {})

    Returns:
        (dict): with multimers updated
    """
    pisa_multimer = pisa_d['multimers']

    if pisa_multimer[(1, 1)] != 'MONOMER':  # if the pdb is not a monomer
        bio_ass = {}
        for set_complex in pisa_multimer:
            biological_assembly = pisa_multimer[set_complex]['pdb_BA']
            if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                if biological_assembly in bio_ass:  # is it already in BA dict?
                    bio_ass[biological_assembly].add(set_complex)
                else:  # add to the BA dict
                    bio_ass[biological_assembly] = {set_complex}

        # or (0, 0) in pisa_multimer:
        if not bio_ass:  # PISA couldn't identify the assembly from crystal data, must be deposited biological assembly?
            print('No bioassembly found for %s' % pdb_code)
            # link the interfaces in the deposited BA complex with the PISA interface number
            pisa_multimer[(0, 0)] = get_complex_interfaces(pdb_code.lower())
            pisa_d['multimers'] = pisa_multimer
            return pisa_d

    return None


def sort_pdb_interfaces_by_contact_type(pisa_d, interface_numbers, assembly_confirmed=None):
    """From a PISA entry, find all PDB interfaces and sort into corresponding Biological assemblies, Crystal Contacts,
    and 'other' contacts. Other contacts are Biological assemblies that are not statistically confirmed, but
    hypothesized as biologically significant. When finding candidate other interfaces, we have to look for alternative
    interface ID's as the multimers and ParsePisa may have a different set of complex biological assembly references.

    Args:
        pisa_d (dict): {'multimers': {}, 'interfaces': {}, 'chains': {})
        interface_numbers (set): The set of PISA interfaces of interest
    Keyword Args:
        assembly_confirmed=None (Iterable[int]): Collection of PDB assemblies (int) that are confirmed oligomers
    Returns:
        (dict[mapping[str,set]]): {'bio': set(), 'xtal': set(), 'unknown_bio': set()}
    """
    pisa_multimer = pisa_d['multimers']
    pisa_interfaces = pisa_d['interfaces']

    final_bio_int, final_xtal_int, final_unknown_bio_int = set(), set(), set()
    if pisa_multimer[(1, 1)] == 'MONOMER':  # we have a monomer and all interface are xtal
        final_xtal_int = interface_numbers
    else:  # the pdb is not a monomer
        bio_ass = {}
        for set_complex, set_complex_data in pisa_multimer.items():
            biological_assembly = set_complex_data.get('pdb_BA')
            if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                if biological_assembly in bio_ass:  # is it already in BA dict?
                    bio_ass[biological_assembly].add(set_complex)
                else:  # add to the BA dict
                    bio_ass[biological_assembly] = {set_complex}

        # Grab all unique interface numbers from PDB assigned BA invoked in multimer entries
        possible_bio_int = set()
        for ba, set_complexes in bio_ass.items():
            for set_complex in set_complexes:
                possible_bio_int.update(pisa_multimer[set_complex]['interfaces'].keys())

        # check for matching interface ids for each interface id. BA may have a different intID than ParsePisa
        other_candidates = set()
        for candidate_int in possible_bio_int:
            for int_type in pisa_interfaces['all_ids']:
                if candidate_int in pisa_interfaces['all_ids'][int_type]:
                    other_candidates.update(pisa_interfaces['all_ids'][int_type])
        all_possible_bio_int = possible_bio_int.union(other_candidates)

        # Todo use the hetero interfaces identified by UniProtID (and chain)
        #  'UniProtID': {'partners': {partner_unpid: {'all': ['1ABC.A' (pdb, chain), ...], 'top': None}, ...}, ...}
        #  to map interface, chain ids as hetero interface_ids
        #  'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        #                    't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104,
        #                    'int_res': {'87': 23.89, '89': 45.01, ...},
        #                 2: ...}},
        # Finally, check if the possible biological_interfaces have confirmed assemblies
        if assembly_confirmed:  # there are assemblies confirmed
            # Only use ba if they are confirmed
            final_unknown_bio_int = set()
            for ba, set_complexes in bio_ass.items():
                if ba not in assembly_confirmed:  # ensure biological assembly is QSBio confirmed assembly
                    for set_complex in set_complexes:
                        final_unknown_bio_int.update(pisa_multimer[set_complex]['interfaces'].keys())
                        # for interface in pisa_multimer[set_complex]['interfaces'].keys():
                        #     all_possible_bio_int.remove(interface)

            # take the intersection (&) of interface_numbers and bio_int to account for the BA id versus ParsePisa id
            final_bio_int = interface_numbers.intersection(all_possible_bio_int.difference(final_unknown_bio_int))
            final_xtal_int = interface_numbers.difference(final_bio_int)
        else:  # the pdb does not have a confirmed assembly
            # take the difference of interface_numbers and all_possible_bio_int
            final_unknown_bio_int = all_possible_bio_int
            # all interface identified, minus possible BA
            final_xtal_int = interface_numbers.difference(final_unknown_bio_int)

    return {'bio': final_bio_int, 'xtal': final_xtal_int, 'unknown_bio': final_unknown_bio_int}


def sort_pdbs_to_uniprot_d(pdbs, pdb_uniprot_d, master_dictionary=None):
    """Add PDB codes to the UniProt ID sorted dictionary and """
    if not master_dictionary:
        unp_master = {}
    else:
        unp_master = master_dictionary
    no_unpid = set()
    for pdb in pdbs:
        for chain in pdb_uniprot_d[pdb]['dbref']:
            if pdb_uniprot_d[pdb]['dbref'][chain]['db'] == 'UNP':  # only add the pdb and chain if dbref is UniProtID
                if 'space' in pdb_uniprot_d[pdb]['struct']:
                    space_group = pdb_uniprot_d[pdb]['struct']['space']
                else:  # Todo NMR, EM? they wouldn't have xtal contacts...
                    space_group = None

                unpid = pdb_uniprot_d[pdb]['dbref'][chain]['accession']
                if unpid in unp_master:
                    unp_master[unpid]['all'].add('%s.%s' % (pdb, chain))
                    unp_master[unpid]['unique_pdb'].add(pdb)
                    if space_group:
                        if space_group not in unp_master[unpid]['space_groups']:
                            unp_master[unpid]['space_groups'][space_group] = {'all': {pdb}}  # , 'top': None}
                        else:
                            unp_master[unpid]['space_groups'][space_group]['all'].add(pdb)
                else:
                    if space_group:
                        entry = {space_group: {'all': set(pdb)}}  # , 'top': None}}
                    else:
                        entry = {}
                    unp_master[unpid] = {'all': {'%s.%s' % (pdb, chain)}, 'unique_pdb': {pdb}, 'space_groups': entry}
                    # , 'top': None, 'hetero': set(), 'partners': {},
            else:
                no_unpid.add('%s.%s' % (pdb, chain))

    return unp_master, no_unpid


def add_partners_to_uniprot_entry(uniprot_id, unp_d, pdb_uniprot_info):
    unp_d['partners'] = {}
    # Check for pdbs which comprise Self and other Uniprot IDs. These represent distinct heteromeric interfaces
    for pdb in unp_d['unique_pdb']:
        if len(pdb_uniprot_info[pdb]['dbref']) > 1:  # Check if there is more than one chain in the pdb
            for chain in pdb_uniprot_info[pdb]['dbref']:
                if pdb_uniprot_info[pdb]['dbref'][chain]['db'] == 'UNP':  # only add if there is corresponding UniProtID
                    partner_unpid = pdb_uniprot_info[pdb]['dbref'][chain]['accession']
                    if partner_unpid != uniprot_id:  # only get other Uniprot IDs
                        if partner_unpid in unp_d['partners']:
                            unp_d['partners'][partner_unpid]['all'].append('%s.%s' % (pdb, chain))
                        else:
                            unp_d['partners'][partner_unpid] = {'all': ['%s.%s' % (pdb, chain)], 'top': None}


def find_representative_pdbs_for_uniprot_entry(unp_d, pdb_uniprot_info, min_resolution_threshold=3.):
    """Filter a UniProt ID/PDB dictionary for PDB entries with unique interfacial contacts including biological,
        crystalline, and hetero partners. Finds the highest resolution structure for each category.
    """
    # Find Uniprot ID with highest resolution and save PDB Code
    min_res = copy(min_resolution_threshold)
    unp_d['top'] = None
    for pdb_code in unp_d['unique_pdb']:
        pdb_res = pdb_uniprot_info[pdb_code]['res']
        if pdb_res <= min_res:  # structural resolution is less than threshold
            unp_d['top'], min_res = pdb_code, pdb_res

    # unp_d['top'] = top_pdb
    # Add to Biological Assembly list
    # unp_master[unpid]['bio'].add(top_pdb)  # Todo why? Could be a monomer

    # Add the highest resolution UniProt Partners to the Hetero Biological Assemblies
    unp_d['hetero'] = set()
    for partner, partner_data in unp_d['partners'].items():
        min_partner_res = copy(min_resolution_threshold)
        top_partner = None
        for pdb_code_chain in partner_data['all']:
            pdb_code, chain = pdb_code_chain.split('.')
            partner_res = pdb_uniprot_info[pdb_code]['res']
            if partner_res <= min_partner_res:  # always puts last pdb chain from [partners][all] if > 1 chain in pdb
                top_partner, min_partner_res = pdb_code_chain, partner_res
        if top_partner:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['partners'][partner]['top'] = top_partner
            unp_d['hetero'].add(top_partner)

    # Save highest Resolution PDB Codes for each Space Group
    for cryst, cryst_data in unp_d['space_groups']:
        min_xtal_res = copy(min_resolution_threshold)
        top_xtal_pdb = None
        for pdb_code in cryst_data['all']:
            xtal_res = pdb_uniprot_info[pdb_code]['res']
            if xtal_res <= min_xtal_res:
                top_xtal_pdb, min_xtal_res = pdb_code, xtal_res
        if top_xtal_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['space_groups'][cryst]['top'] = top_xtal_pdb

    return unp_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Chain/Chain Interfaces from a PDB or PDB Library\n')
    parser.add_argument('-f', '--file_list', type=str,
                        help='%s. Can be newline or comma separated.' % ex_path('pdblist.file'))
    parser.add_argument('-d', '--download', type=bool, help='Whether files should be downloaded. Default=False',
                        default=False)
    parser.add_argument('-p', '--input_pdb_directory', type=str, help='Where should reference PDB files be found? '
                                                                      'Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input_pisa_directory', type=str, help='Where should reference PISA files be found? '
                                                                       'Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output_directory', type=str, help='Where should interface files be saved?')
    parser.add_argument('-q', '--query_web', action='store_true', help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    parser.add_argument('-F', '--fix_pisa', action='store_true', help='Should the pisa files be corrected?')
    args = parser.parse_args()

    logger = start_log(name=os.path.basename(__file__), propagate=True)

    # Input data/files
    # pdb_directory = '/databases/pdb'
    # pisa_directory = '/home/kmeador/yeates/fragment_database/all/pisa_files'

    pdb_directory = pdb_db  # ends with .ent not sub_directoried
    pisa_directory = pisa_db  # subdirectoried here
    if args.fix_pisa:
        all_pisa_files = get_all_file_paths(pisa_directory, extension='.pkl')

        # for pdb_code in all_pisa_codes:
        #     pisa_path = retrieve_pisa_file_path(pdb_code)
        for i, pisa_path in enumerate(all_pisa_files, 1):
            print(i)
            # if pisa_path:
            pisa_d = unpickle(pisa_path)
            pdb_code = os.path.basename(pisa_path)[:4]
            pisa_d = verify_pisa(pisa_d, pdb_code)
            if pisa_d:  # we modified the pisa_d and should re-pickle
                pickle_object(pisa_d, pisa_path, out_path='')
        exit()
    current_interface_file_path = '/yeates1/kmeador/fragment_database/current_pdb_lists/'  # Todo parameterize

    # pdb_directory = '/yeates1/kmeador/fragment_database/all_pdb'  # sub directoried
    # pdb_directory = 'all_pdbs'

    # Variables
    representative_metric = 'res'
    pdb_resolution_threshold = 3.
    write_to_file = True  # TODO parameterize. MySQL DB option would make this False?
    write_to_db = True  # Todo parameterize

    pdbs_of_interest_file_name = 'under3A_protein_no_multimodel_no_mmcif_no_bad_pisa.txt'  # Todo parameterize
    all_protein_file = os.path.join(current_interface_file_path, pdbs_of_interest_file_name)
    if args.query_web:
        pdbs_of_interest = qPDB.retrieve_pdb_entries_by_advanced_query()
    else:
        pdbs_of_interest = to_iterable(all_protein_file, ensure_file=True)

    # Retrieve the verified set of biological assemblies
    # qsbio_file_name = qs_bio
    # qsbio_file = os.path.join(current_interface_file_path, qsbio_file_name)
    # qsbio_confirmed_d = unpickle(qsbio_file)
    # qsbio_monomers_file_name = 'QSbio_Monomers.csv'
    # qsbio_monomers_file = os.path.join(current_interface_file_path, qsbio_monomers_file_name)
    # qsbio_monomers = to_iterable(qsbio_monomers_file)
    # TODO add the list of QSBio confirmed hetero oligomers
    qsbio_confirmed_d = unpickle(qs_bio)  # Todo make each list of interface ids a set()
    qsbio_monomers = to_iterable(qs_bio_monomers_file, ensure_file=True)  # Todo remove monomers from confirmed assemblies in source
    qsbio_monomers_d = {}
    for pdb_assembly in qsbio_monomers:
        pdb, assembly = pdb_assembly.split('_')
        # if pdb in qsbio_monomers_d:
        #     qsbio_monomers_d[pdb].add(assembly)
        monomer_set = qsbio_monomers_d.get(pdb)
        if monomer_set:
            monomer_set.add(assembly)
        else:
            qsbio_monomers_d[pdb] = set(assembly)

    logger.info('The number of monomers found in %s: %d' % (qs_bio_monomers_file, len(qsbio_monomers_d)))
    # remove monomeric assemblies from qsbio_confirmed_oligomers
    for pdb, assemblies in qsbio_monomers_d.items():
        for assembly in assemblies:
            if pdb in qsbio_confirmed_d:
                if assembly in qsbio_confirmed_d[pdb]:
                    qsbio_confirmed_d[pdb].remove(assembly)
        if not qsbio_confirmed_d[pdb]:  # if the values list is now empty, remove the pdb from the dictionary
            qsbio_confirmed_d.pop(pdb)

    # Current, DEPRECIATE!
    interfaces_dir = '/yeates1/kmeador/fragment_database/all_interfaces'
    all_interface_pdb_paths = get_all_file_paths(interfaces_dir, extension='.pdb')
    pdb_interface_codes = [file_ext_split[0] for file_ext_split in map(os.path.splitext,
                                                                       map(os.path.basename, all_interface_pdb_paths))]
    pdb_interface_d = set_up_interface_dict(pdb_interface_codes)
    # contains {pdb names: [interface ids]}

    # Optimal
    pdb_interface_file_name = 'PDBInterfaces.pkl'  # Todo parameterize
    pdb_interface_file = os.path.join(current_interface_file_path, pdb_interface_file_name)
    sorted_file_name = 'PDBInterfacesSortedByType.pkl'  # Todo parameterize
    sorted_interfaces_file = os.path.join(current_interface_file_path, sorted_file_name)
    missing_pisa_file_name = 'MissingPISA.txt'

    if os.path.exists(sorted_interfaces_file) and os.path.exists(pdb_interface_file):
        interface_sort_d = unpickle(sorted_interfaces_file)
        pdb_interface_d = unpickle(pdb_interface_file)
        missing_pisa = set(pdbs_of_interest) - set(pdb_interface_d.keys())
    else:  # make the pdb - interfaces dictionaries
        interface_sort_d, pdb_interface_d, missing_pisa = {}, {}, set()
        if args.query_web:
            dummy = None  # TODO combine all pdb codes into chunks of PISA requests using pp.download_pisa()

        else:
            for pdb_code in pdbs_of_interest:
                pisa_path = retrieve_pisa_file_path(pdb_code, directory=pisa_directory)
                if pisa_path:
                    pisa_d = unpickle(pisa_path)  # Todo try except FileNotFoundError
                    interface_data = pisa_d['interfaces']
                    interface_ids = list(interface_data.keys())  # interface_ids are only from interface type rep
                    interface_ids.remove('all_ids')
                    # Remove ligand interfaces from PISA interface_data
                    polymer_interface_ids = {int_id for int_id in interface_ids
                                             if pisa_polymer_interface(interface_data[int_id])}
                    pdb_interface_d[pdb_code] = polymer_interface_ids

                    assembly = None
                    if pdb_code in qsbio_confirmed_d:
                        assembly = qsbio_confirmed_d[pdb_code]
                    interface_sort_d[pdb_code] = sort_pdb_interfaces_by_contact_type(pisa_d, polymer_interface_ids,
                                                                                     assembly_confirmed=assembly)
                else:
                    missing_pisa.add(pdb_code)
        # Save the objects
        # {1AB3: {1, 3, 5}, ...}
        pickle_object(pdb_interface_d, pdb_interface_file, out_path='')
        # {1AB3: {'bio': {}, 'xtal': {3, 5}, 'unknown_bio': {1}, ...}
        sorted_interfaces_file = pickle_object(interface_sort_d, sorted_interfaces_file, out_path='')

    # Remove missing pdbs from pdb_of_interest
    pdbs_of_interest = set(pdbs_of_interest) - missing_pisa
    missing_pisa_file = os.path.join(current_interface_file_path, missing_pisa_file_name)
    with open(missing_pisa_file, 'w') as f:
        f.write('\n'.join(missing_pisa))
    logger.info('There were %d PDB\'s missing the corresponding PISA data. Missing PDB entry ID\'s can be found at %s'
                % (len(missing_pisa), missing_pisa_file))

    # First, sort all possible interfaces from PDB PISA to respective biological, crystal, or non-determined sets
    # sorted_file_name = 'PDBInterfacesSorted'  # Todo parameterize
    # sorted_interfaces_file = os.path.join(current_interface_file_path, sorted_file_name)
    # if os.path.exists(sorted_interfaces_file):
    #     interface_sort_d = unpickle(sorted_interfaces_file)
    # else:
    # interface_sort_d = {}
    # missing_pisa_paths = []
    # for pdb_code in pdb_interface_d:  # {1AB3: {1, 3, 5}, ...}
    #     pisa_path = retrieve_pisa_file_path(pdb_code)
    #     if pisa_path:
    #         pisa_d = unpickle(pisa_path)
    #         interface_sort_d[pdb_code] = sort_pdb_interfaces_by_contact_type(pisa_d, pdb_interface_d[pdb_code],
    #                                                                          set(qsbio_confirmed_d[pdb_code]))
    #     else:
    #         missing_pisa_paths.append(pdb_code)
    # sorted_interfaces_file = pickle_object(interface_sort_d, sorted_interfaces_file, out_path='')

    # OLD - Depreciate
    old_pdb_uniprot_file_name = '200206_MostCompleteAllPDBSpaceGroupUNPResInfo.pkl'  # Todo parameterize
    # 200121 has 'ref' and 'path': '/home/kmeador/yeates/fragment_database/all/all_pdbs/kn/3KN7.pdb', not entity
    #
    # Generate all PDB Metadata with format
    #  {'entity': {1: ['A']}, 'res': 2.6, 'ref': {'A': {'db': 'UNP', 'accession': 'P35755'})
    #   'cryst': {'space': 'P 21 21 2', 'a_b_c': (106.464, 75.822, 34.109), 'ang_a_b_c': (90.0, 90.0, 90.0)}}
    # will be 'dbref' instead of 'ref'

    pdb_uniprot_file_name = 'PDBMetadata.pkl'
    all_pdb_uniprot_file = os.path.join(current_interface_file_path, pdb_uniprot_file_name)
    if os.path.exists(all_pdb_uniprot_file):  # retrieve the pdb, DBreference, resolution, and crystal dictionary
        pdb_uniprot_info = unpickle(all_pdb_uniprot_file)
        pdb_uniprot_info.update({pdb_code: qPDB.get_pdb_info_by_entry(pdb_code)
                                 for pdb_code in set(pdbs_of_interest).difference(pdb_uniprot_info.keys())})
    else:
        # if args.query_web:  # Retrieve from the PDB web
        pdb_uniprot_info = {pdb_code: qPDB.get_pdb_info_by_entry(pdb_code) for pdb_code in pdbs_of_interest}
        # else:  # From the database of files
        #     pdb_uniprot_info = {}
        #     for pdb_code in pdbs_of_interest:
        #         pdb = PDB(file=retrieve_pdb_file_path(pdb_code, directory=pdb_directory), coordinates_only=False)
        #         pdb_uniprot_info[pdb_code] = {'entity': pdb.entities, 'cryst': pdb.cryst, 'ref': pdb.dbref,
        #                                       'res': pdb.res}

        pickle_object(pdb_uniprot_info, all_pdb_uniprot_file, out_path='')

    # Output data/files for the interface and chain sorting
    uniprot_master_file_name = 'UniProtPDBMapping'
    uniprot_heterooligomer_interface_file_name = 'UniquePDBHeteroOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_homooligomer_interface_file_name = 'UniquePDBHomoOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_unknown_bio_interface_file_name = 'UniquePDBUnknownOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_xtal_interface_file_name = 'UniquePDBXtalInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    unique_chains_file_name = 'UniquePDBChains'  # was 200121_FinalIntraDict  # Todo parameterize

    # TODO Back-end: Figure out how to convert these sets into MySQL tables so that the PDB files present are
    #  easily identified based on their assembly. Write load PDB from MySQL function. Also ensure I have a standard
    #  query format to grab different table schemas depending on the step of the fragment processing.

    # Next, find all those with homomeric biological interfaces according to PISA and QSBio, as well as monomers
    # # all_biological = {pdb for pdb in interface_sort_d if interface_sort_d[pdb]['homo'] != set()}
    # all_missing_biological = set(interface_sort_d) - \
    #                          {pdb for pdb in interface_sort_d if interface_sort_d[pdb]['homo'] != set()}  # all_biological
    # proposed_qsbio_monomer_set = list(qsbio_monomers_d.keys())
    # {pdb for pdb in all_missing_biological if pdb in qsbio_confirmed_d}
    # assert len(proposed_qsbio_monomer_set - qsbio_monomers) == 0, 'The set of proposed monomers has PDB\'s that are '\
    #                                                               'not monomers!'

    # NEXT Sorting the interfaces by UniProtID
    # Gather all UniProtIDs from the pdbs_of_interest
    uniprot_master_file = os.path.join(current_interface_file_path, uniprot_master_file_name)
    if os.path.exists(uniprot_master_file):  # If this file already exists, then we should add to it
        uniprot_master = unpickle(uniprot_master_file)
        # Next, add any additional pdb ids to the uniprot_master
        # uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdbs_of_interest, pdb_uniprot_info,
        #                                                      master_dictionary=uniprot_master)
    else:
        uniprot_master = None
    uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdbs_of_interest, pdb_uniprot_info,
                                                         master_dictionary=uniprot_master)
    logger.info('Total UniProtID\'s added: %d\nTotal without UniProtID: %s' % (len(uniprot_sorted), len(no_unp_code)))
    # sort all interfaces by UniProt ID to account for duplicated PDB Structures, i.e. 1ABC and 2XYZ are protein P012345
    uniprot_partners = {uniprot_id: add_partners_to_uniprot_entry(uniprot_id, uniprot_entry, pdb_uniprot_info)
                        for uniprot_id, uniprot_entry in uniprot_sorted.items()}

    uniprot_master = {uniprot_id:
                      find_representative_pdbs_for_uniprot_entry(uniprot_entry, pdb_uniprot_info,
                                                                 min_resolution_threshold=pdb_resolution_threshold)
                      for uniprot_id, uniprot_entry in uniprot_partners.items()}

    sorted_interfaces_file = pickle_object(uniprot_master, uniprot_master_file, out_path='')

    # Next, search for specific types of interfaces from the Master UniProt sorted dictionary
    # Find unique UniProtIDs which are representative of the homo-oligomer and unknown PDB interfaces
    unknown_bio_interfaces = {pdb: interface_sort_d[pdb]['unknown_bio'] for pdb in interface_sort_d
                              if interface_sort_d[pdb]['unknown_bio']}
    # each of these has the data type - {pdb: set()}
    bio_interfaces = {pdb: interface_sort_d[pdb]['bio'] for pdb in interface_sort_d
                      if interface_sort_d[pdb]['bio']}
    # ^ this contains hetero and homo interfaces
    # TODO need other source than PISA for interfaces
    #  if EM or NMR structures are added we won't have xtal contacts.
    #  if interface_sort_d[pdb]['xtal']:  # check is there is an empty set()
    #      bio_interfaces[pdb] = interface_sort_d[pdb]['bio']

    # homo_uniprot_ids, unknown_uniprot_ids = set(), set()
    homo_uniprot_ids, unknown_uniprot_ids = {}, {}
    uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, unique_pdb_chains = {}, {}, {}
    for uniprot_id, data in uniprot_master.items():
        # Get the highest resolution heteromeric biological interfaces
        # Todo must reconcile the use of hetero interfaces with the usage of QSBio confirmed hetero interfaces to pull
        #  out those hetero that are confirmed from those that are not. I think the entire parsing script needs an
        #  overhaul to adapt to the different types of interfaces
        for pdb_code in data['hetero']:
            uniprot_hetero_bio_interfaces[pdb_code] = interface_sort_d[pdb_code]['unknown_bio']
            # uniprot_hetero_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']  # This has no heteromers in it
        # Get the highest resolution pdb for each space group along with the interfaces.
        # Each space group is assumed to comprise separate crystal contacts
        for space_group, space_group_data in data['space_groups'].items():
            pdb_code = space_group_data['top']
            uniprot_filtered_xtal_interfaces[pdb_code] = interface_sort_d[pdb_code]['xtal']
        # Get all the highest resolution unique pdb and chain
        for pdb_code_chain in data['all']:
            pdb_code, chain = pdb_code_chain.split('.')
            if pdb_code == data['top']:
                # unique_pdb_chains[uniprot_id] = pdb_chain  # Was this, but the format should match the others...
                unique_pdb_chains[pdb_code] = chain
                break
        # From all unique_pdbs, see if they are in set of bio or unknown_bio interfaces
        for pdb_code in data['unique_pdb']:
            if pdb_code in bio_interfaces:
                if uniprot_id in homo_uniprot_ids:
                    homo_uniprot_ids[uniprot_id].add(pdb_code)
                else:
                    homo_uniprot_ids[uniprot_id] = set(pdb_code)
            if pdb_code in unknown_bio_interfaces:
                if uniprot_id in unknown_uniprot_ids:
                    unknown_uniprot_ids[uniprot_id].add(pdb_code)
                else:
                    unknown_uniprot_ids[uniprot_id] = set(pdb_code)

    # For each UniProt ID, sort the corresponding set of PDB's to find the representative PDB
    # Perform sort for each homo_oligomer and unknown_oligomer
    uniprot_homo_bio_interfaces, uniprot_unknown_bio_interfaces = {}, {}
    biological_interface_types = [(homo_uniprot_ids, uniprot_homo_bio_interfaces, bio_interfaces),
                                  (unknown_uniprot_ids, uniprot_unknown_bio_interfaces, unknown_bio_interfaces)]
    for bio_interface_uniprot_id_dict, uniprot_id_bio_dict, interface_type_number_sets in biological_interface_types:
        for uniprot_id, pdb_codes in bio_interface_uniprot_id_dict.items():
            top_pdb = uniprot_master[uniprot_id]['top']
            if top_pdb in pdb_codes:
                uniprot_id_bio_dict[top_pdb] = interface_type_number_sets[top_pdb]
            else:
                best_metric = copy(pdb_resolution_threshold)
                highest_pdb_code = None
                for pdb_code in pdb_codes:
                    pdb_metric = pdb_uniprot_info[pdb_code][representative_metric]
                    if pdb_metric < best_metric:
                        best_metric, highest_pdb_code = pdb_metric, pdb_code
                # if highest_res_pdb:  # this can't possibly be none as their is always one pdb code
                uniprot_id_bio_dict[highest_pdb_code] = interface_type_number_sets[highest_pdb_code]

    logger.info('Found a total of:\nHeteromeric interfaces = %d\nHomomeric interfaces = %d\nUnknown interfaces = %d\n'
                'Crystalline interfaces = %d\nUnique PDB Chains = %d' %
                (len(uniprot_hetero_bio_interfaces), len(uniprot_homo_bio_interfaces),
                 len(uniprot_unknown_bio_interfaces), len(uniprot_filtered_xtal_interfaces), len(unique_pdb_chains)))

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
    if write_to_db:
        fragdb = FragmentDatabase(sql=True)  # Todo this isn't ready to write
        # fragdb.start_mysql_connection()
        table_name = 'pdb_interfaces'
        columns = ('pdb', 'pisa_interface_number', 'interface_type', 'ch1', 'ch2', 'ch1_rot_id', 'ch1_tx',
                   'ch2_rot_id', 'ch2_tx')  # 'id' <-- auto incremented
        interface_type_key = ['hetero', 'xtal', 'intra', 'homo', 'unknown_bio']
        interface_types = [uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces,
                           unique_pdb_chains, uniprot_homo_bio_interfaces, uniprot_unknown_bio_interfaces]
        # interface_types = {'hetero': uniprot_hetero_bio_interfaces, 'xtal': uniprot_filtered_xtal_interfaces,
        #                    'intra': unique_pdb_chains, 'homo': uniprot_homo_bio_interfaces,
        #                    'unknown_bio': uniprot_unknown_bio_interfaces}
        identity_matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        null_tx = [0., 0., 0.]
        data_packets = []
        inserted, missing_insert = {}, {}
        for interface_type_idx, pdb_interface_ids in enumerate(interface_types):
            inserted[interface_type_idx] = []
            missing_insert[interface_type_idx] = []
            if interface_type_idx == 'intra':
                # interface_id is the chain, so pisa_interface_number = 0, chain1 == chain2,
                data = [(pdb_code, 0, interface_type_idx, interface_id, interface_id,
                         identity_matrix, null_tx, identity_matrix, null_tx)
                        for pdb_code, interface_id in pdb_interface_ids.items()]
                try:
                    fragdb.fragdb.insert_multiple(table_name, columns, data)
                    inserted[interface_type_idx] = [pdb_interface_ids.values()]
                except Error as err:
                    if err.errno == 1062:
                        logger.error('Something related to multimodels?: %s' % err)
                    else:
                        logger.error('Something went wrong: %s' % err)
                    missing_insert[interface_type_idx] = [pdb_interface_ids.values()]
                continue
            else:
                for pdb_code, interface_ids in pdb_interface_ids.items():
                    # if pisa_file_path:
                    pisa_data = unpickle(retrieve_pisa_file_path(pdb_code, directory=pisa_directory))
                    data = []
                    for interface_id in interface_ids:
                        # if pisa_polymer_interface(pisa_data['interfaces'][interface_id]):
                        interface_chain_data = pisa_data['interfaces'][interface_id]['chain_data']
                        # 'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        #                    't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104,
                        #                    'int_res': {'87': 23.89, '89': 45.01, ...}},
                        #                2: ...},
                        chains = [{'chain': chain_d['chain'], 'rot': chain_d['r_mat'], 'tx': chain_d['t_vec']}
                                  for chain_d in interface_chain_data.values()]

                        data.append((pdb_code, interface_id, interface_type_idx, chains[0]['chain'], chains[1]['chain'],
                                     chains[0]['rot'], chains[0]['tx'], chains[1]['rot'], chains[1]['tx']))
                    # data_packets.append(data)
                    # for packet in data_packets:
                    try:
                        fragdb.fragdb.insert_multiple(table_name, columns, data)
                        inserted[interface_type_idx].append((pdb_code, interface_ids))
                    except Error as err:
                        if err.errno == 1062:
                            logger.error('Something related to multimodels?: %s' % err)
                        else:
                            logger.error('Something went wrong: %s' % err)
                        missing_insert[interface_type_idx].append((pdb_code, interface_ids))

    # # Old method
    # interface_list = [uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, uniprot_unknown_bio_interfaces]
    # # final_interface_sort = [[] for i in range(len(interface_list))]
    # finalInterfaceDict = {'bio': [], 'xtal': [], 'unknown_bio': []}
    # for i in range(len(interface_list)):
    #     for entry in interface_list[i]:
    #         for interface in interface_list[i][entry]:
    #             if i == 0:
    #                 finalInterfaceDict['bio'].append(entry + '-' + interface)
    #             elif i == 1:
    #                 finalInterfaceDict['xtal'].append(entry + '-' + interface)
    #             elif i == 2:
    #                 finalInterfaceDict['unknown_bio'].append(entry + '-' + interface)
    # #             final_interface_sort[i].append(entry + '-' + interface)
    #
    # inverse_final_interface = {}
    # i = 0
    # total = 0
    # for entry in finalInterfaceDict:
    #     total += len(set(finalInterfaceDict[entry]))
    #     for pdb in finalInterfaceDict[entry]:
    #         inverse_final_interface[pdb] = entry
    #         i += 1
    # print('Length of Inverse Final: %d, Actual operations: %d, Total number considered: %d' % (
    #     len(inverse_final_interface), i, total))
