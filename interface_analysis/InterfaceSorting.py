import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from copy import deepcopy
from itertools import chain as iter_chain
import requests
import pandas as pd

from PDB import PDB
import ParsePisa as pp
from SymDesignUtils import start_log, pickle_object, unpickle, get_all_pdb_file_paths, to_iterable, \
    download_pisa, pisa_ref_d
from Pose import retrieve_pdb_file_path

import QueryProteinData.QueryPDB as qPDB

# Globals
pisa_type_extensions = {'multimers': '.xml', 'interfaces': '.xml', 'multimer': '.pdb', 'pisa': '.pkl'}


def pisa_polymer_interface(interface):
    for chain_id in interface['chain_data']:
        if not interface['chain_data'][chain_id]['chain']:
            return False
        else:
            return True


def return_pdb_interface(pdb_code, interface_id, full_chain=True, db=False):
    try:
        # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
        if not db:   # This makes flexible with MySQL
            pdb_file_path = retrieve_pdb_file_path(pdb_code, directory=pdb_directory)
            pisa_file_path = retrieve_pisa_file_path(pdb_code, directory=pisa_directory)
            if pisa_file_path and pdb_file_path:
                source_pdb = PDB(file=pdb_file_path)
                pisa_data = unpickle(pisa_file_path)
            else:
                return None
        else:
            print("Connection to MySQL DB not yet supported")
            exit()

        # interface_data = pisa_data['interfaces']
        if pisa_polymer_interface(pisa_data['interfaces'][interface_id]):
            interface_chain_data = pisa_data['interfaces'][interface_id]['chain_data']

            return extract_interface(pdb, interface_chain_data, full_chain=full_chain)
        else:
            return None

    except Exception as e:
        print(e.__doc__)
        print(e, pdb_code)

        return pdb_code


def extract_interface(pdb, chain_data_d, full_chain=True):
    """
    'interfaces': {interface_ID: {interface stats, {chain data}}, ...}
        Ex: {1: {'occ': 2, 'area': 998.23727478, 'solv_en': -11.928783903, 'stab_en': -15.481081211,
             'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104, 'int_res': {'87': 23.89, '89': 45.01, ...},
                            2: ...}},
             2: {'occ': ..., },
             'all_ids': {interface_type: [interface_id1, matching_id2], ...}
            } interface_type and id connect the interfaces that are the same, but present in multiple PISA complexes

    """
    # if one were trying to map the interface fragments created in a fragment extraction back to the pdb, they would
    # want to use the interface in interface_data and the chain id (renamed from letter to ID #) to find the chain and
    # the translation from the pisa.xml file
    # pdb_code, subdirectory = return_and_make_pdb_code_and_subdirectory(pdb_file_path)
    # out_path = os.path.join(os.getcwd(), subdirectory)
    # try:
    #     # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
    #     # This makes flexible with MySQL
    #     source_pdb = PDB(file=pdb_file_path)
    #     pisa_data = unpickle(pisa_file_path)  # Get PISA data
    #     interface_data = pisa_data['interfaces']
    #     # interface_data, chain_data = pp.parse_pisa_interfaces_xml(pisa_file_path)
    #     for interface_id in interface_data:
    #         if not interface_id.is_digit():  # == 'all_ids':
    #             continue
    # interface_pdb = PDB.PDB()
    temp_names = ('.', ',')
    interface_chain_pdbs = []
    temp_chain_d = {}
    for temp_name_idx, chain_id in enumerate(chain_data_d):
        # chain_pdb = PDB.PDB()
        chain = chain_data_d[chain_id]['chain']
        # if not chain:  # for instances of ligands, stop process, this is not a protein-protein interface
        #     break
        # else:
        if full_chain:  # get the entire chain
            interface_atoms = deepcopy(pdb.get_chain_atoms(chain))
        else:  # get only the specific residues at the interface
            residue_numbers = chain_data_d[chain_id]['int_res']
            interface_atoms = pdb.chain(chain).get_residue_atoms(residue_numbers)
            # interface_atoms = []
            # for residue_number in residues:
            #     residue_atoms = pdb.get_residue_atoms(chain, residue_number)
            #     interface_atoms.extend(deepcopy(residue_atoms))
            # interface_atoms = list(iter_chain.from_iterable(interface_atoms))
        chain_pdb = PDB(atoms=deepcopy(interface_atoms))
        # chain_pdb.read_atom_list(interface_atoms)

        rot = chain_data_d[chain_id]['r_mat']
        trans = chain_data_d[chain_id]['t_vec']
        chain_pdb.apply(rot, trans)
        chain_pdb.rename_chain(chain, temp_names[temp_name_idx])  # ensure that chain names are not the same
        temp_chain_d[temp_names[temp_name_idx]] = str(chain_id)
        interface_chain_pdbs.append(chain_pdb)
        # interface_pdb.read_atom_list(chain_pdb.atoms)

    interface_pdb = PDB(atoms=iter_chain.from_iterable([chain_pdb.get_atoms() for chain_pdb in interface_chain_pdbs]))
    if len(interface_pdb.chain_id_list) == 2:
        for temp_name in temp_chain_d:
            interface_pdb.rename_chain(temp_name, temp_chain_d[temp_name])

    return interface_pdb


# Todo finalize PISA path, move to ParsePisa.py?
def retrieve_pisa_file_path(pdb_code, directory='/home/kmeador/yeates/fragment_database/all/pisa_files',
                            pisa_data_type='pisa'):
    """Returns the PISA path that corresponds to the PDB code from the local PISA Database.
    Attempts a download if the files are not found
    """
    if pisa_data_type in pisa_ref_d:  # type_extensions:
        pdb_code = pdb_code.upper()
        sub_dir = pdb_code[1:3].lower()  # Todo If subdir is used in directory structure
        root_path = os.path.join(directory, sub_dir)
        specific_file = ' %s_%s' % (pdb_code, pisa_ref_d[pisa_data_type]['ext'])  # file_type, type_extensions[file_type])
        if os.path.exists(os.path.join(root_path, specific_file)):
            return os.path.join(root_path, specific_file)
        else:  # attempt to make the file if not pickled or download if requisite files don't exist
            if pisa_data_type == 'pisa':
                if extract_pisa_files_and_pickle(root_path, pdb_code):  # return True if requisite files found
                    return os.path.join(root_path, specific_file)
                else:  # try to download required files, they were not found
                    logger.info('Attempting to download PISA files')
                    for pisa_data_type in pisa_ref_d:
                        download_pisa(pdb, pisa_data_type, out_path=directory)
                    if extract_pisa_files_and_pickle(root_path, pdb_code):  # return True if requisite files found
                        return os.path.join(root_path, specific_file)
            else:
                download_pisa(pdb, pisa_data_type, out_path=directory)

    return None  # if Download failed, or the files are not found


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


def sort_pdb_interfaces_by_contact_type(pisa_d, interface_number_set, assembly_confirmed):
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
            for int_type in pisa_interfaces['all_ids']:
                if candidate_int in pisa_interfaces['all_ids'][int_type]:  # check if the interface has matching interfaces
                    other_candidates += pisa_interfaces['all_ids'][int_type]
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

    return {'bio': final_bio_int, 'xtal': final_xtal_int, 'unknown_bio': final_unknown_bio_int}


def sort_pdbs_to_uniprot_d(pdbs, pdb_uniprot_d, master_dictionary=None):
    """Add PDB codes to the UniProt ID sorted dictionary and """
    if not master_dictionary:
        unp_master = {}
    else:
        unp_master = master_dictionary
    no_unpid = {}
    for pdb in pdbs:
        for chain in pdb_uniprot_d[pdb]['ref']:
            if pdb_uniprot_d[pdb]['ref'][chain]['db'] == 'UNP':  # only add the pdb and chain if corresponding UniProtID
                unpid = pdb_uniprot_d[pdb]['ref'][chain]['accession']
                if unpid in unp_master:
                    unp_master[unpid]['all'].add('%s.%s' % (pdb, chain))
                    unp_master[unpid]['unique_pdb'].add(pdb)
                    space_group = pdb_uniprot_d[pdb]['cryst']['space']  # Todo ensure can be none
                    if not space_group:
                        pass
                    else:
                        if space_group not in unp_master[unpid]['space_groups']:
                            unp_master[unpid]['space_groups'][space_group] = {'all': {pdb}, 'top': None}
                        else:
                            unp_master[unpid]['space_groups'][space_group]['all'].add(pdb)
                else:
                    unp_master[unpid] = {'all': {'%s.%s' % (pdb, chain)}, 'unique_pdb': {pdb}, 'top': None,
                                         'bio': set(), 'partners': {}, 'space_groups': {
                            pdb_uniprot_d[pdb]['cryst']['space']: {'all': {pdb}, 'top': None}}}
            else:
                no_unpid.add('%s.%s' % (pdb, chain))

    return unp_master, no_unpid


def process_uniprot_entry(uniprot_id, unp_d, pdb_uniprot_info, min_resolution_threshold=3.0):
    """Filter a UniProt ID/PDB dictionary for PDB entries with unique interfacial contacts including biological,
        crystalline, and hetero partners. Finds the highest resolution structure for each category.
    """

    min_res = deepcopy(min_resolution_threshold)
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
        min_partner_res = deepcopy(min_resolution_threshold)
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
        min_xtal_res = deepcopy(min_resolution_threshold)
        top_xtal_pdb = None
        for pdb in unp_d['space_groups'][cryst]['all']:
            if pdb_uniprot_info[pdb]['res'] <= min_xtal_res:
                top_xtal_pdb = pdb
                min_xtal_res = pdb_uniprot_info[pdb]['res']
        if top_xtal_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['space_groups'][cryst]['top'] = top_xtal_pdb

    return unp_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Chain/Chain Interfaces from a PDB or PDB Library\n')
    parser.add_argument('-f', '--file_list', type=str, help='path/to/pdblist.file. Can be newline or comma separated.')
    parser.add_argument('-d', '--download', type=bool, help='Whether files should be downloaded. Default=False',
                        default=False)
    parser.add_argument('-p', '--input_pdb_directory', type=str, help='Where should reference PDB files be found? '
                                                                      'Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input_pisa_directory', type=str, help='Where should reference PISA files be found? '
                                                                       'Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output_directory', type=str, help='Where should interface files be saved?')
    parser.add_argument('-q', '--query_web', action='store_true',
                        help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    logger = start_log(name=os.path.basename(__file__), level=2)

    # Input data/files
    pdb_directory = '/databases/pdb'  # TODO ends with .ent not sub_directoried
    pisa_directory = '/home/kmeador/yeates/fragment_database/all/pisa_files'  # TODO, subdirectoried here
    current_interface_file_path = '/yeates1/kmeador/fragment_database/current_pdb_lists/'  # Todo parameterize
    # pdb_directory = '/yeates1/kmeador/fragment_database/all_pdb'  # sub directoried
    # pdb_directory = 'all_pdbs'

    # Variables
    pdb_resolution_threshold = 3.0
    write_to_file = True  # TODO MySQL DB option would make this false

    pdbs_of_interest_file_name = 'under3A_protein_no_multimodel_no_mmcif_no_bad_pisa.txt'  # Todo parameterize
    all_protein_file = os.path.join(current_interface_file_path, pdbs_of_interest_file_name)
    if args.query_web:
        pdbs_of_interest = qPDB.retrieve_pdb_entries_by_advanced_query()
    else:
        pdbs_of_interest = to_iterable(all_protein_file)

    # Retrieve the verified set of biological assemblies
    qsbio_file_name = 'QSbio_GreaterThanHigh_Assemblies'  # .pkl'  # Todo parameterize
    qsbio_file = os.path.join(current_interface_file_path, qsbio_file_name)
    qsbio_confirmed_d = unpickle(qsbio_file)

    # TODO script this file creation ?
    #  dls = "http://www.muellerindustries.com/uploads/pdf/UW SPD0114.xls"
    #  resp = requests.get(dls)
    #  with open(qsbio_file, 'wb') as output
    #      output.write(resp.content)
    #  qsbio_df = pd.DataFrame(qsbio_file)
    #  qsbio_df.groupby('QSBio Confidence', inplace=True)

    qsbio_data_url = 'https://www.weizmann.ac.il/sb/faculty_pages/ELevy/downloads/QSbio.xlsx'
    response = requests.get(qsbio_data_url)
    print(response.content[:100])
    qsbio_df = pd.DataFrame(response.content)

    qsbio_monomers_file_name = 'QSbio_Monomers.csv'  # Todo parameterize
    qsbio_monomers_file = os.path.join(current_interface_file_path, qsbio_monomers_file_name)
    qsbio_monomers = to_iterable(qsbio_monomers_file)
    print('The number of monomers found in %s: %d' % (qsbio_monomers_file, len(set(qsbio_monomers))))

    # Current, DEPRECIATE!
    interfaces_dir = '/yeates1/kmeador/fragment_database/all_interfaces'
    all_interface_pdb_paths = get_all_pdb_file_paths(interfaces_dir)
    pdb_interface_codes = list(file_ext_split[0]
                               for file_ext_split in map(os.path.splitext, map(os.path.basename, all_interface_pdb_paths)))
    pdb_interface_d = set_up_interface_dict(pdb_interface_codes)

    # Optimal
    pdb_interface_file_name = 'AllPDBInterfaces'
    pdb_interface_file = os.path.join(current_interface_file_path, pdb_interface_file_name)
    sorted_file_name = 'PDBInterfacesSorted'  # Todo parameterize
    sorted_interfaces_file = os.path.join(current_interface_file_path, sorted_file_name)
    if os.path.exists(sorted_interfaces_file):
        interface_sort_d = unpickle(sorted_interfaces_file)

    if os.path.exists(pdb_interface_file):  # retrieve the pdb, [interfaces] dictionary
        pdb_interface_d = unpickle(pdb_interface_file)
        missing_pisa = set(pdb_interface_d.keys()) - set(pdbs_of_interest)
    else:  # make the PISA dictionary
        if args.query_web:
            dummy = None  # TODO combine all pdb codes into chunks of PISA requests using SDUtils.download_pisa()

        else:
            interface_sort_d, pdb_interface_d, missing_pisa = {}, {}, {}
            for pdb_code in pdbs_of_interest:
                pisa_path = retrieve_pisa_file_path(pdb_code, directory=pisa_directory)
                if pisa_path:
                    pisa_d = unpickle(pisa_path)
                    # if pisa_d:
                    interface_data = pisa_d['interfaces']
                    interface_ids = list(interface_data.keys())
                    interface_ids.remove('all_ids')
                    # Remove ligand interfaces from PISA interface_data
                    polymer_interface_ids = {int_id for int_id in interface_ids
                                             if pisa_polymer_interface(interface_data[int_id])}
                    pdb_interface_d[pdb_code] = polymer_interface_ids
                    interface_sort_d[pdb_code] = sort_pdb_interfaces_by_contact_type(pisa_d, polymer_interface_ids,
                                                                                     set(qsbio_confirmed_d[pdb_code]))
                else:
                    missing_pisa.add(pdb_code)
        # Save the objects
        # {1AB3: {1, 3, 5}, ...}
        pickle_object(pdb_interface_d, pdb_interface_file, out_path='')
        sorted_interfaces_file = pickle_object(interface_sort_d, sorted_interfaces_file, out_path='')

    # remove missing pdbs from pdb_of_interest
    pdbs_of_interest = set(pdbs_of_interest) - missing_pisa

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

    all_pdb_uniprot_file_name = '200206_MostCompleteAllPDBSpaceGroupUNPResInfo'  # .pkl'  # Todo parameterize
    all_pdb_uniprot_file = os.path.join(current_interface_file_path, all_pdb_uniprot_file_name)
    # Dictionary Structure - will be 'dbref' instead of 'ref'
    #  {'entity': {1: ['A']}, 'res': 2.6, 'ref': {'A': {'db': 'UNP', 'accession': 'P35755'})
    #   'cryst': {'space': 'P 21 21 2', 'a_b_c': (106.464, 75.822, 34.109), 'ang_a_b_c': (90.0, 90.0, 90.0)}}
    # 200121 has 'path': '/home/kmeador/yeates/fragment_database/all/all_pdbs/kn/3KN7.pdb', not entity
    if os.path.exists(all_pdb_uniprot_file):  # retrieve the pdb, DBreference, resolution, and crystal dictionary
        pdb_uniprot_info = unpickle(all_pdb_uniprot_file)
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

    # Next, find all those with biological interfaces according to PISA and QSBio, as well as monomers
    all_biological = {pdb for pdb in interface_sort_d if interface_sort_d[pdb]['bio'] != set()}
    all_missing_biological = set(interface_sort_d) - all_biological
    proposed_qsbio_monomer_set = {pdb for pdb in all_missing_biological if pdb in qsbio_confirmed_d}
    assert len(proposed_qsbio_monomer_set - qsbio_monomers) == 0, 'The set of proposed monomers has PDB\'s that are not monomers!'

    # NEXT Sorting the interfaces by UniProtID
    # Gather all UniProtIDs from the pdbs_of_interest
    uniprot_master_file = os.path.join(current_interface_file_path, uniprot_master_file_name)
    if os.path.exists(uniprot_master_file):  # If this file already exists, then we should add to it
        uniprot_master = unpickle(uniprot_master_file)
        # Next, add any additional pdb ids to the uniprot_master
        # uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdbs_of_interest, pdb_uniprot_info, master_dictionary=uniprot_master)
    else:
        uniprot_master = None
    uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdbs_of_interest, pdb_uniprot_info, master_dictionary=uniprot_master)
    print('Total UniProtID\'s added: %d\nTotal without UniProtID: %s' % (len(uniprot_sorted), len(no_unp_code)))
    uniprot_master = {uniprot_id: process_uniprot_entry(uniprot_id, uniprot_sorted[uniprot_id], pdb_uniprot_info,
                                                        min_resolution_threshold=pdb_resolution_threshold)
                      for uniprot_id in uniprot_sorted}

    sorted_interfaces_file = pickle_object(uniprot_master, uniprot_master_file, out_path='')

    # Next, search for specific types of interfaces from the Master UniProt sorted dictionary
    # Find unique UniProtIDs which are representative of the homo-oligomer and unknown PDB interfaces
    all_unknown_bio_interfaces, all_homo_bio_interfaces = {}, {}
    for pdb in interface_sort_d:
        if interface_sort_d[pdb]['unknown_bio']:  # check is there is an empty set()
            all_unknown_bio_interfaces[pdb] = interface_sort_d[pdb]['unknown_bio']
        if interface_sort_d[pdb]['bio']:  # check is there is an empty set()
            all_homo_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']
        # if EM or NMR structures are added we won't have xtal contacts. TODO need other source for interface than PISA
        # if interface_sort_d[pdb]['xtal']:  # check is there is an empty set()
        #     all_homo_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']

    # all_homo_uniprot_ids, all_unk_uniprot_ids = set(), set()
    all_homo_uniprot_ids, all_unk_uniprot_ids = {}, {}
    uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, unique_pdb_chains = {}, {}, {}
    for uniprot_id in uniprot_master:
        # Get the highest resolution heteromeric biological interfaces
        for pdb in uniprot_master[uniprot_id]['bio']:
            uniprot_hetero_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']
        # Get the highest resolution pdb for each space group along with the interfaces. These are assumed to comprise separate crystal contacts
        for space_group in uniprot_master[uniprot_id]['space_groups']:
            pdb = uniprot_master[uniprot_id]['space_groups'][space_group]['top']
            uniprot_filtered_xtal_interfaces[pdb] = interface_sort_d[pdb]['xtal']
        # Get all the highest resolution unique pdb and chain
        for pdb_chain in uniprot_master[uniprot_id]['all']:
            pdb = pdb_chain.split('.')[0]
            if pdb == uniprot_master[uniprot_id]['top']:
                # unique_pdb_chains[uniprot_id] = pdb_chain  # Was this, but the format should match the others...
                unique_pdb_chains[pdb] = pdb_chain.split('.')[1]
                break
        # From all unique_pdbs, see if they are in set of homo_ or unknown_bio interfaces
        for pdb in uniprot_master[uniprot_id]['unique_pdb']:
            if pdb in all_homo_bio_interfaces:
                if uniprot_id in all_homo_uniprot_ids:
                    all_homo_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_homo_uniprot_ids[uniprot_id] = {pdb}
            if pdb in all_unknown_bio_interfaces:
                if uniprot_id in all_unk_uniprot_ids:
                    all_unk_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_unk_uniprot_ids[uniprot_id] = {pdb}

    # Sort the set of UniProt ID's to find the top representative PDB for each homo_oligomer and unknown_oligomer
    uniprot_unknown_bio_interfaces, uniprot_homo_bio_interfaces = {}, {}
    for uniprot_id in all_homo_uniprot_ids:
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_homo_uniprot_ids[uniprot_id]:
            uniprot_homo_bio_interfaces[top_pdb] = all_homo_bio_interfaces[top_pdb]
        else:
            highest_res = deepcopy(pdb_resolution_threshold)
            highest_res_pdb = None
            for pdb in all_homo_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_homo_bio_interfaces[highest_res_pdb] = all_homo_bio_interfaces[highest_res_pdb]

    for uniprot_id in all_unk_uniprot_ids:
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_unk_uniprot_ids[uniprot_id]:
            uniprot_unknown_bio_interfaces[top_pdb] = all_unknown_bio_interfaces[top_pdb]
        else:
            highest_res = pdb_resolution_threshold
            highest_res_pdb = None
            for pdb in all_unk_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_unknown_bio_interfaces[highest_res_pdb] = all_unknown_bio_interfaces[highest_res_pdb]

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
