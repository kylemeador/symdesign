# import pickle5 as pickle  # python 3.8 pickling protocol compatible
import copy
import logging
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from glob import glob
from itertools import chain
from json import loads, dumps

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import IUPACData
from sklearn.neighbors import BallTree

import CmdUtils as CUtils
import PDB
import PathUtils as PUtils
# logging.getLogger().setLevel(logging.INFO)
from SequenceProfile import populate_design_dict

# Globals

index_offset = 1
rmsd_threshold = 1.0
layer_groups = {'P 1': 'p1', 'P 2': 'p2', 'P 21': 'p21', 'C 2': 'pg', 'P 2 2 2': 'p222', 'P 2 2 21': 'p2221',
                'P 2 21 21': 'p22121', 'C 2 2 2': 'c222', 'P 4': 'p4', 'P 4 2 2': 'p422',
                'P 4 21 2': 'p4121', 'P 3': 'p3', 'P 3 1 2': 'p312', 'P 3 2 1': 'p321', 'P 6': 'p6', 'P 6 2 2': 'p622'}
viable = {'p6', 'p4', 'p3', 'p312', 'p4121', 'p622'}
pisa_ref_d = {'multimers': {'ext': 'multimers.xml', 'source': 'pisa', 'mod': ''},
              'interfaces': {'ext': 'interfaces.xml', 'source': 'pisa', 'mod': ''},
              'multimer': {'ext': 'bioassembly.pdb', 'source': 'pdb', 'mod': ':1,1'}, 'pisa': '.pkl'}
point_group_d = {8: 'I32', 14: 'I52', 56: 'I53', 4: 'T32', 52: 'T33'}
# layer_group_d = {8: 'I23'}

##########
# ERRORS
##########


class DesignError(Exception):  # TODO make error messages one line instead of string iteration
    # SymDesignUtils.DesignError: ('I', 'n', 'v', 'a', 'l', 'i', 'd', ' ', 'P', 'D', 'B', ' ', 'i', 'n', 'p', 'u', 't',
    # ',', ' ', 'n', 'o', ' ', 'S', 'E', 'Q', 'R', 'E', 'S', ' ', 'r', 'e', 'c', 'o', 'r', 'd', ' ', 'f', 'o', 'u', 'n',
    # 'd')

    def __init__(self, message):
        self.args = message


def handle_errors_f(errors=(Exception, )):
    """Decorator to wrap a function with try: ... except errors: finally:

    Keyword Args:
        errors=(Exception, ) (tuple): A tuple of exceptions to monitor, even if single exception
    Returns:
        return, error (tuple): [0] is function return upon proper execution, else None, tuple[1] is error if exception
            raised, else None
    """
    def wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as e:
                return None
        return wrapped
    return wrapper


def handle_errors(errors=(Exception, )):  # TODO refactor handle_errors to handle_errors_DesDir
    """Decorator to wrap a function with try: ... except errors: finally:

    Keyword Args:
        errors=(Exception, ) (tuple): A tuple of exceptions to monitor, even if single exception
    Returns:
        return, error (tuple): [0] is function return upon proper execution, else None, tuple[1] is error if exception
            raised, else None
    """
    def wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs), None
            except errors as e:
                return None, (args[0], e)  # requires a directory identifier as args[0]
                # return None, (args[0].path, e)
            # finally:  TODO figure out how to run only when uncaught exception is found
            #     print('Error occurred in %s' % args[0].path)
        return wrapped
    return wrapper


############
# Symmetry
############


def handle_symmetry(symmetry_entry_number):
    # group = cryst1_record.split()[-1]/
    if symmetry_entry_number not in point_group_d.keys():
        if symmetry_entry_number in layer_group_d.keys():
            return 2
        else:
            return 3
    else:
        return 0


def sdf_lookup(point_type, dummy=False):
    if dummy:
        return os.path.join(PUtils.symmetry_def_files, 'dummy.symm')
    else:
        symmetry_name = point_group_d[point_type]

    for root, dirs, files in os.walk(PUtils.symmetry_def_files):
        for file in files:
            if symmetry_name in file:
                return os.path.join(PUtils.symmetry_def_files, file)


#####################
# Runtime Utilities
#####################


def start_log(name='', handler=1, level=2, location=os.getcwd(), propagate=True):
    """Create a logger to handle program messages

    Keyword Args:
        name='' (str): The name of the logger
        handler=1 (int): Whether to handle to stream (1-default) or a file (2)
        level=2 (int): What level of messages to emit (1-debug, 2-info (default), 3-warning, 4-error, 5-critical)
        location=os.getcwd() (str): If a FileHandler is used (handler=2) where should file be written?
            .log is appended to file
        propagate=True (bool): Whether to pass messages to parent level loggers
    Returns:
        _logger (Logger): Logger object to handle messages
    """
    # log_handler = {1: logging.StreamHandler(), 2: logging.FileHandler(location + '.log'), 3: logging.NullHandler}
    log_level = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}
    log_format = logging.Formatter('[%(levelname)s] %(module)s: %(message)s')

    _logger = logging.getLogger(name)
    _logger.setLevel(log_level[level])
    if not propagate:
        _logger.propagate = False
    # lh = log_handler[handler]
    if handler == 1:
        lh = logging.StreamHandler()
    elif handler == 2:
        lh = logging.FileHandler(location + '.log')
    else:  # handler == 3:
        return _logger
    lh.setLevel(log_level[level])
    lh.setFormatter(log_format)
    _logger.addHandler(lh)

    return _logger


logger = start_log(name=__name__, handler=3, level=1)


@handle_errors_f(errors=(FileNotFoundError, ))
def unpickle(file_name):
    """Unpickle (deserialize) and return a python object located at filename"""
    if '.pkl' not in file_name:
        file_name = '%s.pkl' % file_name

    with open(file_name, 'rb') as serial_f:
        new_object = pickle.load(serial_f)

    return new_object


def pickle_object(target_object, name, out_path=os.getcwd(), protocol=pickle.HIGHEST_PROTOCOL):
    """Pickle (serialize) an object into a file named 'out_path/name.pkl'

    Args:
        target_object (any): Any python object
        name (str): The name of the pickled file
    Keyword Args:
        out_path=os.getcwd() (str): Where the file should be written
    Returns:
        (str): The pickled filename
    """
    if '.pkl' not in name:
        name = '%s.pkl' % name

    file_name = os.path.join(out_path, name)
    with open(file_name, 'wb') as f:
        pickle.dump(target_object, f, protocol)

    return file_name


def clean_dictionary(dictionary, keys, remove=True):
    """Clean specified keys from a dictionary. Default removes the specified keys

    Args:
        dictionary (dict): {outer_dictionary: {key: value, key2: value2, ...}, ...}
        keys (iter): [key2, key10] Iterator of keys to be removed from dictionary
    Keyword Args:
        remove=True (bool): Whether or not to remove (True) or keep (False) specified keys
    Returns:
        (dict): {outer_dictionary: {key: value, ...}, ...} - Cleaned dictionary
    """
    if remove:
        for key in keys:
            dictionary.pop(key)

        return dictionary
    else:
        return {key: dictionary[key] for key in keys if key in dictionary}


def clean_interior_keys(dictionary, keys, remove=True):
    """Clean specified keys from a dictionaries internal dictionary. Default removes the specified keys

    Args:
        dictionary (dict): {outer_dictionary: {key: value, key2: value2, ...}, ...}
        keys (iter): [key2, key10] Iterator of keys to be removed from dictionary
    Keyword Args:
        remove=True (bool): Whether or not to remove (True) or keep (False) specified keys
    Returns:
        (dict): {outer_dictionary: {key: value, ...}, ...} - Cleaned dictionary
    """
    if remove:
        for entry in dictionary:
            for key in keys:
                try:
                    dictionary[entry].pop(key)
                except KeyError:
                    pass

        return dictionary
    else:
        new_dictionary = {entry: {key: dictionary[entry][key] for key in dictionary[entry] if key in keys}
                          for entry in dictionary}
        # for entry in dictionary:
        #     new_dictionary[entry] = {}
        #     for key in dictionary[entry]:
        #         if key in keys:
        #             new_dictionary[entry][key] = dictionary[entry][key]

        return new_dictionary


def index_intersection(indices):
    """Find the overlap of sets in a dictionary
    """
    final_indices = set()
    # find all set union
    for metric in indices:
        final_indices = set(final_indices) | set(indices[metric])
    # find all set intersection
    for metric in indices:
        final_indices = set(final_indices) & set(indices[metric])


    return list(final_indices)


# def reduce_pose_to_chains(pdb, chains):  # UNUSED
#     new_pdb = PDB.PDB()
#     new_pdb.read_atom_list(pdb.chains(chains))
#
#     return new_pdb
#
#
# def combine_pdb(pdb_1, pdb_2, name):  # UNUSED
#     """Take two pdb objects and write them to the same file
#
#     Args:
#         pdb_1 (PDB): First PDB to concatentate
#         pdb_2 (PDB): Second PDB
#         name (str): Name of the output file
#     """
#     pdb_1.write(name)
#     with open(name, 'a') as full_pdb:
#         for atom in pdb_2.all_atoms:
#             full_pdb.write(str(atom))  # .strip() + '\n')
#
#
# def identify_interface_chains(pdb1, pdb2):  # UNUSED
#     distance = 12  # Angstroms
#     pdb1_chains = []
#     pdb2_chains = []
#     # Get Queried CB Tree for all PDB2 Atoms within 12A of PDB1 CB Atoms
#     query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance)
#
#     for pdb2_query_index in range(len(query)):
#         if query[pdb2_query_index].tolist() != list():
#             pdb2_chains.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain)
#             for pdb1_query_index in query[pdb2_query_index]:
#                 pdb1_chains.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain)
#
#     pdb1_chains = list(set(pdb1_chains))
#     pdb2_chains = list(set(pdb2_chains))
#
#     return pdb1_chains, pdb2_chains
#
#
# def rosetta_score(pdb):  # UNUSED
#     # this will also format your output in rosetta numbering
#     cmd = [PUtils.rosetta, 'score_jd2.default.linuxgccrelease', '-renumber_pdb', '-ignore_unrecognized_res', '-s', pdb,
#            '-out:pdb']
#     subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#
#     return pdb + '_0001.pdb'
#
#
# def duplicate_ssm(pssm_dict, copies):  # UNUSED
#     duplicated_ssm = {}
#     duplication_start = len(pssm_dict)
#     for i in range(int(copies)):
#         if i == 0:
#             offset = 0
#         else:
#             offset = duplication_start * i
#         # j = 0
#         for line in pssm_dict:
#             duplicated_ssm[line + offset] = pssm_dict[line]
#             # j += 1
#
#     return duplicated_ssm
#
#
# def get_all_cluster(pdb, residue_cluster_id_list, db=PUtils.bio_fragmentDB):  # UNUSED DEPRECIATED
#     # generate an interface specific scoring matrix from the fragment library
#     # assuming residue_cluster_id_list has form [(1_2_24, [78, 87]), ...]
#     cluster_list = []
#     for cluster in residue_cluster_id_list:
#         cluster_loc = cluster[0].split('_')
#         res1 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[0], cluster[1][0]))
#         res2 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[1], cluster[1][1]))
#         filename = os.path.join(db, cluster_loc[0], cluster_loc[0] + '_' + cluster_loc[1], cluster_loc[0] +
#                                 '_' + cluster_loc[1] + '_' + cluster_loc[2], cluster[0] + '.pkl')
#         cluster_list.append([[res1.ca, res2.ca], unpickle(filename)])
#
#     # OUTPUT format: [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}], ...]
#     return cluster_list
#
#
# def convert_to_frag_dict(interface_residue_list, cluster_dict):  #UNUSED
#     # Make PDB/ATOM objects and dictionary into design dictionary
#     # INPUT format: interface_residue_list = [[[Atom_ca.residue1, Atom_ca.residue2], '1_2_45'], ...]
#     interface_residue_dict = {}
#     for residue_dict_pair in interface_residue_list:
#         residues = residue_dict_pair[0]
#         for i in range(len(residues)):
#             residues[i] = residues[i].residue_number
#         hash_ = (residues[0], residues[1])
#         interface_residue_dict[hash_] = cluster_dict[residue_dict_pair[1]]
#     # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
#     return interface_residue_dict
#
#
# def convert_to_rosetta_num(pdb, pose, interface_residue_list):  # UNUSED
#     # DEPRECIATED in favor of updating PDB/ATOM objects
#     # INPUT format: interface_residue_list = [[[78, 87], {'IJKClusterDict - such as 1_2_45'}], [[64, 87], {...}], ...]
#     component_chains = [pdb.chain_id_list[0], pdb.chain_id_list[-1]]
#     interface_residue_dict = {}
#     for residue_dict_pair in interface_residue_list:
#         residues = residue_dict_pair[0]
#         dict_ = residue_dict_pair[1]
#         new_key = []
#         pair_index = 0
#         for chain in component_chains:
#             new_key.append(pose.pdb_info().pdb2pose(chain, residues[pair_index]))
#             pair_index = 1
#         hash_ = (new_key[0], new_key[1])
#
#         interface_residue_dict[hash_] = dict_
#     # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
#     return interface_residue_dict
#
#
# def get_residue_list_atom(pdb, residue_list, chain=None):  # UNUSED DEPRECIATED
#     if chain is None:
#         chain = pdb.chain_id_list[0]
#     residues = []
#     for residue in residue_list:
#         res_atoms = PDB.Residue(pdb.getResidueAtoms(chain, residue))
#         residues.append(res_atoms)
#
#     return residues
#
#
# def get_residue_atom_list(pdb, residue_list, chain=None):  # UNUSED
#     if chain is None:
#         chain = pdb.chain_id_list[0]
#     residues = []
#     for residue in residue_list:
#         residues.append(pdb.getResidueAtoms(chain, residue))
#
#     return residues
#
#
# def make_issm(cluster_freq_dict, background):  # UNUSED
#     for residue in cluster_freq_dict:
#         for aa in cluster_freq_dict[residue]:
#             cluster_freq_dict[residue][aa] = round(2 * (math.log2((cluster_freq_dict[residue][aa] / background[aa]))))
#     issm = []
#
#     return issm


def subdirectory(name):  # TODO PDBdb
    return name


###################
# Bio.PDB Handling
###################


def get_rmsd_atoms(filepaths, function):
    all_rmsd_atoms = []
    for filepath in filepaths:
        parser = PDBParser()
        pdb_name = os.path.splitext(os.path.basename(filepath))[0]
        pdb = parser.get_structure(pdb_name, filepath)
        all_rmsd_atoms.append(function(pdb))

    return all_rmsd_atoms


def get_biopdb_ca(structure):
    return [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']


def superimpose(atoms):  # , rmsd_thresh):
    # biopdb_1_id = atoms[0][0].get_full_id()[0]
    # biopdb_2_id = atoms[1][0].get_full_id()[0]
    sup = Superimposer()
    # sup.set_atoms(atoms[0], atoms[1])
    sup.set_atoms(*atoms)
    # if sup.rms <= rmsd_thresh:
    return sup.rms
    # return biopdb_1_id, biopdb_2_id, sup.rms
    # else:
    #     return None


###################
# PDB Handling # TODO PDB.py
###################


def download_pdb(pdb, location=os.getcwd(), asu=False):
    """Download a pdbs from a file, a supplied list, or a single entry

    Args:
        pdb (str, list): PDB's of interest. If asu=False, code_# is format for biological assembly specific pdb.
            Ex: 1bkh_2 fetches biological assembly 2
    Keyword Args:
        asu=False (bool): Whether or not to download the asymmetric unit file
    Returns:
        (None)
    """
    clean_list = to_iterable(pdb)

    failures = []
    for pdb in clean_list:
        clean_pdb = pdb[0:4]  # .upper() redundant call
        if asu:
            assembly = ''
        else:
            assembly = pdb[-3:]
            try:
                assembly = assembly.split('_')[1]
            except IndexError:
                assembly = '1'

        clean_pdb = '%s.pdb%s' % (clean_pdb, assembly)
        file_name = os.path.join(location, clean_pdb)
        current_file = glob(file_name)
        # current_files = os.listdir(location)
        # if clean_pdb not in current_files:
        if not current_file:  # glob will return an empty list if the file is missing and therefore should be downloaded
            # TODO subprocess.POPEN()
            status = os.system('wget -q -O %s http://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            if status != 0:
                failures.append(pdb)

    if failures:
        logger.error('PDB download ran into the following failures:\n%s' % ', '.join(failures))

    return file_name  # if list then will only return the last file


def download_pisa(pdb, pisa_type, out_path=os.getcwd(), force_singles=False):
    """Downloads PISA .xml files from http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/
    Args:
        pdb (str,list): Either a single pdb code, a list of pdb codes, or a file with pdb codes, comma or newline delimited
        pisa_type (str): Either 'multimers', 'interfaces', or 'multimer' to designate the PISA File Source
    Keyword Args:
        out_path=os.getcwd() (str): Path to download PISA files
        force_singles=False (bool): Whether to force downloading of one file at a time
    Returns:
        None
    """
    import xml.etree.ElementTree as ETree

    def retrieve_pisa(pdb_code, _type, filename):
        p = subprocess.Popen(['wget', '-q', '-O', filename, 'https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/%s.%s?%s' %
                              (_type, pisa_ref_d[_type]['source'], pdb_code)])
        if p.returncode != 0:  # Todo if p.returncode
            return False
        else:
            return True

    def separate_entries(tree, ext, out_path=os.getcwd()):
        for pdb_entry in tree.findall('pdb_entry'):
            if pdb_entry.find('status').text.lower() != 'ok':
                failures.extend(modified_pdb_code.split(','))
            else:
                # PDB code is uppercase when returned from PISA interfaces, but lowercase when returned from PISA Multimers
                filename = os.path.join(out_path, '%s_%s' % (pdb_entry.find('pdb_code').text.upper(), ext))
                add_root = ETree.Element('pisa_%s' % pisa_type)
                add_root.append(pdb_entry)
                new_xml = ETree.ElementTree(add_root)
                new_xml.write(open(filename, 'w'), encoding='unicode')  # , pretty_print=True)
                successful_downloads.append(pdb_entry.find('pdb_code').text.upper())

    def process_download(pdb_code, file):
        # nonlocal fail
        nonlocal failures
        if retrieve_pisa(pdb_code, pisa_type, file):  # download was successful
            # Check to see if <status>Ok</status> for the download
            etree = ETree.parse(file)
            if force_singles:
                if etree.find('status').text.lower() == 'ok':
                    successful_downloads.append(pdb_code)
                    # successful_downloads.extend(modified_pdb_code.split(','))
                else:
                    failures.extend(modified_pdb_code.split(','))
            else:
                separate_entries(etree, pisa_ref_d[pisa_type]['ext'])
        else:  # download failed
            failures.extend(modified_pdb_code.split(','))

    if pisa_type not in pisa_ref_d:
        logger.error('%s is not a valid PISA file type' % pisa_type)
        sys.exit()
    if pisa_type == 'multimer':
        force_singles = True

    file = None
    clean_list = to_iterable(pdb)
    count, total_count = 0, 0
    multiple_mod_code, successful_downloads, failures = [], [], []
    for pdb in clean_list:
        pdb_code = pdb[0:4].lower()
        file = os.path.join(out_path, '%s_%s' % (pdb_code.upper(), pisa_ref_d[pisa_type]['ext']))
        if file not in os.listdir(out_path):
            if not force_singles:  # concatenate retrieval
                count += 1
                multiple_mod_code.append(pdb_code)
                if count == 50:
                    count = 0
                    total_count += count
                    logger.info('Iterations: %d' % total_count)
                    modified_pdb_code = ','.join(multiple_mod_code)
                else:
                    continue
            else:
                modified_pdb_code = '%s%s' % (pdb_code, pisa_ref_d[pisa_type]['mod'])
                logger.info('Fetching: %s' % pdb_code)

            process_download(modified_pdb_code, file)
            multiple_mod_code = []

    # Handle remaining codes in concatenation instances where the number remaining is < 50
    if count > 0 and multiple_mod_code != list():
        modified_pdb_code = ','.join(multiple_mod_code)
        process_download(modified_pdb_code, file)

    # Remove successfully downloaded files from the input
    # duplicates = []
    for pdb_code in successful_downloads:
        if pdb_code in clean_list:
            # try:
            clean_list.remove(pdb_code)
        # except ValueError:
        #     duplicates.append(pdb_code)
    # if duplicates:
    #     logger.info('These files may be duplicates:', ', '.join(duplicates))

    if not clean_list:
        return True
    else:
        failures.extend(clean_list)  # should just match clean list ?!
        failures = remove_duplicates(failures)
        logger.warning('Download PISA Failures:\n[%s]' % failures)
        io_save(failures)

        return False


def retrieve_pdb_file_path(code, directory=PUtils.pdb_db):
    """Fetch PDB object of each chain from PDBdb or PDB server

        Args:
            code (iter): Any iterable of PDB codes
        Keyword Args:
            location= : Location of the  on disk
        Returns:
            (str): path/to/your_pdb.pdb
        """
    if PUtils.pdb_source == 'download_pdb':
        get_pdb = download_pdb
        # doesn't return anything at the moment
    else:
        get_pdb = (lambda pdb_code, location=None: glob(os.path.join(location, 'pdb%s.ent' % pdb_code.lower())))
        # The below set up is my local pdb and the format of escher. cassini is slightly different, ughhh
        # get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_db, subdirectory(pdb_code),
        #                                                      '%s.pdb' % pdb_code)))
        # returns a list with matching file (should only be one)

    # pdb_file = get_pdb(code, location)
    pdb_file = get_pdb(code, location=directory)
    # pdb_file = get_pdb(code, location=des_dir.pdbs)
    assert len(pdb_file) == 1, 'More than one matching file found for PDB: %s' % code
    assert pdb_file != list(), 'No matching file found for PDB: %s' % code

    return pdb_file[0]


def fetch_pdbs(codes, location=PUtils.pdb_db):  # UNUSED
    """Fetch PDB object of each chain from PDBdb or PDB server

    Args:
        codes (iter): Any iterable of PDB codes
    Keyword Args:
        location= : Location of the  on disk
    Returns:
        (dict): {pdb_code: PDB.py object, ...}
    """
    if PUtils.pdb_source == 'download_pdb':
        get_pdb = download_pdb
        # doesn't return anything at the moment
    else:
        get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_location, subdirectory(pdb_code),
                                                             '%s.pdb' % pdb_code)))
        # returns a list with matching file (should only be one)
    oligomers = {}
    for code in codes:
        pdb_file_name = get_pdb(code, location=des_dir.pdbs)
        assert len(pdb_file_name) == 1, 'More than one matching file found for pdb code %s' % code
        oligomers[code] = read_pdb(pdb_file_name[0])
        oligomers[code].set_name(code)
        oligomers[code].reorder_chains()

    return oligomers


def read_pdb(file):  # DEPRECIATE
    """Wrapper on the PDB __init__ and readfile functions

    Args:
        file (str): Disk location of pdb file
    Returns:
        (PDB): Initialized PDB object
    """
    return PDB.PDB(file=file)


def fill_pdb(atom_list=None):  # DEPRECIATE
    """Wrapper on the PDB __init__ and readfile functions

    Args:
        atom_list (list): List of Atom objects
    Returns:
        pdb (PDB): Initialized PDB object
    """
    if not atom_list:
        atom_list = []

    pdb = PDB.PDB()
    pdb.read_atom_list(atom_list)

    return pdb


def extract_asu(file, chain='A', outpath=None):  # DEPRECIATE
    """Takes a PDB file and extracts an ASU. ASU is defined as chain, plus all unique entities in contact with chain"""
    if outpath:
        asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(file))[0] + '.pdb')
        # asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(file))[0] + '_%s' % 'asu.pdb')
    else:
        asu_file_name = os.path.splitext(file)[0] + '_asu.pdb'

    pdb = read_pdb(file)
    asu_pdb = PDB.PDB()
    asu_pdb.__dict__ = pdb.__dict__.copy()
    asu_pdb.all_atoms = pdb.get_asu(chain)
    # asu_pdb = fill_pdb(pdb.get_asu(chain))
    asu_pdb.write(asu_file_name, cryst1=asu_pdb.cryst)

    return asu_file_name


def residue_interaction_graph(pdb, distance=8, gly_ca=True):
    """Create a atom tree using CB atoms from two PDB's

    Args:
        pdb (PDB): First PDB to query against
    Keyword Args:
        distance=8 (int): The distance to query in Angstroms
        gly_ca=True (bool): Whether glycine CA should be included in the tree
    Returns:
        query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
    """
    # Get CB Atom Coordinates including CA coordinates for Gly residues
    coords = np.array(pdb.extract_CB_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    query = pdb1_tree.query_radius(coords, distance)

    return query


def construct_cb_atom_tree(pdb1, pdb2, distance=8, gly_ca=True):
    """Create a atom tree using CB atoms from two PDB's

    Args:
        pdb1 (PDB): First PDB to query against
        pdb2 (PDB): Second PDB which will be tested against pdb1
    Keyword Args:
        distance=8 (int): The distance to query in Angstroms
        gly_ca=True (bool): Whether glycine CA should be included in the tree
    Returns:
        query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
        pdb1_cb_indices (list): List of all CB indices from pdb1
        pdb2_cb_indices (list): List of all CB indices from pdb2
    """
    # Get CB Atom Coordinates including CA coordinates for Gly residues
    pdb1_coords = np.array(pdb1.extract_CB_coords(InclGlyCA=gly_ca))
    pdb2_coords = np.array(pdb2.extract_CB_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(pdb1_coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    query = pdb1_tree.query_radius(pdb2_coords, distance)

    # Map Coordinates to Atoms
    pdb1_cb_indices = pdb1.get_cb_indices(InclGlyCA=gly_ca)
    pdb2_cb_indices = pdb2.get_cb_indices(InclGlyCA=gly_ca)

    return query, pdb1_cb_indices, pdb2_cb_indices


def find_interface_pairs(pdb1, pdb2, distance=8):
    """Get pairs of residues across an interface within a certain distance

        Args:
            pdb1 (PDB): First pdb to measure interface between
            pdb2 (PDB): Second pdb to measure interface between
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
        Returns:
            interface_pairs (list(tuple): A list of interface residue pairs across the interface
    """
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance=distance)

    # Map Coordinates to Residue Numbers
    interface_pairs = []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            pdb2_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number
            for pdb1_index in query[pdb2_index]:
                pdb1_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number
                interface_pairs.append((pdb1_res_num, pdb2_res_num))

    return interface_pairs


def split_interface_pairs(interface_pairs):
    residues1, residues2 = zip(*interface_pairs)
    return set(sorted(set(residues1), key=int)), set(sorted(set(residues2), key=int))


def find_interface_residues(pdb1, pdb2, distance=8):
    """Get unique residues from each pdb across an interface

        Args:
            pdb1 (PDB): First pdb to measure interface between
            pdb2 (PDB): Second pdb to measure interface between
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
        Returns:
            (tuple(set): A tuple of interface residue sets across an interface
    """
    return split_interface_pairs(find_interface_pairs(pdb1, pdb2, distance=distance))


def print_atoms(atom_list):  # DEBUG
    print(''.join(str(atom) for atom in atom_list))
    # for atom in atom_list:
    #     for atom in residue:
    #     logger.info(str(atom))
    #     print(str(atom))


######################
# Matrix Handling
######################


def all_vs_all(iterable, func, symmetrize=True):  # TODO SDUtils
    """Calculate an all versus all comparison using a defined function. Matrix is symmetrized by default

    Args:
        iterable (iter): Dict or array like object
        func (function): Function to calculate different iterations of the iterable
    Keyword Args:
        symmetrize=True (Bool): Whether or not to make the resulting matrix symmetric
    Returns:
        all_vs_all (numpy array): Matrix with resulting calculations
    """
    if type(iterable) == dict:
        # func(iterable[obj1], iterable[obj2])
        _dict = iterable
    else:
        _dict = None
    pairwise = np.zeros((len(iterable), (len(iterable))))
    for i, obj1 in enumerate(iterable):
        for j, obj2 in enumerate(iterable):
            if j < i:
                continue
            # if type(iterable) == dict:  # _dict
            pairwise[i][j] = func(obj1, obj2, d=_dict)
            # pairwise[i][j] = func(obj1, obj2, iterable, d=_dict)
            # else:
            #     pairwise[i][j] = func(obj1, obj2, iterable, d=_dict)

    if symmetrize:
        return sym(pairwise)
    else:
        return pairwise


def sym(a):
    """Symmetrize a NumPy array. i.e. if a_ij = 0, then the returned array is such that _ij = a_ji

    Args:
        a (numpy array): square NumPy array
    Returns:
        (numpy array): Symmetrized NumPy array
    """
    return a + a.T - np.diag(a.diagonal())


def condensed_to_square(k, n):
    """Return the i, j indices of a scipy condensed matrix from element k and matrix dimension n"""
    def calc_row_idx(_k, _n):
        return int(math.ceil((1 / 2.) * (- (-8 * _k + 4 * _n ** 2 - 4 * _n - 7) ** 0.5 + 2 * _n - 1) - 1))

    def elem_in_i_rows(_i, _n):
        return _i * (_n - 1 - _i) + (_i * (_i + 1)) // 2

    def calc_col_idx(_k, _i, _n):
        return int(_n - elem_in_i_rows(_i + 1, _n) + _k)
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)

    return i, j


#################
# File Handling
#################


def io_save(data, filename=None):
    """Take an iterable and either output to user, write to a file, or both. User defined choice

    Returns
        None
    """
    # file = os.path.join(os.getcwd(), 'missing_UNP_PDBS.txt')
    def write_file():
        if not filename:
            filename = input('What is your desired filename? (appended to current working directory)\n')
            filename = os.path.join(os.getcwd(), filename)
        with open(filename, 'w') as f:
            f.write('\n'.join(data))
        print('File \'%s\' was written' % filename)

    while True:
        _input = input('Enter P to print Data, W to write Data to file, or B for both:').upper()
        if _input == 'W':
            write_file()
            break
        elif _input == 'P':
            print(data)
            break
        elif _input == 'B':
            print(data)
            # if not filename:
            #     filename = input('What is your desired filename? (appended to current directory)\n')
            #     filename = os.path.join(os.getcwd(), filename)
            # with open(filename, 'w') as f:
            #     f.write('\n'.join(data))
            # print('File \'%s\' was written' % filename)
            # break
            write_file()
        else:
            print('Invalid Input...')


def to_iterable(_obj):
    """Take a file/object and return a list of individual objects splitting on newline or comma"""
    try:
        with open(_obj, 'r') as f:
            _list = f.readlines()
    except FileNotFoundError:
        if isinstance(_obj, list):
            _list = _obj
        else:  # assumes obj is a string
            _list = [_obj]

    clean_list = []
    for it in _list:
        it_list = it.split(',')
        clean_list.extend([_it.strip() for _it in it_list])

    # remove duplicates but keep the order
    return remove_duplicates(clean_list)


def remove_duplicates(_iter):
    seen = set()
    seen_add = seen.add
    return [x for x in _iter if not (x in seen or seen_add(x))]


def write_shell_script(command, name='script', outpath=os.getcwd(), additional=None, shell='bash'):
    """Take a list with command flags formatted for subprocess and write to a name.sh script

    Args:
        command (str): The command formatted using subprocess.list2cmdline(list())
    Keyword Args:
        name='script' (str): The name of the output shell script
        outpath=os.getcwd() (str): The location where the script will be written
        additional=None (iter): Additional commands also formatted using subprocess.list2cmdline
        shell='bash' (str): The shell which should interpret the script
    Returns:
        (str): The name of the file
    """
    file_name = os.path.join(outpath, name + '.sh')
    with open(file_name, 'w') as f:
        f.write('#!/bin/%s\n\n%s\n' % (shell, command))
        if additional:
            f.write('%s\n' % '\n'.join(x for x in additional))

    return file_name


def write_commands(command_list, name='all_commands', loc=os.getcwd()):  # TODO loc, location, outpath. Standardize!!!
    file = os.path.join(loc, name + '.cmd')
    with open(file, 'w') as f:
        for command in command_list:
            f.write(command + '\n')

    return file


def write_list_to_file(_list, name=None, location=os.getcwd()):
    file_name = os.path.join(location, name)  # + '.cmd')
    with open(file_name, 'w') as f:
        f.write('\n'.join(item for item in _list))

    return file_name


def pdb_list_file(refined_pdb, total_pdbs=1, suffix='', loc=os.getcwd(), additional=None):
    file_name = os.path.join(loc, 'pdb_file_list.txt')
    with open(file_name, 'w') as f:
        f.write(refined_pdb + '\n')  # run second round of metrics on input as well
        for i in range(index_offset, total_pdbs + index_offset):
            file_line = os.path.splitext(refined_pdb)[0] + suffix + '_' + str(i).zfill(4) + '.pdb\n'
            f.write(file_line)
        if additional:
            for pdb in additional:
                f.write(pdb + '\n')

    return file_name


def prepare_rosetta_flags(flag_variables, stage, outpath=os.getcwd()):
    """Prepare a protocol specific Rosetta flags file with program specific variables

    Args:
        flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
        stage (str): The protocol stage or flag suffix to name the specific flags file
    Keyword Args:
        outpath=cwd (str): Disk location to write the flags file
    Returns:
        flag_file (str): Disk location of the written flags file
    """
    output_flags = ['-out:path:pdb ' + os.path.join(outpath, PUtils.pdbs_outdir),
                    '-out:path:score ' + os.path.join(outpath, PUtils.scores_outdir)]

    def make_flags_file(flag_list):
        with open(os.path.join(outpath, 'flags_' + stage), 'w') as f:
            for flag in flag_list:
                f.write(flag + '\n')

        return 'flags_' + stage

    _flags = copy.deepcopy(CUtils.flags)
    _flags += output_flags
    _options = CUtils.flag_options[stage]
    for variable in _options:
        _flags.append(variable)

    variables_for_flag_file = '-parser:script_vars'
    for variable, value in flag_variables:
        variables_for_flag_file += ' ' + str(variable) + '=' + str(value)

    _flags.append(variables_for_flag_file)

    return make_flags_file(_flags)


@handle_errors_f(errors=(FileNotFoundError, ))
def parse_flags_file(directory, name='design', flag_variable=None):
    """Returns the design flags passed to Rosetta from a design directory

    Args:
        directory (str): Location of design directory on disk
    Keyword Args:
        name='design' (str): The flags file suffix
        flag_variable=None (str): The name of a specific variable to retrieve
    Returns:
        variable_dict (dict): {'interfaceA': 15A,21A,25A,..., 'dssm_file': , ...}
    """
    parser_vars = '-parser:script_vars'
    with open(os.path.join(directory, 'flags_%s' % name), 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            if line[:19] == parser_vars:
                variables = line.lstrip(parser_vars).strip().split()
                variable_dict = {}
                for variable in variables:
                    variable_dict[variable.split('=')[0]] = variable.split('=')[1]
                if flag_variable:
                    return variable_dict[flag_variable]
                else:
                    return variable_dict


def get_interface_residues(design_variables, pdb=True, zero=False):
    """Returns a list of interface residues from flags_design parameters

    Args:
        design_variables (dict): {'interfaceA': 15A,21A,25A,..., 'dssm_file': , ...}
    Keyword Args:
        pdb=True (bool): Whether residues are in PDB format (True) or pose format (False)
        zero=True (bool): Whether residues are zero indexed (True) or one indexed (False)
    Returns:
          int_residues (list): [13, 16, 40, 88, 129, 130, 131, 190, 300]
    """
    _slice, offset = 0, 0
    if pdb:
        _slice = -1
    if zero:
        offset = index_offset
    int_residues = []
    for var in design_variables:
        if var.startswith('interface'):
            int_residues += [int(n[:_slice]) - offset for n in design_variables[var].split(',')]

    return int_residues


def change_filename(original, new=None, increment=None):
    """Take a json formatted score.sc file and rename decoy ID's

    Args:
        original (str): Location on disk of file
    Keyword Args:
        new=None (str): The filename to replace the file with
        increment=None (int): The number to increment each decoy by
    """
    dirname = os.path.dirname(original)
    # basename = os.path.basename(original)  # .split('_')[:-1]
    basename = os.path.splitext(os.path.basename(original))[0]  # .split('_')[:-1]
    ext = os.path.splitext(os.path.basename(original))[-1]  # .split('_')[:-1]
    old_id = basename.split('_')[-1]
    # old_id = int(os.path.basename(original).split('_')[-1])
    if increment:
        new_id = int(old_id) + increment
    else:
        new_id = int(old_id)

    if new:
        new_file = os.path.join(dirname, '%s_%04d%s' % (new, new_id, ext))
    else:
        new_file = os.path.join(dirname, '%s%04d%s' % (basename[:-len(old_id)], new_id, ext))

    p = subprocess.Popen(['mv', original, new_file])


def modify_decoys(file, increment=0):
    """Take a json formatted score.sc file and rename decoy ID's

    Args:
        file (str): Location on disk of scorefile
    Keyword Args:
        increment=None (int): The number to increment each decoy by
    Returns:
        score_dict (dict): {design_name: {all_score_metric_keys: all_score_metric_values}, ...}
    """
    with open(file, 'r+') as f:
        scores = [loads(score) for score in f.readlines()]
        for i, score in enumerate(scores):
            design_id = score['decoy'].split('_')[-1]
            if design_id[-1].isdigit():
                decoy_name = score['decoy'][:-len(design_id)]
                score['decoy'] = '%s%04d' % (decoy_name, int(design_id) + increment)
            scores[i] = score

        f.seek(0)
        f.write('\n'.join(dumps(score) for score in scores))
        f.truncate()


def rename_decoy_protocols(des_dir, rename_dict):
    score_file = os.path.join(des_dir.scores, PUtils.scores_file)
    with open(score_file, 'r+') as f:
        scores = [loads(score) for score in f.readlines()]
        for i, score in enumerate(scores):
            for protocol in rename_dict:
                if protocol in score:
                    score[protocol] = rename_dict[protocol]
            scores[i] = score

        f.seek(0)
        f.write('\n'.join(dumps(score) for score in scores))
        f.truncate()


####################
# MULTIPROCESSING
####################


def calculate_mp_threads(mpi=False, maximum=False, no_model=False):
    """Calculate the number of multiprocessing threads to use for a specific application

    Keyword Args:
        mpi=False (bool): If commands use MPI
        maximum=False (bool): Whether to use the maximum number of cpu's, leaving one available for the machine
        no_model=False (bool): If pose initialization is completed without any modelling
    Returns:
        (int): The number of threads to use
    """
    # int returns same as math.floor()
    if mpi:
        return int(mp.cpu_count() / CUtils.mpi)
    elif maximum:
        # # leave at least a CPU available for computer, see also len(os.sched_getaffinity(0)), mp.cpu_count() - 1
        return mp.cpu_count() - 1
    elif no_model:
        # current cap for use with hhblits and multiprocessing. TODO Change to take into account memory constraints
        return int(mp.cpu_count() / (CUtils.hhblits_threads + 5))
    else:
        # leave at least a CPU available for computer
        return int(mp.cpu_count() / CUtils.min_cores_per_job) - 1


def set_worker_affinity():
    """When a new worker process is created, use this initialization function to set the affinity for all CPUs.
    Especially important for multiprocessing in the context of numpy, scipy, pandas
    FROM Stack Overflow:
    https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy

    See: http://manpages.ubuntu.com/manpages/precise/en/man1/taskset.1.html
        -p is a mask for the logial cpu processers to use, the pid allows the affinity for an existing process to be
        specified instead of a new process being spawned
    """
    # print("I'm the process %d, setting affinity to all CPUs." % os.getpid())
    _cmd = ['taskset', '-p', '0x%s' % 'f' * int((os.cpu_count() / 4)), str(os.getpid())]
    logger.debug(subprocess.list2cmdline(_cmd))
    subprocess.Popen(_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def mp_map(function, arg, threads=1):
    """Maps input argument to a function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        arg (var): Argument to be unpacked in the defined function
        threads (int): How many workers/threads should be spawned to handle function(arguments)?
    Returns:
        results (list): The results produced from the function and arg
    """
    with mp.get_context('spawn').Pool(processes=threads) as p:  # maxtasksperchild=1
        results = p.map(function, arg)
    p.join()

    return results


def mp_try_starmap(function, process_args, threads, context='spawn'):  # UNUSED
    """Maps iterable to a try/except function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
        threads (int): How many workers/threads should be spawned to handle function(arguments)?
    Keyword Args:
        context='spawn' (str): One of 'spawn', 'fork', or 'forkserver'
    Returns:
        results (list): The results produced from the function and process_args
    """
    with mp.get_context(context).Pool(processes=threads) as p:  # maxtasksperchild=1
        results = p.starmap(function, process_args)
    p.join()

    return results


# to make mp compatible with 2.7
from contextlib import contextmanager


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def mp_starmap_python2(function, process_args, threads=1):
    with poolcontext(processes=threads) as p:
        results = p.map(function, process_args)
    p.join()

    return results


def mp_starmap(function, process_args, threads=1, context='spawn'):
    """Maps iterable to a function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
    Keyword Args:
        threads=1 (int): How many workers/threads should be spawned to handle function(arguments)?
        context='spawn' (str): One of 'spawn', 'fork', or 'forkserver'
    Returns:
        results (list): The results produced from the function and process_args
    """
    with mp.get_context(context).Pool(processes=threads, initializer=set_worker_affinity, maxtasksperchild=100) as p:
        results = p.starmap(function, process_args)  # , chunksize=1
    p.join()

    return results


######################
# Directory Handling
######################


# def set_up_dock_dir(path, suffix=None):  # DEPRECIATED
#     """Saves the path of the docking directory as DesignDirectory.path attribute. Tries to populate further using
#     typical directory structuring"""
#     dock_dir = DesignDirectory(path, auto_structure=False)
#     # try:
#     # dock_dir.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implementation
#     dock_dir.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses%s' % str(suffix or '')))  # design_recap
#     dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.symmetry]  # TODO change to PUtils
#     # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
#     dock_dir.building_blocks, dock_dir.building_block_logs = [], []
#     for k, _sym in enumerate(dock_dir.symmetry):
#         dock_dir.building_blocks.append(list())
#         dock_dir.building_block_logs.append(list())
#         for bb_dir in next(os.walk(_sym))[1]:
#             if os.path.exists(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils
#                 dock_dir.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
#                 dock_dir.building_blocks[k].append(bb_dir)
#
#     # dock_dir.building_blocks = [next(os.walk(dir))[1] for dir in dock_dir.symmetry]
#     # dock_dir.building_block_logs = [[os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)  # make a log path TODO PUtils
#     #                                  for bb_dir in dock_dir.building_blocks[k]]  # for each building_block combo in _sym index of dock_dir.building_blocks
#     #                                 for k, _sym in enumerate(dock_dir.symmetry)]  # for each sym in symmetry
#
#     return dock_dir


def get_pose_by_id(design_directories, ids):
    return [des_dir for des_dir in design_directories if str(des_dir) in ids]


def get_all_base_root_paths(directory):
    return [root for root, dirs, files in os.walk(directory) if not dirs]


def get_all_pdb_file_paths(pdb_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(pdb_dir) for file in files if '.pdb' in file]


def get_directory_pdb_file_paths(pdb_dir):
    return glob(os.path.join(pdb_dir, '*.pdb*'))


def get_base_nanohedra_dirs(base_dir):
    """Find all master directories corresponding to the highest output level of Nanohedra.py outputs. This corresponds
    to the DesignDirectory symmetry attribute
    """
    nanohedra_dirs = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if 'master_log.txt' in files:
            nanohedra_dirs.append(root)
            del dirs[:]
            # print('found %d directories' % len(nanohedra_dirs))

    return nanohedra_dirs


def get_docked_directories(base_directory, directory_type='NanohedraEntry'):  # '*DockedPoses'
    """Useful for when your docked directory is basically known but the """
    all_directories = []
    for root, dirs, files in os.walk(base_directory):
        # if directory_type in dirs:
        for _dir in dirs:
            if directory_type in _dir:
                all_directories.append(os.path.join(root, _dir))

    return sorted(set(all_directories))
    #
    # return sorted(set(map(os.path.dirname, glob('%s/*/*%s' % (base_directory, directory_type)))))


def get_docked_dirs_from_base(base):
    return sorted(set(map(os.path.dirname, glob('%s/*/*/*/*/' % base))))
    # want to find all NanohedraEntry1DockedPoses/1abc_2xyz/DEGEN_1_1/ROT_1_1/tx_139

    # for root1, dirs1, files1 in os.walk(base):  # NanohedraEntry1DockedPoses/
    #     for dir1 in dirs1:  # 1abc_2xyz
    #         for root2, dirs2, files2 in os.walk(os.path.join(base, dir1)):  # NanohedraEntry1DockedPoses/1abc_2xyz/
    #             for dir2 in dirs2:  # DEGEN_1_1
    #                 for root3, dirs3, files3 in os.walk(os.path.join(base, dir1, dir2)):  # NanohedraEntry1DockedPoses/1abc_2xyz/DEGEN_1_1/
    #                     for dir3 in dirs3:  # ROT_1_1
    #                         for root4, dirs4, files4 in os.walk(os.path.join(base, dir1, dir2, dir3)):  # NanohedraEntry1DockedPoses/1abc_2xyz/DEGEN_1_1/ROT_1_1/
    #                             for dir4 in dirs4: # tx_139
    #                                 if dir4.startswith('tx_'):

    # baseline testing with
    # timeit.timeit("import glob;glob.glob('*/*/*/*/')", number=1) Vs.
    # timeit.timeit("import glob;get_design_directories('/share/gscratch/kmeador/crystal_design/
    #     NanohedraEntry65MinMatched6_FULL')", setup="from __main__ import get_design_directories", number=1)
    # gives 2.4859059400041588 versus 13.074574943981133


def collect_directories(directory, file=None, dir_type='design'):
    """Grab all poses from an input source

    Args:
        directory (str): Disk location of the design directory
    Keyword Args:
        file=None (str): Disk location of file containing design directories
    Returns:
        (list), (location): All pose directories found, Path to where they were located
    """
    if file:
        _file = file
        if not os.path.exists(_file):
            _file = os.path.join(os.getcwd(), file)
            if not os.path.exists(_file):
                logger.critical('No %s file found in \'%s\'! Please ensure correct location/name!' % (file, directory))
                sys.exit(1)
        with open(_file, 'r') as f:
            all_directories = [location.strip() for location in f.readlines()]
        location = file
    else:
        if dir_type == 'dock':
            all_directories = get_docked_directories(directory)
        elif dir_type == 'design':
            base_directories = get_base_nanohedra_dirs(directory)
            all_directories = list(chain.from_iterable([get_docked_dirs_from_base(base) for base in base_directories]))
        location = directory

    return sorted(set(all_directories)), location


# # DEPRECIATED
# def get_dock_directories(base_directory, directory_type='vflip_dock.pkl'):  # removed a .pkl 9/29/20 9/17/20 run used .pkl.pkl TODO remove vflip
#     all_directories = []
#     for root, dirs, files in os.walk(base_directory):
#         # for _dir in dirs:
#         if 'master_log.txt' in files:
#             if file.endswith(directory_type):
#                 all_directories.append(root)
#     #
#     #
#     # return sorted(set(all_directories))
#
#     return sorted(set(map(os.path.dirname, glob('%s/*/*%s' % (base_directory, directory_type)))))


##############
# Alignments
##############


def find_gapped_columns(alignment_dict):
    target_seq_index = []
    n = 1
    for aa in alignment_dict['meta']['query']:
        if aa != '-':
            target_seq_index.append(n)
        n += 1

    return target_seq_index


def update_alignment_meta(alignment_dict):  # UNUSED UNFINISHED
    all_meta = []
    for alignment in alignment_dict:
        all_meta.append(alignment_dict[alignment]['meta'])

    meta_strings = ['' for i in range(len(next(all_meta)))]
    for meta in all_meta:
        j = 0
        for data in meta:
            meta_strings[j] += meta[data]

    return alignment_dict


def modify_index(count_dict, index_start=0):  # UNUSED NOT Working
    return {i + index_start: count_dict[i] for i in count_dict}


def modify_alignment_dict_index(alignment_dict, index=0):  # UNUSED UNFINISHED
    alignment_dict['counts'] = modify_index(alignment_dict['counts'], index_start=index)
    alignment_dict['rep'] = modify_index(alignment_dict['rep'], index_start=index)

    return alignment_dict


def merge_alignment_dicts(alignment_merge):  # UNUSED UNFINISHED
    length = [0]
    for i, alignment in enumerate(alignment_merge):
        alignment_dict = modify_alignment_dict_index(alignment_merge[alignment], index=length[i])
        length.append(len(alignment_merge[alignment]['meta']['query']))
        merged_alignment_dict = {'meta': update_alignment_meta(alignment)} # alignment_dict
    for alignment in alignment_merge:
        merged_alignment_dict.update(alignment_merge[alignment])

    return merged_alignment_dict


def clean_gapped_columns(alignment_dict, correct_index):
    """Cleans an alignment dictionary by revising key list with correctly indexed positions. 0 indexed"""
    return {i: alignment_dict[index] for i, index in enumerate(correct_index)}


def weight_sequences(msa_dict, alignment):
    """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment

    Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count)) as was described by Heinkoff and Heinkoff, 1994
    Args:
        msa_dict (dict): { 1: {'A': 31, 'C': 0, ...}, 2: {}, ...}
        alignment (biopython.MSA):
    Returns:
        seq_weight_dict (dict): { 1: 2.390, 2: 2.90, 3:5.33, 4: 1.123, ...} - sequence_in_MSA: sequence_weight_factor
    """
    col_tot_aa_count_dict = {}
    for i in range(len(msa_dict)):
        s = 0  # column amino acid representation
        for aa in msa_dict[i]:
            if aa == '-':
                continue
            elif msa_dict[i][aa] > 0:
                s += 1
        col_tot_aa_count_dict[i] = s

    seq_weight_dict = {}
    for k, record in enumerate(alignment):
        s = 0  # "diversity/surprise"
        for j, aa in enumerate(record.seq):
            s += (1 / (col_tot_aa_count_dict[j] * msa_dict[j][aa]))
        seq_weight_dict[k] = s

    return seq_weight_dict


def generate_msa_dictionary(alignment, alphabet=IUPACData.protein_letters, weighted_dict=None, weight=False):
    """Generate an alignment dictinary from a Biopython MultipleSeqAlignment object

    Args:
        alignment (MultipleSeqAlignment): List of SeqRecords
    Keyword Args:
        alphabet=IUPACData.protein_letters (str): 'ACDEFGHIKLMNPQRSTVWY'
        weighted_dict=None (dict): A weighted sequence dictionary with weights for each alignment sequence
        weight=False (bool): If weights should be used to weight the alignment
    Returns:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {0: {'A': 13, 'C': 1, 'D': 23, ...}, 1: {}, ...})
    """
    aligned_seq = str(alignment[0].seq)
    # Add Info to 'meta' record as needed
    alignment_dict = {'meta': {'num_sequences': len(alignment), 'query': aligned_seq.replace('-', ''),
                               'query_with_gaps': aligned_seq}}
    # Populate Counts Dictionary
    alignment_counts_dict = populate_design_dict(alignment.get_alignment_length(), alphabet, counts=True)
    if weight:
        for record in alignment:
            for i, aa in enumerate(record.seq):
                alignment_counts_dict[i][aa] += weighted_dict[i]
    else:
        for record in alignment:
            for i, aa in enumerate(record.seq):
                alignment_counts_dict[i][aa] += 1
    alignment_dict['counts'] = alignment_counts_dict

    return alignment_dict


def add_column_weight(counts_dict, gaps=False):
    """Find total representation for each column in the alignment

    Args:
        counts_dict (dict): {'counts': {0: {'A': 13, 'C': 1, 'D': 23, ...}, 1: {}, ...}
    Keyword Args:
        gaps=False (bool): Whether the alignment contains gaps
    Returns:
        counts_dict (dict): {0: 210, 1: 211, 2:211, ...}
    """
    return {i: sum_column_weight(counts_dict[i], gaps=gaps) for i in range(len(counts_dict))}


def sum_column_weight(column, gaps=False):
    """Sum the column weight for a single alignment dict column

    Args:
        column (dict): {'A': 13, 'C': 1, 'D': 23, ...}
    Keyword Args:
        gaps=False (bool): Whether to count gaps or not
    Returns:
        s (int): Total counts in the alignment
    """
    s = 0
    if gaps:
        for key in column:
            s += column[key]
    else:
        for key in column:
            if key == '-':
                continue
            else:
                s += column[key]

    return s


def msa_to_prob_distribution(alignment_dict):
    """Turn Alignment dictionary into a probability distribution

    Args:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {0: {'A': 13, 'C': 1, 'D': 23, ...}, 1: {}, ...},
                                'rep': {0: 210, 1: 211, 2:211, ...}}
    Returns:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {0: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 1: {}, ...},
                                'rep': {0: 210, 1: 211, 2:211, ...}}
    """
    for residue in alignment_dict['counts']:
        total_weight_in_column = alignment_dict['rep'][residue]
        assert total_weight_in_column != 0, '%s: Processing error... Downstream cannot divide by 0. Position = %s' % \
                                            (msa_to_prob_distribution.__name__, residue)
        for aa in alignment_dict['counts'][residue]:
            alignment_dict['counts'][residue][aa] /= total_weight_in_column
            # cleaned_msa_dict[i][aa] = round(cleaned_msa_dict[i][aa], 3)

    return alignment_dict


def compute_jsd(msa, bgd_freq, jsd_lambda=0.5):
    """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict

    Args:
        msa (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        bgd_freq (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        divergence (float): 0.732, Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    divergence_dict = {}
    for residue in msa:
        sum_prob1, sum_prob2 = 0, 0
        for aa in IUPACData.protein_letters:
            p = msa[residue][aa]
            q = bgd_freq[aa]
            r = (jsd_lambda * p) + ((1 - jsd_lambda) * q)
            if r == 0:
                continue
            if q != 0:
                prob2 = (q * math.log2(q / r))
                sum_prob2 += prob2
            if p != 0:
                prob1 = (p * math.log2(p / r))
                sum_prob1 += prob1
        divergence = jsd_lambda * sum_prob1 + (1 - jsd_lambda) * sum_prob2
        divergence_dict[residue] = round(divergence, 3)

    return divergence_dict


def weight_gaps(divergence, representation, alignment_length):
    for i in range(len(divergence)):
        divergence[i] = divergence[i] * representation[i] / alignment_length

    return divergence


def window_score(score_dict, window_len, score_lambda=0.5):
    """Takes a MSA score dict and transforms so that each position is a weighted average of the surrounding positions.
    Positions with scores less than zero are not changed and are ignored calculation

    Modified from Capra and Singh 2007 code
    Args:
        score_dict (dict):
        window_len (int): Number of residues on either side of the current residue
    Keyword Args:
        lamda=0.5 (float): Float between 0 and 1
    Returns:
        window_scores (dict):
    """
    if window_len == 0:
        return score_dict
    else:
        window_scores = {}
        for i in range(len(score_dict) + index_offset):
            s, number_terms = 0, 0
            if i <= window_len:
                for j in range(1, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            elif i + window_len > len(score_dict):
                for j in range(i - window_len, len(score_dict) + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            else:
                for j in range(i - window_len, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            window_scores[i] = (1 - score_lambda) * (s / number_terms) + score_lambda * score_dict[i]

        return window_scores


def rank_possibilities(probability_dict):
    """Gather alternative residues and sort them by probability.

    Args:
        probability_dict (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
    Returns:
         sorted_alternates_dict (dict): {15: ['S', 'A', 'T'], ... }
    """
    sorted_alternates_dict = {}
    for residue in probability_dict:
        residue_probability_list = []
        for aa in probability_dict[residue]:
            if probability_dict[residue][aa] > 0:
                residue_probability_list.append((aa, round(probability_dict[residue][aa], 5)))  # tuple instead of list
        residue_probability_list.sort(key=lambda tup: tup[1], reverse=True)
        # [('S', 0.13190), ('A', 0.0500), ...]
        sorted_alternates_dict[residue] = [aa[0] for aa in residue_probability_list]

    return sorted_alternates_dict


def process_alignment(bio_alignment_object, gaps=False):
    """Take a Biopython MultipleSeqAlignment object and process for residue specific information

    gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
    probability of residue i in the specific column more accurately.
    Args:
        bio_alignment_object (MultipleSeqAlignment): List of SeqRecords
    Keyword Args:
        gaps=False (bool): Whether gaps (-) should be counted in column weights
    Returns:
        probability_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                  'query_with_gaps': 'MGS---THLVLK...'}}
                                  'counts': {0: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 1: {}, ...},
                                  'rep': {0: 210, 1: 211, 2:211, ...}}
            Zero-indexed counts and rep dictionary elements
    """
    alignment_dict = generate_msa_dictionary(bio_alignment_object)
    alignment_dict['rep'] = add_column_weight(alignment_dict['counts'], gaps=gaps)
    probability_dict = msa_to_prob_distribution(alignment_dict)

    return probability_dict
