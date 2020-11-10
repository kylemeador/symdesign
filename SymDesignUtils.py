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
from itertools import repeat, chain
from json import loads, dumps

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import IUPACData
from Bio.SubsMat import MatrixInfo as matlist
from sklearn.neighbors import BallTree

import CmdUtils as CUtils
import PDB
import PathUtils as PUtils

# logging.getLogger().setLevel(logging.INFO)

# Globals
index_offset = 1
rmsd_threshold = 1.0
alph_3_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
aa_weight_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
                         'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
layer_groups = {'P 1': 'p1', 'P 2': 'p2', 'P 21': 'p21', 'C 2': 'pg', 'P 2 2 2': 'p222', 'P 2 2 21': 'p2221',
                'P 2 21 21': 'p22121', 'C 2 2 2': 'c222', 'P 4': 'p4', 'P 4 2 2': 'p422',
                'P 4 21 2': 'p4121', 'P 3': 'p3', 'P 3 1 2': 'p312', 'P 3 2 1': 'p321', 'P 6': 'p6', 'P 6 2 2': 'p622'}
viable = {'p6', 'p4', 'p3', 'p312', 'p4121', 'p622'}


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


def handle_symmetry(cryst1_record):
    group = cryst1_record.split()[-1]
    if group in layer_groups:
        symmetry = 2
        return 2
    else:
        return 3


def sdf_lookup(point_type):
    # TODO
    for root, dirs, files in os.walk(PUtils.symmetry_def_files):
        placeholder = None
    symm = os.path.join(PUtils.symmetry_def_files, 'dummy.symm')
    return symm


def scout_sdf_chains(pdb):
    """Search for the chains involved in a complex using a truncated make_symmdef_file.pl script

    perl $SymDesign/dependencies/rosetta/sdf/scout_symmdef_file.pl -p 3l8r_1ho1/DEGEN_1_1/ROT_36_1/tx_4/1ho1_tx_4.pdb
    -i B C D E F G H

    """
    num_chains = len(pdb.chain_id_list)
    scout_cmd = ['perl', PUtils.scout_symmdef, '-p', pdb.filepath, '-i'] + pdb.chain_id_list[1:]
    logger.info(subprocess.list2cmdline(scout_cmd))
    p = subprocess.run(scout_cmd, capture_output=True)
    lines = p.stdout.decode('utf-8').strip().split('\n')
    rotation_dict = {}
    max_sym, max_chain = 0, None
    for line in lines:
        chain = line[0]
        symmetry = int(line.split(':')[1][:6].rstrip('-fold'))
        axis = list(map(float, line.split(':')[2].strip().split()))
        rotation_dict[chain] = {'sym': symmetry, 'axis': np.array(axis)}
        if symmetry > max_sym:
            max_sym = symmetry
            max_chain = chain

    assert max_chain, logger.warning('%s: No symmetry found for SDF creation' % pdb.filepath)
    # if max_chain:
    #     chain_string = max_chain
    # else:
    #     raise DesignError('%s: No symmetry found for SDF creation' % pdb.filepath)

    # Check for dihedral symmetry, ensuring selected chain is orthogonal to max symmetry axis
    if num_chains / max_sym == 2:
        for chain in rotation_dict:
            if rotation_dict[chain]['sym'] == 2:
                if np.dot(rotation_dict[max_chain]['axis'], rotation_dict[chain]['axis']) < 0.01:
                    max_chain += ' ' + chain
                    break

    return max_chain


def make_sdf(pdb, modify_sym_energy=False, energy=2):
    """Use the make_symmdef_file.pl script from Rosetta on an input structure

    perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb -i B -q
    """
    chains = scout_sdf_chains(pdb)
    dihedral = False
    if len(chains) > 1:
        dihedral = True
    number_chains = len(pdb.chain_id_list)
    sdf_file_name = os.path.join(os.path.dirname(pdb.filepath), pdb.name + '.sdf')
    sdf_cmd = ['perl', PUtils.make_symmdef, '-p', pdb.filepath, '-i', chains, '-q']
    logger.info(subprocess.list2cmdline(sdf_cmd))
    with open(sdf_file_name, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=file, stderr=subprocess.DEVNULL)
        p.communicate()

    assert p.returncode == 0, logger.error('%s: Symmetry Definition File generation failed' % pdb.filepath)

    # Ensure proper formatting before proceeding # if dihedral:
    subunits, virtuals, jumps_com, jumps_subunit, trunk = [], [], [], [], []
    with open(sdf_file_name, 'r+') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('xyz'):
                virtual = lines[i].split()[1]
                if virtual.endswith('_base'):
                    subunits.append(virtual)
                else:
                    virtuals.append(virtual.lstrip('VRT'))
                # last_vrt = i + 1
            elif lines[i].startswith('connect_virtual'):
                jump = lines[i].split()[1].lstrip('JUMP')
                if jump.endswith('_to_com'):
                    jumps_com.append(jump[:-7])
                elif jump.endswith('_to_subunit'):
                    jumps_subunit.append(jump[:-11])
                else:
                    trunk.append(jump)
                last_jump = i + 1
        assert set(trunk) - set(virtuals) == set(), logger.error('%s: Symmetry Definition File VRTS are malformed'
                                                                 % pdb.filepath)
        assert number_chains == len(subunits), logger.error('%s: Symmetry Definition File VRTX_base are malformed'
                                                            % pdb.filepath)

        if dihedral:
            # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
            virtuals = [virtual for virtual in virtuals if len(virtual) > 1]  # subunit_
        else:
            if '' in virtuals:
                virtuals.remove('')

        jumps_com_to_add = set(virtuals) - set(jumps_com)
        count = 0
        if jumps_com_to_add != set():
            for jump_com in jumps_com_to_add:
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_com VRT%s VRT%s_base\n'
                             % (jump_com, jump_com, jump_com))
                count += 1
            lines[-2] = lines[-2].strip() + (len(jumps_com_to_add) * ' JUMP%s_to_subunit') \
                % tuple(jump_subunit for jump_subunit in jumps_com_to_add)
            lines[-2] += '\n'
        jumps_subunit_to_add = set(virtuals) - set(jumps_subunit)
        if jumps_subunit_to_add != set():
            for jump_subunit in jumps_subunit_to_add:
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_subunit VRT%s_base SUBUNIT\n'
                             % (jump_subunit, jump_subunit))
                count += 1
            lines[-1] = lines[-1].strip() + (len(jumps_subunit_to_add) * ' JUMP%s_to_subunit') \
                % tuple(jump_subunit for jump_subunit in jumps_subunit_to_add)
            lines[-1] += '\n'
        if modify_sym_energy:
            # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits,
            # so num_subunits - 1
            new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes that subunits are read in alphanumerical order
            new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
            lines[1] = new_energy + '\n'

        file.seek(0)
        for line in lines:
            file.write(line)
        file.truncate()
        if count != 0:
            logger.warning('%s: Symmetry Definition File for %s missing %d lines, fix was attempted. Modelling may be '
                           'affected for pose' % (os.path.dirname(pdb.filepath), os.path.basename(pdb.filepath), count))

    return sdf_file_name


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
def unpickle(filename):
    """Unpickle (deserialize) and return a python object located at filename"""
    with open(filename, 'rb') as serial_f:
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
    file_name = os.path.join(out_path, '%s.pkl' % name)
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


def reduce_pose_to_chains(pdb, chains):  # UNUSED
    new_pdb = PDB.PDB()
    new_pdb.read_atom_list(pdb.chains(chains))

    return new_pdb


def combine_pdb(pdb_1, pdb_2, name):  # UNUSED
    """Take two pdb objects and write them to the same file

    Args:
        pdb_1 (PDB): First PDB to concatentate
        pdb_2 (PDB): Second PDB
        name (str): Name of the output file
    """
    pdb_1.write(name)
    with open(name, 'a') as full_pdb:
        for atom in pdb_2.all_atoms:
            full_pdb.write(str(atom))  # .strip() + '\n')


def identify_interface_chains(pdb1, pdb2):  # UNUSED
    distance = 12  # Angstroms
    pdb1_chains = []
    pdb2_chains = []
    # Get Queried CB Tree for all PDB2 Atoms within 12A of PDB1 CB Atoms
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance)

    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_chains.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain)
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_chains.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain)

    pdb1_chains = list(set(pdb1_chains))
    pdb2_chains = list(set(pdb2_chains))

    return pdb1_chains, pdb2_chains


def rosetta_score(pdb):  # UNUSED
    # this will also format your output in rosetta numbering
    cmd = [PUtils.rosetta, 'score_jd2.default.linuxgccrelease', '-renumber_pdb', '-ignore_unrecognized_res', '-s', pdb,
           '-out:pdb']
    subprocess.Popen(cmd, start_new_session=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    return pdb + '_0001.pdb'


def duplicate_ssm(pssm_dict, copies):  # UNUSED
    duplicated_ssm = {}
    duplication_start = len(pssm_dict)
    for i in range(int(copies)):
        if i == 0:
            offset = 0
        else:
            offset = duplication_start * i
        # j = 0
        for line in pssm_dict:
            duplicated_ssm[line + offset] = pssm_dict[line]
            # j += 1

    return duplicated_ssm


def get_all_cluster(pdb, residue_cluster_id_list, db=PUtils.bio_fragmentDB):  # UNUSED DEPRECIATED
    # generate an interface specific scoring matrix from the fragment library
    # assuming residue_cluster_id_list has form [(1_2_24, [78, 87]), ...]
    cluster_list = []
    for cluster in residue_cluster_id_list:
        cluster_loc = cluster[0].split('_')
        res1 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[0], cluster[1][0]))
        res2 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[1], cluster[1][1]))
        filename = os.path.join(db, cluster_loc[0], cluster_loc[0] + '_' + cluster_loc[1], cluster_loc[0] +
                                '_' + cluster_loc[1] + '_' + cluster_loc[2], cluster[0] + '.pkl')
        cluster_list.append([[res1.ca, res2.ca], unpickle(filename)])

    # OUTPUT format: [[[residue1_ca_atom, residue2_ca_atom], {'IJKClusterDict - such as 1_2_45'}], ...]
    return cluster_list


def convert_to_frag_dict(interface_residue_list, cluster_dict):  #UNUSED
    # Make PDB/ATOM objects and dictionary into design dictionary
    # INPUT format: interface_residue_list = [[[Atom_ca.residue1, Atom_ca.residue2], '1_2_45'], ...]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        for i in range(len(residues)):
            residues[i] = residues[i].residue_number
        hash_ = (residues[0], residues[1])
        interface_residue_dict[hash_] = cluster_dict[residue_dict_pair[1]]
    # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
    return interface_residue_dict


def convert_to_rosetta_num(pdb, pose, interface_residue_list):  # UNUSED
    # DEPRECIATED in favor of updating PDB/ATOM objects
    # INPUT format: interface_residue_list = [[[78, 87], {'IJKClusterDict - such as 1_2_45'}], [[64, 87], {...}], ...]
    component_chains = [pdb.chain_id_list[0], pdb.chain_id_list[-1]]
    interface_residue_dict = {}
    for residue_dict_pair in interface_residue_list:
        residues = residue_dict_pair[0]
        dict_ = residue_dict_pair[1]
        new_key = []
        pair_index = 0
        for chain in component_chains:
            new_key.append(pose.pdb_info().pdb2pose(chain, residues[pair_index]))
            pair_index = 1
        hash_ = (new_key[0], new_key[1])

        interface_residue_dict[hash_] = dict_
    # OUTPUT format: interface_residue_dict = {(78, 256): {'IJKClusterDict - such as 1_2_45'}, (64, 256): {...}, ...}
    return interface_residue_dict


def get_residue_list_atom(pdb, residue_list, chain=None):  # UNUSED DEPRECIATED
    if chain is None:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        res_atoms = PDB.Residue(pdb.getResidueAtoms(chain, residue))
        residues.append(res_atoms)

    return residues


def get_residue_atom_list(pdb, residue_list, chain=None):  # UNUSED
    if chain is None:
        chain = pdb.chain_id_list[0]
    residues = []
    for residue in residue_list:
        residues.append(pdb.getResidueAtoms(chain, residue))

    return residues


def make_issm(cluster_freq_dict, background):  # UNUSED
    for residue in cluster_freq_dict:
        for aa in cluster_freq_dict[residue]:
            cluster_freq_dict[residue][aa] = round(2 * (math.log2((cluster_freq_dict[residue][aa] / background[aa]))))
    issm = []

    return issm


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


def clean_to_iterable(code):
    """Takes a file, a list or a string, and converts to cleaned list removing excess punctuation capitalizing string"""
    pdb_list = []
    try:
        with open(code, 'r') as f:
            pdb_list = f.readlines()
    except FileNotFoundError:
        if isinstance(code, list):
            pdb_list = code
        else:
            pdb_list.append(code)

    clean_list = []
    for pdb in pdb_list:
        pdb = pdb.strip().split(',')
        pdb = list(map(str.strip, pdb))
        for i in pdb:
            clean_list.append(i.upper())

    clean_list = list(set(clean_list))

    return clean_list


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
    clean_list = clean_to_iterable(pdb)

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
        if not current_file:  # glob should return an empty list
            # TODO subprocess.POPEN()
            status = os.system('wget -q -O %s http://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            if status != 0:
                failures.append(pdb)

    if failures:
        logger.error('PDB download ran into the following failures:\n%s' % ', '.join(failures))

    return file_name  # if list then will only return the last file


def fetch_pdb(code, location=PUtils.pdb_db):
    """Fetch PDB object of each chain from PDBdb or PDB server

        Args:
            code (iter): Any iterable of PDB codes
        Keyword Args:
            location= : Location of the  on disk
        Returns:
            (dict): {pdb_code: PDB.py object, ...}
        """
    if PUtils.pdb_source == 'download_pdb':
        get_pdb = download_pdb
        # doesn't return anything at the moment
    else:
        get_pdb = (lambda pdb_code, location=None: glob(os.path.join(PUtils.pdb_db, 'pdb%s.ent' % pdb_code.lower())))
        # The below set up is my local pdb and the format of escher. cassini is slightly different, ughhh
        # get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_db, subdirectory(pdb_code),
        #                                                      '%s.pdb' % pdb_code)))
        # returns a list with matching file (should only be one)

    # pdb_file = get_pdb(code, location)
    pdb_file = get_pdb(code, location=location)
    # pdb_file = get_pdb(code, location=des_dir.pdbs)
    assert len(pdb_file) == 1, 'More than one matching file found for PDB: %s' % code
    assert pdb_file != list(), 'No matching file found for PDB: %s' % code

    return pdb_file[0]
    # pdb = read_pdb(pdb_file[0])
    # pdb.AddName(code)
    # pdb.reorder_chains()
    #
    # return pdb


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
        oligomers[code].AddName(code)
        oligomers[code].reorder_chains()

    return oligomers


def read_pdb(file, coordinates_only=True):
    """Wrapper on the PDB __init__ and readfile functions

    Args:
        file (str): Disk location of pdb file
    Returns:
        pdb (PDB): Initialized PDB object
    """
    pdb = PDB.PDB()
    pdb.readfile(file, remove_alt_location=True, coordinates_only=coordinates_only)

    return pdb


def fill_pdb(atom_list=[]):
    """Wrapper on the PDB __init__ and readfile functions

    Args:
        atom_list (list): List of Atom objects
    Returns:
        pdb (PDB): Initialized PDB object
    """
    pdb = PDB.PDB()
    pdb.read_atom_list(atom_list)

    return pdb


def extract_asu(file, chain='A', outpath=None):
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


def find_interface_residues(pdb1, pdb2, dist=8):
    """Get Queried CB Tree for all PDB2 Atoms within 8A of PDB1 CB Atoms

    Args:
        pdb1 (PDB): First pdb to measure interface between
        pdb2 (PDB): Second pdb to measure interface between
    Keyword Args:
        dist=8 (int): The distance to query in Angstroms
    Returns:
        residues1 (list): A sorted list of unique residue numbers from pdb1
        residues2 (list): A sorted list of unique residue numbers from pdb2
    """
    query, pdb1_cb_indices, pdb2_cb_indices = construct_cb_atom_tree(pdb1, pdb2, distance=dist)

    # Map Coordinates to Residue Numbers
    residues1, residues2 = [], []
    for pdb2_index in range(len(query)):
        if query[pdb2_index].tolist() != list():
            residues2.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number)
            for pdb1_index in query[pdb2_index]:
                residues1.append(pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number)
    residues1 = sorted(set(residues1), key=int)
    residues2 = sorted(set(residues2), key=int)
    return residues1, residues2
    # return {pdb1.name: residues1, pdb2.name: residues2}


def print_atoms(atom_list):  # DEBUG
    for residue in atom_list:
        for atom in residue:
            logger.info(str(atom))


##################
# FRAGMENT DB
##################


def get_db_statistics(database):
    """Retrieve summary statistics for a specific fragment database

    Args:
        database (str): Disk location of a fragment database
    Returns:
        stats (dict): {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
            ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
    """
    for file in os.listdir(database):
        if file.endswith('statistics.pkl'):
            unpickle(os.path.join(database, file))

    return stats


def get_db_aa_frequencies(database):
    """Retrieve database specific interface background AA frequencies

    Args:
        database (str): Location of database on disk
    Returns:
        (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    """
    return get_db_statistics(database)['frequencies']


def get_cluster_dicts(db='biological_interfaces', id_list=None):  # TODO Rename
    """Generate an interface specific scoring matrix from the fragment library

    Args:
    Keyword Args:
        info_db=PUtils.biological_fragmentDB
        id_list=None: [1_2_24, ...]
    Returns:
         cluster_dict: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    """
    info_db = PUtils.frag_directory[db]
    if id_list is None:
        directory_list = get_all_base_root_paths(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        cluster_dict[os.path.basename(cluster)] = unpickle(filename)

    return cluster_dict


def return_cluster_id_string(cluster_rep, index_number=3):
    while len(cluster_rep) < 3:
        cluster_rep += '0'
    if len(cluster_rep.split('_')) != 3:
        index = [cluster_rep[:1], cluster_rep[1:2], cluster_rep[2:]]
    else:
        index = cluster_rep.split('_')

    info = []
    n = 0
    for i in range(index_number):
        info.append(index[i])
        n += 1
    while n < 3:
        info.append('0')
        n += 1

    return '_'.join(info)


def parameterize_frag_length(length):
    """Generate fragment length range parameters for use in fragment functions"""
    _range = math.floor(length / 2)
    if length % 2 == 1:
        return 0 - _range, 0 + _range + index_offset
    else:
        logger.critical('%d is an even integer which is not symmetric about a single residue. '
                        'Ensure this is what you want and modify %s' % (length, parameterize_frag_length.__name__))
        raise DesignError('Function not supported: Even fragment length \'%d\'' % length)


def format_frequencies(frequency_list, flip=False):
    """Format list of paired frequency data into parsable paired format

    Args:
        frequency_list (list): [(('D', 'A'), 0.0822), (('D', 'V'), 0.0685), ...]
    Keyword Args:
        flip=False (bool): Whether to invert the mapping of internal tuple
    Returns:
        (dict): {'A': {'S': 0.02, 'T': 0.12}, ...}
    """
    if flip:
        i, j = 1, 0
    else:
        i, j = 0, 1
    freq_d = {}
    for tup in frequency_list:
        aa_mapped = tup[0][i]  # 0
        aa_paired = tup[0][j]  # 1
        freq = tup[1]
        if aa_mapped in freq_d:
            freq_d[aa_mapped][aa_paired] = freq
        else:
            freq_d[aa_mapped] = {aa_paired: freq}

    return freq_d


def fragment_overlap(residues, interaction_graph, freq_map):
    """Take fragment contact list to find the possible AA types allowed in fragment pairs from the contact list

    Args:
        residues (iter): Iterable of residue numbers
        interaction_graph (dict): {52: [54, 56, 72, 206], ...}
        freq_map (dict): {(78, 87, ...): {'A': {'S': 0.02, 'T': 0.12}, ...}, ...}
    Returns:
        overlap (dict): {residue: {'A', 'I', 'M', 'V'}, ...}
    """
    overlap = {}
    for res in residues:
        overlap[res] = set()
        if res in interaction_graph:  # check for existence as some fragment info is not in the interface set
            # overlap[res] = set()
            for partner in interaction_graph[res]:
                if (res, partner) in freq_map:
                    overlap[res] |= set(freq_map[(res, partner)].keys())

    for res in residues:
        if res in interaction_graph:  # check for existence as some fragment info is not in the interface set
            for partner in interaction_graph[res]:
                if (res, partner) in freq_map:
                    overlap[res] &= set(freq_map[(res, partner)].keys())

    return overlap


def overlap_consensus(issm, aa_set):
    """Find the overlap constrained consensus sequence

    Args:
        issm (dict): {1: {'A': 0.1, 'C': 0.0, ...}, 14: {...}, ...}
        aa_set (dict): {residue: {'A', 'I', 'M', 'V'}, ...}
    Returns:
        (dict): {23: 'T', 29: 'A', ...}
    """
    consensus = {}
    for res in aa_set:
        max_freq = 0.0
        for aa in aa_set[res]:
            # if max_freq < issm[(res, partner)][]:
            if issm[res][aa] > max_freq:
                max_freq = issm[res][aa]
                consensus[res] = aa

    return consensus

######################
# Sequence Handling
######################


def populate_design_dict(n, alph, counts=False):
    """Return a dictionary with n elements and alph subelements.

    Args:
        n (int): number of residues in a design
        alph (iter): alphabet of interest
    Keyword Args:
        counts=False (bool): If true include an integer placeholder for counting
     Returns:
         (dict): {0: {alph1: {}, alph2: {}, ...}, 1: {}, ...}
            Custom length, 0 indexed dictionary with residue number keys
     """
    if counts:
        return {residue: {i: 0 for i in alph} for residue in range(n)}
    else:
        return {residue: {i: dict() for i in alph} for residue in range(n)}


def offset_index(dictionary, to_zero=False):
    """Modify the index of a sequence dictionary. Default is to one-indexed. to_zero=True gives zero-indexed"""
    if to_zero:
        return {residue - index_offset: dictionary[residue] for residue in dictionary}
    else:
        return {residue + index_offset: dictionary[residue] for residue in dictionary}


def residue_number_to_object(pdb, residue_dict):  # TODO supplement with names info and pull out by names
    """Convert sets of residue numbers to sets of PDB.Residue objects

    Args:
        pdb (PDB): PDB object to extract residues from. Chain order matches residue order in residue_dict
        residue_dict (dict): {'key1': [(78, 87, ...),], ...} - Entry mapped to residue sets
    Returns:
        residue_dict - {'key1': [(residue1_ca_atom, residue2_ca_atom, ...), ...] ...}
    """
    for entry in residue_dict:
        pairs = []
        for _set in range(len(residue_dict[entry])):
            residue_obj_set = []
            for i, residue in enumerate(residue_dict[entry][_set]):
                resi_object = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[i], residue)).ca
                assert resi_object, DesignError('Residue \'%s\' missing from PDB \'%s\'' % (residue, pdb.filepath))
                residue_obj_set.append(resi_object)
            pairs.append(tuple(residue_obj_set))
        residue_dict[entry] = pairs

    return residue_dict


def residue_object_to_number(residue_dict):  # TODO supplement with names info and pull out by names
    """Convert sets of PDB.Residue objects to residue numbers

    Args:
        pdb (PDB): PDB object to extract residues from. Chain order matches residue order in residue_dict
        residue_dict (dict): {'key1': [(residue1_ca_atom, residue2_ca_atom, ...), ...] ...}
    Returns:
        residue_dict (dict): {'key1': [(78, 87, ...),], ...} - Entry mapped to residue sets
    """
    for entry in residue_dict:
        pairs = []
        # for _set in range(len(residue_dict[entry])):
        for j, _set in enumerate(residue_dict[entry]):
            residue_num_set = []
            # for i, residue in enumerate(residue_dict[entry][_set]):
            for residue in _set:
                resi_number = residue.residue_number
                # resi_object = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[i], residue)).ca
                # assert resi_object, DesignError('Residue \'%s\' missing from PDB \'%s\'' % (residue, pdb.filepath))
                residue_num_set.append(resi_number)
            pairs.append(tuple(residue_num_set))
        residue_dict[entry] = pairs

    return residue_dict


def convert_to_residue_cluster_map(residue_cluster_dict, frag_range):
    """Make a residue and cluster/fragment index map

    Args:
        residue_cluster_dict (dict): {'1_2_45': [(residue1_ca_atom, residue2_ca_atom), ...] ...}
        frag_range (dict): A range of the fragment size to search over. Ex: (-2, 3) for fragments of length 5
    Returns:
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
            Where the key is the 0 indexed residue id
    """
    cluster_map = {}
    for cluster in residue_cluster_dict:
        for pair in range(len(residue_cluster_dict[cluster])):
            for i, residue_atom in enumerate(residue_cluster_dict[cluster][pair]):
                # for each residue in map add the same cluster to the range of fragment residue numbers
                residue_num = residue_atom.residue_number - index_offset  # zero index
                for j in range(*frag_range):
                    if residue_num + j not in cluster_map:
                        if i == 0:
                            cluster_map[residue_num + j] = {'chain': 'mapped', 'cluster': []}
                        else:
                            cluster_map[residue_num + j] = {'chain': 'paired', 'cluster': []}
                    cluster_map[residue_num + j]['cluster'].append((j, cluster))

    return cluster_map


def deconvolve_clusters(cluster_dict, design_dict, cluster_map):
    """Add frequency information from a fragment database to a design dictionary

    The frequency information is added in a fragment index dependent manner. If multiple fragment indices are present in
    a single residue, a new observation is created for that fragment index.

    Args:
        cluster_dict (dict): {1_1_54: {'mapped': {aa_freq}, 'paired': {aa_freq}}, ...}
            mapped/paired aa_dict = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1: {}, ...}
                Where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
        design_dict (dict): {0: {-2: {'A': 0.1, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    Returns:
        design_dict (dict): {0: {-2: {O: {'A': 0.1, 'C': 0.0, ...}, 1: {}}, -1: {}, ... }, 1: {}, ...}
    """

    for resi in cluster_map:
        dict_type = cluster_map[resi]['chain']
        observation = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for index_cluster_pair in cluster_map[resi]['cluster']:
            aa_freq = cluster_dict[index_cluster_pair[1]][dict_type][index_cluster_pair[0]]
            # Add the aa_freq from cluster to the residue/frag_index/observation
            try:
                design_dict[resi][index_cluster_pair[0]][observation[index_cluster_pair[0]]] = aa_freq
            except KeyError:
                raise DesignError('Missing residue %d in %s.' % (resi, deconvolve_clusters.__name__))
            observation[index_cluster_pair[0]] += 1
    
    return design_dict


def flatten_for_issm(design_cluster_dict, keep_extras=True):
    """Take a multi-observation, mulit-fragment index, fragment frequency dictionary and flatten to single frequency

    Args:
        design_cluster_dict (dict): {0: {-2: {'A': 0.1, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
            Dictionary containing fragment frequency and statistics across a design sequence
    Keyword Args:
        keep_extras=True (bool): If true, keep values for all design dictionary positions that are missing fragment data
    Returns:
        design_cluster_dict (dict): {0: {'A': 0.1, 'C': 0.0, ...}, 13: {...}, ...}
            Weighted average design dictionary combining all fragment profile information at a single residue
    """
    no_design = []
    for res in design_cluster_dict:
        total_residue_weight = 0
        num_frag_weights_observed = 0
        for index in design_cluster_dict[res]:
            if design_cluster_dict[res][index] != dict():
                total_obs_weight = 0
                for obs in design_cluster_dict[res][index]:
                    total_obs_weight += design_cluster_dict[res][index][obs]['stats'][1]
                if total_obs_weight > 0:
                    total_residue_weight += total_obs_weight
                    obs_aa_dict = copy.deepcopy(aa_weight_counts_dict)
                    obs_aa_dict['stats'][1] = total_obs_weight
                    for obs in design_cluster_dict[res][index]:
                        num_frag_weights_observed += 1
                        obs_weight = design_cluster_dict[res][index][obs]['stats'][1]
                        for aa in design_cluster_dict[res][index][obs]:
                            if aa != 'stats':
                                # Add all occurrences to summed frequencies list
                                obs_aa_dict[aa] += design_cluster_dict[res][index][obs][aa] * (obs_weight /
                                                                                               total_obs_weight)
                    design_cluster_dict[res][index] = obs_aa_dict
                else:
                    # Case where no weights associated with observations (side chain not structurally significant)
                    design_cluster_dict[res][index] = dict()

        if total_residue_weight > 0:
            res_aa_dict = copy.deepcopy(aa_weight_counts_dict)
            res_aa_dict['stats'][1] = total_residue_weight
            res_aa_dict['stats'][0] = num_frag_weights_observed
            for index in design_cluster_dict[res]:
                if design_cluster_dict[res][index] != dict():
                    index_weight = design_cluster_dict[res][index]['stats'][1]
                    for aa in design_cluster_dict[res][index]:
                        if aa != 'stats':
                            # Add all occurrences to summed frequencies list
                            res_aa_dict[aa] += design_cluster_dict[res][index][aa] * (index_weight / total_residue_weight)
            design_cluster_dict[res] = res_aa_dict
        else:
            # Add to list for removal from the design dict
            no_design.append(res)

    # Remove missing residues from dictionary
    if keep_extras:
        for res in no_design:
            design_cluster_dict[res] = aa_weight_counts_dict
    else:
        for res in no_design:
            design_cluster_dict.pop(res)

    return design_cluster_dict


def psiblast(query, outpath=None, remote=False):  # UNUSED
    """Generate an position specific scoring matrix using PSI-BLAST subprocess

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
        remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
    Returns:
        outfile_name (str): Name of the file generated by psiblast
        p (subprocess): Process object for monitoring progress of psiblast command
    """
    # I would like the background to come from Uniref90 instead of BLOSUM62 #TODO
    if outpath is not None:
        outfile_name = os.path.join(outpath, query + '.pssm')
        direct = outpath
    else:
        outfile_name = query + '.hmm'
        direct = os.getcwd()
    if query + '.pssm' in os.listdir(direct):
        cmd = ['echo', 'PSSM: ' + query + '.pssm already exists']
        p = subprocess.Popen(cmd)

        return outfile_name, p

    cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
           '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
    if remote:
        cmd.append('-remote')
    else:
        cmd.append('-num_threads')
        cmd.append('8')

    p = subprocess.Popen(cmd)

    return outfile_name, p


def hhblits(query, threads=CUtils.hhblits_threads, outpath=os.getcwd()):
    """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
        threads (int): Number of cpu's to use for the process
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
    Returns:
        outfile_name (str): Name of the file generated by hhblits
        p (subprocess): Process object for monitoring progress of hhblits command
    """

    outfile_name = os.path.join(outpath, os.path.splitext(os.path.basename(query))[0] + '.hmm')

    cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(threads)]
    logger.info('%s Profile Command: %s' % (query, subprocess.list2cmdline(cmd)))
    p = subprocess.Popen(cmd)

    return outfile_name, p


@handle_errors_f(errors=(FileNotFoundError, ))
def parse_pssm(file):
    """Take the contents of a pssm file, parse, and input into a pose profile dictionary.

    Resulting residue dictionary is zero-indexed
    Args:
        file (str): The name/location of the file on disk
    Returns:
        pose_dict (dict): Dictionary containing residue indexed profile information
            Ex: {0: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                {...}}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    for line in lines:
        line_data = line.strip().split()
        if len(line_data) == 44:
            resi = int(line_data[0]) - index_offset
            pose_dict[resi] = copy.deepcopy(aa_counts_dict)
            for i, aa in enumerate(alph_3_aa_list, 22):  # pose_dict[resi], 22):
                # Get normalized counts for pose_dict
                pose_dict[resi][aa] = (int(line_data[i]) / 100.0)
            pose_dict[resi]['lod'] = {}
            for i, aa in enumerate(alph_3_aa_list, 2):
                pose_dict[resi]['lod'][aa] = line_data[i]
            pose_dict[resi]['type'] = line_data[1]
            pose_dict[resi]['info'] = float(line_data[42])
            pose_dict[resi]['weight'] = float(line_data[43])

    return pose_dict


def get_lod(aa_freq_dict, bg_dict, round_lod=True):
    """Get the lod scores for an aa frequency distribution compared to a background frequency
    Args:
        aa_freq_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
        bg_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
    Keyword Args:
        round_lod=True (bool): Whether or not to round the lod values to an integer
    Returns:
         lods (dict): {'A': 2, 'C': -9, 'D': -1, ...}
    """
    lods = {}
    iteration = 0
    for a in aa_freq_dict:
        if aa_freq_dict[a] == 0:
            lods[a] = -9
        elif a != 'stats':
            lods[a] = float((2.0 * math.log2(aa_freq_dict[a]/bg_dict[a])))  # + 0.0
            if lods[a] < -9:
                lods[a] = -9
            if round_lod:
                lods[a] = round(lods[a])
            iteration += 1

    return lods


@handle_errors_f(errors=(FileNotFoundError, ))
def parse_hhblits_pssm(file, null_background=True):
    # Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order
    dummy = 0.00
    null_bg = {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228, 'I': 0.0534,
               'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599, 'S': 0.0791,
               'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}  # 'uniclust30_2018_08'

    def to_freq(value):
        if value == '*':
            # When frequency is zero
            return 0.0001
        else:
            # Equation: value = -1000 * log_2(frequency)
            freq = 2 ** (-int(value)/1000)
            return freq

    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    read = False
    for line in lines:
        if not read:
            if line[0:1] == '#':
                read = True
        else:
            if line[0:4] == 'NULL':
                if null_background:
                    # use the provided null background from the profile search
                    background = line.strip().split()
                    null_bg = {i: {} for i in alph_3_aa_list}
                    for i, aa in enumerate(alph_3_aa_list, 1):
                        null_bg[aa] = to_freq(background[i])

            if len(line.split()) == 23:
                items = line.strip().split()
                resi = int(items[1]) - index_offset  # make zero index so dict starts at 0
                pose_dict[resi] = {}
                for i, aa in enumerate(IUPACData.protein_letters, 2):
                    pose_dict[resi][aa] = to_freq(items[i])
                pose_dict[resi]['lod'] = get_lod(pose_dict[resi], null_bg)
                pose_dict[resi]['type'] = items[0]
                pose_dict[resi]['info'] = dummy
                pose_dict[resi]['weight'] = dummy

    # Output: {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
    # 'weight': 0.00}, {...}}
    return pose_dict


def make_pssm_file(pssm_dict, name, outpath=os.getcwd()):
    """Create a PSI-BLAST format PSSM file from a PSSM dictionary

    Args:
        pssm_dict (dict): A pssm dictionary which has the fields 'A', 'C', (all aa's), 'lod', 'type', 'info', 'weight'
        name (str): The name of the file
    Keyword Args:
        outpath=cwd (str): A specific location to write the .pssm file to
    Returns:
        out_file (str): Disk location of newly created .pssm file
    """
    lod_freq, counts_freq = False, False
    separation_string1, separation_string2 = 3, 3
    if type(pssm_dict[0]['lod']['A']) == float:
        lod_freq = True
        separation_string1 = 4
    if type(pssm_dict[0]['A']) == float:
        counts_freq = True

    header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in alph_3_aa_list) \
             + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in alph_3_aa_list) + '\n'
    footer = ''
    out_file = os.path.join(outpath, name)  # + '.pssm'
    with open(out_file, 'w') as f:
        f.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            lod_string = ''
            if lod_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
            counts_string = ''
            if counts_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3.0f} '.format(math.floor(pssm_dict[res][aa] * 100))
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3d} '.format(pssm_dict[res][aa])
            info = pssm_dict[res]['info']
            weight = pssm_dict[res]['weight']
            line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + index_offset, aa_type, lod_string,
                                                                             counts_string, round(info, 4),
                                                                             round(weight, 4))
            f.write(line)
        f.write(footer)

    return out_file


def combine_pssm(pssms):
    """To a first pssm, append subsequent pssms incrementing the residue number in each additional pssm

    Args:
        pssms (list(dict)): List of pssm dictionaries to concatentate
    Returns:
        combined_pssm (dict): Concatentated PSSM
    """
    combined_pssm = {}
    new_key = 0
    for i in range(len(pssms)):
        # requires python 3.6+ to maintain sorted dictionaries
        # for old_key in pssms[i]:
        for old_key in sorted(list(pssms[i].keys())):
            combined_pssm[new_key] = pssms[i][old_key]
            new_key += 1

    return combined_pssm


def combine_ssm(pssm, issm, alpha, db='biological_interfaces', favor_fragments=True, boltzmann=False, a=0.5):
    """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

    All input must be zero indexed
    Args:
        pssm (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W',
            'info': 0.00, 'weight': 0.00}, {...}}
              PSIBLAST -  {0: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
                          'weight': 0.73}, {...}} CURRENTLY IMPOSSIBLE, NEED TO CHANGE THE LOD SCORE IN PARSING
        issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        alpha (dict): {48: 0.5, 50: 0.321, ...}
    Keyword Args:
        db='biological_interfaces': Disk location of fragment database
        favor_fragments=True (bool): Whether to favor fragment profile in the lod score of the resulting profile
        boltzmann=True (bool): Whether to weight the fragment profile by the Boltzmann probability. If false, residues
            are weighted by a local maximum over the residue scaled to a maximum provided in the standard Rosetta per
            residue reference weight.
        a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
    Returns:
        pssm (dict): {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
            'weight': 0.00}, ...}} - combined PSSM dictionary
    """

    # Combine fragment and evolutionary probability profile according to alpha parameter
    for entry in alpha:
        for aa in IUPACData.protein_letters:
            pssm[entry][aa] = (alpha[entry] * issm[entry][aa]) + ((1 - alpha[entry]) * pssm[entry][aa])
        logger.info('Residue %d Combined evolutionary and fragment profile: %.0f%% fragment'
                    % (entry + index_offset, alpha[entry] * 100))

    if favor_fragments:
        # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
        # Used to weight fragments higher in design
        boltzman_energy = 1
        favor_seqprofile_score_modifier = 0.2 * CUtils.reference_average_residue_weight
        db = PUtils.frag_directory[db]
        stat_dict_bkg = get_db_aa_frequencies(db)
        null_residue = get_lod(stat_dict_bkg, stat_dict_bkg)
        null_residue = {aa: float(null_residue[aa]) for aa in null_residue}

        for entry in pssm:
            pssm[entry]['lod'] = null_residue
        for entry in issm:
            pssm[entry]['lod'] = get_lod(issm[entry], stat_dict_bkg, round_lod=False)
            partition, max_lod = 0, 0.0
            for aa in pssm[entry]['lod']:
                # for use with a boltzman probability weighting, Z = sum(exp(score / kT))
                if boltzmann:
                    pssm[entry]['lod'][aa] = math.exp(pssm[entry]['lod'][aa] / boltzman_energy)
                    partition += pssm[entry]['lod'][aa]
                # remove any lod penalty
                elif pssm[entry]['lod'][aa] < 0:
                    pssm[entry]['lod'][aa] = 0
                # find the maximum/residue (local) lod score
                if pssm[entry]['lod'][aa] > max_lod:
                    max_lod = pssm[entry]['lod'][aa]
            modified_entry_alpha = (alpha[entry] / a) * favor_seqprofile_score_modifier
            if boltzmann:
                modifier = partition
                modified_entry_alpha /= (max_lod / partition)
            else:
                modifier = max_lod
            for aa in pssm[entry]['lod']:
                pssm[entry]['lod'][aa] /= modifier
                pssm[entry]['lod'][aa] *= modified_entry_alpha
            logger.info('Residue %d Fragment lod ratio generated with alpha=%f'
                        % (entry + index_offset, alpha[entry] / a))

    return pssm


def find_alpha(issm, cluster_map, db='biological_interfaces', a=0.5):
    """Find fragment contribution to design with cap at alpha

    Args:
        issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    Keyword Args:
        db='biological_interfaces': Disk location of fragment database
        a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
    Returns:
        alpha (dict): {48: 0.5, 50: 0.321, ...}
    """
    db = PUtils.frag_directory[db]
    stat_dict = get_db_statistics(db)
    alpha = {}
    for entry in issm:  # cluster_map
        if cluster_map[entry]['chain'] == 'mapped':
            i = 0
        else:
            i = 1

        contribution_total = 0.0
        for count, residue_cluster_pair in enumerate(cluster_map[entry]['cluster'], 1):
            cluster_id = return_cluster_id_string(residue_cluster_pair[1], index_number=2)
            contribution_total += stat_dict[cluster_id][0][i]
        stats_average = contribution_total / count
        entry_ave_frag_weight = issm[entry]['stats'][1] / count  # total weight for issm entry / number of fragments
        if entry_ave_frag_weight < stats_average:  # if design frag weight is less than db cluster average weight
            # modify alpha proportionally to cluster average weight
            alpha[entry] = a * (entry_ave_frag_weight / stats_average)
        else:
            alpha[entry] = a

    return alpha


def consensus_sequence(pssm):
    """Return the consensus sequence from a PSSM

    Args:
        pssm (dict): pssm dictionary
    Return:
        consensus_identities (dict): {1: 'M', 2: 'H', ...} One-indexed
    """
    consensus_identities = {}
    for residue in pssm:
        max_lod = 0
        max_res = pssm[residue]['type']
        for aa in alph_3_aa_list:
            if pssm[residue]['lod'][aa] > max_lod:
                max_lod = pssm[residue]['lod'][aa]
                max_res = aa
        consensus_identities[residue + index_offset] = max_res

    return consensus_identities


def sequence_difference(seq1, seq2, d=None, matrix='blosum62'):  # TODO AMS
    """Returns the sequence difference between two sequence iterators

    Args:
        seq1 (any): Either an iterable with residue type as array, or key, with residue type as d[seq1][residue]['type']
        seq2 (any): Either an iterable with residue type as array, or key, with residue type as d[seq2][residue]['type']
    Keyword Args:
        d=None (dict): The dictionary to look up seq1 and seq2 if they are keys and the iterable is a dictionary
        matrix='blosum62' (str): The type of matrix to score the sequence differences on
    Returns:
        (float): The computed sequence difference between seq1 and seq2
    """
    # s = 0
    if d:
        # seq1 = d[seq1]
        # seq2 = d[seq2]
        # for residue in d[seq1]:
            # s.append((d[seq1][residue]['type'], d[seq2][residue]['type']))
        pairs = [(d[seq1][residue]['type'], d[seq2][residue]['type']) for residue in d[seq1]]
    else:
        pairs = [(seq1_res, seq2[i]) for i, seq1_res in enumerate(seq1)]
            # s.append((seq1[i], seq2[i]))
    #     residue_iterator1 = seq1
    #     residue_iterator2 = seq2
    m = getattr(matlist, matrix)
    s = 0
    for tup in pairs:
        try:
            s += m[tup]
        except KeyError:
            s += m[(tup[1], tup[0])]

    return s


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
    while True:
        _input = input('Enter P to print Data, W to write Data to file, or B for both:')
        if _input == 'W':
            if not filename:
                filename = input('What is your desired filename? (appended to current directory)\n')
                filename = os.path.join(os.getcwd(), filename)
            with open(filename, 'w') as f:
                f.write('\n'.join(data))
            print('File \'%s\' was written' % filename)
            break
        elif _input == 'P':
            print(data)
            break
        elif _input == 'B':
            print(data)
            if not filename:
                filename = input('What is your desired filename? (appended to current directory)\n')
                filename = os.path.join(os.getcwd(), filename)
            with open(filename, 'w') as f:
                f.write('\n'.join(data))
            print('File \'%s\' was written' % filename)
            break
        else:
            print('Invalid Input...')


def to_iterable(_obj):
    """Take a file/object and return a list of individual objects splitting on newline, space, or comma"""
    _list = []
    try:
        with open(_obj, 'r') as f:
            _list = f.readlines()
    except FileNotFoundError:
        if isinstance(_obj, list):
            _list = _obj
        else:
            _list.append(_obj)

    clean_list = []
    for it in _list:
        # pdb = pdb.strip()
        it_list = it.split(',')
        # if isinstance(pdb, list):
        # pdb = list(map(str.strip(), pdb))
        clean_list + [_it.strip() for _it in it_list]
        # else:  # unreachable
        #     clean_list.append(pdb.upper())  # unreachable
    # clean_list = list(set(clean_list))

    return clean_list


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


@handle_errors_f(errors=(FileNotFoundError, ))
def gather_docking_metrics(log_file):
    with open(log_file, 'r') as master_log:  # os.path.join(base_directory, 'master_log.txt')
        parameters = master_log.readlines()
        for line in parameters:
            if "PDB 1 Directory Path: " in line:
                pdb_dir1_path = line.split(':')[-1].strip()
            elif "PDB 2 Directory Path: " in line:
                pdb_dir2_path = line.split(':')[-1].strip()
            elif 'Master Output Directory: ' in line:
                master_outdir = line.split(':')[-1].strip()
            elif "Symmetry Entry Number: " in line:
                sym_entry_number = int(line.split(':')[-1].strip())
            elif "Oligomer 1 Symmetry: " in line:
                oligomer_symmetry_1 = line.split(':')[-1].strip()
            elif "Oligomer 2 Symmetry: " in line:
                oligomer_symmetry_2 = line.split(':')[-1].strip()
            elif "Design Point Group Symmetry: " in line:
                design_symmetry = line.split(':')[-1].strip()
            elif "Oligomer 1 Internal ROT DOF: " in line:  # ,
                internal_rot1 = line.split(':')[-1].strip()
            elif "Oligomer 2 Internal ROT DOF: " in line:  # ,
                internal_rot2 = line.split(':')[-1].strip()
            elif "Oligomer 1 ROT Sampling Range: " in line:
                rot_range_deg_pdb1 = int(line.split(':')[-1].strip())
            elif "Oligomer 2 ROT Sampling Range: " in line:
                rot_range_deg_pdb2 = int(line.split(':')[-1].strip())
            elif "Oligomer 1 ROT Sampling Step: " in line:
                rot_step_deg1 = int(line.split(':')[-1].strip())
            elif "Oligomer 2 ROT Sampling Step: " in line:
                rot_step_deg2 = int(line.split(':')[-1].strip())
            elif "Oligomer 1 Internal Tx DOF: " in line:  # ,
                internal_zshift1 = line.split(':')[-1].strip()
            elif "Oligomer 2 Internal Tx DOF: " in line:  # ,
                internal_zshift2 = line.split(':')[-1].strip()
            elif "Oligomer 1 Reference Frame Tx DOF: " in line:  # ,
                ref_frame_tx_dof1 = line.split(':')[-1].strip()
            elif "Oligomer 2 Reference Frame Tx DOF: " in line:  # ,
                ref_frame_tx_dof2 = line.split(':')[-1].strip()
            elif "Oligomer 1 Setting Matrix: " in line:
                set_mat1 = np.array(eval(line.split(':')[-1].strip()))
            elif "Oligomer 2 Setting Matrix: " in line:
                set_mat2 = np.array(eval(line.split(':')[-1].strip()))
            elif "Resulting Design Symmetry: " in line:
                result_design_sym = line.split(':')[-1].strip()
            elif "Design Dimension: " in line:
                design_dim = int(line.split(':')[-1].strip())
            elif "Unit Cell Specification: " in line:
                uc_spec_string = line.split(':')[-1].strip()
            elif 'Degeneracies Found for Oligomer 1' in line:
                degen1 = line.split()[0]
                if degen1.isdigit():
                    degen1 = int(degen1) + 1
                else:
                    degen1 = 1  # No degens becomes a single degen
            elif 'Degeneracies Found for Oligomer 2' in line:
                degen2 = line.split()[0]
                if degen2.isdigit():
                    degen2 = int(degen2) + 1
                else:
                    degen2 = 1  # No degens becomes a single degen

    return pdb_dir1_path, pdb_dir2_path, master_outdir, sym_entry_number, oligomer_symmetry_1, oligomer_symmetry_2,\
        design_symmetry, internal_rot1, internal_rot2, rot_range_deg_pdb1, rot_range_deg_pdb2, rot_step_deg1, \
        rot_step_deg2, internal_zshift1, internal_zshift2, ref_frame_tx_dof1, ref_frame_tx_dof2, set_mat1, set_mat2,\
        result_design_sym, design_dim, uc_spec_string, degen1, degen2


def pdb_input_parameters(args):
    return args[0:1]


def symmetry_parameters(args):
    return args[3:6]


def rotation_parameters(args):
    return args[9:13]


def degeneracy_parameters(args):
    return args[-2:]


def degen_and_rotation_parameters(args):
    return degeneracy_parameters(args), rotation_parameters(args)


def compute_last_rotation_state(range1, range2, step1, step2):
    number_steps1 = range1 / step1
    number_steps2 = range2 / step2

    return int(number_steps1), int(number_steps2)


@handle_errors_f(errors=(FileNotFoundError, ))
def gather_fragment_metrics(_des_dir, init=False, score=False):
    """Gather docking metrics from Nanohedra output
    Args:
        _des_dir (DesignDirectory): DesignDirectory Object
    Keyword Args:
        init=False (bool): Whether the information requested is for pose initialization
        score=False (bool): Whether to return the score information only
    Returns:
        (dict): Either {'nanohedra_score': , 'average_fragment_z_score': , 'unique_fragments': }
            transform_d {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}
            when clusters=False or {'1_2_24': [(78, 87, ...), ...], ...}
    """
    with open(os.path.join(_des_dir.path, PUtils.frag_file), 'r') as f:
        frag_match_info_file = f.readlines()
        residue_cluster_d, transform_d, z_value_dict = {}, {}, {}
        for line in frag_match_info_file:
            if line[:12] == 'Cluster ID: ':
                cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
                if cluster not in residue_cluster_d:
                    # residue_cluster_d[cluster] = []  # TODO make compatible
                    residue_cluster_d[cluster] = {'pair': []}
                continue
            elif line[:40] == 'Cluster Central Residue Pair Frequency: ':
                # pair_freq = loads(line[40:])
                # pair_freq = list(eval(line[40:].lstrip('[').rstrip(']')))  # .split(', ')
                pair_freq = list(eval(line[40:]))  # .split(', ')
                # pair_freq = list(map(eval, pair_freq_list))
                residue_cluster_d[cluster]['freq'] = pair_freq
                continue
            # Cluster Central Residue Pair Frequency:
            # [(('L', 'Q'), 0.2429), (('A', 'D'), 0.0571), (('V', 'D'), 0.0429), (('L', 'E'), 0.0429),
            # (('T', 'L'), 0.0429), (('L', 'S'), 0.0429), (('T', 'D'), 0.0429), (('V', 'L'), 0.0286),
            # (('I', 'K'), 0.0286), (('V', 'E'), 0.0286), (('L', 'L'), 0.0286), (('L', 'M'), 0.0286),
            # (('L', 'K'), 0.0286), (('T', 'Q'), 0.0286), (('S', 'D'), 0.0286), (('Y', 'G'), 0.0286),
            # (('I', 'F'), 0.0286), (('T', 'K'), 0.0286), (('V', 'I'), 0.0143), (('W', 'I'), 0.0143),
            # (('V', 'Q'), 0.0143), (('I', 'L'), 0.0143), (('F', 'G'), 0.0143), (('E', 'H'), 0.0143),
            # (('L', 'D'), 0.0143), (('N', 'M'), 0.0143), (('K', 'D'), 0.0143), (('L', 'H'), 0.0143),
            # (('L', 'V'), 0.0143), (('L', 'R'), 0.0143)]

            elif line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                # Always contains I fragment? #JOSH
                res_chain1 = int(line[43:].strip())
                continue
            elif line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                # Always contains J fragment and Guide Atoms? #JOSH
                res_chain2 = int(line[43:].strip())
                # residue_cluster_d[cluster].append((res_chain1, res_chain2))
                residue_cluster_d[cluster]['pair'].append((res_chain1, res_chain2))
                continue
            elif line[:17] == 'Overlap Z-Value: ':
                try:
                    z_value_dict[cluster] = float(line[17:].strip())
                except ValueError:
                    print('%s has misisng Z-value in frag_info_file.txt' % _des_dir)
                    z_value_dict[cluster] = float(1.0)
                continue
            elif line[:17] == 'Nanohedra Score: ':
                nanohedra_score = float(line[17:].strip())
                continue
        #             if line[:39] == 'Unique Interface Fragment Match Count: ':
        #                 int_match = int(line[39:].strip())
        #             if line[:39] == 'Unique Interface Fragment Total Count: ':
        #                 int_total = int(line[39:].strip())
            elif line[:20] == 'ROT/DEGEN MATRIX PDB':
                # _matrix = np.array(loads(line[23:]))
                _matrix = np.array(eval(line[23:]))
                transform_d[int(line[20:21])] = {'rot/deg': _matrix}  # dict[pdb# (1, 2)] = {'transform_type': matrix}
                continue
            elif line[:15] == 'INTERNAL Tx PDB':  # without PDB1 or PDB2
                # _matrix = np.array(loads(line[18:]))
                _matrix = np.array(eval(line[18:]))
                transform_d[int(line[15:16])]['tx_int'] = _matrix
                continue
            elif line[:18] == 'SETTING MATRIX PDB':
                # _matrix = np.array(loads(line[21:]))
                _matrix = np.array(eval(line[21:]))
                transform_d[int(line[18:19])]['setting'] = _matrix
                continue
            elif line[:21] == 'REFERENCE FRAME Tx PDB':
                # _matrix = np.array(loads(line[24:]))
                _matrix = np.array(eval(line[24:]))
                transform_d[int(line[21:22])]['tx_ref'] = _matrix
                continue
            elif 'Residue-Level Summation Score:' in line:
                score = float(line[30:].rstrip())
            # elif line[:23] == 'ROT/DEGEN MATRIX PDB1: ':
            # elif line[:18] == 'INTERNAL Tx PDB1: ':  # with PDB1 or PDB2
            # elif line[:21] == 'SETTING MATRIX PDB1: ':
            # elif line[:24] == 'REFERENCE FRAME Tx PDB1: ':

            # ROT/DEGEN MATRIX PDB1: [[1.0, -0.0, 0], [0.0, 1.0, 0], [0, 0, 1]]
            # INTERNAL Tx PDB1: [0, 0, 45.96406061067895]
            # SETTING MATRIX PDB1: [[0.707107, 0.408248, 0.57735], [-0.707107, 0.408248, 0.57735], [0.0, -0.816497, 0.57735]]
            # REFERENCE FRAME Tx PDB1: None

    if init:
        for cluster in residue_cluster_d:
            residue_cluster_d[cluster]['pair'] = list(set(residue_cluster_d[cluster]['pair']))

        return residue_cluster_d, transform_d
    elif score:
        return score
    else:
        fragment_z_total = 0
        for cluster in z_value_dict:
            fragment_z_total += z_value_dict[cluster]
        num_fragments = len(z_value_dict)
        ave_z = fragment_z_total / num_fragments
        return {'nanohedra_score': nanohedra_score, 'average_fragment_z_score': ave_z,
                'unique_fragments': num_fragments}  # , 'int_total': int_total}


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


class DesignDirectory:

    def __init__(self, directory, mode='design', auto_structure=True, symmetry=None):
        self.mode = mode
        self.path = directory
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
        self.symmetry = None
        # design_symmetry (P432)
        self.protein_data = None  # TODO
        # design_symmetry/protein_data (P432/Protein_Data)
        self.pdbs = None  # TODO
        # design_symmetry/protein_data/pdbs (P432/Protein_Data/PDBs)
        self.sequences = None
        # design_symmetry/sequences (P432/Sequence_Info)
        # design_symmetry/protein_data/sequences (P432/Protein_Data/Sequence_Info)  # TODO
        self.all_scores = None
        # design_symmetry/all_scores (P432/All_Scores)
        self.trajectories = None
        # design_symmetry/all_scores/str(self)_Trajectories.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Trajectories.csv)
        self.residues = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Residues.csv)
        self.design_sequences = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Sequences.pkl)
        self.building_blocks = None
        # design_symmetry/building_blocks (P432/4ftd_5tch)
        self.building_block_logs = []
        self.scores = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.design_pdbs = None  # TODO .designs?
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/rosetta_pdbs)
        self.frags = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/matching_fragment_representatives
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.data = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data
        self.source = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/central_asu.pdb
        self.asu = None
        self.oligomer_names = []
        self.oligomers = {}
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu.pdb
        self.info = {}
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data/stats.pkl
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.log = None
        # v ^ both used in dock_dir set up
        self.building_block_logs = None

        if auto_structure:
            if symmetry:
                if len(self.path.split(os.sep)) == 1:
                    self.directory_string_to_path()
            if self.mode == 'design':
                self.design_directory_structure(symmetry=symmetry)
            elif self.mode == 'dock':
                self.dock_directory_structure(symmetry=symmetry)

    def __str__(self):
        if self.symmetry:
            return self.path.replace(self.symmetry + os.sep, '').replace(os.sep, '-')  # TODO how integrate with designDB?
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]

    def directory_string_to_path(self):  # string, symmetry
        self.path = self.path.replace('-', os.sep)

    def design_directory_structure(self, symmetry=None):
        # Prepare Output Directory/Files. path always has format:
        if symmetry:
            self.symmetry = symmetry.rstrip(os.sep)
            self.path = os.path.join(symmetry, self.path)
        else:
            self.symmetry = self.path[:self.path.find(self.path.split(os.sep)[-4]) - 1]

        self.protein_data = os.path.join(self.symmetry, 'Protein_Data')
        self.pdbs = os.path.join(self.protein_data, 'PDBs')
        self.sequences = os.path.join(self.protein_data, PUtils.sequence_info)
        self.all_scores = os.path.join(self.symmetry, 'All_' + PUtils.scores_outdir.title())  # TODO db integration
        self.trajectories = os.path.join(self.all_scores, '%s_Trajectories.csv' % self.__str__())
        self.residues = os.path.join(self.all_scores, '%s_Residues.csv' % self.__str__())
        self.design_sequences = os.path.join(self.all_scores, '%s_Sequences.pkl' % self.__str__())
        self.building_blocks = self.path[:self.path.find(self.path.split(os.sep)[-3]) - 1]
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.design_pdbs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)

        self.source = os.path.join(self.path, PUtils.asu)
        self.asu = os.path.join(self.path, PUtils.clean)

        if not os.path.exists(self.path):
            # raise DesignError('Path does not exist!\n%s' % self.path)
            logger.warning('%s: Path does not exist!' % self.path)
        else:
            # TODO ensure these are only created with Pose Processing is called... New method probably
            if not os.path.exists(self.protein_data):
                os.makedirs(self.protein_data)
            if not os.path.exists(self.pdbs):
                os.makedirs(self.pdbs)
            if not os.path.exists(self.sequences):
                os.makedirs(self.sequences)
            if not os.path.exists(self.all_scores):
                os.makedirs(self.all_scores)
            if not os.path.exists(self.scores):
                os.makedirs(self.scores)
            if not os.path.exists(self.design_pdbs):
                os.makedirs(self.design_pdbs)
            if not os.path.exists(self.data):
                os.makedirs(self.data)
            else:
                if os.path.exists(os.path.join(self.data, 'info.pkl')):

                    # raise DesignError('%s: No information found for pose. Have you initialized it?\n'
                    #                   'Try \'python %s ... pose ...\' or inspect the directory for correct files' %
                    #                   (self.path, PUtils.program_name))
                    self.info = unpickle(os.path.join(self.data, 'info.pkl'))

    def dock_directory_structure(self, symmetry=None):
        """Saves the path of the docking directory as DesignDirectory.path attribute. Tries to populate further using
        typical directory structuring"""
        # dock_dir.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implementation?
        self.symmetry = glob(os.path.join(self.path, 'NanohedraEntry*DockedPoses%s' % str(symmetry or '')))  # for design_recap
        self.log = [os.path.join(_sym, 'master_log.txt') for _sym in self.symmetry]  # TODO PUtils
        for k, _sym in enumerate(self.symmetry):
            self.building_blocks.append(list())
            self.building_block_logs.append(list())
            # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
            for bb_dir in next(os.walk(_sym))[1]:  # grabs the directories from os.walk, yielding just top level results
                if os.path.exists(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils
                    self.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
                    self.building_blocks[k].append(bb_dir)

    def get_oligomers(self):
        if self.mode == 'design':
            self.oligomer_names = os.path.basename(self.building_blocks).split('_')
            for name in self.oligomer_names:
                name_pdb_file = glob(os.path.join(self.path, '%s*_tx_*.pdb' % name))
                assert len(name_pdb_file) == 1, 'Incorrect match [%d != 1] found using %s*_tx_*.pdb!\nCheck %s' % \
                                                (len(name_pdb_file), name, self.__str__())
                self.oligomers[name] = read_pdb(name_pdb_file[0])
                self.oligomers[name].AddName(name)
                self.oligomers[name].reorder_chains()

    # TODO generators for the various directory levels using the stored directory pieces
    def get_building_block_dir(self, building_block):
        for sym_idx, symm in enumerate(self.symmetry):
            try:
                bb_idx = self.building_blocks[sym_idx].index(building_block)
                return os.path.join(self.symmetry[sym_idx], self.building_blocks[sym_idx][bb_idx])
            except ValueError:
                continue
        return None

    def return_symmetry_stats(self):
        return len(symm for symm in self.symmetry)

    def return_building_block_stats(self):
        return len(bb for symm_bb in self.building_blocks for bb in symm_bb)

    def return_unique_pose_stats(self):
        return len(bb for symm in self.building_blocks for bb in symm)

    def start_log(self, name=None, level=2):
        _name = __name__
        if name:
            _name = name
        self.log = start_log(name=_name, handler=2, level=level,
                             location=os.path.join(self.path, os.path.basename(self.path)))


def set_up_directory_objects(design_list, mode='design', symmetry=None):
    """Create DesignDirectory objects from a directory iterable. Add symmetry if using DesignDirectory strings"""
    return [DesignDirectory(design, mode=mode, symmetry=symmetry) for design in design_list]


def set_up_dock_dir(path, suffix=None):  # DEPRECIATED
    """Saves the path of the docking directory as DesignDirectory.path attribute. Tries to populate further using
    typical directory structuring"""
    dock_dir = DesignDirectory(path, auto_structure=False)
    # try:
    # dock_dir.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implementation
    dock_dir.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses%s' % str(suffix or '')))  # design_recap
    dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.symmetry]  # TODO change to PUtils
    # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
    dock_dir.building_blocks, dock_dir.building_block_logs = [], []
    for k, _sym in enumerate(dock_dir.symmetry):
        dock_dir.building_blocks.append(list())
        dock_dir.building_block_logs.append(list())
        for bb_dir in next(os.walk(_sym))[1]:
            if os.path.exists(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils
                dock_dir.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
                dock_dir.building_blocks[k].append(bb_dir)

    # dock_dir.building_blocks = [next(os.walk(dir))[1] for dir in dock_dir.symmetry]
    # dock_dir.building_block_logs = [[os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)  # make a log path TODO PUtils
    #                                  for bb_dir in dock_dir.building_blocks[k]]  # for each building_block combo in _sym index of dock_dir.building_blocks
    #                                 for k, _sym in enumerate(dock_dir.symmetry)]  # for each sym in symmetry

    return dock_dir


def set_up_pseudo_design_dir(path, directory, score):  # changed 9/30/20 to locate paths of interest at .path
    pseudo_dir = DesignDirectory(path, auto_structure=False)
    # pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(path)
    pseudo_dir.design_pdbs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir


def get_pose_by_id(design_directories, ids):
    return [des_dir for des_dir in design_directories if str(des_dir) in ids]


def get_all_base_root_paths(directory):
    dir_paths = []
    for root, dirs, files in os.walk(directory):
        if not dirs:
            dir_paths.append(root)

    return dir_paths


def get_all_pdb_file_paths(pdb_dir):
    filepaths = []
    for root, dirs, files in os.walk(pdb_dir):
        for file in files:
            if '.pdb' in file:
                filepaths.append(os.path.join(root, file))

    return filepaths


def get_directory_pdb_file_paths(pdb_dir):
    return glob(os.path.join(pdb_dir, '*.pdb*'))


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


# DEPRECIATED
def get_dock_directories(base_directory, directory_type='vflip_dock.pkl'):  # removed a .pkl 9/29/20 9/17/20 run used .pkl.pkl TODO remove vflip
    all_directories = []
    for root, dirs, files in os.walk(base_directory):
        # for _dir in dirs:
        if 'master_log.txt' in files:
            if file.endswith(directory_type):
                all_directories.append(root)
    #
    #
    # return sorted(set(all_directories))

    return sorted(set(map(os.path.dirname, glob('%s/*/*%s' % (base_directory, directory_type)))))


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
