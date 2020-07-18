import os
import sys
import math
import subprocess
import logging
import pickle
import copy
import glob
from json import loads, dumps
import numpy as np
import multiprocessing as mp
import sklearn.neighbors
from itertools import repeat
import PDB
from Bio.SeqUtils import IUPACData
from Bio.SubsMat import MatrixInfo as matlist
from Bio import pairwise2
import PathUtils as PUtils
import CmdUtils as CUtils
# logging.getLogger().setLevel(logging.INFO)

# Globals
index_offset = 1
alph_3_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
aa_weight_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
                         'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
layer_groups = {'P 1': 'p1', 'P 2': 'p2', 'P 21': 'p21', 'C 2': 'pg', 'P 2 2 2': 'p222', 'P 2 2 21': 'p2221',
                'P 2 21 21': 'p22121', 'C 2 2 2': 'c222', 'P 4': 'p4', 'P 4 2 2': 'p422',
                'P 4 21 2': 'p4121', 'P 3': 'p3', 'P 3 1 2': 'p312', 'P 3 2 1': 'p321', 'P 6': 'p6', 'P 6 2 2': 'p622'}
viable = {'p6', 'p4', 'p3', 'p312', 'p4121', 'p622'}


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


def unpickle(filename):
    """Unpickle (deserialize) and return a python object located at filename"""
    if os.path.getsize(filename) > 0:
        with open(filename, 'rb') as infile:
            new_object = pickle.load(infile)

        return new_object
    else:
        return None


def pickle_object(target_object, name, out_path=os.getcwd()):
    """Pickle (serialize) an object into a file named 'out_path/name.pkl'

    Args:
        target_object (any): Any python object
        name (str): The name of the pickled file
    Keyword Args:
        out_path=os.getcwd() (str): Where the file should be written
    Returns:
        (str): The pickled filename
    """
    with open(os.path.join(out_path, name + '.pkl'), 'wb') as f:
        pickle.dump(target_object, f, pickle.HIGHEST_PROTOCOL)

    return os.path.join(out_path + name) + '.pkl'


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
        filename = os.path.join(db, cluster_loc[0], cluster_loc[0] + '_' + cluster_loc[1], cluster_loc[0] +
                                '_' + cluster_loc[1] + '_' + cluster_loc[2], cluster[0] + '.pkl')
        res1 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[0], cluster[1][0]))
        res2 = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[1], cluster[1][1]))
        with open(filename, 'rb') as f:
            cluster_list.append([[res1.ca, res2.ca], pickle.load(f)])

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


###################
# PDB Handling # TODO PDB.py
###################


def read_pdb(file):
    """Wrapper on the PDB __init__ and readfile functions
    Args:
        file (str): disk location of pdb file
    Returns:
        pdb (PDB): Initialized PDB object
    """
    pdb = PDB.PDB()
    pdb.readfile(file)

    return pdb


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
    pdb1_tree = sklearn.neighbors.BallTree(pdb1_coords)

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
            with open(os.path.join(database, file), 'rb') as f:
                stats = pickle.load(f)

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
        pssm (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                         'weight': 0.00}, {...}}
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
        pssm (dict): combined PSSM dictionary
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
        consensus_identities (dict): {1: 'M', 2: 'H', ...}
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


def gather_profile_info(pdb, des_dir, names):
    """For a given PDB, find the chain wise profile (pssm) then combine into one continuous pssm

    Args:
        pdb (PDB): PDB to generate a profile from. Sequence is taken from the ATOM record
        des_dir (DesignDirectory): Location of which to write output files in the design tree
        names (dict): The pdb names and corresponding chain of each protomer in the pdb object
        log_stream (logging): Which log to pass logging directives to
    Returns:
        pssm_file (str): Location of the combined pssm file written to disk
        full_pssm (dict): A combined pssm with all chains concatenated in the same order as pdb sequence
    """
    pssm_files, pdb_seq, errors, pdb_seq_file, pssm_process = {}, {}, {}, {}, {}
    logger.debug('Fetching PSSM Files')

    # Extract/Format Sequence Information
    for n, name in enumerate(names):
        # if pssm_files[name] == dict():
        logger.debug('%s is chain %s in ASU' % (name, names[name](n)))
        pdb_seq[name], errors[name] = extract_aa_seq(pdb, chain=names[name](n))
        logger.debug('%s Sequence=%s' % (name, pdb_seq[name]))
        if errors[name]:
            logger.warning('Sequence generation ran into the following residue errors: %s' % ', '.join(errors[name]))
        pdb_seq_file[name] = write_fasta_file(pdb_seq[name], name + '_' + os.path.basename(des_dir.path),
                                              outpath=des_dir.sequences)
        if not pdb_seq_file[name]:
            logger.error('Unable to parse sequence. Check if PDB \'%s\' is valid.' % name)
            raise DesignError('Unable to parse sequence in %s' % des_dir.path)

    # Make PSSM of PDB sequence POST-SEQUENCE EXTRACTION
    for name in names:
        logger.info('Generating PSSM file for %s' % name)
        pssm_files[name], pssm_process[name] = hhblits(pdb_seq_file[name], outpath=des_dir.sequences)
        logger.debug('%s seq file: %s' % (name, pdb_seq_file[name]))

    # Wait for PSSM command to complete
    for name in names:
        pssm_process[name].communicate()

    # Extract PSSM for each protein and combine into single PSSM
    pssm_dict = {}
    for name in names:
        pssm_dict[name] = parse_hhblits_pssm(pssm_files[name])
    full_pssm = combine_pssm([pssm_dict[name] for name in pssm_dict])
    pssm_file = make_pssm_file(full_pssm, PUtils.msa_pssm, outpath=des_dir.path)

    return pssm_file, full_pssm


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

def write_shell_script(command, name='script', outpath=os.getcwd(), additional=None):
    with open(os.path.join(outpath, name + '.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('%s\n' % command)
        if additional:
            f.write('%s\n' % '\n'.join(x for x in additional))
            # f.write('%s\n' % additional)


def write_commands(command_list, name='all_commands', loc=os.getcwd()):
    file = os.path.join(loc, name + '.cmd')
    with open(file, 'w') as f:
        for command in command_list:
            f.write(command + '\n')

    return file


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


def gather_fragment_metrics(_des_dir, clusters=False):
    """Gather docking metrics from Nanohedra output
    Args:
        _des_dir (DesignDirectory): DesignDirectory Object
    Returns:
        (dict): Either {'nanohedra_score': , 'average_fragment_z_score': , 'unique_fragments': } when clusters=False or
            {'1_2_24': [(78, 87, ...), ...], ...}
    """
    with open(os.path.join(_des_dir.path, PUtils.frag_file), 'r') as f:
        frag_match_info_file = f.readlines()
        residue_cluster_dict, z_value_dict = {}, {}
        for line in frag_match_info_file:
            if line[:12] == 'Cluster ID: ':
                cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
                if cluster not in residue_cluster_dict:
                    residue_cluster_dict[cluster] = []
                continue
            if line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                # Always contains I fragment? #JOSH
                res_chain1 = int(line[43:].strip())
                continue
            if line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                # Always contains J fragment and Guide Atoms? #JOSH
                res_chain2 = int(line[43:].strip())
                residue_cluster_dict[cluster].append((res_chain1, res_chain2))
                continue
            if line[:17] == 'Overlap Z-Value: ':
                z_value_dict[cluster] = float(line[17:].strip())
                continue
            if line[:17] == 'Nanohedra Score: ':
                nanohedra_score = float(line[17:].strip())
                continue
        #             if line[:39] == 'Unique Interface Fragment Match Count: ':
        #                 int_match = int(line[39:].strip())
        #             if line[:39] == 'Unique Interface Fragment Total Count: ':
        #                 int_total = int(line[39:].strip())
    if clusters:
        return residue_cluster_dict
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


def mp_map(function, process_args, threads=1):
    """Maps input to a function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
        threads (int): How many workers/threads should be spawned to handle function(arguments)?
    Returns:
        results (list): The results produced from the function and process_args
    """
    with mp.get_context('spawn').Pool(processes=threads, maxtasksperchild=1) as p:
        results = p.map(function, process_args)
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
    with mp.get_context(context).Pool(processes=threads, maxtasksperchild=1) as p:
        results = p.starmap(function, process_args)
    p.join()

    return results


def mp_starmap(function, process_args, threads=1, context='spawn'):
    """Maps iterable to a function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
        threads (int): How many workers/threads should be spawned to handle function(arguments)?
    Keyword Args:
        context='spawn' (str): One of 'spawn', 'fork', or 'forkserver'
    Returns:
        results (list): The results produced from the function and process_args
    """
    with mp.get_context(context).Pool(processes=threads, maxtasksperchild=1) as p:
        results = p.starmap(function, process_args)  # , chunksize=1
    p.join()

    return results


##########
# ERRORS
##########

class DesignError(Exception):

    def __init__(self, message):
        self.args = message


def handle_errors(errors=(Exception, )):
    """Decorator to wrap a function with try: ... except errors: finally:

    Keyword Args:
        errors=(Exception, ) (tuple): A tuple of exceptions to monitor
    Returns:
        return, error (tuple): [0] is function return upon proper execution, else None, tuple[1] is error if exception
            raised, else None
    """
    def wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs), None
            except errors as e:
                return None, (args[0].path, e)
            # finally:  TODO figure out how to run only when uncaught exception is found
            #     print('Error occurred in %s' % args[0].path)
        return wrapped
    return wrapper


######################
# Directory Handling
######################


class DesignDirectory:

    def __init__(self, directory, auto_structure=True, symmetry=None):
        self.symmetry = None
        # design_symmetry (P432)
        self.sequences = None
        # design_symmetry/sequences (P432/Sequence_Info)
        self.all_scores = None
        # design_symmetry/all_scores (P432/All_Scores)
        self.building_blocks = None
        # design_symmetry/building_blocks (P432/4ftd_5tch)
        self.path = directory
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
        self.scores = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.design_pdbs = None  # TODO .designs
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/rosetta_pdbs)
        self.frags = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/matching_fragment_representatives
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.data = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.log = None
        if auto_structure:
            if symmetry:
                if len(self.path.split(os.sep)) == 1:
                    self.directory_string_to_path()
            self.make_directory_structure(symmetry=symmetry)

    def __str__(self):
        if self.symmetry:
            return self.path.replace(self.symmetry + os.sep, '').replace(os.sep, '-')  # TODO integration with DB what to do?
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]

    def directory_string_to_path(self):  # string, symmetry
        self.path = self.path.replace('-', os.sep)

    def make_directory_structure(self, symmetry=None):
        # Prepare Output Directory/Files. path always has format:
        if symmetry:
            self.symmetry = symmetry.rstrip(os.sep)
            self.path = os.path.join(symmetry, self.path)
        else:
            self.symmetry = self.path[:self.path.find(self.path.split(os.sep)[-4]) - 1]
        self.sequences = os.path.join(self.symmetry, PUtils.sequence_info)
        self.all_scores = os.path.join(self.symmetry, 'All_' + PUtils.scores_outdir.title())
        self.building_blocks = self.path[:self.path.find(self.path.split(os.sep)[-3]) - 1]
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.design_pdbs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)

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

    def start_log(self, name=None, level=2):
        _name = __name__
        if name:
            _name = name
        self.log = start_log(name=_name, handler=2, level=level,
                             location=os.path.join(self.path, os.path.basename(self.path)))


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
            if file.endswith('.pdb'):
                filepaths.append(os.path.join(root, file))

    return filepaths


def get_directory_pdb_file_paths(pdb_dir):
    return glob.glob(os.path.join(pdb_dir, '*.pdb'))


def collect_designs(directory, file=None):
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
        location = file
        with open(_file, 'r') as f:
            all_directories = [location.strip() for location in f.readlines()]
    else:
        location = directory
        all_directories = get_design_directories(directory)

    return all_directories, location


def get_design_directories(base_directory):
    """Returns a sorted list of all unique directories that contain designable poses

    Args:
        base_directory (str): Location on disk to search for Nanohedra.py poses
    Returns:
        all_design_directories (list): List containing all paths to designable poses
    """
    all_design_directories = []
    for design_root, design_dirs, design_files in os.walk(base_directory):
        if os.path.basename(design_root).startswith(PUtils.pose_prefix):
            all_design_directories.append(design_root)
        else:
            for directory in design_dirs:
                if directory.startswith(PUtils.pose_prefix):
                    all_design_directories.append(os.path.join(design_root, directory))
    return sorted(set(all_design_directories))


def set_up_directory_objects(design_list, symmetry=None):
    """Create DesignDirectory objects from a directory iterable. Add symmetry if using DesignDirectory strings"""
    return [DesignDirectory(design, symmetry=symmetry) for design in design_list]


def set_up_pseudo_design_dir(wildtype, directory, score):
    pseudo_dir = DesignDirectory(wildtype, auto_structure=False)
    pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(wildtype)
    pseudo_dir.design_pdbs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir


#####################
# Sequence handling
#####################


def write_fasta_file(sequence, name, outpath=os.getcwd()):  # , multi_sequence=False):
    """Write a fasta file from sequence(s)

    Args:
        sequence (iterable): One of either list, dict, or string. If list, can be list of tuples(seq second),
            list of lists, etc. Smart solver using object type
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
    Returns:
        (str): The name of the output file
    """
    file_name = os.path.join(outpath, name + '.fasta')
    with open(file_name, 'w') as outfile:
        if type(sequence) is list:
            # if multi_sequence:
            #     with open(outfile_name, 'w') as outfile:
            #         for seq in sequence:
            #             header = '>' + seq[0] + '\n'
            #             line = seq[2] + '\n'
            #             outfile.write(header + line)
            if type(sequence[0]) is list:  # Where inside list is of alphabet (AA or DNA)
                for idx, seq in enumerate(sequence):
                    outfile.write('>%s_%d\n' % (name, idx))  # header
                    if len(seq[0]) == 3:  # Check if alphabet is 3 letter protein
                        outfile.write(' '.join(aa for aa in seq))
                    else:
                        outfile.write(''.join(aa for aa in seq))
            elif isinstance(sequence[0], str):
                outfile.write('>%s\n%s\n' % name, ' '.join(aa for aa in sequence))
            elif type(sequence[0]) is tuple:  # where
                # for seq in sequence:
                #     header = seq[0]
                #     line = seq[1]
                outfile.write('\n'.join('>%s\n%s' % seq for seq in sequence))
            else:
                raise DesignError('Cannot parse data to make fasta')
        elif isinstance(sequence, dict):
            f.write('\n'.join('>%s\n%s' % (seq_name, sequences[seq_name]) for seq_name in sequences))
        elif isinstance(sequence, str):
            outfile.write('>%s\n%s\n' % (name, sequence))
        else:
            raise DesignError('Cannot parse data to make fasta')

    return file_name


def write_multi_line_fasta_file(sequences, name, path=os.getcwd()):  # REDUNDANT DEPRECIATED
    """Write a multi-line fasta file from a dictionary where the keys are >headers and values are sequences

    Args:
        sequences (dict): {'my_protein': 'MSGFGHKLGNLIGV...', ...}
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
    Returns:
        (str): The name of the output file
    """
    file_name = os.path.join(path, name)
    with open(file_name, 'r') as f:
        # f.write('>%s\n' % seq)
        f.write('\n'.join('>%s\n%s' % (seq_name, sequences[seq_name]) for seq_name in sequences))

    return file_name


def extract_aa_seq(pdb, aa_code=1, source='atom', chain=0):
    # Extracts amino acid sequence from either ATOM or SEQRES record of PDB object
    if type(chain) == int:
        chain = pdb.chain_id_list[chain]
    final_sequence = None
    sequence_list = []
    failures = []
    aa_code = int(aa_code)

    if source == 'atom':
        # Extracts sequence from ATOM records
        if aa_code == 1:
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and (atom.alt_location == '' or atom.alt_location == 'A'):
                    try:
                        sequence_list.append(IUPACData.protein_letters_3to1[atom.residue_type.title()])
                    except KeyError:
                        sequence_list.append('X')
                        failures.append((atom.residue_number, atom.residue_type))
            final_sequence = ''.join(sequence_list)
        elif aa_code == 3:
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and atom.alt_location == '' or atom.alt_location == 'A':
                    sequence_list.append(atom.residue_type)
            final_sequence = sequence_list
        else:
            logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))

    elif source == 'seqres':
        # Extract sequence from the SEQRES record
        fail = False
        while True:
            if chain in pdb.sequence_dictionary:
                sequence = pdb.sequence_dictionary[chain]
                break
            else:
                if not fail:
                    temp_pdb = PDB.PDB()
                    temp_pdb.readfile(pdb.filepath, coordinates_only=False)
                    fail = True
                else:
                    raise DesignError('Invalid PDB input, no SEQRES record found')
        if aa_code == 1:
            final_sequence = sequence
            for i in range(len(sequence)):
                if sequence[i] == 'X':
                    failures.append((i, sequence[i]))
        elif aa_code == 3:
            for i, residue in enumerate(sequence):
                sequence_list.append(IUPACData.protein_letters_1to3[residue])
                if residue == 'X':
                    failures.append((i, residue))
            final_sequence = sequence_list
        else:
            logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))
    else:
        raise DesignError('Invalid sequence input')

    return final_sequence, failures


def pdb_to_pose_num(reference_dict):
    """Take a dictionary with chain name as keys and return the length of values as reference length"""
    offset_dict = {}
    prior_chain, prior_chains_len = None, 0
    for i, chain in enumerate(reference_dict):
        if i > 0:
            prior_chains_len += len(reference_dict[prior_chain])
        offset_dict[chain] = prior_chains_len
        # insert function here? Make this a decorator!?

        # for pdb in mutation_dict:
        #     if i == 0:
        #         new_mutation_dict[pdb] = {}
        #     new_mutation_dict[pdb][chain] = {}
        #     for mutation in mutation_dict[pdb][chain]:
        #         # mutation_dict[pdb][chain][mutation + prior_chain_len] = mutation_dict[pdb][chain].pop(mutation)
        #         new_mutation_dict[pdb][chain][mutation + prior_chain_len] = mutation_dict[pdb][chain][mutation]
        prior_chain = chain

    return offset_dict


def extract_sequence_from_pdb(pdb_class_dict, aa_code=1, seq_source='atom', mutation=False, pose_num=True,
                              outpath=None):  # offset=True
    """Extract the sequence from PDB objects

    Args:
        pdb_class_dict (dict): {pdb_code: PDB object, ...}
    Keyword Args:
        aa_code=1 (int): Whether to return sequence with one-letter or three-letter code [1,3]
        seq_source='atom' (str): Whether to return the ATOM or SEQRES record ['atom','seqres','compare']
        mutations=False (bool): Whether to return mutations in sequences compared to a reference.
            Specified by pdb_code='ref'
        outpath=None (str): Where to save the results to disk
    Returns:
        mutation_dict (dict): IF mutations=True {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...},
            ...}
        or
        sequence_dict (dict): ELSE {pdb: {chain_id: 'Sequence', ...}, ...}
    """
    reference_seq_dict = None
    if mutation:
        # If looking for mutations, the reference PDB object should be given as 'ref' in the dictionary
        if 'ref' not in pdb_class_dict:
            sys.exit('No reference sequence specified, but mutations requested. Include a key \'ref\' in PDB dict!')
        reference_seq_dict = {}
        fail_ref = []
        reference = pdb_class_dict['ref']
        #     # TEST to see if the length check is necessary
        # if len(chain_dict['ref']) > 1:
        for chain in pdb_class_dict['ref'].chain_id_list:
            reference_seq_dict[chain], fail = extract_aa_seq(reference, aa_code, seq_source, chain)
            if fail != list():
                fail_ref.append((reference, chain, fail))
        # else:
        #     reference_seq_dict[chain_dict['ref']], fail = extract_aa_seq(reference, aa_code, seq_source,
        #                                                                  chain_dict['ref'])
        if fail_ref:
            logger.error('Ran into following errors generating mutational analysis reference:\n%s' % str(fail_ref))

    if seq_source == 'compare':
        mutation = True

    def handle_extraction(pdb_code, _pdb, _aa, _source, _chain):
        if _source == 'compare':
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, 'atom', _chain)
            sequence2, failures2 = extract_aa_seq(_pdb, _aa, 'seqres', _chain)
        else:
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, _source, _chain)
            sequence2 = reference_seq_dict[_chain]
            sequence_dict[pdb_code][_chain] = sequence1
        if mutation:
            seq_mutations = generate_mutations_from_seq(sequence1, sequence2, offset=False, remove_blanks=False)
            # offset=offset
            mutation_dict[pdb_code][_chain] = seq_mutations
        if failures1:
            error_list.append((_pdb, _chain, failures1))

    error_list = []
    sequence_dict = {}
    mutation_dict = {}
    for pdb in pdb_class_dict:
        sequence_dict[pdb] = {}
        mutation_dict[pdb] = {}
        # if pdb == 'ref':
        # if len(chain_dict[pdb]) > 1:
        #     for chain in chain_dict[pdb]:
        for chain in pdb_class_dict[pdb].chain_id_list:
            handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain)
        # else:
        #     handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain_dict[pdb])

    if outpath:
        sequences = {}
        for pdb in sequence_dict:
            for chain in sequence_dict[pdb]:
                sequences[pdb + '_' + chain] = sequence_dict[pdb][chain]
        filepath = write_multi_line_fasta_file(sequences, 'sequence_extraction.fasta', path=outpath)
        logger.info('The following file was written:\n%s' % filepath)

    if error_list:
        logger.error('The following residues were not extracted:\n%s' % str(error_list))

    if mutation:
        # if offset:
        #     pass
        # else:
        # all_mutated_residues = {}
        # for chain in reference_seq_dict:
        #     all_mutated_residues[chain] = []
        #     for pdb in mutation_dict:
        #         for mutation in mutation_dict[pdb][chain]:
        #             all_mutated_residues[chain].append(mutation)
        #     # for chain in reference_seq_dict:
        #     all_mutated_residues[chain] = set(all_mutated_residues[chain])
        #     # print(all_mutated_residues[chain])
        #     for mutation in all_mutated_residues[chain]:
        #         mutation_dict['ref'][chain][mutation] = {'from': reference_seq_dict[chain][mutation - index_offset],
        #                                                  'to': reference_seq_dict[chain][mutation - index_offset]}
        for chain in reference_seq_dict:
            for i, aa in enumerate(reference_seq_dict[chain]):
                mutation_dict['ref'][chain][i + index_offset] = {'from': reference_seq_dict[chain][i],
                                                                 'to': reference_seq_dict[chain][i]}
        if pose_num:
            new_mutation_dict = {}
            # prior_chain, prior_chain_len = None, 0
            # for i, chain in enumerate(reference_seq_dict):
            #     if i > 0:
            #         prior_chain_len += len(reference_seq_dict[prior_chain])
            offset_dict = pdb_to_pose_num(reference_seq_dict)
            for chain in offset_dict:
                for pdb in mutation_dict:
                    if pdb not in new_mutation_dict:
                        new_mutation_dict[pdb] = {}
                    new_mutation_dict[pdb][chain] = {}
                    for mutation in mutation_dict[pdb][chain]:
                        # new_mutation_dict[pdb][chain][mutation + prior_chain_len] = mutation_dict[pdb][chain][mutation]
                        new_mutation_dict[pdb][chain][mutation + offset_dict[chain]] = mutation_dict[pdb][chain][mutation]
                # prior_chain = chain
            mutation_dict = new_mutation_dict

        return mutation_dict
    else:
        return sequence_dict


def make_sequences_from_mutations(wild_type, mutation_dict, aligned=False, output=False, name=None):  # TODO AMS
    """Takes a list of sequence mutations and returns the mutated form on wildtype

    Args:
        wild_type (str): Sequence to mutate
        mutation_dict (dict): {name: {mutation_index: {'from': AA, 'to': AA}, ...}, ...}, ...}
    Keyword Args:
        aligned=False (bool): Whether the input sequences are already aligned
        output=False (bool): Whether to make a fasta file of the sequence
    Returns:
        all_sequences (dict): {name: sequence, ...}
    """
    all_sequences = {}
    for pdb in mutation_dict:
        all_sequences[pdb] = make_mutations(wild_type, mutation_dict[pdb], find_orf=not aligned)

    if output:
        _file_list = []
        for seq in all_sequences:
            seq_file = write_fasta_file(all_sequences[seq], name, multi_sequence=True)
            _file_list.append(seq_file)
        filepath = write_multi_line_fasta_file(all_sequences, pdb)
        return filepath
    else:
        return all_sequences


def make_mutations(seq, mutation_dict, find_orf=True):  # TODO AMS
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Args:
        seq (str): 'Wild-type' sequence to mutate
        mutation_dict (dict): {mutation_index: {'from': AA, 'to': AA}, ...}
    Keyword Args:
        find_orf=True (bool): Whether or not to fing the correct ORF for the mutations and the seq
    Returns:
        seq (str): The mutated sequence
    """
    # Seq can be either list or string
    def find_orf_offset():
        met_offset_list = []
        for i, aa in enumerate(seq):
            if aa == 'M':
                met_offset_list.append(i)
        if met_offset_list:
            # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
            which_met_offset_counts = []
            for index in met_offset_list:
                index -= index_offset
                s = 0
                for mut in mutation_dict:
                    try:
                        if seq[mut + index] == mutation_dict[mut][0]:
                            s += 1
                    except IndexError:
                        break
                which_met_offset_counts.append(s)
            max_count = np.max(which_met_offset_counts)
        else:
            max_count = 0

        # Check if likely ORF has been identified (count < number mutations/2). If not, MET is missing/not the ORF start
        if max_count < len(mutation_dict) / 2:
            upper_range = 50  # This corresponds to how far away the max seq start is from the ORF MET start site
            offset_list = []
            for i in range(0, upper_range):
                s = 0
                for mut in mutation_dict:
                    if seq[mut + i] == mutation_dict[mut]['from']:
                        s += 1
                offset_list.append(s)
            max_count = np.max(offset_list)
            # find likely orf offset index
            orf_offset = offset_list.index(max_count)  # + lower_range  # + mut_index_correct
        else:
            orf_offset = met_offset_list[which_met_offset_counts.index(max_count)] - index_offset

        return orf_offset

    if find_orf:
        offset = -find_orf_offset()
        logger.info('Found ORF. Offset = %d' % -offset)
    else:
        offset = index_offset

    # zero index seq and 1 indexed mutation_dict
    index_errors = []
    for key in mutation_dict:
        try:
            if seq[key - offset] == mutation_dict[key]['from']:  # adjust seq for zero index slicing
                seq = seq[:key - offset] + mutation_dict[key]['to'] + seq[key - offset + 1:]
            else:  # find correct offset, or mark mutation source as doomed
                index_errors.append(key)
        except IndexError:
            print(key - offset)
    if index_errors:
        logger.warning('Index errors:\n%s' % str(index_errors))

    return seq


def parse_mutations(mutation_list):  # UNUSED  # TODO AMS
    if isinstance(mutation_list, str):
        mutation_list = mutation_list.split(', ')

    # Takes a list of mutations in the form A37K and parses the index (37), the FROM aa (A), and the TO aa (K)
    # output looks like {37: ('A', 'K'), 440: ('K', 'Y'), ...}
    mutation_dict = {}
    for mutation in mutation_list:
        to_letter = mutation[-1]
        from_letter = mutation[0]
        index = int(mutation[1:-1])
        mutation_dict[index] = (from_letter, to_letter)

    return mutation_dict


def generate_mutations_from_seq(seq1, seq2, offset=True, remove_blanks=True):  # TODO AMS
    """Create mutations with format A5K, one-indexed

    Index so residue value starts at 1. For PDB file comparison, seq1 should be crystal sequence (ATOM), seq2 should be
     expression sequence (SEQRES)
    Args:
        seq1 (str): Mutant sequence
        seq2 (str): Wild-type sequence
    Keyword Args:
        offset=True (bool): Whether to calculate alignment offset
        remove_blanks=True (bool): Whether to remove all sequence that has zero index or missing residues
    Returns:
        mutations (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    """
    if offset:
        alignment = generate_alignment(seq1, seq2)
        align_seq_1 = alignment[0][0]
        align_seq_2 = alignment[0][1]
    else:
        align_seq_1 = seq1
        align_seq_2 = seq2

    # Extract differences from the alignment
    starting_index_of_seq2 = align_seq_2.find(seq2[0])
    i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    mutations = {}
    for seq1_aa, seq2_aa in zip(align_seq_1, align_seq_2):
        if seq1_aa != seq2_aa:
            mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
            # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))
        i += 1

    if remove_blanks:
        # Remove any blank mutations and negative/zero indices
        remove_mutation_list = []
        for entry in mutations:
            if entry > 0:
                # if mutations[entry].find('-') == -1:
                for index in mutations[entry]:
                    if mutations[entry][index] == '-':
                    # if mutations[entry][index] == '0':
                        remove_mutation_list.append(entry)

        for entry in remove_mutation_list:
            mutations.pop(entry)

    return mutations


##############
# Alignments
##############


def generate_alignment(seq1, seq2, matrix='blosum62'):
    """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences*"""
    _matrix = getattr(matlist, matrix)
    gap_penalty = -10
    gap_ext_penalty = -1
    # Create sequence alignment
    alignment = pairwise2.align.localds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)

    return alignment


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
