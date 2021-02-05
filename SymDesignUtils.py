# import pickle5 as pickle  # python 3.8 pickling protocol compatible
import logging
import math
import multiprocessing as mp
import operator
import os
import pickle
import subprocess
import sys
from functools import reduce
from glob import glob
from itertools import chain
from json import loads, dumps

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from sklearn.neighbors import BallTree

import CmdUtils as CUtils
import PathUtils as PUtils

# logging.getLogger().setLevel(logging.INFO)

# Globals
from utils.GeneralUtils import euclidean_squared_3d

index_offset = 1
rmsd_threshold = 1.0
layer_groups = {'P 1': 'p1', 'P 2': 'p2', 'P 21': 'p21', 'C 2': 'pg', 'P 2 2 2': 'p222', 'P 2 2 21': 'p2221',
                'P 2 21 21': 'p22121', 'C 2 2 2': 'c222', 'P 4': 'p4', 'P 4 2 2': 'p422',
                'P 4 21 2': 'p4121', 'P 3': 'p3', 'P 3 1 2': 'p312', 'P 3 2 1': 'p321', 'P 6': 'p6', 'P 6 2 2': 'p622'}
layer_group_d = {2, 4, 10, 12, 17, 19, 20, 21, 23,
                 27, 29, 30, 37, 38, 42, 43, 53, 59, 60, 64, 65, 68,
                 71, 78, 74, 78, 82, 83, 84, 89, 93, 97, 105, 111, 115}

# Todo get SDF files for all commented out
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T',  # O32': 'O', 'O42': 'O', 'O43': 'O',
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'T', 'T:{C2}{C3}': 'T',
                       'T:{C3}{C3}': 'T',  # 'O:{C2}{C3}': 'O', 'O:{C2}{C4}': 'O', 'O:{C3}{C4}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I', 'T:{C3}{C2}': 'T',
                       # 'O:{C3}{C2}': 'O', 'O:{C4}{C2}': 'O', 'O:{C4}{C3}': 'I'
                       'T': 'T', 'O': 'O', 'I': 'I',
                       # layer groups
                       # 'p6', 'p4', 'p3', 'p312', 'p4121', 'p622',
                       # space groups  # Todo
                       }
# Todo space and cryst
all_sym_entry_dict = {'T': {'C2': {'C3': 5}, 'C3': {'C2': 5, 'C3': 54}, 'T': -1},
                      'O': {'C2': {'C3': 7, 'C4': 13}, 'C3': {'C2': 7, 'C4': 56}, 'C4': {'C2': 13, 'C3': 56}, 'O': -2},
                      'I': {'C2': {'C3': 9, 'C5': 16}, 'C3': {'C2': 9, 'C5': 58}, 'C5': {'C2': 16, 'C3': 58}}}

point_group_sdf_map = {9: 'I32', 16: 'I52', 58: 'I53', 5: 'T32', 54: 'T33',  # 7: 'O32', 13: 'O42', 56: 'O43',
                       -1: 'T3', -2: 'O3'}


def parse_symmetry_to_nanohedra_entry(symmetry_string):
    symmetry_string = symmetry_string.strip()
    if len(symmetry_string) > 3:
        symmetry_split = symmetry_string.split('{')
        clean_split = [split.strip('}:') for split in symmetry_split]
    elif len(symmetry_string) == 3:  # Rosetta Formatting
        clean_split = ('%s C%s C%s' % (symmetry_string[0], symmetry_string[-1], symmetry_string[1])).split()
    elif symmetry_string in ['T', 'O']:  # , 'I']:
        clean_split = [symmetry_string, symmetry_string]  # , symmetry_string]
    else:  # C2, D6, C34
        raise ValueError('%s is not a supported symmetry yet!' % symmetry_string)

    logger.debug('Symmetry parsing split: %s' % clean_split)
    try:
        sym_entry = dictionary_lookup(all_sym_entry_dict, clean_split)
    except KeyError:
        # the prescribed symmetry was a plane or space group or point group that isn't recognized/ not in nanohedra
        sym_entry = symmetry_string
        raise ValueError('%s is not a supported symmetry!' % symmetry_string)

    logger.debug('Found Symmetry Entry %s for %s.' % (sym_entry, symmetry_string))
    return sym_entry


def dictionary_lookup(dictionary, items):
    """Return the values of a dictionary for the item pairs nested within"""
    return reduce(operator.getitem, items, dictionary)


def set_dictionary_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    dictionary_lookup(root, items[:-1])[items[-1]] = value


##########
# ERRORS
##########


def handle_errors_f(errors=(Exception, )):
    """Decorator to wrap a function with try: ... except errors: finally: Specifically for file reading functions

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


def handle_design_errors(errors=(Exception,)):
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
                args[0].log.error(e)  # This might forgo termination exceptions reporting if args[0] is des_dir
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
    if symmetry_entry_number not in point_group_sdf_map.keys():
        if symmetry_entry_number in layer_group_d:  # .keys():
            return 2
        else:
            return 3
    else:
        return 0


def sdf_lookup(symmetry_entry, dummy=False):
    if not symmetry_entry or dummy:
        return os.path.join(PUtils.symmetry_def_files, 'dummy.sym')
    else:
        symmetry_name = point_group_sdf_map[symmetry_entry]

    for root, dirs, files in os.walk(PUtils.symmetry_def_files):
        for file in files:
            if symmetry_name in file:
                return os.path.join(PUtils.symmetry_def_files, file)

    logger.warning('Error locating specified symmetry entry: %s' % str(symmetry_entry))
    return os.path.join(PUtils.symmetry_def_files, 'dummy.sym')

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
        (logging.Logger): Logger object to handle messages
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
        return {entry: {key: dictionary[entry][key] for key in dictionary[entry] if key in keys}
                for entry in dictionary}


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


def residue_interaction_graph(pdb, distance=8, gly_ca=True):  # Todo PDB.py
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
    except (FileNotFoundError, TypeError):
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


def write_shell_script(command, name='script', out_path=os.getcwd(), additional=None, shell='bash',
                       status_wrap=None):
    """Take a command and write to a name.sh script. By default bash is used as the shell interpreter

    Args:
        command (str): The command formatted using subprocess.list2cmdline(list())
    Keyword Args:
        name='script' (str): The name of the output shell script
        out_path=os.getcwd() (str): The location where the script will be written
        additional=None (iter): Additional commands also formatted using subprocess.list2cmdline
        shell='bash' (str): The shell which should interpret the script
        status_wrap=None (str): The name of a file in which to check and set the status of the command in the shell
    Returns:
        (str): The name of the file
    """
    if status_wrap:
        modifier = '&&'
        check = subprocess.list2cmdline(['python', os.path.join(PUtils.source, 'CommandDistributer.py'), '--stage',
                                         name, '--info', status_wrap, '--check', modifier, '\n'])
        _set = subprocess.list2cmdline(['python', os.path.join(PUtils.source, 'CommandDistributer.py'), '--stage', name,
                                       '--info', status_wrap, '--set'])
    else:
        check, _set, modifier = '', '', ''
    if name.endswith('.sh'):
        name = name[:-3]

    file_name = os.path.join(out_path, '%s.sh' % name)
    with open(file_name, 'w') as f:
        f.write('#!/bin/%s\n\n%s%s %s\n' % (shell, check, command, modifier))
        if additional:
            f.write('%s\n' % ('\n'.join('%s %s' % (x, modifier) for x in additional)))
        f.write('%s\n' % _set)

    return file_name


def write_commands(command_list, name='all_commands', out_path=os.getcwd()):  # TODO loc, location, outpath. Standardize!!!
    if len(command_list) > 1:
        extension = '.cmds'
    else:
        extension = '.cmd'
    file = os.path.join(out_path, name + extension)
    with open(file, 'w') as f:
        f.write('%s\n' % '\n'.join(command for command in command_list))

    return file


def write_list_to_file(_list, name=None, location=os.getcwd()):
    file_name = os.path.join(location, name)  # + '.cmd')
    with open(file_name, 'w') as f:
        f.write('\n'.join(item for item in _list))

    return file_name


def pdb_list_file(refined_pdb, total_pdbs=1, suffix='', out_path=os.getcwd(), additional=None):
    file_name = os.path.join(out_path, 'design_files.txt')
    with open(file_name, 'w') as f:
        f.write('%s\n' % refined_pdb)  # run second round of metrics on input as well
        f.write('\n'.join('%s%s_%s.pdb' % (os.path.splitext(refined_pdb)[0], suffix, str(idx).zfill(4))
                          for idx in enumerate(total_pdbs, 1)))
        if additional:
            f.write('\n'.join(pdb for pdb in additional))

    return file_name


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
        function (Callable): Which function should be executed
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
#     # dock_dir.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))
#     dock_dir.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses%s' % str(suffix or '')))  # design_recap
#     dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.program_root]
#     # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
#     dock_dir.building_blocks, dock_dir.building_block_logs = [], []
#     for k, _sym in enumerate(dock_dir.program_root):
#         dock_dir.building_blocks.append(list())
#         dock_dir.building_block_logs.append(list())
#         for bb_dir in next(os.walk(_sym))[1]:
#             if os.path.exists(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils
#                 dock_dir.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
#                 dock_dir.building_blocks[k].append(bb_dir)
#
#     # dock_dir.building_blocks = [next(os.walk(dir))[1] for dir in dock_dir.program_root]
#     # dock_dir.building_block_logs = [[os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir)  # make a log path TODO PUtils
#     #                                  for bb_dir in dock_dir.building_blocks[k]]  # for each building_block combo in _sym index of dock_dir.building_blocks
#     #                                 for k, _sym in enumerate(dock_dir.program_root)]  # for each sym in symmetry
#
#     return dock_dir


# def get_pose_by_id(design_directories, ids):  # DEPRECIATED
#     return [des_dir for des_dir in design_directories if str(des_dir) in ids]


def get_all_base_root_paths(directory):
    return [os.path.abspath(root) for root, dirs, files in os.walk(directory) if not dirs]


def get_all_file_paths(pdb_dir, extension=None):
    if extension:
        return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(pdb_dir) for file in files
                if extension in file]
    else:
        return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(pdb_dir) for file in files]


def get_all_pdb_file_paths(pdb_dir):  # Todo DEPRECIATE
    return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(pdb_dir) for file in files
            if '.pdb' in file]


# def get_directory_pdb_file_paths(pdb_dir):  # DEPRECIATED
#     return glob(os.path.join(pdb_dir, '*.pdb*'))


def get_base_nanohedra_dirs(base_dir):
    """Find all master directories corresponding to the highest output level of Nanohedra.py outputs. This corresponds
    to the DesignDirectory symmetry attribute
    """
    nanohedra_dirs = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if PUtils.master_log in files:
            nanohedra_dirs.append(root)
            del dirs[:]
            # print('found %d directories' % len(nanohedra_dirs))

    return nanohedra_dirs


def get_docked_directories(base_directory, directory_type='NanohedraEntry'):  # '*DockedPoses'
    """Useful for when your docked directory is basically known but the """
    return [os.path.join(root, _dir) for root, dirs, files in os.walk(base_directory) for _dir in dirs
            if directory_type in _dir]
    # all_directories.append(os.path.join(root, _dir))
    # return sorted(set(all_directories))
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


def collect_directories(directory, file=None, dir_type=None):
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
                exit()
        with open(_file, 'r') as f:
            all_paths = [location.strip() for location in f.readlines()]
        location = file
    else:
        if dir_type == 'dock':
            all_paths = get_docked_directories(directory)
        elif dir_type == PUtils.nano or PUtils.nano in directory:  # Todo this logic can be wrong
            base_directories = get_base_nanohedra_dirs(directory)
            all_paths = list(chain.from_iterable([get_docked_dirs_from_base(base) for base in base_directories]))
        else:
            base_directories = get_base_symdesign_dirs(directory)
            if base_directories:
                # return all design directories within the base directory ->/base/projects/project/designs
                all_paths = list(chain.from_iterable([get_symdesign_dirs_from_base(base) for base in base_directories]))
            else:  # This is probably an uninitialized project and we should grab all .pdb files
                all_paths = get_all_file_paths(directory, extension='.pdb')
        location = directory

    return sorted(set(all_paths)), location


def get_base_symdesign_dirs(directory):
    if PUtils.program_name in directory:   # directory1/SymDesignOutput/directory2/directory3
        return ['/%s' % os.path.join(*directory.split(os.sep)[:idx])
                for idx, dirname in enumerate(directory.split(os.sep), 1)
                if dirname == PUtils.program_output]
    elif PUtils.program_name in os.listdir(directory):  # directory_provided/SymDesignOutput
        return [os.path.join(directory, sub_directory) for sub_directory in os.listdir(directory)
                if sub_directory == PUtils.program_output]
    else:
        return False
    # for root, dirs, files in os.walk(directory):


def get_symdesign_dirs_from_base(base):
    return sorted(set(map(os.path.dirname, glob('%s/*/*/*/' % base))))  # /base/projects/project/designs

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
class DesignError(Exception):
    pass
    # def __init__(self, message):
    #     super().__init__(message)
    #     self.args = message
    #
    # def __eq__(self, other):
    #     return self.__str__() == other


def calculate_overlap(coords1=None, coords2=None, coords_rmsd_reference=None):
    e1 = euclidean_squared_3d(coords1[0], coords2[0])
    e2 = euclidean_squared_3d(coords1[1], coords2[1])
    e3 = euclidean_squared_3d(coords1[2], coords2[2])
    s = e1 + e2 + e3
    mean = s / float(3)
    rmsd = math.sqrt(mean)

    # Calculate Guide Atom Overlap Z-Value
    # and Calculate Score Term for Nanohedra Residue Level Summation Score
    z_val = rmsd / float(max(coords_rmsd_reference, 0.01))

    match_score = 1 / float(1 + (z_val ** 2))

    return match_score, z_val


def z_value_from_match_score(match_score):
    return math.sqrt((1 / match_score) - 1)


def match_score_from_z_value(z_value):
    """Return the match score from a fragment z-value. Bounded between 0 and 1"""
    return 1 / float(1 + (z_value ** 2))


def filter_euler_lookup_by_zvalue(index_pairs, ghost_frags, coords_l1, surface_frags, coords_l2, z_value_func=None,
                                  max_z_value=2):
    """Filter an EulerLookup by a specified z-value, where the z-value is calculated by a passed function which has
    two sets of coordinates and and rmsd as args

    Returns:
        (list[tuple]): (Function overlap parameter, z-value of function)
    """
    # Todo, signature = coords_list1, coords_list2, rmsd_list
    # Todo make coords_list1, coords_list2, rmsd_list all np array. Calling this function on array will return array
    overlap_results = []
    for index_pair in index_pairs:
        ghost_frag = ghost_frags[index_pair[0]]
        coords1 = coords_l1[index_pair[0]]
        # or guide_coords aren't numpy, so np.matmul gets them there, if not matmul, (like 3, 1)
        # coords1 = np.matmul(qhost_frag.get_guide_coords(), rot1_mat_np_t)
        surf_frag = surface_frags[index_pair[1]]
        coords2 = coords_l2[index_pair[1]]
        # surf_frag.get_guide_coords()
        if surf_frag.get_i_type() == ghost_frag.get_j_frag_type():  # Todo remove to mask outside
            result, z_value = z_value_func(coords1=coords1, coords2=coords2,
                                           coords_rmsd_reference=ghost_frag.get_rmsd())
            if z_value <= max_z_value:
                overlap_results.append((result, z_value))
            else:
                overlap_results.append(False)
        else:
            overlap_results.append(False)

    return overlap_results
