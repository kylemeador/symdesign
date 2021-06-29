import os
import logging
import math
import multiprocessing as mp
from operator import getitem
import string
import pickle
import subprocess
from functools import reduce, wraps
from glob import glob
from itertools import chain, repeat
from json import loads, dumps
from collections import defaultdict

import numpy as np
# from numba import njit
from sklearn.neighbors import BallTree
from Bio import SeqIO
from Bio.PDB import PDBParser, Superimposer

# import CommandDistributer
import PathUtils as PUtils
from classes.SymEntry import SymEntry


# Globals
index_offset = 1
rmsd_threshold = 1.0
layer_group_d = {'P 1': 'p1', 'P 2': 'p2', 'P 21': 'p21', 'C 2': 'pg', 'P 2 2 2': 'p222', 'P 2 2 21': 'p2221',
                 'P 2 21 21': 'p22121', 'C 2 2 2': 'c222', 'P 4': 'p4', 'P 4 2 2': 'p422',
                 'P 4 21 2': 'p4121', 'P 3': 'p3', 'P 3 1 2': 'p312', 'P 3 2 1': 'p321', 'P 6': 'p6', 'P 6 2 2': 'p622'}
layer_groups = {2, 4, 10, 12, 17, 19, 20, 21, 23,
                27, 29, 30, 37, 38, 42, 43, 53, 59, 60, 64, 65, 68,
                71, 78, 74, 78, 82, 83, 84, 89, 93, 97, 105, 111, 115}
space_groups = {'P23', 'P4222', 'P321', 'P6322', 'P312', 'P622', 'F23', 'F222', 'P6222', 'I422', 'I213', 'R32', 'P4212',
                'I432', 'P4132', 'I4132', 'P3', 'P6', 'I4122', 'P4', 'C222', 'P222', 'P432', 'F4132', 'P422', 'P213',
                'F432', 'P4232'}
space_group_to_sym_entry = {}

# Todo get SDF files for all commented out
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T',  # O32': 'O', 'O42': 'O', 'O43': 'O',
                       'I23': 'I', 'I25': 'I', 'I35': 'I', 'T23': 'T',  # O23': 'O', 'O24': 'O', 'O34': 'O',
                       'T:{C2}{C3}': 'T', 'T:{C3}{C2}': 'T', 'T:{C3}{C3}': 'T',
                       # 'O:{C2}{C3}': 'O', 'O:{C2}{C4}': 'O', 'O:{C3}{C4}': 'I',
                       # 'O:{C3}{C2}': 'O', 'O:{C4}{C2}': 'O', 'O:{C4}{C3}': 'I'
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I',
                       'T': 'T', 'O': 'O', 'I': 'I',
                       # layer groups
                       # 'p6', 'p4', 'p3', 'p312', 'p4121', 'p622',
                       # space groups  # Todo
                       # 'cryst': 'cryst'
                       }
# Todo space and cryst
all_sym_entry_dict = {'T': {'C2': {'C3': 5}, 'C3': {'C2': 5, 'C3': 54}, 'T': -1},
                      'O': {'C2': {'C3': 7, 'C4': 13}, 'C3': {'C2': 7, 'C4': 56}, 'C4': {'C2': 13, 'C3': 56}, 'O': -2},
                      'I': {'C2': {'C3': 9, 'C5': 16}, 'C3': {'C2': 9, 'C5': 58}, 'C5': {'C2': 16, 'C3': 58}}}

point_group_sdf_map = {9: 'I32', 16: 'I52', 58: 'I53', 5: 'T32', 54: 'T33',  # 7: 'O32', 13: 'O42', 56: 'O43',
                       }


def parse_symmetry_to_sym_entry(symmetry_string):
    symmetry_string = symmetry_string.strip()
    if len(symmetry_string) > 3:
        symmetry_split = symmetry_string.split('{')
        clean_split = [split.strip('}:') for split in symmetry_split]
    elif len(symmetry_string) == 3:  # Rosetta Formatting
        clean_split = ('%s C%s C%s' % (symmetry_string[0], symmetry_string[-1], symmetry_string[1])).split()
    elif symmetry_string in ['T', 'O']:  # , 'I']:
        logger.warning('This functionality is not working properly yet!')
        clean_split = [symmetry_string, symmetry_string]  # , symmetry_string]
    else:  # C2, D6, C34
        raise ValueError('%s is not a supported symmetry yet!' % symmetry_string)

    # logger.debug('Symmetry parsing split: %s' % clean_split)
    try:
        sym_entry = dictionary_lookup(all_sym_entry_dict, clean_split)
    except KeyError:
        # the prescribed symmetry was a plane or space group or point group that isn't recognized/ not in nanohedra
        sym_entry = symmetry_string
        raise ValueError('%s is not a supported symmetry!' % symmetry_string)

    # logger.debug('Found Symmetry Entry %s for %s.' % (sym_entry, symmetry_string))
    return SymEntry(sym_entry)


def dictionary_lookup(dictionary, items):
    """Return the values of a dictionary for the item pairs nested within

    Args:
        dictionary (dict): The dictionary to search
        items (tuple): The tuple of keys to search for
    Returns:
        (any): The value specified by dictionary keys
    """
    return reduce(getitem, items, dictionary)


def set_dictionary_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    dictionary_lookup(root, items[:-1])[items[-1]] = value


##########
# ERRORS
##########


def handle_design_errors(errors=(Exception,)):
    """Decorator to wrap a function with try: ... except errors: and log errors to a DesignDirectory

    Keyword Args:
        errors=(Exception, ) (tuple): A tuple of exceptions to monitor, even if single exception
    Returns:
        (function): Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as error:
                design_directory = args[0]
                design_directory.log.error(error)  # Allows exception reporting using DesignDirectory
                design_directory.info['error'] = error  # include the error code in the design state

                return error
        return wrapped
    return wrapper


def handle_errors(errors=(Exception,)):
    """Decorator to wrap a function with try: ... except errors:

    Keyword Args:
        errors=(Exception, ) (tuple): A tuple of exceptions to monitor, even if single exception
    Returns:
        (function): Function return upon proper execution, else is error if exception raised, else None
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as error:
                return error
        return wrapped
    return wrapper


############
# Symmetry
############


def handle_symmetry(symmetry_entry_number):
    # group = cryst1_record.split()[-1]/
    if symmetry_entry_number not in point_group_sdf_map.keys():
        if symmetry_entry_number in layer_groups:  # .keys():
            return 2
        else:
            return 3
    else:
        return 0


def sdf_lookup(symmetry=None):
    """From the set of possible point groups, locate the proper symmetry definition file depending on the specified
    symmetry. If none specified (default) a viable, but completely garbage symmetry definition file will be returned

    Keyword Args:
        symmetry=None (union[str, int]): Can be one of the valid_point_groups or a point group SymmetryEntry number
    Returns:
        (str): The location of the symmetry definition file on disk
    """
    if not symmetry:
        return os.path.join(PUtils.symmetry_def_files, 'dummy.sym')
    elif isinstance(symmetry, int):
        symmetry_name = point_group_sdf_map[symmetry]
    else:
        symmetry_name = symmetry

    for file in os.listdir(PUtils.symmetry_def_files):
        if symmetry_name in file:
            return os.path.join(PUtils.symmetry_def_files, file)

    raise DesignError('Error locating specified symmetry entry: %s' % symmetry_name)


#####################
# Runtime Utilities
#####################


def start_log(name='', handler=1, level=2, location=os.getcwd(), propagate=True, format_log=True, no_log_name=False,
              set_logger_level=False):
    """Create a logger to handle program messages

    Keyword Args:
        name='' (str): The name of the logger. By default the root logger is returned
        handler=1 (int): Whether to handle to stream (1-default) or a file (2)
        level=2 (int): What level of messages to emit (1-debug, 2-info (default), 3-warning, 4-error, 5-critical)
        location=os.getcwd() (str): If a FileHandler is used (handler=2) where should file be written?
            .log is appended to file
        propagate=True (bool): Whether to propagate messages to parent loggers (such as root or parent.current_logger)
    Returns:
        (logging.Logger): Logger object to handle messages
    """
    # Todo make a mechanism to only emit warning or higher if propagate=True
    # log_handler = {1: logging.StreamHandler(), 2: logging.FileHandler(location + '.log'), 3: logging.NullHandler}
    log_level = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}

    _logger = logging.getLogger(name)
    if set_logger_level:
        _logger.setLevel(log_level[level])
    if not propagate:
        _logger.propagate = False
    # lh = log_handler[handler]
    if handler == 1:
        lh = logging.StreamHandler()
    elif handler == 2:
        if os.path.splitext(location)[1] == '':  # no extension, should add one
            lh = logging.FileHandler('%s.log' % location)
        else:  # already has extension
            lh = logging.FileHandler(location)
    else:  # handler == 3:
        lh = logging.NullHandler()
        # return _logger
    lh.setLevel(log_level[level])
    _logger.addHandler(lh)

    if format_log:
        if no_log_name:
            log_format = logging.Formatter('%(levelname)s: %(message)s')
        else:
            log_format = logging.Formatter('[%(name)s]-%(levelname)s: %(message)s')
        lh.setFormatter(log_format)

    return _logger


logger = start_log(name=__name__)
null_log = start_log(name='null', handler=3, propagate=False)


def pretty_format_table(rows, justification=None):
    """Present a table in readable format.

    Args:
        rows (iter): The rows of data you would like to populate the table
    Keyword Args:
        justification=None (list): A list with either 'l'/'left', 'r'/'right', or 'c'/'center' as the text
        justification values
    """
    justification_d = {'l': str.ljust, 'r': str.rjust, 'c': str.center,
                       'left': str.ljust, 'right': str.rjust, 'center': str.center}
    widths = get_table_column_widths(rows)
    if not justification:
        justifications = list(str.ljust for _ in widths)
    else:
        # try:
        justifications = [justification_d.get(key.lower(), str.ljust) for key in justification]
        # except KeyError:
        #     raise KeyError('%s: The justification \'%s\' is not of the allowed types (%s).'
        #                    % (pretty_format_table.__name__, key, list(justification_d.keys())))

    return [' '.join(justifications[idx](str(col), width) for idx, (col, width) in enumerate(zip(row, widths)))
            for row in rows]


def get_table_column_widths(rows):
    return tuple(max(map(len, map(str, col))) for col in zip(*rows))


# @handle_errors(errors=(FileNotFoundError,))
def unpickle(file_name):  # , protocol=pickle.HIGHEST_PROTOCOL):
    """Unpickle (deserialize) and return a python object located at filename"""
    if '.pkl' not in file_name:
        file_name = '%s.pkl' % file_name
    try:
        with open(file_name, 'rb') as serial_f:
            new_object = pickle.load(serial_f)
    except EOFError as ex:
        raise DesignError('The object serialized at location %s couldn\'t be accessed. No data present!' % file_name)

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


def filter_dictionary_keys(dictionary, keys, remove=False):
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
            dictionary.pop(key, None)

        return dictionary
    else:
        return {key: dictionary[key] for key in keys if key in dictionary}


def remove_interior_keys(dictionary, keys, keep=False):
    """Clean specified keys from a dictionaries internal dictionary. Default removes the specified keys

    Args:
        dictionary (dict): {outer_dictionary: {key: value, key2: value2, ...}, ...}
        keys (iter): [key2, key10] Iterator of keys to be removed from dictionary
    Keyword Args:
        remove=True (bool): Whether or not to remove (True) or keep (False) specified keys
    Returns:
        (dict): {outer_dictionary: {key: value, ...}, ...} - Cleaned dictionary
    """
    if not keep:
        for entry in dictionary:
            for key in keys:
                dictionary[entry].pop(key, None)

        return dictionary
    else:
        return {entry: {key: dictionary[entry][key] for key in dictionary[entry] if key in keys}
                for entry in dictionary}


def index_intersection(indices):
    """Find the overlap of sets in a dictionary
    """
    final_indices = set()
    # find all set union
    for values in indices.values():
        final_indices = final_indices.union(values)
    # find all set intersection
    for values in indices.values():
        final_indices = final_indices.intersection(values)

    return list(final_indices)


def digit_keeper():
    table = defaultdict(type(None))
    table.update({ord(c): c for c in string.digits})

    return table


digit_translate_table = digit_keeper()

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
    coords = np.array(pdb.extract_cb_coords(InclGlyCA=gly_ca))

    # Construct CB Tree for PDB1
    pdb1_tree = BallTree(coords)

    # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    query = pdb1_tree.query_radius(coords, distance)

    return query


def split_interface_pairs(interface_pairs):
    """Used to split Residue pairs and sort by Residue.number"""
    if interface_pairs:
        residues1, residues2 = zip(*interface_pairs)
        return sorted(set(residues1), key=lambda residue: residue.number), \
            sorted(set(residues2), key=lambda residue: residue.number)
    else:
        return [], []


# def split_interface_pairs(interface_pairs):
#     """Used to split residue number pairs and sort"""
#     if interface_pairs:
#         residues1, residues2 = zip(*interface_pairs)
#         return sorted(set(residues1), key=int), sorted(set(residues2), key=int)
#     else:
#         return [], []


#################
# File Handling
#################


def io_save(data, filename=None):
    """Take an iterable and either output to user, write to a file, or both. User defined choice

    Returns
        None
    """
    # file = os.path.join(os.getcwd(), 'missing_UNP_PDBS.txt')
    def write_file(filename):
        if not filename:
            filename = input('What is your desired filename? (appended to current working directory)\n')
            filename = os.path.join(os.getcwd(), filename)
        with open(filename, 'w') as f:
            f.write('\n'.join(data))
        print('File \'%s\' was written' % filename)

    while True:
        _input = input('Enter P to print Data, W to write Data to file, or B for both:').upper()
        if _input == 'W':
            write_file(filename)
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
            write_file(filename)
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


def write_shell_script(command, name='script', out_path=os.getcwd(), additional=None, shell='bash', status_wrap=None):
    """Take a command and write to a name.sh script. By default bash is used as the shell interpreter

    Args:
        command (str): The command formatted using subprocess.list2cmdline(list())
    Keyword Args:
        name='script' (str): The name of the output shell script
        out_path=os.getcwd() (str): The location where the script will be written
        additional=None (list): Additional commands also formatted using subprocess.list2cmdline()
        shell='bash' (str): The shell which should interpret the script
        status_wrap=None (str): The name of a file in which to check and set the status of the command in the shell
    Returns:
        (str): The name of the file
    """
    if status_wrap:
        modifier = '&&'
        check = subprocess.list2cmdline(['python', os.path.join(PUtils.source, 'CommandDistributer.py'), '--stage',
                                         name, 'status', '--info', status_wrap, '--check', modifier, '\n'])
        _set = subprocess.list2cmdline(['python', os.path.join(PUtils.source, 'CommandDistributer.py'), '--stage', name,
                                       'status', '--info', status_wrap, '--set'])
    else:
        check, _set, modifier = '', '', ''

    file_name = os.path.join(out_path, name if name.endswith('.sh') else '%s.sh' % name)
    with open(file_name, 'w') as f:
        f.write('#!/bin/%s\n\n%s%s %s\n' % (shell, check, command, modifier))
        if additional:
            f.write('%s\n' % ('\n\n'.join('%s %s' % (x, modifier) for x in additional)))
        f.write('%s\n' % _set)

    return file_name


def write_commands(command_list, name='all_commands', out_path=os.getcwd()):
    """Write a list of commands out to a file

    Args:
        command_list (iterable): An iterable with the commands as values
    Keyword Args:
        name='all_commands' (str): The name of the file. Will be appended with '.cmd(s)'
        out_path=os.getcwd(): The directory where the file will be written
    Returns:
        (str): The filename of the new file
    """
    file = os.path.join(out_path, '%s.cmds' % name if len(command_list) > 1 else '%s.cmd' % name)
    with open(file, 'w') as f:
        f.write('%s\n' % '\n'.join(command for command in command_list))

    return file


def write_list_to_file(_list, name=None, location=os.getcwd()):
    file_name = os.path.join(location, name)  # + '.cmd')
    with open(file_name, 'w') as f:
        f.write('\n'.join(str(item) for item in _list))

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


@handle_errors(errors=(FileNotFoundError,))
def parse_flags_file(directory, name=PUtils.interface_design, flag_variable=None):
    """Returns the design flags passed to Rosetta from a design directory

    Args:
        directory (str): Location of design directory on disk
    Keyword Args:
        name=PUtils.interface_design (str): The flags file suffix
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


def read_fasta_file(file_name, **kwargs):
    """Returns an iterator of SeqRecords. Ex. [record1, record2, ...]"""
    return SeqIO.parse(file_name, 'fasta')


def write_fasta(sequence_records, file_name=None):  # Todo, consolidate (self.)write_fasta_file() with here
    """Writes an iterator of SeqRecords to a file with .fasta appended. The file name is returned"""
    if not file_name:
        return None
    if '.fasta' in file_name:
        file_name = file_name.rstrip('.fasta')
    SeqIO.write(sequence_records, '%s.fasta' % file_name, 'fasta')

    return '%s.fasta' % file_name


def concatenate_fasta_files(file_names, output='concatenated_fasta'):
    """Take multiple fasta files and concatenate into a single file"""
    seq_records = [read_fasta_file(file) for file in file_names]
    return write_fasta(list(chain.from_iterable(seq_records)), file_name=output)


def write_fasta_file(sequence, name, out_path=os.getcwd(), csv=False):
    """Write a fasta file from sequence(s)

    Args:
        sequence (iterable): One of either list, dict, or string. If list, can be list of tuples(name, sequence),
            list of lists, etc. Smart solver using object type
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
        csv=False (bool): Whether the file should be written as a .csv
    Returns:
        (str): The name of the output file
    """
    extension = '%s.fasta' if not csv else '%s.csv'
    file_name = os.path.join(out_path, extension % name)
    with open(file_name, 'w') as outfile:
        if type(sequence) is list:
            if type(sequence[0]) is list:  # where inside list is of alphabet (AA or DNA)
                for idx, seq in enumerate(sequence):
                    outfile.write('>%s_%d\n' % (name, idx))  # header
                    if len(seq[0]) == 3:  # Check if alphabet is 3 letter protein
                        outfile.write(' '.join(aa for aa in seq))
                    else:
                        outfile.write(''.join(aa for aa in seq))
            elif isinstance(sequence[0], str):
                outfile.write('>%s\n%s\n' % name, ' '.join(aa for aa in sequence))
            elif type(sequence[0]) is tuple:  # where seq[0] is header, seq[1] is seq
                outfile.write('\n'.join('>%s\n%s' % seq for seq in sequence))
            else:
                raise DesignError('Cannot parse data to make fasta')
        elif isinstance(sequence, dict):
            if csv:
                outfile.write('\n'.join('%s,%s' % item for item in sequence.items()))
            else:
                outfile.write('\n'.join('>%s\n%s' % item for item in sequence.items()))
        elif isinstance(sequence, str):
            outfile.write('>%s\n%s\n' % (name, sequence))
        else:
            raise DesignError('Cannot parse data to make fasta')

    return file_name


####################
# MULTIPROCESSING
####################


def calculate_mp_threads(cores=None, mpi=False):
    """Calculate the number of multiprocessing threads to use for a specific application

    Keyword Args:
        mpi=False (bool): If commands use MPI
        maximum=False (bool): Whether to use the maximum number of cpu's, leaving one available for the machine
        no_model=False (bool): If pose initialization is completed without any modelling
    Returns:
        (int): The number of threads to use
    """
    allocated_cpus = os.environ.get('SLURM_CPUS_PER_TASK', None)
    if allocated_cpus:  # we are in a SLURM environment and should follow allocation but allow hyper-threading (* 2)
        max_cpus_to_use = int(allocated_cpus) * 2
    else:
        max_cpus_to_use = mp.cpu_count() - 1  # leave CPU available for computer, see also len(os.sched_getaffinity(0))

    if cores:
        return cores
    elif mpi:
        return int(max_cpus_to_use / 6)  # CommandDistributer.mpi)
    else:
        return max_cpus_to_use


def set_worker_affinity():
    """When a new worker process is created, use this initialization function to set the affinity for all CPUs.
    Especially important for multiprocessing in the context of numpy, scipy, pandas
    FROM Stack Overflow:
    https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy

    See: http://manpages.ubuntu.com/manpages/precise/en/man1/taskset.1.html
        -p is a mask for the logical cpu processors to use, the pid allows the affinity for an existing process to be
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
    # with mp.get_context('spawn').Pool(processes=threads, initializer=set_worker_affinity) as p:  # maxtasksperchild=1
    with mp.get_context('spawn').Pool(processes=threads) as p:  # maxtasksperchild=1
        results = p.map(function, arg)
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
        (list): The results produced from the function and process_args
    """
    # with mp.get_context(context).Pool(processes=threads, initializer=set_worker_affinity, maxtasksperchild=100) as p:
    with mp.get_context(context).Pool(processes=threads, maxtasksperchild=100) as p:
        results = p.starmap(function, process_args)  # , chunksize=1
    p.join()

    return results


# # to make mp compatible with 2.7
# from contextlib import contextmanager
#
#
# @contextmanager
# def poolcontext(*args, **kwargs):
#     pool = mp.Pool(*args, **kwargs)
#     yield pool
#     pool.terminate()
#
#
# def mp_starmap_python2(function, process_args, threads=1):
#     with poolcontext(processes=threads) as p:
#         results = p.map(function, process_args)
#     p.join()
#
#     return results


######################
# Directory Handling
######################


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


def collect_nanohedra_designs(files=None, directory=None, dock=False):
    """Grab all poses from an input Nanohedra output

    Keyword Args:
        files=None (iterable): Iterable with disk location of files containing design directories
        directory=None (str): Disk location of the program directory
        project=False (bool): Whether or not the designs are in a docking run
    Returns:
        (tuple[(list), (str)]): All pose directories found, The location where they are located
    """
    if files:
        all_paths = []
        for file in files:
            _file = file
            if not os.path.exists(_file):
                _file = os.path.join(os.getcwd(), file)
                if not os.path.exists(_file):
                    logger.critical('No \'%s\' file found! Please ensure correct location/name!' % file)
                    exit()
            with open(_file, 'r') as f:
                paths = map(str.rstrip, [location.strip() for location in f.readlines() if location.strip() != ''],
                            repeat(os.sep))  # only strip the trailing '/' separator in case file names are passed
            all_paths.extend(paths)
            location = _file
    elif directory:
        location = directory
        if dock:
            all_paths = get_docked_directories(directory)
        else:
            base_directories = get_base_nanohedra_dirs(directory)
            all_paths = []
            for base in base_directories:  # Todo we shouldn't allow multiple, it complicates SymEntry matching
                all_paths.extend(get_docked_dirs_from_base(base))
    else:  # this shouldn't happen
        all_paths = []
        location = None

    return sorted(set(all_paths)), location


def get_base_nanohedra_dirs(base_dir):
    """Find all master directories corresponding to the highest output level of Nanohedra.py outputs. This corresponds
    to the DesignDirectory symmetry attribute
    """
    nanohedra_dirs = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if PUtils.master_log in files:
            nanohedra_dirs.append(root)
            del dirs[:]

    return nanohedra_dirs


def get_docked_directories(base_directory, directory_type='NanohedraEntry'):  # '*DockedPoses'
    """Useful for when your docked directory is basically known but the """
    return [os.path.join(root, _dir) for root, dirs, files in os.walk(base_directory) for _dir in dirs
            if directory_type in _dir]


def get_docked_dirs_from_base(base):
    return sorted(set(map(os.path.dirname, glob('%s/*/*/*/*/' % base))))


def collect_designs(files=None, directory=None, project=None, single=None):
    """Grab all poses from an input source

    Keyword Args:
        files=None (iterable): Iterable with disk location of files containing design directories
        directory=None (str): Disk location of the program directory
        project=None (str): Disk location of a project directory
        single=None (str): Disk location of a single design directory
    Returns:
        (tuple[(list), (str)]): All pose directories found, The location where they are located
    """
    if files:
        all_paths = []
        for file in files:
            _file = file
            if not os.path.exists(_file):
                _file = os.path.join(os.getcwd(), file)
                if not os.path.exists(_file):
                    logger.critical('No \'%s\' file found! Please ensure correct location/name!' % file)
                    exit()
            with open(_file, 'r') as f:
                paths = map(str.rstrip, [location.strip() for location in f.readlines() if location.strip() != ''],
                            repeat(os.sep))  # only strip the trailing '/' separator in case file names are passed
            all_paths.extend(paths)
            location = _file
    elif directory:
        location = directory
        base_directories = get_base_symdesign_dirs(directory)
        if not base_directories:
            # This is probably an uninitialized project and we should grab all .pdb files then initialize
            all_paths = get_all_file_paths(directory, extension='.pdb')
        else:
            # return all design directories within the base directory ->/base/Projects/project/design
            all_paths = []
            for base in base_directories:
                all_paths.extend(get_symdesign_dirs(base=base))
            all_paths = map(os.path.dirname, all_paths)

    elif project:
        all_paths = get_symdesign_dirs(project=project)
        location = project
    elif single:
        all_paths = get_symdesign_dirs(single=single)
        location = single
    else:  # this shouldn't happen
        all_paths = []
        location = None

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
        return []


def get_symdesign_dirs(base=None, project=None, single=None):
    """Return the specific design directories from the specified hierarchy with the format
    /base(SymDesignOutput)/Projects/project/design
    """
    if single:
        return map(os.path.dirname, glob('%s/' % single))  # sorted(set())
    elif project:
        return map(os.path.dirname, glob('%s/*/' % project))  # sorted(set())
    else:
        return map(os.path.dirname, glob('%s/*/*/*/' % base))  # sorted(set())


class DesignError(Exception):
    pass


######################
# Fragment Handling
######################


# @njit
def calculate_overlap(coords1=None, coords2=None, coords_rmsd_reference=None, max_z_value=2.0):
    """Calculate the overlap between two sets of coordinates given a reference rmsd

    Keyword Args:
        coords1=None (numpy.ndarray): The first set of coordinates
        coords2=None (numpy.ndarray): The second set of coordinates
        coords_rmsd_reference=None (numpy.ndarray): The reference RMSD to compared each pair of coordinates against
        max_z_value=2.0 (float): The z-score deviation threshold of the overlap to be considered a match
    Returns:
        (numpy.ndarray): The overlap z-value where the RMSD between coords1 and coords2 is < max_z_value
    """
    rmsds = rmsd(coords1, coords2)
    # Calculate Guide Atom Overlap Z-Value
    z_values = rmsds / coords_rmsd_reference
    # filter z_values by passing threshold
    return np.where(z_values < max_z_value, z_values, False)


# @njit mean doesn't take arguments
def rmsd(coords1=None, coords2=None):
    """Calculate the RMSD over sets of coordinates in two numpy.arrays. The first axis (0) contains instances of
    coordinate sets, the second axis (1) contains a set of coordinates, and the third axis (2) contains the x, y, z
    values for a coordinate

    Returns:
        (np.ndarray)
    """
    difference_squared = (coords1 - coords2) ** 2
    # axis 2 gets the sum of the rows 0[1[2[],2[],2[]], 1[2[],2[],2[]]]
    sum_difference_squared = difference_squared.sum(axis=2)
    # axis 1 gets the mean of the rows 0[1[]], 1[]]
    mean_sum_difference_squared = sum_difference_squared.mean(axis=1)

    return np.sqrt(mean_sum_difference_squared)


def z_value_from_match_score(match_score):
    return math.sqrt((1 / match_score) - 1)


# @njit
def match_score_from_z_value(z_value):
    """Return the match score from a fragment z-value. Bounded between 0 and 1"""
    return 1 / (1 + (z_value ** 2))


######################
# Matrix Handling
######################


def all_vs_all(iterable, func, symmetrize=True):
    """Calculate an all versus all comparison using a defined function. Matrix is symmetrized by default

    Args:
        iterable (Iterable): Dict or array like object
        func (function): Function to calculate different iterations of the iterable
    Keyword Args:
        symmetrize=True (Bool): Whether or not to make the resulting matrix symmetric
    Returns:
        (numpy.ndarray): Matrix with resulting calculations
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
