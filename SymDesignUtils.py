import os
import logging
import math
import multiprocessing as mp
import time
from csv import reader, Dialect, QUOTE_MINIMAL
from operator import getitem
from string import digits
import pickle
import subprocess
from functools import reduce, wraps
from glob import glob
from itertools import repeat
from json import loads, dumps
from collections import defaultdict
from typing import List, Union, Iterable, Iterator, Tuple, Sequence

import numpy as np
# from numba import njit
# from Bio.PDB import PDBParser, Superimposer

# import CommandDistributer
import PathUtils as PUtils
# from Query.utils import validate_input

# Globals
input_string = '\nInput: '
index_offset = 1
rmsd_threshold = 1.0

# Todo get SDF files for all commented out
# Todo space and cryst

# from colorbrewer (https://colorbrewer2.org/)
color_arrays = [
    # pink to cyan
    ['#fff7fb', '#ece2f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636'],
    # light steel blue to magenta
    ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    # pale orange pink to electric lavender
    ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177', '#49006a'],
    # yellow to green
    ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529'],
    # pale yellow to salmon
    ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000']
]
large_color_array = []
for array in color_arrays:
    large_color_array.extend(array)


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
        (Union[Callable, Exception]): Function return upon proper execution, else the Exception if one was raised
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


#####################
# Runtime Utilities
#####################


def timestamp():
    return time.strftime('%y-%m-%d-%H%M%S')


starttime = timestamp()


def start_log(name='', handler=1, level=2, location=os.getcwd(), propagate=False, format_log=True, no_log_name=False,
              set_logger_level=False):
    """Create a logger to handle program messages

    Keyword Args:
        name='' (str): The name of the logger. By default the root logger is returned
        handler=1 (int): Whether to handle to stream (1-default) or a file (2)
        level=2 (int): What level of messages to emit (1-debug, 2-info (default), 3-warning, 4-error, 5-critical)
        location=os.getcwd() (str): If a FileHandler is used (handler=2) where should file be written?
            .log is appended to file
        propagate=True (bool): Whether to propagate messages to parent loggers (such as root or parent.current_logger)
        format_log=True (bool): Whether to format the log with logger specific formatting otherwise use message format
        no_log_name=False (bool): Whether to omit the logger name from the output
        set_logger_level=False (bool): Whether to set the level for the logger overall in addition to the logHandler
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
            # log_format = logging.Formatter('%(levelname)s: %(message)s')
            log_format = logging.Formatter('\033[38;5;208m%(levelname)s\033[0;0m: %(message)s')
        else:
            # log_format = logging.Formatter('[%(name)s]-%(levelname)s: %(message)s')  # \033[48;5;69m background
            log_format = logging.Formatter('\033[38;5;93m%(name)s\033[0;0m-\033[38;5;208m%(levelname)s\033[0;0m: %(message)s')
        lh.setFormatter(log_format)

    return _logger


logger = start_log(name=__name__)
null_log = start_log(name='null', handler=3)


def set_logging_to_debug():
    for logger_name in logging.root.manager.loggerDict:
        _logger = logging.getLogger(logger_name)
        _logger.setLevel(logging.DEBUG)
        _logger.propagate = False


def pretty_format_table(data, justification=None, header=None, header_justification=None) -> List[str]:
    """Present a table in readable format by sizing and justifying columns in a nested data structure
    i.e. [row1[column1, column2, ...], row2[], ...]

    Args:
        data (Iterable): Where each successive element is a row and each row's sub-elements are unique columns.
            The typical data structure would be [[i, j, k], [yes, 4, 0.1], [no, 5, 0.3]]
    Keyword Args:
        justification=None (Iterable): Iterable with elements 'l'/'left', 'r'/'right', or 'c'/'center' as text
            justification values
        header=None (Iterable): The names of values to place in the table header
        header_justification=None (Iterable): Iterable with elements 'l'/'left', 'r'/'right', or 'c'/'center' as text
            justification values
    Returns:
        (list[str]): The formatted data with each input row justified as an individual element in the list
    """
    justification_d = {'l': str.ljust, 'r': str.rjust, 'c': str.center,
                       'left': str.ljust, 'right': str.rjust, 'center': str.center}
    if isinstance(data, dict):  # incase data is pased as a dictionary, we should turn into an iterator of key, value
        data = data.items()

    widths = get_table_column_widths(data)
    row_length = len(widths)
    if not justification:
        justifications = list(str.ljust for _ in range(row_length))
    elif len(justification) == row_length:
        justifications = [justification_d.get(key.lower(), str.ljust) for key in justification]
    else:
        raise RuntimeError('The justification length (%d) doesn\'t match the number of columns (%d)'
                           % (len(justification), row_length))
    if header:
        if len(header) == row_length:
            data = [[column for column in row] for row in data]  # format as list so can insert
            data.insert(0, list(header))
            if not header_justification:
                header_justification = list(str.center for _ in range(row_length))
            elif len(header_justification) == row_length:
                header_justification = [justification_d.get(key.lower(), str.center) for key in header_justification]
            else:
                raise RuntimeError('The header_justification length (%d) doesn\'t match the number of columns (%d)'
                                   % (len(header_justification), row_length))
        else:
            raise RuntimeError('The header length (%d) doesn\'t match the number of columns (%d)'
                               % (len(header), row_length))

    return [' '.join(header_justification[idx](str(col), width) if not idx and header else justifications[idx](str(col), width)
                     for idx, (col, width) in enumerate(zip(row, widths))) for row in data]


def get_table_column_widths(data):
    """Find the widths of each column in a nested data structure

    Args:
        data (Iterable): Where each successive element is a row and each row's sub-elements are unique columns

    Returns:
        (tuple): A tuple containing the width of each column from the input data
    """
    return tuple(max(map(len, map(str, column))) for column in zip(*data))


def make_path(path, condition=True):
    """Make all required directories in specified path if it doesn't exist, and optional condition is True

    Keyword Args:
        condition=True (bool): A condition to check before the path production is executed
    """
    if condition:
        os.makedirs(path, exist_ok=True)


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
    """Clean a dictionary by passing specified keys. Default keeps all specified keys

    Args:
        dictionary (dict): {outer_dictionary: {key: value, key2: value2, ...}, ...}
        keys (Iterable): [key2, key10] Iterator of keys to be removed from dictionary
    Keyword Args:
        remove=True (bool): Whether or not to remove (True) specified keys
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


def index_intersection(index_groups):
    """Find the overlap of sets in a dictionary
    Args:
        index_groups (Iterable[Iterable]): Groups of indices

    Returns:
        (list): The union of all provided indices
    """
    final_indices = set()
    # find all set union
    for indices in index_groups:
        final_indices = final_indices.union(indices)
    # find all set intersection
    for indices in index_groups:
        final_indices = final_indices.intersection(indices)

    return list(final_indices)


def digit_keeper():
    table = defaultdict(type(None))
    table.update({ord(digit): digit for digit in digits})  # '0123456789'

    return table


digit_translate_table = digit_keeper()


def clean_comma_separated_string(string):
    return map(str.strip, string.strip().split(','))


def format_index_string(index_string):
    """From a string with indices of interest, comma separated or in a range, format into individual, integer indices

    Args:
        index_string (str): 23, 34,35,56-89, 290
    Returns:
        (list[int]): Indices in Pose formatting
    """
    final_index = []
    for index in clean_comma_separated_string(index_string):
        if '-' in index:  # we have a range, extract ranges
            for _idx in range(*tuple(map(int, index.split('-')))):
                final_index.append(_idx)
            final_index.append(_idx + 1)  # inclusive of the last integer in range
        else:  # single index
            final_index.append(int(index))

    return final_index
    # return list(map(int, final_index))  # why was this necessary?


###################
# PDB Handling
###################


# def residue_interaction_graph(pdb, distance=8, gly_ca=True):
#     """Create a atom tree using CB atoms from two PDB's
#
#     Args:
#         pdb (PDB): First PDB to query against
#     Keyword Args:
#         distance=8 (int): The distance to query in Angstroms
#         gly_ca=True (bool): Whether glycine CA should be included in the tree
#     Returns:
#         query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
#     """
#     # Get CB Atom Coordinates including CA coordinates for Gly residues
#     coords = np.array(pdb.extract_cb_coords(InclGlyCA=gly_ca))
#
#     # Construct CB Tree for PDB1
#     pdb1_tree = BallTree(coords)
#
#     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
#     query = pdb1_tree.query_radius(coords, distance)
#
#     return query


def split_interface_residues(interface_pairs):
    """Used to split Residue pairs and sort by Residue.number"""
    if interface_pairs:
        residues1, residues2 = zip(*interface_pairs)
        return sorted(set(residues1), key=lambda residue: residue.number), \
            sorted(set(residues2), key=lambda residue: residue.number)
    else:
        return [], []


# def split_interface_numbers(interface_pairs):
#     """Used to split residue number pairs"""
#     if interface_pairs:
#         numbers1, numbers2 = zip(*interface_pairs)
#         return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
#     else:
#         return [], []


def split_number_pairs_and_sort(pairs):
    """Used to split residue number pairs and sort"""
    if pairs:
        numbers1, numbers2 = zip(*pairs)
        return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
    else:
        return [], []


#################
# File Handling
#################


def write_file(data, file_name=None):
    if not file_name:
        file_name = os.path.join(os.getcwd(), input('What is your desired filename? (appended to current working '
                                                    'directory)%s' % input_string))
    with open(file_name, 'w') as f:
        f.write('%s\n' % '\n'.join(data))
    logger.info('The file \'%s\' was written' % file_name)


def validate_input(prompt, response=None):  # exact copy as in Query.utils
    _input = input(prompt)
    while _input not in response:
        _input = input('Invalid input... \'%s\' not a valid response. Try again%s' % (_input, input_string))

    return _input


def io_save(data, file_name=None):
    """Take an iterable and either output to user, write to a file, or both. User defined choice

    Args:
        data (Iterable): The data to write to file
    Keyword Args:
        file_name=None (str): The name of the file to write to
    Returns:
        (None)
    """
    io_prompt = 'Enter "P" to print Data, "W" to write Data to file, or "B" for both%s' % input_string
    response = ['w', 'p', 'b']
    _input = validate_input(io_prompt, response=response).lower()

    if _input in ['w', 'b']:
        write_file(file_name)

    if _input in ['b', 'p']:
        logger.info('\n%s' % data)

    return file_name


def to_iterable(_obj, ensure_file=False, skip_comma=False):
    """Take a file/object and return a list of individual objects splitting on newline or comma"""
    try:
        with open(_obj, 'r') as f:
            _list = f.readlines()
    except (FileNotFoundError, TypeError) as e:
        if isinstance(e, FileNotFoundError) and ensure_file:
            raise e
            # logger.info('No file was found for %s:\n%s' % (_obj, e))
        if isinstance(_obj, list):
            _list = _obj
        else:  # assumes obj is a string
            _list = [_obj]

    clean_list = []
    for it in _list:
        if skip_comma:
            it_list = [it]
        else:
            it_list = it.split(',')
        clean_list.extend(map(str.strip, it_list))

    # remove duplicates but keep the order
    clean_set = remove_duplicates(clean_list)
    try:
        clean_set.pop(clean_set.index(''))  # remove any missing values
    except ValueError:
        pass
    return clean_set


def remove_duplicates(_iter):
    seen = set()
    seen_add = seen.add
    return [x for x in _iter if not (x in seen or seen_add(x))]


def write_shell_script(command, name='script', out_path=os.getcwd(), additional=None, shell='bash', status_wrap=None) \
        -> str:
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
        f.write('#!/bin/%s\n\n%s%s %s\n\n' % (shell, check, command, modifier))
        if additional:
            f.write('%s\n\n' % ('\n\n'.join('%s %s' % (x, modifier) for x in additional)))
        f.write('%s\n' % _set)

    return file_name


def write_commands(command_list, name='all_commands', out_path=os.getcwd()):
    """Write a list of commands out to a file

    Args:
        command_list (Iterable[str]): An iterable with the commands as values
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


# @handle_errors(errors=(FileNotFoundError,))
# def parse_flags_file(directory, name=PUtils.interface_design, flag_variable=None):  # UNUSED
#     """Returns the design flags passed to Rosetta from a design directory
#
#     Args:
#         directory (str): Location of design directory on disk
#     Keyword Args:
#         name=PUtils.interface_design (str): The flags file suffix
#         flag_variable=None (str): The name of a specific variable to retrieve
#     Returns:
#         variable_dict (dict): {'interfaceA': 15A,21A,25A,..., 'dssm_file': , ...}
#     """
#     parser_vars = '-parser:script_vars'
#     with open(os.path.join(directory, 'flags_%s' % name), 'r') as f:
#         all_lines = f.readlines()
#         for line in all_lines:
#             if line[:19] == parser_vars:
#                 variables = line.lstrip(parser_vars).strip().split()
#                 variable_dict = {}
#                 for variable in variables:
#                     variable_dict[variable.split('=')[0]] = variable.split('=')[1]
#                 if flag_variable:
#                     return variable_dict[flag_variable]
#                 else:
#                     return variable_dict


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
                outfile.write('%s\n' % '\n'.join('%s,%s' % item for item in sequence.items()))
            else:
                outfile.write('%s\n' % '\n'.join('>%s\n%s' % item for item in sequence.items()))
        elif isinstance(sequence, str):
            outfile.write('>%s\n%s\n' % (name, sequence))
        else:
            raise DesignError('Cannot parse data to make fasta')

    return file_name


####################
# MULTIPROCESSING
####################


def calculate_mp_threads(cores=None, mpi=False, jobs=None, hyperthreading=True):
    """Calculate the number of multiprocessing threads to use for a specific application.

    Keyword Args:
        cores=None (int): How many cpu's to attempt to use, leaving at least one available for the machine
        mpi=False (bool): If commands use MPI
        jobs=None (int): How many jobs to attempt
        hyperthreading=True (bool): Whether the number of requested cores should be doubled due to hyperthread capacity
    Returns:
        (int): The number of threads to use taking the minimum of cores[if hyperthreading * 2], jobs, and max available
            cpus
    """
    allocated_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated_cpus:  # we are in a SLURM environment and should follow allocation but allow hyper-threading (* 2)
        # Todo reconsider the * 2 if not on cassini
        max_cpus_to_use = int(allocated_cpus) * (2 if hyperthreading else 1)
    else:
        max_cpus_to_use = mp.cpu_count() - 1  # leave CPU available for computer, see also len(os.sched_getaffinity(0))

    if jobs:  # test if cores or jobs is None, then take the minimal
        return min((cores * (2 if hyperthreading else 1) or max_cpus_to_use),
                   (jobs or max_cpus_to_use), max_cpus_to_use)
    if mpi:
        # Todo grab an evironmental variable for mpi threads?
        return int(max_cpus_to_use / 6)  # CommandDistributer.mpi)
    else:
        return min((cores or max_cpus_to_use), max_cpus_to_use)  # (jobs or max_cpus_to_use),


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


def get_all_file_paths(dir, extension=None):
    if extension:
        return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(dir, followlinks=True)
                for file in files if extension in file]
    else:
        return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(dir, followlinks=True)
                for file in files]


# def get_all_pdb_file_paths(pdb_dir):
#     return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(pdb_dir) for file in files
#             if '.pdb' in file]


def collect_nanohedra_designs(files: Sequence = None, directory: str = None, dock: bool = False) -> Tuple[List, str]:
    """Grab all poses from an input Nanohedra output

    Args:
        files: Iterable with disk location of files containing design directories
        directory: Disk location of the program directory
        dock: Whether the designs are in a docking run
    Returns:
        All pose directories found, the location where they are located
    """
    if files:
        all_paths = []
        for file in files:
            if not os.path.exists(file):
                logger.critical('No \'%s\' file found! Please ensure correct location/name!' % file)
                exit()
            if '.pdb' in file:  # single .pdb files were passed as input and should be loaded as such
                all_paths.append(file)
            else:  # assume a file that specifies individual designs was passed and load all design names in that file
                try:
                    with open(file, 'r') as f:
                        # only strip the trailing 'os.sep' in case file names are passed
                        paths = map(str.rstrip, [location.strip() for location in f.readlines()
                                                 if location.strip() != ''], repeat(os.sep))
                except IsADirectoryError:
                    raise DesignError('%s is a directory not a file. Did you mean to run with --file?' % file)
                all_paths.extend(paths)
    elif directory:
        if dock:
            all_paths = get_docked_directories(directory)
        else:
            base_directories = get_base_nanohedra_dirs(directory)
            all_paths = []
            for base in base_directories:  # Todo we shouldn't allow multiple, it complicates SymEntry matching
                all_paths.extend(get_docked_dirs_from_base(base))
    else:  # this shouldn't happen
        all_paths = []
    location = (files or directory)

    return sorted(set(all_paths)), location if isinstance(location, str) else location[0]


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


def collect_designs(files: Sequence = None, directory: str = None, projects: Sequence = None, singles: Sequence = None)\
        -> Tuple[List, str]:
    """Grab all poses from an input source

    Args:
        files: Iterable with disk location of files containing design directories
        directory: Disk location of the program directory
        projects: Disk location of a project directory
        singles: Disk location of a single design directory
    Returns:
        All pose directories found, the location where they are located
    """
    if files:
        all_paths = []
        for file in files:
            if not os.path.exists(file):
                logger.critical('No \'%s\' file found! Please ensure correct location/name!' % file)
                exit()
            if '.pdb' in file:  # single .pdb files were passed as input and should be loaded as such
                all_paths.append(file)
            else:  # assume a file that specifies individual designs was passed and load all design names in that file
                try:
                    with open(file, 'r') as f:
                        # only strip the trailing 'os.sep' in case file names are passed
                        paths = map(str.rstrip, [location.strip() for location in f.readlines()
                                                 if location.strip() != ''], repeat(os.sep))
                except IsADirectoryError:
                    raise DesignError('%s is a directory not a file. Did you mean to run with --file?' % file)
                all_paths.extend(paths)
    else:
        base_directory = get_base_symdesign_dir(directory)
        # return all design directories within:
        #  base directory -> /base/Projects/project1, ... /base/Projects/projectN
        #  specified projects -> /base/Projects/project1, /base/Projects/project2, ...
        #  specified singles -> /base/Projects/project/design1, /base/Projects/project/design2, ...
        if base_directory or projects or singles:
            all_paths = get_symdesign_dirs(base=base_directory, projects=projects, singles=singles)
        elif directory:  # This is probably an uninitialized project. Grab all .pdb files
            all_paths = get_all_file_paths(directory, extension='.pdb')
        else:  # function was called with all set to None. This shouldn't happen
            raise RuntimeError('Can\'t collect_designs when no arguments were passed!')

    location = (files or directory or projects or singles)

    return sorted(set(all_paths)), location if isinstance(location, str) else location[0]


def get_base_symdesign_dir(directory: str = None) -> Union[None, str]:
    base_dir = None
    if not directory:
        pass
    elif PUtils.program_output in directory:   # directory1/SymDesignOutput/directory2/directory3
        for idx, dirname in enumerate(directory.split(os.sep), 1):
            if dirname == PUtils.program_output:
                base_dir = '%s%s' % (os.sep, os.path.join(*directory.split(os.sep)[:idx]))
                break
    elif PUtils.program_output in os.listdir(directory):  # directory_provided/SymDesignOutput
        for sub_directory in os.listdir(directory):
            if sub_directory == PUtils.program_output:
                base_dir = os.path.join(directory, sub_directory)
                break

    return base_dir


def get_symdesign_dirs(base: str = None, projects: Iterable = None, singles: Iterable = None) -> Iterator:
    """Return the specific design directories from the specified hierarchy with the format
    /base(SymDesignOutput)/Projects/project/design
    """
    paths = []
    if base:
        # base/Projects/*/*/
        paths = glob('%s%s%s%s' % (base, '%s%s' % (os.sep, PUtils.projects), 2 * ('%s*' % os.sep), os.sep))
    elif projects:
        for project in projects:
            paths.extend(glob('%s%s*%s' % (project, os.sep, os.sep)))  # project/*/
    else:  # if single:
        for single, extension in map(os.path.splitext, singles):
            paths.extend(glob('%s%s' % (single, os.sep)))  # single/
    return map(os.path.dirname, paths)


class DesignSpecification(Dialect):
    delimiter = ','
    doublequote = True
    escapechar = None
    lineterminator = '\r\n'
    quotechar = '"'
    quoting = QUOTE_MINIMAL
    skipinitialspace = False
    strict = False

    def __init__(self, file):
        super().__init__()
        # self.delimiter = ','
        # self.doublequote = True
        # self.escapechar = None
        # self.lineterminator = '\r\n'
        # self.quotechar = '"'
        # self.quoting = QUOTE_MINIMAL
        # self.skipinitialspace = False
        # self.strict = False
        self.directive_delimiter = ':'
        self.file = file
        # self.reader()

        # def reader(self):
        with open(self.file) as file:
            all_poses, design_names, all_design_directives, *_ = zip(*reader(file, dialect=self))
            self.all_poses, self.design_names = list(map(str.strip, all_poses)), list(map(str.strip, design_names))

        self.directives = []
        # first split directives by white space, then by directive_delimiter
        for design_directives in all_design_directives:
            # print('Design Directives',design_directives)
            residue_directives = []
            # print('splitting residues', design_directives.split())
            # print('splitting directives', list(map(str.split, design_directives.split(), repeat(self.directive_delimiter))))
            for residues_s, directive in map(str.split, design_directives.split(), repeat(self.directive_delimiter)):
                # residues_s, directive = _directive.split(self.directive_delimiter)
                residues = format_index_string(residues_s)
                residue_directives.extend((residue, directive) for residue in residues)
            # print('Residue Directives', residue_directives)
            self.directives.append(dict(residue_directive for residue_directive in residue_directives))
        # print('Total Design Directives', self.directives)

    def return_directives(self):
        return zip(self.all_poses, self.design_names, self.directives)

    # def validate(self):


class DesignError(Exception):
    pass


######################
# Fragment Handling
######################


# @njit
def calculate_match(coords1=None, coords2=None, coords_rmsd_reference=None):
    """Calculate the overlap between two sets of coordinates given a reference rmsd

    Keyword Args:
        coords1=None (numpy.ndarray): The first set of coordinates
        coords2=None (numpy.ndarray): The second set of coordinates
        coords_rmsd_reference=None (numpy.ndarray): The reference RMSD to compared each pair of coordinates against
        max_z_value=2.0 (float): The z-score deviation threshold of the overlap to be considered a match
    Returns:
        (numpy.ndarray): The match score between coords1 and coords2
    """
    rmsds = rmsd(coords1, coords2)
    # Calculate Guide Atom Overlap Z-Value
    z_values = rmsds / coords_rmsd_reference
    # filter z_values by passing threshold
    return match_score_from_z_value(z_values)


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
    return np.sqrt((1 / match_score) - 1)


# @njit
def match_score_from_z_value(z_value):
    """Return the match score from a fragment z-value. Bounded between 0 and 1"""
    return 1 / (1 + (z_value ** 2))


# @njit
def z_score(sample: Union[float, np.ndarray], mean: Union[float, np.ndarray], stdev: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """From a sample(s), calculate the positional z-score

    Args:
        sample: An array with the sample at every position
        mean: An array with the mean at every position
        stdev: An array with the standard deviation at every position
    Returns:
        An array with the z-score of every position
    """
    try:
        return (sample - mean) / stdev
    except ZeroDivisionError:
        logger.error('The passed standard deviation was 0! Z-score calculation failed')


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
    if isinstance(iterable, dict):
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


def ex_path(*string):
    return os.path.join('path', 'to', *string)
