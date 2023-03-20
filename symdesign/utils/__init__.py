from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import string
import subprocess
import time
from collections import defaultdict
from functools import reduce, wraps
from glob import glob
from itertools import repeat
from logging import Logger, DEBUG, INFO, WARNING, ERROR, CRITICAL, getLogger, StreamHandler, FileHandler, NullHandler, \
    Formatter, root as root_logger
from operator import getitem
from typing import Any, Callable, Iterable, AnyStr, Sequence, Iterator, Literal, Type, get_args

import numpy as np
import psutil
import torch

from . import path as putils

# Globals
logger = logging.getLogger(__name__)
# null_logger = logging.getLogger('null')
np_torch_int_types = (np.int8, np.int16, np.int32, np.int64,
                      torch.int, torch.int8, torch.int16, torch.int32, torch.int64)
np_torch_float_types = (np.float16, np.float32, np.float64,
                        torch.float, torch.float16, torch.float32, torch.float64)
np_torch_int_float_types = np_torch_int_types + np_torch_float_types
input_string = '\nInput: '
zero_offset = 1
rmsd_threshold = 1.
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


def dictionary_lookup(dictionary: dict, items: tuple[Any, ...]) -> Any:
    """Return the values of a dictionary for the item pairs nested within

    Args:
        dictionary: The dictionary to search
        items: The tuple of keys to search for
    Returns:
        The value specified by dictionary keys
    """
    return reduce(getitem, items, dictionary)


def set_dictionary_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    dictionary_lookup(root, items[:-1])[items[-1]] = value


##########
# ERRORS
##########

def handle_errors(errors: tuple[Type[Exception], ...] = (Exception,)) -> Any:
    """Decorator to wrap a function with try: ... except errors:

    Args:
        errors: A tuple of exceptions to monitor, even if single exception
    Returns:
        Function return upon proper execution, else the Exception if one was raised
    """
    def wrapper(func: Callable) -> Any:
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as error:
                return error
        return wrapped
    return wrapper


#####################
# Runtime Utilities
#####################


def timestamp() -> str:
    """Return the date/time formatted as YR-MO-DA-HRMNSC. Ex: 2022-Jan-01-245959"""
    return time.strftime('%y-%m-%d-%H%M%S')


def datestamp(short: bool = False) -> str:
    """Return the date/time formatted as Year-Mon-DA.

    Args:
        short: Whether to return the short date
    Returns:
        Ex: 2022-Jan-01 or 01-Jan-22 if short
    """
    if short:
        return time.strftime('%d-%b-%y')  # Desired PDB format
    else:
        return time.strftime('%Y-%b-%d')  # Preferred format


short_start_date = datestamp(short=True)
long_start_date = datestamp()
starttime = timestamp()
log_handler = {1: StreamHandler, 2: FileHandler, 3: NullHandler}
logging_level_literal = Literal[
    1, 2, 3, 4, 5, 10, 20, 30, 40, 50,
    '1', '2', '3', '4', '5', '10', '20', '30', '40', '50',
    'debug', 'info', 'warning', 'error', 'critical', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
]
log_level_keys: tuple[str | int, ...] = get_args(logging_level_literal)
logging_levels = [DEBUG, INFO, WARNING, ERROR, CRITICAL]
log_level = dict(zip(log_level_keys, [DEBUG, INFO, WARNING, ERROR, CRITICAL, DEBUG, INFO, WARNING, ERROR, CRITICAL,
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, DEBUG, INFO, WARNING, ERROR, CRITICAL,
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, DEBUG, INFO, WARNING, ERROR, CRITICAL
                                      ]))
"""log_level = {
'debug': DEBUG, 'info': INFO, 'warning': WARNING, 'error': ERROR, 'critical': CRITICAL,
'DEBUG': DEBUG, 'INFO': INFO, 'WARNING': WARNING, 'ERROR': ERROR, 'CRITICAL': CRITICAL,
1: DEBUG, 2: INFO, 3: WARNING, 4: ERROR, 5: CRITICAL,
10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR, 50: CRITICAL}
"""


def start_log(name: str = '', handler: int = 1, level: logging_level_literal = 2, location: AnyStr = os.getcwd(),
              propagate: bool = False, format_log: bool = True, no_log_name: bool = False,
              handler_level: logging_level_literal = None) -> Logger:
    """Create a logger to handle program messages

    Args:
        name: The name of the logger. By default, the root logger is returned
        handler: Whether to handle to stream (1), a file (2), or a NullHandler (3+)
        level: What level of messages to emit (1-debug, 2-info, 3-warning, 4-error, 5-critical)
        location: If a FileHandler is used (handler=2) where should file be written? .log is appended to the filename
        propagate: Whether to propagate messages to parent loggers (such as root or parent.current_logger)
        format_log: Whether to format the log with logger specific formatting otherwise use message format
        no_log_name: Whether to omit the logger name from the output
        handler_level: Whether to set the level for the logger handler on top of the overall level
    Returns:
        Logger object to handle messages
    """
    _logger = getLogger(name)
    _logger.setLevel(log_level[level])
    # Todo make a mechanism to only emit warning or higher if propagate=True
    #  See below this function for adding handler[0].addFilter()
    _logger.propagate = propagate
    if format_log:
        if no_log_name:
            # log_format = Formatter('%(levelname)s: %(message)s')
            # log_format = Formatter('\033[38;5;208m%(levelname)s\033[0;0m: %(message)s')
            message_fmt = '\033[38;5;208m{levelname}\033[0;0m: {message}'
        else:
            # log_format = Formatter('[%(name)s]-%(levelname)s: %(message)s')  # \033[48;5;69m background
            # log_format = Formatter('\033[38;5;93m%(name)s\033[0;0m-\033[38;5;208m%(levelname)s\033[0;0m: %(message)s')
            message_fmt = '\033[38;5;93m{name}\033[0;0m-\033[38;5;208m{levelname}\033[0;0m: {message}'
    else:
        message_fmt = '{message}'

    _handler = log_handler[handler]
    if handler == 2:
        # Check for extension. If one doesn't exist, add ".log"
        lh = _handler(f'{location}.log' if os.path.splitext(location)[1] == '' else location, delay=True)
        # Set delay=True to prevent the log from opening until the first emit() is called
        # Remove any coloring from the log
        message_fmt = message_fmt.replace('\033[38;5;208m', '').replace('\033[38;5;93m', '').replace('\033[0;0m', '')
    else:
        lh = _handler()

    if handler_level is not None:
        lh.setLevel(log_level[handler_level])

    log_format = Formatter(fmt=message_fmt, style='{')
    lh.setFormatter(log_format)
    _logger.addHandler(lh)

    return _logger


# logger = start_log(name=__name__)
# def emit_info_and_lower(record) -> int:
#     if record.levelno < 21:  # logging.INFO and logging.DEBUG
#         return 1
#     else:
#         return 0
# # Reject any message that is warning or greater to let root handle
# logger.handlers[0].addFilter(emit_info_and_lower)


def set_logging_to_level(level: logging_level_literal = None, handler_level: logging_level_literal = None):
    """For each Logger in current run time, set the Logger or the Logger.handlers level to level

    level is debug by default if no arguments are specified

    Args:
        level: The level to set all loggers to
        handler_level: The level to set all logger handlers to
    """
    if level is not None:
        _level = log_level[level]
        set_level_func = Logger.setLevel
    elif handler_level is not None:  # Todo possibly rework this to accept both arguments
        _level = log_level[handler_level]

        def set_level_func(logger_: Logger, level_: int):
            for handler in logger_.handlers:
                handler.setLevel(level_)
    else:  # if level is None and handler_level is None:
        _level = log_level[1]
        set_level_func = Logger.setLevel

    # print(root_logger.manager.loggerDict)
    for logger_name in root_logger.manager.loggerDict:
        _logger = getLogger(logger_name)
        set_level_func(_logger, _level)
        # _logger.setLevel(_level)


def set_loggers_to_propagate():
    """For each Logger in current run time, set the Logger to propagate"""
    for logger_name in root_logger.manager.loggerDict:
        _logger = getLogger(logger_name)
        _logger.propagate = True


def pretty_format_table(data: Iterable, justification: Sequence = None, header: Sequence = None,
                        header_justification: Sequence = None) -> list[str]:
    """Present a table in readable format by sizing and justifying columns in a nested data structure
    i.e. [row1[column1, column2, ...], row2[], ...]

    Args:
        data: Where each successive element is a row and each row's sub-elements are unique columns.
            The typical data structure would be [[i, j, k], [yes, 4, 0.1], [no, 5, 0.3]]
        justification: Iterable with elements 'l'/'left', 'r'/'right', or 'c'/'center' as justification values
        header: The names of values to place in the table header
        header_justification: Iterable with elements 'l'/'left', 'r'/'right', or 'c'/'center' as justification values
    Returns:
        The formatted data with each input row justified as an individual element in the list
    """
    justification_d = {'l': str.ljust, 'r': str.rjust, 'c': str.center,
                       'left': str.ljust, 'right': str.rjust, 'center': str.center}
    # Incase data is passed as a dictionary, we should turn into an iterator of key, value
    if isinstance(data, dict):
        data = data.items()

    column_widths = get_table_column_widths(data)
    number_columns = len(column_widths)
    if not justification:
        justifications = list(str.ljust for _ in range(number_columns))
    elif len(justification) == number_columns:
        justifications = [justification_d.get(key.lower(), str.ljust) for key in justification]
    else:
        raise RuntimeError(f"The justification length ({len(justification)}) doesn't match the "
                           f"number of columns ({number_columns})")
    if header is not None:
        if len(header) == number_columns:
            # Format data as list so we can insert header
            data = [[column for column in row] for row in data]
            data.insert(0, list(header))
            if header_justification is None:
                header_justification = list(str.center for _ in range(number_columns))
            elif len(header_justification) == number_columns:
                header_justification = [justification_d.get(key.lower(), str.center) for key in header_justification]
            else:
                raise RuntimeError(f"The header_justification length ({len(header_justification)}) doesn't match the "
                                   f"number of columns ({number_columns})")
        else:
            raise RuntimeError(f"The header length ({len(header)}) doesn't match the "
                               f"number of columns ({number_columns})")

    return [' '.join(header_justification[idx](column, column_widths[idx]) if row_idx == 0 and header is not None
                     else justifications[idx](column, column_widths[idx])
                     for idx, column in enumerate(map(str, row_entry)))
            for row_idx, row_entry in enumerate(data)]


def get_table_column_widths(data: Iterable) -> tuple[int]:
    """Find the widths of each column in a nested data structure

    Args:
        data: Where each successive element is a row and each row's sub-elements are unique columns
    Returns:
        A tuple containing the width of each column from the input data
    """
    return tuple(max(map(len, map(str, column))) for column in zip(*data))


def read_json(file_name, **kwargs) -> dict | None:
    """Use json.load to read an object from a file

    Args:
        file_name: The location of the file to write
    Returns:
        The json data in the file
    """
    with open(file_name, 'r') as f_save:
        data = json.load(f_save)

    return data


def write_json(data: Any, file_name: AnyStr, **kwargs) -> AnyStr:
    """Use json.dump to write an object to a file

    Args:
        data: The object to write
        file_name: The location of the file to write
    Returns:
        The name of the written file
    """
    with open(file_name, 'w') as f_save:
        json.dump(data, f_save, **kwargs)

    return file_name


# @handle_errors(errors=(FileNotFoundError,))
def unpickle(file_name: AnyStr) -> Any:  # , protocol=pickle.HIGHEST_PROTOCOL):
    """Unpickle (deserialize) and return a python object located at filename"""
    if '.pkl' not in file_name and '.pickle' not in file_name:
        file_name = '%s.pkl' % file_name
    try:
        with open(file_name, 'rb') as serial_f:
            new_object = pickle.load(serial_f)
    except EOFError as ex:
        raise InputError(f"The serialized file '{file_name}' contains no data present.")

    return new_object


def pickle_object(target_object: Any, name: str = None, out_path: AnyStr = os.getcwd(),
                  protocol: int = pickle.HIGHEST_PROTOCOL) -> AnyStr:
    """Pickle (serialize) an object into a file named "out_path/name.pkl". Automatically adds extension

    Args:
        target_object: Any python object
        name: The name of the pickled file
        out_path: Where the file should be written
        protocol: The pickling protocol to use
    Returns:
        The pickled filename
    """
    if name is None:
        file_name = out_path
    else:
        file_name = os.path.join(out_path, name)

    if not file_name.endswith('.pkl'):
        file_name = f'{file_name}.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump(target_object, f, protocol)

    return file_name


# def filter_dictionary_keys(dictionary: dict, keys: Iterable, keep: bool = True) -> dict[Any, dict[Any, Any]]:
#     """Clean a dictionary by passing specified keys. Default keeps all specified keys
#
#     Args:
#         dictionary: {outer_dictionary: {key: value, key2: value2, ...}, ...}
#         keys: [key2, key10] Iterator of keys to be removed from dictionary
#         keep: Whether to keep (True) or remove (False) specified keys
#     Returns:
#         {outer_dictionary: {key: value, ...}, ...} - Cleaned dictionary
#     """
#     if keep:
#         return {key: dictionary[key] for key in keys if key in dictionary}
#     else:
#         for key in keys:
#             dictionary.pop(key, None)
#
#         return dictionary

def remove_interior_keys(dictionary: dict, keys: Iterable, keep: bool = False) -> dict[Any, dict[Any, Any]]:
    """Clean specified keys from a dictionaries internal dictionary. Default removes the specified keys

    Args:
        dictionary: {outer_dictionary: {key: value, key2: value2, ...}, ...}
        keys: [key2, key10] Iterator of keys to be removed from dictionary
        keep: Whether to keep (True) or remove (False) specified keys
    Returns:
        {outer_dictionary: {key: value, ...}, ...} - Cleaned dictionary
    """
    if keep:
        return {entry: {key: dictionary[entry][key] for key in dictionary[entry] if key in keys}
                for entry in dictionary}
    else:
        for entry in dictionary:
            for key in keys:
                dictionary[entry].pop(key, None)

        return dictionary


def digit_keeper() -> defaultdict:
    return defaultdict(type(None), dict(zip(map(ord, string.digits), string.digits)))  # '0123456789'


def digit_remover() -> defaultdict:
    non_numeric_chars = string.printable[10:]
    # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    keep_chars = dict(zip(map(ord, non_numeric_chars), non_numeric_chars))

    return defaultdict(type(None), keep_chars)


keep_digit_table = digit_keeper()
remove_digit_table = digit_remover()
# This doesn't work
# >>> to_number = dict(zip(map(ord, numeric_chars), range(10)))
# >>> string.printable[4:20].translate(to_number)
# '\x04\x05\x06\x07\x08\tabcdefghij'


def clean_comma_separated_string(string: str) -> list[str]:
    """Return a list from a comma separated string"""
    return list(map(str.strip, string.strip().split(',')))


def format_index_string(index_string: str) -> list[int]:
    """From a string with indices of interest, comma separated or in a range, format into individual, integer indices

    Args:
        index_string: 23, 34,35,56-89, 290
    Returns:
        Indices in Pose formatting
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


#################
# File Handling
#################

def write_file(data: Iterable, file_name: AnyStr = None) -> AnyStr:
    """Take an iterable and either output to user, write to a file, or both. User defined choice

    Args:
        data: The data to write to file
        file_name: The name of the file to write to
    Returns:
        The name of the output file
    """
    if not file_name:
        file_name = os.path.join(os.getcwd(), input('What is your desired filename? (appended to current working '
                                                    f'directory){input_string}'))
    with open(file_name, 'w') as f:
        f.write('%s\n' % '\n'.join(map(str, data)))

    return file_name


def validate_input(prompt: str, response: Iterable[str]) -> str:  # Exact copy as in Query.utils
    """Following a provided prompt, validate that the user input is a valid response then return the response outcome

    Args:
        prompt: The desired prompt
        response: The response values to accept
    Returns:
        The data matching the chosen response key
    """
    _input = input(f'{prompt}\nChoose from [{", ".join(response)}]{input_string}')
    while _input not in response:
        _input = input(f'Invalid input... "{_input}" not a valid response. Try again{input_string}')

    return _input


def io_save(data: Iterable, file_name: AnyStr = None) -> AnyStr:
    """Take an iterable and either output to user, write to a file, or both. User defined choice

    Args:
        data: The data to write to file
        file_name: The name of the file to write to
    Returns:
        The name of the output file
    """
    io_prompt = f'Enter "P" to print Data, "W" to write Data to file, or "B" for both{input_string}'
    response = ['W', 'P', 'B', 'w', 'p', 'b']
    _input = validate_input(io_prompt, response=response).lower()

    if _input in 'bp':
        logger.info('%s\n' % '\n'.join(map(str, data)))

    if _input in 'wb':
        write_file(data, file_name)

    return file_name


def to_iterable(obj: AnyStr | list, ensure_file: bool = False, skip_comma: bool = False) -> list[str]:
    """Take an object and return a list of individual objects splitting on newline or comma

    Args:
        obj: The object to convert to an Iterable
        ensure_file: Whether to ensure the passed obj is a file
        skip_comma: Whether to skip commas when converting the records to an iterable
    Returns:
        The Iterable formed from the input obj
    """
    try:
        with open(obj, 'r') as f:
            iterable = f.readlines()
    except (FileNotFoundError, TypeError) as error:
        if isinstance(error, FileNotFoundError) and ensure_file:
            raise error
        if isinstance(obj, list):
            iterable = obj
        else:  # Assume that obj is a string
            iterable = [obj]

    clean_list = []
    for item in iterable:
        if skip_comma:
            it_list = [item]
        else:
            it_list = item.split(',')
        clean_list.extend(map(str.strip, it_list))

    # # Remove duplicates but keep the order
    # clean_list = remove_duplicates(clean_list)
    try:
        clean_list.pop(clean_list.index(''))  # Remove any missing values
    except ValueError:
        pass
    return clean_list


def remove_duplicates(iter_: Iterable[Any]) -> list[Any]:
    """An efficient, order maintaining, set function to remove duplicates"""
    seen = set()
    seen_add = seen.add
    return [x for x in iter_ if not (x in seen or seen_add(x))]


# def change_filename(original, new=None, increment=None):
#     """Take a json formatted score.sc file and rename decoy ID's
#
#     Args:
#         original (str): Location on disk of file
#     Keyword Args:
#         new=None (str): The filename to replace the file with
#         increment=None (int): The number to increment each decoy by
#     """
#     dirname = os.path.dirname(original)
#     # basename = os.path.basename(original)  # .split('_')[:-1]
#     basename = os.path.splitext(os.path.basename(original))[0]  # .split('_')[:-1]
#     ext = os.path.splitext(os.path.basename(original))[-1]  # .split('_')[:-1]
#     old_id = basename.split('_')[-1]
#     # old_id = int(os.path.basename(original).split('_')[-1])
#     if increment:
#         new_id = int(old_id) + increment
#     else:
#         new_id = int(old_id)
#
#     if new:
#         new_file = os.path.join(dirname, '%s_%04d%s' % (new, new_id, ext))
#     else:
#         new_file = os.path.join(dirname, '%s%04d%s' % (basename[:-len(old_id)], new_id, ext))
#
#     p = subprocess.Popen(['mv', original, new_file])


# def modify_decoys(file, increment=0):
#     """Take a json formatted score.sc file and rename decoy ID's
#
#     Args:
#         file (str): Location on disk of scorefile
#     Keyword Args:
#         increment=None (int): The number to increment each decoy by
#     Returns:
#         score_dict (dict): {design_name: {all_score_metric_keys: all_score_metric_values}, ...}
#     """
#     with open(file, 'r+') as f:
#         scores = [loads(score) for score in f.readlines()]
#         for i, score in enumerate(scores):
#             design_id = score['decoy'].split('_')[-1]
#             if design_id[-1].isdigit():
#                 decoy_name = score['decoy'][:-len(design_id)]
#                 score['decoy'] = '%s%04d' % (decoy_name, int(design_id) + increment)
#             scores[i] = score
#
#         f.seek(0)
#         f.write('\n'.join(dumps(score) for score in scores))
#         f.truncate()


# def rename_decoy_protocols(des_dir, rename_dict):
#     score_file = os.path.join(des_dir.scores, putils.scores_file)
#     with open(score_file, 'r+') as f:
#         scores = [loads(score) for score in f.readlines()]
#         for i, score in enumerate(scores):
#             for protocol in rename_dict:
#                 if protocol in score:
#                     score[protocol] = rename_dict[protocol]
#             scores[i] = score
#
#         f.seek(0)
#         f.write('\n'.join(dumps(score) for score in scores))
#         f.truncate()


####################
# MULTIPROCESSING
####################


def calculate_mp_cores(cores: int = None, mpi: bool = False, jobs: int = None) -> int:
    """Calculate the number of multiprocessing cores to use for a specific application

    Default options specify to leave at least one CPU available for the machine. If a SLURM environment is used,
    the number of cores will reflect the environmental variable SLURM_CPUS_PER_TASK
    Args:
        cores: How many cpu's to use
        mpi: If commands use MPI
        jobs: How many jobs to use
    Returns:
        The number of cores to use taking the minimum of cores, jobs, and max cpus available
    """
    allocated_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated_cpus:  # we are in a SLURM environment and should follow allocation
        max_cpus_to_use = int(allocated_cpus)
    else:  # logical=False only uses physical cpus, not logical threads
        max_cpus_to_use = psutil.cpu_count(logical=False) - 1  # leave CPU available for computer

    if cores or jobs:  # test if cores or jobs is None, then take the minimum
        infinity = float('inf')
        return min((cores or infinity), (jobs or infinity))

    if mpi:  # Todo grab an environmental variable for mpi cores?
        return int(max_cpus_to_use / 6)  # distribute.mpi)
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
    _cmd = ['taskset', '-p', f'0x{"f" * int((psutil.cpu_count() / 4))}', str(os.getpid())]
    logger.debug(subprocess.list2cmdline(_cmd))
    p = subprocess.Popen(_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    p.communicate()


def mp_map(function: Callable, arg: Iterable, processes: int = 1, context: str = 'spawn') -> list[Any]:
    """Maps an interable input with a single argument to a function using multiprocessing Pool

    Args:
        function: Which function should be executed
        arg: Arguments to be unpacked in the defined function, order specific
        processes: How many workers/cores should be spawned to handle function(arguments)?
        context: How to start new processes? One of 'spawn', 'fork', or 'forkserver'.
    Returns:
        The results produced from the function and arg
    """
    # with mp.get_context(context).Pool(processes=processes) as p:  # , maxtasksperchild=100
    with mp.get_context(context).Pool(processes=processes, initializer=set_worker_affinity) as p:
        results = p.map(function, arg)

    return results


def mp_starmap(function: Callable, star_args: Iterable[tuple], processes: int = 1, context: str = 'spawn') -> list[Any]:
    """Maps an iterable input with multiple arguments to a function using multiprocessing Pool

    Args:
        function: Which function should be executed
        star_args: Arguments to be unpacked in the defined function, order specific
        processes: How many workers/cores should be spawned to handle function(arguments)?
        context: How to start new processes? One of 'spawn', 'fork', or 'forkserver'.
    Returns:
        The results produced from the function and star_args
    """
    # with mp.get_context(context).Pool(processes=processes) as p:  # , maxtasksperchild=100
    with mp.get_context(context).Pool(processes=processes, initializer=set_worker_affinity) as p:
        results = p.starmap(function, star_args)

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
# def mp_starmap_python2(function, process_args, cores=1):
#     with poolcontext(processes=cores) as p:
#         results = p.map(function, process_args)
#     p.join()
#
#     return results


def bytes2human(number: int, return_format: str = "{:.1f} {}") -> str:
    """Convert bytes to a human-readable format

    See: http://goo.gl/zeJZl
    >>> bytes2human(10000)
    '9.8 K'
    >>> bytes2human(100001221)
    '95.4 M'

    Args:
        number: The number of bytes
        return_format: The desired return format with '{}'.format() compatibility
    Returns:
        The human-readable expression of bytes from a number of bytes
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {symbol: 1 << idx * 10 for idx, symbol in enumerate(symbols)}

    for symbol, symbol_number in reversed(prefix.items()):
        if number >= symbol_number:
            value = number / symbol_number
            break
    else:  # Smaller than the smallest
        symbol = symbols[0]
        value = number
    return return_format.format(value, symbol)


SYMBOLS = {
    'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'iotta'),
    'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi'),
}


def human2bytes(human_byte_str: AnyStr) -> int:
    """Convert human-readable bytes to a numeric format

    See: http://goo.gl/zeJZl
    >>> human2bytes('0 B')
    0
    >>> human2bytes('1 K')
    1024
    >>> human2bytes('1 M')
    1048576
    >>> human2bytes('1 Gi')
    1073741824
    >>> human2bytes('1 tera')
    1099511627776
    >>> human2bytes('0.5kilo')
    512
    >>> human2bytes('0.1  byte')
    0
    >>> human2bytes('1 k')  # k is an alias for K
    1024
    >>> human2bytes('12 foo')

    Raises:
        ValueError if input can't be parsed
    Returns:
        The number of bytes from a human-readable expression of bytes
    """
    # Find the scale prefix/abbreviation
    letter = human_byte_str.translate(remove_digit_table).replace('.', '').replace(' ', '')
    for name, symbol_set in SYMBOLS.items():
        if letter in symbol_set:
            break
    else:
        # if letter == 'k':
        #     # treat 'k' as an alias for 'K' as per: http://goo.gl/kTQMs
        #     sset = SYMBOLS['customary']
        #     letter = letter.upper()
        # else:
        raise ValueError(f"{human2bytes.__name__}: Can't interpret {human_byte_str}")

    # Find the size value
    number = human_byte_str.strip(letter).strip()
    try:
        number = float(number)
    except ValueError:
        raise ValueError(f"{human2bytes.__name__}: Can't interpret {human_byte_str}")
    else:
        # Convert to numeric bytes
        letter_index = symbol_set.index(letter)
        return int(number * (1 << letter_index * 10))


def get_available_memory(human_readable: bool = False, gpu: bool = False) -> int:
    """

    Args:
        human_readable: Whether the return value should be human-readable
        gpu: Whether a GPU should be used
    Returns:
        The available memory (in bytes) depending on the compute environment
    """
    # Check if job is allocated by SLURM
    if 'SLURM_JOB_ID' in os.environ:
        jobid = os.environ['SLURM_JOB_ID']  # SLURM_JOB_ID
        # array_jobid = os.environ.get('SLURM_ARRAY_TASK_ID')
        # if array_jobid:
        #     jobid = f'{jobid}_{array_jobid}'  # SLURM_ARRAY_TASK_ID
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            jobid = f'{jobid}_{os.environ["SLURM_ARRAY_TASK_ID"]}'  # SLURM_ARRAY_TASK_ID
            logger.debug(f'The job is managed by SLURM with SLURM_ARRAY_TASK_ID={jobid}')
        else:
            logger.debug(f'The job is managed by SLURM with SLURM_JOB_ID={jobid}')

        # Run the command 'scontrol show job {jobid}'
        p = subprocess.Popen(['scontrol', 'show', 'job', jobid], stdout=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode('UTF-8')
        """ When --mem-per-cpu=20G, searching for the line
        MinCPUsNode=1 MinMemoryCPU=210000M MinTmpDiskNode=0
        Features=(null) DelayBoot=00:00:00
        """
        """ OR when --mem=20G, searching for the line
        MinMemoryNode = 20G
        """
        """ Additionally, the line with 
        TRES=cpu=1,mem=20G,node=1,billing=1
        Is the same with either submission
        """
        start_index = out.find('MinMemoryCPU=') + 13  # <- 13 is length of search string
        """
        Since default value is in M (MB), memory shouldn't be more than ~1000000 (1000 GB RAM?!)
        Use plus 10 characters to parse. Value could be 50 I suppose and the split will get this variable only...
        """
        # try:
        memory_constraint = out[start_index:start_index + 10].split()[0]
        # except IndexError:
        #     print(out)
        #     print(f"start_index where 'MinMemoryCPU=' '=' was found: {start_index}")
        logger.debug(f'Found memory allocated of: {memory_constraint}')
        if human_readable:
            pass
        else:
            # try:
            memory_constraint = human2bytes(memory_constraint)
            # except ValueError:
            #     print(out)
            #     print(f"start_index where 'MinMemoryCPU=' '=' was found: {start_index}")
    else:
        memory_constraint = psutil.virtual_memory().available
        if human_readable:
            memory_constraint = bytes2human(memory_constraint)

    return memory_constraint


##############################
# Directory Handling
##############################


def get_base_root_paths_recursively(directory: AnyStr, sort: bool = True) -> list[AnyStr]:
    """Retrieve the bottom most directories recursively from a root directory

    Args:
        directory: The root directory of interest
        sort: Whether the files should be filtered by name before returning
    Returns:
        The list of directories matching the search
    """
    file_generator = (os.path.abspath(root) for root, dirs, files in os.walk(directory) if not dirs)
    return sorted(file_generator) if sort else list(file_generator)


def get_file_paths_recursively(directory: AnyStr, extension: str = None, sort: bool = True) -> list[AnyStr]:
    """Retrieve files recursively from a directory

    Args:
        directory: The directory of interest
        extension: A extension to filter by
        sort: Whether the files should be filtered by name before returning
    Returns:
        The list of files matching the search
    """
    if extension is not None:
        file_generator = (os.path.join(os.path.abspath(root), file)
                          for root, dirs, files in os.walk(directory, followlinks=True) for file in files
                          if extension in file)
    else:
        file_generator = (os.path.join(os.path.abspath(root), file)
                          for root, dirs, files in os.walk(directory, followlinks=True) for file in files)

    return sorted(file_generator) if sort else list(file_generator)


def get_directory_file_paths(directory: AnyStr, suffix: str = '', extension: str = '', sort: bool = True) -> \
        list[AnyStr]:
    """Return all files in a directory with specified extensions and suffixes

    Args:
        directory: The directory of interest
        suffix: A string to match before the extension. A glob pattern is built as follows "*suffix*extension"
            ex: suffix="model" matches "design_model.pdb" and "model1.pdb"
        extension: A extension to filter by. Include the "." if there is one
        sort: Whether the files should be filtered by name before returning
    Returns:
        The list of files matching the search
    """
    directory = os.path.abspath(directory)
    file_generator = (file for file in glob(os.path.join(directory, f'*{suffix}*{extension}')))

    return sorted(file_generator) if sort else list(file_generator)


def collect_nanohedra_designs(files: Sequence = None, directory: str = None, dock: bool = False) -> \
        tuple[list[AnyStr], str]:
    """Grab all poses from a Nanohedra directory via a file or a directory

    Args:
        files: Iterable with disk location of files containing design directories
        directory: Disk location of the program directory
        dock: Whether the designs are in current docking run
    Returns:
        The absolute paths to Nanohedra output directories for all pose directories found
    """
    if files:
        all_paths = []
        for file in files:
            if not os.path.exists(file):
                logger.critical(f'No "{file}" file found! Please ensure correct location/name!')
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
                    raise InputError(f'{file} is a directory not a file. Did you mean to run with --directory?')
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
    to the PoseJob symmetry attribute
    """
    nanohedra_dirs = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if putils.master_log in files:
            nanohedra_dirs.append(root)
            del dirs[:]

    return nanohedra_dirs


def get_docked_directories(base_directory, directory_type='NanohedraEntry'):  # '*DockedPoses'
    """Useful for when your docked directory is basically known but the """
    return [os.path.join(root, _dir) for root, dirs, files in os.walk(base_directory) for _dir in dirs
            if directory_type in _dir]


def get_docked_dirs_from_base(base: str) -> list[AnyStr]:
    """Find every Nanohedra output base directory where each of the poses and files is contained

    Args:
        base: The base of the filepath corresponding to the Nanohedra master output directory

    Returns:
        The absolute path to every directory containing Nanohedra output
    """
    # base/building_blocks/degen/rot/tx/'
    # abspath removes trailing separator as well
    return sorted(set(map(os.path.abspath, glob(f'{base}{f"{os.sep}*" * 4}{os.sep}'))))


class SymDesignException(Exception):
    pass


class InputError(SymDesignException):
    pass


class SymmetryInputError(SymDesignException):
    pass


class ReportException(SymDesignException):
    pass


class MetricsError(SymDesignException):
    pass


def collect_designs(files: Sequence = None, directory: AnyStr = None, projects: Sequence = None,
                    singles: Sequence = None) -> tuple[list, str]:
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
                logger.critical(f'No "{file}" file found! Please ensure correct location/name!')
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
                    raise InputError(f'{file} is a directory not a file. Did you mean to run with --file?')
                all_paths.extend(paths)
    else:
        base_directory = get_program_root_directory(directory)
        # return all design directories within:
        #  base directory -> /base/Projects/project1, ... /base/Projects/projectN
        #  specified projects -> /base/Projects/project1, /base/Projects/project2, ...
        #  specified singles -> /base/Projects/project/design1, /base/Projects/project/design2, ...
        if base_directory or projects or singles:
            all_paths = get_program_directories(base=base_directory, projects=projects, singles=singles)
        elif directory:  # This is probably an uninitialized project. Grab all .pdb files
            all_paths = get_directory_file_paths(directory, extension='.pdb')
            directory = os.path.basename(directory)  # This is for the location variable return
        else:  # function was called with all set to None. This shouldn't happen
            raise InputError(f"Can't {type(collect_designs.__name__)} with no arguments passed")

    location = (files or directory or projects or singles)

    return sorted(set(all_paths)), location  # if isinstance(location, str) else location[0]  # Grab first index


def get_program_root_directory(search_path: str = None) -> AnyStr | None:
    """Find the program_output variable in the specified path and return the path to it

    Args:
        search_path: The path to search
    Returns:
        The absolute path of the identified program root
    """
    base_dir = None
    if search_path is not None:
        search_path = os.path.abspath(search_path)
        if putils.program_output in search_path:   # directory1/program_output/directory2/directory3
            for idx, dirname in enumerate(search_path.split(os.sep), 1):
                if dirname == putils.program_output:
                    base_dir = f'{os.sep}{os.path.join(*search_path.split(os.sep)[:idx])}'
                    break
        else:
            try:
                all_files = os.listdir(search_path)
            except FileNotFoundError:
                all_files = []
            if putils.program_output in all_files:  # directory_provided/program_output
                for sub_directory in all_files:
                    if sub_directory == putils.program_output:
                        base_dir = os.path.join(search_path, sub_directory)
                        break

    return base_dir


def get_program_directories(base: str = None, projects: Iterable = None, singles: Iterable = None) -> Iterator:
    """Return the specific design directories from the specified hierarchy with the format
    /base(program_output)/Projects/project/design
    """
    paths = []
    if base:
        paths.extend(glob(f'{base}{os.sep}{putils.projects}{os.sep}*{os.sep}*{os.sep}'))  # base/Projects/*/*/
    if projects:
        for project in projects:
            paths.extend(glob(f'{project}{os.sep}*{os.sep}'))  # base/Projects/project/*/
    if singles:
        for single, extension in map(os.path.splitext, singles):  # Remove extensions
            paths.extend(glob(f'{single}{os.sep}'))  # base/Projects/project/single/
    return map(os.path.abspath, paths)


# class PoseSpecification(csv.Dialect):
#     delimiter = ','
#     quotechar = '"'
#     escapechar = None
#     doublequote = True
#     skipinitialspace = False
#     lineterminator = '\n'
#     quoting = csv.QUOTE_MINIMAL
#     strict = False
class PoseSpecification:
    def __init__(self, file: AnyStr):
        super().__init__()
        self.directive_delimiter: str = ':'
        self.file: AnyStr = file
        self.directives: list[dict[int, str]] = []

        all_poses, design_names, all_design_directives, = [], [], []
        with open(self.file) as f:
            # pose_identifiers, design_names, all_design_directives, *_ = zip(*reader(file, dialect=self))
            all_info = list(zip(*csv.reader(f)))  # dialect=self)))

        for idx in range(len(all_info)):
            if idx == 0:
                all_poses = all_info[idx]
            elif idx == 1:
                design_names = all_info[idx]
            elif idx == 2:
                all_design_directives = all_info[idx]

        # logger.debug(f'Found poses {all_poses}')
        # logger.debug(f'Found designs {design_names}')
        # logger.debug(f'Found directives {all_design_directives}')
        self.pose_identifiers: list[str] = list(map(str.strip, all_poses))
        self.design_names: list[str] = list(map(str.strip, design_names))

        # First, split directives by white space, then by directive_delimiter
        # self.directives = \
        #     [dict((residue, directive) for residues_s, directive in [residue_directive.split(self.directive_delimiter)
        #                                                              for residue_directive in design_directives.split()]
        #           for residue in format_index_string(residues_s)) for design_directives in all_design_directives]
        for design_directives in all_design_directives:
            # print('Design Directives', design_directives)
            residue_directives = []
            # print('splitting residues', design_directives.split())
            # print('splitting directives', list(map(str.split, design_directives.split(), repeat(self.directive_delimiter))))
            for residues_s, directive in map(str.split, design_directives.split(), repeat(self.directive_delimiter)):
                # residues_s, directive = _directive.split(self.directive_delimiter)
                residue_directives.extend([(residue, directive) for residue in format_index_string(residues_s)])
            # print('Residue Directives', residue_directives)
            self.directives.append(dict(residue_directive for residue_directive in residue_directives))
        # print('Total Design Directives', self.directives)

    def get_directives(self) -> Iterator[tuple[str, list[str] | None, list[dict[int, str]] | None]]:
        """Retrieve the parsed PoseID, Design Name, and Mutation Directive information from a Specification file

        Returns:
            An iterator which returns tuples containing the PoseID followed by corresponding
        """
        # Calculate whether there are multiple designs present per pose
        found_poses = {}
        for idx, pose in enumerate(self.pose_identifiers):
            if pose in found_poses:
                found_poses[pose].append(idx)
            else:
                found_poses[pose] = [idx]

        # Ensure correctly sized inputs. Create blank data otherwise
        number_pose_identifiers = len(self.pose_identifiers)
        if self.directives:
            if number_pose_identifiers != len(self.directives):
                raise ValueError('The inputs to the PoseSpecification have different lengths!')
        else:
            directives = list(repeat(None, number_pose_identifiers))

        if self.design_names:  # design_file
            if number_pose_identifiers != len(self.design_names):
                raise ValueError('The inputs to the PoseSpecification have different lengths!')
        else:
            design_names = list(repeat(None, number_pose_identifiers))

        # Group the pose_identifiers with the design_names and directives
        if len(found_poses) == number_pose_identifiers:  # There is one design per pose
            if self.directives:
                directives = [[directive] for directive in self.directives]
            if self.design_names:
                design_names = [[design_name] for design_name in self.design_names]
        else:  # More than one
            if self.directives:
                directives = []
                for indices in found_poses.values():
                    directives.append([self.directives[index] for index in indices])
            if self.design_names:
                design_names = []
                for indices in found_poses.values():
                    design_names.append([self.design_names[index] for index in indices])

        return zip(self.pose_identifiers, design_names, directives)


######################
# Fragment Handling
######################


def format_guide_coords_as_atom(coordinate_sets: Iterable[np.array]):
    atom_string = 'ATOM  %s {:>4s}{:1s}%s %s%s{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'
    alt_location = ''
    code_for_insertion = ''
    occ = 1
    temp_fact = 20.
    charge = ''
    atom_types = ['O', 'N', 'C']
    # chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chain_numbers = '0123456789'
    atom_idx = 1

    # add placeholder atoms for setting matrix transform
    atoms = []
    # it_chain_numbers = iter(chain_numbers)
    # chain = next(it_chain_numbers)
    chain = chain_numbers[0]
    for set_idx, points in enumerate(coordinate_sets, 1):
        for point_idx, point in enumerate(points):
            atom_type = atom_types[point_idx]
            atoms.append(
                atom_string.format(format(atom_type, '<3s'), alt_location, code_for_insertion, occ, temp_fact,
                                   atom_type, charge)
                % (format(atom_idx, '5d'), 'XXX', chain, format(set_idx, '4d'),
                   '{:8.3f}{:8.3f}{:8.3f}'.format(*point)))
            atom_idx += 1

    return atoms


######################
# Matrix Handling
######################


def all_vs_all(iterable: Iterable, func: Callable, symmetrize: bool = True) -> np.ndarray:
    """Calculate an all versus all comparison using a defined function. Matrix is symmetrized by default

    Args:
        iterable: Dictionary or array like object
        func: Function to calculate different iterations of the iterable
        symmetrize: Whether to make the resulting matrix symmetric
    Returns:
        Matrix with resulting calculations
    """
    if isinstance(iterable, dict):
        # func(iterable[obj1], iterable[obj2])
        _dict = iterable
    else:
        _dict = None

    pairwise = np.zeros((len(iterable), (len(iterable))))
    for i, obj1 in enumerate(iterable[:-1]):
        j = i+1
        for j, obj2 in enumerate(iterable[j:], j):
            # if type(iterable) == dict:  # _dict
            pairwise[i][j] = func(obj1, obj2, d=_dict)
            # pairwise[i][j] = func(obj1, obj2, iterable, d=_dict)
            # else:
            #     pairwise[i][j] = func(obj1, obj2, iterable, d=_dict)

    if symmetrize:
        return sym(pairwise)
    else:
        return pairwise


def sym(a: np.ndarray) -> np.ndarray:
    """Symmetrize a numpy array. i.e. if a_ij = 0, then the returned array is such that a_ji = a_ij

    Args:
        a: A 2D square array
    Returns:
        Symmetrized array
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

# from . import cluster
# from . import distribute
from . import rosetta, SymEntry, symmetry
