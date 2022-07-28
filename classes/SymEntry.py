from __future__ import annotations

import math
import os
import warnings
from typing import AnyStr

import numpy as np

import PathUtils as PUtils
from SymDesignUtils import start_log, dictionary_lookup, DesignError, SymmetryError, format_guide_coords_as_atom
from utils.SymmetryUtils import valid_subunit_number, space_group_symmetry_operators, point_group_symmetry_operators, \
    all_sym_entry_dict, rotation_range, setting_matrices, identity_matrix, sub_symmetries, flip_y_matrix, max_sym, \
    valid_symmetries

# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

logger = start_log(name=__name__)
null_log = start_log(name='null', handler=3)
symmetry_combination_format = 'ResultingSymmetry:{Component1Symmetry}{Component2Symmetry}{...}'
# SYMMETRY COMBINATION MATERIAL TABLE (T.O.Y and J.L, 2020)
nanohedra_symmetry_combinations = {
    1: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C2', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 4, 2],
    2: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C3', ['r:<0,0,1,c>'], 1, '<e,0.577350*e,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 6],
    3: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 4, 2],
    4: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 5, 6],
    5: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'T', 'T', 0, 'N/A', 4, 3],
    6: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,e,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'T', 'I213', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 5, 10],
    7: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 4],
    8: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 5, 10],
    9: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 7, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 5],
    10: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C4', ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C4', 'p4', 2, '(2*e, 2*e), 90', 4, 4],
    11: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 4, 2],
    12: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<e,0,0>', 'D4', 'p4212', 2, '(2*e, 2*e), 90', 5, 4],
    13: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 3],
    14: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 5, 8],
    15: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D5', 'D5', 0, 'N/A', 4, 2],
    16: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 9, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 3],
    17: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C6', ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 3],
    18: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C6', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D6', 'D6', 0, 'N/A', 4, 2],
    19: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'C6', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 5, 4],
    20: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,f,0>', 'D2', [], 1, '<0,0,0>', 'D2', 'c222', 2, '(4*e, 4*f), 90', 4, 4],
    21: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,0,0>', 'D2', [], 1, '<e,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 4],
    22: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,f>', 'D2', [], 5, '<0,0,0>', 'D4', 'I4122', 3, '(4*e, 4*e, 8*f), (90, 90, 90)', 4, 6],
    23: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 10, '<0,0,0>', 'D2', [], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 3],
    24: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 10, '<0,0,e>', 'D2', [], 1, '<f,0,0>', 'D6', 'P6222', 3, '(2*f, 2*f, 6*e), (90, 90, 120)', 4, 6],
    25: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D2', [], 5, '<2*e,0,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 4],
    26: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<-2*e,3*e,0>', 'D2', [], 5, '<0,2*e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    27: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'D3', [], 11, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 3, 3],
    28: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,f>', 'D3', [], 1, '<0,0,0>', 'D3', 'R32', 3, '(3.4641*e, 3.4641*e, 3*f), (90, 90, 120)', 4, 4],
    29: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D3', [], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    30: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    31: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,f>', 'D6', 'P6322', 3, '(2*e, 2*e, 4*f), (90, 90, 120)', 4, 4],
    32: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', [], 4, '<e,e,e>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    33: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,2*e,0>', 'D3', [], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    34: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D3', [], 4, '<e,e,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 4],
    35: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,-2*e>', 'D3', [], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    36: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,-2*e>', 'D3', [], 4, '<3*e,3*e,3*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    37: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    38: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    39: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,e,f>', 'D4', [], 1, '<0,0,0>', 'D4', 'I422', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 4, 4],
    40: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D4', [], 1, '<0,0,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 3],
    41: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'D4', [], 1, '<2*e,2*e,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    42: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    43: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    44: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,f>', 'D6', [], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 4, 4],
    45: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    46: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'T', [], 1, '<0,0,0>', 'T', 'F23', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    47: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,3*e,0>', 'T', [], 1, '<0,4*e,0>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    48: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    49: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'O', [], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    50: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<e,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    51: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,0>', 'O', [], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    52: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<-e,e,e>', 'O', [], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    53: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', ['r:<0,0,1,c>'], 1, '<e,0.57735*e,0>', 'C3', 'p3', 2, '(2*e, 2*e), 120', 4, 3],
    54: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 12, '<0,0,0>', 'T', 'T', 0, 'N/A', 4, 2],
    55: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 12, '<e,0,0>', 'T', 'P213', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 5],
    56: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 2],
    57: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<e,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 6],
    58: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 7, '<0,0,0>', 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 9, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 2],
    59: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'C6', ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 2],
    60: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D2', [], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    61: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', [], 1, '<e,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 3],
    62: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', [], 3, '<e,0,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    63: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', [], 3, '<2*e,e,0>', 'O', 'I4132', 3, '(8*e,8*e, 8*e), (90, 90, 90)', 3, 2],
    64: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D3', [], 11, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 3, 2],
    65: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D3', [], 1, '<0,0,0>', 'D3', 'p321', 2, '(2*e, 2*e), 120', 3, 2],
    66: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 12, '<4*e,0,0>', 'D3', [], 4, '<3*e,3*e,3*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 4],
    67: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D4', [], 1, '<0,0,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    68: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    69: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<e,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'F23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    70: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<e,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    71: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C4', ['r:<0,0,1,c>'], 1, '<e,e,0>', 'C4', 'p4', 2, '(2*e, 2*e), 90', 4, 2],
    72: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 2, '<0,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 4],
    73: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', [], 1, '<e,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    74: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D2', [], 5, '<0,0,0>', 'D4', 'p4212', 2, '(2*e, 2*e), 90', 3, 2],
    75: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D2', [], 3, '<2*e,e,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    76: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', [], 3, '<e,0,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    77: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', [], 4, '<e,e,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    78: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    79: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D4', [], 1, '<e,e,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    80: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'T', [], 1, '<e,e,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    81: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'O', [], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    82: ['C6', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', [], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    83: ['C6', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 2, 2],
    84: ['D2', [], 1, '<0,0,0>', 'D2', [], 1, '<e,f,0>', 'D2', 'p222', 2, '(2*e, 2*f), 90', 2, 2],
    85: ['D2', [], 1, '<0,0,0>', 'D2', [], 1, '<e,f,g>', 'D2', 'F222', 3, '(4*e, 4*f, 4*g), (90, 90, 90)', 3, 3],
    86: ['D2', [], 1, '<e,0,0>', 'D2', [], 5, '<0,0,f>', 'D4', 'P4222', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 2, 2],
    87: ['D2', [], 1, '<e,0,0>', 'D2', [], 13, '<0,0,-f>', 'D6', 'P6222', 3, '(2*e, 2*e, 6*f), (90, 90, 120)', 2, 2],
    88: ['D2', [], 3, '<0,e,2*e>', 'D2', [], 5, '<0,2*e,e>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    89: ['D2', [], 1, '<e,0,0>', 'D3', [], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    90: ['D2', [], 1, '<e,0,0>', 'D3', [], 11, '<e,0.57735*e,f>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    91: ['D2', [], 1, '<0,0,2*e>', 'D3', [], 4, '<e,e,e>', 'D6', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    92: ['D2', [], 3, '<2*e,e,0>', 'D3', [], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    93: ['D2', [], 1, '<e,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 1, 1],
    94: ['D2', [], 1, '<e,0,f>', 'D4', [], 1, '<0,0,0>', 'D4', 'P422', 3, '(2*e, 2*e, 2*f), (90, 90, 90)', 2, 2],
    95: ['D2', [], 5, '<e,0,f>', 'D4', [], 1, '<0,0,0>', 'D4', 'I422', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 2, 2],
    96: ['D2', [], 3, '<0,e,2*e>', 'D4', [], 1, '<0,0,2*e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    97: ['D2', [], 1, '<e,0,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    98: ['D2', [], 1, '<e,0,f>', 'D6', [], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    99: ['D2', [], 1, '<e,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    100: ['D2', [], 1, '<e,e,0>', 'T', [], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 2],
    101: ['D2', [], 3, '<e,0,e>', 'T', [], 1, '<e,e,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    102: ['D2', [], 3, '<2*e,e,0>', 'T', [], 1, '<0,0,0>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    103: ['D2', [], 3, '<e,0,e>', 'O', [], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    104: ['D2', [], 3, '<2*e,e,0>', 'O', [], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    105: ['D3', [], 11, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 1, 1],
    106: ['D3', [], 11, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,f>', 'D3', 'P312', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    107: ['D3', [], 1, '<0,0,0>', 'D3', [], 11, '<e,0.57735*e,f>', 'D6', 'P6322', 3, '(2*e, 2*e, 4*f), (90, 90, 120)', 2, 2],
    108: ['D3', [], 4, '<e,e,e>', 'D3', [], 12, '<e,3*e,e>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    109: ['D3', [], 4, '<3*e,3*e,3*e>', 'D3', [], 12, '<e,3*e,5*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    110: ['D3', [], 4, '<e,e,e>', 'D4', [], 1, '<0,0,2*e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    111: ['D3', [], 11, '<e,0.57735*e,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    112: ['D3', [], 11, '<e,0.57735*e,f>', 'D6', [], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    113: ['D3', [], 4, '<e,e,e>', 'T', [], 1, '<0,0,0>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    114: ['D3', [], 4, '<e,e,e>', 'O', [], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    115: ['D4', [], 1, '<0,0,0>', 'D4', [], 1, '<e,e,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 1, 1],
    116: ['D4', [], 1, '<0,0,0>', 'D4', [], 1, '<e,e,f>', 'D4', 'P422', 3, '(2*e, 2*e, 2*f), (90, 90,90)', 2, 2],
    117: ['D4', [], 1, '<0,0,e>', 'D4', [], 2, '<0,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    118: ['D4', [], 1, '<0,0,e>', 'O', [], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    119: ['D4', [], 1, '<e,e,0>', 'O', [], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    120: ['T', [], 1, '<0,0,0>', 'T', [], 1, '<e,e,e>', 'T', 'F23', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    121: ['T', [], 1, '<0,0,0>', 'T', [], 1, '<e,0,0>', 'T', 'F23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    122: ['T', [], 1, '<e,e,e>', 'O', [], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    123: ['O', [], 1, '<0,0,0>', 'O', [], 1, '<e,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    124: ['O', [], 1, '<0,0,0>', 'O', [], 1, '<e,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
}
# KM Custom entries
symmetry_combinations = {
    125: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C1', 'C1', 0, 'N/A', 0, 1],  # C1 alone
    126: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>',
          'C1', 'C1', 0, 'N/A', 6, 1],
    130: ['C2', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C2', 'C2', 0, 'N/A', 0, 1],  # C2 alone
    131: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C2', [], 1, '<0,0,0>', 'C2', 'C2', 0, 'N/A', 6, 1],
    135: ['C3', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C3', 'C3', 0, 'N/A', 0, 1],  # C3 alone
    136: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C3', [], 1, '<0,0,0>', 'C3', 'C3', 0, 'N/A', 6, 1],
    140: ['C4', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C4', 'C4', 0, 'N/A', 0, 1],  # C4 alone
    141: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C4', [], 1, '<0,0,0>', 'C4', 'C4', 0, 'N/A', 6, 1],
    145: ['C5', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C5', 'C5', 0, 'N/A', 0, 1],  # C5 alone
    146: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C5', [], 1, '<0,0,0>', 'C5', 'C5', 0, 'N/A', 6, 1],
    150: ['C6', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'C6', 'C6', 0, 'N/A', 0, 1],  # C6 alone
    151: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'C6', [], 1, '<0,0,0>', 'C6', 'C6', 0, 'N/A', 6, 1],
    155: ['D2', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 0, 1],  # D2 alone
    156: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'D2', [], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 6, 1],
    157: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', [], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 2, 1],
    158: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D2', [], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 2, 1],
    160: ['D3', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 0, 1],  # D3 alone
    161: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'D3', [], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 6, 1],
    162: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D3', [], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 2, 1],
    163: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', [], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 2, 1],
    165: ['D4', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 0, 1],  # D4 alone
    166: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 6, 1],
    167: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 2, 1],
    168: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 2, 1],
    169: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D4', [], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 2, 1],
    170: ['D5', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'D5', 'D5', 0, 'N/A', 0, 1],  # D5 alone
    171: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'D5', [], 1, '<0,0,0>', 'D5', 'D5', 0, 'N/A', 6, 1],
    175: ['D6', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'D6', 'D6', 0, 'N/A', 0, 1],  # D6 alone
    176: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'D6', [], 1, '<0,0,0>', 'D6', 'D6', 0, 'N/A', 6, 1],
    200: ['T', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 0, 1],  # T alone
    201: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 6, 1],
    202: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 2, 1],
    203: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 2, 1],
    210: ['O', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 0, 1],  # O alone
    211: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 6, 1],
    212: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 2, 1],
    213: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 2, 1],
    214: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 2, 1],
    220: ['I', [], 1, '<0,0,0>', None, [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 0, 1],  # I alone
    221: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 6, 1],
    222: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 2, 1],
    223: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 7, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 2, 1],
    224: ['C5', ['r:<0,0,1,a>', 't:<0,0,b>'], 9, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 2, 1],
    # KM 3 component entries
    # 301: {'components': [{'symmetry': 'C1', 'dof_internal': ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C2', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C3', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 4, 'dof_external': '<0,0,0>'}]
    #                       , 'result': ['T', 'T', 0, 'N/A', 1, 1]},
}
# Standard T:{C3}{C3}
# Number   grp1 grp1_idx            grp1_internal_dof grp1_set_mat grp1_external_dof
# 54:     ['C3',      2, ['r:<0,0,1,a>', 't:<0,0,b>'],          4,        '<0,0,0>',
#          grp2 grp2_idx            grp2_internal_dof grp2_set_mat grp2_external_dof
#          'C3',      2, ['r:<0,0,1,c>', 't:<0,0,d>'],         12,        '<0,0,0>',
#          pnt_grp final_sym dim  unit_cell tot_dof ring_size
#          'T',         'T',  0,     'N/A',      4,       2],
# Modified T:{C3}{C3} with group 1 internal DOF allowed, group 2, internal DOF disabled
# 54:     ['C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'],  4, '<0,0,0>',
#          'C3', 2,                           [], 12, '<0,0,0>',
#          'T', 'T', 0, 'N/A', 4, 2],
custom_entries = list(symmetry_combinations.keys())
symmetry_combinations.update(nanohedra_symmetry_combinations)
# reformat the symmetry_combinations to account for groups and results separately
parsed_symmetry_combinations: dict[int, tuple[list[tuple[str, list | int | str]], str | int]] = \
    {entry_number: ([(entry[0], entry[1:4]), (entry[4], entry[5:8])], entry[-6:])
     for entry_number, entry in symmetry_combinations.items()}
space_group_to_sym_entry = {}
# ROTATION SETTING MATRICES - All descriptions are with view on the positive side of respective axis
# These specify combinations of symmetric point groups which can be used to construct a larger point group
point_group_setting_matrix_members = {
    # Up until 'T' are all correct for original 124, but dynamic dict creation is preferred with additional combinations
    # 'C3': {'C2': {1}, 'C3': {1}},
    # # 53
    # 'C4': {'C2': {1}, 'C4': {1}},
    # # 10, 71
    # 'C6': {'C2': {1}, 'C3': {1}, 'C6': {1}},
    # # 2, 17, 59
    # 'D2': {'C2': {1, 2}, 'D2': {1}},
    # # 1, 20, 84, 85
    # 'D3': {'C2': {2, 6}, 'C3': {1}, 'D3': {1, 11}},
    # # 3, 4, 27, 28, 64, 65,
    # 'D4': {'C2': {1, 2, 8}, 'C4': {1}, 'D2': {1, 5}, 'D4': {1}},
    # # 11, 12, 21, 22, 37-39, 73, 74, 78, 86, 93-95, 115, 116
    # 'D5': {'C2': {2}, 'C5': {1}},
    # # 15
    # 'D6': {'C2': {1, 2, 6, 10}, 'C3': {1}, 'C6': {1}, 'D2': {1, 13}, 'D3': {1, 4, 11}, 'D6': {1}},
    # # 18, 19, 23, 24, 29-31, 42-44, 60, 68, 82, 83, 87, 89-91, 97, 98, 107, 111, 112
    # 'T': {'C2': {1}, 'C3': {4, 12}},  # might have to check using degeneracy mat mul to first setting matrix 6(4)=12
    # 'O': {'C2': {3}, 'C3': {4, 12}, 'C4': {1}},
    # 'I': {'C2': {1}, 'C3': {7}, 'C5': {9}},
}
for entry_number, ent in symmetry_combinations.items():
    group_1, _, setting_1, _, group_2, _, setting_2, _, point_group, _, _, _, _, _ = ent
    result_entry = point_group_setting_matrix_members.get(point_group, None)
    if result_entry:
        if group_1 in result_entry:
            result_entry[group_1].add(setting_1)
        else:
            result_entry[group_1] = {setting_1}

        if group_2 in result_entry:
            result_entry[group_2].add(setting_2)
        else:
            result_entry[group_2] = {setting_2}
    else:
        point_group_setting_matrix_members[point_group] = {group_1: {setting_1}, group_2: {setting_2}}


class SymEntry:
    __external_dof: list[np.ndarray]
    _degeneracy_matrices: np.ndarray | None
    _external_dof: np.ndarray
    _int_dof_groups: list[str]
    _ref_frame_tx_dof: list[str]
    _setting_matrices: list[np.ndarray]
    cycle_size: int
    dimension: int
    entry_number: int
    groups: list[str]
    n_dof_external: int
    point_group_symmetry: str
    resulting_symmetry: str
    sym_map: list[str]
    total_dof: int
    unit_cell: list[list[str], list[str]]
    expand_matrices: np.ndarray

    def __init__(self, entry: int, sym_map: list[str] = None):
        try:
            # group1, self.int_dof_group1, self.rot_set_group1, self.ref_frame_tx_dof1, \
            #     group2, self.int_dof_group2, self.rot_set_group2, self.ref_frame_tx_dof2, \
            #     self.point_group_symmetry, self.resulting_symmetry, self.dimension, self.unit_cell, self.tot_dof, \
            #     self.cycle_size = nanohedra_symmetry_combinations.get(entry)
            group_info, result_info = parsed_symmetry_combinations.get(entry)
            # returns
            #  {'group1': [self.int_dof_group1, self.rot_set_group1, self.ref_frame_tx_dof1],
            #   'group2': [self.int_dof_group2, self.rot_set_group2, self.ref_frame_tx_dof2],
            #   ...},
            #  [point_group_symmetry, resulting_symmetry, dimension, unit_cell, tot_dof, cycle_size]
            self.point_group_symmetry, self.resulting_symmetry, self.dimension, self.unit_cell, self.total_dof, \
                self.cycle_size = result_info
        except KeyError:
            raise ValueError(f'Invalid symmetry entry "{entry}". Supported values are Nanohedra entries: '
                             f'{1}-{len(nanohedra_symmetry_combinations)} and custom entries: '
                             f'{", ".join(map(str, custom_entries))}')
        self.entry_number = entry
        entry_groups = [group_name for group_name, group_params in group_info]
        if not sym_map:  # assume standard SymEntry
            # assumes 2 component symmetry. index with only 2 options
            self.groups = entry_groups
            self.sym_map = [self.resulting_symmetry] + self.groups
        else:  # requires full specification of all symmetry groups
            self.groups = []
            self.sym_map = sym_map
            result, *sym_map = self.sym_map  # remove the result and pass the groups
            for idx, sub_symmetry in enumerate(sym_map, 1):
                if sub_symmetry not in valid_symmetries:
                    if sub_symmetry is None:
                        pass  # Todo ignore for now. Need to refactor symmetry_combinations for any number of elements
                    else:
                        raise ValueError(f'The symmetry "{sub_symmetry}" specified at index "{idx}" is not a valid '
                                         f'sub-symmetry')
                if sub_symmetry not in entry_groups:  # Todo add sub_symmetry specification to group info
                    raise SymmetryError('This functionality hasn\'t been implemented yet!')
                self.groups.append(sub_symmetry)

        self._int_dof_groups, self._setting_matrices, self._ref_frame_tx_dof, self.__external_dof = [], [], [], []
        for group_idx, group_symmetry in enumerate(self.groups, 1):
            for entry_group_symmetry, (int_dof, set_mat, ext_dof) in group_info:
                if group_symmetry == entry_group_symmetry:
                    self._int_dof_groups.append(int_dof)
                    self._setting_matrices.append(setting_matrices[set_mat])
                    ref_frame_tx_dof = list(map(str.strip, ext_dof.strip('<>').split(',')))
                    self._ref_frame_tx_dof.append(ref_frame_tx_dof)
                    if group_idx <= 1:
                        # this wouldn't be possible with more than 2 groups unless we tether group to an existing group
                        self.__external_dof.append(construct_uc_matrix(ref_frame_tx_dof))
                    else:
                        if getattr(self, f'is_ref_frame_tx_dof{group_idx}'):
                            raise SymmetryError('Cannot yet create a SymEntry with external degrees of freedom and > 2 '
                                                'groups!')
        # Reformat reference_frame entries
        # self.is_ref_frame_tx_dof1 = False if self.ref_frame_tx_dof1 == '<0,0,0>' else True
        # self.is_ref_frame_tx_dof2 = False if self.ref_frame_tx_dof2 == '<0,0,0>' else True
        # self.ref_frame_tx_dof1 = list(map(str.strip, self.ref_frame_tx_dof1.strip('<>').split(',')))
        # self.ref_frame_tx_dof2 = list(map(str.strip, self.ref_frame_tx_dof2.strip('<>').split(',')))
        # self.external_dof1 = construct_uc_matrix(self.ref_frame_tx_dof1)
        # [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # self.external_dof2 = construct_uc_matrix(self.ref_frame_tx_dof2)
        # [[0, 0, 1], [0, 0, 0], [0, 0, 0]]

        # if not self.is_ref_frame_tx_dof1 and not self.is_ref_frame_tx_dof2:
        #     self.ext_dof = np.empty((0, 3), float)  # <- np.array([[0.], [0.], [0.]])
        # else:
        #     difference_matrix = self.external_dof2 - self.external_dof1
        #     # for entry 6 - string_vector is 4*e, 4*e, 4*e
        #     # which is uc_dimension_matrix of [[4, 4, 4], [0, 0, 0], [0, 0, 0]]
        #     #  (^).sum(axis=-1)) = [12, 0, 0]
        #     # component1 string vector is ['0', 'e', '0']
        #     #  [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        #     # component2 string vector is ['0', '0', '0']
        #     #  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        #     # difference_matrix = [[0, -1, 0], [0, 0, 0], [0, 0, 0]]
        #     #  (^).sum(axis=-1)) = [-1, 0, 0]
        #     self.ext_dof = difference_matrix[np.nonzero(difference_matrix.sum(axis=-1))]

        self.n_dof_external = self.external_dof.shape[0]
        # Check construction is valid
        if self.point_group_symmetry not in valid_symmetries:
            raise ValueError(f'Invalid point group symmetry {self.point_group_symmetry}')
        try:
            if self.dimension == 0:
                self.expand_matrices = point_group_symmetry_operators[self.resulting_symmetry]
            elif self.dimension in [2, 3]:
                self.expand_matrices = space_group_symmetry_operators[self.resulting_symmetry]
            else:
                raise ValueError('Invalid symmetry entry. Supported design dimensions are 0, 2, and 3')
        except KeyError:
            raise SymmetryError(f'The symmetry result "{self.resulting_symmetry}" is not allowed')
        self.unit_cell = None if self.unit_cell == 'N/A' else \
            [dim.strip('()').replace(' ', '').split(',') for dim in self.unit_cell.split('), ')]

    @property
    def number_of_operations(self) -> int:
        """The number of symmetric copies in the full symmetric system"""
        return self.expand_matrices.shape[0]

    @property
    def group_subunit_numbers(self) -> list[int]:
        """Returns the number of subunits for each symmetry group"""
        try:
            return self._group_subunit_numbers
        except AttributeError:
            self._group_subunit_numbers = [valid_subunit_number[group] for group in self.groups]
            return self._group_subunit_numbers

    @property
    def combination_string(self) -> str:
        return '%s:{%s}' % (self.resulting_symmetry, '}{'.join(self.groups))

    @property
    def simple_combination_string(self) -> str:
        return f'{self.resulting_symmetry}{"".join(self.groups)}'

    @property
    def uc_specification(self):
        return self.unit_cell

    @property
    def rotation_range1(self) -> float:
        """Return the rotation range according the first symmetry group operator"""
        try:
            return self._rotation_range[0]
        except AttributeError:
            self._rotation_range = [rotation_range.get(group, 0) for group in self.groups]
        return self._rotation_range[0]

    @property
    def rotation_range2(self) -> float:
        """Return the rotation range according the second symmetry group operator"""
        try:
            return self._rotation_range[1]
        except AttributeError:
            self._rotation_range = [rotation_range.get(group, 0) for group in self.groups]
        return self._rotation_range[1]

    @property
    def rotation_range3(self) -> float:
        """Return the rotation range according the third symmetry group operator"""
        try:
            return self._rotation_range[2]
        except AttributeError:
            self._rotation_range = [rotation_range.get(group, 0) for group in self.groups]
        return self._rotation_range[2]

    @property
    def group1(self) -> str:
        return self.groups[0]

    @property
    def group2(self) -> str:
        return self.groups[1]

    @property
    def group3(self) -> str:
        return self.groups[2]

    @property
    def setting_matrices(self) -> list[np.ndarray]:
        return self._setting_matrices

    @property
    def setting_matrix1(self) -> np.ndarray:
        return self._setting_matrices[0]

    @property
    def setting_matrix2(self) -> np.ndarray:
        return self._setting_matrices[1]

    @property
    def setting_matrix3(self) -> np.ndarray:
        return self._setting_matrices[2]

    @property
    def is_internal_rot1(self) -> bool:
        return 'r:<0,0,1,a>' in self._int_dof_groups[0]

    @property
    def is_internal_rot2(self) -> bool:
        return 'r:<0,0,1,c>' in self._int_dof_groups[1]

    @property
    def is_internal_tx1(self) -> bool:
        return 't:<0,0,b>' in self._int_dof_groups[0]

    @property
    def is_internal_tx2(self) -> bool:
        return 't:<0,0,d>' in self._int_dof_groups[1]

    @property
    def is_ref_frame_tx_dof1(self) -> bool:
        return self._ref_frame_tx_dof[0] != ['0', '0', '0']
        # return self._ref_frame_tx_dof[0] != '<0,0,0>'

    @property
    def is_ref_frame_tx_dof2(self) -> bool:
        return self._ref_frame_tx_dof[1] != ['0', '0', '0']
        # return self._ref_frame_tx_dof[1] != '<0,0,0>'

    @property
    def is_ref_frame_tx_dof3(self) -> bool:
        return self._ref_frame_tx_dof[2] != ['0', '0', '0']
        # return self._ref_frame_tx_dof[1] != '<0,0,0>'

    @property
    def external_dof(self) -> np.ndarray:
        """Return the total external degrees of freedom as a number DOF externalx3 array"""
        try:
            return self._external_dof
        except AttributeError:
            if not self.is_ref_frame_tx_dof1 and not self.is_ref_frame_tx_dof2:
                self._external_dof = np.empty((0, 3), float)  # <- np.array([[0.], [0.], [0.]])
            else:
                difference_matrix = self.__external_dof[1] - self.__external_dof[0]
                # for entry 6 - string_vector is 4*e, 4*e, 4*e
                # which is uc_dimension_matrix of [[4, 4, 4], [0, 0, 0], [0, 0, 0]]
                #  (^).sum(axis=-1)) = [12, 0, 0]
                # component1 string vector is ['0', 'e', '0']
                #  [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
                # component2 string vector is ['0', '0', '0']
                #  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                # difference_matrix = [[0, -1, 0], [0, 0, 0], [0, 0, 0]]
                #  (^).sum(axis=-1)) = [-1, 0, 0]
                self._external_dof = difference_matrix[np.nonzero(difference_matrix.sum(axis=-1))]
            return self._external_dof

    @property
    def external_dof1(self) -> np.ndarray:
        """Return the 3x3 external degrees of freedom for component1"""
        return self.__external_dof[0]

    @property
    def external_dof2(self) -> np.ndarray:
        """Return the 3x3 external degrees of freedom for component2"""
        return self.__external_dof[1]

    @property
    def degeneracy_matrices1(self) -> np.ndarray:
        """Returns the (number of degeneracies, 3, 3) degeneracy matrices for component1"""
        try:
            return self._degeneracy_matrices[0]
        except AttributeError:
            self._create_degeneracy_matrices()
            return self._degeneracy_matrices[0]

    @property
    def degeneracy_matrices2(self) -> np.ndarray:
        """Returns the (number of degeneracies, 3, 3) degeneracy matrices for component2"""
        try:
            return self._degeneracy_matrices[1]
        except AttributeError:
            self._create_degeneracy_matrices()
            return self._degeneracy_matrices[1]

    def _create_degeneracy_matrices(self):
        """From the intended point group symmetry and a single component, find the degeneracy matrices that produce all
        viable configurations of the single component in the final symmetry

        Sets:
            self._degeneracy_matrices list[numpy.ndarray, ...]:
                The degeneracy matrices for each group where the matrix for each group has shape
                (number of degeneracies, 3, 3). Will always return the identity matrix if no degeneracies,
                with shape (1, 3, 3)
        """
        self._degeneracy_matrices = []
        # Todo now that SymEntry working with more than 2 symmetries, enumerate when to search these others
        for idx, group in enumerate(self.groups, 1):
            degeneracy_matrices = None
            # For cages, only one of the two groups need to be flipped. By convention, we flip oligomer 2
            if self.dimension == 0 and idx == 2:
                degeneracy_matrices = np.array([identity_matrix, flip_y_matrix])
            # For layers that obey a cyclic point group symmetry and that are constructed from two entities that both
            # obey cyclic symmetry only one of the two entities need to be flipped. By convention, we flip oligomer 2
            elif self.dimension == 2 and idx == 2 and \
                    (self.groups[0][0], self.groups[1][0], self.point_group_symmetry[0]) == ('C', 'C', 'C'):
                degeneracy_matrices = np.array([identity_matrix, flip_y_matrix])
            # else:
            #     if oligomer_symmetry[0] == "C" and design_symmetry[0] == "C":
            #         degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y
            elif group in ['D3', 'D4', 'D6'] and self.point_group_symmetry in ['D3', 'D4', 'D6', 'T', 'O']:
                # commented out "if" statement below because all possible translations are not always tested for D3
                # example: in entry 82, only translations along <e,0.577e> are sampled.
                # This restriction only considers 1 out of the 2 equivalent Wyckoff positions.
                # <0,e> would also have to be searched as well to remove the "if" statement below.
                # if (oligomer_symmetry, design_symmetry_pg) != ('D3', 'D6'):
                if group == 'D3':
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[.5, -0.866025, 0.], [.866025, .5, 0.], [0., 0., 1.]]])  # 60 deg about z
                elif group == 'D4':
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[.707107, .707107, 0.], [-0.707107, .707107, 0.], [0., 0., 1.]]])  # 45 deg about z
                elif group == 'D6':
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[0.866025, -0.5, 0.], [0.5, 0.866025, 0.], [0., 0., 1.]]])  # 30 deg about z
            elif group == 'D2' and self.point_group_symmetry != 'O':
                if self.point_group_symmetry == 'T':
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])  # 90 deg about z

                elif self.point_group_symmetry == 'D4':
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],  # z,x,y
                                  [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])  # y,z,x

                elif self.point_group_symmetry in ['D2', 'D6']:
                    degeneracy_matrices = \
                        np.array([identity_matrix,
                                  [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                                  [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
                                  [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]],
                                  [[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]],
                                  [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]])
            elif group == 'T' and self.point_group_symmetry == 'T':
                degeneracy_matrices = \
                    np.array([identity_matrix,
                              [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])  # 90 deg about z

            if degeneracy_matrices is None:  # No degeneracies, use the identity
                degeneracy_matrices = identity_matrix[None, :, :]  # expand to shape (1, 3, 3)

            self._degeneracy_matrices.append(degeneracy_matrices)

    def get_optimal_external_tx_vector(self, optimal_ext_dof_shifts: np.ndarray, group_number: int = 1) -> np.ndarray:
        """From the DOF and the computed shifts, return the translation vector

        Args:
            optimal_ext_dof_shifts: The parameters for an ideal shift of a single component in the resulting material
            group_number: The number of the group to find the vector for
        Returns:
            The optimal vector for translation
        """
        optimal_shifts_t = getattr(self, f'external_dof{group_number}').T * optimal_ext_dof_shifts
        return optimal_shifts_t.T.sum(axis=0)

    def get_uc_dimensions(self, optimal_shift_vec: np.ndarray) -> np.ndarray | None:
        """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
        basis vectors with component translation degrees of freedom

        Args:
            optimal_shift_vec: An Nx3 array where N is the number of shift instances
                and 3 is number of possible external degrees of freedom (even if they are not utilized)
        Returns:
            The Unit Cell dimensions for each optimal shift vector passed
        """
        if not self.unit_cell:
            return None
        string_lengths, string_angles = self.unit_cell
        # for entry 6 - string_vector is 4*e, 4*e, 4*e
        # construct_uc_matrix() = [[4, 4, 4], [0, 0, 0], [0, 0, 0]]
        uc_mat = construct_uc_matrix(string_lengths) * optimal_shift_vec[:, :, None]
        # [:, :, None] <- expands axis so multiplication is accurate. eg. [[[1.], [0.], [0.]], [[0.], [0.], [0.]]]
        lengths = np.abs(uc_mat.sum(axis=-2))
        #               (^).sum(axis=-2) = [4, 4, 4]
        if len(string_lengths) == 2:
            lengths[:, 2] = 1.

        if len(string_angles) == 1:
            angles = [90., 90., float(string_angles[0])]
        else:
            # angles = [float(string_angle) for string_angle in string_angles]
            angles = [0., 0., 0.]  # initialize incase there are < 1 string_angles
            for idx, string_angle in enumerate(string_angles):
                angles[idx] = float(string_angle)

        # return np.concatenate(lengths, np.tile(angles, len(lengths)))
        return np.hstack((lengths, np.tile(angles, len(lengths)).reshape(-1, 3)))

    def get_optimal_shift_from_uc_dimensions(self, a: float, b: float, c: float, *angles: list) -> np.ndarray | None:
        """Return the optimal shifts provided unit cell dimensions and the external translation degrees of freedom

        Args:
            a: The unit cell parameter for the lattice dimension 'a'
            b: The unit cell parameter for the lattice dimension 'b'
            c: The unit cell parameter for the lattice dimension 'c'
            angles: The unit cell parameters for the lattice angles alpha, beta, gamma. Not utilized!
        Returns:
            The optimal shifts in each direction a, b, and c if they are allowed
        """
        if not self.unit_cell:
            return None
        string_lengths, string_angles = self.unit_cell
        # uc_mat = construct_uc_matrix(string_lengths) * optimal_shift_vec[:, :, None]  # <- expands axis so mult accurate
        uc_mat = construct_uc_matrix(string_lengths)
        # to reverse the values from the incoming a, b, and c, we should divide by the uc_matrix_constraints
        # given the matrix should only ever have one value in each column (max) a sum over the column should produce the
        # desired vector to calculate the optimal shift.
        # There is a possibility of returning inf when we divide 0 by a value so ignore this warning
        with warnings.catch_warnings() as w:
            # Cause all warnings to always be ignored
            warnings.simplefilter('ignore')
            external_translation_shifts = [a, b, c] / np.abs(uc_mat.sum(axis=-2))
            # replace any inf with zero
            external_translation_shifts = np.nan_to_num(external_translation_shifts, copy=False, posinf=0., neginf=0.)

        if len(string_lengths) == 2:
            external_translation_shifts[2] = 1.

        return external_translation_shifts

    def sdf_lookup(self) -> AnyStr:
        """Locate the proper symmetry definition file depending on the specified symmetry

        Returns:
            The location of the symmetry definition file on disk
        """
        if self.dimension > 0:
            return os.path.join(PUtils.symmetry_def_files, 'C1.sym')

        symmetry = self.simple_combination_string
        for file, ext in map(os.path.splitext, os.listdir(PUtils.symmetry_def_files)):
            if symmetry == file:
                return os.path.join(PUtils.symmetry_def_files, file + ext)

        symmetry = self.resulting_symmetry
        for file, ext in map(os.path.splitext, os.listdir(PUtils.symmetry_def_files)):
            if symmetry == file:
                return os.path.join(PUtils.symmetry_def_files, file + ext)

        raise FileNotFoundError(f'Couldn\'t locate correct symmetry definition file at "{PUtils.symmetry_def_files}" '
                                f'for SymEntry: {self.entry_number}')


class SymEntryFactory:
    """Return a SymEntry instance by calling the Factory instance with the SymEntry entry number and symmetry map
    (sym_map)

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the SymEntry
    """
    def __init__(self, **kwargs):
        self._entries = {}

    def __call__(self, entry: int, sym_map: list[str] = None, **kwargs) -> SymEntry:
        """Return the specified SymEntry object singleton

        Args:
            entry: The entry number
            sym_map: The particular mapping of the symmetric groups
        Returns:
            The instance of the specified SymEntry
        """
        if sym_map is not None:
            sym_map_string = '|'.join('None' if sym is None else sym for sym in sym_map)
        else:
            sym_map_string = 'None'

        entry_key = f'{entry}|{sym_map_string}'
        symmetry = self._entries.get(entry_key)
        if symmetry:
            return symmetry
        else:
            self._entries[entry_key] = SymEntry(entry, sym_map=sym_map)
            return self._entries[entry_key]

    def get(self, entry: int, sym_map: list[str] = None, **kwargs) -> SymEntry:
        """Return the specified SymEntry object singleton

        Args:
            entry: The entry number
            sym_map: The particular mapping of the symmetric groups
        Returns:
            The instance of the specified SymEntry
        """
        return self.__call__(entry, sym_map, **kwargs)


symmetry_factory = SymEntryFactory()


def construct_uc_matrix(string_vector: list[str]) -> np.ndarray:
    """Calculate a matrix specifying the degrees of freedom in each dimension of the unit cell

    Args:
        string_vector: The string vector as parsed from the symmetry combination table
    Returns:
        3x3 float array with the values to specify unit cell dimensions from basis vector constraints
    """
    string_position = {'e': 0, 'f': 1, 'g': 2}
    variable_matrix = np.zeros((3, 3))  # default is float
    for col_idx, string in enumerate(string_vector):  # ex ['4*e', 'f', '0']
        if string[-1] != '0':
            row_idx = string_position[string[-1]]
            variable_matrix[row_idx][col_idx] = float(string.split('*')[0]) if '*' in string else 1.

            if '-' in string:
                variable_matrix[row_idx][col_idx] *= -1

    # for entry 6 - unit cell string_vector is ['4*e', '4*e', '4*e']
    #  [[4, 4, 4], [0, 0, 0], [0, 0, 0]]
    #  component1 string vector is ['0', 'e', '0']
    #   [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    #  component2 string vector is ['0', '0', '0']
    #   [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # for entry 85 - string_vector is ['4*e', '4*f', '4*g']
    #  [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    #  component1 string vector is ['0', '0', '0']
    #   [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    #  component2 string vector is ['e', 'f', 'g']
    #   [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return variable_matrix


# def construct_tx_dof_ref_frame_matrix(string_vector):
#     """
#
#     Args:
#         string_vector (list[str]):
#     Returns:
#         (numpy.ndarray): 3x3 array
#     """
#     string_position = {'e': 0, 'f': 1, 'g': 2}
#     variable_matrix = np.zeros((3, 3))
#     for col_idx, string in enumerate(string_vector):
#         if string[-1] != '0':
#             row_idx = string_position.get(string[-1])
#             variable_matrix[row_idx][col_idx] = float(string.split('*')[0]) if '*' in string else 1.
#
#             if '-' in string:
#                 variable_matrix[row_idx][col_idx] *= -1
#
#     return variable_matrix


def get_tx_dof_ref_frame_var_vec(string_vec, var):
    return_vec = [0.0, 0.0, 0.0]
    for i in range(3):
        if var in string_vec[i] and '*' in string_vec[i]:
            return_vec[i] = float(string_vec[i].split('*')[0])
        elif "-" + var in string_vec[i]:
            return_vec[i] = -1.0
        elif var == string_vec[i]:
            return_vec[i] = 1.0
    return return_vec


def parse_ref_tx_dof_str_to_list(ref_frame_tx_dof_string):
    return list(map(str.strip, ref_frame_tx_dof_string.strip('<>').split(',')))


def get_optimal_external_tx_vector(ref_frame_tx_dof, optimal_ext_dof_shifts):
    """From the DOF and the computed shifts, return the translation vector

    Args:
        ref_frame_tx_dof:
        optimal_ext_dof_shifts:

    Returns:
        (list[list[list]])
    """
    ext_dof_variables = ['e', 'f', 'g']

    parsed_ref_tx_vec = parse_ref_tx_dof_str_to_list(ref_frame_tx_dof)

    optimal_external_tx_vector = np.array([0.0, 0.0, 0.0])
    for idx, dof_shift in enumerate(optimal_ext_dof_shifts):
        var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_tx_vec, ext_dof_variables[idx])
        optimal_external_tx_vector += np.array(var_vec) * dof_shift

    return optimal_external_tx_vector.tolist()


def get_rot_matrices(step_deg: int | float, axis: str = 'z', rot_range_deg: int | float = 360) -> np.ndarray | None:
    """Return a group of rotation matrices to rotate coordinates about a specified axis in set step increments

    Args:
        step_deg: The number of degrees for each rotation step
        axis: The axis about which to rotate
        rot_range_deg: The range with which rotation is possible
    Returns:
        The rotation matrices with shape (rotations, 3, 3)
    """
    if rot_range_deg == 0:
        return None

    # Todo use scipy.Rotation to create these!
    rot_matrices = []
    axis = axis.lower()
    if axis == 'x':
        for step in range(0, int(rot_range_deg // step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[1, 0, 0], [0, math.cos(rad), -1 * math.sin(rad)], [0, math.sin(rad), math.cos(rad)]])
    elif axis == 'y':
        for step in range(0, int(rot_range_deg // step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-1 * math.sin(rad), 0, math.cos(rad)]])
    elif axis == 'z':
        for step in range(0, int(rot_range_deg // step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[math.cos(rad), -1 * math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
    else:
        raise ValueError(f'Axis "{axis}" is not supported')

    return np.array(rot_matrices)


def make_rotations_degenerate(rotations: np.ndarray | list[np.ndarray] | list[list[list[float]]] = None,
                              degeneracies: np.ndarray | list[np.ndarray] | list[list[list[float]]] = None) \
        -> np.ndarray:
    """From a set of degeneracy matrices and a set of rotation matrices, produce the complete combination of the
    specified transformations

    Args:
        rotations: A group of rotations with shape (rotations, 3, 3)
        degeneracies: A group of degeneracies with shape (degeneracies, 3, 3)
    Returns:
        The matrices resulting from the multiplication of each rotation by each degeneracy.
            Product has length = (rotations x degeneracies, 3, 3) where the first 3x3 array on axis 0 is the identity
    """
    if rotations is None:
        rotations = identity_matrix[None, :, :]  # expand to shape (1, 3, 3)
    elif np.all(rotations[0] == identity_matrix):
        pass  # This is correct
    else:
        logger.warning(f'{make_rotations_degenerate.__name__}: The argument "rotations" is missing an identity '
                       'matrix which is recommended to produce the correct matrices. Adding now')
        #                'Ensure you add this matrix to your degeneracies before calling')
        rotations = [identity_matrix] + list(rotations)

    if degeneracies is None:
        degeneracies = identity_matrix[None, :, :]  # expand to shape (1, 3, 3)
    elif np.all(degeneracies[0] == identity_matrix):
        pass  # This is correct
    else:
        logger.warning(f'{make_rotations_degenerate.__name__}: The argument "degeneracies" is missing an identity '
                       'matrix which is recommended to produce the correct matrices. Adding now')
        #                'Ensure you add this matrix to your degeneracies before calling')
        degeneracies = [identity_matrix] + list(degeneracies)

    return np.concatenate([np.matmul(rotations, degen_mat) for degen_mat in degeneracies])


def parse_uc_str_to_tuples(uc_string):
    """Acquire unit cell parameters from specified external degrees of freedom string"""
    def s_to_l(string):
        s1 = string.replace('(', '')
        s2 = s1.replace(')', '')
        l1 = s2.split(',')
        l2 = [x.replace(' ', '') for x in l1]
        return l2

    # if '),' in uc_string:
    #     l = uc_string.split('),')
    # else:
    #     l = [uc_string]

    return [s_to_l(s) for s in uc_string.split('), ')]


def get_uc_var_vec(string_vec, var):
    """From the length specification return the unit vector"""
    return_vec = [0.0, 0.0, 0.0]
    for i in range(len(string_vec)):
        if var in string_vec[i] and '*' in string_vec[i]:
            return_vec[i] = (float(string_vec[i].split('*')[0]))
        elif var == string_vec[i]:
            return_vec.append(1.0)
    return return_vec


def get_uc_dimensions(uc_string, e=1, f=0, g=0):
    """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
    basis vectors with component translation degrees of freedom"""
    string_vec_lens, string_vec_angles = parse_uc_str_to_tuples(uc_string)
    e_vec = get_uc_var_vec(string_vec_lens, 'e')
    f_vec = get_uc_var_vec(string_vec_lens, 'f')
    g_vec = get_uc_var_vec(string_vec_lens, 'g')
    e1 = [e_vec_val * e for e_vec_val in e_vec]
    f1 = [f_vec_val * f for f_vec_val in f_vec]
    g1 = [g_vec_val * g for g_vec_val in g_vec]

    lengths = [0.0, 0.0, 0.0]
    for i in range(len(string_vec_lens)):
        lengths[i] = abs((e1[i] + f1[i] + g1[i]))
    if len(string_vec_lens) == 2:
        lengths[2] = 1.0

    if len(string_vec_angles) == 1:
        angles = [90.0, 90.0, float(string_vec_angles[0])]
    else:
        angles = [0.0, 0.0, 0.0]
        for idx, string_vec_angle in enumerate(string_vec_angles):
            angles[idx] = float(string_vec_angle)

    return lengths + angles


highest_point_group_msg = f'If this is a point group. You likely need to modify the current highest cyclic symmetry ' \
                          f'{max_sym} in {PUtils.path_to_sym_utils}, then run the file using "python ' \
                          f'{PUtils.path_to_sym_utils}".'
example_symmetry_specification = 'RESULT:{SUBSYMMETRY}{SUBSYMMETRY}...'


def parse_symmetry_to_sym_entry(sym_entry: int = None, symmetry: str = None, sym_map: list[str] = None) -> \
        SymEntry | None:
    """Take a symmetry specified in a number of ways and return the symmetry parameters in a SymEntry

    Args:
        sym_entry: The integer corresponding to the desired SymEntry
        symmetry: The symmetry specified by a string
        sym_map: A symmetry map where each successive entry is the corresponding symmetry group number for the structure
    Returns:
        An instance of the SymEntry
    """
    if not sym_map:  # find sym_map from symmetry
        if symmetry:
            symmetry = symmetry.strip()
            if symmetry in space_group_symmetry_operators:  # ops keywords are Hermann-Mauguin notation
                # we only have the resulting symmetry, set it and then solve by lookup_sym_entry_by_symmetry_combination
                sym_map = [symmetry]
            elif len(symmetry) > 3:
                if ':{' in symmetry:  # we have a symmetry specification of typical type result:{subsymmetry}{}...
                    sym_map = [split.strip('}:') for split in symmetry.split('{')]
                elif 'cryst' in symmetry.lower():  # this is crystal specification
                    return  # we will have to set this up after parsing cryst records
                else:  # this is some Rosetta based symmetry?
                    sym_str1, sym_str2, sym_str3, *_ = symmetry
                    sym_map = f'{sym_str1} C{sym_str2} C{sym_str3}'.split()
                    logger.error(f'Symmetry specification "{symmetry}" is not understood, trying to solve anyway\n\n')
            elif symmetry in valid_symmetries:
                logger.critical(f'{parse_symmetry_to_sym_entry.__name__}: The functionality of passing symmetry as '
                                f'{symmetry} hasn\'t been tested thoroughly yet!')
                sym_map = [symmetry, symmetry, None]  # specified as [result, entity1, no other entities]
            elif len(symmetry) == 3 and symmetry[1].isdigit() and symmetry[2].isdigit():  # like I32, O43 format
                sym_map = [*symmetry]
            else:  # C35
                raise ValueError(f'{symmetry} is not a supported symmetry! {highest_point_group_msg}')
        else:
            raise \
                SymmetryError(f'{parse_symmetry_to_sym_entry.__name__}: Can\'t initialize without symmetry or sym_map!')

    if not sym_entry:
        try:
            sym_entry = dictionary_lookup(all_sym_entry_dict, sym_map)
            if not isinstance(sym_entry, int):
                raise TypeError
        except (KeyError, TypeError):  # when the entry is not specified in the all_sym_entry_dict
            # the prescribed symmetry is a point, plane, or space group that isn't in nanohedra. try a custom input
            sym_entry = lookup_sym_entry_by_symmetry_combination(*sym_map)

    return symmetry_factory.get(sym_entry, sym_map=sym_map)


def sdf_lookup(symmetry: str = None) -> AnyStr:
    """From the set of possible point groups, locate the proper symmetry definition file depending on the specified
    symmetry. If none is specified, a C1 symmetry will be returned (this doesn't make sense but is completely viable)

    Args:
        symmetry: Can be a valid_point_group, or None
    Returns:
        The location of the symmetry definition file on disk
    """
    if not symmetry:
        return os.path.join(PUtils.symmetry_def_files, 'C1.sym')
    else:
        symmetry = symmetry.upper()

    for file, ext in map(os.path.splitext, os.listdir(PUtils.symmetry_def_files)):
        if symmetry == file:
            return os.path.join(PUtils.symmetry_def_files, file + ext)

    raise DesignError('Couldn\'t locate correct symmetry definition file at "%s" for symmetry: %s'
                      % (PUtils.symmetry_def_files, symmetry))


repeat_with_sym_entry = f'Cannot distinguish between the desired entry. Please repeat your command, however, ' \
                        f'additionally specify the preferred Entry Number (ex: --{PUtils.sym_entry} 1) to proceed'
header_format_string = '{:5s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}'
query_output_format_string = '{:>5s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}'


def print_query_header():
    print(header_format_string.format("ENTRY", "GROUP1", "IntDofRot1", "IntDofTx1", "ReferenceFrameDof1", "GROUP2",
                                      "IntDofRot2", "IntDofTx2", "ReferenceFrameDof2", "RESULT"))


def lookup_sym_entry_by_symmetry_combination(result: str, *symmetry_operators: str) -> int:
    """Given the resulting symmetry and the symmetry operators for each Entity, solve for the SymEntry

    Args:
        result: The final symmetry
        symmetry_operators: Additional operators which specify sub-symmetric systems in the larger result
    Returns:
        The entry number of the SymEntry
    """

    def print_matching_entries(entries):
        if not entries:
            return
        print_query_header()
        for _entry in entries:
            _group1, _int_dof_group1, _, _ref_frame_tx_dof_group1, _group2, _int_dof_group2, _, \
                _ref_frame_tx_dof_group2, _, _, _, _, _, _ = symmetry_combinations[_entry]
            int_rot1, int_tx1, int_rot2, int_tx2 = 0, 0, 0, 0
            for int_dof in _int_dof_group1:
                if int_dof.startswith('r'):
                    int_rot1 = 1
                if int_dof.startswith('t'):
                    int_tx1 = 1
            for int_dof in _int_dof_group2:
                if int_dof.startswith('r'):
                    int_rot2 = 1
                if int_dof.startswith('t'):
                    int_tx2 = 1
            print(query_output_format_string.format(str(_entry), _group1, str(int_rot1), str(int_tx1),
                                                    _ref_frame_tx_dof_group1, _group2, str(int_rot2),
                                                    str(int_tx2), _ref_frame_tx_dof_group2, result))
    result = str(result)
    result_entries, matching_entries = [], []
    for entry_number, entry in symmetry_combinations.items():
        group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
            ref_frame_tx_dof_group2, _, resulting_symmetry, dimension, _, _, _ = entry
        if resulting_symmetry == result:
            result_entries.append(entry_number)
            # find all sub_symmetries that are viable in the component group members
            group1_members = sub_symmetries.get(group1, [None])
            # group1_members.extend('C2') if 'D' in group1 else None
            # group1_dihedral = True if 'D' in group1 else False
            group2_members = sub_symmetries.get(group2, [None])
            # group2_members.extend('C2') if 'D' in group2 else None
            # group2_dihedral = True if 'D' in group2 else False
            required_sym_operators = True  # assume correct until proven incorrect
            for sym_operator in symmetry_operators:
                if sym_operator in [resulting_symmetry, group1, group2]:
                    continue
                elif sym_operator in group1_members + group2_members:
                    continue
                else:
                    required_sym_operators = False

            if symmetry_operators and required_sym_operators:
                matching_entries.append(entry_number)  # TODO include the groups?

    if matching_entries:
        if len(matching_entries) == 1:
            return matching_entries[0]
        else:
            print('\033[1mFound multiple specified symmetries matching including %s\033[0m'
                  % (', '.join(map(str, matching_entries))))
            print_matching_entries(matching_entries)
            print(repeat_with_sym_entry)
            exit()
    elif symmetry_operators:
        raise ValueError('The specified symmetries "%s" could not be coerced to make the resulting symmetry "%s". '
                         'Try to reformat your symmetry specification if this is the result of a typo to include only '
                         'symmetries that are group members of the resulting symmetry such as %s\nUse the format %s '
                         'during your specification'
                         % (', '.join(symmetry_operators), result,
                            ', '.join(all_sym_entry_dict.get(result, {}).keys()), example_symmetry_specification))
    else:  # no symmetry_operators
        if result_entries:
            print('\033[1mFound specified symmetries matching including %s\033[0m'
                  % (', '.join(map(str, result_entries))))
            print_matching_entries(result_entries)
            print(repeat_with_sym_entry)
            exit()
        else:  # no matches
            raise ValueError(f'The resulting symmetry "{result}" didn\'t match any possible symmetry_combinations. You '
                             f'are likely requesting a symmetry that is outside of the parameterized SymEntry entries.'
                             f'If this is a chiral plane/space group, modify the function '
                             f'{lookup_sym_entry_by_symmetry_combination.__name__} to use the non Nanohedra compatible '
                             f'chiral space_group_symmetry_operators. {highest_point_group_msg}')


def query_combination(combination_list):
    if isinstance(combination_list, list) and len(combination_list) == 2:
        matching_entries = []
        for entry_number, entry in nanohedra_symmetry_combinations.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _, _, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if combination_list == [group1, group2] or combination_list == [group2, group1]:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                          str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                          str(int_rot2), str(int_tx2),
                                                                          ref_frame_tx_dof_group2, result))
        if not matching_entries:
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s + %s" % (combination_list[0], combination_list[1]) +
                  '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def query_result(desired_result):
    if isinstance(desired_result, str):
        matching_entries = []
        for entry_number, entry in nanohedra_symmetry_combinations.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _, _, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if desired_result == result:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                          str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                          str(int_rot2), str(int_tx2),
                                                                          ref_frame_tx_dof_group2, result))
        if not matching_entries:
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s" % desired_result + '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def query_counterpart(query_group):
    if isinstance(query_group, str):
        matching_entries = []
        for entry_number, entry in nanohedra_symmetry_combinations.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _, _, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if query_group in [group1, group2]:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(
                    "{:>5s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}".format(
                        str(entry_number), group1, str(int_rot1), str(int_tx1), ref_frame_tx_dof_group1, group2,
                        str(int_rot2), str(int_tx2), ref_frame_tx_dof_group2, result))
        if not matching_entries:
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s" % query_group + '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def all_entries():
    all_entries_list = []
    for entry_number, entry in nanohedra_symmetry_combinations.items():
        group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
            ref_frame_tx_dof_group2, _, result, dimension, _, _, _ = entry
        # group2 = entry[6]
        # int_dof_group1 = entry[3]
        # int_dof_group2 = entry[8]
        # ref_frame_tx_dof_group1 = entry[5]
        # ref_frame_tx_dof_group2 = entry[10]
        # result = entry[12]
        int_rot1 = 0
        int_tx1 = 0
        int_rot2 = 0
        int_tx2 = 0
        for int_dof in int_dof_group1:
            if int_dof.startswith('r'):
                int_rot1 = 1
            if int_dof.startswith('t'):
                int_tx1 = 1
        for int_dof in int_dof_group2:
            if int_dof.startswith('r'):
                int_rot2 = 1
            if int_dof.startswith('t'):
                int_tx2 = 1
        all_entries_list.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                  str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                  str(int_rot2), str(int_tx2), ref_frame_tx_dof_group2,
                                                                  result))
    print('\033[1m' + "ALL ENTRIES" + '\033[0m')
    print_query_header()
    for entry in all_entries_list:
        print(entry)


def dimension(dim):
    if dim in [0, 2, 3]:
        matching_entries_list = []
        for entry_number, entry in nanohedra_symmetry_combinations.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _, _, _ = entry
            # group1 = entry[1]
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if dimension == dim:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries_list.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                               str(int_tx1), ref_frame_tx_dof_group1,
                                                                               group2, str(int_rot2), str(int_tx2),
                                                                               ref_frame_tx_dof_group2, result))
        print('\033[1m' + 'ALL ENTRIES FOUND WITH DIMENSION %d: ' % dim + '\033[0m')
        print_query_header()
        for entry in matching_entries_list:
            print(entry)
    else:
        print('DIMENSION NOT SUPPORTED, VALID DIMENSIONS ARE: 0, 2 or 3')


if __name__ == '__main__':
    # To figure out if there is translation internally (on Z) then externally on Z - There indeed is, however, this
    # analysis doesn't take into consideration the setting matrices might counteract the internal translation so that
    # this translation only lies on another axis
    # double_translation = []
    # for entry in symmetry_combinations:
    #     sym_entry = symmetry_factory.get(entry)
    #     tx1, tx2 = False, False
    #     if sym_entry.ref_frame_tx_dof1[-1] != '0' and sym_entry.is_internal_tx1:
    #         tx1 = True
    #     if sym_entry.ref_frame_tx_dof2[-1] != '0' and sym_entry.is_internal_tx2:
    #         tx2 = True
    #     if tx1 or tx2:
    #         if tx1:
    #             entry_result = (entry, '1')
    #         if tx2:
    #             entry_result = (entry, '2')
    #         if tx1 and tx2:
    #             entry_result = (entry, '1+2')
    #         double_translation.append(entry_result)
    #
    # print('\n'.join('%d: %s' % tup for tup in double_translation))

    point_cloud_scale = 2
    # Oxy is X, N is Y, C is origin
    point_cloud = np.array([[point_cloud_scale, 0, 0], [0, point_cloud_scale, 0], [0, 0, 0]])
    # point_cloud += np.array([0, 0, 1])
    transformed_points = np.empty((len(setting_matrices), 3, 3))
    for idx, matrix in enumerate(setting_matrices.values()):
        transformed_points[idx] = np.matmul(point_cloud, np.transpose(matrix))

    atom_string = 'ATOM  %s {:^4s}{:1s}%s %s%s{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'
    alt_location = ''
    code_for_insertion = ''
    occ = 1
    temp_fact = 20.
    charge = ''

    atoms = format_guide_coords_as_atom(transformed_points.tolist())

    # add origin
    atom_idx = 1
    atoms.append(atom_string.format(format('C', '3s'), alt_location, code_for_insertion, occ, temp_fact,
                                    'C', charge)
                 % (format(atom_idx, '5d'), 'GLY', 'O', format(0, '4d'),
                    '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple([0, 0, 0]))))
    # add axis
    axis_length = 2 * point_cloud_scale
    axis_list = [0, 0, 0]
    axis_type = ['X', 'Y', 'Z']
    for axis_idx, _ in enumerate(axis_list):
        atom_idx += 1
        axis_point = axis_list.copy()
        axis_point[axis_idx] = axis_length
        atoms.append(atom_string.format(format('C', '3s'), alt_location, code_for_insertion, occ, temp_fact,
                                        'C', charge)
                     % (format(atom_idx, '5d'), 'GLY', axis_type[axis_idx], format(axis_idx + 1, '4d'),
                        '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple(axis_point))))

    # write to file
    with open(os.path.join(os.getcwd(), 'setting_matrix_points.pdb'), 'w') as f:
        f.write('%s\n' % '\n'.join(atoms))
