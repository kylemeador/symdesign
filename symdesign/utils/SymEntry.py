from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from typing import AnyStr, Iterable, Literal, get_args

import numpy as np

from symdesign import resources, utils
from symdesign.utils import path as putils
from symdesign.utils.symmetry import valid_subunit_number, space_group_symmetry_operators, \
    point_group_symmetry_operators, all_sym_entry_dict, rotation_range, setting_matrices, identity_matrix, \
    sub_symmetries, flip_y_matrix, MAX_SYMMETRY, valid_symmetries

__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

logger = logging.getLogger(__name__)
symmetry_combination_format = 'ResultingSymmetry:{Component1Symmetry}{Component2Symmetry}{...}'
# SYMMETRY COMBINATION MATERIAL TABLE (T.O.Y and J.L, 2020)
# Guide to table interpretation
# Standard T:{C3}{C3}
# Number   grp1             grp1_internal_dof grp1_set_mat grp1_external_dof
# 54:     ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'],          4,             None,
#          grp2             grp2_internal_dof grp2_set_mat grp2_external_dof
#          'C3', ['r:<0,0,1,c>', 't:<0,0,d>'],         12,             None,
#          pnt_grp final_sym dim  unit_cell tot_dof ring_size
#          'T',         'T',  0,       None,      4,       2],
# Modified T:{C3}{C3} with group 1 internal DOF allowed, group 2, internal DOF disabled
# 54:     ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'],  4, None,
#          'C3',                           [], 12, None,
#          'T', 'T', 0, None, None, 4, 2],
nanohedra_symmetry_combinations = {
    1: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'C2', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D2', 'D2', 0, None, None, 4, 2],
    2: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'C3', ['r:<0,0,1,c>'], 1, 'e,0.577350*e,0', 'C6', 'p6', 2, ('2*e', '2*e'), (120,), 4, 6],
    3: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D3', 'D3', 0, None, None, 4, 2],
    4: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, 'e,0,0', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D3', 'p312', 2, ('2*e', '2*e'), (120,), 5, 6],
    5: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, None, 'T', 'T', 0, None, None, 4, 3],
    6: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '0,e,0', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, None, 'T', 'I213', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 5, 10],
    7: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, None, 'O', 'O', 0, None, None, 4, 4],
    8: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '2*e,e,0', 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 4, None, 'O', 'P4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 5, 10],
    9: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 7, None, 'I', 'I', 0, None, None, 4, 5],
    10: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'C4', ['r:<0,0,1,c>'], 1, None, 'C4', 'p4', 2, ('2*e', '2*e'), (90,), 4, 4],
    11: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D4', 'D4', 0, None, None, 4, 2],
    12: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, 'e,0,0', 'D4', 'p4212', 2, ('2*e', '2*e'), (90,), 5, 4],
    13: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'O', 'O', 0, None, None, 4, 3],
    14: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '2*e,e,0', 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 5, 8],
    15: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D5', 'D5', 0, None, None, 4, 2],
    16: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 9, None, 'I', 'I', 0, None, None, 4, 3],
    17: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'C6', ['r:<0,0,1,c>'], 1, None, 'C6', 'p6', 2, ('2*e', '2*e'), (120,), 4, 3],
    18: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'C6', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D6', 'D6', 0, None, None, 4, 2],
    19: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, 'e,0,0', 'C6', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 5, 4],
    20: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,f,0', 'D2', [], 1, None, 'D2', 'c222', 2, ('4*e', '4*f'), (90,), 4, 4],
    21: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, None, 'D2', [], 1, 'e,0,0', 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 3, 4],
    22: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '0,e,f', 'D2', [], 5, None, 'D4', 'I4122', 3, ('4*e', '4*e', '8*f'), (90, 90, 90), 4, 6],
    23: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 10, None, 'D2', [], 1, 'e,0,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 3],
    24: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 10, '0,0,e', 'D2', [], 1, 'f,0,0', 'D6', 'P6222', 3, ('2*f', '2*f', '6*e'), (90, 90, 120), 4, 6],
    25: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'D2', [], 5, '2*e,0,e', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 4],
    26: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '-2*e,3*e,0', 'D2', [], 5, '0,2*e,e', 'O', 'I4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 3],
    27: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, 'e,0,0', 'D3', [], 11, None, 'D3', 'p312', 2, ('2*e', '2*e'), (120,), 3, 3],
    28: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '0,e,f', 'D3', [], 1, None, 'D3', 'R32', 3, ('3.4641*e', '3.4641*e', '3*f'), (90, 90, 120), 4, 4],
    29: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'D3', [], 11, 'e,0.57735*e,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    30: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D3', [], 11, 'e,0.57735*e,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    31: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D3', [], 11, 'e,0.57735*e,f', 'D6', 'P6322', 3, ('2*e', '2*e', '4*f'), (90, 90, 120), 4, 4],
    32: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D3', [], 4, 'e,e,e', 'O', 'F4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 3],
    33: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '0,2*e,0', 'D3', [], 4, 'e,e,e', 'O', 'I4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 2],
    34: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'D3', [], 4, 'e,e,e', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 4],
    35: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '0,e,-2*e', 'D3', [], 4, 'e,e,e', 'O', 'I4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 2],
    36: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '0,e,-2*e', 'D3', [], 4, '3*e,3*e,3*e', 'O', 'P4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 3],
    37: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'D4', [], 1, None, 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 3, 2],
    38: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '0,e,0', 'D4', [], 1, None, 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 3, 2],
    39: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '0,e,f', 'D4', [], 1, None, 'D4', 'I422', 3, ('2*e', '2*e', '4*f'), (90, 90, 90), 4, 4],
    40: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'D4', [], 1, '0,0,e', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 3],
    41: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '2*e,e,0', 'D4', [], 1, '2*e,2*e,0', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    42: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'D6', [], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    43: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, 'e,0,0', 'D6', [], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    44: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 6, 'e,0,f', 'D6', [], 1, None, 'D6', 'P622', 3, ('2*e', '2*e', '2*f'), (90, 90, 120), 4, 4],
    45: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'T', [], 1, None, 'T', 'P23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    46: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,e,0', 'T', [], 1, None, 'T', 'F23', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 3],
    47: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '2*e,3*e,0', 'T', [], 1, '0,4*e,0', 'O', 'F4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 2],
    48: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'O', [], 1, None, 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    49: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,e,0', 'O', [], 1, None, 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    50: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, 'e,0,0', 'O', [], 1, None, 'O', 'F432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    51: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '0,e,0', 'O', [], 1, None, 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    52: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '-e,e,e', 'O', [], 1, None, 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    53: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C3', ['r:<0,0,1,c>'], 1, 'e,0.57735*e,0', 'C3', 'p3', 2, ('2*e', '2*e'), (120,), 4, 3],
    54: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 12, None, 'T', 'T', 0, None, None, 4, 2],
    55: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'C3', ['r:<0,0,1,c>', 't:<0,0,d>'], 12, 'e,0,0', 'T', 'P213', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 5, 5],
    56: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, None, 'O', 'O', 0, None, None, 4, 2],
    57: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 1, 'e,0,0', 'O', 'F432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 5, 6],
    58: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 7, None, 'C5', ['r:<0,0,1,c>', 't:<0,0,d>'], 9, None, 'I', 'I', 0, None, None, 4, 2],
    59: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0.57735*e,0', 'C6', ['r:<0,0,1,c>'], 1, None, 'C6', 'p6', 2, ('2*e', '2*e'), (120,), 4, 2],
    60: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0.57735*e,0', 'D2', [], 1, 'e,0,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    61: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'D2', [], 1, 'e,0,0', 'T', 'P23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 3],
    62: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'D2', [], 3, 'e,0,e', 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 3],
    63: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'D2', [], 3, '2*e,e,0', 'O', 'I4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 2],
    64: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0.57735*e,0', 'D3', [], 11, None, 'D3', 'p312', 2, ('2*e', '2*e'), (120,), 3, 2],
    65: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0.57735*e,0', 'D3', [], 1, None, 'D3', 'p321', 2, ('2*e', '2*e'), (120,), 3, 2],
    66: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 12, '4*e,0,0', 'D3', [], 4, '3*e,3*e,3*e', 'O', 'P4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 3, 4],
    67: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'D4', [], 1, '0,0,e', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    68: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0.57735*e,0', 'D6', [], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    69: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, 'e,0,0', 'T', [], 1, None, 'T', 'F23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    70: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, 'e,0,0', 'O', [], 1, None, 'O', 'F432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    71: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C4', ['r:<0,0,1,c>'], 1, 'e,e,0', 'C4', 'p4', 2, ('2*e', '2*e'), (90,), 4, 2],
    72: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'C4', ['r:<0,0,1,c>', 't:<0,0,d>'], 2, '0,e,e', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 5, 4],
    73: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D2', [], 1, 'e,0,0', 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 3, 2],
    74: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,0,0', 'D2', [], 5, None, 'D4', 'p4212', 2, ('2*e', '2*e'), (90,), 3, 2],
    75: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D2', [], 3, '2*e,e,0', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    76: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D2', [], 3, 'e,0,e', 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 3],
    77: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D3', [], 4, 'e,e,e', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    78: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,e,0', 'D4', [], 1, None, 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 3, 2],
    79: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D4', [], 1, 'e,e,0', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    80: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'T', [], 1, 'e,e,e', 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 3, 2],
    81: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, 'e,e,0', 'O', [], 1, None, 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 3, 2],
    82: ['C6', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D2', [], 1, 'e,0,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 3, 2],
    83: ['C6', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D3', [], 11, 'e,0.57735*e,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 2, 2],
    84: ['D2', [], 1, None, 'D2', [], 1, 'e,f,0', 'D2', 'p222', 2, ('2*e', '2*f'), (90,), 2, 2],
    85: ['D2', [], 1, None, 'D2', [], 1, 'e,f,g', 'D2', 'F222', 3, ('4*e', '4*f, 4*g'), (90, 90, 90), 3, 3],
    86: ['D2', [], 1, 'e,0,0', 'D2', [], 5, '0,0,f', 'D4', 'P4222', 3, ('2*e', '2*e', '4*f'), (90, 90, 90), 2, 2],
    87: ['D2', [], 1, 'e,0,0', 'D2', [], 13, '0,0,-f', 'D6', 'P6222', 3, ('2*e', '2*e', '6*f'), (90, 90, 120), 2, 2],
    88: ['D2', [], 3, '0,e,2*e', 'D2', [], 5, '0,2*e,e', 'O', 'P4232', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    89: ['D2', [], 1, 'e,0,0', 'D3', [], 11, 'e,0.57735*e,0', 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 1, 1],
    90: ['D2', [], 1, 'e,0,0', 'D3', [], 11, 'e,0.57735*e,f', 'D6', 'P622', 3, ('2*e', '2*e', '2*f'), (90, 90, 120), 2, 2],
    91: ['D2', [], 1, '0,0,2*e', 'D3', [], 4, 'e,e,e', 'D6', 'P4232', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    92: ['D2', [], 3, '2*e,e,0', 'D3', [], 4, 'e,e,e', 'O', 'I4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 1, 1],
    93: ['D2', [], 1, 'e,0,0', 'D4', [], 1, None, 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 1, 1],
    94: ['D2', [], 1, 'e,0,f', 'D4', [], 1, None, 'D4', 'P422', 3, ('2*e', '2*e', '2*f'), (90, 90, 90), 2, 2],
    95: ['D2', [], 5, 'e,0,f', 'D4', [], 1, None, 'D4', 'I422', 3, ('2*e', '2*e', '4*f'), (90, 90, 90), 2, 2],
    96: ['D2', [], 3, '0,e,2*e', 'D4', [], 1, '0,0,2*e', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    97: ['D2', [], 1, 'e,0,0', 'D6', [], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 1, 1],
    98: ['D2', [], 1, 'e,0,f', 'D6', [], 1, None, 'D6', 'P622', 3, ('2*e', '2*e', '2*f'), (90, 90, 120), 2, 2],
    99: ['D2', [], 1, 'e,0,0', 'T', [], 1, None, 'T', 'P23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    100: ['D2', [], 1, 'e,e,0', 'T', [], 1, None, 'T', 'P23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 2],
    101: ['D2', [], 3, 'e,0,e', 'T', [], 1, 'e,e,e', 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    102: ['D2', [], 3, '2*e,e,0', 'T', [], 1, None, 'O', 'P4232', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    103: ['D2', [], 3, 'e,0,e', 'O', [], 1, None, 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    104: ['D2', [], 3, '2*e,e,0', 'O', [], 1, None, 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    105: ['D3', [], 11, None, 'D3', [], 11, 'e,0.57735*e,0', 'D3', 'p312', 2, ('2*e', '2*e'), (120,), 1, 1],
    106: ['D3', [], 11, None, 'D3', [], 11, 'e,0.57735*e,f', 'D3', 'P312', 3, ('2*e', '2*e', '2*f'), (90, 90, 120), 2, 2],
    107: ['D3', [], 1, None, 'D3', [], 11, 'e,0.57735*e,f', 'D6', 'P6322', 3, ('2*e', '2*e', '4*f'), (90, 90, 120), 2, 2],
    108: ['D3', [], 4, 'e,e,e', 'D3', [], 12, 'e,3*e,e', 'O', 'P4232', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    109: ['D3', [], 4, '3*e,3*e,3*e', 'D3', [], 12, 'e,3*e,5*e', 'O', 'P4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 1, 1],
    110: ['D3', [], 4, 'e,e,e', 'D4', [], 1, '0,0,2*e', 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 2],
    111: ['D3', [], 11, 'e,0.57735*e,0', 'D6', [], 1, None, 'D6', 'p622', 2, ('2*e', '2*e'), (120,), 1, 1],
    112: ['D3', [], 11, 'e,0.57735*e,f', 'D6', [], 1, None, 'D6', 'P622', 3, ('2*e', '2*e', '2*f'), (90, 90, 120), 2, 2],
    113: ['D3', [], 4, 'e,e,e', 'T', [], 1, None, 'O', 'F4132', 3, ('8*e', '8*e', '8*e'), (90, 90, 90), 1, 1],
    114: ['D3', [], 4, 'e,e,e', 'O', [], 1, None, 'O', 'I432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    115: ['D4', [], 1, None, 'D4', [], 1, 'e,e,0', 'D4', 'p422', 2, ('2*e', '2*e'), (90,), 1, 1],
    116: ['D4', [], 1, None, 'D4', [], 1, 'e,e,f', 'D4', 'P422', 3, ('2*e', '2*e', '2*f'), (90, 90, 90), 2, 2],
    117: ['D4', [], 1, '0,0,e', 'D4', [], 2, '0,e,e', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    118: ['D4', [], 1, '0,0,e', 'O', [], 1, None, 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    119: ['D4', [], 1, 'e,e,0', 'O', [], 1, None, 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    120: ['T', [], 1, None, 'T', [], 1, 'e,e,e', 'T', 'F23', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    121: ['T', [], 1, None, 'T', [], 1, 'e,0,0', 'T', 'F23', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    122: ['T', [], 1, 'e,e,e', 'O', [], 1, None, 'O', 'F432', 3, ('4*e', '4*e', '4*e'), (90, 90, 90), 1, 1],
    123: ['O', [], 1, None, 'O', [], 1, 'e,e,e', 'O', 'P432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
    124: ['O', [], 1, None, 'O', [], 1, 'e,0,0', 'O', 'F432', 3, ('2*e', '2*e', '2*e'), (90, 90, 90), 1, 1],
}
# KM Custom entries
symmetry_combinations = {
    # C1 alone
    125: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, None, [], 1, None, 'C1', 'C1', 0, None, None, 0, 1],
    126: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None,
          'C1', 'C1', 0, None, None, 6, 1],
    # C2 alone
    130: ['C2', [], 1, None, None, [], 1, None, 'C2', 'C2', 0, None, None, 0, 1],
    131: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C2', [], 1, None, 'C2', 'C2', 0, None, None, 6, 1],
    # C3 alone
    135: ['C3', [], 1, None, None, [], 1, None, 'C3', 'C3', 0, None, None, 0, 1],
    136: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C3', [], 1, None, 'C3', 'C3', 0, None, None, 6, 1],
    # C4 alone
    140: ['C4', [], 1, None, None, [], 1, None, 'C4', 'C4', 0, None, None, 0, 1],
    141: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C4', [], 1, None, 'C4', 'C4', 0, None, None, 6, 1],
    # C5 alone
    145: ['C5', [], 1, None, None, [], 1, None, 'C5', 'C5', 0, None, None, 0, 1],
    146: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C5', [], 1, None, 'C5', 'C5', 0, None, None, 6, 1],
    # C6 alone
    150: ['C6', [], 1, None, None, [], 1, None, 'C6', 'C6', 0, None, None, 0, 1],
    151: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'C6', [], 1, None, 'C6', 'C6', 0, None, None, 6, 1],
    # D2 alone
    155: ['D2', [], 1, None, None, [], 1, None, 'D2', 'D2', 0, None, None, 0, 1],
    156: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'D2', [], 1, None, 'D2', 'D2', 0, None, None, 6, 1],
    157: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D2', [], 1, None, 'D2', 'D2', 0, None, None, 2, 1],
    158: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D2', [], 1, None, 'D2', 'D2', 0, None, None, 2, 1],
    # D3 alone
    160: ['D3', [], 1, None, None, [], 1, None, 'D3', 'D3', 0, None, None, 0, 1],
    161: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'D3', [], 1, None, 'D3', 'D3', 0, None, None, 6, 1],
    162: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D3', [], 1, None, 'D3', 'D3', 0, None, None, 2, 1],
    163: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D3', [], 1, None, 'D3', 'D3', 0, None, None, 2, 1],
    # D4 alone
    165: ['D4', [], 1, None, None, [], 1, None, 'D4', 'D4', 0, None, None, 0, 1],
    166: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'D4', [], 1, None, 'D4', 'D4', 0, None, None, 6, 1],
    167: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D4', [], 1, None, 'D4', 'D4', 0, None, None, 2, 1],
    168: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 2, None, 'D4', [], 1, None, 'D4', 'D4', 0, None, None, 2, 1],
    169: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'D4', [], 1, None, 'D4', 'D4', 0, None, None, 2, 1],
    # D5 alone
    170: ['D5', [], 1, None, None, [], 1, None, 'D5', 'D5', 0, None, None, 0, 1],
    171: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'D5', [], 1, None, 'D5', 'D5', 0, None, None, 6, 1],
    # D6 alone
    175: ['D6', [], 1, None, None, [], 1, None, 'D6', 'D6', 0, None, None, 0, 1],
    176: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'D6', [], 1, None, 'D6', 'D6', 0, None, None, 6, 1],
    # T alone
    200: ['T', [], 1, None, None, [], 1, None, 'T', 'T', 0, None, None, 0, 1],
    201: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'T', [], 1, None, 'T', 'T', 0, None, None, 6, 1],
    202: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'T', [], 1, None, 'T', 'T', 0, None, None, 2, 1],
    203: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'T', [], 1, None, 'T', 'T', 0, None, None, 2, 1],
    # O alone
    210: ['O', [], 1, None, None, [], 1, None, 'O', 'O', 0, None, None, 0, 1],
    211: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'O', [], 1, None, 'O', 'O', 0, None, None, 6, 1],
    212: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, None, 'O', [], 1, None, 'O', 'O', 0, None, None, 2, 1],
    213: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, None, 'O', [], 1, None, 'O', 'O', 0, None, None, 2, 1],
    214: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'O', [], 1, None, 'O', 'O', 0, None, None, 2, 1],
    # I alone
    220: ['I', [], 1, None, None, [], 1, None, 'I', 'I', 0, None, None, 0, 1],
    221: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None, 'I', [], 1, None, 'I', 'I', 0, None, None, 6, 1],
    222: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, None, 'I', [], 1, None, 'I', 'I', 0, None, None, 2, 1],
    223: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 7, None, 'I', [], 1, None, 'I', 'I', 0, None, None, 2, 1],
    224: ['C5', ['r:<0,0,1,a>', 't:<0,0,b>'], 9, None, 'I', [], 1, None, 'I', 'I', 0, None, None, 2, 1],
    # KM 3 component entries
    # 301: {'components': [{'symmetry': 'C1', 'dof_internal': ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C2', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C3', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 4, 'dof_external': '<0,0,0>'}]
    #                       , 'result': ['T', 'T', 0, None, None, 1, 1]},
}

custom_entries = list(symmetry_combinations.keys())
symmetry_combinations.update(nanohedra_symmetry_combinations)
# Reformat the symmetry_combinations to account for groups and results separately
parsed_symmetry_combinations: dict[int, tuple[list[tuple[str, list | int | str]], list[str | int]]] = \
    {entry_number: ([(entry[0], entry[1:4]), (entry[4], entry[5:8])], entry[-7:])
     for entry_number, entry in symmetry_combinations.items()}
# Set the special CRYST1 Record symmetry combination
parsed_symmetry_combinations[0] = ([], [])
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
    group_1, _, setting_1, _, group_2, _, setting_2, _, point_group, *_ = ent
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


def construct_uc_matrix(string_vector: Iterable[str]) -> np.ndarray:
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


class SymEntry:
    __external_dof: list[np.ndarray]
    _cryst_record: str | None
    _degeneracy_matrices: np.ndarray | None
    _external_dof: np.ndarray
    _group_subunit_numbers: list[int]
    _int_dof_groups: list[str]
    _number_dof_external: int
    _number_dof_rotation: int
    _number_dof_translation: int
    _ref_frame_tx_dof: list[list[str] | None]
    _rotation_range: list[int]
    _setting_matrices: list[np.ndarray]
    _setting_matrices_numbers: list[int]
    _uc_dimensions: tuple[float, float, float, float, float, float] | None
    cycle_size: int
    dimension: int
    number: int
    groups: list[str]
    point_group_symmetry: str
    resulting_symmetry: str
    sym_map: list[str]
    total_dof: int
    unit_cell: tuple[tuple[str], tuple[int]] | None
    cell_lengths: tuple[str] | None
    cell_angles: tuple[int] | None
    expand_matrices: np.ndarray

    @classmethod
    def from_cryst(cls, space_group: str, **kwargs):  # uc_dimensions: Iterable[float],
        """Create a SymEntry from a specified symmetry in Hermann-Mauguin notation and the unit-cell dimensions"""
        return cls(0, space_group=space_group, **kwargs)

    def __init__(self, entry: int, sym_map: list[str] = None, **kwargs):
        try:
            self.group_info, result_info = parsed_symmetry_combinations[entry]
            # group_info, result_info = parsed_symmetry_combinations[entry]
            # returns
            #  {'group1': [self.int_dof_group1, self.rot_set_group1, self.ref_frame_tx_dof1],
            #   'group2': [self.int_dof_group2, self.rot_set_group2, self.ref_frame_tx_dof2],
            #   ...},
            #  [point_group_symmetry, resulting_symmetry, dimension, unit_cell, tot_dof, cycle_size]
        except KeyError:
            raise ValueError(
                f"Invalid symmetry entry '{entry}'. Supported values are Nanohedra entries: "
                f'{1}-{len(nanohedra_symmetry_combinations)} and custom entries: '
                f'{", ".join(map(str, custom_entries))}')

        try:  # To unpack the result_info. This will fail if a CRYST1 record placeholder
            self.point_group_symmetry, self.resulting_symmetry, self.dimension, self.cell_lengths, self.cell_angles, \
                self.total_dof, self.cycle_size = result_info
        except ValueError:  # Not enough values to unpack, probably a CRYST token
            # Todo - Crystallographic symmetry could coincide with group symmetry...
            self.group_info = [('C1', [['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None])]  # Assume for now that the groups are C1
            # group_info = [('C1', [['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, None])]  # Assume for now that the groups are C1
            self.point_group_symmetry = None
            # self.resulting_symmetry = kwargs.get('resulting_symmetry', None)
            if 'space_group' in kwargs:
                self.resulting_symmetry = kwargs['space_group']
            elif sym_map is None:
                # self.resulting_symmetry = None
                raise utils.SymmetryInputError(
                    f"Can't create a {self.__class__.__name__} without passing 'space_group' or 'sym_map'")
            else:
                self.resulting_symmetry, *_ = sym_map
            self.dimension = 2 if self.resulting_symmetry in utils.symmetry.layer_group_cryst1_fmt_dict else 3
            self.cell_lengths = self.cell_angles = None
            self.total_dof = self.cycle_size = 0

        self._int_dof_groups, self._setting_matrices, self._setting_matrices_numbers, self._ref_frame_tx_dof, \
            self.__external_dof = [], [], [], [], []
        self.number = entry
        self.entry_groups = [group_name for group_name, group_params in self.group_info if group_name]  # Ensure not None
        # group1, group2, *extra = entry_groups
        if sym_map is None:  # Assume standard SymEntry
            # Assumes 2 component symmetry. index with only 2 options
            self.sym_map = [self.resulting_symmetry] + self.entry_groups
            groups = self.entry_groups
        else:  # Requires full specification of all symmetry groups
            self.sym_map = sym_map
            result, *groups = sym_map  # Remove the result and pass the groups

        # Solve the group information for each passed symmetry
        self.groups = []
        for idx, group in enumerate(groups, 1):
            self.append_group(group)
            # self.groups = []
            # for idx, group in enumerate(groups, 1):
            #     self.append_group(group)
            #     # if group not in valid_symmetries:
            #     #     if group is None:
            #     #         # Todo
            #     #         #  Need to refactor symmetry_combinations for any number of elements
            #     #         continue
            #     #     else:  # Recurse to see if it is yet another symmetry specification
            #     #         # raise ValueError(
            #     #         logger.warning(
            #     #             f"The symmetry group '{group}' specified at index '{idx}' isn't a valid sub-symmetry. "
            #     #             f"Trying to correct by applying another {self.__class__.__name__}()")
            #     #         raise NotImplementedError()
            #     #
            #     # if group not in entry_groups:
            #     #     # This is probably a sub-symmetry of one of the groups. Is it allowed?
            #     #     if not symmetry_groups_are_allowed_in_entry(groups, *entry_groups, result=self.resulting_symmetry):
            #     #                                                 # group1=group1, group2=group2):
            #     #         viable_groups = [group for group in entry_groups if group is not None]
            #     #         raise utils.SymmetryInputError(
            #     #             f"The symmetry group '{group}' isn't an allowed sub-symmetry of the result "
            #     #             f'{self.resulting_symmetry}, or the group(s) {", ".join(viable_groups)}')
            #     # self.groups.append(group)

        # def add_group():
        #     # Todo
        #     #  Can the accuracy of this creation method be guaranteed with the usage of the same symmetry
        #     #  operator and different orientations? Think T33
        #     self._int_dof_groups.append(int_dof)
        #     self._setting_matrices.append(setting_matrices[set_mat_number])
        #     self._setting_matrices_numbers.append(set_mat_number)
        #     if ext_dof is None:
        #         self._ref_frame_tx_dof.append(ext_dof)
        #         self.__external_dof.append(construct_uc_matrix(('0', '0', '0')))
        #     else:
        #         ref_frame_tx_dof = ext_dof.split(',')
        #         self._ref_frame_tx_dof.append(ref_frame_tx_dof)
        #         if group_idx <= 2:
        #             # This isn't possible with more than 2 groups unless the groups is tethered to existing
        #             self.__external_dof.append(construct_uc_matrix(ref_frame_tx_dof))
        #         else:
        #             if ref_frame_tx_dof:
        #                 raise utils.SymmetryInputError(
        #                     f"Can't create {self.__class__.__name__} with external degrees of freedom and > 2 groups")
        #
        # for group_idx, group_symmetry in enumerate(self.groups, 1):
        #     if isinstance(group_symmetry, SymEntry):
        #         group_symmetry = group_symmetry.resulting_symmetry
        #     # for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in self.group_info:
        #     for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in group_info:
        #         if group_symmetry == entry_group_symmetry:
        #             add_group()
        #             break
        #     else:  # None was found for this group_symmetry
        #         # raise utils.SymmetryInputError(
        #         logger.critical(
        #             f"Trying to assign the group '{group_symmetry}' at index {group_idx} to "
        #             f"{self.__class__.__name__}.number={self.number}")
        #         # See if the group is a sub-symmetry of a known group
        #         # for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in self.group_info:
        #         for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in group_info:
        #             entry_sub_groups = sub_symmetries.get(entry_group_symmetry, [None])
        #             if group_symmetry in entry_sub_groups:
        #                 add_group()
        #                 break
        #         else:
        #             raise utils.SymmetryInputError(
        #                 f"Assignment of the group '{group_symmetry}' failed")

        # Check construction is valid
        if self.point_group_symmetry not in valid_symmetries:
            if not self.is_cryst_record():  # Anything besides CRYST entry
                raise utils.SymmetryInputError(
                    f'Invalid point group symmetry {self.point_group_symmetry}')
        try:
            if self.dimension == 0:
                self.expand_matrices = point_group_symmetry_operators[self.resulting_symmetry]
            elif self.dimension in [2, 3]:
                self.expand_matrices, expand_translations = space_group_symmetry_operators[self.resulting_symmetry]
            else:
                raise utils.SymmetryInputError(
                    'Invalid symmetry entry. Supported dimensions are 0, 2, and 3')
        except KeyError:
            raise utils.SymmetryInputError(
                f"The symmetry result '{self.resulting_symmetry}' isn't allowed as there aren't group operators "
                "available for it")

        if self.cell_lengths:
            self.unit_cell = (self.cell_lengths, self.cell_angles)
        else:
            self.unit_cell = None

    def append_group(self, group: str):
        """Add an additional symmetry group to the SymEntry"""
        if group not in valid_symmetries:
            if group is None:
                # Todo
                #  Need to refactor symmetry_combinations for any number of elements
                return
            else:  # Recurse to see if it is yet another symmetry specification
                # raise ValueError(
                logger.warning(
                    f"The symmetry group '{group}' at index {len(self.groups)} isn't a valid sub-symmetry. "
                    f"Trying to correct by applying another {self.__class__.__name__}()")
                raise NotImplementedError()

        if group not in self.entry_groups:
            # This is probably a sub-symmetry of one of the groups. Is it allowed?
            if not symmetry_groups_are_allowed_in_entry(groups, *self.entry_groups, result=self.resulting_symmetry):
                # group1=group1, group2=group2):
                viable_groups = [group for group in self.entry_groups if group is not None]
                raise utils.SymmetryInputError(
                    f"The symmetry group '{group}' isn't an allowed sub-symmetry of the result "
                    f'{self.resulting_symmetry}, or the group(s) {", ".join(viable_groups)}')
        self.groups.append(group)

        def add_group():
            # Todo
            #  Can the accuracy of this creation method be guaranteed with the usage of the same symmetry
            #  operator and different orientations? Think T33
            self._int_dof_groups.append(int_dof)
            self._setting_matrices.append(setting_matrices[set_mat_number])
            self._setting_matrices_numbers.append(set_mat_number)
            if ext_dof is None:
                self._ref_frame_tx_dof.append(ext_dof)
                self.__external_dof.append(construct_uc_matrix(('0', '0', '0')))
            else:
                ref_frame_tx_dof = ext_dof.split(',')
                self._ref_frame_tx_dof.append(ref_frame_tx_dof)
                if len(self.groups) <= 2:
                    # This isn't possible with more than 2 groups unless the groups is tethered to existing
                    self.__external_dof.append(construct_uc_matrix(ref_frame_tx_dof))
                else:
                    if ref_frame_tx_dof:
                        raise utils.SymmetryInputError(
                            f"Can't create {self.__class__.__name__} with external degrees of freedom and > 2 groups")

        if isinstance(group, SymEntry):
            group = group.resulting_symmetry
        for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in self.group_info:
            if group == entry_group_symmetry:
                add_group()
                break
        else:  # None was found for this group
            # raise utils.SymmetryInputError(
            logger.critical(
                f"Trying to assign the group '{group}' at index {len(self.groups)} to "
                f"{self.__class__.__name__}.number={self.number}")
            # See if the group is a sub-symmetry of a known group
            for entry_group_symmetry, (int_dof, set_mat_number, ext_dof) in self.group_info:
                entry_sub_groups = sub_symmetries.get(entry_group_symmetry, [None])
                if group in entry_sub_groups:
                    add_group()
                    break
            else:
                raise utils.SymmetryInputError(
                    f"Assignment of the group '{group}' failed")

    @property
    def number_of_operations(self) -> int:
        """The number of symmetric copies in the full symmetric system"""
        return len(self.expand_matrices)

    @property
    def group_subunit_numbers(self) -> list[int]:
        """Returns the number of subunits for each symmetry group"""
        try:
            return self._group_subunit_numbers
        except AttributeError:
            self._group_subunit_numbers = [valid_subunit_number[group] for group in self.groups]
            return self._group_subunit_numbers

    @property
    def specification(self) -> str:
        """Return the specification for the instance. Ex: RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}... -> (T:{C3}{C3})"""
        return '%s:{%s}' % (self.resulting_symmetry, '}{'.join(self.groups))

    @property
    def simple_specification(self) -> str:
        """Return the simple specification for the instance. Ex: 'RESULTSUBSYMMETRY1SUBSYMMETRY2... -> (T33)"""
        return f'{self.resulting_symmetry}{"".join(self.groups)}'

    @property
    def uc_specification(self) -> tuple[tuple[str] | None, tuple[int] | None]:
        """The external dof and angle parameters which constitute a viable lattice"""
        return self.cell_lengths, self.cell_angles

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
    def number_of_groups(self) -> int:
        return len(self.groups)

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
    def setting_matrices_numbers(self) -> list[int]:
        return self._setting_matrices_numbers

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
    def number_dof_rotation(self) -> int:
        """Return the number of internal rotational degrees of freedom"""
        try:
            return self._number_dof_rotation
        except AttributeError:
            self._number_dof_rotation = \
                sum([self.__getattribute__(f'is_internal_rot{idx}') for idx, group in enumerate(self.groups, 1)])
            return self._number_dof_rotation

    @property
    def is_internal_rot1(self) -> bool:
        """Whether there are rotational degrees of freedom for group 1"""
        return 'r:<0,0,1,a>' in self._int_dof_groups[0]

    @property
    def is_internal_rot2(self) -> bool:
        """Whether there are rotational degrees of freedom for group 2"""
        return 'r:<0,0,1,c>' in self._int_dof_groups[1]

    @property
    def number_dof_translation(self) -> int:
        """Return the number of internal translational degrees of freedom"""
        try:
            return self._number_dof_translation
        except AttributeError:
            self._number_dof_translation = \
                sum([self.__getattribute__(f'is_internal_tx{idx}') for idx, group in enumerate(self.groups, 1)])
            return self._number_dof_translation

    @property
    def is_internal_tx1(self) -> bool:
        """Whether there are internal translational degrees of freedom for group 1"""
        return 't:<0,0,b>' in self._int_dof_groups[0]

    @property
    def is_internal_tx2(self) -> bool:
        """Whether there are internal translational degrees of freedom for group 2"""
        return 't:<0,0,d>' in self._int_dof_groups[1]

    @property
    def ref_frame_tx_dof(self) -> bool:
        return self._ref_frame_tx_dof

    @property
    def number_dof_external(self) -> int:
        """Return the number of external degrees of freedom"""
        try:
            return self._number_dof_external
        except AttributeError:
            self._number_dof_external = len(self.external_dof)
            return self._number_dof_external

    @property
    def external_dof(self) -> np.ndarray:
        """Return the total external degrees of freedom as a number DOF externalx3 array"""
        try:
            return self._external_dof
        except AttributeError:
            if not any(self._ref_frame_tx_dof):
                self._external_dof = np.empty((0, 3), float)  # <- np.array([[0.], [0.], [0.]])
            else:
                difference_matrix = self.__external_dof[1] - self.__external_dof[0]
                # for entry 6 - string_vector is ('4*e', '4*e', '4*e')
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

    @property
    def cryst_record(self) -> str | None:
        """Get the CRYST1 record associated with this SymEntry"""
        return None

    def is_cryst_record(self) -> bool:
        """Is the SymEntry utilizing a provided CRYST1 record"""
        return self.number == 0

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

    # # UNUSED
    # def get_optimal_external_tx_vector(self, optimal_ext_dof_shifts: np.ndarray, group_number: int = 1) -> np.ndarray:
    #     """From the DOF and the computed shifts, return the translation vector
    #
    #     Args:
    #         optimal_ext_dof_shifts: The parameters for an ideal shift of a single component in the resulting material
    #         group_number: The number of the group to find the vector for
    #     Returns:
    #         The optimal vector for translation
    #     """
    #     optimal_shifts_t = getattr(self, f'external_dof{group_number}').T * optimal_ext_dof_shifts
    #     return optimal_shifts_t.T.sum(axis=0)

    def get_uc_dimensions(self, optimal_shift_vec: np.ndarray) -> np.ndarray | None:
        """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
        basis vectors with component translation degrees of freedom

        Args:
            optimal_shift_vec: An Nx3 array where N is the number of shift instances
                and 3 is number of possible external degrees of freedom (even if they are not utilized)
        Returns:
            The unit cell dimensions for each optimal shift vector passed
        """
        if self.unit_cell is None:
            return None
        # for entry 6 - self.cell_lengths is ('4*e', '4*e', '4*e')
        # construct_uc_matrix() = [[4, 4, 4], [0, 0, 0], [0, 0, 0]]
        uc_mat = construct_uc_matrix(self.cell_lengths) * optimal_shift_vec[:, :, None]
        # [:, :, None] <- expands axis so multiplication is accurate. eg. [[[1.], [0.], [0.]], [[0.], [0.], [0.]]]
        lengths = np.abs(uc_mat.sum(axis=-2))
        #               (^).sum(axis=-2) = [4, 4, 4]
        if len(self.cell_lengths) == 2:
            lengths[:, 2] = 1.

        if len(self.cell_angles) == 1:
            angles = [90., 90., float(self.cell_angles[0])]
        else:
            angles = [0., 0., 0.]  # Initialize incase there are < 1 self.cell_angles
            for idx, string_angle in enumerate(self.cell_angles):
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
        if self.unit_cell is None:
            return None
        # uc_mat = construct_uc_matrix(string_lengths) * optimal_shift_vec[:, :, None]  # <- expands axis so mult accurate
        uc_mat = construct_uc_matrix(self.cell_lengths)
        # to reverse the values from the incoming a, b, and c, we should divide by the uc_matrix_constraints
        # given the matrix should only ever have one value in each column (max) a sum over the column should produce the
        # desired vector to calculate the optimal shift.
        # There is a possibility of returning inf when we divide 0 by a value so ignore this warning
        with warnings.catch_warnings():
            # Cause all warnings to always be ignored
            warnings.simplefilter('ignore')
            external_translation_shifts = [a, b, c] / np.abs(uc_mat.sum(axis=-2))
            # replace any inf with zero
            external_translation_shifts = np.nan_to_num(external_translation_shifts, copy=False, posinf=0., neginf=0.)

        if len(self.cell_lengths) == 2:
            external_translation_shifts[2] = 1.

        return external_translation_shifts

    def sdf_lookup(self) -> AnyStr:
        """Locate the proper symmetry definition file depending on the specified symmetry

        Returns:
            The location of the symmetry definition file on disk
        """
        if self.dimension > 0:
            return os.path.join(putils.symmetry_def_files, 'C1.sym')

        symmetry = self.simple_specification
        for file, ext in map(os.path.splitext, os.listdir(putils.symmetry_def_files)):
            if symmetry == file:
                return os.path.join(putils.symmetry_def_files, file + ext)

        symmetry = self.resulting_symmetry
        for file, ext in map(os.path.splitext, os.listdir(putils.symmetry_def_files)):
            if symmetry == file:
                return os.path.join(putils.symmetry_def_files, file + ext)

        raise FileNotFoundError(
            f"Couldn't locate symmetry definition file at '{putils.symmetry_def_files}' for {self.__class__.__name__} "
            f"{self.number}")

    def log_parameters(self):
        """Log the SymEntry Parameters"""
        #                pdb1_path, pdb2_path, master_outdir
        # log.info('NANOHEDRA PROJECT INFORMATION')
        # log.info(f'Oligomer 1 Input: {pdb1_path}')
        # log.info(f'Oligomer 2 Input: {pdb2_path}')
        # log.info(f'Master Output Directory: {master_outdir}\n')
        logger.info('SYMMETRY COMBINATION MATERIAL INFORMATION')
        logger.info(f'Nanohedra Entry Number: {self.number}')
        logger.info(f'Oligomer 1 Point Group Symmetry: {self.group1}')
        logger.info(f'Oligomer 2 Point Group Symmetry: {self.group2}')
        logger.info(f'SCM Point Group Symmetry: {self.point_group_symmetry}')
        # logger.debug(f'Oligomer 1 Internal ROT DOF: {self.is_internal_rot1}')
        # logger.debug(f'Oligomer 2 Internal ROT DOF: {self.is_internal_rot2}')
        # logger.debug(f'Oligomer 1 Internal Tx DOF: {self.is_internal_tx1}')
        # logger.debug(f'Oligomer 2 Internal Tx DOF: {self.is_internal_tx2}')
        # Todo textwrap.textwrapper() prettify these matrices
        logger.debug(f'Oligomer 1 Setting Matrix: {self.setting_matrix1.tolist()}')
        logger.debug(f'Oligomer 2 Setting Matrix: {self.setting_matrix2.tolist()}')
        ext_tx_dof1, ext_tx_dof2, *_ = self.ref_frame_tx_dof
        logger.debug(f'Oligomer 1 Reference Frame Tx DOF: {ext_tx_dof1}')
        logger.debug(f'Oligomer 2 Reference Frame Tx DOF: {ext_tx_dof2}')
        logger.info(f'Resulting SCM Symmetry: {self.resulting_symmetry}')
        logger.info(f'SCM Dimension: {self.dimension}')
        logger.info(f'SCM Unit Cell Specification: {self.uc_specification}\n')
        # rot_step_deg1, rot_step_deg2 = get_rotation_step(self, rot_step_deg1, rot_step_deg2, initial=True, log=logger)
        logger.info('ROTATIONAL SAMPLING INFORMATION')
        logger.info(f'Oligomer 1 ROT Sampling Range: '
                    f'{self.rotation_range1 if self.is_internal_rot1 else None}')
        logger.info('Oligomer 2 ROT Sampling Range: '
                    f'{self.rotation_range2 if self.is_internal_rot2 else None}')
        # logger.info('Oligomer 1 ROT Sampling Step: '
        #             f'{rot_step_deg1 if self.is_internal_rot1 else None}')
        # logger.info('Oligomer 2 ROT Sampling Step: '
        #             f'{rot_step_deg2 if self.is_internal_rot2 else None}\n')
        # Get Degeneracy Matrices
        # logger.info('Searching For Possible Degeneracies')
        if self.degeneracy_matrices1 is None:
            logger.info('No Degeneracies Found for Oligomer 1')
        elif len(self.degeneracy_matrices1) == 1:
            logger.info('1 Degeneracy Found for Oligomer 1')
        else:
            logger.info(f'{len(self.degeneracy_matrices1)} Degeneracies Found for Oligomer 1')
        if self.degeneracy_matrices2 is None:
            logger.info('No Degeneracies Found for Oligomer 2\n')
        elif len(self.degeneracy_matrices2) == 1:
            logger.info('1 Degeneracy Found for Oligomer 2\n')
        else:
            logger.info(f'{len(self.degeneracy_matrices2)} Degeneracies Found for Oligomer 2\n')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.specification})'


class CrystSymEntry(SymEntry):
    deorthogonalization_matrix: np.ndarray
    orthogonalization_matrix: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(0, **kwargs)

    @property
    def uc_dimensions(self) -> tuple[float, float, float, float, float, float] | None:
        """The unit cell dimensions for the lattice specified by lengths a, b, c and angles alpha, beta, gamma

        Returns:
            length a, length b, length c, angle alpha, angle beta, angle gamma
        """
        try:
            return self._uc_dimensions
        except AttributeError:
            return None

    @uc_dimensions.setter
    def uc_dimensions(self, uc_dimensions: tuple[float, float, float, float, float, float]):
        """Set the unit cell dimensions according to the lengths a, b, and c, and angles alpha, beta, and gamma

        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
        """
        try:
            a, b, c, alpha, beta, gamma = self._uc_dimensions = uc_dimensions
        except (TypeError, ValueError):  # Unpacking didn't work
            return

        degree_to_radians = math.pi / 180.
        gamma *= degree_to_radians

        # unit cell volume
        a_cos = math.cos(alpha * degree_to_radians)
        b_cos = math.cos(beta * degree_to_radians)
        g_cos = math.cos(gamma)
        g_sin = float(math.sin(gamma))
        uc_volume = float(a * b * c * math.sqrt(1 - a_cos**2 - b_cos**2 - g_cos**2 + 2*a_cos*b_cos*g_cos))

        # deorthogonalization matrix m
        # m0 = [1./a, -g_cos / (a*g_sin),
        #       ((b*g_cos*c*(a_cos - b_cos*g_cos) / g_sin) - b*c*b_cos*g_sin) * (1/self.uc_volume)]
        # m1 = [0., 1./b*g_sin, -(a*c*(a_cos - b_cos*g_cos) / (self.uc_volume*g_sin))]
        # m2 = [0., 0., a*b*g_sin/self.uc_volume]
        self.deorthogonalization_matrix = np.array(
            [[1. / a, -g_cos / (a*g_sin),
              ((b * g_cos * c * (a_cos - b_cos*g_cos) / g_sin) - b*c*b_cos*g_sin) * (1/uc_volume)],
             [0., 1. / (b*g_sin), -(a * c * (a_cos-b_cos*g_cos) / (uc_volume*g_sin))],
             [0., 0., a * b * g_sin / uc_volume]])

        # orthogonalization matrix m_inv
        # m_inv_0 = [a, b*g_cos, c*b_cos]
        # m_inv_1 = [0., b*g_sin, (c*(a_cos - b_cos*g_cos))/g_sin]
        # m_inv_2 = [0., 0., self.uc_volume/(a*b*g_sin)]
        self.orthogonalization_matrix = np.array([[a, b * g_cos, c * b_cos],
                                                  [0., b * g_sin, (c * (a_cos - b_cos*g_cos)) / g_sin],
                                                  [0., 0., uc_volume / (a*b*g_sin)]])

    @property
    def cryst_record(self) -> str | None:
        """Get the CRYST1 record associated with this SymEntry"""
        try:
            return self._cryst_record
        except AttributeError:
            self._cryst_record = None
            return self._cryst_record

    @cryst_record.setter
    def cryst_record(self, cryst_record: str | None):
        self._cryst_record = cryst_record


# Set up the baseline crystalline entry which will allow for flexible adaptation of non-Nanohedra SymEntry instances
CrystRecord = CrystSymEntry(space_group='P1')


class SymEntryFactory:
    """Return a SymEntry instance by calling the Factory instance with the SymEntry entry number and symmetry map
    (sym_map)

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the SymEntry
    """
    def __init__(self, **kwargs):
        self._entries = {}

    def destruct(self, **kwargs):
        self._entries = {}

    def __call__(self, entry: int, sym_map: list[str] = None, **kwargs) -> SymEntry:
        """Return the specified SymEntry object singleton

        Args:
            entry: The entry number
            sym_map: The particular mapping of the symmetric groups
        Returns:
            The instance of the specified SymEntry
        """
        if sym_map is None:
            sym_map_string = 'None'
        else:
            sym_map_string = '|'.join('None' if sym is None else sym for sym in sym_map)

        if entry == 0:
            # Don't add this to the self._entries
            return CrystSymEntry(sym_map=sym_map, **kwargs)

        entry_key = f'{entry}|{sym_map_string}'
        symmetry = self._entries.get(entry_key)
        if symmetry:
            return symmetry
        else:
            self._entries[entry_key] = sym_entry = SymEntry(entry, sym_map=sym_map, **kwargs)
            return sym_entry

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


# def get_tx_dof_ref_frame_var_vec(string_vec, var):  # UNUSED
#     return_vec = [0., 0., 0.]
#     for i in range(3):
#         if var in string_vec[i] and '*' in string_vec[i]:
#             return_vec[i] = float(string_vec[i].split('*')[0])
#         elif "-" + var in string_vec[i]:
#             return_vec[i] = -1.
#         elif var == string_vec[i]:
#             return_vec[i] = 1.
#     return return_vec


# def parse_ref_tx_dof_str_to_list(ref_frame_tx_dof_string):  # UNUSED
#     return list(map(str.strip, ref_frame_tx_dof_string.strip('<>').split(',')))


# def get_optimal_external_tx_vector(ref_frame_tx_dof, optimal_ext_dof_shifts):  # UNUSED
#     """From the DOF and the computed shifts, return the translation vector
#
#     Args:
#         ref_frame_tx_dof:
#         optimal_ext_dof_shifts:
#
#     Returns:
#         (list[list[list]])
#     """
#     ext_dof_variables = ['e', 'f', 'g']
#
#     parsed_ref_tx_vec = parse_ref_tx_dof_str_to_list(ref_frame_tx_dof)
#
#     optimal_external_tx_vector = np.array([0.0, 0.0, 0.0])
#     for idx, dof_shift in enumerate(optimal_ext_dof_shifts):
#         var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_tx_vec, ext_dof_variables[idx])
#         optimal_external_tx_vector += np.array(var_vec) * dof_shift
#
#     return optimal_external_tx_vector.tolist()


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
        for step in range(0, int(rot_range_deg//step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[1., 0., 0.], [0., math.cos(rad), -math.sin(rad)], [0., math.sin(rad), math.cos(rad)]])
    elif axis == 'y':
        for step in range(0, int(rot_range_deg//step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[math.cos(rad), 0., math.sin(rad)], [0., 1., 0.], [-math.sin(rad), 0., math.cos(rad)]])
    elif axis == 'z':
        for step in range(0, int(rot_range_deg//step_deg)):
            rad = math.radians(step * step_deg)
            rot_matrices.append([[math.cos(rad), -math.sin(rad), 0.], [math.sin(rad), math.cos(rad), 0.], [0., 0., 1.]])
    else:
        raise ValueError(f"Axis '{axis}' isn't supported")

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
        rotations = identity_matrix[None, :, :]  # Expand to shape (1, 3, 3)
    elif np.all(identity_matrix == rotations[0]):
        pass  # This is correct
    else:
        logger.warning(f'{make_rotations_degenerate.__name__}: The argument "rotations" is missing an identity '
                       'matrix which is recommended to produce the correct matrices. Adding now')
        rotations = [identity_matrix] + list(rotations)

    if degeneracies is None:
        degeneracies = identity_matrix[None, :, :]  # Expand to shape (1, 3, 3)
    elif np.all(identity_matrix == degeneracies[0]):
        pass  # This is correct
    else:
        logger.warning(f'{make_rotations_degenerate.__name__}: The argument "degeneracies" is missing an identity '
                       'matrix which is recommended to produce the correct matrices. Adding now')
        degeneracies = [identity_matrix] + list(degeneracies)

    return np.concatenate([np.matmul(rotations, degen_mat) for degen_mat in degeneracies])


# def parse_uc_str_to_tuples(uc_string):  # UNUSED
#     """Acquire unit cell parameters from specified external degrees of freedom string"""
#     def s_to_l(string):
#         s1 = string.replace('(', '')
#         s2 = s1.replace(')', '')
#         l1 = s2.split(',')
#         l2 = [x.replace(' ', '') for x in l1]
#         return l2
#
#     # if '),' in uc_string:
#     #     l = uc_string.split('),')
#     # else:
#     #     l = [uc_string]
#
#     return [s_to_l(s) for s in uc_string.split('), ')]


# def get_uc_var_vec(string_vec, var):  # UNUSED
#     """From the length specification return the unit vector"""
#     return_vec = [0.0, 0.0, 0.0]
#     for i in range(len(string_vec)):
#         if var in string_vec[i] and '*' in string_vec[i]:
#             return_vec[i] = (float(string_vec[i].split('*')[0]))
#         elif var == string_vec[i]:
#             return_vec.append(1.0)
#     return return_vec


# def get_uc_dimensions(uc_string, e=1, f=0, g=0):  # UNUSED
#     """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
#     basis vectors with component translation degrees of freedom"""
#     string_vec_lens, string_vec_angles = parse_uc_str_to_tuples(uc_string)
#     e_vec = get_uc_var_vec(string_vec_lens, 'e')
#     f_vec = get_uc_var_vec(string_vec_lens, 'f')
#     g_vec = get_uc_var_vec(string_vec_lens, 'g')
#     e1 = [e_vec_val * e for e_vec_val in e_vec]
#     f1 = [f_vec_val * f for f_vec_val in f_vec]
#     g1 = [g_vec_val * g for g_vec_val in g_vec]
#
#     lengths = [0.0, 0.0, 0.0]
#     for i in range(len(string_vec_lens)):
#         lengths[i] = abs((e1[i] + f1[i] + g1[i]))
#     if len(string_vec_lens) == 2:
#         lengths[2] = 1.0
#
#     if len(string_vec_angles) == 1:
#         angles = [90.0, 90.0, float(string_vec_angles[0])]
#     else:
#         angles = [0.0, 0.0, 0.0]
#         for idx, string_vec_angle in enumerate(string_vec_angles):
#             angles[idx] = float(string_vec_angle)
#
#     return lengths + angles


highest_point_group_msg = f'If this is a point group. You likely need to modify the current highest cyclic symmetry ' \
                          f'{MAX_SYMMETRY} in {putils.path_to_sym_utils}, then run the file using "python ' \
                          f'{putils.path_to_sym_utils}".'
example_symmetry_specification = 'RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}...'


def parse_symmetry_specification(specification: str) -> list[str]:
    """Parse the typical symmetry specification string with format RESULT:{SUBSYMMETRY1}{SUBSYMMETRY2}... to a list

    Args:
        specification: The specification string
    Returns:
        The parsed string with each member split into a list - ['RESULT', 'SUBSYMMETRY1', 'SUBSYMMETRY2', ...]
    """
    return [split.strip('}:') for split in specification.split('{')]


def parse_symmetry_to_sym_entry(sym_entry_number: int = None, symmetry: str = None, sym_map: list[str] = None) -> \
        SymEntry | None:
    """Take a symmetry specified in a number of ways and return the symmetry parameters in a SymEntry instance

    Args:
        sym_entry_number: The integer corresponding to the desired SymEntry
        symmetry: The symmetry specified by a string
        sym_map: A symmetry map where each successive entry is the corresponding symmetry group number for the structure
    Returns:
        The SymEntry instance or None if parsing failed
    """
    if sym_map is None:  # Find sym_map from symmetry
        if symmetry is not None:
            symmetry = symmetry.strip()
            if symmetry in space_group_symmetry_operators:  # space_group_symmetry_operators in Hermann-Mauguin notation
                # Only have the resulting symmetry, set it and then solve by lookup_sym_entry_by_symmetry_combination()
                sym_map = [symmetry]
            elif len(symmetry) > 3:
                if ':{' in symmetry:  # Symmetry specification of typical type result:{subsymmetry}{}...
                    sym_map = parse_symmetry_specification(symmetry)
                elif CRYST in symmetry.upper():  # This is crystal specification
                    return None  # Have to set this up after parsing cryst records
                else:  # This is some Rosetta based symmetry?
                    sym_str1, sym_str2, sym_str3, *_ = symmetry
                    sym_map = f'{sym_str1} C{sym_str2} C{sym_str3}'.split()
                    logger.error(f"Symmetry specification '{symmetry}' isn't understood, trying to solve anyway\n\n")
            elif symmetry in valid_symmetries:
                # logger.debug(f'{parse_symmetry_to_sym_entry.__name__}: The functionality of passing symmetry as '
                #              f"{symmetry} hasn't been tested thoroughly yet")
                # Specify as [result, entity1, no other entities]
                sym_map = [symmetry, symmetry, None]
            elif len(symmetry) == 3 and symmetry[1].isdigit() and symmetry[2].isdigit():  # like I32, O43 format
                sym_map = [*symmetry]
            else:  # C35
                raise ValueError(
                    f"{symmetry} isn't a supported symmetry... {highest_point_group_msg}")
        elif sym_entry_number is not None:
            return symmetry_factory.get(sym_entry_number)
        else:
            raise utils.SymmetryInputError(
                f"{parse_symmetry_to_sym_entry.__name__}: Can't initialize without 'symmetry' or 'sym_map'")

    if sym_entry_number is None:
        try:  # To lookup in the all_sym_entry_dict
            sym_entry_number = utils.dictionary_lookup(all_sym_entry_dict, sym_map)
            if not isinstance(sym_entry_number, int):
                raise TypeError
        except (KeyError, TypeError):
            # The prescribed symmetry is a point, plane, or space group that isn't in Nanohedra symmetry combinations.
            # Try to load a custom input
            sym_entry_number = lookup_sym_entry_by_symmetry_combination(*sym_map)

    return symmetry_factory.get(sym_entry_number, sym_map=sym_map)


def sdf_lookup(symmetry: str = None) -> AnyStr:
    """From the set of possible point groups, locate the proper symmetry definition file depending on the specified
    symmetry. If none is specified, a C1 symmetry will be returned (this doesn't make sense but is completely viable)

    Args:
        symmetry: Can be a valid_point_group, or None
    Returns:
        The location of the symmetry definition file on disk
    """
    if not symmetry or symmetry.upper() == 'C1':
        return os.path.join(putils.symmetry_def_files, 'C1.sym')
    else:
        symmetry = symmetry.upper()

    for file, ext in map(os.path.splitext, os.listdir(putils.symmetry_def_files)):
        if symmetry == file:
            return os.path.join(putils.symmetry_def_files, file + ext)

    raise FileNotFoundError(
        f"For symmetry: {symmetry}, couldn't locate correct symmetry definition file at '{putils.symmetry_def_files}'")


repeat_with_sym_entry = "Can't distinguish between the desired entries. " \
                        'Repeat your command and specify the preferred ENTRY to proceed.\nEx:\n    --sym-entry 1'
# {flags.format_args(flags.sym_entry_args)}
query_output_format_string = \
    '{:>5d} {:>6s} {:>9d} {:>6s} {:>10d} {:>9d} {:^20s} {:>6s} {:>10d} {:>9d} {:^20s} {:>8d} {:>8d} {:>8d}'


def print_query_header():
    # number_of_groups: int
    print('\033[1m{:5s} {:6s} {:9} {:6s} {:10s} {:9s} {:^20s} {:6s} {:10s} {:9s} {:^20s} {:8s} {:8s} {:8s}\033['
          '0m'.format(
              'Entry', 'Result', 'Dimension',
              f'Group1', 'IntDofRot1', 'IntDofTx1', 'ReferenceFrameDof1',
              'Group2', 'IntDofRot2', 'IntDofTx2', 'ReferenceFrameDof2', 'TotalDof', 'RingSize', 'Dockable'))
    # *((f'Group{i}', f'IntDofRot{i}', f'IntDofTx{i}', f'ReferenceFrameDof{i}' for i in range(number_of_groups))


def symmetry_groups_are_allowed_in_entry(symmetry_operators: Iterable[str], *groups: Iterable[str], result: str = None,
                                         entry_number: int = None) -> bool:
    """Check if the provided symmetry operators are allowed in a SymEntry

    Args:
        symmetry_operators: The symmetry operators of interest
        groups: The groups provided in the symmetry
        result: The resulting symmetry
        entry_number: The SymEntry number of interest
    Returns:
        True if the symmetry operators are valid, False otherwise
    """
    if result is not None:
        # if group1 is None and group2 is None:
        if not groups:
            raise ValueError(
                f"When using the argument 'result', must provide at least 1 group. Got {groups}")
    elif entry_number is not None:
        entry = symmetry_combinations.get(entry_number)
        if entry is None:
            raise utils.SymmetryInputError(
                f"The entry number {entry_number} isn't an available {SymEntry.__name__}")

        group1, _, _, _, group2, _, _, _, _, result, *_ = entry
        groups = (group1, group2)  # Todo modify for more than 2
    else:
        raise ValueError(
            'Must provide entry_number, or the result and *groups arguments. None were provided')

    # Find all sub_symmetries that are viable in the component group members
    for group in groups:
        group_members = sub_symmetries.get(group, [None])
        for sym_operator in symmetry_operators:
            if sym_operator in [result, *groups]:
                continue
            elif sym_operator in group_members:
                continue
            else:
                return False

    return True  # Assume correct unless proven incorrect


def get_int_dof(*groups: Iterable[str]) -> list[tuple[int, int], ...]:
    """Usage
    int_dof1, int_dof2, *_ = get_int_dof(int_dof_group1, int_dof_group2)
    """
    group_int_dofs = []
    for group_int_dof in groups:
        int_rot = int_tx = 0
        for int_dof in group_int_dof:
            if int_dof.startswith('r'):
                int_rot = 1
            if int_dof.startswith('t'):
                int_tx = 1
        group_int_dofs.append((int_rot, int_tx))

    return group_int_dofs


def lookup_sym_entry_by_symmetry_combination(result: str, *symmetry_operators: str) -> int:
    """Given the resulting symmetry and the symmetry operators for each Entity, solve for the SymEntry

    Args:
        result: The global symmetry
        symmetry_operators: Additional operators which specify sub-symmetric systems in the larger result
    Returns:
        The entry number of the SymEntry
    """

    # def print_matching_entries(entries):
    #     if not entries:
    #         return
    #     print_query_header()
    #     for _entry in entries:
    #         _group1, _int_dof_group1, _, _ref_frame_tx_dof_group1, _group2, _int_dof_group2, _, \
    #             _ref_frame_tx_dof_group2, _, _result, dimension, *_ = symmetry_combinations[_entry]
    #         int_dof1, int_dof2, *_ = get_int_dof(_int_dof_group1, _int_dof_group2)
    #         print(query_output_format_string.format(
    #             _entry, result, dimension,
    #             _group1, *int_dof1, str(_ref_frame_tx_dof_group1),
    #             _group2, *int_dof2, str(_ref_frame_tx_dof_group2)))

    def report_multiple_solutions(entries: list[int]):
        # entries = sorted(entries)
        # print(f'\033[1mFound specified symmetries matching including {", ".join(map(str, entries))}\033[0m')
        # print(f'\033[1mFound specified symmetries matching\033[0m')
        print_matching_entries(entries)
        print(repeat_with_sym_entry)

    result = str(result)
    result_entries = []
    matching_entries = []
    for entry_number, entry in symmetry_combinations.items():
        group1, _, _, _, group2, _, _, _, _, resulting_symmetry, *_ = entry
        if resulting_symmetry == result:
            result_entries.append(entry_number)

            if symmetry_operators and \
                    symmetry_groups_are_allowed_in_entry(symmetry_operators, entry_number=entry_number):
                matching_entries.append(entry_number)  # Todo include the groups?

    if matching_entries:
        if len(matching_entries) != 1:
            # Try to solve
            # good_matches: dict[int, list[str]] = defaultdict(list)
            good_matches: dict[int, list[str]] = {entry_number: [None, None] for entry_number in matching_entries}
            for entry_number in matching_entries:
                group1, _, _, _, group2, _, _, _, _, resulting_symmetry, *_ = symmetry_combinations[entry_number]
                match_tuple = good_matches[entry_number]
                for sym_op in symmetry_operators:
                    if sym_op == group1 and match_tuple[0] is None:
                        match_tuple[0] = sym_op
                    elif sym_op == group2 and match_tuple[1] is None:
                        match_tuple[1] = sym_op
            # logger.debug(f'good matches: {good_matches}')

            # max_ops = 0
            exact_matches = []
            for entry_number, ops in good_matches.items():
                number_ops = sum(False if op is None else True for op in ops)
                if number_ops == len(symmetry_operators):
                    exact_matches.append(entry_number)
                # elif number_ops > max_ops:
                #     max_ops = number_ops

            if exact_matches:
                if len(exact_matches) == 1:
                    matching_entries = exact_matches
                else:  # Still equal, report bad
                    report_multiple_solutions(exact_matches)
                    sys.exit(1)
                    # -------- TERMINATE --------
            else:  # symmetry_operations are 3 or greater. Get the highest symmetry
                all_matches = []
                for entry_number, matching_ops in good_matches.items():
                    if all(matching_ops):
                        all_matches.append(entry_number)

                if all_matches:
                    if len(all_matches) == 1:
                        matching_entries = all_matches
                    else:  # Still equal, report bad
                        report_multiple_solutions(exact_matches)
                        sys.exit(1)
                        # -------- TERMINATE --------
                else:  # None match all, this must be 2 or more sub-symmetries
                    # max_symmetry_number = 0
                    # symmetry_number_to_entries = defaultdict(list)
                    # for entry_number, matching_ops in good_matches.items():
                    #     if len(matching_ops) == max_ops:
                    #         total_symmetry_number = sum([valid_subunit_number[op] for op in matching_ops])
                    #         symmetry_number_to_entries[total_symmetry_number].append(entry_number)
                    #         if total_symmetry_number > max_symmetry_number:
                    #             max_symmetry_number = total_symmetry_number
                    #
                    # exact_matches = symmetry_number_to_entries[max_symmetry_number]
                    # if len(exact_matches) == 1:
                    #     matching_entries = exact_matches
                    # else:  # Still equal, report bad
                    # print('non-exact matches')
                    report_multiple_solutions(exact_matches)
                    sys.exit(1)
                    # -------- TERMINATE --------
    elif symmetry_operators:
        if result in space_group_symmetry_operators:  # space_group_symmetry_operators in Hermann-Mauguin notation
            matching_entries = [0]  # 0 = CrystSymEntry
        else:
            raise ValueError(
                f"The specified symmetries '{', '.join(symmetry_operators)}' couldn't be coerced to make the resulting "
                f"symmetry '{result}'. Try to reformat your symmetry specification if this is the result of a typo to "
                'include only symmetries that are group members of the resulting symmetry such as '
                f'{", ".join(all_sym_entry_dict.get(result, {}).keys())}\nUse the format {example_symmetry_specification} '
                'during your specification')
    else:  # No symmetry_operators
        if result_entries:
            report_multiple_solutions(result_entries)
            sys.exit()
        else:  # no matches
            raise ValueError(
                f"The resulting symmetry {result} didn't match any possible symmetry_combinations. You are likely "
                'requesting a symmetry that is outside of the parameterized SymEntry entries. If this is a '
                '\033[1mchiral\033[0m plane/space group, modify the function '
                f'{lookup_sym_entry_by_symmetry_combination.__name__} to use '
                f'non-Nanohedra compatible chiral space_group_symmetry_operators. {highest_point_group_msg}')

    logger.debug(f'Found matching SymEntry.number {matching_entries[0]}')
    return matching_entries[0]


def print_matching_entries(match_string, matching_entries: Iterable[int]):
    """Report the relevant information from passed SymEntry entry numbers

    Args:
        match_string: The string inserted into "All entries found matching {match_string}:"
        matching_entries: The matching entry numbers
    Returns:
        None
    """
    if not matching_entries:
        print(f'\033[1mNo entries found matching {match_string}\033[0m\n')
        return
    else:
        matching_entries = sorted(matching_entries)

    print(f'\033[1mAll entries found matching {match_string}:\033[0m')
    print_query_header()
    for entry in matching_entries:
        group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
            ref_frame_tx_dof_group2, result_point_group, result, dimension, cell_lengths, cell_angles, tot_dof, \
            ring_size = symmetry_combinations[entry]
        int_dof1, int_dof2, *_ = get_int_dof(int_dof_group1, int_dof_group2)
        # print(entry, result, dimension,
        #       group1, *int_dof1, ref_frame_tx_dof_group1,
        #       group2, *int_dof2, ref_frame_tx_dof_group2, tot_dof, ring_size)
        if ref_frame_tx_dof_group1 is None:
            ref_frame_tx_dof_group1 = '0,0,0'
        if ref_frame_tx_dof_group2 is None:
            ref_frame_tx_dof_group2 = '0,0,0'
        print(query_output_format_string.format(
                entry, result, dimension,
                group1, *int_dof1, f'<{ref_frame_tx_dof_group1}>',
                str(group2), *int_dof2, f'<{ref_frame_tx_dof_group2}>', tot_dof, ring_size,
                True if entry in nanohedra_symmetry_combinations else False))


query_modes_literal = Literal['all-entries', 'combination', 'group', 'dimension', 'result']
query_mode_args = get_args(query_modes_literal)


def query(mode: query_modes_literal, *additional_mode_args, nanohedra: bool = True):
    """Perform a query of the symmetry combinations

    Args:
        mode: The type of query to perform. Viable options are:
            'all-entries', 'combination', 'counterpart', 'dimension', and 'result'
        *additional_mode_args: Additional query args required
        nanohedra: True if only Nanohedra docking symmetries should be queried
    Returns:
        None
    """
    if nanohedra:
        symmetry_combinations_of_interest = nanohedra_symmetry_combinations
    else:
        symmetry_combinations_of_interest = symmetry_combinations

    matching_entries = []
    if mode == 'all-entries':
        match_string = mode
        # all_entries()
        # def all_entries():
        matching_entries.extend(symmetry_combinations_of_interest.keys())
    else:
        if not additional_mode_args:
            # raise ValueError(
            #     f"Can't query with mode '{mode}' without additional arguments")
            instructions = defaultdict(str, {
                'combination': 'Provide multiple symmetry groups from the possible groups '
                               f'{", ".join(valid_symmetries)}\n'})
            mode_instructions = instructions[mode]
            more_info_prompt = f"For the query mode '{mode}', more information is needed\n" \
                               f"{mode_instructions}What {mode} is requested?"
            additional_mode_args = resources.query.format_input(more_info_prompt)

        if mode == 'combination':
            combination, *_ = additional_mode_args
            match_string = f"{mode} {''.join('{%s}' % group for group in combination)}"
            # query_combination(*additional_mode_args)
            # def query_combination(*combination):
            for entry_number, entry in symmetry_combinations_of_interest.items():
                group1, _, _, _, group2, *_ = entry
                if combination == (group1, group2) or combination == (group2, group1):
                    matching_entries.append(entry_number)
        elif mode == 'result':
            result, *_ = additional_mode_args
            match_string = f'{mode}={result}'
            # query_result(result)
            # def query_result(desired_result: str):
            for entry_number, entry in symmetry_combinations_of_interest.items():
                _, _, _, _, _, _, _, _, _, entry_result, *_ = entry
                if result == entry_result:
                    matching_entries.append(entry_number)
        elif mode == 'group':
            group, *_ = additional_mode_args
            match_string = f'{mode}={group}'
            # query_counterpart(counterpart)
            # def query_counterpart(group: str):
            for entry_number, entry in symmetry_combinations_of_interest.items():
                group1, _, _, _, group2, *_ = entry
                if group in (group1, group2):
                    matching_entries.append(entry_number)

        elif mode == 'dimension':
            dim, *_ = additional_mode_args
            match_string = f'{mode}={dim}'
            # dimension(dim)
            # def dimension(dim: int):
            try:
                dim = int(dim)
            except ValueError:
                pass

            if dim in [0, 2, 3]:
                for entry_number, entry in symmetry_combinations_of_interest.items():
                    _, _, _, _, _, _, _, _, _, _, dimension, *_ = entry
                    if dimension == dim:
                        matching_entries.append(entry_number)
            else:
                print(f"Dimension '{dim}' isn't supported. Valid dimensions are: 0, 2 or 3'")
                sys.exit()
        else:
            raise ValueError(
                f"The mode '{mode}' isn't available")

    # Report those found
    # print(matching_entries)
    print_matching_entries(match_string, matching_entries)


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

    atoms = utils.format_guide_coords_as_atom(transformed_points.tolist())

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
