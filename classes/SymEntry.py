import math
import os

import numpy as np

import PathUtils as PUtils
from SymDesignUtils import start_log, dictionary_lookup, DesignError

from utils.SymmetryUtils import valid_subunit_number, get_ptgrp_sym_op, get_sg_sym_op

# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

logger = start_log(name=__name__)
null_log = start_log(name='null', handler=3, propagate=False)

# SYMMETRY COMBINATION MATERIAL TABLE (T.O.Y and J.L, 2020)
sym_comb_dict = {
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
    # KM Custom entries
    200: ['T', [], 1, '<0,0,0>', 'None', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 1, 1],  # T alone
    201: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 1, 1],
    202: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 1, 1],
    203: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'T', [], 1, '<0,0,0>', 'T', 'T', 0, 'N/A', 1, 1],
    210: ['O', [], 1, '<0,0,0>', 'None', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 1, 1],  # O alone
    211: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 1, 1],
    212: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 1, 1],
    213: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 1, 1],
    214: ['C4', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'O', [], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 1, 1],
    220: ['I', [], 1, '<0,0,0>', 'None', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 1, 1],  # I alone
    221: ['C1', ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 1, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 1, 1],
    222: ['C2', ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 1, 1],
    223: ['C3', ['r:<0,0,1,a>', 't:<0,0,b>'], 7, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 1, 1],
    224: ['C5', ['r:<0,0,1,a>', 't:<0,0,b>'], 9, '<0,0,0>', 'I', [], 1, '<0,0,0>', 'I', 'I', 0, 'N/A', 1, 1],
    # KM 3 component entries
    # 301: {'components': [{'symmetry': 'C1', 'dof_internal': ['r:<1,1,1,h,i,a>', 't:<j,k,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C2', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 1, 'dof_external': '<0,0,0>'},
    #                      {'symmetry': 'C3', 'dof_internal': ['r:<0,0,1,a>', 't:<0,0,b>'], 'setting': 4, 'dof_external': '<0,0,0>'}]
    #                       , 'result': ['T', 'T', 0, 'N/A', 1, 1]},
}
# Standard T:{C3}{C3}
# 54: [54, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 12, '<0,0,0>',
#      'T', 'T', 0, 'N/A', 4, 2],
#
# Number   grp1 grp1_idx            grp1_internal_dof grp1_set_mat grp1_external_dof
# 54: [54, 'C3',      2, ['r:<0,0,1,a>', 't:<0,0,b>'],          4,        '<0,0,0>',
#          grp2 grp2_idx            grp2_internal_dof grp2_set_mat grp2_external_dof
#          'C3',      2, ['r:<0,0,1,c>', 't:<0,0,d>'],         12,        '<0,0,0>',
#          pnt_grp final_sym dim  unit_cell tot_dof ring_size
#          'T',         'T',  0,     'N/A',      4,       2],

# Modified T:{C3}{C3} with group 1 internal DOF allowed, group 2, internal DOF disabled
# 54: [54, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'],  4, '<0,0,0>',
#          'C3', 2,                           [], 12, '<0,0,0>',
#          'T', 'T', 0, 'N/A', 4, 2],

custom_entries = [200, 201, 202, 203, 210, 211, 212, 213, 214, 220, 221, 222, 223, 224]

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
all_sym_entry_dict = {'T': {'C2': {'C3': 5}, 'C3': {'C2': 5, 'C3': 54}, 'T': -1},
                      'O': {'C2': {'C3': 7, 'C4': 13}, 'C3': {'C2': 7, 'C4': 56}, 'C4': {'C2': 13, 'C3': 56}, 'O': -2},
                      'I': {'C2': {'C3': 9, 'C5': 16}, 'C3': {'C2': 9, 'C5': 58}, 'C5': {'C2': 16, 'C3': 58}}}
point_group_sdf_map = {9: 'I32', 16: 'I52', 58: 'I53', 5: 'T32', 54: 'T33',  # 7: 'O32', 13: 'O42', 56: 'O43',
                       }

sub_symmetries = {'C1': ['C1'],
                  'C2': ['C1', 'C2'],
                  'C3': ['C1', 'C3'],
                  'C4': ['C1', 'C2', 'C4'],
                  'C5': ['C1', 'C5'],
                  'C6': ['C1', 'C2', 'C3', 'C6'],
                  'T': ['C1', 'C2', 'C3', 'T'],
                  'O': ['C1', 'C2', 'C3', 'C4', 'O'],
                  'I': ['C1', 'C2', 'C3', 'C5', 'I'],
                  }
# ROTATION RANGE DEG
C2 = 180
C3 = 120
C4 = 90
C5 = 72
C6 = 60
rotation_range = {'C1': 360, 'C2': 180, 'C3': 120, 'C4': 90, 'C5': 72, 'C6': 60}
# ROTATION SETTING MATRICES
setting_matrices = {1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # identity
                    2: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # 90 degrees CC on Y
                    3: [[0.707107, 0.0, 0.707107], [0.0, 1.0, 0.0], [-0.707107, 0.0, 0.707107]],  # 2-fold axis in T, O
                    4: [[0.707107, 0.408248, 0.577350], [-0.707107, 0.408248, 0.577350], [0.0, -0.816497, 0.577350]],  # 3-fold axis in T, O
                    5: [[0.707107, 0.707107, 0.0], [-0.707107, 0.707107, 0.0], [0.0, 0.0, 1.0]],
                    6: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],  # # 90 degrees CC on X
                    7: [[1.0, 0.0, 0.0], [0.0, 0.934172, 0.356822], [0.0, -0.356822, 0.934172]],  # 3-fold axis in I
                    8: [[0.0, 0.707107, 0.707107], [0.0, -0.707107, 0.707107], [1.0, 0.0, 0.0]],
                    9: [[0.850651, 0.0, 0.525732], [0.0, 1.0, 0.0], [-0.525732, 0.0, 0.850651]],  # 5-fold axis in I
                    10: [[0.0, 0.5, 0.866025], [0.0, -0.866025, 0.5], [1.0, 0.0, 0.0]],
                    11: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    12: [[0.707107, -0.408248, 0.577350], [0.707107, 0.408248, -0.577350], [0.0, 0.816497, 0.577350]],  # opposite 3-fold axis in T, O
                    13: [[0.5, -0.866025, 0.0], [0.866025, 0.5, 0.0], [0.0, 0.0, 1.0]]}
flip_x_matrix = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # rot 180x
flip_y_matrix = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]  # rot 180y
identity_matrix = setting_matrices[1]


class SymEntry:
    def __init__(self, entry, sym_map=None):
        sym_entry = sym_comb_dict.get(entry)
        try:
            self.group1, self.int_dof_group1, self.rot_set_group1, self.ref_frame_tx_dof1, \
                self.group2, self.int_dof_group2, self.rot_set_group2, self.ref_frame_tx_dof2, \
                self.pt_grp, self.resulting_symmetry, self.dimension, self.unit_cell, self.tot_dof, self.cycle_size = \
                sym_entry
        except TypeError:
            raise ValueError('\nINVALID SYMMETRY ENTRY \'%s\'. SUPPORTED VALUES ARE: %d to %d and CUSTOM ENTRIES: %s\n'
                             % (entry, 1, len(sym_comb_dict), ', '.join(map(str, custom_entries))))
        self.entry_number = entry
        # Reformat reference_frame entries
        self._is_ref_frame_tx_dof1 = True if self.ref_frame_tx_dof1 != '<0,0,0>' else False
        self._is_ref_frame_tx_dof2 = True if self.ref_frame_tx_dof2 != '<0,0,0>' else False
        self.ref_frame_tx_dof1 = list(map(str.strip, self.ref_frame_tx_dof1.strip('<>').split(',')))
        self.ref_frame_tx_dof2 = list(map(str.strip, self.ref_frame_tx_dof2.strip('<>').split(',')))
        self.group_external_dof1 = construct_tx_dof_ref_frame_matrix(self.ref_frame_tx_dof1)
        self.group_external_dof2 = construct_tx_dof_ref_frame_matrix(self.ref_frame_tx_dof2)

        ext_dof_indices = []
        if not self.is_ref_frame_tx_dof1 and not self.is_ref_frame_tx_dof2:
            self.ext_dof = np.array([0., 0., 0.])
        else:
            difference_matrix = self.group_external_dof2 - self.group_external_dof1
            difference_sum = difference_matrix.sum(axis=1)
            for idx in range(3):
                if difference_sum[idx] != 0:
                    ext_dof_indices.append(idx)
            self.ext_dof = difference_matrix[ext_dof_indices]

        self.n_dof_external = len(ext_dof_indices)
        self.unit_cell = None if self.unit_cell == 'N/A' else \
            [[x.replace(' ', '') for x in dim.replace('()', '').split(',')] for dim in self.unit_cell.split('), ')]

        if self.dimension == 0:
            self.expand_matrices = get_ptgrp_sym_op(self.resulting_symmetry)
        elif self.dimension in [2, 3]:
            self.expand_matrices = get_sg_sym_op(self.resulting_symmetry)
        else:
            raise ValueError('\nINVALID SYMMETRY ENTRY. SUPPORTED DESIGN DIMENSIONS: %s\n'
                             % ', '.join(map(str, [0, 2, 3])))
        self.degeneracy_matrices_1, self.degeneracy_matrices_2 = self.get_degeneracy_matrices()

        if not sym_map:
            self.sym_map = {1: self.group1, 2: self.group2}  # assumes a 2 component symmetry. index with only 2 options
        else:  # requires full specification of all symmetry groups
            for idx, sym_operator in enumerate(sym_map, 1):
                setattr(self, 'group%d' % idx, sym_operator)
            self.sym_map = {idx: getattr(self, 'group%d' % idx) for idx, sym_operator in enumerate(sym_map, 1)}

    # @property
    # def group1_sym(self):
    #     return self.group1

    # @property
    # def group2_sym(self):
    #     return self.group2

    @property
    def point_group_sym(self):
        return self.pt_grp

    @property
    def rotation_range1(self):
        try:
            return self._rotation_range1
        except AttributeError:
            self._rotation_range1 = rotation_range.get(self.group1, 0)
        return self._rotation_range1

    @property
    def rotation_range2(self):
        try:
            return self._rotation_range2
        except AttributeError:
            self._rotation_range2 = rotation_range.get(self.group2, 0)
        return self._rotation_range2

    @property
    def setting_matrix1(self):
        try:
            return self._setting_matrix1
        except AttributeError:
            self._setting_matrix1 = np.array(setting_matrices[self.rot_set_group1])
        return self._setting_matrix1

    # @property
    # def ref_frame_tx_dof1(self):
    #     return self.ref_frame_tx_dof1

    @property
    def setting_matrix2(self):
        try:
            return self._setting_matrix2
        except AttributeError:
            self._setting_matrix2 = np.array(setting_matrices[self.rot_set_group2])
        return self._setting_matrix2

    # @property
    # def ref_frame_tx_dof2(self):
    #     return self.ref_frame_tx_dof2

    # @property
    # def resulting_symmetry(self):
    #     """The final symmetry of the symmetry combination material"""
    #     return self.result

    # @property
    # def dimension(self):
    #     return self.dim

    @property
    def uc_specification(self):
        return self.unit_cell

    @property
    def is_internal_tx1(self):
        try:
            return self._internal_tx1
        except AttributeError:
            if 't:<0,0,b>' in self.int_dof_group1:
                self._internal_tx1 = True
            else:
                self._internal_tx1 = False
        return self._internal_tx1

    @property
    def is_internal_tx2(self):
        try:
            return self._internal_tx2
        except AttributeError:
            if 't:<0,0,d>' in self.int_dof_group2:
                self._internal_tx2 = True
            else:
                self._internal_tx2 = False
        return self._internal_tx2

    def is_internal_rot1(self):
        if 'r:<0,0,1,a>' in self.int_dof_group1:
            return True
        else:
            return False

    def is_internal_rot2(self):
        if 'r:<0,0,1,c>' in self.int_dof_group2:
            return True
        else:
            return False

    def is_ref_frame_tx_dof1(self):
        return self._is_ref_frame_tx_dof1

    def is_ref_frame_tx_dof2(self):
        return self._is_ref_frame_tx_dof2

    # @property
    # def ext_dof(self):
    #     """Return the external degrees of freedom given a symmetry entry
    #
    #     Returns:
    #         (numpy.ndarray)
    #     """
    #     return self.ext_dof

    def get_degeneracy_matrices(self):
        """From the intended point group symmetry and a single component, find the degeneracy matrices that produce all
        viable configurations of the single component in the final symmetry

        Returns:
            (tuple[list[list[list[float]]] or None])
        """
        # here allows for D5. Is this bad? .pop('D5') The sym_entries are hardcoded...
        valid_pt_gp_symm_list = list(valid_subunit_number.keys())
        valid_pt_gp_symm_list.remove('D5')

        if self.group1 not in valid_pt_gp_symm_list:
            raise ValueError('Invalid Point Group Symmetry')

        if self.group2 not in valid_pt_gp_symm_list:
            raise ValueError('Invalid Point Group Symmetry')

        if self.point_group_sym not in valid_pt_gp_symm_list:
            raise ValueError('Invalid Point Group Symmetry')

        if self.dimension not in [0, 2, 3]:
            raise ValueError('Invalid Design Dimension')

        degeneracies = []
        for i in range(2):  # Todo expand to situations where more than 2 symmetries...
            oligomer_symmetry = self.group1 if i == 0 else self.group2

            degeneracy_matrices = None
            # For cages, only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            if self.dimension == 0 and i == 1:
                degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            # For layers that obey a cyclic point group symmetry and that are constructed from two oligomers that both
            # obey cyclic symmetry only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            elif self.dimension == 2 and i == 1 and \
                    (self.group1[0], self.group2[0], self.point_group_sym[0]) == ('C', 'C', 'C'):
                degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            # else:
            #     if oligomer_symmetry[0] == "C" and design_symmetry[0] == "C":
            #         degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            elif oligomer_symmetry in ['D3', 'D4', 'D6'] and self.point_group_sym in ['D3', 'D4', 'D6', 'T', 'O']:
                # commented out "if" statement below because all possible translations are not always tested for D3
                # example: in entry 82, only translations along <e,0.577e> are sampled.
                # This restriction only considers 1 out of the 2 equivalent Wyckoff positions.
                # <0,e> would also have to be searched as well to remove the "if" statement below.
                # if (oligomer_symmetry, design_symmetry_pg) != ('D3', 'D6'):
                if oligomer_symmetry == 'D3':
                    # ROT 60 degrees about z
                    degeneracy_matrices = [[[0.5, -0.86603, 0.0], [0.86603, 0.5, 0.0], [0.0, 0.0, 1.0]]]
                elif oligomer_symmetry == 'D4':
                    # 45 degrees about z; z unaffected; x goes to [1,-1,0] direction
                    degeneracy_matrices = [[[0.707107, 0.707107, 0.0], [-0.707107, 0.707107, 0.0], [0.0, 0.0, 1.0]]]
                elif oligomer_symmetry == 'D6':
                    # ROT 30 degrees about z
                    degeneracy_matrices = [[[0.86603, -0.5, 0.0], [0.5, 0.86603, 0.0], [0.0, 0.0, 1.0]]]

            elif oligomer_symmetry == 'D2' and self.point_group_sym != 'O':
                if self.point_group_sym == 'T':
                    degeneracy_matrices = [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]  # ROT90z

                elif self.point_group_sym == 'D4':
                    degeneracy_matrices = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]  # z,x,y and y,z,x

                elif self.point_group_sym == 'D2' or self.point_group_sym == 'D6':
                    degeneracy_matrices = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                                           [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]]

            elif oligomer_symmetry == 'T' and self.point_group_sym == 'T':
                degeneracy_matrices = [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]  # ROT90z

            degeneracies.append(np.array(degeneracy_matrices)) if degeneracy_matrices is not None \
                else degeneracies.append(None)

        return degeneracies

    def get_optimal_external_tx_vector(self, optimal_ext_dof_shifts, group_number=1):
        """From the DOF and the computed shifts, return the translation vector

        Args:
            optimal_ext_dof_shifts:
        Keyword Args:
            group_number=1 (int): The number of the group to find the vector for
        Returns:
            (numpy.ndarray[float]): The optimal vector for translation
        """
        optimal_shifts_t = getattr(self, 'group_external_dof%d' % group_number).T * optimal_ext_dof_shifts
        return optimal_shifts_t.T.sum(axis=0)

    def get_uc_dimensions(self, optimal_shift_vec):
        """Return an array with the three unit cell lengths and three angles [20, 20, 20, 90, 90, 90] by combining UC
        basis vectors with component translation degrees of freedom

        Args:
            optimal_shift_vec (numpy.ndarray)
        Returns:
            (Union[numpy.ndarray, None]): The Unit Cell dimensions for each optimal shift vector passed
        """
        if not self.unit_cell:
            return None
        string_lengths, string_angles = self.unit_cell
        uc_mat = construct_uc_matrix(string_lengths) * optimal_shift_vec[:, :, None]  # <- expands axis so mult accurate

        lengths = np.abs(uc_mat.sum(axis=1))
        # lengths = [0.0, 0.0, 0.0]
        # for i in range(len(string_vec_lens)):
        #     lengths[i] = abs((e1[i] + f1[i] + g1[i]))
        if len(string_lengths) == 2:
            lengths[:, 2] = 1.0

        if len(string_angles) == 1:
            angles = [90.0, 90.0, float(string_angles[0])]
        else:
            # angles = [float(string_angle) for string_angle in string_angles]
            angles = [0.0, 0.0, 0.0]  # need this incase there are only 2 angles... which there aren't right now
            for idx, string_angle in enumerate(string_angles):
                angles[idx] = float(string_angle)

        return np.concatenate(lengths, angles)


def construct_uc_matrix(string_vector):
    """

    Args:
        string_vector (list[str]):

    Returns:
        (numpy.ndarray)
    """
    string_position = {'e': 0, 'f': 1, 'g': 2}
    variable_matrix = np.zeros((3, 3))
    for col_idx, string in enumerate(string_vector):
        if string[-1] != '0':
            row_idx = string_position.get(string[-1])
            variable_matrix[row_idx][col_idx] = float(string.split('*')[0]) if '*' in string else 1.

            if '-' in string:
                variable_matrix[row_idx][col_idx] *= -1

    return variable_matrix


def construct_tx_dof_ref_frame_matrix(string_vector):
    """

    Args:
        string_vector (list[str]):
    Returns:
        (numpy.ndarray)
    """
    string_position = {'e': 0, 'f': 1, 'g': 2}
    variable_matrix = np.zeros((3, 3))
    for col_idx, string in enumerate(string_vector):
        if string[-1] != '0':
            row_idx = string_position.get(string[-1])
            variable_matrix[row_idx][col_idx] = float(string.split('*')[0]) if '*' in string else 1.

            if '-' in string:
                variable_matrix[row_idx][col_idx] *= -1

    return variable_matrix


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


def get_rot_matrices(step_deg, axis='z', rot_range_deg=360):
    """Return a group of rotation matrices to rotate coordinates about a specified axis in set step increments

    Args:
        step_deg (int): The number of degrees for each rotation step
    Keyword Args:
        axis='z' (str): The axis about which to rotate
        rot_range_deg=360 (int): The range with which rotation is possible
    Returns:
        (numpy.ndarray): The rotation matrices with shape (rotations, 3, 3) # list[list[list]])
    """
    if rot_range_deg == 0:
        return

    rot_matrices = []
    axis = axis.lower()
    if axis == 'x':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[1, 0, 0], [0, math.cos(rad), -1 * math.sin(rad)], [0, math.sin(rad), math.cos(rad)]])
    elif axis == 'y':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-1 * math.sin(rad), 0, math.cos(rad)]])
    elif axis == 'z':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[math.cos(rad), -1 * math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
    else:
        print('Axis \'%s\' is not supported' % axis)
        return

    return np.array(rot_matrices)


def get_degen_rotmatrices(degeneracy_matrices=None, rotation_matrices=None):
    """From a set of degeneracy matrices and a set of rotation matrices, produce the complete combination of the
    specified transformations.

    Keyword Args:
        degeneracy_matrices (np.ndarray): column major with shape (degeneracies, 3, 3)
            # [[[x, y, z], [x, y, z], [x, y, z]], ...]
        rotation_matrices (np.ndarray): row major with shape (rotations, 3, 3)
            # [[[x, y, z], [x, y, z], [x, y, z]], ...]
    Returns:
        (list[np.ndarray])  # (list[list[list[list]]])
    """
    if rotation_matrices is not None and degeneracy_matrices is not None:
        degen_rot_matrices = [np.matmul(rotation_matrices, degen_mat) for degen_mat in degeneracy_matrices]
        # degen_rot_matrices = \
        #     [[np.matmul(rotation_matrices, degen_mat)] for degen_mat in degeneracy_matrices]
        return [rotation_matrices] + degen_rot_matrices

    elif rotation_matrices is not None and degeneracy_matrices is None:
        return [rotation_matrices]

    elif rotation_matrices is None and degeneracy_matrices is not None:  # is this ever true? list addition seems wrong
        return [[np.array(identity_matrix)]] + [[degen_mat] for degen_mat in degeneracy_matrices]

    elif rotation_matrices is None and degeneracy_matrices is None:
        return [[np.array(identity_matrix)]]


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
            angles[i] = float(string_vec_angle)

    return lengths + angles


def parse_symmetry_to_sym_entry(symmetry_string):
    symmetry_string = symmetry_string.strip()
    clean_split = None
    if len(symmetry_string) > 3:
        clean_split = [split.strip('}:') for split in symmetry_string.split('{')]
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
    except (KeyError, TypeError):  # when the entry is not specified in the all_sym_entry_dict
        # the prescribed symmetry was a plane, space group, or point group that isn't in nanohedra. try a custom input
        # raise ValueError('%s is not a supported symmetry!' % symmetry_string)
        sym_entry = lookup_sym_entry_by_symmetry_combination(*clean_split)

    # logger.debug('Found Symmetry Entry %s for %s.' % (sym_entry, symmetry_string))
    return SymEntry(sym_entry, sym_map=clean_split[1:])  # remove the result


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


def lookup_sym_entry_by_symmetry_combination(result, *symmetry_operators):
    if isinstance(result, str):
        matching_entries = []
        for entry_number, entry in sym_comb_dict.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, resulting_symmetry, dimension, _, _, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if resulting_symmetry == result:
                required_sym_operators = True
                group1_members = sub_symmetries.get(group1.replace('D', 'C'), list())
                group1_members.extend('C2') if 'D' in group1 else None
                # group1_dihedral = True if 'D' in group1 else False
                group2_members = sub_symmetries.get(group2.replace('D', 'C'), list())
                group2_members.extend('C2') if 'D' in group2 else None
                # group2_dihedral = True if 'D' in group2 else False
                for sym_operator in symmetry_operators:
                    if sym_operator in [resulting_symmetry, group1, group2]:
                        continue
                    elif sym_operator in [group1_members, group2_members]:
                        continue
                    else:
                        required_sym_operators = False

                if required_sym_operators:
                    matching_entries.append(entry_number)  # TODO include the groups?
                # else:
                #     matching_entries.append()
        if len(matching_entries) == 1:
            return matching_entries[0]
        else:
            raise ValueError('The specified symmetries "%s" could not be coerced to make the resulting symmetry "%s".'
                             'Try to reformat your symmetry specification to include only symmetries that are group '
                             'members of the resulting symmetry'
                             % (', '.join(symmetry_operators), result))
    else:
        raise ValueError('The arguments passed to %s are improperly formatted!'
                         % lookup_sym_entry_by_symmetry_combination.__name__)


header_format_string = "{:5s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}"
query_output_format_string = "{:>5s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}"


def print_query_header():
    print(header_format_string.format("ENTRY", "GROUP1", "IntDofRot1", "IntDofTx1", "ReferenceFrameDof1", "GROUP2",
                                      "IntDofRot2", "IntDofTx2", "ReferenceFrameDof2", "RESULT"))


def query_combination(combination_list):
    if isinstance(combination_list, list) and len(combination_list) == 2:
        matching_entries = []
        for entry_number, entry in sym_comb_dict.items():
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
        for entry_number, entry in sym_comb_dict.items():
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
        for entry_number, entry in sym_comb_dict.items():
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
        if matching_entries == []:
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
    for entry_number, entry in sym_comb_dict.items():
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
        for entry_number, entry in sym_comb_dict.items():
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
