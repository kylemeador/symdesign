import math

import numpy as np

from utils.ExpandAssemblyUtils import get_ptgrp_sym_op, get_sg_sym_op


# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

# SYMMETRY COMBINATION MATERIAL TABLE (T.O.Y and J.L, 2020)
from utils.SymmUtils import valid_subunit_number

sym_comb_dict = {
    1: [1, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C2', 1, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D2', 'D2', 0, 'N/A', 4, 2],
    2: [2, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C3', 2, ['r:<0,0,1,c>'], 1, '<e,0.577350*e,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 6],
    3: [3, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D3', 'D3', 0, 'N/A', 4, 2],
    4: [4, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 5, 6],
    5: [5, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'T', 'T', 0, 'N/A', 4, 3],
    6: [6, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,e,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'T', 'I213', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 5, 10],
    7: [7, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 4],
    8: [8, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 4, '<0,0,0>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 5, 10],
    9: [9, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 7, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 5],
    10: [10, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C4', 3, ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C4', 'p4', 2, '(2*e, 2*e), 90', 4, 4],
    11: [11, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D4', 'D4', 0, 'N/A', 4, 2],
    12: [12, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<e,0,0>', 'D4', 'p4212', 2, '(2*e, 2*e), 90', 5, 4],
    13: [13, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 3],
    14: [14, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 5, 8],
    15: [15, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C5', 4, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D5', 'D5', 0, 'N/A', 4, 2],
    16: [16, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C5', 4, ['r:<0,0,1,c>', 't:<0,0,d>'], 9, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 3],
    17: [17, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'C6', 5, ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 3],
    18: [18, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'C6', 5, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D6', 'D6', 0, 'N/A', 4, 2],
    19: [19, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'C6', 5, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 5, 4],
    20: [20, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,f,0>', 'D2', 6, ['None'], 1, '<0,0,0>', 'D2', 'c222', 2, '(4*e, 4*f), 90', 4, 4],
    21: [21, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 4],
    22: [22, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,f>', 'D2', 6, ['None'], 5, '<0,0,0>', 'D4', 'I4122', 3, '(4*e, 4*e, 8*f), (90, 90, 90)', 4, 6],
    23: [23, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 10, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 3],
    24: [24, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 10, '<0,0,e>', 'D2', 6, ['None'], 1, '<f,0,0>', 'D6', 'P6222', 3, '(2*f, 2*f, 6*e), (90, 90, 120)', 4, 6],
    25: [25, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D2', 6, ['None'], 5, '<2*e,0,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 4],
    26: [26, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<-2*e,3*e,0>', 'D2', 6, ['None'], 5, '<0,2*e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    27: [27, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'D3', 7, ['None'], 11, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 3, 3],
    28: [28, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,f>', 'D3', 7, ['None'], 1, '<0,0,0>', 'D3', 'R32', 3, '(3.4641*e, 3.4641*e, 3*f), (90, 90, 120)', 4, 4],
    29: [29, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    30: [30, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    31: [31, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,f>', 'D6', 'P6322', 3, '(2*e, 2*e, 4*f), (90, 90, 120)', 4, 4],
    32: [32, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    33: [33, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,2*e,0>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    34: [34, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 4],
    35: [35, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,-2*e>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    36: [36, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,-2*e>', 'D3', 7, ['None'], 4, '<3*e,3*e,3*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 3],
    37: [37, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    38: [38, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,e,0>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    39: [39, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 8, '<0,e,f>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'I422', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 4, 4],
    40: [40, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,0,0>', 'D4', 8, ['None'], 1, '<0,0,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 3],
    41: [41, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,e,0>', 'D4', 8, ['None'], 1, '<2*e,2*e,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    42: [42, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    43: [43, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,0>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    44: [44, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 6, '<e,0,f>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 4, 4],
    45: [45, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    46: [46, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'T', 'F23', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    47: [47, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<2*e,3*e,0>', 'T', 10, ['None'], 1, '<0,4*e,0>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 2],
    48: [48, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    49: [49, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    50: [50, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<e,0,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    51: [51, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<0,e,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    52: [52, 'C2', 1, ['r:<0,0,1,a>', 't:<0,0,b>'], 3, '<-e,e,e>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    53: [53, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>'], 1, '<e,0.57735*e,0>', 'C3', 'p3', 2, '(2*e, 2*e), 120', 4, 3],
    54: [54, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 12, '<0,0,0>', 'T', 'T', 0, 'N/A', 4, 2],
    55: [55, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C3', 2, ['r:<0,0,1,c>', 't:<0,0,d>'], 12, '<e,0,0>', 'T', 'P213', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 5],
    56: [56, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<0,0,0>', 'O', 'O', 0, 'N/A', 4, 2],
    57: [57, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 1, '<e,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 6],
    58: [58, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 7, '<0,0,0>', 'C5', 4, ['r:<0,0,1,c>', 't:<0,0,d>'], 9, '<0,0,0>', 'I', 'I', 0, 'N/A', 4, 2],
    59: [59, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'C6', 5, ['r:<0,0,1,c>'], 1, '<0,0,0>', 'C6', 'p6', 2, '(2*e, 2*e), 120', 4, 2],
    60: [60, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    61: [61, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 3],
    62: [62, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', 6, ['None'], 3, '<e,0,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    63: [63, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D2', 6, ['None'], 3, '<2*e,e,0>', 'O', 'I4132', 3, '(8*e,8*e, 8*e), (90, 90, 90)', 3, 2],
    64: [64, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D3', 7, ['None'], 11, '<0,0,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 3, 2],
    65: [65, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D3', 7, ['None'], 1, '<0,0,0>', 'D3', 'p321', 2, '(2*e, 2*e), 120', 3, 2],
    66: [66, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 12, '<4*e,0,0>', 'D3', 7, ['None'], 4, '<3*e,3*e,3*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 3, 4],
    67: [67, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<0,0,0>', 'D4', 8, ['None'], 1, '<0,0,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    68: [68, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0.57735*e,0>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    69: [69, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<e,0,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'T', 'F23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    70: [70, 'C3', 2, ['r:<0,0,1,a>', 't:<0,0,b>'], 4, '<e,0,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    71: [71, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>'], 1, '<e,e,0>', 'C4', 'p4', 2, '(2*e, 2*e), 90', 4, 2],
    72: [72, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'C4', 3, ['r:<0,0,1,c>', 't:<0,0,d>'], 2, '<0,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 5, 4],
    73: [73, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    74: [74, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,0,0>', 'D2', 6, ['None'], 5, '<0,0,0>', 'D4', 'p4212', 2, '(2*e, 2*e), 90', 3, 2],
    75: [75, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D2', 6, ['None'], 3, '<2*e,e,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    76: [76, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', 6, ['None'], 3, '<e,0,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 3],
    77: [77, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    78: [78, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 3, 2],
    79: [79, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 2, '<0,0,0>', 'D4', 8, ['None'], 1, '<e,e,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    80: [80, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'T', 10, ['None'], 1, '<e,e,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 3, 2],
    81: [81, 'C4', 3, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<e,e,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 3, 2],
    82: [82, 'C6', 5, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 3, 2],
    83: [83, 'C6', 5, ['r:<0,0,1,a>', 't:<0,0,b>'], 1, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 2, 2],
    84: [84, 'D2', 6, ['None'], 1, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,f,0>', 'D2', 'p222', 2, '(2*e, 2*f), 90', 2, 2],
    85: [85, 'D2', 6, ['None'], 1, '<0,0,0>', 'D2', 6, ['None'], 1, '<e,f,g>', 'D2', 'F222', 3, '(4*e, 4*f, 4*g), (90, 90, 90)', 3, 3],
    86: [86, 'D2', 6, ['None'], 1, '<e,0,0>', 'D2', 6, ['None'], 5, '<0,0,f>', 'D4', 'P4222', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 2, 2],
    87: [87, 'D2', 6, ['None'], 1, '<e,0,0>', 'D2', 6, ['None'], 13, '<0,0,-f>', 'D6', 'P6222', 3, '(2*e, 2*e, 6*f), (90, 90, 120)', 2, 2],
    88: [88, 'D2', 6, ['None'], 3, '<0,e,2*e>', 'D2', 6, ['None'], 5, '<0,2*e,e>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    89: [89, 'D2', 6, ['None'], 1, '<e,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    90: [90, 'D2', 6, ['None'], 1, '<e,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,f>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    91: [91, 'D2', 6, ['None'], 1, '<0,0,2*e>', 'D3', 7, ['None'], 4, '<e,e,e>', 'D6', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    92: [92, 'D2', 6, ['None'], 3, '<2*e,e,0>', 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 'I4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    93: [93, 'D2', 6, ['None'], 1, '<e,0,0>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 1, 1],
    94: [94, 'D2', 6, ['None'], 1, '<e,0,f>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'P422', 3, '(2*e, 2*e, 2*f), (90, 90,90)', 2, 2],
    95: [95, 'D2', 6, ['None'], 5, '<e,0,f>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'I422', 3, '(2*e, 2*e, 4*f), (90, 90,90)', 2, 2],
    96: [96, 'D2', 6, ['None'], 3, '<0,e,2*e>', 'D4', 8, ['None'], 1, '<0,0,2*e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    97: [97, 'D2', 6, ['None'], 1, '<e,0,0>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    98: [98, 'D2', 6, ['None'], 1, '<e,0,f>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    99: [99, 'D2', 6, ['None'], 1, '<e,0,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    100: [100, 'D2', 6, ['None'], 1, '<e,e,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'T', 'P23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 2],
    101: [101, 'D2', 6, ['None'], 3, '<e,0,e>', 'T', 10, ['None'], 1, '<e,e,e>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    102: [102, 'D2', 6, ['None'], 3, '<2*e,e,0>', 'T', 10, ['None'], 1, '<0,0,0>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    103: [103, 'D2', 6, ['None'], 3, '<e,0,e>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    104: [104, 'D2', 6, ['None'], 3, '<2*e,e,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    105: [105, 'D3', 7, ['None'], 11, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D3', 'p312', 2, '(2*e, 2*e), 120', 1, 1],
    106: [106, 'D3', 7, ['None'], 11, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,f>', 'D3', 'P312', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    107: [107, 'D3', 7, ['None'], 1, '<0,0,0>', 'D3', 7, ['None'], 11, '<e,0.57735*e,f>', 'D6', 'P6322', 3, '(2*e, 2*e, 4*f), (90, 90, 120)', 2, 2],
    108: [108, 'D3', 7, ['None'], 4, '<e,e,e>', 'D3', 7, ['None'], 12, '<e,3*e,e>', 'O', 'P4232', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    109: [109, 'D3', 7, ['None'], 4, '<3*e,3*e,3*e>', 'D3', 7, ['None'], 12, '<e,3*e,5*e>', 'O', 'P4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    110: [110, 'D3', 7, ['None'], 4, '<e,e,e>', 'D4', 8, ['None'], 1, '<0,0,2*e>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 2],
    111: [111, 'D3', 7, ['None'], 11, '<e,0.57735*e,0>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'p622', 2, '(2*e, 2*e), 120', 1, 1],
    112: [112, 'D3', 7, ['None'], 11, '<e,0.57735*e,f>', 'D6', 9, ['None'], 1, '<0,0,0>', 'D6', 'P622', 3, '(2*e, 2*e, 2*f), (90, 90, 120)', 2, 2],
    113: [113, 'D3', 7, ['None'], 4, '<e,e,e>', 'T', 10, ['None'], 1, '<0,0,0>', 'O', 'F4132', 3, '(8*e, 8*e, 8*e), (90, 90, 90)', 1, 1],
    114: [114, 'D3', 7, ['None'], 4, '<e,e,e>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'I432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    115: [115, 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 8, ['None'], 1, '<e,e,0>', 'D4', 'p422', 2, '(2*e, 2*e), 90', 1, 1],
    116: [116, 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 8, ['None'], 1, '<e,e,f>', 'D4', 'P422', 3, '(2*e, 2*e, 2*f), (90, 90,90)', 2, 2],
    117: [117, 'D4', 8, ['None'], 1, '<0,0,e>', 'D4', 8, ['None'], 2, '<0,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    118: [118, 'D4', 8, ['None'], 1, '<0,0,e>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    119: [119, 'D4', 8, ['None'], 1, '<e,e,0>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    120: [120, 'T', 10, ['None'], 1, '<0,0,0>', 'T', 10, ['None'], 1, '<e,e,e>', 'T', 'F23', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    121: [121, 'T', 10, ['None'], 1, '<0,0,0>', 'T', 10, ['None'], 1, '<e,0,0>', 'T', 'F23', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    122: [122, 'T', 10, ['None'], 1, '<e,e,e>', 'O', 11, ['None'], 1, '<0,0,0>', 'O', 'F432', 3, '(4*e, 4*e, 4*e), (90, 90, 90)', 1, 1],
    123: [123, 'O', 11, ['None'], 1, '<0,0,0>', 'O', 11, ['None'], 1, '<e,e,e>', 'O', 'P432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1],
    124: [124, 'O', 11, ['None'], 1, '<0,0,0>', 'O', 11, ['None'], 1, '<e,0,0>', 'O', 'F432', 3, '(2*e, 2*e, 2*e), (90, 90, 90)', 1, 1]}

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
#          'C3', 2,                     ['None'], 12, '<0,0,0>',
#          'T', 'T', 0, 'N/A', 4, 2],

# ROTATION RANGE DEG
C2 = 180
C3 = 120
C4 = 90
C5 = 72
C6 = 60
RotRangeDict = {"C2": 180, "C3": 120, "C4": 90, "C5": 72, "C6": 60}
# ROTATION SETTING MATRICES
RotSetDict = {1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # identity
              2: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # 90 degrees CC on Y
              3: [[0.707107, 0.0, 0.707107], [0.0, 1.0, 0.0], [-0.707107, 0.0, 0.707107]],
              4: [[0.707107, 0.408248, 0.577350], [-0.707107, 0.408248, 0.577350], [0.0, -0.816497, 0.577350]],
              5: [[0.707107, 0.707107, 0.0], [-0.707107, 0.707107, 0.0], [0.0, 0.0, 1.0]],
              6: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],  # # 90 degrees CC on X
              7: [[1.0, 0.0, 0.0], [0.0, 0.934172, 0.356822], [0.0, -0.356822, 0.934172]],
              8: [[0.0, 0.707107, 0.707107], [0.0, -0.707107, 0.707107], [1.0, 0.0, 0.0]],
              9: [[0.850651, 0.0, 0.525732], [0.0, 1.0, 0.0], [-0.525732, 0.0, 0.850651]],
              10: [[0.0, 0.5, 0.866025], [0.0, -0.866025, 0.5], [1.0, 0.0, 0.0]],
              11: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
              12: [[0.707107, -0.408248, 0.577350], [0.707107, 0.408248, -0.577350], [0.0, 0.816497, 0.577350]],
              13: [[0.5, -0.866025, 0.0], [0.866025, 0.5, 0.0], [0.0, 0.0, 1.0]]}
identity_matrix = RotSetDict[1]


class SymEntry:
    def __init__(self, entry):
        if type(entry) == int and entry in sym_comb_dict:
            # GETTING ENTRY INFORMATION FROM sym_comb_dict
            self.entry_number = entry
            sym_comb_info = sym_comb_dict[self.entry_number]

            # ASSIGNING CLASS VARIABLES
            self.group1 = sym_comb_info[1]
            self.group1_indx = sym_comb_info[2]
            self.int_dof_group1 = sym_comb_info[3]
            self.rot_set_group1 = sym_comb_info[4]
            self.ref_frame_tx_dof_group1 = sym_comb_info[5]
            self.group2 = sym_comb_info[6]
            self.group2_indx = sym_comb_info[7]
            self.int_dof_group2 = sym_comb_info[8]
            self.rot_set_group2 = sym_comb_info[9]
            self.ref_frame_tx_dof_group2 = sym_comb_info[10]
            self.pt_grp = sym_comb_info[11]
            self.result = sym_comb_info[12]
            self.dim = sym_comb_info[13]
            self.unit_cell = sym_comb_info[14]
            self.tot_dof = sym_comb_info[15]
            self.cycle_size = sym_comb_info[16]

            if self.get_design_dim() == 0:
                self.expand_matrices = get_ptgrp_sym_op(self.get_result_design_sym())
            elif self.get_design_dim() in [2, 3]:
                self.expand_matrices = get_sg_sym_op(self.get_result_design_sym())
            else:
                raise ValueError('\nINVALID SYMMETRY ENTRY. SUPPORTED DESIGN DIMENSIONS: %s\n'
                                 % ', '.join(map(str, [0, 2, 3])))
            self.degeneracy_matrices_1, self.degeneracy_matrices_2 = self.get_degeneracy_matrices()

        else:
            raise ValueError("\nINVALID SYMMETRY ENTRY. SUPPORTED VALUES ARE: %d to %d\n" % (1, len(sym_comb_dict)))

    def get_group1_sym(self):
        return self.group1

    def get_group2_sym(self):
        return self.group2

    def get_pt_grp_sym(self):
        return self.pt_grp

    def get_rot_range_deg_1(self):
        if self.group1 in RotRangeDict:
            return RotRangeDict[self.group1]
        else:
            return 0

    def get_rot_range_deg_2(self):
        if self.group2 in RotRangeDict:
            return RotRangeDict[self.group2]
        else:
            return 0

    def get_rot_set_mat_group1(self):
        return RotSetDict[self.rot_set_group1]

    def get_ref_frame_tx_dof_group1(self):
        return self.ref_frame_tx_dof_group1

    def get_rot_set_mat_group2(self):
        return RotSetDict[self.rot_set_group2]

    def get_ref_frame_tx_dof_group2(self):
        return self.ref_frame_tx_dof_group2

    def get_result_design_sym(self):
        return self.result

    def get_design_dim(self):
        return self.dim

    def get_uc_spec_string(self):
        return self.unit_cell

    def is_internal_tx1(self):
        if 't:<0,0,b>' in self.int_dof_group1:
            return True
        else:
            return False

    def is_internal_tx2(self):
        if 't:<0,0,d>' in self.int_dof_group2:
            return True
        else:
            return False

    def get_internal_tx1(self):
        if 't:<0,0,b>' in self.int_dof_group1:
            return 't:<0,0,b>'
        else:
            return None

    def get_internal_tx2(self):
        if 't:<0,0,d>' in self.int_dof_group2:
            return 't:<0,0,d>'
        else:
            return None

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

    def get_internal_rot1(self):
        if 'r:<0,0,1,a>' in self.int_dof_group1:
            return 'r:<0,0,1,a>'
        else:
            return None

    def get_internal_rot2(self):
        if 'r:<0,0,1,c>' in self.int_dof_group2:
            return 'r:<0,0,1,c>'
        else:
            return None

    def is_ref_frame_tx_dof1(self):
        if self.ref_frame_tx_dof_group1 != '<0,0,0>':
            return True
        else:
            return False

    def is_ref_frame_tx_dof2(self):
        if self.ref_frame_tx_dof_group2 != '<0,0,0>':
            return True
        else:
            return False

    def get_ext_dof(self):
        """Return the external degrees of freedom given a symmetry entry

        Returns:
            (numpy.ndarray)
        """
        parsed_ref_frame_tx_dof1 = parse_ref_tx_dof_str_to_list(self.get_ref_frame_tx_dof_group1())
        parsed_ref_frame_tx_dof2 = parse_ref_tx_dof_str_to_list(self.get_ref_frame_tx_dof_group2())

        if parsed_ref_frame_tx_dof1 == ['0', '0', '0'] and parsed_ref_frame_tx_dof2 == ['0', '0', '0']:
            return np.empty((0, 3), float)

        e1_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof1, 'e')
        f1_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof1, 'f')
        g1_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof1, 'g')

        e2_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof2, 'e')
        f2_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof2, 'f')
        g2_var_vec = get_tx_dof_ref_frame_var_vec(parsed_ref_frame_tx_dof2, 'g')

        e2e1_diff = (np.array(e2_var_vec) - np.array(e1_var_vec)).tolist()
        f2f1_diff = (np.array(f2_var_vec) - np.array(f1_var_vec)).tolist()
        g2g1_diff = (np.array(g2_var_vec) - np.array(g1_var_vec)).tolist()

        ext_dof = []
        if e2e1_diff != [0, 0, 0]:
            ext_dof.append(e2e1_diff)

        if f2f1_diff != [0, 0, 0]:
            ext_dof.append(f2f1_diff)

        if g2g1_diff != [0, 0, 0]:
            ext_dof.append(g2g1_diff)

        return np.array(ext_dof)

    def get_degeneracy_matrices(self):
        """From the intended point group symmetry and a single component, find the degeneracy matrices that produce all
        viable configurations of the single component in the final symmetry

        Returns:
            (list[list[list[list[float]]]])
        """
        # Todo matches orient valid operators, consolidate
        # valid_pt_gp_symm_list = ["C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"]
        # here allows for D5. Is this bad? .pop('D5') The sym_entries are hardcoded...
        valid_pt_gp_symm_list = valid_subunit_number.keys()

        if self.get_group1_sym() not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.get_group2_sym() not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.get_pt_grp_sym() not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.get_design_dim() not in [0, 2, 3]:
            raise ValueError("Invalid Design Dimension")

        degeneracies = [None, None]

        for i in range(2):

            degeneracy_matrices = None

            oligomer_symmetry = self.get_group1_sym() if i == 0 else self.get_group2_sym()

            # For cages, only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            if self.get_design_dim() == 0 and i == 1:
                degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            # For layers that obey a cyclic point group symmetry
            # and that are constructed from two oligomers that both obey cyclic symmetry
            # only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            elif self.get_design_dim() == 2 and i == 1 \
                    and (self.get_group1_sym()[0], self.get_group2_sym()[0], self.get_pt_grp_sym()[0]) == ("C", "C", "C"):
                degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            # else:
            #     if oligomer_symmetry[0] == "C" and design_symmetry[0] == "C":
            #         degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            elif oligomer_symmetry in ["D3", "D4", "D6"] and self.get_pt_grp_sym() in ["D3", "D4", "D6", "T", "O"]:
                # commented out "if" statement below because all possible translations are not always being tested for D3
                # example: in entry 82, only translations along <e,0.577e> are sampled.
                # This restriction only considers 1 out of the 2 equivalent Wyckoff positions.
                # <0,e> would also have to be searched as well to remove the "if" statement below.
                # if (oligomer_symmetry, design_symmetry_pg) != ("D3", "D6"):
                if oligomer_symmetry == "D3":
                    # ROT 60 degrees about z
                    degeneracy_matrices = [[[0.5, -0.86603, 0.0], [0.86603, 0.5, 0.0], [0.0, 0.0, 1.0]]]
                elif oligomer_symmetry == "D4":
                    # 45 degrees about z; z unaffected; x goes to [1,-1,0] direction
                    degeneracy_matrices = [[[0.707107, 0.707107, 0.0], [-0.707107, 0.707107, 0.0], [0.0, 0.0, 1.0]]]
                elif oligomer_symmetry == "D6":
                    # ROT 30 degrees about z
                    degeneracy_matrices = [[[0.86603, -0.5, 0.0], [0.5, 0.86603, 0.0], [0.0, 0.0, 1.0]]]

            elif oligomer_symmetry == "D2" and self.get_pt_grp_sym() != "O":
                if self.get_pt_grp_sym() == "T":
                    degeneracy_matrices = [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]  # ROT90z

                elif self.get_pt_grp_sym() == "D4":
                    degeneracy_matrices = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]  # z,x,y and y,z,x

                elif self.get_pt_grp_sym() == "D2" or self.get_pt_grp_sym() == "D6":
                    degeneracy_matrices = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                                           [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                                           [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
                                           [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]]

            elif oligomer_symmetry == "T" and self.get_pt_grp_sym() == "T":
                degeneracy_matrices = [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]  # ROT90z

            degeneracies[i] = degeneracy_matrices

        return degeneracies


def parse_ref_tx_dof_str_to_list(ref_frame_tx_dof_string):
    return list(map(str.strip, ref_frame_tx_dof_string.strip('<>').split(',')))


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


def get_rot_matrices(step_deg, axis, rot_range_deg):
    """Return a group of rotation matrices to rotate coordinates about a specified axis in set step increments
    Args:
        step_deg (int): The number of degrees for each rotation step
        axis (str): The axis about which to rotate
        rot_range_deg (int): The range with which rotation is possible

    Returns:
        (list[list[list]])
    """
    if rot_range_deg == 0:
        # return [identity_matrix]
        return None

    rot_matrices = []
    if axis == 'x':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[1, 0, 0], [0, math.cos(rad), -1 * math.sin(rad)], [0, math.sin(rad), math.cos(rad)]])
        return rot_matrices  # [[[], [], []], ...]

    elif axis == 'y':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-1 * math.sin(rad), 0, math.cos(rad)]])
        return rot_matrices  # [[[], [], []], ...]

    elif axis == 'z':
        for angle_deg in range(0, rot_range_deg, step_deg):
            rad = math.radians(float(angle_deg))
            rot_matrices.append([[math.cos(rad), -1 * math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
        return rot_matrices  # [[[], [], []], ...]

    else:
        print("AXIS SELECTED FOR SAMPLING IS NOT SUPPORTED")
        return None


def get_degen_rotmatrices(degeneracy_matrices, rotation_matrices):
    """From a set of degeneracy matrices and a set of rotation matrices, produce the complete combination of the
    specified transformations.

    Args:
         degeneracy_matrices (list): [[[x, y, z], [x, y, z], [x, y, z]], ...] (column major)
         rotation_matrices (list[list[list]]): [[[x, y, z], [x, y, z], [x, y, z]], ...] (row major)
    Returns:
        (list[list[list[list]]]) or (list[list[2D numpy.ndarray like]])
    """
    if rotation_matrices is not None and degeneracy_matrices is not None:
        degen_rot_matrices = [[np.matmul(rot, degen_mat).tolist() for rot in rotation_matrices]
                             for degen_mat in degeneracy_matrices]
        # degen_rotmatrices = [rotation_matrices]
        # for degen in degeneracy_matrices:
        #     degen_rotmatrices.append([np.matmul(rot, degen).tolist() for rot in rotation_matrices])
        return [rotation_matrices] + degen_rot_matrices  # (list[list[2Darray]])

    elif rotation_matrices is not None and degeneracy_matrices is None:
        return [rotation_matrices]  # # (list[list[2Darray]]) Todo make list[list] in FragDock.py

    elif rotation_matrices is None and degeneracy_matrices is not None:  # is this ever true? list addition seems wrong
        return [[identity_matrix]] + [[degen_mat] for degen_mat in degeneracy_matrices]

    elif rotation_matrices is None and degeneracy_matrices is None:
        return [[identity_matrix]]
    else:
        print('This shouldn\'t be possible')
        exit()