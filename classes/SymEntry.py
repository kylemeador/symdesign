import math

import numpy as np

from utils.SymmetryUtils import valid_subunit_number, get_ptgrp_sym_op, get_sg_sym_op

# Copyright 2020 Joshua Laniado and Todd O. Yeates.
__author__ = "Joshua Laniado and Todd O. Yeates"
__copyright__ = "Copyright 2020, Nanohedra"
__version__ = "1.0"

# SYMMETRY COMBINATION MATERIAL TABLE (T.O.Y and J.L, 2020)

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
    94: [94, 'D2', 6, ['None'], 1, '<e,0,f>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'P422', 3, '(2*e, 2*e, 2*f), (90, 90, 90)', 2, 2],
    95: [95, 'D2', 6, ['None'], 5, '<e,0,f>', 'D4', 8, ['None'], 1, '<0,0,0>', 'D4', 'I422', 3, '(2*e, 2*e, 4*f), (90, 90, 90)', 2, 2],
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
rotation_range = {"C2": 180, "C3": 120, "C4": 90, "C5": 72, "C6": 60}
# ROTATION SETTING MATRICES
setting_matrices = {1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # identity
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
flip_x_matrix = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # rot 180x
flip_y_matrix = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]  # rot 180y
identity_matrix = setting_matrices[1]


class SymEntry:
    def __init__(self, entry):
        sym_entry = sym_comb_dict.get(entry)
        try:
            _, self.group1, self.group1_indx, self.int_dof_group1, self.rot_set_group1, self.ref_frame_tx_dof1, \
                self.group2, self.group2_indx, self.int_dof_group2, self.rot_set_group2, self.ref_frame_tx_dof2, \
                self.pt_grp, self.result, self.dim, self.unit_cell, self.tot_dof, self.cycle_size = sym_entry
        except TypeError:
            raise ValueError("\nINVALID SYMMETRY ENTRY \'%s\'. SUPPORTED VALUES ARE: %d to %d\n"
                             % (entry, 1, len(sym_comb_dict)))
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

    @property
    def group1_sym(self):
        return self.group1

    @property
    def group2_sym(self):
        return self.group2

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

    @property
    def resulting_symmetry(self):
        """The final symmetry of the symmetry combination material"""
        return self.result

    @property
    def dimension(self):
        return self.dim

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
            return self._internal_tx1
        except AttributeError:
            if 't:<0,0,b>' in self.int_dof_group1:
                self._internal_tx1 = True
            else:
                self._internal_tx1 = False
        return self._internal_tx1

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
        # valid_pt_gp_symm_list = ["C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"]
        # here allows for D5. Is this bad? .pop('D5') The sym_entries are hardcoded...
        valid_pt_gp_symm_list = list(valid_subunit_number.keys())
        valid_pt_gp_symm_list.remove('D5')

        if self.group1_sym not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.group2_sym not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.point_group_sym not in valid_pt_gp_symm_list:
            raise ValueError("Invalid Point Group Symmetry")

        if self.dimension not in [0, 2, 3]:
            raise ValueError("Invalid Design Dimension")

        degeneracies = []
        for i in range(2):
            oligomer_symmetry = self.group1_sym if i == 0 else self.group2_sym

            degeneracy_matrices = None
            # For cages, only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            if self.dimension == 0 and i == 1:
                degeneracy_matrices = [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]  # ROT180y

            # For layers that obey a cyclic point group symmetry
            # and that are constructed from two oligomers that both obey cyclic symmetry
            # only one of the two oligomers need to be flipped. By convention we flip oligomer 2.
            elif self.dimension == 2 and i == 1 and \
                    (self.group1_sym[0], self.group2_sym[0], self.point_group_sym[0]) == ('C', 'C', 'C'):
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

            degeneracies.append(np.array(degeneracy_matrices))

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


def get_rot_matrices(step_deg, axis, rot_range_deg):
    """Return a group of rotation matrices to rotate coordinates about a specified axis in set step increments
    Args:
        step_deg (int): The number of degrees for each rotation step
        axis (str): The axis about which to rotate
        rot_range_deg (int): The range with which rotation is possible
    Returns:
        (np.array): The rotation matrics with shape (rotations, 3, 3) # list[list[list]])
    """
    if rot_range_deg == 0:
        return None

    rot_matrices = []
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
        return None

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
