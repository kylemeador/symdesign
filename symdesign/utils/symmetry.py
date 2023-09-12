from __future__ import annotations

import math
import os
from typing import List, Union, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from . import path as putils, pickle_object, unpickle
except ImportError:
    from symdesign.utils import path as putils, pickle_object, unpickle

CRYST = 'CRYST'
chiral_space_groups = [
    'P1',  # TRICLINIC
    'P121', 'P1211', 'C121',  # MONOCLINIC
    'P222', 'P2221', 'P21212', 'P212121', 'C2221', 'C222', 'F222', 'I222', 'I212121',  # ORTHORHOMBIC
    'P4', 'P41', 'P42', 'P43', 'I4', 'I41', 'P422', 'P4212', 'P4122', 'P41212', 'P4222', 'P42212', 'P4322', 'P43212',
    'I422', 'I4122',  # TETRAGONAL
    'P3', 'P31', 'P32', 'R3', 'P312', 'P321', 'P3112', 'P3121', 'P3212', 'P3221', 'R32',  # TRIGONAL
    'P6', 'P61', 'P65', 'P62', 'P64', 'P63', 'P622', 'P6122', 'P6522', 'P6222', 'P6422', 'P6322',  # HEXAGONAL
    'P23', 'F23', 'I23', 'P213', 'I213', 'P432', 'P4232', 'F432', 'F4132', 'I432', 'P4332', 'P4132', 'I4132'  # CUBIC
]
nanohedra_space_groups = {'P23', 'P4222', 'P321', 'P6322', 'P312', 'P622', 'F23', 'F222', 'P6222', 'I422', 'I213',
                          'R32', 'P4212', 'I432', 'P4132', 'I4132', 'P3', 'P6', 'I4122', 'P4', 'C222', 'P222', 'P432',
                          'F4132', 'P422', 'P213', 'F432', 'P4232'}
space_group_cryst1_fmt_dict = {
    'P1': 'P 1',  # TRICLINIC
    'P121': 'P 1 2 1', 'P1211': 'P 1 21 1', 'C121': 'C 1 2 1',  # MONOCLINIC
    'P222': 'P 2 2 2', 'P2221': 'P 2 2 21', 'P21212': 'P 21 21 2', 'P212121': 'P 21 21 21', 'C2221': 'C 2 2 21',
    'C222': 'C 2 2 2', 'F222': 'F 2 2 2', 'I222': 'I 2 2 2', 'I212121': 'I 21 21 21',  # ORTHORHOMBIC
    'P4': 'P 4', 'P41': 'P 41', 'P42': 'P 42', 'P43': 'P 43', 'I4': 'I 4', 'I41': 'I 41', 'P422': 'P 4 2 2',
    'P4212': 'P 4 21 2', 'P4122': 'P 41 2 2', 'P41212': 'P 41 21 2', 'P4222': 'P 42 2 2', 'P42212': 'P 42 21 2',
    'P4322': 'P 43 2 2', 'P43212': 'P 43 21 2', 'I422': 'I 4 2 2', 'I4122': 'I 41 2 2',  # TETRAGONAL
    'P3': 'P 3', 'P31': 'P 31', 'P32': 'P 32', 'R3': 'R 3', 'P312': 'P 3 1 2', 'P321': 'P 3 2 1', 'P3112': 'P 31 1 2',
    'P3121': 'P 31 2 1', 'P3212': 'P 32 1 2', 'P3221': 'P 32 2 1', 'R32': 'R 3 2',  # TRIGONAL
    'P6': 'P 6', 'P61': 'P 61', 'P65': 'P 65', 'P62': 'P 62', 'P64': 'P 64', 'P63': 'P 63',
    'P622': 'P 6 2 2', 'P6122': 'P 61 2 2', 'P6522': 'P 65 2 2', 'P6222': 'P 62 2 2', 'P6422': 'P 64 2 2',
    'P6322': 'P 63 2 2',  # HEXAGONAL
    'P23': 'P 2 3', 'F23': 'F 2 3', 'I23': 'I 2 3', 'P213': 'P 21 3', 'I213': 'I 21 3', 'P432': 'P 4 3 2',
    'P4232': 'P 42 3 2', 'F432': 'F 4 3 2', 'F4132': 'F 41 3 2', 'I432': 'I 4 3 2', 'P4332': 'P 43 3 2',
    'P4132': 'P 41 3 2', 'I4132': 'I 41 3 2'}  # CUBIC
space_group_hm_to_simple_dict = {v: k for k, v in space_group_cryst1_fmt_dict.items()}
layer_groups = [
    'p1'
    'p2', 'p21', 'pg', 'p222', 'p2221', 'p22121',
    'c222',
    'p3', 'p312', 'p321',
    'p4', 'p422', 'p4212',
    'p6', 'p622'
]
nanohedra_layer_groups = ['p222', 'c222', 'p3', 'p312', 'p321', 'p4', 'p422', 'p4212', 'p6', 'p622']
layer_group_cryst1_fmt_dict = {
    'p1': 'P 1',
    'p2': 'P 2', 'p21': 'P 21', 'pg': 'C 2', 'p222': 'P 2 2 2', 'p2221': 'P 2 2 21', 'p22121': 'P 2 21 21',
    'c222': 'C 2 2 2',
    'p3': 'P 3', 'p312': 'P 3 1 2', 'p321': 'P 3 2 1',
    'p4': 'P 4', 'p422': 'P 4 2 2', 'p4212': 'P 4 21 2',
    'p6': 'P 6', 'p622': 'P 6 2 2'
}
layer_group_hm_to_simple_dict = {v: k for k, v in layer_group_cryst1_fmt_dict.items()}
layer_group_entry_numbers = {2, 4, 10, 12, 17, 19, 20, 21, 23,
                             27, 29, 30, 37, 38, 42, 43, 53, 59, 60, 64, 65, 68,
                             71, 78, 74, 78, 82, 83, 84, 89, 93, 97, 105, 111, 115}
space_group_number_operations = \
    {'P1': 1, 'P121': 2, 'P1211': 2, 'C121': 4, 'P2221': 4, 'P21212': 4, 'P212121': 4, 'C2221': 8, 'I222': 8,
     'I212121': 8, 'P41': 4, 'P42': 4, 'P43': 4, 'I4': 8, 'I41': 8, 'P4122': 8, 'P41212': 8, 'P42212': 8, 'P4322': 8,
     'P43212': 8, 'P31': 3, 'P32': 3, 'R3': 9, 'P3112': 6, 'P3121': 6, 'P3212': 6, 'P3221': 6, 'P61': 6, 'P65': 6,
     'P62': 6, 'P64': 6, 'P63': 6, 'P6122': 12, 'P6522': 12, 'P6422': 12, 'I23': 24, 'P4332': 24,  # above added 5/3/22
     'P23': 12, 'P4222': 8, 'P321': 6, 'P6322': 12, 'P312': 12, 'P622': 12, 'F23': 48, 'F222': 16, 'P6222': 12,
     'I422': 16, 'I213': 24, 'R32': 6, 'P4212': 8, 'I432': 48, 'P4132': 24, 'I4132': 48, 'P3': 3, 'P6': 6,
     'I4122': 16, 'P4': 4, 'C222': 8, 'P222': 4, 'P213': 12, 'F4132': 96, 'P422': 8, 'P432': 24, 'F432': 96,
     'P4232': 24}
for sg, z_number in list(space_group_number_operations.items()):
    space_group_number_operations[space_group_cryst1_fmt_dict[sg]] = z_number

cubic_point_groups = ['T', 'O', 'I']
point_group_symmetry_operators: dict[str, np.ndarray] = \
    unpickle(putils.point_group_symmetry_operator_location)
point_group_symmetry_operatorsT: dict[str, np.ndarray] = \
    unpickle(putils.point_group_symmetry_operatorT_location)
"""A mapping of the point group name (in Hermann-Mauguin notation) to the corresponding symmetry operators
Formatted as {'symmetry': rotations[N, 3, 3], ...} where the rotations are pre-transposed to match requirements of 
np.matmul(coords, rotation)"""
space_group_symmetry_operators: dict[str, np.ndarray] = \
    unpickle(putils.space_group_symmetry_operator_location)
space_group_symmetry_operatorsT: dict[str, np.ndarray] = \
    unpickle(putils.space_group_symmetry_operatorT_location)
"""A mapping of the space group name (in Hermann-Mauguin notation) to the corresponding symmetry operators
Formatted as {'symmetry': (rotations[N, 3, 3], translations[N, 1, 3]), ...} where the rotations are pre-transposed to 
match requirements of np.matmul(coords, rotation)"""
# Todo modify to use only this or all_sym_entry_dict
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T', 'O32': 'O', 'O42': 'O', 'O43': 'O',
                       'I23': 'I', 'I25': 'I', 'I35': 'I', 'T23': 'T', 'O23': 'O', 'O24': 'O', 'O34': 'O',
                       'T': 'T', 'T:{C2}': 'T', 'T:{C3}': 'T',
                       'T:{C2}{C3}': 'T', 'T:{C3}{C2}': 'T', 'T:{C3}{C3}': 'T',
                       'O': 'O', 'O:{C2}': 'O', 'O:{C3}': 'O', 'O:{C4}': 'O',
                       'O:{C2}{C3}': 'O', 'O:{C2}{C4}': 'O', 'O:{C3}{C4}': 'O',
                       # 'O:234': 'O', 'O:324': 'O', 'O:342': 'O', 'O:432': 'O', 'O:423': 'O', 'O:243': 'O',
                       # 'O:{C2}{C3}{C4}': 'O', 'O:{C3}{C2}{C4}': 'O', 'O:{C3}{C4}{C2}': 'O', 'O:{C4}{C3}{C2}': 'O',
                       # 'O:{C4}{C2}{C3}': 'O', 'O:{C2}{C4}{C3}': 'O',
                       'O:{C3}{C2}': 'O', 'O:{C4}{C2}': 'O', 'O:{C4}{C3}': 'O',
                       'I': 'I', 'I:{C2}': 'I', 'I:{C3}': 'I', 'I:{C5}': 'I',
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I',
                       # 'I:235': 'I', 'I:325': 'I', 'I:352': 'I', 'I:532': 'I', 'I:253': 'I', 'I:523': 'I',
                       # 'I:{C2}{C3}{C5}': 'I', 'I:{C3}{C2}{C5}': 'I', 'I:{C3}{C5}{C2}': 'I', 'I:{C5}{C3}{C2}': 'I',
                       # 'I:{C2}{C5}{C3}': 'I', 'I:{C5}{C2}{C3}': 'I',
                       'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'C5': 'C5', 'C6': 'C6',
                       'D2': 'D2', 'D3': 'D3', 'D4': 'D4', 'D5': 'D5', 'D6': 'C6',
                       # layer groups
                       # 'p6', 'p4', 'p3', 'p312', 'p4121', 'p622',
                       # space groups  # Todo
                       # 'cryst': 'cryst'
                       }
all_sym_entry_dict = {'T': {'C2': {'C3': 5}, 'C3': {'C2': 5, 'C3': 54}, 'T': 200},
                      'O': {'C2': {'C3': 7, 'C4': 13}, 'C3': {'C2': 7, 'C4': 56}, 'C4': {'C2': 13, 'C3': 56}, 'O': 210},
                      'I': {'C2': {'C3': 9, 'C5': 16}, 'C3': {'C2': 9, 'C5': 58}, 'C5': {'C2': 16, 'C3': 58}, 'I': 220}}
MAX_SYMMETRY = 6
rotation_range = {f'C{int(i)}': 360 / i for i in map(float, range(1, MAX_SYMMETRY + 1))}

# All rotational comments below are described according to a vector emanating from the origin on the specified axis
# The dihedral angles (in radians) are sourced from https://en.wikipedia.org/wiki/Table_of_polyhedron_dihedral_angles
setting_matrices = {
    1: np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    # Identity
    2: np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
    # 90 degrees CW on Y
    3: np.array([[.707107, 0., .707107], [0., 1.0, 0.], [-.707107, 0., .707107]]),
    # 3: Rotation.from_euler('y', -math.pi/2),  # Todo test
    # 45 degrees CW on Y, which is 2-fold axis in O
    # 4: np.array([[0.707107, 0.408248, 0.577350], [-0.707107, 0.408248, 0.577350], [0.0, -0.816497, 0.577350]]),
    4: Rotation.from_euler('xz', (-math.acos(-1/3) / 2, -math.pi / 4)).as_matrix(),
    # ~54.73 degrees CCW on X, 45 degrees CCW on Z, which is X,Y,Z body diagonal or 3-fold axis in T, O
    5: np.array([[.707107, .707107, 0.], [-.707107, .707107, 0.0], [0., 0., 1.]]),
    # 5: Rotation.from_euler('z', -math.pi/4),  # Todo test
    # 45 degrees CCW on Z
    6: np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]),
    # 90 degrees CW on X
    # 7: np.array([[1., 0., 0.], [0., .934172, .356822], [0., -.356822, .934172]]),
    7: Rotation.from_euler('x', (math.pi - math.acos(-math.sqrt(5)/3)) / 2).as_matrix(),
    # ~20.91 degrees CCW on X which is 3-fold axis in I (2-fold is positive Z)
    8: np.array([[0., .707107, .707107], [0., -.707107, .707107], [1., 0., 0.]]),
    # 8: Rotation.from_euler('yz', (-math.pi/2, -math.pi*3/4)),  # Todo test
    # 90 degrees CCW on Y, 135 degrees CCW on Z, which is 45 degree X,Y plane diagonal in D4
    # 9: np.array([[.850651, 0., .525732], [0., 1., 0.], [-0.525732, 0., .850651]]),
    9: Rotation.from_euler('y', (math.pi - math.acos(-math.sqrt(5) / 5)) / 2).as_matrix(),
    # ~31.72 degrees CW on Y which is 5-fold axis in I (2-fold is positive Z)
    10: np.array([[0., .5, .866025], [0., -.866025, .5], [1., 0., 0.]]),
    # 10: Rotation.from_euler('yz', (-math.pi/2, -math.pi/5)),  # Todo test
    # 90 degrees CCW on Y, 150 degrees CCW on Z, which is 60 degree X,Y plane diagonal in D6
    11: np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
    # 90 degrees CCW on Z
    # 12: np.array([[0.707107, -0.408248, 0.577350], [0.707107, 0.408248, -0.577350], [0.0, 0.816497, 0.577350]]),
    12: Rotation.from_euler('xz', (math.acos(-1/3) / 2, math.pi / 4)).as_matrix(),
    # ~54.73 degrees CW on X, 45 degrees CW on Z, which is X,-Y,Z body diagonal or opposite 3-fold in T, O
    13: np.array([[.5, -.866025, 0.], [.866025, .5, 0.], [0., 0., 1.]])
    # 13: Rotation.from_euler('z', math.pi/3),  # Todo test
    # 60 degrees CW on Z
    }
inv_setting_matrices = {key: np.linalg.inv(setting_matrix) for key, setting_matrix in setting_matrices.items()}
flip_x_matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # rot 180x
flip_y_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])  # rot 180y
point_group_degeneracy_matrices = {
    'T': 6,
}
sub_symmetries = {'C1': ['C1'],
                  'C2': ['C1', 'C2'],
                  'C3': ['C1', 'C3'],
                  'C4': ['C1', 'C2', 'C4'],
                  'C5': ['C1', 'C5'],
                  'C6': ['C1', 'C2', 'C3', 'C6'],
                  # all dihedral have C2 operator orthogonal to main symmetry axis
                  'D2': ['C1', 'C2'],
                  'D3': ['C1', 'C3', 'C2'],
                  'D4': ['C1', 'C2', 'C4'],
                  'D5': ['C1', 'C5', 'C2'],
                  'D6': ['C1', 'C2', 'C3', 'C6'],
                  'T': ['C1', 'C2', 'C3'],  # , 'T'
                  'O': ['C1', 'C2', 'C3', 'C4'],  # , 'O'
                  'I': ['C1', 'C2', 'C3', 'C5'],  # , 'I'
                  }


def multicomponent_by_number(number):
    return [multiplier * number for multiplier in range(1, 10)]


# Todo make dynamic based on max_sym
valid_subunit_number = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6,
                        'D2': 4, 'D3': 6, 'D4': 8, 'D5': 10, 'D6': 12,
                        'T': 12, 'O': 24, 'I': 60}
subunit_number_to_symmetry = dict(zip(valid_subunit_number.values(), valid_subunit_number.keys()))
valid_symmetries = list(valid_subunit_number.keys())
multicomponent_valid_subunit_number = \
    {sym: multicomponent_by_number(copy_number) for sym, copy_number in valid_subunit_number.items()}


def generate_cryst1_record(dimensions: list[float], space_group: str) -> str:
    """Format the CRYST1 record from specified unit cell dimensions and space group for a .pdb file

    Args:
        dimensions: Containing a, b, c (Angstroms) alpha, beta, gamma (degrees)
        space_group: The space group of interest in compact format
    Returns:
        The CRYST1 record
    """
    if space_group in space_group_cryst1_fmt_dict or space_group in space_group_hm_to_simple_dict:
        formatted_space_group = space_group_cryst1_fmt_dict.get(space_group, space_group)
    elif space_group in layer_group_cryst1_fmt_dict or space_group in layer_group_hm_to_simple_dict:
        formatted_space_group = layer_group_cryst1_fmt_dict.get(space_group, space_group)
        dimensions[2] = 1.
        dimensions[4] = dimensions[5] = 90.
    else:
        raise ValueError(
            f"The space group '{space_group}' isn't supported")

    try:
        return 'CRYST1{dim[0]:9.3f}{dim[1]:9.3f}{dim[2]:9.3f}{dim[3]:7.2f}{dim[4]:7.2f}{dim[5]:7.2f} {sg:<11s}{z:4d}\n'\
            .format(dim=dimensions, sg=formatted_space_group, z=space_group_number_operations[space_group])
    except TypeError:  # NoneType was passed
        raise ValueError(
            f"Can't {generate_cryst1_record.__name__} without passing 'dimensions'")


def get_central_asu(pdb, uc_dimensions, design_dimension):  # Todo remove from FragDock then Depreciate
    asu_com_frac = cart_to_frac(pdb.center_of_mass, uc_dimensions)

    # array_range = np.full(21, 1)
    # asu_com_frac_array = array_range[:, np.newaxis] * asu_com_frac
    shift_range = np.arange(-10, 11)
    # shift_zeros = np.zeros(len(shift_range))
    shift = np.array([shift_range, shift_range, shift_range]).T
    # y_shift = np.array([shift_zeros, shift_range, shift_zeros]).T
    # z_shift = np.array([shift_zeros, shift_zeros, shift_range]).T

    asu_com_shifted_frac_array = asu_com_frac + shift
    # asu_com_y_shifted_frac_array = asu_com_frac + y_shift
    # asu_com_z_shifted_frac_array = asu_com_frac + z_shift
    asu_com_shifted_cart_array = frac_to_cart(asu_com_shifted_frac_array, uc_dimensions)
    # asu_com_y_shifted_cart_array = frac_to_cart(asu_com_y_shifted_frac_array, uc_dimensions)
    # asu_com_z_shifted_cart_array = frac_to_cart(asu_com_z_shifted_frac_array, uc_dimensions)
    min_shift_idx = np.abs(asu_com_shifted_cart_array).argmin(axis=0)
    # y_min_shift_idx = abs(asu_com_y_shifted_cart_array).argmin(axis=0)[1]
    # z_min_shift_idx = abs(asu_com_z_shifted_cart_array).argmin(axis=0)[2]

    xyz_min_shift_vec_frac = asu_com_shifted_frac_array[min_shift_idx, [0, 1, 2]]
    #                        asu_com_y_shifted_frac_array[y_min_shift_idx],
    #                        asu_com_z_shifted_frac_array[z_min_shift_idx]]

    if design_dimension == 2:
        xyz_min_shift_vec_frac[2] = 0

    if xyz_min_shift_vec_frac.sum() == 0:
        return pdb
    else:
        xyz_min_shifted_pdb_asu_coords_frac = cart_to_frac(pdb.coords, uc_dimensions) + xyz_min_shift_vec_frac
        pdb.coords = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)
        # xyz_min_shifted_asu_pdb = copy.copy(pdb)
        # xyz_min_shifted_asu_pdb.set_coords(xyz_min_shifted_pdb_asu_coords_cart)

        # xyz_min_shifted_cart_tx = frac_to_cart(xyz_min_shift_vec_frac, uc_dimensions)
        # xyz_min_shifted_asu_pdb = copy.copy(pdb)
        # xyz_min_shifted_asu_pdb.set_coords(pdb.coords + xyz_min_shifted_cart_tx)
        # return pdb.get_transformed_copy(translation=xyz_min_shifted_cart_tx)
        # xyz_min_shifted_asu_pdb.set_atom_coordinates(xyz_min_shifted_pdb_asu_coords_cart)
        return pdb


def get_ptgrp_sym_op(sym_type: str,
                     expand_matrix_dir: Union[str, bytes] = os.path.join(putils.sym_op_location,
                                                                         'POINT_GROUP_SYMM_OPERATORS')) -> List[List]:
    """Get the symmetry operations for a specified point group oriented in the canonical orientation

    Args:
        sym_type: The name of the symmetry
        expand_matrix_dir: The disk location of a directory with symmetry name labeled expand matrices
    Returns:
        The rotation matrices to perform point group expansion
    """
    expand_matrix_filepath = os.path.join(expand_matrix_dir, f'{sym_type}.txt')
    with open(expand_matrix_filepath, 'r') as f:
        line_count = 0
        rotation_matrices = []
        mat = []
        for line in f.readlines():
            line = line.split()
            if len(line) == 3:
                mat.append([float(s) for s in line])
                line_count += 1
                if line_count % 3 == 0:
                    rotation_matrices.append(mat)
                    mat = []

        return rotation_matrices


# def get_sg_sym_op(sym_type, space_group_operator_dir=os.path.join(putils.sym_op_location,
#                                                                   "SPACE_GROUP_SYMM_OPERATORS")):
#     """Get the symmetry operations for a specified space group oriented in the canonical orientation
#     Returns:
#         (list[tuple[list[list], list]])
#     """
#     sg_op_filepath = os.path.join(space_group_operator_dir, '%s.pickle' % sym_type.upper())
#     with open(sg_op_filepath, 'rb') as sg_op_file:
#         sg_sym_op = pickle.load(sg_op_file)
#
#     return sg_sym_op


def get_sg_sym_op(sym_type):
    is_sg = False
    expand_uc_matrices = []
    rot_mat, tx_mat = [], []
    line_count = 0
    for line in sg_op_lines:
        if "'" in line:  # Ensure only sg lines are parsed either before or after sg
            is_sg = False
        if "'%s'" % sym_type in line:
            is_sg = True
        if is_sg and "'" not in line and ":" not in line and not line[0].isdigit():
            line_float = [float(s) for s in line.split()]
            rot_mat.append(line_float[0:3])
            tx_mat.append(line_float[-1])
            line_count += 1
            if line_count % 3 == 0:
                expand_uc_matrices.append((rot_mat, tx_mat))
                rot_mat, tx_mat = [], []

    return expand_uc_matrices


def get_all_sg_sym_ops(space_group_operation_lines: Sequence[str]):
    expand_uc_matrices = []
    sg_syms = {}
    rot_mat, tx = [], []
    line_count = 0
    name = None
    for line in space_group_operation_lines:
        if "'" in line:  # New sg, add the old one, then reset all variables
            if name:
                sg_syms[name] = expand_uc_matrices
            expand_uc_matrices = []
            rot_mat, tx = [], []
            line_count = 0
            number, name, *_ = line.strip().replace("'", '').split()
        # if "'%s'" % sym_type in line:
        #     is_sg = True
        if "'" not in line and ':' not in line and not line[0].isdigit():
            line_float = [float(s) for s in line.split()]
            rot_mat.append(line_float[0:3])
            tx.append(line_float[-1])
            line_count += 1
            if line_count % 3 == 0:
                expand_uc_matrices.append((rot_mat, tx))
                rot_mat, tx = [], []

    return sg_syms


def generate_sym_op_txtfiles():
    for group in nanohedra_space_groups:
        sym_op_outfile_path = os.path.join(putils.sym_op_location, 'SPACE_GROUP_SYMM_OPERATORS_TXT',
                                           f'{symmetry_group}.txt')
        with open(sym_op_outfile_path, 'w') as f:
            symmetry_op = get_sg_sym_op(group)
            for op in symmetry_op:
                f.write(str(op) + '\n')


identity_matrix = setting_matrices[1]
origin = np.array([0., 0., 0.])
if __name__ == '__main__':
    print('\nRunning this script creates the symmetry operators fresh from text files. '
          'If all is correct, two prompts should appear and their corresponding file names\n')
    # Missing identity operators for most part. P1 not
    sg_op_filepath = os.path.join(putils.sym_op_location, 'spacegroups_op.txt')
    with open(sg_op_filepath, "r") as f:
        sg_op_lines = f.readlines()
    full_space_group_operator_dict = get_all_sg_sym_ops(sg_op_lines)

    space_group_operators = {}
    for symmetry_group in chiral_space_groups:
        if 'R' in symmetry_group:
            _symmetry_group = symmetry_group + ':H'  # Have to add hexagonal notation found in spacegroup_op.txt
        else:
            _symmetry_group = symmetry_group

        sym_op = full_space_group_operator_dict[_symmetry_group]
        rotations, translations = zip(*sym_op)
        rotations = np.array(rotations)
        translations = np.array(translations)
        number_of_rotations = len(rotations)
        number_of_operators = space_group_number_operations.get(symmetry_group, None)
        if number_of_operators:
            if number_of_operators != number_of_rotations:  # insert the idenity matrix in the front
                rotations = np.insert(rotations, 0, identity_matrix, axis=0)
                translations = np.insert(translations, 0, origin, axis=0)
                # print(f'{symmetry_group} incorrect number of operators. Adding identity')
            else:
                pass
                # print(f'{symmetry_group} found the correct number of operators')
        else:
            if np.all(rotations[0] == identity_matrix):
                print(f'"{symmetry_group}": {number_of_rotations} | NO operator number found. Found IDENTITY')
            else:
                print(f'"{symmetry_group}": {number_of_rotations + 1} | NO operator number found. Adding identity '
                      f'because no match')
                rotations = np.insert(rotations, 0, identity_matrix, axis=0)
                translations = np.insert(translations, 0, origin, axis=0)

        space_group_operators[symmetry_group] = (rotations, translations)

    continue1 = input('Save these results? Yes hits "Enter". Ctrl-C is quit: ')
    space_group_file = pickle_object(space_group_operators,
                                     out_path=putils.space_group_symmetry_operator_location)
    print(space_group_file)
    # Pre-transpose the rotation matrix so program operation doesn't have to
    space_group_operators_t = {}
    for symmetry_group, operators in space_group_operators.items():
        rotations, translations = operators
        rotations = rotations.swapaxes(-1, -2)
        space_group_operators_t[symmetry_group] = (rotations, translations[:, None, :])

    space_group_file_t = pickle_object(space_group_operators_t,
                                       out_path=putils.space_group_symmetry_operatorT_location)
    print(space_group_file_t)

    pg_symmetry_groups = ['C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'T', 'O', 'I']
    # Todo for cyclic and dihedral use
    #  point_group_operators = {}
    #  for symmetry in pg_symmetry_groups:
    #      if symmetry not in cubic_point_groups:
    #          cyclic = get_rot_matrices(rotation_range[symmetry])
    #          point_group_operators[symmetry] = cyclic
    #          dihedral = make_rotations_degenerate(rotations=cyclic, degeneracies=[identity_matrix, flip_x_matrix])
    #          point_group_operators[symmetry] = dihedral
    #      elif symmetry.startswith('D'):
    #          pass
    #      else:
    #          rotations = np.array(get_ptgrp_sym_op(symmetry))  # Todo generate from rotations...
    #          point_group_operators[symmetry] = rotations
    # Each rotation operations will have the shape (N, 3, 3)
    point_group_operators = {'C1': identity_matrix[None, :, :]}  # Expand to shape (1, 3, 3)
    for symmetry in pg_symmetry_groups:
        # These are pretransposed in their files...
        rotations = np.array(get_ptgrp_sym_op(symmetry))
        point_group_operators[symmetry] = rotations

    continue2 = input('Save these results? Yes hits "Enter". Ctrl-C is quit: ')
    point_group_file = pickle_object(point_group_operators,
                                     out_path=putils.point_group_symmetry_operatorT_location)
    print(point_group_file)
    # Pre-transpose the rotation matrix so routine program operation doesn't have to
    point_group_operators_t = {}
    for symmetry_group, rotations in point_group_operators.items():
        point_group_operators_t[symmetry_group] = rotations.swapaxes(-1, -2)
    point_group_file_t = pickle_object(point_group_operators_t,
                                       out_path=putils.point_group_symmetry_operator_location)
    print(point_group_file_t)
