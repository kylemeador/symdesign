import os

import numpy as np

from PathUtils import sym_op_location
from SymDesignUtils import pickle_object, unpickle
from classes.SymEntry import identity_matrix, origin
from utils.SymmetryUtils import get_ptgrp_sym_op, space_group_operation_number

sg_op_filepath = os.path.join(sym_op_location, 'spacegroups_op.txt')  # missing identity operators for most part. P1 not
with open(sg_op_filepath, "r") as f:
    sg_op_lines = f.readlines()


def get_sg_sym_op(sym_type):
    is_sg = False
    expand_uc_matrices = []
    rot_mat, tx_mat = [], []
    line_count = 0
    for line in sg_op_lines:
        if "'" in line:  # ensure only sg lines are parsed either before or after sg
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


def get_all_sg_sym_ops():
    expand_uc_matrices = []
    sg_syms = {}
    rot_mat, tx_mat = [], []
    line_count = 0
    name = None
    for line in sg_op_lines:
        if "'" in line:  # we have a new sg, add the old one, then reset all variables
            if name:
                sg_syms[name] = expand_uc_matrices
            expand_uc_matrices = []
            rot_mat, tx_mat = [], []
            line_count = 0
            number, name, *_ = line.strip().replace("'", '').split()
        # if "'%s'" % sym_type in line:
        #     is_sg = True
        if "'" not in line and ':' not in line and not line[0].isdigit():
            line_float = [float(s) for s in line.split()]
            rot_mat.append(line_float[0:3])
            tx_mat.append(line_float[-1])
            line_count += 1
            if line_count % 3 == 0:
                expand_uc_matrices.append((rot_mat, tx_mat))
                rot_mat, tx_mat = [], []

    return sg_syms


def generate_sym_op_txtfiles(sg_op_filepath):
    for symmetry_group in sg_symmetry_groups:
        sym_op_outfile_path = os.path.join(sym_op_location, 'SPACE_GROUP_SYMM_OPERATORS_TXT', '%s.txt' % symmetry_group)
        with open(sym_op_outfile_path, 'w') as f:
            sym_op = get_sg_sym_op(symmetry_group)
            for op in sym_op:
                f.write(str(op) + '\n')


def generate_sym_op_pickles(sg_op_filepath):
    for symmetry_group in sg_symmetry_groups:
        # sym_op_outfile_path = os.path.join(sym_op_location, '%s.pkl' % symmetry_group)
        sym_op = get_sg_sym_op(symmetry_group)
        pickle_object(sym_op, name=symmetry_group, out_path=pickled_dir)


if __name__ =='__main__':
    full_space_group_operator_dict = get_all_sg_sym_ops()
    chiral_space_groups = [
        'P1',  # TRICLINIC
        'P121', 'P1211', 'C121',  # MONOCLINIC
        'P222', 'P2221', 'P21212', 'P212121', 'C2221', 'C222', 'F222', 'I222', 'I212121',  # ORTHORHOMBIC
        'P4', 'P41', 'P42', 'P43', 'I4', 'I41', 'P422', 'P4212', 'P4122', 'P41212', 'P4222', 'P42212', 'P4322', 'P43212', 'I422', 'I4122',  # TETRAGONAL
        'P3', 'P31', 'P32', 'R3', 'P312', 'P321', 'P3112', 'P3121', 'P3212', 'P3221', 'R32',  # TRIGONAL
        'P6', 'P61', 'P65', 'P62', 'P64', 'P63', 'P622', 'P6122', 'P6522', 'P6222', 'P6422', 'P6322',  # HEXAGONAL
        'P23', 'F23', 'I23', 'P213', 'I213', 'P432', 'P4232', 'F432', 'F4132', 'I432', 'P4332', 'P4132', 'I4132'  # CUBIC
    ]
    sg_symmetry_groups = \
        ['P23', 'P4222', 'P321', 'P6322', 'P312', 'P622', 'F23', 'F222', 'P6222', 'I422', 'I213', 'R32', 'P4212',
         'I432', 'P4132', 'I4132', 'P3', 'P6', 'I4122', 'P4', 'C222', 'P222', 'P432', 'F4132', 'P422', 'P213',
         'F432', 'P4232']
    pickled_dir = os.path.join(sym_op_location, 'pickled')
    # os.makedirs(pickled_dir)
    space_group_operators = {}
    # for symmetry_group in sg_symmetry_groups:
    for symmetry_group in chiral_space_groups:
        # sym_op_in_path = os.path.join(sym_op_location, 'SPACE_GROUP_SYMM_OPERATORS', '%s.pkl' % symmetry_group)
        # sym_op = unpickle(sym_op_in_path)
        if 'R' in symmetry_group:
            _symmetry_group = symmetry_group + ':H'  # have to add hexagonal notation found in spacegroup_op.txt
        else:
            _symmetry_group = symmetry_group
        sym_op = full_space_group_operator_dict[_symmetry_group]
        rotations, translations = zip(*sym_op)
        rotations = np.array(rotations)
        translations = np.array(translations)
        number_of_rotations = len(rotations)
        number_of_operators = space_group_operation_number.get(symmetry_group, None)
        if number_of_operators:
            if number_of_operators != number_of_rotations:  # insert the idenity matrix in the front
                rotations = np.insert(rotations, 0, identity_matrix, axis=0)
                translations = np.insert(translations, 0, origin, axis=0)
                # print('%s incorrect number of operators. Adding identity' % symmetry_group)
            else:
                pass
                # print('%s found the correct number of operators' % symmetry_group)
        else:
            if np.all(rotations[0] == identity_matrix):
                print('\'%s\': %s | NO operator number found. Found IDENTITY'
                      % (symmetry_group, number_of_rotations))
            else:
                print('\'%s\': %s | NO operator number found. Adding identity because no match'
                      % (symmetry_group, number_of_rotations + 1))
                rotations = np.insert(rotations, 0, identity_matrix, axis=0)
                translations = np.insert(translations, 0, origin, axis=0)
        # print(rotations)
        # print(translations)
        # exit()
        space_group_operators[symmetry_group] = (rotations, translations[:, None, :])
        # sym_op_outfile_path = os.path.join(sym_op_location, '%s.pkl' % symmetry_group)
        # pickle_object(sym_op, name=symmetry_group, out_path=pickled_dir)
    # print('Last spacegroup found:', space_group_operators[symmetry_group])
    continue_ = input('Save these results? Yes is "Enter". Ctrl-C is quit')
    pickle_object(space_group_operators, name='space_group_operators', out_path=pickled_dir)
    # sym_op_outfile = open(sym_op_outfile_path, "w")
    # pickle.dump(sym_op, sym_op_outfile)
    # sym_op_outfile.close()

    pg_symmetry_groups = ['D2', 'D3', 'D4', 'D5', 'D6', 'T', 'O', 'I']
    point_group_operators = {}
    for symmetry in pg_symmetry_groups:
        expand_matrix = get_ptgrp_sym_op(symmetry)
        rotations = np.array(expand_matrix)
        point_group_operators[symmetry] = rotations

    # print('Last pointgroup found:', point_group_operators[symmetry])
    continue_ = input('Save these results? Yes is "Enter". Ctrl-C is quit')
    pickle_object(point_group_operators, name='point_group_operators', out_path=pickled_dir)
    # print({notation: notation.replace(' ', '') for notation in hg_notation})
    # generate_sym_op_pickles(os.path.join(sym_op_location, 'spacegroups_op.txt'))
