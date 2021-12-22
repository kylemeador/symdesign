import copy
import os
import pickle

import numpy as np

from PathUtils import sym_op_location


sg_cryst1_fmt_dict = {'F222': 'F 2 2 2', 'P6222': 'P 62 2 2', 'I4132': 'I 41 3 2', 'P432': 'P 4 3 2',
                      'P6322': 'P 63 2 2', 'I4122': 'I 41 2 2', 'I213': 'I 21 3', 'I422': 'I 4 2 2',
                      'I432': 'I 4 3 2', 'P4222': 'P 42 2 2', 'F23': 'F 2 3', 'P23': 'P 2 3', 'P213': 'P 21 3',
                      'F432': 'F 4 3 2', 'P622': 'P 6 2 2', 'P4232': 'P 42 3 2', 'F4132': 'F 41 3 2',
                      'P4132': 'P 41 3 2', 'P422': 'P 4 2 2', 'P312': 'P 3 1 2', 'R32': 'R 3 2'}
sg_cryst1_to_hm_notation = {'F 2 2 2': 'F222', 'P 62 2 2': 'P6222', 'I 41 3 2': 'I4132', 'P 4 3 2': 'P432',
                            'P 63 2 2': 'P6322', 'I 41 2 2': 'I4122', 'I 21 3': 'I213', 'I 4 2 2': 'I422',
                            'I 4 3 2': 'I432', 'P 42 2 2': 'P4222', 'F 2 3': 'F23', 'P 2 3': 'P23', 'P 21 3': 'P213',
                            'F 4 3 2': 'F432', 'P 6 2 2': 'P622', 'P 42 3 2': 'P4232', 'F 41 3 2': 'F4132',
                            'P 41 3 2': 'P4132', 'P 4 2 2': 'P422', 'P 3 1 2': 'P312', 'R 3 2': 'R32'}
pg_cryst1_fmt_dict = {'p3': 'P 3', 'p321': 'P 3 2 1', 'p622': 'P 6 2 2', 'p4': 'P 4', 'p222': 'P 2 2 2',
                      'p422': 'P 4 2 2', 'p4212': 'P 4 21 2', 'p6': 'P 6', 'p312': 'P 3 1 2', 'c222': 'C 2 2 2'}
pg_cryst1_to_hm_notation = {'P 3': 'p3', 'P 3 2 1': 'p321', 'P 6 2 2': 'p622', 'P 4': 'p4', 'P 2 2 2': 'p222',
                            'P 4 2 2': 'p422', 'P 4 21 2': 'p4212', 'P 6': 'p6', 'P 3 1 2': 'p312', 'C 2 2 2': 'c222'}
sg_zvalues = {'P23': 12, 'P4222': 8, 'P321': 6, 'P6322': 12, 'P312': 12, 'P622': 12, 'F23': 48, 'F222': 16, 'P6222': 12,
              'I422': 16, 'I213': 24, 'R32': 6, 'P4212': 8, 'I432': 48, 'P4132': 24, 'I4132': 48, 'P3': 3, 'P6': 6,
              'I4122': 16, 'P4': 4, 'C222': 8, 'P222': 4, 'P213': 12, 'F4132': 96, 'P422': 8, 'P432': 24, 'F432': 96,
              'P4232': 24}
valid_subunit_number = {'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D2': 4, 'D3': 6, 'D4': 8, 'D5': 10, 'D6': 12,
                        'T': 12, 'O': 24, 'I': 60}


def multicomponent_by_number(number):
    return [multiplier * number for multiplier in range(1, 10)]


multicomponent_valid_subunit_number = \
    {sym: multicomponent_by_number(copy_number) for sym, copy_number in valid_subunit_number.items()}
# multicomponent_valid_subunit_number = \
#     {'C2': multicomponent_by_number(2), 'C3': multicomponent_by_number(3), 'C4': multicomponent_by_number(4),
#      'C5': multicomponent_by_number(5), 'C6': multicomponent_by_number(6), 'D2': multicomponent_by_number(4),
#      'D3': multicomponent_by_number(6), 'D4': multicomponent_by_number(8), 'D5': multicomponent_by_number(10),
#      'D6': multicomponent_by_number(12), 'T': multicomponent_by_number(12), 'O': multicomponent_by_number(24),
#      'I': multicomponent_by_number(60)}


def generate_cryst1_record(dimensions, space_group):
    """Format the CRYST1 record from specified unit cell dimensions and space group for a .pdb file

    Args:
        dimensions (union[list, tuple]): Containing a, b, c (Angstroms) alpha, beta, gamma (degrees)
        space_group (str): The space group of interest in compact format
    Returns:
        (str): The CRYST1 record
    """
    if space_group in sg_cryst1_fmt_dict:
        fmt_sg = sg_cryst1_fmt_dict[space_group]
    elif space_group in pg_cryst1_fmt_dict:
        fmt_sg = pg_cryst1_fmt_dict[space_group]
        dimensions[2] = 1.0
        dimensions[3] = 90.0  # Todo this hard coding should be wrong for hexagonal plane groups
        dimensions[4] = 90.0  #  also here
    else:
        raise ValueError('SPACEGROUP NOT SUPPORTED')

    return 'CRYST1{dim[0]:9.3f}{dim[1]:9.3f}{dim[2]:9.3f}{dim[3]:7.2f}{dim[4]:7.2f}{dim[5]:7.2f} {sg:<11s}{z:4d}\n'\
        .format(dim=dimensions, sg=fmt_sg, z=sg_zvalues[space_group])


def cart_to_frac(cart_coords, dimensions):
    # http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
    if len(dimensions) == 6:
        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * (
                    np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # deorthogonalization matrix M
        M_0 = [1 / a, -(np.cos(gamma) / float(a * np.sin(gamma))), (((b * np.cos(gamma) * c * (
                    np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))) - (
                                                                                b * c * np.cos(beta) * np.sin(
                                                                            gamma))) * (1 / float(v))]
        M_1 = [0, 1 / (b * np.sin(gamma)),
               -((a * c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(v * np.sin(gamma)))]
        M_2 = [0, 0, (a * b * np.sin(gamma)) / float(v)]
        M = np.array([M_0, M_1, M_2])

        frac_coords = np.matmul(np.array(cart_coords), np.transpose(M))

        return frac_coords

    else:
        raise ValueError(
            "UNIT CELL DIMENSIONS INCORRECTLY SPECIFIED. CORRECT FORMAT IS: [a, b, c,  alpha, beta, gamma]")


def frac_to_cart(frac_coords, dimensions):
    # http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm
    if len(dimensions) == 6:
        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # volume
        v = a * b * c * np.sqrt((1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * (
                    np.cos(alpha) * np.cos(beta) * np.cos(gamma))))

        # orthogonalization matrix M_inv
        M_inv_0 = [a, b * np.cos(gamma), c * np.cos(beta)]
        M_inv_1 = [0, b * np.sin(gamma), (c * (np.cos(alpha) - (np.cos(beta) * np.cos(gamma)))) / float(np.sin(gamma))]
        M_inv_2 = [0, 0, v / float(a * b * np.sin(gamma))]
        M_inv = np.array([M_inv_0, M_inv_1, M_inv_2])

        cart_coords = np.matmul(np.array(frac_coords), np.transpose(M_inv))

        return cart_coords

    else:
        raise ValueError(
            "UNIT CELL DIMENSIONS INCORRECTLY SPECIFIED. CORRECT FORMAT IS: [a, b, c,  alpha, beta, gamma]")


def get_central_asu(pdb, uc_dimensions, design_dimension):
    asu_com_frac = cart_to_frac(pdb.center_of_mass, uc_dimensions)

    # array_range = np.full(21, 1)
    # asu_com_frac_array = array_range[:, np.newaxis] * asu_com_frac
    shift_range = np.arange(-10, 11)
    shift_zeros = np.zeros(len(shift_range))
    shift = np.array([shift_range, shift_range, shift_range]).T
    # y_shift = np.array([shift_zeros, shift_range, shift_zeros]).T
    # z_shift = np.array([shift_zeros, shift_zeros, shift_range]).T

    asu_com_shifted_frac_array = asu_com_frac + shift
    # asu_com_y_shifted_frac_array = asu_com_frac + y_shift
    # asu_com_z_shifted_frac_array = asu_com_frac + z_shift
    asu_com_shifted_cart_array = frac_to_cart(asu_com_shifted_frac_array, uc_dimensions)
    # asu_com_y_shifted_cart_array = frac_to_cart(asu_com_y_shifted_frac_array, uc_dimensions)
    # asu_com_z_shifted_cart_array = frac_to_cart(asu_com_z_shifted_frac_array, uc_dimensions)
    min_shift_idx = abs(asu_com_shifted_cart_array).argmin(axis=0)
    # y_min_shift_idx = abs(asu_com_y_shifted_cart_array).argmin(axis=0)[1]
    # z_min_shift_idx = abs(asu_com_z_shifted_cart_array).argmin(axis=0)[2]

    xyz_min_shift_vec_frac = asu_com_shifted_frac_array[min_shift_idx]
    #                        asu_com_y_shifted_frac_array[y_min_shift_idx],
    #                        asu_com_z_shifted_frac_array[z_min_shift_idx]]

    if design_dimension == 2:
        xyz_min_shift_vec_frac[2] = 0

    if xyz_min_shift_vec_frac == [0, 0, 0]:
        return pdb
    else:
        # xyz_min_shifted_pdb_asu_coords_frac = cart_to_frac(pdb.coords, uc_dimensions) + xyz_min_shift_vec_frac
        # xyz_min_shifted_pdb_asu_coords_cart = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)
        # xyz_min_shifted_asu_pdb = copy.copy(pdb)
        # xyz_min_shifted_asu_pdb.set_coords(xyz_min_shifted_pdb_asu_coords_cart)

        xyz_min_shifted_cart_tx = frac_to_cart(xyz_min_shift_vec_frac, uc_dimensions)
        # xyz_min_shifted_asu_pdb = copy.copy(pdb)
        # xyz_min_shifted_asu_pdb.set_coords(pdb.coords + xyz_min_shifted_cart_tx)
        return pdb.return_transformed_copy(translation=xyz_min_shifted_cart_tx)
        # xyz_min_shifted_asu_pdb.set_atom_coordinates(xyz_min_shifted_pdb_asu_coords_cart)
        # return xyz_min_shifted_asu_pdb


def get_ptgrp_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, "POINT_GROUP_SYMM_OPERATORS")):
    expand_matrix_filepath = expand_matrix_dir + "/" + sym_type + ".txt"
    expand_matrix_file = open(expand_matrix_filepath, "r")
    expand_matrix_lines = expand_matrix_file.readlines()
    expand_matrix_file.close()
    line_count = 0
    expand_matrices = []
    mat = []
    for line in expand_matrix_lines:
        line = line.split()
        if len(line) == 3:
            line_float = [float(s) for s in line]
            mat.append(line_float)
            line_count += 1
            if line_count % 3 == 0:
                expand_matrices.append(mat)
                mat = []
    return expand_matrices


def get_expanded_ptgrp_pdb(pdb_asu, expand_matrices):
    """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
    asu_symm_mates = []
    # asu_coords = pdb_asu.extract_coords()
    # asu_coords = pdb_asu.extract_all_coords()
    for r in expand_matrices:
        # r_asu_coords = np.matmul(asu_coords, np.transpose(np.array(r)))
        asu_sym_mate_pdb = pdb_asu.return_transformed_copy(rotation=np.array(r))
        # asu_sym_mate_pdb = PDB()
        # asu_sym_mate_pdb_atom_list = []
        # atom_count = 0
        # for atom in pdb_asu.atoms:
        #     x_transformed = r_asu_coords[atom_count][0]
        #     y_transformed = r_asu_coords[atom_count][1]
        #     z_transformed = r_asu_coords[atom_count][2]
        #     atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
        #                             atom.get_residue_type(), atom.get_chain(),
        #                             atom.get_residue_number(),
        #                             atom.get_code_for_insertion(), x_transformed, y_transformed,
        #                             z_transformed,
        #                             atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
        #                             atom.get_atom_charge())
        #     atom_count += 1
        #     asu_sym_mate_pdb_atom_list.append(atom_transformed)

        # asu_sym_mate_pdb.set_all_atoms(asu_sym_mate_pdb_atom_list)
        asu_symm_mates.append(asu_sym_mate_pdb)

    return asu_symm_mates


def get_sg_sym_op(sym_type, space_group_operator_dir=os.path.join(sym_op_location, "SPACE_GROUP_SYMM_OPERATORS")):
    """Get the symmetry operations for a specified space group oriented in the canonical orientation
    Returns:
        (list[tuple[list[list], list]])
    """
    sg_op_filepath = os.path.join(space_group_operator_dir, '%s.pickle' % sym_type)
    with open(sg_op_filepath, 'rb') as sg_op_file:
        sg_sym_op = pickle.load(sg_op_file)

    return sg_sym_op


def get_unit_cell_sym_mates(pdb_asu, expand_matrices, uc_dimensions):
    """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
    unit_cell_sym_mates = [pdb_asu]

    asu_cart_coords = pdb_asu.extract_coords()
    # asu_cart_coords = pdb_asu.extract_all_coords()
    asu_frac_coords = cart_to_frac(asu_cart_coords, uc_dimensions)

    for rot, tx in expand_matrices:
        copy_pdb_asu = copy.copy(pdb_asu)
        t_vec = np.array(tx)
        tr_asu_frac_coords = np.matmul(asu_frac_coords, np.transpose(rot)) + t_vec

        tr_asu_cart_coords = frac_to_cart(tr_asu_frac_coords, uc_dimensions).tolist()
        # asu_sym_mate_pdb = pdb_asu.return_transformed_copy(rotation=np.array(r), translation=tx)
        unit_cell_sym_mate_pdb = copy_pdb_asu.replace_coords(tr_asu_cart_coords)

        # unit_cell_sym_mate_pdb = PDB()
        # unit_cell_sym_mate_pdb_atom_list = []
        # atom_count = 0
        # for atom in pdb_asu.atoms():
        #     x_transformed = tr_asu_cart_coords[atom_count][0]
        #     y_transformed = tr_asu_cart_coords[atom_count][1]
        #     z_transformed = tr_asu_cart_coords[atom_count][2]
        #     atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
        #                             atom.get_residue_type(), atom.get_chain(),
        #                             atom.get_residue_number(),
        #                             atom.get_code_for_insertion(), x_transformed, y_transformed,
        #                             z_transformed,
        #                             atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
        #                             atom.get_atom_charge())
        #     atom_count += 1
        #     unit_cell_sym_mate_pdb_atom_list.append(atom_transformed)

        # unit_cell_sym_mate_pdb.set_all_atoms(unit_cell_sym_mate_pdb_atom_list)
        unit_cell_sym_mates.append(unit_cell_sym_mate_pdb)

    return unit_cell_sym_mates


# def get_surrounding_unit_cells(unit_cell_sym_mates, uc_dimensions, dimension=None, return_side_chains=False):
#     """Returns a grid of unit cells for a symmetry group. Each unit cell is a list of ASU's in total grid list"""
#     if dimension == 3:
#         z_shifts, uc_copy_number = [-1, 0, 1], 8
#     elif dimension == 2:
#         z_shifts, uc_copy_number = [0], 26
#     else:
#         return None
#
#     if return_side_chains:  # get different function calls depending on the return type
#         extract_pdb_atoms = getattr(PDB, 'atoms')
#         extract_pdb_coords = getattr(PDB, 'coords')
#     else:
#         extract_pdb_atoms = getattr(PDB, 'backbone_atoms')
#         extract_pdb_coords = getattr(PDB, 'backbone_coords')
#
#     asu_atom_template = extract_pdb_atoms(unit_cell_sym_mates[0])
#     # asu_bb_atom_template = unit_cell_sym_mates[0].backbone_atoms
#
#     central_uc_cart_coords = []
#     for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
#         central_uc_cart_coords.extend(extract_pdb_coords(unit_cell_sym_mate_pdb))
#         # central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
#     central_uc_frac_coords = cart_to_frac(central_uc_cart_coords, uc_dimensions)
#
#     all_surrounding_uc_frac_coords = []
#     for x_shift in [-1, 0, 1]:
#         for y_shift in [-1, 0, 1]:
#             for z_shift in z_shifts:
#                 if [x_shift, y_shift, z_shift] != [0, 0, 0]:
#                     shifted_uc_frac_coords = central_uc_frac_coords + [x_shift, y_shift, z_shift]
#                     all_surrounding_uc_frac_coords.extend(shifted_uc_frac_coords)
#
#     all_surrounding_uc_cart_coords = frac_to_cart(all_surrounding_uc_frac_coords, uc_dimensions)
#     all_surrounding_uc_cart_coords = np.split(all_surrounding_uc_cart_coords, uc_copy_number)
#
#     all_surrounding_unit_cells = []
#     for surrounding_uc_cart_coords in all_surrounding_uc_cart_coords:
#         all_uc_sym_mates_cart_coords = np.split(surrounding_uc_cart_coords, len(unit_cell_sym_mates))
#         one_surrounding_unit_cell = []
#         for uc_sym_mate_cart_coords in all_uc_sym_mates_cart_coords:
#             uc_sym_mate_atoms = []
#             for atom_count, atom in enumerate(asu_atom_template):
#                 x_transformed = uc_sym_mate_cart_coords[atom_count][0]
#                 y_transformed = uc_sym_mate_cart_coords[atom_count][1]
#                 z_transformed = uc_sym_mate_cart_coords[atom_count][2]
#                 atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
#                                         atom.get_residue_type(), atom.get_chain(),
#                                         atom.get_residue_number(),
#                                         atom.get_code_for_insertion(), x_transformed, y_transformed,
#                                         z_transformed,
#                                         atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
#                                         atom.get_atom_charge())
#                 uc_sym_mate_atoms.append(atom_transformed)
#
#             uc_sym_mate_pdb = PDB(atoms=uc_sym_mate_atoms)
#             one_surrounding_unit_cell.append(uc_sym_mate_pdb)
#
#         all_surrounding_unit_cells.append(one_surrounding_unit_cell)
#
#     return all_surrounding_unit_cells


def expand_asu(asu, symmetry, uc_dimensions=None, return_side_chains=False):  # unused
    """Return the expanded material from the input ASU, symmetry specification, and unit cell dimensions

    Args:
        asu (PDB): PDB object that contains the minimal protein for the specified material
        symmetry (str): The Herman Melville symmetry nomenclature of the symmetric group, ex: P432, F23, I, etc.
    Keyword Args:
        uc_dimensions=None (list): [57, 57, 57, 90, 90, 90] lengths a, b, and c, then angles
        return_side_chains=False (bool): Whether to return all side chain atoms
    Returns:
        (list(PDB)): Expanded to entire point group, 3x3 layer group, or 3x3x3 space group
    """
    if symmetry.upper() in ['T', 'O', 'I']:
        expand_matrices = get_ptgrp_sym_op(symmetry.upper())
        return get_expanded_ptgrp_pdb(asu, expand_matrices)
    else:
        if symmetry in pg_cryst1_fmt_dict:
            dimension = 2
        elif symmetry in sg_cryst1_fmt_dict:
            dimension = 3
        else:
            return None
        expand_matrices = get_sg_sym_op(symmetry)

        return expand_uc(asu, expand_matrices, uc_dimensions, dimension, return_side_chains=return_side_chains)


def expand_uc(pdb_asu, expand_matrices, uc_dimensions, dimension, return_side_chains=False):
    """Return the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell

    Returns
        (list(list(PDB))):
    """
    unit_cell_pdbs = get_unit_cell_sym_mates(pdb_asu, expand_matrices, uc_dimensions)
    if dimension in [2, 3]:
        dummy = True
        # all_surrounding_unit_cells = get_surrounding_unit_cells(unit_cell_pdbs, uc_dimensions, dimension=dimension, return_side_chains=return_side_chains)
        # all_surrounding_unit_cells = get_surrounding_unit_cells_2d(unit_cell_pdbs, uc_dimensions)
    # elif dimension == 3:
    #     all_surrounding_unit_cells = get_surrounding_unit_cells_3d(unit_cell_pdbs, uc_dimensions)
    else:
        return None

    return unit_cell_pdbs
    # return all_surrounding_unit_cells