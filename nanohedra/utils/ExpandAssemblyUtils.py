import copy
import os
import pickle

import numpy as np
import sklearn.neighbors

from nanohedra.classes.Atom import Atom
from nanohedra.classes.PDB import PDB
from nanohedra.utils.GeneralUtils import center_of_mass_3d

# Globals
sg_cryst1_fmt_dict = {'F222': 'F 2 2 2', 'P6222': 'P 62 2 2', 'I4132': 'I 41 3 2', 'P432': 'P 4 3 2',
                      'P6322': 'P 63 2 2', 'I4122': 'I 41 2 2', 'I213': 'I 21 3', 'I422': 'I 4 2 2',
                      'I432': 'I 4 3 2', 'P4222': 'P 42 2 2', 'F23': 'F 2 3', 'P23': 'P 2 3', 'P213': 'P 21 3',
                      'F432': 'F 4 3 2', 'P622': 'P 6 2 2', 'P4232': 'P 42 3 2', 'F4132': 'F 41 3 2',
                      'P4132': 'P 41 3 2', 'P422': 'P 4 2 2', 'P312': 'P 3 1 2', 'R32': 'R 3 2'}
pg_cryst1_fmt_dict = {'p3': 'P 3', 'p321': 'P 3 2 1', 'p622': 'P 6 2 2', 'p4': 'P 4', 'p222': 'P 2 2 2',
                      'p422': 'P 4 2 2', 'p4212': 'P 4 21 2', 'p6': 'P 6', 'p312': 'P 3 1 2', 'c222': 'C 2 2 2'}
zvalue_dict = {'P 2 3': 12, 'P 42 2 2': 8, 'P 3 2 1': 6, 'P 63 2 2': 12, 'P 3 1 2': 12, 'P 6 2 2': 12, 'F 2 3': 48,
               'F 2 2 2': 16, 'P 62 2 2': 12, 'I 4 2 2': 16, 'I 21 3': 24, 'R 3 2': 6, 'P 4 21 2': 8, 'I 4 3 2': 48,
               'P 41 3 2': 24, 'I 41 3 2': 48, 'P 3': 3, 'P 6': 6, 'I 41 2 2': 16, 'P 4': 4, 'C 2 2 2': 8,
               'P 2 2 2': 4, 'P 21 3': 12, 'F 41 3 2': 96, 'P 4 2 2': 8, 'P 4 3 2': 24, 'F 4 3 2': 96,
               'P 42 3 2': 24}


def generate_cryst1_record(dimensions, spacegroup):
    # dimensions is a python list containing a, b, c (Angstroms) alpha, beta, gamma (degrees)

    if spacegroup in sg_cryst1_fmt_dict:
        fmt_spacegroup = sg_cryst1_fmt_dict[spacegroup]
        zvalue = zvalue_dict[fmt_spacegroup]
    elif spacegroup in pg_cryst1_fmt_dict:
        fmt_spacegroup = pg_cryst1_fmt_dict[spacegroup]
        zvalue = zvalue_dict[fmt_spacegroup]
        dimensions[2] = 1.0
        dimensions[3] = 90.0
        dimensions[4] = 90.0
    else:
        raise ValueError("SPACEGROUP NOT SUPPORTED")

    cryst1_fmt = "CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}""{ang[0]:7.2f}{ang[1]:7.2f}{ang[2]:7.2f} ""{spacegroup:<11s}{zvalue:4d}\n"
    return cryst1_fmt.format(box=dimensions[:3], ang=dimensions[3:], spacegroup=fmt_spacegroup, zvalue=zvalue)


def get_ptgrp_sym_op(sym_type, expand_matrix_dir=os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))) + "/ExpandMatrices/POINT_GROUP_SYMM_OPERATORS"):
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


def get_expanded_ptgrp_pdbs(pdb1_asu, pdb2_asu, expand_matrices):
    pdb_asu = PDB()
    pdb_asu.set_all_atoms(pdb1_asu.get_all_atoms() + pdb2_asu.get_all_atoms())

    return get_expanded_ptgrp_pdb(pdb_asu, expand_matrices)


def get_expanded_ptgrp_pdb(pdb_asu, expand_matrices):
    """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
    asu_symm_mates = []
    asu_coords = pdb_asu.extract_all_coords()
    for r in expand_matrices:
        r_mat = np.transpose(np.array(r))
        r_asu_coords = np.matmul(asu_coords, r_mat)

        asu_sym_mate_pdb = PDB()
        asu_sym_mate_pdb_atom_list = []
        atom_count = 0
        for atom in pdb_asu.get_all_atoms():
            x_transformed = r_asu_coords[atom_count][0]
            y_transformed = r_asu_coords[atom_count][1]
            z_transformed = r_asu_coords[atom_count][2]
            atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
                                    atom.get_residue_type(), atom.get_chain(),
                                    atom.get_residue_number(),
                                    atom.get_code_for_insertion(), x_transformed, y_transformed,
                                    z_transformed,
                                    atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                    atom.get_atom_charge())
            atom_count += 1
            asu_sym_mate_pdb_atom_list.append(atom_transformed)

        asu_sym_mate_pdb.set_all_atoms(asu_sym_mate_pdb_atom_list)
        asu_symm_mates.append(asu_sym_mate_pdb)

    return asu_symm_mates


def write_expanded_ptgrp(expanded_ptgrp_pdbs, outfile_path):  # TODO DEPRECIATE
    outfile = open(outfile_path, "w")
    model_count = 1
    for pdb in expanded_ptgrp_pdbs:  # TODO enumerate
        outfile.write("MODEL     {:>4s}\n".format(str(model_count)))
        model_count += 1
        for atom in pdb.all_atoms:
            outfile.write(str(atom))
        outfile.write("ENDMDL\n")
    outfile.close()


def expanded_ptgrp_is_clash(expanded_ptgrp_pdbs, clash_distance=2.2):
    asu = expanded_ptgrp_pdbs[0]
    symm_mates_wo_asu = expanded_ptgrp_pdbs[1:]

    asu_bb_coords = asu.extract_backbone_coords()
    symm_mates_wo_asu_bb_coords = []
    for sym_mate_pdb in symm_mates_wo_asu:
        symm_mates_wo_asu_bb_coords.extend(sym_mate_pdb.extract_backbone_coords())

    kdtree_central_asu_bb = sklearn.neighbors.BallTree(np.array(asu_bb_coords))
    cb_clash_count = kdtree_central_asu_bb.two_point_correlation(symm_mates_wo_asu_bb_coords, [clash_distance])

    ### CLASH TEST ###
    # asu_bb_indices = asu.get_bb_indices()
    # query_list = kdtree_central_asu_bb.query_radius(symm_mates_wo_asu_bb_coords, clash_distance)
    # for symm_mates_wo_asu_bb_coords_index in range(len(query_list)):
    #     for kdtree_central_asu_bb_index in query_list[symm_mates_wo_asu_bb_coords_index]:
    #         print asu.all_atoms[asu_bb_indices[kdtree_central_asu_bb_index]]
    ##################

    if cb_clash_count[0] == 0:
        return False  # "NO CLASH"

    else:
        return True  # "CLASH!!"


def get_sg_sym_op(sym_type, space_group_operator_dir=os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))) + "/ExpandMatrices/SPACE_GROUP_SYMM_OPERATORS"):
    sg_op_filepath = space_group_operator_dir + "/" + sym_type + ".pickle"
    with open(sg_op_filepath, "rb") as sg_op_file:
        sg_sym_op = pickle.load(sg_op_file)

    return sg_sym_op


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


def get_central_asu_pdb_2d(pdb1, pdb2, uc_dimensions):
    pdb_asu = PDB()
    pdb_asu.read_atom_list(pdb1.get_all_atoms() + pdb2.get_all_atoms())

    pdb_asu_coords_cart = pdb_asu.extract_all_coords()

    asu_com_cart = center_of_mass_3d(pdb_asu_coords_cart)
    asu_com_frac = cart_to_frac(asu_com_cart, uc_dimensions)

    asu_com_x_min_cart = float('inf')
    x_min_shift_vec_frac = None
    for x in range(-10, 11):
        asu_com_x_shifted_coords_frac = asu_com_frac + [x, 0, 0]
        asu_com_x_shifted_coords_cart = frac_to_cart(asu_com_x_shifted_coords_frac, uc_dimensions)
        if abs(asu_com_x_shifted_coords_cart[0]) < abs(asu_com_x_min_cart):
            asu_com_x_min_cart = asu_com_x_shifted_coords_cart[0]
            x_min_shift_vec_frac = [x, 0, 0]

    asu_com_y_min_cart = float('inf')
    y_min_shift_vec_frac = None
    for y in range(-10, 11):
        asu_com_y_shifted_coords_frac = asu_com_frac + [0, y, 0]
        asu_com_y_shifted_coords_cart = frac_to_cart(asu_com_y_shifted_coords_frac, uc_dimensions)
        if abs(asu_com_y_shifted_coords_cart[1]) < abs(asu_com_y_min_cart):
            asu_com_y_min_cart = asu_com_y_shifted_coords_cart[1]
            y_min_shift_vec_frac = [0, y, 0]

    if x_min_shift_vec_frac is not None and y_min_shift_vec_frac is not None:
        xyz_min_shift_vec_frac = [x_min_shift_vec_frac[0], y_min_shift_vec_frac[1], 0]

        if xyz_min_shift_vec_frac == [0, 0, 0]:
            return pdb_asu

        else:
            pdb_asu_coords_frac = cart_to_frac(pdb_asu_coords_cart, uc_dimensions)
            xyz_min_shifted_pdb_asu_coords_frac = pdb_asu_coords_frac + xyz_min_shift_vec_frac
            xyz_min_shifted_pdb_asu_coords_cart = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)

            xyz_min_shifted_asu_pdb = copy.deepcopy(pdb_asu)
            xyz_min_shifted_asu_pdb.replace_coords(xyz_min_shifted_pdb_asu_coords_cart)

            return xyz_min_shifted_asu_pdb

    else:
        return pdb_asu


def get_central_asu_pdb_3d(pdb1, pdb2, uc_dimensions):
    pdb_asu = PDB()
    pdb_asu.read_atom_list(pdb1.get_all_atoms() + pdb2.get_all_atoms())

    pdb_asu_coords_cart = pdb_asu.extract_all_coords()

    asu_com_cart = center_of_mass_3d(pdb_asu_coords_cart)
    asu_com_frac = cart_to_frac(asu_com_cart, uc_dimensions)

    asu_com_x_min_cart = float('inf')
    x_min_shift_vec_frac = None
    for x in range(-10, 11):
        asu_com_x_shifted_coords_frac = asu_com_frac + [x, 0, 0]
        asu_com_x_shifted_coords_cart = frac_to_cart(asu_com_x_shifted_coords_frac, uc_dimensions)
        if abs(asu_com_x_shifted_coords_cart[0]) < abs(asu_com_x_min_cart):
            asu_com_x_min_cart = asu_com_x_shifted_coords_cart[0]
            x_min_shift_vec_frac = [x, 0, 0]

    asu_com_y_min_cart = float('inf')
    y_min_shift_vec_frac = None
    for y in range(-10, 11):
        asu_com_y_shifted_coords_frac = asu_com_frac + [0, y, 0]
        asu_com_y_shifted_coords_cart = frac_to_cart(asu_com_y_shifted_coords_frac, uc_dimensions)
        if abs(asu_com_y_shifted_coords_cart[1]) < abs(asu_com_y_min_cart):
            asu_com_y_min_cart = asu_com_y_shifted_coords_cart[1]
            y_min_shift_vec_frac = [0, y, 0]

    asu_com_z_min_cart = float('inf')
    z_min_shift_vec_frac = None
    for z in range(-10, 11):
        asu_com_z_shifted_coords_frac = asu_com_frac + [0, 0, z]
        asu_com_z_shifted_coords_cart = frac_to_cart(asu_com_z_shifted_coords_frac, uc_dimensions)
        if abs(asu_com_z_shifted_coords_cart[2]) < abs(asu_com_z_min_cart):
            asu_com_z_min_cart = asu_com_z_shifted_coords_cart[2]
            z_min_shift_vec_frac = [0, 0, z]

    if x_min_shift_vec_frac is not None and y_min_shift_vec_frac is not None and z_min_shift_vec_frac is not None:
        xyz_min_shift_vec_frac = [x_min_shift_vec_frac[0], y_min_shift_vec_frac[1], z_min_shift_vec_frac[2]]

        if xyz_min_shift_vec_frac == [0, 0, 0]:
            return pdb_asu

        else:
            pdb_asu_coords_frac = cart_to_frac(pdb_asu_coords_cart, uc_dimensions)
            xyz_min_shifted_pdb_asu_coords_frac = pdb_asu_coords_frac + xyz_min_shift_vec_frac
            xyz_min_shifted_pdb_asu_coords_cart = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)

            xyz_min_shifted_asu_pdb = copy.deepcopy(pdb_asu)
            xyz_min_shifted_asu_pdb.replace_coords(xyz_min_shifted_pdb_asu_coords_cart)

            return xyz_min_shifted_asu_pdb

    else:
        return pdb_asu


def get_unit_cell_sym_mates(pdb_asu, expand_matrices, uc_dimensions):
    """Return all symmetry mates as a list of PDB objects. Chain names will match the ASU"""
    unit_cell_sym_mates = [pdb_asu]

    asu_cart_coords = pdb_asu.extract_all_coords()
    asu_frac_coords = cart_to_frac(asu_cart_coords, uc_dimensions)

    for r, t in expand_matrices:
        t_vec = np.array(t)
        r_mat = np.transpose(np.array(r))

        r_asu_frac_coords = np.matmul(asu_frac_coords, r_mat)
        tr_asu_frac_coords = r_asu_frac_coords + t_vec

        tr_asu_cart_coords = frac_to_cart(tr_asu_frac_coords, uc_dimensions).tolist()

        unit_cell_sym_mate_pdb = PDB()
        unit_cell_sym_mate_pdb_atom_list = []
        atom_count = 0
        for atom in pdb_asu.get_all_atoms():
            x_transformed = tr_asu_cart_coords[atom_count][0]
            y_transformed = tr_asu_cart_coords[atom_count][1]
            z_transformed = tr_asu_cart_coords[atom_count][2]
            atom_transformed = Atom(atom_count, atom.get_type(), atom.get_alt_location(),
                                    atom.get_residue_type(), atom.get_chain(),
                                    atom.get_residue_number(),
                                    atom.get_code_for_insertion(), x_transformed, y_transformed,
                                    z_transformed,
                                    atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                    atom.get_atom_charge())
            atom_count += 1
            unit_cell_sym_mate_pdb_atom_list.append(atom_transformed)

        unit_cell_sym_mate_pdb.set_all_atoms(unit_cell_sym_mate_pdb_atom_list)
        unit_cell_sym_mates.append(unit_cell_sym_mate_pdb)

    return unit_cell_sym_mates


def get_surrounding_unit_cells(unit_cell_sym_mates, uc_dimensions, dimension=None, return_side_chains=False):
    """Returns a grid of unit cells for a symmetry group. Each unit cell is a list of ASU's in total grid list"""
    if dimension == 3:
        z_shifts, uc_copy_number = [-1, 0, 1], 8
    elif dimension == 2:
        z_shifts, uc_copy_number = [0], 26
    else:
        return None

    if return_side_chains:  # get different function calls depending on the return type
        extract_pdb_atoms = getattr(PDB, 'get_all_atoms')
        extract_pdb_coords = getattr(PDB, '.extract_all_coords')
    else:
        extract_pdb_atoms = getattr(PDB, 'get_backbone_atoms')
        extract_pdb_coords = getattr(PDB, 'extract_backbone_coords')

    asu_atom_template = extract_pdb_atoms(unit_cell_sym_mates[0])
    # asu_bb_atom_template = unit_cell_sym_mates[0].get_backbone_atoms()

    central_uc_cart_coords = []
    for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
        central_uc_cart_coords.extend(extract_pdb_coords(unit_cell_sym_mate_pdb))
        # central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
    central_uc_frac_coords = cart_to_frac(central_uc_cart_coords, uc_dimensions)

    all_surrounding_uc_frac_coords = []
    for x_shift in [-1, 0, 1]:
        for y_shift in [-1, 0, 1]:
            for z_shift in z_shifts:
                if [x_shift, y_shift, z_shift] != [0, 0, 0]:
                    shifted_uc_frac_coords = central_uc_frac_coords + [x_shift, y_shift, z_shift]
                    all_surrounding_uc_frac_coords.extend(shifted_uc_frac_coords)

    all_surrounding_uc_cart_coords = frac_to_cart(all_surrounding_uc_frac_coords, uc_dimensions)
    all_surrounding_uc_cart_coords = np.split(all_surrounding_uc_cart_coords, uc_copy_number)

    all_surrounding_unit_cells = []
    for surrounding_uc_cart_coords in all_surrounding_uc_cart_coords:
        all_uc_sym_mates_cart_coords = np.split(surrounding_uc_cart_coords, len(unit_cell_sym_mates))
        one_surrounding_unit_cell = []
        for uc_sym_mate_cart_coords in all_uc_sym_mates_cart_coords:
            uc_sym_mate_pdb = PDB()
            uc_sym_mate_atoms = []
            # atom_count = 0
            for atom_count, atom in enumerate(asu_atom_template):
                x_transformed = uc_sym_mate_cart_coords[atom_count][0]
                y_transformed = uc_sym_mate_cart_coords[atom_count][1]
                z_transformed = uc_sym_mate_cart_coords[atom_count][2]
                atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                        atom.get_residue_type(), atom.get_chain(),
                                        atom.get_residue_number(),
                                        atom.get_code_for_insertion(), x_transformed, y_transformed,
                                        z_transformed,
                                        atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                        atom.get_atom_charge())
                uc_sym_mate_atoms.append(atom_transformed)
                # atom_count += 1

            uc_sym_mate_pdb.set_all_atoms(uc_sym_mate_atoms)
            # uc_sym_mate_pdb = SDUtils.fill_pdb(uc_sym_mate_atoms) TODO
            one_surrounding_unit_cell.append(uc_sym_mate_pdb)

        all_surrounding_unit_cells.append(one_surrounding_unit_cell)

    return all_surrounding_unit_cells


def get_surrounding_unit_cells_2d(unit_cell_sym_mates, uc_dimensions):  # DEPRECIATE
    """Returns a 3x3 grid of unit cells for a layer group"""
    all_surrounding_unit_cells = []

    asu_bb_atom_template = unit_cell_sym_mates[0].get_backbone_atoms()
    unit_cell_sym_mates_len = len(unit_cell_sym_mates)

    central_uc_bb_cart_coords = []
    for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
        central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
    central_uc_bb_frac_coords = cart_to_frac(central_uc_bb_cart_coords, uc_dimensions)

    all_surrounding_uc_bb_frac_coords = []
    for x_shift in [-1, 0, 1]:
        for y_shift in [-1, 0, 1]:
            if [x_shift, y_shift] != [0, 0]:
                shifted_uc_bb_frac_coords = central_uc_bb_frac_coords + [x_shift, y_shift, 0]
                all_surrounding_uc_bb_frac_coords.extend(shifted_uc_bb_frac_coords)

    all_surrounding_uc_bb_cart_coords = frac_to_cart(all_surrounding_uc_bb_frac_coords, uc_dimensions)
    all_surrounding_uc_bb_cart_coords = np.split(all_surrounding_uc_bb_cart_coords, 8)

    for surrounding_uc_bb_cart_coords in all_surrounding_uc_bb_cart_coords:
        all_uc_sym_mates_bb_cart_coords = np.split(surrounding_uc_bb_cart_coords, unit_cell_sym_mates_len)
        one_surrounding_unit_cell = []
        for uc_sym_mate_bb_cart_coords in all_uc_sym_mates_bb_cart_coords:
            uc_sym_mate_bb_pdb = PDB()
            uc_sym_mate_bb_atoms = []
            atom_count = 0
            for atom in asu_bb_atom_template:
                x_transformed = uc_sym_mate_bb_cart_coords[atom_count][0]
                y_transformed = uc_sym_mate_bb_cart_coords[atom_count][1]
                z_transformed = uc_sym_mate_bb_cart_coords[atom_count][2]
                atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                        atom.get_residue_type(), atom.get_chain(),
                                        atom.get_residue_number(),
                                        atom.get_code_for_insertion(), x_transformed, y_transformed,
                                        z_transformed,
                                        atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                        atom.get_atom_charge())
                uc_sym_mate_bb_atoms.append(atom_transformed)
                atom_count += 1

            uc_sym_mate_bb_pdb.set_all_atoms(uc_sym_mate_bb_atoms)
            one_surrounding_unit_cell.append(uc_sym_mate_bb_pdb)

        all_surrounding_unit_cells.append(one_surrounding_unit_cell)

    return all_surrounding_unit_cells


def get_surrounding_unit_cells_3d(unit_cell_sym_mates, uc_dimensions):  # DEPRECIATE
    """Returns a 3x3x3 grid of unit cells for a space group. Each unit cell is a list in list,
    each ASU an item in the unit cell list"""
    asu_bb_atom_template = unit_cell_sym_mates[0].get_backbone_atoms()

    central_uc_bb_cart_coords = []
    for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
        central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
    central_uc_bb_frac_coords = cart_to_frac(central_uc_bb_cart_coords, uc_dimensions)

    all_surrounding_uc_bb_frac_coords = []
    for x_shift in [-1, 0, 1]:
        for y_shift in [-1, 0, 1]:
            for z_shift in [-1, 0, 1]:
                if [x_shift, y_shift, z_shift] != [0, 0, 0]:
                    shifted_uc_bb_frac_coords = central_uc_bb_frac_coords + [x_shift, y_shift, z_shift]
                    all_surrounding_uc_bb_frac_coords.extend(shifted_uc_bb_frac_coords)

    all_surrounding_uc_bb_cart_coords = frac_to_cart(all_surrounding_uc_bb_frac_coords, uc_dimensions)
    all_surrounding_uc_bb_cart_coords = np.split(all_surrounding_uc_bb_cart_coords, 26)

    all_surrounding_unit_cells = []
    for surrounding_uc_bb_cart_coords in all_surrounding_uc_bb_cart_coords:
        all_uc_sym_mates_bb_cart_coords = np.split(surrounding_uc_bb_cart_coords, len(unit_cell_sym_mates))
        one_surrounding_unit_cell = []
        for uc_sym_mate_bb_cart_coords in all_uc_sym_mates_bb_cart_coords:
            uc_sym_mate_bb_pdb = PDB()
            uc_sym_mate_bb_atoms = []
            atom_count = 0
            for atom in asu_bb_atom_template:
                x_transformed = uc_sym_mate_bb_cart_coords[atom_count][0]
                y_transformed = uc_sym_mate_bb_cart_coords[atom_count][1]
                z_transformed = uc_sym_mate_bb_cart_coords[atom_count][2]
                atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
                                        atom.get_residue_type(), atom.get_chain(),
                                        atom.get_residue_number(),
                                        atom.get_code_for_insertion(), x_transformed, y_transformed,
                                        z_transformed,
                                        atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
                                        atom.get_atom_charge())
                uc_sym_mate_bb_atoms.append(atom_transformed)
                atom_count += 1

            uc_sym_mate_bb_pdb.set_all_atoms(uc_sym_mate_bb_atoms)
            one_surrounding_unit_cell.append(uc_sym_mate_bb_pdb)

        all_surrounding_unit_cells.append(one_surrounding_unit_cell)

    return all_surrounding_unit_cells


def write_unit_cell_sym_mates(unit_cell_sym_mates, outfile_path):  # Todo integrate with Model.py
    f = open(outfile_path, "a+")
    model_count = 0
    for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
        model_count += 1
        model_line = "MODEL     {:>4s}\n".format(str(model_count))
        end_model_line = "ENDMDL\n"

        f.write(model_line)
        for atom in unit_cell_sym_mate_pdb.get_all_atoms():
            f.write(str(atom))
        f.write(end_model_line)
    f.close()


def write_surrounding_unit_cells(surrounding_unit_cells, outfile_path):  # Todo integrate with Model.py
    f = open(outfile_path, "a+")

    model_count = 0
    for unit_cell in surrounding_unit_cells:  # Todo remove the extra nest on UC generation
        for unit_cell_sym_mate_pdb in unit_cell:
            model_count += 1
            model_line = "MODEL     {:>4s}\n".format(str(model_count))
            end_model_line = "ENDMDL\n"

            f.write(model_line)
            for atom in unit_cell_sym_mate_pdb.get_all_atoms():
                f.write(str(atom))
            f.write(end_model_line)

    f.close()


def uc_expansion_is_clash(central_unit_cell, clash_distance=2.2):
    central_asu_pdb = central_unit_cell[0]
    central_unit_cell_wo_central_asu = central_unit_cell[1:]

    central_asu_pdb_bb_coords = central_asu_pdb.extract_backbone_coords()
    central_unit_cell_wo_central_asu_bb_coords = []
    for unit_cell_sym_mate_pdb in central_unit_cell_wo_central_asu:
        central_unit_cell_wo_central_asu_bb_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())

    kdtree_central_asu_bb = sklearn.neighbors.BallTree(np.array(central_asu_pdb_bb_coords))
    cb_clash_count = kdtree_central_asu_bb.two_point_correlation(central_unit_cell_wo_central_asu_bb_coords,
                                                                 [clash_distance])

    if cb_clash_count[0] == 0:
        return False  # "NO CLASH"

    else:
        return True  # "CLASH!!"


def surrounding_uc_is_clash(central_unit_cell, surrounding_unit_cells, clash_distance=2.2):
    central_asu_pdb = central_unit_cell[0]
    all_unit_cells_wo_central_asu = surrounding_unit_cells + [central_unit_cell[1:]]

    central_asu_pdb_bb_coords = central_asu_pdb.extract_backbone_coords()
    all_unit_cells_wo_central_asu_bb_coords = []
    for unit_cell in all_unit_cells_wo_central_asu:
        for unit_cell_sym_mate_pdb in unit_cell:
            all_unit_cells_wo_central_asu_bb_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())

    kdtree_central_asu_bb = sklearn.neighbors.BallTree(np.array(central_asu_pdb_bb_coords))
    cb_clash_count = kdtree_central_asu_bb.two_point_correlation(all_unit_cells_wo_central_asu_bb_coords,
                                                                 [clash_distance])

    if cb_clash_count[0] == 0:
        return False  # "NO CLASH"

    else:
        return True  # "CLASH!!"


def expand_asu(asu, symmetry, uc_dimensions=None, return_side_chains=False):
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
        all_surrounding_unit_cells = get_surrounding_unit_cells(unit_cell_pdbs, uc_dimensions, dimension=dimension, return_side_chains=return_side_chains)
        # all_surrounding_unit_cells = get_surrounding_unit_cells_2d(unit_cell_pdbs, uc_dimensions)
    # elif dimension == 3:
    #     all_surrounding_unit_cells = get_surrounding_unit_cells_3d(unit_cell_pdbs, uc_dimensions)
    else:
        return None

    return all_surrounding_unit_cells


def expanded_design_is_clash(asu_pdb_1, asu_pdb_2, design_dim, result_design_sym, expand_matrices, uc_dimensions=None,
                             outdir=None, output_exp_assembly=False, output_uc=False, output_surrounding_uc=False):
    if design_dim == 0:
        expanded_ptgrp_pdbs = get_expanded_ptgrp_pdbs(asu_pdb_1, asu_pdb_2, expand_matrices)

        is_clash = expanded_ptgrp_is_clash(expanded_ptgrp_pdbs)

        if not is_clash and outdir is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if output_exp_assembly:
                write_expanded_ptgrp(expanded_ptgrp_pdbs, outdir + "/expanded_assembly.pdb")
            pdb_asu = expanded_ptgrp_pdbs[0]
            pdb_asu.write(outdir + "/asu.pdb")

        return is_clash

    elif design_dim == 2:
        pdb_asu = get_central_asu_pdb_2d(asu_pdb_1, asu_pdb_2, uc_dimensions)

        cryst1_record = generate_cryst1_record(uc_dimensions, result_design_sym)

        unit_cell_pdbs = get_unit_cell_sym_mates(pdb_asu, expand_matrices, uc_dimensions)

        is_uc_exp_clash = uc_expansion_is_clash(unit_cell_pdbs)
        if is_uc_exp_clash:
            return is_uc_exp_clash

        all_surrounding_unit_cells = get_surrounding_unit_cells_2d(unit_cell_pdbs, uc_dimensions)
        is_clash = surrounding_uc_is_clash(unit_cell_pdbs, all_surrounding_unit_cells)

        if not is_clash and outdir is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if output_uc:
                write_unit_cell_sym_mates(unit_cell_pdbs, outdir + "/central_uc.pdb")
            if output_surrounding_uc:
                write_surrounding_unit_cells(all_surrounding_unit_cells, outdir + "/surrounding_unit_cells.pdb")
            pdb_asu.write(outdir + "/central_asu.pdb", cryst1=cryst1_record)

        return is_clash

    elif design_dim == 3:

        # get_cryst1_time_start = time.time()
        cryst1_record = generate_cryst1_record(uc_dimensions, result_design_sym)
        # get_cryst1_time_stop = time.time()
        # get_cryst1_time = get_cryst1_time_stop - get_cryst1_time_start
        # print "TOOK %s seconds to get CRYST1" % str(get_cryst1_time)

        # get_central_asu_time_start = time.time()
        pdb_asu = get_central_asu_pdb_3d(asu_pdb_1, asu_pdb_2, uc_dimensions)
        # get_central_asu_time_stop = time.time()
        # get_central_asu_time = get_central_asu_time_stop - get_central_asu_time_start
        # print "TOOK %s seconds to get central ASU" % str(get_central_asu_time)

        # get_uc_time_start = time.time()
        unit_cell_pdbs = get_unit_cell_sym_mates(pdb_asu, expand_matrices, uc_dimensions)
        # get_uc_time_stop = time.time()
        # get_uc_time = get_uc_time_stop - get_uc_time_start
        # print "TOOK %s seconds to get UC PDBs" % str(get_uc_time)

        # uc_clash_test_time_start = time.time()
        is_uc_exp_clash = uc_expansion_is_clash(unit_cell_pdbs)
        # uc_clash_test_time_stop = time.time()
        # uc_clash_test_time = uc_clash_test_time_stop - uc_clash_test_time_start
        # print "TOOK %s seconds to test for clash in UC" % str(uc_clash_test_time)

        if is_uc_exp_clash:
            # print "\n\n"
            return is_uc_exp_clash

        # get_all_surrounding_uc_time_start = time.time()
        all_surrounding_unit_cells = get_surrounding_unit_cells_3d(unit_cell_pdbs, uc_dimensions)
        # get_all_surrounding_uc_time_stop = time.time()
        # get_all_surrounding_uc_time = get_all_surrounding_uc_time_stop - get_all_surrounding_uc_time_start
        # print "TOOK %s seconds to get surrounding UCs" % str(get_all_surrounding_uc_time)

        # surrounding_uc_test_time_start = time.time()
        is_clash = surrounding_uc_is_clash(unit_cell_pdbs, all_surrounding_unit_cells)
        # surrounding_uc_test_time_stop = time.time()
        # surrounding_uc_test_time = surrounding_uc_test_time_stop - surrounding_uc_test_time_start
        # print "TOOK %s seconds to test for clash in surrounding UCs" % str(surrounding_uc_test_time)

        # write_time_start = time.time()
        if not is_clash and outdir is not None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if output_uc:
                write_unit_cell_sym_mates(unit_cell_pdbs, outdir + "/central_uc.pdb")
            if output_surrounding_uc:
                write_surrounding_unit_cells(all_surrounding_unit_cells, outdir + "/surrounding_unit_cells.pdb")
            pdb_asu.write(outdir + "/central_asu.pdb", cryst1=cryst1_record)
        # write_time_stop = time.time()
        # write_time = write_time_stop - write_time_start
        # print "TOOK %s seconds to write out asu, uc and surrounding UCs" % str(write_time)
        # print "\n\n"

        return is_clash

    else:
        raise ValueError(
            "%s is an Invalid Design Dimension. The Only Valid Dimensions are: 0, 2, 3\n" % str(design_dim))
