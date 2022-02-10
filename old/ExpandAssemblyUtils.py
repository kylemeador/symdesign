import copy
import os

import numpy as np
import sklearn.neighbors


# Globals
from utils.SymmetryUtils import generate_cryst1_record, cart_to_frac, frac_to_cart, \
    get_expanded_ptgrp_pdb, get_unit_cell_sym_mates


# def get_expanded_ptgrp_pdbs(pdb1_asu, pdb2_asu, expand_matrices):
#     pdb_asu = PDB()
#     pdb_asu.set_all_atoms(pdb1_asu.atoms + pdb2_asu.atoms)
#
#     return get_expanded_ptgrp_pdb(pdb_asu, expand_matrices)


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

    # asu_bb_coords = asu.get_backbone_and_cb_coords()
    # asu_bb_coords = asu.extract_backbone_coords()
    # symm_mates_wo_asu_bb_coords = [sym_mate_pdb.get_backbone_and_cb_coords() for sym_mate_pdb in symm_mates_wo_asu]
    symm_mates_wo_asu_bb_coords = []
    for sym_mate_pdb in symm_mates_wo_asu:
        symm_mates_wo_asu_bb_coords.extend(sym_mate_pdb.extract_backbone_coords())

    # kdtree_central_asu_bb = sklearn.neighbors.BallTree(np.array(asu_bb_coords))
    kdtree_central_asu_bb = sklearn.neighbors.BallTree(asu.get_backbone_and_cb_coords())
    cb_clash_count = kdtree_central_asu_bb.two_point_correlation(symm_mates_wo_asu_bb_coords, [clash_distance])

    if cb_clash_count[0] == 0:
        return False  # "NO CLASH"

    else:
        return True  # "CLASH!!"


# def get_central_asu_pdb_2d(pdb1, pdb2, uc_dimensions):
#     pdb_asu = PDB()
#     pdb_asu.read_atom_list(pdb1.atoms() + pdb2.atoms())
#
#     # pdb_asu_coords_cart = pdb_asu.extract_coords()  # TODO
#     pdb_asu_coords_cart = pdb_asu.extract_all_coords()
#
#     asu_com_cart = center_of_mass_3d(pdb_asu_coords_cart)
#     asu_com_frac = cart_to_frac(asu_com_cart, uc_dimensions)
#
#     asu_com_x_min_cart = float('inf')
#     x_min_shift_vec_frac = None
#     for x in range(-10, 11):
#         asu_com_x_shifted_coords_frac = asu_com_frac + [x, 0, 0]
#         asu_com_x_shifted_coords_cart = frac_to_cart(asu_com_x_shifted_coords_frac, uc_dimensions)
#         if abs(asu_com_x_shifted_coords_cart[0]) < asu_com_x_min_cart:
#             asu_com_x_min_cart = abs(asu_com_x_shifted_coords_cart[0])
#             x_min_shift_vec_frac = [x, 0, 0]
#
#     asu_com_y_min_cart = float('inf')
#     y_min_shift_vec_frac = None
#     for y in range(-10, 11):
#         asu_com_y_shifted_coords_frac = asu_com_frac + [0, y, 0]
#         asu_com_y_shifted_coords_cart = frac_to_cart(asu_com_y_shifted_coords_frac, uc_dimensions)
#         if abs(asu_com_y_shifted_coords_cart[1]) < asu_com_y_min_cart:
#             asu_com_y_min_cart = abs(asu_com_y_shifted_coords_cart[1])
#             y_min_shift_vec_frac = [0, y, 0]
#
#     if x_min_shift_vec_frac is not None and y_min_shift_vec_frac is not None:
#         xyz_min_shift_vec_frac = [x_min_shift_vec_frac[0], y_min_shift_vec_frac[1], 0]
#
#         if xyz_min_shift_vec_frac == [0, 0, 0]:
#             return pdb_asu
#
#         else:
#             pdb_asu_coords_frac = cart_to_frac(pdb_asu_coords_cart, uc_dimensions)
#             xyz_min_shifted_pdb_asu_coords_frac = pdb_asu_coords_frac + xyz_min_shift_vec_frac
#             xyz_min_shifted_pdb_asu_coords_cart = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)
#
#             xyz_min_shifted_asu_pdb = copy.deepcopy(pdb_asu)
#             xyz_min_shifted_asu_pdb.replace_coords(xyz_min_shifted_pdb_asu_coords_cart)
#             # xyz_min_shifted_asu_pdb.set_atom_coordinates(xyz_min_shifted_pdb_asu_coords_cart)
#
#             return xyz_min_shifted_asu_pdb
#
#     else:
#         return pdb_asu
#
#
# def get_central_asu_pdb_3d(pdb1, pdb2, uc_dimensions):
#     pdb_asu = PDB()
#     pdb_asu.read_atom_list(pdb1.atoms() + pdb2.atoms())
#
#     # pdb_asu_coords_cart = pdb_asu.extract_coords()  # TODO
#     pdb_asu_coords_cart = pdb_asu.extract_all_coords()
#
#     asu_com_cart = center_of_mass_3d(pdb_asu_coords_cart)
#     asu_com_frac = cart_to_frac(asu_com_cart, uc_dimensions)
#
#     asu_com_x_min_cart = float('inf')
#     x_min_shift_vec_frac = None
#     for x in range(-10, 11):
#         asu_com_x_shifted_coords_frac = asu_com_frac + [x, 0, 0]
#         asu_com_x_shifted_coords_cart = frac_to_cart(asu_com_x_shifted_coords_frac, uc_dimensions)
#         if abs(asu_com_x_shifted_coords_cart[0]) < abs(asu_com_x_min_cart):
#             asu_com_x_min_cart = asu_com_x_shifted_coords_cart[0]
#             x_min_shift_vec_frac = [x, 0, 0]
#
#     asu_com_y_min_cart = float('inf')
#     y_min_shift_vec_frac = None
#     for y in range(-10, 11):
#         asu_com_y_shifted_coords_frac = asu_com_frac + [0, y, 0]
#         asu_com_y_shifted_coords_cart = frac_to_cart(asu_com_y_shifted_coords_frac, uc_dimensions)
#         if abs(asu_com_y_shifted_coords_cart[1]) < abs(asu_com_y_min_cart):
#             asu_com_y_min_cart = asu_com_y_shifted_coords_cart[1]
#             y_min_shift_vec_frac = [0, y, 0]
#
#     asu_com_z_min_cart = float('inf')
#     z_min_shift_vec_frac = None
#     for z in range(-10, 11):
#         asu_com_z_shifted_coords_frac = asu_com_frac + [0, 0, z]
#         asu_com_z_shifted_coords_cart = frac_to_cart(asu_com_z_shifted_coords_frac, uc_dimensions)
#         if abs(asu_com_z_shifted_coords_cart[2]) < abs(asu_com_z_min_cart):
#             asu_com_z_min_cart = asu_com_z_shifted_coords_cart[2]
#             z_min_shift_vec_frac = [0, 0, z]
#
#     if x_min_shift_vec_frac is not None and y_min_shift_vec_frac is not None and z_min_shift_vec_frac is not None:
#         xyz_min_shift_vec_frac = [x_min_shift_vec_frac[0], y_min_shift_vec_frac[1], z_min_shift_vec_frac[2]]
#
#         if xyz_min_shift_vec_frac == [0, 0, 0]:
#             return pdb_asu
#
#         else:
#             pdb_asu_coords_frac = cart_to_frac(pdb_asu_coords_cart, uc_dimensions)
#             xyz_min_shifted_pdb_asu_coords_frac = pdb_asu_coords_frac + xyz_min_shift_vec_frac
#             xyz_min_shifted_pdb_asu_coords_cart = frac_to_cart(xyz_min_shifted_pdb_asu_coords_frac, uc_dimensions)
#
#             xyz_min_shifted_asu_pdb = copy.deepcopy(pdb_asu)
#             xyz_min_shifted_asu_pdb.replace_coords(xyz_min_shifted_pdb_asu_coords_cart)
#             # xyz_min_shifted_asu_pdb.set_atom_coordinates(xyz_min_shifted_pdb_asu_coords_cart)
#
#             return xyz_min_shifted_asu_pdb
#
#     else:
#         return pdb_asu
#
#
# def get_surrounding_unit_cells_2d(unit_cell_sym_mates, uc_dimensions):  # DEPRECIATE
#     """Returns a 3x3 grid of unit cells for a layer group"""
#     all_surrounding_unit_cells = []
#
#     asu_bb_atom_template = unit_cell_sym_mates[0].backbone_atoms
#     unit_cell_sym_mates_len = len(unit_cell_sym_mates)
#
#     central_uc_bb_cart_coords = []
#     for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
#         central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
#     central_uc_bb_frac_coords = cart_to_frac(central_uc_bb_cart_coords, uc_dimensions)
#
#     all_surrounding_uc_bb_frac_coords = []
#     for x_shift in [-1, 0, 1]:
#         for y_shift in [-1, 0, 1]:
#             if [x_shift, y_shift] != [0, 0]:
#                 shifted_uc_bb_frac_coords = central_uc_bb_frac_coords + [x_shift, y_shift, 0]
#                 all_surrounding_uc_bb_frac_coords.extend(shifted_uc_bb_frac_coords)
#
#     all_surrounding_uc_bb_cart_coords = frac_to_cart(all_surrounding_uc_bb_frac_coords, uc_dimensions)
#     all_surrounding_uc_bb_cart_coords = np.split(all_surrounding_uc_bb_cart_coords, 8)
#
#     for surrounding_uc_bb_cart_coords in all_surrounding_uc_bb_cart_coords:
#         all_uc_sym_mates_bb_cart_coords = np.split(surrounding_uc_bb_cart_coords, unit_cell_sym_mates_len)
#         one_surrounding_unit_cell = []
#         for uc_sym_mate_bb_cart_coords in all_uc_sym_mates_bb_cart_coords:
#             uc_sym_mate_bb_pdb = PDB()
#             uc_sym_mate_bb_atoms = []
#             atom_count = 0
#             for atom in asu_bb_atom_template:
#                 x_transformed = uc_sym_mate_bb_cart_coords[atom_count][0]
#                 y_transformed = uc_sym_mate_bb_cart_coords[atom_count][1]
#                 z_transformed = uc_sym_mate_bb_cart_coords[atom_count][2]
#                 atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
#                                         atom.get_residue_type(), atom.get_chain(),
#                                         atom.get_residue_number(),
#                                         atom.get_code_for_insertion(), x_transformed, y_transformed,
#                                         z_transformed,
#                                         atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
#                                         atom.get_atom_charge())
#                 uc_sym_mate_bb_atoms.append(atom_transformed)
#                 atom_count += 1
#
#             uc_sym_mate_bb_pdb.set_all_atoms(uc_sym_mate_bb_atoms)
#             one_surrounding_unit_cell.append(uc_sym_mate_bb_pdb)
#
#         all_surrounding_unit_cells.append(one_surrounding_unit_cell)
#
#     return all_surrounding_unit_cells
#
#
# def get_surrounding_unit_cells_3d(unit_cell_sym_mates, uc_dimensions):  # DEPRECIATE
#     """Returns a 3x3x3 grid of unit cells for a space group. Each unit cell is a list in list,
#     each ASU an item in the unit cell list"""
#     asu_bb_atom_template = unit_cell_sym_mates[0].backbone_atoms
#
#     central_uc_bb_cart_coords = []
#     for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
#         central_uc_bb_cart_coords.extend(unit_cell_sym_mate_pdb.extract_backbone_coords())
#     central_uc_bb_frac_coords = cart_to_frac(central_uc_bb_cart_coords, uc_dimensions)
#
#     all_surrounding_uc_bb_frac_coords = []
#     for x_shift in [-1, 0, 1]:
#         for y_shift in [-1, 0, 1]:
#             for z_shift in [-1, 0, 1]:
#                 if [x_shift, y_shift, z_shift] != [0, 0, 0]:
#                     shifted_uc_bb_frac_coords = central_uc_bb_frac_coords + [x_shift, y_shift, z_shift]
#                     all_surrounding_uc_bb_frac_coords.extend(shifted_uc_bb_frac_coords)
#
#     all_surrounding_uc_bb_cart_coords = frac_to_cart(all_surrounding_uc_bb_frac_coords, uc_dimensions)
#     all_surrounding_uc_bb_cart_coords = np.split(all_surrounding_uc_bb_cart_coords, 26)
#
#     all_surrounding_unit_cells = []
#     for surrounding_uc_bb_cart_coords in all_surrounding_uc_bb_cart_coords:
#         all_uc_sym_mates_bb_cart_coords = np.split(surrounding_uc_bb_cart_coords, len(unit_cell_sym_mates))
#         one_surrounding_unit_cell = []
#         for uc_sym_mate_bb_cart_coords in all_uc_sym_mates_bb_cart_coords:
#             uc_sym_mate_bb_pdb = PDB()
#             uc_sym_mate_bb_atoms = []
#             atom_count = 0
#             for atom in asu_bb_atom_template:
#                 x_transformed = uc_sym_mate_bb_cart_coords[atom_count][0]
#                 y_transformed = uc_sym_mate_bb_cart_coords[atom_count][1]
#                 z_transformed = uc_sym_mate_bb_cart_coords[atom_count][2]
#                 atom_transformed = Atom(atom.get_number(), atom.get_type(), atom.get_alt_location(),
#                                         atom.get_residue_type(), atom.get_chain(),
#                                         atom.get_residue_number(),
#                                         atom.get_code_for_insertion(), x_transformed, y_transformed,
#                                         z_transformed,
#                                         atom.get_occ(), atom.get_temp_fact(), atom.get_element_symbol(),
#                                         atom.get_atom_charge())
#                 uc_sym_mate_bb_atoms.append(atom_transformed)
#                 atom_count += 1
#
#             uc_sym_mate_bb_pdb.set_all_atoms(uc_sym_mate_bb_atoms)
#             one_surrounding_unit_cell.append(uc_sym_mate_bb_pdb)
#
#         all_surrounding_unit_cells.append(one_surrounding_unit_cell)
#
#     return all_surrounding_unit_cells


def write_unit_cell_sym_mates(unit_cell_sym_mates, outfile_path):  # Todo integrate with Model.py
    f = open(outfile_path, "a+")
    model_count = 0
    for unit_cell_sym_mate_pdb in unit_cell_sym_mates:
        model_count += 1
        model_line = "MODEL     {:>4s}\n".format(str(model_count))
        end_model_line = "ENDMDL\n"

        f.write(model_line)
        for atom in unit_cell_sym_mate_pdb.atoms():
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
            for atom in unit_cell_sym_mate_pdb.atoms():
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


def expanded_design_is_clash(asu_pdb_1, asu_pdb_2, design_dim, result_design_sym, expand_matrices, uc_dimensions=None,
                             outdir=None, output_exp_assembly=False, output_uc=False, output_surrounding_uc=False):
    if design_dim == 0:
        # Todo Pose from pdbs
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
