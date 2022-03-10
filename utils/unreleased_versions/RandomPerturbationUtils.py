import ast
import copy
import math
import os
import random
import re

import numpy as np

from PDB import PDB
from classes.SymEntry import get_optimal_external_tx_vector, get_uc_dimensions
from utils.SymmetryUtils import generate_cryst1_record


def get_rot_matrix_z(angle_deg):
    rad = math.radians(float(angle_deg))
    rotmatrix = [[math.cos(rad), -1 * math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]]
    return rotmatrix


def get_docked_pose_info(docked_pose_infofile_path):
    info_file = open(docked_pose_infofile_path, "r")
    info_file_lines = info_file.readlines()
    info_file.close()

    pdb1_path = None
    pdb2_path = None

    set_matrix1 = None
    set_matrix2 = None
    rot_degen_matrix1 = None
    rot_degen_matrix2 = None
    internal_tx1 = None
    internal_tx2 = None

    ref_frame_tx_dof1 = None
    ref_frame_tx_dof2 = None

    ref_frame_tx_param_e = None
    ref_frame_tx_param_f = None
    ref_frame_tx_param_g = None

    result_design_sym = None
    uc_spec_string = None

    has_int_rot_dof_1 = None
    has_int_rot_dof_2 = None

    for line in info_file_lines:
        if line.startswith('Original PDB 1 Path:'):
            pdb1_path = line.rstrip().split()[4]

        if line.startswith('Original PDB 2 Path:'):
            pdb2_path = line.rstrip().split()[4]

        if line.startswith('Oligomer 1 Reference Frame Tx DOF:'):
            ref_frame_tx_dof1 = line.rstrip().split()[6]

        if line.startswith('Oligomer 2 Reference Frame Tx DOF:'):
            ref_frame_tx_dof2 = line.rstrip().split()[6]

        if line.startswith("ROT/DEGEN MATRIX PDB1:"):
            rot_degen_matrix1_string = line.rstrip().split(': ')[1]
            rot_degen_matrix1 = ast.literal_eval(rot_degen_matrix1_string)

        if line.startswith("INTERNAL Tx PDB1:"):
            internal_tx1_string = line.rstrip().split(': ')[1]
            internal_tx1 = ast.literal_eval(internal_tx1_string)

        if line.startswith("SETTING MATRIX PDB1:"):
            set_matrix1_string = line.rstrip().split(': ')[1]
            set_matrix1 = ast.literal_eval(set_matrix1_string)

        if line.startswith("ROT/DEGEN MATRIX PDB2:"):
            rot_degen_matrix2_string = line.rstrip().split(': ')[1]
            rot_degen_matrix2 = ast.literal_eval(rot_degen_matrix2_string)

        if line.startswith("INTERNAL Tx PDB2:"):
            internal_tx2_string = line.rstrip().split(': ')[1]
            internal_tx2 = ast.literal_eval(internal_tx2_string)

        if line.startswith("SETTING MATRIX PDB2:"):
            set_matrix2_string = line.rstrip().split(': ')[1]
            set_matrix2 = ast.literal_eval(set_matrix2_string)

        if line.startswith("REFERENCE FRAME Tx PARAMETER(S):"):
            ref_frame_tx_param_line_string = line.rstrip()

            e_string = re.split('[, \s]', ref_frame_tx_param_line_string)[5]
            f_string = re.split('[, \s]', ref_frame_tx_param_line_string)[8]
            g_string = re.split('[, \s]', ref_frame_tx_param_line_string)[11]

            if e_string != 'None':
                ref_frame_tx_param_e = float(e_string)
            if f_string != 'None':
                ref_frame_tx_param_f = float(f_string)
            if g_string != 'None':
                ref_frame_tx_param_g = float(g_string)

        if line.startswith("Resulting Design Symmetry:"):
            result_design_sym = line.rstrip().split(': ')[1]

        if line.startswith("Unit Cell Specification:"):
            uc_spec_string = line.rstrip().split(': ')[1]

        if line.startswith("Oligomer 1 Internal ROT DOF:"):
            has_int_rot_dof_1_string = line.rstrip().split(': ')[1]
            has_int_rot_dof_1 = ast.literal_eval(has_int_rot_dof_1_string)

        if line.startswith("Oligomer 2 Internal ROT DOF:"):
            has_int_rot_dof_2_string = line.rstrip().split(': ')[1]
            has_int_rot_dof_2 = ast.literal_eval(has_int_rot_dof_2_string)

    return pdb1_path, pdb2_path, set_matrix1, set_matrix2, rot_degen_matrix1, rot_degen_matrix2, internal_tx1, internal_tx2, ref_frame_tx_dof1, ref_frame_tx_dof2, ref_frame_tx_param_e, ref_frame_tx_param_f, ref_frame_tx_param_g, result_design_sym, uc_spec_string, has_int_rot_dof_1, has_int_rot_dof_2


def rand_perturb(pdb1_path, pdb2_path, set_matrix1, set_matrix2, rot_degen_matrix1, rot_degen_matrix2, internal_tx1,
                 internal_tx2, ref_frame_tx_dof1, ref_frame_tx_dof2, ref_frame_tx_param_e, ref_frame_tx_param_f,
                 ref_frame_tx_param_g, result_design_sym, uc_spec_string, has_int_rot_dof_1, has_int_rot_dof_2,
                 number_of_perturbations, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pdb1 = PDB.from_file(pdb1_path)

    pdb2 = PDB.from_file(pdb2_path)

    for perturb_number in range(number_of_perturbations):
        pdb1_copy = copy.copy(pdb1)
        pdb2_copy = copy.copy(pdb2)

        pdb1_rand_perturb_path = outdir + "/" + os.path.splitext(os.path.basename(pdb1.filepath))[
            0] + "_rand_perturb_" + str(perturb_number) + ".pdb"
        pdb2_rand_perturb_path = outdir + "/" + os.path.splitext(os.path.basename(pdb2.filepath))[
            0] + "_rand_perturb_" + str(perturb_number) + ".pdb"

        logfile = open(outdir + "/" + "randomly_perturbed_log.txt", "a+")
        logfile.write("***** PERTURBATION %s *****\n" % str(perturb_number))
        logfile.write("PDB1 RANDOMLY PERTURBED PATH: %s\n" % pdb1_rand_perturb_path)
        logfile.write("PDB2 RANDOMLY PERTURBED PATH: %s\n\n" % pdb2_rand_perturb_path)

        # ROTATE
        if rot_degen_matrix1 is not None and has_int_rot_dof_1:
            rot1_perturbation_matrix = get_rot_matrix_z(random.choice(np.arange(-1.0, 1.1, 0.1)))
            rot1_matrix_perturbed = np.matmul(rot_degen_matrix1, rot1_perturbation_matrix)
            pdb1_copy.rotate(rot=rot1_matrix_perturbed)

            logfile.write("PDB1 ORIGINAL ROT/DEGEN MATRIX: %s\n" % rot_degen_matrix1)
            logfile.write("PDB1 RANDOM PERTURBATION ROT MATRIX: %s\n" % rot1_perturbation_matrix)
            logfile.write("PDB1 PERTURBED ROT/DEGEN MATRIX: %s\n\n" % rot1_matrix_perturbed.tolist())

        else:
            logfile.write("PDB1 ORIGINAL ROT/DEGEN MATRIX: %s\n" % rot_degen_matrix1)
            logfile.write("PDB1 RANDOM PERTURBATION ROT MATRIX: None\n")
            logfile.write("PDB1 PERTURBED ROT/DEGEN MATRIX: None\n\n")

        if rot_degen_matrix2 is not None and has_int_rot_dof_2:
            rot2_perturbation_matrix = get_rot_matrix_z(random.choice(np.arange(-1.0, 1.1, 0.1)))
            rot2_matrix_perturbed = np.matmul(rot_degen_matrix2, rot2_perturbation_matrix)
            pdb2_copy.rotate(rot=rot2_matrix_perturbed)

            logfile.write("PDB2 ORIGINAL ROT/DEGEN MATRIX: %s\n" % rot_degen_matrix2)
            logfile.write("PDB2 RANDOM PERTURBATION ROT MATRIX: %s\n" % rot2_perturbation_matrix)
            logfile.write("PDB2 PERTURBED ROT/DEGEN MATRIX: %s\n\n" % rot2_matrix_perturbed.tolist())

        else:
            logfile.write("PDB2 ORIGINAL ROT/DEGEN MATRIX: %s\n" % rot_degen_matrix2)
            logfile.write("PDB2 RANDOM PERTURBATION ROT MATRIX: None\n")
            logfile.write("PDB2 PERTURBED ROT/DEGEN MATRIX: None\n\n\n")

        # TRANSLATE INTERNALLY
        if internal_tx1 is not None:
            internal_tx1_z_perturbation = random.choice(np.arange(-1.0, 1.1, 0.1))
            internal_tx1_z_perturbed = [internal_tx1[0], internal_tx1[1], internal_tx1[2] + internal_tx1_z_perturbation]
            pdb1_copy.translate(internal_tx1_z_perturbed)

            logfile.write("PDB1 ORIGINAL INTERNAL Tx: : %s\n" % internal_tx1)
            logfile.write("PDB1 RANDOM PERTURBATION Tx: %s\n" % internal_tx1_z_perturbation)
            logfile.write("PDB1 PERTURBED INTERNAL Tx: %s\n\n" % internal_tx1_z_perturbed)

        else:
            logfile.write("PDB1 ORIGINAL INTERNAL Tx: None\n")
            logfile.write("PDB1 RANDOM PERTURBATION Tx: None\n")
            logfile.write("PDB1 PERTURBED INTERNAL Tx: None\n\n")

        if internal_tx2 is not None:
            internal_tx2_z_perturbation = random.choice(np.arange(-1.0, 1.1, 0.1))
            internal_tx2_z_perturbed = [internal_tx2[0], internal_tx2[1], internal_tx2[2] + internal_tx2_z_perturbation]
            pdb2_copy.translate(internal_tx2_z_perturbed)

            logfile.write("PDB2 ORIGINAL INTERNAL Tx: : %s\n" % internal_tx2)
            logfile.write("PDB2 RANDOM PERTURBATION Tx: %s\n" % internal_tx2_z_perturbation)
            logfile.write("PDB2 PERTURBED INTERNAL Tx: %s\n\n" % internal_tx2_z_perturbed)

        else:
            logfile.write("PDB2 ORIGINAL INTERNAL Tx: None\n")
            logfile.write("PDB2 RANDOM PERTURBATION Tx: None\n")
            logfile.write("PDB2 PERTURBED INTERNAL Tx: None\n\n")

        # SET
        pdb1_copy.rotate(rot=set_matrix1)
        logfile.write("PDB1 ORIGINAL SETTING MATRIX: %s\n" % set_matrix1)
        pdb2_copy.rotate(rot=set_matrix2)
        logfile.write("PDB2 ORIGINAL SETTING MATRIX: %s\n\n" % set_matrix2)

        # TRANSLATE IN REFERENCE FRAME
        e_perturbed = None
        f_perturbed = None
        g_perturbed = None
        ext_tx_vector1 = None
        ext_tx_vector2 = None
        perturbed_ext_dof_shifts = []
        if ref_frame_tx_param_e is not None:
            e_perturbed = ref_frame_tx_param_e + random.choice(np.arange(-1.0, 1.1, 0.1))
            perturbed_ext_dof_shifts.append(e_perturbed)
        if ref_frame_tx_param_f is not None:
            f_perturbed = ref_frame_tx_param_f + random.choice(np.arange(-1.0, 1.1, 0.1))
            perturbed_ext_dof_shifts.append(f_perturbed)
        if ref_frame_tx_param_g is not None:
            g_perturbed = ref_frame_tx_param_g + random.choice(np.arange(-1.0, 1.1, 0.1))
            perturbed_ext_dof_shifts.append(g_perturbed)

        if ref_frame_tx_dof1 != '<0,0,0>':
            ext_tx_vector1 = get_optimal_external_tx_vector(ref_frame_tx_dof1, perturbed_ext_dof_shifts)
            pdb1_copy.translate(ext_tx_vector1)

        if ref_frame_tx_dof2 != '<0,0,0>':
            ext_tx_vector2 = get_optimal_external_tx_vector(ref_frame_tx_dof2, perturbed_ext_dof_shifts)
            pdb2_copy.translate(ext_tx_vector2)

        logfile.write("PDB1 ORIGINAL REFERENCE FRAME Tx DOF: %s\n" % ref_frame_tx_dof1)
        logfile.write("PDB2 ORIGINAL REFERENCE FRAME Tx DOF: %s\n" % ref_frame_tx_dof2)
        logfile.write("ORIGINAL REFERENCE FRAME Tx PARAMETERS: e: %s, f: %s, g: %s\n" % (
        ref_frame_tx_param_e, ref_frame_tx_param_f, ref_frame_tx_param_g))
        logfile.write(
            "PERTURBED REFERENCE FRAME Tx PARAMETERS: e: %s, f: %s, g: %s\n" % (e_perturbed, f_perturbed, g_perturbed))
        logfile.write("PDB1 PERTURBED REFERENCE FRAME Tx: %s\n" % ext_tx_vector1)
        logfile.write("PDB2 PERTURBED REFERENCE FRAME Tx: %s\n\n" % ext_tx_vector2)

        # NEW CRYST1 RECORD
        cryst1_record = None
        if uc_spec_string != 'N/A':
            _e_perturbed = e_perturbed
            _f_perturbed = f_perturbed
            _g_perturbed = g_perturbed
            if e_perturbed is None:
                _e_perturbed = 0
            if f_perturbed is None:
                _f_perturbed = 0
            if g_perturbed is None:
                _g_perturbed = 0

            uc_dimensions = get_uc_dimensions(uc_spec_string, _e_perturbed, _f_perturbed, _g_perturbed)
            cryst1_record = generate_cryst1_record(uc_dimensions, result_design_sym)

        logfile.write("CRYST1 RECORD: " + str(cryst1_record) + "\n\n")

        # WRITE OUT RANDOMLY PERTURBED STRUCTURES
        pdb1_copy.write(pdb1_rand_perturb_path)
        pdb2_copy.write(pdb2_rand_perturb_path)

        logfile.close()


def sample_rot_tx_dof_coords(pdb, rot_step_deg=1, rot_range_deg=0, tx_step=1, start_tx_range=0, end_tx_range=0, axis="z", rotational_setting_matrix=None, degeneracy=None):

    def get_rot_matrices(step_deg, axis, rot_range_deg):
        rot_matrices = []
        if axis == 'x':
            for angle_deg in range(0, rot_range_deg, step_deg):
                rad = math.radians(float(angle_deg))
                rotmatrix = [[1, 0, 0], [0, math.cos(rad), -1 * math.sin(rad)], [0, math.sin(rad), math.cos(rad)]]
                rot_matrices.append(rotmatrix)
            return rot_matrices

        elif axis == 'y':
            for angle_deg in range(0, rot_range_deg, step_deg):
                rad = math.radians(float(angle_deg))
                rotmatrix = [[math.cos(rad), 0, math.sin(rad)], [0, 1, 0], [-1 * math.sin(rad), 0, math.cos(rad)]]
                rot_matrices.append(rotmatrix)
            return rot_matrices

        elif axis == 'z':
            for angle_deg in range(0, rot_range_deg, step_deg):
                rad = math.radians(float(angle_deg))
                rotmatrix = [[math.cos(rad), -1 * math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1]]
                rot_matrices.append(rotmatrix)
            return rot_matrices

        else:
            pdb.log.error('Axis selected for sampling is not supported!')
            return

    def get_tx_matrices(step, axis, start_range, end_range):
        if axis == "x":
            tx_matrices = []
            for dist in range(start_range, end_range, step):
                tx_matrices.append([dist, 0, 0])
            return tx_matrices

        elif axis == "y":
            tx_matrices = []
            for dist in range(start_range, end_range, step):
                tx_matrices.append([0, dist, 0])
            return tx_matrices

        elif axis == "z":
            tx_matrices = []
            for dist in range(start_range, end_range, step):
                tx_matrices.append([0, 0, dist])
            return tx_matrices

        else:
            pdb.log.error('Invalid sampling axis!')
            return

    def generate_sampled_coordinates_np(pdb_coordinates, rotation_matrices, translation_matrices, degeneracy_matrices):
        pdb_coords_np = np.array(pdb_coordinates)
        rot_matrices_np = np.array(rotation_matrices)
        degeneracy_matrices_rot_mat_np = np.array(degeneracy_matrices)

        if rotation_matrices is not None and translation_matrices is not None:
            if rotation_matrices == [] and translation_matrices == []:
                if degeneracy_matrices is not None:
                    degen_coords_np = np.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                    pdb_coords_degen_np = np.concatenate((degen_coords_np, np.expand_dims(pdb_coords_np, axis=0)))
                    return pdb_coords_degen_np
                else:
                    return np.expand_dims(pdb_coords_np, axis=0)

            elif rotation_matrices == [] and translation_matrices != []:
                if degeneracy_matrices is not None:
                    degen_coords_np = np.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                    pdb_coords_degen_np = np.concatenate((degen_coords_np, np.expand_dims(pdb_coords_np, axis=0)))
                    tx_sampled_coords = []
                    for tx_mat in translation_matrices:
                        tx_coords_np = pdb_coords_degen_np + tx_mat
                        tx_sampled_coords.extend(tx_coords_np)
                    return np.array(tx_sampled_coords)
                else:
                    tx_sampled_coords = []
                    for tx_mat in translation_matrices:
                        tx_coords_np = pdb_coords_np + tx_mat
                        tx_sampled_coords.append(tx_coords_np)
                    return np.array(tx_sampled_coords)

            elif rotation_matrices != [] and translation_matrices == []:
                if degeneracy_matrices is not None:
                    degen_coords_np = np.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                    pdb_coords_degen_np = np.concatenate((degen_coords_np, np.expand_dims(pdb_coords_np, axis=0)))
                    degen_rot_pdb_coords = []
                    for degen_coord_set in pdb_coords_degen_np:
                        degen_rot_np = np.matmul(degen_coord_set, rot_matrices_np)
                        degen_rot_pdb_coords.extend(degen_rot_np)
                    return np.array(degen_rot_pdb_coords)
                else:
                    rot_coords_np = np.matmul(pdb_coords_np, rot_matrices_np)
                    return rot_coords_np
            else:
                if degeneracy_matrices is not None:
                    degen_coords_np = np.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                    pdb_coords_degen_np = np.concatenate((degen_coords_np, np.expand_dims(pdb_coords_np, axis=0)))
                    degen_rot_pdb_coords = []
                    for degen_coord_set in pdb_coords_degen_np:
                        degen_rot_np = np.matmul(degen_coord_set, rot_matrices_np)
                        degen_rot_pdb_coords.extend(degen_rot_np)
                    degen_rot_pdb_coords_np = np.array(degen_rot_pdb_coords)
                    tx_sampled_coords = []
                    for tx_mat in translation_matrices:
                        tx_coords_np = degen_rot_pdb_coords_np + tx_mat
                        tx_sampled_coords.extend(tx_coords_np)
                    return np.array(tx_sampled_coords)
                else:
                    rot_coords_np = np.matmul(pdb_coords_np, rot_matrices_np)
                    tx_sampled_coords = []
                    for tx_mat in translation_matrices:
                        tx_coords_np = rot_coords_np + tx_mat
                        tx_sampled_coords.extend(tx_coords_np)
                    return np.array(tx_sampled_coords)
        else:
            return

    rot_matrices = get_rot_matrices(rot_step_deg, axis, rot_range_deg)
    tx_matrices = get_tx_matrices(tx_step, axis, start_tx_range, end_tx_range)
    sampled_coords_np = generate_sampled_coordinates_np(pdb.coords, rot_matrices, tx_matrices, degeneracy)

    if sampled_coords_np is not None:
        if rotational_setting_matrix is not None:
            rotational_setting_matrix_np = np.array(rotational_setting_matrix)
            rotational_setting_matrix_np_t = np.transpose(rotational_setting_matrix_np)
            sampled_coords_orient_canon_np = np.matmul(sampled_coords_np, rotational_setting_matrix_np_t)
            return sampled_coords_orient_canon_np.tolist()
        else:
            return sampled_coords_np.tolist()
    else:
        return


if __name__ == '__main__':
    infofile_path = '/Users/jlaniado/Desktop/MP_TEST_P432/4f47_4grd/DEGEN_1_1/ROT_3_1/tx_5/matching_fragment_representatives/frag_match_info_file.txt'
    number_of_perturbations = 50
    outdir = '/Users/jlaniado/Desktop/RAND_PERTURB'
    pdb1_path, pdb2_path, set_matrix1, set_matrix2, rot_degen_matrix1, rot_degen_matrix2, internal_tx1, internal_tx2, \
        ref_frame_tx_dof1, ref_frame_tx_dof2, ref_frame_tx_param_e, ref_frame_tx_param_f, ref_frame_tx_param_g, \
        result_design_sym, uc_spec_string, has_int_rot_dof_1, has_int_rot_dof_2 = get_docked_pose_info(infofile_path)
    rand_perturb(pdb1_path, pdb2_path, set_matrix1, set_matrix2, rot_degen_matrix1, rot_degen_matrix2, internal_tx1,
                 internal_tx2, ref_frame_tx_dof1, ref_frame_tx_dof2, ref_frame_tx_param_e, ref_frame_tx_param_f,
                 ref_frame_tx_param_g, result_design_sym, uc_spec_string, has_int_rot_dof_1, has_int_rot_dof_2,
                 number_of_perturbations, outdir)
