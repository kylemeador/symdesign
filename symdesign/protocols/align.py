from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from itertools import count
from typing import Generator, Iterable, Iterator

import numpy as np
# from memory_profiler import profile
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from symdesign import flags, utils, structure
from symdesign.protocols.pose import PoseJob
from symdesign.resources import job as symjob, sql
from symdesign.structure.base import Residue, SS_HELIX_IDENTIFIERS, Structure, termini_literal
from symdesign.structure.coords import superposition3d
from symdesign.structure.model import Chain, Entity, Model, Pose
from symdesign.structure.utils import chain_id_generator, DesignError
from symdesign.utils import types
from symdesign.utils.SymEntry import SymEntry, parse_symmetry_to_sym_entry
putils = utils.path
logger = logging.getLogger(__name__)


class AngleDistance:
    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2
        self.vec1 = [axis1[2][0] - axis1[0][0], axis1[2][1] - axis1[0][1], axis1[2][2] - axis1[0][2]]
        self.vec2 = [axis2[2][0] - axis2[0][0], axis2[2][1] - axis2[0][1], axis2[2][2] - axis2[0][2]]
        self.length_1 = self.length(self.vec1)
        self.length_2 = self.length(self.vec2)

    def length(self, vec):
        length = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
        return length

    def cos_angle(self):
        if self.length_1 != 0 and self.length_2 !=0:
            cosangle = (self.vec1[0] / self.length_1) * (self.vec2[0] / self.length_2) + (self.vec1[1] / self.length_1) * (self.vec2[1] / self.length_2) +(self.vec1[2] / self.length_1) * (self.vec2[2] / self.length_2)
            return cosangle
        else:
            return 0

    def angle(self):
        angle = (math.acos(abs(self.cos_angle()))*180)/math.pi
        return angle

    def distance(self):
        crossproduct = [self.vec1[1] * self.vec2[2] - self.vec1[2] * self.vec2[1], self.vec1[2] * self.vec2[0] - self.vec1[0] * self.vec2[2], self.vec1[0] * self.vec2[1] - self.vec1[1] * self.vec2[0]]
        crossproduct_length = math.sqrt((crossproduct[0] * crossproduct[0]) + (crossproduct[1] * crossproduct[1]) + (crossproduct[2] * crossproduct[2]))
        connect_vec1_vec2 = [self.axis1[0][0] - self.axis2[0][0], self.axis1[0][1] - self.axis2[0][1], self.axis1[0][2] - self.axis2[0][2]]
        distance = abs(crossproduct[0] * connect_vec1_vec2[0] + crossproduct[1] * connect_vec1_vec2[1] + crossproduct[2] * connect_vec1_vec2[2]) / float(crossproduct_length)
        return distance

    def is_parallel(self, err=5):
        if (self.angle() >= 180 - err and self.angle() <= 180) or (self.angle() >= 0 and self.angle() <= 0 + err):
            return True
        else:
            return False

    def is_90(self, err=10):
        if (self.angle() >= 90 - err and self.angle() <= 90) or (self.angle() >= 90 and self.angle() <= 90 + err):
            return True
        else:
            return False

    def is_35(self, err=10):
        if (self.angle() >= 35 - err and self.angle() <= 35) or (self.angle() >= 35 and self.angle() <= 35 + err):
            return True
        else:
            return False

    def is_55(self, err=10):
        if (self.angle() >= 55 - err and self.angle() <= 55) or (self.angle() >= 55 and self.angle() <= 55 + err):
            return True
        else:
            return False


class HelixFusion:
    def __init__(self, target_protein_path,  targetprotein_term, targetprotein_symm, orient_target, add_target_helix, oligomer_list_path, oligomer_term, oligomer_symm, work_dir):
        self.target_protein_path = target_protein_path
        self.targetprotein_term = targetprotein_term
        self.targetprotein_symm = targetprotein_symm
        self.orient_target = orient_target
        self.add_target_helix = add_target_helix  # bool?, termini, chain id
        self.oligomer_list_path = oligomer_list_path
        self.oligomer_term = oligomer_term
        self.oligomer_symm = oligomer_symm
        self.work_dir = work_dir

    def run(self):
        # Make Directory for Design Candidates if it Doesn't Exist Already
        design_directory = os.path.join(self.work_dir, 'DESIGN_CANDIDATES')
        # if not os.path.exists(design_directory):
        os.makedirs(design_directory, exist_ok=True)
        # Orient Target Protein if desired
        if os.path.exists(self.target_protein_path):
            target_protein = structure.model.Model.from_file(self.target_protein_path)
            if self.orient_target:
                print('Orienting Target Molecule')
                target_protein.orient(symmetry=self.targetprotein_symm)
                print('Done Orienting Target Molecule')
        else:
            print('Could Not Find Target PDB File')
            return -1

        # Add Ideal 10 Ala Helix to Target if desired
        if self.add_target_helix[0]:
            print('Adding Ideal Ala Helix to Target Molecule')
            target_protein.add_ideal_helix(self.add_target_helix[1], self.add_target_helix[2])
            if self.add_target_helix[1] == 'N':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[0].residue_number
            elif self.add_target_helix[1] == 'C':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[-1].residue_number - 9

            print("Done Adding Ideal Ala Helix to Target Molecule")
        else:
            target_term_resi = self.add_target_helix[1]

        # Add Axis / Axes to Target Molecule
        raise NotImplementedError("Axis addition isn't supported to Model class")
        if self.targetprotein_symm[0:1] == 'C':
            target_protein.AddCyclicAxisZ()
        elif self.targetprotein_symm == 'D2':
            target_protein.AddD2Axes()
        else:
            print('Target Protein Symmetry Not Supported')
            return -1

        # Fetch Oligomer PDB files
        fetch_oligomers = FetchPDBBA(self.oligomer_list_path)
        fetch_oligomers.fetch()

        # Try To Correct State issues
        print('Trying To Correct State Issues')
        oligomer_id_listfile = ListFile()
        oligomer_id_listfile.read(self.oligomer_list_path)
        oligomer_id_list = oligomer_id_listfile.list_file
        for oligomer_id in oligomer_id_list:
            oligomer_filepath = os.path.join(self.work_dir, f'{oligomer_id}.pdb1')
            correct_oligomer_state = structure.model.Model.from_file(oligomer_filepath)
            correct_state_out_path = os.path.splitext(oligomer_filepath)[0] + '.pdb'
            correct_oligomer_state.write(out_path=correct_state_out_path)

        for oligomer_id in oligomer_id_list:
            correct_state_oligomer_filepath = os.path.join(self.work_dir, f'{oligomer_id}.pdb')
            print('Orienting Oligomer')
            # Read in Moving PDB
            pdb_oligomer = Model.from_file(correct_state_oligomer_filepath)
            print('Done Orienting Oligomer')

            print('Fusing Target To Oligomer')
            for i in range(6):
                # Run Stride On Oligomer
                if self.oligomer_term in 'NnCc':
                    if pdb_oligomer.is_termini_helical(self.oligomer_term):
                        oligomer_term_resi = \
                            getattr(pdb_oligomer, f'{self.oligomer_term.lower()}_terminial_residue').number
                    else:
                        print("Oligomer termini isn't helical")
                        continue
                else:
                    print('Select N or C Terminus For Oligomer')
                    return -1

                if isinstance(oligomer_term_resi, int):
                    # Add Axis / Axes to Oligomers
                    if self.oligomer_symm[:1] == 'C':
                        pdb_oligomer.AddCyclicAxisZ()
                    elif self.targetprotein_symm == 'D2':
                        pdb_oligomer.AddD2Axes()
                    else:
                        print('Oligomer Symmetry Not Supported')
                        return -1

                    # Extract coordinates of segment to be overlapped from PDB Fixed
                    pdb_fixed_coords = target_protein.chain(self.add_target_helix[2])\
                        .get_coords_subset(start=target_term_resi + i, end=target_term_resi + 4 + i)
                    # Extract coordinates of segment to be overlapped from PDB Moving
                    pdb_mobile_coords = pdb_oligomer.get_coords_subset(
                        start=oligomer_term_resi, end=oligomer_term_resi + 4)

                    rmsd, rot, tx = superposition3d(pdb_fixed_coords, pdb_mobile_coords)

                    # Apply optimal rot and tx to PDB moving axis (does NOT change axis coordinates in instance)
                    pdb_moving_axes = PDB()  # Todo this is outdated

                    if self.oligomer_symm == 'D2':
                        pdb_moving_axes.AddD2Axes()
                        pdb_moving_axes.transform(rot, tx)
                        moving_axis_x = pdb_moving_axes.axisX()
                        moving_axis_y = pdb_moving_axes.axisY()
                        moving_axis_z = pdb_moving_axes.axisZ()
                    elif self.oligomer_symm[0:1] == 'C':
                        pdb_moving_axes.AddCyclicAxisZ()
                        pdb_moving_axes.transform(rot, tx)
                        moving_axis_z = pdb_moving_axes.axisZ()
                    else:
                        print('Oligomer Symmetry Not Supported')
                        return -1

                    # # Check Angle Between Fixed and Moved Axes

                    # D2_D2 3D Crystal Check
                    #angle_check_1 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    # is_parallel_1 = angle_check_1.is_parallel()
                    #
                    # if is_parallel_1:
                    #     pdb_oligomer.apply(rot, tx)
                    #     pdb_oligomer.rename_chains(target_protein.chain_ids)
                    #
                    #     PDB_OUT = PDB()
                    #     PDB_OUT.read_atom_list(target_protein.all_atoms + pdb_oligomer.all_atoms)
                    #
                    #     out_path = design_directory + "/" + os.path.basename(self.target_protein_path)[0:4] + "_" + oligomer_id + "_" + str(i) + ".pdb"
                    #     outfile = open(out_path, "w")
                    #     for atom in PDB_OUT.all_atoms:
                    #         outfile.write(str(atom))
                    #     outfile.close()

                    # D2_C3 3D Crystal I4132 Check
                    # angle_check_1 = AngleDistance(target_protein.axisX(), moving_axis_z)
                    # is_90_1 = angle_check_1.is_90()
                    # angle_check_2 = AngleDistance(target_protein.axisY(), moving_axis_z)
                    # is_90_2 = angle_check_2.is_90()
                    # angle_check_3 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    # is_90_3 = angle_check_3.is_90()
                    #
                    # angle_check_4 = AngleDistance(target_protein.axisX(), moving_axis_z)
                    # is_35_1 = angle_check_4.is_35()
                    # angle_check_5 = AngleDistance(target_protein.axisY(), moving_axis_z)
                    # is_35_2 = angle_check_5.is_35()
                    # angle_check_6 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    # is_35_3 = angle_check_6.is_35()
                    #
                    # angle_check_7 = AngleDistance(target_protein.axisX(), moving_axis_z)
                    # is_55_1 = angle_check_7.is_55()
                    # angle_check_8 = AngleDistance(target_protein.axisY(), moving_axis_z)
                    # is_55_2 = angle_check_8.is_55()
                    # angle_check_9 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    # is_55_3 = angle_check_9.is_55()
                    #
                    # check_90 = [is_90_1, is_90_2, is_90_3]
                    # check_35 = [is_35_1, is_35_2, is_35_3]
                    # check_55= [is_55_1, is_55_2, is_55_3]
                    #
                    # count_90 = 0
                    # for test in check_90:
                    #     if test is True:
                    #         count_90 = count_90 + 1
                    #
                    # count_35 = 0
                    # for test in check_35:
                    #     if test is True:
                    #         count_35 = count_35 + 1
                    #
                    # count_55 = 0
                    # for test in check_55:
                    #     if test is True:
                    #         count_55 = count_55 + 1
                    #
                    # if count_90 > 0 and count_35 > 0 and count_55 > 0:
                    #     for k in [0, 1, 2]:
                    #         if check_90[k] is True:
                    #             check_90_index = k
                    #
                    #     if check_90_index == 0:
                    #         axis_90 = target_protein.axisX()
                    #     elif check_90_index == 1:
                    #         axis_90 = target_protein.axisY()
                    #     else:
                    #         axis_90 = target_protein.axisZ()
                    #
                    #     distance_check_1 = AngleDistance(axis_90, moving_axis_z)
                    #
                    #     if distance_check_1.distance() <= 5:
                    #
                    #             pdb_oligomer.apply(rot, tx)
                    #             pdb_oligomer.rename_chains(target_protein.chain_ids)
                    #
                    #             PDB_OUT = PDB()
                    #             PDB_OUT.read_atom_list(target_protein.all_atoms + pdb_oligomer.all_atoms)
                    #
                    #             out_path = design_directory + "/" + os.path.basename(self.target_protein_path)[0:4] + "_" + oligomer_id + "_" + str(i) + ".pdb"
                    #             outfile = open(out_path, "w")
                    #             for atom in PDB_OUT.all_atoms:
                    #                 outfile.write(str(atom))
                    #             outfile.close()

                    # D2_C3 2D Layer Check p622 Check
                    angle_check_1 = AngleDistance(target_protein.axisX(), moving_axis_z)
                    is_parallel_1 = angle_check_1.is_parallel()
                    angle_check_2 = AngleDistance(target_protein.axisY(), moving_axis_z)
                    is_parallel_2 = angle_check_2.is_parallel()
                    angle_check_3 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    is_parallel_3 = angle_check_3.is_parallel()

                    check_parallel = [is_parallel_1, is_parallel_2, is_parallel_3]
                    count_parallel = 0
                    for test in check_parallel:
                        if test is True:
                            count_parallel = count_parallel + 1

                    angle_check_4 = AngleDistance(target_protein.axisX(), moving_axis_z)
                    is_90_1 = angle_check_4.is_90()
                    angle_check_5 = AngleDistance(target_protein.axisY(), moving_axis_z)
                    is_90_2 = angle_check_5.is_90()
                    angle_check_6 = AngleDistance(target_protein.axisZ(), moving_axis_z)
                    is_90_3 = angle_check_6.is_90()

                    check_90 = [is_90_1, is_90_2, is_90_3]
                    count_90 = 0
                    for test in check_90:
                        if test is True:
                            count_90 = count_90 + 1

                    if count_parallel > 0 and count_90 > 0:
                        for k in [0, 1, 2]:
                            if check_90[k] is True:
                                check_90_index = k

                        if check_90_index == 0:
                            axis_90 = target_protein.axisX()
                        elif check_90_index == 1:
                            axis_90 = target_protein.axisY()
                        else:
                            axis_90 = target_protein.axisZ()

                        distance_check_1 = AngleDistance(axis_90, moving_axis_z)

                        if distance_check_1.distance() <= 3:
                            pdb_oligomer.apply(rot, tx)
                            pdb_oligomer.rename_chains(exclude_chains=target_protein.chain_ids)

                            out_pdb = Model.from_atoms(target_protein.atoms + pdb_oligomer.atoms)

                            out_path = os.path.join(design_directory,
                                                    '%s_%s_%d.pdb' % (os.path.basename(self.target_protein_path)[0:4], oligomer_id, i))
                            out_pdb.write(out_path=out_path)

        print('Done')


modes3x4_F = np.array([
    [[0.9961, 0.0479, 0.0742, -0.020],
     [-0.0506, 0.9981, 0.0345, -0.042],
     [-0.0724, -0.0381, 0.9966, -0.029]],
    [[0.9985, -0.0422, 0.0343, 0.223],
     [0.0425, 0.9991, -0.0082, 0.039],
     [-0.0340, 0.0097, 0.9994, -0.120]],
    [[1.0000, -0.0027, -0.0068, 0.001],
     [0.0023, 0.9981, -0.0622, -0.156],
     [0.0069, 0.0622, 0.9980, -0.191]],
    [[0.9999, -0.0092, 0.0084, -0.048],
     [0.0091, 0.9999, 0.0128, -0.108],
     [-0.0085, -0.0127, 0.9999, 0.043]],
    [[0.9999, 0.0055, 0.0121, -0.105],
     [-0.0055, 1.0000, -0.0009, 0.063],
     [-0.0121, 0.0008, 0.9999, 0.051]],
    [[0.9999, 0.0011, -0.0113, -0.027],
     [-0.0012, 1.0000, -0.0071, 0.009],
     [0.0113, 0.0071, 0.9999, -0.102]],
    [[1.0000, 0.0020, -0.0002, 0.022],
     [-0.0020, 1.0000, -0.0009, 0.030],
     [0.0002, 0.0009, 1.0000, -0.005]],
    [[1.0000, -0.0019, 0.0001, 0.011],
     [0.0019, 1.0000, 0.0001, -0.016],
     [-0.0001, -0.0001, 1.0000, 0.001]],
    [[1.0000, 0.0020, 0.0001, 0.013],
     [-0.0020, 1.0000, 0.0000, 0.007],
     [-0.0001, -0.0000, 1.0000, 0.001]]
])
modes3x4_R = np.array([
    [[0.9984, 0.0530, 0.0215, -0.023],
     [-0.0546, 0.9951, 0.0820, 0.082],
     [-0.0170, -0.0830, 0.9964, 0.026]],
    [[0.9985, 0.0543, 0.0027, -0.080],
     [-0.0541, 0.9974, -0.0473, 0.179],
     [-0.0052, 0.0471, 0.9989, 0.075]],
    [[0.9979, -0.0042, -0.0639, 0.157],
     [0.0032, 0.9999, -0.0156, 0.062],
     [0.0640, 0.0154, 0.9978, -0.205]],
    [[0.9999, 0.0002, 0.0120, 0.050],
     [-0.0002, 1.0000, 0.0008, 0.171],
     [-0.0120, -0.0008, 0.9999, -0.014]],
    [[1.0000, 0.0066, -0.0033, -0.086],
     [-0.0066, 0.9999, 0.0085, 0.078],
     [0.0034, -0.0085, 1.0000, 0.053]],
    [[0.9999, -0.0026, 0.0097, 0.023],
     [0.0025, 0.9999, 0.0129, -0.017],
     [-0.0097, -0.0129, 0.9999, 0.123]],
    [[1.0000, -0.0019, -0.0017, -0.029],
     [0.0019, 1.0000, -0.0014, -0.031],
     [0.0017, 0.0014, 1.0000, -0.018]],
    [[1.0000, -0.0035, 0.0002, -0.011],
     [0.0035, 1.0000, 0.0002, -0.017],
     [-0.0002, -0.0002, 1.0000, 0.002]],
    [[1.0000, 0.0007, -0.0001, -0.017],
     [-0.0007, 1.0000, -0.0001, 0.008],
     [0.0001, 0.0001, 1.0000, -0.001]]
])


def vdot3(a, b):
    dot = 0.
    for i in range(3):
        dot += a[i] * b[i]

    return dot


def vnorm3(a):
    b = [0., 0., 0.]
    dot = 0.
    for i in a:
        dot += i ** 2

    dot_root = math.sqrt(dot)
    for idx, i in enumerate(a):
        b[idx] = i / dot_root

    return b


def vcross(a, b):
    c = [0., 0., 0.]
    for i in range(3):
        c[i] = a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3]

    return c


def norm(a):
    b = np.dot(a, a)
    return a / np.sqrt(b)


def cross(a, b):
    c = np.zeros(3)
    for i in range(3):
        c[i] = a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3]
    return c


def make_guide(n_ca_c_atoms: np.ndarray, scale: float) -> np.ndarray:
    """
    Take 3 atom positions in a 3x3 array (vectors as columns) representing
    N, Ca, C, atoms, and return 3 guide position vectors.  The 1st vector is the
    Ca position, the second is displaced from the Ca position
    along the direction to the C atom with a length
    set by the scale quantity. The 3rd position is likewise offset from the
    Ca position along a direction in the
    plane of the 3 atoms given, also with length given by scale

    Args:
        n_ca_c_atoms:
        scale:

    Returns:

    """
    ca = n_ca_c_atoms[:, 1].flatten()
    v1 = n_ca_c_atoms[:, 2].flatten() - ca
    v2 = n_ca_c_atoms[:, 0].flatten() - ca
    v1n = norm(v1)
    v2t = v2 - v1n * np.dot(v2, v1n)
    v2tn = norm(v2t)

    #    print(np.dot(v1n,v2tn))

    guide1 = ca + scale * v1n
    guide2 = ca + scale * v2tn
    #
    #    print(ca,guide1,guide2)
    guide = np.zeros((3, 3))
    guide[:, 0], guide[:, 1], guide[:, 2] = ca, guide1, guide2

    return guide


def get_frame_from_joint(joint_points: np.ndarray) -> np.ndarray:
    """Create a 'frame' which consists of a matrix with

    Returns:
        The Fortran ordered array with shape (3, 4) that contains 3 basis vectors (x, y, z) of the point in question
        along the first 3 columns, then the 4th column is the translation to the provided joint_point
    """
    guide_target_1 = make_guide(joint_points, 1.)
    ca = joint_points[:, 1].flatten()
    v1 = guide_target_1[:, 1] - ca
    v2 = guide_target_1[:, 2] - ca
    v3 = cross(v1, v2)
    rot = np.array([v1, v2, v3]).T
    # print ('frame rot: ', rot)
    # print ('frame trans: ', guide_points[:,1], guide_target_1[:,0])
    frame_out = np.zeros((3, 4))
    frame_out[:, 0:3] = rot
    frame_out[:, 3] = joint_points[:, 1]
    return frame_out


def invert_3x4(in3x4: np.ndarray) -> np.ndarray:
    rin = in3x4[:, 0:3]
    tin = in3x4[:, 3]

    rout = np.linalg.inv(rin)
    tout = -np.matmul(rout, tin)
    out3x4 = np.zeros((3, 4))
    out3x4[:, 0:3] = rout
    out3x4[:, 3] = tout

    return out3x4


def compose_3x4(a3x4: np.ndarray, b3x4: np.ndarray) -> np.ndarray:
    """Apply a rotation and translation of one array with shape (3, 4) to another array with same shape"""
    r1 = a3x4[:, 0:3]
    t1 = a3x4[:, 3]
    # r2=b3x4[:,0:3]
    # t2=b3x4[:,3]

    c3x4 = np.matmul(r1, b3x4)
    c3x4[:, 3] += t1
    # print('rot: ', np.matmul(r2,r1))
    # print('trans: ', np.matmul(r2,t1) + t2)

    return c3x4


def combine_modes(modes3x4: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    rot_delta = np.zeros([3, 3])
    tx_delta = np.zeros(3)
    roti = np.identity(3)
    for i in range(len(coeffs)):
        c = coeffs[i]
        tx_delta += modes3x4[i, :, 3] * c
        rot_delta += (modes3x4[i, :, 0:3] - roti) * c
    rtmp = roti + rot_delta
    # print ('unnormalized rot:\n', rtmp)
    # normalize r
    rc1 = rtmp[:, 0]
    rc1n = np.array(vnorm3(rc1))
    rc2 = rtmp[:, 1]
    dot12 = vdot3(rc1n, rc2)
    rc2p = np.array(rc2) - dot12 * rc1n
    rc2pn = np.array(vnorm3(rc2p))
    rc3 = np.array(vcross(rc1n, rc2pn))

    rot_out = np.array([rc1n, rc2pn, rc3]).T
    # print ('normalized rot:\n', rot_out)
    # rcheck = np.matmul(rot_out, rot_out.T)
    # print (rcheck)
    blended_mode = np.concatenate((rot_out, np.array([tx_delta]).T), axis=1)
    # print ('blended mode output:\n', blended_mode)

    return blended_mode


# model_ideal_helix = Structure.from_atoms(alpha_helix_15_atoms)


def generate_bend_transformations(joint_residue: Residue, direction: termini_literal = None) \
        -> Generator[types.TransformationMapping, None, None]:
    """Generate transformations compatible with bending a helix at a Residue in that helix in either the 'n' or 'c'
    terminal direction

    Args:
        joint_residue: The Residue where the bending should be applied
        direction: Specify which direction, compared to the Residue, should be bent.
            'c' bends the c-terminal residues, 'n' the n-terminal residues
    Returns:
        A generator which yields a transformation mapping for a single 'bending mode' upon each access
    """
    # Todo KM changed F to c, R to n where the side that should bend is c-terminal of the specified index
    #  So c is Todd's F direction
    if direction == 'c':  # 'F'
        # modes3x4 = get_3x4_modes_d(job.job.direction)
        modes3x4 = modes3x4_F
    elif direction == 'n':  # 'R'
        modes3x4 = modes3x4_R
    else:
        raise ValueError(
            f"'direction' must be either 'n' or 'c', not {direction}")

    # model_residues = model.get_residues(indices=list(range(joint_index - 2, joint_index + 3)))
    # model_coords = model.get_coords_subset(residue_numbers=[r.number for r in model_residues], dtype='backbone')
    # model_coords = model.get_coords_subset(indices=list(range(joint_index - 2, joint_index + 3)), dtype='backbone')
    # Todo make dependent on the helix length?
    # helix_residue_num = 3
    # ideal_coords = model_ideal_helix.get_coords_subset(
    #     residue_numbers=list(range(helix_residue_num - 2, helix_residue_num + 3)), dtype='backbone')
    # if len(model_coords) != len(ideal_coords):
    #     # The residue selection failed
    #     # raise DesignError(
    #     # flags.format_args(flags.joint_residue_args)}
    #     logger.warning(
    #         # f"The number of residues selected from the index {joint_index}, {len(model_residues)} != "
    #         # f"{len(list(range(helix_residue_num - 2, helix_residue_num + 3)))}, length of aligned residues. "
    #         f"Couldn't perform superposition. The number of model coords, {len(model_coords)} != {len(ideal_coords)}, "
    #         f"the number of ideal coords")
    #     return []

    # rmsd, rot_ideal_onto_fixed, tx_ideal_onto_fixed = superposition3d(model_coords, ideal_coords)
    # model_ideal_helix.transform(rotation=rot_ideal_onto_fixed, translation=tx_ideal_onto_fixed)
    # Todo remove the transform. It doesn't appear to be necessary.
    #  Just get the coords of the center residue from model
    # Get the Nitrogen, Ca and C atoms of the ideal_moved helix
    # ideal_center_residue = model_ideal_helix.residue(helix_residue_num)
    # ideal_joint_in_fixed_frame = np.array(
    #     [ideal_center_residue.n.coords, ideal_center_residue.ca.coords, ideal_center_residue.c.coords]).T
    # joint_frame = get_frame_from_joint(ideal_joint_in_fixed_frame)

    # joint_residue = model.residues[joint_index]
    model_frame = np.array(
        [joint_residue.n.coords, joint_residue.ca.coords, joint_residue.c.coords]).T
    joint_frame = get_frame_from_joint(model_frame)
    # print ('joint_frame:\n',joint_frame)
    jinv = invert_3x4(joint_frame)
    # Fixed parameters
    bend_dim = 4
    bend_scale = 1.
    # ntaper = 5

    # Generate various transformations
    while True:
        bend_coeffs = np.random.normal(size=bend_dim) * bend_scale
        blend_mode = combine_modes(modes3x4, bend_coeffs)

        # Compose a trial bending mode in the frame of the fixed structure
        tmp1 = compose_3x4(blend_mode, jinv)
        mode_in_frame = compose_3x4(joint_frame, tmp1)
        # print('mode_in_frame:\n', mode_in_frame)

        # Separate the operations to their components
        rotation = mode_in_frame[:, 0:3]
        translation = mode_in_frame[:, 3].flatten()
        yield dict(rotation=rotation, translation=translation)


def bend(pose: Pose, joint_residue: Residue, direction: termini_literal, samples: int = 1,
         additional_entity_ids: Iterator[str] = None) -> list[np.ndarray]:
    """Bend a Pose at a helix specified by a Residue on the helix according to typical helix bending modes

    Args:
        pose: The Pose of interest to generate bent coordinates for
        joint_residue: The Residue where the bending should be applied
        direction: Specify which termini of the joint_residue Entity, compared to the joint_residue, should be bent.
            'c' bends the c-terminal coordinates, 'n' the n-terminal coordinates
        samples: How many times should the coordinates be bent
        additional_entity_ids: If there are additional Entity instances desired to be carried through bending,
            pass their Entity.name attributes
    Returns:
        A list of the transformed pose coordinates at the bent site
    """
    residue_chain = pose.chain(joint_residue.chain_id)
    # bending_entity = pose.match_entity_by_seq(residue_chain.sequence)
    bending_entity = pose.entity(residue_chain.entity_id)

    # Todo KM changed F to c, R to n where the side that should bend is c-terminal of the specified index
    #  So c is Todd's F direction
    if direction == 'n':
        set_coords_slice = slice(bending_entity.n_terminal_residue.start_index, joint_residue.start_index)
    elif direction == 'c':
        set_coords_slice = slice(joint_residue.end_index + 1, bending_entity.c_terminal_residue.end_index)
    else:
        raise ValueError(
            f"'direction' must be either 'n' or 'c', not {direction}")
    additional_entities = []
    if additional_entity_ids:
        for name in additional_entity_ids:
            entity = pose.entity(name)
            if entity is None:
                raise ValueError(
                    f"The entity_id '{name}' wasn't found in the {repr(pose)}. "
                    f"Available entity_ids={', '.join(entity.name for entity in pose.entities)}")
            else:
                additional_entities.append(entity)

    # Get the model coords before
    pose_coords = pose.coords
    entity_coords_to_bend = pose.coords[set_coords_slice]
    bent_coords_samples = []
    for trial, transformation in enumerate(generate_bend_transformations(joint_residue, direction=direction), 1):

        bent_coords = np.matmul(entity_coords_to_bend, transformation['rotation'].T) \
                      + transformation['translation']
        copied_pose_coords = pose_coords.copy()
        copied_pose_coords[set_coords_slice] = bent_coords

        for entity in additional_entities:
            copied_pose_coords[entity.atom_indices] = np.matmul(entity.coords, transformation['rotation'].T) \
                                                      + transformation['translation']

        bent_coords_samples.append(copied_pose_coords)
        if trial == samples:
            break

    return bent_coords_samples


def prepare_alignment_motif(model: Structure, model_start: int, motif_length: int,
                            termini: termini_literal, extension_length: int = 0, alignment_length: int = 5) -> tuple[Structure, Chain]:
    """From a Structure, select helices of interest from a termini of the model and separate the model into the
    original model and the selected helix

    Args:
        model: The model to acquire helices from
        model_start: The residue index to start motif selection at
        motif_length: The length of the helical motif
        termini: The termini to utilize
        extension_length: How many residues should the helix be extended
        alignment_length: The number of residues used to calculation overlap of the target to the ideal helix
    Returns:
        The original model without the selected helix and the selected helix
    """
    model_residue_indices = list(range(model_start, model_start + motif_length))
    helix_residues = model.get_residues(indices=model_residue_indices)
    helix_model = Chain.from_residues(helix_residues)

    if extension_length:
        # if termini is None:
        #     n_terminal_number = helix_model.n_terminal_residue.number
        #     c_terminal_number = helix_model.c_terminal_residue.number
        #     if n_terminal_number in model_residue_range or n_terminal_number < model_start:
        #         termini = 'n'
        #     elif c_terminal_number in model_residue_range or c_terminal_number > model_start + alignment_length:
        #         termini = 'c'
        #     else:  # Can't extend...
        #         raise ValueError(f"Couldn't automatically determine the desired termini to extend")
        helix_model.add_ideal_helix(termini=termini, length=extension_length, alignment_length=alignment_length)

    if termini == 'n':
        remove_indices = list(range(model.n_terminal_residue.index, helix_residues[-1].index + 1))
    else:
        remove_indices = list(range(helix_residues[0].index, model.c_terminal_residue.index + 1))

    deleted_model = model.copy()
    deleted_model.delete_residues(indices=remove_indices)

    return deleted_model, helix_model


# def align_model_to_helix(model: Structure, model_start: int, helix_model: Structure, helix_start: int,
#                          alignment_length: int) -> tuple[float, np.ndarray, np.ndarray]:  # Structure:  # , termini: termini_literal
#     """Take two Structure Models and align the second one to the first along a helical termini
#
#     Args:
#         model: The second model. This model will be moved during the procedure
#         model_start: The first residue to use for alignment from model
#         helix_model: The first model. This model will be fixed during the procedure
#         helix_start: The first residue to use for alignment from helix_model
#         alignment_length: The length of the helical alignment
#     Returns:
#         The rmsd, rotation, and translation to align the model to the helix_model
#         The aligned Model with the overlapping residues removed
#     """
#     # termini: The termini to utilize
#     # residues1 = helix_model.get_residues(indices=list(range(helix_start, helix_start + alignment_length)))
#     # residues2 = model.get_residues(indices=list(range(model_start, model_start + alignment_length)))
#     # if len(residues1) != len(residues2):
#     #     raise ValueError(
#     #         f"The aligned lengths aren't equal. helix_model length, {len(residues1)} != {len(residues2)},"
#     #         ' the aligned model length')
#     # residue_numbers1 = [residue.number for residue in residues1]
#     # coords1 = helix_model.get_coords_subset(residue_numbers=residue_numbers1, dtype='backbone')
#     coords1 = helix_model.get_coords_subset(
#         indices=list(range(helix_start, helix_start + alignment_length)), dtype='backbone')
#     # residue_numbers2 = [residue.number for residue in residues2]
#     coords2 = model.get_coords_subset(
#         indices=list(range(model_start, model_start + alignment_length)), dtype='backbone')
#     # coords2 = model.get_coords_subset(residue_numbers=residue_numbers2, dtype='backbone')
#
#     return superposition3d(coords1, coords2)


# def solve_termini_start_index(secondary_structure: str, termini: termini_literal) -> int:  # tuple[int, int]:
def get_terminal_helix_start_index_and_length(secondary_structure: str, termini: termini_literal,
                                              start_index: int = None, end_index: int = None) -> tuple[int, int]:
    """

    Args:
        secondary_structure: The secondary structure sequence of characters to search
        termini: The termini to search
        start_index: The termini to search
        start_index: The first index in the terminal helix
        end_index: The last index in the terminal helix
    Returns:
        A tuple of the index from the secondary_structure argument where SS_HELIX_IDENTIFIERS begin at the provided
            termini and the length that they persist for
    """
    # The start index of the specified termini
    # Todo debug
    if termini == 'n':
        if not start_index:
            start_index = secondary_structure.find(SS_HELIX_IDENTIFIERS)

        if not end_index:
            # Search for the last_index in a contiguous block of helical residues at the n-termini
            for idx, sstruct in enumerate(secondary_structure[start_index:], start_index):
                if sstruct != 'H':
                    # end_index = idx
                    break
            # else:  # The whole structure is helical
            end_index = idx  # - 1
    else:  # termini == 'c':
        if not end_index:
            end_index = secondary_structure.rfind(SS_HELIX_IDENTIFIERS) + 1

        if not start_index:
            # Search for the last_index in a contiguous block of helical residues at the c-termini
            # input(secondary_structure[end_index::-1])
            # for idx, sstruct in enumerate(secondary_structure[end_index::-1]):
            # Add 1 to the end index to include end_index secondary_structure in the reverse search
            # for idx, sstruct in enumerate(reversed(secondary_structure[:end_index + 1])):
            for idx, sstruct in enumerate(reversed(secondary_structure[:end_index])):
                if sstruct != 'H':
                    # idx -= 1
                    break
            # else:  # The whole structure is helical
            start_index = end_index - idx

    alignment_length = end_index - start_index

    return start_index, alignment_length


# @profile  # Need to perform not on cassini
def align_helices(models: Iterable[Structure]) -> list[PoseJob] | list:
    """

    Args:
        models: The Structure instances to be used in docking
    Returns:
        The PoseJob instances created as a result of fusion
    """
    align_time_start = time.time()
    # Retrieve symjob.JobResources for all flags
    job = symjob.job_resources_factory.get()

    sym_entry: SymEntry = job.sym_entry
    """The SymmetryEntry object describing the material"""

    # Initialize incoming Structures
    if len(models) != 2:
        raise ValueError(
            f"Can't perform {align_helices.__name__} with {len(models)} models. Only 2 are allowed")
    # models = []
    # """The Structure instances to be used in docking"""
    # Ensure models are oligomeric with make_oligomer()
    for idx, input_model in enumerate(models):
        for entity, symmetry in zip(input_model.entities, sym_entry.groups):
            if entity.is_symmetric():
                pass
            else:
                # Only respect the symmetry of the first input_model
                if idx == 0:
                    entity.make_oligomer(symmetry=symmetry)
                else:
                    logger.info(f'{align_helices.__name__}: Skipping symmetry for model {idx + 1} as no symmetry is '
                                f'allowed for the second model')
        # # Make, then save a new model based on the symmetric version of each Entity in the Model
        # # model = input_model.assembly  # This is essentially what is happening here
        # model = Model.from_chains([chain for entity in input_model.entities for chain in entity.chains],
        #                           entities=False, name=input_model.name)
        # model.fragment_db = job.fragment_db
        # models.append(model)

    model1: Pose
    model2: Model
    model1, model2 = models

    if job.alignment_length:
        alignment_length = job.alignment_length
    else:
        alignment_length = default_alignment_length = 5

    project = f'Alignment_{model1.name}-{model2.name}'
    project_dir = os.path.join(job.projects, project)
    putils.make_path(project_dir)
    protocol_name = 'helix_align'
    # fusion_chain_id = 'A'
    pose_jobs = []
    opposite_termini = {'n': 'c', 'c': 'n'}
    # Limit the calculation to a particular piece of the model
    if job.target_chain is None:  # Use the whole model
        selected_models1 = model1.entities
        remaining_entities1 = [entity for entity in model1.entities]
    else:
        selected_chain1 = model1.chain(job.target_chain)
        if not selected_chain1:
            raise ValueError(
                f"The provided {flags.format_args(flags.target_chain_args)} '{job.target_chain}' wasn't found in the "
                f"target model. Available chains = {', '.join(model1.chain_ids)}")
        # Todo make selection based off Entity
        entity1 = model1.match_entity_by_seq(selected_chain1.sequence)
        # Place a None token where the selected entity should be so that the SymEntry is accurate upon fusion
        remaining_entities1 = [entity if entity != entity1 else None for entity in model1.entities]
        selected_models1 = [entity1]

    if job.output_trajectory:
        # Create a Models instance to collect each model
        raise NotImplementedError('Make iterative saving more reliable. See output_pose()')
        trajectory_models = Models()

    model2_entities_after_fusion = model2.number_of_entities - 1
    # Create the corresponding SymEntry from the original SymEntry and the fusion
    if sym_entry:
        model1.set_symmetry(sym_entry=sym_entry)
        # Todo currently only C1 can be fused. Remove hard coding when changed
        symmetry = sym_entry.specification + '{C1}' * model2_entities_after_fusion
        sym_entry_chimera = parse_symmetry_to_sym_entry(symmetry=symmetry)
    else:
        sym_entry_chimera = None

    # Create the entity_transformations for model1
    model1_entity_transformations = []
    for transformation, set_mat_number in zip(model1.entity_transformations,
                                              sym_entry.setting_matrices_numbers):
        if transformation:
            if transformation['translation'] is None:
                internal_tx_x = internal_tx_y = internal_tx_z = None
            else:
                internal_tx_x, internal_tx_y, internal_tx_z = transformation['translation']
            if transformation['translation2'] is None:
                external_tx_x = external_tx_y = external_tx_z = None
            else:
                external_tx_x, external_tx_y, external_tx_z = transformation['translation2']

            entity_transform = dict(
                # rotation_x=rotation_degrees_x,
                # rotation_y=rotation_degrees_y,
                # rotation_z=rotation_degrees_z,
                internal_translation_z=internal_tx_z,
                setting_matrix=set_mat_number,
                external_translation_x=external_tx_x,
                external_translation_y=external_tx_y,
                external_translation_z=external_tx_z)
        else:
            entity_transform = {}
        model1_entity_transformations.append(entity_transform)
    model2_entity_transformations = [{} for _ in range(model2_entities_after_fusion)]

    def output_pose(name: str):
        """Handle output of the identified pose

        Args:
            name: The name to associate with this job
        Returns:
            None
        """
        # Output the pose as a PoseJob
        # name = f'{termini}-term{helix_start_index + 1}'
        pose_job = PoseJob.from_name(name, project=project, protocol=protocol_name)

        # if job.output:
        if job.output_fragments:
            pose.find_and_split_interface()
            # Query fragments
            pose.generate_interface_fragments()
        if job.output_trajectory:
            # Todo copy the update from fragdock
            pass
        # # Set the ASU, then write to a file
        # pose.set_contacting_asu()
        try:  # Remove existing cryst_record
            del pose._cryst_record
        except AttributeError:
            pass
        try:
            pose.uc_dimensions = model1.uc_dimensions
        except AttributeError:  # model1 isn't a crystalline symmetry
            pass

        # Todo the number of entities and the number of transformations could be different
        # entity_transforms = []
        for entity, transform in zip(pose.entities, model1_entity_transformations + model2_entity_transformations):
            transformation = sql.EntityTransform(**transform)
            # entity_transforms.append(transformation)
            pose_job.entity_data.append(sql.EntityData(
                meta=entity.metadata,
                metrics=entity.metrics,
                transform=transformation)
            )

        # session.add_all(entity_transforms)  # + entity_data)
        # # Need to generate the EntityData.id
        # session.flush()

        pose_job.pose = pose
        # pose_job.calculate_pose_design_metrics(session)
        putils.make_path(pose_job.pose_directory)
        pose_job.output_pose(path=pose_job.pose_path)
        pose_job.source_path = pose_job.pose_path
        pose_job.pose = None
        if job.output_to_directory:
            logger.info(f'Alignment output -> {pose_job.output_pose_path}')
        else:
            logger.info(f'Alignment output -> {pose_job.pose_path}')

        pose_jobs.append(pose_job)

    # Set parameters as null
    observed_protein_data = {}
    # Start the alignment search
    for selected_idx1, entity1 in enumerate(selected_models1):
        logger.info(f'Target component {entity1.name}')
        entity1 = entity1.copy()
        if job.trim_termini:
            # Remove any unstructured termini from the Entity to enable most successful fusion
            entity1.delete_termini_to_helices()
            # entity.delete_unstructured_termini()

        if all(remaining_entities1):
            additional_entities1 = remaining_entities1.copy()
            additional_entities1[selected_idx1] = None
        else:
            additional_entities1 = remaining_entities1

        # Solve for target/aligned residue features
        half_entity1_length = entity1.number_of_residues / 2

        # Check target_end first as the secondary_structure1 slice is dependent on lengths
        if job.target_end:
            target_end_residue = entity1.residue(job.target_end)
            if target_end_residue is None:
                raise DesignError(
                    f"Couldn't find the {flags.format_args(flags.target_end_args)} residue number {job.target_end} "
                    f"in the {entity1.__class__.__name__}, {entity1.name}")
            target_end_index_ = target_end_residue.index
            # See if the specified aligned_start/aligned_end lie in this termini orientation
            if target_end_index_ < half_entity1_length:
                desired_end_target_termini = 'n'
            else:  # Closer to c-termini
                desired_end_target_termini = 'c'
        else:
            desired_end_target_termini = target_end_index_ = None

        if job.target_start:
            target_start_residue = entity1.residue(job.target_start)
            if target_start_residue is None:
                raise DesignError(
                    f"Couldn't find the {flags.format_args(flags.target_start_args)} residue number "
                    f"{job.target_start} in the {entity1.__class__.__name__}, {entity1.name}")
            target_start_index_ = target_start_residue.index
            # See if the specified aligned_start/aligned_end lie in this termini orientation
            if target_start_index_ < half_entity1_length:
                desired_start_target_termini = 'n'
            else:  # Closer to c-termini
                desired_start_target_termini = 'c'
        else:
            desired_start_target_termini = target_start_index_ = None

        # Set up desired_aligned_termini
        if desired_start_target_termini:
            if desired_end_target_termini:
                if desired_start_target_termini != desired_end_target_termini:
                    raise DesignError(
                        f"Found different termini specified for addition by your flags "
                        f"{flags.format_args(flags.aligned_start_args)} ({desired_start_target_termini}-termini) "
                        f"and{flags.format_args(flags.aligned_end_args)} ({desired_end_target_termini}-termini)")

            termini = desired_start_target_termini
        elif desired_end_target_termini:
            termini = desired_end_target_termini
        else:
            termini = None

        if job.target_termini:
            desired_termini = job.target_termini.copy()
            if termini and termini not in desired_termini:
                if desired_start_target_termini:
                    flag = flags.target_start_args
                    arg = job.target_start
                else:
                    flag = flags.target_end_args
                    arg = job.target_end
                raise DesignError(
                    f"The {flags.format_args(flags.target_termini_args)} '{job.target_termini}' isn't compatible with "
                    f"your flag {flags.format_args(flag)} '{arg}' which would specify the {termini}-termini")
        elif termini:
            desired_termini = [termini]
        else:  # None specified, try all
            desired_termini = ['n', 'c']

        # Check for helical termini on the target building block and remove those that are not available
        for termini in reversed(desired_termini):
            if not entity1.is_termini_helical(termini, window=alignment_length):
                logger.error(f"The specified termini '{termini}' isn't helical")
                desired_termini.remove(termini)

        if not desired_termini:
            logger.info(f'Target component {entity1.name} has no termini remaining')
            continue

        if job.aligned_chain is None:  # Use the whole model
            selected_models2 = model2.entities
            remaining_entities2 = [entity for entity in model2.entities]
        else:
            selected_chain2 = model2.chain(job.aligned_chain)
            if not selected_chain2:
                raise ValueError(
                    f"The provided {flags.format_args(flags.aligned_chain_args)} '{job.aligned_chain}' wasn't found in "
                    f"the aligned model. Available chains = {', '.join(model2.chain_ids)}")
            # Todo make selection based off Entity
            entity2 = model2.match_entity_by_seq(selected_chain2.sequence)
            remaining_entities2 = [entity for entity in model2.entities if entity != entity2]
            selected_models2 = [entity2]

        for selected_idx2, entity2 in enumerate(selected_models2):
            logger.info(f'Aligned component {entity2.name}')
            entity2 = entity2.copy()
            if job.trim_termini:
                # Remove any unstructured termini from the Entity to enable most successful fusion
                entity2.delete_termini_to_helices()
                # entity.delete_unstructured_termini()

            # Set variables for the additional entities
            if len(remaining_entities2) == model2.number_of_entities:
                additional_entities2 = remaining_entities2.copy()
                additional_entities2.pop(selected_idx2)
            else:
                additional_entities2 = remaining_entities2
            additional_entity_ids2 = [entity.name for entity in additional_entities2]

            # Throw away chain ids that are in use by model1 to increment additional model2 entities to correct chain_id
            available_chain_ids = chain_id_generator()
            chain_id = next(available_chain_ids)
            while chain_id in model1.chain_ids:
                chain_id = next(available_chain_ids)

            for add_ent2 in additional_entities2:
                add_ent2.chain_id = chain_id
                chain_id = next(available_chain_ids)

            half_entity2_length = entity2.number_of_residues / 2
            if job.aligned_start:
                aligned_start_residue = entity2.residue(job.aligned_start)
                if aligned_start_residue is None:
                    raise DesignError(
                        f"Couldn't find the {flags.format_args(flags.aligned_start_args)} residue number "
                        f"{job.aligned_start} in the {entity2.__class__.__name__}, {entity2.name}")
                aligned_start_index_ = aligned_start_residue.index

                # See if the specified aligned_start/aligned_end lie in this termini orientation
                if aligned_start_index_ < half_entity2_length:
                    desired_start_aligned_termini = 'n'
                else:  # Closer to c-termini
                    desired_start_aligned_termini = 'c'
            else:
                desired_start_aligned_termini = aligned_start_index_ = None

            if job.aligned_end:
                aligned_end_residue = entity2.residue(job.aligned_end)
                if aligned_end_residue is None:
                    raise DesignError(
                        f"Couldn't find the {flags.format_args(flags.aligned_end_args)} residue number {job.aligned_end} in "
                        f"the {entity2.__class__.__name__}, {entity2.name}")

                aligned_end_index_ = aligned_end_residue.index
                # See if the specified aligned_start/aligned_end lie in this termini orientation
                if aligned_end_index_ < half_entity2_length:
                    desired_end_aligned_termini = 'n'
                else:  # Closer to c-termini
                    desired_end_aligned_termini = 'c'
            else:
                desired_end_aligned_termini = aligned_end_index_ = None

            # Set the desired_aligned_termini
            if desired_start_aligned_termini:
                if desired_end_aligned_termini:
                    if desired_start_aligned_termini != desired_end_aligned_termini:
                        raise DesignError(
                            f"Found different termini specified for addition by your flags "
                            f"{flags.format_args(flags.aligned_start_args)} ({desired_start_aligned_termini}-termini) and"
                            f"{flags.format_args(flags.aligned_end_args)} ({desired_end_aligned_termini}-termini)")

                desired_aligned_termini = desired_start_aligned_termini
            elif desired_end_aligned_termini:
                desired_aligned_termini = desired_end_aligned_termini
            else:
                desired_aligned_termini = None

            termini_to_align = []
            for termini in desired_termini:
                # Check if the desired termini in the aligned structure is available
                align_termini = opposite_termini[termini]
                if entity2.is_termini_helical(align_termini, window=alignment_length):
                    if desired_aligned_termini and align_termini == desired_aligned_termini:
                        # This is the correct termini
                        termini_to_align.append(termini)
                        break  # As only one termini can be specified and this was it
                    else:
                        termini_to_align.append(termini)
                else:
                    logger.info(f"{align_helices.__name__} isn't possible for target {termini} to aligned "
                                f'{align_termini} since {model2.name} is missing a helical {align_termini}-termini')

            if not termini_to_align:
                logger.info(f'Target component {entity2.name} has no termini remaining')

            for termini in termini_to_align:
                logger.info(f'Starting {entity1.name} {termini}-termini')
                # Get the target_start_index the length_of_target_helix
                logger.debug(f'Checking {termini}-termini for helices:\n\t{entity1.secondary_structure}')
                target_start_index, length_of_target_helix = \
                    get_terminal_helix_start_index_and_length(entity1.secondary_structure, termini,
                                                              start_index=target_start_index_,
                                                              end_index=target_end_index_)
                logger.debug(f'Found {termini}-termini start index {target_start_index} and '
                             f'length {length_of_target_helix}')
                if job.extension_length:
                    extension_length = job.extension_length
                    # Add the extension length to the residue window if an ideal helix was added
                    # length_of_helix_model = length_of_target_helix + extension_length
                else:
                    extension_length = 0

                truncated_entity1, helix_model = prepare_alignment_motif(
                    entity1, target_start_index, length_of_target_helix,
                    termini=termini, extension_length=extension_length, alignment_length=alignment_length)
                # Rename the models to enable fusion
                chain_id = truncated_entity1.chain_id
                helix_model.chain_id = chain_id
                entity2.chain_id = chain_id

                length_of_helix_model = helix_model.number_of_residues
                logger.debug(f'length_of_helix_model: {length_of_helix_model}')
                max_target_helix_length = length_of_helix_model - alignment_length
                logger.debug(f'Number of helical positions on target: {max_target_helix_length}')
                target_start_indices_sequence = range(max_target_helix_length)

                # Get the aligned_start_index and length_of_aligned_helix
                align_termini = opposite_termini[termini]
                logger.debug(f'Checking {align_termini}-termini for helices:\n\t{entity2.secondary_structure}')
                aligned_start_index, length_of_aligned_helix = \
                    get_terminal_helix_start_index_and_length(entity2.secondary_structure, align_termini,
                                                              start_index=aligned_start_index_,
                                                              end_index=aligned_end_index_)
                logger.debug(f'Found {align_termini}-termini start index {aligned_start_index} and '
                             f'length {length_of_aligned_helix}')

                # Scan along the aligned helix length
                logger.debug(f'length_of_aligned_helix: {length_of_aligned_helix}')
                logger.debug(f'alignment_length: {alignment_length}')
                aligned_length = length_of_aligned_helix + 1 - alignment_length
                if aligned_length < 1:
                    logger.info(
                        f"Aligned component {entity2.name} {align_termini}-termini isn't long enough for alignment")
                    continue

                sample_all_alignments = False  # Debugging True  # First draft
                if sample_all_alignments:
                    align_iteration_direction = iter
                    target_iteration_direction = iter
                    target_start_iterator = target_start_indices_sequence
                else:
                    # Todo
                    #  targeted method which searches only possible
                    # Need to work out the align and sample function to wrap each index pair inside to clean for loops
                    if termini == 'n':
                        align_iteration_direction = iter
                        target_iteration_direction = reversed
                        target_pause_index_during_aligned_loop = max_target_helix_length - 1
                    else:  # termini == 'c'
                        align_iteration_direction = reversed
                        target_iteration_direction = iter
                        target_pause_index_during_aligned_loop = 0
                    target_start_iterator = [target_pause_index_during_aligned_loop]

                aligned_count = count(1)
                aligned_range_end = aligned_start_index + aligned_length
                align_start_indices_sequence = range(aligned_start_index, aligned_range_end)
                for aligned_start_index in align_iteration_direction(align_start_indices_sequence):
                    aligned_idx = next(aligned_count)
                    # logger.debug(f'aligned_idx: {aligned_idx}')
                    logger.debug(f'aligned_start_index: {aligned_start_index}')
                    # logger.debug(f'number of residues: {entity2.number_of_residues}')

                    # # Todo use full model_helix mode
                    # truncated_entity2, helix_model2 = prepare_alignment_motif(
                    #     entity2, aligned_start_index, alignment_length, termini=align_termini)
                    # # Todo? , extend_helix=extension_length)
                    # # Rename the models to enable fusion
                    # truncated_entity2.chain_id = chain_id

                    aligned_end_index = aligned_start_index + alignment_length
                    # Calculate the entity2 indices to delete after alignment position is found
                    if align_termini == 'c':
                        delete_indices2 = list(range(aligned_end_index, entity2.c_terminal_residue.index + 1))
                    else:
                        delete_indices2 = list(range(entity2.n_terminal_residue.index, aligned_start_index))
                    # Get aligned coords
                    coords2 = entity2.get_coords_subset(
                        indices=list(range(aligned_start_index, aligned_end_index)),
                        dtype='backbone')

                    # Todo
                    #  For every iteration of the aligned_start_index, perform the alignment procedure
                    #  Need to perform short target loops during aligned loop or make alignment procedure a function...

                    if aligned_idx == aligned_length:
                        # The maximum number of aligned_start_index have been reached, iterate over the target now
                        target_start_iterator = target_iteration_direction(target_start_indices_sequence)
                    # else:
                    #     target_start_iterator = []

                    # # Scan along the target helix length
                    # # helix_start_index = 0
                    # max_target_helix_length = length_of_helix_model - alignment_length
                    # logger.debug(f'Number of helical positions on target: {max_target_helix_length}')
                    # target_start_indices_sequence = range(max_target_helix_length)
                    # for helix_start_index in target_iteration_direction(target_start_indices_sequence):
                    #
                    for helix_start_index in target_start_iterator:
                        logger.debug(f'helix_start_index: {helix_start_index}')
                        helix_end_index = helix_start_index + alignment_length

                        # if helix_end_index > maximum_helix_alignment_length:
                        #     break  # This isn't allowed
                        sampling_index = f'{aligned_idx}/{aligned_length}, ' \
                                         f'{helix_start_index + 1}/{max_target_helix_length}'
                        # Get target coords
                        coords1 = helix_model.get_coords_subset(
                            indices=list(range(helix_start_index, helix_end_index)),
                            dtype='backbone')
                        try:
                            # # Use helix_model2 start index = 0 as helix_model2 is always truncated
                            # # Use transformed_entity2 mode
                            # rmsd, rot, tx = align_model_to_helix(
                            #     entity2, aligned_start_index, helix_model, helix_start_index, alignment_length)
                            rmsd, rot, tx = superposition3d(coords1, coords2)
                            # # Use full model_helix mode
                            # rmsd, rot, tx = align_model_to_helix(
                            #     helix_model2, 0, helix_model, helix_start_index, alignment_length)
                        except ValueError as error:  # The lengths of the coords aren't equal, report and proceed
                            logger.warning(str(error))
                            continue
                        else:
                            logger.info(f'{entity1.name} {termini}-termini to {entity2.name} {align_termini}-termini '
                                        f'alignment {sampling_index} has RMSD of {rmsd:.4f}')

                        # Transform and copy to facilitate delete
                        transformed_entity2 = entity2.get_transformed_copy(rotation=rot, translation=tx)
                        # Delete overhanging residues
                        transformed_entity2.delete_residues(indices=delete_indices2)

                        # Order the models, slice the helix for overlapped segments
                        if termini == 'n':
                            ordered_entity1 = transformed_entity2
                            ordered_entity2 = truncated_entity1
                            helix_model_start_index = helix_end_index
                            helix_model_range = range(helix_model_start_index, length_of_helix_model)

                            # Get Residue instances that mark the boundary of the fusion
                            start_residue1 = transformed_entity2.n_terminal_residue
                            end_residue1 = transformed_entity2.c_terminal_residue  # residues[aligned_end_index - 1]
                            if helix_end_index < extension_length:  # Zero-indexed <= one-indexed
                                entity1_first_residue_index = target_start_index
                                extension_str = f'-extend-{extension_length - helix_end_index}'
                            else:  # First time this runs, it adds 0 to target_start_index
                                entity1_first_residue_index = target_start_index + helix_end_index - extension_length
                                extension_str = ''
                            start_residue2 = entity1.residues[entity1_first_residue_index]
                            end_residue2 = entity1.c_terminal_residue
                        else:  # termini == 'c'
                            ordered_entity1 = truncated_entity1
                            ordered_entity2 = transformed_entity2
                            helix_model_start_index = 0
                            helix_model_range = range(helix_model_start_index, helix_start_index)

                            # Get Residue instances that mark the boundary of the fusion
                            start_residue1 = entity1.n_terminal_residue
                            if helix_start_index <= length_of_target_helix:  # Zero-indexed < one-indexed
                                entity1_last_residue_index = target_start_index + helix_start_index - 1
                                extension_str = ''
                            else:
                                entity1_last_residue_index = target_start_index + length_of_target_helix - 1
                                extension_str = f'-extend-{helix_start_index - length_of_target_helix}'
                            end_residue1 = entity1.residues[entity1_last_residue_index]  # c_terminal_residue
                            start_residue2 = transformed_entity2.n_terminal_residue  # residues[aligned_start_index]
                            end_residue2 = transformed_entity2.c_terminal_residue

                        # Get the new fusion name
                        # alignment_numbers = f'{start_residue1.number}-{end_residue1.number}+{extension_str}/' \
                        #                     f'{start_residue2.number}-{end_residue2.number}'
                        fusion_name = f'{ordered_entity1.name}_{start_residue1.number}-' \
                                      f'{end_residue1.number}_fused{extension_str}-to' \
                                      f'_{ordered_entity2.name}_{start_residue2.number}-' \
                                      f'{end_residue2.number}'
                        # Reformat residues for new chain
                        helix_n_terminal_residue_number = ordered_entity1.c_terminal_residue.number + 1
                        helix_model.renumber_residues(index=helix_model_start_index, at=helix_n_terminal_residue_number)
                        helix_residues = helix_model.get_residues(indices=list(helix_model_range))
                        ordered_entity2.renumber_residues(at=helix_n_terminal_residue_number + len(helix_residues))

                        if job.bend:  # Get the joint_residue for later manipulation
                            joint_residue = transformed_entity2.residues[aligned_start_index + alignment_length//2]

                        # Create fused Entity and rename select attributes
                        fused_entity = Entity.from_residues(
                            ordered_entity1.residues + helix_residues + ordered_entity2.residues,
                            name=fusion_name, chain_ids=[chain_id],
                            reference_sequence=ordered_entity1.sequence
                            + ''.join(r.type1 for r in helix_residues) + ordered_entity2.sequence,
                            uniprot_ids=tuple(uniprot_id for entity in [ordered_entity1, ordered_entity2]
                                              for uniprot_id in entity.uniprot_ids)
                        )

                        # ordered_entity1.write(out_path='DEBUG_1.pdb')
                        # helix_model.write(out_path='DEBUG_H.pdb')
                        # ordered_entity2.write(out_path='DEBUG_2.pdb')
                        # fused_entity.write(out_path='DEBUG_FUSED.pdb')

                        # Correct the .metadata attribute for each entity in the full assembly
                        # This is crucial for sql usage
                        protein_metadata = observed_protein_data.get(fusion_name)
                        if not protein_metadata:
                            protein_metadata = sql.ProteinMetadata(
                                entity_id=fused_entity.name,  # model_source=None
                                reference_sequence=fused_entity.sequence,
                                thermophilicity=sum((entity1.thermophilicity, entity2.thermophilicity)),
                                # symmetry_group=sym_entry_chimera.groups[entity_idx],
                                n_terminal_helix=ordered_entity1.is_termini_helical(),
                                c_terminal_helix=ordered_entity2.is_termini_helical('c'),
                                uniprot_entities=tuple(uniprot_entity for entity in [ordered_entity1, ordered_entity2]
                                                       for uniprot_entity in entity.metadata.uniprot_entities))
                            observed_protein_data[fusion_name] = protein_metadata
                        fused_entity.metadata = protein_metadata

                        # Create the list of Entity instances for the new Pose
                        # logger.debug(f'Loading pose')
                        if additional_entities1:
                            all_entities = []
                            for entity_idx, entity in enumerate(additional_entities1):
                                if entity is None:
                                    fused_entity.metadata.symmetry_group = sym_entry_chimera.groups[entity_idx]
                                    all_entities.append(fused_entity)
                                else:
                                    all_entities.append(entity)
                        else:
                            fused_entity.metadata.symmetry_group = sym_entry_chimera.groups[0]
                            all_entities = [fused_entity]

                        if additional_entities2:
                            transformed_additional_entities2 = [
                                entity.get_transformed_copy(rotation=rot, translation=tx)
                                for entity in additional_entities2]
                            # transformed_additional_entities2[0].write(out_path='DEBUG_Additional2.pdb')
                            # input('DEBUG_Additional2.pdb')
                            all_entities += transformed_additional_entities2

                        pose = Pose.from_entities(all_entities, sym_entry=sym_entry_chimera)
                        # pose.entities[0].write(oligomer=True, out_path='DEBUG_oligomer.pdb')
                        # pose.write(out_path='DEBUG_POSE.pdb', increment_chains=True)
                        # pose.write(assembly=True, out_path='DEBUG_ASSEMBLY.pdb')  # , increment_chains=True)

                        name = fusion_name
                        # name = f'{termini}-term_{aligned_idx + 1}-{helix_start_index + 1}'
                        if job.bend:
                            central_aligned_residue = pose.chain(chain_id).residue(joint_residue.number)
                            if central_aligned_residue is None:
                                logger.warning(f"Couldn't locate the joint_residue with residue number "
                                               f"{joint_residue.number} from chain {chain_id}")
                                continue
                            # print(central_aligned_residue)
                            bent_coords = bend(pose, central_aligned_residue, termini, samples=job.bend,
                                               additional_entity_ids=[entity.name for entity in pose.entities
                                                                      if entity.name in additional_entity_ids2])
                            for bend_idx, coords in enumerate(bent_coords, 1):
                                pose.coords = coords
                                if pose.is_clash(warn=False, silence_exceptions=True):
                                    logger.info(f'Alignment {fusion_name}, bend {bend_idx} clashes')
                                    continue
                                if pose.is_symmetric() and not job.design.ignore_symmetric_clashes and \
                                        pose.symmetric_assembly_is_clash(warn=False):
                                    logger.info(f'Alignment {fusion_name}, bend {bend_idx} has '
                                                f'symmetric clashes')
                                    continue

                                output_pose(name + f'-bend{bend_idx}')
                        else:
                            if not job.design.ignore_pose_clashes and \
                                    pose.is_clash(warn=False, silence_exceptions=True):
                                logger.info(f'Alignment {fusion_name} clashes')
                                continue

                            if pose.is_symmetric() and not job.design.ignore_symmetric_clashes and \
                                    pose.symmetric_assembly_is_clash(warn=False):
                                logger.info(f'Alignment {fusion_name} has symmetric clashes')
                                continue

                            output_pose(name)

    def terminate(pose_jobs: list[PoseJob]) -> list[PoseJob]:
        # , poses_df_: pd.DataFrame, residues_df_: pd.DataFrame) -> list[PoseJob]:
        """Finalize any remaining work and return to the caller"""
        # Add PoseJobs to the database
        error_count = count(1)
        while True:
            pose_name_pose_jobs = {pose_job.name: pose_job for pose_job in pose_jobs}
            session.add_all(pose_jobs)
            try:  # Flush PoseJobs to the current session to generate ids
                session.flush()
            except IntegrityError:  # PoseJob.project/.name already inserted
                session.rollback()
                number_flush_attempts = next(error_count)
                pose_names = list(pose_name_pose_jobs.keys())
                logger.info(f'rollback() #{number_flush_attempts}')
                logger.info(f'From {len(pose_names)} pose_names:\n{sorted(pose_names)}')
                # Find the actual pose_jobs_to_commit and place in session
                fetch_jobs_stmt = select(PoseJob).where(PoseJob.project.is_(project)) \
                    .where(PoseJob.name.in_(pose_names))
                existing_pose_jobs = session.scalars(fetch_jobs_stmt).all()
                # Note: Values are sorted by alphanumerical, not numerical
                # ex, design 11 is processed before design 2
                existing_pose_names = {pose_job_.name for pose_job_ in existing_pose_jobs}
                new_pose_names = set(pose_names).difference(existing_pose_names)
                logger.info(f'Found {len(new_pose_names)} new_pose_names:\n{sorted(new_pose_names)}')
                if not new_pose_names:  # No new PoseJobs
                    return existing_pose_jobs
                else:
                    pose_jobs = [pose_name_pose_jobs[pose_name] for pose_name in new_pose_names]
                    if number_flush_attempts == 2:
                        # Try to attach existing protein_metadata.entity_id
                        # possibly_new_uniprot_to_prot_metadata = {}
                        possibly_new_uniprot_to_prot_metadata = defaultdict(list)
                        # pose_name_to_prot_metadata = defaultdict(list)
                        for pose_job in pose_jobs:
                            for entity_data in pose_job.entity_data:
                                possibly_new_uniprot_to_prot_metadata[
                                    entity_data.meta.uniprot_ids].append(entity_data.meta)

                        all_uniprot_id_to_prot_data = sql.initialize_metadata(
                            session, possibly_new_uniprot_to_prot_metadata)

                        # logger.debug([[data.meta.entity_id for data in pose_job.entity_data] for pose_job in pose_jobs])
                        # Get all uniprot_entities, and fix ProteinMetadata that is already loaded
                        for pose_name, pose_job in pose_name_pose_jobs.items():
                            for entity_data in pose_job.entity_data:
                                entity_id = entity_data.meta.entity_id
                                # Search the updated ProteinMetadata
                                for protein_metadata in all_uniprot_id_to_prot_data.values():
                                    for data in protein_metadata:
                                        if entity_id == data.entity_id:
                                            # Set with the valid ProteinMetadata
                                            entity_data.meta = data
                                            break
                                    else:  # No break occurred, continue with outer loop
                                        continue
                                    break  # outer loop too
                                else:
                                    insp = inspect(entity_data)
                                    logger.critical(
                                        f'Missing the {sql.ProteinMetadata.__name__} instance for {entity_data} with '
                                        f'entity_id {entity_id}')
                                    logger.info(f'\tThis instance is transient? {insp.transient}, pending?'
                                                f' {insp.pending}, persistent? {insp.persistent}')
                        logger.info(f'Found the newly added Session instances:\n{session.new}')
                    elif number_flush_attempts == 3:
                        attrs_of_interest = \
                            ['id', 'entity_id', 'reference_sequence', 'thermophilicity', 'symmetry_group', 'model_source']
                        properties = []
                        for pose_job in pose_jobs:
                            for entity_data in pose_job.entity_data:
                                properties.append('\t'.join([f'{attr}={getattr(entity_data.meta, attr)}'
                                                             for attr in attrs_of_interest]))
                        pose_job_properties = '\n\t'.join(properties)
                        logger.critical(f"The remaining PoseJob instances have the following "
                                        f"{sql.ProteinMetadata.__name__} properties:\n\t{pose_job_properties}")
                        # This is another error
                        raise
            else:
                break

        # # Format output data, fix missing
        # if job.db:
        #     pose_ids = [pose_job.id for pose_job in pose_jobs]
        # else:
        #     pose_ids = pose_names

        # # Extract transformation parameters for output
        # def populate_pose_metadata():
        #     """Add all required PoseJob information to output the created Pose instances for persistent storage"""
        #     # nonlocal poses_df_, residues_df_
        #     # Save all pose transformation information
        #     # From here out, the transforms used should be only those of interest for outputting/sequence design
        #
        #     # # Format pose transformations for output
        #     # # Get all rotations in terms of the degree of rotation along the z-axis
        #     # # Using the x, y rotation to enforce the degeneracy matrix...
        #     # rotation_degrees_x, rotation_degrees_y, rotation_degrees_z = \
        #     #     zip(*Rotation.from_matrix(rotation).as_rotvec(degrees=True).tolist())
        #     #
        #     # blank_parameter = [None, None, None]
        #     # if sym_entry.is_internal_tx1:
        #     #     # nonlocal full_int_tx1
        #     #     if len(full_int_tx1) > 1:
        #     #         full_int_tx1 = full_int_tx1.squeeze()
        #     #     z_height1 = full_int_tx1[:, -1]
        #     # else:
        #     #     z_height1 = blank_parameter
        #     #
        #     # set_mat1_number, set_mat2_number, *_extra = sym_entry.setting_matrices_numbers
        #     # # if sym_entry.unit_cell:
        #     # #     full_uc_dimensions = full_uc_dimensions[passing_symmetric_clash_indices_perturb]
        #     # #     full_ext_tx1 = full_ext_tx1[:]
        #     # #     full_ext_tx2 = full_ext_tx2[:]
        #     # #     full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
        #     # external_translation_x, external_translation_y, external_translation_z = \
        #     #     blank_parameter if ext_translation is None else ext_translation
        #
        #     # Update the sql.EntityData with transformations
        #     # for idx, pose_job in enumerate(pose_jobs):
        #     #     # Update sql.EntityData, sql.EntityMetrics, sql.EntityTransform
        #     #     # pose_id = pose_job.id
        #
        #
        #     # if job.db:
        #     #     # Update the poses_df_ and residues_df_ index to reflect the new pose_ids
        #     #     poses_df_.index = pd.Index(pose_ids, name=sql.PoseMetrics.pose_id.name)
        #     #     # Write dataframes to the sql database
        #     #     metrics.sql.write_dataframe(session, poses=poses_df_)
        #     #     output_residues = False
        #     #     if output_residues:  # Todo job.metrics.residues
        #     #         residues_df_.index = pd.Index(pose_ids, name=sql.PoseResidueMetrics.pose_id.name)
        #     #         metrics.sql.write_dataframe(session, pose_residues=residues_df_)
        #     # else:  # Write to disk
        #     #     residues_df_.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)  # ascending=False
        #     #     putils.make_path(job.all_scores)
        #     #     residue_metrics_csv = os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Residues.csv')
        #     #     residues_df_.to_csv(residue_metrics_csv)
        #     #     logger.info(f'Wrote residue metrics to {residue_metrics_csv}')
        #     #     trajectory_metrics_csv = \
        #     #         os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Trajectories.csv')
        #     #     job.dataframe = trajectory_metrics_csv
        #     #     poses_df_ = pd.concat([poses_df_], keys=[('dock', 'pose')], axis=1)
        #     #     poses_df_.columns = poses_df_.columns.swaplevel(0, 1)
        #     #     poses_df_.sort_index(level=2, axis=1, inplace=True, sort_remaining=False)
        #     #     poses_df_.sort_index(level=1, axis=1, inplace=True, sort_remaining=False)
        #     #     poses_df_.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
        #     #     poses_df_.to_csv(trajectory_metrics_csv)
        #     #     logger.info(f'Wrote trajectory metrics to {trajectory_metrics_csv}')
        #
        # # Populate the database with pose information. Has access to nonlocal session
        # populate_pose_metadata()

        # Write trajectory if specified
        if job.output_trajectory:
            if sym_entry.unit_cell:
                logger.warning('No unit cell dimensions applicable to the trajectory file.')

            trajectory_models.write(out_path=os.path.join(project_dir, 'trajectory_oligomeric_models.pdb'),
                                    oligomer=True)
        return pose_jobs

    logger.info(f'Total {project} trajectory took {time.time() - align_time_start:.2f}s')
    if not pose_jobs:
        logger.info(f'Found no viable outputs')
        return []

    with job.db.session(expire_on_commit=False) as session:
        pose_jobs = terminate(pose_jobs)  # , poses_df, residues_df)
        session.commit()
        metrics_stmt = select(PoseJob).where(PoseJob.id.in_([pose_job.id for pose_job in pose_jobs])) \
            .execution_options(populate_existing=True) \
            .options(selectinload(PoseJob.metrics))
        pose_jobs = session.scalars(metrics_stmt).all()
        # # Load all the committed metrics to the PoseJob instances
        # for pose_job in pose_jobs:
        #     pose_job.metrics

    return pose_jobs
