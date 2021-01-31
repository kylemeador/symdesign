import copy
import math
import os
import subprocess
from shutil import copyfile, move

import numpy

from PathUtils import free_sasa_exe_path, orient_dir, orient_exe_path, orient_log_file
from classes.Atom import Atom
from classes.Stride import Stride
from utils.SymmUtils import valid_subunit_number


class PDB:
    def __init__(self):
        self.all_atoms = []  # python list of Atoms
        self.filepath = None  # PDB filepath if instance is read from PDB file
        self.chain_id_list = []  # list of unique chain IDs in PDB
        self.name = None
        self.pdb_ss_asg = []
        self.cb_coords = []
        self.bb_coords = []

    def AddName(self, name):
        self.name = name

    def set_all_atoms(self, atom_list):
        self.all_atoms = atom_list

    def set_chain_id_list(self, chain_id_list):
        self.chain_id_list = chain_id_list

    def set_filepath(self, filepath):
        self.filepath = filepath

    def get_atoms(self):
        return self.all_atoms

    def get_chain_id_list(self):
        return self.chain_id_list

    def get_filepath(self):
        return self.filepath

    def get_secondary_structure(self, chain_id="A"):  # , stride_exe_path=stride_exe_path):
        pdb_stride = Stride(self.filepath, chain_id)
        pdb_stride.run()
        self.pdb_ss_asg = pdb_stride.ss_asg

        return self.pdb_ss_asg

    def readfile(self, filepath, remove_alt_location=False):
        # reads PDB file and feeds PDB instance
        self.filepath = filepath

        f = open(filepath, "r")
        pdb = f.readlines()
        f.close()

        available_chain_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T',
                               'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                               'm', 'n',
                               'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4']

        chain_ids = []
        multimodel = False
        start_of_new_model = False
        model_chain_index = -1
        model_chain_id = None
        curr_chain_id = None
        for line in pdb:
            line = line.rstrip()
            if line[0:5] == "MODEL":
                start_of_new_model = True
                multimodel = True
                model_chain_index += 1
                model_chain_id = available_chain_ids[model_chain_index]
            elif line[0:4] == "ATOM":
                number = int(line[6:11].strip())
                type = line[12:16].strip()
                alt_location = line[16:17].strip()
                residue_type = line[17:20].strip()
                if multimodel:
                    if line[21:22].strip() != curr_chain_id:
                        curr_chain_id = line[21:22].strip()
                        if not start_of_new_model:
                            model_chain_index += 1
                            model_chain_id = available_chain_ids[model_chain_index]
                    start_of_new_model = False
                    chain = model_chain_id
                else:
                    chain = line[21:22].strip()
                residue_number = int(line[22:26].strip())
                code_for_insertion = line[26:27].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occ = float(line[54:60].strip())
                temp_fact = float(line[60:66].strip())
                element_symbol = line[76:78].strip()
                atom_charge = line[78:80].strip()
                atom = Atom(number, type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y,
                            z, occ, temp_fact, element_symbol, atom_charge)
                if remove_alt_location:
                    if alt_location == "" or alt_location == "A":
                        if atom.chain not in chain_ids:
                            chain_ids.append(atom.chain)
                        self.all_atoms.append(atom)
                else:
                    if atom.chain not in chain_ids:
                        chain_ids.append(atom.chain)
                    self.all_atoms.append(atom)
        self.chain_id_list = chain_ids

    def read_atom_list(self, atom_list, store_cb_and_bb_coords=False):
        # reads a python list of Atoms and feeds PDB instance
        if store_cb_and_bb_coords:
            chain_ids = []
            for atom in atom_list:
                self.all_atoms.append(atom)
                if atom.is_backbone():
                    [x, y, z] = [atom.x, atom.y, atom.z]
                    self.bb_coords.append([x, y, z])
                if atom.is_CB(InclGlyCA=False):
                    [x, y, z] = [atom.x, atom.y, atom.z]
                    self.cb_coords.append([x, y, z])
                if atom.chain not in chain_ids:
                    chain_ids.append(atom.chain)
            self.chain_id_list = chain_ids
        else:
            chain_ids = []
            for atom in atom_list:
                self.all_atoms.append(atom)
                if atom.chain not in chain_ids:
                    chain_ids.append(atom.chain)
            self.chain_id_list = chain_ids

    def chain(self, chain_id):
        # returns a python list of Atoms containing the subset of Atoms in the PDB instance that belong to the selected chain ID
        selected_atoms = []
        for atom in self.all_atoms:
            if atom.chain == chain_id:
                selected_atoms.append(atom)
        return selected_atoms

    def chains(self, chain_id_list):
        # returns a python list of Atoms containing the subset of Atoms in the PDB instance that belong to the selected chain IDs
        selected_atoms = []
        for atom in self.all_atoms:
            if atom.chain in chain_id_list:
                selected_atoms.append(atom)
        return selected_atoms

    # def extract_coords(self):  # TODO
    def extract_all_coords(self):
        coords = []
        for atom in self.all_atoms:
            [x, y, z] = [atom.x, atom.y, atom.z]
            coords.append([x, y, z])
        return coords

    def extract_backbone_coords(self):
        coords = []
        for atom in self.all_atoms:
            if atom.is_backbone():
                [x, y, z] = [atom.x, atom.y, atom.z]
                coords.append([x, y, z])
        return coords

    def extract_CA_coords(self):
        coords = []
        for atom in self.all_atoms:
            if atom.is_CA():
                [x, y, z] = [atom.x, atom.y, atom.z]
                coords.append([x, y, z])
        return coords

    # def get_ca_atoms(self):  # TODO
    def get_CA_atoms(self):
        ca_atoms = []
        for atom in self.all_atoms:
            if atom.is_CA():
                ca_atoms.append(atom)
        return ca_atoms

    def get_backbone_atoms(self):
        bb_atoms = []
        for atom in self.all_atoms:
            if atom.is_backbone():
                bb_atoms.append(atom)
        return bb_atoms

    def extract_CB_coords(self, InclGlyCA=False):
        coords = []
        for atom in self.all_atoms:
            if atom.is_CB(InclGlyCA=InclGlyCA):
                [x, y, z] = [atom.x, atom.y, atom.z]
                coords.append([x, y, z])
        return coords

    def extract_CB_coords_chain(self, chain, InclGlyCA=False):
        coords = []
        for atom in self.all_atoms:
            if atom.is_CB(InclGlyCA=InclGlyCA) and atom.chain == chain:
                [x, y, z] = [atom.x, atom.y, atom.z]
                coords.append([x, y, z])
        return coords

    def get_coords(self):
        return [[atom.x, atom.y, atom.z] for atom in self.all_atoms]

    def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):

        coords = []
        cb_indices = []

        for i in range(len(self.all_atoms)):
            if self.all_atoms[i].is_CB(InclGlyCA=InclGlyCA):
                [x, y, z] = [self.all_atoms[i].x, self.all_atoms[i].y, self.all_atoms[i].z]
                coords.append([x, y, z])

                if ReturnWithCBIndices:
                    cb_indices.append(i)

        if ReturnWithCBIndices:
            return coords, cb_indices

        else:
            return coords

    def replace_coords(self, new_cords):
        for i in range(len(self.all_atoms)):
            self.all_atoms[i].x, self.all_atoms[i].y, self.all_atoms[i].z = new_cords[i][0], new_cords[i][1], \
                                                                            new_cords[i][2]

    def mat_vec_mul3(self, a, b):  # rot_mat, tx_v
        c = [0. for i in range(3)]
        for i in range(3):
            # c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]
        return c

    def rotate_translate(self, rot, tx):
        for atom in self.all_atoms:
            coord = [atom.x, atom.y, atom.z]
            coord_rot = self.mat_vec_mul3(rot, coord)
            newX = coord_rot[0] + tx[0]
            newY = coord_rot[1] + tx[1]
            newZ = coord_rot[2] + tx[2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def translate(self, tx):
        for atom in self.all_atoms:
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] + tx[0]
            newY = coord[1] + tx[1]
            newZ = coord[2] + tx[2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def rotate(self, rot, store_cb_and_bb_coords=False):
        if store_cb_and_bb_coords:
            for atom in self.all_atoms:
                x, y, z = self.mat_vec_mul3(rot, [atom.x, atom.y, atom.z])
                atom.x, atom.y, atom.z = x, y, z
                if atom.is_backbone():
                    self.bb_coords.append([atom.x, atom.y, atom.z])
                if atom.is_CB(InclGlyCA=False):
                    self.cb_coords.append([atom.x, atom.y, atom.z])
        else:
            for atom in self.all_atoms:
                x, y, z = self.mat_vec_mul3(rot, [atom.x, atom.y, atom.z])
                atom.x, atom.y, atom.z = x, y, z

    def rotate_along_principal_axis(self, degrees=90.0, axis='x'):
        """
        Rotate the coordinates about the given axis
        """
        deg = math.radians(float(degrees))

        # define the rotation matrices
        if axis == 'x':
            rotmatrix = [[1, 0, 0], [0, math.cos(deg), -1 * math.sin(deg)], [0, math.sin(deg), math.cos(deg)]]
        elif axis == 'y':
            rotmatrix = [[math.cos(deg), 0, math.sin(deg)], [0, 1, 0], [-1 * math.sin(deg), 0, math.cos(deg)]]
        elif axis == 'z':
            rotmatrix = [[math.cos(deg), -1 * math.sin(deg), 0], [math.sin(deg), math.cos(deg), 0], [0, 0, 1]]
        else:
            print("Axis does not exists!")

        for atom in self.all_atoms:
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] * rotmatrix[0][0] + coord[1] * rotmatrix[0][1] + coord[2] * rotmatrix[0][2]
            newY = coord[0] * rotmatrix[1][0] + coord[1] * rotmatrix[1][1] + coord[2] * rotmatrix[1][2]
            newZ = coord[0] * rotmatrix[2][0] + coord[1] * rotmatrix[2][1] + coord[2] * rotmatrix[2][2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def ReturnRotatedPDB(self, degrees=90.0, axis='x', store_cb_and_bb_coords=False):
        """
        Rotate the coordinates about the given axis
        """
        deg = math.radians(float(degrees))

        # define the rotation matrices
        if axis == 'x':
            rotmatrix = [[1, 0, 0], [0, math.cos(deg), -1 * math.sin(deg)], [0, math.sin(deg), math.cos(deg)]]
        elif axis == 'y':
            rotmatrix = [[math.cos(deg), 0, math.sin(deg)], [0, 1, 0], [-1 * math.sin(deg), 0, math.cos(deg)]]
        elif axis == 'z':
            rotmatrix = [[math.cos(deg), -1 * math.sin(deg), 0], [math.sin(deg), math.cos(deg), 0], [0, 0, 1]]
        else:
            print("Axis does not exists!")

        rotated_atoms = []
        for atom in self.all_atoms:
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] * rotmatrix[0][0] + coord[1] * rotmatrix[0][1] + coord[2] * rotmatrix[0][2]
            newY = coord[0] * rotmatrix[1][0] + coord[1] * rotmatrix[1][1] + coord[2] * rotmatrix[1][2]
            newZ = coord[0] * rotmatrix[2][0] + coord[1] * rotmatrix[2][1] + coord[2] * rotmatrix[2][2]
            rot_atom = copy.deepcopy(atom)
            rot_atom.x, rot_atom.y, rot_atom.z = newX, newY, newZ
            rotated_atoms.append(rot_atom)

        rotated_pdb = PDB()
        rotated_pdb.read_atom_list(rotated_atoms, store_cb_and_bb_coords)

        return rotated_pdb

    def ReturnTranslatedPDB(self, tx, store_cb_and_bb_coords=False):
        translated_atoms = []
        for atom in self.all_atoms:
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] + tx[0]
            newY = coord[1] + tx[1]
            newZ = coord[2] + tx[2]
            tx_atom = copy.deepcopy(atom)
            tx_atom.x, tx_atom.y, tx_atom.z = newX, newY, newZ
            translated_atoms.append(tx_atom)

        translated_pdb = PDB()
        translated_pdb.read_atom_list(translated_atoms, store_cb_and_bb_coords)

        return translated_pdb

    def ReturnRotatedPDBMat(self, rot):
        rotated_coords = []
        return_atoms = []
        return_pdb = PDB()
        # for coord in self.extract_coords():  # TODO
        for coord in self.extract_all_coords():
            coord_copy = copy.deepcopy(coord)
            coord_rotated = self.mat_vec_mul3(rot, coord_copy)
            rotated_coords.append(coord_rotated)
        for i in range(len(self.all_atoms)):
            atom_copy = copy.deepcopy(self.all_atoms[i])
            atom_copy.x = rotated_coords[i][0]
            atom_copy.y = rotated_coords[i][1]
            atom_copy.z = rotated_coords[i][2]
            return_atoms.append(atom_copy)
        return_pdb.read_atom_list(return_atoms)
        return return_pdb

    def rename_chains(self, chain_list_fixed):
        lf = chain_list_fixed
        lm = self.chain_id_list[:]

        l_abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9']

        l_av = []
        for e in l_abc:
            if e not in lm:
                if e not in lf:
                    l_av.append(e)

        j = 0
        for i in range(len(lm)):
            if lm[i] in lf:
                lm[i] = l_av[j]
                j += 1

        self.chain_id_list = lm

        prev = self.all_atoms[0].chain
        c = 0
        l3 = []
        for i in range(len(self.all_atoms)):
            if prev != self.all_atoms[i].chain:
                c += 1
            l3.append(lm[c])
            prev = self.all_atoms[i].chain

        for i in range(len(self.all_atoms)):
            self.all_atoms[i].chain = l3[i]

    def getResidueAtoms(self, residue_chain_id, residue_number):
        residue_atoms = []
        for atom in self.all_atoms:
            if atom.chain == residue_chain_id and atom.residue_number == residue_number:
                residue_atoms.append(atom)
        return residue_atoms

    def write(self, out_path, cryst1=None):
        outfile = open(out_path, "w")
        if cryst1 is not None and isinstance(cryst1, str) and cryst1.startswith("CRYST1"):
            outfile.write(str(cryst1))
        for atom in self.all_atoms:
            outfile.write(str(atom))
        outfile.close()

    # def calculate_ss(self, chain_id="A"):  # , stride_exe_path=stride_exe_path):
    #     pdb_stride = Stride(self.filepath, chain_id)
    #     pdb_stride.run()
    #     self.pdb_ss_asg = pdb_stride.ss_asg

    def orient(self, sym=None, out_dir=None):
        """Orient a symmetric PDB at the origin with it's symmetry axis cannonically set on axis defined by symmetry
        file"""
        orient_log = os.path.join(out_dir, orient_log_file)

        pdb_file_name = os.path.basename(self.filepath)
        error_string = 'orient_oligomer could not orient %s check %s for more information\n' % (pdb_file_name,
                                                                                                orient_log)
        with open(orient_log_file, 'a+') as log_f:
            number_of_subunits = len(self.chain_id_list)
            if number_of_subunits != valid_subunit_number[sym]:
                log_f.write("%s\n Oligomer could not be oriented: It has %d subunits while %d are expected for %s "
                            "symmetry\n\n" % (pdb_file_name, number_of_subunits, valid_subunit_number[sym], sym))
                raise ValueError(error_string)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            orient_input = os.path.join(orient_dir, 'input.pdb')
            orient_output = os.path.join(orient_dir, 'output.pdb')

            def clean_orient_input_output():
                if os.path.exists(orient_input):
                    os.remove(orient_input)
                if os.path.exists(orient_output):
                    os.remove(orient_output)

            clean_orient_input_output()
            copyfile(self.filepath, orient_input)

            p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=orient_dir)
            in_symm_file = os.path.join(orient_dir, 'symm_files', sym)
            stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
            stderr = stderr.decode()  # 'utf-8' implied
            stdout = pdb_file_name + stdout.decode()[28:]
            # stdout = pdb_file_name + stdout[28:]

            log_f.write(stdout)
            log_f.write('%s\n' % stderr)
            oriented_file = os.path.join(out_dir, pdb_file_name)
            if os.path.exists(orient_output) and os.stat(orient_output).st_size != 0:
                move(orient_output, oriented_file)

            clean_orient_input_output()

            if not os.path.exists(oriented_file):
                raise RuntimeError(error_string)

    # def orient(self, sym, orient_dir, generate_oriented_pdb=False):
    #     os.system('cp %s input.pdb' % self.filepath)
    #     os.system('%s/orient_oligomer >> orient.out 2>&1 << eof\n%s/%s_symm.txt\neof' % (orient_dir, orient_dir, sym))
    #     os.system('mv output.pdb %s_orient.pdb' % os.path.splitext(self.filepath)[0])
    #     os.system('rm input.pdb')
    #     if os.path.exists('%s_orient.pdb' % os.path.splitext(self.filepath)[0]):
    #         if not generate_oriented_pdb:
    #             oriented_pdb = PDB()
    #             oriented_pdb.readfile('%s_orient.pdb' % os.path.splitext(self.filepath)[0])
    #             os.system('rm %s_orient.pdb' % os.path.splitext(self.filepath)[0])
    #             return oriented_pdb
    #         else:
    #             return 0
    #     else:
    #         return None

    def center_of_mass(self):
        # coords = self.extract_coords()  # TODO
        coords = self.extract_all_coords()
        n = len(coords)
        if n != 0:
            cm = [0. for j in range(3)]
            for i in range(n):
                for j in range(3):
                    cm[j] = cm[j] + coords[i][j]
            for j in range(3):
                cm[j] = cm[j] / n
            return cm
        else:
            print("ERROR CALCULATING CENTER OF MASS")
            return None

    def sample_rot_tx_dof_coords(self, rot_step_deg=1, rot_range_deg=0, tx_step=1, start_tx_range=0, end_tx_range=0,
                                 axis="z", rotational_setting_matrix=None, degeneracy=None):

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
                print("AXIS SELECTED FOR SAMPLING IS NOT SUPPORTED")
                return None

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
                print("INVALID SAMPLING AXIS")
                return None

        def generate_sampled_coordinates_np(pdb_coordinates, rotation_matrices, translation_matrices,
                                            degeneracy_matrices):
            pdb_coords_np = numpy.array(pdb_coordinates)
            rot_matrices_np = numpy.array(rotation_matrices)
            degeneracy_matrices_rot_mat_np = numpy.array(degeneracy_matrices)

            if rotation_matrices is not None and translation_matrices is not None:
                if rotation_matrices == [] and translation_matrices == []:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate(
                            (degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
                        return pdb_coords_degen_np
                    else:
                        return numpy.expand_dims(pdb_coords_np, axis=0)

                elif rotation_matrices == [] and translation_matrices != []:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate(
                            (degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
                        tx_sampled_coords = []
                        for tx_mat in translation_matrices:
                            tx_coords_np = pdb_coords_degen_np + tx_mat
                            tx_sampled_coords.extend(tx_coords_np)
                        return numpy.array(tx_sampled_coords)
                    else:
                        tx_sampled_coords = []
                        for tx_mat in translation_matrices:
                            tx_coords_np = pdb_coords_np + tx_mat
                            tx_sampled_coords.append(tx_coords_np)
                        return numpy.array(tx_sampled_coords)

                elif rotation_matrices != [] and translation_matrices == []:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate(
                            (degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
                        degen_rot_pdb_coords = []
                        for degen_coord_set in pdb_coords_degen_np:
                            degen_rot_np = numpy.matmul(degen_coord_set, rot_matrices_np)
                            degen_rot_pdb_coords.extend(degen_rot_np)
                        return numpy.array(degen_rot_pdb_coords)
                    else:
                        rot_coords_np = numpy.matmul(pdb_coords_np, rot_matrices_np)
                        return rot_coords_np
                else:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate(
                            (degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
                        degen_rot_pdb_coords = []
                        for degen_coord_set in pdb_coords_degen_np:
                            degen_rot_np = numpy.matmul(degen_coord_set, rot_matrices_np)
                            degen_rot_pdb_coords.extend(degen_rot_np)
                        degen_rot_pdb_coords_np = numpy.array(degen_rot_pdb_coords)
                        tx_sampled_coords = []
                        for tx_mat in translation_matrices:
                            tx_coords_np = degen_rot_pdb_coords_np + tx_mat
                            tx_sampled_coords.extend(tx_coords_np)
                        return numpy.array(tx_sampled_coords)
                    else:
                        rot_coords_np = numpy.matmul(pdb_coords_np, rot_matrices_np)
                        tx_sampled_coords = []
                        for tx_mat in translation_matrices:
                            tx_coords_np = rot_coords_np + tx_mat
                            tx_sampled_coords.extend(tx_coords_np)
                        return numpy.array(tx_sampled_coords)
            else:
                return None

        # pdb_coords = self.extract_coords()  # TODO
        pdb_coords = self.extract_all_coords()
        rot_matrices = get_rot_matrices(rot_step_deg, axis, rot_range_deg)
        tx_matrices = get_tx_matrices(tx_step, axis, start_tx_range, end_tx_range)
        sampled_coords_np = generate_sampled_coordinates_np(pdb_coords, rot_matrices, tx_matrices, degeneracy)

        if sampled_coords_np is not None:
            if rotational_setting_matrix is not None:
                rotational_setting_matrix_np = numpy.array(rotational_setting_matrix)
                rotational_setting_matrix_np_t = numpy.transpose(rotational_setting_matrix_np)
                sampled_coords_orient_canon_np = numpy.matmul(sampled_coords_np, rotational_setting_matrix_np_t)
                return sampled_coords_orient_canon_np.tolist()
            else:
                return sampled_coords_np.tolist()
        else:
            return None

    def get_cb_indices(self, InclGlyCA=False):
        cb_indices = []
        for i in range(len(self.all_atoms)):
            if self.all_atoms[i].is_CB(InclGlyCA=InclGlyCA):
                cb_indices.append(i)
        return cb_indices

    def get_surface_atoms(self, free_sasa_exe_path=free_sasa_exe_path, chain_selection="all", probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        # proc = subprocess.Popen(
        #     '%s --format=seq --probe-radius %s %s' % (free_sasa_exe_path, str(probe_radius), self.filepath),
        #     stdout=subprocess.PIPE, shell=True)
        # (out, err) = proc.communicate()
        # out_lines = out.split("\n")
        proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath],
                                stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        out_lines = out.decode('utf-8').split("\n")
        sasa_out = []
        for line in out_lines:
            if line != "\n" and line != "" and not line.startswith("#"):
                chain_id = line[4:5]
                res_num = int(line[5:10])
                sasa = float(line[16:])
                if sasa > sasa_thresh:
                    sasa_out.append((chain_id, res_num))

        if chain_selection == "all":
            surface_atoms = []
            for atom in self.all_atoms:
                if (atom.chain, atom.residue_number) in sasa_out:
                    surface_atoms.append(atom)
            return surface_atoms

        else:
            surface_atoms = []
            for atom in self.chain(chain_selection):
                if (atom.chain, atom.residue_number) in sasa_out:
                    surface_atoms.append(atom)
            return surface_atoms

    def get_surface_residue_info(self, free_sasa_exe_path=free_sasa_exe_path, probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath],
                                stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        out_lines = out.decode('utf-8').split("\n")
        sasa_out = []
        for line in out_lines:
            if line != "\n" and line != "" and not line.startswith("#"):
                chain_id = line[4:5]
                res_num = int(line[5:10])
                sasa = float(line[16:])
                if sasa > sasa_thresh:
                    sasa_out.append((chain_id, res_num))

        return sasa_out
