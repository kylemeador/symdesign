#!/home/kmeador/miniconda3/bin/python
import copy
from Atom import Atom
from Residue import Residue
from Bio.SeqUtils import IUPACData
import subprocess
from Stride import Stride
# from functions import extract_aa_seq
import numpy
import os
import math


class PDB:
    def __init__(self):
        self.all_atoms = []  # python list of Atoms
        self.res = None
        self.cryst_record = None
        self.cryst = None
        self.dbref = {}
        self.header = []
        self.sequence_dictionary = {}  # dictionary of SEQRES entries. key is chainID, value is ['3 letter AA Seq']. Ex: {'A': ['ALA GLN GLY PHE...']}
        self.filepath = None  # PDB filepath if instance is read from PDB file
        self.chain_id_list = []  # list of unique chain IDs in PDB
        self.name = None
        self.pdb_ss_asg = []
        self.cb_coords = []
        self.bb_coords = []

    def AddName(self, name):
        self.name = name

    def readfile(self, filepath, remove_alt_location=False, coordinates_only=True):
        # reads PDB file and feeds PDB instance
        self.filepath = filepath

        f = open(filepath, "r")
        pdb = f.readlines()
        f.close()

        available_chain_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
                               '2', '3', '4']
        chain_ids = []
        seq_list = []
        multimodel = False
        start_of_new_model = False
        model_chain_index = -1
        model_chain_id = None
        curr_chain_id = None
        for line in pdb:
            line = line.rstrip()
            if not coordinates_only:  # KM added 02/04/20 to deal with handling PDB headers
                if line[:22] == 'REMARK   2 RESOLUTION.':
                    try:
                        self.res = float(line[22:30].strip().split()[0])
                    except ValueError:
                        self.res = None
                        continue
                elif line[0:6] == 'SEQRES':
                    chain = line[11:12].strip()
                    sequence = line[20:71].strip().split()
                    if chain in self.sequence_dictionary:
                        self.sequence_dictionary[chain] += sequence
                    else:
                        self.sequence_dictionary[chain] = sequence
                    # seq_list.append([chain, sequence])
                    continue
                elif line[0:6] == "CRYST1" or line[0:5] == "SCALE":
                    self.header.append(line)
                    continue
                elif line[0:5] == 'DBREF':
                    line = line.strip()
                    chain = line[12:14].strip().upper()
                    if line[5:6] == '2':
                        db_accession_id = line[18:40].strip()
                    else:
                        db = line[26:33].strip()
                        if line[5:6] == '1':
                            continue
                        db_accession_id = line[33:42].strip()
                    self.dbref[chain] = {'db': db, 'accession': db_accession_id}
                    continue
            if line[0:4] == "ATOM" or line[17:20] == 'MSE' and line[0:6] == 'HETATM':  # KM modified 2/10/20 for MSE
                # coordinates_only = False
                number = int(line[6:11].strip())
                type = line[12:16].strip()
                alt_location = line[16:17].strip()
                if line[17:20] == "MSE":  # KM added 2/10/20
                    residue_type = "MET"  # KM added 2/10/20
                else:  # KM added 2/10/20
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
            elif line[0:5] == "MODEL":
                start_of_new_model = True
                multimodel = True
                model_chain_index += 1
                model_chain_id = available_chain_ids[model_chain_index]
            elif line[0:6] == 'CRYST1':
                self.cryst_record = line
                try:
                    a = float(line[6:15].strip())
                    b = float(line[15:24].strip())
                    c = float(line[24:33].strip())
                    ang_a = float(line[33:40].strip())
                    ang_b = float(line[40:47].strip())
                    ang_c = float(line[47:54].strip())
                except ValueError:
                    a, b, c = 0.0, 0.0, 0.0
                    ang_a, ang_b, ang_c = a, b, c
                space_group = line[55:66].strip()
                self.cryst = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
                continue
        self.chain_id_list = chain_ids
        self.clean_sequences()
        # self.retrieve_sequences(seq_list)

    # KM added 7/25/19 to deal with SEQRES info
    def clean_sequences(self):
        if self.sequence_dictionary:
            for chain in self.sequence_dictionary:
                # sequence = self.sequence_dictionary[chain].strip().split(' ')  # split each 3 AA into list
                # self.sequence_dictionary[chain] = []
                for i, residue in enumerate(self.sequence_dictionary[chain]):
                # for i, residue in enumerate(sequence):
                    try:
                        self.sequence_dictionary[chain][i] = IUPACData.protein_letters_3to1_extended[residue.title()]
                    except KeyError:
                        if residue.title() == 'Mse':
                            self.sequence_dictionary[chain][i] = 'M'
                        else:
                            self.sequence_dictionary[chain][i] = 'X'
                # self.sequence_dictionary[chain] = ''.join(sequence)
    # SEQRES   1 H  112  MSE PHE TYR GLU ILE ARG THR TYR ARG LEU LYS ASN GLY
    # def retrieve_sequences(self, seq_list):
    #     if seq_list != list():
    #         for line in seq_list:
    #             if line[0] not in self.sequence_dictionary:
    #                 self.sequence_dictionary[line[0]] = line[1]
    #             else:
    #                 self.sequence_dictionary[line[0]] += line[1]
    #
    #         for chain in self.sequence_dictionary:
    #             self.sequence_dictionary[chain] = self.sequence_dictionary[chain].strip().split(' ')  # split each 3 AA
    #             sequence = []
    #             for residue in self.sequence_dictionary[chain]:
    #                 try:
    #                     sequence.append(IUPACData.protein_letters_3to1_extended[residue.title()])
    #                 except KeyError:
    #                     if residue.title() == 'Mse':
    #                         sequence.append('M')
    #                     else:
    #                         sequence.append('X')
    #             self.sequence_dictionary[chain] = ''.join(sequence)
    #     else:
    #         self.sequence_dictionary = None
    #         # # File originated from outside the official PDB distribution. Probably a design
    #         # print('%s has no SEQRES, extracting sequence from ATOM record' % self.filepath)
    #         # for chain in self.chain_id_list:
    #         #     sequence, failures = extract(self.all_atoms, chain=chain)
    #         #     self.sequence_dictionary[chain] = sequence

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

    # def retrieve_chain_ids(self):  # KM added 2/3/20 to deal with updating chain names after rename_chain(s) functions
    #     # creates a list of unique chain IDs in PDB and feeds it into chain_id_list maintaining order
    #     chain_ids = []
    #     for atom in self.all_atoms:
    #         chain_ids.append(atom.chain)
    #     chain_ids = list(set(chain_ids))
    #     # chain_ids.sort(key=lambda x: (x[0].isdigit(), x))
    #     self.chain_id_list = chain_ids

    def get_chain_index(self, index):
        return self.chain_id_list[index]

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

    def get_CA_atoms(self):
        ca_atoms = []
        for atom in self.all_atoms:
            if atom.is_CA():
                ca_atoms.append(atom)
        return ca_atoms

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

    def replace_coords(self, new_cords):
        for i in range(len(self.all_atoms)):
            self.all_atoms[i].x, self.all_atoms[i].y, self.all_atoms[i].z = \
                new_cords[i][0], new_cords[i][1], new_cords[i][2]

    def mat_vec_mul3(self, a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            c[i] = 0.
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
            print('Axis does not exists!')

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
            print('Axis does not exists!')

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

        self.chain_id_list = lm

    def rename_chain(self, chain_of_interest, new_chain):  # KM Added 8/19 Caution, will rename to already taken chain
        chain_atoms = self.chain(chain_of_interest)
        for i in range(len(chain_atoms)):
            self.chain_atoms[i].chain = new_chain

        self.chain_id_list[self.chain_id_list.index(chain_of_interest)] = new_chain

    def reorder_chains(self, exclude_chains_list=None):  # KM Added 12/16/19
        # Renames chains starting from the first chain as A and the last as l_abc[len(self.chain_id_list) - 1]
        l_abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6']
        # , '7', '8', '9']

        if exclude_chains_list:
            l_available = []
            for chain in l_abc:
                if chain not in exclude_chains_list:
                    l_available.append(chain)
        else:
            l_available = l_abc

        l_moved = []
        for i in range(len(self.chain_id_list)):
            l_moved.append(l_available[i])

        prev = self.all_atoms[0].chain
        chain_index = 0
        l3 = []
        for i in range(len(self.all_atoms)):
            if prev != self.all_atoms[i].chain:
                chain_index += 1
            l3.append(l_moved[chain_index])
            prev = self.all_atoms[i].chain

        for i in range(len(self.all_atoms)):
            self.all_atoms[i].chain = l3[i]
        # Update chain_id_list
        self.chain_id_list = l_moved

    def renumber_residues(self):  # KM Added 12/16/19
        # Starts numbering PDB residues at 1 and numbers sequentially until reaches last atom in file
        last_atom_index = len(self.all_atoms)
        residues = len(self.get_all_residues())
        idx, offset = 0, 1
        for j in range(residues):
            current_res_num = self.all_atoms[idx].residue_number
            while self.all_atoms[idx].residue_number == current_res_num:
                self.all_atoms[idx].residue_number = j + offset
                idx += 1
                if idx == last_atom_index:
                    break

    def AddZAxis(self):
        z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
        z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")

        axis = [z_axis_a, z_axis_b, z_axis_c]
        self.all_atoms.extend(axis)

    def AddXYZAxes(self):
        z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
        z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")

        y_axis_a = Atom(1, "CA", " ", "GLY", "8", 1, " ", 0.000, 80.000, 0.000, 1.00, 20.00, "C", "")
        y_axis_b = Atom(2, "CA", " ", "GLY", "8", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        y_axis_c = Atom(3, "CA", " ", "GLY", "8", 3, " ", 0.000, -80.000, 0.000, 1.00, 20.00, "C", "")

        x_axis_a = Atom(1, "CA", " ", "GLY", "9", 1, " ", 80.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        x_axis_b = Atom(2, "CA", " ", "GLY", "9", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        x_axis_c = Atom(3, "CA", " ", "GLY", "9", 3, " ", -80.000, 0.000, 0.000, 1.00, 20.00, "C", "")

        axes = [z_axis_a, z_axis_b, z_axis_c, y_axis_a, y_axis_b, y_axis_c, x_axis_a, x_axis_b, x_axis_c]
        self.all_atoms.extend(axes)

    def getTermCAAtom(self, term, chain_id):
        if term == "N":
            for atom in self.chain(chain_id):
                if atom.type == "CA":
                    return atom
        elif term == "C":
            for atom in self.chain(chain_id)[::-1]:
                if atom.type == "CA":
                    return atom
        else:
            print('Select N or C Term')
            return None

    def getResidueAtoms(self, residue_chain_id, residue_number):
        residue_atoms = []
        for atom in self.all_atoms:
            if atom.chain == residue_chain_id and atom.residue_number == residue_number:
                residue_atoms.append(atom)
        return residue_atoms

    def get_residue(self, chain_id, residue_number):  # KM added 04/15/20 to query single residue objects
        return Residue(self.getResidueAtoms(chain_id, residue_number))

    def get_residues_chain(self, chain_id):
        current_residue_number = self.chain(chain_id)[0].residue_number
        current_residue = []
        all_residues = []
        for atom in self.chain(chain_id):
            if atom.residue_number == current_residue_number:
                current_residue.append(atom)
            else:
                all_residues.append(Residue(current_residue))
                current_residue = []
                current_residue.append(atom)
                current_residue_number = atom.residue_number
        all_residues.append(Residue(current_residue))
        return all_residues

    def get_all_residues(self):
        current_residue_number = self.all_atoms[0].residue_number
        current_residue = []
        all_residues = []
        for atom in self.all_atoms:
            if atom.residue_number == current_residue_number:
                current_residue.append(atom)
            else:
                all_residues.append(Residue(current_residue))
                current_residue = []
                current_residue.append(atom)
                current_residue_number = atom.residue_number
        all_residues.append(Residue(current_residue))
        return all_residues

    def write(self, out_path, cryst1=None):
        if not cryst1:
            cryst1 = self.cryst_record
        outfile = open(out_path, "w")
        if cryst1 and isinstance(cryst1, str) and cryst1.startswith("CRYST1"):
            outfile.write(str(cryst1) + "\n")
        for atom in self.all_atoms:
            outfile.write(str(atom))
        outfile.close()

    def calculate_ss(self, chain_id="A", stride_exe_path='./stride/stride'):
        pdb_stride = Stride(self.filepath, chain_id, stride_exe_path)
        pdb_stride.run()
        self.pdb_ss_asg = pdb_stride.ss_asg

    def getStructureSequence(self, chain_id):
        sequence_list = []
        for atom in self.chain(chain_id):
            if atom.is_CA():
                sequence_list.append(atom.residue_type)
        one_letter = ''.join([IUPACData.protein_letters_3to1[k.title()] for k in sequence_list if k.title() in IUPACData.protein_letters_3to1_extended])
        return one_letter

    def orient(self, symm, orient_dir, generate_oriented_pdb=False):
        os.system('cp %s input.pdb' % self.filepath)
        os.system('%s/orient_oligomer >> orient.out 2>&1 << eof\n%s/%s_symm.txt\neof' % (orient_dir, orient_dir, symm))
        os.system('mv output.pdb %s_orient.pdb' % os.path.splitext(self.filepath)[0])
        os.system('rm input.pdb')
        if os.path.exists('%s_orient.pdb' % os.path.splitext(self.filepath)[0]):
            if not generate_oriented_pdb:
                oriented_pdb = PDB()
                oriented_pdb.readfile('%s_orient.pdb' % os.path.splitext(self.filepath)[0])
                os.system('rm %s_orient.pdb' % os.path.splitext(self.filepath)[0])
                return oriented_pdb
            else:
                return 0
        else:
            return None

    def center_of_mass(self):
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
            print('ERROR CALCULATING CENTER OF MASS')
            return None

    def sample_rot_tx_dof_coords(self, rot_step_deg=1, rot_range_deg=0, tx_step=1, start_tx_range=0, end_tx_range=0, axis="z", rotational_setting_matrix=None, degeneracy=None):

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
                print('AXIS SELECTED FOR SAMPLING IS NOT SUPPORTED')
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
                print('INVALID SAMPLING AXIS')
                return None

        def generate_sampled_coordinates_np(pdb_coordinates, rotation_matrices, translation_matrices, degeneracy_matrices):
            pdb_coords_np = numpy.array(pdb_coordinates)
            rot_matrices_np = numpy.array(rotation_matrices)
            degeneracy_matrices_rot_mat_np = numpy.array(degeneracy_matrices)

            if rotation_matrices is not None and translation_matrices is not None:
                if rotation_matrices == [] and translation_matrices == []:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate((degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
                        return pdb_coords_degen_np
                    else:
                        return numpy.expand_dims(pdb_coords_np, axis=0)

                elif rotation_matrices == [] and translation_matrices != []:
                    if degeneracy_matrices is not None:
                        degen_coords_np = numpy.matmul(pdb_coords_np, degeneracy_matrices_rot_mat_np)
                        pdb_coords_degen_np = numpy.concatenate((degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
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
                        pdb_coords_degen_np = numpy.concatenate((degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
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
                        pdb_coords_degen_np = numpy.concatenate((degen_coords_np, numpy.expand_dims(pdb_coords_np, axis=0)))
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

    def get_bb_indices(self):
        bb_indices = []
        for i in range(len(self.all_atoms)):
            if self.all_atoms[i].is_backbone():
                bb_indices.append(i)
        return bb_indices

    def get_cb_indices(self, InclGlyCA=False):
        cb_indices = []
        for i in range(len(self.all_atoms)):
            if self.all_atoms[i].is_CB(InclGlyCA=InclGlyCA):
                cb_indices.append(i)
        return cb_indices

    def get_term_ca_indices(self, term):
        if term == "N":
            ca_term_list = []
            chain_id = None
            for i in range(len(self.all_atoms)):
                atom = self.all_atoms[i]
                if atom.chain != chain_id and atom.type == "CA":
                    ca_term_list.append(i)
                    chain_id = atom.chain
            return ca_term_list

        elif term == "C":
            ca_term_list = []
            chain_id = self.all_atoms[0].chain
            current_ca_indx = None
            for i in range(len(self.all_atoms)):
                atom = self.all_atoms[i]
                if atom.chain != chain_id:
                    ca_term_list.append(current_ca_indx)
                    chain_id = atom.chain
                if atom.type == "CA":
                    current_ca_indx = i
            ca_term_list.append(current_ca_indx)
            return ca_term_list

        else:
            print('Select N or C Term')
            return []

    def get_helix_cb_indices(self, stride_exe_path):
        # only works for monomers or homo-complexes
        h_cb_indices = []
        stride = Stride(self.filepath, self.chain_id_list[0], stride_exe_path)
        stride.run()
        stride_ss_asg = stride.ss_asg
        for i in range(len(self.all_atoms)):
            atom = self.all_atoms[i]
            if atom.is_CB():
                if (atom.residue_number, "H") in stride_ss_asg:
                    h_cb_indices.append(i)
        return h_cb_indices

    def get_surface_helix_cb_indices(self, stride_exe_path, free_sasa_exe_path, probe_radius=1.4, sasa_thresh=1):
        # only works for monomers or homo-complexes
        proc = subprocess.Popen('%s --format=seq --probe-radius %s %s' %(free_sasa_exe_path, str(probe_radius), self.filepath), stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out_lines = out.split("\n")
        sasa_out = []
        for line in out_lines:
            if line != "\n" and line != "" and not line.startswith("#"):
                res_num = int(line[5:10])
                sasa = float(line[16:])
                if sasa >= sasa_thresh:
                    if res_num not in sasa_out:
                        sasa_out.append(res_num)

        h_cb_indices = []
        stride = Stride(self.filepath, self.chain_id_list[0], stride_exe_path)
        stride.run()
        stride_ss_asg = stride.ss_asg
        for i in range(len(self.all_atoms)):
            atom = self.all_atoms[i]
            if atom.is_CB():
                if (atom.residue_number, "H") in stride_ss_asg and atom.residue_number in sasa_out:
                    h_cb_indices.append(i)
        return h_cb_indices

    def get_surface_atoms(self, free_sasa_exe_path, chain_selection="all", probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        proc = subprocess.Popen('%s --format=seq --probe-radius %s %s' %(free_sasa_exe_path, str(probe_radius), self.filepath), stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out_lines = out.split("\n")
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

    def get_surface_resdiue_info(self, free_sasa_exe_path, probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        proc = subprocess.Popen('%s --format=seq --probe-radius %s %s' %
                                (free_sasa_exe_path, str(probe_radius), self.filepath),
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out_lines = out.split("\n")
        sasa_out = []
        for line in out_lines:
            if line != "\n" and line != "" and not line.startswith("#"):
                chain_id = line[4:5]
                res_num = int(line[5:10])
                sasa = float(line[16:])
                if sasa > sasa_thresh:
                    sasa_out.append((chain_id, res_num))

        return sasa_out

    def mutate_to(self, chain, residue, res_id='ALA'):  # KM added 12/31/19 to mutate pdb Residue objects to alanine
        # if using residue number, then residue_atom_list[i] is necessary
        # else using Residue object, residue.atom_list[i] is necessary
        residue_atom_list = self.getResidueAtoms(chain, residue)  # residue.atom_list
        delete = []
        for i in range(len(residue_atom_list)):
            if residue_atom_list[i].is_backbone() or residue_atom_list[i].is_CB():
                residue_atom_list[i].residue_type = res_id
            else:
                delete.append(i)

        if delete:
            delete = sorted(delete, reverse=True)
            for j in delete:
                i = residue_atom_list[j]
                self.all_atoms.remove(i)

    def apply(self, rot, tx):  # KM added 02/10/20 to run extract_pdb_interfaces.py
        moved = []
        for coord in self.extract_all_coords():
            coord_moved = self.mat_vec_mul3(rot, coord)
            for j in range(3):
                coord_moved[j] += tx[j]
            moved.append(coord_moved)
        self.replace_coords(moved)

    def get_ave_residue_b_factor(self, chain, residue):
        residue_atoms = self.getResidueAtoms(chain, residue)
        temp = 0
        for atom in residue_atoms:
            temp += atom.temp_fact

        return round(temp / len(residue_atoms), 2)
