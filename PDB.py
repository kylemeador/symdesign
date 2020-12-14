#!/home/kmeador/miniconda3/bin/python
import copy
import math
import os
import subprocess

# from functions import extract_aa_seq
import numpy
from Bio import pairwise2
from Bio.SeqUtils import IUPACData
from sklearn.neighbors import BallTree

from Atom import Atom
from Residue import Residue
from Stride import Stride


class PDB:
    def __init__(self):
        self.all_atoms = []  # python list of Atoms
        self.res = None
        self.cryst_record = None
        self.cryst = None
        self.dbref = {}
        self.header = []
        self.seqres_sequences = {}  # SEQRES entries. key is chainID, value is 'AGHKLAIDL'
        self.atom_sequences = {}  # ATOM sequences. key is chain, value is 'AGHKLAIDL'
        self.filepath = None  # PDB filepath if instance is read from PDB file
        self.chain_id_list = []  # list of unique chain IDs in PDB
        self.entities = {}  # {0: {'chains': [], 'seq': 'GHIPLF...'}
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

    def get_all_atoms(self):
        return self.all_atoms

    def get_chain_id_list(self):
        return self.chain_id_list

    def get_filepath(self):
        return self.filepath

    def update_attributes_from_pdb(self, pdb):
        # self.all_atoms = pdb.all_atoms
        self.res = pdb.res
        self.cryst_record = pdb.cryst_record
        self.cryst = pdb.cryst
        self.dbref = pdb.dbref
        self.header = pdb.header
        self.seqres_sequences = pdb.seqres_sequences
        # self.atom_sequences = pdb.atom_sequences
        self.filepath = pdb.filepath
        # self.chain_id_list = pdb.chain_id_list
        self.entities = pdb.entities
        self.name = pdb.name
        self.pdb_ss_asg = pdb.pdb_ss_asg
        self.cb_coords = pdb.cb_coords
        self.bb_coords = pdb.bb_coords

    def get_ss_asg(self, chain_id="A", stride_exe_path='./stride/stride'):
        pdb_stride = Stride(self.filepath, chain_id, stride_exe_path)
        pdb_stride.run()
        self.pdb_ss_asg = pdb_stride.ss_asg

        return self.pdb_ss_asg

    def readfile(self, filepath, remove_alt_location=False, coordinates_only=True):
        # reads PDB file and feeds PDB instance
        self.filepath = filepath

        with open(self.filepath, 'r') as f:
            pdb = f.readlines()

        available_chain_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
                               '2', '3', '4']
        chain_ids = []
        multimodel, start_of_new_model = False, False
        model_chain_id, curr_chain_id = None, None
        model_chain_index = 0
        entity = None
        for line in pdb:
            line = line.rstrip()
            if not coordinates_only:  # KM added 02/04/20 to deal with handling PDB headers
                if line[0:6] == 'SEQRES':  # KM added 7/25/19 to deal with SEQRES info
                    chain = line[11:12].strip()
                    sequence = line[19:71].strip().split()
                    if chain in self.seqres_sequences:
                        self.seqres_sequences[chain] += sequence
                    else:
                        self.seqres_sequences[chain] = sequence
                elif line[:6] == 'COMPND' and 'MOL_ID' in line:
                    entity = int(line[line.rfind(':') + 1: line.rfind(';')].strip())
                elif line[:6] == 'COMPND' and 'CHAIN' in line and entity:
                    self.entities[entity] = {'chains': line[line.rfind(':') + 1:].strip().rstrip(';').split(',')}
                    entity = None
                elif line[0:6] == 'CRYST1' or line[0:5] == 'SCALE':
                    self.header.append(line)
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
                elif line[:21] == 'REMARK   2 RESOLUTION':
                    try:
                        self.res = float(line[22:30].strip().split()[0])
                    except ValueError:
                        self.res = None
                continue

            if line[0:4] == 'ATOM' or line[17:20] == 'MSE' and line[0:6] == 'HETATM':  # KM modified 2/10/20 for MSE
                # coordinates_only = False
                number = int(line[6:11].strip())
                type = line[12:16].strip()
                alt_location = line[16:17].strip()
                if line[17:20] == 'MSE':  # KM added 2/10/20
                    residue_type = 'MET'  # KM added 2/10/20
                else:  # KM added 2/10/20
                    residue_type = line[17:20].strip()
                if multimodel:
                    if start_of_new_model or line[21:22].strip() != curr_chain_id:
                        curr_chain_id = line[21:22].strip()
                        model_chain_id = available_chain_ids[model_chain_index]
                        model_chain_index += 1
                        start_of_new_model = False
                    # if line[21:22].strip() != curr_chain_id:
                    #     curr_chain_id = line[21:22].strip()
                    #     if not start_of_new_model:  # used as a check of the outer elif for multimodels with multiple chains
                    #         model_chain_id = available_chain_ids[model_chain_index]
                    #         model_chain_index += 1
                    # start_of_new_model = False
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
                    if alt_location == '' or alt_location == 'A':
                        if atom.chain not in chain_ids:
                            chain_ids.append(atom.chain)
                        self.all_atoms.append(atom)
                else:
                    if atom.chain not in chain_ids:
                        chain_ids.append(atom.chain)
                    self.all_atoms.append(atom)
            elif line[0:5] == 'MODEL':
                multimodel = True
                start_of_new_model = True  # signifies that the next line comes after a new model
                # model_chain_id = available_chain_ids[model_chain_index]
                # model_chain_index += 1
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

        self.chain_id_list = chain_ids
        self.renumber_atoms()
        if self.seqres_sequences:
            self.clean_sequences()
        self.update_chain_sequences()

    def clean_sequences(self):  # Ensure the SEQRES information is accurate and convert to 1 AA format and {key: value}
        # if self.sequences:
        for chain in self.seqres_sequences:
            for i, residue in enumerate(self.seqres_sequences[chain]):
                try:
                    self.seqres_sequences[chain][i] = IUPACData.protein_letters_3to1_extended[residue.title()]
                except KeyError:
                    if residue.title() == 'Mse':
                        self.seqres_sequences[chain][i] = 'M'
                    else:
                        self.seqres_sequences[chain][i] = 'X'
            self.seqres_sequences[chain] = ''.join(self.seqres_sequences[chain])

    def get_all_atoms(self):
        return self.all_atoms

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
            self.chain_id_list += chain_ids
        else:
            chain_ids = []
            for atom in atom_list:
                self.all_atoms.append(atom)
                if atom.chain not in chain_ids:
                    chain_ids.append(atom.chain)
            self.chain_id_list += chain_ids
        self.renumber_atoms()
        self.update_chain_sequences()

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

    def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):
        coords, cb_indices = [], []
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
        # Caution, doesn't update SEQRES chain info
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
        self.update_chain_sequences()

    def rename_chain(self, chain_of_interest, new_chain):  # KM Added 8/19 
        # Caution, will rename to already taken chain. Also, doesn't update SEQRES chain info
        
        chain_atoms = self.chain(chain_of_interest)
        for atom in chain_atoms:
            atom.chain = new_chain
            # chain_atoms[i].chain = new_chain

        self.chain_id_list[self.chain_id_list.index(chain_of_interest)] = new_chain
        self.update_chain_sequences()

    def reorder_chains(self, exclude_chains_list=None):  # KM Added 12/16/19
        # Caution, doesn't update SEQRES_sequences chain info
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
        self.update_chain_sequences()

    def renumber_atoms(self):
        for idx, atom in enumerate(self.all_atoms, 1):
            atom.number = idx

    def pose_numbering(self):  # KM Added 12/16/19
        # Starts numbering PDB residues at 1 and numbers sequentially until reaches last atom in file
        last_atom_index = len(self.all_atoms)
        # residues = len(self.get_all_residues())
        idx = 0  # offset , 1
        for i, residue in enumerate(self.get_all_residues(), 1):
            # current_res_num = self.all_atoms[idx].residue_number
            current_res_num = residue.number
            while self.all_atoms[idx].residue_number == current_res_num:
                self.all_atoms[idx].residue_number = i  # + offset
                idx += 1
                if idx == last_atom_index:
                    break
        self.renumber_atoms()  # should be unnecessary

    def reindex_all_chain_residues(self):
        for chain in self.chain_id_list:
            self.reindex_chain_residues(chain)

    def reindex_chain_residues(self, chain):
        # Starts numbering chain residues at 1 and numbers sequentially until reaches last atom in chain
        chain_atoms = self.chain(chain)
        idx = chain_atoms[0].number - 1  # offset to 0
        last_atom_index = idx + len(chain_atoms)
        for i, residue in enumerate(self.get_residues_chain(chain), 1):
            current_res_num = residue.number
            while self.all_atoms[idx].residue_number == current_res_num:
                self.all_atoms[idx].residue_number = i
                idx += 1
                if idx == last_atom_index:
                    break
        self.renumber_atoms()  # should be unnecessary

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
        with open(out_path, "w") as outfile:
            if cryst1 and isinstance(cryst1, str) and cryst1.startswith("CRYST1"):
                outfile.write(str(cryst1) + "\n")
            outfile.write('\n'.join(str(atom) for atom in self.all_atoms))

    def calculate_ss(self, chain_id="A", stride_exe_path='./stride/stride'):
        pdb_stride = Stride(self.filepath, chain_id, stride_exe_path)
        pdb_stride.run()
        self.pdb_ss_asg = pdb_stride.ss_asg

    def getStructureSequence(self, chain_id):
        sequence_list = []
        for atom in self.chain(chain_id):
            if atom.is_CA():
                sequence_list.append(atom.residue_type)
        one_letter = ''.join([IUPACData.protein_letters_3to1_extended[k.title()]
                              if k.title() in IUPACData.protein_letters_3to1_extended else '-' for k in sequence_list])

        return one_letter

    def update_chain_sequences(self):
        self.atom_sequences = {chain: self.getStructureSequence(chain) for chain in self.chain_id_list}

    def orient(self, symm, orient_dir, generate_oriented_pdb=True):
        # self.reindex_all_chain_residues()  TODO test efficacy. It could be that this screws up more than helps.
        self.write('input.pdb')
        # os.system('cp %s input.pdb' % self.filepath)
        # os.system('%s/orient_oligomer_rmsd >> orient.out 2>&1 << eof\n%s/%s\neof' % (orient_dir, orient_dir, symm))
        os.system('%s/orient_oligomer >> orient.out 2>&1 << eof\n%s/%s_symm.txt\neof' % (orient_dir, orient_dir, symm))
        os.system('mv output.pdb %s_orient.pdb' % os.path.splitext(self.filepath)[0])  # Todo this could be removed
        os.system('rm input.pdb')
        if os.path.exists('%s_orient.pdb' % os.path.splitext(self.filepath)[0]):
            if generate_oriented_pdb:
                oriented_pdb = PDB()
                oriented_pdb.readfile('%s_orient.pdb' % os.path.splitext(self.filepath)[0], remove_alt_location=True)
                os.system('rm %s_orient.pdb' % os.path.splitext(self.filepath)[0])  # Todo this could be removed
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

    def get_cb_indices_chain(self, chain, InclGlyCA=False):
        cb_indices = []
        for i in range(len(self.all_atoms)):
            if self.all_atoms[i].chain == chain:
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

    def get_sasa(self, free_sasa_exe_path, probe_radius=1.4, sasa_thresh=0):
        proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath]
                                , stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        out_lines = out.decode('utf-8').split('\n')

        sasa_out_chain, sasa_out_res, sasa_out = [], [], []
        for line in out_lines:
            if line != "\n" and line != "" and not line.startswith("#"):
                chain_id = line[4:5]
                res_num = int(line[5:10])
                sasa = float(line[16:])
                if sasa >= sasa_thresh:
                    sasa_out_chain.append(chain_id)
                    sasa_out_res.append(res_num)
                    sasa_out.append(sasa)

        return sasa_out_chain, sasa_out_res, sasa_out

    def get_surface_helix_cb_indices(self, stride_exe_path, free_sasa_exe_path, probe_radius=1.4, sasa_thresh=1):
        # only works for monomers or homo-complexes
        # proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath]
        #                         , stdout=subprocess.PIPE)
        # (out, err) = proc.communicate()
        # out_lines = out.decode('utf-8').split('\n')
        # sasa_out = []
        # for line in out_lines:
        #     if line != "\n" and line != "" and not line.startswith("#"):
        #         res_num = int(line[5:10])
        #         sasa = float(line[16:])
        #         if sasa >= sasa_thresh:
        #             if res_num not in sasa_out:
        #                 sasa_out.append(res_num)
        sasa_chain, sasa_res, sasa = self.get_sasa(free_sasa_exe_path, probe_radius=probe_radius, sasa_thresh=sasa_thresh)

        h_cb_indices = []
        stride = Stride(self.filepath, self.chain_id_list[0], stride_exe_path)
        stride.run()
        stride_ss_asg = stride.ss_asg
        for i in range(len(self.all_atoms)):
            atom = self.all_atoms[i]
            if atom.is_CB():
                if (atom.residue_number, "H") in stride_ss_asg and atom.residue_number in sasa_res:
                    h_cb_indices.append(i)
        return h_cb_indices

    def get_surface_atoms(self, free_sasa_exe_path, chain_selection="all", probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        # proc = subprocess.Popen('%s --format=seq --probe-radius %s %s' %(free_sasa_exe_path, str(probe_radius), self.filepath), stdout=subprocess.PIPE, shell=True)
        # (out, err) = proc.communicate()
        # out_lines = out.split("\n")
        # sasa_out = []
        # for line in out_lines:
        #     if line != "\n" and line != "" and not line.startswith("#"):
        #         chain_id = line[4:5]
        #         res_num = int(line[5:10])
        #         sasa = float(line[16:])
        #         if sasa > sasa_thresh:
        #             sasa_out.append((chain_id, res_num))
        sasa_chain, sasa_res, sasa = self.get_sasa(free_sasa_exe_path, probe_radius=probe_radius, sasa_thresh=sasa_thresh)
        sasa_chain_res_l = zip(sasa_chain, sasa_res)

        surface_atoms = []
        if chain_selection == "all":
            for atom in self.all_atoms:
                if (atom.chain, atom.residue_number) in sasa_chain_res_l:
                    surface_atoms.append(atom)
        else:
            for atom in self.chain(chain_selection):
                if (atom.chain, atom.residue_number) in sasa_chain_res_l:
                    surface_atoms.append(atom)

        return surface_atoms

    def get_surface_resdiue_info(self, free_sasa_exe_path, probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        # proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath]
        #                         , stdout=subprocess.PIPE)
        # (out, err) = proc.communicate()
        # out_lines = out.decode('utf-8').split('\n')
        # sasa_out = []
        # for line in out_lines:
        #     if line != "\n" and line != "" and not line.startswith("#"):
        #         chain_id = line[4:5]
        #         res_num = int(line[5:10])
        #         sasa = float(line[16:])
        #         if sasa > sasa_thresh:
        #             sasa_out.append((chain_id, res_num))
        sasa_chain, sasa_res, sasa = self.get_sasa(free_sasa_exe_path, probe_radius=probe_radius, sasa_thresh=sasa_thresh)

        return list(set(zip(sasa_chain, sasa_res)))

    def get_chain_residue_surface_area(self, chain_residue_pairs, free_sasa_exe_path, probe_radius=2.2):
        # only works for monomers or homo-complexes
        # proc = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath]
        #                         , stdout=subprocess.PIPE)
        # (out, err) = proc.communicate()
        # out_lines = out.decode('utf-8').split('\n')
        # sasa_out = 0
        # for line in out_lines:
        #     if line != "\n" and line != "" and not line.startswith("#"):
        #         chain_id = line[4:5]
        #         res_num = int(line[5:10])
        #         sasa = float(line[16:])
        #         if (chain_id, res_num) in chain_residue_pairs:
        #             sasa_out += sasa

        sasa_chain, sasa_res, sasa = self.get_sasa(free_sasa_exe_path, probe_radius=probe_radius)
        total_sasa = 0
        for chain, res, sasa in zip(sasa_chain, sasa_res, sasa):
            if (chain, res) in chain_residue_pairs:
                total_sasa += sasa

        return total_sasa

    def mutate_to(self, chain, residue, res_id='ALA'):  # KM added 12/31/19 to mutate pdb Residue objects to alanine
        """Mutate specific chain and residue to a new residue type. Type can be 1 or 3 letter format"""
        # if using residue number, then residue_atom_list[i] is necessary
        # else using Residue object, residue.atom_list[i] is necessary
        if res_id in IUPACData.protein_letters_1to3:
            res_id = IUPACData.protein_letters_1to3[res_id]

        residue_atom_list = self.getResidueAtoms(chain, residue)  # residue.atom_list
        delete = []
        # for i in range(len(residue_atom_list)):
        for i, atom in enumerate(residue_atom_list):
            # if residue_atom_list[i].is_backbone() or residue_atom_list[i].is_CB():
            #     residue_atom_list[i].residue_type = res_id.upper()
            if atom.is_backbone() or atom.is_CB():
                residue_atom_list[i].residue_type = res_id.upper()
            else:
                delete.append(i)
        # TODO using AA reference lib, align the backbone + CB atoms of the residue then insert all side chain atoms
        if delete:
            # delete = sorted(delete, reverse=True)
            # for j in delete:
            for j in reversed(delete):
                i = residue_atom_list[j]
                self.all_atoms.remove(i)
            # self.delete_atoms(residue_atom_list[j] for j in reversed(delete))  # TODO use this instead
            self.renumber_atoms()

    def insert_residue(self, chain, residue, residue_type):  # KM added 08/01/20, only works for pose_numbering now
        if residue_type.title() in IUPACData.protein_letters_3to1_extended:
            residue_type = IUPACData.protein_letters_3to1_extended[residue_type.title()]
        else:
            residue_type = residue_type.upper()

        # Find atom insertion index, should be last atom in preceding residue
        if residue == 1:
            insert_atom_idx = 0
        else:
            # This assumes atom numbers are proper idx
            residue_atoms = self.getResidueAtoms(chain, residue)
            if residue_atoms:
                insert_atom_idx = residue_atoms[0].number - 1  # subtract one from atom number to get the atom index
            else:  # Atom index is not an insert as this is the C-term
                # use length of all_chain_atoms + length of all prior chains
                # prior_index = self.getResidueAtoms(chain, residue)[0].number - 1
                chain_atoms = self.chain(chain)
                insert_atom_idx = len(chain_atoms) + chain_atoms[0].number - 1

            # insert_atom_idx = self.getResidueAtoms(chain, residue)[0].number

        # Change all downstream residues
        for atom in self.all_atoms[insert_atom_idx:]:
            # atom.number += len(insert_atoms)
            # if atom.chain == chain: TODO uncomment for pdb numbering
            atom.residue_number += 1

        # Grab the reference atom coordinates and push into the atom list
        ref_aa = PDB()  # TODO clean up speed by include in Atom.py or Residue.py or as read_atom_list()
        ref_aa.readfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AAreference.pdb'))
        insert_atoms = ref_aa.getResidueAtoms('A', IUPACData.protein_letters.find(residue_type))
        for atom in reversed(insert_atoms):  # essentially a push
            # atom.number += insert_atom_idx + 1
            atom.chain = chain
            atom.residue_number = residue
            atom.occ = 0
            self.all_atoms.insert(insert_atom_idx, atom)
        self.renumber_atoms()

    def delete_residue(self, chain, residue):  # KM added 08/25/20 to remove missing residues between two files
        # start = len(self.all_atoms)
        # print(len(self.all_atoms))
        self.delete_atoms(self.getResidueAtoms(chain, residue))
        self.renumber_atoms()
        # print('Deleted: %d atoms' % (start - len(self.all_atoms)))

    def delete_atoms(self, atoms):
        # Need to call self.renumber_atoms() after every call to delete_atoms()
        for atom in atoms:
            self.all_atoms.remove(atom)

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

    def get_all_entities(self):  # KM added 08/21/20 to format or the ASU
        """Find all unique entities in the pdb file, these are unique sequence/structure objects"""
        # seq_d = {chain: self.getStructureSequence(chain) for chain in self.chain_id_list}
        # self.entities[copy.copy(count)] = {'chains': [self.chain_id_list[0]], 'seq': seq_d[self.chain_id_list[0]]}
        count = 0
        for chain in self.atom_sequences:
            new_entity = True  # assume all chains are unique entities
            for entity in self.entities:
                if self.atom_sequences[chain] == self.entities[entity]['seq']:
                    score = len(self.atom_sequences[chain])
                else:
                    alignment = pairwise2.align.localxx(self.atom_sequences[chain], self.entities[entity]['seq'])
                    score = alignment[0][2]  # first alignment, score value
                if score / len(self.entities[entity]['seq']) > 0.9:
                    # rmsd = Bio.Superimposer()
                    # if rmsd > 1:
                    self.entities[entity]['chains'].append(chain)
                    new_entity = False  # The entity is not unique, do not add
                    break
            if new_entity:  # nothing was found
                self.entities[copy.copy(count)] = {'chains': [chain], 'seq': self.atom_sequences[chain]}
                count += 1

    def find_entity(self, chain):
        for entity in self.entities:
            if chain in self.entities[entity]['chains']:
                return entity

    def match_entity_by_struct(self, other_struct=None, entity=None, force_closest=False):
        return None  # TODO when entities are structure compatible

    def match_entity_by_seq(self, other_seq=None, force_closest=False, threshold=0.7):
        """From another sequence or set of atoms, returns the first matching chain from the corresponding entity"""

        if force_closest:
            alignment_score_d = {}
            for entity in self.entities:
                # TODO get a gap penalty and rework entire alignment function...
                alignment = pairwise2.align.localxx(other_seq, self.entities[entity]['seq'])
                max_align_score, max_alignment = 0, None
                for i, align in enumerate(alignment):
                    if align.score > max_align_score:
                        max_align_score = align.score
                        max_alignment = i
                alignment_score_d[entity] = alignment[max_alignment].score
                # alignment_score_d[entity] = alignment[0][2]

            max_score, max_score_entity = 0, None
            for entity in alignment_score_d:
                normalized_score = alignment_score_d[entity] / len(self.entities[entity]['seq'])
                if normalized_score > max_score:
                    max_score = normalized_score  # alignment_score_d[entity]
                    max_score_entity = entity
            if max_score > threshold:
                return self.entities[max_score_entity]['chains'][0]
            else:
                return None
        else:
            for entity in self.entities:
                if other_seq == self.entities[entity]['seq']:
                    return self.entities[entity]['chains'][0]

    def chain_interface_contacts(self, chain_id, distance=8, gly_ca=False):
        """Create a atom tree using CB atoms from one chain and all other atoms

        Args:
            chain_id (PDB): First PDB to query against
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
            gly_ca=False (bool): Whether glycine CA should be included in the tree
        Returns:
            chain_atoms, all_contact_atoms (list, list): Chain interface atoms, all contacting interface atoms
        """
        # Get chain CB Atom Coordinates into a numpy array [[x, y, z], ...]
        chain_coords = numpy.array(self.extract_CB_coords_chain(chain_id, InclGlyCA=gly_ca))

        # Construct CB Tree for the chain
        chain_tree = BallTree(chain_coords)

        # Get CB Atom indices for the chain CB and all_atoms CB
        chain_cb_indices = self.get_cb_indices_chain(chain_id, InclGlyCA=gly_ca)
        all_cb_indices = self.get_cb_indices(InclGlyCA=gly_ca)
        chain_coord_indices, contact_cb_indices = [], []
        # Find all the contacting CB indices and the indices where chain specific coords are located in all_coords
        for i, idx in enumerate(all_cb_indices):
            if idx not in chain_cb_indices:
                contact_cb_indices.append(idx)
            else:
                chain_coord_indices.append(i)

        # Get all CB Atom Coordinates into a numpy array [[x, y, z], ...]
        all_coords = numpy.array(self.extract_CB_coords(InclGlyCA=gly_ca))
        # Remove chain specific coords from all coords by deleting them from numpy
        contact_coords = numpy.delete(all_coords, chain_coord_indices, axis=0)
        # Query chain CB Tree for all contacting Atoms within distance
        chain_query = chain_tree.query_radius(contact_coords, distance)

        all_contact_atoms, chain_atoms = [], []
        for contact_idx, contacts in enumerate(chain_query):
            if chain_query[contact_idx].tolist() != list():
                all_contact_atoms.append(self.all_atoms[contact_cb_indices[contact_idx]])
                # residues2.append(pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number)
                # for pdb1_index in chain_query[contact_idx]:
                for chain_idx in contacts:
                    chain_atoms.append(self.all_atoms[chain_cb_indices[chain_idx]])

        return chain_atoms, all_contact_atoms

    def get_asu(self, chain=None, extra=False):
        """Return the atoms involved in the ASU with the provided chain

        Keyword Args:
            chain=None (str): The identity of the target asu
        Returns:
            (list): List of atoms involved in the identified asu
        """
        self.get_all_entities()
        if not chain:
            chain = self.chain_id_list[0]

        def get_unique_contacts(chain, chain_entity=0, iteration=0, extra=False, partner_entity=None):
            unique_chains_entity = {}
            # unique_chains_entity, chain_entity, iteration = {}, None, 0
            while unique_chains_entity == dict():
                # print(iteration, chain_entity)
                if iteration != 0:  # search through the chains found in an entity
                    chain = self.entities[chain_entity]['chains'][iteration]
                    # print(chain)
                chain_interface_atoms, all_contacting_interface_atoms = self.chain_interface_contacts(chain, gly_ca=True)
                # print(self.entities)
                interface_d = {}
                for atom in all_contacting_interface_atoms:
                    if atom.chain not in interface_d:
                        interface_d[atom.chain] = [atom]
                    else:
                        interface_d[atom.chain].append(atom)
                # print(interface_d)
                # copy.deepcopy(interface_d)
                partner_interface_d, self_interface_d = {}, {}
                for _chain in self.entities[chain_entity]['chains']:
                    if _chain != chain:
                        if _chain in interface_d:
                            self_interface_d[_chain] = interface_d[_chain]
                partner_interface_d = {_chain: interface_d[_chain] for _chain in interface_d
                                       if _chain not in self_interface_d}

                if not partner_entity:  # if an entity in particular is desired as in the extras recursion
                    partner_entity = set(self.entities.keys()) - {chain_entity}

                if not extra:
                    # Find the top contacting chain from each unique partner entity
                    for p_entity in partner_entity:
                        max_contact, max_contact_chain = 0, None
                        for _chain in partner_interface_d:
                            # print('Partner: %s' % _chain)
                            if _chain not in self.entities[p_entity]['chains']:
                                continue  # ensure that the chain is relevant to this entity
                            if len(partner_interface_d[_chain]) > max_contact:
                                # print('Partner GREATER!: %s' % _chain)
                                max_contact = len(partner_interface_d[_chain])
                                max_contact_chain = _chain
                        if max_contact_chain:
                            unique_chains_entity[max_contact_chain] = p_entity  # set the max partner for this entity

                    # return list(unique_chains_entity.keys())
                else:  # solve the asu by expansion to extra contacts
                    # partner_entity_chains_first_entity_contact_d = {} TODO define here if iterate over all entities?
                    extra_first_entity_chains, first_entity_chain_contacts = [], []
                    for p_entity in partner_entity:  # search over all entities
                        # Find all partner chains in the entity in contact with chain of interest
                        # partner_chains_entity = {partner_chain: p_entity for partner_chain in partner_interface_d
                        #                          if partner_chain in self.entities[p_entity]['chains']}
                        # partner_chains_entity = [partner_chain for partner_chain in partner_interface_d
                        #                          if partner_chain in self.entities[p_entity]['chains']]
                        # found_chains += [found_chain for found_chain in unique_chains_entity.keys()]

                        # Get the most contacted chain from first entity, in contact with chain of the partner entity
                        for partner_chain in partner_interface_d:
                            if partner_chain in self.entities[p_entity]['chains']:
                                # print(partner_chain)
                                partner_chains_first_entity_contact = \
                                    get_unique_contacts(partner_chain, chain_entity=p_entity,
                                                        partner_entity=chain_entity)
                                print('Partner entity %s, original chain contacts: %s' %
                                      (p_entity, partner_chains_first_entity_contact))
                                # Only include chain/partner entities that are also in contact with chain of interest
                        # for partner_chain in partner_chains_first_entity_contact_d:
                                # Caution: this logic is flawed when contact is part of oligomer, but not touching
                                # original chain... EX: A,B,C,D tetramer with first entity '0', chain C touching second
                                # entity '1', chain R and chain R touching first entity '0', chain A. A and C don't
                                # contact though. Is this even possible?

                                # Choose only the partner chains that don't have first entity chain as the top contact
                                if partner_chains_first_entity_contact[0] != chain:
                                    # Check if there is a first entity contact as well, if so partner chain is a hit
                                    # original_chain_partner_contacts = list(
                                    #     set(self_interface_d.keys())  # this scope is still valid
                                    #     & set(partner_entity_chains_first_entity_contact_d[p_entity][partner_chain]))
                                    # if original_chain_partner_contacts != list():
                                    if partner_chains_first_entity_contact in self_interface_d.keys():
                                        # chain_contacts += list(
                                        #     set(self_interface_d.keys())
                                        #     & set(partner_entity_partner_chain_first_chain_d[entity][partner_chain]))
                                        first_entity_chain_contacts.append(partner_chain)
                                        extra_first_entity_chains += partner_chains_first_entity_contact

                            # chain_contacts += list(
                            #     set(self_interface_d.keys())
                            #     & set(partner_entity_partner_chain_first_chain_d[entity][partner_chain]))
                    print('All original chain contacts: %s' % extra_first_entity_chains)
                    all_asu_chains = list(set(first_entity_chain_contacts)) + extra_first_entity_chains
                    unique_chains_entity = {_chain: self.find_entity(_chain) for _chain in all_asu_chains}
                    # need to make sure that the partner entity chains are all contacting as well...
                    # for chain in found_chains:
                # print('partners: %s' % unique_entity_chains)
                iteration += 1
            return list(unique_chains_entity.keys())
            # return list(set(first_entity_chain_contacts)) + extra_first_entity_chains
            # return unique_chains_entities

        unique_chains = get_unique_contacts(chain, chain_entity=self.find_entity(chain), extra=extra)

        asu = self.chain(chain)
        for atoms in [self.chain(partner_chain) for partner_chain in unique_chains]:
            asu += atoms

        return asu

    def return_asu(self, chain='A'):  # , outpath=None):
        """Returns the ASU as a new PDB object. See self.get_asu() for method"""
        asu_pdb = PDB()
        # asu_pdb.__dict__ = self.__dict__.copy()
        asu_pdb.read_atom_list(self.get_asu(chain=chain))
        return asu_pdb

        # if outpath:
        #     asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(self.filepath))[0] + '.pdb')
        #     # asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(file))[0] + '_%s' % 'asu.pdb')
        # else:
        #     asu_file_name = os.path.splitext(self.filepath)[0] + '_asu.pdb'

        # asu_pdb = fill_pdb(pdb.get_asu(chain))
        # asu_pdb.write(asu_file_name, cryst1=asu_pdb.cryst)
        #
        # return asu_file_name
