import argparse
import math
import os

from PDB import PDB


# from Stride import Stride
from Structure import superposition3d


class ListFile:
    def __init__(self):
        self.path = None
        self.list_file = []

    def read(self, path):
        self.path = path
        f = open(self.path, "r")
        file_lines = f.readlines()
        f.close()

        for line in file_lines:
            if line != '\n':
                line = line.rstrip()
                self.list_file.append(line)


class FetchPDBBA:
    def __init__(self, pdb_listfile_path):
        lf = ListFile()
        lf.read(pdb_listfile_path)
        self.pdblist = lf.list_file

    def fetch(self):
        print('FETCHING PDB FILES')
        for pdb in self.pdblist:
            os.system('wget https://files.rcsb.org/download/%s.pdb1 >> fetch_pdb.out 2>&1' % pdb.rstrip())
        print('DONE FETCHING PDB FILES')


# class Atom:
#     def __init__(self, number, type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y, z, occ, temp_fact, element_symbol, atom_charge):
#         self.number = number
#         self.type = type
#         self.alt_location = alt_location
#         self.residue_type = residue_type
#         self.chain = chain
#         self.residue_number = residue_number
#         self.code_for_insertion = code_for_insertion
#         self.x = x
#         self.y = y
#         self.z = z
#         self.occ = occ
#         self.temp_fact = temp_fact
#         self.element_symbol = element_symbol
#         self.atom_charge = atom_charge
#
#     def __str__(self):
#         # prints Atom in PDB format
#         return "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format("ATOM", self.number, self.type, self.alt_location, self.residue_type, self.chain, self.residue_number, self.code_for_insertion, self.x, self.y, self.z, self.occ, self.temp_fact, self.element_symbol, self.atom_charge)
#
#     def is_backbone(self):
#         # returns True if atom is part of the proteins backbone and False otherwise
#         backbone_specific_atom_type = ["N", "CA", "C", "O"]
#         if self.type in backbone_specific_atom_type:
#             return True
#         else:
#             return False
#
#     def is_cb(self):
#         return self.type == "CB" or (self.type== "H" and self.residue_type == "GLY" )
#
#     def distance(self, atom, intra=False):
#         # returns distance (type float) between current instance of Atom and another instance of Atom
#         if self.chain == atom.chain and not intra:
#             print("Atoms Are In The Same Chain")
#             return None
#         else:
#             distance = math.sqrt((self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2)
#             return distance
#
#     def distance_squared(self, atom, intra = False):
#         # returns distance (type float) between current instance of Atom and another instance of Atom
#         if self.chain == atom.chain and not intra:
#             print("Atoms Are In The Same Chain")
#             return None
#         else:
#             distance = (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2
#             return distance
#
#     def is_axis(self):
#         if self.chain == "7" or self.chain == "8" or self.chain == "9":
#             return True
#         else:
#             return False
#
#     def __eq__(self, other):
#         return self.number == other.number and self.chain == other.chain


class AtomPair:
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
        self.distance = atom1.distance(atom2)

    def __str__(self):
        return "%s - %s: %f" % (self.atom1.residue_number, self.atom2.residue_number, self.distance)


# class PDB:  # REMOVED to get rid of multiple PDB classes. Some functions important here may not be there. In that case they should be added
#     def __init__(self):
#         self.all_atoms = []  # python list of Atoms
#         self.res = None
#         self.cryst = None
#         self.dbref = {}
#         self.header = []
#         self.sequence_dictionary = {} # python dictionary of SEQRES entries. key is chainID, value is [ 'length', '3 letter AA Seq']. Ex: {'A': ['124', 'ALA GLN GLY PHE...']}
#         self.filepath = None  # PDB filepath if instance is read from PDB file
#         self.chain_ids = []  # list of unique chain IDs in PDB
#         self.interface_atoms = []
#         self.name = None
#         self.crystal_linker_term_dist_1 = None
#         self.crystal_linker_term_dist_2 = None
#
#     def AddName(self, name):
#         self.name = name
#
#     def retrieve_chain_ids(self):
#         # creates a list of unique chain IDs in PDB and feeds it into chain_ids
#         chain_ids = []
#         for atom in self.all_atoms:
#             chain_ids.append(atom.chain)
#         chain_ids = list(set(chain_ids))
#         chain_ids.sort(key=lambda x: (x[0].isdigit(), x))
#         self.chain_ids = chain_ids
#
#     def readfile(self, filepath):
#         # reads PDB file and feeds PDB instance
#         self.filepath = filepath
#
#         f = open(filepath, "r")
#         pdb = f.readlines()
#         f.close()
#         multimodel = 0
#         _number_of_chains_model_1 = 0
#         _unique_chain_list = []
#         for line in pdb:
#             line = line.rstrip()
#             if line[:22] == 'REMARK   2 RESOLUTION.':
#                 try:
#                     self.res = float(line[22:30].strip().split()[0])
#                 except ValueError:
#                     self.res = None
#             elif line[0:6] == "CRYST1" or line[0:5] == "SCALE":
#                 self.header.append(line)
#                 if line[0:6] == 'CRYST1':
#                     try:
#                         a = float(line[6:15].strip())
#                         b = float(line[15:24].strip())
#                         c = float(line[24:33].strip())
#                         ang_a = float(line[33:40].strip())
#                         ang_b = float(line[40:47].strip())
#                         ang_c = float(line[47:54].strip())
#                     except ValueError:
#                         a, b, c = 0.0, 0.0, 0.0
#                         ang_a, ang_b, ang_c = a, b, c
#                     space_group = line[55:66].strip()
#                     self.cryst = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
#             elif line[0:5] == 'DBREF':
#                 line = line.strip()
#                 chain = line[12:14].strip().upper()
#                 if line[5:6] == '2':
#                     db_accession_id = line[18:40].strip()
#                 else:
#                     db = line[26:33].strip()
#                     if line[5:6] == '1':
#                         continue
#                     db_accession_id = line[33:42].strip()
#                 self.dbref[chain] = {'db': db, 'accession': db_accession_id}
#             elif line[0:5] == "MODEL" and int(line[10:14].strip()) > 1:
#                 multimodel += 1
#             elif line[0:4] == "ATOM":
#                 header = False
#                 if line[21:22].strip() not in _unique_chain_list and multimodel == 0:
#                     _unique_chain_list.append(line[21:22].strip())
#                     _number_of_chains_model_1 += 1
#                 number = int(line[6:11].strip())
#                 type = line[12:16].strip()
#                 alt_location = line[16:17].strip()
#                 residue_type = line[17:20].strip()
#                 chain = line[21:22].strip()
#                 if multimodel > 0:
#                     chain = chr(ord(chain) + (multimodel * _number_of_chains_model_1))
#                 residue_number = int(line[22:26].strip())
#                 code_for_insertion = line[26:27].strip()
#                 x = float(line[30:38].strip())
#                 y = float(line[38:46].strip())
#                 z = float(line[46:54].strip())
#                 occ = float(line[54:60].strip())
#                 temp_fact = float(line[60:66].strip())
#                 element_symbol = line[76:78].strip()
#                 atom_charge = line[78:80].strip()
#                 atom = Atom(number, type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y, z, occ, temp_fact, element_symbol, atom_charge)
#                 self.all_atoms.append(atom)
#
#         self.retrieve_chain_ids()
#         self.retrieve_sequences(pdb, multimodel)
#
#     # KM added 7/25/19 to deal with SEQRES info
#     def retrieve_sequences(self, pdb, multimodel):
#         # with open(self.filepath, "r") as f:
#         #     pdb = f.readlines()
#         seq_list = []
#         for line in pdb:
#             if line[0:6] == "SEQRES":
#                 chain = line[11:12].strip()
#                 # sequence_length = line[13:18].strip()
#                 sequence = line[19:71]
#                 seq_line = [chain, sequence]  # sequence_length, sequence]
#                 seq_list.append(seq_line)
#         for line in seq_list:
#             if line[0] not in self.sequence_dictionary:
#                 self.sequence_dictionary[line[0]] = line[1]  # 2]
#             else:
#                 self.sequence_dictionary[line[0]] += line[1]  # 2]
#         try:
#             first_chain = self.chain_ids[0]
#             try:
#                 first_seq = self.sequence_dictionary[first_chain].strip().split(' ')
#                 if multimodel > 0:
#                     print('CAUTION: Multimodel sequence extraction')
#                     for chain in self.chain_ids:
#                         self.sequence_dictionary[chain] = first_seq
#                 else:
#                     for chain in self.chain_ids:
#                         try:
#                             self.sequence_dictionary[chain] = self.sequence_dictionary[chain].strip().split(' ')
#                         except KeyError:  # when there are fewer SEQRES than chains, make all chains have first sequence
#                             self.sequence_dictionary[chain] = first_seq
#             except (KeyError, AttributeError):
#                 for chain in self.chain_ids:
#                     # where PDB originated from outside the official PDB distribution. Probably a design
#                     # print('This PDB file has no SEQRES, taking sequence from ATOM record')
#         except IndexError:
#             self.sequence_dictionary = {}
#
#     # KM added 9/13/19
#     def modify_temp_fact(self, dictionary, data):
#         check = self.all_atoms[0]
#         residue_string = []
#         offset = 0
#
#         # ensure that the starting residue of the atom record aligns with the starting residue of the dictionary
#         if check.residue_number != 1:
#             # if not, first, grab the first 4 residues of the atom record
#             start = check.residue_number
#             residue_string.append(protein_letters_3to1[check.residue_type.title()])
#             i = start + 1
#             for atom in self.all_atoms:
#                 if i == start + 4:
#                     break
#                 elif atom.type == 'N' and atom.residue_number == i:
#                     residue_string.append(protein_letters_3to1[atom.residue_type.title()])
#                     i += 1
#                     continue
#
#             # search for index where the first 4 residues of pdb.all_atoms matches first 4 residues of the dictionary
#             index, match = 1, 0
#             while match != 4:
#                 for i in range(4):
#                     if residue_string[i] == dictionary[index + i]['aa']:
#                         match += 1
#                         continue
#                     else:
#                         index += 1
#                         match = 0
#                         break
#             offset = start - index
#
#         keys = True
#         while keys:
#             try:
#                 if dictionary[1][data]:
#                     for atom in self.all_atoms:
#                         atom.temp_fact = dictionary[atom.residue_number - offset][data]
#                     keys = False
#             except KeyError:
#                 print('\n')
#                 print('Key \"%s\" not found in score dictionary...' % data)
#                 print('Try from list of possible scores:')
#                 print(dictionary[1].keys())
#                 print('\n')
#                 data = input('Which are you interested in?: ')
#
#     def read_atom_list(self, atom_list):
#         # reads a python list of Atoms and feeds PDB instance
#         for atom in atom_list:
#             self.all_atoms.append(atom)
#         self.retrieve_chain_ids()
#
#     def chain(self, chain_id):
#         # returns a python list of Atoms containing the subset of Atoms in the PDB instance that belong to the selected chain ID
#         selected_atoms = []
#         for atom in self.all_atoms:
#             if atom.chain == chain_id:
#                 selected_atoms.append(atom)
#         return selected_atoms
#     #
#     # def chains(self, chain_ids):
#     #     # returns a python list of Atoms containing the subset of Atoms in the PDB instance that belong to the selected chain IDs
#     #     selected_atoms = []
#     #     for atom in self.all_atoms:
#     #         if atom.chain in chain_ids:
#     #             selected_atoms.append(atom)
#     #     return selected_atoms
#     #
#     # def getAtom(self, other_atom):
#     #     # returns Atom in self that has the same number, type, residue_type, chain and residue_number as atom if it exists
#     #     # returns None otherwise
#     #     out_atom = None
#     #     for self_atom in self.all_atoms:
#     #         if self_atom == other_atom:
#     #             out_atom = self_atom
#     #             break
#     #     return out_atom
#
#     def is_dimer(self):
#         # returns True if PDB instance is a dimer (contains exactly 2 chains) and False otherwise
#         if len(self.chain_ids) == 2:
#             return True
#         else:
#             return False
#
#     def is_tetramer(self):
#         # returns True if PDB instance is a Tetramer (contains exactly 4 chains) and False otherwise
#         if len(self.chain_ids) == 4:
#             return True
#         else:
#             return False
#
#     def dimer_interface_residues(self, CA_DIST_THRESH=10):
#         # returns a python list of Atoms, which are the subset of Atoms in the PDB instance that are at the interface
#         # only works if PDB instance is a dimer
#         if self.is_dimer():
#
#             # define interface chains
#             protomer1 = self.chain(self.chain_ids[0])
#             protomer2 = self.chain(self.chain_ids[1])
#
#             # create list of CA atom pairs
#             atompairs = []
#             for atom1 in protomer1:
#                 for atom2 in protomer2:
#                     if atom1.type == "CA" and atom2.type == "CA":
#                         pair = AtomPair(atom1, atom2)
#                         atompairs.append(pair)
#
#             atompairs_under_thresh = []
#             for atompair in atompairs:
#                 if atompair.distance < CA_DIST_THRESH:
#                     atompairs_under_thresh.append(atompair)
#
#             interface_residue_numbers_1 = []
#             interface_residue_numbers_2 = []
#             for atompair in atompairs_under_thresh:
#                 interface_residue_numbers_1.append(atompair.atom1.residue_number)
#                 interface_residue_numbers_2.append(atompair.atom2.residue_number)
#
#             interface_residue_numbers_1 = list(set(interface_residue_numbers_1))
#             interface_residue_numbers_2 = list(set(interface_residue_numbers_2))
#
#             self.interface_atoms = []
#             for atom1 in protomer1:
#                 if atom1.residue_number in interface_residue_numbers_1:
#                     self.interface_atoms.append(atom1)
#
#             for atom2 in protomer2:
#                 if atom2.residue_number in interface_residue_numbers_2:
#                     self.interface_atoms.append(atom2)
#
#             allinterface_atoms = self.interface_atoms
#
#             return self.removeduplicates(allinterface_atoms)
#
#         else:
#             print("THIS PDB IS NOT A DIMER: ") + self.filepath + '\n'
#             return []
#
#     def removeduplicates(self, atomlist):
#         nodup = []
#         for x in atomlist:
#             if x not in nodup:
#                 nodup.append(x)
#         return nodup
#
#     def extract_coords_subset(self, res_start, res_end, chain_index, CA):
#         if CA:
#             selected_atoms = []
#             for atom in self.chain(self.chain_ids[chain_index]):
#                 if atom.type == "CA":
#                     if atom.residue_number >= res_start and atom.residue_number <= res_end:
#                         selected_atoms.append(atom)
#             out_coords = []
#             for atom in selected_atoms:
#                 [x, y, z] = [atom.x, atom.y, atom.z]
#                 out_coords.append([x, y, z])
#             return out_coords
#         else:
#             selected_atoms = []
#             for atom in self.chain(self.chain_ids[chain_index]):
#                 if atom.residue_number >= res_start and atom.residue_number <= res_end:
#                     selected_atoms.append(atom)
#             out_coords = []
#             for atom in selected_atoms:
#                 [x, y, z] = [atom.x, atom.y, atom.z]
#                 out_coords.append([x, y, z])
#             return out_coords
#
#     def extract_coords(self):
#         coords = []
#         for atom in self.all_atoms:
#             [x, y, z] = [atom.x, atom.y, atom.z]
#             coords.append([x, y, z])
#         return coords
#
#     def extract_backbone_coords(self):
#         coords = []
#         for atom in self.all_atoms:
#             if atom.is_backbone():
#                 [x, y, z] = [atom.x, atom.y, atom.z]
#                 coords.append([x, y, z])
#         return coords
#
#     def set_atom_coordinates(self, new_cords):
#         for i in range(len(self.all_atoms)):
#             self.all_atoms[i].x, self.all_atoms[i].y, self.all_atoms[i].z = new_cords[i][0], new_cords[i][1], \
#                                                                             new_cords[i][2]
#
#     def mat_vec_mul3(self, a, b):
#         c = [0. for i in range(3)]
#         for i in range(3):
#             c[i] = 0.
#             for j in range(3):
#                 c[i] += a[i][j] * b[j]
#         return c
#
#     def apply(self, rot, tx):
#         moved = []
#         for coord in self.extract_coords():
#             coord_moved = self.mat_vec_mul3(rot, coord)
#             for j in range(3):
#                 coord_moved[j] += tx[j]
#             moved.append(coord_moved)
#         self.set_atom_coordinates(moved)
#
#     def apply_transformation_to_D2axes(self, rot, tx):
#         moved_axis_x = []
#         for coord in self.axisX():
#             coord_moved_x = self.mat_vec_mul3(rot, coord)
#             for j in range(3):
#                 coord_moved_x[j] += tx[j]
#             moved_axis_x.append(coord_moved_x)
#
#         moved_axis_y = []
#         for coord in self.axisY():
#             coord_moved_y = self.mat_vec_mul3(rot, coord)
#             for j in range(3):
#                 coord_moved_y[j] += tx[j]
#             moved_axis_y.append(coord_moved_y)
#
#         moved_axis_z = []
#         for coord in self.axisZ():
#             coord_moved_z = self.mat_vec_mul3(rot, coord)
#             for j in range(3):
#                 coord_moved_z[j] += tx[j]
#             moved_axis_z.append(coord_moved_z)
#
#         return moved_axis_x, moved_axis_y, moved_axis_z
#
#     # def translate3d(self, tx):
#     #     translated = []
#     #     for coord in self.extract_coords():
#     #         for j in range(3):
#     #             coord[j] += tx[j]
#     #         translated.append(coord)
#     #     self.set_atom_coordinates(translated)
#
#     def rotatePDB(self, degrees=90.0, axis='x'):
#         """
#         Rotate the coordinates about the given axis
#         """
#         deg = math.radians(float(degrees))
#
#         # define the rotation matrices
#         if axis == 'x':
#             rotmatrix = [[1, 0, 0], [0, math.cos(deg), -1 * math.sin(deg)], [0, math.sin(deg), math.cos(deg)]]
#         elif axis == 'y':
#             rotmatrix = [[math.cos(deg), 0, math.sin(deg)], [0, 1, 0], [-1 * math.sin(deg), 0, math.cos(deg)]]
#         elif axis == 'z':
#             rotmatrix = [[math.cos(deg), -1 * math.sin(deg), 0], [math.sin(deg), math.cos(deg), 0], [0, 0, 1]]
#         else:
#             print("Axis does not exist!")
#
#         rotated = []
#         for coord in self.extract_coords():
#             newX = coord[0] * rotmatrix[0][0] + coord[1] * rotmatrix[0][1] + coord[2] * rotmatrix[0][2]
#             newY = coord[0] * rotmatrix[1][0] + coord[1] * rotmatrix[1][1] + coord[2] * rotmatrix[1][2]
#             newZ = coord[0] * rotmatrix[2][0] + coord[1] * rotmatrix[2][1] + coord[2] * rotmatrix[2][2]
#             rotated.append([newX, newY, newZ])
#         self.set_atom_coordinates(rotated)
#
#     def rename_chains(self, chain_list_fixed):
#         lf = chain_list_fixed
#         lm = self.chain_ids[:]
#
#         l_abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
#                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
#                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4']
#
#         l_av = []
#         for e in l_abc:
#             if e not in lm:
#                 if e not in lf:
#                     l_av.append(e)
#
#         j = 0
#         for i in range(len(lm)):
#             if lm[i] in lf:
#                 lm[i] = l_av[j]
#                 j += 1
#
#         self.chain_ids = lm
#
#         prev = self.all_atoms[0].chain
#         c = 0
#         l3 = []
#         for i in range(len(self.all_atoms)):
#             if prev != self.all_atoms[i].chain:
#                 c += 1
#             l3.append(lm[c])
#             prev = self.all_atoms[i].chain
#
#         for i in range(len(self.all_atoms)):
#             self.all_atoms[i].chain = l3[i]
#
#     def rename_chain(self, chain_of_interest, new_chain):
#         for i in range(len(self.all_atoms)):
#             if self.all_atoms[i].chain == chain_of_interest:
#                 self.all_atoms[i].chain = new_chain
#
#     def AddD2Axes(self):
#         z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
#         z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")
#
#         y_axis_a = Atom(4, "CA", " ", "GLY", "8", 1, " ", 0.000, 80.000, 0.000, 1.00, 20.00, "C", "")
#         y_axis_b = Atom(5, "CA", " ", "GLY", "8", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         y_axis_c = Atom(6, "CA", " ", "GLY", "8", 3, " ", 0.000, -80.000, 0.000, 1.00, 20.00, "C", "")
#
#         x_axis_a = Atom(7, "CA", " ", "GLY", "9", 1, " ", 80.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         x_axis_b = Atom(8, "CA", " ", "GLY", "9", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         x_axis_c = Atom(9, "CA", " ", "GLY", "9", 3, " ", -80.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#
#         axes = [z_axis_a, z_axis_b, z_axis_c, y_axis_a, y_axis_b, y_axis_c, x_axis_a, x_axis_b, x_axis_c]
#
#         self.all_atoms = self.all_atoms + axes
#         self.retrieve_chain_ids()
#
#     def AddCyclicAxisZ(self):
#         z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
#         z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")
#
#         axis = [z_axis_a, z_axis_b, z_axis_c]
#
#         self.all_atoms = self.all_atoms + axis
#         self.retrieve_chain_ids()
#
#     def AddO4Folds(self):
#         # works when 3-folds are along z
#         z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.81650 * 100, 0.000 * 100, 0.57735 * 100, 1.00, 20.00, "C",
#                         "")
#         z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.81650 * -100, 0.000 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         y_axis_a = Atom(4, "CA", " ", "GLY", "8", 1, " ", -0.40824 * 100, 0.70711 * 100, 0.57735 * 100, 1.00, 20.00,
#                         "C", "")
#         y_axis_b = Atom(5, "CA", " ", "GLY", "8", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         y_axis_c = Atom(6, "CA", " ", "GLY", "8", 3, " ", -0.40824 * -100, 0.70711 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         x_axis_a = Atom(7, "CA", " ", "GLY", "9", 1, " ", -0.40824 * 100, -0.70711 * 100, 0.57735 * 100, 1.00, 20.00,
#                         "C", "")
#         x_axis_b = Atom(8, "CA", " ", "GLY", "9", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         x_axis_c = Atom(9, "CA", " ", "GLY", "9", 3, " ", -0.40824 * -100, -0.70711 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         axes = [z_axis_a, z_axis_b, z_axis_c, y_axis_a, y_axis_b, y_axis_c, x_axis_a, x_axis_b, x_axis_c]
#
#         self.all_atoms = self.all_atoms + axes
#         self.retrieve_chain_ids()
#
#     def AddT2Folds(self):
#         # works when 3-folds are along z
#         z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.81650 * 100, 0.000 * 100, 0.57735 * 100, 1.00, 20.00, "C",
#                         "")
#         z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.81650 * -100, 0.000 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         y_axis_a = Atom(4, "CA", " ", "GLY", "8", 1, " ", -0.40824 * 100, 0.70711 * 100, 0.57735 * 100, 1.00, 20.00,
#                         "C", "")
#         y_axis_b = Atom(5, "CA", " ", "GLY", "8", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         y_axis_c = Atom(6, "CA", " ", "GLY", "8", 3, " ", -0.40824 * -100, 0.70711 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         x_axis_a = Atom(7, "CA", " ", "GLY", "9", 1, " ", -0.40824 * 100, -0.70711 * 100, 0.57735 * 100, 1.00, 20.00,
#                         "C", "")
#         x_axis_b = Atom(8, "CA", " ", "GLY", "9", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
#         x_axis_c = Atom(9, "CA", " ", "GLY", "9", 3, " ", -0.40824 * -100, -0.70711 * -100, 0.57735 * -100, 1.00, 20.00,
#                         "C", "")
#
#         axes = [z_axis_a, z_axis_b, z_axis_c, y_axis_a, y_axis_b, y_axis_c, x_axis_a, x_axis_b, x_axis_c]
#
#         self.all_atoms = self.all_atoms + axes
#         self.retrieve_chain_ids()
#
#     def axisZ(self):
#         axes_list = []
#         for atom in self.all_atoms:
#             if atom.chain in ["7", "8", "9"]:
#                 axes_list.append(atom)
#         a = [axes_list[0].x, axes_list[0].y, axes_list[0].z]
#         b = [axes_list[1].x, axes_list[1].y, axes_list[1].z]
#         c = [axes_list[2].x, axes_list[2].y, axes_list[2].z]
#         return [a, b, c]
#
#     def axisY(self):
#         axes_list = []
#         for atom in self.all_atoms:
#             if atom.chain in ["7", "8", "9"]:
#                 axes_list.append(atom)
#         a = [axes_list[3].x, axes_list[3].y, axes_list[3].z]
#         b = [axes_list[4].x, axes_list[4].y, axes_list[4].z]
#         c = [axes_list[5].x, axes_list[5].y, axes_list[5].z]
#         return [a, b, c]
#
#     def axisX(self):
#         axes_list = []
#         for atom in self.all_atoms:
#             if atom.chain in ["7", "8", "9"]:
#                 axes_list.append(atom)
#         a = [axes_list[6].x, axes_list[6].y, axes_list[6].z]
#         b = [axes_list[7].x, axes_list[7].y, axes_list[7].z]
#         c = [axes_list[8].x, axes_list[8].y, axes_list[8].z]
#         return [a, b, c]
#
#     def getAxes(self):
#         axes_list = []
#         for atom in self.all_atoms:
#             if atom.chain in ["7", "8", "9"]:
#                 axes_list.append(atom)
#         return axes_list
#
#     def higestZ(self):
#         highest = self.all_atoms[0].z
#         for atom in self.all_atoms:
#             if not atom.is_axis():
#                 if atom.z > highest:
#                     highest = atom.z
#         return highest
#
#     def maxZchain(self):
#         highest = self.all_atoms[0].z
#         highest_chain = self.all_atoms[0].chain
#         for atom in self.all_atoms:
#             if not atom.is_axis():
#                 if atom.z > highest:
#                     highest = atom.z
#                     highest_chain = atom.chain
#         return highest_chain
#
#     def minZchain(self):
#         lowest = self.all_atoms[0].z
#         lowest_chain = self.all_atoms[0].chain
#         for atom in self.all_atoms:
#             if not atom.is_axis():
#                 if atom.z < lowest:
#                     lowest = atom.z
#                     lowest_chain = atom.chain
#         return lowest_chain
#
#     def higestCBZ_atom(self):
#         highest = - sys.maxint
#         highest_atom = None
#         for atom in self.all_atoms:
#             if atom.z > highest and atom.type == "CB":
#                 highest = atom.z
#                 highest_atom = atom
#         return highest_atom
#
#     def lowestZ(self):
#         lowest = self.all_atoms[0].z
#         for atom in self.all_atoms:
#             if not atom.is_axis():
#                 if atom.z < lowest:
#                     lowest = atom.z
#         return lowest
#
#     def lowestCBZ_atom(self):
#         lowest = sys.maxint
#         lowest_atom = None
#         for atom in self.all_atoms:
#             if atom.z < lowest and atom.type == "CB":
#                 lowest = atom.z
#                 lowest_atom = atom
#         return lowest_atom
#
#     def getTermCAAtom(self, term, chain_id):  # updated name to getTermCAAtom 6/01/20
#         if term == "N":
#             for atom in self.chain(chain_id):
#                 if atom.type == "CA":  # atom.is_ca()
#                     return atom
#         elif term == "C":
#             for atom in self.chain(chain_id)[::-1]:
#                 if atom.type == "CA":  # atom.is_ca()
#                     return atom
#         else:
#             print("Select N or C Term")
#             return None
#
#     def CBMinDist(self, pdb):
#         cb_distances = []
#         for atom_1 in self.all_atoms:
#             if atom_1.type == "CB":
#                 for atom_2 in pdb.all_atoms:
#                     if atom_2.type == "CB":
#                         d = atom_1.distance(atom_2, intra=True)
#                         cb_distances.append(d)
#         return min(cb_distances)
#
#     def CBMinDist_singlechain_to_all(self, self_chain_id, pdb):
#         # returns min CB-CB distance between selected chain in self and all chains in other pdb
#         cb_distances = []
#         for atom_1 in self.chain(self_chain_id):
#             if atom_1.type == "CB":
#                 for atom_2 in pdb.all_atoms:
#                     if atom_2.type == "CB":
#                         d = atom_1.distance(atom_2, intra=True)
#                         cb_distances.append(d)
#         return min(cb_distances)
#
#     def MinDist_singlechain_to_all(self, self_chain_id, pdb):
#         # returns tuple (min distance between selected chain in self and all chains in other pdb, atom1, atom2)
#         min_dist = sys.maxint
#         atom_1_min = None
#         atom_2_min = None
#         for atom_1 in self.chain(self_chain_id):
#             for atom_2 in pdb.all_atoms:
#                 d = atom_1.distance(atom_2, intra=True)
#                 if d < min_dist:
#                     min_dist = d
#                     atom_1_min = atom_1
#                     atom_2_min = atom_2
#         return (min_dist, atom_1_min, atom_2_min)
#
#     def CBMinDistSquared_singlechain_to_all(self, self_chain_id, pdb):
#         # returns min CB-CB squared distance between selected chain in self and all chains in other pdb
#         cb_distances = []
#         for atom_1 in self.chain(self_chain_id):
#             if atom_1.type == "CB":
#                 for atom_2 in pdb.all_atoms:
#                     if atom_2.type == "CB":
#                         d = atom_1.distance_squared(atom_2, intra=True)
#                         cb_distances.append(d)
#         return min(cb_distances)
#
#     def CBMinDistSquared_highestZ_to_all(self, pdb):
#         # returns min squared distance between Highest CB Z Atom in self and all CB atoms in other pdb
#         cb_distances = []
#         higestZatom = self.higestCBZ_atom()
#         for atom in pdb.all_atoms:
#             if atom.type == "CB":
#                 d = higestZatom.distance_squared(atom, intra=True)
#                 cb_distances.append(d)
#         return min(cb_distances)
#
#     def CBMinDistSquared_lowestZ_to_all(self, pdb):
#         # returns min squared distance between Lowest CB Z Atom in self and all CB atoms in other pdb
#         cb_distances = []
#         lowestZatom = self.lowestCBZ_atom()
#         for atom in pdb.all_atoms:
#             if atom.type == "CB":
#                 d = lowestZatom.distance_squared(atom, intra=True)
#                 cb_distances.append(d)
#         return min(cb_distances)
#
#     def chain_id_to_chain_index(self, chain_id):
#         try:
#             return self.chain_ids.index(chain_id)
#         except ValueError:
#             print('INVALID CHAIN')
#
#     def add_ideal_helix(self, term, chain):
#         if isinstance(chain, str):
#             chain_index = self.chain_id_to_chain_index(chain)
#         else:
#             chain_index = chain
#
#         alpha_helix_10 = [Atom(1, "N", " ", "ALA", "5", 1, " ", 27.128, 20.897, 37.943, 1.00, 0.00, "N", ""),
#                           Atom(2, "CA", " ", "ALA", "5", 1, " ", 27.933, 21.940, 38.546, 1.00, 0.00, "C", ""),
#                           Atom(3, "C", " ", "ALA", "5", 1, " ", 28.402, 22.920, 37.481, 1.00, 0.00, "C", ""),
#                           Atom(4, "O", " ", "ALA", "5", 1, " ", 28.303, 24.132, 37.663, 1.00, 0.00, "O", ""),
#                           Atom(5, "CB", " ", "ALA", "5", 1, " ", 29.162, 21.356, 39.234, 1.00, 0.00, "C", ""),
#                           Atom(6, "N", " ", "ALA", "5", 2, " ", 28.914, 22.392, 36.367, 1.00, 0.00, "N", ""),
#                           Atom(7, "CA", " ", "ALA", "5", 2, " ", 29.395, 23.219, 35.278, 1.00, 0.00, "C", ""),
#                           Atom(8, "C", " ", "ALA", "5", 2, " ", 28.286, 24.142, 34.793, 1.00, 0.00, "C", ""),
#                           Atom(9, "O", " ", "ALA", "5", 2, " ", 28.508, 25.337, 34.610, 1.00, 0.00, "O", ""),
#                           Atom(10, "CB", " ", "ALA", "5", 2, " ", 29.857, 22.365, 34.102, 1.00, 0.00, "C", ""),
#                           Atom(11, "N", " ", "ALA", "5", 3, " ", 27.092, 23.583, 34.584, 1.00, 0.00, "N", ""),
#                           Atom(12, "CA", " ", "ALA", "5", 3, " ", 25.956, 24.355, 34.121, 1.00, 0.00, "C", ""),
#                           Atom(13, "C", " ", "ALA", "5", 3, " ", 25.681, 25.505, 35.079, 1.00, 0.00, "C", ""),
#                           Atom(14, "O", " ", "ALA", "5", 3, " ", 25.488, 26.639, 34.648, 1.00, 0.00, "O", ""),
#                           Atom(15, "CB", " ", "ALA", "5", 3, " ", 24.703, 23.490, 34.038, 1.00, 0.00, "C", ""),
#                           Atom(16, "N", " ", "ALA", "5", 4, " ", 25.662, 25.208, 36.380, 1.00, 0.00, "N", ""),
#                           Atom(17, "CA", " ", "ALA", "5", 4, " ", 25.411, 26.214, 37.393, 1.00, 0.00, "C", ""),
#                           Atom(18, "C", " ", "ALA", "5", 4, " ", 26.424, 27.344, 37.270, 1.00, 0.00, "C", ""),
#                           Atom(19, "O", " ", "ALA", "5", 4, " ", 26.055, 28.516, 37.290, 1.00, 0.00, "O", ""),
#                           Atom(20, "CB", " ", "ALA", "5", 4, " ", 25.519, 25.624, 38.794, 1.00, 0.00, "C", ""),
#                           Atom(21, "N", " ", "ALA", "5", 5, " ", 27.704, 26.987, 37.142, 1.00, 0.00, "N", ""),
#                           Atom(22, "CA", " ", "ALA", "5", 5, " ", 28.764, 27.968, 37.016, 1.00, 0.00, "C", ""),
#                           Atom(23, "C", " ", "ALA", "5", 5, " ", 28.497, 28.876, 35.825, 1.00, 0.00, "C", ""),
#                           Atom(24, "O", " ", "ALA", "5", 5, " ", 28.602, 30.096, 35.937, 1.00, 0.00, "O", ""),
#                           Atom(25, "CB", " ", "ALA", "5", 5, " ", 30.115, 27.292, 36.812, 1.00, 0.00, "C", ""),
#                           Atom(26, "N", " ", "ALA", "5", 6, " ", 28.151, 28.278, 34.682, 1.00, 0.00, "N", ""),
#                           Atom(27, "CA", " ", "ALA", "5", 6, " ", 27.871, 29.032, 33.478, 1.00, 0.00, "C", ""),
#                           Atom(28, "C", " ", "ALA", "5", 6, " ", 26.759, 30.040, 33.737, 1.00, 0.00, "C", ""),
#                           Atom(29, "O", " ", "ALA", "5", 6, " ", 26.876, 31.205, 33.367, 1.00, 0.00, "O", ""),
#                           Atom(30, "CB", " ", "ALA", "5", 6, " ", 27.429, 28.113, 32.344, 1.00, 0.00, "C", ""),
#                           Atom(31, "N", " ", "ALA", "5", 7, " ", 25.678, 29.586, 34.376, 1.00, 0.00, "N", ""),
#                           Atom(32, "CA", " ", "ALA", "5", 7, " ", 24.552, 30.444, 34.682, 1.00, 0.00, "C", ""),
#                           Atom(33, "C", " ", "ALA", "5", 7, " ", 25.013, 31.637, 35.507, 1.00, 0.00, "C", ""),
#                           Atom(34, "O", " ", "ALA", "5", 7, " ", 24.652, 32.773, 35.212, 1.00, 0.00, "O", ""),
#                           Atom(35, "CB", " ", "ALA", "5", 7, " ", 23.489, 29.693, 35.478, 1.00, 0.00, "C", ""),
#                           Atom(36, "N", " ", "ALA", "5", 8, " ", 25.814, 31.374, 36.543, 1.00, 0.00, "N", ""),
#                           Atom(37, "CA", " ", "ALA", "5", 8, " ", 26.321, 32.423, 37.405, 1.00, 0.00, "C", ""),
#                           Atom(38, "C", " ", "ALA", "5", 8, " ", 27.081, 33.454, 36.583, 1.00, 0.00, "C", ""),
#                           Atom(39, "O", " ", "ALA", "5", 8, " ", 26.874, 34.654, 36.745, 1.00, 0.00, "O", ""),
#                           Atom(40, "CB", " ", "ALA", "5", 8, " ", 25.581, 31.506, 36.435, 1.00, 0.00, "C", ""),
#                           Atom(41, "N", " ", "ALA", "5", 9, " ", 27.963, 32.980, 35.700, 1.00, 0.00, "N", ""),
#                           Atom(42, "CA", " ", "ALA", "5", 9, " ", 28.750, 33.859, 34.858, 1.00, 0.00, "C", ""),
#                           Atom(43, "C", " ", "ALA", "5", 9, " ", 27.834, 34.759, 34.042, 1.00, 0.00, "C", ""),
#                           Atom(44, "O", " ", "ALA", "5", 9, " ", 28.052, 35.967, 33.969, 1.00, 0.00, "O", ""),
#                           Atom(45, "CB", " ", "ALA", "5", 9, " ", 29.621, 33.061, 33.894, 1.00, 0.00, "C", ""),
#                           Atom(46, "N", " ", "ALA", "5", 10, " ", 26.807, 34.168, 33.427, 1.00, 0.00, "N", ""),
#                           Atom(47, "CA", " ", "ALA", "5", 10, " ", 25.864, 34.915, 32.620, 1.00, 0.00, "C", ""),
#                           Atom(48, "C", " ", "ALA", "5", 10, " ", 25.230, 36.024, 33.448, 1.00, 0.00, "C", ""),
#                           Atom(49, "O", " ", "ALA", "5", 10, " ", 25.146, 37.165, 33.001, 1.00, 0.00, "O", ""),
#                           Atom(50, "CB", " ", "ALA", "5", 10, " ", 24.752, 34.012, 32.097, 1.00, 0.00, "C", ""),
#                           Atom(51, "N", " ", "ALA", "5", 11, " ", 24.783, 35.683, 34.660, 1.00, 0.00, "N", ""),
#                           Atom(52, "CA", " ", "ALA", "5", 11, " ", 24.160, 36.646, 35.544, 1.00, 0.00, "C", ""),
#                           Atom(53, "C", " ", "ALA", "5", 11, " ", 25.104, 37.812, 35.797, 1.00, 0.00, "C", ""),
#                           Atom(54, "O", " ", "ALA", "5", 11, " ", 24.699, 38.970, 35.714, 1.00, 0.00, "O", ""),
#                           Atom(55, "CB", " ", "ALA", "5", 11, " ", 23.810, 36.012, 36.887, 1.00, 0.00, "C", ""),
#                           Atom(56, "N", " ", "ALA", "5", 12, " ", 26.365, 37.503, 36.107, 1.00, 0.00, "N", ""),
#                           Atom(57, "CA", " ", "ALA", "5", 12, " ", 27.361, 38.522, 36.370, 1.00, 0.00, "C", ""),
#                           Atom(58, "C", " ", "ALA", "5", 12, " ", 27.477, 39.461, 35.177, 1.00, 0.00, "C", ""),
#                           Atom(59, "O", " ", "ALA", "5", 12, " ", 27.485, 40.679, 35.342, 1.00, 0.00, "O", ""),
#                           Atom(60, "CB", " ", "ALA", "5", 12, " ", 28.730, 37.900, 36.625, 1.00, 0.00, "C", ""),
#                           Atom(61, "N", " ", "ALA", "5", 13, " ", 27.566, 38.890, 33.974, 1.00, 0.00, "N", ""),
#                           Atom(62, "CA", " ", "ALA", "5", 13, " ", 27.680, 39.674, 32.761, 1.00, 0.00, "C", ""),
#                           Atom(63, "C", " ", "ALA", "5", 13, " ", 26.504, 40.634, 32.645, 1.00, 0.00, "C", ""),
#                           Atom(64, "O", " ", "ALA", "5", 13, " ", 26.690, 41.815, 32.360, 1.00, 0.00, "O", ""),
#                           Atom(65, "CB", " ", "ALA", "5", 13, " ", 27.690, 38.779, 31.527, 1.00, 0.00, "C", ""),
#                           Atom(66, "N", " ", "ALA", "5", 14, " ", 25.291, 40.121, 32.868, 1.00, 0.00, "N", ""),
#                           Atom(67, "CA", " ", "ALA", "5", 14, " ", 24.093, 40.932, 32.789, 1.00, 0.00, "C", ""),
#                           Atom(68, "C", " ", "ALA", "5", 14, " ", 24.193, 42.112, 33.745, 1.00, 0.00, "C", ""),
#                           Atom(69, "O", " ", "ALA", "5", 14, " ", 23.905, 43.245, 33.367, 1.00, 0.00, "O", ""),
#                           Atom(70, "CB", " ", "ALA", "5", 14, " ", 22.856, 40.120, 33.158, 1.00, 0.00, "C", ""),
#                           Atom(71, "N", " ", "ALA", "5", 15, " ", 24.604, 41.841, 34.986, 1.00, 0.00, "N", ""),
#                           Atom(72, "CA", " ", "ALA", "5", 15, " ", 24.742, 42.878, 35.989, 1.00, 0.00, "C", ""),
#                           Atom(73, "C", " ", "ALA", "5", 15, " ", 25.691, 43.960, 35.497, 1.00, 0.00, "C", ""),
#                           Atom(74, "O", " ", "ALA", "5", 15, " ", 25.390, 45.147, 35.602, 1.00, 0.00, "O", ""),
#                           Atom(75, "CB", " ", "ALA", "5", 15, " ", 24.418, 41.969, 34.808, 1.00, 0.00, "C", "")]
#
#         alpha_helix_10_pdb = PDB()
#         alpha_helix_10_pdb.read_atom_list(alpha_helix_10)
#
#         if term == "N":
#             first_residue_number = self.chain(self.chain_ids[chain_index])[0].residue_number
#             fixed_coords = self.extract_coords_subset(first_residue_number, first_residue_number + 4, chain_index,
#                                                       True)
#             moving_coords = alpha_helix_10_pdb.extract_coords_subset(11, 15, 0, True)
#             helix_overlap = PDBOverlap(fixed_coords, moving_coords)
#             rot, tx, rmsd, coords_moved = helix_overlap.overlap()
#             alpha_helix_10_pdb.apply(rot, tx)
#
#             # rename alpha helix chain
#             for atom in alpha_helix_10_pdb.all_atoms:
#                 atom.chain = self.chain_ids[chain_index]
#
#             # renumber residues in concerned chain
#             if first_residue_number > 10:
#                 shift = -(first_residue_number - 11)
#             else:
#                 shift = 11 - first_residue_number
#
#             for atom in self.all_atoms:
#                 if atom.chain == self.chain_ids[chain_index]:
#                     atom.residue_number = atom.residue_number + shift
#
#             # only keep non overlapping atoms in helix
#             helix_to_add = []
#             for atom in alpha_helix_10_pdb.all_atoms:
#                 if atom.residue_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#                     helix_to_add.append(atom)
#
#             # create a helix-chain atom list
#             chain_and_helix = helix_to_add + self.chain(self.chain_ids[chain_index])
#
#             # place chain_and_helix atoms in the same order as in original PDB file
#             ordered_atom_list = []
#             for chain_id in self.chain_ids:
#                 if chain_id != self.chain_ids[chain_index]:
#                     ordered_atom_list = ordered_atom_list + self.chain(chain_id)
#                 else:
#                     ordered_atom_list = ordered_atom_list + chain_and_helix
#
#             # renumber all atoms in PDB
#             atom_number = 1
#             for atom in ordered_atom_list:
#                 atom.number = atom_number
#                 atom_number = atom_number + 1
#
#             self.all_atoms = ordered_atom_list
#
#         elif term == "C":
#             last_residue_number = self.chain(self.chain_ids[chain_index])[-1].residue_number
#             fixed_coords = self.extract_coords_subset(last_residue_number - 4, last_residue_number, chain_index, True)
#             moving_coords = alpha_helix_10_pdb.extract_coords_subset(1, 5, 0, True)
#             helix_overlap = PDBOverlap(fixed_coords, moving_coords)
#             rot, tx, rmsd, coords_moved = helix_overlap.overlap()
#             alpha_helix_10_pdb.apply(rot, tx)
#
#             # rename alpha helix chain
#             for atom in alpha_helix_10_pdb.all_atoms:
#                 atom.chain = self.chain_ids[chain_index]
#
#             # only keep non overlapping atoms in helix
#             helix_to_add = []
#             for atom in alpha_helix_10_pdb.all_atoms:
#                 if atom.residue_number in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
#                     helix_to_add.append(atom)
#
#             # renumber residues in helix
#             shift = last_residue_number - 5
#             for atom in helix_to_add:
#                 atom.residue_number = atom.residue_number + shift
#
#             # create a helix-chain atom list
#             chain_and_helix = self.chain(self.chain_ids[chain_index]) + helix_to_add
#
#             # place chain_and_helix atoms in the same order as in original PDB file
#             ordered_atom_list = []
#             for chain_id in self.chain_ids:
#                 if chain_id != self.chain_ids[chain_index]:
#                     ordered_atom_list = ordered_atom_list + self.chain(chain_id)
#                 else:
#                     ordered_atom_list = ordered_atom_list + chain_and_helix
#
#             # renumber all atoms in PDB
#             atom_number = 1
#             for atom in ordered_atom_list:
#                 atom.number = atom_number
#                 atom_number = atom_number + 1
#
#             self.all_atoms = ordered_atom_list
#
#         else:
#             print("Select N or C Terminus")
#
#     def get_all_residues(self, chain_id):  # added get_all_ to function name 6/01/20
#         current_residue_number = self.chain(chain_id)[0].residue_number
#         current_residue = []
#         all_residues = []
#         for atom in self.chain(chain_id):
#             if atom.residue_number == current_residue_number:
#                 current_residue.append(atom)
#             else:
#                 all_residues.append(current_residue)
#                 current_residue = []
#                 current_residue.append(atom)
#                 current_residue_number = atom.residue_number
#         all_residues.append(current_residue)
#         return all_residues
#
#     def getResidueAtoms(self, residue_chain_id, residue_number):
#         residue_atoms = []
#         for atom in self.all_atoms:
#             if atom.chain == residue_chain_id and atom.residue_number == residue_number:
#                 residue_atoms.append(atom)
#         return residue_atoms
#
#     def write(self, out_path):
#         outfile = open(out_path, "w")
#         for atom in self.all_atoms:
#             outfile.write(str(atom))
#         outfile.close()
#
#     def adjust_rotZ_to_parallel(self, axis1, axis2, rotate_half=False):
#         def length(vec):
#             length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
#             return length
#
#         def cos_angle(vec1, vec2):
#             length_1 = length(vec1)
#             length_2 = length(vec2)
#             if length_1 != 0 and length_2 != 0:
#                 cosangle = (vec1[0] / length_1) * (vec2[0] / length_2) + (vec1[1] / length_1) * (vec2[1] / length_2) + (vec1[2] / length_1) * (vec2[2] / length_2)
#                 return cosangle
#             else:
#                 return 0
#
#         def angle(vec1, vec2):
#             angle = (math.acos(abs(cos_angle(vec1, vec2))) * 180) / math.pi
#             return angle
#
#         vec1 = [axis1[2][0] - axis1[0][0], axis1[2][1] - axis1[0][1], axis1[2][2] - axis1[0][2]]
#         vec2 = [axis2[2][0] - axis2[0][0], axis2[2][1] - axis2[0][1], axis2[2][2] - axis2[0][2]]
#
#         corrected_vec1 = [vec1[0], vec1[1], 0]
#         corrected_vec2 = [vec2[0], vec2[1], 0]
#
#         crossproduct = [corrected_vec1[1] * corrected_vec2[2] - corrected_vec1[2] * corrected_vec2[1], corrected_vec1[2] * corrected_vec2[0] - corrected_vec1[0] * corrected_vec2[2], corrected_vec1[0] * corrected_vec2[1] - corrected_vec1[1] * corrected_vec2[0]]
#         dotproduct_of_crossproduct_and_z_axis = crossproduct[0] * 0 + crossproduct[1] * 0 + crossproduct[2] * 1
#
#         #print(angle(corrected_vec1, corrected_vec2))
#
#         if rotate_half is False:
#             if dotproduct_of_crossproduct_and_z_axis < 0 and angle(corrected_vec1, corrected_vec2) <= 10:
#                 self.rotatePDB(angle(corrected_vec1, corrected_vec2), "z")
#             else:
#                 self.rotatePDB(-angle(corrected_vec1, corrected_vec2), "z")
#         else:
#             if dotproduct_of_crossproduct_and_z_axis < 0 and angle(corrected_vec1, corrected_vec2) <= 10:
#                 self.rotatePDB(angle(corrected_vec1, corrected_vec2) / 2, "z")
#             else:
#                 self.rotatePDB(-angle(corrected_vec1, corrected_vec2) / 2, "z")


class PDBOverlap:
    def __init__(self, coords_fixed, coords_moving):
        self.coords_fixed = coords_fixed
        self.coords_moving = coords_moving

    def vdot3(self, a, b):
        dot = 0.
        for i in range(3):
            dot += a[i] * b[i]
        return dot

    def vnorm3(self, a):
        b = [0., 0., 0.]
        dot = 0.
        for i in range(3):
            dot += a[i] * a[i]
        for i in range(3):
            b[i] = a[i] / math.sqrt(dot)
        return b

    def vcross(self, a, b):
        c = [0., 0., 0.]
        for i in range(3):
            c[i] = a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3]
        return c

    def inv3(self, a):
        ainv = [[0. for j in range(3)] for i in range(3)]
        det = 0.
        for i in range(3):
            det += a[(i + 0) % 3][0] * a[(i + 1) % 3][1] * a[(i + 2) % 3][2]
            det -= a[(i + 0) % 3][2] * a[(i + 1) % 3][1] * a[(i + 2) % 3][0]

        for i in range(3):
            for j in range(3):
                ainv[j][i] = (a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3] - a[(i + 2) % 3][(j + 1) % 3] *
                              a[(i + 1) % 3][(j + 2) % 3]) / det
        return ainv

    def mat_vec_mul3(self, a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]
        return c

    def apply(self, rot, tx, moving):
        moved = []
        for coord in moving:
            coord_moved = self.mat_vec_mul3(rot, coord)
            for j in range(3):
                coord_moved[j] += tx[j]
            moved.append(coord_moved)
        return moved

    def get_rmsd(self, moving, fixed):
        n = len(moving)
        rmsd = 0.
        for i in range(n):
            for j in range(3):
                rmsd += (moving[i][j] - fixed[i][j]) ** 2
        rmsd = math.sqrt(rmsd / n)
        return rmsd

    def overlap(self):
        n = len(self.coords_moving)
        m = len(self.coords_fixed)
        if (n != m):
            print("Length of matching coordinates must match!")
        #print("# of atoms for overlapping = ", n)

        # calculate centers of mass
        else:
            cm_fixed = [0. for j in range(3)]
            cm_moving = [0. for j in range(3)]
            n = len(self.coords_fixed)
            for i in range(n):
                # print(coords_fixed[i][0], coords_fixed[i][1], coords_fixed[i][2])
                for j in range(3):
                    cm_fixed[j] = cm_fixed[j] + self.coords_fixed[i][j]
            for j in range(3):
                cm_fixed[j] = cm_fixed[j] / n
                # print(cm_fixed)
            n = len(self.coords_moving)
            for i in range(n):
                # print(coords_moving[i][0], coords_moving[i][1], coords_moving[i][2])
                for j in range(3):
                    cm_moving[j] = cm_moving[j] + self.coords_moving[i][j]
            for j in range(3):
                cm_moving[j] = cm_moving[j] / n
                # print(cm_moving)

            # form 3x3 matrices as the sums of outer products
            f_m_mat = [[0. for j in range(3)] for i in range(3)]
            m_m_mat = [[0. for j in range(3)] for i in range(3)]
            for k in range(n):
                for i in range(3):
                    for j in range(3):
                        f_m_mat[i][j] += (self.coords_fixed[k][i] - cm_fixed[i]) * (self.coords_moving[k][j] - cm_moving[j])
                        m_m_mat[i][j] += (self.coords_moving[k][i] - cm_moving[i]) * (self.coords_moving[k][j] - cm_moving[j])

            # solve for best transformation matrix (which could include a stretch)
            m_m_inv = self.inv3(m_m_mat)
            rot_mat = [[0. for j in range(3)] for i in range(3)]
            rot = [[0. for j in range(3)] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        rot_mat[i][j] = rot_mat[i][j] + f_m_mat[i][k] * m_m_inv[k][j]

            # extract rotation part from transformation matrix
            rot[0] = self.vnorm3(rot_mat[0])
            dotp = self.vdot3(rot_mat[1], rot[0])
            for i in range(3):
                rot[1][i] = rot_mat[1][i] - dotp * rot[0][i]
            rot[1] = self.vnorm3(rot[1])
            rot[2] = self.vcross(rot[0], rot[1])

            # for translational part of transformation from rot mat and centers of mass
            tx = self.mat_vec_mul3(rot, cm_moving)
            for i in range(3):
                tx[i] = cm_fixed[i] - tx[i]

            # apply transformation to moving coordinates
            coords_moved = self.apply(rot, tx, self.coords_moving)

            # calculate rmsd between moved coordinates and fixed coordinates
            rmsd = self.get_rmsd(coords_moved, self.coords_fixed)

            return rot, tx, rmsd, coords_moved


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


# class Stride:
#     def __init__(self, pdbfilepath):
#         self.pdbfilepath = pdbfilepath
#         self.ss_asg = []
#
#     def run(self):
#         try:
#             with open(os.devnull, 'w') as devnull:
#                 stride_out = subprocess.check_output(['./helix_fusion_tool/stride', '%s' %self.pdbfilepath, '-cA'], stderr=devnull)
#
#         except:
#             stride_out = None
#
#         if stride_out is not None:
#             lines = stride_out.split('\n')
#             for line in lines:
#                 if line[0:3] == "ASG":
#                     self.ss_asg.append((int(filter(str.isdigit, line[10:15].strip())), line[24:25]))
#
#     def is_n_term_helical(self):
#         if len(self.ss_asg) >= 10:
#             for i in range(5):
#                 temp_window = ''.join([self.ss_asg[0+i:5+i][j][1] for j in range(5)])
#                 res_number = self.ss_asg[0+i:5+i][0][0]
#                 if "HHHHH" in temp_window:
#                     return True, res_number
#         return False, None
#
#     def is_c_term_helical(self):
#         if len(self.ss_asg) >= 10:
#             for i in range(5):
#                 reverse_ss_asg = self.ss_asg[::-1]
#                 temp_window = ''.join([reverse_ss_asg[0+i:5+i][j][1] for j in range(5)])
#                 res_number = reverse_ss_asg[0+i:5+i][4][0]
#                 if "HHHHH" in temp_window:
#                     return True, res_number
#         return False, None


class Orient:
    def __init__(self, pdblist):
        self.pdblist = pdblist

    def run(self, symm):
        for pdb in self.pdblist:
            if os.path.exists(pdb):
                os.system('cp %s input.pdb' %pdb)
                os.system('./helix_fusion_tool/orient_oligomer >> orient.out 2>&1 << eof\n./helix_fusion_tool/%s_symm.txt\neof' %symm)
                os.system('mv output.pdb %s_orient.pdb' %os.path.splitext(pdb)[0])
            os.system('find -type f -size 0 -delete')
            os.system('rm input.pdb')


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
        if self.orient_target:
            if os.path.exists(self.target_protein_path):
                print('Orienting Target Molecule')
                orient_target = Orient([self.target_protein_path])
                orient_target.run(self.targetprotein_symm)
            else:
                print('Could Not Find Target PDB File')
                return -1

        # Read in Fixed PDB file or Oriented PDB File
        if self.orient_target:
            orient_target_path = os.path.splitext(self.target_protein_path)[0] + '_orient.pdb'
            if os.path.exists(orient_target_path):
                print('Done Orienting Target Molecule')
                target_protein = PDB.from_file(orient_target_path)
            else:
                print('Could Not Orient Target Molecule')
                return -1
        else:
            target_protein = PDB.from_file(self.target_protein_path)

        # Add Ideal 10 Ala Helix to Target if desired
        if self.add_target_helix[0]:
            print('Adding Ideal Ala Helix to Target Molecule')
            target_protein.add_ideal_helix(self.add_target_helix[1], self.add_target_helix[2])
            if self.add_target_helix[1] == 'N':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[0].residue_number
            elif self.add_target_helix[1] == 'C':
                target_term_resi = target_protein.chain(target_protein.chain_ids[self.add_target_helix[2]])[-1].residue_number - 9
            else:
                print('Select N or C Terminus for Target Molecule')
                return -1
            print("Done Adding Ideal Ala Helix to Target Molecule")

        # Run Stride On Target Protein
        # else:
        #     if self.targetprotein_term == "N" or self.targetprotein_term == "C":
        #         stride_target = Stride(self.target_protein_path)
        #         stride_target.run()
        #         if len(stride_target.ss_asg) > 0:
        #             if self.targetprotein_term == "N":
        #                 target_term_resi = stride_target.is_n_term_helical()[1]
        #                 print("Done Running Stride On Target Molecule")
        #             else:
        #                 target_term_resi = stride_target.is_c_term_helical()[1]
        #                 print("Done Running Stride On Target Molecule")
        #         else:
        #             print("Error Running Stride On Target Molecule")
        #             return -1
        #     else:
        #         print("Select N or C Terminus for Target Molecule")
        #         return -1

        else:
            target_term_resi = self.add_target_helix[1]

        # Add Axis / Axes to Target Molecule
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
            oligomer_filepath = os.path.join(self.work_dir, '%s.pdb1' % oligomer_id)
            correct_oligomer_state = PDB.from_file(oligomer_filepath)
            correct_sate_out_path = os.path.splitext(oligomer_filepath)[0] + '.pdb'
            correct_sate_out = open(correct_sate_out_path, 'w')
            for atom in correct_oligomer_state.all_atoms:
                correct_sate_out.write(str(atom))
            correct_sate_out.close()

        # Orient Oligomers
        correct_state_oligomer_filepath_list = []
        for oligomer_id in oligomer_id_list:
            correct_state_oligomer_filepath = os.path.join(self.work_dir, '%s.pdb' % oligomer_id)
            correct_state_oligomer_filepath_list.append(correct_state_oligomer_filepath)
        print('Orienting Oligomers')
        orient_oligomers = Orient(correct_state_oligomer_filepath_list)
        orient_oligomers.run(self.oligomer_symm)
        print('Done Orienting Oligomers')

        print('Fusing Target To Oligomers')
        for oligomer_id in oligomer_id_list:
            oriented_oligomer_filepath = os.path.join(self.work_dir, '%s_orient.pdb' % oligomer_id)
            if os.path.isfile(oriented_oligomer_filepath):
                for i in range(6):
                    # Read in Moving PDB
                    pdb_oligomer = PDB.from_file(oriented_oligomer_filepath)

                    # Run Stride On Oligomer
                    if self.oligomer_term in ['N', 'C']:
                        raise RuntimeError('Need to rework Stride execution here')
                        # stride_oligomer = Stride(oriented_oligomer_filepath)
                        # stride_oligomer.run()
                        if self.oligomer_term == 'N':
                            oligomer_term_resi = stride_oligomer.is_n_term_helical()[1]
                        elif self.oligomer_term == 'C':
                            oligomer_term_resi = stride_oligomer.is_c_term_helical()[1]
                    else:
                        print('Select N or C Terminus For Oligomer')
                        return -1

                    if type(oligomer_term_resi) is int:
                        # Add Axis / Axes to Oligomers
                        if self.oligomer_symm[:1] == 'C':
                            pdb_oligomer.AddCyclicAxisZ()
                        elif self.targetprotein_symm == 'D2':
                            pdb_oligomer.AddD2Axes()
                        else:
                            print('Oligomer Symmetry Not Supported')
                            return -1

                        # Extract coordinates of segment to be overlapped from PDB Fixed
                        pdb_fixed_coords = target_protein.chain(self.add_target_helix[2]).get_coords_subset(target_term_resi + i, target_term_resi + 4 + i)
                        # Extract coordinates of segment to be overlapped from PDB Moving
                        pdb_moble_coords = pdb_oligomer.chains[0].get_coords_subset(oligomer_term_resi, oligomer_term_resi + 4)

                        # Create PDBOverlap instance
                        # pdb_overlap = PDBOverlap(pdb_fixed_coords, pdb_moble_coords)
                        rmsd, rot, tx, _ = superposition3d(pdb_fixed_coords, pdb_moble_coords)

                        # if pdb_overlap.overlap() != 'lengths mismatch':
                        #     # Calculate Optimal (rot, tx, rmsd, coords_moved)
                        #     rot, tx, rmsd, coords_moved = pdb_overlap.overlap()

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
                                pdb_oligomer.reorder_chains(exclude_chains=target_protein.chain_ids)

                                out_pdb = PDB.from_atoms(target_protein.atoms + pdb_oligomer.atoms)

                                out_path = os.path.join(design_directory,
                                                        '%s_%s_%d.pdb' % (os.path.basename(self.target_protein_path)[0:4], oligomer_id, i))
                                out_pdb.write(out_path=out_path)

        print('Done')


def align(pdb1_path, start_1, end_1, chain_1, pdb2_path, start_2, end_2, chain_2, extend_helix=False):
        pdb1 = PDB.from_file(pdb1_path)
        pdb2 = PDB.from_file(pdb2_path)

        if extend_helix:
            n_terminus = pdb1.chain(chain_1).n_terminal_residue.number
            if n_terminus in range(start_1, end_1) or n_terminus < start_1:
                term = 'N'
            else:
                term = 'C'
            print('Adding ideal helix to %s-terminus of reference molecule' % term)
            pdb1.add_ideal_helix(term, chain_1)  # terminus, chain number
        coords1 = pdb1.chain(chain_1).get_coords_subset(start_1, end_1)
        coords2 = pdb2.chain(chain_2).get_coords_subset(start_2, end_2)

        rmsd, rot, tx, _ = superposition3d(coords1, coords2)
        pdb2.transform(rot, tx)

        return pdb2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=os.path.basename(__file__) +
                                     '\nTool for aligning terminal helices of two proteins')
    parser.add_argument('-r', '--reference_pdb', type=str, help='The disk location of pdb file to serve as reference')
    parser.add_argument('-s', '--ref_start_res', type=int, help='First residue in a range to serve as reference for '
                                                                'alignment')
    parser.add_argument('-e', '--ref_end_res', type=int, help='Last residue to serve as reference for alignment')
    parser.add_argument('-c', '--ref_chain', help='Chain ID of the reference moleulce, Default=A', default='A')
    parser.add_argument('-a', '--aligned_pdb', type=str, help='The disk location of pdb file to be aligned to the '
                                                              'reference')
    parser.add_argument('-as', '--align_start_res', type=int, help='First residue to align to reference')
    parser.add_argument('-ae', '--align_end_res', type=int, help='Last residue to align to reference')
    parser.add_argument('-ac', '--aligned_chain', help='Chain Id of the moving molecule')
    parser.add_argument('-x', '--extend_helical_termini', action='store_true',
                        help='Whether to extend the termini in question with an ideal 10 residue alpha helix. All '
                             'residue ranges will be modified accordingly. '
                             'Ex. --extend_helical_termini --ref_start_res 1 --ref_end_res 9 '
                             'will insert a 10 residue alpha helix to the reference range and perform alignment from '
                             'residue 1-9 of the extended alpha helix. Default=False')
    parser.add_argument('-o', '--out_file_path', type=str, help='The disk location of file containing a pdb to be moved')
    args = parser.parse_args()

    aligned_pdb = align(args.reference_pdb, args.ref_start_res, args.ref_end_res, args.ref_chain,
                        args.aligned_pdb, args.align_start_res, args.align_end_res, args.aligned_chain,
                        extend_helix=args.extend_helical_termini)
    aligned_pdb.write(args.out_file_path)
