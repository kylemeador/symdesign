#!/home/kmeador/miniconda3/bin/python
import copy
import math
import os
import subprocess
from collections.abc import Iterable
from copy import copy, deepcopy
from itertools import repeat
from shutil import move

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import IUPACData
# from Bio.Alphabet import IUPAC
from sklearn.neighbors import BallTree

from PathUtils import free_sasa_exe_path, stride_exe_path, scout_symmdef, make_symmdef, orient_exe_path, \
    orient_log_file, orient_dir
from Query.PDB import get_pdb_info_by_entry, retrieve_entity_id_by_sequence
from SequenceProfile import write_fasta
from Stride import Stride
from Structure import Structure, Chain, Atom, Coords, Entity
from SymDesignUtils import remove_duplicates, start_log  # logger

logger = start_log(name=__name__, level=2)  # was from SDUtils logger, but moved here per standard suggestion


class PDB(Structure):
    """The base object for PDB file manipulation"""
    available_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 'abcdefghijklmnopqrstuvwyz0123456789~!@#$%^&*()-+={}[]|:;<>?'

    def __init__(self, file=None, atoms=None, metadata=None, log=None, no_entities=None, **kwargs):
        if not log:
            log = start_log()
        super().__init__(log=log, **kwargs)  #
        self.log.debug('Initializing PDB')
        # if log:  # from Super Structure
        #     self.log = log
        # else:
        #     self.log = start_log()

        # self.accession_entity_map = {}
        # self.atoms = []  # captured from Structure
        self.api_entry = None
        # {'entity': {1: {'A', 'B'}, ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
        #  'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
        self.atom_sequences = {}  # ATOM record sequence - {chain: 'AGHKLAIDL'}
        self.bb_coords = []
        self.cb_coords = []
        # self.center_of_mass = None  # captured from Structure
        self.chain_id_list = []  # unique chain IDs in PDB Todo refactor
        self.chains = []
        # self.coords = []  # captured from Structure
        self.cryst = None  # {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
        self.cryst_record = None
        self.dbref = {}  # {'chain': {'db: 'UNP', 'accession': P12345}, ...}
        self.design = False  # assume not a design unless explicitly found to be a design
        self.entities = []
        self.entity_d = {}  # {1: {'chains': [Chain objs], 'seq': 'GHIPLF...', 'representative': 'A'}
        # ^ ZERO-indexed for recap project!!!
        # self.entity_accession_map = {}
        self.filepath = None  # PDB filepath if instance is read from PDB file
        self.header = []
        # self.name = None  # captured from Structure
        self.profile = {}
        self.reference_aa = None
        self.resolution = None
        # self.residues = []  # captured from Structure
        self.rotation_d = {}
        # self.secondary_structure = None  # captured from Structure
        self.reference_sequence = {}  # SEQRES or PDB API entries. key is chainID, value is 'AGHKLAIDL'
        self.space_group = None
        self.uc_dimensions = []
        self.max_symmetry = None
        self.dihedral_chain = None

        if file:
            self.readfile(file, no_entities=no_entities)
        if atoms:
            self.set_atoms(atoms)  # sets all atoms and residues in PDB
            self.chain_id_list = remove_duplicates([residue.chain for residue in self.residues])
            if metadata and isinstance(metadata, PDB):
                self.copy_metadata(metadata)
            self.process_pdb(coords=[atom.coords for atom in atoms])

    @classmethod
    def from_file(cls, file, **kwargs):
        return cls(file=file, **kwargs)

    # @classmethod
    # def from_entities(cls, entities):  # Todo
    #     return cls()

    @property
    def number_of_chains(self):
        return len(self.chain_id_list)

    @property
    def symmetry(self):
        return {'symmetry': self.space_group, 'uc_dimensions': self.uc_dimensions, 'cryst_record': self.cryst_record,
                'cryst': self.cryst, 'max_symmetry': self.max_symmetry}

    def set_chain_id_list(self, chain_id_list):
        self.chain_id_list = chain_id_list

    def get_chain_id_list(self):
        return self.chain_id_list

    def set_filepath(self, filepath):
        self.filepath = filepath

    def get_filepath(self):
        return self.filepath

    def get_uc_dimensions(self):
        return list(self.cryst['a_b_c']) + list(self.cryst['ang_a_b_c'])

    def copy_metadata(self, other_pdb):
        temp_metadata = {'api_entry': other_pdb.__dict__['api_entry'],
                         'cryst_record': other_pdb.__dict__['cryst_record'],
                         'cryst': other_pdb.__dict__['cryst'],
                         'design': other_pdb.__dict__['design'],
                         'entity_d': other_pdb.__dict__['entity_d'],  # Todo
                         'name': other_pdb.__dict__['name'],
                         'space_group': other_pdb.__dict__['space_group'],
                         'uc_dimensions': other_pdb.__dict__['uc_dimensions']}
        # temp_metadata = copy(other_pdb.__dict__)
        # temp_metadata.pop('atoms')
        # temp_metadata.pop('residues')
        # temp_metadata.pop('secondary_structure')
        # temp_metadata.pop('number_of_atoms')
        # temp_metadata.pop('number_of_residues')
        self.__dict__.update(temp_metadata)

    def update_attributes_from_pdb(self, pdb):  # Todo copy full attribute dict without selected elements
        # self.atoms = pdb.atoms
        self.resolution = pdb.resolution
        self.cryst_record = pdb.cryst_record
        self.cryst = pdb.cryst
        self.dbref = pdb.dbref
        self.design = pdb.design
        self.header = pdb.header
        self.reference_sequence = pdb.reference_sequence
        # self.atom_sequences = pdb.atom_sequences
        self.filepath = pdb.filepath
        # self.chain_id_list = pdb.chain_id_list
        self.entity_d = pdb.entity_d
        self.name = pdb.name
        self.secondary_structure = pdb.secondary_structure
        self.cb_coords = pdb.cb_coords
        self.bb_coords = pdb.bb_coords

    def readfile(self, filepath, remove_alt_location=True, **kwargs):
        """Reads .pdb file and feeds PDB instance"""
        self.filepath = filepath
        formatted_filename = os.path.splitext(os.path.basename(filepath))[0].rstrip('pdb').lstrip('pdb')
        underscore_idx = formatted_filename.rfind('_') if formatted_filename.rfind('_') != 1 else None
        self.name = formatted_filename[:underscore_idx]

        with open(self.filepath, 'r') as f:
            pdb_lines = f.readlines()
            pdb_lines = list(map(str.rstrip, pdb_lines))
        available_chain_ids = [sec + first for sec in [''] + [PDB.available_letters] for first in PDB.available_letters]
        chain_ids = []
        seq_res_lines = []
        multimodel, start_of_new_model = False, False
        model_chain_id, curr_chain_id = None, None
        model_chain_index = 0
        entity = None
        atom_idx = 0
        coords, atom_info = [], []
        for line in pdb_lines:
            if line[0:4] == 'ATOM' or line[17:20] == 'MSE' and line[0:6] == 'HETATM':  # KM modified 2/10/20 for MSE
                number = int(line[6:11].strip())
                atom_type = line[12:16].strip()
                alt_location = line[16:17].strip()
                if line[17:20] == 'MSE':  # KM added 2/10/20
                    residue_type = 'MET'  # KM added 2/10/20
                    if atom_type == 'SE':
                        atom_type = 'SD'  # change type from Selenium to Sulfur delta
                else:  # KM added 2/10/20
                    residue_type = line[17:20].strip()
                if multimodel:
                    if start_of_new_model or line[21:22].strip() != curr_chain_id:
                        curr_chain_id = line[21:22].strip()
                        model_chain_id = available_chain_ids[model_chain_index]
                        model_chain_index += 1
                        start_of_new_model = False
                    chain = model_chain_id
                else:
                    chain = line[21:22].strip()
                residue_number = int(line[22:26].strip())
                code_for_insertion = line[26:27].strip()
                occ = float(line[54:60].strip())
                temp_fact = float(line[60:66].strip())
                element_symbol = line[76:78].strip()
                atom_charge = line[78:80].strip()
                _atom = (atom_idx, number, atom_type, alt_location, residue_type, chain, residue_number,
                         code_for_insertion, occ, temp_fact, element_symbol, atom_charge)
                if remove_alt_location:
                    if alt_location == '' or alt_location == 'A':
                        if chain not in chain_ids:
                            chain_ids.append(chain)
                        # Store the atomic coordinates in a numpy array
                        coords.append(
                            [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
                        atom_info.append(_atom)
                        atom_idx += 1
                else:
                    if chain not in chain_ids:  # Todo move this outside of the alt_location? Might be some weird files
                        chain_ids.append(chain)
                    # Store the atomic coordinates in a numpy array
                    coords.append([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
                    atom_info.append(_atom)
                    atom_idx += 1

            elif line[0:5] == 'MODEL':
                multimodel = True
                start_of_new_model = True  # signifies that the next line comes after a new model
                # model_chain_id = available_chain_ids[model_chain_index]
                # model_chain_index += 1
            elif line[0:6] == 'SEQRES':
                seq_res_lines.append(line[11:])
            elif line[0:5] == 'DBREF':
                # line = line.strip()
                chain = line[12:14].strip().upper()
                if line[5:6] == '2':
                    db_accession_id = line[18:40].strip()
                else:
                    db = line[26:33].strip()
                    if line[5:6] == '1':  # skip grabbing db_accession_id until DBREF2
                        continue
                    db_accession_id = line[33:42].strip()
                self.dbref[chain] = {'db': db, 'accession': db_accession_id}  # implies each chain has only one id
            elif line[:21] == 'REMARK   2 RESOLUTION':
                try:
                    self.resolution = float(line[22:30].strip().split()[0])
                except ValueError:
                    self.resolution = None
            elif line[:6] == 'COMPND' and 'MOL_ID' in line:
                entity = int(line[line.rfind(':') + 1: line.rfind(';')].strip())
            elif line[:6] == 'COMPND' and 'CHAIN' in line and entity:  # retrieve from standard .pdb file notation
                # entity number (starting from 1) = {'chains' : {A, B, C}}
                self.entity_d[entity] = {'chains': line[line.rfind(':') + 1:].strip().rstrip(';').split(',')}
                entity = None
            elif line[0:5] == 'SCALE':
                self.header.append(line)
            elif line[0:6] == 'CRYST1':
                self.header.append(line)
                self.cryst_record = line.strip()
                self.uc_dimensions, self.space_group = self.parse_cryst_record(self.cryst_record)
                a, b, c, ang_a, ang_b, ang_c = self.uc_dimensions
                self.cryst = {'space': self.space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
                # try:
                #     a = float(line[6:15].strip())
                #     b = float(line[15:24].strip())
                #     c = float(line[24:33].strip())
                #     ang_a = float(line[33:40].strip())
                #     ang_b = float(line[40:47].strip())
                #     ang_c = float(line[47:54].strip())
                # except ValueError:
                #     a, b, c = 0.0, 0.0, 0.0
                #     ang_a, ang_b, ang_c = a, b, c
                # space_group = line[55:66].strip()

        self.coords = Coords(coords)
        # todo set atom coords
        self.set_atoms([Atom.from_info(*info) for info in atom_info])
        # self.atoms = [Atom.from_info(*info) for info in atom_info]
        # self.renumber_atoms()
        self.chain_id_list = chain_ids
        self.log.debug('Multimodel %s, Chains %s' % (True if multimodel else False, self.chain_id_list))
        # self.coords = Coords(coords)  # np.array(coords)
        # self.set_atom_coordinates(self.coords)
        # assert len(self.atoms) == self.coords.shape[0], '%s: ERROR parseing Atoms (%d) and Coords (%d). Wrong size!' \
        #                                                 % (self.filepath, len(self.atoms), self.coords.shape[0])
        # for atom_idx, atom in enumerate(self.atoms):
        #     atom.coords = self.coords[atom_idx]
        self.process_pdb(coords=coords, seqres=seq_res_lines, multimodel=multimodel, **kwargs)

        # if seq_res_lines:
        #     self.parse_seqres(seq_res_lines)
        # else:
        #     self.reference_sequence = {chain_id: None for chain_id in self.chain_id_list}
        #     self.design = True
            # entity.retrieve_sequence_from_api(entity_id=None)

        # self.get_chain_sequences()
        # self.generate_entity_accession_map()
        # if not self.entity_d:
        #     self.update_entities(source='atom')  # pulls entities from the Atom records not RCSB API ('pdb')
        # else:
        #     self.update_entity_d()

    def process_symmetry(self):
        """Find symmetric copies in the PDB and tether Residues and Entities to a single ASU (One chain)"""
        return None

    def process_pdb(self, coords=None, reference_sequence=None, seqres=None, multimodel=False, no_entities=False):
        """Process all Structure Atoms and Residues to PDB, Chain, and Entity compliant objects"""
        if coords:
            self.coords = Coords(coords)
            # replace the Atom and Residue Coords
            self.set_residues_attributes(coords=self._coords)

        self.renumber_atoms()
        self.renumber_residues()
        # self.find_center_of_mass()
        # self.create_residues()
        if seqres:
            self.parse_seqres(seqres)
        else:  # elif reference_sequence:
            self.reference_sequence = {chain_id: None for chain_id in self.chain_id_list}
            self.design = True

        if multimodel:
            self.create_chains(solve_discrepancy=False)
        else:
            self.create_chains()
        if not no_entities:
            self.create_entities()
        self.get_chain_sequences()
        # or
        # for entity in self.entity_d:
        #     self.create_entity(entity, entity_name='%s_%s' % (self.name, entity))

        # if self.design:  # Todo maybe??
        #     self.process_symmetry()
        # self.find_chain_symmetry()  # Todo worry about this later, for now use Nanohedra full project from Pose
        # chains = self.find_max_chain_symmetry()
        # if len(chains) > 1:
        #     dihedral = True
        # the highest order symmetry operation chain in a pdb plus any dihedral related chains

    def parse_seqres(self, seqres_lines):
        """Convert SEQRES information to single amino acid dictionary format

        Args:
            seqres_lines (list): The list of lines containing SEQRES information
        Sets:
            self.reference_sequence
        """
        for line in seqres_lines:
            chain = line.split()[0]
            sequence = line[19:71].strip().split()
            if chain in self.reference_sequence:
                self.reference_sequence[chain].extend(sequence)
            else:
                self.reference_sequence[chain] = sequence

        for chain in self.reference_sequence:
            for i, residue in enumerate(self.reference_sequence[chain]):
                # try:
                if residue.title() in IUPACData.protein_letters_3to1_extended:
                    self.reference_sequence[chain][i] = IUPACData.protein_letters_3to1_extended[residue.title()]
                # except KeyError:
                else:
                    if residue.title() == 'Mse':
                        self.reference_sequence[chain][i] = 'M'
                    else:
                        self.reference_sequence[chain][i] = '-'
            self.reference_sequence[chain] = ''.join(self.reference_sequence[chain])

    def read_atom_list(self, atom_list, store_cb_and_bb_coords=False):  # TODO DEPRECIATE
        """Reads a python list of Atoms and feeds PDB instance updating chain info"""
        if store_cb_and_bb_coords:
            chain_ids = []
            for atom in atom_list:
                self.atoms.append(atom)
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
                self.atoms.append(atom)
                if atom.chain not in chain_ids:
                    chain_ids.append(atom.chain)
            self.chain_id_list += chain_ids
        self.renumber_atoms()
        self.get_chain_sequences()
        # self.update_entities()  # get entity information from the PDB

    # def retrieve_chain_ids(self):  # KM added 2/3/20 to deal with updating chain names after rename_chain(s) functions
    #     # creates a list of unique chain IDs in PDB and feeds it into chain_id_list maintaining order
    #     chain_ids = []
    #     for atom in self.atoms:
    #         chain_ids.append(atom.chain)
    #     chain_ids = list(set(chain_ids))
    #     # chain_ids.sort(key=lambda x: (x[0].isdigit(), x))
    #     self.chain_id_list = chain_ids

    def get_chain_index(self, index):  # Todo Depreciate
        """Return the chain name associated with a set of Atoms when the chain name for those Atoms is changed"""
        return self.chain_id_list[index]

    # def get_entity_atoms(self, entity_id):
    #     """Return list of Atoms containing the subset of Atoms that belong to the selected Entity"""
    #     return [atom for atom in self.get_atoms() if atom.chain in self.entity_d[entity_id]['chains']]

    # def extract_coords(self):
    #     """Grab all the coordinates from the PDB object"""
    #     return [[atom.x, atom.y, atom.z] for atom in self.get_atoms()]
    #
    # def extract_backbone_coords(self):
    #     return [[atom.x, atom.y, atom.z] for atom in self.get_atoms() if atom.is_backbone()]
    #
    # def extract_backbone_and_cb_coords(self):
    #     # inherently gets all glycine CA's
    #     return [[atom.x, atom.y, atom.z] for atom in self.get_atoms() if atom.is_backbone() or
    #             atom.is_CB()]
    #
    # def extract_CA_coords(self):
    #     return [[atom.x, atom.y, atom.z] for atom in self.get_atoms() if atom.is_CA()]
    #
    # def extract_CB_coords(self, InclGlyCA=False):
    #     return [[atom.x, atom.y, atom.z] for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]

    def extract_CB_coords_chain(self, chain, InclGlyCA=False):
        return [[atom.x, atom.y, atom.z] for atom in self.get_atoms()
                if atom.is_CB(InclGlyCA=InclGlyCA) and atom.chain == chain]

    def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):
        coords, cb_indices = [], []
        for idx, atom in enumerate(self.get_atoms()):
            if atom.is_CB(InclGlyCA=InclGlyCA):
                coords.append([atom.x, atom.y, atom.z])

                if ReturnWithCBIndices:
                    cb_indices.append(idx)

        if ReturnWithCBIndices:
            return coords, cb_indices

        else:
            return coords

    def extract_coords_subset(self, res_start, res_end, chain_index, CA):
        if CA:
            selected_atoms = []
            for atom in self.chain(self.chain_id_list[chain_index]):
                if atom.type == "CA":
                    if atom.residue_number >= res_start and atom.residue_number <= res_end:
                        selected_atoms.append(atom)
            out_coords = []
            for atom in selected_atoms:
                [x, y, z] = [atom.x, atom.y, atom.z]
                out_coords.append([x, y, z])
            return out_coords
        else:
            selected_atoms = []
            for atom in self.chain(self.chain_id_list[chain_index]):
                if atom.residue_number >= res_start and atom.residue_number <= res_end:
                    selected_atoms.append(atom)
            out_coords = []
            for atom in selected_atoms:
                [x, y, z] = [atom.x, atom.y, atom.z]
                out_coords.append([x, y, z])
            return out_coords

    # def set_atom_coordinates(self, coords):
    #     """Replate all Atom coords with coords specified. Ensure the coords are the same length"""
    #     assert len(self.atoms) == coords.shape[0], '%s: ERROR setting Atom coordinates, # Atoms (%d) !=  # Coords (%d)'\
    #                                                % (self.filepath, len(self.atoms), coords.shape[0])
    #     self.coords = coords
    #     for idx, atom in enumerate(self.get_atoms()):
    #         atom.coords = coords[idx]
    #         # atom.x, atom.y, atom.z = coords[idx][0], coords[idx][1], coords[idx][2]

    def get_term_ca_indices(self, term):  # Todo DEPRECIATE
        if term == "N":
            ca_term_list = []
            chain_id = None
            for idx, atom in enumerate(self.get_atoms()):
                # atom = self.atoms[i]
                if atom.chain != chain_id and atom.type == "CA":
                    ca_term_list.append(idx)
                    chain_id = atom.chain
            return ca_term_list

        elif term == "C":
            ca_term_list = []
            chain_id = self.atoms[0].chain
            current_ca_idx = None
            for idx, atom in enumerate(self.get_atoms()):
                # atom = self.atoms[i]
                if atom.chain != chain_id:
                    ca_term_list.append(current_ca_idx)
                    chain_id = atom.chain
                if atom.type == "CA":
                    current_ca_idx = idx
            ca_term_list.append(current_ca_idx)
            return ca_term_list

        else:
            self.log.error('Select N or C Term')
            return []

    def mat_vec_mul3(self, a, b):
        c = [0. for i in range(3)]
        for i in range(3):
            c[i] = 0.
            for j in range(3):
                c[i] += a[i][j] * b[j]
        return c

    def rotate_translate(self, rot, tx):  # Todo Depreciate
        for atom in self.get_atoms():
            coord = [atom.x, atom.y, atom.z]
            coord_rot = self.mat_vec_mul3(rot, coord)
            newX = coord_rot[0] + tx[0]
            newY = coord_rot[1] + tx[1]
            newZ = coord_rot[2] + tx[2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def translate(self, tx):  # Todo Depreciate
        for atom in self.get_atoms():
            newX = atom.x + tx[0]
            newY = atom.y + tx[1]
            newZ = atom.z + tx[2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def rotate(self, rot, store_cb_and_bb_coords=False):  # Todo Depreciate
        if store_cb_and_bb_coords:
            for atom in self.get_atoms():
                atom.x, atom.y, atom.z = self.mat_vec_mul3(rot, [atom.x, atom.y, atom.z])
                if atom.is_backbone():
                    self.bb_coords.append([atom.x, atom.y, atom.z])
                if atom.is_CB(InclGlyCA=False):
                    self.cb_coords.append([atom.x, atom.y, atom.z])
        else:
            for atom in self.get_atoms():
                atom.x, atom.y, atom.z = self.mat_vec_mul3(rot, [atom.x, atom.y, atom.z])

    def rotate_along_principal_axis(self, degrees=90.0, axis='x'):  # Todo Depreciate
        """Rotate the coordinates about the given axis
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
            self.log.error('Axis does not exists!')

        for atom in self.get_atoms():
            coord = [atom.x, atom.y, atom.z]
            # Todo replace below with atom.x, atom.y, atom.z = np.matmul(rotmatrix * coord)
            newX = coord[0] * rotmatrix[0][0] + coord[1] * rotmatrix[0][1] + coord[2] * rotmatrix[0][2]
            newY = coord[0] * rotmatrix[1][0] + coord[1] * rotmatrix[1][1] + coord[2] * rotmatrix[1][2]
            newZ = coord[0] * rotmatrix[2][0] + coord[1] * rotmatrix[2][1] + coord[2] * rotmatrix[2][2]
            atom.x, atom.y, atom.z = newX, newY, newZ

    def ReturnRotatedPDB(self, degrees=90.0, axis='x', store_cb_and_bb_coords=False):  # Todo Depreciate
        """Rotate the coordinates about the given axis
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
            self.log.error('Axis does not exists!')

        rotated_atoms = []
        for atom in self.get_atoms():
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] * rotmatrix[0][0] + coord[1] * rotmatrix[0][1] + coord[2] * rotmatrix[0][2]
            newY = coord[0] * rotmatrix[1][0] + coord[1] * rotmatrix[1][1] + coord[2] * rotmatrix[1][2]
            newZ = coord[0] * rotmatrix[2][0] + coord[1] * rotmatrix[2][1] + coord[2] * rotmatrix[2][2]
            rot_atom = deepcopy(atom)
            rot_atom.x, rot_atom.y, rot_atom.z = newX, newY, newZ
            rotated_atoms.append(rot_atom)

        # rotated_pdb = PDB(atoms=)
        rotated_pdb = PDB()
        rotated_pdb.read_atom_list(rotated_atoms, store_cb_and_bb_coords=store_cb_and_bb_coords)

        return rotated_pdb

    def ReturnTranslatedPDB(self, tx, store_cb_and_bb_coords=False):  # Todo Depreciate
        translated_atoms = []
        for atom in self.get_atoms():
            coord = [atom.x, atom.y, atom.z]
            newX = coord[0] + tx[0]
            newY = coord[1] + tx[1]
            newZ = coord[2] + tx[2]
            tx_atom = deepcopy(atom)
            tx_atom.x, tx_atom.y, tx_atom.z = newX, newY, newZ
            translated_atoms.append(tx_atom)

        translated_pdb = PDB()
        translated_pdb.read_atom_list(translated_atoms, store_cb_and_bb_coords)

        return translated_pdb

    def ReturnRotatedPDBMat(self, rot):  # Todo Depreciate
        rotated_coords = []
        return_atoms = []
        return_pdb = PDB()
        for coord in self.extract_coords():
            coord_copy = deepcopy(coord)
            coord_rotated = self.mat_vec_mul3(rot, coord_copy)
            rotated_coords.append(coord_rotated)
        for i in range(len(self.get_atoms())):
            atom_copy = deepcopy(self.atoms[i])
            atom_copy.x = rotated_coords[i][0]
            atom_copy.y = rotated_coords[i][1]
            atom_copy.z = rotated_coords[i][2]
            return_atoms.append(atom_copy)
        return_pdb.read_atom_list(return_atoms)

        return return_pdb

    def rename_chains(self, chain_list_fixed):  # Todo Depreciate
        # Caution, doesn't update self.reference_sequence chain info
        lf = chain_list_fixed
        lm = self.chain_id_list[:]

        l_abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9']

        l_av = []
        for e in PDB.available_letters:
            if e not in lm:
                if e not in lf:
                    l_av.append(e)

        j = 0
        for i in range(len(lm)):
            if lm[i] in lf:
                lm[i] = l_av[j]
                j += 1

        prev = self.atoms[0].chain
        c = 0
        l3 = []
        for i in range(len(self.get_atoms())):
            if prev != self.atoms[i].chain:
                c += 1
            l3.append(lm[c])
            prev = self.atoms[i].chain

        for i in range(len(self.get_atoms())):
            self.atoms[i].chain = l3[i]

        self.chain_id_list = lm
        self.get_chain_sequences()

    # def rename_chain(self, chain_of_interest, new_chain):
    #     """Rename a single chain to a identifier of your choice.
    #     Caution, will rename to already taken chain and doesn't update self.reference_sequence chain info
    #     """
    #     for atom in self.chain(chain_of_interest).get_atoms():
    #         atom.chain = new_chain
    #
    #     self.chain_id_list[self.chain_id_list.index(chain_of_interest)] = new_chain
    #     self.get_chain_sequences()

    def reorder_chains(self, exclude_chains_list=None):
        """Renames chains starting from the first chain as A and the last as l_abc[len(self.chain_id_list) - 1]
        Caution, doesn't update self.reference_sequence chain info
        """
        if exclude_chains_list:
            available_chains = list(set(PDB.available_letters) - set(exclude_chains_list))
        else:
            available_chains = PDB.available_letters

        moved_chains = [available_chains[idx] for idx in range(self.number_of_chains)]

        for idx, chain in enumerate(self.chain_id_list):
            self.chain(chain).set_atoms_attributes(chain=moved_chains[idx])
        # prev_chain = self.atoms[0].chain
        # chain_index = 0
        # l3 = []
        #
        # for atom in self.get_atoms():
        #     if atom.chain != prev_chain:
        #         chain_index += 1
        #     l3.append(moved_chains[chain_index])
        #     prev_chain = atom.chain
        #
        # # Update all atom chain names
        # for idx, atom in enumerate(self.get_atoms()):
        #     atom.chain = l3[idx]

        # Update chain_id_list
        self.chain_id_list = moved_chains
        self.get_chain_sequences()

    def renumber_pdb(self):
        self.renumber_atoms()
        self.renumber_residues()

    def renumber_residues_by_chain(self):
        for chain in self.chains:
            chain.renumber_residues()

    # def reindex_chain_residues(self, chain):
        # Starts numbering chain residues at 1 and numbers sequentially until reaches last atom in chain
        # chain_atoms = self.get_chain_atoms(chain)
        # idx = chain_atoms[0].number - 1  # offset to chain 0
        # last_atom_index = idx + len(chain_atoms)
        # for i, residue in enumerate(self.get_chain_residues(chain), 1):
        #     current_res_num = residue.number
        #     while self.atoms[idx].residue_number == current_res_num:
        #         self.atoms[idx].residue_number = i
        #         idx += 1
        #         if idx == last_atom_index:
        #             break
        # self.renumber_atoms()  # should be unnecessary

    def AddZAxis(self):
        z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
        z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")

        axis = [z_axis_a, z_axis_b, z_axis_c]
        self.atoms.extend(axis)

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
        self.atoms.extend(axes)

    # def get_terminal_residue(self, termini='c'):
    #     if termini == "N":
    #         for atom in self.get_chain_atoms(chain_id):
    #             if atom.type == "CA":
    #                 return atom
    #     elif termini == "C":
    #         for atom in self.get_chain_atoms(chain_id)[::-1]:
    #             if atom.type == "CA":
    #                 return atom
    #     else:
    #         print('Select N or C Term')
    #         return None
    #
    # def get_residue_atoms(self, chain_id, residue_numbers):
    #     if not isinstance(residue_numbers, list):
    #         residue_numbers = list(residue_numbers)
    #
    #     atoms = []
    #     # for residue in self.residues:
    #     #     if residue.chain == chain_id and residue.number in residue_numbers:
    #     #         atoms.extend(residue.get_atoms())
    #
    #     _residues = self.chain(chain_id).get_residues(numbers=residue_numbers)
    #     for _residue in _residues:
    #         atoms.extend(_residue.get_atoms())
    #
    #     return atoms

    def create_chains(self, solve_discrepancy=True):  # Todo fix the discrepancy to not modify chain names if possible
        """For all the Residues in the PDB, create Chain objects which contain their member Residues"""
        if solve_discrepancy:
            chain_idx = 0
            chain_id = PDB.available_letters[chain_idx]
            chain_residues = {chain_id: [self.residues[0]]}
            # previous_chain = self.residues[0].chain
            for idx, residue in enumerate(self.residues[1:], 1):  # start at the second index to avoid off by one
                if residue.number_pdb < self.residues[idx - 1].number_pdb \
                        or residue.chain != self.residues[idx - 1].chain:
                    # Decreased number should only happen with new chain therefore this SHOULD satisfy a malformed PDB
                    chain_idx += 1
                    # if residue.chain == previous_chain:  # residue[idx - 1].chain:
                    chain_id = PDB.available_letters[chain_idx]  # get a new chain id
                    # else:
                    #     chain = residue.chain
                    chain_residues[chain_id] = [residue]
                else:
                    chain_residues[chain_id].append(residue)

            for idx, (chain_id, residues) in enumerate(chain_residues.items()):
                # for residue in residues:
                #     residue.set_atoms_attributes(chain=chain_id)
                self.chains.append(Chain(name=chain_id, coords=self._coords, residues=residues, log=self.log))  # ,
                #                        sequence=self.reference_sequence[chain_id]))
                self.chains[idx].set_residues_attributes(chain=chain_id)
        else:
            for chain_id in self.chain_id_list:
                self.chains.append(Chain(name=chain_id, coords=self._coords, log=self.log,
                                         residues=[residue for residue in self.residues if residue.chain == chain_id]))
                #                        sequence = self.reference_sequence[chain_id],

    def get_chains(self, names=None):
        """Retrieve Chains in PDB. Returns all by default. If a list of names is provided, the selected Chains are
        returned"""
        if names and isinstance(names, Iterable):
            return [chain for chain in self.chains if chain.name in names]
        else:
            return self.chains

    def chain(self, chain_id):
        """Return the Chain object specified by the passed chain ID. If not found, return None
        Returns:
            (Chain)
        """
        for chain in self.chains:
            if chain.name == chain_id:
                return chain
        return None

    def get_chain_atoms(self, chain_ids):  # Todo Depreciate
        """Return list of Atoms containing the subset of Atoms that belong to the selected Chain(s)"""
        if not isinstance(chain_ids, list):
            chain_ids = list(chain_ids)

        atoms = []
        for chain in self.chains:
            if chain.name in chain_ids:
                atoms.extend(chain.get_atoms())
        return atoms
        # return [atom for atom in self.atoms if atom.chain == chain_id]

    # def get_chain_residues(self, chain_id):  # Todo Depreciate
    #     """Return the Residues included in a particular chain"""
    #     return [residue for residue in self.residues if residue.chain == chain_id]

    # def write(self, out_path=None, cryst1=None):  # Todo Depreciate
    #     if not cryst1:
    #         cryst1 = self.cryst_record
    #     with open(out_path, "w") as outfile:
    #         if cryst1 and isinstance(cryst1, str) and cryst1.startswith("CRYST1"):
    #             outfile.write(str(cryst1) + "\n")
    #         outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))

    def get_chain_sequences(self):
        self.atom_sequences = {chain.name: chain.sequence for chain in self.chains}
        # self.atom_sequences = {chain: self.chain(chain).get_structure_sequence() for chain in self.chain_id_list}
    #
    # # def orient(self, sym=None, orient_dir=os.getcwd(), generate_oriented_pdb=True):
    #     self.write('input.pdb')
    #     # os.system('cp %s input.pdb' % self.filepath)
    #     # os.system('%s/orient_oligomer_rmsd >> orient.out 2>&1 << eof\n%s/%s\neof' % (orient_dir, orient_dir, symm))
    #     os.system('%s/orient_oligomer >> orient.out 2>&1 << eof\n%s/%s_symm.txt\neof' % (orient_dir, orient_dir, sym))
    #     os.system('mv output.pdb %s_orient.pdb' % os.path.splitext(self.filepath)[0])  # Todo this could be removed
    #     os.system('rm input.pdb')
    #     if os.path.exists('%s_orient.pdb' % os.path.splitext(self.filepath)[0]):
    #         if generate_oriented_pdb:
    #             oriented_pdb = PDB()
    #             oriented_pdb.readfile('%s_orient.pdb' % os.path.splitext(self.filepath)[0], remove_alt_location=True)
    #             os.system('rm %s_orient.pdb' % os.path.splitext(self.filepath)[0])  # Todo this could be removed
    #             return oriented_pdb
    #         else:
    #             return 0
    #     else:
    #         return None

    def orient(self, sym=None, out_dir=os.getcwd(), generate_oriented_pdb=False):
        """Orient a symmetric PDB at the origin with it's symmetry axis cannonically set on axis defined by symmetry
        file"""
        valid_subunit_number = {"C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "D2": 4, "D3": 6, "D4": 8, "D5": 10,
                                "D6": 12, "I": 60, "O": 24, "T": 12}
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

            # orient_input = 'input.pdb'
            # orient_output = 'output.pdb'
            orient_input = os.path.join(orient_dir, 'input.pdb')
            orient_output = os.path.join(orient_dir, 'output.pdb')

            def clean_orient_input_output():
                if os.path.exists(orient_input):
                    os.remove(orient_input)
                if os.path.exists(orient_output):
                    os.remove(orient_output)

            clean_orient_input_output()
            # self.reindex_all_chain_residues()  TODO test efficacy. It could be that this screws up more than helps.
            self.write(orient_input)
            # self.write('input.pdb')
            # copyfile(self.filepath, orient_input)

            p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=orient_dir)
            in_symm_file = os.path.join(orient_dir, 'symm_files', sym)
            stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
            stderr = stderr.decode()  # turn from bytes to string 'utf-8' implied
            stdout = pdb_file_name + stdout.decode()[28:]

            log_f.write(stdout)
            log_f.write('%s\n' % stderr)
            if os.path.exists(orient_output) and os.stat(orient_output).st_size != 0:
                if generate_oriented_pdb:
                    oriented_file = os.path.join(out_dir, pdb_file_name)
                    move(orient_output, oriented_file)
                    new_pdb = None
                else:
                    new_pdb = PDB(file=orient_output)
            else:
                raise RuntimeError(error_string)

            clean_orient_input_output()
            return new_pdb

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
                self.log.error('Axis selected for sampling is not supported!')
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
                self.log.error('Invalid sampling axis!')
                return None

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
                return None

        pdb_coords = self.extract_coords()
        rot_matrices = get_rot_matrices(rot_step_deg, axis, rot_range_deg)
        tx_matrices = get_tx_matrices(tx_step, axis, start_tx_range, end_tx_range)
        sampled_coords_np = generate_sampled_coordinates_np(pdb_coords, rot_matrices, tx_matrices, degeneracy)

        if sampled_coords_np is not None:
            if rotational_setting_matrix is not None:
                rotational_setting_matrix_np = np.array(rotational_setting_matrix)
                rotational_setting_matrix_np_t = np.transpose(rotational_setting_matrix_np)
                sampled_coords_orient_canon_np = np.matmul(sampled_coords_np, rotational_setting_matrix_np_t)
                return sampled_coords_orient_canon_np.tolist()
            else:
                return sampled_coords_np.tolist()
        else:
            return None

    def get_sasa(self, probe_radius=1.4, sasa_thresh=0):
        """Use FreeSASA to calculate the surface area of a PDB.

        Must be run with the self.filepath attribute. Entities/chains could have this, but don't currently"""
        current_pdb_file = self.write(out_path='sasa_input.pdb')
        self.log.debug(current_pdb_file)
        p = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius),
                              current_pdb_file], stdout=subprocess.PIPE)
        # p = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius), self.filepath],
        #                      stdout=subprocess.PIPE)
        out, err = p.communicate()
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

        os.system('rm %s' % current_pdb_file)
        return sasa_out_chain, sasa_out_res, sasa_out

    def stride(self, chain=None):
        """Use Stride to calculate the secondary structure of a PDB.

        Must be run with the self.filepath attribute. Entities/chains could have this, but don't currently"""
        # REM  -------------------- Secondary structure summary -------------------  XXXX
        # REM                .         .         .         .         .               XXXX
        # SEQ  1    IVQQQNNLLRAIEAQQHLLQLTVWGIKQLQAGGWMEWDREINNYTSLIHS   50          XXXX
        # STR       HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  HHHHHHHHHHHHHHHHH               XXXX
        # REM                                                                        XXXX
        # SEQ  51   LIEESQN                                              57          XXXX
        # STR       HHHHHH                                                           XXXX
        # REM                                                                        XXXX
        # LOC  AlphaHelix   ILE     3 A      ALA     33 A                            XXXX
        # LOC  AlphaHelix   TRP    41 A      GLN     63 A                            XXXX
        # REM                                                                        XXXX
        # REM  --------------- Detailed secondary structure assignment-------------  XXXX
        # REM                                                                        XXXX
        # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      XXXX
        # ASG  ILE A    3    1    H    AlphaHelix    360.00    -29.07     180.4      XXXX
        # ASG  VAL A    4    2    H    AlphaHelix    -64.02    -45.93      99.8      XXXX
        # ASG  GLN A    5    3    H    AlphaHelix    -61.99    -39.37      82.2      XXXX

        # try:
            # with open(os.devnull, 'w') as devnull:
        stride_cmd = [stride_exe_path, '%s' % self.filepath]
        #   -rId1Id2..  Read only chains Id1, Id2 ...
        #   -cId1Id2..  Process only Chains Id1, Id2 ...
        if chain:
            stride_cmd.append('-c%s' % chain)

        p = subprocess.Popen(stride_cmd, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        out_lines = out.decode('utf-8').split('\n')
        # except:
        #     stride_out = None

        # if stride_out is not None:
        #     lines = stride_out.split('\n')

        for line in out_lines:
            if line[0:3] == 'ASG' and line[10:15].strip().isdigit():
                self.chain(line[9:10]).residue(int(line[10:15].strip())).set_secondary_structure(line[24:25])
        self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]
        # self.secondary_structure = {int(line[10:15].strip()): line[24:25] for line in out_lines
        #                             if line[0:3] == 'ASG' and line[10:15].strip().isdigit()}

    def calculate_secondary_structure(self, chain=None):  # different from Josh PDB
        self.stride(chain=chain)

    def get_secondary_structure_chain(self, chain=None):
        if self.secondary_structure:
            return self.chain(chain).get_secondary_structure()
        else:
            self.fill_secondary_structure()
            if list(filter(None, self.secondary_structure)):  # check if there is at least 1 secondary struc assignment
                return self.chain(chain).get_secondary_structure()
            else:
                return None

    def get_surface_helix_cb_indices(self, probe_radius=1.4, sasa_thresh=1):
        # only works for monomers or homo-complexes
        sasa_chain, sasa_res, sasa = self.get_sasa(probe_radius=probe_radius, sasa_thresh=sasa_thresh)

        h_cb_indices = []
        stride = Stride(self.filepath, self.chain_id_list[0], stride_exe_path)
        stride.run()
        stride_ss_asg = stride.ss_asg
        for i in range(len(self.get_atoms())):
            atom = self.atoms[i]
            if atom.is_CB():
                if (atom.residue_number, "H") in stride_ss_asg and atom.residue_number in sasa_res:
                    h_cb_indices.append(i)
        return h_cb_indices

    def get_surface_atoms(self, chain_selection="all", probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        sasa_chain, sasa_res, sasa = self.get_sasa(probe_radius=probe_radius, sasa_thresh=sasa_thresh)
        sasa_chain_res_l = zip(sasa_chain, sasa_res)

        surface_atoms = []
        if chain_selection == "all":
            for atom in self.get_atoms():
                if (atom.chain, atom.residue_number) in sasa_chain_res_l:
                    surface_atoms.append(atom)
        else:
            for atom in self.get_chain_atoms(chain_selection):
                if (atom.chain, atom.residue_number) in sasa_chain_res_l:
                    surface_atoms.append(atom)

        return surface_atoms

    def get_surface_residue_info(self, probe_radius=2.2, sasa_thresh=0):
        # only works for monomers or homo-complexes
        sasa_chain, sasa_res, sasa = self.get_sasa(probe_radius=probe_radius, sasa_thresh=sasa_thresh)

        return sasa_res
        # return list(set(zip(sasa_chain, sasa_res)))  # for use with chains

    def get_surface_area_residues(self, residue_numbers, probe_radius=2.2):
        """CURRENTLY UNUSEABLE Must be run when the PDB is in Pose numbering, chain data is not connected"""
        sasa_chain, sasa_res, sasa = self.get_sasa(probe_radius=probe_radius)
        total_sasa = 0
        for chain, residue_number, sasa in zip(sasa_chain, sasa_res, sasa):
            if residue_number in residue_numbers:
                total_sasa += sasa

        return total_sasa

    def insert_residue(self, chain_id, residue_number, residue_type):  # Todo Chain compatible
        """Insert a residue into the PDB. Only works for pose_numbering (1 to N). Assumes atom numbers are properly
        indexed"""
        # Convert 3 letter aa to uppercase, 1 letter aa
        if residue_type.title() in IUPACData.protein_letters_3to1_extended:
            residue_type_1 = IUPACData.protein_letters_3to1_extended[residue_type.title()]
        else:  # Why would this be useful?
            residue_type_1 = residue_type.upper()

        # Find atom insertion index, should be last atom in preceding residue
        if residue_number == 1:
            insert_atom_idx = 0
        else:
            residue_atoms = self.chain(chain_id).residue(residue_number).get_atoms()
            # residue_atoms = self.get_residue_atoms(chain_id, residue_number)
            if residue_atoms:
                insert_atom_idx = residue_atoms[0].number - 1  # subtract 1 from first atom number to get insertion idx
            else:  # Atom index is not an insert operation as the location is at the C-term of the chain
                # prior_index = self.getResidueAtoms(chain, residue)[0].number - 1
                prior_chain_length = self.chain(chain_id).residues[0].get_atoms()[0].number - 1
                # chain_atoms = self.chain(chain_id).get_atoms()
                # chain_atoms = self.get_chain_atoms(chain_id)

                # use length of all prior chains + length of all_chain_atoms
                insert_atom_idx = prior_chain_length + self.chain(chain_id).number_of_atoms  # ()
                # insert_atom_idx = len(chain_atoms) + chain_atoms[0].number - 1

            # insert_atom_idx = self.getResidueAtoms(chain, residue)[0].number

        # Change all downstream residues
        for atom in self.atoms[insert_atom_idx:]:
            # atom.number += len(insert_atoms)
            # if atom.chain == chain: TODO uncomment for pdb numbering
            atom.residue_number += 1

        # Grab the reference atom coordinates and push into the atom list
        if not self.reference_aa:
            # TODO load in Residue.py
            self.reference_aa = PDB.from_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                                                           'AAreference.pdb'),
                                              log=start_log(handler=3), no_entities=True)
        insert_atoms = self.reference_aa.chain('A').residue(IUPACData.protein_letters.find(residue_type_1)).get_atoms()

        for atom in reversed(insert_atoms):  # essentially a push
            atom.chain = chain_id
            atom.residue_number = residue_number
            atom.occ = 0
            self.atoms.insert(insert_atom_idx, atom)

        self.renumber_pdb()

    def delete_residue(self, chain_id, residue_number):
        start = len(self.atoms)
        self.log.debug(len(self.atoms))
        # residue = self.get_residue(chain, residue_number)
        chain = self.chain(chain_id)
        residue = chain.residue(residue_number)
        # residue.delete_atoms()  # deletes Atoms from Residue. unneccessary?
        self.delete_atoms(residue.get_atoms())  # deletes Atoms from PDB
        chain.residues.remove(residue)  # deletes Residue from Chain
        self.residues.remove(residue)  # deletes Residue from PDB
        self.renumber_pdb()
        self.log.debug('Deleted: %d atoms' % (start - len(self.atoms)))

    def delete_atoms(self, atoms):
        # Need to call self.renumber_atoms() after every call to delete_atoms()
        # Todo find every Structure container with Atom and remove the Atom references from it. Chain, Entity, Residue,
        #  PDB. Atom may hold on in memory because of refernce to Coords.
        for atom in atoms:
            self.atoms.remove(atom)

    def apply(self, rot=None, tx=None):  # Todo move to Structure?
        """Apply a transformation to the PDB object"""
        # moved = []
        # for coord in self.extract_coords():
        #    coord_moved = self.mat_vec_mul3(rot, coord)
        #    for j in range(3):
        #         coord_moved[j] += tx[j]
        #    moved.append(coord_moved)
        if rot is not None:
            moved = np.matmul(np.array(rot), self.coords)
        else:
            moved = self.coords
        if tx is not None:
            moved = moved + np.array(tx)

        self.coords = Coords(moved)
        # self.set_atom_coordinates(moved)

    def get_ave_residue_b_factor(self, chain_id, residue_number):
        residue_atoms = self.chain(chain_id).residue(residue_number).get_atoms()
        # residue_atoms = self.get_residue_atoms(chain, residue)
        temp = 0
        for atom in residue_atoms:
            temp += atom.temp_fact

        return round(temp / len(residue_atoms), 2)

    def retrieve_pdb_info_from_api(self, pdb_code=None):  # Todo doesn't really need pdb_code currently. When would it?
        if not pdb_code:
            # if self.name:
            pdb_code = self.name
            # else:
            #     print('No PDB code associated with file! Cannot retrieve Entity information from PDB API')
            #     return None
        if len(pdb_code) == 4:
            if not self.api_entry:
                self.api_entry = get_pdb_info_by_entry(pdb_code)
                if self.api_entry:
                    return True
                self.log.info('PDB code \'%s\' was not found with the PDB API.' % pdb_code)
        else:
            self.log.info('PDB code \'%s\' is not of the required format and will not be found with the PDB API.'
                          % pdb_code)
        return False
        # if not self.api_entry and self.name and len(self.name) == 4:
        #     self.api_entry = get_pdb_info_by_entry(self.name)
        # self.get_dbref_info_from_api()
        # self.get_entity_info_from_api()

    # def get_dbref_info_from_api(self, pdb_code=None):
    #     if self.retrieve_pdb_info_from_api(pdb_code=pdb_code):
    #     # if not self.api_entry and self.name and len(self.name) == 4:
    #     #     self.api_entry = get_pdb_info_by_entry(self.name)
    #         self.dbref = self.api_entry['dbref']
    #         return True
    #     else:
    #         return False

    def entity(self, entity_id):
        for entity in self.entities:
            if entity_id == entity.name:
                return entity
        return None

    def create_entities(self):  # , entity_names=None, pdb_code=None):
        """Create all Entities in the PDB object searching for the required information if it was not found during
        parsing. First search the PDB API if a PDB entry_id is attached to instance, next from Atoms in instance

        Keyword Args:
            entity_names=None (list): The list of names for each Entity is names are provided, otherwise, Entity names
            will increment from order identified in ATOM records
            pdb_code=None (str): The four character code specifying the entry ID from the PDB
        """
        if not self.entity_d:  # we didn't find information from the PDB file
            self.get_entity_info_from_api()  # pdb_code=pdb_code)
        if not self.entity_d:  # API didn't work for pdb_name
            self.get_entity_info_from_atoms()
            for entity, atom_info in self.entity_d.items():
                pdb_api_name = retrieve_entity_id_by_sequence(atom_info['seq'])
                if pdb_api_name:
                    self.entity_d[pdb_api_name] = self.entity_d.pop(entity)
            self.log.info('Found Entities \'%s\' by PDB API sequence search' % ', '.join(self.entity_d.keys()))
        else:
            for entity, info in self.entity_d.items():
                info['chains'] = [self.chain(chain_id) for chain_id in info['chains']]  # ['representative']
                info['chains'] = filter(None, info['chains'])

        # # solve this upon chain loading...?
        # if len(self.entity_d) > self.number_of_chains:
        #     # the chain info is certainly messed up, flexibly solve chains using residue increment
        #     self.log.warning('%s: Incorrect chain (%d) and entity (%d) count. Chains probably have the same id! '
        #                      'Attempting to correct' % (self.filepath, self.number_of_chains, len(self.entity_d)))
        #     self.create_chains(solve_discrepancy=True)
        self.update_entity_d()  # pdb_code=pdb_code)

        for entity, info in self.entity_d.items():
            # chain_l = [self.chain(chain_id) for chain_id in info['chains']]  # ['representative']
            # chain_l = filter(None, chain_l)
            if len(entity.split('_')) == 2:  # we have an name generated from a PDB API sequence search
                entity_name = entity
            else:  #
                entity_name = '%s_%d' % (self.name, entity)
            # self.entities.append(self.create_entity(representative_chain=self.entity_d[entity]['representative'],
            #                                         chains=chain_l, entity_id=entity_name,
            #                                         uniprot_id=self.entity_accession_map[entity]))
            self.entities.append(Entity.from_representative(chains=info['chains'], name=entity_name,
                                                            coords=self._coords,
                                                            uniprot_id=info['accession'], log=self.log,
                                                            representative=info['representative']))

    # def update_entities(self):  # , pdb_code=None):
    #     """Add Entity information to the PDB object using the PDB API if pdb_code is specified or .pdb filename is a
    #     four letter code. If not, gather Entitiy information from the ATOM records"""
    #     # if pdb_code or self.name:
    #     #     self.get_entity_info_from_api(pdb_code=pdb_code)
    #     # else:
    #     #     self.get_entity_info_from_atoms()
    #
    #     # self.update_entity_d()
    #     self.update_entity_representatives()
    #     self.update_entity_sequences()
    #     self.update_entity_accession_id()

    def update_entity_d(self):
        """Update a complete entity_d with the required information
        Todo choose the most symmetrically average chain if Entity is symmetric!
        For each Entity, gather the sequence of the chain representative"""
        self.log.debug('Entity Dictionary: %s' % self.entity_d)
        for entity, info in self.entity_d.items():
            info['representative'] = info['chains'][0]  # We may get an index error someday. If so, fix upstream logic
            info['seq'] = info['representative'].sequence

        self.update_entity_accession_id()

    def get_entity_info_from_atoms(self):
        """Find all unique Entities in the input .pdb file. These are unique sequence objects"""
        entity_count = 1
        # for chain in self.atom_sequences:
        for chain in self.chains:
            new_entity = True  # assume all chains are unique entities
            self.log.debug('Searching for matching Entities for Chain %s' % chain.name)
            for entity in self.entity_d:
                # check if the sequence associated with the atom chain is in the entity dictionary
                # if self.atom_sequences[chain] == self.entity_d[entity]['seq']:
                if chain.sequence == self.entity_d[entity]['seq']:
                    # score = len(self.atom_sequences[chain])
                    score = len(chain.sequence)
                else:
                    # alignment = pairwise2.align.localxx(self.atom_sequences[chain], self.entity_d[entity]['seq'])
                    alignment = pairwise2.align.localxx(chain.sequence, self.entity_d[entity]['seq'])
                    score = alignment[0][2]  # first alignment from localxx, grab score value
                self.log.debug('Chain %s has score of %d for Entity %d' % (chain.name, score, entity))
                self.log.debug('Chain %s has match score of %f' % (chain.name,
                                                                   score / len(self.entity_d[entity]['seq'])))
                if score / len(self.entity_d[entity]['seq']) > 0.9:  # if score/length is > 90% similar, entity exists
                    # rmsd = Bio.Superimposer()
                    # if rmsd > 1:
                    self.entity_d[entity]['chains'].append(chain)
                    new_entity = False  # The entity is not unique, do not add
                    break
            if new_entity:  # no existing entity matches, add new entity
                self.entity_d[copy(entity_count)] = {'chains': [chain], 'seq': chain.sequence}
                entity_count += 1
        self.log.info('Entities were generated from ATOM records.')

    def get_entity_info_from_api(self, pdb_code=None):
        """Query the PDB API for the PDB entry_ID to find the corresponding Enitity information"""
        if self.retrieve_pdb_info_from_api(pdb_code=pdb_code):
            self.entity_d = {entity: {'chains': self.api_entry['entity'][entity]} for entity in
                             self.api_entry['entity']}
        # else:
        #     self.get_entity_info_from_atoms()

    # def update_entity_representatives(self):
    #     """For each Entity, gather the chain representative by choosing the first chain in the file
    #     Todo choose the most symmetrically average chain if Entity is symmetric!
    #     """
    #     for entity, info in self.entity_d.items():
    #         info['representative'] = info['chains'][0]  # We may get an index error someday. If so, fix upstream logic

    # def update_entity_sequences(self):
    #     """For each Entity, gather the sequence of the chain representative"""
    #     self.log.debug(self.entity_d)
    #     for entity, info in self.entity_d.items():
    #         info['seq'] = info['representative'].sequence

    def update_entity_accession_id(self):
        """Create a map (dictionary) between identified entities (not yet Entity objs) and their accession code
        If entities from psuedo generation, then may not.
        """
        # dbref will not be generated unless specified by call to API or from .pdb file
        if not self.dbref:  # check if from .pdb file
            if not self.api_entry:  # check if from API
                for entity, info in self.entity_d.items():
                    info['accession'] = None
                return None
            else:
                self.dbref = self.api_entry['dbref']

        for entity, info in self.entity_d.items():
            info['accession'] = self.dbref[info['representative'].name]['accession']
        # self.entity_accession_map = {entity: self.dbref[self.entity_d[entity]['representative']]['accession']
        #                              for entity in self.entity_d}

    def entity_from_chain(self, chain_id):
        """Return the entity associated with a particular chain id"""
        # for entity, info in self.entity_d.items():
        for entity in self.entities:
            if chain_id == entity.chain_id:
                return entity
        return None

    def entity_from_residue(self, residue_number):
        """Return the entity associated with a particular Residue number"""
        for entity in self.entities:
            if entity.get_residues(numbers=[residue_number]):
                return entity
        return None

    def match_entity_by_struct(self, other_struct=None, entity=None, force_closest=False):
        """From another set of atoms, returns the first matching chain from the corresponding entity"""
        return None  # TODO when entities are structure compatible

    def match_entity_by_seq(self, other_seq=None, force_closest=False, threshold=0.7):
        """From another sequence, returns the first matching chain from the corresponding entity"""

        if force_closest:
            alignment_score_d = {}
            for entity in self.entity_d:
                # TODO get a gap penalty and rework entire alignment function...
                alignment = pairwise2.align.localxx(other_seq, self.entity_d[entity]['seq'])
                max_align_score, max_alignment = 0, None
                for i, align in enumerate(alignment):
                    if align.score > max_align_score:
                        max_align_score = align.score
                        max_alignment = i
                alignment_score_d[entity] = alignment[max_alignment].score
                # alignment_score_d[entity] = alignment[0][2]

            max_score, max_score_entity = 0, None
            for entity in alignment_score_d:
                normalized_score = alignment_score_d[entity] / len(self.entity_d[entity]['seq'])
                if normalized_score > max_score:
                    max_score = normalized_score  # alignment_score_d[entity]
                    max_score_entity = entity
            if max_score > threshold:
                return self.entity_d[max_score_entity]['chains'][0]
            else:
                return None
        else:
            for entity in self.entity_d:
                if other_seq == self.entity_d[entity]['seq']:
                    return self.entity_d[entity]['representative'][0]

    def chain_interface_contacts(self, chain_id, distance=8, gly_ca=False):  # Todo very similar to Pose with entities
        """Create a atom tree using CB atoms from one chain and all other atoms

        Args:
            chain_id (PDB): First PDB to query against
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
            gly_ca=False (bool): Whether glycine CA should be included in the tree
        Returns:
            chain_atoms, all_contact_atoms (list, list): Chain interface atoms, all contacting interface atoms
        """
        # Get all CB Atom & chain CB Atom Coordinates into a numpy array [[x, y, z], ...]
        all_coords = self.get_cb_coords(InclGlyCA=gly_ca)
        # all_coords = np.array(self.extract_CB_coords(InclGlyCA=gly_ca))
        chain_coords = self.chain(chain_id).get_cb_coords(InclGlyCA=gly_ca)
        # chain_coords = np.array(self.extract_CB_coords_chain(chain_id, InclGlyCA=gly_ca))

        # Construct CB Tree for the chain
        chain_tree = BallTree(chain_coords)

        # Get CB Atom indices for the atoms CB and chain CB
        all_cb_indices = self.get_cb_indices(InclGlyCA=gly_ca)
        chain_cb_indices = self.chain(chain_id).get_cb_indices(InclGlyCA=gly_ca)
        # chain_cb_indices = self.get_cb_indices_chain(chain_id, InclGlyCA=gly_ca)
        chain_coord_indices, contact_cb_indices = [], []
        # Find the contacting CB indices and chain specific indices
        for i, idx in enumerate(all_cb_indices):
            if idx not in chain_cb_indices:
                contact_cb_indices.append(idx)
            else:
                chain_coord_indices.append(i)

        # Remove chain specific coords from all coords by deleting them from numpy
        contact_coords = np.delete(all_coords, chain_coord_indices, axis=0)
        # Query chain CB Tree for all contacting Atoms within distance
        chain_query = chain_tree.query_radius(contact_coords, distance)

        all_contact_atoms, chain_atoms = [], []
        for contact_idx, contacts in enumerate(chain_query):
            if chain_query[contact_idx].tolist() != list():
                all_contact_atoms.append(self.atoms[contact_cb_indices[contact_idx]])
                # residues2.append(pdb2.atoms[pdb2_cb_indices[pdb2_index]].residue_number)
                # for pdb1_index in chain_query[contact_idx]:
                for chain_idx in contacts:
                    chain_atoms.append(self.atoms[chain_cb_indices[chain_idx]])

        return chain_atoms, all_contact_atoms  # Todo return as interface pairs?

    def get_asu(self, chain=None, extra=False):
        """Return the atoms involved in the ASU with the provided chain

        Keyword Args:
            chain=None (str): The identity of the target ASU. By default the first Chain is chosen
            extra=False (bool): If True, search for additional contacts outside the ASU, but in contact ASU
        Returns:
            (list): List of atoms involved in the identified asu
        """
        # self.get_entity_info_from_atoms()
        if not chain:
            chain = self.chain_id_list[0]

        def get_unique_contacts(chain, entity=0, iteration=0, extra=False, partner_entity=None):
            unique_chains_entity = {}
            # unique_chains_entity, chain_entity, iteration = {}, None, 0
            while not unique_chains_entity:
                self.log.debug(iteration, chain)
                if iteration != 0:  # search through the chains found in an entity
                    chain = self.entity_d[entity]['chains'][iteration]
                    self.log.debug(chain)
                chain_interface_atoms, all_contacting_interface_atoms = self.chain_interface_contacts(chain, gly_ca=True)
                self.log.debug(self.entities)
                interface_d = {}
                for atom in all_contacting_interface_atoms:
                    if atom.chain not in interface_d:
                        interface_d[atom.chain] = [atom]
                    else:
                        interface_d[atom.chain].append(atom)
                self.log.debug(interface_d)
                # deepcopy(interface_d)
                partner_interface_d, self_interface_d = {}, {}
                for _chain in self.entity_d[entity]['chains']:
                    if _chain != chain:
                        if _chain in interface_d:
                            self_interface_d[_chain] = interface_d[_chain]
                partner_interface_d = {_chain: interface_d[_chain] for _chain in interface_d
                                       if _chain not in self_interface_d}

                if not partner_entity:  # if an entity in particular is desired as in the extras recursion
                    partner_entity = set(self.entity_d.keys()) - {entity}

                if not extra:
                    # Find the top contacting chain from each unique partner entity
                    for p_entity in partner_entity:
                        max_contact, max_contact_chain = 0, None
                        for _chain in partner_interface_d:
                            self.log.debug('Partner: %s' % _chain)
                            if _chain not in self.entity_d[p_entity]['chains']:
                                continue  # ensure that the chain is relevant to this entity
                            if len(partner_interface_d[_chain]) > max_contact:
                                self.log.debug('Partner GREATER!: %s' % _chain)
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
                            if partner_chain in self.entity_d[p_entity]['chains']:
                                self.log.debug(partner_chain)
                                partner_chains_first_entity_contact = \
                                    get_unique_contacts(partner_chain, entity=p_entity,
                                                        partner_entity=entity)
                                self.log.info('Partner entity %s, original chain contacts: %s' %
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
                    self.log.info('All original chain contacts: %s' % extra_first_entity_chains)
                    all_asu_chains = list(set(first_entity_chain_contacts)) + extra_first_entity_chains
                    # Todo entity_from_chain returns Entity now
                    unique_chains_entity = {_chain: self.entity_from_chain(_chain) for _chain in all_asu_chains}
                    # need to make sure that the partner entity chains are all contacting as well...
                    # for chain in found_chains:
                self.log.info('partners: %s' % unique_chains_entity)
                iteration += 1
            return list(unique_chains_entity.keys())
            # return list(set(first_entity_chain_contacts)) + extra_first_entity_chains
            # return unique_chains_entities

        # Todo entity_from_chain returns Entity now
        unique_chains = get_unique_contacts(chain, entity=self.entity_from_chain(chain), extra=extra)

        asu = self.chain(chain).get_atoms()
        for atoms in [self.chain(partner_chain).get_atoms() for partner_chain in unique_chains]:
            asu.extend(atoms)

        return asu

    def return_asu(self, chain='A'):  # , outpath=None):
        """Returns the ASU as a new PDB object. See self.get_asu() for method"""
        asu_pdb = PDB.from_atoms(deepcopy(self.get_asu(chain=chain)), metadata=self)
        # asu_pdb.copy_metadata(self)

        return asu_pdb

        # if outpath:
        #     asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(self.filepath))[0] + '.pdb')
        #     # asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(file))[0] + '_%s' % 'asu.pdb')
        # else:
        #     asu_file_name = os.path.splitext(self.filepath)[0] + '_asu.pdb'

        # asu_pdb.write(asu_file_name, cryst1=asu_pdb.cryst)
        #
        # return asu_file_name

    def __len__(self):
        return len([0 for residue in self.residues])

    def scout_symmetry(self):
        self.find_chain_symmetry()
        self.find_max_chain_symmetry()

    def find_chain_symmetry(self):
        """Search for the chains involved in a complex using a truncated make_symmdef_file.pl script

        Requirements - all chains are the same length
        This script translates the PDB center of mass to the origin then uses quaternion geometry to solve for the rotations
        which superimpose chains provided by -i onto a designated chain (usually A). It returns the order of the rotation
        as well as the axis along which the rotation must take place. The axis of the rotation only needs to be translated
        to the center of mass to recapitulate the specific symmetry operation.

        > perl $SymDesign/dependencies/rosetta/sdf/scout_symmdef_file.pl -p 3l8r_1ho1/DEGEN_1_1/ROT_36_1/tx_4/1ho1_tx_4.pdb
            -i B C D E F G H

        """
        # Todo Create a temporary pdb file for the operation then remove file. This is necessary for changes since parsing
        scout_cmd = ['perl', scout_symmdef, '-p', self.filepath, '-a', self.chain_id_list[0], '-i'] + self.chain_id_list[1:]
        logger.info(subprocess.list2cmdline(scout_cmd))
        p = subprocess.run(scout_cmd, capture_output=True)
        # Todo institute a check to ensure proper output
        lines = p.stdout.decode('utf-8').strip().split('\n')
        # rotation_dict = {}
        for line in lines:
            chain = line[0]
            symmetry = int(line.split(':')[1][:6].rstrip('-fold'))
            axis = list(map(float, line.split(':')[2].strip().split()))  # emanating from origin
            self.rotation_d[chain] = {'sym': symmetry, 'axis': np.array(axis)}

    def find_max_chain_symmetry(self):
        """From a dictionary specifying the rotation axis and the symmetry order, find the unique set of significant chains
        """
        # find the highest order symmetry in the pdb
        max_sym, max_chain = 0, None
        for chain in self.rotation_d:
            if self.rotation_d[chain]['sym'] > max_sym:
                max_sym = self.rotation_d[chain]['sym']
                max_chain = chain

        self.max_symmetry = max_chain

    def is_dihedral(self):
        if not self.max_symmetry:
            self.scout_symmetry()
        # Check if PDB is dihedral, ensuring selected chain is orthogonal to maximum symmetry axis
        if len(self.chain_id_list) / self.rotation_d[self.max_symmetry]['sym'] == 2:
            for chain in self.rotation_d:
                if self.rotation_d[chain]['sym'] == 2:
                    if np.dot(self.rotation_d[self.max_symmetry]['axis'], self.rotation_d[chain]['axis']) < 0.01:
                        self.dihedral_chain = chain
                        return True
        elif 1 < len(self.chain_id_list) / self.rotation_d[self.max_symmetry]['sym'] < 2:
            logger.critical('The symmetry of PDB %s is malformed! Highest symmetry (%d) is less than 2x greater than '
                            'the number of chains (%d)!'
                            % (self.name, self.rotation_d[self.max_symmetry]['sym'], len(self.chain_id_list)))
            return False
        else:
            return False

    def make_sdf(self, out_path=os.getcwd(), **kwargs):  # modify_sym_energy=False, energy=2):
        """Use the make_symmdef_file.pl script from Rosetta on an input structure

        perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb -i B -q

        Keyword Args:
            modify_sym_energy=False (bool): Whether the symmetric energy produced in the file should be modified
            energy=2 (int): The scaler to modify the energy by
        Returns:
            (str): Symmetry definition filename
        """
        self.scout_symmetry()
        if self.is_dihedral():
            chains = '%s %s' % (self.max_symmetry, self.dihedral_chain)
            kwargs['dihedral'] = True
        else:
            chains = self.max_symmetry

        sdf_cmd = ['perl', make_symmdef, '-p', self.filepath, '-a', self.chain_id_list[0], '-i', chains, '-q']
        logger.info(subprocess.list2cmdline(sdf_cmd))
        with open(out_path, 'w') as file:
            p = subprocess.Popen(sdf_cmd, stdout=file, stderr=subprocess.DEVNULL)
            p.communicate()
        assert p.returncode == 0, logger.error('%s: Symmetry Definition File generation failed' % self.filepath)

        self.format_sdf(out_path=out_path, **kwargs)  # modify_sym_energy=False, energy=2)

        return out_path

    def format_sdf(self, out_path=None, dihedral=False, modify_sym_energy=False, energy=2):
        """Ensure proper sdf formatting before proceeding"""
        subunits, virtuals, jumps_com, jumps_subunit, trunk = [], [], [], [], []
        with open(out_path, 'r+') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('xyz'):
                    virtual = lines[i].split()[1]
                    if virtual.endswith('_base'):
                        subunits.append(virtual)
                    else:
                        virtuals.append(virtual.lstrip('VRT'))
                    # last_vrt = i + 1
                elif lines[i].startswith('connect_virtual'):
                    jump = lines[i].split()[1].lstrip('JUMP')
                    if jump.endswith('_to_com'):
                        jumps_com.append(jump[:-7])
                    elif jump.endswith('_to_subunit'):
                        jumps_subunit.append(jump[:-11])
                    else:
                        trunk.append(jump)
                    last_jump = i + 1  # find index of lines where the VRTs and connect_virtuals end. The "last jump"

            assert set(trunk) - set(virtuals) == set(), logger.error('%s: Symmetry Definition File VRTS are malformed'
                                                                     % self.filepath)
            assert len(self.chain_id_list) == len(subunits), logger.error('%s: Symmetry Definition File VRTX_base are '
                                                                          'malformed' % self.filepath)

            # if len(chains) > 1:  #
            if dihedral:
                # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
                virtuals = [virtual for virtual in virtuals if len(virtual) > 1]  # subunit_
            else:
                if '' in virtuals:
                    virtuals.remove('')

            jumps_com_to_add = set(virtuals) - set(jumps_com)
            count = 0
            if jumps_com_to_add != set():
                for jump_com in jumps_com_to_add:
                    lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_com VRT%s VRT%s_base\n'
                                 % (jump_com, jump_com, jump_com))
                    count += 1
                lines[-2] = lines[-2].strip() + (len(jumps_com_to_add) * ' JUMP%s_to_subunit') \
                            % tuple(jump_subunit for jump_subunit in jumps_com_to_add)
                lines[-2] += '\n'

            jumps_subunit_to_add = set(virtuals) - set(jumps_subunit)
            if jumps_subunit_to_add != set():
                for jump_subunit in jumps_subunit_to_add:
                    lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_subunit VRT%s_base SUBUNIT\n'
                                 % (jump_subunit, jump_subunit))
                    count += 1
                lines[-1] = lines[-1].strip() + (len(jumps_subunit_to_add) * ' JUMP%s_to_subunit') \
                            % tuple(jump_subunit for jump_subunit in jumps_subunit_to_add)
                lines[-1] += '\n'
            if modify_sym_energy:
                # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits,
                # so num_subunits - 1
                new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes that subunits are read in alphanumerical order
                new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
                lines[1] = new_energy + '\n'

            file.seek(0)
            for line in lines:
                file.write(line)
            file.truncate()
            if count != 0:
                logger.warning('%s: Symmetry Definition File for %s missing %d lines, fix was attempted. Modelling may be '
                               'affected for pose' % (os.path.dirname(self.filepath), os.path.basename(self.filepath), count))

    @staticmethod
    def parse_cryst_record(cryst1_string):
        """Get the unit cell length, height, width, and angles alpha, beta, gamma and the space group
        Returns:
            (tuple[list, str])
        """
        try:
            a = float(cryst1_string[6:15].strip())
            b = float(cryst1_string[15:24].strip())
            c = float(cryst1_string[24:33].strip())
            ang_a = float(cryst1_string[33:40].strip())
            ang_b = float(cryst1_string[40:47].strip())
            ang_c = float(cryst1_string[47:54].strip())
        except ValueError:
            a, b, c = 0.0, 0.0, 0.0
            ang_a, ang_b, ang_c = a, b, c
        space_group = cryst1_string[55:66].strip()  # not in
        return [a, b, c, ang_a, ang_b, ang_c], space_group

    # @staticmethod
    # def create_entity(representative_chain=None, chains=None, entity_name=None, uniprot_id=None):
    #     """Create an Entity
    #
    #     Keyword Args:
    #         representative_chain=None (str): The name of the chain to represent the Entity
    #         chains=None (list): A list of all chains that match the Entity
    #         entity_name=None (str): The name for the Entity. Typically PDB.name is used to make PDB compatible form
    #         PDB EntryID_EntityID
    #         uniprot_id=None (str): The unique UniProtID for the Entity
    #     """
    #     return Entity.from_representative(representative_chain=representative_chain, chains=chains,
    #                                       name=entity_name, uniprot_id=uniprot_id)


def generate_sequence_template(pdb_file):
    pdb = PDB.from_file(pdb_file)
    sequence = SeqRecord(Seq(''.join(pdb.atom_sequences.values()), 'Protein'), id=pdb.filepath)
    sequence_mask = copy(sequence)
    sequence_mask.id = 'design_selector'
    sequences = [sequence, sequence_mask]
    return write_fasta(sequences, file_name='%s_design_selector_sequence' % os.path.splitext(pdb.filepath)[0])
