import copy
import math
import os
import subprocess
from collections.abc import Iterable
from copy import copy, deepcopy
from itertools import chain as iter_chain  # repeat,
from shutil import move

import numpy as np
from sklearn.neighbors import BallTree
from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1_extended

from PathUtils import orient_exe_path, orient_log_file, orient_dir  # reference_aa_file, scout_symmdef, make_symmdef
from Query.PDB import get_pdb_info_by_entry, retrieve_entity_id_by_sequence
from Structure import Structure, Chain, Entity, Atom, Residues, Structures, superposition3d
from SymDesignUtils import remove_duplicates, start_log, DesignError, split_interface_residues
from utils.SymmetryUtils import valid_subunit_number, multicomponent_valid_subunit_number

logger = start_log(name=__name__)


class PDB(Structure):
    """The base object for PDB file reading and Atom manipulation
    Can pass atoms, residues, chains, entities, coords, metadata (PDB), name, seqres, multimodel, pose_format,
    and solve_discrepancy to initialize
    """
    def __init__(self, file=None, atoms=None, residues=None, chains=None, entities=None, coords=None, metadata=None,
                 log=False, **kwargs):
        # let structure start a log if log is False
        super().__init__(log=log, **kwargs)
        self.api_entry = None
        # {'entity': {1: {'A', 'B'}, ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
        #  'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
        self.atom_sequences = {}  # ATOM record sequence - {chain: 'AGHKLAIDL'}
        self.chain_id_list = []  # unique chain IDs in PDB Todo refactor
        self.chains = []
        self.cryst = kwargs.get('cryst', None)  # {space: space_group, a_b_c: (a, b, c), ang_a_b_c: (ang_a, _b, _c)}
        self.cryst_record = kwargs.get('cryst_record', None)
        self.dbref = {}  # {'chain': {'db: 'UNP', 'accession': P12345}, ...}
        self.design = kwargs.get('design', False)  # assume not a design unless explicitly found to be a design
        self.entities = []
        self.entity_d = {}  # {1: {'chains': [Chain objs], 'seq': 'GHIPLF...', 'representative': 'A'}
        # ^ ZERO-indexed for recap project!!!
        self.filepath = file  # PDB filepath if instance is read from PDB file
        self.header = []
        # self.reference_aa = None  # object for reference residue coordinates
        self.resolution = kwargs.get('resolution', None)
        self.reference_sequence = {}  # SEQRES or PDB API entries. key is chainID, value is 'AGHKLAIDL'
        # self.sasa_chain = []
        # self.sasa_residues = []
        # self.sasa = []
        self.space_group = kwargs.get('space_group', None)
        self.structure_containers.extend(['chains', 'entities'])
        self.uc_dimensions = []

        if file:
            if entities is not None:  # if no entities are requested a False argument could be provided
                kwargs['entities'] = entities
            if chains is not None:  # if no chains are requested a False argument could be provided
                kwargs['chains'] = chains
            self.readfile(**kwargs)
        else:
            if atoms is not None:
                if coords is None:
                    raise DesignError('Can\'t initialize Structure with Atom objects without passing coords! Pass '
                                      'desired coords.')
                self.chain_id_list = remove_duplicates([atom.chain for atom in atoms])
                self.process_pdb(atoms=atoms, coords=coords, **kwargs)
            elif residues:
                if coords is None:
                    try:
                        coords = np.concatenate([residue.coords for residue in residues])
                    except AttributeError:
                        raise DesignError('Without passing coords, can\'t initialize Structure with Residue objects '
                                          'lacking coords! Either pass Residue objects with coords or pass coords.')
                self.chain_id_list = remove_duplicates([residue.chain for residue in residues])
                self.process_pdb(residues=residues, coords=coords, **kwargs)
            # Todo add residues, atoms back to kwargs?
            elif chains:
                self.process_pdb(chains=chains, entities=entities, **kwargs)
            elif entities:
                self.process_pdb(entities=entities, chains=chains, **kwargs)
            # # elif isinstance(chains, (list, Structures)) and chains:
            # elif isinstance(chains, list) and chains:
            #     atoms, residues = [], []
            #     for chain in chains:
            #         atoms.extend(chain.atoms)
            #         residues.extend(chain.residues)
            #     self.atom_indices = list(range(len(atoms)))
            #     self.residue_indices = list(range(len(residues)))
            #     self.atoms = atoms
            #     residues = Residues(residues)
            #     # have to copy Residues object to set new attributes on each member Residue
            #     self.residues = copy(residues)
            #     # set residue attributes, index according to new Atoms/Coords index
            #     self.set_residues_attributes(_atoms=self._atoms)  # , _coords=self._coords) <-done in set_coords
            #     self._residues.reindex_residue_atoms()
            #     self.set_coords(coords=np.concatenate([chain.coords for chain in chains]))
            #
            #     self.chains = copy(chains)
            #     self.copy_structures()
            #     # available_chain_ids = (first + second for first in [''] + list(PDB.available_letters)
            #     #                        for second in PDB.available_letters)
            #     # for idx, chain in enumerate(self.chains):
            #     #     chain.chain_id = next(available_chain_ids)
            #     # self.chain_id_list = [chain.name for chain in self.chains]  # ([res.chain for res in residues])
            #
            #     self.chains[0].start_indices(dtype='residue', at=0)
            #     self.chains[0].start_indices(dtype='atom', at=0)
            #     for prior_idx, chain in enumerate(self.chains[1:]):
            #         chain.start_indices(dtype='residue', at=self.chains[prior_idx].residue_indices[-1] + 1)
            #         chain.start_indices(dtype='atom', at=self.chains[prior_idx].atom_indices[-1] + 1)
            #     # set the arrayed attributes for all PDB containers
            #     self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
            #     # rename chains
            #     self.reorder_chains()
            #
            #     if not kwargs.get('pose_format', True):
            #         self.renumber_structure()
            #         # self.create_entities()
            #
            # # elif isinstance(entities, (list, Structures)) and entities:
            # elif isinstance(entities, list) and entities:
            #     # there was a strange error when this function was passed three entities, 2 and 3 were the same,
            #     # however, when clashes were checked, 2 was clashing with itself as it was referencing residues from 3,
            #     # while 3 was clashing with itself as it was referencing residues from 2. watch out
            #     atoms, residues = [], []  # chains = []
            #     for entity in entities:  # grab only the Atom and Residue objects representing the Entity
            #         atoms.extend(entity.atoms)
            #         residues.extend(entity.residues)
            #         # chains.extend(entity.chains)  # won't be included in the main PDB object
            #     self.atom_indices = list(range(len(atoms)))
            #     self.residue_indices = list(range(len(residues)))
            #     self.atoms = atoms
            #     residues = Residues(residues)
            #     # have to copy Residues object to set new attributes on each member Residue
            #     self.residues = copy(residues)
            #     # set residue attributes, index according to new Atoms/Coords index
            #     self.set_residues_attributes(_atoms=self._atoms)  # , _coords=self._coords) <-done in set_coords
            #     self._residues.reindex_residue_atoms()
            #     self.set_coords(coords=np.concatenate([entity.coords for entity in entities]))
            #
            #     self.entities = copy(entities)  # copy the passed Structure list
            #     # self.chains = copy(chains)
            #     self.copy_structures()  # copy all individual Structures in the Structure list. In this case entities
            #     # if chains:
            #     #     self.log.warning('The passed Entities hold associated Chain Structures. These Chains are being '
            #     #                      'removed as the coordinates are untracked and transformations will result in '
            #     #                      'unpredictable behavior. This may be fixed in future versions. For now, you can '
            #     #                      'regenerate them by calling Entity.make_oligomer() with the required arguments.')
            #     #     for entity in self.entities:
            #     #         entity.chains.clear()
            #             # for chain in entity.chains[1:]:  # remove every Chain other than index 0 which is rep
            #             #     entity.chains.remove(chain)
            #
            #         # self.log.warning('The passed Entities hold associated Chain Structures. Ownership of these Chains '
            #         #                  'is being transferred to the respective Entity which will control their '
            #         #                  'coordinates in case transformations are applied.')
            #         # for entity in self.entities:
            #         #     entity.set_up_captain_chain()
            #     # print('Entity %s start indices' % self.entities[0].name, self.entities[0].atom_indices)
            #     self.entities[0].start_indices(dtype='residue', at=0)
            #     self.entities[0].start_indices(dtype='atom', at=0)
            #     for prior_idx, entity in enumerate(self.entities[1:]):
            #         entity.start_indices(dtype='residue', at=self.entities[prior_idx].residue_indices[-1] + 1)
            #         entity.start_indices(dtype='atom', at=self.entities[prior_idx].atom_indices[-1] + 1)
            #     # set the arrayed attributes for all PDB containers (chains, entities)
            #     self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
            #     # set each successive Entity to have an incrementally higher chain id
            #     available_chain_ids = self.return_chain_generator()
            #     for idx, entity in enumerate(self.entities):
            #         # print('Entity %s update indices' % entity.name, entity.atom_indices)
            #         entity.chain_id = next(available_chain_ids)
            #         self.log.debug('Entity %s new chain identifier %s' % (entity.name, entity.residues[0].chain))
            #     # because we don't care for chains attributes (YET) we update after everything is set
            #     # self.chains = chains
            #     # self.reorder_chains()
            #     # self.chain_id_list = [chain.name for chain in self.chains]
            #     # self.chain_id_list = [chain.name for chain in chains]
            #     if not kwargs.get('pose_format', True):
            #         self.renumber_structure()
            else:
                raise DesignError('The PDB object could not be initialized due to missing/malformed arguments')
            if metadata and isinstance(metadata, PDB):
                self.copy_metadata(metadata)

    @classmethod
    def from_file(cls, file, **kwargs):
        return cls(file=file, **kwargs)

    @classmethod
    def from_chains(cls, chains, **kwargs):
        return cls(chains=chains, **kwargs)

    @classmethod
    def from_entities(cls, entities, **kwargs):
        return cls(entities=entities, **kwargs)

    @property
    def number_of_chains(self):
        return len(self.chains)

    @property
    def symmetry(self):
        return {'symmetry': self.space_group, 'uc_dimensions': self.uc_dimensions, 'cryst_record': self.cryst_record,
                'cryst': self.cryst}  # , 'max_symmetry': self.max_symmetry}

    # def return_transformed_copy(self, **kwargs):
    #     new_pdb = super().return_transformed_copy(**kwargs)
    #     # this shouldn't be required as the set of self._coords.coords updates all coords references
    #     # new_pdb.update_attributes(coords=new_pdb._coords)
    #
    #     return new_pdb

    # def set_chain_attributes(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all Chains in the Structure"""
    #     for chain in self.chains:
    #         for kwarg, value in kwargs.items():
    #             setattr(chain, kwarg, value)
    #
    # def set_entity_attributes(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all Chains in the Structure"""
    #     for chain in self.chains:
    #         for kwarg, value in kwargs.items():
    #             setattr(chain, kwarg, value)

    def get_uc_dimensions(self):
        return list(self.cryst['a_b_c']) + list(self.cryst['ang_a_b_c'])

    def copy_metadata(self, other):
        temp_metadata = \
            {'api_entry': other.__dict__['api_entry'],
             'cryst_record': other.__dict__['cryst_record'],
             'cryst': other.__dict__['cryst'],
             'design': other.__dict__['design'],
             'entity_d': other.__dict__['entity_d'],  # Todo
             '_name': other.__dict__['_name'],
             'space_group': other.__dict__['space_group'],
             'uc_dimensions': other.__dict__['uc_dimensions'],
             'header': other.__dict__['header'],
             # 'reference_aa': other.__dict__['reference_aa'],
             'resolution': other.__dict__['resolution'],
             'rotation_d': other.__dict__['rotation_d'],
             'max_symmetry': other.__dict__['max_symmetry'],
             'dihedral_chain': other.__dict__['dihedral_chain'],
             }
        # temp_metadata = copy(other.__dict__)
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
        # self.cb_coords = pdb.cb_coords
        # self.bb_coords = pdb.bb_coords

    def readfile(self, pdb_lines=None, **kwargs):  # pose_format=True,
        """Reads .pdb file and populates PDB instance"""
        if not pdb_lines:
            with open(self.filepath, 'r') as f:
                pdb_lines = f.readlines()

        if not self.name:
            # formatted_filename = os.path.splitext(os.path.basename(self.filepath))[0].replace('pdb', '')
            # underscore_idx = formatted_filename.rfind('_') if formatted_filename.rfind('_') != -1 else None
            # self.name = formatted_filename[:underscore_idx]
            self.name = os.path.splitext(os.path.basename(self.filepath))[0].replace('pdb', '').lower()

        seq_res_lines = []
        multimodel, start_of_new_model = False, False
        model_chain_id, curr_chain_id = None, None
        entity = None
        coords, atom_info = [], []
        atom_idx = 0
        matrices = []
        current_operation = -1
        for line in pdb_lines:
            if line[:4] == 'ATOM' or line[17:20] == 'MSE' and line[:6] == 'HETATM':
                alt_location = line[16:17].strip()
                # if remove_alt_location and alt_location not in ['', 'A']:
                if alt_location not in ['', 'A']:
                    continue
                number = int(line[6:11])
                atom_type = line[12:16].strip()
                if line[17:20] == 'MSE':
                    residue_type = 'MET'
                    if atom_type == 'SE':
                        atom_type = 'SD'  # change type from Selenium to Sulfur delta
                else:
                    residue_type = line[17:20].strip()
                if multimodel:
                    if start_of_new_model or line[21:22] != curr_chain_id:
                        curr_chain_id = line[21:22]
                        model_chain_id = next(available_chain_ids)
                        start_of_new_model = False
                    chain = model_chain_id
                else:
                    chain = line[21:22]
                residue_number = int(line[22:26])
                code_for_insertion = line[26:27].strip()
                occ = float(line[54:60])
                temp_fact = float(line[60:66])
                element_symbol = line[76:78].strip()
                atom_charge = line[78:80].strip()
                if chain not in self.chain_id_list:
                    self.chain_id_list.append(chain)
                # prepare the atomic coordinates for addition to numpy array
                atom_info.append((atom_idx, number, atom_type, alt_location, residue_type, chain, residue_number,
                                  code_for_insertion, occ, temp_fact, element_symbol, atom_charge))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                atom_idx += 1
            elif line[:5] == 'MODEL':
                # start_of_new_model signifies that the next line comes after a new model
                start_of_new_model = True
                if not multimodel:
                    multimodel = True
                    available_chain_ids = self.return_chain_generator()
            # elif pose_format:
            #     continue
            elif line[:6] == 'SEQRES':
                seq_res_lines.append(line[11:])
            elif line[:18] == 'REMARK 350   BIOMT':
                # integration of the REMARK 350 BIOMT
                # REMARK 350
                # REMARK 350 BIOMOLECULE: 1
                # REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: TRIMERIC
                # REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: TRIMERIC
                # REMARK 350 SOFTWARE USED: PISA
                # REMARK 350 TOTAL BURIED SURFACE AREA: 6220 ANGSTROM**2
                # REMARK 350 SURFACE AREA OF THE COMPLEX: 28790 ANGSTROM**2
                # REMARK 350 CHANGE IN SOLVENT FREE ENERGY: -42.0 KCAL/MOL
                # REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B, C
                # REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
                # REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
                # REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
                # tokens = line.split()
                rem, tag, biomt, operation_number, x, y, z, tx = line.split()
                if operation_number != current_operation:  # we reached a new transformation matrix
                    current_operation = operation_number
                    matrices.append([])
                # add the transformation to the current matrix
                matrices[-1].append(list(map(float, [x, y, z, tx])))
            elif line[:5] == 'DBREF':
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
            elif line[:5] == 'SCALE':
                self.header.append(line.strip())
            elif line[:6] == 'CRYST1':
                self.header.append(line.strip())
                self.cryst_record = line.strip()
                self.uc_dimensions, self.space_group = self.parse_cryst_record(self.cryst_record)
                a, b, c, ang_a, ang_b, ang_c = self.uc_dimensions
                self.cryst = {'space': self.space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

        self.log.debug('File found with Multimodel: %s, Chains: %s' % (multimodel, ','.join(self.chain_id_list)))
        if not atom_info:
            raise DesignError('The file %s has no atom records!' % self.filepath)
        self.process_pdb(atoms=[Atom.from_info(*info) for info in atom_info], coords=coords,
                         seqres=seq_res_lines, multimodel=multimodel, **kwargs)  # pose_format=pose_format,

    def process_pdb(self, atoms=None, residues=None, coords=None, chains=True, entities=True,
                    seqres=None, multimodel=False, pose_format=True, solve_discrepancy=True, **kwargs):
        #           reference_sequence=None
        """Process Structure Atoms, Residues, Chain, and Entity to compliant Structure objects"""
        if atoms:
            # create Atoms object and Residue objects
            self.set_atoms(atoms)
        if residues:  # Todo ensure that atoms is also None?
            # sets Atoms and Residues
            self.set_residue_slice(residues)
            # self.set_residues(residues)

        if coords is not None:
            # inherently replace the Atom and Residue Coords
            self.set_coords(coords)

        if isinstance(chains, (list, Structures)) or isinstance(entities, (list, Structures)):  # create from existing
            atoms, residues = [], []
            # add lists together, only one is populated from class construction
            structures = ([] if not isinstance(chains, (list, Structures)) else chains) + \
                         ([] if not isinstance(entities, (list, Structures)) else entities)
            for structure in structures:
                atoms.extend(structure.atoms)
                residues.extend(structure.residues)
            self.atom_indices = list(range(len(atoms)))
            self.residue_indices = list(range(len(residues)))
            self.atoms = atoms
            residues = Residues(residues)
            # have to copy Residues object to set new attributes on each member Residue
            self.residues = copy(residues)
            # set residue attributes, index according to new Atoms/Coords index
            self.set_residues_attributes(_atoms=self._atoms)  # , _coords=self._coords) <-done in set_coords
            self._residues.reindex_residue_atoms()
            self.set_coords(coords=np.concatenate([structure.coords for structure in structures]))

        if chains:
            if isinstance(chains, list):  # create the instance from existing chains
                self.chains = copy(chains)  # copy the passed chains list
                self.copy_structures()  # copy all individual Structures in Structure container attributes
                self.chains[0].start_indices(dtype='residue', at=0)
                self.chains[0].start_indices(dtype='atom', at=0)
                for prior_idx, chain in enumerate(self.chains[1:]):
                    chain.start_indices(dtype='residue', at=self.chains[prior_idx].residue_indices[-1] + 1)
                    chain.start_indices(dtype='atom', at=self.chains[prior_idx].atom_indices[-1] + 1)
                # set the arrayed attributes for all PDB containers
                self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
                # rename chains
                self.reorder_chains()
            else:  # create Chains from Residues
                if multimodel:  # discrepancy is not possible
                    self.create_chains(solve_discrepancy=False)
                else:
                    self.create_chains(solve_discrepancy=solve_discrepancy)
                self.log.debug('New Chains: %s' % ','.join(self.chain_id_list))

        if seqres:
            self.parse_seqres(seqres)
        else:  # elif reference_sequence:
            self.reference_sequence = {chain_id: None for chain_id in self.chain_id_list}
            self.design = True

        if entities:
            if isinstance(entities, list):  # create the instance from existing entities
                self.entities = copy(entities)  # copy the passed entities list
                self.copy_structures()  # copy all individual Structures in Structure container attributes
                self.entities[0].start_indices(dtype='residue', at=0)
                self.entities[0].start_indices(dtype='atom', at=0)
                for prior_idx, entity in enumerate(self.entities[1:]):
                    entity.start_indices(dtype='residue', at=self.entities[prior_idx].residue_indices[-1] + 1)
                    entity.start_indices(dtype='atom', at=self.entities[prior_idx].atom_indices[-1] + 1)
                # set the arrayed attributes for all PDB containers (chains, entities)
                self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
                # set each successive Entity to have an incrementally higher chain id
                available_chain_ids = self.return_chain_generator()
                for idx, entity in enumerate(self.entities):
                    # print('Entity %s update indices' % entity.name, entity.atom_indices)
                    entity.chain_id = next(available_chain_ids)
                    self.log.debug('Entity %s new chain identifier %s' % (entity.name, entity.residues[0].chain))
                # because we don't care for chains attributes (YET) we update after everything is set
                # self.chains = chains
                # self.reorder_chains()
                # self.chain_id_list = [chain.name for chain in self.chains]
                # self.chain_id_list = [chain.name for chain in chains]
            else:
                # create Entities from Chain.Residues
                self.create_entities(**kwargs)

        if pose_format:
            self.renumber_structure()
        # if self.design:  # Todo maybe??
        #     self.process_symmetry()
        # self.entities.make_oligomers()  # Todo institute if needed
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
                if residue.title() in protein_letters_3to1_extended:
                    self.reference_sequence[chain][i] = protein_letters_3to1_extended[residue.title()]
                # except KeyError:
                else:
                    if residue.title() == 'Mse':
                        self.reference_sequence[chain][i] = 'M'
                    else:
                        self.reference_sequence[chain][i] = '-'
            self.reference_sequence[chain] = ''.join(self.reference_sequence[chain])

    # def read_atom_list(self, atom_list, store_cb_and_bb_coords=False):
    #     """Reads a python list of Atoms and feeds PDB instance updating chain info"""
    #     if store_cb_and_bb_coords:
    #         chain_ids = []
    #         for atom in atom_list:
    #             self.atoms.append(atom)
    #             if atom.is_backbone():
    #                 [x, y, z] = [atom.x, atom.y, atom.z]
    #                 self.bb_coords.append([x, y, z])
    #             if atom.is_CB(InclGlyCA=False):
    #                 [x, y, z] = [atom.x, atom.y, atom.z]
    #                 self.cb_coords.append([x, y, z])
    #             if atom.chain not in chain_ids:
    #                 chain_ids.append(atom.chain)
    #         self.chain_id_list += chain_ids
    #     else:
    #         chain_ids = []
    #         for atom in atom_list:
    #             self.atoms.append(atom)
    #             if atom.chain not in chain_ids:
    #                 chain_ids.append(atom.chain)
    #         self.chain_id_list += chain_ids
    #     self.renumber_atoms()
    #     self.get_chain_sequences()
    #     # self.update_entities()  # get entity information from the PDB

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

    # def extract_CB_coords_chain(self, chain, InclGlyCA=False):
    #     return [[atom.x, atom.y, atom.z] for atom in self.atoms
    #             if atom.is_CB(InclGlyCA=InclGlyCA) and atom.chain == chain]

    # def get_CB_coords(self, ReturnWithCBIndices=False, InclGlyCA=False):
    #     coords, cb_indices = [], []
    #     for idx, atom in enumerate(self.atoms):
    #         if atom.is_CB(InclGlyCA=InclGlyCA):
    #             coords.append([atom.x, atom.y, atom.z])
    #
    #             if ReturnWithCBIndices:
    #                 cb_indices.append(idx)
    #
    #     if ReturnWithCBIndices:
    #         return coords, cb_indices
    #
    #     else:
    #         return coords

    def extract_coords_subset(self, res_start, res_end, chain_index, CA):  # Todo Depreciate
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


    def reorder_chains(self, exclude_chains=None):
        """Renames chains using PDB.available_letter. Caution, doesn't update self.reference_sequence chain info
        """
        available_chain_ids = list(self.return_chain_generator())
        if exclude_chains:
            available_chains = sorted(set(available_chain_ids).difference(exclude_chains))
        else:
            available_chains = available_chain_ids

        # Update chain_id_list, then each chain
        self.chain_id_list = available_chains[:self.number_of_chains]
        for chain, new_id in zip(self.chains, self.chain_id_list):
            chain.chain_id = new_id

        self.get_chain_sequences()

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

    def AddZAxis(self):  # Todo, modernize
        z_axis_a = Atom(1, "CA", " ", "GLY", "7", 1, " ", 0.000, 0.000, 80.000, 1.00, 20.00, "C", "")
        z_axis_b = Atom(2, "CA", " ", "GLY", "7", 2, " ", 0.000, 0.000, 0.000, 1.00, 20.00, "C", "")
        z_axis_c = Atom(3, "CA", " ", "GLY", "7", 3, " ", 0.000, 0.000, -80.000, 1.00, 20.00, "C", "")

        axis = [z_axis_a, z_axis_b, z_axis_c]
        self.atoms.extend(axis)

    def AddXYZAxes(self):  # Todo, modernize
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

    def create_chains(self, solve_discrepancy=True):
        """For all the Residues in the PDB, create Chain objects which contain their member Residues"""
        if solve_discrepancy:
            chain_idx = 0
            chain_residues = {chain_idx: [0]}  # self.residues[0].index]}  <- should always be zero
            for prior_idx, residue in enumerate(self.residues[1:]):  # start at the second index to avoid off by one
                if residue.number_pdb < self.residues[prior_idx].number_pdb \
                        or residue.chain != self.residues[prior_idx].chain:
                    # Decreased number should only happen with new chain therefore this SHOULD satisfy a malformed PDB
                    chain_idx += 1
                    chain_residues[chain_idx] = [prior_idx + 1]  # residue.index]
                    # chain_residues[chain_idx] = [residue]
                else:
                    chain_residues[chain_idx].append(prior_idx + 1)  # residue.index)
                    # chain_residues[chain_idx].append(residue)
            available_chain_ids = self.return_chain_generator()
            for idx, (chain_idx, residue_indices) in enumerate(chain_residues.items()):
                if chain_idx < len(self.chain_id_list):  # Todo this logic is flawed when chains come in out of order
                    chain_id = self.chain_id_list[chain_idx]
                    discard_chain = next(available_chain_ids)
                else:  # when there are more chains than supplied by file, chose the next available
                    chain_id = next(available_chain_ids)
                self.chains.append(Chain(name=chain_id, residue_indices=residue_indices, residues=self._residues,
                                         coords=self._coords, log=self.log))
                # self.chains[idx].set_atoms_attributes(chain=chain_id)
            self.chain_id_list = [chain.name for chain in self.chains]
        else:
            for chain_id in self.chain_id_list:
                self.chains.append(Chain(name=chain_id, coords=self._coords, log=self.log, residues=self._residues,
                                         residue_indices=[idx for idx, residue in enumerate(self.residues)
                                                          if residue.chain == chain_id]))
        self.get_chain_sequences()  # Todo maybe depreciate in favor of entities?

    def get_chains(self, names=None):
        """Retrieve Chains in PDB. Returns all by default. If a list of names is provided, the selected Chains are
        returned"""
        if names and isinstance(names, Iterable):
            return [chain for chain in self.chains if chain.name in names]
        else:
            return self.chains

    def chain(self, chain_name):
        """Return the Chain object specified by the passed chain ID. If not found, return None
        Returns:
            (Chain)
        """
        for chain in self.chains:
            if chain.name == chain_name:
                return chain
        return None

    # def get_chain_atoms(self, chain_ids):
    #     """Return list of Atoms containing the subset of Atoms that belong to the selected Chain(s)"""
    #     if not isinstance(chain_ids, list):
    #         chain_ids = list(chain_ids)
    #
    #     atoms = []
    #     for chain in self.chains:
    #         if chain.name in chain_ids:
    #             atoms.extend(chain.atoms())
    #     return atoms
        # return [atom for atom in self.atoms if atom.chain == chain_id]

    # def get_chain_residues(self, chain_id):
    #     """Return the Residues included in a particular chain"""
    #     return [residue for residue in self.residues if residue.chain == chain_id]

    def write(self, out_path=None, **kwargs):
        """Write PDB Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
        one was written

        Returns:
            (str): The name of the written file
        """
        if not kwargs.get('header') and self.cryst_record:
            kwargs['header'] = self.cryst_record

        return super().write(out_path=out_path, **kwargs)

    def get_chain_sequences(self):
        self.atom_sequences = {chain.name: chain.sequence for chain in self.chains}
        # self.atom_sequences = {chain: self.chain(chain).get_structure_sequence() for chain in self.chain_id_list}

    def orient(self, sym=None, out_dir=os.getcwd(), generate_oriented_pdb=False, log=logger):
        """Orient a symmetric PDB at the origin with it's symmetry axis canonically set on axes defined by symmetry
        file. Automatically produces files in PDB numbering for proper orient execution

        Keyword Args:
            sym=None (str): What is the symmetry of the specified PDB?
            out_dir=os.getcwd() (str): Where to save a file to disk
            generate_oriented_pdb=False (bool): Whether to save an oriented file in the out_dir
            log=logger (logging.Logger): Where to log results
        Returns:
            (Union[PDB, str]): Oriented PDB or the path if generate_oriented_pdb=True
        """
        # orient_oligomer.f program notes
        # C		Will not work in any of the infinite situations where a PDB file is f***ed up,
        # C		in ways such as but not limited to:
        # C     equivalent residues in different chains don't have the same numbering; different subunits
        # C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
        # C		of separate IDs; multiple conformations are written out for the same subunit structure
        # C		(as in an NMR ensemble), negative residue numbers, etc. etc.
        # must format the input.pdb in an acceptable manner
        if sym not in valid_subunit_number:
            raise ValueError('Symmetry %s is not a valid symmetry. Please try one of: %s' %
                             (sym, ', '.join(valid_subunit_number.keys())))

        if self.filepath:
            pdb_file_name = os.path.basename(self.filepath)
        else:
            pdb_file_name = '%s.pdb' % self.name
        # Todo change output to logger with potential for file and stdout

        # with open(orient_log, 'a+') as log_f:
        number_of_subunits = len(self.chain_id_list)
        if number_of_subunits != valid_subunit_number[sym]:
            if number_of_subunits not in multicomponent_valid_subunit_number[sym]:
                error = '%s\n Oligomer could not be oriented: It has %d subunits while a multiple of %d are expected ' \
                        'for %s symmetry\n\n' % (pdb_file_name, number_of_subunits, valid_subunit_number[sym], sym)
                raise ValueError(error)
            else:
                multicomponent = True
        else:
            multicomponent = False

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
        # self.reindex_all_chain_residues()  TODO test efficacy. It could be that this screws up more than helps
        # have to change residue numbering to PDB numbering
        if multicomponent:
            chain1 = self.chains[0]
            chain1.write(orient_input, pdb_number=True)
        else:
            self.write(orient_input, pdb_number=True)
        # self.renumber_residues_by_chain()

        p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=orient_dir)
        in_symm_file = os.path.join(orient_dir, 'symm_files', sym)
        stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
        stderr = stderr.decode()  # turn from bytes to string 'utf-8' implied
        stdout = pdb_file_name + stdout.decode()[28:]

        log.info(stdout)
        log.info('%s\n' % stderr)
        if not os.path.exists(orient_output) or os.stat(orient_output).st_size == 0:
            orient_log = os.path.join(out_dir, orient_log_file)
            error_string = 'orient_oligomer could not orient %s. Check %s for more information\n' \
                           % (pdb_file_name, orient_log)
            # Todo fix this to be more precise
            raise RuntimeError(error_string)

        if multicomponent:
            oriented_pdb = PDB.from_file(orient_output)
            _, rot, tx, _ = superposition3d(oriented_pdb.chains[0].get_cb_coords(), chain1.get_cb_coords())
            self.transform(rotation=rot, translation=tx)
            if generate_oriented_pdb:
                oriented_pdb = self.write(out_path=os.path.join(out_dir, pdb_file_name))
            else:
                oriented_pdb = self
        else:
            if generate_oriented_pdb:
                oriented_pdb = os.path.join(out_dir, pdb_file_name)
                move(orient_output, oriented_pdb)
            else:
                oriented_pdb = PDB.from_file(orient_output, name=self.name, pose_format=False, log=self.log)
        clean_orient_input_output()

        return oriented_pdb

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

        pdb_coords = self.get_coords()
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

    # def stride(self, chain=None):
    #     """Use Stride to calculate the secondary structure of a PDB.
    #
    #     Sets:
    #         Residue.secondary_structure
    #     """
    #     # REM  -------------------- Secondary structure summary -------------------  XXXX
    #     # REM                .         .         .         .         .               XXXX
    #     # SEQ  1    IVQQQNNLLRAIEAQQHLLQLTVWGIKQLQAGGWMEWDREINNYTSLIHS   50          XXXX
    #     # STR       HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  HHHHHHHHHHHHHHHHH               XXXX
    #     # REM                                                                        XXXX
    #     # SEQ  51   LIEESQN                                              57          XXXX
    #     # STR       HHHHHH                                                           XXXX
    #     # REM                                                                        XXXX
    #     # LOC  AlphaHelix   ILE     3 A      ALA     33 A                            XXXX
    #     # LOC  AlphaHelix   TRP    41 A      GLN     63 A                            XXXX
    #     # REM                                                                        XXXX
    #     # REM  --------------- Detailed secondary structure assignment-------------  XXXX
    #     # REM                                                                        XXXX
    #     # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      XXXX
    #     # ASG  ILE A    3    1    H    AlphaHelix    360.00    -29.07     180.4      XXXX
    #     # ASG  VAL A    4    2    H    AlphaHelix    -64.02    -45.93      99.8      XXXX
    #     # ASG  GLN A    5    3    H    AlphaHelix    -61.99    -39.37      82.2      XXXX
    #
    #     # ASG    Detailed secondary structure assignment
    #     #    Format:  6-8  Residue name
    #     #       10-10 Protein chain identifier
    #     #       12-15 PDB	residue	number
    #     #       17-20 Ordinal residue number
    #     #       25-25 One	letter secondary structure code	**)
    #     #       27-39 Full secondary structure name
    #     #       43-49 Phi	angle
    #     #       53-59 Psi	angle
    #     #       65-69 Residue solvent accessible area
    #
    #     current_pdb_file = self.write(out_path='stride_input-%s-%d.pdb' % (self.name, random() * 100000))
    #     stride_cmd = [stride_exe_path, current_pdb_file]
    #     #   -rId1Id2..  Read only Chains Id1, Id2 ...
    #     #   -cId1Id2..  Process only Chains Id1, Id2 ...
    #     if chain:
    #         stride_cmd.append('-c%s' % chain)
    #
    #     p = subprocess.Popen(stride_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    #     out, err = p.communicate()
    #     if out:
    #         out_lines = out.decode('utf-8').split('\n')
    #     else:
    #         self.log.warning('%s: No secondary structure assignment found with Stride' % self.name)
    #         return None
    #     # except:
    #     #     stride_out = None
    #
    #     # if stride_out is not None:
    #     #     lines = stride_out.split('\n')
    #     os.system('rm %s' % current_pdb_file)
    #
    #     residue_idx = 0
    #     # print(out_lines)
    #     residues = self.residues
    #     for line in out_lines:
    #         # residue_idx = int(line[10:15])
    #         if line[0:3] == 'ASG':
    #             # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
    #             # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
    #             # self.chain(line[9:10]).residue(int(line[10:15].strip())).secondary_structure = line[24:25]
    #             residues[residue_idx].secondary_structure = line[24:25]
    #             residue_idx += 1
    #     self.secondary_structure = [residue.secondary_structure for residue in self.residues]
    #     # self.secondary_structure = {int(line[10:15].strip()): line[24:25] for line in out_lines
    #     #                             if line[0:3] == 'ASG' and line[10:15].strip().isdigit()}

    # def get_secondary_structure(self, chain=None):  # wrapper for stride, could change program eventually
    #     self.stride()

    # def get_secondary_structure_chain(self, chain=None):
    #     if self.secondary_structure:
    #         return self.chain(chain).get_secondary_structure()
    #     else:
    #         self.fill_secondary_structure()
    #         if list(filter(None, self.secondary_structure)):  # check if there is at least 1 sec. struc assignment
    #             return self.chain(chain).get_secondary_structure()
    #         else:
    #             return None

    # def get_surface_helix_cb_indices(self, probe_radius=1.4, sasa_thresh=1):
    #     # only works for monomers or homo-complexes
    #     sasa_chain, sasa_res, sasa = self.get_sasa(probe_radius=probe_radius, sasa_thresh=sasa_thresh)
    #
    #     h_cb_indices = []
    #     stride = Stride(self.filepath, self.chain_id_list[0], stride_exe_path)
    #     stride.run()
    #     stride_ss_asg = stride.ss_asg
    #     for idx, atom in enumerate(self.atoms):
    #         # atom = self.atoms[i]
    #         if atom.is_CB():
    #             if (atom.residue_number, "H") in stride_ss_asg and atom.residue_number in sasa_res:
    #                 h_cb_indices.append(idx)
    #     return h_cb_indices

    # def get_surface_fragments(self):
    #     """Using Sasa, return the 5 residue surface fragments for each surface residue on each chain"""
    #     surface_frags = []
    #     # for (chain, res_num) in self.get_surface_residues():
    #     for res_num in self.get_surface_residues():
    #         frag_res_nums = [res_num - 2, res_num - 1, res_num, res_num + 1, res_num + 2]
    #         ca_count = 0
    #
    #         # for atom in pdb.get_chain_atoms(chain):
    #         # for atom in pdb.chain(chain):
    #         # frag_atoms = pdb.chain(chain).get_residue_atoms(numbers=frag_res_nums, pdb=True)
    #         frag_atoms = []
    #         for atom in self.get_residue_atoms(numbers=frag_res_nums):
    #             # if atom.pdb_residue_number in frag_res_nums:
    #             if atom.residue_number in frag_res_nums:
    #                 frag_atoms.append(atom)
    #                 if atom.is_CA():
    #                     ca_count += 1
    #         if ca_count == 5:
    #             # surface_frags.append(PDB.from_atoms(frag_atoms, coords=self._coords, pose_format=False,
    #                                                   entities=False, log=self.log))
    #             surface_frags.append(Structure.from_atoms(atoms=frag_atoms, coords=self._coords, log=None))
    #
    #     return surface_frags

    def mutate_residue(self, residue=None, number=None, to='ALA', **kwargs):
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Keyword Args:
            residue=None (Residue): A Residue object to mutate
            number=None (int): A Residue number to select the Residue of interest by
            to='ALA' (str): The type of amino acid to mutate to
            pdb=False (bool): Whether to pull the Residue by PDB number
        """
        delete_indices = super().mutate_residue(residue=residue, number=number, to=to, **kwargs)
        if not delete_indices:  # there are no indices
            return
        delete_length = len(delete_indices)
        # remove these indices from the Structure atom_indices (If other structures, must update their atom_indices!)
        for structures in [self.chains, self.entities]:
            for structure in structures:
                try:
                    atom_delete_index = structure._atom_indices.index(delete_indices[0])
                    for _ in iter(delete_indices):
                        structure._atom_indices.pop(atom_delete_index)
                    structure.reindex_atoms(start_at=atom_delete_index, offset=delete_length)
                except (ValueError, IndexError):  # this should happen if the Atom is not in the Structure of interest
                    continue

    def insert_residue_type(self, residue_type, at=None, chain=None):
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed!

        Args:
            residue_type (str): Either the 1 or 3 letter amino acid code for the residue in question
        Keyword Args:
            at=None (int): The pose numbered location which a new Residue should be inserted into the Structure
            chain=None (str): The chain identifier to associate the new Residue with
        """
        new_residue = super().insert_residue_type(residue_type, at=at, chain=chain)
        # If other structures, must update their atom_indices!
        residue_index = at - 1  # since at is one-indexed integer
        for structures in [self.chains, self.entities]:
            for idx, structure in enumerate(structures):
                try:  # update each Structures residue_ and atom_indices with additional indices
                    res_insert_idx = structure.residue_indices.index(residue_index)
                    structure.insert_indices(at=res_insert_idx, new_indices=[residue_index], dtype='residue')
                    atom_insert_idx = structure.atom_indices.index(new_residue.start_index)
                    structure.insert_indices(at=atom_insert_idx, new_indices=new_residue.atom_indices, dtype='atom')
                    break  # must move to the next container to update the indices by a set increment
                    # structure.residue_indices = structure.residue_indices.insert(res_insertion_idx, residue_index)
                    # for idx in reversed(new_residue.atom_indices):
                    #     structure.atom_indices = structure.atom_indices.insert(new_residue.start_index, idx)
                    # below are not necessary
                    # structure.coords_indexed_residues = \
                    #     [(res_idx, res_atom_idx) for res_idx, residue in enumerate(structure.residues)
                    #      for res_atom_idx in residue.range]
                except (ValueError, IndexError):  # this should happen if the Atom is not in the Structure of interest
                    # try:  # edge case where the index is being appended to the c-terminus
                    if residue_index - 1 == structure.residue_indices[-1] and new_residue.chain == structure.chain_id:
                        res_insert_idx, atom_insert_idx = len(structure.residue_indices), len(structure.atom_indices)
                        # res_insertion_idx = structure.residue_indices.index(residue_index - 1)
                        structure.insert_indices(at=res_insert_idx, new_indices=[residue_index], dtype='residue')
                        # atom_insertion_idx = structure.atom_indices.index(new_residue.start_index - 1)
                        structure.insert_indices(at=atom_insert_idx, new_indices=new_residue.atom_indices, dtype='atom')
                        break  # must move to the next container to update the indices by a set increment
                    # except (ValueError, IndexError):
                    # else:
                    #     continue
            # for each subsequent structure in the structure container, update the indices with the last indices from
            # the prior structure
            for prior_idx, structure in enumerate(structures[idx + 1:], idx):
                structure.start_indices(dtype='residue', at=structures[prior_idx].residue_indices[-1] + 1)
                structure.start_indices(dtype='atom', at=structures[prior_idx].atom_indices[-1] + 1)

    # def insert_residue(self, chain_id, number, residue_type):
    #     """Insert a residue into the PDB. Only works for pose_numbering (1 to N). Assumes atom numbers are properly
    #     indexed"""
    #     # Find atom insertion index, should be last atom in preceding residue
    #     if number == 1:
    #         insert_atom_idx = 0
    #     else:
    #         try:
    #             residue_atoms = self.chain(chain_id).residue(number).atoms
    #             # residue_atoms = self.get_residue_atoms(chain_id, residue_number)
    #             # if residue_atoms:
    #             insert_atom_idx = residue_atoms[0].number - 1  # subtract 1 from first atom number to get insertion idx
    #         # else:  # Atom index is not an insert operation as the location is at the C-term of the chain
    #         except AttributeError:  # Atom index is not an insert operation as the index is at the C-term
    #             # prior_index = self.getResidueAtoms(chain, residue)[0].number - 1
    #             prior_chain_length = self.chain(chain_id).residues[0].atoms[0].number - 1
    #             # chain_atoms = self.chain(chain_id).atoms
    #             # chain_atoms = self.get_chain_atoms(chain_id)
    #
    #             # use length of all prior chains + length of all_chain_atoms
    #             insert_atom_idx = prior_chain_length + self.chain(chain_id).number_of_atoms  # ()
    #             # insert_atom_idx = len(chain_atoms) + chain_atoms[0].number - 1
    #
    #         # insert_atom_idx = self.getResidueAtoms(chain, residue)[0].number
    #
    #     # Change all downstream residues
    #     for atom in self.atoms[insert_atom_idx:]:
    #         # atom.number += len(insert_atoms)
    #         # if atom.chain == chain: TODO uncomment for pdb numbering
    #         atom.residue_number += 1
    #
    #     # Grab the reference atom coordinates and push into the atom list
    #     if not self.reference_aa:
    #         self.reference_aa = PDB.from_file(reference_aa_file, log=None, entities=False)
    #     # Convert incoming aa to residue index so that AAReference can fetch the correct amino acid
    #     residue_index = protein_letters.find(protein_letters_3to1_extended.get(residue_type.title(),
    #                                                                            residue_type.upper())) + 1  # offset
    #     insert_atoms = deepcopy(self.reference_aa.chain('A').residue(residue_index).atoms)
    #
    #     raise DesignError('This function \'%s\' is currently broken' % self.insert_residue.__name__)  # TODO BROKEN
    #     for atom in reversed(insert_atoms):  # essentially a push
    #         atom.chain = chain_id
    #         atom.residue_number = residue_number
    #         atom.occ = 0
    #         self.atoms = np.concatenate((self.atoms[:insert_atom_idx], insert_atoms, self.atoms[insert_atom_idx:]))
    #
    #     self.renumber_structure()

    def delete_residue(self, chain_id, residue_number):
        # raise DesignError('This function is broken')  # TODO TEST
        # start = len(self.atoms)
        # self.log.debug(start)
        # residue = self.get_residue(chain, residue_number)
        # residue.delete_atoms()  # deletes Atoms from Residue. unneccessary?

        delete = self.chain(chain_id).residue(residue_number).atom_indices
        # Atoms() should handle all Atoms containers for the object
        self._atoms.atoms = np.delete(self._atoms.atoms, delete)
        # self.delete_atoms(residue.atoms)  # deletes Atoms from PDB
        # chain._residues.remove(residue)  # deletes Residue from Chain
        # self._residues.remove(residue)  # deletes Residue from PDB
        self.renumber_structure()
        self._residues.reindex_residue_atoms()
        # remove these indices from the Structure atom_indices (If other structures, must update their atom_indices!)
        atom_delete_index = self._atom_indices.index(delete[0])
        for iteration in range(len(delete)):
            self._atom_indices.pop(atom_delete_index)
        for structures in [self.chains, self.entities]:
            for structure in structures:
                try:
                    atom_delete_index = structure._atom_indices.index(delete[0])
                    for iteration in range(len(delete)):
                        structure._atom_indices.pop(atom_delete_index)
                except ValueError:
                    continue
        # self.log.debug('Deleted: %d atoms' % (start - len(self.atoms)))

    def retrieve_pdb_info_from_api(self):  # pdb_code=None
        """Query the PDB API for information on the PDB code found as the PDB object .name attribute

        Returns:
            (dict): {'entity': {1: {'A', 'B'}, ...},
                     'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
                     'res': resolution,
                     'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
                     }
        """
        if self.name and len(self.name) == 4:
            if not self.api_entry:
                self.api_entry = get_pdb_info_by_entry(self.name)
                if self.api_entry:
                    return True
                self.log.debug('PDB code \'%s\' was not found with the PDB API.' % self.name)
        else:
            self.log.debug('PDB code \'%s\' is not of the required format and will not be found with the PDB API.'
                           % self.name)
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

    def entity(self, entity_id: str):
        for entity in self.entities:
            if entity_id == entity.name:
                return entity
        return

    def create_entities(self, query_by_sequence=True, entity_names=None, **kwargs):
        """Create all Entities in the PDB object searching for the required information if it was not found during
        parsing. First search the PDB API if a PDB entry_id is attached to instance, next from Atoms in instance

        Keyword Args:
            query_by_sequence=True (bool): Whether the PDB API should be queried for an Entity name by matching sequence
            entity_names (list): Names explicitly passed for the Entity instances. Length must equal number of entities
        """
        self.retrieve_pdb_info_from_api()  # sets api_entry
        if not self.entity_d and self.api_entry:  # self.api_entry = {1: {'A', 'B'}, ...}
            self.entity_d = \
                {ent_number: {'chains': chains} for ent_number, chains in self.api_entry.get('entity').items()}
        else:  # still nothing, then API didn't work for pdb_name so we solve by file information
            self.get_entity_info_from_atoms()
            if entity_names:
                for idx, entity_number in enumerate(list(self.entity_d.keys())):  # make a copy as update occurs w/ iter
                    try:
                        self.entity_d[entity_names[idx]] = self.entity_d.pop(entity_number)
                        self.log.debug('Entity %d now named \'%s\', as directed by supplied entity_names'
                                       % (entity_number, entity_names[idx]))
                    except IndexError:
                        raise IndexError('The number of indices in entity_names must equal %d' % len(self.entity_d))
            elif query_by_sequence:
                for entity_number, atom_info in list(self.entity_d.items()):  # make a copy as update occurs with iter
                    pdb_api_name = retrieve_entity_id_by_sequence(atom_info['seq'])
                    if pdb_api_name:
                        pdb_api_name = pdb_api_name.lower()
                        self.entity_d[pdb_api_name] = self.entity_d.pop(entity_number)
                        self.log.info('Entity %d now named \'%s\', as found by PDB API sequence search'
                                      % (entity_number, pdb_api_name))

        # For each Entity, get the chain representative Todo choose most symmetrically average if Entity is symmetric
        for entity_name, info in self.entity_d.items():
            chains = info.get('chains')  # v make Chain objects (if they are names)
            info['chains'] = [self.chain(chain) if isinstance(chain, str) else chain for chain in chains]
            info['chains'] = [chain for chain in info['chains'] if chain]
            info['representative'] = info['chains'][0]
            accession = self.dbref.get(info['representative'].chain_id, None)
            info['accession'] = accession['accession'] if accession else accession
            # info['seq'] = info['representative'].sequence

        # self.update_entity_accession_id()  # only useful if retrieve_pdb_info_from_api() is called
        # for entity_name, info in self.entity_d.items():  # generated from a PDB API sequence search v
            entity_name = '%s_%d' % (self.name, entity_name) if isinstance(entity_name, int) else entity_name
            self.entities.append(
                Entity.from_representative(representative=info['representative'], name=entity_name, log=self.log,
                                           chains=info['chains'], uniprot_id=info['accession']))

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

    # def update_entity_d(self):
    #     """Update a complete entity_d with the required information
    #     For each Entity, gather the sequence of the chain representative"""
    #     for entity, info in self.entity_d.items():
    #         info['representative'] = info['chains'][0]  # We may get an index error someday. If so, fix upstream logic
    #         info['seq'] = info['representative'].sequence
    #
    #     self.update_entity_accession_id()

    def get_entity_info_from_atoms(self, tolerance=0.9):
        """Find all unique Entities in the input .pdb file. These are unique sequence objects

        Keyword Args:
            tolerance=0.1 (float): The acceptable difference between chains to consider them the same Entity.
            Tuning this parameter is necessary if you have chains which should be considered different entities,
            but are fairly similar. Alternatively, the use of a structural match could be used.
            For example, when each chain in an ASU is structurally deviating, but they all share the same sequence
        """
        assert tolerance <= 1, '%s tolerance cannot be greater than 1!' % self.get_entity_info_from_atoms.__name__
        entity_count = 1
        self.entity_d[entity_count] = {'chains': [self.chains[0]], 'seq': self.chains[0].sequence}
        for chain in self.chains[1:]:
            new_entity = True  # assume all chains are unique entities
            self.log.debug('Searching for matching Entities for Chain %s' % chain.name)
            for entity in self.entity_d:
                # rmsd, rot, tx, rescale = superposition3d()  # Todo implement structure check
                # if rmsd < 3:  # 3A threshold needs testing
                #     self.entity_d[entity]['chains'].append(chain)
                #     new_entity = False  # The entity is not unique, do not add
                #     break
                # check if the sequence associated with the atom chain is in the entity dictionary
                if chain.sequence == self.entity_d[entity]['seq']:
                    score = len(chain.sequence)
                else:
                    alignment = pairwise2.align.localxx(chain.sequence, self.entity_d[entity]['seq'])
                    score = alignment[0][2]  # first alignment from localxx, grab score value
                match_score = score / len(self.entity_d[entity]['seq'])  # could also use which ever sequence is greater
                length_proportion = abs(len(chain.sequence) - len(self.entity_d[entity]['seq'])) \
                    / len(self.entity_d[entity]['seq'])
                self.log.debug('Chain %s matches Entity %d with %0.2f identity and length difference of %0.2f'
                               % (chain.name, entity, match_score, length_proportion))
                if match_score >= tolerance and length_proportion <= 1 - tolerance:
                    # if number of sequence matches is > tolerance, and the length difference < tolerance
                    # the current chain is the same as the Entity, add to chains, and move on to the next chain
                    self.entity_d[entity]['chains'].append(chain)
                    new_entity = False  # The entity is not unique, do not add
                    break
            if new_entity:  # no existing entity matches, add new entity
                entity_count += 1
                self.entity_d[entity_count] = {'chains': [chain], 'seq': chain.sequence}
        self.log.debug('Entities were generated from ATOM records.')

    def get_entity_info_from_api(self):  # , pdb_code=None):  UNUSED
        """Query the PDB API for the PDB entry_ID to find the corresponding Entity information"""
        if self.retrieve_pdb_info_from_api():  # pdb_code=pdb_code):
            self.entity_d = {ent: {'chains': self.api_entry['entity'][ent]} for ent in self.api_entry['entity']}

    # def update_entity_representatives(self):
    #     """For each Entity, gather the chain representative by choosing the first chain in the file
    #     """
    #     for entity, info in self.entity_d.items():
    #         info['representative'] = info['chains'][0]  # We may get an index error someday. If so, fix upstream logic

    # def update_entity_sequences(self):
    #     """For each Entity, gather the sequence of the chain representative"""
    #     self.log.debug(self.entity_d)
    #     for entity, info in self.entity_d.items():
    #         info['seq'] = info['representative'].sequence

    # def update_entity_accession_id(self):  # UNUSED
    #     """Create a map (dictionary) between identified entities (not yet Entity objs) and their accession code
    #     If entities from psuedo generation, then may not.
    #     """
    #     # dbref will not be generated unless specified by call to API or from .pdb file
    #     if not self.dbref:  # check if from .pdb file
    #         if not self.api_entry:  # check if from API
    #             for info in self.entity_d.values():
    #                 info['accession'] = None
    #             return
    #         else:
    #             self.dbref = self.api_entry['dbref']
    #
    #     for info in self.entity_d.values():
    #         info['accession'] = self.dbref.get(info['representative'].chain_id)['accession']
    #     # self.entity_accession_map = {entity: self.dbref[self.entity_d[entity]['representative']]['accession']
    #     #                              for entity in self.entity_d}

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

    def chain_interface_contacts(self, chain, distance=8):  # Todo very similar to Pose with entities
        """Create a atom tree using CB atoms from one chain and all other atoms

        Args:
            chain (PDB): First PDB to query against
        Keyword Args:
            distance=8 (int): The distance to query in Angstroms
            gly_ca=False (bool): Whether glycine CA should be included in the tree
        Returns:
            chain_atoms, all_contact_atoms (list, list): Chain interface atoms, all contacting interface atoms
        """
        # Get CB Atom indices for the atoms CB and chain CB
        all_cb_indices = self.cb_indices
        chain_cb_indices = chain.cb_indices
        # chain_cb_indices = self.get_cb_indices_chain(chain_id, InclGlyCA=gly_ca)
        # chain_coord_indices, contact_cb_indices = [], []
        # # Find the contacting CB indices and chain specific indices
        # for i, idx in enumerate(all_cb_indices):
        #     if idx in chain_cb_indices:
        #         chain_coord_indices.append(i)
        #     else:
        #         contact_cb_indices.append(idx)

        contact_cb_indices = list(set(all_cb_indices).difference(chain_cb_indices))
        # assuming that coords is for the whole structure
        contact_coords = self.coords[contact_cb_indices]  # InclGlyCA=gly_ca)
        # all_cb_coords = self.get_cb_coords()  # InclGlyCA=gly_ca)
        # all_cb_coords = np.array(self.extract_CB_coords(InclGlyCA=gly_ca))
        # Remove chain specific coords from all coords by deleting them from numpy
        # contact_coords = np.delete(all_cb_coords, chain_coord_indices, axis=0)

        # Construct CB Tree for the chain
        chain_tree = BallTree(chain.get_cb_coords())
        # Query chain CB Tree for all contacting Atoms within distance
        chain_contact_query = chain_tree.query_radius(contact_coords, distance)
        pdb_atoms = self.atoms
        return [(pdb_atoms[chain_cb_indices[chain_idx]].residue_number,
                 pdb_atoms[contact_cb_indices[contact_idx]].residue_number)
                for contact_idx, contacts in enumerate(chain_contact_query) for chain_idx in contacts]
        # all_contact_atoms, chain_atoms = [], []
        # for contact_idx, contacts in enumerate(chain_contact_query):
        #     if chain_contact_query[contact_idx].tolist():
        #         all_contact_atoms.append(pdb_atoms[contact_cb_indices[contact_idx]])
        #         # residues2.append(pdb2.atoms[pdb2_cb_indices[pdb2_index]].residue_number)
        #         # for pdb1_index in chain_contact_query[contact_idx]:
        #         chain_atoms.extend([pdb_atoms[chain_cb_indices[chain_idx]] for chain_idx in contacts])
        #
        # return chain_atoms, all_contact_atoms

    def get_asu(self, chain=None, extra=False):
        """Return the atoms involved in the ASU with the provided chain

        Keyword Args:
            chain=None (str): The identity of the target ASU. By default the first Chain is chosen
            extra=False (bool): If True, search for additional contacts outside the ASU, but in contact ASU
        Returns:
            (list): List of atoms involved in the identified asu
        """
        if not chain:
            chain = self.chain_id_list[0]

        def get_unique_contacts(chain, entity, iteration=0, extra=False, partner_entities=None):
            """

            Args:
                chain (Chain):
                entity (Entity):
                iteration (int):
                extra:
                partner_entities (list[Entity]):

            Returns:
                (list[Entity])
            """
            unique_chains_entity = {}
            while not unique_chains_entity:
                if iteration == 0:  # use the provided chain
                    pass
                elif iteration < len(entity.chains):  # search through the chains found in an entity
                    chain = entity.chains[iteration]
                else:
                    raise DesignError('The ASU couldn\'t be found! Debugging may be required %s'
                                      % get_unique_contacts.__name__)
                iteration += 1
                self.log.debug('Iteration %d, Chain %s' % (iteration, chain.chain_id))
                chain_residue_numbers, contacting_residue_numbers = \
                    split_interface_residues(self.chain_interface_contacts(chain))

                # find all chains in contact and their corresponding atoms
                interface_d = {}
                for residue in self.get_residues(numbers=contacting_residue_numbers):
                    if residue.chain in interface_d:
                        interface_d[residue.chain].append(residue)
                    else:  # add to chain
                        interface_d[residue.chain] = [residue]
                self.log.debug('Interface chains: %s' % interface_d)

                # find all chains that are in the entity in question
                self_interface_d = {}
                for _chain in entity.chains:
                    if _chain != chain:
                        if _chain.chain_id in interface_d:
                            # {chain: [Residue, Residue, ...]}
                            self_interface_d[_chain.chain_id] = interface_d[_chain.chain_id]
                self.log.debug('Self interface chains: %s' % self_interface_d.keys())

                # all others are in the partner entity
                # {chain: int}
                partner_interface_d = {self.chain(chain_id): len(residues) for chain_id, residues in interface_d.items()
                                       if chain_id not in self_interface_d}
                self.log.debug('Partner interface chains %s' % partner_interface_d.keys())
                if not partner_entities:  # if no partner entity is specified
                    partner_entities = set(self.entities).difference({entity})
                # else:  # particular entity is desired in the extras recursion
                #     pass
                self.log.debug('Partner Entities: %s' % ', '.join(entity.name for entity in partner_entities))

                if not extra:
                    # Find the top contacting chain from each unique partner entity
                    for p_entity in partner_entities:
                        max_contacts, max_contact_chain = 0, None
                        for p_chain, contacts in partner_interface_d.items():
                            if p_chain not in p_entity.chains:  # if more than 2 Entities this is okay
                                # self.log.error('Chain %s was found in the list of partners but isn\'t in this Entity'
                                #                % p_chain.chain_id)
                                continue  # ensure that the chain is relevant to this entity
                            if contacts > max_contacts:  # length is number of atoms
                                self.log.debug('Partner GREATER!: %s' % p_chain)
                                max_contacts = contacts
                                max_contact_chain = p_chain
                            else:
                                self.log.debug('Partner LESS THAN/EQUAL: %d' % contacts)
                        if max_contact_chain:
                            unique_chains_entity[max_contact_chain] = p_entity

                    # return list(unique_chains_entity.keys())
                else:  # TODO this doesn't work yet. Solve the asu by expansion to extra contacts
                    raise DesignError('This functionality \'get_asu(extra=True)\' is not working!')
                    # partner_entity_chains_first_entity_contact_d = {} TODO define here if iterate over all entities?
                    extra_first_entity_chains, first_entity_chain_contacts = [], []
                    for p_entity in partner_entities:  # search over all entities
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
                                    get_unique_contacts(partner_chain, p_entity, partner_entities=[entity])
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
                    unique_chains_entity = {_chain: self.entity_from_chain(_chain) for _chain in all_asu_chains}
                    # need to make sure that the partner entity chains are all contacting as well...
                    # for chain in found_chains:
                self.log.info('Partner chains: %s' % unique_chains_entity.keys())

            return unique_chains_entity.keys()

        chain_of_interest = self.chain(chain)
        partner_chains = get_unique_contacts(chain_of_interest, entity=self.entity_from_chain(chain), extra=extra)
        # partner_chains = get_unique_contacts(chain, entity=self.entity_from_chain(chain), extra=extra)
        # partner_chains = get_unique_contacts(chain, entity=self.entity_from_chain(chain).name, extra=extra)

        asu_atoms = chain_of_interest.atoms
        for atoms in [chain.atoms for chain in partner_chains]:
            asu_atoms.extend(atoms)
        asu_coords = np.concatenate([chain_of_interest.coords] + [chain.coords for chain in partner_chains])

        return asu_atoms, asu_coords

    def return_asu(self, chain='A'):
        """Returns the ASU as a new PDB object. See self.get_asu() for method"""
        asu_pdb_atoms, asu_pdb_coords = self.get_asu(chain=chain)
        asu = PDB.from_atoms(atoms=deepcopy(asu_pdb_atoms), coords=asu_pdb_coords, metadata=self, log=self.log)
        asu.reorder_chains()

        return asu

        # if outpath:
        #     asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(self.filepath))[0] + '.pdb')
        #     # asu_file_name = os.path.join(outpath, os.path.splitext(os.path.basename(file))[0] + '_%s' % 'asu.pdb')
        # else:
        #     asu_file_name = os.path.splitext(self.filepath)[0] + '_asu.pdb'

        # asu_pdb.write(asu_file_name, cryst1=asu_pdb.cryst)
        #
        # return asu_file_name

    @staticmethod
    def get_cryst_record(file):
        with open(file, 'r') as f:
            for line in f.readlines():
                if line[0:6] == 'CRYST1':
                    uc_dimensions, space_group = PDB.parse_cryst_record(line.strip())
                    a, b, c, ang_a, ang_b, ang_c = uc_dimensions
                    return {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

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

    # def update_attributes(self, **kwargs):
    #     """Update PDB attributes for all member containers specified by keyword args"""
    #     # super().update_attributes(**kwargs)  # this is required to set the base Structure with the kwargs
    #     self.set_structure_attributes(self.chains, **kwargs)
    #     self.set_structure_attributes(self.entities, **kwargs)
    #
    # def copy_structures(self):
    #     super().copy_structures([self.entities, self.chains])
    def __len__(self):
        try:
            return self.number_of_residues
        except TypeError:  # This catches an empty instance with no data
            return False

    def __copy__(self):
        other = super().__copy__()
        # create a copy of all chains and entities
        # structures = [other.chains, other.entities]
        # other.copy_structures(structures)
        other.copy_structures()  # uses self.structure_containers
        # these were updated in the super().__copy__, now need to set attributes in copied chains and entities
        # other.update_attributes(residues=copy(self._residues), coords=copy(self._coords))
        # print('Updating new copy of \'%s\' attributes' % self.name)
        # This style v accomplishes the update that the super().__copy__() started using self.structure_containers
        # providing references to new, shared objects to each individual Structure container in the PDB
        other.update_attributes(_residues=other._residues, _coords=other._coords)
        # memory_l = [self, self.chains[0], self.entities[0], self.entities[0].chains[0]]
        # memory_o = [other, other.chains[0], other.entities[0], other.entities[0].chains[0]]
        # print('The id in memory of self : %s\nstored coordinates is: %s' % (memory_l, list(map(getattr, memory_l, repeat('_coords')))))
        # print('The id in memory of other: %s\nstored coordinates is: %s' % (memory_o, list(map(getattr, memory_o, repeat('_coords')))))

        # # This routine replaces the .chains container of .entities with all chains that are PDB copies
        # # UPDATE built in deepcopy would handle this perfectly fine..., it has a bit of overhead in copying though
        # for idx_ch, chain in enumerate(self.chains):
        #     for idx_ent, entity in enumerate(self.entities):
        #         if entity.is_oligomeric:
        #             # This check prevents copies when the Entity has full control over it's chains
        #             break
        #         elif chain in entity.chains:
        #             equivalent_idx = entity.chains.index(chain)
        #             other.entities[idx_ent].chains.pop(equivalent_idx)
        #             other.entities[idx_ent].chains.insert(equivalent_idx, other.chains[idx_ch])
        #             # break  # There shouldn't be any chains which belong to multiple Entities
        # memory_n = [other, other.chains[0], other.entities[0], other.entities[0].chains[0]]
        # print('The id in memory of Nothr: %s\nstored coordinates is: %s' % (memory_n, list(map(getattr, memory_n, repeat('_coords')))))

        return other


def extract_interface(pdb, chain_data_d, full_chain=True):
    """
    'interfaces': {interface_ID: {interface stats, {chain data}}, ...}
        Ex: {1: {'occ': 2, 'area': 998.23727478, 'solv_en': -11.928783903, 'stab_en': -15.481081211,
             'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104, 'int_res': {'87': 23.89, '89': 45.01, ...},
                            2: ...}},
             2: {'occ': ..., },
             'all_ids': {interface_type: [interface_id1, matching_id2], ...}
            } interface_type and id connect the interfaces that are the same, but present in multiple PISA complexes

    """
    # if one were trying to map the interface fragments created in a fragment extraction back to the pdb, they would
    # want to use the interface in interface_data and the chain id (renamed from letter to ID #) to find the chain and
    # the translation from the pisa.xml file
    # pdb_code, subdirectory = return_and_make_pdb_code_and_subdirectory(pdb_file_path)
    # out_path = os.path.join(os.getcwd(), subdirectory)
    # try:
    #     # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
    #     # This makes flexible with MySQL
    #     source_pdb = PDB(file=pdb_file_path)
    #     pisa_data = unpickle(pisa_file_path)  # Get PISA data
    #     interface_data = pisa_data['interfaces']
    #     # interface_data, chain_data = pp.parse_pisa_interfaces_xml(pisa_file_path)
    #     for interface_id in interface_data:
    #         if not interface_id.is_digit():  # == 'all_ids':
    #             continue
    # interface_pdb = PDB.PDB()
    temp_names = ('.', ',')
    interface_chain_pdbs = []
    temp_chain_d = {}
    for temp_name_idx, chain_id in enumerate(chain_data_d):
        # chain_pdb = PDB.PDB()
        chain = chain_data_d[chain_id]['chain']
        # if not chain:  # for instances of ligands, stop process, this is not a protein-protein interface
        #     break
        # else:
        if full_chain:  # get the entire chain
            interface_atoms = deepcopy(pdb.chain(chain).atoms)
        else:  # get only the specific residues at the interface
            residue_numbers = chain_data_d[chain_id]['int_res']
            interface_atoms = pdb.chain(chain).get_residue_atoms(residue_numbers)
            # interface_atoms = []
            # for residue_number in residues:
            #     residue_atoms = pdb.get_residue_atoms(chain, residue_number)
            #     interface_atoms.extend(deepcopy(residue_atoms))
            # interface_atoms = list(iter_chain.from_iterable(interface_atoms))
        chain_pdb = PDB.from_atoms(deepcopy(interface_atoms))
        # chain_pdb.read_atom_list(interface_atoms)

        rot = chain_data_d[chain_id]['r_mat']
        trans = chain_data_d[chain_id]['t_vec']
        chain_pdb.apply(rot, trans)
        chain_pdb.chain(chain).set_atoms_attributes(chain=temp_names[temp_name_idx])  # ensure that chain names are not the same
        # Todo edit this mechanism! ^
        temp_chain_d[temp_names[temp_name_idx]] = str(chain_id)
        interface_chain_pdbs.append(chain_pdb)
        # interface_pdb.read_atom_list(chain_pdb.atoms)

    interface_pdb = PDB.from_atoms(list(iter_chain.from_iterable([chain_pdb.atoms
                                                                  for chain_pdb in interface_chain_pdbs])))
    if len(interface_pdb.chain_id_list) == 2:
        for temp_name in temp_chain_d:
            interface_pdb.chain(temp_name).set_atoms_attributes(chain=temp_chain_d[temp_name])
            # Todo edit this mechanism! ^

    return interface_pdb

# ref_aa = PDB.from_file('/home/kylemeador/symdesign/data/AAreference.pdb', log=None, pose_format=False, entities=False)
# pickle_object(ref_aa.residues, name='/home/kylemeador/symdesign/data/AAreferenceResidues.pkl', out_path='')
