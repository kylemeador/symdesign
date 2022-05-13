import copy
from logging import Logger
import math
import os
import subprocess
from collections.abc import Iterable
from copy import copy, deepcopy
from glob import glob
from itertools import chain as iter_chain  # repeat,
# from random import randint
# from time import sleep
from typing import Union, Dict, Sequence, List, Container, Optional

import numpy as np
from sklearn.neighbors import BallTree
from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1_extended, protein_letters_1to3_extended

from PathUtils import orient_exe_path, orient_dir, pdb_db, qs_bio, reference_aa_file, reference_residues_pkl
from Query.PDB import get_pdb_info_by_entry, retrieve_entity_id_by_sequence, get_pdb_info_by_assembly
from SequenceProfile import generate_alignment
from Structure import Structure, Chain, Entity, Atom, Residues, Structures, superposition3d
from SymDesignUtils import remove_duplicates, start_log, DesignError, split_interface_residues, to_iterable, \
    unpickle, pickle_object
from utils.SymmetryUtils import valid_subunit_number, multicomponent_valid_subunit_number, valid_symmetries

logger = start_log(name=__name__)
seq_res_len = 52


class PDB(Structure):
    """The base object for PDB file reading and Atom manipulation
    Can pass atoms, residues, chains, entities, coords, metadata (PDB), name, seqres, multimodel, pose_format,
    and solve_discrepancy to initialize
    """
    def __init__(self, file=None, atoms=None, residues=None, chains=None, entities=None, coords=None, metadata=None,
                 log=False, **kwargs):
        # PDB defaults to Structure logger (log is False)
        super().__init__(log=log, **kwargs)
        self.api_entry = None
        # {'entity': {1: {'A', 'B'}, ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
        #  'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
        self.assembly = False
        self.atom_sequences = {}  # ATOM record sequence - {chain: 'AGHKLAIDL'}
        # self.biomt = []
        # self.biomt_header = ''
        self.chain_ids = []  # unique chain IDs
        self.chains = []
        self.cryst = kwargs.get('cryst', None)  # {space: space_group, a_b_c: (a, b, c), ang_a_b_c: (ang_a, _b, _c)}
        self.cryst_record = kwargs.get('cryst_record', None)
        self.dbref = {}  # {'chain': {'db: 'UNP', 'accession': P12345}, ...}
        self.design = kwargs.get('design', False)  # assume not a design unless explicitly found to be a design
        self.entities = []
        self.entity_info = []  # [{'chains': [Chain objs], 'seq': 'GHIPLF...', 'name': 'A'}, ...]
        # ^ ZERO-indexed for recap project!!!
        self.filepath = file  # PDB filepath if instance is read from PDB file
        self.header = []
        # self.reference_aa = None  # object for reference residue coordinates
        self.multimodel = False
        self.multimodel_chain_map = {}  # {model number: [chain_ids], ...}
        self.resolution = kwargs.get('resolution', None)
        self.reference_sequence = {}  # SEQRES or PDB API entries. key is chainID, value is 'AGHKLAIDL'
        # self.sasa_chain = []
        # self.sasa_residues = []
        # self.sasa = []
        self.space_group = kwargs.get('space_group', None)
        self.structure_containers.extend(['chains', 'entities'])
        self.uc_dimensions = kwargs.get('uc_dimensions', None)

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
                self.chain_ids = remove_duplicates([atom.chain for atom in atoms])
                self.process_pdb(atoms=atoms, coords=coords, **kwargs)
            elif residues:
                if coords is None:
                    try:
                        coords = np.concatenate([residue.coords for residue in residues])
                    except AttributeError:
                        raise DesignError('Without passing coords, can\'t initialize Structure with Residue objects '
                                          'lacking coords! Either pass Residue objects with coords or pass coords.')
                self.chain_ids = remove_duplicates([residue.chain for residue in residues])
                self.process_pdb(residues=residues, coords=coords, **kwargs)
            # Todo add residues, atoms back to kwargs?
            elif chains:
                self.process_pdb(chains=chains, entities=entities, **kwargs)
            elif entities:
                self.process_pdb(entities=entities, chains=chains, **kwargs)
                # for idx, entity in enumerate(self.entities, 1):
                #     filename = entity.write_oligomer(out_path='%s%d_post_process_pdb_oligomer-%d.pdb'
                #                                               % (entity.name, idx, randint(0, 10000)))
                #     self.log.info('Wrote %s' % filename)
                # self.log.info('After process_pdb')
                # sleep(20)
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
    def number_of_chains(self) -> int:
        """Return the number of Chain objects in the PDB"""
        return len(self.chains)

    @property
    def number_of_entities(self) -> int:
        """Return the number of Entity objects in the PDB"""
        return len(self.entities)

    @property
    def symmetry(self) -> Dict:
        """Return the symmetry parameters of the PDB"""
        sym_attrbutes = ['symmetry', 'uc_dimensions', 'cryst_record', 'cryst']  # , 'max_symmetry': self.max_symmetry}
        return {sym_attrbutes[idx]: sym_attr
                for idx, sym_attr in enumerate([self.space_group, self.uc_dimensions, self.cryst_record, self.cryst])
                if sym_attr is not None}

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

    @property
    def uc_dimensions(self) -> List:
        try:
            return self._uc_dimensions
        except AttributeError:
            self._uc_dimensions = list(self.cryst['a_b_c']) + list(self.cryst['ang_a_b_c'])
            return self._uc_dimensions

    @uc_dimensions.setter
    def uc_dimensions(self, dimensions):
        self._uc_dimensions = dimensions

    def copy_metadata(self, other):
        temp_metadata = \
            {'api_entry': other.__dict__['api_entry'],
             'cryst_record': other.__dict__['cryst_record'],
             'cryst': other.__dict__['cryst'],
             'design': other.__dict__['design'],
             'entity_info': other.__dict__['entity_info'],
             '_name': other.__dict__['_name'],
             'space_group': other.__dict__['space_group'],
             '_uc_dimensions': other.__dict__['_uc_dimensions'],
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
        # self.chain_ids = pdb.chain_ids
        self.entity_info = pdb.entity_info
        self.name = pdb.name
        self.secondary_structure = pdb.secondary_structure
        # self.cb_coords = pdb.cb_coords
        # self.bb_coords = pdb.bb_coords

    def readfile(self, pdb_lines=None, **kwargs):  # pose_format=True,
        """Reads .pdb file and populates PDB instance"""
        if not pdb_lines:
            with open(self.filepath, 'r') as f:
                pdb_lines = f.readlines()

        if os.path.splitext(self.filepath)[-1][-1].isdigit():  # pull the extension, last character
            # If not a letter, then the file is an assembly, or the filename was provided weird
            self.assembly = True

        if not self.name:
            self.name = os.path.splitext(os.path.basename(self.filepath))[0]  # .replace('pdb', '')

        seq_res_lines = []
        start_of_new_model = False
        curr_chain_id = None
        entity = None
        coords, atom_info = [], []
        atom_idx = 0
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
                if start_of_new_model:  # if self.multimodel:
                    start_of_new_model = False
                    if line[21:22] == curr_chain_id:  # chain naming is not incremental
                        curr_chain_id = line[21:22]
                        chain = next(available_chain_ids)
                    else:  # line[21:22] != curr_chain_id  Chain naming IS incremental
                        curr_chain_id, chain = line[21:22], line[21:22]
                        discard = next(available_chain_ids)  # getting rid of a chain is prudent
                    self.multimodel_chain_map[chain] = curr_chain_id
                else:
                    chain = line[21:22]
                residue_number = int(line[22:26])
                code_for_insertion = line[26:27].strip()
                occ = float(line[54:60])
                temp_fact = float(line[60:66])
                element_symbol = line[76:78].strip()
                atom_charge = line[78:80].strip()
                if chain not in self.chain_ids:
                    self.chain_ids.append(chain)
                # prepare the atomic coordinates for addition to numpy array
                atom_info.append((atom_idx, number, atom_type, alt_location, residue_type, chain, residue_number,
                                  code_for_insertion, occ, temp_fact, element_symbol, atom_charge))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                atom_idx += 1
            elif line[:5] == 'MODEL':
                # start_of_new_model signifies that the next line comes after a new model
                start_of_new_model = True
                # model_number = line[6:].strip()
                # self.multimodel_chain_map[model_number] = {}
                if not self.multimodel:
                    self.multimodel = True
                    available_chain_ids = self.return_chain_generator()
            # elif pose_format:
            #     continue
            elif line[:6] == 'SEQRES':
                seq_res_lines.append(line[11:])
            elif line[:18] == 'REMARK 350':
            # elif line[:18] == 'REMARK 350   BIOMT':
                self.biomt_header += line
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
                _, _, biomt, operation_number, x, y, z, tx = line.split()
                if biomt == 'BIOMT':
                    if operation_number != current_operation:  # we reached a new transformation matrix
                        current_operation = operation_number
                        self.biomt.append([])
                    # add the transformation to the current matrix
                    self.biomt[-1].append(list(map(float, [x, y, z, tx])))
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
                # self.entity_info[entity] = \
                    # {'chains': list(map(str.strip, line[line.rfind(':') + 1:].strip().rstrip(';').split(',')))}
                self.entity_info.append(
                    {'chains': list(map(str.strip, line[line.rfind(':') + 1:].strip().rstrip(';').split(','))),
                     'name': entity})
                entity = None
            elif line[:5] == 'SCALE':
                self.header.append(line.strip())
            elif line[:6] == 'CRYST1':
                self.header.append(line.strip())
                self.cryst_record = line  # .strip()
                self.uc_dimensions, self.space_group = self.parse_cryst_record(self.cryst_record)
                a, b, c, ang_a, ang_b, ang_c = self.uc_dimensions
                self.cryst = {'space': self.space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

        if self.multimodel:  # ensure we have a multimodel file, not just one with MODEL record
            self.multimodel = False
            for new_chain, old_chain in self.multimodel_chain_map.items():
                if new_chain != old_chain:
                    self.multimodel = True
                    self.log.debug('Multimodel file found. Original Chains: %s' % ','.join(self.multimodel_chain_map.values()))
                    break
            if self.multimodel:
                self.log.debug('Multimodel file not respected, chains are all different')
        if not atom_info:
            raise DesignError('The file %s has no atom records!' % self.filepath)
        self.process_pdb(atoms=[Atom(*info) for info in atom_info], coords=coords,
                         seqres=seq_res_lines, **kwargs)  # pose_format=pose_format,

    def process_pdb(self, atoms=None, residues=None, coords=None, chains=True, entities=True,
                    seqres=None, pose_format=True, solve_discrepancy=True, rename_chains=False, **kwargs):
        #           reference_sequence=None, multimodel=False,
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
            atoms, residues, coords = [], [], []
            # add lists together, only one is populated from class construction
            structures = ([] if not isinstance(chains, (list, Structures)) else chains) + \
                         ([] if not isinstance(entities, (list, Structures)) else entities)
            for structure in structures:
                atoms.extend(structure.atoms)
                residues.extend(structure.residues)
                coords.append(structure.coords)
            self.atom_indices = list(range(len(atoms)))
            self.residue_indices = list(range(len(residues)))
            self.atoms = atoms
            residues = Residues(residues)
            # have to copy Residues object to set new attributes on each member Residue
            self.residues = copy(residues)
            # set residue attributes, index according to new Atoms/Coords index
            self.set_residues_attributes(_atoms=self._atoms)  # , _coords=self._coords) <- done in set_coords
            # self._residues.set_attributes(_atoms=self._atoms)  # , _coords=self._coords) <- done in set_coords
            self._residues.reindex_residue_atoms()
            self.set_coords(coords=np.concatenate(coords))
            self.chain_ids = remove_duplicates([residue.chain for residue in residues])

        if chains:
            if isinstance(chains, (list, Structures)):  # create the instance from existing chains
                self.chains = copy(chains)  # copy the passed chains list
                self.copy_structures()  # copy all individual Structures in Structure container attributes
                # Reindex all residue and atom indices
                self.chains[0].reset_indices_attributes()
                self.chains[0].start_indices(dtype='residue', at=0)
                self.chains[0].start_indices(dtype='atom', at=0)
                for prior_idx, chain in enumerate(self.chains[1:]):
                    chain.reset_indices_attributes()
                    chain.start_indices(dtype='residue', at=self.chains[prior_idx].residue_indices[-1] + 1)
                    chain.start_indices(dtype='atom', at=self.chains[prior_idx].atom_indices[-1] + 1)
                # set the arrayed attributes for all PDB containers
                self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
                if rename_chains:
                    self.reorder_chains()
                # else:
                #     self.chain_ids = remove_duplicates([atom.chain for atom in atoms])
            else:  # create Chains from Residues
                if self.multimodel:  # discrepancy is not possible
                    self.create_chains(solve_discrepancy=False)
                else:
                    self.create_chains(solve_discrepancy=solve_discrepancy)
                self.log.debug('Loaded with Chains: %s' % ','.join(self.chain_ids))

        if seqres:
            self.parse_seqres(seqres)
        else:
            # Todo get the reference sequence from Entities?
            # self.reference_sequence = {chain_id: None for chain_id in self.chain_ids}
            self.design = True

        if entities:
            if isinstance(entities, (list, Structures)):  # create the instance from existing entities
                self.entities = copy(entities)  # copy the passed entities list
                self.copy_structures()  # copy all individual Structures in Structure container attributes
                # Reindex all residue and atom indices
                self.entities[0].reset_indices_attributes()
                self.entities[0].start_indices(dtype='residue', at=0)
                self.entities[0].start_indices(dtype='atom', at=0)
                for prior_idx, entity in enumerate(self.entities[1:]):
                    entity.reset_indices_attributes()
                    entity.start_indices(dtype='residue', at=self.entities[prior_idx].residue_indices[-1] + 1)
                    entity.start_indices(dtype='atom', at=self.entities[prior_idx].atom_indices[-1] + 1)
                # set the arrayed attributes for all PDB containers (chains, entities)
                self.update_attributes(_atoms=self._atoms, _residues=self._residues, _coords=self._coords)
                if rename_chains:  # set each successive Entity to have an incrementally higher chain id
                    available_chain_ids = self.return_chain_generator()
                    for idx, entity in enumerate(self.entities):
                        entity.chain_id = next(available_chain_ids)
                        self.log.debug('Entity %s new chain identifier %s' % (entity.name, entity.chain_id))
                # update chains after everything is set
                chains = []
                for entity in self.entities:
                    chains.extend(entity.chains)
                self.chains = chains
                self.chain_ids = [chain.name for chain in self.chains]
            else:  # create Entities from Chain.Residues
                self.create_entities(**kwargs)

        if pose_format:
            self.renumber_structure()

    # def format_header(self):
    #     return self.format_biomt() + self.format_seqres()

    def format_biomt(self, **kwargs) -> str:
        """Return the BIOMT record for the PDB if there was one parsed

        Keyword Args:
            **kwargs
        Returns:
            (str)
        """
        if self.biomt_header != '':
            return self.biomt_header
        elif self.biomt:
            return '%s\n' \
                % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}'.format(v_idx, m_idx, *vec)
                            for m_idx, matrix in enumerate(self.biomt, 1) for v_idx, vec in enumerate(matrix, 1))
        else:
            return ''

    def format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            **kwargs
        Returns:
            (str)
        """
        if self.reference_sequence:
            formated_reference_sequence = \
                {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in sequence)))
                 for chain, sequence in self.reference_sequence.items()}
            chain_lengths = {chain: len(sequence) for chain, sequence in self.reference_sequence.items()}
            return '%s\n' \
                % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
                            % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                            for chain, sequence in formated_reference_sequence.items()
                            for line_number in range(1, 1 + math.ceil(len(sequence)/seq_res_len)))
        else:
            return ''

    def parse_seqres(self, seqres_lines):
        """Convert SEQRES information to single amino acid dictionary format

        Args:
            seqres_lines (list): The list of lines containing SEQRES information
        Sets:
            self.reference_sequence
        """
        # SEQRES   1 A  182  THR THR ALA SER THR SER GLN VAL ARG GLN ASN TYR HIS
        # SEQRES   2 A  182  GLN ASP SER GLU ALA ALA ILE ASN ARG GLN ILE ASN LEU
        # SEQRES   3 A  182  GLU LEU TYR ALA SER TYR VAL TYR LEU SER MET SER TYR
        # SEQRES ...
        # SEQRES  16 C  201  SER TYR ILE ALA GLN GLU
        for line in seqres_lines:
            chain, length, *sequence = line.split()  # [0]
            if chain in self.reference_sequence:
                self.reference_sequence[chain].extend(sequence)
            else:
                self.reference_sequence[chain] = sequence

        for chain, sequence in self.reference_sequence.items():
            for idx, aa in enumerate(sequence):
                # try:
                if aa.title() in protein_letters_3to1_extended:
                    self.reference_sequence[chain][idx] = protein_letters_3to1_extended[aa.title()]
                # except KeyError:
                else:
                    if aa.title() == 'Mse':
                        self.reference_sequence[chain][idx] = 'M'
                    else:
                        self.reference_sequence[chain][idx] = '-'
            self.reference_sequence[chain] = ''.join(self.reference_sequence[chain])

    def reorder_chains(self, exclude_chains=None):
        """Renames chains using PDB.available_letter. Caution, doesn't update self.reference_sequence chain info
        """
        available_chain_ids = self.return_chain_generator()
        if exclude_chains:
            available_chains = sorted(set(available_chain_ids).difference(exclude_chains))
        else:
            available_chains = list(available_chain_ids)

        # Update chain_ids, then each chain
        self.chain_ids = available_chains[:self.number_of_chains]
        for chain, new_id in zip(self.chains, self.chain_ids):
            chain.chain_id = new_id

        self.get_chain_sequences()

    def renumber_residues_by_chain(self):
        for chain in self.chains:
            chain.renumber_residues()

    def create_chains(self, solve_discrepancy: bool = True):
        """For all the Residues in the PDB, create Chain objects which contain their member Residues

        Sets:
            self.chains
        """
        if solve_discrepancy:
            chain_idx = 0
            chain_residues = [[0]]  # self.residues[0].index]}  <- should always be zero
            for prior_idx, residue in enumerate(self.residues[1:]):  # start at the second index to avoid off by one
                if residue.number_pdb < self.residues[prior_idx].number_pdb \
                        or residue.chain != self.residues[prior_idx].chain:
                    # Decreased number should only happen with new chain therefore this SHOULD satisfy a malformed PDB
                    chain_idx += 1
                    chain_residues.append([prior_idx + 1])  # residue.index]
                    # chain_residues[chain_idx] = [residue]
                else:
                    chain_residues[chain_idx].append(prior_idx + 1)  # residue.index)
                    # chain_residues[chain_idx].append(residue)
            available_chain_ids = self.return_chain_generator()
            for chain_idx, residue_indices in enumerate(chain_residues):
                if chain_idx < len(self.chain_ids):  # Todo this logic is flawed when chains come in out of order
                    chain_id = self.chain_ids[chain_idx]
                    discard_chain = next(available_chain_ids)
                else:  # when there are more chains than supplied by file, chose the next available
                    chain_id = next(available_chain_ids)
                self.chains.append(Chain(name=chain_id, coords=self._coords, log=self._log, residues=self._residues,
                                         residue_indices=residue_indices))
                # self.chains[idx].set_atoms_attributes(chain=chain_id)
            self.chain_ids = [chain.name for chain in self.chains]
        else:
            for chain_id in self.chain_ids:
                self.chains.append(Chain(name=chain_id, coords=self._coords, log=self._log, residues=self._residues,
                                         residue_indices=[idx for idx, residue in enumerate(self.residues)
                                                          if residue.chain == chain_id]))
        self.get_chain_sequences()  # Todo maybe depreciate in favor of entities?

    def get_chains(self, names: Container = None) -> List:
        """Retrieve Chains in PDB. Returns all by default. If a list of names is provided, the selected Chains are
        returned"""
        if names and isinstance(names, Iterable):
            return [chain for chain in self.chains if chain.name in names]
        else:
            return self.chains

    def chain(self, chain_name: str) -> Chain:
        """Return the Chain object specified by the passed chain ID. If not found, return None
        Returns:
            (Chain)
        """
        for chain in self.chains:
            if chain.name == chain_name:
                return chain
        return

    def write(self, **kwargs) -> Optional[str]:
        """Write PDB Atoms to a file specified by out_path or with a passed file_handle

        Returns:
            The name of the written file if out_path is used
        """
        if not kwargs.get('header') and self.cryst_record:
            kwargs['header'] = self.cryst_record

        return super().write(**kwargs)

    def get_chain_sequences(self):
        self.atom_sequences = {chain.name: chain.sequence for chain in self.chains}
        # self.atom_sequences = {chain: self.chain(chain).get_structure_sequence() for chain in self.chain_ids}

    def orient(self, symmetry: str = None, log: os.PathLike = None):
        """Orient a symmetric PDB at the origin with its symmetry axis canonically set on axes defined by symmetry
        file. Automatically produces files in PDB numbering for proper orient execution

        Keyword Args:
            symmetry=None (str): What is the symmetry of the specified PDB?
            log=None (os.PathLike): If there is a log specific for orienting
        """
        # orient_oligomer.f program notes
        # C		Will not work in any of the infinite situations where a PDB file is f***ed up,
        # C		in ways such as but not limited to:
        # C     equivalent residues in different chains don't have the same numbering; different subunits
        # C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
        # C		of separate IDs; multiple conformations are written out for the same subunit structure
        # C		(as in an NMR ensemble), negative residue numbers, etc. etc.
        # must format the input.pdb in an acceptable manner
        try:
            subunit_number = valid_subunit_number[symmetry]
        except KeyError:
            raise ValueError('Symmetry %s is not a valid symmetry. Please try one of: %s' %
                             (symmetry, ', '.join(valid_symmetries)))
        if not log:
            log = self.log

        if self.filepath:
            pdb_file_name = os.path.basename(self.filepath)
        else:
            pdb_file_name = '%s.pdb' % self.name
        # Todo change output to logger with potential for file and stdout

        number_of_subunits = len(self.chain_ids)
        if number_of_subunits > 1:
            if number_of_subunits != subunit_number:
                if number_of_subunits in multicomponent_valid_subunit_number.get(symmetry):
                    multicomponent = True
                else:
                    raise ValueError('%s\n Oligomer could not be oriented: It has %d subunits while a multiple of %d '
                                     'are expected for %s symmetry\n\n'
                                     % (pdb_file_name, number_of_subunits, subunit_number, symmetry))
            else:
                multicomponent = False
        else:
            raise ValueError('%s: Cannot orient a Structure with only a single chain. No symmetry present!' % self.name)

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
            self.entities[0].write_oligomer(orient_input, pdb_number=True)
            # entity1_chains = self.entities[0].chains
            # entity1_chains[0].write(orient_input, pdb_number=True)
            # with open(orient_input, 'w') as f:
            #     for chain in entity1_chains[1:]:
            #         chain.write(file_handle=f, pdb_number=True)
        else:
            self.write(orient_input, pdb_number=True)
        # self.renumber_residues_by_chain()

        p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=orient_dir)
        in_symm_file = os.path.join(orient_dir, 'symm_files', symmetry)
        stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
        # stderr = stderr.decode()  # turn from bytes to string 'utf-8' implied
        # stdout = pdb_file_name + stdout.decode()[28:]
        log.info(pdb_file_name + stdout.decode()[28:])
        log.info(stderr.decode()) if stderr else None
        if not os.path.exists(orient_output) or os.stat(orient_output).st_size == 0:
            # orient_log = os.path.join(out_dir, orient_log_file)
            log_file = getattr(log.handlers[0], 'baseFilename', None)
            log_message = '. Check %s for more information' % log_file if log_file else ''
            error_string = 'orient_oligomer could not orient %s%s' % (pdb_file_name, log_message)
            raise RuntimeError(error_string)

        oriented_pdb = PDB.from_file(orient_output, name=self.name, pose_format=False, log=log)
        orient_fixed_struct = oriented_pdb.chains[0]
        if multicomponent:
            moving_struct = self.entities[0]
            # _, rot, tx, _ = superposition3d(oriented_pdb.chains[0].get_cb_coords(), self.entities[0].get_cb_coords())
        else:
            # orient_fixed_struct = oriented_pdb.chains[0]
            moving_struct = self.chains[0]
        try:
            _, rot, tx, _ = superposition3d(orient_fixed_struct.get_cb_coords(), moving_struct.get_cb_coords())
        except ValueError:  # we have the wrong lengths, lets subtract a certain amount by performing a seq alignment
            # rot, tx = None, None
            orient_fixed_seq = orient_fixed_struct.sequence
            moving_seq = moving_struct.sequence
            # while not rot:
            #     try:
            # moving coords are from the pre-orient structure where orient may have removed residues
            # lets try to remove those residues by doing an alignment
            align_orient_seq, align_moving_seq, *_ = generate_alignment(orient_fixed_seq, moving_seq, local=True)
            # align_seq_1.replace('-', '')
            # orient_idx1 = moving_seq.find(align_orient_seq.replace('-', '')[0])
            for orient_idx1, aa in enumerate(align_orient_seq):
                if aa != '-':  # we found the first aligned residue
                    break
            orient_idx2 = orient_idx1 + len(align_orient_seq.replace('-', ''))
            # starting_index_of_seq2 = moving_seq.find(align_moving_seq.replace('-', '')[0])
            # # get the first matching index of the moving_seq from the first aligned residue
            # ending_index_of_seq2 = starting_index_of_seq2 + align_moving_seq.rfind(moving_seq[-1])  # find last index of reference
            _, rot, tx, _ = superposition3d(orient_fixed_struct.get_cb_coords(),
                                            moving_struct.get_cb_coords()[orient_idx1:orient_idx2])
            # except ValueError:
            #     rot, tx = None, None
        self.transform(rotation=rot, translation=tx)
        clean_orient_input_output()

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
    #         atom.occupancy = 0
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
            (dict): {'assembly': {1: ['A', 'B'], ...}
                     'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
                     'entity': {1: ['A', 'B'], ...},
                     'res': resolution,
                     'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
                     }
        """
        if self.api_entry:
            return

        if self.name and len(self.name) == 4:
            # if not self.api_entry:
            self.api_entry = get_pdb_info_by_entry(self.name)
            if self.assembly:
                # self.api_entry.update(get_pdb_info_by_assembly(self.name))
                self.api_entry['assembly'] = get_pdb_info_by_assembly(self.name)
            if self.api_entry:
                return  # True
            self.log.debug('PDB code \'%s\' was not found with the PDB API.' % self.name)
        else:
            self.log.debug('PDB code \'%s\' is not of the required format and will not be found with the PDB API'
                           % self.name)
        return  # False
        # if not self.api_entry and self.name and len(self.name) == 4:
        #     self.api_entry = get_pdb_info_by_entry(self.name)
        # self.get_dbref_info_from_api()
        # self.get_entity_info_from_api()

    def entity(self, entity_id: str) -> Union[Entity, None]:
        for entity in self.entities:
            if entity_id == entity.name:
                return entity
        return

    def create_entities(self, entity_names: Sequence = None, query_by_sequence: bool = True, **kwargs):
        """Create all Entities in the PDB object searching for the required information if it was not found during
        file parsing. First, search the PDB API using an attached PDB entry_id, dependent on the presence of a
        biological assembly file and/or multimodel file. Finally, initialize them from the Residues in each Chain
        instance using a specified threshold of sequence homology

        Keyword Args:
            entity_names (Sequence): Names explicitly passed for the Entity instances. Length must equal number of entities.
                Names will take precedence over query_by_sequence if passed.
            query_by_sequence=True (bool): Whether the PDB API should be queried for an Entity name by matching sequence
        """
        if not self.entity_info:  # we didn't get the info from the file, so we have to try and piece together
            # the file is either from a program that has modified the original PDB file, was a model that hasn't been
            # formatted properly, or may be some sort of PDB assembly. If it is a PDB assembly, the file will have a
            # final numeric suffix after the .pdb extension. If not, it may be an assembly file from another source, in
            # which case we have to solve by atomic info. If we have to solve by atomic info, then the number of
            # chains in the structure and the number of Entity chains will not be equal after prior attempts
            self.retrieve_pdb_info_from_api()  # First try to set self.api_entry if possible. This is probably safe
            if self.api_entry:  # self.api_entry = {'entity': {1: ['A', 'B'], ...}, ...}
                if self.assembly:  # When PDB API is returning information on the asu and assembly is different
                    if self.multimodel:  # ensure the renaming of chains is handled correctly
                        for ent_idx, chains in self.api_entry.get('entity').items():
                            # chain_set = set(chains)
                            success = False
                            for cluster_idx, cluster_chains in self.api_entry.get('assembly').items():
                                # if set(cluster_chains) == chain_set:  # we found the right cluster
                                if not set(cluster_chains).difference(chains):  # we found the right cluster
                                    self.entity_info.append(
                                        {'chains': [new_chn for new_chn, old_chn in self.multimodel_chain_map.items()
                                                    if old_chn in chains], 'name': ent_idx})
                                    success = True
                                    break  # this should be fine since entities will cluster together, unless they don't
                            if not success:
                                self.log.error('Unable to find the chains corresponding from asu (%s) to assembly (%s)'
                                               % (self.api_entry.get('entity'), self.api_entry.get('assembly')))
                    else:  # chain names should be the same as the assembly API if the file is sourced from PDB
                        self.entity_info = [{'chains': chains, 'name': ent_idx}
                                            for ent_idx, chains in self.api_entry.get('assembly').items()]
                else:
                    self.entity_info = [{'chains': chains, 'name': ent_idx}
                                        for ent_idx, chains in self.api_entry.get('entity').items()]
                # check to see that the entity_info is in line with the number of chains already parsed
                found_entity_chains = [chain for info in self.entity_info for chain in info.get('chains', [])]
                if len(self.chain_ids) != len(found_entity_chains):
                    self.get_entity_info_from_atoms(**kwargs)  # tolerance=0.9
            else:  # Still nothing, then API didn't work for pdb_name. Solve by atom information
                self.get_entity_info_from_atoms(**kwargs)  # tolerance=0.9
                if query_by_sequence and not entity_names:
                    for data in self.entity_info:
                        pdb_api_name = retrieve_entity_id_by_sequence(data['sequence'])
                        if pdb_api_name:
                            pdb_api_name = pdb_api_name.lower()
                            self.log.info('Entity %d now named \'%s\', as found by PDB API sequence search'
                                          % (data['name'], pdb_api_name))
                            data['name'] = pdb_api_name
        if entity_names:
            for idx, data in enumerate(self.entity_info):
                try:
                    data['name'] = entity_names[idx]
                    self.log.debug('Entity %d now named \'%s\', as directed by supplied entity_names'
                                   % (idx + 1, entity_names[idx]))
                except IndexError:
                    raise IndexError('The number of indices in entity_names (%d) must equal the number of entities (%d)'
                                     % (len(entity_names), len(self.entity_info)))

        # For each Entity, get chains
        for data in self.entity_info:
            # v make Chain objects (if they are names)
            data['chains'] = [self.chain(chain) if isinstance(chain, str) else chain for chain in data.get('chains')]
            data['chains'] = [chain for chain in data['chains'] if chain]  # remove any missing chains
            # get uniprot ID if the file is from the PDB and has a DBREF remark
            accession = self.dbref.get(data['chains'][0].chain_id, None)
            data['uniprot_id'] = accession['accession'] if accession and accession['db'] == 'UNP' else accession
            #                                               generated from a PDB API sequence search v
            data['name'] = '%s_%d' % (self.name, data['name']) if isinstance(data['name'], int) else data['name']
            self.entities.append(Entity.from_chains(**data, log=self._log))

    def get_entity_info_from_atoms(self, tolerance=0.9, **kwargs):
        """Find all unique Entities in the input .pdb file. These are unique sequence objects

        Keyword Args:
            tolerance=0.1 (float): The acceptable difference between chains to consider them the same Entity.
                Tuning this parameter is necessary if you have chains which should be considered different entities,
                but are fairly similar. Alternatively, the use of a structural match could be used.
                For example, when each chain in an ASU is structurally deviating, but they all share the same sequence
        Sets:
            self.entity_info
        """
        assert tolerance <= 1, '%s tolerance cannot be greater than 1!' % self.get_entity_info_from_atoms.__name__
        entity_idx = 1
        self.entity_info.clear()  # get rid of any information already acquired
        self.entity_info.append({'chains': [self.chains[0]], 'sequence': self.chains[0].sequence, 'name': entity_idx})
        for chain in self.chains[1:]:
            self.log.debug('Searching for matching Entities for Chain %s' % chain.name)
            new_entity = True  # assume all chains are unique entities
            for data in self.entity_info:
                # Todo implement structure check
                #  rmsd_threshold = 1.  # threshold needs testing
                #  rmsd, rot, tx, _ = superposition3d()
                #  if rmsd < rmsd_threshold:
                #      data['chains'].append(chain)
                #      new_entity = False  # The entity is not unique, do not add
                #      break
                # check if the sequence associated with the atom chain is in the entity dictionary
                if chain.sequence == data['sequence']:
                    score = len(chain.sequence)
                else:
                    alignment = pairwise2.align.localxx(chain.sequence, data['sequence'])
                    score = alignment[0][2]  # first alignment from localxx, grab score value
                match_score = score / len(data['sequence'])  # could also use which ever sequence is greater
                length_proportion = abs(len(chain.sequence) - len(data['sequence'])) / len(data['sequence'])
                self.log.debug('Chain %s matches Entity %d with %0.2f identity and length difference of %0.2f'
                               % (chain.name, data['name'], match_score, length_proportion))
                if match_score >= tolerance and length_proportion <= 1 - tolerance:
                    # if number of sequence matches is > tolerance, and the length difference < tolerance
                    # the current chain is the same as the Entity, add to chains, and move on to the next chain
                    data['chains'].append(chain)
                    new_entity = False  # The entity is not unique, do not add
                    break

            if new_entity:  # no existing entity matches, add new entity
                entity_idx += 1
                self.entity_info.append({'chains': [chain], 'sequence': chain.sequence, 'name': entity_idx})
        self.log.debug('Entities were generated from ATOM records.')

    def entity_from_chain(self, chain_id: str) -> Union[Entity, None]:
        """Return the entity associated with a particular chain id

        Returns:
            (Union[Entity, None])
        """
        for entity in self.entities:
            if chain_id == entity.chain_id:
                return entity
        return

    def entity_from_residue(self, residue_number: int) -> Union[Entity, None]:
        """Return the entity associated with a particular Residue number

        Returns:
            (Union[Entity, None])
        """
        for entity in self.entities:
            if entity.get_residues(numbers=[residue_number]):
                return entity
        return

    def match_entity_by_struct(self, other_struct=None, entity=None, force_closest=False):
        """From another set of atoms, returns the first matching chain from the corresponding entity"""
        return  # TODO when entities are structure compatible

    def match_entity_by_seq(self, other_seq: str = None, force_closest: bool = True, threshold: float = 0.7) \
            -> Union[Entity, None]:
        """From another sequence, returns the first matching chain from the corresponding Entity

        Returns
            (Union[Entity, None])
        """
        for entity in self.entities:
            if other_seq == entity.sequence:
                return entity

        # we didn't find an ideal match
        if force_closest:
            alignment_score_d = {}
            for entity in self.entities:
                # TODO get a gap penalty and rework entire alignment function...
                alignment = pairwise2.align.localxx(other_seq, entity.sequence)
                max_align_score, max_alignment = 0, None
                for idx, align in enumerate(alignment):
                    if align.score > max_align_score:
                        max_align_score = align.score
                        max_alignment = idx
                alignment_score_d[entity] = alignment[max_alignment].score

            max_score, max_score_entity = 0, None
            for entity, score in alignment_score_d.items():
                normalized_score = score / len(entity.sequence)
                if normalized_score > max_score:
                    max_score = normalized_score  # alignment_score_d[entity]
                    max_score_entity = entity
            if max_score > threshold:
                return max_score_entity

        return

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
                # {Chain: int}
                partner_interface_d = {self.chain(chain_id): len(residues) for chain_id, residues in interface_d.items()
                                       if chain_id not in self_interface_d}
                self.log.debug('Partner interface Chains %s' % partner_interface_d.keys())
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
                            if partner_chain in p_entity.chains:  # this wouldn't be true if the chains were symmetrized
                                self.log.debug('Partner Chain: %s' % partner_chain.name)
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

        if not chain:
            chain = self.chain_ids[0]
        chain_of_interest = self.chain(chain)
        if not chain_of_interest:
            raise ValueError('The Chain %s is not found in the Structure' % chain)
        partner_chains = get_unique_contacts(chain_of_interest, entity=self.entity_from_chain(chain), extra=extra)
        # partner_chains = get_unique_contacts(chain, entity=self.entity_from_chain(chain), extra=extra)
        # partner_chains = get_unique_contacts(chain, entity=self.entity_from_chain(chain).name, extra=extra)

        asu_atoms = chain_of_interest.atoms
        for atoms in [chain.atoms for chain in partner_chains]:
            asu_atoms.extend(atoms)
        asu_coords = np.concatenate([chain_of_interest.coords] + [chain.coords for chain in partner_chains])

        return asu_atoms, asu_coords

    def return_asu(self, chain='A'):  # Todo Depreciate in favor of Pose.get_contacting_asu()
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
    if len(interface_pdb.chain_ids) == 2:
        for temp_name in temp_chain_d:
            interface_pdb.chain(temp_name).set_atoms_attributes(chain=temp_chain_d[temp_name])
            # Todo edit this mechanism! ^

    return interface_pdb


def fetch_pdb(pdb_codes: Union[str, list], assembly: int = 1, asu: bool = False,
              out_dir: Union[str, bytes] = os.getcwd(), **kwargs) -> List[Union[str, bytes]]:  # Todo mmcif
    """Download PDB files from pdb_codes provided in a file, a supplied list, or a single entry
    Can download a specific biological assembly if asu=False.
    Ex: fetch_pdb('1bkh', assembly=2) fetches 1bkh biological assembly 2 "1bkh.pdb2"

    Args:
        pdb_codes: PDB IDs of interest.
        assembly: The integer of the assembly to fetch
        asu: Whether to download the asymmetric unit file
        out_dir: The location to save downloaded files to
    Returns:
        Filenames of the retrieved files
    """
    file_names = []
    for pdb_code in to_iterable(pdb_codes):
        clean_pdb = pdb_code[:4].lower()
        if asu:
            clean_pdb = '%s.pdb' % clean_pdb
        else:
            # assembly = pdb[-3:]
            # try:
            #     assembly = assembly.split('_')[1]
            # except IndexError:
            #     assembly = '1'
            clean_pdb = '%s.pdb%d' % (clean_pdb, assembly)

        # clean_pdb = '%s.pdb%d' % (clean_pdb, assembly)
        file_name = os.path.join(out_dir, clean_pdb)
        current_file = sorted(glob(file_name))
        # print('Found the files %s' % current_file)
        # current_files = os.listdir(location)
        # if clean_pdb not in current_files:
        if not current_file:  # glob will return an empty list if the file is missing and therefore should be downloaded
            # Always returns files in lowercase
            status = os.system('wget -q -O %s https://files.rcsb.org/download/%s' % (file_name, clean_pdb))
            # TODO subprocess.POPEN()
            if status != 0:
                logger.error('PDB download failed for: %s. If you believe this PDB ID is correct, there may only be a '
                             '.cif file available for this entry. ' % clean_pdb)
                # todo parse .cif file.
                #  Super easy as the names of the columns are given in a loop and the ATOM records still start with ATOM
                #  The additional benefits is that the records contain entity IDS as well as the residue index and the
                #  author residue number. I think I will prefer this format from now on once parsing is possible.

            # file_request = requests.get('https://files.rcsb.org/download/%s' % clean_pdb)
            # if file_request.status_code == 200:
            #     with open(file_name, 'wb') as f:
            #         f.write(file_request.content)
            # else:
            #     logger.error('PDB download failed for: %s' % pdb)
        file_names.append(file_name)

    return file_names


def fetch_pdb_file(pdb_code: str, asu: bool = True, location: Union[str, bytes] = pdb_db, **kwargs) -> \
        Optional[Union[str, bytes]]:  # assembly=None, out_dir=os.getcwd(),
    """Fetch PDB object from PDBdb or download from PDB server

    Args:
        pdb_code: The PDB ID/code. If the biological assembly is desired, supply 1ABC_1 where '_1' is assembly ID
        asu: Whether to fetch the ASU
        location: Location of a local PDB mirror if one is linked on disk
        assembly=None (Optional[int]): Location of a local PDB mirror if one is linked on disk
        out_dir=os.getcwd() (Union[str, bytes]): The location to save retrieved files if fetched from PDB
    Returns:
        The path to the file if located successfully
    """
    # if location == pdb_db and asu:
    if os.path.exists(location) and asu:
        get_pdb = (lambda pdb_code, location=None, **kwargs:  # asu=None, assembly=None, out_dir=None
                   sorted(glob(os.path.join(location, 'pdb%s.ent' % pdb_code.lower()))))
        logger.debug('Searching for PDB file at \'%s\'' % os.path.join(location, 'pdb%s.ent' % pdb_code.lower()))
        # Cassini format is above, KM local pdb and the escher PDB mirror is below
        # get_pdb = (lambda pdb_code, asu=None, assembly=None, out_dir=None:
        #            glob(os.path.join(pdb_db, subdirectory(pdb_code), '%s.pdb' % pdb_code)))
        # print(os.path.join(pdb_db, subdirectory(pdb_code), '%s.pdb' % pdb_code))
    else:
        get_pdb = fetch_pdb

    # return a list where the matching file is the first (should only be one anyway)
    pdb_file = get_pdb(pdb_code, asu=asu, location=location, **kwargs)
    if not pdb_file:
        logger.warning('No matching file found for PDB: %s' % pdb_code)
    else:  # we should only find one file, therefore, return the first
        return pdb_file[0]


def orient_pdb_file(file: os.PathLike, log: Logger = logger, symmetry: str = None,
                    out_dir: os.PathLike = None) -> Optional[str]:
    """For a specified pdb filename and output directory, orient the PDB according to the provided symmetry where the
        resulting .pdb file will have the chains symmetrized and oriented in the coordinate frame as to have the major axis
        of symmetry along z, and additional axis along canonically defined vectors. If the symmetry is C1, then the monomer
        will be transformed so the center of mass resides at the origin

        Args:
            file: The location of the .pdb file to be oriented
            log: A log to report on operation success
            symmetry: The symmetry type to be oriented. Possible types in SymmetryUtils.valid_subunit_number
            out_dir: The directory that should be used to output files
        Returns:
            Filepath of oriented PDB
        """
    pdb_filename = os.path.basename(file)
    oriented_file_path = os.path.join(out_dir, pdb_filename)
    if os.path.exists(oriented_file_path):
        return oriented_file_path
    # elif sym in valid_subunit_number:
    else:
        pdb = PDB.from_file(file, log=log, pose_format=False)  #, entities=False)
        try:
            pdb.orient(symmetry=symmetry)
            pdb.write(out_path=oriented_file_path)
            log.info('Oriented: %s' % pdb_filename)
            return oriented_file_path
        except (ValueError, RuntimeError) as error:
            log.error(str(error))


def query_qs_bio(pdb_entry_id: str) -> int:
    qsbio_confirmed = unpickle(qs_bio)
    biological_assemblies = qsbio_confirmed.get(pdb_entry_id)
    if biological_assemblies:  # first   v   assembly in matching oligomers
        assembly = biological_assemblies[0]
    else:
        assembly = 1
        logger.warning('No confirmed biological assembly for entry %s'
                       ' using PDB default assembly %d' % (pdb_entry_id, assembly))

    return assembly
# ref_aa = PDB.from_file(reference_aa_file, log=None, pose_format=False, entities=False)
# from shutil import move
# move(reference_residues_pkl, '%s.bak' % reference_residues_pkl)
# pickle_object(ref_aa.residues, name=reference_residues_pkl, out_path='')
