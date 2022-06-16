from __future__ import annotations

import os
import subprocess
from collections import UserList, defaultdict
from collections.abc import Iterable, Generator
from copy import copy  # , deepcopy
from itertools import repeat
from logging import Logger
from math import ceil
from random import random  # , randint
from typing import IO, Sequence, Container, Literal, get_args

import numpy as np
from Bio.Data.IUPACData import protein_letters, protein_letters_1to3, protein_letters_3to1_extended, \
    protein_letters_1to3_extended
from numpy.linalg import eigh, LinAlgError
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree  # , KDTree, NearestNeighbors

from JobResources import parameterize_frag_length
from PathUtils import free_sasa_exe_path, stride_exe_path, errat_exe_path, make_symmdef, scout_symmdef, \
    reference_residues_pkl, free_sasa_configuration_path
from Query.PDB import get_entity_reference_sequence, get_pdb_info_by_entity, retrieve_entity_id_by_sequence
from SequenceProfile import SequenceProfile, generate_mutations
# from ProteinExpression import find_expression_tags, remove_expression_tags
from SymDesignUtils import start_log, null_log, DesignError, unpickle
from classes.SymEntry import get_rot_matrices, make_rotations_degenerate
from utils.GeneralUtils import transform_coordinate_sets
from utils.SymmetryUtils import valid_subunit_number, cubic_point_groups, point_group_symmetry_operators, \
    rotation_range, identity_matrix, origin, flip_x_matrix

# globals
logger = start_log(name=__name__)
seq_res_len = 52
directives = Literal['special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding',
                     'branched']
mutation_directives: tuple[directives, ...] = get_args(directives)
# mutation_directives = \
#     ['special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding', 'branched']
residue_properties = {'ALA': {'hydrophobic', 'apolar'},
                      'CYS': {'special', 'hydrophobic', 'apolar', 'polar', 'hbonding'},
                      'ASP': {'charged', 'polar', 'hbonding'},
                      'GLU': {'charged', 'polar', 'hbonding'},
                      'PHE': {'hydrophobic', 'apolar', 'aromatic'},
                      'GLY': {'special'},
                      'HIS': {'charged', 'polar', 'aromatic', 'hbonding'},
                      'ILE': {'hydrophobic', 'apolar', 'branched'},
                      'LYS': {'charged', 'polar', 'hbonding'},
                      'LEU': {'hydrophobic', 'apolar', 'branched'},
                      'MET': {'hydrophobic', 'apolar'},
                      'ASN': {'polar', 'hbonding'},
                      'PRO': {'special', 'hydrophobic', 'apolar'},
                      'GLN': {'polar', 'hbonding'},
                      'ARG': {'charged', 'polar', 'hbonding'},
                      'SER': {'polar', 'hbonding'},
                      'THR': {'polar', 'hbonding', 'branched'},
                      'VAL': {'hydrophobic', 'apolar', 'branched'},
                      'TRP': {'hydrophobic', 'apolar', 'aromatic', 'hbonding'},
                      'TYR': {'hydrophobic', 'apolar', 'aromatic', 'hbonding'}}
# useful in generating aa_by_property from mutation_directives and residue_properties
# aa_by_property = {}
# for type_ in mutation_directives:
#     aa_by_property[type_] = set()
#     for res in residue_properties:
#         if type_ in residue_properties[res]:
#             aa_by_property[type_].append(res)
#     aa_by_property[type_] = list(aa_by_property[type_])
aa_by_property = \
    {'special': {'CYS', 'GLY', 'PRO'},
     'charged': {'ARG', 'GLU', 'ASP', 'HIS', 'LYS'},
     'polar': {'CYS', 'ASP', 'GLU', 'HIS', 'LYS', 'ASN', 'GLN', 'ARG', 'SER', 'THR'},
     'apolar': {'ALA', 'CYS', 'PHE', 'ILE', 'LEU', 'MET', 'PRO', 'VAL', 'TRP', 'TYR'},
     'hydrophobic': {'ALA', 'CYS', 'PHE', 'ILE', 'LEU', 'MET', 'PRO', 'VAL', 'TRP', 'TYR'},  # same as apolar
     'aromatic': {'PHE', 'HIS', 'TRP', 'TYR'},
     'hbonding': {'CYS', 'ASP', 'GLU', 'HIS', 'LYS', 'ASN', 'GLN', 'ARG', 'SER', 'THR', 'TRP', 'TYR'},
     'branched': {'ILE', 'LEU', 'THR', 'VAL'}}
gxg_sasa = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
            'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
            'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167, 'GLU': 223, 'GLN': 225, 'GLY': 104, 'HIS': 224,
            'ILE': 197, 'LEU': 201, 'LYS': 236, 'MET': 224, 'PHE': 240, 'PRO': 159, 'SER': 155, 'THR': 172, 'TRP': 285,
            'TYR': 263, 'VAL': 174}  # from table 1, theoretical values of Tien et al. 2013


def unknown_index():
    return -1

atomic_polarity_table = {  # apolar = 0, polar = 1
    'ALA': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0}),
    'ARG': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'NE': 1, 'CZ': 0, 'NH1': 1, 'NH2': 1}),
    'ASN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'ND2': 1}),
    'ASP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'OD2': 1}),
    'CYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'SG': 1}),
    'GLN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'NE2': 1}),
    'GLU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'OE2': 1}),
    'GLY': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1}),
    'HIS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'ND1': 1, 'CD2': 0, 'CE1': 0, 'NE2': 1}),
    'ILE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0, 'CD1': 0}),
    'LEU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0}),
    'LYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'CE': 0, 'NZ': 1}),
    'MET': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'SD': 1, 'CE': 0}),
    'PHE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0, 'CE2': 0, 'CZ': 0,}),
    'PRO': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0}),
    'SER': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG': 1}),
    'THR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG1': 1, 'CG2': 0}),
    'TRP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'NE1': 1, 'CE2': 0, 'CE3': 0, 'CZ2': 0, 'CZ3': 0, 'CH2': 0}),
    'TYR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0, 'CE2': 0, 'CZ': 0, 'OH': 1}),
    'VAL': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0})}
hydrogens = {
    'ALA': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '3HB': 0},
    'ARG': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, 'HE': 1, '1HH1': 1, '2HH1': 1, '1HH2': 1, '2HH2': 1},
    'ASN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HD2': 1, '2HD2': 1,
            '1HD1': 1, '2HD1': 1},  # these are the alternative specification
    'ASP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0},
    'CYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 1},
    'GLN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE2': 1, '2HE2': 1,
            '1HE1': 1, '2HE1': 1},  # these are the alternative specification
    'GLU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0},
    'GLY': {'2HA': 0, 'H': 1, '1HA': 0, 'HA3': 0},  # last entry is from PDB version
    'HIS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 1, 'HD2': 0, 'HE1': 0, 'HE2': 1},  # this assumes HD1 is on ND1, HE2 is on NE2
    'ILE': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0, '1HD1': 0, '2HD1': 0, '3HD1': 0,
            '3HG1': 0},  # this is the alternative specification
    'LEU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 0, '1HD1': 0, '2HD1': 0, '3HD1': 0, '1HD2': 0, '2HD2': 0, '3HD2': 0},
    'LYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, '1HE': 0, '2HE': 0, '1HZ': 1, '2HZ': 1, '3HZ': 1},
    'MET': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE': 0, '2HE': 0, '3HE': 0},
    'PHE': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HZ': 0},
    'PRO': {'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 1},
    'SER': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 1},
    'THR': {'HA': 0, 'HB': 0, 'H': 1, 'HG1': 1, '1HG2': 0, '2HG2': 0, '3HG2': 0,
            'HG2': 1, '1HG1': 0, '2HG1': 0, '3HG1': 0},  # these are the alternative specification
    'TRP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HE1': 1, 'HE3': 0, 'HZ2': 0, 'HZ3': 0, 'HH2': 0,  # assumes HE1 is on NE1
            'HE2': 0, 'HZ1': 0, 'HH1': 0, 'HH3': 0},  # none of these should be possible given standard nomenclature, but including incase
    'TYR': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HH': 1},
    'VAL': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '3HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0}}
termini = {'1H': 1, '2H': 1, '3H': 1, 'OXT': 1}
for residue_type, residue_atoms in atomic_polarity_table.items():
    residue_atoms.update(termini)
    residue_atoms.update(hydrogens[residue_type])


class Log:
    def __init__(self, log: Logger = None):
        self.log = log


class Coords:
    coords: np.ndarray

    def __init__(self, coords: list | np.ndarray = None):
        if coords is None:
            self.coords = np.array([])
        else:
            self.coords = np.array(coords)

    def delete(self, indices: Sequence[int]):
        self.coords = np.delete(self.coords, indices, axis=0)

    def insert(self, new_coords: list | np.ndarray, at: int = None):
        """Insert Atom objects into the Atoms container

        Args:
            new_coords: The coords to include into Coords
            at: The index to perform the insert at
        """
        self.coords = \
            np.concatenate((self.coords[:at] if 0 <= at <= len(self.coords) else self.coords, new_coords,
                            self.coords[at:])
                           if at else (self.coords[:at] if 0 <= at <= len(self.coords) else self.coords, new_coords))

    def __len__(self) -> int:
        return self.coords.shape[0]


class Atom:
    """An Atom container with the full Structure coordinates and the Atom unique data. Pass a reference to the full
    Structure coordinates for Keyword Arg coords=self.coords"""
    index: int | None
    number: int | None
    type: str | None
    alt_location: str | None
    residue_type: str | None
    chain: str | None
    pdb_residue_number: int | None
    residue_number: int | None
    code_for_insertion: str | None
    occupancy: float | None
    b_factor: float | None
    element_symbol: str | None
    atom_charge: str | None

    def __init__(self, index: int = None, number: int = None, atom_type: str = None, alt_location: str = None,
                 residue_type: str = None, chain: str = None, residue_number: int = None,
                 code_for_insertion: str = None, occupancy: float = None, b_factor: float = None,
                 element_symbol: str = None, atom_charge: str = None):  # , sasa=None, sasa=0., coords=None
        self.index = index
        self.number = number
        self.type = atom_type
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain = chain
        self.pdb_residue_number = residue_number
        self.residue_number = residue_number  # originally set the same as parsing
        self.code_for_insertion = code_for_insertion
        self.occupancy = occupancy
        self.b_factor = b_factor
        self.element_symbol = element_symbol
        self.atom_charge = atom_charge
        # self.sasa = sasa
        # if coords:
        #     self.coords = coords

    # @classmethod
    # def from_info(cls, *args):
    #     # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occupancy, b_factor,
    #     # element_symbol, atom_charge
    #     """Initialize without coordinates"""
    #     return cls(*args)

    @property
    def sasa(self) -> float:
        """The Solvent accessible surface area for the Atom. Raises AttributeError if .sasa isn't set"""
        # try:  # let the Residue owner handle this error
        return self._sasa
        # except AttributeError:
        #     raise AttributeError

    @sasa.setter
    def sasa(self, sasa: float):
        self._sasa = sasa

    # @property
    # def coords(self):
    #     """This holds the atomic Coords which is a view from the Structure that created them"""
    #     # print(self._coords, len(self._coords.coords), self.index)
    #     # returns self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
    #     return self._coords.coords[self.index]  # [self.x, self.y, self.z]
    #
    # @coords.setter
    # def coords(self, coords):
    #     if isinstance(coords, Coords):
    #         self._coords = coords
    #     else:
    #         raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    #                              'view. To pass the Coords object for a Structure, use the private attribute _coords')

    def is_backbone(self) -> bool:
        """Is the Atom is a backbone Atom? These include N, CA, C, and O"""
        # backbone_specific_atom_type = ['N', 'CA', 'C', 'O']  # Todo make a class attribute
        return self.type in ['N', 'CA', 'C', 'O']

    def is_cb(self, gly_ca: bool = True) -> bool:
        """Is the Atom a CB atom? Default returns True if Glycine and Atom is CA

        Args:
            gly_ca: Whether to include Glycine CA in the boolean evaluation
        """
        if gly_ca:
            return self.type == 'CB' or (self.residue_type == 'GLY' and self.type == 'CA')
        else:
            #                                    When Rosetta assigns, it is this  v  but PDB assigns as this  v
            return self.type == 'CB' or (self.residue_type == 'GLY' and (self.type == '2HA' or self.type == 'HA3'))

    def is_ca(self) -> bool:
        """Is the Atom a CA atom?"""
        return self.type == 'CA'

    # def distance(self, atom, intra=False):
    #     """returns distance (type float) between current instance of Atom and another instance of Atom"""
    #     if self.chain == atom.chain and not intra:
    #         # self.log.error('Atoms Are In The Same Chain')
    #         return None
    #     else:
    #         distance = sqrt((self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2)
    #         return distance

    # def distance_squared(self, atom, intra=False):
    #     """returns squared distance (type float) between current instance of Atom and another instance of Atom"""
    #     if self.chain == atom.chain and not intra:
    #         # self.log.error('Atoms Are In The Same Chain')
    #         return None
    #     else:
    #         distance = (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2
    #         return distance

    def __key(self) -> tuple[float, str]:
        return self.b_factor, self.type  # self.index, self.type

    def __str__(self, **kwargs) -> str:  # type=None, number=None, pdb=False, chain=None,
        """Represent Atom in PDB format"""
        # this annoyingly doesn't comply with the PDB format specifications because of the atom type field
        # ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- PDB format
        # ATOM     32 CG2  VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- fstring print
        # return'{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'
        return '{:6s}%s {:^4s}{:1s}%s %s%s{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'\
            .format('ATOM', self.type, self.alt_location, self.code_for_insertion, self.occupancy, self.b_factor,
                    self.element_symbol, self.atom_charge)
        # ^ For future implement in residue writes
        # v old atom writes
        # return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'\
        #        .format('ATOM', self.number, self.type, self.alt_location, self.residue_type, (chain or self.chain),
        #                getattr(self, '%sresidue_number' % ('pdb_' if pdb else '')), self.code_for_insertion,
        #                self.occupancy, self.b_factor, self.element_symbol, self.atom_charge)

    def __eq__(self, other: Atom) -> bool:
        return (self.number == other.number and self.chain == other.chain and self.type == other.type and
                self.residue_type == other.residue_type)

    def __hash__(self) -> int:
        return hash(self.__key())


class Atoms:
    atoms: np.ndarray

    def __init__(self, atoms: list | np.ndarray = None):
        if atoms is None:
            self.atoms = np.array([])
        else:
            self.atoms = np.array(atoms)

    def delete(self, indices: Sequence[int]):
        self.atoms = np.delete(self.atoms, indices)

    def insert(self, new_atoms: list | np.ndarray, at: int = None):
        """Insert Atom objects into the Atoms container

        Args:
            new_atoms: The residues to include into Residues
            at: The index to perform the insert at
        """
        self.atoms = np.concatenate((self.atoms[:at] if 0 <= at <= len(self.atoms) else self.atoms,
                                     new_atoms if isinstance(new_atoms, Iterable) else [new_atoms],
                                     self.atoms[at:] if at is not None else []))

    def __copy__(self) -> Atoms:
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.atoms = self.atoms.copy()
        # copy all Atom
        for idx, atom in enumerate(other.atoms):
            other.atoms[idx] = copy(atom)
        # copy all attributes. No! most are unchanged...
        # # must copy any residue specific attributes
        # for attr in Atoms.residue_specific_attributes:
        #     try:
        #         for idx, atom in enumerate(other):
        #             setattr(other.atoms[idx], attr, copy(getattr(other.atoms[idx], attr)))
        #     except AttributeError:  # the attribute may not be set yet, so we should ignore all and move on
        #         continue

        return other

    def __len__(self) -> int:
        return self.atoms.shape[0]

    def __iter__(self) -> Atom:
        yield from self.atoms.tolist()


class Residue:
    # atom_indices: list[int]
    atoms: Atoms
    # backbone_indices: list[int]
    # backbone_and_cb_indices: list[int]
    chain: str
    coords: Coords
    # sidechain_indices: list[int]
    # heavy_atom_indices: list[int]
    local_density: float
    number: int
    number_pdb: int
    type: str

    def __init__(self, atom_indices: list[int] = None, atoms: Atoms = None, coords: Coords = None, log: Log = None):
        #        index=None
        # self.index = index
        self.log = log
        self.atom_indices = atom_indices
        self.atoms = atoms
        if coords:
            self.coords = coords
        # self.secondary_structure = None
        self.local_density = 0.
        self._contact_order = 0

    @property
    def log(self) -> Logger:
        """Access to the Structure logger"""
        return self._log.log

    @log.setter
    def log(self, log_object: Log):
        self._log = log_object

    @property
    def start_index(self) -> int:
        """The first atomic index of the Residue"""
        return self._start_index

    @start_index.setter
    def start_index(self, index: int):
        self._start_index = index
        self._atom_indices = list(range(index, index + self.number_of_atoms))

    @property
    def range(self) -> list[int]:
        """The range of indices corresponding to the Residue atoms"""
        return list(range(self.number_of_atoms))

    @property
    def atom_indices(self) -> list[int]:
        """The particular indices in which the Residue Atoms are located in the shared Atoms container"""
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices: list[int]):
        """Set the Structure atom indices to a list of integers"""
        self._atom_indices = indices
        try:
            self._start_index = indices[0]
        except (TypeError, IndexError):
            raise IndexError('The Residue wasn\'t passed any atom_indices which are required for initialization')

    @property
    def atoms(self) -> list[Atom]:
        """The particular Atom objects that the Residue owns"""
        return self._atoms.atoms[self._atom_indices].tolist()

    @atoms.setter
    def atoms(self, atoms: Atoms):
        if isinstance(atoms, Atoms):
            self._atoms = atoms
        else:
            raise AttributeError('The supplied atoms are not of the class Atoms! Pass an Atoms object not a Atoms view.'
                                 ' To pass the Atoms object for a Structure, use the private attribute ._atoms')

        side_chain, heavy_atoms = [], []
        for idx, atom in enumerate(self.atoms):
            if atom.type == 'N':
                self.n_index = idx
                # self.n = atom.index
            elif atom.type == 'CA':
                self.ca_index = idx
                # self.ca = atom.index
                if atom.residue_type == 'GLY':
                    self.cb_index = idx
                    # self.cb = atom.index
            elif atom.type == 'CB':  # atom.is_cb(InclGlyCA=True):
                self.cb_index = idx
                # self.cb = atom.index
            elif atom.type == 'C':
                self.c_index = idx
                # self.c = atom.index
            elif atom.type == 'O':
                self.o_index = idx
                # self.o = atom.index
            elif atom.type == 'H':
                self.h_index = idx
                # self.h = atom.index
            else:
                side_chain.append(idx)
                if 'H' not in atom.type:
                    heavy_atoms.append(idx)
        self.backbone_indices = \
            [getattr(self, index, None) for index in ['_n_index', '_ca_index', '_c_index', '_o_index']]
        self.backbone_and_cb_indices = getattr(self, '_cb_index', getattr(self, '_ca_index', None))
        self.sidechain_indices = side_chain
        self.heavy_atom_indices = self._bb_cb_indices + heavy_atoms
        self.chain = atom.chain
        self.number = atom.residue_number
        self.number_pdb = atom.pdb_residue_number
        self.type = atom.residue_type
        # if not self.ca_index:  # this is likely a NH or a C=O so we don't have a full residue
        #     self.log.error('Residue %d has no CA atom!' % self.number)
        #     # Todo this residue should be built, but as of 4/28/22 it only could be deleted
        #     self.ca_index = idx  # use the last found index as a rough guess
        #     self.secondary_structure = 'C'  # just a placeholder since stride shouldn't work
        #     # raise DesignError('Residue %d has no CA atom!' % self.number)

    # # This is the setter for all atom properties available above
    # def set_atoms_attributes(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all atoms in the Residue"""
    #     for kwarg, value in kwargs.items():
    #         for atom in self.atoms:
    #             setattr(atom, kwarg, value)

    @property
    def backbone_indices(self) -> list[int]:
        """The atom indices that belong to the backbone atoms"""
        return [self._atom_indices[idx] for idx in self._bb_indices]

    @backbone_indices.setter
    def backbone_indices(self, indices: Iterable[int]):
        self._bb_indices = [idx for idx in indices if idx]

    @property
    def backbone_and_cb_indices(self) -> list[int]:
        """The atom indices that belong to the backbone and CB atoms"""
        return [self._atom_indices[idx] for idx in self._bb_cb_indices]

    @backbone_and_cb_indices.setter
    def backbone_and_cb_indices(self, cb_index: int):
        self._bb_cb_indices = self._bb_indices + ([cb_index] if cb_index else [])

    @property
    def sidechain_indices(self) -> list[int]:
        """The atom indices that belong to the side chain atoms"""
        return [self._atom_indices[idx] for idx in self._sc_indices]

    @sidechain_indices.setter
    def sidechain_indices(self, indices: Sequence[int]):
        self._sc_indices = indices

    @property
    def heavy_atom_indices(self) -> list[int]:
        """The atom indices that belong to the heavy atoms (non-hydrogen)"""
        return [self._atom_indices[idx] for idx in self._heavy_atom_indices]

    @heavy_atom_indices.setter
    def heavy_atom_indices(self, indices: Sequence[int]):
        """Returns: (list[int])"""
        self._heavy_atom_indices = indices

    @property
    def contains_hydrogen(self) -> bool:
        """Returns whether the Residue contains hydrogen atoms"""
        return self._heavy_atom_indices != self._atom_indices

    @property
    def coords(self) -> np.ndarray:
        """The Residue atomic coords. Provides a view from the Structure that the Residue belongs too"""
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
        return self._coords.coords[self._atom_indices]

    @property
    def heavy_coords(self) -> np.ndarray:
        """The Residue atomic coords. Provides a view from the Structure that the Residue belongs too"""
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
        return self._coords.coords[[self._atom_indices[idx] for idx in self._heavy_atom_indices]]

    @property
    def backbone_coords(self) -> np.ndarray:
        """The backbone atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[idx] for idx in self._bb_indices]]

    @property
    def backbone_and_cb_coords(self) -> np.ndarray:
        """The backbone and CB atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[idx] for idx in self._bb_cb_indices]]

    @property
    def sidechain_coords(self) -> np.ndarray:
        """The backbone and CB atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[index] for index in self._sc_indices]]

    @coords.setter
    def coords(self, coords: Coords):
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords! Pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute ._coords')

    @property
    def backbone_atoms(self) -> list[Atom]:
        """Return the Residue backbone Atom objects"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._bb_indices]]

    @property
    def backbone_and_cb_atoms(self) -> list[Atom]:
        """Return the Residue backbone and CB Atom objects"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._bb_cb_indices]]

    @property
    def sidechain_atoms(self) -> list[Atom]:
        """Return the Residue side chain Atom objects"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._sc_indices]]

    @property
    def n(self) -> Atom | None:
        """Return the amide N Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._n_index]]
        except AttributeError:
            return

    @property
    def n_coords(self) -> np.ndarry | None:
        """Return the amide N Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._n_index]]
        except AttributeError:
            return

    @property
    def n_atom_index(self) -> int | None:
        """Return the index of the amide N Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._n_index]
        except AttributeError:
            return

    @property
    def n_index(self) -> int | None:
        """Return the index of the amide N Atom in the Residue Atoms"""
        try:
            return self._n_index
        except AttributeError:
            return

    @n_index.setter
    def n_index(self, index: int):
        self._n_index = index

    @property
    def h(self) -> Atom | None:
        """Return the amide H Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._h_index]]
        except AttributeError:
            return

    @property
    def h_coords(self) -> np.ndarry | None:
        """Return the amide H Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._h_index]]
        except AttributeError:
            return

    @property
    def h_atom_index(self) -> int | None:
        """Return the index of the amide H Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._h_index]
        except AttributeError:
            return

    @property
    def h_index(self) -> int | None:
        """Return the index of the amide H Atom in the Residue Atoms"""
        try:
            return self._h_index
        except AttributeError:
            return

    @h_index.setter
    def h_index(self, index):
        self._h_index = index

    @property
    def ca(self) -> Atom | None:
        """Return the CA Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._ca_index]]
        except AttributeError:
            return

    @property
    def ca_coords(self) -> np.ndarry | None:
        """Return the CA Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._ca_index]]
        except AttributeError:
            return

    @property
    def ca_atom_index(self) -> int | None:
        """Return the index of the CA Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._ca_index]
        except AttributeError:
            return

    @property
    def ca_index(self) -> int | None:
        """Return the index of the CA Atom in the Residue Atoms"""
        try:
            return self._ca_index
        except AttributeError:
            return

    @ca_index.setter
    def ca_index(self, index):
        self._ca_index = index

    @property
    def cb(self) -> Atom | None:
        """Return the CB Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._cb_index]]
        except AttributeError:
            return

    @property
    def cb_coords(self) -> np.ndarry | None:
        """Return the CB Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._cb_index]]
        except AttributeError:
            return

    @property
    def cb_atom_index(self) -> int | None:
        """Return the index of the CB Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._cb_index]
        except AttributeError:
            return

    @property
    def cb_index(self) -> int | None:
        """Return the index of the CB Atom in the Residue Atoms"""
        try:
            return self._cb_index
        except AttributeError:
            return

    @cb_index.setter
    def cb_index(self, index):
        self._cb_index = index

    @property
    def c(self) -> Atom | None:
        """Return the carbonyl C Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._c_index]]
        except AttributeError:
            return

    @property
    def c_coords(self) -> np.ndarry | None:
        """Return the carbonyl C Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._c_index]]
        except AttributeError:
            return

    @property
    def c_atom_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._c_index]
        except AttributeError:
            return

    @property
    def c_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Residue Atoms"""
        try:
            return self._c_index
        except AttributeError:
            return

    @c_index.setter
    def c_index(self, index):
        self._c_index = index

    @property
    def o(self) -> Atom | None:
        """Return the carbonyl O Atom object"""
        try:
            return self._atoms.atoms[self._atom_indices[self._o_index]]
        except AttributeError:
            return

    @property
    def o_coords(self) -> np.ndarry | None:
        """Return the carbonyl O Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._o_index]]
        except AttributeError:
            return

    @property
    def o_atom_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Structure Atoms"""
        try:
            return self._atom_indices[self._o_index]
        except AttributeError:
            return

    @property
    def o_index(self) -> int | None:
        """Return the index of the carbonyl O Atom in the Residue Atoms"""
        try:
            return self._o_index
        except AttributeError:
            return

    @o_index.setter
    def o_index(self, index):
        self._o_index = index

    # @property
    # def number(self):
    #     return self._number
    #     # try:
    #     #     return self.ca.residue_number
    #     # except AttributeError:
    #     #     return self.n.residue_number
    #
    # @number.setter
    # def number(self, number):
    #     self._number = number
    #
    # @property
    # def number_pdb(self):
    #     return self._number_pdb
    #
    # @number_pdb.setter
    # def number_pdb(self, number_pdb):
    #     self._number_pdb = number_pdb
    #     # try:
    #     #     return self.ca.pdb_residue_number
    #     # except AttributeError:
    #     #     return self.n.pdb_residue_number
    #
    # @property
    # def chain(self):
    #     return self._chain
    #     # try:
    #     #     return self.ca.chain
    #     # except AttributeError:
    #     #     return self.n.chain
    # @chain.setter
    # def chain(self, chain):
    #     self._chain = chain
    #
    # @property
    # def type(self):
    #     return self._type
    #     # try:
    #     #     return self.ca.residue_type
    #     # except AttributeError:
    #     #     return self.n.chain
    #
    # @type.setter
    # def type(self, _type):
    #     self._type = _type

    @property
    def secondary_structure(self) -> str:
        """Return the secondary structure designation as defined by a secondary structure calculation"""
        try:
            return self._secondary_structure
        except AttributeError:
            raise AttributeError(f'Residue {self.number}{self.chain} has no ".secondary_structure" attribute! Ensure '
                                 f'you call Structure.get_secondary_structure() before you request Residue '
                                 f'secondary structure information')

    @secondary_structure.setter
    def secondary_structure(self, ss_code: str):
        self._secondary_structure = ss_code

    def _segregate_sasa(self):
        """Separate sasa into apolar and polar attributes according to underlying Atoms"""
        residue_atom_polarity = atomic_polarity_table[self.type]
        polarity_list = [[], [], []]  # apolar = 0, polar = 1, unknown = 2 (-1)
        for atom in self.atoms:
            polarity_list[residue_atom_polarity.get(atom.type)].append(atom.sasa)

        self._sasa_apolar, self._sasa_polar, _ = map(sum, polarity_list)
        # if _ > 0:
        #     print('Found %f unknown surface area' % _)

    @property
    def sasa(self) -> float:
        """Return the solvent accessible surface area as calculated by a solvent accessible surface area calculator"""
        try:
            return self._sasa
        except AttributeError:
            try:
                self._sasa = self.sasa_apolar + self.sasa_polar
                return self._sasa
            except AttributeError:
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".sasa" attribute! Ensure you call '
                                     f'Structure.get_sasa() before you request Residue SASA information')

    @sasa.setter
    def sasa(self, sasa: float):
        self._sasa = sasa

    @property
    def sasa_apolar(self) -> float:
        """Return the apolar solvent accessible surface area as calculated by a solvent accessible surface area
        calculator
        """
        try:
            return self._sasa_apolar
        except AttributeError:
            try:
                self._segregate_sasa()
                return self._sasa_apolar
            except AttributeError:  # missing atom.sasa
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".sasa_apolar" attribute! Ensure you '
                                     f'call Structure.get_sasa() before you request Residue SASA information')

    @sasa_apolar.setter
    def sasa_apolar(self, sasa: float | int):
        self._sasa_apolar = sasa

    @property
    def sasa_polar(self) -> float:
        """Return the polar solvent accessible surface area as calculated by a solvent accessible surface area
        calculator
        """
        try:
            return self._sasa_polar
        except AttributeError:
            try:
                self._segregate_sasa()
                return self._sasa_polar
            except AttributeError:  # missing atom.sasa
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".sasa_polar" attribute! Ensure you '
                                     f'call Structure.get_sasa() before you request Residue SASA information')

    @sasa_polar.setter
    def sasa_polar(self, sasa: float | int):
        self._sasa_polar = sasa

    @property
    def relative_sasa(self) -> float:
        """The solvent accessible surface area relative to the standard surface accessibility of the Residue type"""
        # try:  # catch this in self.sasa
        return self.sasa / gxg_sasa[self.type]
        # except AttributeError:
        #     raise AttributeError(f'Residue {self.number}{self.chain} has no ".sasa" attribute! Ensure you call '
        #                          f'Structure.get_sasa() before you request Residue specific SASA information')

    @property
    def contact_order(self) -> float:
        """The Residue contact order, which describes how far away each Residue makes contacts in the polymer chain"""
        try:
            return self._contact_order
        except AttributeError:
            raise AttributeError(f'Residue {self.number}{self.chain} has no ".contact_order" attribute! Ensure you call'
                                 ' Structure.contact_order before you request Residue contact order information')

    @contact_order.setter
    def contact_order(self, contact_order: float):
        self._contact_order = contact_order

    @property
    def number_of_atoms(self) -> int:
        return len(self._atom_indices)

    @property
    def number_of_heavy_atoms(self) -> int:
        return len(self._heavy_atom_indices)

    @property
    def b_factor(self) -> float:
        try:
            return sum(atom.b_factor for atom in self.atoms) / self.number_of_atoms
        except ZeroDivisionError:
            return 0.

    @b_factor.setter
    def b_factor(self, dtype: str | Iterable[float] = None, **kwargs):
        """Set the temperature factor for the Atoms in the Residue

        Args:
            dtype: The data type that should fill the temperature_factor from Residue attributes
                or an iterable containing the explicit b_factor float values
        """
        try:
            for atom in self.atoms:
                atom.b_factor = getattr(self, dtype)
        except AttributeError:
            raise AttributeError(f'The attribute {dtype} was not found in the Residue {self.number}{self.chain}. Are '
                                 f'you sure this is the attribute you want?')
        except TypeError:
            # raise TypeError(f'{type(dtype)} is not a string. To set b_factor, you must provide the dtype as a string')
            try:
                for atom, b_fact in zip(self.atoms, dtype):
                    atom.b_factor = b_fact
            except TypeError:
                raise TypeError(f'{type(dtype)} is not a string nor an iterable. To set b_factor, you must provide the '
                                f'dtype as a string specifying a Residue attribute or an iterable with length = '
                                f'Residue.number_of_atoms')

    def mutation_possibilities_from_directive(self, directive: directives = None, background: set[str] = None,
                                              special: bool = False, **kwargs) -> set[str]:
        """Select mutational possibilities for each Residue based on the Residue and a directive

        Args:
            directive: Where the choice is one of 'special', 'same', 'different', 'charged', 'polar', 'apolar',
                'hydrophobic', 'aromatic', 'hbonding', 'branched'
            background: The background amino acids to compare possibilities against
            special: Whether to include special residues

        Returns:
            The possible amino acid types available given the mutational directive
        """
        if not directive or directive not in mutation_directives:
            self.log.debug(f'{self.mutation_possibilities_from_directive.__name__}: The mutation directive {directive} '
                           f'is not a valid directive yet. Possible directives are: {", ".join(mutation_directives)}')
            return set()
            # raise TypeError('%s: The mutation directive %s is not a valid directive yet. Possible directives are: %s'
            #                 % (self.mutation_possibilities_from_directive.__name__, directive,
            #                    ', '.join(mutation_directives)))

        current_properties = residue_properties[self.type]
        if directive == 'same':
            properties = current_properties
        elif directive == 'different':  # hmm not right -> .difference({hbonding, branched}) <- for ex. polar if apolar
            properties = set(aa_by_property.keys()).difference(current_properties)
        else:
            properties = [directive]
        available_aas = set(aa for prop in properties for aa in aa_by_property[prop])

        if directive != 'special' and not special:
            available_aas = available_aas.difference(aa_by_property['special'])
        if background:
            available_aas = background.intersection(available_aas)

        return available_aas

    def distance(self, other: Residue, dtype: str = 'ca') -> float:
        """Return the distance from this Residue to another specified by atom type "dtype"

        Args:
            other: The other Residue to measure against
        Returns:
            The euclidean distance between the specified Atom type
        """
        return np.linalg.norm(getattr(self, f'.{dtype}_coords') - getattr(other, f'.{dtype}_coords'))

    # def in_contact(self, other_residue, distance_thresh=4.5, side_chain_only=False):
    #     if side_chain_only:
    #         for atom in self.atoms:
    #             if not atom.is_backbone():
    #                 for other_atom in other_residue.atoms:
    #                     if not other_atom.is_backbone():
    #                         if atom.distance(other_atom, intra=True) < distance_thresh:
    #                             return True
    #     else:
    #         for atom in self.atoms:
    #             for other_atom in other_residue.atoms:
    #                 if atom.distance(other_atom, intra=True) < distance_thresh:
    #                     return True
    #     return False

    # def in_contact_residuelist(self, residuelist, distance_thresh=4.5, side_chain_only=False):  # UNUSED
    #     for residue in residuelist:
    #         if self.in_contact(residue, distance_thresh, side_chain_only):
    #             return True
    #     return False

    # def residue_string(self, pdb: bool = False, chain: str = None, **kwargs) -> tuple[str, str, str]:
    #     """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
    #     formatting
    #
    #     Args:
    #         pdb: Whether the Residue representation should use the pdb number at file parsing
    #         chain: The ID of the chain to use
    #     Returns:
    #         Tuple of formatted Residue attributes
    #     """
    #     return format(self.type, '3s'), (chain or self.chain), \
    #         format(getattr(self, f'number{"_pdb" if pdb else ""}'), '4d')

    def __getitem__(self, idx) -> Atom:
        return self.atoms[idx]

    def __key(self) -> tuple[int, int, str]:
        return self._start_index, self.number_of_atoms, self.type

    def __eq__(self, other: Residue) -> bool:
        if isinstance(other, Residue):
            return self.__key() == other.__key()
        raise NotImplementedError(f'Can\' compare {Residue.__name__} instance to {other.__name__} instance')

    def __str__(self, pdb: bool = False, chain: str = None, atom_offset: int = 0, **kwargs) -> str:
        #         type=None, number=None
        """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
        formatting

        Args:
            pdb: Whether the Residue representation should use the pdb number at file parsing
            chain: The ID of the chain to use
            atom_offset: How much to offset the atom index
        Returns:
            The archived .pdb formatted ATOM record for the Residue
        """
        # format the string returned from each Atom, such as
        #  'ATOM  %s  CG2 %s %s%s    %s  1.00 17.48           C  0'
        #       AtomIdx  TypeChNumberCoords
        # To
        #  'ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  0'
        # self.type, self.alt_location, self.code_for_insertion, self.occupancy, self.b_factor,
        #                     self.element_symbol, self.atom_charge)
        # res_str = self.residue_string(**kwargs)
        res_str = format(self.type, '3s'), (chain or self.chain), \
            format(getattr(self, f'number{"_pdb" if pdb else ""}'), '4d')
        offset = 1 + atom_offset
        # limit idx + offset with [-5:] to keep pdb string to a minimum
        return '\n'.join(self._atoms.atoms[idx].__str__(**kwargs)
                         % (format(idx + offset, '5d')[-5:], *res_str, '{:8.3f}{:8.3f}{:8.3f}'.format(*coord))
                         for idx, coord in zip(self._atom_indices, self.coords.tolist()))

    def __hash__(self) -> int:
        return hash(self.__key())


class Residues:
    residues: np.ndarray

    def __init__(self, residues: list[Residue]):
        self.residues = np.array(residues)

    def reindex_residue_atoms(self, start_at: int = 0):  # , offset=None):
        """Set each member Residue indices according to incremental Atoms/Coords index

        Args:
            start_at: The integer to start renumbering Residue, Atom objects at
        """
        if start_at > 0:  # if not 0 or negative
            if start_at < self.residues.shape[0]:  # if in the Residues index range
                prior_residue = self.residues[start_at - 1]
                # prior_residue.start_index = start_at
                for residue in self.residues[start_at:].tolist():
                    residue.start_index = prior_residue._atom_indices[-1] + 1
                    prior_residue = residue
            else:
                # self.residues[-1].start_index = self.residues[-2]._atom_indices[-1] + 1
                raise DesignError('%s: Starting index is outside of the allowable indices in the Residues object!'
                                  % Residues.reindex_residue_atoms.__name__)
        else:  # when start_at is 0 or less
            prior_residue = self.residues[start_at]
            prior_residue.start_index = start_at
            for residue in self.residues[start_at + 1:].tolist():
                residue.start_index = prior_residue._atom_indices[-1] + 1
                prior_residue = residue

    def insert(self, new_residues: list | np.ndarray, at: int = None):
        """Insert Residue(s) into the Residues object

        Args:
            new_residues: The residues to include into Residues
            at: The index to perform the insert at
        """
        self.residues = np.concatenate((self.residues[:at] if 0 <= at <= len(self.residues) else self.residues,
                                        new_residues if isinstance(new_residues, Iterable) else [new_residues],
                                        self.residues[at:] if at is not None else []))

    def set_attributes(self, **kwargs):
        """Set Residue attributes passed by keyword to their corresponding value"""
        for residue in self:
            for key, value in kwargs.items():
                setattr(residue, key, value)

    def set_attribute_from_array(self, **kwargs):  # UNUSED
        """For each Residue, set the attribute passed by keyword to the attribute corresponding to the Residue index in
        a provided array

        Ex: residues.attribute_from_array(mutation_rate=residue_mutation_rate_array)
        """
        for idx, residue in enumerate(self):
            for key, value in kwargs.items():
                setattr(residue, key, value[idx])

    def __copy__(self) -> Residues:
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.residues = self.residues.copy()
        for idx, residue in enumerate(other.residues):
            other.residues[idx] = copy(residue)

        return other

    def __len__(self) -> int:
        return self.residues.shape[0]  # len(self.residues)

    def __iter__(self) -> Residue:
        yield from self.residues.tolist()


class GhostFragment:
    guide_coords: np.ndarray
    i_type: int
    j_type: int
    k_type: int
    rmsd: float
    aligned_fragment: MonoFragment

    def __init__(self, guide_coords: np.ndarray, i_type: int, j_type: int, k_type: int, ijk_rmsd: float,
                 aligned_fragment: MonoFragment):  # structure
        self.guide_coords = guide_coords
        self.i_type = i_type
        self.j_type = j_type
        self.k_type = k_type
        self.rmsd = ijk_rmsd
        self.aligned_fragment = aligned_fragment

    def get_ijk(self) -> tuple[int, int, int]:
        """Return the fragments corresponding cluster index information

        Returns:
            I cluster index, J cluster index, K cluster index
        """
        return self.i_type, self.j_type, self.k_type

    def get_aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the MonoFragment identifiers that the GhostFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.aligned_fragment.central_residue.chain, self.aligned_fragment.number

    @property
    def number(self) -> int:
        """The Residue number of the aligned Residue"""
        return self.aligned_fragment.number

    # @property
    # def structure(self):
    #     return self._structure
    #
    # @structure.setter
    # def structure(self, structure):
    #     self._structure = structure

    # def get_guide_coords(self):  # UNUSED
    #     return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class MonoFragment:
    central_residue: Residue
    guide_coords: np.ndarray
    i_type: int
    rotation: np.ndarray
    translation: np.ndarray
    template_coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])

    def __init__(self, residues: Sequence[Residue], representatives: dict[int, np.ndarray] = None,
                 fragment_type: int = None, guide_coords: np.ndarray = None, fragment_length: int = 5,
                 rmsd_thresh: float = 0.75):
        self.i_type = fragment_type
        self.guide_coords = guide_coords
        self.central_residue = residues[int(fragment_length/2)]

        if not residues:
            raise ValueError(f'Can\'t find {MonoFragment.__name__} without passing residues')
        elif not representatives:
            raise ValueError(f'Can\'t find {MonoFragment.__name__} without passing representatives')

        frag_ca_coords = np.array([residue.ca_coords for residue in residues])
        min_rmsd = float('inf')
        for cluster_type, cluster_coords in representatives.items():
            rmsd, rot, tx, _ = superposition3d(frag_ca_coords, cluster_coords)
            if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                self.i_type = cluster_type
                min_rmsd, self.rotation, self.translation = rmsd, rot, tx

        if self.i_type:
            self.guide_coords = \
                np.matmul(MonoFragment.template_coords, np.transpose(self.rotation)) + self.translation

    @property
    def transformation(self) -> dict[str, np.ndarray]:
        return dict(rotation=self.rotation, translation=self.translation)

    def get_central_res_tup(self) -> tuple[str, int]:
        return self.central_residue.chain, self.central_residue.number

    # def get_guide_coords(self):  # UNUSED
    #     return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul([0.33333, 0.33333, 0.33333], self.guide_coords)
    #     else:
    #         return None

    # @property
    # def structure(self):
    #     return self._structure

    # @structure.setter
    # def structure(self, structure):
    #     self._structure = structure

    @property
    def number(self) -> int:
        """The Residue number"""
        return self.central_residue.number
    #
    # def chain(self):
    #     return self.central_residue.chain

    def get_ghost_fragments(self, indexed_ghost_fragments: dict, bb_balltree: BallTree, clash_dist: float = 2.2) -> \
            list | list[GhostFragment]:
        """Find all the GhostFragments associated with the MonoFragment that don't clash with the original structure
        backbone

        Args:
            indexed_ghost_fragments: The paired fragment database to match to the MonoFragment instance
            bb_balltree: The backbone of the structure to assign fragments to
            clash_dist: The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        ghost_i_type = indexed_ghost_fragments.get(self.i_type, None)
        if not ghost_i_type:
            return []

        stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array = ghost_i_type
        transformed_bb_coords = transform_coordinate_sets(stacked_bb_coords, **self.transformation)
        transformed_guide_coords = transform_coordinate_sets(stacked_guide_coords, **self.transformation)
        neighbors = bb_balltree.query_radius(transformed_bb_coords.reshape(-1, 3), clash_dist)  # queries on a np.view
        neighbor_counts = np.array([neighbor.size for neighbor in neighbors])
        # reshape to original size then query for existence of any neighbors for each fragment individually
        viable_indices = neighbor_counts.reshape(transformed_bb_coords.shape[0], -1).any(axis=1)
        ghost_frag_info = \
            zip(list(transformed_guide_coords[~viable_indices]), *zip(*ijk_types[~viable_indices].tolist()),
                rmsd_array[~viable_indices].tolist(), repeat(self))

        return [GhostFragment(*info) for info in ghost_frag_info]

    # Methods below make MonoFragment compatible with pose symmetry operations
    @property
    def coords(self) -> np.ndarray:
        return self.guide_coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self.guide_coords = coords

    def return_transformed_copy(self, rotation: list | np.ndarray = None, translation: list | np.ndarray = None,
                                rotation2: list | np.ndarray = None, translation2: list | np.ndarray = None) -> \
            MonoFragment:
        """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space

        Transformation proceeds by matrix multiplication with the order of operations as:
        rotation, translation, rotation2, translation2

        Args:
            rotation: The first rotation to apply, expected general rotation matrix shape (3, 3)
            translation: The first translation to apply, expected shape (3)
            rotation2: The second rotation to apply, expected general rotation matrix shape (3, 3)
            translation2: The second translation to apply, expected shape (3)
        Returns:
            A transformed copy of the original object
        """
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.guide_coords, np.transpose(rotation))
        else:
            new_coords = self.guide_coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(new_coords, np.transpose(rotation2))

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        new_structure = copy(self)
        new_structure.guide_coords = new_coords

        return new_structure

    def replace_coords(self, new_coords: np.ndarray):
        self.guide_coords = new_coords


class StructureBase:
    """Collect extra keyword arguments"""
    def __init__(self, chains=None, entities=None, seqres=None, multimodel=None, pose_format=None, sequence=None,
                 cryst=None, cryst_record=None, design=None, resolution=None, space_group=None, query_by_sequence=True,
                 entity_names=None, rename_chains=None, biomt=None, biomt_header=None, **kwargs):
        try:
            super().__init__(**kwargs)
        except TypeError:
            raise TypeError('The argument(s) passed to the Structure object were not recognized: %s' %
                            ', '.join(kwargs.keys()))


class Structure(StructureBase):
    """Structure object handles Atom/Residue/Coords manipulation of all Structure containers.
    Must pass atoms, residues, residue_indices, or coords to use most methods without issues
    to initialize
    """
    _residue_indices: list[int] | None
    available_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # '0123456789~!@#$%^&*()-+={}[]|:;<>?'

    def __init__(self, atoms=None, residues=None, residue_indices=None, name=None, coords=None, log=None, **kwargs):
        self._atoms = None
        self._atom_indices = None
        self._coords = None
        self._coords_indexed_residues = None
        self._residues = None
        self._residue_indices = None
        self.biomt = kwargs.get('biomt', [])  # list of vectors to format
        self.biomt_header = kwargs.get('biomt_header', '')  # str with already formatted header
        self.name = name
        self.secondary_structure = None
        self.sasa = None
        self.structure_containers = []

        if log:
            if isinstance(log, Log):
                self._log = log
            else:
                self._log = Log(log)
                # self.log = log
        elif log is None:
            # self.log = Log(null_log)
            self._log = Log(null_log)
        else:  # When log is explicitly passed as False, use the module logger
            # self.log = Log(logger)
            self._log = Log(logger)

        if atoms is not None:  # Todo make look like below residues init!
            self.atoms = atoms
            if coords is None:  # assumes that this is a Structure init without existing shared coords
                try:
                    # coords = [atom.coords for atom in atoms]
                    self.set_coords(coords=np.concatenate([atom.coords for atom in atoms]))
                except AttributeError:
                    raise DesignError('Can\'t initialize Structure with Atom objects lacking coords when no Coords '
                                      'object is passed! Either pass Atom objects with coords attribute or pass Coords')
        if residues is not None:
            if not residue_indices:  # assume that the passed residues shouldn't be bound to an existing Structure
                atoms = []
                for residue in residues:
                    atoms.extend(residue.atoms)
                self.atom_indices = list(range(len(atoms)))
                self.residue_indices = list(range(len(residues)))  # residue_indices
                self.atoms = atoms
                if isinstance(residues, Residues):  # already have a residues object
                    pass
                else:  # must create the residues object
                    residues = Residues(residues)
                # have to copy Residues object to set new attributes on each member Residue
                self.residues = copy(residues)
                # set residue attributes, index according to new Atoms/Coords index
                # self._residues.set_attributes(_atoms=self._atoms)  # , _coords=self._coords) <- done in set_coords
                self.set_residues_attributes(_atoms=self._atoms)  # , _coords=self._coords) <- done in set_coords
                self._residues.reindex_residue_atoms()
                self.set_coords(coords=np.concatenate([residue.coords for residue in residues]))
            else:
                self.residue_indices = residue_indices
                self.set_residue_slice(residues)
                self.coords = coords
                # self.set_residues(residues)
            # if coords is None:  # assumes that this is a Structure init without existing shared coords
            #     # try:
            #     #     self.coords = self.residues[0]._coords
            #     coords = np.concatenate([residue.coords for residue in residues])
            #     self.set_coords(coords=coords)
            #     # except (IndexError, AssertionError):  # self.residues[0]._coords isn't the correct size
            #     #     self.coords = None
        # if coords is not None:  # must go after Atom containers as atoms don't have any/right coordinate info
        #     self.coords = coords

        super().__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms=None, coords=None, **kwargs):
        return cls(atoms=atoms, coords=coords, **kwargs)
        # new_structure = cls(atoms=atoms, coords=coords, **kwargs)
        # new_structure.set_coords(coords)
        # return new_structure

    @classmethod
    def from_residues(cls, residues=None, residue_indices=None, coords=None, **kwargs):
        return cls(residues=residues, residue_indices=residue_indices, coords=coords, **kwargs)
        # new_structure = cls(residues=residues, residue_indices=residue_indices, coords=coords, **kwargs)
        # new_structure.set_coords(coords)
        # return new_structure

    # @property
    # def name(self):
    #     """Returns: (str)"""
    #     return self._name
    #
    # @name.setter
    # def name(self, name):
    #     self._name = name

    @property
    def log(self) -> Logger:
        """Returns the log object holding the Logger"""
        return self._log.log

    @log.setter
    def log(self, log):  # log_object):
        """Set the Structure, Atom, and Residue log with specified Log Object"""
        # try:
        #     log_object.log
        # except AttributeError:
        #     log_object = Log(log_object)
        # self._log = log_object
        if isinstance(log, Logger):  # prefer this protection method versus Log.log property overhead?
            self._log.log = log
        else:
            raise TypeError('The log type (%s) is not of the specified type logging.Logger' % type(log))

    # def set_log(self, log=None):
    #     """Set the log for the Structure as a Log object. Additionally, updates all member Residues with the
    #     Log object and maps the atom/coordinate index to each Residue, residue atom index pair.
    #
    #     Keyword Args:
    #         log=None (numpy.ndarray | list): The coordinates to set for the structure
    #     """
    #     self.log = log
    #     # self.set_atoms_attributes(coords=self._coords)  # atoms doesn't have coords now
    #     self._residues.set_attributes(log=self._log)
    #     # self.set_residues_attributes(log=self._log)

    @property
    def contains_hydrogen(self) -> bool:
        """Returns whether the Structure contains hydrogen atoms"""
        return self.residues[0].contains_hydrogen

    @property
    def sequence(self) -> str:
        """Holds the Structure amino acid sequence"""
        # Todo if the Structure is mutated, this mechanism will cause errors, must re-extract sequence
        try:
            return self._sequence
        except AttributeError:
            self._sequence = self.get_structure_sequence()
            return self._sequence

    @sequence.setter
    def sequence(self, sequence):
        self._sequence = sequence

    @property
    def coords(self) -> np.ndarray:
        """Return the atomic coordinates for the Atoms in the Structure"""
        return self._coords.coords[self._atom_indices]

    @coords.setter
    def coords(self, coords):
        """Replace the Structure, Atom, and Residue coordinates with specified Coords Object or numpy.ndarray"""
        try:
            coords.coords
        except AttributeError:  # not yet a Coords object, so create one
            coords = Coords(coords)
        self._coords = coords

        if self._coords.coords.shape[0] != 0:
            assert len(self.atoms) == len(self.coords), '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
                                                        % (self.name, len(self.atoms), len(self.coords))

    def set_coords(self, coords: np.ndarray | list = None):
        """Set the coordinates for the Structure as a Coord object. Additionally, updates all member Residues with the
        Coords object and maps the atom/coordinate index to each Residue, residue atom index pair.

        Only use set_coords once per Structure object creation otherwise Structures with multiple containers will be
        corrupted
        Args:
            coords: The coordinates to set for the structure
        """
        self.coords = coords
        # self.set_atoms_attributes(coords=self._coords)  # atoms doesn't have coords now
        self.set_residues_attributes(coords=self._coords)
        # self._residues.set_attributes(coords=self._coords)
        # self.store_coordinate_index_residue_map()
        # self.coords_indexed_residues = \
        #     [(res_idx, res_atom_idx) for res_idx, residue in enumerate(self.residues) for res_atom_idx in residue.range]
        # res_idx_atom_idx = \
        #     [(res_idx, res_atom_idx) for res_idx, residue in enumerate(self.residues) for res_atom_idx in residue.range]
        # residue_indices, residue_atom_indices = zip(*res_idx_atom_idx)
        residues_atom_idx = [(residue, res_atom_idx) for residue in self.residues for res_atom_idx in residue.range]
        self.coords_indexed_residues, self.coords_indexed_residue_atoms = zip(*residues_atom_idx)
        range_idx, prior_range_idx = 0, 0
        residue_indexed_ranges = []
        for residue in self.residues:
            range_idx += residue.number_of_atoms
            residue_indexed_ranges.append(list(range(prior_range_idx, range_idx)))
            prior_range_idx = range_idx
        self.residues_indexed_coords_indices = residue_indexed_ranges

    @property
    def is_structure_owner(self) -> bool:
        """Check to see if the Structure is the owner of it's Coord and Atom attributes or if there is a larger
        Structure that maintains them"""
        return self._coords_indexed_residues is not None

    @property
    def atom_indices(self) -> list[int]:  # In Residue too
        """Return the atom indices which belong to the Structure"""
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices: list[int]):
        self._atom_indices = indices

    def start_indices(self, dtype=None, at=0):
        """Modify Structure container indices by a set integer amount"""
        try:
            indices = getattr(self, '%s_indices' % dtype)
        except AttributeError:
            raise AttributeError('The dtype "%s"_indices was not found the Structure object. Possible dtype values '
                                 'are atom or residue' % dtype)
        first_index = indices[0]
        setattr(self, '%s_indices' % dtype, [at + prior_idx - first_index for prior_idx in indices])

    def insert_indices(self, at=0, new_indices=None, dtype=None):
        """Modify Structure container indices by a set integer amount"""
        if new_indices is None:
            new_indices = []
        try:
            indices = getattr(self, '%s_indices' % dtype)
        except AttributeError:
            raise AttributeError('The dtype %s_indices was not found the Structure object. Possible values of dtype are'
                                 ' atom or residue' % dtype)
        number_new = len(new_indices)
        setattr(self, '%s_indices' % dtype, indices[:at] + new_indices + [idx + number_new for idx in indices[at:]])

    @property
    def atoms(self) -> list[Atom]:
        """Return the Atom objects in the Structure"""
        return self._atoms.atoms[self._atom_indices].tolist()

    @atoms.setter
    def atoms(self, atoms: Atoms | list[Atom]):
        """Set the Structure atoms to an Atoms object"""
        if isinstance(atoms, Atoms):
            self._atoms = atoms
        else:
            self._atoms = Atoms(atoms)

    def set_atoms(self, atoms):
        """Set the Structure atom indices, atoms to an Atoms object, and create Residue objects"""
        self.atom_indices = list(range(len(atoms)))  # [atom.index for atom in atoms]
        self.atoms = atoms
        # self.atom_indices = list(range(len(atom_list)))  # can't set here as may contain other atoms
        self.create_residues()
        # self.set_residues_attributes(_atoms=atoms)

    @property
    def number_of_atoms(self) -> int:
        """Return the number of atoms/coords in the Structure"""
        return len(self._atom_indices)

    @property
    def residue_indices(self) -> list[int]:
        """Return the residue indices which belong to the Structure"""
        return self._residue_indices

    @residue_indices.setter
    def residue_indices(self, indices: list[int]):
        self._residue_indices = indices  # np.array(indices)

    @property
    def residues(self) -> list[Residue]:  # TODO Residues iteration
        """Return the Residue objects in the Structure"""
        return self._residues.residues[self._residue_indices].tolist()

    @residues.setter
    def residues(self, residues: Residues | list[Residue]):
        """Set the Structure atoms to a Residues object"""
        if isinstance(residues, Residues):
            self._residues = residues
        else:
            self._residues = Residues(residues)

    # def store_coordinate_index_residue_map(self):
    #     self.coords_indexed_residues = [(residue, res_idx) for residue in self.residues for res_idx in residue.range]

    # @property
    # def coords_indexed_residues(self):
    #     """Returns a map of the Residues and Residue atom_indices for each Coord in the Structure
    #
    #     Returns:
    #         (list[tuple[Residue, int]]): Indexed by the by the Residue position in the corresponding .coords attribute
    #     """
    #     try:
    #         return [(self._residues.residues[res_idx], res_atom_idx)
    #                 for res_idx, res_atom_idx in self._coords_indexed_residues[self.atom_indices].tolist()]
    #     except (AttributeError, TypeError):
    #         raise AttributeError('The current Structure object "%s" doesn\'t "own" it\'s coordinates. The attribute '
    #                              '.coords_indexed_residues can only be accessed by the Structure object that owns these'
    #                              ' coordinates and therefore owns this Structure' % self.name)
    #
    # @coords_indexed_residues.setter
    # def coords_indexed_residues(self, index_pairs):
    #     """Create a map of the coordinate indices to the Residue and Residue atom index"""
    #     self._coords_indexed_residues = np.array(index_pairs)

    @property
    def residues_indexed_coords_indices(self) -> list[list[int]]:
        """Returns a map of the Residue atom/coord_indices, indexed to the .coords indices for each Residue in the
        Structure

        Returns:
            Residue objects indexed by the Residue position in the corresponding .coords attribute
        """
        try:
            return self._residues_indexed_coords_indices.tolist()  # [self.atom_indices]
        except (AttributeError, TypeError):
            raise AttributeError('The Structure "%s" doesn\'t "own" it\'s coordinates. The attribute '
                                 '.residues_indexed_coords_indices can only be accessed by the Structure object that '
                                 'owns these coordinates and therefore owns this Structure' % self.name)

    @residues_indexed_coords_indices.setter
    def residues_indexed_coords_indices(self, indices):
        """Create a map of the coordinate indices to the Residue and Residue atom index"""
        self._residues_indexed_coords_indices = np.array(indices, dtype=object)

    @property
    def coords_indexed_residues(self) -> list[Residue]:
        """Returns a map of the Residues for each Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .coords attribute
        """
        try:
            return self._coords_indexed_residues[self.atom_indices].tolist()
        except (AttributeError, TypeError):
            raise AttributeError('The Structure "%s" doesn\'t "own" it\'s coordinates. The attribute '
                                 '.coords_indexed_residues can only be accessed by the Structure object that owns these'
                                 ' coordinates and therefore owns this Structure' % self.name)  # Todo self.owner

    @coords_indexed_residues.setter
    def coords_indexed_residues(self, residues):
        """Create a map of the coordinate indices to the Residue"""
        self._coords_indexed_residues = np.array(residues)

    @property
    def coords_indexed_residue_atoms(self) -> list[int]:
        """Returns a map of the Residue atom_indices for each Coord in the Structure

        Returns:
            Index of the Atom position in the Residue for the index of the .coords attribute
        """
        try:
            return self._coords_indexed_residue_atoms[self.atom_indices].tolist()
        except (AttributeError, TypeError):
            raise AttributeError('The Structure "%s" doesn\'t "own" it\'s coordinates. The attribute '
                                 '.coords_indexed_residue_atoms can only be accessed by the Structure object that owns '
                                 'these coordinates and therefore owns this Structure' % self.name)  # Todo self.owner

    @coords_indexed_residue_atoms.setter
    def coords_indexed_residue_atoms(self, indices):
        """Create a map of the coordinate indices to the Residue and Residue atom index"""
        self._coords_indexed_residue_atoms = np.array(indices)

    @property
    def number_of_residues(self) -> int:
        """Access the number of Residues in the Structure"""
        return len(self._residue_indices)

    @property
    def center_of_mass(self) -> np.ndarray:
        """Returns: (numpy.ndarray)"""
        structure_length = self.number_of_atoms
        return np.matmul(np.full(structure_length, 1 / structure_length), self.coords)
        # try:
        #     return self._center_of_mass
        # except AttributeError:
        #     self.find_center_of_mass()
        #     return self._center_of_mass

    # def find_center_of_mass(self):
    #     """Retrieve the center of mass for the specified Structure"""
    #     divisor = 1 / self.number_of_atoms
    #     self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

    # def get_coords(self):
    #     """Return a view of the Coords from the Structure
    #
    #     Returns:
    #         (Numpy.ndarray)
    #     """
    #     return self._coords[self.atom_indices]

    def get_backbone_coords(self):
        """Return a view of the Coords from the Structure with only backbone atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [atom.index for atom in self.atoms if atom.is_backbone()]
        return self._coords.coords[self.backbone_indices]

    def get_backbone_and_cb_coords(self):
        """Return a view of the Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [atom.index for atom in self.atoms if atom.is_backbone() or atom.is_cb()]
        return self._coords.coords[self.backbone_and_cb_indices]

    def get_ca_coords(self):
        """Return a view of the Coords from the Structure with CA atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [residue.ca.index for residue in self.residues]
        # index_mask = [atom.index for atom in self.atoms if atom.is_ca()]
        return self._coords.coords[self.ca_indices]

    def get_cb_coords(self):
        """Return a view of the Coords from the Structure with CB atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [residue.cb.index for residue in self.residues]
        # index_mask = [atom.index for atom in self.atoms if atom.is_cb(InclGlyCA=InclGlyCA)]
        # return self._coords.coords[index_mask]
        return self._coords.coords[self.cb_indices]

    def get_coords_subset(self, res_start, res_end, ca=True):
        selected_atoms, out_coords = [], []
        if ca:
            for residue in self.get_residues(range(res_start, res_end + 1)):
                out_coords.extend(residue.ca_coords)
        else:
            for residue in self.get_residues(range(res_start, res_end + 1)):
                out_coords.extend(residue.coords)

        return out_coords
    # def atoms(self):
    #     """Retrieve Atoms in structure. Returns all by default. If numbers=(list) selected Atom numbers are returned
    #     Returns:
    #         (list[Atom])
    #     """
    #     if numbers and isinstance(numbers, Iterable):
    #         return [atom for atom in self.atoms if atom.number in numbers]
    #     else:
    #         return self._atoms

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the Structure instance"""
        raise DesignError('This function (add_atoms) is currently broken')  # TODO BROKEN
        atoms = self.atoms.tolist()
        atoms.extend(atom_list)
        self.atoms = atoms
        # Todo need to update all referrers
        # Todo need to add the atoms to coords

    def update_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all Structure container members"""
        for structure_type in self.structure_containers:
            structure = getattr(self, structure_type)
            # print('Updating %s attributes %s' % (structure, kwargs))
            self.set_structure_attributes(structure, **kwargs)
        # # self.set_structure_attributes(self.atoms, **kwargs)
        # for kwarg, value in kwargs.items():
        #     setattr(self, kwarg, value)
        # # self.set_structure_attributes(self.residues, **kwargs)

    def set_atoms_attributes(self, **kwargs):  # Same function in Residue
        """Set attributes specified by key, value pairs for all Atoms in the Structure"""
        for atom in self.atoms:
            for kwarg, value in kwargs.items():
                setattr(atom, kwarg, value)

    def set_residues_attributes(self, numbers=None, **kwargs):  # Depreciated in favor of _residues.set_attributes()
        """Set attributes specified by key, value pairs for all Residues in the Structure"""
        for residue in self.get_residues(numbers=numbers, **kwargs):
            for kwarg, value in kwargs.items():
                setattr(residue, kwarg, value)

    # def set_residues_attributes_from_array(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all Residues in the Structure"""
    #     # self._residues.set_attribute_from_array(**kwargs)
    #     for idx, residue in enumerate(self.residues):
    #         for key, value in kwargs.items():
    #             setattr(residue, key, value[idx])

    @staticmethod
    def set_structure_attributes(structure, **kwargs):
        """Set structure attributes specified by key, value pairs for all object instances in the structure iterator"""
        for obj in structure:
            for kwarg, value in kwargs.items():
                setattr(obj, kwarg, value)

    # def update_structure(self, atom_list):  # UNUSED
    #     # self.reindex_atoms()
    #     # self.coords = np.append(self.coords, [atom.coords for atom in atom_list])
    #     # self.set_atom_coordinates(self.coords)
    #     # self.create_residues()
    #     # self.set_length()

    # def get_atoms_by_indices(self, indices=None):  # UNUSED
    #     """Retrieve Atoms in the Structure specified by indices. Returns all by default
    #
    #     Returns:
    #         (list[Atom])
    #     """
    #     return [self.atoms[index] for index in indices]

    def get_residue_atom_indices(self, numbers=None, **kwargs):
        """Retrieve Atom indices for Residues in the Structure. Returns all by default. If residue numbers are provided
         the selected Residues are returned

        Returns:
            (list[int])
        """
        # return [atom.index for atom in self.get_residue_atoms(numbers=numbers, **kwargs)]
        atom_indices = []
        for residue in self.get_residues(numbers=numbers, **kwargs):
            atom_indices.extend(residue.atom_indices)
        return atom_indices

    def get_residues_by_atom_indices(self, indices=None):
        """Retrieve Residues in the Structure specified by Atom indices.

        Returns:
            (list[Residue])
        """
        if indices:
            residues = set(residue for residue, atom_index in self._coords_indexed_residues[indices].tolist())
            return sorted(residues, key=lambda residue: residue.number)
        else:
            return self.residues
        # residue_numbers = set(atom.residue_number for atom in atoms)
        # if residue_numbers:
        #     return self.get_residues(numbers=residue_numbers)
        # else:
        #     return None

    def reset_indices_attributes(self):
        """Upon loading a new Structure with old Structure object, remove any indices that might have been saved
        Including:
            self._coords_indexed_backbone_indices
            self._coords_indexed_backbone_and_cb_indices
            self._coords_indexed_cb_indices
            self._coords_indexed_ca_indices
            self._backbone_indices
            self._backbone_and_cb_indices
            self._cb_indices
            self._ca_indices
            self._heavy_atom_indices
            self._helix_cb_indices
        """
        structure_indices = ['_coords_indexed_backbone_indices', '_coords_indexed_backbone_and_cb_indices',
                             '_coords_indexed_cb_indices', '_coords_indexed_ca_indices', '_backbone_indices',
                             '_backbone_and_cb_indices', '_cb_indices', '_ca_indices', '_heavy_atom_indices',
                             '_helix_cb_indices']
        # structure_indices = [attribute for attribute in self.__dict__ if attribute.endswith('_indices')]
        # self.log.info('Deleting the following indices: %s' % structure_indices)
        for structure_index in structure_indices:
            self.__dict__.pop(structure_index, None)

    @property
    def coords_indexed_backbone_indices(self):
        """Return backbone Atom indices from the Structure indexed to the Coords view

        Returns:
            (list[int])
        """
        try:
            return self._coords_indexed_backbone_indices
        except AttributeError:
            # for idx, (atom_idx, bb_idx) in enumerate(zip(self.atom_indices, self.backbone_indices)):
            # backbone_indices = []
            # for residue, res_atom_idx in self.coords_indexed_residues:
            #     backbone_indices.extend(residue.backbone_indices)
            test_indices = self.backbone_indices
            self._coords_indexed_backbone_indices = \
                [idx for idx, atom_idx in enumerate(self.atom_indices) if atom_idx in test_indices]
        return self._coords_indexed_backbone_indices

    @property
    def coords_indexed_backbone_and_cb_indices(self):
        """Return backbone and cb Atom indices from the Structure indexed to the Coords view

        Returns:
            (list[int])
        """
        try:
            return self._coords_indexed_backbone_and_cb_indices
        except AttributeError:
            test_indices = self.backbone_and_cb_indices
            self._coords_indexed_backbone_and_cb_indices = \
                [idx for idx, atom_idx in enumerate(self.atom_indices) if atom_idx in test_indices]
        return self._coords_indexed_backbone_and_cb_indices

    @property
    def coords_indexed_cb_indices(self):
        """Return ca Atom indices from the Structure indexed to the Coords view

        Returns:
            (list[int])
        """
        try:
            return self._coords_indexed_cb_indices
        except AttributeError:
            test_indices = self.cb_indices
            self._coords_indexed_cb_indices = \
                [idx for idx, atom_idx in enumerate(self.atom_indices) if atom_idx in test_indices]
        return self._coords_indexed_cb_indices

    @property
    def coords_indexed_ca_indices(self):
        """Return cb Atom indices from the Structure indexed to the Coords view

        Returns:
            (list[int])
        """
        try:
            return self._coords_indexed_ca_indices
        except AttributeError:
            test_indices = self.ca_indices
            self._coords_indexed_ca_indices = \
                [idx for idx, atom_idx in enumerate(self.atom_indices) if atom_idx in test_indices]
        return self._coords_indexed_ca_indices

    @property
    def backbone_indices(self):
        """Return backbone Atom indices from the Structure

        Returns:
            (list[int])
        """
        try:
            return self._backbone_indices
        except AttributeError:
            self._backbone_indices = []
            for residue in self.residues:
                self._backbone_indices.extend(residue.backbone_indices)
            return self._backbone_indices

    @property
    def backbone_and_cb_indices(self):
        """Return backbone and CB Atom indices from the Structure. Inherently gets glycine CA's

        Returns:
            (list[int])
        """
        try:
            return self._backbone_and_cb_indices
        except AttributeError:
            self._backbone_and_cb_indices = []
            for residue in self.residues:
                self._backbone_and_cb_indices.extend(residue.backbone_and_cb_indices)
            return self._backbone_and_cb_indices

    @property
    def cb_indices(self):
        """Return CB Atom indices from the Structure. Inherently gets glycine Ca's and Ca's of Residues missing Cb

        Returns:
            (list[int])
        """
        try:
            return self._cb_indices
        except AttributeError:
            # self._cb_indices = [residue.cb_atom_index for residue in self.residues if residue.cb_atom_index]
            self._cb_indices = [residue.cb_atom_index if residue.cb_atom_index else residue.ca_atom_index
                                for residue in self.residues]
            return self._cb_indices

    @property
    def ca_indices(self):
        """Return CB Atom indices from the Structure

        Returns:
            (list[int])
        """
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = [residue.ca_atom_index for residue in self.residues if residue.ca_atom_index]
            return self._ca_indices

    @property
    def heavy_atom_indices(self):
        """Return Heavy Atom indices from the Structure

        Returns:
            (list[int])
        """
        try:
            return self._heavy_atom_indices
        except AttributeError:
            self._heavy_atom_indices = []
            for residue in self.residues:
                self._heavy_atom_indices.extend(residue.heavy_atom_indices)
            return self._heavy_atom_indices

    @property
    def helix_cb_indices(self):
        """Only works on secondary structure assigned structures!

        Returns:
            (list[int])
        """
        try:
            return self._helix_cb_indices
        except AttributeError:
            h_cb_indices = []
            for residue in self.residues:
                if not residue.secondary_structure:
                    self.log.error('Secondary Structures must be set before finding helical CB\'s! Error at Residue %s'
                                   % residue.number)
                    return
                elif residue.secondary_structure == 'H':
                    h_cb_indices.append(residue.cb_atom_index)
            self._helix_cb_indices = h_cb_indices
            return self._helix_cb_indices

    @property
    def ca_atoms(self):
        """Return CA Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.ca_indices]

    @property
    def cb_atoms(self):
        """Return CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.cb_indices]

    @property
    def backbone_atoms(self):
        """Return backbone Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.backbone_indices]

    @property
    def backbone_and_cb_atoms(self):
        """Return backbone and CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.backbone_and_cb_indices]

    def atom(self, atom_number):
        """Retrieve the Atom specified by atom number

        Returns:
            (list[Atom])
        """
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return

    def renumber_structure(self):
        """Change the Atom and Residue numbering. Access the readtime Residue number in .pdb_number attribute"""
        self.renumber_atoms()
        self.renumber_residues()
        self.log.debug('%s was formatted in Pose numbering (residues now 1 to %d)'
                       % (self.name, self.number_of_residues))

    def renumber_atoms(self):
        """Renumber all Atom objects sequentially starting with 1"""
        for idx, atom in enumerate(self.atoms, 1):
            atom.number = idx

    def renumber_residues(self):
        """Renumber Residue objects sequentially starting with 1"""
        for idx, residue in enumerate(self.residues, 1):
            residue.number = idx

    def reindex_atoms(self, start_at: int = 0, offset: int = None):
        """Reindex all Atom objects after the start_at index in the self.atoms attribute

        Args:
            start_at: The integer to start reindexing Atom objects at
            offset: The integer to offset the index by. Defaults to a subtracting the offset from all subsequent Atoms
        """
        if start_at:
            if offset:
                self.atom_indices = \
                    self.atom_indices[:start_at] + [idx - offset for idx in self.atom_indices[start_at:]]
            else:
                raise ValueError('Must include an offset when re-indexing atoms from a start_at position!')
        else:
            # WARNING, this shouldn't be used for a Structure object whose self._atoms is shared with another Structure!
            self.atom_indices = list(range(len(self.atom_indices)))
        # for idx, atom in enumerate(self.atoms):
        #     self.atoms[idx].index = idx

    # def set_atom_coordinates(self, coords):
    #     """Set/Replace all Atom coordinates with coords specified. Must be in the same order to apply correctly!"""
    #     assert len(self.atoms) == coords.shape[0], '%s: ERROR setting Atom coordinates, # Atoms (%d) !=  # Coords (%d)'\
    #                                                % (self.name, len(self.atoms), coords.shape[0])
    #     self.coords = coords
    #     for idx, atom in enumerate(self.get_atoms):
    #         atom.coords = coords[idx]
    #         # atom.x, atom.y, atom.z = coords[idx][0], coords[idx][1], coords[idx][2]

    def get_residues(self, numbers: Container = None, pdb: bool = False, **kwargs) -> list[Residue]:
        """Retrieve Residue objects in Structure. Returns all by default. If a list of numbers is provided, the selected
        Residues numbers are returned

        Returns:
            The requested Residue objects
        """
        if numbers:
            if isinstance(numbers, Container):
                number_source = 'number_pdb' if pdb else 'number'
                return [residue for residue in self.residues if getattr(residue, number_source) in numbers]
            else:
                self.log.error(f'The passed residue numbers type "{type(numbers)}" must be a Container. Returning all '
                               f'Residues instead')
        return self.residues

    def set_residue_slice(self, residues):
        """Set the Structure Residues to Residues object. Set the Structure Atoms and atom_indices"""
        self.residues = residues
        self.atom_indices = [idx for residue in self.residues for idx in residue.atom_indices]
        self.atoms = self.residues[0]._atoms

    # def set_residues(self, residues):
    #     """Set the Structure Residues to Residues object. Set the Structure Atoms and atom_indices"""
    #     self.residues = residues
    #     # self.atom_indices = [atom.index for residue in self.residues for atom in residue.atoms]
    #     atom_indices = []
    #     for residue in self.residues:
    #         atom_indices.extend(residue.atom_indices)
    #     self.atom_indices = atom_indices
    #     self.atoms = self.residues[0]._atoms

    # def add_residues(self, residue_list):
    #     """Add Residue objects in a list to the Structure instance"""
    #     raise DesignError('This function is broken')  # TODO BROKEN
    #     residues = self.residues
    #     residues.extend(residue_list)
    #     self.set_residues(residues)
    #     # Todo need to add the residue coords to coords

    # update_structure():
    #  self.reindex_atoms() -> self.coords = np.append(self.coords, [atom.coords for atom in atoms]) ->
    #  self.set_atom_coordinates(self.coords) -> self.create_residues() -> self.set_length()

    def create_residues(self):
        """For the Structure, create all possible Residue instances. Doesn't allow for alternative atom locations"""
        # required_types = {'N', 'CA', 'O'}  # 'C', Removing this for fragment library guide atoms...
        required_types = {'N', 'CA', 'C', 'O'}
        new_residues, remove_atom_indices, found_types = [], [], set()
        current_residue_number = self.atoms[0].residue_number
        start_atom_index = 0
        for idx, atom in enumerate(self.atoms):
            # if the current residue number is the same as the prior number and the atom.type is not already present
            # We get rid of alternate conformations upon PDB load, so must be a new residue with bad numbering
            if atom.residue_number == current_residue_number and atom.type not in found_types:
                # atom_indices.append(idx)
                found_types.add(atom.type)
            else:
                if required_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
                    remove_atom_indices.extend(list(range(start_atom_index, idx)))
                else:  # proper format
                    new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx)), atoms=self._atoms,
                                                coords=self._coords, log=self._log))
                start_atom_index = idx
                found_types = {atom.type}  # atom_indices = [idx]
                current_residue_number = atom.residue_number

        # ensure last residue is added after stop iteration
        if required_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
            remove_atom_indices.extend(list(range(start_atom_index, idx + 1)))
        else:  # proper format. For each need to increment one higher than the last v
            new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx + 1)), atoms=self._atoms,
                                        coords=self._coords, log=self._log))

        self.residue_indices = list(range(len(new_residues)))
        self.residues = new_residues

        # remove bad atom_indices
        atom_indices = self.atom_indices
        for index in remove_atom_indices[::-1]:  # ensure popping happens in reverse
            atom_indices.pop(index)
        self.atom_indices = atom_indices

    # in an alt_location world, there may be some use to this. however above works great with current parsing
    # def create_residues(self):
    #     """For the Structure, create all possible Residue instances. Doesn't allow for alternative atom locations"""
    #     # required_types = {'N', 'CA', 'O'}  # 'C', Removing this for fragment library guide atoms...
    #     required_types = {'N', 'CA', 'C', 'O'}
    #     start_indices, residue_ranges = [], []
    #     remove_atom_indices = []
    #     remove_indices = []
    #     new_residues = []
    #     atom_indices, found_types = [], set()
    #     current_residue_number = self.atoms[0].residue_number
    #     start_atom_index, residue_idx = 0, 0
    #     for idx, atom in enumerate(self.atoms):
    #         # if the current residue number is the same as the prior number and the atom.type is not already present
    #         # We get rid of alternate conformations upon PDB load, so must be a new residue with bad numbering
    #         if atom.residue_number == current_residue_number and atom.type not in found_types:
    #             atom_indices.append(idx)
    #             found_types.add(atom.type)
    #         # if atom.residue_number == current_residue_number:  # current residue number is the same as the prior number
    #         #     if atom.type not in found_types:  # the atom.type is not already present
    #         #         # atom_indices.append(idx)
    #         #         found_types.add(atom.type)
    #         #     else:  # atom is already present. We got rid of alternate conformations upon PDB load, so new residue
    #         #         remove_indices.append(idx)
    #         else:  # we are starting a new residue
    #             if required_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
    #                 remove_atom_indices.append(list(range(start_atom_index, idx)))  # remove_indices
    #             else:  # proper format
    #                 start_indices.append(start_atom_index)
    #                 residue_ranges.append(len(found_types))
    #                 # only add those indices that are duplicates was used without alternative conformations
    #                 remove_atom_indices.append(remove_indices)  # <- empty list
    #             # remove_indices = []
    #             start_atom_index = idx
    #             found_types = {atom.type}  # atom_indices = [idx]
    #             current_residue_number = atom.residue_number
    #
    #     # ensure last residue is added after stop iteration
    #     if required_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
    #         remove_atom_indices.append(atom_indices)
    #     else:  # proper format
    #         start_indices.append(start_atom_index)
    #         residue_ranges.append(len(found_types))
    #         # only add those indices that are duplicates was used without alternative conformations
    #         remove_atom_indices.append(remove_indices)  # <- empty list
    #
    #     # remove bad atoms and correct atom_indices
    #     # atom_indices = self.atom_indices
    #     atoms = self.atoms
    #     for indices in remove_atom_indices[::-1]:  # ensure popping happens in reverse
    #         for index in indices[::-1]:  # ensure popping happens in reverse
    #             atoms.pop(index)  # , atom_indices.pop(index)
    #
    #     self.atom_indices = list(range(len(atoms)))  # atom_indices
    #     self.atoms = atoms
    #
    #     for start_index, residue_range in zip(start_indices, residue_ranges):
    #         new_residues.append(Residue(atom_indices=list(range(start_atom_index, start_atom_index + residue_range)),
    #                                     atoms=self._atoms, coords=self._coords, log=self._log))
    #     self.residue_indices = list(range(len(new_residues)))
    #     self.residues = new_residues

    def residue(self, residue_number, pdb=False):
        """Retrieve the Residue specified

        Returns:
            (Residue)
        """
        if pdb:
            number_source = 'number_pdb'
        else:
            number_source = 'number'

        for residue in self.residues:
            if getattr(residue, number_source) == residue_number:
                return residue
        return

    @property
    def n_terminal_residue(self):
        """Retrieve the Residue from the specified termini

        Returns:
            (Residue)
        """
        return self.residues[0]

    @property
    def c_terminal_residue(self):
        """Retrieve the Residue from the specified termini

        Returns:
            (Residue)
        """
        return self.residues[-1]

    @property
    def radius(self):
        """The largest point from the center of mass of the Structure

        Returns:
            (float)
        """
        return np.max(np.linalg.norm(self.coords - self.center_of_mass, axis=1))

    def get_residue_atoms(self, numbers=None, **kwargs):
        """Return the Atoms contained in the Residue objects matching a set of residue numbers

        Returns:
            (list[Atoms])
        """
        atoms = []
        for residue in self.get_residues(numbers=numbers, **kwargs):
            atoms.extend(residue.atoms)
        return atoms
        # return [residue.get_atoms() for residue in self.get_residues(numbers=residue_numbers)]

    def residue_from_pdb_numbering(self, residue_number):
        """Returns the Residue object from the Structure according to PDB residue number

        Returns:
            (Residue)
        """
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue
        return

    def residue_number_from_pdb(self, residue_number):
        """Returns the pose residue number from the queried .pdb number

        Returns:
            (int)
        """
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue.number
        return

    def residue_number_to_pdb(self, residue_number):
        """Returns the .pdb residue number from the queried pose number

        Returns:
            (int)
        """
        for residue in self.residues:
            if residue.number == residue_number:
                return residue.number_pdb
        return

    # def renumber_residues(self):
    #     """Starts numbering Residues at 1 and number sequentially until last Residue"""
    #     atoms = self.atoms
    #     last_atom_index = len(atoms)
    #     idx = 0  # offset , 1
    #     for i, residue in enumerate(self.residues, 1):
    #         # current_res_num = self.atoms[idx].residue_number
    #         # try:
    #         current_res_num = residue.number
    #         # except AttributeError:
    #         #     print('\n'.join(str(atom) for atom in residue.atoms))
    #         while atoms[idx].residue_number == current_res_num:
    #             atoms[idx].residue_number = i  # + offset
    #             idx += 1
    #             if idx == last_atom_index:
    #                 break
    #     # self.renumber_atoms()  # should be unnecessary

    def mutate_residue(self, residue: Residue = None, number: int = None, to: str = 'ALA', **kwargs) -> list[int]:
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Args:
            residue: A Residue object to mutate
            number: A Residue number to select the Residue of interest with
            to: The type of amino acid to mutate to
        Keyword Args:
            pdb=False (bool): Whether to pull the Residue by PDB number
        Returns:
            The indices of the Atoms being removed from the Structure
        """
        # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
        # if to.upper() in protein_letters_1to3:
        to = protein_letters_1to3.get(to.upper(), to).upper()

        if not residue:
            if not number:
                raise DesignError('Cannot mutate Residue without Residue object or number!')
            else:
                residue = self.residue(number, **kwargs)
        # for idx, atom in zip(residue.backbone_indices, residue.backbone_atoms):
        # for atom in residue.backbone_atoms:
        residue.type = to
        for atom in residue.atoms:
            atom.residue_type = to

        # Find the corresponding Residue Atom indices to delete (side-chain only)
        delete_indices = residue.sidechain_indices
        if not delete_indices:  # there are no indices
            return delete_indices
        # self.log.debug('Deleting indices from Residue: %s' % delete_indices)
        # self.log.debug('Indices in Residue: %s' % delete_indices)
        residue_delete_index = residue._atom_indices.index(delete_indices[0])
        for _ in iter(delete_indices):
            residue._atom_indices.pop(residue_delete_index)
        # must re-index all succeeding residues
        # This applies to all Residue objects, not only Structure Residue objects because modifying Residues object
        self._residues.reindex_residue_atoms()  # Todo start_at=residue.index)
        # self.log.debug('Deleting indices from Atoms: %s' % delete_indices)
        # self.log.debug('Range of indices in Atoms: %s' % self._atoms.atoms.shape[0])
        # self.log.debug('Last Residue atom_indices: %s' % self._residues.residues[-1].atom_indices)
        self._atoms.delete(delete_indices)
        self._coords.delete(delete_indices)
        # remove these indices from the Structure atom_indices (If other structures, must update their atom_indices!)
        # try:
        atom_delete_index = self._atom_indices.index(delete_indices[0])
        # except ValueError:
        #     print('Delete has %s:' % delete_indices)
        #     print('length of atom_indices %s:' % len(self._atom_indices))
        #     print('residue is %d%s, chain %s, pdb number %d:' % (residue.number, residue.type, residue.chain,
        #                                                          residue.number_pdb))
        #     print('structure._atom_indices has %s:' % self._atom_indices)
        #     exit()
        for _ in iter(delete_indices):
            self._atom_indices.pop(atom_delete_index)
        # must re-index all succeeding atoms
        # This doesn't apply to all PDB Atoms only Structure Atoms! Need to modify at PDB level
        self.reindex_atoms(start_at=atom_delete_index, offset=len(delete_indices))

        return delete_indices

    def insert_residue_type(self, residue_type: str, at: int = None, chain: str = None) -> Residue:
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed!

        Args:
            residue_type: Either the 1 or 3 letter amino acid code for the residue in question
            at: The pose numbered location which a new Residue should be inserted into the Structure
            chain: The chain identifier to associate the new Residue with
        Returns:
            The newly inserted Residue object
        """
        if not self.is_structure_owner:
            raise DesignError('This Structure "%s" is not the owner of it\'s attributes and therefore cannot handle '
                              'residue insertion!' % self.name)
        # Convert incoming aa to residue index so that AAReference can fetch the correct amino acid
        reference_index = protein_letters.find(protein_letters_3to1_extended.get(residue_type.title(),
                                                                                 residue_type.upper()))
        # Grab the reference atom coordinates and push into the atom list
        new_residue = copy(reference_aa.residue(reference_index))
        # new_residue = copy(Structure.reference_aa.residue(reference_index))
        new_residue.number = at
        residue_index = at - 1  # since at is one-indexed integer
        # insert the new_residue atoms and coords into the Structure Atoms
        # new_atoms = new_residue.atoms
        # new_coords = new_residue.coords
        self._atoms.insert(new_residue.atoms, at=new_residue.start_index)
        self._coords.insert(new_residue.coords, at=new_residue.start_index)
        # insert the new_residue into the Structure Residues
        self._residues.insert(new_residue, at=residue_index)  # take from pose numbering to zero-indexed
        self._residues.reindex_residue_atoms(start_at=residue_index)
        # self._atoms.insert(new_atoms, at=self._residues)
        new_residue._atoms = self._atoms
        new_residue.coords = self._coords
        # set this Structures new residue_indices. Must be the owner of all residues for this to work
        # self._residue_indices.insert(residue_index, residue_index)
        self.insert_indices(at=residue_index, new_indices=[residue_index], dtype='residue')
        # self.residue_indices = self.residue_indices.insert(residue_index, residue_index)
        # set this Structures new atom_indices. Must be the owner of all residues for this to work
        # for idx in reversed(range(new_residue.number_of_atoms)):
        #     self._atom_indices.insert(new_residue.start_index, idx + new_residue.start_index)
        self.insert_indices(at=new_residue.start_index, new_indices=new_residue.atom_indices, dtype='atom')
        # self.atom_indices = self.atom_indices.insert(new_residue.start_index, idx + new_residue.start_index)
        self.renumber_structure()

        # n_termini, c_termini = False, False
        prior_residue = self.residues[residue_index - 1]
        try:
            next_residue = self.residues[residue_index + 1]
        except IndexError:  # c_termini = True
            next_residue = None
        # if n_termini and not c_termini:
        # set the residues new chain_id, must occur after self.residue_indices update if chain isn't provided
        if chain:
            new_residue.chain = chain
        elif not next_residue:
            new_residue.chain = prior_residue.chain
        elif prior_residue.number > new_residue.number:  # we have a negative index, n_termini = True
            new_residue.chain = next_residue.chain
        elif prior_residue.chain == next_residue.chain:
            new_residue.chain = prior_residue.chain
        else:
            raise DesignError('Can\'t solve for the new Residue polymer association automatically! If the new '
                              'Residue is at a Structure termini in a multi-Structure Structure container, you must'
                              ' specify which Structure it belongs to by passing chain=')
        new_residue.number_pdb = prior_residue.number_pdb + 1
        # re-index the coords and residues map
        if self.secondary_structure:
            self.secondary_structure.insert('C', residue_index)  # ASSUME the insertion is disordered and coiled segment

        # remake residue indexing
        residues_atom_idx = [(residue, res_atom_idx) for residue in self.residues for res_atom_idx in residue.range]
        self.coords_indexed_residues, self.coords_indexed_residue_atoms = zip(*residues_atom_idx)
        range_idx = 0
        residue_indexed_ranges = []
        for residue in self.residues:
            residue_indexed_ranges.append(list(range(range_idx, range_idx + residue.number_of_atoms)))
            range_idx += residue.number_of_atoms
        self.residues_indexed_coords_indices = residue_indexed_ranges

        return new_residue

    def get_structure_sequence(self):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'

        Returns:
            (str): The amino acid sequence of the Structure Residues
        """
        return ''.join([protein_letters_3to1_extended.get(res.type.title(), '-') for res in self.residues])

    def translate(self, translation):
        """Perform a translation to the Structure ensuring only the Structure container of interest is translated
        ensuring the underlying coords are not modified"""
        translation_array = np.zeros(self._coords.coords.shape)
        translation_array[self.atom_indices] = np.array(translation)
        new_coords = self._coords.coords + translation_array
        self.replace_coords(new_coords)

    def rotate(self, rotation):
        """Perform a rotation to the Structure ensuring only the Structure container of interest is rotated ensuring the
        underlying coords are not modified"""
        rotation_array = np.tile(identity_matrix, (self._coords.coords.shape[0], 1, 1))
        rotation_array[self.atom_indices] = np.array(rotation)
        new_coords = np.matmul(self._coords.coords, rotation_array.swapaxes(-2, -1))  # essentially a transpose
        self.replace_coords(new_coords)

    def transform(self, rotation=None, translation=None, rotation2=None, translation2=None):
        """Perform a specific transformation to the Structure ensuring only the Structure container of interest is
        transformed ensuring the underlying coords are not modified"""
        if rotation is not None:  # required for np.ndarray or None checks
            rotation_array = np.tile(identity_matrix, (self._coords.coords.shape[0], 1, 1))
            rotation_array[self.atom_indices] = np.array(rotation)
            new_coords = np.matmul(self._coords.coords.reshape(-1, 1, 3), rotation_array.swapaxes(-2, -1))
        else:
            new_coords = self._coords.coords.reshape(-1, 1, 3)

        if translation is not None:  # required for np.ndarray or None checks
            translation_array = np.zeros(new_coords.shape)
            translation_array[self.atom_indices] = np.array(translation)
            new_coords += translation_array

        if rotation2 is not None:  # required for np.ndarray or None checks
            rotation_array2 = np.tile(identity_matrix, (self._coords.coords.shape[0], 1, 1))
            rotation_array2[self.atom_indices] = np.array(rotation2)
            new_coords = np.matmul(new_coords, rotation_array2.swapaxes(-2, -1))  # essentially transpose

        if translation2 is not None:  # required for np.ndarray or None checks
            translation_array2 = np.zeros(new_coords.shape)
            translation_array2[self.atom_indices] = np.array(translation2)
            new_coords += translation_array2
        self.replace_coords(new_coords.reshape(-1, 3))

    def return_transformed_copy(self, rotation=None, translation=None, rotation2=None, translation2=None):
        """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space

        Transformation proceeds by matrix multiplication with the order of operations as:
        rotation, translation, rotation2, translation2

        Keyword Args:
            rotation=None (numpy.ndarray): The first rotation to apply, expected general rotation matrix shape (3, 3)
            translation=None (numpy.ndarray): The first translation to apply, expected shape (3)
            rotation2=None (numpy.ndarray): The second rotation to apply, expected general rotation matrix shape (3, 3)
            translation2=None (numpy.ndarray): The second translation to apply, expected shape (3)
        Returns:
            (Structure): A transformed copy of the original object
        """
        if rotation is not None:  # required for np.ndarray or None checks
            # new_coords = np.matmul(self.coords, np.transpose(rotation))  # returns arrays with improper indices
            new_coords = np.matmul(self._coords.coords, np.transpose(rotation))
        else:
            # new_coords = self.coords  # returns arrays with improper indices
            new_coords = self._coords.coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(new_coords, np.transpose(rotation2))

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        # print(new_coords)
        new_structure = self.__copy__()
        # print('BEFORE', new_structure.coords)
        # this v should replace the actual numpy array located at coords after the _coords object has been copied
        new_structure.replace_coords(new_coords)
        # print('AFTER', new_structure.coords)
        # where as this v will set the _coords object to a new Coords object thus requiring all other _coords be updated
        # new_structure.set_coords(new_coords)
        return new_structure

    def replace_coords(self, new_coords):
        """Replace the current Coords array with a new Coords array"""
        try:
            new_coords.shape
            self._coords.coords = new_coords
        except AttributeError:
            raise ValueError('The coords passed to %s must be a numpy.ndarray' % self.replace_coords.__name__)

    def local_density(self, residue_numbers: list[int] = None, distance: float = 12.) -> list[float]:
        """Find the number of Atoms within a distance of each Atom in the Structure and add the density as an average
        value over each Residue

        Args:
            residue_numbers: The number of the Residues to include in the calculation
            distance: The cutoff distance with which Atoms should be included in local density
        Returns:
            An array like containing the local density around each Residue
        """
        if residue_numbers:
            coords = []
            residues = self.get_residues(numbers=residue_numbers)
            for residue in residues:
                coords.extend(residue.heavy_coords)
            coords_indexed_residues = [residue for residue in residues for _ in residue.heavy_atom_indices]
        else:
            heavy_atom_indices = self.heavy_atom_indices
            coords = self.coords[heavy_atom_indices]
            coords_indexed_residues = \
                [residue for idx, residue in enumerate(self.coords_indexed_residues) if idx in heavy_atom_indices]

        all_atom_tree = BallTree(coords)  # faster 131 msec/loop
        all_atom_counts_query = all_atom_tree.query_radius(coords, distance, count_only=True)
        # residue_neighbor_counts, current_residue = 0, coords_indexed_residues[0]
        current_residue = coords_indexed_residues[0]
        for residue, atom_neighbor_counts in zip(coords_indexed_residues, all_atom_counts_query):  # should be same len
            if residue == current_residue:
                current_residue.local_density += atom_neighbor_counts
            else:  # we have a new residue
                current_residue.local_density /= current_residue.number_of_heavy_atoms  # find the average
                current_residue = residue
                current_residue.local_density += atom_neighbor_counts
        # ensure the last residue is calculated
        current_residue.local_density /= current_residue.number_of_heavy_atoms  # find the average

        return [residue.local_density for residue in self.residues]

    def is_clash(self, distance: float = 2.1) -> bool:
        """Check if the Structure contains any self clashes. If clashes occur with the Backbone, return True. Reports
        the Residue where the clash occurred and the clashing Atoms

        Args:
            distance: The distance which clashes should be checked
        Returns:
            True if the Structure clashes, False if not
        """
        # heavy_atom_indices = self.heavy_atom_indices
        # all_atom_tree = BallTree(self.coords[heavy_atom_indices])  # faster 131 msec/loop
        # temp_coords_indexed_residues = self.coords_indexed_residues
        # coords_indexed_residues = [temp_coords_indexed_residues[idx] for idx in heavy_atom_indices]
        # temp_coords_indexed_residue_atoms = self.coords_indexed_residue_atoms
        # coords_indexed_residue_atoms = [temp_coords_indexed_residue_atoms[idx] for idx in heavy_atom_indices]
        all_atom_tree = BallTree(self.coords)  # faster 131 msec/loop
        coords_indexed_residues = self.coords_indexed_residues
        # atoms = self.atoms
        coords_indexed_residue_atoms = self.coords_indexed_residue_atoms
        residues_indexed_coords_indices = self.residues_indexed_coords_indices
        number_residues = self.number_of_residues
        residues = self.residues
        backbone_clashes, side_chain_clashes = [], []
        for prior_residue_idx, residue in enumerate(residues, -1):
            residue_query = all_atom_tree.query_radius(residue.backbone_and_cb_coords, distance)
            # reduce the dimensions and format as a single array
            all_contacts = set(np.concatenate(residue_query).ravel().tolist())
            # We must subtract the N and C atoms from the adjacent residues for each residue as these are within a bond
            # For the edge cases (N- & C-term), use other termini C & N atoms.
            # We might miss a clash here! It would be peculiar for the C-terminal C clashing with the N-terminus atoms
            # and vice-versa. This also allows a Structure with permuted sequence to be handled properly!
            prior_residue_range = residues_indexed_coords_indices[prior_residue_idx]
            prior_residue = residues[prior_residue_idx]
            residue_range = residues_indexed_coords_indices[-number_residues + prior_residue_idx + 1]
            next_residue_idx = -number_residues + prior_residue_idx + 2
            next_residue_range = residues_indexed_coords_indices[next_residue_idx]
            next_residue = residues[next_residue_idx]
            # using prior_residue and prior_residue_range
            residue_indices_and_bonded_c_and_n = \
                residue_range + [prior_residue_range[prior_residue.c_index], prior_residue_range[prior_residue.o_index],
                                 next_residue_range[next_residue.n_index]]
            # # using prior_residue atom_indices
            # residue_indices_and_bonded_c_and_n = \
            #     residue.atom_indices + [prior_residue.c_atom_index, prior_residue.o_atom_index,
            #                             residues[-number_residues + prior_residue_idx + 2].n_atom_index]
            # v is old way of specifying before refactoring to ^
            # residue_indices_and_bonded_c_and_n = \
            #     residue.atom_indices + [prior_residue.c_index, prior_residue.o_index,
            #                             residues[-number_residues + prior_residue_idx + 2].n_index]
            clashes = all_contacts.difference(residue_indices_and_bonded_c_and_n)
            if any(clashes):
                for clashing_atom_idx in clashes:
                    other_residue = coords_indexed_residues[clashing_atom_idx]
                    atom_idx = coords_indexed_residue_atoms[clashing_atom_idx]
                    # atom = atoms[clashing_atom_idx]
                    atom = other_residue[atom_idx]
                    if atom.is_backbone() or atom.is_cb():
                        backbone_clashes.append((residue, other_residue, atom_idx))
                    elif 'H' not in atom.type:
                        side_chain_clashes.append((residue, other_residue, atom_idx))

        if backbone_clashes:
            bb_info = '\n\t'.join('Residue %4d: %s' % (residue.number, str(other).split('\n')[atom_idx])
                                  for residue, other, atom_idx in backbone_clashes)
            self.log.critical('%s contains %d backbone clashes from the following Residues to the corresponding Atom:'
                              '\n\t%s' % (self.name, len(backbone_clashes), bb_info))
            if side_chain_clashes:
                self.log.warning('Additional side_chain clashes were identified but are being silenced by importance')
            return True
        else:
            if side_chain_clashes:
                sc_info = '\n\t'.join('Residue %5d: %s' % (residue.number, str(other).split('\n')[atom_idx])
                                      for residue, other, atom_idx in side_chain_clashes)
                self.log.warning('%s contains %d side-chain clashes from the following Residues to the corresponding '
                                 'Atom:\n\t%s' % (self.name, len(side_chain_clashes), sc_info))
            return False

    def get_sasa(self, probe_radius: float = 1.4, atom: bool = True):
        """Use FreeSASA to calculate the surface area of residues in the Structure object.

        Args:
            probe_radius: The radius which surface area should be generated
            atom: Whether the output should be generated for each atom. If False, will be generated for each Residue
        Sets:
            self.sasa, self.residue(s).sasa
        """
        if atom:
            out_format = 'pdb'
        # --format=pdb --depth=atom
        # REMARK 999 This PDB file was generated by FreeSASA 2.0.
        # REMARK 999 In the ATOM records temperature factors have been
        # REMARK 999 replaced by the SASA of the atom, and the occupancy
        # REMARK 999 by the radius used in the calculation.
        # MODEL        1                                        [radii][sasa]
        # ATOM   2557  C   PHE C 113      -2.627 -17.654  13.108  1.61  1.39
        # ATOM   2558  O   PHE C 113      -2.767 -18.772  13.648  1.42 39.95
        # ATOM   2559  CB  PHE C 113      -1.255 -16.970  11.143  1.88 13.46
        # ATOM   2560  CG  PHE C 113      -0.886 -17.270   9.721  1.61  1.98
        # ATOM   2563 CE1  PHE C 113      -0.041 -18.799   8.042  1.76 28.76
        # ATOM   2564 CE2  PHE C 113      -0.694 -16.569   7.413  1.76  2.92
        # ATOM   2565  CZ  PHE C 113      -0.196 -17.820   7.063  1.76  4.24
        # ATOM   2566 OXT  PHE C 113      -2.515 -16.590  13.750  1.46 15.09
        # ...
        # TER    7913      GLU A 264
        # ENDMDL EOF
        # if residue:
        else:
            out_format = 'seq'
        # --format=seq
        # Residues in ...
        # SEQ A    1 MET :   74.46
        # SEQ A    2 LYS :   96.30
        # SEQ A    3 VAL :    0.00
        # SEQ A    4 VAL :    0.00
        # SEQ A    5 VAL :    0.00
        # SEQ A    6 GLN :    0.00
        # SEQ A    7 ILE :    0.00
        # SEQ A    8 LYS :    0.87
        # SEQ A    9 ASP :    1.30
        # SEQ A   10 PHE :   64.55
        # ...
        # \n EOF
        if self.contains_hydrogen:
            include_hydrogen = ['--hydrogen']  # the addition of hydrogen skews results quite a bit
        else:
            include_hydrogen = []
        p = subprocess.Popen([free_sasa_exe_path, '--format=%s' % out_format, '--probe-radius', str(probe_radius),
                              '-c', free_sasa_configuration_path] + include_hydrogen,
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=self.return_atom_string().encode('utf-8'))
        # if err:  # usually results from Hydrogen atoms, silencing
        #     self.log.warning('\n%s' % err.decode('utf-8'))
        sasa_output = out.decode('utf-8').split('\n')
        if_idx = 0
        if atom:
            # slice removes first REMARK, MODEL and final TER, MODEL regardless of # of chains, TER inclusion
            # since return_atom_string doesn't have models, these won't be present and no option to freesasa about model
            # would be provided with above subprocess call
            # coords_indexed_residues = self.coords_indexed_residues
            # coords_indexed_residue_atoms = self.coords_indexed_residue_atoms
            atoms = self.atoms
            for line_split in map(str.split, sasa_output[5:-2]):  # slice could remove need for if ATOM
                if line_split[0] == 'ATOM':  # this seems necessary as MODEL can be added if MODEL is written
                    atoms[if_idx].sasa = float(line_split[-1])
                    if_idx += 1
        else:
            residues = self.residues
            for idx, line in enumerate(sasa_output[1:-1]):  # slice removes need for if == 'SEQ'
                if line[:3] == 'SEQ':  # doesn't seem to be the case that we can do this ^
                    residues[if_idx].sasa = float(line[16:])
                    if_idx += 1
        # Todo change to sasa property to call this automatically if AttributeError?
        self.sasa = sum([residue.sasa for residue in self.residues])

    def get_surface_residues(self, probe_radius: float = 2.2, sasa_thresh: float = 0.) -> list[int]:  # sasa_thresh=0.25
        """Get the residues who reside on the surface of the molecule

        Args:
            probe_radius: The radius which surface area should be generated
            sasa_thresh: The area threshold that the residue should have before it is considered "surface"
        Returns:
            The surface residue numbers
        """
        if not self.sasa:
            self.get_sasa(probe_radius=probe_radius, atom=False)  # , sasa_thresh=sasa_thresh)

        return [residue.number for residue in self.residues if residue.sasa > sasa_thresh]
        # Todo make dynamic based on relative threshold seen with Levy 2010
        # return [residue.number for residue in self.residues if residue.relative_sasa > sasa_thresh]

    # def get_residue_surface_area(self, residue_number, probe_radius=2.2):
    #     """Get the surface area for specified residues
    #
    #     Returns:
    #         (float): Angstrom^2 of surface area
    #     """
    #     if not self.sasa:
    #         self.get_sasa(probe_radius=probe_radius)
    #
    #     # return self.sasa[self.residues.index(residue_number)]
    #     return self.sasa[self.residues.index(residue_number)]

    def get_surface_area_residues(self, numbers, probe_radius=2.2):
        """Get the surface area for specified residues

        Returns:
            (float): Angstrom^2 of surface area
        """
        if not self.sasa:
            self.get_sasa(probe_radius=probe_radius)

        # return sum([sasa for residue_number, sasa in zip(self.sasa_residues, self.sasa) if residue_number in numbers])
        return sum([residue.sasa for residue in self.residues if residue.number in numbers])

    def errat(self, out_path: str | bytes = os.getcwd()) -> tuple[float, np.ndarray]:
        """Find the overall and per residue Errat accuracy for the given Structure

        Args:
            out_path: The path where Errat files should be written
        Returns:
            Overall Errat score, Errat value/residue array
        """
        # name = 'errat_input-%s-%d.pdb' % (self.name, random() * 100000)
        # current_struc_file = self.write(out_path=os.path.join(out_path, name))
        # errat_cmd = [errat_exe_path, os.path.splitext(name)[0], out_path]  # for writing file first
        # os.system('rm %s' % current_struc_file)
        out_path = out_path if out_path[-1] == os.sep else out_path + os.sep  # errat needs trailing "/"
        errat_cmd = [errat_exe_path, out_path]  # for passing atoms by stdin
        # p = subprocess.Popen(errat_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # out, err = p.communicate(input=self.return_atom_string().encode('utf-8'))
        # logger.info(self.return_atom_string()[:120])
        iteration = 1
        all_residue_scores = []
        while iteration < 5:
            p = subprocess.run(errat_cmd, input=self.return_atom_string(), encoding='utf-8', capture_output=True)
            all_residue_scores = p.stdout.strip().split('\n')
            if len(all_residue_scores) - 1 == self.number_of_residues:  # subtract overall_score from all_residue_scores
                break
            iteration += 1

        if iteration == 5:
            self.log.error('Errat couldn\'t generate the correct output length (%d) != number_of_residues (%d)'
                           % (len(all_residue_scores) - 1, self.number_of_residues))
        # errat_output_file = os.path.join(out_path, '%s.ps' % name)
        # errat_output_file = os.path.join(out_path, 'errat.ps')
        # else:
        # print(subprocess.list2cmdline(['grep', 'Overall quality factor**: ', errat_output_file]))
        # p = subprocess.Popen(['grep', 'Overall quality factor', errat_output_file],
        #                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # errat_out, errat_err = p.communicate()
        try:
            # overall_score = set(errat_out.decode().split('\n'))
            # all_residue_scores = list(map(str.strip, errat_out.split('\n'), 'Residue '))
            # all_residue_scores = errat_out.split('\n')
            overall_score = all_residue_scores.pop(-1)
            # print('all_residue_scores has %d records\n' % len(all_residue_scores), list(map(str.split, all_residue_scores)))
            return float(overall_score.split()[-1]), \
                np.array([float(score[-1]) for score in map(str.split, all_residue_scores)])
        except (IndexError, AttributeError):
            self.log.warning('%s: Failed to generate ERRAT measurement. Errat returned %s'
                             % (self.name, all_residue_scores))
            return 0., np.array([0. for _ in range(self.number_of_residues)])

    def stride(self, to_file: str | bytes = None):
        """Use Stride to calculate the secondary structure of a PDB.

        Args
            to_file: The location of a file to save the Stride output
        Sets:
            Residue.secondary_structure
        """
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

        # ASG    Detailed secondary structure assignment
        # Format:
        #  5-8  Residue type
        #  9-10 Protein chain identifier
        #  11-15 PDB residue number
        #  16-20 Ordinal residue number
        #  24-25 One letter secondary structure code **)
        #  26-39 Full secondary structure name
        #  42-49 Phi angle
        #  52-59 Psi angle
        #  61-69 Residue solvent accessible area
        #
        # -rId1Id2..  Read only Chains Id1, Id2 ...
        # -cId1Id2..  Process only Chains Id1, Id2 ...

        # The Stride based secondary structure names of each unique element where possible values are
        #  H:Alpha helix,
        #  G:3-10 helix,
        #  I:PI-helix,
        #  E:Extended conformation,
        #  B/b:Isolated bridge,
        #  T:Turn,
        #  C:Coil (none of the above)'
        current_struc_file = self.write(out_path=f'stride_input-{self.name}-{random() * 100000}.pdb')
        p = subprocess.Popen([stride_exe_path, current_struc_file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        os.system(f'rm {current_struc_file}')

        if out:
            if to_file:
                with open(to_file, 'wb') as f:
                    f.write(out)
            stride_output = out.decode('utf-8').split('\n')
        else:
            self.log.warning(f'{self.name}: No secondary structure assignment found with Stride')
            return
        # except:
        #     stride_out = None

        # if stride_out is not None:
        #     lines = stride_out.split('\n')
        residue_idx = 0
        residues = self.residues
        for line in stride_output:
            # residue_idx = int(line[10:15])
            if line[0:3] == 'ASG':
                # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
                # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
                # self.chain(line[9:10]).residue(int(line[10:15].strip())).secondary_structure = line[24:25]
                residues[residue_idx].secondary_structure = line[24:25]
                residue_idx += 1
        self.secondary_structure = ''.join(residue.secondary_structure for residue in residues)

    def parse_stride(self, stride_file: str | bytes, **kwargs):
        """From a Stride file, parse information for residue level secondary structure assignment

        Sets:
            self.secondary_structure
        """
        with open(stride_file, 'r') as f:
            stride_output = f.readlines()

        # residue_idx = 0
        # residues = self.residues
        for line in stride_output:
            # residue_idx = int(line[10:15])
            if line[0:3] == 'ASG':
                # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
                # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
                self.residue(int(line[10:15].strip()), pdb=True).secondary_structure = line[24:25]
                # residues[residue_idx].secondary_structure = line[24:25]
                # residue_idx += 1
        self.secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)

    def is_termini_helical(self, termini: str = 'n', window: int = 5) -> int:
        """Using assigned secondary structure, probe for a helical C-termini using a segment of 'window' residues

        Args:
            termini: Either n or c should be specified
            window: The segment size to search
        Returns:
            Whether the termini has a stretch of helical residues with length of the window (1) or not (0)
        """
        residues = list(reversed(self.residues)) if termini.lower() == 'c' else self.residues
        if not residues[0].secondary_structure:
            raise DesignError(f'You must call .get_secondary_structure on {self.name} before querying for helical termini')
        term_window = ''.join(residue.secondary_structure for residue in residues[:window * 2])
        if 'H' * window in term_window:
            return 1  # True
        else:
            return 0  # False

    def get_secondary_structure(self):
        if self.secondary_structure:
            return self.secondary_structure
        else:
            self.fill_secondary_structure()
            if self.secondary_structure:  # check if there is at least 1 secondary struc assignment
                return self.secondary_structure
            else:
                return

    def fill_secondary_structure(self, secondary_structure=None):
        if secondary_structure:
            self.secondary_structure = secondary_structure
            if len(self.secondary_structure) == self.number_of_residues:
                for idx, residue in enumerate(self.residues):
                    residue.secondary_structure = secondary_structure[idx]
            else:
                self.log.warning(f'The passed secondary_structure length ({len(self.secondary_structure)}) is not equal'
                                 f' to the number of residues ({self.number_of_residues}). Recalculating...')
                self.stride()  # we tried for efficiency, but its inaccurate, recalculate
        else:
            if self.residues[0].secondary_structure:
                self.secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)
            else:
                self.stride()

    def termini_proximity_from_reference(self, termini: str = 'n', reference: np.ndarray = None) -> float:
        """From an Entity, find the orientation of the termini from the origin (default) or from a reference point

        Args:
            termini: Either n or c should be specified
            reference: The reference where the point should be measured from
        Returns:
            The distance from the reference point to the furthest point
        """
        if termini.lower() == 'n':
            residue_coords = self.residues[0].n_coords
        elif termini.lower() == 'c':
            residue_coords = self.residues[-1].c_coords
        else:
            raise ValueError(f'Termini must be either "n" or "c", not "{termini}"!')

        if reference:
            coord_distance = np.linalg.norm(residue_coords - reference)
        else:
            coord_distance = np.linalg.norm(residue_coords)

        max_distance = self.distance_to_reference(reference=reference, measure='max')
        min_distance = self.distance_to_reference(reference=reference, measure='min')
        if abs(coord_distance - max_distance) < abs(coord_distance - min_distance):
            return 1  # termini is further from the reference
        else:
            return -1  # termini is closer to the reference

    # def furthest_point_from_reference(self, reference=None):
    #     """From an Structure, find the furthest coordinate from the origin (default) or from a reference.
    #
    #     Keyword Args:
    #         reference=None (numpy.ndarray): The reference where the point should be measured from. Default is origin
    #     Returns:
    #         (float): The distance from the reference point to the furthest point
    #     """
    #     if reference:
    #         return np.max(np.linalg.norm(self.coords - reference, axis=1))
    #     else:
    #         return np.max(np.linalg.norm(self.coords, axis=1))

    # def closest_point_to_reference(self, reference=None):  # todo combine with above into distance from reference
    #     """From an Structure, find the furthest coordinate from the origin (default) or from a reference.
    #
    #     Keyword Args:
    #         reference=None (numpy.ndarray): The reference where the point should be measured from. Default is origin
    #     Returns:
    #         (float): The distance from the reference point to the furthest point
    #     """
    #     if reference:
    #         return np.min(np.linalg.norm(self.coords - reference, axis=1))
    #     else:
    #         return np.min(np.linalg.norm(self.coords, axis=1))

    def distance_to_reference(self, reference=None, measure='mean'):  # todo combine with above into distance from reference
        """From an Structure, find the furthest coordinate from the origin (default) or from a reference.

        Keyword Args:
            reference=None (numpy.ndarray): The reference where the point should be measured from. Default is origin
            measure='mean' (str): The measurement to take with respect to the reference. Could be mean, min, max, or any
                numpy function to describe computed distance scalars
        Returns:
            (float): The distance from the reference point to the furthest point
        """
        if not reference:
            reference = origin

        return getattr(np, measure)(np.linalg.norm(self.coords - reference, axis=1))

    def return_atom_string(self, **kwargs):
        """Provide the Structure Atoms as a PDB file string"""
        # atom_atrings = '\n'.join(str(atom) for atom in self.atoms)
        # '%d, %d, %d' % tuple(element.tolist())
        # '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'
        # atom_atrings = '\n'.join(str(atom) % '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple(coord))
        return '\n'.join(residue.__str__(**kwargs) for residue in self.residues)
        # return '\n'.join(atom.__str__(**kwargs) % '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple(coord))
        #                  for atom, coord in zip(self.atoms, self.coords.tolist()))

    def format_header(self, **kwargs):
        if type(self).__name__ in ['Entity', 'PDB']:
            return self.format_biomt(**kwargs) + self.format_seqres(**kwargs)
        # elif type(self).__name__ in ['Entity', 'Chain']:
        #     return self.format_biomt() + self.format_seqres()
        else:
            return ''

    def format_biomt(self, **kwargs):
        """Return the BIOMT record for the PDB if there was one parsed

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

    def write_header(self, file_handle, header=None, **kwargs) -> None:
        """Handle writing of Structure header information to the file

        Args:
            file_handle (FileObject): An open file object where the header should be written
        Keyword Args
            header (None | str): A string that is desired at the top of the .pdb file
            **kwargs:
        Returns:
            (None)
        """
        _header = self.format_header(**kwargs)  # biomt and seqres
        if header and isinstance(header, Iterable):
            if isinstance(header, str):  # used for cryst_record now...
                _header += (header if header[-2:] == '\n' else '%s\n' % header)
            # else:  # TODO
            #     location.write('\n'.join(header))
        if _header != '':
            file_handle.write('%s' % _header)

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, **kwargs) -> str | None:
        #     header: str = None, increment_chains: bool = False,
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
        Returns:
            The name of the written file if out_path is used
        """
        if file_handle:
            file_handle.write('%s\n' % self.return_atom_string(**kwargs))
            return

        if out_path:
            with open(out_path, 'w') as outfile:
                self.write_header(outfile, **kwargs)
                outfile.write('%s\n' % self.return_atom_string(**kwargs))

            return out_path

    def get_fragments(self, residues: list[Residue] = None, residue_numbers: list[int] = None, fragment_length: int = 5,
                      **kwargs) -> list[MonoFragment]:
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Args:
            residues: The specific Residues to search for
            residue_numbers: The specific residue numbers to search for
            fragment_length: The length of the fragment observations used
        Keyword Args:
            representatives=None (dict[int, PDB]):
        Returns:
            The MonoFragments found on the Structure
        """
        if not residues and not residue_numbers:
            return []

        # residues = self.residues
        # ca_stretches = [[residues[idx + i].ca for i in range(-2, 3)] for idx, residue in enumerate(residues)]
        # compare ca_stretches versus monofrag ca_stretches
        # monofrag_array = repeat([ca_stretch_frag_index1, ca_stretch_frag_index2, ...]
        # monofrag_indices = filter_euler_lookup_by_zvalue(ca_stretches, monofrag_array, z_value_func=fragment_overlap,
        #                                                  max_z_value=rmsd_threshold)
        fragment_lower_range, fragment_upper_range = parameterize_frag_length(fragment_length)
        fragments = []
        for residue_number in residue_numbers:
            # frag_residue_numbers = [residue_number + i for i in range(fragment_lower_range, fragment_upper_range)]
            ca_count = 0
            frag_residues = self.get_residues(numbers=[residue_number + i for i in range(fragment_lower_range,
                                                                                         fragment_upper_range)])
            for residue in frag_residues:
                if residue.ca:
                    ca_count += 1

            if ca_count == 5:
                fragment = MonoFragment(residues=frag_residues, fragment_length=fragment_length, **kwargs)
                if fragment.i_type:
                    fragments.append(fragment)
                # fragments.append(Structure.from_residues(frag_residues, coords=self._coords, log=None))
                # fragments.append(Structure.from_residues(deepcopy(frag_residues), log=None))

        # for structure in fragments:
        #     structure.chain_ids = [structure.residues[0].chain]

        return fragments

    @property
    def contact_order(self) -> np.ndarray:
        """Return the contact order on a per Residue basis

        Returns:
            The array representing the contact order for each residue in the Structure
        """
        try:
            return self._contact_order
        except AttributeError:
            self._contact_order = self.contact_order_per_residue()
            return self._contact_order

    @contact_order.setter
    def contact_order(self, contact_order: Sequence):
        """Set the contact order for each Residue

        Args:
            contact_order: A zero-indexed per residue measure of the contact order
        """
        for idx, residue in enumerate(self.residues):
            residue.contact_order = contact_order[idx]

    def contact_order_per_residue(self, sequence_distance_cutoff: float = 2., distance: float = 6.) -> np.ndarray:
        """Calculate the contact order on a per residue basis

        Args:
            sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
            distance: The distance in angstroms to measure atomic contact distances in contact
        Returns:
            The array representing the contact order for each residue in the Structure
        """
        # distance of 6 angstroms between heavy atoms was used for 1998 contact order work,
        # subsequent residue wise contact order has focused on the Cb Cb heuristic of 12 A
        # I think that an atom-atom based measure is more accurate, if slightly more time
        # The BallTree creation is the biggest time cost regardless

        # Get CB Atom Coordinates including CA coordinates for Gly residues
        tree = BallTree(self.coords)  # [self.heavy_atom_indices])  # Todo
        # entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want
        query = tree.query_radius(self.coords, distance)  # get v residue w/ [0]
        coords_indexed_residues = self.coords_indexed_residues
        contacting_pairs = set((coords_indexed_residues[idx1], coords_indexed_residues[idx2])
                               for idx2, contacts in enumerate(query) for idx1 in contacts)
        for residue1, residue2 in contacting_pairs:
            residue_distance = abs(residue1.number - residue2.number)
            if residue_distance >= sequence_distance_cutoff:
                residue1.contact_order += residue_distance

        number_residues = self.number_of_residues
        for residue in self.residues:
            residue.contact_order /= number_residues

        return np.array([residue.contact_order for residue in self.residues])

    def format_resfile_from_directives(self, residue_directives, include=None, background=None, **kwargs):
        """Format Residue mutational potentials given Residues/residue numbers and corresponding mutation directive.
        Optionally, include specific amino acids and limit to a specific background. Both dictionaries accessed by same
        keys as residue_directives

        Args:
            residue_directives (dict[mapping[Residue | int],str]): {Residue object: 'mutational_directive', ...}
        Keyword Args:
            include=None (dict[mapping[Residue | int],set[str]]):
                Include a set of specific amino acids for each residue
            background=None (dict[mapping[Residue | int],set[str]]):
                The background amino acids to compare possibilities against
            special=False (bool): Whether to include special residues
        Returns:
            (list[str]): Formatted resfile lines for each Residue with a PIKAA and amino acid type string
        """
        if not background:
            background = {}
        if not include:
            include = {}

        res_file_lines = []
        if isinstance(next(iter(residue_directives)), int):  # this isn't a residue object, instead residue numbers
            for residue_number, directive in residue_directives.items():
                residue = self.residue(residue_number)
                allowed_aas = residue. \
                    mutation_possibilities_from_directive(directive, background=background.get(residue_number),
                                                          **kwargs)
                allowed_aas = {protein_letters_3to1_extended[aa.title()] for aa in allowed_aas}
                allowed_aas = allowed_aas.union(include.get(residue_number, {}))
                res_file_lines.append('%d %s PIKAA %s' % (residue.number, residue.chain, ''.join(sorted(allowed_aas))))
                # res_file_lines.append('%d %s %s' % (residue.number, residue.chain,
                #                                     'PIKAA %s' % ''.join(sorted(allowed_aas)) if len(allowed_aas) > 1
                #                                     else 'NATAA'))
        else:
            for residue, directive in residue_directives.items():
                allowed_aas = residue.\
                    mutation_possibilities_from_directive(directive, background=background.get(residue), **kwargs)
                allowed_aas = {protein_letters_3to1_extended[aa.title()] for aa in allowed_aas}
                allowed_aas = allowed_aas.union(include.get(residue, {}))
                res_file_lines.append('%d %s PIKAA %s' % (residue.number, residue.chain, ''.join(sorted(allowed_aas))))
                # res_file_lines.append('%d %s %s' % (residue.number, residue.chain,
                #                                     'PIKAA %s' % ''.join(sorted(allowed_aas)) if len(allowed_aas) > 1
                #                                     else 'NATAA'))

        return res_file_lines

    def make_resfile(self, residue_directives, out_path=os.getcwd(), header=None, **kwargs):
        """Format a resfile for the Rosetta Packer from Residue mutational directives

        Args:
            residue_directives (dict[Residue | int, str]): {Residue/int: 'mutational_directive', ...}
        Keyword Args:
            out_path=os.getcwd() (str): Directory to write the file
            header=None (list[str]): A header to constrain all Residues for packing
            include=None (dict[Residue | int, set[str]]):
                Include a set of specific amino acids for each residue
            background=None (dict[Residue | int, set[str]]):
                The background amino acids to compare possibilities against
            special=False (bool): Whether to include special residues
        Returns:
            (str): The path to the resfile
        """
        residue_lines = self.format_resfile_from_directives(residue_directives, **kwargs)
        res_file = os.path.join(out_path, '%s.resfile' % self.name)
        with open(res_file, 'w') as f:
            # format the header
            f.write('%s\n' % ('\n'.join(header + ['start']) if header else 'start'))
            # start the body
            f.write('%s\n' % '\n'.join(residue_lines))

        return res_file

    # def read_secondary_structure(self, filename=None, source='stride'):
    #     if source == 'stride':
    #         secondary_structure = self.parse_stride(filename)
    #     elif source == 'dssp':
    #         secondary_structure = None
    #     else:
    #         raise DesignError('Must pass a source to %s' % Structure.read_secondary_structure.__name__)
    #
    #     return secondary_structure
    def set_b_factor_data(self, dtype=None):
        """Set the b-factor entry for every Residue to a Residue attribute

        Keyword Args:
            dtype=None (str): The attribute of interest
        """
        # kwargs = dict(b_factor=dtype)
        # self._residues.set_attributes(b_factor=dtype)  # , **kwargs)
        self.set_residues_attributes(b_factor=dtype)  # , **kwargs)

    def copy_structures(self):
        """Copy all member Structures that reside in Structure containers"""
        for structure_type in self.structure_containers:
            structures = getattr(self, structure_type)
            for idx, structure in enumerate(structures):
                structures[idx] = copy(structure)

    @staticmethod
    def return_chain_generator() -> Generator[str, None, None]:
        """Provide a generator which produces all combinations of chain strings useful in producing viable Chain objects

        Returns
            The generator producing a 2 character string
        """
        return (first + second for modification in ['upper', 'lower']
                for first in [''] + list(getattr(Structure.available_letters, modification)())
                for second in list(getattr(Structure.available_letters, 'upper')()) +
                list(getattr(Structure.available_letters, 'lower')()))

    def __key(self) -> tuple[str, int, ...]:
        return self.name, *self._residue_indices

    def __copy__(self) -> Structure:
        other = self.__class__.__new__(self.__class__)
        other.__dict__ = self.__dict__.copy()
        # other.__dict__ = {}  # Todo
        # for attr, value in self.__dict__.items():  # Todo
        for attr, value in other.__dict__.items():
            other.__dict__[attr] = copy(value)
        # other._residues.set_attributes(_coords=other._coords)  # , _log=other._log)  # , _atoms=other._atoms)
        other.set_residues_attributes(_coords=other._coords)  # , _log=other._log)  # , _atoms=other._atoms)

        return other

    def __eq__(self, other):
        if isinstance(other, Structure):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.name


class Structures(Structure, UserList):
    # todo
    #  The inheritance of both a Structure and UserClass may get sticky...
    #  As all the functions I have here overwrite Structure class functions, the inheritance may not be necessary
    """Keep track of groups of Structure objects"""
    def __init__(self, structures: list = None, **kwargs):  # log=None,
        super().__init__(**kwargs)
        # print('Initializing Structures')
        # super().__init__()  # without passing **kwargs, there is no need to ensure base Object class is protected
        # if log:
        #     self.log = log
        # elif log is None:
        #     self.log = null_log
        # else:  # When log is explicitly passed as False, use the module logger
        #     self.log = logger

        if isinstance(structures, list):
            if all([True if isinstance(structure, Structure) else False for structure in structures]):
                # self.structures = structures
                self.data = structures
        #     else:
        #         # self.structures = []
        #         self.data = []
        # else:
        #     # self.structures = []
        #     self.data = []

    # @classmethod
    # def from_file(cls, file, **kwargs):
    #     """Construct Models from multimodel PDB file using the PDB.chains
    #     Ex: [Chain1, Chain1, ...]
    #     """
    #     pdb = PDB.from_file(file, **kwargs)  # Todo make independent parsing function
    #     # new_model = cls(models=pdb.chains)
    #     return cls(structures=pdb.chains, **kwargs)

    @property
    def structures(self):
        return self.data

    @property
    def name(self) -> str:
        try:
            return self._name
        except AttributeError:
            self._name = Structures.__name__
            return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def number_of_structures(self):
        # return len(self.structures)
        return len(self.data)

    @property
    def coords(self) -> np.ndarray:
        """Return a view of the Coords from the Structures"""
        try:
            return self._coords.coords
        except AttributeError:
            # coords = [structure.coords for structure in self.structures]
            coords = [structure.coords for structure in self.data]
            # coords = []
            # for structure in self.structures:
            #     coords.extend(structure.coords)
            self._coords = Coords(np.concatenate(coords))

            return self._coords.coords

    @property
    def atoms(self):
        """Return a view of the Atoms from the Structures"""
        try:
            return self._atoms.atoms.tolist()
        except AttributeError:
            atoms = []
            # for structure in self.structures:
            for structure in self.data:
                atoms.extend(structure.atoms)
            self._atoms = Atoms(atoms)
            return self._atoms.atoms.tolist()

    @property
    def number_of_atoms(self):
        return len(self.atoms)
        # return len(self.coords)

    @property
    def residues(self):  # TODO Residues iteration
        try:
            return self._residues.residues.tolist()
        except AttributeError:
            residues = []
            for structure in self.data:
                residues.extend(structure.residues)
            self._residues = Residues(residues)
            return self._residues.residues.tolist()

    @property
    def number_of_residues(self):
        return len(self.residues)

    @property
    def coords_indexed_residues(self):
        try:
            return self._coords_indexed_residues
        except AttributeError:
            self._coords_indexed_residues = [residue for residue in self.residues for _ in residue.range]
            return self._coords_indexed_residues

    @property
    def coords_indexed_residue_atoms(self):
        try:
            return self._coords_indexed_residue_atoms
        except AttributeError:
            self._coords_indexed_residue_atoms = \
                [res_atom_idx for residue in self.residues for res_atom_idx in residue.range]
            return self._coords_indexed_residue_atoms

    @property
    def residues_indexed_coords_indices(self):
        """Returns a map of the Residue atom/coord_indices, indexed to the coords indices for each Residue in the
        Structure

        Returns:
            (list[list[int]]): Indexed by the Residue position in the corresponding .coords attribute
        """
        try:
            return self._residues_indexed_coords_indices
        except AttributeError:
            range_idx = 0
            self._residues_indexed_coords_indices = []
            for residue in self.residues:
                self._residues_indexed_coords_indices.append(list(range(range_idx, range_idx + residue.number_of_atoms)))
                range_idx += residue.number_of_atoms
            return self._residues_indexed_coords_indices

    # @property
    # def model_coords(self):  # TODO RECONCILE with coords, SymmetricModel, and State variation
    #     """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
    #     return self._model_coords.coords
    #
    # @model_coords.setter
    # def model_coords(self, coords):
    #     if isinstance(coords, Coords):
    #         self._model_coords = coords
    #     else:
    #         raise AttributeError(
    #             'The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    #             'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    @property
    def backbone_indices(self):
        # Todo these below are incorrect because the coords indexed indices are each 0-N ... reindex all to coords
        try:
            return self._backbone_indices
        except AttributeError:
            test_indices = self.backbone_indices
            self._coords_indexed_backbone_indices = \
                [idx for idx, atom_idx in enumerate(self.atom_indices) if atom_idx in test_indices]

            self._backbone_indices = []
            for structure in self.data:
                self._backbone_indices.extend(structure.coords_indexed_backbone_indices)
            return self._backbone_indices

    @property
    def backbone_and_cb_indices(self):
        try:
            return self._backbone_and_cb_indices
        except AttributeError:
            self._backbone_and_cb_indices = []
            for structure in self.data:
                self._backbone_and_cb_indices.extend(structure.coords_indexed_backbone_and_cb_indices)
            return self._backbone_and_cb_indices

    @property
    def cb_indices(self):
        try:
            return self._cb_indices
        except AttributeError:
            self._cb_indices = []
            for structure in self.data:
                self._cb_indices.extend(structure.coords_indexed_cb_indices)
            return self._cb_indices

    @property
    def ca_indices(self):
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = []
            for structure in self.structures:
                self._ca_indices.extend(structure.coords_indexed_ca_indices)
            return self._ca_indices

    # @property
    # def center_of_mass(self):
    #     """Returns: (numpy.ndarray)"""
    #     structure_length = self.number_of_atoms
    #     return np.matmul(np.full(structure_length, 1 / structure_length), self.coords)

    def translate(self, tx):
        for structure in self.data:
            structure.translate(tx)

    def rotate(self, rotation):
        for structure in self.data:
            structure.rotate(rotation)

    def transform(self, **kwargs):  # rotation=None, translation=None):
        for structure in self.data:
            structure.transform(**kwargs)

    # def replace_coords(self, new_coords):
    #     """Replace the current Coords array with a new Coords array"""
    #     self._coords.coords = new_coords
    #     total_atoms = 0
    #     for structure in self.data:
    #         new_atoms = total_atoms + structure.number_of_atoms
    #         self._coords.coords[total_atoms: new_atoms] = structure.coords
    #         total_atoms = new_atoms

    # @classmethod
    # def return_transformed_copy(cls, **kwargs):  # rotation=None, translation=None, rotation2=None, translation2=None):
    #     # return Structures(structures=[structure.return_transformed_copy(**kwargs) for structure in self.structures])
    #     return cls([structure.return_transformed_copy(**kwargs) for structure in self.structures])
    #     # return cls(structures=[structure.return_transformed_copy(**kwargs) for structure in self.structures])
    #
    def return_transformed_copy(self, **kwargs):  # rotation=None, translation=None, rotation2=None, translation2=None):
        # print('Structures type %s' % self.__class__)
        new_structures = self.__new__(self.__class__)
        # print('Transformed Structure type (__new__) %s' % type(new_structures))
        # print('self.__dict__ is %s' % self.__dict__)
        new_structures.__init__([structure.return_transformed_copy(**kwargs) for structure in self.data],
                                log=self.log)  # self.__dict__
        # print('Transformed Structures, structures %s' % [structure for structure in new_structures.structures])
        # print('Transformed Structures, models %s' % [structure for structure in new_structures.models])
        return new_structures
        # return Structures(structures=[structure.return_transformed_copy(**kwargs) for structure in self.structures])

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, increment_chains: bool = True,
              header: str = None, **kwargs) -> str | None:
        """Write Structures to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            increment_chains: Whether to write each Structure with a new chain name, otherwise write as a new Model
            header: If there is header information that should be included. Pass new lines with a "\n"
        Returns:
            The name of the written file if out_path is used
        """
        if file_handle:  # Todo increment_chains compatibility
            file_handle.write('%s\n' % self.return_atom_string(**kwargs))
            return

        with open(out_path, 'w') as f:
            if header:
                if isinstance(header, str):
                    f.write(header)
                # if isinstance(header, Iterable):

            if increment_chains:
                available_chain_ids = self.return_chain_generator()
                for structure in self.structures:
                    chain = next(available_chain_ids)
                    structure.write(file_handle=f, chain=chain)
                    c_term_residue = structure.c_terminal_residue
                    f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
                                                                          c_term_residue.type, chain,
                                                                          c_term_residue.number))
            else:
                for model_number, structure in enumerate(self.structures, 1):
                    f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                    structure.write(file_handle=f)
                    c_term_residue = structure.c_terminal_residue
                    f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
                                                                          c_term_residue.type, structure.chain_id,
                                                                          c_term_residue.number))
                    f.write('ENDMDL\n')

        return out_path

    def __repr__(self):
        return '<Structure.Structures object at %s>' % id(self)

    # def __str__(self):
    #     return self.name

    # def __len__(self) -> int:
    #     return len(self.structures)

    # def __iter__(self):
    #     yield from iter(self.structures)

    # def __getitem__(self, idx):
    #     return self.structures[idx]


class Chain(Structure):
    def __init__(self, **kwargs):  # name=None, residues=None,  residue_indices=None, coords=None, log=None
        super().__init__(**kwargs)  # name=name, residues=residues, residue_indices=residue_indices, coords=coords,
        # log=log
        # self.chain_id = self.name

    @property
    def chain_id(self):
        return self.name

    @chain_id.setter
    def chain_id(self, chain_id):
        self.name = chain_id
        # self._residues.set_attributes(chain=chain_id)
        self.set_residues_attributes(chain=chain_id)


class Entity(Chain, SequenceProfile):
    """Entity
    Initialize with Keyword Args:
        chains=None (list): A list of all Chain objects that match the Entity
        uniprot_id=None (str): The unique UniProtID for the Entity
        sequence=None (str): The sequence for the Entity
        name=None (str): The name for the Entity. Typically, PDB.name is used to make a PDB compatible form
        PDB EntryID_EntityID
    """
    def __init__(self, chains=None, uniprot_id=None, **kwargs):
        """When init occurs chain_ids are set if chains were passed. If not, then they are auto generated"""
        self.api_entry = None  # {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
        self.dihedral_chain = None
        self.is_oligomeric = False
        self.max_symmetry = None
        self.rotation_d = {}
        self.symmetry = None
        # Todo choose most symmetrically average if Entity is symmetric
        representative = chains[0]
        super().__init__(residues=representative._residues, residue_indices=representative.residue_indices,
                         coords=representative._coords, **kwargs)
        self._chains = []
        chain_ids = [representative.name]
        self.chain_transforms.append(dict(rotation=identity_matrix, translation=origin))
        if len(chains) > 1:
            self.is_oligomeric = True  # inherent in Entity type is a single sequence. Therefore, must be oligomeric
            for idx, chain in enumerate(chains[1:]):
                if chain.number_of_residues == self.number_of_residues and chain.sequence == self.sequence:
                    # do an apples to apples comparison
                    # length alone is inaccurate if chain is missing first residue and self is missing it's last...
                    _, rot, tx, _ = superposition3d(chain.get_cb_coords(), self.get_cb_coords())
                else:  # do an alignment, get selective indices, then follow with superposition
                    self.log.warning('Chain %s passed to Entity %s doesn\'t have the same number of residues'
                                     % (chain.name, self.name))
                    mutations = generate_mutations(self.sequence, chain.sequence, blanks=True, return_all=True)
                    # get only those indices where there is an aligned aa on the opposite chain
                    fixed_polymer_indices, moving_polymer_indices = [], []
                    to_idx, from_idx = 0, 0
                    # from is moving, to is fixed
                    for mutation in mutations.values():
                        if mutation['from'] == '-':  # increment to_idx/fixed_idx
                            to_idx += 1
                        elif mutation['to'] == '-':  # increment from_idx/moving_idx
                            from_idx += 1
                        else:
                            fixed_polymer_indices.append(to_idx)
                            to_idx += 1
                            moving_polymer_indices.append(from_idx)
                            from_idx += 1
                    # mov_indices_str, from_str, to_str, fix_indices_str = '', '', '', ''
                    # moving_polymer_indices_iter, fixed_polymer_indices_iter = iter(moving_polymer_indices), iter(fixed_polymer_indices)
                    # for idx, mutation in enumerate(mutations.values()):
                    #     try:
                    #         mov_indices_str += ('%2d' % next(moving_polymer_indices_iter) if mutation['from'] != '-' and mutation['to'] != '-' else ' -')
                    #     except StopIteration:
                    #         mov_indices_str += '-'
                    #     try:
                    #         fix_indices_str += ('%2d' % next(fixed_polymer_indices_iter) if mutation['from'] != '-' and mutation['to'] != '-' else ' -')
                    #     except StopIteration:
                    #         fix_indices_str += '-'
                    # print(mov_indices_str)
                    # # print(from_str)
                    # print(' '.join(mutation['from'] for mutation in mutations.values()))  # from_str
                    # # print(to_str)
                    # print(' '.join(mutation['to'] for mutation in mutations.values()))  # to_str
                    # print(fix_indices_str)
                    _, rot, tx, _ = superposition3d(chain.get_cb_coords()[fixed_polymer_indices],
                                                    self.get_cb_coords()[moving_polymer_indices])
                self.chain_transforms.append(dict(rotation=rot, translation=tx))
                chain_ids.append(chain.name)
            self.number_of_monomers = len(chains)
        self.chain_ids = chain_ids
        self.prior_ca_coords = self.get_ca_coords()
        # else:  # elif len(chains) == 1:
        #     self.chain_transforms.append(dict(rotation=identity_matrix, translation=origin))
        # else:
        #     self.chain_ids = [self.chain_id]
        #     self.chain_transforms.append(dict(rotation=identity_matrix, translation=origin))
        # self._uniprot_id = None
        self.uniprot_id = uniprot_id

    @classmethod
    def from_chains(cls, chains=None, uniprot_id=None, **kwargs):
        # if isinstance(representative, Structure):
        return cls(chains=chains, uniprot_id=uniprot_id, **kwargs)
        # else:
        #     raise DesignError('When initializing an Entity, you must pass a representative Structure object. This is '
        #                       'typically a Chain, but could be another collection of residues in a Structure object')

    @Structure.coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            #                         and setter is not happening because of a copy (no new info, update unimportant)
            if self.is_oligomeric:  # and not (coords.coords == self._coords.coords).all():
                # each mate chain coords are dependent on the representative (captain) coords, find the transform
                chains = self.chains
                self.chain_transforms.clear()
                for chain in chains:
                    # rmsd, rot, tx, _ = superposition3d(coords.coords[self.cb_indices], self.get_cb_coords())
                    _, rot, tx, _ = superposition3d(coords.coords[self.cb_indices], chain.get_cb_coords())
                    self.chain_transforms.append(dict(rotation=rot, translation=tx))
                self._chains.clear()
                # then apply to mates
                # for chain in self.chains:
                #     # rotate then translate the Entity coords and pass to mate chains
                #     new_coords = np.matmul(chain.coords, np.transpose(rot)) + tx
                #     chain.coords = new_coords
                # finally set the Entity coords
                self._coords = coords
                # self.update_attributes(coords=self._coords)
            else:  # chains will be modified by another Structure owner/accept the new copy coords
                self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute _coords')

    @property
    def uniprot_id(self) -> str | None:
        """The UniProt ID for the Entity used for accessing genomic and homology features"""
        try:
            return self._uniprot_id
        except AttributeError:
            self.api_entry = get_pdb_info_by_entity(self.name)  # {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
            for chain, api_data in self.api_entry.items():  # [next(iter(self.api_entry))]
                # print('Retrieving UNP ID for %s\nAPI DATA for chain %s:\n%s' % (self.name, chain, api_data))
                if api_data.get('db') == 'UNP':
                    # set the first found chain. They are likely all the same anyway
                    self._uniprot_id = api_data.get('accession')
            try:
                return self._uniprot_id
            except AttributeError:
                pass
        self.log.warning('No uniprot_id found for Entity %s' % self.name)
        return
        # # Make a random pseudo UniProtID
        # self.uniprot_id = '%s%d' % ('R', randint(10000, 99999))

    @uniprot_id.setter
    def uniprot_id(self, uniprot_id):
        if uniprot_id:
            self._uniprot_id = uniprot_id

    @property
    def chain_id(self) -> str:
        """The Chain name for the Entity instance"""
        return self.residues[0].chain

    @chain_id.setter
    def chain_id(self, chain_id):
        # self._residues.set_attributes(chain=chain_id)
        self.set_residues_attributes(chain=chain_id)
        try:
            self._chain_ids[0] = chain_id
        except AttributeError:
            # if _chain_ids is an attribute, then it will be length 1. If not set, will be set accordingly later
            pass

    @property
    def number_of_monomers(self) -> int:
        """The number of copies of the Entity in the Oligomer"""
        try:
            return self._number_of_monomers
        except AttributeError:  # set based on the symmetry, unless that fails then find using chain_ids
            self._number_of_monomers = valid_subunit_number.get(self.symmetry, len(self._chain_ids))
            return self._number_of_monomers

    @number_of_monomers.setter
    def number_of_monomers(self, value):
        self._number_of_monomers = value

    @property
    def chain_ids(self) -> list:  # Also used in PDB
        """The names of each Chain found in the Entity"""
        try:
            return self._chain_ids
        except AttributeError:  # This shouldn't be possible with the constructor available
            chain_gen = self.return_chain_generator()
            chain_id = self.chain_id
            self._chain_ids = [chain_id]
            for _ in range(self.number_of_monomers):
                next_chain = next(chain_gen)
                if next_chain != chain_id:
                    self._chain_ids.append(next_chain)
            return self._chain_ids

    @chain_ids.setter
    def chain_ids(self, chain_ids):
        self._chain_ids = chain_ids

    @property
    def chain_transforms(self) -> list[dict]:
        """The specific transformation operators to generate all mate chains of the Oligomer

        Returns:
            (list[dict])
        """
        # self.log.info('%s chain_transform %s' % (self.name, 'start'))
        try:
            return self._chain_transforms
        except AttributeError:
            try:  # this section is only useful if the current instance is an Entity copy
                # self.log.info('%s chain_transform %s' % (self.name, 'AttributeError'))
                self._chain_transforms = []
                # if self.prior_ca_coords is not None:
                #     self.log.info('prior_ca_coords has not been set but it is not None')
                #     getattr(self, 'prior_ca_coords')  # try to get exception raised here?
                # missing_at = 'prior_ca_coords'
                # self.prior_ca_coords
                self.__chain_transforms
                if self.is_oligomeric:  # True if multiple chains
                    current_ca_coords = self.get_ca_coords()
                    _, new_rot, new_tx, _ = superposition3d(current_ca_coords, self.prior_ca_coords)
                    # self._chain_transforms.extend([dict(rotation=np.matmul(transform['rotation'], rot),
                    #                                     translation=transform['translation'] + tx)
                    #                                for transform in self.__chain_transforms[1:]])
                    # self._chain_transforms.extend([dict(rotation=transform['rotation'], translation=transform['translation'],
                    #                                     rotation2=rot, translation2=tx)
                    #                                for transform in self.__chain_transforms[1:]])
                    # missing_at = '__chain_transforms'
                    for transform in self.__chain_transforms[1:]:
                        chain_coords = np.matmul(np.matmul(self.prior_ca_coords, np.transpose(transform['rotation']))
                                                 + transform['translation'], np.transpose(new_rot)) + new_tx
                        _, rot, tx, _ = superposition3d(chain_coords, current_ca_coords)
                        self._chain_transforms.append(dict(rotation=rot, translation=tx))
                self._chain_transforms.insert(0, dict(rotation=identity_matrix, translation=origin))
            except AttributeError:  # no prior_ca_coords or __chain_transforms
                pass
                # self.log.info('%s chain_transform %s because missing %s' % (self.name, 'LastAttributeError', missing_at))

            return self._chain_transforms

    @chain_transforms.setter
    def chain_transforms(self, value):
        self._chain_transforms = value

    def remove_chain_transforms(self):
        """Remove chain_transforms attribute in preparation for coordinate movement"""
        self._chains.clear()
        self.__chain_transforms = self.chain_transforms
        del self._chain_transforms
        self.prior_ca_coords = self.get_ca_coords()

    @property
    def chains(self):
        """Returns:
            (list[Entity]): Transformed copies of the Entity itself
        """
        if self._chains:  # check if empty list in the case that coords have been changed and chains cleared
            return self._chains
        else:  # empty list, populate with entity copies
            self._chains = [self.return_transformed_copy(**transform) for transform in self.chain_transforms]
            chain_ids = self.chain_ids
            self.log.debug('Entity chains property has %s chains because the underlying chain_transforms has %d. '
                           'chain_ids has %d' % (len(self._chains), len(self.chain_transforms), len(chain_ids)))
            for idx, chain in enumerate(self._chains):
                # set entity.chain_id which sets all residues
                chain.chain_id = chain_ids[idx]
            return self._chains

    @property
    def reference_sequence(self) -> str:
        """Return the reference sequence constituting the entire open reading frame for an Entity instance

        Returns:
            (str)
        """
        try:
            return self._reference_sequence
        except AttributeError:
            self.retrieve_sequence_from_api()
            if not self._reference_sequence:
                self.log.warning('The reference sequence could not be found. Using the observed Residue sequence '
                                 'instead')
                self._reference_sequence = self.structure_sequence
            return self._reference_sequence

    @reference_sequence.setter
    def reference_sequence(self, sequence):
        self._reference_sequence = sequence

    @property
    def disorder(self) -> dict[int, dict[str, str]]:
        """Return the Residue number keys where disordered residues are found by comparison of the genomic (construct)
        sequence with that of the structure sequence

        Returns:
            Mutation index to mutations in the format of {1: {'from': 'A', 'to': 'K'}, ...}
        """
        try:
            return self._disorder
        except AttributeError:
            self._disorder = generate_mutations(self.reference_sequence, self.structure_sequence, only_gaps=True)
            return self._disorder

    def chain(self, chain_name):
        """Fetch and return a Chain by name

        Returns:
            (Entity)
        """
        for idx, chain_id in enumerate(self.chain_ids):
            if chain_id == chain_name:
                try:
                    return self._chains[idx]
                except IndexError:  # could make all the chains too?
                    chain = self.return_transformed_copy(**self.chain_transforms[idx])
                    chain.chain_id = chain_name
                    return chain
        return

    def retrieve_sequence_from_api(self, entity_id=None):
        if not entity_id:
            if len(self.name.split('_')) != 2:  # Todo if name=None then throws attribute error
                self.log.warning('For Entity method .%s, if an entity_id isn\'t passed and the Entity name %s is not '
                                 'the correct format (1abc_1), the query will fail. Retrieving closest entity_id by PDB'
                                 ' API structure sequence' % (self.retrieve_sequence_from_api.__name__, self.name))
                entity_id = retrieve_entity_id_by_sequence(self.sequence)
                if not entity_id:
                    self.reference_sequence = None
                    return
            else:
                entity_id = self.name
        self.log.debug('Retrieving Entity reference sequence from PDB')
        self.reference_sequence = get_entity_reference_sequence(entity_id=entity_id)

    # def retrieve_info_from_api(self):
    #     """Retrieve information from the PDB API about the Entity
    #
    #     Sets:
    #         self.api_entry (dict): {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
    #     """
    #     self.api_entry = get_pdb_info_by_entity(self.name)

    # def set_up_captain_chain(self):
    #     raise DesignError('This function is not implemented yet')
    #     self.is_oligomeric = True
    #     for chain in self.chains:
    #         dum = True
    #         # find the center of mass for all chains
    #         # transform the entire group to the origin by subtracting com
    #         # superimpose each chain on the captain, returning the quaternion
    #         # using the quaternion, find the major, minor axis which should be close to integers
    #         # Check for bad position by non-canonical rotation integers
    #         # orient the chains so that they are in a canonical orientation
    #         # for each chain find the rotation which aligns it with its captain
    #         # using the inverse of the orient rotation, apply to the aligned rotation to generate a cumulative rotation
    #         # invert the translation of the center of mass to the origin
    #         # for all use of these chains in the future, ensure the found transformations are applied to each chain

    @property
    def oligomer(self) -> Structures:
        """Access the oligomeric Structure which is a copy of the Entity plus any additional symmetric mate chains

        Returns:
            Structures object with the underlying chains in the oligomer
        """
        try:
            return self._oligomer
        except AttributeError:
            # if not self.is_oligomeric:
            #     self.log.warning('The oligomer was requested but the Entity %s is not oligomeric. Returning the Entity '
            #                      'instead' % self.name)
            self._oligomer = self.chains
            # self._oligomer = Structures(self.chains)
            return self._oligomer

    def remove_mate_chains(self):
        """Clear the Entity of all Chain and Oligomer information"""
        self.chain_transforms = [dict(rotation=identity_matrix, translation=origin)]
        self.number_of_monomers = 1
        # try:
        self.chains.clear()
        # except AttributeError:
        #     pass
        try:
            del self._chain_ids
        except AttributeError:
            pass

    def make_oligomer(self, symmetry=None, rotation=None, translation=None, rotation2=None, translation2=None):
        """Given a symmetry and transformational mapping, generate oligomeric copies of the Entity

        Assumes that the symmetric system treats the canonical symmetric axis as the Z-axis, and if the Entity is not at
        the origin, that a transformation describing its current position relative to the origin is passed so that it
        can be moved to the origin. At the origin, makes the required oligomeric rotations, to generate an oligomer
        where symmetric copies are stored in the .chains attribute then reverses the operations back to original
        reference frame if any was provided

        Sets:
            # self.chains
            self.chain_transforms
            self.chain_ids
        """
        try:
            if symmetry == 'C1':  # not symmetric
                return
            elif symmetry in cubic_point_groups:
                # must transpose these along last axis as they are pre-transposed upon creation
                rotation_matrices = point_group_symmetry_operators[symmetry].swapaxes(-2, -1)
                degeneracy_matrices = None  # Todo may need to add T degeneracy here!
            elif 'D' in symmetry:  # provide a 180 degree rotation along x (all D orient symmetries have axis here)
                rotation_matrices = get_rot_matrices(rotation_range[symmetry.replace('D', 'C')], 'z', 360)
                degeneracy_matrices = [identity_matrix, flip_x_matrix]
            else:  # symmetry is cyclic
                rotation_matrices = get_rot_matrices(rotation_range[symmetry], 'z', 360)
                degeneracy_matrices = None
            degeneracy_rotation_matrices = make_rotations_degenerate(rotation_matrices, degeneracy_matrices)
        except KeyError:
            raise ValueError('The symmetry %s is not a viable symmetry! You should try to add compatibility for it'
                             ' if you believe this is a mistake' % symmetry)
        self.symmetry = symmetry
        self.is_oligomeric = True
        if rotation is None:
            rotation, inv_rotation = identity_matrix, identity_matrix
        else:
            inv_rotation = np.linalg.inv(rotation)
        if translation is None:
            translation = origin

        if rotation2 is None:
            rotation2, inv_rotation2 = identity_matrix, identity_matrix
        else:
            inv_rotation2 = np.linalg.inv(rotation2)
        if translation2 is None:
            translation2 = origin
        # this is helpful for dihedral symmetry as entity must be transformed to origin to get canonical dihedral
        # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
        # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
        # and undo symmetry expansion matrices
        # centered_coords = transform_coordinate_sets(self.coords, translation=-translation2,
        # centered_coords = transform_coordinate_sets(self._coords.coords, translation=-translation2)
        cb_coords = self.get_cb_coords()
        centered_coords = transform_coordinate_sets(cb_coords, translation=-translation2)

        centered_coords_inv = transform_coordinate_sets(centered_coords, rotation=inv_rotation2,
                                                        translation=-translation, rotation2=inv_rotation)
        # debug_pdb = self.chain_representative.__copy__()
        # debug_pdb.replace_coords(centered_coords_inv)
        # debug_pdb.write(out_path='invert_set_invert_rot%s.pdb' % self.name)

        # set up copies to match the indices of entity
        # self.chain_representative.start_indices(dtype='atom', at=self.atom_indices[0])
        # self.chain_representative.start_indices(dtype='residue', at=self.residue_indices[0])
        # self.chains.append(self.chain_representative)
        self.chain_transforms.clear()
        # self.chain_ids.clear()
        # try:
        #     del self._number_of_monomers
        # except AttributeError:
        #     pass
        try:
            del self._chain_ids
        except AttributeError:
            pass

        number_of_monomers = 0
        for degeneracy_matrices in degeneracy_rotation_matrices:
            for rotation_matrix in degeneracy_matrices:
                number_of_monomers += 1
                rot_centered_coords = transform_coordinate_sets(centered_coords_inv, rotation=rotation_matrix)
                # debug_pdb2 = self.chain_representative.__copy__()
                # debug_pdb2.replace_coords(rot_centered_coords)
                # debug_pdb2.write(out_path='invert_set_invert_rot_ROT-%d%s.pdb' % (idx, self.name))
                new_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation, translation=translation,
                                                       rotation2=rotation2, translation2=translation2)
                # final_coords = transform_coordinate_sets(temp_coords, rotation=rot_op, translation=translation2)
                # # Entity representative stays in the .chains attribute as chain[0] given the iterator slice above
                # sub_symmetry_mate_pdb = self.chain_representative.__copy__()
                # sub_symmetry_mate_pdb.replace_coords(new_coords)
                # sub_symmetry_mate_pdb.set_atoms_attributes(chain=Structure.available_letters[idx])
                # sub_symmetry_mate_pdb.name = Structure.available_letters[idx]
                # sub_symmetry_mate_pdb.write(out_path='make_oligomer_transformed_CHAIN-%d%s.pdb' % (idx, self.name))
                # self.chains.append(sub_symmetry_mate_pdb)
                # # self.chains[idx] = sub_symmetry_mate_pdb
                _, rot, tx, _ = superposition3d(new_coords, cb_coords)
                self.chain_transforms.append(dict(rotation=rot, translation=tx))
        self.number_of_monomers = number_of_monomers
        # self.chain_ids = list(self.return_chain_generator())[:self.number_of_monomers]
        # self.log.debug('After make_oligomers, the chain_ids for %s are %s' % (self.name, self.chain_ids))

    def translate(self, translation):
        """Perform a translation to the Structure ensuring only the Structure container of interest is translated
        ensuring the underlying coords are not modified"""
        self.remove_chain_transforms()
        super().translate(translation)

    def rotate(self, rotation):
        """Perform a rotation to the Structure ensuring only the Structure container of interest is rotated ensuring the
        underlying coords are not modified"""
        self.remove_chain_transforms()
        super().rotate(rotation)

    def transform(self, **kwargs):
        """Perform a specific transformation to the Structure ensuring only the Structure container of interest is
        transformed ensuring the underlying coords are not modified"""
        self.remove_chain_transforms()
        super().transform(**kwargs)

    def format_seqres(self, asu=True, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            asu=True (bool): Whether to output the Entity ASU or the full oligomer
            **kwargs
        Returns:
            (str)
        """
        formated_reference_sequence = \
            ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in self.reference_sequence)))
        chain_length = len(self.reference_sequence)
        asu_slice = 1 if asu else None
        # chains = self.chains if asu else None
        return '%s\n' \
            % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain.chain_id, chain_length)
                        % formated_reference_sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                        for chain in self.chains[:asu_slice]
                        for line_number in range(1, 1 + ceil(len(formated_reference_sequence)/seq_res_len)))

    # Todo overwrite Structure.write() method with oligomer=True flag?
    def write_oligomer(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, **kwargs) -> str | None:
        #               header=None,
        """Write oligomeric Structure Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
        Returns:
            The name of the written file if out_path is used
        """
        offset = 0
        if file_handle:
            if self.chains:
                for chain in self.chains:
                    file_handle.write('%s\n' % chain.return_atom_string(atom_offset=offset, **kwargs))
                    offset += chain.number_of_atoms

        if out_path:
            if self.chains:
                with open(out_path, 'w') as outfile:
                    self.write_header(outfile, asu=False, **kwargs)  # function implies we want all chains, i.e. asu=False
                    for idx, chain in enumerate(self.chains, 1):
                        outfile.write('%s\n' % chain.return_atom_string(atom_offset=offset, **kwargs))
                        offset += chain.number_of_atoms

            return out_path

    # def orient(self, symmetry: str = None, log: os.PathLike = None):
    #     """Orient a symmetric PDB at the origin with its symmetry axis canonically set on axes defined by symmetry
    #     file. Automatically produces files in PDB numbering for proper orient execution
    #
    #     Keyword Args:
    #         symmetry=None (str): What is the symmetry of the specified PDB?
    #         log=None (os.PathLike): If there is a log specific for orienting
    #     """
    #     # orient_oligomer.f program notes
    #     # C		Will not work in any of the infinite situations where a PDB file is f***ed up,
    #     # C		in ways such as but not limited to:
    #     # C     equivalent residues in different chains don't have the same numbering; different subunits
    #     # C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
    #     # C		of separate IDs; multiple conformations are written out for the same subunit structure
    #     # C		(as in an NMR ensemble), negative residue numbers, etc. etc.
    #     # must format the input.pdb in an acceptable manner
    #     subunit_number = valid_subunit_number.get(symmetry, None)
    #     if not subunit_number:
    #         raise ValueError('Symmetry %s is not a valid symmetry. Please try one of: %s' %
    #                          (symmetry, ', '.join(valid_subunit_number.keys())))
    #     if not log:
    #         log = self.log
    #
    #     if self.filepath:
    #         pdb_file_name = os.path.basename(self.filepath)
    #     else:
    #         pdb_file_name = '%s.pdb' % self.name
    #     # Todo change output to logger with potential for file and stdout
    #
    #     if self.number_of_monomers < 2:
    #         raise ValueError('Cannot orient a file with only a single chain. No symmetry present!')
    #     elif self.number_of_monomers != subunit_number:
    #         raise ValueError('%s\n Oligomer could not be oriented: It has %d subunits while %d '
    #                          'are expected for %s symmetry\n\n'
    #                          % (pdb_file_name, self.number_of_monomers, subunit_number, symmetry))
    #
    #     orient_input = os.path.join(orient_dir, 'input.pdb')
    #     orient_output = os.path.join(orient_dir, 'output.pdb')
    #
    #     def clean_orient_input_output():
    #         if os.path.exists(orient_input):
    #             os.remove(orient_input)
    #         if os.path.exists(orient_output):
    #             os.remove(orient_output)
    #
    #     clean_orient_input_output()
    #     # self.reindex_all_chain_residues()  TODO test efficacy. It could be that this screws up more than helps
    #     # have to change residue numbering to PDB numbering
    #     self.write_oligomer(orient_input, pdb_number=True)
    #     # self.renumber_residues_by_chain()
    #
    #     p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    #                          stderr=subprocess.PIPE, cwd=orient_dir)
    #     in_symm_file = os.path.join(orient_dir, 'symm_files', symmetry)
    #     stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
    #     # stderr = stderr.decode()  # turn from bytes to string 'utf-8' implied
    #     # stdout = pdb_file_name + stdout.decode()[28:]
    #     log.info(pdb_file_name + stdout.decode()[28:])
    #     log.info(stderr.decode())
    #     if not os.path.exists(orient_output) or os.stat(orient_output).st_size == 0:
    #         # orient_log = os.path.join(out_dir, orient_log_file)
    #         log_file = getattr(log.handlers[0], 'baseFilename', None)
    #         log_message = '. Check %s for more information' % log_file if log_file else ''
    #         error_string = 'orient_oligomer could not orient %s%s' % (pdb_file_name, log_message)
    #         raise RuntimeError(error_string)
    #
    #     oriented_pdb = PDB.from_file(orient_output, name=self.name, log=log)
    #     orient_fixed_struct = oriented_pdb.chains[0]
    #     moving_struct = self.chains[0]
    #     try:
    #         _, rot, tx, _ = superposition3d(orient_fixed_struct.get_cb_coords(), moving_struct.get_cb_coords())
    #     except ValueError:  # we have the wrong lengths, lets subtract a certain amount by performing a seq alignment
    #         # rot, tx = None, None
    #         orient_fixed_seq = orient_fixed_struct.sequence
    #         moving_seq = moving_struct.sequence
    #         # while not rot:
    #         #     try:
    #         # moving coords are from the pre-orient structure where orient may have removed residues
    #         # lets try to remove those residues by doing an alignment
    #         align_orient_seq, align_moving_seq, *_ = generate_alignment(orient_fixed_seq, moving_seq, local=True)
    #         # align_seq_1.replace('-', '')
    #         # orient_idx1 = moving_seq.find(align_orient_seq.replace('-', '')[0])
    #         for orient_idx1, aa in enumerate(align_orient_seq):
    #             if aa != '-':  # we found the first aligned residue
    #                 break
    #         orient_idx2 = orient_idx1 + len(align_orient_seq.replace('-', ''))
    #         # starting_index_of_seq2 = moving_seq.find(align_moving_seq.replace('-', '')[0])
    #         # # get the first matching index of the moving_seq from the first aligned residue
    #         # ending_index_of_seq2 = starting_index_of_seq2 + align_moving_seq.rfind(moving_seq[-1])  # find last index of reference
    #         _, rot, tx, _ = superposition3d(orient_fixed_struct.get_cb_coords(),
    #                                         moving_struct.get_cb_coords()[orient_idx1:orient_idx2])
    #         # except ValueError:
    #         #     rot, tx = None, None
    #     self.transform(rotation=rot, translation=tx)
    #     clean_orient_input_output()

    def find_chain_symmetry(self, struct_file: str | bytes = None) -> str | bytes:
        """Search for the chains involved in a complex using a truncated make_symmdef_file.pl script

        Requirements - all chains are the same length
        This script translates the PDB center of mass to the origin then uses quaternion geometry to solve for the
        rotations which superimpose chains provided by -i onto a designated chain (usually A). It returns the order of
        the rotation as well as the axis along which the rotation must take place. The axis of the rotation only needs
        to be translated to the center of mass to recapitulate the specific symmetry operation.

        perl symdesign/dependencies/rosetta/sdf/scout_symmdef_file.pl -p 1ho1_tx_4.pdb -i B C D E F G H
        >B:3-fold axis: -0.00800197 -0.01160998 0.99990058
        >C:3-fold axis: 0.00000136 -0.00000509 1.00000000
        Args:
            struct_file: The location of the input .pdb file
        Sets:
            self.rotation_d (dict[str, dict[str, int | np.ndarray]])
        Returns:
            The name of the file written for symmetry definition file creation
        """
        if not struct_file:
            struct_file = self.write_oligomer(out_path='make_sdf_input-%s-%d.pdb' % (self.name, random() * 100000))

        # todo initiate this process in house using superposition3D for every chain
        scout_cmd = ['perl', scout_symmdef, '-p', struct_file, '-a', self.chain_ids[0], '-i'] + self.chain_ids[1:]
        self.log.debug('Scouting chain symmetry: %s' % subprocess.list2cmdline(scout_cmd))
        p = subprocess.Popen(scout_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()

        for line in out.decode('utf-8').strip().split('\n'):
            chain, symmetry, axis = line.split(':')
            self.rotation_d[chain] = \
                {'sym': int(symmetry[:6].rstrip('-fold')), 'axis': np.array(list(map(float, axis.strip().split())))}
            # the returned axis is from a center of mass at the origin as Structure has been translated there

        return struct_file

    def find_max_chain_symmetry(self):
        """Find the highest order symmetry in the Structure

        Sets:
            self.max_symmetry (str)
        """
        max_sym, max_chain = 0, None
        for chain, data in self.rotation_d.items():
            if data['sym'] > max_sym:
                max_sym = data['sym']
                max_chain = chain

        self.max_symmetry = max_chain

    def scout_symmetry(self, **kwargs) -> str | bytes:
        """Check the PDB for the required symmetry parameters to generate a proper symmetry definition file

        Sets:
            self.rotation_d (dict[str, dict[str, int | np.ndarray]])
            self.max_symmetry (str)
        Returns:
            The location of the oligomeric Structure
        """
        struct_file = self.find_chain_symmetry(**kwargs)
        self.find_max_chain_symmetry()

        return struct_file

    def is_dihedral(self, **kwargs) -> bool:
        """Report whether a structure is dihedral or not

        Sets:
            self.rotation_d (dict[str, dict[str, int | np.ndarray]])
            self.max_symmetry (str)
            self.dihedral_chain (str): The name of the chain that is dihedral
        Returns:
            True if the Structure is dihedral, False if not
        """
        if not self.max_symmetry:
            self.scout_symmetry(**kwargs)
        # ensure if the structure is dihedral a selected dihedral_chain is orthogonal to the maximum symmetry axis
        max_symmetry_data = self.rotation_d[self.max_symmetry]
        if self.number_of_monomers / max_symmetry_data['sym'] == 2:
            for chain, data in self.rotation_d.items():
                if data['sym'] == 2:
                    axis_dot_product = np.dot(max_symmetry_data['axis'], data['axis'])
                    if axis_dot_product < 0.01:
                        if np.allclose(data['axis'], [1, 0, 0]):
                            self.log.debug('The relation between %s and %s would result in a malformed .sdf file'
                                           % (self.max_symmetry, chain))
                            pass  # this will not work in the make_symmdef.pl script, we should choose orthogonal y-axis
                        else:
                            self.dihedral_chain = chain
                            return True
        elif 1 < self.number_of_monomers / max_symmetry_data['sym'] < 2:
            self.log.critical('The symmetry of %s is malformed! Highest symmetry (%d-fold) is less than 2x greater than'
                              ' the number (%d) of chains'
                              % (self.name, max_symmetry_data['sym'], self.number_of_monomers))

        return False

    def make_sdf(self, struct_file: str | bytes = None, out_path: str | bytes = os.getcwd(), **kwargs) -> \
            str | bytes:
        """Use the make_symmdef_file.pl script from Rosetta to make a symmetry definition file on the Structure

        perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb.pdb -i B -q

        Args:
            struct_file: The location of the input .pdb file
            out_path: The location the symmetry definition file should be written
        Keyword Args:
            modify_sym_energy_for_cryst=False (bool): Whether the symmetric energy produced in the file should be modified
            energy=2 (int): Scalar to modify the Rosetta energy by
        Returns:
            Symmetry definition filename
        """
        out_file = os.path.join(out_path, '%s.sdf' % self.name)
        # Todo Master branch reinstate
        # if os.path.exists(out_file):
        #     return out_file

        # if self.symmetry == 'C1':
        #     return
        # el
        if self.symmetry in cubic_point_groups:
            # if not struct_file:
            #     struct_file = self.write_oligomer(out_path='make_sdf_input-%s-%d.pdb' % (self.name, random() * 100000))
            sdf_mode = 'PSEUDO'
            self.log.warning('Using experimental symmetry definition file generation, proceed with caution as Rosetta '
                             'runs may fail due to improper set up')
        else:
            # if not struct_file:
            #     struct_file = self.scout_symmetry(struct_file=struct_file)
            sdf_mode = 'NCS'

        if not struct_file:
            struct_file = self.scout_symmetry(struct_file=struct_file)
        dihedral = self.is_dihedral(struct_file=struct_file)  # include so we don't write another struct_file
        if dihedral:  # dihedral_chain will be set
            chains = [self.max_symmetry, self.dihedral_chain]
        else:
            chains = [self.max_symmetry]

        sdf_cmd = \
            ['perl', make_symmdef, '-m', sdf_mode, '-q', '-p', struct_file, '-a', self.chain_ids[0], '-i'] + chains
        self.log.info('Creating symmetry definition file: %s' % subprocess.list2cmdline(sdf_cmd))
        # with open(out_file, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        if os.path.exists(struct_file):
            os.system('rm %s' % struct_file)
        assert p.returncode == 0, 'Symmetry definition file creation failed for %s' % self.name

        self.format_sdf(out.decode('utf-8').split('\n')[:-1], to_file=out_file, dihedral=dihedral, **kwargs)
        #               modify_sym_energy_for_cryst=False, energy=2)

        return out_file

    def format_sdf(self, lines: list, to_file: str | bytes = None,
                   out_path: str | bytes = os.getcwd(), dihedral: bool = False,
                   modify_sym_energy_for_cryst: bool = False, energy: int = None) -> str | bytes:
        """Ensure proper sdf formatting before proceeding

        Args:
            lines: The symmetry definition file lines
            to_file: The name of the symmetry definition file
            out_path: The location the symmetry definition file should be written
            dihedral: Whether the assembly is in dihedral symmetry
            modify_sym_energy_for_cryst: Whether the symmetric energy should match crystallographic systems
            energy: Scalar to modify the Rosetta energy by
        Returns:
            The location the symmetry definition file was written
        """
        subunits, virtuals, jumps_com, jumps_subunit, trunk = [], [], [], [], []
        for idx, line in enumerate(lines, 1):
            if line.startswith('xyz'):
                virtual = line.split()[1]
                if virtual.endswith('_base'):
                    subunits.append(virtual)
                else:
                    virtuals.append(virtual.lstrip('VRT'))
                # last_vrt = line + 1
            elif line.startswith('connect_virtual'):
                jump = line.split()[1].lstrip('JUMP')
                if jump.endswith('_to_com'):
                    jumps_com.append(jump[:-7])
                elif jump.endswith('_to_subunit'):
                    jumps_subunit.append(jump[:-11])
                else:
                    trunk.append(jump)
                last_jump = idx  # index where the VRTs and connect_virtuals end. The "last jump"

        assert set(trunk) - set(virtuals) == set(), 'Symmetry Definition File VRTS are malformed'
        assert self.number_of_monomers == len(subunits), 'Symmetry Definition File VRTX_base are malformed'

        if dihedral:  # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
            virtuals = [virtual for virtual in virtuals if len(virtual) > 1]  # subunit_
        else:
            if '' in virtuals:
                virtuals.remove('')

        jumps_com_to_add = set(virtuals).difference(jumps_com)
        count = 0
        if jumps_com_to_add != set():
            for count, jump_com in enumerate(jumps_com_to_add, count):
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_com VRT%s VRT%s_base'
                             % (jump_com, jump_com, jump_com))
            lines[-2] = lines[-2].strip() + (len(jumps_com_to_add) * ' JUMP%s_to_subunit') % tuple(jumps_com_to_add)

        jumps_subunit_to_add = set(virtuals).difference(jumps_subunit)
        if jumps_subunit_to_add != set():
            for count, jump_subunit in enumerate(jumps_subunit_to_add, count):
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_subunit VRT%s_base SUBUNIT'
                             % (jump_subunit, jump_subunit))
            lines[-1] = \
                lines[-1].strip() + (len(jumps_subunit_to_add) * ' JUMP%s_to_subunit') % tuple(jumps_subunit_to_add)

        if modify_sym_energy_for_cryst:
            # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits
            # where complex subunits = num_subunits - 1
            # new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes subunits are read in alphanumerical order
            # new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
            lines[1] = 'E = 2*%s+%s' \
                % (subunits[0], '+'.join('1*(%s:%s)' % (subunits[0], pair) for pair in subunits[1:]))
        else:
            if not energy:
                energy = len(subunits)
            lines[1] = 'E = %d*%s+%s' \
                % (energy, subunits[0], '+'.join('%d*(%s:%s)' % (energy, subunits[0], pair) for pair in subunits[1:]))

        if not to_file:
            to_file = os.path.join(out_path, '%s.sdf' % self.name)

        with open(to_file, 'w') as f:
            f.write('%s\n' % '\n'.join(lines))
        if count != 0:
            self.log.info('Symmetry Definition File "%s" was missing %d lines, so a fix was attempted. '
                          'Modelling may be affected' % (to_file, count))
        return to_file

    def format_missing_loops_for_design(self, max_loop_length=12, exclude_n_term=True, ignore_termini=False, **kwargs):
        """Process missing residue information to prepare the various files for loop modelling

        Keyword Args:
            max_loop_length=12 (int): The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modelling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file
        Returns:
            (tuple[list[tuple], dict[mapping[int, int]], int]):
                Loop start and ending indices, loop indices mapped to disordered residues, index of n-terminal residue
        """
        disordered_residues = self.disorder  # {residue_number: {'from': ,'to': }, ...}
        # disorder_indices = list(disordered_residues.keys())
        # disorder_indices = []  # holds the indices that should be inserted into the total residues to be modelled
        loop_indices, disorder_indices = [], {}  # holds the indices that should be inserted into the total residues to be modelled
        start_idx = 0  # initialize as an impossible value for blueprint formatting list comprehension
        excluded_disorder, segment_length = 0, 0  # iterate each missing segment length, and total excluded disorder
        loop_start, loop_end, n_term = None, None, False
        for idx, residue_number in enumerate(disordered_residues.keys(), 1):
            segment_length += 1
            if residue_number - 1 not in disordered_residues:
                # print('Residue number -1 not in loops', residue_number)
                loop_start = residue_number - 1 - excluded_disorder
                if loop_start <= 0:
                    # disordered_residues[loop_start] = residues[loop_start]
                # else:
                    # the disordered locations include the n-terminus, set start_idx to idx (should equal 1)
                    # start_idx = idx
                    n_term = True
            if residue_number + 1 not in disordered_residues and residue_number != len(self.reference_sequence):
                # print('Residue number +1 not in loops', residue_number)
                # print('Adding loop with length', segment_length)
                if segment_length <= max_loop_length:  # ensure modelling will be useful
                    if exclude_n_term and n_term:  # check if the n_terminus should be included
                        excluded_disorder += segment_length
                        n_term = False
                    else:  # include the segment in the disorder_indices
                        # print('Start index', start_idx)
                        loop_end = residue_number + 1 - excluded_disorder
                        loop_indices.append((loop_start, loop_end))
                        # disorder_indices.extend([loop_start, loop_end])
                        for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                            disorder_indices[residue_index] = residue_number - (segment_length - it)
                        disorder_indices[loop_start], disorder_indices[loop_end] = -1, -1  # out of bounds number
                        # disordered_residues[loop_start], disordered_residues[loop_end] = \
                        #     residues[loop_start - 1], residues[loop_end - 1]  # offset index
                        if n_term and idx != 1:
                            # if n-term was identified and not 1 (only start Met missing), save last idx of n-term insertion
                            start_idx = loop_end  # idx
                else:
                    excluded_disorder += segment_length
                segment_length = 0
                loop_start, loop_end = None, None

        if loop_start and not ignore_termini:  # when insertion is at the c-term, we hit a residue that didn't complete
            if segment_length <= max_loop_length:
                loop_end = loop_start + 1 + segment_length  # - excluded_disorder
                # disorder_indices.append(loop_start)
                loop_indices.append((loop_start, loop_end - 1))  # subtract as no c-terminal res attachment at c-termini
                # disorder_indices.extend(range(loop_start, loop_end))  # NO +1 don't need to include final index in range
                for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                    disorder_indices[residue_index] = residue_number - (segment_length - it)
                disorder_indices[loop_start], disorder_indices[loop_end] = -1, -1  # out of bounds number
                # disordered_residues[loop_start] = residues[loop_start - 1]

        return loop_indices, disorder_indices, start_idx

    # Todo move both of these to Structure/Pose. Requires using .reference_sequence in Structure/ or maybe Pose better
    def make_loop_file(self, out_path=os.getcwd(), **kwargs):
        """Format a loops file according to Rosetta specifications

        Keyword Args:
            out_path=os.getcwd() (str): The location the file should be written
            max_loop_length=12 (int): The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modelling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file
        Returns:
            (str): The path of the file
        """
        loop_file = os.path.join(out_path, '%s.loops' % self.name)
        loop_indices, _, _ = self.format_missing_loops_for_design(**kwargs)
        with open(loop_file, 'w') as f:
            f.write('%s\n' % '\n'.join('LOOP %d %d 0 0 1' % start_stop for start_stop in loop_indices))

        return loop_file

    def make_blueprint_file(self, out_path=os.getcwd(), **kwargs):
        """Format a blueprint file according to Rosetta specifications

        Keyword Args:
            out_path=os.getcwd() (str): The location the file should be written
            max_loop_length=12 (int): The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modelling due to Remodel Bug
        Returns:
            (str): The path of the file
        """
        out_file = os.path.join(out_path, '%s.blueprint' % self.name)
        disordered_residues = self.disorder  # {residue_number: {'from': ,'to': }, ...}
        # trying to remove tags at this stage runs into a serious indexing problem where tags need to be deleted from
        # disordered_residues and then all subsequent indices adjusted.

        # # look for existing tag to remove from sequence and save identity
        # available_tags = find_expression_tags(self.reference_sequence)
        # if available_tags:
        #     loop_sequences = ''.join(mutation['from'] for mutation in disordered_residues)
        #     remove_loop_pairs = []
        #     for tag in available_tags:
        #         tag_location = loop_sequences.find(tag['sequences'])
        #         if tag_location != -1:
        #             remove_loop_pairs.append((tag_location, len(tag['sequences'])))
        #     for tag_start, tag_length in remove_loop_pairs:
        #         for
        #
        #     # untagged_seq = remove_expression_tags(loop_sequences, [tag['sequence'] for tag in available_tags])

        _, disorder_indices, start_idx = self.format_missing_loops_for_design(**kwargs)

        residues = self.residues
        # for residue_number in sorted(disorder_indices):  # ensure ascending order, insert is dependent on prior inserts
        for residue_index, disordered_index in disorder_indices.items():  # ensure ascending order, insert is dependent on prior inserts
            mutation = disordered_residues.get(disordered_index)
            if mutation:  # add disordered residues if they exist
                residues.insert(residue_index - 1, mutation['from'])  # offset to match residues zero-index

        #              index AA SS Choice AA
        structure_str   = '%d %s %s'
        loop_str        = '%d X %s PIKAA %s'
        with open(out_file, 'w') as f:
            f.write('%s\n'
                    % '\n'.join([structure_str % (residue.number, protein_letters_3to1_extended.get(residue.type.title()),
                                                  'L PIKAA %s' % protein_letters_3to1_extended.get(residue.type.title())
                                                  if idx in disorder_indices else '.')
                                 if isinstance(residue, Residue)
                                 else loop_str % (1 if idx < start_idx else 0, 'L', residue)
                                 for idx, residue in enumerate(residues, 1)]))
        return out_file

    # def update_attributes(self, **kwargs):
    #     """Update attributes specified by keyword args for all member containers"""
    #     # super().update_attributes(**kwargs)  # this is required to set the base Structure with the kwargs
    #     # self.set_structure_attributes(self.chains, **kwargs)
    #     for structure in self.structure_containers:
    #         self.set_structure_attributes(structure, **kwargs)

    # def copy_structures(self):
    #     # super().copy_structures([self.chains])
    #     super().copy_structures(self.structure_containers)

    def __copy__(self):
        other = super().__copy__()
        # # create a copy of all chains
        # # structures = [other.chains]
        # # other.copy_structures(structures)
        # other.copy_structures()  # NEVERMIND uses self.structure_containers... does this use Entity version?
        # # attributes were updated in the super().__copy__, now need to set attributes in copied chains
        # # This style v accomplishes the update that the super().__copy__() started using self.structure_containers...
        # other.update_attributes(residues=other._residues, coords=other._coords)
        if other.is_oligomeric:
            # self.log.info('Copy Entity. Clearing chains, chain_transforms')
            # other._chains.clear()
            other.remove_chain_transforms()
            # other.__chain_transforms = other.chain_transforms  # requires update before copy
            # del other._chain_transforms
            # other.prior_ca_coords = other.get_ca_coords()  # update these as next generation will rely on them for chain_transforms

        return other

    # def __key(self):
    #     return (self.uniprot_id, *super().__key())  # without uniprot_id, could equal a chain...
    #     # return self.uniprot_id


def superposition3d(fixed_coords, moving_coords, a_weights=None, allow_rescale=False, report_quaternion=False):
    """Takes two arrays of xyz coordinates (same length), and attempts to superimpose them using rotations,translations,
    and (optionally) rescale operations in order to minimize the root-mean-squared-distance (RMSD) between them. These
    operations should be applied to the "moving_coords" argument.

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    The quaternion_matrix has the first row storing cos(θ/2) (where θ is the rotation angle). The following 3 rows
    form a vector (of length sin(θ/2)), pointing along the axis of rotation.
    Details: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        fixed_coords (numpy.ndarray): The coordinates for the 'frozen' object
        moving_coords (numpy.ndarray): The coordinates for the 'mobile' object
    Keyword Args:
        a_weights=None (numpy.ndarray): Weights for the calculation of RMSD
        allow_rescale=False (bool): Attempt to rescale the mobile point cloud in addition to translation/rotation?
        report_quaternion=False (bool): Whether to report the rotation angle and axis in typical quaternion fashion
    Returns:
        (tuple[float, numpy.ndarray, numpy.ndarray, float]): rmsd, rotation/quaternion_matrix, translation_vector,
        scale_factor
    """
    # convert input lists to numpy arrays
    # fixed_coords = np.array(fixed_coords)
    # moving_coords = np.array(moving_coords)

    if fixed_coords.shape[0] != moving_coords.shape[0]:
        raise ValueError('%s: Inputs should have the same size. Input 1=%d, 2=%d' %
                         (superposition3d.__name__, fixed_coords.shape[0], moving_coords.shape[0]))

    number_of_points = fixed_coords.shape[0]
    # Find the center of mass of each object:
    # convert weights into array
    if not a_weights or len(a_weights) == 0:
        a_weights = np.full((number_of_points, 1), 1.0)
    else:
        # reshape aWeights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)

    a_center_f = np.sum(fixed_coords * a_weights, axis=0)
    a_center_m = np.sum(moving_coords * a_weights, axis=0)
    sum_weights = np.sum(a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    if sum_weights != 0:
        a_center_f /= sum_weights
        a_center_m /= sum_weights
    aa_xf = fixed_coords - a_center_f
    aa_xm = moving_coords - a_center_m

    # Calculate the "M" array from the Diamond paper (equation 16)
    m = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate Q (equation 17)
    q = m + m.T - 2 * np.eye(3) * np.trace(m)

    # Calculate v (equation 18)  #KM this appears to be the cross product...
    v = np.empty(3)
    v[0] = m[1][2] - m[2][1]
    v[1] = m[2][0] - m[0][2]
    v[2] = m[0][1] - m[1][0]

    # Calculate "P" (equation 22)
    P = np.zeros((4, 4))
    P[:3, :3] = q
    P[3, :3] = v
    P[:3, 3] = v
    # [[ q[0][0] q[0][1] q[0][2] v[0] ]
    #  [ q[1][0] q[1][1] q[1][2] v[1] ]
    #  [ q[2][0] q[2][1] q[2][2] v[2] ]
    #  [ v[0]    v[1]    v[2]    0    ]]

    # Calculate "p".
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below.)
    # First, specify the default value for p:
    p = np.zeros(4)
    p[3] = 1.0           # p = [0,0,0,1]    default value
    pPp = 0.0            # = p^T * P * p    (zero by default)
    singular = (number_of_points < 2)   # (it doesn't make sense to rotate a single point)

    try:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
        a_eigenvals, aa_eigenvects = eigh(P)
    except LinAlgError:
        singular = True  # (I have never seen this happen.)

    if not singular:  # (don't crash if the caller supplies nonsensical input)
        i_eval_max = np.argmax(a_eigenvals)
        pPp = np.max(a_eigenvals)
        p[:] = aa_eigenvects[:, i_eval_max]  # pull out the largest magnitude eigenvector

    # normalize the vector
    # (It should be normalized already, but just in case it is not, do it again)
    p /= np.linalg.norm(p)

    # Finally, calculate the rotation matrix corresponding to "p"
    # (p is in backwards-quaternion format)
    """

    aa_rotate = np.empty((3, 3))
    aa_rotate[0][0] = (p[0]*p[0])-(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[1][1] = -(p[0]*p[0])+(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[2][2] = -(p[0]*p[0])-(p[1]*p[1])+(p[2]*p[2])+(p[3]*p[3])
    aa_rotate[0][1] = 2*(p[0]*p[1] - p[2]*p[3])
    aa_rotate[1][0] = 2*(p[0]*p[1] + p[2]*p[3])
    aa_rotate[1][2] = 2*(p[1]*p[2] - p[0]*p[3])
    aa_rotate[2][1] = 2*(p[1]*p[2] + p[0]*p[3])
    aa_rotate[0][2] = 2*(p[0]*p[2] + p[1]*p[3])
    aa_rotate[2][0] = 2*(p[0]*p[2] - p[1]*p[3])

    # Alternatively, in modern python versions, this code also works:
    """
    the_rotation = Rotation.from_quat(p)
    aa_rotate = the_rotation.as_matrix()

    # Optional: Decide the scale factor, c
    c = 1.0   # by default, don't rescale the coordinates
    if allow_rescale and not singular:
        weightaxaixai_moving = np.sum(a_weights * aa_xm ** 2)
        weightaxaixai_fixed = np.sum(a_weights * aa_xf ** 2)

        c = (weightaxaixai_fixed + pPp) / weightaxaixai_moving

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    e0 = np.sum((aa_xf - c * aa_xm) ** 2)
    sum_sqr_dist = max(0, e0 - c * 2.0 * pPp)

    rmsd = 0.0
    if sum_weights != 0.0:
        rmsd = np.sqrt(sum_sqr_dist/sum_weights)

    # Lastly, calculate the translational offset:
    # Recall that:
    # RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = a_center_f[]       and       a_center_m[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    a_translate = a_center_f - np.matmul(c * aa_rotate, a_center_m).T.reshape(3,)

    if report_quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        # q = np.empty(4)
        # q[0], q[1], q[2], q[3] = p[3], p[0], p[1], p[2]
        q = np.array([p[3], p[0], p[1], p[2]])
        return rmsd, q, a_translate, c
    else:
        return rmsd, aa_rotate, a_translate, c


def parse_stride(stride_file, **kwargs):
    """From a Stride file, parse information for residue level secondary structure assignment

    Sets:
        self.secondary_structure
    """
    with open(stride_file, 'r') as f:
        stride_output = f.readlines()

    return ''.join(line[24:25] for line in stride_output if line[0:3] == 'ASG')


reference_residues = unpickle(reference_residues_pkl)  # zero-indexed 1 letter alphabetically sorted aa at the origin
reference_aa = Structure.from_residues(residues=reference_residues)
# pickle_object(ref, '/home/kylemeador/symdesign/data/AAreferenceStruct.pkl', out_path='')
