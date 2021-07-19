import os
import subprocess
from copy import copy  # , deepcopy
from collections.abc import Iterable
from itertools import repeat
from random import random, randint

import numpy as np
from numpy.linalg import eigh, LinAlgError
from sklearn.neighbors import BallTree  # , KDTree, NearestNeighbors
from scipy.spatial.transform import Rotation
from Bio.Data.IUPACData import protein_letters, protein_letters_1to3, protein_letters_3to1_extended

from PathUtils import free_sasa_exe_path, stride_exe_path, errat_exe_path, make_symmdef, scout_symmdef, \
    reference_residues_pkl
from SymDesignUtils import start_log, null_log, DesignError, unpickle
from Query.PDB import get_entity_reference_sequence, get_pdb_info_by_entity  # get_pdb_info_by_entry, query_entity_id
from SequenceProfile import SequenceProfile
from classes.SymEntry import identity_matrix, get_rot_matrices, RotRangeDict, flip_x_matrix, get_degen_rotmatrices
from utils.GeneralUtils import transform_coordinate_sets

# globals
logger = start_log(name=__name__)
gxg_sasa = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
            'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
            'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167, 'GLU': 223, 'GLN': 225, 'GLY': 104, 'HIS': 224,
            'ILE': 197, 'LEU': 201, 'LYS': 236, 'MET': 224, 'PHE': 240, 'PRO': 159, 'SER': 155, 'THR': 172, 'TRP': 285,
            'TYR': 263, 'VAL': 174}  # from table 1, theoretical values of Tien et al. 2013
origin = np.array([0, 0, 0])


class StructureBase:
    """Collect extra keyword arguments such as:
        chains, entities, seqres, multimodel, lazy, solve_discrepancy
    """
    def __init__(self, chains=None, entities=None, seqres=None, multimodel=None, lazy=None, solve_discrepancy=None,
                 sequence=None, cryst=None, cryst_record=None, design=None, resolution=None, space_group=None,
                 query_by_sequence=True, entity_names=None, **kwargs):
        super().__init__(**kwargs)


class Structure(StructureBase):
    """Structure object handles Atom/Residue/Coords manipulation of all Structure containers.
    Must pass atoms, residues, residue_indices, or coords to use most methods without issues
    to initialize
    """
    available_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 'abcdefghijklmnopqrstuvwyz0123456789~!@#$%^&*()-+={}[]|:;<>?'

    def __init__(self, atoms=None, residues=None, residue_indices=None, name=None, coords=None, log=None, **kwargs):
        self._coords = None
        self._atoms = None
        self._atom_indices = None
        self._residues = None
        self._residue_indices = None
        self._coords_residue_index = None
        self.name = name
        self.secondary_structure = None
        self.sasa = None
        self.structure_containers = []

        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if atoms is not None:
            self.atoms = atoms
            if coords is None:
                try:
                    coords = [atom.coords for atom in atoms]
                except AttributeError:
                    raise DesignError('Can\'t initialize Structure with Atom objects lacking coords if no _coords are '
                                      'passed! Either pass Atom objects with coords or pass _coords.')
                self.coords = coords
        if residues is not None:
            if residue_indices:
                self.residue_indices = residue_indices
                self.set_residues(residues)
                if coords is None:
                    self.coords = self.residues[0]._coords
                #     try:
                #         coords = [atom.coords for residue in residues for atom in residue.atoms]
                #     except AttributeError:
                #         raise DesignError('Can\'t initialize Structure with Atom objects '
                #                           'lacking coords! Either pass Atom objects with coords or pass coords.')
                #     self.reindex_atoms()
                #     self.coords = coords
            else:
                raise DesignError('Without passing residue_indices, can\'t initialize Structure with residue objects '
                                  'lacking coords! Either pass Atom objects with coords or pass coords.')
        if coords is not None:  # must go after Atom containers as atoms don't have any/right coordinate info
            self.coords = coords

        super().__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms=None, coords=None, **kwargs):
        return cls(atoms=atoms, coords=coords, **kwargs)

    @classmethod
    def from_residues(cls, residues=None, residue_indices=None, coords=None, **kwargs):
        return cls(residues=residues, residue_indices=residue_indices, coords=coords, **kwargs)

    @property  # Todo these do nothing and could be removed
    def name(self):
        """Returns: (str)"""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def coords(self):
        """From the larger array of Coords attached to a PDB object, get the specific Coords for the subset of Atoms
        belonging to the specific Structure instance
        Returns:
            (Numpy.ndarray)
        """
        return self._coords.coords[self._atom_indices]

    @coords.setter
    def coords(self, coords):
        """Replace the Structure, Atom, and Residue coordinates with specified Coords Object or numpy.ndarray"""
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            self._coords = Coords(coords)
        assert len(self.atoms) == len(self.coords), '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
                                                    % (self.name, len(self.atoms), len(self.coords))

    def set_coords(self, coords):
        """Set the coordinates for the Structure as a Coord object. Additionally, updates all member Residues with the
        Coords object and maps the atom/coordinate index to each Residue, residue atom index pair.

        Only use set_coords once per Structure object creation otherwise Structures with multiple containers will be
        corrupted"""
        self.coords = coords
        # self.set_atoms_attributes(coords=self._coords)  # atoms doesn't have coords now
        self.set_residues_attributes(coords=self._coords)
        # self.store_coordinate_index_residue_map()
        self.coords_indexed_residues = [(res_idx, res_atom_idx) for res_idx, residue in enumerate(self.residues)
                                        for res_atom_idx in residue.range]

    @property
    def is_structure_owner(self):
        """Check to see if the Structure is the owner of it's Coord and Atom attributes or if there is a larger
        Structure that maintains them"""
        return True if self._coords_residue_index is not None else False

    @property
    def atom_indices(self):  # In Residue too
        """Returns: (list[int])"""
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices):
        """Set the Structure atom indices to a list of integers"""
        self._atom_indices = indices

    def start_indices(self, dtype=None, at=0):
        """Modify Structure container indices by a set integer amount"""
        try:
            indices = getattr(self, '%s_indices' % dtype)
        except AttributeError:
            raise AttributeError('The dtype %s_indices was not found the Structure object. Possible values of dtype are'
                                 ' atom or residue' % dtype)
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
    def atoms(self):
        """Returns: (list[Atom])"""
        return self._atoms.atoms[self._atom_indices].tolist()

    @atoms.setter
    def atoms(self, atoms):
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
    def number_of_atoms(self):
        """Returns: (int)"""
        return len(self._atom_indices)

    @property
    def residue_indices(self):
        """Returns: (list[int])"""
        return self._residue_indices

    @residue_indices.setter
    def residue_indices(self, indices):
        """Set the Structure residue indices to a list of integers"""
        self._residue_indices = indices  # np.array(indices)

    @property
    def residues(self):
        """Returns: (list[Residue])"""
        return self._residues.residues[self._residue_indices].tolist()

    @residues.setter
    def residues(self, residues):
        """Set the Structure atoms to an Residues object"""
        if isinstance(residues, Residues):
            self._residues = residues
        else:
            self._residues = Residues(residues)

    # def store_coordinate_index_residue_map(self):
    #     self.coords_residue_index = [(residue, res_idx) for residue in self.residues for res_idx in residue.range]

    @property
    def coords_indexed_residues(self):
        """Returns a map of the Residues and Residue atom_indices for each Coord in the Structure

        Returns:
            (list[tuple[Residue, int]]): Indexed by the by the Residue position in the corresponding .coords attribute
        """
        try:
            return [(self._residues.residues[res_idx], res_atom_idx)
                    for res_idx, res_atom_idx in self._coords_residue_index[self.atom_indices].tolist()]
        except (AttributeError, TypeError):
            raise AttributeError('The current Structure object \'%s\' doesn\'t "own" it\'s coordinates. The attribute '
                                 '.coords_indexed_residues can only be accessed by the Structure object that owns these'
                                 ' coordinates and therefore owns this Structure' % self.name)  # Todo self.owner

    @coords_indexed_residues.setter
    def coords_indexed_residues(self, index_pairs):
        """Create a map of the coordinate indices to the Residue and Residue atom index"""
        self._coords_residue_index = np.array(index_pairs)

    @property
    def number_of_residues(self):
        """Returns: (int)"""
        return len(self._residue_indices)

    @property
    def center_of_mass(self):
        """Returns: (Numpy.ndarray)"""
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
        return self._coords.coords[self.get_backbone_indices()]

    def get_backbone_and_cb_coords(self):
        """Return a view of the Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [atom.index for atom in self.atoms if atom.is_backbone() or atom.is_CB()]
        return self._coords.coords[self.get_backbone_and_cb_indices()]

    def get_ca_coords(self):
        """Return a view of the Coords from the Structure with CA atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [residue.ca.index for residue in self.residues]
        # index_mask = [atom.index for atom in self.atoms if atom.is_CA()]
        return self._coords.coords[self.get_ca_indices()]

    def get_cb_coords(self):
        """Return a view of the Coords from the Structure with CB atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        # index_mask = [residue.cb.index for residue in self.residues]
        # index_mask = [atom.index for atom in self.atoms if atom.is_CB(InclGlyCA=InclGlyCA)]
        # return self._coords.coords[index_mask]
        return self._coords.coords[self.get_cb_indices()]

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

    def set_residues_attributes(self, numbers=None, **kwargs):
        """Set attributes specified by key, value pairs for all Residues in the Structure"""
        for residue in self.get_residues(numbers=numbers, **kwargs):
            for kwarg, value in kwargs.items():
                setattr(residue, kwarg, value)

    def set_residues_attributes_from_array(self, **kwargs):
        """Set attributes specified by key, value pairs for all Residues in the Structure"""
        # self._residues.set_attribute_from_array(**kwargs)
        for idx, residue in enumerate(self.residues):
            for key, value in kwargs.items():
                setattr(residue, key, value[idx])

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
            residues = set(residue for residue, atom_index in self._coords_residue_index[indices].tolist())
            return sorted(residues, key=lambda residue: residue.number)
        else:
            return self.residues
        # residue_numbers = set(atom.residue_number for atom in atoms)
        # if residue_numbers:
        #     return self.get_residues(numbers=residue_numbers)
        # else:
        #     return None

    def get_backbone_indices(self):
        """Return backbone Atom indices from the Structure

        Returns:
            (list[int])
        """
        indices = []
        for residue in self.residues:
            indices.extend(residue.backbone_indices)
        return indices

    def get_backbone_and_cb_indices(self):
        """Return backbone and CB Atom indices from the Structure. Inherently gets glycine CA's

        Returns:
            (list[int])
        """
        indices = []
        for residue in self.residues:
            indices.extend(residue.backbone_and_cb_indices)
        return indices

    def get_ca_indices(self):
        """Return CB Atom indices from the Structure

        Returns:
            (list[int])
        """
        return [residue.ca_index for residue in self.residues if residue.ca_index]

    def get_cb_indices(self):
        """Return CB Atom indices from the Structure. Inherently gets glycine Ca's and Ca's of Residues missing Cb

        Returns:
            (list[int])
        """
        return [residue.cb_index if residue.cb_index else residue.ca_index for residue in self.residues]

    def get_heavy_atom_indices(self):
        """Return Heavy Atom indices from the Structure

        Returns:
            (list[int])
        """
        indices = []
        for residue in self.residues:
            indices.extend(residue.heavy_atom_indices)
        return indices

    def get_helix_cb_indices(self):
        """Only works on secondary structure assigned structures!

        Returns:
            (list[int])
        """
        h_cb_indices = []
        for idx, residue in enumerate(self.residues):
            if not residue.secondary_structure:
                self.log.error('Secondary Structures must be set before finding helical CB\'s! Error at Residue %s'
                               % residue.number)
                return None
            elif residue.secondary_structure == 'H':
                h_cb_indices.append(residue.cb)

        return h_cb_indices

    def get_ca_atoms(self):
        """Return CA Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.get_ca_indices()]

    def get_cb_atoms(self):
        """Return CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.get_cb_indices()]

    def get_backbone_atoms(self):
        """Return backbone Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.get_backbone_indices()]

    def get_backbone_and_cb_atoms(self):
        """Return backbone and CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return self.atoms[self.get_backbone_and_cb_indices()]

    def atom(self, atom_number):
        """Retrieve the Atom specified by atom number

        Returns:
            (list[Atom])
        """
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    def renumber_structure(self):
        """Change the Atom and Residue numbering for the Structure"""
        self.renumber_atoms()
        self.renumber_residues()
        self.log.debug('%s was formatted in Pose numbering (residues now 1 to %d)'
                       % (self.name, self.number_of_residues))

    def renumber_atoms(self):
        """Renumber all atom entries one-indexed according to list order"""
        for idx, atom in enumerate(self.atoms, 1):
            atom.number = idx

    def renumber_residues(self):
        """Starts numbering Residues at 1 and number sequentially until last Residue"""
        for idx, residue in enumerate(self.residues, 1):
            residue.number = idx

    def reindex_atoms(self, start_at=0, offset=None):
        """Reindex all Atom objects to the current index in the self.atoms attribute"""
        if start_at:
            if offset:
                self.atom_indices = self.atom_indices[:start_at] + [idx - offset for idx in self.atom_indices[start_at:]]
            else:
                raise DesignError('Must include an offset when re-indexing atoms from a start_at position!')
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

    def get_residues(self, numbers=None, pdb=False, **kwargs):
        """Retrieve Residues in Structure. Returns all by default. If a list of numbers is provided, the selected
        Residues numbers are returned

        Returns:
            (list[Residue])
        """
        if numbers and isinstance(numbers, Iterable):
            if pdb:
                number_source = 'number_pdb'
            else:
                number_source = 'number'
            return [residue for residue in self.residues if getattr(residue, number_source) in numbers]
        else:
            return self.residues

    def set_residues(self, residues):
        """Set the Structure Residues to Residues object. Set the Structure Atoms and atom_indices"""
        self.residues = residues
        # self.atom_indices = [atom.index for residue in self.residues for atom in residue.atoms]
        atom_indices = []
        for residue in self.residues:
            atom_indices.extend(residue.atom_indices)
        self.atom_indices = atom_indices
        self.atoms = self.residues[0]._atoms

    def add_residues(self, residue_list):
        """Add Residue objects in a list to the Structure instance"""
        raise DesignError('This function is broken')  # TODO BROKEN
        residues = self.residues
        residues.extend(residue_list)
        self.set_residues(residues)
        # Todo need to add the residue coords to coords

    # update_structure():
    #  self.reindex_atoms() -> self.coords = np.append(self.coords, [atom.coords for atom in atoms]) ->
    #  self.set_atom_coordinates(self.coords) -> self.create_residues() -> self.set_length()

    def create_residues(self):
        """For the Structure, create all possible Residue instances. Doesn't allow for alternative atom locations"""
        new_residues = []
        residue_indices, found_types = [], []
        current_residue_number = self.atoms[0].residue_number
        # residue_idx = 0
        for idx, atom in enumerate(self.atoms):
            # if the current residue number is the same as the prior number and the atom.type is not already present
            if atom.residue_number == current_residue_number and atom.type not in found_types:
                residue_indices.append(idx)
                found_types.append(atom.type)
            else:
                new_residues.append(Residue(atom_indices=residue_indices, atoms=self._atoms, coords=self._coords))
                #                           index=residue_idx,
                # residue_idx += 1
                found_types, residue_indices = [atom.type], [idx]
                current_residue_number = atom.residue_number
        # ensure last residue is added after iteration is complete
        new_residues.append(Residue(atom_indices=residue_indices, atoms=self._atoms, coords=self._coords))
        #                           index=residue_idx,
        self.residue_indices = list(range(len(new_residues)))
        self.residues = new_residues

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

    def mutate_residue(self, residue=None, number=None, to='ALA', **kwargs):
        """Mutate specific residue to a new residue type. Type can be 1 or 3 letter format

        Keyword Args:
            residue=None (Residue): A Residue object to mutate
            number=None (int): A Residue number to select the Residue of interest by
            to='ALA' (str): The type of amino acid to mutate to
            pdb=False (bool): Whether to pull the Residue by PDB number
        Returns:
            (list[int]): The indices of the Atoms being removed from the Structure
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
        # residue.type = to.upper()
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

    def insert_residue_type(self, residue_type, at=None, chain=None):
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed!

        Args:
            residue_type (str): Either the 1 or 3 letter amino acid code for the residue in question
        Keyword Args:
            at=None (int): The pose numbered location which a new Residue should be inserted into the Structure
            chain=None (str): The chain identifier to associate the new Residue with
        """
        if not self.is_structure_owner:
            raise DesignError('This Structure \'%s\' is not the owner of it\'s attributes and therefore cannot handle '
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
        self.coords_indexed_residues = [(res_idx, res_atom_idx) for res_idx, residue in enumerate(self.residues)
                                        for res_atom_idx in residue.range]

        return new_residue

    def get_structure_sequence(self):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'

        Returns:
            (str): The amino acid sequence of the Structure Residues
        """
        return ''.join([protein_letters_3to1_extended.get(res.type.title(), '-') for res in self.residues])

    def translate(self, tx):
        new_coords = self.coords + tx
        self.replace_coords(new_coords)

    def rotate(self, rotation):
        new_coords = np.matmul(self.coords, np.transpose(rotation))
        self.replace_coords(new_coords)

    def transform(self, rotation=None, translation=None):
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))
        else:
            new_coords = self.coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)
        self.replace_coords(new_coords)

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
        self._coords.coords = new_coords
        # self.set_atoms_attributes(coords=self._coords)
        # self.reindex_atoms()
        # self.set_residues_attributes(coords=self._coords)
        # self.renumber_atoms()

    def is_clash(self, distance=2.1):
        """Check if the Structure contains any self clashes. If clashes occur with the Backbone, return True. Reports
        the Residue where the clash occurred and the clashing Atoms

        Keyword Args:
            distance=2.1 (float): The distance which clashes should be checked
        Returns:
            (bool)
        """
        # heavy_atom_indices = self.get_heavy_atom_indices()
        # all_atom_tree = BallTree(self.coords[heavy_atom_indices])  # faster 131 msec/loop
        # temp_atoms = self.atoms
        # atoms = [temp_atoms[idx] for idx in heavy_atom_indices]
        # temp_coords_indexed_residues = self.coords_indexed_residues
        # coords_indexed_residues = [temp_coords_indexed_residues[idx] for idx in heavy_atom_indices]
        all_atom_tree = BallTree(self.coords)  # faster 131 msec/loop
        atoms = self.atoms
        coords_indexed_residues = self.coords_indexed_residues
        number_residues = self.number_of_residues
        backbone_clashes, side_chain_clashes = [], []
        for prior_idx, residue in enumerate(self.residues, -1):
            residue_query = all_atom_tree.query_radius(residue.backbone_and_cb_coords, distance)
            # reduce the dimensions and format as a single array
            all_contacts = set(np.concatenate(residue_query).ravel().tolist())
            # We must subtract the N and C atoms from the adjacent residues for each residue as these are within a bond
            # For the edge cases (N- & C-term), use other termini C & N atoms.
            # We might miss a clash here! It would be peculiar for the C-terminal C clashing with the N-terminus atoms
            # and vice-versa. This also allows a PDB with permuted sequence to be handled properly!
            residue_indices_and_bonded_c_and_n = \
                residue.atom_indices + [self.residues[prior_idx].c_index, self.residues[prior_idx].o_index,
                                        self.residues[-number_residues + prior_idx + 2].n_index]
            clashes = all_contacts.difference(residue_indices_and_bonded_c_and_n)
            if any(clashes):
                for clashing_atom_idx in clashes:
                    atom = atoms[clashing_atom_idx]
                    other_residue, atom_idx = coords_indexed_residues[clashing_atom_idx]
                    if atom.is_backbone() or atom.is_CB():
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
                self.log.warning(
                    '%s contains %d side-chain clashes from the following Residues to the corresponding Atom:'
                    '\n\t%s' % (self.name, len(side_chain_clashes), sc_info))
            return False

    def get_sasa(self, probe_radius=1.4):  # , sasa_thresh=0):
        """Use FreeSASA to calculate the surface area of residues in the Structure object.
        Entities/chains could have this, but don't currently"""
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
        p = subprocess.Popen([free_sasa_exe_path, '--format=seq', '--probe-radius', str(probe_radius)],
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=self.return_atom_string().encode('utf-8'))
        if err:
            self.log.warning('\n%s' % err.decode('utf-8'))
        # self.sasa = [float(line[16:]) for line in out.decode('utf-8').split('\n') if line[:3] == 'SEQ']
        residues = self.residues
        idx = 0
        for line in out.decode('utf-8').split('\n'):
            if line[:3] == 'SEQ':
                residues[idx].sasa = float(line[16:])
                idx += 1

        self.sasa = sum([residue.sasa for residue in self.residues])
        # for line in out.decode('utf-8').split('\n'):
        #     if line[:3] == 'SEQ':
        #         self.sasa_chain.append(line[4:5])
        #         self.sasa_residues.append(int(line[5:10]))
        #         self.sasa.append(float(line[16:]))

    def get_surface_residues(self, probe_radius=2.2, sasa_thresh=0):
        """Get the residues who reside on the surface of the molecule

        Returns:
            (list[int]): The surface residue numbers
        """
        if not self.sasa:
            self.get_sasa(probe_radius=probe_radius)  # , sasa_thresh=sasa_thresh)

        # Todo make dynamic based on relative threshold seen with Levy 2010
        # return [residue.number for residue, sasa in zip(self.residues, self.sasa) if sasa > sasa_thresh]
        return [residue.number for residue in self.residues if residue.sasa > sasa_thresh]

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

    def errat(self, out_path=os.getcwd()):
        # name = 'errat_input-%s-%d.pdb' % (self.name, random() * 100000)
        # current_struc_file = self.write(out_path=os.path.join(out_path, name))
        # errat_cmd = [errat_exe_path, os.path.splitext(name)[0], out_path]  # for writing file first
        # print(subprocess.list2cmdline(errat_cmd))
        # os.system('rm %s' % current_struc_file)
        out_path = out_path if out_path[-1] == os.sep else out_path + os.sep  # errat needs trailing /
        errat_cmd = [errat_exe_path, out_path]  # for passing atoms by stdin
        # p = subprocess.Popen(errat_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # out, err = p.communicate(input=self.return_atom_string().encode('utf-8'))
        # logger.info(self.return_atom_string()[:120])
        iteration = 1
        # print('Errat:')
        all_residue_scores = []
        while iteration < 5:
            p = subprocess.run(errat_cmd, input=self.return_atom_string(), encoding='utf-8', capture_output=True)
            # print('Errat Returned: %s' % p.stdout)
            # errat_out = p.stdout
            all_residue_scores = p.stdout.split('\n')
            if len(all_residue_scores) - 1 == self.number_of_residues:  # subtract overall score
                # print('Broke from correct output')
                break
            iteration += 1
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

    def stride(self, to_file=None):
        """Use Stride to calculate the secondary structure of a PDB.

        Keyword Args
            to_file=None (str): The location of a file to save the Stride output
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
        #    Format:  6-8  Residue name
        #       10-10 Protein chain identifier
        #       12-15 PDB	residue	number
        #       17-20 Ordinal residue number
        #       25-25 One	letter secondary structure code	**)
        #       27-39 Full secondary structure name
        #       43-49 Phi	angle
        #       53-59 Psi	angle
        #       65-69 Residue solvent accessible area
        #
        #   -rId1Id2..  Read only Chains Id1, Id2 ...
        #   -cId1Id2..  Process only Chains Id1, Id2 ...
        # if chain:
        #     stride_cmd = [stride_exe_path, current_structure_file]
        #     stride_cmd.append('-c%s' % chain)

        current_struc_file = self.write(out_path='stride_input-%s-%d.pdb' % (self.name, random() * 100000))
        p = subprocess.Popen([stride_exe_path, current_struc_file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        os.system('rm %s' % current_struc_file)

        if out:
            if to_file:
                with open(to_file, 'wb') as f:
                    f.write(out)
            stride_output = out.decode('utf-8').split('\n')
        else:
            self.log.warning('%s: No secondary structure assignment found with Stride' % self.name)
            return None
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

    def parse_stride(self, stride_file, **kwargs):
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

    def is_termini_helical(self, termini='n', window=5):
        """Using assigned secondary structure, probe for a helical C-termini using a segment of 'window' residues

        Keyword Args:
            termini='n' (str): The segment size to search
            window=5 (int): The segment size to search
        Returns:
            (int): Whether the termini has a stretch of helical residues with length of the window (1) or not (0)
        """
        residues = list(reversed(self.residues)) if termini.lower() == 'c' else self.residues
        if not residues[0].secondary_structure:
            raise DesignError('You must call .get_secondary_structure on %s before querying for helical termini'
                              % self.name)
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
                return None

    def fill_secondary_structure(self, secondary_structure=None):
        if secondary_structure:
            self.secondary_structure = secondary_structure
            if len(self.secondary_structure) == self.number_of_residues:
                for idx, residue in enumerate(self.residues):
                    residue.secondary_structure = secondary_structure[idx]
            else:
                raise DesignError('The length of the passed secondary_structure (%d) is not equal to the number of '
                                  'residues (%d)' % (len(self.secondary_structure), self.number_of_residues))
        else:
            if self.residues[0].secondary_structure:
                self.secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)
            else:
                self.stride()

    def termini_proximity_from_reference(self, termini='n', reference=None):
        """From an Entity, find the orientation of the termini from the origin (default) or from a reference point

        Keyword Args:
            termini='n' (str): Either n or c terminus should be specified
            reference=None (numpy.ndarray): The reference where the point should be measured from
        Returns:
            (float): The distance from the reference point to the furthest point
        """
        if termini.lower() == 'n':
            residue_coords = self.residues[0].n_coords
        elif termini.lower() == 'c':
            residue_coords = self.residues[-1].c_coords
        else:
            raise DesignError('Termini must be either \'n\' or \'c\', not \'%s\'!' % termini)

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

    def write(self, out_path=None, file_handle=None, header=None, **kwargs):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle

        Keyword Args:
            out_path=None (str):
            file_handle=None (FileObject) #todo:
            header=None (str):
        Returns:
            (str): The name of the written file if out_path is used
        """
        def write_header(location):
            if header and isinstance(header, Iterable):
                if isinstance(header, str):
                    location.write(header)
                # else:  # TODO
                #     location.write('\n'.join(header))

        if file_handle:
            # write_header(file_handle)
            file_handle.write('%s\n' % self.return_atom_string(**kwargs))

        if out_path:
            with open(out_path, 'w') as outfile:
                write_header(outfile)
                outfile.write('%s\n' % self.return_atom_string(**kwargs))

            return out_path

    def get_fragments(self, residues=None, residue_numbers=None, representatives=None, fragment_length=5):
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Keyword Args:
            residues=None (list): The specific Residues to search for
            residue_numbers=None (list): The specific residue numbers to search for
        Returns:
            (list[MonoFragment]): The MonoFragments found on the Structure
        """
        if not residues and not residue_numbers:
            return []

        # residues = self.residues
        # ca_stretches = [[residues[idx + i].ca for i in range(-2, 3)] for idx, residue in enumerate(residues)]
        # compare ca_stretches versus monofrag ca_stretches
        # monofrag_array = repeat([ca_stretch_frag_index1, ca_stretch_frag_index2, ...]
        # monofrag_indices = filter_euler_lookup_by_zvalue(ca_stretches, monofrag_array, z_value_func=fragment_overlap,
        #                                                  max_z_value=rmsd_threshold)

        fragments = []
        for residue_number in residue_numbers:
            frag_residue_numbers = [residue_number + i for i in range(-2, 3)]  # Todo parameterize range
            ca_count = 0
            frag_residues = self.get_residues(numbers=frag_residue_numbers)
            for residue in frag_residues:
                # frag_atoms.extend(residue.get_atoms)
                if residue.ca:
                    ca_count += 1

            if ca_count == 5:
                fragment = MonoFragment(residues=frag_residues, fragment_representatives=representatives,
                                        fragment_length=fragment_length)
                if fragment.i_type:
                    fragments.append(fragment)
                # fragments.append(Structure.from_residues(frag_residues, coords=self._coords, log=None))
                # fragments.append(Structure.from_residues(deepcopy(frag_residues), log=None))

        # for structure in fragments:
        #     structure.chain_id_list = [structure.residues[0].chain]

        return fragments

    @property
    def contact_order(self):
        """Return the contact order on a per Residue basis

        Returns:
            (numpy.ndarray): The array representing the contact order for each residue in the Structure
        """
        return self.contact_order_per_residue()  # np.array([residue.contact_order for residue in self.residues])

    @contact_order.setter
    def contact_order(self, contact_order):
        """Set the contact order for each Residue

        Args:
            contact_order (Sequence)
        """
        for idx, residue in enumerate(self.residues):
            residue.contact_order = contact_order[idx]

    def contact_order_per_residue(self, sequence_distance_cutoff=2.0, distance=6.0):
        """Calculate the contact order on a per residue basis

        Keyword Args:
            sequence_distance_cutoff=2.0 (float): The residue spacing required to count a contact as a true contact
            distance=6.0 (float): The distance in angstroms to measure atomic contact distances in contact

        Returns:
            (numpy.ndarray): The array representing the contact order for each residue in the Structure
        """
        # distance of 6 angstroms between heavy atoms was used for 1998 contact order work,
        # subsequent residue wise contact order has focused on the Cb Cb heuristic of 12 A
        # I think that an atom-atom based measure is more accurate, if slightly more time
        # The BallTree creation is the biggest time cost regardless

        # Get CB Atom Coordinates including CA coordinates for Gly residues
        # indices = self.get_cb_indices()
        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        # query_coords = self.coords[indices]  # only get the coordinate indices we want
        tree = BallTree(self.coords)  # [self.get_heavy_atom_indices()])  # Todo
        # entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want
        query = tree.query_radius(self.coords, distance)  # get v residue w/ [0]
        coords_indexed_residues = self.coords_indexed_residues
        contacting_pairs = set((coords_indexed_residues[idx1][0], coords_indexed_residues[idx2][0])
                               for idx2, contacts in enumerate(query) for idx1 in contacts)
        # residues1, residues2 = split_interface_pairs(contacting_pairs)
        contact_number = len(contacting_pairs)
        for residue1, residue2 in contacting_pairs:
            # if residue1.number < residue2.number:  # only get distances for one direction
            #     continue
            # residue_distance = residue1.number - residue2.number
            residue_distance = abs(residue1.number - residue2.number)
            if residue_distance < sequence_distance_cutoff:
                continue
            residue1.contact_order += residue_distance

        number_residues = self.number_of_residues
        for residue in self.residues:
            residue.contact_order /= number_residues

        return np.array([residue.contact_order for residue in self.residues])

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
        self.set_residues_attributes(b_factor=dtype)  # , **kwargs)

    def copy_structures(self):
        """Copy all member Structures that residue in Structure containers"""
        for structure_type in self.structure_containers:
            structure = getattr(self, structure_type)
            for idx, instance in enumerate(structure):
                structure[idx] = copy(instance)

    @staticmethod
    def return_chain_generator():
        return (first + second for modification in ['upper', 'lower']
                for first in [''] + list(getattr(Structure.available_letters, modification)())
                for second in getattr(Structure.available_letters, modification)())

    def __key(self):
        return self.name, (*self._residue_indices)
        # return (self.name, *tuple(self.center_of_mass))  # , self.number_of_atoms

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        other.__dict__ = self.__dict__.copy()
        for attr, value in other.__dict__.items():
            other.__dict__[attr] = copy(value)
        other.set_residues_attributes(_coords=other._coords)  # , _atoms=other._atoms)

        return other

    def __eq__(self, other):
        if isinstance(other, Structure):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.name


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
        self.set_residues_attributes(chain=chain_id)
        # self.set_atoms_attributes(chain=chain_id)

    @property
    def sequence(self):  # Todo if the chain is mutated, this mechanism will cause errors, must clear the sequence if so
        try:
            return self._sequence
        except AttributeError:
            self._sequence = self.get_structure_sequence()
            return self._sequence

    @sequence.setter
    def sequence(self, sequence):
        self._sequence = sequence

    # @property
    # def reference_sequence(self):
    #     return self._reference_sequence
    #
    # @reference_sequence.setter
    # def reference_sequence(self, sequence):
    #     self._reference_sequence = sequence

    # def __key(self):
    #     return (self.name, self._residue_indices)

    # def __copy__(self):
    #     """Overwrite Structure.__copy__() with standard copy() method.
    #     This fails to update any attributes such as .residues or .coords, so these must be provided by another method.
    #     Theoretically, these should be updated regardless.
    #     If the Structure is owned by another Structure (Entity, PDB), the shared object will override the
    #     copy, but not implementing them removes the usability of making a copy for this Structure itself.
    #     """
    #     other = self.__class__.__new__(self.__class__)
    #     other.__dict__ = self.__dict__.copy()
    #     # for attr, value in other.__dict__.items():
    #     #     other.__dict__[attr] = copy(value)
    #
    #     return other


class Entity(Chain, SequenceProfile):
    """Entity
    Initialize with Keyword Args:
        representative=None (Chain): The Chain that should represent the Entity
        chains=None (list): A list of all Chain objects that match the Entity
        uniprot_id=None (str): The unique UniProtID for the Entity
        sequence=None (str): The sequence for the Entity
        name=None (str): The name for the Entity. Typically PDB.name is used to make a PDB compatible form
        PDB EntryID_EntityID
    """
    def __init__(self, representative=None, uniprot_id=None, **kwargs):
        #                                                                             name=None, coords=None, log=None):
        # assert isinstance(representative, Chain), 'Error: Cannot initiate a Entity without a Chain object! Pass a ' \
        #                                           'Chain object as the representative!'
        # Sets up whether Entity has full control over it's member Chain attributes
        self.is_oligomeric = False
        self.symmetry = None
        self.rotation_d = {}
        self.max_symmetry = None
        self.dihedral_chain = None
        super().__init__(residues=representative._residues, residue_indices=representative.residue_indices, **kwargs)
        self._chains = []
        self.chain_ops = []
        chains = kwargs.get('chains', list())  # [Chain objs]
        if chains and len(chains) > 1:
            self.is_oligomeric = True
            chain_ids = []
            for idx, chain in enumerate(chains):  # one of these is the representative, but we can treat it the same
                if chain.number_of_residues == self.number_of_residues:  # v this won't work if they are different len
                    _, rot, tx, _ = superposition3d(chain.get_cb_coords(), self.get_cb_coords())
                    self.chain_ops.append(dict(rotation=rot, translation=tx))
                    chain_ids.append(chain.name)
                else:
                    self.log.warning('The Chain %s passed to %s doesn\'t have the same number of residues'
                                     % (chain.name, self.name))
            self.chain_ids = chain_ids
            # self.chain_ids = [chain.name for chain in chains]
            # self.structure_containers.extend(['chains'])
        self.api_entry = None
        self.reference_sequence = kwargs.get('sequence', self.get_structure_sequence())
        # self._uniprot_id = None
        self.uniprot_id = uniprot_id

    @classmethod
    def from_representative(cls, representative=None, uniprot_id=None, **kwargs):  # chains=None,
        if isinstance(representative, Structure):
            return cls(representative=representative, uniprot_id=uniprot_id, **kwargs)  # chains=chains,
        else:
            raise DesignError('When initializing an Entity, you must pass a representative Structure object. This is '
                              'typically a Chain, but could be another collection of residues in a Structure object')

    @Structure.coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            #                         and setter is not happening because of a copy (no new info, update unimportant)
            if self.is_oligomeric:  # and not (coords.coords == self._coords.coords).all():
                # each mate chain coords are dependent on the representative (captain) coords, find the transform
                self.chain_ops.clear()
                for chain in self.chains:
                    # rmsd, rot, tx, _ = superposition3d(coords.coords[self.get_cb_indices()], self.get_cb_coords())
                    rmsd, rot, tx, _ = superposition3d(coords.coords[self.get_cb_indices()], chain.get_cb_coords())
                    self.chain_ops.append(dict(rotation=rot, translation=tx))
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
    def uniprot_id(self):
        try:
            return self._uniprot_id
        except AttributeError:
            try:
                chain_api_data = self.api_entry[next(iter(self.api_entry))]
                if chain_api_data.get('db', None) == 'UNP':
                    self._uniprot_id = chain_api_data.get('accession', None)
                return self._uniprot_id
            except AttributeError:
                self.log.warning('No uniprot_id found for Entity %s' % self.name)
                return
                # # Make a random pseudo UniProtID
                # self.uniprot_id = '%s%d' % ('R', randint(10000, 99999))

    @uniprot_id.setter
    def uniprot_id(self, uniprot_id):
        if uniprot_id:
            self._uniprot_id = uniprot_id

    @property
    def chain_id(self):
        return self.residues[0].chain

    @chain_id.setter
    def chain_id(self, chain_id):
        self.set_residues_attributes(chain=chain_id)
        # self.set_atoms_attributes(chain=chain_id)

    @property
    def chain_ids(self):
        try:
            return self._chain_ids
        except AttributeError:
            return list()

    @chain_ids.setter
    def chain_ids(self, chain_ids):
        self._chain_ids = chain_ids

    @property
    def chains(self):
        if self._chains:
            return self._chains
        else:
            self._chains = [self.return_transformed_copy(**transformation) for transformation in self.chain_ops]
            for idx, chain in enumerate(self._chains):
                # set the entity.chain_id (which sets all atoms/residues...)
                chain.chain_id = self.chain_ids[idx]
            return self._chains

    @property
    def reference_sequence(self):
        """With the default init, all Entity instances will have the structure sequence as reference_sequence"""
        return self._reference_sequence

    @reference_sequence.setter
    def reference_sequence(self, sequence):
        self._reference_sequence = sequence

    def chain(self, chain_name):
        for idx, chain_id in enumerate(self.chain_ids):
            if chain_id == chain_name:
                try:
                    return self._chains[idx]
                except IndexError:  # could make all the chains too?
                    chain = self.return_transformed_copy(**self.chain_ops[idx])
                    chain.chain_id = chain_name
                    return chain
        return None

    def retrieve_sequence_from_api(self, entity_id=None):
        if not entity_id:
            if len(self.name.split('_')) != 2:
                self.log.warning('For Entity method .%s, if an entity_id isn\'t passed and the Entity name %s is not '
                                 'the correct format (1abc_1), the query will surely fail. Ensure this is the desired '
                                 'behavior!' % (self.retrieve_sequence_from_api.__name__, self.name))
            entity_id = self.name
        self.reference_sequence = get_entity_reference_sequence(entity_id)

    def retrieve_info_from_api(self):
        """Retrieve information from the PDB API about the Entity

        Sets:
            self.api_entry (dict): {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
        """
        self.api_entry = get_pdb_info_by_entity(self.name)

    def set_up_captain_chain(self):
        raise DesignError('This function is not implemented yet')
        self.is_oligomeric = True
        for chain in self.chains:
            dum = True
            # find the center of mass for all chains
            # transform the entire group to the origin by subtracting com
            # superimpose each chain on the captain, returning the quaternion
            # using the quaternion, find the major, minor axis which should be close to integers
            # Check for bad position by non-canonical rotation integers
            # orient the chains so that they are in a canonical orientation
            # for each chain find the rotation which aligns it with its captain
            # using the inverse of the orient rotation, apply to the aligned rotation to generate a cumulative rotation
            # invert the translation of the center of mass to the origin
            # for all use of these chains in the future, ensure the found transformations are applied to each chain

    def make_oligomer(self, sym=None, rotation=None, translation=None, rotation2=None, translation2=None):
        #                   transform=None):
        """Given a symmetry and an optional transformational mapping, generate oligomeric copies of the Entity as Chains

        Assumes that the symmetric system treats the canonical symmetric axis as the Z-axis, and if the Entity is not at
        the origin, that a transformation describing it's current position relative to the origin is passed so that it
        can be moved to the origin. At the origin, makes the required oligomeric rotations, to generate an oligomer
        where symmetric copies are stored in the .chains attribute then reverses the operations back to original
        reference frame if any was provided

        Sets:
            self.chains
        """
        # if transform:
        #     translation, rotation, ext_translation, setting_rotation
        self.is_oligomeric = True
        origin = np.array([0., 0., 0.])
        if rotation is None:
            rotation = identity_matrix
        if translation is None:
            translation = origin
        if rotation2 is None:
            rotation2 = identity_matrix
        if translation2 is None:
            translation2 = origin

        self.symmetry = sym
        if 'D' in sym:  # provide a 180 degree rotation along x (all D orient symmetries have axis here)
            rotation_matrices = get_rot_matrices(RotRangeDict[sym.replace('D', 'C')], 'z', 360)
            # apparently passing the degeneracy matrix first without any specification towards the row/column major
            # worked for Josh. I am not sure that I understand his degeneracy (rotation) matrices orientation enough to
            # understand if he hardcoded the column "majorness" into situations with rot and degen np.matmul(rot, degen)
            degeneracy_rotation_matrices = get_degen_rotmatrices([flip_x_matrix], rotation_matrices)
        else:
            rotation_matrices = get_rot_matrices(RotRangeDict[sym], 'z', 360)
            degeneracy_rotation_matrices = get_degen_rotmatrices(None, rotation_matrices)
        # this is helpful for dihedral symmetry as entity must be transformed to origin to get canonical dihedral
        inv_rotation = np.linalg.inv(rotation)
        inv_setting = np.linalg.inv(rotation2)
        # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
        # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
        # and undo symmetry expansion matrices
        # centered_coords = transform_coordinate_sets(self.coords, translation=-translation2,
        # centered_coords = transform_coordinate_sets(self._coords.coords, translation=-translation2)
        cb_coords = self.get_cb_coords()
        centered_coords = transform_coordinate_sets(cb_coords, translation=-translation2)

        centered_coords_inv = transform_coordinate_sets(centered_coords, rotation=inv_setting,
                                                        translation=-translation, rotation2=inv_rotation)
        # debug_pdb = self.chain_representative.__copy__()
        # debug_pdb.replace_coords(centered_coords_inv)
        # debug_pdb.write(out_path='invert_set_invert_rot%s.pdb' % self.name)

        # set up copies to match the indices of entity
        # self.chain_representative.start_indices(dtype='atom', at=self.atom_indices[0])
        # self.chain_representative.start_indices(dtype='residue', at=self.residue_indices[0])
        # self.chains.append(self.chain_representative)
        self.chain_ops.clear()
        # for idx, rot in enumerate(degeneracy_rotation_matrices[1:], 1):  # exclude the first rotation matrix as it is identity
        for degeneracy_matrices in degeneracy_rotation_matrices:
            for rotation_matrix in degeneracy_matrices:
                rot_centered_coords = transform_coordinate_sets(centered_coords_inv, rotation=np.array(rotation_matrix))

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
                rmsd, rot, tx, _ = superposition3d(new_coords, cb_coords)
                self.chain_ops.append(dict(rotation=rot, translation=tx))
        self.chain_ids = list(self.return_chain_generator())[:len(self.chain_ops)]
        # self.log.debug('After make_oligomers, the chain_ids for %s are %s' % (self.name, self.chain_ids))

    @property
    def oligomer(self,):
        """Access the oligomeric Structure

        Returns:
            (list[Structure]): The underlying chains in the oligomer
        """
        if self.is_oligomeric:
            return self.chains
        else:
            self.log.warning('The oligomer was requested but the Entity %s is not oligomeric. Returning the Entity '
                             'instead' % self.name)
            return [self]

    def write_oligomer(self, out_path=None, file_handle=None, header=None, **kwargs):
        """Write oligomeric Structure Atoms to a file specified by out_path or with a passed file_handle

        Keyword Args:
            out_path=None (str):
            file_handle=None (FileObject) #todo:
            header=None (str):
        Returns:
            (str): The name of the written file if out_path is used
        """
        def write_header(location):
            if header and isinstance(header, Iterable):
                if isinstance(header, str):
                    location.write(header)
                # else:  # TODO
                #     location.write('\n'.join(header))
        offset = 0
        if file_handle:
            # write_header(file_handle)
            if self.chains:
                for chain in self.chains:
                    file_handle.write('%s\n' % chain.return_atom_string(atom_offset=offset, **kwargs))
                    offset += chain.number_of_atoms
            else:
                self.write(file_handle=file_handle, header=header)

        if out_path:
            if self.chains:
                with open(out_path, 'w') as outfile:
                    write_header(outfile)
                    for chain in self.chains:
                        outfile.write('%s\n' % chain.return_atom_string(atom_offset=offset, **kwargs))
                        offset += chain.number_of_atoms
            else:
                self.write(out_path=out_path, header=header)

            return out_path

    def find_chain_symmetry(self, struct_file=None):
        """Search for the chains involved in a complex using a truncated make_symmdef_file.pl script

        Keyword Args:
            struct_file=None (str): The location of the input .pdb file
        Requirements - all chains are the same length
        This script translates the PDB center of mass to the origin then uses quaternion geometry to solve for the
        rotations which superimpose chains provided by -i onto a designated chain (usually A). It returns the order of
        the rotation as well as the axis along which the rotation must take place. The axis of the rotation only needs
        to be translated to the center of mass to recapitulate the specific symmetry operation.

        perl symdesign/dependencies/rosetta/sdf/scout_symmdef_file.pl -p 1ho1_tx_4.pdb -i B C D E F G H
        >B:3-fold axis: -0.00800197 -0.01160998 0.99990058
        >C:3-fold axis: 0.00000136 -0.00000509 1.00000000

        Returns:
            (str): The name of the file written for symmetry definition file creation
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

    def scout_symmetry(self, **kwargs):
        """Check the PDB for the required symmetry parameters to generate a proper symmetry definition file"""
        struct_file = self.find_chain_symmetry(**kwargs)
        self.find_max_chain_symmetry()

        return struct_file

    def is_dihedral(self):
        """Report whether a structure is dihedral or not

        Sets:
            self.dihedral_chain (str): The name of the chain that is dihedral
        Returns:
            (bool): True if the Structure is dihedral, False if not
        """
        if not self.max_symmetry:
            self.scout_symmetry()
        # ensure if the structure is dihedral a selected dihedral_chain is orthogonal to the maximum symmetry axis
        max_symmetry_data = self.rotation_d[self.max_symmetry]
        if len(self.chain_ids) / max_symmetry_data['sym'] == 2:
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
        elif 1 < len(self.chain_ids) / max_symmetry_data['sym'] < 2:
            self.log.critical('The symmetry of %s is malformed! Highest symmetry (%d-fold) is less than 2x greater than'
                              ' the number (%d) of chains' % (self.name, max_symmetry_data['sym'], len(self.chain_ids)))

        return False

    def make_sdf(self, struct_file=None, out_path=os.getcwd(), **kwargs):
        """Use the make_symmdef_file.pl script from Rosetta to make a symmetry definition file on the Structure

        perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb.pdb -i B -q

        Keyword Args:
            struct_file=None (str): The location of the input .pdb file
            out_path=os.getcwd() (str): The location the symmetry definition file should be written
            # dihedral=False (bool): Whether the assembly is in dihedral symmetry
            # modify_sym_energy=False (bool): Whether the symmetric energy produced in the file should be modified
            # energy=2 (int): Scalar to modify the Rosetta energy by
        Returns:
            (str): Symmetry definition filename
        """
        out_file = os.path.join(out_path, '%s.sdf' % self.name)
        if os.path.exists(out_file):
            return out_file

        struct_file = self.scout_symmetry(struct_file=struct_file)
        dihedral = self.is_dihedral()
        if dihedral:  # dihedral_chain will be set
            chains = [self.max_symmetry, self.dihedral_chain]
        else:
            chains = [self.max_symmetry]

        # if not struct_file:
        #     struct_file = self.write(out_path='make_sdf_input-%s-%d.pdb' % (self.name, random() * 100000))
        sdf_cmd = ['perl', make_symmdef, '-q', '-p', struct_file, '-a', self.chain_ids[0], '-i'] + chains
        self.log.info('Creating symmetry definition file: %s' % subprocess.list2cmdline(sdf_cmd))
        # with open(out_file, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        assert p.returncode == 0, 'Symmetry definition file creation failed for %s' % self.name
        if os.path.exists(struct_file):
            os.system('rm %s' % struct_file)

        self.format_sdf(out.decode('utf-8').split('\n'), to_file=out_file, dihedral=dihedral, **kwargs)
        #               modify_sym_energy=False, energy=2)

        return out_file

    def format_sdf(self, lines, to_file=None, out_path=os.getcwd(), dihedral=False, modify_sym_energy=False, energy=2):
        """Ensure proper sdf formatting before proceeding

        Keyword Args:
            to_file=None (str): The name of the symmetry definition file
            out_path=os.getcwd() (str): The location the symmetry definition file should be written
            dihedral=False (bool): Whether the assembly is in dihedral symmetry
            modify_sym_energy=False (bool): Whether the symmetric energy produced in the file should be modified
            energy=2 (int): Scalar to modify the Rosetta energy by
        Returns:
            (str): The location the symmetry definition file was written
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
                last_jump = idx  # index of lines where the VRTs and connect_virtuals end. The "last jump"

        assert set(trunk) - set(virtuals) == set(), 'Symmetry Definition File VRTS are malformed'
        assert len(self.chain_ids) == len(subunits), 'Symmetry Definition File VRTX_base are malformed'

        if dihedral:  # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
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
            # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits
            # where complex subunits = num_subunits - 1
            # new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes subunits are read in alphanumerical order
            # new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
            lines[1] = '%s\n' % 'E = %d*%s + %s' \
                % (energy, subunits[0], ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:])))

        if not to_file:
            to_file = os.path.join(out_path, '%s.sdf' % self.name)

        with open(to_file, 'w') as f:
            f.write('%s\n' % '\n'.join(lines))
        if count != 0:
            self.log.info('Symmetry Definition File was missing %d lines, so a fix was attempted. '
                          'Modelling may be affected' % count)
        return to_file

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
        if self.is_oligomeric:
            other._chains.clear()

        return other

    # def __key(self):
    #     return (self.uniprot_id, *super().__key())  # without uniprot_id, could equal a chain...
    #     # return self.uniprot_id


class Residues:
    def __init__(self, residues):
        self.residues = np.array(residues)

    def reindex_residue_atoms(self, start_at=0):  # , offset=None):
        """Set each member Residue indices according to incremental Atoms/Coords index"""
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

    def insert(self, new_residues, at=None):
        """Insert Residue(s) into the Residues object"""
        self.residues = np.concatenate((self.residues[:at] if 0 <= at <= len(self.residues) else self.residues,
                                        new_residues if isinstance(new_residues, Iterable) else [new_residues],
                                        self.residues[at:] if at is not None else []))

    def set_attributes(self, **kwargs):
        """Set Residue attributes passed by keyword to their corresponding value"""
        for residue in self.residues.tolist():
            for key, value in kwargs.items():
                setattr(residue, key, value)

    def set_attribute_from_array(self, **kwargs):
        """For all Residues, set the Residue attribute passed by keyword to the value with the Residue index in the
        passed array

        Ex: residues.attribute_from_array(mutation_rate=residue_mutation_rate_array)
        """
        for idx, residue in enumerate(self.residues.tolist()):
            for key, value in kwargs.items():
                setattr(residue, key, value[idx])

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.residues = self.residues.copy()
        for idx, residue in enumerate(other.residues):
            other.residues[idx] = copy(residue)

        return other


class Residue:
    def __init__(self, atom_indices=None, index=None, atoms=None, coords=None):
        # self.index = index
        self.atom_indices = atom_indices
        self.atoms = atoms
        if coords:
            self.coords = coords
        self.secondary_structure = None
        self._contact_order = 0

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, index):
        self._start_index = index
        self._atom_indices = list(range(index, index + self.number_of_atoms))

    @property
    def range(self):
        return list(range(self.number_of_atoms))

    @property
    def atom_indices(self):
        """Returns: (list[int])"""
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices):
        """Set the Structure atom indices to a list of integers"""
        self._atom_indices = indices
        try:
            self._start_index = indices[0]
        except (TypeError, IndexError):
            raise IndexError('The Residue wasn\'t passed any atom_indices which are required for initialization')

    @property
    def atoms(self):
        return self._atoms.atoms[self._atom_indices].tolist()

    @atoms.setter
    def atoms(self, atoms):
        if isinstance(atoms, Atoms):
            self._atoms = atoms
        else:
            raise AttributeError('The supplied atoms are not of the class Atoms! Pass an Atoms object not a Atoms view.'
                                 ' To pass the Atoms object for a Structure, use the private attribute ._atoms')
        side_chain, heavy_atoms = [], []
        for idx, atom in enumerate(self.atoms):
            if atom.type == 'N':
                self.n = idx
                # self.n = atom.index
            elif atom.type == 'CA':
                self.ca = idx
                # self.ca = atom.index
                if atom.residue_type == 'GLY':
                    self.cb = idx
                    # self.cb = atom.index
            elif atom.type == 'CB':  # atom.is_CB(InclGlyCA=True):
                self.cb = idx
                # self.cb = atom.index
            elif atom.type == 'C':
                self.c = idx
                # self.c = atom.index
            elif atom.type == 'O':
                self.o = idx
                # self.o = atom.index
            elif atom.type == 'H':
                self.h = idx
                # self.h = atom.index
            else:
                side_chain.append(idx)
                if 'H' not in atom.type:
                    heavy_atoms.append(idx)
        self.backbone_indices = [getattr(self, index, None) for index in ['_n', '_ca', '_c', '_o']]
        self.backbone_and_cb_indices = getattr(self, '_cb', None)
        self.sidechain_indices = side_chain
        self.heavy_atom_indices = self._bb_cb_indices + heavy_atoms
        self.number_pdb = atom.pdb_residue_number
        self.number = atom.residue_number
        self.type = atom.residue_type
        self.chain = atom.chain
    # # This is the setter for all atom properties available above
    # def set_atoms_attributes(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all atoms in the Residue"""
    #     for kwarg, value in kwargs.items():
    #         for atom in self.atoms:
    #             setattr(atom, kwarg, value)

    @property
    def backbone_indices(self):
        """Returns: (list[int])"""
        return [self._atom_indices[index] for index in self._bb_indices]

    @backbone_indices.setter
    def backbone_indices(self, indices):
        """Returns: (list[int])"""
        self._bb_indices = [index for index in indices if index]

    @property
    def backbone_and_cb_indices(self):
        """Returns: (list[int])"""
        return [self._atom_indices[index] for index in self._bb_cb_indices]

    @backbone_and_cb_indices.setter
    def backbone_and_cb_indices(self, index):
        """Returns: (list[int])"""
        self._bb_cb_indices = self._bb_indices + ([index] if index else [])

    @property
    def sidechain_indices(self):
        """Returns: (list[int])"""
        return [self._atom_indices[idx] for idx in self._sc_indices]

    @sidechain_indices.setter
    def sidechain_indices(self, indices):
        """Returns: (list[int])"""
        self._sc_indices = indices

    @property
    def heavy_atom_indices(self):
        """Returns: (list[int])"""
        return [self._atom_indices[idx] for idx in self._heavy_atom_indices]

    @heavy_atom_indices.setter
    def heavy_atom_indices(self, indices):
        """Returns: (list[int])"""
        self._heavy_atom_indices = indices

    @property
    def coords(self):  # in structure too
        """The Residue atomic coords. Provides a view from the Structure that the Residue belongs too"""
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
        return self._coords.coords[self._atom_indices]

    @property
    def backbone_coords(self):
        """The backbone atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[index] for index in self._bb_indices]]

    @property
    def backbone_and_cb_coords(self):  # in structure too
        """The backbone and CB atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[index] for index in self._bb_cb_indices]]

    @property
    def sidechain_coords(self):
        """The backbone and CB atomic coords. Provides a view from the Structure that the Residue belongs too"""
        return self._coords.coords[[self._atom_indices[index] for index in self._sc_indices]]

    @coords.setter
    def coords(self, coords):  # in structure too
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords! Pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute ._coords')

    @property
    def backbone_atoms(self):
        """Returns: (list[int])"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._bb_indices]]

    @property
    def backbone_and_cb_atoms(self):
        """Returns: (list[int])"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._bb_cb_indices]]

    @property
    def sidechain_atoms(self):
        """Returns: (list[int])"""
        return self._atoms.atoms[[self._atom_indices[index] for index in self._sc_indices]]

    @property
    def n_coords(self):
        try:
            return self._coords.coords[self._atom_indices[self._n]]
        except AttributeError:
            return None

    @property
    def n(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._n]]
        except AttributeError:
            return None

    @property
    def n_index(self):
        try:
            return self._atom_indices[self._n]
        except AttributeError:
            return None

    @n.setter
    def n(self, index):
        self._n = index

    @property
    def h(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._h]]
        except AttributeError:
            return None

    @property
    def h_index(self):
        try:
            return self._atom_indices[self._h]
        except AttributeError:
            return None

    @h.setter
    def h(self, index):
        self._h = index

    @property
    def ca_coords(self):
        try:
            return self._coords.coords[self._atom_indices[self._ca]]
        except AttributeError:
            return None

    @property
    def ca_index(self):
        try:
            return self._atom_indices[self._ca]
        except AttributeError:
            return None

    @property
    def ca(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._ca]]
        except AttributeError:
            return None

    @ca.setter
    def ca(self, index):
        self._ca = index

    @property
    def cb_coords(self):
        try:
            return self._coords.coords[self._atom_indices[self._cb]]
        except AttributeError:
            return None

    @property
    def cb_index(self):
        try:
            return self._atom_indices[self._cb]
        except AttributeError:
            return None

    @property
    def cb(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._cb]]
        except AttributeError:
            return None

    @cb.setter
    def cb(self, index):
        self._cb = index

    @property
    def c_coords(self):
        try:
            return self._coords.coords[self._atom_indices[self._c]]
        except AttributeError:
            return None

    @property
    def c(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._c]]
        except AttributeError:
            return None

    @property
    def c_index(self):
        try:
            return self._atom_indices[self._c]
        except AttributeError:
            return None

    @c.setter
    def c(self, index):
        self._c = index

    @property
    def o(self):
        try:
            return self._atoms.atoms[self._atom_indices[self._o]]
        except AttributeError:
            return None

    @property
    def o_index(self):
        try:
            return self._atom_indices[self._o]
        except AttributeError:
            return None

    @o.setter
    def o(self, index):
        self._o = index

    @property
    def number(self):  # Todo remove these properties to standard attributes
        return self._number
        # try:
        #     return self.ca.residue_number
        # except AttributeError:
        #     return self.n.residue_number

    @number.setter
    def number(self, number):
        self._number = number

    @property
    def number_pdb(self):
        return self._number_pdb

    @number_pdb.setter
    def number_pdb(self, number_pdb):
        self._number_pdb = number_pdb
        # try:
        #     return self.ca.pdb_residue_number
        # except AttributeError:
        #     return self.n.pdb_residue_number

    @property
    def chain(self):
        return self._chain
        # try:
        #     return self.ca.chain
        # except AttributeError:
        #     return self.n.chain
    @chain.setter
    def chain(self, chain):
        self._chain = chain

    @property
    def type(self):
        return self._type
        # try:
        #     return self.ca.residue_type
        # except AttributeError:
        #     return self.n.chain

    @type.setter
    def type(self, _type):
        self._type = _type

    @property
    def secondary_structure(self):
        try:
            return self._secondary_structure
        except AttributeError:
            raise DesignError('This residue has no \'.secondary_structure\' attribute! Ensure you call '
                              'Structure.get_secondary_structure() on your Structure before you request Residue '
                              'specific secondary structure information')

    @secondary_structure.setter
    def secondary_structure(self, ss_code):
        self._secondary_structure = ss_code

    @property
    def sasa(self):
        try:
            return self._sasa
        except AttributeError:
            raise DesignError('Residue %d%s has no \'.sasa\' attribute! Ensure you call Structure.get_sasa() on your '
                              'Structure before you request Residue specific SASA information'
                              % (self.number, self.chain))

    @sasa.setter
    def sasa(self, sasa):
        self._sasa = sasa

    @property
    def relative_sasa(self):
        return self._sasa / gxg_sasa[self._type]

    @property
    def contact_order(self):
        return self._contact_order

    @contact_order.setter
    def contact_order(self, contact_order):
        self._contact_order = contact_order

    @property
    def number_of_atoms(self):
        return len(self._atom_indices)

    @property
    def b_factor(self):
        try:
            return sum(atom.temp_fact for atom in self.atoms) / self.number_of_atoms
        except ZeroDivisionError:
            return 0.

    @b_factor.setter
    def b_factor(self, dtype, **kwargs):
        """Set the temperature factor for every Atom in the Residue

        Keyword Args:
            dtype=None (str): The data type that should fill the temperature_factor
        """
        try:
            for atom in self.atoms:
                atom.temp_fact = getattr(self, dtype)
        except TypeError:
            raise TypeError('The b_factor must be set with a string. %s is not a string' % dtype)
        except AttributeError:
            raise AttributeError('The attribute %s was not found in the Residue. Are you sure this is the attribute you'
                                 ' want?' % dtype)

    def distance(self, other_residue):  # Todo make for Ca to Ca
        min_dist = float('inf')
        for atom in self.atoms:
            for other_atom in other_residue.atoms:
                d = atom.distance(other_atom, intra=True)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def in_contact(self, other_residue, distance_thresh=4.5, side_chain_only=False):
        if side_chain_only:
            for atom in self.atoms:
                if not atom.is_backbone():
                    for other_atom in other_residue.atoms:
                        if not other_atom.is_backbone():
                            if atom.distance(other_atom, intra=True) < distance_thresh:
                                return True
            return False
        else:
            for atom in self.atoms:
                for other_atom in other_residue.atoms:
                    if atom.distance(other_atom, intra=True) < distance_thresh:
                        return True
            return False

    def in_contact_residuelist(self, residuelist, distance_thresh=4.5, side_chain_only=False):  # UNUSED
        for residue in residuelist:
            if self.in_contact(residue, distance_thresh, side_chain_only):
                return True
        return False

    def residue_string(self):
        return format(self.type, '3s'), self.chain, format(self.number, '4d')

    def __key(self):
        return self._start_index, self.number_of_atoms, self.type  # self.ca  # Uses CA atom.

    def __eq__(self, other):
        if isinstance(other, Residue):
            return self.__key() == other.__key()
        return NotImplemented

    def __str__(self, pdb=False, chain=None, atom_offset=0, **kwargs):  # type=None, number=None, **kwargs
        residue_str = format(self.type, '3s'), (chain or self.chain), \
                      format(getattr(self, 'number%s' % ('_pdb' if pdb else '')), '4d')
        offset = 1 + atom_offset
        return '\n'.join(self._atoms.atoms[idx].__str__(**kwargs)
                         % (format(idx + offset, '5d'), *residue_str, '{:8.3f}{:8.3f}{:8.3f}'.format(*tuple(coord)))
                         for idx, coord in zip(self._atom_indices, self.coords.tolist()))

    def __hash__(self):
        return hash(self.__key())


class GhostFragment:
    def __init__(self, guide_coords, i_type, j_type, k_type, ijk_rmsd, aligned_fragment):  # structure
        #        aligned_chain_residue_tuple, guide_coords=None):
        # self.structure = structure
        # if not guide_coords:
        self.guide_coords = guide_coords
        # self.guide_coords = self.structure.chain('9').coords
        # else:
        #     self.guide_coords = guide_coords
        self.i_type = i_type
        self.j_type = j_type
        self.k_type = k_type
        self.rmsd = ijk_rmsd
        self.aligned_fragment = aligned_fragment
        # self.aligned_surf_frag_central_res_tup = aligned_chain_residue_tuple

    def get_ijk(self):
        """Return the fragments corresponding cluster index information

        Returns:
            (tuple[int, int, int]): I cluster index, J cluster index, K cluster index
        """
        return self.i_type, self.j_type, self.k_type

    def get_aligned_fragment(self):
        """
        Returns:
            (Structure): The fragment the GhostFragment instance is aligned to
        """
        return self.aligned_fragment

    def get_aligned_chain_and_residue(self):
        """Return the fragment information the GhostFragment instance is aligned to
        Returns:
            (tuple[str,int]): aligned chain, aligned residue_number"""
        return self.aligned_fragment.central_residue.chain, self.aligned_fragment.central_residue.number
        # return self.aligned_surf_frag_central_res_tup

    def get_i_type(self):
        return self.i_type

    def get_j_type(self):
        return self.j_type

    def get_k_type(self):
        return self.k_type

    def get_rmsd(self):
        return self.rmsd

    # @property
    # def structure(self):
    #     return self._structure
    #
    # @structure.setter
    # def structure(self, structure):
    #     self._structure = structure

    def get_guide_coords(self):
        return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class MonoFragment:
    def __init__(self, residues, fragment_representatives=None, fragment_type=None, guide_coords=None,
                 fragment_length=5, rmsd_thresh=0.75):
        self.i_type = fragment_type
        self.guide_coords = guide_coords
        self.central_residue = residues[int(fragment_length/2)]

        if residues and fragment_representatives:
            frag_ca_coords = np.array([residue.ca_coords for residue in residues])
            min_rmsd = float('inf')
            for cluster_type, cluster_rep in fragment_representatives.items():
                rmsd, rot, tx, rescale = superposition3d(frag_ca_coords, cluster_rep.get_ca_coords())
                if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                    self.i_type = cluster_type
                    min_rmsd, self.rotation, self.translation = rmsd, rot, tx

            if self.i_type:
                guide_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
                self.guide_coords = np.matmul(guide_coords, np.transpose(self.rotation)) + self.translation

    # @classmethod
    # def from_residue(cls):
    #     return cls()

    # @classmethod
    # def from_database(cls, residues=None, representatives=None):
    #     return cls(residues=residues, fragment_representatives=representatives)

    # @classmethod
    # def from_fragment(cls, residues=None, fragment_type=None, guide_coords=None, central_res_num=None,
    #                   central_res_chain_id=None):
    #     return cls(residues=residues, fragment_type=fragment_type, guide_coords=guide_coords,
    #                central_res_num=central_res_num, central_res_chain_id=central_res_chain_id)

    @property
    def transformation(self):
        return dict(rotation=self.rotation, translation=self.translation)

    @property
    def coords(self):  # this makes compatible with pose symmetry operations
        return self.guide_coords

    @coords.setter
    def coords(self, coords):
        self.guide_coords = coords

    def get_central_res_tup(self):
        return self.central_residue.chain, self.central_residue.number

    def get_guide_coords(self):
        return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul([0.33333, 0.33333, 0.33333], self.guide_coords)
    #     else:
    #         return None

    def get_i_type(self):
        return self.i_type

    # @property
    # def structure(self):
    #     return self._structure

    # @structure.setter
    # def structure(self, structure):
    #     self._structure = structure

    # def residue_number(self):
    #     return self.central_residue.number
    #
    # def chain(self):
    #     return self.central_residue.chain

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

    def replace_coords(self, new_coords):  # makes compatible with pose symmetry operations. Same as @coords.setter
        self.guide_coords = new_coords

    def get_ghost_fragments(self, indexed_ghost_fragments, bb_balltree, clash_dist=2.2):
        """Find all the GhostFragments associated with the MonoFragment that don't clash with the original structure
        backbone

        Args:
            indexed_ghost_fragments (dict): The paired fragment database to match to the MonoFragment instance
            bb_balltree (sklearn.neighbors.BallTree): The backbone of the structure to assign fragments to
        Keyword Args:
            clash_dist=2.2 (float): The distance to check for backbone clashes
        Returns:
            (list[GhostFragment])
        """
        if self.i_type not in indexed_ghost_fragments:
            return []

        stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array = indexed_ghost_fragments[self.i_type]
        transformed_bb_coords = transform_coordinate_sets(stacked_bb_coords, **self.transformation)
        transformed_guide_coords = transform_coordinate_sets(stacked_guide_coords, **self.transformation)
        neighbors = bb_balltree.query_radius(transformed_bb_coords.reshape(-1, 3), clash_dist)  # queries on a np.view
        neighbor_counts = np.array([neighbor.size for neighbor in neighbors])
        # reshape to original size then query for existence of any neighbors for each fragment individually
        viable_indices = neighbor_counts.reshape(transformed_bb_coords.shape[0], -1).any(axis=1)
        ghost_frag_info = \
            zip(transformed_guide_coords[~viable_indices].tolist(), *zip(*ijk_types[~viable_indices].tolist()),
                rmsd_array[~viable_indices].tolist(), repeat(self))

        return [GhostFragment(*info) for info in ghost_frag_info]


class Atoms:
    def __init__(self, atoms):
        self.atoms = np.array(atoms)

    def delete(self, indices):
        self.atoms = np.delete(self.atoms, indices)

    def insert(self, new_atoms, at=None):
        self.atoms = np.concatenate((self.atoms[:at] if 0 <= at <= len(self.atoms) else self.atoms,
                                     new_atoms if isinstance(new_atoms, Iterable) else [new_atoms],
                                     self.atoms[at:] if at is not None else []))

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.atoms = self.atoms.copy()
        # for idx, atom in enumerate(other.atoms):
        #     other.atoms[idx] = copy(atom)

        return other

    def __len__(self):
        return self.atoms.shape[0]


class Atom:
    """An Atom container with the full Structure coordinates and the Atom unique data. Pass a reference to the full
    Structure coordinates for Keyword Arg coords=self.coords"""
    def __init__(self, index=None, number=None, atom_type=None, alt_location=None, residue_type=None, chain=None,
                 residue_number=None, code_for_insertion=None, occ=None, temp_fact=None, element_symbol=None,
                 atom_charge=None, coords=None):
        self.index = index
        self.number = number
        self.type = atom_type
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain = chain
        self.pdb_residue_number = residue_number
        self.residue_number = residue_number
        self.code_for_insertion = code_for_insertion
        self.occ = occ
        self.temp_fact = temp_fact
        self.element_symbol = element_symbol
        self.atom_charge = atom_charge
        # if coords:
        #     self.coords = coords

    @classmethod
    def from_info(cls, *args):
        # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occ, temp_fact,
        # element_symbol, atom_charge
        """Initialize without coordinates"""
        return cls(*args)

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

    def is_backbone(self):
        """Check if the Atom is a backbone Atom
         Returns:
             (bool)"""
        backbone_specific_atom_type = ['N', 'CA', 'C', 'O']
        if self.type in backbone_specific_atom_type:
            return True
        else:
            return False

    def is_CB(self, InclGlyCA=True):
        if InclGlyCA:
            return self.type == 'CB' or (self.residue_type == 'GLY' and self.type == 'CA')
        else:
            #                                    When Rosetta assigns, it is this  v  but PDB assigns as this  v
            return self.type == 'CB' or (self.residue_type == 'GLY' and (self.type == '2HA' or self.type == 'HA3'))

    def is_CA(self):
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

    def get_index(self):
        return self.index

    def get_number(self):
        return self.number

    def get_type(self):
        return self.type

    def get_alt_location(self):
        return self.alt_location

    def get_residue_type(self):
        return self.residue_type

    def get_chain(self):
        return self.chain

    def get_pdb_residue_number(self):
        return self.pdb_residue_number

    def get_residue_number(self):
        return self.residue_number

    def get_code_for_insertion(self):
        return self.code_for_insertion

    # @property
    # def x(self):
    #     return self.coords[0]  # x
    #
    # @x.setter
    # def x(self, x):
    #     self._coords.coords[self.index][0] = x
    #     # self.coords[0] = x
    #
    # @property
    # def y(self):
    #     return self.coords[1]  # y
    #
    # @y.setter
    # def y(self, y):
    #     self._coords.coords[self.index][1] = y
    #     # self.coords[1] = y
    #
    # @property
    # def z(self):
    #     return self.coords[2]  # z
    #
    # @z.setter
    # def z(self, z):
    #     self._coords.coords[self.index][2] = z
    #     # self.coords[2] = z

    def get_occ(self):
        return self.occ

    def get_temp_fact(self):
        return self.temp_fact

    def get_element_symbol(self):
        return self.element_symbol

    def get_atom_charge(self):
        return self.atom_charge

    def __key(self):
        return self.number, self.type

    def __str__(self, **kwargs):  # type=None, number=None, pdb=False, chain=None,
        """Represent Atom in PDB format"""
        # this annoyingly doesn't comply with the PDB format specifications because of the atom type field
        # ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- PDB format
        # ATOM     32 CG2  VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- fstring print
        # return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'\
        return '{:6s}%s {:^4s}{:1s}%s %s%s{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'\
            .format('ATOM', self.type, self.alt_location, self.code_for_insertion, self.occ, self.temp_fact,
                    self.element_symbol, self.atom_charge)
        # ^ For future implement in residue writes
        # v old atom writes
        # return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   %s{:6.2f}{:6.2f}          {:>2s}{:2s}'\
        #        .format('ATOM', self.number, self.type, self.alt_location, self.residue_type, (chain or self.chain),
        #                getattr(self, '%sresidue_number' % ('pdb_' if pdb else '')), self.code_for_insertion,
        #                self.occ, self.temp_fact, self.element_symbol, self.atom_charge)

    def __eq__(self, other):
        return (self.number == other.number and self.chain == other.chain and self.type == other.type and
                self.residue_type == other.residue_type)

    def __hash__(self):  # Todo current key is mutable so this hash is invalid
        return hash(self.__key())


class Coords:
    def __init__(self, coords=None):
        # self.coords = np.array(coords)  # Todo simplify to this, remove properties
        if coords is not None:
            self.coords = coords
        else:
            self.coords = []

    @property
    def coords(self):
        """This holds the atomic coords which is a view from the Structure that created them"""
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = np.array(coords)

    def delete(self, indices):
        self._coords = np.delete(self._coords, indices, axis=0)

    def insert(self, new_coords, at=None):
        self._coords = \
            np.concatenate((self._coords[:at] if 0 <= at <= len(self._coords) else self._coords, new_coords,
                            self._coords[at:])
                           if at else (self._coords[:at] if 0 <= at <= len(self._coords) else self._coords, new_coords))

    def __len__(self):
        return self._coords.shape[0]


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
        fixed_coords (numpy.ndarray): The coordinates for the "frozen" object
        moving_coords (numpy.ndarray): The coordinates for the "mobile" object
    Keyword Args:
        aWeights=None (numpy.ndarray): The optional weights for the calculation of RMSD
        allow_rescale=False (bool): Attempt to rescale the mobile point cloud in addition to translation/rotation?
        report_quaternion=False (bool): Whether to report the rotation angle and axis in typical quaternion fashion
    Returns:
        (tuple[float,numpy.ndarray,numpy.ndarray,float]): rmsd, rotation_matrix/quaternion_matrix, translation_vector,
        scale_factor
    """
    # convert input lists to numpy arrays
    # fixed_coords = np.array(fixed_coords)
    # moving_coords = np.array(moving_coords)

    if fixed_coords.shape[0] != moving_coords.shape[0]:
        raise ValueError("%s: Inputs should have the same size." % superposition3d.__name__)

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
reference_aa = Structure.from_residues(residues=reference_residues, residue_indices=list(range(len(reference_residues))))
# pickle_object(ref, '/home/kylemeador/symdesign/data/AAreferenceStruct.pkl', out_path='')
