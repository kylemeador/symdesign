from copy import deepcopy, copy
from math import sqrt
from collections.abc import Iterable
from random import random

from sklearn.neighbors import KDTree
import numpy as np
from Bio.SeqUtils import IUPACData
from numpy.linalg import eigh, LinAlgError

from BioPDBUtils import biopdb_superimposer, biopdb_aligned_chain_old
from Query.PDB import get_sequence_by_entity_id, get_pdb_info_by_entry, query_entity_id, get_pdb_info_by_entity
from SequenceProfile import SequenceProfile
from SymDesignUtils import start_log, DesignError


null_log = start_log(name='null', handler=3, propagate=False)


class StructureBase:
    """Collect extra keyword arguments such as:
        chains, entities, seqres, multimodel, lazy, solve_discrepancy
    """
    def __init__(self, chains=None, entities=None, seqres=None, multimodel=None, lazy=None, solve_discrepancy=None,
                 **kwargs):
        super().__init__(**kwargs)


class Structure(StructureBase):  # (Coords):
    def __init__(self, atoms=None, residues=None, name=None, coords=None, log=None, **kwargs):
        # super().__init__(coords=coords)  # gets self.coords
        # self.atoms = []  # atoms
        # self.residues = []  # residues
        self.name = name
        self.secondary_structure = None

        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, create a new log
            self.log = start_log(name=self.name)

        if atoms is not None:
            self.atoms = atoms
            if coords is None:
                try:
                    coords = [atom.coords for atom in atoms]
                except AttributeError:
                    raise DesignError('Without passing coords, can\'t initialize Structure with Atom objects lacking '
                                      'coords! Either pass Atom objects with coords or pass coords.')
                self.reindex_atoms()
                self.coords = coords
        if residues:
            self.set_residues(residues)
            if coords is None:
                try:
                    coords = [atom.coords for residue in residues for atom in residue.atoms]
                except AttributeError:
                    raise DesignError('Without passing coords, can\'t initialize Structure with Atom objects lacking '
                                      'coords! Either pass Atom objects with coords or pass coords.')
                self.reindex_atoms()
                self.coords = coords
        if coords is not None:  # must go after Atom containers as atoms don't have any/right coordinate info
            self.coords = coords

        super().__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms=None, coords=None, **kwargs):
        return cls(atoms=atoms, coords=coords, **kwargs)

    @classmethod
    def from_residues(cls, residues=None, coords=None, **kwargs):
        return cls(residues=residues, coords=coords, **kwargs)

    @property
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
        return self._coords.coords[self.atom_indices]

    @coords.setter
    def coords(self, coords):
        """Replace the Structure, Atom, and Residue coordinates with specified Coords Object or numpy.ndarray"""
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            self._coords = Coords(coords)

        assert len(self.atoms) <= len(self.coords), '%s: ERROR number of Atoms (%d) > number of Coords (%d)!' \
                                                    % (self.name, len(self.atoms), len(self.coords))
        self.set_atoms_attributes(coords=self._coords)
        self.set_residues_attributes(coords=self._coords)

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
        """Make a deepcopy of the Structure object with the coordinates transformed in cartesian space
        Returns:
            (Structure)
        """
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))
        else:
            new_coords = self.coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(new_coords, np.transpose(rotation2))

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        new_structure = self.__copy__()
        new_structure.coords = new_coords
        return new_structure

    def replace_coords(self, new_coords):
        # if not isinstance(new_coords, Coords):
        #     new_coords = Coords(new_coords)
        self.coords = new_coords
        # self.set_atoms_attributes(coords=self._coords)
        self.reindex_atoms()  # Todo Is this poisoning the return of return_transformed_copy ?
        # self.set_residues_attributes(coords=self._coords)
        # self.renumber_atoms()

    @property
    def atom_indices(self):  # In Residue too
        """Returns: (list[int])"""
        return [atom.index for atom in self.atoms]
    #     try:
    #         return self._atom_indices
    #     except AttributeError:
    #         self.atom_indices = [atom.index for atom in self.atoms]
    #         return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices):
        self._atom_indices = np.array(indices)

    @property
    def number_of_atoms(self):
        """Returns: (int)"""
        return len(self.atoms)
    #     try:
    #         return self._number_of_atoms
    #     except AttributeError:
    #         self.set_length()
    #         return self._number_of_atoms
    #
    # @number_of_atoms.setter
    # def number_of_atoms(self, length):
    #     self._number_of_atoms = length

    @property
    def number_of_residues(self):
        """Returns: (int)"""
        return len(self.residues)
    #     try:
    #         return self._number_of_residues
    #     except AttributeError:
    #         self.set_length()
    #         return self._number_of_residues
    #
    # @number_of_residues.setter
    # def number_of_residues(self, length):
    #     self._number_of_residues = length

    @property
    def center_of_mass(self):
        """Returns: (Numpy.ndarray)"""
        divisor = 1 / self.number_of_atoms
        return np.matmul(np.full(self.number_of_atoms, divisor), self.get_coords())
        # try:
        #     return self._center_of_mass
        # except AttributeError:
        #     self.find_center_of_mass()
        #     return self._center_of_mass

    # def set_length(self):
    #     self.number_of_atoms = len(self.get_atoms())
    #     self.number_of_residues = len(self.get_residues())

    # def find_center_of_mass(self):
    #     """Retrieve the center of mass for the specified Structure"""
    #     divisor = 1 / self.number_of_atoms
    #     self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

    def get_coords(self):
        """Return a view of the Coords from the Structure

        Returns:
            (Numpy.ndarray)
        """
        return self.coords[self.atom_indices]

    def get_backbone_coords(self):
        """Return a view of the Coords from the Structure with only backbone atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.atoms if atom.is_backbone()]
        return self.coords[index_mask]

    def get_backbone_and_cb_coords(self):
        """Return a view of the Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.atoms if atom.is_backbone() or atom.is_CB()]
        return self.coords[index_mask]

    def get_ca_coords(self):
        """Return a view of the Coords from the Structure with CA atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.atoms if atom.is_CA()]
        return self.coords[index_mask]

    def get_cb_coords(self, InclGlyCA=True):
        """Return a view of the Coords from the Structure with CB atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.atoms if atom.is_CB(InclGlyCA=InclGlyCA)]
        return self.coords[index_mask]

    # def atoms(self):
    #     """Retrieve Atoms in structure. Returns all by default. If numbers=(list) selected Atom numbers are returned
    #     Returns:
    #         (list[Atom])
    #     """
    #     if numbers and isinstance(numbers, Iterable):
    #         return [atom for atom in self.atoms if atom.number in numbers]
    #     else:
    #         return self._atoms

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atom_list):
        """Set the Structure atoms to Atoms in atom_list and creates Residue objects"""
        self._atoms = np.array(atom_list, dtype=Atom)
        # self.atom_indices = list(range(len(atom_list)))  # can't set here as may contain other atoms
        self.create_residues()

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the structure instance"""
        atoms = self.atoms.tolist()
        atoms.extend(atom_list)
        self.atoms = atoms
        # Todo need to add the atoms to coords

    def set_residues_attributes(self, numbers=None, **kwargs):
        """Set attributes specified by key, value pairs for all Residues in the Structure"""
        for residue in self.get_residues(numbers=numbers, **kwargs):
            for kwarg, value in kwargs.items():
                setattr(residue, kwarg, value)
            # residue.set_atoms_attributes(**kwargs)

    def set_atoms_attributes(self, **kwargs):  # Same function in Residue
        """Set attributes specified by key, value pairs for all Atoms in the Structure"""
        for atom in self.atoms:
            for kwarg, value in kwargs.items():
                setattr(atom, kwarg, value)

    # def update_structure(self, atom_list):  # UNUSED
    #     # self.reindex_atoms()
    #     # self.coords = np.append(self.coords, [atom.coords for atom in atom_list])
    #     # self.set_atom_coordinates(self.coords)
    #     # self.create_residues()
    #     # self.set_length()

    def get_atoms_by_indices(self, indices=None):  # Todo overlap with self.atom_indices above...
        """Retrieve Atoms in the Structure specified by indices. Returns all by default

        Returns:
            (list[Atom])
        """
        return [self.atoms[index] for index in indices]

    def get_residue_atom_indices(self, numbers=None, **kwargs):
        """Retrieve Atom indices for Residues in the Structure. Returns all by default. If residue numbers are provided
         the selected Residues are returned

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.get_residue_atoms(numbers=numbers, **kwargs)]

    def get_residues_by_atom_indices(self, indices=None):
        """Retrieve Residues in the Structure specified by Atom indices.

        Returns:
            (list[Residue])
        """
        atoms = self.get_atoms_by_indices(indices)
        residue_numbers = [atom.residue_number for atom in atoms]
        if residue_numbers:
            return self.get_residues(numbers=residue_numbers)
        else:
            return None

    def get_backbone_indices(self):
        """Return backbone Atom indices from the Structure

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.atoms if atom.is_backbone()]

    def get_backbone_and_cb_indices(self):
        """Return backbone and CB Atom indices from the Structure inherently gets all glycine CA's

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.atoms if atom.is_backbone() or atom.is_CB()]

    def get_cb_indices(self, InclGlyCA=True):
        """Return CB Atom indices from the Structure. By default, inherently gets all glycine CA's

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.atoms if atom.is_CB(InclGlyCA=InclGlyCA)]

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

    # def get_CA_atoms(self):  # compatibility
    #     return self.get_ca_atoms()

    def get_ca_atoms(self):
        """Return CA Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.atoms if atom.is_CA()]

    def get_cb_atoms(self):
        """Return CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.atoms if atom.is_CB()]

    def get_backbone_atoms(self):
        """Return backbone Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.atoms if atom.is_backbone()]

    def get_backbone_and_cb_atoms(self):
        """Return backbone and CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.atoms if atom.is_backbone() or atom.is_CB()]

    def atom(self, atom_number):
        """Retrieve the Atom specified by atom number

        Returns:
            (list[Atom])
        """
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    def renumber_atoms(self):
        """Renumber all atom entries one-indexed according to list order"""
        self.log.debug('Atoms in %s were renumbered from 1 to %s' % (self.name, self.number_of_atoms))
        for idx, atom in enumerate(self.atoms):
            self.atoms[idx].number = idx + 1

    def reindex_atoms(self):
        """Reindex all Atom objects to the current index in the self.atoms attribute"""
        for idx, atom in enumerate(self.atoms):
            self.atoms[idx].index = idx

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
        if pdb:
            number_source = 'number_pdb'
        else:
            number_source = 'number'

        if numbers and isinstance(numbers, Iterable):
            return [residue for residue in self.residues if getattr(residue, number_source) in numbers]
        else:
            return self._residues

    @property
    def residues(self):
        return self._residues

    @residues.setter
    def residues(self, residues):
        self._residues = residues
        self.atom_indices = [atom.index for residue in residues for atom in residue.atoms]

    def set_residues(self, residues):
        """Set the Structure residues to Residue objects provided in a list"""
        self.atoms = residues[0]._atoms
        self.residues = residues
        # self._atoms = [atom for residue in residues for atom in residue.atoms]

    def add_residues(self, residue_list):
        """Add Residue objects in a list to the Structure instance"""
        residues = self.residues
        residues.extend(residue_list)
        self.set_residues(residues)
        # Todo need to add the residues to coords

    # update_structure():
    #  self.reindex_atoms() -> self.coords = np.append(self.coords, [atom.coords for atom in atoms]) ->
    #  self.set_atom_coordinates(self.coords) -> self.create_residues() -> self.set_length()

    def create_residues(self):
        """For the Structure, create all possible Residue instances. Doesn't allow for alternative atom locations"""
        new_residues = []
        residue_indices, found_types = [], []
        current_residue_number = self.atoms[0].residue_number
        for idx, atom in enumerate(self.atoms):
            # if the current residue number is the same as the prior number and the atom.type is not already present
            if atom.residue_number == current_residue_number and atom.type not in found_types:
                residue_indices.append(idx)
                found_types.append(atom.type)
            else:
                new_residues.append(Residue(atom_indices=residue_indices, atoms=self.atoms))  # , coords=self._coords))
                found_types, residue_indices = [atom.type], [idx]
                current_residue_number = atom.residue_number
        # ensure last residue is added after iteration is complete
        new_residues.append(Residue(atom_indices=residue_indices, atoms=self.atoms))  # , coords=self._coords))
        self.residues = new_residues

    def residue(self, residue_number):
        """Retrieve the Residue specified

        Returns:
            (Residue)
        """
        for residue in self.residues:
            if residue.number == residue_number:
                return residue
        return None

    def get_terminal_residue(self, termini='c'):
        """Retrieve the Residue from the specified termini

        Returns:
            (Residue)
        """
        if termini.lower() == 'n':
            return self.residues[0]
        elif termini.lower() == 'c':
            return self.residues[-1]
        else:
            self.log.error('%s: N or C are only allowed inputs!' % self.get_terminal_residue.__name__)
            return None

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
        return None

    def residue_number_from_pdb(self, residue_number):
        """Returns the pose residue number from the queried .pdb number

        Returns:
            (int)
        """
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue.number
        return None

    def residue_number_to_pdb(self, residue_number):
        """Returns the .pdb residue number from the queried pose number

        Returns:
            (int)
        """
        for residue in self.residues:
            if residue.number == residue_number:
                return residue.number_pdb
        return None

    def renumber_residues(self):
        """Starts numbering Residues at 1 and number sequentially until last Residue"""
        last_atom_index = len(self.atoms)
        idx = 0  # offset , 1
        for i, residue in enumerate(self.residues, 1):
            # current_res_num = self.atoms[idx].residue_number
            current_res_num = residue.number
            while self.atoms[idx].residue_number == current_res_num:
                self.atoms[idx].residue_number = i  # + offset
                idx += 1
                if idx == last_atom_index:
                    break
        # self.renumber_atoms()  # should be unnecessary

    def mutate_residue(self, residue_number, to='ALA'):
        """Mutate specific residue to a new residue type. Type can be 1 or 3 letter format"""
        if to.upper() in IUPACData.protein_letters_1to3:
            to = IUPACData.protein_letters_1to3[to.upper()]

        residue = self.residue(residue_number)
        delete = []
        for atom in residue.atoms:
            if atom.is_backbone():
                self.atoms[atom.index].residue_type = to.upper()
            else:  # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
                # delete.append(i)
                delete.append(atom.index)

        self.atoms = np.delete(self.atoms, delete)  # todo delete atoms
        # for atom in reversed(delete):
        #     self._atoms.remove(atom)
        #     residue.atoms.remove(atom)
        self.renumber_atoms()
        self.reindex_atoms()

    def get_structure_sequence(self):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'

        Returns:
            (str)
        """
        sequence_list = [residue.type for residue in self.residues]
        sequence = ''.join([IUPACData.protein_letters_3to1_extended[k.title()]
                            if k.title() in IUPACData.protein_letters_3to1_extended else '-'
                            for k in sequence_list])

        return sequence

    def is_clash(self, clash_distance=2.1):
        """Check if the Structure contains any self clashes. If clashes occur with the Backbone, return True. Reports
        the Residue where the clash occurred and the clashing Atoms

        Returns:
            (bool)
        """
        all_atom_tree = KDTree(self.coords)
        # all_atom_tree = KDTree(self.get_backbone_and_cb_coords())
        number_of_residues = self.number_of_residues
        # non_residue_indices = np.ones(self.number_of_atoms, dtype=bool)
        backbone_clashes, side_chain_clashes = [], []
        for idx, residue in enumerate(self.residues, -1):
            # return a np.array((residue length, all_atom coords)) KDTree
            # residue_query = all_atom_tree.query_radius(residue.coords, clash_distance)
            residue_query = all_atom_tree.query_radius(residue.backbone_and_cb_coords, clash_distance)
            # reduce the dimensions and format as a single array
            all_contacts = np.concatenate(residue_query).ravel()  # .reshape(-1)

            # We must subtract the N and C atoms from the adjacent residues for each residue as these are within a bond
            # For the edge cases (N- & C-term), use other termini C & N atoms.
            # We might miss a clash here! It would be peculiar for the C-terminal C clashing with the N-terminus atoms
            # and vice-versa. This also allows a PDB with permuted sequence to be handled properly!
            residue_indices_and_bonded_c_and_n = np.array(residue.atom_indices +
                                                          [self.residues[idx].c.index,
                                                           self.residues[-number_of_residues + 2 + idx].n.index])
            # non_residue_indices[residue_indices_and_bonded_c_and_n] = False
            # clashes = residue_query[residue_indices_and_bonded_c_and_n].flatten()
            # clashes = residue_query[:, non_residue_indices].flatten()
            # all_contacts = residue_query.ravel()
            # for all_clashing_indices in residue_query:
            #     atom_clash = list(set(all_clashing_indices).difference(residue_indices_and_bonded_c_and_n))
            clashes = np.setdiff1d(all_contacts, residue_indices_and_bonded_c_and_n)
            # clashes = list(set(all_contacts).difference(residue_indices_and_bonded_c_and_n))
            if any(clashes):
                # atom_clash = list(set(all_clashing_indices).difference(residue_indices_and_bonded_c_and_n))
                # if atom_clash:
                for clash in clashes:
                    if self.atoms[clash].is_backbone() or self.atoms[clash].is_CB():
                        backbone_clashes.append((residue, self.atoms[clash]))
                    else:
                        side_chain_clashes.append((residue, self.atoms[clash]))
                # backbone_clashes.extend([(residue, self.atoms[clash]) for clash in clashes if self.atoms[clash].is_backbone()
                #                     or self.atoms[clash].is_CB()])
                # raise DesignError('%s contains %d clashing atoms at Residue %d! Backbone clashes are not '
                #                   'permitted. See:\n%s'
                #                   % (self.name, len(clashes), residue.number, self.atoms[clash]))
                # self.log.critical('%s contains %d clashing atoms at Residue %d! Backbone clashes are not '
                #                   'permitted. See:\n%s'
                #                   % (self.name, len(clashes), residue.number, str(self.atoms[clash])))
                # self.log.warning('%s contains %d clashing atoms at residue %d! See:\n\t%s'
                #                  % (self.name, len(clashes), residue.number,
                #                     '\n\t'.join(str(self.atoms[clash]) for clash in clashes)))
        if side_chain_clashes:
            self.log.warning('%s contains %d side-chain clashes at the following Residues!\n\t%s'
                             % (self.name, len(backbone_clashes), '\n\t'.join('Residue %d: %s' % (residue.number, atom)
                                                                         for residue, atom in backbone_clashes)))
        if backbone_clashes:
            self.log.critical('%s contains %d backbone clashes at the following Residues!\n\t%s'
                              % (self.name, len(backbone_clashes), '\n\t'.join('Residue %d: %s' % (residue.number, atom)
                                                                          for residue, atom in backbone_clashes)))
            return True
        else:
            return False

    # def stride(self, chain=None):
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
    #     # try:
    #         # with open(os.devnull, 'w') as devnull:
    #     stride_cmd = [stride_exe_path, '%s' % self.filepath]
    #     #   -rId1Id2..  Read only chains Id1, Id2 ...
    #     #   -cId1Id2..  Process only Chains Id1, Id2 ...
    #     if chain:
    #         stride_cmd.append('-c%s' % chain_id)
    #
    #     p = subprocess.Popen(stride_cmd, stderr=subprocess.DEVNULL)
    #     out, err = p.communicate()
    #     out_lines = out.decode('utf-8').split('\n')
    #     # except:
    #     #     stride_out = None
    #
    #     # if stride_out is not None:
    #     #     lines = stride_out.split('\n')
    #
    #     for line in out_lines:
    #         if line[0:3] == 'ASG' and line[10:15].strip().isdigit():   # Todo sort out chain issues
    #             self.chain(line[9:10]).residue(int(line[10:15].strip())).secondary_structure = line[24:25]
    #     self.secondary_structure = [residue.secondary_structure for residue in self.get_residues()]
    #     # self.secondary_structure = {int(line[10:15].strip()): line[24:25] for line in out_lines
    #     #                             if line[0:3] == 'ASG' and line[10:15].strip().isdigit()}
    #
    def is_n_term_helical(self, window=5):
        """Using assigned secondary structure, probe for a helical N-termini using a sequence seqment of five residues

        Keyword Args:
            window=5 (int): The segment size to search
        Returns:
            (bool): Whether the termini has a stretch of helical residues with length of the window
        """
        if self.secondary_structure and len(self.secondary_structure) >= 2 * window:
            for idx, residue_secondary_structure in enumerate(self.secondary_structure):
                temp_window = ''.join(self.secondary_structure[idx + j] for j in range(window))
                # res_number = self.secondary_structure[0 + i:5 + i][0][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == window:
                    break
        return False  # , None

    def is_c_term_helical(self, window=5):
        """Using assigned secondary structure, probe for a helical C-termini using a sequence seqment of five residues

        Keyword Args:
            window=5 (int): The segment size to search
        Returns:
            (bool): Whether the termini has a stretch of helical residues with length of the window
        """
        if self.secondary_structure and len(self.secondary_structure) >= 2 * window:
            # for i in range(5):
            for idx, residue_secondary_structure in enumerate(reversed(self.secondary_structure)):
                # reverse_ss_asg = self.secondary_structure[::-1]
                temp_window = ''.join(self.secondary_structure[idx + j] for j in range(-window + 1, 1))
                # res_number = reverse_ss_asg[0+i:5+i][4][0]
                if 'H' * window in temp_window:
                    return True  # , res_number
                if idx == window:
                    break
        return False  # ,

    def get_secondary_structure(self):
        if self.secondary_structure:
            return self.secondary_structure
        else:
            self.fill_secondary_structure()
            if list(filter(None, self.secondary_structure)):  # check if there is at least 1 secondary struc assignment
                return self.secondary_structure
            else:
                return None

    def fill_secondary_structure(self, secondary_structure=None):
        if secondary_structure:
            self.secondary_structure = secondary_structure
        else:
            self.secondary_structure = [residue.secondary_structure for residue in self.residues]

    def write(self, out_path=None, header=None, file_handle=None):
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
        one was written"""
        def write_header(location):
            if header and isinstance(header, Iterable):
                if isinstance(header, str):
                    location.write(header)
                else:
                    location.write('\n'.join(line for line in header))

        if file_handle:
            write_header(file_handle)
            file_handle.write('\n'.join(str(atom) for atom in self.atoms))

        if out_path:
            with open(out_path, 'w') as outfile:
                write_header(outfile)
                outfile.write('\n'.join(str(atom) for atom in self.atoms))

            return out_path

    def get_fragments(self, residue_numbers=None, fragment_representatives=None, fragment_length=5):
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Keyword Args:
            residue_numbers=None (list): The specific residue numbers to search for
        """
        if not residue_numbers:
            return None

        # residues = self.residues
        # ca_stretches = [[residues[idx + i].ca for i in range(-2, 3)] for idx, residue in enumerate(residues)]
        # compare ca_stretches versus monofrag ca_stretches
        # monofrag_array = repeat([ca_stretch_frag_index1, ca_stretch_frag_index2, ...]
        # monofrag_indices = filter_euler_lookup_by_zvalue(ca_stretches, monofrag_array, z_value_func=fragment_overlap,
        #                                                  max_z_value=rmsd_threshold)

        fragments = []
        for residue_number in residue_numbers:
            frag_residue_numbers = [residue_number + i for i in range(-2, 3)]  # Todo parameterize
            ca_count = 0
            frag_residues = self.get_residues(numbers=frag_residue_numbers)
            for residue in frag_residues:
                # frag_atoms.extend(residue.get_atoms)
                if residue.ca:
                    ca_count += 1

            if ca_count == 5:
                fragment = MonoFragment(residues=frag_residues, fragment_representatives=fragment_representatives,
                                        fragment_length=fragment_length)
                if fragment.i_type:
                    fragments.append(fragment)
                # fragments.append(Structure.from_residues(frag_residues, coords=self._coords, log=None))
                # fragments.append(Structure.from_residues(deepcopy(frag_residues), log=None))

        # for structure in fragments:
        #     structure.chain_id_list = [structure.residues[0].chain]

        return fragments

    def __key(self):
        return (self.name, *tuple(self.center_of_mass))  # , self.number_of_atoms

    def __copy__(self):
        other = Structure.__new__(Structure)
        other.__dict__ = self.__dict__.copy()
        other.atoms = copy(self.atoms)
        return other

    def __eq__(self, other):
        # return self.ca == other_residue.ca
        if isinstance(other, Structure):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.name


class Chain(Structure):
    def __init__(self, **kwargs):  # name=None, residues=None, coords=None, log=None
        super().__init__(**kwargs)  # residues=residues, name=name, coords=coords, log=log

    @property
    def sequence(self):
        try:
            return self._sequence
        except AttributeError:
            self.sequence = self.get_structure_sequence()
            return self._sequence

    @sequence.setter
    def sequence(self, sequence):
        self._sequence = sequence

    # @property
    # def reference_sequence(self):
    #     return self._ref_sequence
    #
    # @reference_sequence.setter
    # def reference_sequence(self, sequence):
    #     self._ref_sequence = sequence


class Entity(Chain, SequenceProfile):  # Structure):
    """Entity
    Initialize with Keyword Args:
        representative=None (Chain): The Chain that should represent the Entity
        chains=None (list): A list of all Chain objects that match the Entity
        uniprot_id=None (str): The unique UniProtID for the Entity
        sequence=None (str): The sequence for the Entity
        name=None (str): The name for the Entity. Typically PDB.name is used to make PDB compatible form
        PDB EntryID_EntityID
    """
    def __init__(self, representative=None, chains=None, sequence=None, uniprot_id=None, **kwargs):
        #                                                                             name=None, coords=None, log=None):

        # assert isinstance(representative, Chain), 'Error: Cannot initiate a Entity without a Chain object! Pass a ' \
        #                                           'Chain object as the representative!'
        super().__init__(residues=representative.residues, structure=self, **kwargs)
        self.chain_id = representative.name
        self.chains = chains  # [Chain objs]
        self.api_entry = None
        if sequence:
            self.reference_sequence = sequence
        else:
            self.reference_sequence = self.get_structure_sequence()  # get_structure_sequence()

        if uniprot_id:
            self.uniprot_id = uniprot_id
        else:
            self.uniprot_id = '%s%d' % ('R', int(random() * 100000))  # Make a pseudo uniprot ID
        # self.representative_chain = representative_chain
        # use the self.structure __init__ from SequenceProfile for the structure identifier
        # Chain init

        # super().__init__(chain_name=representative_chain.name, residues=representative_chain.get_residues(),
        #                  coords=representative_chain.coords)
        # super().__init__(chain_name=entity_id, residues=self.chain(representative_chain).get_residues(), coords=coords)
        # SequenceProfile init
        # super().__init__(structure=self)
        # self.representative = representative  # Chain obj
        # super().__init__(structure=self.representative)  # SequenceProfile init
        # self.residues = self.chain(representative).get_residues()  # reflected above in super() call to Chain
        # self.name = entity_id  # reflected above in super() call to Chain

        # self.entity_id = entity_id

    @classmethod
    def from_representative(cls, representative=None, chains=None, uniprot_id=None, name=None, coords=None, log=None):
        return cls(representative=representative, chains=chains, uniprot_id=uniprot_id, name=name, coords=coords,
                   log=log)  # **kwargs

    @property
    def reference_sequence(self):
        return self._ref_sequence

    @reference_sequence.setter
    def reference_sequence(self, sequence):
        self._ref_sequence = sequence

    def chain(self, chain_id):  # Also in PDB
        for chain in self.chains:
            if chain.name == chain_id:
                return chain
        return None

    def retrieve_sequence_from_api(self, entity_id=None):
        if not entity_id:
            self.log.warning('For Entity method \'%s\', the entity_id must be passed!'
                             % self.retrieve_sequence_from_api.__name__)
            return None
        self.reference_sequence = get_sequence_by_entity_id(entity_id)
        # self.sequence_source = 'seqres'

    def retrieve_info_from_api(self):
        """Retrieve information from the PDB API about the Entity

        Sets:
            self.api_entry (dict): {chain: db_reference, ...}
        """
        self.api_entry = get_pdb_info_by_entity(self.name)

    # Todo set up captain chain and mate chain dependency

    def __key(self):
        return (self.uniprot_id, *super().__key())  # without uniprot_id, could equal a chain...
        # return self.uniprot_id

    # def __eq__(self, other):
    #     # return self.ca == other_residue.ca
    #     if isinstance(other, Entity):
    #         return self.__key() == other.__key()
    #     return NotImplemented

    # def __hash__(self):
    #     return hash(self.__key())


class Residue:
    def __init__(self, atom_indices=None, atoms=None, coords=None):
        # self._n = None
        # self._h = None
        # self._ca = None
        # self._cb = None
        # self._c = None
        # self._o = None
        self.atom_indices = atom_indices
        self.atoms = atoms
        if coords:
            self.coords = coords
        self.secondary_structure = None

    @property
    def atom_indices(self):  # in structure too
        return self._atom_indices

    #     return [atom.index for atom in self.atoms]
    # #     try:
    # #         return self._atom_indices
    # #     except AttributeError:
    # #         self.atom_indices = [atom.index for atom in self.atoms]
    # #         return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, indices):  # in structure too
        self._atom_indices = np.array(indices)

    @property
    def atoms(self):
        # return self._atoms
        return self._atoms[self.atom_indices]

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms
        # self.atom_indices = [atom.index for atom in atoms]
        for atom in self.atoms:
            if atom.type == 'N':
                self.n = atom.index
            elif atom.type == 'H':
                self.h = atom.index
            elif atom.type == 'CA':
                self.ca = atom.index
            elif atom.is_CB(InclGlyCA=True):
                self.cb = atom.index
            elif atom.type == 'C':
                self.c = atom.index
            elif atom.type == 'O':
                self.o = atom.index
        # Todo handle if the atom is missing backbone?

    # # This is the setter for all atom properties available above
    # def set_atoms_attributes(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all atoms in the Residue"""
    #     for kwarg, value in kwargs.items():
    #         for atom in self.atoms:
    #             setattr(atom, kwarg, value)

    @property
    def coords(self):  # in structure too
        """This holds the atomic coords which is a view from the Structure that created them"""
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
        return self._coords.coords[self.atom_indices]

    @property
    def backbone_and_cb_coords(self):  # in structure too
        """This holds the atomic coords which is a view from the Structure that created them"""
        return self._coords.coords[[atom.index for atom in [self.n, self.ca, self.cb, self.c, self.o] if atom]]
        # return self._coords.coords[np.array(bb_cb_indices)]
        # return self._coords.coords[np.array([self._n, self._ca, self._cb, self._c, self._o])]
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]

    @coords.setter
    def coords(self, coords):  # in structure too
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute _coords')

    @property
    def n(self):
        try:
            return self._atoms[self._n]
        except TypeError:
            return None

    @n.setter
    def n(self, index):
        self._n = index

    @property
    def h(self):
        try:
            return self._atoms[self._h]
        except TypeError:
            return None

    @h.setter
    def h(self, index):
        self._h = index

    @property
    def ca(self):
        try:
            return self._atoms[self._ca]
        except TypeError:
            return None

    @ca.setter
    def ca(self, index):
        self._ca = index

    @property
    def cb(self):
        try:
            return self._atoms[self._cb]
        except TypeError:
            return None

    @cb.setter
    def cb(self, index):
        self._cb = index

    @property
    def c(self):
        try:
            return self._atoms[self._c]
        except TypeError:
            return None

    @c.setter
    def c(self, index):
        self._c = index

    @property
    def o(self):
        try:
            return self._atoms[self._o]
        except TypeError:
            return None

    @o.setter
    def o(self, index):
        self._o = index

    @property
    def number(self):
        return self.ca.residue_number

    @property
    def number_pdb(self):
        return self.ca.pdb_residue_number

    @property
    def chain(self):
        return self.ca.chain

    @property
    def type(self):
        return self.ca.residue_type

    @property
    def secondary_structure(self):
        return self._secondary_structure

    @secondary_structure.setter
    def secondary_structure(self, ss_code):
        self._secondary_structure = ss_code

    @property
    def number_of_atoms(self):
        return len(self._atom_indices)

    #     try:
    #         return self._number_of_atoms
    #     except AttributeError:
    #         self.number_of_atoms = len(self.atoms)
    #         return self._number_of_atoms
    #
    # @number_of_atoms.setter
    # def number_of_atoms(self, length):
    #     self._number_of_atoms = length

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

    @staticmethod
    def get_residue(number, chain, residue_type, residuelist):  # UNUSED
        for residue in residuelist:
            if residue.number == number and residue.chain == chain and residue.type == residue_type:
                return residue
        return None

    def __key(self):
        return self.ca  # Uses CA atom. # self.number, self.chain, self.type

    def __eq__(self, other):
        # return self.ca == other_residue.ca
        if isinstance(other, Residue):
            return self.__key() == other.__key()
        return NotImplemented

    def __str__(self):
        return '\n'.join(str(atom) for atom in self.atoms)

    def __hash__(self):
        return hash(self.__key())


class GhostFragment:
    def __init__(self, structure, i_type, j_type, k_type, ijk_rmsd, aligned_fragment):
        #        aligned_chain_residue_tuple, guide_coords=None):
        self.structure = structure
        self.i_type = i_type
        self.j_type = j_type
        self.k_type = k_type
        self.rmsd = ijk_rmsd
        self.aligned_fragment = aligned_fragment
        # self.aligned_surf_frag_central_res_tup = aligned_chain_residue_tuple

        # if not guide_coords:
        self.guide_coords = self.structure.chain('9').get_coords()
        # else:
        #     self.guide_coords = guide_coords

    def get_ijk(self):
        """Return the fragments corresponding cluster index information

        Returns:
            (tuple[str, str, str]): I cluster index, J cluster index, K cluster index
        """
        return self.i_type, self.j_type, self.k_type

    def get_aligned_fragment(self):
        """Return the fragment information the GhostFragment instance is aligned to
        Returns:
            (tuple[str,int]): aligned chain, aligned residue_number"""
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

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure

    def get_guide_coords(self):
        return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class MonoFragment:
    def __init__(self, residues, fragment_representatives=None, fragment_type=None, guide_coords=None,
                 fragment_length=5, rmsd_thresh=0.75):  # central_res_num=None, central_res_chain_id=None,
        # self.structure = pdb
        self.i_type = fragment_type
        self.guide_coords = guide_coords
        self.central_residue = residues[int(fragment_length/2)]
        # self.central_res_num = central_res_num
        # self.central_res_chain_id = central_res_chain_id

        if residues and fragment_representatives:
            frag_ca_atoms = [residue.ca for residue in residues]
            # central_residue = frag_ca_atoms[2]
            # self.central_res_num = central_residue.residue_number
            # self.central_res_chain_id = central_residue.chain
            min_rmsd = float('inf')
            for cluster_type, cluster_rep in fragment_representatives.items():
                # if len(frag_ca_atoms) != len(cluster_rep.get_ca_atoms()):
                #     print('Atom list lengths are not equal! %d != %d' % (len(frag_ca_atoms),
                #                                                          len(cluster_rep.get_ca_atoms())),
                #           self.get_central_res_tup(), cluster_rep.filepath)
                #     continue
                rmsd, rot, tx = biopdb_superimposer(frag_ca_atoms, cluster_rep.get_ca_atoms())
                if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                    self.i_type = cluster_type
                    min_rmsd, self.rot, self.tx = rmsd, np.transpose(rot), tx

            if self.i_type:
                guide_coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
                # rot is returned in column major, therefore no need to transpose when transforming guide coordinates
                self.guide_coords = np.matmul(guide_coords, rot) + self.tx

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

    def get_central_res_tup(self):
        return self.central_residue.chain, self.central_residue.number

    def get_guide_coords(self):
        return self.guide_coords

    # def get_center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)
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

    def get_central_res_num(self):  # Todo rename to residue_number?
        return self.central_residue.number

    def get_central_res_chain_id(self):  # Todo rename to chain?
        return self.central_residue.chain

    def get_ghost_fragments(self, intfrag_cluster_rep, kdtree_oligomer_backbone, intfrag_cluster_info, clash_dist=2.2):
        """Find all the GhostFragments associated with the MonoFragment that don't clash with the original structure
        backbone

        Args:
            intfrag_cluster_rep (dict): The paired fragment database to match to the MonoFragment instance
            kdtree_oligomer_backbone (sklearn.neighbors.KDTree): The backbone of the structure to assign fragments to
            intfrag_cluster_info (dict): The paired fragment database info
        Keyword Args:
            clash_dist=2.2 (float): The distance to check for backbone clashes
        Returns:
            (list[GhostFragment])
        """
        if self.i_type not in intfrag_cluster_rep:
            return []

        count_check = 0  # TOdo
        ghost_fragments = []
        for j_type, j_dictionary in intfrag_cluster_rep[self.i_type].items():
            for k_type, (frag_pdb, frag_mapped_chain, frag_paired_chain) in j_dictionary.items():
                # intfrag = intfrag_cluster_rep[self.i_type][j_type][k_type]
                # frag_pdb = intfrag[0]
                # frag_paired_chain = intfrag[1]
                # # frag_mapped_chain = intfrag[1]
                # # intfrag_mapped_chain_central_res_num = intfrag[2]
                # # intfrag_partner_chain_id = intfrag[3]
                # # intfrag_partner_chain_central_res_num = intfrag[4]
                # fixed = self.structure.get_ca_atoms()
                # moving = frag_pdb.chain(frag_mapped_chain).get_ca_atoms()
                # if len(fixed) != len(moving):
                #     print('Atom list lengths are not equal! %d != %d' % (len(fixed), len(moving)),
                #           self.get_central_res_tup(), frag_pdb.filepath)
                #     continue
                # rot, tr = biopdb_align_atom_lists(fixed, moving)  # self.central_res_chain_id,
                # aligned_ghost_frag_pdb = frag_pdb.return_transformed_copy(rotation=rot, translation=tr)
                aligned_ghost_frag_pdb = frag_pdb.return_transformed_copy(rotation=self.rot, translation=self.tx)
                # is this what is not working?

                # ghost_frag_chain = (set(frag_pdb.chain_id_list) - {'9', frag_mapped_chain}).pop()
                g_frag_bb_coords = aligned_ghost_frag_pdb.chain(frag_paired_chain).get_backbone_coords()
                # Only keep ghost fragments that don't clash with oligomer backbone
                # Note: guide atoms, mapped chain atoms and non-backbone atoms not included
                cb_clash_count = kdtree_oligomer_backbone.two_point_correlation(g_frag_bb_coords, [clash_dist])

                if cb_clash_count[0] == 0:
                    rmsd = intfrag_cluster_info[self.i_type][j_type][k_type].get_rmsd()
                    ghost_fragments.append(GhostFragment(aligned_ghost_frag_pdb, self.i_type, j_type, k_type, rmsd,
                                                         self.get_central_res_tup()))
                else:  # TOdo
                    count_check += 1  # TOdo
        print('Found %d clashing fragments' % count_check)  # TOdo
        return ghost_fragments

    # def get_ghost_fragments(self, intfrag_cluster_rep_dict, kdtree_oligomer_backbone, intfrag_cluster_info_dict,
    #                         clash_dist=2.2):
    #     if self.i_type in intfrag_cluster_rep_dict:
    #
    #         count_check = 0  # TOdo
    #         ghost_fragments = []
    #         for j_type in intfrag_cluster_rep_dict[self.i_type]:
    #             for k_type in intfrag_cluster_rep_dict[self.i_type][j_type]:
    #                 intfrag = intfrag_cluster_rep_dict[self.i_type][j_type][k_type]
    #                 intfrag_pdb = intfrag[0]
    #                 intfrag_mapped_chain_id = intfrag[1]
    #                 #                                  This has been added in Structure.get_fragments  v
    #                 aligned_ghost_frag_pdb = biopdb_aligned_chain_old(self.structure, self.structure.chain_id_list[0],
    #                                                                   intfrag_pdb, intfrag_mapped_chain_id)
    #
    #                 # Only keep ghost fragments that don't clash with oligomer backbone
    #                 # Note: guide atoms, mapped chain atoms and non-backbone atoms not included
    #                 g_frag_bb_coords = []
    #                 for atom in aligned_ghost_frag_pdb.atoms:
    #                     if atom.chain != "9" and atom.chain != intfrag_mapped_chain_id and atom.is_backbone():
    #                         g_frag_bb_coords.append([atom.x, atom.y, atom.z])
    #
    #                 cb_clash_count = kdtree_oligomer_backbone.two_point_correlation(g_frag_bb_coords, [clash_dist])
    #
    #                 if cb_clash_count[0] == 0:
    #                     rmsd = intfrag_cluster_info_dict[self.i_type][j_type][k_type].get_rmsd()
    #                     ghost_fragments.append(
    #                         GhostFragment(aligned_ghost_frag_pdb, self.i_type, j_type, k_type, rmsd,
    #                                       self.get_central_res_tup()))  # ghostfrag_central_res_tup,
    #                 else:  # TOdo
    #                     count_check += 1  # TOdo
    #         print('Found %d clashing fragments' % count_check)  # TOdo
    #
    #         return ghost_fragments
    #
    #     else:
    #         return None


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
        if coords:
            self.coords = coords

    @classmethod
    def from_info(cls, *args):
        # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occ, temp_fact,
        # element_symbol, atom_charge
        """Initialize without coordinates"""
        return cls(*args)

    @property
    def coords(self):
        """This holds the atomic Coords which is a view from the Structure that created them"""
        # print(self._coords, len(self._coords.coords), self.index)
        # returns self.Coords.coords(which returns a np.array)[slicing that by the atom.index]
        return self._coords.coords[self.index]  # [self.x, self.y, self.z]

    @coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute _coords')

    def is_backbone(self):
        """Check if the Atom is a backbone Atom
         Returns:
             (bool)"""
        backbone_specific_atom_type = ['N', 'CA', 'C', 'O']
        if self.type in backbone_specific_atom_type:
            return True
        else:
            return False

    def is_CB(self, InclGlyCA=False):
        if InclGlyCA:
            return self.type == 'CB' or (self.type == 'CA' and self.residue_type == 'GLY')
        else:  # When Rosetta assigns, it is this  v  but PDB assigns as this  v
            return self.type == 'CB' or ((self.type == '2HA' or self.type == 'HA3') and self.residue_type == 'GLY')

    def is_CA(self):
        return self.type == 'CA'

    def distance(self, atom, intra=False):
        """returns distance (type float) between current instance of Atom and another instance of Atom"""
        if self.chain == atom.chain and not intra:  # todo depreciate
            # self.log.error('Atoms Are In The Same Chain')
            return None
        else:
            distance = sqrt((self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2)
            return distance

    def distance_squared(self, atom, intra=False):
        """returns squared distance (type float) between current instance of Atom and another instance of Atom"""
        if self.chain == atom.chain and not intra:
            # self.log.error('Atoms Are In The Same Chain')
            return None
        else:
            distance = (self.x - atom.x)**2 + (self.y - atom.y)**2 + (self.z - atom.z)**2
            return distance

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

    @property
    def x(self):
        return self.coords[0]  # x

    @x.setter
    def x(self, x):
        self._coords.coords[self.index][0] = x
        # self.coords[0] = x

    @property
    def y(self):
        return self.coords[1]  # y

    @y.setter
    def y(self, y):
        self._coords.coords[self.index][1] = y
        # self.coords[1] = y

    @property
    def z(self):
        return self.coords[2]  # z

    @z.setter
    def z(self, z):
        self._coords.coords[self.index][2] = z
        # self.coords[2] = z

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

    def __str__(self):
        """Represent Atom in PDB format"""
        return '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'\
               .format('ATOM', self.number, self.type, self.alt_location, self.residue_type, self.chain,
                       self.residue_number, self.code_for_insertion, self.x, self.y, self.z, self.occ,
                       self.temp_fact, self.element_symbol, self.atom_charge)

    def __eq__(self, other):
        return (self.number == other.number and self.chain == other.chain and self.type == other.type and
                self.residue_type == other.residue_type)

    def __hash__(self):  # Todo current key is mutable so this hash is invalid
        return hash(self.__key())


class Coords:
    def __init__(self, coords=None):
        if coords is not None:
            self.coords = coords
        else:
            self.coords = []
        # self.indices = None

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        return self._coords  # [self.indices]  # [self.x, self.y, self.z]

    @coords.setter
    def coords(self, coords):
        self._coords = np.array(coords)

    def get_indices(self, indicies=None):
        if indicies.any():
            return self._coords[indicies]
        else:
            return self.coords

    def __len__(self):
        return self.coords.shape[0]

    # @property
    # def x(self):
    #     return self.coords[0]  # x
    #
    # @x.setter
    # def x(self, x):
    #     self.coords[0] = x
    #
    # @property
    # def y(self):
    #     return self.coords[1]  # y
    #
    # @y.setter
    # def y(self, y):
    #     self.coords[1] = y
    #
    # @property
    # def z(self):
    #     return self.coords[2]  # z
    #
    # @z.setter
    # def z(self, z):
    #     self.coords[2] = z


def superposition3d(aa_xf_orig, aa_xm_orig, a_weights=None, allow_rescale=False, report_quaternion=False):
    """
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

    Superpose3D() takes two lists of xyz coordinates (same length), and attempts to superimpose them using rotations,
     translations, and (optionally) rescale operations in order to minimize the root-mean-squared-distance (RMSD)
     between them. These operations should be applied to the "aa_xf_orig" argument.

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    Args:
        aa_xf_orig (numpy.array): The coordinates for the "frozen" object
        aa_xm_orig (numpy.array): The coordinates for the "mobile" object
    Keyword Args:
        aWeights=None (numpy.array): The optional weights for the calculation of RMSD
        allow_rescale=False (bool): Attempt to rescale the mobile point cloud in addition to translation/rotation?
        report_quaternion=False (bool): Whether to report the rotation angle and axis in typical quaternion fashion
    Returns:
        (float, numpy.array, numpy.array, float): Corresponding to the rmsd, optimal rotation_matrix or
        quaternion_matrix (if report_quaternion=True), optimal_translation_vector, and optimal_scale_factor.
        The quaternion_matrix has the first row storing cos(θ/2) (where θ is the rotation angle). The following 3 rows
        form a vector (of length sin(θ/2)), pointing along the axis of rotation.
        Details here: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    # convert input lists as to numpy arrays

    aa_xf_orig = np.array(aa_xf_orig)
    aa_xm_orig = np.array(aa_xm_orig)

    if aa_xf_orig.shape[0] != aa_xm_orig.shape[0]:
        raise ValueError("Inputs should have the same size.")

    number_of_points = aa_xf_orig.shape[0]
    # Find the center of mass of each object:
    """ # old code (using for-loops)
    if (aWeights == None) or (len(aWeights) == 0):
        aWeights = np.full(number_of_points, 1.0)
    a_center_f = np.zeros(3)
    a_center_m = np.zeros(3)
    sum_weights = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            a_center_f[d] += aaXf_orig[n][d]*aWeights[n]
            a_center_m[d] += aaXm_orig[n][d]*aWeights[n]
        sum_weights += aWeights[n]
    """
    # new code (avoiding for-loops)
    # convert weights into array
    if not a_weights or (len(a_weights) == 0):
        a_weights = np.full((number_of_points, 1), 1.0)
    else:
        # reshape aWeights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)

    a_center_f = np.sum(aa_xf_orig * a_weights, axis=0)
    a_center_m = np.sum(aa_xm_orig * a_weights, axis=0)
    sum_weights = np.sum(a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    """ # old code (using for-loops)
    if sum_weights != 0:
        for d in range(0, 3):
            a_center_f[d] /= sum_weights
            a_center_m[d] /= sum_weights
    for n in range(0, number_of_points):
        for d in range(0, 3):
            aa_xf[n][d] = aaXf_orig[n][d] - a_center_f[d]
            aa_xm[n][d] = aaXm_orig[n][d] - a_center_m[d]
    """
    # new code (avoiding for-loops)
    if sum_weights != 0:
        a_center_f /= sum_weights
        a_center_m /= sum_weights
    aa_xf = aa_xf_orig - a_center_f
    aa_xm = aa_xm_orig - a_center_m

    # Calculate the "M" array from the Diamond paper (equation 16)
    """ # old code (using for-loops)
    M = np.zeros((3,3))
    for n in range(0, number_of_points):
        for i in range(0, 3):
            for j in range(0, 3):
                M[i][j] += aWeights[n] * aa_xm[n][i] * aa_xf[n][j]
    """
    M = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate Q (equation 17)

    """ # old code (using for-loops)
    traceM = 0.0
    for i in range(0, 3):
        traceM += M[i][i]
    Q = np.empty((3,3))
    for i in range(0, 3):
        for j in range(0, 3):
            Q[i][j] = M[i][j] + M[j][i]
            if i==j:
                Q[i][j] -= 2.0 * traceM
    """
    Q = M + M.T - 2 * np.eye(3) * np.trace(M)

    # Calculate v (equation 18)
    v = np.empty(3)
    v[0] = M[1][2] - M[2][1]
    v[1] = M[2][0] - M[0][2]
    v[2] = M[0][1] - M[1][0]

    # Calculate "P" (equation 22)
    """ # old code (using for-loops)
    P = np.empty((4,4))
    for i in range(0,3):
        for j in range(0,3):
            P[i][j] = Q[i][j]
    P[0][3] = v[0]
    P[3][0] = v[0]
    P[1][3] = v[1]
    P[3][1] = v[1]
    P[2][3] = v[2]
    P[3][2] = v[2]
    P[3][3] = 0.0
    """
    P = np.zeros((4, 4))
    P[:3, :3] = Q
    P[3, :3] = v
    P[:3, 3] = v

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
        """ # old code (using for-loops)
        eval_max = a_eigenvals[0]
        i_eval_max = 0
        for i in range(1, 4):
            if a_eigenvals[i] > eval_max:
                eval_max = a_eigenvals[i]
                i_eval_max = i
        p[0] = aa_eigenvects[0][i_eval_max]
        p[1] = aa_eigenvects[1][i_eval_max]
        p[2] = aa_eigenvects[2][i_eval_max]
        p[3] = aa_eigenvects[3][i_eval_max]
        pPp = eval_max
        """
        # new code (avoiding for-loops)
        i_eval_max = np.argmax(a_eigenvals)
        pPp = np.max(a_eigenvals)
        p[:] = aa_eigenvects[:, i_eval_max]

    # normalize the vector
    # (It should be normalized already, but just in case it is not, do it again)
    p /= np.linalg.norm(p)

    # Finally, calculate the rotation matrix corresponding to "p"
    # (p is in backwards-quaternion format)

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
    from scipy.spatial.transform import Rotation as R
    the_rotation = R.from_quat(p)
    aa_rotate = the_rotation.as_matrix()
    """

    # Optional: Decide the scale factor, c
    c = 1.0   # by default, don't rescale the coordinates
    if allow_rescale and (not singular):
        """ # old code (using for-loops)
        Waxaixai = 0.0
        WaxaiXai = 0.0
        for a in range(0, number_of_points):
            for i in range(0, 3):
                Waxaixai += aWeights[a] * aa_xm[a][i] * aa_xm[a][i]
                WaxaiXai += aWeights[a] * aa_xm[a][i] * aa_xf[a][i]
        """
        # new code (avoiding for-loops)
        Waxaixai = np.sum(a_weights * aa_xm ** 2)
        WaxaiXai = np.sum(a_weights * aa_xf ** 2)

        c = (WaxaiXai + pPp) / Waxaixai

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper

    """ # old code (using for-loops)
    E0 = 0.0
    for n in range(0, number_of_points):
        for d in range(0, 3):
            # (remember to include the scale factor "c" that we inserted)
            E0 += aWeights[n] * ((aa_xf[n][d] - c*aa_xm[n][d])**2)
    sum_sqr_dist = E0 - c*2.0*pPp
    if sum_sqr_dist < 0.0: #(edge case due to rounding error)
        sum_sqr_dist = 0.0
    """
    # new code (avoiding for-loops)
    E0 = np.sum((aa_xf - c * aa_xm) ** 2)
    sum_sqr_dist = max(0, E0 - c * 2.0 * pPp)

    rmsd = 0.0
    if sum_weights != 0.0:
        rmsd = np.sqrt(sum_sqr_dist/sum_weights)

    # Lastly, calculate the translational offset:
    # Recall that:
    #RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = a_center_f[]       and       a_center_m[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    """ # old code (using for-loops)
    a_translate = np.empty(3)
    for i in range(0,3):
        a_translate[i] = a_center_f[i]
        for j in range(0,3):
            a_translate[i] -= c*aa_rotate[i][j]*a_center_m[j]
    """
    # new code (avoiding for-loops)
    a_translate = a_center_f - np.matmul(c * aa_rotate, a_center_m).T.reshape(3,)

    if report_quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        q = np.empty(4)
        q[0] = p[3]
        q[1] = p[0]
        q[2] = p[1]
        q[3] = p[2]
        return rmsd, q, a_translate, c
    else:
        return rmsd, aa_rotate, a_translate, c
