from copy import deepcopy
from math import sqrt
from collections.abc import Iterable
from random import random

from sklearn.neighbors import KDTree
import numpy as np
from Bio.SeqUtils import IUPACData
from numpy.linalg import eigh, LinAlgError

from Query.PDB import get_sequence_by_entity_id, get_pdb_info_by_entry, query_entity_id, get_pdb_info_by_entity
from SequenceProfile import SequenceProfile
from SymDesignUtils import start_log, DesignError


class Structure:  # (Coords):
    def __init__(self, atoms=None, residues=None, name=None, coords=None, log=None, **kwargs):
        # self.coords = coords
        # super().__init__(coords=coords)  # gets self.coords
        self.atoms = []  # atoms
        self.residues = []  # residues
        # self.id = None
        self.name = name
        self.secondary_structure = None
        # self.center_of_mass = None
        # self.sequence = None

        if log:
            self.log = log
        else:
            # print('Structure starting log')  # Todo when Structure is base class?
            # self.log = start_log()
            dummy = True

        if atoms:
            self.set_atoms(atoms)
        if residues:  # Todo, the structure can not have Coords! if from_atoms or from_residues lacks them
            self.set_residues(residues)
        # if isinstance(coords, np.ndarray) and coords.any():
        if coords:  # and isinstance(coords, Coords):
            self.coords = coords

        super().__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls(atoms=atoms, **kwargs)

    @classmethod
    def from_residues(cls, residues, **kwargs):
        return cls(residues=residues, **kwargs)

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
        return self._coords.coords  # [self.atom_indices]
        # return self._coords.get_indices(self.atom_indices)

    @coords.setter
    def coords(self, coords):
        # assert len(self.atoms) == coords.shape[0], '%s: ERROR number of Atoms (%d) != number of Coords (%d)!' \
        #                                                 % (self.name, len(self.atoms), self.coords.shape[0])
        if isinstance(coords, Coords):
            self._coords = coords
            self.set_atoms_attributes(coords=self._coords)
            # for atom in self.atoms:
            #     atom.coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def return_transformed_copy(self, rotation=None, translation=None):
        """Make a deepcopy of the Structure object with the coordinates transformed in cartesian space
        Returns:
            (Structure)
        """
        new_coords = np.array(self.extract_coords())
        if rotation:
            new_coords = np.matmul(new_coords, np.transpose(np.array(rotation)))
        if translation:
            new_coords += np.array(translation)

        new_structure = deepcopy(self)
        new_structure.replace_coords(Coords(new_coords))
        return new_structure

    def replace_coords(self, new_coords):
        if not isinstance(new_coords, Coords):
            new_coords = Coords(new_coords)
        self.coords = new_coords
        self.set_atoms_attributes(coords=self._coords)
        self.reindex_atoms()
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
    #
    # @atom_indices.setter
    # def atom_indices(self, indices):
    #     self._atom_indices = np.array(indices)

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
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone()]
        # return self._coords[index_mask]
        return self.coords[index_mask]

    def get_backbone_and_cb_coords(self):
        """Return a view of the Coords from the Structure with backbone and CB atom coordinates
        inherently gets all glycine CA's

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]
        # return self._coords[index_mask]
        return self.coords[index_mask]

    def get_ca_coords(self):
        """Return a view of the Coords from the Structure with CA atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CA()]
        # return self._coords[index_mask]
        return self.coords[index_mask]

    def get_cb_coords(self, InclGlyCA=True):
        """Return a view of the Coords from the Structure with CB atom coordinates

        Returns:
            (Numpy.ndarray)
        """
        index_mask = [atom.index for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]
        # return self._coords[index_mask]
        return self.coords[index_mask]

    def extract_all_coords(self):  # compatibility
        return self.extract_coords()

    def extract_coords(self):  # compatibility
        """Grab all the coordinates from the Structure's Coords, returns a list with views of the Coords array

        Returns:
            (list[Numpy.ndarray])
        """
        return [atom.coords for atom in self.get_atoms()]

    def extract_backbone_coords(self):  # compatibility
        return [atom.coords for atom in self.get_atoms() if atom.is_backbone()]

    def extract_backbone_and_cb_coords(self):  # compatibility
        # inherently gets all glycine CA's
        return [atom.coords for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]

    def extract_CA_coords(self):  # compatibility
        return [atom.coords for atom in self.get_atoms() if atom.is_CA()]

    def extract_CB_coords(self, InclGlyCA=False):  # compatibility
        return [atom.coords for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]

    # @property Todo
    def get_atoms(self, numbers=None):
        """Retrieve Atoms in structure. Returns all by default. If numbers=(list) selected Atom numbers are returned

        Returns:
            (list[Atom])
        """
        if numbers and isinstance(numbers, Iterable):
            return [atom for atom in self.atoms if atom.number in numbers]
        else:
            return self.atoms

    def set_atoms(self, atom_list):
        """Set the Structure atoms to Atoms in atom_list"""
        self.atoms = atom_list
        # self.renumber_atoms()
        # self.reindex_atoms()
        self.create_residues()
        # self.update_structure(atom_list)
        # self.set_length()

    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the structure instance"""
        self.atoms.extend(atom_list)
        # self.renumber_atoms()  # Todo this logic can't hold if the structure contains atoms in another structure!
        # self.reindex_atoms()
        self.create_residues()
        # self.update_structure(atom_list)
        # self.set_length()

    def set_residues_attributes(self, numbers=None, **kwargs):
        """Set attributes specified by key, value pairs for all Residues in the Structure"""
        # for kwarg, value in kwargs.items():
        for residue in self.get_residues(numbers=numbers):
            # setattr(residue, kwarg, value)
            residue.set_atoms_attributes(**kwargs)

    def set_atoms_attributes(self, numbers=None, **kwargs):  # Same function as in Residue
        """Set attributes specified by key, value pairs for all atoms in the Structure"""
        for kwarg, value in kwargs.items():
            for atom in self.get_atoms(numbers=numbers):
                setattr(atom, kwarg, value)

    # def update_structure(self, atom_list):  # UNUSED
    #     # self.reindex_atoms()
    #     # self.coords = np.append(self.coords, [atom.coords for atom in atom_list])
    #     # self.set_atom_coordinates(self.coords)
    #     # self.create_residues()
    #     # self.set_length()

    def get_atom_indices(self, numbers=None):
        """Retrieve Atom indices for Atoms in the Structure. Returns all by default. If atom numbers are provided
         the selected Atoms are returned

        Returns:
            (list[int])
        """
        # if numbers and isinstance(numbers, Iterable):
        return [atom.index for atom in self.get_atoms(numbers=numbers)]
        # else:
        #     return [atom.index for atom in self.get_atoms()]

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
        return [atom.index for atom in self.get_atoms() if atom.is_backbone()]

    def get_backbone_and_cb_indices(self):
        """Return backbone and CB Atom indices from the Structure inherently gets all glycine CA's

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]

    def get_cb_indices(self, InclGlyCA=True):
        """Return CB Atom indices from the Structure. By default, inherently gets all glycine CA's

        Returns:
            (list[int])
        """
        return [atom.index for atom in self.get_atoms() if atom.is_CB(InclGlyCA=InclGlyCA)]

    def get_helix_cb_indices(self):
        """Only works on secondary structure assigned structures!

        Returns:
            (list[int])
        """
        h_cb_indices = []
        for idx, residue in enumerate(self.get_residues()):
            if not residue.get_secondary_structure():
                self.log.error('Secondary Structures must be set before finding helical CB\'s! Error at Residue %s'
                               % residue.number)
                return None
            elif residue.get_secondary_structure() == 'H':
                h_cb_indices.append(residue.get_cb())

        return h_cb_indices

    def get_CA_atoms(self):  # compatibility
        return self.get_ca_atoms()

    def get_ca_atoms(self):
        """Return CA Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.get_atoms() if atom.is_CA()]

    def get_cb_atoms(self):
        """Return CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.get_atoms() if atom.is_CB()]

    def get_backbone_atoms(self):
        """Return backbone Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.get_atoms() if atom.is_backbone()]

    def get_backbone_and_cb_atoms(self):
        """Return backbone and CB Atoms from the Structure

        Returns:
            (list[Atom])
        """
        return [atom for atom in self.get_atoms() if atom.is_backbone() or atom.is_CB()]

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
        for idx, atom in enumerate(self.atoms, 1):
            atom.number = idx

    def reindex_atoms(self):  # Unused
        for idx, atom in enumerate(self.atoms):
            atom.index = idx

    # def set_atom_coordinates(self, coords):
    #     """Set/Replace all Atom coordinates with coords specified. Must be in the same order to apply correctly!"""
    #     assert len(self.atoms) == coords.shape[0], '%s: ERROR setting Atom coordinates, # Atoms (%d) !=  # Coords (%d)'\
    #                                                % (self.name, len(self.atoms), coords.shape[0])
    #     self.coords = coords
    #     for idx, atom in enumerate(self.get_atoms()):
    #         atom.coords = coords[idx]
    #         # atom.x, atom.y, atom.z = coords[idx][0], coords[idx][1], coords[idx][2]

    # @property Todo
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
            return self.residues

    def set_residues(self, residue_list):
        """Set the Structure residues to Residue objects provided in a list"""
        self.residues = residue_list  # []
        self.atoms = [atom for residue in residue_list for atom in residue.get_atoms()]
        # self.update_structure(atom_list)
        # self.set_length()

    def add_residues(self, residue_list):
        """Add Residue objects in a list to the Structure instance"""
        self.residues.extend(residue_list)
        atom_list = [atom for atom in residue_list.get_atoms()]
        self.atoms.extend(atom_list)
        # self.update_structure(atom_list)
        # self.set_length()

    # update_structure():
    #  self.reindex_atoms() -> self.coords = np.append(self.coords, [atom.coords for atom in atoms]) ->
    #  self.set_atom_coordinates(self.coords) -> self.create_residues() -> self.set_length()

    def create_residues(self):
        """For the Structure, create all possible Residue instances"""
        current_residue_number = self.atoms[0].residue_number
        current_residue = []
        for atom in self.atoms:
            if atom.residue_number == current_residue_number:
                current_residue.append(atom)
            else:
                self.residues.append(Residue(atoms=current_residue, coords=self._coords))
                current_residue = [atom]
                current_residue_number = atom.residue_number
        # ensure last residue is added after iteration is complete
        self.residues.append(Residue(atoms=current_residue, coords=self._coords))

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
            atoms.extend(residue.get_atoms())
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
        for atom in residue.get_atoms():
            if atom.is_backbone():
                atom.residue_type = to.upper()  # should be fine? Atom is an Atom object reference by others
            else:  # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
                # delete.append(i)
                delete.append(atom)

        for atom in reversed(delete):
            self.atoms.remove(atom)
            residue.atoms.remove(atom)
        self.renumber_atoms()

    def get_structure_sequence(self):
        """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'

        Returns:
            (str)
        """
        sequence_list = [residue.type for residue in self.get_residues()]
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
        all_clashes, side_chain_clashes = [], []
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
                        all_clashes.append((residue, self.atoms[clash]))
                    else:
                        side_chain_clashes.append((residue, self.atoms[clash]))
                # all_clashes.extend([(residue, self.atoms[clash]) for clash in clashes if self.atoms[clash].is_backbone()
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
                             % (self.name, len(all_clashes), '\n\t'.join('Residue %d: %s' % (residue.number, atom)
                                                                         for residue, atom in all_clashes)))
        if all_clashes:
            self.log.critical('%s contains %d backbone clashes at the following Residues!\n\t%s'
                              % (self.name, len(all_clashes), '\n\t'.join('Residue %d: %s' % (residue.number, atom)
                                                                          for residue, atom in all_clashes)))
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
    #             self.chain(line[9:10]).residue(int(line[10:15].strip())).set_secondary_structure(line[24:25])
    #     self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]
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
            self.secondary_structure = [residue.get_secondary_structure() for residue in self.get_residues()]

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
            file_handle.write('\n'.join(str(atom) for atom in self.get_atoms()))

        if out_path:
            with open(out_path, 'w') as outfile:
                write_header(outfile)
                outfile.write('\n'.join(str(atom) for atom in self.get_atoms()))

            return out_path

    def get_fragments(self, residue_numbers=None, fragment_length=5):
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Keyword Args:
            residue_numbers=None (list): The specific residue numbers to search for
        """
        if not residue_numbers:
            return None

        # residues = self.get_residues()
        # ca_stretches = [[residues[idx + i].ca for i in range(-2, 3)] for idx, residue in enumerate(residues)]
        # compare ca_stretches versus monofrag ca_stretches
        # monofrag_array = repeat([ca_stretch_frag_index1, ca_stretch_frag_index2, ...]
        # monofrag_indices = filter_euler_lookup_by_zvalue(ca_stretches, monofrag_array, z_value_func=fragment_overlap,
        #                                                  max_z_value=rmsd_threshold)

        interface_frags = []
        for residue_number in residue_numbers:
            frag_residue_numbers = [residue_number + i for i in range(-2, 3)]  # Todo parameterize
            ca_count = 0
            for residue in self.get_residues(frag_residue_numbers):
                # frag_atoms.extend(residue.get_atoms())
                if residue.get_ca():
                    ca_count += 1
            # todo reduce duplicate calculation
            if ca_count == 5:
                interface_frags.append(Structure.from_residues(self.get_residues(frag_residue_numbers), log=False))

        for structure in interface_frags:
            structure.chain_id_list = [structure.get_residues()[0].chain]

        return interface_frags

    def __key(self):
        return (self.name, *tuple(self.center_of_mass))  # , self.number_of_atoms

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
    def __init__(self, **kwargs):  # name=None, residues=None, coords=None, log=None, sequence=None,
        super().__init__(**kwargs)  # residues=residues, name=name, coords=coords, log=log,
        # self.residues = residues
        # self.id = name
        # if sequence:
        #     self.reference_sequence = sequence
        # else:
        #     self.reference_sequence = self.sequence  # get_structure_sequence()

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
        name=None (str): The name for the Entity. Typically PDB.name is used to make PDB compatible form
        PDB EntryID_EntityID
        uniprot_id=None (str): The unique UniProtID for the Entity
    """
    def __init__(self, representative=None, chains=None, sequence=None, uniprot_id=None, **kwargs):
        #        name=None, coords=None, log=None):
        assert isinstance(representative, Chain), 'Error: Cannot initiate a Entity without a Chain object! Pass a ' \
                                                  'Chain object as the representative!'
        super().__init__(residues=representative.get_residues(), structure=self, **kwargs)  # name=name, coords=coords,
        #                                                                                     log=log,
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
    def from_representative(cls, representative=None, chains=None, name=None, uniprot_id=None, coords=None, log=None):
        return cls(representative=representative, chains=chains, name=name, uniprot_id=uniprot_id, coords=coords,
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
    def __init__(self, atoms=None, coords=None):
        self.atoms = atoms
        self.secondary_structure = None
        self._n = None
        self._ca = None
        self._cb = None
        self._c = None
        self._o = None
        self.set_atoms()
        if coords:
            self.coords = coords

    @property
    def n(self):
        try:
            return self.atoms[self._n]
        except TypeError:
            return None

    @n.setter
    def n(self, index):
        self._n = index

    @property
    def ca(self):
        try:
            return self.atoms[self._ca]
        except TypeError:
            return None

    @ca.setter
    def ca(self, index):
        self._ca = index

    @property
    def cb(self):
        try:
            return self.atoms[self._cb]
        except TypeError:
            return None

    @cb.setter
    def cb(self, index):
        self._cb = index

    @property
    def c(self):
        try:
            return self.atoms[self._c]
        except TypeError:
            return None

    @c.setter
    def c(self, index):
        self._c = index

    @property
    def o(self):
        try:
            return self.atoms[self._o]
        except TypeError:
            return None

    @o.setter
    def o(self, index):
        self._o = index

    @property
    def number(self):
        return self.ca.get_residue_number()

    @property
    def number_pdb(self):
        return self.ca.get_pdb_residue_number()

    @property
    def chain(self):
        return self.ca.get_chain()

    @property
    def type(self):
        return self.ca.get_residue_type()

    def set_atoms(self):
        # self.atoms = atoms
        for idx, atom in enumerate(self.atoms):
            if atom.is_n():
                self.n = idx
            elif atom.is_CA():
                self.ca = idx
            elif atom.is_CB(InclGlyCA=True):
                self.cb = idx
            elif atom.is_c():
                self.c = idx
            elif atom.is_o():
                self.o = idx

    # This is the setter for all atom properties available above
    def set_atoms_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for all atoms in the Residue"""
        for kwarg, value in kwargs.items():
            for atom in self.atoms:
                setattr(atom, kwarg, value)

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        return self._coords.coords[self.atom_indices]
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]

    @property
    def backbone_and_cb_coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        # bb_cb_indices = [atom.index for atom in [self.n, self.ca, self.cb, self.c, self.o] if atom]
        # print(bb_cb_indices)
        return self._coords.coords[[atom.index for atom in [self.n, self.ca, self.cb, self.c, self.o] if atom]]
        # return self._coords.coords[np.array(bb_cb_indices)]
        # return self._coords.coords[np.array([self._n, self._ca, self._cb, self._c, self._o])]
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]

    @coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    @property
    def atom_indices(self):  # in structure too
        return [atom.index for atom in self.atoms]
    #     try:
    #         return self._atom_indices
    #     except AttributeError:
    #         self.atom_indices = [atom.index for atom in self.atoms]
    #         return self._atom_indices
    #
    # @atom_indices.setter
    # def atom_indices(self, indices):
    #     self._atom_indices = np.array(indices)

    @property
    def number_of_atoms(self):
        return len(self.get_atoms())
    #     try:
    #         return self._number_of_atoms
    #     except AttributeError:
    #         self.number_of_atoms = len(self.get_atoms())
    #         return self._number_of_atoms
    #
    # @number_of_atoms.setter
    # def number_of_atoms(self, length):
    #     self._number_of_atoms = length

    # @property  # todo
    def get_atoms(self):
        return self.atoms

    def get_secondary_structure(self):
        return self.secondary_structure

    def set_secondary_structure(self, ss_code):
        self.secondary_structure = ss_code

    def get_ca(self):
        for atom in self.atoms:
            if atom.is_CA():
                return atom
        return None

    def get_cb(self, include_glycine=True):
        for atom in self.atoms:
            if atom.is_CB(InclGlyCA=include_glycine):
                return atom
        return None

    def distance(self, other_residue):
        min_dist = float('inf')
        for self_atom in self.atoms:
            for other_atom in other_residue.atoms:
                d = self_atom.distance(other_atom, intra=True)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def in_contact(self, other_residue, distance_thresh=4.5, side_chain_only=False):
        if side_chain_only:
            for self_atom in self.atoms:
                if not self_atom.is_backbone():
                    for other_atom in other_residue.atoms:
                        if not other_atom.is_backbone():
                            if self_atom.distance(other_atom, intra=True) < distance_thresh:
                                return True
            return False
        else:
            for self_atom in self.atoms:
                for other_atom in other_residue.atoms:
                    if self_atom.distance(other_atom, intra=True) < distance_thresh:
                        return True
            return False

    def in_contact_residuelist(self, residuelist, distance_thresh=4.5, side_chain_only=False):
        for residue in residuelist:
            if self.in_contact(residue, distance_thresh, side_chain_only):
                return True
        return False

    @staticmethod
    def get_residue(number, chain, residue_type, residuelist):
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


class Atom:  # (Coords):
    """An Atom container with the full Structure coordinates and the Atom unique data. Pass a reference to the full
    Structure coordinates for Keyword Arg coords=self.coords"""
    def __init__(self, index=None, number=None, atom_type=None, alt_location=None, residue_type=None, chain=None,
                 residue_number=None, code_for_insertion=None, occ=None, temp_fact=None, element_symbol=None,
                 atom_charge=None, coords=None):  # x, y, z,
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
            # super().__init__(coords=coords)  # init from Coords class
        # self.x = x
        # self.y = y
        # self.z = z

    @classmethod
    def from_info(cls, *args):  # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occ, temp_fact, element_symbol, atom_charge
        """Initialize without coordinates"""
        return cls(*args)

    @property
    def coords(self):  # , transformation_operator=None):
        """This holds the atomic coords which is a view from the Structure that created them"""
        # if transformation_operator:
        #     return np.matmul([self.x, self.y, self.z], transformation_operator)
        # else:
        # print(self.index, self.number, self.type, self.alt_location, self.residue_type, self.residue_number)
        # print(len(self._coords.coords))
        return self._coords.coords[self.index]  # [self.x, self.y, self.z]
        # return self.Coords.coords(which returns a np.array)[slicing that by the atom.index]

    @coords.setter
    def coords(self, coords):
        if isinstance(coords, Coords):
            self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def is_backbone(self):
        """Check if the Atom is a backbone Atom
         Returns:
             (bool)"""
        # backbone_specific_atom_type = ["N", "CA", "C", "O"]
        # if self.type in backbone_specific_atom_type:
        #     return True
        # else:
        #     return False
        return self.is_n() or self.is_CA() or self.is_c() or self.is_o()

    def is_n(self):
        return self.type == 'N'

    def is_CB(self, InclGlyCA=False):
        if InclGlyCA:
            return self.type == 'CB' or (self.type == 'CA' and self.residue_type == 'GLY')
        else:
            return self.type == 'CB' or (self.type == 'H' and self.residue_type == 'GLY')

    def is_CA(self):
        return self.type == 'CA'

    def is_c(self):
        return self.type == 'C'

    def is_o(self):
        return self.type == 'O'

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

    def translate(self, tx):
        self.coords = self.coords + tx

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