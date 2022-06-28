from __future__ import annotations

import os
from copy import copy, deepcopy
from itertools import chain as iter_chain, combinations_with_replacement, combinations, product
# from logging import Logger
from math import sqrt, cos, sin, prod, ceil, pi
from typing import Iterable, IO, Any

import numpy as np
# from numba import njit, jit
from Bio.Data.IUPACData import protein_letters_1to3_extended
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.neighbors._ball_tree import BinaryTree  # this typing implementation supports BallTree or KDTree

import PathUtils as PUtils
from DesignMetrics import calculate_match_metrics, fragment_metric_template, format_fragment_metrics
from JobResources import fragment_factory, Database, FragmentDatabase
from PDB import PDB, parse_cryst_record
from SequenceProfile import SequenceProfile
from Structure import Coords, Structure, Structures, Chain, Entity, Residue, Residues, GhostFragment, MonoFragment, \
    write_frag_match_info_file, Fragment, StructureBase
from SymDesignUtils import DesignError, calculate_overlap, z_value_from_match_score, start_log, null_log, \
    match_score_from_z_value, dictionary_lookup, digit_translate_table
from classes.EulerLookup import EulerLookup, euler_factory
from classes.SymEntry import get_rot_matrices, make_rotations_degenerate, SymEntry, point_group_setting_matrix_members, \
    symmetry_combination_format, parse_symmetry_to_sym_entry, symmetry_factory
from utils.GeneralUtils import transform_coordinate_sets
from utils.SymmetryUtils import valid_subunit_number, space_group_cryst1_fmt_dict, layer_group_cryst1_fmt_dict, \
    generate_cryst1_record, space_group_number_operations, point_group_symmetry_operators, \
    space_group_symmetry_operators, possible_symmetries, rotation_range, setting_matrices, inv_setting_matrices, \
    origin, flip_x_matrix, identity_matrix, SymmetryError

# from operator import itemgetter

# Globals
logger = start_log(name=__name__)
index_offset = 1
seq_res_len = 52
config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location


class MultiModel:
    """Class for working with iterables of State objects of macromolecular polymers (proteins for now). Each State
    container comprises Structure object(s) which can also be accessed as a unique Model by slicing the Structure across
    States.

    self.structures holds each of the individual Structure objects which are involved in the MultiModel. As of now,
    no checks are made whether the identity of these is the same across States"""
    def __init__(self, model=None, models=None, state=None, states=None, independent=False, log=None, **kwargs):
        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if model:
            if not isinstance(model, Model):
                model = Model(model)

            self.models = [model]
            self.states = [[state] for state in model.models]
            # self.structures = [[model.states]]

        if isinstance(models, list):
            self.models = models
            self.states = [[model[state_idx] for model in models] for state_idx in range(len(models[0].models))]
            # self.states = [[] for state in models[0].models]
            # for model in models:
            #     for state_idx, state in enumerate(model.models):
            #         self.states[state_idx].append(state)

            # self.structures = [[model.states] for model in models]

            # self.structures = models
            # structures = [[] for model in models]
            # for model in models:
            #     for idx, state in enumerate(model):
            #         structures[idx].append(state)

        # collect the various structures and corresponding states of separate Structures
        if state:
            if not isinstance(state, State):
                state = State(state)

            self.states = [state]
            self.models = [[structure] for structure in state.structures]
            # self.structures = [[structure] for structure in state.structures]
        if isinstance(states, list):
            # modify loop order by separating Structure objects in same state to individual Entity containers
            self.states = states
            self.models = [[state[model_idx] for state in states] for model_idx in range(len(states[0].structures))]
            # self.models = [state.structures for state in states]

            # self.structures = [[] for structure in states[0]]
            # for state in states:
            #     for idx, structure in enumerate(state.structures):
            #         self.structures[idx].append(structure)

        # indicate whether each structure is an independent set of models by setting dependent to corresponding tuple
        dependents = [] if independent else range(self.number_of_models)
        self.dependents = set(dependents)  # tuple(dependents)

    @classmethod
    def from_model(cls, model, **kwargs):
        """Construct a MultiModel from a Structure object container with or without multiple states
        Ex: [Structure1_State1, Structure1_State2, ...]
        """
        return cls(model=model, **kwargs)

    @classmethod
    def from_models(cls, models, independent=False, **kwargs):
        """Construct a MultiModel from an iterable of Structure object containers with or without multiple states
        Ex: [Model[Structure1_State1, Structure1_State2, ...], Model[Structure2_State1, ...]]

        Keyword Args:
            independent=False (bool): Whether the models are independent (True) or dependent on each other (False)
        """
        return cls(models=models, independent=independent, **kwargs)

    @classmethod
    def from_state(cls, state, **kwargs):
        """Construct a MultiModel from a Structure object container, representing a single Structural state.
        For instance, one trajectory in a sequence design with multiple polymers or a SymmetricModel
        Ex: [Model_State1[Structure1, Structure2, ...], Model_State2[Structure1, Structure2, ...]]
        """
        return cls(state=state, **kwargs)

    @classmethod
    def from_states(cls, states, independent=False, **kwargs):
        """Construct a MultiModel from an iterable of Structure object containers, each representing a different state
        of the Structures. For instance, multiple trajectories in a sequence design
        Ex: [Model_State1[Structure1, Structure2, ...], Model_State2[Structure1, Structure2, ...]]

        Keyword Args:
            independent=False (bool): Whether the models are independent (True) or dependent on each other (False)
        """
        return cls(states=states, independent=independent, **kwargs)

    # @property
    # def number_of_structures(self):
    #     return len(self.structures)
    #
    # @property
    # def number_of_states(self):
    #     return max(map(len, self.structures))

    @property
    def number_of_models(self):
        return len(self.models)

    @property
    def number_of_states(self):
        return len(self.states)
        # return max(map(len, self.models))

    def get_models(self):
        return [Models(model, log=self.log) for model in self.models]

    def get_states(self):
        return [State(state, log=self.log) for state in self.states]

    # @property
    # def models(self):
    #     return [Model(model) for model in self._models]
    #
    # @models.setter
    # def models(self, models):
    #     self._models = models
    #
    # @property
    # def states(self):
    #     return [State(state) for state in self._states]
    #
    # @states.setter
    # def states(self, states):
    #     self._states = states

    @property
    def independents(self) -> set[int]:
        """Retrieve the indices of the Structures whose model information is independent of other Structures"""
        return set(range(self.number_of_models)).difference(self.dependents)

    def add_state(self, state):
        """From a state, incorporate the Structures in the state into the existing Model

        Sets:
            self.states
            self.models
        """
        self.states.append(state)  # Todo ensure correct methods once State is subclassed as UserList
        try:
            for idx, structure in enumerate(self.models):
                structure.append(state[idx])
            del self._model_iterator
            # delattr(self, '_model_iterator')
        except IndexError:  # Todo handle mismatched lengths, either passed or existing
            raise IndexError('The added State contains fewer Structures than present in the MultiModel. Only pass a '
                             'State that has the same number of Structures (%d) as the MultiModel' % self.number_of_models)

    def add_model(self, model, independent=False):
        """From a Structure with multiple states, incorporate the Model into the existing Model

        Sets:
            self.states
            self.models
            self.dependents
        """
        self.models.append(model)  # Todo ensure correct methods once Model is subclassed as UserList
        try:
            for idx, state in enumerate(self.states):
                state.append(model[idx])
            del self._model_iterator
            # delattr(self, '_model_iterator')
        except IndexError:  # Todo handle mismatched lengths, either passed or existing
            raise IndexError('The added Model contains fewer models than present in the MultiModel. Only pass a Model '
                             'that has the same number of States (%d) as the MultiModel' % self.number_of_states)

        if not independent:
            self.dependents.add(self.number_of_models - 1)

    def enumerate_models(self) -> list:
        """Given the MultiModel Structures and dependents, construct an iterable of all States in the MultiModel"""
        # print('enumerating_models, states', self.states, 'models', self.models)
        # First, construct tuples of independent structures if available
        independents = self.independents
        if not independents:  # all dependents are already in order
            return self.get_states()
            # return zip(self.structures)
        else:
            independent_sort = sorted(independents)
        independent_gen = product(*[self.models[idx] for idx in independent_sort])
        # independent_gen = combinations([self.structures[idx] for idx in independents], len(independents))

        # Next, construct tuples of dependent structures
        dependent_sort = sorted(self.dependents)
        if not dependent_sort:  # all independents are already in order and combined
            # independent_gen = list(independent_gen)
            # print(independent_gen)
            return [State(state, log=self.log) for state in independent_gen]
            # return list(independent_gen)
        else:
            dependent_zip = zip(self.models[idx] for idx in dependent_sort)

        # Next, get all model possibilities in an unordered fashion
        unordered_structure_model_gen = product(dependent_zip, independent_gen)
        # unordered_structure_model_gen = combinations([dependent_zip, independent_gen], self.number_of_structures)
        # unordered_structure_models = zip(dependents + independents)
        unordered_structure_models = \
            list(zip(*(dep_structs + indep_structs for dep_structs, indep_structs in unordered_structure_model_gen)))

        # Finally, repackage in an ordered fashion
        models = []
        for idx in range(self.number_of_models):
            dependent_index = dependent_sort.index(idx)
            if dependent_index == -1:  # no index found, idx is in independents
                independent_index = independent_sort.index(idx)
                if independent_index == -1:  # no index found? Where is it
                    raise IndexError('The index was not found in either independent or dependent models!')
                else:
                    models.append(unordered_structure_models[len(dependent_sort) + independent_index])
            else:  # index found, idx is in dependents
                models.append(unordered_structure_models[dependent_index])

        return [State(state, log=self.log) for state in models]

    @property
    def model_iterator(self):
        try:
            return iter(self._model_iterator)
        except AttributeError:
            self._model_iterator = self.enumerate_models()
            return iter(self._model_iterator)

    def __len__(self):
        try:
            return len(self._model_iterator)
        except AttributeError:
            self._model_iterator = self.enumerate_models()
            return len(self._model_iterator)

    def __iter__(self):
        yield from self.model_iterator
        # yield from self.enumerate_models()


# (BaseModel)?
class State(Structures):
    """A collection of Structure objects comprising one distinct configuration"""
    # def __init__(self, structures=None, **kwargs):  # log=None,
    #     super().__init__(**kwargs)
    #     # super().__init__()  # without passing **kwargs, there is no need to ensure base Object class is protected
    #     # if log:
    #     #     self.log = log
    #     # elif log is None:
    #     #     self.log = null_log
    #     # else:  # When log is explicitly passed as False, use the module logger
    #     #     self.log = logger
    #
    #     if isinstance(structures, list):
    #         if all([True if isinstance(structure, Structure) else False for structure in structures]):
    #             self.structures = structures
    #             # self.data = structures
    #         else:
    #             self.structures = []
    #             # self.data = []
    #     else:
    #         self.structures = []
    #         # self.data = []
    #
    # @property
    # def number_of_structures(self):
    #     return len(self.structures)
    #
    # @property
    # def coords(self):
    #     """Return a view of the Coords from the Structures"""
    #     try:
    #         coords_exist = self._coords.shape  # check on first call for attribute, if not, make, else, replace coords
    #         total_atoms = 0
    #         for structure in self.structures:
    #             new_atoms = total_atoms + structure.number_of_atoms
    #             self._coords[total_atoms: new_atoms] = structure.coords
    #             total_atoms += total_atoms
    #         return self._coords
    #     except AttributeError:
    #         coords = [structure.coords for structure in self.structures]
    #         # coords = []
    #         # for structure in self.structures:
    #         #     coords.extend(structure.coords)
    #         self._coords = np.concatenate(coords)
    #
    #         return self._coords
    #
    # # @coords.setter
    # # def coords(self, coords):
    # #     if isinstance(coords, Coords):
    # #         self._coords = coords
    # #     else:
    # #         raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    # #                              'view. To pass the Coords object for a Structure, use the private attribute _coords')

    # @property
    # def model_coords(self):  # TODO RECONCILE with coords, SymmetricModel variation
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

    # @property
    # def atoms(self):
    #     """Return a view of the Atoms from the Structures"""
    #     try:
    #         return self._atoms
    #     except AttributeError:
    #         atoms = []
    #         for structure in self.structures:
    #             atoms.extend(structure.atoms)
    #         self._atoms = Atoms(atoms)
    #         return self._atoms
    #
    # @property
    # def number_of_atoms(self):
    #     return len(self.coords)
    #
    # @property
    # def residues(self):  # TODO Residues iteration
    #     try:
    #         return self._residues.residues.tolist()
    #     except AttributeError:
    #         residues = []
    #         for structure in self.structures:
    #             residues.extend(structure.residues)
    #         self._residues = Residues(residues)
    #         return self._residues.residues.tolist()
    #
    # @property
    # def number_of_residues(self):
    #     return len(self.residues)
    #
    # @property
    # def coords_indexed_residues(self):
    #     try:
    #         return self._coords_indexed_residues
    #     except AttributeError:
    #         self._coords_indexed_residues = \
    #             [residue for residue in self.residues for _ in residue.range]
    #         return self._coords_indexed_residues
    #
    # @property
    # def coords_indexed_residue_atoms(self):
    #     try:
    #         return self._coords_indexed_residue_atoms
    #     except AttributeError:
    #         self._coords_indexed_residue_atoms = \
    #             [res_atom_idx for residue in self.residues for res_atom_idx in residue.range]
    #         return self._coords_indexed_residue_atoms
    #
    # # @property  # SAME implementation in Structure
    # # def center_of_mass(self):
    # #     """The center of mass for the model Structure, either an asu, or other pdb
    # #
    # #     Returns:
    # #         (numpy.ndarray)
    # #     """
    # #     return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms), self.coords)
    #
    # @property
    # def backbone_indices(self):
    #     try:
    #         return self._backbone_indices
    #     except AttributeError:
    #         self._backbone_indices = []
    #         for structure in self.structures:
    #             self._backbone_indices.extend(structure.coords_indexed_backbone_indices)
    #         return self._backbone_indices
    #
    # @property
    # def backbone_and_cb_indices(self):
    #     try:
    #         return self._backbone_and_cb_indices
    #     except AttributeError:
    #         self._backbone_and_cb_indices = []
    #         for structure in self.structures:
    #             self._backbone_and_cb_indices.extend(structure.coords_indexed_backbone_and_cb_indices)
    #         return self._backbone_and_cb_indices
    #
    # @property
    # def cb_indices(self):
    #     try:
    #         return self._cb_indices
    #     except AttributeError:
    #         self._cb_indices = []
    #         for structure in self.structures:
    #             self._cb_indices.extend(structure.coords_indexed_cb_indices)
    #         return self._cb_indices
    #
    # @property
    # def ca_indices(self):
    #     try:
    #         return self._ca_indices
    #     except AttributeError:
    #         self._ca_indices = []
    #         for structure in self.structures:
    #             self._ca_indices.extend(structure.coords_indexed_ca_indices)
    #         return self._ca_indices
    #

    def write(self, increment_chains: bool = False, **kwargs) -> str | None:
        """Write Structures to a file specified by out_path or with a passed file_handle.

        Keyword Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            increment_chains: Whether to write each Structure with a new chain name, otherwise write as a new Model
            header: If there is header information that should be included. Pass new lines with a "\n"
        Returns:
            The name of the written file if out_path is used
        """
        self.log.warning('The ability to write States to file has not been thoroughly debugged. If your State consists '
                         'of various types of Structure containers (PDB, Structures, chains, or entities, check your '
                         'file is as expected before preceeding')
        return super().write(increment_chains=increment_chains, **kwargs)

        # if file_handle:  # Todo handle with multiple Structure containers
        #     file_handle.write('%s\n' % self.return_atom_string(**kwargs))
        #     return
        #
        # with open(out_path, 'w') as f:
        #     if header:
        #         if isinstance(header, str):
        #             f.write(header)
        #         # if isinstance(header, Iterable):
        #
        #     if increment_chains:
        #         available_chain_ids = self.return_chain_generator()
        #         for structure in self.structures:
        #             # for entity in structure.entities:  # Todo handle with multiple Structure containers
        #             chain = next(available_chain_ids)
        #             structure.write(file_handle=f, chain=chain)
        #             c_term_residue = structure.c_terminal_residue
        #             f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER',
        #                                                                   c_term_residue.atoms[-1].number + 1,
        #                                                                   c_term_residue.type, chain,
        #                                                                   c_term_residue.number))
        #     else:
        #         for model_number, structure in enumerate(self.structures, 1):
        #             f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
        #             # for entity in structure.entities:  # Todo handle with multiple Structure containers
        #             structure.write(file_handle=f)
        #             c_term_residue = structure.c_terminal_residue
        #             f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER',
        #                                                                   c_term_residue.atoms[-1].number + 1,
        #                                                                   c_term_residue.type, structure.chain_id,
        #                                                                   c_term_residue.number))
        #             f.write('ENDMDL\n')
    #
    # def __getitem__(self, idx):
    #     return self.structures[idx]


class Model(PDB):
    """Manipulate Structure objects containing multiple Chain or Entity objects

    If you have multiple Models or States, use the MultiModel class to store and retrieve that data
    """
    def __init__(self, model: Structure = None, **kwargs):

        if model:
            if isinstance(model, Structure):
                super().__init__(**model.get_structure_containers(), **kwargs)
            else:
                raise NotImplementedError(f'Setting {type(self).__name__} with a {type(model).__name__} isn\'t '
                                          f'supported')
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_model(cls, model, **kwargs):
        """Initialize from an existing Model"""
        return cls(model=model, **kwargs)

    @property
    def chain_breaks(self) -> list[int]:  # Todo KEEP
        return [entity.c_terminal_residue.number for entity in self.entities]

    @property
    def atom_indices_per_chain(self) -> list[list[int]]:
        """Return the atom indices for each Chain in the Model"""
        return [chain.atom_indices for chain in self.chains]

    @property
    def atom_indices_per_entity(self) -> list[list[int]]:
        """Return the atom indices for each Entity in the Model"""
        return [entity.atom_indices for entity in self.entities]

    @property
    def residue_indices_per_chain(self) -> list[list[int]]:
        return [chain.residue_indices for chain in self.chains]

    @property
    def residue_indices_per_entity(self) -> list[list[int]]:
        return [entity.residue_indices for entity in self.entities]

    @property
    def number_of_atoms_per_chain(self) -> list[int]:
        return [chain.number_of_atoms for chain in self.chains]

    @property
    def number_of_atoms_per_entity(self) -> list[int]:
        return [entity.number_of_atoms for entity in self.entities]

    @property
    def number_of_residues_per_chain(self) -> list[int]:
        return [chain.number_of_residues for chain in self.chains]

    @property
    def number_of_residues_per_entity(self) -> list[int]:
        return [entity.number_of_residues for entity in self.entities]

    # @property
    # def coords_indexed_residues(self):  # TODO CONNECT
    #     try:
    #         return self._coords_indexed_residues
    #     except AttributeError:
    #         self._coords_indexed_residues = \
    #             [residue for residue in self.residues for _ in residue.range]
    #         return self._coords_indexed_residues
    #
    # @property
    # def coords_indexed_residue_atoms(self):  # TODO CONNECT
    #     try:
    #         return self._coords_indexed_residue_atoms
    #     except AttributeError:
    #         self._coords_indexed_residue_atoms = \
    #             [res_atom_idx for residue in self.residues for res_atom_idx in residue.range]
    #         return self._coords_indexed_residue_atoms

    # @property
    # def model_coords(self) -> np.ndarray:  # DONE RECONCILE with coords, SymmetricModel, and State variation
    #     """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
    #     return self._model_coords.coords
    #
    # @model_coords.setter
    # def model_coords(self, coords: Coords):
    #     # if isinstance(coords, Coords):
    #     try:
    #         coords.coords  # are they Coords?
    #         self._model_coords = coords
    #     # else:
    #     except AttributeError:
    #         raise AttributeErr('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    #                              'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    # def format_seqres(self, **kwargs) -> str:
    #     """Format the reference sequence present in the SEQRES remark for writing to the output header
    #
    #     Keyword Args:
    #         **kwargs
    #     Returns:
    #         The PDB formatted SEQRES record
    #     """
    #     # if self.pdb.reference_sequence:
    #     #     formated_reference_sequence = {entity.chain_id: entity.reference_sequence for entity in self.entities}
    #     #     formated_reference_sequence = \
    #     #         {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in sequence)))
    #     #          for chain, sequence in formated_reference_sequence.items()}
    #     if self.pdb.reference_sequence:
    #         formated_reference_sequence = \
    #             {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in sequence)))
    #              for chain, sequence in self.pdb.reference_sequence.items()}  # .reference_sequence doesn't have chains
    #         chain_lengths = {chain: len(sequence) for chain, sequence in self.pdb.reference_sequence.items()}
    #         return '%s\n' \
    #                % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
    #                            % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
    #                            for chain, sequence in formated_reference_sequence.items()
    #                            for line_number in range(1, 1 + ceil(len(sequence)/seq_res_len)))
    #     else:
    #         return ''

    def format_header(self, **kwargs) -> str:
        """Return the BIOMT and the SEQRES records based on the Model

        Returns:
            The header with PDB file formatting
        """
        return self.format_biomt(**kwargs) + self.format_seqres(**kwargs)


class Models(Model):
    """Keep track of different variations of the same Model object such as altered coordinates (different decoy's or
    symmetric copies) [or mutated Residues]. In PDB parlance, this would be a multimodel, however could be multiple
    PDB files that share a common element.
    """
    _model_coords: Coords

    def __init__(self, models: Iterable[Model] = None, **kwargs):
        if models:
            for model in models:
                if not isinstance(model, Model):
                    raise TypeError(f'Can\'t initialize {type(self).__name__} with a {type(model).__name__}. Must be an'
                                    f' iterable of Model')
            self.models = [model for model in models]
        else:
            super().__init__(**kwargs)
            self.models = []

    @classmethod
    def from_models(cls, models: Iterable[Model], **kwargs):
        """Initialize from an iterable of Model"""
        return cls(models=models, **kwargs)

    # @classmethod
    # def from_file(cls, file, **kwargs):
    #     """Construct Models from multimodel PDB file using the PDB.chains
    #     Ex: [Chain1, Chain1, ...]
    #     """
    #     pdb = PDB.from_file(file, **kwargs)  # Todo make independent parsing function
    #     # new_model = cls(models=pdb.chains)
    #     return cls(models=pdb.chains, **kwargs)
    #
    # @classmethod
    # def from_pdb(cls, pdb, **kwargs):
    #     """Construct Models from multimodel PDB file using the PDB.chains
    #     Ex: [Chain1, Chain1, ...]
    #     """
    #     return cls(models=pdb.chains, **kwargs)

    @property
    def number_of_models(self) -> int:
        """The number of unique models that are found in the Models object"""
        return len(self.models)

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

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, increment_chains: bool = False,
              **kwargs) -> str | bytes | None:
        """Write Model Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            increment_chains: Whether to write each Structure with a new chain name, otherwise write as a new Model
        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'Models is writing')

        def models_write(handle):
            self.write_header(handle, **kwargs)
            # self.models is populated
            if increment_chains:  # assembly requested, check on the mechanism of symmetric writing
                # we won't allow incremental chains when the Model is plain as the models are all the same and
                # therefore belong with the models label
                available_chain_ids = self.return_chain_generator()
                for structure in self.models:
                    for entity in structure.entities:
                        chain = next(available_chain_ids)
                        entity.write(file_handle=handle, chain=chain)
                        c_term_residue = entity.c_terminal_residue
                        handle.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.
                                     format('TER', c_term_residue.atoms[-1].number + 1, c_term_residue.type, chain,
                                            c_term_residue.number))
            else:
                for model_number, structure in enumerate(self.models, 1):
                    handle.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                    for entity in structure.entities:
                        entity.write(file_handle=handle)
                        c_term_residue = entity.c_terminal_residue
                        handle.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.
                                     format('TER', c_term_residue.atoms[-1].number + 1, c_term_residue.type,
                                            entity.chain_id, c_term_residue.number))
                    handle.write('ENDMDL\n')

        if file_handle:
            models_write(file_handle)
            return
        else:
            with open(out_path, 'w') as outfile:
                models_write(outfile)
            return out_path

    def __getitem__(self, idx: int) -> Structure:
        return self.models[idx]


class SymmetricModel(Models):
    _asu_model_idx: int
    _oligomeric_model_indices: dict[Entity, list[int]]
    assembly_tree: BinaryTree | None
    expand_matrices: np.ndarray | list[list[float]] | None
    expand_translations: np.ndarray | list[float] | None
    uc_dimensions: list[float] | None

    def __init__(self, sym_entry: SymEntry | int = None, symmetry: str = None,
                 uc_dimensions: list[float] = None, expand_matrices: np.ndarray | list = None,
                 surrounding_uc: bool = True, **kwargs):
        """

        Args:
            sym_entry: The SymEntry which specifies all symmetry parameters
            symmetry: The name of a symmetry to be searched against the existing compatible symmetries
            uc_dimensions: Whether the symmetric coords should be generated from the ASU coords
            expand_matrices: A set of custom expansion matrices
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        #     generate_assembly_coords: Whether the symmetric coords should be generated from the ASU coords
        #     generate_symmetry_mates: Whether the symmetric models should be generated from the ASU model
        #          asu: PDB = None, asu_file: str = None
        super().__init__(**kwargs)  # log=log,
        # if asu and isinstance(asu, Structure):
        #     self.asu = asu  # the pose specific asu
        # elif asu_file:
        #     self.asu = PDB.from_file(asu_file, log=self.log, **kwargs)
        # # add stripped kwargs back
        # kwargs['symmetry'] = symmetry
        # kwargs['sym_entry'] = sym_entry
        # self.models = []  # from Models
        # self.model_coords = [] <- designated as symmetric_coords
        self.assembly_tree = None  # stores a sklearn tree for coordinate searching
        # self._asu_model_idx = None
        self.expand_matrices = None
        self.expand_translations = None
        # self._oligomeric_model_indices = {}
        self.uc_dimensions = None  # uc_dimensions

        # initialize symmetry
        self.set_symmetry(sym_entry=sym_entry, symmetry=symmetry, uc_dimensions=uc_dimensions,
                          expand_matrices=expand_matrices)
        if self.symmetry:  # this is set if symmetry keyword args were passed
            self.generate_symmetric_coords(surrounding_uc=surrounding_uc)  # default has surrounding_uc=True
            # if generate_symmetry_mates:  # always set to False before. commenting out
            #     self.generate_assembly_symmetry_models(**kwargs)

    @classmethod
    def from_assembly(cls, assembly: list, sym_entry: SymEntry | int = None, symmetry: str = None, **kwargs):
        """Initialize from a symmetric assembly"""
        if not symmetry or not sym_entry:
            raise ValueError(f'Can\'t initialize {cls.from_assembly.__name__} without symmetry! Pass symmetry or '
                             f'sym_entry with {cls.from_assembly.__name__}')
        return cls(models=assembly, sym_entry=sym_entry, symmetry=symmetry, **kwargs)

    # @classmethod
    # def from_asu(cls, asu, **kwargs):  # generate_symmetry_mates=True
    #     """From a Structure representing an asu, return the SymmetricModel with generated symmetry mates
    #
    #     Keyword Args:
    #         # generate_symmetry_mates=True (bool): Whether the symmetric copies of the ASU model should be generated
    #         surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
    #     """
    #     return cls(asu=asu, **kwargs)  # generate_symmetry_mates=generate_symmetry_mates,
    #
    # @classmethod
    # def from_asu_file(cls, asu_file, **kwargs):
    #     return cls(asu_file=asu_file, **kwargs)

    def set_symmetry(self, sym_entry: SymEntry | int = None, symmetry: str = None,
                     uc_dimensions: list[float] = None, expand_matrices: np.ndarray | list = None):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required

        Args:
            sym_entry: The SymEntry which specifies all symmetry parameters
            symmetry: The name of a symmetry to be searched against the existing compatible symmetries
            uc_dimensions: Whether the symmetric coords should be generated from the ASU coords
            expand_matrices: A set of custom expansion matrices
        """
        # try to solve for symmetry as we want uc_dimensions if available for cryst ops
        if self.cryst_record:  # was populated from file parsing
            if not uc_dimensions and not symmetry:  # only if user didn't provide both
                uc_dimensions, symmetry = parse_cryst_record(self.cryst_record)

        if symmetry:  # ensure conversion to Hermann–Mauguin notation. ex: P23 not P 2 3
            symmetry = ''.join(symmetry.split())

        if sym_entry:
            if isinstance(sym_entry, SymEntry):  # attach if SymEntry class set up
                self.sym_entry = sym_entry
            else:  # try to solv using integer and any info in symmetry. Fails upon non Nanohedra chiral space-group...
                self.sym_entry = parse_symmetry_to_sym_entry(sym_entry=sym_entry, symmetry=symmetry)
        elif symmetry:  # either provided or solved from cryst_record
            # existing sym_entry takes precedence since the user specified it
            try:  # Fails upon non Nanohedra chiral space-group...
                if not self.sym_entry:  # ensure conversion to Hermann–Mauguin notation. ex: P23 not P 2 3
                    self.sym_entry = parse_symmetry_to_sym_entry(symmetry=symmetry)
            except ValueError as error:  # let's print the error and move on since this is likely just parsed
                logger.warning(str(error))
                self.symmetry = symmetry
                # not sure if cryst record can differentiate between 2D and 3D. 3D will be wrong if actually 2D
                self.dimension = 2 if symmetry in layer_group_cryst1_fmt_dict else 3

            # if symmetry in layer_group_cryst1_fmt_dict:  # not available yet for non-Nanohedra PG's
            #     self.dimension = 2
            #     self.symmetry = symmetry
            # elif symmetry in space_group_cryst1_fmt_dict:  # not available yet for non-Nanohedra SG's
            #     self.dimension = 3
            #     self.symmetry = symmetry
            # elif symmetry in possible_symmetries:
            #     self.symmetry = possible_symmetries[symmetry]
            #     self.point_group_symmetry = possible_symmetries[symmetry]
            #     self.dimension = 0

            # elif self.uc_dimensions is not None:
            #     raise DesignError('Symmetry %s is not available yet! If you didn\'t provide it, the symmetry was likely'
            #                       ' set from a PDB file. Get the symmetry operations from the international'
            #                       ' tables and add to the pickled operators if this displeases you!' % symmetry)
            # else:  # when a point group besides T, O, or I is provided
            #     raise DesignError('Symmetry %s is not available yet! Get the canonical symm operators from %s and add '
            #                       'to the pickled operators if this displeases you!' % (symmetry, PUtils.orient_dir))
        else:  # no symmetry was provided
            # since this is now subclassed by Pose, lets ignore this error since self.symmetry is explicitly False
            return
            # raise SymmetryError('A SymmetricModel was initiated without any symmetry! Ensure you specify the symmetry '
            #                     'upon class initialization by passing symmetry=, or sym_entry=')

        # set the uc_dimensions if they were parsed or provided
        if self.dimension > 0 and uc_dimensions is not None:
            self.uc_dimensions = uc_dimensions

        if expand_matrices:  # perhaps these would be from a fiber or some sort of BIOMT?
            if isinstance(expand_matrices, tuple) and len(expand_matrices) == 2:
                self.log.critical('Providing expansion matrices may result in program crash if you '
                                  'don\'t work on the SymmetricModel class! Proceed with caution')
                expand_matrices, expand_translations = expand_matrices
                self.expand_translations = \
                    np.ndarray(expand_translations) if not isinstance(expand_translations, np.ndarray) \
                    else expand_translations
                # lets assume expand_matrices were provided in a standard orientation and transpose
                # using .swapaxes(-2, -1) call instead of .transpose() for safety
                self.expand_matrices = \
                    np.ndarray(expand_matrices).swapaxes(-2, -1) if not isinstance(expand_matrices, np.ndarray) \
                    else expand_matrices
            else:
                raise SymmetryError(f'The expand matrix form {expand_matrices} is not supported! Must provide a tuple '
                                    f'of array like objects with the form (expand_matrix(s), expand_translation(s))')
        else:
            if self.dimension == 0:
                self.expand_matrices, self.expand_translations = point_group_symmetry_operators[self.symmetry], origin
            else:
                self.expand_matrices, self.expand_translations = space_group_symmetry_operators[self.symmetry]

        # Todo?
        #  remove any existing symmetry attr from the Model
        #  if not self.sym_entry:
        #      del self._symmetry

        # if generate_assembly_coords:  # if self.asu and generate_assembly_coords:
        #     self.generate_symmetric_coords(**kwargs)
        #     if generate_symmetry_mates:
        #         self.generate_assembly_symmetry_models(**kwargs)

    # def set_asu_coords(self, coords):
    #     # overwrite all the coords for each member Entity
    #     self.pdb.replace_coords(coords)
    #     self.generate_symmetric_coords()
    #     # Todo delete any saved attributes from the SymmetricModel
    #     #  self.symmetric_coords
    #     #  self.asu_equivalent_model_idx
    #     #  self.oligomeric_equivalent_model_idxs

    @property
    def chains(self) -> list[Entity]:
        """Return all the Chain objects including symmetric chains"""
        return [chain for entity in self.entities for chain in entity.chains]

    def chain(self, chain_id: str) -> Chain:
        """Return the Entity corresponding to the provided chain_id"""
        return self.entity_from_chain(chain_id)

    # each of the below functions with raise NotImplementedError need to be removed or solved
    @property
    def atom_indices_per_chain(self) -> list[list[int]]:
        """Return the atom indices for each Chain in the Model"""
        raise NotImplementedError(f'This function is not implemented for a {type(self).__name__}')
        return [chain.atom_indices for chain in self.chains]

    @property
    def residue_indices_per_chain(self) -> list[list[int]]:
        raise NotImplementedError(f'This function is not implemented for a {type(self).__name__}')
        return [chain.residue_indices for chain in self.chains]

    @property
    def number_of_atoms_per_chain(self) -> list[int]:
        raise NotImplementedError(f'This function is not implemented for a {type(self).__name__}')
        return [chain.number_of_atoms for chain in self.chains]

    @property
    def atom_indices_per_entity_model(self) -> list[list[int]]:
        # Todo
        #   alternative solution may be quicker by performing the following multiplication then .flatten()
        #   broadcast entity_indices ->
        #   (np.arange(model_number) * coords_length).T
        #   |
        #   v
        number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        return [[idx + (number_of_atoms * model_number) for model_number in range(self.number_of_models)
                 for idx in entity_indices] for entity_indices in self.atom_indices_per_entity]

    @property
    def sequence(self) -> str:
        """Holds the SymmetricModel amino acid sequence"""
        return ''.join(entity.sequence for entity in self.entities)

    @property
    def reference_sequence(self) -> str:
        """Return the entire SymmetricModel sequence, constituting all Residues, not just structurally modelled ones

        Returns:
            The sequence according to each of the Entity references
        """
        return ''.join(entity.reference_sequence for entity in self.entities)

    @property
    def sym_entry(self) -> SymEntry | None:
        """The SymEntry specifies the symmetric parameters for the utilized symmetry"""
        try:
            return self._sym_entry
        except AttributeError:
            # raise SymmetryError('No symmetry entry was specified!')
            self._sym_entry = None
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: SymEntry | int):
        if isinstance(sym_entry, SymEntry):
            self._sym_entry = sym_entry
        else:  # try to convert
            self._sym_entry = symmetry_factory.get(sym_entry)
        # Todo remove hidden symmetric ._ attributes if set

    @property
    def symmetry(self) -> str | None:
        """The resulting_symmetry of the SymEntry"""
        try:
            return self._symmetry
        except AttributeError:
            self._symmetry = getattr(self.sym_entry, 'resulting_symmetry', None)
            return self._symmetry

    @symmetry.setter
    def symmetry(self, symmetry: str | None):
        self._symmetry = symmetry

    @property
    def point_group_symmetry(self) -> str | None:
        """The point group underlying the resulting SymEntry"""
        try:
            return self._point_group_symmetry
        except AttributeError:
            self._point_group_symmetry = getattr(self.sym_entry, 'point_group_symmetry', None)
            return self._point_group_symmetry

    @point_group_symmetry.setter
    def point_group_symmetry(self, point_group_symmetry: str | None):
        self._point_group_symmetry = point_group_symmetry

    @property
    def dimension(self) -> int | None:
        """The dimension of the symmetry from 0, 2, or 3"""
        try:
            return self._dimension
        except AttributeError:
            self._dimension = getattr(self.sym_entry, 'dimension', None)
            return self._dimension

    @dimension.setter
    def dimension(self, dimension: int | None):
        self._dimension = dimension

    @property
    def cryst_record(self) -> str | None:
        """Return the symmetry parameters as a CRYST1 entry"""
        # Todo should we always use a generated _cryst_record? If read from file, but a Nanohedra based cryst was made
        #  then it would be wrong since it wouldn't be used
        try:
            return self._cryst_record
        except AttributeError:  # for now don't use if the structure wasn't symmetric and no attribute was parsed
            self._cryst_record = None if not self.symmetry or self.dimension == 0 \
                else generate_cryst1_record(self.uc_dimensions, self.symmetry)
            return self._cryst_record

    @cryst_record.setter
    def cryst_record(self, cryst_record: str | None):
        self._cryst_record = cryst_record

    # @property
    # def uc_dimensions(self) -> list[float]:
    #     try:
    #         return self._uc_dimensions
    #     except AttributeError:
    #         self._uc_dimensions = list(self.cryst['a_b_c']) + list(self.cryst['ang_a_b_c'])
    #         return self._uc_dimensions
    #
    # @uc_dimensions.setter
    # def uc_dimensions(self, dimensions: list[float]):
    #     self._uc_dimensions = dimensions

    @property
    def number_of_symmetry_mates(self) -> int:
        """Describes the number of symmetric copies present in the coordinates"""
        try:
            return self._number_of_symmetry_mates
        except AttributeError:
            self._number_of_symmetry_mates = getattr(self.sym_entry, 'number_of_operations', 1)
            return self._number_of_symmetry_mates

    @number_of_symmetry_mates.setter
    def number_of_symmetry_mates(self, number_of_symmetry_mates: int):
        self._number_of_symmetry_mates = number_of_symmetry_mates

    @property
    def number_of_uc_symmetry_mates(self) -> int:
        """Describes the number of symmetry mates present in the unit cell"""
        try:
            return space_group_number_operations[self.symmetry]
        except KeyError:
            raise SymmetryError(f'The symmetry "{self.symmetry}" is not an available unit cell at this time. If this is'
                                f' a point group, adjust your code, otherwise, help expand the code to include the '
                                f'symmetry operators for this symmetry group')

    # @number_of_uc_symmetry_mates.setter
    # def number_of_uc_symmetry_mates(self, number_of_uc_symmetry_mates):
    #     self._number_of_uc_symmetry_mates = number_of_uc_symmetry_mates

    @property
    def atom_indices_per_entity_symmetric(self):
        # Todo make Structure .atom_indices a numpy array
        #  Need to modify delete_residue and insert residue ._atom_indices attribute access
        # alt solution may be quicker by performing the following addition then .flatten()
        # broadcast entity_indices ->
        # (np.arange(model_number) * number_of_atoms).T
        # |
        # v
        number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        return [[idx + (number_of_atoms * model_number) for model_number in range(self.number_of_symmetry_mates)
                 for idx in entity_indices] for entity_indices in self.atom_indices_per_entity]

    @property
    def asu_coords(self) -> np.ndarray:
        """Return a view of the ASU Coords"""
        return self._coords.coords[self.asu_indices]

    @asu_coords.setter
    def asu_coords(self, coords: Coords):
        self.coords = coords
        # set the symmetric coords according to the ASU
        self.generate_symmetric_coords()
        # Todo delete any saved attributes from the SymmetricModel
        #  self._asu_model_idx
        #  self._oligomeric_model_indices

    @property
    def asu_indices(self) -> slice:  # list[int]
        """Return the ASU indices"""
        # Todo Always the same as _atom_indices due to sym/coords nature. Save slice mechanism, remove overhead!
        try:
            return self._asu_indices
        except AttributeError:
            self._asu_indices = self.get_asu_atom_indices(as_slice=True)
            return self._asu_indices

    @property
    def symmetric_coords(self) -> np.ndarray:
        """Return a view of the symmetric Coords"""
        return self._coords.coords

    @symmetric_coords.setter
    def symmetric_coords(self, coords: Coords):
        self._coords = coords

    @property
    def symmetric_coords_split(self) -> list[np.ndarray]:
        """A view of the symmetric coords split at different symmetric models"""
        try:
            return self._symmetric_coords_split
        except AttributeError:
            self._symmetric_coords_split = np.split(self.symmetric_coords, self.number_of_symmetry_mates)
            #                     np.array()  # seems costly
            return self._symmetric_coords_split

    @property
    def symmetric_coords_split_by_entity(self) -> list[list[np.ndarray]]:
        """A view of the symmetric coords split for each symmetric model by the Pose Entity indices"""
        try:
            return self._symmetric_coords_split_by_entity
        except AttributeError:
            symmetric_coords_split = self.symmetric_coords_split
            self._symmetric_coords_split_by_entity = []
            for entity_indices in self.atom_indices_per_entity:
                # self._symmetric_coords_split_by_entity.append(symmetric_coords_split[:, entity_indices])
                self._symmetric_coords_split_by_entity.append([symmetric_split[entity_indices]
                                                               for symmetric_split in symmetric_coords_split])

            return self._symmetric_coords_split_by_entity

    @property
    def symmetric_coords_by_entity(self) -> list[np.ndarray]:
        """A view of the symmetric coords for each Entity in order of the Pose Entity indices"""
        try:
            return self._symmetric_coords_by_entity
        except AttributeError:
            self._symmetric_coords_by_entity = []
            for entity_indices in self.atom_indices_per_entity_symmetric:
                self._symmetric_coords_by_entity.append(self.symmetric_coords[entity_indices])

            return self._symmetric_coords_by_entity

    @property
    def center_of_mass_symmetric(self) -> np.ndarray:
        """The center of mass for the entire symmetric system"""
        # number_of_symmetry_atoms = len(self.symmetric_coords)
        # return np.matmul(np.full(number_of_symmetry_atoms, 1 / number_of_symmetry_atoms), self.symmetric_coords)
        # v since all symmetry by expand_matrix anyway
        return self.center_of_mass_symmetric_models.mean(axis=-2)

    @property
    def center_of_mass_symmetric_models(self) -> np.ndarray:
        """The individual centers of mass for each model in the symmetric system"""
        # number_of_atoms = self.number_of_atoms
        # return np.matmul(np.full(number_of_atoms, 1 / number_of_atoms), self.symmetric_coords_split)
        return np.matmul(self.center_of_mass, self.expand_matrices)

    @property
    def center_of_mass_symmetric_entities(self) -> list[np.ndarray]:
        """The individual centers of mass for each Entity in the symmetric system"""
        # if self.symmetry:
        # self._center_of_mass_symmetric_entities = []
        # for num_atoms, entity_coords in zip(self.number_of_atoms_per_entity, self.symmetric_coords_split_by_entity):
        #     self._center_of_mass_symmetric_entities.append(np.matmul(np.full(num_atoms, 1 / num_atoms),
        #                                                              entity_coords))
        # return self._center_of_mass_symmetric_entities
        return [np.matmul(entity.center_of_mass, self.expand_matrices) for entity in self.entities]

    @property
    def assembly(self) -> Structure:
        """Provides the Structure object containing all symmetric chains in the assembly unless the design is 2- or 3-D
        then the assembly only contains the contacting models"""
        try:
            return self._assembly
        except AttributeError:
            if self.dimension > 0:
                self._assembly = self.assembly_minimally_contacting
            else:
                if not self.models:
                    self.generate_assembly_symmetry_models()
                chains = []
                for model in self.models:
                    chains.extend(model.chains)
                self._assembly = PDB.from_chains(chains, name='assembly', log=self.log, entities=False,
                                                 biomt_header=self.format_biomt(), cryst_record=self.cryst_record)
            return self._assembly

    @property
    def assembly_minimally_contacting(self) -> Structure:  # Todo reconcile mechanism with Entity.oligomer
        """Provides the Structure object only containing the Symmetric Models contacting the ASU"""
        try:
            return self._assembly_minimally_contacting
        except AttributeError:
            if not self.models:
                self.generate_assembly_symmetry_models()  # defaults to surrounding_uc generation
            # only return contacting
            interacting_model_indices = self.get_asu_interaction_model_indices()
            self.log.debug(f'Found selected models {interacting_model_indices} for assembly')

            chains = []
            for idx in [0] + interacting_model_indices:  # add the ASU to the model first
                chains.extend(self.models[idx].chains)
            self._assembly_minimally_contacting = \
                PDB.from_chains(chains, name='assembly', log=self.log, biomt_header=self.format_biomt(),
                                cryst_record=self.cryst_record, entities=False)
            return self._assembly_minimally_contacting

    def generate_symmetric_coords(self, surrounding_uc: bool = True):
        """Expand the asu using self.symmetry for the symmetry specification, and optional unit cell dimensions if
        self.dimension > 0. Expands assembly to complete point group, unit cell, or surrounding unit cells

        Args:
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        # if not self.symmetry:
        #     raise SymmetryError(f'{self.generate_symmetric_coords.__name__}: No symmetry set for {self.name}!')

        if self.dimension == 0:
            # self.generate_point_group_coords(**kwargs)

    # def generate_point_group_coords(self, **kwargs):  # return_side_chains=True,
    #     """Find the coordinates of the symmetry mates using the coordinates and the input expansion matrices
    #
    #     Sets:
    #         self.number_of_symmetry_mates (int)
    #         self.symmetric_coords (Coords)
    #     """
        # if return_side_chains:  # get different function calls depending on the return type # todo
        #     # get_pdb_coords = getattr(PDB, 'coords')
        #     self.coords_type = 'all'
        # else:
        #     # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
        #     self.coords_type = 'bb_cb'

        # self.number_of_symmetry_mates = valid_subunit_number[self.symmetry]
            symmetric_coords = Coords((np.matmul(np.tile(self.coords, (self.number_of_symmetry_mates, 1, 1)),
                                                 self.expand_matrices) + self.expand_translations).reshape(-1, 3))
        # number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        # model_coords = np.empty((number_of_atoms * self.number_of_symmetry_mates, 3), dtype=float)
        # for idx, rotation in enumerate(self.expand_matrices):
        #     model_coords[idx * number_of_atoms: (idx + 1) * number_of_atoms] = \
        #         np.matmul(self.coords, np.transpose(rotation))
        # self.symmetric_coords = Coords(model_coords)

    # def generate_lattice_coords(self, surrounding_uc: bool = True, **kwargs):  # return_side_chains=True
    #     """Generates unit cell coordinates for a symmetry group. Modifies model_coords to include all in the unit cell
    #
    #     Args:
    #         surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
    #     Sets:
    #         self.number_of_symmetry_mates (int)
    #         self.symmetric_coords (Coords)
    #     """
        # if return_side_chains:  # get different function calls depending on the return type  # todo
        #     # get_pdb_coords = getattr(PDB, 'coords')
        #     self.coords_type = 'all'
        # else:
        #     # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
        #     self.coords_type = 'bb_cb'
        else:
            if surrounding_uc:
                shift_3d = [0., 1., -1.]
                if self.dimension == 3:
                    z_shifts, uc_number = shift_3d, 27
                elif self.dimension == 2:
                    z_shifts, uc_number = [0.], 9
                else:
                    raise SymmetryError(f'The specified dimension "{self.dimension}" is not crystalline')

                # set the number_of_symmetry_mates to account for the unit cell number
                self.number_of_symmetry_mates = self.number_of_uc_symmetry_mates * uc_number
                uc_frac_coords = self.return_unit_cell_coords(self.coords, fractional=True)
                surrounding_frac_coords = \
                    np.concatenate([uc_frac_coords + [x, y, z] for x in shift_3d for y in shift_3d for z in z_shifts])
                symmetric_coords = self.frac_to_cart(surrounding_frac_coords)
            else:
                # must set number_of_symmetry_mates before self.return_unit_cell_coords as it relies on copy number
                # self.number_of_symmetry_mates = self.number_of_uc_symmetry_mates
                # uc_number = 1
                symmetric_coords = self.return_unit_cell_coords(self.coords)

        self.symmetric_coords = Coords(symmetric_coords)

    def cart_to_frac(self, cart_coords: np.ndarray | Iterable | int | float) -> np.ndarray:
        """Return fractional coordinates from cartesian coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            cart_coords: The cartesian coordinates of a unit cell
        Returns:
            The fractional coordinates of a unit cell
        """
        if self.uc_dimensions is None:
            raise ValueError('Can\'t manipulate unit cell, no unit cell dimensions were passed')

        degree_to_radians = pi / 180.
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= degree_to_radians
        beta *= degree_to_radians
        gamma *= degree_to_radians

        # unit cell volume
        a_cos = cos(alpha)
        b_cos = cos(beta)
        g_cos = cos(gamma)
        g_sin = sin(gamma)
        v = a * b * c * sqrt(1 - a_cos ** 2 - b_cos ** 2 - g_cos ** 2 + 2 * (a_cos * b_cos * g_cos))

        # deorthogonalization matrix m
        m0 = [1 / a, -(g_cos / float(a * g_sin)),
              (((b * g_cos * c * (a_cos - (b_cos * g_cos))) / float(g_sin)) - (b * c * b_cos * g_sin)) * (1 / float(v))]
        m1 = [0, 1 / (b * g_sin), -((a * c * (a_cos - (b_cos * g_cos))) / float(v * g_sin))]
        m2 = [0, 0, (a * b * g_sin) / float(v)]
        m = [m0, m1, m2]

        return np.matmul(cart_coords, np.transpose(m))

    def frac_to_cart(self, frac_coords: np.ndarray | Iterable | int | float) -> np.ndarray:
        """Return cartesian coordinates from fractional coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            frac_coords: The fractional coordinates of a unit cell
        Returns:
            The cartesian coordinates of a unit cell
        """
        if self.uc_dimensions is None:
            raise ValueError('Can\'t manipulate unit cell, no unit cell dimensions were passed')

        degree_to_radians = pi / 180.
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= degree_to_radians
        beta *= degree_to_radians
        gamma *= degree_to_radians

        # unit cell volume
        a_cos = cos(alpha)
        b_cos = cos(beta)
        g_cos = cos(gamma)
        g_sin = sin(gamma)
        v = a * b * c * sqrt(1 - a_cos**2 - b_cos**2 - g_cos**2 + 2 * (a_cos * b_cos * g_cos))

        # orthogonalization matrix m_inv
        m_inv_0 = [a, b * g_cos, c * b_cos]
        m_inv_1 = [0, b * g_sin, (c * (a_cos - (b_cos * g_cos))) / float(g_sin)]
        m_inv_2 = [0, 0, v / float(a * b * g_sin)]
        m_inv = [m_inv_0, m_inv_1, m_inv_2]

        return np.matmul(frac_coords, np.transpose(m_inv))

    def get_assembly_symmetry_models(self, **kwargs) -> list[Structure]:
        """Return symmetry mates as a collection of Structures with symmetric coordinates

        Keyword Args:
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            All symmetry mates where Chain names match the ASU
        """
        if self.number_of_symmetry_mates != self.number_of_models:  # we haven't generated symmetry models
            self.generate_assembly_symmetry_models(**kwargs)
            if self.number_of_symmetry_mates != self.number_of_models:
                raise SymmetryError(f'{self.get_assembly_symmetry_models.__name__}: The assembly couldn\'t be '
                                    f'returned')

        return self.models

    def generate_assembly_symmetry_models(self, surrounding_uc: bool = True, **kwargs):
        # , return_side_chains=True):
        """Generate symmetry mates as a collection of Structures with symmetric coordinates

        Args:
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Sets:
            self.models (list[Structure]): All symmetry mates where each mate has Chain names matching the ASU
        """
        if not self.symmetry:
            # self.log.critical('%s: No symmetry set for %s! Cannot get symmetry mates'  # Todo
            #                   % (self.generate_assembly_symmetry_models.__name__, self.name))
            raise SymmetryError(f'{self.generate_assembly_symmetry_models.__name__}: No symmetry set for {self.name}! '
                                f'Cannot get symmetry mates')
        # if return_side_chains:  # get different function calls depending on the return type
        #     extract_pdb_atoms = getattr(PDB, 'atoms')
        # else:
        #     extract_pdb_atoms = getattr(PDB, 'backbone_and_cb_atoms')

        # prior_idx = self.asu.number_of_atoms
        # if self.dimension > 0:
        #     number_of_models = self.number_of_symmetry_mates
        # else:  # layer or space group
        if surrounding_uc:  # if the surrounding_uc is requested, we might need to generate it
            if self.number_of_symmetry_mates == self.number_of_uc_symmetry_mates:  # ensure surrounding coords exist
                self.generate_symmetric_coords(surrounding_uc=surrounding_uc)
                # raise SymmetryError('Cannot return the surrounding unit cells as no coordinates were generated '
                #                     f'for them. Try passing surrounding_uc=True to '
                #                     f'{self.generate_symmetric_coords.__name__}')
        # else:
        # number_of_models = self.number_of_symmetry_mates

        number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        for coord_idx in range(self.number_of_symmetry_mates):
            self.log.critical(f'Ensure the output of symmetry mate creation is correct. The copy of a '
                              f'{type(self).__name__} is being taken which is probably relying on PDB.__copy__ or '
                              f'Structure.__copy__. These may not be adequate and need to be overwritten')
            symmetry_mate_pdb = copy(self)
            # old-style
            # symmetry_mate_pdb.replace_coords(self.symmetric_coords[(coord_idx * number_of_atoms):
            #                                                        ((coord_idx + 1) * number_of_atoms)])
            # new-style
            symmetry_mate_pdb.coords = self.symmetric_coords[(coord_idx * number_of_atoms):
                                                             ((coord_idx + 1) * number_of_atoms)]
            self.models.append(symmetry_mate_pdb)

    @property
    def asu_model_index(self) -> int:
        """The asu equivalent model in the SymmetricModel. Zero-indexed"""
        try:
            # if self._asu_model_idx:  # we already found this information
            #     self.log.debug('Skipping ASU identification as information already exists')
            #     return
            return self._asu_model_idx
        except AttributeError:
            template_residue = self.n_terminal_residue
            atom_ca_coord, atom_idx = template_residue.ca_coords, template_residue.ca_atom_index
            # entity1_number, entity2_number = self.number_of_residues_per_entity
            # entity2_n_term_residue_idx = entity1_number + 1
            # entity2_n_term_residue = self.residues[entity2_n_term_residue_idx]
            # entity2_ca_idx = entity2_n_term_residue.ca_atom_index
            number_of_atoms = self.number_of_atoms
            # number_of_atoms = len(self.coords)
            for model_idx in range(self.number_of_symmetry_mates):
                if np.allclose(atom_ca_coord, self.symmetric_coords[(model_idx * number_of_atoms) + atom_idx]):
                    # if (atom_ca_coord ==
                    #         self.symmetric_coords[(model_idx * number_of_atoms) + atom_idx]).all():
                    self._asu_model_idx = model_idx
                    return self._asu_model_idx

            self.log.error(f'FAILED to find {self.asu_model_index.__name__}')

    @property
    def oligomeric_model_indices(self) -> dict[Entity, list[int]] | dict:
        try:
            return self._oligomeric_model_indices
        except AttributeError:
            self._find_oligomeric_model_indices()
            return self._oligomeric_model_indices

    def _find_oligomeric_model_indices(self, epsilon: float = 0.5):
        """From an Entity's Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass

        Args:
            epsilon: The distance measurement tolerance to find similar symmetric models to the oligomer
        """
        number_of_atoms = self.number_of_atoms
        for entity in self.entities:
            if not entity.is_oligomeric():
                self._oligomeric_model_indices[entity] = []
                continue
            # need to slice through the specific Entity coords once we have the model
            entity_indices = entity.atom_indices
            entity_start, entity_end = entity_indices[0], entity_indices[-1]
            entity_length = entity.number_of_atoms
            entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
            equivalent_models = []
            for chain in entity.chains:
                # chain_length = chain.number_of_atoms
                # chain_center_of_mass = np.matmul(np.full(chain_length, 1 / chain_length), chain.coords)
                chain_center_of_mass = chain.center_of_mass
                # print('Chain', chain_center_of_mass.astype(int))
                for model_num in range(self.number_of_symmetry_mates):  # Todo modify below to be with symmetric coms
                    sym_model_center_of_mass = \
                        np.matmul(entity_center_of_mass_divisor,
                                  self.symmetric_coords[(model_num * number_of_atoms) + entity_start:
                                                        (model_num * number_of_atoms) + entity_end + 1])
                    #                                             have to add 1 for slice ^
                    # print('Sym Model', sym_model_center_of_mass)
                    # if np.allclose(chain_center_of_mass.astype(int), sym_model_center_of_mass.astype(int)):
                    # if np.allclose(chain_center_of_mass, sym_model_center_of_mass):  # using np.rint()
                    if np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass) < epsilon:
                        equivalent_models.append(model_num)
                        break

            if len(equivalent_models) != len(entity.chains):
                raise SymmetryError(f'The number of equivalent models ({len(equivalent_models)}) '
                                    f'!= the number of chains ({len(entity.chains)})')

            self._oligomeric_model_indices[entity] = equivalent_models

    def get_asu_interaction_model_indices(self, calculate_contacts: bool = True, distance: float = 8., **kwargs) ->\
            list[int]:
        """From an ASU, find the symmetric models that immediately surround the ASU

        Args:
            calculate_contacts: Whether to calculate interacting models by atomic contacts
            distance: When calculate_contacts is True, the CB distance which nearby symmetric models should be found
                When calculate_contacts is False, uses the ASU radius plus the maximum Entity radius
        Returns:
            The indices of the models that contact the asu
        """
        if calculate_contacts:
            # Select only coords that are BB or CB from the model coords
            # bb_cb_indices = None if self.coords_type == 'bb_cb' else self.backbone_and_cb_indices
            bb_cb_indices = self.backbone_and_cb_indices
            self.generate_assembly_tree()
            asu_query = self.assembly_tree.query_radius(self.coords[bb_cb_indices], distance)
            # coords_length = len(bb_cb_indices)
            # contacting_model_indices = [assembly_idx // coords_length
            #                             for asu_idx, assembly_contacts in enumerate(asu_query)
            #                             for assembly_idx in assembly_contacts]
            # interacting_models = sorted(set(contacting_model_indices))
            # combine each subarray of the asu_query and divide by the assembly_tree interval length len(asu_query)
            interacting_models = (np.unique(np.concatenate(asu_query) // len(asu_query)) + 1).tolist()
            # asu is missing from assembly_tree so add 1 model to total symmetric index  ^
        else:
            # distance = self.asu.radius * 2  # value too large self.radius * 2
            # The furthest point from the ASU COM + the max individual Entity radius
            distance = self.radius + max([entity.radius for entity in self.entities])  # all the radii
            center_of_mass = self.center_of_mass
            interacting_models = [idx for idx, sym_model_com in enumerate(self.center_of_mass_symmetric_models)
                                  if np.linalg.norm(center_of_mass - sym_model_com) <= distance]
            # print('interacting_models com', self.center_of_mass_symmetric_models[interacting_models])

        return interacting_models

    def get_asu_atom_indices(self, as_slice: bool = False) -> list[int] | slice:
        """Find the coordinate indices of the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            The indices in the SymmetricModel where the ASU is also located
        """
        asu_model_idx = self.asu_model_index
        number_of_atoms = self.number_of_atoms
        start_idx = number_of_atoms * asu_model_idx
        end_idx = number_of_atoms * (asu_model_idx + 1)

        if as_slice:
            return slice(start_idx, end_idx)
        else:
            return list(range(start_idx, end_idx))

    def get_oligomeric_atom_indices(self, entity: Entity) -> list[int]:
        """Find the coordinate indices of the intra-oligomeric equivalent models in the SymmetricModel. Zero-indexed

        Args:
            entity: The Entity with oligomeric chains to query for corresponding symmetry mates
        Returns:
            The indices in the SymmetricModel where the intra-oligomeric contacts are located
        """
        number_of_atoms = self.number_of_atoms
        oligomeric_atom_indices = []
        for model_number in self.oligomeric_model_indices.get(entity):
            # start_idx = number_of_atoms * model_number
            # end_idx = number_of_atoms * (model_number + 1)
            oligomeric_atom_indices.extend(list(range(number_of_atoms * model_number,
                                                      number_of_atoms * (model_number + 1))))

        return oligomeric_atom_indices

    def get_asu_interaction_indices(self, **kwargs) -> list[int]:
        """Find the coordinate indices for the models in the SymmetricModel interacting with the asu. Zero-indexed

        Keyword Args:
            calculate_contacts=True (bool): Whether to calculate interacting models by atomic contacts
            distance=8.0 (float): When calculate_contacts is True, the CB distance which nearby symmetric models should be found
                When calculate_contacts is False, uses the ASU radius plus the maximum Entity radius
        Returns:
            The indices in the SymmetricModel where the asu contacts other models
        """
        model_numbers = self.get_asu_interaction_model_indices(**kwargs)
        interacting_indices = []
        number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        for model_number in model_numbers:
            start_idx = number_of_atoms * model_number
            end_idx = number_of_atoms * (model_number + 1)
            interacting_indices.extend(list(range(start_idx, end_idx)))

        return interacting_indices

    def return_symmetric_copies(self, structure: Structure, return_side_chains: bool = True,
                                surrounding_uc: bool = True, **kwargs) -> list[Structure]:
        """Expand the provided Structure using self.symmetry for the symmetry specification

        Args:
            structure: A Structure object with .coords/.get_backbone_and_cb_coords() methods
            return_side_chains: Whether to make the structural copy with side chains
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            The symmetric copies of the input structure
        """
        self.log.critical(f'Ensure the output of symmetry mate creation is correct. The copy of a '
                          f'{type(self).__name__} is being taken which is probably relying on PDB.__copy__ or '
                          f'Structure.__copy__. These may not be adequate and need to be overwritten')
        # Caution, this function will return poor if the number of atoms in the structure is 1!
        coords = structure.coords if return_side_chains else structure.get_backbone_and_cb_coords()

        if self.dimension == 0:
            # return self.return_point_group_copies(structure, **kwargs)
            number_of_symmetry_mates = self.number_of_symmetry_mates
            # favoring this as it is more explicit
            sym_coords = (np.matmul(np.tile(coords, (self.number_of_symmetry_mates, 1, 1)),
                                    self.expand_matrices) + self.expand_translations).reshape(-1, 3)
            # coords_length = sym_coords.shape[1]
            # sym_mates = []
            # for model_num in range(self.number_of_symmetry_mates):
            #     symmetry_mate_pdb = copy(structure)
            #     symmetry_mate_pd.replace_coords(sym_coords[model_num * coords_length:(model_num + 1) * coords_length])
            #     sym_mates.append(symmetry_mate_pdb)
            # return sym_mates
        else:
            # return self.return_lattice_copies(structure, **kwargs)
            if surrounding_uc:
                # return self.return_surrounding_unit_cell_symmetry_mates(structure, **kwargs)  # return_side_chains
                shift_3d = [0., 1., -1.]
                if self.dimension == 3:
                    z_shifts, uc_number = shift_3d, 27
                elif self.dimension == 2:
                    z_shifts, uc_number = [0.], 9
                else:
                    raise SymmetryError(f'The specified dimension "{self.dimension}" is not crystalline')

                number_of_symmetry_mates = self.number_of_uc_symmetry_mates * uc_number
                uc_frac_coords = self.return_unit_cell_coords(coords, fractional=True)
                surrounding_frac_coords = \
                    np.concatenate([uc_frac_coords + [x, y, z] for x in shift_3d for y in shift_3d for z in z_shifts])
                sym_coords = self.frac_to_cart(surrounding_frac_coords)
            else:
                number_of_symmetry_mates = self.number_of_uc_symmetry_mates
                sym_coords = self.return_unit_cell_coords(coords)

        # coords_length = coords.shape[0]
        sym_mates = []
        for coord_set in np.split(sym_coords, number_of_symmetry_mates):  # uc_number):
            # for model_num in range(self.number_of_symmetry_mates):
            symmetry_mate_pdb = copy(structure)
            # old-style
            # symmetry_mate_pdb.replace_coords(coord_set)  # [model_num * coords_length:(model_num + 1) * coords_length])
            # new-style
            symmetry_mate_pdb.coords = coord_set
            sym_mates.append(symmetry_mate_pdb)

        assert len(sym_mates) == uc_number * self.number_of_symmetry_mates, \
            f'Number of models ({len(sym_mates)}) is incorrect! ' \
            f'Should be {uc_number * self.number_of_uc_symmetry_mates}'
        return sym_mates

    # def return_point_group_copies(self, structure: Structure, return_side_chains: bool = True, **kwargs) -> \
    #         list[Structure]:
    #     """Expand the coordinates for every symmetric copy within the point group assembly
    #
    #     Args:
    #         structure: A Structure containing some collection of Residues
    #         return_side_chains: Whether to make the structural copy with side chains
    #     Returns:
    #         The symmetric copies of the input structure
    #     """
    #     # Caution, this function will return poor if the number of atoms in the structure is 1!
    #     coords = structure.coords if return_side_chains else structure.get_backbone_and_cb_coords()
    #     # Favoring this alternative way as it is more explicit
    #     coord_set = (np.matmul(np.tile(coords, (self.number_of_symmetry_mates, 1, 1)),
    #                            self.expand_matrices) + self.expand_translations).reshape(-1, 3)
    #     coords_length = coord_set.shape[1]
    #     sym_mates = []
    #     for model_num in range(self.number_of_symmetry_mates):
    #         symmetry_mate_pdb = copy(structure)
    #         symmetry_mate_pdb.replace_coords(coord_set[model_num * coords_length:(model_num + 1) * coords_length])
    #         sym_mates.append(symmetry_mate_pdb)
    #     return sym_mates
    #
    # def return_lattice_copies(self, structure: Structure, surrounding_uc: bool = True, return_side_chains: bool = True,
    #                           **kwargs) -> list[Structure]:
    #     """Expand the coordinates for every symmetric copy within the unit cell
    #
    #     Args:
    #         structure: A Structure containing some collection of Residues
    #         surrounding_uc: Whether to return the surrounding unit cells along with the central unit cell
    #         return_side_chains: Whether to make the structural copy with side chains
    #     Returns:
    #         The symmetric copies of the input structure
    #     """
    #     # Caution, this function will return poor if the number of atoms in the structure is 1!
    #     coords = structure.coords if return_side_chains else structure.get_backbone_and_cb_coords()
    #
    #     if surrounding_uc:
    #         # return self.return_surrounding_unit_cell_symmetry_mates(structure, **kwargs)  # return_side_chains
    #         shift_3d = [0., 1., -1.]
    #         if self.dimension == 3:
    #             z_shifts, uc_number = shift_3d, 27
    #         elif self.dimension == 2:
    #             z_shifts, uc_number = [0.], 9
    #         else:
    #             raise SymmetryError(f'The specified dimension "{self.dimension}" is not crystalline')
    #
    #         uc_frac_coords = self.return_unit_cell_coords(coords, fractional=True)
    #         surrounding_frac_coords = np.concatenate([uc_frac_coords + [x, y, z] for x in shift_3d for y in shift_3d
    #                                                   for z in z_shifts])
    #         sym_coords = self.frac_to_cart(surrounding_frac_coords)
    #     else:
    #         uc_number = 1
    #         sym_coords = self.return_unit_cell_coords(coords)
    #
    #     coords_length = coords.shape[0]
    #     sym_mates = []
    #     for coord_set in np.split(sym_coords, uc_number):
    #         for model_num in range(self.number_of_symmetry_mates):
    #             symmetry_mate_pdb = copy(structure)
    #             symmetry_mate_pdb.replace_coords(coord_set[model_num * coords_length:(model_num + 1) * coords_length])
    #             sym_mates.append(symmetry_mate_pdb)
    #
    #     assert len(sym_mates) == uc_number * self.number_of_uc_symmetry_mates, \
    #         f'Number of models ({len(sym_mates)}) is incorrect! ' \
    #         f'Should be {uc_number * self.number_of_uc_symmetry_mates}'
    #     return sym_mates

    def return_symmetric_coords(self, coords: list | np.ndarray, surrounding_uc: bool = True) -> np.ndarray:
        """Provided an input set of coordinates, return the symmetrized coordinates corresponding to the SymmetricModel

        Args:
            coords: The coordinates to symmetrize
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            The symmetrized coordinates
        """
        if self.dimension == 0:
            # coords_len = 1 if not isinstance(coords[0], (list, np.ndarray)) else len(coords)
            # model_coords = np.empty((coords_length * self.number_of_symmetry_mates, 3), dtype=float)
            # for idx, rotation in enumerate(self.expand_matrices):
            #     model_coords[idx * coords_len: (idx + 1) * coords_len] = np.matmul(coords, np.transpose(rotation))

            return (np.matmul(np.tile(coords, (self.number_of_symmetry_mates, 1, 1)),
                              self.expand_matrices) + self.expand_translations).reshape(-1, 3)
        else:
            if surrounding_uc:
                shift_3d = [0., 1., -1.]
                if self.dimension == 3:
                    z_shifts = shift_3d
                elif self.dimension == 2:
                    z_shifts = [0.]
                else:
                    raise SymmetryError(f'The specified dimension "{self.dimension}" is not crystalline')

                uc_frac_coords = self.return_unit_cell_coords(coords, fractional=True)
                surrounding_frac_coords = \
                    np.concatenate([uc_frac_coords + [x, y, z] for x in shift_3d for y in shift_3d for z in z_shifts])
                return self.frac_to_cart(surrounding_frac_coords)
            else:
                # must set number_of_symmetry_mates before self.return_unit_cell_coords as it relies on copy number
                # self.number_of_symmetry_mates = self.number_of_uc_symmetry_mates
                # uc_number = 1
                return self.return_unit_cell_coords(coords)

    def return_unit_cell_coords(self, coords: np.ndarray, fractional: bool = False) -> np.ndarray:
        """Return the unit cell coordinates from a set of coordinates for the specified SymmetricModel

        Args:
            coords: The cartesian coordinates to expand to the unit cell
            fractional: Whether to return coordinates in fractional or cartesian (False) unit cell frame
        Returns:
            All unit cell coordinates
        """
        # asu_frac_coords = self.cart_to_frac(coords)
        model_coords = (np.matmul(np.tile(self.cart_to_frac(coords), (self.number_of_uc_symmetry_mates, 1, 1)),
                                  self.expand_matrices) + self.expand_translations).reshape(-1, 3)
        # coords_length = 1 if not isinstance(coords[0], (list, np.ndarray)) else len(coords)
        # model_coords = np.empty((coords_length * self.number_of_uc_symmetry_mates, 3), dtype=float)
        # model_coords[:coords_length] = asu_frac_coords
        # for idx, (rotation, translation) in enumerate(self.expand_matrices, 1):  # since no identity, start idx at 1
        #     model_coords[idx * coords_length: (idx + 1) * coords_length] = \
        #         np.matmul(asu_frac_coords, np.transpose(rotation)) + translation

        if fractional:
            return model_coords
        else:
            return self.frac_to_cart(model_coords)

    def assign_entities_to_sub_symmetry(self):
        """From a symmetry entry, find the entities which belong to each sub-symmetry (the component groups) which make
        the global symmetry. Construct the sub-symmetry by copying each symmetric chain to the Entity's .chains
        attribute"""
        raise NotImplementedError('Cannot assign entities to sub symmetry yet! Need to debug this function')
        if not self.symmetry:
            raise SymmetryError('Must set a global symmetry to assign entities to sub symmetry!')

        # Get the rotation matrices for each group then orient along the setting matrix "axis"
        if self.sym_entry.group1 in ['D2', 'D3', 'D4', 'D6'] or self.sym_entry.group2 in ['D2', 'D3', 'D4', 'D6']:
            group1 = self.sym_entry.group1.replace('D', 'C')
            group2 = self.sym_entry.group2.replace('D', 'C')
            rotation_matrices_only1 = get_rot_matrices(rotation_range[group1], 'z', 360)
            rotation_matrices_only2 = get_rot_matrices(rotation_range[group2], 'z', 360)
            # provide a 180 degree rotation along x (all D orient symmetries have axis here)
            flip_x = [identity_matrix, flip_x_matrix]
            rotation_matrices_group1 = make_rotations_degenerate(rotation_matrices_only1, flip_x)
            rotation_matrices_group2 = make_rotations_degenerate(rotation_matrices_only2, flip_x)
            # group_set_rotation_matrices = {1: np.matmul(degen_rot_mat_1, np.transpose(set_mat1)),
            #                                2: np.matmul(degen_rot_mat_2, np.transpose(set_mat2))}
            raise DesignError('Using dihedral symmetry has not been implemented yet! It is required to change the code'
                              ' before continuing with design of symmetry entry %d!' % self.sym_entry.entry_number)
        else:
            group1 = self.sym_entry.group1
            group2 = self.sym_entry.group2
            rotation_matrices_group1 = get_rot_matrices(rotation_range[group1], 'z', 360)  # np.array (rotations, 3, 3)
            rotation_matrices_group2 = get_rot_matrices(rotation_range[group2], 'z', 360)

        # Assign each Entity to a symmetry group
        # entity_coms = [entity.center_of_mass for entity in self.asu]
        # all_entities_com = np.matmul(np.full(len(entity_coms), 1 / len(entity_coms)), entity_coms)
        all_entities_com = self.center_of_mass
        # check if global symmetry is centered at the origin. If not, translate to the origin with ext_tx
        self.log.debug('The symmetric center of mass is: %s' % str(self.center_of_mass_symmetric))
        if np.isclose(self.center_of_mass_symmetric, origin):  # is this threshold loose enough?
            # the com is at the origin
            self.log.debug('The symmetric center of mass is at the origin')
            ext_tx = origin
            expand_matrices = self.expand_matrices
        else:
            self.log.debug('The symmetric center of mass is NOT at the origin')
            # Todo find ext_tx from input without Nanohedra input
            #  In Nanohedra, the origin will work for many symmetries, I believe all! Given the reliance on
            #  crystallographic tables, their symmetry operations are centered around the lattice point, which I see
            #  only complicating things to move besides the origin
            if self.dimension > 0:
                # Todo we have different set up required here. The expand matrices can be derived from a point group in
                #  the layer or space setting, however we must ensure that the required external tx is respected
                #  (i.e. subtracted) at the required steps such as from coms_group1/2 in return_symmetric_coords
                #  (generated using self.expand_matrices) and/or the entity_com as this is set up within a
                #  cartesian expand matrix environment is going to yield wrong results on the expand matrix indexing
                assert self.number_of_symmetry_mates == self.number_of_uc_symmetry_mates, \
                    'Cannot have more models (%d) than a single unit cell (%d)!' \
                    % (self.number_of_symmetry_mates, self.number_of_uc_symmetry_mates)
                expand_matrices = point_group_symmetry_operators[self.point_group_symmetry]
            else:
                expand_matrices = self.expand_matrices
            ext_tx = self.center_of_mass_symmetric  # only works for unit cell or point group NOT surrounding UC
            # This is typically centered at the origin for the symmetric assembly... NEED rigourous testing.
            # Maybe this route of generation is too flawed for layer/space? Nanohedra framework gives a comprehensive
            # handle on all these issues though

        # find the approximate scalar translation of the asu center of mass from the reference symmetry origin
        approx_entity_com_reference_tx = np.linalg.norm(all_entities_com - ext_tx)
        approx_entity_z_tx = np.array([0., 0., approx_entity_com_reference_tx])
        # apply the setting matrix for each group to the approximate translation
        set_mat1 = self.sym_entry.setting_matrix1
        set_mat2 = self.sym_entry.setting_matrix2
        # TODO test transform_coordinate_sets has the correct input format (numpy.ndarray)
        com_group1 = \
            transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat1, translation2=ext_tx)
        com_group2 = \
            transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat2, translation2=ext_tx)
        # expand the tx'd, setting matrix rot'd, approximate coms for each group using self.expansion operators
        coms_group1 = self.return_symmetric_coords(com_group1)
        coms_group2 = self.return_symmetric_coords(com_group2)

        # measure the closest distance from each entity com to the setting matrix transformed approx group coms to find
        # which group the entity belongs to. Save the group and the operation index of the expansion matrices. With both
        # of these, it is possible to find a new setting matrix that is symmetry equivalent and will generate the
        # correct sub-symmetry symmetric copies for each provided Entity
        group_entity_rot_ops = {1: {}, 2: {}}
        # min_dist1, min_dist2, min_1_entity, min_2_entity = float('inf'), float('inf'), None, None
        for entity in self.entities:
            entity_com = entity.center_of_mass
            min_dist, min_entity_group_operator = float('inf'), None
            for idx in range(len(expand_matrices)):  # has the length of the symmetry operations
                com1_distance = np.linalg.norm(entity_com - coms_group1[idx])
                com2_distance = np.linalg.norm(entity_com - coms_group2[idx])
                if com1_distance < com2_distance:
                    if com1_distance < min_dist:
                        min_dist = com1_distance
                        min_entity_group_operator = (group2, expand_matrices[idx])
                    # # entity_min_group = 1
                    # entity_group_d[1].append(entity)
                else:
                    if com2_distance < min_dist:
                        min_dist = com2_distance
                        min_entity_group_operator = (group2, expand_matrices[idx])
                    # # entity_min_group = 2
                    # entity_group_d[2].append(entity)
            if min_entity_group_operator:
                group, operation = min_entity_group_operator
                group_entity_rot_ops[group][entity] = operation
                # {1: {entity1: [[],[],[]]}, 2: {entity2: [[],[],[]]}}

        set_mat = {1: set_mat1, 2: set_mat2}
        inv_set_matrix = {1: np.linalg.inv(set_mat1), 2: np.linalg.inv(set_mat2)}
        group_rotation_matrices = {1: rotation_matrices_group1, 2: rotation_matrices_group2}
        # Multiplication is not possible in this way apparently!
        # group_set_rotation_matrices = {1: np.matmul(rotation_matrices_group1, np.transpose(set_mat1)),
        #                                2: np.matmul(rotation_matrices_group2, np.transpose(set_mat2))}

        # Apply the rotation matrices to the identified group Entities. First modify the Entity by the inverse expansion
        # and setting matrices to orient along Z axis. Apply the rotation matrix, then reverse operations back to start
        for idx, (group, entity_ops) in enumerate(group_entity_rot_ops.items()):
            for entity, rot_op in entity_ops.items():
                dummy_rotation = False
                dummy_translation = False
                # Todo need to reverse the expansion matrix first to get the entity coords to the "canonical" setting
                #  matrix as expected by Nanohedra. I can then make_oligomers
                entity.make_oligomer(symmetry=group, **dict(rotation=dummy_rotation, translation=dummy_translation,
                                                            rotation2=set_mat[idx], translation2=ext_tx))
                # # Todo if this is a fractional rot/tx pair this won't work
                # #  I converted the space group external tx and design_pg_symmetry to rot_matrices so I should
                # #  test if the change to local point group symmetry in a layer or space group is sufficient
                # inv_expand_matrix = np.linalg.inv(rot_op)
                # inv_rotation_matrix = np.linalg.inv(dummy_rotation)
                # # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
                # # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
                # centered_coords = transform_coordinate_sets(entity.coords, translation=-ext_tx)
                # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_expand_matrix,
                #                                             rotation2=inv_set_matrix[group])
                # TODO                                        NEED DIHEDRAl rotation v back to canonical
                # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_rotation_matrix,
                # TODO                                        as well as v translation (not approx, dihedral won't work)
                #                                             translation=approx_entity_z_tx)
                # # now rotate, then undo symmetry expansion matrices
                # # for rot in group_rotation_matrices[group][1:]:  # exclude the first rotation matrix as it is identity
                # for rot in group_rotation_matrices[group]:
                #     temp_coords = transform_coordinate_sets(sym_on_z_coords, rotation=np.array(rot), rotation2=set_mat[group])
                #     # rot_centered_coords = transform_coordinate_sets(sym_on_z_coords, rotation=rot)
                #     # final_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation,
                #     #                                          translation=translation, <-NEED^ for DIHEDRAL
                #     #                                          rotation2=rotation2, translation2=translation2)
                #     final_coords = transform_coordinate_sets(temp_coords, rotation=rot_op, translation=ext_tx)
                #     # Entity representative stays in the .chains attribute as chain[0] given the iterator slice above
                #     sub_symmetry_mate_pdb = copy(entity.chain_representative)
                #     sub_symmetry_mate_pdb.replace_coords(final_coords)
                #     entity.chains.append(sub_symmetry_mate_pdb)
                #     # need to take the cyclic system generated and somehow transpose it on the dihedral group.
                #     # an easier way would be to grab the assembly from the SymDesignOutput/Data/PDBs and set the
                #     # oligomer onto the ASU. The .chains would then be populated for the non-transposed chains
                #     # if dihedral:  # TODO
                #     #     dummy = True

    def assign_pose_transformation(self) -> list[dict]:
        """Using the symmetry entry and symmetric material, find the specific transformations necessary to establish the
        individual symmetric components in the global symmetry

        Returns:
            The specific transformation dictionaries which place each Entity with proper symmetry axis in the Pose
        """
        if not self.symmetry:
            raise SymmetryError(f'Must set a global symmetry to {self.assign_pose_transformation.__name__}!')

        # get optimal external translation
        if self.dimension == 0:
            external_tx = [None for _ in self.sym_entry.groups]
        else:
            try:
                optimal_external_shifts = self.sym_entry.get_optimal_shift_from_uc_dimensions(*self.uc_dimensions)
            except AttributeError as error:
                print(f'\n\n\n{self.assign_pose_transformation.__name__}: Couldn\'t '
                      f'{SymEntry.get_optimal_shift_from_uc_dimensions.__name__} with dimensions: {self.uc_dimensions}'
                      f'\nAnd sym_entry.unit_cell specification: {self.sym_entry.unit_cell}\nThis is likely because '
                      f'{self.symmetry} isn\'t a lattice with parameterized external translations\n\n\n')
                raise error
            # external_tx1 = optimal_external_shifts[:, None] * self.sym_entry.external_dof1
            # external_tx2 = optimal_external_shifts[:, None] * self.sym_entry.external_dof2
            # external_tx = [external_tx1, external_tx2]
            self.log.critical('This functionality has never been tested! Inspect all outputs before trusting results')
            external_tx = \
                [(optimal_external_shifts[:, None] * getattr(self.sym_entry, f'external_dof{idx}')).sum(axis=-2)
                 for idx, group in enumerate(self.sym_entry.groups, 1)]

        center_of_mass_symmetric_entities = self.center_of_mass_symmetric_entities
        # self.log.critical('center_of_mass_symmetric_entities = %s' % center_of_mass_symmetric_entities)
        transform_solutions = []
        asu_indices = []
        for group_idx, sym_group in enumerate(self.sym_entry.groups):
            # find groups for which the oligomeric parameters do not apply or exist by nature of orientation [T, O, I]
            if sym_group == self.symmetry:  # molecule should be oriented already and expand matrices handle oligomers
                transform_solutions.append(dict())  # rotation=rot, translation=tx
                asu_indices.append(list(range(len(center_of_mass_symmetric_entities[group_idx]))))
                continue
            elif sym_group == 'C1':  # no oligomer possible
                transform_solutions.append(dict())  # rotation=rot, translation=tx
                asu_indices.append(list(range(len(center_of_mass_symmetric_entities[group_idx]))))
                continue
            # search through the sub_symmetry group setting matrices that make up the resulting point group symmetry
            # apply setting matrix to the entity centers of mass indexed to the proper group number
            internal_tx = None
            setting_matrix = None
            entity_asu_indices = None
            group_subunit_number = valid_subunit_number[sym_group]
            current_best_minimal_central_offset = float('inf')
            # sym_group_setting_matrices = point_group_setting_matrix_members[self.point_group_symmetry].get(sym_group)
            for setting_matrix_idx in point_group_setting_matrix_members[self.point_group_symmetry].get(sym_group, []):
                # self.log.critical('Setting_matrix_idx = %d' % setting_matrix_idx)
                temp_model_coms = np.matmul(center_of_mass_symmetric_entities[group_idx],
                                            np.transpose(inv_setting_matrices[setting_matrix_idx]))
                # self.log.critical('temp_model_coms = %s' % temp_model_coms)
                # find groups of COMs with equal z heights
                possible_height_groups = {}
                for idx, com in enumerate(temp_model_coms.round(decimals=2)):  # 2 decimals may be required precision
                    z_coord = com[-1]
                    if z_coord in possible_height_groups:
                        possible_height_groups[z_coord].append(idx)
                    else:
                        possible_height_groups[z_coord] = [idx]
                # find the most centrally disposed, COM grouping with the correct number of COMs in the group
                # not necessarily positive...
                centrally_disposed_group_height = None
                minimal_central_offset = float('inf')
                for height, indices in possible_height_groups.items():
                    # if height < 0:  # this may be detrimental. Increased search cost not worth missing solution
                    #     continue
                    if len(indices) == group_subunit_number:
                        x = (temp_model_coms[indices] - [0, 0, height])[0]  # get first point. Norms are equivalent
                        central_offset = np.sqrt(x.dot(x))  # np.abs()
                        # self.log.debug('central_offset = %f' % central_offset)
                        if central_offset < minimal_central_offset:
                            minimal_central_offset = central_offset
                            centrally_disposed_group_height = height
                            # self.log.debug('centrally_disposed_group_height = %d' % centrally_disposed_group_height)
                        elif central_offset == minimal_central_offset and centrally_disposed_group_height < 0 < height:
                            centrally_disposed_group_height = height
                            # self.log.debug('centrally_disposed_group_height = %d' % centrally_disposed_group_height)
                        else:  # The central offset is larger
                            pass
                # if a viable group was found save the group COM as an internal_tx and setting_matrix used to find it
                if centrally_disposed_group_height is not None:
                    if setting_matrix is not None and internal_tx is not None:
                        # There is an alternative solution. Is it better? Or is it a degeneracy?
                        if minimal_central_offset < current_best_minimal_central_offset:
                            # the new one if it is less offset
                            entity_asu_indices = possible_height_groups[centrally_disposed_group_height]
                            internal_tx = temp_model_coms[entity_asu_indices].mean(axis=-2)
                            setting_matrix = setting_matrices[setting_matrix_idx]
                        elif minimal_central_offset == current_best_minimal_central_offset:
                            # chose the positive one in the case that there are degeneracies (most likely)
                            self.log.info('There are multiple pose transformation solutions for the symmetry group '
                                          '%s (specified in position {%d} of %s). The solution with a positive '
                                          'translation was chosen by convention. This may result in inaccurate behavior'
                                          % (sym_group, group_idx + 1, self.sym_entry.combination_string))
                            if internal_tx[-1] < 0 < centrally_disposed_group_height:
                                entity_asu_indices = possible_height_groups[centrally_disposed_group_height]
                                internal_tx = temp_model_coms[entity_asu_indices].mean(axis=-2)
                                setting_matrix = setting_matrices[setting_matrix_idx]
                        else:  # The central offset is larger
                            pass
                    else:  # these were not set yet
                        entity_asu_indices = possible_height_groups[centrally_disposed_group_height]
                        internal_tx = temp_model_coms[entity_asu_indices].mean(axis=-2)
                        setting_matrix = setting_matrices[setting_matrix_idx]
                        current_best_minimal_central_offset = minimal_central_offset
                else:  # no viable group probably because the setting matrix was wrong. Continue with next
                    pass

            if entity_asu_indices is not None:
                transform_solutions.append(dict(rotation2=setting_matrix, translation2=external_tx[group_idx],
                                                translation=internal_tx))
                asu_indices.append(entity_asu_indices)
            else:
                raise ValueError('Using the supplied Model (%s) and the specified symmetry (%s), there was no solution '
                                 'found for Entity #%d. A possible issue could be that the supplied Model has it\'s '
                                 'Entities out of order for the assumed symmetric entry "%s". If the order of the '
                                 'Entities in the file is different than the provided symmetry please supply the '
                                 'correct order with the symmetry combination format "%s" to the flag --%s. Another '
                                 'possibility is that the symmetry is generated improperly or imprecisely. Please '
                                 'ensure your inputs are symmetrically viable for the desired symmetry'
                                 % (self.name, self.symmetry, group_idx + 1, self.sym_entry.combination_string,
                                    symmetry_combination_format, 'symmetry'))

        # Todo find the particular rotation to orient the Entity oligomer to a cannonical orientation. This must
        #  accompany standards required for the SymDesign Database for actions like refinement

        # this routine uses the same logic at the get_contacting_asu however using the COM of the found
        # pose_transformation coordinates to find the ASU entities. These will then be used to make oligomers
        # assume a globular nature to entity chains
        # therefore the minimal com to com dist is our asu and therefore naive asu coords
        if len(asu_indices) == 1:
            selected_asu_indices = [asu_indices[0][0]]  # choice doesn't matter, grab the first
        else:
            all_coms = []
            for group_idx, indices in enumerate(asu_indices):
                all_coms.append(center_of_mass_symmetric_entities[group_idx][indices])
                # pdist()
            # self.log.critical('all_coms: %s' % all_coms)
            idx = 0
            asu_indices_combinations = []
            asu_indices_index, asu_coms_index = [], []
            com_offsets = np.zeros(sum(map(prod, combinations((len(indices) for indices in asu_indices), 2))))
            for idx1, idx2 in combinations(range(len(asu_indices)), 2):
                # for index1 in asu_indices[idx1]:
                for idx_com1, com1 in enumerate(all_coms[idx1]):
                    for idx_com2, com2 in enumerate(all_coms[idx2]):
                        asu_indices_combinations.append((idx1, idx_com1, idx2, idx_com2))
                        asu_indices_index.append((idx1, idx2))
                        asu_coms_index.append((idx_com1, idx_com2))
                        dist = com2 - com1
                        com_offsets[idx] = np.sqrt(dist.dot(dist))
                        idx += 1
            # self.log.critical('com_offsets: %s' % com_offsets)
            minimal_com_distance_index = com_offsets.argmin()
            entity_index1, com_index1, entity_index2, com_index2 = asu_indices_combinations[minimal_com_distance_index]
            # entity_index1, entity_index2 = asu_indices_index[minimal_com_distance_index]
            # com_index1, com_index2 = asu_coms_index[minimal_com_distance_index]
            core_indices = [(entity_index1, com_index1), (entity_index2, com_index2)]
            # asu_index2 = asu_indices[entity_index2][com_index2]
            additional_indices = []
            if len(asu_indices) != 2:  # we have to find more indices
                # find the indices where either of the minimally distanced coms are utilized
                # selected_asu_indices_indices = {idx for idx, (ent_idx1, com_idx1, ent_idx2, com_idx2) in enumerate(asu_indices_combinations)
                #                                 if entity_index1 == ent_idx1 or entity_index2 == ent_idx2 and not }
                selected_asu_indices_indices = {idx for idx, ent_idx_pair in enumerate(asu_indices_index)
                                                if entity_index1 in ent_idx_pair or entity_index2 in ent_idx_pair and
                                                asu_indices_index[idx] != ent_idx_pair}
                remaining_indices = set(range(len(asu_indices))).difference({entity_index1, entity_index2})
                for index in remaining_indices:
                    # find the indices where the missing index is utilized
                    remaining_index_indices = \
                        {idx for idx, ent_idx_pair in enumerate(asu_indices_index) if index in ent_idx_pair}
                    # only use those where found asu indices already occur
                    viable_remaining_indices = list(remaining_index_indices.intersection(selected_asu_indices_indices))
                    index_min_com_dist_idx = com_offsets[viable_remaining_indices].argmin()
                    for new_index_idx, new_index in enumerate(asu_indices_index[viable_remaining_indices[index_min_com_dist_idx]]):
                        if index == new_index:
                            additional_indices.append((index, asu_coms_index[viable_remaining_indices[index_min_com_dist_idx]][new_index_idx]))
            new_asu_indices = core_indices + additional_indices

            # selected_asu_indices = [asu_indices[entity_idx][com_idx] for entity_idx, com_idx in core_indices + additional_indices]
            selected_asu_indices = []
            for range_idx in range(len(asu_indices)):
                for entity_idx, com_idx in new_asu_indices:
                    if range_idx == entity_idx:
                        selected_asu_indices.append(asu_indices[entity_idx][com_idx])

        symmetric_coords_split_by_entity = self.symmetric_coords_split_by_entity
        asu_coords = [symmetric_coords_split_by_entity[group_idx][sym_idx]
                      for group_idx, sym_idx in enumerate(selected_asu_indices)]
        # self.log.critical('asu_coords: %s' % asu_coords)
        # self.set_asu_coords(np.concatenate(asu_coords))
        self.asu_coords = Coords(np.concatenate(asu_coords))  # Todo ensure correct
        # for idx, entity in enumerate(self.entities):
        #     entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **transform_solutions[idx])

        return transform_solutions

    def find_contacting_asu(self, distance: int = 8, **kwargs) -> list[Entity]:
        """From the Pose Entities, find the maximal contacting Chain for each Entity

        Args:
            distance: The distance to check for contacts
        Returns:
            The minimal set of Entities containing the maximally touching configuration
        """
        entities = self.entities
        if self.number_of_entities != 1:
            idx = 0
            chain_combinations: list[tuple[Entity, Entity]] = []
            entity_combinations: list[tuple[Entity, Entity]] = []
            contact_count = \
                np.zeros(sum(map(prod, combinations((entity.number_of_monomers for entity in entities), 2))))
            for entity1, entity2 in combinations(entities, 2):
                for chain1 in entity1.chains:
                    chain_cb_coord_tree = BallTree(chain1.get_cb_coords())
                    for chain2 in entity2.chains:
                        entity_combinations.append((entity1, entity2))
                        chain_combinations.append((chain1, chain2))
                        contact_count[idx] = \
                            chain_cb_coord_tree.two_point_correlation(chain2.get_cb_coords(), [distance])[0]
                        idx += 1

            max_contact_idx = contact_count.argmax()
            additional_chains = []
            max_chains = list(chain_combinations[max_contact_idx])
            if len(max_chains) != self.number_of_entities:
                # find the indices where either of the maximally contacting chains are utilized
                selected_chain_indices = {idx for idx, chain_pair in enumerate(chain_combinations)
                                          if max_chains[0] in chain_pair or max_chains[1] in chain_pair}
                remaining_entities = set(entities).difference(entity_combinations[max_contact_idx])
                for entity in remaining_entities:  # get the max contacts and the associated entity and chain indices
                    # find the indices where the missing entity is utilized
                    remaining_indices = \
                        {idx for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair}
                    # pair_position = [0 if entity_pair[0] == entity else 1
                    #                  for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair]
                    # only use those where found asu chains already occur
                    viable_remaining_indices = list(remaining_indices.intersection(selected_chain_indices))
                    # out of the viable indices where the selected chains are matched with the missing entity,
                    # find the highest contact
                    max_idx = contact_count[viable_remaining_indices].argmax()
                    for entity_idx, entity_in_combo in enumerate(entity_combinations[viable_remaining_indices[max_idx]]):
                        if entity == entity_in_combo:
                            additional_chains.append(chain_combinations[viable_remaining_indices[max_idx]][entity_idx])

            new_entities = max_chains + additional_chains
            # print([(entity.name, entity.chain_id) for entity in entities])
            # rearrange the entities to have the same order as provided
            # for new_entity in new_entities:
            #     print('NEW', new_entity.name, new_entity.chain)
            # for entity in entities:
            #     print('OLD', entity.name, entity.chain)
            entities = [new_entity for entity in entities for new_entity in new_entities if entity == new_entity]
            # print([(entity.name, entity.chain_id) for entity in entities])
        return entities

    def return_contacting_asu(self, **kwargs) -> PDB:
        """From the Pose Entities, find the maximal contacting Chain for each of the entities and return the ASU

        If the chain IDs of the asu are the same, then chain IDs will automatically be renamed

        Returns:
            A PDB object with the minimal set of Entities containing the maximally touching configuration
        """
        entities = self.find_contacting_asu(**kwargs)
        found_chain_ids = []
        for entity in entities:
            if entity.chain_id in found_chain_ids:
                kwargs['rename_chains'] = True
                break
            else:
                found_chain_ids.append(entity.chain_id)

        return PDB.from_entities(entities, name='asu', log=self.log, biomt_header=self.format_biomt(),
                                 cryst_record=self.cryst_record, **kwargs)

    def set_contacting_asu(self, **kwargs):
        """From the Pose Entities, find the maximal contacting Chain for each of the entities and set the Pose.asu

        Sets:
            self._pdb: To a PDB object with the minimal set of Entities containing the maximally touching configuration
        """
        entities = self.find_contacting_asu(**kwargs)
        # self = PDB.from_entities(entities, name='asu', log=self.log, **kwargs)  # Todo remove .pdb
        self._pdb = PDB.from_entities(entities, name='asu', log=self.log, **kwargs)

    # def make_oligomers(self):
    #     """Generate oligomers for each Entity in the SymmetricModel"""
    #     for idx, entity in enumerate(self.entities):
    #         entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **self.transformations[idx])

    def symmetric_assembly_is_clash(self, distance: float = 2.1) -> bool:  # Todo design_selector
        """Returns True if the SymmetricModel presents any clashes. Checks only backbone and CB atoms

        Args:
            distance: The cutoff distance for the coordinate overlap
        Returns:
            True if the symmetric assembly clashes with the asu, False otherwise
        """
        if not self.symmetry:
            raise SymmetryError('Cannot check if the assembly is clashing as it has no symmetry!')
        # elif self.number_of_symmetry_mates == 1:
        #     raise ValueError(f'Cannot check if the assembly is clashing without first calling '
        #                      f'{self.generate_symmetric_coords.__name__}')

        # if self.coords_type != 'bb_cb':
        # Need to select only coords that are BB or CB from the model coords
        # asu_indices = self.backbone_and_cb_indices
        # else:
        #     asu_indices = None

        self.generate_assembly_tree()
        # clashes = asu_coord_tree.two_point_correlation(self.symmetric_coords[model_indices_without_asu], [distance])
        clashes = self.assembly_tree.two_point_correlation(self.coords[self.backbone_and_cb_indices], [distance])
        if clashes[0] > 0:
            self.log.warning(f'{self.name}: Found {clashes[0]} clashing sites! Pose is not a viable symmetric assembly')
            return True  # clash
        else:
            return False  # no clash

    def generate_assembly_tree(self):
        """Create a tree structure from all the coordinates in the symmetric assembly

        Sets:
            self.assembly_tree (sklearn.neighbors._ball_tree.BinaryTree): The constructed coordinate tree
        """
        if self.assembly_tree:
            return

        # model_asu_indices = self.get_asu_atom_indices()
        # if self.coords_type == 'bb_cb':  # grab every coord in the model
        #     model_indices = np.arange(len(self.symmetric_coords))
        #     asu_indices = []
        # else:  # Select only coords that are BB or CB from the model coords
        number_asu_atoms = self.number_of_atoms
        asu_indices = self.backbone_and_cb_indices
        # We have all the BB/CB indices from ASU, must multiply this int's in self.number_of_symmetry_mates
        # to get every BB/CB coord in the model
        # Finally we take out those indices that are inclusive of the model_asu_indices like below
        model_indices = np.array([idx + (model_number * number_asu_atoms)
                                  for model_number in range(self.number_of_symmetry_mates) for idx in asu_indices])

        # # make a boolean mask where the model indices of interest are True
        # without_asu_mask = np.logical_or(model_indices < model_asu_indices[0],
        #                                  model_indices > model_asu_indices[-1])
        # model_indices_without_asu = model_indices[without_asu_mask]
        # take the boolean mask and filter the model indices mask to leave only symmetry mate bb/cb indices, NOT asu
        model_indices_without_asu = model_indices[len(asu_indices):]
        # selected_assembly_coords = len(model_indices_without_asu) + len(asu_indices)
        # all_assembly_coords_length = len(asu_indices) * self.number_of_symmetry_mates
        # assert selected_assembly_coords == all_assembly_coords_length, \
        #     '%s: Ran into an issue indexing' % self.symmetric_assembly_is_clash.__name__

        # asu_coord_tree = BallTree(self.coords[asu_indices])
        # return BallTree(self.symmetric_coords[model_indices_without_asu])
        self.assembly_tree = BallTree(self.symmetric_coords[model_indices_without_asu])

    def format_biomt(self, **kwargs) -> str:
        """Return the SymmetricModel expand_matrices as a BIOMT record

        Returns:
            The BIOMT REMARK 350 with PDB file formatting
        """
        if self.dimension == 0:
            return '%s\n' % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}'
                                      .format(v_idx, m_idx, *vec, 0.)
                                      for m_idx, mat in enumerate(self.expand_matrices.swapaxes(-2, -1).tolist(), 1)
                                      for v_idx, vec in enumerate(mat, 1))
        else:  # TODO write so that the oligomeric units are populated?
            return ''

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, assembly: bool = False, **kwargs) -> \
            str | bytes | None:
        """Write Model Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            assembly: Whether to write the full assembly
        Keyword Args:
            increment_chains=False (bool): Whether to write each Structure with a new chain name, otherwise write as a
                new Model
            surrounding_uc=True (bool): Write the surrounding unit cell if assembly is True and self.dimension > 1
        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'SymmetricModel is writing')

        def symmetric_model_write(handle):
            self.write_header(handle, **kwargs)
            if assembly:  # will make models and use next logic steps to write them out
                self.generate_assembly_symmetry_models(**kwargs)
                # self.models is populated, use Models.write() to finish
                super().write(file_handle=handle, **kwargs)
            else:  # skip models, write asu using biomt_record/cryst_record for sym
                for entity in self.entities:
                    entity.write(file_handle=handle, **kwargs)

        def model_write(handle):
            self.write_header(handle, **kwargs)
            super(Model, Models).write(file_handle=handle, **kwargs)

        if file_handle:  # Todo should the use of self.write_header() with file_handle be disabled?
            if self.symmetry:
                symmetric_model_write(file_handle)
            else:
                model_write(file_handle)
            return
        else:
            with open(out_path, 'w') as outfile:
                if self.symmetry:
                    symmetric_model_write(outfile)
                else:
                    model_write(outfile)
            return out_path


class Pose(SymmetricModel, SequenceProfile):  # Todo consider moving SequenceProfile to first in MRO
    """A Pose is made of single or multiple Structure objects such as Entities, Chains, or other structures.
    All objects share a common feature such as the same symmetric system or the same general atom configuration in
    separate models across the Structure or sequence.
    """
    design_selector: dict[str, dict[str, dict[str, set[int] | set[str] | None]]] | None
    design_selector_entities: set[Entity]
    design_selector_indices: set[int]
    euler_lookup: EulerLookup | None
    fragment_metrics: dict
    fragment_pairs: list[tuple[GhostFragment, Fragment, float]] | list
    fragment_queries: dict[tuple[Entity, Entity], list[dict[str, Any]]]
    ignore_clashes: bool
    interface_residues: dict[tuple[Entity, Entity], tuple[list[Residue], list[Residue]]]
    required_indices: set[int]
    required_residues: list[Residue] | None
    resource_db: Database | None
    split_interface_residues: dict[int, list[tuple[Residue, Entity]]]
    split_interface_ss_elements: dict[int, list[int]]
    ss_index_array: list[int]
    ss_type_array: list[str]

    def __init__(self, fragment_db: FragmentDatabase = None, resource_db: Database = None,
                 design_selector: dict[str, dict[str, dict[str, set[int] | set[str] | None]]] = None, **kwargs):
        #          euler_lookup: EulerLookup = None,
        self.design_selector = design_selector  # kwargs.get('design_selector', {})
        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.euler_lookup = euler_factory()  # kwargs.get('euler_lookup', None)
        self.fragment_metrics = {}
        self.fragment_pairs = []
        self.fragment_queries = {}
        self.ignore_clashes = kwargs.get(PUtils.ignore_clashes, False)
        self.interface_residues = {}
        self.required_indices = set()
        self.required_residues = None
        self.resource_db = resource_db  # kwargs.get('resource_db', None)
        self.split_interface_residues = {}  # {1: [(Residue obj, Entity obj), ...], 2: [(Residue obj, Entity obj), ...]}
        self.split_interface_ss_elements = {}  # {1: [0, 1, 2] , 2: [9, 13, 19]]}
        self.ss_index_array = []  # stores secondary structure elements by incrementing index
        self.ss_type_array = []  # stores secondary structure type ('H', 'S', ...)

        # Model init will handle Structure set up if a structure file is present
        # SymmetricModel init will generate_symmetric_coords() if symmetry specification present
        super().__init__(**kwargs)
        if self.is_clash():
            if not self.ignore_clashes:
                raise DesignError(f'{self.name} contains Backbone clashes and is not being considered further!')

        # need to set up after load Entities so that they can have this added to their SequenceProfile
        self.fragment_db = fragment_db  # kwargs.get('fragment_db', None)
        self.create_design_selector()  # **self.design_selector)
        self.log.debug(f'Entities: {", ".join(entity.name for entity in self.entities)}')
        self.log.debug(f'Active Entities: {", ".join(entity.name for entity in self.active_entities)}')

    @property
    def fragment_db(self) -> FragmentDatabase:
        """The FragmentDatabase with which information about fragment usage will be extracted"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: FragmentDatabase):
        if not isinstance(fragment_db, FragmentDatabase):
            self.log.warning(f'The passed fragment_db is being set to the default since {fragment_db} was passed which '
                             f'is not of the required type {FragmentDatabase.__name__}')
            fragment_db = fragment_factory(source=PUtils.biological_interfaces)  # Todo add fragment_length, sql kwargs

        self._fragment_db = fragment_db
        for entity in self.entities:
            entity.fragment_db = fragment_db

    @SymmetricModel.asu.setter
    def asu(self, asu):
        self.pdb = asu  # process incoming structure as normal
        if self.number_of_entities != self.number_of_chains:  # ensure the structure is an asu
            # self.log.debug('self.number_of_entities (%d) self.number_of_chains (%d)'
            #                % (self.number_of_entities, self.number_of_chains))
            self.log.debug('Setting Pose ASU to the ASU with the most contacting interface')
            self.set_contacting_asu()  # find maximally touching ASU and set ._pdb

    @property
    def active_entities(self):
        try:
            return self._active_entities
        except AttributeError:
            self._active_entities = [entity for entity in self.entities if entity in self.design_selector_entities]
            return self._active_entities

    def create_design_selector(self):
        """Set up a design selector for the Pose including selections, masks, and required Entities and Atoms

        Sets:
            self.design_selector_entities (set[Entity])
            self.design_selector_indices (set[int])
            self.required_indices (set[int])
        """
        def grab_indices(entities: set[str] = None, chains: set[str] = None, residues: set[int] = None,
                         pdb_residues: set[int] = None, start_with_none: bool = False) -> tuple[set[Entity], set[int]]:
            #              atoms: set[int] = None
            """Parse the residue selector to a set of entities and a set of atom indices"""
            if start_with_none:
                entity_set = set()
                atom_indices = set()
                set_function = getattr(set, 'union')
            else:  # start with all indices and include those of interest
                entity_set = set(self.entities)
                atom_indices = set(self._atom_indices)
                set_function = getattr(set, 'intersection')

            if entities:
                atom_indices = set_function(atom_indices, iter_chain.from_iterable([self.entity(entity).atom_indices
                                                                                   for entity in entities]))
                entity_set = set_function(entity_set, [self.entity(entity) for entity in entities])
            if chains:
                # vv This is for the intersectional model
                atom_indices = set_function(atom_indices, iter_chain.from_iterable([self.chain(chain_id).atom_indices
                                                                                   for chain_id in chains]))
                # atom_indices.union(iter_chain.from_iterable(self.chain(chain_id).get_residue_atom_indices(numbers=residues)
                #                                     for chain_id in chains))
                # ^^ This is for the additive model
                entity_set = set_function(entity_set, [self.chain(chain_id) for chain_id in chains])
            if residues:
                atom_indices = set_function(atom_indices, self.get_residue_atom_indices(numbers=residues))
            if pdb_residues:
                atom_indices = set_function(atom_indices, self.get_residue_atom_indices(numbers=residues, pdb=True))
            # if atoms:
            #     atom_indices = set_function(atom_indices, [idx for idx in self._atom_indices if idx in atoms])

            return entity_set, atom_indices

        selection = self.design_selector.get('selection')
        if selection:
            self.log.debug(f'The design_selection includes: {selection}')
            entity_selection, atom_selection = grab_indices(**selection)
        else:  # use all the entities and indices
            entity_selection, atom_selection = set(self.entities), set(self._atom_indices)

        mask = self.design_selector.get('mask')
        if mask:
            self.log.debug(f'The design_mask includes: {mask}')
            entity_mask, atom_mask = grab_indices(**mask, start_with_none=True)
        else:
            entity_mask = atom_mask = set()

        self.design_selector_entities = entity_selection.difference(entity_mask)
        self.design_selector_indices = atom_selection.difference(atom_mask)

        required = self.design_selector.get('required')
        if required:
            self.log.debug(f'The required_residues includes: {required}')
            entity_required, self.required_indices = grab_indices(**required, start_with_none=True)
            # Todo create a separte variable for required_entities?
            self.design_selector_entities = self.design_selector_entities.union(entity_required)
            if self.required_indices:  # only if indices are specified should we grab them
                self.required_residues = self.get_residues_by_atom_indices(atom_indices=self.required_indices)
        else:
            entity_required = self.required_indices = set()

    def return_interface(self, distance: float = 8.) -> Structure:
        """Provide a view of the Pose interface by generating a Structure containing only interface Residues

        Args:
            distance: The distance across the interface to query for Residue contacts
        Returns:
            The Structure containing only the Residues in the interface
        """
        raise NotImplementedError('This function has not been properly converted to deal with non symmetric poses')
        number_of_models = self.number_of_symmetry_mates
        # find all pertinent interface residues from results of find_interface_residues()
        residues_entities = []
        for residue_entities in self.split_interface_residues.values():
            residues_entities.extend(residue_entities)
        interface_residues, interface_entities = list(zip(*residues_entities))

        # interface_residues = []
        # interface_core_coords = []
        # for residues1, residues2 in self.interface_residues.values():
        #     if not residues1 and not residues2:  # no interface
        #         continue
        #     elif residues1 and not residues2:  # symmetric case
        #         interface_residues.extend(residues1)
        #         # This was useful when not doing the symmetrization below...
        #         # symmetric_residues = []
        #         # for _ in range(number_of_models):
        #         #     symmetric_residues.extend(residues1)
        #         # residues1_coords = np.concatenate([residue.coords for residue in residues1])
        #         # # Add the number of symmetric observed structures to a single new Structure
        #         # symmetric_residue_structure = Structure.from_residues(residues=symmetric_residues)
        #         # # symmetric_residues2_coords = self.return_symmetric_coords(residues1_coords)
        #         # symmetric_residue_structure.replace_coords(self.return_symmetric_coords(residues1_coords))
        #         # # use a single instance of the residue coords to perform a distance query against symmetric coords
        #         # residues_tree = BallTree(residues1_coords)
        #         # symmetric_query = residues_tree.query_radius(symmetric_residue_structure.coords, distance)
        #         # # symmetric_indices = [symmetry_idx for symmetry_idx, asu_contacts in enumerate(symmetric_query)
        #         # #                      if asu_contacts.any()]
        #         # # finally, add all correctly located, asu interface indexed symmetrical residues to the interface
        #         # coords_indexed_residues = symmetric_residue_structure.coords_indexed_residues
        #         # interface_residues.extend(set(coords_indexed_residues[sym_idx]
        #         #                               for sym_idx, asu_contacts in enumerate(symmetric_query)
        #         #                               if asu_contacts.any()))
        #     else:  # non-symmetric case
        #         interface_core_coords.extend([residue.cb_coords for residue in residues1])
        #         interface_core_coords.extend([residue.cb_coords for residue in residues2])
        #         interface_residues.extend(residues1), interface_residues.extend(residues2)

        # return Structure.from_residues(residues=sorted(interface_residues, key=lambda residue: residue.number))
        interface_asu_structure = \
            Structure.from_residues(residues=sorted(set(interface_residues), key=lambda residue: residue.number))
        # interface_symmetry_mates = self.return_symmetric_copies(interface_asu_structure)
        # interface_coords = interface_asu_structure.coords
        coords_length = interface_asu_structure.number_of_atoms
        # interface_cb_indices = interface_asu_structure.cb_indices
        # print('NUMBER of RESIDUES:', interface_asu_structure.number_of_residues,
        #       '\nNUMBER of CB INDICES', len(interface_cb_indices))
        # residue_number = interface_asu_structure.number_of_residues
        # [interface_asu_structure.cb_indices + (residue_number * model) for model in self.number_of_symmetry_mates]
        symmetric_cb_indices = np.array([idx + (coords_length * model_num) for model_num in range(number_of_models)
                                         for idx in interface_asu_structure.cb_indices])
        # print('Number sym CB INDICES:\n', len(symmetric_cb_indices))
        symmetric_interface_coords = self.return_symmetric_coords(interface_asu_structure.coords)
        # from the interface core, find the mean position to seed clustering
        entities_asu_com = self.center_of_mass
        initial_interface_coords = self.return_symmetric_coords(entities_asu_com)
        # initial_interface_coords = self.return_symmetric_coords(np.array(interface_core_coords).mean(axis=0))

        # index_cluster_labels = KMeans(n_clusters=self.number_of_symmetry_mates).fit_predict(symmetric_interface_coords)
        # symmetric_interface_cb_coords = symmetric_interface_coords[symmetric_cb_indices]
        # print('Number sym CB COORDS:\n', len(symmetric_interface_cb_coords))
        # initial_cluster_indices = [interface_cb_indices[0] + (coords_length * model_number)
        #                            for model_number in range(self.number_of_symmetry_mates)]
        # fit a KMeans model to the symmetric interface cb coords
        kmeans_cluster_model: KMeans = KMeans(n_clusters=number_of_models, init=initial_interface_coords, n_init=1)\
            .fit(symmetric_interface_coords[symmetric_cb_indices])
        # kmeans_cluster_model = \
        #     KMeans(n_clusters=self.number_of_symmetry_mates, init=symmetric_interface_coords[initial_cluster_indices],
        #            n_init=1).fit(symmetric_interface_cb_coords)
        index_cluster_labels = kmeans_cluster_model.labels_
        # find the label where the asu is nearest too
        asu_label = kmeans_cluster_model.predict(entities_asu_com[None, :])  # add new first axis
        # asu_interface_labels = kmeans_cluster_model.predict(interface_asu_structure.get_cb_coords())

        # closest_interface_indices = np.where(index_cluster_labels == 0, True, False)
        # [False, False, False, True, True, True, True, True, True, False, False, False, False, False, ...]
        # symmetric_residues = interface_asu_structure.residues * self.number_of_symmetry_mates
        # [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, ...]
        # asu_index = np.median(asu_interface_labels)
        # grab the symmetric indices for a single interface cluster, matching spatial proximity to the asu_index
        # closest_asu_sym_cb_indices = symmetric_cb_indices[index_cluster_labels == asu_index]
        closest_asu_sym_cb_indices = np.where(index_cluster_labels == asu_label, symmetric_cb_indices, 0)
        # # find the cb indices of the closest interface asu
        # closest_asu_cb_indices = closest_asu_sym_cb_indices % coords_length
        # interface_asu_structure.coords_indexed_residues
        # find the model indices of the closest interface asu
        # print('Normal sym CB INDICES\n:', closest_asu_sym_cb_indices)
        flat_sym_model_indices = closest_asu_sym_cb_indices.reshape((number_of_models, -1)).sum(axis=0)
        # print('FLATTENED CB INDICES to get MODEL\n:', flat_sym_model_indices)
        symmetric_model_indices = flat_sym_model_indices // coords_length
        # print('FLOORED sym CB INDICES to get MODEL\n:', symmetric_model_indices)
        symmetry_mate_index_symmetric_coords = symmetric_interface_coords.reshape((number_of_models, -1, 3))
        # print('RESHAPED SYMMETRIC COORDS SHAPE:', symmetry_mate_index_symmetric_coords.shape,
        #       '\nCOORDS length:', coords_length)
        closest_interface_coords = \
            np.concatenate([symmetry_mate_index_symmetric_coords[symmetric_model_indices[idx]][residue.atom_indices]
                            for idx, residue in enumerate(interface_asu_structure.residues)])
        # closest_symmetric_coords = \
        #     np.where(index_cluster_labels[:, None] == asu_index, symmetric_interface_coords, np.array([0.0, 0.0, 0.0]))
        # closest_interface_coords = \
        #     closest_symmetric_coords.reshape((self.number_of_symmetry_mates, interface_coords.shape[0], -1)).sum(axis=0)
        # old-style
        # interface_asu_structure.replace_coords(closest_interface_coords)
        # new-style
        interface_asu_structure.coords = closest_interface_coords

        return interface_asu_structure

    def find_interface_pairs(self, entity1: Entity = None, entity2: Entity = None, distance: float = 8.) -> \
            list[tuple[Residue, Residue]] | None:
        """Get pairs of Residues that have CB Atoms within a distance between two Entities

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms
        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware. Will remove interface residues if not active under the design selector

        Args:
            entity1: First entity to measure interface between
            entity2: Second entity to measure interface between
            distance: The distance to query the interface in Angstroms
        Returns:
            A list of interface residue numbers across the interface
        """
        self.log.debug(f'Entity {entity1.name} | Entity {entity2.name} interface query')
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_indices = entity1.cb_indices
        entity2_indices = entity2.cb_indices

        if self.design_selector_indices:  # subtract the masked atom indices from the entity indices
            before = len(entity1_indices) + len(entity2_indices)
            entity1_indices = list(set(entity1_indices).intersection(self.design_selector_indices))
            entity2_indices = list(set(entity2_indices).intersection(self.design_selector_indices))
            self.log.debug('Applied design selection to interface identification. Number of indices before '
                           f'selection = {before}. Number after = {len(entity1_indices) + len(entity2_indices)}')

        if not entity1_indices or not entity2_indices:
            return

        # coords_length = len(self.coords)
        number_of_atoms = self.number_of_atoms
        if self.symmetry:
            sym_string = 'symmetric '
            self.log.debug(f'Number of Atoms in Pose: {number_of_atoms}')
            # get all symmetric indices
            entity2_indices = [idx + (number_of_atoms * model_number)
                               for model_number in range(self.number_of_symmetry_mates) for idx in entity2_indices]
            # solve for entity2_indices to query
            if entity1 == entity2:  # We don't want symmetry interactions with the asu model or intra-oligomeric models
                if entity1.is_oligomeric():  # remove oligomeric protomers (contains asu)
                    remove_indices = self.get_oligomeric_atom_indices(entity1)
                    self.log.info('Removing indices from models %s due to detected oligomer'
                                  % ', '.join(map(str, self.oligomeric_model_indices.get(entity1))))
                    self.log.debug(f'Removing {len(remove_indices)} indices from symmetric query due to oligomer')
                else:  # remove asu
                    remove_indices = self.get_asu_atom_indices()
                self.log.debug(f'Number of indices before removal of "self" indices: {len(entity2_indices)}')
                entity2_indices = list(set(entity2_indices).difference(remove_indices))
                self.log.debug(f'Final indices remaining after removing "self": {len(entity2_indices)}')
            entity2_coords = self.symmetric_coords[entity2_indices]  # only get the coordinate indices we want
        elif entity1 == entity2:
            # without symmetry, we can't measure this, unless intra-oligomeric contacts are desired
            self.log.warning('Entities are the same, but no symmetry is present. The interface between them will not be'
                             ' detected!')
            raise NotImplementedError('These entities shouldn\'t necessarily be equal. This issue needs to be addressed'
                                      'by expanding the __eq__ method of Entity to more accurately reflect what a '
                                      'Structure object represents programmatically')
            return
        else:
            sym_string = ''
            entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want
        entity1_tree = BallTree(entity1_coords)
        if len(entity2_coords) == 0:  # ensure the array is not empty
            return
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return residue numbers of identified coordinates
        self.log.info(f'Querying {len(entity1_indices)} CB residues in Entity {entity1.name} versus, '
                      f'{len(entity2_indices)} CB residues in {sym_string}Entity {entity2.name}')
        # contacting_pairs = [(pdb_residues[entity1_indices[entity1_idx]],
        #                      pdb_residues[entity2_indices[entity2_idx]])
        #                    for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

        # contacting_pairs = [(pdb_atoms[entity1_indices[entity1_idx]].residue_number,
        #                      pdb_atoms[entity2_indices[entity2_idx]].residue_number)
        #                     for entity2_idx, entity1_contacts in enumerate(entity2_query)
        #                     for entity1_idx in entity1_contacts]
        coords_indexed_residues = self.coords_indexed_residues
        contacting_pairs = [(coords_indexed_residues[entity1_indices[entity1_idx]],
                             coords_indexed_residues[entity2_indices[entity2_idx] % number_of_atoms])
                            for entity2_idx, entity1_contacts in enumerate(entity2_query)
                            for entity1_idx in entity1_contacts]
        if entity1 == entity2:  # solve symmetric results for asymmetric contacts
            asymmetric_contacting_pairs, found_pairs = [], []
            for pair1, pair2 in contacting_pairs:
                # only add to contacting pair if we have never observed either
                if (pair1, pair2) not in found_pairs or (pair2, pair1) not in found_pairs:
                    asymmetric_contacting_pairs.append((pair1, pair2))
                # add both pair orientations (1, 2) or (2, 1) regardless
                found_pairs.extend([(pair1, pair2), (pair2, pair1)])

            return asymmetric_contacting_pairs
        else:
            return contacting_pairs

    def find_interface_residues(self, entity1: Entity = None, entity2: Entity = None, **kwargs):
        """Get unique Residues across an interface provided by two Entities

        If the interface occurs between the same Entity which is non-symmetrically defined, but happens to occur along a
        dimeric axis of symmetry (evaluates to True when the same Residue is found on each side of the interface), then
        the residues are returned belonging to only one side of the interface

        Args:
            entity1: First Entity to measure interface between
            entity2: Second Entity to measure interface between
        Keyword Args:
            distance=8. (float): The distance to measure Residues across an interface
        Sets:
            self.interface_residues (dict[tuple[Entity, Entity], tuple[list[Residue], list[Residue]]]):
                The Entity1/Entity2 interface mapped to the interface Residues
        """
        entity1_residues, entity2_residues = \
            split_residue_pairs(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))

        if not entity1_residues or not entity2_residues:
            self.log.info(f'Interface search at {entity1.name} | {entity2.name} found no interface residues')
            self.interface_residues[(entity1, entity2)] = ([], [])
            return
        # else:
        if entity1 == entity2:  # symmetric query
            for residue in entity2_residues:  # entity2 usually has fewer residues, this might be quickest
                if residue in entity1_residues:  # the interface is dimeric and should only have residues on one side
                    entity1_residues, entity2_residues = \
                        sorted(set(entity1_residues).union(entity2_residues), key=lambda res: res.number), []
                    break
        self.log.info(f'At Entity {entity1.name} | Entity {entity2.name} interface:'
                      f'\n\t{entity1.name} found residue numbers: {", ".join(str(r.number) for r in entity1_residues)}'
                      f'\n\t{entity2.name} found residue numbers: {", ".join(str(r.number) for r in entity2_residues)}')

        self.interface_residues[(entity1, entity2)] = (entity1_residues, entity2_residues)
        # entities = [entity1, entity2]
        # self.log.debug(f'Added interface_residues: {", ".join(f"{residue.number}{entities[idx].chain_id}")}'
        #                for idx, entity_residues in enumerate(self.interface_residues[(entity1, entity2)])
        #                for residue in entity_residues)

    def find_interface_atoms(self, entity1: Entity = None, entity2: Entity = None, distance: float = 4.68) -> \
            list[tuple[int, int]] | None:
        """Get pairs of heavy atom indices that are within a distance at the interface between two Entities

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms

        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized

        Args:
            entity1: First Entity to measure interface between
            entity2: Second Entity to measure interface between
            distance: The distance to measure contacts between atoms. Default = CB radius + 2.8 H2O probe Was 3.28
        Returns:
            The Atom indices for the interface
        """
        residues1, residues2 = self.interface_residues.get((entity1, entity2))
        if not residues1:
            return
        if not residues2:  # check if the interface is a self and all residues are in residues1
            residues2 = residues1

        entity1_indices: list[int] = []
        for residue in residues1:
            entity1_indices.extend(residue.heavy_atom_indices)

        entity2_indices: list[int] = []
        for residue in residues2:
            entity2_indices.extend(residue.heavy_atom_indices)

        if self.symmetry:  # get all symmetric indices
            # number_of_atoms = len(self.coords)
            number_of_atoms = self.number_of_atoms
            entity2_indices = [idx + (number_of_atoms * model_number)
                               for model_number in range(self.number_of_symmetry_mates) for idx in entity2_indices]

        interface_atom_tree = BallTree(self.coords[entity1_indices])
        # self.coords uses the symmetric coords if a symmetric model
        atom_query = interface_atom_tree.query_radius(self.coords[entity2_indices], distance)
        contacting_pairs = [(entity1_indices[entity1_idx], entity2_indices[entity2_idx])
                            for entity2_idx, entity1_contacts in enumerate(atom_query)
                            for entity1_idx in entity1_contacts]
        return contacting_pairs

    def interface_local_density(self, distance: float = 12.) -> float:
        """Returns the average density of heavy atoms which neighbor the Atoms in the Pose interface

        Args:
            distance: The cutoff distance with which Atoms should be included in local density
        Returns:
            The local atom density around the interface
        """
        interface_indices1, interface_indices2 = [], []
        for entity1, entity2 in self.interface_residues:
            atoms_indices1, atoms_indices2 = \
                split_number_pairs_and_sort(self.find_interface_atoms(entity1=entity1, entity2=entity2))
            interface_indices1.extend(atoms_indices1), interface_indices2.extend(atoms_indices2)

        # operation assumes ASU is model_coords group 1
        # self.coords uses the symmetric coords if a symmetric model
        interface_coords = self.coords[list(set(interface_indices1).union(interface_indices2))]
        interface_tree = BallTree(interface_coords)
        interface_counts = interface_tree.query_radius(interface_coords, distance, count_only=True)

        return interface_counts.mean()

    def query_interface_for_fragments(self, entity1: Entity = None, entity2: Entity = None):
        """For all found interface residues in an Entity/Entity interface, search for corresponding fragment pairs

        Args:
            entity1: The first Entity to measure for interface fragments
            entity2: The second Entity to measure for interface fragments
        Sets:
            self.fragment_queries (dict[tuple[Entity, Entity], list[dict[str, Any]]])
        """
        entity1_residues, entity2_residues = self.interface_residues.get((entity1, entity2))
        if not entity1_residues or not entity2_residues:
            self.log.info(f'No residues at the {entity1.name} | {entity2.name} interface. Fragments not available')
            self.fragment_queries[(entity1, entity2)] = []
            return
        # Todo NOT Correct! .is_oligomeric checks to see if interface is 2-fold but could be 2-fold with Entity monomer
        if entity1 == entity2 and entity1.is_oligomeric:
            entity1_residues = set(entity1_residues + entity2_residues)
            entity2_residues = entity1_residues

        residue_numbers1 = sorted(residue.number for residue in entity1_residues)
        residue_numbers2 = sorted(residue.number for residue in entity2_residues)
        self.log.debug(f'At Entity {entity1.name} | Entity {entity2.name} interface, searching for fragments at the '
                       f'surface of:\n\tEntity {entity1.name}: Residues {", ".join(map(str, residue_numbers1))}'
                       f'\n\tEntity {entity2.name}: Residues {", ".join(map(str, residue_numbers2))}')

        # surface_frags1 = entity1.get_fragments(residue_numbers=residue_numbers1, representatives=self.fragment_db.reps)
        # surface_frags2 = entity2.get_fragments(residue_numbers=residue_numbers2, representatives=self.fragment_db.reps)
        frag_residues1 = entity1.assign_fragments(residue_numbers=entity1_residues, representatives=self.fragment_db.reps)
        frag_residues2 = entity2.assign_fragments(residue_numbers=entity2_residues, representatives=self.fragment_db.reps)

        if not frag_residues1 or not frag_residues2:
            self.log.info(f'No fragments found at the {entity1.name} | {entity2.name} interface')
            self.fragment_queries[(entity1, entity2)] = []
            return
        else:
            self.log.debug(f'At Entity {entity1.name} | Entity {entity2.name} interface:\t'
                           f'{entity1.name} has {len(frag_residues1)} interface fragments\t'
                           f'{entity2.name} has {len(frag_residues2)} interface fragments')

        if self.symmetry:
            # even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            # asu frag subtraction is unnecessary THIS IS ALL WRONG DEPENDING ON THE CONTEXT
            if entity1 == entity2:
                # We don't want interactions with the intra-oligomeric contacts
                if entity1.is_oligomeric():  # remove oligomeric protomers (contains asu)
                    skip_models = self.oligomeric_model_indices.get(entity1)
                    self.log.info(f'Skipping oligomeric models: {", ".join(map(str, skip_models))}')
                else:  # probably a C1
                    skip_models = []
            else:
                skip_models = []
            symmetric_surface_frags2 = [self.return_symmetric_copies(residue) for residue in frag_residues2]
            entity2_residues.clear()
            for frag_mates in symmetric_surface_frags2:
                entity2_residues.extend(frag for sym_idx, frag in enumerate(frag_mates) if sym_idx not in skip_models)
            self.log.debug(f'Entity {entity2.name} has {len(frag_residues2)} symmetric fragments')

        entity1_coords = entity1.get_backbone_and_cb_coords()  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = find_fragment_overlap(entity1_coords, frag_residues1, frag_residues2,
                                                            fragdb=self.fragment_db, euler_lookup=self.euler_lookup)
        self.log.info(f'Found {len(ghostfrag_surfacefrag_pairs)} overlapping fragment pairs at the {entity1.name} | '
                      f'{entity2.name} interface')
        self.fragment_queries[(entity1, entity2)] = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
        # add newly found fragment pairs to the existing fragment observations
        self.fragment_pairs.extend(ghostfrag_surfacefrag_pairs)

    def score_interface(self, entity1=None, entity2=None):
        """Generate the fragment metrics for a specified interface between two entities

        Returns:
            (dict): Fragment metrics as key (metric type) value (measurement) pairs
        """
        if (entity1, entity2) not in self.fragment_queries or (entity2, entity1) not in self.fragment_queries:
            self.find_interface_residues(entity1=entity1, entity2=entity2)
            self.query_interface_for_fragments(entity1=entity1, entity2=entity2)

        return self.return_fragment_metrics(by_interface=True, entity1=entity1, entity2=entity2)

    def find_and_split_interface(self):
        """Locate the interface residues for the designable entities and split into two interfaces

        Sets:
            self.split_interface_residues (dict[int, list[tuple[Residue, Entity]]]): Residue/Entity id of each residue
                at the interface identified by interface id as split by topology
        """
        self.log.debug('Find and split interface using active_entities: %s' %
                       ', '.join(entity.name for entity in self.active_entities))
        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.find_interface_residues(*entity_pair)

        self.check_interface_topology()

    def check_interface_topology(self):
        """From each pair of entities that share an interface, split the identified residues into two distinct groups.
        If an interface can't be composed into two distinct groups, raise DesignError

        Sets:
            self.split_interface_residues (dict[int, list[tuple[Residue, Entity]]]): Residue/Entity id of each residue
                at the interface identified by interface id as split by topology
        """
        first_side, second_side = 0, 1
        interface = {first_side: {}, second_side: {}, 'self': [False, False]}  # assume no symmetric contacts to start
        terminate = False
        # self.log.debug('Pose contains interface residues: %s' % self.interface_residues)
        for entity_pair, entity_residues in self.interface_residues.items():
            entity1, entity2 = entity_pair
            residues1, residues2 = entity_residues
            # if not entity_residues:
            if not residues1:  # no residues were found at this interface
                continue
            else:  # Partition residues from each entity to the correct interface side
                # check for any existing symmetry
                if entity1 == entity2:  # if query is with self, have to record it
                    _self = True
                    if not residues2:  # the interface is symmetric dimer and residues were removed from interface 2
                        residues2 = copy(residues1)  # add residues1 to residues2
                else:
                    _self = False

                if not interface[first_side]:  # This is first interface observation
                    # add the pair to the dictionary in their indexed order
                    interface[first_side][entity1], interface[second_side][entity2] = copy(residues1), copy(residues2)
                    # indicate whether the interface is a self symmetric interface by marking side 2 with _self
                    interface['self'][second_side] = _self
                else:  # We have interface assigned, so interface observation >= 2
                    # Need to check if either Entity is in either side before adding correctly
                    if entity1 in interface[first_side]:  # is Entity1 on the interface side 1?
                        if interface['self'][first_side]:
                            # is an Entity in interface1 here as a result of self symmetric interaction?
                            # if so, flip Entity1 to interface side 2, add new Entity2 to interface side 1
                            # Ex4 - self Entity was added to index 0 while ASU added to index 1
                            interface[second_side][entity1].extend(residues1)
                            interface[first_side][entity2] = copy(residues2)
                        else:  # Entities are properly indexed, extend the first index
                            interface[first_side][entity1].extend(residues1)
                            # Because of combinations with replacement Entity search, the second Entity is not in
                            # interface side 2, UNLESS the Entity self interaction is on interface 1 (above if check)
                            # Therefore, add without checking for overwrite
                            interface[second_side][entity2] = copy(residues2)
                            # if _self:  # This can't happen, it would VIOLATE RULES
                            #     interface['self'][second] = _self
                    # Entity1 is not in the first index. It may be in the second, it may not
                    elif entity1 in interface[second_side]:  # it is, add to interface side 2
                        interface[second_side][entity1].extend(residues1)
                        # also add it's partner entity to the first index
                        # Entity 2 can't be in interface side 1 due to combinations with replacement check
                        interface[first_side][entity2] = copy(residues2)  # Ex5
                        if _self:  # only modify if self is True, don't want to overwrite an existing True value
                            interface['self'][first_side] = _self
                    # If Entity1 is missing, check Entity2 to see if it has been identified yet
                    elif entity2 in interface[second_side]:  # this is more likely from combinations with replacement
                        # Possible in an iteration Ex: (A:D) (C:D)
                        interface[second_side][entity2].extend(residues2)
                        # entity 1 was not in first interface (from if #1), therefore we can set directly
                        interface[first_side][entity1] = copy(residues1)
                        if _self:  # only modify if self is True, don't want to overwrite an existing True value
                            interface['self'][first_side] = _self  # Ex3
                    elif entity2 in interface[first_side]:
                        # the first Entity wasn't found in either interface, but both interfaces are already set,
                        # therefore Entity pair isn't self, so the only way this works is if entity1 is further in the
                        # iterative process which is an impossible topology, and violates interface separation rules
                        interface[second_side][entity1] = False
                        terminate = True
                        break
                    # Neither of our Entities were found, thus we would add 2 entities to each interface side, violation
                    else:
                        interface[first_side][entity1], interface[second_side][entity2] = False, False
                        terminate = True
                        break

            interface1, interface2, self_check = tuple(interface.values())
            if len(interface1) == 2 and len(interface2) == 2 and all(self_check):
                pass
            elif len(interface1) == 1 or len(interface2) == 1:
                pass
            else:
                terminate = True
                break

        self_indications = interface.pop('self')
        if terminate:
            self.log.critical('The set of interfaces found during interface search generated a topologically '
                              'disallowed combination.\n\t %s\n This cannot be modelled by a simple split for residues '
                              'on either side while respecting the requirements of polymeric Entities. '
                              '%sPlease correct your design_selectors to reduce the number of Entities you are '
                              'attempting to design'
                              % (' | '.join(':'.join(entity.name for entity in interface_entities)
                                            for interface_entities in interface.values()),
                                 'Symmetry was set which may have influenced this unfeasible topology, you can try to '
                                 'set it False. ' if self.symmetry else ''))
            raise DesignError('The specified interfaces generated a topologically disallowed combination! Check the log'
                              ' for more information.')

        for key, entity_residues in interface.items():
            all_residues = [(residue, entity) for entity, residues in entity_residues.items() for residue in residues]
            self.split_interface_residues[key + 1] = sorted(all_residues, key=lambda res_ent: res_ent[0].number)

        if not self.split_interface_residues[1]:
            # Todo return an error but don't raise anything
            raise DesignError('Interface was unable to be split because no residues were found on one side of the'
                              ' interface! Check that your input has an interface or your flags aren\'t too stringent')
        else:
            self.log.debug('The interface is split as:\n\tInterface 1: %s\n\tInterface 2: %s'
                           % tuple(','.join('%d%s' % (res.number, ent.chain_id) for res, ent in residues_entities)
                                   for residues_entities in self.split_interface_residues.values()))

    def interface_secondary_structure(self):
        """From a split interface, curate the secondary structure topology for each

        Sets:
            self.ss_index_array (list[int]): The indices where the secondary structure transitoins to another type
            self.ss_type_array (list[str]): The ordered secondary structure type for the Pose
            self.split_interface_ss_elements (dict[int, list[int]]): The secondary structure split across the interface
        """
        pose_secondary_structure = ''
        for entity in self.active_entities:
            if not entity.secondary_structure:
                if self.resource_db:
                    parsed_secondary_structure = self.resource_db.stride.retrieve_data(name=entity.name)
                    if parsed_secondary_structure:
                        entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                    else:
                        entity.stride(to_file=self.resource_db.stride.store(entity.name))
                # if source_dir:
                #     entity.parse_stride(os.path.join(source_dir, '%s.stride' % entity.name))
                else:
                    entity.stride()
            pose_secondary_structure += entity.secondary_structure

        # increment a secondary structure index which changes with every secondary structure transition
        # simultaneously, map the secondary structure type to an array of pose length (offset for residue number)
        self.ss_index_array.clear(), self.ss_type_array.clear()  # clear any information if it exists
        self.ss_type_array.append(pose_secondary_structure[0])
        ss_increment_index = 0
        self.ss_index_array.append(ss_increment_index)
        for prior_idx, ss_type in enumerate(pose_secondary_structure[1:], 0):
            if ss_type != pose_secondary_structure[prior_idx]:
                self.ss_type_array.append(ss_type)
                ss_increment_index += 1
            self.ss_index_array.append(ss_increment_index)

        for number, residues_entities in self.split_interface_residues.items():
            self.split_interface_ss_elements[number] = []
            for residue, entity in residues_entities:
                try:
                    self.split_interface_ss_elements[number].append(self.ss_index_array[residue.number - 1])
                except IndexError:
                    raise IndexError('The index %d, from entity %s, residue %d is not found in ss_index_array size %d'
                                     % (residue.number - 1, entity.name, residue.number, len(self.ss_index_array)))

        self.log.debug(f'Found interface secondary structure: {self.split_interface_ss_elements}')

    def interface_design(self, evolution=True, fragments=True, write_fragments=True, des_dir=None):
        # Todo deprec. des_dir
        """Compute calculations relevant to interface design.

        Sets:
            self.pssm_file (str | bytes)
        """
        # self.log.debug('Entities: %s' % ', '.join(entity.name for entity in self.entities))
        # self.log.debug('Active Entities: %s' % ', '.join(entity.name for entity in self.active_entities))

        # we get interface residues for the designable entities as well as interface_topology at PoseDirectory level
        if fragments:
            # if query_fragments:  # search for new fragment information
            self.generate_interface_fragments(out_path=des_dir.frags, write_fragments=write_fragments)
            # else:  # No fragment query, add existing fragment information to the pose
            #     if fragment_source is None:
            #         raise DesignError(f'Fragments were set for design but there were none found! Try excluding '
            #                           f'--{PUtils.no_term_constraint} in your input flags and rerun this command, or '
            #                           f'generate them separately with "{PUtils.program_command} '
            #                           f'{PUtils.generate_fragments}"')
            #
            #     self.log.debug('Fragment data found from prior query. Solving query index by Pose numbering/Entity '
            #                    'matching')
            #     self.add_fragment_query(query=fragment_source)

            for query_pair, fragment_info in self.fragment_queries.items():
                self.log.debug('Query Pair: %s, %s\n\tFragment Info:%s' % (query_pair[0].name, query_pair[1].name,
                                                                           fragment_info))
                for query_idx, entity in enumerate(query_pair):
                    entity.assign_fragments(fragments=fragment_info,
                                            alignment_type=SequenceProfile.idx_to_alignment_type[query_idx])
        for entity in self.entities:
            # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
            if entity not in self.active_entities:  # we shouldn't design, add a null profile instead
                entity.add_profile(null=True)
            else:  # add a real profile
                if self.resource_db:
                    profiles_path = self.resource_db.hhblits_profiles.location
                    entity.sequence_file = self.resource_db.sequences.retrieve_file(name=entity.name)
                    entity.evolutionary_profile = self.resource_db.hhblits_profiles.retrieve_data(name=entity.name)
                    if not entity.evolutionary_profile:
                        entity.add_evolutionary_profile(out_path=profiles_path)
                    else:  # ensure the file is attached as well
                        entity.pssm_file = self.resource_db.hhblits_profiles.retrieve_file(name=entity.name)

                    if not entity.pssm_file:  # still no file found. this is likely broken
                        raise DesignError(f'{entity.name} has no profile generated. To proceed with this design/'
                                          f'protocol you must generate the profile!')
                    if len(entity.evolutionary_profile) != entity.number_of_residues:
                        # profile was made with reference or the sequence has inserts and deletions of equal length
                        # A more stringent check could move through the evolutionary_profile[idx]['type'] key versus the
                        # entity.sequence[idx]
                        entity.fit_evolutionary_profile_to_structure()
                else:
                    profiles_path = des_dir.profiles

                if not entity.sequence_file:  # Todo move up to line 2749?
                    entity.write_sequence_to_fasta('reference', out_path=des_dir.sequences)
                entity.add_profile(evolution=evolution, fragments=fragments, out_path=profiles_path)

        # Update PoseDirectory with design information
        if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
            self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
            fragment_pssm_file = self.write_pssm_file(self.fragment_profile, PUtils.fssm, out_path=des_dir.data)

        if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
            self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
            self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=des_dir.data)

        self.combine_profile([entity.profile for entity in self.entities])
        design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=des_dir.data)
        # -------------------------------------------------------------------------
        # self.solve_consensus()
        # -------------------------------------------------------------------------

    def return_fragment_observations(self) -> list[dict[str, str | int | float]]:
        """Return the fragment observations identified on the pose regardless of Entity binding

        Returns:
            The fragment observations formatted as [{'mapped': int, 'paired': int, 'cluster': str, 'match': float}, ...]
        """
        observations = []
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        for query_pair, fragment_matches in self.fragment_queries.items():
            observations.extend(fragment_matches)

        return observations

    def return_fragment_metrics(self, fragments: list[dict] = None, by_interface: bool = False, by_entity: bool = False,
                                entity1: Structure = None, entity2: Structure = None) -> dict:
        """Return fragment metrics from the Pose. Entire Pose unless by_interface or by_entity is used

        Uses data from self.fragment_queries unless fragments are passed
        Args:
            fragments: A list of fragment observations
            by_interface: Return fragment metrics for each particular interface found in the Pose
            by_entity: Return fragment metrics for each Entity found in the Pose
            entity1: The first Entity object to identify the interface if per_interface=True
            entity2: The second Entity object to identify the interface if per_interface=True
        Returns:
            {query1: {all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap,
                      central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d}, ... }
        """
        # Todo consolidate return to (dict[(dict)]) like by_entity
        # Todo once moved to pose, incorporate these?
        #  'fragment_cluster_ids': ','.join(clusters),
        #  'total_interface_residues': total_residues,
        #  'percent_residues_fragment_total': percent_interface_covered,
        #  'percent_residues_fragment_center': percent_interface_matched,

        if fragments:
            return format_fragment_metrics(calculate_match_metrics(fragments))

        # self.calculate_fragment_query_metrics()  # populates self.fragment_metrics
        if not self.fragment_metrics:
            for query_pair, fragment_matches in self.fragment_queries.items():
                self.fragment_metrics[query_pair] = calculate_match_metrics(fragment_matches)

        if by_interface:
            if entity1 and entity2:
                for query_pair, metrics in self.fragment_metrics.items():
                    if not metrics:
                        continue
                    if (entity1, entity2) in query_pair or (entity2, entity1) in query_pair:
                        return format_fragment_metrics(metrics)
                self.log.info('Couldn\'t locate query metrics for Entity pair %s, %s' % (entity1.name, entity2.name))
            else:
                self.log.error('%s: entity1 or entity1 can\'t be None!' % self.return_fragment_metrics.__name__)

            return fragment_metric_template
        elif by_entity:
            metric_d = {}
            for query_pair, metrics in self.fragment_metrics.items():
                if not metrics:
                    continue
                for idx, entity in enumerate(query_pair):
                    if entity not in metric_d:
                        metric_d[entity] = fragment_metric_template

                    align_type = SequenceProfile.idx_to_alignment_type[idx]
                    metric_d[entity]['center_residues'].union(metrics[align_type]['center']['residues'])
                    metric_d[entity]['total_residues'].union(metrics[align_type]['total']['residues'])
                    metric_d[entity]['nanohedra_score'] += metrics[align_type]['total']['score']
                    metric_d[entity]['nanohedra_score_center'] += metrics[align_type]['center']['score']
                    metric_d[entity]['multiple_fragment_ratio'] += metrics[align_type]['multiple_ratio']
                    metric_d[entity]['number_fragment_residues_total'] += metrics[align_type]['total']['number']
                    metric_d[entity]['number_fragment_residues_center'] += metrics[align_type]['center']['number']
                    metric_d[entity]['number_fragments'] += metrics['total']['observations']
                    metric_d[entity]['percent_fragment_helix'] += metrics[align_type]['index_count'][1]
                    metric_d[entity]['percent_fragment_strand'] += metrics[align_type]['index_count'][2]
                    metric_d[entity]['percent_fragment_coil'] += (metrics[align_type]['index_count'][3] +
                                                                  metrics[align_type]['index_count'][4] +
                                                                  metrics[align_type]['index_count'][5])
            for entity in metric_d:
                metric_d[entity]['percent_fragment_helix'] /= metric_d[entity]['number_fragments']
                metric_d[entity]['percent_fragment_strand'] /= metric_d[entity]['number_fragments']
                metric_d[entity]['percent_fragment_coil'] /= metric_d[entity]['number_fragments']

            return metric_d
        else:
            metric_d = fragment_metric_template
            for query_pair, metrics in self.fragment_metrics.items():
                if not metrics:
                    continue
                metric_d['center_residues'].union(
                    metrics['mapped']['center']['residues'].union(metrics['paired']['center']['residues']))
                metric_d['total_residues'].union(
                    metrics['mapped']['total']['residues'].union(metrics['paired']['total']['residues']))
                metric_d['nanohedra_score'] += metrics['total']['total']['score']
                metric_d['nanohedra_score_center'] += metrics['total']['center']['score']
                metric_d['multiple_fragment_ratio'] += metrics['total']['multiple_ratio']
                metric_d['number_fragment_residues_total'] += metrics['total']['total']['number']
                metric_d['number_fragment_residues_center'] += metrics['total']['center']['number']
                metric_d['number_fragments'] += metrics['total']['observations']
                metric_d['percent_fragment_helix'] += metrics['total']['index_count'][1]
                metric_d['percent_fragment_strand'] += metrics['total']['index_count'][2]
                metric_d['percent_fragment_coil'] += (metrics['total']['index_count'][3] +
                                                      metrics['total']['index_count'][4] +
                                                      metrics['total']['index_count'][5])
            try:
                metric_d['percent_fragment_helix'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
                metric_d['percent_fragment_strand'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
                metric_d['percent_fragment_coil'] /= (metric_d['number_fragments'] * 2)  # account for 2x observations
            except ZeroDivisionError:
                metric_d['percent_fragment_helix'], metric_d['percent_fragment_strand'], \
                    metric_d['percent_fragment_coil'] = 0, 0, 0

            return metric_d

    # def calculate_fragment_query_metrics(self):
    #     """From the profile's fragment queries, calculate and store the query metrics per query"""
    #     for query_pair, fragment_matches in self.fragment_queries.items():
    #         self.fragment_metrics[query_pair] = calculate_match_metrics(fragment_matches)

    # def return_fragment_info(self):
    #     clusters, residue_numbers, match_scores = [], [], []
    #     for query_pair, fragments in self.fragment_queries.items():
    #         for query_idx, entity_name in enumerate(query_pair):
    #             clusters.extend([fragment['cluster'] for fragment in fragments])

    # Todo use this or below if columns are cleaned
    def residue_processing(self, design_scores: dict[str, dict[str, float | str]], columns: list[str]) -> \
            dict[str, dict[int, dict[str, float | list]]]:
        """Process Residue Metrics from Rosetta score dictionary (One-indexed residues)

        Args:
            design_scores: {'001': {'buns': 2.0, 'per_res_energy_complex_15A': -2.71, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...'}, ...}
            columns: ['per_res_energy_complex_5', 'per_res_energy_1_unbound_5', ...]
        Returns:
            {'001': {15: {'type': 'T', 'energy': {'complex': -2.71, 'unbound': [-1.9, 0]}, 'fsp': 0., 'cst': 0.}, ...},
             ...}
        """
        # energy_template = {'complex': 0., 'unbound': 0., 'fsp': 0., 'cst': 0.}
        residue_template = {'energy': {'complex': 0., 'unbound': [0. for ent in self.entities], 'fsp': 0., 'cst': 0.}}
        pose_length = self.number_of_residues
        # adjust the energy based on pose specifics
        pose_energy_multiplier = self.number_of_symmetry_mates  # will be 1 if not symmetric
        entity_energy_multiplier = [entity.number_of_monomers for entity in self.entities]

        warn = False
        parsed_design_residues = {}
        for design, scores in design_scores.items():
            residue_data = {}
            for column in columns:
                if column not in scores:
                    continue
                metadata = column.strip('_').split('_')
                # remove chain_id in rosetta_numbering="False"
                # if we have enough chains, weird chain characters appear "per_res_energy_complex_19_" which mess up
                # split. Also numbers appear, "per_res_energy_complex_1161" which may indicate chain "1" or residue 1161
                residue_number = int(metadata[-1].translate(digit_translate_table))
                if residue_number > pose_length:
                    if not warn:
                        warn = True
                        logger.warning(
                            'Encountered %s which has residue number > the pose length (%d). If this system is '
                            'NOT a large symmetric system and output_as_pdb_nums="true" was used in Rosetta '
                            'PerResidue SimpleMetrics, there is an error in processing that requires your '
                            'debugging. Otherwise, this is likely a numerical chain and will be treated under '
                            'that assumption. Always ensure that output_as_pdb_nums="true" is set'
                            % (column, pose_length))
                    residue_number = residue_number[:-1]
                if residue_number not in residue_data:
                    residue_data[residue_number] = deepcopy(residue_template)  # deepcopy(energy_template)

                metric = metadata[2]  # energy [or sasa]
                if metric != 'energy':
                    continue
                pose_state = metadata[-2]  # unbound or complex [or fsp (favor_sequence_profile) or cst (constraint)]
                entity_or_complex = metadata[3]  # 1,2,3,... or complex

                # use += because instances of symmetric residues from symmetry related chains are summed
                try:  # to convert to int. Will succeed if we have an entity value, ex: 1,2,3,...
                    entity = int(entity_or_complex) - index_offset
                    residue_data[residue_number][metric][pose_state][entity] += \
                        (scores.get(column, 0) / entity_energy_multiplier[entity])
                except ValueError:  # complex is the value, use the pose state
                    residue_data[residue_number][metric][pose_state] += (scores.get(column, 0) / pose_energy_multiplier)
            parsed_design_residues[design] = residue_data

        return parsed_design_residues

    def rosetta_residue_processing(self, design_scores: dict[str, dict[str, float | str]]) -> \
            dict[str, dict[int, dict[str, float | list]]]:
        """Process Residue Metrics from Rosetta score dictionary (One-indexed residues) accounting for symmetric energy

        Args:
            design_scores: {'001': {'buns': 2.0, 'per_res_energy_complex_15A': -2.71, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...'}, ...}
        Returns:
            {'001': {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                          'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                          'fsp': 0., 'cst': 0.},
                     ...},
             ...}
        """
        energy_template = {'complex': 0., 'bound': [0. for _ in self.entities], 'unbound': [0. for _ in self.entities],
                           'solv_complex': 0., 'solv_bound': [0. for _ in self.entities],
                           'solv_unbound': [0. for _ in self.entities], 'fsp': 0., 'cst': 0.}
        # residue_template = {'energy': {'complex': 0., 'fsp': 0., 'cst': 0.,
        #                                'unbound': [0. for _ in self.entities], 'bound': [0. for _ in self.entities]}}
        pose_length = self.number_of_residues
        # adjust the energy based on pose specifics
        pose_energy_multiplier = self.number_of_symmetry_mates  # will be 1 if not symmetric
        entity_energy_multiplier = [entity.number_of_monomers for entity in self.entities]

        warn = False
        parsed_design_residues = {}
        for design, scores in design_scores.items():
            residue_data = {}
            for key, value in scores.items():
                if not key.startswith('per_res_'):
                    continue
                # per_res_energysolv_complex_15W or per_res_energysolv_2_bound_415B
                metadata = key.strip('_').split('_')
                # remove chain_id in rosetta_numbering="False"
                # if we have enough chains, weird chain characters appear "per_res_energy_complex_19_" which mess up
                # split. Also numbers appear, "per_res_energy_complex_1161" which may indicate chain "1" or residue 1161
                residue_number = int(metadata[-1].translate(digit_translate_table))
                if residue_number > pose_length:
                    if not warn:
                        warn = True
                        logger.warning(
                            f'Encountered {key} which has residue number > the pose length ({pose_length}). If this '
                            'system is NOT a large symmetric system and output_as_pdb_nums="true" was used in Rosetta '
                            'PerResidue SimpleMetrics, there is an error in processing that requires your '
                            'debugging. Otherwise, this is likely a numerical chain and will be treated under '
                            'that assumption. Always ensure that output_as_pdb_nums="true" is set'
                        )
                    residue_number = residue_number[:-1]
                if residue_number not in residue_data:
                    residue_data[residue_number] = deepcopy(energy_template)  # deepcopy(residue_template)
                metric = metadata[2]  # energy [or sasa]
                if metric == 'energy':
                    pose_state = metadata[-2]  # un, bound, complex [or fsp (favor_sequence_profile), cst (constraint)]
                    entity_or_complex = metadata[3]  # 1,2,3,... or complex
                    # use += because instances of symmetric residues from symmetry related chains are summed
                    try:  # to convert to int. Will succeed if we have an entity as a string integer, ex: 1,2,3,...
                        entity = int(entity_or_complex) - index_offset
                        residue_data[residue_number][pose_state][entity] += (value / entity_energy_multiplier[entity])
                    except ValueError:  # complex is the value, use the pose state
                        residue_data[residue_number][pose_state] += (value / pose_energy_multiplier)
                elif metric == 'energysolv':
                    pose_state = metadata[-2]  # unbound, bound, complex
                    entity_or_complex = metadata[3]  # 1,2,3,... or complex
                    # use += because instances of symmetric residues from symmetry related chains are summed
                    try:  # to convert to int. Will succeed if we have an entity as a string integer, ex: 1,2,3,...
                        entity = int(entity_or_complex) - index_offset
                        residue_data[residue_number][f'solv_{pose_state}'][entity] += \
                            (value / entity_energy_multiplier[entity])
                    except ValueError:  # complex is the value, use the pose state
                        residue_data[residue_number][f'solv_{pose_state}'] += (value / pose_energy_multiplier)
                # else:  # sasa or something else old
                #     pass
            parsed_design_residues[design] = residue_data

        return parsed_design_residues

    # def renumber_fragments_to_pose(self, fragments):
    #     for idx, fragment in enumerate(fragments):
    #         # if self.pdb.residue_from_pdb_numbering():
    #         # only assign the new fragment number info to the fragments if the residue is found
    #         map_pose_number = self.residue_number_from_pdb(fragment['mapped'])
    #         fragment['mapped'] = map_pose_number if map_pose_number else fragment['mapped']
    #         pair_pose_number = self.residue_number_from_pdb(fragment['paired'])
    #         fragment['paired'] = pair_pose_number if pair_pose_number else fragment['paired']
    #         # fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
    #         # fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
    #         fragments[idx] = fragment
    #
    #     return fragments

    # def add_fragment_query(self, entity1: Entity = None, entity2: Entity = None, query=None, pdb_numbering: bool = False):
    #     """For a fragment query loaded from disk between two entities, add the fragment information to the Pose"""
    #     # Todo This function has logic pitfalls if residue numbering is in PDB format. How easy would
    #     #  it be to refactor fragment query to deal with the chain info from the frag match file?
    #     if pdb_numbering:  # Renumber self.fragment_map and self.fragment_profile to Pose residue numbering
    #         query = self.renumber_fragments_to_pose(query)
    #         # for idx, fragment in enumerate(fragment_source):
    #         #     fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
    #         #     fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
    #         #     fragment_source[idx] = fragment
    #         if entity1 and entity2 and query:
    #             self.fragment_queries[(entity1, entity2)] = query
    #     else:
    #         entity_pairs = [(self.entity_from_residue(fragment['mapped']),
    #                          self.entity_from_residue(fragment['paired'])) for fragment in query]
    #         if all([all(pair) for pair in entity_pairs]):
    #             for entity_pair, fragment in zip(entity_pairs, query):
    #                 if entity_pair in self.fragment_queries:
    #                     self.fragment_queries[entity_pair].append(fragment)
    #                 else:
    #                     self.fragment_queries[entity_pair] = [fragment]
    #         else:
    #             raise DesignError('%s: Couldn\'t locate Pose Entities passed by residue number. Are the residues in '
    #                               'Pose Numbering? This may be occurring due to fragment queries performed on the PDB '
    #                               'and not explicitly searching using pdb_numbering = True. Retry with the appropriate'
    #                               ' modifications' % self.add_fragment_query.__name__)

    # def connect_fragment_database(self, source: str = PUtils.biological_interfaces, **kwargs):
    #     """Generate a FragmentDatabase connection
    #
    #     Args:
    #         source: The type of FragmentDatabase to connect
    #     Sets:
    #         self.fragment_db (FragmentDatabase)
    #     """
    #     self.fragment_db = fragment_factory(source=source, **kwargs)

    def generate_interface_fragments(self, write_fragments: bool = True, out_path: str | bytes = None):
        """Generate fragments between the Pose interface(s). Finds interface(s) if not already available

        Args:
            write_fragments: Whether to write the located fragments
            out_path: The location to write each fragment file
        """
        if not self.interface_residues:
            self.find_and_split_interface()

        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.log.debug('Querying Entity pair: %s, %s for interface fragments'
                           % tuple(entity.name for entity in entity_pair))
            self.query_interface_for_fragments(*entity_pair)

        if write_fragments:
            self.write_fragment_pairs(self.fragment_pairs, out_path=out_path)
            frag_file = os.path.join(out_path, PUtils.frag_text_file)
            if os.path.exists(frag_file):
                os.system(f'rm {frag_file}')  # ensure old file is removed before new write
            for match_count, (ghost_frag, surface_frag, match) in enumerate(self.fragment_pairs, 1):
                write_frag_match_info_file(ghost_frag=ghost_frag, matched_frag=surface_frag,
                                           overlap_error=z_value_from_match_score(match),
                                           match_number=match_count, out_path=out_path)

    def write_fragment_pairs(self, ghost_mono_frag_pairs: list[tuple[GhostFragment, MonoFragment, float]],
                             out_path: str | bytes = os.getcwd()):
        ghost_frag: GhostFragment
        mono_frag: MonoFragment
        for idx, (ghost_frag, mono_frag, match_score) in enumerate(ghost_mono_frag_pairs, 1):
            ijk = ghost_frag.get_ijk()
            fragment_pdb, _ = dictionary_lookup(self.fragment_db.paired_frags, ijk)
            trnsfmd_fragment = fragment_pdb.return_transformed_copy(**ghost_frag.transformation)
            trnsfmd_fragment.write(out_path=os.path.join(out_path, f'%d_%d_%d_fragment_match_{idx}.pdb' % ijk))

    def format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            **kwargs
        Returns:
            The PDB formatted SEQRES record
        """
        # if self.reference_sequence:
        formated_reference_sequence = {entity.chain_id: entity.reference_sequence for entity in self.entities}
        formated_reference_sequence = \
            {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in sequence)))
             for chain, sequence in formated_reference_sequence.items()}
        chain_lengths = {chain: len(sequence) for chain, sequence in formated_reference_sequence.items()}
        return '%s\n' \
               % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
                           % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                           for chain, sequence in formated_reference_sequence.items()
                           for line_number in range(1, 1 + ceil(len(sequence)/seq_res_len)))
        # else:
        #     return ''

    def debug_pdb(self, tag: str = None):
        """Write out all Structure objects for the Pose PDB"""
        with open(f'{f"{tag}_" if tag else ""}POSE_DEBUG_{self.name}.pdb', 'w') as f:
            available_chain_ids = self.return_chain_generator()
            for entity_idx, entity in enumerate(self.entities, 1):
                f.write('REMARK 999   Entity %d - ID %s\n' % (entity_idx, entity.name))
                entity.write(file_handle=f, chain=next(available_chain_ids))
                for chain_idx, chain in enumerate(entity.chains, 1):
                    f.write('REMARK 999   Entity %d - ID %s   Chain %d - ID %s\n'
                            % (entity_idx, entity.name, chain_idx, chain.chain_id))
                    chain.write(file_handle=f, chain=next(available_chain_ids))

    # def get_interface_surface_area(self):
    #     # pdb1_interface_sa = entity1.get_surface_area_residues(entity1_residue_numbers)
    #     # pdb2_interface_sa = entity2.get_surface_area_residues(self.interface_residues or entity2_residue_numbers)
    #     # interface_buried_sa = pdb1_interface_sa + pdb2_interface_sa
    #     return


def subdirectory(name):
    return name[1:2]


# @njit
def find_fragment_overlap(entity1_coords: np.ndarray, residues1: list[Residue] | Residues,
                          residues2: list[Residue] | Residues, fragdb: FragmentDatabase = None,
                          euler_lookup: EulerLookup = None, max_z_value: float = 2.) -> \
        list[tuple[GhostFragment, Fragment, float]]:
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From two sets of Residues, score the fragment overlap according to Nanohedra's fragment matching

    Args:
        entity1_coords:
        residues1:
        residues2:
        fragdb:
        euler_lookup:
        max_z_value:
    """
    if not fragdb:
        fragdb = fragment_factory()  # FragmentDB()

    if not euler_lookup:
        euler_lookup = euler_factory()

    # logger.debug('Starting Ghost Frag Lookup')
    oligomer1_bb_tree = BallTree(entity1_coords)
    ghost_frags1: list[GhostFragment] = []
    for residue in residues1:
        ghost_frags1.extend(residue.get_ghost_fragments(fragdb.indexed_ghosts, clash_tree=oligomer1_bb_tree))
    # for frag1 in interface_frags1:
    #     ghostfrags = frag1.get_ghost_fragments(fragdb.indexed_ghosts, clash_tree=oligomer1_bb_tree)
    #     if ghostfrags:
    #         ghost_frags1.extend(ghostfrags)
    logger.debug('Finished Ghost Frag Lookup')

    # Get fragment guide coordinates
    residue1_ghost_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in ghost_frags1])
    residue2_guide_coords = np.array([residue.guide_coords for residue in residues2])
    # interface_surf_frag_guide_coords = np.array([residue.guide_coords for residue in interface_residues2])

    # Check for matching Euler angles
    # TODO create a stand alone function
    # logger.debug('Starting Euler Lookup')
    overlapping_ghost_indices, overlapping_frag_indices = \
        euler_lookup.check_lookup_table(residue1_ghost_guide_coords, residue2_guide_coords)
    # logger.debug('Finished Euler Lookup')
    logger.debug(f'Found {len(overlapping_ghost_indices)} overlapping fragments')
    # filter array by matching type for surface (i) and ghost (j) frags
    ghost_type_array = np.array([ghost_frags1[idx].frag_type for idx in overlapping_ghost_indices.tolist()])
    mono_type_array = np.array([residues2[idx].frag_type for idx in overlapping_frag_indices.tolist()])
    ij_type_match = np.where(mono_type_array == ghost_type_array, True, False)

    passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
    passing_frag_indices = overlapping_frag_indices[ij_type_match]
    logger.debug(f'Found {len(passing_ghost_indices)} overlapping fragments in the same i/j type')

    passing_ghost_coords = residue1_ghost_guide_coords[passing_ghost_indices]
    passing_frag_coords = residue2_guide_coords[passing_frag_indices]
    # precalculate the reference_rmsds for each ghost fragment
    reference_rmsds = np.array([ghost_frags1[ghost_idx].rmsd for ghost_idx in passing_ghost_indices.tolist()])
    reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)

    # logger.debug('Calculating passing fragment overlaps by RMSD')
    all_fragment_overlap = \
        calculate_overlap(passing_ghost_coords, passing_frag_coords, reference_rmsds, max_z_value=max_z_value)
    # logger.debug('Finished calculating fragment overlaps')
    passing_overlap_indices = np.flatnonzero(all_fragment_overlap)
    logger.debug(f'Found {len(passing_overlap_indices)} overlapping fragments under the {max_z_value} threshold')

    # interface_ghostfrags = [ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()]
    # interface_monofrags2 = [residues2[idx] for idx in passing_surf_indices[passing_overlap_indices].tolist()]
    # passing_z_values = all_fragment_overlap[passing_overlap_indices]
    # match_scores = match_score_from_z_value(all_fragment_overlap[passing_overlap_indices])

    return list(zip([ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()],
                    [residues2[idx] for idx in passing_frag_indices[passing_overlap_indices].tolist()],
                    match_score_from_z_value(all_fragment_overlap[passing_overlap_indices]).tolist()))


def get_matching_fragment_pairs_info(ghostfrag_surffrag_pairs):
    """From a ghost fragment/surface fragment pair and corresponding match score, return the pertinent interface
    information

    Args:
        ghostfrag_surffrag_pairs (list[tuple]): Observed ghost and surface fragment overlaps and their match score
    Returns:
        (list[dict[mapping[str,any]]])
    """
    fragment_matches = []
    for interface_ghost_frag, interface_surf_frag, match_score in ghostfrag_surffrag_pairs:
        _, surffrag_resnum1 = interface_ghost_frag.get_aligned_chain_and_residue()  # surffrag_ch1,
        _, surffrag_resnum2 = interface_surf_frag.get_central_res_tup()  # surffrag_ch2,
        # Todo
        #  surf_frag_central_res_num1 = interface_ghost_residue.number
        #  surf_frag_central_res_num2 = interface_surf_residue.number
        fragment_matches.append(dict(zip(('mapped', 'paired', 'match', 'cluster'),
                                     (surffrag_resnum1, surffrag_resnum2,  match_score,
                                      '%d_%d_%d' % interface_ghost_frag.get_ijk()))))
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


# def calculate_interface_score(interface_pdb, write=False, out_path=os.getcwd()):
#     """Takes as input a single PDB with two chains and scores the interface using fragment decoration"""
#     interface_name = interface_pdb.name
#
#     entity1 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_ids[0]).atoms)
#     entity1.update_attributes_from_pdb(interface_pdb)
#     entity2 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_ids[-1]).atoms)
#     entity2.update_attributes_from_pdb(interface_pdb)
#
#     interacting_residue_pairs = find_interface_pairs(entity1, entity2)
#
#     entity1_interface_residue_numbers, entity2_interface_residue_numbers = \
#         get_interface_fragment_residue_numbers(entity1, entity2, interacting_residue_pairs)
#     # entity1_ch_interface_residue_numbers, entity2_ch_interface_residue_numbers = \
#     #     get_interface_fragment_chain_residue_numbers(entity1, entity2)
#
#     entity1_interface_sa = entity1.get_surface_area_residues(entity1_interface_residue_numbers)
#     entity2_interface_sa = entity2.get_surface_area_residues(entity2_interface_residue_numbers)
#     interface_buried_sa = entity1_interface_sa + entity2_interface_sa
#
#     interface_frags1 = entity1.get_fragments(residue_numbers=entity1_interface_residue_numbers)
#     interface_frags2 = entity2.get_fragments(residue_numbers=entity2_interface_residue_numbers)
#     entity1_coords = entity1.coords
#
#     ghostfrag_surfacefrag_pairs = find_fragment_overlap(entity1_coords, interface_frags1, interface_frags2)
#     # fragment_matches = find_fragment_overlap(entity1, entity2, entity1_interface_residue_numbers,
#     #                                                       entity2_interface_residue_numbers)
#     fragment_matches = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
#     if write:
#         write_fragment_pairs(ghostfrag_surfacefrag_pairs, out_path=out_path)
#
#     # all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
#     #     central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d = \
#     #     calculate_match_metrics(fragment_matches)
#
#     match_metrics = calculate_match_metrics(fragment_matches)
#     # Todo
#     #   'mapped': {'center': {'residues' (int): (set), 'score': (float), 'number': (int)},
#     #                         'total': {'residues' (int): (set), 'score': (float), 'number': (int)},
#     #                         'match_scores': {residue number(int): (list[score (float)]), ...},
#     #                         'index_count': {index (int): count (int), ...},
#     #                         'multiple_ratio': (float)}
#     #              'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
#     #              'total': {'center': {'score': , 'number': },
#     #                        'total': {'score': , 'number': },
#     #                        'index_count': , 'multiple_ratio': , 'observations': (int)}
#     #              }
#
#     total_residues = {'A': set(), 'B': set()}
#     for pair in interacting_residue_pairs:
#         total_residues['A'].add(pair[0])
#         total_residues['B'].add(pair[1])
#
#     total_residues = len(total_residues['A']) + len(total_residues['B'])
#
#     percent_interface_matched = central_residues_with_fragment_overlap / total_residues
#     percent_interface_covered = total_residues_with_fragment_overlap / total_residues
#
#     interface_metrics = {'nanohedra_score': all_residue_score,
#                          'nanohedra_score_central': center_residue_score,
#                          'fragments': fragment_matches,
#                          'multiple_fragment_ratio': multiple_frag_ratio,
#                          'number_fragment_residues_central': central_residues_with_fragment_overlap,
#                          'number_fragment_residues_all': total_residues_with_fragment_overlap,
#                          'total_interface_residues': total_residues,
#                          'number_fragments': len(fragment_matches),
#                          'percent_residues_fragment_total': percent_interface_covered,
#                          'percent_residues_fragment_center': percent_interface_matched,
#                          'percent_fragment_helix': fragment_content_d['1'],
#                          'percent_fragment_strand': fragment_content_d['2'],
#                          'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
#                          + fragment_content_d['5'],
#                          'interface_area': interface_buried_sa}
#
#     return interface_name, interface_metrics


def get_interface_fragment_residue_numbers(pdb1, pdb2, interacting_pairs):
    # Get interface fragment information
    pdb1_residue_numbers, pdb2_residue_numbers = set(), set()
    for pdb1_central_res_num, pdb2_central_res_num in interacting_pairs:
        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.residue_number in pdb1_res_num_list:
                if atom.is_ca():
                    frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.residue_number in pdb2_res_num_list:
                if atom.is_ca():
                    frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            pdb1_residue_numbers.add(pdb1_central_res_num)
            pdb2_residue_numbers.add(pdb2_central_res_num)

    return pdb1_residue_numbers, pdb2_residue_numbers


def get_interface_fragment_chain_residue_numbers(pdb1, pdb2, cb_distance=8):
    """Given two PDBs, return the unique chain and interacting residue lists"""
    pdb1_cb_coords = pdb1.get_cb_coords()
    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_coords = pdb2.get_cb_coords()
    pdb2_cb_indices = pdb2.cb_indices

    pdb1_cb_kdtree = BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))


def get_multi_chain_interface_fragment_residue_numbers(pdb1, pdb2, interacting_pairs):
    # Get interface fragment information
    pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list = [], []
    for pair in interacting_pairs:

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    if atom.is_ca():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    if atom.is_ca():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_central_chainid_resnum_unique_list:
                pdb1_central_chainid_resnum_unique_list.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_central_chainid_resnum_unique_list:
                pdb2_central_chainid_resnum_unique_list.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list


def split_residue_pairs(interface_pairs: list[tuple[Residue, Residue]]) -> tuple[list[Residue], list[Residue]]:
    """Used to split Residue pairs, sort by Residue.number, and return pairs separated by index"""
    if interface_pairs:
        residues1, residues2 = zip(*interface_pairs)
        return sorted(set(residues1), key=lambda residue: residue.number), \
            sorted(set(residues2), key=lambda residue: residue.number)
    else:
        return [], []


# def split_interface_numbers(interface_pairs) -> tuple[list[int], list[int]]:
#     """Used to split residue number pairs"""
#     if interface_pairs:
#         numbers1, numbers2 = zip(*interface_pairs)
#         return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
#     else:
#         return [], []


def split_number_pairs_and_sort(pairs: list[tuple[int, int]]) -> tuple[list, list]:
    """Used to split integer pairs and sort, and return pairs separated by index"""
    if pairs:
        numbers1, numbers2 = zip(*pairs)
        return sorted(set(numbers1), key=int), sorted(set(numbers2), key=int)
    else:
        return [], []
