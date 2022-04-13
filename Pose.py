import os
from copy import copy
from pickle import load
from itertools import chain as iter_chain, combinations_with_replacement, combinations, product
from math import sqrt, cos, sin, prod, ceil
from typing import Set, List, Iterable
# from operator import itemgetter

import numpy as np
# from numba import njit, jit
from Bio.Data.IUPACData import protein_letters_1to3_extended
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
# import requests

import PathUtils as PUtils
from SymDesignUtils import pickle_object, DesignError, calculate_overlap, z_value_from_match_score, \
    start_log, null_log, match_score_from_z_value, split_interface_residues, dictionary_lookup, \
    split_interface_numbers
from classes.SymEntry import get_rot_matrices, rotation_range, get_degen_rotmatrices, SymEntry, flip_x_matrix, \
    possible_symmetries, point_group_setting_matrix_members, setting_matrices, symmetry_combination_format
from utils.GeneralUtils import write_frag_match_info_file, transform_coordinate_sets
from utils.SymmetryUtils import valid_subunit_number, sg_cryst1_fmt_dict, pg_cryst1_fmt_dict, sg_zvalues, \
    get_ptgrp_sym_op, generate_cryst1_record
from classes.EulerLookup import EulerLookup
from PDB import PDB
from SequenceProfile import SequenceProfile
from DesignMetrics import calculate_match_metrics, fragment_metric_template, format_fragment_metrics
from Structure import Coords, Structure, Structures  # Atoms, Residues,
from Database import FragmentDB, FragmentDatabase

# Globals
logger = start_log(name=__name__)
seq_res_len = 52
config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location

# Todo
# class BaseModel:
#     """Grab all unused args that could be based to any model type class"""
#     def __init__(self, pdb=None, models=None, log=None,
#                  TODO put all structure stuff here
#                  **kwargs):
#         super().__init__(**kwargs)


class MultiModel:
    """Class for working with iterables of State objects of macromolecular polymers (proteins for now). Each State
    container comprises Structure object(s) which can also be accessed as a unique Model by slicing the Structure across
    States.

    self.structures holds each of the individual Structure objects which are involved in the MultiModel. As of now,
    no checks are made as to whether the identity of these is the same accross States"""
    def __init__(self, model=None, models=None, state=None, states=None, independent=False, log=None, **kwargs):
        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if model:
            # if not isinstance(model, Model):  # TODO?
            if not isinstance(model, Models):
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
    def independents(self) -> Set[int]:
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

    def enumerate_models(self) -> List:
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
# class State(Structure):  # todo subclass UserList (https://docs.python.org/3/library/collections.html#userlist-objects
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

    def write(self, increment_chains=False, **kwargs):  # out_path=os.getcwd(), file_handle=None, header=None,
        """Write Structures to a file specified by out_path or with a passed file_handle.

        Keyword Args:
            out_path=os.getcwd() (str): The path to write the Models to
            file_handle=None (io.TextIOWrapper): A file handle to write the Models to
            header=None (str): If there is header information that should be included. Pass new lines with a \n
            increment_chains=False (bool): Whether or not to write each Model with a new chain name.
                Default (False) writes as a new MODEL for each Model
            kwargs
        Returns:
            (str): The filename if one was written
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


class Models(Structures):
    """Keep track of different variations of the same Structure object such as altered coordinates (different decoy's or
     symmetric copies) or where Residues are mutated. In PDB parlance, this would be a multimodel with a single chain,
     but could be multiple PDB's with some common element.

    If you have multiple Structures with Multiple States, use the MultiModel class to store and retrieve that data
    """
    def __init__(self, models=None, **kwargs):  # log=None,
        super().__init__(structures=models, **kwargs)
        # print('Initializing Models')

        # super().__init__()  # without passing **kwargs, there is no need to ensure base Object class is protected
        # if log:
        #     self.log = log
        # elif log is None:
        #     self.log = null_log
        # else:  # When log is explicitly passed as False, use the module logger
        #     self.log = logger

        if self.structures:
            self.models = self.structures  # Todo is this reference to structures via models stable? ENSURE it is

    @classmethod
    def from_file(cls, file, **kwargs):
        """Construct Models from multimodel PDB file using the PDB.chains
        Ex: [Chain1, Chain1, ...]
        """
        pdb = PDB.from_file(file, **kwargs)  # Todo make independent parsing function
        # new_model = cls(models=pdb.chains)
        return cls(models=pdb.chains, **kwargs)

    @classmethod
    def from_PDB(cls, pdb, **kwargs):
        """Construct Models from multimodel PDB file using the PDB.chains
        Ex: [Chain1, Chain1, ...]
        """
        return cls(models=pdb.chains, **kwargs)

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

    def write(self, increment_chains=False, **kwargs):  # out_path=os.getcwd(), file_handle=None, header=None,
        """Write Structures to a file specified by out_path or with a passed file_handle.

        Keyword Args:
            out_path=os.getcwd() (str): The path to write the Models to
            file_handle=None (io.TextIOWrapper): A file handle to write the Models to
            header=None (str): If there is header information that should be included. Pass new lines with a \n
            increment_chains=False (bool): Whether or not to write each Model with a new chain name.
                Default (False) writes as a new MODEL for each Model
            kwargs
        Returns:
            (str): The filename if one was written
        """
        return super().write(increment_chains=increment_chains, **kwargs)
        # if file_handle:  # Todo increment_chains compatibility
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
        #             chain = next(available_chain_ids)
        #             structure.write(file_handle=f, chain=chain)
        #             c_term_residue = structure.c_terminal_residue
        #             f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
        #                                                                   c_term_residue.type, chain,
        #                                                                   c_term_residue.number))
        #     else:
        #         for model_number, structure in enumerate(self.structures, 1):
        #             f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
        #             structure.write(file_handle=f)
        #             c_term_residue = structure.c_terminal_residue
        #             f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
        #                                                                   c_term_residue.type, structure.chain_id,
        #                                                                   c_term_residue.number))
        #             f.write('ENDMDL\n')


# (BaseModel)?
class Model:  # Todo (Structure)
    """Keep track of different variations of the same Structure object such as altered coordinates (different decoy's or
     symmetric copies) or where Residues are mutated. In PDB parlance, this would be a multimodel with a single chain,
     but could be multiple PDB's with some common element.

    If you have multiple Structures with Multiple States, use the MultiModel class to store and retrieve that data
    """
    def __init__(self, pdb=None, models=None, log=None, **kwargs):
        # super().__init__(**kwargs)
        super().__init__()  # without passing **kwargs, there is no need to ensure base Object class is protected
        # self.pdb = self.models[0]
        # elif isinstance(pdb, PDB):
        # self.biomt_header = ''
        # self.biomt = []
        if log:
            self.log = log
        elif log is None:
            self.log = null_log
        else:  # When log is explicitly passed as False, use the module logger
            self.log = logger

        if pdb and isinstance(pdb, Structure):
            self.pdb = pdb  # TODO DISCONNECT HERE

        if models and isinstance(models, list):
            self.models = models
        else:
            self.models = []

    @classmethod
    def from_file(cls, file, **kwargs):
        """Construct Models from multimodel PDB file using the PDB.chains
        Ex: [Chain1, Chain1, ...]
        """
        pdb = PDB.from_file(file, **kwargs)  # Todo make independent parsing function for multimodels
        # new_model = cls(models=pdb.chains)
        return cls(models=pdb.chains, **kwargs)

    @property
    def number_of_models(self):
        return len(self.models)

    @property
    def pdb(self):  # TODO DISCONNECT HERE
        return self._pdb

    @pdb.setter
    def pdb(self, pdb):  # TODO DISCONNECT HERE
        self._pdb = pdb
        # self.coords = pdb._coords

    @property
    def coords(self):  # TODO DISCONNECT HERE
        """Return a view of the representative Coords from the Model. These may be the ASU if a SymmetricModel"""
        return self.pdb._coords.coords
        # return self._coords.coords

    @coords.setter
    def coords(self, coords):  # TODO DISCONNECT HERE
        if isinstance(coords, Coords):
            self.pdb._coords = coords
            # self._coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Structure, use the private attribute _coords')

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return self.pdb.name  # TODO DISCONNECT HERE

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def entities(self):  # TODO COMMENT OUT .pdb
        return self.pdb.entities

    @property
    def chains(self):
        return [chain for entity in self.entities for chain in entity.chains]

    @property
    def active_chains(self):
        return [chain for entity in self.active_entities for chain in entity.chains]

    @property
    def chain_breaks(self):
        return [entity.c_terminal_residue.number for entity in self.entities]

    @property
    def residues(self):  # TODO COMMENT OUT .pdb
        return self.pdb.residues

    @property
    def reference_sequence(self) -> str:
        # return ''.join(self.pdb.reference_sequence.values())
        return ''.join(entity.reference_sequence for entity in self.entities)

    def entity(self, entity):  # TODO COMMENT OUT .pdb
        return self.pdb.entity(entity)

    def chain(self, chain):  # TODO COMMENT OUT .pdb
        return self.pdb.entity_from_chain(chain)

    @property
    def atom_indices_per_entity(self):
        return [entity.atom_indices for entity in self.pdb.entities]

    @property
    def residue_indices_per_entity(self):
        return [entity.residue_indices for entity in self.pdb.entities]

    @property
    def number_of_atoms_per_entity(self):  # TODO COMMENT OUT .pdb
        return [entity.number_of_atoms for entity in self.pdb.entities]

    @property
    def number_of_atoms(self):  # TODO COMMENT OUT .pdb
        return self.pdb.number_of_atoms

    @property
    def number_of_residues(self):  # TODO COMMENT OUT .pdb
        return self.pdb.number_of_residues

    @property
    def coords_indexed_residues(self):  # TODO COMMENT OUT .pdb
        return self.pdb.coords_indexed_residues
        # try:
        #     return self._coords_indexed_residues
        # except AttributeError:
        #     self._coords_indexed_residues = [residue for residue in self.pdb.coords_indexed_residues]
        #     return self._coords_indexed_residues

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

    @property
    def center_of_mass(self):  # TODO COMMENT OUT ONCE Structure SUBCLASSED
        """The center of mass for the model Structure, either an asu, or other pdb

        Returns:
            (numpy.ndarray)
        """
        return np.matmul(np.full(self.number_of_atoms, 1 / self.number_of_atoms), self.coords)

    @property
    def model_coords(self):  # TODO RECONCILE with coords, SymmetricModel, and State variation
        """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
        return self._model_coords.coords

    @model_coords.setter
    def model_coords(self, coords):
        if isinstance(coords, Coords):
            self._model_coords = coords
        else:
            raise AttributeError('The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
                                 'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

    def format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            **kwargs
        Returns:
            (str)
        """
        if self.pdb.reference_sequence:  # TODO DISCONNECT HERE
            formated_reference_sequence = \
                {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended[aa] for aa in sequence)))
                 for chain, sequence in self.pdb.reference_sequence.items()}
            chain_lengths = {chain: len(sequence) for chain, sequence in self.pdb.reference_sequence.items()}
            return '%s\n' \
                   % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
                               % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                               for chain, sequence in formated_reference_sequence.items()
                               for line_number in range(1, 1 + ceil(len(sequence)/seq_res_len)))
        else:
            return ''

    def format_header(self, **kwargs):
        if type(self).__name__ in ['Model']:
            return self.format_biomt(**kwargs) + self.format_seqres(**kwargs)
        elif type(self).__name__ in ['Pose', 'SymmetricModel']:
            return self.format_biomt(**kwargs) + self.format_seqres(**kwargs)
        else:
            return ''

    def format_biomt(self, **kwargs):
        """Return the BIOMT record for the PDB if there was one parsed

        Returns:
            (str)
        """
        # Todo test
        if self.pdb.biomt_header != '':  # TODO DISCONNECT HERE
            return self.pdb.biomt_header
        elif self.pdb.biomt:
            return '%s\n' \
                   % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}'.format(v_idx, m_idx, *vec)
                               for m_idx, matrix in enumerate(self.pdb.biomt, 1) for v_idx, vec in enumerate(matrix, 1))
        else:
            return ''

    def write_header(self, file_handle, header=None, **kwargs) -> None:
        """Handle writing of Structure header information to the file

        Args:
            file_handle (FileObject): An open file object where the header should be written
        Keyword Args
            header (Union[None, str]): A string that is desired at the top of the .pdb file
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

    def write(self, out_path=os.getcwd(), file_handle=None, assembly=False, increment_chains=False, **kwargs) -> str:
        # header=None,
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
        one was written

        Keyword Args:
            file_handle=None (FileObject): An open file object where the header should be written
            assembly=False (bool): Whether to write the full assembly to a file
            surrounding_uc=True (bool): Whether to write the surrounding unit cell if assembly is true and
                self.dimension > 1
        Returns:
            (str)
        """
        if file_handle:  # Todo handle with multiple Structure containers
            file_handle.write('%s\n' % self.return_atom_string(**kwargs))
            return

        with open(out_path, 'w') as outfile:
            self.write_header(outfile, **kwargs)
            # if header:
            #     if isinstance(header, str):
            #         f.write(header)
            #     # if isinstance(header, Iterable):

            if type(self).__name__ in ['SymmetricModel', 'Pose']:
                if not self.symmetry:  # When Pose isn't symmetric, we don't have to consider symmetric issues
                    pass
                elif assembly:  # will make models and use next logic steps to write them out
                    self.get_assembly_symmetry_mates(**kwargs)
                # elif self.output_asu or
                # elif not self.models:
                else:  # when assembly not explicitly requested, skip models, using biomt_record/cryst_record for sym
                    for entity in self.pdb.entities:
                        entity.write(file_handle=outfile, **kwargs)
                    # Todo with Structure subclass
                    #  super().write(out_path=out_path, **kwargs)
                    return out_path

                if increment_chains:  # assembly requested, check on the mechanism of symmetric writing
                    # we won't allow incremental chains when the Model is plain as the models are all the same and
                    # therefore belong with the models label
                    available_chain_ids = Structure.return_chain_generator()
                    for structure in self.models:
                        for entity in structure.entities:  # Todo handle with multiple Structure containers
                            chain = next(available_chain_ids)
                            entity.write(file_handle=outfile, chain=chain)
                            c_term_residue = entity.c_terminal_residue
                            outfile.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER',
                                                                                        c_term_residue.atoms[-1].number + 1,
                                                                                        c_term_residue.type, chain,
                                                                                        c_term_residue.number))
                    return out_path
            # else:
            if self.models:  # these were generated if assembly=True, therefore user doesn't want to increment chains
                for model_number, structure in enumerate(self.models, 1):
                    outfile.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
                    for entity in structure.entities:  # Todo handle with multiple Structure containers
                        entity.write(file_handle=outfile)
                        c_term_residue = entity.c_terminal_residue
                        outfile.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER',
                                                                                    c_term_residue.atoms[-1].number + 1,
                                                                                    c_term_residue.type, entity.chain_id,
                                                                                    c_term_residue.number))
                    outfile.write('ENDMDL\n')
            else:
                self.pdb.write(file_handle=outfile, **kwargs)

        return out_path

    def __getitem__(self, idx):
        return self.models[idx]


class SymmetricModel(Model):
    def __init__(self, asu=None, **kwargs):
        super().__init__(**kwargs)  # log=log,
        if asu and isinstance(asu, Structure):
            self.asu = asu  # the pose specific asu
        # self.pdb = pdb
        # self.models = []
        # self.coords = []
        # self.model_coords = []
        self.assembly_tree = None  # stores a sklearn tree for coordinate searching
        self.asu_equivalent_model_idx = None
        self.coords_type = None  # coords_type
        self.dimension = None  # dimension
        self.expand_matrices = None  # expand_matrices  # Todo make expand_matrices numpy
        self.sym_entry = None
        self.symmetry = None  # symmetry  # also defined in PDB as self.space_group
        self.point_group_symmetry = None
        self.oligomeric_equivalent_model_idxs = {}
        # self.output_asu = True
        self.uc_dimensions = None  # uc_dimensions  # also defined in PDB

        if self.asu.space_group:
            kwargs.update(self.asu.symmetry.copy())
        self.set_symmetry(**kwargs)

    @classmethod
    def from_assembly(cls, assembly, symmetry=None):
        assert symmetry, 'Currently, can\'t initialize a symmetric model without the symmetry! Pass symmetry during ' \
                         'Class initialization. Or add a scout symmetry Class method to SymmetricModel.'
        return cls(models=assembly, **symmetry)

    @classmethod
    def from_asu(cls, asu, generate_symmetry_mates=True, **kwargs):
        """From an Structure representing an asu, return the SymmetricModel with generated symmetry mates

        Keyword Args:
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            (SymmetricModel)
        """
        return cls(asu=asu, generate_symmetry_mates=generate_symmetry_mates, **kwargs)

    @property
    def asu(self):
        return self._asu

    @asu.setter
    def asu(self, asu):
        self._asu = asu
        self.pdb = asu  # TODO COMMENT OUT

    @property
    def cryst_record(self) -> str:
        """Return the symmetry parameters as a CRYST1 entry

        Returns:
            (str)
        """
        try:
            return self._cryst_record
        except AttributeError:
            self._cryst_record = None if self.dimension == 0 \
                else generate_cryst1_record(self.uc_dimensions, self.symmetry)
            return self._cryst_record

    @property
    def number_of_symmetry_mates(self) -> int:
        """Describes the number of symmetry mates present in the Model

        Returns:
            (int)
        """
        try:
            return self._number_of_symmetry_mates
        except AttributeError:
            self._number_of_symmetry_mates = 1
            return self._number_of_symmetry_mates

    @number_of_symmetry_mates.setter
    def number_of_symmetry_mates(self, number_of_symmetry_mates):
        self._number_of_symmetry_mates = number_of_symmetry_mates

    @property
    def number_of_uc_symmetry_mates(self):
        """Describes the number of symmetry mates present in the Model of the unit cell

        Returns:
            (int)
        """
        try:
            return sg_zvalues[self.symmetry]
        except KeyError:
            raise KeyError('The symmetry \'%s\' is not an available unit cell at this time. If this is a point group, '
                           'adjust your code, otherwise, help expand the code to include the symmetry operators for '
                           'this symmetry group')

    # @number_of_uc_symmetry_mates.setter
    # def number_of_uc_symmetry_mates(self, number_of_uc_symmetry_mates):
    #     self._number_of_uc_symmetry_mates = number_of_uc_symmetry_mates

    @property
    def center_of_mass_symmetric(self):
        """The center of mass for the entire symmetric system

        Returns:
            (numpy.ndarray)
        """
        if self.symmetry:
            return np.matmul(np.full(self.number_of_atoms * self.number_of_symmetry_mates,
                                     1 / self.number_of_atoms * self.number_of_symmetry_mates), self.model_coords)

    @property
    def center_of_mass_symmetric_models(self):
        """The individual centers of mass for each model in the symmetric system

        Returns:
            (numpy.ndarray)
        """
        if self.symmetry:
            number_of_atoms = self.number_of_atoms
            return np.matmul(np.full(number_of_atoms, 1 / number_of_atoms),
                             np.split(self.model_coords, self.number_of_symmetry_mates))

    @property
    def center_of_mass_symmetric_entities(self):
        """The individual centers of mass for each Entity in the symmetric system

        Returns:
            (numpy.ndarray)
        """
        if self.symmetry:
            split_models = np.split(self.model_coords, self.number_of_symmetry_mates)
            self._center_of_mass_symmetric_entities = []
            for number_of_atoms, entity_indices in zip(self.number_of_atoms_per_entity, self.atom_indices_per_entity):
                self._center_of_mass_symmetric_entities.append(
                    np.matmul(np.full(number_of_atoms, 1 / number_of_atoms), split_models[:, [entity_indices]]))

            return self._center_of_mass_symmetric_entities

    @property
    def assembly(self) -> Structure:
        """Provides the Structure object containing all symmetric chains in the assembly unless the design is 2- or 3-D
        then the assembly only contains the contacting models

        Returns:
            (Structure)
        """
        try:
            return self._assembly
        except AttributeError:
            if not self.models:
                self.get_assembly_symmetry_mates()  # default to surrounding_uc generation, but only return contacting
            if self.dimension > 0:
                selected_models = self.return_asu_interaction_models()
                self.log.debug('Found selected models %s for assembly' % selected_models)
            else:
                selected_models = list(range(self.number_of_models))
            self._assembly = \
                PDB.from_chains(list(iter_chain.from_iterable(model.chains for idx, model in enumerate(self.models)
                                                              if idx in selected_models)),
                                name='assembly', log=self.log, biomt_record=self.format_biomt(),
                                cryst_record=self.cryst_record)
            return self._assembly

    # def write_assembly(self, out_path=os.getcwd(), file_handle=None, increment_chains=False, **kwargs):
    #     self.assembly.write()

    def set_symmetry(self, sym_entry=None, expand_matrices=None, symmetry=None, cryst1=None, uc_dimensions=None,
                     generate_assembly=True, generate_symmetry_mates=False, **kwargs):
        """Set the model symmetry using the CRYST1 record, or the unit cell dimensions and the Hermann–Mauguin symmetry
        notation (in CRYST1 format, ex P 4 3 2) for the Model assembly. If the assembly is a point group,
        only the symmetry is required

        Keyword Args:
            sym_entry=None (SymEntry):
            expand_matrices=None (list[list[list]]]):
            symmetry=None (str):
            cryst1=None (str):
            uc_dimensions=None (Iterable): # TODO
            generate_assembly=True (bool):
            generate_symmetry_mates=False (bool):
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        """
        if cryst1:
            uc_dimensions, symmetry = PDB.parse_cryst_record(cryst1_string=cryst1)

        if sym_entry and isinstance(sym_entry, SymEntry):
            self.sym_entry = sym_entry
            self.symmetry = sym_entry.resulting_symmetry
            self.dimension = sym_entry.dimension
            self.point_group_symmetry = sym_entry.point_group_symmetry
            if self.dimension > 0:
                self.uc_dimensions = uc_dimensions

        elif symmetry:
            if uc_dimensions:
                self.uc_dimensions = uc_dimensions
                self.symmetry = ''.join(symmetry.split())

            if symmetry in pg_cryst1_fmt_dict:  # not available yet for non-Nanohedra PG's
                self.dimension = 2
                self.symmetry = symmetry
            elif symmetry in sg_cryst1_fmt_dict:  # not available yet for non-Nanohedra SG's
                self.dimension = 3
                self.symmetry = symmetry
            elif symmetry in possible_symmetries:  # ['T', 'O', 'I']:
                self.symmetry = possible_symmetries[symmetry]
                self.point_group_symmetry = possible_symmetries[symmetry]
                self.dimension = 0

            elif self.uc_dimensions:
                raise DesignError('Symmetry %s is not available yet! If you didn\'t provide it, the symmetry was likely'
                                  ' set from a PDB file. Get the symmetry operations from the international'
                                  ' tables and add to the pickled operators if this displeases you!' % symmetry)
            else:  # when a point group besides T, O, or I is provided
                raise DesignError('Symmetry %s is not available yet! Get the canonical symm operators from %s and add '
                                  'to the pickled operators if this displeases you!' % (symmetry, PUtils.orient_dir))
        elif not symmetry:
            return  # no symmetry was provided

        if expand_matrices:
            self.expand_matrices = expand_matrices
        else:
            self.expand_matrices = get_ptgrp_sym_op(self.symmetry) if self.dimension == 0 \
                else self.get_sg_sym_op(self.symmetry)  # ensure symmetry is Hermann–Mauguin notation

        if self.asu and generate_assembly:
            self.generate_symmetric_assembly(**kwargs)
            if generate_symmetry_mates:
                self.get_assembly_symmetry_mates(**kwargs)

    def generate_symmetric_assembly(self, **kwargs):
        """Expand the asu using self.symmetry for the symmetry specification, and optional unit cell dimensions if
        self.dimension > 0. Expands assembly to complete point group, unit cell, or surrounding unit cells

        Keyword Args:
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
        """
        if self.dimension == 0:
            self.get_point_group_coords(**kwargs)
        else:
            # self.expand_uc_coords(**kwargs)
            self.get_unit_cell_coords(**kwargs)

        self.log.info('Generated %d Symmetric Models' % self.number_of_symmetry_mates)

    # def expand_uc_coords(self, **kwargs):
    #     """Expand the backbone coordinates for every symmetric copy within the unit cells surrounding a central cell
    #
    #     Keyword Args:
    #         surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
    #         return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
    #     """
    #     self.get_unit_cell_coords(**kwargs)

    def cart_to_frac(self, cart_coords):
        """Takes a numpy array of coordinates and finds the fractional coordinates from cartesian coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            cart_coords (Union[numpy.ndarray, list[list]]): The cartesian coordinates of a unit cell
        Returns:
            (Union[numpy.ndarray, None]): The fractional coordinates of a unit cell
        """
        if not self.uc_dimensions:
            self.log.error('No unit cell dimensions were passed')
            return

        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * \
            sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2 * (cos(alpha) * cos(beta) * cos(gamma)))

        # deorthogonalization matrix M
        m0 = [1 / a, -(cos(gamma) / float(a * sin(gamma))),
              (((b * cos(gamma) * c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(sin(gamma))) -
               (b * c * cos(beta) * sin(gamma))) * (1 / float(v))]
        m1 = [0, 1 / (b * np.sin(gamma)), -((a * c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(v * sin(gamma)))]
        m2 = [0, 0, (a * b * sin(gamma)) / float(v)]
        m = [m0, m1, m2]

        return np.matmul(cart_coords, np.transpose(m))

    def frac_to_cart(self, frac_coords):
        """Takes a numpy array of coordinates and finds the cartesian coordinates from fractional coordinates
        From http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm

        Args:
            frac_coords (Union[numpy.ndarray, list[list]]): The fractional coordinates of a unit cell
        Returns:
            (Union[numpy.ndarray, None]): The cartesian coordinates of a unit cell
        """
        if not self.uc_dimensions:
            self.log.error('No unit cell dimensions were passed')
            return

        a2r = np.pi / 180.0
        a, b, c, alpha, beta, gamma = self.uc_dimensions
        alpha *= a2r
        beta *= a2r
        gamma *= a2r

        # unit cell volume
        v = a * b * c * \
            sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2 * (cos(alpha) * cos(beta) * cos(gamma)))

        # orthogonalization matrix m_inv
        m_inv_0 = [a, b * cos(gamma), c * cos(beta)]
        m_inv_1 = [0, b * sin(gamma), (c * (cos(alpha) - (cos(beta) * cos(gamma)))) / float(sin(gamma))]
        m_inv_2 = [0, 0, v / float(a * b * sin(gamma))]
        m_inv = [m_inv_0, m_inv_1, m_inv_2]

        return np.matmul(frac_coords, np.transpose(m_inv))

    def get_point_group_coords(self, return_side_chains=True, **kwargs):
        """Find the coordinates of the symmetry mates using the coordinates and the input expansion matrices

        Sets:
            self.number_of_symmetry_mates (int)
            self.model_coords (numpy.ndarray)
        """
        if return_side_chains:  # get different function calls depending on the return type # todo
            # get_pdb_coords = getattr(PDB, 'coords')
            self.coords_type = 'all'
        else:
            # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        coords_length = len(self.coords)
        self.number_of_symmetry_mates = valid_subunit_number[self.symmetry]
        model_coords = np.empty((coords_length * self.number_of_symmetry_mates, 3), dtype=float)
        for idx, rot in enumerate(self.expand_matrices):
            model_coords[idx * coords_length: (idx + 1) * coords_length] = np.matmul(self.coords, np.transpose(rot))
        self.model_coords = Coords(model_coords)

    def get_unit_cell_coords(self, return_side_chains=True, surrounding_uc=True, **kwargs):
        """Generates unit cell coordinates for a symmetry group. Modifies model_coords to include all in a unit cell

        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False returns backbone and CB atoms
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Sets:
            self.number_of_symmetry_mates (int)
            self.model_coords (numpy.ndarray)
        """
        if return_side_chains:  # get different function calls depending on the return type  # todo
            # get_pdb_coords = getattr(PDB, 'coords')
            self.coords_type = 'all'
        else:
            # get_pdb_coords = getattr(PDB, 'get_backbone_and_cb_coords')
            self.coords_type = 'bb_cb'

        if surrounding_uc:
            if self.dimension == 3:
                z_shifts, uc_number = [0., 1., -1.], 27
            elif self.dimension == 2:
                z_shifts, uc_number = [0.], 9
            else:
                return

            uc_frac_coords = self.return_unit_cell_coords(self.coords, fractional=True)
            surrounding_frac_coords = \
                np.concatenate([uc_frac_coords + [x_shift, y_shift, z_shift] for x_shift in [0., 1., -1.]
                                for y_shift in [0., 1., -1.] for z_shift in z_shifts])
            coords = self.frac_to_cart(surrounding_frac_coords)
            self.number_of_symmetry_mates = self.number_of_uc_symmetry_mates * uc_number
        else:
            self.number_of_symmetry_mates = self.number_of_uc_symmetry_mates
            # uc_number = 1
            coords = self.return_unit_cell_coords(self.coords)

        self.model_coords = Coords(coords)

    def return_assembly_symmetry_mates(self, **kwargs) -> List[Structure]:
        """Return symmetry mates as a collection of Structures with symmetric coordinates

        Keyword Args:
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            (list[Structure]): All symmetry mates where Chain names match the ASU
        """
        if len(self.number_of_symmetry_mates) != self.number_of_models:
            self.get_assembly_symmetry_mates(**kwargs)
            if len(self.number_of_symmetry_mates) != self.number_of_models:
                raise \
                    DesignError('%s: The assembly couldn\'t be returned' % self.return_assembly_symmetry_mates.__name__)

        return self.models

    def get_assembly_symmetry_mates(self, surrounding_uc=True, **kwargs):  # -> List[Structure]:
        # , return_side_chains=True):
        """Generate symmetry mates as a collection of Structures with symmetric coordinates

        Keyword Args:
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Sets:
            self.models (list[Structure]): All symmetry mates where each mate has Chain names matching the ASU
        """
        if not self.symmetry:
            # self.log.critical('%s: No symmetry set for %s! Cannot get symmetry mates'  # Todo
            #                   % (self.get_assembly_symmetry_mates.__name__, self.asu.name))
            raise DesignError('%s: No symmetry set for %s! Cannot get symmetry mates'
                              % (self.get_assembly_symmetry_mates.__name__, self.name))
        # if return_side_chains:  # get different function calls depending on the return type
        #     extract_pdb_atoms = getattr(PDB, 'atoms')
        # else:
        #     extract_pdb_atoms = getattr(PDB, 'backbone_and_cb_atoms')

        # prior_idx = self.asu.number_of_atoms  # TODO modify by extract_pdb_atoms
        if self.dimension > 0 and surrounding_uc:
            if self.number_of_symmetry_mates > self.number_of_uc_symmetry_mates:  # ensure surrounding coordinates exist
                number_of_models = self.number_of_symmetry_mates
            else:
                raise ValueError('Cannot return the surrounding unit cells as no coordinates were generated for them.'
                                 'Try passing surrounding_uc=True to .set_symmetry()')
            # else:
            #     number_of_models = self.number_of_uc_symmetry_mates  # set to the uc only
        else:
            number_of_models = self.number_of_symmetry_mates

        number_of_atoms = self.number_of_atoms
        for model_idx in range(number_of_models):
            symmetry_mate_pdb = copy(self.asu)
            symmetry_mate_pdb.replace_coords(self.model_coords[(model_idx * number_of_atoms):
                                                               ((model_idx + 1) * number_of_atoms)])
            self.models.append(symmetry_mate_pdb)

    def find_asu_equivalent_symmetry_model(self):
        """Find the asu equivalent model in the SymmetricModel. Zero-indexed

        Returns:
            (int): The index of the number of models where the ASU can be found
        """
        if self.asu_equivalent_model_idx:  # we already found this information
            self.log.debug('Skipping ASU identification as information already exists')
            return

        template_atom = self.asu.n_terminal_residue
        template_atom_coords, template_atom_index = template_atom.ca_coords, template_atom.ca_atom_index
        asu_size = len(self.coords)
        for model_num in range(self.number_of_symmetry_mates):
            # print(self.model_coords[(model_num * coords_length) + template_atom_index])
            # print(template_atom_coords ==
            #         self.model_coords[(model_num * coords_length) + template_atom_index])
            if np.allclose(template_atom_coords, self.model_coords[(model_num * asu_size) + template_atom_index]):
                # if (template_atom_coords ==
                #         self.model_coords[(model_num * coords_length) + template_atom_index]).all():
                self.asu_equivalent_model_idx = model_num
                return

        self.log.error('%s FAILED to find model' % self.find_asu_equivalent_symmetry_model.__name__)

    def find_intra_oligomeric_equivalent_symmetry_models(self, entity, epsilon=3):  # todo put back to 0.5 after SUCCES
        """From an Entity's Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass

        Args:
            entity (Entity): The Entity with oligomeric chains that should be queried
        Keyword Args:
            epsilon=0.5 (float): The distance measurement tolerance to find similar symmetric models to the oligomer
        """
        if self.oligomeric_equivalent_model_idxs.get(entity):  # we already found this information
            self.log.debug('Skipping oligomeric identification as information already exists')
            return
        asu_size = len(self.coords)
        # need to slice through the specific Entity coords once we have the model
        entity_start, entity_end = entity.atom_indices[0], entity.atom_indices[-1]
        entity_length = entity.number_of_atoms
        entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
        equivalent_models = []
        for chain in entity.chains:
            # chain_length = chain.number_of_atoms
            # chain_center_of_mass = np.matmul(np.full(chain_length, 1 / chain_length), chain.coords)
            chain_center_of_mass = chain.center_of_mass
            # print('Chain', chain_center_of_mass.astype(int))
            for model_num in range(self.number_of_symmetry_mates):
                sym_model_center_of_mass = \
                    np.matmul(entity_center_of_mass_divisor,
                              self.model_coords[(model_num * asu_size) + entity_start:
                                                (model_num * asu_size) + entity_end + 1])
                #                                             have to add 1 for slice ^
                # print('Sym Model', sym_model_center_of_mass)
                # if np.allclose(chain_center_of_mass.astype(int), sym_model_center_of_mass.astype(int)):
                # if np.allclose(chain_center_of_mass, sym_model_center_of_mass):  # using np.rint()
                if np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass) < epsilon:
                    equivalent_models.append(model_num)
                    break

        assert len(equivalent_models) == len(entity.chains), \
            'The number of equivalent models (%d) does not equal the expected number of chains (%d)!'\
            % (len(equivalent_models), len(entity.chains))

        self.oligomeric_equivalent_model_idxs[entity] = equivalent_models

    def return_asu_interaction_models(self, calculate_contacts=False, distance=None, **kwargs):
        """From an asu, find the SymmetricModel equivalent models that immediately surround the asu

        Keyword Args:
            calculate_contacts=False (bool): Whether to calculate interacting models by atomic contacts. If this
                argument is True, the value passed to distance will be the contact distance
            distance=None (float): The distance measurement to find nearby symmetric models to the asu.
                If no distance is provided, will use 2x the calculated asu radius (essentially the diameter).
                If calculate_contacts=True, distance is 8.0 by default
        Returns:
            (list): The indices of the models that contact the asu
        """
        if not distance:
            if calculate_contacts:
                distance = 8.  # default contact distance
            else:
                # distance = self.asu.radius * 2  # value too large self.pdb.radius * 2
                distance = self.asu.radius  # Todo adjust this value still!!!

        if calculate_contacts:
            self.generate_assembly_tree()
            # Need to select only coords that are BB or CB from the model coords
            asu_indices = self.asu.backbone_and_cb_indices if self.coords_type != 'bb_cb' else None
            asu_query = self.assembly_tree.query_radius(self.coords[asu_indices], distance)
            # coords_length = len(asu_indices)
            # contacting_model_indices = [assembly_idx // coords_length
            #                             for asu_idx, assembly_contacts in enumerate(asu_query)
            #                             for assembly_idx in assembly_contacts]
            # interacting_models = sorted(set(contacting_model_indices))
            # interacting_models = np.unique(asu_query // coords_length).tolist()
            interacting_models = np.unique(asu_query // len(asu_query)).tolist()
        else:
            center_of_mass = self.center_of_mass
            interacting_models = [model_idx for model_idx, sym_model_com in enumerate(self.center_of_mass_symmetric_models)
                                  if np.linalg.norm(center_of_mass - sym_model_com) < distance]
            # print('interacting_models com', self.center_of_mass_symmetric_models[interacting_models])

        return interacting_models

    def return_asu_equivalent_symmetry_mate_indices(self):
        """Find the coordinate indices of the asu equivalent model in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which by default is True
        Returns:
            (list): The indices in the SymmetricModel where the ASU is also located
        """
        self.find_asu_equivalent_symmetry_model()
        start_idx = self.number_of_atoms * self.asu_equivalent_model_idx
        end_idx = self.number_of_atoms * (self.asu_equivalent_model_idx + 1)

        return list(range(start_idx, end_idx))

    def return_intra_oligomeric_symmetry_mate_indices(self, entity):
        """Find the coordinate indices of the intra-oligomeric equivalent models in the SymmetricModel. Zero-indexed

        self.model_coords must be from all atoms which is True by default
        Args:
            entity (Entity): The Entity with oligomeric chains to query for corresponding symmetry mates
        Returns:
            (list): The indices in the SymmetricModel where the intra-oligomeric contacts are located
        """
        self.find_intra_oligomeric_equivalent_symmetry_models(entity)
        oligomeric_indices = []
        for model_number in self.oligomeric_equivalent_model_idxs.get(entity):
            start_idx = self.number_of_atoms * model_number
            end_idx = self.number_of_atoms * (model_number + 1)
            oligomeric_indices.extend(list(range(start_idx, end_idx)))

        return oligomeric_indices

    def find_asu_interaction_indices(self, **kwargs):
        """Find the coordinate indices for the models in the SymmetricModel interacting with the asu. Zero-indexed

        self.model_coords must be from all atoms which is True by default
        Keyword Args:
            calculate_contacts=False (bool): Whether to calculate interacting models by atomic contacts. If this
                argument is True, the value passed to distance will be the contact distance
            distance=None (float): The distance measurement to find nearby symmetric models to the asu.
                If no distance is provided, will use 2x the max asu diameter.
                If calculate_contacts=True, distance is 8.0 by default
        Returns:
            (list): The indices in the SymmetricModel where the asu contacts other models
        """
        model_numbers = self.return_asu_interaction_models(**kwargs)
        interacting_indices = []
        for model_number in model_numbers:
            start_idx = self.number_of_atoms * model_number
            end_idx = self.number_of_atoms * (model_number + 1)
            interacting_indices.extend(list(range(start_idx, end_idx)))

        return interacting_indices

    def return_symmetry_mates(self, structure, **kwargs):
        """Expand the asu using self.symmetry for the symmetry specification, and optional unit cell dimensions if
        self.dimension > 0. Expands assembly to complete point group, or the unit cell

        Args:
            structure (Structure): A Structure containing some collection of Residues
        Keyword Args:
            return_side_chains=True (bool): Whether to return all side chain atoms. False gives backbone and CB atoms
            surrounding_uc=True (bool): Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            (list[Structure]): The symmetric copies of the input structure
        """
        if self.dimension == 0:
            return self.return_point_group_symmetry_mates(structure)
        else:
            return self.return_crystal_symmetry_mates(structure, **kwargs)

    def return_point_group_symmetry_mates(self, structure):
        """Expand the coordinates for every symmetric copy within the point group assembly

        Args:
            structure (Structure): A Structure containing some collection of Residues
        Returns:
            (list[Structure]): The symmetric copies of the input structure
        """
        return [structure.return_transformed_copy(rotation=rot) for rot in self.expand_matrices]

    def return_crystal_symmetry_mates(self, structure, surrounding_uc=True, **kwargs):
        """Expand the coordinates for every symmetric copy within the unit cell surrounding a asu

        Args:
            structure (Structure): A Structure containing some collection of Residues
        Returns:
            (list[Structure]): The symmetric copies of the input structure
        """
        if surrounding_uc:
            return self.return_surrounding_unit_cell_symmetry_mates(structure, **kwargs)  # return_side_chains
        else:
            return self.return_unit_cell_symmetry_mates(structure, **kwargs)  # return_side_chains

    def return_symmetric_coords(self, coords):
        """Return the coordinates from a set of coordinates for the specified SymmetricModel

        Args:
            coords (Union[numpy.ndarray, list]): The coordinates to symmetrize
        Returns:
            (numpy.ndarray): The symmetrized coordinates
        """
        if self.dimension == 0:
            coords_length = 1 if not isinstance(coords[0], (list, np.ndarray)) else len(coords)
            model_coords = np.empty((coords_length * self.number_of_symmetry_mates, 3), dtype=float)
            for idx, rot in enumerate(self.expand_matrices):
                rot_coords = np.matmul(coords, np.transpose(rot))
                model_coords[idx * coords_length: (idx + 1) * coords_length] = rot_coords

            return model_coords
        else:
            return self.return_unit_cell_coords(coords)

    def return_unit_cell_coords(self, coords, fractional=False):  # Todo surrounding_uc=True
        """Return the unit cell coordinates from a set of coordinates for the specified SymmetricModel

        Args:
            coords (numpy.ndarray): The cartesian coordinates to expand to the unit cell
        Keyword Args:
            fractional=False (bool): Whether to return coordinates in fractional or cartesian (False) unit cell frame
        Returns:
            (numpy.ndarray): All unit cell coordinates
        """
        asu_frac_coords = self.cart_to_frac(coords)
        coords_length = 1 if not isinstance(coords[0], (list, np.ndarray)) else len(coords)
        model_coords = np.empty((coords_length * self.number_of_uc_symmetry_mates, 3), dtype=float)
        # model_coords = np.empty((coords_length * self.number_of_symmetry_mates, 3), dtype=float)
        # Todo pickled operators don't have identity matrix (currently), so we add the asu
        model_coords[:coords_length] = asu_frac_coords
        for idx, (rot, tx) in enumerate(self.expand_matrices, 1):  # since no identity, start idx at 1
            rt_asu_frac_coords = np.matmul(asu_frac_coords, np.transpose(rot)) + tx
            model_coords[idx * coords_length: (idx + 1) * coords_length] = rt_asu_frac_coords

        if fractional:
            return model_coords
        else:
            return self.frac_to_cart(model_coords)

    def return_unit_cell_symmetry_mates(self, structure, return_side_chains=True):  # For returning PDB copies
        """Expand the coordinates for the structure to every symmetry mate in the unit cell based on symmetry matrices

        Args:
            structure (Structure): A Structure containing some collection of Residues
        Keyword Args:
            return_side_chains=True (bool): Whether the return the side chain coordinates in addition to backbone
        Returns:
            (list[Structure]): The symmetric copies of the input structure
        """
        # Caution, this function will return poor if the number of atoms in the structure is 1!
        if return_side_chains:  # get different function calls depending on the return type
            # extract_pdb_atoms = getattr(PDB, 'atoms')  # Not using. The copy() versus PDB() changes residue objs
            coords = structure.coords
        else:
            # extract_pdb_atoms = getattr(PDB, 'backbone_and_cb_atoms')
            coords = structure.get_backbone_and_cb_coords()

        sym_cart_coords = self.return_unit_cell_coords(coords)
        coords_length = sym_cart_coords.shape[0]
        sym_mates = []
        for model_num in range(self.number_of_symmetry_mates):
            symmetry_mate_pdb = copy(structure)
            symmetry_mate_pdb.replace_coords(sym_cart_coords[model_num * coords_length:
                                                             (model_num + 1) * coords_length])
            sym_mates.append(symmetry_mate_pdb)

        return sym_mates

    def return_surrounding_unit_cell_symmetry_mates(self, structure, return_side_chains=True, **kwargs):
        """Expand the coordinates for the structure to every surrounding unit cell based on symmetry matrices

        Args:
            structure (Structure): A Structure containing some collection of Residues
        Keyword Args:
            return_side_chains=True (bool): Whether the return the side chain coordinates in addition to backbone
        Returns:
            (list[Structure]): The symmetric copies of the input structure
        """
        if return_side_chains:  # get different function calls depending on the return type
            # extract_pdb_atoms = getattr(PDB, 'atoms')  # Not using. The copy() versus PDB() changes residue objs
            coords = structure.coords
        else:
            # extract_pdb_atoms = getattr(PDB, 'backbone_and_cb_atoms')
            coords = structure.get_backbone_and_cb_coords()

        if self.dimension == 3:
            z_shifts, uc_number = [0, 1, -1], 9
        elif self.dimension == 2:
            z_shifts, uc_number = [0], 27
        else:
            return

        # pdb_coords = extract_pdb_atoms
        uc_frac_coords = self.return_unit_cell_coords(coords, fractional=True)
        surrounding_frac_coords = [uc_frac_coords + [x_shift, y_shift, z_shift] for x_shift in [0, 1, -1]
                                   for y_shift in [0, 1, -1] for z_shift in z_shifts]

        surrounding_cart_coords = self.frac_to_cart(surrounding_frac_coords)

        coords_length = len(uc_frac_coords)
        sym_mates = []
        for coord_set in surrounding_cart_coords:
            for model in range(self.number_of_uc_symmetry_mates):
                symmetry_mate_pdb = copy(structure)
                symmetry_mate_pdb.replace_coords(coord_set[(model * coords_length): ((model + 1) * coords_length)])
                sym_mates.append(symmetry_mate_pdb)

        number_of_symmetry_mates = uc_number * self.number_of_uc_symmetry_mates
        assert len(sym_mates) == number_of_symmetry_mates, \
            'Number of models (%d) is incorrect! Should be %d' % (len(sym_mates), number_of_symmetry_mates)
        return sym_mates

    def assign_entities_to_sub_symmetry(self):
        """From a symmetry entry, find the entities which belong to each sub-symmetry (the component groups) which make
        the global symmetry. Construct the sub-symmetry by copying each symmetric chain to the Entity's .chains
        attribute
        """
        if not self.symmetry:
            raise DesignError('Must set a global symmetry to assign entities to sub symmetry!')

        # Get the rotation matrices for each group then orient along the setting matrix "axis"
        if self.sym_entry.group1 in ['D2', 'D3', 'D4', 'D6'] or self.sym_entry.group2 in ['D2', 'D3', 'D4', 'D6']:
            group1 = self.sym_entry.group1.replace('D', 'C')
            group2 = self.sym_entry.group2.replace('D', 'C')
            rotation_matrices_only1 = get_rot_matrices(rotation_range[group1], 'z', 360)
            rotation_matrices_only2 = get_rot_matrices(rotation_range[group2], 'z', 360)
            # provide a 180 degree rotation along x (all D orient symmetries have axis here)
            # apparently passing the degeneracy matrix first without any specification towards the row/column major
            # worked for Josh. I am not sure that I understand his degeneracy (rotation) matrices orientation enough to
            # understand if he hardcoded the column "majorness" into situations with rot and degen np.matmul(rot, degen)
            rotation_matrices_group1 = get_degen_rotmatrices(flip_x_matrix, rotation_matrices_only1)
            rotation_matrices_group2 = get_degen_rotmatrices(flip_x_matrix, rotation_matrices_only2)
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
        origin = np.array([0., 0., 0.])
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
                expand_matrices = get_ptgrp_sym_op(self.point_group_symmetry)
            else:
                expand_matrices = self.expand_matrices
            ext_tx = self.center_of_mass_symmetric  # only works for unit cell or point group NOT surrounding UC
            # This is typically centered at the origin for the symmetric assembly... NEED rigourous testing.
            # Maybe this route of generation is too flawed for layer/space? Nanohedra framework gives a comprehensive
            # handle on all these issues though

        # Solution as of 4/8/22 v
        # get optimal external translation
        if self.dimension > 0:
            optimal_external_shifts = self.sym_entry.get_optimal_shift_from_uc_dimensions(*self.uc_dimensions)
            external_tx1 = optimal_external_shifts[:, None] * self.sym_entry.external_dof1
            external_tx2 = optimal_external_shifts[:, None] * self.sym_entry.external_dof2
            external_tx = [external_tx1, external_tx2]
        else:
            # external_tx1, external_tx2 = None, None
            external_tx = [None for _ in len(self.sym_entry.groups)]

        inv_setting_matrices = {key: np.linalg.inv(setting_matrix) for key, setting_matrix in setting_matrices.items()}
        center_of_mass_symmetric_entities = self.center_of_mass_symmetric_entities
        transform_solutions = []
        for group_idx, sym_group in enumerate(self.sym_entry.groups):
            # search through the sub_symmetry group setting matrices that make up the resulting point group symmetry
            # apply setting matrix to the entity centers of mass indexed to the proper group number
            internal_tx = None
            setting_matrix = None
            current_best_minimal_central_offset = float('inf')
            # sym_group_setting_matrices = point_group_setting_matrix_members[self.point_group_symmetry].get(sym_group)
            for setting_matrix_idx in point_group_setting_matrix_members[self.point_group_symmetry].get(sym_group, []):
                temp_model_coms = np.matmul(center_of_mass_symmetric_entities[group_idx],
                                            np.transpose(inv_setting_matrices[setting_matrix_idx]))
                # find groups of COMs with equal z heights
                possible_height_groups = {}
                for idx, com in enumerate(temp_model_coms.round(decimals=2)):  # Todo maybe 1 is close enough
                    if com in possible_height_groups:
                        possible_height_groups[com].add(idx)
                    else:
                        possible_height_groups[com] = {idx}
                # find the most centrally disposed, COM grouping with the correct number of COMs in the group
                # not necessarily positive...
                centrally_disposed_group = None
                minimal_central_offset = float('inf')
                for height, indices in possible_height_groups.items():
                    # if height < 0:  # this may be detrimental. Increased search cost not worth missing solution
                    #     continue
                    if len(indices) == valid_subunit_number[sym_group]:
                        # central_offset = pdist(temp_model_coms[indices]).sum()
                        x = (temp_model_coms[indices] - height)[0]  # get first point as they are symmetric/equivalent
                        central_offset = np.sqrt(x.dot(x))
                        if central_offset < minimal_central_offset:
                            minimal_central_offset = central_offset
                            centrally_disposed_group = height

                # if a viable group was found save the group COM as an internal_tx and setting_matrix used to find it
                if centrally_disposed_group:
                    if setting_matrix and internal_tx:  # we have competing solutions, this is probably a degeneracy?
                        # choose the one that specifies the COM of individual Entities the most central
                        self.log.warning('There are multiple pose transformation solutions for the symmetry group '
                                         '%s (specified in position {%d} of %s). The solution with the minial '
                                         'central offset from the Entity center of mass to the specified symmetry '
                                         'axis will be chosen. This may result in inaccurate behavior'
                                         % (sym_group, group_idx + 1, self.sym_entry.combination_string))
                        if minimal_central_offset < current_best_minimal_central_offset:
                            setting_matrix = setting_matrices[setting_matrix_idx]
                            internal_tx = temp_model_coms[possible_height_groups[centrally_disposed_group]].mean()
                        else:
                            # The central offset of this setting matrix is larger than the other. This logic may be
                            # wrong but the heuristic indicates the primary axis of this group is not optimal
                            pass
                    else:  # these were not set yet
                        setting_matrix = setting_matrices[setting_matrix_idx]
                        internal_tx = temp_model_coms[possible_height_groups[centrally_disposed_group]].mean()
                        current_best_minimal_central_offset = minimal_central_offset
                else:  # no viable group probably because the setting matrix was wrong. Continue with next
                    pass

            if setting_matrix and internal_tx:
                transform_solutions.append(dict(rotation2=setting_matrix, translation2=external_tx[group_idx],
                                                translation=internal_tx))
            else:
                raise ValueError('Using the supplied Model (%s) and the specified symmetry (%s), there was no solution '
                                 'found for Entity #%d. A possible issue could be that the supplied Model has it\'s '
                                 'Entities out of order for the assumed symmetric entry "%s". If the order is different'
                                 ' please supply the correct order with the symmetry combination format "%s" to the '
                                 'flag --%s'
                                 % (self.name, self.symmetry, group_idx + 1, self.sym_entry.combination_string,
                                    symmetry_combination_format, PUtils.sym_entry))

        # find the particular rotation to orient the Entity oligomer to a cannonical orientation
        # Solution as of 4/8/22 ^

        # # find the approximate scalar translation of the asu center of mass from the reference symmetry origin
        # approx_entity_com_reference_tx = np.linalg.norm(all_entities_com - ext_tx)
        # approx_entity_z_tx = [0., 0., approx_entity_com_reference_tx]
        # # apply the setting matrix for each group to the approximate translation
        # set_mat1 = self.sym_entry.setting_matrix1
        # set_mat2 = self.sym_entry.setting_matrix2
        # # TODO test transform_coordinate_sets has the correct input format (numpy.ndarray)
        # com_group1 = \
        #     transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat1, translation2=ext_tx)
        # com_group2 = \
        #     transform_coordinate_sets(origin, translation=approx_entity_z_tx, rotation2=set_mat2, translation2=ext_tx)
        # # expand the tx'd, setting matrix rot'd, approximate coms for each group using self.expansion operators
        # coms_group1 = self.return_symmetric_coords(com_group1)
        # coms_group2 = self.return_symmetric_coords(com_group2)
        #
        # # measure the closest distance from each entity com to the setting matrix transformed approx group coms to find
        # # which group the entity belongs to. Save the group and the operation index of the expansion matrices. With both
        # # of these, it is possible to find a new setting matrix that is symmetry equivalent and will generate the
        # # correct sub-symmetry symmetric copies for each provided Entity
        # group_entity_rot_ops = {1: {}, 2: {}}
        # # min_dist1, min_dist2, min_1_entity, min_2_entity = float('inf'), float('inf'), None, None
        # for entity in self.asu.entities:
        #     entity_com = entity.center_of_mass
        #     min_dist, min_entity_group_operator = float('inf'), None
        #     for idx in range(len(expand_matrices)):  # has the length of the symmetry operations
        #         com1_distance = np.linalg.norm(entity_com - coms_group1[idx])
        #         com2_distance = np.linalg.norm(entity_com - coms_group2[idx])
        #         if com1_distance < com2_distance:
        #             if com1_distance < min_dist:
        #                 min_dist = com1_distance
        #                 min_entity_group_operator = (group2, expand_matrices[idx])
        #             # # entity_min_group = 1
        #             # entity_group_d[1].append(entity)
        #         else:
        #             if com2_distance < min_dist:
        #                 min_dist = com2_distance
        #                 min_entity_group_operator = (group2, expand_matrices[idx])
        #             # # entity_min_group = 2
        #             # entity_group_d[2].append(entity)
        #     if min_entity_group_operator:
        #         group, operation = min_entity_group_operator
        #         group_entity_rot_ops[group][entity] = operation
        #         # {1: {entity1: [[],[],[]]}, 2: {entity2: [[],[],[]]}}
        #
        # set_mat = {1: set_mat1, 2: set_mat2}
        # inv_set_matrix = {1: np.linalg.inv(set_mat1), 2: np.linalg.inv(set_mat2)}
        # group_rotation_matrices = {1: rotation_matrices_group1, 2: rotation_matrices_group2}
        # # Multiplication is not possible in this way apparently!
        # # group_set_rotation_matrices = {1: np.matmul(rotation_matrices_group1, np.transpose(set_mat1)),
        # #                                2: np.matmul(rotation_matrices_group2, np.transpose(set_mat2))}
        #
        # # Apply the rotation matrices to the identified group Entities. First modify the Entity by the inverse expansion
        # # and setting matrices to orient along Z axis. Apply the rotation matrix, then reverse operations back to start
        # for idx, (group, entity_ops) in enumerate(group_entity_rot_ops.items()):
        #     for entity, rot_op in entity_ops.items():
        #         dummy_rotation = False
        #         dummy_translation = False
        #         # Todo need to reverse the expansion matrix first to get the entity coords to the "canonical" setting
        #         #  matrix as expected by Nanohedra. I can then make_oligomers
        #         entity.make_oligomer(symmetry=group, **dict(rotation=dummy_rotation, translation=dummy_translation,
        #                                                     rotation2=set_mat[idx], translation2=ext_tx))
        #         # # Todo if this is a fractional rot/tx pair this won't work
        #         # #  I converted the space group external tx and design_pg_symmetry to rot_matrices so I should
        #         # #  test if the change to local point group symmetry in a layer or space group is sufficient
        #         # inv_expand_matrix = np.linalg.inv(rot_op)
        #         # inv_rotation_matrix = np.linalg.inv(dummy_rotation)
        #         # # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
        #         # # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
        #         # centered_coords = transform_coordinate_sets(entity.coords, translation=-ext_tx)
        #         # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_expand_matrix,
        #         #                                             rotation2=inv_set_matrix[group])
        #         # TODO                                        NEED DIHEDRAl rotation v back to canonical
        #         # sym_on_z_coords = transform_coordinate_sets(centered_coords, rotation=inv_rotation_matrix,
        #         # TODO                                        as well as v translation (not approx, dihedral won't work)
        #         #                                             translation=approx_entity_z_tx)
        #         # # now rotate, then undo symmetry expansion matrices
        #         # # for rot in group_rotation_matrices[group][1:]:  # exclude the first rotation matrix as it is identity
        #         # for rot in group_rotation_matrices[group]:
        #         #     temp_coords = transform_coordinate_sets(sym_on_z_coords, rotation=np.array(rot), rotation2=set_mat[group])
        #         #     # rot_centered_coords = transform_coordinate_sets(sym_on_z_coords, rotation=rot)
        #         #     # final_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation,
        #         #     #                                          translation=translation, <-NEED^ for DIHEDRAL
        #         #     #                                          rotation2=rotation2, translation2=translation2)
        #         #     final_coords = transform_coordinate_sets(temp_coords, rotation=rot_op, translation=ext_tx)
        #         #     # Entity representative stays in the .chains attribute as chain[0] given the iterator slice above
        #         #     sub_symmetry_mate_pdb = copy(entity.chain_representative)
        #         #     sub_symmetry_mate_pdb.replace_coords(final_coords)
        #         #     entity.chains.append(sub_symmetry_mate_pdb)
        #         #     # need to take the cyclic system generated and somehow transpose it on the dihedral group.
        #         #     # an easier way would be to grab the assembly from the SymDesignOutput/Data/PDBs and set the
        #         #     # oligomer onto the ASU. The .chains would then be populated for the non-transposed chains
        #         #     # if dihedral:  # TODO
        #         #     #     dummy = True

    def symmetric_assembly_is_clash(self, distance=2.1):  # Todo design_selector
        """Returns True if the SymmetricModel presents any clashes. Checks only backbone and CB atoms

        Keyword Args:
            distance=2.1 (float): The cutoff distance for the coordinate overlap

        Returns:
            (bool): True if the symmetric assembly clashes with the asu, False otherwise
        """
        if not self.symmetry:
            raise DesignError('Cannot check if the assembly is clashing as it has no symmetry!')
        elif self.number_of_symmetry_mates == 1:
            raise DesignError('Cannot check if the assembly is clashing without first calling %s'
                              % self.generate_symmetric_assembly.__name__)

        if self.coords_type != 'bb_cb':
            # Need to select only coords that are BB or CB from the model coords
            asu_indices = self.asu.backbone_and_cb_indices
        else:
            asu_indices = None

        self.generate_assembly_tree()
        # clashes = asu_coord_tree.two_point_correlation(self.model_coords[model_indices_without_asu], [distance])
        clashes = self.assembly_tree.two_point_correlation(self.coords[asu_indices], [distance])
        if clashes[0] > 0:
            self.log.warning('%s: Found %d clashing sites! Pose is not a viable symmetric assembly'
                             % (self.name, clashes[0]))
            return True  # clash
        else:
            return False  # no clash

    def generate_assembly_tree(self):
        """Create a tree structure from all the coordinates in the symmetric assembly

        Sets:
            self.assembly_tree (sklearn.neighbors._ball_tree.BallTree): The constructed coordinate tree
        """
        if self.assembly_tree:
            return

        model_asu_indices = self.return_asu_equivalent_symmetry_mate_indices()
        if self.coords_type != 'bb_cb':
            # Need to select only coords that are BB or CB from the model coords
            number_asu_atoms = self.number_of_atoms
            asu_indices = self.asu.backbone_and_cb_indices
            # We have all the BB/CB indices from ASU, must multiply this int's in self.number_of_symmetry_mates
            # to get every BB/CB coord in the model
            # Finally we take out those indices that are inclusive of the model_asu_indices like below
            model_indices_filter = np.array([idx + (model_number * number_asu_atoms)
                                             for model_number in range(self.number_of_symmetry_mates)
                                             for idx in asu_indices])
        else:  # we will grab every coord in the model
            model_indices_filter = np.array(list(range(len(self.model_coords))))
            asu_indices = None

        # make a boolean mask where the model indices of interest are True
        without_asu_mask = np.logical_or(model_indices_filter < model_asu_indices[0],
                                         model_indices_filter > model_asu_indices[-1])
        # take the boolean mask and filter the model indices mask to leave only symmetry mate bb/cb indices, NOT asu
        model_indices_without_asu = model_indices_filter[without_asu_mask]
        selected_assembly_coords = len(model_indices_without_asu) + len(asu_indices)
        all_assembly_coords_length = len(asu_indices) * self.number_of_symmetry_mates
        assert selected_assembly_coords == all_assembly_coords_length, \
            '%s: Ran into an issue indexing' % self.symmetric_assembly_is_clash.__name__

        # asu_coord_tree = BallTree(self.coords[asu_indices])
        # return BallTree(self.model_coords[model_indices_without_asu])
        self.assembly_tree = BallTree(self.model_coords[model_indices_without_asu])

    def format_biomt(self, **kwargs):
        """Return the expand_matrices as a BIOMT record

        Returns:
            (str)
        """
        if self.dimension == 0:
            return '%s\n' \
                   % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}'.format(v_idx, m_idx, *vec, 0.)
                               for m_idx, rot in enumerate(self.expand_matrices, 1) for v_idx, vec in enumerate(rot, 1))
        # for np expand_matrices for m_idx, matrix in enumerate(self.expand_matrices.tolist(), 1) for v_idx, vec in enumerate(matrix, 1))
        else:  # TODO change this so that the oligomeric units are populated?
            return ''

    # def write(self, out_path=os.getcwd(), header=None, increment_chains=False):  # , cryst1=None):  # Todo write symmetry, name, location
    #     """Write Structure Atoms to a file specified by out_path or with a passed file_handle. Return the filename if
    #     one was written"""

    # @staticmethod
    # def get_ptgrp_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, 'POINT_GROUP_SYMM_OPERATORS')):
    #     """Get the symmetry operations for a specified point group oriented in the canonical orientation
    #     Returns:
    #         (list[list])
    #     """
    #     expand_matrix_filepath = os.path.join(expand_matrix_dir, '%s.txt' % sym_type)
    #     with open(expand_matrix_filepath, "r") as expand_matrix_f:
    #         # Todo pickle these to match SDUtils
    #         line_count = 0
    #         expand_matrices = []
    #         mat = []
    #         for line in expand_matrix_f.readlines():
    #             line = line.split()
    #             if len(line) == 3:
    #                 line_float = [float(s) for s in line]
    #                 mat.append(line_float)
    #                 line_count += 1
    #                 if line_count % 3 == 0:
    #                     expand_matrices.append(mat)
    #                     mat = []
    #
    #         return expand_matrices

    @staticmethod  # Todo clean this pickle to match SDUtils
    def get_sg_sym_op(sym_type, expand_matrix_dir=os.path.join(sym_op_location, 'SPACE_GROUP_SYMM_OPERATORS')):
        sg_op_filepath = os.path.join(expand_matrix_dir, '%s.pickle' % sym_type)
        with open(sg_op_filepath, 'rb') as sg_op_file:
            sg_sym_op = load(sg_op_file)

        return sg_sym_op


class Pose(SymmetricModel, SequenceProfile):  # Model
    """A Pose is made of single or multiple PDB objects such as Entities, Chains, or other tsructures.
    All objects share a common feature such as the same symmetric system or the same general atom configuration in
    separate models across the Structure or sequence.
    """
    def __init__(self, asu_file=None, pdb_file=None, **kwargs):  # asu=None, pdb=None,
        # self.pdbs_d = {}
        self.fragment_pairs = []
        self.fragment_metrics = {}
        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.required_indices = set()
        self.required_residues = None
        self.interface_residues = {}  # {(entity1, entity2): ([entity1_residues], [entity2_residues]), ...}
        self.source_db = kwargs.get('source_db', None)
        self.split_interface_residues = {}  # {1: [(Residue obj, Entity obj), ...], 2: [(Residue obj, Entity obj), ...]}
        self.split_interface_ss_elements = {}  # {1: [0,1,2] , 2: [9,13,19]]}
        self.ss_index_array = []  # stores secondary structure elements by incrementing index
        self.ss_type_array = []  # stores secondary structure type ('H', 'S', ...)
        # self.handle_flags(**kwargs)
        # self.ignore_clashes = False
        self.ignore_clashes = kwargs.get('ignore_clashes', False)
        # self.design_selector = {}
        # if kwargs.get('design_selector'):
        self.design_selector = kwargs.get('design_selector', {})
        # else:
        #     self.design_selector = {}
        # if asu and isinstance(asu, Structure):
        #     self.asu = asu
        if asu_file:  # TODO COMMENT OUT .asu initialize Pose from SymmetricModel?
            self.asu = PDB.from_file(asu_file, log=self.log)  # **kwargs todo create kwarg collector to protect Object
        # elif pdb and isinstance(pdb, Structure):
        #     self.pdb = pdb
        elif pdb_file:  # TODO COMMENT OUT .pdb initialize Pose from Model?
            self.pdb = PDB.from_file(pdb_file, log=self.log)  # **kwargs

        super().__init__(**kwargs)  # will only generate an assembly if an ASU is present
        # super().__init__(**kwargs)
        # self.set_symmetry(**symmetry_kwargs)

        frag_db = kwargs.get('frag_db')
        if frag_db:  # Attach existing FragmentDB to the Pose
            self.attach_fragment_database(db=frag_db)
            # self.frag_db = frag_db  # Todo property
            for entity in self.entities:
                entity.attach_fragment_database(db=frag_db)

        self.euler_lookup = kwargs.get('euler_lookup', None)
        # for entity in self.entities:  # No need to attach to entities
        #     entity.euler_lookup = euler_lookup

    @classmethod
    def from_pdb(cls, pdb, **kwargs):
        return cls(pdb=pdb, **kwargs)

    @classmethod
    def from_pdb_file(cls, pdb_file, **kwargs):
        return cls(pdb_file=pdb_file, **kwargs)

    @classmethod
    def from_asu(cls, asu, **kwargs):
        return cls(asu=asu, **kwargs)

    @classmethod
    def from_asu_file(cls, asu_file, **kwargs):
        return cls(asu_file=asu_file, **kwargs)

    # @property
    # def asu(self):
    #     return self._asu
    #
    # @asu.setter
    # def asu(self, asu):
    #     self._asu = asu
    #     self.pdb = asu

    # @property
    # def pdb(self):
    #     return self._pdb

    @Model.pdb.setter
    def pdb(self, pdb):
        # self.log.debug('Adding PDB \'%s\' to Pose' % pdb.name)
        # super(Model, self).pdb = pdb
        self._pdb = pdb
        self.coords = pdb._coords
        if not self.ignore_clashes:  # TODO add this check to SymmetricModel initialization
            if pdb.is_clash():
                raise DesignError('%s contains Backbone clashes and is not being considered further!' % self.name)
        # self.pdbs_d[pdb.name] = pdb
        self.create_design_selector()  # **self.design_selector) TODO rework this whole mechanism

    # @property
    # def name(self):
    #     try:
    #         return self._name
    #     except AttributeError:
    #         return self.pdb.name
    #
    # @name.setter
    # def name(self, name):
    #     self._name = name

    @property
    def active_entities(self):
        try:
            return self._active_entities
        except AttributeError:
            self._active_entities = [entity for entity in self.entities if entity in self.design_selector_entities]
            return self._active_entities

    # @property
    # def entities(self):  # TODO COMMENT OUT .pdb
    #     return self.pdb.entities
    #
    # @property
    # def chains(self):
    #     return [chain for entity in self.entities for chain in entity.chains]
    #
    # @property
    # def active_chains(self):
    #     return [chain for entity in self.active_entities for chain in entity.chains]
    #
    # @property
    # def chain_breaks(self):
    #     return [entity.c_terminal_residue.number for entity in self.entities]
    #
    # @property
    # def residues(self):  # TODO COMMENT OUT .pdb
    #     return self.pdb.residues
    #
    # @property
    # def reference_sequence(self) -> str:
    #     # return ''.join(self.pdb.reference_sequence.values())
    #     return ''.join(entity.reference_sequence for entity in self.entities)
    #
    # def entity(self, entity):  # TODO COMMENT OUT .pdb
    #     return self.pdb.entity(entity)
    #
    # def chain(self, chain):  # TODO COMMENT OUT .pdb
    #     return self.pdb.entity_from_chain(chain)

    # def find_center_of_mass(self):
    #     """Retrieve the center of mass for the specified Structure"""
    #     if self.symmetry:
    #         divisor = 1 / len(self.model_coords)
    #         self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.model_coords)
    #     else:
    #         divisor = 1 / len(self.coords)  # must use coords as can have reduced view of the full coords, i.e. BB & CB
    #         self._center_of_mass = np.matmul(np.full(self.number_of_atoms, divisor), self.coords)

    # def add_pdb(self, pdb):  # UNUSED
    #     """Add a PDB to the Pose PDB as well as the member PDB container"""
    #     # self.debug_pdb(tag='add_pdb')  # here, pdb chains are still in the oriented configuration
    #     self.pdbs_d[pdb.name] = pdb
    #     self.add_entities_to_pose(pdb)
    #
    # def add_entities_to_pose(self, pdb):  # UNUSED
    #     """Add each unique Entity in a PDB to the Pose PDB. Multiple PDB's become one PDB representative"""
    #     current_pdb_entities = self.entities
    #     for idx, entity in enumerate(pdb.entities):
    #         current_pdb_entities.append(entity)
    #     self.log.debug('Adding the entities \'%s\' to the Pose'
    #                    % ', '.join(list(entity.name for entity in current_pdb_entities)))
    #
    #     self.pdb = PDB.from_entities(current_pdb_entities, metadata=self.pdb, log=self.log)

    def get_contacting_asu(self, distance=8, **kwargs):
        """From the Pose PDB and the associated active Entities, find the maximal contacting ASU for each of the
        entities

        Keyword Args:
            distance=8 (int): The distance to check for contacts
        Returns:
            (PDB): The PDB object with the minimal set of Entities containing the maximally touching configuration
        """
        # self.debug_pdb(tag='get_contacting')
        if len(self.active_entities) == 1:
            entities = self.active_entities
        else:
            idx = 0
            chain_combinations, entity_combinations = [], []
            contact_count = \
                np.zeros(sum(map(prod, combinations((entity.number_of_monomers for entity in self.active_entities), 2))))
            # contact_count = np.zeros((prod([len(entity.chains) for entity in self.active_entities])))
            for entity1, entity2 in combinations(self.active_entities, 2):
                for chain1 in entity1.chains:
                    chain_cb_coord_tree = BallTree(chain1.get_cb_coords())
                    for chain2 in entity2.chains:
                        chain_combinations.append((chain1, chain2))
                        entity_combinations.append((entity1, entity2))
                        contact_count[idx] = chain_cb_coord_tree.two_point_correlation(chain2.get_cb_coords(),
                                                                                       [distance])[0]
                        idx += 1
            max_contact_idx = contact_count.argmax()
            # second_contact_idx = contact_count.argsort()[-2]
            additional_chains = []
            max_chains = list(chain_combinations[max_contact_idx])
            if len(max_chains) != len(self.active_entities):
                # find the indices where either of the maximally contacting chains are utilized
                selected_chain_indices = [idx for idx, chain_pair in enumerate(chain_combinations)
                                          if max_chains[0] in chain_pair or max_chains[1] in chain_pair]
                remaining_entities = set(self.active_entities).difference(entity_combinations[max_contact_idx])
                for entity in remaining_entities:  # get the maximum contacts and the associated entity and chain indices
                    # find the indices where the missing entity is utilized
                    remaining_indices = [idx for idx, entity_pair in enumerate(entity_combinations)
                                         if entity in entity_pair]
                    # pair_position = [0 if entity_pair[0] == entity else 1
                    #                  for idx, entity_pair in enumerate(entity_combinations) if entity in entity_pair]
                    viable_remaining_indices = list(set(remaining_indices).intersection(selected_chain_indices))
                    # out of the viable indices where the selected chains are matched with the missing entity,
                    # find the highest contact
                    max_index = contact_count[viable_remaining_indices].argmax()
                    for entity_idx, entity_in_combo in enumerate(entity_combinations[viable_remaining_indices[max_index]]):
                        if entity == entity_in_combo:
                            additional_chains.append(chain_combinations[viable_remaining_indices[max_index]][entity_idx])

            entities = max_chains + additional_chains

        # chain_2 = PDB.from_entities(list(chain_combinations[second_contact_idx]), name='asu', log=self.log, pose_format=False,
        #                   biomt_header=self.format_biomt(), cryst_record=self.cryst_record, **kwargs)
        # chain_2.write(out_path='SecondContactingChain.pdb')
        return PDB.from_entities(entities, name='asu', log=self.log, pose_format=False,
                                 biomt_header=self.format_biomt(), cryst_record=self.cryst_record, **kwargs)

    # def handle_flags(self, design_selector=None, frag_db=None, ignore_clashes=False, **kwargs):
    #     self.ignore_clashes = ignore_clashes
    #     if design_selector:
    #         self.design_selector = design_selector
    #     else:
    #         self.design_selector = {}
    #     # if design_selector:
    #     #     self.create_design_selector(**design_selector)
    #     # else:
    #     #     # self.create_design_selector(selection={}, mask={}, required={})
    #     #     self.design_selector_entities = self.design_selector_entities.union(set(self.entities))
    #     #     self.design_selector_indices = self.design_selector_indices.union(set(self.pdb.atom_indices))
    #     if frag_db:
    #         # Attach an existing FragmentDB to the Pose
    #         self.attach_fragment_database(db=frag_db)
    #         for entity in self.entities:
    #             entity.attach_fragment_database(db=frag_db)

    def create_design_selector(self):  # , selection=None, mask=None, required=None):
        """Set up a design selector for the Pose including selections, masks, and required Entities and Atoms

        Sets:
            self.design_selector_indices (set[int])
            self.design_selector_entities (set[Entity])
            self.required_indices (set[int])
        """
        # if len(self.pdbs_d) > 1:
        #     self.log.debug('The design_selector may be incorrect as the Pose was initialized with multiple PDB '
        #                    'files. Proceed with caution if this is not what you expected!')

        def grab_indices(pdbs=None, entities=None, chains=None, residues=None, pdb_residues=None, atoms=None,
                         start_with_none=False):
            if start_with_none:
                entity_set = set()
                atom_indices = set()
                set_function = getattr(set, 'union')
            else:  # start with all indices and include those of interest
                entity_set = set(self.entities)
                atom_indices = set(self.pdb.atom_indices)  # TODO COMMENT OUT .pdb
                set_function = getattr(set, 'intersection')

            if pdbs:
                # atom_selection = set(self.pdb.get_residue_atom_indices(numbers=residues))
                raise DesignError('Can\'t select residues by PDB yet!')
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
                atom_indices = set_function(atom_indices, self.pdb.get_residue_atom_indices(numbers=residues))
            if pdb_residues:  # TODO COMMENT OUT .pdb v ^
                atom_indices = set_function(atom_indices, self.pdb.get_residue_atom_indices(numbers=residues, pdb=True))
            if atoms:  # TODO COMMENT OUT .pdb v
                atom_indices = set_function(atom_indices, [idx for idx in self.pdb.atom_indices if idx in atoms])

            return entity_set, atom_indices

        if 'selection' in self.design_selector:
            self.log.debug('The design_selection includes: %s' % self.design_selector['selection'])
            entity_selection, atom_selection = grab_indices(**self.design_selector['selection'])
        else:  # TODO COMMENT OUT .pdb v
            entity_selection, atom_selection = set(self.entities), set(self.pdb.atom_indices)

        if 'mask' in self.design_selector:
            self.log.debug('The design_mask includes: %s' % self.design_selector['mask'])
            entity_mask, atom_mask = grab_indices(**self.design_selector['mask'], start_with_none=True)
        else:
            entity_mask, atom_mask = set(), set()

        self.design_selector_entities = entity_selection.difference(entity_mask)
        self.design_selector_indices = atom_selection.difference(atom_mask)

        if 'required' in self.design_selector:
            self.log.debug('The required_residues includes: %s' % self.design_selector['required'])
            entity_required, self.required_indices = grab_indices(**self.design_selector['required'],
                                                                  start_with_none=True)
            if self.required_indices:  # only if indices are specified should we grab them
                self.required_residues = self.pdb.get_residues_by_atom_indices(indices=self.required_indices)
        else:  # TODO COMMENT OUT .pdb ^
            entity_required, self.required_indices = set(), set()

    # def construct_cb_atom_tree(self, entity1, entity2, distance=8):  # UNUSED
    #     """Create a atom tree using CB atoms from two PDB's
    #
    #     Args:
    #         entity1 (Structure): First PDB to query against
    #         entity2 (Structure): Second PDB which will be tested against pdb1
    #     Keyword Args:
    #         distance=8 (int): The distance to query in Angstroms
    #         include_glycine=True (bool): Whether glycine CA should be included in the tree
    #     Returns:
    #         query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
    #         pdb1_cb_indices (list): List of all CB indices from pdb1
    #         pdb2_cb_indices (list): List of all CB indices from pdb2
    #     """
    #     # Get CB Atom Coordinates including CA coordinates for Gly residues
    #     entity1_indices = np.array(self.asu.entity(entity1).cb_indices)
    #     # mask = np.ones(self.asu.number_of_atoms, dtype=int)  # mask everything
    #     # mask[index_array] = 0  # we unmask the useful coordinates
    #     entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want!
    #
    #     entity2_indices = np.array(self.asu.entity(entity2).cb_indices)
    #     entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want!
    #
    #     # Construct CB Tree for PDB1
    #     entity1_tree = BallTree(entity1_coords)
    #
    #     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
    #     return entity1_tree.query_radius(entity2_coords, distance)

    def return_interface(self, distance=8.0):
        """Provide a view of the Pose interface by generating a Structure containing only interface Residues

        Keyword Args:
            distance=8.0 (float): The distance across the interface to query for Residue contacts
        Returns:
            (Structure): The Structure containing only the Residues in the interface
        """
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
        # interface_symmetry_mates = self.return_symmetry_mates(interface_asu_structure)
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
        kmeans_cluster_model = KMeans(n_clusters=number_of_models, init=initial_interface_coords, n_init=1)\
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
        interface_asu_structure.replace_coords(closest_interface_coords)

        return interface_asu_structure

    def find_interface_pairs(self, entity1=None, entity2=None, distance=8):
        """Get pairs of Residues that have CB Atoms within a distance between two Entities

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms
        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.
        design_selector aware. Will remove interface residues if not active under the design selector

        Keyword Args:
            entity1=None (Entity): First entity to measure interface between
            entity2=None (Entity): Second entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
        Returns:
            (list[tuple]): A list of interface residue numbers across the interface
        """
        self.log.debug('Entity %s | Entity %s interface query' % (entity1.name, entity2.name))
        # Get CB Atom Coordinates including CA coordinates for Gly residues
        entity1_indices = entity1.cb_indices
        entity2_indices = entity2.cb_indices

        if self.design_selector_indices:  # subtract the masked atom indices from the entity indices
            before = len(entity1_indices) + len(entity2_indices)
            entity1_indices = list(set(entity1_indices).intersection(self.design_selector_indices))
            entity2_indices = list(set(entity2_indices).intersection(self.design_selector_indices))
            self.log.debug('Applied design selection to interface identification. Number of indices before '
                           'selection = %d. Number after = %d' % (before, len(entity1_indices) + len(entity2_indices)))

        if not entity1_indices or not entity2_indices:
            return

        coords_length = len(self.coords)
        if self.symmetry:
            sym_string = 'symmetric '
            self.log.debug('Number of Atoms in Pose: %s' % coords_length)
            # get all symmetric indices
            entity2_indices = [idx + (coords_length * model_number)
                               for model_number in range(self.number_of_symmetry_mates) for idx in entity2_indices]
            if entity1 == entity2:
                # We don't want interactions with the symmetric asu model or intra-oligomeric contacts
                if entity1.is_oligomeric:  # remove oligomeric protomers (contains asu)
                    remove_indices = self.return_intra_oligomeric_symmetry_mate_indices(entity1)
                    self.log.debug('Removing indices from models %s due to detected oligomer'
                                   % ','.join(map(str, self.oligomeric_equivalent_model_idxs.get(entity1))))
                    self.log.debug('Removing %d indices from symmetric query due to detected oligomer'
                                   % (len(remove_indices)))
                else:  # remove asu
                    remove_indices = self.return_asu_equivalent_symmetry_mate_indices()
                self.log.debug('Number of indices before removal of \'self\' indices: %s' % len(entity2_indices))
                entity2_indices = list(set(entity2_indices).difference(remove_indices))
                self.log.debug('Final indices remaining after removing \'self\': %s' % len(entity2_indices))
            entity2_coords = self.model_coords[entity2_indices]  # only get the coordinate indices we want
        elif entity1 == entity2:
            # without symmetry, we can't measure this, unless intra-oligomeric contacts are desired
            self.log.warning('Entities are the same, but no symmetry is present. The interface between them will not be'
                             ' detected!')
            return
        else:
            sym_string = ''
            entity2_coords = self.coords[entity2_indices]  # only get the coordinate indices we want

        # Construct CB tree for entity1 and query entity2 CBs for a distance less than a threshold
        entity1_coords = self.coords[entity1_indices]  # only get the coordinate indices we want
        entity1_tree = BallTree(entity1_coords)
        if len(entity2_coords) == 0:  # ensure the array is not empty
            return []
        entity2_query = entity1_tree.query_radius(entity2_coords, distance)

        # Return residue numbers of identified coordinates
        self.log.info('Querying %d CB residues in Entity %s versus, %d CB residues in %sEntity %s'
                      % (len(entity1_indices), entity1.name, len(entity2_indices), sym_string, entity2.name))
        # contacting_pairs = [(pdb_residues[entity1_indices[entity1_idx]],
        #                      pdb_residues[entity2_indices[entity2_idx]])
        #                    for entity2_idx in range(entity2_query.size) for entity1_idx in entity2_query[entity2_idx]]

        # contacting_pairs = [(pdb_atoms[entity1_indices[entity1_idx]].residue_number,
        #                      pdb_atoms[entity2_indices[entity2_idx]].residue_number)
        #                     for entity2_idx, entity1_contacts in enumerate(entity2_query)
        #                     for entity1_idx in entity1_contacts]
        contacting_pairs = [(self.coords_indexed_residues[entity1_indices[entity1_idx]],
                             self.coords_indexed_residues[entity2_indices[entity2_idx] % coords_length])
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

    def find_interface_residues(self, entity1=None, entity2=None, **kwargs):  # distance=8
        """Get unique Residues across an interface provided by two Entities

        Keyword Args:
            entity1=None (Entity): First Entity to measure interface between
            entity2=None (Entity): Second Entity to measure interface between
            distance=8 (int): The distance to query the interface in Angstroms
        Sets:
            self.interface_residues (dict[mapping[tuple[Structure,Structure],tuple[list[Residue],list[Residue]]]):
            The Entity1/Entity2 interface mapped to the interface Residues
        """
        entity1_residues, entity2_residues = \
            split_interface_residues(self.find_interface_pairs(entity1=entity1, entity2=entity2, **kwargs))

        if not entity1_residues or not entity2_residues:
            self.log.info('Interface search at %s | %s found no interface residues' % (entity1.name, entity2.name))
            self.interface_residues[(entity1, entity2)] = ([], [])
            return
        # else:
        if entity1 == entity2:  # symmetric query
            for residue in entity2_residues:  # entity2 usually has fewer residues, this might be quickest
                if residue in entity1_residues:  # the interface is dimeric and should only have residues on one side
                    entity1_residues, entity2_residues = \
                        sorted(set(entity1_residues).union(entity2_residues), key=lambda res: res.number), []
                    break
        self.log.info(
            'At Entity %s | Entity %s interface:\n\t%s found residue numbers: %s\n\t%s found residue numbers: %s'
            % (entity1.name, entity2.name, entity1.name, ', '.join(str(res.number) for res in entity1_residues),
               entity2.name, ', '.join(str(res.number) for res in entity2_residues)))

        self.interface_residues[(entity1, entity2)] = (entity1_residues, entity2_residues)
        entities = [entity1, entity2]
        self.log.debug('Added interface_residues: %s' % ['%d%s' % (residue.number, entities[idx].chain_id)
                       for idx, entity_residues in enumerate(self.interface_residues[(entity1, entity2)])
                       for residue in entity_residues])

    def find_interface_atoms(self, entity1=None, entity2=None, distance=4.68):
        """Get pairs of heavy atom indices that are within a distance at the interface between two Entities

        Caution: Pose must have Coords representing all atoms! Residue pairs are found using CB indices from all atoms
        Symmetry aware. If symmetry is used, by default all atomic coordinates for entity2 are symmeterized.

        Keyword Args:
            entity1=None (Entity): First Entity to measure interface between
            entity2=None (Entity): Second Entity to measure interface between
            distance=3.28 (float): The distance to measure contacts between atoms. Default = CB radius + 2.8 H2O probe
        Returns:
            (list[tuple]): The Atom indices for the interface
        """
        residues1, residues2 = self.interface_residues.get((entity1, entity2))
        if not residues1:
            return
        if not residues2:  # check if the interface is a self and all residues are in residues1
            residues2 = copy(residues1)

        entity1_indices = []
        for residue in residues1:
            entity1_indices.extend(residue.heavy_atom_indices)

        entity2_indices = []
        for residue in residues2:
            entity2_indices.extend(residue.heavy_atom_indices)

        coords_length = len(self.coords)
        if self.symmetry:
            # get all symmetric indices
            entity2_indices = [idx + (coords_length * model_number)
                               for model_number in range(self.number_of_symmetry_mates) for idx in entity2_indices]

        interface_atom_tree = BallTree(self.coords[entity1_indices])
        atom_query = interface_atom_tree.query_radius(self.model_coords[entity2_indices], distance)
        contacting_pairs = [(entity1_indices[entity1_idx], entity2_indices[entity2_idx])
                            for entity2_idx, entity1_contacts in enumerate(atom_query)
                            for entity1_idx in entity1_contacts]
        return contacting_pairs

    def interface_local_density(self, distance=12.):
        """Find the number of Atoms within a distance of each Atom in the Structure and add the density as an average
           value over each Residue

        Keyword Args:
            distance=12.0 (float): The cutoff distance with which Atoms should be included in local density
        Returns:
            (float): The local density around each interface
        """
        interface_indices1, interface_indices2 = [], []
        for entity1, entity2 in self.interface_residues:
            atoms_indices1, atoms_indices2 = \
                split_interface_numbers(self.find_interface_atoms(entity1=entity1, entity2=entity2))
            interface_indices1.extend(atoms_indices1), interface_indices2.extend(atoms_indices2)
            # print('INDICES1', atoms_indices1, '\nINDICES2', atoms_indices2)

        interface_indices = list(set(interface_indices1).union(interface_indices2))
        # print('INTERFACE ATOMS SET len', len(interface_indices))
        interface_coords = self.model_coords[interface_indices]  # OPERATION ASSUMES ASU IS MODEL_COORDS GROUP 1
        interface_tree = BallTree(interface_coords)
        interface_counts = interface_tree.query_radius(interface_coords, distance, count_only=True)
        # print('COUNTS LEN', len(interface_counts))
        # print('COUNTS', interface_counts)
        return interface_counts.mean()

    def query_interface_for_fragments(self, entity1=None, entity2=None):
        """For all found interface residues in a Entity/Entity interface, search for corresponding fragment pairs

        Keyword Args:
            entity1=None (Structure): The first Entity to measure for an interface
            entity2=None (Structure): The second Entity to measure for an interface
        Sets:
            self.fragment_queries (dict[mapping[tuple[Structure,Structure], list[dict[mapping[str,any]]]]])
        """
        entity1_residues, entity2_residues = self.interface_residues.get((entity1, entity2))
        if not entity1_residues or not entity2_residues:
            self.log.info('No residues found at the %s | %s interface. Fragments not available'
                          % (entity1.name, entity2.name))
            self.fragment_queries[(entity1, entity2)] = []
            return
        if entity1 == entity2 and entity1.is_oligomeric:
            # surface_frags1.extend(surface_frags2)
            # surface_frags2 = surface_frags1
            entity1_residues = set(entity1_residues + entity2_residues)
            entity2_residues = entity1_residues
        # else:
        #     _entity1_residues, _entity2_residues = entity1_residues, entity2_residues
        # Todo make so that the residue objects support fragments instead of converting back
        entity1_res_numbers = sorted(residue.number for residue in entity1_residues)
        entity2_res_numbers = sorted(residue.number for residue in entity2_residues)
        self.log.debug('At Entity %s | Entity %s interface, searching for fragments at the surface of:%s%s'
                       % (entity1.name, entity2.name,
                          '\n\tEntity %s: Residues %s' % (entity1.name, ', '.join(map(str, entity1_res_numbers))),
                          '\n\tEntity %s: Residues %s' % (entity2.name, ', '.join(map(str, entity2_res_numbers)))))

        surface_frags1 = entity1.get_fragments(residue_numbers=entity1_res_numbers, representatives=self.frag_db.reps)
        surface_frags2 = entity2.get_fragments(residue_numbers=entity2_res_numbers, representatives=self.frag_db.reps)

        if not surface_frags1 or not surface_frags2:
            self.log.info('No fragments found at the %s | %s interface' % (entity1.name, entity2.name))
            self.fragment_queries[(entity1, entity2)] = []
            return
        else:
            self.log.debug(
                'At Entity %s | Entity %s interface:\t%s has %d interface fragments\t%s has %d interface fragments'
                % (entity1.name, entity2.name, entity1.name, len(surface_frags1), entity2.name, len(surface_frags2)))

        if self.symmetry:
            # even if entity1 == entity2, only need to expand the entity2 fragments due to surface/ghost frag mechanics
            # asu frag subtraction is unnecessary THIS IS ALL WRONG DEPENDING ON THE CONTEXT
            if entity1 == entity2:
                skip_models = self.oligomeric_equivalent_model_idxs[entity1]  # + [self.asu_equivalent_model_idx]
                self.log.debug('Skipping oligomeric models %s' % skip_models)
            else:
                skip_models = []
            surface_frags2_nested = [self.return_symmetry_mates(frag) for frag in surface_frags2]
            surface_frags2.clear()
            for frag_mates in surface_frags2_nested:
                surface_frags2.extend(frag for sym_idx, frag in enumerate(frag_mates) if sym_idx not in skip_models)
            self.log.debug('Entity %s has %d symmetric fragments' % (entity2.name, len(surface_frags2)))

        entity1_coords = entity1.get_backbone_and_cb_coords()  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = \
            find_fragment_overlap_at_interface(entity1_coords, surface_frags1, surface_frags2, fragdb=self.frag_db,
                                               euler_lookup=self.euler_lookup)
        self.log.info('Found %d overlapping fragment pairs at the %s | %s interface.'
                      % (len(ghostfrag_surfacefrag_pairs), entity1.name, entity2.name))
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
            self.split_interface_residues (dict): Residue/Entity id of each residue at the interface identified by interface id
                as split by topology
        """
        self.log.debug('Find and split interface using active_entities: %s' %
                       [entity.name for entity in self.active_entities])
        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.find_interface_residues(*entity_pair)

        self.check_interface_topology()

    def check_interface_topology(self):
        """From each pair of entities that share an interface, split the identified residues into two distinct groups.
        If an interface can't be composed into two distinct groups, raise a DesignError

        Sets:
            self.split_interface_residues (dict): Residue/Entity id of each residue at the interface identified by interface id
                as split by topology
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
            raise DesignError('Interface was unable to be split because no residues were found on one side of the'
                              ' interface!')
        else:
            self.log.debug('The interface is split as:\n\tInterface 1: %s\n\tInterface 2: %s'
                           % tuple(','.join('%d%s' % (res.number, ent.chain_id) for res, ent in residues_entities)
                                   for residues_entities in self.split_interface_residues.values()))

    def interface_secondary_structure(self):
        """From a split interface, curate the secondary structure topology for each

        Keyword Args:
            source_db=None (Database): A Database object connected to secondary structure db
            source_dir=None (str): The location of the directory containing Stride files
        """
        pose_secondary_structure = ''
        for entity in self.active_entities:
            if not entity.secondary_structure:
                if self.source_db:
                    parsed_secondary_structure = self.source_db.stride.retrieve_data(name=entity.name)
                    if parsed_secondary_structure:
                        entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                    else:
                        entity.stride(to_file=self.source_db.stride.store(entity.name))
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
                    raise IndexError('The index %d, from entity %s, residue %d is not found in self.ss_index_array with length %d'
                                     % (residue.number - 1, entity.name, residue.number, len(self.ss_index_array)))

        self.log.debug('Found interface secondary structure: %s' % self.split_interface_ss_elements)

    def interface_design(self, evolution=True, fragments=True, query_fragments=False, fragment_source=None,
                         write_fragments=True, frag_db='biological_interfaces', des_dir=None):  # Todo deprec. des_dir
        """Take the provided PDB, and use the ASU to compute calculations relevant to interface design.

        This process identifies the ASU (if one is not explicitly provided, enables Pose symmetry,

        Sets:
            design_dir.info['fragments'] to True is fragments are queried
        """
        # Ensure ASU. This should be done on loading from PDB file with Pose.from_asu()/Pose.from_pdb()
        # save self.asu to design_dir.asu now that we have cleaned any chain issues and renumbered residues
        # self.pdb.write(out_path=design_dir.asu)

        # if design_dir.nano:
        #     # num_chains = len(self.pdb.chain_id_list)
        #     # if num_chains != len(design_dir.oligomers):
        #     #     # oligomer_file = glob(os.path.join(des_dir.path, pdb_codes[0] + '_tx_*.pdb'))
        #     #     # assert len(oligomer_file) == 1, 'More than one matching file found with %s' % pdb_codes[0] + '_tx_*.pdb'
        #     #     # # assert len(oligomer_file) == 1, '%s: More than one matching file found with %s' % \
        #     #     # #                                 (des_dir.path, pdb_codes[0] + '_tx_*.pdb')
        #     #     # first_oligomer = PDB(file=oligomer_file[0])
        #     #     # # first_oligomer = SDUtils.read_pdb(oligomer_file[0])
        #     #     # # find the number of ATOM records for template_pdb chain1 using the same oligomeric chain as model
        #     #     # for atom_idx in range(len(first_oligomer.chain(template_pdb.chain_id_list[0]))):
        #     #     for atom_idx in range(len(design_dir.oligomers.chain(self.pdb.entity(1)))):
        #     #         self.pdb.atoms[atom_idx].chain = self.pdb.chain_id_list[0].lower()
        #     #     self.pdb.chain_id_list = [self.pdb.chain_id_list[0].lower(), self.pdb.chain_id_list[0]]
        #     #     num_chains = len(self.pdb.chain_id_list)
        #     #     logger.warning(
        #     #         '%s: Incorrect chain count: %d. Chains probably have the same id! Temporarily changing IDs\'s to'
        #     #         ' %s' % (design_dir.path, num_chains, self.pdb.chain_id_list))
        #     #     # Save the renamed chain PDB to the source.pdb
        #     #     self.pdb.write(out_path=design_dir.source)
        #
        #     assert len(design_dir.oligomers) == num_chains, \
        #         'Number of chains \'%d\' in ASU doesn\'t match number of building blocks \'%d\'' \
        #         % (num_chains, len(design_dir.oligomers))
        # else:
        #     # sdf_file_name = os.path.join(os.path.dirname(oligomer[name].filepath), '%s.sdf' % oligomer[name].name)
        #     # sym_definition_files[name] = oligomer[name].make_sdf(out_path=sdf_file_name, modify_sym_energy=True)
        self.log.debug('Entities: %s' % ', '.join(entity.name for entity in self.entities))
        self.log.debug('Active Entities: %s' % ', '.join(entity.name for entity in self.active_entities))
        # self.log.debug('Designable Residues: %s' % ', '.join(entity.name for entity in self.design_selector_indices))
        # self.log.info('Symmetry is: %s' % symmetry)
        # if symmetry and isinstance(symmetry, dict):
        #     self.set_symmetry(**symmetry)

        # we get interface residues for the designable entities as well as interface_topology at DesignDirectory level
        if fragments:
            if query_fragments:  # search for new fragment information
                self.generate_interface_fragments(out_path=des_dir.frags, write_fragments=write_fragments)
            else:  # No fragment query, add existing fragment information to the pose
                if not self.frag_db:
                    self.connect_fragment_database(source=frag_db, init=False)  # no need to initialize
                    # Attach an existing FragmentDB to the Pose
                    self.attach_fragment_database(db=self.frag_db)
                    for entity in self.entities:
                        entity.attach_fragment_database(db=self.frag_db)

                if fragment_source is None:
                    raise DesignError('Fragments were set for design but there were none found! Try including '
                                      '--query_fragments in your input flags and rerun this command or generate '
                                      'them separately with \'%s %s\''
                                      % (PUtils.program_command, PUtils.generate_fragments))

                # Must provide des_dir.fragment_observations then specify whether the Entity in question is from the
                # mapped or paired chain (entity1 is mapped, entity2 is paired from Nanohedra). Then, need to renumber
                # fragments to Pose residue numbering when added to fragment queries
                # if design_dir.nano:  # Todo depreciate this check as Nanohedra outputs will be in Pose Numbering
                #     if len(self.entities) > 2:  # Todo compatible with > 2 entities
                #         raise DesignError('Not able to solve fragment/residue membership with more than 2 Entities!')
                #     self.log.debug('Fragment data found in Nanohedra docking. Solving fragment membership for '
                #                    'Entity\'s: %s by PDB numbering correspondence'
                #                    % ','.join(entity.name for entity in self.entities))
                #     self.add_fragment_query(entity1=self.entities[0], entity2=self.entities[1], query=fragment_source,
                #                             pdb_numbering=True)
                # else:  # assuming the input is in Pose numbering!
                self.log.debug('Fragment data found from prior query. Solving query index by Pose numbering/Entity '
                               'matching')
                self.add_fragment_query(query=fragment_source)

            for query_pair, fragment_info in self.fragment_queries.items():
                self.log.debug('Query Pair: %s, %s\n\tFragment Info:%s' % (query_pair[0].name, query_pair[1].name,
                                                                           fragment_info))
                for query_idx, entity in enumerate(query_pair):
                    # # Attach an existing FragmentDB to the Pose
                    # entity.connect_fragment_database(location=frag_db, db=design_dir.frag_db)
                    entity.attach_fragment_database(db=self.frag_db)
                    entity.assign_fragments(fragments=fragment_info,
                                            alignment_type=SequenceProfile.idx_to_alignment_type[query_idx])
        for entity in self.entities:
            # entity.retrieve_sequence_from_api(entity_id=entity)  # Todo
            # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
            if entity not in self.active_entities:  # we shouldn't design, add a null profile instead
                entity.add_profile(null=True)
            else:  # add a real profile
                if self.source_db:
                    entity.sequence_file = self.source_db.sequences.retrieve_file(name=entity.name)
                    entity.evolutionary_profile = self.source_db.hhblits_profiles.retrieve_data(name=entity.name)
                    if not entity.pssm_file:
                        entity.pssm_file = self.source_db.hhblits_profiles.retrieve_file(name=entity.name)
                    if len(entity.evolutionary_profile) != entity.number_of_residues:
                        # profile was made with reference or the sequence has inserts and deletions of equal length
                        # A more stringent check could move through the evolutionary_profile[idx]['type'] key versus the
                        # entity.sequence[idx]
                        entity.fit_evolutionary_profile_to_structure()
                    profiles_path = self.source_db.hhblits_profiles.location
                else:
                    profiles_path = des_dir.profiles

                if not entity.sequence_file:  # Todo move up to line 2749?
                    entity.write_fasta_file(entity.reference_sequence, name=entity.name, out_path=des_dir.sequences)
                entity.add_profile(evolution=evolution, fragments=fragments, out_path=profiles_path)

        # Update DesignDirectory with design information # Todo include in DesignDirectory initialization by args?
        # This info is pulled out in AnalyzeOutput from Rosetta currently

        if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
            self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
            self.fragment_pssm_file = self.write_pssm_file(self.fragment_profile, PUtils.fssm, out_path=des_dir.data)
            # design_dir.info['fragment_profile'] = self.fragment_pssm_file
            # design_dir.info['fragment_database'] = frag_db
            # self.log.debug('Fragment Specific Scoring Matrix: %s' % str(self.fragment_profile))
            # this dictionary is removed of all entries that are not fragment populated.
            clean_fragment_profile = dict((residue, data) for residue, data in self.fragment_profile.items()
                                          if data.get('stats', (None,))[0])  # [0] must contain a fragment observation
            self.interface_data_file = \
                pickle_object(clean_fragment_profile, '%s_fragment_profile' % self.frag_db.source,
                              out_path=des_dir.data)
            # design_dir.info['fragment_data'] = self.interface_data_file

        if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
            self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
            # self.log.debug('Position Specific Scoring Matrix: %s' % str(self.evolutionary_profile))
            self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=des_dir.data)
            # design_dir.info['evolutionary_profile'] = self.pssm_file

        self.combine_profile([entity.profile for entity in self.entities])
        # self.log.debug('Design Specific Scoring Matrix: %s' % str(self.profile))
        self.design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=des_dir.data)
        # design_dir.info['design_profile'] = self.design_pssm_file
        # -------------------------------------------------------------------------
        # self.solve_consensus()
        # -------------------------------------------------------------------------

    def return_fragment_observations(self):
        """Return the fragment observations identified on the pose regardless of Entity binding

        Returns:
            (list[dict]): [{'mapped': int, 'paired': int, 'cluster': str, 'match': float}, ...]
        """
        observations = []
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        for query_pair, fragment_matches in self.fragment_queries.items():
            observations.extend(fragment_matches)

        return observations

    def return_fragment_metrics(self, fragments=None, by_interface=False, entity1=None, entity2=None, by_entity=False):
        """From self.fragment_queries, return the specified fragment metrics. By default returns the entire Pose

        Keyword Args:
            metrics=None (list): A list of calculated metrics
            by_interface=False (bool): Return fragment metrics for each particular interface found in the Pose
            entity1=None (Entity): The first Entity object to identify the interface if per_interface=True
            entity2=None (Entity): The second Entity object to identify the interface if per_interface=True
            by_entity=False (bool): Return fragment metrics for each Entity found in the Pose
        Returns:
            (dict): {query1: {all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap,
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

    def renumber_fragments_to_pose(self, fragments):
        for idx, fragment in enumerate(fragments):
            # if self.pdb.residue_from_pdb_numbering():
            # only assign the new fragment number info to the fragments if the residue is found
            map_pose_number = self.pdb.residue_number_from_pdb(fragment['mapped'])  # TODO COMMENT OUT .pdb
            fragment['mapped'] = map_pose_number if map_pose_number else fragment['mapped']
            pair_pose_number = self.pdb.residue_number_from_pdb(fragment['paired'])  # TODO COMMENT OUT .pdb
            fragment['paired'] = pair_pose_number if pair_pose_number else fragment['paired']
            # fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
            # fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
            fragments[idx] = fragment

        return fragments

    def add_fragment_query(self, entity1=None, entity2=None, query=None, pdb_numbering=False):
        """For a fragment query loaded from disk between two entities, add the fragment information to the Pose"""
        # Todo This function has logic pitfalls if residue numbering is in PDB format. How easy would
        #  it be to refactor fragment query to deal with the chain info from the frag match file?
        if pdb_numbering:  # Renumber self.fragment_map and self.fragment_profile to Pose residue numbering
            query = self.renumber_fragments_to_pose(query)
            # for idx, fragment in enumerate(fragment_source):
            #     fragment['mapped'] = self.pdb.residue_number_from_pdb(fragment['mapped'])
            #     fragment['paired'] = self.pdb.residue_number_from_pdb(fragment['paired'])
            #     fragment_source[idx] = fragment
            if entity1 and entity2 and query:
                self.fragment_queries[(entity1, entity2)] = query
        else:
            entity_pairs = [(self.pdb.entity_from_residue(fragment['mapped']),   # TODO COMMENT OUT .pdb < and v
                             self.pdb.entity_from_residue(fragment['paired'])) for fragment in query]
            if all([all(pair) for pair in entity_pairs]):
                for entity_pair, fragment in zip(entity_pairs, query):
                    if entity_pair in self.fragment_queries:
                        self.fragment_queries[entity_pair].append(fragment)
                    else:
                        self.fragment_queries[entity_pair] = [fragment]
            else:
                raise DesignError('%s: Couldn\'t locate Pose Entities passed by residue number. Are the residues in '
                                  'Pose Numbering? This may be occurring due to fragment queries performed on the PDB '
                                  'and not explicitly searching using pdb_numbering = True. Retry with the appropriate'
                                  ' modifications' % self.add_fragment_query.__name__)

    def connect_fragment_database(self, source=None, init=False, **kwargs):  # Todo Clean up
        """Generate a new connection. Initialize the representative library by passing init=True"""
        if not source:  # Todo fix once multiple are available
            source = 'biological_interfaces'
        self.frag_db = FragmentDatabase(source=source, init_db=init)
        #                               source=source, init_db=init_db)

    def generate_interface_fragments(self, write_fragments=True, out_path=None, new_db=False):
        """Using the attached fragment database, generate interface fragments between the Pose interfaces

        Keyword Args:
            write_fragments=True (bool): Whether or not to write the located fragments
            out_path=None (str): The location to write each fragment file
            new_db=False (bool): Whether a fragment database should be initialized for the interface fragment search
        """
        if not self.frag_db:  # There is no fragment database connected
            # Connect to a new DB, Todo parameterize which one should be used with source=
            self.connect_fragment_database(init=True)  # default init=False, we need an initiated one to generate frags

        if not self.interface_residues:
            self.find_and_split_interface()  # shouldn't occur with the set up on 3/17/21, upstream funcs call this 1st
            # for entity_pair in combinations_with_replacement(self.active_entities, 2):
            #     self.find_interface_residues(*entity_pair)

        for entity_pair in combinations_with_replacement(self.active_entities, 2):
            self.log.debug('Querying Entity pair: %s, %s for interface fragments'
                           % tuple(entity.name for entity in entity_pair))
            self.query_interface_for_fragments(*entity_pair)

        if write_fragments:
            self.write_fragment_pairs(self.fragment_pairs, out_path=out_path)
            frag_file = os.path.join(out_path, PUtils.frag_text_file)
            if os.path.exists(frag_file):
                os.system('rm %s' % frag_file)  # ensure old file is removed before new write
            for match_count, (ghost_frag, surface_frag, match) in enumerate(self.fragment_pairs, 1):
                write_frag_match_info_file(ghost_frag=ghost_frag, matched_frag=surface_frag,
                                           overlap_error=z_value_from_match_score(match),
                                           match_number=match_count, out_path=out_path)

    def write_fragment_pairs(self, ghostfrag_surffrag_pairs, out_path=os.getcwd()):
        for idx, (interface_ghost_frag, interface_mono_frag, match_score) in enumerate(ghostfrag_surffrag_pairs, 1):
            fragment, _ = dictionary_lookup(self.frag_db.paired_frags, interface_ghost_frag.get_ijk())
            trnsfmd_fragment = fragment.return_transformed_copy(**interface_ghost_frag.aligned_fragment.transformation)
            trnsfmd_fragment.write(out_path=os.path.join(out_path, '%d_%d_%d_fragment_match_%d.pdb'
                                                         % (*interface_ghost_frag.get_ijk(), idx)))
            # interface_ghost_frag.structure.write(out_path=os.path.join(out_path, '%d_%d_%d_fragment_overlap_match_%d.pdb'
            #                                                            % (*interface_ghost_frag.get_ijk(), idx)))

    def return_symmetry_parameters(self):
        """Return the symmetry parameters from a SymmetricModel

        Returns:
            (dict): {symmetry: (str), dimension: (int), uc_dimensions: (list), expand_matrices: (list[list])}
        """
        return {'symmetry': self.__dict__['symmetry'],
                'uc_dimensions': self.__dict__['uc_dimensions'],
                'expand_matrices': self.__dict__['expand_matrices'],
                'dimension': self.__dict__['dimension']}

    def format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            **kwargs
        Returns:
            (str)
        """
        # if self.reference_sequence:
        formated_reference_sequence = {entity.chain_id: entity.reference_sequence for entity in self.entities}
        chain_lengths = {chain: len(sequence) for chain, sequence in formated_reference_sequence.items()}
        formated_reference_sequence = \
            {chain: ' '.join(map(str.upper, map(protein_letters_1to3_extended.get, sequence)))
             for chain, sequence in formated_reference_sequence.items()}
        return '%s\n' \
               % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
                           % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                           for chain, sequence in formated_reference_sequence.items()
                           for line_number in range(1, 1 + ceil(len(sequence)/seq_res_len)))
        # else:
        #     return ''

    def debug_pdb(self, tag=None):
        """Write out all Structure objects for the Pose PDB"""
        with open('%sDEBUG_POSE_PDB_%s.pdb' % ('%s_' % tag if tag else '', self.name), 'w') as f:
            available_chain_ids = self.pdb.return_chain_generator()  # TODO COMMENT OUT .pdb
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


# def fetch_pdbs(codes, location=PUtils.pdb_db):  # UNUSED
#     """Fetch PDB object of each chain from PDBdb or PDB server
#
#     Args:
#         codes (iter): Any iterable of PDB codes
#     Keyword Args:
#         location= : Location of the  on disk
#     Returns:
#         (dict): {pdb_code: PDB.py object, ...}
#     """
#     if PUtils.pdb_source == 'download_pdb':
#         get_pdb = download_pdb
#         # doesn't return anything at the moment
#     else:
#         get_pdb = (lambda pdb_code, dummy: glob(os.path.join(PUtils.pdb_location, subdirectory(pdb_code),
#                                                              '%s.pdb' % pdb_code)))
#         # returns a list with matching file (should only be one)
#     oligomers = {}
#     for code in codes:
#         pdb_file_name = get_pdb(code, location=des_dir.pdbs)
#         assert len(pdb_file_name) == 1, 'More than one matching file found for pdb code %s' % code
#         oligomers[code] = PDB(file=pdb_file_name[0])
#         oligomers[code].name = code
#         oligomers[code].reorder_chains()
#
#     return oligomers


# def construct_cb_atom_tree(pdb1, pdb2, distance=8):  # UNUSED
#     """Create a atom tree using CB atoms from two PDB's
#
#     Args:
#         pdb1 (PDB): First PDB to query against
#         pdb2 (PDB): Second PDB which will be tested against pdb1
#     Keyword Args:
#         distance=8 (int): The distance to query in Angstroms
#         include_glycine=True (bool): Whether glycine CA should be included in the tree
#     Returns:
#         query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
#         pdb1_cb_indices (list): List of all CB indices from pdb1
#         pdb2_cb_indices (list): List of all CB indices from pdb2
#     """
#     # Get CB Atom Coordinates including CA coordinates for Gly residues
#     pdb1_coords = np.array(pdb1.get_cb_coords())  # InclGlyCA=gly_ca))
#     pdb2_coords = np.array(pdb2.get_cb_coords())  # InclGlyCA=gly_ca))
#
#     # Construct CB Tree for PDB1
#     pdb1_tree = BallTree(pdb1_coords)
#
#     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
#     return pdb1_tree.query_radius(pdb2_coords, distance)
#
#
# def find_interface_pairs(pdb1, pdb2, distance=8):  # , gly_ca=True):  # UNUSED
#     """Get pairs of residues across an interface within a certain distance
#
#         Args:
#             pdb1 (PDB): First pdb to measure interface between
#             pdb2 (PDB): Second pdb to measure interface between
#         Keyword Args:
#             distance=8 (int): The distance to query in Angstroms
#         Returns:
#             interface_pairs (list(tuple): A list of interface residue pairs across the interface
#     """
#     query = construct_cb_atom_tree(pdb1, pdb2, distance=distance)
#
#     # Map Coordinates to Atoms
#     pdb1_cb_indices = pdb1.cb_indices
#     pdb2_cb_indices = pdb2.cb_indices
#
#     # Map Coordinates to Residue Numbers
#     interface_pairs = []
#     for pdb2_index in range(len(query)):
#         if query[pdb2_index].tolist() != list():
#             pdb2_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_index]].residue_number
#             for pdb1_index in query[pdb2_index]:
#                 pdb1_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_index]].residue_number
#                 interface_pairs.append((pdb1_res_num, pdb2_res_num))
#
#     return interface_pairs


# def split_interface_pairs(interface_pairs):
#     residues1, residues2 = zip(*interface_pairs)
#     return sorted(set(residues1), key=int), sorted(set(residues2), key=int)
#
#
# def find_interface_residues(pdb1, pdb2, distance=8):
#     """Get unique residues from each pdb across an interface
#
#         Args:
#             pdb1 (PDB): First pdb to measure interface between
#             pdb2 (PDB): Second pdb to measure interface between
#         Keyword Args:
#             distance=8 (int): The distance to query in Angstroms
#         Returns:
#             (tuple(set): A tuple of interface residue sets across an interface
#     """
#     return split_interface_pairs(find_interface_pairs(pdb1, pdb2, distance=distance))


# @njit
def find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2, fragdb=None,
                                       euler_lookup=None, max_z_value=2):
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From two Structure's, score the interface between them according to Nanohedra's fragment matching"""
    if not fragdb:
        fragdb = FragmentDB()
        fragdb.get_monofrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_rep_dict()
        fragdb.get_intfrag_cluster_info_dict()
    if not euler_lookup:
        euler_lookup = EulerLookup()

    # logger.debug('Starting Ghost Frag Lookup')
    oligomer1_bb_tree = BallTree(entity1_coords)
    interface_ghost_frags1 = []
    for frag1 in interface_frags1:
        ghostfrags = frag1.get_ghost_fragments(fragdb.indexed_ghosts, oligomer1_bb_tree)
        if ghostfrags:
            interface_ghost_frags1.extend(ghostfrags)
    logger.debug('Finished Ghost Frag Lookup')

    # Get fragment guide coordinates
    interface_ghostfrag_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in interface_ghost_frags1])
    interface_surf_frag_guide_coords = np.array([frag2.guide_coords for frag2 in interface_frags2])

    # Check for matching Euler angles
    # TODO create a stand alone function
    # logger.debug('Starting Euler Lookup')
    overlapping_ghost_indices, overlapping_surf_indices = \
        euler_lookup.check_lookup_table(interface_ghostfrag_guide_coords, interface_surf_frag_guide_coords)
    # logger.debug('Finished Euler Lookup')
    logger.debug('Found %d overlapping fragments' % len(overlapping_ghost_indices))
    # filter array by matching type for surface (i) and ghost (j) frags
    surface_type_i_array = np.array([interface_frags2[idx].i_type for idx in overlapping_surf_indices.tolist()])
    ghost_type_j_array = np.array([interface_ghost_frags1[idx].j_type for idx in overlapping_ghost_indices.tolist()])
    ij_type_match = np.where(surface_type_i_array == ghost_type_j_array, True, False)

    passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
    passing_ghost_coords = interface_ghostfrag_guide_coords[passing_ghost_indices]

    passing_surf_indices = overlapping_surf_indices[ij_type_match]
    passing_surf_coords = interface_surf_frag_guide_coords[passing_surf_indices]
    logger.debug('Found %d overlapping fragments in the same i/j type' % len(passing_ghost_indices))
    # precalculate the reference_rmsds for each ghost fragment
    reference_rmsds = np.array([interface_ghost_frags1[ghost_idx].rmsd for ghost_idx in passing_ghost_indices.tolist()])
    reference_rmsds = np.where(reference_rmsds == 0, 0.01, reference_rmsds)

    # logger.debug('Calculating passing fragment overlaps by RMSD')
    all_fragment_overlap = calculate_overlap(passing_ghost_coords, passing_surf_coords, reference_rmsds,
                                             max_z_value=max_z_value)
    # logger.debug('Finished calculating fragment overlaps')
    passing_overlap_indices = np.flatnonzero(all_fragment_overlap)
    logger.debug('Found %d overlapping fragments under the %f threshold' % (len(passing_overlap_indices), max_z_value))

    interface_ghostfrags = [interface_ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()]
    interface_monofrags2 = [interface_frags2[idx] for idx in passing_surf_indices[passing_overlap_indices].tolist()]
    passing_z_values = all_fragment_overlap[passing_overlap_indices]
    match_scores = match_score_from_z_value(passing_z_values)

    return list(zip(interface_ghostfrags, interface_monofrags2, match_scores))


def get_matching_fragment_pairs_info(ghostfrag_surffrag_pairs):
    """From a ghost fragment/surface fragment pair and corresponding match score, return the pertinent interface
    information

    Args:
        ghostfrag_surffrag_pairs (list[tuple]): Observed ghost and surface fragment overlaps and their match score
    Returns:
        (list[dict[mapping[str,any]]])
    """
    fragment_matches = []
    for interface_ghost_frag, interface_mono_frag, match_score in ghostfrag_surffrag_pairs:
        surffrag_ch1, surffrag_resnum1 = interface_ghost_frag.get_aligned_chain_and_residue()
        surffrag_ch2, surffrag_resnum2 = interface_mono_frag.get_central_res_tup()
        fragment_matches.append(dict(zip(('mapped', 'paired', 'match', 'cluster'),
                                     (surffrag_resnum1, surffrag_resnum2,  match_score,
                                      '%d_%d_%d' % interface_ghost_frag.get_ijk()))))
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


def calculate_interface_score(interface_pdb, write=False, out_path=os.getcwd()):
    """Takes as input a single PDB with two chains and scores the interface using fragment decoration"""
    interface_name = interface_pdb.name

    entity1 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[0]).atoms)
    entity1.update_attributes_from_pdb(interface_pdb)
    entity2 = PDB.from_atoms(interface_pdb.chain(interface_pdb.chain_id_list[-1]).atoms)
    entity2.update_attributes_from_pdb(interface_pdb)

    interacting_residue_pairs = find_interface_pairs(entity1, entity2)

    entity1_interface_residue_numbers, entity2_interface_residue_numbers = \
        get_interface_fragment_residue_numbers(entity1, entity2, interacting_residue_pairs)
    # entity1_ch_interface_residue_numbers, entity2_ch_interface_residue_numbers = \
    #     get_interface_fragment_chain_residue_numbers(entity1, entity2)

    entity1_interface_sa = entity1.get_surface_area_residues(entity1_interface_residue_numbers)
    entity2_interface_sa = entity2.get_surface_area_residues(entity2_interface_residue_numbers)
    interface_buried_sa = entity1_interface_sa + entity2_interface_sa

    interface_frags1 = entity1.get_fragments(residue_numbers=entity1_interface_residue_numbers)
    interface_frags2 = entity2.get_fragments(residue_numbers=entity2_interface_residue_numbers)
    entity1_coords = entity1.coords

    ghostfrag_surfacefrag_pairs = find_fragment_overlap_at_interface(entity1_coords, interface_frags1, interface_frags2)
    # fragment_matches = find_fragment_overlap_at_interface(entity1, entity2, entity1_interface_residue_numbers,
    #                                                       entity2_interface_residue_numbers)
    fragment_matches = get_matching_fragment_pairs_info(ghostfrag_surfacefrag_pairs)
    if write:
        write_fragment_pairs(ghostfrag_surfacefrag_pairs, out_path=out_path)

    # all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
    #     central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d = \
    #     calculate_match_metrics(fragment_matches)

    match_metrics = calculate_match_metrics(fragment_matches)
    # Todo
    #   'mapped': {'center': {'residues' (int): (set), 'score': (float), 'number': (int)},
    #                         'total': {'residues' (int): (set), 'score': (float), 'number': (int)},
    #                         'match_scores': {residue number(int): (list[score (float)]), ...},
    #                         'index_count': {index (int): count (int), ...},
    #                         'multiple_ratio': (float)}
    #              'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
    #              'total': {'center': {'score': , 'number': },
    #                        'total': {'score': , 'number': },
    #                        'index_count': , 'multiple_ratio': , 'observations': (int)}
    #              }

    total_residues = {'A': set(), 'B': set()}
    for pair in interacting_residue_pairs:
        total_residues['A'].add(pair[0])
        total_residues['B'].add(pair[1])

    total_residues = len(total_residues['A']) + len(total_residues['B'])

    percent_interface_matched = central_residues_with_fragment_overlap / total_residues
    percent_interface_covered = total_residues_with_fragment_overlap / total_residues

    interface_metrics = {'nanohedra_score': all_residue_score,
                         'nanohedra_score_central': center_residue_score,
                         'fragments': fragment_matches,
                         'multiple_fragment_ratio': multiple_frag_ratio,
                         'number_fragment_residues_central': central_residues_with_fragment_overlap,
                         'number_fragment_residues_all': total_residues_with_fragment_overlap,
                         'total_interface_residues': total_residues,
                         'number_fragments': len(fragment_matches),
                         'percent_residues_fragment_total': percent_interface_covered,
                         'percent_residues_fragment_center': percent_interface_matched,
                         'percent_fragment_helix': fragment_content_d['1'],
                         'percent_fragment_strand': fragment_content_d['2'],
                         'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
                         + fragment_content_d['5'],
                         'interface_area': interface_buried_sa}

    return interface_name, interface_metrics


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
