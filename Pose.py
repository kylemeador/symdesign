from __future__ import annotations

import os
import subprocess
from copy import copy, deepcopy
from itertools import chain as iter_chain, combinations_with_replacement, combinations, product
from math import sqrt, cos, sin, prod, ceil, pi
from pathlib import Path
from typing import Iterable, IO, Any, Sequence, AnyStr

import numpy as np
# from numba import njit, jit
from Bio.Data.IUPACData import protein_letters_1to3_extended
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.neighbors._ball_tree import BinaryTree  # this typing implementation supports BallTree or KDTree

import PathUtils as PUtils
import fragment
import wrapapi
from DesignMetrics import calculate_match_metrics, fragment_metric_template, format_fragment_metrics
from Query.PDB import retrieve_entity_id_by_sequence, query_pdb_by, get_entity_reference_sequence
from SequenceProfile import SequenceProfile, alignment_types, generate_alignment, get_equivalent_indices
from Structure import Coords, Structure, Structures, Chain, Entity, Residue, Residues, GhostFragment, MonoFragment, \
    write_frag_match_info_file, Fragment, StructureBase, ContainsAtomsMixin, superposition3d, ContainsChainsMixin
from SymDesignUtils import DesignError, ClashError, SymmetryError, z_value_from_match_score, start_log, null_log, \
    dictionary_lookup, digit_translate_table, calculate_match
from classes.EulerLookup import EulerLookup, euler_factory
from classes.SymEntry import get_rot_matrices, make_rotations_degenerate, SymEntry, point_group_setting_matrix_members,\
    symmetry_combination_format, parse_symmetry_to_sym_entry, symmetry_factory
from utils.GeneralUtils import transform_coordinate_sets
from utils.SymmetryUtils import valid_subunit_number, layer_group_cryst1_fmt_dict, \
    generate_cryst1_record, space_group_number_operations, point_group_symmetry_operators, \
    space_group_symmetry_operators, rotation_range, setting_matrices, inv_setting_matrices, \
    origin, flip_x_matrix, identity_matrix, valid_symmetries, multicomponent_valid_subunit_number


# Globals
logger = start_log(name=__name__)
index_offset = 1
seq_res_len = 52
config_directory = PUtils.pdb_db
sym_op_location = PUtils.sym_op_location


def subdirectory(name):
    return name[1:2]


# @njit
def find_fragment_overlap(entity1_coords: np.ndarray, residues1: list[Residue] | Residues,
                          residues2: list[Residue] | Residues, frag_db: fragment.FragmentDatabase = None,
                          euler_lookup: EulerLookup = None, min_match_value: float = 0.2) -> \
        list[tuple[GhostFragment, Fragment, float]]:
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From two sets of Residues, score the fragment overlap according to Nanohedra's fragment matching

    Args:
        entity1_coords:
        residues1:
        residues2:
        frag_db:
        euler_lookup:
        min_match_value: The minimum value which constitutes an acceptable fragment match
    """
    if not frag_db:
        frag_db = fragment.fragment_factory()

    if not euler_lookup:
        euler_lookup = euler_factory()

    # logger.debug('Starting Ghost Frag Lookup')
    oligomer1_bb_tree = BallTree(entity1_coords)
    ghost_frags1: list[GhostFragment] = []
    for residue in residues1:
        ghost_frags1.extend(residue.get_ghost_fragments(frag_db.indexed_ghosts, clash_tree=oligomer1_bb_tree))
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
    logger.debug(f'Found {len(overlapping_ghost_indices)} overlapping fragments in the same Euler rotational space')
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

    # logger.debug('Calculating passing fragment overlaps by RMSD')
    all_fragment_match = calculate_match(passing_ghost_coords, passing_frag_coords, reference_rmsds)
    passing_overlaps_indices = np.flatnonzero(all_fragment_match > min_match_value)
    # all_fragment_overlap = \
    #     calculate_overlap(passing_ghost_coords, passing_frag_coords, reference_rmsds, max_z_value=max_z_value)
    # logger.debug('Finished calculating fragment overlaps')
    # passing_overlap_indices = np.flatnonzero(all_fragment_overlap)
    logger.debug(f'Found {len(passing_overlaps_indices)} overlapping fragments over the {min_match_value} threshold')

    # interface_ghostfrags = [ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()]
    # interface_monofrags2 = [residues2[idx] for idx in passing_surf_indices[passing_overlap_indices].tolist()]
    # passing_z_values = all_fragment_overlap[passing_overlap_indices]
    # match_scores = match_score_from_z_value(all_fragment_overlap[passing_overlap_indices])

    return list(zip([ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlaps_indices].tolist()],
                    [residues2[idx] for idx in passing_frag_indices[passing_overlaps_indices].tolist()],
                    all_fragment_match[passing_overlaps_indices].tolist()))


def get_matching_fragment_pairs_info(ghostfrag_frag_pairs: list[tuple[GhostFragment, Fragment, float]]) -> \
        list[dict[str, float | str]]:
    """From a ghost fragment/surface fragment pair and corresponding match score, return the pertinent interface
    information

    Args:
        ghostfrag_frag_pairs: Observed ghost and surface fragment overlaps and their match score
    Returns:
        The formatted fragment information for each pair
    """
    fragment_matches = []
    for interface_ghost_frag, interface_surf_frag, match_score in ghostfrag_frag_pairs:
        _, surffrag_resnum1 = interface_ghost_frag.get_aligned_chain_and_residue()  # surffrag_ch1,
        _, surffrag_resnum2 = interface_surf_frag.get_central_res_tup()  # surffrag_ch2,
        # Todo
        #  surf_frag_central_res_num1 = interface_ghost_residue.number
        #  surf_frag_central_res_num2 = interface_surf_residue.number
        fragment_matches.append(dict(zip(('mapped', 'paired', 'match', 'cluster'),
                                     (surffrag_resnum1, surffrag_resnum2,  match_score,
                                      '%d_%d_%d' % interface_ghost_frag.ijk))))
    logger.debug('Fragments for Entity1 found at residues: %s' % [fragment['mapped'] for fragment in fragment_matches])
    logger.debug('Fragments for Entity2 found at residues: %s' % [fragment['paired'] for fragment in fragment_matches])

    return fragment_matches


# def calculate_interface_score(interface_pdb, write=False, out_path=os.getcwd()):
#     """Takes as input a single PDB with two chains and scores the interface using fragment decoration"""
#     interface_name = interface_pdb.name
#
#     entity1 = Model.from_atoms(interface_pdb.chain(interface_pdb.chain_ids[0]).atoms)
#     entity1.update_attributes_from_pdb(interface_pdb)
#     entity2 = Model.from_atoms(interface_pdb.chain(interface_pdb.chain_ids[-1]).atoms)
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
    pdb1_cb_coords = pdb1.cb_coords
    pdb1_cb_indices = pdb1.cb_indices
    pdb2_cb_coords = pdb2.cb_coords
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


def parse_cryst_record(cryst_record) -> tuple[list[float], str]:
    """Get the unit cell length, height, width, and angles alpha, beta, gamma and the space group
    Args:
        cryst_record: The CRYST1 record in a .pdb file
    """
    try:
        cryst, a, b, c, ang_a, ang_b, ang_c, *space_group = cryst_record.split()
        # a = float(cryst1_record[6:15])
        # b = float(cryst1_record[15:24])
        # c = float(cryst1_record[24:33])
        # ang_a = float(cryst1_record[33:40])
        # ang_b = float(cryst1_record[40:47])
        # ang_c = float(cryst1_record[47:54])
    except ValueError:  # split and unpacking went wrong
        a = b = c = ang_a = ang_b = ang_c = 0

    return list(map(float, [a, b, c, ang_a, ang_b, ang_c])), cryst_record[55:66].strip()


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

    # Todo Modernize
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
        #     file_handle.write('%s\n' % self.return_atom_record(**kwargs))
        #     return
        #
        # with open(out_path, 'w') as f:
        #     if header:
        #         if isinstance(header, str):
        #             f.write(header)
        #         # if isinstance(header, Iterable):
        #
        #     if increment_chains:
        #         available_chain_ids = self.chain_id_generator()
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


class Model(Structure, ContainsChainsMixin):
    """The base object for Structure file (.pdb/.cif) parsing and manipulation, particularly containing multiple Chain
    or Entity instances

    Can initialize by passing a file, or passing Atom/Residue/Chain/Entity instances

    If you have multiple Models or States, use the MultiModel class to store and retrieve that data

    Args:
        metadata
    Keyword Args:
        pose_format: bool = False - Whether to renumber the Model to use Residue numbering from 1 to N
        rename_chains: bool = False - Whether to name each chain an incrementally new Alphabetical character
        log
        name
    """
    api_entry: dict[str, dict[Any] | float] | None
    biological_assembly: str | int | None
    chain_ids: list[str]
    chains: list[Chain] | Structures | bool | None
    # cryst: dict[str, str | tuple[float]] | None
    cryst_record: str | None
    # dbref: dict[str, dict[str, str]]
    design: bool
    entities: list[Entity] | Structures | bool | None
    entity_info: dict[str, dict[dict | list | str]] | dict
    # file_path: AnyStr | None
    header: list
    # multimodel: bool
    original_chain_ids: list[str]
    resolution: float | None
    api_db: wrapapi.APIDatabase
    _reference_sequence: dict[str, str]
    # space_group: str | None
    # uc_dimensions: list[float] | None

    def __init__(self, model: Structure = None,
                 biological_assembly: str | int = None,
                 chains: list[Chain] | Structures | bool = None, entities: list[Entity] | Structures | bool = None,
                 cryst_record: str = None, design: bool = False,
                 # dbref: dict[str, dict[str, str]] = None,
                 entity_info: dict[str, dict[dict | list | str]] = None,
                 # multimodel: bool = False,
                 resolution: float = None,
                 # api_db: wrapapi.APIDatabase = None,
                 reference_sequence: dict[str, str] = None,
                 # metadata: Model = None,
                 **kwargs):
        # kwargs passed to Structure
        #          atoms: list[Atom] | Atoms = None, residues: list[Residue] | Residues = None, name: str = None,
        #          residue_indices: list[int] = None,
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        # Unused args now
        #        cryst: dict[str, str | tuple[float]] = None, space_group: str = None, uc_dimensions: list[float] = None
        if model:
            if isinstance(model, Structure):
                super().__init__(**model.get_structure_containers(), **kwargs)
            else:
                raise NotImplementedError(f'Setting {type(self).__name__} with a {type(model).__name__} isn\'t '
                                          f'supported')
        else:
            super().__init__(**kwargs)
            self.api_entry = None
            # {'entity': {1: {'A', 'B'}, ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
            #  'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
            self.biological_assembly = biological_assembly
            # self.chain_ids = []  # unique chain IDs
            self.chains = []
            # self.cryst = cryst
            # {space: space_group, a_b_c: (a, b, c), ang_a_b_c: (ang_a, _b, _c)}
            self.cryst_record = cryst_record
            # self.dbref = dbref if dbref else {}  # {'chain': {'db: 'UNP', 'accession': P12345}, ...}
            self.design = design  # assumes not a design unless explicitly found to be a design
            self.entities = []
            self.entity_info = entity_info if entity_info is not None else {}
            # [{'chains': [Chain objs], 'seq': 'GHIPLF...', 'name': 'A'}, ...]
            # ^ ZERO-indexed for recap project!!!
            # self.file_path = file_path
            self.header = []
            # self.multimodel = multimodel
            # self.original_chain_ids = []  # [original_chain_id1, id2, ...]
            self.resolution = resolution
            self._reference_sequence = reference_sequence if reference_sequence else {}
            # ^ SEQRES or PDB API entries. key is chainID, value is 'AGHKLAIDL'
            # self.space_group = space_group
            # Todo standardize path with some state variable?
            # self.api_db = api_db if api_db else wrapapi.api_database_factory()

            # self.uc_dimensions = uc_dimensions
            self.structure_containers.extend(['chains', 'entities'])

            # only pass arguments if they are not None
            if entities is not None:  # if no entities are requested a False argument could be provided
                kwargs['entities'] = entities
            if chains is not None:  # if no chains are requested a False argument could be provided
                kwargs['chains'] = chains
            # finish processing the model
            self._process_model(**kwargs)
            # below was depreciated in favor of single call above using kwargs unpacking
            # if self.residues:  # we should have residues if Structure init, otherwise we have None
            #     if entities is not None:  # if no entities are requested a False argument could be provided
            #         kwargs['entities'] = entities
            #     if chains is not None:  # if no chains are requested a False argument could be provided
            #         kwargs['chains'] = chains
            #     self._process_model(**kwargs)
            # elif chains:  # pass the chains which should be a Structure type and designate whether entities should be made
            #     self._process_model(chains=chains, entities=entities, **kwargs)
            # elif entities:  # pass the entities which should be a Structure type and designate whether chains should be made
            #     self._process_model(entities=entities, chains=chains, **kwargs)
            # else:
            #     raise ValueError(f'{type(self).__name__} couldn\'t be initialized as there is no specified Structure type')

            # if metadata and isinstance(metadata, PDB):
            #     self.copy_metadata(metadata)

    @classmethod
    def from_chains(cls, chains: list[Chain] | Structures, **kwargs):
        """Create a new Model from a container of Chain objects. Automatically renames all chains"""
        return cls(chains=chains, rename_chains=True, **kwargs)

    @classmethod
    def from_entities(cls, entities: list[Entity] | Structures, **kwargs):
        """Create a new Model from a container of Entity objects"""
        return cls(entities=entities, chains=False, **kwargs)

    @classmethod
    def from_model(cls, model, **kwargs):
        """Initialize from an existing Model"""
        return cls(model=model, **kwargs)

    @property
    def chain_breaks(self) -> list[int]:
        return [structure.c_terminal_residue.number for structure in self.chains]

    @property
    def entity_breaks(self) -> list[int]:
        return [structure.c_terminal_residue.number for structure in self.entities]

    @property
    def atom_indices_per_chain(self) -> list[list[int]]:  # UNUSED
        """Return the atom indices for each Chain in the Model"""
        return [structure.atom_indices for structure in self.chains]

    @property
    def atom_indices_per_entity(self) -> list[list[int]]:
        """Return the atom indices for each Entity in the Model"""
        return [structure.atom_indices for structure in self.entities]

    @property
    def residue_indices_per_chain(self) -> list[list[int]]:  # UNUSED
        return [structure.residue_indices for structure in self.chains]

    @property
    def residue_indices_per_entity(self) -> list[list[int]]:
        return [structure.residue_indices for structure in self.entities]

    @property
    def number_of_atoms_per_chain(self) -> list[int]:  # UNUSED
        return [structure.number_of_atoms for structure in self.chains]

    @property
    def number_of_atoms_per_entity(self) -> list[int]:  # UNUSED
        return [structure.number_of_atoms for structure in self.entities]

    @property
    def number_of_residues_per_chain(self) -> list[int]:  # UNUSED
        return [structure.number_of_residues for structure in self.chains]

    @property
    def number_of_residues_per_entity(self) -> list[int]:  # UNUSED
        return [structure.number_of_residues for structure in self.entities]

    def format_header(self, **kwargs) -> str:
        """Return the BIOMT and the SEQRES records based on the Model

        Returns:
            The header with PDB file formatting
        """
        return self.format_biomt(**kwargs) + self.format_seqres(**kwargs)

    @property
    def number_of_entities(self) -> int:
        """Return the number of Entity instances in the Structure"""
        return len(self.entities)

    def is_multimodel(self) -> bool:
        """Return whether the parsed file contains multiple models, aka a 'multimodel'"""
        return self.chain_ids == self.original_chain_ids

    @property
    def reference_sequence(self) -> str:  # Todo this needs to be reconciled with Pose and Entity and Chain
        """Return the entire Model sequence, constituting all Residues, not just structurally modelled ones

        Returns:
            The sequence according to each of the Entity references
        """
        return ''.join(self._reference_sequence.values())

    def _process_model(self, pose_format: bool = False, chains: bool | list[Chain] | Structures = True,
                       rename_chains: bool = False, entities: bool | list[Entity] | Structures = True,
                       **kwargs):
        #               atoms: Union[Atoms, List[Atom]] = None, residues: Union[Residues, List[Residue]] = None,
        #               coords: Union[List[List], np.ndarray, Coords] = None,
        #               reference_sequence=None
        """Process various types of Structure containers to update the Model with the corresponding information

        Args:
            pose_format: Whether to initialize Structure with residue numbering from 1 until the end
            chains:
            rename_chains: Whether to name each chain an incrementally new Alphabetical character
            entities:
        """
        # add lists together, only one is populated from class construction
        structures = (chains if isinstance(chains, (list, Structures)) else []) + \
                     (entities if isinstance(entities, (list, Structures)) else [])
        if structures:  # create from existing
            atoms, residues, coords = [], [], []
            for structure in structures:
                atoms.extend(structure.atoms)
                residues.extend(structure.residues)
                coords.append(structure.coords)
            self._assign_residues(residues, atoms=atoms, coords=coords)

        if chains:
            if isinstance(chains, (list, Structures)):  # create the instance from existing chains
                self.chains = copy(chains)  # copy the passed chains
                self._copy_structure_containers()  # copy each Chain in chains
                # Reindex all residue and atom indices
                self.chains[0].reset_state()
                self.chains[0]._start_indices(at=0, dtype='atom')
                self.chains[0]._start_indices(at=0, dtype='residue')
                for prior_idx, chain in enumerate(self.chains[1:]):
                    chain.reset_state()
                    chain._start_indices(at=self.chains[prior_idx].atom_indices[-1] + 1, dtype='atom')
                    chain._start_indices(at=self.chains[prior_idx].residue_indices[-1] + 1, dtype='residue')

                # set the parent attribute for all containers
                self._update_structure_container_attributes(_parent=self)
                # By using extend, we set self.original_chain_ids too
                self.chain_ids.extend([chain.chain_id for chain in self.chains])
            else:  # Create Chain instances from Residues
                self._create_chains()
                # Todo this isn't super accurate
                #  Ideally we get correct solution from PDB or UniProt API.
                #  If no _reference_sequence passed then this will be nothing, so that isn't great.
                #  It should be at least the structure sequence
                self._reference_sequence = dict(zip(self.chain_ids, self._reference_sequence.values()))

            if rename_chains:
                self.rename_chains()
            self.log.debug(f'Original chain_ids={",".join(self.original_chain_ids)} | '
                           f'Loaded chain_ids={",".join(self.chain_ids)}')

        if entities:
            if isinstance(entities, (list, Structures)):  # create the instance from existing entities
                self.entities = copy(entities)  # copy the passed entities list
                self._copy_structure_containers()  # copy each Entity in entities
                # Reindex all residue and atom indices
                self.entities[0].reset_state()
                self.entities[0]._start_indices(at=0, dtype='atom')
                self.entities[0]._start_indices(at=0, dtype='residue')
                for prior_idx, entity in enumerate(self.entities[1:]):
                    entity.reset_state()
                    entity._start_indices(at=self.entities[prior_idx].atom_indices[-1] + 1, dtype='atom')
                    entity._start_indices(at=self.entities[prior_idx].residue_indices[-1] + 1, dtype='residue')

                # set the parent attribute for all containers
                self._update_structure_container_attributes(_parent=self)
                if rename_chains:  # set each successive Entity to have an incrementally higher chain id
                    available_chain_ids = self.chain_id_generator()
                    for idx, entity in enumerate(self.entities):
                        entity.chain_id = next(available_chain_ids)
                        self.log.debug(f'Entity {entity.name} new chain identifier {entity.chain_id}')

                # update chains to entities after everything is set
                self.chains = self.entities
                # self.chain_ids = [chain.name for chain in self.chains]
            else:  # create Entities from Chain.Residues
                self._create_entities(**kwargs)

            if not self.chain_ids:  # set according to self.entities
                self.chain_ids.extend([entity.chain_id for entity in self.entities])

        if pose_format:
            self.renumber_structure()

    # def copy_metadata(self, other):  # Todo, rework for all Structure
    #     temp_metadata = \
    #         {'api_entry': other.__dict__['api_entry'],
    #          'cryst_record': other.__dict__['cryst_record'],
    #          # 'cryst': other.__dict__['cryst'],
    #          'design': other.__dict__['design'],
    #          'entity_info': other.__dict__['entity_info'],
    #          '_name': other.__dict__['_name'],
    #          # 'space_group': other.__dict__['space_group'],
    #          # '_uc_dimensions': other.__dict__['_uc_dimensions'],
    #          'header': other.__dict__['header'],
    #          # 'reference_aa': other.__dict__['reference_aa'],
    #          'resolution': other.__dict__['resolution'],
    #          'rotation_d': other.__dict__['rotation_d'],
    #          'max_symmetry': other.__dict__['max_symmetry'],
    #          'dihedral_chain': other.__dict__['dihedral_chain'],
    #          }
    #     # temp_metadata = copy(other.__dict__)
    #     # temp_metadata.pop('atoms')
    #     # temp_metadata.pop('residues')
    #     # temp_metadata.pop('secondary_structure')
    #     # temp_metadata.pop('number_of_atoms')
    #     # temp_metadata.pop('number_of_residues')
    #     self.__dict__.update(temp_metadata)

    # def update_attributes_from_pdb(self, pdb):  # Todo copy full attribute dict without selected elements
    #     # self.atoms = pdb.atoms
    #     self.resolution = pdb.resolution
    #     self.cryst_record = pdb.cryst_record
    #     # self.cryst = pdb.cryst
    #     self.dbref = pdb.dbref
    #     self.design = pdb.design
    #     self.header = pdb.header
    #     self.reference_sequence = pdb._reference_sequence
    #     # self.atom_sequences = pdb.atom_sequences
    #     self.file_path = pdb.file_path
    #     # self.chain_ids = pdb.chain_ids
    #     self.entity_info = pdb.entity_info
    #     self.name = pdb.name
    #     self.secondary_structure = pdb.secondary_structure
    #     # self.cb_coords = pdb.cb_coords
    #     # self.bb_coords = pdb.bb_coords

    def format_seqres(self, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Keyword Args:
            **kwargs
        Returns:
            The PDB formatted SEQRES record
        """
        if self._reference_sequence:
            formated_reference_sequence = \
                {chain: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX') for aa in sequence)))
                 for chain, sequence in self._reference_sequence.items()}
            chain_lengths = {chain: len(sequence) for chain, sequence in self._reference_sequence.items()}
            return '%s\n' \
                % '\n'.join('SEQRES{:4d} {:1s}{:5d}  %s         '.format(line_number, chain, chain_lengths[chain])
                            % sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]
                            for chain, sequence in formated_reference_sequence.items()
                            for line_number in range(1, 1 + ceil(len(sequence)/seq_res_len)))
        else:
            return ''

    def write(self, **kwargs) -> AnyStr | None:  # Todo Depreciate. require Pose or self.cryst_record -> Structure?
        """Write Atoms to a file specified by out_path or with a passed file_handle

        Keyword Args
            header: None | str - A string that is desired at the top of the .pdb file
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'Model is writing')
        if not kwargs.get('header') and self.cryst_record:
            kwargs['header'] = self.cryst_record

        return super().write(**kwargs)

    def orient(self, symmetry: str = None, log: AnyStr = None):  # similar function in Entity
        """Orient a symmetric PDB at the origin with its symmetry axis canonically set on axes defined by symmetry
        file. Automatically produces files in PDB numbering for proper orient execution

        Args:
            symmetry: What is the symmetry of the specified PDB?
            log: If there is a log specific for orienting
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
            self.log.error(f'{self.orient.__name__}: Symmetry {symmetry} is not a valid symmetry. '
                           f'Please try one of: {", ".join(valid_symmetries)}')
            return

        if not log:
            log = self.log

        if self.file_path:
            file_name = os.path.basename(self.file_path)
        else:
            file_name = f'{self.name}.pdb'
        # Todo change output to logger with potential for file and stdout

        number_of_subunits = self.number_of_chains
        multicomponent = False
        if symmetry == 'C1':
            log.debug('C1 symmetry doesn\'t have a cannonical orientation')
            self.translate(-self.center_of_mass)
            return
        elif number_of_subunits > 1:
            if number_of_subunits != subunit_number:
                if number_of_subunits in multicomponent_valid_subunit_number.get(symmetry):
                    multicomponent = True
                else:
                    raise ValueError(f'{file_name} could not be oriented: It has {number_of_subunits} subunits '
                                     f'while a multiple of {subunit_number} are expected for {symmetry} symmetry')
        else:
            raise ValueError(f'{self.name}: Cannot orient a Structure with only a single chain. No symmetry present!')

        orient_input = Path(PUtils.orient_dir, 'input.pdb')
        orient_output = Path(PUtils.orient_dir, 'output.pdb')

        def clean_orient_input_output():
            orient_input.unlink(missing_ok=True)
            orient_output.unlink(missing_ok=True)

        clean_orient_input_output()
        # Have to change residue numbering to PDB numbering
        if multicomponent:
            self.entities[0].write_oligomer(out_path=str(orient_input), pdb_number=True)
        else:
            self.write(out_path=orient_input, pdb_number=True)

        # Todo superposition3d -> quaternion
        p = subprocess.Popen([PUtils.orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=PUtils.orient_dir)
        in_symm_file = os.path.join(PUtils.orient_dir, 'symm_files', symmetry)
        stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
        log.info(file_name + stdout.decode()[28:])
        log.info(stderr.decode()) if stderr else None
        if not orient_output.exists() or orient_output.stat().st_size == 0:
            log_file = getattr(log.handlers[0], 'baseFilename', None)
            log_message = f'. Check {log_file} for more information' if log_file else ''
            raise RuntimeError(f'orient_oligomer could not orient {file_name}{log_message}')

        oriented_pdb = Model.from_file(str(orient_output), name=self.name, entities=False, log=log)
        orient_fixed_struct = oriented_pdb.chains[0]
        if multicomponent:
            moving_struct = self.entities[0]
        else:
            moving_struct = self.chains[0]

        orient_fixed_seq = orient_fixed_struct.sequence
        moving_seq = moving_struct.sequence

        if orient_fixed_struct.number_of_residues == moving_struct.number_of_residues and orient_fixed_seq == moving_seq:
            # do an apples to apples comparison
            # length alone is inaccurate if chain is missing first residue and self is missing it's last...
            _, rot, tx = superposition3d(orient_fixed_struct.cb_coords, moving_struct.cb_coords)
        else:  # do an alignment, get selective indices, then follow with superposition
            self.log.warning(f'{moving_struct.name} and {orient_fixed_struct.name} require alignment to '
                             f'{self.orient.__name__}')
            fixed_indices, moving_indices = get_equivalent_indices(orient_fixed_seq, moving_seq)
            _, rot, tx = superposition3d(orient_fixed_struct.cb_coords[fixed_indices],
                                         moving_struct.cb_coords[moving_indices])

        self.transform(rotation=rot, translation=tx)
        clean_orient_input_output()

    def mutate_residue(self, residue: Residue = None, number: int = None, to: str = 'ALA', **kwargs):  # Todo Structures
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Args:
            residue: A Residue object to mutate
            number: A Residue number to select the Residue of interest with
            to: The type of amino acid to mutate to
        Keyword Args:
            pdb=False (bool): Whether to pull the Residue by PDB number
        """
        delete_indices = super().mutate_residue(residue=residue, number=number, to=to, **kwargs)
        if not delete_indices:  # there are no indices
            return
        delete_length = len(delete_indices)
        # remove these indices from the Structure atom_indices (If other structures, must update their atom_indices!)
        for structure_type in self.structure_containers:
            for structure in getattr(self, structure_type):  # iterate over Structures in each structure_container
                try:
                    atom_delete_index = structure.atom_indices.index(delete_indices[0])
                    for _ in iter(delete_indices):
                        structure.atom_indices.pop(atom_delete_index)
                    structure._offset_indices(start_at=atom_delete_index, offset=-delete_length)
                except (ValueError, IndexError):  # this should happen if the Atom is not in the Structure of interest
                    continue

    def insert_residue_type(self, residue_type: str, at: int = None, chain: str = None):  # Todo Structures
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed!

        Args:
            residue_type: Either the 1 or 3 letter amino acid code for the residue in question
            at: The pose numbered location which a new Residue should be inserted into the Structure
            chain: The chain identifier to associate the new Residue with
        """
        new_residue = super().insert_residue_type(residue_type, at=at, chain=chain)
        # must update other Structures indices
        residue_index = at - 1  # since at is one-indexed integer
        # for structures in [self.chains, self.entities]:
        for structure_type in self.structure_containers:
            structures = getattr(self, structure_type)
            idx = 0
            for idx, structure in enumerate(structures):  # iterate over Structures in each structure_container
                try:  # update each Structures _residue_ and _atom_indices with additional indices
                    structure._insert_indices(at=structure.residue_indices.index(residue_index),
                                              new_indices=[residue_index], dtype='residue')
                    structure._insert_indices(at=structure.atom_indices.index(new_residue.start_index),
                                              new_indices=new_residue.atom_indices, dtype='atom')
                    break  # move to the next container to update the indices by a set increment
                except (ValueError, IndexError):  # this should happen if the Atom is not in the Structure of interest
                    # edge case where the index is being appended to the c-terminus
                    if residue_index - 1 == structure.residue_indices[-1] and new_residue.chain == structure.chain_id:
                        structure._insert_indices(at=structure.number_of_residues, new_indices=[residue_index],
                                                  dtype='residue')
                        structure._insert_indices(at=structure.number_of_atoms, new_indices=new_residue.atom_indices,
                                                  dtype='atom')
                        break  # must move to the next container to update the indices by a set increment
            # for each subsequent structure in the structure container, update the indices with the last indices from
            # the prior structure
            for prior_idx, structure in enumerate(structures[idx + 1:], idx):
                structure._start_indices(at=structures[prior_idx].atom_indices[-1] + 1, dtype='atom')
                structure._start_indices(at=structures[prior_idx].residue_indices[-1] + 1, dtype='residue')

    def delete_residue(self, chain_id: str, residue_number: int):  # Todo Move to Structure
        self.log.critical(f'{self.delete_residue.__name__} This function requires testing')  # TODO TEST
        # start = len(self.atoms)
        # self.log.debug(start)
        # residue = self.get_residue(chain, residue_number)
        # residue.delete_atoms()  # deletes Atoms from Residue. unneccessary?

        delete_indices = self.chain(chain_id).residue(residue_number).atom_indices
        # Atoms() handles all Atom instances for the object
        self._atoms.delete(delete_indices)
        # self.delete_atoms(residue.atoms)  # deletes Atoms from PDB
        # chain._residues.remove(residue)  # deletes Residue from Chain
        # self._residues.remove(residue)  # deletes Residue from PDB
        self.renumber_structure()
        self._residues.reindex_atoms()
        # remove these indices from all Structure atom_indices including structure_containers
        # Todo, turn this loop into Structure routine and implement for self, and structure_containers
        atom_delete_index = self._atom_indices.index(delete_indices[0])
        for iteration in range(len(delete_indices)):
            self._atom_indices.pop(atom_delete_index)
        # for structures in [self.chains, self.entities]:
        for structure_type in self.structure_containers:
            for structure in getattr(self, structure_type):  # iterate over Structures in each structure_container
                try:
                    atom_delete_index = structure.atom_indices.index(delete_indices[0])
                    for iteration in range(len(delete_indices)):
                        structure.atom_indices.pop(atom_delete_index)
                except ValueError:
                    continue
        # self.log.debug('Deleted: %d atoms' % (start - len(self.atoms)))

    def retrieve_pdb_info_from_api(self):
        """Query the PDB API for information on the PDB code found as the PDB object .name attribute

        Makes 1 + num_of_entities calls to the PDB API. If file is assembly, makes one more

        Sets:
            self.api_entry (dict[str, dict[Any] | float]):
                {'assembly': [['A', 'B'], ...],
                 'entity': {'EntityID': {'chains': ['A', 'B', ...],
                                         'dbref': {'accession': 'Q96DC8', 'db': 'UNP'}
                                         'reference_sequence': 'MSLEHHHHHH...'},
                                         ...},
                 'res': resolution,
                 'struct': {'space': space_group, 'a_b_c': (a, b, c),
                            'ang_a_b_c': (ang_a, ang_b, ang_c)}
                }
        """
        if self.api_entry is not None:  # we already tried solving this
            return
        # if self.api_db:
        try:
            # retrieve_api_info = self.api_db.pdb.retrieve_data
            retrieve_api_info = wrapapi.api_database_factory().pdb.retrieve_data
        except AttributeError:
            retrieve_api_info = query_pdb_by

        # if self.name:  # try to solve API details from name
        parsed_name = self.name
        splitter = ['_', '-']  # [entity, assembly]
        idx = -1
        extra = None
        while len(parsed_name) != 4:
            try:  # to parse the name using standard PDB API entry ID's
                idx += 1
                parsed_name, *extra = parsed_name.split(splitter[idx])
            except IndexError:  # idx > len(splitter)
                # we can't find entry in parsed_name from splitting typical PDB formatted strings
                # it may be completely incorrect or something unplanned.
                bad_format_msg = \
                    f'PDB entry "{self.name}" is not of the required format and wasn\'t queried from the PDB API'
                self.log.debug(bad_format_msg)
                break  # the while loop and handle

        if idx < len(splitter):  # len(parsed_name) == 4 at some point
            # query_args = dict(entry=parsed_name)
            # # self.api_entry = _get_entry_info(parsed_name)
            if self.biological_assembly:
                # query_args.update(assembly_integer=self.assembly)
                # # self.api_entry.update(_get_assembly_info(self.name))
                self.api_entry = retrieve_api_info(entry=parsed_name)
                self.api_entry['assembly'] = \
                    retrieve_api_info(entry=parsed_name, assembly_integer=self.biological_assembly)
                # ^ returns [['A', 'A', 'A', ...], ...]
            elif extra:  # extra not None or []. use of elif means we couldn't have 1ABC_1.pdb2
                # try to parse any found extra to an integer denoting entity or assembly ID
                integer, *non_sense = extra
                if integer.isdigit() and not non_sense:
                    integer = int(integer)
                    if idx == 0:  # entity integer, such as 1ABC_1.pdb
                        # query_args.update(entity_integer=integer)
                        self.api_entry = dict(entity=retrieve_api_info(entry=parsed_name, entity_integer=integer))
                        # retrieve_api_info returns
                        # {'EntityID': {'chains': ['A', 'B', ...],
                        #               'dbref': {'accession': 'Q96DC8', 'db': 'UNP'}
                        #               'reference_sequence': 'MSLEHHHHHH...'},
                        #  ...}
                    else:  # get entry alone. This is an assembly or unknown conjugation. Either way we need entry info
                        self.api_entry = retrieve_api_info(entry=parsed_name)

                        if idx == 1:  # assembly integer, such as 1ABC-1.pdb
                            # query_args.update(assembly_integer=integer)
                            self.api_entry['assembly'] = \
                                retrieve_api_info(entry=parsed_name, assembly_integer=integer)
                else:  # this isn't an integer or there are extra characters
                    # It's likely they are extra characters that won't be of help. Try to collect anyway
                    # self.log.debug(bad_format_msg)
                    self.api_entry = {}
                    self.log.debug('Found extra file name information that can\'t be coerced to match the PDB API')
                    # self.api_entry = retrieve_api_info(entry=parsed_name)
            elif extra is None:  # we didn't get extra as it was correct length to begin with, just query entry
                self.api_entry = retrieve_api_info(entry=parsed_name)
            else:
                raise RuntimeError('This logic was not expected and shouldn\'t be allowed to persist:'
                                   f'self.name={self.name}, parse_name={parsed_name}, extra={extra}, idx={idx}')

            if self.api_entry:
                self.log.debug(f'Found PDB API information: '
                               f'{", ".join(f"{k}={v}" for k, v in self.api_entry.items())}')
                # set the identified name
                self.name = self.name.lower()
        else:
            self.api_entry = {}
        #     self.log.debug('No name was found for this Model. PDB API won\'t be searched')

    def entity(self, entity_id: str) -> Entity | None:
        """Retrieve an Entity by name from the PDB object

        Args:
            entity_id: The name of the Entity to query
        Returns:
            The Entity if one was found
        """
        for entity in self.entities:
            if entity_id == entity.name:
                return entity

    def _create_entities(self, entity_names: Sequence = None, query_by_sequence: bool = True, **kwargs):
        """Create all Entities in the PDB object searching for the required information if it was not found during
        file parsing. First, search the PDB API using an attached PDB entry_id, dependent on the presence of a
        biological assembly file and/or multimodel file. Finally, initialize them from the Residues in each Chain
        instance using a specified threshold of sequence homology

        Args:
            entity_names: Names explicitly passed for the Entity instances. Length must equal number of entities.
                Names will take precedence over query_by_sequence if passed
            query_by_sequence: Whether the PDB API should be queried for an Entity name by matching sequence. Only used
                if entity_names not provided
        """
        if not self.entity_info:  # we didn't get from the file (probaly not PDB), so we have to try and piece together
            # The file is either from a program that has modified an original PDB file, or may be some sort of PDB
            # assembly. If it is a PDB assembly, the only way to know is that the file would have a final numeric suffix
            # after the .pdb extension (.pdb1). If not, it may be an assembly file from another source, in which case we
            # have to solve by using the atomic info
            self.retrieve_pdb_info_from_api()  # First try to set self.api_entry
            if self.api_entry:
                if self.biological_assembly:
                    # As API returns information on the asu, assembly may be different. We got API info for assembly, so
                    # we try to reconcile
                    multimodel = self.is_multimodel()
                    for entity_name, data in self.api_entry.get('entity', {}).items():
                        chains = data['chains']
                        for cluster_chains in self.api_entry.get('assembly', []):
                            if not set(cluster_chains).difference(chains):  # nothing missing, correct cluster
                                self.entity_info[entity_name] = data
                                if multimodel:  # ensure the renaming of chains is handled correctly
                                    self.entity_info[entity_name].update(
                                        {'chains': [new_chn for new_chn, old_chn in zip(self.chain_ids,
                                                                                        self.original_chain_ids)
                                                    if old_chn in chains]})
                                # else:  # chain names should be the same as assembly API if file is sourced from PDB
                                #     self.entity_info[entity_name] = data
                                break  # we satisfied this cluster, move on
                        else:  # if we didn't satisfy a cluster, report and move to the next
                            self.log.error('Unable to find the chains corresponding from entity (%s) to assembly (%s)'
                                           % (entity_name, self.api_entry.get('assembly', {})))
                else:
                    for entity_name, data in self.api_entry.get('entity', {}).items():
                        self.entity_info[entity_name] = data
                # Todo this was commented out because nucleotides can't be parsed. This issue still needs solving
                # Check to see that the entity_info is in line with the number of chains already parsed
                # found_entity_chains = [chain for info in self.entity_info for chain in info.get('chains', [])]
                # if len(self.chain_ids) != len(found_entity_chains):
                #     self._get_entity_info_from_atoms(**kwargs)
            else:  # Still nothing, the API didn't work for self.name. Solve by atom information
                self._get_entity_info_from_atoms(**kwargs)
                if query_by_sequence and not entity_names:
                    for entity_name, data in list(self.entity_info.items()):  # Make a new list to prevent pop issues
                        pdb_api_name = retrieve_entity_id_by_sequence(data['sequence'])
                        if pdb_api_name:
                            pdb_api_name = pdb_api_name.lower()
                            self.log.info(f'Entity {entity_name} now named "{pdb_api_name}", as found by PDB API '
                                          f'sequence search')
                            self.entity_info[pdb_api_name] = self.entity_info.pop(entity_name)
        if entity_names:
            # if self.api_db:
            try:
                # retrieve_api_info = self.api_db.pdb.retrieve_data
                retrieve_api_info = wrapapi.api_database_factory().pdb.retrieve_data
            except AttributeError:
                retrieve_api_info = query_pdb_by

            api_entry_entity = self.api_entry.get('entity', {})
            if not api_entry_entity:
                self.api_entry['entity'] = {}

            for idx, entity_name in enumerate(list(self.entity_info.keys())):  # Make a new list to prevent pop issues
                try:
                    new_entity_name = entity_names[idx]
                except IndexError:
                    raise IndexError(f'The number of indices in entity_names ({len(entity_names)}) must equal the '
                                     f'number of entities ({len(self.entity_info)})')

                # Get any info already solved using the old name
                self.entity_info[new_entity_name] = self.entity_info.pop(entity_name)
                entity_api_info = retrieve_api_info(entity_id=new_entity_name)
                if entity_api_info and new_entity_name not in self.api_entry.get('entity', {}):
                    # Add the new info. If the new_entity_name is already present, we could expect that
                    # self.entity_info is already solved and new_entity_name probably == entity_name
                    self.api_entry['entity'].update(entity_api_info)
                    # Respect any found info in self.entity_info
                    if self.entity_info[new_entity_name].get('chains', {}):
                        # Remove the entity_api_info 'chains' indication and use the entity_info chains
                        entity_api_info[new_entity_name].pop('chains')
                    # Update the entity_api_info to the entity_info, preserving self.entity_info[new_entity_name] data
                    self.entity_info[new_entity_name].update(entity_api_info[new_entity_name])

                self.log.debug(f'Entity {entity_name} now named "{new_entity_name}", as supplied by entity_names')

        # For each Entity, get matching Chain instances
        for entity_name, data in self.entity_info.items():
            chains = [self.chain(chain) if isinstance(chain, str) else chain for chain in data.get('chains')]
            data['chains'] = [chain for chain in chains if chain]  # remove any missing chains
            # # get uniprot ID if the file is from the PDB and has a DBREF remark
            # try:
            #     accession = self.dbref.get(data['chains'][0].chain_id, None)
            # except IndexError:  # we didn't find any chains. It may be a nucleotide structure
            #     continue
            try:  # Todo clean here and the above entity vs chain len checks with nucleotide parsing
                chain_check_to_account_for_inability_to_parse_nucleotides = data['chains'][0]
            except IndexError:  # we didn't find any chains. It may be a nucleotide structure
                self.log.debug(f'Missing associated chains for the Entity {entity_name} with data '
                               f'{", ".join(f"{k}={v}" for k, v in data.items())}')
                continue
            #     raise DesignError('Missing Chain object for %s %s! entity_info=%s, assembly=%s and '
            #                       'api_entry=%s, original_chain_ids=%s'
            #                       % (self.name, self._create_entities.__name__, self.entity_info,
            #                       self.biological_assembly, self.api_entry, self.original_chain_ids))
            if 'dbref' not in data:
                data['dbref'] = {}
            # data['chains'] = [chain for chain in chains if chain]  # remove any missing chains
            #                                               generated from a PDB API sequence search v
            # if isinstance(entity_name, int):
            #     data['name'] = f'{self.name}_{data["name"]}'

            # ref_seq = data.get('reference_sequence')
            if 'reference_sequence' not in data:
                if 'sequence' in data:  # We set from Atom info
                    data['reference_sequence'] = data['sequence']
                else:  # We should try to set using the entity_name
                    data['reference_sequence'] = get_entity_reference_sequence(entity_id=entity_name)
            # data has attributes chains, dbref, and reference_sequence
            self.entities.append(Entity.from_chains(**data, name=entity_name, parent=self))

    def _get_entity_info_from_atoms(self, method: str = 'sequence', tolerance: float = 0.9, **kwargs):  # Todo define inside _create_entities?
        """Find all unique Entities in the input .pdb file. These are unique sequence objects

        Args:
            method: The method used to extract information. One of 'sequence' or 'structure'
            tolerance: The acceptable difference between chains to consider them the same Entity.
                Tuning this parameter is necessary if you have chains which should be considered different entities,
                but are fairly similar. Alternatively, the use of a structural match should be used.
                For example, when each chain in an ASU is structurally deviating, but they all share the same sequence
        Sets:
            self.entity_info
        """
        if tolerance > 1:
            raise ValueError(f'{self._get_entity_info_from_atoms.__name__} tolerance={tolerance}. Can\'t be > 1')
        entity_idx = 1
        # get rid of any information already acquired
        self.entity_info = {f'{self.name}_{entity_idx}':
                            dict(chains=[self.chains[0]], sequence=self.chains[0].sequence)}
        for chain in self.chains[1:]:
            self.log.debug(f'Searching for matching Entities for Chain {chain.name}')
            new_entity = True  # assume all chains are unique entities
            for entity_name, data in self.entity_info.items():
                # Todo implement structure check
                #  rmsd_threshold = 1.  # threshold needs testing
                #  try:
                #       rmsd, *_, = superposition3d()
                #  except ValueError:  # the chains are different lengths
                #      try to use the code identified in Entity to match lengths..., if not, continue
                #  if rmsd < rmsd_threshold:
                #      data['chains'].append(chain)
                #      new_entity = False  # The entity is not unique, do not add
                #      break
                # check if the sequence associated with the Chain is in entity_info
                sequence = data['sequence']
                if chain.sequence == sequence:
                    score = len(chain.sequence)
                else:
                    alignment = generate_alignment(chain.sequence, sequence, local=True)
                    # alignment = pairwise2.align.localxx(chain.sequence, sequence)
                    score = alignment[2]  # grab score value
                sequence_length = len(sequence)
                match_score = score / sequence_length  # could also use which ever sequence is greater
                length_proportion = abs(len(chain.sequence) - sequence_length) / sequence_length
                self.log.debug(f'Chain {chain.name} matches Entity {entity_name} with '
                               f'%0.2f identity and length difference of %0.2f' % (match_score, length_proportion))
                if match_score >= tolerance and length_proportion <= 1 - tolerance:
                    # if number of sequence matches is > tolerance, and the length difference < tolerance
                    # the current chain is the same as the Entity, add to chains, and move on to the next chain
                    data['chains'].append(chain)
                    new_entity = False  # The entity is not unique, do not add
                    break

            if new_entity:  # no existing entity matches, add new entity
                entity_idx += 1
                self.entity_info[f'{self.name}_{entity_idx}'] = dict(chains=[chain], sequence=chain.sequence)
        self.log.debug(f'Entity information was solved by {method} match')

    def entity_from_chain(self, chain_id: str) -> Entity | None:
        """Return the entity associated with a particular chain id"""
        for entity in self.entities:
            if chain_id == entity.chain_id:
                return entity
        return None

    # def entity_from_residue(self, residue_number: int) -> Union[Entity, None]:  # Todo ResidueSelectors/fragment query
    #     """Return the entity associated with a particular Residue number
    #
    #     Returns:
    #         (Union[Entity, None])
    #     """
    #     for entity in self.entities:
    #         if entity.get_residues(numbers=[residue_number]):
    #             return entity
    #     return
    #
    # def match_entity_by_struct(self, other_struct=None, entity=None, force_closest=False):
    #     """From another set of atoms, returns the first matching chain from the corresponding entity"""
    #     return  # TODO when entities are structure compatible

    def match_entity_by_seq(self, other_seq: str = None, force_closest: bool = True, tolerance: float = 0.7) \
            -> Entity | None:
        """From another sequence, returns the first matching chain from the corresponding Entity

        Args:
            other_seq: The sequence to query
            force_closest: Whether to force the search if a perfect match isn't identified
            tolerance: The acceptable difference between sequences to consider them the same Entity.
                Tuning this parameter is necessary if you have sequences which should be considered different entities,
                but are fairly similar
        Returns
            The matching Entity if one was found
        """
        for entity in self.entities:
            if other_seq == entity.sequence:
                return entity

        # we didn't find an ideal match
        if force_closest:
            entity_alignment_scores = {}
            for entity in self.entities:
                alignment = generate_alignment(other_seq, entity.sequence, local=True)
                entity_alignment_scores[entity] = alignment.score

            max_score, max_score_entity = 0, None
            for entity, score in entity_alignment_scores.items():
                normalized_score = score / len(entity.sequence)
                if normalized_score > max_score:
                    max_score = normalized_score  # alignment_score_d[entity]
                    max_score_entity = entity

            if max_score > tolerance:
                return max_score_entity

        return None

# All methods below come with no intention of working with Model, but contain useful code to generate axes and display
# them for symmetric systems. Adaptation to an Axis class could be helpful for visualization
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


class Models(Model):
    """Keep track of different variations of the same Model object such as altered coordinates (different decoy's or
    symmetric copies) [or mutated Residues]. In PDB parlance, this would be a multimodel, however could be multiple
    PDB files that share a common element.
    """
    _models_coords: Coords
    state_attributes: set[str] = Model.state_attributes | {'_models_coords'}

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

    @property
    def number_of_models(self) -> int:
        """The number of unique models that are found in the Models object"""
        return len(self.models)

    @property
    def models_coords(self) -> np.ndarray | None:
        """Return a concatenated view of the Coords from all models"""
        try:
            return self._models_coords.coords
        except AttributeError:
            return None

    @models_coords.setter
    def models_coords(self, coords: Coords | np.ndarray | list[list[float]]):
        if isinstance(coords, Coords):
            self._models_coords = coords
        else:
            self._models_coords = Coords(coords)

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, increment_chains: bool = False,
              **kwargs) -> AnyStr | None:
        """Write Model Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            increment_chains: Whether to write each Structure with a new chain name, otherwise write as a new Model
        Keyword Args
            header: None | str - A string that is desired at the top of the .pdb file
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'Models is writing')

        def models_write(handle):
            # self.models is populated
            if increment_chains:  # assembly requested, check on the mechanism of symmetric writing
                # we won't allow incremental chains when the Model is plain as the models are all the same and
                # therefore belong with the models label
                available_chain_ids = self.chain_id_generator()
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
            # self.write_header(file_handle, **kwargs)
            models_write(file_handle)
            return None
        else:
            with open(out_path, 'w') as outfile:
                self.write_header(outfile, **kwargs)
                models_write(outfile)
            return out_path

    def __getitem__(self, idx: int) -> Structure:
        return self.models[idx]


class SymmetricModel(Models):
    _assembly: Model
    _assembly_minimally_contacting: Model
    _assembly_tree: BinaryTree  # stores a sklearn tree for coordinate searching
    _asu_indices: slice  # list[int]
    _asu_model_idx: int
    _center_of_mass_symmetric_entities: list[np.ndarray]
    _center_of_mass_symmetric_models: np.ndarray
    _cryst_record: str
    _dimension: int
    _number_of_symmetry_mates: int
    _point_group_symmetry: str
    _oligomeric_model_indices: dict[Entity, list[int]]
    _sym_entry: SymEntry
    _symmetry: str
    _symmetric_coords_by_entity: list[np.ndarray]
    _symmetric_coords_split: list[np.ndarray]
    _symmetric_coords_split_by_entity: list[list[np.ndarray]]
    expand_matrices: np.ndarray | list[list[float]] | None
    expand_translations: np.ndarray | list[float] | None
    uc_dimensions: list[float] | None
    state_attributes: set[str] = Models.state_attributes | \
        {'_assembly', '_assembly_minimally_contacting', '_assembly_tree', '_asu_indices', '_asu_model_idx',
         '_center_of_mass_symmetric_entities', '_center_of_mass_symmetric_models',
         '_oligomeric_model_indices', '_symmetric_coords_by_entity', '_symmetric_coords_split',
         '_symmetric_coords_split_by_entity'}

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
        # Initialize symmetry first
        self.expand_matrices = None
        self.expand_translations = None
        self.uc_dimensions = None  # uc_dimensions
        self.set_symmetry(sym_entry=sym_entry, symmetry=symmetry, uc_dimensions=uc_dimensions,
                          expand_matrices=expand_matrices)
        super().__init__(**kwargs)

        if self.symmetry:  # True if symmetry keyword args were passed
            # Ensure that the symmetric system is set up properly which could require finding the ASU
            if self.number_of_entities != self.number_of_chains:  # ensure the structure is an asu
                self.log.debug('Setting Pose ASU to the ASU with the most contacting interface')
                self.set_contacting_asu()
            elif self.symmetric_coords is None:
                # We need to generate the symmetric coords
                self.log.debug('Setting symmetric coords')
                self.generate_symmetric_coords(surrounding_uc=surrounding_uc)  # default has surrounding_uc=True
            # if generate_symmetry_mates:  # always set to False before. commenting out
            #     self.generate_assembly_symmetry_models(**kwargs)

    @classmethod
    def from_assembly(cls, assembly: list[Structure], sym_entry: SymEntry | int = None, symmetry: str = None, **kwargs):
        """Initialize from a symmetric assembly"""
        if symmetry is None and sym_entry is None:
            raise ValueError(f'Can\'t initialize {type(cls).__name__} without symmetry! Pass symmetry or '
                             f'sym_entry to constructor {cls.from_assembly.__name__}')
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
            if not uc_dimensions and not symmetry:  # only if user didn't provide either
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
        if uc_dimensions is not None and self.dimension > 0:
            self.uc_dimensions = uc_dimensions

        if expand_matrices is not None:  # perhaps these would be from a fiber or some sort of BIOMT?
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

    # @property
    # def chains(self) -> list[Entity]:
    #     """Return all the Chain objects including symmetric chains"""
    #     return [chain for entity in self.entities for chain in entity.chains]

    # Todo this is same as atom_indices_per_entity_symmetric
    # @property
    # def atom_indices_per_entity_model(self) -> list[list[int]]:
    #     # Todo
    #     #   alternative solution may be quicker by performing the following multiplication then .flatten()
    #     #   broadcast entity_indices ->
    #     #   (np.arange(model_number) * coords_length).T
    #     #   |
    #     #   v
    #     number_of_atoms = self.number_of_atoms
    #     # number_of_atoms = len(self.coords)
    #     return [[idx + (number_of_atoms * model_number) for model_number in range(self.number_of_models)
    #              for idx in entity_indices] for entity_indices in self.atom_indices_per_entity]
    #  Todo this is used in atom_indices_per_entity_symmetric
    #     return [self.get_symmetric_indices(entity_indices) for entity_indices in self.atom_indices_per_entity]

    @property
    def sequence(self) -> str:
        """Holds the SymmetricModel amino acid sequence"""
        return ''.join(entity.sequence for entity in self.chains)

    @property
    def reference_sequence(self) -> str:
        """Return the entire SymmetricModel sequence, constituting all Residues, not just structurally modelled ones

        Returns:
            The sequence according to each of the Entity references
        """
        return ''.join(entity.reference_sequence for entity in self.chains)

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

        symmetry_state = ['_symmetry',
                          '_point_group_symmetry',
                          '_dimension',
                          '_cryst_record',
                          '_number_of_symmetry_mates']
        for attribute in symmetry_state:
            try:
                delattr(self, attribute)
            except AttributeError:
                continue

    @property
    def symmetry(self) -> str | None:
        """The overall symmetric state, ie, the symmetry result. Uses SymEntry.resulting_symmetry"""
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
        # number_of_atoms = self.number_of_atoms
        # number_of_atoms = len(self.coords)
        return [self.get_symmetric_indices(entity_indices) for entity_indices in self.atom_indices_per_entity]
        # return [[idx + (number_of_atoms * model_number) for model_number in range(self.number_of_symmetry_mates)
        #          for idx in entity_indices] for entity_indices in self.atom_indices_per_entity]

    # def set_asu_coords(self, coords: Coords | np.ndarray | list[list[float]]):
    #     """Set the coordinates corresponding to the asymmetric unit for the SymmetricModel"""
    #     self._coords.set(coords)
    #     if self.symmetry:
    #         self.generate_symmetric_coords()

    # @property
    # def asu_coords(self) -> np.ndarray:
    #     """Return a view of the ASU Coords"""
    #     return self._coords.coords[self.asu_indices]
    #
    # @asu_coords.setter
    # def asu_coords(self, coords: Coords):
    #     self.coords = coords
    #     # set the symmetric coords according to the ASU
    #     self.generate_symmetric_coords()

    @StructureBase.coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        # self.coords = coords
        super(Structure, Structure).coords.fset(self, coords)  # prefer this over below, as this mechanism could change
        # self._coords.replace(self._atom_indices, coords)
        if self.symmetry:  # set the symmetric coords according to the ASU
            self.generate_symmetric_coords()

        # delete any saved attributes from the SymmetricModel (or Model)
        self.reset_state()

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
    def symmetric_coords(self) -> np.ndarray | None:
        """Return a view of the symmetric models Coords"""
        try:
            return self._models_coords.coords
        except AttributeError:
            return None

    # @symmetric_coords.setter
    # def symmetric_coords(self, coords: np.ndarray | list[list[float]]):
    #     if isinstance(coords, Coords):
    #         self._symmetric_coords = coords
    #     else:
    #         self._symmetric_coords = Coords(coords)
    #     Todo make below standard (like StructureBase) once symmetric_coords are part of other Structures (Entity?)
    #     self._symmetric_coords.replace(self.get_symmetric_indices(self._atom_indices), coords)

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
        # return np.matmul(self.center_of_mass, self.expand_matrices)
        try:
            return self._center_of_mass_symmetric_models
        except AttributeError:
            self._center_of_mass_symmetric_models = self.return_symmetric_coords(self.center_of_mass)
            return self._center_of_mass_symmetric_models

    @property
    def center_of_mass_symmetric_entities(self) -> list[np.ndarray]:
        """The individual centers of mass for each Entity in the symmetric system"""
        # if self.symmetry:
        # self._center_of_mass_symmetric_entities = []
        # for num_atoms, entity_coords in zip(self.number_of_atoms_per_entity, self.symmetric_coords_split_by_entity):
        #     self._center_of_mass_symmetric_entities.append(np.matmul(np.full(num_atoms, 1 / num_atoms),
        #                                                              entity_coords))
        # return self._center_of_mass_symmetric_entities
        # return [np.matmul(entity.center_of_mass, self.expand_matrices) for entity in self.entities]
        try:
            return self._center_of_mass_symmetric_entities
        except AttributeError:
            self._center_of_mass_symmetric_entities = [self.return_symmetric_coords(entity.center_of_mass)
                                                       for entity in self.entities]
            return self._center_of_mass_symmetric_entities

    @property
    def assembly(self) -> Model:
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
                self._assembly = Model.from_chains(chains, name='assembly', log=self.log, entities=False,
                                                   biomt_header=self.format_biomt(), cryst_record=self.cryst_record)
            return self._assembly

    @property
    def assembly_minimally_contacting(self) -> Model:  # Todo reconcile mechanism with Entity.oligomer
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
                Model.from_chains(chains, name='assembly', log=self.log, biomt_header=self.format_biomt(),
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
        #     # get_pdb_coords = getattr(PDB, 'backbone_and_cb_coords')
        #     self.coords_type = 'bb_cb'

        # self.number_of_symmetry_mates = valid_subunit_number[self.symmetry]
            print('self.cords.shape', self.coords.shape)
            print('self.expand_matrices.shape', self.expand_matrices.shape)
            print('self.expand_translations.shape', self.expand_translations.shape)
            print('self.number_of_symmetry_mates', self.number_of_symmetry_mates)
            symmetric_coords = (np.matmul(np.tile(self.coords, (self.number_of_symmetry_mates, 1, 1)),
                                          self.expand_matrices) + self.expand_translations).reshape(-1, 3)
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
        #     # get_pdb_coords = getattr(PDB, 'backbone_and_cb_coords')
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

        self._models_coords = Coords(symmetric_coords)

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
        if self.dimension > 0 and surrounding_uc:  # if the surrounding_uc is requested, we might need to generate it
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
                              f'{type(self).__name__} is being taken which is relying on Structure.__copy__. This may '
                              f'not be adequate and need to be overwritten')
            symmetry_mate = copy(self)
            # old-style
            # symmetry_mate_pdb.replace_coords(self.symmetric_coords[(coord_idx * number_of_atoms):
            #                                                        ((coord_idx + 1) * number_of_atoms)])
            # new-style
            symmetry_mate.coords = self.symmetric_coords[(coord_idx * number_of_atoms):
                                                         ((coord_idx + 1) * number_of_atoms)]
            self.models.append(symmetry_mate)

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
            self._oligomeric_model_indices = {}
            self._find_oligomeric_model_indices()
            return self._oligomeric_model_indices

    def _find_oligomeric_model_indices(self, epsilon: float = 0.5):
        """From an Entity's Chain members, find the SymmetricModel equivalent models using Chain center or mass
        compared to the symmetric model center of mass

        Args:
            epsilon: The distance measurement tolerance to find similar symmetric models to the oligomer
        """
        # number_of_atoms = self.number_of_atoms
        for entity_symmetric_centers_of_mass, entity in zip(self.center_of_mass_symmetric_entities, self.entities):
            if not entity.is_oligomeric():
                self._oligomeric_model_indices[entity] = []
                continue
            # need to slice through the specific Entity coords once we have the model
            # entity_indices = entity.atom_indices
            # entity_start, entity_end = entity_indices[0], entity_indices[-1]
            # entity_length = entity.number_of_atoms
            # entity_center_of_mass_divisor = np.full(entity_length, 1 / entity_length)
            equivalent_models = []
            for chain in entity.chains:
                # chain_length = chain.number_of_atoms
                # chain_center_of_mass = np.matmul(np.full(chain_length, 1 / chain_length), chain.coords)
                chain_center_of_mass = chain.center_of_mass
                # print('Chain', chain_center_of_mass.astype(int))
                for model_idx, sym_model_center_of_mass in enumerate(entity_symmetric_centers_of_mass):
                    # sym_model_center_of_mass = \
                    #     np.matmul(entity_center_of_mass_divisor,
                    #               self.symmetric_coords[(model_idx * number_of_atoms) + entity_start:
                    #                                     (model_idx * number_of_atoms) + entity_end + 1])
                    # #                                             have to add 1 for slice ^
                    # print('Sym Model', sym_model_center_of_mass)
                    # if np.allclose(chain_center_of_mass.astype(int), sym_model_center_of_mass.astype(int)):
                    # if np.allclose(chain_center_of_mass, sym_model_center_of_mass):  # using np.rint()
                    if np.linalg.norm(chain_center_of_mass - sym_model_center_of_mass) < epsilon:
                        equivalent_models.append(model_idx)
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
            # self.generate_assembly_tree()
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

    def get_symmetric_indices(self, indices: list[int]) -> list[int]:
        """Extend asymmetric indices across the symmetric_coords using the symmetry state

        Args:
            indices: The asymmetric indices to symmetrize
        Returns:
            The symmetric indices of the asymmetric input
        """
        atom_num = self.number_of_atoms
        return [idx + (atom_num * model_num) for model_num in range(self.number_of_symmetry_mates) for idx in indices]

    def return_symmetric_copies(self, structure: ContainsAtomsMixin, return_side_chains: bool = True,
                                surrounding_uc: bool = True, **kwargs) -> list[ContainsAtomsMixin]:
        """Expand the provided Structure using self.symmetry for the symmetry specification

        Args:
            structure: A ContainsAtomsMixin Structure object with .coords/.backbone_and_cb_coords methods
            return_side_chains: Whether to make the structural copy with side chains
            surrounding_uc: Whether the 3x3 layer group, or 3x3x3 space group should be generated
        Returns:
            The symmetric copies of the input structure
        """
        self.log.critical(f'Ensure the output of symmetry mate creation is correct. The copy of a '
                          f'{type(structure).__name__} is being taken which is relying on '
                          f'{type(structure).__name__}.__copy__. This may not be adequate and need to be overwritten')
        # Caution, this function will return poor if the number of atoms in the structure is 1!
        coords = structure.coords if return_side_chains else structure.backbone_and_cb_coords
        uc_number = 1
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
            symmetry_mate = copy(structure)
            # old-style
            # symmetry_mate_pdb.replace_coords(coord_set)  # [model_num * coords_length:(model_num + 1) * coords_length])
            # new-style
            symmetry_mate.coords = coord_set
            sym_mates.append(symmetry_mate)

        if len(sym_mates) != uc_number * self.number_of_symmetry_mates:
            raise SymmetryError(f'Number of models ({len(sym_mates)}) is incorrect! Should be '
                                f'{uc_number * self.number_of_uc_symmetry_mates}')
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
    #     coords = structure.coords if return_side_chains else structure.backbone_and_cb_coords
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
    #     coords = structure.coords if return_side_chains else structure.backbone_and_cb_coords
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
        self.coords = np.concatenate(asu_coords)
        # self.asu_coords = Coords(np.concatenate(asu_coords))
        # for idx, entity in enumerate(self.entities):
        #     entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **transform_solutions[idx])

        return transform_solutions

    def find_contacting_asu(self, distance: float = 8., **kwargs) -> list[Entity]:
        """Find the maximally contacting symmetry mate for each Entity and return the corresponding Entity instances

        Args:
            distance: The distance to check for contacts
        Returns:
            The minimal set of Entities containing the maximally touching configuration
        """
        if not self.entities:
            # The SymmetricModel was probably set without them. Create them, then try to find the asu
            self._create_entities()

        entities = self.entities
        if self.number_of_entities != 1:
            idx = 0
            chain_combinations: list[tuple[Entity, Entity]] = []
            entity_combinations: list[tuple[Entity, Entity]] = []
            contact_count = \
                np.zeros(sum(map(prod, combinations((entity.number_of_symmetry_mates for entity in entities), 2))))
            for entity1, entity2 in combinations(entities, 2):
                for chain1 in entity1.chains:
                    chain_cb_coord_tree = BallTree(chain1.cb_coords)
                    for chain2 in entity2.chains:
                        entity_combinations.append((entity1, entity2))
                        chain_combinations.append((chain1, chain2))
                        contact_count[idx] = \
                            chain_cb_coord_tree.two_point_correlation(chain2.cb_coords, [distance])[0]
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
            # Rearrange the entities to have the same order as provided
            entities = [new_entity for entity in entities for new_entity in new_entities if entity == new_entity]

        return entities

    def get_contacting_asu(self, distance: float = 8., **kwargs) -> Model:
        """Find the maximally contacting symmetry mate for each Entity and return the corresponding Entity instances as
         a new Pose

        If the chain IDs of the asu are the same, then chain IDs will automatically be renamed

        Args:
            distance: The distance to check for contacts
        Returns:
            A new Model with the minimal set of Entity instances. Will also be symmetric
        """
        entities = self.find_contacting_asu(distance=distance, **kwargs)
        found_chain_ids = []
        for entity in entities:
            if entity.chain_id in found_chain_ids:
                kwargs['rename_chains'] = True
                break
            else:
                found_chain_ids.append(entity.chain_id)

        return type(self).from_entities(entities, name=f'{self.name}-asu', log=self.log, sym_entry=self.sym_entry,
                                        biomt_header=self.format_biomt(), cryst_record=self.cryst_record, **kwargs)

    def set_contacting_asu(self, **kwargs):
        """Find the maximally contacting symmetry mate for each Entity, then set the Pose with this info

        Keyword Args:
            distance: float = 8.0 - The distance to check for contacts
        Sets:
            self: To a SymmetricModel with the minimal set of Entities containing the maximally touching configuration
        """
        entities = self.find_contacting_asu(**kwargs)
        # self = Model.from_entities(entities, name='asu', log=self.log, **kwargs)
        # self._pdb = Model.from_entities(entities, name='asu', log=self.log, **kwargs)
        # Todo
        #  With perfect symmetry, v this is sufficient. Need to figure out setting _process_model keyword args
        #   self.coords = np.concatenate([entity.coords for entity in entities])
        self._process_model(entities=entities, chains=False, **kwargs)

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

        # self.generate_assembly_tree()
        # clashes = asu_coord_tree.two_point_correlation(self.symmetric_coords[model_indices_without_asu], [distance])
        clashes = self.assembly_tree.two_point_correlation(self.coords[self.backbone_and_cb_indices], [distance])
        if clashes[0] > 0:
            self.log.warning(f'{self.name}: Found {clashes[0]} clashing sites! Pose is not a viable symmetric assembly')
            return True  # clash
        else:
            return False  # no clash

    @property
    def assembly_tree(self) -> BinaryTree:
        """Holds the tree structure of the symmetric_coords"""
        try:
            return self._assembly_tree
        except AttributeError:
            # model_asu_indices = self.get_asu_atom_indices()
            # if self.coords_type == 'bb_cb':  # grab every coord in the model
            #     model_indices = np.arange(len(self.symmetric_coords))
            #     asu_indices = []
            # else:  # Select only coords that are BB or CB from the model coords
            asu_indices = self.backbone_and_cb_indices
            # We have all the BB/CB indices from ASU, must multiply this int's in self.number_of_symmetry_mates
            # to get every BB/CB coord in the model
            # Finally we take out those indices that are inclusive of the model_asu_indices like below
            # model_indices = self.get_symmetric_indices(asu_indices)
            # # # make a boolean mask where the model indices of interest are True
            # # without_asu_mask = np.logical_or(model_indices < model_asu_indices[0],
            # #                                  model_indices > model_asu_indices[-1])
            # # model_indices_without_asu = model_indices[without_asu_mask]
            # # take the boolean mask and filter the model indices mask to leave only symmetry mate indices, NOT asu
            # model_indices_without_asu = self.get_symmetric_indices(asu_indices)[len(asu_indices):]
            # selected_assembly_coords = len(model_indices_without_asu) + len(asu_indices)
            # all_assembly_coords_length = len(asu_indices) * self.number_of_symmetry_mates
            # assert selected_assembly_coords == all_assembly_coords_length, \
            #     '%s: Ran into an issue indexing' % self.symmetric_assembly_is_clash.__name__

            # asu_coord_tree = BallTree(self.coords[asu_indices])
            # return BallTree(self.symmetric_coords[model_indices_without_asu])
            self._assembly_tree = \
                BallTree(self.symmetric_coords[self.get_symmetric_indices(asu_indices)[len(asu_indices):]])
            return self._assembly_tree

    def orient(self, symmetry: str = None, log: AnyStr = None):  # similar function in Entity
        """Orient is not available for SymmetricModel"""
        raise NotImplementedError(f'{self.orient.__name__} is not available for {type(self).__name__}')
        # Todo is this method at all useful? Could there be a situation where the symmetry is right,
        #  but the axes aren't in their canonical locations?

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
            AnyStr | None:
        """Write SymmetricModel Atoms to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            assembly: Whether to write the full assembly. Default writes only the ASU
        Keyword Args:
            increment_chains: bool = False - Whether to write each Structure with a new chain name, otherwise write as
                a new Model
            surrounding_uc: bool = True - Write the surrounding unit cell if assembly is True and self.dimension > 1
            header: None | str - A string that is desired at the top of the .pdb file
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
        Returns:
            The name of the written file if out_path is used
        """
        self.log.debug(f'SymmetricModel is writing')

        def symmetric_model_write(handle):
            if assembly:  # will make models and use next logic steps to write them out
                self.generate_assembly_symmetry_models(**kwargs)
                # self.models is populated, use Models.write() to finish
                super(SymmetricModel, SymmetricModel).write(self, file_handle=handle, **kwargs)
            else:  # skip models, write asu using biomt_record/cryst_record for sym
                for entity in self.entities:
                    entity.write(file_handle=handle, **kwargs)

        def model_write(handle):
            super(Models, Models).write(self, file_handle=handle, **kwargs)

        if file_handle:
            # self.write_header(file_handle, **kwargs)
            if self.symmetry:
                symmetric_model_write(file_handle)
            else:
                model_write(file_handle)
            return None
        else:
            with open(out_path, 'w') as outfile:
                self.write_header(outfile, **kwargs)
                if self.symmetry:
                    symmetric_model_write(outfile)
                else:
                    model_write(outfile)
            return out_path


class Pose(SequenceProfile, SymmetricModel):
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
    split_interface_residues: dict[int, list[tuple[Residue, Entity]]]
    split_interface_ss_elements: dict[int, list[int]]
    ss_index_array: list[int]
    ss_type_array: list[str]

    def __init__(self, fragment_db: fragment.FragmentDatabase = None, ignore_clashes: bool = False,
                 design_selector: dict[str, dict[str, dict[str, set[int] | set[str] | None]]] = None, **kwargs):
        # unused args
        #           euler_lookup: EulerLookup = None,
        self.design_selector = design_selector if design_selector else {}  # kwargs.get('design_selector', {})
        self.design_selector_entities = set()
        self.design_selector_indices = set()
        self.euler_lookup = euler_factory()  # kwargs.get('euler_lookup', None)
        self.fragment_metrics = {}
        self.fragment_pairs = []
        self.fragment_queries = {}
        self.ignore_clashes = ignore_clashes
        self.interface_residues = {}
        self.required_indices = set()
        self.required_residues = None
        self.split_interface_residues = {}  # {1: [(Residue obj, Entity obj), ...], 2: [(Residue obj, Entity obj), ...]}
        self.split_interface_ss_elements = {}  # {1: [0, 1, 2] , 2: [9, 13, 19]]}
        self.ss_index_array = []  # stores secondary structure elements by incrementing index
        self.ss_type_array = []  # stores secondary structure type ('H', 'S', ...)

        # Model init will handle Structure set up if a structure file is present
        # SymmetricModel init will generate_symmetric_coords() if symmetry specification present
        super().__init__(**kwargs)
        if self.is_clash():
            if not self.ignore_clashes:
                raise ClashError(f'{self.name} contains Backbone clashes and is not being considered further!')

        # need to set up after load Entities so that they can have this added to their SequenceProfile
        self.fragment_db = fragment_db  # kwargs.get('fragment_db', None)
        self.create_design_selector()  # **self.design_selector)
        self.log.debug(f'Entities: {", ".join(entity.name for entity in self.entities)}')
        self.log.debug(f'Active Entities: {", ".join(entity.name for entity in self.active_entities)}')

    @property
    def fragment_db(self) -> fragment.FragmentDatabase:
        """The FragmentDatabase with which information about fragment usage will be extracted"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: fragment.FragmentDatabase):
        if not isinstance(fragment_db, fragment.FragmentDatabase):
            self.log.warning(f'The passed fragment_db is being set to the default since {fragment_db} was passed which '
                             f'is not of the required type {fragment.FragmentDatabase.__name__}')
            # Todo add fragment_length, sql kwargs
            fragment_db = fragment.fragment_factory(source=PUtils.biological_interfaces)

        self._fragment_db = fragment_db
        for entity in self.entities:
            entity.fragment_db = fragment_db

    # @SymmetricModel.asu.setter
    # def asu(self, asu):
    #     self.pdb = asu  # process incoming structure as normal
    #     if self.number_of_entities != self.number_of_chains:  # ensure the structure is an asu
    #         # self.log.debug('self.number_of_entities (%d) self.number_of_chains (%d)'
    #         #                % (self.number_of_entities, self.number_of_chains))
    #         self.log.debug('Setting Pose ASU to the ASU with the most contacting interface')
    #         self.set_contacting_asu()  # find maximally touching ASU and set ._pdb

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
            entity_mask, atom_mask = set(), set()

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
            entity_required, self.required_indices = set(), set()

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
        # asu_interface_labels = kmeans_cluster_model.predict(interface_asu_structure.cb_coords)

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

        if self.symmetry:  # get the symmetric indices of interest
            entity2_indices = self.get_symmetric_indices(entity2_indices)
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
            entity2_coords = self.symmetric_coords[entity2_indices]  # get the symmetric indices from Entity 2
            sym_string = 'symmetric '
        elif entity1 == entity2:
            # without symmetry, we can't measure this, unless intra-oligomeric contacts are desired
            self.log.warning('Entities are the same, but no symmetry is present. The interface between them will not be'
                             ' detected!')
            raise NotImplementedError('These entities shouldn\'t necessarily be equal. This issue needs to be addressed'
                                      'by expanding the __eq__ method of Entity to more accurately reflect what a '
                                      'Structure object represents programmatically')
            # return
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

        coords_indexed_residues = self.coords_indexed_residues
        # get the modulus of the number_of_atoms to account for symmetry if used
        number_of_atoms = self.number_of_atoms
        contacting_pairs = [(coords_indexed_residues[entity1_indices[entity1_idx]],
                             coords_indexed_residues[entity2_indices[entity2_idx] % number_of_atoms])
                            for entity2_idx, entity1_contacts in enumerate(entity2_query)
                            for entity1_idx in entity1_contacts]
        if entity1 == entity2:  # solve symmetric results for asymmetric contacts
            asymmetric_contacting_pairs, found_pairs = [], set()
            for pair1, pair2 in contacting_pairs:
                # only add to contacting pair if we have never observed either
                if (pair1, pair2) not in found_pairs:  # or (pair2, pair1) not in found_pairs:
                    asymmetric_contacting_pairs.append((pair1, pair2))
                # add both pair orientations (1, 2) and (2, 1) regardless
                found_pairs.update([(pair1, pair2), (pair2, pair1)])

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

        if entity1 == entity2:  # if symmetric query
            # is the interface is across a dimeric interface?
            for residue in entity2_residues:  # entity2 usually has fewer residues, this might be quickest
                if residue in entity1_residues:  # the interface is dimeric
                    # include all residues found to only one side and move on
                    entity1_residues = sorted(set(entity1_residues).union(entity2_residues), key=lambda res: res.number)
                    entity2_residues = []
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
        try:
            residues1, residues2 = self.interface_residues[(entity1, entity2)]
        except KeyError:  # when interface_residues haven't been set
            self.find_interface_residues(entity1=entity1, entity2=entity2)
            try:
                residues1, residues2 = self.interface_residues[(entity1, entity2)]
            except KeyError:
                raise DesignError(f'{self.find_interface_atoms.__name__} can\'t access interface_residues as the Entity'
                                  f' pair {entity1.name}, {entity2.name} hasn\'t located interface_residues')

        if not residues1:
            return
        entity1_indices: list[int] = []
        for residue in residues1:
            entity1_indices.extend(residue.heavy_indices)

        if not residues2:  # check if the interface is a symmetric self dimer and all residues are in residues1
            # residues2 = residues1
            entity2_indices = entity1_indices
        else:
            entity2_indices: list[int] = []
            for residue in residues2:
                entity2_indices.extend(residue.heavy_indices)

        if self.symmetry:  # get all symmetric indices for entity2
            query_coords = self.symmetric_coords[self.get_symmetric_indices(entity2_indices)]
        else:
            query_coords = self.coords[entity2_indices]

        interface_atom_tree = BallTree(self.coords[entity1_indices])
        atom_query = interface_atom_tree.query_radius(query_coords, distance)
        contacting_pairs = [(entity1_indices[entity1_idx], entity2_indices[entity2_idx])
                            for entity2_idx, entity1_contacts in enumerate(atom_query)
                            for entity1_idx in entity1_contacts]
        return contacting_pairs

    def local_density_interface(self, distance: float = 12.) -> float:
        """Returns the density of heavy Atoms neighbors within 'distance' Angstroms to Atoms in the Pose interface

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

        if self.symmetry:
            interface_coords = self.symmetric_coords[list(set(interface_indices1).union(interface_indices2))]
        else:
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
        # because the way self.interface_residues is set, when there is not interface, a check on entity1_residues is
        # sufficient, however entity2_residues is empty with an interface present across a non-oligomeric dimeric 2-fold
        if not entity1_residues:  # or not entity2_residues:
            self.log.info(f'No residues at the {entity1.name} | {entity2.name} interface. Fragments not available')
            self.fragment_queries[(entity1, entity2)] = []
            return

        # residue_numbers1 = sorted(residue.number for residue in entity1_residues)
        # surface_frags1 = entity1.get_fragments(residue_numbers=residue_numbers1,
        #                                        representatives=self.fragment_db.reps)
        frag_residues1 = \
            entity1.get_fragment_residues(residues=entity1_residues, representatives=self.fragment_db.reps)
        if not entity2_residues:  # entity1 == entity2 and not entity2_residues:
            # entity1_residues = set(entity1_residues + entity2_residues)
            entity2_residues = entity1_residues
            frag_residues2 = frag_residues1
            # residue_numbers2 = residue_numbers1
        else:
            # residue_numbers2 = sorted(residue.number for residue in entity2_residues)
            # surface_frags2 = entity2.get_fragments(residue_numbers=residue_numbers2,
            #                                        representatives=self.fragment_db.reps)
            frag_residues2 = \
                entity2.get_fragment_residues(residues=entity2_residues, representatives=self.fragment_db.reps)

        # self.log.debug(f'At Entity {entity1.name} | Entity {entity2.name} interface, searching for fragments at the '
        #                f'surface of:\n\tEntity {entity1.name}: Residues {", ".join(map(str, residue_numbers1))}'
        #                f'\n\tEntity {entity2.name}: Residues {", ".join(map(str, residue_numbers2))}')

        if not frag_residues1 or not frag_residues2:
            self.log.info(f'No fragments found at the {entity1.name} | {entity2.name} interface')
            self.fragment_queries[(entity1, entity2)] = []
            return
        else:
            self.log.debug(f'At Entity {entity1.name} | Entity {entity2.name} interface:\t'
                           f'{entity1.name} has {len(frag_residues1)} interface fragments at residues {",".join(map(str, [res.number for res in frag_residues1]))}\t'
                           f'{entity2.name} has {len(frag_residues2)} interface fragments at residues {",".join(map(str, [res.number for res in frag_residues2]))}')

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
            frag_residues2.clear()
            for frag_mates in symmetric_surface_frags2:
                frag_residues2.extend([frag for sym_idx, frag in enumerate(frag_mates) if sym_idx not in skip_models])
            self.log.debug(f'Entity {entity2.name} has {len(frag_residues2)} symmetric fragments')

        entity1_coords = entity1.backbone_and_cb_coords  # for clash check, we only want the backbone and CB
        ghostfrag_surfacefrag_pairs = find_fragment_overlap(entity1_coords, frag_residues1, frag_residues2,
                                                            frag_db=self.fragment_db, euler_lookup=self.euler_lookup)
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
        # if self.api_db:
        try:
            # retrieve_api_info = self.api_db.pdb.retrieve_data
            retrieve_stride_info = wrapapi.api_database_factory().stride.retrieve_data
        except AttributeError:
            retrieve_stride_info = Structure.stride

        pose_secondary_structure = ''
        for entity in self.active_entities:
            if not entity.secondary_structure:
                parsed_secondary_structure = retrieve_stride_info(name=entity.name)
                if parsed_secondary_structure:
                    entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                else:
                    entity.stride(to_file=self.api_db.stride.path_to(entity.name))

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

    # def interface_design(self, evolution=True, fragments=True, write_fragments=True, des_dir=None):
    #     """Compute calculations relevant to interface design.
    #
    #     Sets:
    #         self.pssm_file (AnyStr)
    #     """
    #     # self.log.debug('Entities: %s' % ', '.join(entity.name for entity in self.entities))
    #     # self.log.debug('Active Entities: %s' % ', '.join(entity.name for entity in self.active_entities))
    #
    #     # we get interface residues for the designable entities as well as interface_topology at PoseDirectory level
    #     if fragments:
    #         # if query_fragments:  # search for new fragment information
    #         self.generate_interface_fragments(out_path=des_dir.frags, write_fragments=write_fragments)
    #         # else:  # No fragment query, add existing fragment information to the pose
    #         #     if fragment_source is None:
    #         #         raise DesignError(f'Fragments were set for design but there were none found! Try excluding '
    #         #                           f'--{PUtils.no_term_constraint} in your input flags and rerun this command, or '
    #         #                           f'generate them separately with "{PUtils.program_command} '
    #         #                           f'{PUtils.generate_fragments}"')
    #         #
    #         #     self.log.debug('Fragment data found from prior query. Solving query index by Pose numbering/Entity '
    #         #                    'matching')
    #         #     self.add_fragment_query(query=fragment_source)
    #
    #         for query_pair, fragment_info in self.fragment_queries.items():
    #             self.log.debug('Query Pair: %s, %s\n\tFragment Info:%s' % (query_pair[0].name, query_pair[1].name,
    #                                                                        fragment_info))
    #             for query_idx, entity in enumerate(query_pair):
    #                 entity.map_fragments_to_profile(fragments=fragment_info, alignment_type=alignment_types[query_idx])
    #     for entity in self.entities:
    #         # TODO Insert loop identifying comparison of SEQRES and ATOM before SeqProf.calculate_design_profile()
    #         if entity not in self.active_entities:  # we shouldn't design, add a null profile instead
    #             entity.add_profile(null=True)
    #         else:  # add a real profile
    #             if self.api_db:
    #                 profiles_path = self.api_db.hhblits_profiles.location
    #                 entity.sequence_file = self.api_db.sequences.retrieve_file(name=entity.name)
    #                 entity.evolutionary_profile = self.api_db.hhblits_profiles.retrieve_data(name=entity.name)
    #                 if not entity.evolutionary_profile:
    #                     entity.add_evolutionary_profile(out_path=profiles_path)
    #                 else:  # ensure the file is attached as well
    #                     entity.pssm_file = self.api_db.hhblits_profiles.retrieve_file(name=entity.name)
    #
    #                 if not entity.pssm_file:  # still no file found. this is likely broken
    #                     raise DesignError(f'{entity.name} has no profile generated. To proceed with this design/'
    #                                       f'protocol you must generate the profile!')
    #                 if len(entity.evolutionary_profile) != entity.number_of_residues:
    #                     # profile was made with reference or the sequence has inserts and deletions of equal length
    #                     # A more stringent check could move through the evolutionary_profile[idx]['type'] key versus the
    #                     # entity.sequence[idx]
    #                     entity.fit_evolutionary_profile_to_structure()
    #             else:
    #                 profiles_path = des_dir.profiles
    #
    #             if not entity.sequence_file:
    #                 entity.write_sequence_to_fasta('reference', out_path=des_dir.sequences)
    #             entity.add_profile(evolution=evolution, fragments=fragments, out_path=profiles_path)
    #
    #     # Update PoseDirectory with design information
    #     if fragments:  # set pose.fragment_profile by combining entity frag profile into single profile
    #         self.combine_fragment_profile([entity.fragment_profile for entity in self.entities])
    #         fragment_pssm_file = self.write_pssm_file(self.fragment_profile, PUtils.fssm, out_path=des_dir.data)
    #
    #     if evolution:  # set pose.evolutionary_profile by combining entity evo profile into single profile
    #         self.combine_pssm([entity.evolutionary_profile for entity in self.entities])
    #         self.pssm_file = self.write_pssm_file(self.evolutionary_profile, PUtils.pssm, out_path=des_dir.data)
    #
    #     self.combine_profile([entity.profile for entity in self.entities])
    #     design_pssm_file = self.write_pssm_file(self.profile, PUtils.dssm, out_path=des_dir.data)
    #     # -------------------------------------------------------------------------
    #     # self.solve_consensus()
    #     # -------------------------------------------------------------------------

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
        # Todo incorporate these
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
                self.log.info(f'Couldn\'t locate query metrics for Entity pair {entity1.name}, {entity2.name}')
            else:
                self.log.error(f'{self.return_fragment_metrics.__name__}: entity1 or entity1 can\'t be None!')

            return fragment_metric_template
        elif by_entity:
            metric_d = {}
            for query_pair, metrics in self.fragment_metrics.items():
                if not metrics:
                    continue
                for idx, entity in enumerate(query_pair):
                    if entity not in metric_d:
                        metric_d[entity] = fragment_metric_template

                    align_type = alignment_types[idx]
                    metric_d[entity]['center_residues'].update(metrics[align_type]['center']['residues'])
                    metric_d[entity]['total_residues'].update(metrics[align_type]['total']['residues'])
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
                metric_d['center_residues'].update(
                    metrics['mapped']['center']['residues'].union(metrics['paired']['center']['residues']))
                metric_d['total_residues'].update(
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
                    metric_d['percent_fragment_coil'] = 0., 0., 0.

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
        entity_energy_multiplier = [entity.number_of_symmetry_mates for entity in self.entities]

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
        entity_energy_multiplier = [entity.number_of_symmetry_mates for entity in self.entities]

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
    #         #     fragment['mapped'] = self.residue_number_from_pdb(fragment['mapped'])
    #         #     fragment['paired'] = self.residue_number_from_pdb(fragment['paired'])
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
    #     """Generate a fragment.FragmentDatabase connection
    #
    #     Args:
    #         source: The type of FragmentDatabase to connect
    #     Sets:
    #         self.fragment_db (fragment.FragmentDatabase)
    #     """
    #     self.fragment_db = fragment.fragment_factory(source=source, **kwargs)

    def generate_interface_fragments(self, write_fragments: bool = True, out_path: AnyStr = None):
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
                             out_path: AnyStr = os.getcwd()):
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
            available_chain_ids = self.chain_id_generator()
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
