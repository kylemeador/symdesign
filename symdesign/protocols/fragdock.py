from __future__ import annotations

import copy
import logging
import math
import os
import time
from collections.abc import Iterable
from itertools import repeat, count
from math import prod
from typing import AnyStr, Container

import numpy as np
import pandas as pd
import psutil
import scipy
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.neighbors._ball_tree import BinaryTree  # This typing implementation supports BallTree or KDTree

from symdesign import flags, resources, protocols
from symdesign.metrics import calculate_collapse_metrics, calculate_residue_surface_area, errat_1_sigma, errat_2_sigma,\
    multiple_sequence_alignment_dependent_metrics, profile_dependent_metrics, columns_to_new_column, \
    delta_pairs, division_pairs, interface_composition_similarity, clean_up_intermediate_columns, \
    sum_per_residue_metrics, hydrophobic_collapse_index, cross_entropy, collapse_significance_threshold
from symdesign.resources import ml, job as symjob
from symdesign.structure.base import Structure, Residue
from symdesign.structure.coords import transform_coordinate_sets
from symdesign.structure.fragment import GhostFragment
from symdesign.structure.fragment.visuals import write_fragment_pairs_as_accumulating_states
from symdesign.structure.model import Pose, Model, get_matching_fragment_pairs_info, Models
from symdesign.structure.sequence import generate_mutations_from_reference, numeric_to_sequence, concatenate_profile, \
    pssm_as_array, MultipleSequenceAlignment
from symdesign.structure.utils import chain_id_generator
from symdesign.utils import z_score, rmsd_z_score, z_value_from_match_score, match_score_from_z_value, path as putils
from symdesign.utils.SymEntry import SymEntry, get_rot_matrices, make_rotations_degenerate
from symdesign.utils.symmetry import generate_cryst1_record, identity_matrix

# Globals
logger = logging.getLogger(__name__)
# logger = start_log(name=__name__, format_log=False)
zero_offset = 1
zero_probability_frag_value = -20


# TODO decrease amount of work by saving each index array and reusing...
#  such as stacking each j_index, guide_coords, rmsd, etc and pulling out by index
def is_frag_type_same(frags1, frags2, dtype='ii'):
    frag1_type = f'{dtype[0]}_type'
    frag2_type = f'{dtype[1]}_type'
    frag1_indices = np.array([getattr(frag, frag1_type) for frag in frags1])
    frag2_indices = np.array([getattr(frag, frag2_type) for frag in frags2])
    # np.where(frag1_indices_repeat == frag2_indices_tile)
    frag1_indices_repeated = np.repeat(frag1_indices, len(frag2_indices))
    frag2_indices_tiled = np.tile(frag2_indices, len(frag1_indices))
    return np.where(frag1_indices_repeated == frag2_indices_tiled, True, False).reshape(
        len(frag1_indices), -1)


def compute_ij_type_lookup(indices1: np.ndarray | Iterable, indices2: np.ndarray | Iterable) -> np.ndarray:
    """Compute a lookup table where the array elements are indexed to boolean values if the indices match.
    Axis 0 is indices1, Axis 1 is indices2

    Args:
        indices1: The array elements from group 1
        indices2: The array elements from group 2
    Returns:
        A 2D boolean array where the first index maps to the input 1, second index maps to index 2
    """
    # TODO use broadcasting to compute true/false instead of tiling (memory saving)
    indices1_repeated = np.repeat(indices1, len(indices2))
    len_indices1 = len(indices1)
    indices2_tiled = np.tile(indices2, len_indices1)
    # TODO keep as table or flatten? Can use one or the other as memory and take a view of the other as needed...
    # return np.where(indices1_repeated == indices2_tiled, True, False).reshape(len_indices1, -1)
    return (indices1_repeated == indices2_tiled).reshape(len_indices1, -1)


def get_perturb_matrices(rotation_degrees: float, number: int = 10) -> np.ndarray:
    """Using a sampled degree of rotation, create z-axis rotation matrices in equal increments between +/- rotation_degrees/2

    Args:
        rotation_degrees: The range of degrees to create matrices for. Will be centered at the identity perturbation
        number: The number of steps to take
    Returns:
        A 3D numpy array where each subsequent rotation is along axis=0,
            and each 3x3 rotation matrix is along axis=1/2
    """
    half_grid_range, remainder = divmod(number, 2)
    # If the number is odd we should center on 0 else center with more on the negative side
    if remainder:
        upper_range = half_grid_range + 1
    else:
        upper_range = half_grid_range

    step_degrees = rotation_degrees / number
    perturb_matrices = []
    for step in range(-half_grid_range, upper_range):  # Range from -5 to 4(5) for example. 0 is identity matrix
        rad = math.radians(step * step_degrees)
        rad_s = math.sin(rad)
        rad_c = math.cos(rad)
        # Perform rotational perturbation on z-axis
        perturb_matrices.append([[rad_c, -rad_s, 0.], [rad_s, rad_c, 0.], [0., 0., 1.]])

    return np.array(perturb_matrices)


def create_perturbation_transformations(sym_entry: SymEntry, rotation_number: int = 1, translation_number: int = 1,
                                        rotation_range: Iterable[float] = None,
                                        translation_range: Iterable[float] = None) -> dict[str, np.ndarray]:
    """From a specified SymEntry and sampling schedule, create perturbations to degrees of freedom for each available

    Args:
        sym_entry: The SymEntry whose degrees of freedom should be expanded
        rotation_number: The number of times to sample from the allowed rotation space. 1 means no perturbation
        translation_number: The number of times to sample from the allowed translation space. 1 means no perturbation
        rotation_range: The range to sample rotations +/- the identified rotation in degrees.
            Expected type is an iterable of length comparable to the number of rotational degrees of freedom
        translation_range: The range to sample translations +/- the identified translation in Angstroms
            Expected type is an iterable of length comparable to the number of translational degrees of freedom
    Returns:
        A mapping between the perturbation type and the corresponding transformations
    """
    # Get the perturbation parameters
    # Total number of perturbations desired using the total_dof possible in the symmetry and those requested
    target_dof = total_dof = sym_entry.total_dof
    rotational_dof = sym_entry.number_dof_rotation
    translational_dof = sym_entry.number_dof_translation
    n_dof_external = sym_entry.number_dof_external

    if rotation_number < 1:
        raise ValueError(f"Can't create perturbation transformations with rotation_number of {rotation_number}")
    elif rotation_number == 1:
        target_dof -= rotational_dof

    if translation_number < 1:
        raise ValueError(f"Can't create perturbation transformations with translation_number of {translation_number}")
    elif translation_number == 1:
        target_dof -= translational_dof

    # # Make a vector of the perturbation number [1, 2, 2, 3, 3, 1] with 1 as constants on each end
    # dof_number_perturbations = [1] \
    #     + [rotation_number for dof in range(rotational_dof)] \
    # Make a vector of the perturbation number [2, 2, 3, 3]
    dof_number_perturbations = \
        [rotation_number for dof in range(rotational_dof)] \
        + [translation_number for dof in range(translational_dof)] \
        # + [1]
    # translation_stack_size = translation_number**translational_dof
    # stack_size = rotation_number**rotational_dof * translation_stack_size
    # number = rotation_number + translation_number

    stack_size = prod(dof_number_perturbations)
    # Initialize a translation grid for any translational degrees of freedom
    translation_grid = np.zeros((stack_size, 3), dtype=float)
    # # Begin with total dof minus 1
    # remaining_dof = total_dof - 1
    # # Begin with 0
    # seen_dof = 0
    dof_idx = 0
    # Default rotation range is 1. degree
    default_rotation = 1.
    # Default translation range is 0.5 Angstroms
    default_translation = .5

    if rotation_range is None:
        rotation_range = tuple(repeat(default_rotation, sym_entry.number_of_groups))

    if translation_range is None:
        translation_range = tuple(repeat(default_translation, sym_entry.number_of_groups))
        ext_translation_range = tuple(repeat(default_translation, n_dof_external))
    else:
        logger.warning(f'Currently using the default value of {default_translation} for external translation range')
        ext_translation_range = tuple(repeat(default_translation, n_dof_external))

    perturbation_mapping = {}
    for idx, group in enumerate(sym_entry.groups):
        group_idx = idx + 1
        if getattr(sym_entry, f'is_internal_rot{group_idx}'):
            rotation_step = rotation_range[idx]  # * 2
            perturb_matrices = get_perturb_matrices(rotation_step, number=rotation_number)
            # Repeat (tile then reshape) the matrices according to the number of perturbations raised to the power of
            # the remaining dof (remaining_dof), then tile that by how many dof have been seen (seen_dof)
            # perturb_matrices = \
            #     np.tile(np.tile(perturb_matrices,
            #                     (1, 1, rotation_number**remaining_dof * translation_stack_size)).reshape(-1, 3, 3),
            #             (number**seen_dof, 1, 1))
            # remaining_dof -= 1
            # seen_dof += 1
            # Get the product of the number of perturbations before and after the current index
            repeat_number = prod(dof_number_perturbations[dof_idx + 1:])
            tile_number = prod(dof_number_perturbations[:dof_idx])
            # Repeat (tile then reshape) the matrices according to the product of the remaining dof,
            # number of perturbations, then tile by the product of how many perturbations have been seen
            perturb_matrices = np.tile(np.tile(perturb_matrices, (1, repeat_number, 1)).reshape(-1, 3, 3),
                                       (tile_number, 1, 1))
            # Increment the dof seen
            dof_idx += 1
        else:
            # This is a requirement as currently, all SymEntry are assumed to have rotations
            # np.tile the identity matrix to make equally sized.
            perturb_matrices = np.tile(identity_matrix, (stack_size, 1, 1))

        perturbation_mapping[f'rotation{group_idx}'] = perturb_matrices

    for idx, group in enumerate(sym_entry.groups):
        group_idx = idx + 1
        if getattr(sym_entry, f'is_internal_tx{group_idx}'):
            # Repeat the translation according to the number of perturbations raised to the power of the
            # remaining dof (remaining_dof), then tile that by how many dof have been seen (seen_dof)
            internal_translation_grid = copy.copy(translation_grid)
            translation_perturb_vector = \
                np.linspace(-translation_range[idx], translation_range[idx], translation_number)
            # internal_translation_grid[:, 2] = np.tile(np.repeat(translation_perturb_vector, number**remaining_dof),
            #                                           number**seen_dof)
            # remaining_dof -= 1
            # seen_dof += 1
            # Get the product of the number of perturbations before and after the current index
            repeat_number = prod(dof_number_perturbations[dof_idx + 1:])
            tile_number = prod(dof_number_perturbations[:dof_idx])
            internal_translation_grid[:, 2] = np.tile(np.repeat(translation_perturb_vector, repeat_number),
                                                      tile_number)
            # Increment the dof seen
            dof_idx += 1
            perturbation_mapping[f'translation{group_idx}'] = internal_translation_grid

    # if n_dof_external:
    #     # sym_entry.number_dof_external are included in the sym_entry.total_dof calculation
    #     # Need to perturb this many dofs. Each additional ext DOF increments e, f, g.
    #     # So 2 number_dof_external gives e, f. 3 gives e, f, g. This way the correct number of axis can be perturbed..
    #     # This solution doesn't vary the translation_grid in all dofs
    #     # ext_dof_perturbs[:, :number_dof_external] = np.tile(translation_grid, (number_dof_external, 1)).T
    # This solution iterates over the translation_grid, adding a new grid over all remaining dofs
    external_translation_grid = copy.copy(translation_grid)
    for idx, ext_idx in enumerate(range(n_dof_external)):
        translation_perturb_vector = \
            np.linspace(-ext_translation_range[idx], ext_translation_range[idx], translation_number)
        # external_translation_grid[:, ext_idx] = np.tile(np.repeat(translation_perturb_vector,
        #                                                           number**remaining_dof),
        #                                                 number**seen_dof)
        # remaining_dof -= 1
        # seen_dof += 1
        # Get the product of the number of perturbations before and after the current index
        repeat_number = prod(dof_number_perturbations[dof_idx + 1:])
        tile_number = prod(dof_number_perturbations[:dof_idx])
        external_translation_grid[:, ext_idx] = np.tile(np.repeat(translation_perturb_vector, repeat_number),
                                                        tile_number)
        # Increment the dof seen
        dof_idx += 1

    perturbation_mapping['external_translations'] = external_translation_grid

    # if remaining_dof + 1 != 0 and seen_dof != total_dof:
    #     logger.critical(f'The number of perturbations is unstable! {remaining_dof + 1} != 0 and '
    #                     f'{seen_dof} != {total_dof} total_dof')
    if dof_idx != target_dof:
        logger.critical(f'The number of perturbations is unstable! '
                        f'perturbed dof used {dof_idx} != {target_dof} the targeted dof to perturb resulting from '
                        f'{total_dof} total_dof, {rotational_dof} rotational_dof, '
                        f'and {translational_dof} translational_dof')

    return perturbation_mapping


def fragment_dock(models: Iterable[Structure | AnyStr], **kwargs) -> list[protocols.PoseDirectory] | list:
    # model1: Structure | AnyStr, model2: Structure | AnyStr,
    """Perform the fragment docking routine described in Laniado, Meador, & Yeates, PEDS. 2021

    Args:
        models: The Structures to be used in docking
    Returns:
        The resulting Poses satisfying docking criteria
    """
    frag_dock_time_start = time.time()
    # Get Building Blocks in pose format to remove need for fragments to use chain info
    models = list(models)
    """The Structure instances to be used in docking"""
    for idx, model in enumerate(models):
        if not isinstance(model, Structure):
            models[idx] = Model.from_file(model)

    # Set up output mechanism
    # Retrieve symjob.JobResources for all flags
    job = symjob.job_resources_factory.get()
    sym_entry: SymEntry = job.sym_entry
    """The SymmetryEntry object describing the material"""
    entry_string = f'NanohedraEntry{sym_entry.entry_number}'
    building_blocks = '-'.join(model.name for model in models)
    entry_and_building_blocks = f'{entry_string}_{building_blocks}_{putils.pose_directory}'
    out_dir = os.path.join(job.projects, entry_and_building_blocks)
    putils.make_path(out_dir)

    euler_lookup = job.fragment_db.euler_lookup
    # This is used in clustering algorithms to define an observation outside the found clusters
    outlier = -1
    initial_z_value = job.dock.initial_z_value
    """The acceptable standard deviation z score for initial fragment overlap identification. Smaller values lead to 
    more stringent matching criteria
    """
    min_matched = job.dock.min_matched
    """How many high quality fragment pairs should be present before a pose is identified?"""
    high_quality_match_value = job.dock.match_value
    """The value to exceed before a high quality fragment is matched. When z-value was used this was 1.0, however, 0.5
    when match score is used
    """
    rotation_step1 = job.dock.rotation_step1
    rotation_step2 = job.dock.rotation_step2
    # Todo set below as parameters?
    low_quality_match_value = .2
    """The lower bounds on an acceptable match. Was upper bound of 2 using z-score"""
    clash_dist: float = 2.2
    """The distance to measure for clashing atoms"""
    cb_distance = 9.  # change to 8.?
    """The distance to measure for interface atoms"""
    # Testing if this is too strict when strict overlaps are used
    cluster_transforms = not job.dock.contiguous_ghosts  # True
    # Todo set above as parameters?
    translation_epsilon = 1  # 1 seems to work well at recapitulating the results without it. More stringent -> 0.75
    high_quality_z_value = z_value_from_match_score(high_quality_match_value)
    low_quality_z_value = z_value_from_match_score(low_quality_match_value)

    if job.dock.perturb_dof_tx:
        if sym_entry.unit_cell:
            logger.critical(f"{create_perturbation_transformations.__name__} hasn't been tested for lattice symmetries")

    # Ensure models are oligomeric with make_oligomer()
    entity_count = count(1)
    for idx, (model, symmetry) in enumerate(zip(models, sym_entry.groups)):
        for entity in model.entities:
            # Precompute reference sequences if available
            # dummy = entity.reference_sequence  # use the incomming SEQRES REMARK for now
            if entity.is_oligomeric():
                continue
            else:
                entity.make_oligomer(symmetry=symmetry)

            if next(entity_count) > 2:
                # Todo remove able to take more than 2 Entity
                raise NotImplementedError(f"Can't dock 2 Models with > 2 total Entity instances")
        # Make, then save a new model based on the symmetric version of each Entity in the Model
        _model = Model.from_chains([chain for entity in model.entities for chain in entity.chains], name=model.name)
        _model.file_path = model.file_path
        _model.fragment_db = job.fragment_db
        models[idx] = _model

    # Todo reimplement this feature to write a log to the Project directory
    # # Setup logger
    # if logger is None:
    #     log_file_path = os.path.join(out_dir, f'{building_blocks}_log.txt')
    # else:
    #     try:
    #         log_file_path = getattr(logger.handlers[0], 'baseFilename', None)
    #     except IndexError:  # No handler attached to this logger. Probably passing to a parent logger
    #         log_file_path = None
    #
    # if log_file_path:  # Start logging to a file in addition
    #     logger = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False, propagate=True)

    for model in models:
        model.log = logger

    # Todo figure out for single component
    model1: Model
    model2: Model
    model1, model2 = models
    logger.info(f'DOCKING {model1.name} TO {model2.name}\n'
                f'Oligomer 1 Path: {model1.file_path}\nOligomer 2 Path: {model2.file_path}')

    # Set up Building Block2
    # Get Surface Fragments With Guide Coordinates Using COMPLETE Fragment Database
    get_complete_surf_frags2_time_start = time.time()
    complete_surf_frags2 = \
        model2.get_fragment_residues(residues=model2.surface_residues, fragment_db=job.fragment_db)

    # Calculate the initial match type by finding the predominant surface type
    surf_guide_coords2 = np.array([surf_frag.guide_coords for surf_frag in complete_surf_frags2])
    surf_residue_numbers2 = np.array([surf_frag.number for surf_frag in complete_surf_frags2])
    surf_i_indices2 = np.array([surf_frag.i_type for surf_frag in complete_surf_frags2])
    fragment_content2 = np.bincount(surf_i_indices2)
    initial_surf_type2 = np.argmax(fragment_content2)
    init_surf_frag_indices2 = \
        [idx for idx, surf_frag in enumerate(complete_surf_frags2) if surf_frag.i_type == initial_surf_type2]
    init_surf_guide_coords2 = surf_guide_coords2[init_surf_frag_indices2]
    init_surf_residue_numbers2 = surf_residue_numbers2[init_surf_frag_indices2]
    idx = 2
    logger.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords2.shape}')
    logger.debug(f'Found surface residue numbers {idx} with shape {surf_residue_numbers2.shape}')
    logger.debug(f'Found surface indices {idx} with shape {surf_i_indices2.shape}')
    logger.debug(f'Found {init_surf_residue_numbers2.shape[0]} initial surface {idx} fragments with type: {initial_surf_type2}')

    logger.info(f'Retrieved oligomer {idx} surface fragments and guide coordinates took '
                f'{time.time() - get_complete_surf_frags2_time_start:8f}s')

    # logger.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # logger.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # logger.debug('init_surf_residue_numbers2: %s' % slice_variable_for_log(init_surf_residue_numbers2))

    # Set up Building Block1
    get_complete_surf_frags1_time_start = time.time()
    surf_frags1 = model1.get_fragment_residues(residues=model1.surface_residues, fragment_db=job.fragment_db)

    # Calculate the initial match type by finding the predominant surface type
    fragment_content1 = np.bincount([surf_frag.i_type for surf_frag in surf_frags1])
    initial_surf_type1 = np.argmax(fragment_content1)
    init_surf_frags1 = [surf_frag for surf_frag in surf_frags1 if surf_frag.i_type == initial_surf_type1]
    # init_surf_guide_coords1 = np.array([surf_frag.guide_coords for surf_frag in init_surf_frags1])
    # init_surf_residue_numbers1 = np.array([surf_frag.number for surf_frag in init_surf_frags1])
    # surf_frag1_residues = [surf_frag.number for surf_frag in surf_frags1]
    idx = 1
    # logger.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords1.shape}')
    # logger.debug(f'Found surface residue numbers {idx} with shape {surf_residue_numbers1.shape}')
    # logger.debug(f'Found surface indices {idx} with shape {surf_i_indices1.shape}')
    logger.debug(f'Found {len(init_surf_frags1)} initial surface {idx} fragments with type: {initial_surf_type1}')
    # logger.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    # logger.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # logger.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # logger.debug('init_surf_residue_numbers2: %s' % slice_variable_for_log(init_surf_residue_numbers2))
    # logger.debug('init_surf_guide_coords1: %s' % slice_variable_for_log(init_surf_guide_coords1))
    # logger.debug('init_surf_residue_numbers1: %s' % slice_variable_for_log(init_surf_residue_numbers1))

    logger.info(f'Retrieved oligomer {idx} surface fragments and guide coordinates took '
                f'{time.time() - get_complete_surf_frags1_time_start:8f}s')

    #################################
    # Get component 1 ghost fragments and associated data from complete fragment database
    oligomer1_backbone_cb_tree = BallTree(model1.backbone_and_cb_coords)
    get_complete_ghost_frags1_time_start = time.time()
    ghost_frags_by_residue1 = \
        [frag.get_ghost_fragments(clash_tree=oligomer1_backbone_cb_tree) for frag in surf_frags1]

    complete_ghost_frags1: list[GhostFragment] = \
        [ghost for ghosts in ghost_frags_by_residue1 for ghost in ghosts]

    ghost_guide_coords1 = np.array([ghost_frag.guide_coords for ghost_frag in complete_ghost_frags1])
    ghost_rmsds1 = np.array([ghost_frag.rmsd for ghost_frag in complete_ghost_frags1])
    ghost_residue_numbers1 = np.array([ghost_frag.number for ghost_frag in complete_ghost_frags1])
    ghost_j_indices1 = np.array([ghost_frag.j_type for ghost_frag in complete_ghost_frags1])

    # Whether to use the overlap potential on the same component to filter ghost fragments
    if job.dock.contiguous_ghosts:
        # Identify surface/ghost frag overlap originating from the same oligomer
        # Set up the output array with the number of residues by the length of the max number of ghost fragments
        max_ghost_frags = max([len(ghost_frags) for ghost_frags in ghost_frags_by_residue1])
        number_or_surface_frags = len(surf_frags1)
        same_component_overlapping_ghost_frags = np.zeros((number_or_surface_frags, max_ghost_frags), dtype=int)
        # Set up the input array types with the various information needed for each pairwise check
        ghost_frag_type_by_residue = [[ghost.frag_type for ghost in residue_ghosts]
                                      for residue_ghosts in ghost_frags_by_residue1]
        ghost_frag_rmsds_by_residue = np.zeros_like(same_component_overlapping_ghost_frags, dtype=float)
        ghost_guide_coords_by_residue1 = np.zeros((number_or_surface_frags, max_ghost_frags, 3, 3))
        for idx, residue_ghosts in enumerate(ghost_frags_by_residue1):
            number_of_ghosts = len(residue_ghosts)
            # Set any viable index to 1 to distinguish between padding with 0
            same_component_overlapping_ghost_frags[idx, :number_of_ghosts] = 1
            ghost_frag_rmsds_by_residue[idx, :number_of_ghosts] = [ghost.rmsd for ghost in residue_ghosts]
            ghost_guide_coords_by_residue1[idx, :number_of_ghosts] = [ghost.guide_coords for ghost in residue_ghosts]
        # ghost_frag_rmsds_by_residue = np.array([[ghost.rmsd for ghost in residue_ghosts]
        #                                         for residue_ghosts in ghost_frags_by_residue1], dtype=object)
        # ghost_guide_coords_by_residue1 = np.array([[ghost.guide_coords for ghost in residue_ghosts]
        #                                            for residue_ghosts in ghost_frags_by_residue1], dtype=object)
        # surface_frag_residue_numbers = [residue.number for residue in surf_frags1]

        # Query for residue-residue distances for each surface fragment
        # surface_frag_cb_coords = np.concatenate([residue.cb_coords for residue in surf_frags1], axis=0)
        surface_frag_cb_coords = np.array([residue.cb_coords for residue in surf_frags1])
        model1_surface_cb_ball_tree = BallTree(surface_frag_cb_coords)
        residue_contact_query: list[list[int]] = \
            model1_surface_cb_ball_tree.query_radius(surface_frag_cb_coords, cb_distance)
        surface_frag_residue_indices = list(range(number_or_surface_frags))
        contacting_residue_pairs: list[tuple[int, int]] = \
            [(surface_frag_residue_indices[idx1], surface_frag_residue_indices[idx2])
             for idx2 in range(residue_contact_query.size) for idx1 in residue_contact_query[idx2]]

        # Separate residue-residue contacts into a unique set of residue pairs
        asymmetric_contacting_residue_pairs, found_pairs = [], []
        for residue_idx1, residue_idx2 in contacting_residue_pairs:
            # Add to unique set (asymmetric_contacting_residue_pairs) if we have never observed either
            if residue_idx1 == residue_idx2:
                continue  # We don't need to add because this check rules possibility out
            elif (residue_idx1, residue_idx2) not in found_pairs:
                # or (residue2, residue1) not in found_pairs
                # Checking both directions isn't required because the overlap would be the same...
                asymmetric_contacting_residue_pairs.append((residue_idx1, residue_idx2))
            # Add both pair orientations (1, 2) or (2, 1) regardless
            found_pairs.extend([(residue_idx1, residue_idx2), (residue_idx2, residue_idx1)])

        # Now, use asymmetric_contacting_residue_pairs indices to find the ghost_fragments that overlap for each residue
        # Todo, there are multiple indexing steps for residue_idx1/2 which only occur once if below code was used
        #  found_pairs = []
        #  for residue_idx2 in range(residue_contact_query.size):
        #      residue_ghost_frag_type2 = ghost_frag_type_by_residue[residue_idx2]
        #      residue_ghost_guide_coords2 = ghost_guide_coords_by_residue1[residue_idx2]
        #      residue_ghost_reference_rmsds2 = ghost_frag_rmsds_by_residue[residue_idx2]
        #      for residue_idx1 in residue_contact_query[residue_idx2]]
        #          # Check if the pair has been seen before. Work from second visited index(1) to first visited(2)
        #          if (residue_idx1, residue_idx2) in found_pairs:
        #              continue
        #          # else:
        #          found_pairs.append((residue_idx1, residue_idx2))
        #          type_bool_matrix = compute_ij_type_lookup(ghost_frag_type_by_residue[residue_idx1],
        #                                                    residue_ghost_frag_type2)
        #          # Separate indices for each type-matched, ghost fragment in the residue pair
        #          residue_idx1_ghost_indices, residue_idx2_ghost_indices = np.nonzero(type_bool_matrix)
        for residue_idx1, residue_idx2 in asymmetric_contacting_residue_pairs:
            # Check if each of the associated ghost frags have the same secondary structure type
            #   Fragment1
            # F T  F  F
            # R F  F  T
            # A F  F  F
            # G F  F  F
            # 2 T  T  F
            type_bool_matrix = compute_ij_type_lookup(ghost_frag_type_by_residue[residue_idx1],
                                                      ghost_frag_type_by_residue[residue_idx2])
            # Separate indices for each type-matched, ghost fragment in the residue pair
            residue_idx1_ghost_indices, residue_idx2_ghost_indices = np.nonzero(type_bool_matrix)
            # # Iterate over each matrix rox/column to pull out necessary guide coordinate pairs
            # # HERE v
            # # ij_matching_ghost1_indices = (type_bool_matrix * np.arange(type_bool_matrix.shape[0]))[type_bool_matrix]

            # These should pick out each instance of the guide_coords identified as overlapping by indexing bool type
            # Resulting instances should be present multiple times from residue_idxN_ghost_indices
            ghost_coords_residue1 = ghost_guide_coords_by_residue1[residue_idx1][residue_idx1_ghost_indices]
            ghost_coords_residue2 = ghost_guide_coords_by_residue1[residue_idx2][residue_idx2_ghost_indices]
            if len(ghost_coords_residue1) != len(residue_idx1_ghost_indices):
                raise IndexError('There was an issue indexing')
            ghost_reference_rmsds_residue1 = ghost_frag_rmsds_by_residue[residue_idx1][residue_idx1_ghost_indices]

            # Perform the overlap calculation and find indices with overlap
            overlapping_z_score = rmsd_z_score(ghost_coords_residue1,
                                               ghost_coords_residue2,
                                               ghost_reference_rmsds_residue1)  # , max_z_value=initial_z_value)
            same_component_overlapping_indices = np.flatnonzero(overlapping_z_score <= initial_z_value)
            # Increment indices of overlapping ghost fragments for each residue
            # same_component_overlapping_ghost_frags[
            #     residue_idx1, residue_idx1_ghost_indices[same_component_overlapping_indices]
            # ] += 1
            # This double counts as each is actually overlapping at the same location in space.
            # Just need one for initial overlap check...
            # Todo situation where more info needed?
            same_component_overlapping_ghost_frags[
                residue_idx2, residue_idx2_ghost_indices[same_component_overlapping_indices]
            ] += 1
            #       Ghost Fragments
            # S F - 1  0  0
            # U R - 0  0  0
            # R A - 0  0  0
            # F G - 1  1  0
            # Todo
            #  Could use the identified ghost fragments to further stitch together continuous ghost fragments...
            #  This proceeds by doing guide coord overlap between fragment adjacent CA (i.e. +-2 fragment length 5) and
            #  other identified ghost fragments
            #  Or guide coords cross product points along direction of persistence and euler angles are matched between
            #  all overlap ghosts to identify those which share "moment of extension"

        # Measure where greater than 0 as a value of 1 was included to mark viable fragment indices
        fragment_overlap_counts = \
            same_component_overlapping_ghost_frags[np.nonzero(same_component_overlapping_ghost_frags)]
        # logger.debug(f'fragment_overlap_counts.shape: {fragment_overlap_counts.shape}')
        viable_same_component_overlapping_ghost_frags = np.flatnonzero(fragment_overlap_counts > 1)
        # logger.debug(f'viable_same_component_overlapping_ghost_frags.shape: '
        #              f'{viable_same_component_overlapping_ghost_frags.shape}')
        # logger.debug(f'viable_same_component_overlapping_ghost_frags[:10]: '
        #              f'{viable_same_component_overlapping_ghost_frags[:10]}')
        # Prioritize search at those fragments which have same component, ghost fragment overlap
        initial_ghost_frags1 = \
            [complete_ghost_frags1[idx] for idx in viable_same_component_overlapping_ghost_frags.tolist()]
        init_ghost_guide_coords1 = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags1])
        init_ghost_rmsds1 = np.array([ghost_frag.rmsd for ghost_frag in initial_ghost_frags1])
    else:
        init_ghost_frag_indices1 = \
            [idx for idx, ghost_frag in enumerate(complete_ghost_frags1) if ghost_frag.j_type == initial_surf_type2]
        init_ghost_guide_coords1: np.ndarray = ghost_guide_coords1[init_ghost_frag_indices1]
        init_ghost_rmsds1: np.ndarray = ghost_rmsds1[init_ghost_frag_indices1]
        # init_ghost_residue_numbers1: np.ndarray = ghost_residue_numbers1[init_ghost_frag_indices1]

    idx = 1
    logger.debug(f'Found ghost guide coordinates {idx} with shape: {ghost_guide_coords1.shape}')
    logger.debug(f'Found ghost residue numbers {idx} with shape: {ghost_residue_numbers1.shape}')
    logger.debug(f'Found ghost indices {idx} with shape: {ghost_j_indices1.shape}')
    logger.debug(f'Found ghost rmsds {idx} with shape: {ghost_rmsds1.shape}')
    logger.debug(f'Found {init_ghost_guide_coords1.shape[0]} initial ghost {idx} fragments with type:'
                 f' {initial_surf_type2}')

    logger.info(f'Retrieved oligomer {idx}, {model1.name} ghost fragments and guide coordinates '
                f'took {time.time() - get_complete_ghost_frags1_time_start:8f}s')
    #################################
    # Implemented for Todd to work on C1 instances
    if job.only_write_frag_info:
        # Whether to write fragment information to a directory (useful for fragment based docking w/o Nanohedra)
        guide_file_ghost = os.path.join(out_dir, f'{model1.name}_ghost_coords.txt')
        with open(guide_file_ghost, 'w') as f:
            for coord_group in ghost_guide_coords1.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_ghost_idx = os.path.join(out_dir, f'{model1.name}_ghost_coords_index.txt')
        with open(guide_file_ghost_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_j_indices1.tolist())))
        guide_file_ghost_res_num = os.path.join(out_dir, f'{model1.name}_ghost_coords_residue_number.txt')
        with open(guide_file_ghost_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_residue_numbers1.tolist())))

        guide_file_surf = os.path.join(out_dir, f'{model2.name}_surf_coords.txt')
        with open(guide_file_surf, 'w') as f:
            for coord_group in surf_guide_coords2.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_surf_idx = os.path.join(out_dir, f'{model2.name}_surf_coords_index.txt')
        with open(guide_file_surf_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_i_indices2.tolist())))
        guide_file_surf_res_num = os.path.join(out_dir, f'{model2.name}_surf_coords_residue_number.txt')
        with open(guide_file_surf_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_residue_numbers2.tolist())))

        # write_fragment_pairs_as_accumulating_states(complete_ghost_frags1[:50],
        # input([len(frags) for frags in ghost_frags_by_residue1])
        start_slice = 0
        visualize_number = 15
        indices_of_interest = [0, 3, 5, 10]
        for idx, frags in enumerate(ghost_frags_by_residue1):
            if idx in indices_of_interest:
                number_of_fragments = len(frags)
                step_size = number_of_fragments // visualize_number
                # end_slice = start_slice * step_size
                residue_number = frags[0].number
                write_fragment_pairs_as_accumulating_states(
                    ghost_frags_by_residue1[idx][start_slice:number_of_fragments:step_size],
                    os.path.join(out_dir, f'{model1.name}_{residue_number}_paired_frags_'
                                          f'{start_slice}:{number_of_fragments}:{visualize_number}.pdb'))
        # write_fragment_pairs_as_accumulating_states(ghost_frags_by_residue1[3][20:40],
        #                                             os.path.join(out_dir, f'{model1.name}_frags4_{20}:{40}.pdb'))
        # write_fragment_pairs_as_accumulating_states(ghost_frags_by_residue1[5][20:40],
        #                                             os.path.join(out_dir, f'{model1.name}_frags6_{20}:{40}.pdb'))
        raise RuntimeError(f'Suspending operation of {model1.name}/{model2.name} after write')

    ij_type_match_lookup_table = compute_ij_type_lookup(ghost_j_indices1, surf_i_indices2)
    # Axis 0 is ghost frag, 1 is surface frag
    # ij_matching_ghost1_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[0]))[ij_type_match_lookup_table]
    # ij_matching_surf2_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[1])[:, None])[
    #         ij_type_match_lookup_table]
    # Todo apparently this should work to grab the flattened indices where there is overlap
    #  row_indices, column_indices = np.indices(ij_type_match_lookup_table.shape)
    #  # row index vary with ghost, column surf
    #  # transpose to index the first axis (axis=0) along the 1D row indices
    #  ij_matching_ghost1_indices = row_indices[ij_type_match_lookup_table.T]
    #  ij_matching_surf2_indices = column_indices[ij_type_match_lookup_table]
    #  >>> j = np.ones(22)
    #  >>> k = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    #  >>> k.shape
    #  (2, 22)
    #  >>> j[k]
    #  array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #          1., 1., 1., 1., 1., 1.]])
    #  This will allow pulling out the indices where there is overlap which may be useful
    #  for limiting scope of overlap checks

    # Get component 2 ghost fragments and associated data from complete fragment database
    bb_cb_coords2 = model2.backbone_and_cb_coords
    bb_cb_tree2 = BallTree(bb_cb_coords2)
    get_complete_ghost_frags2_time_start = time.time()
    complete_ghost_frags2 = []
    for frag in complete_surf_frags2:
        complete_ghost_frags2.extend(frag.get_ghost_fragments(clash_tree=bb_cb_tree2))
    init_ghost_frags2 = [ghost_frag for ghost_frag in complete_ghost_frags2 if ghost_frag.j_type == initial_surf_type1]
    init_ghost_guide_coords2 = np.array([ghost_frag.guide_coords for ghost_frag in init_ghost_frags2])
    # init_ghost_residue_numbers2 = np.array([ghost_frag.number for ghost_frag in init_ghost_frags2])
    # ghost_frag2_residues = [ghost_frag.number for ghost_frag in complete_ghost_frags2]

    idx = 2
    # logger.debug(f'Found ghost guide coordinates {idx} with shape {ghost_guide_coords2.shape}')
    # logger.debug(f'Found ghost residue numbers {idx} with shape {ghost_residue_numbers2.shape}')
    # logger.debug(f'Found ghost indices {idx} with shape {ghost_j_indices2.shape}')
    # logger.debug(f'Found ghost rmsds {idx} with shape {ghost_rmsds2.shape}')
    logger.debug(f'Found {len(init_ghost_guide_coords2)} initial ghost {idx} fragments with type {initial_surf_type1}')
    # logger.debug('init_ghost_guide_coords2: %s' % slice_variable_for_log(init_ghost_guide_coords2))
    # logger.debug('init_ghost_residue_numbers2: %s' % slice_variable_for_log(init_ghost_residue_numbers2))
    # Prepare precomputed arrays for fast pair lookup
    # ghost1_residue_array = np.repeat(init_ghost_residue_numbers1, len(init_surf_residue_numbers2))
    # ghost2_residue_array = np.repeat(init_ghost_residue_numbers2, len(init_surf_residue_numbers1))
    # surface1_residue_array = np.tile(init_surf_residue_numbers1, len(init_ghost_residue_numbers2))
    # surface2_residue_array = np.tile(init_surf_residue_numbers2, len(init_ghost_residue_numbers1))

    logger.info(f'Retrieved oligomer {idx}, {model2.name} ghost fragments and guide coordinates '
                f'took {time.time() - get_complete_ghost_frags2_time_start:8f}s')

    logger.info('Obtaining rotation/degeneracy matrices\n')

    rotation_steps = [rotation_step1, rotation_step2]
    """The number of degrees to increment the rotational degrees of freedom search for each model"""
    number_of_degens = []
    number_of_rotations = []
    rotation_matrices = []
    for idx, rotation_step in enumerate(rotation_steps, 1):
        if getattr(sym_entry, f'is_internal_rot{idx}'):  # if rotation step required
            if rotation_step is None:
                rotation_step = 3  # Set rotation_step to default
            # Set sym_entry.rotation_step
            setattr(sym_entry, f'rotation_step{idx}', rotation_step)
        else:
            if rotation_step:
                logger.warning(f"Specified rotation_step{idx} was ignored. Oligomer {idx} doesn't have rotational DOF")
            rotation_step = 1  # Set rotation step to 1

        rotation_steps[idx - 1] = rotation_step
        degeneracy_matrices = getattr(sym_entry, f'degeneracy_matrices{idx}')
        # Todo make reliant on scipy...Rotation
        # rotation_matrix = \
        #     scipy.spatial.transform.Rotation.from_euler('Z', [step * rotation_step
        #                                                       for step in range(number_of_steps)],
        #                                                 degrees=True).as_matrix()
        # rotations = \
        #     scipy.spatial.transform.Rotation.from_euler('Z', np.linspace(0, getattr(sym_entry, f'rotation_range{idx}'),
        #                                                                  number_of_steps),
        #                                                 degrees=True).as_matrix()
        # rot_degen_matrices = []
        # for idx in range(degeneracy_matrices):
        #    rot_degen_matrices = rotations * degeneracy_matrices[idx]
        # rot_degen_matrices = rotations * degeneracy_matrices
        # rotation_matrix = rotations.as_matrix()
        rotation_matrix = get_rot_matrices(rotation_step, 'z', getattr(sym_entry, f'rotation_range{idx}'))
        rot_degen_matrices = make_rotations_degenerate(rotation_matrix, degeneracy_matrices)
        logger.debug(f'Degeneracy shape for component {idx}: {degeneracy_matrices.shape}')
        logger.debug(f'Combined rotation/degeneracy shape for component {idx}: {rot_degen_matrices.shape}')
        number_of_degens.append(degeneracy_matrices.shape[0])
        # logger.debug(f'Rotation shape for component {idx}: {rot_degen_matrices.shape}')
        number_of_rotations.append(rot_degen_matrices.shape[0] // degeneracy_matrices.shape[0])
        rotation_matrices.append(rot_degen_matrices)

    set_mat1, set_mat2 = sym_entry.setting_matrix1, sym_entry.setting_matrix2

    # def check_forward_and_reverse(test_ghost_guide_coords, stack_rot1, stack_tx1,
    #                               test_surf_guide_coords, stack_rot2, stack_tx2,
    #                               reference_rmsds):
    #     """Debug forward versus reverse guide coordinate fragment matching
    #
    #     All guide_coords and reference_rmsds should be indexed to the same length and overlap
    #     """
    #     mismatch = False
    #     inv_set_mat1 = np.linalg.inv(set_mat1)
    #     for shift_idx in range(1):
    #         rot1 = stack_rot1[shift_idx]
    #         tx1 = stack_tx1[shift_idx]
    #         rot2 = stack_rot2[shift_idx]
    #         tx2 = stack_tx2[shift_idx]
    #
    #         tnsfmd_ghost_coords = transform_coordinate_sets(test_ghost_guide_coords,
    #                                                         rotation=rot1,
    #                                                         translation=tx1,
    #                                                         rotation2=set_mat1)
    #         tnsfmd_surf_coords = transform_coordinate_sets(test_surf_guide_coords,
    #                                                        rotation=rot2,
    #                                                        translation=tx2,
    #                                                        rotation2=set_mat2)
    #         int_euler_matching_ghost_indices, int_euler_matching_surf_indices = \
    #             euler_lookup.check_lookup_table(tnsfmd_ghost_coords, tnsfmd_surf_coords)
    #
    #         all_fragment_match = calculate_match(tnsfmd_ghost_coords[int_euler_matching_ghost_indices],
    #                                              tnsfmd_surf_coords[int_euler_matching_surf_indices],
    #                                              reference_rmsds[int_euler_matching_ghost_indices])
    #         high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
    #         high_qual_match_count = len(high_qual_match_indices)
    #         if high_qual_match_count < min_matched:
    #             logger.info(
    #                 f'\t{high_qual_match_count} < {min_matched} Which is Set as the Minimal Required Amount of '
    #                 f'High Quality Fragment Matches')
    #
    #         # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
    #         passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
    #         number_passing_overlaps = len(passing_overlaps_indices)
    #         logger.info(
    #             f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} '
    #             f'Matches Found in Complete Fragment Library')
    #
    #         # now try inverse
    #         inv_rot_mat1 = np.linalg.inv(rot1)
    #         tnsfmd_surf_coords_inv = transform_coordinate_sets(tnsfmd_surf_coords,
    #                                                            rotation=inv_set_mat1,
    #                                                            translation=tx1 * -1,
    #                                                            rotation2=inv_rot_mat1)
    #         int_euler_matching_ghost_indices_inv, int_euler_matching_surf_indices_inv = \
    #             euler_lookup.check_lookup_table(test_ghost_guide_coords, tnsfmd_surf_coords_inv)
    #
    #         all_fragment_match = calculate_match(test_ghost_guide_coords[int_euler_matching_ghost_indices_inv],
    #                                              tnsfmd_surf_coords_inv[int_euler_matching_surf_indices_inv],
    #                                              reference_rmsds[int_euler_matching_ghost_indices_inv])
    #         high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
    #         high_qual_match_count = len(high_qual_match_indices)
    #         if high_qual_match_count < min_matched:
    #             logger.info(
    #                 f'\tINV {high_qual_match_count} < {min_matched} Which is Set as the Minimal Required Amount'
    #                 f' of High Quality Fragment Matches')
    #
    #         # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
    #         passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
    #         number_passing_overlaps = len(passing_overlaps_indices)
    #
    #         logger.info(
    #             f'\t{high_qual_match_count} High Quality Fragments Out of {number_passing_overlaps} '
    #             f'Matches Found in Complete Fragment Library')
    #
    #         def investigate_mismatch():
    #             logger.info(f'Euler True ghost/surf indices forward and inverse don\'t match. '
    #                      f'Shapes: Forward={int_euler_matching_ghost_indices.shape}, '
    #                      f'Inverse={int_euler_matching_ghost_indices_inv.shape}')
    #             logger.debug(f'tnsfmd_ghost_coords.shape {tnsfmd_ghost_coords.shape}')
    #             logger.debug(f'tnsfmd_surf_coords.shape {tnsfmd_surf_coords.shape}')
    #             int_euler_matching_array = \
    #                 euler_lookup.check_lookup_table(tnsfmd_ghost_coords, tnsfmd_surf_coords, return_bool=True)
    #             int_euler_matching_array_inv = \
    #                 euler_lookup.check_lookup_table(test_ghost_guide_coords, tnsfmd_surf_coords_inv, return_bool=True)
    #             # Change the shape to allow for relation to guide_coords
    #             different = np.where(int_euler_matching_array != int_euler_matching_array_inv,
    #                                  True, False).reshape(tnsfmd_ghost_coords.shape[0], -1)
    #             ghost_indices, surface_indices = np.nonzero(different)
    #             logger.debug(f'different.shape {different.shape}')
    #
    #             different_ghosts = tnsfmd_ghost_coords[ghost_indices]
    #             different_surf = tnsfmd_surf_coords[surface_indices]
    #             tnsfmd_ghost_ints1, tnsfmd_ghost_ints2, tnsfmd_ghost_ints3 = \
    #                 euler_lookup.get_eulint_from_guides(different_ghosts)
    #             tnsfmd_surf_ints1, tnsfmd_surf_ints2, tnsfmd_surf_ints3 = \
    #                 euler_lookup.get_eulint_from_guides(different_surf)
    #             stacked_ints = np.stack([tnsfmd_ghost_ints1, tnsfmd_ghost_ints2, tnsfmd_ghost_ints3,
    #                                      tnsfmd_surf_ints1, tnsfmd_surf_ints2, tnsfmd_surf_ints3], axis=0).T
    #             logger.info(
    #                 f'The mismatched forward Euler ints are\n{[ints for ints in list(stacked_ints)[:10]]}\n')
    #
    #             different_ghosts_inv = test_ghost_guide_coords[ghost_indices]
    #             different_surf_inv = tnsfmd_surf_coords_inv[surface_indices]
    #             tnsfmd_ghost_ints_inv1, tnsfmd_ghost_ints_inv2, tnsfmd_ghost_ints_inv3 = \
    #                 euler_lookup.get_eulint_from_guides(different_ghosts_inv)
    #             tnsfmd_surf_ints_inv1, tnsfmd_surf_ints_inv2, tnsfmd_surf_ints_inv3 = \
    #                 euler_lookup.get_eulint_from_guides(different_surf_inv)
    #
    #             stacked_ints_inv = \
    #                 np.stack([tnsfmd_ghost_ints_inv1, tnsfmd_ghost_ints_inv2, tnsfmd_ghost_ints_inv3,
    #                           tnsfmd_surf_ints_inv1, tnsfmd_surf_ints_inv2, tnsfmd_surf_ints_inv3], axis=0).T
    #             logger.info(
    #                 f'The mismatched inverse Euler ints are\n{[ints for ints in list(stacked_ints_inv)[:10]]}\n')
    #
    #         if not np.array_equal(int_euler_matching_ghost_indices, int_euler_matching_ghost_indices_inv):
    #             mismatch = True
    #
    #         if not np.array_equal(int_euler_matching_surf_indices, int_euler_matching_surf_indices_inv):
    #             mismatch = True
    #
    #         if mismatch:
    #             investigate_mismatch()

    # if job.skip_transformation:
    #     transformation1 = unpickle(kwargs.get('transformation_file1'))
    #     full_rotation1, full_int_tx1, full_setting1, full_ext_tx1 = transformation1.values()
    #     transformation2 = unpickle(kwargs.get('transformation_file2'))
    #     full_rotation2, full_int_tx2, full_setting2, full_ext_tx2 = transformation2.values()
    #     # make arbitrary degen, rot, and tx counts
    #     degen_counts = [(idx, idx) for idx in range(1, len(full_rotation1) + 1)]
    #     rot_counts = [(idx, idx) for idx in range(1, len(full_rotation1) + 1)]
    #     tx_counts = list(range(1, len(full_rotation1) + 1))
    # else:

    # Set up internal translation parameters
    # zshift1/2 must be 2d array, thus the , 2:3].T instead of , 2].T
    # [:, None, 2] would also work
    if sym_entry.is_internal_tx1:  # add the translation to Z (axis=1)
        full_int_tx1 = []
        zshift1 = set_mat1[:, None, 2].T
    else:
        full_int_tx1 = zshift1 = None

    if sym_entry.is_internal_tx2:
        full_int_tx2 = []
        zshift2 = set_mat2[:, None, 2].T
    else:
        full_int_tx2 = zshift2 = None

    # Set up external translation parameters
    if sym_entry.unit_cell:
        full_optimal_ext_dof_shifts = []
        positive_indices = None
    else:
        full_optimal_ext_dof_shifts = []
        # Ensure we slice by nothing, as None alone creates a new axis
        positive_indices = slice(None)

    # Initialize the OptimalTx object
    logger.debug(f'zshift1 = {zshift1}, zshift2 = {zshift2}, max_z_value={initial_z_value:2f}')
    optimal_tx = resources.OptimalTx.from_dof(sym_entry.external_dof, zshift1=zshift1, zshift2=zshift2,
                                              max_z_value=initial_z_value)

    number_of_init_ghost = init_ghost_guide_coords1.shape[0]
    number_of_init_surf = init_surf_guide_coords2.shape[0]
    total_ghost_surf_combinations = number_of_init_ghost * number_of_init_surf
    # fragment_pairs = []
    rot_counts, degen_counts, tx_counts = [], [], []
    full_rotation1, full_rotation2 = [], []
    rotation_matrices1, rotation_matrices2 = rotation_matrices
    rotation_matrices_len1, rotation_matrices_len2 = rotation_matrices1.shape[0], rotation_matrices2.shape[0]
    number_of_rotations1, number_of_rotations2 = number_of_rotations
    # number_of_degens1, number_of_degens2 = number_of_degens

    # Perform Euler integer extraction for all rotations
    init_translation_time_start = time.time()
    # Rotate Oligomer1 surface and ghost guide coordinates using rotation_matrices1 and set_mat1
    # Must add a new axis so that the multiplication is broadcast
    ghost_frag1_guide_coords_rot_and_set = \
        transform_coordinate_sets(init_ghost_guide_coords1[None, :, :, :],
                                  rotation=rotation_matrices1[:, None, :, :],
                                  rotation2=set_mat1[None, None, :, :])
    # Unstack the guide coords to be shape (N, 3, 3)
    # eulerint_ghost_component1_1, eulerint_ghost_component1_2, eulerint_ghost_component1_3 = \
    #     euler_lookup.get_eulint_from_guides(ghost_frag1_guide_coords_rot_and_set.reshape((-1, 3, 3)))
    eulerint_ghost_component1 = \
        euler_lookup.get_eulint_from_guides_as_array(ghost_frag1_guide_coords_rot_and_set.reshape((-1, 3, 3)))

    # Next, for component 2
    surf_frags2_guide_coords_rot_and_set = \
        transform_coordinate_sets(init_surf_guide_coords2[None, :, :, :],
                                  rotation=rotation_matrices2[:, None, :, :],
                                  rotation2=set_mat2[None, None, :, :])
    # Reshape with the first axis (0) containing all the guide coordinate rotations stacked
    eulerint_surf_component2 = \
        euler_lookup.get_eulint_from_guides_as_array(surf_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))
    # eulerint_surf_component2_1, eulerint_surf_component2_2, eulerint_surf_component2_3 = \
    #     euler_lookup.get_eulint_from_guides(surf_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))

    # Reshape the reduced dimensional eulerint_components to again have the number_of_rotations length on axis 0,
    # the number of init_guide_coords on axis 1, and the 3 euler intergers on axis 2
    stacked_surf_euler_int2 = eulerint_surf_component2.reshape((rotation_matrices_len2, -1, 3))
    stacked_ghost_euler_int1 = eulerint_ghost_component1.reshape((rotation_matrices_len1, -1, 3))

    # stacked_surf_euler_int2_1 = eulerint_surf_component2_1.reshape((rotation_matrices_len2, -1))
    # stacked_surf_euler_int2_2 = eulerint_surf_component2_2.reshape((rotation_matrices_len2, -1))
    # stacked_surf_euler_int2_3 = eulerint_surf_component2_3.reshape((rotation_matrices_len2, -1))
    # stacked_ghost_euler_int1_1 = eulerint_ghost_component1_1.reshape((rotation_matrices_len1, -1))
    # stacked_ghost_euler_int1_2 = eulerint_ghost_component1_2.reshape((rotation_matrices_len1, -1))
    # stacked_ghost_euler_int1_3 = eulerint_ghost_component1_3.reshape((rotation_matrices_len1, -1))

    # The fragments being added to the pose are different than the fragments generated on the pose. This function
    # helped me elucidate that this was occurring
    # def check_offset_index(title):
    #     if pose.entities[-1].offset_index == 0:
    #         raise RuntimeError('The offset_index has changed to 0')
    #     else:
    #         print(f'{title} offset_index: {pose.entities[-1].offset_index}')
    if job.development:
        rotations_to_perform1 = min(rotation_matrices1.shape[0], 13)
        rotations_to_perform2 = min(rotation_matrices2.shape[0], 12)
        logger.critical(f'Development: Only sampling {rotations_to_perform1} by {rotations_to_perform2} rotations')
    else:
        rotations_to_perform1 = rotation_matrices1.shape[0]
        rotations_to_perform2 = rotation_matrices2.shape[0]

    # Todo resolve. Below uses eulerints
    # Get rotated oligomer1 ghost fragment, oligomer2 surface fragment guide coodinate pairs in the same Euler space
    for idx1 in range(rotations_to_perform1):
        rot1_count = idx1%number_of_rotations1 + 1
        degen1_count = idx1//number_of_rotations1 + 1
        rot_mat1 = rotation_matrices1[idx1]
        rotation_ghost_euler_ints1 = stacked_ghost_euler_int1[idx1]
        for idx2 in range(rotations_to_perform2):
            # Rotate oligomer2 surface and ghost fragment guide coordinates using rot_mat2 and set_mat2
            rot2_count = idx2%number_of_rotations2 + 1
            degen2_count = idx2//number_of_rotations2 + 1
            rot_mat2 = rotation_matrices2[idx2]

            logger.info(f'***** OLIGOMER 1: Degeneracy {degen1_count} Rotation {rot1_count} | '
                        f'OLIGOMER 2: Degeneracy {degen2_count} Rotation {rot2_count} *****')

            euler_start = time.time()
            # euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
            #     euler_lookup.lookup_by_euler_integers(stacked_surf_euler_int2_1[idx2],
            #                                           stacked_surf_euler_int2_2[idx2],
            #                                           stacked_surf_euler_int2_3[idx2],
            #                                           stacked_ghost_euler_int1_1[idx1],
            #                                           stacked_ghost_euler_int1_2[idx1],
            #                                           stacked_ghost_euler_int1_3[idx1],
            #                                           )
            euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
                euler_lookup.lookup_by_euler_integers_as_array(stacked_surf_euler_int2[idx2],
                                                               rotation_ghost_euler_ints1)
            # # euler_lookup.lookup_by_euler_integers_as_array(eulerint_ghost_component2.reshape(number_of_rotations2,
            # #                                                                                  1, 3),
            # #                                                eulerint_surf_component1.reshape(number_of_rotations1,
            # #                                                                                 1, 3))
            # Todo resolve. eulerints

    # Todo resolve. Below uses guide coords
    # # for idx1 in range(rotation_matrices):
    # # Iterating over more than 2 rotation matrix sets becomes hard to program dynamically owing to the permutations
    # # of the rotations and the application of the rotation/setting to each set of fragment information. It would be a
    # # bit easier if the same logic that is applied to the following routines, (similarity matrix calculation) putting
    # # the rotation of the second set of fragment information into the setting of the first by applying the inverse
    # # rotation and setting matrices to the second (or third...) set of fragments. Forget about this for now
    # init_time_start = time.time()
    # for idx1 in range(rotation_matrices1.shape[0]):  # min(rotation_matrices1.shape[0], 5)):  # Todo remove min
    #     # Rotate Oligomer1 Surface and Ghost Fragment Guide Coordinates using rot_mat1 and set_mat1
    #     rot1_count = idx1 % number_of_rotations1 + 1
    #     degen1_count = idx1 // number_of_rotations1 + 1
    #     rot_mat1 = rotation_matrices1[idx1]
    #     ghost_guide_coords_rot_and_set1 = \
    #         transform_coordinate_sets(init_ghost_guide_coords1, rotation=rot_mat1, rotation2=set_mat1)
    #     # surf_guide_coords_rot_and_set1 = \
    #     #     transform_coordinate_sets(init_surf_guide_coords1, rotation=rot_mat1, rotation2=set_mat1)
    #
    #     for idx2 in range(rotation_matrices2.shape[0]):  # min(rotation_matrices2.shape[0], 5)):  # Todo remove min
    #         # Rotate Oligomer2 Surface and Ghost Fragment Guide Coordinates using rot_mat2 and set_mat2
    #         rot2_count = idx2 % number_of_rotations2 + 1
    #         degen2_count = idx2 // number_of_rotations2 + 1
    #         rot_mat2 = rotation_matrices2[idx2]
    #         surf_guide_coords_rot_and_set2 = \
    #             transform_coordinate_sets(init_surf_guide_coords2, rotation=rot_mat2, rotation2=set_mat2)
    #         # ghost_guide_coords_rot_and_set2 = \
    #         #     transform_coordinate_sets(init_ghost_guide_coords2, rotation=rot_mat2, rotation2=set_mat2)
    #
    #         logger.info(f'***** OLIGOMER 1: Degeneracy {degen1_count} Rotation {rot1_count} | '
    #                     f'OLIGOMER 2: Degeneracy {degen2_count} Rotation {rot2_count} *****')
    #
    #         euler_start = time.time()
    #         # First returned variable has indices increasing 0,0,0,0,1,1,1,1,1,2,2,2,3,...
    #         # Second returned variable has indices increasing 2,3,4,14,...
    #         euler_matched_surf_indices2, euler_matched_ghost_indices1 = \
    #             euler_lookup.check_lookup_table(surf_guide_coords_rot_and_set2,
    #                                             ghost_guide_coords_rot_and_set1)
    #         # euler_matched_ghost_indices_rev2, euler_matched_surf_indices_rev1 = \
    #         #     euler_lookup.check_lookup_table(ghost_guide_coords_rot_and_set2,
    #         #                                     surf_guide_coords_rot_and_set1)
    # Todo resolve. guide coords
    #  Timings on these from improved protocola shows about similar times to euler_lookup and calculate_overlap
    #  even with vastly different scales of the arrays. This ignores the fact that calculate_overlap uses a
    #  number of indexing steps including making the ij_match array formation, indexing against the ghost and
    #  surface arrays, the rmsd_reference construction
    #  Given the lookups sort of irrelevance to the scoring (given very poor alignment), I could remove that
    #  step if it interfered with differentiability
            logger.debug(f'\tEuler search took {time.time() - euler_start:8f}s for '
                         f'{total_ghost_surf_combinations} ghost/surf pairs')

            # Ensure pairs are similar between euler_matched_surf_indices2 and euler_matched_ghost_indices_rev2
            # by indexing the residue_numbers
            # forward_reverse_comparison_start = time.time()
            # # logger.debug(f'Euler indices forward, index 0: {euler_matched_surf_indices2[:10]}')
            # forward_surface_numbers2 = init_surf_residue_numbers2[euler_matched_surf_indices2]
            # # logger.debug(f'Euler indices forward, index 1: {euler_matched_ghost_indices1[:10]}')
            # forward_ghosts_numbers1 = init_ghost_residue_numbers1[euler_matched_ghost_indices1]
            # # logger.debug(f'Euler indices reverse, index 0: {euler_matched_ghost_indices_rev2[:10]}')
            # reverse_ghosts_numbers2 = init_ghost_residue_numbers2[euler_matched_ghost_indices_rev2]
            # # logger.debug(f'Euler indices reverse, index 1: {euler_matched_surf_indices_rev1[:10]}')
            # reverse_surface_numbers1 = init_surf_residue_numbers1[euler_matched_surf_indices_rev1]

            # Make an index indicating where the forward and reverse euler lookups have the same residue pairs
            # Important! This method only pulls out initial fragment matches that go both ways, i.e. component1
            # surface (type1) matches with component2 ghost (type1) and vice versa, so the expanded checks of
            # for instance the surface loop (i type 3,4,5) with ghost helical (i type 1) matches is completely
            # unnecessary during euler look up as this will never be included
            # Also, this assumes that the ghost fragment display is symmetric, i.e. 1 (i) 1 (j) 10 (K) has an
            # inverse transform at 1 (i) 1 (j) 230 (k) for instance

            # prior = 0
            # number_overlapping_pairs = euler_matched_ghost_indices1.shape[0]
            # possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)
            # # Residue numbers are in order for forward_surface_numbers2 and reverse_ghosts_numbers2
            # for residue in init_surf_residue_numbers2:
            #     # Where the residue number of component 2 is equal pull out the indices
            #     forward_index = np.flatnonzero(forward_surface_numbers2 == residue)
            #     reverse_index = np.flatnonzero(reverse_ghosts_numbers2 == residue)
            #     # Next, use residue number indices to search for the same residue numbers in the extracted pairs
            #     # The output array slice is only valid if the forward_index is the result of
            #     # forward_surface_numbers2 being in ascending order, which for check_lookup_table is True
            #     current = prior + forward_index.shape[0]
            #     possible_overlaps[prior:current] = \
            #         np.in1d(forward_ghosts_numbers1[forward_index], reverse_surface_numbers1[reverse_index])
            #     prior = current

            # # Use for residue number debugging
            # possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)

            # forward_ghosts_numbers1[possible_overlaps]
            # forward_surface_numbers2[possible_overlaps]

            # indexing_possible_overlap_time = time.time() - indexing_possible_overlap_start

            # number_of_successful = possible_overlaps.sum()
            # logger.info(f'\tIndexing {number_overlapping_pairs * euler_matched_surf_indices2.shape[0]} '
            #             f'possible overlap pairs found only {number_of_successful} possible out of '
            #             f'{number_overlapping_pairs} (took {time.time() - forward_reverse_comparison_start:8f}s)')

            # Get optimal shift parameters for initial (Ghost Fragment, Surface Fragment) guide coordinate pairs
            # Take the boolean index of the indices
            # possible_ghost_frag_indices = euler_matched_ghost_indices1[possible_overlaps]
            # # possible_surf_frag_indices = euler_matched_surf_indices2[possible_overlaps]

            # reference_rmsds = init_ghost_rmsds1[possible_ghost_frag_indices]
            # passing_ghost_coords = ghost_guide_coords_rot_and_set1[possible_ghost_frag_indices]
            # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2[possible_overlaps]]
            # # Todo these are from Guides
            # passing_ghost_coords = ghost_guide_coords_rot_and_set1[euler_matched_ghost_indices1]
            # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2]
            # # Todo these are from Guides
            # Todo debug With EulerInteger calculation
            passing_ghost_coords = ghost_frag1_guide_coords_rot_and_set[idx1, euler_matched_ghost_indices1]
            # passing_ghost_coords = transform_coordinate_sets(init_ghost_guide_coords1[euler_matched_ghost_indices1],
            #                                                  rotation=rot_mat1, rotation2=set_mat1)
            passing_surf_coords = surf_frags2_guide_coords_rot_and_set[idx2, euler_matched_surf_indices2]
            # passing_surf_coords = transform_coordinate_sets(init_surf_guide_coords2[euler_matched_surf_indices2],
            #                                                 rotation=rot_mat2, rotation2=set_mat2)
            # Todo debug With EulerInteger calculation
            reference_rmsds = init_ghost_rmsds1[euler_matched_ghost_indices1]

            optimal_shifts_start = time.time()
            transform_passing_shifts = \
                optimal_tx.solve_optimal_shifts(passing_ghost_coords, passing_surf_coords, reference_rmsds)
            optimal_shifts_time = time.time() - optimal_shifts_start

            pre_cluster_passing_shifts = transform_passing_shifts.shape[0]
            if pre_cluster_passing_shifts == 0:
                # logger.debug(f'optimal_shifts length: {len(optimal_shifts)}')
                # logger.debug(f'transform_passing_shifts shape: {transform_passing_shifts.shape[0]}')
                logger.info(f'\tNo transforms were found passing optimal shift criteria '
                            f'(took {optimal_shifts_time:8f}s)')
                continue
            elif cluster_transforms:
                cluster_time_start = time.time()
                translation_cluster = \
                    DBSCAN(eps=translation_epsilon, min_samples=min_matched).fit(transform_passing_shifts)
                transform_passing_shifts = transform_passing_shifts[translation_cluster.labels_ != outlier]
                cluster_time = time.time() - cluster_time_start
                logger.debug(f'Clustering {pre_cluster_passing_shifts} possible transforms (took {cluster_time:8f}s)')
            # else:  # Use all translations
            #     pass

            if sym_entry.unit_cell:
                # Must take the optimal_ext_dof_shifts and multiply the column number by the corresponding row
                # in the sym_entry.external_dof#
                # optimal_ext_dof_shifts[0] scalar * sym_entry.group_external_dof[0] (1 row, 3 columns)
                # Repeat for additional DOFs, then add all up within each row.
                # For a single DOF, multiplication won't matter as only one matrix element will be available
                #
                # Must find positive indices before external_dof1 multiplication in case negatives there
                positive_indices = np.flatnonzero(np.all(transform_passing_shifts[:, :sym_entry.number_dof_external] >= 0,
                                                         axis=1))
                number_passing_shifts = positive_indices.shape[0]
                optimal_ext_dof_shifts = np.zeros((number_passing_shifts, 3), dtype=float)
                optimal_ext_dof_shifts[:, :sym_entry.number_dof_external] = \
                    transform_passing_shifts[positive_indices, :sym_entry.number_dof_external]
                # ^ I think for the sake of cleanliness, I need to make this matrix

                full_optimal_ext_dof_shifts.append(optimal_ext_dof_shifts)
            else:
                number_passing_shifts = transform_passing_shifts.shape[0]

            # logger.debug(f'\tFound {number_passing_shifts} transforms after clustering from '
            #              f'{pre_cluster_passing_shifts} possible transforms (took '
            #              f'{time.time() - cluster_time_start:8f}s)')

            # Prepare the transformation parameters for storage in full transformation arrays
            # Use of [:, None] transforms the array into an array with each internal dof sored as a scalar in
            # axis 1 and each successive index along axis 0 as each passing shift

            # Stack each internal parameter along with a blank vector, this isolates the tx vector along z axis
            if full_int_tx1 is not None:
                # stacked_internal_tx_vectors1 = np.zeros((number_passing_shifts, 3), dtype=float)
                # stacked_internal_tx_vectors1[:, -1] = transform_passing_shifts[:, sym_entry.number_dof_external]
                # internal_tx_params1 = transform_passing_shifts[:, None, sym_entry.number_dof_external]
                # stacked_internal_tx_vectors1 = np.hstack((blank_vector, blank_vector, internal_tx_params1))
                # Store transformation parameters, indexing only those that are positive in the case of lattice syms
                full_int_tx1.extend(transform_passing_shifts[positive_indices,
                                                             sym_entry.number_dof_external].tolist())

            if full_int_tx2 is not None:
                # stacked_internal_tx_vectors2 = np.zeros((number_passing_shifts, 3), dtype=float)
                # stacked_internal_tx_vectors2[:, -1] = transform_passing_shifts[:, sym_entry.number_dof_external + 1]
                # internal_tx_params2 = transform_passing_shifts[:, None, sym_entry.number_dof_external + 1]
                # stacked_internal_tx_vectors2 = np.hstack((blank_vector, blank_vector, internal_tx_params2))
                # Store transformation parameters, indexing only those that are positive in the case of lattice syms
                full_int_tx2.extend(transform_passing_shifts[positive_indices,
                                                             sym_entry.number_dof_external + 1].tolist())

            # full_int_tx1.append(stacked_internal_tx_vectors1[positive_indices])
            # full_int_tx2.append(stacked_internal_tx_vectors2[positive_indices])
            full_rotation1.append(np.tile(rot_mat1, (number_passing_shifts, 1, 1)))
            full_rotation2.append(np.tile(rot_mat2, (number_passing_shifts, 1, 1)))

            degen_counts.extend([(degen1_count, degen2_count) for _ in range(number_passing_shifts)])
            rot_counts.extend([(rot1_count, rot2_count) for _ in range(number_passing_shifts)])
            tx_counts.extend(list(range(1, number_passing_shifts + 1)))
            logger.debug(f'\tOptimal shift search took {optimal_shifts_time:8f}s for '
                         f'{euler_matched_ghost_indices1.shape[0]} guide coordinate pairs')
            logger.info(f'\t{number_passing_shifts if number_passing_shifts else "No"} initial interface '
                        f'match{"es" if number_passing_shifts != 1 else ""} found '
                        f'(took {time.time() - euler_start:8f}s)')

            # # Todo remove debug
            # # tx_param_list = []
            # init_pass_ghost_numbers = init_ghost_residue_numbers1[possible_ghost_frag_indices]
            # init_pass_surf_numbers = init_surf_residue_numbers2[possible_surf_frag_indices]
            # for index in range(passing_ghost_coords.shape[0]):
            #     o = OptimalTxOLD(set_mat1, set_mat2, sym_entry.is_internal_tx1, sym_entry.is_internal_tx2,
            #                      reference_rmsds[index],
            #                      passing_ghost_coords[index], passing_surf_coords[index], sym_entry.external_dof)
            #     o.solve_optimal_shift()
            #     if o.get_zvalue() <= initial_z_value:
            #         # logger.debug(f'overlap found at ghost/surf residue pair {init_pass_ghost_numbers[index]} | '
            #         #              f'{init_pass_surf_numbers[index]}')
            #         fragment_pairs.append((init_pass_ghost_numbers[index], init_pass_surf_numbers[index],
            #                                initial_ghost_frags1[possible_ghost_frag_indices[index]].guide_coords))
            #         all_optimal_shifts = o.get_all_optimal_shifts()  # [OptimalExtDOFShifts, OptimalIntDOFShifts]
            #         tx_param_list.append(all_optimal_shifts)
            #
            # logger.info(f'\t{len(tx_param_list) if tx_param_list else "No"} Initial Interface Fragment '
            #             f'Matches Found')
            # tx_param_list = np.array(tx_param_list)
            # logger.debug(f'Equality of vectorized versus individual tx array: '
            #              f'{np.all(tx_param_list == transform_passing_shifts)}')
            # logger.debug(f'ALLCLOSE Equality of vectorized versus individual tx array: '
            #              f'{np.allclose(tx_param_list, transform_passing_shifts)}')
            # check_forward_and_reverse(init_ghost_guide_coords1[possible_ghost_frag_indices],
            #                           [rot_mat1], stacked_internal_tx_vectors1,
            #                           init_surf_guide_coords2[euler_matched_surf_indices2[possible_overlaps]],
            #                           [rot_mat2], stacked_internal_tx_vectors2,
            #                           reference_rmsds)
            # # Todo remove debug

    ##############
    # Here represents an important break in the execution of this code.
    # Below create vectors for cluster transformations
    # Then we perform asu clash testing, scoring, and finally symmetric clash testing
    ##############
    if sym_entry.unit_cell:
        # optimal_ext_dof_shifts[:, :, None] <- None expands the axis to make multiplication accurate
        full_optimal_ext_dof_shifts = np.concatenate(full_optimal_ext_dof_shifts, axis=0)
        unsqueezed_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts[:, :, None]
        full_ext_tx1 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof1, axis=-2)
        full_ext_tx2 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof2, axis=-2)
        # full_ext_tx1 = np.concatenate(full_ext_tx1, axis=0)  # .sum(axis=-2)
        # full_ext_tx2 = np.concatenate(full_ext_tx2, axis=0)  # .sum(axis=-2)
        full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
    else:
        # stacked_external_tx1, stacked_external_tx2 = None, None
        full_ext_tx1 = full_ext_tx2 = full_optimal_ext_dof_shifts = None
        # full_optimal_ext_dof_shifts = list(repeat(None, number_passing_shifts))
        full_ext_tx_sum = None

    # fragment_pairs = np.array(fragment_pairs)
    # Make full, numpy vectorized transformations overwriting individual variables for memory management
    full_rotation1 = np.concatenate(full_rotation1, axis=0)
    full_rotation2 = np.concatenate(full_rotation2, axis=0)
    starting_transforms = full_rotation1.shape[0]

    if not starting_transforms:  # There were no successful transforms
        logger.warning(f'No optimal translations found. Terminating {building_blocks} docking')
        return []
        # ------------------ TERM ------------------------
    else:
        logger.info(f'Initial optimal translation search found {starting_transforms} total transforms '
                    f'in {time.time() - init_translation_time_start:8f}s')

    if sym_entry.is_internal_tx1:
        stacked_internal_tx_vectors1 = np.zeros((starting_transforms, 3), dtype=float)
        # Add the translation to Z (axis=1)
        stacked_internal_tx_vectors1[:, -1] = full_int_tx1
        full_int_tx1 = stacked_internal_tx_vectors1
        del stacked_internal_tx_vectors1

    if sym_entry.is_internal_tx2:
        stacked_internal_tx_vectors2 = np.zeros((starting_transforms, 3), dtype=float)
        # Add the translation to Z (axis=1)
        stacked_internal_tx_vectors2[:, -1] = full_int_tx2
        full_int_tx2 = stacked_internal_tx_vectors2
        del stacked_internal_tx_vectors2

    # full_int_tx1 = np.concatenate(full_int_tx1, axis=0)
    # full_int_tx2 = np.concatenate(full_int_tx2, axis=0)
    # starting_transforms = len(full_int_tx1)
    # logger.debug(f'Shape of full_rotation1: {full_rotation1.shape}')
    # logger.debug(f'Shape of full_rotation2: {full_rotation2.shape}')
    # logger.debug(f'Shape of full_int_tx1: {full_int_tx1.shape}')
    # logger.debug(f'Shape of full_int_tx2: {full_int_tx2.shape}')

    # Make inverted transformations
    inv_setting1 = np.linalg.inv(set_mat1)
    full_inv_rotation1 = np.linalg.inv(full_rotation1)
    _full_rotation2 = full_rotation2.copy()
    if sym_entry.is_internal_tx1:
        # Invert by multiplying by -1
        full_int_tx_inv1 = full_int_tx1 * -1
    else:
        full_int_tx_inv1 = None
    if sym_entry.is_internal_tx2:
        _full_int_tx2 = full_int_tx2.copy()
    else:
        _full_int_tx2 = None

    # Define functions for removing indices from the active transformation arrays

    def remove_non_viable_indices_inverse(passing_indices: np.ndarray | list[int]):
        """Responsible for updating docking intermediate transformation parameters for inverse transform operations
        These include: full_inv_rotation1, _full_rotation2, full_int_tx_inv1, _full_int_tx2, and full_ext_tx_sum
        """
        nonlocal full_inv_rotation1, _full_rotation2, full_int_tx_inv1, _full_int_tx2, full_ext_tx_sum
        full_inv_rotation1 = full_inv_rotation1[passing_indices]
        _full_rotation2 = _full_rotation2[passing_indices]
        if sym_entry.is_internal_tx1:
            full_int_tx_inv1 = full_int_tx_inv1[passing_indices]
        if sym_entry.is_internal_tx2:
            _full_int_tx2 = _full_int_tx2[passing_indices]
        if sym_entry.unit_cell:
            full_ext_tx_sum = full_ext_tx_sum[passing_indices]

    def remove_non_viable_indices(passing_indices: np.ndarray | list[int]):
        """Responsible for updating docking transformation parameters for transform operations. These include:
        full_rotation1, full_rotation2, full_int_tx1, full_int_tx2, full_optimal_ext_dof_shifts, full_ext_tx1, and
        full_ext_tx2
        """
        nonlocal full_rotation1, full_rotation2, full_int_tx1, full_int_tx2
        full_rotation1 = full_rotation1[passing_indices]
        full_rotation2 = full_rotation2[passing_indices]
        if sym_entry.is_internal_tx1:
            full_int_tx1 = full_int_tx1[passing_indices]
        if sym_entry.is_internal_tx2:
            full_int_tx2 = full_int_tx2[passing_indices]

        if sym_entry.unit_cell:
            nonlocal full_optimal_ext_dof_shifts, full_ext_tx1, full_ext_tx2
            full_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts[passing_indices]
            full_ext_tx1 = full_ext_tx1[passing_indices]
            full_ext_tx2 = full_ext_tx2[passing_indices]

    # Find the clustered transformations to expedite search of ASU clashing
    # Todo
    #  Can I use cluster.cluster_transformation_pairs distance graph to provide feedback on other aspects of the dock?
    #  seems that I could use the distances to expedite clashing checks, especially for more time consuming expansion
    #  checks such as the full material...

    if cluster_transforms:
        clustering_start = time.time()
        # Todo tune DBSCAN epsilon (distance) to be reflective of the data, should be related to radius in
        #  NearestNeighbors but smaller by some amount. Ideal amount would be the distance between two transformed
        #  guide coordinate sets of a similar tx and a 3 degree step of rotation.
        # Must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
        transform_neighbor_tree, transform_cluster = \
            protocols.cluster.cluster_transformation_pairs(dict(rotation=full_rotation1,
                                                                translation=None if full_int_tx1 is None
                                                                else full_int_tx1[:, None, :],
                                                                rotation2=set_mat1,
                                                                translation2=None if full_ext_tx1 is None
                                                                else full_ext_tx1[:, None, :]),
                                                           dict(rotation=full_rotation2,
                                                                translation=None if full_int_tx2 is None
                                                                else full_int_tx2[:, None, :],
                                                                rotation2=set_mat2,
                                                                translation2=None if full_ext_tx2 is None
                                                                else full_ext_tx2[:, None, :]),
                                                           minimum_members=min_matched)
        # cluster_representative_indices, cluster_labels = \
        #     find_cluster_representatives(transform_neighbor_tree, transform_cluster)
        del transform_neighbor_tree
        # representative_labels = cluster_labels[cluster_representative_indices]
        # Todo?
        #  _, cluster_labels = find_cluster_representatives(transform_neighbor_tree, transform_cluster)
        cluster_labels = transform_cluster.labels_
        # logger.debug(f'Shape of cluster_labels: {cluster_labels.shape}')
        passing_transforms = cluster_labels != -1
        sufficiently_dense_indices = np.flatnonzero(passing_transforms)
        number_of_dense_transforms = len(sufficiently_dense_indices)

        logger.info(f'After clustering, {starting_transforms - number_of_dense_transforms} are missing the minimum '
                    f'number of close transforms to be viable. {number_of_dense_transforms} transforms '
                    f'remain (took {time.time() - clustering_start:8f}s)')
        if not number_of_dense_transforms:  # There were no successful transforms
            logger.warning(f'No viable transformations found. Terminating {building_blocks} docking')
            return []
        # ------------------ TERM ------------------------
        # Update the transformation array and counts with the sufficiently_dense_indices
        # Remove non-viable transforms by indexing sufficiently_dense_indices
        remove_non_viable_indices_inverse(sufficiently_dense_indices)
    else:
        sufficiently_dense_indices = np.arange(starting_transforms)
        number_of_dense_transforms = starting_transforms

    # Transform coords to query for clashes
    # Set up chunks of coordinate transforms for clash testing
    check_clash_coords_start = time.time()
    memory_constraint = psutil.virtual_memory().available
    # Assume each element is np.float64
    element_memory = 8  # where each element is np.float64
    # guide_coords_elements = 9  # For a single guide coordinate with shape (3, 3)
    # coords_multiplier = 2
    number_of_elements_available = memory_constraint / element_memory
    model_elements = prod(bb_cb_coords2.shape)
    # total_elements_required = model_elements * number_of_dense_transforms
    # Start with the assumption that all tested clashes are clashing
    asu_clash_counts = np.ones(number_of_dense_transforms)
    clash_vect = [clash_dist]
    # The batch_length indicates how many models could fit in the allocated memory. Using floor division to get integer
    # Reduce scale by factor of divisor to be safe
    start_divisor = 16
    batch_length = int(number_of_elements_available // model_elements // start_divisor)

    # Setup function that must be performed before the function isexecuted
    def np_tile_wrap(length: int, coords: np.ndarray, *args, **kwargs):
        return dict(query_points=np.tile(coords, (length, 1, 1)))

    # Create the balltree clash check as a batched function
    @resources.ml.batch_calculation(size=number_of_dense_transforms, batch_length=batch_length, setup=np_tile_wrap,
                                    compute_failure_exceptions=(np.core._exceptions._ArrayMemoryError,))
    def check_tree_for_query_overlap(batch_slice: slice,
                                     binarytree: BinaryTree = None, query_points: np.ndarray = None,
                                     rotation: np.ndarray = None, translation: np.ndarray = None,
                                     rotation2: np.ndarray = None, translation2: np.ndarray = None,
                                     rotation3: np.ndarray = None, translation3: np.ndarray = None,
                                     rotation4: np.ndarray = None, translation4: np.ndarray = None) \
            -> dict[str, list]:
        """Check for overlapping coordinates between a BinaryTree and a collection of query_points.
        Transform the query over multiple iterations

        Args:
            binarytree: The tree to check all queries against
            query_points: The points to transform, then query against the tree
            rotation:
            translation:
            rotation2:
            translation2:
            rotation3:
            translation3:
            rotation4:
            translation4:

        Returns:
            The number of overlaps found at each transformed query point as a dictionary
        """
        # These variables are accessed from within the resources.ml.batch_calculation scope
        # nonlocal actual_batch_length, batch_slice
        _rotation = rotation[batch_slice]
        # actual_batch_length = batch_slice.stop - batch_slice.start
        actual_batch_length = _rotation.shape[0]
        # Transform the coordinates
        # Todo for performing broadcasting of this operation
        #  s_broad = np.matmul(tiled_coords2[None, :, None, :], _full_rotation2[:, None, :, :])
        #  produces a shape of (_full_rotation2.shape[0], tiled_coords2.shape[0], 1, 3)
        #  inverse_transformed_model2_tiled_coords = transform_coordinate_sets(transform_coordinate_sets()).squeeze()
        transformed_query_points = \
            transform_coordinate_sets(
                transform_coordinate_sets(query_points[:actual_batch_length],  # Slice ensures same size
                                          rotation=_rotation,
                                          translation=None if translation is None
                                          else translation[batch_slice, None, :],
                                          rotation2=rotation2,  # setting matrix, no slice
                                          translation2=None if translation2 is None
                                          else translation2[batch_slice, None, :]),
                rotation=rotation3,  # setting matrix, no slice
                translation=None if translation3 is None else translation3[batch_slice, None, :],
                rotation2=rotation4[batch_slice],
                translation2=None if translation4 is None else translation4[batch_slice, None, :])

        overlap_counts = \
            [binarytree.two_point_correlation(transformed_query_points[idx], clash_vect)[0]
             for idx in range(actual_batch_length)]

        return {'overlap_counts': overlap_counts}

    # Using the inverse transform of the model2 backbone and cb (surface fragment) coordinates, check for clashes
    # with the model1 backbone and cb coordinates BinaryTree
    ball_tree_kwargs = dict(binarytree=oligomer1_backbone_cb_tree,
                            rotation=_full_rotation2, translation=_full_int_tx2,
                            rotation2=set_mat2, translation2=full_ext_tx_sum,
                            rotation3=inv_setting1, translation3=full_int_tx_inv1,
                            rotation4=full_inv_rotation1)

    overlap_return = check_tree_for_query_overlap(**ball_tree_kwargs,
                                                  return_containers={'overlap_counts': asu_clash_counts},
                                                  setup_args=(bb_cb_coords2,))
    # Extract the data
    asu_clash_counts = overlap_return['overlap_counts']
    # Find those indices where the asu_clash_counts is not zero (inverse of nonzero by using the array == 0)
    asu_is_viable_indices = np.flatnonzero(asu_clash_counts == 0)
    number_non_clashing_transforms = asu_is_viable_indices.shape[0]
    # Update the passing_transforms
    # passing_transforms contains all the transformations that are still passing
    # index the previously passing indices (sufficiently_dense_indices) by new pasing indices (asu_is_viable_indices)
    # and set each of these indices to 1 (True)
    # passing_transforms[sufficiently_dense_indices[asu_is_viable_indices]] = 1
    logger.info(f"Clash testing for identified poses found {number_non_clashing_transforms} viable ASU's out of "
                f'{number_of_dense_transforms} (took {time.time() - check_clash_coords_start:8f}s)')

    if not number_non_clashing_transforms:  # There were no successful asus that don't clash
        logger.warning(f'No viable asymmetric units. Terminating {building_blocks} docking')
        return []
    # ------------------ TERM ------------------------
    # Remove non-viable transforms by indexing asu_is_viable_indices
    remove_non_viable_indices_inverse(asu_is_viable_indices)

    # logger.debug('Checking rotation and translation fidelity after removing non-viable asu indices')
    # check_forward_and_reverse(ghost_guide_coords1,
    #                           full_rotation1, full_int_tx_inv1,
    #                           surf_guide_coords2,
    #                           _full_rotation2, _full_int_tx2,
    #                           ghost_rmsds1)

    #################
    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    # alternative route to measure clashes of each transform. Move copies of component2 to interact with model1 ORIGINAL
    int_cb_and_frags_start = time.time()
    # Transform the CB coords of oligomer 2 to each identified transformation
    # Transforming only surface frags has large speed benefits from not having to transform all ghosts
    inverse_transformed_model2_tiled_cb_coords = \
        transform_coordinate_sets(transform_coordinate_sets(np.tile(model2.cb_coords,
                                                                    (number_non_clashing_transforms, 1, 1)),
                                                            rotation=_full_rotation2,
                                                            translation=None if full_int_tx2 is None
                                                            else _full_int_tx2[:, None, :],
                                                            rotation2=set_mat2,
                                                            translation2=None if sym_entry.unit_cell is None
                                                            else full_ext_tx_sum[:, None, :]),
                                  rotation=inv_setting1,
                                  translation=None if full_int_tx1 is None else full_int_tx_inv1[:, None, :],
                                  rotation2=full_inv_rotation1)

    # Transform the surface guide coords of oligomer 2 to each identified transformation
    # Makes a shape (full_rotations.shape[0], surf_guide_coords.shape[0], 3, 3)
    inverse_transformed_surf_frags2_guide_coords = \
        transform_coordinate_sets(transform_coordinate_sets(surf_guide_coords2[None, :, :, :],
                                                            rotation=_full_rotation2[:, None, :, :],
                                                            translation=None if full_int_tx2 is None
                                                            else _full_int_tx2[:, None, None, :],
                                                            rotation2=set_mat2[None, None, :, :],
                                                            translation2=None if sym_entry.unit_cell is None
                                                            else full_ext_tx_sum[:, None, None, :]),
                                  rotation=inv_setting1[None, None, :, :],
                                  translation=None if full_int_tx1 is None else full_int_tx_inv1[:, None, None, :],
                                  rotation2=full_inv_rotation1[:, None, :, :])

    logger.info(f'\tTransformation of viable oligomer 2 CB atoms and surface fragments took '
                f'{time.time() - int_cb_and_frags_start:8f}s')

    # Todo if using individual Poses
    #  def clone_pose(idx: int) -> Pose:
    #      # Create a copy of the base Pose
    #      new_pose = copy.copy(pose)
    #      if sym_entry.unit_cell:
    #          # Set the next unit cell dimensions
    #          new_pose.uc_dimensions = full_uc_dimensions[idx]
    #      # Update the Pose coords
    #      new_pose.coords = np.concatenate(new_coords)
    #      return new_pose

    # Use below instead of this until can TODO vectorize asu_interface_residue_processing
    # asu_interface_residues = \
    #     np.array([oligomer1_backbone_cb_tree.query_radius(inverse_transformed_model2_tiled_cb_coords[idx],
    #                                                       cb_distance)
    #               for idx in range(inverse_transformed_model2_tiled_cb_coords.shape[0])])

    # Full Interface Fragment Match
    # Gather the data for efficient querying of model1 and model2 interactions
    model1_cb_balltree = BallTree(model1.cb_coords)
    model1_cb_indices = model1.cb_indices
    model1_coords_indexed_residues = model1.coords_indexed_residues
    model2_cb_indices = model2.cb_indices
    model2_coords_indexed_residues = model2.coords_indexed_residues
    zero_counts = []

    # Save all the indices were matching fragments are identified
    interface_is_viable = []
    # all_passing_ghost_indices = []
    # all_passing_surf_indices = []
    # all_passing_z_scores = []
    # Get residue number for all model1, model2 CB Pairs that interact within cb_distance
    for idx in range(number_non_clashing_transforms):
        # query/contact pairs/isin  - 0.028367  <- I predict query is about 0.015
        # indexing guide_coords     - 0.000389
        # total get_int_frags_time  - 0.028756 s

        # indexing guide_coords     - 0.000389
        # Euler Lookup              - 0.008161 s for 71435 fragment pairs
        # Overlap Score Calculation - 0.000365 s for 2949 fragment pairs
        # Total Match time          - 0.008915 s

        # query                     - 0.000895 s <- 100 fold shorter than predict
        # contact pairs             - 0.019595
        # isin indexing             - 0.008992 s
        # indexing guide_coords     - 0.000438
        # get_int_frags_time        - 0.029920 s

        # indexing guide_coords     - 0.000438
        # Euler Lookup              - 0.005603 s for 35400 fragment pairs
        # Overlap Score Calculation - 0.000209 s for 887 fragment pairs
        # Total Match time          - 0.006250 s

        int_frags_time_start = time.time()
        model2_query = model1_cb_balltree.query_radius(inverse_transformed_model2_tiled_cb_coords[idx], cb_distance)
        # model1_cb_balltree_time = time.time() - int_frags_time_start

        contacting_residue_pairs = [(model1_coords_indexed_residues[model1_cb_indices[model1_idx]].number,
                                     model2_coords_indexed_residues[model2_cb_indices[model2_idx]].number)
                                    for model2_idx, model1_contacts in enumerate(model2_query)
                                    for model1_idx in model1_contacts]
        try:
            interface_residue_numbers1, interface_residue_numbers2 = map(list, map(set, zip(*contacting_residue_pairs)))
        except ValueError:  # Interface contains no residues, so not enough values to unpack
            logger.warning('Interface contains no residues')
            continue

        # Find the indices where the fragment residue numbers are found the interface residue numbers
        # is_in_index_start = time.time()
        # Since *_residue_numbers1/2 are the same index as the complete fragment arrays, these interface indices are the
        # same index as the complete guide coords and rmsds as well
        # Both residue numbers are one-indexed vv
        # Todo make ghost_residue_numbers1 unique -> unique_ghost_residue_numbers1
        #  index selected numbers against per_residue_ghost_indices 2d (number surface frag residues,
        ghost_indices_in_interface1 = \
            np.flatnonzero(np.isin(ghost_residue_numbers1, interface_residue_numbers1))
        surf_indices_in_interface2 = \
            np.flatnonzero(np.isin(surf_residue_numbers2, interface_residue_numbers2, assume_unique=True))

        # is_in_index_time = time.time() - is_in_index_start
        all_fragment_match_time_start = time.time()

        # unique_interface_frag_count_model1, unique_interface_frag_count_model2 = \
        #     ghost_indices_in_interface1.shape[0], surf_indices_in_interface2.shape[0]
        # get_int_frags_time = time.time() - int_frags_time_start
        # logger.debug(f'\tNewly formed interface contains {unique_interface_frag_count_model1} unique Fragments on '
        #              f'Oligomer 1 from {len(interface_residue_numbers1)} Residues and '
        #              f'{unique_interface_frag_count_model2} on Oligomer 2 from {len(interface_residue_numbers2)} '
        #              f'Residues\n\t(took {get_int_frags_time:8f}s to get interface fragments, including '
        #              f'{model1_cb_balltree_time:8f}s to query distances, '
        #              f'{is_in_index_time:8f}s to index residue numbers)')

        int_surf_shape = surf_indices_in_interface2.shape[0]
        int_ghost_shape = ghost_indices_in_interface1.shape[0]
        # maximum_number_of_pairs = int_ghost_shape*int_surf_shape
        # if maximum_number_of_pairs < euler_lookup_size_threshold:
        # Todo there may be memory leak by Pose objects sharing memory with persistent objects
        #  that prevent garbage collection and stay attached to the run
        # Skipping EulerLookup as it has issues with precision
        index_ij_pairs_start_time = time.time()
        ghost_indices_repeated = np.repeat(ghost_indices_in_interface1, int_surf_shape)
        surf_indices_tiled = np.tile(surf_indices_in_interface2, int_ghost_shape)
        ij_type_match = ij_type_match_lookup_table[ghost_indices_repeated, surf_indices_tiled]
        # DEBUG: If ij_type_match needs to be removed for testing
        # ij_type_match = np.array([True for _ in range(len(ij_type_match))])
        # Surface selecting
        # [0, 1, 3, 5, ...] with fancy indexing [0, 1, 5, 10, 12, 13, 34, ...]
        possible_fragments_pairs = ghost_indices_repeated.shape[0]
        passing_ghost_indices = ghost_indices_repeated[ij_type_match]
        passing_surf_indices = surf_indices_tiled[ij_type_match]
        # else:  # Narrow candidates by EulerLookup
        #     Get (Oligomer1 Interface Ghost Fragment, Oligomer2 Interface Surface Fragment) guide coordinate pairs
        #     in the same Euler rotational space bucket
        #     DON'T think this is crucial! ###
        #     int_euler_matching_ghost_indices1, int_euler_matching_surf_indices2 = \
        #         euler_lookup.check_lookup_table(int_trans_ghost_guide_coords, int_trans_surf_guide_coords2)
        #     logger.debug('Euler lookup')
        #     logger.warning(f'The interface size is too large ({maximum_number_of_pairs} maximum pairs). '
        #                    f'Trimming possible fragments by EulerLookup')
        #     eul_lookup_start_time = time.time()
        #     int_ghost_guide_coords1 = ghost_guide_coords1[ghost_indices_in_interface1]
        #     int_trans_surf_guide_coords2 = inverse_transformed_surf_frags2_guide_coords[idx, surf_indices_in_interface2]
        #     # Todo Debug skipping EulerLookup to see if issues with precision
        #     int_euler_matching_ghost_indices1, int_euler_matching_surf_indices2 = \
        #         euler_lookup.check_lookup_table(int_ghost_guide_coords1, int_trans_surf_guide_coords2)
        #     # logger.debug(f'int_euler_matching_ghost_indices1: {int_euler_matching_ghost_indices1[:5]}')
        #     # logger.debug(f'int_euler_matching_surf_indices2: {int_euler_matching_surf_indices2[:5]}')
        #     eul_lookup_time = time.time() - eul_lookup_start_time
        #
        #     # Find the ij_type_match which is the same length as the int_euler_matching indices
        #     # this has data type bool so indexing selects al original
        #     index_ij_pairs_start_time = time.time()
        #     ij_type_match = \
        #         ij_type_match_lookup_table[
        #             ghost_indices_in_interface1[int_euler_matching_ghost_indices1],
        #             surf_indices_in_interface2[int_euler_matching_surf_indices2]]
        #     possible_fragments_pairs = int_euler_matching_ghost_indices1.shape[0]
        #
        #     # Get only euler matching fragment indices that pass ij filter. Then index their associated coords
        #     passing_ghost_indices = int_euler_matching_ghost_indices1[ij_type_match]
        #     # passing_ghost_coords = int_trans_ghost_guide_coords[passing_ghost_indices]
        #     # passing_ghost_coords = int_ghost_guide_coords1[passing_ghost_indices]
        #     passing_surf_indices = int_euler_matching_surf_indices2[ij_type_match]
        #     # passing_surf_coords = int_trans_surf_guide_coords2[passing_surf_indices]
        #     DON'T think this is crucial! ###

        # Calculate z_value for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        # Calculate match score for the selected (Ghost Fragment, Interface Fragment) guide coordinate pairs
        overlap_score_time_start = time.time()

        all_fragment_z_score = rmsd_z_score(ghost_guide_coords1[passing_ghost_indices],
                                            inverse_transformed_surf_frags2_guide_coords[idx, passing_surf_indices],
                                            ghost_rmsds1[passing_ghost_indices])
        # all_fragment_match = calculate_match(ghost_guide_coords1[passing_ghost_indices],
        #                                      inverse_transformed_surf_frags2_guide_coords[idx, passing_surf_indices],
        #                                      ghost_rmsds1[passing_ghost_indices])
        logger.debug(
            # f'\tEuler Lookup found {int_euler_matching_ghost_indices1.shape[0]} passing overlaps '
            #      f'(took {eul_lookup_time:8f}s) for '
            #      f'{unique_interface_frag_count_model1 * unique_interface_frag_count_model2} fragment pairs and '
            f'\tZ-score calculation took {time.time() - overlap_score_time_start:8f}s for '
            f'{passing_ghost_indices.shape[0]} successful ij type matches (indexing time '
            f'{overlap_score_time_start - index_ij_pairs_start_time:8f}s) from '
            f'{possible_fragments_pairs} possible fragment pairs')
        # logger.debug(f'Found ij_type_match with shape {ij_type_match.shape}')
        # logger.debug(f'And Data: {ij_type_match[:3]}')
        # logger.debug(f'Found all_fragment_match with shape {all_fragment_match.shape}')
        # logger.debug(f'And Data: {all_fragment_match[:3]}')

        # Check if the pose has enough high quality fragment matches
        # high_qual_match_indices = np.flatnonzero(all_fragment_match >= high_quality_match_value)
        high_qual_match_indices = np.flatnonzero(all_fragment_z_score <= high_quality_z_value)
        high_qual_match_count = len(high_qual_match_indices)
        all_fragment_match_time = time.time() - all_fragment_match_time_start
        if high_qual_match_count < min_matched:
            logger.debug(f'\t{high_qual_match_count} < {min_matched}, the minimal high quality fragment matches '
                         f'(took {all_fragment_match_time:8f}s)')
            # Debug. Why are there no matches... cb_distance?
            if high_qual_match_count == 0:
                zero_counts.append(1)
            continue
        else:
            # Find the passing overlaps to limit the output to only those passing the low_quality_match_value
            # passing_overlaps_indices = np.flatnonzero(all_fragment_match >= low_quality_match_value)
            passing_overlaps_indices = np.flatnonzero(all_fragment_z_score <= low_quality_z_value)
            number_passing_overlaps = passing_overlaps_indices.shape[0]
            logger.info(f'\t{high_qual_match_count} high quality fragments out of {number_passing_overlaps} matches '
                        f'found (took {all_fragment_match_time:8f}s)')
            # Return the indices sorted by z_value in ascending order, truncated at the number of passing
            # sorted_fragment_indices = np.argsort(all_fragment_z_score)[:number_passing_overlaps]
            # sorted_match_scores = match_score_from_z_value(sorted_z_values)
            # logger.debug('Overlapping Match Scores: %s' % sorted_match_scores)
            # sorted_overlap_indices = passing_overlaps_indices[sorted_fragment_indices]
            # interface_ghost_frags = \
            #     complete_ghost_frags1[interface_ghost_indices1][passing_ghost_indices[sorted_overlap_indices]]
            # interface_surf_frags = \
            #     complete_surf_frags2[surf_indices_in_interface2][passing_surf_indices[sorted_overlap_indices]]
            # overlap_passing_ghosts = passing_ghost_indices[sorted_fragment_indices]
            # all_passing_ghost_indices.append(passing_ghost_indices[sorted_fragment_indices])
            # all_passing_surf_indices.append(passing_surf_indices[sorted_fragment_indices])
            # all_passing_z_scores.append(all_fragment_z_score[sorted_fragment_indices])
            interface_is_viable.append(idx)
            # logger.debug(f'\tInterface fragment search time took {time.time() - int_frags_time_start:8f}')
            continue

    logger.debug(f'Found {len(zero_counts)} zero counts')
    number_viable_pose_interfaces = len(interface_is_viable)
    if number_viable_pose_interfaces == 0:  # There were no successful transforms
        logger.warning(f'No interfaces have enough fragment matches. Terminating {building_blocks} docking')
        return []
    # ------------------ TERM ------------------------
    logger.info(f'Found {number_viable_pose_interfaces} poses with viable interfaces')
    # Generate the Pose for output handling
    entity_names = [entity.name for model in models for entity in model.entities]
    # entity_bb_coords = [entity.backbone_coords for model in models for entity in model.entities]
    entity_start_coords = [entity.coords for model in models for entity in model.entities]
    entity_idx = count(0)
    transform_indices = {next(entity_idx): transform_idx
                         for transform_idx, model in enumerate(models)
                         for _ in model.entities}
    entity_info = {entity_name: data for model in models
                   for entity_name, data in model.entity_info.items()}
    chain_gen = chain_id_generator()
    for entity_name, data in entity_info.items():
        data['chains'] = [next(chain_gen)]

    pose = Pose.from_entities([entity for model in models for entity in model.entities],
                              entity_info=entity_info, entity_names=entity_names, name='asu', log=logger,
                              sym_entry=sym_entry, surrounding_uc=job.output_surrounding_uc,
                              fragment_db=job.fragment_db, ignore_clashes=True, rename_chains=True)

    # Calculate metrics on input Pose before any manipulation
    pose_length = pose.number_of_residues
    residue_numbers = list(range(1, pose_length + 1))
    entity_tuple = tuple(pose.entities)
    # model_tuple = tuple(models)

    # residue_numbers = [residue.number for residue in pose.residues]
    # entity_energies = tuple(0. for _ in pose.entities)
    # pose_source_residue_info = \
    #     {residue.number: {'complex': 0.,
    #                       # 'bound': 0.,  # copy(entity_energies),
    #                       'unbound': 0.,  # copy(entity_energies),
    #                       # 'solv_complex': 0., 'solv_bound': 0.,  # copy(entity_energies),
    #                       # 'solv_unbound': 0.,  # copy(entity_energies),
    #                       # 'fsp': 0., 'cst': 0.,
    #                       'type': protein_letters_3to1.get(residue.type),
    #                       # 'hbond': 0
    #                       }
    #      for residue in pose.residues}
    # This needs to be calculated before iterating over each pose
    # residue_info = {pose_source: pose_source_residue_info}
    # residue_info[pose_source] = pose_source_residue_info
    if job.design.sequences and job.design.structures:
        source_contact_order, source_errat = [], []
        for idx, entity in enumerate(pose.entities):
            # Contact order is the same for every design in the Pose and not dependent on pose
            source_contact_order.append(entity.contact_order)
            # Replace 'errat_deviation' measurement with uncomplexed entities
            # oligomer_errat_accuracy, oligomeric_errat = entity_oligomer.errat(out_path=os.path.devnull)
            # Todo translate the source pose
            # Todo when Entity.oligomer works
            #  _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
            entity_oligomer = Model.from_chains(entity.chains, log=logger, entities=False)
            _, oligomeric_errat = entity_oligomer.errat(out_path=os.devnull)
            source_errat.append(oligomeric_errat[:entity.number_of_residues])

        pose_source_contact_order_s = pd.Series(np.concatenate(source_contact_order), index=residue_numbers)
        pose_source_errat_s = pd.Series(np.concatenate(source_errat), index=residue_numbers)

        per_residue_data = {putils.pose_source: {
            # 'type': list(pose.sequence),
            'contact_order': pose_source_contact_order_s,
            'errat_deviation': pose_source_errat_s}}
    else:  # Set up per_residue_data to be empty and populated later. Only populated by proteinmpnn_used = True logic
        per_residue_data = {}

    # Define functions for updating the single Pose instance coordinates
    def update_pose_coords(idx: int):
        """Take the current transformation index and update the reference coordinates with the provided transforms

        Args:
            idx: The index of the transformation to select
        """
        # Get contacting PDB 1 ASU and PDB 2 ASU
        copy_model_start = time.time()
        if sym_entry.is_internal_tx1:
            internal_tx_param1 = full_int_tx1[idx]
        else:
            internal_tx_param1 = None

        if sym_entry.is_internal_tx2:
            internal_tx_param2 = full_int_tx2[idx]
        else:
            internal_tx_param2 = None

        if sym_entry.unit_cell:
            external_tx1 = full_ext_tx1[idx]
            external_tx2 = full_ext_tx2[idx]
            # uc_dimensions = full_uc_dimensions[idx]
            # Set the next unit cell dimensions
            pose.uc_dimensions = full_uc_dimensions[idx]
        else:
            external_tx1 = external_tx2 = None

        specific_transformation1 = dict(rotation=full_rotation1[idx], translation=internal_tx_param1,
                                        rotation2=set_mat1, translation2=external_tx1)
        specific_transformation2 = dict(rotation=full_rotation2[idx], translation=internal_tx_param2,
                                        rotation2=set_mat2, translation2=external_tx2)
        specific_transformations = [specific_transformation1, specific_transformation2]

        # Transform each starting coords to the candidate pose coords then update the Pose coords
        new_coords = []
        for entity_idx, entity in enumerate(pose.entities):
            # logger.debug(f'transform_indices[entity_idx]={transform_indices[entity_idx]}'
            #              f'entity_idx={entity_idx}')
            # tsnfmd = transform_coordinate_sets(entity_start_coords[entity_idx],
            #                                    **specific_transformations[transform_indices[entity_idx]])
            # logger.debug(f'Equality of tsnfmd and original {np.allclose(tsnfmd, entity_start_coords[entity_idx])}')
            # logger.debug(f'tsnfmd: {tsnfmd[:5]}')
            # logger.debug(f'start_coords: {entity_start_coords[entity_idx][:5]}')
            new_coords.append(transform_coordinate_sets(entity_start_coords[entity_idx],
                                                        **specific_transformations[transform_indices[entity_idx]]))
        pose.coords = np.concatenate(new_coords)

        logger.debug(f'\tCopy and Transform Oligomer1 and Oligomer2 (took {time.time() - copy_model_start:8f}s)')

    def find_viable_symmetric_indices(viable_pose_indices: list[int]) -> np.ndarray:
        """Using the nonlocal Pose and transformation indices, check each transformation index for symmetric viability

        Args:
            viable_pose_indices: The indices from the transform array to test for clashes
        Returns:
            An array with the transformation indices that passed clash testing
        """
        # Assume the pose will fail the clash test (0), otherwise, (1) for passing
        _passing_symmetric_clashes = [0 for _ in range(len(viable_pose_indices))]
        for result_idx, transform_idx in enumerate(viable_pose_indices):
            # exp_des_clash_time_start = time.time()
            # Find the pose
            update_pose_coords(transform_idx)
            if not pose.symmetric_assembly_is_clash():
                _passing_symmetric_clashes[result_idx] = 1
            #     logger.info(f'\tNO Backbone Clash when pose is expanded (took '
            #                 f'{time.time() - exp_des_clash_time_start:8f}s)')
            # else:
            #     logger.info(f'\tBackbone Clash when pose is expanded (took '
            #                 f'{time.time() - exp_des_clash_time_start:8f}s)')

        return np.flatnonzero(_passing_symmetric_clashes)

    # Make the indices into an array
    interface_is_viable = np.array(interface_is_viable, dtype=int)

    # Update the passing_transforms
    # passing_transforms contains all the transformations that are still passing
    # index the previously passing indices (sufficiently_dense_indices) and (asu_is_viable_indices)
    # by new passing indices (interface_is_viable)
    # and set each of these indices to 1 (True)
    # passing_transforms[sufficiently_dense_indices[asu_is_viable_indices[interface_is_viable]]] = 1
    # # Remove non-viable transforms from the original transformation parameters by indexing interface_is_viable
    # passing_transforms_indices = np.flatnonzero(passing_transforms)
    # # remove_non_viable_indices(passing_transforms_indices)
    passing_transforms_indices = sufficiently_dense_indices[asu_is_viable_indices[interface_is_viable]]

    if job.design.ignore_symmetric_clashes:
        logger.warning(f'Not checking for symmetric clashes per requested flag --{flags.ignore_symmetric_clashes}')
    else:
        if sym_entry.unit_cell:
            # Calculate the vectorized uc_dimensions
            full_uc_dimensions = sym_entry.get_uc_dimensions(full_optimal_ext_dof_shifts)

        # passing_symmetric_clash_indices = find_viable_symmetric_indices(number_viable_pose_interfaces)
        passing_symmetric_clash_indices = find_viable_symmetric_indices(passing_transforms_indices.tolist())
        number_passing_symmetric_clashes = passing_symmetric_clash_indices.shape[0]
        logger.info(f'After symmetric clash testing, found {number_passing_symmetric_clashes} viable poses')

        if number_passing_symmetric_clashes == 0:  # There were no successful transforms
            logger.warning(f'No viable poses without symmetric clashes. Terminating {building_blocks} docking')
            return []
        # ------------------ TERM ------------------------
        # Update the passing_transforms
        # passing_transforms contains all the transformations that are still passing
        # index the previously passing indices (sufficiently_dense_indices) and (asu_is_viable_indices) and (interface_is_viable)
        # by new passing indices (passing_symmetric_clash_indices)
        # and set each of these indices to 1 (True)
        # passing_transforms_indices = \
        #     sufficiently_dense_indices[asu_is_viable_indices[interface_is_viable[passing_symmetric_clash_indices]]]
        passing_transforms_indices = passing_transforms_indices[passing_symmetric_clash_indices]
        # Todo could this be used?
        # passing_transforms[passing_transforms_indices] = 1

    # Remove non-viable transforms from the original transformations due to clashing
    remove_non_viable_indices(passing_transforms_indices)
    # passing_transforms_indices = np.flatnonzero(passing_transforms)
    degen_counts, rot_counts, tx_counts = zip(*[(degen_counts[idx], rot_counts[idx], tx_counts[idx])
                                                for idx in passing_transforms_indices.tolist()])
    # all_passing_ghost_indices = [all_passing_ghost_indices[idx] for idx in passing_symmetric_clash_indices.tolist()]
    # all_passing_surf_indices = [all_passing_surf_indices[idx] for idx in passing_symmetric_clash_indices.tolist()]
    # all_passing_z_scores = [all_passing_z_scores[idx] for idx in passing_symmetric_clash_indices.tolist()]

    if sym_entry.unit_cell:
        # Calculate the vectorized uc_dimensions
        full_uc_dimensions = sym_entry.get_uc_dimensions(full_optimal_ext_dof_shifts)

    # Perform perturbations to the allowed degrees of freedom
    number_of_transforms = passing_transforms_indices.shape[0]
    if job.dock.perturb_dof_rot or job.dock.perturb_dof_tx:
        perturb_rotation1, perturb_rotation2, perturb_int_tx1, perturb_int_tx2, perturb_optimal_ext_dof_shifts = \
            [], [], [], [], []

        # Define a function to stack the transforms
        def stack_viable_transforms(passing_indices: np.ndarray | list[int]):
            """From indices with viable transformations, stack there corresponding transformations into full
            perturbation transformations

            Args:
                passing_indices: The indices that should be selected from the full transformation sets
            """
            # nonlocal perturb_rotation1, perturb_rotation2, perturb_int_tx1, perturb_int_tx2
            logger.debug(f'Perturb expansion found {len(passing_indices)} passing_perturbations')
            perturb_rotation1.append(full_rotation1[passing_indices])
            perturb_rotation2.append(full_rotation2[passing_indices])
            if sym_entry.is_internal_tx1:
                perturb_int_tx1.extend(full_int_tx1[passing_indices, 2])
            if sym_entry.is_internal_tx2:
                perturb_int_tx2.extend(full_int_tx2[passing_indices, 2])

            if sym_entry.unit_cell:
                nonlocal full_optimal_ext_dof_shifts, full_ext_tx1, full_ext_tx2
                perturb_optimal_ext_dof_shifts.append(full_optimal_ext_dof_shifts[passing_indices])
                # full_uc_dimensions = full_uc_dimensions[passing_indices]
                # full_ext_tx1 = full_ext_tx1[passing_indices]
                # full_ext_tx2 = full_ext_tx2[passing_indices]

        # Expand successful poses from coarse search of transformational space to randomly perturbed offset
        # By perturbing the transformation a random small amount, we generate transformational diversity from
        # the already identified solutions.
        perturbations = create_perturbation_transformations(sym_entry, rotation_number=job.dock.perturb_dof_steps_rot,
                                                            translation_number=job.dock.perturb_dof_steps_tx,
                                                            rotation_range=rotation_steps)
        # Extract perturbation parameters and set the original transformation parameters to a new variable
        # if sym_entry.is_internal_rot1:
        original_rotation1 = full_rotation1
        rotation_perturbations1 = perturbations['rotation1']
        # Compute the length of each perturbation to separate into unique perturbation spaces
        total_perturbation_size, *_ = rotation_perturbations1.shape
        # logger.debug(f'rotation_perturbations1.shape: {rotation_perturbations1.shape}')
        # logger.debug(f'rotation_perturbations1[:5]: {rotation_perturbations1[:5]}')

        # if sym_entry.is_internal_rot2:
        original_rotation2 = full_rotation2
        rotation_perturbations2 = perturbations['rotation2']
        # logger.debug(f'rotation_perturbations2.shape: {rotation_perturbations2.shape}')
        # logger.debug(f'rotation_perturbations2[:5]: {rotation_perturbations2[:5]}')
        # blank_parameter = list(repeat([None, None, None], number_of_transforms))
        if sym_entry.is_internal_tx1:
            original_int_tx1 = full_int_tx1
            translation_perturbations1 = perturbations['translation1']
            # logger.debug(f'translation_perturbations1.shape: {translation_perturbations1.shape}')
            # logger.debug(f'translation_perturbations1[:5]: {translation_perturbations1[:5]}')
        # else:
        #     translation_perturbations1 = blank_parameter

        if sym_entry.is_internal_tx2:
            original_int_tx2 = full_int_tx2
            translation_perturbations2 = perturbations['translation2']
            # logger.debug(f'translation_perturbations2.shape: {translation_perturbations2.shape}')
            # logger.debug(f'translation_perturbations2[:5]: {translation_perturbations2[:5]}')
        # else:
        #     translation_perturbations2 = blank_parameter

        if sym_entry.unit_cell:
            ext_dof_perturbations = perturbations['external_translations']
            original_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts
            # original_ext_tx1 = full_ext_tx1
            # original_ext_tx2 = full_ext_tx2
        else:
            full_ext_tx1 = full_ext_tx2 = full_ext_tx_sum = None

        # Apply the perturbation to each existing transformation
        for idx in range(number_of_transforms):
            # Rotate the unique rotation by the perturb_matrix_grid and set equal to the full_rotation* array
            full_rotation1 = np.matmul(original_rotation1[idx], rotation_perturbations1.swapaxes(-1, -2))  # rotation1
            full_inv_rotation1 = np.linalg.inv(full_rotation1)
            full_rotation2 = np.matmul(original_rotation2[idx], rotation_perturbations2.swapaxes(-1, -2))  # rotation2

            # Translate the unique translation according to the perturb_translation_grid
            if sym_entry.is_internal_tx1:
                full_int_tx1 = original_int_tx1[idx] + translation_perturbations1  # translation1
            if sym_entry.is_internal_tx2:
                full_int_tx2 = original_int_tx2[idx] + translation_perturbations2  # translation2
            if sym_entry.unit_cell:
                # perturbed_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts[None] + ext_dof_perturbations
                # full_ext_tx_perturb1 = (perturbed_optimal_ext_dof_shifts[:, :, None] * sym_entry.external_dof1).sum(axis=-2)
                # full_ext_tx_perturb2 = (perturbed_optimal_ext_dof_shifts[:, :, None] * sym_entry.external_dof2).sum(axis=-2)
                # Below is for the individual perturbation
                # optimal_ext_dof_shift = full_optimal_ext_dof_shifts[idx]
                # perturbed_ext_dof_shift = optimal_ext_dof_shift + ext_dof_perturbations
                unsqueezed_perturbed_ext_dof_shifts = \
                    (original_optimal_ext_dof_shifts[idx] + ext_dof_perturbations)[:, :, None]
                # unsqueezed_perturbed_ext_dof_shifts = perturbed_ext_dof_shift[:, :, None]
                full_ext_tx1 = np.sum(unsqueezed_perturbed_ext_dof_shifts * sym_entry.external_dof1, axis=-2)
                full_ext_tx2 = np.sum(unsqueezed_perturbed_ext_dof_shifts * sym_entry.external_dof2, axis=-2)
                full_ext_tx_sum = full_ext_tx2 - full_ext_tx1

            # Check for ASU clashes again
            # Using the inverse transform of the model2 backbone and cb coordinates, check for clashes with the model1
            # backbone and cb coordinates BallTree
            ball_tree_kwargs = dict(binarytree=oligomer1_backbone_cb_tree,
                                    rotation=full_rotation2, translation=full_int_tx2,
                                    rotation2=set_mat2, translation2=full_ext_tx_sum,
                                    rotation3=inv_setting1,
                                    translation3=None if full_int_tx1 is None else full_int_tx1 * -1,
                                    rotation4=full_inv_rotation1)
            # Create a fresh asu_clash_counts
            asu_clash_counts = np.ones(total_perturbation_size)
            clash_time_start = time.time()
            overlap_return = check_tree_for_query_overlap(**ball_tree_kwargs,
                                                          return_containers={'overlap_counts': asu_clash_counts},
                                                          setup_args=(bb_cb_coords2,))
            logger.debug(f'Perturb clash took {time.time() - clash_time_start:8f}s')

            # Extract the data
            asu_clash_counts = overlap_return['overlap_counts']
            logger.debug(f'Perturb expansion found asu_clash_counts:\n{asu_clash_counts}')
            passing_perturbations = np.flatnonzero(asu_clash_counts == 0)
            # Check for symmetric clashes again
            if not job.design.ignore_symmetric_clashes:
                passing_symmetric_clash_indices_perturb = find_viable_symmetric_indices(passing_perturbations.tolist())
            # Index the passing ASU indices with the passing symmetric indices and keep all viable transforms
            # Stack the viable perturbed transforms
            stack_viable_transforms(passing_perturbations[passing_symmetric_clash_indices_perturb])

        # Concatenate the stacked perturbations
        full_rotation1 = np.concatenate(perturb_rotation1, axis=0)
        logger.debug(f'After perturbation, found full_rotation1.shape: {full_rotation1.shape}')
        full_rotation2 = np.concatenate(perturb_rotation2, axis=0)
        number_of_transforms = full_rotation1.shape[0]
        if sym_entry.is_internal_tx1:
            stacked_internal_tx_vectors1 = np.zeros((number_of_transforms, 3), dtype=float)
            # Add the translation to Z (axis=1)
            stacked_internal_tx_vectors1[:, -1] = perturb_int_tx1
            full_int_tx1 = stacked_internal_tx_vectors1

        if sym_entry.is_internal_tx2:
            stacked_internal_tx_vectors2 = np.zeros((number_of_transforms, 3), dtype=float)
            # Add the translation to Z (axis=1)
            stacked_internal_tx_vectors2[:, -1] = perturb_int_tx2
            full_int_tx2 = stacked_internal_tx_vectors2

        if sym_entry.unit_cell:
            # optimal_ext_dof_shifts[:, :, None] <- None expands the axis to make multiplication accurate
            full_optimal_ext_dof_shifts = np.concatenate(perturb_optimal_ext_dof_shifts, axis=0)
            unsqueezed_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts[:, :, None]
            full_ext_tx1 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof1, axis=-2)
            full_ext_tx2 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof2, axis=-2)

        # total_number_of_perturbations = number_of_transforms * total_perturbation_size
    else:
        # total_number_of_perturbations = number_of_transforms
        total_perturbation_size = 1

    # From here out, the transforms used should be only those of interest for outputting/sequence design
    # remove_non_viable_indices() <- This is done above
    # Format pose transformations for output
    blank_parameter = list(repeat([None, None, None], number_of_transforms))
    full_ext_tx1 = blank_parameter if full_ext_tx1 is None else full_ext_tx1.squeeze()
    full_ext_tx2 = blank_parameter if full_ext_tx2 is None else full_ext_tx2.squeeze()

    set_mat1_number, set_mat2_number, *_extra = sym_entry.setting_matrices_numbers
    rotations1 = scipy.spatial.transform.Rotation.from_matrix(full_rotation1)
    rotations2 = scipy.spatial.transform.Rotation.from_matrix(full_rotation2)
    # Get all rotations in terms of the degree of rotation along the z-axis
    rotation_degrees1 = rotations1.as_rotvec(degrees=True)[:, -1]
    rotation_degrees2 = rotations2.as_rotvec(degrees=True)[:, -1]
    # Todo get the degeneracy_degrees
    # degeneracy_degrees1 = rotations1.as_rotvec(degrees=True)[:, :-1]
    # degeneracy_degrees2 = rotations2.as_rotvec(degrees=True)[:, :-1]
    if sym_entry.is_internal_tx1:
        if full_int_tx1.shape[0] > 1:
            full_int_tx1 = full_int_tx1.squeeze()
        z_heights1 = full_int_tx1[:, -1]
    else:
        z_heights1 = blank_parameter

    if sym_entry.is_internal_tx2:
        if full_int_tx2.shape[0] > 1:
            full_int_tx2 = full_int_tx2.squeeze()
        z_heights2 = full_int_tx2[:, -1]
    else:
        z_heights2 = blank_parameter
    # if sym_entry.unit_cell:
    #     full_uc_dimensions = full_uc_dimensions[passing_symmetric_clash_indices_perturb]
    #     full_ext_tx1 = full_ext_tx1[:]
    #     full_ext_tx2 = full_ext_tx2[:]
    #     full_ext_tx_sum = full_ext_tx2 - full_ext_tx1

    # Save all pose transformation information
    def create_pose_id(_idx: int) -> str:
        """Create a PoseID from the sampling conditions

        Args:
            _idx: The current sampling index
        Returns:
            The PoseID with format building_blocks-degeneracy-rotation-transform-perturb if perturbation used
                Ex: '****_#-****_#-d_#_#-r_#_#-t_#-p_#' OR '****_#-****_#-d_#_#-r_#_#-t_#' (no perturbation)
        """
        transform_idx = _idx // total_perturbation_size
        _pose_id = f'd_{"_".join(map(str, degen_counts[transform_idx]))}' \
                   f'-r_{"_".join(map(str, rot_counts[transform_idx]))}' \
                   f'-t_{tx_counts[transform_idx]}'  # translation idx
        if total_perturbation_size > 1:
            # perturb_idx = idx % total_perturbation_size
            _pose_id = f'{_pose_id}-p_{_idx%total_perturbation_size + 1}'

        return f'{building_blocks}-{_pose_id}'

    # Todo move after job.dock.score tabulation
    pose_transformations = {}
    for idx in range(number_of_transforms):
        external_translation1_x, external_translation1_y, external_translation1_z = full_ext_tx1[idx]
        external_translation2_x, external_translation2_y, external_translation2_z = full_ext_tx2[idx]
        pose_transformations[create_pose_id(idx)] = \
            dict(rotation1=rotation_degrees1[idx],
                 internal_translation1=z_heights1[idx],
                 setting_matrix1=set_mat1_number,
                 external_translation1_x=external_translation1_x,
                 external_translation1_y=external_translation1_y,
                 external_translation1_z=external_translation1_z,
                 rotation2=rotation_degrees2[idx],
                 internal_translation2=z_heights2[idx],
                 setting_matrix2=set_mat2_number,
                 external_translation2_x=external_translation2_x,
                 external_translation2_y=external_translation2_y,
                 external_translation2_z=external_translation2_z)

    # Capture all the pose_ids from the transformation dictionary
    pose_ids = list(pose_transformations.keys())

    def add_fragments_to_pose(overlap_ghosts: list[int] = None, overlap_surf: list[int] = None,
                              sorted_z_scores: np.ndarray = None):
        """Add observed fragments to the Pose or generate new observations given the Pose state

        If no arguments are passed, the fragment observations will be generated new
        """
        # First, clear any pose information and force identification of the interface
        pose.split_interface_residues = {}
        pose.find_and_split_interface(distance=cb_distance)

        # Next, set the interface fragment info for gathering of interface metrics
        if overlap_ghosts is None or overlap_surf is None or sorted_z_scores is None:
            # Remove old fragments
            pose.fragment_queries = {}
            # Query fragments
            pose.generate_interface_fragments()
        else:  # Process with provided data
            # Return the indices sorted by z_value in ascending order, truncated at the number of passing
            sorted_match_scores = match_score_from_z_value(sorted_z_scores)

            # These are indexed outside this function
            # overlap_ghosts = passing_ghost_indices[sorted_fragment_indices]
            # overlap_surf = passing_surf_indices[sorted_fragment_indices]

            sorted_int_ghostfrags: list[GhostFragment] = [complete_ghost_frags1[idx] for idx in overlap_ghosts]
            sorted_int_surffrags2: list[Residue] = [complete_surf_frags2[idx] for idx in overlap_surf]
            # For all matched interface fragments
            # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
            # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
            # chid_resnum_scores_dict_model1, chid_resnum_scores_dict_model2 = {}, {}
            # Number of unique interface mono fragments matched
            # unique_frags_info1, unique_frags_info2 = set(), set()
            # res_pair_freq_info_list = []
            fragment_pairs = list(zip(sorted_int_ghostfrags, sorted_int_surffrags2, sorted_match_scores))
            frag_match_info = get_matching_fragment_pairs_info(fragment_pairs)
            # pose.fragment_queries = {(model1, model2): frag_match_info}
            fragment_metrics = pose.fragment_db.calculate_match_metrics(frag_match_info)
            # Todo when able to take more than 2 Entity
            #  The entity_tuple must contain the same Entity instances as in the Pose!
            # entity_tuple = models_tuple
            # These two pose attributes must be set
            pose.fragment_queries = {entity_tuple: frag_match_info}
            pose.fragment_metrics = {entity_tuple: fragment_metrics}

    # Should interface metrics be performed to inform on docking success?
    interface_metrics = {}
    # # Todo use this to control the output of this section
    # match job.dock.score:
    #     case 'proteinmpnn':
    #         check_dock_for_designability()
    #     case 'nanohedra':
    #         # Todo make the below function
    #         check_dock_for_nanohedra()
    # # Todo use this to control the output of this section

    proteinmpnn_used = False
    if job.dock_only:  # Only get pose outputs, no sequences or metrics
        design_ids = pose_ids
        # logger.info(f'Total {building_blocks} dock trajectory took {time.time() - frag_dock_time_start:.2f}s')
        # terminate()  # End of docking run
        # return pose_paths
    elif job.dock.proteinmpnn_score or job.design.sequences:  # Initialize proteinmpnn for dock/design
        proteinmpnn_used = True
        # Load profiles of interest into the analysis
        profile_background = {}
        if job.design.evolution_constraint:
            # Add Entity information to the Pose
            measure_evolution = measure_alignment = True
            warn = False
            for entity in pose.entities:
                # entity.sequence_file = job.api_db.sequences.retrieve_file(name=entity.name)
                # if not entity.sequence_file:
                #     entity.write_sequence_to_fasta('reference', out_dir=job.sequences)
                #     # entity.add_evolutionary_profile(out_dir=job.api_db.hhblits_profiles.location)
                # else:
                profile = job.api_db.hhblits_profiles.retrieve_data(name=entity.name)
                if not profile:
                    measure_evolution = False
                    warn = True
                else:
                    entity.evolutionary_profile = profile

                if not entity.verify_evolutionary_profile():
                    entity.fit_evolutionary_profile_to_structure()

                try:  # To fetch the multiple sequence alignment for further processing
                    msa = job.api_db.alignments.retrieve_data(name=entity.name)
                    if not msa:
                        measure_evolution = False
                        warn = True
                    else:
                        entity.msa = msa
                except ValueError as error:  # When the Entity reference sequence and alignment are different lengths
                    # raise error
                    logger.info(f'Entity reference sequence and provided alignment are different lengths: {error}')
                    warn = True

            if warn:
                if not measure_evolution and not measure_alignment:
                    logger.info(f'Metrics relying on multiple sequence alignment data are not being collected as '
                                f'there were none found. These include: '
                                f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
                elif not measure_alignment:
                    logger.info(f'Metrics relying on a multiple sequence alignment are not being collected as '
                                f'there was no MSA found. These include: '
                                f'{", ".join(multiple_sequence_alignment_dependent_metrics)}')
                else:
                    logger.info(f'Metrics relying on an evolutionary profile are not being collected as '
                                f'there was no profile found. These include: '
                                f'{", ".join(profile_dependent_metrics)}')

            if measure_evolution:
                pose.evolutionary_profile = \
                    concatenate_profile([entity.evolutionary_profile for entity in pose.entities])
            else:
                pose.log.info('No evolution information')
                pose.evolutionary_profile = pose.create_null_profile()

            if pose.evolutionary_profile:
                profile_background['evolution'] = evolutionary_profile_array = pssm_as_array(pose.evolutionary_profile)
                batch_evolutionary_profile = torch.from_numpy(np.tile(evolutionary_profile_array,
                                                                      (batch_length, 1, 1)))
                torch_log_evolutionary_profile = torch.from_numpy(np.log(evolutionary_profile_array))
            # else:
            #     pose.log.info('No evolution information')

        if job.fragment_db is not None:
            # Todo ensure the AA order is the same as MultipleSequenceAlignment.from_dictionary(pose_sequences) below
            interface_bkgd = np.array(list(job.fragment_db.aa_frequencies.values()))
            profile_background['interface'] = np.tile(interface_bkgd, (pose.number_of_residues, 1))

        # Gather folding metrics for the pose for comparison to the designed sequences
        contact_order_per_res_z, reference_collapse, collapse_profile = pose.get_folding_metrics()
        if collapse_profile.size:  # Not equal to zero
            collapse_profile_mean, collapse_profile_std = \
                np.nanmean(collapse_profile, axis=-2), np.nanstd(collapse_profile, axis=-2)

        # Extract parameters to run ProteinMPNN design and modulate memory requirements
        # Retrieve the ProteinMPNN model
        mpnn_model = ml.proteinmpnn_factory()  # Todo accept model_name arg. Now just use the default
        # Set up parameters and model sampling type based on symmetry
        if pose.is_symmetric():
            # number_of_symmetry_mates = pose.number_of_symmetry_mates
            mpnn_sample = mpnn_model.tied_sample
            number_of_residues = pose_length * pose.number_of_symmetry_mates
        else:
            mpnn_sample = mpnn_model.sample
            number_of_residues = pose_length

        if job.design.ca_only:
            coords_type = 'ca_coords'
            num_model_residues = 1
        else:
            coords_type = 'backbone_coords'
            num_model_residues = 4

        # Translate the coordinates along z in increments of 1000 to separate coordinates
        entity_unbound_coords = [getattr(entity, coords_type) for model in models for entity in model.entities]
        unbound_transform = np.array([0, 0, 1000])
        if pose.is_symmetric():
            coord_func = pose.return_symmetric_coords
        else:
            def coord_func(coords): return coords

        for idx, coords in enumerate(entity_unbound_coords):
            entity_unbound_coords[idx] = coord_func(coords + unbound_transform*idx)

        device = mpnn_model.device
        if device.type == 'cpu':
            mpnn_memory_constraint = psutil.virtual_memory().available
            logger.debug(f'The available cpu memory is: {mpnn_memory_constraint}')
        else:
            mpnn_memory_constraint, gpu_memory_total = torch.cuda.mem_get_info()
            logger.debug(f'The available gpu memory is: {mpnn_memory_constraint}')

        element_memory = 4  # where each element is np.int/float32
        number_of_elements_available = mpnn_memory_constraint / element_memory
        logger.debug(f'The number_of_elements_available is: {number_of_elements_available}')
        number_of_mpnn_model_parameters = sum([math.prod(param.size()) for param in mpnn_model.parameters()])
        model_elements = number_of_mpnn_model_parameters
        # Todo use 5 as ideal CB is added by the model later with ca_only = False
        model_elements += prod((number_of_residues, num_model_residues, 3))  # X,
        model_elements += number_of_residues  # S.shape
        model_elements += number_of_residues  # chain_mask.shape
        model_elements += number_of_residues  # chain_encoding.shape
        model_elements += number_of_residues  # residue_idx.shape
        model_elements += number_of_residues  # mask.shape
        model_elements += number_of_residues  # residue_mask.shape
        model_elements += prod((number_of_residues, 21))  # omit_AA_mask.shape
        model_elements += number_of_residues  # pssm_coef.shape
        model_elements += prod((number_of_residues, 20))  # pssm_bias.shape
        model_elements += prod((number_of_residues, 20))  # pssm_log_odds_mask.shape
        model_elements += number_of_residues  # tied_beta.shape
        model_elements += prod((number_of_residues, 21))  # bias_by_res.shape
        logger.debug(f'The number of model_elements is: {model_elements}')

        size = full_rotation1.shape[0]  # This is the number of transformations, i.e. the number_of_designs
        # The batch_length indicates how many models could fit in the allocated memory. Using floor division to get integer
        # Reduce scale by factor of divisor to be safe
        # start_divisor = divisor = 512  # 256 # 128  # 2048 breaks when there is a gradient for training
        # batch_length = 10
        # batch_length = int(number_of_elements_available//model_elements//start_divisor)
        batch_length = 6  # works for 24 GiB mem with 6264 residue T input, 7 is too much
        # once, twice = False, False

        # Set up Pose parameters
        parameters = pose.get_proteinmpnn_params()
        # Todo
        #  Must calculate randn individually if using some feature to describe order
        parameters['randn'] = pose.generate_proteinmpnn_decode_order()  # to_device=device)

        # Add a parameter for the unbound version of X to X
        X_unbound = np.concatenate(entity_unbound_coords).reshape((number_of_residues, num_model_residues, 3))
        # extra_batch_parameters = ml.proteinmpnn_to_device(device, **ml.batch_proteinmpnn_input(size=batch_length,
        #                                                                                        X=X_unbound))
        parameters['X_unbound'] = X_unbound
        # Disregard X, chain_M_pos, and bias_by_res parameters return and use the pose specific data from below
        # parameters.pop('X')  # overwritten by X_unbound
        parameters.pop('chain_M_pos')
        parameters.pop('bias_by_res')
        # tied_pos = parameters.pop('tied_pos')
        # # Todo if modifying the amount of weight given to each of the copies
        # tied_beta = parameters.pop('tied_beta')
        # # Set the design temperature
        # temperature = job.design.temperatures[0]

        proteinmpnn_time_start = time.time()

        @torch.no_grad()  # Ensure no gradients are produced
        @resources.ml.batch_calculation(size=size, batch_length=batch_length,
                                        setup=ml.setup_pose_batch_for_proteinmpnn,
                                        compute_failure_exceptions=(RuntimeError,
                                                                    np.core._exceptions._ArrayMemoryError))
        def check_dock_for_designability(batch_slice: slice,
                                         S: torch.Tensor = None,
                                         chain_mask: torch.Tensor = None,
                                         chain_encoding: torch.Tensor = None,
                                         residue_idx: torch.Tensor = None,
                                         mask: torch.Tensor = None,
                                         randn: torch.Tensor = None,
                                         pose_length: int = None,
                                         tied_pos: Iterable[Container] = None,
                                         **batch_parameters) -> dict[str, np.ndarray]:
            actual_batch_length = batch_slice.stop - batch_slice.start
            # Get the null_idx
            mpnn_null_idx = resources.ml.MPNN_NULL_IDX
            if pose_length is None:
                batch_length, pose_length, *_ = S.shape
            else:
                batch_length, *_ = S.shape
            # Initialize pose data structures for interface design
            residue_mask_cpu = np.zeros((actual_batch_length, pose_length),
                                        dtype=np.int32)  # (batch, number_of_residues)
            # Stack the entity coordinates to make up a contiguous block for each pose
            # If entity_bb_coords are stacked, then must concatenate along axis=1 to get full pose
            new_coords = np.zeros((actual_batch_length, pose_length * num_model_residues, 3),
                                  dtype=np.float32)  # (batch, number_of_residues, coords_length)

            fragment_profiles = []
            design_profiles = []
            # Use batch_idx to set new numpy arrays, transform_idx (includes perturb_idx) to set coords
            for batch_idx, transform_idx in enumerate(range(batch_slice.start, batch_slice.stop)):
                # Get the transformations based on the global index from batch_length
                update_pose_coords(transform_idx)
                new_coords[batch_idx] = getattr(pose, coords_type)

                # pose.find_and_split_interface(distance=cb_distance)
                # This is done in the below call
                add_fragments_to_pose()  # <- here generating fragments fresh
                # Reset the fragment_profile and fragment_map for each Entity before calculate_fragment_profile
                for entity in pose.entities:
                    entity._fragment_profile = {}
                    entity.fragment_map = {}
                    # entity.alpha.clear()

                # Load fragment_profile into the analysis
                pose.calculate_fragment_profile()
                # if pose.fragment_profile:
                fragment_profiles.append(pose.fragment_profile.as_array())
                # else:
                #     fragment_profiles.append(pose.fragment_profile.as_array())

                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()
                # # Remove saved pose attributes from the prior iteration calculations
                # pose.ss_index_array.clear(), pose.ss_type_array.clear()
                # pose.fragment_metrics.clear(), pose.fragment_pairs.clear()
                # for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
                #     try:
                #         delattr(pose, attribute)
                #     except AttributeError:
                #         pass
                #
                # # Calculate pose metrics
                # interface_metrics[design_id] = pose.interface_metrics()
                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()
                pose.calculate_profile()
                design_profiles.append(pssm_as_array(pose.profile))

                # Add all interface residues
                design_residues = []
                for number, residues_entities in pose.split_interface_residues.items():
                    design_residues.extend([residue.index for residue, _ in residues_entities])

                # Residues to design are 1, others are 0
                residue_mask_cpu[batch_idx, design_residues] = 1

            # If entity_bb_coords are individually transformed, then axis=0 works
            perturbed_bb_coords = np.concatenate(new_coords, axis=0)

            # Format the bb coords for ProteinMPNN
            if pose.is_symmetric():
                # Make each set of coordinates symmetric. Lattice cases have uc_dimensions passed in update_pose_coords
                _perturbed_bb_coords = []
                for idx in range(perturbed_bb_coords.shape[0]):
                    _perturbed_bb_coords.append(pose.return_symmetric_coords(perturbed_bb_coords[idx]))

                # Let -1 fill in the pose length dimension with the number of residues
                # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
                perturbed_bb_coords = np.concatenate(_perturbed_bb_coords)

                # Symmetrize other arrays
                number_of_symmetry_mates = pose.number_of_symmetry_mates
                # (batch, number_of_sym_residues, ...)
                residue_mask_cpu = np.tile(residue_mask_cpu, (1, number_of_symmetry_mates))
                # bias_by_res = np.tile(bias_by_res, (1, number_of_symmetry_mates, 1))

            # Reshape for ProteinMPNN
            logger.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')
            X = perturbed_bb_coords.reshape((actual_batch_length, -1, num_model_residues, 3))
            logger.debug(f'X.shape: {X.shape}')

            # Update different parameters to the identified device
            batch_parameters.update(ml.proteinmpnn_to_device(device, X=X, chain_M_pos=residue_mask_cpu))
            # Different across poses
            X = batch_parameters.pop('X')
            residue_mask = batch_parameters.get('chain_M_pos', None)
            # # Potentially different across poses
            # bias_by_res = batch_parameters.get('bias_by_res', None)
            # Todo calculate individually if using some feature to describe order
            #  MUST reinstate the removal from scope after finished with this batch
            # decoding_order = pose.generate_proteinmpnn_decode_order(to_device=device)
            # decoding_order.repeat(actual_batch_length, 1)
            # with torch.no_grad():  # Ensure no gradients are produced
            # Unpack constant parameters and slice reused parameters only once
            # X_unbound = batch_parameters.pop('X')  # Remove once batch_calculation()
            # chain_mask = batch_parameters.pop('chain_mask')
            # chain_encoding = batch_parameters.pop('chain_encoding')
            # residue_idx = batch_parameters.pop('residue_idx')
            # mask = batch_parameters.pop('mask')
            # randn = batch_parameters.pop('randn')

            # Todo remove S from above unpacking
            #
            # ml.proteinmpnn_batch_design(batch_slice,
            #                             mpnn_model: ProteinMPNN,
            #                             temperatures=job.design.temperatures,
            #                             **parameters,  # (randn, S, chain_mask, chain_encoding, residue_idx, mask, temperatures, pose_length, bias_by_res, tied_pos, X_unbound)
            #                             **batch_parameters  # (X, chain_M_pos, bias_by_res)
            # ml.proteinmpnn_batch_design(batch_slice: slice, proteinmpnn: ProteinMPNN,
            #                             X: torch.Tensor = None,
            #                             randn: torch.Tensor = None,
            #                             S: torch.Tensor = None,
            #                             chain_mask: torch.Tensor = None,
            #                             chain_encoding: torch.Tensor = None,
            #                             residue_idx: torch.Tensor = None,
            #                             mask: torch.Tensor = None,
            #                             temperatures: Sequence[float] = (0.1,),
            #                             pose_length: int = None,
            #                             bias_by_res: torch.Tensor = None,
            #                             tied_pos: Iterable[Container] = None,
            #                             X_unbound: torch.Tensor = None,
            #                             **batch_parameters
            #                             ) -> dict[str, np.ndarray]:

            # Clone the data from the sequence tensor so that it can be set with the null token below
            S_design_null = S.detach().clone()
            # Get the provided batch_length from wrapping function. actual_batch_length may be smaller on last batch
            # batch_length = X.shape[0]
            # batch_length = X_unbound.shape[0]
            if actual_batch_length != batch_length:
                # Slice these for the last iteration
                # X_unbound = X_unbound[:actual_batch_length]  # , None)
                chain_mask = chain_mask[:actual_batch_length]  # , None)
                chain_encoding = chain_encoding[:actual_batch_length]  # , None)
                residue_idx = residue_idx[:actual_batch_length]  # , None)
                mask = mask[:actual_batch_length]  # , None)
                randn = randn[:actual_batch_length]
                S_design_null = S_design_null[:actual_batch_length]  # , None)
                # # Unpack, unpacked keyword args
                # omit_AA_mask = batch_parameters.get('omit_AA_mask')
                # pssm_coef = batch_parameters.get('pssm_coef')
                # pssm_bias = batch_parameters.get('pssm_bias')
                # pssm_log_odds_mask = batch_parameters.get('pssm_log_odds_mask')
                # # Set keyword args
                # batch_parameters['omit_AA_mask'] = omit_AA_mask[:actual_batch_length]
                # batch_parameters['pssm_coef'] = pssm_coef[:actual_batch_length]
                # batch_parameters['pssm_bias'] = pssm_bias[:actual_batch_length]
                # batch_parameters['pssm_log_odds_mask'] = pssm_log_odds_mask[:actual_batch_length]

            # Use the sequence as an unknown token then guess the probabilities given the remaining
            # information, i.e. the sequence and the backbone
            S_design_null[residue_mask.type(torch.bool)] = mpnn_null_idx
            chain_residue_mask = chain_mask * residue_mask

            decoding_order = ml.create_decoding_order(randn, chain_mask, tied_pos=tied_pos, to_device=device)
            # See if the pose is useful to design based on constraints of collapse

            # Measure the conditional amino acid probabilities at each residue to see
            # how they compare to various profiles from the Pose multiple sequence alignment
            # If conditional_probs() are measured, then we need a batched_decoding order
            # conditional_start_time = time.time()
            # Calculations with this are done using cpu memory and numpy
            conditional_log_probs_null_seq = \
                mpnn_model(X, S_design_null, mask, chain_residue_mask, residue_idx, chain_encoding,
                           None,  # This argument is provided but with below args, is not used
                           use_input_decoding_order=True, decoding_order=decoding_order).cpu()
            _residue_indices_of_interest = residue_mask_cpu[:, :pose_length].astype(bool)
            #  Taking the KL divergence would indicate how divergent the interfaces are from the
            #  surface. This should be simultaneously minimized (i.e. lowest evolutionary divergence)
            #  while the aa frequency distribution cross_entropy compared to the fragment profile is
            #  minimized
            asu_conditional_softmax_null_seq = \
                np.exp(conditional_log_probs_null_seq[:, :pose_length])
            # asu_conditional_softmax
            # tensor([[[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          ...,
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]],
            #          ...
            #         [[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          ...,
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]]])

            if pose.fragment_profile:
                # Process the fragment_profiles into an array for cross entropy
                fragment_profile_array = np.array(fragment_profiles)
                # RuntimeWarning: divide by zero encountered in log
                # np.log causes -inf at 0, thus we need to correct these to a very large number
                batch_fragment_profile = torch.from_numpy(np.nan_to_num(fragment_profile_array, copy=False, nan=np.nan))
                # print('batch_fragment_profile', batch_fragment_profile[:, 20:23])
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_fragment_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_fragment_profile,
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
                # All per_residue metrics look the same. Shape batch_length, number_of_residues
                # per_residue_evolution_cross_entropy[batch_slice]
                # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
                #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
                #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
            else:
                _per_residue_fragment_cross_entropy = np.empty_like(_residue_indices_of_interest, dtype=np.float32)
                _per_residue_fragment_cross_entropy[:] = np.nan

            if pose.evolutionary_profile:
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_evolution_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_evolutionary_profile[:actual_batch_length],
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
            else:  # Populate with null data
                _per_residue_evolution_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
                _per_residue_evolution_cross_entropy[:] = np.nan

            if pose.profile:
                # Process the design_profiles into an array for cross entropy
                batch_design_profile = torch.from_numpy(np.array(design_profiles))
                # print('batch_design_profile', batch_design_profile[:, 20:23])
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_design_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_design_profile,
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
                # All per_residue metrics look the same. Shape batch_length, number_of_residues
                # per_residue_evolution_cross_entropy[batch_slice]
                # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
                #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
                #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
            else:  # Populate with null data
                _per_residue_design_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
                _per_residue_design_cross_entropy[:] = np.nan

            if collapse_profile.size:  # Not equal to zero
                # Take the hydrophobic collapse of the log probs to understand the profiles "folding"
                _poor_collapse = []
                _per_residue_mini_batch_collapse_z = \
                    np.empty((actual_batch_length, pose_length), dtype=np.float32)
                for pose_idx in range(actual_batch_length):
                    # Only include the residues in the ASU
                    design_probs_collapse = \
                        hydrophobic_collapse_index(asu_conditional_softmax_null_seq[pose_idx],
                                                   # asu_unconditional_softmax,
                                                   alphabet_type=ml.mpnn_alphabet)
                    # Todo?
                    #  design_probs_collapse = \
                    #      hydrophobic_collapse_index(asu_conditional_softmax,
                    #                                 alphabet_type=ml.mpnn_alphabet)
                    # Compare the sequence collapse to the pose collapse
                    # USE:
                    #  contact_order_per_res_z, reference_collapse, collapse_profile
                    # print('HCI profile mean', collapse_profile_mean)
                    # print('HCI profile std', collapse_profile_std)
                    _per_residue_mini_batch_collapse_z[pose_idx] = collapse_z = \
                        z_score(design_probs_collapse, collapse_profile_mean, collapse_profile_std)
                    # folding_loss = ml.sequence_nllloss(S_sample, design_probs_collapse)  # , mask_for_loss)
                    pose_idx_residues_of_interest = _residue_indices_of_interest[pose_idx]
                    designed_indices_collapse_z = collapse_z[pose_idx_residues_of_interest]
                    # magnitude_of_collapse_z_deviation = np.abs(designed_indices_collapse_z)
                    # Check if dock has collapse larger than collapse_significance_threshold and increased collapse
                    if np.any(np.logical_and(design_probs_collapse[pose_idx_residues_of_interest]
                                             > collapse_significance_threshold,
                                             designed_indices_collapse_z > 0)):
                        # Todo save this
                        print('design_probs_collapse', design_probs_collapse[_residue_indices_of_interest[pose_idx]])
                        print('designed_indices_collapse_z', designed_indices_collapse_z)
                        # design_probs_collapse [0.1229698  0.14987233 0.23318215 0.23268045 0.23882663 0.24801936
                        #  0.25622816 0.44975936 0.43138875 0.3607946  0.3140504  0.28207788
                        #  0.27033003 0.27388856 0.28031376 0.28897327 0.14254868 0.13711281
                        #  0.12078322 0.11563808 0.13515363 0.16421124 0.16638894 0.16817969
                        #  0.16234223 0.19553652 0.20065537 0.1901575  0.17455298 0.17621328
                        #  0.20747318 0.21465868 0.22461864 0.21520302 0.21346277 0.2054776
                        #  0.17700449 0.15074518 0.11202089 0.07674509 0.08504518 0.09990609
                        #  0.16057604 0.14554144 0.14646661 0.15743639 0.2136532  0.23222249
                        #  0.26718637]
                        # designed_indices_collapse_z [-0.80368181 -1.2787087   0.71124918  1.04688287  1.26099661 -0.17269616
                        #  -0.06417628  1.16625098  0.94364294  0.62500235  0.53019078  0.5038286
                        #   0.59372686  0.82563642  1.12022683  1.1989269  -1.07529947 -1.27769417
                        #  -1.24323295 -0.95376269  0.55229076  1.05845308  0.62604691  0.20474606
                        #  -0.20987778 -0.45545679 -0.40602295 -0.54974293 -0.72873982 -0.84489538
                        #  -0.8104777  -0.80596935 -0.71591074 -0.79774316 -0.75114322 -0.77010185
                        #  -0.63265472 -0.61240502 -0.69975283 -1.11501543 -0.81130281 -0.64497745
                        #  -0.10221637 -0.32925792 -0.53646227 -0.54949522 -0.35537453 -0.28560236
                        #   0.23599237]
                        # print('magnitude greater than 1', magnitude_of_collapse_z_deviation > 1)
                        logger.warning(f'***Collapse is larger than one standard deviation.'
                                       f' Pose is *** being considered')
                        _poor_collapse.append(1)
                    else:
                        _poor_collapse.append(0)
                    #     logger.critical(
                    #         f'Total deviation={magnitude_of_collapse_z_deviation.sum()}. '
                    #         f'Mean={designed_indices_collapse_z.mean()}'
                    #         f'Standard Deviation={designed_indices_collapse_z.std()}')
                # _total_collapse_favorability.extend(_poor_collapse)
                # per_residue_design_indices[batch_slice] = _residue_indices_of_interest
                # per_residue_batch_collapse_z[batch_slice] = _per_residue_mini_batch_collapse_z
            else:  # Populate with null data
                _per_residue_mini_batch_collapse_z = _per_residue_evolution_cross_entropy.copy()
                _per_residue_mini_batch_collapse_z[:] = np.nan
                _poor_collapse = _per_residue_mini_batch_collapse_z[:, 0]

            return {
                # The below structures have a shape (batch_length, pose_length)
                'design_cross_entropy': _per_residue_design_cross_entropy,
                'evolution_cross_entropy': _per_residue_evolution_cross_entropy,
                'fragment_cross_entropy': _per_residue_fragment_cross_entropy,
                'collapse_z': _per_residue_mini_batch_collapse_z,
                'design_indices': _residue_indices_of_interest,
                'collapse_violation': _poor_collapse,
            }

        @torch.no_grad()  # Ensure no gradients are produced
        @resources.ml.batch_calculation(size=size, batch_length=batch_length,
                                        setup=ml.setup_pose_batch_for_proteinmpnn,
                                        compute_failure_exceptions=(RuntimeError,
                                                                    np.core._exceptions._ArrayMemoryError))
        def fragdock_design(batch_slice: slice,
                            pose_length: int = None,
                            **batch_parameters) -> dict[str, np.ndarray]:
            actual_batch_length = batch_slice.stop - batch_slice.start
            # Initialize pose data structures for interface design

            residue_mask_cpu = np.zeros((actual_batch_length, pose_length),
                                        dtype=np.int32)  # (batch, number_of_residues)
            bias_by_res = np.zeros((actual_batch_length, pose_length, 21),
                                   dtype=np.float32)  # (batch, number_of_residues, alphabet_length)
            # Stack the entity coordinates to make up a contiguous block for each pose
            # If entity_bb_coords are stacked, then must concatenate along axis=1 to get full pose
            new_coords = np.zeros((actual_batch_length, pose_length * num_model_residues, 3),
                                  dtype=np.float32)  # (batch, number_of_residues, coords_length)

            fragment_profiles = []
            # Use batch_idx to set new numpy arrays, transform_idx (includes perturb_idx) to set coords
            for batch_idx, transform_idx in enumerate(range(batch_slice.start, batch_slice.stop)):
                # Get the transformations based on the global index from batch_length
                update_pose_coords(transform_idx)
                new_coords[batch_idx] = getattr(pose, coords_type)

                # pose.find_and_split_interface(distance=cb_distance)
                # This is done in the below call
                add_fragments_to_pose()  # <- here generating fragments fresh
                # Reset the fragment_profile and fragment_map for each Entity before calculate_fragment_profile
                for entity in pose.entities:
                    entity._fragment_profile = {}
                    entity.fragment_map = {}
                    # entity.alpha.clear()

                # Load fragment_profile into the analysis
                pose.calculate_fragment_profile()
                # if pose.fragment_profile:
                fragment_profiles.append(pose.fragment_profile.as_array())
                # else:
                #     fragment_profiles.append(pose.fragment_profile.as_array())

                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()
                # # Remove saved pose attributes from the prior iteration calculations
                # pose.ss_index_array.clear(), pose.ss_type_array.clear()
                # pose.fragment_metrics.clear(), pose.fragment_pairs.clear()
                # for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
                #     try:
                #         delattr(pose, attribute)
                #     except AttributeError:
                #         pass
                #
                # # Calculate pose metrics
                # interface_metrics[design_id] = pose.interface_metrics()
                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()

                # Add all interface residues
                design_residues = []
                for number, residues_entities in pose.split_interface_residues.items():
                    design_residues.extend([residue.index for residue, _ in residues_entities])

                # Residues to design are 1, others are 0
                residue_mask_cpu[batch_idx, design_residues] = 1
                # Todo Should I use this?
                #  bias_by_res[batch_idx] = pose.fragment_profile.as_array()
                #  OR
                #  bias_by_res[batch_idx, fragment_residues] = pose.fragment_profile.as_array()[fragment_residues]
                #  If tied_beta is modified
                #  tied_beta[batch_idx] = ...

            # If entity_bb_coords are individually transformed, then axis=0 works
            perturbed_bb_coords = np.concatenate(new_coords, axis=0)

            # Format the bb coords for ProteinMPNN
            if pose.is_symmetric():
                # Make each set of coordinates symmetric. Lattice cases have uc_dimensions passed in update_pose_coords
                _perturbed_bb_coords = []
                for idx in range(perturbed_bb_coords.shape[0]):
                    _perturbed_bb_coords.append(pose.return_symmetric_coords(perturbed_bb_coords[idx]))

                # Let -1 fill in the pose length dimension with the number of residues
                # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
                perturbed_bb_coords = np.concatenate(_perturbed_bb_coords)

                # Symmetrize other arrays
                number_of_symmetry_mates = pose.number_of_symmetry_mates
                # (batch, number_of_sym_residues, ...)
                residue_mask_cpu = np.tile(residue_mask_cpu, (1, number_of_symmetry_mates))
                bias_by_res = np.tile(bias_by_res, (1, number_of_symmetry_mates, 1))

            # Reshape for ProteinMPNN
            logger.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')
            X = perturbed_bb_coords.reshape((actual_batch_length, -1, num_model_residues, 3))
            logger.debug(f'X.shape: {X.shape}')

            # Update different parameters to the identified device
            batch_parameters.update(ml.proteinmpnn_to_device(device, X=X, chain_M_pos=residue_mask_cpu,
                                                             bias_by_res=bias_by_res))
            # Different across poses
            # X = batch_parameters.pop('X')
            # residue_mask = batch_parameters.get('chain_M_pos', None)

            # # Potentially different across poses
            # bias_by_res = batch_parameters.get('bias_by_res', None)
            # Todo calculate individually if using some feature to describe order
            #  MUST reinstate the removal from scope after finished with this batch
            # decoding_order = pose.generate_proteinmpnn_decode_order(to_device=device)
            # decoding_order.repeat(actual_batch_length, 1)
            # with torch.no_grad():  # Ensure no gradients are produced
            # Unpack constant parameters and slice reused parameters only once
            # X_unbound = batch_parameters.pop('X')  # Remove once batch_calculation()
            # chain_mask = batch_parameters.pop('chain_mask')
            # chain_encoding = batch_parameters.pop('chain_encoding')
            # residue_idx = batch_parameters.pop('residue_idx')
            # mask = batch_parameters.pop('mask')
            # randn = batch_parameters.pop('randn')

            # Todo remove S from above unpacking
            #
            return ml.proteinmpnn_batch_design(batch_slice,
                                               mpnn_model,
                                               # temperatures=job.design.temperatures,
                                               pose_length=pose_length,
                                               # **parameters,  # (randn, S, chain_mask, chain_encoding, residue_idx, mask, temperatures, pose_length, bias_by_res, tied_pos, X_unbound)
                                               **batch_parameters  # (X, chain_M_pos, bias_by_res)
                                               )

        @torch.no_grad()  # Ensure no gradients are produced
        @resources.ml.batch_calculation(size=size, batch_length=batch_length,
                                        setup=ml.setup_pose_batch_for_proteinmpnn,
                                        compute_failure_exceptions=(RuntimeError,
                                                                    np.core._exceptions._ArrayMemoryError))
        def pose_batch_to_protein_mpnn(batch_slice: slice,
                                       X_unbound: torch.Tensor = None,
                                       S: torch.Tensor = None,
                                       chain_mask: torch.Tensor = None,
                                       chain_encoding: torch.Tensor = None,
                                       residue_idx: torch.Tensor = None,
                                       mask: torch.Tensor = None,
                                       randn: torch.Tensor = None,
                                       tied_pos: Iterable[Container] = None,
                                       **batch_parameters
                                       ) -> dict[str, np.ndarray]:
            actual_batch_length = batch_slice.stop - batch_slice.start
            # Get the null_idx
            mpnn_null_idx = resources.ml.MPNN_NULL_IDX
            # This parameter is pass as X for compatibility reasons
            # X_unbound = X
            # Initialize pose data structures for interface design
            residue_mask_cpu = np.zeros((actual_batch_length, pose_length),
                                        dtype=np.int32)  # (batch, number_of_residues)
            bias_by_res = np.zeros((actual_batch_length, pose_length, 21),
                                   dtype=np.float32)  # (batch, number_of_residues, alphabet_length)
            # Stack the entity coordinates to make up a contiguous block for each pose
            # If entity_bb_coords are stacked, then must concatenate along axis=1 to get full pose
            new_coords = np.zeros((actual_batch_length, pose_length * num_model_residues, 3),
                                  dtype=np.float32)  # (batch, number_of_residues, coords_length)

            fragment_profiles = []
            design_profiles = []
            # Use batch_idx to set new numpy arrays, transform_idx (includes perturb_idx) to set coords
            for batch_idx, transform_idx in enumerate(range(batch_slice.start, batch_slice.stop)):
                # Get the transformations based on the global index from batch_length
                update_pose_coords(transform_idx)
                new_coords[batch_idx] = getattr(pose, coords_type)

                # pose.find_and_split_interface(distance=cb_distance)
                # This is done in the below call
                add_fragments_to_pose()  # <- here generating fragments fresh
                # Reset the fragment_profile and fragment_map for each Entity before calculate_fragment_profile
                for entity in pose.entities:
                    entity._fragment_profile = {}
                    entity.fragment_map = {}
                    # entity.alpha.clear()

                # Load fragment_profile into the analysis
                pose.calculate_fragment_profile()
                # if pose.fragment_profile:
                fragment_profiles.append(pose.fragment_profile.as_array())
                # else:
                #     fragment_profiles.append(pose.fragment_profile.as_array())

                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()
                # # Remove saved pose attributes from the prior iteration calculations
                # pose.ss_index_array.clear(), pose.ss_type_array.clear()
                # pose.fragment_metrics.clear(), pose.fragment_pairs.clear()
                # for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
                #     try:
                #         delattr(pose, attribute)
                #     except AttributeError:
                #         pass
                #
                # # Calculate pose metrics
                # interface_metrics[design_id] = pose.interface_metrics()
                # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.interface_metrics()
                pose.calculate_profile()
                design_profiles.append(pssm_as_array(pose.profile))

                # Add all interface residues
                design_residues = []
                for number, residues_entities in pose.split_interface_residues.items():
                    design_residues.extend([residue.index for residue, _ in residues_entities])

                # Residues to design are 1, others are 0
                residue_mask_cpu[batch_idx, design_residues] = 1
                # Todo Should I use this?
                #  bias_by_res[batch_idx] = pose.fragment_profile.as_array()
                #  OR
                #  bias_by_res[batch_idx, fragment_residues] = pose.fragment_profile.as_array()[fragment_residues]
                #  If tied_beta is modified
                #  tied_beta[batch_idx] = ...

            # If entity_bb_coords are individually transformed, then axis=0 works
            perturbed_bb_coords = np.concatenate(new_coords, axis=0)

            # Format the bb coords for ProteinMPNN
            if pose.is_symmetric():
                # Make each set of coordinates symmetric. Lattice cases have uc_dimensions passed in update_pose_coords
                _perturbed_bb_coords = []
                for idx in range(perturbed_bb_coords.shape[0]):
                    _perturbed_bb_coords.append(pose.return_symmetric_coords(perturbed_bb_coords[idx]))

                # Let -1 fill in the pose length dimension with the number of residues
                # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
                perturbed_bb_coords = np.concatenate(_perturbed_bb_coords)

                # Symmetrize other arrays
                number_of_symmetry_mates = pose.number_of_symmetry_mates
                # (batch, number_of_sym_residues, ...)
                residue_mask_cpu = np.tile(residue_mask_cpu, (1, number_of_symmetry_mates))
                bias_by_res = np.tile(bias_by_res, (1, number_of_symmetry_mates, 1))

            # Reshape for ProteinMPNN
            logger.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')
            X = perturbed_bb_coords.reshape((actual_batch_length, -1, num_model_residues, 3))
            logger.debug(f'X.shape: {X.shape}')

            # Update different parameters to the identified device
            batch_parameters.update(ml.proteinmpnn_to_device(device, X=X, chain_M_pos=residue_mask_cpu,
                                                             bias_by_res=bias_by_res))
            # Different across poses
            X = batch_parameters.pop('X')
            residue_mask = batch_parameters.get('chain_M_pos', None)
            # # Potentially different across poses
            # bias_by_res = batch_parameters.get('bias_by_res', None)
            # Todo calculate individually if using some feature to describe order
            #  MUST reinstate the removal from scope after finished with this batch
            # decoding_order = pose.generate_proteinmpnn_decode_order(to_device=device)
            # decoding_order.repeat(actual_batch_length, 1)
            # with torch.no_grad():  # Ensure no gradients are produced
            # Unpack constant parameters and slice reused parameters only once
            # X_unbound = batch_parameters.pop('X')  # Remove once batch_calculation()
            # chain_mask = batch_parameters.pop('chain_mask')
            # chain_encoding = batch_parameters.pop('chain_encoding')
            # residue_idx = batch_parameters.pop('residue_idx')
            # mask = batch_parameters.pop('mask')
            # randn = batch_parameters.pop('randn')
            # Clone the data from the sequence tensor so that it can be set with the null token below
            S_design_null = S.detach().clone()
            # Get the provided batch_length from wrapping function. actual_batch_length may be smaller on last batch
            batch_length = X_unbound.shape[0]
            if actual_batch_length != batch_length:
                # Slice these for the last iteration
                X_unbound = X_unbound[:actual_batch_length]  # , None)
                chain_mask = chain_mask[:actual_batch_length]  # , None)
                chain_encoding = chain_encoding[:actual_batch_length]  # , None)
                residue_idx = residue_idx[:actual_batch_length]  # , None)
                mask = mask[:actual_batch_length]  # , None)
                randn = randn[:actual_batch_length]
                S_design_null = S_design_null[:actual_batch_length]  # , None)
                # Unpack, unpacked keyword args
                omit_AA_mask = batch_parameters.get('omit_AA_mask')
                pssm_coef = batch_parameters.get('pssm_coef')
                pssm_bias = batch_parameters.get('pssm_bias')
                pssm_log_odds_mask = batch_parameters.get('pssm_log_odds_mask')
                # Set keyword args
                batch_parameters['omit_AA_mask'] = omit_AA_mask[:actual_batch_length]
                batch_parameters['pssm_coef'] = pssm_coef[:actual_batch_length]
                batch_parameters['pssm_bias'] = pssm_bias[:actual_batch_length]
                batch_parameters['pssm_log_odds_mask'] = pssm_log_odds_mask[:actual_batch_length]

            # Use the sequence as an unknown token then guess the probabilities given the remaining
            # information, i.e. the sequence and the backbone
            S_design_null[residue_mask.type(torch.bool)] = mpnn_null_idx
            chain_residue_mask = chain_mask * residue_mask

            decoding_order = ml.create_decoding_order(randn, chain_mask, tied_pos=tied_pos, to_device=device)
            # See if the pose is useful to design based on constraints of collapse

            # Measure the conditional amino acid probabilities at each residue to see
            # how they compare to various profiles from the Pose multiple sequence alignment
            # If conditional_probs() are measured, then we need a batched_decoding order
            # conditional_start_time = time.time()
            # Calculations with this are done using cpu memory and numpy
            # logger.critical(f'Before model forward pass\nmemory_allocated: {torch.cuda.memory_allocated()}'
            #                 f'\nmemory_reserved: {torch.cuda.memory_reserved()}')
            conditional_log_probs_null_seq = \
                mpnn_model(X, S_design_null, mask, chain_residue_mask, residue_idx, chain_encoding,
                           None,  # This argument is provided but with below args, is not used
                           use_input_decoding_order=True, decoding_order=decoding_order).cpu()
            _residue_indices_of_interest = residue_mask_cpu[:, :pose_length].astype(bool)
            #  Taking the KL divergence would indicate how divergent the interfaces are from the
            #  surface. This should be simultaneously minimized (i.e. lowest evolutionary divergence)
            #  while the aa frequency distribution cross_entropy compared to the fragment profile is
            #  minimized
            asu_conditional_softmax_null_seq = \
                np.exp(conditional_log_probs_null_seq[:, :pose_length])
            # asu_conditional_softmax
            # tensor([[[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          ...,
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]],
            #          ...
            #         [[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
            #          ...,
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
            #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]]])

            if pose.fragment_profile:
                # Process the fragment_profiles into an array for cross entropy
                fragment_profile_array = np.array(fragment_profiles)
                # RuntimeWarning: divide by zero encountered in log
                # np.log causes -inf at 0, thus we need to correct these to a very large number
                batch_fragment_profile = torch.from_numpy(np.nan_to_num(fragment_profile_array, copy=False, nan=np.nan))
                # print('batch_fragment_profile', batch_fragment_profile[:, 20:23])
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_fragment_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_fragment_profile,
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
                # All per_residue metrics look the same. Shape batch_length, number_of_residues
                # per_residue_evolution_cross_entropy[batch_slice]
                # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
                #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
                #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
            else:
                _per_residue_fragment_cross_entropy = np.empty_like(_residue_indices_of_interest, dtype=np.float32)
                _per_residue_fragment_cross_entropy[:] = np.nan

            if pose.evolutionary_profile:
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_evolution_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_evolutionary_profile[:actual_batch_length],
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
            else:  # Populate with null data
                _per_residue_evolution_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
                _per_residue_evolution_cross_entropy[:] = np.nan

            if pose.profile:
                # Process the design_profiles into an array for cross entropy
                batch_design_profile = torch.from_numpy(np.array(design_profiles))
                # print('batch_design_profile', batch_design_profile[:, 20:23])
                # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
                _per_residue_design_cross_entropy = \
                    cross_entropy(asu_conditional_softmax_null_seq[:, :, :mpnn_null_idx],
                                  batch_design_profile,
                                  per_entry=True)
                #                 mask=_residue_indices_of_interest,
                #                 axis=1)
                # All per_residue metrics look the same. Shape batch_length, number_of_residues
                # per_residue_evolution_cross_entropy[batch_slice]
                # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
                #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
                #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
            else:  # Populate with null data
                _per_residue_design_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
                _per_residue_design_cross_entropy[:] = np.nan

            if collapse_profile.size:  # Not equal to zero
                # Take the hydrophobic collapse of the log probs to understand the profiles "folding"
                _poor_collapse = []
                _per_residue_mini_batch_collapse_z = \
                    np.empty((actual_batch_length, pose_length), dtype=np.float32)
                for pose_idx in range(actual_batch_length):
                    # Only include the residues in the ASU
                    design_probs_collapse = \
                        hydrophobic_collapse_index(asu_conditional_softmax_null_seq[pose_idx],
                                                   # asu_unconditional_softmax,
                                                   alphabet_type=ml.mpnn_alphabet)
                    # Todo?
                    #  design_probs_collapse = \
                    #      hydrophobic_collapse_index(asu_conditional_softmax,
                    #                                 alphabet_type=ml.mpnn_alphabet)
                    # Compare the sequence collapse to the pose collapse
                    # USE:
                    #  contact_order_per_res_z, reference_collapse, collapse_profile
                    # print('HCI profile mean', collapse_profile_mean)
                    # print('HCI profile std', collapse_profile_std)
                    _per_residue_mini_batch_collapse_z[pose_idx] = collapse_z = \
                        z_score(design_probs_collapse, collapse_profile_mean, collapse_profile_std)
                    # folding_loss = ml.sequence_nllloss(S_sample, design_probs_collapse)  # , mask_for_loss)
                    pose_idx_residues_of_interest = _residue_indices_of_interest[pose_idx]
                    designed_indices_collapse_z = collapse_z[pose_idx_residues_of_interest]
                    # magnitude_of_collapse_z_deviation = np.abs(designed_indices_collapse_z)
                    # Check if dock has collapse larger than collapse_significance_threshold and increased collapse
                    if np.any(np.logical_and(design_probs_collapse[pose_idx_residues_of_interest]
                                             > collapse_significance_threshold,
                                             designed_indices_collapse_z > 0)):
                        # Todo save this
                        print('design_probs_collapse', design_probs_collapse[pose_idx_residues_of_interest])
                        print('designed_indices_collapse_z', designed_indices_collapse_z)
                        # print('magnitude greater than 1', magnitude_of_collapse_z_deviation > 1)
                        logger.warning(f'***Collapse is larger than one standard deviation.'
                                       f' Pose is *** being considered')
                        _poor_collapse.append(1)
                    else:
                        _poor_collapse.append(0)
                    #     logger.critical(
                    #         f'Total deviation={magnitude_of_collapse_z_deviation.sum()}. '
                    #         f'Mean={designed_indices_collapse_z.mean()}'
                    #         f'Standard Deviation={designed_indices_collapse_z.std()}')
                # _total_collapse_favorability.extend(_poor_collapse)
                # per_residue_design_indices[batch_slice] = _residue_indices_of_interest
                # per_residue_batch_collapse_z[batch_slice] = _per_residue_mini_batch_collapse_z
            else:  # Populate with null data
                _per_residue_mini_batch_collapse_z = np.empty_like(_per_residue_evolution_cross_entropy)
                # _per_residue_mini_batch_collapse_z = _per_residue_evolution_cross_entropy.copy()
                _per_residue_mini_batch_collapse_z[:] = np.nan
                _poor_collapse = _per_residue_mini_batch_collapse_z[:, 0]

            dock_fit_parameters = {
                # The below structures have a shape (batch_length, pose_length)
                'design_cross_entropy': _per_residue_design_cross_entropy,
                'evolution_cross_entropy': _per_residue_evolution_cross_entropy,
                'fragment_cross_entropy': _per_residue_fragment_cross_entropy,
                'collapse_z': _per_residue_mini_batch_collapse_z,
                'design_indices': _residue_indices_of_interest,
                'collapse_violation': _poor_collapse,
            }

            batch_sequences = []
            _per_residue_complex_sequence_loss = []
            _per_residue_unbound_sequence_loss = []
            number_of_temps = len(job.design.temperatures)
            for temperature in job.design.temperatures:
                # Todo add _total_collapse_favorability skipping to the selection mechanism?
                sample_start_time = time.time()
                sample_dict = mpnn_sample(X, randn,  # decoding_order,
                                          S_design_null,  # S[:actual_batch_length],
                                          chain_mask, chain_encoding, residue_idx, mask,
                                          temperature=temperature,
                                          # omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                                          # chain_M_pos=residue_mask,  # separate_parameters
                                          # omit_AA_mask=omit_AA_mask[:actual_batch_length],
                                          # pssm_coef=pssm_coef[:actual_batch_length],
                                          # pssm_bias=pssm_bias[:actual_batch_length],
                                          # pssm_multi=pssm_multi,  # batch_parameters
                                          # pssm_log_odds_flag=pssm_log_odds_flag,  # batch_parameters
                                          # pssm_log_odds_mask=pssm_log_odds_mask[:actual_batch_length],
                                          # pssm_bias_flag=pssm_bias_flag,  # batch_parameters
                                          tied_pos=tied_pos,  # parameters
                                          # tied_beta=tied_beta,  # parameters
                                          # bias_by_res=bias_by_res,  # separate_parameters
                                          # bias_by_res=bias_by_res[:actual_batch_length],
                                          **batch_parameters)
                logger.info(f'Sample calculation took {time.time() - sample_start_time:8f}')
                S_sample = sample_dict['S']
                # decoding_order_out = sample_dict['decoding_order']
                decoding_order_out = decoding_order  # When using the same decoding order for all
                # _X_unbound = X_unbound[:actual_batch_length]
                unbound_log_prob_start_time = time.time()
                unbound_log_probs = \
                    mpnn_model(X_unbound, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                               None,  # This argument is provided but with below args, is not used
                               use_input_decoding_order=True, decoding_order=decoding_order_out).cpu()

                logger.debug(f'Unbound log prob calculation took {time.time() - unbound_log_prob_start_time:8f}')
                log_probs_start_time = time.time()
                complex_log_probs = \
                    mpnn_model(X, S_sample, mask, chain_residue_mask, residue_idx, chain_encoding,
                               None,  # This argument is provided but with below args, is not used
                               use_input_decoding_order=True, decoding_order=decoding_order_out).cpu()
                # complex_log_probs is
                # tensor([[[-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
                #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
                #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
                #          ...,
                #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
                #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772],
                #          [-2.7691, -3.5265, -2.9001,  ..., -3.3623, -3.0247, -4.2772]],
                #         [[-2.6934, -4.0610, -2.6506, ..., -4.2404, -3.4620, -4.8641],
                #          [-2.8753, -4.3959, -2.4042,  ..., -4.4922, -3.5962, -5.1403],
                #          [-2.5235, -4.0181, -2.7738,  ..., -4.2454, -3.4768, -4.8088],
                #          ...,
                #          [-3.4500, -4.4373, -3.7814,  ..., -5.1637, -4.6107, -5.2295],
                #          [-0.9690, -4.9492, -3.9373,  ..., -2.0154, -2.2262, -4.3334],
                #          [-3.1118, -4.3809, -3.8763,  ..., -4.7145, -4.1524, -5.3076]]])
                logger.debug(f'Log prob calculation took {time.time() - log_probs_start_time:8f}')
                # Score the redesigned structure-sequence
                # mask_for_loss = chain_mask_and_mask*residue_mask
                # batch_scores = ml.sequence_nllloss(S_sample, complex_log_probs, mask_for_loss, per_residue=False)
                # batch_scores is
                # tensor([2.1039, 2.0618, 2.0802, 2.0538, 2.0114, 2.0002], device='cuda:0')
                # Format outputs
                _batch_sequences = S_sample[:, :pose_length].cpu()
                batch_sequences.append(_batch_sequences)
                _per_residue_complex_sequence_loss.append(
                    ml.sequence_nllloss(_batch_sequences, complex_log_probs[:, :pose_length]).numpy())
                _per_residue_unbound_sequence_loss.append(
                    ml.sequence_nllloss(_batch_sequences, unbound_log_probs[:, :pose_length]).numpy())

            # return {
            _return = {
                # The below structures have a shape (batch_length, number_of_temperatures, pose_length)
                'sequences':
                    np.concatenate(batch_sequences, axis=1).reshape(actual_batch_length, number_of_temps,
                                                                    pose_length),
                'complex_sequence_loss':
                    np.concatenate(_per_residue_complex_sequence_loss, axis=1).reshape(actual_batch_length,
                                                                                       number_of_temps,
                                                                                       pose_length),
                'unbound_sequence_loss':
                    np.concatenate(_per_residue_unbound_sequence_loss, axis=1).reshape(actual_batch_length,
                                                                                       number_of_temps,
                                                                                       pose_length),
            }
            dock_fit_parameters.update(_return)

            return dock_fit_parameters

        # Initialize correct data for the calculation
        proteinmpnn_kwargs = dict(pose_length=pose_length,
                                  temperatures=job.design.temperatures)
        number_of_temperatures = len(job.design.temperatures)
        # probabilities = np.empty((size, number_of_residues, mpnn_alphabet_length, dtype=np.float32))
        # Include design indices in both dock and design
        per_residue_design_indices = np.zeros((size, pose_length), dtype=bool)
        if job.dock.proteinmpnn_score:
            # Set up ProteinMPNN output data structures
            # To use torch.nn.NLLL() must use dtype Long -> np.int64, not Int -> np.int32
            # generated_sequences = np.empty((size, pose_length), dtype=np.int64)
            per_residue_evolution_cross_entropy = np.empty((size, pose_length), dtype=np.float32)
            per_residue_fragment_cross_entropy = np.empty_like(per_residue_evolution_cross_entropy)
            per_residue_design_cross_entropy = np.empty_like(per_residue_evolution_cross_entropy)
            per_residue_batch_collapse_z = np.zeros_like(per_residue_evolution_cross_entropy)
            collapse_violation = np.zeros((size,), dtype=bool)
            dock_returns = {'design_cross_entropy': per_residue_design_cross_entropy,
                            'evolution_cross_entropy': per_residue_evolution_cross_entropy,
                            'fragment_cross_entropy': per_residue_fragment_cross_entropy,
                            'collapse_z': per_residue_batch_collapse_z,
                            'design_indices': per_residue_design_indices,
                            'collapse_violation': collapse_violation}
        else:
            dock_returns = {}

        if job.design.sequences:
            generated_sequences = np.empty((size, number_of_temperatures, pose_length), dtype=np.int64)
            per_residue_complex_sequence_loss = np.empty(generated_sequences.shape, dtype=np.float32)
            per_residue_unbound_sequence_loss = np.empty_like(per_residue_complex_sequence_loss)
            design_returns = {'sequences': generated_sequences,
                              'complex_sequence_loss': per_residue_complex_sequence_loss,
                              'unbound_sequence_loss': per_residue_unbound_sequence_loss,
                              'design_indices': per_residue_design_indices
                              }
        else:
            design_returns = {}

        # Perform the calculation
        if job.dock.proteinmpnn_score and job.design.sequences:
            # This is used for October 2022 working dock/design protocol
            calculation_method = 'docking analysis/sequence design'
            dock_design_returns = {**dock_returns, **design_returns}
            sequences_and_scores = pose_batch_to_protein_mpnn(**proteinmpnn_kwargs,
                                                              return_containers=dock_design_returns,
                                                              setup_args=(device,),
                                                              setup_kwargs=parameters
                                                              )
            per_residue_design_cross_entropy = sequences_and_scores['design_cross_entropy']
            per_residue_evolution_cross_entropy = sequences_and_scores['evolution_cross_entropy']
            per_residue_fragment_cross_entropy = sequences_and_scores['fragment_cross_entropy']
            per_residue_batch_collapse_z = sequences_and_scores['collapse_z']
            per_residue_design_indices = sequences_and_scores['design_indices']
            collapse_violation = sequences_and_scores['collapse_violation']
            generated_sequences = sequences_and_scores['sequences']
            per_residue_complex_sequence_loss = sequences_and_scores['complex_sequence_loss']
            per_residue_unbound_sequence_loss = sequences_and_scores['unbound_sequence_loss']
        elif job.dock.proteinmpnn_score:
            calculation_method = 'docking analysis'
            # This is used for understanding dock fit only
            sequences_and_scores = check_dock_for_designability(**proteinmpnn_kwargs,
                                                                return_containers=dock_returns,
                                                                setup_args=(device,),
                                                                setup_kwargs=parameters)

            per_residue_design_cross_entropy = sequences_and_scores['design_cross_entropy']
            per_residue_evolution_cross_entropy = sequences_and_scores['evolution_cross_entropy']
            per_residue_fragment_cross_entropy = sequences_and_scores['fragment_cross_entropy']
            per_residue_batch_collapse_z = sequences_and_scores['collapse_z']
            per_residue_design_indices = sequences_and_scores['design_indices']
            collapse_violation = sequences_and_scores['collapse_violation']
            generated_sequences = per_residue_complex_sequence_loss = per_residue_unbound_sequence_loss = None
        elif job.design.sequences:  # This is used for design only
            calculation_method = 'sequence design'
            sequences_and_scores = fragdock_design(**proteinmpnn_kwargs,
                                                   return_containers=design_returns,
                                                   setup_args=(device,),
                                                   setup_kwargs=parameters
                                                   )
            generated_sequences = sequences_and_scores['sequences']
            per_residue_complex_sequence_loss = sequences_and_scores['complex_sequence_loss']
            per_residue_unbound_sequence_loss = sequences_and_scores['unbound_sequence_loss']
            per_residue_design_indices = sequences_and_scores['design_indices']
            per_residue_design_cross_entropy = per_residue_evolution_cross_entropy = \
                per_residue_fragment_cross_entropy = per_residue_batch_collapse_z = \
                collapse_violation = None
        else:
            raise RuntimeError(f"Logic shouldn't allow this to happen")

        logger.info(f'ProteinMPNN {calculation_method} took {time.time() - proteinmpnn_time_start:8f}')

        # # Truncate the sequences to the ASU
        # if pose.is_symmetric():
        #     generated_sequences = generated_sequences[:, :pose_length]
        #     per_residue_complex_sequence_loss = per_residue_complex_sequence_loss[:, :pose_length]
        #     per_residue_unbound_sequence_loss = per_residue_unbound_sequence_loss[:, :pose_length]
        #     # probabilities = probabilities[:, :pose_length]
        #     # sequences = sequences[:, :pose_length]
        # Create design_ids for each of the pose_ids plus the identified sequence
        design_ids = [f'{pose_id}-design{design_idx:04d}' for pose_id in pose_ids
                      for design_idx in range(1, 1 + number_of_temperatures)]
        # pose_ids, design_ids = list(zip(*[(pose_id, f'{pose_id}-design{design_idx:04d}') for pose_id in pose_ids
        #                                   for design_idx in range(1, 1 + number_of_temperatures)]))
        design_id_iterator = iter(design_ids)
        for pose_id in pose_ids:
            _pose_transformation = pose_transformations.pop(pose_id)
            for design_idx in range(1, 1 + number_of_temperatures):
                pose_transformations[next(design_id_iterator)] = _pose_transformation
        # sequences = numeric_to_sequence(generated_sequences)
        # # Format the sequences from design with shape (size, number_of_temperatures, pose_length)
        # # to (size * number_of_temperatures, pose_length)
        # # generated_sequences = generated_sequences.reshape(-1, pose_length)
        # # per_residue_complex_sequence_loss = per_residue_complex_sequence_loss.reshape(-1, pose_length)
        # # per_residue_unbound_sequence_loss = per_residue_unbound_sequence_loss.reshape(-1, pose_length)
        # sequences = sequences.reshape(-1, pose_length)
    else:
        design_ids = pose_ids

    # Sort cluster members by the highest metric and take some highest number of members
    # Todo filter by score for the best
    #  Collect scores according to job.dock.score
    #
    filter = False
    if filter:
        if job.dock.score:
            metric = sequences_and_scores.get(job.dock.score)
            if metric is None:
                raise ValueError(f"The metric {job.dock.score} wasn't collected and therefore, can't be selected")

            # The use of perturbation is inherently handled. Don't need check
            # if job.dock.perturb_dof:
            number_of_chosen_hits = math.sqrt(total_perturbation_size)
            number_of_transform_seeds = int(number_of_transforms // total_perturbation_size)
            perturb_passing_indices = []
            for idx in range(number_of_transform_seeds):
                # Slice the chosen metric along the local perturbation
                perturb_metric = metric[idx * total_perturbation_size: (idx+1) * total_perturbation_size]
                # Sort the slice and take the top number of hits
                top_candidate_indices = perturb_metric.argsort()[:number_of_chosen_hits]
                # perturb_metric[top_candidate_indices]
                # Scale the selected indices into the transformation indices reference
                perturb_passing_indices.extend((top_candidate_indices + idx*total_perturbation_size).tolist())

            remove_non_viable_indices(perturb_passing_indices)
        else:  # Output all
            design_ids = pose_ids

    # Create a Models instance to collect each model
    if job.write_trajectory:
        trajectory_models = Models()

    # Define functions for outputting docked poses
    def output_pose(_out_dir: AnyStr, _pose_id: AnyStr, uc_dimensions: np.ndarray = None):
        """Format the current Pose for output using the job parameters

        Args:
            _out_dir: The directory to write files
            _pose_id: The particular identifier for the pose
            uc_dimensions: If this is a lattice, the crystal dimensions
        """
        out_path = os.path.join(_out_dir, _pose_id)
        putils.make_path(out_path)

        # Set the ASU, then write to a file
        pose.set_contacting_asu(distance=cb_distance)
        if sym_entry.unit_cell:  # 2, 3 dimensions
            cryst_record = generate_cryst1_record(uc_dimensions, sym_entry.resulting_symmetry)
        else:
            cryst_record = None

        if job.write_structures:
            pose_path = os.path.join(out_path, putils.asu)
            pose.write(out_path=pose_path, header=cryst_record)
        else:
            pose_path = None

        if job.write_trajectory:
            raise NotImplementedError('make more reliable!!')
            nonlocal idx
            if idx % 2 == 0:
                new_pose = copy.copy(pose)
                # new_pose = copy.copy(pose.models[0])
                for entity in new_pose.chains[1:]:  # new_pose.entities[1:]:
                    entity.chain_id = 'D'
                    # Todo make more reliable
                    # Todo NEED TO MAKE SymmetricModel copy .entities and .chains correctly!
                trajectory_models.append_model(new_pose)

        # Todo group by input model... not entities
        # Write Model1, Model2
        if job.write_oligomers:
            for entity in pose.entities:
                entity.write(oligomer=True, out_path=os.path.join(out_path, f'{entity.name}_{_pose_id}.pdb'))

        # Write assembly files
        if job.output_assembly:
            if sym_entry.unit_cell:  # 2, 3 dimensions
                if job.output_surrounding_uc:
                    assembly_path = os.path.join(out_path, f'{_pose_id}_{putils.surrounding_unit_cells}')
                else:
                    assembly_path = os.path.join(out_path, f'{_pose_id}_{putils.assembly}')
            else:  # 0 dimension
                assembly_path = os.path.join(out_path, f'{_pose_id}_{putils.assembly}')
            pose.write(assembly=True, out_path=assembly_path, header=cryst_record,
                       surrounding_uc=job.output_surrounding_uc)

        # Write fragment files
        if job.write_fragments:
            # Make directories to output matched fragment files
            matching_fragments_dir = os.path.join(out_path, putils.frag_dir)
            putils.make_path(matching_fragments_dir)
            # high_qual_match for fragments that were matched with z values <= 1, otherwise, low_qual_match
            # high_quality_matches_dir = os.path.join(matching_fragments_dir, 'high_qual_match')
            # low_quality_matches_dir = os.path.join(matching_fragments_dir, 'low_qual_match')
            pose.write_fragment_pairs(out_path=matching_fragments_dir)

        logger.info(f'\tSUCCESSFUL DOCKED POSE: {out_path}')

        return pose_path

    def terminate():
        """Finalize any remaining work and return to the caller"""
        if job.write_trajectory:
            if sym_entry.unit_cell:
                logger.warning('No unit cell dimensions applicable to the trajectory file.')

            trajectory_models.write(out_path=os.path.join(out_dir, 'trajectory_oligomeric_models.pdb'),
                                    oligomer=True)

    # Get metrics for each Pose
    # Set up data structures
    idx_slice = pd.IndexSlice
    interface_local_density = {}
    # all_pose_divergence = []
    # all_probabilities = {}
    fragment_profile_frequencies = []
    pose_paths = []
    nan_blank_data = list(repeat(np.nan, pose_length))
    for idx, design_id in enumerate(design_ids):
        # Add the next set of coordinates
        update_pose_coords(idx)

        # if total_perturbation_size > 1:
        add_fragments_to_pose()  # <- here generating fresh
        # else:
        #     # Here, loading fragments. No self-symmetric interactions will be generated!
        #     # where idx is the actual transform idx
        #     add_fragments_to_pose(all_passing_ghost_indices[idx],
        #                           all_passing_surf_indices[idx],
        #                           all_passing_z_scores[idx])

        if job.output:
            pose_paths.append(output_pose(out_dir, design_id))

        # Reset the fragment_map and fragment_profile for each Entity before calculate_fragment_profile
        for entity in pose.entities:
            entity._fragment_profile = {}
            entity.fragment_map = {}
            # entity.alpha.clear()

        # Load fragment_profile into the analysis
        pose.calculate_fragment_profile()
        # This could be an empty array if no fragments were found
        fragment_profile_array = pose.fragment_profile.as_array()

        # Remove saved pose attributes from the prior iteration calculations
        pose.ss_index_array.clear(), pose.ss_type_array.clear()
        pose.fragment_metrics.clear(), pose.fragment_pairs.clear()
        for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
            try:
                delattr(pose, attribute)
            except AttributeError:
                pass

        # Calculate pose metrics
        interface_metrics[design_id] = pose.interface_metrics()

        # if job.design.sequences:
        if proteinmpnn_used:
            # Todo, hook this into analysis
            #  This assumes that the pose already has .evolutionary_profile and .fragment_profile attributes
            #  pose.add_profile(evolution=job.design.evolution_constraint,
            #                   fragments=job.generate_fragments)
            #  # if pose.profile:
            #  design_profile_array = pssm_as_array(pose.profile)
            #  # else:
            #  #     pose.log.info('Design has no fragment information')
            if job.dock.proteinmpnn_score:
                # dock_per_residue_design_cross_entropy = per_residue_design_cross_entropy[idx]
                # dock_per_residue_evolution_cross_entropy = per_residue_evolution_cross_entropy[idx]
                # dock_per_residue_fragment_cross_entropy = per_residue_fragment_cross_entropy[idx]
                # dock_per_residue_design_indices = per_residue_design_indices[idx]
                # dock_per_residue_batch_collapse_z = per_residue_batch_collapse_z[idx]
                design_dock_params = {
                    'proteinmpnn_v_design_cross_entropy': per_residue_design_cross_entropy[idx],
                    'proteinmpnn_v_evolution_cross_entropy': per_residue_evolution_cross_entropy[idx],
                    'proteinmpnn_v_fragment_cross_entropy': per_residue_fragment_cross_entropy[idx],
                    'designed_residues_total': per_residue_design_indices[idx],
                    'collapse_profile_z': per_residue_batch_collapse_z[idx],
                }
                # 'collapse_profile_z' - The z score for the collapse profile as measured from an evolutionary
                # collapse profile
            else:
                design_dock_params = {}

            if job.design.sequences:
                dock_per_residue_design_indices = per_residue_design_indices[idx]
                designed_sequences = generated_sequences[idx]
                dock_per_residue_complex_sequence_loss = per_residue_complex_sequence_loss[idx]
                dock_per_residue_unbound_sequence_loss = per_residue_unbound_sequence_loss[idx]

                for temp_idx, design_idx in enumerate(range(idx * number_of_temperatures,
                                                            (idx+1) * number_of_temperatures)):
                    design_id = design_ids[design_idx]
                    if job.design.structures:
                        # Todo use the template protocol from protocols.py
                        #  if job.design.alphafold:
                        #      pose.predict_structure()
                        #  else:
                        #      pose.refine()
                        interface_local_density[design_id] = pose.local_density_interface()
                        per_res_interface_metrics = pose.get_per_residue_interface_metrics()
                    else:
                        per_res_interface_metrics = {}
                    # For each Pose, save each sequence design data such as energy # probabilites
                    # all_probabilities[design_id] = probabilities[idx]
                    # Todo process the all_probabilities to a DataFrame?
                    #  The probabilities are the actual probabilities at each residue for each AA
                    #  These differ from the log_probabilities in that those are scaled by the log()
                    #  and therefore are negative. The use of probabilities is how I have calculated divergence.
                    #  Perhaps I should transition to take the log of probabilities and calculate the loss.
                    # all_probabilities is
                    # {'2gtr-3m6n-DEGEN_1_1-ROT_13_10-TX_1-PT_1':
                    #  array([[1.55571969e-02, 6.64833433e-09, 3.03523801e-03, ...,
                    #          2.94689467e-10, 8.92133514e-08, 6.75683381e-12],
                    #         [9.43517406e-03, 2.54900701e-09, 4.43358254e-03, ...,
                    #          2.19431431e-10, 8.18614296e-08, 4.94338381e-12],
                    #         [1.50658926e-02, 1.43449803e-08, 3.27082584e-04, ...,
                    #          1.70684064e-10, 8.77646258e-08, 6.67974660e-12],
                    #         ...,
                    #         [1.23516358e-07, 2.98688293e-13, 3.48888407e-09, ...,
                    #          1.17041141e-14, 4.72279464e-12, 5.79130243e-16],
                    #         [9.99999285e-01, 2.18584519e-19, 3.87702094e-16, ...,
                    #          7.12933229e-07, 5.22657113e-13, 3.19411591e-17],
                    #         [2.11755684e-23, 2.32944583e-23, 3.86148234e-23, ...,
                    #          1.16764793e-22, 1.62743156e-23, 7.65081924e-23]]),
                    #  '2gtr-3m6n-DEGEN_1_1-ROT_13_10-TX_1-PT_2':
                    #  array([[1.72123183e-02, 7.31348226e-09, 3.28084361e-03, ...,
                    #          3.16341731e-10, 9.09206364e-08, 7.41259137e-12],
                    #         [6.17256807e-03, 1.86070248e-09, 2.70802877e-03, ...,
                    #          1.61229460e-10, 5.94660143e-08, 3.73394328e-12],
                    #         [1.28052337e-02, 1.10993081e-08, 3.89973022e-04, ...,
                    #          2.21829027e-10, 1.03226760e-07, 8.43660298e-12],
                    #         ...,
                    #         [1.31807008e-06, 2.47859654e-12, 2.27575967e-08, ...,
                    #          5.34223104e-14, 2.06900348e-11, 3.35126595e-15],
                    #         [9.99999821e-01, 1.26853575e-19, 2.05691231e-16, ...,
                    #          2.02439509e-07, 5.02121131e-13, 1.38719620e-17],
                    #         [2.01858383e-23, 2.29340987e-23, 3.59583879e-23, ...,
                    #          1.13548109e-22, 1.60868618e-23, 7.25537526e-23]])}

                    # Calculate sequence statistics
                    # Todo get the below mechanism clean
                    # Before calculation, we must set this (v) to get the correct values from the profile
                    pose._sequence_numeric = designed_sequences[temp_idx]
                    # Todo these are not Softmax probabilities
                    try:
                        fragment_profile_frequencies.append(
                            pose.get_sequence_probabilities_from_profile(precomputed=fragment_profile_array))
                    except IndexError as error:  # We are missing fragments for this Pose
                        logger.warning(f"We didn't find any fragment information... due to: {error}")
                        #                "\nSetting the pose.fragment_profile = None")
                        # raise IndexError(f'With new updates to calculate_fragment_profile this code should be '
                        #                  f'unreachable. Original error:\n{error}')
                        # pose.fragment_profile = None

                    # observed, divergence = \
                    #     calculate_sequence_observations_and_divergence(pose_alignment,
                    #                                                    profile_background,
                    #                                                    interface_indexer)
                    # # Get pose sequence divergence
                    # divergence_s = pd.Series({f'{divergence_type}_per_residue': _divergence.mean()
                    #                           for divergence_type, _divergence in divergence.items()},
                    #                          name=design_id)
                    # all_pose_divergence.append(divergence_s)
                    # Todo extract the observed values out of the observed dictionary
                    #  Each Pose only has one trajectory, so measurement of divergence is pointless (no distribution)
                    # observed_dfs = []
                    # # Todo must ensure the observed_values is the length of the design_ids
                    # # for profile, observed_values in observed.items():
                    # #     scores_df[f'observed_{profile}'] = observed_values.mean(axis=1)
                    # #     observed_dfs.append(pd.DataFrame(data=observed_values, index=design_id,
                    # #                                      columns=pd.MultiIndex.from_product([residue_numbers,
                    # #                                                                          [f'observed_{profile}']]))
                    # #                         )
                    # # Add observation information into the residue_df
                    # residue_df = pd.concat([residue_df] + observed_dfs, axis=1)
                    # Todo get divergence?
                    # Get the negative log likelihood of the .evolutionary_ and .fragment_profile
                    torch_numeric = torch.from_numpy(pose.sequence_numeric)
                    # if pose.evolutionary_profile:
                    if measure_evolution:
                        per_residue_evolutionary_profile_scores = \
                            ml.sequence_nllloss(torch_numeric, torch_log_evolutionary_profile)
                    else:
                        per_residue_evolutionary_profile_scores = nan_blank_data

                    if pose.fragment_profile:
                        # np.log causes -inf at 0, thus we correct these to a very large number
                        corrected_frag_array = np.nan_to_num(np.log(fragment_profile_array), copy=False,
                                                             nan=np.nan, neginf=zero_probability_frag_value)
                        per_residue_fragment_profile_scores = \
                            ml.sequence_nllloss(torch_numeric, torch.from_numpy(corrected_frag_array))
                        # Find the non-zero sites in the profile
                        # interface_indexer = [residue.index for residue in pose.interface_residues]
                        # interface_observed_from_fragment_profile = fragment_profile_frequencies[idx][interface_indexer]
                    else:
                        per_residue_fragment_profile_scores = nan_blank_data

                    per_residue_data[design_id] = {
                        **per_res_interface_metrics,
                        **design_dock_params,
                        'designed_residues_total': dock_per_residue_design_indices,
                        'complex': dock_per_residue_complex_sequence_loss[temp_idx],
                        'unbound': dock_per_residue_unbound_sequence_loss[temp_idx],
                        # 'proteinmpnn_v_design_cross_entropy': dock_per_residue_design_cross_entropy,
                        # 'proteinmpnn_v_evolution_cross_entropy': dock_per_residue_evolution_cross_entropy,
                        # 'proteinmpnn_v_fragment_cross_entropy': dock_per_residue_fragment_cross_entropy,
                        # 'collapse_profile_z': dock_per_residue_batch_collapse_z,
                        'evolution_sequence_loss': per_residue_evolutionary_profile_scores,
                        'fragment_sequence_loss': per_residue_fragment_profile_scores,
                        # 'bound': 0.,  # copy(entity_energies),
                        # copy(entity_energies),
                        # 'solv_complex': 0., 'solv_bound': 0.,
                        # copy(entity_energies),
                        # 'solv_unbound': 0.,  # copy(entity_energies),
                        # 'fsp': 0., 'cst': 0.,
                        # 'type': protein_letters_3to1.get(residue.type),
                        # 'hbond': 0
                    }
            else:
                for temp_idx, design_idx in enumerate(range(idx * number_of_temperatures,
                                                            (idx+1) * number_of_temperatures)):
                    per_residue_data[design_ids[design_idx]] = design_dock_params

    # Todo get the keys right here
    # all_pose_divergence_df = pd.DataFrame()
    # all_pose_divergence_df = pd.concat(all_pose_divergence, keys=[('sequence', 'pose')], axis=1)
    interface_metrics_df = pd.DataFrame(interface_metrics).T

    # Initialize the main scoring DataFrame
    # scores_df = pd.DataFrame(pose_transformations).T
    scores_df = pd.concat([pd.DataFrame(pose_transformations).T, interface_metrics_df], axis=1)

    # Collect sequence metrics on every designed Pose
    if proteinmpnn_used:
        # Construct per_residue_df
        per_residue_df = pd.concat({design_id: pd.DataFrame(data, index=residue_numbers)
                                    for design_id, data in per_residue_data.items()}).unstack().swaplevel(0, 1, axis=1)
        if job.design.sequences:
            sequences = numeric_to_sequence(generated_sequences)
            # Format the sequences from design with shape (size, number_of_temperatures, pose_length)
            # to (size * number_of_temperatures, pose_length)
            sequences = sequences.reshape(-1, pose_length)
            per_residue_sequence_df = pd.DataFrame(sequences, index=design_ids,
                                                   columns=pd.MultiIndex.from_product([residue_numbers, ['type']]))
            per_residue_sequence_df.loc[putils.pose_source, :] = list(pose.sequence)
            # per_residue_sequence_df.append(pd.DataFrame(list(pose.sequence), columns=[putils.pose_source]).T)
            pose_sequences = dict(zip(design_ids, [''.join(sequence) for sequence in sequences.tolist()]))
            # Todo This is pretty much already done!
            #  pose_alignment = MultipleSequenceAlignment.from_array(sequences)
            # Todo make this capability
            #  pose_sequences = dict(zip(design_ids, pose_alignment.tolist()]))
            pose_alignment = MultipleSequenceAlignment.from_dictionary(pose_sequences)
            # Perform a frequency extraction for each background profile
            background_frequencies = {profile: pose_alignment.get_probabilities_from_profile(background)
                                      for profile, background in profile_background.items()}

            interface_observed_from_fragment_profile = np.array(fragment_profile_frequencies)
            background_frequencies.update({'fragment': interface_observed_from_fragment_profile})

            # Get profile mean observed
            # Todo
            #  Ensure that the interface residues are selected, not only by those that are 0 as interface can be 0!
            #  This could be transitioned to during design to ease the selection of thes
            interface_observed_from_fragment_profile[interface_observed_from_fragment_profile == 0] = np.nan
            # Todo RuntimeWarning: Mean of empty slice
            scores_df['observed_fragment_interface_mean'] = np.nanmean(interface_observed_from_fragment_profile, axis=1)
            scores_df['observed_evolution_mean'] = background_frequencies['evolution'].mean(axis=1)
            if collapse_profile.size:  # Not equal to zero
                scores_df['collapse_violation_design_residues'] = collapse_violation

            per_residue_background_frequencies = \
                pd.concat([pd.DataFrame(background, index=design_ids,
                                        columns=pd.MultiIndex.from_product([residue_numbers, [f'observed_{profile}']]))
                           for profile, background in background_frequencies.items()], axis=1)

            # Can't use below as each pose is different
            # index_residues = list(pose.interface_design_residue_numbers)
            # residue_df = pd.merge(residue_df.loc[:, idx_slice[index_residues, :]],
            #                       per_residue_df.loc[:, idx_slice[index_residues, :]],
            #                       left_index=True, right_index=True)

            # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
            # residue_info = process_residue_info(residue_info)  # Only useful in Rosetta
            # residue_info = incorporate_mutation_info(residue_info, all_mutations)
            # residue_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()

            # Calculate hydrophobic collapse for each design
            # Separate sequences by entity
            all_sequences_split = []
            for entity in pose.entities:
                entity_slice = slice(entity.n_terminal_residue.index, 1 + entity.c_terminal_residue.index)
                all_sequences_split.append(sequences[:, entity_slice].tolist())

            all_sequences_by_entity = list(zip(*all_sequences_split))
            # Todo, should the reference pose be used? -> + [entity.sequence for entity in pose.entities]
            # Include the pose as the pose_source in the measured designs
            # contact_order_per_res_z, reference_collapse, collapse_profile = pose.get_folding_metrics()
            folding_and_collapse = calculate_collapse_metrics(all_sequences_by_entity,
                                                              contact_order_per_res_z, reference_collapse, collapse_profile)
            per_residue_collapse_df = pd.concat({design_id: pd.DataFrame(data, index=residue_numbers)
                                                 for design_id, data in zip(design_ids, folding_and_collapse)},
                                                ).unstack().swaplevel(0, 1, axis=1)
            # Calculate mutational content
            all_mutations = \
                generate_mutations_from_reference(pose.sequence, pose_sequences, zero_index=True, return_to=True)
            all_mutations.pop('reference', None)  # Throw the reference away for now
            # s = pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
            scores_df['number_of_mutations'] = \
                pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
            scores_df['percent_mutations'] = \
                scores_df['number_of_mutations'] / scores_df['pose_length']

            idx = 1
            for idx, entity in enumerate(pose.entities, idx):
                c_terminal_residue_index_in_pose = entity.c_terminal_residue.index
                scores_df[f'entity_{idx}_number_of_mutations'] = \
                    pd.Series({design: len([1 for mutation_idx in mutations
                                            if mutation_idx < c_terminal_residue_index_in_pose])
                               for design, mutations in all_mutations.items()})
                scores_df[f'entity_{idx}_percent_mutations'] = \
                    scores_df[f'entity_{idx}_number_of_mutations'] \
                    / scores_df[f'entity_{idx}_number_of_residues']
            per_residue_df = per_residue_df.join([per_residue_sequence_df, per_residue_background_frequencies,
                                                  per_residue_collapse_df])
        # else:
        #     per_residue_sequence_df = per_residue_background_frequencies = per_residue_collapse_df = pd.DataFrame()

        if job.design.structures:
            scores_df['interface_local_density'] = pd.Series(interface_local_density)
            # Make buried surface area (bsa) columns, and residue classification
            per_residue_df = calculate_residue_surface_area(per_residue_df)  # .loc[:, idx_slice[index_residues, :]])

        # Calculate new metrics from combinations of other metrics
        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        summed_scores_df = sum_per_residue_metrics(per_residue_df)  # .loc[:, idx_slice[index_residues, :]])
        scores_df = scores_df.join(summed_scores_df)

        # scores_df['interface_area_polar'] = per_residue_df.loc[:, idx_slice[:, 'bsa_polar']].sum(axis=1)
        # scores_df['interface_area_hydrophobic'] = per_residue_df.loc[:, idx_slice[:, 'bsa_hydrophobic']].sum(axis=1)
        # scores_df['interface_area_total'] = \
        #     residue_df.loc[not_pose_source_indices, idx_slice[index_residues, 'bsa_total']].sum(axis=1)
        if job.design.structures:
            scores_df['interface_area_total'] = bsa_assembly_df = \
                scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']
            # Find the proportion of the residue surface area that is solvent accessible versus buried in the interface
            scores_df['interface_area_to_residue_surface_ratio'] = \
                (bsa_assembly_df / (bsa_assembly_df+scores_df['sasa_total_complex']))
            #      / scores_df['total_interface_residues']

            # Make scores_df errat_deviation that takes into account the pose_source sequence errat_deviation
            # This overwrites the sum_per_residue_metrics() value
            # Include in errat_deviation if errat score is < 2 std devs and isn't 0 to begin with
            source_errat_inclusion_boolean = \
                np.logical_and(pose_source_errat_s < errat_2_sigma, pose_source_errat_s != 0.)
            errat_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            # find where designs deviate above wild-type errat scores
            errat_sig_df = errat_df.sub(pose_source_errat_s, axis=1) > errat_1_sigma  # axis=1 Series is column oriented
            # then select only those residues which are expressly important by the inclusion boolean
            scores_df['errat_deviation'] = (errat_sig_df.loc[:, source_errat_inclusion_boolean] * 1).sum(axis=1)

        # Drop unused particular scores_df columns that have been summed
        scores_drop_columns = ['hydrophobic_collapse', 'sasa_relative_bound', 'sasa_relative_complex']
        scores_df = scores_df.drop(scores_drop_columns, errors='ignore', axis=1)
        scores_df = scores_df.rename(columns={'type': 'sequence'})
        #                                       'evolution': 'evolution_sequence_loss',
        #                                       'fragment': 'fragment_sequence_loss',
        #                                       'designed': 'designed_residues_total'})
        designed_df = per_residue_df.loc[:, idx_slice[:, 'designed_residues_total']].droplevel(1, axis=1)

        if job.dock.proteinmpnn_score:
            scores_df['proteinmpnn_v_design_cross_entropy_designed_mean'] = \
                (per_residue_df.loc[:, idx_slice[:, 'proteinmpnn_v_design_cross_entropy']].droplevel(1, axis=1)
                 * designed_df).mean(axis=1)
            # The per designed residue average proteinmpnn versus evolution cross entropy
            scores_df['proteinmpnn_v_evolution_cross_entropy_designed_mean'] = \
                (per_residue_df.loc[:, idx_slice[:, 'proteinmpnn_v_evolution_cross_entropy']].droplevel(1, axis=1)
                 * designed_df).mean(axis=1)
            # The per designed residue average proteinmpnn versus evolution cross entropy
            scores_df['proteinmpnn_v_fragment_cross_entropy_designed_mean'] = \
                (per_residue_df.loc[:, idx_slice[:, 'proteinmpnn_v_fragment_cross_entropy']].droplevel(1, axis=1)
                 * designed_df).mean(axis=1)
            # The per designed residue average proteinmpnn versus fragment cross entropy
            scores_df['proteinmpnn_v_evolution_cross_entropy_per_residue'] = \
                scores_df['proteinmpnn_v_evolution_cross_entropy'] / scores_df['pose_length']
            # The per residue average proteinmpnn versus evolution cross entropy in the pose

        if job.design.sequences:
            scores_df[putils.protocol] = 'proteinmpnn'
            scores_df['proteinmpnn_score_complex'] = \
                scores_df['interface_energy_complex'] / scores_df['pose_length']
            # The per residue average complexed proteinmpnn score in the pose
            scores_df['proteinmpnn_score_unbound'] = \
                scores_df['interface_energy_unbound'] / scores_df['pose_length']
            # The per residue average unbound proteinmpnn score in the pose
            scores_df['proteinmpnn_score_designed_complex'] = \
                (per_residue_df.loc[:, idx_slice[:, 'complex']].droplevel(1, axis=1) * designed_df).mean(axis=1)
            # The per designed residue average complexed proteinmpnn score in the pose
            scores_df['proteinmpnn_score_designed_unbound'] = \
                (per_residue_df.loc[:, idx_slice[:, 'unbound']].droplevel(1, axis=1) * designed_df).mean(axis=1)
            # The per designed residue average unbound proteinmpnn score in the pose
            scores_df['proteinmpnn_score_designed_delta'] = \
                scores_df['proteinmpnn_score_designed_complex'] - scores_df['proteinmpnn_score_designed_unbound']
            # The delta between the average complexed and unbound proteinmpnn designed residue score

        # # Drop unused particular per_residue_df columns that have been summed
        # per_residue_drop_columns = per_residue_energy_states + energy_metric_names + per_residue_sasa_states \
        #                            + collapse_metrics + residue_classification \
        #                            + ['errat_deviation', 'hydrophobic_collapse', 'contact_order'] \
        #                            + ['hbond', 'evolution', 'fragment', 'type'] + ['surface', 'interior']
        # # Slice each of these columns as the first level residue number needs to be accounted for in MultiIndex
        # per_residue_df = per_residue_df.drop(
        #     list(per_residue_df.loc[:, idx_slice[:, per_residue_drop_columns]].columns),
        #     errors='ignore', axis=1)
        per_residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)  # ascending=False
        # sum columns using list[0] + list[1] + list[n]
        # Todo We are not taking these measurements w/o Rosetta...
        # summation_pairs = \
        #     {'buns_unbound': list(filter(re.compile('buns_[0-9]+_unbound$').match, scores_columns)),  # Rosetta
        #      # 'interface_energy_bound':
        #      #     list(filter(re_compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
        #      # 'interface_energy_unbound':
        #      #     list(filter(re_compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
        #      # 'interface_solvation_energy_bound':
        #      #     list(filter(re_compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
        #      # 'interface_solvation_energy_unbound':
        #      #     list(filter(re_compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
        #      'interface_connectivity':
        #          list(filter(re.compile('interface_connectivity_[0-9]+').match, scores_columns)),  # Rosetta
        #      }
        # 'sasa_hydrophobic_bound':
        #     list(filter(re_compile('sasa_hydrophobic_[0-9]+_bound').match, scores_columns)),
        # 'sasa_polar_bound': list(filter(re_compile('sasa_polar_[0-9]+_bound').match, scores_columns)),
        # 'sasa_total_bound': list(filter(re_compile('sasa_total_[0-9]+_bound').match, scores_columns))}
        # scores_df = columns_to_new_column(scores_df, summation_pairs)
        scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
        scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
        if job.design.structures:
            scores_df['interface_composition_similarity'] = scores_df.apply(interface_composition_similarity, axis=1)
        scores_df.drop(clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')
    # else:  # Get metrics and output
    #     # Generate placeholder all_mutations which only contains "reference"
    #     # all_mutations = generate_mutations_from_reference(pose.sequence, pose_sequences, return_to=True)
    #     # per_residue_sequence_df = per_residue_background_frequencies = per_residue_collapse_df = pd.DataFrame()
    #     # all_pose_divergence_df = pd.DataFrame()
    #     # residue_df = pd.DataFrame()

    # Get the average thermophilicity for all entities
    scores_df['pose_thermophilicity'] = \
        scores_df.loc[:, [f'entity_{idx}_thermophile' for idx in range(1, pose.number_of_entities)]
                      ].sum(axis=1) / pose.number_of_entities

    scores_columns = scores_df.columns.to_list()
    logger.debug(f'Metrics present: {scores_columns}')

    # interface_metrics_s = pd.Series(interface_metrics_df)
    # Concatenate all design information after parsing data sources
    # interface_metrics_df = pd.concat([interface_metrics_df], keys=[('dock', 'pose')])
    # scores_df = pd.concat([scores_df], keys=[('dock', 'pose')], axis=1)
    # Todo incorporate full sequence ProteinMPNN summation into scores_df. Find meaning of probabilities
    # Todo incorporate residue_df summation into scores_df
    #  observed_*, solvation_energy, etc.
    scores_df = pd.concat([scores_df], keys=[('dock', 'pose')], axis=1)

    # CONSTRUCT: Create pose series and format index names
    pose_df = scores_df.swaplevel(0, 1, axis=1)
    # pose_df = pd.concat([scores_df, interface_metrics_df, all_pose_divergence_df]).swaplevel(0, 1)
    # Remove pose specific metrics from pose_df and sort
    pose_df.sort_index(level=2, axis=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_df.sort_index(level=1, axis=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)  # ascending=False
    pose_df.name = str(building_blocks)

    save = True
    if save:
        putils.make_path(job.all_scores)
        trajectory_metrics_csv = os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Trajectories.csv')
        pose_df.to_csv(trajectory_metrics_csv)
        logger.info(f'Wrote trajectory metrics to {trajectory_metrics_csv}')
        if job.design.sequences:
            residue_metrics_csv = os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Residues.csv')
            per_residue_df.to_csv(residue_metrics_csv)
            logger.info(f'Wrote per residue metrics to {residue_metrics_csv}')

    # Finalize docking run
    terminate()
    logger.info(f'Total {building_blocks} dock trajectory took {time.time() - frag_dock_time_start:.2f}s')

    return [protocols.PoseDirectory.from_file(file, entity_names=entity_names) for file in pose_paths]
    # ------------------ TERM ------------------------
