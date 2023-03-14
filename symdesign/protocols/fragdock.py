from __future__ import annotations

import logging
import sys
import math
import os
import time
from warnings import catch_warnings, simplefilter
from collections.abc import Iterable
from itertools import repeat, count
from math import prod

import numpy as np
import pandas as pd
import psutil
import scipy
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.neighbors._ball_tree import BinaryTree  # This typing implementation supports BallTree or KDTree
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from . import cluster
from .pose import load_evolutionary_profile, PoseJob
from symdesign import flags, metrics, resources, utils
from symdesign.resources import ml, job as symjob, sql
from symdesign.sequence import protein_letters_alph1
from symdesign.structure.base import Structure, Residue
from symdesign.structure.coords import transform_coordinate_sets, superposition3d
from symdesign.structure.fragment import GhostFragment
from symdesign.structure.fragment.db import TransformHasher
from symdesign.structure.fragment.metrics import rmsd_z_score, z_value_from_match_score
from symdesign.structure.fragment.visuals import write_fragment_pairs_as_accumulating_states
from symdesign.structure.model import Pose, Model, Models
from symdesign.structure.sequence import pssm_as_array
from symdesign.structure.utils import chain_id_generator
from symdesign.utils.SymEntry import SymEntry, get_rot_matrices, make_rotations_degenerate
from symdesign.utils.symmetry import identity_matrix
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
zero_offset = 1


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


def create_perturbation_transformations(sym_entry: SymEntry, number_of_rotations: int = 1, number_of_translations: int = 1,
                                        rotation_steps: Iterable[float] = None,
                                        translation_steps: Iterable[float] = None) -> dict[str, np.ndarray]:
    """From a specified SymEntry and sampling schedule, create perturbations to degrees of freedom for each available

    Args:
        sym_entry: The SymEntry whose degrees of freedom should be expanded
        number_of_rotations: The number of times to sample from the allowed rotation space. 1 means no perturbation
        number_of_translations: The number of times to sample from the allowed translation space. 1 means no perturbation
        rotation_steps: The step to sample rotations +/- the identified rotation in degrees.
            Expected type is an iterable of length comparable to the number of rotational degrees of freedom
        translation_steps: The step to sample translations +/- the identified translation in Angstroms
            Expected type is an iterable of length comparable to the number of translational degrees of freedom
    Returns:
        A mapping between the perturbation type and the corresponding transformation operation
    """
    # Get the perturbation parameters
    # Total number of perturbations desired using the total_dof possible in the symmetry and those requested
    target_dof = total_dof = sym_entry.total_dof
    rotational_dof = sym_entry.number_dof_rotation
    translational_dof = sym_entry.number_dof_translation
    n_dof_external = sym_entry.number_dof_external

    if number_of_rotations < 1:
        logger.warning(f"Can't create perturbation transformations with rotation_number of {number_of_rotations}. "
                       f"Setting to 1")
        number_of_rotations = 1
        # raise ValueError(f"Can't create perturbation transformations with rotation_number of {rotation_number}")
    if number_of_rotations == 1:
        target_dof -= rotational_dof
        rotational_dof = 0

    if number_of_translations < 1:
        logger.warning(f"Can't create perturbation transformations with translation_number of {number_of_translations}. "
                       f"Setting to 1")
        number_of_translations = 1

    if number_of_translations == 1:
        target_dof -= translational_dof
        translational_dof = 0
        # Set to 0 so there is no deviation from the current value and remove any provided values
        default_translation = 0
        translation_steps = None
    else:
        # Default translation range is 0.5 Angstroms
        default_translation = .5

    if translation_steps is None:
        translation_steps = tuple(repeat(default_translation, sym_entry.number_of_groups))

    if n_dof_external:
        if translation_steps is None:
            logger.warning(f'Using the default value of {default_translation} for external translation steps')
            ext_translation_steps = tuple(repeat(default_translation, n_dof_external))
        else:
            # Todo allow ext_translation_steps to be a parameter
            ext_translation_steps = translation_steps
    # # Make a vector of the perturbation number [1, 2, 2, 3, 3, 1] with 1 as constants on each end
    # dof_number_perturbations = [1] \
    #     + [rotation_number for dof in range(rotational_dof)] \
    # Make a vector of the perturbation number [2, 2, 3, 3]
    dof_number_perturbations = \
        [number_of_rotations for dof in range(rotational_dof)] \
        + [number_of_translations for dof in range(translational_dof)] \
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

    if rotation_steps is None:
        # Default rotation range is 1. degree
        default_rotation = 1.
        rotation_steps = tuple(repeat(default_rotation, sym_entry.number_of_groups))

    dof_idx = 0
    perturbation_mapping = {}
    for idx, group in enumerate(sym_entry.groups):
        group_idx = idx + 1
        if getattr(sym_entry, f'is_internal_rot{group_idx}'):
            rotation_step = rotation_steps[idx]  # * 2
            perturb_matrices = get_perturb_matrices(rotation_step, number=number_of_rotations)
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
            internal_translation_grid = translation_grid.copy()
            translation_perturb_vector = \
                np.linspace(-translation_steps[idx], translation_steps[idx], number_of_translations)
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
    external_translation_grid = translation_grid.copy()
    for idx, ext_idx in enumerate(range(n_dof_external)):
        translation_perturb_vector = \
            np.linspace(-ext_translation_steps[idx], ext_translation_steps[idx], number_of_translations)
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
                        f'perturbed dof used {dof_idx} != {target_dof}, the targeted dof to perturb resulting from '
                        f'{total_dof} total_dof = {rotational_dof} rotational_dof, {translational_dof} '
                        f'translational_dof, and {n_dof_external} external_translational_dof')

    return perturbation_mapping, target_dof


def make_contiguous_ghosts(ghost_frags_by_residue: list[list[GhostFragment]], residues: list[Residue],
                           distance: float = 16., initial_z_value: float = 1.) -> np.ndarray:
    #     -> tuple[np.ndarray, np.ndarray, np.ndarray]
    """Identify GhostFragment overlap originating from a group of residues related bby spatial proximity

    Args:
        ghost_frags_by_residue: The list of GhostFragments separated by Residue instances.
            GhostFragments should align with residues
        residues: Residue instance for which the ghost_frags_by_residue belong to
        # Todo residues currently requires a cb_coord to measure Residue-Residue distances. This could be some other feature
        #  of fragments such as a coordinate that identifies its spatial proximity to other fragments
        distance: The distance to measure neighbors between residues
        initial_z_value: The acceptable standard deviation z score for initial fragment overlap identification. \
            Smaller values lead to more stringent matching criteria
    Returns:
        The indices of the ghost fragments (when the ghost_frags_by_residue) is unstacked such as by itertool.chain
        chained
    """
    # Set up the output array with the number of residues by the length of the max number of ghost fragments
    max_ghost_frags = max([len(ghost_frags) for ghost_frags in ghost_frags_by_residue])
    number_or_surface_frags = len(residues)
    same_component_overlapping_ghost_frags = np.zeros((number_or_surface_frags, max_ghost_frags), dtype=int)
    ghost_frag_rmsds_by_residue = np.zeros_like(same_component_overlapping_ghost_frags, dtype=float)
    ghost_guide_coords_by_residue = np.zeros((number_or_surface_frags, max_ghost_frags, 3, 3))
    for idx, residue_ghosts in enumerate(ghost_frags_by_residue):
        number_of_ghosts = len(residue_ghosts)
        # Set any viable index to 1 to distinguish between padding with 0
        same_component_overlapping_ghost_frags[idx, :number_of_ghosts] = 1
        ghost_frag_rmsds_by_residue[idx, :number_of_ghosts] = [ghost.rmsd for ghost in residue_ghosts]
        ghost_guide_coords_by_residue[idx, :number_of_ghosts] = [ghost.guide_coords for ghost in residue_ghosts]
    # ghost_frag_rmsds_by_residue = np.array([[ghost.rmsd for ghost in residue_ghosts]
    #                                         for residue_ghosts in ghost_frags_by_residue], dtype=object)
    # ghost_guide_coords_by_residue = np.array([[ghost.guide_coords for ghost in residue_ghosts]
    #                                           for residue_ghosts in ghost_frags_by_residue], dtype=object)
    # surface_frag_residue_numbers = [residue.number for residue in residues]

    # Query for residue-residue distances for each surface fragment
    # surface_frag_cb_coords = np.concatenate([residue.cb_coords for residue in residues], axis=0)
    surface_frag_cb_coords = np.array([residue.cb_coords for residue in residues])
    model1_surface_cb_ball_tree = BallTree(surface_frag_cb_coords)
    residue_contact_query: np.ndarray = \
        model1_surface_cb_ball_tree.query_radius(surface_frag_cb_coords, distance)
    surface_frag_residue_indices = list(range(number_or_surface_frags))
    contacting_residue_idx_pairs: list[tuple[int, int]] = \
        [(surface_frag_residue_indices[idx1], surface_frag_residue_indices[idx2])
         for idx2, idx1_contacts in enumerate(residue_contact_query.tolist())
         for idx1 in idx1_contacts.tolist()]

    # Separate residue-residue contacts into a unique set of residue pairs
    asymmetric_contacting_residue_pairs, found_pairs = [], []
    for residue_idx1, residue_idx2 in contacting_residue_idx_pairs:
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
    # Todo 3, there are multiple indexing steps for residue_idx1/2 which only occur once if below code was used
    #  found_pairs = []
    #  for residue_idx2 in range(residue_contact_query.size):
    #      residue_ghost_frag_type2 = ghost_frag_type_by_residue[residue_idx2]
    #      residue_ghost_guide_coords2 = ghost_guide_coords_by_residue[residue_idx2]
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
    # Set up the input array types with the various information needed for each pairwise check
    ghost_frag_type_by_residue = [[ghost.frag_type for ghost in residue_ghosts]
                                  for residue_ghosts in ghost_frags_by_residue]
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
        ghost_coords_residue1 = ghost_guide_coords_by_residue[residue_idx1][residue_idx1_ghost_indices]
        ghost_coords_residue2 = ghost_guide_coords_by_residue[residue_idx2][residue_idx2_ghost_indices]
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
    return viable_same_component_overlapping_ghost_frags
    # # Prioritize search at those fragments which have same component, ghost fragment overlap
    # initial_ghost_frags = \
    #     [complete_ghost_frags1[idx] for idx in viable_same_component_overlapping_ghost_frags.tolist()]
    # init_ghost_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags])
    # init_ghost_rmsds = np.array([ghost_frag.rmsd for ghost_frag in initial_ghost_frags])
    # init_ghost_residue_indices = np.array([ghost_frag.index for ghost_frag in initial_ghost_frags])
    # return init_ghost_guide_coords, init_ghost_rmsds, init_ghost_residue_indices


def fragment_dock(models: Iterable[Structure], **kwargs) -> list[PoseJob] | list:
    # model1: Structure | AnyStr, model2: Structure | AnyStr,
    """Perform the fragment docking routine described in Laniado, Meador, & Yeates, PEDS. 2021

    Args:
        models: The Structures to be used in docking
    Returns:
        The resulting Poses satisfying docking criteria
    """
    frag_dock_time_start = time.time()

    # Todo reimplement this feature to write a log to the Project directory?
    # # Setup logger
    # if logger is None:
    #     log_file_path = os.path.join(project_dir, f'{building_blocks}_log.txt')
    # else:
    #     try:
    #         log_file_path = getattr(logger.handlers[0], 'baseFilename', None)
    #     except IndexError:  # No handler attached to this logger. Probably passing to a parent logger
    #         log_file_path = None
    #
    # if log_file_path:  # Start logging to a file in addition
    #     logger = start_log(name=building_blocks, handler=2, location=log_file_path, format_log=False, propagate=True)

    # Retrieve symjob.JobResources for all flags
    job = symjob.job_resources_factory.get()

    sym_entry: SymEntry = job.sym_entry
    """The SymmetryEntry object describing the material"""
    if sym_entry:
        protocol_name = 'nanohedra'
    else:
        protocol_name = 'fragment_docking'
    #
    # protocol = Protocol(name=protocol_name)

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
    # Todo 3 set below as parameters?
    measure_interface_during_dock = True
    low_quality_match_value = .2
    """The lower bounds on an acceptable match. Was upper bound of 2 using z-score"""
    clash_dist: float = 2.2
    """The distance to measure for clashing atoms"""
    cb_distance = 9.  # change to 8.?
    """The distance to measure for interface atoms"""
    # Testing if this is too strict when strict overlaps are used
    cluster_translations = not job.dock.contiguous_ghosts  # True
    translation_cluster_epsilon = 1
    # 1 works well at recapitulating the results without it while reducing number of checks
    # More stringent -> 0.75
    cluster_transforms = False  # True
    """Whether the entire transformation space should be clustered. This was found to be redundant with a translation
    clustering search only, and instead, decreases Pose solutions at the edge of oligomeric search slices  
    """
    transformation_cluster_epsilon = 1
    # 1 seems to work well at recapitulating the results without it
    # less stringent -> 0.75, removes about 20% found solutions
    # stringent -> 0.5, removes about %50 found solutions
    forward_reverse = False  # True
    # Todo 3 set above as parameters?
    high_quality_z_value = z_value_from_match_score(high_quality_match_value)
    low_quality_z_value = z_value_from_match_score(low_quality_match_value)

    if job.dock.perturb_dof_tx:
        if sym_entry.unit_cell:
            logger.critical(f"{create_perturbation_transformations.__name__} hasn't been tested for lattice symmetries")

    # Get score functions from input
    if job.dock.weight and isinstance(job.dock.weight, dict):
        # Todo actually use these during optimize_found_transformations_by_metrics()
        #  score_functions = metrics.pose.format_metric_functions(job.dock.weight.keys())
        default_weight_metric = None
    else:
        #  score_functions = {}
        if job.dock.proteinmpnn_score:
            weight_method = f'{putils.nanohedra}+{putils.proteinmpnn}'
        else:
            weight_method = putils.nanohedra

        default_weight_metric = resources.config.default_weight_parameter[weight_method]

    # Initialize incoming Structures
    models = list(models)
    """The Structure instances to be used in docking"""
    # Ensure models are oligomeric with make_oligomer()
    # Assumes model is oriented with major axis of symmetry along z
    entity_count = count(1)
    for idx, (model, symmetry) in enumerate(zip(models, sym_entry.groups)):
        for entity in model.entities:
            if entity.is_symmetric():  # oligomeric():
                pass
            else:
                # Remove any unstructured termini from the Entity to allow best secondary structure docking
                entity.delete_unstructured_termini()
                entity.make_oligomer(symmetry=symmetry)

            if next(entity_count) > 2:
                # Todo 2 remove able to take more than 2 Entity
                raise NotImplementedError(f"Can't dock 2 Model instances with > 2 total Entity instances")

        # Make, then save a new model based on the symmetric version of each Entity in the Model
        _model = Model.from_chains([chain for entity in model.entities
                                    for chain in entity.chains],  # log=logger
                                   entity_info=model.entity_info, name=model.name)
        _model.file_path = model.file_path
        _model.fragment_db = job.fragment_db
        # Ensure we pass the .metadata attribute to each entity in the full assembly
        # This is crucial for sql usage
        for _entity, entity in zip(_model.entities, model.entities):
            _entity.metadata = entity.metadata
        models[idx] = _model

    # Todo 2 figure out for single component
    model1: Model
    model2: Model
    model1, model2 = models
    logger.info(f'DOCKING {model1.name} TO {model2.name}')
    #            f'\nOligomer 1 Path: {model1.file_path}\nOligomer 2 Path: {model2.file_path}')

    # Set up output mechanism
    entry_string = f'NanohedraEntry{sym_entry.number}'
    building_blocks = '-'.join(model.name for model in models)
    if job.prefix:
        project = f'{job.prefix}{building_blocks}'
    else:
        project = building_blocks
    if job.suffix:
        project = f'{project}{job.suffix}'

    project = f'{entry_string}_{project}'
    project_dir = os.path.join(job.projects, project)
    putils.make_path(project_dir)
    if job.output_trajectory:
        # Create a Models instance to collect each model
        raise NotImplementedError('Make iterative saving more reliable. See output_pose()')
        trajectory_models = Models()

    # Set up the TransformHasher to assist in scoring/pose output
    radius1 = model1.distance_from_reference(measure='max')
    radius2 = model2.distance_from_reference(measure='max')
    # Assume the maximum distance the box could get is the radius of each plus the interface distance
    box_width = radius1 + radius2 + cb_distance
    model_transform_hasher = TransformHasher(box_width)

    # Set up Building Block1
    get_complete_surf_frags1_time_start = time.time()
    surf_frags1 = model1.get_fragment_residues(residues=model1.surface_residues, fragment_db=job.fragment_db)

    # Calculate the initial match type by finding the predominant surface type
    fragment_content1 = np.bincount([surf_frag.i_type for surf_frag in surf_frags1])
    initial_surf_type1 = np.argmax(fragment_content1)
    init_surf_frags1 = [surf_frag for surf_frag in surf_frags1 if surf_frag.i_type == initial_surf_type1]
    # For reverse/forward matching these two arrays must be made
    if forward_reverse:
        init_surf_guide_coords1 = np.array([surf_frag.guide_coords for surf_frag in init_surf_frags1])
        init_surf_residue_indices1 = np.array([surf_frag.index for surf_frag in init_surf_frags1])
    # surf_frag1_indices = [surf_frag.index for surf_frag in surf_frags1]
    idx = 1
    # logger.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords1.shape}')
    # logger.debug(f'Found surface residue numbers {idx} with shape {surf_residue_numbers1.shape}')
    # logger.debug(f'Found surface indices {idx} with shape {surf_i_indices1.shape}')
    logger.debug(f'Found {len(init_surf_frags1)} initial surface {idx} fragments with type: {initial_surf_type1}')
    # logger.debug('Found oligomer 2 fragment content: %s' % fragment_content2)
    # logger.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # logger.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # logger.debug('init_surf_residue_indices2: %s' % slice_variable_for_log(init_surf_residue_indices2))
    # logger.debug('init_surf_guide_coords1: %s' % slice_variable_for_log(init_surf_guide_coords1))
    # logger.debug('init_surf_residue_indices1: %s' % slice_variable_for_log(init_surf_residue_indices1))

    logger.info(f'Retrieved oligomer{idx}-{model1.name} surface fragments and guide coordinates took '
                f'{time.time() - get_complete_surf_frags1_time_start:8f}s')

    #################################
    # Set up Building Block2
    # Get Surface Fragments With Guide Coordinates Using COMPLETE Fragment Database
    get_complete_surf_frags2_time_start = time.time()
    surf_frags2 = \
        model2.get_fragment_residues(residues=model2.surface_residues, fragment_db=job.fragment_db)

    # Calculate the initial match type by finding the predominant surface type
    surf_guide_coords2 = np.array([surf_frag.guide_coords for surf_frag in surf_frags2])
    surf_residue_indices2 = np.array([surf_frag.index for surf_frag in surf_frags2])
    surf_i_indices2 = np.array([surf_frag.i_type for surf_frag in surf_frags2])
    fragment_content2 = np.bincount(surf_i_indices2)
    initial_surf_type2 = np.argmax(fragment_content2)
    init_surf_frag_indices2 = \
        [idx for idx, surf_frag in enumerate(surf_frags2) if surf_frag.i_type == initial_surf_type2]
    init_surf_guide_coords2 = surf_guide_coords2[init_surf_frag_indices2]
    init_surf_residue_indices2 = surf_residue_indices2[init_surf_frag_indices2]
    idx = 2
    logger.debug(f'Found surface guide coordinates {idx} with shape {surf_guide_coords2.shape}')
    logger.debug(f'Found surface residue numbers {idx} with shape {surf_residue_indices2.shape}')
    logger.debug(f'Found surface indices {idx} with shape {surf_i_indices2.shape}')
    logger.debug(
        f'Found {len(init_surf_residue_indices2)} initial surface {idx} fragments with type: {initial_surf_type2}')

    logger.info(f'Retrieved oligomer{idx}-{model2.name} surface fragments and guide coordinates took '
                f'{time.time() - get_complete_surf_frags2_time_start:8f}s')

    # logger.debug('init_surf_frag_indices2: %s' % slice_variable_for_log(init_surf_frag_indices2))
    # logger.debug('init_surf_guide_coords2: %s' % slice_variable_for_log(init_surf_guide_coords2))
    # logger.debug('init_surf_residue_indices2: %s' % slice_variable_for_log(init_surf_residue_indices2))

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
    ghost_residue_indices1 = np.array([ghost_frag.index for ghost_frag in complete_ghost_frags1])
    ghost_j_indices1 = np.array([ghost_frag.j_type for ghost_frag in complete_ghost_frags1])

    # Whether to use the overlap potential on the same component to filter ghost fragments
    if job.dock.contiguous_ghosts:
        # Prioritize search at those fragments which have same component, ghost fragment overlap
        contiguous_ghost_indices1 = make_contiguous_ghosts(ghost_frags_by_residue1, surf_frags1,
                                                           # distance=cb_distance,
                                                           initial_z_value=initial_z_value)
        initial_ghost_frags1 = [complete_ghost_frags1[idx] for idx in contiguous_ghost_indices1.tolist()]
        init_ghost_guide_coords1 = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags1])
        init_ghost_rmsds1 = np.array([ghost_frag.rmsd for ghost_frag in initial_ghost_frags1])
        init_ghost_residue_indices1 = np.array([ghost_frag.index for ghost_frag in initial_ghost_frags1])
        # init_ghost_guide_coords1, init_ghost_rmsds1, init_ghost_residue_indices1 = \
        #     make_contiguous_ghosts(ghost_frags_by_residue1, surf_frags)
    else:
        init_ghost_frag_indices1 = \
            [idx for idx, ghost_frag in enumerate(complete_ghost_frags1) if ghost_frag.j_type == initial_surf_type2]
        init_ghost_guide_coords1: np.ndarray = ghost_guide_coords1[init_ghost_frag_indices1]
        init_ghost_rmsds1: np.ndarray = ghost_rmsds1[init_ghost_frag_indices1]
        init_ghost_residue_indices1: np.ndarray = ghost_residue_indices1[init_ghost_frag_indices1]

    idx = 1
    logger.debug(f'Found ghost guide coordinates {idx} with shape: {ghost_guide_coords1.shape}')
    logger.debug(f'Found ghost residue numbers {idx} with shape: {ghost_residue_indices1.shape}')
    logger.debug(f'Found ghost indices {idx} with shape: {ghost_j_indices1.shape}')
    logger.debug(f'Found ghost rmsds {idx} with shape: {ghost_rmsds1.shape}')
    logger.debug(f'Found {len(init_ghost_guide_coords1)} initial ghost {idx} fragments with type:'
                 f' {initial_surf_type2}')

    logger.info(f'Retrieved oligomer{idx}-{model1.name} ghost fragments and guide coordinates '
                f'took {time.time() - get_complete_ghost_frags1_time_start:8f}s')
    #################################
    # Implemented for Todd to work on C1 instances
    if job.only_write_frag_info:
        # Whether to write fragment information to a directory (useful for fragment based docking w/o Nanohedra)
        guide_file_ghost = os.path.join(project_dir, f'{model1.name}_ghost_coords.txt')
        with open(guide_file_ghost, 'w') as f:
            for coord_group in ghost_guide_coords1.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_ghost_idx = os.path.join(project_dir, f'{model1.name}_ghost_coords_index.txt')
        with open(guide_file_ghost_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_j_indices1.tolist())))
        guide_file_ghost_res_num = os.path.join(project_dir, f'{model1.name}_ghost_coords_residue_number.txt')
        with open(guide_file_ghost_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, ghost_residue_indices1.tolist())))

        guide_file_surf = os.path.join(project_dir, f'{model2.name}_surf_coords.txt')
        with open(guide_file_surf, 'w') as f:
            for coord_group in surf_guide_coords2.tolist():
                f.write('%s\n' % ' '.join('%f,%f,%f' % tuple(coords) for coords in coord_group))
        guide_file_surf_idx = os.path.join(project_dir, f'{model2.name}_surf_coords_index.txt')
        with open(guide_file_surf_idx, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_i_indices2.tolist())))
        guide_file_surf_res_num = os.path.join(project_dir, f'{model2.name}_surf_coords_residue_number.txt')
        with open(guide_file_surf_res_num, 'w') as f:
            f.write('%s\n' % '\n'.join(map(str, surf_residue_indices2.tolist())))

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
                    os.path.join(project_dir, f'{model1.name}_{residue_number}_paired_frags_'
                                              f'{start_slice}:{number_of_fragments}:{visualize_number}.pdb'))

        raise RuntimeError(f'Suspending operation of {model1.name}/{model2.name} after write')

    ij_type_match_lookup_table = compute_ij_type_lookup(ghost_j_indices1, surf_i_indices2)
    # Axis 0 is ghost frag, 1 is surface frag
    # ij_matching_ghost1_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[0]))[ij_type_match_lookup_table]
    # ij_matching_surf2_indices = \
    #     (ij_type_match_lookup_table * np.arange(ij_type_match_lookup_table.shape[1])[:, None])[
    #         ij_type_match_lookup_table]
    # Tod0 apparently this works to grab the flattened indices where there is overlap
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

    # Whether to use the overlap potential on the same component to filter ghost fragments
    if forward_reverse:
        bb_cb_tree2 = BallTree(bb_cb_coords2)
        get_complete_ghost_frags2_time_start = time.time()
        ghost_frags_by_residue2 = \
            [frag.get_ghost_fragments(clash_tree=bb_cb_tree2) for frag in surf_frags2]

        complete_ghost_frags2: list[GhostFragment] = \
            [ghost for ghosts in ghost_frags_by_residue2 for ghost in ghosts]
        # complete_ghost_frags2 = []
        # for frag in surf_frags2:
        #     complete_ghost_frags2.extend(frag.get_ghost_fragments(clash_tree=bb_cb_tree2))

        if job.dock.contiguous_ghosts:
            # Prioritize search at those fragments which have same component, ghost fragment overlap
            contiguous_ghost_indices2 = make_contiguous_ghosts(ghost_frags_by_residue2, surf_frags2,
                                                               # distance=cb_distance,
                                                               initial_z_value=initial_z_value)
            initial_ghost_frags2 = [complete_ghost_frags2[idx] for idx in contiguous_ghost_indices2.tolist()]
            init_ghost_guide_coords2 = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags2])
            # init_ghost_rmsds2 = np.array([ghost_frag.rmsd for ghost_frag in initial_ghost_frags2])
            init_ghost_residue_indices2 = np.array([ghost_frag.index for ghost_frag in initial_ghost_frags2])
            # init_ghost_guide_coords1, init_ghost_rmsds1, init_ghost_residue_indices1 = \
            #     make_contiguous_ghosts(ghost_frags_by_residue1, surf_frags)
        else:
            # init_ghost_frag_indices2 = \
            #     [idx for idx, ghost_frag in enumerate(complete_ghost_frags2) if ghost_frag.j_type == initial_surf_type1]
            # init_ghost_guide_coords2: np.ndarray = ghost_guide_coords2[init_ghost_frag_indices2]
            # # init_ghost_rmsds2: np.ndarray = ghost_rmsds2[init_ghost_frag_indices2]
            # init_ghost_residue_indices2: np.ndarray = ghost_residue_indices2[init_ghost_frag_indices2]
            initial_ghost_frags2 = [ghost_frag for ghost_frag in complete_ghost_frags2 if
                                    ghost_frag.j_type == initial_surf_type1]
            init_ghost_guide_coords2 = np.array([ghost_frag.guide_coords for ghost_frag in initial_ghost_frags2])
            init_ghost_residue_indices2 = np.array([ghost_frag.index for ghost_frag in initial_ghost_frags2])

        idx = 2
        logger.debug(
            f'Found {len(init_ghost_guide_coords2)} initial ghost {idx} fragments with type {initial_surf_type1}')
        # logger.debug('init_ghost_guide_coords2: %s' % slice_variable_for_log(init_ghost_guide_coords2))
        # logger.debug('init_ghost_residue_indices2: %s' % slice_variable_for_log(init_ghost_residue_indices2))
        # ghost2_residue_array = np.repeat(init_ghost_residue_indices2, len(init_surf_residue_indices1))
        # surface1_residue_array = np.tile(init_surf_residue_indices1, len(init_ghost_residue_indices2))
        logger.info(f'Retrieved oligomer{idx}-{model2.name} ghost fragments and guide coordinates '
                    f'took {time.time() - get_complete_ghost_frags2_time_start:8f}s')

    # logger.debug(f'Found ghost guide coordinates {idx} with shape {ghost_guide_coords2.shape}')
    # logger.debug(f'Found ghost residue numbers {idx} with shape {ghost_residue_numbers2.shape}')
    # logger.debug(f'Found ghost indices {idx} with shape {ghost_j_indices2.shape}')
    # logger.debug(f'Found ghost rmsds {idx} with shape {ghost_rmsds2.shape}')
    # Prepare precomputed arrays for fast pair lookup
    # ghost1_residue_array = np.repeat(init_ghost_residue_indices1, len(init_surf_residue_indices2))
    # surface2_residue_array = np.tile(init_surf_residue_indices2, len(init_ghost_residue_indices1))

    logger.info('Obtaining rotation/degeneracy matrices\n')

    translation_perturb_steps = tuple(.5 for _ in range(sym_entry.number_dof_translation))
    """The number of angstroms to increment the translation degrees of freedom search for each model"""
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
        # Todo 3 make reliant on scipy...Rotation
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
        degen_len = len(degeneracy_matrices)
        number_of_degens.append(degen_len)
        # logger.debug(f'Rotation shape for component {idx}: {rot_degen_matrices.shape}')
        number_of_rotations.append(len(rot_degen_matrices) // degen_len)
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
    # else:

    # Set up internal translation parameters
    # zshift1/2 must be 2d array, thus the , 2:3].T instead of , 2].T
    # [:, None, 2] would also work
    if sym_entry.is_internal_tx1:  # Add the translation to Z (axis=1)
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
    logger.debug(f'zshift1={zshift1}, zshift2={zshift2}, max_z_value={initial_z_value:2f}')
    optimal_tx = resources.OptimalTx.from_dof(sym_entry.external_dof, zshift1=zshift1, zshift2=zshift2,
                                              max_z_value=initial_z_value)

    number_of_init_ghost = len(init_ghost_guide_coords1)
    number_of_init_surf = len(init_surf_guide_coords2)
    total_ghost_surf_combinations = number_of_init_ghost * number_of_init_surf
    # fragment_pairs = []
    full_rotation1, full_rotation2 = [], []
    rotation_matrices1, rotation_matrices2 = rotation_matrices
    rotation_matrices_len1, rotation_matrices_len2 = len(rotation_matrices1), len(rotation_matrices2)
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
    if forward_reverse:
        surf_frag1_guide_coords_rot_and_set = \
            transform_coordinate_sets(init_surf_guide_coords1[None, :, :, :],
                                      rotation=rotation_matrices1[:, None, :, :],
                                      rotation2=set_mat1[None, None, :, :])
        ghost_frags2_guide_coords_rot_and_set = \
            transform_coordinate_sets(init_ghost_guide_coords2[None, :, :, :],
                                      rotation=rotation_matrices2[:, None, :, :],
                                      rotation2=set_mat2[None, None, :, :])
        eulerint_surf_component1 = \
            euler_lookup.get_eulint_from_guides_as_array(surf_frag1_guide_coords_rot_and_set.reshape((-1, 3, 3)))
        eulerint_ghost_component2 = \
            euler_lookup.get_eulint_from_guides_as_array(ghost_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))

        stacked_surf_euler_int1 = eulerint_surf_component1.reshape((rotation_matrices_len1, -1, 3))
        stacked_ghost_euler_int2 = eulerint_ghost_component2.reshape((rotation_matrices_len2, -1, 3))
        # Improve indexing time by precomputing python objects
        stacked_ghost_euler_int2 = list(stacked_ghost_euler_int2)
        stacked_surf_euler_int1 = list(stacked_surf_euler_int1)
    # eulerint_surf_component2_1, eulerint_surf_component2_2, eulerint_surf_component2_3 = \
    #     euler_lookup.get_eulint_from_guides(surf_frags2_guide_coords_rot_and_set.reshape((-1, 3, 3)))

    # Reshape the reduced dimensional eulerint_components to again have the number_of_rotations length on axis 0,
    # the number of init_guide_coords on axis 1, and the 3 euler intergers on axis 2
    stacked_surf_euler_int2 = eulerint_surf_component2.reshape((rotation_matrices_len2, -1, 3))
    stacked_ghost_euler_int1 = eulerint_ghost_component1.reshape((rotation_matrices_len1, -1, 3))
    # Improve indexing time by precomputing python objects
    stacked_surf_euler_int2 = list(stacked_surf_euler_int2)
    stacked_ghost_euler_int1 = list(stacked_ghost_euler_int1)

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
    if job.dock.quick:  # job.development:
        rotations_to_perform1 = min(rotation_matrices1.shape[0], 13)
        rotations_to_perform2 = min(rotation_matrices2.shape[0], 12)
        logger.critical(f'Development: Only sampling {rotations_to_perform1} by {rotations_to_perform2} rotations')
    else:
        rotations_to_perform1 = rotation_matrices1.shape[0]
        rotations_to_perform2 = rotation_matrices2.shape[0]

    # Todo 2 multiprocessing
    def initial_euler_search():
        pass

    if job.multi_processing:
        raise NotImplementedError(f"Can't perform {fragment_dock.__name__} using --{flags.multi_processing} just yet")
        rotation_pairs = None
        results = utils.mp_map(initial_euler_search, rotation_pairs, processes=job.cores)
    else:
        pass
        # results = []
        # for rot_pair in rotation_pairs:
        #     results.append(initial_euler_search(rot_pair))

    # Todo 3 resolve which mechanisms to use. guide coords or eulerints
    #  Below uses eulerints which work just fine.
    #  Timings on these from improved protocols shows about similar times to euler_lookup and calculate_overlap
    #  even with vastly different scales of the arrays. This ignores the fact that calculate_overlap uses a
    #  number of indexing steps including making the ij_match array formation, indexing against the ghost and
    #  surface arrays, the rmsd_reference construction
    #  |
    #  Given the lookups sort of irrelevance to the scoring (given very poor alignment), I could remove that
    #  step if it interfered with differentiability
    #  |
    #  Majority of time is spent indexing the 6D euler overlap arrays which should be quite easy to speed up given
    #  understanding of different computational efficiencies at this check
    # Get rotated oligomer1 ghost fragment, oligomer2 surface fragment guide coodinate pairs in the same Euler space
    perturb_dof = job.dock.perturb_dof
    for idx1 in range(rotations_to_perform1):
        rot1_count = idx1%number_of_rotations1 + 1
        degen1_count = idx1//number_of_rotations1 + 1
        rot_mat1 = rotation_matrices1[idx1]
        rotation_ghost_euler_ints1 = stacked_ghost_euler_int1[idx1]
        if forward_reverse:
            rotation_surf_euler_ints1 = stacked_surf_euler_int1[idx1]
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
            if forward_reverse:
                euler_matched_ghost_indices_rev2, euler_matched_surf_indices_rev1 = \
                    euler_lookup.lookup_by_euler_integers_as_array(stacked_ghost_euler_int2[idx2],
                                                                   rotation_surf_euler_ints1)
            # Todo 3 resolve. eulerints

    # Todo 3 resolve. Below uses guide coords
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
    # Todo 3 resolve. guide coords
            logger.debug(f'\tEuler search took {time.time() - euler_start:8f}s for '
                         f'{total_ghost_surf_combinations} ghost/surf pairs')

            if forward_reverse:
                # Ensure pairs are similar between euler_matched_surf_indices2 and euler_matched_ghost_indices_rev2
                # by indexing the residue_numbers
                # forward_reverse_comparison_start = time.time()
                # logger.debug(f'Euler indices forward, index 0: {euler_matched_surf_indices2[:10]}')
                forward_surface_indices2 = init_surf_residue_indices2[euler_matched_surf_indices2]
                # logger.debug(f'Euler indices forward, index 1: {euler_matched_ghost_indices1[:10]}')
                forward_ghosts_indices1 = init_ghost_residue_indices1[euler_matched_ghost_indices1]
                # logger.debug(f'Euler indices reverse, index 0: {euler_matched_ghost_indices_rev2[:10]}')
                reverse_ghosts_indices2 = init_ghost_residue_indices2[euler_matched_ghost_indices_rev2]
                # logger.debug(f'Euler indices reverse, index 1: {euler_matched_surf_indices_rev1[:10]}')
                reverse_surface_indices1 = init_surf_residue_indices1[euler_matched_surf_indices_rev1]

                # Make an index indicating where the forward and reverse euler lookups have the same residue pairs
                # Important! This method only pulls out initial fragment matches that go both ways, i.e. component1
                # surface (type1) matches with component2 ghost (type1) and vice versa, so the expanded checks of
                # for instance the surface loop (i type 3,4,5) with ghost helical (i type 1) matches is completely
                # unnecessary during euler look up as this will never be included
                # Also, this assumes that the ghost fragment display is symmetric, i.e. 1 (i) 1 (j) 10 (K) has an
                # inverse transform at 1 (i) 1 (j) 230 (k) for instance

                prior = 0
                number_overlapping_pairs = len(euler_matched_ghost_indices1)
                possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)
                # Residue numbers are in order for forward_surface_indices2 and reverse_ghosts_indices2
                for residue_index in init_surf_residue_indices2:
                    # Where the residue number of component 2 is equal pull out the indices
                    forward_index = np.flatnonzero(forward_surface_indices2 == residue_index)
                    reverse_index = np.flatnonzero(reverse_ghosts_indices2 == residue_index)
                    # Next, use residue number indices to search for the same residue numbers in the extracted pairs
                    # The output array slice is only valid if the forward_index is the result of
                    # forward_surface_indices2 being in ascending order, which for check_lookup_table is True
                    current = prior + forward_index.shape[0]
                    possible_overlaps[prior:current] = \
                        np.in1d(forward_ghosts_indices1[forward_index], reverse_surface_indices1[reverse_index])
                    prior = current

            # # Use for residue number debugging
            # possible_overlaps = np.ones(number_overlapping_pairs, dtype=np.bool8)

            # forward_ghosts_indices1[possible_overlaps]
            # forward_surface_indices2[possible_overlaps]

            # indexing_possible_overlap_time = time.time() - indexing_possible_overlap_start

            # number_of_successful = possible_overlaps.sum()
            # logger.info(f'\tIndexing {number_overlapping_pairs * euler_matched_surf_indices2.shape[0]} '
            #             f'possible overlap pairs found only {number_of_successful} possible out of '
            #             f'{number_overlapping_pairs} (took {time.time() - forward_reverse_comparison_start:8f}s)')

            # passing_ghost_coords = ghost_guide_coords_rot_and_set1[possible_ghost_frag_indices]
            # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2[possible_overlaps]]

            # Get optimal shift parameters for initial (Ghost Fragment, Surface Fragment) guide coordinate pairs
            # # Todo these are from Guides
            # passing_ghost_coords = ghost_guide_coords_rot_and_set1[euler_matched_ghost_indices1]
            # passing_surf_coords = surf_guide_coords_rot_and_set2[euler_matched_surf_indices2]
            # # Todo these are from Guides
            # Todo debug With EulerInteger calculation
            if forward_reverse:
                # Take the boolean index of the indices
                possible_ghost_frag_indices = euler_matched_ghost_indices1[possible_overlaps]
                # possible_surf_frag_indices = euler_matched_surf_indices2[possible_overlaps]
                passing_ghost_coords = \
                    ghost_frag1_guide_coords_rot_and_set[idx1, possible_ghost_frag_indices]
                passing_surf_coords = \
                    surf_frags2_guide_coords_rot_and_set[idx2, euler_matched_surf_indices2[possible_overlaps]]
                reference_rmsds = init_ghost_rmsds1[possible_ghost_frag_indices]
            else:
                passing_ghost_coords = ghost_frag1_guide_coords_rot_and_set[idx1, euler_matched_ghost_indices1]
                # passing_ghost_coords = transform_coordinate_sets(init_ghost_guide_coords1[euler_matched_ghost_indices1],
                #                                                  rotation=rot_mat1, rotation2=set_mat1)
                passing_surf_coords = surf_frags2_guide_coords_rot_and_set[idx2, euler_matched_surf_indices2]
                # passing_surf_coords = transform_coordinate_sets(init_surf_guide_coords2[euler_matched_surf_indices2],
                #                                                 rotation=rot_mat2, rotation2=set_mat2)
                reference_rmsds = init_ghost_rmsds1[euler_matched_ghost_indices1]
            # Todo debug With EulerInteger calculation

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
            elif cluster_translations:
                cluster_time_start = time.time()
                translation_cluster = \
                    DBSCAN(eps=translation_cluster_epsilon, min_samples=min_matched).fit(transform_passing_shifts)
                if perturb_dof:  # Later will be sampled more finely, so
                    # Get the core indices, i.e. the most dense translation regions only
                    transform_passing_shift_indexer = translation_cluster.core_sample_indices_
                else:  # Get any transform which isn't an outlier
                    transform_passing_shift_indexer = translation_cluster.labels_ != outlier
                transform_passing_shifts = transform_passing_shifts[transform_passing_shift_indexer]
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
                positive_indices = \
                    np.flatnonzero(np.all(transform_passing_shifts[:, :sym_entry.number_dof_external] >= 0, axis=1))
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
                # Store transformation parameters, indexing only those that are positive in the case of lattice syms
                full_int_tx1.extend(transform_passing_shifts[positive_indices,
                                                             sym_entry.number_dof_external].tolist())

            if full_int_tx2 is not None:
                # Store transformation parameters, indexing only those that are positive in the case of lattice syms
                full_int_tx2.extend(transform_passing_shifts[positive_indices,
                                                             sym_entry.number_dof_external + 1].tolist())

            full_rotation1.append(np.tile(rot_mat1, (number_passing_shifts, 1, 1)))
            full_rotation2.append(np.tile(rot_mat2, (number_passing_shifts, 1, 1)))

            logger.debug(f'\tOptimal shift search took {optimal_shifts_time:8f}s for '
                         f'{euler_matched_ghost_indices1.shape[0]} guide coordinate pairs')
            logger.info(f'\t{number_passing_shifts if number_passing_shifts else "No"} initial interface '
                        f'match{"es" if number_passing_shifts != 1 else ""} found '
                        f'(took {time.time() - euler_start:8f}s)')

            # # Tod0 debug
            # # tx_param_list = []
            # init_pass_ghost_numbers = init_ghost_residue_indices1[possible_ghost_frag_indices]
            # init_pass_surf_numbers = init_surf_residue_indices2[possible_surf_frag_indices]
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
            # # Tod0 debug

    # -----------------------------------------------------------------------------------------------------------------
    # Below creates vectors for cluster transformations
    # Then asu clash testing, scoring, and symmetric clash testing are performed
    # -----------------------------------------------------------------------------------------------------------------
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
        # ------------------ TERMINATE DOCKING ------------------------
    else:
        logger.info(f'Initial optimal translation search found {starting_transforms} total transforms '
                    f'in {time.time() - init_translation_time_start:8f}s')

    if sym_entry.is_internal_tx1:
        stacked_internal_tx_vectors1 = np.zeros((starting_transforms, 3), dtype=float)
        # Add the translation to Z (axis=1)
        stacked_internal_tx_vectors1[:, -1] = full_int_tx1
        full_int_tx1 = stacked_internal_tx_vectors1
        # del stacked_internal_tx_vectors1

    if sym_entry.is_internal_tx2:
        stacked_internal_tx_vectors2 = np.zeros((starting_transforms, 3), dtype=float)
        # Add the translation to Z (axis=1)
        stacked_internal_tx_vectors2[:, -1] = full_int_tx2
        full_int_tx2 = stacked_internal_tx_vectors2
        # del stacked_internal_tx_vectors2

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

    # Define functions for making active transformation arrays and removing indices from them
    def create_transformation_group() -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None]]:
        """Create the transformation mapping for each transformation in the current docking trajectory

        Returns:
            Every stacked transformation operation for the two separate models being docked in two separate dictionaries
        """
        return (
            dict(rotation=full_rotation1, translation=None if full_int_tx1 is None else full_int_tx1[:, None, :],
                 rotation2=set_mat1, translation2=None if full_ext_tx1 is None else full_ext_tx1[:, None, :]),
            dict(rotation=full_rotation2, translation=None if full_int_tx2 is None else full_int_tx2[:, None, :],
                 rotation2=set_mat2, translation2=None if full_ext_tx2 is None else full_ext_tx2[:, None, :])
        )

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

    def filter_transforms_by_indices(passing_indices: np.ndarray | list[int]):
        """Responsible for updating docking transformation parameters for transform operations. Will set the
        transformation in the order of the passing_indices

        These include:
            full_rotation1, full_rotation2, full_int_tx1, full_int_tx2, full_optimal_ext_dof_shifts, full_ext_tx1, and
            full_ext_tx2

        Args:
            passing_indices: The indices which should be kept
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
    if cluster_transforms:
        clustering_start = time.time()
        # Todo 3
        #  Can I use cluster.cluster_transformation_pairs distance graph to provide feedback on other aspects of the
        #  dock? Seems that I could use the distances to expedite clashing checks, especially for more time consuming
        #  expansion checks such as the full material...
        # Must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
        transform_neighbor_tree, transform_cluster = \
            cluster.cluster_transformation_pairs(*create_transformation_group(),
                                                 distance=transformation_cluster_epsilon,
                                                 minimum_members=min_matched
                                                 )
        # cluster_representative_indices, cluster_labels = \
        #     find_cluster_representatives(transform_neighbor_tree, transform_cluster)
        del transform_neighbor_tree
        # representative_labels = cluster_labels[cluster_representative_indices]
        # Todo 3
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
        # ------------------ TERMINATE DOCKING ------------------------
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
        # Todo 3 for performing broadcasting of this operation
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

    logger.info(f'Testing found transforms for ASU clashes')
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
    # ------------------ TERMINATE DOCKING ------------------------
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

    # Todo 3 if using individual Poses
    #  def clone_pose(idx: int) -> Pose:
    #      # Create a copy of the base Pose
    #      new_pose = copy.copy(pose)
    #      if sym_entry.unit_cell:
    #          # Set the next unit cell dimensions
    #          new_pose.uc_dimensions = full_uc_dimensions[idx]
    #      # Update the Pose coords
    #      new_pose.coords = np.concatenate(new_coords)
    #      return new_pose

    # Use below instead of this until can Todo 3 vectorize asu_interface_residue_processing
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

        # int_frags_time_start = time.time()
        model2_query = model1_cb_balltree.query_radius(inverse_transformed_model2_tiled_cb_coords[idx], cb_distance)
        # model1_cb_balltree_time = time.time() - int_frags_time_start

        contacting_residue_idx_pairs = [(model1_coords_indexed_residues[model1_cb_indices[model1_idx]].index,
                                         model2_coords_indexed_residues[model2_cb_indices[model2_idx]].index)
                                        for model2_idx, model1_contacts in enumerate(model2_query.tolist())
                                        for model1_idx in model1_contacts.tolist()]
        try:
            interface_residue_indices1, interface_residue_indices2 = \
                map(list, map(set, zip(*contacting_residue_idx_pairs)))
        except ValueError:  # Interface contains no residues, so not enough values to unpack
            logger.warning('Interface contains no residues')
            continue

        # Find the indices where the fragment residue numbers are found the interface residue numbers
        # is_in_index_start = time.time()
        # Since *_residue_numbers1/2 are the same index as the complete fragment arrays, these interface indices are the
        # same index as the complete guide coords and rmsds as well
        # Both residue numbers are one-indexed vv
        # Todo 3 make ghost_residue_indices1 unique -> unique_ghost_residue_numbers1
        #  index selected numbers against per_residue_ghost_indices 2d (number surface frag residues,
        ghost_indices_in_interface1 = \
            np.flatnonzero(np.isin(ghost_residue_indices1, interface_residue_indices1))
        surf_indices_in_interface2 = \
            np.flatnonzero(np.isin(surf_residue_indices2, interface_residue_indices2, assume_unique=True))

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

        number_int_surf = surf_indices_in_interface2.shape[0]
        number_int_ghost = ghost_indices_in_interface1.shape[0]
        # maximum_number_of_pairs = number_int_ghost*number_int_surf
        # if maximum_number_of_pairs < euler_lookup_size_threshold:
        # Tod0 at one point, there might have been a memory leak by Pose objects sharing memory with persistent objects
        #  that prevent garbage collection and stay attached to the run
        # Skipping EulerLookup as it has issues with precision
        index_ij_pairs_start_time = time.time()
        ghost_indices_repeated = np.repeat(ghost_indices_in_interface1, number_int_surf)
        surf_indices_tiled = np.tile(surf_indices_in_interface2, number_int_ghost)
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
            # I think it is the accuracy of binned euler_angle lookup
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
            #     surf_frags2[surf_indices_in_interface2][passing_surf_indices[sorted_overlap_indices]]
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
    # ------------------ TERMINATE DOCKING ------------------------
    logger.info(f'Found {number_viable_pose_interfaces} poses with viable interfaces')
    # Generate the Pose for output handling
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
                              log=None,
                              name='asu', entity_info=entity_info,  # entity_names=entity_names,  # log=logger,
                              sym_entry=sym_entry, surrounding_uc=job.output_surrounding_uc,
                              fragment_db=job.fragment_db, ignore_clashes=True, rename_chains=True)

    # Ensure we pass the .metadata attribute to each entity in the full assembly
    # This is crucial for sql usage
    entity_idx = count(0)
    for model in models:
        for entity in model.entities:
            pose.entities[next(entity_idx)].metadata = entity.metadata

    # Define functions for updating the single Pose instance coordinates
    # def create_specific_transformation(idx: int) -> tuple[dict[str, np.ndarray], ...]:
    #     """Take the current transformation index and create a mapping of the transformation operations
    #
    #     Args:
    #         idx: The index of the transformation to select
    #     Returns:
    #         A tuple containing the transformation operations for each model
    #     """
    #     if sym_entry.is_internal_tx1:
    #         internal_tx_param1 = full_int_tx1[idx]
    #     else:
    #         internal_tx_param1 = None
    #
    #     if sym_entry.is_internal_tx2:
    #         internal_tx_param2 = full_int_tx2[idx]
    #     else:
    #         internal_tx_param2 = None
    #
    #     if sym_entry.unit_cell:
    #         external_tx1 = full_ext_tx1[idx]
    #         external_tx2 = full_ext_tx2[idx]
    #     else:
    #         external_tx1 = external_tx2 = None
    #
    #     specific_transformation1 = dict(rotation=full_rotation1[idx], translation=internal_tx_param1,
    #                                     rotation2=set_mat1, translation2=external_tx1)
    #     specific_transformation2 = dict(rotation=full_rotation2[idx], translation=internal_tx_param2,
    #                                     rotation2=set_mat2, translation2=external_tx2)
    #     return specific_transformation1, specific_transformation2

    def update_pose_coords(idx: int):
        """Take the current transformation index and update the reference coordinates with the provided transforms

        Args:
            idx: The index of the transformation to select
        """
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
            # logger.debug(f'entity_start_coords{entity_idx + 1}: {entity_start_coords[entity_idx][:2]}')
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
    # # filter_transforms_by_indices(passing_transforms_indices)
    passing_transforms_indices = sufficiently_dense_indices[asu_is_viable_indices[interface_is_viable]]

    if job.design.ignore_symmetric_clashes:
        logger.warning(f'Not checking for symmetric clashes per requested flag --{flags.ignore_symmetric_clashes}')
        passing_symmetric_clash_indices_perturb = slice(None)
    else:
        logger.info('Checking solutions for symmetric clashes')
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
        # ------------------ TERMINATE DOCKING ------------------------
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
    filter_transforms_by_indices(passing_transforms_indices)
    number_of_transforms = passing_transforms_indices.shape[0]

    # # all_passing_ghost_indices = [all_passing_ghost_indices[idx] for idx in passing_symmetric_clash_indices.tolist()]
    # # all_passing_surf_indices = [all_passing_surf_indices[idx] for idx in passing_symmetric_clash_indices.tolist()]
    # # all_passing_z_scores = [all_passing_z_scores[idx] for idx in passing_symmetric_clash_indices.tolist()]

    if sym_entry.unit_cell:
        # Calculate the vectorized uc_dimensions
        full_uc_dimensions = sym_entry.get_uc_dimensions(full_optimal_ext_dof_shifts)

    def perturb_transformations() -> tuple[np.ndarray, list[int], int]:
        """From existing transformation parameters, sample parameters within a range of spatial perturbation

        Returns:
            A tuple consisting of the elements (
            transformation hash - Integer mapping the possible 3D space for docking to each perturbed transformation,
            size of each perturbation cluster - Number of perturbed transformations possible from starting transform,
            degrees of freedom sampled - How many degrees of freedom were perturbed
            )
        """
        logger.info(f'Perturbing transformations')
        perturb_rotation1, perturb_rotation2, perturb_int_tx1, perturb_int_tx2, perturb_optimal_ext_dof_shifts = \
            [], [], [], [], []

        # Define a function to stack the transforms
        def stack_viable_transforms(passing_indices: np.ndarray | list[int]):
            """From indices with viable transformations, stack the corresponding transformations into full
            perturbation transformations

            Args:
                passing_indices: The indices that should be selected from the full transformation sets
            """
            # nonlocal perturb_rotation1, perturb_rotation2, perturb_int_tx1, perturb_int_tx2
            logger.debug(f'Perturb expansion found {len(passing_indices)} passing_perturbations')
            perturb_rotation1.append(full_rotation1[passing_indices])
            perturb_rotation2.append(full_rotation2[passing_indices])
            if sym_entry.is_internal_tx1:
                perturb_int_tx1.extend(full_int_tx1[passing_indices, -1])
            if sym_entry.is_internal_tx2:
                perturb_int_tx2.extend(full_int_tx2[passing_indices, -1])

            if sym_entry.unit_cell:
                nonlocal full_optimal_ext_dof_shifts  # , full_ext_tx1, full_ext_tx2
                perturb_optimal_ext_dof_shifts.append(full_optimal_ext_dof_shifts[passing_indices])
                # full_uc_dimensions = full_uc_dimensions[passing_indices]
                # full_ext_tx1 = full_ext_tx1[passing_indices]
                # full_ext_tx2 = full_ext_tx2[passing_indices]

        # Expand successful poses from coarse search of transformational space to randomly perturbed offset
        # By perturbing the transformation a random small amount, we generate transformational diversity from
        # the already identified solutions.
        perturbations, n_perturbed_dof = \
            create_perturbation_transformations(sym_entry, number_of_rotations=job.dock.perturb_dof_steps_rot,
                                                number_of_translations=job.dock.perturb_dof_steps_tx,
                                                rotation_steps=rotation_steps,
                                                translation_steps=translation_perturb_steps)
        # Extract perturbation parameters and set the original transformation parameters to a new variable
        # if sym_entry.is_internal_rot1:  # Todo 2
        nonlocal number_of_transforms, full_rotation1, full_rotation2
        nonlocal number_perturbations_applied
        original_rotation1 = full_rotation1
        rotation_perturbations1 = perturbations['rotation1']
        # Compute the length of each perturbation to separate into unique perturbation spaces
        number_perturbations_applied, *_ = rotation_perturbations1.shape
        # logger.debug(f'rotation_perturbations1.shape: {rotation_perturbations1.shape}')
        # logger.debug(f'rotation_perturbations1[:5]: {rotation_perturbations1[:5]}')

        # if sym_entry.is_internal_rot2:  # Todo 2
        original_rotation2 = full_rotation2
        rotation_perturbations2 = perturbations['rotation2']
        # logger.debug(f'rotation_perturbations2.shape: {rotation_perturbations2.shape}')
        # logger.debug(f'rotation_perturbations2[:5]: {rotation_perturbations2[:5]}')
        # blank_parameter = list(repeat([None, None, None], number_of_transforms))
        if sym_entry.is_internal_tx1:
            nonlocal full_int_tx1
            original_int_tx1 = full_int_tx1
            translation_perturbations1 = perturbations['translation1']
            # logger.debug(f'translation_perturbations1.shape: {translation_perturbations1.shape}')
            # logger.debug(f'translation_perturbations1[:5]: {translation_perturbations1[:5]}')
        # else:
        #     translation_perturbations1 = blank_parameter

        if sym_entry.is_internal_tx2:
            nonlocal full_int_tx2
            original_int_tx2 = full_int_tx2
            translation_perturbations2 = perturbations['translation2']
            # logger.debug(f'translation_perturbations2.shape: {translation_perturbations2.shape}')
            # logger.debug(f'translation_perturbations2[:5]: {translation_perturbations2[:5]}')
        # else:
        #     translation_perturbations2 = blank_parameter

        if sym_entry.unit_cell:
            nonlocal full_optimal_ext_dof_shifts
            nonlocal full_ext_tx1, full_ext_tx2
            ext_dof_perturbations = perturbations['external_translations']
            original_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts
            # original_ext_tx1 = full_ext_tx1
            # original_ext_tx2 = full_ext_tx2
        else:
            full_ext_tx1 = full_ext_tx2 = full_ext_tx_sum = None

        # Apply the perturbation to each existing transformation
        logger.info(f'Perturbing each transform {number_perturbations_applied} times')
        for idx in range(number_of_transforms):
            logger.info(f'Perturbing transform {idx + 1}')
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
            asu_clash_counts = np.ones(number_perturbations_applied)
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
            else:
                passing_symmetric_clash_indices_perturb = slice(None)
            # Index the passing ASU indices with the passing symmetric indices and keep all viable transforms
            # Stack the viable perturbed transforms
            stack_viable_transforms(passing_perturbations[passing_symmetric_clash_indices_perturb])

        # Concatenate the stacked perturbations
        full_rotation1 = np.concatenate(perturb_rotation1, axis=0)
        full_rotation2 = np.concatenate(perturb_rotation2, axis=0)
        number_of_transforms = full_rotation1.shape[0]
        logger.info(f'After perturbation, found {number_of_transforms} viable solutions')
        if sym_entry.is_internal_tx1:
            full_int_tx1 = np.zeros((number_of_transforms, 3), dtype=float)
            # Add the translation to Z (axis=1)
            full_int_tx1[:, -1] = perturb_int_tx1
            # full_int_tx1 = stacked_internal_tx_vectors1

        if sym_entry.is_internal_tx2:
            full_int_tx2 = np.zeros((number_of_transforms, 3), dtype=float)
            # Add the translation to Z (axis=1)
            full_int_tx2[:, -1] = perturb_int_tx2
            # full_int_tx2 = stacked_internal_tx_vectors2

        if sym_entry.unit_cell:
            # optimal_ext_dof_shifts[:, :, None] <- None expands the axis to make multiplication accurate
            full_optimal_ext_dof_shifts = np.concatenate(perturb_optimal_ext_dof_shifts, axis=0)
            unsqueezed_optimal_ext_dof_shifts = full_optimal_ext_dof_shifts[:, :, None]
            full_ext_tx1 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof1, axis=-2)
            full_ext_tx2 = np.sum(unsqueezed_optimal_ext_dof_shifts * sym_entry.external_dof2, axis=-2)

        transform_hashes = create_transformation_hash()
        logger.debug(f'Found the TransformHasher.translation_bin_width={model_transform_hasher.translation_bin_width}, '
                     f'.rotation_bin_width={model_transform_hasher.rotation_bin_width}\n'
                     f'Current range of sampled translations={sum(translation_perturb_steps)}, '
                     f'rotations={sum(rotation_steps)}')
        # print(model_transform_hasher.translation_bin_width > sum(translation_perturb_steps))
        # print(model_transform_hasher.rotation_bin_width > sum(rotation_steps))
        if model_transform_hasher.translation_bin_width > sum(translation_perturb_steps) or \
                model_transform_hasher.rotation_bin_width > sum(rotation_steps):
            # The translation/rotation is smaller than bins, so further exploration only possible without minimization
            # Get the shape of the passing perturbations
            perturbation_shape = [len(perturb) for perturb in perturb_rotation1]
            # sorted_unique_transform_hashes = transform_hashes
        else:
            # Minimize perturbation space by unique transform hashes
            # Using the current transforms, create a hash to uniquely label them and apply to the indices
            sorted_unique_transform_hashes, unique_indices = np.unique(transform_hashes, return_index=True)
            # Create array to mark which are unique
            unique_transform_hashes = np.zeros_like(transform_hashes)
            unique_transform_hashes[unique_indices] = 1
            # Filter by unique_indices, sorting the indices to maintain the order of the transforms
            # filter_transforms_by_indices(unique_indices)
            unique_indices.sort()
            filter_transforms_by_indices(unique_indices)
            transform_hashes = transform_hashes[unique_indices]
            # sorted_transform_hashes = np.sort(transform_hashes, axis=None)
            # unique_sorted_transform_hashes = np.zeros_like(sorted_transform_hashes, dtype=bool)
            # unique_sorted_transform_hashes[1:] = sorted_transform_hashes[1:] == sorted_transform_hashes[:-1]
            # unique_sorted_transform_hashes[0] = True
            # # Alternative
            # unique_transform_hashes = pd.Index(transform_hashes).duplicated('first')

            # total_number_of_perturbations = number_of_transforms * number_perturbations_applied
            # Get the shape of the passing perturbations
            perturbation_shape = []
            # num_zeros = 0
            last_perturb_start = 0
            for perturb in perturb_rotation1:
                perturb_end = last_perturb_start + len(perturb)
                perturbation_shape.append(unique_transform_hashes[last_perturb_start:perturb_end].sum())
                # Use if removing zero counts...
                # shape = unique_transform_hashes[last_perturb_start:perturb_end].sum()
                # if shape:
                #     perturbation_shape.append(shape)
                # else:
                #     num_zeros += 1
                last_perturb_start = perturb_end

            number_of_transforms = full_rotation1.shape[0]
            logger.info(f'After culling duplicated transforms, found {number_of_transforms} viable solutions')
            num_zeros = perturbation_shape.count(0)
            if num_zeros:
                logger.info(f'A total of {num_zeros} original transformations had no unique perturbations')
                # Could use if removing zero counts... but probably less clear than above
                # pop_zero_index = perturbation_shape.index(0)
                # while pop_zero_index != -1:
                #     perturbation_shape.pop(pop_zero_index)
                #     pop_zero_index = perturbation_shape.index(0)

        return transform_hashes, perturbation_shape, n_perturbed_dof

    # Calculate metrics on input Pose before any manipulation
    pose_length = pose.number_of_residues
    residue_indices = list(range(pose_length))
    # residue_numbers = [residue.number for residue in pose.residues]
    # entity_tuple = tuple(pose.entities)
    # model_tuple = tuple(models)

    def add_fragments_to_pose(overlap_ghosts: list[int] = None, overlap_surf: list[int] = None,
                              sorted_z_scores: np.ndarray = None):
        """Add observed fragments to the Pose or generate new observations given the Pose state

        If no arguments are passed, the fragment observations will be generated new
        """
        # First, clear any pose information and force identification of the interface
        # del pose._interface_residues
        pose.interface_residues_by_interface = {}
        pose.find_and_split_interface(distance=cb_distance)

        # # Next, set the interface fragment info for gathering of interface metrics
        # if overlap_ghosts is None or overlap_surf is None or sorted_z_scores is None:
        # Remove old fragments
        pose.fragment_queries = {}
        pose.fragment_pairs.clear()
        # Query fragments
        pose.generate_interface_fragments()
        # else:  # Process with provided data
        #     # Return the indices sorted by z_value in ascending order, truncated at the number of passing
        #     sorted_match_scores = match_score_from_z_value(sorted_z_scores)
        #
        #     # These are indexed outside this function
        #     # overlap_ghosts = passing_ghost_indices[sorted_fragment_indices]
        #     # overlap_surf = passing_surf_indices[sorted_fragment_indices]
        #
        #     sorted_int_ghostfrags: list[GhostFragment] = [complete_ghost_frags1[idx] for idx in overlap_ghosts]
        #     sorted_int_surffrags2: list[Residue] = [surf_frags2[idx] for idx in overlap_surf]
        #     # For all matched interface fragments
        #     # Keys are (chain_id, res_num) for every residue that is covered by at least 1 fragment
        #     # Values are lists containing 1 / (1 + z^2) values for every (chain_id, res_num) residue fragment match
        #     # chid_resnum_scores_dict_model1, chid_resnum_scores_dict_model2 = {}, {}
        #     # Number of unique interface mono fragments matched
        #     # unique_frags_info1, unique_frags_info2 = set(), set()
        #     # res_pair_freq_info_list = []
        #     fragment_pairs = list(zip(sorted_int_ghostfrags, sorted_int_surffrags2, sorted_match_scores))
        #     frag_match_info = get_matching_fragment_pairs_info(fragment_pairs)
        #     # pose.fragment_queries = {(model1, model2): frag_match_info}
        #     fragment_metrics = pose.fragment_db.calculate_match_metrics(frag_match_info)
        #     # Tod0 2 when able to take more than 2 Entity
        #     #  The entity_tuple must contain the same Entity instances as in the Pose!
        #     # entity_tuple = models_tuple
        #     # These two pose attributes must be set
        #     pose.fragment_queries = {entity_tuple: frag_match_info}
        #     pose.fragment_metrics = {entity_tuple: fragment_metrics}

    # if job.dock_only:  # Only get pose outputs, no sequences or metrics
    #     pass
    #     # design_ids = pose_names
    #     # logger.info(f'Total {building_blocks} dock trajectory took {time.time() - frag_dock_time_start:.2f}s')
    #     # terminate()  # End of docking run
    #     # return pose_paths
    # elif job.dock.proteinmpnn_score or job.design.sequences:  # Initialize proteinmpnn for dock/design
    # if job.dock.proteinmpnn_score or job.design.sequences:  # Initialize proteinmpnn for dock/design
    pose_length_nan = [np.nan for _ in range(pose_length)]

    # Load evolutionary profiles of interest for optimization/analysis
    if job.design.evolution_constraint:
        # profile_background = {}
        measure_evolution, measure_alignment = load_evolutionary_profile(job.api_db, pose)

        # if pose.evolutionary_profile:
        # profile_background['evolution'] = evolutionary_profile_array = pssm_as_array(pose.evolutionary_profile)
        evolutionary_profile_array = pssm_as_array(pose.evolutionary_profile)
        batch_evolutionary_profile = \
            torch.from_numpy(np.tile(evolutionary_profile_array, (batch_length, 1, 1)))
        # torch_log_evolutionary_profile = torch.from_numpy(np.log(evolutionary_profile_array))
        # else:
        #     pose.log.info('No evolution information')
    else:  # Make an empty collapse_profile
        measure_evolution = measure_alignment = False
        collapse_profile = np.empty(0)
        evolutionary_profile_array = None

    # Calculate hydrophobic collapse for each dock using the collapse_profile if it was calculated
    if measure_evolution:
        hydrophobicity = 'expanded'
    else:
        hydrophobicity = 'standard'
    contact_order_per_res_z, reference_collapse, collapse_profile = \
        pose.get_folding_metrics(hydrophobicity=hydrophobicity)
    if measure_evolution:  # collapse_profile.size:  # Not equal to zero, use the profile instead
        reference_collapse = collapse_profile
    #     reference_mean = np.nanmean(collapse_profile, axis=-2)
    #     reference_std = np.nanstd(collapse_profile, axis=-2)
    #     # How different are the collapse of the MSA profile and the mean of the collapse profile?
    #     reference_collapse = metrics.hydrophobic_collapse_index(evolutionary_profile_array,
    #                                                             alphabet_type=protein_letters_alph1,
    #                                                             hydrophobicity='expanded')
    #     # seq_reference_collapse = reference_collapse
    #     # reference_difference1 = reference_collapse - seq_reference_collapse
    #     # logger.critical('Found a collapse difference between the MSA profile and the reference collapse'
    #     #                 f' of {reference_difference1.sum()}')
    #     # reference_difference2 = reference_collapse - reference_mean
    #     # logger.critical('Found a collapse difference between the MSA profile and the mean of the collapse'
    #                       f'profile of {reference_difference2.sum()}')
    # else:
    #     reference_mean = reference_std = None

    # Todo
    #  enable precise metric acquisition
    # def collect_dock_metrics(score_functions: dict[str, Callable]) -> dict[str, np.ndarray]:
    #     """Perform analysis on the docked Pose instances"""
    #     pose_functions = {}
    #     residue_functions = {}
    #     for score, function in score_functions:
    #         if getattr(sql.PoseMetrics, score, None):
    #             pose_functions[score] = function
    #         else:
    #             residue_functions[score] = function
    #
    #     pose_metrics = []
    #     per_residue_metrics = []
    #     for idx in range(number_of_transforms):
    #         # Add the next set of coordinates
    #         update_pose_coords(idx)
    #
    #         # if number_perturbations_applied > 1:
    #         add_fragments_to_pose()  # <- here generating fragments fresh
    #
    #         pose_metrics.append({score: function(pose) for score, function in pose_functions})
    #         per_residue_metrics.append({score: function(pose) for score, function in residue_functions})
    #
    #         # Reset the fragment_map and fragment_profile for each Entity before calculate_fragment_profile
    #         for entity in pose.entities:
    #             entity.fragment_map = None
    #             # entity.alpha.clear()
    #
    #         # Load fragment_profile (and fragment_metrics) into the analysis
    #         pose.calculate_fragment_profile()
    #         # This could be an empty array if no fragments were found
    #         fragment_profile_array = pose.fragment_profile.as_array()
    #         with catch_warnings():
    #             simplefilter('ignore', category=RuntimeWarning)
    #             # np.log causes -inf at 0, thus we correct these to a 'large' number
    #             corrected_frag_array = np.nan_to_num(np.log(fragment_profile_array), copy=False,
    #                                                  nan=np.nan, neginf=metrics.zero_probability_frag_value)
    #         per_residue_fragment_profile_loss = \
    #             resources.ml.sequence_nllloss(torch_numeric_sequence, torch.from_numpy(corrected_frag_array))
    #
    #         # Remove saved pose attributes from the prior iteration calculations
    #         pose.ss_sequence_indices.clear(), pose.ss_type_sequence.clear()
    #         pose.fragment_metrics.clear()
    #         for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
    #             try:
    #                 delattr(pose, attribute)
    #             except AttributeError:
    #                 pass
    #
    #         # Save pose metrics
    #         # pose_metrics[pose_id] = {
    #         pose_metrics.append({
    #             **pose.calculate_metrics(),  # Also calculates entity.metrics
    #             'dock_collapse_violation': collapse_violation[idx],
    #         })

    # def format_docking_metrics(metrics_: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """From the current pool of docked poses and their collected metrics, format the metrics for selection/output
    #
    #     Args:
    #         metrics_: A dictionary of metric name to metric value where the values are per-residue measurements the
    #             length of the active transformation pool
    #     Returns:
    #         A tuple of DataFrames representing the per-pose and the per-residue metrics. Each has indices from 0-N
    #     """

    def collect_dock_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:  # -> dict[str, np.ndarray]:
        """Perform analysis on the docked Pose instances

        Returns:
            A tuple of DataFrames representing the per-pose and the per-residue metrics. Each has indices from 0-N
        """
        logger.info(f'Collecting metrics for {number_of_transforms} active Poses')

        idx_slice = pd.IndexSlice
        # Unpack scores for output
        collapse_violation = list(repeat(None, number_of_transforms))
        # Get metrics for each Pose
        # nan_blank_data = list(repeat(np.nan, pose_length))
        # unbound_errat = []
        # for idx, entity in enumerate(pose.entities):
        #     _, oligomeric_errat = entity.oligomer.errat(out_path=os.path.devnull)
        #     unbound_errat.append(oligomeric_errat[:entity.number_of_residues])

        torch_numeric_sequence = torch.from_numpy(pose.sequence_numeric)
        if evolutionary_profile_array is not None:
            # evolutionary_profile_array = pssm_as_array(pose.evolutionary_profile)
            # batch_evolutionary_profile = np.tile(evolutionary_profile_array, (number_of_sequences, 1, 1))
            # torch_log_evolutionary_profile = torch.from_numpy(np.log(batch_evolutionary_profile))
            torch_log_evolutionary_profile = torch.from_numpy(np.log(evolutionary_profile_array))
            per_residue_evolutionary_profile_loss = \
                resources.ml.sequence_nllloss(torch_numeric_sequence, torch_log_evolutionary_profile)
            profile_loss = {
                # 'sequence_loss_design': per_residue_design_profile_loss,
                'sequence_loss_evolution': per_residue_evolutionary_profile_loss,
            }
        else:
            profile_loss = {}

        sequence_params = {
            **pose.per_residue_contact_order(),
            # 'errat_deviation': np.concatenate(unbound_errat),
            'type': tuple(pose.sequence),
            **profile_loss
            # Todo 1 each pose...
            #  'sequence_loss_fragment': per_residue_fragment_profile_loss
        }

        # Initialize proteinmpnn for dock/design analysis
        if job.dock.proteinmpnn_score:
            # Retrieve the ProteinMPNN model
            mpnn_model = ml.proteinmpnn_factory()  # Todo 1 accept model_name arg. Now just use the default
            # Set up model sampling type based on symmetry
            if pose.is_symmetric():
                # number_of_symmetry_mates = pose.number_of_symmetry_mates
                # mpnn_sample = mpnn_model.tied_sample
                number_of_residues = pose_length * pose.number_of_symmetry_mates
            else:
                # mpnn_sample = mpnn_model.sample
                number_of_residues = pose_length

            # Modulate memory requirements
            size = len(full_rotation1)  # This is the number of transformations, i.e. the number_of_designs
            # The batch_length indicates how many models could fit in the allocated memory
            batch_length = ml.calculate_proteinmpnn_batch_length(mpnn_model, number_of_residues)
            logger.info(f'Found ProteinMPNN batch_length={batch_length}')

            # Set up parameters to run ProteinMPNN design
            if job.design.ca_only:
                coords_type = 'ca_coords'
                num_model_residues = 1
            else:
                coords_type = 'backbone_coords'
                num_model_residues = 4

            # Set up Pose parameters
            parameters = pose.get_proteinmpnn_params(ca_only=job.design.ca_only,
                                                     interface=measure_interface_during_dock)
            # Todo 2 reinstate if conditional_log_probs
            # # Todo
            # #  Must calculate randn individually if using some feature to describe order
            # parameters['randn'] = pose.generate_proteinmpnn_decode_order()  # to_device=device)

            # Set up interface unbound coordinates
            mpnn_null_idx = resources.ml.MPNN_NULL_IDX
            if measure_interface_during_dock:
                X_unbound = pose.get_proteinmpnn_unbound_coords(ca_only=job.design.ca_only)
                # Add a parameter for the unbound version of X to X
                # extra_batch_parameters = ml.proteinmpnn_to_device(device, **ml.batch_proteinmpnn_input(size=batch_length,
                #                                                                                        X=X_unbound))
                # parameters['X_unbound'] = X_unbound
                # unbound_batch = ml.proteinmpnn_to_device(
                #     device=mpnn_model.device,
                #     **ml.batch_proteinmpnn_input(size=1, X_unbound=X_unbound, mask=parameters['mask'],
                #                                  residue_idx=parameters['residue_idx'],
                #                                  chain_encoding=parameters['chain_encoding'])
                # )
                unbound_batch = \
                    ml.setup_pose_batch_for_proteinmpnn(1, mpnn_model.device, X_unbound=X_unbound,
                                                        mask=parameters['mask'], residue_idx=parameters['residue_idx'],
                                                        chain_encoding=parameters['chain_encoding'])
                # X_unbound = unbound_batch['X_unbound']
                # mask = unbound_batch['mask']
                # residue_idx = unbound_batch['residue_idx']
                # chain_encoding = unbound_batch['chain_encoding']
                with torch.no_grad():
                    unconditional_log_probs_unbound = \
                        mpnn_model.unconditional_probs(unbound_batch['X_unbound'], unbound_batch['mask'],
                                                       unbound_batch['residue_idx'],
                                                       unbound_batch['chain_encoding']).cpu()
                    asu_conditional_softmax_seq_unbound = \
                        np.exp(unconditional_log_probs_unbound[:, :pose_length, :mpnn_null_idx])
                # Remove any reserved GPU memory...
                del unbound_batch
            else:
                raise NotImplementedError(f"{fragment_dock.__name__} isn't written to only measure the complexed state")
                asu_conditional_softmax_seq_unbound = None
            # Disregard X, chain_M_pos, and bias_by_res parameters return and use the pose specific data from below
            # parameters.pop('X')  # overwritten by X_unbound
            parameters.pop('chain_M_pos')
            parameters.pop('bias_by_res')
            # tied_pos = parameters.pop('tied_pos')
            # # Todo 2 if modifying the amount of weight given to each of the copies
            # tied_beta = parameters.pop('tied_beta')
            # # Set the design temperature
            # temperature = job.design.temperatures[0]

            batch_parameters = ml.setup_pose_batch_for_proteinmpnn(batch_length, mpnn_model.device, **parameters)
            mask = batch_parameters['mask']
            residue_idx = batch_parameters['residue_idx']
            chain_encoding = batch_parameters['chain_encoding']

            def proteinmpnn_score_batched_coords(batched_coords: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
                """"""
                actual_batch_length = len(batched_coords)
                if actual_batch_length != batch_length:
                    # if not actual_batch_length:
                    #     return []
                    _mask = mask[:actual_batch_length]
                    _residue_idx = residue_idx[:actual_batch_length]
                    _chain_encoding = chain_encoding[:actual_batch_length]
                else:
                    _mask = mask[:actual_batch_length]
                    _residue_idx = residue_idx[:actual_batch_length]
                    _chain_encoding = chain_encoding[:actual_batch_length]

                # Format the bb coords for ProteinMPNN
                if pose.is_symmetric():
                    # Make each set of coordinates symmetric
                    # Lattice cases have .uc_dimensions set in update_pose_coords()
                    perturbed_bb_coords = np.concatenate(
                        [pose.return_symmetric_coords(coords_) for coords_ in batched_coords])

                    # Todo 2 reinstate if conditional_log_probs
                    # # Symmetrize other arrays
                    # number_of_symmetry_mates = pose.number_of_symmetry_mates
                    # # (batch, number_of_sym_residues, ...)
                    # residue_mask_cpu = np.tile(residue_mask_cpu, (1, number_of_symmetry_mates))
                    # # bias_by_res = np.tile(bias_by_res, (1, number_of_symmetry_mates, 1))
                else:
                    # If entity_bb_coords are individually transformed, then axis=0 works
                    perturbed_bb_coords = np.concatenate(batched_coords, axis=0)

                # Reshape for ProteinMPNN
                # Let -1 fill in the pose length dimension with the number of residues
                # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
                # logger.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')
                X = perturbed_bb_coords.reshape((actual_batch_length, -1, num_model_residues, 3))
                # logger.debug(f'X.shape: {X.shape}')
                with torch.no_grad():
                    X = torch.from_numpy(X).to(dtype=torch.float32, device=mpnn_model.device)
                # X = ml.proteinmpnn_to_device(mpnn_model.device, X=X)

                # Start a unit of work
                #  Taking the KL divergence would indicate how divergent the interfaces are from the
                #  surface. This should be simultaneously minimized (i.e. the lowest evolutionary divergence)
                #  while the aa frequency distribution cross_entropy compared to the fragment profile is
                #  minimized
                # -------------------------------------------
                    unconditional_log_probs = \
                        mpnn_model.unconditional_probs(X, _mask, _residue_idx, _chain_encoding).cpu()
                # Use the sequence as an unknown token then guess the probabilities given the remaining
                # information, i.e. the sequence and the backbone
                # Calculations with this are done using cpu memory and numpy
                # Todo 2 reinstate if conditional_log_probs
                # S_design_null[residue_mask.type(torch.bool)] = mpnn_null_idx
                # chain_residue_mask = chain_mask * residue_mask * mask
                # decoding_order = \
                #     ml.create_decoding_order(randn, chain_residue_mask, tied_pos=tied_pos, to_device=device)
                # conditional_log_probs_null_seq = \
                #     mpnn_model(X, S_design_null, mask, chain_residue_mask, residue_idx, chain_encoding,
                #                None,  # This argument is provided but with below args, is not used
                #                use_input_decoding_order=True, decoding_order=decoding_order).cpu()

                # Remove the gaps index from the softmax input with :mpnn_null_idx]
                asu_conditional_softmax_seq = \
                    np.exp(unconditional_log_probs[:, :pose_length, :mpnn_null_idx])
                # asu_conditional_softmax_null_seq = \
                #     np.exp(conditional_log_probs_null_seq[:, :pose_length, :mpnn_null_idx])
                per_residue_dock_cross_entropy = \
                    metrics.cross_entropy(asu_conditional_softmax_seq,
                                          asu_conditional_softmax_seq_unbound[:actual_batch_length],
                                          per_entry=True)
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

                per_residue_design_indices = np.zeros((actual_batch_length, pose_length), dtype=np.int32)
                # Has shape (batch, number_of_residues)
                # Residues to design are 1, others are 0
                for chunk_idx, design_residues in enumerate(interface_mask):
                    per_residue_design_indices[chunk_idx, design_residues] = 1

                if pose.fragment_profile:
                    # Process the fragment_profiles into an array for cross entropy
                    fragment_profile_array = np.nan_to_num(np.array(fragment_profiles), copy=False, nan=np.nan)
                    # RuntimeWarning: divide by zero encountered in log
                    # np.log causes -inf at 0, thus we need to correct these to a very large number
                    batch_fragment_profile = torch.from_numpy(fragment_profile_array)
                    per_residue_fragment_cross_entropy = \
                        metrics.cross_entropy(asu_conditional_softmax_seq,
                                              batch_fragment_profile,
                                              per_entry=True)
                    #                         mask=per_residue_design_indices,
                    #                         axis=1)
                    # print('batch_fragment_profile', batch_fragment_profile[:, 20:23])
                    # All per_residue metrics look the same. Shape batch_length, number_of_residues
                    # per_residue_evolution_cross_entropy[batch_slice]
                    # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
                    #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
                    #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
                else:
                    per_residue_fragment_cross_entropy = np.empty_like(per_residue_design_indices, dtype=np.float32)
                    per_residue_fragment_cross_entropy[:] = np.nan

                if pose.evolutionary_profile:
                    per_residue_evolution_cross_entropy = \
                        metrics.cross_entropy(asu_conditional_softmax_seq,
                                              batch_evolutionary_profile[:actual_batch_length],
                                              per_entry=True)
                    #                         mask=per_residue_design_indices,
                    #                         axis=1)
                else:  # Populate with null data
                    per_residue_evolution_cross_entropy = np.empty_like(per_residue_fragment_cross_entropy)
                    per_residue_evolution_cross_entropy[:] = np.nan

                if pose.profile:
                    # Process the design_profiles into an array for cross entropy
                    # Todo 1
                    #  need to make scipy.softmax(design_profiles) so scaling matches
                    batch_design_profile = torch.from_numpy(np.array(design_profiles))
                    per_residue_design_cross_entropy = \
                        metrics.cross_entropy(asu_conditional_softmax_seq,
                                              batch_design_profile,
                                              per_entry=True)
                    #                         mask=per_residue_design_indices,
                    #                         axis=1)
                else:  # Populate with null data
                    per_residue_design_cross_entropy = np.empty_like(per_residue_fragment_cross_entropy)
                    per_residue_design_cross_entropy[:] = np.nan

                # Convert to axis=0 list's for below indexing
                per_residue_design_indices = list(per_residue_design_indices)
                per_residue_dock_cross_entropy = list(per_residue_dock_cross_entropy)
                per_residue_design_cross_entropy = list(per_residue_design_cross_entropy)
                per_residue_evolution_cross_entropy = list(per_residue_evolution_cross_entropy)
                per_residue_fragment_cross_entropy = list(per_residue_fragment_cross_entropy)
                _per_residue_data_batched = []
                for idx in range(actual_batch_length):
                    _per_residue_data_batched.append({
                        # This is required to save the interface_residues
                        'interface_residue': per_residue_design_indices[idx],
                        'proteinmpnn_dock_cross_entropy_loss': per_residue_dock_cross_entropy[idx],
                        'proteinmpnn_v_design_probability_cross_entropy_loss':
                            per_residue_design_cross_entropy[idx],
                        'proteinmpnn_v_evolution_probability_cross_entropy_loss':
                            per_residue_evolution_cross_entropy[idx],
                        'proteinmpnn_v_fragment_probability_cross_entropy_loss':
                            per_residue_fragment_cross_entropy[idx],
                    })

                if collapse_profile.size:  # Not equal to zero
                    # # Make data structures
                    # per_residue_collapse = np.zeros((actual_batch_length, pose_length), dtype=np.float32)
                    # per_residue_dock_islands = np.zeros_like(per_residue_collapse)
                    # per_residue_dock_island_significance = np.zeros_like(per_residue_collapse)
                    # per_residue_dock_collapse_significance_by_contact_order_z = np.zeros_like(
                    #     per_residue_collapse)
                    # per_residue_dock_collapse_increase_significance_by_contact_order_z = \
                    #     np.zeros_like(per_residue_collapse)
                    # per_residue_dock_collapse_increased_z = np.zeros_like(per_residue_collapse)
                    # per_residue_dock_collapse_deviation_magnitude = np.zeros_like(per_residue_collapse)
                    # per_residue_dock_sequential_peaks_collapse_z = np.zeros_like(per_residue_collapse)
                    # per_residue_dock_collapse_sequential_z = np.zeros_like(per_residue_collapse)

                    # Include new axis for the sequence iteration to work on an array v
                    collapse_by_pose = \
                        metrics.collapse_per_residue(asu_conditional_softmax_seq[:, None],
                                                     contact_order_per_res_z, reference_collapse,
                                                     alphabet_type=protein_letters_alph1,
                                                     hydrophobicity='expanded')
                    for _data_batched, collapse_metrics in zip(_per_residue_data_batched, collapse_by_pose):
                        _data_batched.update(collapse_metrics)
                        #     {
                        #     'dock_collapse_deviation_magnitude': per_residue_dock_collapse_deviation_magnitude[idx],
                        #     'dock_collapse_increase_significance_by_contact_order_z': per_residue_dock_collapse_increase_significance_by_contact_order_z[idx],
                        #     'dock_collapse_increased_z': per_residue_dock_collapse_increased_z[idx],
                        #     'dock_collapse_new_positions': per_residue_dock_islands[idx],
                        #     'dock_collapse_new_position_significance': per_residue_dock_island_significance[idx],
                        #     'dock_collapse_significance_by_contact_order_z': per_residue_dock_collapse_significance_by_contact_order_z[idx],
                        #     'dock_collapse_sequential_peaks_z': per_residue_dock_sequential_peaks_collapse_z[idx],
                        #     'dock_collapse_sequential_z': per_residue_dock_collapse_sequential_z[idx],
                        #     'dock_hydrophobic_collapse': per_residue_collapse[idx],
                        # })
                        # # Unpack each metric set and add to the batch arrays
                        # per_residue_dock_collapse_deviation_magnitude[pose_idx] = collapse_metrics['collapse_deviation_magnitude']
                        # per_residue_dock_collapse_increase_significance_by_contact_order_z[pose_idx] = collapse_metrics['collapse_increase_significance_by_contact_order_z']
                        # per_residue_dock_collapse_increased_z[pose_idx] = collapse_metrics['collapse_increased_z']
                        # per_residue_dock_islands[pose_idx] = collapse_metrics['collapse_new_positions']
                        # per_residue_dock_island_significance[pose_idx] = collapse_metrics['collapse_new_position_significance']
                        # per_residue_dock_collapse_significance_by_contact_order_z[pose_idx] = collapse_metrics['collapse_significance_by_contact_order_z']
                        # per_residue_dock_sequential_peaks_collapse_z[pose_idx] = collapse_metrics['collapse_sequential_peaks_z']
                        # per_residue_dock_collapse_sequential_z[pose_idx] = collapse_metrics['collapse_sequential_z']
                        # per_residue_collapse[pose_idx] = collapse_metrics['hydrophobic_collapse']

                    # # Check if there are new collapse islands and count
                    # # If there are any then there is a collapse violation
                    # # number_collapse_new_positions_per_designed = \
                    # dock_collapse_violation = \
                    #     per_residue_dock_islands[per_residue_design_indices].sum(axis=-1)
                    # # if np.any(np.logical_and(_per_residue_dock_islands[per_residue_design_indices],
                    # #                          _per_residue_dock_collapse_increased_z[per_residue_design_indices])):
                    # # _poor_collapse = designed_collapse_new_positions > 0
                    #
                    # pose_metrics_batched.extend([{'dock_collapse_violation': violation}
                    #                              for violation in dock_collapse_violation])

                return _per_residue_data_batched
        else:
            batch_parameters = {}
            mask = residue_idx = chain_encoding = None

        # Initialize pose data
        pose_metrics = []
        per_residue_data = []
        pose_ids = list(range(number_of_transforms))
        # ProteinMPNN
        batch_idx = 0
        # pose_metrics_batched = []
        per_residue_data_batched = []
        design_profiles = []
        fragment_profiles = []
        interface_mask = []
        # Stack the entity coordinates to make up a contiguous block for each pose
        # If entity_bb_coords are stacked, then must concatenate along axis=1 to get full pose
        new_coords = []
        for idx in tqdm(pose_ids, total=number_of_transforms):
            # logger.info(f'Metrics for Pose {idx + 1}/{number_of_transforms}')
            # Add the next set of coordinates
            update_pose_coords(idx)

            # if number_perturbations_applied > 1:
            add_fragments_to_pose()  # <- here generating fragments fresh

            # Reset the fragment_map and fragment_profile for each Entity before calculate_fragment_profile
            for entity in pose.entities:
                entity.fragment_map = None
                # entity.alpha.clear()

            # Load fragment_profile (and fragment_metrics) into the analysis
            pose.calculate_fragment_profile()
            # This could be an empty array if no fragments were found
            fragment_profile_array = pose.fragment_profile.as_array()
            with catch_warnings():
                simplefilter('ignore', category=RuntimeWarning)
                # np.log causes -inf at 0, thus we correct these to a 'large' number
                corrected_frag_array = np.nan_to_num(np.log(fragment_profile_array), copy=False,
                                                     nan=np.nan, neginf=metrics.zero_probability_frag_value)
            per_residue_fragment_profile_loss = \
                resources.ml.sequence_nllloss(torch_numeric_sequence, torch.from_numpy(corrected_frag_array))
            # per_residue_data[pose_id] = {
            per_residue_data.append({
                **sequence_params,
                'sequence_loss_fragment': per_residue_fragment_profile_loss
            })

            # Remove saved pose attributes from the prior iteration calculations
            pose.ss_sequence_indices.clear(), pose.ss_type_sequence.clear()
            pose.fragment_metrics.clear()
            for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
                try:
                    delattr(pose, attribute)
                except AttributeError:
                    pass

            # Save pose metrics
            # pose_metrics[pose_id] = {
            pose_metrics.append(pose.calculate_metrics())

            if job.dock.proteinmpnn_score:
                # Save profiles
                fragment_profiles.append(pose.fragment_profile.as_array())
                pose.calculate_profile()
                design_profiles.append(pssm_as_array(pose.profile))

                # Add all interface residues
                # if measure_interface_during_dock:  # job.design.interface:
                design_residues = []
                for number, residues in pose.interface_residues_by_interface_unique.items():
                    design_residues.extend([residue.index for residue in residues])
                # else:
                #     design_residues = list(range(pose_length))
                interface_mask.append(design_residues)
                # Set coords
                new_coords.append(getattr(pose, coords_type))

                batch_idx += 1
                # If the current iteration marks a batch"-sized" unit of work, execute it
                if batch_idx == batch_length:  # or idx + 1 == number_of_transforms:
                    per_residue_data_batched.extend(proteinmpnn_score_batched_coords(new_coords))

                    # Set batch containers to zero for the next iteration
                    batch_idx = 0
                    design_profiles = []
                    fragment_profiles = []
                    interface_mask = []
                    new_coords = []

        if job.dock.proteinmpnn_score:
            # Finish the routine with any remaining proteinmpnn calculations
            if new_coords:
                per_residue_data_batched.extend(proteinmpnn_score_batched_coords(new_coords))
            # Consolidate the unbatched and batched data
            for data, batched_data in zip(per_residue_data, per_residue_data_batched):
                data.update(batched_data)
            # for data, batched_data in zip(pose_metrics, pose_metrics_batched):
            #     data.update(batched_data)
        # else:
        #     pass

        # Construct the main DataFrames, poses_df and residues_df
        poses_df = pd.DataFrame.from_dict(dict(zip(pose_ids, pose_metrics)), orient='index')
        residues_df = pd.concat({pose_id: pd.DataFrame(data, index=residue_indices)
                                 for pose_id, data in zip(pose_ids, per_residue_data)}) \
            .unstack().swaplevel(0, 1, axis=1)

        # Calculate new metrics from combinations of other metrics
        # Add summed residue information to make poses_df
        # if job.dock.proteinmpnn_score:
        #     _per_res_columns = [
        #         'dock_hydrophobic_collapse',  # dock by default not included
        #         'dock_collapse_deviation_magnitude',
        #     ]
        #     _mean_columns = [
        #         'dock_hydrophobicity',
        #         'dock_collapse_variance'
        #     ]
        # Set up column renaming
        if job.dock.proteinmpnn_score:
            collapse_metrics = (
                'collapse_deviation_magnitude',
                'collapse_increase_significance_by_contact_order_z',
                'collapse_increased_z',
                'collapse_new_positions',
                'collapse_new_position_significance',
                'collapse_significance_by_contact_order_z',
                'collapse_sequential_peaks_z',
                'collapse_sequential_z',
                'hydrophobic_collapse')
            unique_columns = residues_df.columns.unique(level=-1)
            _columns = unique_columns.tolist()
            remap_columns = dict(zip(_columns, _columns))
            remap_columns.update(dict(zip(collapse_metrics, (f'dock_{metric_}' for metric_ in collapse_metrics))))
            residues_df.columns = residues_df.columns.set_levels(unique_columns.map(remap_columns), level=-1)
            per_res_columns = [
                # collapse_profile required
                'dock_hydrophobic_collapse',  # dock by default not included
                'dock_collapse_deviation_magnitude',
                # proteinmpnn_score required
                'proteinmpnn_v_design_probability_cross_entropy_loss',
                'proteinmpnn_v_evolution_probability_cross_entropy_loss'
            ]
            mean_columns = [
                # collapse_profile required
                'dock_hydrophobicity',
                'dock_collapse_variance',
                # proteinmpnn_score required
                'proteinmpnn_v_design_probability_cross_entropy_per_residue',
                'proteinmpnn_v_evolution_probability_cross_entropy_per_residue'
            ]
            _rename = dict(zip(per_res_columns, mean_columns))
        else:
            mean_columns = []
            _rename = {}

        summed_poses_df = metrics.sum_per_residue_metrics(
            residues_df, rename_columns=_rename, mean_metrics=mean_columns)
        poses_df = poses_df.join(summed_poses_df.drop('number_residues_interface', axis=1))
        # # Need to remove sequence as it is in pose.calculate_metrics()
        # poses_df = poses_df.join(summed_poses_df.drop('sequence', axis=1))
        if job.dock.proteinmpnn_score:
            interface_df = residues_df.loc[:, idx_slice[:, 'interface_residue']]
            poses_df['dock_collapse_new_positions'] = \
                (residues_df.loc[:, idx_slice[:, 'dock_collapse_new_positions']]
                 * interface_df).sum(axis=1)
            # Update the total loss according to those residues that were actually specified as designable
            poses_df['proteinmpnn_dock_cross_entropy_per_residue'] = \
                (residues_df.loc[:, idx_slice[:, 'proteinmpnn_dock_cross_entropy_loss']]
                 * interface_df).mean(axis=1)
            poses_df['proteinmpnn_v_design_probability_cross_entropy_loss'] = \
                (residues_df.loc[:, idx_slice[:, 'proteinmpnn_v_design_probability_cross_entropy_loss']]
                 * interface_df).mean(axis=1)
            poses_df['proteinmpnn_v_evolution_probability_cross_entropy_loss'] = \
                (residues_df.loc[:, idx_slice[:, 'proteinmpnn_v_evolution_probability_cross_entropy_loss']]
                 * interface_df).mean(axis=1)
            poses_df['proteinmpnn_v_fragment_probability_cross_entropy_loss'] = \
                (residues_df.loc[:, idx_slice[:, 'proteinmpnn_v_fragment_probability_cross_entropy_loss']]
                 * interface_df).mean(axis=1)

            # scores_df['collapse_new_positions'] /= scores_df['pose_length']
            # scores_df['collapse_new_position_significance'] /= scores_df['pose_length']
            poses_df['dock_collapse_significance_by_contact_order_z_mean'] = \
                poses_df['dock_collapse_significance_by_contact_order_z'] / \
                (residues_df.loc[:, idx_slice[:, 'dock_collapse_significance_by_contact_order_z']] != 0) \
                .sum(axis=1)
            # if measure_alignment:
            dock_collapse_increased_df = residues_df.loc[:, idx_slice[:, 'dock_collapse_increased_z']]
            total_increased_collapse = (dock_collapse_increased_df != 0).sum(axis=1)
            # scores_df['dock_collapse_increase_significance_by_contact_order_z_mean'] = \
            #     scores_df['dock_collapse_increase_significance_by_contact_order_z'] / \
            #     total_increased_collapse
            poses_df['dock_collapse_increased_z_mean'] = \
                dock_collapse_increased_df.sum(axis=1) / total_increased_collapse
            # poses_df['dock_collapse_variance'] = \
            #     poses_df['dock_collapse_deviation_magnitude'] / pose_length
            poses_df['dock_collapse_sequential_peaks_z_mean'] = \
                poses_df['dock_collapse_sequential_peaks_z'] / total_increased_collapse
            poses_df['dock_collapse_sequential_z_mean'] = \
                poses_df['dock_collapse_sequential_z'] / total_increased_collapse

            # Update the per_residue loss according to those residues involved in the scoring
            poses_df['proteinmpnn_v_fragment_probability_cross_entropy_per_residue'] = \
                poses_df['proteinmpnn_v_fragment_probability_cross_entropy_loss'] \
                / poses_df['number_residues_interface_fragment_total']
            # poses_df['proteinmpnn_v_design_probability_cross_entropy_per_residue'] = \
            #     poses_df['proteinmpnn_v_design_probability_cross_entropy_loss'] / pose_length
            # poses_df['proteinmpnn_v_evolution_probability_cross_entropy_per_residue'] = \
            #     poses_df['proteinmpnn_v_evolution_probability_cross_entropy_loss'] / pose_length

        # scores_df = metrics.columns_to_new_column(scores_df, metrics.delta_pairs, mode='sub')
        # scores_df = metrics.columns_to_new_column(scores_df, metrics.division_pairs, mode='truediv')
        # if job.design.structures:
        #     scores_df['interface_composition_similarity'] = \
        #         scores_df.apply(metrics.interface_composition_similarity, axis=1)
        # poses_df.drop(metrics.clean_up_intermediate_columns, axis=1, inplace=True, errors='ignore')

        logger.debug(f'Found poses_df with columns: {poses_df.columns.tolist()}')
        logger.debug(f'Found poses_df with index: {poses_df.index.tolist()}')
        return poses_df, residues_df

    # def collect_dock_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:  # -> dict[str, np.ndarray]:
    #     """Perform analysis on the docked Pose instances
    #
    #     Returns:
    #         A tuple of DataFrames representing the per-pose and the per-residue metrics. Each has indices from 0-N
    #     """
    #     logger.info(f'Collecting metrics for {number_of_transforms} active Poses')
    #
    #     # Initialize proteinmpnn for dock/design analysis
    #     if job.dock.proteinmpnn_score:
    #         # Retrieve the ProteinMPNN model
    #         mpnn_model = ml.proteinmpnn_factory()  # Todo 1 accept model_name arg. Now just use the default
    #         # Set up model sampling type based on symmetry
    #         if pose.is_symmetric():
    #             # number_of_symmetry_mates = pose.number_of_symmetry_mates
    #             # mpnn_sample = mpnn_model.tied_sample
    #             number_of_residues = pose_length * pose.number_of_symmetry_mates
    #         else:
    #             # mpnn_sample = mpnn_model.sample
    #             number_of_residues = pose_length
    #
    #         # Modulate memory requirements
    #         size = full_rotation1.shape[0]  # This is the number of transformations, i.e. the number_of_designs
    #         # The batch_length indicates how many models could fit in the allocated memory
    #         batch_length = ml.calculate_proteinmpnn_batch_length(mpnn_model, number_of_residues)
    #         logger.info(f'Found ProteinMPNN batch_length={batch_length}')
    #
    #         # Set up Pose parameters
    #         parameters = pose.get_proteinmpnn_params(ca_only=job.design.ca_only, interface=measure_interface_during_dock)
    #         # Todo 2 reinstate if conditional_log_probs
    #         # # Todo
    #         # #  Must calculate randn individually if using some feature to describe order
    #         # parameters['randn'] = pose.generate_proteinmpnn_decode_order()  # to_device=device)
    #
    #         # Set up interface unbound coordinates
    #         if measure_interface_during_dock:
    #             X_unbound = pose.get_proteinmpnn_unbound_coords(ca_only=job.design.ca_only)
    #             # Add a parameter for the unbound version of X to X
    #             # extra_batch_parameters = ml.proteinmpnn_to_device(device, **ml.batch_proteinmpnn_input(size=batch_length,
    #             #                                                                                        X=X_unbound))
    #             # parameters['X_unbound'] = X_unbound
    #             unbound_batch = ml.proteinmpnn_to_device(
    #                 device=mpnn_model.device,
    #                 **ml.batch_proteinmpnn_input(size=1, X_unbound=X_unbound, mask=parameters['mask'],
    #                                              residue_idx=parameters['residue_idx'],
    #                                              chain_encoding=parameters['chain_encoding'])
    #             )
    #
    #             # X_unbound = torch.from_numpy(unbound_batch['X_unbound']).to(device=device)
    #             # mask = torch.from_numpy(unbound_batch['mask']).to(device=device)
    #             # residue_idx = torch.from_numpy(unbound_batch['residue_idx']).to(device=device)
    #             # chain_encoding = torch.from_numpy(unbound_batch['chain_encoding']).to(device=device)
    #
    #             X_unbound = unbound_batch['X_unbound']
    #             mask = unbound_batch['mask']
    #             residue_idx = unbound_batch['residue_idx']
    #             chain_encoding = unbound_batch['chain_encoding']
    #             with torch.no_grad():
    #                 unconditional_log_probs_unbound = \
    #                     mpnn_model.unconditional_probs(X_unbound, mask, residue_idx, chain_encoding).cpu()
    #                 mpnn_null_idx = resources.ml.MPNN_NULL_IDX
    #                 asu_conditional_softmax_seq_unbound = \
    #                     np.exp(unconditional_log_probs_unbound[:, :pose_length, :mpnn_null_idx])
    #         else:
    #             raise NotImplementedError(f"{fragment_dock.__name__} isn't written to only measure the complexed state")
    #             asu_conditional_softmax_seq_unbound = None
    #         # Disregard X, chain_M_pos, and bias_by_res parameters return and use the pose specific data from below
    #         # parameters.pop('X')  # overwritten by X_unbound
    #         parameters.pop('chain_M_pos')
    #         parameters.pop('bias_by_res')
    #         # tied_pos = parameters.pop('tied_pos')
    #         # # Todo 2 if modifying the amount of weight given to each of the copies
    #         # tied_beta = parameters.pop('tied_beta')
    #         # # Set the design temperature
    #         # temperature = job.design.temperatures[0]
    #
    #         proteinmpnn_time_start = time.time()
    #
    #         @torch.no_grad()  # Ensure no gradients are produced
    #         @resources.ml.batch_calculation(size=size, batch_length=batch_length,
    #                                         setup=ml.setup_pose_batch_for_proteinmpnn,
    #                                         compute_failure_exceptions=(RuntimeError,
    #                                                                     np.core._exceptions._ArrayMemoryError))
    #         def check_dock_for_designability(batch_slice: slice,
    #                                          S: torch.Tensor = None,
    #                                          chain_encoding: torch.Tensor = None,
    #                                          residue_idx: torch.Tensor = None,
    #                                          mask: torch.Tensor = None,
    #                                          pose_length: int = None,
    #                                          # X_unbound: torch.Tensor = None,
    #                                          # Todo 2 reinstate if conditional_log_probs
    #                                          # chain_mask: torch.Tensor = None,
    #                                          # randn: torch.Tensor = None,
    #                                          # tied_pos: Iterable[Container] = None,
    #                                          **batch_parameters) -> dict[str, np.ndarray]:
    #             actual_batch_length = batch_slice.stop - batch_slice.start
    #             # # Get the null_idx
    #             # mpnn_null_idx = resources.ml.MPNN_NULL_IDX
    #             # Get the batch_length
    #             if pose_length is None:
    #                 batch_length, pose_length, *_ = S.shape
    #             else:
    #                 batch_length, *_ = S.shape
    #
    #             # Initialize pose data structures for interface design
    #             residue_mask_cpu = np.zeros((actual_batch_length, pose_length),
    #                                         dtype=np.int32)  # (batch, number_of_residues)
    #             # Stack the entity coordinates to make up a contiguous block for each pose
    #             # If entity_bb_coords are stacked, then must concatenate along axis=1 to get full pose
    #             new_coords = np.zeros((actual_batch_length, pose_length * num_model_residues, 3),
    #                                   dtype=np.float32)  # (batch, number_of_residues, coords_length)
    #
    #             fragment_profiles = []
    #             design_profiles = []
    #             # Use batch_idx to set new numpy arrays, transform_idx (includes perturb_idx) to set coords
    #             for batch_idx, transform_idx in enumerate(range(batch_slice.start, batch_slice.stop)):
    #                 # Get the transformations based on the global index from batch_length
    #                 update_pose_coords(transform_idx)
    #
    #                 # pose.find_and_split_interface(distance=cb_distance)
    #                 # This is done in the below call
    #                 add_fragments_to_pose()  # <- here generating fragments fresh
    #                 # Reset the fragment_profile and fragment_map for each Entity before calculate_fragment_profile
    #                 for entity in pose.entities:
    #                     entity.fragment_map = None
    #                     # entity.alpha.clear()
    #
    #                 # Load fragment_profile into the analysis
    #                 pose.calculate_fragment_profile()
    #                 # if pose.fragment_profile:
    #                 fragment_profiles.append(pose.fragment_profile.as_array())
    #                 # else:
    #                 #     fragment_profiles.append(pose.fragment_profile.as_array())
    #
    #                 # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.calculate_metrics()
    #                 # # Remove saved pose attributes from the prior iteration calculations
    #                 # pose.ss_sequence_indices.clear(), pose.ss_type_sequence.clear()
    #                 # pose.fragment_metrics.clear()
    #                 # for attribute in ['_design_residues', '_interface_residues']:  # _assembly_minimally_contacting
    #                 #     try:
    #                 #         delattr(pose, attribute)
    #                 #     except AttributeError:
    #                 #         pass
    #                 #
    #                 # # Calculate pose metrics
    #                 # interface_metrics[design_id] = pose.calculate_metrics()
    #                 # # Todo use the below calls to grab fragments and thus nanohedra_score from pose.calculate_metrics()
    #                 pose.calculate_profile()
    #                 # # Todo if want to throw away missing fragments
    #                 # if sum(pose.alpha) == 0:  # No useful fragment observations
    #                 #     actual_batch_length -= 1
    #                 #     continue
    #                 design_profiles.append(pssm_as_array(pose.profile))
    #
    #                 # Add all interface residues
    #                 if measure_interface_during_dock:  # job.design.interface:
    #                     design_residues = []
    #                     for number, residues in pose.residues_by_interface.items():
    #                         design_residues.extend([residue.index for residue in residues])
    #                 else:
    #                     design_residues = list(range(pose_length))
    #
    #                 # Todo 2 reinstate if conditional_log_probs
    #                 # Residues to design are 1, others are 0
    #                 residue_mask_cpu[batch_idx, design_residues] = 1
    #                 # Set coords
    #                 new_coords[batch_idx] = getattr(pose, coords_type)
    #
    #             # If entity_bb_coords are individually transformed, then axis=0 works
    #             perturbed_bb_coords = np.concatenate(new_coords, axis=0)
    #             # # Todo if want to throw away missing fragments
    #             # perturbed_bb_coords = np.concatenate(new_coords[:actual_batch_length], axis=0)
    #
    #             # Format the bb coords for ProteinMPNN
    #             if pose.is_symmetric():
    #                 # Make each set of coordinates symmetric. Lattice cases have uc_dimensions passed in update_pose_coords
    #                 _perturbed_bb_coords = []
    #                 for idx in range(perturbed_bb_coords.shape[0]):
    #                     _perturbed_bb_coords.append(pose.return_symmetric_coords(perturbed_bb_coords[idx]))
    #
    #                 # Let -1 fill in the pose length dimension with the number of residues
    #                 # 4 is shape of backbone coords (N, Ca, C, O), 3 is x,y,z
    #                 perturbed_bb_coords = np.concatenate(_perturbed_bb_coords)
    #
    #                 # Todo 2 reinstate if conditional_log_probs
    #                 # # Symmetrize other arrays
    #                 # number_of_symmetry_mates = pose.number_of_symmetry_mates
    #                 # # (batch, number_of_sym_residues, ...)
    #                 # residue_mask_cpu = np.tile(residue_mask_cpu, (1, number_of_symmetry_mates))
    #                 # # bias_by_res = np.tile(bias_by_res, (1, number_of_symmetry_mates, 1))
    #
    #             # Reshape for ProteinMPNN
    #             logger.debug(f'perturbed_bb_coords.shape: {perturbed_bb_coords.shape}')
    #             X = perturbed_bb_coords.reshape((actual_batch_length, -1, num_model_residues, 3))
    #             logger.debug(f'X.shape: {X.shape}')
    #             # Save design_indices
    #             _residue_indices_of_interest = residue_mask_cpu.astype(bool)
    #             # Todo 2 reinstate if conditional_log_probs
    #             # _residue_indices_of_interest = residue_mask_cpu[:, :pose_length].astype(bool)
    #
    #             # Update different parameters to the identified device
    #             batch_parameters.update(ml.proteinmpnn_to_device(device, X=X))  # , chain_M_pos=residue_mask_cpu))
    #             # Different across poses
    #             X = batch_parameters.pop('X')
    #             # Todo 2 reinstate if conditional_log_probs
    #             # residue_mask = batch_parameters.get('chain_M_pos', None)
    #             # # Potentially different across poses
    #             # bias_by_res = batch_parameters.get('bias_by_res', None)
    #             # Todo calculate individually if using some feature to describe order
    #             #  MUST reinstate the removal from scope after finished with this batch
    #             # decoding_order = pose.generate_proteinmpnn_decode_order(to_device=device)
    #             # decoding_order.repeat(actual_batch_length, 1)
    #             # with torch.no_grad():  # Ensure no gradients are produced
    #             # Unpack constant parameters and slice reused parameters only once
    #             # X_unbound = batch_parameters.pop('X')  # Remove once batch_calculation()
    #             # chain_mask = batch_parameters.pop('chain_mask')
    #             # chain_encoding = batch_parameters.pop('chain_encoding')
    #             # residue_idx = batch_parameters.pop('residue_idx')
    #             # mask = batch_parameters.pop('mask')
    #             # randn = batch_parameters.pop('randn')
    #
    #             # ml.proteinmpnn_batch_design(batch_slice,
    #             #                             mpnn_model: ProteinMPNN,
    #             #                             temperatures=job.design.temperatures,
    #             #                             **parameters,  # (randn, S, chain_mask, chain_encoding, residue_idx, mask, temperatures, pose_length, bias_by_res, tied_pos, X_unbound)
    #             #                             **batch_parameters  # (X, chain_M_pos, bias_by_res)
    #             # ml.proteinmpnn_batch_design(batch_slice: slice, proteinmpnn: ProteinMPNN,
    #             #                             X: torch.Tensor = None,
    #             #                             randn: torch.Tensor = None,
    #             #                             S: torch.Tensor = None,
    #             #                             chain_mask: torch.Tensor = None,
    #             #                             chain_encoding: torch.Tensor = None,
    #             #                             residue_idx: torch.Tensor = None,
    #             #                             mask: torch.Tensor = None,
    #             #                             temperatures: Sequence[float] = (0.1,),
    #             #                             pose_length: int = None,
    #             #                             bias_by_res: torch.Tensor = None,
    #             #                             tied_pos: Iterable[Container] = None,
    #             #                             X_unbound: torch.Tensor = None,
    #             #                             **batch_parameters
    #             #                             ) -> dict[str, np.ndarray]:
    #
    #             # Todo 2 reinstate if conditional_log_probs
    #             # # Clone the data from the sequence tensor so that it can be set with the null token below
    #             # S_design_null = S.detach().clone()
    #             # Get the provided batch_length from wrapping function. actual_batch_length may be smaller on last batch
    #             # batch_length = X.shape[0]
    #             # batch_length = X_unbound.shape[0]
    #             if actual_batch_length != batch_length:
    #                 # Slice these for the last iteration
    #                 # X_unbound = X_unbound[:actual_batch_length]  # , None)
    #                 chain_encoding = chain_encoding[:actual_batch_length]  # , None)
    #                 residue_idx = residue_idx[:actual_batch_length]  # , None)
    #                 mask = mask[:actual_batch_length]  # , None)
    #                 # Todo 2 reinstate if conditional_log_probs
    #                 # chain_mask = chain_mask[:actual_batch_length]  # , None)
    #                 # randn = randn[:actual_batch_length]
    #                 # S_design_null = S_design_null[:actual_batch_length]  # , None)
    #                 # # Unpack, unpacked keyword args
    #                 # omit_AA_mask = batch_parameters.get('omit_AA_mask')
    #                 # pssm_coef = batch_parameters.get('pssm_coef')
    #                 # pssm_bias = batch_parameters.get('pssm_bias')
    #                 # pssm_log_odds_mask = batch_parameters.get('pssm_log_odds_mask')
    #                 # # Set keyword args
    #                 # batch_parameters['omit_AA_mask'] = omit_AA_mask[:actual_batch_length]
    #                 # batch_parameters['pssm_coef'] = pssm_coef[:actual_batch_length]
    #                 # batch_parameters['pssm_bias'] = pssm_bias[:actual_batch_length]
    #                 # batch_parameters['pssm_log_odds_mask'] = pssm_log_odds_mask[:actual_batch_length]
    #
    #             # See if the pose is useful to design based on constraints of collapse
    #             # Measure the conditional amino acid probabilities at each residue to see
    #             # how they compare to various profiles from the Pose multiple sequence alignment
    #             # If conditional_probs() are measured, then we need a batched_decoding order
    #             # conditional_start_time = time.time()
    #
    #             # Use the sequence as an unknown token then guess the probabilities given the remaining
    #             # information, i.e. the sequence and the backbone
    #             # Calculations with this are done using cpu memory and numpy
    #             # Todo 2 reinstate if conditional_log_probs
    #             # S_design_null[residue_mask.type(torch.bool)] = mpnn_null_idx
    #             # chain_residue_mask = chain_mask * residue_mask * mask
    #             # decoding_order = ml.create_decoding_order(randn, chain_residue_mask, tied_pos=tied_pos, to_device=device)
    #             # conditional_log_probs_null_seq = \
    #             #     mpnn_model(X, S_design_null, mask, chain_residue_mask, residue_idx, chain_encoding,
    #             #                None,  # This argument is provided but with below args, is not used
    #             #                use_input_decoding_order=True, decoding_order=decoding_order).cpu()
    #             unconditional_log_probs = \
    #                 mpnn_model.unconditional_probs(X, mask, residue_idx, chain_encoding).cpu()
    #             #  Taking the KL divergence would indicate how divergent the interfaces are from the
    #             #  surface. This should be simultaneously minimized (i.e. lowest evolutionary divergence)
    #             #  while the aa frequency distribution cross_entropy compared to the fragment profile is
    #             #  minimized
    #             # Remove the gaps index from the softmax input -> ... :, :mpnn_null_idx]
    #             # asu_conditional_softmax_null_seq = \
    #             #     np.exp(conditional_log_probs_null_seq[:, :pose_length, :mpnn_null_idx])
    #             asu_conditional_softmax_seq = \
    #                 np.exp(unconditional_log_probs[:, :pose_length, :mpnn_null_idx])
    #             _per_residue_proteinmpnn_dock_cross_entropy = \
    #                 metrics.cross_entropy(asu_conditional_softmax_seq,
    #                                       asu_conditional_softmax_seq_unbound[:actual_batch_length],
    #                                       per_entry=True)
    #             # asu_conditional_softmax
    #             # tensor([[[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          ...,
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]],
    #             #          ...
    #             #         [[0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          [0.0273, 0.0125, 0.0200,  ..., 0.0073, 0.0102, 0.0052],
    #             #          ...,
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059],
    #             #          [0.0091, 0.0078, 0.0101,  ..., 0.0038, 0.0029, 0.0059]]])
    #
    #             if pose.fragment_profile:
    #                 # Process the fragment_profiles into an array for cross entropy
    #                 fragment_profile_array = np.array(fragment_profiles)
    #                 # RuntimeWarning: divide by zero encountered in log
    #                 # np.log causes -inf at 0, thus we need to correct these to a very large number
    #                 batch_fragment_profile = torch.from_numpy(np.nan_to_num(fragment_profile_array, copy=False, nan=np.nan))
    #                 _per_residue_fragment_cross_entropy = \
    #                     metrics.cross_entropy(asu_conditional_softmax_seq,
    #                                           batch_fragment_profile,
    #                                           per_entry=True)
    #                 #                         mask=_residue_indices_of_interest,
    #                 #                         axis=1)
    #                 # print('batch_fragment_profile', batch_fragment_profile[:, 20:23])
    #                 # All per_residue metrics look the same. Shape batch_length, number_of_residues
    #                 # per_residue_evolution_cross_entropy[batch_slice]
    #                 # [[-3.0685883 -3.575249  -2.967545  ... -3.3111317 -3.1204746 -3.1201541]
    #                 #  [-3.0685873 -3.5752504 -2.9675443 ... -3.3111336 -3.1204753 -3.1201541]
    #                 #  [-3.0685952 -3.575687  -2.9675474 ... -3.3111277 -3.1428783 -3.1201544]]
    #             else:
    #                 _per_residue_fragment_cross_entropy = np.empty_like(_residue_indices_of_interest, dtype=np.float32)
    #                 _per_residue_fragment_cross_entropy[:] = np.nan
    #
    #             if pose.evolutionary_profile:
    #                 _per_residue_evolution_cross_entropy = \
    #                     metrics.cross_entropy(asu_conditional_softmax_seq,
    #                                           batch_evolutionary_profile[:actual_batch_length],
    #                                           per_entry=True)
    #                 #                         mask=_residue_indices_of_interest,
    #                 #                         axis=1)
    #             else:  # Populate with null data
    #                 _per_residue_evolution_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
    #                 _per_residue_evolution_cross_entropy[:] = np.nan
    #
    #             if pose.profile:
    #                 # Process the design_profiles into an array for cross entropy
    #                 # Todo 1
    #                 #  need to make scipy.softmax(design_profiles) so scaling matches
    #                 batch_design_profile = torch.from_numpy(np.array(design_profiles))
    #                 _per_residue_design_cross_entropy = \
    #                     metrics.cross_entropy(asu_conditional_softmax_seq,
    #                                           batch_design_profile,
    #                                           per_entry=True)
    #                 #                         mask=_residue_indices_of_interest,
    #                 #                         axis=1)
    #             else:  # Populate with null data
    #                 _per_residue_design_cross_entropy = np.empty_like(_per_residue_fragment_cross_entropy)
    #                 _per_residue_design_cross_entropy[:] = np.nan
    #
    #             if collapse_profile.size:  # Not equal to zero
    #                 # Make data structures
    #                 _per_residue_collapse = np.zeros((actual_batch_length, pose_length), dtype=np.float32)
    #                 _per_residue_dock_islands = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_island_significance = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_collapse_significance_by_contact_order_z = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_collapse_increase_significance_by_contact_order_z = \
    #                     np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_collapse_increased_z = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_collapse_deviation_magnitude = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_sequential_peaks_collapse_z = np.zeros_like(_per_residue_collapse)
    #                 _per_residue_dock_collapse_sequential_z = np.zeros_like(_per_residue_collapse)
    #
    #                 # Include new axis for the sequence iteration to work on an array... v
    #                 collapse_by_pose = \
    #                     metrics.collapse_per_residue(asu_conditional_softmax_seq[:, np.newaxis],
    #                                                  contact_order_per_res_z, reference_collapse,
    #                                                  alphabet_type=protein_letters_alph1,
    #                                                  hydrophobicity='expanded')
    #                 for pose_idx, collapse_metrics in enumerate(collapse_by_pose):
    #                     # Unpack each metric set and add to the batch arrays
    #                     _per_residue_dock_collapse_deviation_magnitude[pose_idx] = \
    #                         collapse_metrics['collapse_deviation_magnitude']
    #                     _per_residue_dock_collapse_increase_significance_by_contact_order_z[pose_idx] = \
    #                         collapse_metrics['collapse_increase_significance_by_contact_order_z']
    #                     _per_residue_dock_collapse_increased_z[pose_idx] = collapse_metrics['collapse_increased_z']
    #                     _per_residue_dock_islands[pose_idx] = collapse_metrics['collapse_new_positions']
    #                     _per_residue_dock_island_significance[pose_idx] = \
    #                         collapse_metrics['collapse_new_position_significance']
    #                     _per_residue_dock_collapse_significance_by_contact_order_z[pose_idx] = \
    #                         collapse_metrics['collapse_significance_by_contact_order_z']
    #                     _per_residue_dock_sequential_peaks_collapse_z[pose_idx] = \
    #                         collapse_metrics['collapse_sequential_peaks_z']
    #                     _per_residue_dock_collapse_sequential_z[pose_idx] = collapse_metrics['collapse_sequential_z']
    #                     _per_residue_collapse[pose_idx] = collapse_metrics['hydrophobic_collapse']
    #
    #                 # Check if there are new collapse islands and count
    #                 # If there are any then there is a collapse violation
    #                 number_collapse_new_positions_per_designed = \
    #                     _per_residue_dock_islands[_residue_indices_of_interest].sum(axis=-1)
    #                 # if np.any(np.logical_and(_per_residue_dock_islands[_residue_indices_of_interest],
    #                 #                          _per_residue_dock_collapse_increased_z[_residue_indices_of_interest])):
    #                 # _poor_collapse = designed_collapse_new_positions > 0
    #                 collapse_fit_parameters = {
    #                     # The below structures have a shape (batch_length, pose_length)
    #                     'collapse_new_positions': _per_residue_dock_islands,
    #                     'collapse_new_position_significance': _per_residue_dock_island_significance,
    #                     'collapse_significance_by_contact_order_z':
    #                         _per_residue_dock_collapse_significance_by_contact_order_z,
    #                     'collapse_increase_significance_by_contact_order_z':
    #                         _per_residue_dock_collapse_increase_significance_by_contact_order_z,
    #                     'collapse_increased_z': _per_residue_dock_collapse_increased_z,
    #                     'collapse_deviation_magnitude': _per_residue_dock_collapse_deviation_magnitude,
    #                     'collapse_sequential_peaks_z': _per_residue_dock_sequential_peaks_collapse_z,
    #                     'collapse_sequential_z': _per_residue_dock_collapse_sequential_z,
    #                     # The below structure has shape (batch_length,)
    #                     'collapse_violation': number_collapse_new_positions_per_designed > 0,
    #                     'hydrophobic_collapse': _per_residue_collapse
    #                 }
    #             else:
    #                 # _per_residue_dock_islands = _per_residue_dock_island_significance = \
    #                 #     _per_residue_dock_collapse_significance_by_contact_order_z = \
    #                 #     _per_residue_dock_collapse_increase_significance_by_contact_order_z =
    #                 #     _per_residue_dock_collapse_increased_z = _per_residue_dock_collapse_deviation_magnitude = \
    #                 #     _per_residue_dock_sequential_peaks_collapse_z = _per_residue_dock_collapse_sequential_z = \
    #                 #     np.zeros((actual_batch_length, pose_length), dtype=np.float32)
    #                 # Initialize collapse measurement container
    #                 # _poor_collapse = [0 for _ in range(actual_batch_length)]
    #                 collapse_fit_parameters = {}
    #
    #             # if collapse_profile.size:  # Not equal to zero
    #             #     # Take the hydrophobic collapse of the log probs to understand the profiles "folding"
    #             #     _poor_collapse = [0 for _ in range(actual_batch_length)]
    #             #     _per_residue_mini_batch_collapse_z = \
    #             #         np.zeros((actual_batch_length, pose_length), dtype=np.float32)
    #             #     for pose_idx in range(actual_batch_length):
    #             #         # Only include the residues in the ASU
    #             #         design_probs_collapse = \
    #             #             hydrophobic_collapse_index(asu_conditional_softmax_null_seq[pose_idx],
    #             #                                        # asu_unconditional_softmax,
    #             #                                        alphabet_type=ml.mpnn_alphabet)
    #             #         # Compare the sequence collapse to the pose collapse
    #             #         # USE:
    #             #         #  contact_order_per_res_z, reference_collapse, collapse_profile
    #             #         # print('HCI profile mean', collapse_profile_mean)
    #             #         # print('HCI profile std', collapse_profile_std)
    #             #         _per_residue_mini_batch_collapse_z[pose_idx] = collapse_z = \
    #             #             z_score(design_probs_collapse, collapse_profile_mean, collapse_profile_std)
    #             #         # folding_loss = ml.sequence_nllloss(S_sample, design_probs_collapse)  # , mask_for_loss)
    #             #         pose_idx_residues_of_interest = _residue_indices_of_interest[pose_idx]
    #             #         designed_indices_collapse_z = collapse_z[pose_idx_residues_of_interest]
    #             #         # magnitude_of_collapse_z_deviation = np.abs(designed_indices_collapse_z)
    #             #         # Check if dock has collapse larger than collapse_significance_threshold and increased collapse
    #             #         if np.any(np.logical_and(design_probs_collapse[pose_idx_residues_of_interest]
    #             #                                  > collapse_significance_threshold,
    #             #                                  designed_indices_collapse_z > 0)):
    #             #             _poor_collapse[pose_idx] = 1
    #             #             # logger.warning(f'***Collapse is larger than one standard deviation. '
    #             #             #                f'Pose is *** being considered')
    #             #             # print('design_probs_collapse', design_probs_collapse[pose_idx_residues_of_interest])
    #             #             # This is the collapse value at each residue_of_interest
    #             #             # print('designed_indices_collapse_z', designed_indices_collapse_z)
    #             #             # This is the collapse z score from the Pose profile at each residue_of_interest
    #             #             # design_probs_collapse
    #             #             # [0.1229698  0.14987233 0.23318215 0.23268045 0.23882663 0.24801936
    #             #             #  0.25622816 0.44975936 0.43138875 0.3607946  0.3140504  0.28207788
    #             #             #  0.27033003 0.27388856 0.28031376 0.28897327 0.14254868 0.13711281
    #             #             #  0.12078322 0.11563808 0.13515363 0.16421124 0.16638894 0.16817969
    #             #             #  0.16234223 0.19553652 0.20065537 0.1901575  0.17455298 0.17621328
    #             #             #  0.20747318 0.21465868 0.22461864 0.21520302 0.21346277 0.2054776
    #             #             #  0.17700449 0.15074518 0.11202089 0.07674509 0.08504518 0.09990609
    #             #             #  0.16057604 0.14554144 0.14646661 0.15743639 0.2136532  0.23222249
    #             #             #  0.26718637]
    #             #             # designed_indices_collapse_z
    #             #             # [-0.80368181 -1.2787087   0.71124918  1.04688287  1.26099661 -0.17269616
    #             #             #  -0.06417628  1.16625098  0.94364294  0.62500235  0.53019078  0.5038286
    #             #             #   0.59372686  0.82563642  1.12022683  1.1989269  -1.07529947 -1.27769417
    #             #             #  -1.24323295 -0.95376269  0.55229076  1.05845308  0.62604691  0.20474606
    #             #             #  -0.20987778 -0.45545679 -0.40602295 -0.54974293 -0.72873982 -0.84489538
    #             #             #  -0.8104777  -0.80596935 -0.71591074 -0.79774316 -0.75114322 -0.77010185
    #             #             #  -0.63265472 -0.61240502 -0.69975283 -1.11501543 -0.81130281 -0.64497745
    #             #             #  -0.10221637 -0.32925792 -0.53646227 -0.54949522 -0.35537453 -0.28560236
    #             #             #   0.23599237]
    #             #         # else:
    #             #         #     _poor_collapse.append(0)
    #             #         #     logger.critical(
    #             #         #         f'Total deviation={magnitude_of_collapse_z_deviation.sum()}. '
    #             #         #         f'Mean={designed_indices_collapse_z.mean()}'
    #             #         #         f'Standard Deviation={designed_indices_collapse_z.std()}')
    #             #     # _total_collapse_favorability.extend(_poor_collapse)
    #             #     # per_residue_design_indices[batch_slice] = _residue_indices_of_interest
    #             #     # per_residue_batch_collapse_z[batch_slice] = _per_residue_mini_batch_collapse_z
    #             # else:  # Populate with null data
    #             #     _per_residue_mini_batch_collapse_z = _per_residue_evolution_cross_entropy.copy()
    #             #     _per_residue_mini_batch_collapse_z[:] = np.nan
    #             #     _poor_collapse = _per_residue_mini_batch_collapse_z[:, 0]
    #
    #             return {
    #                 **collapse_fit_parameters,
    #                 # The below structures have a shape (batch_length, pose_length)
    #                 'design_indices': _residue_indices_of_interest,
    #                 'dock_cross_entropy': _per_residue_proteinmpnn_dock_cross_entropy,
    #                 'design_cross_entropy': _per_residue_design_cross_entropy,
    #                 'evolution_cross_entropy': _per_residue_evolution_cross_entropy,
    #                 'fragment_cross_entropy': _per_residue_fragment_cross_entropy,
    #             }
    #
    #         # Initialize correct data for the calculation
    #         proteinmpnn_kwargs = dict(pose_length=pose_length,
    #                                   temperatures=job.design.temperatures)
    #         # probabilities = np.empty((size, number_of_residues, mpnn_alphabet_length, dtype=np.float32))
    #         # if job.dock.proteinmpnn_score:
    #         # Set up ProteinMPNN output data structures
    #         # To use torch.nn.NLLL() must use dtype Long -> np.int64, not Int -> np.int32
    #         # generated_sequences = np.empty((size, pose_length), dtype=np.int64)
    #         per_residue_proteinmpnn_dock_cross_entropy = np.empty((size, pose_length), dtype=np.float32)
    #         per_residue_evolution_cross_entropy = np.empty_like(per_residue_proteinmpnn_dock_cross_entropy)
    #         per_residue_fragment_cross_entropy = np.empty_like(per_residue_proteinmpnn_dock_cross_entropy)
    #         per_residue_design_cross_entropy = np.empty_like(per_residue_proteinmpnn_dock_cross_entropy)
    #         dock_returns = {
    #             'dock_cross_entropy': per_residue_proteinmpnn_dock_cross_entropy,
    #             'design_cross_entropy': per_residue_design_cross_entropy,
    #             'evolution_cross_entropy': per_residue_evolution_cross_entropy,
    #             'fragment_cross_entropy': per_residue_fragment_cross_entropy,
    #         }
    #
    #         if collapse_profile.size:
    #             per_residue_collapse = np.empty((size, pose_length), dtype=np.float32)
    #             collapse_returns = {'collapse_new_positions': per_residue_collapse,
    #                                 'collapse_new_position_significance': np.zeros_like(per_residue_collapse),
    #                                 'collapse_significance_by_contact_order_z': np.zeros_like(per_residue_collapse),
    #                                 'collapse_increase_significance_by_contact_order_z': np.zeros_like(per_residue_collapse),
    #                                 'collapse_increased_z': np.zeros_like(per_residue_collapse),
    #                                 'collapse_deviation_magnitude': np.zeros_like(per_residue_collapse),
    #                                 'collapse_sequential_peaks_z': np.zeros_like(per_residue_collapse),
    #                                 'collapse_sequential_z': np.zeros_like(per_residue_collapse),
    #                                 'collapse_violation': np.zeros((size,), dtype=bool),
    #                                 'hydrophobic_collapse': np.zeros_like(per_residue_collapse),
    #                                 }
    #         else:
    #             collapse_returns = {}
    #
    #         # Perform the calculation
    #         all_returns = {
    #             # Include design indices in both dock and design (used for output of residues_df)
    #             'design_indices': np.zeros((size, pose_length), dtype=bool),
    #             **dock_returns,
    #             **collapse_returns
    #         }
    #         logger.info(f'Starting scoring with ProteinMPNN')
    #         # This is used for understanding dock fit only
    #         scores = _check_dock_for_designability(**proteinmpnn_kwargs,
    #                                                return_containers=all_returns,
    #                                                setup_args=(device,),
    #                                                setup_kwargs=parameters)
    #         logger.info(f'ProteinMPNN docking analysis took {time.time() - proteinmpnn_time_start:8f}')
    #     else:  # Interface metrics be captured in format_docking_metrics, return empty
    #         scores = {}
    #
    #     # return scores
    #     # Format metrics for each pose
    #     return format_docking_metrics(scores)

    def optimize_found_transformations_by_metrics() -> tuple[pd.DataFrame, list[int]]:  # float:
        """Perform a cycle of (optional) transformation perturbation, and then score and select those which are ranked
        highest

        Returns:
            A tuple containing the DataFrame containing the selected metrics for selected Poses and the identifiers for
                those Poses
            # The mean value of the acquired metric for all found poses
        """
        nonlocal poses_df, residues_df
        nonlocal total_dof_perturbed
        # nonlocal number_of_transforms, optimize_round, poses_df, residues_df
        # # Set the cluster number to the incoming number_of_transforms
        # number_of_transform_clusters = number_of_transforms
        # total_dof_perturbed = sym_entry.total_dof
        # if job.dock.perturb_dof_rot or job.dock.perturb_dof_tx:
        if any((sym_entry.number_dof_rotation, sym_entry.number_dof_translation)):
            nonlocal rotation_steps, translation_perturb_steps
            # Perform perturbations to the allowed degrees of freedom
            # Modify the perturbation amount by half as the space is searched to completion
            # Reduce rotation_step before as the original step size was already searched
            rotation_steps = tuple(step * .5 for step in rotation_steps)
            current_transformation_ids, number_of_perturbs_per_cluster, total_dof_perturbed = perturb_transformations()
            # Sets number_perturbations_applied, number_of_transforms,
            #  full_rotation1, full_rotation2, full_int_tx1, full_int_tx2,
            #  full_optimal_ext_dof_shifts, full_ext_tx1, full_ext_tx2
            # Reduce translation_perturb_steps after as the original step size was never searched
            translation_perturb_steps = tuple(step * .5 for step in translation_perturb_steps)
        # elif sym_entry.external_dof:  # The DOF are not such that perturbation would be of much benefit
        else:
            raise NotImplementedError(f"Can't perturb external dof only quite yet")

            # Perform optimization by means of optimal_tx
            def minimize_translations():
                """"""
                logger.info(f'Optimizing transformations')
                # The application of total_dof_perturbed might not be necessary as this optimizes fully
                total_dof_perturbed = sym_entry.total_dof
                # Remake the optimal shifts given each of the passing ghost fragment/surface fragment pairs
                optimal_ext_dof_shifts = np.zeros((number_of_transforms, 3), dtype=float)
                for idx in range(number_of_transforms):
                    update_pose_coords(idx)
                    add_fragments_to_pose()
                    passing_ghost_coords = []
                    passing_surf_coords = []
                    reference_rmsds = []
                    for ghost_frag, surf_frag, match_score in pose.fragment_pairs:
                        passing_ghost_coords.append(ghost_frag.guide_coords)
                        passing_surf_coords.append(surf_frag.guide_coords)
                        reference_rmsds.append(ghost_frag.rmsd)

                    transform_passing_shifts = \
                        optimal_tx.solve_optimal_shifts(passing_ghost_coords, passing_surf_coords, reference_rmsds)
                    mean_transform = transform_passing_shifts.mean(axis=0)

                    # Inherent in minimize_translations() call due to DOF requirements of preceding else:
                    if sym_entry.unit_cell:
                        # Must take the optimal_ext_dof_shifts and multiply the column number by the corresponding row
                        # in the sym_entry.external_dof#
                        # optimal_ext_dof_shifts[0] scalar * sym_entry.group_external_dof[0] (1 row, 3 columns)
                        # Repeat for additional DOFs, then add all up within each row.
                        # For a single DOF, multiplication won't matter as only one matrix element will be available
                        #
                        # # Must find positive indices before external_dof1 multiplication in case negatives there
                        # positive_indices = \
                        #     np.flatnonzero(np.all(transform_passing_shifts[:, :sym_entry.number_dof_external] >= 0, axis=1))
                        # number_passing_shifts = positive_indices.shape[0]
                        optimal_ext_dof_shifts[idx, :sym_entry.number_dof_external] = \
                            mean_transform[:sym_entry.number_dof_external]

            # Using the current transforms, create a hash to uniquely label them and apply to the indices
            current_transformation_ids = create_transformation_hash()
            minimize_translations()

        weighted_trajectory_df: pd.DataFrame = prioritize_transforms_by_selection()
        weighted_trajectory_df_index = weighted_trajectory_df.index

        if number_perturbations_applied > 1:
            # Sort each perturbation cluster members by the prioritized metric
            # # Round down the sqrt of the number_perturbations_applied
            # top_perturb_hits = int(math.sqrt(number_perturbations_applied) + .5)
            # Used to progressively limit search as clusters deepen
            if optimize_round == 1:
                top_perturb_hits = total_dof_perturbed
            else:
                top_perturb_hits = 1
            logger.info(f'Selecting the top {top_perturb_hits} transformations from each perturbation')
            # top_perturb_hits = int(total_dof_perturbed/optimize_round + .5)

            top_transform_cluster_indices: list[int] = []
            perturb_passing_indices: list[list[int]] = []
            # # number_of_transform_clusters is approximate here due to clashing perturbations
            # number_of_transform_clusters = int(number_of_transforms // number_perturbations_applied)
            # for idx in range(number_of_transform_clusters):
            lower_perturb_idx = 0
            for cluster_idx, number_of_perturbs in enumerate(number_of_perturbs_per_cluster):
                if not number_of_perturbs:
                    # All perturbations were culled due to overlap
                    continue
                # Set up the cluster range
                upper_perturb_idx = lower_perturb_idx + number_of_perturbs
                perturb_indices = list(range(lower_perturb_idx, upper_perturb_idx))
                lower_perturb_idx = upper_perturb_idx
                # Grab the cluster range indices
                perturb_indexer = np.isin(weighted_trajectory_df_index, perturb_indices)
                if perturb_indexer.any():
                    if optimize_round == 1:
                        # Slice the cluster range indices by the top hits
                        selected_perturb_indices = \
                            weighted_trajectory_df_index[perturb_indexer][:top_perturb_hits].tolist()
                        # if selected_perturb_indices:
                        # Save the top transform and the top X transforms from each cluster
                        top_transform_cluster_indices.append(selected_perturb_indices[0])
                        perturb_passing_indices.append(selected_perturb_indices)
                    else:  # Just grab the top hit
                        top_transform_cluster_indices.append(weighted_trajectory_df_index[perturb_indexer][0])
                else:  # Update that no perturb_indices present after filter
                    number_of_perturbs_per_cluster[cluster_idx] = 0
                #     perturb_passing_indices.append([])

            if optimize_round == 1:
                nonlocal round1_cluster_shape
                round1_cluster_shape = [len(indices) for indices in perturb_passing_indices]
            elif optimize_round == 2:
                # This is only required if perturb_passing_indices.append([]) is used above
                # Adjust the shape to account for any perturbations that were culled due to overlap
                cluster_start = 0
                for cluster_idx, cluster_shape in enumerate(round1_cluster_shape):
                    cluster_end = cluster_start + cluster_shape
                    for number_of_perturbs in number_of_perturbs_per_cluster[cluster_start:cluster_end]:
                        if not number_of_perturbs:
                            # All perturbations were culled due to overlap
                            cluster_shape -= 1
                    # for indices in perturb_passing_indices[cluster_start:cluster_end]:
                    #     if not indices:
                    #         cluster_shape -= 1
                    # Set the cluster shape with the results of the perturbation trials
                    round1_cluster_shape[cluster_idx] = cluster_shape
                    cluster_start = cluster_end

                number_top_indices = len(top_transform_cluster_indices)  # sum(round1_cluster_shape)
                round1_number_of_clusters = len(round1_cluster_shape)
                logger.info(f'Reducing round 1 expanded cluster search from {number_top_indices} '
                            f'to {round1_number_of_clusters} transformations')
                top_scores_s = weighted_trajectory_df.loc[top_transform_cluster_indices,
                metrics.selection_weight_column]
                # Filter down to the size of the original transforms from the cluster expansion
                top_index_of_cluster = []
                cluster_lower_bound = 0
                for cluster_idx, cluster_shape in enumerate(round1_cluster_shape):
                    if cluster_shape > 0:
                        cluster_upper_bound = cluster_lower_bound + cluster_shape
                        top_cluster_score = top_scores_s.iloc[cluster_lower_bound:cluster_upper_bound].argmax()
                        top_index_of_cluster.append(cluster_lower_bound + top_cluster_score)
                        # Set new lower bound
                        cluster_lower_bound = cluster_upper_bound

                top_transform_cluster_indices = [top_transform_cluster_indices[idx] for idx in top_index_of_cluster]
        else:
            top_transform_cluster_indices = list(range(number_of_transforms))

        # # Finally take from each of the top perturbation "kernels"
        # # With each additional optimize_round, there is exponential increase in the number of transforms
        # # unless there is filtering to take the top
        # # Taking the sqrt (or equivalent function), needs to be incremented for each perturbation increase
        # # so this doesn't get out of hand as the amount grows
        # # For example, iteration 1 gets no sqrt, iteration 2 gets sqrt, 3 gets cube root
        # # root_to_take = 1/iteration
        # top_cluster_root_to_take = 1 / optimize_round
        # # Take the metric at each of the top positions and sort these
        # top_cluster_hits = int((number_of_transform_clusters**top_cluster_root_to_take) + .5)

        # Operation NOTE:
        # During the perturbation selection, this is the sqrt of the number_of_transform_clusters
        # So, from 18 transforms (idx=1), expanded to 81 dof perturbs (idx=2), to get 1458 possible,
        # 1392 didn't clash. 4 top_cluster_hits were selected and 9 transforms from each.
        # This is a lot of sampling for the other 14 that were never chosen. They might've
        # not been discovered without the perturb since the top score came from one of the 81
        # possible perturb transforms
        # if optimize_round > 1:
        #     cluster_divisor = (total_dof_perturbed / (optimize_round-1))
        # else:
        #     cluster_divisor = total_dof_perturbed
        # Divide the clusters by the total applied degrees of freedom
        # top_cluster_hits = int(number_of_transform_clusters/cluster_divisor + .5)
        #
        # # Grab the cluster range indices
        # cluster_representative_indexer = np.isin(weighted_trajectory_s.index, top_transform_cluster_indices)
        # selected_cluster_indices = \
        #     weighted_trajectory_s[cluster_representative_indexer][:top_cluster_hits].index.tolist()
        # # selected_cluster_hits = top_transform_cluster_indices[:top_cluster_overall_hits]
        # # # Use .loc here as we have a list used to index...
        # # selected_cluster_indices = weighted_trajectory_s.loc[selected_cluster_hits].index.tolist()

        # # Grab the top cluster indices in the order of the weighted_trajectory_df
        # cluster_representative_indexer = np.isin(weighted_trajectory_df_index, top_transform_cluster_indices)
        # selected_cluster_indices = weighted_trajectory_df_index[cluster_representative_indexer].tolist()
        # if number_perturbations_applied > 1 and optimize_round == 1:
        #     # For each of the top perturbation clusters, add all the indices picked from the above logic
        #     selected_indices = []
        #     for selected_idx in selected_cluster_indices:
        #         reference_idx = top_transform_cluster_indices.index(selected_idx)
        #         selected_indices.extend(perturb_passing_indices[reference_idx])
        # else:
        #     selected_indices = selected_cluster_indices

        # Grab the top cluster indices in the transformation order
        if number_perturbations_applied > 1 and optimize_round == 1:
            # For each of the top perturbation clusters, add all the indices picked from the above logic
            selected_indices = []
            for cluster_idx, top_index in enumerate(top_transform_cluster_indices):
                selected_indices.extend(perturb_passing_indices[cluster_idx])
        else:
            selected_indices = top_transform_cluster_indices

        # Handle results
        # # Using the current transforms, create a hash to uniquely label them and apply to the indices
        # current_transformation_ids = create_transformation_hash()

        # Filter hits down in the order of the selected indices
        filter_transforms_by_indices(selected_indices)
        # Narrow down the metrics by the selected_indices. If this is the last cycle, they will be written
        poses_df = poses_df.loc[selected_indices]
        residues_df = residues_df.loc[selected_indices]
        # Reset the DataFrame.index
        poses_df.index = residues_df.index = pd.RangeIndex(len(selected_indices))
        # selected_transformation_ids = weighted_trajectory_df_index[selected_indices].tolist()
        selected_transformation_ids = [current_transformation_ids[idx] for idx in selected_indices]

        # Filer down the current_transformation_ids by the filter passing indices and update the index
        # logger.critical(f'Found length of current_transformation_ids: {len(current_transformation_ids)}\n'
        #                 f'weighted_trajectory_df_index: {weighted_trajectory_df_index.tolist()}')
        weighted_current_transforms = current_transformation_ids[weighted_trajectory_df_index]
        weighted_trajectory_df.index = pd.Index(weighted_current_transforms)
        # weighted_trajectory_df.index = \
        #     weighted_trajectory_df.index.map(dict(zip(weighted_trajectory_df_index,
        #                                               pd.Index(current_transformation_ids))))
        # Add the weighted_trajectory_df to the total_results_df to keep global results
        append_total_results(weighted_trajectory_df)

        return weighted_trajectory_df, selected_transformation_ids

        # if metric.dim == 3:
        #     num_transform_dim, per_res_dim, per_aa_dim = metric.shape
        # elif metric.dim == 2:
        #     num_transform_dim, per_res_dim = metric.shape
        # elif metric.dim == 1:
        #     num_transform_dim = metric.shape
        # else:
        #     logger.critical(f"Can't sort transforms by a single scalar")
        #     return 0
        #
        # return metric

    def create_transformation_hash() -> np.ndarray:  # list[int]:
        """Using the currently available transformation parameters for the two Model instances, create the
        transformation hash to describe the orientation of the second model in relation to the first. This hash will be
        unique over the sampling space when discrete differences exceed the TransformHasher.rotation_bin_width and
        .translation_bin_width

        Returns:
            An integer hashing the currently active transforms to distinct orientational offset in the described space
        """
        # Needs to be completed outside of individual naming function as it is stacked
        # transforms = create_transformation_group()
        # input(f'len(transforms[0]["rotation"]): {len(transforms[0]["rotation"])}')
        guide_coordinates_model1, guide_coordinates_model2 = \
            cluster.apply_transform_groups_to_guide_coordinates(*create_transformation_group())
        # input(guide_coordinates_model1[:3])
        rotations = [None for _ in range(len(guide_coordinates_model1))]
        # input(len(rotations))
        translations = rotations.copy()
        # Only turn the outermost array into a list. Keep the guide coordinate 3x3 arrays as arrays for superposition3d
        for transform_idx, (guide_coord2, guide_coord1) in enumerate(
                zip(list(guide_coordinates_model2), list(guide_coordinates_model1))):
            # Reverse the orientation so that the rot, tx indicate the movement of guide_coord1 onto guide_coord2
            rmsd, rot, tx = superposition3d(guide_coord2, guide_coord1)
            rotations[transform_idx] = rot  # rotations.append(rot)
            translations[transform_idx] = tx  # translations.append(tx)

        # logger.debug(f'before rotations[:3]: {rotations[:3]}')
        # logger.debug(f'before translations[:3]: {translations[:3]}')
        hashed_transforms = model_transform_hasher.transforms_to_hash(rotations, translations)
        # rotations, translations = model_transform_hasher.hash_to_transforms(hashed_transforms)
        # logger.debug(f'after rotations[:3]: {rotations[:3]}')
        # logger.debug(f'after translations[:3]: {translations[:3]}')

        return hashed_transforms

    # Collect metrics and filter/weight for each active transform
    def prioritize_transforms_by_selection() -> pd.DataFrame:
        """Using the active transformations, measure the Pose metrics and filter/weight according to defaults/provided
        parameters

        Sets:
            poses_df and residues_df according to the current transformations
        Returns:
            The DataFrame that has been sorted according to the specified filters/weights
        """
        nonlocal poses_df, residues_df
        poses_df, residues_df = collect_dock_metrics()
        # Todo
        #  enable precise metric acquisition
        # dock_metrics = collect_dock_metrics(score_functions)
        weighted_trajectory_df = \
            metrics.prioritize_design_indices(poses_df, filters=job.dock.filter, weights=job.dock.weight,
                                              default_weight=default_weight_metric)
        # Set the metrics_of_interest to the default weighting metric name as well as any weights that are specified
        metrics_of_interest = [metrics.selection_weight_column]
        if job.dock.weight:
            metrics_of_interest += list(job.dock.weight.keys())
        #     weighted_trajectory_df = weighted_trajectory_df.loc[:, list(job.dock.weight.keys())]
        # else:
        #     weighted_trajectory_df = weighted_trajectory_df.loc[:, metrics.selection_weight_column]
        weighted_trajectory_df = weighted_trajectory_df.loc[:, metrics_of_interest]
        # weighted_trajectory_s = metrics.pareto_optimize_trajectories(poses_df, weights=job.dock.weight,
        #                                                              default_sort=default_weight_metric)
        # weighted_trajectory_s is sorted with best transform in index 0, regardless of whether it is ascending or not
        return weighted_trajectory_df

    def append_total_results(additional_trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """Combine existing metrics with the new metrics
        Args:
            additional_trajectory_df: The additional DataFrame to add to the existing global metrics
        Returns:
            The full global metrics DataFrame
        """
        nonlocal total_results_df
        # Add new metrics to existing and keep the newly added if there are overlap
        total_results_df = pd.concat([total_results_df, additional_trajectory_df], axis=0)
        total_results_df = total_results_df[~total_results_df.index.duplicated(keep='last')]
        return total_results_df

    # def calculate_results_for_stopping(target_df: pd.DataFrame, indices: list[int | str]) -> float | pd.Series:
    #     """Given a DataFrame with metrics from a round of optimization, calculate the optimization results to report on whether a stopping condition has been met
    #
    #     Args:
    #         target_df: The DataFrame that resulted from the most recent optimization
    #         indices: The indices which have been selected from the target_df
    #     Returns:
    #         The resulting values from the DataFrame based on the target metrics
    #     """
    #     # Find the value of the new metrics in relation to the old to calculate the result from optimization
    #     if job.dock.weight:
    #         selected_columns = list(job.dock.weight.keys())
    #     else:
    #         selected_columns = metrics.selection_weight_column
    #
    #     selected_metrics_df = target_df.loc[indices, selected_columns]
    #     # other_metrics_df = selected_metrics_df.drop(indices)
    #     # Find the difference between the selected and the other
    #     return selected_metrics_df.mean(axis=0)
    #     # other_metrics_df.mean(axis=1)

    # Initialize output DataFrames which are set in prioritize_transforms_by_selection()
    poses_df = residues_df = pd.DataFrame()
    weighted_trajectory_df = prioritize_transforms_by_selection()
    # # Get selected indices (sorted in original order)
    # selected_indices = weighted_trajectory_df.index.sort_values().tolist()
    # Get selected indices (sorted in weighted_trajectory_df order)
    selected_indices = weighted_trajectory_df.index.tolist()
    # current_transformation_ids = create_transformation_hash() DELETE
    # Filter/sort transforms and metrics by the selected_indices
    filter_transforms_by_indices(selected_indices)
    poses_df = poses_df.loc[selected_indices]
    residues_df = residues_df.loc[selected_indices]
    # # Reorder the transformation identifiers DELETE
    # passing_transform_ids = [current_transformation_ids[idx] for idx in selected_indices] DELETE
    # """The transformation hash indices that are passing in the order of the --dock-weight selection""" DELETE
    passing_transform_ids = create_transformation_hash()
    weighted_trajectory_df.index = pd.Index(passing_transform_ids)
    # -----------------------------------------------------------------------------------------------------------------
    # Below creates perturbations to sampled transformations and scores the resulting Pose
    # -----------------------------------------------------------------------------------------------------------------
    # Set nonlocal perturbation/metric variables that are used in optimize_found_transformations_by_metrics()
    number_of_transforms = number_of_original_transforms = len(full_rotation1)
    number_perturbations_applied = 1
    if job.dock.perturb_dof:
        # Set the weighted_trajectory_df as total_results_df to keep a record of global results
        # total_results_df = poses_df <- this contains all filtered results too
        total_results_df = weighted_trajectory_df
        # Initialize docking score search
        round1_cluster_shape = []
        total_dof_perturbed = 1
        optimize_round = 0
        logger.info(f'Starting optimize with {number_of_transforms} transformations')
        if job.dock.weight:
            selected_columns = list(job.dock.weight.keys())
        else:
            selected_columns = [metrics.selection_weight_column]
        result = total_results_df.loc[:, selected_columns].mean(axis=0)
        # result = calculate_results_for_stopping(total_results_df, passing_transform_ids)
        # threshold = 0.05  # 0.1 <- not convergent # 1 <- too lenient with pareto_optimize_trajectories
        threshold_percent = 0.05
        if isinstance(result, float):
            def result_func(result_): return result_
            thresholds = result * threshold_percent
            last_result = 0.
        else:  # pd.Series
            def result_func(result_): return result_.values
            result = result_func(result)
            thresholds = tuple(result * threshold_percent)
            last_result = tuple(0. for _ in thresholds)

        # The condition sum(translation_perturb_steps) < 0.1 is True after 4 optimization rounds...
        # To ensure that the abs doesn't produce worse values, need to compare results in an unbounded scale
        # (i.e. not between 0-1), which also indicates using a global scale. This way, iteration can tell if they are
        # better. It is a feature of perturb_transformations() that the same transformation is always included in grid
        # search at the moment, so the routine should never arrive at worse scores...
        # Everything below could really be expedited with a Bayseian optimization search strategy
        # while sum(translation_perturb_steps) > 0.1 and all(tuple(abs(last_result - result) > thresholds)):
        # Todo the tuple(abs(last_result - result) > thresholds)) with a float won't convert to an iterable
        while (optimize_round < 2 or all(tuple(abs(last_result - result) > thresholds))) \
                and sum(translation_perturb_steps) > 0.1:
            optimize_round += 1
            logger.info(f'{optimize_found_transformations_by_metrics.__name__} round {optimize_round}')
            last_result = result
            # Perform scoring and a possible iteration of dock perturbation
            weighted_trajectory_df, passing_transform_ids = optimize_found_transformations_by_metrics()
            # IMPORTANT:
            #  passing_transform_ids is in the order of the current transformations
            # # De-duplicate the passing_transform_ids if the optimization has surpassed their bin size
            # passing_transform_ids = utils.remove_duplicates(passing_transform_ids)
            # result = calculate_results_for_stopping(total_results_df, passing_transform_ids)
            # Todo? Could also index the
            #  top_results_df = total_results_df.loc[passing_transform_ids, selected_columns]
            #  result = top_results_df.mean(axis=0)
            top_results_df = weighted_trajectory_df.loc[passing_transform_ids, selected_columns]
            result = result_func(top_results_df.mean(axis=0))
            number_of_transforms = len(full_rotation1)
            logger.info(f'Found {number_of_transforms} transformations after '
                        f'{optimize_found_transformations_by_metrics.__name__} round {optimize_round} '
                        f'with result={result}. last_result={last_result}')
        else:
            if optimize_round == 1:
                number_top_indices = number_of_transforms
                round1_number_of_clusters = len(round1_cluster_shape)
                logger.info(f'Reducing round 1 expanded cluster search from {number_top_indices} '
                            f'to {round1_number_of_clusters} transformations')
                # Reduce the top_transform_cluster_indices to the best remaining in each optimize_round 1 cluster
                # Todo? Could also use
                #  top_scores_s = total_results_df.loc[passing_transform_ids, metrics.selection_weight_column].mean(
                #      axis=0)
                top_scores_s = weighted_trajectory_df.loc[passing_transform_ids, metrics.selection_weight_column]
                # top_indices_of_cluster = []
                # for i in range(round1_number_of_clusters):
                #     # Slice by the expanded cluster amount
                #     cluster_lower_bound = total_dof_perturbed * i
                #     top_cluster_score = top_scores_s.iloc[cluster_lower_bound:
                #                                           cluster_lower_bound + total_dof_perturbed].argmax()
                #     top_indices_of_cluster.append(cluster_lower_bound + top_cluster_score)

                # Filter down to the size of the original transforms from the cluster expansion
                top_indices_of_cluster = []
                cluster_lower_bound = 0
                for cluster_idx, cluster_shape in enumerate(round1_cluster_shape):
                    # This can't be less than 1 here...
                    # if cluster_shape > 0:
                    cluster_upper_bound = cluster_lower_bound + cluster_shape
                    top_cluster_score = top_scores_s.iloc[cluster_lower_bound:cluster_upper_bound].argmax()
                    top_indices_of_cluster.append(cluster_lower_bound + top_cluster_score)
                    # Set new lower bound
                    cluster_lower_bound = cluster_upper_bound

                passing_transform_ids = top_scores_s.iloc[top_indices_of_cluster].index.tolist()
                # Filter hits down
                filter_transforms_by_indices(top_indices_of_cluster)
                # Narrow down the metrics by the selected_indices. If this is the last cycle, they will be written
                poses_df = poses_df.loc[top_indices_of_cluster]
                residues_df = residues_df.loc[top_indices_of_cluster]
                # Reset the DataFrame.index
                poses_df.index = residues_df.index = pd.RangeIndex(len(top_indices_of_cluster))
                number_of_transforms = len(passing_transform_ids)

            # Grab the passing_transform_ids according to the order of provided job.dock.weight
            passing_transform_indexer = np.isin(weighted_trajectory_df.index, passing_transform_ids)
            weighted_transform_ids = weighted_trajectory_df.index[passing_transform_indexer].tolist()
            ordered_indices = [passing_transform_ids.index(_id) for _id in weighted_transform_ids]
            # Reorder hits and metrics by the ordered_indices
            filter_transforms_by_indices(ordered_indices)
            poses_df = poses_df.loc[ordered_indices]
            residues_df = residues_df.loc[ordered_indices]

        logger.info(f'Optimization complete, with {number_of_transforms} final transformations')
    # Set the passing transformation identifiers as the trajectory metrics index
    # These should all be the same order as w/ or w/o optimize_found_transformations_by_metrics() the order of
    # passing_transform_ids is fetched from the order of the selected_indices and each _df is sorted accordingly
    _pose_names = [f'{identifier:d}' for identifier in passing_transform_ids]
    passing_index = pd.Index(_pose_names, name=sql.PoseMetrics.pose_id.name)
    # Deduplicate the indices by keeping the first instance
    # The above sorting ensures that the first instance is the "best"
    deduplicated_indices = ~passing_index.duplicated(keep='first')
    poses_df = poses_df[deduplicated_indices]
    residues_df = residues_df[deduplicated_indices]
    poses_df.index = residues_df.index = passing_index[deduplicated_indices]
    pose_names = poses_df.index.tolist()
    number_of_transforms = len(pose_names)
    filter_transforms_by_indices(np.flatnonzero(deduplicated_indices))

    def terminate(pose: Pose, poses_df_: pd.DataFrame, residues_df_: pd.DataFrame):
        """Finalize any remaining work and return to the caller"""

        # Extract transformation parameters for output
        nonlocal number_of_transforms, pose_names
        # Create PoseJob pose_names using the transformations
        # pose_names = create_transformation_hash()
        # pose_names = [f'{pose_name:d}' for pose_name in passing_transform_ids]

        # Add the PoseJobs to the database
        while True:
            pose_jobs = [PoseJob.from_name(pose_name, project=project, protocol=protocol_name)
                         for pose_name in pose_names]
            session.add_all(pose_jobs)
            try:  # Flush PoseJobs to the current session to generate ids
                session.flush()
            except SQLAlchemyError:  # We already inserted this PoseJob.project/.name
                session.rollback()
                # Find the actual pose_jobs_to_commit and place in session
                # pose_identifiers = [pose_job.new_pose_identifier for pose_job in pose_jobs]
                fetch_jobs_stmt = select(PoseJob).where(PoseJob.project.is_(project)) \
                    .where(PoseJob.name.in_(pose_names))
                existing_pose_jobs = list(session.scalars(fetch_jobs_stmt))
                # Note: Values are sorted by alphanumerical, not numerical
                # ex, design 11 is processed before design 2
                existing_pose_names = {pose_job_.name for pose_job_ in existing_pose_jobs}
                new_pose_names = set(pose_names).difference(existing_pose_names)
                if not new_pose_names:  # No new PoseJobs
                    return existing_pose_jobs
                else:
                    pose_names_ = []
                    for name in new_pose_names:
                        pose_name_index = pose_names.index(name)
                        if pose_name_index != -1:
                            pose_names_.append((pose_name_index, name))
                    # Finally, sort all the names to ensure that the indices from the first pass are accurate
                    # with the new set
                    existing_indices_, pose_names = zip(*sorted(pose_names_, key=lambda name: name[0]))
                    # Select poses_df/residues_df by existing_indices_
                    # poses_df_ = poses_df_.loc[existing_indices_, :]
                    # residues_df_ = residues_df_.loc[existing_indices_, :]
                    poses_df_ = poses_df_.loc[pose_names, :]
                    residues_df_ = residues_df_.loc[pose_names, :]
                    logger.critical(f'Reset the new poses with attributes:\n'
                                    f'\tpose_names={pose_names}\n'
                                    f'\texisting_indices_={existing_indices_}\n'
                                    f'\tposes_df_.index={poses_df_.index.tolist()}\n'
                                    f'\tresidues_df_.index={residues_df_.index.tolist()}\n'
                                    f'')
                    number_of_transforms = len(pose_names)
                    filter_transforms_by_indices(list(existing_indices_))
            else:
                break

        # trajectory = TrajectoryMetadata(poses=pose_jobs, protocol=protocol)
        # session.add(trajectory)

        # Format output data, fix missing
        if job.db:
            pose_ids = [pose.id for pose in pose_jobs]
        else:
            pose_ids = pose_names
        project_str = f'{project}{os.sep}'
        project_pose_names = [f'{project_str}{pose_name}' for pose_name in pose_names]

        def populate_pose_metadata():
            """Add all required PoseJob information to output the created Pose instances for persistent storage"""
            nonlocal poses_df_, residues_df_
            # Save all pose transformation information
            # From here out, the transforms used should be only those of interest for outputting/sequence design
            # filter_transforms_by_indices() <- This is done above

            # Format pose transformations for output
            blank_parameter = list(repeat([None, None, None], number_of_transforms))
            rotations1 = scipy.spatial.transform.Rotation.from_matrix(full_rotation1)
            rotations2 = scipy.spatial.transform.Rotation.from_matrix(full_rotation2)
            # Get all rotations in terms of the degree of rotation along the z-axis
            # rotation_degrees_z1 = rotations1.as_rotvec(degrees=True)[:, -1]
            # rotation_degrees_z2 = rotations2.as_rotvec(degrees=True)[:, -1]
            rotation_degrees_x1, rotation_degrees_y1, rotation_degrees_z1 = \
                zip(*rotations1.as_rotvec(degrees=True).tolist())
            rotation_degrees_x2, rotation_degrees_y2, rotation_degrees_z2 = \
                zip(*rotations2.as_rotvec(degrees=True).tolist())
            # Using the x, y rotation to enforce the degeneracy matrix...
            # Tod0 get the degeneracy_degrees
            # degeneracy_degrees1 = rotations1.as_rotvec(degrees=True)[:, :-1]
            # degeneracy_degrees2 = rotations2.as_rotvec(degrees=True)[:, :-1]
            if sym_entry.is_internal_tx1:
                nonlocal full_int_tx1
                if full_int_tx1.shape[0] > 1:
                    full_int_tx1 = full_int_tx1.squeeze()
                z_heights1 = full_int_tx1[:, -1]
            else:
                z_heights1 = blank_parameter

            if sym_entry.is_internal_tx2:
                nonlocal full_int_tx2
                if full_int_tx2.shape[0] > 1:
                    full_int_tx2 = full_int_tx2.squeeze()
                z_heights2 = full_int_tx2[:, -1]
            else:
                z_heights2 = blank_parameter

            set_mat1_number, set_mat2_number, *_extra = sym_entry.setting_matrices_numbers
            # if sym_entry.unit_cell:
            #     full_uc_dimensions = full_uc_dimensions[passing_symmetric_clash_indices_perturb]
            #     full_ext_tx1 = full_ext_tx1[:]
            #     full_ext_tx2 = full_ext_tx2[:]
            #     full_ext_tx_sum = full_ext_tx2 - full_ext_tx1
            _full_ext_tx1 = blank_parameter if full_ext_tx1 is None else full_ext_tx1.squeeze()
            _full_ext_tx2 = blank_parameter if full_ext_tx2 is None else full_ext_tx2.squeeze()

            for idx, pose_job in enumerate(pose_jobs):
                # Add the next set of coordinates
                update_pose_coords(idx)

                # if number_perturbations_applied > 1:
                add_fragments_to_pose()  # <- here generating fragments fresh

                if job.output:
                    if job.output_trajectory:
                        if idx % 2 == 0:
                            new_pose = pose.copy()
                            # new_pose = pose.models[0]copy()
                            for entity in new_pose.chains[1:]:  # new_pose.entities[1:]:
                                entity.chain_id = 'D'
                                # Todo make more reliable
                                # Todo NEED TO MAKE SymmetricModel copy .entities and .chains correctly!
                            trajectory_models.append_model(new_pose)
                    # Set the ASU, then write to a file
                    pose.set_contacting_asu(distance=cb_distance)
                    try:  # Remove existing cryst_record
                        del pose._cryst_record
                    except AttributeError:
                        pass
                    # pose.uc_dimensions
                    # if sym_entry.unit_cell:  # 2, 3 dimensions
                    #     cryst_record = generate_cryst1_record(full_uc_dimensions[idx], sym_entry.resulting_symmetry)
                    # else:
                    #     cryst_record = None
                    # Todo make a copy of the Pose and add to the PoseJob, then no need for PoseJob.pose = None
                    pose_job.pose = pose
                    putils.make_path(pose_job.pose_directory)
                    pose_job.output_pose(path=pose_job.pose_path)
                    pose_job.source_path = pose_job.pose_path
                    pose_job.pose = None
                    # # Modify the pose_name to get rid of the project
                    # output_pose(pose_name)  # .replace(project_str, ''))
                    # # pose_paths.append(output_pose(pose_name))
                    logger.info(f'OUTPUT POSE: {pose_job.pose_directory}')

                # Update the sql.EntityData with transformations
                external_translation_x1, external_translation_y1, external_translation_z1 = _full_ext_tx1[idx]
                external_translation_x2, external_translation_y2, external_translation_z2 = _full_ext_tx2[idx]
                entity_transformations = [
                    dict(
                        rotation_x=rotation_degrees_x1[idx],
                        rotation_y=rotation_degrees_y1[idx],
                        rotation_z=rotation_degrees_z1[idx],
                        internal_translation_z=z_heights1[idx],
                        setting_matrix=set_mat1_number,
                        external_translation_x=external_translation_x1,
                        external_translation_y=external_translation_y1,
                        external_translation_z=external_translation_z1),
                    dict(
                        rotation_x=rotation_degrees_x2[idx],
                        rotation_y=rotation_degrees_y2[idx],
                        rotation_z=rotation_degrees_z2[idx],
                        internal_translation_z=z_heights2[idx],
                        setting_matrix=set_mat2_number,
                        external_translation_x=external_translation_x2,
                        external_translation_y=external_translation_y2,
                        external_translation_z=external_translation_z2)
                ]

                # Update sql.EntityData, sql.EntityMetrics, sql.EntityTransform
                # pose_id = pose_job.id
                entity_data = []
                entity_transforms = []
                # Todo the number of entities and the number of transformations could be different
                for entity, transform in zip(pose.entities, entity_transformations):
                    transformation = sql.EntityTransform(**transform)
                    entity_transforms.append(transformation)
                    entity_data.append(sql.EntityData(
                        pose=pose_job,
                        meta=entity.metadata,
                        metrics=entity.metrics,
                        transform=transformation)
                    )

                job.current_session.add_all(entity_transforms + entity_data)
                # # Update the PoseJob with sql.EntityData
                # pose_job.entity_data.extend(entity_data)

            if job.db:
                # Update the poses_df_ and residues_df_ index to reflect the new pose_ids
                poses_df_.index = pd.Index(pose_ids, name=sql.PoseMetrics.pose_id.name)
                residues_df_.index = pd.Index(pose_ids, name=sql.PoseResidueMetrics.pose_id.name)
                # Write dataframes to the sql database
                metrics.sql.write_dataframe(job.current_session, poses=poses_df_)
                metrics.sql.write_dataframe(job.current_session, pose_residues=residues_df_)
            else:  # Write to disk
                residues_df_.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)  # ascending=False
                putils.make_path(job.all_scores)
                residue_metrics_csv = os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Residues.csv')
                residues_df_.to_csv(residue_metrics_csv)
                logger.info(f'Wrote residue metrics to {residue_metrics_csv}')
                trajectory_metrics_csv = \
                    os.path.join(job.all_scores, f'{building_blocks}_docked_poses_Trajectories.csv')
                job.dataframe = trajectory_metrics_csv
                poses_df_ = pd.concat([poses_df_], keys=[('dock', 'pose')], axis=1)
                poses_df_.columns = poses_df_.columns.swaplevel(0, 1)
                poses_df_.sort_index(level=2, axis=1, inplace=True, sort_remaining=False)
                poses_df_.sort_index(level=1, axis=1, inplace=True, sort_remaining=False)
                poses_df_.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                poses_df_.to_csv(trajectory_metrics_csv)
                logger.info(f'Wrote trajectory metrics to {trajectory_metrics_csv}')

        # Populate the database with pose information. Has access to nonlocal session
        populate_pose_metadata()

        # Todo 2 modernize with the new SQL database and 6D transform aspirations
        # Cluster by perturbation if perturb_dof:
        if number_perturbations_applied > 1:
            perturbation_identifier = '-p_'
            cluster_type_str = 'ByPerturbation'
            seed_transforms = utils.remove_duplicates(
                [pose_name.split(perturbation_identifier)[0] for pose_name in pose_names])
            cluster_map = {seed_transform: pose_names[idx * number_perturbations_applied:
                                                      (idx + 1) * number_perturbations_applied]
                           for idx, seed_transform in enumerate(seed_transforms)}
            # for pose_name in pose_names:
            #     seed_transform, *perturbation = pose_name.split(perturbation_identifier)
            #     clustered_transformations[seed_transform].append(pose_name)

            # Set the number of poses to cluster equal to the sqrt of the search area
            job.cluster.number = math.sqrt(number_perturbations_applied)
        else:
            cluster_type_str = 'ByTransformation'
            cluster_map = cluster.cluster_by_transformations(*create_transformation_group(),
                                                             values=project_pose_names)
        # Output clustering results
        job.cluster.map = utils.pickle_object(cluster_map,
                                              name=putils.default_clustered_pose_file.format('', cluster_type_str),
                                              out_path=project_dir)
        logger.info(f'Found {len(cluster_map)} unique clusters from {len(pose_names)} pose inputs. '
                    f'Wrote cluster map to {job.cluster.map}')

        # Write trajectory if specified
        if job.output_trajectory:
            if sym_entry.unit_cell:
                logger.warning('No unit cell dimensions applicable to the trajectory file.')

            trajectory_models.write(out_path=os.path.join(project_dir, 'trajectory_oligomeric_models.pdb'),
                                    oligomer=True)

        return pose_jobs

    # Clean up, save data/output results
    # Todo 1 atomize session for concurrent database access
    # with job.db.session(expire_on_commit=False) as session:
    session = job.current_session
    pose_jobs = terminate(pose, poses_df, residues_df)
    session.commit()
    logger.info(f'Total {building_blocks} dock trajectory took {time.time() - frag_dock_time_start:.2f}s')

    return pose_jobs
    # ------------------ TERMINATE DOCKING ------------------------
