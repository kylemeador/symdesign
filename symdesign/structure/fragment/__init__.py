from __future__ import annotations

import abc
import logging
import os
from abc import ABC
from collections.abc import Iterable
from itertools import repeat
from typing import IO, Sequence, AnyStr

import numpy as np
from sklearn.neighbors._ball_tree import BinaryTree, BallTree

from symdesign import utils, structure
from . import db, info, metrics
from ..coordinates import guide_superposition, superposition3d, transform_coordinate_sets

# Globals
logger = logging.getLogger(__name__)


class GhostFragment:
    _guide_coords: np.ndarray
    """The guide coordinates according to the representative ghost fragment"""
    _representative: 'structure.base.Structure'
    aligned_fragment: Fragment
    """The Fragment instance that this GhostFragment is aligned too. 
    Must support .chain_id, .number, .index, .rotation, .translation, and .transformation attributes
    """
    fragment_db: db.FragmentDatabase
    i_type: int
    j_type: int
    k_type: int
    rmsd: float
    """The deviation from the representative ghost fragment"""

    def __init__(self, guide_coords: np.ndarray, i_type: int, j_type: int, k_type: int, ijk_rmsd: float,
                 aligned_fragment: Fragment):
        self._guide_coords = guide_coords
        self.i_type = i_type
        self.j_type = self.frag_type = j_type
        self.k_type = k_type
        self.rmsd = ijk_rmsd
        self.aligned_fragment = aligned_fragment
        # Assign both attributes for API compatibility with Fragment
        self.fragment_db = self._fragment_db = aligned_fragment.fragment_db

    @property
    def type(self) -> int:
        """The secondary structure of the Fragment"""
        return self.j_type

    # @type.setter
    # def type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.j_type = frag_type

    # @property
    # def frag_type(self) -> int:
    #     """The secondary structure of the Fragment"""
    #     return self.j_type

    # @frag_type.setter
    # def frag_type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.j_type = frag_type

    @property
    def ijk(self) -> tuple[int, int, int]:
        """The Fragment cluster index information

        Returns:
            I cluster index, J cluster index, K cluster index
        """
        return self.i_type, self.j_type, self.k_type

    @property
    def aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the GhostFragment was mapped to

        Returns:
            aligned chain_id, aligned residue_number
        """
        return self.aligned_fragment.chain_id, self.aligned_fragment.number

    @property
    def number(self) -> int:
        """The Residue number of the aligned Fragment"""
        return self.aligned_fragment.number

    @property
    def index(self) -> int:
        """The Residue index of the aligned Fragment"""
        return self.aligned_fragment.index

    @property
    def guide_coords(self) -> np.ndarray:
        """Return the guide coordinates of the GhostFragment"""
        rotation, translation = self.aligned_fragment.transformation  # Updates the transformation on the fly
        return np.matmul(self._guide_coords, np.transpose(rotation)) + translation

    @property
    def rotation(self) -> np.ndarray:
        """The rotation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.rotation

    @property
    def translation(self) -> np.ndarray:
        """The translation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.translation

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the aligned Fragment from the Fragment Database

        Returns:
            The rotation (3, 3) and the translation (3,)
        """
        return self.aligned_fragment.transformation

    @property
    def representative(self) -> 'structure.base.Structure':
        """Access the Representative GhostFragment Structure"""
        try:
            return self._representative.get_transformed_copy()
        except AttributeError:
            self._representative, _ = self._fragment_db.paired_frags[self.ijk]

        return self._representative.get_transformed_copy()

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None, **kwargs) \
            -> AnyStr | None:
        """Write the GhostFragment to a file specified by out_path or with a passed file_handle

        If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        Keyword Args:
            chain_id: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The name of the written file if out_path is used
        """
        if file_handle:
            file_handle.write(f'{self.representative.get_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.representative.format_header(**kwargs)
            if header is not None and isinstance(header, str):  # Used for cryst_record now...
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.representative.get_atom_record(**kwargs)}\n')
            return out_path

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class Fragment(ABC):
    _fragment_ca_coords: np.ndarray
    _fragment_coords: np.ndarray | None
    """Holds coordinates (currently backbone) that represent the fragment during spatial transformation"""
    _fragment_db: db.FragmentDatabase | None
    _guide_coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
    # frag_lower_range: int
    # frag_upper_range: int
    ghost_fragments: list | list[GhostFragment] | None
    i_type: int | None
    rmsd_thresh: float = 0.75
    rotation: np.ndarray
    translation: np.ndarray

    def __init__(self, fragment_type: int = None, fragment_db: db.FragmentDatabase = None, **kwargs):
        self._fragment_coords = None
        self.ghost_fragments = None
        self.i_type = fragment_type
        self.rotation = utils.symmetry.identity_matrix
        self.translation = utils.symmetry.origin
        self._fragment_db = fragment_db

        super().__init__(**kwargs)  # Fragment
        # May need a FragmentBase to clean extra kwargs for proper method resolution order

    # These property getter and setter exist so that Residue instances can be initialized w/o FragmentDatabase
    # and then provided this argument upon fragment assignment
    @property
    def fragment_db(self) -> db.FragmentDatabase:
        """The FragmentDatabase that the Fragment was created from"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: db.FragmentDatabase):
        """Set the Fragment instances FragmentDatabase to assist with creation and manipulation"""
        self._fragment_db = fragment_db

    @property
    def frag_type(self) -> int | None:
        """The secondary structure of the Fragment"""
        return self.i_type

    @frag_type.setter
    def frag_type(self, frag_type: int):
        """Set the secondary structure of the Fragment"""
        self.i_type = frag_type

    @property
    def aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            aligned chain_id, aligned residue_number
        """
        return self.chain_id, self.number

    @property
    @abc.abstractmethod
    def chain_id(self) -> str:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            The aligned Residue.chain_id attribute
        """

    @property
    @abc.abstractmethod
    def number(self) -> int:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            The aligned Residue.number attribute
        """

    @property
    @abc.abstractmethod
    def index(self) -> int:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            The aligned Residue.index attribute
        """

    @property
    def guide_coords(self) -> np.ndarray:
        """Return the guide coordinates of the mapped Fragment"""
        rotation, translation = self.transformation  # Updates the transformation on the fly
        return np.matmul(self._guide_coords, np.transpose(rotation)) + translation

    # @guide_coords.setter
    # def guide_coords(self, coords: np.ndarray):
    #     self.guide_coords = coords

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the Fragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        return self.rotation, self.translation
        # return dict(rotation=self.rotation, translation=self.translation)

    # def center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul([0.33333, 0.33333, 0.33333], self.guide_coords)
    #     else:
    #         return None

    def find_ghost_fragments(self, clash_tree: BinaryTree = None, clash_dist: float = 2.1):
        """Find all the GhostFragments associated with the Fragment

        Args:
            clash_tree: Allows clash prevention during search. Typical use is the backbone and CB atoms of the
                Structure that the Fragment is assigned
            clash_dist: The distance to check for backbone clashes
        """
        ghost_i_type_arrays = self._fragment_db.indexed_ghosts.get(self.i_type, None)
        if ghost_i_type_arrays is None:
            self.ghost_fragments = []
            return

        stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array = ghost_i_type_arrays
        # No need to transform stacked_guide_coords as these will be transformed upon .guide_coords access
        if clash_tree is None:
            # Ensure we slice by nothing, as None alone creates a new axis
            viable_indices = slice(None)
        else:
            # Ensure that the backbone coords are transformed to the Fragment reference frame
            transformed_bb_coords = transform_coordinate_sets(stacked_bb_coords, *self.transformation)
            # with .reshape(), we query on a np.view saving memory
            neighbors = clash_tree.query_radius(transformed_bb_coords.reshape(-1, 3), clash_dist)
            neighbor_counts = np.array([neighbor.size for neighbor in neighbors.tolist()])
            # reshape to original size then query for existence of any neighbors for each fragment individually
            clashing_indices = neighbor_counts.reshape(len(transformed_bb_coords), -1).any(axis=1)
            viable_indices = ~clashing_indices

        # self.ghost_fragments = [GhostFragment(*info) for info in zip(list(transformed_guide_coords[viable_indices]),
        self.ghost_fragments = [
            GhostFragment(*info_)
            for info_ in zip(
                list(stacked_guide_coords[viable_indices]), *zip(*ijk_types[viable_indices].tolist()),
                rmsd_array[viable_indices].tolist(), repeat(self))
        ]

    def get_ghost_fragments(self, **kwargs) -> list | list[GhostFragment]:
        """Retrieve the GhostFragments associated with the Fragment. Will generate if none are available, otherwise,
        will return the already generated instances.

        Optionally, check clashing with the original structure backbone by passing clash_tree

        Keyword Args:
            clash_tree: sklearn.neighbors._ball_tree.BinaryTree = None - Allows clash prevention during search.
                Typical use is the backbone and CB coordinates of the Structure that the Fragment is assigned
            clash_dist: float = 2.1 - The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        #         Args:
        #             indexed_ghost_fragments: The paired fragment database to match to the Fragment instance
        # self.find_ghost_fragments(**kwargs)
        if self.ghost_fragments is None:
            self.find_ghost_fragments(**kwargs)
        else:
            # This routine is necessary when the ghost_fragments are already generated on a residue,
            # but that residue is copied.
            # Todo Perhaps, this routine should be called by the copy function...
            logger.debug('Using previously generated ghost fragments. Updating their .aligned_fragment attribute')
            for ghost in self.ghost_fragments:
                ghost.aligned_fragment = self

        return self.ghost_fragments

    # def __copy__(self):  # Todo -> Self: in python 3.11
    #     other = self.__class__.__new__(self.__class__)
    #     other.__dict__ = copy(self.__dict__)
    #     other.__dict__['ghost_fragments'] = copy(self.ghost_fragments)


class MonoFragment(Fragment):
    """Used to represent Fragment information when treated as a continuous Fragment of length fragment_length"""
    central_residue: 'structure.base.Residue'

    def __init__(self, residues: Sequence['structure.base.Residue'], **kwargs):
        """

        Args:
            residues: The Residue instances which comprise the MonoFragment
            **kwargs:
        """
        super().__init__(**kwargs)  # MonoFragment

        try:
            fragment_length = self.fragment_db.fragment_length
        except AttributeError:  # self.fragment_db is None
            raise ValueError(
                f"Can't construct {self.__class__.__name__} without passing 'fragment_db'")

        if not residues:
            raise ValueError(
                f"Can't find {self.__class__.__name__} without passing {fragment_length} Residue instances")
        self.central_residue = residues[int(fragment_length / 2)]

        try:
            fragment_representatives = self.fragment_db.representatives
        except AttributeError:
            raise TypeError(
                f"The 'fragment_db' is not of the required type '{db.FragmentDatabase.__name__}'")

        fragment_ca_coords = np.array([residue.ca_coords for residue in residues])
        min_rmsd = float('inf')

        for fragment_type, representative in fragment_representatives.items():
            rmsd, rot, tx = superposition3d(fragment_ca_coords, representative.ca_coords)
            if rmsd <= self.rmsd_thresh and rmsd <= min_rmsd:
                self.i_type = fragment_type
                min_rmsd, self.rotation, self.translation = rmsd, rot, tx

        if self.i_type:
            # self.guide_coords = \
            #     np.matmul(self.template_coords, np.transpose(self.rotation)) + self.translation
            self._fragment_coords = fragment_representatives[self.i_type].backbone_coords

    @property
    def chain_id(self) -> str:
        """The Residue chainID"""
        return self.central_residue.chain_id

    @property
    def number(self) -> int:
        """The Residue number"""
        return self.central_residue.number

    @property
    def index(self) -> int:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            The aligned Residue.index attribute
        """
        return self.central_residue.index

    # Methods below make compatible with Pose symmetry operations
    @property
    def coords(self) -> np.ndarray:
        return self.guide_coords

    @coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        if coords.shape == (3, 3):
            # Move the transformation accordingly
            _, self.rotation, self.translation = superposition3d(coords, self.template_coords)
            # self.guide_coords = coords
        else:
            raise ValueError(
                f'{self.__class__.__name__} coords.shape ({coords.shape}) != (3, 3)')


class ResidueFragment(Fragment, ABC):
    """Represent Fragment information for a single Residue"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # ResidueFragment

    @property
    @abc.abstractmethod
    def backbone_coords(self) -> np.ndarray:
        """"""

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the ResidueFragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        # *Slower than the below functions
        # rotation, *_ = Rotation.align_vectors(self.backbone_coords, self._fragment_coords)
        # self.rotation = rotation.as_matrix()
        # self.translation = self.backbone_coords.mean(axis=0)\
        #     - np.matmul(self.rotation, self._fragment_coords.mean(axis=0))
        # *Slower
        # _, self.rotation, self.translation = \
        #     superposition3d(self.backbone_coords, self._fragment_coords)
        #     superposition3d(self._fragment_coords, self._fragment_coords)
        self.rotation, self.translation = guide_superposition(self.backbone_coords, self._fragment_coords)
        return self.rotation, self.translation


def find_fragment_overlap(fragments1: Iterable[Fragment], fragments2: Sequence[Fragment],
                          clash_coords: np.ndarray = None, min_match_value: float = 2.,  # .2,
                          **kwargs) -> list[tuple[GhostFragment, Fragment, float]]:
    #           entity1, entity2, entity1_interface_residue_numbers, entity2_interface_residue_numbers, max_z_value=2):
    """From two sets of Residues, score the fragment overlap according to Nanohedra's fragment matching

    Args:
        fragments1: The Fragment instances that will be used to search for GhostFragment instances
        fragments2: The Fragment instances to pair against fragments1 GhostFragment instances
        clash_coords: The coordinates to use for checking for GhostFragment clashes
        min_match_value: The minimum value which constitutes an acceptable fragment z_score
    Returns:
        The GhostFragment, Fragment pairs, along with their match score
    """
    # min_match_value: The minimum value which constitutes an acceptable fragment match score
    # 0.2 with match score, 2 with z_score
    # Todo memoize this variable into a function default... The load time is kinda significant and shouldn't be used
    #  until needed. Getting the factory everytime is a small overhead that is really unnecessary. Perhaps this function
    #  should be refactored to structure.fragment.db or something and imported upon usage...

    # logger.debug('Starting Ghost Frag Lookup')
    if clash_coords is not None:
        clash_tree = BallTree(clash_coords)
    else:
        clash_tree = None

    ghost_frags1: list[GhostFragment] = []
    for fragment in fragments1:
        ghost_frags1.extend(fragment.get_ghost_fragments(clash_tree=clash_tree))

    logger.debug(f'Residues 1 has {len(ghost_frags1)} ghost fragments')

    # Get fragment guide coordinates
    residue1_ghost_guide_coords = np.array([ghost_frag.guide_coords for ghost_frag in ghost_frags1])
    residue2_guide_coords = np.array([fragment.guide_coords for fragment in fragments2])
    # interface_surf_frag_guide_coords = np.array([residue.guide_coords for residue in interface_residues2])

    # Check for matching Euler angles
    # Todo create a stand alone function
    # logger.debug('Starting Euler Lookup')
    euler_lookup = fragments2[0].fragment_db.euler_lookup

    overlapping_ghost_indices, overlapping_frag_indices = \
        euler_lookup.check_lookup_table(residue1_ghost_guide_coords, residue2_guide_coords)
    # logger.debug('Finished Euler Lookup')
    logger.debug(f'Found {len(overlapping_ghost_indices)} overlapping fragments in the same Euler rotational space')
    # filter array by matching type for surface (i) and ghost (j) frags
    ghost_type_array = np.array([ghost_frags1[idx].frag_type for idx in overlapping_ghost_indices.tolist()])
    mono_type_array = np.array([fragments2[idx].frag_type for idx in overlapping_frag_indices.tolist()])
    ij_type_match = mono_type_array == ghost_type_array

    passing_ghost_indices = overlapping_ghost_indices[ij_type_match]
    passing_frag_indices = overlapping_frag_indices[ij_type_match]
    logger.debug(f'Found {len(passing_ghost_indices)} overlapping fragments in the same i/j type')

    passing_ghost_coords = residue1_ghost_guide_coords[passing_ghost_indices]
    passing_frag_coords = residue2_guide_coords[passing_frag_indices]
    # # Todo keep without euler_lookup?
    # ghost_type_array = np.array([ghost_frag.frag_type for ghost_frag in ghost_frags1])
    # mono_type_array = np.array([residue.frag_type for residue in fragments2])
    # # Using only ij_type_match, no euler_lookup
    # int_ghost_shape = len(ghost_frags1)
    # int_surf_shape = len(fragments2)
    # # maximum_number_of_pairs = int_ghost_shape*int_surf_shape
    # ghost_indices_repeated = np.repeat(ghost_type_array, int_surf_shape)
    # surf_indices_tiled = np.tile(mono_type_array, int_ghost_shape)
    # # ij_type_match = ij_type_match_lookup_table[ghost_indices_repeated, surf_indices_tiled]
    # # ij_type_match = np.where(ghost_indices_repeated == surf_indices_tiled, True, False)
    # # ij_type_match = ghost_indices_repeated == surf_indices_tiled
    # ij_type_match_lookup_table = (ghost_indices_repeated == surf_indices_tiled).reshape(int_ghost_shape, -1)
    # ij_type_match = ij_type_match_lookup_table[ghost_indices_repeated, surf_indices_tiled]
    # # possible_fragments_pairs = ghost_indices_repeated.shape[0]
    # passing_ghost_indices = ghost_indices_repeated[ij_type_match]
    # passing_surf_indices = surf_indices_tiled[ij_type_match]
    # # passing_ghost_coords = residue1_ghost_guide_coords[ij_type_match]
    # # passing_frag_coords = residue2_guide_coords[ij_type_match]
    # passing_ghost_coords = residue1_ghost_guide_coords[passing_ghost_indices]
    # passing_frag_coords = residue2_guide_coords[passing_surf_indices]
    # Precalculate the reference_rmsds for each ghost fragment
    reference_rmsds = np.array([ghost_frags1[ghost_idx].rmsd for ghost_idx in passing_ghost_indices.tolist()])
    # # Todo keep without euler_lookup?
    # reference_rmsds = np.array([ghost_frag.rmsd for ghost_frag in ghost_frags1])[passing_ghost_indices]

    # logger.debug('Calculating passing fragment overlaps by RMSD')
    # all_fragment_match = metrics.calculate_match(passing_ghost_coords, passing_frag_coords, reference_rmsds)
    # passing_overlaps_indices = np.flatnonzero(all_fragment_match > min_match_value)
    all_fragment_z_score = metrics.rmsd_z_score(passing_ghost_coords, passing_frag_coords, reference_rmsds)
    passing_overlaps_indices = np.flatnonzero(all_fragment_z_score < min_match_value)
    # logger.debug('Finished calculating fragment overlaps')
    # logger.debug(f'Found {len(passing_overlaps_indices)} overlapping fragments over the {min_match_value} threshold')
    logger.debug(f'Found {len(passing_overlaps_indices)} overlapping fragments under the {min_match_value} threshold')

    # interface_ghostfrags = [ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlap_indices].tolist()]
    # interface_monofrags2 = [fragments2[idx] for idx in passing_surf_indices[passing_overlap_indices].tolist()]
    # passing_z_values = all_fragment_overlap[passing_overlap_indices]
    # match_scores = utils.match_score_from_z_value(all_fragment_overlap[passing_overlap_indices])

    return list(zip([ghost_frags1[idx] for idx in passing_ghost_indices[passing_overlaps_indices].tolist()],
                    [fragments2[idx] for idx in passing_frag_indices[passing_overlaps_indices].tolist()],
                    # all_fragment_match[passing_overlaps_indices].tolist()))
                    metrics.match_score_from_z_value(all_fragment_z_score[passing_overlaps_indices]).tolist()))
    #
    # # Todo keep without euler_lookup?
    # return list(zip([ghost_frags1[idx] for idx in passing_overlaps_indices.tolist()],
    #                 [fragments2[idx] for idx in passing_overlaps_indices.tolist()],
    #                 all_fragment_match[passing_overlaps_indices].tolist()))


def create_fragment_info_from_pairs(
    ghostfrag_frag_pairs: list[tuple[GhostFragment, Fragment, float]]
) -> list[db.FragmentInfo]:
    """From a ghost fragment/surface fragment pair and corresponding match score, return the pertinent interface
    information

    Args:
        ghostfrag_frag_pairs: Observed ghost and surface fragment overlaps and their match score
    Returns:
        The formatted fragment information for each pair
            {'mapped': int, 'paired': int, 'match': float, 'cluster': tuple(int, int, int)}
    """
    fragment_matches = [db.FragmentInfo(mapped=ghost_frag.index, paired=surf_frag.index,
                                        match=match_score, cluster=ghost_frag.ijk)
                        for ghost_frag, surf_frag, match_score in ghostfrag_frag_pairs]

    logger.debug(f'Fragments for Entity1 found at indices: {[frag.mapped for frag in fragment_matches]}')
    logger.debug(f'Fragments for Entity2 found at indices: {[frag.paired for frag in fragment_matches]}')

    return fragment_matches


def write_frag_match_info_file(ghost_frag: GhostFragment = None, matched_frag: Fragment = None,
                               overlap_error: float = None, match_number: int = None,
                               out_path: AnyStr = os.getcwd(), pose_identifier: str = None):
    # central_frequencies=None,
    # ghost_residue: Residue = None, matched_residue: Residue = None,

    # if not ghost_frag and not matched_frag and not overlap_error and not match_number:
    #     raise DesignError(
    #       f'{write_frag_match_info_file.__name__}: Missing required information')

    with open(os.path.join(out_path, utils.path.frag_text_file), 'a+') as out_info_file:
        if match_number == 1:
            out_info_file.write(f'DOCKED POSE ID: {pose_identifier}\n\n')
            out_info_file.write('***** ALL FRAGMENT MATCHES *****\n\n')
        cluster_id = 'i{}_j{}_k{}'.format(*ghost_frag.ijk)
        out_info_file.write(f'MATCH {match_number}\n')
        out_info_file.write(f'z-val: {overlap_error}\n')
        out_info_file.write('CENTRAL RESIDUES\ncomponent1 ch, resnum: {}, {}\ncomponent2 ch, resnum: {}, {}\n'.format(
            *ghost_frag.aligned_chain_and_residue, *matched_frag.aligned_chain_and_residue))
        # Todo
        #  out_info_file.write('component1 ch, resnum: %s, %d\n' % (ghost_residue.chain_id, ghost_residue.residue))
        #  out_info_file.write('component2 ch, resnum: %s, %d\n' % (matched_residue.chain_id, matched_residue.residue))
        out_info_file.write('FRAGMENT CLUSTER\n')
        out_info_file.write(f'id: {cluster_id}\n')
        out_info_file.write(f'mean rmsd: {ghost_frag.rmsd}\n')
        out_info_file.write(f'aligned rep: int_frag_{cluster_id}_{match_number}.pdb\n')
        # out_info_file.write(f'central res pair freqs:\n{central_frequencies}\n\n')
