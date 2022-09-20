from __future__ import annotations

import os
from abc import ABC
from itertools import repeat
from typing import IO, Sequence, AnyStr

import numpy as np
from sklearn.neighbors._ball_tree import BinaryTree

import structure
from structure.coords import superposition3d, transform_coordinate_sets
from utils.path import frag_text_file
from utils.symmetry import identity_matrix, origin


class GhostFragment:
    _guide_coords: np.ndarray
    """The guide coordinates according to the representative ghost fragment"""
    _representative: 'structure.base.Structure'
    aligned_fragment: Fragment
    """Must support .chain, .number, and .transformation attributes"""
    fragment_db: structure.fragment.db.FragmentDatabase
    # index: int
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
        self.fragment_db = aligned_fragment.fragment_db

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
            aligned chain, aligned residue_number
        """
        return self.aligned_fragment.chain, self.aligned_fragment.number

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
        rotation, translation = self.aligned_fragment.transformation  # self.transformation
        return np.matmul(self._guide_coords, np.transpose(rotation)) + translation

    @property
    def rotation(self) -> np.ndarray:
        """The rotation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.rotation

    @property
    def translation(self) -> np.ndarray:
        """The rotation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.translation

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the aligned Fragment from the Fragment Database

        Returns:
            The rotation (3, 3), the translation (3,)
        """
        return self.aligned_fragment.transformation

    @property
    def representative(self) -> 'structure.base.Structure':
        """Access the Representative GhostFragment Structure"""
        try:
            return self._representative.get_transformed_copy()
        except AttributeError:
            self._representative, _ = self.fragment_db.paired_frags[self.ijk]

        return self._representative.get_transformed_copy()

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None, **kwargs) -> \
            str | None:
        """Write the GhostFragment to a file specified by out_path or with a passed file_handle

        If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        """
        if file_handle:
            file_handle.write(f'{self.representative.get_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.representative.format_header(**kwargs)
            if header is not None and isinstance(header, str):  # used for cryst_record now...
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.representative.get_atom_record(**kwargs)}\n')
            return out_path

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class Fragment:
    _fragment_ca_coords: np.ndarray
    _representative_ca_coords: np.ndarray
    chain: str
    frag_lower_range: int
    frag_upper_range: int
    fragment_db: structure.fragment.db.FragmentDatabase
    ghost_fragments: list | list[GhostFragment] | None
    # guide_coords: np.ndarray | None
    i_type: int | None
    index: int
    number: int
    rmsd_thresh: float = 0.75
    rotation: np.ndarray
    template_coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
    translation: np.ndarray

    def __init__(self, fragment_type: int = None,
                 # guide_coords: np.ndarray = None,
                 # fragment_length: int = 5,
                 fragment_db: structure.fragment.db.FragmentDatabase = None,
                 **kwargs):
        self.ghost_fragments = None
        self.i_type = fragment_type
        # self.guide_coords = guide_coords
        # self.fragment_length = fragment_length
        self.rotation = identity_matrix
        self.translation = origin
        # if fragment_db is not None:
        self.fragment_db = fragment_db
        # self.frag_lower_range, self.frag_upper_range = fragment_db.fragment_range
        super().__init__(**kwargs)
        # may need FragmentBase to clean extras for proper method resolution order (MRO)

    @property
    def fragment_db(self) -> object | None:
        """The secondary structure of the Fragment"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: structure.fragment.db.FragmentDatabase):
        """Set the secondary structure of the Fragment"""
        self._fragment_db = fragment_db
        if fragment_db is not None:
            self.frag_lower_range, self.frag_upper_range = fragment_db.fragment_range
            self.fragment_length = fragment_db.fragment_length

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
            aligned chain, aligned residue_number
        """
        return self.chain, self.number

    @property
    def index(self) -> int:
        """Return the Fragment identifiers that the Fragment was mapped to

        Returns:
            The aligned Residue.index attribute
        """
        return self.index

    @property
    def _representative_coords(self) -> np.ndarray:
        """Return the CA coordinates of the mapped fragment"""
        try:
            return self._representative_ca_coords
        except AttributeError:
            self._representative_ca_coords = self.fragment_db.reps[self.i_type]
            return self._representative_ca_coords

    @_representative_coords.setter
    def _representative_coords(self, coords: np.ndarray):
        self._representative_ca_coords = coords

    @property
    def guide_coords(self) -> np.ndarray:
        """Return the guide coordinates of the mapped Fragment"""
        rotation, translation = self.transformation  # This updates the transformation on the fly if possible
        return np.matmul(self.template_coords, np.transpose(rotation)) + translation
        # return np.matmul(self.template_coords, np.transpose(self.rotation)) + self.translation

    # @guide_coords.setter
    # def guide_coords(self, coords: np.ndarray):
    #     self.guide_coords = coords

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the Fragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        return self.rotation, self.translation
        # return dict(rotation=self.rotation, translation=self.translation)

    # def get_center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul([0.33333, 0.33333, 0.33333], self.guide_coords)
    #     else:
    #         return None

    def find_ghost_fragments(self,
                             # indexed_ghost_fragments: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
                             clash_tree: BinaryTree = None, clash_dist: float = 2.2):
        """Find all the GhostFragments associated with the Fragment

        Args:
            clash_tree: Allows clash prevention during search. Typical use is the backbone and CB atoms of the
                Structure that the Fragment is assigned
            clash_dist: The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        #             indexed_ghost_fragments: The paired fragment database to match to the Fragment instance
        # ghost_i_type = indexed_ghost_fragments.get(self.i_type, None)

        # self.fragment_db.indexed_ghosts : dict[int,
        #                                        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ghost_i_type_arrays = self.fragment_db.indexed_ghosts.get(self.i_type, None)
        if ghost_i_type_arrays is None:
            self.ghost_fragments = []
            return

        stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array = ghost_i_type_arrays
        # transformed_guide_coords = transform_coordinate_sets(stacked_guide_coords, *self.transformation)
        if clash_tree is None:
            viable_indices = None
        else:
            transformed_bb_coords = transform_coordinate_sets(stacked_bb_coords, *self.transformation)
            # with .reshape(), we query on a np.view saving memory
            neighbors = clash_tree.query_radius(transformed_bb_coords.reshape(-1, 3), clash_dist)
            neighbor_counts = np.array([neighbor.size for neighbor in neighbors])
            # reshape to original size then query for existence of any neighbors for each fragment individually
            clashing_indices = neighbor_counts.reshape(transformed_bb_coords.shape[0], -1).any(axis=1)
            viable_indices = ~clashing_indices

        # self.ghost_fragments = [GhostFragment(*info) for info in zip(list(transformed_guide_coords[viable_indices]),
        self.ghost_fragments = [GhostFragment(*info) for info in zip(list(stacked_guide_coords[viable_indices]),
                                                                     *zip(*ijk_types[viable_indices].tolist()),
                                                                     rmsd_array[viable_indices].tolist(), repeat(self))]

    def get_ghost_fragments(self,
                            # indexed_ghost_fragments: dict,
                            **kwargs) -> list | list[GhostFragment]:
        """Find and return all the GhostFragments associated with the Fragment. Optionally check clashing with the
        original structure backbone

        Keyword Args:
            clash_tree: sklearn.neighbors._ball_tree.BinaryTree = None - Allows clash prevention during search.
                Typical use is the backbone and CB coordinates of the Structure that the Fragment is assigned
            clash_dist: float = 2.2 - The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        #         Args:
        #             indexed_ghost_fragments: The paired fragment database to match to the Fragment instance
        self.find_ghost_fragments(**kwargs)
        return self.ghost_fragments

    # def __copy__(self):  # -> Self # Todo python3.11
    #     other = self.__class__.__new__(self.__class__)
    #     other.__dict__ = copy(self.__dict__)
    #     other.__dict__['ghost_fragments'] = copy(self.ghost_fragments)


class MonoFragment(Fragment):
    """Used to represent Fragment information when treated as a continuous Structure Fragment of length fragment_length
    """
    _fragment_coords: np.ndarray  # This is a property in ResidueFragment
    central_residue: 'structure.base.Residue'

    def __init__(self, residues: Sequence['structure.base.Residue'],
                 fragment_db: structure.fragment.db.FragmentDatabase = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.central_residue = residues[int(self.fragment_length/2)]

        if not residues:
            raise ValueError(f'Can\'t find {type(self).__name__} without passing residues with length '
                             f'{self.fragment_length}')
        elif fragment_db is None:
            raise ValueError(f"Can't find {type(self).__name__} without passing fragment_db")
        else:
            try:
                representatives: dict[int, np.ndarray] = fragment_db.reps
            except AttributeError:
                raise TypeError(f'The passed fragment_db is not of the required type "FragmentDatabase"')

        self._fragment_coords = np.array([residue.ca_coords for residue in residues])
        min_rmsd = float('inf')
        for cluster_type, cluster_coords in representatives.items():
            rmsd, rot, tx = superposition3d(self._fragment_coords, cluster_coords)
            if rmsd <= self.rmsd_thresh and rmsd <= min_rmsd:
                self.i_type = cluster_type
                min_rmsd, self.rotation, self.translation = rmsd, rot, tx

        if self.i_type:
            # self.guide_coords = \
            #     np.matmul(self.template_coords, np.transpose(self.rotation)) + self.translation
            self._representative_ca_coords = representatives[self.i_type]

    @property
    def aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the MonoFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.central_residue.chain, self.central_residue.number

    @property
    def chain(self) -> str:
        """The Residue number"""
        return self.central_residue.chain

    @property
    def number(self) -> int:
        """The Residue number"""
        return self.central_residue.number

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
            raise ValueError(f'{type(self).__name__} coords must be shape (3, 3), not {coords.shape}')

    # def get_transformed_copy(self, rotation: list | np.ndarray = None, translation: list | np.ndarray = None,
    #                             rotation2: list | np.ndarray = None, translation2: list | np.ndarray = None) -> \
    #         MonoFragment:
    #     """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space
    #
    #     Transformation proceeds by matrix multiplication with the order of operations as:
    #     rotation, translation, rotation2, translation2
    #
    #     Args:
    #         rotation: The first rotation to apply, expected array shape (3, 3)
    #         translation: The first translation to apply, expected array shape (3,)
    #         rotation2: The second rotation to apply, expected array shape (3, 3)
    #         translation2: The second translation to apply, expected array shape (3,)
    #     Returns:
    #         A transformed copy of the original object
    #     """
    #     if rotation is not None:  # required for np.ndarray or None checks
    #         new_coords = np.matmul(self.guide_coords, np.transpose(rotation))
    #     else:
    #         new_coords = self.guide_coords
    #
    #     if translation is not None:  # required for np.ndarray or None checks
    #         new_coords += np.array(translation)
    #
    #     if rotation2 is not None:  # required for np.ndarray or None checks
    #         new_coords = np.matmul(new_coords, np.transpose(rotation2))
    #
    #     if translation2 is not None:  # required for np.ndarray or None checks
    #         new_coords += np.array(translation2)
    #
    #     new_structure = copy(self)
    #     new_structure.guide_coords = new_coords
    #
    #     return new_structure


class ResidueFragment(Fragment, ABC):
    """Represent Fragment information for a single Residue"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @property
    # def frag_type(self) -> int | None:
    #     """The secondary structure of the Fragment"""
    #     return self.i_type
    #
    # @frag_type.setter
    # def frag_type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.i_type = frag_type

    @property
    def _fragment_coords(self) -> np.ndarray:
        """Return the CA coordinates of the neighboring Residues which specify the ResidueFragment"""
        # try:
        #     return self._fragment_ca_coords
        # except AttributeError:
        return np.array([res.ca_coords for res in self.get_upstream(self.frag_lower_range)
                         + [self] + self.get_downstream(self.frag_upper_range-1)])
        #     return self._fragment_ca_coords

    # @_fragment_coords.setter
    # def _fragment_coords(self, coords: np.ndarray):
    #     self._fragment_ca_coords = coords

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the Fragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        _, self.rotation, self.translation = superposition3d(self._fragment_coords, self._representative_coords)
        return self.rotation, self.translation
        # return dict(rotation=self.rotation, translation=self.translation)

    @property
    def aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the ResidueFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.chain, self.number


def write_frag_match_info_file(ghost_frag: GhostFragment = None, matched_frag: Fragment = None,
                               overlap_error: float = None, match_number: int = None,
                               out_path: AnyStr = os.getcwd(), pose_id: str = None):
    # central_frequencies=None,
    # ghost_residue: Residue = None, matched_residue: Residue = None,

    # if not ghost_frag and not matched_frag and not overlap_error and not match_number:  # TODO
    #     raise DesignError('%s: Missing required information for writing!' % write_frag_match_info_file.__name__)

    with open(os.path.join(out_path, frag_text_file), 'a+') as out_info_file:
        # if is_initial_match:
        if match_number == 1:
            out_info_file.write(f'DOCKED POSE ID: {pose_id}\n\n')
            out_info_file.write('***** ALL FRAGMENT MATCHES *****\n\n')
            # out_info_file.write("***** INITIAL MATCH FROM REPRESENTATIVES OF INITIAL FRAGMENT CLUSTERS *****\n\n")
        cluster_id = 'i{}_j{}_k{}'.format(*ghost_frag.ijk)
        out_info_file.write(f'MATCH {match_number}\n')
        out_info_file.write(f'z-val: {overlap_error}\n')
        out_info_file.write('CENTRAL RESIDUES\noligomer1 ch, resnum: {}, {}\noligomer2 ch, resnum: {}, {}\n'.format(
            *ghost_frag.aligned_chain_and_residue, *matched_frag.aligned_chain_and_residue))
        # Todo
        #  out_info_file.write('oligomer1 ch, resnum: %s, %d\n' % (ghost_residue.chain, ghost_residue.residue))
        #  out_info_file.write('oligomer2 ch, resnum: %s, %d\n' % (matched_residue.chain, matched_residue.residue))
        out_info_file.write('FRAGMENT CLUSTER\n')
        out_info_file.write(f'id: {cluster_id}\n')
        out_info_file.write(f'mean rmsd: {ghost_frag.rmsd}\n')
        out_info_file.write(f'aligned rep: int_frag_{cluster_id}_{match_number}.pdb\n')
        # out_info_file.write(f'central res pair freqs:\n{central_frequencies}\n\n')

        # if is_initial_match:
        #     out_info_file.write("***** ALL MATCH(ES) FROM REPRESENTATIVES OF ALL FRAGMENT CLUSTERS *****\n\n")
