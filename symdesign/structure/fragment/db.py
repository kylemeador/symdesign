from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Annotated, Literal, get_args, Type, Union

import numpy as np
import scipy.spatial.transform

from . import info
from symdesign import utils, structure
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
alignment_types_literal = Literal['mapped', 'paired']
alignment_types: tuple[str, ...] = get_args(alignment_types_literal)
fragment_info_keys = Literal[alignment_types_literal, 'match', 'cluster']
fragment_info_type = Type[dict[fragment_info_keys, Union[int, float, tuple[int, int, int]]]]
RELOAD_DB = 123


class Representative:
    backbone_coords: np.ndarray
    ca_coords: np.ndarray
    register: tuple[str, ...] = ('backbone_coords', 'ca_coords')

    def __init__(self, struct: 'structure.base.Structure', fragment_db: FragmentDatabase):
        for item in self.register:
            self.__setattr__(item, getattr(struct, item))

        for idx, index in enumerate(range(*fragment_db.fragment_range)):
            if index == 0:
                representative_residue_idx = list(range(fragment_db.fragment_length))[idx]
                logger.info(f'Found the representative residue index {representative_residue_idx}')
                self.backbone_coords = struct.residues[representative_residue_idx].backbone_coords
                break
        else:
            raise ValueError(f"Couldn't get the representative residue index upon initialization")


class FragmentDatabase(info.FragmentInfo):
    cluster_representatives_path: str
    cluster_info_path: str
    euler_lookup: EulerLookup
    indexed_ghosts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | dict
    # dict[int, tuple[3x3, 1x3, tuple[int, int, int], float]]
    # monofrag_representatives_path: str
    paired_frags: dict[tuple[int, int, int], tuple['structure.model.Model', str]]
    reps: dict[int, np.ndarray]

    def __init__(self, **kwargs):  # init_db: bool = True, fragment_length: int = 5
        super().__init__(**kwargs)
        if self.source == putils.biological_interfaces:  # Todo parameterize
            self.cluster_representatives_path = putils.intfrag_cluster_rep_dirpath
            self.monofrag_representatives_path = putils.monofrag_cluster_rep_dirpath

        self.representatives = {}
        self.paired_frags = {}
        self.indexed_ghosts = {}
        # self.euler_lookup = euler_factory()

        # if init_db:
        #     logger.info(f'Initializing {self.source} FragmentDatabase from disk. This may take awhile...')
        #     # self.get_monofrag_cluster_rep_dict()
        #     self.representatives = {int(os.path.splitext(file)[0]):
        #                             structure.base.Structure.from_file(
        #                                 os.path.join(root, file), entities=False, log=None).ca_coords
        #                             for root, dirs, files in os.walk(self.monofrag_representatives_path) for file in files}
        #     self.paired_frags = load_paired_fragment_representatives(self.cluster_representatives_path)
        #     self.load_cluster_info()  # Using my generated data instead of Josh's for future compatibility and size
        #     # self.load_cluster_info_from_text()
        #     self._index_ghosts()
        # else:
        #     self.representatives = {}
        #     self.paired_frags = {}
        #     self.indexed_ghosts = {}

    def _index_ghosts(self):
        """From a loaded fragment database, precompute all required data into numpy arrays to populate Ghost Fragments

        keeping each data type as numpy array allows facile indexing for allowed ghosts if a clash test is performed
        """
        ijk_types = list(sorted(self.paired_frags.keys()))
        stacked_bb_coords, stacked_guide_coords = [], []
        for ijk in ijk_types:
            frag_model, frag_paired_chain = self.paired_frags[ijk]
            # Look up the partner coords by using stored frag_paired_chain
            stacked_bb_coords.append(frag_model.chain(frag_paired_chain).backbone_coords)
            # Todo store these as a numpy array instead of as a chain
            stacked_guide_coords.append(frag_model.chain('9').coords)
        logger.debug(f'Last representative file: {frag_model}, Paired chain: {frag_paired_chain}')

        stacked_bb_coords = np.array(stacked_bb_coords)
        stacked_guide_coords = np.array(stacked_guide_coords)
        logger.debug(f'stacked_bb_coords {stacked_bb_coords.shape}')
        logger.debug(f'stacked_guide_coords {stacked_guide_coords.shape}')

        stacked_rmsds = np.array([self.info[ijk].rmsd for ijk in ijk_types])
        # Todo ensure rmsd rounded correct upon creation
        stacked_rmsds = np.where(stacked_rmsds == 0, 0.0001, stacked_rmsds)
        logger.debug(f'stacked_rmsds {stacked_rmsds.shape}')

        # Split data into separate i_types
        prior_idx = 0
        prior_i_type, prior_j_type, prior_k_type = ijk_types[prior_idx]
        for idx, (i_type, j_type, k_type) in enumerate(ijk_types):
            if i_type != prior_i_type:
                self.indexed_ghosts[prior_i_type] = \
                    (stacked_bb_coords[prior_idx:idx], stacked_guide_coords[prior_idx:idx],
                     np.array(ijk_types[prior_idx:idx]), stacked_rmsds[prior_idx:idx])
                prior_i_type = i_type
                prior_idx = idx

        # One more time for the final index
        self.indexed_ghosts[prior_i_type] = \
            (stacked_bb_coords[prior_idx:idx], stacked_guide_coords[prior_idx:idx],
             np.array(ijk_types[prior_idx:idx]), stacked_rmsds[prior_idx:idx])

    # Todo
    #  An ideal measure of the central importance would weight central fragment observations to new center
    #  fragments higher than repeated observations. Say mapped residue 45 to paired 201. If there are two
    #  observations of this pair, just different ijk indices, then these would take the additive form:
    #  SUMi->n 1/i*2
    #  If the observation went from residue 45 to paired residue 204, they constitute separate, but less important
    #  observations as these are part of the same SS and it is implied that if 201 contacts, 204 (and 198) are
    #  possible. In the case where the interaction goes from a residue to a different secondary structure, then
    #  this is the most important, say residue 45 to residue 322 on a separate helix. This indicates that
    #  residue 45 is ideally placed to interact with a separate SS and add them separately to the score
    #  |
    #  How the data structure to build this relationship looks is likely a graph, which has significant overlap
    #  to work I did on the consensus sequence higher order relationships in 8/20. I may find some ground in
    #  the middle of these two ideas where the graph theory could take hold and a more useful scoring metric
    #  could emerge, also possibly with a differentiable equation I could relate pose transformations to
    #  favorable fragments found in the interface.
    def calculate_match_metrics(self, fragment_matches: list[fragment_info_type]) -> dict:
        """Return the various metrics calculated by overlapping fragments at the interface of two proteins

        Args:
            fragment_matches: [{'mapped': entity1_index, 'paired': entity2_index,
                                'cluster': tuple(int, int, int), 'match': score_term}, ...]
        Returns:
            {'mapped': {'center': {'indices': (set[int]), 'score': (float),},
                        'total': {'indices': (set[int]), 'score': (float),},
                        'match_scores': {residue number(int): (list[score (float)]), ...},
                        'index_count': {index (int): count (int), ...},
                        'multiple_ratio': (float)}
             'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
             'total':  {'center': {'score': , 'number': },
                        'total': {'score': , 'number': },
                        'index_count': , 'multiple_ratio': , 'observations': (int)}
             }
        """
        if not fragment_matches:
            return {}

        fragment_i_index_count_d = {frag_idx: 0 for frag_idx in range(1, self.fragment_length + 1)}
        fragment_j_index_count_d = fragment_i_index_count_d.copy()
        total_fragment_content = fragment_i_index_count_d.copy()

        # entity1_center_match_scores, entity2_center_match_scores = {}, {}
        # entity1_match_scores, entity2_match_scores = {}, {}
        entity1_center_match_scores = defaultdict(list)
        entity2_center_match_scores = defaultdict(list)
        entity1_match_scores = defaultdict(list)
        entity2_match_scores = defaultdict(list)
        total_observations = len(fragment_matches)
        separated_fragment_metrics = {}
        # separated_fragment_metrics = {'mapped': {'center': {'indices': set()}, 'total': {'indices': set()}},
        #                               'paired': {'center': {'indices': set()}, 'total': {'indices': set()}},
        #                               'total': {'observations': len(fragment_matches), 'center': {}, 'total': {}}}
        for fragment in fragment_matches:
            center_residx1, center_residx2, match_score = fragment['mapped'], fragment['paired'], fragment['match']
            i, j, k = fragment['cluster']
            fragment_i_index_count_d[i] += 1
            fragment_j_index_count_d[j] += 1

            entity1_center_match_scores[center_residx1].append(match_score)
            entity2_center_match_scores[center_residx2].append(match_score)

            for idx1, idx2 in [(center_residx1 + j, center_residx2 + j) for j in self.fragment_range]:
                entity1_match_scores[idx1].append(match_score)
                entity2_match_scores[idx2].append(match_score)

        # -------------------------------------------
        # Score the interface individually
        mapped_center_score = nanohedra_fragment_match_score(entity1_center_match_scores)
        paired_center_score = nanohedra_fragment_match_score(entity2_center_match_scores)
        mapped_total_score = nanohedra_fragment_match_score(entity1_match_scores)
        paired_total_score = nanohedra_fragment_match_score(entity2_match_scores)
        # Combine
        all_residue_score = mapped_total_score + paired_total_score
        center_residue_score = mapped_center_score + paired_center_score
        # -------------------------------------------
        entity1_indices = set(entity1_center_match_scores.keys())
        entity1_total_indices = set(entity1_match_scores.keys())
        entity2_indices = set(entity2_center_match_scores.keys())
        entity2_total_indices = set(entity2_match_scores.keys())
        # Get individual number of CENTRAL residues with overlapping fragments given z_value criteria
        mapped_central_residues_with_fragment_overlap = len(entity1_indices)
        paired_central_residues_with_fragment_overlap = len(entity2_indices)
        # Combine
        central_residues_with_fragment_overlap = \
            mapped_central_residues_with_fragment_overlap + paired_central_residues_with_fragment_overlap
        # -------------------------------------------
        # Get the individual number of TOTAL residues with overlapping fragments given z_value criteria
        mapped_total_indices_with_fragment_overlap = len(entity1_total_indices)
        paired_total_indices_with_fragment_overlap = len(entity2_total_indices)
        # Combine
        total_indices_with_fragment_overlap = \
            mapped_total_indices_with_fragment_overlap + paired_total_indices_with_fragment_overlap
        # -------------------------------------------
        # Get the individual multiple fragment observation ratio observed for each side of the fragment query
        mapped_multiple_frag_ratio = total_observations / mapped_central_residues_with_fragment_overlap
        paired_multiple_frag_ratio = total_observations / paired_central_residues_with_fragment_overlap
        # Combine
        multiple_frag_ratio = total_observations*2 / central_residues_with_fragment_overlap
        # -------------------------------------------
        # Turn individual index counts into paired counts
        for index, count in fragment_i_index_count_d.items():
            total_fragment_content[index] += count
            # separated_fragment_metrics['mapped']['index'][index_count] = count / separated_fragment_metrics['number']
        for index, count in fragment_j_index_count_d.items():
            total_fragment_content[index] += count
            # separated_fragment_metrics['paired']['index'][index_count] = count / separated_fragment_metrics['number']
        # combined
        # for index, count in total_fragment_content.items():
        #     total_fragment_content[index] = count / (separated_fragment_metrics['total']['observations'] * 2)
        # -------------------------------------------
        separated_fragment_metrics['mapped'] = dict(
            center=dict(score=mapped_center_score,
                        # number=mapped_central_residues_with_fragment_overlap,
                        indices=entity1_indices),
            # center_match_scores=entity1_center_match_scores,
            # match_scores=entity1_match_scores,
            # index_count=fragment_i_index_count_d,
            total=dict(score=mapped_total_score,
                       # number=mapped_total_indices_with_fragment_overlap,
                       indices=entity1_total_indices),
            multiple_ratio=mapped_multiple_frag_ratio
        )
        separated_fragment_metrics['paired'] = dict(
            center=dict(score=paired_center_score,
                        # number=paired_central_residues_with_fragment_overlap,
                        indices=entity2_indices),
            # center_match_scores=entity2_center_match_scores,
            # match_scores=entity2_match_scores,
            # index_count=fragment_j_index_count_d,
            total=dict(score=paired_total_score,
                       # number=paired_total_indices_with_fragment_overlap,
                       indices=entity2_total_indices),
            multiple_ratio=paired_multiple_frag_ratio
        )
        # separated_fragment_metrics['mapped']['center']['score'] = mapped_center_score
        # separated_fragment_metrics['mapped']['center']['number'] = mapped_central_residues_with_fragment_overlap
        # separated_fragment_metrics['mapped']['total']['score'] = mapped_total_score
        # separated_fragment_metrics['mapped']['total']['number'] = mapped_total_indices_with_fragment_overlap
        # separated_fragment_metrics['mapped']['multiple_ratio'] = mapped_multiple_frag_ratio
        # separated_fragment_metrics['paired']['center']['score'] = paired_center_score
        # separated_fragment_metrics['paired']['center']['number'] = paired_central_residues_with_fragment_overlap
        # separated_fragment_metrics['paired']['total']['score'] = paired_total_score
        # separated_fragment_metrics['paired']['total']['number'] = paired_total_indices_with_fragment_overlap
        # separated_fragment_metrics['paired']['multiple_ratio'] = paired_multiple_frag_ratio

        separated_fragment_metrics['total'] = dict(
            observations=total_observations,
            multiple_ratio=multiple_frag_ratio,
            index_count=total_fragment_content,
            center=dict(score=center_residue_score, number=central_residues_with_fragment_overlap),
            total=dict(score=all_residue_score, number=total_indices_with_fragment_overlap)
        )

        return separated_fragment_metrics

    @staticmethod
    def format_fragment_metrics(metrics: dict) -> dict:
        """For a set of fragment metrics, return the formatted total fragment metrics

        Args:
            metrics:
        Returns:
            {center_indices, total_indices,
             nanohedra_score, nanohedra_score_center, multiple_fragment_ratio, number_residues_fragment_total,
             number_residues_fragment_center, number_fragments,
             percent_fragment_helix, percent_fragment_strand, percent_fragment_coil}
        """
        return {
            'center_indices': metrics['mapped']['center']['indices'].union(metrics['paired']['center']['indices']),
            'total_indices': metrics['mapped']['total']['indices'].union(metrics['paired']['total']['indices']),
            'nanohedra_score': metrics['total']['total']['score'],
            'nanohedra_score_center': metrics['total']['center']['score'],
            'multiple_fragment_ratio': metrics['total']['multiple_ratio'],
            'number_residues_fragment_total': metrics['total']['total']['number'],
            'number_residues_fragment_center': metrics['total']['center']['number'],
            'number_fragments': metrics['total']['observations'],
            # Todo ensure these metrics are accounted for if using a different cluster index
            'percent_fragment_helix': (metrics['total']['index_count'][1] / (metrics['total']['observations'] * 2)),
            'percent_fragment_strand': (metrics['total']['index_count'][2] / (metrics['total']['observations'] * 2)),
            'percent_fragment_coil': (metrics['total']['index_count'][3] + metrics['total']['index_count'][4]
                                      + metrics['total']['index_count'][5]) / (metrics['total']['observations'] * 2)}


class FragmentDatabaseFactory:
    """Return a FragmentDatabase instance by calling the Factory instance with the FragmentDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named FragmentDatabase
    """
    def __init__(self, **kwargs):
        self._databases = {}

    def __call__(self, source: str = putils.biological_interfaces, token: int = None, **kwargs) \
            -> FragmentDatabase | None:
        """Return the specified FragmentDatabase object singleton

        Args:
            source: The FragmentDatabase source name
        Returns:
            The instance of the specified FragmentDatabase
        """
        fragment_db = self._databases.get(source)
        if fragment_db:
            return fragment_db
        elif source == putils.biological_interfaces:
            if token == RELOAD_DB:
                return None
            try:
                self._databases[source] = utils.unpickle(putils.biological_fragment_db_pickle)
            except ModuleNotFoundError:
                raise RuntimeError(
                    f"Couldn't access the serialized {FragmentDatabase.__name__} which is required for operation. "
                    f'Please reload this by executing "{putils.pickle_program_requirements_cmd}"')
        else:
            self._databases[source] = FragmentDatabase(source=source, **kwargs)

        logger.info(f'Initializing {FragmentDatabase.__name__}({source})')
        # Attach the euler_lookup class
        self._databases[source].euler_lookup = euler_factory()

        return self._databases[source]

    def get(self, **kwargs) -> FragmentDatabase:
        """Return the specified FragmentDatabase object singleton

        Keyword Args:
            source: The FragmentDatabase source name
        Returns:
            The instance of the specified FragmentDatabase
        """
        return self.__call__(**kwargs)


fragment_factory: Annotated[FragmentDatabaseFactory,
                            'Calling this factory method returns the single instance of the FragmentDatabase class '
                            'containing fragment information specified by the "source" keyword argument'] = \
    FragmentDatabaseFactory()
"""Calling this factory method returns the single instance of the FragmentDatabase class containing fragment information
 specified by the "source" keyword argument"""


def nanohedra_fragment_match_score(per_residue_match_scores: dict[int, list[float]]) -> float:
    """Calculate the Nanohedra score from a dictionary with the 'center' residues and 'match_scores'

    Args:
        per_residue_match_scores: The residue mapped to its match score measurements
    Returns:
        The Nanohedra score
    """
    score = 0
    for residue_number, scores in per_residue_match_scores.items():
        n = 1
        for observation in sorted(scores, reverse=True):
            score += observation / n
            n *= 2

    return score


class TransformHasher:
    def __init__(self, max_width: float, translation_bin_width: float = 0.2, rotation_bin_width: float = 0.5,
                 dimensions: int = 6):
        """"""
        half_box_width = max_width
        self.box_width = 2 * half_box_width
        self.dimensions = dimensions
        self.translation_bin_width = translation_bin_width  # Angstrom minimum amount of search difference
        self.rotation_bin_width = rotation_bin_width  # Degree to bin euler angles
        self.offset_scipy_euler_to_bin = [180, 90, 180]
        logger.debug(f'translation_bin_width: {self.translation_bin_width}')
        logger.debug(f'rotation_bin_width: {self.rotation_bin_width}')
        # half_box_width = raidus1 - radius2
        # box_width = half_box_width * 2
        # half_box_width = max_width / 2
        self.box_lower = np.array([-half_box_width, -half_box_width, -half_box_width])
        logger.debug(f'self.box_lower: {self.box_lower}')
        # box_upper = self.box_lower * -1

        # Create dimsizes_
        number_of_bins = math.ceil(self.box_width / self.translation_bin_width)
        # number_tx_dimensions = np.full(3, number_of_bins, np.int)
        number_tx_bins = [number_of_bins, number_of_bins, number_of_bins]
        self.index_with_360 = [0, 2]
        number_of_rot_bins_360, remainder360 = divmod(360, self.rotation_bin_width)
        number_of_rot_bins_180, remainder180 = divmod(180, self.rotation_bin_width)
        # Ensure the values are integers
        number_of_rot_bins_360 = int(number_of_rot_bins_360)
        number_of_rot_bins_180 = int(number_of_rot_bins_180)
        # The number of bins always should be an integer. Ensure quotient is an integer for both 360/X and 180/X
        # otherwise the bins wouldn't space correctly as they wrap around
        if remainder180 != 0 or remainder360 != 0:
            raise ValueError(
                f"The value of the rotation_bin_size {self.rotation_bin_width} must be an integer denominator of"
                f" both 360 and 180. Got 360/rotation_bin_size={360 / self.rotation_bin_width},"
                f" 180/rotation_bin_size={180 / self.rotation_bin_width}")
        # Keeping with convention, keep the smaller euler angle on the second axis...
        # number_euler_dimensions = \
        #     np.array([number_of_rot_bins_360, number_of_rot_bins_180, number_of_rot_bins_360], np.int)
        number_euler_bins = [number_of_rot_bins_360, number_of_rot_bins_180, number_of_rot_bins_360]

        # Create dimprods_ . The form of this is different given the placement of the "theta" angle in the last position
        # of the euler array in SixDHasher.cc and using the conventional 2nd position here
        # dimension_products = np.ones(6, np.int)
        # for i in range(2, 7):  # [-2,-3,-4,-5,-6]
        #     dimension_products[-i] =
        dimension_products = [1, 0, 0, 0, 0, 0]
        try:
            for idx, (product, dimension_size) in enumerate(zip(dimension_products,
                                                                number_euler_bins
                                                                + number_tx_bins), 1):
                # input(f'product:{product}dimension_size:{dimension_size}')
                dimension_products[idx] = product * dimension_size
        except IndexError:  # Reached the end of dimension_products
            # Reverse the order for future usage
            self.dimension_products = [int(product) for product in dimension_products[::-1]]
            logger.debug(f'Found dimension_products: {self.dimension_products} from {dimension_products}')

        self.number_tx_bins = np.array(number_tx_bins)
        self.number_euler_bins = np.array(number_euler_bins)

    def transforms_to_bin_integers(self, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
        """From transformations consisting of stacked rotations and translations, identify the bin that each
        transformation falls into along each transformation dimension


        Args:
            rotations: Array with shape (N, 3, 3) where each N is the rotation paired with the translation N
            translations: Array with shape (N, 3) where each N is the translation paired with the rotation N
        Returns:
            Array with shape (number_of_transforms, dimensions) representing the unique bins along each dimension,
                that each transform occupies
        """
        #     -> tuple[int, int, int, int, int, int]:
        # For each of the translations, set in reference to the lower bound and divide by the bin widths to find the bin
        translation_bin_int = (translations - self.box_lower) // self.translation_bin_width
        if (translation_bin_int < 0).any():
            raise ValueError(f'Found a value for the binned translation integers less than 0. The transform box for the'
                             f" requested {self.__class__.__name__} can't properly handle the passed transformation")
        rotation = scipy.spatial.transform.Rotation.from_matrix(rotations)
        # 'xyz' lowercase denotes the rotation is external
        euler_angles = rotation.as_euler('xyz', degrees=True)
        euler_angles += self.offset_scipy_euler_to_bin
        rotation_bin_int = euler_angles // self.rotation_bin_width
        # Using the modulo operator should enable the returned euler_angles, which are in the range:
        # -180,180, -90,90, and -180,180
        # To be in the correct bins between 0,360, 0,180, 0,360
        # rotation_bin_int %= self.number_euler_bins
        # rotation_bin_int[:, self.index_with_360] %= self.number_euler_bins
        logger.debug(f'euler_angles: {euler_angles[:3]}')
        logger.debug(f'rotation_bin_int: {rotation_bin_int[:3]}')
        logger.debug(f'translation_bin_int: {translation_bin_int[:3]}')

        return np.concatenate((translation_bin_int.astype(int), rotation_bin_int.astype(int)), axis=1)

    def hash_bin_integers(self, bin_integers: np.ndarray) -> np.ndarray:
        """

        Args:
            bin_integers: Array with shape (number_of_transforms, dimensions) representing the unique bins along each
                dimension, that each transform occupies
        Returns:
            An integer hashing the binned integers which describe a position in the described 3D space
        """
        # # This is used for boost library c++ stuff
        # self.max_int32_prime = 999879
        # return (bin_integers * self.dimension_products).sum(axis=-1) % self.max_int32_prime
        # logger.debug(f'bin_integers[:3]: {bin_integers[:3]}')
        # result = (bin_integers * self.dimension_products).sum(axis=-1)
        # input(result[:3])
        # input(result.dtype)
        return (bin_integers * self.dimension_products).sum(axis=-1).astype(int)

    def transforms_to_hash(self, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
        """From pairs of rotations and translations, create a hash to describe the complete transformation

        Args:
            rotations: Array with shape (N, 3, 3) where each N is the rotation paired with the translation N
            translations: Array with shape (N, 3) where each N is the translation paired with the rotation N
        Returns:
            An integer hashing the transforms which relates to distinct orientational offset in the described space
        """
        return self.hash_bin_integers(self.transforms_to_bin_integers(rotations, translations))

    def hash_to_bins(self, index: int | np.ndarray) -> np.ndarray:
        """From a hash or multiple hashes describing a transformation, return the bins that that hash belongs too

        Args:
            index: The hash(es) to calculate bins for
        Returns:
            Array with shape (number_of_transforms, dimensions) representing the unique bins along each dimension,
                that each hash maps to the transformation space
        """
        try:
            bins = np.zeros((len(index), self.dimensions))
        except TypeError:  # int doesn't have len()
            bins = [0 for _ in range(self.dimensions)]
            for idx, product in enumerate(self.dimension_products):
                bins[idx], index = divmod(index, product)
            bins = np.array(bins)
        else:
            for idx, product in enumerate(self.dimension_products):
                bins[:, idx], index = divmod(index, product)

        return bins

    def bins_to_transform(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            bins: The integers which express particular bins in the transformation space
        Returns:
            The tuple of stacked rotations and stacked translations
        """
        translation_bin_int: np.ndarray = bins[:, :3]
        translations = translation_bin_int*self.translation_bin_width + self.box_lower
        rotation_bin_int: np.ndarray = bins[:, 3:]
        offset_euler_angles = rotation_bin_int * self.rotation_bin_width
        euler_angles = offset_euler_angles - self.offset_scipy_euler_to_bin
        # 'xyz' lowercase denotes the rotation is external
        rotation = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles, degrees=True)
        rotations = rotation.as_matrix()

        return rotations, translations

    def hash_to_transforms(self, index: int | np.ndarray) -> np.ndarray:
        """From a hash or multiple hashes describing a transformation, return the bins that that hash belongs too

        Args:
            index: The hash(es) to calculate bins for
        Returns:
            Array with shape (number_of_transforms, dimensions) representing the unique bins along each dimension,
                that each hash maps to the transformation space
        """
        # bins = self.hash_to_bins(index)
        # input(bins[:3])
        return self.bins_to_transform(self.hash_to_bins(index))


# @jitclass
class EulerLookup:
    def __init__(self, scale: float = 3.):
        # 6-d bool array [[[[[[True, False, ...], ...]]]]] with shape (37, 19, 37, 37, 19, 37)
        self.eul_lookup_40 = np.load(putils.binary_lookup_table_path)['a']
        """Indicates whether two sets of triplet integer values for each Euler angle are within an acceptable angular 
        offset. Acceptable offset is approximately +/- 40 degrees, or +/- 3 integers in one of the rotation angles and
        a couple of integers in another i.e.
        eul_lookup_40[1,5,1,1,8,1] -> True
        eul_lookup_40[1,5,1,1,9,1] -> False
        eul_lookup_40[1,5,1,1,7,4] -> True
        KM doesn't completely know how Todd made these
        """
        self.indices_lens = [0, 0]
        self.normalization = 1. / scale
        self.one_tolerance = 1. - 1.e-6
        self.eulint_divisor = 180. / np.pi * 0.1 * self.one_tolerance

    # @njit
    def get_eulint_from_guides_as_array(self, guide_coords: np.ndarray) -> np.ndarray:
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_coords[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # Todo Alternative
        e_array = np.zeros_like(guide_coords)
        # Subtract the guide coord origin from the other two dimensions to get unit basis vectors
        e_array[:, :2, :] = (guide_coords[:, 1:, :] - guide_coords[:, :1, :]) * self.normalization
        # Cross the basis vectors to get the orthogonal vector
        e_array[:, 2, :] = np.cross(e_array[:, 0], e_array[:, 1])
        # Bound by a min of -1 and max of 1 as arccos is valid in the domain of [1 to -1]
        e_array[:, 2, 2] = np.minimum(1, e_array[:, 2, 2])
        e_array[:, 2, 2] = np.maximum(-1, e_array[:, 2, 2])
        third_angle_degenerate = np.abs(e_array[:, 2, 2]) > self.one_tolerance
        # Put the results in the first position of every instances first vector
        # Second and third position are disregarded
        e_array[:, 0, 0] = np.where(third_angle_degenerate,
                                    np.arctan2(*e_array[:, :2, 0].T),
                                    np.arctan2(e_array[:, 0, 2], -e_array[:, 1, 2]))
        # Put the results in the first position of every instances second vector
        # arccos returns values of [0 to pi]
        e_array[:, 1, 0] = np.arccos(e_array[:, 2, 2])
        # Put the results in the first position of every instances third vector
        e_array[:, 2, 0] = np.where(third_angle_degenerate, 0, np.arctan2(*e_array[:, 2, :2].T))
        # Values in range (-18 to 18) (not inclusive) v. Then add 36 and divide by 36 to get the abs
        np.rint(e_array * self.eulint_divisor, out=e_array)
        e_array[:, [0, 2]] += 36
        e_array[:, [0, 2]] %= 36
        # # This way allows unpacking for check_lookup_table()
        # return e_array[:, :, 0].T.astype(int)
        # This returns in the intuitive column orientation with "C" array ordering
        return e_array[:, :, 0].astype(int)
        # Todo Alternative

    # @njit
    def get_eulint_from_guides(self, guide_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Take a set of guide atoms (3 xyz positions) and return integer indices for the euler angles describing the
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        """
        v1_a: An array of vectors containing the first vector which is orthogonal to v2_a (canonically on x)
        v2_a: An array of vectors containing the second vector which is orthogonal to v1_a (canonically on y)
        v3_a: An array of vectors containing the third vector which is the cross product of v1_a and v2_a
        """
        v1_a = (guide_coords[:, 1, :] - guide_coords[:, 0, :]) * self.normalization
        v2_a = (guide_coords[:, 2, :] - guide_coords[:, 0, :]) * self.normalization
        v3_a = np.cross(v1_a, v2_a)

        """Convert rotation matrix to euler angles in the form of an integer triplet (integer values are degrees
        divided by 10; these become indices for a lookup table)
        """
        # Bound by a min of -1 and max of 1 as arccos is valid in the domain of [1 to -1]
        e2_v = np.minimum(1, v3_a[:, 2])
        np.maximum(-1, e2_v, out=e2_v)

        # Check if the third angle is degenerate
        third_angle_degenerate = np.abs(e2_v) > self.one_tolerance
        e1_v = np.where(third_angle_degenerate,
                        np.arctan2(v2_a[:, 0], v1_a[:, 0]),
                        np.arctan2(v1_a[:, 2], -v2_a[:, 2]))

        # arccos returns values of [0 to pi]
        np.arccos(e2_v, out=e2_v)
        e3_v = np.where(third_angle_degenerate, 0, np.arctan2(*v3_a[:, :2].T))

        # np.floor(e1_v*self.eulint_divisor + .5, out=e1_v)  # More accurate
        np.rint(e1_v * self.eulint_divisor, out=e1_v)
        e1_v += 36
        e1_v %= 36
        # np.floor(e3_v*self.eulint_divisor + .5, out=e3_v)  # More accurate
        np.rint(e3_v * self.eulint_divisor, out=e3_v)
        e3_v += 36
        e3_v %= 36
        # Values in range (0 to 18) (not inclusive)
        # np.floor(e2_v*self.eulint_divisor + .5, out=e2_v)  # More accurate
        np.rint(e2_v * self.eulint_divisor, out=e2_v)

        return e1_v.astype(int), e2_v.astype(int), e3_v.astype(int)

    # @njit
    def lookup_by_euler_integers(self, *args: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        # Unpack each of the integer arrays
        eulintarray1_1, eulintarray1_2, eulintarray1_3, eulintarray2_1, eulintarray2_2, eulintarray2_3 = args

        indices1_len, indices2_len = len(eulintarray1_1), len(eulintarray2_1)

        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # check lookup table
        # start = time.time()
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # logger.debug(f'took {time.time() - start:8f}s')
        # overlap = self.eul_lookup_40[*stacked_eulintarray1.T,
        #                              stacked_eulintarray2.T]
        # there may be some solution where numpy.ix_ is used to broadcast the index operation

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs

    # @njit
    def lookup_by_euler_integers_as_array(self, eulintarray1: np.ndarray, eulintarray2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        indices1_len, indices2_len = len(eulintarray1), len(eulintarray2)

        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # stacked_eulintarray1 = np.tile(eulintarray1, (indices2_len, 1))
        # stacked_eulintarray2 = np.tile(eulintarray2, (indices1_len, 1))
        # # check lookup table
        # overlap = self.eul_lookup_40[stacked_eulintarray1[:, 0],
        #                              stacked_eulintarray1[:, 1],
        #                              stacked_eulintarray1[:, 2],
        #                              stacked_eulintarray2[:, 0],
        #                              stacked_eulintarray2[:, 1],
        #                              stacked_eulintarray2[:, 2]]
        # Transpose the intarray to have each of the columns unpacked as individual arrays
        eulintarray1_1, eulintarray1_2, eulintarray1_3 = eulintarray1.T
        eulintarray2_1, eulintarray2_2, eulintarray2_3 = eulintarray2.T

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # there may be some solution where numpy.meshgrid is used to broadcast the euler ints
        # check lookup table
        # start = time.time()
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # logger.debug(f'took {time.time() - start:8f}s')
        # overlap = self.eul_lookup_40[*stacked_eulintarray1.T,
        #                              stacked_eulintarray2.T]
        # there may be some solution where numpy.ix_ is used to broadcast the index operation

        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs

    def check_lookup_table(self, guide_coords1: np.ndarray, guide_coords2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #                    return_bool: bool = False
        """Returns a tuple with the index of the first fragment and second fragment where they overlap
        """
        # Ensure the atoms are passed as an array of (n, 3x3) matrices
        try:
            for idx, guide_coords in enumerate([guide_coords1, guide_coords2]):
                indices_len, *remainder = guide_coords.shape
                if remainder != [3, 3]:
                    logger.error(f'ERROR: guide coordinate array with wrong dimensions. '
                                 f'{guide_coords.shape} != (n, 3, 3)')
                    return np.array([]), np.array([])
                self.indices_lens[idx] = indices_len
        except (AttributeError, ValueError):  # guide_coords are the wrong format or the shape couldn't be unpacked
            logger.error(f'ERROR: guide coordinate array wrong type {type(guide_coords).__name__} != (n, 3, 3)')
            return np.array([]), np.array([])

        eulintarray1_1, eulintarray1_2, eulintarray1_3 = self.get_eulint_from_guides(guide_coords1)
        eulintarray2_1, eulintarray2_2, eulintarray2_3 = self.get_eulint_from_guides(guide_coords2)

        indices1_len, indices2_len = self.indices_lens
        index_array1 = np.repeat(np.arange(indices1_len), indices2_len)
        index_array2 = np.tile(np.arange(indices2_len), indices1_len)

        # Construct the correctly sized arrays to lookup euler space matching pairs from the all to all guide_coords
        # eulintarray1_1_r = np.repeat(eulintarray1_1, indices2_len)
        # eulintarray1_2_r = np.repeat(eulintarray1_2, indices2_len)
        # eulintarray1_3_r = np.repeat(eulintarray1_3, indices2_len)
        # eulintarray2_1_r = np.tile(eulintarray2_1, indices1_len)
        # eulintarray2_2_r = np.tile(eulintarray2_2, indices1_len)
        # eulintarray2_3_r = np.tile(eulintarray2_3, indices1_len)
        # check lookup table
        overlap = self.eul_lookup_40[np.repeat(eulintarray1_1, indices2_len),
                                     np.repeat(eulintarray1_2, indices2_len),
                                     np.repeat(eulintarray1_3, indices2_len),
                                     np.tile(eulintarray2_1, indices1_len),
                                     np.tile(eulintarray2_2, indices1_len),
                                     np.tile(eulintarray2_3, indices1_len)]
        # if return_bool:
        #     return overlap
        # else:
        return index_array1[overlap], index_array2[overlap]  # these are the overlapping ij pairs


class EulerLookupV1:
    def __init__(self, scale=3.0):
        self.eul_lookup_40 = np.load(putils.binary_lookup_table_path)['a']  # 6-d bool array [[[[[[True, False, ...], ...]]]]]
        self.scale = scale

    @staticmethod
    def get_eulerint10_from_rot(rot):
        # convert rotation matrix to euler angles in the form of an integer triplet
        # (integer values are degrees divided by 10; these become indices for a lookup table)
        tolerance = 1.e-6
        eulint = np.zeros(3, dtype=int)
        rot[2, 2] = min(rot[2, 2], 1.)
        rot[2, 2] = max(rot[2, 2], -1.)

        # if |rot[2,2]|~1, let the 3rd angle (which becomes degenerate with the 1st) be zero
        if rot[2, 2] > 1. - tolerance:
            e3 = 0.
            e1 = np.arctan2(rot[1, 0], rot[0, 0])
            e2 = 0.
        else:
            if rot[2, 2] < -(1. - tolerance):
                e3 = 0.
                e1 = np.arctan2(rot[1, 0], rot[0, 0])
                e2 = np.pi
            else:
                e2 = np.arccos(rot[2, 2])
                e1 = np.arctan2(rot[0, 2], -rot[1, 2])
                e3 = np.arctan2(rot[2, 0], rot[2, 1])

        eulint[0] = (np.rint(e1 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36
        eulint[1] = np.rint(e2 * 180. / np.pi * 0.1 * 0.999999)
        eulint[2] = (np.rint(e3 * 180. / np.pi * 0.1 * 0.999999) + 36) % 36

        return eulint

    def get_eulint_from_guides(self, guide_ats):
        # take a set of guide atoms (3 xyz positions) and return integer indices
        # for the euler angles describing the orientations of the axes they form
        # Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a
        # 3x3 array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]
        # use known scale value to normalize, to save repeated sqrt calculations

        if guide_ats.ndim != 3 or guide_ats.shape[1] != 3 or guide_ats.shape[2] != 3:
            print ('ERROR: guide atom array with wrong dimensions')

        nfrags = guide_ats.shape[0]
        rot = np.zeros((3, 3))
        eulintarray = np.zeros((nfrags, 3), dtype=int)

        # form the 2 difference vectors, normalize, then cross product
        for i in range(nfrags):
            v1 = (guide_ats[i, :, 1] - guide_ats[i, :, 0]) * 1. / self.scale
            v2 = (guide_ats[i, :, 2] - guide_ats[i, :, 0]) * 1. / self.scale
            v3 = np.cross(v1, v2)
            rot = np.array([v1, v2, v3])

            # get the euler indices
            eulintarray[i, :] = self.get_eulerint10_from_rot(rot)

        return eulintarray

    def check_lookup_table(self, guide_coords_list1, guide_coords_list2):
        return_tup_list = []

        guide_list_1_np = np.array(guide_coords_list1)
        guide_list_1_np_T = np.array([atoms_coords_1.T for atoms_coords_1 in guide_list_1_np])

        guide_list_2_np = np.array(guide_coords_list2)
        guide_list_2_np_T = np.array([atoms_coords_2.T for atoms_coords_2 in guide_list_2_np])

        eulintarray1 = self.get_eulint_from_guides(guide_list_1_np_T)
        eulintarray2 = self.get_eulint_from_guides(guide_list_2_np_T)

        # check lookup table
        for i in range(len(eulintarray1)):
            for j in range(len(eulintarray2)):
                (e1, e2, e3) = eulintarray1[i, :].flatten()
                (f1, f2, f3) = eulintarray2[j, :].flatten()
                if self.eul_lookup_40[e1, e2, e3, f1, f2, f3]:
                    return_tup_list.append((i, j))

        return map(np.array, zip(*return_tup_list))


class EulerLookupFactory:
    """Return an EulerLookup instance by calling the Factory instance

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer
    """

    def __init__(self, **kwargs):
        self._lookup_tables = {}

    def __call__(self, **kwargs) -> EulerLookup:
        """Return the specified EulerLookup object singleton

        Returns:
            The instance of the specified EulerLookup
        """
        lookup = self._lookup_tables.get('euler')
        if lookup:
            return lookup
        else:
            logger.info(f'Initializing {EulerLookup.__name__}()')
            self._lookup_tables['euler'] = EulerLookup(**kwargs)

        return self._lookup_tables['euler']

    def get(self, **kwargs) -> EulerLookup:
        """Return the specified EulerLookup object singleton

        Returns:
            The instance of the specified EulerLookup
        """
        return self.__call__(**kwargs)


euler_factory: Annotated[EulerLookupFactory,
                         'Calling this factory method returns the single instance of the EulerLookup class'] = \
    EulerLookupFactory()
"""Calling this factory method returns the single instance of the EulerLookup class"""
