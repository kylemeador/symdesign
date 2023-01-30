from __future__ import annotations

import logging
from typing import Annotated, Literal, get_args, Type, Union

import numpy as np

from . import info
from symdesign import utils, structure
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
alignment_types_literal = Literal['mapped', 'paired']
alignment_types: tuple[alignment_types_literal] = get_args(alignment_types_literal)
fragment_info_keys = Literal[alignment_types_literal, 'match', 'cluster']
fragment_info_type = Type[dict[fragment_info_keys, Union[int, float, tuple[int, int, int]]]]
RELOAD_DB = 123


class Representative:
    backbone_coords: np.ndarray
    ca_coords: np.ndarray
    register: tuple[str, ...] = ('backbone_coords', 'ca_coords')

    def __init__(self, struct: 'structure.base.Structure', fragment_db: FragmentDatabase):
        for item in self.register:
            setattr(self, item, getattr(struct, item))

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

    def calculate_match_metrics(self, fragment_matches: list[fragment_info_type]) -> dict:
        """Return the various metrics calculated by overlapping fragments at the interface of two proteins

        Args:
            fragment_matches: [{'mapped': entity1_resnum, 'paired': entity2_resnum,
                                'cluster': tuple(int, int, int), 'match': score_term}, ...]
        Returns:
            {'mapped': {'center': {'indices': (set[int]), 'score': (float), 'number': (int)},
                        'total': {'indices': (set[int]), 'score': (float), 'number': (int)},
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

        entity1_center_match_scores, entity2_center_match_scores = {}, {}
        entity1_match_scores, entity2_match_scores = {}, {}
        separated_fragment_metrics = {'mapped': {'center': {'indices': set()}, 'total': {'indices': set()}},
                                      'paired': {'center': {'indices': set()}, 'total': {'indices': set()}},
                                      'total': {'observations': len(fragment_matches), 'center': {}, 'total': {}}}
        for fragment in fragment_matches:
            center_residx1, center_residx2, match_score = fragment['mapped'], fragment['paired'], fragment['match']
            separated_fragment_metrics['mapped']['center']['indices'].add(center_residx1)
            separated_fragment_metrics['paired']['center']['indices'].add(center_residx2)
            # i, j, k = list(map(int, fragment['cluster'].split('_')))
            i, j, k = fragment['cluster']
            fragment_i_index_count_d[i] += 1
            fragment_j_index_count_d[j] += 1
            # TODO an ideal measure of the central importance would weight central fragment observations to new center
            #  fragments higher than repeated observations. Say mapped residue 45 to paired 201. If there are two
            #  observations of this pair, just different ijk indices, then these would take the SUMi->n 1/i*2 additive form.
            #  If the observation went from residue 45 to paired residue 204, they constitute separate, but less important
            #  observations as these are part of the same SS and it is implied that if 201 contacts, 204 (and 198) are
            #  possible. In the case where the interaction goes from a residue to a different secondary structure, then
            #  this is the most important, say residue 45 to residue 322 on a separate helix. This indicates that
            #  residue 45 is ideally placed to interact with a separate SS and add them separately to the score
            #  |
            #  How the data structure to build this relationship looks is likely a graph, which has significant overlap to
            #  work I did on the consensus sequence higher order relationships in 8/20. I may find some ground in the middle
            #  of these two ideas where the graph theory could take hold and a more useful scoring metric could emerge,
            #  also possibly with a differentiable equation I could relate pose transformations to favorable fragments found
            #  in the interface.
            if center_residx1 not in entity1_center_match_scores:
                entity1_center_match_scores[center_residx1] = [match_score]
            else:
                entity1_center_match_scores[center_residx1].append(match_score)

            if center_residx2 not in entity2_center_match_scores:
                entity2_center_match_scores[center_residx2] = [match_score]
            else:
                entity2_center_match_scores[center_residx2].append(match_score)

            for resnum1, resnum2 in [(center_residx1 + j, center_residx2 + j) for j in self.fragment_range]:
                separated_fragment_metrics['mapped']['total']['indices'].add(resnum1)
                separated_fragment_metrics['paired']['total']['indices'].add(resnum2)

                if resnum1 not in entity1_match_scores:
                    entity1_match_scores[resnum1] = [match_score]
                else:
                    entity1_match_scores[resnum1].append(match_score)

                if resnum2 not in entity2_match_scores:
                    entity2_match_scores[resnum2] = [match_score]
                else:
                    entity2_match_scores[resnum2].append(match_score)

        separated_fragment_metrics['mapped']['center_match_scores'] = entity1_center_match_scores
        separated_fragment_metrics['paired']['center_match_scores'] = entity2_center_match_scores
        separated_fragment_metrics['mapped']['match_scores'] = entity1_match_scores
        separated_fragment_metrics['paired']['match_scores'] = entity2_match_scores
        separated_fragment_metrics['mapped']['index_count'] = fragment_i_index_count_d
        separated_fragment_metrics['paired']['index_count'] = fragment_j_index_count_d
        # -------------------------------------------
        # Score the interface individually
        mapped_total_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped']['match_scores'])
        mapped_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped']['center_match_scores'])
        paired_total_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired']['match_scores'])
        paired_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired']['center_match_scores'])
        # Combine
        all_residue_score = mapped_total_score + paired_total_score
        center_residue_score = mapped_center_score + paired_center_score
        # -------------------------------------------
        # Get individual number of CENTRAL residues with overlapping fragments given z_value criteria
        mapped_central_residues_with_fragment_overlap = len(separated_fragment_metrics['mapped']['center']['indices'])
        paired_central_residues_with_fragment_overlap = len(separated_fragment_metrics['paired']['center']['indices'])
        # Combine
        central_residues_with_fragment_overlap = \
            mapped_central_residues_with_fragment_overlap + paired_central_residues_with_fragment_overlap
        # -------------------------------------------
        # Get the individual number of TOTAL residues with overlapping fragments given z_value criteria
        mapped_total_indices_with_fragment_overlap = len(separated_fragment_metrics['mapped']['total']['indices'])
        paired_total_indices_with_fragment_overlap = len(separated_fragment_metrics['paired']['total']['indices'])
        # Combine
        total_indices_with_fragment_overlap = \
            mapped_total_indices_with_fragment_overlap + paired_total_indices_with_fragment_overlap
        # -------------------------------------------
        # Get the individual multiple fragment observation ratio observed for each side of the fragment query
        mapped_multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] / mapped_central_residues_with_fragment_overlap
        paired_multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] / paired_central_residues_with_fragment_overlap
        # Combine
        multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] * 2 / central_residues_with_fragment_overlap
        # -------------------------------------------
        # Turn individual index counts into paired counts # and percentages <- not accurate if summing later, need counts
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
        # if paired:
        separated_fragment_metrics['mapped']['center']['score'] = mapped_center_score
        separated_fragment_metrics['paired']['center']['score'] = paired_center_score
        separated_fragment_metrics['mapped']['center']['number'] = mapped_central_residues_with_fragment_overlap
        separated_fragment_metrics['paired']['center']['number'] = paired_central_residues_with_fragment_overlap
        separated_fragment_metrics['mapped']['total']['score'] = mapped_total_score
        separated_fragment_metrics['paired']['total']['score'] = paired_total_score
        separated_fragment_metrics['mapped']['total']['number'] = mapped_total_indices_with_fragment_overlap
        separated_fragment_metrics['paired']['total']['number'] = paired_total_indices_with_fragment_overlap
        separated_fragment_metrics['mapped']['multiple_ratio'] = mapped_multiple_frag_ratio
        separated_fragment_metrics['paired']['multiple_ratio'] = paired_multiple_frag_ratio
        #     return separated_fragment_metrics
        # else:
        separated_fragment_metrics['total']['center']['score'] = center_residue_score
        separated_fragment_metrics['total']['center']['number'] = central_residues_with_fragment_overlap
        separated_fragment_metrics['total']['total']['score'] = all_residue_score
        separated_fragment_metrics['total']['total']['number'] = total_indices_with_fragment_overlap
        separated_fragment_metrics['total']['multiple_ratio'] = multiple_frag_ratio
        separated_fragment_metrics['total']['index_count'] = total_fragment_content

        return separated_fragment_metrics

    @staticmethod
    def format_fragment_metrics(metrics: dict) -> dict:
        """For a set of fragment metrics, return the formatted total fragment metrics

        Args:
            metrics:
        Returns:
            {center_indices, total_indices,
             nanohedra_score, nanohedra_score_center, multiple_fragment_ratio, number_fragment_residues_total,
             number_fragment_residues_center, number_of_fragments,
             percent_fragment_helix, percent_fragment_strand, percent_fragment_coil}
        """
        return {
            'center_indices': metrics['mapped']['center']['indices'].union(metrics['paired']['center']['indices']),
            'total_indices': metrics['mapped']['total']['indices'].union(metrics['paired']['total']['indices']),
            'nanohedra_score': metrics['total']['total']['score'],
            'nanohedra_score_center': metrics['total']['center']['score'],
            'multiple_fragment_ratio': metrics['total']['multiple_ratio'],
            'number_fragment_residues_total': metrics['total']['total']['number'],
            'number_fragment_residues_center': metrics['total']['center']['number'],
            'number_of_fragments': metrics['total']['observations'],
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
                raise RuntimeError(f"Couldn't access the serialize {FragmentDatabase.__name__} which is required for "
                                   "operation. Please reload this by executing "
                                   f'"{putils.pickle_program_requirements_cmd}"')
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


def nanohedra_fragment_match_score(fragment_metrics: dict) -> float:
    """Calculate the Nanohedra score from a dictionary with the 'center' residues and 'match_scores'

    Args:
        fragment_metrics: {'center': {'residues' (int): (set)},
                           'total': {'residues' (int): (set)},
                           'center_match_scores': {residue number(int): (list[score (float)]), ...},
                           'match_scores': {residue number(int): (list[score (float)]), ...},
                           'index_count': {index (int): count (int), ...}}
    Returns:
        The Nanohedra score
    """
    # Generate Nanohedra score for all match scores
    score = 0
    for residue_number, res_scores in fragment_metrics.items():
        n = 1
        for peripheral_score in sorted(res_scores, reverse=True):
            score += peripheral_score * (1 / float(n))
            n *= 2

    return score


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
        orientations of the axes they form. Note that the positions are in a 3D array. Each guide_ats[i,:,:] is a 3x3
        array with the vectors stored *in columns*, i.e. one vector is in [i,:,j]. Use known scale value to normalize,
        to save repeated sqrt calculations
        """
        # Todo Alternative
        e_array = np.zeros(guide_coords.shape)
        e_array[:, :2, :] = (guide_coords[:, 1:, :] - guide_coords[:, :1, :]) * self.normalization
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

        indices1_len, indices2_len = eulintarray1_1.shape[0], eulintarray2_1.shape[0]

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
        indices1_len, indices2_len = eulintarray1.shape[0], eulintarray2.shape[0]

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
        # ensure the atoms are passed as an array of (n, 3x3) matrices
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

        # if |rot[2,2]|~1, let the 3rd angle (which becomes degernate with the 1st) be zero
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
