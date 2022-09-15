from __future__ import annotations

import os
from copy import copy
from typing import Annotated

import numpy as np

from utils.path import intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, monofrag_cluster_rep_dirpath, \
    biological_interfaces, biological_fragment_db_pickle
import structure
from utils import dictionary_lookup, start_log, unpickle
from resources.info import FragmentInfo

# Globals
logger = start_log(name=__name__)


class ClusterInfoFile:
    def __init__(self, infofile_path):
        self.name = os.path.splitext(os.path.basename(infofile_path))[0]
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []

        with open(infofile_path, 'r') as f:
            info_lines = f.readlines()

        is_res_freq_line = False
        for line in info_lines:
            # if line.startswith("CLUSTER NAME:"):
            #     self.name = line.split()[2]
            if line.startswith('CLUSTER SIZE:'):
                self.size = int(line.split()[2])
            elif line.startswith('CLUSTER RMSD:'):
                self.rmsd = float(line.split()[2])
            elif line.startswith('CLUSTER REPRESENTATIVE NAME:'):
                self.representative_filename = line.split()[3]
            elif line.startswith('CENTRAL RESIDUE PAIR COUNT:'):
                is_res_freq_line = False
            elif is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            elif line.startswith('CENTRAL RESIDUE PAIR FREQUENCY:'):
                is_res_freq_line = True


class FragmentDatabase(FragmentInfo):
    cluster_representatives_path: str
    cluster_info_path: str
    indexed_ghosts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | dict
    # dict[int, tuple[3x3, 1x3, tuple[int, int, int], float]]
    info: dict[int, dict[int, dict[int, ClusterInfoFile]]]
    # monofrag_representatives_path: str
    paired_frags: dict[int, dict[int, dict[int, tuple['structure.model.Model', str]]]]
    reps: dict[int, np.ndarray]

    def __init__(self, init_db: bool = True, **kwargs):  # fragment_length: int = 5
        super().__init__(**kwargs)
        if self.source == biological_interfaces:
            self.cluster_representatives_path = intfrag_cluster_rep_dirpath
            self.cluster_info_path = intfrag_cluster_info_dirpath
            self.monofrag_representatives_path = monofrag_cluster_rep_dirpath

        if init_db:
            logger.info(f'Initializing {self.source} FragmentDatabase from disk. This may take awhile...')
            # self.get_monofrag_cluster_rep_dict()
            self.reps = {int(os.path.splitext(file)[0]):
                         structure.base.Structure.from_file(os.path.join(root, file), entities=False, log=None).ca_coords
                         for root, dirs, files in os.walk(self.monofrag_representatives_path) for file in files}
            self.get_intfrag_cluster_rep_dict()
            self.get_intfrag_cluster_info_dict()
            self.index_ghosts()
            # self._load_cluster_info()
        else:
            self.reps = {}
            self.paired_frags = {}
            self.info = {}
            self.indexed_ghosts = {}

    def get_intfrag_cluster_rep_dict(self):
        self.paired_frags = {}
        for root, dirs, files in os.walk(self.cluster_representatives_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in self.paired_frags:
                    self.paired_frags[i_cluster_type] = {}
                if j_cluster_type not in self.paired_frags[i_cluster_type]:
                    self.paired_frags[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    ijk_frag_cluster_rep_pdb = structure.model.Model.from_file(os.path.join(root, file), entities=False, log=None)
                    # mapped_chain_idx = file.find('mappedchain')
                    # ijk_cluster_rep_mapped_chain = file[mapped_chain_idx + 12:mapped_chain_idx + 13]
                    # must look up the partner coords later by using chain_id stored in file
                    partner_chain_idx = file.find('partnerchain')
                    ijk_cluster_rep_partner_chain = file[partner_chain_idx + 13:partner_chain_idx + 14]
                    self.paired_frags[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)  # ijk_cluster_rep_mapped_chain,

    def get_intfrag_cluster_info_dict(self):
        self.info = {}
        for root, dirs, files in os.walk(self.cluster_info_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in self.info:
                    self.info[i_cluster_type] = {}
                if j_cluster_type not in self.info[i_cluster_type]:
                    self.info[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    self.info[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        ClusterInfoFile(os.path.join(root, file))

    def index_ghosts(self):
        """From the fragment database, precompute all required data into numpy arrays to populate Ghost Fragments

        keeping each data type as numpy array allows facile indexing for allowed ghosts if a clash test is performed
        """
        self.indexed_ghosts = {}
        for i_type in self.paired_frags:
            # Must look up the partner coords by using stored chain_id
            stacked_bb_coords = np.array([frag_pdb.chain(frag_paired_chain).backbone_coords
                                          for j_dict in self.paired_frags[i_type].values()
                                          for frag_pdb, frag_paired_chain in j_dict.values()])
            # Todo store these as a numpy array instead of as a chain
            # Guide coords are stored with chain_id "9"
            stacked_guide_coords = np.array([frag_pdb.chain('9').coords
                                             for j_dict in self.paired_frags[i_type].values()
                                             for frag_pdb, _, in j_dict.values()])
            ijk_types = np.array([(i_type, j_type, k_type)
                                  for j_type, j_dict in self.paired_frags[i_type].items()
                                  for k_type in j_dict])
            # rmsd_array = np.array([self.info.cluster(type_set).rmsd for type_set in ijk_types])  # Todo
            rmsd_array = np.array([dictionary_lookup(self.info, type_set).rmsd for type_set in ijk_types])
            rmsd_array = np.where(rmsd_array == 0, 0.0001, rmsd_array)  # Todo ensure rmsd rounded correct upon creation
            self.indexed_ghosts[i_type] = stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array

    def calculate_match_metrics(self, fragment_matches: list[dict]) -> dict:
        """Return the various metrics calculated by overlapping fragments at the interface of two proteins

        Args:
            fragment_matches: [{'mapped': entity1_resnum, 'match': score_term, 'paired': entity2_resnum,
                                'culster': cluster_id}, ...]
        Returns:
            {'mapped': {'center': {'residues': (set[int]), 'score': (float), 'number': (int)},
                        'total': {'residues': (set[int]), 'score': (float), 'number': (int)},
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
        # fragment_i_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        fragment_j_index_count_d = copy(fragment_i_index_count_d)
        total_fragment_content = copy(fragment_i_index_count_d)

        entity1_center_match_scores, entity2_center_match_scores = {}, {}
        entity1_match_scores, entity2_match_scores = {}, {}
        separated_fragment_metrics = {'mapped': {'center': {'residues': set()}, 'total': {'residues': set()}},
                                      'paired': {'center': {'residues': set()}, 'total': {'residues': set()}},
                                      'total': {'observations': len(fragment_matches), 'center': {}, 'total': {}}}
        for fragment in fragment_matches:
            center_resnum1, center_resnum2, match_score = fragment['mapped'], fragment['paired'], fragment['match']
            separated_fragment_metrics['mapped']['center']['residues'].add(center_resnum1)
            separated_fragment_metrics['paired']['center']['residues'].add(center_resnum2)

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
            if center_resnum1 not in entity1_center_match_scores:
                entity1_center_match_scores[center_resnum1] = [match_score]
            else:
                entity1_center_match_scores[center_resnum1].append(match_score)

            if center_resnum2 not in entity2_center_match_scores:
                entity2_center_match_scores[center_resnum2] = [match_score]
            else:
                entity2_center_match_scores[center_resnum2].append(match_score)

            # for resnum1, resnum2 in [(fragment['mapped'] + j, fragment['paired'] + j) for j in range(-2, 3)]:
            for resnum1, resnum2 in [(center_resnum1+j, center_resnum2+j) for j in self.fragment_range]:
                separated_fragment_metrics['mapped']['total']['residues'].add(resnum1)
                separated_fragment_metrics['paired']['total']['residues'].add(resnum2)

                if resnum1 not in entity1_match_scores:
                    entity1_match_scores[resnum1] = [match_score]
                else:
                    entity1_match_scores[resnum1].append(match_score)

                if resnum2 not in entity2_match_scores:
                    entity2_match_scores[resnum2] = [match_score]
                else:
                    entity2_match_scores[resnum2].append(match_score)

            i, j, k = list(map(int, fragment['cluster'].split('_')))
            fragment_i_index_count_d[i] += 1
            fragment_j_index_count_d[j] += 1

        separated_fragment_metrics['mapped']['center_match_scores'] = entity1_center_match_scores
        separated_fragment_metrics['paired']['center_match_scores'] = entity2_center_match_scores
        separated_fragment_metrics['mapped']['match_scores'] = entity1_match_scores
        separated_fragment_metrics['paired']['match_scores'] = entity2_match_scores
        separated_fragment_metrics['mapped']['index_count'] = fragment_i_index_count_d
        separated_fragment_metrics['paired']['index_count'] = fragment_j_index_count_d
        # -------------------------------------------
        # score the interface individually
        mapped_total_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped']['match_scores'])
        mapped_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped']['center_match_scores'])
        paired_total_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired']['match_scores'])
        paired_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired']['center_match_scores'])
        # combine
        all_residue_score = mapped_total_score + paired_total_score
        center_residue_score = mapped_center_score + paired_center_score
        # -------------------------------------------
        # Get individual number of CENTRAL residues with overlapping fragments given z_value criteria
        mapped_central_residues_with_fragment_overlap = len(separated_fragment_metrics['mapped']['center']['residues'])
        paired_central_residues_with_fragment_overlap = len(separated_fragment_metrics['paired']['center']['residues'])
        # combine
        central_residues_with_fragment_overlap = \
            mapped_central_residues_with_fragment_overlap + paired_central_residues_with_fragment_overlap
        # -------------------------------------------
        # Get the individual number of TOTAL residues with overlapping fragments given z_value criteria
        mapped_total_residues_with_fragment_overlap = len(separated_fragment_metrics['mapped']['total']['residues'])
        paired_total_residues_with_fragment_overlap = len(separated_fragment_metrics['paired']['total']['residues'])
        # combine
        total_residues_with_fragment_overlap = \
            mapped_total_residues_with_fragment_overlap + paired_total_residues_with_fragment_overlap
        # -------------------------------------------
        # get the individual multiple fragment observation ratio observed for each side of the fragment query
        mapped_multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] / mapped_central_residues_with_fragment_overlap
        paired_multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] / paired_central_residues_with_fragment_overlap
        # combine
        multiple_frag_ratio = \
            separated_fragment_metrics['total']['observations'] * 2 / central_residues_with_fragment_overlap
        # -------------------------------------------
        # turn individual index counts into paired counts # and percentages <- not accurate if summing later, need counts
        for index, count in separated_fragment_metrics['mapped']['index_count'].items():
            total_fragment_content[index] += count
            # separated_fragment_metrics['mapped']['index'][index_count] = count / separated_fragment_metrics['number']
        for index, count in separated_fragment_metrics['paired']['index_count'].items():
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
        separated_fragment_metrics['mapped']['total']['number'] = mapped_total_residues_with_fragment_overlap
        separated_fragment_metrics['paired']['total']['number'] = paired_total_residues_with_fragment_overlap
        separated_fragment_metrics['mapped']['multiple_ratio'] = mapped_multiple_frag_ratio
        separated_fragment_metrics['paired']['multiple_ratio'] = paired_multiple_frag_ratio
        #     return separated_fragment_metrics
        # else:
        separated_fragment_metrics['total']['center']['score'] = center_residue_score
        separated_fragment_metrics['total']['center']['number'] = central_residues_with_fragment_overlap
        separated_fragment_metrics['total']['total']['score'] = all_residue_score
        separated_fragment_metrics['total']['total']['number'] = total_residues_with_fragment_overlap
        separated_fragment_metrics['total']['multiple_ratio'] = multiple_frag_ratio
        separated_fragment_metrics['total']['index_count'] = total_fragment_content

        return separated_fragment_metrics


class FragmentDatabaseFactory:
    """Return a FragmentDatabase instance by calling the Factory instance with the FragmentDatabase source name

    Handles creation and allotment to other processes by saving expensive memory load of multiple instances and
    allocating a shared pointer to the named FragmentDatabase
    """
    def __init__(self, **kwargs):
        self._databases = {}

    def __call__(self, source: str = biological_interfaces, **kwargs) -> FragmentDatabase:
        """Return the specified FragmentDatabase object singleton

        Args:
            source: The FragmentDatabase source name
        Returns:
            The instance of the specified FragmentDatabase
        """
        fragment_db = self._databases.get(source)
        if fragment_db:
            return fragment_db
        elif source == biological_interfaces:
            logger.info(f'Initializing {source} {FragmentDatabase.__name__}')
            self._databases[source] = unpickle(biological_fragment_db_pickle)
        else:
            logger.info(f'Initializing {source} {FragmentDatabase.__name__}')
            self._databases[source] = FragmentDatabase(source=source, **kwargs)

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


def format_fragment_metrics(metrics: dict, null: bool = False) -> dict:
    """For a set of fragment metrics, return the formatted total fragment metrics

    Returns:
        {center_residues, total_residues,
         nanohedra_score, nanohedra_score_center, multiple_fragment_ratio, number_fragment_residues_total,
         number_fragment_residues_center, number_of_fragments, percent_fragment_helix, percent_fragment_strand,
         percent_fragment_coil}
    """
    if null or not metrics:
        return fragment_metric_template
    return {'center_residues': metrics['mapped']['center']['residues'].union(metrics['paired']['center']['residues']),
            'total_residues': metrics['mapped']['total']['residues'].union(metrics['paired']['total']['residues']),
            'nanohedra_score': metrics['total']['total']['score'],
            'nanohedra_score_center': metrics['total']['center']['score'],
            'multiple_fragment_ratio': metrics['total']['multiple_ratio'],
            'number_fragment_residues_total': metrics['total']['total']['number'],
            'number_fragment_residues_center': metrics['total']['center']['number'],
            'number_of_fragments': metrics['total']['observations'],
            'percent_fragment_helix': (metrics['total']['index_count'][1] / (metrics['total']['observations'] * 2)),
            'percent_fragment_strand': (metrics['total']['index_count'][2] / (metrics['total']['observations'] * 2)),
            'percent_fragment_coil': ((metrics['total']['index_count'][3] + metrics['total']['index_count'][4]
                                       + metrics['total']['index_count'][5]) / (metrics['total']['observations'] * 2))}


fragment_metric_template = {'center_residues': set(), 'total_residues': set(),
                            'nanohedra_score': 0.0, 'nanohedra_score_center': 0.0, 'multiple_fragment_ratio': 0.0,
                            'number_fragment_residues_total': 0, 'number_fragment_residues_center': 0,
                            'number_of_fragments': 0, 'percent_fragment_helix': 0.0, 'percent_fragment_strand': 0.0,
                            'percent_fragment_coil': 0.0}
