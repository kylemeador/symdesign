from __future__ import annotations

import os
from glob import glob
from itertools import repeat
from typing import AnyStr, Literal, get_args, Sequence

from structure.utils import protein_letters_alph1, protein_letters_literal
from utils.path import biological_interfaces, frag_directory, intfrag_cluster_info_dirpath
from utils import start_log, unpickle, get_base_root_paths_recursively, DesignError, parameterize_frag_length
from utils.sql import Mysql

logger = start_log(name=__name__)
source_literal = Literal['size', 'rmsd', 'rep', 'mapped', 'paired']
weighted_counts_keys = Literal[protein_letters_literal, 'stats']  # could add 'weight', 'count']
aa_weighted_counts_type: dict[weighted_counts_keys, int | tuple[int, int]]


class ClusterInfo:
    def __init__(self, infofile_path: AnyStr = None, name: str = None, size: int = None, rmsd: float = None,
                 representative_filename: str = None, mapped: aa_weighted_counts_type = None,
                 paired: aa_weighted_counts_type = None):
        self.central_residue_pair_freqs = []
        if infofile_path is not None:
            with open(infofile_path, 'r') as f:
                self.name = os.path.splitext(os.path.basename(infofile_path))[0]
                is_res_freq_line = False
                for line in f.readlines():
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
        else:
            self.name = name
            self.size = size
            self.rmsd = rmsd
            self.representative_filename = representative_filename
            self.mapped = mapped
            self.paired = paired

    @classmethod
    def from_file(cls, file: AnyStr):
        return cls(infofile_path=file)


class FragmentInfo:
    # info: dict[tuple[int, int, int], dict[source_literal, int | float | str | aa_weighted_counts_type]]
    info: dict[int, dict[int, dict[int, ClusterInfo]]]
    fragment_length: int
    fragment_range: tuple[int, int]
    source: str
    statistics: dict

    def __init__(self, source: str = biological_interfaces, fragment_length: int = 5, sql: bool = False, **kwargs):
        super().__init__()  # object
        self.cluster_info_path = intfrag_cluster_info_dirpath
        self.fragment_length = fragment_length
        self.fragment_range = parameterize_frag_length(fragment_length)
        self.info = {}
        self.source = source
        self.statistics = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]

        if sql:
            raise NotImplementedError("Can't connect to MySQL database yet")
            self.start_mysql_connection()  # sets self.fragdb  # Todo work out mechanism
            self.db = True
        else:  # self.source == 'directory':
            # Todo initialize as local directory
            self.db = False

        self._load_db_statistics()

    @property
    def location(self) -> AnyStr | None:
        """Provide the location where fragments are stored"""
        return frag_directory.get(self.source, None)

    def _load_db_statistics(self):
        """Retrieve summary statistics for a specific fragment database located on directory

        Returns:
            {cluster_id1: [[mapped_index_average, paired_index_average, {max_weight_counts_mapped}, {paired}],
                           total_fragment_observations], cluster_id2: ...,
             frequencies: {'A': 0.11, ...}}
                ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749], ...}
        """
        if self.db:
            raise NotImplementedError("Can't connect to MySQL database yet")
        else:
            stats_file = sorted(glob(os.path.join(self.location, 'statistics.pkl')))
            if len(stats_file) == 1:
                self.statistics = unpickle(stats_file[0])
            else:
                raise DesignError('There were too many statistics.pkl files found from the fragment database source!')
            # for file in os.listdir(self.location):
            #     if 'statistics.pkl' in file:
            #         self.statistics = unpickle(os.path.join(self.location, file))
            #         return

    @property
    def aa_frequencies(self) -> dict[protein_letters_alph1, float]:
        """Retrieve database specific amino acid representation frequencies

        Returns:
            {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    # Todo rework this and below func for Database
    def retrieve_cluster_info(self, cluster: str = None, source: source_literal = None, index: str = None) -> \
            dict[str, int | float | str | dict[int, dict[protein_letters_literal | str, float | tuple[int, float]]]]:
        """Return information from the fragment information database by cluster_id, information source, and source index

        Args:
            cluster: A cluster_id to get information about
            source: The source of information to retrieve. Must be one of 'size', 'rmsd', 'rep', 'mapped', or 'paired'
            index: The index to gather information from. Source must be one of 'mapped' or 'paired' to use
        Returns:
            {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': indexed_frequencies, 'paired': indexed_frequencies}
            Where indexed_frequencies has format {-2: {'A': 0.1, 'C': 0., ..., 'info': (12, 0.41)}, -1: {}, ..., 2: {}}
        """
        try:
            cluster_data = self.info[cluster]
        except KeyError:
            self.load_cluster_info(ids=[cluster])
            cluster_data = self.info[cluster]

        if source is None:
            return cluster_data
        else:
            if index is None:  # Must check for None, index can be 0
                return cluster_data[source]
            else:  # source in ['mapped', 'paired']:
                try:
                    return cluster_data[source][index]
                except KeyError:
                    raise KeyError(f'The source {source} is not available. '
                                   f'Try one of {", ".join(get_args(source_literal))}')
                except IndexError:
                    raise IndexError(f'The index {index} is outside of the fragment range. '
                                     f'Try one of {", ".join(cluster_data["mapped"].keys())}')
                except TypeError:
                    raise TypeError(f'You must provide "mapped" or "paired" if you wish to use an index')

    def load_cluster_info(self, ids: Sequence[str] = None):
        """Load cluster information from the fragment database source into attribute .info

        Args:
            ids: ['1_2_123', ...]
        Sets:
            self.info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            raise NotImplementedError("Can't connect to MySQL database yet")
        else:
            if ids is None:  # Load all data
                identified_directories = {os.path.basename(cluster_directory): cluster_directory
                                          for cluster_directory in get_base_root_paths_recursively(self.location)}
            else:
                identified_directories = \
                    [(ids[idx], os.path.join(self.location, c_id1, f'{c_id1}_{c_id2}', f'{c_id1}_{c_id2}_{c_id3}'))
                     for idx, (c_id1, c_id2, c_id3) in enumerate(map(str.split, ids, repeat('_')))]

            self.info.update({tuple(cluster_id.split('_')):
                              ClusterInfo(name=cluster_id, **unpickle(os.path.join(cluster_directory,
                                                                                   f'{cluster_id}.pkl')))
                              for cluster_id, cluster_directory in identified_directories})

    def load_cluster_info_from_text(self, ids: Sequence[str] = None):
        """Load cluster information from the fragment database source text files into attribute .info

        Args:
            ids: ['1_2_123', ...]
        Sets:
            self.info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            raise NotImplementedError("Can't connect to MySQL database yet")
        else:
            if ids is None:  # Load all data
                for root, dirs, files in os.walk(self.cluster_info_path):
                    if not dirs:
                        i_cluster_type, j_cluster_type, k_cluster_type = map(int, os.path.basename(root).split('_'))

                        # if i_cluster_type not in self.info:
                        #     self.info[i_cluster_type] = {}
                        # if j_cluster_type not in self.info[i_cluster_type]:
                        #     self.info[i_cluster_type][j_cluster_type] = {}

                        # for file in files:
                        # There is only one file
                        self.info[(i_cluster_type, j_cluster_type, k_cluster_type)] = \
                            ClusterInfo.from_file(os.path.join(root, files[0]))
            else:
                for _id in ids:
                    c_id1, c_id2, c_id3 = map(int, _id.split('_'))
                    self.info[(c_id1, c_id2, c_id3)] = \
                        ClusterInfo.from_file(os.path.join(self.cluster_info_path, c_id1, f'{c_id1}_{c_id2}', f'{c_id1}_{c_id2}_{c_id3}'))

    @staticmethod
    def get_cluster_id(cluster_id: str, index: int = 3) -> str:
        """Returns the cluster identification string according the specified index

        Args:
            cluster_id: The id of the fragment cluster. Ex: 1_2_123
            index: The index on which to return. Ex: index_number=2 gives 1_2
        Returns:
            The cluster_id modified by the requested index_number
        """
        while len(cluster_id) < 3:
            cluster_id += '0'

        cluster_id_split = cluster_id.split('_')
        if len(cluster_id_split) == 1:  # in case of 12123? -> ['12123', '']
            id_l = [cluster_id[:1], cluster_id[1:2], cluster_id[2:]]
        else:
            id_l = cluster_id_split

        info = id_l[:index]

        while len(info) < 3:  # Ensure the returned string has at least 3 indices
            info.append('0')

        return '_'.join(info)

    def start_mysql_connection(self):
        self.fragdb = Mysql(host='cassini-mysql', database='kmeader', user='kmeader', password='km3@d3r')
