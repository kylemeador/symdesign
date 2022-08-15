from __future__ import annotations

import os
from glob import glob
from typing import AnyStr, Iterable, Literal

from Bio.Data.IUPACData import protein_letters

from utils.path import biological_interfaces, frag_directory
from utils import start_log, unpickle, get_base_root_paths_recursively, DesignError, parameterize_frag_length
from utils.MysqlPython import Mysql

logger = start_log(name=__name__)
protein_letter_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


class FragmentInfo:
    cluster_info: dict
    fragment_length: int
    fragment_range: tuple[int, int]
    source: str
    statistics: dict

    def __init__(self, source: str = biological_interfaces, fragment_length: int = 5, sql: bool = False, **kwargs):
        super().__init__()  # object
        # Todo load all statistics files into the pickle!
        self.cluster_info = {}
        self.fragment_length = fragment_length
        self.fragment_range = parameterize_frag_length(fragment_length)
        self.source = source
        self.statistics = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]

        if sql:
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
            logger.warning('No SQL DB connected yet!')  # Todo
            raise NotImplementedError('Can\'t connect to MySQL database yet')
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
    def aa_frequencies(self) -> dict[protein_letters, float]:
        """Retrieve database specific amino acid representation frequencies

        Returns:
            {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    def retrieve_cluster_info(self, cluster: str = None, source: str = None, index: str = None) -> \
            dict[str, int | float | str | dict[int, dict[protein_letter_literal | str, float | tuple[int, float]]]]:
        # Todo rework this and below func for Database
        """Return information from the fragment information database by cluster_id, information source, and source index

        Args:
            cluster: A cluster_id to get information about
            source: The source of information to gather from: ['size', 'rmsd', 'rep', 'mapped', 'paired']
            index: The index to gather information from. Must be from 'mapped' or 'paired'
        Returns:
            {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': indexed_frequencies, 'paired': indexed_frequencies}
            Where indexed_frequencies has format {-2: {'A': 0.1, 'C': 0., ..., 'info': (12, 0.41)}, -1: {}, ..., 2: {}}
        """
        if cluster is not None:
            if cluster not in self.cluster_info:
                self._load_cluster_info(ids=[cluster])
            if source is not None:
                if index is not None and source in ['mapped', 'paired']:  # must check for not None. The index can be 0
                    return self.cluster_info[cluster][source][index]
                else:
                    return self.cluster_info[cluster][source]
            else:
                return self.cluster_info[cluster]
        else:
            return self.cluster_info

    def _load_cluster_info(self, ids: Iterable[str] = None):
        """Load cluster information from the fragment database source into attribute cluster_info
        # Todo change ids to a tuple
        Args:
            ids: ['1_2_123', ...]
        Sets:
            self.cluster_info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')
            raise NotImplementedError('Can\'t connect to MySQL database yet')
        else:
            if ids is None:
                directories = get_base_root_paths_recursively(self.location)
            else:
                directories = []
                for _id in ids:
                    c_id = _id.split('_')
                    _dir = os.path.join(self.location, c_id[0], '%s_%s' % (c_id[0], c_id[1]),
                                        '%s_%s_%s' % (c_id[0], c_id[1], c_id[2]))
                    directories.append(_dir)

            for cluster_directory in directories:
                cluster_id = os.path.basename(cluster_directory)
                self.cluster_info[cluster_id] = unpickle(os.path.join(cluster_directory, '%s.pkl' % cluster_id))

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

        if len(cluster_id.split('_')) != 3:  # in case of 12123?
            id_l = [cluster_id[:1], cluster_id[1:2], cluster_id[2:]]
        else:
            id_l = cluster_id.split('_')

        info = [id_l[i] for i in range(index)]

        while len(info) < 3:  # ensure the returned string has at least 3 indices
            info.append('0')

        return '_'.join(info)

    # def parameterize_frag_length(self, length):
    #     """Generate fragment length range parameters for use in fragment functions"""
    #     _range = math.floor(length / 2)  # get the number of residues extending to each side
    #     if length % 2 == 1:  # fragment length is odd
    #         self.fragment_range = (0 - _range, 0 + _range + index_offset)
    #         # return 0 - _range, 0 + _range + index_offset
    #     else:  # length is even
    #         logger.critical(f'{length} is an even integer which is not symmetric about a single residue. '
    #                         'Ensure this is what you want')
    #         self.fragment_range = (0 - _range, 0 + _range)

    def start_mysql_connection(self):
        self.fragdb = Mysql(host='cassini-mysql', database='kmeader', user='kmeader', password='km3@d3r')
