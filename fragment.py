from __future__ import annotations

import os
from glob import glob
from typing import Annotated

import numpy as np
from Bio.Data.IUPACData import protein_letters

from PathUtils import intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, monofrag_cluster_rep_dirpath, \
    biological_interfaces, biological_fragment_db_pickle, frag_directory
from Pose import Model
from Structure import Structure
from SymDesignUtils import dictionary_lookup, unpickle, parameterize_frag_length, DesignError, \
    get_base_root_paths_recursively, start_log
from utils.MysqlPython import Mysql


# Globals
logger = start_log(name=__name__)


class ClusterInfoFile:
    def __init__(self, infofile_path):
        # self.infofile_path = infofile_path
        self.name = os.path.splitext(os.path.basename(infofile_path))[0]
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []
        # self.central_residue_pair_counts = []
        # self.load_info()

    # def load_info(self):
        with open(infofile_path, 'r') as f:
            info_lines = f.readlines()

        is_res_freq_line = False
        for line in info_lines:
            # if line.startswith("CLUSTER NAME:"):
            #     self.name = line.split()[2]
            if line.startswith("CLUSTER SIZE:"):
                self.size = int(line.split()[2])
            elif line.startswith("CLUSTER RMSD:"):
                self.rmsd = float(line.split()[2])
            elif line.startswith("CLUSTER REPRESENTATIVE NAME:"):
                self.representative_filename = line.split()[3]
            elif line.startswith("CENTRAL RESIDUE PAIR COUNT:"):
                is_res_freq_line = False
            elif is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            elif line.startswith("CENTRAL RESIDUE PAIR FREQUENCY:"):
                is_res_freq_line = True

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def get_rmsd(self):
        return self.rmsd

    def get_representative_filename(self):
        return self.representative_filename

    def get_central_residue_pair_freqs(self):
        return self.central_residue_pair_freqs


class FragmentDB:
    cluster_representatives_path: str
    cluster_info_path: str
    fragment_length: int
    indexed_ghosts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | dict
    # dict[int, tuple[3x3, 1x3, tuple[int, int, int], float]]
    info: dict[int, dict[int, dict[int, ClusterInfoFile]]] | None
    # monofrag_representatives_path: str
    paired_frags: dict[int, dict[int, dict[int, tuple[Model, str]]]] | None
    reps: dict[int, np.ndarray]

    def __init__(self, fragment_length: int = 5):
        self.cluster_representatives_path = intfrag_cluster_rep_dirpath
        self.cluster_info_path = intfrag_cluster_info_dirpath
        self.fragment_length = fragment_length
        self.indexed_ghosts = {}
        self.info = None
        # self.monofrag_representatives_path = monofrag_cluster_rep_dirpath
        self.paired_frags = None
        # self.reps = None

    # def get_monofrag_cluster_rep_dict(self):
        self.reps = {int(os.path.splitext(file)[0]):
                     Structure.from_file(os.path.join(root, file), entities=False, log=None).ca_coords
                     for root, dirs, files in os.walk(monofrag_cluster_rep_dirpath) for file in files}

    def get_intfrag_cluster_rep_dict(self):
        ijk_cluster_representatives = {}
        for root, dirs, files in os.walk(self.cluster_representatives_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in ijk_cluster_representatives:
                    ijk_cluster_representatives[i_cluster_type] = {}
                if j_cluster_type not in ijk_cluster_representatives[i_cluster_type]:
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    ijk_frag_cluster_rep_pdb = Model.from_file(os.path.join(root, file), entities=False, log=None)
                    # mapped_chain_idx = file.find('mappedchain')
                    # ijk_cluster_rep_mapped_chain = file[mapped_chain_idx + 12:mapped_chain_idx + 13]
                    # must look up the partner coords later by using chain_id stored in file
                    partner_chain_idx = file.find('partnerchain')
                    ijk_cluster_rep_partner_chain = file[partner_chain_idx + 13:partner_chain_idx + 14]
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)  # ijk_cluster_rep_mapped_chain,

        self.paired_frags = ijk_cluster_representatives
        if self.info:
            self.index_ghosts()

    def get_intfrag_cluster_info_dict(self):
        intfrag_cluster_info_dict = {}
        for root, dirs, files in os.walk(self.cluster_info_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in intfrag_cluster_info_dict:
                    intfrag_cluster_info_dict[i_cluster_type] = {}
                if j_cluster_type not in intfrag_cluster_info_dict[i_cluster_type]:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        ClusterInfoFile(os.path.join(root, file))

        self.info = intfrag_cluster_info_dict
        if self.paired_frags:
            self.index_ghosts()

    def index_ghosts(self):
        """From the fragment database, precompute all required data into arrays to populate Ghost Fragments"""
        for i_type in self.paired_frags:
            # must look up the partner coords by using stored chain_id
            stacked_bb_coords = np.array([frag_pdb.chain(frag_paired_chain).backbone_coords
                                          for j_dict in self.paired_frags[i_type].values()
                                          for frag_pdb, frag_paired_chain in j_dict.values()])
            # guide coords are stored with chain_id "9"
            stacked_guide_coords = np.array([frag_pdb.chain('9').coords for j_dict in self.paired_frags[i_type].values()
                                             for frag_pdb, _, in j_dict.values()])
            ijk_types = \
                np.array([(i_type, j_type, k_type) for j_type, j_dict in self.paired_frags[i_type].items()
                          for k_type in j_dict])
            # rmsd_array = np.array([self.info.cluster(type_set).rmsd for type_set in ijk_types])  # Todo
            rmsd_array = np.array([dictionary_lookup(self.info, type_set).rmsd for type_set in ijk_types])
            rmsd_array = np.where(rmsd_array == 0, 0.0001, rmsd_array)  # Todo ensure correct upon creation
            self.indexed_ghosts[i_type] = stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array


class FragmentDatabase(FragmentDB):
    def __init__(self, source: str = biological_interfaces, fragment_length: int = 5, init_db: bool = True,
                 sql: bool = False, **kwargs):
        super().__init__()  # FragmentDB
        # self.monofrag_representatives_path = monofrag_representatives_path
        # self.cluster_representatives_path
        # self.cluster_info_path = cluster_info_path
        # self.reps = None
        # self.paired_frags = None
        # self.info = None
        self.source: str = source
        # Todo load all statistics files into the pickle!
        # self.location = frag_directory.get(self.source, None)
        self.statistics: dict = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
        self.fragment_range: tuple[int, int]
        self.cluster_info: dict = {}
        # self.fragdb = None  # Todo

        if sql:
            self.start_mysql_connection()
            self.db = True
        else:  # self.source == 'directory':
            # Todo initialize as local directory
            self.db = False
            if init_db:
                logger.info(f'Initializing {source} FragmentDatabase from disk. This may take awhile...')
                # self.get_monofrag_cluster_rep_dict()
                self.get_intfrag_cluster_rep_dict()
                self.get_intfrag_cluster_info_dict()
                # self.get_cluster_info()

        self.get_db_statistics()
        self.fragment_range = parameterize_frag_length(fragment_length)

    @property
    def location(self) -> str | bytes | None:
        """Provide the location where fragments are stored"""
        return frag_directory.get(self.source, None)

    def get_db_statistics(self) -> dict:
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

    def get_db_aa_frequencies(self) -> dict[protein_letters, float]:
        """Retrieve database specific amino acid representation frequencies

        Returns:
            {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    def retrieve_cluster_info(self, cluster: str = None, source: str = None, index: str = None) -> \
            dict[str, int | float | str | dict[int, dict[protein_letters | str, float | tuple[int, float]]]]:
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
        if cluster:
            if cluster not in self.cluster_info:
                self.get_cluster_info(ids=[cluster])
            if source:
                if index is not None and source in ['mapped', 'paired']:  # must check for not None. The index can be 0
                    return self.cluster_info[cluster][source][index]
                else:
                    return self.cluster_info[cluster][source]
            else:
                return self.cluster_info[cluster]
        else:
            return self.cluster_info

    def get_cluster_info(self, ids: list[str] = None):
        """Load cluster information from the fragment database source into attribute cluster_info
        # todo change ids to a tuple
        Args:
            ids: ['1_2_123', ...]
        Sets:
            self.cluster_info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            if not ids:
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
