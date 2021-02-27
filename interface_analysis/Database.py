import math
import os

from PDB import PDB
from PathUtils import monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, \
    frag_directory
from SymDesignUtils import DesignError, unpickle, get_all_base_root_paths, start_log
from utils.MysqlPython import Mysql


logger = start_log(name=__name__)
index_offset = 1


class ClusterInfoFile:
    def __init__(self, infofile_path):
        self.infofile_path = infofile_path
        self.name = None
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []
        self.central_residue_pair_counts = []
        self.load_info()

    def load_info(self):
        infofile = open(self.infofile_path, "r")
        info_lines = infofile.readlines()
        infofile.close()
        is_res_freq_line = False
        for line in info_lines:

            if line.startswith("CLUSTER NAME:"):
                self.name = line.split()[2]
            if line.startswith("CLUSTER SIZE:"):
                self.size = int(line.split()[2])
            if line.startswith("CLUSTER RMSD:"):
                self.rmsd = float(line.split()[2])
            if line.startswith("CLUSTER REPRESENTATIVE NAME:"):
                self.representative_filename = line.split()[3]

            if line.startswith("CENTRAL RESIDUE PAIR COUNT:"):
                is_res_freq_line = False
            if is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            if line.startswith("CENTRAL RESIDUE PAIR FREQUENCY:"):
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
    def __init__(self):  # , monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath):
        self.monofrag_cluster_rep_dirpath = monofrag_cluster_rep_dirpath
        self.intfrag_cluster_rep_dirpath = intfrag_cluster_rep_dirpath
        self.intfrag_cluster_info_dirpath = intfrag_cluster_info_dirpath
        self.reps = None
        self.paired_frags = None
        self.info = None

    def get_monofrag_cluster_rep_dict(self):
        self.reps = {os.path.splitext(filename)[0]:
                     PDB.from_file(os.path.join(self.monofrag_cluster_rep_dirpath, filename), lazy=True)
                     for root, dirs, files in os.walk(self.monofrag_cluster_rep_dirpath) for filename in files}

    def get_intfrag_cluster_rep_dict(self):
        i_j_k_intfrag_cluster_rep_dict = {}
        for dirpath1, dirnames1, filenames1 in os.walk(self.intfrag_cluster_rep_dirpath):
            if not dirnames1:
                ijk_cluster_name = dirpath1.split("/")[-1]
                i_cluster_type = ijk_cluster_name.split("_")[0]
                j_cluster_type = ijk_cluster_name.split("_")[1]
                k_cluster_type = ijk_cluster_name.split("_")[2]

                if i_cluster_type not in i_j_k_intfrag_cluster_rep_dict:
                    i_j_k_intfrag_cluster_rep_dict[i_cluster_type] = {}

                if j_cluster_type not in i_j_k_intfrag_cluster_rep_dict[i_cluster_type]:
                    i_j_k_intfrag_cluster_rep_dict[i_cluster_type][j_cluster_type] = {}

                for dirpath2, dirnames2, filenames2 in os.walk(dirpath1):
                    for filename in filenames2:
                        ijk_frag_cluster_rep_pdb = PDB.from_file(os.path.join(dirpath1, filename), lazy=True)
                        ijk_frag_cluster_rep_mapped_chain_id = \
                            filename[filename.find("mappedchain") + 12:filename.find("mappedchain") + 13]

                        i_j_k_intfrag_cluster_rep_dict[i_cluster_type][j_cluster_type][k_cluster_type] = \
                            (ijk_frag_cluster_rep_pdb, ijk_frag_cluster_rep_mapped_chain_id)

        self.paired_frags = i_j_k_intfrag_cluster_rep_dict

    def get_intfrag_cluster_info_dict(self):
        intfrag_cluster_info_dict = {}
        for dirpath1, dirnames1, filenames1 in os.walk(self.intfrag_cluster_info_dirpath):
            if not dirnames1:
                ijk_cluster_name = dirpath1.split("/")[-1]
                i_cluster_type = ijk_cluster_name.split("_")[0]
                j_cluster_type = ijk_cluster_name.split("_")[1]
                k_cluster_type = ijk_cluster_name.split("_")[2]

                if i_cluster_type not in intfrag_cluster_info_dict:
                    intfrag_cluster_info_dict[i_cluster_type] = {}

                if j_cluster_type not in intfrag_cluster_info_dict[i_cluster_type]:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type] = {}

                for dirpath2, dirnames2, filenames2 in os.walk(dirpath1):
                    for filename in filenames2:
                        # if ".txt" in filename:
                        intfrag_cluster_info_dict[i_cluster_type][j_cluster_type][k_cluster_type] = ClusterInfoFile(
                            dirpath1 + "/" + filename)

        self.info = intfrag_cluster_info_dict
        # return intfrag_cluster_info_dict


class FragmentDatabase(FragmentDB):
    def __init__(self, source='directory', location=None, length=5, init_db=False):
        super().__init__()  # FragmentDB
        # self.monofrag_cluster_rep_dirpath = monofrag_cluster_rep_dirpath
        # self.intfrag_cluster_rep_dirpath = intfrag_cluster_rep_dirpath
        # self.intfrag_cluster_info_dirpath = intfrag_cluster_info_dirpath
        # self.reps = None
        # self.paired_frags = None
        # self.info = None
        self.source = source
        if location:
            self.location = frag_directory[location]  # location
        else:
            self.location = None
        self.statistics = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
        self.fragment_range = None
        self.cluster_info = {}
        self.fragdb = None

        if self.source == 'DB':
            self.start_mysql_connection()
            self.db = True
        elif self.source == 'directory':
            # Todo initialize as local directory
            self.db = False
            if init_db:
                logger.info('Initializing FragmentDatabase from disk. This may take awhile...')
                self.get_monofrag_cluster_rep_dict()
                self.get_intfrag_cluster_rep_dict()
                self.get_intfrag_cluster_info_dict()
        else:
            self.db = False

        self.get_db_statistics()
        self.parameterize_frag_length(length)

    def get_db_statistics(self):
        """Retrieve summary statistics for a specific fragment database located on directory

        Returns:
            (dict): {cluster_id1: [[mapped_index_average, paired_index_average, {max_weight_counts_mapped}, {paired}],
                                   total_fragment_observations],
                     cluster_id2: ...,
                     frequencies: {'A': 0.11, ...}}
                ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749], ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')  # Todo
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            for file in os.listdir(self.location):
                if 'statistics.pkl' in file:
                    self.statistics = unpickle(os.path.join(self.location, file))

    def get_db_aa_frequencies(self):
        """Retrieve database specific interface background AA frequencies

        Returns:
            (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics['frequencies']

    def retrieve_cluster_info(self, cluster=None, source=None, index=None):
        """Return information from the fragment information database by cluster_id, information source, and source index
         cluster_info takes the form:
            {'1_2_123': {'size': ..., 'rmsd': ..., 'rep': ...,
                         'mapped': indexed_frequency_dict, 'paired': indexed_frequency_dict}
                         indexed_frequency_dict = {-2: {'A': 0.1, 'C': 0., ..., 'info': (12, 0.41)},
                                                   -1: {}, 0: {}, 1: {}, 2: {}}

        Keyword Args:
            cluster=None (str): A cluster_id to get information about
            source=None (str): The source of information to gather from: ['size', 'rmsd', 'rep', 'mapped', 'paired']
            index=None (int): The index to gather information from. Must be from 'mapped' or 'paired'
        Returns:
            (dict)
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

    def get_cluster_info(self, ids=None):
        """Load cluster information from the fragment database source into attribute cluster_info
        # todo change ids to a tuple
        Keyword Args:
            id_list=None: [1_2_123, ...]
        Sets:
            self.cluster_info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')  # Todo
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            if not ids:
                directories = get_all_base_root_paths(self.location)
            else:
                directories = []
                for _id in ids:
                    c_id = _id.split('_')
                    _dir = os.path.join(self.location, c_id[0], '%s_%s' % (c_id[0], c_id[1]),
                                        '%s_%s_%s' % (c_id[0], c_id[1], c_id[2]))
                    directories.append(_dir)

            for cluster_directory in directories:
                cluster_id = os.path.basename(cluster_directory)
                filename = os.path.join(cluster_directory, '%s.pkl' % cluster_id)
                self.cluster_info[cluster_id] = unpickle(filename)

            # return self.cluster_info

    @staticmethod
    def get_cluster_id(cluster_id, index=3):
        """Returns the cluster identification string according the specified index

        Args:
            cluster_id (str): The id of the fragment cluster. Ex: 1_2_123
        Keyword Args:
            index_number=3 (int): The index on which to return. Ex: index_number=2 gives 1_2
        Returns:
            (str): The cluster_id modified by the requested index_number
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

    def parameterize_frag_length(self, length):
        """Generate fragment length range parameters for use in fragment functions"""
        _range = math.floor(length / 2)  # get the number of residues extending to each side
        if length % 2 == 1:
            self.fragment_range = (0 - _range, 0 + _range + index_offset)
            # return 0 - _range, 0 + _range + index_offset
        else:
            self.log.critical('%d is an even integer which is not symmetric about a single residue. '
                              'Ensure this is what you want' % length)
            self.fragment_range = (0 - _range, 0 + _range)

    def start_mysql_connection(self):
        self.fragdb = Mysql(host='cassini-mysql', database='kmeader', user='kmeader', password='km3@d3r')
