import os
import math
from glob import glob

import numpy as np

from PDB import PDB
from PathUtils import monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, \
    frag_directory
from SequenceProfile import parse_hhblits_pssm, MultipleSequenceAlignment, read_fasta_file  # parse_pssm
from Structure import parse_stride
from SymDesignUtils import DesignError, unpickle, get_all_base_root_paths, start_log, dictionary_lookup
from utils.MysqlPython import Mysql
# import dependencies.bmdca as bmdca


# Globals
# for checking out the options to read and write Rosetta runs to a relational DB such as MySQL
# https://new.rosettacommons.org/docs/latest/rosetta_basics/options/Database-options
logger = start_log(name=__name__)
index_offset = 1


class Database:  # Todo ensure that the single object is completely loaded before multiprocessing... Queues and whatnot
    def __init__(self, oriented, oriented_asu, refined, full_models, stride, sequences, hhblits_profiles, sql=None,
                 log=logger):
        if sql:
            raise DesignError('SQL set up has not been completed!')

        self.log = log
        self.oriented = DataStore(location=oriented, extension='.pdb*', sql=sql, log=log)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', sql=sql, log=log)
        self.refined = DataStore(location=refined, extension='.pdb', sql=sql, log=log)
        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', sql=sql, log=log)
        self.stride = DataStore(location=stride, extension='.stride', sql=sql, log=log)
        self.sequences = DataStore(location=sequences, extension='.fasta', sql=sql, log=log)
        self.alignments = DataStore(location=hhblits_profiles, extension='.sto', sql=sql, log=log)
        self.hhblits_profiles = DataStore(location=hhblits_profiles, extension='.hmm', sql=sql, log=log)
        # self.bmdca_fields = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_h_final.bin' % os.sep, sql=sql, log=log)
        # self.bmdca_couplings = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_J_final.bin' % os.sep, sql=sql, log=log)

    def load_all_data(self):
        """For every resource, acquire all existing data in memory"""
        #              self.oriented_asu, self.sequences,
        for source in [self.stride, self.alignments, self.hhblits_profiles, self.oriented, self.refined]:
            source.get_all_data()
        # self.log.debug('The data in the Database is: %s'
        #                % '\n'.join(str(store.__dict__) for store in self.__dict__.values()))

    def source(self, name):
        """Return on of the various DataStores supported by the Database"""
        try:
            return getattr(self, name)
        except AttributeError:
            raise AttributeError('There is no Database source named \'%s\' found. Possible sources are: %s'
                                 % (name, ', '.join(self.__dict__)))

    def retrieve_data(self, source=None, name=None):
        """Return the data requested if loaded into source Database, otherwise, load into the Database from the located
        file. Raise an error if source or file/SQL doesn't exist

        Keyword Args:

        Returns:
            (object): The object requested will depend on the source
        """
        object_db = self.source(source)
        data = getattr(object_db, name, None)
        if not data:
            object_db.name = object_db.load_data(name, log=None)
            data = object_db.name  # store the new data as an attribute

        return data

    def retrieve_file(self, from_source=None, name=None):
        """Retrieve the specified file on disk for subsequent parsing"""
        object_db = getattr(self, from_source, None)
        if not object_db:
            raise DesignError('There is no source named %s found in the Design Database' % from_source)

        return object_db.retrieve_file(name)


class DataStore:
    def __init__(self, location=None, extension='.txt', sql=None, log=logger):
        self.location = location
        self.extension = extension
        self.sql = sql
        self.log = log

        if '.pdb' in extension:
            self.load_file = PDB.from_file
        elif extension == '.fasta':
            self.load_file = read_fasta_file
        elif extension == '.stride':
            self.load_file = parse_stride
        elif extension == '.hmm':  # in ['.hmm', '.pssm']:
            self.load_file = parse_hhblits_pssm  # parse_pssm
        # elif extension == '.fasta' and msa:  # Todo if msa is in fasta format
        elif extension == '.sto':
            self.load_file = MultipleSequenceAlignment.from_stockholm  # parse_stockholm_to_msa
        elif extension == '_bmDCA%sparameters_h_final.bin' % os.sep:
            self.load_file = bmdca.load_fields
        elif extension == '_bmDCA%sparameters_J_final.bin' % os.sep:
            self.load_file = bmdca.load_couplings
        else:  # '.txt' read the file and return the lines
            self.load_file = self.read_file

    def store(self, name):
        """Return the path of the storage location given an entity name"""
        return os.path.join(self.location, '%s%s' % (name, self.extension))

    def retrieve_file(self, name):
        """Returns the actual location by combining the requested name with the stored .location"""
        path = os.path.join(self.location, '%s%s' % (name, self.extension))
        file_location = glob(path)
        if file_location:
            if len(file_location) > 1:
                self.log.error('Found more than one file at %s. Grabbing the first one: %s' % (path, file_location[0]))

            return file_location[0]
        else:
            self.log.error('No files found for \'%s\'. Attempting to incorporate into the Database' % path)

    def retrieve_data(self, name=None):
        """Return the data requested if loaded into source Database, otherwise, load into the Database from the located
        file. Raise an error if source or file/SQL doesn't exist

        Keyword Args:
            name=None (str): The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            (any[object, None]): If the data is available, the object requested will be returned, else None
        """
        data = getattr(self, name, None)
        if data:
            self.log.debug('Info %s%s was retrieved from DataStore' % (name, self.extension))
        else:
            setattr(self, name, self.load_data(name, log=None))  # attempt to store the new data as an attribute
            self.log.debug('Database file %s%s was loaded fresh' % (name, self.extension))  # not necessarily successful
            data = getattr(self, name)

        return data

    def load_data(self, name, **kwargs):
        """Return the data located in a particular entry specified by name"""
        if self.sql:
            dummy = True
        else:
            file = self.retrieve_file(name)
            if file:
                return self.load_file(file, **kwargs)
        return

    def get_all_data(self, **kwargs):
        """Return all data located in the particular DataStore storage location"""
        if self.sql:
            dummy = True
        else:
            for file in glob(os.path.join(self.location, '*%s' % self.extension)):
                data = self.load_file(file)
                setattr(self, os.path.splitext(os.path.basename(file))[0], data)

    @staticmethod
    def read_file(file, **kwargs):
        with open(file, 'r') as f:
            lines = f.readlines()

        return lines


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
    def __init__(self):
        self.monofrag_representatives_path = monofrag_cluster_rep_dirpath
        self.cluster_representatives_path = intfrag_cluster_rep_dirpath
        self.cluster_info_path = intfrag_cluster_info_dirpath
        self.reps = None
        self.paired_frags = None
        self.indexed_ghosts = {}
        self.info = None

    def get_monofrag_cluster_rep_dict(self):
        self.reps = {int(os.path.splitext(file)[0]):
                     PDB.from_file(os.path.join(root, file), solve_discrepancy=False, lazy=True, log=None)
                     for root, dirs, files in os.walk(self.monofrag_representatives_path) for file in files}

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
                    ijk_frag_cluster_rep_pdb = PDB.from_file(os.path.join(root, file), solve_discrepancy=False,
                                                             lazy=True, log=None)
                    # ijk_cluster_rep_mapped_chain = file[file.find('mappedchain') + 12:file.find('mappedchain') + 13]
                    ijk_cluster_rep_partner_chain = file[file.find('partnerchain') + 13:file.find('partnerchain') + 14]
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
            stacked_bb_coords = np.array([frag_pdb.chain(frag_paired_chain).get_backbone_coords()
                                          for j_dict in self.paired_frags[i_type].values()
                                          for frag_pdb, frag_paired_chain in j_dict.values()])
            stacked_guide_coords = np.array([frag_pdb.chain('9').coords for j_dict in self.paired_frags[i_type].values()
                                             for frag_pdb, _, in j_dict.values()])
            ijk_types = \
                np.array([(i_type, j_type, k_type) for j_type, j_dict in self.paired_frags[i_type].items()
                          for k_type in j_dict])
            # rmsd_array = np.array([self.info.cluster(type_set).rmsd for type_set in ijk_types])  # Todo
            rmsd_array = np.array([dictionary_lookup(self.info, type_set).rmsd for type_set in ijk_types])
            self.indexed_ghosts[i_type] = (stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array)


class FragmentDatabase(FragmentDB):
    def __init__(self, source='biological_interfaces', length=5, init_db=False, sql=False):
        super().__init__()  # FragmentDB
        # self.monofrag_representatives_path = monofrag_representatives_path
        # self.cluster_representatives_path
        # self.cluster_info_path = cluster_info_path
        # self.reps = None
        # self.paired_frags = None
        # self.info = None
        self.source = source
        self.location = frag_directory.get(source, None)  # Todo make dynamic upon unpickle and not loaded
        self.statistics = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
        self.fragment_range = None
        self.cluster_info = {}
        self.fragdb = None

        if sql:
            self.start_mysql_connection()
            self.db = True
        else:  # self.source == 'directory':
            # Todo initialize as local directory
            self.db = False
            if init_db:
                logger.info('Initializing FragmentDatabase from disk. This may take awhile...')
                self.get_monofrag_cluster_rep_dict()
                self.get_intfrag_cluster_rep_dict()
                self.get_intfrag_cluster_info_dict()
                # self.get_cluster_info()

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
            stats_file = glob(os.path.join(self.location, 'statistics.pkl'))
            if len(stats_file) == 1:
                self.statistics = unpickle(stats_file[0])
            else:
                raise DesignError('There were too many statistics.pkl files found from the fragment database source!')
            # for file in os.listdir(self.location):
            #     if 'statistics.pkl' in file:
            #         self.statistics = unpickle(os.path.join(self.location, file))
            #         return

    def get_db_aa_frequencies(self):
        """Retrieve database specific interface background AA frequencies

        Returns:
            (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    def retrieve_cluster_info(self, cluster=None, source=None, index=None):  # Todo rework this, below func for Database
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
                self.cluster_info[cluster_id] = unpickle(os.path.join(cluster_directory, '%s.pkl' % cluster_id))

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
