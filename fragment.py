from __future__ import annotations

import os
from typing import Annotated

import numpy as np

from PathUtils import intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, monofrag_cluster_rep_dirpath, \
    biological_interfaces, biological_fragment_db_pickle
import Pose
from Structure import Structure
from SymDesignUtils import dictionary_lookup, unpickle, start_log
from info import FragmentInfo

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


class FragmentDatabase(FragmentInfo):
    cluster_representatives_path: str
    cluster_info_path: str
    indexed_ghosts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | dict
    # dict[int, tuple[3x3, 1x3, tuple[int, int, int], float]]
    info: dict[int, dict[int, dict[int, ClusterInfoFile]]]
    # monofrag_representatives_path: str
    paired_frags: dict[int, dict[int, dict[int, tuple['Pose.Model', str]]]]
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
                             Structure.from_file(os.path.join(root, file), entities=False, log=None).ca_coords
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
                    ijk_frag_cluster_rep_pdb = Pose.Model.from_file(os.path.join(root, file), entities=False, log=None)
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
        """From the fragment database, precompute all required data into arrays to populate Ghost Fragments"""
        self.indexed_ghosts = {}
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
            rmsd_array = np.where(rmsd_array == 0, 0.0001, rmsd_array)  # Todo ensure rmsd rounded correct upon creation
            self.indexed_ghosts[i_type] = stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array


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
