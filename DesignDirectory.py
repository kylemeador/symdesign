import os
import copy
import re
from functools import wraps
# from math import ceil, sqrt
import shutil
from subprocess import Popen, list2cmdline
from glob import glob
from itertools import combinations, repeat  # chain as iter_chain
from typing import Union, Dict, List, Optional, Tuple, Callable, Any
# from textwrap import fill

import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
# from matplotlib.axes import Axes
# from mpl_toolkits.mplot3d import Axes3D
from Bio.Data.IUPACData import protein_letters_3to1, protein_letters_1to3
from cycler import cycler
from matplotlib.ticker import MultipleLocator
from pickle import UnpicklingError
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from Query.UniProt import is_uniprot_thermophilic
from Structure import Structure  # , Structures
from SymDesignUtils import unpickle, start_log, null_log, handle_errors, write_shell_script, DesignError, \
    match_score_from_z_value, pickle_object, filter_dictionary_keys, all_vs_all, \
    condensed_to_square, digit_translate_table, sym, index_intersection, z_score, large_color_array, starttime
# from Query import Flags
from CommandDistributer import reference_average_residue_weight, run_cmds, script_cmd, rosetta_flags, \
    rosetta_variables, relax_flags_cmdline
from PDB import PDB
from Pose import Pose, MultiModel, Models  # , Model
from DesignMetrics import read_scores, necessary_metrics, division_pairs, delta_pairs, \
    columns_to_new_column, unnecessary, rosetta_terms, dirty_hbond_processing, dirty_residue_processing, \
    mutation_conserved, per_res_metric, interface_composition_similarity, \
    significance_columns, df_permutation_test, clean_up_intermediate_columns, fragment_metric_template, \
    protocol_specific_columns, rank_dataframe_by_metric_weights, background_protocol, filter_df_for_index_by_value, \
    multiple_sequence_alignment_dependent_metrics, format_fragment_metrics, calculate_match_metrics, residue_processing
from SequenceProfile import parse_pssm, generate_mutations_from_reference, \
    simplify_mutation_dict, weave_sequence_dict, position_specific_jsd, sequence_difference, \
    jensen_shannon_divergence, hydrophobic_collapse_index, msa_from_dictionary  # multi_chain_alignment,
from classes.SymEntry import SymEntry, sdf_lookup, symmetry_factory
from utils.SymmetryUtils import identity_matrix, origin
from Database import FragmentDatabase, Database

# Globals
logger = start_log(name=__name__)
idx_offset = 1
design_directory_modes = [PUtils.interface_design, 'dock', 'filter']
cst_value = round(0.2 * reference_average_residue_weight, 2)
mean, std = 'mean', 'std'
stats_metrics = [mean, std]
residue_classificiation = ['core', 'rim', 'support']  # 'hot_spot'
errat_1_sigma, errat_2_sigma, errat_3_sigma = 5.76, 11.52, 17.28  # these are approximate magnitude of deviation
collapse_significance_threshold = 0.43


class JobResources:
    """The intention of JobResources is to serve as a singular source of design info which is common accross all
    designs. This includes common paths, databases, and design flags which should only be set once in program operation,
    then shared across all member designs"""
    def __init__(self, program_root, **kwargs):
        """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes"""
        if not os.path.exists(program_root):
            raise DesignError('Path does not exist!\n\t%s' % program_root)
        else:
            self.program_root = program_root
        self.protein_data = os.path.join(self.program_root, PUtils.data.title())
        self.pdbs = os.path.join(self.protein_data, 'PDBs')  # Used to store downloaded PDB's
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.full_model_dir = os.path.join(self.pdbs, 'full_models')
        self.stride_dir = os.path.join(self.pdbs, 'stride')
        self.sequence_info = os.path.join(self.protein_data, PUtils.sequence_info)
        self.sequences = os.path.join(self.sequence_info, 'sequences')
        self.profiles = os.path.join(self.sequence_info, 'profiles')
        # try:
        # if not self.projects:  # used for subclasses
        if not getattr(self, 'projects', None):  # used for subclasses
            self.projects = os.path.join(self.program_root, PUtils.projects)
        # except AttributeError:
        #     self.projects = os.path.join(self.program_root, PUtils.projects)
        self.clustered_poses = os.path.join(self.protein_data, 'ClusteredPoses')
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        # TODO ScoreDatabase integration
        self.all_scores = os.path.join(self.program_root, PUtils.all_scores)
        # self.design_db = None
        # self.score_db = None
        self.make_path(self.protein_data)
        self.make_path(self.job_paths)
        self.make_path(self.sbatch_scripts)
        # sequence database specific
        self.make_path(self.sequence_info)
        self.make_path(self.sequences)
        self.make_path(self.profiles)
        # structure database specific
        self.make_path(self.pdbs)
        self.make_path(self.orient_dir)
        self.make_path(self.orient_asu_dir)
        self.make_path(self.stride_dir)
        self.make_path(self.full_model_dir)
        self.reduce_memory = False
        self.resources = Database(self.orient_dir, self.orient_asu_dir, self.refine_dir, self.full_model_dir,
                                  self.stride_dir, self.sequences, self.profiles, sql=None)  # , log=logger)
        self.symmetry_factory = symmetry_factory
        self.fragment_db = None
        self.euler_lookup = None

    @staticmethod
    def make_path(path, condition=True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Keyword Args:
            condition=True (bool): A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(path, exist_ok=True)


# Todo move PDB coordinate information to Pose. Only use to handle Pose paths/options
class DesignDirectory:  # (JobResources):
    fragment_db: Optional[FragmentDatabase]

    def __init__(self, design_path, pose_id=None, root=None, **kwargs):
        #        project=None, specific_design=None, dock=False, construct_pose=False,
        self.job_resources = kwargs.get('job_resources', None)

        # DesignDirectory flags
        self.construct_pose = kwargs.get('construct_pose', False)
        self.debug = kwargs.get('debug', False)
        # self.dock = kwargs.get('dock', False)
        self.initialized = None
        self.log = None
        # self.master_db = kwargs.get('master_db', None)
        self.nanohedra_output = kwargs.get('nanohedra_output', False)
        if pose_id:
            self.directory_string_to_path(root, design_path)  # sets self.path
            self.source_path = self.path
        else:
            self.source_path = os.path.abspath(design_path)
        self.name = os.path.splitext(os.path.basename(self.source_path))[0]
        # DesignDirectory path attributes
        # self.scores = None  # /program_root/Projects/project_Designs/design/scores
        self.scores_file = None  # /program_root/Projects/project_Designs/design/data/name.sc
        self.data = None  # /program_root/Projects/project_Designs/design/data
        self.designs = None  # /program_root/Projects/project_Designs/design/designs
        # self.sdf_dir = None  # path/to/directory/sdf/
        self.frags = None  # /program_root/Projects/project_Designs/design/matching_fragments
        self.flags = None  # /program_root/Projects/project_Designs/design/scripts/flags
        self.frag_file = None  # /program_root/Projects/project_Designs/design/
        self.output_directory = kwargs.get('output_directory', None)
        self.pose_file = None  # /program_root/Projects/project_Designs/design/
        self.scripts = None  # /program_root/Projects/project_Designs/design/scripts
        self.serialized_info = None  # /program_root/Projects/project_Designs/design/data/info.pkl
        self.asu_path = None  # /program_root/Projects/project_Designs/design/design_name_clean_asu.pdb
        self.assembly_path = None  # /program_root/Projects/project_Designs/design/design_name_assembly.pdb
        self.refine_pdb = None
        # self._fragment_database = {}
        # self._evolutionary_profile = {}
        # self._design_profile = {}
        # self._fragment_data = {}
        # program_root/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_refine.pdb
        self.refined_pdb = None  # /program_root/Projects/project_Designs/design/design_name_refined.pdb
        self.scouted_pdb = None  # /program_root/Projects/project_Designs/design/designs/design_name_scouted.pdb
        self.consensus_pdb = None  # /program_root/Projects/project_Designs/design/design_name_for_consensus.pdb
        self.consensus_design_pdb = None  # /program_root/Projects/project_Designs/design/designs/design_name_for_consensus.pdb
        self.pdb_list = None  # /program_root/Projects/project_Designs/design/scripts/design_files.txt
        self.design_profile_file = None  # /program_root/Projects/project_Designs/design/data/design.pssm
        self.evolutionary_profile_file = None  # /program_root/Projects/project_Designs/design/data/evolutionary.pssm
        # self.fragment_data_pkl = None  # /program_root/Projects/project_Designs/design/data/%s_fragment_profile.pkl

        # Symmetry attributes
        # self._pose_transformation = {}  # dict[pdb# (1, 2)] = {'rotation': matrix, 'translation': vector}
        # self.cryst_record = None
        # self.expand_matrices = None
        self.modify_sym_energy = True if self.design_dimension in [2, 3] else False
        self.sym_def_file = None  # The symmetry definition file for the entire Pose
        if 'sym_entry' in kwargs:
            self.sym_entry = kwargs['sym_entry']
        # elif 'sym_entry_number' in kwargs:
        #     self.sym_entry = symmetry_factory(kwargs['sym_entry_specification'])
        # if symmetry:
        #     if symmetry == 'cryst':
        #         raise DesignError('This functionality is not possible yet. Please pass --symmetry by Symmetry Entry'
        #                           ' Number instead (See Laniado & Yeates, 2020).')
        #         cryst_record_d = PDB.get_cryst_record(
        #             self.source)  # Todo must get self.source before attempt this call
        #         self.sym_entry = space_group_to_sym_entry[cryst_record_d['space_group']]
        self.symmetry_protocol = None
        # self.uc_dimensions = None

        # Design flags  # Todo move to JobResources
        self.consensus = kwargs.get('consensus', None)  # Whether to run consensus or not
        self.design_residues = False  # (set[int])
        self.no_term_constraint = kwargs.get(PUtils.no_term_constraint, True)
        self.directives = kwargs.get('directives', {})
        self.no_evolution_constraint = kwargs.get(PUtils.no_evolution_constraint, True)
        # self.fragment_file = None
        # self.fragment_type = 'biological_interfaces'  # default for now, can be found in fragment_db
        self.force_flags = kwargs.get(PUtils.force_flags, False)
        self.fuse_chains = [tuple(pair.split(':')) for pair in kwargs.get('fuse_chains', [])]
        self.ignore_clashes = kwargs.get('ignore_clashes', False)
        self.increment_chains = kwargs.get('increment_chains', False)
        self.mpi = kwargs.get('mpi', False)
        self.no_hbnet = kwargs.get(PUtils.no_hbnet, False)
        self.number_of_trajectories = kwargs.get(PUtils.number_of_trajectories, PUtils.nstruct)
        self.output_assembly = kwargs.get('output_assembly', False)
        self.run_in_shell = kwargs.get('run_in_shell', False)
        self.pre_refine = False  # True
        self.query_fragments = kwargs.get(PUtils.generate_fragments, False)
        self.scout = kwargs.get(PUtils.scout, False)
        self.sequence_background = kwargs.get('sequence_background', False)
        self.specific_design = kwargs.get('specific_design', None)
        self.specific_protocol = kwargs.get('specific_protocol', False)
        self.structure_background = kwargs.get(PUtils.structure_background, False)
        self.write_frags = kwargs.get('write_fragments', True)
        self.write_oligomers = kwargs.get('write_oligomers', False)
        # Development Flags
        self.command_only = kwargs.get('command_only', False)  # Whether to reissue commands, only if run_in_shell=False
        self.development = kwargs.get('development', False)

        # Analysis flags
        self.skip_logging = kwargs.get('skip_logging', False)
        self.copy_nanohedra = kwargs.get('copy_nanohedra', False)  # no construction specific flags
        self.nanohedra_root = None

        # Design attributes
        self.composition = None  # building_blocks (4ftd_5tch)
        self.design_background = kwargs.get('design_background', PUtils.design_profile)  # by default, grab design profile
        self.design_residue_ids = {}  # {'interface1': '23A,45A,46A,...' , 'interface2': '234B,236B,239B,...'}
        self.design_selector = kwargs.get('design_selector', None)
        self.entity_names = []
        self.fragment_observations = None  # (dict): {'1_2_24': [(78, 87, ...), ...], ...}
        self.info = {}  # internal state info
        self._info = {}  # internal state info at load time
        self.init_pdb = None  # used if the design has never been initialized previously
        self.interface_residues = []
        # self.oligomer_names = []
        self.oligomers = []
        self.pose = None  # contains the design's Pose object
        self.pose_id = None
        self.source = None

        # Metric attributes TODO MOVE Metrics
        self.interface_ss_topology = {}  # {1: 'HHLH', 2: 'HSH'}
        self.interface_ss_fragment_topology = {}  # {1: 'HHH', 2: 'HH'}
        self.center_residue_numbers = []
        self.total_residue_numbers = []
        self.all_residue_score = None
        self.center_residue_score = None
        self.high_quality_int_residues_matched = None
        self.central_residues_with_fragment_overlap = None
        self.fragment_residues_total = None
        self.percent_overlapping_fragment = None
        self.multiple_frag_ratio = None
        self.helical_fragment_content = None
        self.strand_fragment_content = None
        self.coil_fragment_content = None
        self.ave_z = None
        self.total_interface_residues = None
        self.total_non_fragment_interface_residues = None
        self.percent_residues_fragment_total = None
        self.percent_residues_fragment_center = None

        if self.nanohedra_output:
            # source_path is design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2)
            if not os.path.exists(self.source_path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.source_path)
            # self.canonical_pdb1 = None  # canonical pdb orientation
            # self.canonical_pdb2 = None
            # self.rot_step_deg1 = None
            # self.rot_step_deg2 = None

            # if self.dock:  # Todo DockDirectory
            #     # Saves the path of the docking directory as DesignDirectory.path attribute. Try to populate further
            #     # using typical directory structuring
            #     # self.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implement?
            #     self.program_root = self.source_path  # Assuming that output directory (^ or v) of Nanohedra was passed
            #     # v for design_recap
            #     # self.program_root = glob(os.path.join(self.path, 'NanohedraEntry*DockedPoses%s'
            #     #                                                   % str(program_root or '')))
            #     # self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
            #     # self.log = [os.path.join(_sym, PUtils.master_log) for _sym in self.program_root]
            #     # for k, _sym in enumerate(self.program_root):
            #     # for k, _sym in enumerate(next(os.walk(self.program_root))):
            #     # self.building_blocks.append(list())
            #     # self.building_block_logs.append(list())
            #     # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
            #     # for bb_dir in next(os.walk(_sym))[1]:
            #     # v used in dock_dir set up
            #     self.building_block_logs = []
            #     self.building_block_dirs = []
            #     for bb_dir in next(os.walk(self.program_root))[1]:  # [1] grabs dirs from os.walk, yields only top level
            #         if os.path.exists(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils?
            #             self.building_block_dirs.append(bb_dir)
            #             # self.building_block_dirs[k].append(bb_dir)
            #             self.building_block_logs.append(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir))
            #             # self.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
            #
            #     # TODO generators for the various directory levels using the stored directory pieces
            #     def get_building_block_dir(self, building_block):
            #         for sym_idx, symm in enumerate(self.program_root):
            #             try:
            #                 bb_idx = self.building_block_dirs[sym_idx].index(building_block)
            #                 return os.path.join(self.program_root[sym_idx], self.building_block_dirs[sym_idx][bb_idx])
            #             except ValueError:
            #                 continue
            #         return
            # else:  # if self.construct_pose:
            self.initialized = False
            path_components = self.source_path.split(os.sep)
            self.nanohedra_root = os.sep.join(path_components[:-4])  # path_components[-5]
            # design_symmetry (P432)
            # path_components[-3] are the oligomeric names
            self.composition = self.source_path[:self.source_path.find(path_components[-3]) - 1]
            # design_symmetry/building_blocks (P432/4ftd_5tch)
            self.oligomer_names = list(map(str.lower, os.path.basename(self.composition).split('_')))
            self.entity_names = ['%s_1' % name for name in self.oligomer_names]  # assumes the entity is the first
            # self.pose_id = self.source_path[self.source_path.find(path_components[-3]) - 1:]\
            #     .replace(os.sep, '-')
            self.pose_id = '-'.join(path_components[-4:])  # [-5:-1] because of trailing os.sep
            self.name = self.pose_id
            if self.output_directory:
                # self.make_path(self.output_directory)
                self.program_root = self.output_directory  # os.getcwd()
                self.projects = ''
                self.project_designs = self.output_directory  # ''
                self.path = self.output_directory
                # ^ /output_directory<- self.path /design.pdb
            else:
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = \
                    os.path.join(self.projects, '%s_%s' % (path_components[-5], PUtils.design_directory))
                self.path = os.path.join(self.project_designs, self.name)
                self.make_path(self.program_root)
                self.make_path(self.projects)
                self.make_path(self.project_designs)
            # copy the master log
            if not os.path.exists(os.path.join(self.project_designs, PUtils.master_log)):
                shutil.copy(os.path.join(self.nanohedra_root, PUtils.master_log), self.project_designs)

            self.make_path(self.path, condition=(self.copy_nanohedra or self.construct_pose))

            if not self.construct_pose:  # no construction specific flags
                self.write_frags = False
        elif '.pdb' in self.source_path:  # Initial set up of directory -> /program_root/projects/project/design
            self.initialized = False
            if not os.path.exists(self.source_path):
                raise FileNotFoundError('The file "%s" couldn\'t be located! Ensure this location is correct.')
            self.source = self.source_path
            if self.output_directory:
                self.make_path(self.output_directory)
                self.program_root = self.output_directory  # os.getcwd()
                self.projects = ''
                self.project_designs = ''
                self.path = self.output_directory
                # ^ /output_directory<- self.path /design.pdb
            else:
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)  # symmetry.rstrip(os.sep)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = os.path.join(self.projects, '%s_%s' % (self.source_path.split(os.sep)[-2],
                                                                              PUtils.design_directory))
                self.path = os.path.join(self.project_designs, self.name)
                # ^ /program_root/projects/project/design<- self.path /design.pdb
                self.make_path(self.program_root)
                self.make_path(self.projects)
                self.make_path(self.project_designs)
                self.make_path(self.path)
                shutil.copy(self.source_path, self.path)
            # need to start here if I want to load pose through normal mechanism... ugh
            self.set_up_design_directory()
            if not self.entity_names:
                self.init_pdb = PDB.from_file(self.source_path, log=self.log, pose_format=False)
                self.entity_names = [entity.name for entity in self.init_pdb.entities]
                # self.load_pose()  # load the source pdb to find the entity_names
                # self.entity_names = [entity.name for entity in self.pose.entities]
            # self.set_up_design_directory()
        else:  # initialize DesignDirectory with existing /program_root/projects/project/design
            self.initialized = True
            self.path = self.source_path
            if not os.path.exists(self.path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.source_path)
            self.project_designs = os.path.dirname(self.path)
            self.projects = os.path.dirname(self.project_designs)
            self.program_root = os.path.dirname(self.projects)

    @classmethod
    def from_nanohedra(cls, design_path, project=None, **kwargs):
        return cls(design_path, nanohedra_output=True, project=project, **kwargs)

    @classmethod
    def from_file(cls, design_path, project=None, **kwargs):
        return cls(design_path, project=project, **kwargs)

    @classmethod
    def from_pose_id(cls, design_path, root=None, **kwargs):
        return cls(design_path, pose_id=True, root=root, **kwargs)

    # JobResources path attributes
    @property
    def all_scores(self):
        return self.job_resources.all_scores  # program_root/AllScores

    @property
    def clustered_poses(self):
        return self.job_resources.clustered_poses  # program_root/Data/ClusteredPoses

    @property
    def euler_lookup(self):
        return self.job_resources.euler_lookup

    @property
    def fragment_db(self) -> FragmentDatabase:
        """Returns the JobResource FragmentDatabase"""
        return self.job_resources.fragment_db

    @property
    def resources(self):
        return self.job_resources.resources

    @property
    def full_model_dir(self):
        return self.job_resources.full_model_dir  # program_root/Data/PDBs/full_models

    @property
    def job_paths(self):
        return self.job_resources.job_paths  # program_root/JobPaths

    @property
    def orient_dir(self):
        return self.job_resources.orient_dir  # program_root/Data/PDBs/oriented

    @property
    def orient_asu_dir(self):
        return self.job_resources.orient_asu_dir  # program_root/Data/PDBs/oriented_asu

    @property
    def pdbs(self):
        return self.job_resources.pdbs  # program_root/Data/PDBs

    @property
    def profiles(self):
        return self.job_resources.profiles  # program_root/SequenceInfo/profiles

    @property
    def protein_data(self):
        return self.job_resources.protein_data  # program_root/Data

    @property
    def reduce_memory(self):
        return self.job_resources.reduce_memory

    @property
    def refine_dir(self):
        return self.job_resources.refine_dir  # program_root/Data/PDBs/refined

    @property
    def stride_dir(self):
        return self.job_resources.stride_dir  # program_root/Data/PDBs/stride

    @property
    def sequence_info(self):
        return self.job_resources.sequence_info  # program_root/SequenceInfo

    @property
    def sequences(self):
        return self.job_resources.sequences  # program_root/SequenceInfo/sequences

    @property
    def sbatch_scripts(self):
        return self.job_resources.sbatch_scripts  # program_root/Scripts

    # @property
    # def design_sequences(self):
    #     return self.job_resources.design_sequences  # program_root/AllScores/str(self)_Sequences.pkl
    #
    # @property
    # def residues(self):
    #     return self.job_resources.residues  # program_root/AllScores/str(self)_Residues.csv
    #
    # @property
    # def trajectories(self):
    #     return self.job_resources.trajectories  # program_root/AllScores/str(self)_Trajectories.csv

    # SymEntry object attributes
    @property
    def sym_entry(self) -> Optional[SymEntry]:
        """The SymEntry"""
        try:
            return self._sym_entry
        except AttributeError:
            self._sym_entry = symmetry_factory(*self.info['sym_entry_specification']) \
                if 'sym_entry_specification' in self.info else None
            return self._sym_entry

    @sym_entry.setter
    def sym_entry(self, sym_entry: SymEntry):
            self._sym_entry = sym_entry

    @property
    def symmetric(self) -> bool:
        """Is the DesignDirectory symmetric?"""
        return self.sym_entry is not None

    @property
    def design_symmetry(self) -> Optional[str]:
        """The result of the SymEntry"""
        try:
            return self.sym_entry.resulting_symmetry
        except AttributeError:
            return

    @property
    def sym_entry_number(self) -> Optional[int]:
        """The entry number of the SymEntry"""
        try:
            return self.sym_entry.entry_number
        except AttributeError:
            return

    @property
    def sym_entry_map(self) -> Optional[str]:
        """The symmetry map of the SymEntry"""
        try:
            return self.sym_entry.sym_map
        except AttributeError:
            return

    # @property
    # def sym_entry_combination(self) -> Optional[str]:
    #     """The combination string of the SymEntry"""
    #     try:
    #         return self.sym_entry.combination_string
    #     except AttributeError:
    #         return

    @property
    def design_dimension(self) -> Optional[int]:
        """The dimension of the SymEntry"""
        try:
            return self.sym_entry.dimension
        except AttributeError:
            return

    @property
    def number_of_symmetry_mates(self) -> Optional[int]:
        """The number of symmetric copies in the full symmetric system"""
        try:
            return self.sym_entry.number_of_operations
        except AttributeError:
            return

    @property
    def trajectories(self) -> Union[str, bytes]:
        return os.path.join(self.all_scores, '%s_Trajectories.csv' % self.__str__())

    @property
    def residues(self) -> Union[str, bytes]:
        return os.path.join(self.all_scores, '%s_Residues.csv' % self.__str__())

    @property
    def design_sequences(self) -> Union[str, bytes]:
        return os.path.join(self.all_scores, '%s_Sequences.pkl' % self.__str__())

    # SequenceProfile based attributes
    @property
    def design_background(self) -> Dict:
        """Return the amino acid frequencies utilized as the DesignDirectory background frequencies"""
        try:
            return getattr(self, self._design_background)
        except AttributeError:
            self.log.error('For the design_background, couldn\'t locate the background profile "%s". '
                           'By default, using design_profile instead' % self._design_background)
            return self.design_profile

    @design_background.setter
    def design_background(self, background: str):
        self._design_background = background

    @property
    def design_profile(self) -> Dict:
        """Returns the amino acid frequencies observed for each residue's design profile which is a specific mix of
        evolution and fragment frequency information"""
        try:
            return self._design_profile
        except AttributeError:
            try:
                self._design_profile = parse_pssm(self.design_profile_file)
            except FileNotFoundError:
                self._design_profile = {}
            return self._design_profile

    @property
    def evolutionary_profile(self) -> Dict:
        """Returns the amino acid frequencies observed for each residue's evolutionary alignment"""
        try:
            return self._evolutionary_profile
        except AttributeError:
            try:
                self._evolutionary_profile = parse_pssm(self.evolutionary_profile_file)
            except FileNotFoundError:
                self._evolutionary_profile = {}
            return self._evolutionary_profile

    @property
    def fragment_profile(self) -> Dict:
        """Returns the amino acid frequencies observed for each residue's fragment observations"""
        try:
            return self._fragment_profile
        except AttributeError:
            try:
                self._fragment_profile = parse_pssm(self.fragment_profile_file)
            except FileNotFoundError:
                self._fragment_profile = {}
            return self._fragment_profile

    @property
    def fragment_data(self) -> Dict:
        """Returns only the entries in the fragment_profile which are populated"""
        try:
            return self._fragment_data
        except AttributeError:
            try:
                frag_pkl = os.path.join(self.data, '%s_%s.pkl' % (self.fragment_source, PUtils.fragment_profile))
                self._fragment_data = self.info['fragment_data'] if 'fragment_data' in self.info else unpickle(frag_pkl)
            except FileNotFoundError:
                # fragment_profile is removed of all entries that are not fragment populated.
                self._fragment_data = {residue: data for residue, data in self.fragment_profile.items()
                                       if data.get('stats', (None,))[0]}  # [0] must contain a fragment observation
                # self._fragment_data = None
            return self._fragment_data

    @property
    def fragment_source(self) -> str:
        """The identity of the fragment database used in design fragment decoration"""
        try:
            return self._fragment_source
        except AttributeError:
            # self._fragment_database = self.info.get('fragment_database')
            try:
                self._fragment_source = self.info['fragment_source'] if 'fragment_source' in self.info \
                    else self.fragment_db.source
            except AttributeError:  # there is not fragment_db attached
                self._fragment_source = None
            return self._fragment_source

    @property
    def number_of_fragments(self) -> int:
        return len(self.fragment_observations) if self.fragment_observations else 0

    # @property
    # def score(self) -> float:
    #     """The central Nanohedra score (derived from reported in Laniado, Meador, & Yeates, PEDS. 2021)"""
    #     try:
    #         self.get_fragment_metrics()
    #         return self.center_residue_score / self.central_residues_with_fragment_overlap
    #     except ZeroDivisionError:
    #         self.log.error('No fragment information found! Fragment scoring unavailable.')
    #         return 0.

    # def pose_score(self) -> float:
    #     """The Nanohedra score as reported in Laniado, Meador, & Yeates, PEDS. 2021"""
    #     return self.all_residue_score

    def pose_metrics(self) -> Dict:
        """Gather all metrics relating to the Pose and the interfaces within the Pose

        Returns:
            {'nanohedra_score_normalized': , 'nanohedra_score_center_normalized':,
             'nanohedra_score': , 'nanohedra_score_center': , 'number_fragment_residues_total': ,
             'number_fragment_residues_center': , 'multiple_fragment_ratio': ,
             'percent_fragment_helix': , 'percent_fragment_strand': ,
             'percent_fragment_coil': , 'number_of_fragments': , 'total_interface_residues': ,
             'percent_residues_fragment_total': , 'percent_residues_fragment_center': }
        """
        self.get_fragment_metrics()
        # Interface B Factor TODO ensure asu.pdb has B-factors for Nanohedra
        int_b_factor = sum(self.pose.pdb.residue(residue_number).b_factor for residue_number in self.interface_residues)
        metrics = {
            'interface_b_factor_per_residue': round(int_b_factor / self.total_interface_residues, 2),
            'nanohedra_score': self.all_residue_score,
            'nanohedra_score_normalized': self.all_residue_score / self.fragment_residues_total
            if self.fragment_residues_total else 0.0,
            'nanohedra_score_center': self.center_residue_score,
            'nanohedra_score_center_normalized': self.center_residue_score / self.central_residues_with_fragment_overlap
            if self.central_residues_with_fragment_overlap else 0.0,
            'number_fragment_residues_total': self.fragment_residues_total,
            'number_fragment_residues_center': self.central_residues_with_fragment_overlap,
            'multiple_fragment_ratio': self.multiple_frag_ratio,
            'percent_fragment': self.fragment_residues_total / self.total_interface_residues,
            'percent_fragment_helix': self.helical_fragment_content,
            'percent_fragment_strand': self.strand_fragment_content,
            'percent_fragment_coil': self.coil_fragment_content,
            'number_of_fragments': self.number_of_fragments,
            'total_interface_residues': self.total_interface_residues,
            'total_non_fragment_interface_residues': self.total_non_fragment_interface_residues,
            'percent_residues_fragment_total': self.percent_residues_fragment_total,
            'percent_residues_fragment_center': self.percent_residues_fragment_center}

        # pose.secondary_structure attributes are set in the get_fragment_metrics() call
        for number, elements in self.pose.split_interface_ss_elements.items():
            self.interface_ss_topology[number] = ''.join(self.pose.ss_type_array[element] for element in set(elements))
        total_fragment_elements, total_interface_elements = '', ''
        for number, topology in self.interface_ss_topology.items():
            metrics['interface_secondary_structure_topology_%d' % number] = topology
            total_interface_elements += topology
            metrics['interface_secondary_structure_fragment_topology_%d' % number] = \
                self.interface_ss_fragment_topology.get(number, '-')
            total_fragment_elements += self.interface_ss_fragment_topology.get(number, '')

        metrics['interface_secondary_structure_fragment_topology'] = total_fragment_elements
        metrics['interface_secondary_structure_fragment_count'] = len(total_fragment_elements)
        metrics['interface_secondary_structure_topology'] = total_interface_elements
        metrics['interface_secondary_structure_count'] = len(total_interface_elements)

        if self.symmetric:
            metrics['design_dimension'] = self.design_dimension
            for idx, group in enumerate(self.sym_entry.groups, 1):
                metrics['symmetry_group_%d' % idx] = group
        else:
            metrics['design_dimension'] = 'asymmetric'

        total_residue_counts = []
        minimum_radius, maximum_radius = float('inf'), 0
        for ent_idx, entity in enumerate(self.pose.entities, 1):
            ent_com = entity.distance_to_reference()
            min_rad = entity.distance_to_reference(measure='min')
            max_rad = entity.distance_to_reference(measure='max')
            # distances.append(np.array([ent_idx, ent_com, min_rad, max_rad]))
            # distances[entity] = np.array([ent_idx, ent_com, min_rad, max_rad, entity.number_of_residues])
            total_residue_counts.append(entity.number_of_residues)
            metrics.update({
                'entity_%d_symmetry' % ent_idx: entity.symmetry if entity.is_oligomeric else 'monomer',
                'entity_%d_name' % ent_idx: entity.name,
                'entity_%d_number_of_residues' % ent_idx: entity.number_of_residues,
                'entity_%d_radius' % ent_idx: ent_com,
                'entity_%d_min_radius' % ent_idx: min_rad, 'entity_%d_max_radius' % ent_idx: max_rad,
                'entity_%d_n_terminal_helix' % ent_idx: entity.is_termini_helical(),
                'entity_%d_c_terminal_helix' % ent_idx: entity.is_termini_helical(termini='c'),
                'entity_%d_n_terminal_orientation' % ent_idx: entity.termini_proximity_from_reference(),
                'entity_%d_c_terminal_orientation' % ent_idx: entity.termini_proximity_from_reference(termini='c'),
                'entity_%d_thermophile' % ent_idx: is_uniprot_thermophilic(entity.uniprot_id)})
            if min_rad < minimum_radius:
                minimum_radius = min_rad
            if max_rad > maximum_radius:
                maximum_radius = max_rad

        metrics['entity_minimum_radius'] = minimum_radius
        metrics['entity_maximum_radius'] = maximum_radius
        metrics['entity_residue_length_total'] = sum(total_residue_counts)
        # # for distances1, distances2 in combinations(distances, 2):
        # for entity1, entity2 in combinations(self.pose.entities, 2):
        #     # entity_indices = (int(distances1[0]), int(distances2[0]))  # this is a sloppy conversion rn, but oh well
        #     entity_indices = (int(distances[entity1][0]), int(distances[entity2][0]))  # this is kinda sloppy conversion
        #     # entity_ratios = distances1 / distances2
        #     entity_ratios = distances[entity1] / distances[entity2]
        #     metrics.update({'entity_radius_ratio_%dv%d' % entity_indices: entity_ratios[1],
        #                     'entity_min_radius_ratio_%dv%d' % entity_indices: entity_ratios[2],
        #                     'entity_max_radius_ratio_%dv%d' % entity_indices: entity_ratios[3],
        #                     'entity_number_of_residues_ratio_%dv%d' % entity_indices:
        #                         residue_counts[entity_indices[0]] / residue_counts[entity_indices[1]]})

        radius_ratio_sum, min_ratio_sum, max_ratio_sum, residue_ratio_sum = 0, 0, 0, 0
        for counter, (entity_idx1, entity_idx2) in enumerate(combinations(range(1, len(self.pose.entities) + 1), 2), 1):
            radius_ratio = metrics['entity_%d_radius' % entity_idx1] / metrics['entity_%d_radius' % entity_idx2]
            min_ratio = metrics['entity_%d_min_radius' % entity_idx1] / metrics['entity_%d_min_radius' % entity_idx2]
            max_ratio = metrics['entity_%d_max_radius' % entity_idx1] / metrics['entity_%d_max_radius' % entity_idx2]
            residue_ratio = metrics['entity_%d_number_of_residues' % entity_idx1] / metrics['entity_%d_number_of_residues' % entity_idx2]
            radius_ratio_sum += abs(1 - radius_ratio)
            min_ratio_sum += abs(1 - min_ratio)
            max_ratio_sum += abs(1 - max_ratio)
            residue_ratio_sum += abs(1 - residue_ratio)
            metrics.update({'entity_radius_ratio_%dv%d' % (entity_idx1, entity_idx2): radius_ratio,
                            'entity_min_radius_ratio_%dv%d' % (entity_idx1, entity_idx2): min_ratio,
                            'entity_max_radius_ratio_%dv%d' % (entity_idx1, entity_idx2): max_ratio,
                            'entity_number_of_residues_ratio_%dv%d' % (entity_idx1, entity_idx2): residue_ratio})
        metrics.update({'entity_radius_average_deviation': radius_ratio_sum / counter,
                        'entity_min_radius_average_deviation': min_ratio_sum / counter,
                        'entity_max_radius_average_deviation': max_ratio_sum / counter,
                        'entity_number_of_residues_average_deviation': residue_ratio_sum / counter})
        return metrics

    def return_termini_accessibility(self, entity=None, report_if_helix=False):
        """Return the termini which are not buried in the Pose

        Keyword Args:
            entity=None (Structure): The Structure to query which originates in the pose
            report_if_helix=False (bool): Whether the query should additionally report on the helicity of the termini
        Returns:
            (dict): {'n': True, 'c': False}
        """
        # self.pose.get_assembly_symmetry_mates()
        if not self.pose.assembly.sasa:
            # self.pose.assembly.write(os.path.join(self.data, 'ASSEMBLY_DEBUG.pdb'))
            self.pose.assembly.get_sasa()

        # self.pose.assembly.write(out_path=os.path.join(self.path, 'POSE_ASSEMBLY.pdb'))
        entity_chain = self.pose.assembly.chain(entity.chain_id)
        n_term, c_term = False, False
        # Todo add reference when in a crystalline environment  # reference=pose_transformation[idx].get('translation2')
        n_termini_orientation = entity.termini_proximity_from_reference()
        if n_termini_orientation == 1:  # if outward
            if entity_chain.n_terminal_residue.relative_sasa > 0.25:
                n_term = True
        # Todo add reference when in a crystalline environment
        c_termini_orientation = entity.termini_proximity_from_reference(termini='c')
        if c_termini_orientation == 1:  # if outward
            if entity_chain.c_terminal_residue.relative_sasa > 0.25:
                c_term = True

        if report_if_helix:
            if self.resources:
                parsed_secondary_structure = self.resources.stride.retrieve_data(name=entity.name)
                if parsed_secondary_structure:
                    entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                else:
                    entity.stride()
            n_term = True if n_term and entity.is_termini_helical() else False
            c_term = True if c_term and entity.is_termini_helical(termini='c') else False

        return {'n': n_term, 'c': c_term}

    # def clear_pose_transformation(self):
    #     """Remove any pose transformation data from the Pose"""
    #     try:
    #         del self._pose_transformation
    #         self.info.pop('pose_transformation')
    #     except AttributeError:
    #         pass

    @property
    def pose_transformation(self) -> List[Dict[str, np.ndarray]]:
        """Provide the transformation parameters for the design in question

        Returns:
            [{'rotation': numpy.ndarray, 'translation': numpy.ndarray, 'rotation2': numpy.ndarray,
              'translation2': numpy.ndarray}, ...]
        """
        try:
            return self._pose_transformation
        except AttributeError:
            try:
                self._pose_transformation = self.retrieve_pose_transformation_from_file()
            except FileNotFoundError:
                try:
                    self._pose_transformation = self.pose.assign_pose_transformation()
                except DesignError:
                    # Todo
                    #  This must be something outside of the realm of possibilities of Nanohedra symmetry groups
                    #  Perhaps we need to get the parameters for oligomer generation from PISA or other operator source
                    self.log.critical('There was no pose transformation file specified at %s and no transformation '
                                      'found from routine search of Nanohedra docking parameters. Is this pose from the'
                                      ' PDB? You may need to utilize PISA to accurately deduce the locations of pose '
                                      'transformations to make the correct oligomers. For now, using null '
                                      'transformation which is likely not what you want...' % self.pose_file)
                    self._pose_transformation = \
                        [dict(rotation=identity_matrix, translation=None) for _ in self.pose.entities]
                # raise FileNotFoundError('There was no pose transformation file specified at %s' % self.pose_file)
            self.info['pose_transformation'] = self._pose_transformation
            # self.log.debug('Using transformation parameters:\n\t%s'
            #                % '\n\t'.join(pretty_format_table(self._pose_transformation.items())))
            return self._pose_transformation

    @pose_transformation.setter
    def pose_transformation(self, transform):
        if isinstance(transform, list):
            self._pose_transformation = transform
            self.info['pose_transformation'] = self._pose_transformation
        else:
            raise ValueError('The attribute pose_transformation must be a list, not %s' % type(transform))

    # def rotation_parameters(self):
    #     return self.rot_range_deg_pdb1, self.rot_range_deg_pdb2, self.rot_step_deg1, self.rot_step_deg2
    #
    # def degeneracy_parameters(self):
    #     return self.sym_entry.degeneracy_matrices1, self.sym_entry.degeneracy_matrices2
    #
    # def degen_and_rotation_parameters(self):
    #     return self.degeneracy_parameters(), self.rotation_parameters()
    #
    # def compute_last_rotation_state(self):
    #     number_steps1 = self.rot_range_deg_pdb1 / self.rot_step_deg1
    #     number_steps2 = self.rot_range_deg_pdb2 / self.rot_step_deg1
    #
    #     return int(number_steps1), int(number_steps2)
    #
    # def return_symmetry_parameters(self):
    #     return dict(symmetry=self.design_symmetry, design_dimension=self.design_dimension,
    #                 uc_dimensions=self.uc_dimensions, expand_matrices=self.expand_matrices)

    # Decorator static methods: These must be declared above their usage, but made static after each declaration
    def handle_design_errors(errors: Tuple = (Exception,)) -> Callable:
        """Decorator to wrap a method with try: ... except errors: and log errors to the DesignDirectory

        Args:
            errors: A tuple of exceptions to monitor. Must be a tuple even if single exception
        Returns:
            Function return upon proper execution, else is error if exception raised, else None
        """
        def wrapper(func: Callable) -> Any:
            @wraps(func)
            def wrapped(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except errors as error:
                    self.log.error(error)  # Allows exception reporting using self.log
                    # self.info['error'] = error  # include the error code in the design state
                    return error
            return wrapped
        return wrapper

    def close_logs(func):
        """Decorator to close the instance log file after use in an instance method (protocol)"""
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            func_return = func(self, *args, **kwargs)
            # adapted from https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
            for handler in self.log.handlers:
                handler.close()
            return func_return
        return wrapped

    def remove_structure_memory(func):
        """Decorator to remove large memory attributes from the instance after processing is complete"""
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            func_return = func(self, *args, **kwargs)
            if self.reduce_memory:
                self.pose = None
                self.oligomers.clear()
            return func_return
        return wrapped

    def start_log(self, level: int = 2) -> None:
        """Initialize the logger for the Pose"""
        if self.log:
            return

        if self.debug:
            handler, level = 1, 1  # defaults to stdout, debug is level 1
            propagate, no_log_name = False, False
        else:
            handler = 2  # to a file
            propagate, no_log_name = True, True

        if self.skip_logging:  # set up null_logger
            self.log = null_log
        elif self.nanohedra_output and not self.construct_pose:
            self.log = start_log(name=str(self), handler=3, propagate=True, no_log_name=no_log_name)
        else:
            self.log = start_log(name=str(self), handler=handler, level=level,
                                 location=os.path.join(self.path, self.name), propagate=propagate,
                                 no_log_name=no_log_name)

    def directory_string_to_path(self, root: Union[str, bytes], pose_id: str):
        """Set the DesignDirectory self.path to the root/pose-ID where the pose-ID is converted from dash separation to
         path separators"""
        assert root, 'No program directory attribute set! Cannot create a path from a pose_id without a root directory!' \
                     ' Pass both -f with the pose_id\'s and -d with the specified directory'
        if self.nanohedra_output:
            self.path = os.path.join(root, pose_id.replace('-', os.sep))
        else:
            self.path = os.path.join(root, 'Projects', pose_id.replace('_Designs-', '_Designs%s' % os.sep))

    # def link_master_directory(self, master_db=None):  # UNUSED. Could be useful in case where root is unknown
    #     """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes
    #
    #     Keyword Args:
    #         master_db=None (JobResources):
    #     """
    #     # if not os.path.exists(self.program_root):
    #     #     raise DesignError('Path does not exist!\n\t%s' % self.program_root)
    #     if master_db:
    #         self.master_db = master_db
    #
    #     if self.master_db:
    #         self.protein_data = self.master_db.protein_data
    #         self.pdbs = self.master_db.pdbs
    #         self.orient_dir = self.master_db.orient_dir
    #         self.orient_asu_dir = self.master_db.orient_asu_dir
    #         self.refine_dir = self.master_db.refine_dir
    #         self.full_model_dir = self.master_db.full_model_dir
    #         self.stride_dir = self.master_db.stride_dir
    #         # self.sdf_dir = self.master_db.sdf_dir
    #         self.sequence_info = self.master_db.sequence_info
    #         self.sequences = self.master_db.sequences
    #         self.profiles = self.master_db.profiles
    #         self.clustered_poses = self.master_db.clustered_poses
    #         self.job_paths = self.master_db.job_paths
    #         self.sbatch_scripts = self.master_db.sbatch_scripts
    #         self.all_scores = self.master_db.all_scores
    #
    # def link_database(self, resource_db=None, fragment_db=None, design_db=None, score_db=None):
    #     """Connect the design to the master Database object to fetch shared resources"""
    #     if resource_db:
    #         self.resources = resource_db
    #         if self.pose:
    #             self.pose.source_db = resource_db
    #     if fragment_db:
    #         self.fragment_db = fragment_db
    #         if self.pose:
    #             self.pose.fragment_db = fragment_db
    #     # if design_db and isinstance(design_db, FragmentDatabase):
    #     #     self.design_db = design_db
    #     # if score_db and isinstance(score_db, FragmentDatabase):
    #     #     self.score_db = score_db

    @handle_design_errors(errors=(DesignError, ))
    @close_logs
    def set_up_design_directory(self, pre_refine=None, pre_loop_model=None):
        """Prepare output Directory and File locations. Each DesignDirectory always includes this format

        Keyword Args:
            pre_refine=None (Union[None, bool]): Whether the Pose has been refined previously (before loading)
            pre_loop_model=None (Union[None, bool]): Whether the Pose had loops modeled previously (before loading)
        """
        # self.make_path(self.path, condition=(not self.nano or self.copy_nanohedra or self.construct_pose))
        self.start_log()
        # self.scores = os.path.join(self.path, PUtils.scores_outdir)
        # Todo if I use this for design, it opens up a can of worms. Maybe it is better to include only this identifier
        #  for specific modules
        self.output_identifier = '%s_' % self.name if self.output_directory else ''
        self.designs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.scripts = os.path.join(self.path, '%s%s' % (self.output_identifier, PUtils.scripts))
        self.frags = os.path.join(self.path, '%s%s' % (self.output_identifier, PUtils.frag_dir))
        self.flags = os.path.join(self.scripts, 'flags')  # '%sflags' % self.output_identifier)
        self.data = os.path.join(self.path, '%s%s' % (self.output_identifier, PUtils.data))
        self.scores_file = os.path.join(self.data, '%s.sc' % self.name)
        self.serialized_info = os.path.join(self.data, 'info.pkl')  # '%sinfo.pkl' % self.output_identifier)
        self.asu_path = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean_asu))
        self.assembly_path = os.path.join(self.path, '%s_%s' % (self.name, PUtils.assembly))
        self.refine_pdb = '%s_for_refine.pdb' % os.path.splitext(self.asu_path)[0]
        self.consensus_pdb = '%s_for_consensus.pdb' % os.path.splitext(self.asu_path)[0]
        self.consensus_design_pdb = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        self.pdb_list = os.path.join(self.scripts, 'design_files.txt')
        if self.nanohedra_output:
            self.pose_file = os.path.join(self.source_path, PUtils.pose_file)
            self.frag_file = os.path.join(self.source_path, PUtils.frag_dir, PUtils.frag_text_file)
            if self.copy_nanohedra:  # copy nanohedra output directory to the design directory
                if os.path.exists(self.frags):  # only present if a copy or generate fragments command is issued
                    raise DesignError('The directory %s already exists! Can\'t complete set up without an overwrite!')
                shutil.copytree(self.source_path, self.path)
            elif self.construct_pose:
                if not os.path.exists(os.path.join(self.path, PUtils.pose_file)):
                    shutil.copy(self.pose_file, self.path)
                    shutil.copy(self.frag_file, self.path)
                self.info['nanohedra'] = True
                self.info['sym_entry_specification'] = self.sym_entry_number, self.sym_entry_map
                # self.info['sym_entry'] = self.sym_entry
                self.info['oligomer_names'] = self.oligomer_names
                self.pose_transformation = self.retrieve_pose_metrics_from_file()
                # self.entity_names = ['%s_1' % name for name in self.oligomer_names]  # this assumes the entity is the first
                self.info['entity_names'] = self.entity_names  # Todo remove after T33
                # self.info['pre_refine'] = self.pre_refine  # Todo remove after T33
                self.pickle_info()  # save this info on the first copy so that we don't have to construct again
        else:
            self.pose_file = os.path.join(self.path, PUtils.pose_file)
            self.frag_file = os.path.join(self.frags, PUtils.frag_text_file)
            if os.path.exists(self.serialized_info):  # Pose has already been processed, gather state data
                try:
                    self.info = unpickle(self.serialized_info)
                except UnpicklingError as error:  # pickle.UnpicklingError:
                    self.log.error('%s: There was an issue retrieving design state from binary file...' % self.name)
                    raise error
                except ModuleNotFoundError as error:
                    self.log.error('%s: There was an issue retrieving design state from binary file...' % self.name)
                    self.log.critical('Removing %s' % self.serialized_info)
                    # raise error
                    os.remove(self.serialized_info)
                if os.stat(self.serialized_info).st_size > 10000:
                    print('Found pickled file with huge size %d. fragment_database being removed'
                          % os.stat(self.serialized_info).st_size)
                    self.info['fragment_source'] = \
                        getattr(self.info.get('fragment_database'), 'source', 'biological_interfaces')
                    self.pickle_info()  # save immediately so we don't have this issue with reading again!
                self._info = self.info.copy()  # create a copy of the state upon initialization
                self.pre_refine = self.info.get('pre_refine', False)  # default value is False unless set to True
                self.fragment_observations = self.info.get('fragments', None)  # None signifies query wasn't attempted
                self.entity_names = self.info.get('entity_names', [])
                self.oligomer_names = self.info.get('oligomer_names', [])
                # These statements are a temporary patch Todo remove for SymDesign master branch
                if not self.sym_entry:  # none was provided at initiation or in state
                    if 'sym_entry' in self.info:
                        self.sym_entry = self.info['sym_entry']  # get instance
                        self.info.pop('sym_entry')  # remove this object
                        self.info['sym_entry_specification'] = self.sym_entry_number, self.sym_entry_map
                if not self.oligomer_names:
                    self.oligomer_names = self.info.get('entity_names', [])
                if 'design_residue_ids' in self.info:  # format is modern
                    self.design_residue_ids = self.info.get('design_residue_ids', {})
                    self.interface_residues = self.info.get('interface_residues', False)
                else:  # format is old, convert
                    self.design_residue_ids = self.info.get('interface_residues', {})
                    self.interface_residues = self.info.get('interface_residues', False)
                if 'fragment_database' in self.info:
                    self.info['fragment_source'] = self.info.get('fragment_database')
                    self.info.pop('fragment_database')
                fragment_data = self.info.get('fragment_data')
                if fragment_data and not isinstance(fragment_data, dict):  # this is a .pkl file
                    try:
                        self.info['fragment_data'] = unpickle(fragment_data)
                        os.remove(fragment_data)
                    except FileNotFoundError:
                        self.info.pop('fragment_data')
                self._pose_transformation = self.info.get('pose_transformation', None)
                if isinstance(self._pose_transformation, dict):  # old format
                    del self._pose_transformation
                # End temporary patch
                self.design_residues = self.info.get('design_residues', False)  # (set[int])
                if isinstance(self.design_residues, str):  # Todo remove as this conversion updates old directories
                    # 'design_residues' coming in as 234B (residue_number|chain), remove chain, change type to int
                    self.design_residues = \
                        set(int(res.translate(digit_translate_table)) for res in self.design_residues.split(','))
            else:  # we are constructing for the first time. Save all relevant information
                self.info['sym_entry'] = self.sym_entry
                self.info['entity_names'] = self.entity_names
                # self.pickle_info()  # save this info on the first copy so that we don't have to construct again

        if pre_refine is not None:
            self.pre_refine = pre_refine
        # check if the source of the pdb files was refined upon loading
        if self.pre_refine:
            self.refined_pdb = self.asu_path
            self.scouted_pdb = \
                '%s_scout.pdb' % os.path.join(self.designs, os.path.basename(os.path.splitext(self.refined_pdb)[0]))
        else:
            self.refined_pdb = os.path.join(self.designs, os.path.basename(self.refine_pdb))
            self.scouted_pdb = '%s_scout.pdb' % os.path.splitext(self.refined_pdb)[0]

        # configure standard pose loading mechanism with self.source
        if self.specific_design:
            matching_designs = sorted(glob(os.path.join(self.designs, '*%s.pdb' % self.specific_design)))
            self.specific_design = self.name + '_' + self.specific_design
            if matching_designs:
                for matching_design in matching_designs:
                    if os.path.exists(matching_design):
                        self.specific_design_path = matching_design
                        break
                if len(matching_designs) > 1:
                    self.log.warning('Found %d matching designs to your specified design, choosing the first %s'
                                     % (len(matching_designs), matching_designs[0]))
            else:
                raise DesignError('Couldn\'t locate a design matching the specific_design name %s'
                                  % self.specific_design)

            self.source = self.specific_design_path
        elif not self.source:
            if os.path.exists(self.asu_path):  # standard mechanism of loading the pose
                self.source = self.asu_path
            else:
                try:
                    self.source = sorted(glob(os.path.join(self.path, '%s.pdb' % self.name)))[0]
                except IndexError:  # glob found no files
                    self.source = None
        else:  # if the DesignDirectory is loaded as .pdb, the source should be loaded already
            # self.source = self.init_pdb
            pass

        # design specific files
        self.design_profile_file = os.path.join(self.data, 'design.pssm')  # os.path.abspath(self.path), 'data'
        self.evolutionary_profile_file = os.path.join(self.data, 'evolutionary.pssm')
        self.fragment_profile_file = os.path.join(self.data, 'fragment.pssm')
        # self.fragment_data_pkl = os.path.join(self.data, '%s_%s.pkl' % (self.fragment_source, PUtils.fragment_profile))

    def get_wildtype_file(self) -> Union[str, bytes]:
        """Retrieve the wild-type file name from DesignDirectory"""
        wt_file = glob(self.asu_path)
        assert len(wt_file) == 1, 'More than one matching file found during search %s' % self.asu_path

        return wt_file[0]

    def get_designs(self) -> List[Union[str, bytes]]:  # design_type: str = PUtils.interface_design
        """Return the paths of all design files in a DesignDirectory"""
        return sorted(glob(os.path.join(self.designs, '*.pdb')))
        # return sorted(glob(os.path.join(self.designs, '*%s*.pdb' % design_type)))

    # def return_symmetry_stats(self):  # Depreciated
    #     return len(symm for symm in self.program_root)
    #
    # def return_building_block_stats(self):  # Depreciated
    #     return len(bb for symm_bb in self.composition for bb in symm_bb)
    #
    # def return_unique_pose_stats(self):  # Depreciated
    #     return len(bb for symm in self.composition for bb in symm)

    # def return_fragment_metrics(self):
    #     self.all_residue_score, self.center_residue_score, self.fragment_residues_total, \
    #         self.central_residues_with_fragment_overlap, self.multiple_frag_ratio, self.fragment_content_d

    def get_fragment_metrics(self):
        """Calculate fragment metrics for all fragment observations in the design

        Sets:
            self.center_residue_numbers (int)
            self.total_residue_numbers (int)
            self.fragment_residues_total (int)
            self.central_residues_with_fragment_overlap (int)
            self.total_interface_residues (int)
            self.all_residue_score (float)
            self.center_residue_score (float)
            self.multiple_frag_ratio (float)
            self.helical_fragment_content (float)
            self.strand_fragment_content (float)
            self.coil_fragment_content (float)
            self.total_non_fragment_interface_residues (int)
            self.percent_residues_fragment_center (int)
            self.percent_residues_fragment_total (int)
            self.interface_ss_fragment_topology (Dict[int, str])
        """
        # TODO Doesn't need an attached pose if fragments have been generated
        if self.design_residues is False:  # no search yet, so = False
            self.identify_interface()  # sets self.design_residues and self.interface_residues

        self.log.debug('Starting fragment metric collection')
        if self.fragment_observations:  # check if fragment generation has been populated somewhere
            # frag_metrics = self.pose.return_fragment_metrics(fragments=self.info.get('fragments'))
            frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
        else:
            if self.pose.fragment_queries:
                self.log.debug('Fragment observations found in Pose. Adding to the Design state')
                self.fragment_observations = self.pose.return_fragment_observations()
                frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
            elif os.path.exists(self.frag_file):  # try to pull them from disk
                self.log.debug('Fragment observations found on disk. Adding to the Design state')
                self.retrieve_fragment_info_from_file()
                # frag_metrics = format_fragment_metrics(calculate_match_metrics(self.fragment_observations))
                frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
            # fragments were attempted, but returned nothing, set frag_metrics to the template (empty)
            elif self.fragment_observations == list():
                frag_metrics = fragment_metric_template
            # fragments haven't been attempted on this pose
            elif self.fragment_observations is None:
                self.log.warning('There were no fragments generated for this Design! If this isn\'t what you expected, '
                                 'ensure that you haven\'t disabled it with "--%s" or explicitly request it with --%s'
                                 % (PUtils.no_term_constraint, PUtils.generate_fragments))
                #                  'Have you run %s on it yet?' % PUtils.generate_fragments)
                frag_metrics = fragment_metric_template
                # self.log.warning('%s: There are no fragment observations for this Design! Have you run %s on it yet?
                #                  ' Trying %s now...' % (self.path, PUtils.generate_fragments, generate_fragments))
                # self.generate_interface_fragments()
            else:
                raise DesignError('Design hit a snag that shouldn\'t have happened. Please report to the developers')
            self.info['fragments'] = self.fragment_observations
            self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

        self.center_residue_numbers = frag_metrics['center_residues']
        self.total_residue_numbers = frag_metrics['total_residues']
        self.all_residue_score = frag_metrics['nanohedra_score']
        self.center_residue_score = frag_metrics['nanohedra_score_center']
        self.fragment_residues_total = frag_metrics['number_fragment_residues_total']
        # ^ can be more than self.total_interface_residues because each fragment may have members not in the interface
        self.central_residues_with_fragment_overlap = frag_metrics['number_fragment_residues_center']
        self.multiple_frag_ratio = frag_metrics['multiple_fragment_ratio']
        self.helical_fragment_content = frag_metrics['percent_fragment_helix']
        self.strand_fragment_content = frag_metrics['percent_fragment_strand']
        self.coil_fragment_content = frag_metrics['percent_fragment_coil']

        self.total_interface_residues = len(self.interface_residues)
        self.total_non_fragment_interface_residues = \
            max(self.total_interface_residues - self.central_residues_with_fragment_overlap, 0)
        try:
            # if interface_distance is different between interface query and fragment generation these can be < 0 or > 1
            self.percent_residues_fragment_center = \
                min(self.central_residues_with_fragment_overlap / self.total_interface_residues, 1)
            self.percent_residues_fragment_total = min(self.fragment_residues_total / self.total_interface_residues, 1)
        except ZeroDivisionError:
            self.log.warning('%s: No interface residues were found. Is there an interface in your design?'
                             % self.source)
            self.percent_residues_fragment_center, self.percent_residues_fragment_total = 0.0, 0.0

        # todo make dependent on split_interface_residues which doesn't have residues obj, just number (pickle concerns)
        if not self.pose.ss_index_array or not self.pose.ss_type_array:
            self.pose.interface_secondary_structure()  # source_db=self.resources, source_dir=self.stride_dir)
        for number, elements in self.pose.split_interface_ss_elements.items():
            fragment_elements = set()
            # residues, entities = self.pose.split_interface_residues[number]
            for residue, _, element in zip(*zip(*self.pose.split_interface_residues[number]), elements):
                if residue.number in self.center_residue_numbers:
                    fragment_elements.add(element)
            self.interface_ss_fragment_topology[number] = \
                ''.join(self.pose.ss_type_array[element] for element in fragment_elements)

    @handle_errors(errors=(FileNotFoundError,))
    def retrieve_fragment_info_from_file(self):
        """Gather observed fragment metrics from fragment matching output

        Sets:
            self.fragment_observations (List[Dict[str, Union[int, str, float]]])
        """
        fragment_observations = set()
        with open(self.frag_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[:6] == 'z-val:':
                    # overlap_rmsd_divded_by_cluster_rmsd
                    match_score = match_score_from_z_value(float(line[6:].strip()))
                elif line[:21] == 'oligomer1 ch, resnum:':
                    oligomer1_info = line[21:].strip().split(',')
                    # chain1 = oligomer1_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number1 = int(oligomer1_info[1])
                elif line[:21] == 'oligomer2 ch, resnum:':
                    oligomer2_info = line[21:].strip().split(',')
                    # chain2 = oligomer2_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number2 = int(oligomer2_info[1])
                elif line[:3] == 'id:':
                    cluster_id = [index.strip('ijk') for index in line[3:].strip().split('_')]
                    # use with self.entity_names to get mapped and paired oligomer id
                    fragment_observations.add((residue_number1, residue_number2, '_'.join(cluster_id), match_score))
        self.fragment_observations = [dict(zip(('mapped', 'paired', 'cluster', 'match'), frag_obs))
                                      for frag_obs in fragment_observations]

    # @handle_errors(errors=(FileNotFoundError,))
    def retrieve_pose_transformation_from_file(self) -> List[Dict]:
        """Gather pose transformation information for the Pose from Nanohedra output

        Returns:
            The pose transformation arrays as found in the pose_file
        """
        try:
            with open(self.pose_file, 'r') as f:
                pose_transformation = {}
                for line in f.readlines():
                    if line[:20] == 'ROT/DEGEN MATRIX PDB':
                        data = eval(line[22:].strip())  # Todo remove eval(), this is a program vulnerability
                        pose_transformation[int(line[20:21])] = {'rotation': np.array(data)}
                    elif line[:15] == 'INTERNAL Tx PDB':  # all below parsing lacks PDB number suffix such as PDB1 or PDB2
                        data = eval(line[17:].strip())
                        if data:  # == 'None'
                            pose_transformation[int(line[15:16])]['translation'] = np.array(data)
                        else:
                            pose_transformation[int(line[15:16])]['translation'] = np.array([0, 0, 0])
                    elif line[:18] == 'SETTING MATRIX PDB':
                        data = eval(line[20:].strip())
                        pose_transformation[int(line[18:19])]['rotation2'] = np.array(data)
                    elif line[:22] == 'REFERENCE FRAME Tx PDB':
                        data = eval(line[24:].strip())
                        if data:
                            pose_transformation[int(line[22:23])]['translation2'] = np.array(data)
                        else:
                            pose_transformation[int(line[22:23])]['translation2'] = np.array([0, 0, 0])
                    elif 'CRYST1 RECORD:' in line:
                        cryst_record = line[15:].strip()
                        self.cryst_record = None if cryst_record == 'None' else cryst_record

            return [pose_transformation[idx] for idx, _ in enumerate(pose_transformation, 1)]
        except TypeError:
            raise FileNotFoundError('The specified pose metrics file was not declared and cannot be found!')

    def retrieve_pose_metrics_from_file(self) -> List[Dict]:
        """Gather information for the docked Pose from Nanohedra output. Includes coarse fragment metrics

        Returns:
            (list[dict])
        """
        try:
            with open(self.pose_file, 'r') as f:
                pose_transformation = {}
                for line in f.readlines():
                    if line[:15] == 'DOCKED POSE ID:':
                        self.pose_id = line[15:].strip().replace('_DEGEN_', '-DEGEN_').replace('_ROT_', '-ROT_').\
                            replace('_TX_', '-tx_')
                    elif line[:38] == 'Unique Mono Fragments Matched (z<=1): ':
                        self.high_quality_int_residues_matched = int(line[38:].strip())
                    # number of interface residues with fragment overlap potential from other oligomer
                    elif line[:31] == 'Unique Mono Fragments Matched: ':
                        self.central_residues_with_fragment_overlap = int(line[31:].strip())
                    # number of interface residues with 2 residues on either side of central residue
                    elif line[:36] == 'Unique Mono Fragments at Interface: ':
                        self.fragment_residues_total = int(line[36:].strip())
                    elif line[:25] == 'Interface Matched (%): ':  # matched / at interface * 100
                        self.percent_overlapping_fragment = float(line[25:].strip()) / 100
                    elif line[:20] == 'ROT/DEGEN MATRIX PDB':
                        data = eval(line[22:].strip())  # Todo remove eval(), this is a program vulnerability
                        pose_transformation[int(line[20:21])] = {'rotation': np.array(data)}
                    elif line[:15] == 'INTERNAL Tx PDB':  # all below parsing lacks PDB number suffix such as PDB1 or PDB2
                        data = eval(line[17:].strip())
                        if data:  # == 'None'
                            pose_transformation[int(line[15:16])]['translation'] = np.array(data)
                        else:
                            pose_transformation[int(line[15:16])]['translation'] = np.array([0, 0, 0])
                    elif line[:18] == 'SETTING MATRIX PDB':
                        data = eval(line[20:].strip())
                        pose_transformation[int(line[18:19])]['rotation2'] = np.array(data)
                    elif line[:22] == 'REFERENCE FRAME Tx PDB':
                        data = eval(line[24:].strip())
                        if data:
                            pose_transformation[int(line[22:23])]['translation2'] = np.array(data)
                        else:
                            pose_transformation[int(line[22:23])]['translation2'] = np.array([0, 0, 0])
                    elif 'Nanohedra Score:' in line:  # res_lev_sum_score
                        self.all_residue_score = float(line[16:].rstrip())
                    # elif 'CRYST1 RECORD:' in line:
                    #     cryst_record = line[15:].strip()
                    #     self.cryst_record = None if cryst_record == 'None' else cryst_record
                    # elif line[:31] == 'Canonical Orientation PDB1 Path':
                    #     self.canonical_pdb1 = line[:31].strip()
                    # elif line[:31] == 'Canonical Orientation PDB2 Path':
                    #     self.canonical_pdb2 = line[:31].strip()
            return [pose_transformation[idx] for idx, _ in enumerate(pose_transformation, 1)]
        except TypeError:
            raise FileNotFoundError('The specified pose metrics file was not declared and cannot be found!')

    def pickle_info(self):
        """Write any design attributes that should persist over program run time to serialized file"""
        if self.nanohedra_output and not self.construct_pose:  # don't write anything as we are just querying Nanohedra
            return
        self.make_path(self.data)
        if self.info != self._info:  # if the state has changed from the original version
            pickle_object(self.info, self.serialized_info, out_path='')

    def prepare_rosetta_flags(self, symmetry_protocol: Optional[str] = None, sym_def_file: Optional[str] = None,
                              pdb_out_path: Optional[str] = None, out_path: Union[str, bytes] = os.getcwd()) -> str:
        """Prepare a protocol specific Rosetta flags file with program specific variables

        Keyword Args:
            symmetry_protocol: The type of symmetric protocol (specifying design dimension) to use for Rosetta jobs
            sym_def_file: A Rosetta specific file specifying the symmetry system
            pdb_out_path: Disk location to write the resulting design files
            out_path: Disk location to write the flags file
        Returns:
            Disk location of the flags file
        """
        # flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
        self.log.info('Total number of residues in Pose: %d' % self.pose.number_of_residues)

        # Get ASU distance parameters
        if self.design_dimension:  # check for None and dimension 0 simultaneously
            # The furthest point from the ASU COM + the max individual Entity radius
            distance = self.pose.pdb.radius + max([entity.radius for entity in self.pose.entities])  # all the radii
            self.log.info('Expanding ASU into symmetry group by %f Angstroms' % distance)
        else:
            distance = 0

        if self.no_evolution_constraint:
            constraint_percent, free_percent = 0, 1
        else:
            constraint_percent = 0.5
            free_percent = 1 - constraint_percent

        variables = rosetta_variables + [('dist', distance), ('repack', 'yes'),
                                         ('constrained_percent', constraint_percent), ('free_percent', free_percent)]
        variables.extend([(PUtils.design_profile, self.design_profile_file)] if self.design_profile else [])
        variables.extend([(PUtils.fragment_profile, self.fragment_profile_file)] if self.fragment_profile else [])

        self.prepare_symmetry_for_rosetta()
        if not symmetry_protocol:
            symmetry_protocol = self.symmetry_protocol
        if not sym_def_file:
            sym_def_file = self.sym_def_file
        variables.extend([('symmetry', symmetry_protocol), ('sdf', sym_def_file)] if symmetry_protocol else [])
        out_of_bounds_residue = self.pose.number_of_residues * self.number_of_symmetry_mates + 1
        variables.extend([(interface, residues) if residues else (interface, out_of_bounds_residue)
                          for interface, residues in self.design_residue_ids.items()])

        # assign any additional designable residues
        if self.pose.required_residues:
            variables.extend([('required_residues', ','.join('%d%s' % (res.number, res.chain)
                                                             for res in self.pose.required_residues))])
        else:  # get an out-of-bounds index
            variables.extend([('required_residues', out_of_bounds_residue)])

        # allocate any "core" residues based on central fragment information
        if self.center_residue_numbers:
            variables.extend([('core_residues', ','.join(map(str, self.center_residue_numbers)))])
        else:  # get an out of bounds index
            variables.extend([('core_residues', out_of_bounds_residue)])

        flags = copy.copy(rosetta_flags)
        if pdb_out_path:
            flags.extend(['-out:path:pdb %s' % pdb_out_path, '-scorefile %s' % self.scores_file])
        else:
            flags.extend(['-out:path:pdb %s' % self.designs, '-scorefile %s' % self.scores_file])
        flags.append('-in:file:native %s' % self.refined_pdb)
        flags.append('-parser:script_vars %s' % ' '.join('%s=%s' % tuple(map(str, var_val)) for var_val in variables))

        out_file = os.path.join(out_path, 'flags')
        with open(out_file, 'w') as f:
            f.write('%s\n' % '\n'.join(flags))

        return out_file

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def interface_metrics(self):
        """Generate a script capable of running Rosetta interface metrics analysis on the bound and unbound states"""
        # metrics_flags = 'repack=yes'
        main_cmd = copy.copy(script_cmd)
        if self.interface_residues is False or self.design_residues is False:
            # need these ^ for making flags so get them v
            self.identify_interface()
            if not self.design_residues:  # we should always get an empty set if we have got to this point
                raise DesignError('No residues were found with your design criteria... Your flags may be too stringent '
                                  'or incorrect. If you are performing interface design, check that your input has an '
                                  'interface')
        else:
            self.load_pose()
        # interface_secondary_structure
        if not os.path.exists(self.flags) or self.force_flags:
            # self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()  # <-$ needed for prepare_rosetta_flags -> self.center_residue_numbers
            self.make_path(self.scripts)
            self.flags = self.prepare_rosetta_flags(out_path=self.scripts)
            self.log.debug('Pose flags written to: %s' % self.flags)

        pdb_list = os.path.join(self.scripts, 'design_files%s.txt' %
                                ('_%s' % self.specific_protocol if self.specific_protocol else ''))
        generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', pdb_list] + \
            (['-s', self.specific_protocol] if self.specific_protocol else [])
        main_cmd += ['@%s' % self.flags, '-in:file:l', pdb_list,
                     # TODO out:file:score_only file is not respected if out:path:score_file given
                     #  -run:score_only true?
                     '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
        #              '-in:file:native', self.refined_pdb,
        if self.mpi:
            main_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + main_cmd
            self.run_in_shell = False

        metric_cmd_bound = main_cmd + (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + \
            [os.path.join(PUtils.rosetta_scripts, 'interface_%s%s.xml'
                          % (PUtils.stage[3], '_DEV' if self.development else ''))]
        entity_cmd = main_cmd + [os.path.join(PUtils.rosetta_scripts, '%s_entity%s.xml'
                                              % (PUtils.stage[3], '_DEV' if self.development else ''))]
        metric_cmds = [metric_cmd_bound]
        for idx, entity in enumerate(self.pose.entities, 1):
            if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
                entity_sym = 'symmetry=make_point_group'
            else:
                entity_sdf, entity_sym = '', 'symmetry=asymmetric'
            _metric_cmd = entity_cmd + ['-parser:script_vars', 'repack=yes', 'entity=%d' % idx, entity_sym] + \
                ([entity_sdf] if entity_sdf != '' else [])
            self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(_metric_cmd)))
            metric_cmds.append(_metric_cmd)
        # Create executable to gather interface Metrics on all Designs
        if not self.run_in_shell:
            write_shell_script(list2cmdline(generate_files_cmd), name='interface_%s' % PUtils.stage[3],
                               out_path=self.scripts, additional=[list2cmdline(command) for command in metric_cmds])
        else:
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                # Wait for Rosetta Design command to complete
                metrics_process.communicate()

    def custom_rosetta_script(self, script, file_list=None, native=None, suffix=None,
                              score_only=None, variables=None, **kwargs):
        """Generate a custom script to dispatch to the design using a variety of parameters"""
        raise DesignError('This module is outdated, please update it to use')  # Todo reflect modern metrics collection
        cmd = copy.copy(script_cmd)
        script_name = os.path.splitext(os.path.basename(script))[0]
        flags = os.path.join(self.scripts, 'flags')
        if not os.path.exists(self.flags) or self.force_flags:  # Generate a new flags_design file
            # Need to assign the designable residues for each entity to a interface1 or interface2 variable
            self.identify_interface()
            # self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()  # needed for prepare_rosetta_flags -> self.center_residue_numbers
            self.make_path(self.scripts)
            flags = self.prepare_rosetta_flags(out_path=self.scripts)
            self.log.debug('Pose flags written to: %s' % flags)

        cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []

        if file_list:
            pdb_input = os.path.join(self.scripts, 'design_files.txt')
            generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', pdb_input]
        else:
            pdb_input = self.refined_pdb
            generate_files_cmd = []  # empty command

        if native:
            native = getattr(self, native, 'refined_pdb')
        else:
            native = self.refined_pdb

        # if isinstance(suffix, str):
        #     suffix = ['-out:suffix', '_%s' % suffix]
        # if isinstance(suffix, bool):
        if suffix:
            suffix = ['-out:suffix', '_%s' % script_name]
        else:
            suffix = []

        if score_only:
            score = ['-out:file:score_only', self.scores_file]
        else:
            score = []

        if self.number_of_trajectories:
            trajectories = ['-nstruct', str(self.number_of_trajectories)]
        else:
            trajectories = ['-no_nstruct_label true']

        if variables:
            for idx, var_val in enumerate(variables):
                variable, value = var_val.split('=')
                variables[idx] = '%s=%s' % (variable, getattr(self.pose, value, ''))
            variables = ['-parser:script_vars'] + variables
        else:
            variables = []

        cmd += ['@%s' % flags, '-in:file:%s' % ('l' if file_list else 's'), pdb_input, '-in:file:native', native] + \
            score + suffix + trajectories + ['-parser:protocol', script] + variables
        if self.mpi:
            cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + cmd
            self.run_in_shell = False

        write_shell_script(list2cmdline(generate_files_cmd),
                           name=script_name, out_path=self.scripts, additional=[list2cmdline(cmd)])

    def prepare_symmetry_for_rosetta(self):
        """For the specified design, locate/make the symmetry files necessary for Rosetta input

        Sets:
            self.symmetry_protocol (str)
            self.sym_def_file (Union[str, bytes])
        """
        if self.design_dimension is not None:  # symmetric, could be 0
            self.symmetry_protocol = PUtils.protocol[self.design_dimension]
            self.log.info('Symmetry Option: %s' % self.symmetry_protocol)
            self.log.debug('Design has Symmetry Entry Number: %s (Laniado & Yeates, 2020)' % str(self.sym_entry_number))
            self.sym_def_file = self.sym_entry.sdf_lookup()
        else:  # asymmetric
            self.symmetry_protocol = 'asymmetric'
            self.sym_def_file = sdf_lookup()  # grabs C1.sym
            self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it\'s'
                              ' an ASU it may be missing crucial interface contacts. Is this what you want?')

    def rosetta_interface_design(self):
        """For the basic process of sequence design between two halves of an interface, write the necessary files for
        refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)

        Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
        on the DesignDirectory
        """
        # Set up the command base (rosetta bin and database paths)
        if self.scout:
            protocol, protocol_xml1 = PUtils.stage[12], PUtils.stage[12]
            nstruct_instruct = ['-no_nstruct_label', 'true']
            generate_files_cmd, metrics_pdb = [], ['-in:file:s', self.scouted_pdb]
            metrics_flags = 'repack=no'
            additional_cmds, out_file = [], []
        elif self.structure_background:
            protocol, protocol_xml1 = PUtils.stage[14], PUtils.stage[14]
            nstruct_instruct = ['-no_nstruct_label', 'true']
            design_list_file = os.path.join(self.scripts, 'design_files_%s.txt' % protocol)
            generate_files_cmd = \
                ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', design_list_file, '-s', '_' + protocol]
            metrics_pdb = ['-in:file:l', design_list_file]  # self.pdb_list]
            metrics_flags = 'repack=yes'
            additional_cmds, out_file = [], []
        elif self.no_hbnet:  # run the legacy protocol
            protocol, protocol_xml1 = PUtils.stage[2], PUtils.stage[2]
            nstruct_instruct = ['-nstruct', str(self.number_of_trajectories)]
            design_list_file = os.path.join(self.scripts, 'design_files_%s.txt' % protocol)
            generate_files_cmd = \
                ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', design_list_file, '-s', '_' + protocol]
            metrics_pdb = ['-in:file:l', design_list_file]  # self.pdb_list]
            metrics_flags = 'repack=yes'
            additional_cmds, out_file = [], []
        else:  # run hbnet_design_profile protocol
            protocol, protocol_xml1 = PUtils.stage[13], 'hbnet_scout'  # PUtils.stage[14]
            nstruct_instruct = ['-no_nstruct_label', 'true']
            design_list_file = os.path.join(self.scripts, 'design_files_%s.txt' % protocol)
            generate_files_cmd = \
                ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', design_list_file, '-s', '_' + protocol]
            metrics_pdb = ['-in:file:l', design_list_file]  # self.pdb_list]
            metrics_flags = 'repack=yes'
            out_file = ['-out:file:silent', os.path.join(self.data, 'hbnet_silent.o'),
                        '-out:file:silent_struct_type', 'binary']
            additional_cmds = \
                [[PUtils.hbnet_sort, os.path.join(self.data, 'hbnet_silent.o'), str(self.number_of_trajectories)]]
            # silent_file = os.path.join(self.data, 'hbnet_silent.o')
            # additional_commands = \
            #     [
            #      # ['grep', '^SCORE', silent_file, '>', os.path.join(self.data, 'hbnet_scores.sc')],
            #      main_cmd + [os.path.join(self.data, 'hbnet_selected.o')]
            #      [os.path.join(self.data, 'hbnet_selected.tags')]
            #     ]

        main_cmd = copy.copy(script_cmd)
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
        if not os.path.exists(self.flags) or self.force_flags:
            # self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()  # needed for prepare_rosetta_flags -> self.center_residue_numbers
            self.make_path(self.scripts)
            self.flags = self.prepare_rosetta_flags(out_path=self.scripts)
            self.log.debug('Pose flags written to: %s' % self.flags)

        if self.consensus:  # Todo add consensus sbatch generator to the symdesign main
            if not self.no_term_constraint:  # design_with_fragments
                consensus_cmd = main_cmd + relax_flags_cmdline + \
                    ['@%s' % self.flags, '-in:file:s', self.consensus_pdb,
                     # '-in:file:native', self.refined_pdb,
                     '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[5]),
                     '-parser:script_vars', 'switch=%s' % PUtils.stage[5]]
                self.log.info('Consensus Command: %s' % list2cmdline(consensus_cmd))
                if not self.run_in_shell:
                    write_shell_script(list2cmdline(consensus_cmd), name=PUtils.stage[5], out_path=self.scripts)
                else:
                    consensus_process = Popen(consensus_cmd)
                    consensus_process.communicate()
            else:
                self.log.critical('Cannot run consensus design without fragment info and none was found.'
                                  ' Did you mean to design with -generate_fragments False? You will need to run with'
                                  '\'True\' if you want to use fragments')
        # DESIGN: Prepare command and flags file
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        design_cmd = main_cmd + (['-in:file:pssm', self.evolutionary_profile_file] if self.evolutionary_profile else []) + \
            ['@%s' % self.flags, '-in:file:s', self.scouted_pdb if os.path.exists(self.scouted_pdb) else self.refined_pdb,
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % protocol_xml1),
             '-out:suffix', '_%s' % protocol] + nstruct_instruct + out_file
        if additional_cmds:  # this is where hbnet_design_profile.xml is set up, which could be just design_profile.xml
            additional_cmds.append(
                main_cmd +
                (['-in:file:pssm', self.evolutionary_profile_file] if self.evolutionary_profile else []) +
                ['-in:file:silent', os.path.join(self.data, 'hbnet_selected.o'), '@%s' % self.flags,
                 '-in:file:silent_struct_type', 'binary',
                 # '-out:suffix', '_%s' % protocol,
                 '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % protocol)])  # + nstruct_instruct)

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        entity_cmd = script_cmd + metrics_pdb + \
            ['@%s' % self.flags, '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, 'metrics_entity.xml')]

        if self.mpi and not self.scout:
            design_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + design_cmd
            entity_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + entity_cmd
            self.run_in_shell = False

        self.log.info('Design Command: %s' % list2cmdline(design_cmd))
        metric_cmds = []
        for idx, entity in enumerate(self.pose.entities, 1):
            if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
                entity_sym = 'symmetry=make_point_group'
            else:
                entity_sdf, entity_sym = '', 'symmetry=asymmetric'
            _metric_cmd = entity_cmd + ['-parser:script_vars', 'repack=yes', 'entity=%d' % idx, entity_sym] + \
                ([entity_sdf] if entity_sdf != '' else [])
            self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(_metric_cmd)))
            metric_cmds.append(_metric_cmd)
        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if not self.run_in_shell:
            write_shell_script(list2cmdline(design_cmd), name=protocol, out_path=self.scripts,
                               additional=[list2cmdline(command) for command in additional_cmds] +
                               [list2cmdline(generate_files_cmd)] +
                               [list2cmdline(command) for command in metric_cmds])
            #                  status_wrap=self.serialized_info,
        else:
            design_process = Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            for metric_cmd in metric_cmds:
                metrics_process = Popen(metric_cmd)
                metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if self.run_in_shell:
            pose_s = self.design_analysis()
            out_path = os.path.join(self.all_scores, PUtils.analysis_file % (starttime, 'All'))
            if not os.path.exists(out_path):
                header = True
            else:
                header = False
            pose_s.to_csv(out_path, mode='a', header=header)

    def transform_oligomers_to_pose(self, refined=True, oriented=False, **kwargs):
        """Take the set of oligomers involved in a pose composition and transform them from a standard reference frame
        to the pose reference frame using computed pose_transformation parameters. Default is to take the pose from the
        master Database refined source if the oligomers exist there, if they don't, the oriented source is used if it
        exists. Finally, the DesignDirectory will be used as a back up

        Keyword Args:
            refined=True (bool): Whether to use the refined pdb from the refined pdb source directory
            oriented=false (bool): Whether to use the oriented pdb from the oriented pdb source directory
        """
        self.get_oligomers(refined=refined, oriented=oriented)
        if self.pose_transformation:
            self.oligomers = [oligomer.return_transformed_copy(**self.pose_transformation[oligomer_idx])
                              for oligomer_idx, oligomer in enumerate(self.oligomers)]
            # Todo assumes a 1:1 correspondence between entities and oligomers (component group numbers) CHANGE
            # Todo make oligomer, oligomer! If symmetry permits now that storing as refined asu, oriented asu, model asu
            self.log.debug('Oligomers were transformed to the found docking parameters')
        else:
            raise DesignError('The design could not be transformed as it is missing the required transformation '
                              'parameters. Were they generated properly?')

    def transform_structures_to_pose(self, structures, **kwargs):
        """Take the set of oligomers involved in a pose composition and transform them from a standard reference frame
        to the pose reference frame using computed pose_transformation parameters. Default is to take the pose from the
        master Database refined source if the oligomers exist there, if they don't, the oriented source is used if it
        exists. Finally, the DesignDirectory will be used as a back up

        Args:
            structures (Iterable[Structure]): The Structure objects you would like to transform
        """
        if self.pose_transformation:
            self.log.debug('Structures were transformed to the found docking parameters')
            # Todo assumes a 1:1 correspondence between structures and transforms (component group numbers) CHANGE
            return [structure.return_transformed_copy(**self.pose_transformation[idx])
                    for idx, structure in enumerate(structures)]
        else:
            raise DesignError('The design could not be transformed as it is missing the required transformation '
                              'parameters. Were they generated properly?')

    def get_oligomers(self, refined: bool = True, oriented: bool = False):
        """Retrieve oligomeric files from either the design Database, the oriented directory, or the refined directory,
        or the design directory, and load them into job for further processing

        Args:
            refined: Whether to use the refined oligomeric directory
            oriented: Whether to use the oriented oligomeric directory
        Sets:
            self.oligomers (list[PDB])
        """
        source_preference = ['refined', 'oriented', 'design']
        if self.resources:
            if refined:
                source_idx = 0
            elif oriented:
                source_idx = 1
            else:
                source_idx = 2
                self.log.warning('Falling back on oligomers present in the Design source which may not be refined. This'
                                 ' will lead to issues in sequence design if the structure is not refined first...')

            self.oligomers.clear()
            for name in self.entity_names:
                oligomer = None
                while not oligomer:
                    oligomer = self.resources.retrieve_data(source=source_preference[source_idx], name=name)
                    if isinstance(oligomer, Structure):
                        self.log.info('Found oligomer file at %s and loaded into job' % source_preference[source_idx])
                        self.oligomers.append(oligomer)
                    else:
                        self.log.error('Couldn\'t locate the oligomer %s at the specified source %s'
                                       % (name, source_preference[source_idx]))
                        source_idx += 1
                        self.log.error('Falling back to source %s' % source_preference[source_idx])
                        if source_preference[source_idx] == 'design':
                            file = glob(os.path.join(self.path, '%s*.pdb*' % name))
                            if file and len(file) == 1:
                                self.oligomers.append(PDB.from_file(file[0], log=self.log,
                                                                    name=os.path.splitext(os.path.basename(file))[0]))
                            else:
                                raise DesignError('Couldn\'t located the specified oligomer %s' % name)
            if source_idx == 0:
                self.pre_refine = True
        else:  # Todo consolidate this with above as far as iterative mechanism
            if refined:  # prioritize the refined version
                path = self.refine_dir
                for name in self.entity_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % name))[0]):
                        oriented = True  # fall back to the oriented version
                        self.log.debug('Couldn\'t find oligomers in the refined directory')
                        break
                self.pre_refine = True if not oriented else False
            if oriented:
                path = self.orient_dir
                for name in self.entity_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % name))[0]):
                        path = self.path
                        self.log.debug('Couldn\'t find oligomers in the oriented directory')

            if not refined and not oriented:
                path = self.path

            idx = 2  # initialize as 2. it doesn't matter if no names are found, but nominally it should be 2 for now
            oligomer_files = []
            for idx, name in enumerate(self.entity_names, 1):
                oligomer_files.extend(sorted(glob(os.path.join(path, '%s*.pdb*' % name))))  # first * is DesignDirectory
            assert len(oligomer_files) == idx, \
                'Incorrect number of oligomers! Expected %d, %d found. Matched files from \'%s\':\n\t%s' \
                % (idx, len(oligomer_files), os.path.join(path, '*.pdb*'), oligomer_files)

            self.oligomers.clear()  # for every call we should reset the list
            for file in oligomer_files:
                self.oligomers.append(PDB.from_file(file, name=os.path.splitext(os.path.basename(file))[0],
                                                    log=self.log))
        self.log.debug('%d matching oligomers found' % len(self.oligomers))
        assert len(self.oligomers) == len(self.entity_names), \
            'Expected %d oligomers, but found %d' % (len(self.oligomers), len(self.entity_names))

    def load_pose(self, source: str = None, entities: List[Structure] = None):
        """For the design info given by a DesignDirectory source, initialize the Pose with self.source file,
        self.symmetry, self.design_selectors, self.fragment_database, and self.log objects

        Handles clash testing and writing the assembly if those options are True

        Args:
            source: The file path to a source file
            entities: The Entities desired in the Pose
        """
        if self.pose and not source and not entities:  # pose is already loaded and nothing new provided
            return

        rename_chains = True  # because the result of entities, we should rename
        if not entities and not self.source or not os.path.exists(self.source):
            # in case we initialized design without a .pdb or clean_asu.pdb (Nanohedra)
            self.log.info('No source file found. Fetching source from Database and transforming to Pose')
            self.transform_oligomers_to_pose()
            entities = []
            for oligomer in self.oligomers:
                entities.extend(oligomer.entities)
            # because the file wasn't specified on the way in, no chain names should be binding
            # rename_chains = True

        if entities:
            pdb = PDB.from_entities(entities, log=self.log, rename_chains=rename_chains)
            #                       name='%s-asu' % str(self)
        elif self.init_pdb:  # this is a fresh pose, and we already loaded so reuse
            # careful, if some processing to the pdb has since occurred then this will be wrong!
            pdb = self.init_pdb
        else:
            pdb = PDB.from_file(source if source else self.source, entity_names=self.entity_names, log=self.log)
            #                              pass names if available ^
        if self.symmetric:
            self.pose = Pose.from_asu(pdb, sym_entry=self.sym_entry, name='%s-asu' % str(self),
                                      design_selector=self.design_selector, log=self.log,
                                      source_db=self.resources, fragment_db=self.fragment_db,
                                      euler_lookup=self.euler_lookup, ignore_clashes=self.ignore_clashes)
            # generate oligomers for each entity in the pose
            for idx, entity in enumerate(self.pose.entities):
                if entity.number_of_monomers != self.sym_entry.group_subunit_numbers[idx]:
                    entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **self.pose_transformation[idx])
                # write out new oligomers to the DesignDirectory
                if self.write_oligomers:
                    entity.write_oligomer(out_path=os.path.join(self.path, '%s_oligomer.pdb' % entity.name))
        else:
            self.pose = Pose.from_pdb(pdb, name=str(self),
                                      design_selector=self.design_selector, log=self.log,
                                      source_db=self.resources, fragment_db=self.fragment_db,
                                      euler_lookup=self.euler_lookup, ignore_clashes=self.ignore_clashes)
        if not self.entity_names:  # store the entity names if they were never generated
            self.entity_names = [entity.name for entity in self.pose.entities]
            self.log.info('Input Entities: %s' % ', '.join(self.entity_names))
            self.info['entity_names'] = self.entity_names

        # Save renumbered PDB to clean_asu.pdb
        if not self.asu_path or not os.path.exists(self.asu_path):
            if self.nanohedra_output and not self.construct_pose or self.output_directory:
                return

            self.save_asu()

    def save_asu(self):  # , rename_chains=False
        """Save a new Structure from multiple Chain or Entity objects including the Pose symmetry"""
        if self.fuse_chains:
            # try:
            for fusion_nterm, fusion_cterm in self.fuse_chains:
                # rename_success = False
                new_success, same_success = False, False
                for idx, entity in enumerate(self.pose.entities):
                    if entity.chain_id == fusion_cterm:
                        entity_new_chain_idx = idx
                        new_success = True
                    if entity.chain_id == fusion_nterm:
                        # entity_same_chain_idx = idx
                        same_success = True
                        # rename_success = True
                        # break
                # if not rename_success:
                if not new_success and not same_success:
                    raise DesignError('Your requested fusion of chain %s with chain %s didn\'t work!' %
                                      (fusion_nterm, fusion_cterm))
                    # self.log.critical('Your requested fusion of chain %s with chain %s didn\'t work!' %
                    #                   (fusion_nterm, fusion_cterm))
                else:  # won't be accessed unless entity_new_chain_idx is set
                    self.pose.entities[entity_new_chain_idx].chain_id = fusion_nterm
            # except AttributeError:
            #     raise ValueError('One or both of the chain IDs %s were not found in the input model. Possible chain'
            #                      ' ID\'s are %s' % ((fusion_nterm, fusion_cterm), ','.join(new_asu.chain_ids)))
        self.pose.write(out_path=self.asu_path)
        self.info['pre_refine'] = self.pre_refine
        self.log.info('Cleaned PDB: "%s"' % self.asu_path)

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def check_clashes(self, clashing_threshold=0.75):
        """Given a multimodel file, measure the number of clashes is less than a percentage threshold"""
        models = [Models.from_PDB(self.resources.full_models.retrieve_data(name=entity), log=self.log)
                  for entity in self.entity_names]
        # models = [Models.from_file(self.resources.full_models.retrieve_data(name=entity))
        #           for entity in self.entity_names]

        # for each model, transform to the correct space
        models = self.transform_structures_to_pose(models)
        multimodel = MultiModel.from_models(models, independent=True, log=self.log)

        clashes = 0
        prior_clashes = 0
        for idx, state in enumerate(multimodel, 1):
            clashes += (1 if state.is_clash() else 0)
            state.write(out_path=os.path.join(self.path, 'state_%d.pdb' % idx))
            print('State %d - Clashes: %s' % (idx, 'YES' if clashes > prior_clashes else 'NO'))
            prior_clashes = clashes

        if clashes/float(len(multimodel)) > clashing_threshold:
            raise DesignError('The frequency of clashes (%f) exceeds the clashing threshold (%f)'
                              % (clashes/float(len(multimodel)), clashing_threshold))

    @handle_design_errors(errors=(DesignError,))
    @close_logs
    @remove_structure_memory
    def rename_chains(self):
        """Standardize the chain names in incremental order found in the design source file"""
        pdb = PDB.from_file(self.source, log=self.log, pose_format=False)
        pdb.reorder_chains()
        pdb.write(out_path=self.asu_path)

    @handle_design_errors(errors=(DesignError, ValueError, RuntimeError))
    @close_logs
    @remove_structure_memory
    def orient(self, to_design_directory=False):
        """Orient the Pose with the prescribed symmetry at the origin and symmetry axes in canonical orientations
        self.symmetry is used to specify the orientation
        """
        if self.init_pdb:
            pdb = self.init_pdb
        else:
            pdb = PDB.from_file(self.source, log=self.log, pose_format=False)

        if self.design_symmetry:
            if to_design_directory:
                out_path = self.assembly_path
            else:
                out_path = os.path.join(self.orient_dir, '%s.pdb' % pdb.name)
                self.make_path(self.orient_dir)

            pdb.orient(symmetry=self.design_symmetry)

            orient_file = pdb.write(out_path=out_path)
            self.log.critical('The oriented file was saved to %s' % orient_file)
            # self.clear_pose_transformation()
            for entity in pdb.entities:
                entity.remove_mate_chains()
            self.load_pose(entities=pdb.entities)
            # self.save_asu(rename_chains=True)
        else:
            self.log.critical(PUtils.warn_missing_symmetry % self.orient.__name__)

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def refine(self, to_design_directory=False, interface_to_alanine=True, gather_metrics=False):
        """Refine the source PDB using self.symmetry to specify any symmetry"""
        main_cmd = copy.copy(script_cmd)
        stage = PUtils.refine
        if to_design_directory:  # original protocol to refine a pose as provided from Nanohedra
            # self.pose = Pose.from_pdb_file(self.source, symmetry=self.design_symmetry, log=self.log)
            # Todo unnecessary? call self.load_pose with a flag for the type of file? how to reconcile with interface
            #  design and the asu versus pdb distinction. Can asu be implied by symmetry? Not for a trimer input that
            #  needs to be oriented and refined
            # assign designable residues to interface1/interface2 variables, not necessary for non complex PDB jobs
            # try:
            self.identify_interface()
            # except DesignError:  # Todo handle when no interface residues are found and we just want refinement
            #     pass
            if interface_to_alanine:  # Mutate all design positions to Ala before the Refinement
                # mutated_pdb = copy.copy(self.pose.pdb)  # copy method implemented, but incompatible!
                # Have to use self.pose.pdb as Residue objects in entity_residues are from self.pose.pdb and not copy()!
                for entity_pair, interface_residue_sets in self.pose.interface_residues.items():
                    if interface_residue_sets[0]:  # check that there are residues present
                        for idx, interface_residue_set in enumerate(interface_residue_sets):
                            self.log.debug('Mutating residues from Entity %s' % entity_pair[idx].name)
                            for residue in interface_residue_set:
                                self.log.debug('Mutating %d%s' % (residue.number, residue.type))
                                if residue.type != 'GLY':  # no mutation from GLY to ALA as Rosetta will build a CB.
                                    self.pose.pdb.mutate_residue(residue=residue, to='A')

            # self.pose.pdb.write(out_path=self.refine_pdb)
            self.pose.write(out_path=self.refine_pdb)
            self.log.debug('Cleaned PDB for %s: \'%s\'' % (self.refine_pdb, stage.title()))
            flags = os.path.join(self.scripts, 'flags')
            flag_dir = self.scripts
            pdb_out_path = self.designs
            refine_pdb = self.refine_pdb
            refined_pdb = self.refined_pdb
            additional_flags = []
        else:  # protocol to refine input structures, place in a common location, then transform for many jobs to source
            flags = os.path.join(self.refine_dir, 'refine_flags')
            flag_dir = self.refine_dir
            pdb_out_path = self.refine_dir  # os.path.join(self.refine_dir, '%s.pdb' % self.name)
            refine_pdb = self.source
            refined_pdb = os.path.join(pdb_out_path, refine_pdb)
            additional_flags = ['-no_scorefile', 'true']

        if not os.path.exists(self.flags) or self.force_flags:  # Generate a new flags file
            # self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()  # needed for prepare_rosetta_flags -> self.center_residue_numbers
            self.make_path(flag_dir)
            self.make_path(pdb_out_path)
            flags = self.prepare_rosetta_flags(out_path=flag_dir, pdb_out_path=pdb_out_path)
            self.log.debug('Pose flags written to: %s' % flags)

        # RELAX: Prepare command
        relax_cmd = main_cmd + relax_flags_cmdline + additional_flags + \
            (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + \
            ['@%s' % flags, '-no_nstruct_label', 'true', '-in:file:s', refine_pdb,
             '-in:file:native', refine_pdb,  # native is here to block flag file version, not actually useful for refine
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % stage),
             '-parser:script_vars', 'switch=%s' % stage]
        self.log.info('%s Command: %s' % (stage.title(), list2cmdline(relax_cmd)))

        if gather_metrics:
            #                            nullify -native from flags v
            main_cmd += ['-in:file:s', refined_pdb, '@%s' % flags, '-in:file:native', refine_pdb,
                         '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
            if self.mpi:
                main_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + main_cmd
                self.run_in_shell = False

            metric_cmd_bound = main_cmd + (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []) + \
                [os.path.join(PUtils.rosetta_scripts, 'interface_%s%s.xml'
                              % (PUtils.stage[3], '_DEV' if self.development else ''))]
            entity_cmd = main_cmd + [os.path.join(PUtils.rosetta_scripts, '%s_entity%s.xml'
                                                  % (PUtils.stage[3], '_DEV' if self.development else ''))]
            metric_cmds = [metric_cmd_bound]
            for idx, entity in enumerate(self.pose.entities, 1):
                if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                    entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
                    entity_sym = 'symmetry=make_point_group'
                else:
                    entity_sdf, entity_sym = '', 'symmetry=asymmetric'
                _metric_cmd = entity_cmd + ['-parser:script_vars', 'repack=yes', 'entity=%d' % idx, entity_sym] + \
                    ([entity_sdf] if entity_sdf != '' else [])
                self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(_metric_cmd)))
                metric_cmds.append(_metric_cmd)
        else:
            metric_cmds = []

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if not self.run_in_shell:
            write_shell_script(list2cmdline(relax_cmd), name=stage, out_path=flag_dir,
                               additional=[list2cmdline(command) for command in metric_cmds])
            #                  status_wrap=self.serialized_info)
        else:
            relax_process = Popen(relax_cmd)
            relax_process.communicate()
            if gather_metrics:
                for metric_cmd in metric_cmds:
                    metrics_process = Popen(metric_cmd)
                    # Wait for Rosetta Design command to complete
                    metrics_process.communicate()

    @handle_design_errors(errors=(DesignError, AssertionError, FileNotFoundError))
    @close_logs
    @remove_structure_memory
    def find_asu(self):
        """From a PDB with multiple Chains from multiple Entities, return the minimal configuration of Entities.
        ASU will only be a true ASU if the starting PDB contains a symmetric system, otherwise all manipulations find
        the minimal unit of Entities that are in contact
        """
        if self.symmetric:  # if the symmetry isn't known then this wouldn't be a great option
            if os.path.exists(self.assembly_path):
                self.load_pose(source=self.assembly_path)
            else:
                self.load_pose()
            # self.save_asu()  # force saving the Pose.asu
        else:
            self.load_pose()
            raise DesignError('This might cause issues')
            # pdb = PDB.from_file(self.source, log=self.log)
            # asu = pdb.return_asu()
            # Todo ensure asu format matches pose.get_contacting_asu standard
            # asu.update_attributes_from_pdb(pdb)
            # asu_path.write(out_path=self.asu_path)
        self.save_asu()  # force saving the Pose.asu

    def symmetric_assembly_is_clash(self):
        """Wrapper around the Pose symmetric_assembly_is_clash() to check at the Design level for clashes and raise
        DesignError if any are found, otherwise, continue with protocol
        """
        if self.pose.symmetric_assembly_is_clash():
            if self.ignore_clashes:
                self.log.critical('The Symmetric Assembly contains clashes! %s is not viable.' % self.asu_path)
            else:
                raise DesignError('The Symmetric Assembly contains clashes! Design won\'t be considered. If you '
                                  'would like to generate the Assembly anyway, re-submit the command with '
                                  '--ignore_clashes')

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def expand_asu(self):
        """For the design info given by a DesignDirectory source, initialize the Pose with self.source file,
        self.symmetry, and self.log objects then expand the design given the provided symmetry operators and write to a
        file

        Reports on clash testing
        """
        if not self.pose:
            self.load_pose()
        if self.symmetric:
            self.symmetric_assembly_is_clash()
            self.pose.write(assembly=True, out_path=self.assembly_path, increment_chains=self.increment_chains)
            self.log.info('Symmetrically expanded assembly file written to: "%s"' % self.assembly_path)
        else:
            self.log.critical(PUtils.warn_missing_symmetry % self.expand_asu.__name__)
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def generate_interface_fragments(self):
        """For the design info given by a DesignDirectory source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation file
        """
        if not self.fragment_db:
            self.log.warning('There was no FragmentDatabase passed to the Design. But fragment information was '
                             'requested. Each design is loading a separate FragmentDatabase instance. To maximize '
                             'efficiency, pass --%s' % PUtils.generate_fragments)
        self.identify_interface()
        self.make_path(self.frags, condition=self.write_frags)
        self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=self.write_frags)
        self.fragment_observations = self.pose.return_fragment_observations()
        self.info['fragments'] = self.fragment_observations
        self.info['fragment_source'] = self.fragment_source
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    def identify_interface(self):
        """Initialize the design and find the interfaces between entities

        Sets:
            self.design_residue_ids (Dict[str, str]):
                Map each interface to the corresponding residue/chain pairs
            self.design_residues (Set[int]):
                The residues in proximity of the interface, including buried residues
            self.interface_residues (List[int]):
                The residues in contact across the interface
        """
        self.load_pose()
        if self.symmetric:
            self.symmetric_assembly_is_clash()
            if self.output_assembly:
                self.pose.write(assembly=True, out_path=self.assembly_path, increment_chains=self.increment_chains)
                self.log.info('Symmetrically expanded assembly file written to: "%s"' % self.assembly_path)
        self.pose.find_and_split_interface()

        self.design_residues = set()  # update False to set() or replace set() and attempt addition of new residues
        for number, residues_entities in self.pose.split_interface_residues.items():
            self.design_residue_ids['interface%d' % number] = \
                ','.join('%d%s' % (residue.number, entity.chain_id) for residue, entity in residues_entities)
            self.design_residues.update([residue.number for residue, _ in residues_entities])

        self.interface_residues = []  # update False to list or replace list and attempt addition of new residues
        for entity in self.pose.entities:  # Todo v clean as it is redundant with analysis and falls out of scope
            entity_oligomer = PDB.from_chains(entity.oligomer, log=self.log, pose_format=False, entities=False)
            entity_oligomer.get_sasa()
            # for residue_number in self.design_residues:
            for residue in entity_oligomer.get_residues(self.design_residues):
                # residue = entity_oligomer.residue(residue_number)
                # self.log.debug('Design residue: %d - SASA: %f' % (residue_number, residue.sasa))
                # if residue:
                if residue.sasa > 0:
                    self.interface_residues.append(residue.number)

        # # interface1, interface2 = \
        # #     self.design_residue_ids.get('interface1'), self.design_residue_ids.get('interface2')
        # interface_string = []
        # for idx, interface_info in enumerate(self.design_residue_ids.values()):
        #     if interface_info != '':
        #         interface_string.append('interface%d: %s' % (idx, interface_info))
        # if len(interface_string) == len(self.pose.split_interface_residues):
        #     self.log.info('Interface Residues:\n\t%s' % '\n\t'.join(interface_string))
        # else:
        #     self.log.info('No Residues found at the design interface!')

        self.info['design_residues'] = self.design_residues
        self.info['interface_residues'] = self.interface_residues
        self.info['design_residue_ids'] = self.design_residue_ids

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def interface_design(self):
        """For the design info given by a DesignDirectory source, initialize the Pose then prepare all parameters for
        interfacial redesign between Pose Entities. Aware of symmetry, design_selectors, fragments, and
        evolutionary information in interface design
        """
        self.identify_interface()
        if not self.command_only and not self.run_in_shell:  # just reissue the commands
            if self.query_fragments:
                self.make_path(self.frags)
            elif self.fragment_observations or self.fragment_observations == list():
                pass  # fragment generation was run and maybe succeeded. If not ^
            elif not self.no_term_constraint and not self.fragment_observations and os.path.exists(self.frag_file):
                self.retrieve_fragment_info_from_file()
            elif self.no_term_constraint:
                pass
            else:
                raise DesignError('Fragments were specified during design, but observations have not been yet been '
                                  'generated for this Design! Try with the flag --generate_fragments or run %s'
                                  % PUtils.generate_fragments)
            self.make_path(self.data)
            # creates all files which store the evolutionary_profile and/or fragment_profile -> design_profile
            self.pose.interface_design(evolution=not self.no_evolution_constraint,
                                       fragments=not self.no_term_constraint,
                                       query_fragments=self.query_fragments, fragment_source=self.fragment_observations,
                                       write_fragments=self.write_frags, des_dir=self)
            self.make_path(self.designs)
            self.fragment_observations = self.pose.return_fragment_observations()
            self.info['fragments'] = self.fragment_observations
            self.info['fragment_source'] = self.fragment_source

        if not self.pre_refine and not os.path.exists(self.refined_pdb):
            self.refine(to_design_directory=True)

        self.rosetta_interface_design()
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def optimize_designs(self, threshold: float = 0.):
        """To touch up and optimize a design, provide a list of optional directives to view mutational landscape around
        certain residues in the design as well as perform wild-type amino acid reversion to mutated residues

        Args:
            # residue_directives=None (dict[mapping[Union[Residue,int],str]]):
            #     {Residue object: 'mutational_directive', ...}
            # design_file=None (str): The name of a particular design file present in the designs output
            threshold: The threshold above which background amino acid frequencies are allowed for mutation
        """
        self.load_pose()
        # format all amino acids in self.design_residues with frequencies above the threshold to a set
        # Todo, make threshold and return set of strings a property of a profile object
        background = \
            {self.pose.pdb.residue(residue_number):
             {protein_letters_1to3.get(aa).upper() for aa in protein_letters_1to3 if fields.get(aa, -1) > threshold}
             for residue_number, fields in self.design_background.items() if residue_number in self.design_residues}
        # include the wild-type residue from DesignDirectory Pose source and the residue identity of the selected design
        wt = {residue: {self.design_background[residue.number].get('type'), protein_letters_3to1[residue.type.title()]}
              for residue in background}
        directives = dict(zip(background.keys(), repeat(None)))
        directives.update({self.pose.pdb.residue(residue_number): directive
                           for residue_number, directive in self.directives.items()})

        res_file = self.pose.pdb.make_resfile(directives, out_path=self.data, include=wt, background=background)

        protocol = 'optimize_design'
        protocol_xml1 = protocol
        # nstruct_instruct = ['-no_nstruct_label', 'true']
        nstruct_instruct = ['-nstruct', str(self.number_of_trajectories)]
        design_list_file = os.path.join(self.scripts, 'design_files_%s.txt' % protocol)
        generate_files_cmd = \
            ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', design_list_file, '-s', '_' + protocol]

        main_cmd = copy.copy(script_cmd)
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
        if not os.path.exists(self.flags) or self.force_flags:
            # self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()  # needed for prepare_rosetta_flags -> self.center_residue_numbers
            self.make_path(self.scripts)
            self.flags = self.prepare_rosetta_flags(out_path=self.scripts)
            self.log.debug('Pose flags written to: %s' % self.flags)

        # DESIGN: Prepare command and flags file
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        design_cmd = main_cmd + \
            (['-in:file:pssm', self.evolutionary_profile_file] if self.evolutionary_profile else []) + \
            ['-in:file:s', self.specific_design_path if self.specific_design_path else self.refined_pdb,
             '@%s' % self.flags, '-out:suffix', '_%s' % protocol, '-packing:resfile', res_file,
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % protocol_xml1)] + nstruct_instruct

        # metrics_pdb = ['-in:file:l', design_list_file]  # self.pdb_list]
        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        # metric_cmd = main_cmd + ['-in:file:s', self.specific_design if self.specific_design else self.refined_pdb] + \
        entity_cmd = main_cmd + ['-in:file:l', design_list_file] + \
            ['@%s' % self.flags, '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, 'metrics_entity.xml')]

        if self.mpi:
            design_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + design_cmd
            entity_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + entity_cmd
            self.run_in_shell = False

        self.log.info('Design Command: %s' % list2cmdline(design_cmd))
        metric_cmds = []
        for idx, entity in enumerate(self.pose.entities, 1):
            if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
                entity_sym = 'symmetry=make_point_group'
            else:
                entity_sdf, entity_sym = '', 'symmetry=asymmetric'
            _metric_cmd = entity_cmd + ['-parser:script_vars', 'repack=yes', 'entity=%d' % idx, entity_sym] + \
                ([entity_sdf] if entity_sdf != '' else [])
            self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(_metric_cmd)))
            metric_cmds.append(_metric_cmd)
        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if not self.run_in_shell:
            write_shell_script(list2cmdline(design_cmd), name=protocol, out_path=self.scripts,
                               additional=[list2cmdline(generate_files_cmd)] +
                                          [list2cmdline(command) for command in metric_cmds])

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    @remove_structure_memory
    def design_analysis(self, merge_residue_data: bool = False, save_trajectories: bool = True, figures: bool = False) \
            -> pd.Series:  # Todo interface_design_analysis
        """Retrieve all score information from a DesignDirectory and write results to .csv file

        Keyword Args:
            merge_residue_data: Whether to incorporate residue data into Pose DataFrame
            save_trajectories: Whether to save trajectory and residue DataFrames
            figures: Whether to make and save pose figures
        Returns:
            Series containing summary metrics for all designs in the design directory
        """
        if self.interface_residues is False or self.design_residues is False:
            self.identify_interface()
        else:  # we only need to pose active as we already calculated these
            self.load_pose()
        self.log.debug('Found design residues: %s' % ', '.join(map(str, sorted(self.design_residues))))
        if self.query_fragments:
            self.make_path(self.frags, condition=self.write_frags)
            self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=self.write_frags)

        # Gather miscellaneous pose specific metrics
        other_pose_metrics = self.pose_metrics()

        # Find all designs files Todo fold these into Model(s) and attack metrics from Pose objects?
        design_structures = []
        for file in self.get_designs():
            # decoy_name = os.path.splitext(os.path.basename(file))[0]  # name should match scored designs...
            #                     pass names if available v
            design = PDB.from_file(file, entity_names=self.entity_names, log=self.log)  # name=decoy_name,
            if self.symmetric:
                for idx, entity in enumerate(design.entities):
                    entity.make_oligomer(symmetry=self.sym_entry.groups[idx], **self.pose_transformation[idx])
            design_structures.append(design)

        # initialize empty design dataframes
        idx_slice = pd.IndexSlice
        pose_length = self.pose.number_of_residues
        residue_indices = list(range(1, pose_length + 1))
        pose_sequences = {structure.name: structure.sequence for structure in design_structures}
        all_mutations = generate_mutations_from_reference(self.pose.sequence, pose_sequences)
        #    generate_mutations_from_reference(''.join(self.pose.pdb.atom_sequences.values()), pose_sequences)

        # Assumes each structure is the same length
        entity_sequences = \
            {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
                      for design, sequence in pose_sequences.items()} for entity in self.pose.entities}
        # Todo generate_multiple_mutations accounts for offsets from the reference sequence. Not necessary YET
        # sequence_mutations = \
        #     generate_multiple_mutations(self.pose.pdb.atom_sequences, pose_sequences, pose_num=False)
        # sequence_mutations.pop('reference')
        # entity_sequences = generate_sequences(self.pose.pdb.atom_sequences, sequence_mutations)
        # entity_sequences = {chain: keys_from_trajectory_number(named_sequences)
        #                         for chain, named_sequences in entity_sequences.items()}
        wt_design_info = {residue.number: {'energy_delta': 0., 'type': protein_letters_3to1.get(residue.type.title()),
                                           'hbond': 0} for entity in self.pose.entities for residue in entity.residues}
        residue_info = {'wild_type': wt_design_info}
        stat_s, sim_series = pd.Series(), []
        if not os.path.exists(self.scores_file):  # Rosetta scores file isn't present
            self.log.debug('Missing design scores file at %s' % self.scores_file)
            # Todo add relevant missing scores such as those specified as 0 below
            scores_df = pd.DataFrame({structure.name: {PUtils.groups: 'no_metrics'}  # 'metric_keys': 'metric_values'}
                                      for structure in design_structures}).T
            for idx, entity in enumerate(self.pose.entities):
                scores_df['buns_%d_unbound' % idx] = 0
                scores_df['interface_energy_%d_bound' % idx] = 0
                scores_df['interface_energy_%d_unbound' % idx] = 0
                scores_df['solvation_energy_%d_bound' % idx] = 0
                scores_df['solvation_energy_%d_unbound' % idx] = 0
                scores_df['interface_connectivity_%d' % idx] = 0
                # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
                # design_info.update({residue.number: {'energy_delta': 0., 'type': protein_letters_3to1.get(residue.type.title()),
                #                          'hbond': 0} for residue in entity.residues})
            scores_df['number_hbonds'] = 0
            protocol_s = scores_df[PUtils.groups]
            # Todo generate the per residue scores internally which matches output from dirty_residue_processing
            # interface_hbonds = dirty_hbond_processing(all_design_scores)
            residue_info.update({structure_name: wt_design_info for structure_name in scores_df.index.to_list()})
            # residue_info = dirty_residue_processing(all_design_scores, simplify_mutation_dict(all_mutations),
            #                                             hbonds=interface_hbonds)
        else:  # Get the scores from the score file on design trajectory metrics
            self.log.debug('Found design scores in file: %s' % self.scores_file)
            all_design_scores = read_scores(self.scores_file)
            self.log.debug('All designs with scores: %s' % ', '.join(all_design_scores.keys()))

            # Gather mutations for residue specific processing and design sequences
            for design, data in list(all_design_scores.items()):  # make a copy as is removed if no sequence present
                sequence = data.get('final_sequence')
                if sequence:
                    if len(sequence) >= pose_length:
                        pose_sequences[design] = sequence[:pose_length]  # Todo won't work if the design had insertions
                    else:
                        pose_sequences[design] = sequence
                else:
                    self.log.warning('Design %s is missing sequence data, removing from design pool' % design)
                    all_design_scores.pop(design)
            # format {entity: {design_name: sequence, ...}, ...}
            entity_sequences = \
                {entity: {design: sequence[entity.n_terminal_residue.number - 1:entity.c_terminal_residue.number]
                          for design, sequence in pose_sequences.items()} for entity in self.pose.entities}

            scores_df = pd.DataFrame(all_design_scores).T
            # Gather all columns into specific types for processing and formatting
            per_res_columns, hbonds_columns = [], []
            for column in scores_df.columns.to_list():
                if column.startswith('per_res_'):
                    per_res_columns.append(column)
                elif column.startswith('hbonds_res_selection'):
                    hbonds_columns.append(column)

            # Check proper input
            metric_set = necessary_metrics.difference(set(scores_df.columns))
            # self.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
            assert metric_set == set(), 'Missing required metrics: %s' % metric_set
            # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
            protocol_s = scores_df[PUtils.groups]
            # protocol_s.replace({'combo_profile': PUtils.design_profile}, inplace=True)  # ensure proper profile name

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # TODO learn know how to produce score terms in output score file. Not in FastRelax...
            remove_columns = per_res_columns + hbonds_columns + rosetta_terms + unnecessary + [PUtils.groups]
            scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')
            # TODO remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            interface_hbonds = dirty_hbond_processing(all_design_scores)
            # can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
            # interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)
            number_hbonds_s = \
                pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()}, name='number_hbonds')
            # scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)
            scores_df = scores_df.assign(number_hbonds=number_hbonds_s)
            # residue_info = {'energy': {'complex': 0., 'unbound': 0.}, 'type': None, 'hbond': 0}
            residue_info.update(dirty_residue_processing(all_design_scores, simplify_mutation_dict(all_mutations),
                                                         hbonds=interface_hbonds))
            # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
            # residue_info.update(residue_processing(all_design_scores, simplify_mutation_dict(all_mutations),
            #                                        per_res_columns, hbonds=interface_hbonds))

        # Find designs where required data is present
        viable_designs = scores_df.index.to_list()
        assert viable_designs, 'No viable designs remain after processing!'
        self.log.debug('Viable designs remaining after cleaning:\n\t%s' % ', '.join(viable_designs))
        # Find protocols for protocol specific data processing
        unique_protocols = protocol_s.unique().tolist()
        protocol_dict = {idx: protocol for idx, protocol in enumerate(unique_protocols)}
        designs_by_protocol = protocol_s.groupby(protocol_dict).indices
        # remove refine and consensus if present as there was no design done over multiple protocols
        designs_by_protocol.pop(PUtils.refine, None)
        designs_by_protocol.pop(PUtils.stage[5], None)
        # Get unique protocols
        unique_design_protocols = set(designs_by_protocol.keys())
        self.log.info('Unique Design Protocols: %s' % ', '.join(unique_design_protocols))

        other_pose_metrics['observations'] = len(viable_designs)
        pose_sequences = filter_dictionary_keys(pose_sequences, viable_designs)
        # Replace empty strings with numpy.notanumber (np.nan) and convert remaining to float
        scores_df.replace('', np.nan, inplace=True)
        scores_df.fillna(dict(zip(protocol_specific_columns, repeat(0))), inplace=True)
        scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

        # design_assemblies = []  # Todo use to store the poses generated below?
        # per residue data includes every residue in the pose
        per_residue_data = {'errat_deviation': {}, 'hydrophobic_collapse': {},
                            'sasa_hydrophobic_complex': {}, 'sasa_polar_complex': {}, 'sasa_relative_complex': {},
                            'sasa_hydrophobic_bound': {}, 'sasa_polar_bound': {}, 'sasa_relative_bound': {}}
        # 'local_density': {},
        pose_assembly_minimally_contacting = self.pose.assembly_minimally_contacting
        # Favor errat/collapse measurement on a per Entity basis because the wild type is assuming no interface present
        # but the design has the interface
        # atomic_deviation[wild_type] per_residue_errat = pose_assembly_minimally_contacting.errat(out_path=self.data)
        # per_residue_data['errat_deviation']['wild_type'] = per_residue_errat[:pose_length]
        # perform SASA measurements
        pose_assembly_minimally_contacting.get_sasa()
        # per_residue_sasa = [residue.sasa for residue in structure.residues
        #                     if residue.number in self.design_residues]
        per_residue_sasa_complex_apolar = \
            [residue.sasa_apolar for residue in pose_assembly_minimally_contacting.residues[:pose_length]]
        per_residue_sasa_complex_polar = \
            [residue.sasa_polar for residue in pose_assembly_minimally_contacting.residues[:pose_length]]
        per_residue_sasa_complex_relative = \
            [residue.relative_sasa for residue in pose_assembly_minimally_contacting.residues[:pose_length]]
        per_residue_data['sasa_hydrophobic_complex']['wild_type'] = per_residue_sasa_complex_apolar
        per_residue_data['sasa_polar_complex']['wild_type'] = per_residue_sasa_complex_polar
        per_residue_data['sasa_relative_complex']['wild_type'] = per_residue_sasa_complex_relative
        per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = \
            [], [], []
        # Grab metrics for the wild-type file. Assumes self.pose is from non-designed sequence
        msa_metrics = True
        collapse_df, wt_errat, wt_collapse, wt_collapse_bool, wt_collapse_z_score = {}, {}, {}, {}, {}
        inverse_residue_contact_order_z, contact_order = {}, {}
        for entity in self.pose.entities:
            # we need to get the contact order, errat from the symmetric entity
            # entity.oligomer.get_sasa()
            entity_oligomer = PDB.from_chains(entity.oligomer, log=self.log, entities=False)
            entity_oligomer.get_sasa()
            per_residue_sasa_unbound_apolar.extend(
                [residue.sasa_apolar for residue in entity_oligomer.residues[:entity.number_of_residues]])
            per_residue_sasa_unbound_polar.extend(
                [residue.sasa_polar for residue in entity_oligomer.residues[:entity.number_of_residues]])
            per_residue_sasa_unbound_relative.extend(
                [residue.relative_sasa for residue in entity_oligomer.residues[:entity.number_of_residues]])
        # for entity in self.pose.entities:
            # entity_oligomer = PDB.from_chains(entity.oligomer, log=self.log, entities=False)
            # Contact order is the same for every design in the Pose
            # Todo clean this behavior up as it is not good if entity is used downstream...
            #  for contact order we must give a copy of coords_indexed_residues from the pose to each entity...
            entity.coords_indexed_residues = self.pose.pdb._coords_indexed_residues
            residue_contact_order = entity.contact_order
            # residue_contact_order = entity_oligomer.contact_order[:entity.number_of_residues]
            contact_order[entity] = residue_contact_order  # save the contact order for plotting
            _, oligomeric_errat = entity_oligomer.errat(out_path=self.data)
            wt_errat[entity] = oligomeric_errat[:entity.number_of_residues]
            # residue_contact_order_mean, residue_contact_order_std = \
            #     residue_contact_order.mean(), residue_contact_order.std()
            # print('%s residue_contact_order' % entity.name, residue_contact_order)
            # temporary contact order debugging
            # print(residue_contact_order)
            # entity.contact_order = residue_contact_order
            # entity.set_residues_attributes_from_array(collapse=wt_collapse[entity])
            # entity.set_b_factor_data(dtype='collapse')
            # entity.write_oligomer(out_path=os.path.join(self.path, '%s_collapse.pdb' % entity.name))
            residue_contact_order_z = \
                z_score(residue_contact_order, residue_contact_order.mean(), residue_contact_order.std())
            inverse_residue_contact_order_z[entity] = residue_contact_order_z * -1
            wt_collapse[entity] = hydrophobic_collapse_index(entity.sequence)  # TODO comment out, instate below?
            wt_collapse_bool[entity] = np.where(wt_collapse[entity] > collapse_significance_threshold, 1, 0)
            # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
            # entity = self.resources.refined.retrieve_data(name=entity.name))  # Todo always use wild-type?
            entity.msa = self.resources.alignments.retrieve_data(name=entity.name)
            # entity.h_fields = self.resources.bmdca_fields.retrieve_data(name=entity.name)  # Todo reinstate
            # entity.j_couplings = self.resources.bmdca_couplings.retrieve_data(name=entity.name)  # Todo reinstate
            if msa_metrics:
                if not entity.msa:
                    self.log.info('Metrics relying on a multiple sequence alignment are not being collected as '
                                  'there is no MSA found. These include: %s'
                                  % ', '.join(multiple_sequence_alignment_dependent_metrics))
                    # set anything found to null values
                    collapse_df, wt_collapse_z_score = {}, {}
                    msa_metrics = False
                    continue
                collapse = entity.collapse_profile()  # takes ~5-10 seconds depending on the size of the msa
                collapse_df[entity] = collapse
                # wt_collapse[entity] = hydrophobic_collapse_index(self.resources.sequences.retrieve_data(name=entity.name))
                wt_collapse_z_score[entity] = z_score(wt_collapse[entity], collapse.loc[mean, :], collapse.loc[std, :])
        per_residue_data['sasa_hydrophobic_bound']['wild_type'] = per_residue_sasa_unbound_apolar
        per_residue_data['sasa_polar_bound']['wild_type'] = per_residue_sasa_unbound_polar
        per_residue_data['sasa_relative_bound']['wild_type'] = per_residue_sasa_unbound_relative

        wt_errat_concat_s = pd.Series(np.concatenate(list(wt_errat.values())), index=residue_indices)  #, name='wild_type')
        wt_collapse_concat_s = pd.Series(np.concatenate(list(wt_collapse.values())), index=residue_indices)
        per_residue_data['errat_deviation']['wild_type'] = wt_errat_concat_s
        per_residue_data['hydrophobic_collapse']['wild_type'] = wt_collapse_concat_s
        # now that wildtype is included, don't need this anymore...
        # errat_collapse_df = \
        #     pd.concat([pd.concat(dict(errat_deviation=wt_errat_concat_s, hydrophobic_collapse=wt_collapse_concat_s))],
        #               keys=['wild_type']).unstack().unstack()  # .swaplevel(0, 1, axis=1)

        interface_local_density, atomic_deviation = {}, {}
        for structure in design_structures:  # Takes 1-2 seconds for Structure -> assembly -> errat
            if structure.name not in viable_designs:
                continue
            if self.symmetric:
                design_pose = Pose.from_asu(structure, sym_entry=self.sym_entry, name='%s-asu' % structure.name,
                                            design_selector=self.design_selector, log=self.log,
                                            source_db=self.resources, fragment_db=self.fragment_db,
                                            euler_lookup=self.euler_lookup, ignore_clashes=self.ignore_clashes)
            else:
                design_pose = Pose.from_pdb(structure, name='%s-asu' % structure.name,
                                            design_selector=self.design_selector, log=self.log,
                                            source_db=self.resources, fragment_db=self.fragment_db,
                                            euler_lookup=self.euler_lookup, ignore_clashes=self.ignore_clashes)

            # assembly = SymmetricModel.from_asu(structure, sym_entry=self.sym_entry, log=self.log).assembly
            #                                            ,symmetry=self.design_symmetry)

            # assembly.local_density()[:pose_length]  To get every residue in the pose.entities
            # per_residue_data['local_density'][structure.name] = \
            #     [density for residue_number, density in enumerate(assembly.local_density(), 1)
            #      if residue_number in self.design_residues]  # self.interface_residues <- no interior, mas accurate?
            # per_residue_data['local_density'][structure.name] = \
            #     assembly.local_density(residue_numbers=self.interface_residues)[:pose_length]

            # must find interface residues before measure local_density
            design_pose.find_and_split_interface()
            interface_local_density[structure.name] = design_pose.interface_local_density()
            assembly_minimally_contacting = design_pose.assembly_minimally_contacting
            atomic_deviation[structure.name], per_residue_errat = \
                assembly_minimally_contacting.errat(out_path=self.data)
            per_residue_data['errat_deviation'][structure.name] = per_residue_errat[:pose_length]
            # perform SASA measurements
            assembly_minimally_contacting.get_sasa()
            # per_residue_sasa = [residue.sasa for residue in structure.residues
            #                     if residue.number in self.design_residues]
            per_residue_sasa_complex_apolar = \
                [residue.sasa_apolar for residue in assembly_minimally_contacting.residues[:pose_length]]
            per_residue_sasa_complex_polar = \
                [residue.sasa_polar for residue in assembly_minimally_contacting.residues[:pose_length]]
            per_residue_sasa_complex_relative = \
                [residue.relative_sasa for residue in assembly_minimally_contacting.residues[:pose_length]]
            per_residue_data['sasa_hydrophobic_complex'][structure.name] = per_residue_sasa_complex_apolar
            per_residue_data['sasa_polar_complex'][structure.name] = per_residue_sasa_complex_polar
            per_residue_data['sasa_relative_complex'][structure.name] = per_residue_sasa_complex_relative
            per_residue_sasa_unbound_apolar, per_residue_sasa_unbound_polar, per_residue_sasa_unbound_relative = \
                [], [], []
            for entity in structure.entities:
                # entity.oligomer.get_sasa()
                entity_oligomer = PDB.from_chains(entity.oligomer, log=self.log, entities=False)
                entity_oligomer.get_sasa()
                per_residue_sasa_unbound_apolar.extend(
                    [residue.sasa_apolar for residue in entity_oligomer.residues[:entity.number_of_residues]])
                per_residue_sasa_unbound_polar.extend(
                    [residue.sasa_polar for residue in entity_oligomer.residues[:entity.number_of_residues]])
                per_residue_sasa_unbound_relative.extend(
                    [residue.relative_sasa for residue in entity_oligomer.residues[:entity.number_of_residues]])
            per_residue_data['sasa_hydrophobic_bound'][structure.name] = per_residue_sasa_unbound_apolar
            per_residue_data['sasa_polar_bound'][structure.name] = per_residue_sasa_unbound_polar
            per_residue_data['sasa_relative_bound'][structure.name] = per_residue_sasa_unbound_relative

        scores_df['errat_accuracy'] = pd.Series(atomic_deviation)
        scores_df['interface_local_density'] = pd.Series(interface_local_density)

        # Calculate hydrophobic collapse for each design
        # Measure the wild type entity versus the modified entity to find the hci delta
        # Todo if no design, can't measure the wild-type after the normalization...

        # A measure of the sequential, the local, the global, and the significance all constitute interesting
        # parameters which contribute to the outcome. I can use the measure of each to do a post-hoc solubility
        # analysis. In the meantime, I could stay away from any design which causes the global collapse to increase
        # by some percent of total relating to the z-score. This could also be an absolute which would tend to favor
        # smaller proteins. Favor smaller or larger? What is the literature/data say about collapse?
        #
        # A synopsis of my reading is as follows:
        # I hypothesize that the worst offenders in collapse modification will be those that increase in
        # hydrophobicity in sections intended for high contact order packing. Additionally, the establishment of new
        # collapse locales will be detrimental to the folding pathway regardless of their location, however
        # establishment in folding locations before a significant protein core is established are particularly
        # egregious. If there is already collapse occurring, the addition of new collapse could be less important as
        # the structural determinants (geometric satisfaction) of the collapse are not as significant
        #
        # All possible important aspects measured are:
        # X the sequential collapse (earlier is worse than later as nucleation of core is wrong),
        #   sequential_collapse_peaks_z_sum, sequential_collapse_z_sum
        # X the local nature of collapse (is the sequence/structural context amenable to collapse?),
        #   contact_order_collapse_z_sum
        # X the global nature of collapse (how much has collapse increased globally),
        #   hydrophobicity_deviation_magnitude, global_collapse_z_sum,
        # X the change from "non-collapsing" to "collapsing" where collapse passes a threshold and changes folding
        #   new_collapse_islands, new_collapse_island_significance

        # linearly weight residue by sequence position (early > late) with the halfway position (midpoint)
        # weighted at 1
        midpoint = 0.5
        scale = 1 / midpoint
        folding_and_collapse = \
            {'hydrophobicity_deviation_magnitude': {}, 'new_collapse_islands': {},
             'new_collapse_island_significance': {}, 'contact_order_collapse_z_sum': {},
             'sequential_collapse_peaks_z_sum': {}, 'sequential_collapse_z_sum': {}, 'global_collapse_z_sum': {}}
        for design in viable_designs:
            hydrophobicity_deviation_magnitude, new_collapse_islands, new_collapse_island_significance = [], [], []
            contact_order_collapse_z_sum, sequential_collapse_peaks_z_sum, sequential_collapse_z_sum, \
                global_collapse_z_sum, collapse_concatenated = [], [], [], [], []
            for entity in self.pose.entities:
                sequence = entity_sequences[entity][design]
                standardized_collapse = hydrophobic_collapse_index(sequence)
                collapse_concatenated.append(standardized_collapse)
                # Todo -> observed_collapse, standardized_collapse = hydrophobic_collapse_index(sequence)
                # normalized_collapse = standardized_collapse - wt_collapse[entity]
                # find collapse where: delta above standard collapse, collapsable boolean, and successive number
                # collapse_propensity = np.where(standardized_collapse > 0.43, standardized_collapse - 0.43, 0)
                # scale the collapse propensity by the standard collapse threshold and make z score
                collapse_propensity_z = z_score(standardized_collapse, collapse_significance_threshold, 0.05)
                collapse_propensity_positive_z_only = np.where(collapse_propensity_z > 0, collapse_propensity_z, 0)
                # ^ [0, 0, 0, 0, 0.04, 0.06, 0, 0, 0.1, 0.07, ...]
                # collapse_bool = np.where(standardized_collapse > 0.43, 1, 0)  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
                collapse_bool = np.where(collapse_propensity_positive_z_only, 1, 0)  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
                increased_collapse = np.where(collapse_bool - wt_collapse_bool[entity] == 1, 1, 0)
                # check if the increased collapse has made new collapse
                new_collapse = np.zeros(collapse_bool.shape)  # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ...]
                for idx, _bool in enumerate(increased_collapse.tolist()[1:-1], 1):
                    if _bool and (not wt_collapse_bool[entity][idx - 1] or not wt_collapse_bool[entity][idx + 1]):
                        new_collapse[idx] = _bool
                # new_collapse are sites where a new collapse is formed compared to wild-type

                # # we must give a copy of coords_indexed_residues from the pose to each entity...
                # entity.coords_indexed_residues = self.pose.pdb._coords_indexed_residues
                # residue_contact_order = entity.contact_order
                # contact_order_concatenated.append(residue_contact_order)
                # inverse_residue_contact_order = max(residue_contact_order) - residue_contact_order
                # residue_contact_order_mean, residue_contact_order_std = \
                #     residue_contact_order[entity].mean(), residue_contact_order[entity].std()
                # residue_contact_order_z = \
                #     z_score(residue_contact_order, residue_contact_order_mean, residue_contact_order_std)
                # inverse_residue_contact_order_z = residue_contact_order_z * -1

                # use the contact order (or inverse) to multiply by hci in order to understand the designability of
                # the specific area and its resulting folding modification
                # The multiplication by positive collapsing z-score will indicate the degree to which low contact
                # order stretches are reliant on collapse as a folding mechanism, while high contact order are
                # negative and the locations of highly negative values indicate high contact order use of collapse
                collapse_significance = inverse_residue_contact_order_z[entity] * collapse_propensity_positive_z_only

                collapse_peak_start = np.zeros(collapse_bool.shape)  # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...]
                sequential_collapse_points = np.zeros(collapse_bool.shape)  # [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, ...]
                new_collapse_peak_start = np.zeros(collapse_bool.shape)  # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...]
                collapse_iterator = 0
                for idx, _ in enumerate(collapse_bool.tolist()[1:], 1):  # peak_value
                    # check for the new_collapse islands and collapse peak start position by neighbor res similarity
                    if new_collapse[idx] > new_collapse[idx - 1]:  # only true when 0 -> 1 transition
                        new_collapse_peak_start[idx] = 1
                    if collapse_bool[idx] > collapse_bool[idx - 1]:  # only true when 0 -> 1 transition
                        collapse_peak_start[idx] = 1
                        collapse_iterator += 1
                    sequential_collapse_points[idx] = collapse_iterator
                sequential_collapse_points *= collapse_bool  # reduce sequential collapse iter to collapse points

                # for idx, _ in enumerate(collapse_bool):  # peak_value
                #     total_collapse_points * collapse_bool
                # sequential_collapse_weights = \
                #     scale * (1 - (total_collapse_points * sequential_collapse_points / total_collapse_points))
                total_collapse_points = collapse_peak_start.sum()
                step = 1 / total_collapse_points
                add_step_array = collapse_bool * step
                # v [0, 0, 0, 0, 2, 2, 0, 0, 1.8, 1.8, ...]
                sequential_collapse_weights = scale * ((1 - (step * sequential_collapse_points)) + add_step_array)
                # v [2, 1.98, 1.96, 1.94, 1.92, ...]
                sequential_weights = scale * (1 - (np.arange(entity.number_of_residues) / entity.number_of_residues))

                new_collapse_islands.append(new_collapse_peak_start.sum())
                new_collapse_island_significance.append(sum(new_collapse_peak_start * abs(collapse_significance)))

                if msa_metrics:
                    z_array = z_score(standardized_collapse,  # observed_collapse,
                                      collapse_df[entity].loc[mean, :], collapse_df[entity].loc[std, :])
                    # todo test for magnitude of the wt versus profile, remove subtraction?
                    normalized_collapse_z = z_array - wt_collapse_z_score[entity]
                    hydrophobicity_deviation_magnitude.append(sum(abs(normalized_collapse_z)))
                    global_collapse_z = np.where(normalized_collapse_z > 0, normalized_collapse_z, 0)
                    # offset inverse_residue_contact_order_z to center at 1 instead of 0. Todo deal with negatives
                    contact_order_collapse_z_sum.append(
                        sum((inverse_residue_contact_order_z[entity] + 1) * global_collapse_z))
                    sequential_collapse_peaks_z_sum.append(sum(sequential_collapse_weights * global_collapse_z))
                    sequential_collapse_z_sum.append(sum(sequential_weights * global_collapse_z))
                    global_collapse_z_sum.append(global_collapse_z.sum())

            # add the total and concatenated metrics to analysis structures
            folding_and_collapse['hydrophobicity_deviation_magnitude'][design] = sum(hydrophobicity_deviation_magnitude)
            folding_and_collapse['new_collapse_islands'][design] = sum(new_collapse_islands)
            # takes into account new collapse positions contact order and measures the deviation of collapse and
            # contact order to indicate the potential effect to folding
            folding_and_collapse['new_collapse_island_significance'][design] = sum(new_collapse_island_significance)
            folding_and_collapse['contact_order_collapse_z_sum'][design] = sum(contact_order_collapse_z_sum)
            folding_and_collapse['sequential_collapse_peaks_z_sum'][design] = sum(sequential_collapse_peaks_z_sum)
            folding_and_collapse['sequential_collapse_z_sum'][design] = sum(sequential_collapse_z_sum)
            folding_and_collapse['global_collapse_z_sum'][design] = sum(global_collapse_z_sum)
            # collapse_concatenated = np.concatenate(collapse_concatenated)
            collapse_concatenated = pd.Series(np.concatenate(collapse_concatenated), name=design)
            per_residue_data['hydrophobic_collapse'][design] = collapse_concatenated

        pose_collapse_df = pd.DataFrame(folding_and_collapse)
        # turn per_residue_data into a dataframe matching residue_df orientation
        per_residue_df = pd.concat({measure: pd.DataFrame(data, index=residue_indices)
                                    for measure, data in per_residue_data.items()}).T.swaplevel(0, 1, axis=1)
        # With the per_residue_df constructed with wild_type, many metric instances should remove this entry
        not_wt_indices = per_residue_df.index != 'wild_type'
        errat_df = per_residue_df.loc[not_wt_indices, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
        # include if errat score is < 2 std devs and isn't 0.  TODO what about measuring wild-type when no design?
        wt_errat_inclusion_boolean = np.logical_and(wt_errat_concat_s < errat_2_sigma, wt_errat_concat_s != 0.)
        # print('SEPARATE', (wt_errat_concat_s < errat_2_sigma)[30:40], (wt_errat_concat_s != 0.)[30:40])
        # print('LOGICAL AND\n', wt_errat_inclusion_boolean[30:40])
        # errat_sig_df = (errat_df > errat_2_sigma)
        # find where designs deviate above wild-type errat scores
        # print('errat_df', errat_df.iloc[:5, 30:40])
        # print('wt_errat_concat_s', wt_errat_concat_s[30:40])
        # print('SUBTRACTION', errat_df.sub(wt_errat_concat_s, axis=1).iloc[:5, 30:40])
        errat_sig_df = (errat_df.sub(wt_errat_concat_s, axis=1)) > errat_1_sigma  # axis=1 Series is column oriented
        # print('errat_sig_df', errat_sig_df.iloc[:5, 30:40])
        # then select only those residues which are expressly important by the inclusion boolean
        errat_design_significance = errat_sig_df.loc[:, wt_errat_inclusion_boolean].any(axis=1)
        # print('SIGNIFICANCE', errat_design_significance)
        # errat_design_residue_significance = errat_sig_df.loc[:, wt_errat_inclusion_boolean].any(axis=0)
        # print('RESIDUE SIGNIFICANCE', errat_design_residue_significance[errat_design_residue_significance].index.tolist())
        pose_collapse_df['errat_deviation'] = errat_design_significance
        # significant_errat_residues = per_residue_df.index[].remove_unused_levels().levels[0].to_list()

        # Get design information including: interface residues, SSM's, and wild_type/design files
        profile_background = {}
        if self.design_profile:
            profile_background['design'] = self.design_profile
        # else:
        #     self.log.info('Design has no fragment information')
        if self.evolutionary_profile:
            profile_background['evolution'] = self.evolutionary_profile
        else:
            self.log.info('Design has no evolution information')
        if self.fragment_data:
            profile_background['fragment'] = self.fragment_data
        else:
            self.log.info('Design has no fragment information')

        if not profile_background:
            divergence_s = pd.Series()
        else:
            # Calculate amino acid observation percent from residue_info and background SSM's
            observation_d = {profile: {design: mutation_conserved(info, background)
                                       for design, info in residue_info.items()}
                             for profile, background in profile_background.items()}
            # Find the observed background for each profile, for each design in the pose
            pose_observed_bkd = {profile: {design: per_res_metric(freq) for design, freq in design_obs_freqs.items()}
                                 for profile, design_obs_freqs in observation_d.items()}
            for profile, observed_frequencies in pose_observed_bkd.items():
                scores_df['observed_%s' % profile] = pd.Series(observed_frequencies)
            # Add observation information into the residue dictionary
            for design, info in residue_info.items():
                residue_info[design] = \
                    weave_sequence_dict(base_dict=info, **{'observed_%s' % profile: design_obs_freqs[design]
                                                           for profile, design_obs_freqs in observation_d.items()})
            # Calculate sequence statistics
            # first for entire pose
            pose_alignment = msa_from_dictionary(pose_sequences)
            mutation_frequencies = filter_dictionary_keys(pose_alignment.frequencies, self.design_residues)
            # mutation_frequencies = filter_dictionary_keys(pose_alignment['frequencies'], interface_residues)
            # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
            #                                              both mut_freq and profile_background[profile] are one-indexed
            divergence = {'divergence_%s' % profile: position_specific_jsd(mutation_frequencies, background)
                          for profile, background in profile_background.items()}
            interface_bkgd = self.fragment_db.get_db_aa_frequencies()
            if interface_bkgd:
                divergence['divergence_interface'] = jensen_shannon_divergence(mutation_frequencies, interface_bkgd)
            # Get pose sequence divergence
            divergence_stats = {'%s_per_residue' % divergence_type: per_res_metric(stat)
                                for divergence_type, stat in divergence.items()}
            pose_divergence_s = pd.concat([pd.Series(divergence_stats)], keys=[('sequence_design', 'pose')])

            if designs_by_protocol:  # were multiple designs generated with each protocol?
                # find the divergence within each protocol
                divergence_by_protocol = {protocol: {} for protocol in designs_by_protocol}
                for protocol, designs in designs_by_protocol.items():
                    # Todo select from pose_alignment the indices of each design then pass to MultipleSequenceAlignment?
                    protocol_alignment = msa_from_dictionary({design: pose_sequences[design] for design in designs})
                    protocol_mutation_freq = filter_dictionary_keys(protocol_alignment.frequencies,
                                                                    self.design_residues)
                    protocol_res_dict = {'divergence_%s' % profile: position_specific_jsd(protocol_mutation_freq, bkgnd)
                                         for profile, bkgnd in profile_background.items()}  # ^ both are 1-idx
                    if interface_bkgd:
                        protocol_res_dict['divergence_interface'] = \
                            jensen_shannon_divergence(protocol_mutation_freq, interface_bkgd)
                    # Get per residue divergence metric by protocol
                    for divergence, sequence_info in protocol_res_dict.items():
                        divergence_by_protocol[protocol]['%s_per_residue' % divergence] = per_res_metric(sequence_info)
                        # stats_by_protocol[protocol]['%s_per_residue' % key] = per_res_metric(sequence_info)
                        # {protocol: 'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}

                protocol_divergence_s = pd.concat(
                    [pd.Series(divergence) for divergence in divergence_by_protocol.values()],
                    keys=list(zip(repeat('sequence_design'), divergence_by_protocol)))
            else:
                protocol_divergence_s = pd.Series()
            divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])

        # reference_mutations = cleaned_mutations.pop('reference', None)  # save the reference
        scores_df['number_of_mutations'] = \
            pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
        scores_df['percent_mutations'] = \
            scores_df['number_of_mutations'] / other_pose_metrics['entity_residue_length_total']
        # residue_indices_per_entity = self.pose.residue_indices_per_entity
        is_thermophilic = []
        for idx, (entity, entity_indices) in enumerate(zip(self.pose.entities, self.pose.residue_indices_per_entity), 1):
            # entity_indices = residue_indices_per_entity[idx]
            scores_df['entity_%d_number_of_mutations' % idx] = \
                pd.Series({design: len([residue_idx for residue_idx in mutations if residue_idx in entity_indices])
                           for design, mutations in all_mutations.items()})
            scores_df['entity_%d_percent_mutations' % idx] = \
                scores_df['entity_%d_number_of_mutations' % idx] / \
                other_pose_metrics['entity_%d_number_of_residues' % idx]
            is_thermophilic.append(getattr(other_pose_metrics, 'entity_%d_thermophile' % idx, 0))

        other_pose_metrics['entity_thermophilicity'] = sum(is_thermophilic) / idx  # get the average

        # Process mutational frequencies, H-bond, and Residue energy metrics to dataframe
        residue_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
        # returns multi-index column with residue number as first (top) column index, metric as second index
        # during residue_df unstack, all residues with missing dicts are copied as nan
        # Merge interface design specific residue metrics with total per residue metrics
        # residue_df = pd.merge(residue_df, per_residue_df.loc[:, idx_slice[residue_df.columns.levels[0], :]],
        #                       left_index=True, right_index=True)
        residue_df = pd.merge(residue_df.loc[:, idx_slice[self.design_residues, :]],
                              per_residue_df.loc[:, idx_slice[self.design_residues, :]],
                              left_index=True, right_index=True)

        # entity_alignment = multi_chain_alignment(entity_sequences)
        entity_alignments = {entity: msa_from_dictionary(design_sequences)
                             for entity, design_sequences in entity_sequences.items()}
        # pose_collapse_ = pd.concat(pd.DataFrame(folding_and_collapse), axis=1, keys=[('sequence_design', 'pose')])
        dca_design_residues_concat = []
        dca_succeed = True
        # dca_background_energies, dca_design_energies = [], []
        dca_background_energies, dca_design_energies = {}, {}
        for entity in self.pose.entities:
            try:
                # TODO add these to the analysis
                dca_background_residue_energies = entity.direct_coupling_analysis()
                dca_design_residue_energies = entity.direct_coupling_analysis(msa=entity_alignments[entity])
                dca_design_residues_concat.append(dca_design_residue_energies)
                # dca_background_energies.append(dca_background_energies.sum(axis=1))
                # dca_design_energies.append(dca_design_energies.sum(axis=1))
                dca_background_energies[entity] = dca_background_residue_energies.sum(axis=1)  # turns data to 1D
                dca_design_energies[entity] = dca_design_residue_energies.sum(axis=1)
            except AttributeError:
                self.log.warning('For %s, DCA analysis couldn\'t be performed. Missing required parameter files'
                                 % entity.name)
                dca_succeed = False

        if dca_succeed:
            # concatenate along columns, adding residue index to column, design name to row
            dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                               index=list(entity_sequences[entity].keys()), columns=residue_indices)
            dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy']).swaplevel(0, 1, axis=1)
            # merge with per_residue_df
            per_residue_df = pd.merge(per_residue_df, dca_concatenated_df, left_index=True, right_index=True)

        # residue_df = pd.merge(residue_df, per_residue_df.loc[:, idx_slice[residue_df.columns.levels[0], :]],
        #                       left_index=True, right_index=True)
        # Add local_density information to scores_df
        # scores_df['interface_local_density'] = \
        #     residue_df.loc[not_wt_indices, idx_slice[self.interface_residues, 'local_density']].mean(axis=1)

        # Make buried surface area (bsa) columns
        residue_df = residue_df.join(residue_df.loc[:, idx_slice[self.design_residues, 'sasa_hydrophobic_bound']]
                                     .rename(columns={'sasa_hydrophobic_bound': 'bsa_hydrophobic'}) -
                                     residue_df.loc[:, idx_slice[self.design_residues, 'sasa_hydrophobic_complex']]
                                     .rename(columns={'sasa_hydrophobic_complex': 'bsa_hydrophobic'}))
        residue_df = residue_df.join(residue_df.loc[:, idx_slice[self.design_residues, 'sasa_polar_bound']]
                                     .rename(columns={'sasa_polar_bound': 'bsa_polar'}) -
                                     residue_df.loc[:, idx_slice[self.design_residues, 'sasa_polar_complex']]
                                     .rename(columns={'sasa_polar_complex': 'bsa_polar'}))
        residue_df = residue_df.join(residue_df.loc[:, idx_slice[self.design_residues, 'bsa_hydrophobic']]
                                     .rename(columns={'bsa_hydrophobic': 'bsa_total'}) +
                                     residue_df.loc[:, idx_slice[self.design_residues, 'bsa_polar']]
                                     .rename(columns={'bsa_polar': 'bsa_total'}))
        scores_df['interface_area_polar'] = \
            residue_df.loc[not_wt_indices, idx_slice[self.design_residues, 'bsa_polar']].sum(axis=1)
        scores_df['interface_area_hydrophobic'] = \
            residue_df.loc[not_wt_indices, idx_slice[self.design_residues, 'bsa_hydrophobic']].sum(axis=1)
        # scores_df['interface_area_total'] = \
        #     residue_df.loc[not_wt_indices, idx_slice[self.design_residues, 'bsa_total']].sum(axis=1)
        scores_df['interface_area_total'] = scores_df['interface_area_polar'] + scores_df['interface_area_hydrophobic']
        # make sasa_complex_total columns
        residue_df = residue_df.join(residue_df.loc[:, idx_slice[self.design_residues, 'sasa_hydrophobic_bound']]
                                     .rename(columns={'sasa_hydrophobic_bound': 'sasa_total_bound'}) +
                                     residue_df.loc[:, idx_slice[self.design_residues, 'sasa_polar_bound']]
                                     .rename(columns={'sasa_polar_bound': 'sasa_total_bound'}))
        residue_df = residue_df.join(residue_df.loc[:, idx_slice[self.design_residues, 'sasa_hydrophobic_complex']]
                                     .rename(columns={'sasa_hydrophobic_complex': 'sasa_total_complex'}) +
                                     residue_df.loc[:, idx_slice[self.design_residues, 'sasa_polar_complex']]
                                     .rename(columns={'sasa_polar_complex': 'sasa_total_complex'}))
        # find the proportion of the residue surface area that is solvent accessible versus buried in the interface
        sasa_assembly_df = \
            residue_df.loc[not_wt_indices, idx_slice[self.design_residues, 'sasa_total_complex']].droplevel(-1, axis=1)
        bsa_assembly_df = \
            residue_df.loc[not_wt_indices, idx_slice[self.design_residues, 'bsa_total']].droplevel(-1, axis=1)
        total_surface_area_df = sasa_assembly_df + bsa_assembly_df
        # ratio_df = bsa_assembly_df / total_surface_area_df
        scores_df['interface_area_to_residue_surface_ratio'] = (bsa_assembly_df / total_surface_area_df).mean(axis=1)

        # find the relative sasa of the complex and the unbound fraction
        buried_interface_residues = np.asarray(residue_df.loc[:, idx_slice[self.design_residues, 'bsa_total']] > 0)
        core_or_interior = residue_df.loc[:, idx_slice[self.design_residues, 'sasa_relative_complex']] < 0.25
        support_not_rim_or_core = residue_df.loc[:, idx_slice[self.design_residues, 'sasa_relative_bound']] < 0.25
        core_residues = np.logical_and(core_or_interior, buried_interface_residues).rename(
            columns={'sasa_relative_complex': 'core'})  # .replace({False: 0, True: 1}) maybe... shouldn't be necessary
        interior_residues = np.logical_and(core_or_interior, ~buried_interface_residues).rename(
            columns={'sasa_relative_complex': 'interior'})
        support_residues = np.logical_and(support_not_rim_or_core, buried_interface_residues).rename(
            columns={'sasa_relative_bound': 'support'})
        rim_or_surface = np.logical_xor(~support_not_rim_or_core, np.asarray(core_residues))
        rim_residues = np.logical_and(rim_or_surface, buried_interface_residues).rename(
            columns={'sasa_relative_bound': 'rim'})
        surface_residues = np.logical_xor(rim_or_surface, buried_interface_residues).rename(
            columns={'sasa_relative_bound': 'surface'})

        residue_df = pd.concat([residue_df, core_residues, interior_residues, support_residues, rim_residues,
                                surface_residues], axis=1)
        # Check if any columns are > 50% interior (value can be 0 or 1). If so, return True for that column
        # interior_residue_df = residue_df.loc[not_wt_indices, idx_slice[:, 'interior']]
        interior_residue_numbers = \
            interior_residues[interior_residues.mean(axis=1) > 0.5].columns.remove_unused_levels().levels[0].to_list()
        if interior_residue_numbers:
            self.log.info('Design Residues %s are located in the interior' % ', '.join(map(str, interior_residue_numbers)))

        # This shouldn't be much different from the state variable self.interface_residues
        # perhaps the use of residue neighbor energy metrics adds residues which contribute, but not directly
        # interface_residues = set(residue_df.columns.levels[0].unique()).difference(interior_residue_numbers)

        # Add design residue information to scores_df such as how many core, rim, and support residues were measured
        for residue_class in residue_classificiation:
            scores_df[residue_class] = residue_df.loc[not_wt_indices, idx_slice[:, residue_class]].sum(axis=1)

        scores_columns = scores_df.columns.to_list()
        self.log.debug('Score columns present: %s' % scores_columns)
        # Calculate new metrics from combinations of other metrics
        # sum columns using list[0] + list[1] + list[n]
        summation_pairs = \
            {'buns_unbound': list(filter(re.compile('buns_[0-9]+_unbound$').match, scores_columns)),  # Rosetta
             'interface_energy_bound':
                 list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             'interface_energy_unbound':
                 list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             'solvation_energy_bound':
                 list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),  # Rosetta
             'solvation_energy_unbound':
                 list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),  # Rosetta
             'interface_connectivity':
                 list(filter(re.compile('interface_connectivity_[0-9]+').match, scores_columns)),  # Rosetta
             }
        # 'sasa_hydrophobic_bound':
        #     list(filter(re.compile('sasa_hydrophobic_[0-9]+_bound').match, scores_columns)),
        # 'sasa_polar_bound': list(filter(re.compile('sasa_polar_[0-9]+_bound').match, scores_columns)),
        # 'sasa_total_bound': list(filter(re.compile('sasa_total_[0-9]+_bound').match, scores_columns))}
        scores_df = columns_to_new_column(scores_df, summation_pairs)
        scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
        # add total_interface_residues for div_pairs and int_comp_similarity
        scores_df['total_interface_residues'] = other_pose_metrics['total_interface_residues']
        scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
        scores_df['interface_composition_similarity'] = scores_df.apply(interface_composition_similarity, axis=1)
        # dropping 'total_interface_residues' after calculation as it is in other_pose_metrics
        scores_df.drop(clean_up_intermediate_columns + ['total_interface_residues'], axis=1, inplace=True,
                       errors='ignore')
        if scores_df.get('repacking') is not None:
            # set interface_bound_activation_energy = NaN where repacking is 0
            # Currently is -1 for True (Rosetta Filter quirk...)
            scores_df.loc[scores_df[scores_df['repacking'] == 0].index, 'interface_bound_activation_energy'] = np.nan
            scores_df.drop('repacking', axis=1, inplace=True)
        # Process dataframes for missing values and drop refine trajectory if present
        scores_df[PUtils.groups] = protocol_s
        # refine_index = scores_df[scores_df[PUtils.groups] == PUtils.refine].index
        # scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # residue_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
        # residue_info.pop(PUtils.refine, None)  # Remove refine from analysis
        # residues_no_frags = residue_df.columns[residue_df.isna().all(axis=0)].remove_unused_levels().levels[0]
        residue_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns such as obs_interface
        residue_df.fillna(0., inplace=True)
        # residue_indices_no_frags = residue_df.columns[residue_df.isna().all(axis=0)]

        # POSE ANALYSIS
        # refine is not considered sequence design and destroys mean. remove v
        # trajectory_df = scores_df.sort_index().drop(PUtils.refine, axis=0, errors='ignore')
        # consensus cst_weights are very large and destroy the mean.
        # remove this step for consensus or refine if they are run multiple times
        trajectory_df = scores_df.sort_index().drop([PUtils.refine, PUtils.stage[5]], axis=0, errors='ignore')
        # add all docking and pose information to each trajectory
        pose_metrics_df = pd.concat([pd.Series(other_pose_metrics)] * len(trajectory_df), axis=1).T
        pose_metrics_df.rename(index=dict(zip(range(len(trajectory_df)), trajectory_df.index)), inplace=True)
        trajectory_df = pd.concat([pose_metrics_df, trajectory_df, pose_collapse_df], axis=1)

        # assert len(trajectory_df.index.to_list()) > 0, 'No designs left to analyze in this pose!'

        # Get total design statistics for every sequence in the pose and every protocol specifically
        protocol_groups = scores_df.groupby(PUtils.groups)
        # # protocol_groups = trajectory_df.groupby(groups)
        # designs_by_protocol = {protocol: scores_df.index[indices].values.tolist()  # <- df must be from same source
        #                        for protocol, indices in protocol_groups.indices.items()}
        # designs_by_protocol.pop(PUtils.refine, None)  # remove refine if present
        # designs_by_protocol.pop(PUtils.stage[5], None)  # remove consensus if present
        # # designs_by_protocol = {protocol: trajectory_df.index[indices].values.tolist()
        # #                        for protocol, indices in protocol_groups.indices.items()}

        pose_stats, protocol_stats = [], []
        for idx, stat in enumerate(stats_metrics):
            pose_stats.append(getattr(trajectory_df, stat)().rename(stat))
            protocol_stats.append(getattr(protocol_groups, stat)())

        protocol_stats[stats_metrics.index('mean')]['observations'] = protocol_groups.size()
        protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats], keys=stats_metrics)
        pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
        stat_s = pd.concat([protocol_stats_s.dropna(), pose_stats_s.dropna()])  # dropna removes NaN metrics
        # change statistic names for all df that are not groupby means for the final trajectory dataframe
        for idx, stat in enumerate(stats_metrics):
            if stat != 'mean':
                protocol_stats[idx].index = protocol_stats[idx].index.to_series().map(
                    {protocol: '%s_%s' % (protocol, stat) for protocol in unique_design_protocols})
        trajectory_df = pd.concat([trajectory_df, pd.concat(pose_stats, axis=1).T] + protocol_stats)
        # this concat puts back refine and consensus designs since protocol_stats is calculated on scores_df

        # Calculate protocol significance
        pvalue_df = pd.DataFrame()
        scout_protocols = list(filter(re.compile('.*scout').match, unique_protocols))
        similarity_protocols = unique_design_protocols.difference([PUtils.refine] + scout_protocols)
        if background_protocol not in unique_design_protocols:
            self.log.warning('Missing background protocol \'%s\'. No protocol significance measurements available '
                             'for this pose' % background_protocol)
        elif len(similarity_protocols) == 1:  # measure significance
            self.log.info('Can\'t measure protocol significance, only one protocol of interest')
        # missing_protocols = protocols_of_interest.difference(unique_design_protocols)
        # if missing_protocols:
        #     self.log.warning('Missing protocol%s \'%s\'. No protocol significance measurements for this design!'
        #                      % ('s' if len(missing_protocols) > 1 else '', ', '.join(missing_protocols)))
        # elif len(protocols_of_interest) == 1:
        else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
            # for prot1, prot2 in combinations(sorted(protocols_of_interest), 2):
            for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
                select_df = \
                    trajectory_df.loc[designs_by_protocol[prot1] + designs_by_protocol[prot2], significance_columns]
                difference_s = trajectory_df.loc[prot1, :].sub(trajectory_df.loc[prot2, :])  # prot1/2 pull out mean
                pvalue_df[(prot1, prot2)] = df_permutation_test(select_df, difference_s, compare='mean',
                                                                group1_size=len(designs_by_protocol[prot1]))
            pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
            trajectory_df = pd.concat([trajectory_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

            # Compute residue energy/sequence differences between each protocol
            residue_energy_df = residue_df.loc[not_wt_indices, idx_slice[:, 'energy_delta']]

            scaler = StandardScaler()
            res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
            residue_energy_np = scaler.fit_transform(residue_energy_df.values)
            residue_energy_pc = res_pca.fit_transform(residue_energy_np)

            seq_pca = PCA(PUtils.variance)
            designed_residue_info = {design: {residue: info for residue, info in residues_info.items()
                                              if residue in self.design_residues}
                                     for design, residues_info in residue_info.items()}
            pairwise_sequence_diff_np = scaler.fit_transform(all_vs_all(designed_residue_info, sequence_difference))
            seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
            # Make principal components (PC) DataFrame
            residue_energy_pc_df = \
                pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                             columns=['pc%d' % idx for idx, _ in enumerate(res_pca.components_, 1)])
            seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_info.keys()),
                                     columns=['pc%d' % idx for idx, _ in enumerate(seq_pca.components_, 1)])
            # Compute the euclidean distance
            # pairwise_pca_distance_np = pdist(seq_pc)
            # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

            # Merge PC DataFrames with labels
            seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)
            residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)
            # Next group the labels
            sequence_groups = seq_pc_df.groupby(PUtils.groups)
            residue_energy_groups = residue_energy_pc_df.groupby(PUtils.groups)
            # Measure statistics for each group
            # All protocol means have pairwise distance measured to access similarity
            # Gather protocol similarity/distance metrics
            sim_measures = {'sequence_distance': {}, 'energy_distance': {}}
            sim_stdev = {}  # 'similarity': None, 'seq_distance': None, 'energy_distance': None}
            # grouped_pc_seq_df_dict, grouped_pc_energy_df_dict, similarity_stat_dict = {}, {}, {}
            for stat in stats_metrics:
                grouped_pc_seq_df = getattr(sequence_groups, stat)()
                grouped_pc_energy_df = getattr(residue_energy_groups, stat)()
                similarity_stat = getattr(pvalue_df, stat)(axis=1)  # protocol pair : stat Series
                if stat == 'mean':
                    # for each measurement in residue_energy_pc_df, need to take the distance between it and the
                    # structure background mean (if structure background, is the mean is useful too?)
                    background_distance = cdist(residue_energy_pc,
                                                grouped_pc_energy_df.loc[background_protocol, :].values[np.newaxis, :])
                    trajectory_df = \
                        pd.concat([trajectory_df,
                                   pd.Series(background_distance.flatten(), index=residue_energy_pc_df.index,
                                             name='energy_distance_from_%s_mean' % background_protocol)], axis=1)

                    # if renaming is necessary
                    # protocol_stats_s[stat].index = protocol_stats_s[stat].index.to_series().map(
                    #     {protocol: protocol + '_' + stat for protocol in sorted(unique_design_protocols)})
                    # find the pairwise distance from every point to every other point
                    seq_pca_mean_distance_vector = pdist(grouped_pc_seq_df)
                    energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df)
                    # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
                    #                             for k in seq_pca_mean_distance_vector)
                    # find similarity between each protocol by taking row average of all p-values for each metric
                    # mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance Series
                    # mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
                    # sim_measures['similarity'] = mean_pvalue_s
                    similarity_stat.index = pd.MultiIndex.from_tuples(similarity_stat.index)
                    sim_measures['similarity'] = similarity_stat

                    # for vector_idx, seq_dist in enumerate(seq_pca_mean_distance_vector):
                    #     i, j = condensed_to_square(vector_idx, len(grouped_pc_seq_df.index))
                    #     sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                    #                                        grouped_pc_seq_df.index[j])] = seq_dist

                    for vector_idx, (seq_dist, energy_dist) in enumerate(zip(seq_pca_mean_distance_vector,
                                                                             energy_pca_mean_distance_vector)):
                        i, j = condensed_to_square(vector_idx, len(grouped_pc_energy_df.index))
                        sim_measures['sequence_distance'][(grouped_pc_seq_df.index[i],
                                                           grouped_pc_seq_df.index[j])] = seq_dist
                        sim_measures['energy_distance'][(grouped_pc_energy_df.index[i],
                                                         grouped_pc_energy_df.index[j])] = energy_dist
                elif stat == 'std':
                    # sim_stdev['similarity'] = similarity_stat_dict[stat]
                    # Todo need to square each pc, add them up, divide by the group number, then take the sqrt
                    sim_stdev['seq_distance'] = grouped_pc_seq_df
                    sim_stdev['energy_distance'] = grouped_pc_energy_df

            # Find the significance between each pair of protocols
            protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                                       keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
            # squeeze turns the column headers into series indices. Keys appends to make a multi-index

            # Find total protocol similarity for different metrics
            # for measure, values in sim_measures.items():
            #     # measure_s = pd.Series({pair: similarity for pair, similarity in values.items()})
            #     # measure_s = pd.Series(values)
            #     similarity_sum['protocol_%s_sum' % measure] = pd.Series(values).sum()
            similarity_sum = {'protocol_%s_sum' % measure: pd.Series(values).sum()
                              for measure, values in sim_measures.items()}
            similarity_sum_s = pd.concat([pd.Series(similarity_sum)], keys=[('sequence_design', 'pose')])

            # Process similarity between protocols
            sim_measures_s = pd.concat([pd.Series(values) for values in sim_measures.values()],
                                       keys=list(sim_measures.keys()))
            # # Todo test
            # sim_stdev_s = pd.concat(list(sim_stdev.values()),
            #                         keys=list(zip(repeat('std'), sim_stdev.keys()))).swaplevel(1, 2)
            # sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s, sim_stdev_s]
            sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s]

            # if figures:  # Todo ensure output is as expected then move below
                # protocols_by_design = {design: protocol for protocol, designs in designs_by_protocol.items()
                #                        for design in designs}
                # _path = os.path.join(self.all_scores, str(self))
                # # Set up Labels & Plot the PC data
                # protocol_map = {protocol: i for i, protocol in enumerate(designs_by_protocol)}
                # integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
                # pc_labels_group = [protocols_by_design[design] for design in residue_info]
                # # pc_labels_group = np.array([protocols_by_design[design] for design in residue_info])
                # pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_info]
                # fig = plt.figure()
                # # ax = fig.add_subplot(111, projection='3d')
                # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
                # # plt.cla()
                #
                # # for color_int, label in integer_map.items():  # zip(pc_labels_group, pc_labels_int):
                # #     ax.scatter(seq_pc[pc_labels_group == label, 0],
                # #                seq_pc[pc_labels_group == label, 1],
                # #                seq_pc[pc_labels_group == label, 2],
                # #                c=color_int, cmap=plt.cm.nipy_spectral, edgecolor='k')
                # scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral',
                #                      edgecolor='k')
                # # handles, labels = scatter.legend_elements()
                # # # print(labels)  # ['$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$']
                # # ax.legend(handles, labels, loc='upper right', title=groups)
                # # # ax.legend(handles, [integer_map[label] for label in labels], loc="upper right", title=groups)
                # # # plt.axis('equal') # not possible with 3D graphs
                # # plt.legend()  # No handles with labels found to put in legend.
                # colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
                # custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20)
                #                 for c in colors]
                # ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
                #           bbox_to_anchor=(1.0, .5))
                # # # Add group mean to the plot
                # # for name, label in integer_map.items():
                # #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(),
                # #                seq_pc[pc_labels_group == label, 1].mean(),
                # #                seq_pc[pc_labels_group == label, 2].mean(), marker='x')
                # ax.set_xlabel('PC1')
                # ax.set_ylabel('PC2')
                # ax.set_zlabel('PC3')
                # # plt.legend(pc_labels_group)
                # plt.savefig('%s_seq_pca.png' % _path)
                # plt.clf()
                # # Residue PCA Figure to assay multiple interface states
                # fig = plt.figure()
                # # ax = fig.add_subplot(111, projection='3d')
                # ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
                # scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2],
                #                      c=pc_labels_int,
                #                      cmap='Spectral', edgecolor='k')
                # colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
                # custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in
                #                 colors]
                # ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
                #           bbox_to_anchor=(1.0, .5))
                # ax.set_xlabel('PC1')
                # ax.set_ylabel('PC2')
                # ax.set_zlabel('PC3')
                # plt.savefig('%s_res_energy_pca.png' % _path)

        # Format output and save Trajectory, Residue DataFrames, and PDB Sequences
        if save_trajectories:
            trajectory_df.sort_index(inplace=True, axis=1)
            residue_df.sort_index(inplace=True)
            # # Add wild-type residue information in metrics for sequence comparison
            # # find the solvent accessible surface area of the separated entities
            # for entity in self.pose.entities:
            #     entity.get_sasa()

            # errat_collapse_df = \
            #     pd.concat([pd.concat(
            #         {'errat_deviation': pd.Series(np.concatenate(list(wt_errat.values())), index=residue_indices),
            #          'hydrophobic_collapse': pd.Series(np.concatenate(list(wt_collapse.values())),
            #                                            index=residue_indices)})],
            #         keys=['wild_type']).unstack().unstack()  # .swaplevel(0, 1, axis=1)

            # wild_type_residue_info = {}
            # for res_number in residue_info[next(iter(residue_info))].keys():
            #     # bsa_total is actually a sasa, but for formatting sake, I've called it a bsa...
            #     residue = self.pose.pdb.residue(res_number)
            #     wild_type_residue_info[res_number] = \
            #         {'type': protein_letters_3to1.get(residue.type.title()), 'core': None, 'rim': None, 'support': None,
            #          'interior': 0, 'hbond': None, 'energy_delta': None,
            #          'bsa_total': None, 'bsa_polar': None, 'bsa_hydrophobic': None, 'sasa_total': residue.sasa,
            #          'coordinate_constraint': None, 'residue_favored': None, 'observed_design': None,
            #          'observed_evolution': None, 'observed_fragment': None}  # 'hot_spot': None}
            #     if residue.relative_sasa < 0.25:
            #         wild_type_residue_info[res_number]['interior'] = 1
            #     # if res_number in issm_residues and res_number not in residues_no_frags:
            #     #     wild_type_residue_info[res_number]['observed_fragment'] = None

            # wt_df = pd.concat([pd.DataFrame(wild_type_residue_info)], keys=['wild_type']).unstack()
            # wt_df = pd.merge(wt_df, errat_collapse_df.loc[:, idx_slice[wt_df.columns.levels[0], :]],
            #                  left_index=True, right_index=True)
            # wt_df.drop(residue_indices_no_frags, inplace=True, axis=1, errors='ignore')
            # # only sort once as residues are in same order
            # # wt_df.sort_index(level=0, inplace=True, axis=1, sort_remaining=False)
            # # residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
            # residue_df = pd.concat([wt_df, residue_df], sort=False)
            # # residue_df.drop(residue_indices_no_frags, inplace=True, axis=1)
            residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
            residue_df[(PUtils.groups, PUtils.groups)] = protocol_s
            # residue_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
            if merge_residue_data:
                trajectory_df = pd.concat([trajectory_df], axis=1, keys=['metrics'])
                trajectory_df = pd.merge(trajectory_df, residue_df, left_index=True, right_index=True)
            else:
                residue_df.to_csv(self.residues)
            trajectory_df.to_csv(self.trajectories)
            pickle_object(entity_sequences, self.design_sequences, out_path='')

        # Create figures
        if figures:  # for plotting collapse profile, errat data, contact order
            # Plot: Format the collapse data with residues as index and each design as column
            # collapse_graph_df = pd.DataFrame(per_residue_data['hydrophobic_collapse'])
            collapse_graph_df = per_residue_df.loc[:, idx_slice[:, 'hydrophobic_collapse']].droplevel(-1, axis=1)
            # wt_collapse_concatenated_s = pd.Series(np.concatenate(list(wt_collapse.values())), name='clean_asu')
            # collapse_graph_df['clean_asu'] = wt_collapse_concatenated_s
            collapse_graph_df.index += 1  # offset index to residue numbering
            # collapse_graph_df.sort_index(axis=1, inplace=True)
            # graph_collapse = sns.lineplot(data=collapse_graph_df)
            # g = sns.FacetGrid(tip_sumstats, col="sex", row="smoker")
            # graph_collapse = sns.relplot(data=collapse_graph_df, kind='line')  # x='Residue Number'

            # Set the base figure aspect ration for all sequence designs
            figure_aspect_ratio = (pose_length / 25., 20)  # 20 is arbitrary size to fit all information in figure
            color_cycler = cycler(color=large_color_array)
            plt.rc('axes', prop_cycle=color_cycler)
            fig = plt.figure(figsize=figure_aspect_ratio)
            # legend_fill_value = int(15 * pose_length / 100)

            # collapse_ax, contact_ax, errat_ax = fig.subplots(3, 1, sharex=True)
            collapse_ax, errat_ax = fig.subplots(2, 1, sharex=True)
            # add the contact order to a new plot
            wt_contact_order_concatenated_s = \
                pd.Series(np.concatenate(list(contact_order.values())), index=residue_indices, name='contact_order')
            contact_ax = collapse_ax.twinx()
            contact_ax.plot(wt_contact_order_concatenated_s, label='Contact Order',
                            color='#fbc0cb', lw=1, linestyle='-')  # pink
            # contact_ax.scatter(residue_indices, wt_contact_order_concatenated_s, color='#fbc0cb', marker='o')  # pink
            # wt_contact_order_concatenated_min_s = wt_contact_order_concatenated_s.min()
            # wt_contact_order_concatenated_max_s = wt_contact_order_concatenated_s.max()
            # wt_contact_order_range = wt_contact_order_concatenated_max_s - wt_contact_order_concatenated_min_s
            # scaled_contact_order = ((wt_contact_order_concatenated_s - wt_contact_order_concatenated_min_s)
            #                         / wt_contact_order_range)  # / wt_contact_order_range)
            # graph_contact_order = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # collapse_ax1.plot(scaled_contact_order)
            # contact_ax.vlines(self.pose.chain_breaks, 0, 1, transform=contact_ax.get_xaxis_transform(),
            #                   label='Entity Breaks', colors='#cccccc')  # , grey)
            # contact_ax.vlines(design_residues_l, 0, 0.05, transform=contact_ax.get_xaxis_transform(),
            #                   label='Design Residues', colors='#f89938', lw=2)  # , orange)
            contact_ax.set_ylabel('Contact Order')
            # contact_ax.set_xlim(0, pose_length)
            contact_ax.set_ylim(0, None)
            # contact_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))
            # collapse_ax1.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))

            # Get the plot of each collapse profile into a matplotlib axes
            # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax, figsize=figure_aspect_ratio)
            # collapse_ax = collapse_graph_df.plot.line(legend=False, ax=collapse_ax)
            collapse_ax.plot(collapse_graph_df.values, label=collapse_graph_df.columns)
            # collapse_ax = collapse_graph_df.plot.line(ax=collapse_ax)
            collapse_ax.xaxis.set_major_locator(MultipleLocator(20))
            collapse_ax.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            collapse_ax.xaxis.set_minor_locator(MultipleLocator(5))
            collapse_ax.set_xlim(0, pose_length)
            collapse_ax.set_ylim(0, 1)
            # # CAN'T SET FacetGrid object for most matplotlib elements...
            # ax = graph_collapse.axes
            # ax = plt.gca()  # gca <- get current axis
            # labels = [fill(column, legend_fill_value) for column in collapse_graph_df.columns]
            # collapse_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1))
            # collapse_ax.legend(loc='lower left', bbox_to_anchor=(0., 1))
            # Plot the chain break(s) and design residues
            # linestyles={'solid', 'dashed', 'dashdot', 'dotted'}
            collapse_ax.vlines(self.pose.chain_breaks, 0, 1, transform=collapse_ax.get_xaxis_transform(),
                               label='Entity Breaks', colors='#cccccc')  # , grey)
            design_residues_l = list(self.design_residues)
            collapse_ax.vlines(design_residues_l, 0, 0.05, transform=collapse_ax.get_xaxis_transform(),
                               label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            collapse_ax.hlines([collapse_significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                               label='Collapse Threshold', colors='#fc554f', linestyle='dotted')  # tomato
            # collapse_ax.set_xlabel('Residue Number')
            collapse_ax.set_ylabel('Hydrophobic Collapse Index')
            # collapse_ax.set_prop_cycle(color_cycler)
            # ax.autoscale(True)
            # collapse_ax.figure.tight_layout()  # no standardization
            # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse.png'))  # no standardization

            # Plot: Collapse description of total profile against each design
            if msa_metrics:
                profile_mean_collapse_concatenated_s = \
                    pd.concat([collapse_df[entity].loc['mean', :] for entity in self.pose.entities], ignore_index=True)
                profile_std_collapse_concatenated_s = \
                    pd.concat([collapse_df[entity].loc['std', :] for entity in self.pose.entities], ignore_index=True)
                profile_mean_collapse_concatenated_s.index += 1  # offset index to residue numbering
                profile_std_collapse_concatenated_s.index += 1  # offset index to residue numbering
                collapse_graph_describe_df = pd.DataFrame({
                    'std_min': profile_mean_collapse_concatenated_s - profile_std_collapse_concatenated_s,
                    'std_max': profile_mean_collapse_concatenated_s + profile_std_collapse_concatenated_s,
                })
                collapse_graph_describe_df.index += 1  # offset index to residue numbering
                collapse_graph_describe_df['Residue Number'] = collapse_graph_describe_df.index
                collapse_ax.vlines('Residue Number', 'std_min', 'std_max', data=collapse_graph_describe_df,
                                   color='#e6e6fa', linestyle='-', lw=1, alpha=0.8)  # lavender
                # collapse_ax.figure.savefig(os.path.join(self.data, 'hydrophobic_collapse_versus_profile.png'))

            # Plot: Errat Accuracy
            # errat_graph_df = pd.DataFrame(per_residue_data['errat_deviation'])
            # errat_graph_df = per_residue_df.loc[:, idx_slice[:, 'errat_deviation']].droplevel(-1, axis=1)
            errat_graph_df = errat_df
            # wt_errat_concatenated_s = pd.Series(np.concatenate(list(wt_errat.values())), name='clean_asu')
            errat_graph_df['clean_asu'] = wt_errat_concat_s
            errat_graph_df.index += 1  # offset index to residue numbering
            errat_graph_df.sort_index(axis=1, inplace=True)
            # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax, figsize=figure_aspect_ratio)
            # errat_ax = errat_graph_df.plot.line(legend=False, ax=errat_ax)
            # errat_ax = errat_graph_df.plot.line(ax=errat_ax)
            errat_ax.plot(errat_graph_df.values, label=collapse_graph_df.columns)
            errat_ax.xaxis.set_major_locator(MultipleLocator(20))
            errat_ax.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            errat_ax.xaxis.set_minor_locator(MultipleLocator(5))
            # errat_ax.set_xlim(0, pose_length)
            errat_ax.set_ylim(0, None)
            # graph_errat = sns.relplot(data=errat_graph_df, kind='line')  # x='Residue Number'
            # Plot the chain break(s) and design residues
            # labels = [fill(column, legend_fill_value) for column in errat_graph_df.columns]
            # errat_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1.))
            # errat_ax.legend(loc='lower center', bbox_to_anchor=(0., 1.))
            errat_ax.vlines(self.pose.chain_breaks, 0, 1, transform=errat_ax.get_xaxis_transform(),
                            label='Entity Breaks', colors='#cccccc')  # , grey)
            errat_ax.vlines(design_residues_l, 0, 0.05, transform=errat_ax.get_xaxis_transform(),
                            label='Design Residues', colors='#f89938', lw=2)  # , orange)
            # Plot horizontal significance
            errat_ax.hlines([errat_2_sigma], 0, 1, transform=errat_ax.get_yaxis_transform(),
                            label='Significant Error', colors='#fc554f', linestyle='dotted')  # tomato
            errat_ax.set_xlabel('Residue Number')
            errat_ax.set_ylabel('Errat Score')
            # errat_ax.autoscale(True)
            # errat_ax.figure.tight_layout()
            # errat_ax.figure.savefig(os.path.join(self.data, 'errat.png'))
            collapse_handles, collapse_labels = collapse_ax.get_legend_handles_labels()
            contact_handles, contact_labels = contact_ax.get_legend_handles_labels()
            # errat_handles, errat_labels = errat_ax.get_legend_handles_labels()
            # print(handles, labels)
            handles = collapse_handles + contact_handles
            labels = collapse_labels + contact_labels
            # handles = errat_handles + contact_handles
            # labels = errat_labels + contact_labels
            labels = [label.replace('%s_' % self.name, '') for label in labels]
            # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  #, ncol=3, mode='expand')
            # print(labels)
            # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand'
            # v Why the hell doesn't this work??
            # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.), ncol=3,  # , mode='expand')
            # fig.subplots_adjust(bottom=0.1)
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -1.), ncol=3)  # , mode='expand')
            #            bbox_transform=plt.gcf().transFigure)  # , bbox_transform=collapse_ax.transAxes)
            fig.tight_layout()
            fig.savefig(os.path.join(self.data, 'DesignMetricsPerResidues.png'))

        # After parsing data sources
        other_metrics_s = pd.concat([pd.Series(other_pose_metrics)], keys=[('dock', 'pose')])

        # CONSTRUCT: Create pose series and format index names
        pose_s = pd.concat([other_metrics_s, stat_s, divergence_s] + sim_series).swaplevel(0, 1)
        # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
        pose_s.drop([PUtils.groups], level=2, inplace=True, errors='ignore')
        pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
        pose_s.name = str(self)

        return pose_s

    @handle_design_errors(errors=(DesignError, AssertionError))
    @close_logs
    def select_sequences(self, filters=None, weights=None, number=1, protocols=None, **kwargs):
        """Select sequences for further characterization. If weights, then user can prioritize by metrics, otherwise
        sequence with the most neighbors as calculated by sequence distance will be selected. If there is a tie, the
        sequence with the lowest weight will be selected

        Keyword Args:
            filters=None (Iterable): The filters to use in sequence selection
            weights=None (Iterable): The weights to use in sequence selection
            number=1 (int): The number of sequences to consider for each design
            protocols=None (str): Whether particular design protocol(s) should be chosen
        Returns:
            (list[tuple[DesignDirectory, str]]): Containing the selected sequences found
        """
        # Load relevant data from the design directory
        trajectory_df = pd.read_csv(self.trajectories, index_col=0, header=[0])
        trajectory_df.dropna(inplace=True)
        if protocols:
            designs = []
            for protocol in protocols:
                designs.extend(trajectory_df[trajectory_df['protocol'] == protocol].index.to_list())

            if not designs:
                raise DesignError('No designs found for protocols %s!' % protocols)
        else:
            designs = trajectory_df.index.to_list()

        self.log.info('Number of starting trajectories = %d' % len(trajectory_df))
        df = trajectory_df.loc[designs, :]

        if filters:
            self.log.info('Using filter parameters: %s' % str(filters))
            # Filter the DataFrame to include only those values which are le/ge the specified filter
            filtered_designs = index_intersection(filter_df_for_index_by_value(df, filters).values())
            df = trajectory_df.loc[filtered_designs, :]

        if weights:
            # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
            self.log.info('Using weighting parameters: %s' % str(weights))
            design_list = rank_dataframe_by_metric_weights(df, weights=weights, **kwargs).index.to_list()
            self.log.info('Final ranking of trajectories:\n%s' % ', '.join(pose for pose in design_list))

            return list(zip(repeat(self), design_list[:number]))
        else:
            # sequences_pickle = glob(os.path.join(self.all_scores, '%s_Sequences.pkl' % str(self)))
            # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
            #                                    os.path.join(self.all_scores, '%s_Sequences.pkl' % str(self))
            #
            # chain_sequences = SDUtils.unpickle(sequences_pickle[0])
            # {chain: {name: sequence, ...}, ...}
            entity_sequences = unpickle(self.design_sequences)
            concatenated_sequences = [''.join([entity_sequences[entity][design] for entity in entity_sequences])
                                      for design in designs]
            self.log.debug('The final concatenated sequences are:\n%s' % concatenated_sequences)

            # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, sequence_difference)
            # Using concatenated sequences makes the values very similar and inflated as most residues are the same
            # doing min/max normalization to see variation
            pairwise_sequence_diff_l = [sequence_difference(*seq_pair)
                                        for seq_pair in combinations(concatenated_sequences, 2)]
            pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
            _min = min(pairwise_sequence_diff_l)
            # _max = max(pairwise_sequence_diff_l)
            pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
            # self.log.info(pairwise_sequence_diff_l)

            # PCA analysis of distances
            pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
            for k, dist in enumerate(pairwise_sequence_diff_np):
                i, j = condensed_to_square(k, len(designs))
                pairwise_sequence_diff_mat[i, j] = dist
            pairwise_sequence_diff_mat = sym(pairwise_sequence_diff_mat)

            pairwise_sequence_diff_mat = StandardScaler().fit_transform(pairwise_sequence_diff_mat)
            seq_pca = PCA(PUtils.variance)
            seq_pc_np = seq_pca.fit_transform(pairwise_sequence_diff_mat)
            seq_pca_distance_vector = pdist(seq_pc_np)
            # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
            epsilon = seq_pca_distance_vector.mean() * 0.5
            self.log.info('Finding maximum neighbors within distance of %f' % epsilon)

            # self.log.info(pairwise_sequence_diff_np)
            # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
            # epsilon = math.sqrt(seq_pc_np.myean()) * 0.5
            # epsilon = math.sqrt(pairwise_sequence_diff_np.mean()) * 0.5

            # Find the nearest neighbors for the pairwise distance matrix using the X*X^T (PCA) matrix, linear transform
            seq_neighbors = BallTree(seq_pc_np)  # Todo make brute force or automatic, not BallTree
            seq_neighbor_counts = seq_neighbors.query_radius(seq_pc_np, epsilon,
                                                             count_only=True)  # , sort_results=True)
            top_count, top_idx = 0, None
            for count in seq_neighbor_counts:  # idx, enumerate()
                if count > top_count:
                    top_count = count

            sorted_seqs = sorted(seq_neighbor_counts, reverse=True)
            top_neighbor_counts = sorted(set(sorted_seqs[:number]), reverse=True)

            # Find only the designs which match the top x (number) of neighbor counts
            final_designs = {designs[idx]: num_neighbors for num_neighbors in top_neighbor_counts
                             for idx, count in enumerate(seq_neighbor_counts) if count == num_neighbors}
            self.log.info('The final sequence(s) and file(s):\nNeighbors\tDesign\n%s'
                          # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.index_offset,
                          % '\n'.join('\t%d\t%s' % (neighbors, os.path.join(self.designs, design))
                                      for design, neighbors in final_designs.items()))

            # self.log.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(self.designs, seq))
            #                                                         for i, seq in enumerate(final_designs, 1)))

            # Compute the highest density cluster using DBSCAN algorithm
            # seq_cluster = DBSCAN(eps=epsilon)
            # seq_cluster.fit(pairwise_sequence_diff_np)
            #
            # seq_pc_df = pd.DataFrame(seq_pc, index=designs, columns=['pc' + str(x + SDUtils.index_offset)
            #                                                          for x in range(len(seq_pca.components_))])
            # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

            # If final designs contains more sequences than specified, find the one with the lowest energy
            if len(final_designs) > number:
                energy_s = trajectory_df.loc[final_designs.keys(), 'interface_energy']
                energy_s.sort_values(inplace=True)
                final_seqs = zip(repeat(self), energy_s.index.to_list()[:number])
            else:
                final_seqs = zip(repeat(self), final_designs.keys())

            return list(final_seqs)

    @staticmethod
    def make_path(path, condition=True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Keyword Args:
            condition=True (bool): A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(path, exist_ok=True)

    handle_design_errors = staticmethod(handle_design_errors)
    close_logs = staticmethod(close_logs)
    remove_structure_memory = staticmethod(remove_structure_memory)

    def __key(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DesignDirectory):
            return self.__key() == other.__key()
        return NotImplemented

    def __hash__(self):
        return hash(self.__key())

    def __str__(self) -> str:
        if self.nanohedra_output:
            return self.source_path.replace(self.nanohedra_root + os.sep, '').replace(os.sep, '-')
        elif self.output_directory:
            return self.name
        else:
            # TODO integrate with designDB?
            return self.path.replace(self.projects + os.sep, '').replace(os.sep, '-')


def get_sym_entry_from_nanohedra_directory(nanohedra_dir):
    try:
        with open(os.path.join(nanohedra_dir, PUtils.master_log), 'r') as f:
            for line in f.readlines():
                if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                    return SymEntry(int(line.split(':')[-1]))  # sym_map inclusion?
    except FileNotFoundError:
        raise FileNotFoundError('The Nanohedra Output Directory is malformed. Missing required docking file %s'
                                % os.path.join(nanohedra_dir, PUtils.master_log))
    raise DesignError('The Nanohedra Output docking file %s is malformed. Missing required info Nanohedra Entry Number'
                      % os.path.join(nanohedra_dir, PUtils.master_log))


def set_up_directory_objects(design_list, mode=PUtils.interface_design, project=None):
    """Create DesignDirectory objects from a directory iterable. Add program_root if using DesignDirectory strings"""
    return [DesignDirectory.from_nanohedra(design_path, nanohedra_output=True, mode=mode, project=project)
            for design_path in design_list]


def set_up_pseudo_design_dir(path, directory, score):  # changed 9/30/20 to locate paths of interest at .path
    pseudo_dir = DesignDirectory(path, nanohedra_output=False)
    # pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.composition = os.path.dirname(path)
    pseudo_dir.designs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir
