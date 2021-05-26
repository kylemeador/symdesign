import os
import copy
from math import ceil, sqrt
import shutil
import subprocess
from glob import glob
from itertools import combinations, repeat,  # chain as iter_chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from SymDesignUtils import unpickle, start_log, null_log, handle_errors, sdf_lookup, write_shell_script, DesignError, \
    match_score_from_z_value, handle_design_errors, pickle_object, filter_dictionary_keys, \
    all_vs_all, condensed_to_square, space_group_to_sym_entry, digit_translate_table, sym, pretty_format_table
from Query import Flags
from CommandDistributer import reference_average_residue_weight, run_cmds, script_cmd, rosetta_flags
from PDB import PDB
from Pose import Pose
from DesignMetrics import columns_to_rename, read_scores, keys_from_trajectory_number, join_columns, groups, \
    necessary_metrics, columns_to_new_column, delta_pairs, summation_pairs, unnecessary, rosetta_terms, \
    dirty_hbond_processing, dirty_residue_processing, mutation_conserved, per_res_metric, residue_classificiation, \
    interface_residue_composition_similarity, division_pairs, stats_metrics, significance_columns, \
    protocols_of_interest, df_permutation_test, calc_relative_sa, clean_up_intermediate_columns, \
    master_metrics, fragment_metric_template, protocol_specific_columns
from SequenceProfile import parse_pssm, generate_multiple_mutations, get_db_aa_frequencies, simplify_mutation_dict, \
    make_mutations_chain_agnostic, weave_sequence_dict, position_specific_jsd, sequence_difference, \
    jensen_shannon_divergence, multi_chain_alignment  # , format_mutations, generate_sequences
from classes.SymEntry import SymEntry
from interface_analysis.Database import FragmentDatabase
from utils.SymmetryUtils import valid_subunit_number


# Globals
logger = start_log(name=__name__)
idx_offset = 1
design_directory_modes = [PUtils.interface_design, 'dock', 'filter']
cst_value = round(0.2 * reference_average_residue_weight, 2)
relax_flags = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints false',
               '-no_optH false', '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev 0.5',
               '-no_his_his_pairE', '-flip_HNQ', '-nblist_autoupdate true', '-no_nstruct_label true',
               '-relax:bb_move false']


class DesignDirectory:  # Todo move PDB coordinate information to Pose. Only use to handle Pose paths/options

    def __init__(self, design_path, nano=False, construct_pose=False, pose_id=None, dock=False, root=None,
                 **kwargs):  # project=None,
        if pose_id:  # Todo may not be compatible P432
            self.program_root = root
            self.directory_string_to_path(pose_id)
            self.source_path = self.path
        self.source_path = design_path
        self.source = None
        self.name = os.path.splitext(os.path.basename(self.source_path))[0]  # works for all directory and file cases
        self.log = None
        self.debug = False
        self.nano = nano
        self.dock = dock
        self.construct_pose = construct_pose

        self.project_designs = None
        self.database = None
        self.protein_data = None
        # design_symmetry/data (P432/Data)
        self.pdbs = None
        # design_symmetry/data/pdbs (P432/Data/PDBs)
        self.orient_dir = None
        # design_symmetry/data/pdbs/oriented (P432/Data/PDBs/oriented)
        self.orient_asu_dir = None
        # design_symmetry/data/pdbs/oriented (P432/Data/PDBs/oriented_asu)
        self.refine_dir = None
        # design_symmetry/data/pdbs/refined (P432/Data/PDBs/refined)
        self.stride_dir = None
        # design_symmetry/data/pdbs/stride (P432/Data/PDBs/refined)
        self.sequence_info = None
        # design_symmetry/sequence_info (P432/SequenceInfo)
        self.sequences = None
        # design_symmetry/sequence_info/sequences (P432/SequenceInfo/sequences)
        self.profiles = None
        # design_symmetry/sequence_info/profiles (P432/SequenceInfo/profiles)
        self.all_scores = None
        # design_symmetry/all_scores (P432/All_Scores)
        self.trajectories = None
        # design_symmetry/all_scores/str(self)_Trajectories.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Trajectories.csv)
        self.residues = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Residues.csv)
        self.design_sequences = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Sequences.pkl)

        self.composition = None
        # design_symmetry/building_blocks (4ftd_5tch)
        self.scores = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.scores_file = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores
        #  (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores/all_scores.sc)
        self.designs = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/designs (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/designs)
        self.scripts = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scripts (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scripts)
        self.frags = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/matching_fragment_representatives
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.frag_file = None
        self.pose_file = None
        self.data = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/data)
        self.serialized_info = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data/stats.pkl
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.info = {}
        self._info = {}  # internal state info

        self.pose = None  # contains the design's Pose object
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/asu.pdb (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/asu.pdb)
        self.asu = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu.pdb
        self.assembly = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/assembly.pdb
        self.refine_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_refine.pdb
        self.refined_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/designs/clean_asu_for_refine.pdb
        self.consensus = None  # Whether to run consensus or not
        self.consensus_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_consensus.pdb
        self.consensus_design_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/designs/clean_asu_for_consensus.pdb
        self.pdb_list = None

        self.sdf_dir = None
        # path/to/directory/sdf/
        self.sdfs = {}
        self.sym_def_file = None
        self.symmetry_protocol = None
        self.oligomer_names = []
        self.oligomers = []

        # todo integrate these flags with SymEntry and pass to Pose
        # self.sym_entry_number = None
        # self.design_symmetry = None
        # self.design_dimension = None
        self.sym_entry = None
        self.uc_dimensions = None
        self.expand_matrices = None
        self.transform_d = {}  # dict[pdb# (1, 2)] = {'rotation': matrix, 'translation': vector}

        # self.fragment_cluster_residue_d = {}
        self.fragment_observations = None
        self.interface_residue_ids = {}
        # {'interface1': '23A,45A,46A,...' , 'interface2': '234B,236B,239B,...'}
        self.interface_ss_topology = {}
        # {1: 'HHLH', 2: 'HSH'}
        self.interface_ss_fragment_topology = {}
        # {1: 'HHH', 2: 'HH'}

        self.center_residue_numbers = []  # TODO MOVE Metrics
        self.total_residue_numbers = []  # TODO MOVE Metrics
        self.all_residue_score = None  # TODO MOVE Metrics
        self.center_residue_score = None  # TODO MOVE Metrics
        self.high_quality_int_residues_matched = None  # TODO MOVE Metrics
        self.central_residues_with_fragment_overlap = None  # TODO MOVE Metrics
        self.fragment_residues_total = None  # TODO MOVE Metrics
        self.percent_overlapping_fragment = None  # TODO MOVE Metrics
        self.multiple_frag_ratio = None  # TODO MOVE Metrics
        # self.fragment_content_d = None
        self.helical_fragment_content = None  # TODO MOVE Metrics
        self.strand_fragment_content = None  # TODO MOVE Metrics
        self.coil_fragment_content = None  # TODO MOVE Metrics
        self.ave_z = None  # TODO MOVE Metrics
        self.total_interface_residues = None  # TODO MOVE Metrics
        self.total_non_fragment_interface_residues = None  # TODO MOVE Metrics
        self.percent_residues_fragment_total = None  # TODO MOVE Metrics
        self.percent_residues_fragment_center = None  # TODO MOVE Metrics

        # Design flags
        self.number_of_trajectories = None
        self.design_selector = None
        self.evolution = False
        self.design_with_fragments = False
        self.query_fragments = False
        self.write_frags = True
        # self.fragment_file = None
        # self.fragment_type = 'biological_interfaces'  # default for now, can be found in frag_db
        self.euler_lookup = None
        self.frag_db = None
        self.design_db = None
        self.score_db = None
        self.script = True
        self.mpi = False
        self.output_assembly = False
        self.increment_chains = kwargs.get('increment_chains', False)
        self.ignore_clashes = False
        # Analysis flags
        self.analysis = False
        self.skip_logging = False
        self.copy_nanohedra = False
        self.set_flags(**kwargs)  # has to be set before set_up_design_directory
        # if not self.sym_entry:
        #     self.sym_entry = SymEntry(sym_entry)
            # self.design_symmetry = self.sym_entry.get_result_design_sym()
            # self.design_dimension = self.sym_entry.get_design_dim()
        # if not self.nano:
        # check to be sure it's not actually one
        #     if ('DEGEN', 'ROT', 'tx') in self.path:
        #         self.nano = True
        self.cryst_record = None

        if self.nano:
            # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
            if not os.path.exists(self.source_path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.source_path)
            # v used in dock_dir set up
            self.building_block_logs = []
            self.building_block_dirs = []

            self.canonical_pdb1 = None  # canonical pdb orientation
            self.canonical_pdb2 = None
            # self.pdb_dir1_path = None
            # self.pdb_dir2_path = None
            self.rot_step_deg1 = None  # TODO
            self.rot_step_deg2 = None  # TODO
            self.cryst_record = None
            self.pose_id = None
            # self.fragment_cluster_freq_d = {}

            if self.dock:
                # Saves the path of the docking directory as DesignDirectory.path attribute. Try to populate further
                # using typical directory structuring
                # self.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implement?
                self.program_root = self.source_path  # Assuming that output directory (^ or v) of Nanohedra was passed
                # v for design_recap
                # self.program_root = glob(os.path.join(self.path, 'NanohedraEntry*DockedPoses%s'
                #                                                   % str(program_root or '')))
                # self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
                # self.log = [os.path.join(_sym, PUtils.master_log) for _sym in self.program_root]
                # for k, _sym in enumerate(self.program_root):
                # for k, _sym in enumerate(next(os.walk(self.program_root))):
                # self.building_blocks.append(list())
                # self.building_block_logs.append(list())
                # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
                # for bb_dir in next(os.walk(_sym))[1]:
                for bb_dir in next(os.walk(self.program_root))[1]:  # [1] grabs dirs from os.walk, yields only top level
                    if os.path.exists(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils?
                        self.building_block_dirs.append(bb_dir)
                        # self.building_block_dirs[k].append(bb_dir)
                        self.building_block_logs.append(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir))
                        # self.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
            elif self.construct_pose:
                self.info['nanohedra'] = True
                path_components = self.source_path.split(os.sep)
                nanohedra_root = path_components[-5]
                # design_symmetry (P432)
                # self.pose_id = self.source_path[self.source_path.find(path_components[-3]) - 1:]\
                #     .replace(os.sep, '-')
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = os.path.join(self.projects, '%s_%s' % (nanohedra_root, PUtils.design_directory))
                # make the newly required files
                self.make_path(self.program_root)
                self.make_path(self.projects)
                self.make_path(self.project_designs)
                # copy the master log
                if not os.path.exists(os.path.join(self.project_designs, PUtils.master_log)):
                    shutil.copy(os.path.join(nanohedra_root, PUtils.master_log), self.project_designs)

                self.composition = self.source_path[:self.source_path.find(path_components[-3]) - 1]
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                self.oligomer_names = list(map(str.lower, os.path.basename(self.composition).split('_')))
                self.info['oligomer_names'] = self.oligomer_names  # Todo ensure all routes of contruction pickle this!

                self.pose_id = '-'.join(path_components[-4:])  # [-5:-1] because of trailing os.sep
                self.name = self.pose_id
                self.path = os.path.join(self.project_designs, self.name)

                # self.set_up_design_directory()
                # shutil.copy(self.pose_file, self.path)
                # shutil.copy(self.frag_file, self.path)
                # populate the transformation dictionary
                # self.gather_pose_metrics()  # is this necessary yet?
                # self.source = self.asu
                # self.nano_master_log = os.path.join(self.project_designs, PUtils.master_log)
                # ^ /program_root/projects/project_designs/design<- self.path /design.pdb
            # else:  # if self.directory_type in [PUtils.interface_design, 'filter', 'analysis']:
            else:
                # May have issues with the number of open log files
                # if self.directory_type == 'filter':
                #     self.skip_logging = True
                # else:
                #     self.skip_logging = True
                path_components = self.source_path.split(os.sep)
                self.program_root = self.source_path[:self.source_path.find(path_components[-4]) - 1]
                # design_symmetry (P432)
                # self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
                self.composition = os.path.join(self.program_root, path_components[-4])
                self.project_designs = os.path.join(self.composition, path_components[-2])
                self.oligomer_names = list(map(str.lower, os.path.basename(self.composition).split('_')))
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                self.source = os.path.join(self.source_path, PUtils.asu)
                # self.set_up_design_directory()
            # else:
            #     raise DesignError('%s: %s is not an available directory_type. Choose from %s...\n'
            #                       % (DesignDirectory.__name__, self.directory_type, ','.join(design_directory_modes)))

            # if not os.path.exists(self.nano_master_log):
            #     raise DesignError('%s: No %s found for this directory! Cannot perform material design without it.\n'
            #                       'Ensure you have the file \'%s\' located properly before trying this Design!'
            #                       % (self.__str__(), PUtils.master_log, self.nano_master_log))
            # self.gather_docking_metrics()

        else:
            if '.pdb' in self.source_path:  # set up /program_root/projects/project/design
                self.source = self.source_path
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)  # symmetry.rstrip(os.sep)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = os.path.join(self.projects, '%s_%s' % (self.source_path.split(os.sep)[-2],
                                                                              PUtils.design_directory))
                self.path = os.path.join(self.project_designs, self.name)
                # ^ /program_root/projects/project/design<- self.path /design.pdb
                self.make_path(self.program_root)
                self.make_path(self.projects)
                self.make_path(self.project_designs)
                # self.make_path(self.path)

                shutil.copy(self.source_path, self.path)
            else:  # initialize DesignDirectory to recognize existing /program_root/projects/project/design
                self.path = self.source_path
                self.project_designs = os.path.dirname(self.path)
                self.projects = os.path.dirname(self.project_designs)
                self.program_root = os.path.dirname(self.projects)
                # path_components = self.path.split(os.sep)
                # self.program_root = '/%s' % os.path.join(*path_components[:-3])
                # self.projects = '/%s' % os.path.join(*path_components[:-2])
                # self.project_designs = '/%s' % os.path.join(*path_components[:-1])
            # self.set_up_design_directory()
        self.link_master_directory()

    @classmethod
    def from_nanohedra(cls, design_path, project=None, **kwargs):
        return cls(design_path, nano=True, project=project, **kwargs)

    @classmethod
    def from_file(cls, design_path, project=None, **kwargs):
        return cls(design_path, project=project, **kwargs)

    @classmethod
    def from_pose_id(cls, pose_id=None, root=None, **kwargs):
        return cls(None, pose_id=pose_id, root=root, **kwargs)

    @property
    def design_symmetry(self):
        try:
            return self.sym_entry.get_result_design_sym()
        except AttributeError:
            return None

    @property
    def sym_entry_number(self):
        try:
            return self.sym_entry.entry_number
        except AttributeError:
            return None

    @property
    def design_dimension(self):
        try:
            return self.sym_entry.get_design_dim()
        except AttributeError:
            return None

    @property
    def number_of_fragments(self):
        return len(self.fragment_observations) if self.fragment_observations else 0

    @property
    def score(self):
        try:
            self.get_fragment_metrics()
            return self.center_residue_score / self.central_residues_with_fragment_overlap
        except ZeroDivisionError:
            self.log.error('No fragment information found! Fragment scoring unavailable.')
            return 0.0

    def pose_score(self):  # Todo merge with above
        """Returns:
            (float): The Nanohedra score as reported in Laniado, Meador, Yeates 2021
        """
        return self.all_residue_score

    def pose_metrics(self):
        """Gather all metrics relating to the Pose and the interfaces within the Pose

        Returns:
            (dict): {'nanohedra_score_normalized': , 'nanohedra_score_center_normalized':,
                     'nanohedra_score': , 'nanohedra_score_center': , 'number_fragment_residues_total': ,
                     'number_fragment_residues_center': , 'multiple_fragment_ratio': ,
                     'percent_fragment_helix': , 'percent_fragment_strand': ,
                     'percent_fragment_coil': , 'number_of_fragments': , 'total_interface_residues': ,
                     'percent_residues_fragment_total': , 'percent_residues_fragment_center': }
        """
        self.get_fragment_metrics()
        metrics = {
            'nanohedra_score': self.all_residue_score,
            'nanohedra_score_normalized': self.all_residue_score / self.fragment_residues_total
            if self.fragment_residues_total else 0.0,
            'nanohedra_score_center': self.center_residue_score,
            'nanohedra_score_center_normalized': self.center_residue_score / self.central_residues_with_fragment_overlap
            if self.central_residues_with_fragment_overlap else 0.0,
            'number_fragment_residues_total': self.fragment_residues_total,
            'number_fragment_residues_center': self.central_residues_with_fragment_overlap,
            'multiple_fragment_ratio': self.multiple_frag_ratio,
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

        if self.sym_entry:
            metrics['design_dimension'] = self.design_dimension
            for group_idx, name in enumerate(self.oligomer_names, 1):
                metrics['symmetry_group_%d' % group_idx] = getattr(self.sym_entry, 'group%d' % group_idx)
        else:
            metrics['design_dimension'] = 'asymmetric'

        for ent_idx, entity in enumerate(self.pose.entities, 1):
            if entity.is_oligomeric:
                metrics['entity_%d_symmetry' % ent_idx] = entity.symmetry
            metrics.update({'entity_%d_name' % ent_idx: entity.name,
                            'entity_%d_number_of_residues' % ent_idx: entity.number_of_residues,
                            'entity_%d_max_radius' % ent_idx: entity.furthest_point_from_reference(),
                            'entity_%d_n_terminal_helix' % ent_idx: entity.is_termini_helical(),
                            'entity_%d_c_terminal_helix' % ent_idx: entity.is_termini_helical(termini='c'),
                            'entity_%d_n_terminal_orientation' % ent_idx:
                                entity.terminal_residue_orientation_from_reference(),
                            'entity_%d_c_terminal_orientation' % ent_idx:
                                entity.terminal_residue_orientation_from_reference(termini='c')
                            })

        return metrics

    # @property
    # def fragment_observations(self):
    #     """Returns:
    #         (dict): {'1_2_24': [(78, 87, ...), ...], ...}
    #     """
    #     return self._fragment_observations

    @property
    def pose_transformation(self):
        """Provide the transformation parameters for the design in question

        Returns:
            (dict): {1: {'rotation': numpy.ndarray, 'translation': numpy.ndarray, 'rotation2': numpy.ndarray,
                         'translation2': numpy.ndarray},
                     2: {}}
        """
        if not self.transform_d:
            self.gather_pose_metrics()
            self.info['pose_transformation'] = self.transform_d
            self.log.debug('Using transformation parameters:\n\t%s'
                           % '\n\t'.join(pretty_format_table(self.transform_d.items())))

        return self.transform_d

    # def pdb_input_parameters(self):
    #     return self.pdb_dir1_path, self.pdb_dir2_path

    # def symmetry_parameters(self):
    #     return self.sym_entry_number, self.oligomer_symmetry_1, self.oligomer_symmetry_2, self.design_symmetry_pg

    def rotation_parameters(self):
        return self.rot_range_deg_pdb1, self.rot_range_deg_pdb2, self.rot_step_deg1, self.rot_step_deg2

    def degeneracy_parameters(self):
        return self.sym_entry.degeneracy_matrices_1, self.sym_entry.degeneracy_matrices_2

    def degen_and_rotation_parameters(self):
        return self.degeneracy_parameters(), self.rotation_parameters()

    def compute_last_rotation_state(self):
        number_steps1 = self.rot_range_deg_pdb1 / self.rot_step_deg1
        number_steps2 = self.rot_range_deg_pdb2 / self.rot_step_deg1

        return int(number_steps1), int(number_steps2)

    def set_flags(self, symmetry=None, design_with_evolution=True, sym_entry_number=None, sym_entry=None, debug=False,
                  design_with_fragments=True, generate_fragments=True, write_fragments=True,
                  output_assembly=False, design_selector=None, ignore_clashes=False, script=True, mpi=False,
                  number_of_trajectories=PUtils.nstruct, skip_logging=False, analysis=False, copy_nanohedra=False,
                  **kwargs):
        # self.design_symmetry = symmetry
        self.sym_entry = sym_entry
        if not sym_entry and sym_entry_number:
            # self.sym_entry_number = sym_entry_number
            self.sym_entry = SymEntry(sym_entry_number)
        if symmetry:
            if symmetry == 'cryst':
                raise DesignError('This functionality is not possible yet. Please pass --symmetry by Symmetry Entry'
                                  ' Number instead (See Laniado & Yeates, 2020).')
                cryst_record_d = PDB.get_cryst_record(self.source)  # Todo must get self.source before attempt this call
                self.sym_entry = space_group_to_sym_entry[cryst_record_d['space_group']]
        self.design_selector = design_selector
        self.evolution = design_with_evolution
        self.design_with_fragments = design_with_fragments
        # self.fragment_file = fragments_exist
        self.query_fragments = generate_fragments
        self.write_frags = write_fragments
        self.output_assembly = output_assembly
        self.ignore_clashes = ignore_clashes
        self.number_of_trajectories = number_of_trajectories
        self.script = script  # Todo to reflect the run_in_shell flag
        self.mpi = mpi
        self.analysis = analysis
        self.skip_logging = skip_logging
        self.debug = debug
        self.copy_nanohedra = copy_nanohedra

    def return_symmetry_parameters(self):
        return dict(symmetry=self.design_symmetry, design_dimension=self.design_dimension,
                    uc_dimensions=self.uc_dimensions, expand_matrices=self.expand_matrices)

    def start_log(self, level=2):
        if self.skip_logging:  # set up null_logger
            self.log = null_log
            # self.log.debug('Null logger set')
            return None

        if self.debug:
            handler, level = 1, 1
            propagate = False
        else:
            propagate = True
            handler = 2
        self.log = start_log(name=str(self), handler=handler, level=level, location=os.path.join(self.path, self.name),
                             propagate=propagate)

    def connect_db(self, frag_db=None, design_db=None, score_db=None):
        if frag_db and isinstance(frag_db, FragmentDatabase):
            self.frag_db = frag_db

        if design_db and isinstance(design_db, FragmentDatabase):  # Todo DesignDatabase
            self.design_db = design_db

        if score_db and isinstance(score_db, FragmentDatabase):  # Todo ScoreDatabase
            self.score_db = score_db

    def directory_string_to_path(self, pose_id):
        """Set the DesignDirectory self.path to the root/pose-ID where the pose-ID is converted from dash separation to
         path separators"""
        assert self.program_root, 'No program_root attribute set! Cannot create a path from a pose_id without a ' \
                                  'program_root!'
        self.path = os.path.join(self.program_root, pose_id.replace('Projects-', 'Projects%s' % os.sep).replace(
            '_Designs-', '_Designs%s' % os.sep).replace('-', os.sep))

    def link_master_directory(self):
        """For common resources for all SymDesign outputs, ensure paths to these resources are available attributes"""
        if not os.path.exists(self.program_root):
            raise DesignError('Path does not exist!\n\t%s' % self.program_root)
        self.protein_data = os.path.join(self.program_root, PUtils.data.title())
        self.pdbs = os.path.join(self.protein_data, 'PDBs')  # Used to store downloaded PDB's
        self.orient_dir = os.path.join(self.pdbs, 'oriented')
        self.orient_asu_dir = os.path.join(self.pdbs, 'oriented_asu')
        self.refine_dir = os.path.join(self.pdbs, 'refined')
        self.stride_dir = os.path.join(self.pdbs, 'stride')
        # self.sdf_dir = os.path.join(self.pdbs, PUtils.symmetry_def_file_dir)
        self.sequence_info = os.path.join(self.protein_data, PUtils.sequence_info)
        self.sequences = os.path.join(self.sequence_info, 'sequences')
        self.profiles = os.path.join(self.sequence_info, 'profiles')
        self.all_scores = os.path.join(self.program_root, PUtils.all_scores)  # TODO db integration
        self.trajectories = os.path.join(self.all_scores, '%s_Trajectories.csv' % self.__str__())
        self.residues = os.path.join(self.all_scores, '%s_Residues.csv' % self.__str__())
        self.design_sequences = os.path.join(self.all_scores, '%s_Sequences.pkl' % self.__str__())

        # Ensure directories are only created once Pose Processing is called
        # self.make_path(self.protein_data)
        # self.make_path(self.pdbs)
        # self.make_path(self.orient_dir)
        # self.make_path(self.orient_asu_dir)
        # self.make_path(self.refine_dir)
        # self.make_path(self.sdf_dir)

    def link_master_database(self, database):
        """Connect the design to the master Database object to fetch shared resources"""
        self.database = database

    @handle_design_errors(errors=(DesignError, ))
    def set_up_design_directory(self):
        """Prepare output Directory and File locations. Each DesignDirectory always includes this format"""
        self.make_path(self.path)
        # if not os.path.exists(self.path):
        #     raise DesignError('Path does not exist!\n\t%s' % self.path)
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.scores_file = os.path.join(self.scores, PUtils.scores_file)
        self.designs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.scripts = os.path.join(self.path, PUtils.scripts)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)
        self.asu = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean_asu))
        if self.construct_pose:
            # self.source = self.asu
            self.pose_file = os.path.join(self.source_path, PUtils.pose_file)
            self.frag_file = os.path.join(self.source_path, PUtils.frag_dir, PUtils.frag_text_file)
            if self.copy_nanohedra:  # copy nanohedra output directory to the design directory
                if os.path.exists(self.frags):  # only present if a copy or generate fragments command is issued
                    raise DesignError('The directory %s already exists! Can\'t complete set up without an overwrite!')
                shutil.copytree(self.source_path, self.path)
            else:
                shutil.copy(self.pose_file, self.path)
                shutil.copy(self.frag_file, self.path)
        else:
            self.pose_file = os.path.join(self.path, PUtils.pose_file)
            self.frag_file = os.path.join(self.frags, PUtils.frag_text_file)

        if not self.source and os.path.exists(self.asu):
            self.source = self.asu
        else:
            try:
                self.source = glob(os.path.join(self.path, '%s.pdb' % self.name))[0]
            except IndexError:  # glob found no files
                self.source = None
        self.assembly = os.path.join(self.path, '%s_%s' % (self.name, PUtils.assembly))
        self.refine_pdb = '%s_for_refine.pdb' % os.path.splitext(self.asu)[0]
        self.consensus_pdb = '%s_for_consensus.pdb' % os.path.splitext(self.asu)[0]
        self.refined_pdb = os.path.join(self.designs, os.path.basename(self.refine_pdb))
        self.consensus_design_pdb = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        self.serialized_info = os.path.join(self.data, 'info.pkl')
        self.pdb_list = os.path.join(self.scripts, 'design_files.txt')

        if os.path.exists(self.serialized_info):  # Pose has already been processed. We can assume files are available
            self.info = unpickle(self.serialized_info)
            self._info = self.info.copy()  # create a copy of the state upon initialization
            if self.info.get('nanohedra'):
                self.oligomer_names = self.info.get('oligomer_names', list())
                self.transform_d = self.info.get('pose_transformation', dict())
            self.fragment_observations = self.info.get('fragments', None)

            # if 'design' in self.info and self.info['design']:  # Todo, respond to the state
            #     dummy = True

        self.start_log()
        # if os.path.exists(self.frag_file):
        # if self.info['fragments']:
        # self.gather_fragment_info()
        # self.get_fragment_metrics(from_file=True)
        # if os.path.exists(self.pose_file) and not self.nano:
        #     self.gather_pose_metrics()
        #     self.composition = '_'.join(self.pose_id.split('_')[:2])
        #     self.info['composition'] = self.composition

    def get_wildtype_file(self):
        """Retrieve the wild-type file name from Design Directory"""
        wt_file = glob(self.asu)
        assert len(wt_file) == 1, '%s: More than one matching file found with %s' % self.asu

        return wt_file[0]

    def get_designs(self):  # design_type=PUtils.interface_design
        """Return the paths of all design files in a DesignDirectory"""
        return glob('%s/*.pdb' % self.designs)
        # return glob(os.path.join(self.designs, '*%s*' % design_type))

    # TODO generators for the various directory levels using the stored directory pieces
    def get_building_block_dir(self, building_block):
        for sym_idx, symm in enumerate(self.program_root):
            try:
                bb_idx = self.building_block_dirs[sym_idx].index(building_block)
                return os.path.join(self.program_root[sym_idx], self.building_block_dirs[sym_idx][bb_idx])
            except ValueError:
                continue
        return None

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
        """Set/get fragment metrics for all fragment observations in the design"""
        self.log.debug('Starting fragment metric collection')
        # if self.info.get('fragments', None):
        if self.fragment_observations:  # check if fragment generation has been populated somewhere
            # frag_metrics = self.pose.return_fragment_metrics(fragments=self.info.get('fragments'))
            frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
        elif self.pose.fragment_queries:
            self.log.debug('Fragment observations found in Pose. Adding to the Design state')
            self.fragment_observations = self.pose.return_fragment_observations()
            self.info['fragments'] = self.fragment_observations
            frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
        elif os.path.exists(self.frag_file):  # try to pull them from disk
            self.gather_fragment_info()
            # frag_metrics = format_fragment_metrics(calculate_match_metrics(self.fragment_observations))
            frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
        # no fragments were attempted but returned nothing, set frag_metrics empty
        elif self.fragment_observations == list():
            frag_metrics = fragment_metric_template
        elif self.fragment_observations is None:
            # raise DesignError('Fragment observations have not been generated for this Design! Have you run %s on it?'
            #                   % PUtils.generate_fragments)
            self.log.warning('There are no fragment observations for this Design! Returning null values... '
                             'Have you run %s on it yet?' % PUtils.generate_fragments)
            frag_metrics = fragment_metric_template
        else:
            raise DesignError('Design hit a snag that shouldn\'t have happened. Please report this to the developers')
            # self.log.warning('%s: There are no fragment observations for this Design! Have you run %s on it yet?
            #                  ' Trying %s now...'
            #                  % (self.path, PUtils.generate_fragments, PUtils.generate_fragments))
            # self.generate_interface_fragments()

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
        # self.number_of_fragments = frag_metrics['number_fragments']  # Now @property self.number_of_fragments

        # Todo limit design_residues by the SASA accessible residues
        while True:
            design_residues = self.info.get('design_residues', False)
            if design_residues is None:  # when no interface was found
                design_residues = []
                break
            elif not design_residues:  # no attribute yet
                self.identify_interface()
            else:
                design_residues = design_residues.split(',')
                break

        self.total_interface_residues = len(design_residues)
        try:
            self.total_non_fragment_interface_residues = \
                max(self.total_interface_residues - self.central_residues_with_fragment_overlap, 0)
            # if interface_distance is different between interface query and fragment generation these can be < 0 or > 1
            self.percent_residues_fragment_center = \
                min(self.central_residues_with_fragment_overlap / self.total_interface_residues, 1)
            self.percent_residues_fragment_total = min(self.fragment_residues_total / self.total_interface_residues, 1)

        except ZeroDivisionError:
            self.log.warning('%s: No interface residues were found. Is there an interface in your design?'
                             % self.source)
            self.percent_residues_fragment_center, self.percent_residues_fragment_total = 0.0, 0.0

        if not self.pose.ss_index_array or not self.pose.ss_type_array:
            self.pose.interface_secondary_structure()  # source_db=self.database, source_dir=self.stride_dir)
        for number, elements in self.pose.split_interface_ss_elements.items():
            fragment_elements = set()
            # residues, entities = self.pose.split_interface_residues[number]
            for residue, entity, element in zip(*zip(*self.pose.split_interface_residues[number]), elements):
                if residue.number in self.center_residue_numbers:
                    fragment_elements.add(element)
            self.interface_ss_fragment_topology[number] = \
                ''.join(self.pose.ss_type_array[element] for element in fragment_elements)

    # @staticmethod
    # @handle_errors(errors=(FileNotFoundError, ))
    # def gather_docking_metrics(nano_master_log):
    #     with open(nano_master_log, 'r') as master_log:
    #         parameters = master_log.readlines()
    #         for line in parameters:
    #             if 'Oligomer 1 Input Directory: ' in line:  # "PDB 1 Directory Path: " or
    #                 pdb_dir1_path = line.split(':')[-1].strip()
    #             elif 'Oligomer 2 Input Directory: ' in line:  # "PDB 2 Directory Path: " or '
    #                 pdb_dir2_path = line.split(':')[-1].strip()
    #             # elif 'Master Output Directory: ' in line:
    #             #     master_outdir = line.split(':')[-1].strip()
    #             elif 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
    #                 sym_entry_number = int(line.split(':')[-1])
    #             # all commented below are reflected in sym_entry_number
    #             # elif 'Oligomer 1 Point Group Symmetry: ' in line:  # "Oligomer 1 Symmetry: "
    #             #     self.oligomer_symmetry_1 = line.split(':')[-1].strip()
    #             # elif 'Oligomer 2 Point Group Symmetry: ' in line:  # "Oligomer 2 Symmetry: " or
    #             #     self.oligomer_symmetry_2 = line.split(':')[-1].strip()
    #             # elif 'SCM Point Group Symmetry: ' in line:  # underlying point group # "Design Point Group Symmetry: "
    #             #     self.design_symmetry_pg = line.split(':')[-1].strip()
    #             # elif "Oligomer 1 Internal ROT DOF: " in line:  # ,
    #             #     self.internal_rot1 = line.split(':')[-1].strip()
    #             # elif "Oligomer 2 Internal ROT DOF: " in line:  # ,
    #             #     self.internal_rot2 = line.split(':')[-1].strip()
    #             # elif "Oligomer 1 Internal Tx DOF: " in line:  # ,
    #             #     self.internal_zshift1 = line.split(':')[-1].strip()
    #             # elif "Oligomer 2 Internal Tx DOF: " in line:  # ,
    #             #     self.internal_zshift2 = line.split(':')[-1].strip()
    #             # elif "Oligomer 1 Setting Matrix: " in line:
    #             #     self.set_mat1 = np.array(eval(line.split(':')[-1].strip()))
    #             # elif "Oligomer 2 Setting Matrix: " in line:
    #             #     self.set_mat2 = np.array(eval(line.split(':')[-1].strip()))
    #             # elif "Oligomer 1 Reference Frame Tx DOF: " in line:  # ,
    #             #     self.ref_frame_tx_dof1 = line.split(':')[-1].strip()
    #             # elif "Oligomer 2 Reference Frame Tx DOF: " in line:  # ,
    #             #     self.ref_frame_tx_dof2 = line.split(':')[-1].strip()
    #             # elif 'Resulting SCM Symmetry: ' in line:  # "Resulting Design Symmetry: " or
    #             #     self.design_symmetry = line.split(':')[-1].strip()
    #             # elif 'SCM Dimension: ' in line:  # "Design Dimension: " or
    #             #     self.design_dimension = int(line.split(':')[-1].strip())
    #             # elif 'SCM Unit Cell Specification: ' in line:  # "Unit Cell Specification: " or
    #             #     self.uc_spec_string = line.split(':')[-1].strip()
    #             # elif "Oligomer 1 ROT Sampling Range: " in line:
    #             #     self.rot_range_deg_pdb1 = int(line.split(':')[-1].strip())
    #             # elif "Oligomer 2 ROT Sampling Range: " in line:
    #             #     self.rot_range_deg_pdb2 = int(line.split(':')[-1].strip())
    #             elif "Oligomer 1 ROT Sampling Step: " in line:
    #                 rot_step_deg1 = int(line.split(':')[-1].strip())
    #             elif "Oligomer 2 ROT Sampling Step: " in line:
    #                 rot_step_deg2 = int(line.split(':')[-1].strip())
    #             # elif 'Degeneracies Found for Oligomer 1' in line:
    #             #     self.degen1 = line.split()[0]
    #             #     if self.degen1.isdigit():
    #             #         self.degen1 = int(self.degen1) + 1  # number of degens is added to the original orientation
    #             #     else:
    #             #         self.degen1 = 1  # No degens becomes a single degen
    #             # elif 'Degeneracies Found for Oligomer 2' in line:
    #             #     self.degen2 = line.split()[0]
    #             #     if self.degen2.isdigit():
    #             #         self.degen2 = int(self.degen2) + 1  # number of degens is added to the original orientation
    #             #     else:
    #             #         self.degen2 = 1  # No degens becomes a single degen
    #     sym_entry = SymEntry(sym_entry_number)
    #     return sym_entry

    @handle_errors(errors=(FileNotFoundError,))
    def gather_fragment_info(self):
        """Gather observed fragment metrics from fragment matching output"""
        fragment_observations = set()
        with open(self.frag_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[:6] == 'z-val:':
                    # overlap_rmsd_divded_by_cluster_rmsd
                    match_score = match_score_from_z_value(float(line[6:].strip()))
                elif line[:21] == 'oligomer1 ch, resnum:':
                    oligomer1_info = line[21:].strip().split(',')
                    chain1 = oligomer1_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number1 = int(oligomer1_info[1])
                elif line[:21] == 'oligomer2 ch, resnum:':
                    oligomer2_info = line[21:].strip().split(',')
                    chain2 = oligomer2_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number2 = int(oligomer2_info[1])
                elif line[:3] == 'id:':
                    cluster_id = map(str.strip, line[3:].strip().split('_'), 'ijk')
                    # use with self.oligomer_names to get mapped and paired oligomer id
                    fragment_observations.add((residue_number1, residue_number2, '_'.join(cluster_id), match_score))
                    # self.fragment_observations.append({'mapped': residue_number1, 'paired': residue_number2,
                    #                                    'cluster': cluster_id, 'match': match_score})
                # "mean rmsd: %s\n" % str(cluster_rmsd))
                # "aligned rep: int_frag_%s_%s.pdb\n" % (cluster_id, str(match_count)))
                # elif line[:23] == 'central res pair freqs:':
                #     # pair_freq = list(eval(line[22:].strip()))
                #     pair_freq = None
                #     # self.fragment_cluster_freq_d[cluster_id] = pair_freq
                #     # self.fragment_cluster_residue_d[cluster_id]['freq'] = pair_freq
        self.fragment_observations = [dict(zip(('mapped', 'paired', 'cluster', 'match'), frag_obs))
                                      for frag_obs in fragment_observations]
        self.info['fragments'] = self.fragment_observations  # inform the design state that fragments have been produced

    @handle_errors(errors=(FileNotFoundError,))
    def gather_pose_metrics(self):
        """Gather information for the docked Pose from Nanohedra output. Includes coarse fragment metrics"""
        with open(self.pose_file, 'r') as f:
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
                    self.transform_d[int(line[20:21])] = {'rotation': np.array(data)}
                elif line[:15] == 'INTERNAL Tx PDB':  # all below parsing lacks PDB number suffix such as PDB1 or PDB2
                    data = eval(line[17:].strip())
                    if data:  # == 'None'
                        self.transform_d[int(line[15:16])]['translation'] = np.array(data)
                    else:
                        self.transform_d[int(line[15:16])]['translation'] = np.array([0, 0, 0])
                elif line[:18] == 'SETTING MATRIX PDB':
                    data = eval(line[20:].strip())
                    self.transform_d[int(line[18:19])]['rotation2'] = np.array(data)
                elif line[:22] == 'REFERENCE FRAME Tx PDB':
                    data = eval(line[24:].strip())
                    if data:
                        self.transform_d[int(line[22:23])]['translation2'] = np.array(data)
                    else:
                        self.transform_d[int(line[22:23])]['translation2'] = np.array([0, 0, 0])
                elif 'Nanohedra Score:' in line:  # res_lev_sum_score
                    self.all_residue_score = float(line[16:].rstrip())
                elif 'CRYST1 RECORD:' in line:
                    cryst_record = line[15:].strip()
                    self.cryst_record = None if cryst_record == 'None' else cryst_record
                elif line[:31] == 'Canonical Orientation PDB1 Path':
                    self.canonical_pdb1 = line[:31].strip()
                elif line[:31] == 'Canonical Orientation PDB2 Path':
                    self.canonical_pdb2 = line[:31].strip()

    def pickle_info(self):
        """Write any design attributes that should persist over program run time to serialized file"""
        self.make_path(self.data)
        if self.info != self._info:  # if the state has changed from the original version
            pickle_object(self.info, self.serialized_info, out_path='')

    def prepare_rosetta_flags(self, symmetry_protocol=None, sym_def_file=None, pdb_path=None, out_path=os.getcwd()):
        """Prepare a protocol specific Rosetta flags file with program specific variables

        Args:
            symmetry_protocol (str): The type of symmetric protocol to use for Rosetta jobs the flags are valid for
            sym_def_file (str): The file specifying the symmetry system for Rosetta
        Keyword Args:
            out_path=cwd (str): Disk location to write the flags file
        Returns:
            (str): Disk location of the written flags file
        """
        # flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
        chain_breaks = {entity: entity.get_terminal_residue('c').number for entity in self.pose.entities}
        self.log.info('Found the following chain breaks in the ASU:\n\t%s'
                      % ('\n\t'.join('\tEntity %s, Chain %s Residue %d'
                                     % (entity.name, entity.chain_id, residue_number)
                                     for entity, residue_number in list(chain_breaks.items())[:-1])))
        self.log.info('Total number of residues in Pose: %d' % self.pose.number_of_residues)

        # Get ASU distance parameters
        if self.design_dimension:  # when greater than 0
            max_com_dist = 0
            for entity in self.pose.entities:
                com_dist = np.linalg.norm(self.pose.pdb.center_of_mass - entity.center_of_mass)
                # need ASU COM -> self.pose.pdb.center_of_mass, not Sym Mates COM -> (self.pose.center_of_mass)
                if com_dist > max_com_dist:
                    max_com_dist = com_dist
            dist = round(sqrt(ceil(max_com_dist)), 0)
            self.log.info('Expanding ASU into symmetry group by %f Angstroms' % dist)
        else:
            dist = 0

        if self.evolution:
            constraint_percent = 0.5
            free_percent = 1 - constraint_percent
        else:
            constraint_percent, free_percent = 0, 1

        variables = [('scripts', PUtils.rosetta_scripts), ('sym_score_patch', PUtils.sym_weights),
                     ('solvent_sym_score_patch', PUtils.solvent_weights), ('cst_value_sym', (cst_value / 2)),
                     ('dist', dist), ('constrained_percent', constraint_percent), ('free_percent', free_percent)]
        design_profile = self.info.get('design_profile')
        variables.extend([('design_profile', design_profile)] if design_profile else [])
        fragment_profile = self.info.get('fragment_profile')
        variables.extend([('fragment_profile', fragment_profile)] if fragment_profile else [])

        if not symmetry_protocol:
            symmetry_protocol = self.symmetry_protocol
        if not sym_def_file:
            sym_def_file = self.sym_def_file
        variables.extend([('symmetry', symmetry_protocol), ('sdf', sym_def_file)] if symmetry_protocol else [])
        out_of_bound_residue = list(chain_breaks.values())[-1] + 50
        variables.extend([(interface, residues) if residues else (interface, out_of_bound_residue)
                          for interface, residues in self.interface_residue_ids.items()])

        # assign any additional designable residues
        if self.pose.required_residues:
            variables.extend([('required_residues', ','.join(str(res.number) for res in self.pose.required_residues))])
        else:  # get an out of bounds index
            variables.extend([('required_residues', out_of_bound_residue)])

        # allocate any "core" residues based on central fragment information
        if self.center_residue_numbers:
            variables.extend([('core_residues', ','.join(map(str, self.center_residue_numbers)))])
        else:  # get an out of bounds index
            variables.extend([('core_residues', out_of_bound_residue)])

        flags = copy.copy(rosetta_flags)
        if pdb_path:
            flags.extend(['-out:path:pdb %s' % pdb_path, '-scorefile %s' % self.scores_file])
        else:
            flags.extend(['-out:path:pdb %s' % self.designs, '-out:path:score %s' % self.scores,  # TODO necessary?
                          '-scorefile %s' % self.scores_file])
        flags.append('-parser:script_vars %s' % ' '.join('%s=%s' % tuple(map(str, var_val)) for var_val in variables))

        out_file = os.path.join(out_path, 'flags')
        # out_file = os.path.join(out_path, 'flags_%s' % stage)
        with open(out_file, 'w') as f:
            f.write('%s\n' % '\n'.join(flags))

        return out_file

    @handle_design_errors(errors=(DesignError, AssertionError))
    def rosetta_interface_metrics(self, force_flags=False, development=False):
        """Generate a script capable of running Rosetta interface metrics analysis on the bound and unbound states"""
        main_cmd = copy.copy(script_cmd)
        flags = os.path.join(self.scripts, 'flags')
        if force_flags or not os.path.exists(flags):  # Generate a new flags_design file
            # Need to assign the designable residues for each entity to a interface1 or interface2 variable
            self.identify_interface()
            self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()
            self.make_path(self.scripts)
            flags = self.prepare_rosetta_flags(out_path=self.scripts)
            # if self.design_dimension is not None:  # can be 0
            #     protocol = PUtils.protocol[self.design_dimension]
            #     self.log.debug('Design has Symmetry Entry Number: %s (Laniado & Yeates, 2020)'
            #                    % str(self.sym_entry_number))
            #     if self.design_dimension == 0:  # point
            #         sym_def_file = sdf_lookup(self.sym_entry_number)
            #         main_cmd += ['-symmetry_definition', sym_def_file]
            #     else:  # layer or space
            #         sym_def_file = sdf_lookup(None, dummy=True)  # currently grabbing dummy.sym
            #         main_cmd += ['-symmetry_definition', 'CRYST1']
            #     self.log.info('Symmetry Option: %s' % protocol)
            # else:
            #     sym_def_file = 'null'
            #     protocol = PUtils.protocol[-1]  # Make part of self.design_dimension
            #     self.log.critical(
            #         'No symmetry invoked during design. Rosetta will still design your PDB, however, if it\'s an ASU it'
            #         ' may be missing crucial interface contacts. Is this what you want?')
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []

        pdb_list = os.path.join(self.scripts, 'design_files.txt')
        generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', pdb_list]
        main_cmd += ['-in:file:l', pdb_list, '-in:file:native', self.refined_pdb, '@%s' % flags, '-out:file:score_only',
                     self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
        if self.mpi:
            main_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + main_cmd
            self.script = True

        metric_cmd_bound = main_cmd + [os.path.join(PUtils.rosetta_scripts, 'interface_%s%s.xml'
                                                    % (PUtils.stage[3], '_DEV' if development else ''))]
        metric_cmd_unbound = main_cmd + [os.path.join(PUtils.rosetta_scripts, '%s%s.xml'
                                                      % (PUtils.stage[3], '_DEV' if development else ''))]
        metric_cmds = [metric_cmd_bound] + \
                      [metric_cmd_unbound + ['-parser:script_vars', 'interface=%d' % number] for number in [1, 2]]

        # Create executable to gather interface Metrics on all Designs
        write_shell_script(subprocess.list2cmdline(generate_files_cmd), name='interface_%s' % PUtils.stage[3],
                           out_path=self.scripts,  # status_wrap=self.serialized_info,
                           additional=[subprocess.list2cmdline(command) for command in metric_cmds])

    def custom_rosetta_script(self, script, force_flags=False, file_list=None, native=None, suffix=None,
                              score_only=None, variables=None, **kwargs):
        """Generate a custom script to dispatch to the design using a variety of parameters"""
        cmd = copy.copy(script_cmd)
        script_name = os.path.splitext(os.path.basename(script))[0]
        flags = os.path.join(self.scripts, 'flags')
        if force_flags or not os.path.exists(flags):  # Generate a new flags_design file
            # Need to assign the designable residues for each entity to a interface1 or interface2 variable
            self.identify_interface()
            self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()
            self.make_path(self.scripts)
            flags = self.prepare_rosetta_flags(out_path=self.scripts)

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

        cmd += ['-in:file:%s' % ('l' if file_list else 's'), pdb_input, '-in:file:native', native, '@%s' % flags] + \
            score + suffix + trajectories + \
            ['-parser:protocol', script] + variables
        if self.mpi:
            cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + cmd
            self.script = True

        write_shell_script(subprocess.list2cmdline(generate_files_cmd),
                           name=script_name, out_path=self.scripts, additional=[subprocess.list2cmdline(cmd)])

    def prepare_symmetry_for_rosetta(self):
        """For the specified design, locate/make the symmetry files necessary for Rosetta input

        Returns:
            (tuple[str, str]): The protocol to generate symmetry, and the location of the symmetry definition file
        """
        if self.design_dimension is not None:  # can be 0
            self.symmetry_protocol = PUtils.protocol[self.design_dimension]
            self.log.debug('Design has Symmetry Entry Number: %s (Laniado & Yeates, 2020)' % str(self.sym_entry_number))
            if self.design_dimension == 0:  # point
                if self.design_symmetry in ['T', 'O', 'I']:
                    self.sym_def_file = sdf_lookup(self.sym_entry_number)
                elif self.design_symmetry in valid_subunit_number.keys():  # todo standardize oriented versions of these
                    self.make_path(self.sdf_dir)
                    self.sym_def_file = self.pose.pdb.make_sdf(out_path=self.sdf_dir)
                else:
                    raise ValueError('The symmetry %s is unavailable at this time!')
            else:  # layer or space
                self.sym_def_file = sdf_lookup()  # currently grabbing dummy.sym
            self.log.info('Symmetry Option: %s' % self.symmetry_protocol)
        else:
            self.sym_def_file = 'null'
            self.symmetry_protocol = 'asymmetric'
            self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it\'s'
                              ' an ASU it may be missing crucial interface contacts. Is this what you want?')

    def prepare_rosetta_interface_design(self):
        """For the basic process of sequence design between two halves of an interface, write the necessary files for
        refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)

        Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
        on the DesignDirectory
        """
        # Set up the command base (rosetta bin and database paths)
        main_cmd = copy.copy(script_cmd)
        # if self.design_dimension is not None:  # can be 0
        #     protocol = PUtils.protocol[self.design_dimension]
        #     self.log.debug('Design has Symmetry Entry Number: %s (Laniado & Yeates, 2020)' % str(self.sym_entry_number))
        #     if self.design_dimension == 0:  # point
        #         sym_def_file = sdf_lookup(self.sym_entry_number)
        #         main_cmd += ['-symmetry_definition', sym_def_file]
        #     else:  # layer or space
        #         sym_def_file = sdf_lookup(None, dummy=True)  # currently grabbing dummy.sym
        #         main_cmd += ['-symmetry_definition', 'CRYST1']
        #     self.log.info('Symmetry Option: %s' % protocol)
        # else:
        #     sym_def_file = 'null'
        #     protocol = PUtils.protocol[-1]  # Make part of self.design_dimension
        #     self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it\'s'
        #                       ' an ASU it may be missing crucial interface contacts. Is this what you want?')
        main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
        self.log.info('Input Entities: %s' % ', '.join(entity.name for entity in self.pose.entities))

        # chain_breaks = {entity: entity.get_terminal_residue('c').number for entity in self.pose.entities}
        # self.log.info('Found the following chain breaks in the ASU:\n\t%s'
        #               % ('\n\t'.join('\tEntity %s, Chain %s Residue %d' % (entity.name, entity.chain_id, residue_number)
        #                              for entity, residue_number in list(chain_breaks.items())[:-1])))
        # self.log.info('Total number of residues in Pose: %d' % self.pose.number_of_residues)
        #
        # # Get ASU distance parameters
        # if self.design_dimension:  # when greater than 0
        #     max_com_dist = 0
        #     for oligomer in self.oligomers:
        #         com_dist = np.linalg.norm(self.pose.pdb.center_of_mass - oligomer.center_of_mass)
        #         # need ASU COM -> self.pose.pdb.center_of_mass, not Sym Mates COM -> (self.pose.center_of_mass)
        #         if com_dist > max_com_dist:
        #             max_com_dist = com_dist
        #     dist = round(math.sqrt(math.ceil(max_com_dist)), 0)
        #     self.log.info('Expanding ASU into symmetry group by %f Angstroms' % dist)
        # else:
        #     dist = 0
        #
        # # ------------------------------------------------------------------------------------------------------------
        # # Rosetta Execution formatting
        # # ------------------------------------------------------------------------------------------------------------
        # # Prepare flags file
        # # assign any additional designable residues
        # if self.pose.required_residues:
        #     required = ('required_residues', ','.join(str(residue.number) for residue in self.pose.required_residues))
        # else:  # get an out of bounds index
        #     required = ('required_residues', int(list(chain_breaks.values())[-1]) + 50)
        #
        # if self.evolution:
        #     constraint_percent = 0.5
        #     free_percent = 1 - constraint_percent
        # else:
        #     constraint_percent, free_percent = 0, 1
        #
        # flag_variables = [('scripts', PUtils.rosetta_scripts), ('sym_score_patch', PUtils.sym_weights),
        #                   ('solvent_sym_score_patch', PUtils.solvent_weights), ('cst_value_sym', (cst_value / 2)),
        #                   ('symmetry', protocol), ('sdf', sym_def_file), ('dist', dist),
        #                   required, *self.interface_residue_ids.items(),  # interface1 or interface2 variable
        #                   ('constrained_percent', constraint_percent), ('free_percent', free_percent)]
        # design_profile = self.info.get('design_profile')
        # flag_variables.extend([('design_profile', design_profile)] if design_profile else [])
        #
        # self.make_path(self.scripts)
        # flags = self.prepare_rosetta_flags(flag_variables, out_path=self.scripts)
        self.prepare_symmetry_for_rosetta()
        self.get_fragment_metrics()
        self.make_path(self.scripts)
        flags = self.prepare_rosetta_flags(out_path=self.scripts)

        shell_scripts = []
        # RELAX: Prepare command
        # refine_cmd = relax_cmd + ['-in:file:s', self.refine_pdb, '-parser:script_vars', 'switch=%s' % PUtils.stage[1]]
        if self.consensus:
            if self.design_with_fragments:
                consensus_cmd = main_cmd + relax_flags + \
                    ['@%s' % flags, '-in:file:s', self.consensus_pdb, '-in:file:native', self.refined_pdb,
                     '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[5]),
                     '-parser:script_vars', 'switch=%s' % PUtils.stage[5]]
                if self.script:
                    shell_scripts.append(write_shell_script(subprocess.list2cmdline(consensus_cmd),
                                                            name=PUtils.stage[5], out_path=self.scripts,
                                                            status_wrap=self.serialized_info))
                else:
                    self.log.info('Consensus Command: %s' % subprocess.list2cmdline(consensus_cmd))
                    consensus_process = subprocess.Popen(consensus_cmd)
                    consensus_process.communicate()
            else:
                self.log.critical('Cannot run consensus design without fragment info and none was found.'
                                  ' Did you mean to design with -generate_fragments False? You will need to run with'
                                  '\'True\' if you want to use fragments')

        # # Mutate all design positions to Ala before the Refinement
        # # mutated_pdb = copy.deepcopy(self.pose.pdb)  # this method is not implemented safely
        # # mutated_pdb = copy.copy(self.pose.pdb)  # this method is implemented, incompatible with downstream use!
        # # Have to use the self.pose.pdb as the Residue objects in entity_residues are from self.pose.pdb and not copy()!
        # for entity_pair, interface_residue_sets in self.pose.interface_residues.items():
        #     if interface_residue_sets[0]:  # check that there are residues present
        #         for idx, interface_residue_set in enumerate(interface_residue_sets):
        #             self.log.debug('Mutating residues from Entity %s' % entity_pair[idx].name)
        #             for residue in interface_residue_set:
        #                 self.log.debug('Mutating %d%s' % (residue.number, residue.type))
        #                 if residue.type != 'GLY':  # no mutation from GLY to ALA as Rosetta will build a CB.
        #                     self.pose.pdb.mutate_residue(residue=residue, to='A')
        #
        # self.pose.pdb.write(out_path=self.refine_pdb)
        # self.log.debug('Cleaned PDB for Refine: \'%s\'' % self.refine_pdb)
        #
        # # Create executable/Run FastRelax on Clean ASU/Consensus ASU with RosettaScripts
        # if self.script:
        #     self.log.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
        #     shell_scripts.append(write_shell_script(subprocess.list2cmdline(refine_cmd), name=PUtils.stage[1],
        #                                             out_path=self.scripts, status_wrap=self.serialized_info))
        # else:
        #     self.log.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
        #     refine_process = subprocess.Popen(refine_cmd)
        #     # Wait for Rosetta Refine command to complete
        #     refine_process.communicate()

        # DESIGN: Prepare command and flags file
        # flags_design = self.prepare_rosetta_flags(design_variables, PUtils.stage[2], out_path=self.scripts)
        evolutionary_profile = self.info.get('design_profile')
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        design_cmd = main_cmd + ['-in:file:pssm', evolutionary_profile] if evolutionary_profile else [] + \
            ['-in:file:s', self.refined_pdb, '-in:file:native', self.refined_pdb, '@%s' % flags,
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[2]),
             '-out:suffix', '_%s' % PUtils.stage[2], '-nstruct', str(self.number_of_trajectories)]

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        # all_chains = [entity.chain_id for entity in self.pose.entities]  # pose.interface_residue_ids}  # ['A', 'B', 'C']
        # # add symmetry definition files and set metrics up for oligomeric symmetry
        # if self.nano:
        #     design_variables.extend([('sdf%s' % chain, self.sdfs[name])
        #                              for chain, name in zip(all_chains, list(self.sdfs.keys()))])  # REQUIRES py3.6
        # flags_metric = self.prepare_rosetta_flags(refine_variables, PUtils.stage[3], out_path=self.scripts)

        generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', self.pdb_list]
        metric_cmd = main_cmd + \
            ['-in:file:l', self.pdb_list, '-in:file:native', self.refined_pdb, '@%s' % flags,
             '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[3])]

        if self.mpi:
            design_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + design_cmd
            metric_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + metric_cmd
            self.script = True

        metric_cmds = [metric_cmd + ['-parser:script_vars', 'interface=%d' % number] for number in [1, 2]]

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
        if self.script:
            self.log.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(design_cmd), name=PUtils.stage[2],
                                                    out_path=self.scripts, status_wrap=self.serialized_info))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(generate_files_cmd), name=PUtils.stage[3],
                                                    out_path=self.scripts, status_wrap=self.serialized_info,
                                                    additional=[subprocess.list2cmdline(command)
                                                                for command in metric_cmds]))
            for idx, metric_cmd in enumerate(metric_cmds, 1):
                self.log.info('Metrics Command %d: %s' % (idx, subprocess.list2cmdline(metric_cmd)))
        else:
            self.log.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            design_process = subprocess.Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            for idx, metric_cmd in enumerate(metric_cmds, 1):
                self.log.info('Metrics Command %d: %s' % (idx, subprocess.list2cmdline(metric_cmd)))
                metrics_process = subprocess.Popen(metric_cmd)
                metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if self.script:
            pass
        else:
            pose_s = self.design_analysis()
            out_path = os.path.join(self.all_scores, PUtils.analysis_file % ('All', ''))
            if not os.path.exists(out_path):
                header = True
            else:
                header = False
            pose_s.to_csv(out_path, mode='a', header=header)

        write_shell_script('', name=PUtils.interface_design, out_path=self.scripts,
                           additional=['bash %s' % design_script for design_script in shell_scripts])
        self.info['status'] = {PUtils.stage[stage]: False for stage in [1, 2, 3, 4, 5]}  # change active stage
        # write_commands(shell_scripts, name=PUtils.interface_design, out_path=self.scripts)

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
        # for oligomer in self.oligomers:
        #     oligomer.write(out_path=os.path.join(self.path, 'not_tsfmd_%s' % os.path.basename(oligomer.filepath)))
        self.oligomers = [oligomer.return_transformed_copy(**self.pose_transformation[oligomer_number])
                          for oligomer_number, oligomer in enumerate(self.oligomers, 1)]
        self.log.debug('Oligomers were transformed to the found docking parameters')

    def get_oligomers(self, refined=False, oriented=False):
        """Retrieve oligomeric files from either the design Database, the oriented directory, or the refined directory,
        or the design directory, and load them into job for further processing

        Keyword Args:
            master_db=None (Database): The Database object which stores relevant files
            refined=False (bool): Whether or not to use the refined oligomeric directory
            oriented=False (bool): Whether or not to use the oriented oligomeric directory
        Sets:
            self.oligomers (list[PDB])
        """
        if self.database:
            # refined, oriented = False, False
            if refined:
                source = 'refined'
            elif oriented:
                source = 'oriented'
            self.oligomers.clear()
            for oligomer_name in self.oligomer_names:
                self.oligomers.append(self.database.retrieve_data(from_source=source, name=oligomer_name))

        else:
            if refined:  # prioritize the refined version
                path = self.refine_dir
                for oligomer in self.oligomer_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % oligomer))[0]):
                        oriented = True  # fall back to the oriented version
                        self.log.debug('Couldn\'t find oligomers in the refined directory')
                        break
            if oriented:
                path = self.orient_dir
                for oligomer in self.oligomer_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % oligomer))[0]):
                        path = self.path
                        self.log.debug('Couldn\'t find oligomers in the oriented directory')

            if not refined and not oriented:
                path = self.path

            idx = 2  # initialize as 2. it doesn't matter if no names are found, but nominally it should be 2 for now
            oligomer_files = []
            for idx, name in enumerate(self.oligomer_names, 1):
                oligomer_files.extend(glob(os.path.join(path, '%s*.pdb*' % name)))  # first * is for DesignDirectory
            assert len(oligomer_files) == idx, \
                'Incorrect number of oligomers! Expected %d, %d found. Matched files from \'%s\':\n\t%s' \
                % (idx, len(oligomer_files), os.path.join(path, '*.pdb*'), oligomer_files)

            self.oligomers.clear()  # for every call we should reset the list
            for file in oligomer_files:
                self.oligomers.append(PDB.from_file(file, name=os.path.splitext(os.path.basename(file))[0],
                                                    log=self.log))
        self.log.debug('%d matching oligomers found' % len(self.oligomers))
        assert len(self.oligomers) == len(self.oligomer_names), \
            'Expected %d oligomers, but found %d' % (len(self.oligomers), len(self.oligomer_names))

    def load_pose(self):
        """For the design info given by a DesignDirectory source, initialize the Pose with self.source file,
        self.symmetry, self.design_selectors, self.fragment_database, and self.log objects

        Handles clash testing and writing the assembly if those options are True
        """
        # if self.nano:
        #     self.get_oligomers()
        #     if not self.oligomers:
        #         raise DesignError('No oligomers were found for this design! Cannot initialize pose without oligomers')
        #     self.pose = Pose.from_pdb(self.oligomers[0], symmetry=self.design_symmetry, log=self.log,
        #                               design_selector=self.design_selector, frag_db=self.frag_db,
        #                               ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
        #     #                         self.fragment_observations
        #     for oligomer in self.oligomers[1:]:
        #         self.pose.add_pdb(oligomer)
        #     self.pose.asu = self.pose.pdb  # set the asu
        #     self.pose.generate_symmetric_assembly()
        # else:
        if not self.source or not os.path.exists(self.source):
            # in case we initialized design without a .pdb or clean_asu.pdb (Nanohedra)
            # raise DesignError('No source file was found for this design! Cannot initialize pose without a source')
            self.transform_oligomers_to_pose()
            # else:
            #     self.get_oligomers()

            # # unnecessary with a transform_d stored in the design state
            # # write out oligomers to the designdirectory
            # for oligomer in self.oligomers:
            #     oligomer.write(out_path=os.path.join(self.path, 'TSFMD_%s' % os.path.basename(oligomer.filepath)))
            # # write out oligomer chains to the designdirectory
            # for oligomer in self.oligomers:
            #     with open(os.path.join(self.path, 'CHAINS_%s' % os.path.basename(oligomer.filepath)), 'w') as f:
            #         for chain in oligomer.chains:
            #             chain.write(file_handle=f)
            #     with open(os.path.join(self.path, 'ENTITY_CHAINS_%s' % os.path.basename(oligomer.filepath)), 'w') as f:
            #         for entity in oligomer.entities:
            #             for chain in entity.chains:
            #                 chain.write(file_handle=f)
            # # BOTH OF THESE PRODUCE THE SAME FILE WITH THE OLIGOMER TRANSFORMED
            # # SOMETHING IS HAPPENING WHEN THE POSE IS INITIALIZED CAUSING IT TO REVERT TO A NON TRANSFORMED VERSION?!

            asu = PDB.from_entities(chain.from_iterable([oligomer.entities for oligomer in self.oligomers]),
                                    cryst_record=self.cryst_record, log=self.log)
            self.pose = Pose.from_asu(asu, sym_entry=self.sym_entry,  # symmetry=self.design_symmetry,
                                      design_selector=self.design_selector, frag_db=self.frag_db, log=self.log,
                                      ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
            # This mechanism doesn't work as the entities attached chains are not handled correctly
            # self.pose = Pose.from_pdb(self.oligomers[0], sym_entry=self.sym_entry,  # symmetry=self.design_symmetry,
            #                           design_selector=self.design_selector, frag_db=self.frag_db, log=self.log,
            #                           ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
            # #                         self.fragment_observations
            # for oligomer in self.oligomers[1:]:
            #     self.pose.add_pdb(oligomer)
            # self.pose.asu = self.pose.pdb  # set the asu
            # asu = self.pose.get_contacting_asu()
        else:
            self.pose = Pose.from_asu_file(self.source, sym_entry=self.sym_entry,  # symmetry=self.design_symmetry,
                                           design_selector=self.design_selector, frag_db=self.frag_db, log=self.log,
                                           ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
        if self.pose_transformation:
            for idx, entity in enumerate(self.pose.entities, 1):
                entity.make_oligomer(sym=getattr(self.sym_entry, 'group%d' % idx), **self.pose_transformation[idx])
        else:
            # may switch this whole function to align the assembly identified by the asu entities PDB code after
            # download from PDB API
            self.pose.assign_entities_to_sub_symmetry()  # Todo debugggererer

        self.pose.generate_symmetric_assembly()
        # Save renumbered PDB to clean_asu.pdb
        if not os.path.exists(self.asu):
            # self.pose.pdb.write(out_path=os.path.join(self.path, 'pose_pdb.pdb'))  # not necessarily the most contacting
            # self.pose.pdb.write(out_path=self.asu)
            new_asu = self.pose.get_contacting_asu()
            new_asu.write(out_path=self.asu, header=self.cryst_record)
            self.log.info('Cleaned PDB: \'%s\'' % self.asu)

    @handle_design_errors(errors=(DesignError,))
    def rename_chains(self):
        """Standardize the chain names in incremental order found in the design source file"""
        pdb = PDB.from_file(self.source, log=self.log)
        pdb.reorder_chains()
        pdb.write(out_path=self.asu)

    @handle_design_errors(errors=(DesignError, ValueError, RuntimeError))
    def orient(self, to_design_directory=False):
        """Orient the Pose with the prescribed symmetry at the origin and symmetry axes in canonical orientations
        self.symmetry is used to specify the orientation
        """
        pdb = PDB.from_file(self.source, log=self.log)
        oriented_pdb = pdb.orient(sym=self.design_symmetry, out_dir=self.orient_dir)
        if to_design_directory:
            path = self.assembly
        else:
            path = self.orient_dir
            self.make_path(self.orient_dir)

        return oriented_pdb.write(out_path=path)

    @handle_design_errors(errors=(DesignError, AssertionError))
    def refine(self, to_design_directory=False, force_flags=False):
        """Refine the source PDB using self.symmetry to specify any symmetry"""
        relax_cmd = copy.copy(script_cmd)
        stage = PUtils.stage[1]
        if to_design_directory:  # original protocol to refine a pose as provided from Nanohedra
            flags = os.path.join(self.scripts, 'flags')
            flag_dir = self.scripts
            pdb_path = self.refined_pdb
            additional_flags = []
            # self.pose = Pose.from_pdb_file(self.source, symmetry=self.design_symmetry, log=self.log)
            self.load_pose()
            # Todo unnecessary? call self.load_pose with a flag for the type of file? how to reconcile with interface
            #  design and the asu versus pdb distinction. Can asu be implied by symmetry? Not for a trimer input that
            #  needs to be oriented and refined
            # assign designable residues to interface1/interface2 variables, not necessary for non complex PDB jobs
            self.identify_interface()
            # Mutate all design positions to Ala before the Refinement
            # mutated_pdb = copy.deepcopy(self.pose.pdb)  # this method is not implemented safely
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

            self.pose.pdb.write(out_path=self.refine_pdb)
            refine_pdb = self.refine_pdb
            self.log.debug('Cleaned PDB for Refine: \'%s\'' % self.refine_pdb)
        else:  # protocol to refine input structures, place in a common location, then transform for many jobs to source
            flags = os.path.join(self.refine_dir, 'refine_flags')
            flag_dir = self.refine_dir
            pdb_path = self.refine_dir  # os.path.join(self.refine_dir, '%s.pdb' % self.name)
            # out_put_pdb_path = os.path.join(self.refine_dir, '%s.pdb' % self.pose.name)
            refine_pdb = self.source
            additional_flags = ['-no_scorefile', 'true']

        if force_flags or not os.path.exists(flags):  # Generate a new flags file
            self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()
            self.make_path(flag_dir)
            flags = self.prepare_rosetta_flags(pdb_path=pdb_path, out_path=flag_dir)

        # RELAX: Prepare command
        relax_cmd += relax_flags + additional_flags + \
            ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else [] + \
            ['@%s' % flags, '-in:file:s', refine_pdb,
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[1]),
             '-parser:script_vars', 'switch=%s' % stage]
        self.log.info('%s Command: %s' % (stage.title(), subprocess.list2cmdline(relax_cmd)))

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if self.script:
            write_shell_script(subprocess.list2cmdline(relax_cmd), name=stage, out_path=flag_dir,
                               status_wrap=self.serialized_info)
        else:
            relax_process = subprocess.Popen(relax_cmd)
            relax_process.communicate()

    @handle_design_errors(errors=(DesignError, AssertionError, FileNotFoundError))
    def find_asu(self):
        """From a PDB with multiple Chains from multiple Entities, return the minimal configuration of Entities.
        ASU will only be a true ASU if the starting PDB contains a symmetric system, otherwise all manipulations find
        the minimal unit of Entities that are in contact
        """
        pdb = PDB.from_file(self.assembly, log=self.log)
        asu = pdb.return_asu()
        # ensure format matches clean_asu standard
        asu.write(out_path=self.asu)

    def symmetric_assembly_is_clash(self):
        if self.pose.symmetric_assembly_is_clash():
            if self.ignore_clashes:
                self.log.critical('The Symmetric Assembly contains clashes! %s is not viable.' % self.asu)
            else:
                raise DesignError('The Symmetric Assembly contains clashes! Design won\'t be considered. If you '
                                  'would like to generate the Assembly anyway, re-submit the command with '
                                  '--ignore_clashes')

    @handle_design_errors(errors=(DesignError, AssertionError))
    def expand_asu(self):
        """For the design info given by a DesignDirectory source, initialize the Pose with self.source file,
        self.symmetry, and self.log objects then expand the design given the provided symmetry operators and write to a
        file

        Reports on clash testing
        """
        if not self.pose:
            self.load_pose()
        if self.pose.symmetry:
            self.symmetric_assembly_is_clash()
            if self.output_assembly:  # True by default when expand_asu module is used
                self.pose.get_assembly_symmetry_mates()
                self.pose.write(out_path=self.assembly, increment_chains=self.increment_chains)
                self.log.info('Expanded Assembly PDB: \'%s\'' % self.assembly)

    @handle_design_errors(errors=(DesignError, AssertionError))
    def generate_interface_fragments(self):
        """For the design info given by a DesignDirectory source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation
        file
        """
        self.load_pose()
        self.make_path(self.frags)
        if not self.frag_db:
            self.log.warning('There was no FragmentDatabase passed to the Design. But fragment information was '
                             'requested. Each design is loading a separate instance. To maximize efficiency, pass --%s'
                             % Flags.generate_frags)
        self.identify_interface()
        self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=True)  # Todo parameterize write
        # if self.fragment_observations:
        #     self.log.warning('There are fragments already associated with this pose. They are being overwritten with '
        #                      'newly found fragments')
        # what is the below data used for? Seems to serve no purpose
        # self.fragment_observations = []
        # for observation in self.pose.fragment_queries.values():
        #     self.fragment_observations.extend(observation)
        self.info['fragments'] = self.pose.return_fragment_observations()
        # self.pickle_info()

    def identify_interface(self):
        """Initialize the design in a symmetric environment (if one is passed) and find the interfaces between
        entities. Sets the interface_residue_ids to map each interface to the corresponding residues."""
        self.expand_asu()
        self.pose.find_and_split_interface()
        for number, residues_entities in self.pose.split_interface_residues.items():
            self.interface_residue_ids['interface%d' % number] = \
                ','.join('%d%s' % (res.number, ent.chain_id) for res, ent in residues_entities)
        interface1, interface2 = \
            self.interface_residue_ids.get('interface1', None), self.interface_residue_ids.get('interface2', None)
        if interface1 and interface2:
            self.info['design_residues'] = '%s,%s' % (interface1, interface2)
            self.log.info('Interface Residues:\n\t%s'
                          % '\n\t'.join('interface%d: %s' % info for info in enumerate([interface1, interface2], 1)))
        else:
            self.info['design_residues'] = None
            self.log.info('No Interface Residues Found')

    @handle_design_errors(errors=(DesignError, AssertionError))
    def interface_design(self):
        """For the design info given by a DesignDirectory source, initialize the Pose then prepare all parameters for
        interfacial redesign between between Pose Entities. Aware of symmetry, design_selectors, fragments, and
        evolutionary information in interface design
        """
        self.identify_interface()
        if self.query_fragments:
            self.make_path(self.frags)
            # self.info['fragments'] = True
        elif self.fragment_observations or self.fragment_observations == list():
            pass  # fragment generation was run (maybe succeeded)
        elif self.design_with_fragments and not self.fragment_observations and os.path.exists(self.frag_file):
            self.gather_fragment_info()
        else:
            raise DesignError('Fragments were specified during design, but observations have not been yet been '
                              'generated for this Design! Try with the flag --query_fragments or run %s'
                              % PUtils.generate_fragments)
        self.make_path(self.data)
        self.pose.interface_design(evolution=self.evolution, fragments=self.design_with_fragments,
                                   query_fragments=self.query_fragments, fragment_source=self.fragment_observations,
                                   write_fragments=self.write_frags, des_dir=self)  # Todo frag_db=self.frag_db_SOURCE
        self.make_path(self.designs)
        self.make_path(self.scores)
        self.prepare_rosetta_interface_design()
        self.info['fragments'] = self.pose.return_fragment_observations()
        self.info['design_profile'] = self.pose.design_pssm_file
        self.info['evolutionary_profile'] = self.pose.pssm_file
        self.info['fragment_data'] = self.pose.interface_data_file
        self.info['fragment_profile'] = self.pose.fragment_pssm_file
        self.info['fragment_database'] = self.pose.frag_db

    @handle_design_errors(errors=(DesignError, AssertionError))
    def design_analysis(self, merge_residue_data=False, save_trajectories=True, figures=False):
        """Retrieve all score information from a DesignDirectory and write results to .csv file

        Keyword Args:
            merge_residue_data (bool): Whether to incorporate residue data into Pose DataFrame
            save_trajectories=False (bool): Whether to save trajectory and residue DataFrames
            figures=True (bool): Whether to make and save pose figures
        Returns:
            scores_df (pandas.DataFrame): DataFrame containing the average values from the input design directory
        """
        # Get design information including: interface residues, SSM's, and wild_type/design files
        profile_background = {}
        design_profile = self.info.get('design_profile')
        evolutionary_profile = self.info.get('evolutionary_profile')
        fragment_data = self.info.get('fragment_data')
        if design_profile:
            profile_background['design'] = parse_pssm(design_profile)
        if evolutionary_profile:  # lost accuracy ^ and v with round operation during write
            profile_background['evolution'] = parse_pssm(evolutionary_profile)
        if fragment_data:
            profile_background['fragment'] = unpickle(fragment_data)
            # issm_residues = set(profile_background['fragment'].keys())
        else:
            # issm_residues = set()
            self.log.info('Design has no fragment information')
        frag_db_ = self.info.get('fragment_database')
        interface_bkgd = get_db_aa_frequencies(PUtils.frag_directory[frag_db_]) if frag_db_ else {}

        # Gather miscellaneous pose specific metrics
        # ensure oligomers are present and if so, their metrics are pulled out. Happens when pose is scored.
        # self.load_pose()  # given scope of new metrics, using below instead due to interface residue requirement
        self.identify_interface()
        other_pose_metrics = self.pose_metrics()
        if not other_pose_metrics:
            raise DesignError('Design hit a snag that shouldn\'t have happened. Please report this to the developers')
            # raise DesignError('Scoring this design encountered major problems. Check the log (%s) for any errors which
            #                   'may have caused this and fix' % self.log.handlers[0].baseFilename)

        # Todo fold these into Model and attack these metrics from a Pose object
        #  This will get rid of the self.log
        wt_pdb = PDB.from_file(self.get_wildtype_file(), log=self.log)
        self.log.debug('Reordering wild-type chains')
        wt_pdb.reorder_chains()  # ensure chain ordering is A then B to match output from interface_design

        design_residues = self.info.get('design_residues', None)
        if design_residues == list():  # we should always get an empty list if we have got to this point
            raise DesignError('No residues were found with your design criteria... Your flags may be to stringent or '
                              'incorrect. Check input files for interface existance')
        elif design_residues is None:  # this should only happen if something is really failing in pose initialization
            raise DesignError('Design hit a snag that shouldn\'t have happened. Please report this to the developers')
        # 'design_residues' coming in as 234B (residue_number|chain), remove chain from residue, change type to int
        design_residues = set(int(residue.translate(digit_translate_table)) for residue in design_residues.split(','))
        self.log.debug('Found design residues: %s' % design_residues)

        int_b_factor = sum(wt_pdb.residue(residue).get_ave_b_factor() for residue in design_residues)
        other_pose_metrics['interface_b_factor_per_residue'] = round(int_b_factor / len(design_residues), 2)

        # initialize empty design dataframes
        stat_s, divergence_s, sim_series = pd.Series(), pd.Series(), []
        if os.path.exists(self.scores_file):
            self.log.debug('Found design scores in file: %s' % self.scores_file)
            # Get the scores from the score file on design trajectory metrics
            # all_design_scores = keys_from_trajectory_number(read_scores(self.scores_file))
            all_design_scores = read_scores(self.scores_file)
            self.log.debug('All designs with scores: %s' % ', '.join(all_design_scores.keys()))
            # Gather mutations for residue specific processing and design sequences

            # Find all designs which have corresponding pdb files and collect their sequences
            pdb_sequences = {}
            for file in self.get_designs():
                pdb = PDB.from_file(file, name=os.path.splitext(os.path.basename(file))[0], log=None, entities=False)
                pdb_sequences[pdb.name] = pdb.atom_sequences  # {chain: sequence, ...}

            # pulling from design pdbs and reorienting with format {chain: {name: sequence, ...}, ...}
            all_design_sequences = {}
            for design, chain_sequences in pdb_sequences.items():
                for chain, sequence in chain_sequences.items():
                    if chain not in all_design_sequences:
                        all_design_sequences[chain] = {}
                    all_design_sequences[chain][design] = sequence

            # #  v-this-v mechanism accounts for offsets from the reference sequence which aren't necessary YET
            # sequence_mutations = generate_multiple_mutations(wt_pdb.atom_sequences, pdb_sequences, pose_num=False)
            # sequence_mutations.pop('reference')
            # self.log.debug('Sequence Mutations: %s' % {design: {chain: format_mutations(mutations)
            #                                                     for chain, mutations in chain_mutations.items()}
            #                                            for design, chain_mutations in sequence_mutations.items()})
            # all_design_sequences = generate_sequences(wt_pdb.atom_sequences, sequence_mutations)
            # all_design_sequences = {chain: keys_from_trajectory_number(named_sequences)
            #                         for chain, named_sequences in all_design_sequences.items()}

            self.log.debug('Design sequences by chain: %s' % all_design_sequences)
            self.log.debug('All designs with sequences: %s'
                           % ', '.join(all_design_sequences[next(iter(all_design_sequences))].keys()))

            # Ensure data is present for both scores and sequences, then initialize DataFrames
            good_designs = sorted(set(all_design_sequences[next(iter(all_design_sequences))].keys()).
                                  intersection(set(all_design_scores.keys())))
            self.log.info('All designs with both sequence and scores: %s' % ', '.join(good_designs))
            all_design_scores = filter_dictionary_keys(all_design_scores, good_designs)
            all_design_sequences = {chain: filter_dictionary_keys(chain_sequences, good_designs)
                                    for chain, chain_sequences in all_design_sequences.items()}
            self.log.debug('Final design sequences by chain: %s' % all_design_sequences)

            idx_slice = pd.IndexSlice
            scores_df = pd.DataFrame(all_design_scores).T
            # Gather all columns into specific types for processing and formatting TODO move up
            rename_columns, per_res_columns, hbonds_columns = {}, [], []
            for column in scores_df.columns.to_list():
                if column.startswith('R_'):
                    rename_columns[column] = column.replace('R_', '')
                elif column.startswith('symmetry_switch'):
                    other_pose_metrics['symmetry'] = \
                        scores_df.loc[:, column][0].replace('make_', '').replace('_group', '')
                elif column.startswith('per_res_'):
                    per_res_columns.append(column)
                elif column.startswith('hbonds_res_selection'):
                    hbonds_columns.append(column)
            rename_columns.update(columns_to_rename)

            # Rename and Format columns
            scores_df.rename(columns=rename_columns, inplace=True)
            scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
            # Check proper input
            metric_set = necessary_metrics.difference(set(scores_df.columns))
            assert metric_set == set(), 'Missing required metrics: %s' % metric_set
            # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
            protocol_s = scores_df[groups]
            protocol_s.replace({'combo_profile': 'design_profile'}, inplace=True)  # ensure proper profile name

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # TODO learn know how to produce score terms in output score file. Not in FastRelax...
            remove_columns = rosetta_terms + hbonds_columns + per_res_columns + unnecessary + [groups]
            scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')
            self.log.debug('Score columns present: %s' % scores_df.columns.tolist())
            # Replace empty strings with numpy.notanumber (np.nan) and convert remaining to float
            scores_df.replace('', np.nan, inplace=True)
            scores_df.fillna(dict(zip(protocol_specific_columns, repeat(0))), inplace=True)
            scores_df = scores_df.astype(float)  # , copy=False, errors='ignore')

            # TODO remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            interface_hbonds = dirty_hbond_processing(all_design_scores)
            #                                         , offset=offset_dict) <- when hbonds are pose numbering
            # can't use hbond_processing (clean) in the case there is a design without metrics... columns not found!
            # interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)  # , offset=offset_dict)

            # all_mutations = keys_from_trajectory_number(
            #     generate_multiple_mutations(wt_pdb.atom_sequences, pdb_sequences))
            all_mutations = generate_multiple_mutations(wt_pdb.atom_sequences, pdb_sequences)
            all_mutations_no_chains = make_mutations_chain_agnostic(all_mutations)
            cleaned_mutations = simplify_mutation_dict(all_mutations_no_chains)
            residue_info = dirty_residue_processing(all_design_scores, cleaned_mutations, hbonds=interface_hbonds)
            #                                       offset=offset_dict)
            # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
            # residue_info = residue_processing(all_design_scores, cleaned_mutations, per_res_columns,
            #                                   hbonds=interface_hbonds)
            #                                   offset=offset_dict)

            # Calculate amino acid observation percent from residue dict and background SSM's
            observation_d = {profile: {design: mutation_conserved(residue_info, background)
                                       for design, residue_info in residue_info.items()}
                             for profile, background in profile_background.items()}

            # if profile_background.get('fragment'):
            #     # Keep residues in observed fragment if fragment information available for them
            #     observation_d['fragment'] = remove_interior_keys(observation_d['fragment'], issm_residues, keep=True)

            # Add observation information into the residue dictionary
            for design, info in residue_info.items():
                residue_info[design] = weave_sequence_dict(base_dict=info,
                                                           **{'observed_%s' % profile: design_obs_freqs[design]
                                                              for profile, design_obs_freqs in observation_d.items()})

            # Find the observed background for each profile, for each design in the pose
            pose_observed_bkd = {profile: {design: per_res_metric(freq) for design, freq in design_obs_freqs.items()}
                                 for profile, design_obs_freqs in observation_d.items()}
            for profile, observed_frequencies in pose_observed_bkd.items():
                scores_df['observed_%s' % profile] = pd.Series(observed_frequencies)

            # Process H-bond and Residue metrics to dataframe
            residue_df = pd.concat({design: pd.DataFrame(info) for design, info in residue_info.items()}).unstack()
            # returns multi-index column with residue number as first (top) column index, metric as second index
            # during residue_df unstack, all residues with missing dicts are copied as nan

            number_hbonds_s = pd.Series({design: len(hbonds) for design, hbonds in interface_hbonds.items()},
                                        name='number_hbonds')
            scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)
            reference_mutations = cleaned_mutations.pop('reference', None)
            scores_df['number_of_mutations'] = pd.Series({design: len(mutations)
                                                          for design, mutations in cleaned_mutations.items()})
            interior_residue_df = residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == 'interior']]
            # Check if any columns are > 50% interior. If so, return True for that column
            interior_residues = \
                interior_residue_df.columns[interior_residue_df.mean() > 0.5].remove_unused_levels().levels[0].to_list()
            interface_residues = set(residue_df.columns.levels[0].unique()).difference(interior_residues)
            assert len(interface_residues) > 0, 'No interface residues found! Design not considered'
            other_pose_metrics['percent_fragment'] = self.fragment_residues_total / len(interface_residues)
            other_pose_metrics['total_interface_residues'] = len(interface_residues)
            if interface_residues != design_residues:
                self.log.info('Residues %s are located in the interior' %
                              ', '.join(map(str, design_residues.difference(interface_residues))))

            # Interface B Factor TODO ensure asu.pdb has B-factors for Nanohedra
            int_b_factor = 0
            for residue in interface_residues:
                int_b_factor += wt_pdb.residue(residue).get_ave_b_factor()
            # this value updates the prior calculated one with only residues in interface, i.e. not interior
            other_pose_metrics['interface_b_factor_per_residue'] = round(int_b_factor / len(interface_residues), 2)

            # Add design residue information to scores_df such as how many core, rim, and support residues were measured
            for r_class in residue_classificiation:
                scores_df[r_class] = \
                    residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == r_class]].sum(axis=1)

            # Calculate new metrics from combinations of other metrics
            scores_df['total_interface_residues'] = len(interface_residues)
            scores_df = columns_to_new_column(scores_df, summation_pairs)
            scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
            scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
            scores_df['interface_composition_similarity'] = \
                scores_df.apply(interface_residue_composition_similarity, axis=1)
            # dropping 'total_interface_residues' after calculation as it is in other_pose_metrics
            scores_df.drop(clean_up_intermediate_columns + ['total_interface_residues'], axis=1, inplace=True,
                           errors='ignore')

            # Process dataframes for missing values and drop refine trajectory if present
            scores_df[groups] = protocol_s
            refine_index = scores_df[scores_df[groups] == PUtils.stage[1]].index
            # scores_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
            # residue_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
            scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
            residue_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
            # print(residue_df.columns[residue_df.isna().all(axis=0)])
            # residues_no_frags = residue_df.columns[residue_df.isna().all(axis=0)].remove_unused_levels().levels[0]
            residue_indices_no_frags = residue_df.columns[residue_df.isna().all(axis=0)]
            residue_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns such as obs_interface
            scores_na_index = scores_df.index[scores_df.isna().any(axis=1)]  # scores_df.where()
            residue_na_index = residue_df.index[residue_df.isna().any(axis=1)]
            drop_na_index = np.union1d(scores_na_index, residue_na_index)
            if drop_na_index.any():
                self.log.debug('Missing information in score columns: %s'
                               % scores_df.columns[scores_df.isna().any(axis=0)].tolist())
                self.log.debug('Missing information in residue columns: %s'
                               % residue_df.columns[residue_df.isna().any(axis=0)].tolist())
                protocol_s.drop(drop_na_index, inplace=True, errors='ignore')
                scores_df.drop(drop_na_index, inplace=True, errors='ignore')
                residue_df.drop(drop_na_index, inplace=True, errors='ignore')
                for idx in drop_na_index:
                    residue_info.pop(idx, None)
                self.log.warning('Dropped designs from analysis due to missing values: %s' % ', '.join(scores_na_index))
                # might have to remove these from all_design_scores in the case that that is used as a dictionary again
            other_pose_metrics['observations'] = len(scores_df)

            # POSE ANALYSIS
            # cst_weights are very large and destroy the mean. remove v'drop' if consensus is run multiple times
            trajectory_df = scores_df.sort_index().drop(PUtils.stage[5], axis=0, errors='ignore')
            # TODO v consensus only?
            assert len(trajectory_df.index.to_list()) > 0, 'No designs left to analyze in this pose!'

            # Get total design statistics for every sequence in the pose and every protocol specifically
            protocol_groups = scores_df.groupby(groups)
            # protocol_groups = trajectory_df.groupby(groups)
            designs_by_protocol = {protocol: scores_df.index[indices].values.tolist()  # <- df must be from same source
                                   for protocol, indices in protocol_groups.indices.items()}
            designs_by_protocol.pop(PUtils.stage[5], None)  # remove consensus if present
            # designs_by_protocol = {protocol: trajectory_df.index[indices].values.tolist()
            #                        for protocol, indices in protocol_groups.indices.items()}
            # Get unique protocols
            # unique_protocols = trajectory_df[groups].unique().tolist()
            unique_protocols = list(designs_by_protocol.keys())
            self.log.info('Unique Design Protocols: %s' % ', '.join(unique_protocols))
            pose_stats, protocol_stats = [], []
            for idx, stat in enumerate(stats_metrics):
                pose_stats.append(getattr(trajectory_df, stat)().rename(stat))
                protocol_stats.append(getattr(protocol_groups, stat)())

            protocol_stats[stats_metrics.index('mean')]['observations'] = protocol_groups.size()
            protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats],
                                         keys=stats_metrics)  # .swaplevel(0, 1)
            pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
            stat_s = pd.concat([protocol_stats_s, pose_stats_s])
            # change statistic names for all df that are not groupby means
            for idx, stat in enumerate(stats_metrics):
                if stat != 'mean':
                    protocol_stats[idx].index = protocol_stats[idx].index.to_series().map(
                        {protocol: '%s_%s' % (protocol, stat) for protocol in unique_protocols})
            trajectory_df = pd.concat([trajectory_df, pd.concat(pose_stats, axis=1).T] + protocol_stats)
            # this concat puts back consensus design to trajectory_df since protocol_stats is calculated on scores_df

            # Calculate sequence statistics
            # first for entire pose
            pose_alignment = multi_chain_alignment(all_design_sequences)
            mutation_frequencies = filter_dictionary_keys(pose_alignment['counts'], interface_residues)
            # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
            #                                              both mut_freq and profile_background[profile] are one-indexed
            divergence = {'divergence_%s' % profile: position_specific_jsd(mutation_frequencies, background)
                          for profile, background in profile_background.items()}
            if interface_bkgd:
                divergence['divergence_interface'] = jensen_shannon_divergence(mutation_frequencies, interface_bkgd)
            # Get pose sequence divergence
            divergence_stats = {'%s_per_residue' % divergence_type: per_res_metric(stat)
                                for divergence_type, stat in divergence.items()}
            # pose_res_dict['hydrophobic_collapse_index'] = hydrophobic_collapse_index()  # TODO HCI to a single metric?

            # next for each protocol
            # designs_by_protocol = {}
            # stats_by_protocol = {protocol: {} for protocol in unique_protocols}
            # divergence_by_protocol = copy.deepcopy(stats_by_protocol)
            divergence_by_protocol = {protocol: {} for protocol in designs_by_protocol}
            for protocol, designs in designs_by_protocol.items():
                # designs_by_protocol[protocol] = protocol_s.index[protocol_s == protocol].tolist()
                # All of this is DONE FOR WHOLE POSE ABOVE, PULLED BY PROTOCOL instead of this extra work
                # Get per design observed background metric by protocol
                # for profile in profile_background:
                #     stats_by_protocol[protocol]['observed_%s' % profile] = per_res_metric(
                #         {design: pose_observed_bkd[profile][design] for design in designs_by_protocol[protocol]})

                # Gather the average number of residue classifications for each protocol
                # for res_class in residue_classificiation:
                #     stats_by_protocol[protocol][res_class] = \
                #         residue_df.loc[designs_by_protocol[protocol],
                #                        idx_slice[:, residue_df.columns.get_level_values(1) == res_class]].mean().sum()
                # stats_by_protocol[protocol]['observations'] = len(designs_by_protocol[protocol])
                # Get the interface composition similarity for each protocol
                # composition_d = {metric: stats_by_protocol[protocol][metric] for metric in residue_classificiation}
                # composition_d['interface_area_total'] = trajectory_df.loc[protocol, 'interface_area_total']
                # stats_by_protocol[protocol]['interface_composition_similarity'] = \
                #     interface_residue_composition_similarity(composition_d)

                # protocol_sequences = {chain: {design: sequence for design, sequence in design_sequences.items()
                #                               if design in designs_by_protocol[protocol]}
                #                       for chain, design_sequences in all_design_sequences.items()}
                protocol_alignment = multi_chain_alignment({chain: {design: design_seqs[design] for design in designs}
                                                            for chain, design_seqs in all_design_sequences.items()})
                protocol_mutation_freq = filter_dictionary_keys(protocol_alignment['counts'], interface_residues)
                protocol_res_dict = {'divergence_%s' % profile: position_specific_jsd(protocol_mutation_freq, bkgnd)
                                     for profile, bkgnd in profile_background.items()}  # ^ both are 1-idx
                if interface_bkgd:
                    protocol_res_dict['divergence_interface'] = jensen_shannon_divergence(protocol_mutation_freq,
                                                                                          interface_bkgd)
                # Get per residue divergence metric by protocol
                for divergence, sequence_info in protocol_res_dict.items():
                    divergence_by_protocol[protocol]['%s_per_residue' % divergence] = per_res_metric(sequence_info)
                    # stats_by_protocol[protocol]['%s_per_residue' % key] = per_res_metric(sequence_info)
                    # {protocol: 'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}
                # pose_res_dict['hydrophobic_collapse_index'] = hydrophobic_collapse_index()  # TODO

            protocol_divergence_s = pd.concat([pd.Series(divergence) for divergence in divergence_by_protocol.values()],
                                              keys=list(zip(repeat('sequence_design'), divergence_by_protocol)))
            pose_divergence_s = pd.concat([pd.Series(divergence_stats)], keys=[('sequence_design', 'pose')])
            divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])
            # Calculate protocol significance
            pvalue_df = pd.DataFrame()
            missing_protocols = protocols_of_interest.difference(unique_protocols)
            if missing_protocols:
                self.log.warning('Missing protocol%s \'%s\'. No protocol significance measurements for this design!'
                                 % ('s' if len(missing_protocols) > 1 else '', ', '.join(missing_protocols)))
            elif len(protocols_of_interest) == 1:
                self.log.warning('Can\'t measure protocol significance, only one protocol of interest!')
            else:
                # Test significance between all combinations of protocols
                for prot1, prot2 in combinations(sorted(protocols_of_interest), 2):
                    select_df = trajectory_df.loc[designs_by_protocol[prot1] + designs_by_protocol[prot2],
                                                  significance_columns]
                    # difference_s = sig_df.loc[prot1, significance_columns].sub(sig_df.loc[prot2, significance_columns])
                    difference_s = trajectory_df.loc[prot1, :].sub(trajectory_df.loc[prot2, :])
                    pvalue_df[(prot1, prot2)] = df_permutation_test(select_df, difference_s, compare='mean',
                                                                    group1_size=len(designs_by_protocol[prot1]))
                pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
                trajectory_df = pd.concat([trajectory_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

                # Compute sequence differences between each protocol
                residue_energy_df = \
                    residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == 'energy_delta']]

                scaler = StandardScaler()
                res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
                residue_energy_np = scaler.fit_transform(residue_energy_df.values)
                residue_energy_pc = res_pca.fit_transform(residue_energy_np)
                residue_energy_pc_df = \
                    pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                 columns=['pc%d' % idx for idx, _ in enumerate(res_pca.components_, 1)])

                seq_pca = PCA(PUtils.variance)
                residue_info.pop(PUtils.stage[1], None)  # Remove refine from analysis before PC calculation
                pairwise_sequence_diff_np = all_vs_all(residue_info, sequence_difference)
                pairwise_sequence_diff_np = scaler.fit_transform(pairwise_sequence_diff_np)
                seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
                # Compute the euclidean distance
                # pairwise_pca_distance_np = pdist(seq_pc)
                # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

                # Make PC DataFrame
                # First take all the principal components identified from above and merge with labels
                # Next the labels will be grouped and stats are taken for each group (mean is important)
                # All protocol means will have pairwise distance measured as a means of accessing similarity
                # These distance metrics will be reported in the final pose statistics
                seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_info.keys()),
                                         columns=['pc%d' % idx for idx, _ in enumerate(seq_pca.components_, 1)])
                # Merge principle components with labels
                seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)
                residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)

                # Gather protocol similarity/distance metrics
                sim_measures = {'sequence_distance': {}, 'energy_distance': {}}
                sim_stdev = {}  # 'similarity': None, 'seq_distance': None, 'energy_distance': None}
                grouped_pc_seq_df_dict, grouped_pc_energy_df_dict, similarity_stat_dict = {}, {}, {}
                for stat in stats_metrics:
                    grouped_pc_seq_df_dict[stat] = getattr(seq_pc_df.groupby(groups), stat)()
                    grouped_pc_energy_df_dict[stat] = getattr(residue_energy_pc_df.groupby(groups), stat)()
                    similarity_stat_dict[stat] = getattr(pvalue_df, stat)(axis=1)  # protocol pair : stat Series
                    if stat == 'mean':
                        # if renaming is necessary
                        # protocol_stats_s[stat].index = protocol_stats_s[stat].index.to_series().map(
                        #     {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
                        seq_pca_mean_distance_vector = pdist(grouped_pc_seq_df_dict[stat])
                        energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df_dict[stat])
                        # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
                        #                             for k in seq_pca_mean_distance_vector)
                        # find similarity between each protocol by taking row average of all p-values for each metric
                        # mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance Series
                        # mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
                        # sim_measures['similarity'] = mean_pvalue_s
                        similarity_stat_dict[stat].index = pd.MultiIndex.from_tuples(similarity_stat_dict[stat].index)
                        sim_measures['similarity'] = similarity_stat_dict[stat]

                        for vector_idx, seq_dist in enumerate(seq_pca_mean_distance_vector):
                            i, j = condensed_to_square(vector_idx, len(grouped_pc_seq_df_dict[stat].index))
                            sim_measures['sequence_distance'][(grouped_pc_seq_df_dict[stat].index[i],
                                                               grouped_pc_seq_df_dict[stat].index[j])] = seq_dist

                        for vector_idx, energy_dist in enumerate(energy_pca_mean_distance_vector):
                            i, j = condensed_to_square(vector_idx, len(grouped_pc_energy_df_dict[stat].index))
                            sim_measures['energy_distance'][(grouped_pc_energy_df_dict[stat].index[i],
                                                             grouped_pc_energy_df_dict[stat].index[j])] = energy_dist
                    elif stat == 'std':
                        # sim_stdev['similarity'] = similarity_stat_dict[stat]
                        # Todo need to square each pc, add them up, divide by the group number, then take the sqrt
                        sim_stdev['seq_distance'] = grouped_pc_seq_df_dict[stat]
                        sim_stdev['energy_distance'] = grouped_pc_energy_df_dict[stat]

                # for pc_stat in grouped_pc_seq_df_dict:
                #     self.log.info(grouped_pc_seq_df_dict[pc_stat])

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
                # Todo test
                sim_stdev_s = pd.concat(list(sim_stdev.values()),
                                        keys=list(zip(repeat('std'), sim_stdev.keys()))).swaplevel(1, 2)
                # sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s, sim_stdev_s]
                sim_series = [protocol_sig_s, similarity_sum_s, sim_measures_s]

                if figures:  # Todo ensure output is as expected
                    protocols_by_design = {design: protocol for protocol, designs in designs_by_protocol.items()
                                           for design in designs}
                    _path = os.path.join(self.all_scores, str(self))
                    # Set up Labels & Plot the PC data
                    protocol_map = {protocol: i for i, protocol in enumerate(designs_by_protocol)}
                    integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
                    pc_labels_group = [protocols_by_design[design] for design in residue_info]
                    # pc_labels_group = np.array([protocols_by_design[design] for design in residue_info])
                    pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_info]
                    fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
                    # plt.cla()

                    # for color_int, label in integer_map.items():  # zip(pc_labels_group, pc_labels_int):
                    #     ax.scatter(seq_pc[pc_labels_group == label, 0],
                    #                seq_pc[pc_labels_group == label, 1],
                    #                seq_pc[pc_labels_group == label, 2],
                    #                c=color_int, cmap=plt.cm.nipy_spectral, edgecolor='k')
                    scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral',
                                         edgecolor='k')
                    # handles, labels = scatter.legend_elements()
                    # # print(labels)  # ['$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$']
                    # ax.legend(handles, labels, loc='upper right', title=groups)
                    # # ax.legend(handles, [integer_map[label] for label in labels], loc="upper right", title=groups)
                    # # plt.axis('equal') # not possible with 3D graphs
                    # plt.legend()  # No handles with labels found to put in legend.
                    colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
                    custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20)
                                    for c in colors]
                    ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
                              bbox_to_anchor=(1.0, .5))
                    # # Add group mean to the plot
                    # for name, label in integer_map.items():
                    #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(),
                    #                seq_pc[pc_labels_group == label, 1].mean(),
                    #                seq_pc[pc_labels_group == label, 2].mean(), marker='x')
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    # plt.legend(pc_labels_group)
                    plt.savefig('%s_seq_pca.png' % _path)
                    plt.clf()
                    # Residue PCA Figure to assay multiple interface states
                    fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
                    scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2],
                                         c=pc_labels_int,
                                         cmap='Spectral', edgecolor='k')
                    colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
                    custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in
                                    colors]
                    ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left',
                              bbox_to_anchor=(1.0, .5))
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    plt.savefig('%s_res_energy_pca.png' % _path)

            # Format output and save Trajectory, Residue DataFrames, and PDB Sequences
            if save_trajectories:
                trajectory_df.sort_index(inplace=True, axis=1)
                residue_df.sort_index(inplace=True)
                # Add wild-type residue information in metrics for sequence comparison
                # find the solvent acessible surface area of the separated entities
                for entity in wt_pdb.entities:
                    entity.get_sasa()

                wild_type_residue_info = {}
                for res_number in residue_info[next(iter(residue_info))].keys():
                    # bsa_total is actually a sasa, but for formatting sake, I've called it a bsa...
                    wild_type_residue_info[res_number] = \
                        {'type': reference_mutations[res_number], 'core': None, 'rim': None, 'support': None,
                         # Todo implement wt energy metric during oligomer refinement?
                         'interior': 0, 'hbond': None, 'energy_delta': None,
                         'bsa_total': wt_pdb.residue(res_number).sasa, 'bsa_polar': None, 'bsa_hydrophobic': None,
                         'coordinate_constraint': None, 'residue_favored': None, 'observed_design': None,
                         'observed_evolution': None, 'observed_fragment': None}  # 'hot_spot': None}
                    relative_oligomer_sasa = calc_relative_sa(wild_type_residue_info[res_number]['type'],
                                                              wild_type_residue_info[res_number]['bsa_total'])
                    if relative_oligomer_sasa < 0.25:
                        wild_type_residue_info[res_number]['interior'] = 1
                    # if res_number in issm_residues and res_number not in residues_no_frags:
                    #     wild_type_residue_info[res_number]['observed_fragment'] = None

                wt_df = pd.concat([pd.DataFrame(wild_type_residue_info)], keys=['wild_type']).unstack()
                wt_df.drop(residue_indices_no_frags, inplace=True, axis=1)
                # only sort once as residues are in same order
                # wt_df.sort_index(level=0, inplace=True, axis=1, sort_remaining=False)
                # residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                residue_df = pd.concat([wt_df, residue_df], sort=False)
                residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                residue_df[(groups, groups)] = protocol_s
                # residue_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
                if merge_residue_data:
                    trajectory_df = pd.merge(trajectory_df, residue_df, left_index=True, right_index=True)
                else:
                    residue_df.to_csv(self.residues)
                trajectory_df.to_csv(self.trajectories)
                seq_file = pickle_object(all_design_sequences, '%s_Sequences' % str(self), out_path=self.all_scores)

            # Create figures
            # if figures:  # Todo include relevant .ipynb figures
        else:
            self.log.debug('No design scores found at %s' % self.scores_file)
        other_metrics_s = pd.concat([pd.Series(other_pose_metrics)], keys=[('dock', 'pose')])

        # CONSTRUCT: Create pose series and format index names
        pose_s = pd.concat([other_metrics_s, stat_s, divergence_s] + sim_series).swaplevel(0, 1)
        # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
        pose_s.drop([groups], level=2, inplace=True, errors='ignore')
        pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
        pose_s.name = str(self)

        return pose_s

    @handle_design_errors(errors=(DesignError, AssertionError))
    def select_sequences(self, weights=None, number=1, protocol=None):
        """Select sequences for further characterization. If weights, then user can prioritize by metrics, otherwise
         sequence with the most neighbors as calculated by sequence distance will be selected. If there is a tie, the
         sequence with the lowest weight will be selected

        Keyword Args:
            weights=None (iter): The weights to use in sequence selection
            number=1 (int): The number of sequences to consider for each design
            protocol=None (str): Whether a particular design protocol should be chosen
        Returns:
            (list[tuple[DesignDirectory, str]]): Containing the selected sequences found
        """
        # Load relevant data from the design directory
        trajectory_df = pd.read_csv(self.trajectories, index_col=0, header=[0])
        trajectory_df.dropna(inplace=True)
        if protocol:
            # unique_protocols = trajectory_df['protocol'].unique().tolist()
            # while True:
            #     protocol = input(
            #         'Do you have a protocol that you prefer to pull your designs from? Possible '
            #         'protocols include:\n%s' % ', '.join(unique_protocols))
            #     if protocol in unique_protocols:
            #         break
            #     else:
            #         print('%s is not a valid protocol, try again.' % protocol)
            #
            # designs = trajectory_df[trajectory_df['protocol'] == protocol].index.to_list()
            designs = trajectory_df[trajectory_df['protocol'] == protocol].index.to_list()
            if not designs:
                raise DesignError('No designs found for protocol %s!' % protocol)
        else:
            designs = trajectory_df.index.to_list()

        self.log.info('Number of starting trajectories = %d' % len(trajectory_df))

        if weights:
            filter_df = pd.DataFrame(master_metrics)
            # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
            # _df = trajectory_df.loc[final_indices, :]
            df = trajectory_df.loc[designs, :]
            self.log.info('Using weighting parameters: %s' % str(weights))
            _weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': weights[metric]}
                        for metric in weights}
            weight_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
            # weights_s = pd.Series(weights)
            weight_score_s_d = {}
            for metric in _weights:
                weight_score_s_d[metric] = \
                    df[metric].rank(ascending=weight_direction[_weights[metric]['direction']],
                                    method=_weights[metric]['direction'], pct=True) * _weights[metric]['value']

            # design_score_df = pd.DataFrame(weight_score_s_d)  # Todo simplify
            score_df = pd.concat(weight_score_s_d.values(), axis=1)
            design_list = score_df.sum(axis=1).sort_values(ascending=False).index.to_list()
            self.log.info('Final ranking of trajectories:\n%s' % ', '.join(pose for pose in design_list))

            return list(zip(repeat(self), design_list[:number]))
        else:
            # sequences_pickle = glob(os.path.join(self.all_scores, '%s_Sequences.pkl' % str(self)))
            # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
            #                                    os.path.join(self.all_scores, '%s_Sequences.pkl' % str(self))
            #
            # all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
            # {chain: {name: sequence, ...}, ...}
            all_design_sequences = unpickle(self.design_sequences)
            chains = list(all_design_sequences.keys())
            concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains])
                                      for design in designs]
            self.log.debug(chains)
            self.log.debug(concatenated_sequences)

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
                energy_s = trajectory_df.loc[final_designs.keys(), 'interface_energy']  # includes solvation energy
                # try:
                #     energy_s = pd.Series(energy_s)
                # except ValueError:
                #     raise DesignError('no dataframe')
                energy_s.sort_values(inplace=True)
                final_seqs = zip(repeat(self), energy_s.index.to_list()[:number])
            else:
                final_seqs = zip(repeat(self), final_designs.keys())

            return list(final_seqs)

    @staticmethod
    def make_path(path, condition=True):
        """Make a path if it doesn't exist yet"""
        if not os.path.exists(path) and condition:
            os.makedirs(path)

    def __str__(self):
        if self.program_root:
            # TODO integrate with designDB?
            return self.path.replace(self.program_root + os.sep, '').replace(os.sep, '-')
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]


def get_sym_entry_from_nanohedra_directory(nanohedra_dir):
    try:
        with open(os.path.join(nanohedra_dir, PUtils.master_log), 'r') as f:
            for line in f.readlines():
                if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                    return SymEntry(int(line.split(':')[-1]))
    except FileNotFoundError:
        raise FileNotFoundError('The Nanohedra Output Directory is malformed. Missing required docking file %s'
                                % os.path.join(nanohedra_dir, PUtils.master_log))
    raise DesignError('The Nanohedra Output docking file %s is malformed. Missing required info Nanohedra Entry Number'
                      % os.path.join(nanohedra_dir, PUtils.master_log))


def set_up_directory_objects(design_list, mode=PUtils.interface_design, project=None):
    """Create DesignDirectory objects from a directory iterable. Add program_root if using DesignDirectory strings"""
    return [DesignDirectory.from_nanohedra(design_path, nano=True, mode=mode, project=project)
            for design_path in design_list]


def set_up_pseudo_design_dir(path, directory, score):  # changed 9/30/20 to locate paths of interest at .path
    pseudo_dir = DesignDirectory(path, nano=False)
    # pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.composition = os.path.dirname(path)
    pseudo_dir.designs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir
