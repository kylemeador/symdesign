import os
import copy
import re
from math import ceil, sqrt
import shutil
from subprocess import Popen, list2cmdline
from glob import glob
from itertools import combinations, repeat  # chain as iter_chain

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from matplotlib.axes import Axes
# from mpl_toolkits.mplot3d import Axes3D
from Bio.Data.IUPACData import protein_letters_3to1
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

import PathUtils as PUtils
from Structure import Structure
from SymDesignUtils import unpickle, start_log, null_log, handle_errors, sdf_lookup, write_shell_script, DesignError, \
    match_score_from_z_value, handle_design_errors, pickle_object, filter_dictionary_keys, \
    all_vs_all, condensed_to_square, space_group_to_sym_entry, digit_translate_table, sym, pretty_format_table, \
    index_intersection, z_score
from Query import Flags
from CommandDistributer import reference_average_residue_weight, run_cmds, script_cmd, rosetta_flags
from PDB import PDB
from Pose import Pose, SymmetricModel
from DesignMetrics import columns_to_rename, read_scores, join_columns, groups, necessary_metrics, division_pairs, \
    columns_to_new_column, delta_pairs, unnecessary, rosetta_terms, dirty_hbond_processing, dirty_residue_processing, \
    mutation_conserved, per_res_metric, residue_classificiation, interface_residue_composition_similarity, \
    stats_metrics, significance_columns, df_permutation_test, clean_up_intermediate_columns, fragment_metric_template, \
    protocol_specific_columns, rank_dataframe_by_metric_weights, background_protocol, \
    filter_df_for_index_by_value  # calc_relative_sa,
from SequenceProfile import parse_pssm, generate_mutations_from_reference, get_db_aa_frequencies, \
    simplify_mutation_dict, weave_sequence_dict, position_specific_jsd, sequence_difference, jensen_shannon_divergence, \
    multi_chain_alignment, hydrophobic_collapse_index, msa_from_dictionary
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
    def __init__(self, design_path, nano=False, pose_id=None, root=None, construct_pose=False, dock=False,
                 **kwargs):  # project=None,
        self.nano = nano
        if pose_id:
            self.directory_string_to_path(root, design_path)
            self.source_path = self.path
        else:
            self.source_path = os.path.abspath(design_path)
        self.name = os.path.splitext(os.path.basename(self.source_path))[0]
        self.source = None
        self.log = None
        self.debug = False
        self.dock = dock
        self.construct_pose = construct_pose
        # MasterDirectory path attributes
        self.database = None
        self.protein_data = None  # program_root/Data
        self.pdbs = None  # program_root/Data/PDBs
        self.orient_dir = None  # program_root/Data/PDBs/oriented
        self.orient_asu_dir = None  # program_root/Data/PDBs/oriented_asu
        self.refine_dir = None  # program_root/Data/PDBs/refined)
        self.stride_dir = None  # program_root/Data/PDBs/stride
        self.sequence_info = None  # program_root/SequenceInfo
        self.sequences = None  # program_root/SequenceInfo/sequences
        self.profiles = None  # program_root/SequenceInfo/profiles
        self.all_scores = None  # program_root/AllScores
        self.job_paths = None  # program_root/JobPaths
        self.sbatch_scripts = None  # program_root/Scripts
        self.trajectories = None  # program_root/AllScores/str(self)_Trajectories.csv
        self.residues = None  # program_root/AllScores/str(self)_Residues.csv
        self.design_sequences = None  # program_root/AllScores/str(self)_Sequences.pkl
        # DesignDirectory path attributes
        # self.scores = None  # /program_root/Projects/project_Designs/design/scores
        self.scores_file = None  # /program_root/Projects/project_Designs/design/data/name.sc
        self.designs = None  # /program_root/Projects/project_Designs/design/designs
        self.scripts = None  # /program_root/Projects/project_Designs/design/scripts
        # self.sdf_dir = None  # path/to/directory/sdf/
        self.frags = None  # /program_root/Projects/project_Designs/design/matching_fragments
        self.flags = None  # /program_root/Projects/project_Designs/design/scripts/flags
        self.frag_file = None  # /program_root/Projects/project_Designs/design/
        self.pose_file = None  # /program_root/Projects/project_Designs/design/
        self.data = None  # /program_root/Projects/project_Designs/design/data
        self.serialized_info = None  # /program_root/Projects/project_Designs/design/data/info.pkl
        self.asu = None  # /program_root/Projects/project_Designs/design/design_name_clean_asu.pdb
        self.assembly = None  # /program_root/Projects/project_Designs/design/design_name_assembly.pdb
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
        # Symmetry attributes Todo fully integrate with SymEntry
        self.sym_entry = kwargs.get('sym_entry', None)
        self.uc_dimensions = None
        self.expand_matrices = None
        self.transform_d = {}  # dict[pdb# (1, 2)] = {'rotation': matrix, 'translation': vector}
        self.cryst_record = None
        # Design flags
        self.command_only = kwargs.get('command_only', False)
        self.consensus = None  # Whether to run consensus or not
        self.design_with_fragments = False
        self.development = kwargs.get('development', False)
        self.evolution = False
        self.force_flags = kwargs.get('force_flags', False)
        self.legacy = kwargs.get('legacy', False)
        self.number_of_trajectories = None
        self.pre_refine = True
        self.query_fragments = False
        self.scout = kwargs.get('scout', False)
        self.structure_background = kwargs.get('structure_background', False)
        self.sequence_background = kwargs.get('sequence_background', False)
        self.write_frags = True
        # self.fragment_file = None
        # self.fragment_type = 'biological_interfaces'  # default for now, can be found in frag_db
        self.script = True
        self.mpi = False
        self.output_assembly = kwargs.get('output_assembly', False)
        self.increment_chains = kwargs.get('increment_chains', False)
        self.ignore_clashes = False
        # Analysis flags
        self.analysis = False
        self.skip_logging = False
        self.copy_nanohedra = False  # no construction specific flags
        self.nanohedra_root = None

        self.composition = None  # building_blocks (4ftd_5tch)
        self.design_db = None
        self.design_selector = None
        self.entity_names = []
        self.euler_lookup = None
        self.fragment_observations = None  # (dict): {'1_2_24': [(78, 87, ...), ...], ...}
        self.frag_db = None
        self.info = {}  # internal state info
        self._info = {}  # internal state info at load time
        self.oligomer_names = []
        self.oligomers = []
        self.pose = None  # contains the design's Pose object
        self.pose_id = None
        # self.sdfs = {}
        self.sym_def_file = None
        self.symmetry_protocol = None
        self.score_db = None
        # Metric attributes TODO MOVE Metrics
        self.interface_residue_ids = {}  # {'interface1': '23A,45A,46A,...' , 'interface2': '234B,236B,239B,...'}
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

        self.set_flags(**kwargs)  # has to be set before set_up_design_directory

        if self.nano:
            # source_path is design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2)
            if not os.path.exists(self.source_path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.source_path)
            # v used in dock_dir set up
            self.building_block_logs = []
            self.building_block_dirs = []

            self.canonical_pdb1 = None  # canonical pdb orientation
            self.canonical_pdb2 = None
            self.rot_step_deg1 = None  # TODO
            self.rot_step_deg2 = None  # TODO

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
            else:  # if self.construct_pose:
                path_components = self.source_path.split(os.sep)
                self.nanohedra_root = '/'.join(path_components[:-4])  # path_components[-5]
                # design_symmetry (P432)
                # self.pose_id = self.source_path[self.source_path.find(path_components[-3]) - 1:]\
                #     .replace(os.sep, '-')
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = os.path.join(self.projects, '%s_%s' % (path_components[-5],
                                                                              PUtils.design_directory))
                # make the newly required files
                self.make_path(self.program_root)
                self.make_path(self.projects)
                self.make_path(self.project_designs)
                # copy the master log
                if not os.path.exists(os.path.join(self.project_designs, PUtils.master_log)):
                    shutil.copy(os.path.join(self.nanohedra_root, PUtils.master_log), self.project_designs)

                self.composition = self.source_path[:self.source_path.find(path_components[-3]) - 1]
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                self.oligomer_names = list(map(str.lower, os.path.basename(self.composition).split('_')))

                self.pose_id = '-'.join(path_components[-4:])  # [-5:-1] because of trailing os.sep
                self.name = self.pose_id
                self.path = os.path.join(self.project_designs, self.name)

                if not self.construct_pose:  # no construction specific flags
                    self.write_frags = False
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
                self.make_path(self.path)

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
    def from_pose_id(cls, design_path, root=None, **kwargs):
        return cls(design_path, pose_id=True, root=root, **kwargs)

    @property
    def design_symmetry(self):
        try:
            return self.sym_entry.get_result_design_sym()
        except AttributeError:
            return

    @property
    def sym_entry_number(self):
        try:
            return self.sym_entry.entry_number
        except AttributeError:
            return

    @property
    def design_dimension(self):
        try:
            return self.sym_entry.get_design_dim()
        except AttributeError:
            return

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

    @property
    def design_profile(self):
        try:
            return self._design_profile
        except AttributeError:
            # self._design_profile = parse_pssm(os.path.join(os.path.abspath(self.path), self.info.get('design_profile')))
            self._design_profile = parse_pssm(os.path.join(os.path.abspath(self.path), 'data', 'design.pssm'))
            return self._design_profile

    @property
    def evolutionary_profile(self):
        try:
            return self._evolutionary_profile
        except AttributeError:
            # self._evolutionary_profile = \
                # parse_pssm(os.path.join(os.path.abspath(self.path), self.info.get('evolutionary_profile')))
            self._evolutionary_profile = \
                parse_pssm(os.path.join(os.path.abspath(self.path), 'data', 'evolutionary.pssm'))
            return self._evolutionary_profile

    @property
    def fragment_data(self):  # Todo associate fragment_data into info.pkl state as it is a separate I/O operation
        try:
            return self._fragment_data
        except AttributeError:
            # self._fragment_data = unpickle(os.path.join(os.path.abspath(self.path), self.info.get('fragment_data')))
            self._fragment_data = unpickle(os.path.join(os.path.abspath(self.path),
                                                        'data', '%s_fragment_profile.pkl' % self.fragment_database))
            return self._fragment_data

    @property
    def fragment_database(self):
        try:
            return self._fragment_database
        except AttributeError:
            self._fragment_database = self.info.get('fragment_database')
            return self._fragment_database

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
            # Todo must clarify symmetry separation if non-nanohedra
            for group_idx, name in enumerate(self.entity_names, 1):
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
            self.pose.assembly.get_sasa()

        # self.pose.assembly.write(out_path=os.path.join(self.path, 'POSE_ASSEMBLY.pdb'))
        entity_chain = self.pose.assembly.chain(entity.chain_id)
        n_term, c_term = False, False
        # Todo add reference when in a crystalline environment  # reference=pose_transformation[idx].get('translation2')
        n_termini_orientation = entity.terminal_residue_orientation_from_reference()
        if n_termini_orientation == 1:  # if outward
            if entity_chain.n_terminal_residue.relative_sasa > 0.25:
                n_term = True
        # Todo add reference when in a crystalline environment
        c_termini_orientation = entity.terminal_residue_orientation_from_reference(termini='c')
        if c_termini_orientation == 1:  # if outward
            if entity_chain.c_terminal_residue.relative_sasa > 0.25:
                c_term = True

        if report_if_helix:
            if self.database:
                parsed_secondary_structure = self.database.stride.retrieve_data(name=entity.name)
                if parsed_secondary_structure:
                    entity.fill_secondary_structure(secondary_structure=parsed_secondary_structure)
                else:
                    entity.stride()
            n_term = True if n_term and entity.is_termini_helical() else False
            c_term = True if c_term and entity.is_termini_helical(termini='c') else False

        return {'n': n_term, 'c': c_term}

    @property
    def pose_transformation(self):
        """Provide the transformation parameters for the design in question

        Returns:
            (dict): {1: {'rotation': numpy.ndarray, 'translation': numpy.ndarray, 'rotation2': numpy.ndarray,
                         'translation2': numpy.ndarray},
                     2: {}}
        """
        if not self.transform_d:
            self.retrieve_pose_metrics_from_file()
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
                  design_with_fragments=True, generate_fragments=False, write_fragments=True,
                  output_assembly=False, design_selector=None, ignore_clashes=False, script=True, mpi=False,
                  number_of_trajectories=PUtils.nstruct, skip_logging=False, analysis=False, copy_nanohedra=False,
                  **kwargs):
        # self.sym_entry = sym_entry
        if not self.sym_entry and sym_entry_number:
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
        # self.output_assembly = output_assembly
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
            propagate, no_log_name = False, False
        else:
            propagate, no_log_name = True, True
            handler = 2
        if self.nano and not self.construct_pose:
            # self.log = start_log(name=str(self), handler=handler, level=level, propagate=propagate)
            self.log = start_log(name=str(self), handler=3, propagate=True, no_log_name=no_log_name)
        else:
            self.log = start_log(name=str(self), handler=handler, level=level, no_log_name=no_log_name,
                                 location=os.path.join(self.path, self.name), propagate=propagate)

    def connect_db(self, frag_db=None, design_db=None, score_db=None):
        if frag_db and isinstance(frag_db, FragmentDatabase):
            self.frag_db = frag_db

        if design_db and isinstance(design_db, FragmentDatabase):  # Todo DesignDatabase
            self.design_db = design_db

        if score_db and isinstance(score_db, FragmentDatabase):  # Todo ScoreDatabase
            self.score_db = score_db

    def directory_string_to_path(self, root, pose_id):
        """Set the DesignDirectory self.path to the root/pose-ID where the pose-ID is converted from dash separation to
         path separators"""
        assert root, 'No program directory attribute set! Cannot create a path from a pose_id without a root directory!' \
                     ' Pass both -f with the pose_id\'s and -d with the specified directory'
        # assert self.program_root, 'No program_root attribute set! Cannot create a path from a pose_id without a ' \
        #                           'program_root!'
        if self.nano:
            self.path = os.path.join(root, pose_id.replace('-', os.sep))
        else:
            self.path = os.path.join(root, 'Projects', pose_id.replace('_Designs-', '_Designs%s' % os.sep))
        # .replace('Projects-', 'Projects%s' % os.sep)  .replace('-', os.sep))

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
        self.job_paths = os.path.join(self.program_root, 'JobPaths')
        self.sbatch_scripts = os.path.join(self.program_root, 'Scripts')
        self.all_scores = os.path.join(self.program_root, PUtils.all_scores)  # TODO ScoreDatabase integration
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
        self.make_path(self.path, condition=(not self.nano or self.copy_nanohedra or self.construct_pose))
        self.start_log()
        # self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.designs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.scripts = os.path.join(self.path, PUtils.scripts)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.flags = os.path.join(self.scripts, 'flags')
        self.data = os.path.join(self.path, PUtils.data)
        self.scores_file = os.path.join(self.data, '%s.sc' % self.name)
        self.serialized_info = os.path.join(self.data, 'info.pkl')
        self.asu = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean_asu))
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
        self.consensus_design_pdb = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        self.pdb_list = os.path.join(self.scripts, 'design_files.txt')
        if self.nano:
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
                self.info['sym_entry'] = self.sym_entry
                self.info['oligomer_names'] = self.oligomer_names
                self.entity_names = ['%s_1' % name for name in self.oligomer_names]
                # self.info['entity_names'] = self.entity_names  # Todo remove after T33
                # self.info['pre_refine'] = self.pre_refine  # Todo remove after T33
                self.pickle_info()  # save this info on the first copy so that we don't have to construct again
        else:
            self.pose_file = os.path.join(self.path, PUtils.pose_file)
            self.frag_file = os.path.join(self.frags, PUtils.frag_text_file)
            if os.path.exists(self.serialized_info):  # Pose has already been processed, gather state data
                self.info = unpickle(self.serialized_info)
                # if os.stat(self.serialized_info).st_size > 10000:
                #     print('Found pickled file with huge size %d. fragmentdatabase being removed'
                #           % os.stat(self.serialized_info).st_size)
                #     self.info['fragment_database'] = \
                #         getattr(self.info.get('fragment_database'), 'source', 'biological_interfaces')
                #     self.pickle_info()  # save this info so that we don't have this issue again!
                self._info = self.info.copy()  # create a copy of the state upon initialization
                # if self.info.get('nanohedra'):
                self.transform_d = self.info.get('pose_transformation', {})
                if not self.sym_entry:
                    self.sym_entry = self.info.get('sym_entry', None)
                else:
                    self.info['sym_entry'] = self.sym_entry
                self.oligomer_names = self.info.get('oligomer_names', [])
                # self.oligomer_names = self.info.get('entity_names', list())  # Todo TEMP addition
                self.entity_names = self.info.get('entity_names', [])
                # self._info = self.info.copy()  # create a copy of the state upon initialization
                # self.pre_refine = self.info.get('pre_refine', True)  # Todo remove after T33
                self.fragment_observations = self.info.get('fragments', None)  # None signifies query wasn't attempted
                self.interface_residue_ids = self.info.get('interface_residues', {})
        if self.pre_refine:
            self.refined_pdb = self.asu
            self.scouted_pdb = '%s_scout.pdb' \
                               % os.path.join(self.designs, os.path.basename(os.path.splitext(self.refined_pdb)[0]))
        else:
            self.refined_pdb = os.path.join(self.designs, os.path.basename(self.refine_pdb))
            self.scouted_pdb = '%s_scout.pdb' % os.path.splitext(self.refined_pdb)[0]

    def get_wildtype_file(self):
        """Retrieve the wild-type file name from Design Directory"""
        wt_file = glob(self.asu)
        assert len(wt_file) == 1, 'More than one matching file found during search %s' % self.asu

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
        else:
            if self.pose.fragment_queries:
                self.log.debug('Fragment observations found in Pose. Adding to the Design state')
                self.fragment_observations = self.pose.return_fragment_observations()
                self.info['fragments'] = self.fragment_observations
                frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
            elif os.path.exists(self.frag_file):  # try to pull them from disk
                self.log.debug('Fragment observations found on disk. Adding to the Design state')
                self.retrieve_fragment_info_from_file()
                # frag_metrics = format_fragment_metrics(calculate_match_metrics(self.fragment_observations))
                frag_metrics = self.pose.return_fragment_metrics(fragments=self.fragment_observations)
            # fragments were attempted, but returned nothing, set frag_metrics to the template (empty)
            elif self.fragment_observations == list():
                frag_metrics = fragment_metric_template
            elif self.fragment_observations is None:
                self.log.warning('There are no fragment observations for this Design! Returning null values... '
                                 'Have you run %s on it yet?' % PUtils.generate_fragments)
                frag_metrics = fragment_metric_template
                # self.log.warning('%s: There are no fragment observations for this Design! Have you run %s on it yet?
                #                  ' Trying %s now...' % (self.path, PUtils.generate_fragments, generate_fragments))
                # self.generate_interface_fragments()
            else:
                raise DesignError('Design hit a snag that shouldn\'t have happened. Please report to the developers')
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

        self.total_interface_residues = len(set(design_residues))
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
    def retrieve_fragment_info_from_file(self):
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
                    # chain1 = oligomer1_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number1 = int(oligomer1_info[1])
                elif line[:21] == 'oligomer2 ch, resnum:':
                    oligomer2_info = line[21:].strip().split(',')
                    # chain2 = oligomer2_info[0]  # doesn't matter when all subunits are symmetric
                    residue_number2 = int(oligomer2_info[1])
                elif line[:3] == 'id:':
                    cluster_id = map(str.strip, line[3:].strip().split('_'), 'ijk')
                    # use with self.entity_names to get mapped and paired oligomer id
                    fragment_observations.add((residue_number1, residue_number2, '_'.join(cluster_id), match_score))
        self.fragment_observations = [dict(zip(('mapped', 'paired', 'cluster', 'match'), frag_obs))
                                      for frag_obs in fragment_observations]
        self.info['fragments'] = self.fragment_observations  # inform the design state that fragments have been produced

    @handle_errors(errors=(FileNotFoundError,))
    def retrieve_pose_metrics_from_file(self):
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
        if self.nano and not self.construct_pose:
            return
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
        chain_breaks = {entity: entity.c_terminal_residue.number for entity in self.pose.entities}
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
                     ('solvent_sym_score_patch', PUtils.solvent_weights_sym),
                     ('solvent_score_patch', PUtils.solvent_weights),  # Todo put all above here in permanent flags
                     ('dist', dist), ('repack', 'yes'),
                     ('constrained_percent', constraint_percent), ('free_percent', free_percent)]
        # design_profile = self.info.get('design_profile')
        variables.extend([('design_profile', self.design_profile)] if self.design_profile else [])
        # fragment_profile = self.info.get('fragment_profile')
        variables.extend([('fragment_profile', self.fragment_profile)] if self.fragment_profile else [])

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
            flags.extend(['-out:path:pdb %s' % self.designs, '-scorefile %s' % self.scores_file])
        flags.append('-in:file:native %s' % self.refined_pdb)
        flags.append('-parser:script_vars %s' % ' '.join('%s=%s' % tuple(map(str, var_val)) for var_val in variables))

        out_file = os.path.join(out_path, 'flags')
        # out_file = os.path.join(out_path, 'flags_%s' % stage)
        with open(out_file, 'w') as f:
            f.write('%s\n' % '\n'.join(flags))

        return out_file

    @handle_design_errors(errors=(DesignError, AssertionError))
    def rosetta_interface_metrics(self):
        """Generate a script capable of running Rosetta interface metrics analysis on the bound and unbound states"""
        # metrics_flags = 'repack=yes'
        main_cmd = copy.copy(script_cmd)
        # Need to initialize the pose so each entity can get sdf created
        self.identify_interface()
        if not os.path.exists(self.flags) or self.force_flags:
            self.prepare_symmetry_for_rosetta()
            self.get_fragment_metrics()
            self.make_path(self.scripts)
            self.flags = self.prepare_rosetta_flags(out_path=self.scripts)

        pdb_list = os.path.join(self.scripts, 'design_files.txt')
        generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', pdb_list]
        main_cmd += ['-in:file:l', pdb_list, '@%s' % self.flags,
                     '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol']
        #              '-in:file:native', self.refined_pdb,
        if self.mpi:
            main_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + main_cmd
            self.script = True

        metric_cmd_bound = main_cmd + [os.path.join(PUtils.rosetta_scripts, 'interface_%s%s.xml'
                                                    % (PUtils.stage[3], '_DEV' if self.development else ''))]
        entity_cmd = main_cmd + [os.path.join(PUtils.rosetta_scripts, '%s_entity%s.xml'
                                              % (PUtils.stage[3], '_DEV' if self.development else ''))]
        metric_cmds = [metric_cmd_bound + (['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else [])]
        #               [metric_cmd_unbound + ['-parser:script_vars', 'interface=%d' % number] for number in [1, 2]]
        # metric_cmds = []
        for idx, entity in enumerate(self.pose.entities, 1):
            if entity not in self.pose.active_entities:  # todo what about when modifications occur to neighbor residue?
                continue
            if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
            else:
                entity_sdf = ''
            metric_cmd = entity_cmd + ['-parser:script_vars', 'repack=yes', 'entity=%d' % idx, entity_sdf,
                                       'symmetry=%s' % 'make_point_group' if entity.is_oligomeric else 'asymmetric']
            self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(metric_cmd)))
            metric_cmds.append(metric_cmd)
        # Create executable to gather interface Metrics on all Designs
        if self.script:
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
        cmd = copy.copy(script_cmd)
        script_name = os.path.splitext(os.path.basename(script))[0]
        flags = os.path.join(self.scripts, 'flags')
        if self.force_flags or not os.path.exists(flags):  # Generate a new flags_design file
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

        write_shell_script(list2cmdline(generate_files_cmd),
                           name=script_name, out_path=self.scripts, additional=[list2cmdline(cmd)])

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
                    # self.make_path(self.sdf_dir)
                    self.make_path(self.data)
                    self.sym_def_file = self.pose.pdb.make_sdf(out_path=self.data)
                else:
                    raise ValueError('The symmetry %s is unavailable at this time!')
            else:  # layer or space
                self.sym_def_file = sdf_lookup()  # currently grabbing dummy.sym
            self.log.info('Symmetry Option: %s' % self.symmetry_protocol)
        else:
            self.sym_def_file = sdf_lookup()  # currently grabbing dummy.sym
            self.symmetry_protocol = 'asymmetric'
            self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it\'s'
                              ' an ASU it may be missing crucial interface contacts. Is this what you want?')

    # def scout_interface(self):
    #     """For the basic process of sequence design between two halves of an interface, write the necessary files for
    #     refinement (FastRelax), redesign (FastDesign), and metrics collection (Filters & SimpleMetrics)
    #
    #     Stores job variables in a [stage]_flags file and the command in a [stage].sh file. Sets up dependencies based
    #     on the DesignDirectory
    #     """
    #     # Set up the command base (rosetta bin and database paths)
    #     main_cmd = copy.copy(script_cmd)
    #     main_cmd += ['-symmetry_definition', 'CRYST1'] if self.design_dimension > 0 else []
    #     self.prepare_symmetry_for_rosetta()
    #     self.get_fragment_metrics()
    #     self.make_path(self.scripts)
    #     flags = self.prepare_rosetta_flags(out_path=self.scripts)
    #
    #     # DESIGN: Prepare command and flags file
    #     evolutionary_profile = self.info.get('design_profile')
    #     scout_cmd = main_cmd + (['-in:file:pssm', evolutionary_profile] if evolutionary_profile else []) + \
    #         [
    #          '-in:file:s', self.refined_pdb, '@%s' % flags,
    #          '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[12]),
    #          '-out:suffix', '_%s' % PUtils.stage[12], '-no_nstruct_label', 'true']
    #
    #     # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
    #     metric_cmd = main_cmd + \
    #         ['-in:file:s', self.scouted_pdb,
    #          '@%s' % flags, '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true', '-parser:protocol',
    #          os.path.join(PUtils.rosetta_scripts, 'metrics_entity.xml')]
    #     metric_cmds = [metric_cmd + ['-parser:script_vars', 'interface=%d' % number] for number in [1, 2]]
    #
    #     # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
    #     self.log.info('Design Command: %s' % list2cmdline(scout_cmd))
    #     for idx, metric_cmd in enumerate(metric_cmds, 1):
    #         self.log.info('Metrics Command %d: %s' % (idx, list2cmdline(metric_cmd)))
    #     if self.script:
    #         write_shell_script(list2cmdline(scout_cmd), name=PUtils.stage[12], out_path=self.scripts,
    #                            additional=[list2cmdline(command) for command in metric_cmds])
    #     else:
    #         scout_process = Popen(scout_cmd)
    #         # Wait for Rosetta Design command to complete
    #         scout_process.communicate()
    #         for metric_cmd in metric_cmds:
    #             metrics_process = Popen(metric_cmd)
    #             metrics_process.communicate()
    #
    #     # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
    #     if not self.script:
    #         pose_s = self.design_analysis()
    #         out_path = os.path.join(self.all_scores, PUtils.analysis_file % ('All', ''))
    #         if not os.path.exists(out_path):
    #             header = True
    #         else:
    #             header = False
    #         pose_s.to_csv(out_path, mode='a', header=header)
    #
    #     self.info['status'] = {PUtils.stage[stage]: False for stage in [1, 2, 3, 4, 5]}  # change active stage

    def prepare_rosetta_interface_design(self):
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
        elif self.legacy:
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
        self.prepare_symmetry_for_rosetta()
        self.get_fragment_metrics()
        self.make_path(self.scripts)
        if not os.path.exists(self.flags) or self.force_flags:
            self.flags = self.prepare_rosetta_flags(out_path=self.scripts)

        if self.consensus:  # Todo add consensus sbatch generator to the symdesign main
            if self.design_with_fragments:
                consensus_cmd = main_cmd + relax_flags + \
                    ['@%s' % self.flags, '-in:file:s', self.consensus_pdb, '-in:file:native', self.refined_pdb,
                     '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[5]),
                     '-parser:script_vars', 'switch=%s' % PUtils.stage[5]]
                self.log.info('Consensus Command: %s' % list2cmdline(consensus_cmd))
                if self.script:
                    write_shell_script(list2cmdline(consensus_cmd), name=PUtils.stage[5], out_path=self.scripts)
                else:
                    consensus_process = Popen(consensus_cmd)
                    consensus_process.communicate()
            else:
                self.log.critical('Cannot run consensus design without fragment info and none was found.'
                                  ' Did you mean to design with -generate_fragments False? You will need to run with'
                                  '\'True\' if you want to use fragments')
        # DESIGN: Prepare command and flags file
        # evolutionary_profile = self.info.get('design_profile')
        # Todo must set up a blank -in:file:pssm in case the evolutionary matrix is not used. Design will fail!!
        design_cmd = main_cmd + (['-in:file:pssm', self.evolutionary_profile] if self.evolutionary_profile else []) + \
            ['-in:file:s', self.scouted_pdb if os.path.exists(self.scouted_pdb) else self.refined_pdb,
             '@%s' % self.flags, '-out:suffix', '_%s' % protocol,
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % protocol_xml1)] + nstruct_instruct + \
            out_file
        if additional_cmds:  # this is where hbnet_design_profile.xml is set up, which could be just design_profile.xml
            additional_cmds.append(
                main_cmd + (['-in:file:pssm', self.evolutionary_profile] if self.evolutionary_profile else []) +
                ['-in:file:silent', os.path.join(self.data, 'hbnet_selected.o'), '@%s' % self.flags,
                 '-in:file:silent_struct_type', 'binary',
                 # '-out:suffix', '_%s' % protocol,
                 '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % protocol)])  # + nstruct_instruct)

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        metric_cmd = main_cmd + metrics_pdb + \
            ['@%s' % self.flags, '-out:file:score_only', self.scores_file, '-no_nstruct_label', 'true',
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, 'metrics_entity.xml')]

        if self.mpi and not self.scout:
            design_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + design_cmd
            metric_cmd = run_cmds[PUtils.rosetta_extras] + [str(self.mpi)] + metric_cmd
            self.script = True

        self.log.info('Design Command: %s' % list2cmdline(design_cmd))
        metric_cmds = []
        for idx, entity in enumerate(self.pose.entities, 1):
            if entity not in self.pose.active_entities:
                continue
            if entity.is_oligomeric:  # make symmetric energy in line with SymDesign energies v
                entity_sdf = 'sdf=%s' % entity.make_sdf(out_path=self.data, modify_sym_energy=True)
            else:
                entity_sdf = ''
            _metric_cmd = metric_cmd + ['-parser:script_vars', metrics_flags, 'entity=%d' % idx, entity_sdf,
                                        'symmetry=%s' % 'make_point_group' if entity.is_oligomeric else 'asymmetric']
            self.log.info('Metrics Command for Entity %s: %s' % (entity.name, list2cmdline(_metric_cmd)))
            metric_cmds.append(_metric_cmd)
        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics
        if self.script:
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
        if not self.script:
            pose_s = self.design_analysis()
            out_path = os.path.join(self.all_scores, PUtils.analysis_file % ('All', ''))
            if not os.path.exists(out_path):
                header = True
            else:
                header = False
            pose_s.to_csv(out_path, mode='a', header=header)

        self.info['status'] = {PUtils.stage[stage]: False for stage in [1, 2, 3, 4, 5]}  # change active stage

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
            self.oligomers = [oligomer.return_transformed_copy(**self.pose_transformation[oligomer_number])
                              for oligomer_number, oligomer in enumerate(self.oligomers, 1)]
            self.log.debug('Oligomers were transformed to the found docking parameters')
        else:
            raise DesignError('The design could not be transformed as it is missing the required transformation '
                              'parameters. Were they generated properly?')

    def get_oligomers(self, refined=True, oriented=False):
        """Retrieve oligomeric files from either the design Database, the oriented directory, or the refined directory,
        or the design directory, and load them into job for further processing

        Keyword Args:
            master_db=None (Database): The Database object which stores relevant files
            refined=False (bool): Whether or not to use the refined oligomeric directory
            oriented=False (bool): Whether or not to use the oriented oligomeric directory
        Sets:
            self.oligomers (list[PDB])
        """
        source_preference = ['refined', 'oriented', 'design']
        if self.database:
            if refined:
                source_idx = 0
            elif oriented:
                source_idx = 1
            else:
                source_idx = 2
                self.log.warning('Falling back on oligomers present in the Design source which may not be refined. This'
                                 ' will lead to issues in sequence design if the structure is not refined first...')

            self.oligomers.clear()
            for name in self.oligomer_names:
                oligomer = None
                while not oligomer:
                    oligomer = self.database.retrieve_data(source=source_preference[source_idx], name=name)
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
            if source_idx > 0:
                self.pre_refine = False
        else:  # Todo consolidate this with above as far as iterative mechanism
            if refined:  # prioritize the refined version
                path = self.refine_dir
                for name in self.oligomer_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % name))[0]):
                        oriented = True  # fall back to the oriented version
                        self.log.debug('Couldn\'t find oligomers in the refined directory')
                        break
            if oriented:
                path = self.orient_dir
                for name in self.oligomer_names:
                    if not os.path.exists(glob(os.path.join(self.refine_dir, '%s.pdb*' % name))[0]):
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
        if self.pose:
            return
        if not self.source or not os.path.exists(self.source):
            # in case we initialized design without a .pdb or clean_asu.pdb (Nanohedra)
            # raise DesignError('No source file was found for this design! Cannot initialize pose without a source')
            self.log.info('No source file found. Fetching source from Database and transforming to Pose')
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
            entities = []
            for oligomer in self.oligomers:
                entities.extend(oligomer.entities)

            # list(iter_chain.from_iterable([oligomer.entities for oligomer in self.oligomers]))
            asu = PDB.from_entities(entities, name='%s-asu' % str(self), cryst_record=self.cryst_record, log=self.log)
            # self.pose = Pose.from_asu(asu, sym_entry=self.sym_entry, source_db=self.database,
            #                           design_selector=self.design_selector, frag_db=self.frag_db, log=self.log,
            #                           ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
            # self.pose.asu = self.pose.get_contacting_asu() # Todo test out PDB.from_chains() making new entities...
        else:  # |                              pass names if available v
            asu = PDB.from_file(self.source, name='%s-asu' % str(self), entity_names=self.entity_names, log=self.log)

        self.pose = Pose.from_asu(asu, sym_entry=self.sym_entry, source_db=self.database,
                                  design_selector=self.design_selector, frag_db=self.frag_db, log=self.log,
                                  ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
        if not self.entity_names:  # store the entity names if they were never generated
            self.entity_names = [entity.name for entity in self.pose.entities]
            self.log.info('Input Entities: %s' % ', '.join(self.entity_names))
            self.info['entity_names'] = self.entity_names

        if self.pose_transformation:
            for idx, entity in enumerate(self.pose.entities, 1):
                # Todo assumes a 1:1 correspondence between entities and oligomers (component group numbers) CHANGE
                entity.make_oligomer(sym=getattr(self.sym_entry, 'group%d' % idx), **self.pose_transformation[idx])
                # # write out new oligomers to the DesignDirectory TODO add flag to include these
                # out_path = os.path.join(self.path, '%s_oligomer.pdb' % entity.name)
                # entity.write_oligomer(out_path=out_path)
        else:
            # may switch this whole function to align the assembly identified by the asu entities PDB code after
            # download from PDB API
            raise DesignError('The functionality for specifying the pose transformation parameters is not possible yet.'
                              '\nThis pose is not designable with the current version of %s' % PUtils.program_name)
            self.pose.assign_entities_to_sub_symmetry()  # Todo debugggererer

        # self.pose.generate_symmetric_assembly()  # call is redundant with input asu's
        # Save renumbered PDB to clean_asu.pdb
        if not os.path.exists(self.asu):
            if self.nano and not self.construct_pose:
                return
            # self.pose.pdb.write(out_path=os.path.join(self.path, 'pose_pdb.pdb'))  # not necessarily most contacting
            new_asu = self.pose.get_contacting_asu()
            new_asu.write(out_path=self.asu, header=self.cryst_record)
            # self.pose.pdb.write(out_path=self.asu, header=self.cryst_record)
            self.info['pre_refine'] = self.pre_refine
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
        if self.design_symmetry:
            oriented_pdb = pdb.orient(sym=self.design_symmetry, out_dir=self.orient_dir)
            if to_design_directory:
                path = self.assembly
            else:
                path = self.orient_dir
                self.make_path(self.orient_dir)

            return oriented_pdb.write(out_path=path)
        else:
            self.log.critical(PUtils.warn_missing_symmetry % self.orient.__name__)

    @handle_design_errors(errors=(DesignError, AssertionError))
    def refine(self, to_design_directory=False):
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

        if self.force_flags or not os.path.exists(flags):  # Generate a new flags file
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
        self.log.info('%s Command: %s' % (stage.title(), list2cmdline(relax_cmd)))

        # Create executable/Run FastRelax on Clean ASU with RosettaScripts
        if self.script:
            write_shell_script(list2cmdline(relax_cmd), name=stage, out_path=flag_dir,
                               status_wrap=self.serialized_info)
        else:
            relax_process = Popen(relax_cmd)
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
        """Wrapper around the Pose symmetric_assembly_is_clash() to check at the Design level for clashes and raise
        DesignError if any are found, otherwise, continue with protocol
        """
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
            # if self.output_assembly:  # True by default when expand_asu module is used, otherwise False
            self.pose.get_assembly_symmetry_mates()
            self.pose.write(out_path=self.assembly, increment_chains=self.increment_chains)
            self.log.info('Expanded Assembly PDB: \'%s\'' % self.assembly)
        else:
            self.log.critical(PUtils.warn_missing_symmetry % self.expand_asu.__name__)
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError, AssertionError))
    def generate_interface_fragments(self):
        """For the design info given by a DesignDirectory source, initialize the Pose then generate interfacial fragment
        information between Entities. Aware of symmetry and design_selectors in fragment generation
        file
        """
        if not self.frag_db:
            self.log.warning('There was no FragmentDatabase passed to the Design. But fragment information was '
                             'requested. Each design is loading a separate instance. To maximize efficiency, pass --%s'
                             % Flags.generate_frags)
        self.identify_interface()
        self.make_path(self.frags, condition=self.write_frags)
        self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=self.write_frags)
        # if self.fragment_observations:
        #     self.log.warning('There are fragments already associated with this pose. They are being overwritten with '
        #                      'newly found fragments')
        # what is the below data used for? Seems to serve no purpose
        # self.fragment_observations = []
        # for observation in self.pose.fragment_queries.values():
        #     self.fragment_observations.extend(observation)
        self.info['fragments'] = self.pose.return_fragment_observations()
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    def identify_interface(self):
        """Initialize the design in a symmetric environment (if one is passed) and find the interfaces between
        entities. Sets the interface_residue_ids to map each interface to the corresponding residues."""
        # self.expand_asu()  # can't use this as it is a stand in for SymDesign call which needs to catch Errors!
        if not self.pose:
            self.load_pose()
        if self.pose.symmetry:
            self.symmetric_assembly_is_clash()
            if self.output_assembly:  # True by default when expand_asu module is used, otherwise False
                self.pose.get_assembly_symmetry_mates()
                self.pose.write(out_path=self.assembly, increment_chains=self.increment_chains)
                self.log.info('Expanded Assembly PDB: \'%s\'' % self.assembly)
        self.pose.find_and_split_interface()
        for number, residues_entities in self.pose.split_interface_residues.items():
            self.interface_residue_ids['interface%d' % number] = \
                ','.join('%d%s' % (res.number, ent.chain_id) for res, ent in residues_entities)
        interface1, interface2 = \
            self.interface_residue_ids.get('interface1', None), self.interface_residue_ids.get('interface2', None)
        if interface1 and interface2:
            self.info['design_residues'] = '%s,%s' % (interface1, interface2)
            self.info['interface_residues'] = self.interface_residue_ids
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
        if self.command_only:
            pass
        else:
            if self.query_fragments:
                self.make_path(self.frags)
                # self.info['fragments'] = True
            elif self.fragment_observations or self.fragment_observations == list():
                pass  # fragment generation was run and maybe succeeded. If not ^
            elif self.design_with_fragments and not self.fragment_observations and os.path.exists(self.frag_file):
                self.retrieve_fragment_info_from_file()
            elif not self.design_with_fragments:
                pass
            else:
                raise DesignError('Fragments were specified during design, but observations have not been yet been '
                                  'generated for this Design! Try with the flag --generate_fragments or run %s'
                                  % PUtils.generate_fragments)
            self.make_path(self.data)
            self.pose.interface_design(evolution=self.evolution, fragments=self.design_with_fragments,
                                       query_fragments=self.query_fragments, fragment_source=self.fragment_observations,
                                       write_fragments=self.write_frags, des_dir=self)  # Todo frag_db=self.frag_db.source
            self.make_path(self.designs)
            # self.make_path(self.scores)
            self.info['fragments'] = self.pose.return_fragment_observations()
            # Todo edit each of these files to be relative paths?
            self.info['design_profile'] = self.pose.design_pssm_file
            self.info['evolutionary_profile'] = self.pose.pssm_file
            self.info['fragment_data'] = self.pose.interface_data_file
            self.info['fragment_profile'] = self.pose.fragment_pssm_file
            self.info['fragment_database'] = self.pose.frag_db.source
        # if self.scout:
        #     self.scout_interface()
        # else:
        self.prepare_rosetta_interface_design()
        self.pickle_info()  # Todo remove once DesignDirectory state can be returned to the SymDesign dispatch w/ MP

    @handle_design_errors(errors=(DesignError, AssertionError))
    def design_analysis(self, merge_residue_data=False, save_trajectories=True, figures=False):
        """Retrieve all score information from a DesignDirectory and write results to .csv file

        Keyword Args:
            merge_residue_data (bool): Whether to incorporate residue data into Pose DataFrame
            save_trajectories=False (bool): Whether to save trajectory and residue DataFrames
            figures=True (bool): Whether to make and save pose figures
        Returns:
            (pandas.Series): Series containing summary metrics for all designs in the design directory
        """
        # Gather miscellaneous pose specific metrics
        # ensure oligomers are present and if so, their metrics are pulled out. Happens when pose is scored.
        # self.load_pose()  # given scope of new metrics, using below instead due to interface residue requirement
        self.identify_interface()
        if self.query_fragments:
            # self.generate_interface_fragments()  # inherently identifies the interface
            self.make_path(self.frags, condition=self.write_frags)
            self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=self.write_frags)
        other_pose_metrics = self.pose_metrics()
        if not other_pose_metrics:
            raise DesignError('Design hit a snag that shouldn\'t have happened. Please report this to the developers')
            # raise DesignError('Scoring this design encountered major problems. Check the log (%s) for any errors which
            #                   'may have caused this and fix' % self.log.handlers[0].baseFilename)

        # Todo fold these into Model and attack these metrics from a Pose object
        #  This will get rid of the self.log
        # wt_pdb = PDB.from_file(self.get_wildtype_file(), log=self.log)
        # wt_pdb = self.pose.pdb
        # self.log.debug('Reordering wild-type chains')
        # wt_pdb.reorder_chains()  # ensure chain ordering is A then B to match output from interface_design

        design_residues = self.info.get('design_residues')  # None
        if design_residues == list():  # we should always get an empty list if we have got to this point
            raise DesignError('No residues were found with your design criteria... Your flags may be to stringent or '
                              'incorrect. Check input files for interface existance')
        elif design_residues is None:  # this should only happen if something is really failing in pose initialization
            raise DesignError('Design hit a snag that shouldn\'t have happened. Please report this to the developers')
        # 'design_residues' coming in as 234B (residue_number|chain), remove chain from residue, change type to int
        design_residues = set(int(residue.translate(digit_translate_table)) for residue in design_residues.split(','))
        self.log.debug('Found design residues: %s' % design_residues)

        # Interface B Factor TODO ensure asu.pdb has B-factors for Nanohedra
        int_b_factor = sum(self.pose.pdb.residue(residue).b_factor for residue in design_residues)
        other_pose_metrics['interface_b_factor_per_residue'] = round(int_b_factor / len(design_residues), 2)
        # other_pose_metrics['interface_b_factor_per_residue'] = round(int_b_factor / len(interface_residues), 2)

        # initialize empty design dataframes
        stat_s, divergence_s, sim_series = pd.Series(), pd.Series(), []
        if os.path.exists(self.scores_file):
            self.log.debug('Found design scores in file: %s' % self.scores_file)
            # Get the scores from the score file on design trajectory metrics
            # all_design_scores = keys_from_trajectory_number(read_scores(self.scores_file))
            all_design_scores = read_scores(self.scores_file)
            self.log.debug('All designs with scores: %s' % ', '.join(all_design_scores.keys()))
            # Gather mutations for residue specific processing and design sequences Todo won't work if poses change len
            # pose_sequences = {design: data.get('final_sequence')[:self.pose.number_of_residues]
            #                   for design, data in all_design_scores.items()}
            pose_length = self.pose.number_of_residues
            residue_indices = list(range(1, pose_length + 1))
            pose_sequences = {}
            for design, data in list(all_design_scores.items()):
                sequence = data.get('final_sequence')
                if sequence:
                    if len(sequence) >= pose_length:
                        pose_sequences[design] = sequence[:pose_length]
                    else:
                        pose_sequences[design] = sequence
                else:
                    self.log.debug('Design %s is missing sequence data' % design)
                    all_design_scores.pop(design)
            entity_chain_breaks = [(entity, entity.n_terminal_residue.number, entity.c_terminal_residue.number)
                                   for entity in self.pose.entities]
            # # entity_sequences = {design: scores.get('final_sequence')[:self.pose.number_of_residues]
            # #                         for design, scores in all_design_scores.items()}
            entity_sequences = {entity: {design: sequence[n_term - 1:c_term]   # only include v if design is viable
                                         for design, sequence in pose_sequences.items()}  # if design in viable_designs}
                                for entity, n_term, c_term in entity_chain_breaks}
            # Find all designs which have corresponding pdb files and collect their sequences
            # for file in glob(self.designs):
            #     if file in all_design_scores:
            #      pdb = PDB.from_file(file, name=os.path.splitext(os.path.basename(file))[0], log=None, entities=False)
            #         pose_sequences[pdb.name] = pdb.atom_sequences  # {chain: sequence, ...}
            #
            # # pulling from design pdbs and reorienting with format {chain: {name: sequence, ...}, ...}
            # entity_sequences = {}
            # for design, entity_sequences in pose_sequences.items():
            #     for chain, sequence in entity_sequences.items():
            #         if chain not in entity_sequences:
            #             entity_sequences[chain] = {}
            #         entity_sequences[chain][design] = sequence

            # #  v-this-v mechanism accounts for offsets from the reference sequence which aren't necessary YET
            # sequence_mutations = \
            #     generate_multiple_mutations(self.pose.pdb.atom_sequences, pose_sequences, pose_num=False)
            # sequence_mutations.pop('reference')
            # self.log.debug('Sequence Mutations: %s' % {design: {chain: format_mutations(mutations)
            #                                                     for chain, mutations in chain_mutations.items()}
            #                                            for design, chain_mutations in sequence_mutations.items()})
            # entity_sequences = generate_sequences(self.pose.pdb.atom_sequences, sequence_mutations)
            # entity_sequences = {chain: keys_from_trajectory_number(named_sequences)
            #                         for chain, named_sequences in entity_sequences.items()}
            # entity_chain_breaks = [(entity, entity.n_terminal_residue.number, entity.c_terminal_residue.number)
            #                        for entity in self.pose.entities]
            # # # entity_sequences = {design: scores.get('final_sequence')[:self.pose.number_of_residues]
            # # #                         for design, scores in all_design_scores.items()}
            # entity_sequences = {entity.chain_id: {design: sequence[n_term - 1:c_term]
            #                                      for design, sequence in pose_sequences.items()}
            #                    for entity, n_term, c_term in entity_chain_breaks}
            # for chain in chain_break
            # self.log.debug('Design sequences by chain: %s' % entity_sequences)
            # self.log.debug('All designs with sequences: %s'
            #                % ', '.join(entity_sequences[next(iter(entity_sequences))].keys()))

            # Ensure data is present for both scores and sequences, then initialize DataFrames
            # good_designs = sorted(set(entity_sequences[next(iter(entity_sequences))].keys()).
            #                       intersection(set(all_design_scores.keys())))
            # self.log.info('All designs with both sequence and scores: %s' % ', '.join(good_designs))
            # all_design_scores = filter_dictionary_keys(all_design_scores, good_designs)
            # entity_sequences = {chain: filter_dictionary_keys(entity_sequences, good_designs)
            #                         for chain, entity_sequences in entity_sequences.items()}
            # self.log.debug('Final design sequences by chain: %s' % entity_sequences)

            idx_slice = pd.IndexSlice
            scores_df = pd.DataFrame(all_design_scores).T
            # Gather all columns into specific types for processing and formatting
            rename_columns, per_res_columns, hbonds_columns = {}, [], []
            for column in scores_df.columns.to_list():
                if column.startswith('R_'):
                    rename_columns[column] = column.replace('R_', '').replace('S_', '')
                elif column.startswith('per_res_'):
                    per_res_columns.append(column)
                elif column.startswith('hbonds_res_selection'):
                    hbonds_columns.append(column)
                elif column.startswith('symmetry_switch'):
                    other_pose_metrics['symmetry'] = \
                        scores_df.loc[:, column][0].replace('make_', '').replace('_group', '')
                else:
                    rename_columns[column] = column.replace('res_summary_', '').replace('solvation_total', 'solvation')
            rename_columns.update(columns_to_rename)

            # Rename and Format columns
            scores_df.rename(columns=rename_columns, inplace=True)
            scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
            # Check proper input
            metric_set = necessary_metrics.difference(set(scores_df.columns))
            # self.log.debug('Score columns present before required metric check: %s' % scores_df.columns.to_list())
            assert metric_set == set(), 'Missing required metrics: %s' % metric_set
            # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
            protocol_s = scores_df[groups]
            # protocol_s.replace({'combo_profile': 'design_profile'}, inplace=True)  # ensure proper profile name

            # Remove unnecessary (old scores) as well as Rosetta pose score terms besides ref (has been renamed above)
            # TODO learn know how to produce score terms in output score file. Not in FastRelax...
            remove_columns = rosetta_terms + hbonds_columns + per_res_columns + unnecessary + [groups]
            scores_df.drop(remove_columns, axis=1, inplace=True, errors='ignore')
            scores_columns = scores_df.columns.to_list()
            self.log.debug('Score columns present: %s' % scores_columns)
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

            all_mutations = \
                generate_mutations_from_reference(''.join(self.pose.pdb.atom_sequences.values()), pose_sequences)
            # all_mutations = generate_mutations_from_reference(self.pose.pdb.atom_sequences, pose_sequences)
            # all_mutations_no_chains = make_mutations_chain_agnostic(all_mutations)
            # cleaned_mutations = simplify_mutation_dict(all_mutations)
            residue_info = dirty_residue_processing(all_design_scores, simplify_mutation_dict(all_mutations),
                                                    hbonds=interface_hbonds)
            #                                       offset=offset_dict)
            # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
            # residue_info = residue_processing(all_design_scores, cleaned_mutations, per_res_columns,
            #                                   hbonds=interface_hbonds)
            #                                   offset=offset_dict)

            # Get design information including: interface residues, SSM's, and wild_type/design files
            profile_background = {}
            # design_profile = self.info.get('design_profile')
            # evolutionary_profile = self.info.get('evolutionary_profile')
            # fragment_data = self.info.get('fragment_data')
            if self.design_profile:
                # profile_background['design'] = parse_pssm(design_profile)
                profile_background['design'] = self.design_profile
            if self.evolutionary_profile:  # lost accuracy ^ and v with round operation during write
                # profile_background['evolution'] = parse_pssm(evolutionary_profile)
                profile_background['evolution'] = self.evolutionary_profile
            if self.fragment_data:
                profile_background['fragment'] = self.fragment_data
                # issm_residues = set(profile_background['fragment'].keys())
            else:
                # issm_residues = set()
                self.log.info('Design has no fragment information')
            # frag_db_ = self.info.get('fragment_database')
            # interface_bkgd = get_db_aa_frequencies(PUtils.frag_directory[frag_db_]) if frag_db_ else {}
            interface_bkgd = get_db_aa_frequencies(PUtils.frag_directory.get(self.fragment_database, {}))

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
            # reference_mutations = cleaned_mutations.pop('reference', None)  # save the reference
            scores_df['number_of_mutations'] = \
                pd.Series({design: len(mutations) for design, mutations in all_mutations.items()})
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

            # Add design residue information to scores_df such as how many core, rim, and support residues were measured
            for r_class in residue_classificiation:
                scores_df[r_class] = \
                    residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == r_class]].sum(axis=1)

            # Calculate new metrics from combinations of other metrics
            scores_df['total_interface_residues'] = len(interface_residues)
            # sum columns using tuple [0] + [1]
            summation_pairs = \
                {'buns_unbound': list(filter(re.compile('buns_[0-9]+_unbound$').match, scores_columns)),
                 'interface_energy_bound':
                     list(filter(re.compile('interface_energy_[0-9]+_bound').match, scores_columns)),
                 'interface_energy_unbound':
                     list(filter(re.compile('interface_energy_[0-9]+_unbound').match, scores_columns)),
                 'sasa_hydrophobic_bound':
                     list(filter(re.compile('sasa_hydrophobic_[0-9]+_bound').match, scores_columns)),
                 'sasa_polar_bound': list(filter(re.compile('sasa_polar_[0-9]+_bound').match, scores_columns)),
                 'sasa_total_bound': list(filter(re.compile('sasa_total_[0-9]+_bound').match, scores_columns)),
                 'solvation_energy_bound':
                     list(filter(re.compile('solvation_energy_[0-9]+_bound').match, scores_columns)),
                 'solvation_energy_unbound':
                     list(filter(re.compile('solvation_energy_[0-9]+_unbound').match, scores_columns)),
                 'interface_connectivity':
                     list(filter(re.compile('interface_connectivity_[0-9]+').match, scores_columns))
                 # 'buns_hpol_total': ('buns_asu_hpol', 'buns_nano_hpol'),
                 # 'buns_heavy_total': ('buns_asu', 'buns_nano'),
                 }
            scores_df = columns_to_new_column(scores_df, summation_pairs)
            scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
            scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')
            scores_df['interface_composition_similarity'] = \
                scores_df.apply(interface_residue_composition_similarity, axis=1)
            # dropping 'total_interface_residues' after calculation as it is in other_pose_metrics
            scores_df.drop(clean_up_intermediate_columns + ['total_interface_residues'], axis=1, inplace=True,
                           errors='ignore')
            if scores_df.get('repacking') is not None:
                # set interface_bound_activation_energy = NaN where repacking is 0
                # Currently is -1 for True (Rosetta Filter quirk...)
                scores_df.loc[scores_df[scores_df['repacking'] == 0].index, 'interface_bound_activation_energy'] = \
                    np.nan
                scores_df.drop('repacking', axis=1, inplace=True)
            # Process dataframes for missing values and drop refine trajectory if present
            scores_df[groups] = protocol_s
            refine_index = scores_df[scores_df[groups] == PUtils.stage[1]].index
            scores_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
            residue_df.drop(refine_index, axis=0, inplace=True, errors='ignore')
            residue_info.pop(PUtils.stage[1], None)  # Remove refine from analysis
            # residues_no_frags = residue_df.columns[residue_df.isna().all(axis=0)].remove_unused_levels().levels[0]
            residue_df.dropna(how='all', inplace=True, axis=1)  # remove completely empty columns such as obs_interface
            residue_df.fillna(0., inplace=True)
            # residue_indices_no_frags = residue_df.columns[residue_df.isna().all(axis=0)]
            # scores_na_index = scores_df.index[scores_df.isna().any(axis=1)]  # scores_df.where()
            # residue_na_index = residue_df.index[residue_df.isna().any(axis=1)]
            # drop_na_index = np.union1d(scores_na_index, residue_na_index)
            # if drop_na_index.any():
            #     self.log.debug('Missing information in score columns: %s'
            #                    % scores_df.columns[scores_df.isna().any(axis=0)].tolist())
            #     self.log.debug('Missing information in residue columns: %s'
            #                    % residue_df.columns[residue_df.isna().any(axis=0)].tolist())
            #     protocol_s.drop(drop_na_index, inplace=True, errors='ignore')
            #     scores_df.drop(drop_na_index, inplace=True, errors='ignore')
            #     residue_df.drop(drop_na_index, inplace=True, errors='ignore')
            #     for idx in drop_na_index:
            #         residue_info.pop(idx, None)
            #     self.log.warning('Dropped designs from analysis due to missing values: %s' % ', '.join(scores_na_index))
            #     # might have to remove these from all_design_scores in the case that that is used as a dictionary again
            viable_designs = scores_df.index.to_list()
            self.log.debug('Viable designs remaining after cleaning:\n\t%s' % ', '.join(viable_designs))
            other_pose_metrics['observations'] = len(scores_df)
            pose_sequences = filter_dictionary_keys(pose_sequences, viable_designs)
            # entity_alignment = multi_chain_alignment(entity_sequences)
            pose_alignment = msa_from_dictionary(pose_sequences)
            entity_alignments = {entity: msa_from_dictionary(design_sequences)
                                 for entity, design_sequences in entity_sequences.items()}

            atomic_deviation, per_residue_data = {}, {}
            # design_assemblies = []  # maybe use?
            per_residue_data['errat_deviation'] = {}
            for file in self.get_designs():
                decoy_name = os.path.splitext(os.path.basename(file))[0]  # should match scored designs...
                if decoy_name not in scores_df.index:
                    continue
                design_asu = PDB.from_file(file, name=decoy_name, log=self.log, entities=False)  # , lazy=True)
                # atomic_deviation[pdb.name] = pdb.errat(out_path=self.data)
                assembly = SymmetricModel.from_asu(design_asu, sym_entry=self.sym_entry, log=self.log).assembly
                #                                            ,symmetry=self.design_symmetry)
                atomic_deviation[design_asu.name], per_residue_errat = assembly.errat()
                per_residue_data['errat_deviation'][design_asu.name] = per_residue_errat[:design_asu.number_of_residues]
            scores_df['errat_accuracy'] = pd.Series(atomic_deviation)

            # Calculate hydrophobic collapse for each design

            # # design_collapse_z_score = {entity: {design: hydrophobic_collapse_index(sequence)
            # #                          for design, sequence in sequences.items()}
            # #                 for entity, sequences in entity_sequences.items()}
            # # design_collapse_z_score = {entity: {design: z_score(hydrophobic_collapse_index(seq),
            # #                                                  collapse['mean'], collapse['std'])
            # #                                  for design, seq in entity_sequences[entity].items()}
            # #                         for entity, collapse in collapse_df.items()}
            # design_collapse_z_score = \
            #     {design: {entity: z_score(hydrophobic_collapse_index(entity_sequences[entity][design]),
            #                               collapse_df[entity]['mean'], collapse_df[entity]['std'])
            #               for entity in self.pose.entities} for design in viable_designs}
            #
            # # Measure the wild type entity versus the modified entity to find the hci delta
            # # Todo if no design, can't measure the wild-type after the normalization...
            # # delta_sequence_hci = {entity: {design: collapse_z - wt_collapse_z_score[entity]
            # #                                for design, collapse_z in design_collapse_z_score[entity].items()}
            # #                       for entity in self.pose.entities}
            #
            # delta_sequence_hci = {design: {entity: design_collapse_z_score[design][entity] - wt_collapse_z_score[entity]
            #                                for entity in self.pose.entities}
            #                       for design in viable_designs}

            # global_hydrophobicity = {entity: {design: np.where(normalized_collapse_z > 0, normalized_collapse_z, 0).sum()
            #                                   for design, normalized_collapse_z in delta_sequence_hci[entity].items()}
            #                          for entity in self.pose.entities}

            #
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

            collapse_df, wt_errat, wt_collapse, wt_collapse_bool, wt_collapse_z_score = {}, {}, {}, {}, {}
            inverse_residue_contact_order_z, contact_order = {}, {}
            for entity in self.pose.entities:
                entity.msa = self.database.alignments.retrieve_data(name=entity.name)
                # Todo reinstate
                # entity.h_fields = self.database.bmdca_fields.retrieve_data(name=entity.name)
                # entity.j_couplings = self.database.bmdca_couplings.retrieve_data(name=entity.name)
                collapse = entity.collapse_profile()
                collapse_df[entity] = collapse
                # wt_collapse_z_score[entity] = hydrophobic_collapse_index(entity.sequence)
                wt_collapse[entity] = hydrophobic_collapse_index(entity.sequence)
                wt_collapse_bool[entity] = np.where(wt_collapse[entity] > 0.43, 1, 0)  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
                wt_collapse_z_score[entity] = \
                    z_score(wt_collapse[entity], collapse.loc['mean', :], collapse.loc['std', :])
                # we must give a copy of coords_indexed_residues from the pose to each entity...
                entity.coords_indexed_residues = self.pose.pdb._coords_residue_index

                _, wt_errat[entity] = entity.errat()
                # residue_contact_order[entity] = entity.contact_order_per_residue()
                # we need to get the contact order from the symmetric version...
                entity_oligomer = PDB.from_chains(entity.oligomer, log=self.log, entities=False)
                residue_contact_order = entity_oligomer.contact_order_per_residue()[:entity.number_of_residues]
                # residue_contact_order = entity.contact_order_per_residue()
                contact_order[entity] = residue_contact_order
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
            # for graphing the collapse profile
            wt_contact_order_concatenated_s = pd.Series(np.concatenate(list(contact_order.values())), name='contact_order')
            wt_collapse_concatenated_s = pd.Series(np.concatenate(list(wt_collapse.values())), name='wild_type')
            profile_mean_collapse_concatenated_s = \
                pd.concat([collapse_df[entity].loc['mean', :] for entity in self.pose.entities], ignore_index=True)
            profile_mean_collapse_concatenated_s.index += 1
            profile_std_collapse_concatenated_s = \
                pd.concat([collapse_df[entity].loc['std', :] for entity in self.pose.entities], ignore_index=True)
            profile_std_collapse_concatenated_s.index += 1

            folding_and_collapse = \
                {'hydrophobicity_deviation_magnitude': {}, 'new_collapse_islands': {},
                 'new_collapse_island_significance': {}, 'contact_order_collapse_z_sum': {},
                 'sequential_collapse_peaks_z_sum': {}, 'sequential_collapse_z_sum': {}, 'global_collapse_z_sum': {}}
            design_collapse_graph = {}
            per_residue_data['hydrophobic_collapse'] = {}
            for design in viable_designs:
                hydrophobicity_deviation_magnitude, new_collapse_islands, new_collapse_island_significance = [], [], []
                contact_order_collapse_z_sum, sequential_collapse_peaks_z_sum, sequential_collapse_z_sum, global_collapse_z_sum = [], [], [], []
                collapse_concatenated = []
                for entity in self.pose.entities:
                    sequence = entity_sequences[entity][design]
                    standardized_collapse = hydrophobic_collapse_index(sequence)
                    collapse_concatenated.append(standardized_collapse)
                    # Todo -> observed_collapse, standardized_collapse = hydrophobic_collapse_index(sequence)
                    # normalized_collapse = standardized_collapse - wt_collapse[entity]
                    z_array = z_score(standardized_collapse,  # observed_collapse,
                                      collapse_df[entity].loc['mean', :], collapse_df[entity].loc['std', :])
                    # todo test for magnitude of the wt versus profile, remove subtraction?
                    normalized_collapse_z = z_array - wt_collapse_z_score[entity]
                    hydrophobicity_deviation_magnitude.append(sum(abs(normalized_collapse_z)))
                    global_collapse_z = np.where(normalized_collapse_z > 0, normalized_collapse_z, 0)

                    # find collapse where: delta above standard collapse, collapsable boolean, and successive number
                    # collapse_propensity = np.where(standardized_collapse > 0.43, standardized_collapse - 0.43, 0)
                    # scale the collapse propensity by the standard collapse threshold and make z score
                    collapse_propensity_z = z_score(standardized_collapse, 0.43, 0.05)
                    collapse_propensity_positive_z_only = np.where(collapse_propensity_z > 0, collapse_propensity_z, 0)
                    # ^ [0, 0, 0, 0, 0.04, 0.06, 0, 0, 0.1, 0.07, ...]
                    # collapse_bool = np.where(standardized_collapse > 0.43, 1, 0)  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
                    collapse_bool = np.where(collapse_propensity_positive_z_only, 1, 0)  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
                    increased_collapse = np.where(collapse_bool - wt_collapse_bool[entity] == 1, 1, 0)
                    # check if the increased collapse has made new collapse
                    new_collapse = np.zeros(collapse_bool.shape)  # [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ...]
                    for idx, _bool in enumerate(increased_collapse.tolist()[1:], 1):
                        if _bool and (not wt_collapse_bool[entity][idx - 1] or not wt_collapse_bool[entity][idx + 1]):
                            new_collapse[idx] = _bool
                    # new_collapse are sites where a new collapse is formed compared to wild-type

                    # # we must give a copy of coords_indexed_residues from the pose to each entity...
                    # entity.coords_indexed_residues = self.pose.pdb._coords_residue_index
                    # residue_contact_order = entity.contact_order_per_residue()
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
                    # linearly weight residue by sequence position (early > late) with the halfway position (midpoint)
                    # weighted at 1
                    midpoint = 0.5
                    scale = 1 / midpoint
                    # v [0, 0, 0, 0, 2, 2, 0, 0, 1.8, 1.8, ...]
                    sequential_collapse_weights = scale * ((1 - (step * sequential_collapse_points)) + add_step_array)
                    # v [2, 1.98, 1.96, 1.94, 1.92, ...]
                    sequential_weights = \
                        scale * (1 - (np.arange(entity.number_of_residues) / entity.number_of_residues))

                    new_collapse_islands.append(new_collapse_peak_start.sum())
                    new_collapse_island_significance.append(sum(new_collapse_peak_start * abs(collapse_significance)))

                    # offset inverse_residue_contact_order_z to center at 1 instead of 0. Todo deal with negatives
                    contact_order_collapse_z_sum.append(sum((inverse_residue_contact_order_z[entity] + 1) * global_collapse_z))
                    sequential_collapse_peaks_z_sum.append(sum(sequential_collapse_weights * global_collapse_z))
                    sequential_collapse_z_sum.append(sum(sequential_weights * global_collapse_z))
                    global_collapse_z_sum.append(global_collapse_z.sum())

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
                design_collapse_graph[design] = collapse_concatenated
                # design_collapse_graph[design].name = design

            # turn the per_residue data into a dataframe matching orientation of residue_df
            per_residue_df = \
                pd.concat({measure: pd.DataFrame(data, index=residue_indices)
                           for measure, data in per_residue_data.items()}).T.swaplevel(0, 1, axis=1)
            # make a graph of the collapse with residues as index and design as column
            collapse_graph_df = pd.DataFrame(design_collapse_graph)
            collapse_graph_df.index += 1  # offset index to residue numbering
            collapse_graph_df['wild_type'] = wt_collapse_concatenated_s
            # collapse_graph_df['profile_mean'] = profile_mean_collapse_concatenated_s
            # collapse_graph_df['profile_std'] = profile_std_collapse_concatenated_s
            # collapse_graph_df['contact_order'] = wt_contact_order_concatenated_s
            pose_collapse_df = pd.DataFrame(folding_and_collapse)
            # pose_collapse_ = pd.concat(pd.DataFrame(folding_and_collapse), axis=1, keys=[('sequence_design', 'pose')])
            dca_design_residues_concat = []
            dca_succeed = True
            # dca_bkgd_energies, dca_design_energies = [], []
            dca_bkgd_energies, dca_design_energies = {}, {}
            for entity in self.pose.entities:
                try:
                    dca_background_energies = entity.direct_coupling_analysis()
                    dca_design_energies = entity.direct_coupling_analysis(msa=entity_alignments[entity])
                    dca_design_residues_concat.append(dca_design_energies)
                    # dca_background_energies.append(dca_background_energies.sum(axis=1))
                    # dca_design_energies.append(dca_design_energies.sum(axis=1))
                    dca_background_energies[entity] = dca_background_energies.sum(axis=1)  # turns data to 1D
                    dca_design_energies[entity] = dca_design_energies.sum(axis=1)
                except AttributeError:
                    self.log.error('No DCA analysis could be performed, missing required parameters files')
                    # TODO add these to the analysis
                    dca_succeed = False

            if dca_succeed:
                # concatenate along columns, adding residue index to column, design name to row
                dca_concatenated_df = pd.DataFrame(np.concatenate(dca_design_residues_concat, axis=1),
                                                   index=list(entity_sequences[entity].keys()), columns=residue_indices)
                dca_concatenated_df = pd.concat([dca_concatenated_df], keys=['dca_energy']).swaplevel(0, 1, axis=1)
                # merge with per_residue_df
                per_residue_df = pd.merge(per_residue_df, dca_concatenated_df, left_index=True, right_index=True)

            residue_df = pd.merge(residue_df, per_residue_df.loc[:, idx_slice[residue_df.columns.levels[0], :]],
                                  left_index=True, right_index=True)
            residue_indices_no_frags = residue_df.columns[residue_df.isna().all(axis=0)]

            # POSE ANALYSIS
            # cst_weights are very large and destroy the mean. remove v'drop' if consensus is run multiple times
            trajectory_df = scores_df.sort_index().drop(PUtils.stage[5], axis=0, errors='ignore')
            # add all docking and pose information to each trajectory
            pose_metrics_df = pd.concat([pd.Series(other_pose_metrics)] * len(trajectory_df), axis=1).T
            pose_metrics_df.rename(index=dict(zip(range(len(trajectory_df)), trajectory_df.index)), inplace=True)
            trajectory_df = pd.concat([pose_metrics_df, trajectory_df, pose_collapse_df], axis=1)
            # TODO v what about when run on consensus only?
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
            protocol_stats_s = pd.concat([stat_df.T.unstack() for stat_df in protocol_stats], keys=stats_metrics)
            pose_stats_s = pd.concat(pose_stats, keys=list(zip(stats_metrics, repeat('pose'))))
            stat_s = pd.concat([protocol_stats_s.dropna(), pose_stats_s.dropna()])  # dropna removes NaN metrics
            # change statistic names for all df that are not groupby means for the output trajectory.csv
            for idx, stat in enumerate(stats_metrics):
                if stat != 'mean':
                    protocol_stats[idx].index = protocol_stats[idx].index.to_series().map(
                        {protocol: '%s_%s' % (protocol, stat) for protocol in unique_protocols})
            trajectory_df = pd.concat([trajectory_df, pd.concat(pose_stats, axis=1).T] + protocol_stats)
            # this concat puts back consensus design to trajectory_df since protocol_stats is calculated on scores_df

            # Calculate sequence statistics
            # first for entire pose
            mutation_frequencies = filter_dictionary_keys(pose_alignment.frequencies, interface_residues)
            # mutation_frequencies = filter_dictionary_keys(pose_alignment['frequencies'], interface_residues)
            # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
            #                                              both mut_freq and profile_background[profile] are one-indexed
            divergence = {'divergence_%s' % profile: position_specific_jsd(mutation_frequencies, background)
                          for profile, background in profile_background.items()}
            if interface_bkgd:
                divergence['divergence_interface'] = jensen_shannon_divergence(mutation_frequencies, interface_bkgd)
            # Get pose sequence divergence
            divergence_stats = {'%s_per_residue' % divergence_type: per_res_metric(stat)
                                for divergence_type, stat in divergence.items()}

            # Next, for each protocol
            divergence_by_protocol = {protocol: {} for protocol in designs_by_protocol}
            for protocol, designs in designs_by_protocol.items():
                # Todo select from pose_alignment the indices of each design then pass to MultipleSequenceAlignment?
                protocol_alignment = msa_from_dictionary({design: pose_sequences[design] for design in designs})
                # protocol_alignment = multi_chain_alignment({entity: {design: design_seqs[design] for design in designs}
                #                                             for entity, design_seqs in entity_sequences.items()})
                # protocol_mutation_freq = filter_dictionary_keys(protocol_alignment['frequencies'], interface_residues)
                protocol_mutation_freq = filter_dictionary_keys(protocol_alignment.frequencies, interface_residues)
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
                # pose_res_dict['hydrophobic_collapse_index'] = hydrophobic_collapse_index()  # TODO HCI

            protocol_divergence_s = pd.concat([pd.Series(divergence) for divergence in divergence_by_protocol.values()],
                                              keys=list(zip(repeat('sequence_design'), divergence_by_protocol)))
            pose_divergence_s = pd.concat([pd.Series(divergence_stats)], keys=[('sequence_design', 'pose')])
            divergence_s = pd.concat([protocol_divergence_s, pose_divergence_s])
            # Calculate protocol significance
            pvalue_df = pd.DataFrame()
            scout_protocols = filter(re.compile('.*scout').match, protocol_s.index.to_list())  # list()
            similarity_protocols = set(unique_protocols).difference(scout_protocols)
            if background_protocol not in unique_protocols:
                self.log.warning('Missing background protocol \'%s\'. No protocol significance measurements available '
                                 'for this design' % background_protocol)
            elif len(similarity_protocols) == 1:  # measure significance
                self.log.warning('Can\'t measure protocol significance, only one protocol of interest')
            # missing_protocols = protocols_of_interest.difference(unique_protocols)
            # if missing_protocols:
            #     self.log.warning('Missing protocol%s \'%s\'. No protocol significance measurements for this design!'
            #                      % ('s' if len(missing_protocols) > 1 else '', ', '.join(missing_protocols)))
            # elif len(protocols_of_interest) == 1:
            else:  # Test significance between all combinations of protocols by grabbing mean entries per protocol
                # for prot1, prot2 in combinations(sorted(protocols_of_interest), 2):
                for prot1, prot2 in combinations(sorted(similarity_protocols), 2):
                    select_df = \
                        trajectory_df.loc[designs_by_protocol[prot1] + designs_by_protocol[prot2], significance_columns]
                    difference_s = trajectory_df.loc[prot1, :].sub(trajectory_df.loc[prot2, :])
                    pvalue_df[(prot1, prot2)] = df_permutation_test(select_df, difference_s, compare='mean',
                                                                    group1_size=len(designs_by_protocol[prot1]))
                pvalue_df = pvalue_df.T  # transpose significance pairs to indices and significance metrics to columns
                trajectory_df = pd.concat([trajectory_df, pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1)])

                # Compute residue energy/sequence differences between each protocol
                residue_energy_df = \
                    residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == 'energy_delta']]

                scaler = StandardScaler()
                res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
                residue_energy_np = scaler.fit_transform(residue_energy_df.values)
                residue_energy_pc = res_pca.fit_transform(residue_energy_np)

                seq_pca = PCA(PUtils.variance)
                designed_residue_info = {design: {residue: info for residue, info in residues_info.items()
                                                  if residue in design_residues}
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
                sequence_groups = seq_pc_df.groupby(groups)
                residue_energy_groups = residue_energy_pc_df.groupby(groups)
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
                        #     {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
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

                if figures:
                    # Make a hydrophobic collapse figure
                    # collapse_graph_df['profile_mean'] = profile_mean_collapse_concatenated_s
                    # collapse_graph_df['profile_std'] = profile_std_collapse_concatenated_s
                    # collapse_graph_df['contact_order'] = wt_contact_order_concatenated_s
                    collapse_graph_describe = {
                        'std_min': profile_mean_collapse_concatenated_s - profile_std_collapse_concatenated_s,
                        'std_max': profile_mean_collapse_concatenated_s + profile_std_collapse_concatenated_s,
                        'contact_order': (wt_contact_order_concatenated_s - wt_contact_order_concatenated_s.min()) /
                                         (wt_contact_order_concatenated_s.max() - wt_contact_order_concatenated_s.min()),
                        # 'Residue Number': collapse_graph_df.index}
                    }
                    collapse_graph_describe_df = pd.DataFrame(collapse_graph_describe)
                    collapse_graph_describe_df.index += 1
                    collapse_graph_describe_df['Residue Number'] = collapse_graph_describe_df.index
                    # g = sns.FacetGrid(tip_sumstats, col="sex", row="smoker")
                    # collapse_graph_df['Residue Number'] = collapse_graph_df.index
                    # graph = sns.lineplot(data=collapse_graph_df)
                    graph = sns.relplot(data=collapse_graph_df, kind='line')  # x='Residue Number'
                    # ax = graph.axes
                    # ax[0, 0].vlines('Residue Number', 'std_min', 'std_max', data=collapse_graph_describe_df)
                    graph.savefig(os.path.join(self.data, 'hydrophobic_collapse.png'))
                    ax = plt.gca()
                    ax.vlines('Residue Number', 'std_min', 'std_max', data=collapse_graph_describe_df)
                    # axes[0].plot(collapse_graph_df.index, profile_mean_collapse_concatenated_s)
                    graph.set_xlabels('Residue Number')
                    # def errplot(x, y, yerr, **kwargs):
                    #     ax = plt.gca()
                    #     data = kwargs.pop("data")
                    #     data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
                    # graph.map_dataframe(errplot, 'Residue Number', 'mean', 'std')
                    graph.savefig(os.path.join(self.data, 'hydrophobic_collapse+contact.png'))
                    # Todo ensure output is as expected
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
                # Add wild-type residue information in metrics for sequence comparison
                # find the solvent acessible surface area of the separated entities
                for entity in self.pose.pdb.entities:
                    entity.get_sasa()

                # todo simplify this mess...
                errat_collapse_df = \
                    pd.concat([pd.concat({'errat_deviation':
                              pd.Series(np.concatenate(list(wt_errat.values())), index=residue_indices),
                               'hydrophobic_collapse': pd.Series(np.concatenate(list(wt_collapse.values())),
                                                                 index=residue_indices)}
                                         )], keys=['wild_type']).unstack().unstack()  # .swaplevel(0, 1, axis=1)
                # print(errat_collapse_df)
                wild_type_residue_info = {}
                for res_number in residue_info[next(iter(residue_info))].keys():
                    # bsa_total is actually a sasa, but for formatting sake, I've called it a bsa...
                    residue = self.pose.pdb.residue(res_number)
                    wild_type_residue_info[res_number] = \
                        {'type': protein_letters_3to1.get(residue.type.title()), 'core': None, 'rim': None, 'support': None,
                         # Todo implement wt energy metric during oligomer refinement?
                         'interior': 0, 'hbond': None, 'energy_delta': None,
                         'bsa_total': residue.sasa, 'bsa_polar': None, 'bsa_hydrophobic': None,
                         'coordinate_constraint': None, 'residue_favored': None, 'observed_design': None,
                         'observed_evolution': None, 'observed_fragment': None}  # 'hot_spot': None}
                    if residue.relative_sasa < 0.25:
                        wild_type_residue_info[res_number]['interior'] = 1
                    # if res_number in issm_residues and res_number not in residues_no_frags:
                    #     wild_type_residue_info[res_number]['observed_fragment'] = None

                wt_df = pd.concat([pd.DataFrame(wild_type_residue_info)], keys=['wild_type']).unstack()
                wt_df = pd.merge(wt_df, errat_collapse_df.loc[:, idx_slice[wt_df.columns.levels[0], :]],
                                 left_index=True, right_index=True)
                wt_df.drop(residue_indices_no_frags, inplace=True, axis=1)
                # only sort once as residues are in same order
                # wt_df.sort_index(level=0, inplace=True, axis=1, sort_remaining=False)
                # residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                residue_df = pd.concat([wt_df, residue_df], sort=False)
                # residue_df.drop(residue_indices_no_frags, inplace=True, axis=1)
                residue_df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
                residue_df[(groups, groups)] = protocol_s
                # residue_df.sort_index(inplace=True, key=lambda x: x.str.isdigit())  # put wt entry first
                if merge_residue_data:
                    trajectory_df = pd.concat([trajectory_df], axis=1, keys=['metrics'])
                    trajectory_df = pd.merge(trajectory_df, residue_df, left_index=True, right_index=True)
                else:
                    residue_df.to_csv(self.residues)
                trajectory_df.to_csv(self.trajectories)
                self.design_sequences = pickle_object(entity_sequences, self.design_sequences, out_path='')

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
    def select_sequences(self, filters=None, weights=None, number=1, protocol=None):
        """Select sequences for further characterization. If weights, then user can prioritize by metrics, otherwise
        sequence with the most neighbors as calculated by sequence distance will be selected. If there is a tie, the
        sequence with the lowest weight will be selected

        Keyword Args:
            filters=None (Iterable): The filters to use in sequence selection
            weights=None (Iterable): The weights to use in sequence selection
            number=1 (int): The number of sequences to consider for each design
            protocol=None (str): Whether a particular design protocol should be chosen
        Returns:
            (list[tuple[DesignDirectory, str]]): Containing the selected sequences found
        """
        # Load relevant data from the design directory
        trajectory_df = pd.read_csv(self.trajectories, index_col=0, header=[0])
        trajectory_df.dropna(inplace=True)
        if protocol:
            designs = trajectory_df[trajectory_df['protocol'] == protocol].index.to_list()
            if not designs:
                raise DesignError('No designs found for protocol %s!' % protocol)
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
            design_list = rank_dataframe_by_metric_weights(df, weights=weights).index.to_list()
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
        """Make a path if it doesn't exist yet"""
        if not os.path.exists(path) and condition:
            os.makedirs(path)

    def __str__(self):
        if self.nano:
            return self.source_path.replace(self.nanohedra_root + os.sep, '').replace(os.sep, '-')
        else:
            # TODO integrate with designDB?
            return self.path.replace(self.projects + os.sep, '').replace(os.sep, '-')


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
