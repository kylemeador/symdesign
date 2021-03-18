import copy
import math
import os
import shutil
import subprocess
from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import PathUtils as PUtils
from CmdUtils import reference_average_residue_weight, script_cmd, run_cmds, flag_options, rosetta_flags
from Query import Flags
from SymDesignUtils import unpickle, start_log, null_log, handle_errors_f, sdf_lookup, write_shell_script, DesignError,\
    match_score_from_z_value, handle_design_errors, pickle_object, remove_interior_keys, clean_dictionary, all_vs_all, \
    condensed_to_square
from PDB import PDB
from Pose import Pose
from AnalyzeMutatedSequences import generate_all_design_mutations, generate_sequences, multi_chain_alignment, \
    compute_jsd
from AnalyzeOutput import columns_to_remove, columns_to_rename, read_scores, remove_pdb_prefixes, join_columns, groups, \
    necessary_metrics, columns_to_new_column, delta_pairs, summation_pairs, unnecessary, rosetta_terms, \
    dirty_hbond_processing, dirty_residue_processing, mutation_conserved, per_res_metric, residue_classificiation, \
    residue_composition_diff, division_pairs, stats_metrics, protocol_specific_columns, protocols_of_interest, \
    df_permutation_test, remove_score_columns
from SequenceProfile import calculate_match_metrics, return_fragment_interface_metrics, parse_pssm, \
    get_db_aa_frequencies, simplify_mutation_dict, make_mutations_chain_agnostic, weave_sequence_dict, pos_specific_jsd, \
    remove_non_mutations, sequence_difference
from classes.SymEntry import SymEntry
from interface_analysis.Database import FragmentDatabase


# Globals
logger = start_log(name=__name__)
index_offset = 1
design_directory_modes = ['design', 'dock', 'filter']


class DesignDirectory:  # Todo move PDB coordinate information to Pose. Only use to handle Pose paths/options

    def __init__(self, design_path, nano=False, directory_type='design', pose_id=None, root=None, debug=False,
                 **kwargs):  # project=None,
        if pose_id:  # Todo may not be compatible P432
            self.program_root = root
            self.directory_string_to_path(pose_id)
            design_path = self.path
        self.name = os.path.splitext(os.path.basename(design_path))[0]  # works for all cases
        self.log = None
        self.nano = nano
        self.directory_type = directory_type

        self.project_designs = None
        self.protein_data = None
        # design_symmetry/data (P432/Data)
        self.pdbs = None
        # design_symmetry/data/pdbs (P432/Data/PDBs)
        self.sequences = None
        # design_symmetry/sequences (P432/Sequence_Info)
        # design_symmetry/data/sequences (P432/Data/Sequence_Info)
        self.all_scores = None
        # design_symmetry/all_scores (P432/All_Scores)
        self.trajectories = None
        # design_symmetry/all_scores/str(self)_Trajectories.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Trajectories.csv)
        self.residues = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Residues.csv)
        self.design_sequences = None
        # design_symmetry/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Sequences.pkl)

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
        self.info_pickle = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data/stats.pkl
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.info = {}

        self.pose = None  # contains the design's Pose object
        self.source = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/asu.pdb (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/asu.pdb)
        self.asu = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu.pdb
        self.assembly = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/assembly.pdb
        self.refine_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_refine.pdb
        self.refined_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs/clean_asu_for_refine.pdb
        self.consensus = None  # Whether to run consensus or not
        self.consensus_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_consensus.pdb
        self.consensus_design_pdb = None
        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs/clean_asu_for_consensus.pdb

        self.sdf = None
        # path/to/directory/sdf/
        self.sdfs = {}
        self.oligomer_names = []
        self.oligomers = []

        self.sym_entry_number = None
        self.design_symmetry = None
        self.design_dim = None
        self.uc_dimensions = None
        self.expand_matrices = None

        # self.fragment_cluster_residue_d = {}
        self.fragment_observations = []
        self.interface_residue_d = {}

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
        self.percent_residues_fragment_all = None  # TODO MOVE Metrics
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
        self.ignore_clashes = False
        # Analysis flags
        self.analysis = False
        self.skip_logging = False
        self.set_flags(**kwargs)

        # if not self.nano:
        # check to be sure it's not actually one
        #     if ('DEGEN', 'ROT', 'tx') in self.path:
        #         self.nano = True
        if self.nano:
            self.path = design_path
            # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
            if not os.path.exists(self.path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.path)
            # v used in dock_dir set up
            self.building_block_logs = []
            self.building_block_dirs = []

            self.cannonical_pdb1 = None  # cannonical pdb orientation
            self.cannonical_pdb2 = None
            self.pdb_dir1_path = None
            self.pdb_dir2_path = None
            self.master_outdir = None  # same as self.program_root
            self.oligomer_symmetry_1 = None
            self.oligomer_symmetry_2 = None
            self.design_symmetry_pg = None
            self.internal_rot1 = None
            self.internal_rot2 = None
            self.rot_range_deg_pdb1 = None
            self.rot_range_deg_pdb2 = None
            self.rot_step_deg1 = None
            self.rot_step_deg2 = None
            self.internal_zshift1 = None
            self.internal_zshift2 = None
            self.ref_frame_tx_dof1 = None
            self.ref_frame_tx_dof2 = None
            self.set_mat1 = None
            self.set_mat2 = None
            self.uc_spec_string = None
            self.degen1 = None
            self.degen2 = None
            self.cryst_record = None
            self.pose_id = None

            # self.fragment_cluster_freq_d = {}
            self.transform_d = {}  # dict[pdb# (1, 2)] = {'transform_type': matrix/vector}

            if self.directory_type == 'dock':
                # Saves the path of the docking directory as DesignDirectory.path attribute. Try to populate further
                # using typical directory structuring
                # self.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implementation?
                self.program_root = self.path  # Assuming that the output directory (^ or v) of Nanohedra passed as the path
                # v for design_recap
                # self.program_root = glob(os.path.join(self.path, 'NanohedraEntry*DockedPoses%s' % str(program_root or '')))
                self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
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
            else:  # if self.directory_type in ['design', 'filter']:
                # May have issues with the number of open log files
                # if self.directory_type == 'filter':
                #     self.skip_logging = True
                # else:
                #     self.skip_logging = True
                self.program_root = self.path[:self.path.find(self.path.split(os.sep)[-4]) - 1]
                # design_symmetry (P432)
                self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
                self.building_blocks = self.path[:self.path.find(self.path.split(os.sep)[-3]) - 1]
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                self.source = os.path.join(self.path, PUtils.asu)
                self.set_up_design_directory()
            # else:
            #     raise DesignError('%s: %s is not an available directory_type. Choose from %s...\n'
            #                       % (DesignDirectory.__name__, self.directory_type, ','.join(design_directory_modes)))

            if not os.path.exists(self.nano_master_log):
                raise DesignError('%s: No %s found for this directory! Cannot perform material design without it.\n'
                                  'Ensure you have the file \'%s\' located properly before trying this Design!'
                                  % (self.__str__(), PUtils.master_log, self.nano_master_log))
            self.gather_docking_metrics()
            self.sym_entry = SymEntry(self.sym_entry_number)
            self.design_symmetry = self.sym_entry.get_result_design_sym()

        else:
            if '.pdb' in design_path:  # set up /program_root/projects/project/design
                self.program_root = os.path.join(os.getcwd(), PUtils.program_output)  # symmetry.rstrip(os.sep)
                self.projects = os.path.join(self.program_root, PUtils.projects)
                self.project_designs = os.path.join(self.projects, '%s_%s' % (design_path.split(os.sep)[-2],
                                                                              PUtils.design_directory))
                self.path = os.path.join(self.project_designs, self.name)
                # ^ /program_root/projects/project/design<- self.path /design.pdb
                if not os.path.exists(self.program_root):
                    os.makedirs(self.program_root)
                if not os.path.exists(self.projects):
                    os.makedirs(self.projects)
                if not os.path.exists(self.project_designs):
                    os.makedirs(self.project_designs)
                if not os.path.exists(self.path):
                    os.makedirs(self.path)

                self.source = design_path
                shutil.copy(design_path, self.path)
            else:  # initialize DesignDirectory to recognize existing /program_root/projects/project/design
                self.path = design_path
                self.asu = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean))
                if os.path.exists(self.asu):
                    self.source = self.asu
                else:
                    try:
                        self.source = glob(os.path.join(self.path, '%s.pdb' % self.name))[0]
                    except IndexError:
                        self.source = None
                self.program_root = '/%s' % os.path.join(*self.path.split(os.sep)[:-3])  # symmetry.rstrip(os.sep)
                self.projects = '/%s' % os.path.join(*self.path.split(os.sep)[:-2])
                self.project_designs = '/%s' % os.path.join(*self.path.split(os.sep)[:-1])

            self.set_up_design_directory()
        self.start_log(debug=debug)
        # self.log.debug('fragment_observations: %s' % self.fragment_observations)

    @classmethod
    def from_nanohedra(cls, design_path, mode=None, project=None, nano=True, **kwargs):
        return cls(design_path, nano=nano, mode=mode, project=project, **kwargs)

    @classmethod
    def from_file(cls, design_path, project=None, **kwargs):  # directory_type=None
        return cls(design_path, project=project, **kwargs)

    @classmethod
    def from_pose_id(cls, pose_id=None, root=None, **kwargs):  # directory_type=None
        return cls(None, pose_id=pose_id, root=root, **kwargs)

    @property
    def number_of_fragments(self):
        return len(self.fragment_observations)

    @property
    def score(self):
        try:
            if self.center_residue_score and self.central_residues_with_fragment_overlap:
                return self.center_residue_score / self.central_residues_with_fragment_overlap
            else:
                self.get_fragment_metrics()
                if self.center_residue_score is not None and self.central_residues_with_fragment_overlap is not None:
                    return self.center_residue_score / self.central_residues_with_fragment_overlap
                else:
                    return None
        except ZeroDivisionError:
            self.log.error('No fragment information available! Design cannot be scored.')
            return 0.0

    def pose_score(self):  # Todo merge with above
        """Returns:
            (float): The Nanohedra score as reported in Laniado, Meador, Yeates 2021
        """
        return self.all_residue_score

    def pose_metrics(self):
        """Gather all metrics relating to the Pose and the interfaces within the Pose

        Returns:
            (dict): {'nanohedra_score_per_res': , 'number_fragment_residues_total': ,
                     'number_fragment_residues_central': , 'multiple_fragment_ratio': ,
                     'percent_fragment_helix': , 'percent_fragment_strand': ,
                     'percent_fragment_coil': , 'unique_fragments': }
        """
        score = self.score  # inherently calls self.get_fragment_metrics(). Returns None if this fails
        if score is None:  # can be 0.0
            return {}

        metrics = {'nanohedra_score_per_res': score,
                   'nanohedra_score': self.all_residue_score,
                   'nanohedra_score_central': self.center_residue_score,
                   'number_fragment_residues_total': self.fragment_residues_total,
                   'number_fragment_residues_central': self.central_residues_with_fragment_overlap,
                   'multiple_fragment_ratio': self.multiple_frag_ratio,
                   'percent_fragment_helix': self.helical_fragment_content,
                   'percent_fragment_strand': self.strand_fragment_content,
                   'percent_fragment_coil': self.coil_fragment_content,
                   'unique_fragments': self.number_of_fragments,
                   'total_interface_residues': self.total_interface_residues,
                   'percent_residues_fragment_all': self.percent_residues_fragment_all,
                   'percent_residues_fragment_center': self.percent_residues_fragment_center}
        if self.sym_entry:
            # if self.pose:  # Todo test
            #     for oligomer in self.oligomers:
            #         oligomer.calculate_secondary_structure()
            metrics.update(
                {'design_dimension': self.sym_entry.get_design_dim(),
                 'component_1_symmetry': self.sym_entry.get_group1_sym(),
                 # TODO clean oligomers[].entities mechanism
                 'component_1_number_of_residues': self.oligomers[0].entities[0].number_of_residues,
                 'component_2_symmetry': self.sym_entry.get_group2_sym(),
                 'component_2_number_of_residues': self.oligomers[1].entities[0].number_of_residues,
                 # 'component_1_n_terminal_helix': self.oligomers[0].entities[0].is_n_term_helical(),
                 # 'component_1_c_terminal_helix': self.oligomers[0].entities[0].is_c_term_helical(),
                 # 'component_2_n_terminal_helix': self.oligomers[1].entities[0].is_n_term_helical(),
                 # 'component_2_c_terminal_helix': self.oligomers[1].entities[0].is_c_term_helical(),
                 })

        return metrics

    def pose_fragments(self):
        """Returns:
            (dict): {'1_2_24': [(78, 87, ...), ...], ...}
        """
        return self.fragment_observations

    def pose_transformation(self):
        """Returns:
            (dict): {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}
        """
        return self.transform_d  # Todo enable transforms with pdbDB

    def pdb_input_parameters(self):
        return self.pdb_dir1_path, self.pdb_dir2_path

    def symmetry_parameters(self):
        return self.sym_entry_number, self.oligomer_symmetry_1, self.oligomer_symmetry_2, self.design_symmetry_pg

    def rotation_parameters(self):
        return self.rot_range_deg_pdb1, self.rot_range_deg_pdb2, self.rot_step_deg1, self.rot_step_deg2

    def degeneracy_parameters(self):
        return self.degen1, self.degen2

    def degen_and_rotation_parameters(self):
        return self.degeneracy_parameters(), self.rotation_parameters()

    def compute_last_rotation_state(self):
        number_steps1 = self.rot_range_deg_pdb1 / self.rot_step_deg1
        number_steps2 = self.rot_range_deg_pdb2 / self.rot_step_deg1

        return int(number_steps1), int(number_steps2)

    # def add_flags(self, flags_file):  # UNUSED
    #     self.set_flags(**load_flags(flags_file))

    def set_flags(self, symmetry=None, design_with_evolution=True, sym_entry_number=None,
                  design_with_fragments=True, generate_fragments=True, write_fragments=True,  # fragments_exist=None,
                  output_assembly=False, design_selector=None, ignore_clashes=False, script=True, mpi=False,
                  number_of_trajectories=PUtils.nstruct, skip_logging=None, analysis=False, **kwargs):  # nanohedra_output,
        self.design_symmetry = symmetry
        self.sym_entry_number = sym_entry_number
        # self.nano = nanohedra_output
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
        if skip_logging:
            self.skip_logging = skip_logging

    def set_symmetry(self, symmetry=None, dimension=None, uc_dimensions=None, expand_matrices=None, **kwargs):
        #            sym_entry_number=None,
        """{symmetry: (str), dimension: (int), uc_dimensions: (list), expand_matrices: (list[list])}

        (str)
        (int)
        (list)
        (list[tuple[list[list], list]])
        """
        # self.sym_entry_number = sym_entry_number
        self.design_symmetry = symmetry
        self.design_dim = dimension
        self.uc_dimensions = uc_dimensions
        self.expand_matrices = expand_matrices

    def return_symmetry_parameters(self):
        return dict(symmetry=self.design_symmetry, design_dimension=self.design_dim,
                    uc_dimensions=self.uc_dimensions, expand_matrices=self.expand_matrices)

    def start_log(self, debug=False, level=2):
        if self.skip_logging:  # set up null_logger
            self.log = null_log
            return None

        if debug:
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

        if design_db and isinstance(design_db, FragmentDatabase):
            self.design_db = design_db

        if score_db and isinstance(score_db, FragmentDatabase):
            self.score_db = score_db

    def directory_string_to_path(self, pose_id):  # todo
        assert self.program_root, 'No program_root attribute set! Cannot create a path from a pose_id without a ' \
                                  'program_root!'
        self.path = os.path.join(self.program_root, pose_id.replace('Projects-', 'Projects%s' % os.sep).replace(
            '_Designs-', '_Designs%s' % os.sep))

    def set_up_design_directory(self):
        """Prepare output Directory and File locations. Each DesignDirectory always includes this format"""
        if not os.path.exists(self.path):
            raise DesignError('Path does not exist!\n\t%s' % self.path)
            # self.log.warning('%s: Path does not exist!' % self.path)
        self.protein_data = os.path.join(self.program_root, PUtils.data.title())
        self.pdbs = os.path.join(self.protein_data, 'PDBs')  # Used to store downloaded PDB's
        self.sequences = os.path.join(self.protein_data, PUtils.sequence_info)

        self.all_scores = os.path.join(self.program_root, 'All' + PUtils.scores_outdir.title())  # TODO db integration
        self.trajectories = os.path.join(self.all_scores, '%s_Trajectories.csv' % self.__str__())
        self.residues = os.path.join(self.all_scores, '%s_Residues.csv' % self.__str__())
        self.design_sequences = os.path.join(self.all_scores, '%s_Sequences.pkl' % self.__str__())

        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.scores_file = os.path.join(self.scores, PUtils.scores_file)
        self.designs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.scripts = os.path.join(self.path, PUtils.scripts)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)
        self.sdf = os.path.join(self.path, PUtils.symmetry_def_file_dir)
        self.pose_file = os.path.join(self.path, PUtils.pose_file)
        self.frag_file = os.path.join(self.frags, PUtils.frag_text_file)
        self.asu = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean))
        self.assembly = os.path.join(self.path, '%s_%s' % (self.name, PUtils.assembly))
        self.refine_pdb = os.path.join(self.path, '%s_for_refine.pdb' % os.path.splitext(PUtils.clean)[0])
        self.consensus_pdb = os.path.join(self.path, '%s_for_consensus.pdb' % os.path.splitext(PUtils.clean)[0])
        self.refined_pdb = os.path.join(self.designs, os.path.basename(self.refine_pdb))
        self.consensus_design_pdb = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        self.info_pickle = os.path.join(self.data, 'info.pkl')

        if os.path.exists(self.info_pickle):  # Pose has already been processed. We can assume files are available
            self.info = unpickle(self.info_pickle)
            if 'design' in self.info and self.info['design']:  # Todo, respond to the state
                dummy = True
        else:  # Ensure directories are only created once Pose Processing is called
            # self.log.debug('Setting up DesignDirectory for design: %s' % self.source)
            self.make_path(self.protein_data)
            self.make_path(self.pdbs)
            self.make_path(self.sequences)

        self.make_path(self.all_scores, condition=self.analysis)
        self.make_path(self.frags, condition=self.query_fragments)
        self.make_path(self.sdf, condition=self.nano)  # Todo, is this necessary anymore?

        if os.path.exists(self.frag_file):
            # if self.info['fragments']:
            self.gather_fragment_info()
            # self.get_fragment_metrics(from_file=True)

    def get_wildtype_file(self):
        """Retrieve the wild-type file name from Design Directory"""
        wt_file = glob(self.asu)
        assert len(wt_file) == 1, '%s: More than one matching file found with %s' % (self.path, PUtils.asu)
        return wt_file[0]
        # for file in os.listdir(self.building_blocks):
        #     if file.endswith(PUtils.asu):
        #         return os.path.join(self.building_blocks, file)

    def get_designs(self):  # design_type='design'
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
    #     return len(bb for symm_bb in self.building_blocks for bb in symm_bb)
    #
    # def return_unique_pose_stats(self):  # Depreciated
    #     return len(bb for symm in self.building_blocks for bb in symm)

    # def return_fragment_metrics(self):
    #     self.all_residue_score, self.center_residue_score, self.fragment_residues_total, \
    #         self.central_residues_with_fragment_overlap, self.multiple_frag_ratio, self.fragment_content_d

    def get_oligomers(self):
        # if self.directory_type == 'design':
        self.oligomer_names = os.path.basename(self.building_blocks).split('_')
        for idx, name in enumerate(self.oligomer_names):
            pdb_files = glob(os.path.join(self.path, '%s*.pdb' % name))
            assert len(pdb_files) == 1, 'Incorrect match [%d != 1] found using %s*.pdb!' % (len(pdb_files), name)
            self.oligomers.append(PDB.from_file(pdb_files[0], name=name, log=self.log))
            # self.oligomers[idx].name = name
            # TODO Chains must be symmetrized on input before SDF creation, currently raise DesignError
            # sdf_file_name = os.path.join(os.path.dirname(self.oligomers[name].filepath), self.sdf, '%s.sdf' % name)
            # self.sdfs[name] = self.oligomers[name].make_sdf(out_path=sdf_file_name, modify_sym_energy=True)
            # self.oligomers[name].reorder_chains()
        self.log.debug('%s: %d matching oligomers found' % (self.path, len(self.oligomers)))

    def get_fragment_metrics(self):  # , from_file=True, from_pose=False):
        """Set/get fragment metrics for all fragment observations in the design"""
        self.log.debug('Starting fragment metric collection')
        if self.info.get('fragments', None):
            if self.fragment_observations:
                # self.log.debug('Found fragment observations from %s' % self.frag_file)
                design_metrics = return_fragment_interface_metrics(calculate_match_metrics(self.fragment_observations))
            # if from_pose and self.pose:
            elif self.pose:
                self.log.debug('No fragment observations, getting info from Pose')
                design_metrics = self.pose.return_fragment_query_metrics(total=True)
            else:
                # no fragments were found. set design_metrics empty
                design_metrics = return_fragment_interface_metrics(None, null=True)
        else:
            self.log.warning('%s: There are no fragment observations for this Design! Have you run %s on it yet? Trying'
                             ' %s now...'
                             % (self.path, PUtils.generate_fragments, PUtils.generate_fragments))
            self.generate_interface_fragments()
            if self.info.get('fragments', None):
                if self.fragment_observations:
                    design_metrics = return_fragment_interface_metrics(
                        calculate_match_metrics(self.fragment_observations))
                else:
                    # no fragments were found. set design_metrics empty
                    design_metrics = return_fragment_interface_metrics(None, null=True)
            else:
                raise DesignError('Something is wrong in Design logic. This error shouldn\'t be reached.')
                # return None

        self.all_residue_score = design_metrics['nanohedra_score']
        self.center_residue_score = design_metrics['nanohedra_score_central']
        self.fragment_residues_total = design_metrics['number_fragment_residues_total']
        self.central_residues_with_fragment_overlap = design_metrics['number_fragment_residues_central']
        self.multiple_frag_ratio = design_metrics['multiple_fragment_ratio']
        self.helical_fragment_content = design_metrics['percent_fragment_helix']
        self.strand_fragment_content = design_metrics['percent_fragment_strand']
        self.coil_fragment_content = design_metrics['percent_fragment_coil']
        # self.number_of_fragments = design_metrics['number_fragments']  # Now @property self.number_of_fragments

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
            self.percent_residues_fragment_all = self.fragment_residues_total / self.total_interface_residues
            self.percent_residues_fragment_center = \
                self.central_residues_with_fragment_overlap / self.total_interface_residues
        except ZeroDivisionError:
            self.log.warning('%s: No interface residues were found. Is there an interface in your design?' % str(self))
            self.percent_residues_fragment_all, self.percent_residues_fragment_center = 0.0, 0.0

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_docking_metrics(self):
        with open(self.nano_master_log, 'r') as master_log:
            parameters = master_log.readlines()
            for line in parameters:
                if 'Oligomer 1 Input Directory: ' in line:  # "PDB 1 Directory Path: " or
                    self.pdb_dir1_path = line.split(':')[-1].strip()
                elif 'Oligomer 2 Input Directory: 'in line:  #  "PDB 2 Directory Path: " or '
                    self.pdb_dir2_path = line.split(':')[-1].strip()
                elif 'Master Output Directory: ' in line:
                    self.master_outdir = line.split(':')[-1].strip()
                elif 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                    self.sym_entry_number = int(line.split(':')[-1].strip())
                elif 'Oligomer 1 Point Group Symmetry: ' in line:  # "Oligomer 1 Symmetry: "
                    self.oligomer_symmetry_1 = line.split(':')[-1].strip()
                elif 'Oligomer 2 Point Group Symmetry: ' in line:  # "Oligomer 2 Symmetry: " or
                    self.oligomer_symmetry_2 = line.split(':')[-1].strip()
                elif 'SCM Point Group Symmetry: ' in line:  # underlying point group # "Design Point Group Symmetry: "
                    self.design_symmetry_pg = line.split(':')[-1].strip()
                elif "Oligomer 1 Internal ROT DOF: " in line:  # ,
                    self.internal_rot1 = line.split(':')[-1].strip()
                elif "Oligomer 2 Internal ROT DOF: " in line:  # ,
                    self.internal_rot2 = line.split(':')[-1].strip()
                elif "Oligomer 1 Internal Tx DOF: " in line:  # ,
                    self.internal_zshift1 = line.split(':')[-1].strip()
                elif "Oligomer 2 Internal Tx DOF: " in line:  # ,
                    self.internal_zshift2 = line.split(':')[-1].strip()
                elif "Oligomer 1 Setting Matrix: " in line:
                    self.set_mat1 = np.array(eval(line.split(':')[-1].strip()))
                elif "Oligomer 2 Setting Matrix: " in line:
                    self.set_mat2 = np.array(eval(line.split(':')[-1].strip()))
                elif "Oligomer 1 Reference Frame Tx DOF: " in line:  # ,
                    self.ref_frame_tx_dof1 = line.split(':')[-1].strip()
                elif "Oligomer 2 Reference Frame Tx DOF: " in line:  # ,
                    self.ref_frame_tx_dof2 = line.split(':')[-1].strip()
                elif 'Resulting SCM Symmetry: ' in line:  # "Resulting Design Symmetry: " or
                    self.design_symmetry = line.split(':')[-1].strip()
                elif 'SCM Dimension: ' in line:  # "Design Dimension: " or
                    self.design_dim = int(line.split(':')[-1].strip())
                elif 'SCM Unit Cell Specification: ' in line:  # "Unit Cell Specification: " or
                    self.uc_spec_string = line.split(':')[-1].strip()
                elif "Oligomer 1 ROT Sampling Range: " in line:
                    self.rot_range_deg_pdb1 = int(line.split(':')[-1].strip())
                elif "Oligomer 2 ROT Sampling Range: " in line:
                    self.rot_range_deg_pdb2 = int(line.split(':')[-1].strip())
                elif "Oligomer 1 ROT Sampling Step: " in line:
                    self.rot_step_deg1 = int(line.split(':')[-1].strip())
                elif "Oligomer 2 ROT Sampling Step: " in line:
                    self.rot_step_deg2 = int(line.split(':')[-1].strip())
                elif 'Degeneracies Found for Oligomer 1' in line:
                    self.degen1 = line.split()[0]
                    if self.degen1.isdigit():
                        self.degen1 = int(self.degen1) + 1  # number of degens is added to the original orientation
                    else:
                        self.degen1 = 1  # No degens becomes a single degen
                elif 'Degeneracies Found for Oligomer 2' in line:
                    self.degen2 = line.split()[0]
                    if self.degen2.isdigit():
                        self.degen2 = int(self.degen2) + 1  # number of degens is added to the original orientation
                    else:
                        self.degen2 = 1  # No degens becomes a single degen

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_fragment_info(self):
        """Gather observed fragment metrics from Nanohedra output"""
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
                elif line[:23] == 'central res pair freqs:':
                    # pair_freq = list(eval(line[22:].strip()))
                    pair_freq = None
                    # self.fragment_cluster_freq_d[cluster_id] = pair_freq
                    # self.fragment_cluster_residue_d[cluster_id]['freq'] = pair_freq
        self.fragment_observations = [{'mapped': frag_obs[0], 'paired': frag_obs[1], 'cluster': frag_obs[2],
                                       'match': frag_obs[3]} for frag_obs in fragment_observations]
        self.info['fragments'] = True

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_pose_metrics(self):
        """Gather information for the docked Pose from Nanohedra output. Includes coarse fragment metrics"""
        with open(self.pose_file, 'r') as f:
            pose_info_file_lines = f.readlines()
            for line in pose_info_file_lines:
                if line[:15] == 'DOCKED POSE ID:':
                    self.pose_id = line[15:].strip()
                elif line[:38] == 'Unique Mono Fragments Matched (z<=1): ':
                    self.high_quality_int_residues_matched = int(line[38:].strip())
                # number of interface residues with fragment overlap potential from other oligomer
                elif line[:31] == 'Unique Mono Fragments Matched: ':
                    self.central_residues_with_fragment_overlap = int(line[31:].strip())
                # number of interface residues with 2 residues on either side of central residue
                elif line[:36] == 'Unique Mono Fragments at Interface: ':
                    self.fragment_residues_total = int(line[36:].strip())
                elif line[:25] == 'Interface Matched (%): ':  #  matched / at interface * 100
                    self.percent_overlapping_fragment = float(line[25:].strip()) / 100
                elif line[:20] == 'ROT/DEGEN MATRIX PDB':
                    data = eval(line[22:].strip())
                    self.transform_d[int(line[20:21])] = {'rot/deg': np.array(data)}
                elif line[:15] == 'INTERNAL Tx PDB':  # all below parsing lacks PDB number suffix such as PDB1 or PDB2
                    data = eval(line[17:].strip())
                    if data:  # == 'None'
                        self.transform_d[int(line[15:16])]['tx_int'] = np.array([0, 0, 0])
                    else:
                        self.transform_d[int(line[15:16])]['tx_int'] = np.array(data)
                elif line[:18] == 'SETTING MATRIX PDB':
                    data = eval(line[20:].strip())
                    self.transform_d[int(line[18:19])]['setting'] = np.array(data)
                elif line[:22] == 'REFERENCE FRAME Tx PDB':
                    data = eval(line[24:].strip())
                    if data:
                        self.transform_d[int(line[22:23])]['tx_ref'] = np.array([0, 0, 0])
                    else:
                        self.transform_d[int(line[22:23])]['tx_ref'] = np.array(data)
                elif 'Nanohedra Score:' in line:  # res_lev_sum_score
                    self.all_residue_score = float(line[16:].rstrip())
                elif 'CRYST1 RECORD:' in line:
                    self.cryst_record = line[15:].strip()
                elif line[:31] == 'Canonical Orientation PDB1 Path':
                    self.cannonical_pdb1 = line[:31].strip()
                elif line[:31] == 'Canonical Orientation PDB2 Path':
                    self.cannonical_pdb2 = line[:31].strip()

    def pickle_info(self):
        self.make_path(self.data)
        pickle_object(self.info, self.info_pickle, out_path='')

    def prepare_rosetta_flags(self, flag_variables, stage, out_path=os.getcwd()):
        """Prepare a protocol specific Rosetta flags file with program specific variables

        Args:
            flag_variables (list(tuple)): The variable value pairs to be filed in the RosettaScripts XML
            stage (str): The protocol stage or flag suffix to name the specific flags file
        Keyword Args:
            out_path=cwd (str): Disk location to write the flags file
        Returns:
            (str): Disk location of the written flags file
        """
        flags = copy.deepcopy(rosetta_flags)
        flags.extend(flag_options[stage])
        flags.extend(['-out:path:pdb %s' % self.designs, '-out:path:score %s' % self.scores])
        variables_for_flag_file = '-parser:script_vars ' + ' '.join('%s=%s' % (variable, str(value))
                                                                    for variable, value in flag_variables)
        flags.append(variables_for_flag_file)

        out_file = os.path.join(out_path, 'flags_%s' % stage)
        with open(out_file, 'w') as f:
            f.write('\n'.join(flags))
            f.write('\n')

        return out_file

    def prepare_rosetta_commands(self):
        # Set up the command base (rosetta bin and database paths)
        main_cmd = copy.deepcopy(script_cmd)
        if self.design_dim is not None:  # can be 0
            protocol = PUtils.protocol[self.design_dim]
            if self.design_dim == 0:  # point
                self.log.debug('Design has Symmetry Entry Number: %s (Laniado & Yeates, 2020)'
                               % str(self.sym_entry_number))
                sym_def_file = sdf_lookup(self.sym_entry_number)
                main_cmd += ['-symmetry_definition', sym_def_file]
            else:  # layer or space
                sym_def_file = sdf_lookup(None, dummy=True)  # currently grabbing dummy.sym
                main_cmd += ['-symmetry_definition', 'CRYST1']
            self.log.info('Symmetry Option: %s' % protocol)
        else:
            sym_def_file = 'null'
            protocol = PUtils.protocol[-1]  # Make part of self.design_dim
            self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it is'
                              'an ASU, may be missing crucial contacts. Is this what you want?')

        if self.nano:
            self.log.info('Input Oligomers: %s' % ', '.join(oligomer.name for oligomer in self.oligomers))

        chain_breaks = {entity: entity.get_terminal_residue('c').number for entity in self.pose.entities}
        self.log.info('Found the following chain breaks in the ASU:\n\t%s'
                      % ('\n\t'.join('\tEntity %s, Chain %s Residue %d' % (entity.name, entity.chain_id, residue_number)
                                     for entity, residue_number in list(chain_breaks.items())[:-1])))
        self.log.info('Total number of residues in Pose: %d' % self.pose.number_of_residues)

        # self.log.info('Pulling fragment info from clusters: %s' % ', '.join(metrics['fragment_cluster_ids']
        #                                                                    for metrics in interface_metrics.values()))

        # Get ASU distance parameters
        if self.nano:  # Todo adapt to self.design_dim and not nanohedra input
            max_com_dist = 0
            for oligomer in self.oligomers:
                # asu_oligomer_com_dist.append(np.linalg.norm(np.array(template_pdb.get_center_of_mass())
                com_dist = np.linalg.norm(self.pose.pdb.center_of_mass - oligomer.center_of_mass)
                # need to use self.pose.pdb.center+of_mass as we want ASU COM not Sym Mates COM (self.pose.center_of_mass)
                if com_dist > max_com_dist:
                    max_com_dist = com_dist

            dist = round(math.sqrt(math.ceil(max_com_dist)), 0)
            self.log.info('Expanding ASU by %f Angstroms' % dist)
        else:
            dist = 0

        # ------------------------------------------------------------------------------------------------------------
        # Rosetta Execution formatting
        # ------------------------------------------------------------------------------------------------------------
        cst_value = round(0.2 * reference_average_residue_weight, 2)
        shell_scripts = []
        # RELAX: Prepare command and flags file
        refine_variables = [('pdb_reference', self.asu), ('scripts', PUtils.rosetta_scripts),
                            ('sym_score_patch', PUtils.sym_weights), ('symmetry', protocol), ('sdf', sym_def_file),
                            ('dist', dist), ('cst_value', cst_value), ('cst_value_sym', (cst_value / 2))]

        # Need to assign the designable residues for each entity to a interface1 or interface2 variable
        refine_variables.extend(self.interface_residue_d.items())

        # assign any additional designable residues
        if self.pose.required_residues:
            refine_variables.append(('required_residues',
                                     ','.join(str(residue.number) for residue in self.pose.required_residues)))
        else:
            # get an out of bounds index
            refine_variables.append(('required_residues', int(list(chain_breaks.values())[-1]) + 50))

        self.make_path(self.scripts)
        flags_refine = self.prepare_rosetta_flags(refine_variables, PUtils.stage[1], out_path=self.scripts)
        relax_cmd = main_cmd + \
            ['@%s' % flags_refine, '-scorefile', os.path.join(self.scores, PUtils.scores_file),
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[1])]
        refine_cmd = relax_cmd + ['-in:file:s', self.refine_pdb, '-parser:script_vars', 'switch=%s' % PUtils.stage[1]]
        if self.consensus:
            if self.design_with_fragments:
                consensus_cmd = relax_cmd + ['-in:file:s', self.consensus_pdb, '-parser:script_vars',
                                             'switch=%s' % PUtils.stage[5]]
                if self.script:
                    shell_scripts.append(write_shell_script(subprocess.list2cmdline(consensus_cmd),
                                                            name=PUtils.stage[5], out_path=self.scripts,
                                                            status_wrap=self.info_pickle))
                else:
                    self.log.info('Consensus Command: %s' % subprocess.list2cmdline(consensus_cmd))
                    consensus_process = subprocess.Popen(consensus_cmd)
                    # Wait for Rosetta Consensus command to complete
                    consensus_process.communicate()
            else:
                self.log.critical('Cannot run consensus design without fragment info and none was found.'
                                  ' Did you mean to design with -generate_fragments False? You will need to run with'
                                  '\'True\' if you want to use fragments')

        # Mutate all design positions to Ala before the Refinement
        mutated_pdb = copy.deepcopy(self.pose.pdb)
        for entity_pair, residue_pair in self.pose.interface_residues.items():
            for idx, entity_residues in enumerate(residue_pair):
                self.log.debug('Mutating residues from Entity %s' % entity_pair[idx].name)
                for residue in entity_residues:
                    if residue.type != 'GLY':  # no mutation from GLY to ALA as Rosetta will build a CB.
                        mutated_pdb.mutate_residue(residue=residue, to='A')

        mutated_pdb.write(out_path=self.refine_pdb)
        self.log.debug('Cleaned PDB for Refine: \'%s\'' % self.refine_pdb)

        # Create executable/Run FastRelax on Clean ASU/Consensus ASU with RosettaScripts
        if self.script:
            self.log.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(refine_cmd), name=PUtils.stage[1],
                                                    out_path=self.scripts, status_wrap=self.info_pickle))
        else:
            self.log.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
            refine_process = subprocess.Popen(refine_cmd)
            # Wait for Rosetta Refine command to complete
            refine_process.communicate()

        # DESIGN: Prepare command and flags file
        design_variables = copy.deepcopy(refine_variables)
        design_variables.append(('design_profile', self.info['design_profile']))

        constraint_percent, free_percent = 0, 1
        if self.evolution:
            constraint_percent = 0.5
            free_percent -= constraint_percent
        design_variables.append(('constrained_percent', constraint_percent))
        design_variables.append(('free_percent', free_percent))

        flags_design = self.prepare_rosetta_flags(design_variables, PUtils.stage[2], out_path=self.scripts)
        design_cmd = main_cmd + (['-in:file:pssm', self.info['evolutionary_profile']] if self.evolution else []) + \
            ['-in:file:s', self.refined_pdb, '-in:file:native', self.asu, '-nstruct', str(self.number_of_trajectories),
             '@%s' % os.path.join(self.path, flags_design),
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[2] + '.xml'),
             '-scorefile', os.path.join(self.scores, PUtils.scores_file)]

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        # all_chains = [entity.chain_id for entity in self.pose.entities]  # pose.interface_residues}  # ['A', 'B', 'C']
        # # add symmetry definition files and set metrics up for oligomeric symmetry
        # if self.nano:
        #     design_variables.extend([('sdf%s' % chain, self.sdfs[name])
        #                              for chain, name in zip(all_chains, list(self.sdfs.keys()))])  # REQUIRES py3.6
        flags_metric = self.prepare_rosetta_flags(design_variables, PUtils.stage[3], out_path=self.scripts)

        pdb_list = os.path.join(self.scripts, 'design_files.txt')
        generate_files_cmd = ['python', PUtils.list_pdb_files, '-d', self.designs, '-o', pdb_list]
        metric_cmd = main_cmd + \
            ['-in:file:l', pdb_list, '-in:file:native', self.refined_pdb, '@%s' % os.path.join(self.path, flags_metric),
             '-out:file:score_only', os.path.join(self.scores, PUtils.scores_file),
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[3] + '.xml')]

        if self.mpi:
            design_cmd = run_cmds[PUtils.rosetta_extras] + design_cmd
            metric_cmd = run_cmds[PUtils.rosetta_extras] + metric_cmd
            self.script = True

        metric_cmds = [metric_cmd + ['-parser:script_vars', 'interface=%d' % number] for number in [1, 2]]

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
        if self.script:
            self.log.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(design_cmd), name=PUtils.stage[2],
                                                    out_path=self.scripts, status_wrap=self.info_pickle))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(generate_files_cmd), name=PUtils.stage[3],
                                                    out_path=self.scripts, status_wrap=self.info_pickle,
                                                    additional=[subprocess.list2cmdline(command)
                                                                for n, command in enumerate(metric_cmds)]))
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
            # analysis_cmd = '%s -d %s %s' % (PUtils.program_command, self.path, PUtils.stage[4])
            # write_shell_script(analysis_cmd, name=PUtils.stage[4], out_path=self.scripts, status_wrap=self.info_pickle
        else:
            pose_s = self.design_analysis()
            outpath = os.path.join(self.all_scores, PUtils.analysis_file)
            _header = False
            if not os.path.exists(outpath):
                _header = True
            pose_s.to_csv(outpath, mode='a', header=_header)

        write_shell_script('', name=PUtils.interface_design, out_path=self.scripts,
                           additional=['bash %s' % design_script for design_script in shell_scripts])
        self.info['status'] = {PUtils.stage[stage]: False for stage in [1, 2, 3, 4, 5]}  # change active stage
        # write_commands(shell_scripts, name=PUtils.interface_design, out_path=self.scripts)

    def load_pose(self):
        """For the design info given by a DesignDirectory source, initialize the Pose with self.source file,
        self.symmetry, self.design_selectors, self.fragment_database, and self.log objects

        Handles clash testing and writing the assembly if those options are True
        """
        if self.nano:
            self.get_oligomers()
            if not self.oligomers:
                raise DesignError('No oligomers were found for this design! Cannot initialize pose without oligomers.')
            self.pose = Pose.from_pdb(self.oligomers[0], symmetry=self.design_symmetry, log=self.log,
                                      design_selector=self.design_selector, frag_db=self.frag_db,
                                      ignore_clashes=self.ignore_clashes, euler_lookup=self.euler_lookup)
            #                         self.fragment_observations
            for oligomer in self.oligomers[1:]:
                self.pose.add_pdb(oligomer)
            self.pose.asu = self.pose.pdb  # set the asu
            self.pose.generate_symmetric_assembly()
        else:
            if not self.source:
                raise DesignError('No source file was found for this design! Cannot initialize pose without a source.')
            # Todo ensure that the asu has intra-oligomeric contacts accounted for by oligomer alignment (PDB API) or
            #  quaternion rotational sampling
            self.pose = Pose.from_asu_file(self.source, symmetry=self.design_symmetry, log=self.log,
                                           design_selector=self.design_selector, frag_db=self.frag_db,
                                           ignore_clashes=self.ignore_clashes)
        # Save renumbered PDB to clean_asu.pdb
        if not os.path.exists(self.asu):
            self.pose.pdb.write(out_path=self.asu)
            self.log.info('Cleaned PDB: \'%s\'' % self.asu)
        # if self.pose.symmetry:
        #     if self.pose.symmetric_assembly_is_clash():
        #         raise DesignError('The Symmetric Assembly contains clashes! Design won\'t be considered')
        #     if self.output_assembly:
        #         self.pose.get_assembly_symmetry_mates()
        #         self.pose.write(out_path=self.assembly)
        #         self.log.info('Expanded Assembly PDB: \'%s\'' % self.assembly)

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
            if self.pose.symmetric_assembly_is_clash():
                if self.ignore_clashes:
                    self.log.critical('The Symmetric Assembly contains clashes! %s is not viable.' % self.asu)
                else:
                    raise DesignError('The Symmetric Assembly contains clashes! Design won\'t be considered')
            if self.output_assembly:  # True by default when expand_asu module is used
                self.pose.get_assembly_symmetry_mates()
                self.pose.write(out_path=self.assembly)
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
                             'requested. Each design will load a separate instance which takes time. If you wish to '
                             'speed up processing pass the flag -%s' % Flags.generate_frags)
        self.identify_interface()
        self.pose.generate_interface_fragments(out_path=self.frags, write_fragments=True)  # Todo parameterize write
        for observation in self.pose.fragment_queries.values():
            self.fragment_observations.extend(observation)
        self.info['fragments'] = True
        self.pickle_info()

    def identify_interface(self):
        if not self.pose:
            self.load_pose()
        self.pose.find_and_split_interface()
        self.interface_residue_d = {'interface%d' % interface: residues
                                    for interface, residues in self.pose.interface_split.items()}
        if 'interface1' in self.interface_residue_d and 'interface2' in self.interface_residue_d:
            self.info['design_residues'] = '%s,%s' % (self.interface_residue_d['interface1'],
                                                      self.interface_residue_d['interface2'])
            self.log.info('Interface Residues:\n\t%s'
                          % '\n\t'.join('%s : %s' % (interface, res)
                                        for interface, res in self.interface_residue_d.items()))
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
        self.make_path(self.data)
        self.pose.interface_design(design_dir=self,
                                   evolution=self.evolution, symmetry=self.design_symmetry,
                                   fragments=self.design_with_fragments, write_fragments=self.write_frags,
                                   query_fragments=self.query_fragments)
        # TODO add symmetry or oligomer data to self.info. Right now in self.sym_entry
        self.set_symmetry(**self.pose.return_symmetry_parameters())
        self.log.debug('DesignDirectory Symmetry: %s' % self.return_symmetry_parameters())
        self.make_path(self.designs)  # Todo include these after refine_sbatch.sh? Need if commands are run in python!
        self.make_path(self.scores)
        self.prepare_rosetta_commands()
        self.pickle_info()

    @handle_design_errors(errors=(DesignError, AssertionError))
    def design_analysis(self, merge_residue_data=False, save_trajectories=True, figures=True):
        """Retrieve all score information from a design directory and write results to .csv file

        Keyword Args:
            merge_residue_data (bool): Whether to incorporate residue data into Pose dataframe
            save_trajectories=False (bool): Whether to save trajectory and residue dataframes
            figures=True (bool): Whether to make and save pose figures
        Returns:
            scores_df (Dataframe): Dataframe containing the average values from the input design directory
        """
        # TODO add fraction_buried_atoms
        remove_columns = columns_to_remove
        rename_columns = columns_to_rename

        # Get design information including: interface residues, SSM's, and wild_type/design files
        profile_dict = {}
        design_profile = self.info.get('design_profile', None)
        evolutionary_profile = self.info.get('evolutionary_profile', None)
        fragment_profile = self.info.get('fragment_profile', None)
        if design_profile:
            profile_dict['design'] = parse_pssm(design_profile)
        if evolutionary_profile:
            profile_dict['evolution'] = parse_pssm(evolutionary_profile)
        if fragment_profile:
            profile_dict['fragment'] = unpickle(fragment_profile)
            issm_residues = list(set(profile_dict['fragment'].keys()))
        else:
            issm_residues = []
            self.log.info('Design has no fragment information')
        if self.info.get('fragment_database', None):
            interface_bkgd = get_db_aa_frequencies(PUtils.frag_directory[self.info['fragment_database']])

        # Gather miscellaneous pose specific metrics
        # ensure oligomers are present and if so, their metrics are pulled out. Happens when pose is scored.
        self.load_pose()
        other_pose_metrics = self.pose_metrics()  # these are initialized with DesignDirectory init
        if not other_pose_metrics:
            raise DesignError('Scoring this design encountered problems. Check the log (%s) for any errors which may '
                              'have caused this and fix' % self.log.handlers[0].baseFilename)

        # Todo fold these into Model and attack these metrics from a Pose object
        #  This will get rid of the self.log
        wt_pdb = PDB.from_file(self.get_wildtype_file(), log=self.log)
        wt_sequence = wt_pdb.atom_sequences

        design_residues = self.info.get('design_residues', None)
        if design_residues:
            design_residues = [int(residue[:-1]) for residue in design_residues.split(',')]  # remove chain, change type
        else:  # This should never happen as we catch at other_pose_metrics
            raise DesignError('No residues were marked for design. Have you run -%s or -%s?'
                              % (PUtils.generate_fragments, PUtils.interface_design))

        int_b_factor = sum(wt_pdb.residue(residue).get_ave_b_factor() for residue in design_residues)
        other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(design_residues), 2)

        idx_slice = pd.IndexSlice
        if not os.path.exists(self.scores_file):
            # initialize design dataframes as empty
            pose_stat_s, protocol_stat_s, protocol_stats_s, sim_series = {}, {}, [], []

            # if not self.info:
            #     raise DesignError('Has not been initialized for design and therefore can\'t be analyzed. '
            #                       'Initialize and perform interface design if you want to measure this design.')

        else:
            # Get the scores from all design trajectories
            all_design_scores = read_scores(os.path.join(self.scores, PUtils.scores_file))
            all_design_scores = remove_interior_keys(all_design_scores, remove_score_columns)

            # Gather mutations for residue specific processing and design sequences
            # self.log.debug('Design Files: %s' % ', '.join(self.get_designs()))
            sequence_mutations = generate_all_design_mutations(self.get_designs(), self.get_wildtype_file())  # TODO
            # self.log.debug('Design Files: %s' % ', '.join(sequence_mutations))
            # self.log.debug('Chain offset: %s' % str(offset_dict))

            # Remove wt sequence and find all designs which have corresponding pdb files
            sequence_mutations.pop('ref')
            # all_design_sequences = {AnalyzeMutatedSequences.get_pdb_sequences(file) for file in self.get_designs()}
            # for pdb in models:
            #     for chain in pdb.chain_id_list
            #         sequences[chain][pdb.name] = pdb.atom_sequences[chain]
            # Todo just pull from design pdbs... reorient for {chain: {name: sequence, ...}, ...} ^^
            all_design_sequences = generate_sequences(wt_sequence, sequence_mutations)
            all_design_sequences = {chain: remove_pdb_prefixes(all_design_sequences[chain]) for chain in
                                    all_design_sequences}
            all_design_scores = remove_pdb_prefixes(all_design_scores)
            self.log.debug('all_design_sequences: %s' % ', '.join(name for chain in all_design_sequences
                                                                  for name in all_design_sequences[chain]))
            # for chain in all_design_sequences:
            #     all_design_sequences[chain] = remove_pdb_prefixes(all_design_sequences[chain])

            # self.log.debug('all_design_sequences2: %s' % ', '.join(name for chain in all_design_sequences
            #                                                      for name in all_design_sequences[chain]))
            self.log.debug('all_design_scores: %s' % ', '.join(all_design_scores.keys()))
            # Ensure data is present for both scores and sequences, then initialize DataFrames
            good_designs = list(
                set(design for design_sequences in all_design_sequences.values() for design in design_sequences)
                & set(all_design_scores.keys()))
            self.log.info('All Designs: %s' % ', '.join(good_designs))
            all_design_scores = clean_dictionary(all_design_scores, good_designs, remove=False)
            all_design_sequences = {chain: clean_dictionary(all_design_sequences[chain], good_designs, remove=False)
                                    for chain in all_design_sequences}
            self.log.debug('All Sequences: %s' % all_design_sequences)

            scores_df = pd.DataFrame(all_design_scores).T
            # Gather all columns into specific types for processing and formatting TODO move up
            report_columns, per_res_columns, hbonds_columns = {}, [], []
            for column in list(scores_df.columns):
                if column.startswith('R_'):
                    report_columns[column] = column.replace('R_', '')
                elif column.startswith('symmetry_switch'):
                    other_pose_metrics['symmetry'] = scores_df.loc[PUtils.stage[1], column].replace('make_', '')
                elif column.startswith('per_res_'):
                    per_res_columns.append(column)
                elif column.startswith('hbonds_res_selection'):
                    hbonds_columns.append(column)
            rename_columns.update(report_columns)
            rename_columns.update({'R_int_sc': 'shape_complementarity', 'R_full_stability': 'full_stability_complex'})
            #                        'R_full_stability_oligomer_A': 'full_stability_oligomer_A',
            #                        'R_full_stability_oligomer_B': 'full_stability_oligomer_B',
            #                        'R_full_stability_A_oligomer': 'full_stability_oligomer_A',
            #                        'R_full_stability_B_oligomer': 'full_stability_oligomer_B',
            #                        'R_int_energy_context_A_oligomer': 'int_energy_context_oligomer_A',
            #                        'R_int_energy_context_B_oligomer': 'int_energy_context_oligomer_B'})
            #                       TODO TEST if DONE? remove the update when metrics protocol is changed
            res_columns = hbonds_columns + per_res_columns
            remove_columns += res_columns + [groups]

            # Format columns
            scores_df = scores_df.rename(columns=rename_columns)
            scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
            # Check proper input
            metric_set = necessary_metrics.copy() - set(scores_df.columns)
            assert metric_set == set(), 'Missing required metrics: %s' % metric_set
            # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
            # TODO protocol switch or no_design switch?
            protocol_s = scores_df[groups]
            self.log.debug(protocol_s)
            designs = protocol_s.index.to_list()
            self.log.debug('Design indices: %s' % designs)
            # Modify protocol name for refine and consensus
            for stage in PUtils.stage_f:
                if stage in designs:
                    # change design index value to PUtils.stage[i] (for consensus and refine)
                    protocol_s[stage] = stage  # TODO remove in future scripts
                    # protocol_s.at[PUtils.stage[stage], groups] = PUtils.stage[stage]

            # Replace empty strings with numpy.notanumber (np.nan), drop str columns, and convert remaining to float
            scores_df = scores_df.replace('', np.nan)
            scores_df = scores_df.drop(remove_columns, axis=1, errors='ignore').astype(float)
            scores_df = columns_to_new_column(scores_df, summation_pairs)
            scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
            # Remove unnecessary and Rosetta score terms TODO learn know how to produce them. Not in FastRelax...
            scores_df.drop(unnecessary + rosetta_terms, axis=1, inplace=True, errors='ignore')

            # TODO remove dirty when columns are correct (after P432)
            #  and column tabulation precedes residue/hbond_processing
            interface_hbonds = dirty_hbond_processing(
                all_design_scores)  # , offset=offset_dict) when hbonds are pose numbering
            # interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)  # , offset=offset_dict)

            all_mutations = generate_all_design_mutations(self.get_designs(), self.get_wildtype_file(), pose_num=True)
            all_mutations_no_chains = make_mutations_chain_agnostic(all_mutations)
            all_mutations_simplified = simplify_mutation_dict(all_mutations_no_chains)
            cleaned_mutations = remove_pdb_prefixes(all_mutations_simplified)
            residue_dict = dirty_residue_processing(all_design_scores, cleaned_mutations, hbonds=interface_hbonds)
            #                                       offset=offset_dict)
            # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
            # residue_dict = residue_processing(all_design_scores, cleaned_mutations, per_res_columns,
            #                                   hbonds=interface_hbonds)
            #                                   offset=offset_dict)

            # Calculate amino acid observation percent from residue dict and background SSM's
            obs_d = {}
            for profile in profile_dict:
                obs_d[profile] = {design: mutation_conserved(residue_dict[design], profile_dict[profile])
                                  for design in residue_dict}

            if 'fragment' in profile_dict:
                # Remove residues from fragment dict if no fragment information available for them
                obs_d['fragment'] = remove_interior_keys(obs_d['fragment'], issm_residues, keep=True)

            # Add observation information into the residue dictionary
            for design in residue_dict:
                res_dict = {'observed_%s' % profile: obs_d[profile][design] for profile in obs_d}
                residue_dict[design] = weave_sequence_dict(base_dict=residue_dict[design], **res_dict)

            # Find the observed background for each design in the pose
            pose_observed_bkd = {profile: {design: per_res_metric(obs_d[profile][design]) for design in obs_d[profile]}
                                 for profile in profile_dict}
            for profile in profile_dict:
                scores_df['observed_%s' % profile] = pd.Series(pose_observed_bkd[profile])

            # Process H-bond and Residue metrics to dataframe
            residue_df = pd.concat({key: pd.DataFrame(value) for key, value in residue_dict.items()}).unstack()
            # returns multi-index column with residue number as first (top) column index, metric as second index
            # during residue_df unstack, all residues with missing dicts are copied as nan
            number_hbonds = {entry: len(interface_hbonds[entry]) for entry in interface_hbonds}
            # number_hbonds_df = pd.DataFrame(number_hbonds, index=['number_hbonds', ]).T
            number_hbonds_s = pd.Series(number_hbonds, name='number_hbonds')
            scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)

            # Add design residue information to scores_df such as core, rim, and support measures
            for r_class in residue_classificiation:
                scores_df[r_class] = \
                    residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == r_class]].sum(axis=1)
            scores_df['int_composition_similarity'] = scores_df.apply(residue_composition_diff, axis=1)

            interior_residue_df = \
                residue_df.loc[:, idx_slice[:, residue_df.columns.get_level_values(1) == 'interior']].droplevel(1,
                                                                                                                axis=1)
            # Check if any of the values in columns are 1. If so, return True for that column
            interior_residues = interior_residue_df.any().index[interior_residue_df.any()].to_list()
            interface_residues = list(set(residue_df.columns.get_level_values(0).unique()) - set(interior_residues))
            assert len(interface_residues) > 0, 'No interface residues found!'
            other_pose_metrics['observations'] = len(good_designs)
            other_pose_metrics['percent_fragment'] = len(issm_residues) / len(interface_residues)
            scores_df['total_interface_residues'] = len(interface_residues)
            # 'design_residues' coming in as 234B (residue_number|chain)
            if set(interface_residues) != set(design_residues):
                self.log.info('Residues %s are located in the interior' %
                              ', '.join(map(str, set(design_residues) - set(interface_residues))))

            # Interface B Factor TODO ensure asu.pdb has B-factors for Nanohedra
            int_b_factor = 0
            for residue in interface_residues:
                # if residue <= chain_sep:
                int_b_factor += wt_pdb.residue(residue).get_ave_b_factor()
                # else:
                #     int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[1], residue)
            other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(interface_residues), 2)

            pose_alignment = multi_chain_alignment(all_design_sequences)
            mutation_frequencies = clean_dictionary(pose_alignment['counts'], interface_residues, remove=False)
            # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
            pose_res_dict = {}
            for profile in profile_dict:  # both mut_freq and profile_dict[profile] are one-indexed
                pose_res_dict['divergence_%s' % profile] = pos_specific_jsd(mutation_frequencies, profile_dict[profile])
            # if 'fragment' in profile_dict:
            pose_res_dict['divergence_interface'] = compute_jsd(mutation_frequencies, interface_bkgd)
            # pose_res_dict['hydrophobic_collapse_index'] = hci()  # TODO HCI

            # Divide/Multiply column pairs to new columns
            scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')

            # Merge processed dataframes
            scores_df = pd.merge(protocol_s, scores_df, left_index=True, right_index=True)
            protocol_df = pd.DataFrame(protocol_s)
            protocol_df.columns = pd.MultiIndex.from_product([[''], protocol_df.columns])
            residue_df = pd.merge(protocol_df, residue_df, left_index=True, right_index=True)

            # Drop refine row and any rows with nan values
            scores_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
            residue_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
            clean_scores_df = scores_df.dropna()
            residue_df = residue_df.dropna(how='all', axis=1)  # remove completely empty columns (obs_interface)
            clean_residue_df = residue_df.dropna()
            # print(residue_df.isna())  #.any(axis=1).to_list())  # scores_df.where()
            scores_na_index = scores_df[~scores_df.index.isin(clean_scores_df.index)].index.to_list()
            residue_na_index = residue_df[~residue_df.index.isin(clean_residue_df.index)].index.to_list()
            if scores_na_index:
                protocol_s.drop(scores_na_index, inplace=True)
                self.log.warning('%s: Trajectory DataFrame dropped rows with missing values: %s' %
                               (self.path, ', '.join(scores_na_index)))
                # might have to remove these from all_design_scores in the case that that is used as a dictionary again
            if residue_na_index:
                self.log.warning('%s: Residue DataFrame dropped rows with missing values: %s' %
                               (self.path, ', '.join(residue_na_index)))
                for res_idx in residue_na_index:
                    residue_dict.pop(res_idx)
                self.log.debug('Residue_dict:\n\n%s\n\n' % residue_dict)

            # Fix reported per_residue_energy to contain only interface. BUT With delta, these residues should be subtracted
            # int_residue_df = residue_df.loc[:, idx_slice[interface_residues, :]]

            # Get unique protocols for protocol specific metrics and drop unneeded protocol values
            unique_protocols = protocol_s.unique().tolist()
            for value in ['refine']:
                # TODO TEST if remove '' is fixed ## after P432 MinMatch6 upon future script deployment
                try:
                    unique_protocols.remove(value)
                except ValueError:
                    pass
            self.log.info('Unique Protocols: %s' % ', '.join(unique_protocols))

            designs_by_protocol, sequences_by_protocol = {}, {}
            stats_by_protocol = {protocol: {} for protocol in unique_protocols}
            for protocol in unique_protocols:
                designs_by_protocol[protocol] = protocol_s.index[protocol_s == protocol].tolist()
                sequences_by_protocol[protocol] = {chain: {design: all_design_sequences[chain][design]
                                                           for design in all_design_sequences[chain]
                                                           if design in designs_by_protocol[protocol]}
                                                   for chain in all_design_sequences}
                protocol_alignment = multi_chain_alignment(sequences_by_protocol[protocol])
                protocol_mutation_freq = remove_non_mutations(protocol_alignment['counts'], interface_residues)
                protocol_res_dict = {'divergence_%s' % profile: pos_specific_jsd(protocol_mutation_freq,
                                                                                 profile_dict[profile])
                                     for profile in profile_dict}  # both prot_freq and profile_dict[profile] are 0-idx
                protocol_res_dict['divergence_interface'] = compute_jsd(protocol_mutation_freq, interface_bkgd)

                # Get per residue divergence metric by protocol
                for key in protocol_res_dict:
                    stats_by_protocol[protocol]['%s_per_res' % key] = per_res_metric(
                        protocol_res_dict[key])  # , key=key)
                    # {protocol: 'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}
                # Get per design observed background metric by protocol
                for profile in profile_dict:
                    stats_by_protocol[protocol]['observed_%s' % profile] = per_res_metric(
                        {design: pose_observed_bkd[profile][design] for design in designs_by_protocol[protocol]})

                # Gather the average number of residue classifications for each protocol
                for res_class in residue_classificiation:
                    stats_by_protocol[protocol][res_class] = clean_residue_df.loc[
                        designs_by_protocol[protocol],
                        idx_slice[:, clean_residue_df.columns.get_level_values(1) == res_class]].mean().sum()
                stats_by_protocol[protocol]['observations'] = len(designs_by_protocol[protocol])
            protocols_by_design = {v: k for k, _list in designs_by_protocol.items() for v in _list}

            # POSE ANALYSIS: Get total pose design statistics
            # remove below if consensus is run multiple times. the cst_weights are very large and destroy the mean
            trajectory_df = clean_scores_df.drop(PUtils.stage[5], axis=0, errors='ignore')
            assert len(trajectory_df.index.to_list()) > 0, 'No design was done on this pose'

            traj_stats = {}
            protocol_stat_df = {}
            for stat in stats_metrics:
                traj_stats[stat] = getattr(trajectory_df, stat)().rename(stat)
                protocol_stat_df[stat] = getattr(clean_scores_df.groupby(groups), stat)()
                if stat == 'mean':
                    continue
                protocol_stat_df[stat].index = protocol_stat_df[stat].index.to_series().map(
                    {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
            trajectory_df = trajectory_df.append(list(traj_stats.values()))
            # Here we add consensus back to the trajectory_df after removing above
            trajectory_df = trajectory_df.append(list(protocol_stat_df.values()))

            if merge_residue_data:
                trajectory_df = pd.merge(trajectory_df, clean_residue_df, left_index=True, right_index=True)

            # Calculate protocol significance
            # Find all unique combinations of protocols using 'mean' as all protocol combination source. Excludes Consensus
            protocol_subset_df = trajectory_df.loc[:, protocol_specific_columns]
            protocol_intersection = set(protocols_of_interest) & set(unique_protocols)

            if protocol_intersection != set(protocols_of_interest):
                self.log.warning(
                    'Missing %s protocol required for significance measurements! Significance analysis failed!'
                    % ', '.join(set(protocols_of_interest) - protocol_intersection))
                significance = False
                sim_sum_and_divergence_stats, sim_measures = {}, {}
            else:
                significance = True

                sig_df = protocol_stat_df['mean']  # stats_metrics[0]
                assert len(sig_df.index.to_list()) > 1, 'Can\'t measure protocol significance, not enough protocols!'
                pvalue_df = pd.DataFrame()
                for pair in combinations(sorted(sig_df.index.to_list()), 2):
                    select_df = protocol_subset_df.loc[designs_by_protocol[pair[0]] + designs_by_protocol[pair[1]], :]
                    difference_s = sig_df.loc[pair[0], protocol_specific_columns].sub(
                        sig_df.loc[pair[1], protocol_specific_columns])
                    pvalue_df[pair] = df_permutation_test(select_df, difference_s,
                                                          group1_size=len(designs_by_protocol[pair[0]]),
                                                          compare=stats_metrics[0])
                self.log.debug(pvalue_df)
                pvalue_df = pvalue_df.T  # change the significance pairs to the indices and protocol specific columns to columns
                trajectory_df = trajectory_df.append(pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1))

                # Get pose sequence divergence TODO protocol switch
                sim_sum_and_divergence_stats = {'%s_per_res' % key: per_res_metric(pose_res_dict[key]) for key in
                                                pose_res_dict}

                # Compute sequence differences between each protocol
                residue_energy_df = \
                    clean_residue_df.loc[:,
                                         idx_slice[:, clean_residue_df.columns.get_level_values(1) == 'energy_delta']]

                res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
                residue_energy_np = StandardScaler().fit_transform(residue_energy_df.values)
                residue_energy_pc = res_pca.fit_transform(residue_energy_np)
                residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                                    columns=['pc' + str(x + index_offset)
                                                             for x in range(len(res_pca.components_))])
                #                                    ,columns=residue_energy_df.columns)

                seq_pca = copy.deepcopy(res_pca)
                residue_dict.pop(PUtils.stage[1])  # Remove refine from analysis before PC calculation
                pairwise_sequence_diff_np = all_vs_all(residue_dict, sequence_difference)
                pairwise_sequence_diff_np = StandardScaler().fit_transform(pairwise_sequence_diff_np)
                seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
                # Compute the euclidean distance
                # pairwise_pca_distance_np = pdist(seq_pc)
                # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

                # Make PC DataFrame
                # First take all the principal components identified from above and merge with labels
                # Next the labels will be grouped and stats are taken for each group (mean is important)
                # All protocol means will have pairwise distance measured as a means of accessing similarity
                # These distance metrics will be reported in the final pose statistics
                seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_dict.keys()),
                                         columns=['pc' + str(x + index_offset)
                                                  for x in range(len(seq_pca.components_))])
                # Merge principle components with labels
                residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)
                seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

                # Gather protocol similarity/distance metrics
                sim_measures = {'similarity': None, 'seq_distance': {}, 'energy_distance': {}}
                # Find similarity between each type of protocol by taking row average of all p-values for each metric
                mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance
                mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
                sim_measures['similarity'] = mean_pvalue_s
                # sim_measures['similarity'] = pvalue_df.mean(axis=1)

                grouped_pc_stat_df_dict, grouped_pc_energy_df_dict = {}, {}
                for stat in stats_metrics:
                    grouped_pc_stat_df_dict[stat] = getattr(seq_pc_df.groupby(groups), stat)()
                    grouped_pc_energy_df_dict[stat] = getattr(residue_energy_pc_df.groupby(groups), stat)()
                    if stat == 'mean':
                        # if renaming is necessary
                        # protocol_stat_df[stat].index = protocol_stat_df[stat].index.to_series().map(
                        #     {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
                        seq_pca_mean_distance_vector = pdist(grouped_pc_stat_df_dict[stat])
                        energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df_dict[stat])
                        # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
                        #                             for k in seq_pca_mean_distance_vector)
                        for k, dist in enumerate(seq_pca_mean_distance_vector):
                            i, j = condensed_to_square(k, len(grouped_pc_stat_df_dict[stat].index))
                            sim_measures['seq_distance'][(grouped_pc_stat_df_dict[stat].index[i],
                                                          grouped_pc_stat_df_dict[stat].index[j])] = dist

                        for k, e_dist in enumerate(energy_pca_mean_distance_vector):
                            i, j = condensed_to_square(k, len(grouped_pc_energy_df_dict[stat].index))
                            sim_measures['energy_distance'][(grouped_pc_energy_df_dict[stat].index[i],
                                                             grouped_pc_energy_df_dict[stat].index[j])] = e_dist

                for pc_stat in grouped_pc_stat_df_dict:
                    self.log.info(grouped_pc_stat_df_dict[pc_stat])

                # Find total protocol similarity for different metrics
                for measure in sim_measures:
                    measure_s = pd.Series(
                        {pair: sim_measures[measure][pair] for pair in combinations(protocols_of_interest, 2)})
                    sim_sum_and_divergence_stats['protocol_%s_sum' % measure] = measure_s.sum()

            # Create figures
            if figures:  # Todo with relevant .ipynb figures
                _path = os.path.join(self.all_scores, str(self))
                # Set up Labels & Plot the PC data
                protocol_map = {protocol: i for i, protocol in enumerate(unique_protocols)}
                integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
                pc_labels_group = [protocols_by_design[design] for design in residue_dict]
                # pc_labels_group = np.array([protocols_by_design[design] for design in residue_dict])
                pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_dict]
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
                custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]
                ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left', bbox_to_anchor=(1.0, .5))
                # # Add group mean to the plot
                # for name, label in integer_map.items():
                #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(), seq_pc[pc_labels_group == label, 1].mean(),
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
                custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]
                ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left', bbox_to_anchor=(1.0, .5))
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                plt.savefig('%s_res_energy_pca.png' % _path)

            # Save Trajectory, Residue DataFrames, and PDB Sequences
            if save_trajectories:
                # trajectory_df.to_csv('%s_Trajectories.csv' % _path)
                trajectory_df.to_csv(self.trajectories)
                # clean_residue_df.to_csv('%s_Residues.csv' % _path)
                clean_residue_df.to_csv(self.residues)
                seq_file = pickle_object(all_design_sequences, '%s_Sequences' % str(self), out_path=self.all_scores)

            # CONSTRUCT: Create pose series and format index names
            pose_stat_s, protocol_stat_s = {}, {}
            for stat in stats_metrics:
                pose_stat_s[stat] = trajectory_df.loc[stat, :]
                pose_stat_s[stat] = pd.concat([pose_stat_s[stat]], keys=['pose'])
                pose_stat_s[stat] = pd.concat([pose_stat_s[stat]], keys=[stat])
                # Collect protocol specific metrics in series
                suffix = ''
                if stat != 'mean':
                    suffix = '_' + stat
                protocol_stat_s[stat] = pd.concat([protocol_subset_df.loc[protocol + suffix, :]
                                                   for protocol in unique_protocols], keys=unique_protocols)
                protocol_stat_s[stat] = pd.concat([protocol_stat_s[stat]], keys=[stat])

            protocol_stats_s = pd.concat([pd.Series(stats_by_protocol[protocol]) for protocol in stats_by_protocol],
                                         keys=unique_protocols)

            # Add series specific Multi-index names to data
            protocol_stats_s = [pd.concat([protocol_stats_s], keys=['stats'])]

            if significance:
                # Find the significance between each pair of protocols
                protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                                           keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
                # squeeze turns the column headers into series indices. Keys appends to make a multi-index

                other_stats_s = pd.Series(sim_sum_and_divergence_stats)
                other_stats_s = pd.concat([other_stats_s], keys=['pose'])
                other_stats_s = pd.concat([other_stats_s], keys=['seq_design'])

                # Process similarity between protocols
                sim_measures_s = pd.concat([pd.Series(values) for values in sim_measures.values()],
                                           keys=[measure for measure in sim_measures])
                sim_series = [protocol_sig_s, other_stats_s, sim_measures_s]

        other_metrics_s = pd.Series(other_pose_metrics)
        other_metrics_s = pd.concat([other_metrics_s], keys=['pose'])
        other_metrics_s = pd.concat([other_metrics_s], keys=['dock'])
        # Combine all series
        pose_s = pd.concat([pose_stat_s[stat] for stat in pose_stat_s] +
                           [protocol_stat_s[stat] for stat in protocol_stat_s]
                           + protocol_stats_s + [other_metrics_s] + sim_series).swaplevel(0, 1)
        # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df
        pose_s.drop([groups, ], level=2, inplace=True, errors='ignore')
        pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
        pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
        pose_s.name = str(self)

        return pose_s

    @staticmethod
    def make_path(path, condition=True):
        if not os.path.exists(path) and condition:
            os.makedirs(path)

    def __str__(self):
        if self.program_root:
            # TODO integrate with designDB?
            return self.path.replace(self.program_root + os.sep, '').replace(os.sep, '-')
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]


def set_up_directory_objects(design_list, mode='design', symmetry=None, project=None):
    """Create DesignDirectory objects from a directory iterable. Add program_root if using DesignDirectory strings"""
    return [DesignDirectory.from_nanohedra(design_path, nano=True, mode=mode, project=project)
            for design_path in design_list]


def set_up_pseudo_design_dir(path, directory, score):  # changed 9/30/20 to locate paths of interest at .path
    pseudo_dir = DesignDirectory(path, nano=False)
    # pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(path)
    pseudo_dir.designs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir
