import copy
import math
import os
import subprocess
from glob import glob

import numpy as np

import CmdUtils as CUtils
from AnalyzeOutput import analyze_output
from CmdUtils import reference_average_residue_weight, script_cmd, run_cmds

from PDB import PDB
import PathUtils as PUtils
from Pose import Pose
from Query.Flags import load_flags
from SequenceProfile import calculate_match_metrics, FragmentDatabase, return_fragment_interface_metrics
from SymDesignUtils import unpickle, start_log, handle_errors_f, sdf_lookup, write_shell_script, pdb_list_file, \
    DesignError, match_score_from_z_value, handle_design_errors, pickle_object, write_commands


logger = start_log(name=__name__, level=2)  # was from SDUtils logger, but moved here per standard suggestion

# Globals
design_direcotry_modes = ['design', 'dock']


class DesignDirectory:  # Todo move PDB coordinate information to Pose. Only use to handle Pose paths/options

    def __init__(self, design_path, nano=False, mode='design', project=None, pose_id=None, debug=False, **kwargs):
        self.name = os.path.splitext(os.path.basename(design_path))[0]  # works for all cases
        self.log = None
        self.nano = nano
        self.mode = mode

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
        self.oligomers = {}

        self.sym_entry_number = None
        self.design_symmetry = None
        self.design_dim = None
        self.uc_dimensions = None
        self.expand_matrices = None

        # self.fragment_cluster_residue_d = {}
        self.fragment_observations = []
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

        # Design flags
        self.design_selection = None
        self.evolution = True
        self.design_with_fragments = True
        self.query_fragments = True
        self.write_frags = True
        # self.fragment_file = None
        # self.fragment_type = 'biological_interfaces'  # default for now, can be found in frag_db
        self.frag_db = None
        self.design_db = None
        self.score_db = None
        self.script = True
        self.mpi = False
        self.output_assembly = False

        # Analysis flags
        self.analysis = False

        if self.nano:
            if project:
                self.program_root = project.rstrip(os.sep)
                if pose_id:  # Todo may not be compatible P432
                    self.directory_string_to_path(pose_id)
                else:
                    self.path = os.path.join(project, design_path)
            else:
                self.path = design_path
                # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2

            if not os.path.exists(self.path):
                raise FileNotFoundError('The specified DesignDirectory \'%s\' was not found!' % self.path)

            self.start_log(debug=debug)
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

            if self.mode == 'design':
                self.program_root = self.path[:self.path.find(self.path.split(os.sep)[-4]) - 1]
                # design_symmetry (P432)
                self.nano_master_log = os.path.join(self.program_root, PUtils.master_log)
                self.building_blocks = self.path[:self.path.find(self.path.split(os.sep)[-3]) - 1]
                # design_symmetry/building_blocks (P432/4ftd_5tch)
                self.source = os.path.join(self.path, PUtils.asu)
                self.set_up_design_directory()

                # self.gather_pose_metrics()
                # self.gather_fragment_info()
            elif self.mode == 'dock':
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
                for bb_dir in next(os.walk(self.program_root))[1]:  # grab directories from os.walk, yielding only top level
                    if os.path.exists(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils?
                        self.building_block_dirs.append(bb_dir)
                        # self.building_block_dirs[k].append(bb_dir)
                        self.building_block_logs.append(os.path.join(self.program_root, bb_dir, '%s_log.txt' % bb_dir))
                        # self.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))
            else:
                raise DesignError('%s: %s is not an available mode. Choose from %s...\n'
                                  % (DesignDirectory.__name__, self.mode, ','.join(design_direcotry_modes)))

            if not os.path.exists(self.nano_master_log):
                raise DesignError('%s: No %s found for this directory! Cannot perform material design without it.\n'
                                  'Ensure you have the file \'%s\' located properly before trying this Design!'
                                  % (self.__str__(), PUtils.master_log, self.nano_master_log))
            self.gather_docking_metrics()
        else:
            if '.pdb' in design_path:  # set up /program_root/projects/project/design
                self.source = design_path
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
            else:  # initialize DesignDirectory to recognize existing /program_root/projects/project/design
                self.path = design_path
                self.asu = os.path.join(self.path, '%s-%s' % (self.name, PUtils.clean))
                self.source = self.asu
                self.program_root = '/%s' % os.path.join(*self.path.split(os.sep)[:-3])  # symmetry.rstrip(os.sep)
                self.projects = '/%s' % os.path.join(*self.path.split(os.sep)[:-2])
                self.project_designs = '/%s' % os.path.join(*self.path.split(os.sep)[:-1])

            self.start_log(debug=debug)
            self.set_up_design_directory()
            # self.design_from_file(symmetry=symmetry)
        self.set_flags(**kwargs)

    @classmethod
    def from_nanohedra(cls, design_path, mode=None, project=None, **kwargs):
        return cls(design_path, nano=True, mode=mode, project=project, **kwargs)

    @classmethod
    def from_file(cls, design_path, project=None, **kwargs):  # mode=None
        return cls(design_path, project=project, **kwargs)

    @property
    def score(self):
        if self.center_residue_score and self.central_residues_with_fragment_overlap:
            return self.center_residue_score / self.central_residues_with_fragment_overlap
        else:
            self.get_fragment_metrics()
            try:
                return self.center_residue_score / self.central_residues_with_fragment_overlap
            except (AttributeError, ZeroDivisionError):
                raise DesignError('No fragment information available! Design cannot be scored.')

    @property
    def number_of_fragments(self):
        return len(self.fragment_observations)

    def __str__(self):
        if self.program_root:
            return self.path.replace(self.program_root + os.sep, '').replace(os.sep, '-')  # TODO integrate with designDB?
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]

    def start_log(self, debug=False, level=2):
        if debug:
            handler, level = 1, 1
        else:
            handler = 2
        self.log = start_log(name=self.name, handler=handler, level=level, location=os.path.join(self.path, self.name))

    def connect_db(self, frag_db=None, design_db=None, score_db=None):
        if frag_db and isinstance(frag_db, FragmentDatabase):
            self.frag_db = frag_db

        if design_db and isinstance(design_db, FragmentDatabase):
            self.design_db = design_db

        if score_db and isinstance(score_db, FragmentDatabase):
            self.score_db = score_db

    def directory_string_to_path(self, pose_id):  # todo
        assert self.program_root, 'No program_root attribute set! Cannot create a path from a pose_id without a program_root!'
        self.path = os.path.join(self.program_root, pose_id.replace('-', os.sep))

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
        self.designs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.scripts = os.path.join(self.path, PUtils.scripts)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)
        self.sdf = os.path.join(self.path, PUtils.symmetry_def_file_dir)
        self.pose_file = os.path.join(self.path, PUtils.pose_file)
        self.frag_file = os.path.join(self.frags, PUtils.frag_text_file)
        self.asu = os.path.join(self.path, '%s_%s' % (self.name, PUtils.clean))
        self.refine_pdb = os.path.join(self.path, '%s_for_refine.pdb' % os.path.splitext(PUtils.clean)[0])
        self.consensus_pdb = os.path.join(self.path, '%s_for_consensus.pdb' % os.path.splitext(PUtils.clean)[0])
        self.refined_pdb = os.path.join(self.designs, os.path.basename(self.refine_pdb))
        self.consensus_design_pdb = os.path.join(self.designs, os.path.basename(self.consensus_pdb))
        self.info_pickle = os.path.join(self.data, 'info.pkl')

        if os.path.exists(self.info_pickle):  # Pose has already been processed. We can assume files are available
            self.info = unpickle(self.info_pickle)
            if 'design' in self.info and self.info['design']:
                dummy = True

        else:  # Ensure directories are only created once Pose Processing is called
            if not os.path.exists(self.protein_data):
                os.makedirs(self.protein_data)
            if not os.path.exists(self.pdbs):
                os.makedirs(self.pdbs)
            if not os.path.exists(self.sequences):
                os.makedirs(self.sequences)
            if not os.path.exists(self.all_scores) and self.analysis:
                os.makedirs(self.all_scores)

            # if not os.path.exists(self.scores):  # made by Rosetta
            #     os.makedirs(self.scores)
            # if not os.path.exists(self.designs):  # made by Rosetta
            #     os.makedirs(self.designs)
            if not os.path.exists(self.scripts):
                os.makedirs(self.scripts)
            if not os.path.exists(self.frags) and self.query_fragments:
                os.makedirs(self.frags)
            if not os.path.exists(self.data):
                os.makedirs(self.data)
            if not os.path.exists(self.sdf) and self.nano:
                os.makedirs(self.sdf)

        if os.path.exists(self.frag_file):
            # if self.info['fragments']:
            self.gather_fragment_info()
            self.get_fragment_metrics(from_file=True)

    def get_oligomers(self):
        if self.mode == 'design':
            self.oligomer_names = os.path.basename(self.building_blocks).split('_')
            for name in self.oligomer_names:
                name_pdb_file = glob(os.path.join(self.path, '%s*_tx_*.pdb' % name))
                assert len(name_pdb_file) == 1, 'Incorrect match [%d != 1] found using %s*_tx_*.pdb!\nCheck %s' % \
                                                (len(name_pdb_file), name, self.__str__())
                self.oligomers[name] = PDB.from_file(name_pdb_file[0])
                self.oligomers[name].name = name
                # TODO Chains must be symmetrized on input before SDF creation, currently raise DesignError
                sdf_file_name = os.path.join(os.path.dirname(self.oligomers[name].filepath), self.sdf, '%s.sdf' % name)
                self.sdfs[name] = self.oligomers[name].make_sdf(out_path=sdf_file_name, modify_sym_energy=True)
                self.oligomers[name].reorder_chains()
            self.log.debug('%s: %d matching oligomers found' % (self.path, len(self.oligomers)))

    def get_designs(self, design_type='design'):
        """Return the paths of all design files in a DesignDirectory"""
        return glob(os.path.join(self.designs, '*%s*' % design_type))

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

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_docking_metrics(self):
        with open(self.nano_master_log, 'r') as master_log:  # os.path.join(base_directory, 'master_log.txt')
            parameters = master_log.readlines()
            for line in parameters:
                if "PDB 1 Directory Path: " or 'Oligomer 1 Input Directory: ' in line:
                    self.pdb_dir1_path = line.split(':')[-1].strip()
                elif "PDB 2 Directory Path: " or 'Oligomer 2 Input Directory: 'in line:
                    self.pdb_dir2_path = line.split(':')[-1].strip()
                elif 'Master Output Directory: ' in line:
                    self.master_outdir = line.split(':')[-1].strip()
                elif "Symmetry Entry Number: " or 'Nanohedra Entry Number: ' in line:
                    self.sym_entry_number = int(line.split(':')[-1].strip())
                elif "Oligomer 1 Symmetry: " or 'Oligomer 1 Point Group Symmetry: ' in line:
                    self.oligomer_symmetry_1 = line.split(':')[-1].strip()
                elif "Oligomer 2 Symmetry: " or 'Oligomer 2 Point Group Symmetry: ' in line:
                    self.oligomer_symmetry_2 = line.split(':')[-1].strip()
                elif "Design Point Group Symmetry: " or 'SCM Point Group Symmetry: ' in line:  # underlying point group
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
                elif "Resulting Design Symmetry: " or 'Resulting SCM Symmetry: ' in line:  # program_root for total design
                    self.design_symmetry = line.split(':')[-1].strip()
                elif "Design Dimension: " or 'SCM Dimension: ' in line:
                    self.design_dim = int(line.split(':')[-1].strip())
                elif "Unit Cell Specification: " or 'SCM Unit Cell Specification: ' in line:
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

    def pose_score(self):
        """Returns: (dict): """
        return self.all_residue_score

    def pose_metrics(self):
        """Returns: (dict): {'nanohedra_score': , 'average_fragment_z_score': , 'unique_fragments': }
        """
        return {'nanohedra_score': self.all_residue_score, 'average_fragment_z_score': self.ave_z,
                'unique_fragments': self.num_fragments}  # Todo expand with AnalyzeOutput

    def pose_fragments(self):
        """Returns: (dict): {'1_2_24': [(78, 87, ...), ...], ...}
        """
        return self.fragment_observations

    def pose_transformation(self):
        """Returns: (dict): {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}
        """
        return self.transform_d  # Todo enable with pdbDB

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_fragment_info(self):
        """Gather fragment metrics from Nanohedra output"""
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

    # def return_fragment_metrics(self):
    #     self.all_residue_score, self.center_residue_score, self.fragment_residues_total, \
    #         self.central_residues_with_fragment_overlap, self.multiple_frag_ratio, self.fragment_content_d

    def get_fragment_metrics(self, from_file=True, from_pose=False):
        """Set the design fragment metrics for all fragment observations"""
        if from_file and self.fragment_observations:
            total_design_metrics = return_fragment_interface_metrics(calculate_match_metrics(self.fragment_observations))
        elif from_pose and self.pose:
            total_design_metrics = self.pose.return_fragment_query_metrics(total=True)
        else:
            self.log.warning('%s: There are no fragment observations associated with this Design! Have you scored '
                             'it yet? See \'Scoring Interfaces\' in the %s' % (self.path, PUtils.guide_string))
            return None

        self.all_residue_score = total_design_metrics['nanohedra_score']
        self.center_residue_score = total_design_metrics['nanohedra_score_central']
        self.fragment_residues_total = total_design_metrics['number_fragment_residues_total']
        self.central_residues_with_fragment_overlap = total_design_metrics['number_fragment_residues_central']
        self.multiple_frag_ratio = total_design_metrics['multiple_fragment_ratio']
        self.helical_fragment_content = total_design_metrics['percent_fragment_helix']
        self.strand_fragment_content = total_design_metrics['percent_fragment_strand']
        self.coil_fragment_content = total_design_metrics['percent_fragment_coil']

        # 'total_interface_residues': total_residues,
        # 'percent_residues_fragment_all': percent_interface_covered,
        # 'percent_residues_fragment_center': percent_interface_matched,
        # 'fragment_cluster_ids': ','.join(clusters),  # Todo
        # Need to grab the total residue number from these to calculate correctly
        # self.interface_residue_count, self.percent_interface_matched, self.percent_interface_covered,

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_pose_metrics(self):
        """Gather docking metrics from Nanohedra output"""
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

    def add_flags(self, flags_file):
        self.set_flags(**load_flags(flags_file))

    def set_flags(self, symmetry=None, design_with_evolution=True, sym_entry_number=None,
                  design_with_fragments=True, generate_fragments=True, write_fragments=True,  # fragments_exist=None,
                  output_assembly=False, design_selection=None, script=True, mpi=False, **kwargs):  # nanohedra_output,
        self.design_symmetry = symmetry
        self.sym_entry_number = sym_entry_number
        # self.nano = nanohedra_output
        self.design_selection = design_selection
        self.evolution = design_with_evolution
        self.design_with_fragments = design_with_fragments
        # self.fragment_file = fragments_exist
        self.query_fragments = generate_fragments
        self.write_frags = write_fragments
        self.output_assembly = output_assembly
        self.script = script
        self.mpi = mpi
        # self.fragment_type

    def prepare_rosetta_commands(self):  # , script=False, mpi=False):
        cst_value = round(0.2 * reference_average_residue_weight, 2)
        # Set up protocol program_root
        main_cmd = copy.deepcopy(script_cmd)
        # sym_entry_number, oligomer_symmetry_1, oligomer_symmetry_2, design_symmetry = des_dir.symmetry_parameters()
        # sym = SDUtils.handle_symmetry(sym_entry_number)  # This makes the process dependent on the PUtils.master_log file
        if self.design_dim is not None:  # can be 0
            protocol = PUtils.protocol[self.design_dim]
            if self.design_dim > 0:  # layer or space
                sym_def_file = sdf_lookup(None, dummy=True)  # currently grabbing dummy.sym
                main_cmd += ['-symmetry_definition', 'CRYST1']
            else:  # point
                self.log.debug('Design has Symmetry Entry Number: %d (Laniado & Yeates, 2020)' % self.sym_entry_number)
                sym_def_file = sdf_lookup(self.sym_entry_number)
                main_cmd += ['-symmetry_definition', sym_def_file]
            self.log.info('Symmetry Option: %s' % protocol)
        else:
            sym_def_file = 'null'
            protocol = PUtils.protocol[-1]  # Make part of self.design_dim
            self.log.critical('No symmetry invoked during design. Rosetta will still design your PDB, however, if it is'
                              'an ASU, may be missing crucial contacts. Is this what you want?')

        if self.nano:
            self.log.info('Input Oligomers: %s' % ', '.join(name for name in self.oligomers))

        self.log.info('Found the following chain breaks in the ASU:\n%s'
                      % ('\n'.join(['\tEntity %s, Chain %s Residue %d'
                                    % (entity.name, entity.chain_id, entity.get_terminal_residue('c').number)
                                    for entity in self.pose.entities])))
        self.log.info('Total number of residues in Pose: %d' % self.pose.number_of_residues)
        self.log.info('Interface Residues: %s'  # Todo ensure interface is checked even if no fragment info!
                      % ', '.join('%s%s' % (residue.number, entity.chain_id)
                                  for entity, residues in self.pose.interface_residues.values()
                                  for residue in residues))

        # # need to assign the designable residues for each entity to a interfaceA or interfaceB variable
        # interface_metrics = self.pose.return_fragment_query_metrics()
        #
        # self.log.info('Pulling fragment info from clusters: %s' % ', '.join(metrics['fragment_cluster_ids']
        #                                                                    for metrics in interface_metrics.values()))

        # Get ASU distance parameters
        if self.nano:  # Todo adapt to self.design_dim and not nanohedra input
            max_com_dist = 0
            for oligomer in self.oligomers.values():
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
        shell_scripts = []
        # ------------------------------------------------------------------------------------------------------------
        # RELAX: Prepare command and flags file
        refine_variables = [('pdb_reference', self.asu), ('scripts', PUtils.rosetta_scripts),
                            ('sym_score_patch', PUtils.sym_weights), ('symmetry', protocol), ('sdf', sym_def_file),
                            ('dist', dist), ('cst_value', cst_value), ('cst_value_sym', (cst_value / 2))]

        # Assumes all entity chains are renamed from A to Z for entities (1 to n)
        all_chains = [entity.chain_id for entity in self.pose.entities]  # pose.interface_residues}  # ['A', 'B', 'C']
        interface_residue_d = {'interface%s' % chain: '' for chain in all_chains}
        for i, (entity, residues) in enumerate(self.pose.interface_residues.items(), 1):
            interface_residue_d['interface%s' % entity.chain_id] = ','.join('%d%s'
                                                                            % (residue.number, entity.chain_id)
                                                                            for residue in residues)

        refine_variables.extend(interface_residue_d.items())

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
                                                            name=PUtils.stage[5], out_path=self.scripts))
                    # additional_cmd = [subprocess.list2cmdline(consensus_cmd)]
                else:
                    self.log.info('Consensus Command: %s' % subprocess.list2cmdline(consensus_cmd))
                    consensus_process = subprocess.Popen(consensus_cmd)
                    # Wait for Rosetta Consensus command to complete
                    consensus_process.communicate()
            else:
                self.log.critical('Cannot run consensus design without fragment info. Did you mean to design with '
                                  '-fragments_exist None & -generate_fragments False?')

        # Mutate all design positions to Ala before the Refinement
        mutated_pdb = copy.deepcopy(self.pose.pdb)
        for residues in self.pose.interface_residues.values():
            for residue in residues:
                if residue.type != 'GLY':  # no mutation from GLY to ALA as Rosetta will build a CB.
                    mutated_pdb.mutate_residue(residue.number, to='A')

        mutated_pdb.write(out_path=self.refine_pdb)
        self.log.debug('Cleaned PDB for Refine: \'%s\'' % self.refine_pdb)

        # Create executable/Run FastRelax on Clean ASU/Consensus ASU with RosettaScripts
        if self.script:
            self.log.info('Refine Command: %s' % subprocess.list2cmdline(refine_cmd))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(refine_cmd), name=PUtils.stage[1],
                                                    out_path=self.scripts))
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
        # TODO back out nstruct label to command distribution
        design_cmd = main_cmd + ['-in:file:pssm', self.info['evolutionary_profile']] if self.evolution else [] + \
            ['-in:file:s', self.refined_pdb, '-in:file:native', self.asu, '-nstruct', str(PUtils.nstruct),
             '@%s' % os.path.join(self.path, flags_design),
             '-parser:protocol', os.path.join(PUtils.rosetta_scripts, PUtils.stage[2] + '.xml'),
             '-scorefile', os.path.join(self.scores, PUtils.scores_file)]

        # METRICS: Can remove if SimpleMetrics adopts pose metric caching and restoration
        # add symmetry definition files and set metrics up for oligomeric symmetry
        if self.nano:
            design_variables.extend([('sdf%s' % chain, self.sdfs[name])
                                     for chain, name in zip(all_chains, list(self.sdfs.keys()))])  # REQUIRES py3.6 dict
            design_variables.append(('metrics_symmetry', 'oligomer'))
        else:  # This may not always work if the input has lower symmetry
            design_variables.append(('metrics_symmetry', 'no_symmetry'))

        design_variables.extend([('all_but_%s' % chain, ''.join(set(all_chains) - set(chain))) for chain in all_chains])

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

        metric_cmds = {entity.chain_id: metric_cmd + ['-parser:script_vars', 'chain=%s' % entity.chain_id]
                       for entity in self.pose.entities}

        # Create executable/Run FastDesign on Refined ASU with RosettaScripts. Then, gather Metrics on Designs
        if self.script:
            self.log.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(design_cmd), name=PUtils.stage[2],
                                                    out_path=self.scripts))
            shell_scripts.append(write_shell_script(subprocess.list2cmdline(generate_files_cmd), name=PUtils.stage[3],
                                                    out_path=self.scripts,
                                                    additional=[subprocess.list2cmdline(command)
                                                                for n, command in enumerate(metric_cmds.values())]))
            for idx, metric_cmd in enumerate(metric_cmds):
                self.log.info('Metrics Command %d: %s' % (idx, subprocess.list2cmdline(metric_cmd)))
        else:
            self.log.info('Design Command: %s' % subprocess.list2cmdline(design_cmd))
            design_process = subprocess.Popen(design_cmd)
            # Wait for Rosetta Design command to complete
            design_process.communicate()
            for idx, metric_cmd in enumerate(metric_cmds):
                self.log.info('Metrics Command %d: %s' % (idx, subprocess.list2cmdline(metric_cmd)))
                metrics_process = subprocess.Popen(metric_cmd)
                metrics_process.communicate()

        # ANALYSIS: each output from the Design process based on score, Analyze Sequence Variation
        if self.script:
            pass
            # analysis_cmd = '%s -d %s %s' % (PUtils.program_command, self.path, PUtils.stage[4])
            # write_shell_script(analysis_cmd, name=PUtils.stage[4], out_path=self.scripts)
        else:
            pose_s = analyze_output(self)
            outpath = os.path.join(self.all_scores, PUtils.analysis_file)
            _header = False
            if not os.path.exists(outpath):
                _header = True
            pose_s.to_csv(outpath, mode='a', header=_header)

        self.info['status'] = {shell_script: False for shell_script in shell_scripts}

        write_commands(shell_scripts, name=PUtils.interface_design_command, out_path=self.scripts)

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
        flags = copy.deepcopy(CUtils.flags)
        flags.extend(CUtils.flag_options[stage])
        flags.extend(['-out:path:pdb %s' % self.designs, '-out:path:score %s' % self.scores])
        variables_for_flag_file = '-parser:script_vars ' + ' '.join('%s=%s' % (variable, str(value))
                                                                    for variable, value in flag_variables)

        flags.append(variables_for_flag_file)

        out_file = os.path.join(out_path, 'flags_%s' % stage)
        with open(out_file, 'w') as f:
            f.write('\n'.join(flags))

        return out_file  # 'flags_' + stage

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

    @handle_design_errors(errors=(DesignError, AssertionError))
    def load_pose(self):
        self.pose = Pose.from_asu_file(self.source, symmetry=self.design_symmetry, log=self.log,
                                       design_selection=self.design_selection, frag_db=self.frag_db)
        # Save renumbered PDB to clean_asu.pdb
        self.pose.pdb.write(out_path=self.asu)
        self.log.info('Cleaned PDB: \'%s\'' % self.asu)
        if self.pose.symmetry and self.pose.symmetric_assembly_is_clash():
            raise DesignError('The Symmetric Assembly contains clashes! Design (%s) is not being considered'
                              % str(self))
        if self.output_assembly:
            self.pose.get_assembly_symmetry_mates()
            self.pose.write(out_path=os.path.join(self.path, PUtils.assembly))
            self.log.info('Expanded Assembly PDB: \'%s\'' % os.path.join(self.path, PUtils.assembly))

    @handle_design_errors(errors=(DesignError, AssertionError))
    def expand_asu(self):
        print(self.source)
        self.pose = Pose.from_asu_file(self.source, symmetry=self.design_symmetry, log=self.log,
                                       design_selection=self.design_selection, frag_db=self.frag_db)
        # Save renumbered PDB to clean_asu.pdb
        self.pose.pdb.write(out_path=self.asu)
        self.log.info('Cleaned PDB: \'%s\'' % self.asu)
        self.pose.get_assembly_symmetry_mates()
        self.pose.write(out_path=os.path.join(self.path, PUtils.assembly))
        self.log.info('Expanded Assembly PDB: \'%s\'' % os.path.join(self.path, PUtils.assembly))

    @handle_design_errors(errors=(DesignError, AssertionError))
    def generate_interface_fragments(self):
        self.load_pose()
        self.pose.generate_interface_fragments(db=self.frag_db, out_path=self.frags)
        self.info['fragments'] = self.frag_file
        self.info_pickle = pickle_object(self.info, 'info', out_path=self.data)

    @handle_design_errors(errors=(DesignError, AssertionError))
    def interface_design(self):
        self.load_pose()
        # if self.info:  # Todo
        #     return None  # pose has already been initialized for design
        self.pose.interface_design(design_dir=self,  # output_assembly=self.output_assembly,
                                   evolution=self.evolution, symmetry=self.design_symmetry,
                                   fragments=self.design_with_fragments, write_fragments=self.write_frags,
                                   query_fragments=self.query_fragments,)
                                   # frag_db=self.frag_db)  Todo clean up
        self.set_symmetry(**self.pose.return_symmetry_parameters())
        self.log.debug('DesignDirectory Symmetry: %s' % self.return_symmetry_parameters())
        self.info_pickle = pickle_object(self.info, 'info', out_path=self.data)
        self.prepare_rosetta_commands()

    # @handle_errors_f(errors=(FileNotFoundError, ))
    # def gather_fragment_infov0(self):  # DEPRECIATED v0
    #     """Gather fragment metrics from Nanohedra output"""
    #     with open(os.path.join(self.path, PUtils.frag_file), 'r') as f:
    #         frag_match_info_lines = f.readlines()
    #         for line in frag_match_info_lines:
    #             if line[:12] == 'Cluster ID: ':
    #                 cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
    #                 if cluster not in self.fragment_cluster_residue_d:
    #                     self.fragment_cluster_residue_d[cluster] = {}
    #                     # self.fragment_cluster_residue_d[cluster] = {'pair': {}}
    #             elif line[:40] == 'Cluster Central Residue Pair Frequency: ':
    #                 pair_freq = list(eval(line[40:]))
    #                 self.fragment_cluster_freq_d[cluster] = pair_freq
    #                 # Cluster Central Residue Pair Frequency:
    #                 # [(('L', 'Q'), 0.2429), (('A', 'D'), 0.0571), (('V', 'D'), 0.0429), (('L', 'E'), 0.0429),
    #                 # (('T', 'L'), 0.0429), (('L', 'S'), 0.0429), (('T', 'D'), 0.0429), (('V', 'L'), 0.0286),
    #                 # (('I', 'K'), 0.0286), (('V', 'E'), 0.0286), (('L', 'L'), 0.0286), (('L', 'M'), 0.0286),
    #                 # (('L', 'K'), 0.0286), (('T', 'Q'), 0.0286), (('S', 'D'), 0.0286), (('Y', 'G'), 0.0286),
    #                 # (('I', 'F'), 0.0286), (('T', 'K'), 0.0286), (('V', 'I'), 0.0143), (('W', 'I'), 0.0143),
    #                 # (('V', 'Q'), 0.0143), (('I', 'L'), 0.0143), (('F', 'G'), 0.0143), (('E', 'H'), 0.0143),
    #                 # (('L', 'D'), 0.0143), (('N', 'M'), 0.0143), (('K', 'D'), 0.0143), (('L', 'H'), 0.0143),
    #                 # (('L', 'V'), 0.0143), (('L', 'R'), 0.0143)]
    #             elif line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
    #                 res_chain1 = int(line[43:].strip())
    #             elif line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
    #                 res_chain2 = int(line[43:].strip())
    #                 self.fragment_cluster_residue_d[cluster].add((res_chain1, res_chain2))
    #                 # self.fragment_cluster_residue_d[cluster]['pair'].add((res_chain1, res_chain2))
    #             elif line[:17] == 'Overlap Z-Value: ':
    #                 # try:
    #                 self.z_value_dict[cluster] = float(line[17:].strip())
    #                 # except ValueError:
    #                 #     print('%s has misisng Z-value in frag_info_file.txt' % self.__str__())
    #                 #     self.z_value_dict[cluster] = float(1.0)
    #             # elif line[:17] == 'Nanohedra Score: ':  # Depreciated
    #             #     nanohedra_score = float(line[17:].strip())
    #         #             if line[:39] == 'Unique Interface Fragment Match Count: ':
    #         #                 int_match = int(line[39:].strip())
    #         #             if line[:39] == 'Unique Interface Fragment Total Count: ':
    #         #                 int_total = int(line[39:].strip())
    #             elif line[:20] == 'ROT/DEGEN MATRIX PDB':
    #                 # ROT/DEGEN MATRIX PDB1: [[1.0, -0.0, 0], [0.0, 1.0, 0], [0, 0, 1]]
    #                 _matrix = np.array(eval(line[23:]))
    #                 self.transform_d[int(line[20:21])] = {'rot/deg': _matrix}  # dict[pdb# (1, 2)] = {'transform_type': matrix}
    #             elif line[:15] == 'INTERNAL Tx PDB':
    #                 # INTERNAL Tx PDB1: [0, 0, 45.96406061067895]
    #                 _matrix = np.array(eval(line[18:]))
    #                 self.transform_d[int(line[15:16].strip())]['tx_int'] = _matrix
    #             elif line[:18] == 'SETTING MATRIX PDB':
    #                 # SETTING MATRIX PDB1: [[0.707107, 0.408248, 0.57735], [-0.707107, 0.408248, 0.57735], [0.0, -0.816497, 0.57735]]
    #                 _matrix = np.array(eval(line[21:].strip()))
    #                 self.transform_d[int(line[18:19])]['setting'] = _matrix
    #             elif line[:21] == 'REFERENCE FRAME Tx PDB':
    #                 # REFERENCE FRAME Tx PDB1: None
    #                 _matrix = np.array(eval(line[24:].strip()))
    #                 self.transform_d[int(line[21:22])]['tx_ref'] = _matrix
    #             elif 'Residue-Level Summation Score:' in line:
    #                 self.nanohedra_score = float(line[30:].strip())
    #
    #     # for cluster in self.fragment_cluster_residue_d:
    #     #     self.fragment_cluster_residue_d[cluster]['pair'] = list(set(residue_cluster_d[cluster]['pair']))


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
