import os
from glob import glob

import numpy as np

import PathUtils as PUtils
from SymDesignUtils import logger, unpickle, start_log, handle_errors_f


class DesignDirectory:  # Todo remove all PDB specific information and add to Pose. only use to handle Pose paths

    def __init__(self, directory, mode='design', auto_structure=True, symmetry=None):
        self.mode = mode
        self.path = directory
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
        self.symmetry = None
        # design_symmetry_pg (P432)
        self.protein_data = None  # TODO
        # design_symmetry_pg/protein_data (P432/Protein_Data)
        self.pdbs = None  # TODO
        # design_symmetry_pg/protein_data/pdbs (P432/Protein_Data/PDBs)
        self.sequences = None
        # design_symmetry_pg/sequences (P432/Sequence_Info)
        # design_symmetry_pg/protein_data/sequences (P432/Protein_Data/Sequence_Info)  # TODO
        self.all_scores = None
        # design_symmetry_pg/all_scores (P432/All_Scores)
        self.trajectories = None
        # design_symmetry_pg/all_scores/str(self)_Trajectories.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Trajectories.csv)
        self.residues = None
        # design_symmetry_pg/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Residues.csv)
        self.design_sequences = None
        # design_symmetry_pg/all_scores/str(self)_Residues.csv (P432/All_Scores/4ftd_5tch-DEGEN1_2-ROT_1-tx_2_Sequences.pkl)
        self.building_blocks = None
        # design_symmetry_pg/building_blocks (P432/4ftd_5tch)
        self.building_block_logs = []
        self.scores = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.design_pdbs = None  # TODO .designs?
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/rosetta_pdbs)
        self.frags = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/matching_fragment_representatives
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.data = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data
        self.source = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/central_asu.pdb
        self.asu = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu.pdb
        self.refine_pdb = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_refine.pdb
        self.refined_pdb = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs/clean_asu_for_refine.pdb
        self.consensus_pdb = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu_for_consensus.pdb
        self.consensus_design_pdb = None
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs/clean_asu_for_consensus.pdb
        self.oligomer_names = []
        self.oligomers = {}
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/clean_asu.pdb
        self.info = {}
        # design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data/stats.pkl
        #   (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/matching_fragment_representatives)
        self.log = None
        # v ^ both used in dock_dir set up
        self.building_block_logs = None

        if auto_structure:
            if symmetry:
                if len(self.path.split(os.sep)) == 1:
                    self.directory_string_to_path()
            if self.mode == 'design':
                self.design_directory_structure(symmetry=symmetry)
            elif self.mode == 'dock':
                self.dock_directory_structure(symmetry=symmetry)

        self.pdb_dir1_path = None
        self.pdb_dir2_path = None
        self.master_outdir = None
        self.sym_entry_number = None
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
        self.design_symmetry = None
        self.design_dim = None
        self.uc_spec_string = None
        self.degen1 = None
        self.degen2 = None

        self.fragment_cluster_residue_d = {}
        self.fragment_observations = []
        self.fragment_cluster_freq_d = {}
        self.transform_d = {}
        self.z_value_dict = {}
        self.nanohedra_score = None
        self.ave_z = None
        self.num_fragments = None

        self.cannonical_pdb1 = None
        self.cannonical_pdb2 = None

    def __str__(self):
        if self.symmetry:
            return self.path.replace(self.symmetry + os.sep, '').replace(os.sep, '-')  # TODO how integrate with designDB?
        else:
            # When is this relevant?
            return self.path.replace(os.sep, '-')[1:]

    def directory_string_to_path(self):  # string, symmetry
        self.path = self.path.replace('-', os.sep)

    def design_directory_structure(self, symmetry=None):
        # Prepare Output Directory/Files. path always has format:
        if symmetry:
            self.symmetry = symmetry.rstrip(os.sep)
            self.path = os.path.join(symmetry, self.path)
        else:
            self.symmetry = self.path[:self.path.find(self.path.split(os.sep)[-4]) - 1]
        self.log = os.path.join(self.symmetry, PUtils.master_log)
        if not os.path.exists(self.log):
            logger.critical('%s: No %s found in this directory! Cannot perform material design without it.'
                            % (self.__str__(), PUtils.master_log))
            exit()
        self.protein_data = os.path.join(self.symmetry, 'Protein_Data')
        self.pdbs = os.path.join(self.protein_data, 'PDBs')
        self.sequences = os.path.join(self.protein_data, PUtils.sequence_info)
        self.all_scores = os.path.join(self.symmetry, 'All_' + PUtils.scores_outdir.title())  # TODO db integration
        self.trajectories = os.path.join(self.all_scores, '%s_Trajectories.csv' % self.__str__())
        self.residues = os.path.join(self.all_scores, '%s_Residues.csv' % self.__str__())
        self.design_sequences = os.path.join(self.all_scores, '%s_Sequences.pkl' % self.__str__())
        self.building_blocks = self.path[:self.path.find(self.path.split(os.sep)[-3]) - 1]
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.design_pdbs = os.path.join(self.path, PUtils.pdbs_outdir)
        self.frags = os.path.join(self.path, PUtils.frag_dir)
        self.data = os.path.join(self.path, PUtils.data)

        self.source = os.path.join(self.path, PUtils.asu)
        self.asu = os.path.join(self.path, PUtils.clean)
        self.refine_pdb = os.path.join(self.path, '%s_for_refine.pdb' % os.path.splitext(self.asu)[0])
        self.refined_pdb = os.path.join(self.design_pdbs, os.path.basename(self.refine_pdb))
        self.consensus_pdb = os.path.join(self.path, '%s_for_consensus.pdb' % os.path.splitext(self.asu)[0])
        self.consensus_design_pdb = os.path.join(self.design_pdbs, os.path.basename(self.consensus_pdb))

        if not os.path.exists(self.path):
            # raise DesignError('Path does not exist!\n%s' % self.path)
            logger.warning('%s: Path does not exist!' % self.path)
        else:
            # TODO ensure these are only created with Pose Processing is called... New method probably
            if not os.path.exists(self.protein_data):
                os.makedirs(self.protein_data)
            if not os.path.exists(self.pdbs):
                os.makedirs(self.pdbs)
            if not os.path.exists(self.sequences):
                os.makedirs(self.sequences)
            if not os.path.exists(self.all_scores):
                os.makedirs(self.all_scores)
            if not os.path.exists(self.scores):
                os.makedirs(self.scores)
            if not os.path.exists(self.design_pdbs):
                os.makedirs(self.design_pdbs)
            if not os.path.exists(self.data):
                os.makedirs(self.data)
            else:
                if os.path.exists(os.path.join(self.data, 'info.pkl')):

                    # raise DesignError('%s: No information found for pose. Have you initialized it?\n'
                    #                   'Try \'python %s ... pose ...\' or inspect the directory for correct files' %
                    #                   (self.path, PUtils.program_name))
                    self.info = unpickle(os.path.join(self.data, 'info.pkl'))

    def dock_directory_structure(self):  # , symmetry=None):
        """Saves the path of the docking directory as DesignDirectory.path attribute. Tries to populate further using
        typical directory structuring"""
        # self.symmetry = glob(os.path.join(path, 'NanohedraEntry*DockedPoses*'))  # TODO final implementation?
        self.symmetry = self.path  # Assuming that the output directory (^ or v) of Nanohedra is passed as the path
        # self.symmetry = glob(os.path.join(self.path, 'NanohedraEntry*DockedPoses%s' % str(symmetry or '')))  # for design_recap
        self.log = os.path.join(self.symmetry, PUtils.master_log)
        # self.log = [os.path.join(_sym, PUtils.master_log) for _sym in self.symmetry]
        # for k, _sym in enumerate(self.symmetry):
        # for k, _sym in enumerate(next(os.walk(self.symmetry))):
        # self.building_blocks.append(list())
        # self.building_block_logs.append(list())
        # get all dirs from walk('NanohedraEntry*DockedPoses/) Format: [[], [], ...]
        # for bb_dir in next(os.walk(_sym))[1]:  # grabs the directories from os.walk, yielding just top level results
        for bb_dir in next(os.walk(self.symmetry))[1]:  # grab directories from os.walk, yielding just top level results
            if os.path.exists(os.path.join(self.symmetry, bb_dir, '%s_log.txt' % bb_dir)):  # TODO PUtils?
                self.building_blocks.append(bb_dir)
                # self.building_blocks[k].append(bb_dir)
                self.building_block_logs.append(os.path.join(self.symmetry, bb_dir, '%s_log.txt' % bb_dir))
                # self.building_block_logs[k].append(os.path.join(_sym, bb_dir, '%s_log.txt' % bb_dir))

    def get_oligomers(self):
        if self.mode == 'design':
            self.oligomer_names = os.path.basename(self.building_blocks).split('_')
            for name in self.oligomer_names:
                name_pdb_file = glob(os.path.join(self.path, '%s*_tx_*.pdb' % name))
                assert len(name_pdb_file) == 1, 'Incorrect match [%d != 1] found using %s*_tx_*.pdb!\nCheck %s' % \
                                                (len(name_pdb_file), name, self.__str__())
                self.oligomers[name] = PDB(file=name_pdb_file[0])
                # self.oligomers[name] = read_pdb(name_pdb_file[0])
                self.oligomers[name].set_name(name)
                self.oligomers[name].reorder_chains()

    # TODO generators for the various directory levels using the stored directory pieces
    def get_building_block_dir(self, building_block):
        for sym_idx, symm in enumerate(self.symmetry):
            try:
                bb_idx = self.building_blocks[sym_idx].index(building_block)
                return os.path.join(self.symmetry[sym_idx], self.building_blocks[sym_idx][bb_idx])
            except ValueError:
                continue
        return None

    # def return_symmetry_stats(self):  # Depreciated
    #     return len(symm for symm in self.symmetry)
    #
    # def return_building_block_stats(self):  # Depreciated
    #     return len(bb for symm_bb in self.building_blocks for bb in symm_bb)
    #
    # def return_unique_pose_stats(self):  # Depreciated
    #     return len(bb for symm in self.building_blocks for bb in symm)

    def start_log(self, name=None, level=2):
        _name = __name__
        if name:
            _name = name
        self.log = start_log(name=_name, handler=2, level=level,
                             location=os.path.join(self.path, os.path.basename(self.path)))

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_docking_metrics(self):
        with open(self.log, 'r') as master_log:  # os.path.join(base_directory, 'master_log.txt')
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
                elif "Resulting Design Symmetry: " or 'Resulting SCM Symmetry: ' in line:  # symmetry for total design
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
        return self.nanohedra_score

    def pose_metrics(self):
        """Returns: (dict): {'nanohedra_score': , 'average_fragment_z_score': , 'unique_fragments': }
        """
        return {'nanohedra_score': self.nanohedra_score, 'average_fragment_z_score': self.ave_z,
                'unique_fragments': self.num_fragments}

    def pose_fragments(self):
        """Returns: (dict): {'1_2_24': [(78, 87, ...), ...], ...}
        """
        return self.fragment_cluster_residue_d

    def pose_transformation(self):
        """Returns: (dict): {1: {'rot/deg': [[], ...],'tx_int': [], 'setting': [[], ...], 'tx_ref': []}, ...}
        """
        return self.transform_d

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_fragment_info(self):
        """Gather fragment metrics from Nanohedra output"""
        with open(os.path.join(self.path, PUtils.frag_file), 'r') as f:
            frag_match_info_file = f.readlines()
            for line in frag_match_info_file:
                if line[:6] == 'z-val:':
                    overlap_rmsd_divded_by_cluster_rmsd = float(line[6:].strip())
                    match_score = 1 / float(1 + (overlap_rmsd_divded_by_cluster_rmsd ** 2))  # bounds between 0 and 1
                elif line[:21] == 'oligomer1 ch, resnum:':
                    oligomer1_info = line[21:].strip().split(',')
                    chain1 = oligomer1_info[0]  # doesn't matter as all subunits are symmetric
                    residue_number1 = oligomer1_info[1]
                elif line[:21] == 'oligomer2 ch, resnum:':
                    oligomer2_info = line[21:].strip().split(',')
                    chain2 = oligomer2_info[0]  # doesn't matter as all subunits are symmetric
                    residue_number2 = oligomer2_info[1]
                elif line[:3] == 'id:':
                    cluster_id = line[3:].strip()
                    self.fragment_observations.append({'mapped': residue_number1, 'paired': residue_number2,
                                                               'cluster': cluster_id, 'match': match_score})
                    if cluster_id in self.fragment_cluster_residue_d:
                        self.fragment_cluster_residue_d[cluster_id].add((residue_number1, residue_number2))
                        # self.fragment_cluster_residue_d[cluster_id]['pair'].add((residue_number1, residue_number2))
                    else:
                        self.fragment_cluster_residue_d[cluster_id] = {(residue_number1, residue_number2)}
                        # self.fragment_cluster_residue_d[cluster_id] = {'pair': {(residue_number1, residue_number2)}}
                # "mean rmsd: %s\n" % str(cluster_rmsd))
                # "aligned rep: int_frag_%s_%s.pdb\n" % (cluster_id, str(match_count)))
                elif line[:23] == 'central res pair freqs:':
                    pair_freq = list(eval(line[23:].strip()))
                    self.fragment_cluster_freq_d[cluster_id] = pair_freq
                    # self.fragment_cluster_residue_d[cluster_id]['freq'] = pair_freq

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_fragment_infov0(self):  # DEPRECIATED v0
        """Gather fragment metrics from Nanohedra output"""
        with open(os.path.join(self.path, PUtils.frag_file), 'r') as f:
            frag_match_info_lines = f.readlines()
            for line in frag_match_info_lines:
                if line[:12] == 'Cluster ID: ':
                    cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
                    if cluster not in self.fragment_cluster_residue_d:
                        self.fragment_cluster_residue_d[cluster] = {}
                        # self.fragment_cluster_residue_d[cluster] = {'pair': {}}
                elif line[:40] == 'Cluster Central Residue Pair Frequency: ':
                    pair_freq = list(eval(line[40:]))
                    self.fragment_cluster_freq_d[cluster] = pair_freq
                    # Cluster Central Residue Pair Frequency:
                    # [(('L', 'Q'), 0.2429), (('A', 'D'), 0.0571), (('V', 'D'), 0.0429), (('L', 'E'), 0.0429),
                    # (('T', 'L'), 0.0429), (('L', 'S'), 0.0429), (('T', 'D'), 0.0429), (('V', 'L'), 0.0286),
                    # (('I', 'K'), 0.0286), (('V', 'E'), 0.0286), (('L', 'L'), 0.0286), (('L', 'M'), 0.0286),
                    # (('L', 'K'), 0.0286), (('T', 'Q'), 0.0286), (('S', 'D'), 0.0286), (('Y', 'G'), 0.0286),
                    # (('I', 'F'), 0.0286), (('T', 'K'), 0.0286), (('V', 'I'), 0.0143), (('W', 'I'), 0.0143),
                    # (('V', 'Q'), 0.0143), (('I', 'L'), 0.0143), (('F', 'G'), 0.0143), (('E', 'H'), 0.0143),
                    # (('L', 'D'), 0.0143), (('N', 'M'), 0.0143), (('K', 'D'), 0.0143), (('L', 'H'), 0.0143),
                    # (('L', 'V'), 0.0143), (('L', 'R'), 0.0143)]
                elif line[:43] == 'Surface Fragment Oligomer1 Residue Number: ':
                    res_chain1 = int(line[43:].strip())
                elif line[:43] == 'Surface Fragment Oligomer2 Residue Number: ':
                    res_chain2 = int(line[43:].strip())
                    self.fragment_cluster_residue_d[cluster].add((res_chain1, res_chain2))
                    # self.fragment_cluster_residue_d[cluster]['pair'].add((res_chain1, res_chain2))
                elif line[:17] == 'Overlap Z-Value: ':
                    # try:
                    self.z_value_dict[cluster] = float(line[17:].strip())
                    # except ValueError:
                    #     print('%s has misisng Z-value in frag_info_file.txt' % self.__str__())
                    #     self.z_value_dict[cluster] = float(1.0)
                # elif line[:17] == 'Nanohedra Score: ':  # Depreciated
                #     nanohedra_score = float(line[17:].strip())
            #             if line[:39] == 'Unique Interface Fragment Match Count: ':
            #                 int_match = int(line[39:].strip())
            #             if line[:39] == 'Unique Interface Fragment Total Count: ':
            #                 int_total = int(line[39:].strip())
                elif line[:20] == 'ROT/DEGEN MATRIX PDB':
                    # ROT/DEGEN MATRIX PDB1: [[1.0, -0.0, 0], [0.0, 1.0, 0], [0, 0, 1]]
                    _matrix = np.array(eval(line[23:]))
                    self.transform_d[int(line[20:21])] = {'rot/deg': _matrix}  # dict[pdb# (1, 2)] = {'transform_type': matrix}
                elif line[:15] == 'INTERNAL Tx PDB':
                    # INTERNAL Tx PDB1: [0, 0, 45.96406061067895]
                    _matrix = np.array(eval(line[18:]))
                    self.transform_d[int(line[15:16].strip())]['tx_int'] = _matrix
                elif line[:18] == 'SETTING MATRIX PDB':
                    # SETTING MATRIX PDB1: [[0.707107, 0.408248, 0.57735], [-0.707107, 0.408248, 0.57735], [0.0, -0.816497, 0.57735]]
                    _matrix = np.array(eval(line[21:].strip()))
                    self.transform_d[int(line[18:19])]['setting'] = _matrix
                elif line[:21] == 'REFERENCE FRAME Tx PDB':
                    # REFERENCE FRAME Tx PDB1: None
                    _matrix = np.array(eval(line[24:].strip()))
                    self.transform_d[int(line[21:22])]['tx_ref'] = _matrix
                elif 'Residue-Level Summation Score:' in line:
                    self.nanohedra_score = float(line[30:].strip())

        # for cluster in self.fragment_cluster_residue_d:
        #     self.fragment_cluster_residue_d[cluster]['pair'] = list(set(residue_cluster_d[cluster]['pair']))

    @handle_errors_f(errors=(FileNotFoundError, ))
    def gather_pose_metrics(self):
        """Gather docking metrics from Nanohedra output"""
        with open(os.path.join(self.path, PUtils.pose_file), 'r') as f:
            pose_info_file_lines = f.readlines()
            for line in pose_info_file_lines:
                if line[:15] == 'DOCKED POSE ID:':
                    self.pose_id = line[15:].strip()
                elif line[:38] == 'Unique Mono Fragments Matched (z<=1): ':
                    self.high_quallity_int_residues_matched = int(line[38:].strip())
                elif line[:31] == 'Unique Mono Fragments Matched: ':
                    self.int_residues_matched = int(line[31:].strip())
                elif line[:36] == 'Unique Mono Fragments at Interface: ':
                    self.int_residues_total = int(line[36:].strip())
                elif line[:25] == 'Interface Matched (%): ' % '%':
                    self.percent_fragment = float(line[25:].strip())
                elif line[:20] == 'ROT/DEGEN MATRIX PDB':
                    data = eval(line[22:].strip())
                    self.transform_d[int(line[20:21])] = {'rot/deg': np.array(data)}  # dict[pdb# (1, 2)] = {'transform_type': matrix}
                elif line[:15] == 'INTERNAL Tx PDB':  # without PDB1 or PDB2
                    data = eval(line[17:].strip())
                    if data == 'None':
                        self.transform_d[int(line[22:23])]['tx_ref'] = np.array([0, 0, 0])
                    else:
                        self.transform_d[int(line[15:16])]['tx_int'] = np.array(data)

                elif line[:18] == 'SETTING MATRIX PDB':  # 'SETTING MATRIX PDB1: '
                    data = eval(line[20:].strip())
                    self.transform_d[int(line[18:19])]['setting'] = np.array(data)
                # elif line[:19] == 'SETTING MATRIX PDB2':
                #     data = eval(line[20:].strip())
                #     self.transform_d[int(line[22:23])]['setting'] = np.array(data)

                elif line[:22] == 'REFERENCE FRAME Tx PDB':
                    data = eval(line[24:].strip())
                    if data == 'None':
                        self.transform_d[int(line[22:23])]['tx_ref'] = np.array([0, 0, 0])
                    else:
                        self.transform_d[int(line[22:23])]['tx_ref'] = np.array(data)
                # elif line[:23] == 'REFERENCE FRAME Tx PDB2':
                #     data = eval(line[24:].strip())
                #     if data != 'None':
                #         self.transform_d[int(line[22:23].strip())]['tx_ref2'] = np.array(data)
                #     else:
                #         self.transform_d[int(line[22:23].strip())]['tx_ref2'] = np.array([0, 0, 0])

                elif 'Nanohedra Score:' in line:  # res_lev_sum_score
                    self.nanohedra_score = float(line[16:].rstrip())
                elif 'CRYST1 RECORD:' in line:
                    self.cryst_record = line[15:].strip()

                elif line[:31] == 'Canonical Orientation PDB1 Path':
                    self.cannonical_pdb1 = line[:31].strip()
                elif line[:31] == 'Canonical Orientation PDB2 Path':
                    self.cannonical_pdb2 = line[:31].strip()

    def pdb_input_parameters(self):
        return self.pdb_dir1_path, self.pdb_dir2_path  # args[0:2]

    def symmetry_parameters(self):
        return self.sym_entry_number, self.oligomer_symmetry_1, self.oligomer_symmetry_2, self.design_symmetry_pg  # args[3:7]

    def rotation_parameters(self):
        return self.rot_range_deg_pdb1, self.rot_range_deg_pdb2, self.rot_step_deg1, self.rot_step_deg2  # args[9:13]

    def degeneracy_parameters(self):
        return self.degen1, self.degen2  # args[-2:]

    def degen_and_rotation_parameters(self):
        return self.degeneracy_parameters(), self.rotation_parameters()

    def compute_last_rotation_state(self):
        number_steps1 = self.rot_range_deg_pdb1 / self.rot_step_deg1
        number_steps2 = self.rot_range_deg_pdb2 / self.rot_step_deg1

        return int(number_steps1), int(number_steps2)


def set_up_directory_objects(design_list, mode='design', symmetry=None):
    """Create DesignDirectory objects from a directory iterable. Add symmetry if using DesignDirectory strings"""
    return [DesignDirectory(design, mode=mode, symmetry=symmetry) for design in design_list]


def set_up_pseudo_design_dir(path, directory, score):  # changed 9/30/20 to locate paths of interest at .path
    pseudo_dir = DesignDirectory(path, auto_structure=False)
    # pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(path)
    pseudo_dir.design_pdbs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir


class DesignError(Exception):  # TODO make error messages one line instead of string iteration
    # SymDesignUtils.DesignError: ('I', 'n', 'v', 'a', 'l', 'i', 'd', ' ', 'P', 'D', 'B', ' ', 'i', 'n', 'p', 'u', 't',
    # ',', ' ', 'n', 'o', ' ', 'S', 'E', 'Q', 'R', 'E', 'S', ' ', 'r', 'e', 'c', 'o', 'r', 'd', ' ', 'f', 'o', 'u', 'n',
    # 'd')

    def __init__(self, message):
        self.args = message