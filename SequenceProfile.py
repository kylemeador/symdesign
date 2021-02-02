import math
import os
import subprocess
import time
from copy import deepcopy  # copy,
from glob import glob, iglob
from itertools import chain

from Bio import SeqIO  # import write, parse
from Bio import pairwise2
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import IUPACData

import CmdUtils as CUtils
import PathUtils as PUtils
from classes.Fragment import FragmentDB
from utils.MysqlPython import Mysql
from Query.PDB import get_sequence_by_entity_id
from SymDesignUtils import handle_errors_f, unpickle, get_all_base_root_paths, DesignError, start_log, \
    residue_interaction_graph  # logger,


logger = start_log(name=__name__, level=2)  # was from SDUtils logger, but moved here per standard suggestion

# Globals
index_offset = 1
alph_3_aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
aa_weight_counts_dict = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
                         'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
add_fragment_profile_instructions = 'To add fragment information, call PDB.score_interface_with_nanohedra()'  # Todo


class SequenceProfile:
    def __init__(self, structure=None, log=None, **kwargs):
        super().__init__(**kwargs)  # log=log,
        # if log:
        #     self.log = log
        # else:
        #     print('SequenceProfile starting log')
        #     self.log = start_log()

        # self.sequence = None
        self.structure = None  # should be initialized with a Entity/Chain obj, could be used with PDB obj
        self.structure_sequence = None
        self.sequence_source = None
        self.sequence_file = None
        self.pssm_file = None
        self.evolutionary_profile = {}  # position specific scoring matrix
        self.design_pssm_file = None
        self.profile = {}  # design specific scoring matrix
        self.frag_db = None
        self.fragment_queries = {}
        self.fragment_map = {}
        self.fragment_profile = {}
        self.interface_data_file = None
        self.alpha = {}

        if structure:
            self.set_structure(structure)

    @classmethod
    def from_structure(cls, structure=None):
        return cls(structure=structure)

    @property
    def name(self):
        return self.structure.name

    @name.setter
    def name(self, name):
        self.structure.name = name

    @property
    def profile_length(self):
        # Todo in future, this wouldn't handle profile size modifications. Same issue as Structure number_of properties
        try:
            return self._profile_length
        except AttributeError:
            self.set_profile_length()
            return self._profile_length

    @profile_length.setter
    def profile_length(self, length):
        self._profile_length = length

    @property
    def entity_offset(self):
        # Todo this wouldn't handle profile size modifications
        try:
            return self._entity_offset
        except AttributeError:
            self.entity_offset = self.structure.get_residues()[0].number - 1
            return self._entity_offset

    @entity_offset.setter
    def entity_offset(self, offset):
        self._entity_offset = offset

    def set_structure(self, structure):
        self.structure = structure
        self.structure_sequence = self.structure.get_structure_sequence()

    def set_profile_length(self):
        self.profile_length = len(self.profile)

    # def retrieve_sequence_from_api(self, entity_id=None):  # Unused
    #     self.sequence = get_sequence_by_entity_id(entity_id)
    #     self.sequence_source = 'seqres'

    # def get_reference_sequence(self):
    #
    # def get_structure_sequence(self):
    #     self.sequence = self.structure.get_structure_sequence()
    #     self.sequence_source = 'atom'
    #
    # def find_sequence(self):
    #     # if self.sequence_source:
    #     #     if self.sequence_source == 'seqres':
    #     self.get_reference_sequence()
    #     #     else:
    #     self.get_structure_sequence()
    #
    # @property
    # def sequence(self):
    #     try:
    #         return self._sequence
    #     except AttributeError:
    #         self.find_sequence()
    #         return self._sequence
    #
    # @sequence.setter
    # def sequence(self, sequence):
    #     self._sequence = sequence

    def add_profile(self, evolution=True, out_path=os.getcwd(),
                    fragments=True, fragment_observations=None, entities=None, pdb_numbering=True, **kwargs):
        """Add the evolutionary and fragment profiles onto the SequenceProfile

        Keyword Args:
            fragment_source=None (list):
            alignment_type=None (str): Either 'mapped' or 'paired'. Indicates how entity and fragments are aligned
            out_path=os.getcwd() (str): Location where sequence files should be written
            pdb_numbering=True (bool):
        """
        if fragments:  # add fragment information to the SequenceProfile
            if fragment_observations:  # fragments should be provided, then distributed to the SequenceProfile
                if entities:
                    self.add_fragment_query(entity1=entities[0], entity2=entities[1], query=fragment_observations,
                                            pdb_numbering=pdb_numbering)
                    # if pdb_numbering:  # Renumber to Pose residue numbering
                    #     fragment_source = self.renumber_fragments_to_pose(fragment_source)
                    #     for idx, fragment in enumerate(fragment_source):
                    #         fragment['mapped'] = self.structure.residue_number_from_pdb(fragment['mapped'])
                    #         fragment['paired'] = self.structure.residue_number_from_pdb(fragment['paired'])
                    #         fragment_source[idx] = fragment
                    # self.assign_fragments(fragments=fragment_source, alignment_type=alignment_type)
                else:
                    self.log.error('%s: Argument \'entities\' (tuple) is required if fragment_observations are provided'
                                   % self.add_profile.__name__)
                    return None

            elif self.fragment_map and self.frag_db:  # fragments have already been added, connect DB info
                self.frag_db.get_cluster_info(ids=[fragment['cluster'] for idx_d in self.fragment_map.values()
                                                   for frag_obs in idx_d.values() for fragment in frag_obs])
            # process fragment profile from self.fragment_map or self.fragment_query
            self.add_fragment_profile()  # fragment_source=fragment_source, alignment_type=frag_alignment_type)

        if evolution:  # add evolutionary information to the SequenceProfile
            # DesignDirectory.path could be removed as the input oligomers should be perfectly symmetric so the tx won't
            # matter for each entity. right? Todo check this assumption...
            self.add_evolutionary_profile(out_path=out_path, **kwargs)
            # TODO currently using self.structure.reference_sequence which could be ATOM, could be SEQRES.
            #  For the next step, we NEED ATOM. Must resize the profile! Can use the sequence alignment from sequnce
            #  processing
            self.combine_ssm(boltzmann=True, favor_fragments=fragments)

    # def atom_profile(self):  # Todo resolve
    #     """Make the atom profile from the structure.sequence and the profile"""
    #     atom_profile = []
    #     return atom_profile

    def verify_profile(self):  # Todo
        """Check Pose and Profile for equality before proceeding"""
        rerun, second, failed = False, False, False
        while not failed:
            if self.profile_length != self.structure.number_of_residues:  # ():
                self.log.warning('%s: Profile and Pose are different lengths!\nProfile=%d, Pose=%d. '
                                 'Generating a new profile' % (self.structure.file_path, self.profile_length,
                                                               self.structure.number_of_residues))
                rerun = True

            # if not rerun:
            else:
                # Check sequence from Pose and self.profile to compare identity before proceeding
                # pssm_res, pose_res = {}, {}
                for idx, residue in enumerate(self.structure.get_residues(), 1):
                    profile_residue_type = self.profile[idx]['type']
                    # pssm_res[residue.number] = self.profile[idx]['type']
                    pose_residue_type = IUPACData.protein_letters_3to1[residue.type.title()]
                    # pose_res[residue.number] = IUPACData.protein_letters_3to1[residue.type.title()]
                    if profile_residue_type != pose_residue_type:
                        self.log.warning(
                            '%s: Profile and Pose sequences mismatched!\nResidue %d: Profile=%s, Pose=%s. Generating a'
                            ' new profile' % (self.structure.file_path, residue.number, profile_residue_type,
                                              pose_residue_type))
                        rerun = True
                        break

            if rerun:
                if second:
                    # self.log.error('%s: Profile Generation got stuck, design aborted' % des_dir.path)
                    raise DesignError('Profile Generation got stuck, design aborted')
                    # raise DesignError('%s: Profile Generation got stuck, design aborted' % des_dir.path)
                self.add_profile(force=True)
                second = True

    def add_evolutionary_profile(self, out_path=os.getcwd(), profile_source='hhblits', force=False):
        """Add the evolutionary profile to the entity. Profile is generated through a position specific search of
        homologous protein sequences (evolutionary)

        Keyword Args:
            out_path=os.getcwd() (str): Location where sequence files should be written
            profile_source='hhblits' (str): One of 'hhblits' or 'psiblast'
        Sets:
            self.evolutionary_profile
        """
        if profile_source not in ['hhblits', 'psiblast']:
            raise DesignError('%s: Profile generation only possible from \'hhblits\' or \'psiblast\', not %s'
                              % (self.add_evolutionary_profile.__name__, profile_source))

        if force:
            self.sequence_file = None
            self.pssm_file = None
        else:
            # Check to see if the files of interest already exist
            temp_file = os.path.join(out_path, '%s.hold' % self.name)
            out_put_file_search = glob(os.path.join(out_path, '%s.*' % self.name))
            if not out_put_file_search:  # found nothing -> []
                with open(temp_file, 'w') as f:
                    self.log.info('Fetching \'%s\' sequence data.\n' % self.name)
            else:
                for seq_file in out_put_file_search:
                    if seq_file == os.path.join(out_path, '%s.hold' % self.name):
                        self.log.info('Waiting for \'%s\' profile generation...' % self.name)
                        while not os.path.exists(os.path.join(out_path, '%s.hmm' % self.name)):
                            time.sleep(20)
                            if int(time.time()) - int(os.path.getmtime(temp_file)) > 600:  # > 10 minutes have passed
                                os.remove(temp_file)
                                raise DesignError('%s: Generation of the profile for %s took longer than the time '
                                                  'limit. Job killed!'
                                                  % (self.add_evolutionary_profile.__name__, self.name))
                    elif seq_file == os.path.join(out_path, '%s.hmm' % self.name):
                        self.pssm_file = os.path.join(out_path, seq_file)
                        self.log.debug('%s PSSM Files=%s' % (self.name, self.pssm_file))
                        break
                    elif seq_file == os.path.join(out_path, '%s.fasta' % self.name):
                        self.sequence_file = seq_file
                        self.log.debug('%s PSSM File not yet created' % self.name)
                    else:
                        self.log.error('Found the file \'%s\' which was not expected in %s' % (seq_file, out_path))
                        #     with open(temp_file, 'w') as f:
                        #         f.write('Started fetching data. Process will resume once data is gathered\n')

        if not self.sequence_file:
            # Extract/Format Sequence Information
            # if not self.sequence:  # Use structure sequence if atom sequence is missing
            #     self.sequence = self.get_structure_sequence()
            #     self.sequence = self.structure.get_structure_sequence()
            #     self.sequence_source = 'atom'
            self.log.debug('%s Sequence=%s' % (self.name, self.structure.reference_sequence))

            # make self.sequence_file
            self.write_fasta_file(self.structure.reference_sequence, name='%s' % self.name, out_path=out_path)
            self.log.debug('%s fasta file: %s' % (self.name, self.sequence_file))

        if not self.pssm_file:
            # Make PSSM of sequence
            self.log.info('Generating PSSM file for %s' % self.name)
            if profile_source == 'psiblast':
                self.psiblast(out_path=out_path)
                self.parse_psiblast_pssm()
            else:
                self.hhblits(out_path=out_path)
                self.parse_hhblits_pssm()

    def psiblast(self, out_path=None, remote=False):
        """Generate an position specific scoring matrix using PSI-BLAST subprocess

        Keyword Args:
            outpath=None (str): Disk location where generated file should be written
            remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
        Returns:
            outfile_name (str): Name of the file generated by psiblast
            p (subprocess): Process object for monitoring progress of psiblast command
        """
        self.pssm_file = os.path.join(out_path, '%s.pssm' % str(self.name))

        cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', self.sequence_file + '.fasta', '-out_ascii_pssm',
               self.pssm_file, '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']  # Todo ^ iters
        if remote:
            cmd.append('-remote')
        else:
            cmd.append('-num_threads')
            cmd.append('8')

        p = subprocess.Popen(cmd)
        p.communicate()

    @handle_errors_f(errors=(FileNotFoundError,))
    def parse_psiblast_pssm(self):
        """Take the contents of a pssm file, parse, and input into a sequence dictionary.

        Returns:
            pose_dict (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        with open(self.pssm_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_data = line.strip().split()
            if len(line_data) == 44:
                residue_number = int(line_data[0])
                self.evolutionary_profile[residue_number] = deepcopy(aa_counts_dict)
                for i, aa in enumerate(alph_3_aa_list, 22):  # pose_dict[residue_number], 22):
                    # Get normalized counts for pose_dict
                    self.evolutionary_profile[residue_number][aa] = (int(line_data[i]) / 100.0)
                self.evolutionary_profile[residue_number]['lod'] = {}
                for i, aa in enumerate(alph_3_aa_list, 2):
                    self.evolutionary_profile[residue_number]['lod'][aa] = line_data[i]
                self.evolutionary_profile[residue_number]['type'] = line_data[1]
                self.evolutionary_profile[residue_number]['info'] = float(line_data[42])
                self.evolutionary_profile[residue_number]['weight'] = float(line_data[43])

    def hhblits(self, threads=CUtils.hhblits_threads, out_path=os.getcwd()):
        """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

        Keyword Args:
            threads=CUtils.hhblits_threads (int): Number of cpu's to use for the process
            outpath=None (str): Disk location where generated file should be written
        Returns:
            outfile_name (str): Name of the file generated by hhblits
            p (subprocess): Process object for monitoring progress of hhblits command
        """

        self.pssm_file = os.path.join(out_path, '%s.hmm' % str(self.name))

        cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', self.sequence_file, '-ohhm', self.pssm_file, '-v', '1',
               '-cpu', str(threads)]
        self.log.info('%s Profile Command: %s' % (self.name, subprocess.list2cmdline(cmd)))
        p = subprocess.Popen(cmd)
        p.communicate()

    @handle_errors_f(errors=(FileNotFoundError,))
    def parse_hhblits_pssm(self, null_background=True):
        """Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order

        Returns:
            (dict): {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                'weight': 0.00}, {...}}
        """
        dummy = 0.00
        null_bg = {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228,
                   'I': 0.0534,
                   'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599,
                   'S': 0.0791,
                   'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}  # 'uniclust30_2018_08'

        def to_freq(value):
            if value == '*':
                # When frequency is zero
                return 0.0001
            else:
                # Equation: value = -1000 * log_2(frequency)
                freq = 2 ** (-int(value) / 1000)
                return freq

        with open(self.pssm_file, 'r') as f:
            lines = f.readlines()

        read = False
        for line in lines:
            if not read:
                if line[0:1] == '#':
                    read = True
            else:
                if line[0:4] == 'NULL':
                    if null_background:
                        # use the provided null background from the profile search
                        background = line.strip().split()
                        null_bg = {i: {} for i in alph_3_aa_list}
                        for i, aa in enumerate(alph_3_aa_list, 1):
                            null_bg[aa] = to_freq(background[i])

                if len(line.split()) == 23:
                    items = line.strip().split()
                    residue_number = int(items[1])
                    self.evolutionary_profile[residue_number] = {}
                    for i, aa in enumerate(IUPACData.protein_letters, 2):
                        self.evolutionary_profile[residue_number][aa] = to_freq(items[i])
                    self.evolutionary_profile[residue_number]['lod'] = get_lod(self.evolutionary_profile[residue_number], null_bg)
                    self.evolutionary_profile[residue_number]['type'] = items[0]
                    self.evolutionary_profile[residue_number]['info'] = dummy
                    self.evolutionary_profile[residue_number]['weight'] = dummy

    def combine_pssm(self, pssms):
        """Combine a list of PSSMs incrementing the residue number in each additional PSSM

        Args:
            pssms (list(dict)): List of PSSMs to concatenate
        Sets
            self.evolutionary_profile (dict): Using the list of input PSSMs, make a concatenated PSSM
        """
        # combined_pssm = {}
        new_key = 1
        for i in range(len(pssms)):
            for old_key in sorted(pssms[i].keys()):
                self.evolutionary_profile[new_key] = pssms[i][old_key]
                new_key += 1
        # return combined_pssm

    def combine_fragment_profile(self, fragment_profiles):
        """Combine a list of fragment profiles incrementing the residue number in each additional fragment profile

        Args:
            fragment_profiles (list(dict)): List of fragment profiles to concatentate
        Sets
            self.fragment_profile (dict): To a concatentated fragment profile from the input fragment profiles
        """
        new_key = 1
        for i in range(len(fragment_profiles)):
            for old_key in sorted(fragment_profiles[i].keys()):
                self.fragment_profile[new_key] = fragment_profiles[i][old_key]
                new_key += 1

    def combine_profile(self, profiles):
        """Combine a list of DSSMs incrementing the residue number in each additional DSSM

        Args:
            profiles (list(dict)): List of DSSMs to concatentate
        Sets
            self.profile (dict): Using the list of input DSSMs, make a concatentated DSSM
        """
        new_key = 1
        for i in range(len(profiles)):
            for old_key in sorted(profiles[i].keys()):
                self.profile[new_key] = profiles[i][old_key]
                new_key += 1

    def write_fasta_file(self, sequence, name=None, out_path=os.getcwd()):
        """Write a fasta file from sequence(s)

        Keyword Args:
            name=None (str): The name of the file to output
            out_path=os.getcwd() (str): The location on disk to output file
        Returns:
            (str): The name of the output file
        """
        if not name:
            name = self.name
        self.sequence_file = os.path.join(out_path, '%s.fasta' % name)
        with open(self.sequence_file, 'w') as outfile:
            outfile.write('>%s\n%s\n' % (name, sequence))
            # outfile.write('>%s\n%s\n' % (name, self.structure_sequence))

        return self.sequence_file

    def renumber_fragments_to_pose(self, fragments):
        for idx, fragment in enumerate(fragments):
            # if self.structure.residue_from_pdb_numbering():
            # only assign the new fragment number info to the fragments if the residue is found
            map_pose_number = self.structure.residue_number_from_pdb(fragment['mapped'])
            fragment['mapped'] = map_pose_number if map_pose_number else fragment['mapped']
            pair_pose_number = self.structure.residue_number_from_pdb(fragment['paired'])
            fragment['paired'] = pair_pose_number if pair_pose_number else fragment['paired']
            # fragment['mapped'] = self.structure.residue_number_from_pdb(fragment['mapped'])
            # fragment['paired'] = self.structure.residue_number_from_pdb(fragment['paired'])
            fragments[idx] = fragment

        return fragments

    def return_entity_interface_metrics(self):
        """NOT YET ACCURATE TODO split the fragment info before the interface metric calculation...
        From the calculated fragment queries, return the interface metrics attributed to each entity"""
        metrics_by_query = self.return_interface_metrics()

        # prepopulate the dictionary with all queried entities
        entity_metrics_d = {query_pair[0]: {} for query_pair in metrics_by_query}
        entity_metrics_d.update({query_pair[1]: {} for query_pair in metrics_by_query})
        # separate the metrics from a query to an entity
        for query_pair, metrics in metrics_by_query.items():
            for entity in query_pair:
                if entity_metrics_d[entity]:
                    for key in entity_metrics_d[entity]:
                        entity_metrics_d[entity][key] += metrics[key]
                else:
                    entity_metrics_d[entity] = metrics

        return entity_metrics_d

    def return_interface_metrics(self):
        """From the calculated fragment queries, return the interface metrics for each pair
        Todo Expand to other metrics after design?
        Return the various metrics calculated by overlapping fragments at the interface of two proteins

        Returns:
            (dict): {query1: {all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap,
            central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d}, ... }
        """
        interface_metrics_d = {}
        for query, fragment_matches in self.fragment_queries.items():
            all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
                central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d = \
                get_fragment_metrics(fragment_matches)
            # total_residues, percent_interface_matched, percent_interface_covered,

            interface_metrics_d[query] = {'nanohedra_score': all_residue_score,
                                          'nanohedra_score_central': center_residue_score,
                                          'fragments': fragment_matches,
                                          # 'fragment_cluster_ids': ','.join(fragment_indices),
                                          'multiple_fragment_ratio': multiple_frag_ratio,
                                          'number_fragment_residues_all': total_residues_with_fragment_overlap,
                                          'number_fragment_residues_central': central_residues_with_fragment_overlap,
                                          # 'total_interface_residues': total_residues,
                                          'number_fragments': len(fragment_matches),
                                          # 'percent_residues_fragment_all': percent_interface_covered,
                                          # 'percent_residues_fragment_center': percent_interface_matched,
                                          'percent_fragment_helix': fragment_content_d['1'],
                                          'percent_fragment_strand': fragment_content_d['2'],
                                          'percent_fragment_coil': fragment_content_d['3'] + fragment_content_d['4']
                                          + fragment_content_d['5']}

        return interface_metrics_d

    # def return_fragment_info(self):
    #     clusters, residue_numbers, match_scores = [], [], []
    #     for query_pair, fragments in self.fragment_queries.items():
    #         for query_idx, entity_name in enumerate(query_pair):
    #             clusters.extend([fragment['cluster'] for fragment in fragments])
    #             clusters.extend([fragment['cluster'] for fragment in fragments])
    #             clusters.extend([fragment['cluster'] for fragment in fragments])
    #             clusters.extend([fragment['cluster'] for fragment in fragments])

    def add_fragment_query(self, entity1=None, entity2=None, query=None, pdb_numbering=False):
        """This funcion has all sorts of logic pitfalls and may be more trouble than it alleviates. How easy would it
        be to Todo refactor code to deal with the chain info from the frag match file?"""
        if pdb_numbering:  # Renumber self.fragment_map and self.fragment_profile to Pose residue numbering
            query = self.renumber_fragments_to_pose(query)
            # for idx, fragment in enumerate(fragment_source):
            #     fragment['mapped'] = self.structure.residue_number_from_pdb(fragment['mapped'])
            #     fragment['paired'] = self.structure.residue_number_from_pdb(fragment['paired'])
            #     fragment_source[idx] = fragment
            if entity1 and entity2 and query:
                self.fragment_queries[(entity1, entity2)] = query
        else:
            entity_pairs = [(self.structure.entity_from_residue(fragment['mapped']),
                             self.structure.entity_from_residue(fragment['paired'])) for fragment in query]
            if all([all(pair) for pair in entity_pairs]):
                for entity_pair, fragment in zip(entity_pairs, query):
                    if entity_pair in self.fragment_queries:
                        self.fragment_queries[entity_pair].append(fragment)
                    else:
                        self.fragment_queries[entity_pair] = [fragment]
            else:
                raise DesignError('%s: Couldn\'t locate Pose Entities passed by residue number. Are the residues in '
                                  'Pose Numbering? This may be occuring due to fragment queries performed on the PDB '
                                  'and not explicitly searching using pdb_numbering = True. Retry with the appropriate'
                                  ' modifications' % self.add_fragment_query.__name__)

    # fragments
    # [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]
    def add_fragment_profile(self):  # , fragment_source=None, alignment_type=None):
        # v now done at the pose_level
        # self.assign_fragments(fragments=fragment_source, alignment_type=alignment_type)
        if self.fragment_map:
            self.generate_fragment_profile()
            self.simplify_fragment_profile()  # =False added for issm pickling and analyze_output
        else:
            idx_to_alignment_type = {0: 'mapped', 1: 'paired'}
            if self.fragment_queries:
                for query_pair, fragments in self.fragment_queries.items():
                    for query_idx, entity in enumerate(query_pair):
                        if entity.name == self.structure.name:
                            # add to fragment map
                            self.assign_fragments(fragments=fragments, alignment_type=idx_to_alignment_type[query_idx])
            else:
                self.log.error('No fragment information associated with the Entity %s yet! You must add to the profile '
                               'otherwise only evolutionary values will be used.\n%s'
                               % (self.name, add_fragment_profile_instructions))
                return None

    def find_alpha(self, alpha=0.5):
        """Find fragment contribution to design with cap at alpha

        Takes self.fragment_map
            (dict) {1: {-2: [{'chain': 'mapped', 'cluster': '1_2_123', 'match': 0.6}, ...], -1: [], ...},
                    2: {}, ...}
        To identify cluster_id and chain thus returning fragment contribution from the fragment database statistics
            (dict): {cluster_id1: [[mapped_index_average, paired_index_average, {max_weight_counts_mapped}, {_paired}],
                                   total_fragment_observations],
                     cluster_id2: ...,
                     frequencies: {'A': 0.11, ...}}
        Then makes self.alpha
            (dict): {1: 0.5, 2: 0.321, ...}

        Keyword Args:
            alpha=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
        """
        assert 0 <= alpha <= 1, '%s: Alpha parameter must be between 0 and 1' % self.find_alpha.__name__
        alignment_type_to_idx = {'mapped': 0, 'paired': 1}
        match_score_average = 0.5  # when fragment pair rmsd equal to the mean cluster rmsd
        bounded_floor = 0.2
        fragment_stats = self.frag_db.get_db_statistics()
        for entry in self.fragment_profile:
            # match score is bounded between 1 and 0.2
            # match_sum = sum([self.fragment_map[entry][index][obs]['match'] for index in self.fragment_map[entry]
            match_sum = sum(obs['match'] for index_values in self.fragment_map[entry].values() for obs in index_values)
            contribution_total = sum(fragment_stats[self.get_cluster_id(obs['cluster'], index=2)][0]
                                     [alignment_type_to_idx[obs['chain']]]
                                     for index_values in self.fragment_map[entry].values() for obs in index_values)

            # contribution_total = 0.0
            # for index in self.fragment_map[entry]:
            #     # for obs in self.fragment_map[entry][index]['cluster']: # WAS
            #     for obs in self.fragment_map[entry][index]:
            #         # get first two indices from the cluster_id
            #         # cluster_id = return_cluster_id_string(self.fragment_map[entry][index][obs]['fragment'], # WAS
            #         cluster_id = return_cluster_id_string(obs['cluster'], index_number=2)
            #
            #         # from the fragment statistics grab the average index weight for the observed chain alignment type
            #         contribution_total += fragment_stats[cluster_id][0][alignment_type_to_idx[obs['chain']]]

            # can't use the match count as the fragment index may have no useful residue information
            # count = len([1 for obs in self.fragment_map[entry][index] for index in self.fragment_map[entry]]) or 1
            # instead use # of fragments with SC interactions count from the frequency map
            count = self.fragment_profile[entry]['stats'][0]
            if count == 0:
                # ensure that match modifier is 0 so self.alpha[entry] is 0, as there is no fragment information here!
                count = match_sum * 5  # makes the match average = 0.5

            match_average = match_sum / float(count)
            # find the match modifier which spans from 0 to 1
            if match_average < match_score_average:
                match_modifier = ((match_average - bounded_floor) / (match_score_average - bounded_floor))
            else:
                match_modifier = match_score_average / match_score_average  # 1 is the maximum bound

            # get the average contribution of each fragment type
            stats_average = contribution_total / count
            # get entry average fragment weight. total weight for issm entry / count
            frag_weight_average = self.fragment_profile[entry]['stats'][1] / match_sum

            # modify alpha proportionally to cluster average weight and match_modifier
            if frag_weight_average < stats_average:  # if design frag weight is less than db cluster average weight
                self.alpha[entry] = alpha * (frag_weight_average / stats_average) * match_modifier
            else:
                self.alpha[entry] = alpha * match_modifier

    def connect_fragment_database(self, init=False, **kwargs):
        """Generate a new connection. Initialize the representative library by passing init=True"""
        self.frag_db = FragmentDatabase(init_db=init)  # source=source, location=location, init_db=init_db)

    def attach_fragment_database(self, db=None, **kwargs):
        """Attach an existing Fragment Database to the SequenceProfile or generate a new connection. Initialize the
        representative library by passing init=True"""
        #                         source='directory', location='biological_interfaces', length=5, init_db=True):
        if db:
            self.frag_db = db
        else:
            raise DesignError('%s: No fragment database connection was passed!'
                              % self.attach_fragment_database.__name__)

    def assign_fragments(self, fragments=None, alignment_type=None):
        """Distribute fragment information to self.fragment_map. One-indexed residue dictionary

        Keyword Args:
            fragments=None (list): The fragment list to assign to the sequence profile wiht format
            [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]
            alignment_type=None (str): Either mapped or paired
        """
        if alignment_type not in ['mapped', 'paired']:
            return None
        if not fragments:
            return None

        if not self.fragment_map:
            self.fragment_map = self.populate_design_dictionary(self.structure.number_of_residues,
                                                                [j for j in range(*self.frag_db.fragment_range)],
                                                                dtype=list)
            print('New fragment_map')
        print(fragments)
        print(self.name)
        print(self.entity_offset)
        for fragment in fragments:
            residue_number = fragment[alignment_type] - self.entity_offset
            for j in range(*self.frag_db.fragment_range):  # lower_bound, upper_bound
                self.fragment_map[residue_number + j][j].append({'chain': alignment_type,
                                                                 'cluster': fragment['cluster'],
                                                                 'match': fragment['match']})
        # remove entries which don't exist on protein because of fragment_index +- residues
        not_available = [residue_number for residue_number in self.fragment_map
                         if residue_number <= 0 or residue_number > self.structure.number_of_residues]
        for residue_number in not_available:
            self.log.debug('In \'%s\', residue %d is represented by a fragment but there is no Atom record for it. '
                           'Fragment index will be deleted.' % (self.name, residue_number))
            self.fragment_map.pop(residue_number)

        self.log.debug('Residue Cluster Map: %s' % str(self.fragment_map))

    def generate_fragment_profile(self):
        """Add frequency information to the fragment profile using parsed cluster information. Frequency information is
        added in a fragment index dependent manner. If multiple fragment indices are present in a single residue, a new
        observation is created for that fragment index.

        Converts a fragment_map with format:
            (dict): {0: {-2: [{'chain': 'mapped', 'cluster': '1_2_123', 'match': 0.6}, ...], -1: [], ...},
                     1: {}, ...}
        To a fragment_profile with format:
            (dict): {0: {-2: {O: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37], 'match': 0.6}, 1: {}}, -1: {}, ... },
                     1: {}, ...}
        """

        for residue_number, fragment_indices in self.fragment_map.items():
            self.fragment_profile[residue_number] = {}
            for frag_idx, fragments in fragment_indices.items():
                self.fragment_profile[residue_number][frag_idx] = {}
                # observation_d = {}
                for observation_idx, fragment in enumerate(fragments):
                    cluster_id = fragment['cluster']
                    freq_type = fragment['chain']
                    aa_freq = self.frag_db.retrieve_cluster_info(cluster=cluster_id, source=freq_type, index=frag_idx)
                    # {1_1_54: {'mapped': {aa_freq}, 'paired': {aa_freq}}, ...}
                    #  mapped/paired aa_freq = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1: {}, ...}
                    #  Where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
                    self.fragment_profile[residue_number][frag_idx][observation_idx] = aa_freq
                    self.fragment_profile[residue_number][frag_idx][observation_idx]['match'] = fragment['match']
                    # observation_d[obs_idx] = aa_freq
                    # observation_d[obs_idx]['match'] = fragment['match']
                # self.fragment_map[residue_number][frag_index] = observation_d

    def simplify_fragment_profile(self, keep_extras=False):
        """Take a multi-indexed, a multi-observation fragment frequency dictionary and flatten to single frequency for
        each amino acid. Weight the frequency of each observation by the fragment indexed, observation weight and the
        match between the fragment library and the observed fragment overlap

        Takes the fragment_map with format:
            (dict): {1: {-2: {0: 'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37], 'match': 0.6}}, 1: {}}, -1: {}, ... },
                     2: {}, ...}
                Dictionary containing fragment frequency and statistics across a design
        And makes into
            (dict): {1: {'A': 0.23, 'C': 0.01, ..., stats': [1, 0.37]}, 13: {...}, ...}
                Weighted average design dictionary combining all fragment profile information at a single residue where
                stats[0] is number of fragment observations at each residue, and stats[1] is the total fragment weight
                over the entire residue
        Keyword Args:
            keep_extras=True (bool): If true, keep values for all design dictionary positions that are missing data
        """
        no_design = []
        for residue, index_d in self.fragment_profile.items():
            total_fragment_weight = 0
            total_fragment_observations = 0
            for index, observation_d in index_d.items():
                if observation_d:
                    # sum the weight for each fragment observation
                    total_obs_weight = 0.0
                    total_obs_x_match_weight = 0.0
                    # total_match_weight = 0.0
                    for observation, fragment_frequencies in observation_d.items():
                        if fragment_frequencies:
                            total_obs_weight += fragment_frequencies['stats'][1]
                            total_obs_x_match_weight += fragment_frequencies['stats'][1] * fragment_frequencies['match']
                            # total_match_weight += self.fragment_profile[residue][index][obs]['match']

                    # Check if weights are associated with observations, if not side chain isn't significant!
                    if total_obs_weight > 0:
                        total_fragment_weight += total_obs_weight
                        obs_aa_dict = deepcopy(aa_weight_counts_dict)  # {'A': 0, 'C': 0, ..., 'stats': [0, 1]}
                        obs_aa_dict['stats'][1] = total_obs_weight
                        for obs in self.fragment_profile[residue][index]:
                            total_fragment_observations += 1
                            obs_x_match_weight = self.fragment_profile[residue][index][obs]['stats'][1] * \
                                self.fragment_profile[residue][index][obs]['match']
                            # match_weight = self.fragment_profile[residue][index][obs]['match']
                            # obs_weight = self.fragment_profile[residue][index][obs]['stats'][1]
                            for aa in self.fragment_profile[residue][index][obs]:
                                if aa not in ['stats', 'match']:
                                    # Multiply OBS and MATCH
                                    modification_weight = (obs_x_match_weight / total_obs_x_match_weight)
                                    # modification_weight = ((obs_weight + match_weight) /  # WHEN SUMMING OBS and MATCH
                                    #                        (total_obs_weight + total_match_weight))
                                    # modification_weight = (obs_weight / total_obs_weight)
                                    # Add all occurrences to summed frequencies list
                                    obs_aa_dict[aa] += self.fragment_profile[residue][index][obs][aa] * modification_weight
                        self.fragment_profile[residue][index] = obs_aa_dict
                    else:
                        self.fragment_profile[residue][index] = {}

            if total_fragment_weight > 0:
                res_aa_dict = deepcopy(aa_weight_counts_dict)
                res_aa_dict['stats'][1] = total_fragment_weight  # this is over all indices and observations
                res_aa_dict['stats'][0] = total_fragment_observations  # this is over all indices and observations
                for index in self.fragment_profile[residue]:
                    if self.fragment_profile[residue][index]:
                        index_weight = self.fragment_profile[residue][index]['stats'][1]  # total_obs_weight
                        for aa in self.fragment_profile[residue][index]:
                            if aa not in ['stats', 'match']:
                                # Add all occurrences to summed frequencies list
                                res_aa_dict[aa] += self.fragment_profile[residue][index][aa] * (
                                            index_weight / total_fragment_weight)
                self.fragment_profile[residue] = res_aa_dict
            else:
                # Add to list for removal from the design dict
                no_design.append(residue)

        if keep_extras:
            for residue in no_design:
                self.fragment_profile[residue] = aa_weight_counts_dict
        else:  # remove missing residues from dictionary
            for residue in no_design:
                self.fragment_profile.pop(residue)

    def combine_ssm(self, favor_fragments=True, boltzmann=False, alpha=0.5):
        """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

        All input must be zero indexed

        Takes self.evolutionary_profile
            (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                                   'weight': 0.00}, {...}}
                    PSIBLAST - {0: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
                                    'weight': 0.73}, {...}} CURRENTLY IMPOSSIBLE, NEED TO CHANGE LOD SCORE IN PARSING
        self.fragment_profile
            (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        and self.alpha
            (dict): {48: 0.5, 50: 0.321, ...}
        Keyword Args:
            favor_fragments=True (bool): Whether to favor fragment profile in the lod score of the resulting profile
            boltzmann=True (bool): Whether to weight the fragment profile by the Boltzmann probability.
                                   lod = z[i]/Z, Z = sum(exp(score[i]/kT))
                   If=False, residues are weighted by the residue local maximum lod score in a linear fashion.
            All lods are scaled to a maximum provided in the Rosetta REF2015 per residue reference weight.
            alpha=0.5 (float): The maximum alpha value to use, bounded between 0 and 1

        And outputs self.profile
            (dict): {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                         'weight': 0.00}, ...}} - combined PSSM dictionary
        """
        assert 0 <= alpha <= 1, '%s: Alpha parameter must be between 0 and 1' % self.combine_ssm.__name__
        # copy the evol profile to self.profile (design specific scoring matrix)
        self.profile = deepcopy(self.evolutionary_profile)
        # Combine fragment and evolutionary probability profile according to alpha parameter
        for entry in self.alpha:  # will be 0 unless self.fragment_profile is not {}
            for aa in IUPACData.protein_letters:
                self.profile[entry][aa] = (self.alpha[entry] * self.fragment_profile[entry][aa]) + \
                                          ((1 - self.alpha[entry]) * self.profile[entry][aa])
            self.log.info('Residue %d Combined evolutionary and fragment profile: %.0f%% fragment'
                          % (entry + index_offset, self.alpha[entry] * 100))

        if favor_fragments:
            # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
            # Used to weight fragments higher in design
            boltzman_energy = 1
            favor_seqprofile_score_modifier = 0.2 * CUtils.reference_average_residue_weight
            database_background_aa_frequencies = self.frag_db.get_db_aa_frequencies()

            null_residue = get_lod(database_background_aa_frequencies, database_background_aa_frequencies)
            null_residue = {aa: float(null_residue[aa]) for aa in null_residue}

            for entry in self.profile:
                self.profile[entry]['lod'] = null_residue  # Caution all reference same object
            for entry in self.fragment_profile:
                self.profile[entry]['lod'] = get_lod(self.fragment_profile[entry],
                                                     database_background_aa_frequencies, round_lod=False)
                # get the sum for the partition function
                partition, max_lod = 0, 0.0
                for aa in self.profile[entry]['lod']:
                    if boltzmann:  # boltzmann probability distribution scaling, lod = z[i]/Z, Z = sum(exp(score[i]/kT))
                        self.profile[entry]['lod'][aa] = math.exp(self.profile[entry]['lod'][aa] / boltzman_energy)
                        partition += self.profile[entry]['lod'][aa]
                    # linear scaling, remove any lod penalty
                    elif self.profile[entry]['lod'][aa] < 0:
                        self.profile[entry]['lod'][aa] = 0
                    # find the maximum/residue (local) lod score
                    if self.profile[entry]['lod'][aa] > max_lod:
                        max_lod = self.profile[entry]['lod'][aa]
                # takes the percent of max alpha for each entry multiplied by the standard residue scaling factor
                modified_entry_alpha = (self.alpha[entry] / alpha) * favor_seqprofile_score_modifier
                if boltzmann:
                    modifier = partition
                    modified_entry_alpha /= (max_lod / partition)
                else:
                    modifier = max_lod

                # weight the final lod score by the modifier and the scaling factor for the chosen method
                for aa in self.evolutionary_profile[entry]['lod']:
                    self.profile[entry]['lod'][aa] /= modifier  # get percent total (boltzman) or percent max (linear)
                    self.profile[entry]['lod'][aa] *= modified_entry_alpha  # scale by score modifier
                self.log.info('Residue %d Fragment lod ratio generated with alpha=%f'
                              % (entry + index_offset, self.alpha[entry] / alpha))

    def solve_consensus(self, fragment_source=None, alignment_type=None):
        # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER

        # frag_cluster_residue_d = DesignDirectory.gather_pose_metrics(init=True)  Call this function with it
        # ^ Format: {'1_2_24': [(78, 87, ...), ...], ...}
        # Todo Can also re-score the interface upon Pose loading and return this information
        # template_pdb = DesignDirectory.source NOW self.pdb

        # v Used for central pair fragment mapping of the biological interface generated fragments
        cluster_freq_tuple_d = {cluster: fragment_source[cluster]['freq'] for cluster in fragment_source}
        # cluster_freq_tuple_d = {cluster: {cluster_residue_d[cluster]['freq'][0]: cluster_residue_d[cluster]['freq'][1]}
        #                         for cluster in cluster_residue_d}

        # READY for all to all fragment incorporation once fragment library is of sufficient size # TODO all_frags
        # TODO freqs are now separate
        cluster_freq_d = {cluster: self.format_frequencies(fragment_source[cluster]['freq'])
                          for cluster in fragment_source}  # orange mapped to cluster tag
        cluster_freq_twin_d = {cluster: self.format_frequencies(fragment_source[cluster]['freq'], flip=True)
                               for cluster in fragment_source}  # orange mapped to cluster tag
        frag_cluster_residue_d = {cluster: fragment_source[cluster]['pair'] for cluster in fragment_source}

        frag_residue_object_d = residue_number_to_object(self.structure, frag_cluster_residue_d)

        # Parse Fragment Clusters into usable Dictionaries and Flatten for Sequence Design
        # # TODO all_frags
        cluster_residue_pose_d = residue_object_to_number(frag_residue_object_d)
        # self.log.debug('Cluster residues pose number:\n%s' % cluster_residue_pose_d)
        # # ^{cluster: [(78, 87, ...), ...]...}
        residue_freq_map = {residue_set: cluster_freq_d[cluster] for cluster in cluster_freq_d
                            for residue_set in cluster_residue_pose_d[cluster]}  # blue
        # ^{(78, 87, ...): {'A': {'S': 0.02, 'T': 0.12}, ...}, ...}
        # make residue_freq_map inverse pair frequencies with cluster_freq_twin_d
        residue_freq_map.update({tuple(residue for residue in reversed(residue_set)): cluster_freq_twin_d[cluster]
                                 for cluster in cluster_freq_twin_d for residue_set in residue_freq_map})

        # Construct CB Tree for full interface atoms to map residue residue contacts
        # total_int_residue_objects = [res_obj for chain in names for res_obj in int_residue_objects[chain]] Now above
        # interface = PDB(atoms=[atom for residue in total_int_residue_objects for atom in residue.atoms])
        # interface_tree = residue_interaction_graph(interface)
        # interface_cb_indices = interface.get_cb_indices(InclGlyCA=True)

        interface_residue_edges = {}
        for idx, residue_contacts in enumerate(interface_tree):
            if interface_tree[idx].tolist() != list():
                residue = interface.all_atoms[interface_cb_indices[idx]].residue_number
                contacts = {interface.all_atoms[interface_cb_indices[contact_idx]].residue_number
                            for contact_idx in interface_tree[idx]}
                interface_residue_edges[residue] = contacts - {residue}
        # ^ {78: [14, 67, 87, 109], ...}  green

        # solve for consensus residues using the residue graph
        self.assign_fragments(fragments=fragment_source, alignment_type=alignment_type)
        consensus_residues = {}
        all_pose_fragment_pairs = list(residue_freq_map.keys())
        residue_cluster_map = offset_index(self.cluster_map)  # change so it is one-indexed
        # for residue in residue_cluster_map:
        for residue, partner in all_pose_fragment_pairs:
            for idx, cluster in residue_cluster_map[residue]['cluster']:
                if idx == 0:  # check if the fragment index is 0. No current information for other pairs 07/24/20
                    for idx_p, cluster_p in residue_cluster_map[partner]['cluster']:
                        if idx_p == 0:  # check if the fragment index is 0. No current information for other pairs 07/24/20
                            if residue_cluster_map[residue]['chain'] == 'mapped':
                                # choose first AA from AA tuple in residue frequency d
                                aa_i, aa_j = 0, 1
                            else:  # choose second AA from AA tuple in residue frequency d
                                aa_i, aa_j = 1, 0
                            for pair_freq in cluster_freq_tuple_d[cluster]:
                                # if cluster_freq_tuple_d[cluster][k][0][aa_i] in frag_overlap[residue]:
                                if residue in frag_overlap:  # edge case where fragment has no weight but it is center res
                                    if pair_freq[0][aa_i] in frag_overlap[residue]:
                                        # if cluster_freq_tuple_d[cluster][k][0][aa_j] in frag_overlap[partner]:
                                        if partner in frag_overlap:
                                            if pair_freq[0][aa_j] in frag_overlap[partner]:
                                                consensus_residues[residue] = pair_freq[0][aa_i]
                                                break  # because pair_freq's are sorted we end at the highest matching pair

        # # Set up consensus design # TODO all_frags
        # # Combine residue fragment information to find residue sets for consensus
        # # issm_weights = {residue: final_issm[residue]['stats'] for residue in final_issm}
        final_issm = offset_index(final_issm)  # change so it is one-indexed
        frag_overlap = fragment_overlap(final_issm, interface_residue_edges, residue_freq_map)  # all one-indexed

        # consensus = SDUtils.consensus_sequence(dssm)
        self.log.debug('Consensus Residues only:\n%s' % consensus_residues)
        self.log.debug('Consensus:\n%s' % consensus)
        for n, name in enumerate(names):
            for residue in int_res_numbers[name]:  # one-indexed
                mutated_pdb.mutate_to(names[name](n), residue,
                                      res_id=IUPACData.protein_letters_1to3[consensus[residue]].upper())
        mutated_pdb.write(des_dir.consensus_pdb)
        # mutated_pdb.write(consensus_pdb)
        # mutated_pdb.write(consensus_pdb, cryst1=cryst)

    @staticmethod
    def generate_mutations(mutant, reference, offset=True, blanks=False, termini=False, reference_gaps=False,
                           only_gaps=False):
        """Create mutation data in a typical A5K format. One-indexed dictionary keys, mutation data accessed by 'from'
        and 'to' keywords. By default all gapped sequences are excluded from returned mutations

        For PDB file comparison, mutant should be crystal sequence (ATOM), reference should be expression sequence
        (SEQRES). only_gaps=True will return only the gapped area while blanks=True will return all differences between
        the alignment sequences. termini=True returns missing alignments at the termini

        Args:
            mutant (str): Mutant sequence. Will be in the 'to' key
            reference (str): Wild-type sequence or sequence to reference mutations against. Will be in the 'from' key
        Keyword Args:
            offset=True (bool): Whether sequences are different legnths. Creates a new alignment
            blanks=False (bool): Whether to include indices that are outside the reference sequence or missing residues
            termini=False (bool): Whether to include indices that are outside the reference sequence boundaries
            reference_gaps=False (bool): Whether to include indices with missing residues inside the reference sequence
            only_gaps=False (bool): Whether to only include all indices that are missing residues
        Returns:
            (dict): {index: {'from': 'A', 'to': 'K'}, ...}
        """
        # TODO change function name/order of mutant and reference arguments to match logic with 'from' 37 'to' framework
        if offset:
            alignment = generate_alignment(mutant, reference)
            align_seq_1 = alignment[0][0]
            align_seq_2 = alignment[0][1]
        else:
            align_seq_1 = mutant
            align_seq_2 = reference

        # Extract differences from the alignment
        starting_index_of_seq2 = align_seq_2.find(reference[0])
        ending_index_of_seq2 = starting_index_of_seq2 + align_seq_2.rfind(reference[-1])  # find offset end_index
        mutations = {}
        for i, (seq1_aa, seq2_aa) in enumerate(zip(align_seq_1, align_seq_2), -starting_index_of_seq2 + index_offset):
            if seq1_aa != seq2_aa:
                mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
                # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))

        remove_mutation_list = []
        if only_gaps:  # remove the actual mutations
            for entry in mutations:
                if entry > 0 or entry <= ending_index_of_seq2:
                    if mutations[entry]['to'] != '-':
                        remove_mutation_list.append(entry)
            blanks = True
        if blanks:  # if blanks is True, leave all types of blanks, if blanks is False check for requested types
            termini, reference_gaps = True, True
        if not termini:  # Remove indices outside of sequence 2
            for entry in mutations:
                if entry < 0 or entry > ending_index_of_seq2:
                    remove_mutation_list.append(entry)
        if not reference_gaps:  # Remove indices inside sequence 2 where sequence 1 is gapped
            for entry in mutations:
                if entry > 0 or entry <= ending_index_of_seq2:
                    if mutations[entry]['to'] == '-':
                        remove_mutation_list.append(entry)

        for entry in remove_mutation_list:
            if entry in mutations:
                mutations.pop(entry)

        return mutations

    @staticmethod
    def generate_alignment(seq1, seq2, matrix='BLOSUM62'):
        """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences*

        Returns:

        """
        # _matrix = getattr(matlist, matrix)
        _matrix = substitution_matrices.load(matrix)
        gap_penalty = -10
        gap_ext_penalty = -1
        # Create sequence alignment
        return pairwise2.align.localds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)

    def generate_design_mutations(self, all_design_files, wild_type_file, pose_num=False):
        """From a wild-type sequence (original PDB structure), and a collection of structure sequences that have
        undergone design (Atom sequences), generate 'A5K' style mutation data

        Args:
            all_design_files (list): PDB files on disk to extract sequence info and compare
            wild_type_file (str): PDB file on disk which contains a reference sequence
        Returns:
            mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
        """
        # pdb_dict = {'ref': PDB(file=wild_type_file)}
        # for file_name in all_design_files:
        #     pdb = PDB(file=file_name)
        #     pdb.name = os.path.splitext(os.path.basename(file_name))[0])
        #     pdb_dict[pdb.name] = pdb
        #
        # return extract_sequence_from_pdb(pdb_dict, mutation=True, pose_num=pose_num)  # , offset=False)

    @staticmethod
    def populate_design_dictionary(n, alphabet, dtype=dict, zero_index=False):
        """Return a dictionary with n elements, each integer key containing another dictionary with the items in
        alphabet as keys. By default, the data type inside the alphabet dictionary is a dictionary. Can set dtype as any
        viable type including list, set, tuple, or int. If dtype is int, 0 will be added to use for counting

        Args:
            n (int): number of residues in a design
            alphabet (iter): alphabet of interest
        Keyword Args:
            dtype=dict (object): The type of object present in the interior dictionary
            counts=False (bool): If True, include an integer placeholder for counting
            zero_index=False (bool): If True, return the dictionary with zero indexing
         Returns:
             (dict): {1: {alph1: {}, alph2: {}, ...}, 2: {}, ...}
                Custom length, 0 indexed dictionary with residue number keys
         """
        if zero_index:
            offset = 0
        else:
            offset = index_offset

        # if counts:
        #     return {residue + offset: {i: 0 for i in alphabet} for residue in range(n)}
        # else:
        return {residue + offset: {i: dtype() for i in alphabet} for residue in range(n)}

    @staticmethod
    def get_lod(aa_freq, background, round_lod=True):
        """Get the lod scores for an aa frequency distribution compared to a background frequency
        Args:
            aa_freq (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
            background (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
        Keyword Args:
            round_lod=True (bool): Whether or not to round the lod values to an integer
        Returns:
             (dict): {'A': 2, 'C': -9, 'D': -1, ...}
        """
        lods = {aa: None for aa in aa_freq}
        for aa in aa_freq:
            if aa not in ['stats', 'match']:
                lods[aa] = float((2.0 * math.log2(aa_freq[aa] / background[aa])))  # + 0.0
                if aa_freq[aa] == 0 or lods[aa] < -9:
                    lods[aa] = -9
                if round_lod:
                    lods[aa] = round(lods[aa])

        return lods

    @staticmethod
    def make_pssm_file(pssm_dict, name, out_path=os.getcwd()):
        """Create a PSI-BLAST format PSSM file from a PSSM dictionary. Assumes residue numbering is correct!

        Args:
            pssm_dict (dict): A pssm dictionary which has the fields: all aa's ('A', 'C', ...), 'lod', 'type', 'info', 'weight'
            name (str): The name of the file including the extension
        Keyword Args:
            out_path=os.getcwd() (str): A specific location to write the file to
        Returns:
            (str): Disk location of newly created .pssm file
        """
        if not pssm_dict:
            return None

        lod_freq, counts_freq = False, False
        separation_string1, separation_string2 = 3, 3
        if type(pssm_dict[0]['lod']['A']) == float:
            lod_freq = True
            separation_string1 = 4
        if type(pssm_dict[0]['A']) == float:
            counts_freq = True

        header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in alph_3_aa_list) \
                 + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in alph_3_aa_list) + '\n'
        footer = ''
        out_file = os.path.join(out_path, name)
        with open(out_file, 'w') as f:
            f.write(header)
            for res in pssm_dict:
                aa_type = pssm_dict[res]['type']
                lod_string = ''
                if lod_freq:
                    for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                        lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
                else:
                    for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                        lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
                counts_string = ''
                if counts_freq:
                    for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                        counts_string += '{:>3.0f} '.format(math.floor(pssm_dict[res][aa] * 100))
                else:
                    for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                        counts_string += '{:>3d} '.format(pssm_dict[res][aa])
                info = pssm_dict[res]['info']
                weight = pssm_dict[res]['weight']
                line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res, aa_type, lod_string,counts_string,
                                                                                 round(info, 4), round(weight, 4))
                f.write(line)
            f.write(footer)

        return out_file

    @staticmethod
    def format_frequencies(frequency_list, flip=False):
        """Format list of paired frequency data into parsable paired format

        Args:
            frequency_list (list): [(('D', 'A'), 0.0822), (('D', 'V'), 0.0685), ...]
        Keyword Args:
            flip=False (bool): Whether to invert the mapping of internal tuple
        Returns:
            (dict): {'A': {'S': 0.02, 'T': 0.12}, ...}
        """
        if flip:
            i, j = 1, 0
        else:
            i, j = 0, 1
        freq_d = {}
        for tup in frequency_list:
            aa_mapped = tup[0][i]  # 0
            aa_paired = tup[0][j]  # 1
            freq = tup[1]
            if aa_mapped in freq_d:
                freq_d[aa_mapped][aa_paired] = freq
            else:
                freq_d[aa_mapped] = {aa_paired: freq}

        return freq_d


def overlap_consensus(issm, aa_set):
    """Find the overlap constrained consensus sequence

    Args:
        issm (dict): {1: {'A': 0.1, 'C': 0.0, ...}, 14: {...}, ...}
        aa_set (dict): {residue: {'A', 'I', 'M', 'V'}, ...}
    Returns:
        (dict): {23: 'T', 29: 'A', ...}
    """
    consensus = {}
    for res in aa_set:
        max_freq = 0.0
        for aa in aa_set[res]:
            # if max_freq < issm[(res, partner)][]:
            if issm[res][aa] > max_freq:
                max_freq = issm[res][aa]
                consensus[res] = aa

    return consensus


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
            self.location = PUtils.frag_directory[location]  # location
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
            print('No SQL DB connected yet!')  # Todo
            return None
        else:
            for file in os.listdir(self.location):
                if file.endswith('statistics.pkl'):
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
                if index and source in ['mapped', 'paired']:
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
            print('No SQL DB connected yet!')  # Todo
            return None
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
            index = [cluster_id[:1], cluster_id[1:2], cluster_id[2:]]
        else:
            index = cluster_id.split('_')

        info = [index[i] for i in range(index)]

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


def get_db_statistics(database):
    """Retrieve summary statistics for a specific fragment database

    Args:
        database (str): Disk location of a fragment database
    Returns:
        stats (dict): {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
            ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
    """
    for file in os.listdir(database):
        if file.endswith('statistics.pkl'):
            return unpickle(os.path.join(database, file))

    return None  # Should never be called


def get_db_aa_frequencies(database):
    """Retrieve database specific interface background AA frequencies

    Args:
        database (str): Location of database on disk
    Returns:
        (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    """
    return get_db_statistics(database)['frequencies']


def get_cluster_dicts(db='biological_interfaces', id_list=None):  # TODO Rename
    """Generate an interface specific scoring matrix from the fragment library

    Args:
    Keyword Args:
        info_db=PUtils.biological_fragmentDB
        id_list=None: [1_2_24, ...]
    Returns:
         cluster_dict: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    """
    info_db = PUtils.frag_directory[db]
    if id_list is None:
        directory_list = get_all_base_root_paths(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        cluster_dict[os.path.basename(cluster)] = unpickle(filename)

    return cluster_dict


def return_cluster_id_string(cluster_rep, index_number=3):
    while len(cluster_rep) < 3:
        cluster_rep += '0'
    if len(cluster_rep.split('_')) != 3:
        index = [cluster_rep[:1], cluster_rep[1:2], cluster_rep[2:]]
    else:
        index = cluster_rep.split('_')

    info = []
    n = 0
    for i in range(index_number):
        info.append(index[i])
        n += 1
    while n < 3:
        info.append('0')
        n += 1

    return '_'.join(info)


def parameterize_frag_length(length):
    """Generate fragment length range parameters for use in fragment functions"""
    _range = math.floor(length / 2)
    if length % 2 == 1:
        return 0 - _range, 0 + _range + index_offset
    else:
        logger.critical('%d is an even integer which is not symmetric about a single residue. '
                        'Ensure this is what you want and modify %s' % (length, parameterize_frag_length.__name__))
        raise DesignError('Function not supported: Even fragment length \'%d\'' % length)


def format_frequencies(frequency_list, flip=False):
    """Format list of paired frequency data into parsable paired format

    Args:
        frequency_list (list): [(('D', 'A'), 0.0822), (('D', 'V'), 0.0685), ...]
    Keyword Args:
        flip=False (bool): Whether to invert the mapping of internal tuple
    Returns:
        (dict): {'A': {'S': 0.02, 'T': 0.12}, ...}
    """
    if flip:
        i, j = 1, 0
    else:
        i, j = 0, 1
    freq_d = {}
    for tup in frequency_list:
        aa_mapped = tup[0][i]  # 0
        aa_paired = tup[0][j]  # 1
        freq = tup[1]
        if aa_mapped in freq_d:
            freq_d[aa_mapped][aa_paired] = freq
        else:
            freq_d[aa_mapped] = {aa_paired: freq}

    return freq_d


def fragment_overlap(residues, interaction_graph, freq_map):
    """Take fragment contact list to find the possible AA types allowed in fragment pairs from the contact list

    Args:
        residues (iter): Iterable of residue numbers
        interaction_graph (dict): {52: [54, 56, 72, 206], ...}
        freq_map (dict): {(78, 87, ...): {'A': {'S': 0.02, 'T': 0.12}, ...}, ...}
    Returns:
        overlap (dict): {residue: {'A', 'I', 'M', 'V'}, ...}
    """
    overlap = {}
    for res in residues:
        overlap[res] = set()
        if res in interaction_graph:  # check for existence as some fragment info is not in the interface set
            # overlap[res] = set()
            for partner in interaction_graph[res]:
                if (res, partner) in freq_map:
                    overlap[res] |= set(freq_map[(res, partner)].keys())

    for res in residues:
        if res in interaction_graph:  # check for existence as some fragment info is not in the interface set
            for partner in interaction_graph[res]:
                if (res, partner) in freq_map:
                    overlap[res] &= set(freq_map[(res, partner)].keys())

    return overlap


def populate_design_dict(n, alph, counts=False):
    """Return a dictionary with n elements and alph subelements.

    Args:
        n (int): number of residues in a design
        alph (iter): alphabet of interest
    Keyword Args:
        counts=False (bool): If true include an integer placeholder for counting
     Returns:
         (dict): {0: {alph1: {}, alph2: {}, ...}, 1: {}, ...}
            Custom length, 0 indexed dictionary with residue number keys
     """
    if counts:
        return {residue: {i: 0 for i in alph} for residue in range(n)}
    else:
        return {residue: {i: dict() for i in alph} for residue in range(n)}


def offset_index(dictionary, to_zero=False):
    """Modify the index of a sequence dictionary. Default is to one-indexed. to_zero=True gives zero-indexed"""
    if to_zero:
        return {residue - index_offset: dictionary[residue] for residue in dictionary}
    else:
        return {residue + index_offset: dictionary[residue] for residue in dictionary}


def residue_object_to_number(residue_dict):  # TODO supplement with names info and pull out by names
    """Convert sets of PDB.Residue objects to residue numbers

    Args:
        residue_dict (dict): {'key1': [(residue1_ca_atom, residue2_ca_atom, ...), ...] ...}
    Returns:
        residue_dict (dict): {'key1': [(78, 87, ...),], ...} - Entry mapped to residue sets
    """
    for entry in residue_dict:
        pairs = []
        # for _set in range(len(residue_dict[entry])):
        for j, _set in enumerate(residue_dict[entry]):
            residue_num_set = []
            # for i, residue in enumerate(residue_dict[entry][_set]):
            for residue in _set:
                resi_number = residue.residue_number
                # resi_object = PDB.Residue(pdb.getResidueAtoms(pdb.chain_id_list[i], residue)).ca
                # assert resi_object, DesignError('Residue \'%s\' missing from PDB \'%s\'' % (residue, pdb.filepath))
                residue_num_set.append(resi_number)
            pairs.append(tuple(residue_num_set))
        residue_dict[entry] = pairs

    return residue_dict


def convert_to_residue_cluster_map(residue_cluster_dict, frag_range):
    """Make a residue and cluster/fragment index map

    Args:
        residue_cluster_dict (dict): {'1_2_45': [(residue1_ca_atom, residue2_ca_atom), ...] ...}
        frag_range (dict): A range of the fragment size to search over. Ex: (-2, 3) for fragments of length 5
    Returns:
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
            Where the key is the 0 indexed residue id
    """
    cluster_map = {}
    for cluster in residue_cluster_dict:
        for pair in range(len(residue_cluster_dict[cluster])):
            for i, residue_atom in enumerate(residue_cluster_dict[cluster][pair]):
                # for each residue in map add the same cluster to the range of fragment residue numbers
                residue_num = residue_atom.residue_number - index_offset  # zero index
                for j in range(*frag_range):
                    if residue_num + j not in cluster_map:
                        if i == 0:
                            cluster_map[residue_num + j] = {'chain': 'mapped', 'cluster': []}
                        else:
                            cluster_map[residue_num + j] = {'chain': 'paired', 'cluster': []}
                    cluster_map[residue_num + j]['cluster'].append((j, cluster))

    return cluster_map


def deconvolve_clusters(cluster_dict, design_dict, cluster_map):
    """Add frequency information from a fragment database to a design dictionary

    The frequency information is added in a fragment index dependent manner. If multiple fragment indices are present in
    a single residue, a new observation is created for that fragment index.

    Args:
        cluster_dict (dict): {1_1_54: {'mapped': {aa_freq}, 'paired': {aa_freq}}, ...}
            mapped/paired aa_freq = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1: {}, ...}
                Where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
        design_dict (dict): {0: {-2: {'A': 0.0, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...], 'match': 1.2}, ...}
    Returns:
        (dict): {0: {-2: {O: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37], 'match': 1.2}, 1: {}}, -1: {}, ... },
                 1: {}, ...}
    """

    for resi in cluster_map:
        dict_type = cluster_map[resi]['chain']
        observation = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}  # counter for each residue obs along the fragment index
        for index_cluster_pair in cluster_map[resi]['cluster']:
            aa_freq = cluster_dict[index_cluster_pair[1]][dict_type][index_cluster_pair[0]]
            # Add the aa_freq from cluster to the residue/frag_index/observation
            try:
                design_dict[resi][index_cluster_pair[0]][observation[index_cluster_pair[0]]] = aa_freq
                design_dict[resi][index_cluster_pair[0]][observation[index_cluster_pair[0]]]['match'] = \
                    cluster_map[resi]['match']
            except KeyError:
                raise DesignError('Missing residue %d in %s.' % (resi, deconvolve_clusters.__name__))
            observation[index_cluster_pair[0]] += 1

    return design_dict


def flatten_for_issm(design_cluster_dict, keep_extras=True):
    """Take a multi-observation, mulit-fragment index, fragment frequency dictionary and flatten to single frequency

    Args:
        design_cluster_dict (dict): {0: {-2: {'A': 0.1, 'C': 0.0, ...}, -1: {}, ... }, 1: {}, ...}
            Dictionary containing fragment frequency and statistics across a design sequence
    Keyword Args:
        keep_extras=True (bool): If true, keep values for all design dictionary positions that are missing fragment data
    Returns:
        design_cluster_dict (dict): {0: {'A': 0.1, 'C': 0.0, ...}, 13: {...}, ...}
            Weighted average design dictionary combining all fragment profile information at a single residue
    """
    no_design = []
    for res in design_cluster_dict:
        total_residue_weight = 0
        num_frag_weights_observed = 0
        for index in design_cluster_dict[res]:
            if design_cluster_dict[res][index] != dict():
                total_obs_weight = 0
                for obs in design_cluster_dict[res][index]:
                    total_obs_weight += design_cluster_dict[res][index][obs]['stats'][1]
                if total_obs_weight > 0:
                    total_residue_weight += total_obs_weight
                    obs_aa_dict = deepcopy(aa_weight_counts_dict)
                    obs_aa_dict['stats'][1] = total_obs_weight
                    for obs in design_cluster_dict[res][index]:
                        num_frag_weights_observed += 1
                        obs_weight = design_cluster_dict[res][index][obs]['stats'][1]
                        for aa in design_cluster_dict[res][index][obs]:
                            if aa != 'stats':
                                # Add all occurrences to summed frequencies list
                                obs_aa_dict[aa] += design_cluster_dict[res][index][obs][aa] * (obs_weight /
                                                                                               total_obs_weight)
                    design_cluster_dict[res][index] = obs_aa_dict
                else:
                    # Case where no weights associated with observations (side chain not structurally significant)
                    design_cluster_dict[res][index] = dict()

        if total_residue_weight > 0:
            res_aa_dict = deepcopy(aa_weight_counts_dict)
            res_aa_dict['stats'][1] = total_residue_weight
            res_aa_dict['stats'][0] = num_frag_weights_observed
            for index in design_cluster_dict[res]:
                if design_cluster_dict[res][index] != dict():
                    index_weight = design_cluster_dict[res][index]['stats'][1]
                    for aa in design_cluster_dict[res][index]:
                        if aa != 'stats':
                            # Add all occurrences to summed frequencies list
                            res_aa_dict[aa] += design_cluster_dict[res][index][aa] * (index_weight / total_residue_weight)
            design_cluster_dict[res] = res_aa_dict
        else:
            # Add to list for removal from the design dict
            no_design.append(res)

    # Remove missing residues from dictionary
    if keep_extras:
        for res in no_design:
            design_cluster_dict[res] = aa_weight_counts_dict
    else:
        for res in no_design:
            design_cluster_dict.pop(res)

    return design_cluster_dict


def psiblast(query, outpath=None, remote=False):  # UNUSED
    """Generate an position specific scoring matrix using PSI-BLAST subprocess

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
        remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
    Returns:
        outfile_name (str): Name of the file generated by psiblast
        p (subprocess): Process object for monitoring progress of psiblast command
    """
    # I would like the background to come from Uniref90 instead of BLOSUM62 #TODO
    if outpath is not None:
        outfile_name = os.path.join(outpath, query + '.pssm')
        direct = outpath
    else:
        outfile_name = query + '.hmm'
        direct = os.getcwd()
    if query + '.pssm' in os.listdir(direct):
        cmd = ['echo', 'PSSM: ' + query + '.pssm already exists']
        p = subprocess.Popen(cmd)

        return outfile_name, p

    cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
           '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
    if remote:
        cmd.append('-remote')
    else:
        cmd.append('-num_threads')
        cmd.append('8')

    p = subprocess.Popen(cmd)

    return outfile_name, p


def hhblits(query, threads=CUtils.hhblits_threads, outpath=os.getcwd()):
    """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

    Args:
        query (str): Basename of the sequence to use as a query, intended for use as pdb
        threads (int): Number of cpu's to use for the process
    Keyword Args:
        outpath=None (str): Disk location where generated file should be written
    Returns:
        outfile_name (str): Name of the file generated by hhblits
        p (subprocess): Process object for monitoring progress of hhblits command
    """

    outfile_name = os.path.join(outpath, os.path.splitext(os.path.basename(query))[0] + '.hmm')

    cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(threads)]
    logger.info('%s Profile Command: %s' % (query, subprocess.list2cmdline(cmd)))
    p = subprocess.Popen(cmd)

    return outfile_name, p


@handle_errors_f(errors=(FileNotFoundError, ))
def parse_pssm(file):
    """Take the contents of a pssm file, parse, and input into a pose profile dictionary.

    Resulting residue dictionary is zero-indexed
    Args:
        file (str): The name/location of the file on disk
    Returns:
        pose_dict (dict): Dictionary containing residue indexed profile information
            Ex: {0: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                {...}}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    for line in lines:
        line_data = line.strip().split()
        if len(line_data) == 44:
            resi = int(line_data[0]) - index_offset
            pose_dict[resi] = deepcopy(aa_counts_dict)
            for i, aa in enumerate(alph_3_aa_list, 22):  # pose_dict[resi], 22):
                # Get normalized counts for pose_dict
                pose_dict[resi][aa] = (int(line_data[i]) / 100.0)
            pose_dict[resi]['lod'] = {}
            for i, aa in enumerate(alph_3_aa_list, 2):
                pose_dict[resi]['lod'][aa] = line_data[i]
            pose_dict[resi]['type'] = line_data[1]
            pose_dict[resi]['info'] = float(line_data[42])
            pose_dict[resi]['weight'] = float(line_data[43])

    return pose_dict


def get_lod(aa_freq_dict, bg_dict, round_lod=True):
    """Get the lod scores for an aa frequency distribution compared to a background frequency
    Args:
        aa_freq_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
        bg_dict (dict): {'A': 0.10, 'C': 0.0, 'D': 0.04, ...}
    Keyword Args:
        round_lod=True (bool): Whether or not to round the lod values to an integer
    Returns:
         lods (dict): {'A': 2, 'C': -9, 'D': -1, ...}
    """
    lods = {}
    iteration = 0
    for a in aa_freq_dict:
        if aa_freq_dict[a] == 0:
            lods[a] = -9
        elif a != 'stats':
            lods[a] = float((2.0 * math.log2(aa_freq_dict[a]/bg_dict[a])))  # + 0.0
            if lods[a] < -9:
                lods[a] = -9
            if round_lod:
                lods[a] = round(lods[a])
            iteration += 1

    return lods


@handle_errors_f(errors=(FileNotFoundError, ))
def parse_hhblits_pssm(file, null_background=True):
    # Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order
    dummy = 0.00
    null_bg = {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228, 'I': 0.0534,
               'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599, 'S': 0.0791,
               'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}  # 'uniclust30_2018_08'

    def to_freq(value):
        if value == '*':
            # When frequency is zero
            return 0.0001
        else:
            # Equation: value = -1000 * log_2(frequency)
            freq = 2 ** (-int(value)/1000)
            return freq

    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    read = False
    for line in lines:
        if not read:
            if line[0:1] == '#':
                read = True
        else:
            if line[0:4] == 'NULL':
                if null_background:
                    # use the provided null background from the profile search
                    background = line.strip().split()
                    null_bg = {i: {} for i in alph_3_aa_list}
                    for i, aa in enumerate(alph_3_aa_list, 1):
                        null_bg[aa] = to_freq(background[i])

            if len(line.split()) == 23:
                items = line.strip().split()
                resi = int(items[1]) - index_offset  # make zero index so dict starts at 0
                pose_dict[resi] = {}
                for i, aa in enumerate(IUPACData.protein_letters, 2):
                    pose_dict[resi][aa] = to_freq(items[i])
                pose_dict[resi]['lod'] = get_lod(pose_dict[resi], null_bg)
                pose_dict[resi]['type'] = items[0]
                pose_dict[resi]['info'] = dummy
                pose_dict[resi]['weight'] = dummy

    # Output: {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
    # 'weight': 0.00}, {...}}
    return pose_dict


def make_pssm_file(pssm_dict, name, outpath=os.getcwd()):
    """Create a PSI-BLAST format PSSM file from a PSSM dictionary

    Args:
        pssm_dict (dict): A pssm dictionary which has the fields 'A', 'C', (all aa's), 'lod', 'type', 'info', 'weight'
        name (str): The name of the file
    Keyword Args:
        outpath=cwd (str): A specific location to write the .pssm file to
    Returns:
        out_file (str): Disk location of newly created .pssm file
    """
    lod_freq, counts_freq = False, False
    separation_string1, separation_string2 = 3, 3
    if type(pssm_dict[0]['lod']['A']) == float:
        lod_freq = True
        separation_string1 = 4
    if type(pssm_dict[0]['A']) == float:
        counts_freq = True

    header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in alph_3_aa_list) \
             + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in alph_3_aa_list) + '\n'
    footer = ''
    out_file = os.path.join(outpath, name)  # + '.pssm'
    with open(out_file, 'w') as f:
        f.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            lod_string = ''
            if lod_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
            counts_string = ''
            if counts_freq:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3.0f} '.format(math.floor(pssm_dict[res][aa] * 100))
            else:
                for aa in alph_3_aa_list:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3d} '.format(pssm_dict[res][aa])
            info = pssm_dict[res]['info']
            weight = pssm_dict[res]['weight']
            line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + index_offset, aa_type, lod_string,
                                                                             counts_string, round(info, 4),
                                                                             round(weight, 4))
            f.write(line)
        f.write(footer)

    return out_file


def combine_pssm(pssms):
    """To a first pssm, append subsequent pssms incrementing the residue number in each additional pssm

    Args:
        pssms (list(dict)): List of pssm dictionaries to concatenate
    Returns:
        (dict): Concatenated PSSM
    """
    combined_pssm = {}
    new_key = 0
    for i in range(len(pssms)):
        # requires python 3.6+ to maintain sorted dictionaries
        # for old_key in pssms[i]:
        for old_key in sorted(list(pssms[i].keys())):
            combined_pssm[new_key] = pssms[i][old_key]
            new_key += 1

    return combined_pssm


def combine_ssm(pssm, issm, alpha, db='biological_interfaces', favor_fragments=True, boltzmann=False, a=0.5):
    """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

    All input must be zero indexed
    Args:
        pssm (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W',
            'info': 0.00, 'weight': 0.00}, {...}}
              PSIBLAST -  {0: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
                          'weight': 0.73}, {...}} CURRENTLY IMPOSSIBLE, NEED TO CHANGE THE LOD SCORE IN PARSING
        issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        alpha (dict): {48: 0.5, 50: 0.321, ...}
    Keyword Args:
        db='biological_interfaces': Disk location of fragment database
        favor_fragments=True (bool): Whether to favor fragment profile in the lod score of the resulting profile
        boltzmann=True (bool): Whether to weight the fragment profile by the Boltzmann probability. If false, residues
            are weighted by a local maximum over the residue scaled to a maximum provided in the standard Rosetta per
            residue reference weight.
        a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
    Returns:
        pssm (dict): {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
            'weight': 0.00}, ...}} - combined PSSM dictionary
    """

    # Combine fragment and evolutionary probability profile according to alpha parameter
    for entry in alpha:
        for aa in IUPACData.protein_letters:
            pssm[entry][aa] = (alpha[entry] * issm[entry][aa]) + ((1 - alpha[entry]) * pssm[entry][aa])
        logger.info('Residue %d Combined evolutionary and fragment profile: %.0f%% fragment'
                    % (entry + index_offset, alpha[entry] * 100))

    if favor_fragments:
        # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
        # Used to weight fragments higher in design
        boltzman_energy = 1
        favor_seqprofile_score_modifier = 0.2 * CUtils.reference_average_residue_weight
        db = PUtils.frag_directory[db]
        stat_dict_bkg = get_db_aa_frequencies(db)
        null_residue = get_lod(stat_dict_bkg, stat_dict_bkg)
        null_residue = {aa: float(null_residue[aa]) for aa in null_residue}

        for entry in pssm:
            pssm[entry]['lod'] = null_residue
        for entry in issm:
            pssm[entry]['lod'] = get_lod(issm[entry], stat_dict_bkg, round_lod=False)
            partition, max_lod = 0, 0.0
            for aa in pssm[entry]['lod']:
                # for use with a boltzman probability weighting, Z = sum(exp(score / kT))
                if boltzmann:
                    pssm[entry]['lod'][aa] = math.exp(pssm[entry]['lod'][aa] / boltzman_energy)
                    partition += pssm[entry]['lod'][aa]
                # remove any lod penalty
                elif pssm[entry]['lod'][aa] < 0:
                    pssm[entry]['lod'][aa] = 0
                # find the maximum/residue (local) lod score
                if pssm[entry]['lod'][aa] > max_lod:
                    max_lod = pssm[entry]['lod'][aa]
            modified_entry_alpha = (alpha[entry] / a) * favor_seqprofile_score_modifier
            if boltzmann:
                modifier = partition
                modified_entry_alpha /= (max_lod / partition)
            else:
                modifier = max_lod
            for aa in pssm[entry]['lod']:
                pssm[entry]['lod'][aa] /= modifier
                pssm[entry]['lod'][aa] *= modified_entry_alpha
            logger.info('Residue %d Fragment lod ratio generated with alpha=%f'
                        % (entry + index_offset, alpha[entry] / a))

    return pssm


def find_alpha(issm, cluster_map, db='biological_interfaces', a=0.5):
    """Find fragment contribution to design with cap at alpha

    Args:
        issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
        cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
    Keyword Args:
        db='biological_interfaces': Disk location of fragment database
        a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
    Returns:
        alpha (dict): {48: 0.5, 50: 0.321, ...}
    """
    db = PUtils.frag_directory[db]
    stat_dict = get_db_statistics(db)
    alpha = {}
    for entry in issm:  # cluster_map
        if cluster_map[entry]['chain'] == 'mapped':
            i = 0
        else:
            i = 1

        contribution_total, count = 0.0, 1
        # count = len([1 for obs in cluster_map[entry][index] for index in cluster_map[entry]]) or 1
        for count, residue_cluster_pair in enumerate(cluster_map[entry]['cluster'], 1):
            cluster_id = return_cluster_id_string(residue_cluster_pair[1], index_number=2)  # get first two indices
            contribution_total += stat_dict[cluster_id][0][i]  # get the average contribution of each fragment type
        stats_average = contribution_total / count
        entry_ave_frag_weight = issm[entry]['stats'][1] / count  # total weight for issm entry / number of fragments
        if entry_ave_frag_weight < stats_average:  # if design frag weight is less than db cluster average weight
            # modify alpha proportionally to cluster average weight
            alpha[entry] = a * (entry_ave_frag_weight / stats_average)
        else:
            alpha[entry] = a

    return alpha


def consensus_sequence(pssm):
    """Return the consensus sequence from a PSSM

    Args:
        pssm (dict): pssm dictionary
    Return:
        consensus_identities (dict): {1: 'M', 2: 'H', ...} One-indexed
    """
    consensus_identities = {}
    for residue in pssm:
        max_lod = 0
        max_res = pssm[residue]['type']
        for aa in alph_3_aa_list:
            if pssm[residue]['lod'][aa] > max_lod:
                max_lod = pssm[residue]['lod'][aa]
                max_res = aa
        consensus_identities[residue + index_offset] = max_res

    return consensus_identities


def sequence_difference(seq1, seq2, d=None, matrix='blosum62'):  # TODO AMS
    """Returns the sequence difference between two sequence iterators

    Args:
        seq1 (any): Either an iterable with residue type as array, or key, with residue type as d[seq1][residue]['type']
        seq2 (any): Either an iterable with residue type as array, or key, with residue type as d[seq2][residue]['type']
    Keyword Args:
        d=None (dict): The dictionary to look up seq1 and seq2 if they are keys and the iterable is a dictionary
        matrix='blosum62' (str): The type of matrix to score the sequence differences on
    Returns:
        (float): The computed sequence difference between seq1 and seq2
    """
    # s = 0
    if d:
        # seq1 = d[seq1]
        # seq2 = d[seq2]
        # for residue in d[seq1]:
            # s.append((d[seq1][residue]['type'], d[seq2][residue]['type']))
        pairs = [(d[seq1][residue]['type'], d[seq2][residue]['type']) for residue in d[seq1]]
    else:
        pairs = [(seq1_res, seq2[i]) for i, seq1_res in enumerate(seq1)]
            # s.append((seq1[i], seq2[i]))
    #     residue_iterator1 = seq1
    #     residue_iterator2 = seq2
    m = getattr(matlist, matrix)
    s = 0
    for tup in pairs:
        try:
            s += m[tup]
        except KeyError:
            s += m[(tup[1], tup[0])]

    return s


def remove_non_mutations(frequency_msa, residue_list):
    """Keep residues which are present in provided list

    Args:
        frequency_msa (dict): {0: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 1: {}, ...}
        residue_list (list): [15, 16, 18, 20, 34, 35, 67, 108, 119]
    Returns:
        mutation_dict (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
    """
    mutation_dict = {}
    for residue in frequency_msa:
        if residue in residue_list:
            mutation_dict[residue] = frequency_msa[residue]

    return mutation_dict


def return_consensus_design(frequency_sorted_msa):
    for residue in frequency_sorted_msa:
        if residue == 0:
            pass
        else:
            if len(frequency_sorted_msa[residue]) > 2:
                for alternative in frequency_sorted_msa[residue]:
                    # Prepare for Letter sorting SchemA
                    sequence_logo = None
            else:
                # DROP from analysis...
                frequency_sorted_msa[residue] = None


def pos_specific_jsd(msa, background):
    """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency

    Both msa_dictionary and background must be the same index
    Args:
        msa (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        background (dict): {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
            Must contain residue index with inner dictionary of single amino acid types
    Returns:
        divergence_dict (dict): {15: 0.732, 16: 0.552, ...}
    """
    return {residue: res_divergence(msa[residue], background[residue]) for residue in msa if residue in background}


def res_divergence(position_freq, bgd_freq, jsd_lambda=0.5):
    """Calculate residue specific Jensen-Shannon Divergence value

    Args:
        position_freq (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        bgd_freq (dict): {15: {'A': 0, 'R': 0, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        divergence (float): 0.732, Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    sum_prob1, sum_prob2 = 0, 0
    for aa in position_freq:
        p = position_freq[aa]
        q = bgd_freq[aa]
        r = (jsd_lambda * p) + ((1 - jsd_lambda) * q)
        if r == 0:
            continue
        if q != 0:
            prob2 = (q * math.log(q / r, 2))
            sum_prob2 += prob2
        if p != 0:
            prob1 = (p * math.log(p / r, 2))
            sum_prob1 += prob1
    divergence = round(jsd_lambda * sum_prob1 + (1 - jsd_lambda) * sum_prob2, 3)

    return divergence


def create_bio_msa(sequence_dict):
    """
    Args:
        sequence_dict (dict): {name: sequence, ...}
            ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        new_alignment (MultipleSeqAlignment): [SeqRecord(Seq("ACTGCTAGCTAG", generic_dna), id="Alpha"),
                                               SeqRecord(Seq("ACT-CTAGCTAG", generic_dna), id="Beta"), ...]
    """
    sequences = [SeqRecord(Seq(sequence_dict[name], annotations={'molecule_type': 'Protein'}), id=name)
                 for name in sequence_dict]
    # sequences = [SeqIO.SeqRecord(Seq(sequence_dict[name], generic_protein), id=name) for name in sequence_dict]
    new_alignment = MultipleSeqAlignment(sequences)

    return new_alignment


def read_fasta_file(file_name):
    """Returns an iterator of SeqRecords. Ex. [record1, record2, ...]"""
    return SeqIO.parse(file_name, 'fasta')


def write_fasta(sequence_records, file_name=None):  # Todo, consolidate (self.)write_fasta_file() with here
    """Writes an iterator of SeqRecords to a file with .fasta appended. The file name is returned"""
    if not file_name:
        return None
    if '.fasta' in file_name:
        file_name = file_name.rstrip('.fasta')
    SeqIO.write(sequence_records, '%s.fasta' % file_name, 'fasta')

    return '%s.fasta' % file_name


def concatenate_fasta_files(file_names, output='concatenated_fasta'):
    """Take multiple fasta files and concatenate into a single file"""
    seq_records = [read_fasta_file(file) for file in file_names]
    return write_fasta(list(chain.from_iterable(seq_records)), file_name=output)


def write_fasta_file(sequence, name, outpath=os.getcwd()):
    """Write a fasta file from sequence(s)

    Args:
        sequence (iterable): One of either list, dict, or string. If list, can be list of tuples(name, sequence),
            list of lists, etc. Smart solver using object type
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
    Returns:
        (str): The name of the output file
    """
    file_name = os.path.join(outpath, name + '.fasta')
    with open(file_name, 'w') as outfile:
        if type(sequence) is list:
            if type(sequence[0]) is list:  # where inside list is of alphabet (AA or DNA)
                for idx, seq in enumerate(sequence):
                    outfile.write('>%s_%d\n' % (name, idx))  # header
                    if len(seq[0]) == 3:  # Check if alphabet is 3 letter protein
                        outfile.write(' '.join(aa for aa in seq))
                    else:
                        outfile.write(''.join(aa for aa in seq))
            elif isinstance(sequence[0], str):
                outfile.write('>%s\n%s\n' % name, ' '.join(aa for aa in sequence))
            elif type(sequence[0]) is tuple:  # where seq[0] is header, seq[1] is seq
                outfile.write('\n'.join('>%s\n%s' % seq for seq in sequence))
            else:
                raise DesignError('Cannot parse data to make fasta')
        elif isinstance(sequence, dict):
            outfile.write('\n'.join('>%s\n%s' % (seq_name, sequence[seq_name]) for seq_name in sequence))
        elif isinstance(sequence, str):
            outfile.write('>%s\n%s\n' % (name, sequence))
        else:
            raise DesignError('Cannot parse data to make fasta')

    return file_name


def make_mutations(seq, mutations, find_orf=True):
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Args:
        seq (str): 'Wild-type' sequence to mutate
        mutations (dict): {mutation_index: {'from': AA, 'to': AA}, ...}
    Keyword Args:
        find_orf=True (bool): Whether or not to find the correct ORF for the mutations and the seq
    Returns:
        seq (str): The mutated sequence
    """
    # Seq can be either list or string
    if find_orf:
        offset = -find_orf_offset(seq, mutations)
        logger.info('Found ORF. Offset = %d' % -offset)
    else:
        offset = index_offset

    # zero index seq and 1 indexed mutation_dict
    index_errors = []
    for key in mutations:
        try:
            if seq[key - offset] == mutations[key]['from']:  # adjust seq for zero index slicing
                seq = seq[:key - offset] + mutations[key]['to'] + seq[key - offset + 1:]
            else:  # find correct offset, or mark mutation source as doomed
                index_errors.append(key)
        except IndexError:
            print(key - offset)
    if index_errors:
        logger.warning('Index errors:\n%s' % str(index_errors))

    return seq


def find_orf_offset(seq, mutations):
    """Using one sequence and mutation data, find the sequence offset which matches mutations closest

    Args:
        seq (str): 'Wild-type' sequence to mutate in 1 letter format
        mutations (dict): {mutation_index: {'from': AA, 'to': AA}, ...}
    Returns:
        orf_offset_index (int): The index to offset the sequence by in order to match the mutations the best
    """
    unsolvable = False
    # for idx, aa in enumerate(seq):
    #     if aa == 'M':
    #         met_offset_d[idx] = 0
    met_offset_d = {idx: 0 for idx, aa in enumerate(seq) if aa == 'M'}
    methionine_positions = list(met_offset_d.keys())

    while True:
        if met_offset_d:  # == dict():  # MET is missing/not the ORF start
            met_offset_d = {start_idx: 0 for start_idx in range(0, 50)}

        # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
        for test_orf_index in met_offset_d:
            for mutation_index in mutations:
                try:
                    if seq[test_orf_index + mutation_index - index_offset] == mutations[mutation_index]['from']:
                        met_offset_d[test_orf_index] += 1
                except IndexError:
                    break

        max_count = max(list(met_offset_d.values()))
        # Check if likely ORF has been identified (count < number mutations/2). If not, MET is missing/not the ORF start
        if max_count < len(mutations) / 2:
            if unsolvable:
                return 0  # TODO return not index change?
                # break
            unsolvable = True
            met_offset_d = {}
        else:
            for offset in met_offset_d:  # offset is index here
                if max_count == met_offset_d[offset]:
                    orf_offset_index = offset  # + index_offset  # change to one-index
                    break

            closest_met = None
            for met in methionine_positions:
                if met <= orf_offset_index:
                    closest_met = met
                else:
                    if closest_met is not None:
                        orf_offset_index = closest_met  # + index_offset # change to one-index
                    break

            break
            # orf_offset_index = met_offset_d[which_met_offset_counts.index(max_count)] - index_offset

    return orf_offset_index


def generate_alignment(seq1, seq2, matrix='blosum62'):
    """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences*

    Returns:

    """
    _matrix = getattr(matlist, matrix)
    gap_penalty = -10
    gap_ext_penalty = -1
    # Create sequence alignment
    return pairwise2.align.localds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)


def generate_mutations(mutant, reference, offset=True, blanks=False, termini=False, reference_gaps=False,
                       only_gaps=False):
    """Create mutation data in a typical A5K format. One-indexed dictionary keys, mutation data accessed by 'from' and
        'to' keywords. By default all gapped sequences are excluded from returned mutations

    For PDB file comparison, mutant should be crystal sequence (ATOM), reference should be expression sequence (SEQRES).
     only_gaps=True will return only the gapped area while blanks=True will return all differences between the alignment
      sequences. termini=True returns missing alignments at the termini
    Args:
        mutant (str): Mutant sequence. Will be in the 'to' key
        reference (str): Wild-type sequence or sequence to reference mutations against. Will be in the 'from' key
    Keyword Args:
        offset=True (bool): Whether sequences are different legnths. Creates a new alignment
        blanks=False (bool): Whether to include all indices that are outside the reference sequence or missing residues
        termini=False (bool): Whether to include indices that are outside the reference sequence boundaries
        reference_gaps=False (bool): Whether to include indices that are missing residues inside the reference sequence
        only_gaps=False (bool): Whether to only include all indices that are missing residues
    Returns:
        mutations (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    """
    # TODO change function name/order of mutant and reference arguments to match logic with 'from' 37 'to' framework
    if offset:
        alignment = generate_alignment(mutant, reference)
        align_seq_1 = alignment[0][0]
        align_seq_2 = alignment[0][1]
    else:
        align_seq_1 = mutant
        align_seq_2 = reference

    # Extract differences from the alignment
    starting_index_of_seq2 = align_seq_2.find(reference[0])
    ending_index_of_seq2 = starting_index_of_seq2 + align_seq_2.rfind(reference[-1])  # find offset end_index
    mutations = {}
    for i, (seq1_aa, seq2_aa) in enumerate(zip(align_seq_1, align_seq_2), -starting_index_of_seq2 + index_offset):
        if seq1_aa != seq2_aa:
            mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
            # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))

    remove_mutation_list = []
    if only_gaps:  # remove the actual mutations
        for entry in mutations:
            if entry > 0 or entry <= ending_index_of_seq2:
                if mutations[entry]['to'] != '-':
                    remove_mutation_list.append(entry)
        blanks = True
    if blanks:  # if blanks is True, leave all types of blanks, if blanks is False check for requested types
        termini, reference_gaps = True, True
        # for entry in mutations:
        #     for index in mutations[entry]:
        #         if mutations[entry][index] == '-':
        #             remove_mutation_list.append(entry)
    if not termini:  # Remove indices outside of sequence 2
        for entry in mutations:
            if entry < 0 or entry > ending_index_of_seq2:
                remove_mutation_list.append(entry)
    if not reference_gaps:  # Remove indices inside sequence 2 where sequence 1 is gapped
        for entry in mutations:
            if entry > 0 or entry <= ending_index_of_seq2:
                if mutations[entry]['to'] == '-':
                    remove_mutation_list.append(entry)

    for entry in remove_mutation_list:
        if entry in mutations:
            mutations.pop(entry)

    return mutations


def make_mutations_chain_agnostic(mutation_dict):
    """Remove chain identifier from mutation dictionary

    Args:
        mutation_dict (dict): {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Returns:
        flattened_dict (dict): {pdb: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}
    """
    flattened_dict = {}
    for pdb in mutation_dict:
        flattened_dict[pdb] = {}
        for chain in mutation_dict[pdb]:
            flattened_dict[pdb].update(mutation_dict[pdb][chain])

    return flattened_dict


def simplify_mutation_dict(mutation_dict, to=True):
    """Simplify mutation dictionary to 'to'/'from' AA key

    Args:
        mutation_dict (dict): {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Keyword Args:
        to=True (bool): Whether to use 'to' AA (True) or 'from' AA (False)
    Returns:
        mutation_dict (dict): {pdb: {mutation_index: 'K', ...}, ...}
    """
    simplification = get_mutation_to
    if not to:
        simplification = get_mutation_from

    for pdb in mutation_dict:
        for index in mutation_dict[pdb]:
            mutation_dict[pdb][index] = simplification(mutation_dict[pdb][index])

    return mutation_dict


def get_mutation_from(mutation_dict):
    """Remove 'to' identifier from mutation dictionary

    Args:
        mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
    Returns:
        mutation_dict (str): 'A'
    """
    return mutation_dict['from']


def get_mutation_to(mutation_dict):
    """Remove 'from' identifier from mutation dictionary
    Args:
        mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
    Returns:
        mutation_dict (str): 'K'
    """
    return mutation_dict['to']


def get_wildtype_file(des_directory):
    """Retrieve the wild-type file name from Design Directory"""
    # wt_file = glob(os.path.join(des_directory.building_blocks, PUtils.clean))
    wt_file = glob(os.path.join(des_directory.path, PUtils.clean))
    assert len(wt_file) == 1, '%s: More than one matching file found with %s' % (des_directory.path, PUtils.asu)
    return wt_file[0]
    # for file in os.listdir(des_directory.building_blocks):
    #     if file.endswith(PUtils.asu):
    #         return os.path.join(des_directory.building_blocks, file)


def weave_mutation_dict(sorted_freq, mut_prob, resi_divergence, int_divergence, des_divergence):
    """Make final dictionary, index to sequence

    Args:
        sorted_freq (dict): {15: ['S', 'A', 'T'], ... }
        mut_prob (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        resi_divergence (dict): {15: 0.732, 16: 0.552, ...}
        int_divergence (dict): {15: 0.732, 16: 0.552, ...}
        des_divergence (dict): {15: 0.732, 16: 0.552, ...}
    Returns:
        weaved_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    """
    weaved_dict = {}
    for residue in sorted_freq:
        final_resi = residue + index_offset
        weaved_dict[final_resi] = {}
        for aa in sorted_freq[residue]:
            weaved_dict[final_resi][aa] = round(mut_prob[residue][aa], 3)
        weaved_dict[final_resi]['jsd'] = resi_divergence[residue]
        weaved_dict[final_resi]['int_jsd'] = int_divergence[residue]
        weaved_dict[final_resi]['des_jsd'] = des_divergence[residue]

    return weaved_dict


def weave_sequence_dict(base_dict=None, **kwargs):  # *args, # sorted_freq, mut_prob, resi_divergence, int_divergence):
    """Make final dictionary indexed to sequence, from same-indexed, residue numbered, sequence dictionaries

    Args:
        *args (dict)
    Keyword Args:
        base=None (dict): Original dictionary
        **kwargs (dict): key=dictionary pairs to include in the final dictionary
            sorted_freq={15: ['S', 'A', 'T'], ... }, mut_prob={15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...},
                divergence (dict): {15: 0.732, 16: 0.552, ...}
    Returns:
        weaved_dict (dict): {16: {'freq': {'S': 0.134, 'A': 0.050, ...,} 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    """
    if base_dict:
        weaved_dict = base_dict
    else:
        weaved_dict = {}

    # print('kwargs', kwargs)
    for seq_dict in kwargs:
        # print('seq_dict', seq_dict)
        for residue in kwargs[seq_dict]:
            if residue not in weaved_dict:
                weaved_dict[residue] = {}
            # else:
            #     weaved_dict[residue][seq_dict] = {}
            if isinstance(kwargs[seq_dict][residue], dict):  # TODO make endlessly recursive?
                weaved_dict[residue][seq_dict] = {}
                for sub_key in kwargs[seq_dict][residue]:  # kwargs[seq_dict][residue]
                    weaved_dict[residue][seq_dict][sub_key] = kwargs[seq_dict][residue][sub_key]
            else:
                weaved_dict[residue][seq_dict] = kwargs[seq_dict][residue]

    # ensure all residues in weaved_dict have every keyword
    # missing_keys = {}
    # for residue in weaved_dict:
    #     missing_set = set(kwargs.keys()) - set(weaved_dict[residue].keys())
    #     if missing_set:
    #         for missing in missing_set:
    #             weaved_dict[residue][missing] = None
        # missing_keys[residue] = set(kwargs.keys()) - set(weaved_dict[residue].keys())
    # for residue in missing_keys:

    return weaved_dict


def get_fragment_metrics(fragment_matches):
    """Return the various metrics calculated by overlapping fragments at the interface of two proteins

    Returns:
        (tuple): all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap, \
        central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d
    """
    # fragment_matches = [{'mapped': entity1_surffrag_resnum, 'match': score_term,
    #                          'paired': entity2_surffrag_resnum, 'culster': cluster_id}, ...]
    fragment_i_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    fragment_j_index_count_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    entity1_match_scores, entity2_match_scores = {}, {}
    interface_residues_with_fragment_overlap = {'mapped': set(), 'paired': set()}
    for fragment in fragment_matches:

        interface_residues_with_fragment_overlap['mapped'].add(fragment['mapped'])
        interface_residues_with_fragment_overlap['paired'].add(fragment['paired'])
        covered_residues_pdb1 = [(fragment['mapped'] + j) for j in range(-2, 3)]
        covered_residues_pdb2 = [(fragment['paired'] + j) for j in range(-2, 3)]
        for k in range(5):
            resnum1 = covered_residues_pdb1[k]
            resnum2 = covered_residues_pdb2[k]
            if resnum1 not in entity1_match_scores:
                entity1_match_scores[resnum1] = [fragment['match']]
            else:
                entity1_match_scores[resnum1].append(fragment['match'])

            if resnum2 not in entity2_match_scores:
                entity2_match_scores[resnum2] = [fragment['match']]
            else:
                entity2_match_scores[resnum2].append(fragment['match'])

        fragment_i_index_count_d[fragment['cluster'].split('_')[0]] += 1
        fragment_j_index_count_d[fragment['cluster'].split('_')[0]] += 1

    # Generate Nanohedra score for center and all residues
    all_residue_score, center_residue_score = 0, 0
    for residue_number, res_scores in entity1_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['mapped']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    # doing this twice seems unnecessary as there is no new fragment information, but residue observations are
    # weighted by n, number of observations which differs between entities across the interface
    for residue_number, res_scores in entity2_match_scores.items():
        n = 1
        res_scores_sorted = sorted(res_scores, reverse=True)
        if residue_number in interface_residues_with_fragment_overlap['paired']:  # interface_residue_numbers: <- may be at termini
            for central_score in res_scores_sorted:
                center_residue_score += central_score * (1 / float(n))
                n *= 2
        else:
            for peripheral_score in res_scores_sorted:
                all_residue_score += peripheral_score * (1 / float(n))
                n *= 2

    all_residue_score += center_residue_score

    # Get the number of CENTRAL residues with overlapping fragments given z_value criteria
    central_residues_with_fragment_overlap = len(interface_residues_with_fragment_overlap['mapped']) + \
        len(interface_residues_with_fragment_overlap['paired'])

    # Get the number of TOTAL residues with overlapping fragments given z_value criteria
    total_residues_with_fragment_overlap = len(entity1_match_scores) + len(entity2_match_scores)

    if central_residues_with_fragment_overlap > 0:
        multiple_frag_ratio = (len(fragment_matches) * 2) / central_residues_with_fragment_overlap  # paired fragment
    else:
        multiple_frag_ratio = 0

    interface_residue_count = len(interface_residues_with_fragment_overlap['mapped']) + \
        len(interface_residues_with_fragment_overlap['paired'])
    if interface_residue_count > 0:
        percent_interface_matched = central_residues_with_fragment_overlap / float(interface_residue_count)
        percent_interface_covered = total_residues_with_fragment_overlap / float(interface_residue_count)
    else:
        percent_interface_matched, percent_interface_covered = 0, 0

    # Sum the total contribution from each fragment type on both sides of the interface
    fragment_content_d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for index in fragment_i_index_count_d:
        fragment_content_d[index] += fragment_i_index_count_d[index]
        fragment_content_d[index] += fragment_j_index_count_d[index]

    if len(fragment_matches) > 0:
        for index in fragment_content_d:
            fragment_content_d[index] = fragment_content_d[index] / (len(fragment_matches) * 2)  # paired fragment

    return all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
        central_residues_with_fragment_overlap, multiple_frag_ratio, fragment_content_d
        # interface_residue_count, percent_interface_matched, percent_interface_covered,


def generate_sequence_mask(fasta_file):
    """From a sequence with a design_selection, grab the residue indices that should be designed in the target
    structural calculation

    Returns:
        (list): The residue numbers (in pose format) that should be ignored in design
    """
    sequence_and_mask = read_fasta_file(fasta_file)
    sequences = list(sequence_and_mask)
    sequence = sequences[0]
    mask = sequences[1]
    if not len(sequence) == len(mask):
        raise DesignError('The sequence and design_selection are different lengths! Please correct the alignment and lengths '
                          'before proceeding.')

    return [idx for idx, aa in enumerate(mask, 1) if aa != '-']


def clean_comma_separated_string(string):
    return list(map(str.strip, string.strip().split(',')))


def generate_residue_mask(residue_string):
    """From a string with a design_selection, grab the residue indices that should be designable in the target
    structural calculation

    Returns:
        (list): The residue numbers (in pose format) that should be ignored in design
    """
    clean_residue_list = clean_comma_separated_string(residue_string)
    # extract ranges
    final_residues = []
    for residue in clean_residue_list:
        if '-' in residue:  # we have a range
            for residue_idx in range(*tuple(map(int, residue.split('-')))):
                final_residues.append(residue_idx)
            final_residues.append(residue_idx + 1)
        else:
            final_residues.append(residue)

    return list(map(int, final_residues))


def generate_chain_mask(chain_string):
    """From a string with a  design_selection, grab the chains that should be designable in the target
    structural calculation

    Returns:
        (list): The residue numbers (in pose format) that should be ignored in design
    """
    return clean_comma_separated_string(chain_string)
