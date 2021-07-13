import math
import os
from itertools import chain  # repeat
from math import floor, exp, log, log2
import subprocess
import time
from copy import deepcopy, copy
# from glob import glob

import numpy as np
import pandas as pd
from Bio import pairwise2, SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data.IUPACData import protein_letters, extended_protein_letters, protein_letters_3to1

import CommandDistributer
import PathUtils as PUtils
from SymDesignUtils import handle_errors, unpickle, get_all_base_root_paths, DesignError, start_log, pretty_format_table

# Globals
logger = start_log(name=__name__)
index_offset = 1
alph_3_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_counts = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
             'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
aa_weighted_counts = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0,
                      'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'stats': [0, 1]}
add_fragment_profile_instructions = 'To add fragment information, call Pose.generate_interface_fragments()'
subs_matrices = {'BLOSUM62': substitution_matrices.load('BLOSUM62')}


class SequenceProfile:
    """Contains the sequence information for a Structural unit. Should always be subclassed by an object like an Entity,
    basically any structure object with a .reference_sequence attribute could be used"""
    idx_to_alignment_type = {0: 'mapped', 1: 'paired'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evolutionary_profile = {}  # position specific scoring matrix
        self.design_pssm_file = None
        self.profile = {}  # design specific scoring matrix
        self.frag_db = None
        self.fragment_queries = {}
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        self.fragment_map = {}
        self.alpha = {}
        self.fragment_profile = {}
        self.fragment_pssm_file = None
        self.interface_data_file = None
        self.a3m_file = None
        self.msa = None
        self.msa_file = None
        self.pssm_file = None
        self.sequence_source = None
        self.sequence_file = None

    @classmethod
    def from_structure(cls, structure=None):
        return cls(structure=structure)

    @property
    def profile_length(self):
        # return self.structure.number_of_residues
        return self.number_of_residues

    @property
    def offset(self):
        """Return the starting index for the Entity based on pose numbering of the residues"""
        # return self.structure.residues[0].number - 1
        return self.residues[0].number - 1

    # @offset.setter
    # def offset(self, offset):
    #     self._entity_offset = offset

    # def set_structure(self, structure):
    #     self.structure = structure

    @property
    def structure_sequence(self):
        # return self.structure.get_structure_sequence()
        return self.get_structure_sequence()

    # def set_profile_length(self):
    #     self.profile_length = len(self.profile)

    def attach_fragment_database(self, db=None):
        """Attach an existing Fragment Database to the SequenceProfile"""
        if db:
            self.frag_db = db
        else:
            raise DesignError('%s: No fragment database connection was passed!'
                              % self.attach_fragment_database.__name__)

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

    def add_profile(self, evolution=True, out_path=os.getcwd(), null=False, fragments=True, **kwargs):
        #           fragment_observations=None, entities=None, pdb_numbering=True,
        """Add the evolutionary and fragment profiles onto the SequenceProfile

        Keyword Args:
            fragment_source=None (list):
            alignment_type=None (str): Either 'mapped' or 'paired'. Indicates how entity and fragments are aligned
            out_path=os.getcwd() (str): Location where sequence files should be written
            pdb_numbering=True (bool):
        """
        if null or not evolution and not fragments:
            null, evolution, fragments = True, False, False
            # self.add_evolutionary_profile(null=null, **kwargs)

        if evolution:  # add evolutionary information to the SequenceProfile
            if not self.evolutionary_profile:
                self.add_evolutionary_profile(out_path=out_path, **kwargs)
            self.verify_profile()
        else:
            self.null_pssm()

        if fragments:  # add fragment information to the SequenceProfile
            if self.fragment_map and self.frag_db:  # fragments have already been added, connect DB info
                retrieve_fragments = [fragment['cluster'] for idx_d in self.fragment_map.values()
                                      for fragments in idx_d.values() for fragment in fragments
                                      if fragment['cluster'] not in self.frag_db.cluster_info]
                self.frag_db.get_cluster_info(ids=retrieve_fragments)
            else:
                raise DesignError('Fragments were specified but have not been added to the SequenceProfile! '
                                  'The Pose/Entity must call assign_fragments() with fragment information')
            # process fragment profile from self.fragment_map or self.fragment_query
            self.add_fragment_profile()
            self.find_alpha()

        self.calculate_design_profile(boltzmann=True, favor_fragments=fragments)

    def verify_profile(self):
        """Check Pose and evolutionary profile for equality before proceeding"""
        rerun, second, success = False, False, False
        while not success:
            if self.profile_length != len(self.evolutionary_profile):
                self.log.warning('%s: Profile and Pose are different lengths!\nProfile=%d, Pose=%d'
                                 % (self.name, len(self.evolutionary_profile), self.profile_length))
                rerun = True

            if not rerun:
                # Check sequence from Pose and self.profile to compare identity before proceeding
                for idx, residue in enumerate(self.residues, 1):
                    profile_res_type = self.evolutionary_profile[idx]['type']
                    pose_res_type = protein_letters_3to1[residue.type.title()]
                    if profile_res_type != pose_res_type:
                        self.log.warning(
                            'Profile (%s) and Pose (%s) sequences mismatched!\n\tResidue %d: Profile=%s, Pose=%s'
                            % (self.pssm_file, self.sequence_file, residue.number, profile_res_type, pose_res_type))
                        rerun = True
                        break

            if rerun:
                if second:
                    raise DesignError('Profile Generation got stuck, design aborted')
                else:
                    self.log.info('Generating a new profile for %s' % self.name)
                    self.add_evolutionary_profile(force=True, out_path=os.path.dirname(self.pssm_file))
                    second = True
            else:
                success = True

    def add_evolutionary_profile(self, out_path=os.getcwd(), profile_source='hhblits', file=None, force=False):
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
        if file:
            self.pssm_file = file
        else:  # Check to see if the files of interest already exist
            # Extract/Format Sequence Information. SEQRES is prioritized if available
            if not self.sequence_file:  # not made/provided before add_evolutionary_profile, make new one at out_path
                self.write_fasta_file(self.reference_sequence, name=self.name, out_path=out_path)
            elif not os.path.exists(self.sequence_file) or force:
                self.log.debug('%s Sequence=%s' % (self.name, self.reference_sequence))
                self.write_fasta_file(self.reference_sequence, name=self.sequence_file, out_path='')
                self.log.debug('%s fasta file: %s' % (self.name, self.sequence_file))

            temp_file = os.path.join(out_path, '%s.hold' % self.name)
            self.pssm_file = os.path.join(out_path, '%s.hmm' % self.name)
            if not os.path.exists(self.pssm_file) or force:
                if not os.path.exists(temp_file):  # No work on this pssm file has been initiated
                    # Create blocking file to prevent excess work
                    with open(temp_file, 'w') as f:
                        self.log.info('Fetching \'%s\' sequence data.\n' % self.name)
                    self.log.debug('%s Evolutionary Profile not yet created.' % self.name)
                    if profile_source == 'psiblast':
                        self.log.info('Generating PSSM Evolutionary Profile for %s' % self.name)
                        self.psiblast(out_path=out_path)
                    else:
                        self.log.info('Generating HHM Evolutionary Profile for %s' % self.name)
                        self.hhblits(out_path=out_path)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                else:  # Block is in place, another process is working
                    self.log.info('Waiting for \'%s\' profile generation...' % self.name)
                    while not os.path.exists(self.pssm_file):
                        if int(time.time()) - int(os.path.getmtime(temp_file)) > 1800:  # > 30 minutes have passed
                            os.remove(temp_file)
                            raise DesignError('%s: Generation of the profile for %s took longer than the time '
                                              'limit. Job killed!'
                                              % (self.add_evolutionary_profile.__name__, self.name))
                        time.sleep(20)

        if profile_source == 'psiblast':
            self.parse_psiblast_pssm()
        else:
            self.parse_hhblits_pssm()

    def null_pssm(self):
        """Take the contents of a pssm file, parse, and input into a sequence dictionary.

        Sets:
            self.evolutionary_profile (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        self.evolutionary_profile = self.populate_design_dictionary(self.profile_length, alph_3_aa, dtype=int)
        structure_sequence = self.structure_sequence
        for idx, residue_number in enumerate(self.evolutionary_profile):
            self.evolutionary_profile[residue_number]['lod'] = copy(aa_counts)
            self.evolutionary_profile[residue_number]['type'] = structure_sequence[idx]
            self.evolutionary_profile[residue_number]['info'] = 0.0
            self.evolutionary_profile[residue_number]['weight'] = 0.0

    def fit_evolutionary_profile_to_structure(self):
        """From a evolutionary profile input according to a protein reference sequence, align the profile to the
        structure sequence, removing all information for residues not present in the structure

        Sets:
            (dict) self.evolutionary_profile
        """
        # self.retrieve_info_from_api()
        # grab the reference sequence used for translation (expression)
        if not self.reference_sequence:
            self.retrieve_sequence_from_api(entity_id=self.name)
        # generate the disordered indices which are positions in reference that are missing in structure
        disorder = generate_mutations(self.structure_sequence, self.reference_sequence, only_gaps=True)
        # removal of these positions from .evolutionary_profile will produce a properly indexed profile
        new_idx = 1
        structure_evolutionary_profile = {}
        for index, residue_data in self.evolutionary_profile.items():
            if index not in disorder:
                structure_evolutionary_profile[new_idx] = residue_data
                new_idx += 1
        self.evolutionary_profile = structure_evolutionary_profile

    def psiblast(self, out_path=None, remote=False):
        """Generate an position specific scoring matrix using PSI-BLAST subprocess

        Keyword Args:
            out_path=None (str): Disk location where generated file should be written
            remote=False (bool): Whether to perform the search through the web. If False, need blast installed locally!
        Sets:
            self.pssm_file (str): Name of the file generated by psiblast
        """
        self.pssm_file = os.path.join(out_path, '%s.pssm' % str(self.name))
        cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', self.sequence_file + '.fasta', '-out_ascii_pssm',
               self.pssm_file, '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']  # Todo # iters
        if remote:
            cmd.append('-remote')
        else:
            cmd.extend(['-num_threads', '8'])  # Todo

        p = subprocess.Popen(cmd)
        p.wait()  # p.communicate() Todo?

    @handle_errors(errors=(FileNotFoundError,))
    def parse_psiblast_pssm(self, **kwargs):
        """Take the contents of a pssm file, parse, and input into a sequence dictionary.
        # Todo it's CURRENTLY IMPOSSIBLE to use in calculate_design_profile, CHANGE psiblast lod score parsing
        Sets:
            self.evolutionary_profile (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        with open(self.pssm_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_data = line.strip().split()
            if len(line_data) == 44:
                residue_number = int(line_data[0])
                self.evolutionary_profile[residue_number] = copy(aa_counts)
                for i, aa in enumerate(alph_3_aa, 22):  # pose_dict[residue_number], 22):
                    # Get normalized counts for pose_dict
                    self.evolutionary_profile[residue_number][aa] = (int(line_data[i]) / 100.0)
                self.evolutionary_profile[residue_number]['lod'] = {}
                for i, aa in enumerate(alph_3_aa, 2):
                    self.evolutionary_profile[residue_number]['lod'][aa] = line_data[i]
                self.evolutionary_profile[residue_number]['type'] = line_data[1]
                self.evolutionary_profile[residue_number]['info'] = float(line_data[42])
                self.evolutionary_profile[residue_number]['weight'] = float(line_data[43])

    def hhblits(self, out_path=os.getcwd(), threads=CommandDistributer.hhblits_threads, return_command=False, **kwargs):
        """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

        Keyword Args:
            out_path=None (str): Disk location where generated file should be written
            threads=CommandDistributer.hhblits_threads (int): Number of cpu's to use for the process
            return_command=False (bool): Whether to simply return the hhblits command
        Sets:
            self.pssm_file (str): Name of the file generated by psiblast
        """
        self.pssm_file = os.path.join(out_path, '%s.hmm' % str(self.name))
        self.a3m_file = os.path.join(out_path, '%s.a3m' % str(self.name))
        # self.msa_file = os.path.join(out_path, '%s.fasta' % str(self.name))
        self.msa_file = os.path.join(out_path, '%s.sto' % str(self.name))  # preferred
        fasta_msa = os.path.join(out_path, '%s.fasta' % str(self.name))
        # todo for higher performance set up https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
        cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', self.sequence_file,
               '-ohhm', self.pssm_file, '-oa3m', self.a3m_file,  # '-Ofas', self.msa_file,
               '-hide_cons', '-hide_pred', '-hide_dssp', '-E', '1E-06',
               '-v', '1', '-cpu', str(threads)]
        # reformat_msa_cmd1 = [PUtils.reformat_msa_exe_path, self.a3m_file, self.msa_file, '-num', '-uc']
        # reformat_msa_cmd2 = [PUtils.reformat_msa_exe_path, self.a3m_file, fasta_msa, '-M', 'first', '-r']
        if return_command:
            return subprocess.list2cmdline(cmd)  # , subprocess.list2cmdline(reformat_msa_cmd)

        self.log.info('%s Profile Command: %s' % (self.name, subprocess.list2cmdline(cmd)))
        p = subprocess.Popen(cmd)
        p.communicate()
        p = subprocess.Popen([PUtils.reformat_msa_exe_path, self.a3m_file, self.msa_file, '-num', '-uc'])
        p.communicate()
        p = subprocess.Popen([PUtils.reformat_msa_exe_path, self.a3m_file, fasta_msa, '-M', 'first', '-r'])
        p.communicate()
        # os.system('rm %s' % self.a3m_file)

    @handle_errors(errors=(FileNotFoundError,))
    def parse_hhblits_pssm(self, null_background=True, **kwargs):
        """Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order

        Keyword Args:
            null_background=True (bool): Whether to use the null background for the specific protein
        Sets:
            self.evolutionary_profile (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                     'weight': 0.00}, {...}}
        """
        dummy = 0.00
        # 'uniclust30_2018_08'
        null_bg = {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228,
                   'I': 0.0534, 'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401,
                   'R': 0.0599, 'S': 0.0791, 'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}

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

        self.evolutionary_profile = {}
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
                        null_bg = {i: {} for i in alph_3_aa}
                        for i, aa in enumerate(alph_3_aa, 1):
                            null_bg[aa] = to_freq(background[i])

                if len(line.split()) == 23:
                    items = line.strip().split()
                    residue_number = int(items[1])
                    self.evolutionary_profile[residue_number] = {}
                    for i, aa in enumerate(protein_letters, 2):
                        self.evolutionary_profile[residue_number][aa] = to_freq(items[i])
                    self.evolutionary_profile[residue_number]['lod'] = \
                        get_lod(self.evolutionary_profile[residue_number], null_bg)
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
        new_key = 1
        for profile in pssms:
            for position_profile in profile.values():
                self.evolutionary_profile[new_key] = position_profile
                new_key += 1

    def combine_fragment_profile(self, fragment_profiles):
        """Combine a list of fragment profiles incrementing the residue number in each additional fragment profile

        Args:
            fragment_profiles (list(dict)): List of fragment profiles to concatenate
        Sets
            self.fragment_profile (dict): To a concatenated fragment profile from the input fragment profiles
        """
        new_key = 1
        for profile in fragment_profiles:
            for position_profile in profile.values():
                self.fragment_profile[new_key] = position_profile
                new_key += 1

    def combine_profile(self, profiles):
        """Combine a list of DSSMs incrementing the residue number in each additional DSSM

        Args:
            profiles (list(dict)): List of DSSMs to concatenate
        Sets
            self.profile (dict): Using the list of input DSSMs, make a concatenated DSSM
        """
        new_key = 1
        for profile in profiles:
            for position_profile in profile.values():
                self.profile[new_key] = position_profile
                new_key += 1

    def add_msa(self,):  # Todo adapt to different data sources
        """Add a multiple sequence alignment to the profile"""
        if not self.msa_file:
            # self.msa = self.database.alignments.retrieve_data(name=self.name)
            raise AttributeError('No .msa_file attribute is specified yet!')
        # self.msa = MultipleSequenceAlignment.from_stockholm(self.msa_file)
        try:
            self.msa = MultipleSequenceAlignment.from_stockholm(self.msa_file)
            # self.msa = MultipleSequenceAlignment.from_fasta(self.msa_file)
        except FileNotFoundError:
            try:
                self.msa = MultipleSequenceAlignment.from_fasta('%s.fasta' % os.path.splitext(self.msa_file)[0])
                # self.msa = MultipleSequenceAlignment.from_stockholm('%s.sto' % os.path.splitext(self.msa_file)[0])
            except FileNotFoundError:
                raise FileNotFoundError('No multiple sequence alignment exists at %s' % self.msa_file)

    def collapse_profile(self):
        """Find the mean and standard deviation for each index in the SequenceProfile sequence MSA

        Turn each sequence into a HCI array. All sequences have different lengths.
        For each input sequence, make a gap mask into a 2D array (#sequences x length of sequence alignment)
        iterate over the gap mask array adding the ith column of the mask to the row of the iterator array
        where there is a gap there is a 0, sequence present is 1
            iterator array        gap mask  Hydro Collapse Array     Aligned HCI (drop col 2)   HCI
        iter i 1  2  3  4  element 0  1  2      0    1    2          0    1    2            0    1    3
              [0  1  2  2         [1  1  0    [0.5  0.2  0.5       [0.5  0.2   0          [0.5  0.2  ...
               0  0  1  2          0  1  1     0.4  0.7  0.4    =    0   0.4  0.7    ->     0   0.4  ...
               0  0  1  2          0  1  1     0.3  0.6  0.3         0   0.3  0.6           0   0.3  ...

        After the addition, the hydro collapse array index that is accessed by the iterator is multiplied by the gap
        mask to return a null value is there is no value and the hydro collapse value if there is one
        therefore the element such as 3 in the Aligned HCI would be dropped from the array when the aligned sequence
        is removed of any gaps and only the iterations will be left, essentially giving the HCI for the sequence
        profile in the native context, however adjusted to the specific context of the protein/design sequence at hand

        Returns:
            (pandas.DataFrame): DataFrame containing each sequences hydrophobic collapse values for the profile, mean,
                and std
        """
        if not self.msa:
            try:
                self.add_msa()
            except DesignError:
                raise DesignError('Ensure that you have properly set up the .msa for this SequenceProfile. To do this, '
                                  'either link the Structure to the Master Database, call %s, or pass the location of a'
                                  ' multiple sequence alignment. Supported formats:\n%s)'
                                  % (msa_generation_function, pretty_format_table(msa_supported_types.items())))
        msa_np = np.array([list(record) for record in self.msa.alignment], np.character)
        # msa_np = np.array([list(str(record.seq)) for record in self.msa.alignment], np.character)
        # print('msa_np', msa_np[:5, :])
        # aligned_hci_np = np.zeros((self.msa.number_of_sequences, self.msa.length))
        # Make the output array one longer to keep a np.nan value at the 0 index for collapse gaps
        evolutionary_collapse_np = np.zeros((self.msa.number_of_sequences, self.msa.length + 1))  # aligned_hci_np.copy()
        evolutionary_collapse_np[:, 0] = np.nan  # np.nan for all missing indices
        # print('alignment', self.msa.alignment[:5, :])
        for idx, record in enumerate(self.msa.alignment):
            non_gapped_sequence = str(record.seq).replace('-', '')
            evolutionary_collapse_np[idx, 1:len(non_gapped_sequence) + 1] = hydrophobic_collapse_index(non_gapped_sequence)

        # print('evolutionary_collapse_np', evolutionary_collapse_np[:5, :])
        # iterator_np = np.zeros((self.msa.number_of_sequences,), order='F', dtype=int)
        msa_mask = np.isin(msa_np, b'-', invert=True)  # returns bool array '-' = False. Converted during arithmetic
        # print('msa_mask', msa_mask[:5, :])
        iterator_np = np.cumsum(msa_mask, axis=1) * msa_mask
        # print('iterator_np', iterator_np[:5, :])
        # for idx in range(self.msa.length):
        #     # print('iterator shape', iterator_np.shape)
        #     # print('slices mas_mask shape', msa_mask[:, idx].shape)
        #     # print(msa_mask[:, idx])
        #     # aligned_hci_np[:, idx] = evolutionary_collapse_np[:, iterator_np] * msa_mask[:, idx]
        #     aligned_hci_np[:, idx] = evolutionary_collapse_np[np.ix_(:, iterator_np)] * msa_mask[:, idx]
        #     iterator_np += msa_mask[:, idx]
        aligned_hci_np = np.take_along_axis(evolutionary_collapse_np, iterator_np, axis=1)
        # print('aligned_hci_np', aligned_hci_np[:5, :])
        sequence_hci_np = aligned_hci_np[:, msa_mask[0]]  # where the aligned sequence is the first index
        # print('sequence:\n', '     '.join(aa for idx, aa in enumerate(map(str, self.msa.alignment[0].seq)) if msa_mask[0][idx]))
        # print(list(map(round, sequence_hci_np[0, :].tolist(), repeat(2))), '\nsequence_hci_np')
        sequence_hci_df = pd.DataFrame(sequence_hci_np, columns=list(range(1, self.msa.query_length + 1)))
        sequence_hci_mean_s = sequence_hci_df.mean()
        # sequence_hci_mean = pd.Series(sequence_hci_np.mean(axis=1), name='mean')
        # sequence_hci_mean.index += 1
        sequence_hci_std_s = sequence_hci_df.std()
        # sequence_hci_std = pd.Series(sequence_hci_np.std(axis=1), name='std')
        # sequence_hci_std.index += 1
        # sequence_hci_z_value = (aligned_hci_np[0] - sequence_hci_mean) / sequence_hci_std
        return pd.concat([sequence_hci_df,
                          pd.concat([sequence_hci_mean_s, sequence_hci_std_s], axis=1, keys=['mean', 'std']).T])

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

    # fragments
    # [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]
    def add_fragment_profile(self):  # , fragment_source=None, alignment_type=None):
        """From self.fragment_map, add the fragment profile to the SequenceProfile

        Sets:
            self.fragment_profile
        """
        # v now done at the pose_level
        # self.assign_fragments(fragments=fragment_source, alignment_type=alignment_type)
        if self.fragment_map:
            self.generate_fragment_profile()
            self.simplify_fragment_profile(keep_extras=True)
        else:
            if self.fragment_queries:  # Todo refactor this to Pose
                for query_pair, fragments in self.fragment_queries.items():
                    for query_idx, entity in enumerate(query_pair):
                        if entity.name == self.name:
                            # add to fragment map
                            self.assign_fragments(fragments=fragments,
                                                  alignment_type=SequenceProfile.idx_to_alignment_type[query_idx])
            else:
                self.log.error('No fragment information associated with the Entity %s yet! You must add to the profile '
                               'otherwise only evolutionary values will be used.\n%s'
                               % (self.name, add_fragment_profile_instructions))
                return None

    def assign_fragments(self, fragments=None, alignment_type=None):
        """Distribute fragment information to self.fragment_map. One-indexed residue dictionary

        Keyword Args:
            fragments=None (list): The fragment list to assign to the sequence profile with format
            [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]
            alignment_type=None (str): Either mapped or paired
        Sets:
            self.fragment_map (dict): {1: [{'chain']: 'mapped', 'cluster': 1_2_123, 'match': 0.61}, ...], ...}
        """
        if alignment_type not in ['mapped', 'paired']:
            return None
        if not fragments:
            return None

        if not self.fragment_map:
            self.fragment_map = self.populate_design_dictionary(self.profile_length,
                                                                [j for j in range(*self.frag_db.fragment_range)],
                                                                dtype=list)
        #     print('New fragment_map')
        # print(fragments)
        # print(self.name)
        # print(self.offset)
        for fragment in fragments:
            residue_number = fragment[alignment_type] - self.offset
            for j in range(*self.frag_db.fragment_range):  # lower_bound, upper_bound
                self.fragment_map[residue_number + j][j].append({'chain': alignment_type,
                                                                 'cluster': fragment['cluster'],
                                                                 'match': fragment['match']})
        # should be unnecessary when fragments are generated internally
        # remove entries which don't exist on protein because of fragment_index +- residues
        not_available = [residue_number for residue_number in self.fragment_map
                         if residue_number <= 0 or residue_number > self.profile_length]
        for residue_number in not_available:
            self.log.info('In \'%s\', residue %d is represented by a fragment but there is no Atom record for it. '
                          'Fragment index will be deleted.' % (self.name, residue_number))
            self.fragment_map.pop(residue_number)

        # self.log.debug('Residue Cluster Map: %s' % str(self.fragment_map))

    def generate_fragment_profile(self):
        """Add frequency information to the fragment profile using parsed cluster information. Frequency information is
        added in a fragment index dependent manner. If multiple fragment indices are present in a single residue, a new
        observation is created for that fragment index.

        Converts a fragment_map with format:
            (dict): {1: {-2: [{'chain': 'mapped', 'cluster': '1_2_123', 'match': 0.6}, ...], -1: [], ...},
                     2: {}, ...}
        To a fragment_profile with format:
            (dict): {1: {-2: {0: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37], 'match': 0.6}, 1: {}}, -1: {}, ... },
                     2: {}, ...}
        """
        self.log.debug('Generating Fragment Profile from Map')
        for residue_number, fragment_indices in self.fragment_map.items():
            self.fragment_profile[residue_number] = {}
            # if residue_number == 12:
            #     print('At 12, fragment_map: %s' % self.fragment_map)
            # if residue_number == 13:
            #     print('At 12, fragment_map: %s' % self.fragment_profile[12].items())
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

                    # if residue_number == 12:
                    #     print('Observation_index %d\nAA_freq %s\nMatch %f' % (observation_idx, aa_freq, fragment['match']))
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
        # self.log.debug(self.fragment_profile.items())
        database_bkgnd_aa_freq = self.frag_db.get_db_aa_frequencies()
        sequence = self.reference_sequence
        no_design = []
        for residue, index_d in self.fragment_profile.items():
            total_fragment_weight, total_fragment_observations = 0, 0
            for index, observation_d in index_d.items():
                if observation_d:
                    # sum the weight for each fragment observation
                    total_obs_weight, total_obs_x_match_weight = 0.0, 0.0
                    # total_match_weight = 0.0
                    for observation, fragment_frequencies in observation_d.items():
                        if fragment_frequencies:
                            total_obs_weight += fragment_frequencies['stats'][1]
                            total_obs_x_match_weight += fragment_frequencies['stats'][1] * fragment_frequencies['match']
                            # total_match_weight += self.fragment_profile[residue][index][obs]['match']

                    # Check if weights are associated with observations, if not side chain isn't significant!
                    if total_obs_weight > 0:
                        total_fragment_weight += total_obs_weight
                        obs_aa_dict = deepcopy(aa_weighted_counts)  # {'A': 0, 'C': 0, ..., 'stats': [0, 1]}
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
                res_aa_dict = copy(aa_counts)
                for index in self.fragment_profile[residue]:
                    if self.fragment_profile[residue][index]:
                        index_weight = self.fragment_profile[residue][index]['stats'][1]  # total_obs_weight
                        for aa in self.fragment_profile[residue][index]:
                            if aa not in ['stats', 'match']:  # Add all occurrences to summed frequencies list
                                res_aa_dict[aa] += self.fragment_profile[residue][index][aa] * (
                                            index_weight / total_fragment_weight)
                res_aa_dict['lod'] = self.get_lod(res_aa_dict, database_bkgnd_aa_freq)
                res_aa_dict['stats'] = [total_fragment_observations, total_fragment_weight]  # over all idx and obs
                res_aa_dict['type'] = sequence[residue - 1]  # offset to zero index
                self.fragment_profile[residue] = res_aa_dict
            else:  # Add to list for removal from the profile
                no_design.append(residue)

        if keep_extras:
            if self.evolutionary_profile:  # if not an empty dictionary
                for residue in no_design:
                    self.fragment_profile[residue] = self.evolutionary_profile.get(residue)  # TODO, aa_weighted_counts)
            else:
                for residue in no_design:
                    self.fragment_profile[residue] = aa_weighted_counts
        else:  # remove missing residues from dictionary
            for residue in no_design:
                self.fragment_profile.pop(residue)

    def find_alpha(self, alpha=0.5):
        """Find fragment contribution to design with a maximum contribution of alpha. Used subsequently to integrate
         fragment profile during combination with evolutionary profile in calculate_design_profile

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
        if not self.frag_db:
            raise DesignError('%s: No fragment database connected! Cannot calculate optimal fragment contribution '
                              'without this.' % self.find_alpha.__name__)
        assert 0 <= alpha <= 1, '%s: Alpha parameter must be between 0 and 1' % self.find_alpha.__name__
        alignment_type_to_idx = {'mapped': 0, 'paired': 1}  # could move to class, but not used elsewhere
        match_score_average = 0.5  # when fragment pair rmsd equal to the mean cluster rmsd
        bounded_floor = 0.2
        fragment_stats = self.frag_db.statistics
        for entry in self.fragment_profile:
            # can't use the match count as the fragment index may have no useful residue information
            # count = len([1 for obs in self.fragment_map[entry][index] for index in self.fragment_map[entry]]) or 1
            # instead use number of fragments with SC interactions count from the frequency map
            count = self.fragment_profile[entry].get('stats', (None,))[0]
            if not count:  # if the data is missing 'stats' or the 'stats'[0] is 0
                continue  # move on, this isn't a fragment observation or we have no observed fragments
            # match score is bounded between 1 and 0.2
            match_sum = sum(obs['match'] for index_values in self.fragment_map[entry].values() for obs in index_values)
            # if count == 0:
            #     # ensure that match modifier is 0 so self.alpha[entry] is 0, as there is no fragment information here!
            #     count = match_sum * 5  # makes the match average = 0.5

            match_average = match_sum / float(count)
            # find the match modifier which spans from 0 to 1
            if match_average < match_score_average:
                match_modifier = ((match_average - bounded_floor) / (match_score_average - bounded_floor))
            else:
                match_modifier = 1  # match_score_average / match_score_average  # 1 is the maximum bound

            # find the total contribution from a typical fragment of this type
            contribution_total = sum(fragment_stats[self.frag_db.get_cluster_id(obs['cluster'], index=2)][0]
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

            # get the average contribution of each fragment type
            stats_average = contribution_total / count
            # get entry average fragment weight. total weight for issm entry / count
            frag_weight_average = self.fragment_profile[entry]['stats'][1] / match_sum

            # modify alpha proportionally to cluster average weight and match_modifier
            if frag_weight_average < stats_average:  # if design frag weight is less than db cluster average weight
                self.alpha[entry] = alpha * (frag_weight_average / stats_average) * match_modifier
            else:
                self.alpha[entry] = alpha * match_modifier

    def calculate_design_profile(self, favor_fragments=True, boltzmann=False, alpha=0.5):
        """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

        Takes self.evolutionary_profile
            (dict): HHblits - {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                                   'weight': 0.00}, {...}}
                    PSIBLAST - {1: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
                                    'weight': 0.73}, {...}}
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
            (dict): {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                         'weight': 0.00}, ...}} - combined PSSM dictionary
        """
        assert 0 <= alpha <= 1, '%s: Alpha parameter must be between 0 and 1' % self.calculate_design_profile.__name__
        # copy the evol profile to self.profile (design specific scoring matrix)
        self.profile = deepcopy(self.evolutionary_profile)
        # Combine fragment and evolutionary probability profile according to alpha parameter
        self.log.info('At Entity %s, combined evolutionary and fragment profile (Design Profile) with:\n\t%s'
                      % (self.name, '\n\t'.join('Residue %4d: %d%% fragment weight' %
                                                (entry, weight * 100) for entry, weight in self.alpha.items())))
        for entry, weight in self.alpha.items():  # weight will be 0 if the fragment_profile is empty
            for aa in protein_letters:
                self.profile[entry][aa] = \
                    (weight * self.fragment_profile[entry][aa]) + ((1 - weight) * self.profile[entry][aa])

        if favor_fragments:
            # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
            # Used to weight fragments higher in design
            boltzman_energy = 1
            favor_seqprofile_score_modifier = 0.2 * CommandDistributer.reference_average_residue_weight
            database_bkgnd_aa_freq = self.frag_db.get_db_aa_frequencies()

            null_residue = self.get_lod(database_bkgnd_aa_freq, database_bkgnd_aa_freq)
            null_residue = {aa: float(frequency) for aa, frequency in null_residue.items()}

            for entry in self.profile:
                self.profile[entry]['lod'] = null_residue  # Caution all reference same object
            for entry, weight in self.alpha.items():  # self.fragment_profile:
                self.profile[entry]['lod'] = \
                    self.get_lod(self.fragment_profile[entry], database_bkgnd_aa_freq, round_lod=False)
                # get the sum for the partition function
                partition, max_lod = 0, 0.0
                for aa in self.profile[entry]['lod']:
                    if boltzmann:  # boltzmann probability distribution scaling, lod = z[i]/Z, Z = sum(exp(score[i]/kT))
                        self.profile[entry]['lod'][aa] = exp(self.profile[entry]['lod'][aa] / boltzman_energy)
                        partition += self.profile[entry]['lod'][aa]
                    # linear scaling, remove any lod penalty
                    elif self.profile[entry]['lod'][aa] < 0:
                        self.profile[entry]['lod'][aa] = 0
                    # find the maximum/residue (local) lod score
                    if self.profile[entry]['lod'][aa] > max_lod:
                        max_lod = self.profile[entry]['lod'][aa]
                # takes the percent of max alpha for each entry multiplied by the standard residue scaling factor
                modified_entry_alpha = (weight / alpha) * favor_seqprofile_score_modifier
                if boltzmann:
                    modifier = partition
                    modified_entry_alpha /= (max_lod / partition)
                else:
                    modifier = max_lod

                # weight the final lod score by the modifier and the scaling factor for the chosen method
                for aa in self.evolutionary_profile[entry]['lod']:
                    self.profile[entry]['lod'][aa] /= modifier  # get percent total (boltzman) or percent max (linear)
                    self.profile[entry]['lod'][aa] *= modified_entry_alpha  # scale by score modifier
                # self.log.debug('Residue %4d Fragment lod ratio generated with alpha=%f' % (entry, weight / alpha))

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

        frag_residue_object_d = residue_number_to_object(self, frag_cluster_residue_d)

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
        # interface_cb_indices = interface.get_cb_indices()  # InclGlyCA=True)

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
                mutated_pdb.mutate_residue(number=residue)
        mutated_pdb.write(des_dir.consensus_pdb)
        # mutated_pdb.write(consensus_pdb)
        # mutated_pdb.write(consensus_pdb, cryst1=cryst)

    # @staticmethod
    # def generate_mutations(mutant, reference, offset=True, blanks=False, termini=False, reference_gaps=False,
    #                        only_gaps=False):
    #     """Create mutation data in a typical A5K format. One-indexed dictionary keys, mutation data accessed by 'from'
    #     and 'to' keywords. By default all gaped sequences are excluded from returned mutations
    #
    #     For PDB file comparison, mutant should be crystal sequence (ATOM), reference should be expression sequence
    #     (SEQRES). only_gaps=True will return only the gaped area while blanks=True will return all differences between
    #     the alignment sequences. termini=True returns missing alignments at the termini
    #
    #     Args:
    #         mutant (str): Mutant sequence. Will be in the 'to' key
    #         reference (str): Wild-type sequence or sequence to reference mutations against. Will be in the 'from' key
    #     Keyword Args:
    #         offset=True (bool): Whether sequences are different lengths. Creates a new alignment
    #         blanks=False (bool): Whether to include indices that are outside the reference sequence or missing residues
    #         termini=False (bool): Whether to include indices that are outside the reference sequence boundaries
    #         reference_gaps=False (bool): Whether to include indices with missing residues inside the reference sequence
    #         only_gaps=False (bool): Whether to only include indices that are missing residues
    #     Returns:
    #         (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    #     """
    #     # TODO change function name/order of mutant and reference arguments to match logic with 'from' 37 'to' framework
    #     if offset:
    #         alignment = generate_alignment(mutant, reference)
    #         align_seq_1 = alignment[0][0]
    #         align_seq_2 = alignment[0][1]
    #     else:
    #         align_seq_1 = mutant
    #         align_seq_2 = reference
    #
    #     # Extract differences from the alignment
    #     starting_index_of_seq2 = align_seq_2.find(reference[0])
    #     ending_index_of_seq2 = starting_index_of_seq2 + align_seq_2.rfind(reference[-1])  # find offset end_index
    #     mutations = {}
    #     for i, (seq1_aa, seq2_aa) in enumerate(zip(align_seq_1, align_seq_2), -starting_index_of_seq2 + index_offset):
    #         if seq1_aa != seq2_aa:
    #             mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
    #             # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))
    #
    #     remove_mutation_list = []
    #     if only_gaps:  # remove the actual mutations
    #         for entry in mutations:
    #             if entry > 0 or entry <= ending_index_of_seq2:
    #                 if mutations[entry]['to'] != '-':
    #                     remove_mutation_list.append(entry)
    #         blanks = True
    #     if blanks:  # if blanks is True, leave all types of blanks, if blanks is False check for requested types
    #         termini, reference_gaps = True, True
    #     if not termini:  # Remove indices outside of sequence 2
    #         for entry in mutations:
    #             if entry < 0 or entry > ending_index_of_seq2:
    #                 remove_mutation_list.append(entry)
    #     if not reference_gaps:  # Remove indices inside sequence 2 where sequence 1 is gapped
    #         for entry in mutations:
    #             if entry > 0 or entry <= ending_index_of_seq2:
    #                 if mutations[entry]['to'] == '-':
    #                     remove_mutation_list.append(entry)
    #
    #     for entry in remove_mutation_list:
    #         mutations.pop(entry, None)
    #
    #     return mutations

    # @staticmethod
    # def generate_alignment(seq1, seq2, matrix='BLOSUM62'):
    #     """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences*
    #
    #     Returns:
    #         # TODO
    #     """
    #     _matrix = subs_matrices.get(matrix, substitution_matrices.load(matrix))
    #     gap_penalty = -10
    #     gap_ext_penalty = -1
    #     logger.debug('Generating sequence alignment between:\n%s\nAND:\n%s' % (seq1, seq2))
    #     # Create sequence alignment
    #     return pairwise2.align.globalds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)
    #     # return pairwise2.align.localds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)

    # def generate_design_mutations(self, all_design_files, wild_type_file, pose_num=False):
    #     """From a wild-type sequence (original PDB structure), and a collection of structure sequences that have
    #     undergone design (Atom sequences), generate 'A5K' style mutation data
    #
    #     Args:
    #         all_design_files (list): PDB files on disk to extract sequence info and compare
    #         wild_type_file (str): PDB file on disk which contains a reference sequence
    #     Returns:
    #         mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    #     """
    #     # pdb_dict = {'ref': PDB(file=wild_type_file)}
    #     # for file_name in all_design_files:
    #     #     pdb = PDB(file=file_name)
    #     #     pdb.name = os.path.splitext(os.path.basename(file_name))[0])
    #     #     pdb_dict[pdb.name] = pdb
    #     #
    #     # return extract_sequence_from_pdb(pdb_dict, mutation=True, pose_num=pose_num)  # , offset=False)

    @staticmethod
    def populate_design_dictionary(n, alphabet, dtype=dict, zero_index=False):
        """Return a dictionary with n elements, each integer key containing another dictionary with the items in
        alphabet as keys. By default, one-indexed, and data inside the alphabet dictionary is a dictionary.
        Set dtype as any viable container type [list, set, tuple, int, etc.]. If dtype is int, 0 will be added to use
        for counting

        Args:
            n (int): number of residues in a design
            alphabet (iter): alphabet of interest
        Keyword Args:
            dtype=object (object): The type of object present in the interior dictionary
            zero_index=False (bool): If True, return the dictionary with zero indexing
         Returns:
             (dict[mapping[int, dict[mapping[str, Any]]]): {1: {alph1: {}, alph2: {}, ...}, 2: {}, ...}
                Custom length, 0 indexed dictionary with residue number keys
         """
        if zero_index:
            offset = 0
        else:
            offset = index_offset

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
        # lods = {aa: None for aa in aa_freq}
        lods = {}
        for aa in aa_freq:
            if aa not in ['stats', 'match', 'lod', 'type']:
                if aa_freq[aa] == 0:
                    lods[aa] = -9
                else:
                    lods[aa] = float((2.0 * log2(aa_freq[aa] / background[aa])))  # + 0.0
                    if lods[aa] < -9:
                        lods[aa] = -9
                    elif round_lod:
                        lods[aa] = round(lods[aa])

        return lods

    @staticmethod
    def write_pssm_file(pssm_dict, name, out_path=os.getcwd()):
        """Create a PSI-BLAST format PSSM file from a PSSM dictionary. Assumes residue numbering is correct!

        Args:
            pssm_dict (dict): A dictionary which has the keys: 'A', 'C', ... (all aa's), 'lod', 'type', 'info', 'weight'
            name (str): The name of the file including the extension
        Keyword Args:
            out_path=os.getcwd() (str): A specific location to write the file to
        Returns:
            (str): Disk location of newly created .pssm file
        """
        if not pssm_dict:
            return None

        # find out if the pssm has values expressed as frequencies (percentages) or as counts and modify accordingly
        # lod_freq, counts_freq = False, False
        separation1, separation2 = 3, 3
        # first_key = next(iter(pssm_dict.keys()))
        if type(pssm_dict[next(iter(pssm_dict.keys()))]['lod']['A']) == float:
            separation1 = 4
            # lod_freq = True
        # if type(pssm_dict[first_key]['A']) == float:
        #     counts_freq = True

        header = '\n\n            %s%s%s\n' % ((' ' * separation1).join(alph_3_aa), ' ' * separation1,
                                               (' ' * separation2).join(alph_3_aa))
        # footer = ''
        out_file = os.path.join(out_path, name)
        with open(out_file, 'w') as f:
            f.write(header)
            for residue_number, profile in pssm_dict.items():
                aa_type = profile['type']
                # lod_string = ''
                if isinstance(profile['lod']['A'], float):  # lod_freq:  # relevant for favor_fragment
                    # for aa in alph_3_aa:  # ensures alpha_3_aa_list for PSSM format
                    #     lod_string += '{:>4.2f} '.format(profile['lod'][aa])
                    lod_string = '%s ' % ' '.join('{:>4.2f}'.format(profile['lod'][aa]) for aa in alph_3_aa)
                else:
                    # for aa in alph_3_aa:  # ensures alpha_3_aa_list for PSSM format
                    #     lod_string += '{:>3d} '.format(profile['lod'][aa])
                    lod_string = '%s ' % ' '.join('{:>3d}'.format(profile['lod'][aa]) for aa in alph_3_aa)
                # counts_string = ''
                if isinstance(profile['A'], float):  # counts_freq: # relevant for freq calculations
                    # for aa in alph_3_aa:  # ensures alpha_3_aa_list for PSSM format
                    #     counts_string += '{:>3.0f} '.format(floor(profile[aa] * 100))
                    counts_string = '%s ' % ' '.join('{:>3.0f}'.format(floor(profile[aa] * 100)) for aa in alph_3_aa)
                else:
                    # for aa in alph_3_aa:  # ensures alpha_3_aa_list for PSSM format
                    #     counts_string += '{:>3d} '.format(profile[aa])
                    counts_string = '%s ' % ' '.join('{:>3d}'.format(profile[aa]) for aa in alph_3_aa)
                info = profile.get('info', 0.0)
                weight = profile.get('weight', 0.0)
                # line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}\n'.format(residue_number, aa_type, lod_string,
                #                                                                counts_string, round(info, 4),
                #                                                                round(weight, 4))
                f.write('{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}\n'
                        .format(residue_number, aa_type, lod_string, counts_string, round(info, 4), round(weight, 4)))
            # f.write(footer)

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

    return {}  # Should never be called


def get_db_aa_frequencies(database):
    """Retrieve database specific interface background AA frequencies

    Args:
        database (str): Location of database on disk
    Returns:
        (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    """
    return get_db_statistics(database).get('frequencies', {})


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
    _range = floor(length / 2)
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


def residue_object_to_number(residue_dict):  # TODO DEPRECIATE
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
                    obs_aa_dict = deepcopy(aa_weighted_counts)
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
            res_aa_dict = deepcopy(aa_weighted_counts)
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
            design_cluster_dict[res] = aa_weighted_counts
    else:
        for res in no_design:
            design_cluster_dict.pop(res)

    return design_cluster_dict


# def psiblast(query, outpath=None, remote=False):  # UNUSED
#     """Generate an position specific scoring matrix using PSI-BLAST subprocess
#
#     Args:
#         query (str): Basename of the sequence to use as a query, intended for use as pdb
#     Keyword Args:
#         outpath=None (str): Disk location where generated file should be written
#         remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
#     Returns:
#         outfile_name (str): Name of the file generated by psiblast
#         p (subprocess): Process object for monitoring progress of psiblast command
#     """
#     # I would like the background to come from Uniref90 instead of BLOSUM62 #TODO
#     if outpath is not None:
#         outfile_name = os.path.join(outpath, query + '.pssm')
#         direct = outpath
#     else:
#         outfile_name = query + '.hmm'
#         direct = os.getcwd()
#     if query + '.pssm' in os.listdir(direct):
#         cmd = ['echo', 'PSSM: ' + query + '.pssm already exists']
#         p = subprocess.Popen(cmd)
#
#         return outfile_name, p
#
#     cmd = ['psiblast', '-db', PUtils.alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
#            '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']
#     if remote:
#         cmd.append('-remote')
#     else:
#         cmd.append('-num_threads')
#         cmd.append('8')
#
#     p = subprocess.Popen(cmd)
#
#     return outfile_name, p
#
#
# def hhblits(query, threads=CommandDistributer.hhblits_threads, outpath=os.getcwd()):
#     """Generate an position specific scoring matrix from HHblits using Hidden Markov Models
#
#     Args:
#         query (str): Basename of the sequence to use as a query, intended for use as pdb
#         threads (int): Number of cpu's to use for the process
#     Keyword Args:
#         outpath=None (str): Disk location where generated file should be written
#     Returns:
#         outfile_name (str): Name of the file generated by hhblits
#         p (subprocess): Process object for monitoring progress of hhblits command
#     """
#
#     outfile_name = os.path.join(outpath, os.path.splitext(os.path.basename(query))[0] + '.hmm')
#
#     cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(threads)]
#     logger.info('%s Profile Command: %s' % (query, subprocess.list2cmdline(cmd)))
#     p = subprocess.Popen(cmd)
#
#     return outfile_name, p


@handle_errors(errors=(FileNotFoundError,))
def parse_pssm(file, **kwargs):
    """Take the contents of a pssm file, parse, and input into a pose profile dictionary.

    Resulting residue dictionary is zero-indexed
    Args:
        file (str): The name/location of the file on disk
    Returns:
        pose_dict (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                {...}}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    for line in lines:
        line_data = line.strip().split()
        if len(line_data) == 44:
            residue_number = int(line_data[0])
            pose_dict[residue_number] = copy(aa_counts)
            for i, aa in enumerate(alph_3_aa, 22):
                # Get normalized counts for pose_dict
                pose_dict[residue_number][aa] = (int(line_data[i]) / 100.0)
            pose_dict[residue_number]['lod'] = {}
            for i, aa in enumerate(alph_3_aa, 2):
                pose_dict[residue_number]['lod'][aa] = line_data[i]
            pose_dict[residue_number]['type'] = line_data[1]
            pose_dict[residue_number]['info'] = float(line_data[42])
            pose_dict[residue_number]['weight'] = float(line_data[43])

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
            lods[a] = float((2.0 * log2(aa_freq_dict[a]/bg_dict[a])))  # + 0.0
            if lods[a] < -9:
                lods[a] = -9
            if round_lod:
                lods[a] = round(lods[a])
            iteration += 1

    return lods


# @handle_errors(errors=(FileNotFoundError,))
# def parse_stockholm_to_msa(file):
#     """
#     Args:
#         file (str): The location of a file containing the .fasta records of interest
#     Returns:
#         (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
#                  'msa': (Bio.Align.MultipleSeqAlignment)
#                  'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
#                  'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
#                  'rep': {1: 210, 2:211, ...}}
#             The msa formatted with counts and indexed by residue
#     """
#     return generate_msa_dictionary(read_stockholm_file(file)))


# @handle_errors(errors=(FileNotFoundError,))
# def parse_fasta_to_msa(file):
#     """
#     Args:
#         file (str): The location of a file containing the .fasta records of interest
#     Returns:
#         (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
#                  'msa': (Bio.Align.MultipleSeqAlignment)
#                  'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
#                  'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
#                  'rep': {1: 210, 2:211, ...}}
#             The msa formatted with counts and indexed by residue
#     """
#     return generate_msa_dictionary(msa_from_seq_records(read_fasta_file(file)))


@handle_errors(errors=(FileNotFoundError,))
def parse_hhblits_pssm(file, null_background=True, **kwargs):
    """Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order

    Args:
        file (str): The file to parse, typically with the extension '.hmm'
    Keyword Args:
        null_background=True (bool): Whether to use the null background for the specific protein
    Returns:
        (dict): Dictionary containing residue indexed profile information
        Ex: {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                 'weight': 0.00}, {...}}
    """
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
                    null_bg = {i: {} for i in alph_3_aa}
                    for i, aa in enumerate(alph_3_aa, 1):
                        null_bg[aa] = to_freq(background[i])

            if len(line.split()) == 23:
                items = line.strip().split()
                residue_number = int(items[1])
                pose_dict[residue_number] = {}
                for i, aa in enumerate(protein_letters, 2):
                    pose_dict[residue_number][aa] = to_freq(items[i])
                pose_dict[residue_number]['lod'] = get_lod(pose_dict[residue_number], null_bg)
                pose_dict[residue_number]['type'] = items[0]
                pose_dict[residue_number]['info'] = dummy
                pose_dict[residue_number]['weight'] = dummy

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

    header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in alph_3_aa) \
             + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in alph_3_aa) + '\n'
    footer = ''
    out_file = os.path.join(outpath, name)  # + '.pssm'
    with open(out_file, 'w') as f:
        f.write(header)
        for res in pssm_dict:
            aa_type = pssm_dict[res]['type']
            lod_string = ''
            if lod_freq:
                for aa in alph_3_aa:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
            else:
                for aa in alph_3_aa:  # ensure alpha_3_aa_list for PSSM format
                    lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
            counts_string = ''
            if counts_freq:
                for aa in alph_3_aa:  # ensure alpha_3_aa_list for PSSM format
                    counts_string += '{:>3.0f} '.format(floor(pssm_dict[res][aa] * 100))
            else:
                for aa in alph_3_aa:  # ensure alpha_3_aa_list for PSSM format
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
        for aa in protein_letters:
            pssm[entry][aa] = (alpha[entry] * issm[entry][aa]) + ((1 - alpha[entry]) * pssm[entry][aa])
        logger.info('Residue %d Combined evolutionary and fragment profile: %.0f%% fragment'
                    % (entry + index_offset, alpha[entry] * 100))

    if favor_fragments:
        # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
        # Used to weight fragments higher in design
        boltzman_energy = 1
        favor_seqprofile_score_modifier = 0.2 * CommandDistributer.reference_average_residue_weight
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
                    pssm[entry]['lod'][aa] = exp(pssm[entry]['lod'][aa] / boltzman_energy)
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
        for aa in alph_3_aa:
            if pssm[residue]['lod'][aa] > max_lod:
                max_lod = pssm[residue]['lod'][aa]
                max_res = aa
        consensus_identities[residue + index_offset] = max_res

    return consensus_identities


def sequence_difference(seq1, seq2, d=None, matrix='BLOSUM62'):  # TODO AMS
    """Returns the sequence difference between two sequence iterators

    Args:
        seq1 (any): Either an iterable with residue type as array, or key, with residue type as d[seq1][residue]['type']
        seq2 (any): Either an iterable with residue type as array, or key, with residue type as d[seq2][residue]['type']
    Keyword Args:
        d=None (dict): The dictionary to look up seq1 and seq2 if they are keys in the a dictionary
        matrix='BLOSUM62' (str): The type of matrix to score the sequence differences on
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
    m = substitution_matrices.load(matrix)
    s = 0
    for tup in pairs:
        try:
            s += m[tup]
        except KeyError:
            s += m[(tup[1], tup[0])]

    return s


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


def position_specific_jsd(msa, background):
    """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency

    Both msa_dictionary and background must be the same index
    Args:
        msa (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        background (dict): {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
            Must contain residue index with inner dictionary of single amino acid types
    Returns:
        divergence_dict (dict): {15: 0.732, 16: 0.552, ...}
    """
    return {idx: distribution_divergence(freq, background[idx]) for idx, freq in msa.items() if idx in background}


def distribution_divergence(frequencies, bgd_frequencies, lambda_=0.5):
    """Calculate residue specific Jensen-Shannon Divergence value

    Args:
        frequencies (dict): {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        bgd_frequencies (dict): {'A': 0, 'R': 0, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        (float): 0.732, Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    sum_prob1, sum_prob2 = 0, 0
    for item, frequency in frequencies.items():
        bgd_frequency = bgd_frequencies.get(item)
        try:
            r = (lambda_ * frequency) + ((1 - lambda_) * bgd_frequency)
        except TypeError:  # bgd_frequency is None
            continue
        try:
            try:
                prob2 = (bgd_frequency * log(bgd_frequency / r, 2))
                sum_prob2 += prob2
            except ValueError:  # math domain error
                continue
            try:
                prob1 = (frequency * log(frequency / r, 2))
                sum_prob1 += prob1
            except ValueError:  # math domain error
                continue
        except ZeroDivisionError:  # r = 0
            continue

    return lambda_ * sum_prob1 + (1 - lambda_) * sum_prob2


def msa_from_dictionary(named_sequences):
    """Create a BioPython Multiple Sequence Alignment from a collection of sequences a dictionary
    Args:
        named_sequences (dict): {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        (MultipleSequenceAlignment):
        # (Bio.Align.MultipleSeqAlignment): [SeqRecord(Seq('MNTEELQVAAFEI...', ...), id="Alpha"),
        #                                    SeqRecord(Seq('MNTEEL-VAAFEI...', ...), id="Beta"), ...]
    """
    return MultipleSequenceAlignment(MultipleSeqAlignment([SeqRecord(Seq(sequence),
                                                                     annotations={'molecule_type': 'Protein'}, id=name)
                                                           for name, sequence in named_sequences.items()]))


def msa_from_seq_records(seq_records):
    """Create a BioPython Multiple Sequence Alignment from a collection of sequences a dictionary
    Args:
        seq_records (Iterator[SeqRecord]): {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        (Bio.Align.MultipleSeqAlignment): [SeqRecord(Seq('MNTEELQVAAFEI...', ...), id="Alpha"),
                                           SeqRecord(Seq('MNTEEL-VAAFEI...', ...), id="Beta"), ...]
    """
    return MultipleSeqAlignment(seq_records)


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
            logger.error(key - offset)
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
        if not met_offset_d:  # == dict():  # MET is missing/not the ORF start
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


def generate_alignment(seq1, seq2, matrix='BLOSUM62'):
    """Use Biopython's pairwise2 to generate a local alignment. *Only use for generally similar sequences*

    Returns:

    """
    _matrix = subs_matrices.get(matrix, substitution_matrices.load(matrix))
    gap_penalty = -10
    gap_ext_penalty = -1
    logger.debug('Generating sequence alignment between:\n%s\nAND:\n%s' % (seq1, seq2))
    # Create sequence alignment
    return pairwise2.align.globalds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)
    # return pairwise2.align.localds(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty)


def generate_mutations(mutant, reference, offset=True, blanks=False, termini=False, reference_gaps=False,
                       only_gaps=False):
    """Create mutation data in a typical A5K format. One-indexed dictionary keys with the index matching the reference
     sequence index. Sequence mutations accessed by 'from' and 'to' keys. By default, all gaped sequences are excluded
     from returned mutation dictionary

    For PDB file comparison, mutant should be crystal sequence (ATOM), reference should be expression sequence (SEQRES).
    only_gaps=True will return only the gaped area while blanks=True will return all differences between the alignment
    sequences. termini=True returns missing alignments at the termini

    Args:
        mutant (str): Mutant sequence. Will be in the 'to' key
        reference (str): Wild-type sequence or sequence to reference mutations against. Will be in the 'from' key
    Keyword Args:
        offset=True (bool): Whether sequences are different lengths. Creates a new alignment
        blanks=False (bool): Whether to include all indices that are outside the reference sequence or missing residues
        termini=False (bool): Whether to include indices that are outside the reference sequence boundaries
        reference_gaps=False (bool): Whether to include indices that are missing residues inside the reference sequence
        only_gaps=False (bool): Whether to only include all indices that are missing residues
    Returns:
        (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    """
    # TODO change function name/order of mutant and reference arguments to match logic with 'from' 37 'to' framework
    if offset:
        # alignment = generate_alignment(mutant, reference)
        align_seq_1, align_seq_2, *_ = generate_alignment(mutant, reference)[0]  # alignment[0][0], alignment[0][1]
    else:
        align_seq_1, align_seq_2 = mutant, reference

    # Extract differences from the alignment
    starting_index_of_seq2 = align_seq_2.find(reference[0])  # get the first matching index of the reference sequence
    ending_index_of_seq2 = starting_index_of_seq2 + align_seq_2.rfind(reference[-1])  # find last index of reference
    mutations = {}
    for idx, (seq1_aa, seq2_aa) in enumerate(zip(align_seq_1, align_seq_2), -starting_index_of_seq2 + index_offset):
        if seq1_aa != seq2_aa:
            mutations[idx] = {'from': seq2_aa, 'to': seq1_aa}

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
        mutations.pop(entry, None)

    return mutations


def format_mutations(mutations):
    return ['%s%d%s' % (mutation['from'], index, mutation['to']) for index, mutation in mutations.items()]


def make_mutations_chain_agnostic(mutations):
    """Remove chain identifier from mutation dictionary

    Args:
        mutations (dict): {design: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Returns:
        (dict): {design: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}
    """
    flattened_mutations = {}
    for design, chain_mutations in mutations.items():
        flattened_mutations[design] = {}
        for chain, mutations in chain_mutations.items():
            flattened_mutations[design].update(mutations)

    return flattened_mutations


def simplify_mutation_dict(mutations, to=True):
    """Simplify mutation dictionary to 'to'/'from' AA key

    Args:
        mutations (dict): {pdb: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Keyword Args:
        to=True (bool): Whether to use 'to' AA (True) or 'from' AA (False)
    Returns:
        (dict): {pdb: {mutation_index: 'K', ...}, ...}
    """
    if to:
        simplification = 'to'
        # simplification = get_mutation_to
    else:
        simplification = 'from'
        # simplification = get_mutation_from

    for pdb in mutations:
        for index in mutations[pdb]:
            mutations[pdb][index] = mutations[pdb][index][simplification]
            # mutations[pdb][index] = simplification(mutations[pdb][index])

    return mutations


# def get_mutation_from(mutation_dict):
#     """Remove 'to' identifier from mutation dictionary
#
#     Args:
#         mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
#     Returns:
#         mutation_dict (str): 'A'
#     """
#     return mutation_dict['from']
#
#
# def get_mutation_to(mutation_dict):
#     """Remove 'from' identifier from mutation dictionary
#     Args:
#         mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
#     Returns:
#         mutation_dict (str): 'K'
#     """
#     return mutation_dict['to']


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


def weave_sequence_dict(base_dict=None, **kwargs):
    """Weave together a single dictionary with residue numbers as keys, from separate residue keyed, dictionaries
    All supplied dictionaries must be same integer index for accurate function

    Keyword Args:
        base=None (dict): If a dictionary already exists, pass the dictionary to add residue data to
        **kwargs (dict): keyword=dictionary pairs. Ex: sorted_freq={16: ['S', 'A', ...], ... },
            mut_prob={16: {'A': 0.05, 'C': 0.01, ...}, ...}, jsd={16: 0.732, 17: 0.552, ...}
    Returns:
        (dict): {16: {'mut_prob': {'A': 0.05, 'C': 0.01, ...}, 'jsd': 0.732, 'sorted_freq': ['S', 'A', ...]}, ...}
    """
    if not base_dict:
        base_dict = {}

    for observation_type, sequence_data in kwargs.items():
        for residue, value in sequence_data.items():
            if residue not in base_dict:
                base_dict[residue] = {}
            # else:
            #     weaved_dict[residue][observation_type] = {}
            # if isinstance(value, dict):  # TODO make endlessly recursive?
                # base_dict[residue][observation_type] = dict(sub_item for sub_item in value.items())
                # base_dict[residue][observation_type] = value
            # else:
            base_dict[residue][observation_type] = value

    return base_dict


# def get_paired_fragment_metrics(separated_fragment_metrics):
#     # -------------------------------------------
#     # score the interface separately
#     mapped_all_score, mapped_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped'])
#     paired_all_score, paired_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired'])
#     # combine
#     all_residue_score = mapped_all_score + paired_all_score
#     center_residue_score = mapped_center_score + paired_center_score
#     # -------------------------------------------
#     # Get the number of CENTRAL residues with overlapping fragments given z_value criteria TOTAL interface
#     central_residues_with_fragment_overlap = len(interface_residues_with_fragment_overlap['mapped']) + len(interface_residues_with_fragment_overlap['paired'])
#     # Get the number of TOTAL residues with overlapping fragments given z_value criteria
#     total_residues_with_fragment_overlap = len(entity1_match_scores) + len(entity2_match_scores)
#
#     total_residue_observations = len(fragment_matches) * 2
#     if central_residues_with_fragment_overlap > 0:
#         multiple_frag_ratio = total_residue_observations / central_residues_with_fragment_overlap
#     else:
#         multiple_frag_ratio = 0
#
#     interface_residue_count = len(interface_residues_with_fragment_overlap['mapped']) + \
#         len(interface_residues_with_fragment_overlap['paired'])
#     if interface_residue_count > 0:
#         percent_interface_matched = central_residues_with_fragment_overlap / float(interface_residue_count)
#         percent_interface_covered = total_residues_with_fragment_overlap / float(interface_residue_count)
#     else:
#         percent_interface_matched, percent_interface_covered = 0, 0
#
#     # Sum the total contribution from each fragment type on both sides of the interface
#     total_fragment_content = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
#     for index in fragment_i_index_count_d:
#         total_fragment_content[index] += fragment_i_index_count_d[index]
#         total_fragment_content[index] += fragment_j_index_count_d[index]
#
#     if total_residue_observations > 0:
#         for index in total_fragment_content:
#             total_fragment_content[index] = total_fragment_content[index] / total_residue_observations
#
#     return all_residue_score, center_residue_score, total_residues_with_fragment_overlap, \
#         central_residues_with_fragment_overlap, multiple_frag_ratio, total_fragment_content
#         # interface_residue_count, percent_interface_matched, percent_interface_covered,


# def residue_number_to_object(pdb, residue_dict):  # TODO DEPRECIATE
#     """Convert sets of residue numbers to sets of PDB.Residue objects
#
#     Args:
#         pdb (PDB): PDB object to extract residues from. Chain order matches residue order in residue_dict
#         residue_dict (dict): {'key1': [(78, 87, ...),], ...} - Entry mapped to residue sets
#     Returns:
#         residue_dict - {'key1': [(residue1_ca_atom, residue2_ca_atom, ...), ...] ...}
#     """
#     for entry in residue_dict:
#         pairs = []
#         for _set in range(len(residue_dict[entry])):
#             residue_obj_set = []
#             for i, residue in enumerate(residue_dict[entry][_set]):
#                 resi_object = Residue(pdb.getResidueAtoms(pdb.chain_id_list[i], residue)).ca
#                 assert resi_object, DesignError('Residue \'%s\' missing from PDB \'%s\'' % (residue, pdb.filepath))
#                 residue_obj_set.append(resi_object)
#             pairs.append(tuple(residue_obj_set))
#         residue_dict[entry] = pairs
#
#     return residue_dict


def clean_gapped_columns(alignment_dict, correct_index):  # UNUSED
    """Cleans an alignment dictionary by revising key list with correctly indexed positions. 0 indexed"""
    return {i: alignment_dict[index] for i, index in enumerate(correct_index)}


def weight_sequences(alignment, bio_alignment, column_counts=None):
    """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment

    Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count)) as was described by Heinkoff and Heinkoff, 1994
    Args:
        alignment (dict): {1: {'A': 31, 'C': 0, ...}, 2: {}, ...}
        bio_alignment (biopython.MultipleSeqAlignment):
        column_counts=None (dict): The indexed counts for each column in the msa
    Returns:
        (dict): { 1: 2.390, 2: 2.90, 3:5.33, 4: 1.123, ...} - sequence_in_MSA: sequence_weight_factor
    """
    if not column_counts:
        column_counts = {}
        for idx, amino_acid_counts in alignment.items():
            s = 0  # column amino acid representation
            for aa in amino_acid_counts:
                if aa == '-':
                    continue
                elif amino_acid_counts[aa] > 0:
                    s += 1
            column_counts[idx] = s

    sequence_weights = {}
    for k, record in enumerate(bio_alignment):
        s = 0  # "diversity/surprise"
        for j, aa in enumerate(record.seq):
            s += (1 / (column_counts[j] * alignment[j][aa]))
        sequence_weights[k] = s

    return sequence_weights


msa_supported_types = {'fasta': '.fasta', 'stockholm': '.sto'}
msa_generation_function = 'SequenceProfile.hhblits()'


class MultipleSequenceAlignment:  # (MultipleSeqAlignment):

    def __init__(self, alignment=None, aligned_sequence=None, alphabet='-' + extended_protein_letters,
                 weight_alignment_by_sequence=False, sequence_weights=None, **kwargs):
        """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed

        gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
        probability of residue i in the specific column more accurately.
        Keyword Args:
            bio_alignment=None ((Bio.Align.MultipleSeqAlignment)): "Array" of SeqRecords
            aligned_sequence=None (str): Provide the sequence on which the alignment is based, otherwise the first
            sequence will be used
            alphabet=extended_protein_letters (str): '-ACDEFGHIKLMNPQRSTVWYBXZJUO'
            weight_alignment_by_sequence=False (bool): If weighting should be performed. Use in cases of
                unrepresentative sequence population in the MSA
            sequence_weights=None (dict): If the alignment should be weighted, and weights are already available, the
                weights for each sequence
            gaps=False (bool): Whether gaps (-) should be counted in column weights
        Sets:
            alignment - (Bio.Align.MultipleSeqAlignment)
            number_of_sequences - 214
            query - 'MGSTHLVLK...'
            query_with_gaps - 'MGS--THLVLK...'
            counts - {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
            frequencies - {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
            observations - {1: 210, 2:211, ...}}
        """
        if not alignment:
            pass
        else:
            if not aligned_sequence:
                aligned_sequence = str(alignment[0].seq)
            # Add Info to 'meta' record as needed and populate a amino acid count dict (one-indexed)
            self.alignment = alignment
            self.number_of_sequences = len(alignment)
            self.length = alignment.get_alignment_length()
            self.query = aligned_sequence.replace('-', '')
            self.query_length = len(self.query)
            self.query_with_gaps = aligned_sequence
            self.counts =\
                SequenceProfile.populate_design_dictionary(self.length, alphabet, dtype=int)
            for record in self.alignment:
                for i, aa in enumerate(record.seq, 1):
                    self.counts[i][aa] += 1

            self.observations = find_column_observations(self.counts, **kwargs)
            if weight_alignment_by_sequence:
                sequence_weights = weight_sequences(self.counts, self.alignment, column_counts=self.observations)

            if sequence_weights:  # overwrite the current counts with weighted counts
                self.sequence_weights = sequence_weights
                for record in self.alignment:
                    for i, aa in enumerate(record.seq, 1):
                        self.counts[i][aa] += sequence_weights[i]
            else:
                self.sequence_weights = []

            self.frequencies = {}
            self.msa_to_prob_distribution()

    @classmethod
    def from_stockholm(cls, file, **kwargs):
        try:
            return cls(alignment=read_alignment(file, alignment_type='stockholm'), **kwargs)
        except FileNotFoundError:
            raise DesignError('The file requested \'%s\'for multiple sequence alignemnt doesn\'t exist' % file)

    @classmethod
    def from_fasta(cls, file):
        try:
            return cls(alignment=read_alignment(file))
        except FileNotFoundError:
            raise DesignError('The file requested \'%s\'for multiple sequence alignemnt doesn\'t exist' % file)

    def msa_to_prob_distribution(self):
        """Turn Alignment dictionary into a probability distribution

        Sets:
            (dict): self.frequencies - {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...}
        """
        for residue, amino_acid_counts in self.counts.items():
            total_column_weight = self.observations[residue]
            assert total_column_weight != 0, '%s: Processing error... Downstream cannot divide by 0. Position = %s' \
                                             % (MultipleSequenceAlignment.msa_to_prob_distribution.__name__, residue)  # Todo correct?
            self.frequencies[residue] = {aa: count / total_column_weight for aa, count in amino_acid_counts.items()}

# def generate_msa_dictionary(bio_alignment, aligned_sequence=None, alphabet=protein_letters,
#                             weight_alignment_by_sequence=False, sequence_weights=None, **kwargs):
#     """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed
#
#     gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
#     probability of residue i in the specific column more accurately.
#     Args:
#         bio_alignment ((Bio.Align.MultipleSeqAlignment)): "Array" of SeqRecords
#     Keyword Args:
#         aligned_sequence=None (str): Provide the sequence on which the alignment is based, otherwise the first sequence
#             will be used
#         alphabet=protein_letters (str): 'ACDEFGHIKLMNPQRSTVWY'
#         weight_alignment_by_sequence=False (bool): If weighting should be performed
#         sequence_weights=None (dict): If the alignment should be weighted, and weights are already available, the
#             weights for each sequence
#         gaps=False (bool): Whether gaps (-) should be counted in column weights
#     Returns:
#         (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
#                  'msa': (Bio.Align.MultipleSeqAlignment)
#                  'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
#                  'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
#                  'rep': {1: 210, 2:211, ...}}
#             The msa formatted with counts and indexed by residue
#     """
#     if not aligned_sequence:
#         aligned_sequence = str(bio_alignment[0].seq)
#     # Add Info to 'meta' record as needed and populate a amino acid count dict (one-indexed)
#     alignment = {'msa' : bio_alignment,
#         'meta': {'num_sequences': len(bio_alignment), 'query': aligned_sequence.replace('-', ''),
#                   'query_with_gaps': aligned_sequence},
#         'counts': SequenceProfile.populate_design_dictionary(bio_alignment.get_alignment_length(), alphabet, dtype=int)}
#     for record in bio_alignment:
#         for i, aa in enumerate(record.seq, 1):
#             alignment['counts'][i][aa] += 1
#
#     alignment['rep'] = add_column_weight(alignment['counts'], **kwargs)
#     if weight_alignment_by_sequence:
#         sequence_weights = weight_sequences(alignment['counts'], bio_alignment, column_counts=alignment['rep'])
#
#     if sequence_weights:  # overwrite the current counts with weighted counts
#         for record in bio_alignment:
#             for i, aa in enumerate(record.seq, 1):
#                 alignment['counts'][i][aa] += sequence_weights[i]
#
#     return msa_to_prob_distribution(alignment)


def find_column_observations(counts, **kwargs):
    """Find total representation for each column in the alignment

    Args:
        counts (dict): {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...}
    Keyword Args:
        gaps=False (bool): Whether to count gaps (True) or not in the alignment
    Returns:
        (dict): {1: 210, 2:211, ...}
    """
    return {idx: sum_column_observations(aa_counts, **kwargs) for idx, aa_counts in counts.items()}


def sum_column_observations(column, gaps=False, **kwargs):
    """Sum the column weight for a single alignment dict column

    Args:
        column (dict): {'A': 13, 'C': 1, 'D': 23, ...}
    Keyword Args:
        gaps=False (bool): Whether to count gaps (True) or not
    Returns:
        s (int): Total counts in the alignment
    """
    if not gaps:
        column.pop('-')

    return sum(column.values())


def msa_to_prob_distribution(alignment):
    """Turn Alignment dictionary into a probability distribution

    Args:
        alignment (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
                           'msa': (Bio.Align.MultipleSeqAlignment)
                           'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
                           'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
                           'rep': {1: 210, 2:211, ...}}
            The msa formatted with counts and indexed by residue
    Returns:
        (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
                 'msa': (Bio.Align.MultipleSeqAlignment)
                 'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
                 'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
                 'rep': {1: 210, 2:211, ...}}
            The msa formatted with counts and indexed by residue
    """
    alignment['frequencies'] = {}
    for residue, amino_acid_counts in alignment['counts'].items():
        total_column_weight = alignment['rep'][residue]
        assert total_column_weight != 0, '%s: Processing error... Downstream cannot divide by 0. Position = %s' \
                                         % (msa_to_prob_distribution.__name__, residue)  # Todo correct?
        alignment['frequencies'][residue] = {aa: count / total_column_weight for aa, count in amino_acid_counts.items()}

    return alignment


def jensen_shannon_divergence(multiple_sequence_alignment, background_aa_probabilities, lambda_=0.5):
    """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict

    Args:
        multiple_sequence_alignment (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        background_aa_probabilities (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        (dict): {15: 0.732, ...} Divergence per residue bounded between 0 and 1. 1 is more divergent from background
    """
    return {residue: distribution_divergence(aa_probabilities, background_aa_probabilities, lambda_=lambda_)
            for residue, aa_probabilities in multiple_sequence_alignment.items()}


def weight_gaps(divergence, representation, alignment_length):  # UNUSED
    for i in range(len(divergence)):
        divergence[i] = divergence[i] * representation[i] / alignment_length

    return divergence


def window_score(score_dict, window_len, score_lambda=0.5):  # UNUSED
    """Takes a MSA score dict and transforms so that each position is a weighted average of the surrounding positions.
    Positions with scores less than zero are not changed and are ignored calculation

    Modified from Capra and Singh 2007 code
    Args:
        score_dict (dict):
        window_len (int): Number of residues on either side of the current residue
    Keyword Args:
        lamda=0.5 (float): Float between 0 and 1
    Returns:
        (dict):
    """
    if window_len == 0:
        return score_dict
    else:
        window_scores = {}
        for i in range(len(score_dict) + index_offset):
            s, number_terms = 0, 0
            if i <= window_len:
                for j in range(1, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            elif i + window_len > len(score_dict):
                for j in range(i - window_len, len(score_dict) + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            else:
                for j in range(i - window_len, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            window_scores[i] = (1 - score_lambda) * (s / number_terms) + score_lambda * score_dict[i]

        return window_scores


def rank_possibilities(probability_dict):  # UNUSED
    """Gather alternative residues and sort them by probability.

    Args:
        probability_dict (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
    Returns:
         sorted_alternates_dict (dict): {15: ['S', 'A', 'T'], ... }
    """
    sorted_alternates_dict = {}
    for residue in probability_dict:
        residue_probability_list = []
        for aa in probability_dict[residue]:
            if probability_dict[residue][aa] > 0:
                residue_probability_list.append((aa, round(probability_dict[residue][aa], 5)))  # tuple instead of list
        residue_probability_list.sort(key=lambda tup: tup[1], reverse=True)
        # [('S', 0.13190), ('A', 0.0500), ...]
        sorted_alternates_dict[residue] = [aa[0] for aa in residue_probability_list]

    return sorted_alternates_dict


# def process_alignment(bio_alignment, **kwargs):
#     """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed
#
#     gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
#     probability of residue i in the specific column more accurately.
#     Args:
#         bio_alignment (MultipleSeqAlignment): List of SeqRecords
#     Keyword Args:
#         weight_sequences=False (bool): Whether sequences should be weighted by their information content
#         gaps=False (bool): Whether gaps (-) should be counted in column weights
#         aligned_sequence=None (str): Provide the sequence on which the alignment is based, otherwise the first sequence
#             will be used
#         alphabet=protein_letters (str): 'ACDEFGHIKLMNPQRSTVWY'
#         sequence_weights=None (dict): If the alignment should be weighted, a dictionary with weights for each sequence
#     Returns:
#         (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
#                  'msa': (Bio.Align.MultipleSeqAlignment)
#                  'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
#                  'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
#                  'rep': {1: 210, 2:211, ...}}
#             The msa formatted with counts and indexed by residue
#     """
#     if weight_sequences:
#         kwargs['sequence_weights'] = weight_sequences()
#     alignment = generate_msa_dictionary(bio_alignment, **kwargs)
#     # alignment['rep'] = add_column_weight(alignment['counts'], **kwargs)
#
#     return msa_to_prob_distribution(alignment)


def multi_chain_alignment(mutated_sequences):
    """Combines different chain's Multiple Sequence Alignments into a single MSA. One-indexed

    Args:
        mutated_sequences (dict): {chain: {name: sequence, ...}
    Returns:
        (MultipleSequenceAlignment): The MSA object with counts, frequencies, sequences, and indexed by residue
    """
    #         (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...', 'query_with_gaps': 'MGS--THLVLK...'},
    #                  'msa': (Bio.Align.MultipleSeqAlignment)
    #                  'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
    #                  'frequencies': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
    #                  'rep': {1: 210, 2:211, ...}}
    #             The msa formatted with counts and indexed by residue

    # Combine alignments for all chains from design file Ex: A: 1-102, B: 1-130. Alignment: 1-232
    total_alignment = None
    for idx, named_sequences in enumerate(mutated_sequences.values()):
        if idx == 0:
            total_alignment = msa_from_dictionary(named_sequences)[:, :]
        else:
            total_alignment += msa_from_dictionary(named_sequences)[:, :]

    if total_alignment:
        # return generate_msa_dictionary(total_alignment)
        return MultipleSequenceAlignment(alignment=total_alignment)
    else:
        raise DesignError('%s - No sequences were found!' % multi_chain_alignment.__name__)


# def generate_all_design_mutations(all_design_files, wild_type_file, pose_num=False):  # Todo DEPRECIATE
#     """From a list of PDB's and a wild-type PDB, generate a list of 'A5K' style mutations
#
#     Args:
#         all_design_files (list): PDB files on disk to extract sequence info and compare
#         wild_type_file (str): PDB file on disk which contains a reference sequence
#     Returns:
#         mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
#     """
#     wild_type_pdb = PDB.from_file(wild_type_file, log=None, entities=False)
#     pdb_sequences = {}
#     for file_name in all_design_files:
#         pdb = PDB.from_file(file_name, log=None, entities=False)
#         pdb_sequences[pdb.name] = pdb.atom_sequences
#
#     return generate_multiple_mutations(wild_type_pdb.atom_sequences, pdb_sequences, pose_num=pose_num)


def pdb_to_pose_offset(reference_sequence):
    """Take a dictionary with chain name as keys and return the length of Pose numbering offset

    Args:
        reference_sequence (dict(iter)): {key1: 'MSGKLDA...', ...} or {key2: {1: 'A', 2: 'S', ...}, ...}
    Order of dictionary must maintain chain order, so 'A', 'B', 'C'. Python 3.6+ should be used
    Returns:
        (dict): {key1: 0, key2: 123, ...}
    """
    offset = {}
    # prior_chain = None
    prior_chains_len = 0
    for i, key in enumerate(reference_sequence):
        if i > 0:
            prior_chains_len += len(reference_sequence[prior_key])
        offset[key] = prior_chains_len
        # insert function here? Make this a decorator!?
        prior_key = key

    return offset


def generate_multiple_mutations(reference, sequences, pose_num=True):
    """Extract mutation data from multiple sequence dictionaries with regard to a reference. Default is Pose numbering

    Args:
        reference (dict[mapping[str, str]]): {chain: sequence, ...} The reference sequence to compare sequences to
        sequences (dict[mapping[str, dict[mapping[str, str]]): {pdb_code: {chain: sequence, ...}, ...}
    Keyword Args:
        pose_num=True (bool): Whether to return the mutations in Pose numbering with the first Entity as 1 and the
        second Entity as Entity1 last residue + 1
    Returns:
        (dict): {pdb_code: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    """
    # add reference sequence mutations
    mutations = {'reference': {chain: {sequence_idx: {'from': aa, 'to': aa}
                                       for sequence_idx, aa in enumerate(ref_sequence, 1)}
                               for chain, ref_sequence in reference.items()}}
    #                         returns {1: {'from': 'A', 'to': 'K'}, ...}
    # mutations = {pdb: {chain: generate_mutations(sequence, reference[chain], offset=False)
    #                    for chain, sequence in chain_sequences.items()}
    #              for pdb, chain_sequences in pdb_sequences.items()}
    try:
        for name, chain_sequences in sequences.items():
            mutations[name] = {}
            for chain, sequence in chain_sequences.items():
                mutations[name][chain] = generate_mutations(sequence, reference[chain], offset=False)
    except KeyError:
        raise DesignError('The reference sequence and mutated_sequences have different chains! Chain %s isn\'t in the '
                          'reference' % chain)
    if pose_num:
        offset_dict = pdb_to_pose_offset(reference)
        # pose_mutations = {}
        # for chain, offset in offset_dict.items():
        #     for pdb_code in mutations:
        #         if pdb_code not in pose_mutations:
        #             pose_mutations[pdb_code] = {}
        #         pose_mutations[pdb_code][chain] = {}
        #         for mutation_idx in mutations[pdb_code][chain]:
        #             pose_mutations[pdb_code][chain][mutation_idx + offset] = mutations[pdb_code][chain][mutation_idx]
        # mutations = pose_mutations
        mutations = {name: {chain: {idx + offset: mutation for idx, mutation in chain_mutations[chain].iems()}
                            for chain, offset in offset_dict.items()} for name, chain_mutations in mutations.items()}
    return mutations


def generate_mutations_from_reference(reference, sequences):  # , pose_num=True):
    """Extract mutation data from multiple sequence dictionaries with regard to a reference. Default is Pose numbering

    Args:
        reference (str): The reference sequence to compare sequences to
        sequences (dict[mapping[str, str]]): {pdb_code: sequence, ...}
    Keyword Args:
        # pose_num=True (bool): Whether to return the mutations in Pose numbering with the first Entity as 1 and the
        second Entity as Entity1 last residue + 1
    Returns:
        (dict): {pdb_code: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    """
    # mutations = {'reference': {chain: {sequence_idx: {'from': aa, 'to': aa}
    #                                    for sequence_idx, aa in enumerate(ref_sequence, 1)}
    #                            for chain, ref_sequence in reference.items()}}
    #                         returns {1: {'from': 'A', 'to': 'K'}, ...}
    # mutations = {pdb: {chain: generate_mutations(sequence, reference[chain], offset=False)
    #                    for chain, sequence in chain_sequences.items()}
    #              for pdb, chain_sequences in pdb_sequences.items()}
    mutations = {name: generate_mutations(sequence, reference, offset=False) for name, sequence in sequences.items()}
    # try:
    #     for pdb, chain_sequences in sequences.items():
    #         mutations[pdb] = {}
    #         for chain, sequence in chain_sequences.items():
    #             mutations[pdb][chain] = generate_mutations(sequence, reference[chain], offset=False)
    # except KeyError:
    #     raise DesignError('The reference sequence and mutated_sequences have different chains! Chain %s isn\'t in the '
    #                       'reference' % chain)
    # add reference sequence mutations
    mutations['reference'] = {sequence_idx: {'from': aa, 'to': aa} for sequence_idx, aa in enumerate(reference, 1)}

    # if pose_num:
    #     offset_dict = pdb_to_pose_num(reference)
    #     # pose_mutations = {}
    #     # for chain, offset in offset_dict.items():
    #     #     for pdb_code in mutations:
    #     #         if pdb_code not in pose_mutations:
    #     #             pose_mutations[pdb_code] = {}
    #     #         pose_mutations[pdb_code][chain] = {}
    #     #         for mutation_idx in mutations[pdb_code][chain]:
    #     #             pose_mutations[pdb_code][chain][mutation_idx + offset] = mutations[pdb_code][chain][mutation_idx]
    #     # mutations = pose_mutations
    #     mutations = {pdb_code: {chain: {idx + offset: mutation for idx, mutation in chain_mutations[chain].iems()}
    #                             for chain, offset in offset_dict.items()}
    #                  for pdb_code, chain_mutations in mutations.items()}

    return mutations

# def extract_aa_seq(pdb, aa_code=1, source='atom', chain=0):
#     """Extracts amino acid sequence from either ATOM or SEQRES record of PDB object
#     Returns:
#         (str): Sequence of PDB
#     """
#     if type(chain) == int:
#         chain = pdb.chain_id_list[chain]
#     final_sequence = None
#     sequence_list = []
#     failures = []
#     aa_code = int(aa_code)
#
#     if source == 'atom':
#         # Extracts sequence from ATOM records
#         if aa_code == 1:
#             for atom in pdb.all_atoms:
#                 if atom.chain == chain and atom.type == 'N' and (atom.alt_location == '' or atom.alt_location == 'A'):
#                     try:
#                         sequence_list.append(protein_letters_3to1[atom.residue_type.title()])
#                     except KeyError:
#                         sequence_list.append('X')
#                         failures.append((atom.residue_number, atom.residue_type))
#             final_sequence = ''.join(sequence_list)
#         elif aa_code == 3:
#             for atom in pdb.all_atoms:
#                 if atom.chain == chain and atom.type == 'N' and atom.alt_location == '' or atom.alt_location == 'A':
#                     sequence_list.append(atom.residue_type)
#             final_sequence = sequence_list
#         else:
#             logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))
#
#     elif source == 'seqres':
#         # Extract sequence from the SEQRES record
#         sequence = pdb.seqres_sequences[chain]
#         # fail = False
#         # while True:  # TODO WTF is this used for
#         #     if chain in pdb.seqres_sequences:
#         #         sequence = pdb.seqres_sequences[chain]
#         #         break
#         #     else:
#         #         if not fail:
#         #             temp_pdb = PDB.from_file(file=pdb.filepath)
#         #             fail = True
#         #         else:
#         #             raise DesignError('Invalid PDB input, no SEQRES record found')
#         if aa_code == 1:
#             final_sequence = sequence
#             for i in range(len(sequence)):
#                 if sequence[i] == 'X':
#                     failures.append((i, sequence[i]))
#         elif aa_code == 3:
#             for i, residue in enumerate(sequence):
#                 sequence_list.append(protein_letters_1to3[residue])
#                 if residue == 'X':
#                     failures.append((i, residue))
#             final_sequence = sequence_list
#         else:
#             logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))
#     else:
#         raise DesignError('Invalid sequence input')
#
#     return final_sequence, failures


def make_sequences_from_mutations(wild_type, pdb_mutations, aligned=False):
    """Takes a list of sequence mutations and returns the mutated form on wildtype

    Args:
        wild_type (str): Sequence to mutate
        pdb_mutations (dict): {name: {mutation_index: {'from': AA, 'to': AA}, ...}, ...}, ...}
    Keyword Args:
        aligned=False (bool): Whether the input sequences are already aligned
    Returns:
        all_sequences (dict): {name: sequence, ...}
    """
    return {pdb: make_mutations(wild_type, mutations, find_orf=not aligned) for pdb, mutations in pdb_mutations.items()}


def generate_sequences(wild_type_sequences, all_design_mutations):
    """Separate chains from mutation dictionary and generate mutated sequences

    Args:
        wild_type_sequences (dict): {chain: sequence, ...}
        all_design_mutations (dict): {'name': {chain: {mutation_index: {'from': AA, 'to': AA}, ...}, ...}, ...}
            Index so mutation_index starts at 1
    Returns:
        mutated_sequences (dict): {chain: {name: sequence, ...}
    """
    mutated_sequences = {}
    for chain in wild_type_sequences:
        # pdb_chain_mutations = {pdb: chain_mutations.get(chain) for pdb, chain_mutations in all_design_mutations.items()}
        pdb_chain_mutations = {}
        for pdb, chain_mutations in all_design_mutations.items():
            if chain in chain_mutations:
                pdb_chain_mutations[pdb] = all_design_mutations[pdb][chain]
        mutated_sequences[chain] = make_sequences_from_mutations(wild_type_sequences[chain], pdb_chain_mutations,
                                                                 aligned=True)

    return mutated_sequences


def hydrophobic_collapse_index(sequence, hydrophobicity='standard'):  # TODO Validate
    """Calculate hydrophobic collapse index for a particular sequence of an iterable object and return a HCI array

    Args:
        sequence (str): The sequence to measure
    Keyword Args:
        hydrophobicity='standard' (str): The degree of hydrophobicity to consider. Either 'standard' (FILV) or 'expanded' (FILMVWY)
    Returns:
        (numpy.ndarray): 1D array with the mean collapse score for every position on the input sequence
    """
    sequence_length = len(sequence)
    lower_range, upper_range, range_correction = 3, 9, 1
    range_size = upper_range - lower_range + range_correction
    yes = 1
    no = 0
    if hydrophobicity == 'background':  # Todo
        raise DesignError('This function is not yet possible')
        # hydrophobicity_values = \
        #     {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
        #      'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0, 'X': 0, 'Z': 0}
    elif hydrophobicity == 'expanded':
        hydrophobicity_values = \
            {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 1, 'N': 0, 'P': 0,
             'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 1, 'Y': 1, 'B': 0, 'J': 0, 'O': 0, 'U': 0, 'X': 0, 'Z': 0}
    else:  # hydrophobicity == 'standard':
        hydrophobicity_values = \
            {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 0, 'N': 0, 'P': 0,
             'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 0, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0, 'X': 0, 'Z': 0}

    sequence_array = [hydrophobicity_values[aa] for aa in sequence]

    # make an array with # of rows equal to upper range (+1 for indexing), length equal to # of letters in sequence
    window_array = np.zeros((range_size, sequence_length))
    for array_idx, window_size in enumerate(range(lower_range, upper_range + range_correction)):
        # iterate over the window range
        window_spread = math.floor(window_size / 2)
        # check if the range is odd or even, then calculate score accordingly, with cases for N- and C-terminal windows
        if window_size % 2 == 1:  # range is odd
            for seq_idx in range(sequence_length):
                position_sum = 0
                if seq_idx < window_spread:  # N-terminus
                    for window_position in range(seq_idx + window_spread + range_correction):
                        position_sum += sequence_array[window_position]
                elif seq_idx + window_spread >= sequence_length:  # C-terminus
                    for window_position in range(seq_idx - window_spread, sequence_length):
                        position_sum += sequence_array[window_position]
                else:
                    for window_position in range(seq_idx - window_spread, seq_idx + window_spread + range_correction):
                        position_sum += sequence_array[window_position]
                window_array[array_idx][seq_idx] = position_sum / window_size
        else:  # range is even
            for seq_idx in range(sequence_length):
                position_sum = 0
                if seq_idx < window_spread:  # N-terminus
                    for window_position in range(seq_idx + window_spread + range_correction):
                        if window_position == seq_idx + window_spread:
                            position_sum += 0.5 * sequence_array[window_position]
                        else:
                            position_sum += sequence_array[window_position]
                elif seq_idx + window_spread >= sequence_length:  # C-terminus
                    for window_position in range(seq_idx - window_spread, sequence_length):
                        if window_position == seq_idx - window_spread:
                            position_sum += 0.5 * sequence_array[window_position]
                        else:
                            position_sum += sequence_array[window_position]
                else:
                    for window_position in range(seq_idx - window_spread, seq_idx + window_spread + range_correction):
                        if window_position == seq_idx - window_spread \
                                or window_position == seq_idx + window_spread + range_correction:
                            position_sum += 0.5 * sequence_array[window_position]
                        else:
                            position_sum += sequence_array[window_position]
                window_array[array_idx][seq_idx] = position_sum / window_size
    logger.debug('Hydrophobic Collapse window values:\n%s' % window_array)
    hci = window_array.mean(axis=0)
    logger.debug('Hydrophobic Collapse Index:\n%s' % hci)

    return hci
    # hci = np.zeros(sequence_length)  # [0] * (sequence_length + 1)
    # for seq_idx in range(sequence_length):
    #     for window_size in range(lower_range, upper_range + range_correction):
    #         hci[seq_idx] += window_array[window_size][seq_idx]
    #     hci[seq_idx] /= range_size
    #     hci[seq_idx] = round(hci[seq_idx], 3)

    # return hci


@handle_errors(errors=(FileNotFoundError,))
def read_fasta_file(file_name, **kwargs):
    """Open a fasta file and return a parser object to load the sequences to SeqRecords
    Returns:
        (Iterator[SeqRecords]): Ex. [record1, record2, ...]
    """
    return SeqIO.parse(file_name, 'fasta')


@handle_errors(errors=(FileNotFoundError,))
def read_alignment(file_name, alignment_type='fasta', **kwargs):
    """Open a fasta file and return a parser object to load the sequences to SeqRecords
    Returns:
        (Iterator[SeqRecords]): Ex. [record1, record2, ...]
    """
    # return AlignIO.read(file_name, 'stockholm')
    return AlignIO.read(file_name, alignment_type)


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
