from __future__ import annotations

import math
import os
import subprocess
import time
from collections import namedtuple, defaultdict
from copy import deepcopy, copy
from itertools import repeat
from logging import Logger
from math import floor, exp, log2
from pathlib import Path
from typing import Sequence, Any, Iterable, get_args, Literal, Iterator, AnyStr

import numpy as np
import pandas as pd
from Bio import pairwise2, SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.Data.IUPACData import protein_letters, extended_protein_letters, protein_letters_3to1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import CommandDistributer
import info
import PathUtils as PUtils
from SymDesignUtils import handle_errors, unpickle, get_base_root_paths_recursively, DesignError, start_log, pretty_format_table
# import dependencies.bmdca as bmdca

# Globals
logger = start_log(name=__name__)
index_offset = 1
alignment_types_literal = Literal['mapped', 'paired']
alignment_types: tuple[alignment_types_literal, ...] = get_args(alignment_types_literal)
sequence_type_literal = Literal['reference', 'structure']
sequence_types: tuple[sequence_type_literal, ...] = get_args(sequence_type_literal)
alph_3_aa: tuple[info.protein_letter_literal, ...] = get_args(info.protein_letter_literal)
protein_letter_plus_literals = Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                       'T', 'W', 'Y', 'V', 'lod', 'type', 'info', 'weight']
aa_counts = dict(zip(protein_letters, repeat(0)))
aa_weighted_counts = dict(zip(protein_letters, repeat(0)))
aa_weighted_counts.update({'stats': [0, 1]})
hydrophobicity_scale = \
    {'expanded': {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 1, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 1, 'Y': 1, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0},
     'standard': {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 0, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0}}
# add_fragment_profile_instructions = 'To add fragment information, call Pose.generate_interface_fragments()'
subs_matrices = {'BLOSUM62': substitution_matrices.load('BLOSUM62')}

protein_letters_1aa_literal = Literal[tuple(protein_letters)]
# protein_letters_literal: tuple[str, ...] = get_args(protein_letters_1aa_literal)
gapped_protein_letters = protein_letters + '-'
numerical_translation = dict(zip(protein_letters, range(len(protein_letters))))
gapped_numerical_translation = defaultdict(lambda: 20, zip(gapped_protein_letters, range(len(gapped_protein_letters))))
extended_protein_letters_literal = Literal[tuple(extended_protein_letters)]
# extended_protein_letters: tuple[str, ...] = get_args(extended_protein_letters_literal)
extended_protein_letters_and_gap_literal = Literal['-', get_args(extended_protein_letters_literal)]
extended_protein_letters_and_gap: tuple[str, ...] = get_args(extended_protein_letters_and_gap_literal)


class MultipleSequenceAlignment:
    _array: np.ndarray
    _numerical_alignment: np.ndarray
    _sequence_index: np.ndarray
    alignment: MultipleSeqAlignment
    frequencies: np.ndarray
    number_of_sequences: int
    length: int
    query: str
    query_length: int
    query_with_gaps: str
    # counts: list[dict[extended_protein_letters_and_gap, int]]
    counts: list[list[int]] | np.ndarray

    def __init__(self, alignment: MultipleSeqAlignment = None, aligned_sequence: str = None,
                 alphabet: str = gapped_protein_letters,
                 weight_alignment_by_sequence: bool = False, sequence_weights: list[float] = None,
                 count_gaps: bool = False, **kwargs):
        """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed

        gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
        probability of residue i in the specific column more accurately.

        Args:
            alignment: "Array" of SeqRecords
            aligned_sequence: Provide the sequence on which the alignment is based, otherwise the first
                sequence will be used
            alphabet: 'ACDEFGHIKLMNPQRSTVWY-'
            weight_alignment_by_sequence: If weighting should be performed. Use in cases of
                unrepresentative sequence population in the MSA
            sequence_weights: If the alignment should be weighted, and weights are already available, the
                weights for each sequence
            count_gaps: Whether gaps (-) should be counted in column weights
        Sets:
            alignment - (Bio.Align.MultipleSeqAlignment)
            number_of_sequences - 214
            query - 'MGSTHLVLK...'
            query_with_gaps - 'MGS--THLVLK...'
            counts - {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
            frequencies - {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
            observations - {1: 210, 2:211, ...}}
        """
        if alignment is not None:
            if not aligned_sequence:
                aligned_sequence = str(alignment[0].seq)
            # Add Info to 'meta' record as needed and populate an amino acid count dict (one-indexed)
            self.alignment = alignment
            self.number_of_sequences = len(alignment)
            self.length = alignment.get_alignment_length()
            self.query = aligned_sequence.replace('-', '')
            self.query_length = len(self.query)
            self.query_with_gaps = aligned_sequence
            # self.counts = SequenceProfile.populate_design_dictionary(self.length, alphabet)
            # self.counts = [{letter: 0 for letter in alphabet} for _ in range(self.length)]  # list[dict]
            self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
            for record in self.alignment:
                for i, aa in enumerate(record.seq):
                    self._counts[i][gapped_numerical_translation[aa]] += 1
                    # self.counts[i][aa] += 1
            print('OLD self._counts', self._counts)

            print('self.array', self.array)
            print('self.numerical_alignment', self.numerical_alignment)
            numerical_alignment = self.numerical_alignment
            self.counts = np.zeros((self.length, len(gapped_protein_letters)))
            # invert the "typical" format to length of the alignment in axis 0, and the numerical letters in axis 1
            for residue_idx in range(self.length):
                self.counts[residue_idx, :] = np.bincount(numerical_alignment[:, residue_idx])
            print('self.counts', self.counts)

            # self.observations = find_column_observations(self.counts, **kwargs)
            if count_gaps:
                self._observations = [sum(aa_counts) for aa_counts in self._counts]  # list[dict]
                self.observations = [self.number_of_sequences for _ in range(self.length)]
            else:
                # gap_observations = [aa_counts['-'] for aa_counts in self.counts]  # list[dict]
                # gap_observations = [aa_counts[0] for aa_counts in self.counts]  # list[list]
                # self.observations = [counts - gap for counts, gap in zip(self.observations, gap_observations)]
                self._observations = [sum(aa_counts[:-1]) for aa_counts in self._counts]  # list[list]
                self.observations = self.counts[:-1, :].sum(axis=0)  # gaps are the last index
                if not np.any(self.observations):  # check if an observation is 0
                    raise ValueError(f'Can\'t have a MSA column with 0 observations. Found at ('
                                     f'{",".join(map(str, np.flatnonzero(self.observations)))}')
                    #                f'{",".join(str(idx) for idx, pos in enumerate(self.observations) if not pos)}')
            print('self.observations', self.observations)
            print('OLD self._observations', self._observations)

            if weight_alignment_by_sequence:
                # create a 1/ obs * counts = positional_weights
                #               alignment.length - 0   1   2  ...
                #      / obs 0 [[32]   count seq 0 '-' -  2   0   0  ...   [[ 64   0   0 ...]  \
                # 1 / |  obs 1  [33] * count seq 1 'A' - 10  10   0  ... =  [330 330   0 ...]   |
                #      \ obs 2  [33]   count seq 2 'C' -  8   8   1  ...    [270 270  33 ...]] /
                #   ...   ...]               ...  ... ... ...
                position_weights = 1 / (self.observations[None, :] * self.counts)
                # take_along_axis from this with the transposed numerical_alignment (na) where each successive na idx
                # is the sequence position at the na and therefore is grabbing the position_weights by that index
                # finally sum along each sequence
                # The position_weights count seq idx must be taken by a sequence index. This happens to be on NA axis 1
                # at the moment so specified with .T and take using axis=0. Keeping both as axis=0 doen't index
                # correctly. Maybe this is a case where 'F' array ordering is needed?
                sequence_weights = np.take_along_axis(position_weights, numerical_alignment.T, axis=0).sum(axis=0)
                print('sequence_weights', sequence_weights)
                sequence_weights_ = weight_sequences(self._counts, self.alignment, self._observations)
                print('OLD sequence_weights_', sequence_weights_)

            if sequence_weights is not None:  # overwrite the current counts with weighted counts
                self.sequence_weights = sequence_weights
                # Todo update this as well
                self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
                for record in self.alignment:
                    for i, aa in enumerate(record.seq):
                        self._counts[i][gapped_numerical_translation[aa]] += sequence_weights_[i]
                        # self.counts[i][aa] += sequence_weights[i]
                print('OLD sequence_weight self._counts', self._counts)

                # add each sequence weight to the indices indicated by the numerical_alignment
                self.counts = np.zeros((self.length, len(gapped_protein_letters)))
                for idx in range(self.number_of_sequences):
                    self.counts[:, numerical_alignment[idx]] += sequence_weights[idx]
                print('sequence_weight self.counts', self.counts)
            else:
                self.sequence_weights = []

            self._frequencies = [[count/observation for count in amino_acid_counts]
                                 for amino_acid_counts, observation in zip(self._counts, self._observations)]
            print('OLD self._frequencies', self._frequencies)

            self.frequencies = np.zeros(self.counts.shape)
            # self.frequencies = [self.counts[residue_idx] / self.observations[residue_idx]
            #                     for residue_idx in range(self.length)]
            for residue_idx in range(self.length):
                self.frequencies[residue_idx, :] = self.counts[residue_idx] / self.observations[residue_idx]
            print('self.frequencies', self.frequencies)
            # self.msa_to_prob_distribution()

    @classmethod
    def from_stockholm(cls, file, **kwargs):
        try:
            return cls(alignment=read_alignment(file, alignment_type='stockholm'), **kwargs)
        except FileNotFoundError:
            raise DesignError(f'The multiple sequence alignment file "{file}" doesn\'t exist')

    @classmethod
    def from_fasta(cls, file):
        try:
            return cls(alignment=read_alignment(file))
        except FileNotFoundError:
            raise DesignError(f'The multiple sequence alignemnt file "{file}" doesn\'t exist')

    @classmethod
    def from_dictionary(cls, named_sequences: dict[str, str], **kwargs):
        """Create a MultipleSequenceAlignment from a dictionary of named sequences

        Args:
            named_sequences: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
        Returns:
            The MultipleSequenceAlignment object for the provided sequences
        """
        return cls(alignment=MultipleSeqAlignment([SeqRecord(Seq(sequence), annotations={'molecule_type': 'Protein'},
                                                             id=name)
                                                   for name, sequence in named_sequences.items()]), **kwargs)

    @classmethod
    def from_seq_records(cls, seq_records: Iterable[SeqRecord], **kwargs):
        """Create a MultipleSequenceAlignment from a SeqRecord Iterable

        Args:
            seq_records: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
        """
        return cls(alignment=MultipleSeqAlignment(seq_records), **kwargs)

    # def msa_to_prob_distribution(self):
    #     """Find the Alignment probability distribution from the self.counts dictionary
    #
    #     Sets:
    #         self.frequencies (dict[mapping[int, dict[mapping[alphabet,float]]]]):
    #             {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...}
    #     """
    #     for residue, amino_acid_counts in self.counts.items():
    #         total_column_weight = self.observations[residue]
    #         if total_column_weight == 0:
    #             raise ValueError(f'{self.msa_to_prob_distribution.__name__}: Can\'t have a column with 0 observations. Position = {residue}')
    #         self.frequencies[residue] = {aa: count / total_column_weight for aa, count in amino_acid_counts.items()}

    @property
    def query_indices(self) -> np.ndarray:
        """Returns the query as a boolean array (1, length) where gaps ("-") are False"""
        try:
            return self._sequence_index[0]
        except AttributeError:
            self._sequence_index = self.array != b'-'
            # self._sequence_index = np.isin(self.array, b'-', invert=True)
            return self._sequence_index[0]

    @property
    def sequence_indices(self) -> np.ndarray:
        """Returns the alignment as a boolean array (number_of_sequences, length) where gaps ("-") are False"""
        try:
            return self._sequence_index
        except AttributeError:
            self._sequence_index = self.array != b'-'
            # self._sequence_index = np.isin(self.array, b'-', invert=True)
            return self._sequence_index

    @sequence_indices.setter
    def sequence_indices(self, sequence_indices: np.ndarray):
        self._sequence_index = sequence_indices

    @property
    def numerical_alignment(self) -> np.ndarray:
        """Return the alignment as an integer array (number_of_sequences, length) of the amino acid characters

        MultipleSequenceAlignment.gapped_numerical_translation maps "ACDEFGHIKLMNPQRSTVWY-" to the resulting index
        """
        try:
            return self._numerical_alignment
        except AttributeError:
            self._numerical_alignment = np.vectorize(gapped_numerical_translation.__getitem__)(self.array)
            # self._numerical_alignment = \
            #     np.array([[numerical_translation[aa] for aa in record] for record in self.alignment])
            return self._numerical_alignment

    @property
    def array(self) -> np.ndarray:
        """Return the alignment as a character array (number_of_sequences, length) with numpy.string_ dtype"""
        try:
            return self._array
        except AttributeError:
            self._array = np.array([list(record) for record in self.alignment], np.string_)
            return self._array


def sequence_to_numeric(sequence: Sequence) -> np.ndarray:
    """Convert a position specific profile matrix into a numeric array

    Args:
        sequence: The sequence to encode
    Returns:
        The numerically encoded sequence where each entry along axis 0 is the indexed amino acid. Indices are according
            to the 1 letter alphabetical amino acid
    """
    _array = np.array(list(sequence), np.string_)  # for single sequence
    return np.vectorize(numerical_translation.__getitem__)(_array)


def pssm_to_numeric(pssm: dict[int, dict[str, str | float | int | dict[str, int]]]) -> np.ndarray:
    """Convert a position specific profile matrix into a numeric array

    Args:
        pssm: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                  2: {}, ...}
    Returns:
        The numerically encoded pssm where each entry along axis 0 is the position, and the entries on axis 1 are the
            frequency data at every indexed amino acid. Indices are according to the 1 letter alphabetical amino acid,
            i.e array([[0.1, 0.01, 0.12, ...], ...])
    """
    _array = np.array([[residue_info[aa] for aa in protein_letters] for residue_info in pssm.values()], np.string_)
    return np.vectorize(numerical_translation.__getitem__)(_array)


class SequenceProfile:
    """Contains the sequence information for a Structure. Should always be subclassed by a Structure object.
    Currently, Entity and Pose contain the necessary .reference_sequence attribute. Any Structure object with a
    .reference_sequence attribute could be used however
    """
    disorder: dict[int, dict[str, str]]
    fragment_db: info.FragmentInfo
    log: Logger
    name: str
    number_of_residues: int
    reference_sequence: str
    # residues: list[Residue]
    sequence: str

    def __init__(self, **kwargs):
        # super().__init__()
        super().__init__(**kwargs)
        self.evolutionary_profile: dict = {}  # position specific scoring matrix
        # self.design_pssm_file = None
        self.profile: dict = {}  # design specific scoring matrix
        self._fragment_db = None
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        self.fragment_map: dict | None = None  # {}
        self.alpha: dict = {}
        self.fragment_profile: dict = {}
        # self.fragment_pssm_file = None
        # self.interface_data_file = None
        self.a3m_file: AnyStr | None = None
        self.h_fields: np.ndarray | None = None
        self.j_couplings: np.ndarray | None = None
        # self.msa: Optional[MultipleSequenceAlignment] = None
        self.msa_file: AnyStr | None = None
        self.pssm_file: AnyStr | None = None
        # self.sequence_source = None
        self.sequence_file: AnyStr | None = None

    # @classmethod
    # def from_structure(cls, structure=None):
    #     return cls(structure=structure)

    # @property
    # def profile_length(self) -> int:
    #     """Returns the number of Structured residues"""
    #     return self.number_of_residues

    @property
    def offset_index(self) -> int:
        """Return the starting index for the Entity based on pose numbering of the residues. Zero-indexed"""
        return self.residues[0].number - 1

    # @offset_index.setter
    # def offset_index(self, offset_index):
    #     self._entity_offset = offset_index

    @property
    def msa(self) -> MultipleSequenceAlignment | None:
        try:
            return self._msa
        except AttributeError:
            return

    @msa.setter
    def msa(self, msa: MultipleSequenceAlignment):
        if isinstance(msa, MultipleSequenceAlignment):
            self._msa = copy(msa)
            self.fit_msa_to_structure()
        else:
            self.log.warning(f'The passed msa isn\'t of the required type {MultipleSequenceAlignment.__name__}')

    # def disorder(self):
    #     try:
    #         return self._disorder
    #     except AttributeError:
    #         if not self.reference_sequence:
    #             self._retrieve_sequence_from_api(entity_id=self.name)
    #         self._disorder = generate_mutations(self.structure_sequence, self.reference_sequence, only_gaps=True)
    #         return self._disorder

    # def _retrieve_sequence_from_api(self, entity_id=None):  # Unused
    #     self.sequence = get_sequence_by_entity_id(entity_id)
    #     self.sequence_source = 'seqres'

    def add_profile(self, evolution: bool = True, fragments: bool = True, null: bool = False,
                    out_path: AnyStr = os.getcwd(), **kwargs):
        #           fragment_observations=None, entities=None, pdb_numbering=True,
        """Add the evolutionary and fragment profiles onto the SequenceProfile

        Args:
            evolution: Whether to add evolutionary information to the sequence profile
            fragments: Whether to add fragment information to the sequence profile
            null: Whether to use a null profile (non-functional) as the sequence profile
            out_path: Location where sequence files should be written
            # fragment_source=None (list):
            # alignment_type=None (str): Either 'mapped' or 'paired'. Indicates how entity and fragments are aligned
            # pdb_numbering=True (bool):
        """
        if null or (not evolution and not fragments):
            null, evolution, fragments = True, False, False
            # self.add_evolutionary_profile(null=null, **kwargs)

        if evolution:  # add evolutionary information to the SequenceProfile
            if not self.evolutionary_profile:
                self.add_evolutionary_profile(out_path=out_path, **kwargs)

            # Check the profile and try to generate again if it is incorrect
            first = True
            while not self.verify_evolutionary_profile():
                if first:
                    self.log.info(f'Generating a new profile for {self.name}')
                    self.add_evolutionary_profile(force=True, out_path=out_path)
                    first = False
                else:
                    raise DesignError('Profile Generation got stuck, design aborted')

        else:
            self.null_pssm()

        if fragments:  # add fragment information to the SequenceProfile
            if self.fragment_map is None:
                raise AttributeError('Fragments were specified but have not been added to the SequenceProfile! '
                                     f'Call {self.map_fragments_to_profile.__name__} with fragment information')
            elif self.fragment_db:  # fragments have already been added, connect DB info
                retrieve_fragments = [fragment['cluster'] for idx_d in self.fragment_map.values()
                                      for fragments in idx_d.values() for fragment in fragments
                                      if fragment['cluster'] not in self.fragment_db.cluster_info]
                self.fragment_db._load_cluster_info(ids=retrieve_fragments)
            else:
                raise AttributeError('Fragments were specified but there is no fragment database attached. Ensure '
                                     'fragment_db is set before requesting fragment information')

            # process fragment profile from self.fragment_map or self.fragment_query
            self.add_fragment_profile()
            self.find_alpha()

        self.calculate_design_profile(boltzmann=True, favor_fragments=fragments)

    def verify_evolutionary_profile(self) -> bool:
        """Returns True if evolutionary profile and Structure are equal"""
        if self.number_of_residues != len(self.evolutionary_profile):
            self.log.warning(f'{self.name}: Profile and {type(self).__name__} are different lengths! Profile='
                             f'{len(self.evolutionary_profile)}, Pose={self.number_of_residues}')
            return False

        # if not rerun:
        # Check sequence from Pose and self.profile to compare identity before proceeding
        incorrect_count = 0
        for residue, position_data in zip(self.residues, self.evolutionary_profile.values()):
            profile_res_type = position_data['type']
            pose_res_type = protein_letters_3to1[residue.type.title()]
            if profile_res_type != pose_res_type:
                # This may not be the worst thing in the world... If the profile was made off of an entity
                # that is not the exact structure, there should be some reality to it. I think the issue would
                # be with Rosetta loading of the Sequence Profile and not matching. I am trying to mutate the
                # offending residue type in the evolutionary profile to the Pose residue type. The frequencies
                # will reflect the actual values desired, however the surface level will be different.
                # Otherwise, generating evolutionary profiles from individual files will be required which
                # don't contain a reference sequence and therefore have their own caveats. Warning the user
                # will allow the user to understand what is happening at least
                self.log.warning(f'Profile ({self.pssm_file}) and Pose ({self.sequence_file}) sequences '
                                 f'mismatched!\n\tResidue {residue.number}: Profile={profile_res_type}, '
                                 f'Pose={pose_res_type}')
                if position_data[pose_res_type] > 0:  # The residue choice isn't horrible...
                    self.log.critical('The evolutionary profile must have been generated from a different file,'
                                      ' however the evolutionary information contained is still viable. The '
                                      'correct residue from the Pose will be substituted for the missing '
                                      'residue in the profile')
                    incorrect_count += 1
                    if incorrect_count > 2:
                        self.log.critical('This error has occurred at least 3 times and your modelling accuracy'
                                          ' will probably suffer')
                    position_data['type'] = pose_res_type
                else:
                    self.log.critical('The evolutionary profile must have been generated from a different file,'
                                      ' and the evolutionary information contained ISN\'T viable. Regenerating '
                                      'evolutionary profile from the structure sequence instead')
                    return False

        return True

    def add_evolutionary_profile(self, out_path: AnyStr = os.getcwd(), profile_source: str = PUtils.hhblits,
                                 file: AnyStr = None, force: bool = False):
        """Add the evolutionary profile to the entity. Profile is generated through a position specific search of
        homologous protein sequences (evolutionary)

        Args:
            out_path: Location where sequence files should be written
            profile_source: One of 'hhblits' or 'psiblast'
            file: Location where profile file should be loaded from
            force: Whether to force generation of a new profile
        Sets:
            self.evolutionary_profile
        """
        if profile_source not in [PUtils.hhblits, 'psiblast']:
            raise DesignError(f'{self.add_evolutionary_profile.__name__}: Profile generation only possible from '
                              f'"{PUtils.hhblits}" or "psiblast", not {profile_source}')
        if file is not None:
            self.pssm_file = file
        else:  # Check to see if the files of interest already exist
            # Extract/Format Sequence Information. SEQRES is prioritized if available
            if not self.sequence_file:  # not made/provided before add_evolutionary_profile, make a new one
                self.write_sequence_to_fasta('reference', out_path=out_path)
            elif not os.path.exists(self.sequence_file) or force:
                self.log.debug(f'{self.name} Sequence={self.reference_sequence}')
                self.write_sequence_to_fasta('reference', file_name=self.sequence_file)
                self.log.debug(f'{self.name} sequence file: {self.sequence_file}')

            # temp_file = os.path.join(out_path, f'{self.name}.hold')
            temp_file = Path(out_path, f'{self.name}.hold')
            self.pssm_file = os.path.join(out_path, f'{self.name}.hmm')
            if not os.path.exists(self.pssm_file) or force:
                if not os.path.exists(temp_file):  # No work on this pssm file has been initiated
                    # Create blocking file to prevent excess work
                    with open(temp_file, 'w') as f:
                        self.log.info(f'Fetching "{self.name}" sequence data')
                    self.log.debug(f'{self.name} Evolutionary Profile not yet created')
                    # if profile_source == PUtils.hhblits:
                    #     self.log.info(f'Generating HHM Evolutionary Profile for {self.name}')
                    #     self.hhblits(out_path=out_path)
                    # else:
                    self.log.info(f'Generating Evolutionary Profile for {self.name}')
                    # self.psiblast(out_path=out_path)
                    getattr(self, profile_source)(out_path=out_path)
                    temp_file.unlink(missing_ok=True)
                    # if os.path.exists(temp_file):
                    #     os.remove(temp_file)
                else:  # Block is in place, another process is working
                    self.log.info(f'Waiting for "{self.name}" profile generation...')
                    while not os.path.exists(self.pssm_file):
                        if int(time.time()) - int(os.path.getmtime(temp_file)) > 5400:  # > 1 hr 30 minutes have passed
                            # os.remove(temp_file)
                            temp_file.unlink(missing_ok=True)
                            raise DesignError(f'{self.add_evolutionary_profile.__name__}: Generation of the profile for'
                                              f' {self.name} took longer than the time limit. Job killed!')
                        time.sleep(20)

        getattr(self, f'parse_{profile_source}_pssm')()

    def null_pssm(self):
        """Take the contents of a pssm file, parse, and input into a sequence dictionary.

        Sets:
            self.evolutionary_profile (dict): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        self.evolutionary_profile = self.populate_design_dictionary(self.number_of_residues, alph_3_aa)
        sequence = self.sequence
        for idx, residue_number in enumerate(self.evolutionary_profile):
            self.evolutionary_profile[residue_number]['lod'] = copy(aa_counts)
            self.evolutionary_profile[residue_number]['type'] = sequence[idx]
            self.evolutionary_profile[residue_number]['info'] = 0.0
            self.evolutionary_profile[residue_number]['weight'] = 0.0

    def fit_evolutionary_profile_to_structure(self):
        """From an evolutionary profile generated according to a reference sequence, align the profile to the Structure
        sequence, removing information for residues not present in the Structure

        Sets:
            (dict) self.evolutionary_profile
        """
        # generate the disordered indices which are positions in reference that are missing in structure
        disorder = self.disorder
        # removal of these positions from .evolutionary_profile will produce a properly indexed profile
        new_residue_number = 1
        structure_evolutionary_profile = {}
        for residue_number, residue_data in self.evolutionary_profile.items():
            if residue_number not in disorder:
                structure_evolutionary_profile[new_residue_number] = residue_data
                new_residue_number += 1

        self.log.debug(f'{self.fit_evolutionary_profile_to_structure.__name__}:\n\tOld:\n'
                       f'{"".join(res["type"] for res in self.evolutionary_profile.values())}\n\tNew:\n'
                       f'{"".join(res["type"] for res in structure_evolutionary_profile.values())}')
        self.evolutionary_profile = structure_evolutionary_profile

    def fit_msa_to_structure(self):
        """From a multiple sequence alignment to the reference sequence, align the profile to the Structure sequence.
        Removes the view of all data not present in the structure

        Sets:
            (np.ndarray) self.msa.sequence_indices
        """
        # generate the disordered indices which are positions in reference that are missing in structure
        # disorder_indices = [index - 1 for index in self.disorder]
        if len(self.reference_sequence) != self.msa.query_length:
            raise ValueError(f'The {self.name} reference_sequence ({len(self.reference_sequence)}) and '
                             f'MultipleSequenceAlignment query ({self.msa.query_length}) should be the same length!')
        sequence_indices = self.msa.sequence_indices
        sequence_indices[:, [index - 1 for index in self.disorder]] = False
        self.msa.sequence_indices = sequence_indices

    # def fit_secondary_structure_profile_to_structure(self):
    #     """
    #
    #     Sets:
    #         (dict) self.secondary_structure
    #     """
    #     # self.retrieve_info_from_api()
    #     # grab the reference sequence used for translation (expression)
    #     # if not self.reference_sequence:
    #     #     self._retrieve_sequence_from_api(entity_id=self.name)
    #     # generate the disordered indices which are positions in reference that are missing in structure
    #     # disorder = generate_mutations(self.sequence, self.reference_sequence, only_gaps=True)
    #     disorder = self.disorder
    #     # removal of these positions from .evolutionary_profile will produce a properly indexed profile
    #     secondary_structure = ''
    #     for index, ss_data in enumerate(self.secondary_structure, 1):
    #         if index not in disorder:
    #             secondary_structure += ss_data
    #     self.log.debug('Different profile lengths requires %s to be performed:\nOld ss:\n\t%s\nNew ss:\n\t%s'
    #                    % (self.fit_secondary_structure_profile_to_structure.__name__,
    #                       self.secondary_structure, secondary_structure))
    #     self.secondary_structure = secondary_structure

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
        p.communicate()

    # @handle_errors(errors=(FileNotFoundError,))
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

    def hhblits(self, out_path: AnyStr = os.getcwd(), threads: int = CommandDistributer.hhblits_threads,
                return_command: bool = False, **kwargs) -> str | None:
        """Generate an position specific scoring matrix from HHblits using Hidden Markov Models

        Args:
            out_path: Disk location where generated file should be written
            threads: Number of cpu's to use for the process
            return_command: Whether to simply return the hhblits command
        Sets:
            self.pssm_file (str): Name of the file generated by psiblast
        """
        self.pssm_file = os.path.join(out_path, f'{self.name}.hmm')
        self.a3m_file = os.path.join(out_path, f'{self.name}.a3m')
        # self.msa_file = os.path.join(out_path, f'{self.name}.fasta'
        self.msa_file = os.path.join(out_path, f'{self.name}.sto')  # preferred
        # this location breaks with SymDesign norm so we should modify it Todo clean
        fasta_msa = os.path.join(os.path.dirname(out_path), 'sequences', f'{self.name}.fasta')
        # todo for higher performance set up https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
        cmd = [PUtils.hhblits_exe, '-d', PUtils.uniclustdb, '-i', self.sequence_file,
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
        if p.returncode != 0:
            # temp_file = os.path.join(out_path, f'{self.name}.hold')
            temp_file = Path(out_path, f'{self.name}.hold')
            temp_file.unlink(missing_ok=True)
            # if os.path.exists(temp_file):  # remove hold file blocking progress
            #     os.remove(temp_file)
            raise DesignError(f'Profile generation for {self.name} got stuck')  #
            # raise DesignError(f'Profile generation for {self.name} got stuck. See the error for details -> {p.stderr} '
            #                   f'output -> {p.stdout}')  #
        p = subprocess.Popen([PUtils.reformat_msa_exe_path, self.a3m_file, self.msa_file, '-num', '-uc'])
        p.communicate()
        p = subprocess.Popen([PUtils.reformat_msa_exe_path, self.a3m_file, fasta_msa, '-M', 'first', '-r'])
        p.communicate()
        # os.system('rm %s' % self.a3m_file)

    # @handle_errors(errors=(FileNotFoundError,))
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

    def add_msa(self, msa: str | MultipleSequenceAlignment = None):
        """Add a multiple sequence alignment to the profile

        Args:
            msa: The multiple sequence alignment object or file to use for collapse
        """
        if msa:
            if isinstance(msa, MultipleSequenceAlignment):
                self.msa = msa
                return
            else:
                self.msa_file = msa

        if not self.msa_file:
            # self.msa = self.api_db.alignments.retrieve_data(name=self.name)
            raise AttributeError('No .msa_file attribute is specified yet!')
        # self.msa = MultipleSequenceAlignment.from_stockholm(self.msa_file)
        try:
            self.msa = MultipleSequenceAlignment.from_stockholm(self.msa_file)
            # self.msa = MultipleSequenceAlignment.from_fasta(self.msa_file)
        except FileNotFoundError:
            try:
                self.msa = MultipleSequenceAlignment.from_fasta(f'{os.path.splitext(self.msa_file)[0]}.fasta')
                # self.msa = MultipleSequenceAlignment.from_stockholm('%s.sto' % os.path.splitext(self.msa_file)[0])
            except FileNotFoundError:
                raise FileNotFoundError(f'No multiple sequence alignment exists at {self.msa_file}')

    def collapse_profile(self, msa: str | MultipleSequenceAlignment = None) -> pd.DataFrame:
        """Make a profile out of the hydrophobic collapse index (HCI) for each sequence in a multiple sequence alignment

        Calculate HCI for each sequence (different lengths) into an array. For each msa sequence, make a gap array
        (# msa sequences x alignment length) to account for gaps from each individual sequence. Create a map between the
        gap array and the HCI array

        iter array   -   gap mask      -       Hydro Collapse Array     -     Aligned HCI     - -     Final HCI

        ------------

        iter - - - - - - 1 is gap    - - - -     compute for each     -     account for gaps   -  (drop col 3, idx 2)

        it 1 2 3 4  - - 0 | 1 | 2 - - - - - - - - - 0 | 1 | 2 - - - - - - - - 0 | 1 | 2 - - - - - - - 0 | 1 | 3 | ... N

        0 0 1 2 2  - - 1 | 1 | 0 - - - -   - - - - 0.5 0.2 0.5 - -   =   - - 0.5 0.2 0.0 -  ->   - - 0.5 0.2 0.4 ... 0.3

        1 0 0 1 2  - - 0 | 1 | 1 - - - -   - - - - 0.4 0.7 0.4 - -   =   - - 0.0 0.4 0.7 -  ->   - - 0.0 0.4 0.4 ... 0.1

        2 0 0 1 2  - - 0 | 1 | 1 - - - -   - - - - 0.3 0.6 0.3 - -   =   - - 0.0 0.3 0.6 -  ->   - - 0.0 0.3 0.4 ... 0.0

        After the addition, the hydro collapse array index that is accessed by the iterator is multiplied by the gap
        mask to return a null value is there is no value and the hydro collapse value if there is one
        therefore the element such as 3 in the Aligned HCI would be dropped from the array when the aligned sequence
        is removed of any gaps and only the iterations will be left, essentially giving the HCI for the sequence
        profile in the native context, however adjusted to the specific context of the protein/design sequence at hand

        Args:
            msa: The multiple sequence alignment to use for collapse
        Returns:
            DataFrame containing each sequences hydrophobic collapse values for the profile
        """
        if not self.msa:
            try:
                self.add_msa(msa)
            except FileNotFoundError:
                raise DesignError(f'Ensure that you have set up the .msa for this {type(self).__name__}. To do this, '
                                  f'either link to the Master Database, call {msa_generation_function}, or pass the '
                                  f'location of a multiple sequence alignment. '
                                  f'Supported formats:\n{pretty_format_table(msa_supported_types.items())}')

        # Make the output array. Use one additional length to add np.nan value at the 0 index for gaps
        evolutionary_collapse_np = np.zeros((self.msa.number_of_sequences, self.msa.length + 1))  # aligned_hci_np.copy()
        evolutionary_collapse_np[:, 0] = np.nan  # np.nan for all missing indices
        for idx, record in enumerate(self.msa.alignment):
            non_gapped_sequence = str(record.seq).replace('-', '')
            evolutionary_collapse_np[idx, 1:len(non_gapped_sequence) + 1] = \
                hydrophobic_collapse_index(non_gapped_sequence)

        iterator_np = np.cumsum(self.msa.sequence_indices, axis=1) * self.msa.sequence_indices
        aligned_hci_np = np.take_along_axis(evolutionary_collapse_np, iterator_np, axis=1)
        # select only the query sequence indices
        # sequence_hci_np = aligned_hci_np[:, self.msa.query_indices]
        return pd.DataFrame(aligned_hci_np[:, self.msa.query_indices],
                            columns=list(range(1, self.msa.query_length + 1)))  # include last residue
        # summary = pd.concat([sequence_hci_df, pd.concat([sequence_hci_df.mean(), sequence_hci_df.std()], axis=1,
        #                                                 keys=['mean', 'std']).T])

    def direct_coupling_analysis(self, msa: MultipleSequenceAlignment = None) -> np.ndarray:  # , data_dir=None):
        """Using boltzmann machine direct coupling analysis (bmDCA), score each sequence in an alignment based on the
         statistical energy compared to the learn DCA model

        Args:
            msa: A MSA object to score. By default will use self.msa attribute
        Returns:
            The energy for each residue in each sequence of the alignment based on direct coupling
                analysis parameters. Sequences exist on axis 0, residues along axis 1
            # (numpy.ndarray): The energy for each sequence in the alignment based on direct coupling analysis parameters
        """
        if not msa:
            msa = self.msa
        if not self.h_fields or not self.j_couplings:
            raise AttributeError('The required data .h_fields and .j_couplings are not availble. Add them to the Entity'
                                 ' before %s' % self.direct_coupling_analysis.__name__)
            # return np.array([])
        analysis_length = msa.query_length
        idx_range = np.arange(analysis_length)
        # h_fields = bmdca.load_fields(os.path.join(data_dir, '%s_bmDCA' % self.name, 'parameters_h_final.bin'))
        # h_fields = h_fields.T  # this isn't required when coming in Fortran order, i.e. (21, analysis_length)
        # sum the h_fields values for each sequence position in every sequence
        h_values = self.h_fields[msa.numerical_alignment, idx_range[None, :]].sum(axis=1)
        h_sum = h_values.sum(axis=1)

        # coming in as a 4 dimension (analysis_length, analysis_length, alphabet_number, alphabet_number) ndarray
        # j_couplings = bmdca.load_couplings(os.path.join(data_dir, '%s_bmDCA' % self.name, 'parameters_J_final.bin'))
        i_idx = np.repeat(idx_range, analysis_length)
        j_idx = np.tile(idx_range, analysis_length)
        i_aa = np.repeat(msa.numerical_alignment, analysis_length)
        j_aa = np.tile(msa.numerical_alignment, msa.query_length)
        j_values = np.zeros((msa.number_of_sequences, len(i_idx)))
        for idx in range(msa.number_of_sequences):
            j_values[idx] = self.j_couplings[i_idx, j_idx, i_aa, j_aa]
        # this mask is not necessary when the array comes in as a non-symmetry matrix. All i > j result in 0 values...
        # mask = np.triu(np.ones((analysis_length, analysis_length)), k=1).flatten()
        # j_sum = j_values[:, mask].sum(axis=1)
        # sum the j_values for every design (axis 0) at every residue position (axis 1)
        j_values = np.array(np.split(j_values, 3, axis=1)).sum(axis=2).T
        j_sum = j_values.sum(axis=1)
        # couplings_idx = np.stack((i_idx, j_idx, i_aa, j_aa), axis=1)
        # this stacks all arrays like so
        #  [[[ i_idx1, i_idx2, ..., i_idxN],
        #    [ j_idx1, j_idx2, ..., j_idxN],  <- this is for one sequence
        #    [ i_aa 1, i_aa 2, ..., i_aa N],
        #    [ j_aa 1, j_aa 2, ..., j_aa N]],
        #   [[NEXT SEQUENCE],
        #    [
        # this stacks all arrays the transpose, which would match the indexing style on j_couplings much better...
        # couplings_idx = np.stack((i_idx, j_idx, i_aa, j_aa), axis=2)
        # j_sum = np.zeros((self.msa.number_of_sequences, len(couplings_idx)))
        # for idx in range(self.msa.number_of_sequences):
        #     j_sum[idx] = j_couplings[couplings_idx[idx]]
        # return -h_sum - j_sum
        return -h_values - j_values

    def write_sequence_to_fasta(self, sequence: str | sequence_type_literal, file_name: AnyStr = None,
                                name: str = None, out_path: AnyStr = os.getcwd()) -> AnyStr:
        """Write a sequence to a .fasta file with fasta format and save file location as self.sequence_file.
        '.fasta' is appended if not specified in the name argument

        Args:
            sequence: The sequence to write. Can be the specified sequence or the keywords 'reference' or 'structure'
            file_name: The explicit name of the file
            name: The name of the sequence record. If not provided, the instance name will be used.
                Will be used as the default file_name base name if file_name not provided
            out_path: The location on disk to output the file. Only used if file_name not explicitly provided
        Returns:
            The name of the output file
        """
        if sequence in sequence_types:  # get the attribute from the instance
            sequence = getattr(self, f'{sequence}_sequence')
        if not name:
            name = self.name
        if not file_name:
            file_name = os.path.join(out_path, name)

        self.sequence_file = write_sequence_to_fasta(sequence=sequence, name=name, file_name=file_name)

        return self.sequence_file

    # fragments
    # [{'mapped': residue_number1, 'paired': residue_number2, 'cluster': cluster_id, 'match': match_score}]
    def add_fragment_profile(self):  # , fragment_source=None, alignment_type=None):
        """From self.fragment_map, add the fragment profile to the SequenceProfile

        Sets:
            self.fragment_profile
        """
        # v now done at the pose_level
        # self.map_fragments_to_profile(fragments=fragment_source, alignment_type=alignment_type)
        # if self.fragment_map is not None:
        self.generate_fragment_profile()
        self.simplify_fragment_profile()
        # else:  # try to separate any fragment queries to this entity
        #     if self.fragment_queries:
        #         for query_pair, fragments in self.fragment_queries.items():
        #             for query_idx, entity in enumerate(query_pair):
        #                 if entity.name == self.name:
        #                     # add to fragment map
        #                  self.map_fragments_to_profile(fragments=fragments, alignment_type=alignment_types[query_idx])
        #     else:
        #         self.log.error('No fragment information associated with the Entity %s yet! You must add to the profile '
        #                        'otherwise only evolutionary values will be used.\n%s'
        #                        % (self.name, add_fragment_profile_instructions))
        #         return

    def map_fragments_to_profile(self, fragments: list[dict[str, int | str, float]] = None,
                                 alignment_type: alignment_types_literal = None):
        """Distribute fragment information to self.fragment_map. One-indexed residue dictionary

        Args:
            fragments: The fragment list to assign to the sequence profile with format
                [{'mapped': residue_number1 (int), 'paired': residue_number2 (int), 'cluster': cluster_id (str),
                  'match': match_score (float)}]
            alignment_type: Either 'mapped' or 'paired'
        Sets:
            self.fragment_map (dict[int, list[dict[str, str | float]]]):
                {1: [{'chain': 'mapped', 'cluster': '1_2_123', 'match': 0.61}, ...], ...}
        """
        if alignment_type not in alignment_types:
            return

        if not self.fragment_map:
            self.fragment_map = self.populate_design_dictionary(self.number_of_residues,
                                                                [j for j in range(*self.fragment_db.fragment_range)],
                                                                dtype='list')
        if not fragments:
            return
        #     print('New fragment_map')
        # print(fragments)
        # print(self.name)
        # print(self.offset)
        for fragment in fragments:
            residue_number = fragment[alignment_type] - self.offset_index
            for j in range(*self.fragment_db.fragment_range):  # lower_bound, upper_bound
                self.fragment_map[residue_number + j][j].append({'chain': alignment_type,
                                                                 'cluster': fragment['cluster'],
                                                                 'match': fragment['match']})
        # should be unnecessary when fragments are generated internally
        # remove entries which don't exist on protein because of fragment_index +- residues
        not_available = [residue_number for residue_number in self.fragment_map
                         if residue_number <= 0 or residue_number > self.number_of_residues]
        for residue_number in not_available:
            self.log.info(f'In "{self.name}", residue {residue_number} is represented by a fragment but there is no '
                          f'Atom record for it. Fragment index will be deleted')
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
            self.fragment_profile[residue_number] = {}  # this may be unnecessary due to populate_design_dictionary()
            for frag_idx, fragments in fragment_indices.items():
                self.fragment_profile[residue_number][frag_idx] = {}
                for observation_idx, fragment in enumerate(fragments):
                    cluster_id = fragment['cluster']
                    freq_type = fragment['chain']
                    aa_freq = \
                        self.fragment_db.retrieve_cluster_info(cluster=cluster_id, source=freq_type, index=frag_idx)
                    # {1_1_54: {'mapped': {aa_freq}, 'paired': {aa_freq}}, ...}
                    #  mapped/paired aa_freq = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1: {}, ...}
                    #  Where 'stats'[0] is total fragments in cluster, and 'stats'[1] is weight of fragment index
                    self.fragment_profile[residue_number][frag_idx][observation_idx] = aa_freq
                    self.fragment_profile[residue_number][frag_idx][observation_idx]['match'] = fragment['match']

    def simplify_fragment_profile(self, keep_extras: bool = True):
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
            keep_extras: bool = True - If true, keep values for all design dictionary positions that are missing data
        """
        # self.log.debug(self.fragment_profile.items())
        database_bkgnd_aa_freq = self.fragment_db.aa_frequencies
        # Fragment profile is correct size for indexing all STRUCTURAL residues
        #  self.reference_sequence is not used for this. Instead, self.sequence is used in place since the use
        #  of a disorder indicator that removes any disordered residues from input evolutionary profiles is calculated
        #  on the full reference sequence. This ensures that the profile is the right length of the structure and
        #  captures disorder specific evolutionary signals that could be important in the calculation of profiles
        sequence = self.sequence
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
                        for obs, obs_freq_data in self.fragment_profile[residue][index].items():
                            total_fragment_observations += 1
                            obs_x_match_weight = obs_freq_data['stats'][1] * obs_freq_data['match']
                            # match_weight = self.fragment_profile[residue][index][obs]['match']
                            # obs_weight = self.fragment_profile[residue][index][obs]['stats'][1]
                            for aa, frequency in obs_freq_data.items():
                                if aa not in ['stats', 'match']:  # Todo remove to reduce time. zip the aas with freqs?
                                    # Multiply OBS and MATCH
                                    modification_weight = obs_x_match_weight / total_obs_x_match_weight
                                    # modification_weight = ((obs_weight + match_weight) /  # WHEN SUMMING OBS and MATCH
                                    #                        (total_obs_weight + total_match_weight))
                                    # modification_weight = (obs_weight / total_obs_weight)
                                    # Add all occurrences to summed frequencies list
                                    obs_aa_dict[aa] += frequency * modification_weight
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
            if self.evolutionary_profile:
                # Todo currently, if not an empty dictionary, add the corresponding value from evolution because the
                #  calculation of packer pallette is subtractive so the use of an overlapping evolution and
                #  null fragment would result in nothing allowed to design...
                for residue in no_design:
                    self.fragment_profile[residue] = self.evolutionary_profile.get(residue)  # TODO, aa_weighted_counts)
            else:  # Todo add a blank enty. This needs direction if not evolutionary profile
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
        if not self.fragment_db:
            raise AttributeError(f'{self.find_alpha.__name__}: No fragment database connected! Cannot calculate optimal'
                                 f' fragment contribution without one')
        if alpha < 0 or 1 < alpha:
            raise ValueError(f'{self.find_alpha.__name__}: Alpha parameter must be bounded between 0 and 1')
        alignment_type_to_idx = {'mapped': 0, 'paired': 1}  # could move to class, but not used elsewhere
        match_score_average = 0.5  # when fragment pair rmsd equal to the mean cluster rmsd
        bounded_floor = 0.2
        fragment_stats = self.fragment_db.statistics
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
            contribution_total = sum(fragment_stats[self.fragment_db.get_cluster_id(obs['cluster'], index=2)][0]
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
        if self.alpha:
            self.log.info('At Entity %s, combined evolutionary and fragment profiles into Design Profile with:\n\t%s'
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
            database_bkgnd_aa_freq = self.fragment_db.aa_frequencies

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
        raise NotImplementedError('This function needs work')
        # Fetch IJK Cluster Dictionaries and Setup Interface Residues for Residue Number Conversion. MUST BE PRE-RENUMBER

        # frag_cluster_residue_d = PoseDirectory.gather_pose_metrics(init=True)  Call this function with it
        # ^ Format: {'1_2_24': [(78, 87, ...), ...], ...}
        # Todo Can also re-score the interface upon Pose loading and return this information
        # template_pdb = PoseDirectory.source NOW self.pdb

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
        # interface_cb_indices = interface.cb_indices

        interface_residue_edges = {}
        for idx, residue_contacts in enumerate(interface_tree):
            if interface_tree[idx].tolist() != list():
                residue = interface.all_atoms[interface_cb_indices[idx]].residue_number
                contacts = {interface.all_atoms[interface_cb_indices[contact_idx]].residue_number
                            for contact_idx in interface_tree[idx]}
                interface_residue_edges[residue] = contacts - {residue}
        # ^ {78: [14, 67, 87, 109], ...}  green

        # solve for consensus residues using the residue graph
        self.map_fragments_to_profile(fragments=fragment_source, alignment_type=alignment_type)
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
    # def generate_mutations(query, reference, offset=True, blanks=False, termini=False, reference_gaps=False,
    #                        only_gaps=False):
    #     """Create mutation data in a typical A5K format. One-indexed dictionary keys, mutation data accessed by 'from'
    #     and 'to' keywords. By default all gaped sequences are excluded from returned mutations
    #
    #     For PDB file comparison, query should be crystal sequence (ATOM), reference should be expression sequence
    #     (SEQRES). only_gaps=True will return only the gaped area while blanks=True will return all differences between
    #     the alignment sequences. termini=True returns missing alignments at the termini
    #
    #     Args:
    #         query (str): Mutant sequence. Will be in the 'to' key
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
    #     if offset:
    #         alignment = generate_alignment(query, reference)
    #         align_seq_1 = alignment[0]
    #         align_seq_2 = alignment[1]
    #     else:
    #         align_seq_1 = query
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
    dtype_literals = Literal['list', 'set', 'tuple', 'float', 'int']

    @staticmethod
    def populate_design_dictionary(n: int, alphabet: Sequence, zero_index: bool = False, dtype: dtype_literals = 'int')\
            -> dict[int, dict[int | str, Any]]:
        """Return a dictionary with n elements, each integer key containing another dictionary with the items in
        alphabet as keys. By default, one-indexed, and data inside the alphabet dictionary is a dictionary.
        dtype can be any viable type [list, set, tuple, int, etc.]. If dtype is int or float, 0 will be initial value

        Args:
            n: number of entries in the dictionary
            alphabet: alphabet of interest
            zero_index: If True, return the dictionary with zero indexing
            dtype: The type of object present in the interior dictionary
         Returns:
             N length, one indexed dictionary with entry number keys
                ex: {1: {alphabet[0]: dtype, alphabet[1]: dtype, ...}, 2: {}, ...}
         """
        offset = 0 if zero_index else index_offset

        # Todo add
        #  match dtype:
        #       case 'int':
        #       ...
        if dtype == 'int':
            dtype = int
        elif dtype == 'dict':
            dtype = dict
        elif dtype == 'list':
            dtype = list
        elif dtype == 'set':
            dtype = set
        elif dtype == 'float':
            dtype = float

        return {residue: {character: dtype() for character in alphabet} for residue in range(offset, n + offset)}

    @staticmethod
    def get_lod(aa_freqs: dict[info.protein_letter_literal, float],
                background: dict[info.protein_letter_literal, float],
                round_lod: bool = True) -> dict[str, int]:
        """Get the log of the odds that an amino acid is in a frequency distribution compared to a background frequency

        Args:
            aa_freqs: {'A': 0.11, 'C': 0.01, 'D': 0.034, ...}
            background: {'A': 0.10, 'C': 0.02, 'D': 0.04, ...}
            round_lod: Whether to round the lod values to an integer
        Returns:
             The log of odds for each amino acid type {'A': 2, 'C': -9, 'D': -1, ...}
        """
        lods = {}
        for aa, freq in aa_freqs.items():
            try:
                lods[aa] = float((2. * log2(freq / background[aa])))  # + 0.0
            except ValueError:  # math domain error
                lods[aa] = -9
            except KeyError:
                if aa in protein_letters:
                    raise KeyError(f'{aa} was not in the background frequencies: {", ".join(background)}')
                else:  # we shouldn't worry about a missing value if it's not an amino acid
                    continue
            except ZeroDivisionError:  # background is 0. We may need a pseudocount...
                raise ZeroDivisionError(f'{aa} has a background frequency of 0. Consider adding a pseudocount')
            # if lods[aa] < -9:
            #     lods[aa] = -9
            # elif round_lod:
            #     lods[aa] = round(lods[aa])

        if round_lod:
            return {aa: (round(value) if value >= -9 else -9) for aa, value in lods.items()}
        else:  # ensure that -9 is the lowest value (formatting issues if 2 digits)
            return {aa: (value if value >= -9 else -9) for aa, value in lods.items()}

    @staticmethod
    def write_pssm_file(pssm: dict[protein_letter_plus_literals, float | str | tuple | dict], name: str,
                        out_path: AnyStr = os.getcwd()) -> str | None:
        """Create a PSI-BLAST format PSSM file from a PSSM dictionary. Assumes residue numbering is correct!

        Args:
            pssm: A dictionary which has the keys: 'A', 'C', ... (all aa's), 'lod', 'type', 'info', 'weight'
            name: The name of the file including the extension
            out_path: A specific location to write the file to
        Returns:
            Disk location of newly created .pssm file
        """
        if not pssm:
            return

        # find out if the pssm has values expressed as frequencies (percentages) or as counts and modify accordingly
        # lod_freq, counts_freq = False, False
        # first_key = next(iter(pssm.keys()))
        if isinstance(list(pssm.values())[0]['lod']['A'], float):
            separation1 = " " * 4
        else:
            separation1 = " " * 3
            # lod_freq = True
        # if type(pssm[first_key]['A']) == float:
        #     counts_freq = True

        header = f'\n\n{" " * 12}{separation1.join(alph_3_aa)}{separation1}{(" " * 3).join(alph_3_aa)}\n'
        # footer = ''
        out_file = os.path.join(out_path, name)
        with open(out_file, 'w') as f:
            f.write(header)
            for residue_number, profile in pssm.items():
                # aa_type = profile['type']
                if isinstance(profile['lod']['A'], float):  # lod_freq:  # relevant for favor_fragment
                    lod_string = '%s ' % ' '.join('{:>4.2f}'.format(profile['lod'][aa]) for aa in alph_3_aa)
                else:
                    lod_string = '%s ' % ' '.join('{:>3d}'.format(profile['lod'][aa]) for aa in alph_3_aa)
                if isinstance(profile['A'], float):  # counts_freq: # relevant for freq calculations
                    counts_string = '%s ' % ' '.join('{:>3.0f}'.format(floor(profile[aa] * 100)) for aa in alph_3_aa)
                else:
                    counts_string = '%s ' % ' '.join('{:>3d}'.format(profile[aa]) for aa in alph_3_aa)
                # info = profile.get('info', 0.)
                # weight = profile.get('weight', 0.)
                # line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}\n'.format(residue_number, aa_type, lod_string,
                #                                                                counts_string, round(info, 4),
                #                                                                round(weight, 4))
                f.write('{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}\n'
                        .format(residue_number, profile['type'], lod_string, counts_string,
                                round(profile.get('info', 0.), 4), round(profile.get('weight', 0.), 4)))
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


def get_equivalent_indices(sequence1: Sequence, sequence2: Sequence) -> tuple[list[int], list[int]]:
    """From two sequences, find the indices where both sequences are equal

    Args:
        sequence1: The first sequence to compare
        sequence2: The second sequence to compare
    Returns:
        The pair of sequence indices were the sequences align.
            Ex: sequence1 = 'ABCDEF', sequence2 = 'ABDEF', returns [0, 1, 3, 4, 5], [0, 1, 2, 3, 4]
    """
    mutations = generate_mutations(sequence1, sequence2, blanks=True, return_all=True)
    # get only those indices where there is an aligned aa on the opposite chain
    sequence1_indices, sequence2_indices = [], []
    to_idx, from_idx = 0, 0
    # sequence1 'from' is fixed, sequence2 'to' is moving
    for mutation in mutations.values():
        if mutation['from'] == '-':  # increment to_idx/fixed_idx
            to_idx += 1
        elif mutation['to'] == '-':  # increment from_idx/moving_idx
            from_idx += 1
        else:
            sequence1_indices.append(to_idx)
            sequence2_indices.append(from_idx)
            to_idx += 1
            from_idx += 1

    return sequence1_indices, sequence2_indices


# def residue_interaction_graph(pdb, distance=8, gly_ca=True):
#     """Create a atom tree using CB atoms from two PDB's
#
#     Args:
#         pdb (PDB): First PDB to query against
#     Keyword Args:
#         distance=8 (int): The distance to query in Angstroms
#         gly_ca=True (bool): Whether glycine CA should be included in the tree
#     Returns:
#         query (list()): sklearn query object of pdb2 coordinates within dist of pdb1 coordinates
#     """
#     # Get CB Atom Coordinates including CA coordinates for Gly residues
#     coords = np.array(pdb.extract_cb_coords(InclGlyCA=gly_ca))
#
#     # Construct CB Tree for PDB1
#     pdb1_tree = BallTree(coords)
#
#     # Query CB Tree for all PDB2 Atoms within distance of PDB1 CB Atoms
#     query = pdb1_tree.query_radius(coords, distance)
#
#     return query


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


# def get_db_statistics(database: AnyStr) -> dict:
#     """Retrieve summary statistics for a specific fragment database
#
#     Args:
#         database: Disk location of a fragment database
#     Returns:
#         {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
#             ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
#     """
#     for file in os.listdir(database):
#         if file.endswith('statistics.pkl'):
#             return unpickle(os.path.join(database, file))
#
#     return {}
#
#
# def get_db_aa_frequencies(database: AnyStr) -> dict[protein_letters, float]:
#     """Retrieve database specific interface background AA frequencies
#
#     Args:
#         database: Location of database on disk
#     Returns:
#         {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
#     """
#     return get_db_statistics(database).get('frequencies', {})


def get_cluster_dicts(db=PUtils.biological_interfaces, id_list=None):  # TODO Rename
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
        directory_list = get_base_root_paths_recursively(info_db)
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


# def parameterize_frag_length(length):
#     """Generate fragment length range parameters for use in fragment functions"""
#     _range = floor(length / 2)
#     if length % 2 == 1:
#         return 0 - _range, 0 + _range + index_offset
#     else:
#         logger.critical('%d is an even integer which is not symmetric about a single residue. '
#                         'Ensure this is what you want and modify %s' % (length, parameterize_frag_length.__name__))
#         raise DesignError('Function not supported: Even fragment length \'%d\'' % length)


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
                # resi_object = PDB.Residue(pdb.getResidueAtoms(pdb.chain_ids[i], residue)).ca
                # assert resi_object, DesignError('Residue \'%s\' missing from PDB \'%s\'' % (residue, pdb.file_path))
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
# def hhblits(query, cores=CommandDistributer.hhblits_threads, outpath=os.getcwd()):
#     """Generate an position specific scoring matrix from HHblits using Hidden Markov Models
#
#     Args:
#         query (str): Basename of the sequence to use as a query, intended for use as pdb
#         cores (int): Number of cpu's to use for the process
#     Keyword Args:
#         outpath=None (str): Disk location where generated file should be written
#     Returns:
#         outfile_name (str): Name of the file generated by hhblits
#         p (subprocess): Process object for monitoring progress of hhblits command
#     """
#
#     outfile_name = os.path.join(outpath, os.path.splitext(os.path.basename(query))[0] + '.hmm')
#
#     cmd = [PUtils.hhblits, '-d', PUtils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(cores)]
#     logger.info('%s Profile Command: %s' % (query, subprocess.list2cmdline(cmd)))
#     p = subprocess.Popen(cmd)
#
#     return outfile_name, p


# @handle_errors(errors=(FileNotFoundError,))
def parse_pssm(file: AnyStr, **kwargs) -> dict[int, dict[str, str | float | int | dict[str, int]]]:
    """Take the contents of a pssm file, parse, and input into a pose profile dictionary.

    Resulting dictionary is indexed according to the values in the pssm file

    Args:
        file: The location of the file on disk
    Returns:
        Dictionary containing residue indexed profile information
            i.e. {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                  2: {}, ...}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    pose_dict = {}
    for line in lines:
        line_data = line.strip().split()
        if len(line_data) == 44:
            residue_number = int(line_data[0])
            pose_dict[residue_number] = \
                dict(zip(alph_3_aa, [x/100. for x in map(int, line_data[22:len(alph_3_aa)+22])]))
            # pose_dict[residue_number] = copy(aa_counts)
            # for i, aa in enumerate(alph_3_aa, 22):
            #     # Get normalized counts for pose_dict
            #     pose_dict[residue_number][aa] = int(line_data[i]) / 100.

            # for i, aa in enumerate(alph_3_aa, 2):
            #     pose_dict[residue_number]['lod'][aa] = line_data[i]
            pose_dict[residue_number]['lod'] = dict(zip(alph_3_aa, line_data[2:len(alph_3_aa)+2]))
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


# @handle_errors(errors=(FileNotFoundError,))
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


# def combine_ssm(pssm, issm, alpha, db=PUtils.biological_interfaces, favor_fragments=True, boltzmann=False, a=0.5):
#     """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap
#
#     All input must be zero indexed
#     Args:
#         pssm (dict): HHblits - {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W',
#             'info': 0.00, 'weight': 0.00}, {...}}
#               PSIBLAST -  {0: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...}, 'type': 'W', 'info': 3.20,
#                           'weight': 0.73}, {...}} CURRENTLY IMPOSSIBLE, NEED TO CHANGE THE LOD SCORE IN PARSING
#         issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
#         alpha (dict): {48: 0.5, 50: 0.321, ...}
#     Keyword Args:
#         db: Disk location of fragment database
#         favor_fragments=True (bool): Whether to favor fragment profile in the lod score of the resulting profile
#         boltzmann=True (bool): Whether to weight the fragment profile by the Boltzmann probability. If false, residues
#             are weighted by a local maximum over the residue scaled to a maximum provided in the standard Rosetta per
#             residue reference weight.
#         a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
#     Returns:
#         pssm (dict): {0: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
#             'weight': 0.00}, ...}} - combined PSSM dictionary
#     """
#
#     # Combine fragment and evolutionary probability profile according to alpha parameter
#     for entry in alpha:
#         for aa in protein_letters:
#             pssm[entry][aa] = (alpha[entry] * issm[entry][aa]) + ((1 - alpha[entry]) * pssm[entry][aa])
#         logger.info('Residue %d Combined evolutionary and fragment profile: %.0f%% fragment'
#                     % (entry + index_offset, alpha[entry] * 100))
#
#     if favor_fragments:
#         # Modify final lod scores to fragment profile lods. Otherwise use evolutionary profile lod scores
#         # Used to weight fragments higher in design
#         boltzman_energy = 1
#         favor_seqprofile_score_modifier = 0.2 * CommandDistributer.reference_average_residue_weight
#         db = PUtils.frag_directory[db]
#         stat_dict_bkg = get_db_aa_frequencies(db)
#         null_residue = get_lod(stat_dict_bkg, stat_dict_bkg)
#         null_residue = {aa: float(null_residue[aa]) for aa in null_residue}
#
#         for entry in pssm:
#             pssm[entry]['lod'] = null_residue
#         for entry in issm:
#             pssm[entry]['lod'] = get_lod(issm[entry], stat_dict_bkg, round_lod=False)
#             partition, max_lod = 0, 0.0
#             for aa in pssm[entry]['lod']:
#                 # for use with a boltzman probability weighting, Z = sum(exp(score / kT))
#                 if boltzmann:
#                     pssm[entry]['lod'][aa] = exp(pssm[entry]['lod'][aa] / boltzman_energy)
#                     partition += pssm[entry]['lod'][aa]
#                 # remove any lod penalty
#                 elif pssm[entry]['lod'][aa] < 0:
#                     pssm[entry]['lod'][aa] = 0
#                 # find the maximum/residue (local) lod score
#                 if pssm[entry]['lod'][aa] > max_lod:
#                     max_lod = pssm[entry]['lod'][aa]
#             modified_entry_alpha = (alpha[entry] / a) * favor_seqprofile_score_modifier
#             if boltzmann:
#                 modifier = partition
#                 modified_entry_alpha /= (max_lod / partition)
#             else:
#                 modifier = max_lod
#             for aa in pssm[entry]['lod']:
#                 pssm[entry]['lod'][aa] /= modifier
#                 pssm[entry]['lod'][aa] *= modified_entry_alpha
#             logger.info('Residue %d Fragment lod ratio generated with alpha=%f'
#                         % (entry + index_offset, alpha[entry] / a))
#
#     return pssm
#
#
# def find_alpha(issm, cluster_map, db=PUtils.biological_interfaces, a=0.5):
#     """Find fragment contribution to design with cap at alpha
#
#     Args:
#         issm (dict): {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'stats': [4, 0.274]}, 50: {...}, ...}
#         cluster_map (dict): {48: {'chain': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
#     Keyword Args:
#         db: Disk location of fragment database
#         a=0.5 (float): The maximum alpha value to use, should be bounded between 0 and 1
#     Returns:
#         alpha (dict): {48: 0.5, 50: 0.321, ...}
#     """
#     db = PUtils.frag_directory[db]
#     stat_dict = get_db_statistics(db)
#     alpha = {}
#     for entry in issm:  # cluster_map
#         if cluster_map[entry]['chain'] == 'mapped':
#             i = 0
#         else:
#             i = 1
#
#         contribution_total, count = 0.0, 1
#         # count = len([1 for obs in cluster_map[entry][index] for index in cluster_map[entry]]) or 1
#         for count, residue_cluster_pair in enumerate(cluster_map[entry]['cluster'], 1):
#             cluster_id = return_cluster_id_string(residue_cluster_pair[1], index_number=2)  # get first two indices
#             contribution_total += stat_dict[cluster_id][0][i]  # get the average contribution of each fragment type
#         stats_average = contribution_total / count
#         entry_ave_frag_weight = issm[entry]['stats'][1] / count  # total weight for issm entry / number of fragments
#         if entry_ave_frag_weight < stats_average:  # if design frag weight is less than db cluster average weight
#             # modify alpha proportionally to cluster average weight
#             alpha[entry] = a * (entry_ave_frag_weight / stats_average)
#         else:
#             alpha[entry] = a
#
#     return alpha


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


# def position_specific_jsd(msa: dict[int, dict[str, float]], background: dict[int, dict[str, float]]) -> \
#         dict[int, float]:
#     """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency
#
#     Both msa and background must be the same index
#     Args:
#         msa: {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
#         background: {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
#             Containing residue index with inner dictionary of single amino acid types
#     Returns:
#         divergence_dict: {15: 0.732, 16: 0.552, ...}
#     """
#     return {idx: distribution_divergence(freq, background[idx]) for idx, freq in msa.items() if idx in background}
#
#
# def distribution_divergence(frequencies: dict[str, float], bgd_frequencies: dict[str, float], lambda_: float = 0.5) -> \
#         float:
#     """Calculate residue specific Jensen-Shannon Divergence value
#
#     Args:
#         frequencies: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
#         bgd_frequencies: {'A': 0, 'R': 0, ...}
#         lambda_: Value bounded between 0 and 1 to calculate the contribution from the observation versus the background
#     Returns:
#         Bounded between 0 and 1. 1 is more divergent from background frequencies
#     """
#     sum_prob1, sum_prob2 = 0, 0
#     for item, frequency in frequencies.items():
#         bgd_frequency = bgd_frequencies.get(item)
#         try:
#             r = (lambda_ * frequency) + ((1 - lambda_) * bgd_frequency)
#         except TypeError:  # bgd_frequency is None, therefore the frequencies can't be compared. Should error be raised?
#             continue
#         try:
#             with warnings.catch_warnings() as w:
#                 # Cause all warnings to always be ignored
#                 warnings.simplefilter('ignore')
#                 try:
#                     prob2 = (bgd_frequency * log(bgd_frequency / r, 2))
#                     sum_prob2 += prob2
#                 except (ValueError, RuntimeWarning):  # math DomainError doesn't raise, instead RunTimeWarn
#                     pass  # continue
#                 try:
#                     prob1 = (frequency * log(frequency / r, 2))
#                     sum_prob1 += prob1
#                 except (ValueError, RuntimeWarning):  # math domain error
#                     continue
#         except ZeroDivisionError:  # r = 0
#             continue
#
#     return lambda_ * sum_prob1 + (1 - lambda_) * sum_prob2


def jensen_shannon_divergence(sequence_frequencies: np.ndarray, background_aa_freq: np.ndarray, **kwargs) -> np.ndarray:
    """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict

    Args:
        sequence_frequencies: [[0.05, 0.001, 0.1, ...], ...]
        background_aa_freq: [0.11, 0.03, 0.53, ...]
    Keyword Args:
        lambda_: float = 0.5 - Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        The divergence per residue bounded between 0 and 1. 1 is more divergent from background, i.e. [0.732, ...]
    """
    return [distribution_divergence(aa_freq, background_aa_freq, **kwargs) for aa_freq in sequence_frequencies]


def position_specific_jsd(msa: np.ndarray, background: np.ndarray, **kwargs) -> np.ndarray:
    """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency

    Both msa and background must be the same index

    Args:
        msa: {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        background: {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
            Containing residue index with inner dictionary of single amino acid types
    Keyword Args:
        lambda_: float = 0.5 - Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        The divergence values per position, i.e [0.732, 0.552, ...]
    """
    return [distribution_divergence(freq, bgd_freq, **kwargs) for freq, bgd_freq in zip(msa, background)]


def distribution_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray, lambda_: float = 0.5) -> float:
    """Calculate Jensen-Shannon Divergence value from observed and background frequencies

    Args:
        frequencies: [0.05, 0.001, 0.1, ...]
        bgd_frequencies: [0, 0, ...]
        lambda_: Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    r = (lambda_ * frequencies) + ((1 - lambda_) * bgd_frequencies)
    sum_prob1 = (frequencies * np.log2(frequencies / r)).sum()
    sum_prob2 = (bgd_frequencies * np.log2(bgd_frequencies / r)).sum()
    return (lambda_ * sum_prob1) + ((1 - lambda_) * sum_prob2)


# def distribution_divergence(frequencies: Sequence[float], bgd_frequencies: Sequence[float], lambda_: float = 0.5) -> \
#         float:
#     """Calculate Jensen-Shannon Divergence value from observed and background frequencies
#
#     Args:
#         frequencies: [0.05, 0.001, 0.1, ...]
#         bgd_frequencies: [0, 0, ...]
#         lambda_: Bounded between 0 and 1 indicates weight of the observation versus the background
#     Returns:
#         Bounded between 0 and 1. 1 is more divergent from background frequencies
#     """
#     sum_prob1, sum_prob2 = 0, 0
#     for frequency, bgd_frequency in zip(frequencies, bgd_frequencies):
#         # bgd_frequency = bgd_frequencies.get(item)
#         try:
#             r = (lambda_ * frequency) + ((1 - lambda_) * bgd_frequency)
#         except TypeError:  # bgd_frequency is None, therefore the frequencies can't be compared. Should error be raised?
#             continue
#         try:
#             with warnings.catch_warnings() as w:
#                 # Cause all warnings to always be ignored
#                 warnings.simplefilter('ignore')
#                 try:
#                     prob2 = bgd_frequency * log(bgd_frequency / r, 2)
#                 except (ValueError, RuntimeWarning):  # math DomainError doesn't raise, instead RunTimeWarn
#                     prob2 = 0
#                 sum_prob2 += prob2
#                 try:
#                     prob1 = frequency * log(frequency / r, 2)
#                 except (ValueError, RuntimeWarning):  # math domain error
#                     continue
#                 sum_prob1 += prob1
#         except ZeroDivisionError:  # r = 0
#             continue
#
#     return lambda_ * sum_prob1 + (1 - lambda_) * sum_prob2


# def msa_from_dictionary(named_sequences: dict[str, str]) -> MultipleSequenceAlignment:
#     """Create a MultipleSequenceAlignment from a dictionary of named sequences
#
#     Args:
#         named_sequences: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
#     Returns:
#         The MultipleSequenceAlignment object for the provided sequences
#     """
#     return MultipleSequenceAlignment(MultipleSeqAlignment([SeqRecord(Seq(sequence),
#                                                                      annotations={'molecule_type': 'Protein'}, id=name)
#                                                            for name, sequence in named_sequences.items()]))


def msa_from_dictionary(named_sequences: dict[str, str]) -> MultipleSeqAlignment:
    """Create a MultipleSequenceAlignment from a dictionary of named sequences

    Args:
        named_sequences: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        The MultipleSequenceAlignment object for the provided sequences
    """
    return MultipleSeqAlignment([SeqRecord(Seq(sequence), annotations={'molecule_type': 'Protein'}, id=name)
                                 for name, sequence in named_sequences.items()])


def msa_from_seq_records(seq_records: Iterable[SeqRecord]) -> MultipleSeqAlignment:
    """Create a BioPython Multiple Sequence Alignment from a SeqRecord Iterable

    Args:
        seq_records: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        [SeqRecord(Seq('MNTEELQVAAFEI...', ...), id="Alpha"),
         SeqRecord(Seq('MNTEEL-VAAFEI...', ...), id="Beta"), ...]
    """
    return MultipleSeqAlignment(seq_records)


def make_mutations(sequence: Sequence, mutations: dict[int, dict[str, str]], find_orf: bool = True) -> str:
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Args:
        sequence: 'Wild-type' sequence to mutate
        mutations: {mutation_index: {'from': AA, 'to': AA}, ...}
        find_orf: Whether to find the correct ORF for the mutations and the seq
    Returns:
        seq: The mutated sequence
    """
    # Seq can be either list or string
    if find_orf:
        offset = -find_orf_offset(sequence, mutations)
        logger.info(f'Found ORF. Offset = {-offset}')
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
        logger.warning(f'{make_mutations.__name__} index errors: {", ".join(map(str, index_errors))}')

    return seq


def find_orf_offset(sequence: Sequence,  mutations: dict[int, dict[str, str]]) -> int:
    """Using a sequence and mutation data, find the open reading frame that matches mutations closest

    Args:
        sequence: Sequence to search for ORF in 1 letter format
        mutations: {mutation_index: {'from': AA, 'to': AA}, ...} One-indexed sequence dictionary
    Returns:
        The zero-indexed integer to offset the provided sequence to best match the provided mutations
    """
    unsolvable = False
    orf_start_idx = 0
    orf_offsets = {idx: 0 for idx, aa in enumerate(sequence) if aa == 'M'}
    methionine_positions = list(orf_offsets.keys())
    while True:
        if not orf_offsets:  # MET is missing/not the ORF start
            orf_offsets = {start_idx: 0 for start_idx in range(0, 50)}

        # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
        for test_orf_index in orf_offsets:
            for mutation_index, mutation in mutations.items():
                try:
                    if sequence[test_orf_index + mutation_index - index_offset] == mutation['from']:
                        orf_offsets[test_orf_index] += 1
                except IndexError:  # we have reached the end of the sequence
                    break

        max_count = max(list(orf_offsets.values()))
        # Check if likely ORF has been identified (count < number mutations/2). If not, MET is missing/not the ORF start
        if max_count < len(mutations) / 2:
            if unsolvable:
                return orf_start_idx
            orf_offsets = {}
            unsolvable = True  # if we reach this spot again, the problem is deemed unsolvable
        else:  # find the index of the max_count
            for idx, count in orf_offsets.items():
                if max_count == count:  # orf_offsets[offset]:
                    orf_start_idx = idx  # select the first occurrence of the max count
                    break

            # for cases where the orf doesn't begin on Met, try to find a prior Met. Otherwise, selects the id'd Met
            closest_met = None
            for met_index in methionine_positions:
                if met_index <= orf_start_idx:
                    closest_met = met_index
                else:  # we have passed the identified orf_start_idx
                    if closest_met is not None:
                        orf_start_idx = closest_met  # + index_offset # change to one-index
                    break
            break

    return orf_start_idx


Alignment = namedtuple('Alignment', 'seqA, seqB, score, start, end')


def generate_alignment(seq1: Sequence, seq2: Sequence, matrix: str = 'BLOSUM62', local: bool = False,
                       top_aligment: bool = True) -> Alignment | list[Alignment]:
    """Use Biopython's pairwise2 to generate a sequence alignment

    Args:
        seq1: The first sequence to align
        seq2: The second sequence to align
        matrix: The matrix used to compare character similarities
        local: Whether to run a local alignment. Only use for generally similar sequences!
        top_aligment: Only include the highest scoring alignment
    Returns:
        The resulting alignment
    """
    if local:
        _type = 'local'
    else:
        _type = 'global'
    _matrix = subs_matrices.get(matrix, substitution_matrices.load(matrix))
    gap_penalty = -10
    gap_ext_penalty = -1
    logger.debug(f'Generating sequence alignment between:\n{seq1}\n\tAND:\n{seq2}')
    # Create sequence alignment
    align = getattr(pairwise2.align, f'{_type}ds')(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty,
                                                   one_alignment_only=top_aligment)
    return align[0] if top_aligment else align


def generate_mutations(reference: Sequence, query: Sequence, offset: bool = True, blanks: bool = False,
                       remove_termini: bool = True, remove_query_gaps: bool = True, only_gaps: bool = False,
                       zero_index: bool = False, return_all: bool = False) -> dict[int, dict[str, str]]:
    """Create mutation data in a typical A5K format. One-indexed dictionary keys with the index matching the reference
     sequence index. Sequence mutations accessed by "from" and "to" keys. By default, all gaped sequences are excluded
     from returned mutation dictionary

    For PDB comparison, reference should be expression sequence (SEQRES), query should be atomic sequence (ATOM)

    Args:
        reference: Reference sequence to align mutations against. Character values are returned in the "from" key
        query: Query sequence. Character values are returned in the "to" key
        offset: Whether sequences are different lengths. Will create an alignment of the two sequences
        blanks: Include all gaped indices, i.e. outside the reference sequence or missing characters in the sequence
        remove_termini: Remove indices that are outside the reference sequence boundaries
        remove_query_gaps: Remove indices where there are gaps present in the query sequence
        only_gaps: Only include reference indices that are missing query residues. All "to" values will be a gap "-"
        zero_index: Whether to return the indices zero-indexed (like python) or one-indexed
        return_all: Whether to return all the indices and there corresponding mutational data
    Returns:
        Mutation index to mutations in the format of {1: {'from': 'A', 'to': 'K'}, ...}
    """
    if offset:
        align_seq_1, align_seq_2, *_ = generate_alignment(reference, query)
    else:
        align_seq_1, align_seq_2 = reference, query

    if zero_index:
        idx_offset = 0
    else:
        idx_offset = index_offset

    # Extract differences from the alignment
    starting_idx_of_seq1 = align_seq_1.find(reference[0])  # get the first matching index of the reference sequence
    ending_index_of_seq1 = starting_idx_of_seq1 + align_seq_1.rfind(reference[-1])  # find last index of reference
    if return_all:
        mutations = \
            {idx: {'from': seq1, 'to': seq2}  # always ensure sequence1/reference starts at idx 1    v
             for idx, (seq1, seq2) in enumerate(zip(align_seq_1, align_seq_2), -starting_idx_of_seq1 + idx_offset)}
    else:
        mutations = \
            {idx: {'from': seq1, 'to': seq2}  # always ensure sequence1/reference starts at idx 1    v
             for idx, (seq1, seq2) in enumerate(zip(align_seq_1, align_seq_2), -starting_idx_of_seq1 + idx_offset)
             if seq1 != seq2}

    remove_mutation_list = []
    if only_gaps:  # remove the actual mutations, keep internal and external gap indices and the reference sequence
        blanks = True
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if 0 < entry <= ending_index_of_seq1 and mutation['to'] != '-'])
    if blanks:  # leave all types of blanks, otherwise check for each requested type
        remove_termini, remove_query_gaps = False, False

    if remove_termini:  # remove indices outside of sequence 1
        remove_mutation_list.extend([entry for entry in mutations if entry < 0 or ending_index_of_seq1 < entry])

    if remove_query_gaps:  # remove indices where sequence 2 is gaped
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if 0 < entry <= ending_index_of_seq1 and mutation['to'] == '-'])
    for entry in remove_mutation_list:
        mutations.pop(entry, None)

    return mutations


def format_mutations(mutations):
    return [f'{mutation["from"]}{index}{mutation["to"]}' for index, mutation in mutations.items()]


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


def clean_gapped_columns(alignment_dict, correct_index):  # UNUSED
    """Cleans an alignment dictionary by revising key list with correctly indexed positions. 0 indexed"""
    return {i: alignment_dict[index] for i, index in enumerate(correct_index)}


def weight_sequences(alignment_counts: Sequence[Sequence[int]], bio_alignment: MultipleSeqAlignment,
                     column_counts: Sequence[int]) -> list[float]:  # UNUSED
    """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment

    Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count)) as was described by Heinkoff and Heinkoff, 1994
    Args:
        alignment_counts: The counts of each AA in each column [{'A': 31, 'C': 0, ...}, 2: {}, ...]
        bio_alignment:
        column_counts: The indexed counts for each column in the msa that are not gaped
    Returns:
        Weight of each sequence in the MSA - [2.390, 2.90, 5.33, 1.123, ...]
    """
    sequence_weights = []
    for record in bio_alignment:
        s = 0  # "diversity/surprise"
        for j, aa in enumerate(record.seq):
            s += (1 / (column_counts[j] * alignment_counts[j][aa]))
        sequence_weights.append(s)

    return sequence_weights


msa_supported_types = {'fasta': '.fasta', 'stockholm': '.sto'}
msa_generation_function = SequenceProfile.hhblits.__name__


# def find_column_observations(counts, **kwargs):
#     """Find total representation for each column in the alignment
#
#     Args:
#         counts (dict): {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...}
#     Keyword Args:
#         gaps=False (bool): Whether to count gaps (True) or not in the alignment
#     Returns:
#         (dict): {1: 210, 2:211, ...}
#     """
#     return {idx: sum_column_observations(aa_counts, **kwargs) for idx, aa_counts in counts.items()}


# def sum_column_observations(column: dict[str, int], gaps: bool = False, **kwargs) -> int:
#     """Sum the column weight for a single alignment dict column
#
#     Args:
#         column: {'A': 13, 'C': 1, 'D': 23, ...}
#         gaps: Whether to count gaps (True) or not
#     Returns:
#         s (int): Total counts in the alignment
#     """
#     if not gaps:
#         column.pop('-')
#
#     return sum(column.values())


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
                                         % (msa_to_prob_distribution.__name__, residue)
        alignment['frequencies'][residue] = {aa: count / total_column_weight for aa, count in amino_acid_counts.items()}

    return alignment


# def jensen_shannon_divergence(multiple_sequence_alignment, background_aa_probabilities, lambda_: float = 0.5):
#     """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict
#
#     Args:
#         multiple_sequence_alignment (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
#         background_aa_probabilities (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
#         lambda_: Value bounded between 0 and 1 to calculate the contribution from the observation versus the background
#     Returns:
#         (dict): {15: 0.732, ...} Divergence per residue bounded between 0 and 1. 1 is more divergent from background
#     """
#     return {residue_number: distribution_divergence(aa_probabilities, background_aa_probabilities, lambda_=lambda_)
#             for residue_number, aa_probabilities in multiple_sequence_alignment.items()}


def weight_gaps(divergence, representation, alignment_length):  # UNUSED
    for i in range(len(divergence)):
        divergence[i] = divergence[i] * representation[i] / alignment_length

    return divergence


def window_score(score_dict, window_len, score_lambda=0.5):  # UNUSED  incorporate into MultipleSequenceAlignment
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


def rank_possibilities(probability_dict):  # UNUSED  incorporate into MultipleSequenceAlignment
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


def multi_chain_alignment(mutated_sequences, **kwargs):
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
        return MultipleSequenceAlignment(alignment=total_alignment, **kwargs)
    else:
        raise DesignError(f'{multi_chain_alignment.__name__} - No sequences were found!')


# def generate_all_design_mutations(all_design_files, wild_type_file, pose_num=False):
#     """From a list of PDB's and a wild-type PDB, generate a list of 'A5K' style mutations
#
#     Args:
#         all_design_files (list): PDB files on disk to extract sequence info and compare
#         wild_type_file (str): PDB file on disk which contains a reference sequence
#     Returns:
#         mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
#     """
#     wild_type_pdb = Model.from_file(wild_type_file, log=None, entities=False)
#     pdb_sequences = {}
#     for file_name in all_design_files:
#         pdb = Model.from_file(file_name, log=None, entities=False)
#         pdb_sequences[pdb.name] = pdb.atom_sequences
#
#     return generate_multiple_mutations(wild_type_pdb.atom_sequences, pdb_sequences, pose_num=pose_num)


def pdb_to_pose_offset(reference_sequence: dict[Any, Sequence]) -> dict[Any, int]:
    """Take a dictionary with chain name as keys and return the length of Pose numbering offset

    Args:
        reference_sequence: {key1: 'MSGKLDA...', ...} or {key2: {1: 'A', 2: 'S', ...}, ...}
    Returns:
        {key1: 0, key2: 123, ...}
    """
    offset = {}
    # prior_chain = None
    prior_chains_len = prior_key = 0  # prior_key not used as 0 but to ensure initialized nonetheless
    for idx, key in enumerate(reference_sequence):
        if idx > 0:
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
                mutations[name][chain] = generate_mutations(reference[chain], sequence, offset=False)
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


def generate_mutations_from_reference(reference: str, sequences: dict[str, str]) -> \
        dict[str, dict[int, dict[str, str]]]:
    """Generate mutation data from multiple sequences dictionaries with regard to a single reference

    Args:
        reference: The reference sequence to compare each sequence against
        sequences: {alias: sequence, ...}
    Returns:
        {alias: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}
    """
    mutations = {alias: generate_mutations(reference, sequence, offset=False) for alias, sequence in sequences.items()}
    # add reference sequence mutations
    mutations[PUtils.reference_name] = \
        {sequence_idx: {'from': aa, 'to': aa} for sequence_idx, aa in enumerate(reference, 1)}

    return mutations


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


def hydrophobic_collapse_index(sequence: str, hydrophobicity: str = 'standard', lower_window: int = 3,
                               upper_window: int = 9) -> np.ndarray:
    """Calculate hydrophobic collapse index for a particular sequence of an iterable object and return an HCI array

    Args:
        sequence: The sequence to measure
        hydrophobicity: The degree of hydrophobicity to consider. Either 'standard' (FILV) or 'expanded' (FMILYVW)
        lower_window: The smallest window used to measure
        upper_window: The largest window used to measure
    Returns:
        1D array with the mean collapse score for every position on the input sequence
    """
    if hydrophobicity == 'background':  # Todo
        raise DesignError('This function is not yet possible')
    hydrophobicity_values = hydrophobicity_scale.get(hydrophobicity)
    # sequence_array = np.array([hydrophobicity_values.get(aa, 0) for aa in sequence])
    sequence_array = [hydrophobicity_values.get(aa, 0) for aa in sequence]

    # make an array with # of rows equal to range of windows used, length equal to # of characters in sequence
    sequence_length = len(sequence)
    range_size = upper_window + 1 - lower_window  # + 1 makes lower:upper inclusive of upper in range
    window_array = np.zeros((range_size, sequence_length))
    for array_idx, window_size in enumerate(map(float, range(lower_window, upper_window + 1))):  # make divisor float
        half_window = math.floor(window_size / 2)  # how far on each side should the window extend
        # # calculate score accordingly, with cases for N- and C-terminal windows
        # for seq_idx in range(half_window):  # N-terminus windows
        #     # add 1 as high slice not inclusive
        #     window_array[array_idx, seq_idx] = sequence_array[:seq_idx + half_window + 1].sum() / window_size
        # for seq_idx in range(half_window, sequence_length - half_window):  # continuous length windows
        #     # add 1 as high slice not inclusive
        #     window_array[array_idx, seq_idx] = \
        #         sequence_array[seq_idx - half_window: seq_idx + half_window + 1].sum() / window_size
        # for seq_idx in range(sequence_length - half_window, sequence_length):  # C-terminus windows
        #     # No add 1 as low slice inclusive
        #     window_array[array_idx, seq_idx] = sequence_array[seq_idx - half_window:].sum() / window_size
        #
        # # check if the range is even, then subtract 1/2 of the value of trailing and leading window values
        # if window_size % 2 == 0.:
        #     # subtract_half_leading_residue = sequence_array[half_window:] * 0.5 / window_size
        #     window_array[array_idx, :sequence_length - half_window] -= \
        #         sequence_array[half_window:] * 0.5 / window_size
        #     # subtract_half_trailing_residue = sequence_array[:sequence_length - half_window] * 0.5 / window_size
        #     window_array[array_idx, half_window:] -= \
        #         sequence_array[:sequence_length - half_window] * 0.5 / window_size

        # check if the range is odd or even, then calculate score accordingly, with cases for N- and C-terminal windows
        # if window_size % 2 == 1.:  # range is odd
        for seq_idx in range(sequence_length):
            position_sum = 0
            if seq_idx < half_window:  # N-terminus
                for window_position in range(seq_idx + half_window + 1):
                    position_sum += sequence_array[window_position]
            elif seq_idx + half_window >= sequence_length:  # C-terminus
                for window_position in range(seq_idx - half_window, sequence_length):
                    position_sum += sequence_array[window_position]
            else:
                for window_position in range(seq_idx - half_window, seq_idx + half_window + 1):
                    position_sum += sequence_array[window_position]
            window_array[array_idx, seq_idx] = position_sum / window_size
        # else:  # range is even
        #     for seq_idx in range(sequence_length):
        #         position_sum = 0
        #         if seq_idx < half_window:  # N-terminus
        #             for window_position in range(seq_idx + half_window + 1):
        #                 # if window_position == seq_idx + half_window:
        #                 #     position_sum += 0.5 * sequence_array[window_position]
        #                 # else:
        #                 position_sum += sequence_array[window_position]
        #         elif seq_idx + half_window >= sequence_length:  # C-terminus
        #             for window_position in range(seq_idx - half_window, sequence_length):
        #                 # if window_position == seq_idx - half_window:
        #                 #     position_sum += 0.5 * sequence_array[window_position]
        #                 # else:
        #                 position_sum += sequence_array[window_position]
        #         else:
        #             for window_position in range(seq_idx - half_window, seq_idx + half_window + 1):
        #                 # if window_position == seq_idx - half_window \
        #                 #         or window_position == seq_idx + half_window + 1:
        #                 #     position_sum += 0.5 * sequence_array[window_position]
        #                 # else:
        #                 position_sum += sequence_array[window_position]
        #         window_array[array_idx, seq_idx] = position_sum / window_size
        if window_size % 2 == 0.:  # range is even
            even_modifier = 0.5 / window_size
            # subtract_half_leading_residue = sequence_array[half_window:] * 0.5 / window_size
            window_array[array_idx, :sequence_length - half_window] -= \
                [hydrophobicity * even_modifier for hydrophobicity in sequence_array[half_window:]]
            # subtract_half_trailing_residue = sequence_array[:sequence_length - half_window] * 0.5 / window_size
            window_array[array_idx, half_window:] -= \
                [hydrophobicity * even_modifier for hydrophobicity in sequence_array[:sequence_length - half_window]]

    return window_array.mean(axis=0)  # hci


@handle_errors(errors=(FileNotFoundError,))
def read_fasta_file(file_name: AnyStr, **kwargs) -> Iterator[SeqRecord]:
    """Opens a fasta file and return a parser object to load the sequences to SeqRecords

    Args:
        file_name: The location of the file on disk
    Returns:
        An iterator of the sequences in the file [record1, record2, ...]
    """
    return SeqIO.parse(file_name, 'fasta')


@handle_errors(errors=(FileNotFoundError,))
def read_alignment(file_name: AnyStr, alignment_type: str = 'fasta', **kwargs) -> MultipleSeqAlignment:
    """Open an alignment file and parse the alignment to a Biopython MultipleSeqAlignment

    Args:
        file_name: The location of the file on disk
        alignment_type: The type of file that the alignment is stored in. Used for parsing
    Returns:
        The parsed alignment
    """
    return AlignIO.read(file_name, alignment_type)


def write_fasta(sequence_records: Iterable[SeqRecord], name: str = None, out_path: AnyStr = os.getcwd()) -> \
        AnyStr:
    """Write an iterator of SeqRecords to a .fasta file with fasta format. '.fasta' is appended if not specified in name

    Args:
        sequence_records: The sequences to write. Should be Biopython SeqRecord format
        name: The name of the file to output
        out_path: The location on disk to output file
    Returns:
        The name of the output file
    """
    file_name = os.path.join(out_path, name)
    if not file_name.endswith('.fasta'):
        file_name = f'{file_name}.fasta'

    SeqIO.write(sequence_records, file_name, 'fasta')

    return file_name


def write_sequence_to_fasta(sequence: str, name: str, file_name: AnyStr) -> AnyStr:
    """Write an iterator of SeqRecords to a .fasta file with fasta format. '.fasta' is appended if not specified in name

    Args:
        sequence: The sequence to write
        name: The name of the sequence
        file_name: The explicit name of the file
    Returns:
        The name of the output file
    """
    if not file_name.endswith('.fasta'):
        file_name = f'{file_name}.fasta'

    with open(file_name, 'w') as outfile:
        outfile.write(f'>{name}\n{sequence}\n')

    return file_name


def concatenate_fasta_files(file_names: Iterable[AnyStr], output: str = 'concatenated_fasta') -> AnyStr:
    """Take multiple fasta files and concatenate into a single file

    Args:
        file_names: The name of the files to concatenate
        out_path: The location on disk to output file
    Returns:
        The name of the output file
    """
    seq_records = []
    for file in file_names:
        seq_records.extend(list(read_fasta_file(file)))
    return write_fasta(seq_records, out_path=output)
