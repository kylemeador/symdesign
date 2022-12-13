from __future__ import annotations

import logging
import os
import subprocess
import time
from abc import ABC
from collections import namedtuple, UserList
from copy import deepcopy, copy
from itertools import repeat, count
from logging import Logger
from math import floor, exp, log2
from pathlib import Path
from typing import Sequence, Any, Iterable, get_args, Literal, AnyStr, Type

import numpy as np
from Bio import pairwise2, SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import utils
from .fragment import info
from .fragment.db import alignment_types_literal, alignment_types, fragment_info_type
from symdesign import metrics, utils as sdutils
from symdesign.utils import path as putils
# import dependencies.bmdca as bmdca

# Globals
logger = logging.getLogger(__name__)
hhblits_threads = 2
default_fragment_contribution = .5
zero_offset = 1
sequence_type_literal = Literal['reference', 'structure']
sequence_types: tuple[sequence_type_literal, ...] = get_args(sequence_type_literal)
# aa_counts = dict(zip(utils.protein_letters_alph1, repeat(0)))
aa_counts_alph3 = dict(zip(utils.protein_letters_alph3, repeat(0)))
blank_profile_entry = aa_counts_alph3.copy()
"""{utils.profile_keys, repeat(0))}"""
blank_profile_entry.update({'lod': aa_counts_alph3.copy(), 'info': 0., 'weight': 0.})
aa_nan_counts_alph3 = dict(zip(utils.protein_letters_alph3, repeat(np.nan)))
"""{protein_letters_alph3, repeat(numpy.nan))}"""
nan_profile_entry = aa_nan_counts_alph3.copy()
"""{utils.profile_keys, repeat(numpy.nan))}"""
nan_profile_entry.update({'lod': aa_nan_counts_alph3.copy(), 'info': 0., 'weight': 0.})  # 'type': residue_type,

aa_weighted_counts: info.aa_weighted_counts_type = dict(zip(utils.protein_letters_alph1, repeat(0)))
"""{protein_letters_alph1, repeat(0) | 'weight': 1}"""  # 'count': 0,
aa_weighted_counts['weight'] = 1
# aa_weighted_counts.update({'count': 0, 'weight': 1})
# """{protein_letters_alph1, repeat(0) | 'stats'=(0, 1)}"""
# aa_weighted_counts.update({'stats': (0, 1)})
numerical_profile = Type[np.ndarray]
"""The shape should be (number of residues, number of characters in the alphabet"""
subs_matrices = {'BLOSUM62': substitution_matrices.load('BLOSUM62')}
# 'uniclust30_2018_08'
latest_uniclust_background_frequencies = \
    {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228, 'I': 0.0534,
     'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599, 'S': 0.0791,
     'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}

# protein_letters_literal: tuple[str, ...] = get_args(utils.protein_letters_alph1_literal)
# numerical_translation = dict(zip(utils.protein_letters_alph1, range(len(utils.protein_letters_alph1))))
# protein_letters_alph1_extended: tuple[str, ...] = get_args(utils.protein_letters_alph1_extended_literal)


class MultipleSequenceAlignment:
    _alphabet_type: str
    _array: np.ndarray
    _frequencies: np.ndarray
    _gaps_per_position: np.ndarray
    _numerical_alignment: np.ndarray
    _sequence_indices: np.ndarray
    _numeric_translation_type: dict[str, int]
    """Given an amino acid alphabet type, return the corresponding numerical translation table"""
    alignment: MultipleSeqAlignment
    # counts: list[dict[extended_protein_letters_and_gap, int]]
    counts: list[list[int]] | np.ndarray
    frequencies: np.ndarray
    number_of_characters: int
    """The number of sequence characters in the character alphabet"""
    number_of_sequences: int
    """The number of sequences in the alignment"""
    length: int
    """The number of individual characters found in each sequence in the alignment"""
    observations: np.ndarray
    """The number of observations for each sequence index in the alignment"""
    query: str
    """The sequence used to perform the MultipleSequenceAlignment search"""
    query_length: int
    """The length of the query sequence. No gaps"""
    query_with_gaps: str
    """The sequence used to perform the MultipleSequenceAlignment search. May contain gaps from alignment"""

    def __init__(self, alignment: MultipleSeqAlignment = None, aligned_sequence: str = None,
                 alphabet: str = utils.protein_letters_alph1_gapped,
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
            query - 'MGSTHLVLK...' from aligned_sequence argument OR alignment argument, index 0
            query_with_gaps - 'MGS--THLVLK...'
            counts - {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
            frequencies - {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
            observations - {1: 210, 2:211, ...}}
        """
        if alignment is not None:
            self.alignment = alignment
            self.number_of_sequences = len(alignment)
            self.length = alignment.get_alignment_length()
            self.alphabet = alphabet
            self.number_of_characters = len(alphabet)
            if aligned_sequence is None:
                aligned_sequence = str(alignment[0].seq)
            # Add Info to 'meta' record as needed and populate an amino acid count dict (one-indexed)
            self.query = aligned_sequence.replace('-', '')
            self.query_length = len(self.query)
            self.query_with_gaps = aligned_sequence

            numerical_alignment = self.numerical_alignment
            self.counts = np.zeros((self.length, self.number_of_characters))
            # invert the "typical" format to length of the alignment in axis 0, and the numerical letters in axis 1
            for residue_idx in range(self.length):
                self.counts[residue_idx, :] = \
                    np.bincount(numerical_alignment[:, residue_idx], minlength=self.number_of_characters)

            # self.observations = find_column_observations(self.counts, **kwargs)
            self._gap_index = 0
            if count_gaps:
                self.observations = [self.number_of_sequences for _ in range(self.length)]
            else:
                # gap_observations = [_aa_counts['-'] for _aa_counts in self.counts]  # list[dict]
                # gap_observations = [_aa_counts[0] for _aa_counts in self.counts]  # list[list]
                # self.observations = [counts - gap for counts, gap in zip(self.observations, gap_observations)]
                # Find where gaps and unknown start. They are always at the end
                if 'gapped' in self.alphabet_type:
                    self._gap_index -= 1
                if 'unknown' in self.alphabet_type:
                    self._gap_index -= 1

                self.observations = self.counts[:, :self._gap_index].sum(axis=1)
                if not np.any(self.observations):  # Check if an observation is 0
                    raise ValueError("Can't have a MSA column (sequence index) with 0 observations. Found at ("
                                     f'{",".join(map(str, np.flatnonzero(self.observations)))}')
                    #                f'{",".join(str(idx) for idx, pos in enumerate(self.observations) if not pos)}')

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
                self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
                for record in self.alignment:
                    for i, aa in enumerate(record.seq):
                        self._counts[i][utils.numerical_translation_alph1_gapped[aa]] += 1
                        # self.counts[i][aa] += 1
                print('OLD self._counts', self._counts)
                self._observations = [sum(_aa_counts[:self._gap_index]) for _aa_counts in self._counts]  # list[list]

                sequence_weights_ = weight_sequences(self._counts, self.alignment, self._observations)
                print('OLD sequence_weights_', sequence_weights_)

            if sequence_weights is not None:  # overwrite the current counts with weighted counts
                self.sequence_weights = sequence_weights
                # Todo update this as well
                self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
                for record in self.alignment:
                    for i, aa in enumerate(record.seq):
                        self._counts[i][utils.numerical_translation_alph1_gapped[aa]] += sequence_weights_[i]
                        # self.counts[i][aa] += sequence_weights[i]
                print('OLD sequence_weight self._counts', self._counts)

                # add each sequence weight to the indices indicated by the numerical_alignment
                self.counts = np.zeros((self.length, len(utils.protein_letters_alph1_gapped)))
                for idx in range(self.number_of_sequences):
                    self.counts[:, numerical_alignment[idx]] += sequence_weights[idx]
                print('sequence_weight self.counts', self.counts)
            else:
                self.sequence_weights = []

    @classmethod
    def from_stockholm(cls, file, **kwargs):
        try:
            return cls(alignment=read_alignment(file, alignment_type='stockholm'), **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"The multiple sequence alignment file '{file}' doesn't exist")

    @classmethod
    def from_fasta(cls, file):
        try:
            return cls(alignment=read_alignment(file))
        except FileNotFoundError:
            raise FileNotFoundError(f"The multiple sequence alignment file '{file}' doesn't exist")

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
    def alphabet_type(self) -> str:
        """The type of alphabet that the alignment is mapped to numerically"""
        try:
            return self._alphabet_type
        except AttributeError:
            if self.alphabet.startswith('ACD'):  # 1 letter order
                self._alphabet_type = 'protein_letters_alph1'
            else:  # 3 letter order
                self._alphabet_type = 'protein_letters_alph3'

            if 'X' in self.alphabet:  # Unknown
                self._alphabet_type += '_unknown'
            if '-' in self.alphabet:  # Gapped
                self._alphabet_type += '_gapped'

            return self._alphabet_type

    @property
    def query_indices(self) -> np.ndarray:
        """Returns the query as a boolean array (1, length) where gaps ("-") are False"""
        try:
            return self._sequence_indices[0]
        except AttributeError:
            self._sequence_indices = self.array != b'-'
            # self._sequence_indices = np.isin(self.array, b'-', invert=True)
            return self._sequence_indices[0]

    @property
    def sequence_indices(self) -> np.ndarray:
        """Returns the alignment as a boolean array (number_of_sequences, length) where gaps ("-") are False"""
        try:
            return self._sequence_indices
        except AttributeError:
            self._sequence_indices = self.array != b'-'
            # self._sequence_indices = np.isin(self.array, b'-', invert=True)
            return self._sequence_indices

    @sequence_indices.setter
    def sequence_indices(self, sequence_indices: np.ndarray):
        self._sequence_indices = sequence_indices

    @property
    def numerical_alignment(self) -> np.ndarray:
        """Return the alignment as an integer array (number_of_sequences, length) of the amino acid characters

        Maps the instance .alphabet characters to their resulting sequence index
        """
        try:
            return self._numerical_alignment
        except AttributeError:
            try:
                translation_type = self._numeric_translation_type  # Todo clean setting of this with self.alphabet_type
            except AttributeError:
                self._numeric_translation_type = utils.create_translation_tables(self.alphabet_type)
                translation_type = self._numeric_translation_type

            self._numerical_alignment = np.vectorize(translation_type.__getitem__)(self.array)
            return self._numerical_alignment

    @property
    def array(self) -> np.ndarray:
        """Return the alignment as a character array (number_of_sequences, length) with numpy.string_ dtype"""
        try:
            return self._array
        except AttributeError:
            self._array = np.array([list(record) for record in self.alignment], np.string_)
            return self._array

    @property
    def frequencies(self) -> np.ndarray:
        """Access the per residue (axis=0) amino acid frequencies (axis=1) bounded between 0 and 1"""
        # self._frequencies = [[count/observation for count in amino_acid_counts[:self._gap_index]]  # don't use the gap
        #                      for amino_acid_counts, observation in zip(self._counts, self._observations)]
        # print('OLD self._frequencies', self._frequencies)

        # self.frequencies = np.zeros((self.length, len(utils.protein_letters_alph1)))  # self.counts.shape)
        # for residue_idx in range(self.length):
        #     self.frequencies[residue_idx, :] = self.counts[:, :self._gap_index] / self.observations
        try:
            return self._frequencies
        except AttributeError:  # Don't use gapped indices
            self._frequencies = self.counts[:, :self._gap_index] / self.observations[:, None]
        return self._frequencies

    @property
    def gaps_per_postion(self) -> np.ndarray:
        try:
            return self._gaps_per_position
        except AttributeError:
            self._gaps_per_position = self.number_of_sequences - self.observations
        return self._gaps_per_position

    def get_probabilities_from_profile(self, profile: numerical_profile) -> np.ndarry:
        """For each sequence in the alignment, extract the values from a profile corresponding to the amino acid type
        of each residue in each sequence

        Args:
            profile: A profile of values with shape (length, alphabet_length) where length is the number_of_residues
        Returns:
            The array with shape (number_of_sequences, length) with the value for each amino acid index in profile
        """
        # transposed_alignment = self.numerical_alignment.T
        return np.take_along_axis(profile, self.numerical_alignment.T, axis=1).T
        # observed = {profile: np.take_along_axis(background, transposed_alignment, axis=1).T
        #             for profile, background in backgrounds.items()}
        # observed = {profile: np.where(np.take_along_axis(background, transposed_alignment, axis=1) > 0, 1, 0).T
        #             for profile, background in backgrounds.items()}


def sequence_to_numeric(sequence: Sequence) -> numerical_profile:  # np.ndarray:
    """Convert a position specific profile matrix into a numeric array

    Args:
        sequence: The sequence to encode
    Returns:
        The numerically encoded sequence where each entry along axis=0 is the indexed amino acid. Indices are according
            to the 1 letter alphabetical amino acid
    """
    _array = np.array(list(sequence), np.string_)
    return np.vectorize(utils.numerical_translation_alph1_bytes.__getitem__)(_array)


def sequences_to_numeric(sequences: list[Sequence]) -> numerical_profile:  # np.ndarray[numerical_profile]:
    """Convert a position specific profile matrix into a numeric array

    Args:
        sequences: The sequences to encode
    Returns:
        The numerically encoded sequence where each entry along axis=0 is the indexed amino acid. Indices are according
            to the 1 letter alphabetical amino acid
    """
    _array = np.array([list(sequence) for sequence in sequences], np.string_)
    return np.vectorize(utils.numerical_translation_alph1_bytes.__getitem__)(_array)


def numeric_to_sequence(numeric_sequence: np.ndarray, alphabet_order: int = 1) -> np.ndarray:
    """Convert a numeric sequence array into a sequence array

    Args:
        numeric_sequence: The sequence to convert
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3
    Returns:
        The alphabetically encoded sequence where each entry along axis=-1 is the one letter amino acid
    """
    if alphabet_order == 1:
        return np.vectorize(utils.sequence_translation_alph1.__getitem__)(numeric_sequence)
    elif alphabet_order == 3:
        return np.vectorize(utils.sequence_translation_alph3.__getitem__)(numeric_sequence)
    else:
        raise ValueError(f"The alphabet_order {alphabet_order} isn't valid. Choose from either 1 or 3")


def pssm_as_array(pssm: profile_dictionary, alphabet: str = utils.protein_letters_alph1, lod: bool = False) \
        -> np.ndarray:
    """Convert a position specific profile matrix into a numeric array

    Args:
        pssm: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                  2: {}, ...}
        alphabet: The amino acid alphabet to use. Array values will be returned in this order
        lod: Whether to return the array for the log of odds values
    Returns:
        The numerically encoded pssm where each entry along axis 0 is the position, and the entries on axis 1 are the
            frequency data at every indexed amino acid. Indices are according to the specified amino acid alphabet,
            i.e array([[0.1, 0.01, 0.12, ...], ...])
    """
    if lod:
        return np.array([[position_info['lod'][aa] for aa in alphabet]
                         for position_info in pssm.values()], dtype=np.float32)
    else:
        return np.array([[position_info[aa] for aa in alphabet]
                         for position_info in pssm.values()], dtype=np.float32)
        # return np.vectorize(utils.numerical_translation_alph1_bytes.__getitem__)(_array)


# Todo rename to concatenate_to_ordered_dictionary?
def concatenate_profile(profiles: Iterable[Any], start_at: int = 1) -> dict[int, Any]:
    """Combine a list of profiles (parsed PSSMs) by incrementing the entry index for each additional profile

    Args:
        profiles: The profiles to concatenate
        start_at: The integer to start the resulting dictionary at
    Returns
        The concatenated input profiles, make a concatenated PSSM
            {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                 'weight': 0.00}, ...}}
    """
    _count = count(start_at)
    return {next(_count): position_profile for profile in profiles for position_profile in profile.values()}
    # new_key = 1
    # for profile in profiles:
    #     for position_profile in profile.values():
    #         new_profile[new_key] = position_profile
    #         new_key += 1
    #
    # return new_profile


def write_pssm_file(pssm: profile_dictionary, file_name: AnyStr = None, name: str = None,
                    out_dir: AnyStr = os.getcwd()) -> AnyStr | None:
    """Create a PSI-BLAST format PSSM file from a PSSM dictionary. Assumes residue numbering is correct!

    Args:
        pssm: A dictionary which has the keys: 'A', 'C', ... (all aa's), 'lod', 'type', 'info', 'weight'
        file_name: The explicit name of the file
        name: The name of the file. Will be used as the default file_name base name if file_name not provided
        out_dir: The location on disk to output the file. Only used if file_name not explicitly provided
    Returns:
        Disk location of newly created .pssm file
    """
    if not pssm:
        return None

    # Find out if the pssm has values expressed as frequencies (percentages) or as counts and modify accordingly
    if isinstance(list(pssm.values())[0]['lod']['A'], float):
        separation1 = " " * 4
    else:
        separation1 = " " * 3

    if file_name is None:
        if name is None:
            raise ValueError(f'Must provide argument "file_name" or "name" as a str to {write_sequences.__name__}')
        else:
            file_name = os.path.join(out_dir, name)

    if os.path.splitext(file_name)[-1] == '':  # No extension
        file_name = f'{file_name}.pssm'

    with open(file_name, 'w') as f:
        f.write(f'\n\n{" " * 12}{separation1.join(utils.protein_letters_alph3)}'
                f'{separation1}{(" " * 3).join(utils.protein_letters_alph3)}\n')
        for residue_number, profile in pssm.items():
            if isinstance(profile['lod']['A'], float):  # lod_freq:  # relevant for favor_fragment
                lod_string = ' '.join(f'{profile["lod"][aa]:>4.2f}' for aa in utils.protein_letters_alph3) \
                    + ' '
            else:
                lod_string = ' '.join(f'{profile["lod"][aa]:>3d}' for aa in utils.protein_letters_alph3) \
                    + ' '

            if isinstance(profile['A'], float):  # counts_freq: # relevant for freq calculations
                counts_string = ' '.join(f'{floor(profile[aa] * 100):>3.0f}' for aa in utils.protein_letters_alph3) \
                    + ' '
            else:
                counts_string = ' '.join(f'{profile[aa]:>3d}' for aa in utils.protein_letters_alph3) \
                    + ' '
            f.write(f'{residue_number:>5d} {profile["type"]:1s}   {lod_string:80s} {counts_string:80s} '
                    f'{round(profile.get("info", 0.), 4):4.2f} {round(profile.get("weight", 0.), 4):4.2f}\n')

    return file_name


lod_dictionary: dict[utils.protein_letters_literal, int]
profile_values: float | str | lod_dictionary
profile_entry: Type[dict[utils.profile_keys, profile_values]]
profile_dictionary: Type[dict[int, dict[utils.profile_keys, profile_values]]]
"""{1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
        'type': 'W', 'info': 0.00, 'weight': 0.00}, {...}}
"""
alignment_programs_literal = Literal['hhblits', 'psiblast']
alignment_programs: tuple[str, ...] = get_args(alignment_programs_literal)
profile_types = Literal['evolutionary', 'fragment', '']


class Profile(UserList):
    data: list[profile_entry]
    """[{'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...},
         'type': 'W', 'info': 3.20, 'weight': 0.73},
        {}, ...]
    """
    def __init__(self, profile: list[profile_entry] = None, dtype: str = None, **kwargs):
        # Todo?
        if profile is None:
            profile = []
        elif isinstance(profile, list):
            try:
                self.available_keys = tuple(profile[0].keys())
            except IndexError:  # No values in list
                pass

        super().__init__(initlist=profile, **kwargs)  # initlist sets UserList.data to profile_dictionary

        self.dtype = 'profile' if dtype is None else dtype
        if not self.data:  # Set up an empty Profile
            self.available_keys = tuple()
            self.lods = self.types = self.weights = self.info = None
        else:
            if 'lod' in self.available_keys:
                self.lods = [position_info['lod'] for position_info in self]
            if 'type' in self.available_keys:
                self.types = [position_info['type'] for position_info in self]
            if 'weight' in self.available_keys:
                self.weights = [position_info['weight'] for position_info in self]
            if 'info' in self.available_keys:
                self.info = [position_info['info'] for position_info in self]

    def as_array(self, alphabet: str = utils.protein_letters_alph1, lod: bool = False) -> np.ndarray:
        """Convert the Profile into a numeric array

        Args:
            alphabet: The amino acid alphabet to use. Array values will be returned in this order
            lod: Whether to return the array for the log of odds values
        Returns:
            The numerically encoded pssm where each entry along axis 0 is the position, and the entries on axis 1 are
            the frequency data at every indexed amino acid. Indices are according to the specified amino acid alphabet,
                i.e array([[0.1, 0.01, 0.12, ...], ...])
        """
        if lod:
            return np.array([[position_info['lod'][aa] for aa in alphabet]
                             for position_info in self], dtype=np.float32)
        else:
            return np.array([[position_info[aa] for aa in alphabet]
                             for position_info in self], dtype=np.float32)

    def write(self, file_name: AnyStr = None, name: str = None, out_dir: AnyStr = os.getcwd()) -> AnyStr | None:
        """Create a PSI-BLAST format PSSM file from a PSSM dictionary. Assumes residue numbering is 1 to last entry

        Args:
            file_name: The explicit name of the file
            name: The name of the file. Will be used as the default file_name base name if file_name not provided
            out_dir: The location on disk to output the file. Only used if file_name not explicitly provided
        Returns:
            Disk location of newly created .pssm file
        """
        # Format the Profile according to the write_pssm_file mechanism
        # This takes the shape of 1 to the last entry
        if self.dtype in putils.fragment_profile:
            # Need to convert np.nan to zeros
            logger.warning(f'Converting {self.dtype} type Profile np.nan values to 0.0')
            data = np.nan_to_num([[position_info[aa] for aa in utils.protein_letters_alph3]
                                  for position_info in self]).tolist()
        else:
            data = [[position_info[aa] for aa in utils.protein_letters_alph3] for position_info in self]

        # Find out if the pssm has values expressed as frequencies (percentages) or as counts and modify accordingly
        if isinstance(self.lods[0]['A'], float):
            separation1 = " " * 4
        else:
            separation1 = " " * 3
            # lod_freq = True
        # if type(pssm[first_key]['A']) == float:
        #     counts_freq = True

        if file_name is None:
            if name is None:
                raise ValueError(f'Must provide argument "file_name" or "name" as a str to {write_sequences.__name__}')
            else:
                file_name = os.path.join(out_dir, name)

        if os.path.splitext(file_name)[-1] == '':  # No extension
            file_name = f'{file_name}.pssm'

        with open(file_name, 'w') as f:
            f.write(f'\n\n{" " * 12}{separation1.join(utils.protein_letters_alph3)}'
                    f'{separation1}{(" " * 3).join(utils.protein_letters_alph3)}\n')
            for residue_number, (entry, lod, _type, info, weight) in enumerate(
                    zip(data, self.lods, self.types, self.info, self.weights), 1):
                if isinstance(lod['A'], float):  # relevant for favor_fragment
                    lod_string = \
                        ' '.join(f'{lod[aa]:>4.2f}' for aa in utils.protein_letters_alph3) + ' '
                else:
                    lod_string = \
                        ' '.join(f'{lod[aa]:>3d}' for aa in utils.protein_letters_alph3) + ' '

                if isinstance(entry[0], float):  # relevant for freq calculations
                    counts_string = ' '.join(f'{floor(value * 100):>3.0f}' for value in entry) + ' '
                else:
                    counts_string = ' '.join(f'{value:>3d}' for value in entry) + ' '

                f.write(f'{residue_number:>5d} {_type:1s}   {lod_string:80s} {counts_string:80s} '
                        f'{round(info, 4):4.2f} {round(weight, 4):4.2f}\n')

        return file_name


# class Profile(UserDict):
#     data: profile_dictionary
#     """{1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...},
#            'type': 'W', 'info': 3.20, 'weight': 0.73},
#         2: {}, ...}
#     """
#     def __init__(self, profile: profile_dictionary, dtype: str = None, **kwargs):
#         super().__init__(initialdata=profile, **kwargs)  # initialdata sets UserDict.data to profile_dictionary
#
#         self.dtype = 'profile' if dtype is None else dtype
#         if not self.data:  # Set up an empty Profile
#             self.available_keys = tuple()
#         else:
#             self.available_keys = tuple(next(iter(self.values())).keys())
#             if 'lod' in self.available_keys:
#                 self.lods = {position: position_info['lod'] for position, position_info in self.items()}
#             if 'weight' in self.available_keys:
#                 self.weights = {position: position_info['weight'] for position, position_info in self.items()}
#             if 'info' in self.available_keys:
#                 self.info = {position: position_info['info'] for position, position_info in self.items()}
#
#     def as_array(self, alphabet: str = utils.protein_letters_alph1, lod: bool = False) -> np.ndarray:
#         """Convert the Profile into a numeric array
#
#         Args:
#             alphabet: The amino acid alphabet to use. Array values will be returned in this order
#             lod: Whether to return the array for the log of odds values
#         Returns:
#             The numerically encoded pssm where each entry along axis 0 is the position, and the entries on axis 1 are
#             the frequency data at every indexed amino acid. Indices are according to the specified amino acid alphabet,
#                 i.e array([[0.1, 0.01, 0.12, ...], ...])
#         """
#         if lod:
#             return np.array([[position_info['lod'][aa] for aa in alphabet]
#                              for position_info in self.values()], dtype=np.float32)
#         else:
#             return np.array([[position_info[aa] for aa in alphabet]
#                              for position_info in self.values()], dtype=np.float32)


class SequenceProfile(ABC):
    """Contains the sequence information for a Structure. Should always be subclassed by a Structure object.
    Currently, Chain, Entity, Model and Pose contain the necessary .reference_sequence property.
    Any Structure object with a .reference_sequence attribute could be used however
    """
    _alpha: float
    _collapse_profile: np.ndarray  # pd.DataFrame
    _fragment_db: info.FragmentInfo | None
    _fragment_profile: list[list[list]] | list[profile_entry]
    _hydrophobic_collapse: np.ndarray
    _msa: MultipleSequenceAlignment | None
    _sequence_numeric: np.ndarray
    a3m_file: AnyStr | None
    alpha: list[float]
    # alpha: dict[int, float]
    disorder: dict[int, dict[str, str]]
    evolutionary_profile: dict | profile_dictionary
    fragment_map: dict[int, dict[int, list[fragment_info_type]]]
    """{1: {-2: [{'source': 'mapped', 'cluster': (1, 2, 123), 'match': 0.6}, ...], -1: [], ...},
        2: {}, ...}
    """
    fragment_profile: Profile | None  # | profile_dictionary
    h_fields: np.ndarray | None
    j_couplings: np.ndarray | None
    log: Logger
    msa_file: AnyStr | None
    name: str
    number_of_residues: int
    profile: dict | profile_dictionary
    pssm_file: AnyStr | None
    reference_sequence: str
    residues: list['structure.base.Residue']
    sequence_file: AnyStr | None
    sequence: str

    def __init__(self, **kwargs):
        # super().__init__()
        super().__init__(**kwargs)
        # self.design_pssm_file = None
        # {(ent1, ent2): [{mapped: res_num1, paired: res_num2, cluster: id, match: score}, ...], ...}
        # self._fragment_db = None
        self.a3m_file = None
        self.alpha = []  # {}
        # Using .profile as attribute instead
        # self.design_profile = {}  # design specific scoring matrix
        self.evolutionary_profile = {}  # position specific scoring matrix
        self.fragment_map = {}  # fragment information
        self.fragment_profile = None  # fragment specific scoring matrix
        self._fragment_profile = []  # hidden fragment specific scoring matrix
        self.h_fields = None
        self.j_couplings = None
        self.msa_file = None
        self.profile = {}  # design/structure specific scoring matrix
        self.pssm_file = None
        self.sequence_file = None

    @property
    def offset_index(self) -> int:
        """Return the starting index for the SequenceProfile based on pose numbering of the residues. Zero-indexed"""
        return self.residues[0].index

    # @offset_index.setter
    # def offset_index(self, offset_index):
    #     self._entity_offset = offset_index

    @property
    def msa(self) -> MultipleSequenceAlignment | None:
        """The MultipleSequenceAlignment object for the instance"""
        try:
            return self._msa
        except AttributeError:
            return None

    @msa.setter
    def msa(self, msa: MultipleSequenceAlignment):
        if isinstance(msa, MultipleSequenceAlignment):
            self._msa = copy(msa)
            self.fit_msa_to_structure()
        else:
            self.log.warning(f"The passed msa (type: {type(msa).__name__}) isn't of the required type "
                             f"{MultipleSequenceAlignment.__name__}")

    @property
    def sequence_numeric(self) -> np.ndarray:
        """Return the sequence as an integer array (number_of_residuces, alphabet_length) of the amino acid characters

        Maps "ACDEFGHIKLMNPQRSTVWY-" to the resulting index
        """
        try:
            return self._sequence_numeric
        except AttributeError:
            self._sequence_array = np.array(list(self.sequence), np.string_)
            self._sequence_numeric = \
                np.vectorize(utils.numerical_translation_alph1_gapped_bytes.__getitem__, otypes='l')(self._sequence_array)
            # using otypes='i' as the datatype for int32. 'f' would be for float32
            # using otypes='l' as the datatype for int64. 'd' would be for float64
            # self.log.critical(f'The sequence_numeric dtype is {self._sequence_numeric.dtype}. It should be int64')
            # self._sequence_numeric = self._sequence_numeric.astype(np.int32)
            return self._sequence_numeric

    # @property
    def hydrophobic_collapse(self, **kwargs) -> np.array:
        """Return the hydrophobic collapse for the Sequence

        Keyword Args:
            hydrophobicity: int = 'standard' – The hydrophobicity scale to consider. Either 'standard' (FILV),
                'expanded' (FMILYVW), or provide one with 'custom' keyword argument
            custom: mapping[str, float | int] = None – A user defined mapping of amino acid type, hydrophobicity value pairs
            alphabet_type: alphabet_types = None – The amino acid alphabet if the sequence consists of integer characters
            lower_window: int = 3 – The smallest window used to measure
            upper_window: int = 9 – The largest window used to measure
        """
        try:
            return self._hydrophobic_collapse
        except AttributeError:
            self._hydrophobic_collapse = metrics.hydrophobic_collapse_index(self.sequence, **kwargs)
            return self._hydrophobic_collapse

    def get_sequence_probabilities_from_profile(self, dtype: profile_types = None, precomputed: numerical_profile = None)\
            -> numerical_profile:
        """Extract the values from a profile corresponding to the amino acid type of each residue in the sequence

        Args:
            dtype: The profile type to sample from. Can be one of 'evolutionary', 'fragment', or None which
                combines 'evolutionary' and 'fragment'
            precomputed: If the profile is precomputed, pass as precomputed=profile_variable
        Returns:
            The array with shape (number_of_sequences, length) with the value for each amino acid index in profile
        """
        if precomputed is None:
            if dtype is None:
                profile_of_interest = self.profile
            else:
                profile_of_interest = getattr(self, f'{dtype}_profile')

            profile_of_interest = pssm_as_array(profile_of_interest)
        else:
            profile_of_interest = precomputed

        # return profile_of_interest[:, self.sequence_numeric]
        try:
            return np.take_along_axis(profile_of_interest, self.sequence_numeric[:, None], axis=1).squeeze()
        except IndexError:  # The profile_of_interest and sequence are different sizes
            raise IndexError(f'The profile length {profile_of_interest.shape[0]} != '
                             f'the number_of_residues {self.number_of_residues}')

    def add_profile(self, evolution: bool = True, fragments: bool | list[fragment_info_type] = True,
                    null: bool = False, **kwargs):
        """Add the evolutionary and fragment profiles onto the SequenceProfile

        Args:
            evolution: Whether to add evolutionary information to the sequence profile
            fragments: Whether to add fragment information to the sequence profile. Can pass fragment instances as well
            null: Whether to use a null profile (non-functional) as the sequence profile
        Keyword Args:
            alignment_type: alignment_types_literal = 'mapped' – Either 'mapped' or 'paired' indicating how the fragment
                observations were generated relative to this Structure. Is it mapped to this Structure or was it paired
                to it?
            out_dir: AnyStr = os.getcwd() - Location where sequence files should be written
            favor_fragments: bool = False - Whether to favor fragment profile in the lod score of the resulting profile
                Currently this routine is only used for Rosetta designs where the fragments should be favored by a
                particular weighting scheme. By default, the boltzmann weighting scheme is applied
            boltzmann: bool = False - Whether to weight the fragment profile by a Boltzmann probability scaling using
                the formula lods = exp(lods[i]/kT)/Z, where Z = sum(exp(lods[i]/kT)), and kT is 1 by default.
                If False, residues are weighted by the residue local maximum lod score in a linear fashion
                All lods are scaled to a maximum provided in the Rosetta REF2015 per residue reference weight.
        Sets:
            self.profile (profile_dictionary)
        """
        if null or (not evolution and not fragments):
            self.profile = self.evolutionary_profile = self.create_null_profile()
            self.fragment_profile = Profile(list(self.create_null_profile(nan=True, zero_index=True).values()),
                                            dtype='fragment')
            return
            # evolution = fragments = False
            # self.add_evolutionary_profile(null=null, **kwargs)

        if evolution:  # add evolutionary information to the SequenceProfile
            if not self.evolutionary_profile:
                self.add_evolutionary_profile(**kwargs)

            # Check the profile and try to generate again if it is incorrect
            first = True
            while not self.verify_evolutionary_profile():
                if first:
                    self.log.info(f'Generating a new profile for {self.name}')
                    self.add_evolutionary_profile(force=True)
                    first = False
                else:
                    # Todo RuntimeError()
                    raise RuntimeError('evolutionary_profile generation got stuck')
        else:  # Set the evolutionary_profile to null
            self.evolutionary_profile = self.create_null_profile()

        if isinstance(fragments, list):  # Add fragment information to the SequenceProfile
            self.add_fragments_to_profile(fragments, **kwargs)
            self.simplify_fragment_profile()
        elif fragments:  # If was passed as True
            if not self.alpha:
                raise AttributeError('Fragments were specified but have not been added to the SequenceProfile! '
                                     f'Call {self.add_fragments_to_profile.__name__} with fragment information or pass'
                                     f' fragments and alignment_type to {self.add_profile.__name__}')
            # # Fragments have already been added, connect DB info
            # elif self.fragment_db:
            #     retrieve_fragments = [fragment['cluster'] for idx_d in self.fragment_map.values()
            #                           for fragments in idx_d.values() for fragment in fragments]
            #     self.fragment_db.load_cluster_info(ids=retrieve_fragments)
            # else:
            #     raise AttributeError('Fragments were specified but there is no fragment database attached. Ensure '
            #                          f'{self.fragment_db.__name__} is set before requesting fragment information')

            # # Process fragment profile from self.fragment_profile
            # self.process_fragment_profile()

        self.calculate_profile(**kwargs)

    def verify_evolutionary_profile(self) -> bool:
        """Returns True if the evolutionary_profile and Structure sequences are equivalent"""
        if self.number_of_residues != len(self.evolutionary_profile):
            self.log.warning(f'{self.name}: Profile and {type(self).__name__} are different lengths! Profile='
                             f'{len(self.evolutionary_profile)}, Pose={self.number_of_residues}')
            return False

        # if not rerun:
        # Check sequence from Pose and self.profile to compare identity before proceeding
        incorrect_count = 0
        for residue, position_data in zip(self.residues, self.evolutionary_profile.values()):
            profile_res_type = position_data['type']
            pose_res_type = utils.protein_letters_3to1[residue.type]
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

    def add_evolutionary_profile(self, file: AnyStr = None, out_dir: AnyStr = os.getcwd(),
                                 profile_source: alignment_programs_literal = putils.hhblits, force: bool = False):
        """Add the evolutionary profile to the Structure. If the profile isn't provided, it is generated through search
        of homologous protein sequences using the profile_source argument

        Args:
            file: Location where profile file should be loaded from
            out_dir: Location where sequence files should be written
            profile_source: One of 'hhblits' or 'psiblast'
            force: Whether to force generation of a new profile
        Sets:
            self.evolutionary_profile (profile_dictionary)
        """
        if profile_source not in alignment_programs:  # [putils.hhblits, 'psiblast']:
            raise ValueError(f'{self.add_evolutionary_profile.__name__}: Profile generation only possible from '
                             f'{", ".join(alignment_programs)} not {profile_source}')

        if file is not None:
            self.pssm_file = file
        else:  # Check to see if the files of interest already exist
            # Extract/Format Sequence Information. SEQRES is prioritized if available
            if not self.sequence_file:  # not made/provided before add_evolutionary_profile, make a new one
                self.write_sequence_to_fasta('reference', out_dir=out_dir)
            elif not os.path.exists(self.sequence_file) or force:
                self.log.debug(f'{self.name} Sequence={self.reference_sequence}')
                self.write_sequence_to_fasta('reference', file_name=self.sequence_file)
                self.log.debug(f'{self.name} sequence file: {self.sequence_file}')

            # temp_file = os.path.join(out_path, f'{self.name}.hold')
            temp_file = Path(out_dir, f'{self.name}.hold')
            self.pssm_file = os.path.join(out_dir, f'{self.name}.hmm')
            if not os.path.exists(self.pssm_file) or force:
                if not os.path.exists(temp_file):  # No work on this pssm file has been initiated
                    # Create blocking file to prevent excess work
                    with open(temp_file, 'w') as f:
                        self.log.info(f'Fetching "{self.name}" sequence data')
                    self.log.debug(f'{self.name} Evolutionary Profile not yet created')
                    # if profile_source == putils.hhblits:
                    #     self.log.info(f'Generating HHM Evolutionary Profile for {self.name}')
                    #     self.hhblits(out_dir=out_dir)
                    # else:
                    self.log.info(f'Generating Evolutionary Profile for {self.name}')
                    # self.psiblast(out_dir=out_dir)
                    getattr(self, profile_source)(out_path=out_dir)
                    temp_file.unlink(missing_ok=True)
                    # if os.path.exists(temp_file):
                    #     os.remove(temp_file)
                else:  # Block is in place, another process is working
                    self.log.info(f'Waiting for "{self.name}" profile generation...')
                    while not os.path.exists(self.pssm_file):
                        if int(time.time()) - int(os.path.getmtime(temp_file)) > 5400:  # > 1 hr 30 minutes have passed
                            # os.remove(temp_file)
                            temp_file.unlink(missing_ok=True)
                            raise RuntimeError(f'{self.add_evolutionary_profile.__name__}: Generation of the '
                                               f'profile for {self.name} took longer than the time limit. Job killed!')
                        time.sleep(20)

        # These functions set self.evolutionary_profile
        getattr(self, f'parse_{profile_source}_pssm')()

    def create_null_profile(self, nan: bool = True, zero_index: bool = False, **kwargs) -> profile_dictionary:
        """Make a blank profile

        Args:
            nan: Whether to fill the null profile with np.nan
            zero_index: bool = False - If True, return the dictionary with zero indexing
        Returns:
            Dictionary containing profile information with keys as the index (zero or one-indexed), values as PSSM
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        offset = 0 if zero_index else zero_offset

        if nan:
            profile = {residue: nan_profile_entry.copy()
                       for residue in range(offset, self.number_of_residues + offset)}
        else:
            profile = {residue: blank_profile_entry.copy()
                       for residue in range(offset, self.number_of_residues + offset)}

        for residue_data, residue_type in zip(profile.values(), self.sequence):
            residue_data['type'] = residue_type

        return profile

    def fit_evolutionary_profile_to_structure(self):
        """From an evolutionary profile generated according to a reference sequence, align the profile to the Structure
        sequence, removing information for residues not present in the Structure

        Sets:
            self.evolutionary_profile (profile_dictionary)
        """
        # Generate the disordered indices which are positions in reference that are missing in structure
        disorder = self.disorder
        # Removal of these positions from self.evolutionary_profile will produce a properly indexed profile
        new_residue_number = count(1)
        structure_evolutionary_profile = {next(new_residue_number): residue_data
                                          for residue_number, residue_data in self.evolutionary_profile.items()
                                          if residue_number not in disorder}
        # for residue_number, residue_data in self.evolutionary_profile.items():
        #     if residue_number not in disorder:
        #         structure_evolutionary_profile[next(new_residue_number)] = residue_data
        #         # new_residue_number += 1

        self.log.debug(f'{self.fit_evolutionary_profile_to_structure.__name__}:\n\tOld:\n'
                       f'{"".join(res["type"] for res in self.evolutionary_profile.values())}\n\tNew:\n'
                       f'{"".join(res["type"] for res in structure_evolutionary_profile.values())}')
        self.evolutionary_profile = structure_evolutionary_profile

    def fit_msa_to_structure(self):
        """From a multiple sequence alignment to the reference sequence, align the profile to the Structure sequence.
        Removes the view of all data not present in the structure

        Sets:
            self.msa.sequence_indices (np.ndarray)
        """
        # generate the disordered indices which are positions in reference that are missing in structure
        # disorder_indices = [index - 1 for index in self.disorder]
        if len(self.reference_sequence) != self.msa.query_length:
            raise ValueError(f'The {self.name} reference_sequence ({len(self.reference_sequence)}) and '
                             f'MultipleSequenceAlignment query ({self.msa.query_length}) should be the same length!')
        sequence_indices = self.msa.sequence_indices
        disordered_indices = [index - zero_offset for index in self.disorder]
        self.log.debug(f'Removing disordered indices (reference_sequence indices) from the MultipleSequenceAlignment: '
                       f'{disordered_indices}')  # f'{",".join(map(str, disordered_indices))}')
        # Get all non-zero indices. Then, remove the disordered indices from these indices
        msa_disordered_indices = np.flatnonzero(self.msa.query_indices)[disordered_indices]
        # These selected indices are where the msa is populated, but the structure sequence is missing
        sequence_indices[:, msa_disordered_indices] = False
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

    def psiblast(self, out_dir: AnyStr = os.getcwd(), remote: bool = False):
        """Generate a position specific scoring matrix using PSI-BLAST subprocess

        Args:
            out_dir: Disk location where generated file should be written
            remote: Whether to perform the search through the web. If False, need blast installed locally
        Sets:
            self.pssm_file (str): Name of the file generated by psiblast
        """
        self.pssm_file = os.path.join(out_dir, f'{self.name}.pssm')
        cmd = ['psiblast', '-db', putils.alignmentdb, '-query', self.sequence_file + '.fasta', '-out_ascii_pssm',
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
        # Todo it's CURRENTLY IMPOSSIBLE to use in calculate_profile, CHANGE psiblast lod score parsing
        Sets:
            self.evolutionary_profile (profile_dictionary): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        with open(self.pssm_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_data = line.strip().split()
            if len(line_data) == 44:
                residue_number = int(line_data[0])
                self.evolutionary_profile[residue_number] = copy(aa_counts_alph3)
                for i, aa in enumerate(utils.protein_letters_alph3, 22):  # pose_dict[residue_number], 22):
                    # Get normalized counts for pose_dict
                    self.evolutionary_profile[residue_number][aa] = (int(line_data[i]) / 100.0)
                self.evolutionary_profile[residue_number]['lod'] = {}
                for i, aa in enumerate(utils.protein_letters_alph3, 2):
                    self.evolutionary_profile[residue_number]['lod'][aa] = line_data[i]
                self.evolutionary_profile[residue_number]['type'] = line_data[1]
                self.evolutionary_profile[residue_number]['info'] = float(line_data[42])
                self.evolutionary_profile[residue_number]['weight'] = float(line_data[43])

    def hhblits(self, out_dir: AnyStr = os.getcwd(), threads: int = hhblits_threads,
                return_command: bool = False, **kwargs) -> str | None:
        """Generate a position specific scoring matrix from HHblits using Hidden Markov Models

        Args:
            out_dir: Disk location where generated file should be written
            threads: Number of cpu's to use for the process
            return_command: Whether to simply return the hhblits command
        Sets:
            self.pssm_file (str): Name of the file generated by psiblast
        """
        self.pssm_file = os.path.join(out_dir, f'{self.name}.hmm')
        self.a3m_file = os.path.join(out_dir, f'{self.name}.a3m')
        # self.msa_file = os.path.join(out_path, f'{self.name}.fasta'
        self.msa_file = os.path.join(out_dir, f'{self.name}.sto')  # preferred
        # this location breaks with SymDesign norm so we should modify it Todo clean
        fasta_msa = os.path.join(os.path.dirname(out_dir), 'sequences', f'{self.name}.fasta')
        # todo for higher performance set up https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
        cmd = [putils.hhblits_exe, '-d', putils.uniclustdb, '-i', self.sequence_file,
               '-ohhm', self.pssm_file, '-oa3m', self.a3m_file,  # '-Ofas', self.msa_file,
               '-hide_cons', '-hide_pred', '-hide_dssp', '-E', '1E-06',
               '-v', '1', '-cpu', str(threads)]
        # reformat_msa_cmd1 = [putils.reformat_msa_exe_path, self.a3m_file, self.msa_file, '-num', '-uc']
        # reformat_msa_cmd2 = [putils.reformat_msa_exe_path, self.a3m_file, fasta_msa, '-M', 'first', '-r']
        if return_command:
            return subprocess.list2cmdline(cmd)  # , subprocess.list2cmdline(reformat_msa_cmd)

        self.log.info(f'{self.name} Profile Command: {subprocess.list2cmdline(cmd)}')
        p = subprocess.Popen(cmd)
        p.communicate()
        if p.returncode != 0:
            # temp_file = os.path.join(out_path, f'{self.name}.hold')
            temp_file = Path(out_dir, f'{self.name}.hold')
            temp_file.unlink(missing_ok=True)
            # if os.path.exists(temp_file):  # remove hold file blocking progress
            #     os.remove(temp_file)
            raise RuntimeError(f'Profile generation for {self.name} got stuck')  #
            # raise DesignError(f'Profile generation for {self.name} got stuck. See the error for details -> {p.stderr} '
            #                   f'output -> {p.stdout}')  #
        p = subprocess.Popen([putils.reformat_msa_exe_path, self.a3m_file, self.msa_file, '-num', '-uc'])
        p.communicate()
        p = subprocess.Popen([putils.reformat_msa_exe_path, self.a3m_file, fasta_msa, '-M', 'first', '-r'])
        p.communicate()
        # os.system('rm %s' % self.a3m_file)

    # @handle_errors(errors=(FileNotFoundError,))
    def parse_hhblits_pssm(self, null_background: bool = True, **kwargs):
        """Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order

        Args:
            null_background: Whether to use the profile specific null background
        Sets:
            self.evolutionary_profile (profile_dictionary): Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                     'weight': 0.00}, {...}}
        """
        self.evolutionary_profile = parse_hhblits_pssm(self.pssm_file, null_background=null_background)

    def add_msa(self, msa: str | MultipleSequenceAlignment = None):
        """Add a multiple sequence alignment to the profile. Handles correct sizing of the MSA

        Args:
            msa: The multiple sequence alignment object or file to use for collapse
        """
        if msa is not None:
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

    def collapse_profile(self, msa: AnyStr | MultipleSequenceAlignment = None, **kwargs) -> np.ndarray:
        """Make a profile out of the hydrophobic collapse index (HCI) for each sequence in a multiple sequence alignment

        Takes ~5-10 seconds depending on the size of the msa

        Calculate HCI for each sequence in the MSA (which are different lengths). This is the Hydro Collapse array. For
        each sequence, make a Gap mask (# msa sequences x alignment length) to account for gaps from each individual
        sequence. Apply the mask using the map between the Gap mask and the Hydro Collapse array. Finally, drop the
        columns from the array that are gaps in the reference sequence.

        iter array   -   Gap mask      -       Hydro Collapse array     -     Aligned HCI     - -     Final HCI

        ------------

        iter - - - - - - 0 is gap    - - - -     compute for each     -     account for gaps   -  (drop idx 2)

        it 1 2 3 4  - - 0 | 1 | 2 - - - - - - - - - 0 | 1 | 2 - - - - - - - - 0 | 1 | 2 - - - - - - - 0 | 1 | 3 | ... N

        0 0 1 2 2  - - 1 | 1 | 0 - - - -   - - - - 0.5 0.2 0.5 - -   =   - - 0.5 0.2 0.0 -  ->   - - 0.5 0.2 0.4 ... 0.3

        1 0 0 1 2  - - 0 | 1 | 1 - - - -   - - - - 0.4 0.7 0.4 - -   =   - - 0.0 0.4 0.7 -  ->   - - 0.0 0.4 0.4 ... 0.1

        2 0 0 1 2  - - 0 | 1 | 1 - - - -   - - - - 0.3 0.6 0.3 - -   =   - - 0.0 0.3 0.6 -  ->   - - 0.0 0.3 0.4 ... 0.0

        After iteration cumulative summation, the Hydro Collapse array index is accessed by the iterator. This is then
        multiplied by the gap mask to place np.nan value if there is a 0 index (i.e. a gap) and the Hydro Collapse array
        value otherwise. After, the element at index 2 in the Aligned HCI is dropped from the array when the aligned
        sequence is removed of gaps and only the iterations will be left, essentially giving the HCI for the sequence
        profile in the native context, however adjusted to the specific context of the protein/design sequence at hand

        Args:
            msa: The multiple sequence alignment (file or object) to use for collapse.
                Will use the instance .msa if not provided
        Keyword Args:
            hydrophobicity: int = 'standard' – The hydrophobicity scale to consider. Either 'standard' (FILV),
                'expanded' (FMILYVW), or provide one with 'custom' keyword argument
            custom: mapping[str, float | int] = None – A user defined mapping of amino acid type, hydrophobicity value
                pairs
            alphabet_type: alphabet_types = None – The amino acid alphabet if the sequence consists of integer
                characters
            lower_window: int = 3 – The smallest window used to measure
            upper_window: int = 9 – The largest window used to measure
        Returns:
            Array with shape (number_of_sequences, number_of_residues) containing the hydrophobic collapse values for
                per-residue, per-sequence in the profile. The "query" sequence from the MultipleSequenceAlignment.query
                is located at index 0 on axis=0
        """
        try:  # Todo ensure that the file hasn't changed...
            return self._collapse_profile
        except AttributeError:
            if not self.msa:
                # try:
                self.add_msa(msa)
                # except FileNotFoundError:
                #     raise DesignError(f'Ensure that you have set up the .msa for this {type(self).__name__}. To do this'
                #                       f', either link to the Master Database, call {msa_generation_function}, or pass '
                #                       f'the location of a multiple sequence alignment. '
                #                       f'Supported formats:\n{pretty_format_table(msa_supported_types.items())}')

            # Make the output array. Use one additional length to add np.nan value at the 0 index for gaps
            evolutionary_collapse_np = np.zeros((self.msa.number_of_sequences, self.msa.length + 1))
            evolutionary_collapse_np[:, 0] = np.nan  # np.nan for all missing indices
            for idx, record in enumerate(self.msa.alignment):
                non_gapped_sequence = str(record.seq).replace('-', '')
                evolutionary_collapse_np[idx, 1:len(non_gapped_sequence) + 1] = \
                    metrics.hydrophobic_collapse_index(non_gapped_sequence, **kwargs)
            # Todo this should be possible now metrics.hydrophobic_collapse_index(self.msa.array)

            iterator_np = np.cumsum(self.msa.sequence_indices, axis=1) * self.msa.sequence_indices
            aligned_hci_np = np.take_along_axis(evolutionary_collapse_np, iterator_np, axis=1)
            # Select only the query sequence indices
            # sequence_hci_np = aligned_hci_np[:, self.msa.query_indices]
            # print('aligned_hci_np', aligned_hci_np.shape, aligned_hci_np)
            # print('self.msa.query_indices', self.msa.query_indices.shape, self.msa.query_indices)
            self._collapse_profile = aligned_hci_np[:, self.msa.query_indices]
            # self._collapse_profile = pd.DataFrame(aligned_hci_np[:, self.msa.query_indices],
            #                                       columns=list(range(1, self.msa.query_length + 1)))  # One-indexed
            # summary = pd.concat([sequence_hci_df, pd.concat([sequence_hci_df.mean(), sequence_hci_df.std()], axis=1,
            #                                                 keys=['mean', 'std']).T])

            return self._collapse_profile

    def direct_coupling_analysis(self, msa: AnyStr | MultipleSequenceAlignment = None) -> np.ndarray:
        """Using boltzmann machine direct coupling analysis (bmDCA), score each sequence in an alignment based on the
         statistical energy compared to the learned DCA model

        Args:
            msa: The multiple sequence alignment (file or object) to use for collapse.
                Will use the instance .msa if not provided
        Returns:
            Array with shape (number_of_sequences, length) where the values are the energy for each residue/sequence
                based on direct coupling analysis parameters
        """
        if msa is None:
            msa = self.msa
        else:
            if not self.msa:
                # try:
                self.add_msa(msa)

        if not self.h_fields or not self.j_couplings:
            raise AttributeError('The required data .h_fields and .j_couplings are not available. Add them to the '
                                 f'Entity before {self.direct_coupling_analysis.__name__}')
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
                                name: str = None, out_dir: AnyStr = os.getcwd()) -> AnyStr:
        """Write a sequence to a .fasta file with fasta format and save file location as self.sequence_file.
        '.fasta' is appended if not specified in the name argument

        Args:
            sequence: The sequence to write. Can be the specified sequence or the keywords 'reference' or 'structure'
            file_name: The explicit name of the file
            name: The name of the sequence record. If not provided, the instance name will be used.
                Will be used as the default file_name base name if file_name not provided
            out_dir: The location on disk to output the file. Only used if file_name not explicitly provided
        Returns:
            The name of the output file
        """
        if sequence in sequence_types:  # get the attribute from the instance
            sequence = getattr(self, f'{sequence}_sequence')

        if name is None:
            name = self.name
        if file_name is None:
            file_name = os.path.join(out_dir, name)
            if not file_name.endswith('.fasta'):
                file_name = f'{file_name}.fasta'

        self.sequence_file = write_sequence_to_fasta(sequence=sequence, name=name, file_name=file_name)

        return self.sequence_file

    # def process_fragment_profile(self, **kwargs):
    #     """From self.fragment_map, add the fragment profile to the SequenceProfile
    #
    #     Keyword Args:
    #         keep_extras: bool = True - Whether to keep values for all that are missing data
    #         evo_fill: bool = False - Whether to fill missing positions with evolutionary profile values
    #         alpha: float = 0.5 - The maximum contribution of the fragment profile to use, bounded between (0, 1].
    #             0 means no use of fragments in the .profile, while 1 means only use fragments
    #     Sets:
    #         self.fragment_profile (profile_dictionary)
    #     """
    #     self.simplify_fragment_profile(**kwargs)
    #     # self._calculate_alpha(**kwargs)

    def add_fragments_to_profile(self, fragments: Iterable[fragment_info_type],
                                 alignment_type: alignment_types_literal):
        """Distribute fragment information to self.fragment_map. One-indexed residue dictionary

        Args:
            fragments: The fragment list to assign to the sequence profile with format
                [{'mapped': residue_index1 (int), 'paired': residue_index2 (int), 'cluster': tuple(int, int, int),
                  'match': match_score (float)}]
            alignment_type: Either 'mapped' or 'paired' indicating how the fragment observation was generated relative
                to this Structure. Is it mapped to this Structure or was it paired to it?
        Sets:
            self.fragment_map (dict[int, list[dict[str, str | float]]]):
                {1: [{'source': 'mapped', 'cluster': '1_2_123', 'match': 0.61}, ...], ...}
            self._fragment_profile (profile_dictionary):
                [[[{'A': 0.23, 'C': 0.01, ..., 'count': 12, 'weight': 0.37, 'match': 0.6}, ...], [], ...],
                 [[{}, ...], ...],
                 ...]
                Where the keys (first index) are residue numbers (residue index) and each list holds the fragment
                indices for that residue, where each index in the indices (list in list) can have multiple observations
        """
        if alignment_type not in alignment_types:
            raise ValueError(f'Argument alignment_type must be either "mapped" or "paired" not {alignment_type}')

        # Create self.fragment_map to store information about each fragment observation in the profile
        if not self.fragment_map:
            self.fragment_map = populate_design_dictionary(self.number_of_residues,
                                                           list(range(*self._fragment_db.fragment_range)),
                                                           zero_index=True, dtype='list')
        # for fragment in fragments:
        #     residue_index = fragment[alignment_type] - self.offset_index
        #     for frag_idx in range(*self._fragment_db.fragment_range):  # lower_bound, upper_bound
        #         self.fragment_map[residue_index + frag_idx][frag_idx].append({'source': alignment_type,
        #                                                                       'cluster': fragment['cluster'],
        #                                                                       'match': fragment['match']})
        #
        #     # As of 9/18/22 opting to preload this data
        #     # # Ensure fragment information is retrieved from the fragment_db for the particular clusters
        #     # retrieve_fragments = [fragment['cluster'] for residue_indices in self.fragment_map.values()
        #     #                       for fragments in residue_indices.values() for fragment in fragments]
        #     # self.fragment_db.load_cluster_info(ids=retrieve_fragments)

        if not self._fragment_profile:
            # self._fragment_profile = {residue_index: [[] for _ in range(self._fragment_db.fragment_length)]
            #                           for residue_index in range(self.number_of_residues)}
            self._fragment_profile = [[[] for _ in range(self._fragment_db.fragment_length)]
                                      for _ in range(self.number_of_residues)]

        # Add frequency information to the fragment profile using parsed cluster information. Frequency information is
        # added in a fragment index dependent manner. If multiple fragment indices are present in a single residue, a new
        # observation is created for that fragment index.
        # try:
        for fragment in fragments:
            # Offset the specified fragment index to the overall index in the Structure
            residue_index = fragment[alignment_type] - self.offset_index
            # Retrieve the amino acid frequencies for this fragment cluster, for this alignment side
            aa_freq = getattr(self._fragment_db.info[fragment['cluster']], alignment_type)
            for idx, (frag_idx, frequencies) in enumerate(aa_freq.items()):  # 0, (lower_bound - upper_bound), [freqs]
                # observation = dict(match=fragment['match'], **frequencies)
                self.fragment_map[residue_index + frag_idx][frag_idx].append({'source': alignment_type,
                                                                              'cluster': fragment['cluster'],
                                                                              'match': fragment['match']})
                self._fragment_profile[residue_index + frag_idx][idx].append(dict(match=fragment['match'],
                                                                                  **frequencies))
        # except KeyError:
        #     self.log.critical(f'KeyError at {residue_index + frag_idx} with {frag_idx}. Fragment info is {fragment}.'
        #                       f'len(fragment_map)={len(self.fragment_map)} which are mapped to the index. '
        #                       f'len(self._fragment_profile)={len(self._fragment_profile)}'
        #                       f'offset_index={self.offset_index}')
        #     raise RuntimeError('Need to fix this')

    def simplify_fragment_profile(self, evo_fill: bool = False, **kwargs):  # keep_extras: bool = True,
        """Take a multi-indexed, a multi-observation fragment_profile and flatten to single frequency for each residue.

        Weight the frequency of each observation by the fragment indexed, average observation weight, proportionally
        scaled by the match score between the fragment database and the observed fragment overlap

        From self.fragment_map, add the fragment profile to the SequenceProfile

        Args:
            evo_fill: Whether to fill missing positions with evolutionary profile values
        Keyword Args:
            alpha: float = 0.5 - The maximum contribution of the fragment profile to use, bounded between (0, 1].
                0 means no use of fragments in the .profile, while 1 means only use fragments
        Sets:
            self.fragment_profile (Profile)
                [{'A': 0.23, 'C': 0.01, ..., stats': (1, 0.37)}, {...}, ...]
                list of profile_entry that combines all fragment information at a single residue using a weighted
                average. 'count' is number of fragment observations at each residue, and 'weight' is the total
                fragment weight over the entire residue
        """
        # keep_extras: Whether to keep values for all positions that are missing data
        if not self.fragment_map:  # We need this for _calculate_alpha()
            raise RuntimeError(f"Must {self.add_fragments_to_profile.__name__} before "
                               f"{self.simplify_fragment_profile.__name__}. No fragments were set")
        database_bkgnd_aa_freq = self._fragment_db.aa_frequencies
        # Fragment profile is correct size for indexing all STRUCTURAL residues
        #  self.reference_sequence is not used for this. Instead, self.sequence is used in place since the use
        #  of a disorder indicator that removes any disordered residues from input evolutionary profiles is calculated
        #  on the full reference sequence. This ensures that the profile is the right length of the structure and
        #  captures disorder specific evolutionary signals that could be important in the calculation of profiles
        sequence = self.sequence
        no_design = []
        for residue_index, indexed_observations in enumerate(self._fragment_profile):
            total_fragment_observations = total_fragment_weight = 0
            for index, observations in enumerate(indexed_observations):
                if observations:  # If not, will be an empty list
                    # Sum the weight for each fragment observation
                    total_obs_weight, total_obs_x_match_weight = 0., 0.
                    for observation in observations:
                        total_fragment_observations += 1
                        # observation_weight = observation['stats'][1]
                        observation_weight = observation['weight']
                        total_obs_weight += observation_weight
                        total_obs_x_match_weight += observation_weight * observation['match']
                        # total_match_weight += self._fragment_profile[residue_index][index][obs]['match']

                    # Combine all observations at each index
                    # Check if weights are associated with observations. If not, side chain isn't significant
                    observation_frequencies = {}
                    if total_obs_weight > 0:
                        observation_frequencies.update(**aa_counts_alph3)  # {'A': 0, RC': 0, ...}
                        observation_frequencies['weight'] = total_obs_weight
                        total_fragment_weight += total_obs_weight
                        for observation in observations:
                            # Use pop to access and simultaneously remove from observation for the iteration below
                            # obs_x_match_weight = observation.pop('stats')[1] * observation.pop('match')
                            obs_x_match_weight = observation.pop('weight') * observation.pop('match')
                            # match_weight = self._fragment_profile[residue_index][index][obs]['match']
                            # obs_weight = self._fragment_profile[residue_index][index][obs]['stats'][1]
                            scaled_obs_weight = obs_x_match_weight / total_obs_x_match_weight
                            for aa, frequency in observation.items():
                                # if aa not in ['stats', 'match']:
                                # Multiply OBS and MATCH
                                # modification_weight = obs_x_match_weight / total_obs_x_match_weight
                                # modification_weight = ((obs_weight + match_weight) /  # WHEN SUMMING OBS and MATCH
                                #                        (total_obs_weight + total_match_weight))
                                # modification_weight = (obs_weight / total_obs_weight)
                                # Add all occurrences to summed frequencies list
                                observation_frequencies[aa] += frequency * scaled_obs_weight

                    # Add results to intermediate fragment_profile residue position index
                    self._fragment_profile[residue_index][index] = observation_frequencies

            # Combine all index observations into one residue frequency distribution
            # Set stats over all residue indices and observations
            # residue_frequencies: dict[utils.profile_keys, str | lod_dictionary | float | list[float]] = \
            #     aa_counts_alph3.copy()
            residue_frequencies = {'count': total_fragment_observations,
                                   'weight': total_fragment_weight,
                                   'info': 0.}
            if total_fragment_weight > 0:
                residue_frequencies.update(**aa_counts_alph3)  # {'A': 0, 'R': 0, ...}  # NO -> 'stats': [0, 1]}
                # residue_frequencies['count'] = total_fragment_observations
                # residue_frequencies['weight'] = total_fragment_weight
                for observation_frequencies in self._fragment_profile[residue_index]:
                    if observation_frequencies:  # Not an empty dict
                        # index_weight = observation_frequencies.pop('weight')  # total_obs_weight from above
                        scaled_frag_weight = observation_frequencies.pop('weight') / total_fragment_weight
                        # Add all occurrences to summed frequencies list
                        for aa, frequency in observation_frequencies.items():
                            residue_frequencies[aa] += frequency * scaled_frag_weight

                residue_frequencies['lod'] = get_lod(residue_frequencies, database_bkgnd_aa_freq)
                residue_frequencies['type'] = sequence[residue_index]
            else:  # Add to list for removal from the profile
                no_design.append(residue_index)

            # Add results to final fragment_profile residue position
            self._fragment_profile[residue_index] = residue_frequencies
            # Since we either copy from self.evolutionary_profile or remove, an empty dictionary is fine here
            # If this changes, maybe the == 0 condition needs an aa_counts_alph3.copy() instead of {}

        # if keep_extras:
        if evo_fill and self.evolutionary_profile:
            # If not an empty dictionary, add the corresponding value from evolution
            # For Rosetta, the packer palette is subtractive so the use of an overlapping evolution and
            # null fragment would result in nothing allowed during design...
            for residue_index in no_design:
                self._fragment_profile[residue_index] = self.evolutionary_profile.get(residue_index + zero_offset)
        else:
            # null_profile = self.create_null_profile(zero_index=True)
            # for residue_index in no_design:
            #     self._fragment_profile[residue_index] = null_profile[residue_index]  # blank_residue_entry
            # Add blank entries where each value is np.nan
            for residue_index in no_design:
                new_entry = nan_profile_entry.copy()
                new_entry['type'] = sequence[residue_index]
                self._fragment_profile[residue_index] = new_entry
        # else:  # Remove missing residues from dictionary
        #     for residue_index in no_design:
        #         self._fragment_profile.pop(residue_index)

        # Format into fragment_profile Profile object
        self.fragment_profile = Profile(self._fragment_profile, dtype='fragment')

        self._calculate_alpha(**kwargs)

    def _calculate_alpha(self, alpha: float = default_fragment_contribution, **kwargs):
        """Find fragment contribution to design with a maximum contribution of alpha. Used subsequently to integrate
        fragment profile during combination with evolutionary profile in calculate_profile

        Takes self.fragment_profile (Profile)
            [{'A': 0.23, 'C': 0.01, ..., stats': [1, 0.37]}, {...}, ...]
        self.fragment_map (dict)
            {residue_index: {fragment_index1:
                             [{'cluster': cluster_id1, 'match': float, 'source': Literal['mapped', 'paired']},
                              ...], fragment_index2: [], ...},
             ...}
        and self._fragment_db.statistics (dict)
            {cluster_id1 (str): [[mapped_index_average, paired_index_average,
                                 {max_weight_counts_mapped}, {_paired}],
                                 total_fragment_observations],
             cluster_id2: [], ...,
             frequencies: {'A': 0.11, ...}}
        To identify cluster_id and chain thus returning fragment contribution from the fragment database statistics

        Sets:
            self.alpha: (dict[int, float]) - {0: 0.5, 0: 0.321, ...}

        Args:
            alpha: The maximum contribution of the fragment profile to use, bounded between (0, 1].
                0 means no use of fragments in the .profile, while 1 means only use fragments
        """
        if not self._fragment_db:
            raise AttributeError(f'{self._calculate_alpha.__name__}: No fragment database connected! Cannot calculate '
                                 f'optimal fragment contribution without one')
        if alpha <= 0 or 1 <= alpha:
            raise ValueError(f'{self._calculate_alpha.__name__}: Alpha parameter must be bounded between 0 and 1')
        else:
            self._alpha = alpha

        alignment_type_to_idx = {'mapped': 0, 'paired': 1}  # could move to class, but not used elsewhere
        match_score_average = 0.5  # when fragment pair rmsd equal to the mean cluster rmsd
        bounded_floor = 0.2
        fragment_stats = self._fragment_db.statistics
        # self.alpha.clear()  # Reset the data
        self.alpha = [0 for _ in self.residues]  # Reset the data
        for entry, data in enumerate(self.fragment_profile):
            # Can't use the match count as the fragment index may have no useful residue information
            # Instead use number of fragments with SC interactions count from the frequency map
            frag_count = data.get('count', None)
            if not frag_count:  # When data is missing 'stats', or 'stats'[0] is 0
                # self.alpha[entry] = 0.
                continue  # Move on, this isn't a fragment observation, or we have no observed fragments

            # Match score 'match' is bounded between [0.2, 1]
            match_sum = sum(observation['match'] for index_observations in self.fragment_map[entry].values()
                            for observation in index_observations)
            # if count == 0:
            #     # ensure that match modifier is 0 so self.alpha[entry] is 0, as there is no fragment information here!
            #     count = match_sum * 5  # makes the match average = 0.5

            match_average = match_sum / float(frag_count)
            # Find the match modifier which spans from 0 to 1
            if match_average < match_score_average:
                match_modifier = (match_average-bounded_floor) / (match_score_average-bounded_floor)
            else:  # Set modifier to 1, the maximum bound
                match_modifier = 1

            # Find the total contribution from a typical fragment of this type
            contribution_total = sum(fragment_stats[f'{observation["cluster"][0]}_{observation["cluster"][1]}_0']
                                     [0][alignment_type_to_idx[observation['source']]]
                                     for index_observations in self.fragment_map[entry].values()
                                     for observation in index_observations)

            # Get the average contribution of each fragment type
            stats_average = contribution_total / frag_count
            # Get entry average fragment weight. total weight for issm entry / count
            frag_weight_average = data.get('weight') / match_sum

            # Modify alpha proportionally to cluster average weight and match_modifier
            # If design frag weight is less than db cluster average weight
            if frag_weight_average < stats_average:
                # self.alpha.append(self._alpha * match_modifier * (frag_weight_average/stats_average))
                self.alpha[entry] = self._alpha * match_modifier * (frag_weight_average/stats_average)
            else:
                # self.alpha.append(self._alpha * match_modifier)
                self.alpha[entry] = self._alpha * match_modifier

    def calculate_profile(self, favor_fragments: bool = False, boltzmann: bool = True, **kwargs):
        """Combine weights for profile PSSM and fragment SSM using fragment significance value to determine overlap

        Using self.evolutionary_profile
            (profile_dictionary): HHblits - {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
                                                 'type': 'W', 'info': 0.00, 'weight': 0.00}, {...}}
                                  PSIBLAST - {1: {'A': 0.13, 'R': 0.12, ..., 'lod': {'A': -5, 'R': 2, ...},
                                                  'type': 'W', 'info': 3.20, 'weight': 0.73}, {...}}
        self.fragment_profile
            (dict[int, dict[str, float | list[float]]]):
                {48: {'A': 0.167, 'D': 0.028, 'E': 0.056, ..., 'count': 4, 'weight': 0.274}, 50: {...}, ...}
        self.alpha
            (list[float]): [0., 0., 0., 0.5, 0.321, ...]
        Args:
            favor_fragments: Whether to favor fragment profile in the lod score of the resulting profile
                Currently this routine is only used for Rosetta designs where the fragments should be favored by a
                particular weighting scheme. By default, the boltzmann weighting scheme is applied
            boltzmann: Whether to weight the fragment profile by a Boltzmann probability scaling using the formula
                lods = exp(lods[i]/kT)/Z, where Z = sum(exp(lods[i]/kT)), and kT is 1 by default.
                If False, residues are weighted by the residue local maximum lod score in a linear fashion
                All lods are scaled to a maximum provided in the Rosetta REF2015 per residue reference weight.
        Sets:
            self.profile: (profile_dictionary)
                {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
                     'type': 'W', 'info': 0.00, 'weight': 0.00}, ...}, ...}
        """
        if self._alpha == 0:  # We get a division error
            self.log.debug(f'{self.calculate_profile.__name__}: _alpha set with 1e-5 tolerance due to 0 value')
            self._alpha = 0.000001

        # Copy the evolutionary profile to self.profile (structure specific scoring matrix)
        self.profile = deepcopy(self.evolutionary_profile)
        if sum(self.alpha) == 0:  # No fragments to combine
            return

        # Combine fragment and evolutionary probability profile according to alpha parameter
        fragment_profile = self.fragment_profile
        # log_string = []
        for entry, weight in enumerate(self.alpha):
            # Weight will be 0 if the fragment_profile is empty
            if weight > 0:
                # log_string.append(f'Residue {entry + 1:5d}: {weight * 100:.0f}% fragment weight')
                inverse_weight = 1 - weight
                frag_profile_entry = fragment_profile[entry]
                _profile_entry = self.profile[entry + zero_offset]
                _profile_entry.update({aa: weight*frag_profile_entry[aa] + inverse_weight*_profile_entry[aa]
                                       for aa in utils.protein_letters_alph3})
        # if log_string:
        #     # self.log.info(f'At {self.name}, combined evolutionary and fragment profiles into Design Profile with:'
        #     #               f'\n\t%s' % '\n\t'.join(log_string))
        #     pass

        if favor_fragments:
            boltzman_energy = 1
            favor_seqprofile_score_modifier = 0.2 * utils.rosetta.reference_average_residue_weight
            database_bkgnd_aa_freq = self._fragment_db.aa_frequencies

            null_residue = get_lod(database_bkgnd_aa_freq, database_bkgnd_aa_freq, as_int=False)
            # This was needed in the case of domain errors with lod
            # null_residue = {aa: float(frequency) for aa, frequency in null_residue.items()}

            # Set all profile entries to a null entry first
            for entry, data in self.profile.items():
                data['lod'] = null_residue  # Caution, all reference same object

            for entry, data in self.profile.items():
                data['lod'] = get_lod(fragment_profile[entry - zero_offset], database_bkgnd_aa_freq, as_int=False)
                # Adjust scores with particular weighting scheme
                partition = 0.
                for aa, value in data['lod'].items():
                    if boltzmann:  # Boltzmann scaling, sum for the partition function
                        value = exp(value / boltzman_energy)
                        partition += value
                    else:  # if value < 0:
                        # With linear scaling, remove any lod penalty
                        value = max(0, value)

                    data['lod'][aa] = value

                # Find the maximum/residue (local) lod score
                max_lod = max(data['lod'].values())
                # Takes the percent of max alpha for each entry multiplied by the standard residue scaling factor
                modified_entry_alpha = (self.alpha[entry - zero_offset]/self._alpha) * favor_seqprofile_score_modifier
                if boltzmann:
                    # lods = e ** odds[i]/Z, Z = sum(exp(odds[i]/kT))
                    modifier = partition
                    modified_entry_alpha /= (max_lod / partition)
                else:
                    modifier = max_lod

                # Weight the final lod score by the modifier and the scaling factor for the chosen method
                data['lod'] = {aa: value / modifier * modified_entry_alpha for aa, value in data['lod'].items()}
                # Get percent total (boltzman) or percent max (linear) and scale by alpha score modifier

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
        cluster_freq_d = {cluster: format_frequencies(fragment_source[cluster]['freq'])
                          for cluster in fragment_source}  # orange mapped to cluster tag
        cluster_freq_twin_d = {cluster: format_frequencies(fragment_source[cluster]['freq'], flip=True)
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
        self.add_fragments_to_profile(fragments=fragment_source, alignment_type=alignment_type)
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


dtype_literals = Literal['list', 'set', 'tuple', 'float', 'int']


def populate_design_dictionary(n: int, alphabet: Sequence, zero_index: bool = False, dtype: dtype_literals = 'int') \
        -> dict[int, dict[int | str, Any]]:
    """Return a dictionary with n elements, each integer key containing another dictionary with the items in
    alphabet as keys. By default, one-indexed, and data inside the alphabet dictionary is 0 (integer).
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
    offset = 0 if zero_index else zero_offset

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


def get_lod(frequencies: dict[utils.protein_letters_literal, float],
            background: dict[utils.protein_letters_literal, float], as_int: bool = True) -> dict[str, int | float]:
    """Get the log of the odds that an amino acid is in a frequency distribution compared to a background frequency

    Args:
        frequencies: {'A': 0.11, 'C': 0.01, 'D': 0.034, ...}
        background: {'A': 0.10, 'C': 0.02, 'D': 0.04, ...}
        as_int: Whether to round the lod values to an integer
    Returns:
         The log of odds for each amino acid type {'A': 2, 'C': -9, 'D': -1, ...}
    """
    lods = {}
    for aa, freq in frequencies.items():
        try:  # Todo why is this 2. * the log2?
            lods[aa] = float(2. * log2(freq / background[aa]))  # + 0.0
        except ValueError:  # math domain error
            lods[aa] = -9
        except KeyError:
            if aa in utils.protein_letters_alph1:
                raise KeyError(f'{aa} was not in the background frequencies: {", ".join(background)}')
            else:  # we shouldn't worry about a missing value if it's not an amino acid
                continue
        except ZeroDivisionError:  # background is 0. We may need a pseudocount...
            raise ZeroDivisionError(f'{aa} has a background frequency of 0. Consider adding a pseudocount')
        # if lods[aa] < -9:
        #     lods[aa] = -9
        # elif round_lod:
        #     lods[aa] = round(lods[aa])

    if as_int:
        return {aa: (int(value) if value >= -9 else -9) for aa, value in lods.items()}
    else:  # ensure that -9 is the lowest value (formatting issues if 2 digits)
        return {aa: (value if value >= -9 else -9) for aa, value in lods.items()}


def format_frequencies(frequency_list: list, flip: bool = False) -> dict[str, dict[str, float]]:
    """Format list of paired frequency data into parsable paired format

    Args:
        frequency_list: [(('D', 'A'), 0.0822), (('D', 'V'), 0.0685), ...]
        flip: Whether to invert the mapping of internal tuple
    Returns:
        {'A': {'S': 0.02, 'T': 0.12}, ...}
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
    # Get all mutations from the alignment of sequence1 and sequence2
    mutations = generate_mutations(sequence1, sequence2, blanks=True, return_all=True)
    # Get only those indices where there is an aligned aa on the opposite chain
    sequence1_indices, sequence2_indices = [], []
    to_idx, from_idx = 0, 0
    # sequence1 'from' is fixed, sequence2 'to' is moving
    for mutation in mutations.values():
        if mutation['from'] == '-':  # increment to_idx/fixed_idx
            to_idx += 1
        elif mutation['to'] == '-':  # increment from_idx/moving_idx
            from_idx += 1
        else:
            sequence1_indices.append(from_idx)
            sequence2_indices.append(to_idx)
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


def get_cluster_dicts(db=putils.biological_interfaces, id_list=None):  # TODO Rename
    """Generate an interface specific scoring matrix from the fragment library

    Args:
    Keyword Args:
        info_db=putils.biological_fragmentDB
        id_list=None: [1_2_24, ...]
    Returns:
         cluster_dict: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    """
    info_db = putils.frag_directory[db]
    if id_list is None:
        directory_list = sdutils.get_base_root_paths_recursively(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        cluster_dict[os.path.basename(cluster)] = sdutils.unpickle(filename)

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
        return {residue - zero_offset: dictionary[residue] for residue in dictionary}
    else:
        return {residue + zero_offset: dictionary[residue] for residue in dictionary}


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
        cluster_map (dict): {48: {'source': 'mapped', 'cluster': [(-2, 1_1_54), ...]}, ...}
            Where the key is the 0 indexed residue id
    """
    cluster_map = {}
    for cluster in residue_cluster_dict:
        for pair in range(len(residue_cluster_dict[cluster])):
            for i, residue_atom in enumerate(residue_cluster_dict[cluster][pair]):
                # for each residue in map add the same cluster to the range of fragment residue numbers
                residue_num = residue_atom.residue_number - zero_offset  # zero index
                for j in range(*frag_range):
                    if residue_num + j not in cluster_map:
                        if i == 0:
                            cluster_map[residue_num + j] = {'source': 'mapped', 'cluster': []}
                        else:
                            cluster_map[residue_num + j] = {'source': 'paired', 'cluster': []}
                    cluster_map[residue_num + j]['cluster'].append((j, cluster))

    return cluster_map


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
#     cmd = ['psiblast', '-db', putils.alignmentdb, '-query', query + '.fasta', '-out_ascii_pssm', outfile_name,
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
# def hhblits(query, cores=hhblits_threads, outpath=os.getcwd()):
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
#     cmd = [putils.hhblits, '-d', putils.uniclustdb, '-i', query, '-ohhm', outfile_name, '-v', '1', '-cpu', str(cores)]
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
                dict(zip(utils.protein_letters_alph3,
                         [x / 100. for x in map(int, line_data[22:len(utils.protein_letters_alph3) + 22])]))
            # pose_dict[residue_number] = aa_counts_alph3.copy()
            # for i, aa in enumerate(utils.protein_letters_alph3, 22):
            #     # Get normalized counts for pose_dict
            #     pose_dict[residue_number][aa] = int(line_data[i]) / 100.

            # for i, aa in enumerate(utils.protein_letters_alph3, 2):
            #     pose_dict[residue_number]['lod'][aa] = line_data[i]
            pose_dict[residue_number]['lod'] = \
                dict(zip(utils.protein_letters_alph3, line_data[2:len(utils.protein_letters_alph3) + 2]))
            pose_dict[residue_number]['type'] = line_data[1]
            pose_dict[residue_number]['info'] = float(line_data[42])
            pose_dict[residue_number]['weight'] = float(line_data[43])

    return pose_dict


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
def parse_hhblits_pssm(file: AnyStr, null_background: bool = True, **kwargs) -> profile_dictionary:
    """Take contents of protein.hmm, parse file and input into pose_dict. File is Single AA code alphabetical order

    Args:
        file: The file to parse, typically with the extension '.hmm'
        null_background: Whether to use the null background for the specific protein
    Returns:
        Dictionary containing residue indexed profile information
            Ex: {1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...}, 'type': 'W', 'info': 0.00,
                     'weight': 0.00}, {...}}
    """
    null_bg = latest_uniclust_background_frequencies

    def to_freq(value: str) -> float:
        if value == '*':  # When frequency is zero
            return 0.0001
        else:
            # Equation: value = -1000 * log_2(frequency)
            return 2 ** (-int(value) / 1000)

    with open(file, 'r') as f:
        lines = f.readlines()

    evolutionary_profile = {}
    dummy = 0.
    read = False
    for line in lines:
        if not read:
            if line[0:1] == '#':
                read = True
        else:
            if line[0:4] == 'NULL':
                if null_background:  # Use the provided null background from the profile search
                    null, *background_values = line.strip().split()
                    # null = 'NULL', background_values = list[str] ['3706', '5728', ...]
                    null_bg = {aa: to_freq(value) for value, aa in zip(background_values, utils.protein_letters_alph3)}

            if len(line.split()) == 23:
                residue_type, residue_number, *position_values = line.strip().split()
                aa_freqs = {aa: to_freq(value) for value, aa in zip(position_values, utils.protein_letters_alph1)}

                evolutionary_profile[int(residue_number)] = \
                    dict(lod=get_lod(aa_freqs, null_bg), type=residue_type, info=dummy, weight=dummy, **aa_freqs)

    return evolutionary_profile

    # with open(file, 'r') as f:
    #     lines = f.readlines()
    #
    # pose_dict = {}
    # read = False
    # for line in lines:
    #     if not read:
    #         if line[0:1] == '#':
    #             read = True
    #     else:
    #         if line[0:4] == 'NULL':
    #             if null_background:
    #                 # use the provided null background from the profile search
    #                 background = line.strip().split()
    #                 null_bg = {i: {} for i in utils.protein_letters_alph3}
    #                 for i, aa in enumerate(utils.protein_letters_alph3, 1):
    #                     null_bg[aa] = to_freq(background[i])
    #
    #         if len(line.split()) == 23:
    #             items = line.strip().split()
    #             residue_number = int(items[1])
    #             pose_dict[residue_number] = {}
    #             for i, aa in enumerate(utils.protein_letters_alph1, 2):
    #                 pose_dict[residue_number][aa] = to_freq(items[i])
    #             pose_dict[residue_number]['lod'] = get_lod(pose_dict[residue_number], null_bg)
    #             pose_dict[residue_number]['type'] = items[0]
    #             pose_dict[residue_number]['info'] = dummy
    #             pose_dict[residue_number]['weight'] = dummy
    #
    # return pose_dict


# def make_pssm_file(pssm_dict: profile_dictionary, name: str, out_dir: AnyStr = os.getcwd()):
#     """Create a PSI-BLAST format PSSM file from a PSSM dictionary
#
#     Args:
#         pssm_dict: A pssm dictionary which has the fields 'A', 'C', (all aa's), 'lod', 'type', 'info', 'weight'
#         name: The name of the file
#         out_dir: A specific location to write the .pssm file to
#     Returns:
#         The disk location of newly created .pssm file
#     """
#     lod_freq, counts_freq = False, False
#     separation_string1, separation_string2 = 3, 3
#     if type(pssm_dict[0]['lod']['A']) == float:
#         lod_freq = True
#         separation_string1 = 4
#     if type(pssm_dict[0]['A']) == float:
#         counts_freq = True
#
#     header = '\n\n            ' + (' ' * separation_string1).join(aa for aa in utils.protein_letters_alph3) \
#              + ' ' * separation_string1 + (' ' * separation_string2).join(aa for aa in utils.protein_letters_alph3) + '\n'
#     footer = ''
#     out_file = os.path.join(out_dir, name)  # + '.pssm'
#     with open(out_file, 'w') as f:
#         f.write(header)
#         for res in pssm_dict:
#             aa_type = pssm_dict[res]['type']
#             lod_string = ''
#             if lod_freq:
#                 for aa in utils.protein_letters_alph3:  # ensure alpha_3_aa_list for PSSM format
#                     lod_string += '{:>4.2f} '.format(pssm_dict[res]['lod'][aa])
#             else:
#                 for aa in utils.protein_letters_alph3:  # ensure alpha_3_aa_list for PSSM format
#                     lod_string += '{:>3d} '.format(pssm_dict[res]['lod'][aa])
#             counts_string = ''
#             if counts_freq:
#                 for aa in utils.protein_letters_alph3:  # ensure alpha_3_aa_list for PSSM format
#                     counts_string += '{:>3.0f} '.format(floor(pssm_dict[res][aa] * 100))
#             else:
#                 for aa in utils.protein_letters_alph3:  # ensure alpha_3_aa_list for PSSM format
#                     counts_string += '{:>3d} '.format(pssm_dict[res][aa])
#             info = pssm_dict[res]['info']
#             weight = pssm_dict[res]['weight']
#             line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + zero_offset, aa_type, lod_string,
#                                                                              counts_string, round(info, 4),
#                                                                              round(weight, 4))
#             f.write(line)
#         f.write(footer)
#
#     return out_file


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
        for aa in utils.protein_letters_alph3:
            if pssm[residue]['lod'][aa] > max_lod:
                max_lod = pssm[residue]['lod'][aa]
                max_res = aa
        consensus_identities[residue + zero_offset] = max_res

    return consensus_identities


def sequence_difference(seq1: Sequence, seq2: Sequence, d: dict = None, matrix: str = 'BLOSUM62') -> float:  # TODO AMS
    """Returns the sequence difference between two sequence iterators

    Args:
        seq1: Either an iterable with residue type as array, or key, with residue type as d[seq1][residue]['type']
        seq2: Either an iterable with residue type as array, or key, with residue type as d[seq2][residue]['type']
        d: The dictionary to look up seq1 and seq2 if they are keys in d
        matrix: The type of matrix to score the sequence differences on
    Returns:
        The computed sequence difference between seq1 and seq2
    """
    if d is not None:
        # seq1 = d[seq1]
        # seq2 = d[seq2]
        # for residue in d[seq1]:
        #     s.append((d[seq1][residue]['type'], d[seq2][residue]['type']))
        pairs = [(d[seq1][residue]['type'], d[seq2][residue]['type']) for residue in d[seq1]]
    else:
        pairs = zip(seq1, seq2)

    matrix_ = substitution_matrices.load(matrix)
    scores = [matrix_.get((letter1, letter2), (letter2, letter1)) for letter1, letter2 in pairs]

    return sum(scores)


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
        offset = zero_offset

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
                    if sequence[test_orf_index + mutation_index - zero_offset] == mutation['from']:
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
                        orf_start_idx = closest_met  # + zero_offset # change to one-index
                    break
            break

    return orf_start_idx


Alignment = namedtuple('Alignment', 'seqA, seqB, score, start, end')


def generate_alignment(seq1: Sequence[str], seq2: Sequence[str], matrix: str = 'BLOSUM62', local: bool = False,
                       top_alignment: bool = True) -> Alignment | list[Alignment]:
    """Use Biopython's pairwise2 to generate a sequence alignment

    Args:
        seq1: The first sequence to align
        seq2: The second sequence to align
        matrix: The matrix used to compare character similarities
        local: Whether to run a local alignment. Only use for generally similar sequences!
        top_alignment: Only include the highest scoring alignment
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
    # logger.debug(f'Generating sequence alignment between:\n{seq1}\n\tAND:\n{seq2}')
    # Create sequence alignment
    align = getattr(pairwise2.align, f'{_type}ds')(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty,
                                                   one_alignment_only=top_alignment)
    logger.debug(f'Generated alignment:\n{pairwise2.format_alignment(*align[0])}')

    return align[0] if top_alignment else align


mutation_entry = Type[dict[Literal['to', 'from'], utils.protein_letters_alph3_gapped_literal]]
"""Mapping of a reference sequence amino acid type, 'to', and the resulting sequence amino acid type, 'from'"""
mutation_dictionary = dict[int, mutation_entry]
"""The mapping of a residue number to a mutation entry containing the reference, 'to', and sequence, 'from', amino acid 
type
"""
sequence_dictionary = dict[int, utils.protein_letters_alph3_gapped_literal]
"""The mapping of a residue number to the corresponding amino acid type"""


def generate_mutations(reference: Sequence, query: Sequence, offset: bool = True, blanks: bool = False,
                       remove_termini: bool = True, remove_query_gaps: bool = True, only_gaps: bool = False,
                       zero_index: bool = False,
                       return_all: bool = False, return_to: bool = False, return_from: bool = False) \
        -> mutation_dictionary | sequence_dictionary:
    """Create mutation data in a typical A5K format. One-indexed dictionary keys with the index matching the reference
     sequence index. Sequence mutations accessed by "from" and "to" keys. By default, only mutated positions are
     returned and all gaped sequences are excluded

    For PDB comparison, reference should be expression sequence (SEQRES), query should be atomic sequence (ATOM)

    Args:
        reference: Reference sequence to align mutations against. Character values are returned to the "from" key
        query: Query sequence. Character values are returned to the "to" key
        offset: Whether sequences are different lengths. Will create an alignment of the two sequences
        blanks: Include all gaped indices, i.e. outside the reference sequence or missing characters in the sequence
        remove_termini: Remove indices that are outside the reference sequence boundaries
        remove_query_gaps: Remove indices where there are gaps present in the query sequence
        only_gaps: Only include reference indices that are missing query residues. All "to" values will be a gap "-"
        zero_index: Whether to return the indices zero-indexed (like python) or one-indexed
        return_all: Whether to return all the indices and there corresponding mutational data
        return_to: Whether to return only the 'to' amino acid type
        return_from: Whether to return only the 'from' amino acid type
    Returns:
        Mutation index to mutations in the format of {1: {'from': 'A', 'to': 'K'}, ...}
            unless return_to or return_from is True, then {1: 'K', ...}
    """
    if offset:
        align_seq_1, align_seq_2, *_ = generate_alignment(reference, query)
    else:
        align_seq_1, align_seq_2 = reference, query

    idx_offset = 0 if zero_index else zero_offset

    # Get the first matching index of the reference sequence
    starting_idx_of_seq1 = align_seq_1.find(reference[0])
    # Ensure iteration sequence1/reference starts at idx 1       v
    sequence_iterator = enumerate(zip(align_seq_1, align_seq_2), -starting_idx_of_seq1 + idx_offset)
    # Extract differences from the alignment
    if return_all:
        mutations = {idx: {'from': seq1, 'to': seq2} for idx, (seq1, seq2) in sequence_iterator}
    else:
        mutations = {idx: {'from': seq1, 'to': seq2} for idx, (seq1, seq2) in sequence_iterator if seq1 != seq2}

    # Find last index of reference
    ending_index_of_seq1 = starting_idx_of_seq1 + align_seq_1.rfind(reference[-1])
    remove_mutation_list = []
    if only_gaps:  # remove the actual mutations, keep internal and external gap indices and the reference sequence
        blanks = True
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if idx_offset < entry <= ending_index_of_seq1 and mutation['to'] != '-'])
    if blanks:  # leave all types of blanks, otherwise check for each requested type
        remove_termini, remove_query_gaps = False, False

    if remove_termini:  # remove indices outside of sequence 1
        remove_mutation_list.extend([entry for entry in mutations
                                     if entry < idx_offset or ending_index_of_seq1 < entry])

    if remove_query_gaps:  # remove indices where sequence 2 is gaped
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if 0 < entry <= ending_index_of_seq1 and mutation['to'] == '-'])
    for entry in remove_mutation_list:
        mutations.pop(entry, None)

    if return_to:
        mutations = {idx: _mutation_dictionary['to'] for idx, _mutation_dictionary in mutations.items()}
    elif return_from:
        mutations = {idx: _mutation_dictionary['from'] for idx, _mutation_dictionary in mutations.items()}

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


def simplify_mutation_dict(mutations: dict[str, mutation_dictionary], to: bool = True) \
        -> dict[str, mutation_dictionary]:
    """Simplify mutation dictionary to 'to'/'from' AA key

    Args:
        mutations: Ex: {alias: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
        to: Whether to simplify with the 'to' AA key (True) or the 'from' AA key (False)
    Returns:
        The simplified mutation dictionary. Ex: {alias: {mutation_index: 'K', ...}, ...}
    """
    simplification = 'to' if to else 'from'

    for alias, indexed_mutations in mutations.items():
        for index, mutation in indexed_mutations.items():
            mutations[alias][index] = mutation[simplification]

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
        final_resi = residue + zero_offset
        weaved_dict[final_resi] = {}
        for aa in sorted_freq[residue]:
            weaved_dict[final_resi][aa] = round(mut_prob[residue][aa], 3)
        weaved_dict[final_resi]['jsd'] = resi_divergence[residue]
        weaved_dict[final_resi]['int_jsd'] = int_divergence[residue]
        weaved_dict[final_resi]['des_jsd'] = des_divergence[residue]

    return weaved_dict


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
        for i in range(len(score_dict) + zero_offset):
            s, number_terms = 0, 0
            if i <= window_len:
                for j in range(1, i + window_len + zero_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            elif i + window_len > len(score_dict):
                for j in range(i - window_len, len(score_dict) + zero_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            else:
                for j in range(i - window_len, i + window_len + zero_offset):
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
        raise RuntimeError(f'{multi_chain_alignment.__name__} - No sequences were found!')


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
        raise RuntimeError(f"The reference sequence and mutated_sequences have different chains! Chain {chain} "
                           "isn't in the reference")
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


def generate_mutations_from_reference(reference: Sequence[str], sequences: dict[str, Sequence[str]], **kwargs) -> \
        dict[str, mutation_dictionary | sequence_dictionary]:
    """Generate mutation data from multiple alias mapped sequence dictionaries with regard to a single reference.

    Defaults to returning only mutations (return_all=False) and forgoes any sequence alignment (offset=False)

    Args:
        reference: The reference sequence to align each sequence against.
            Character values are returned to the "from" key
        sequences: The template sequences to align, i.e. {alias: sequence, ...}.
            Character values are returned to the "to" key
    Keyword Args:
        offset: (bool) = True - Whether sequences are different lengths. Will create an alignment of the two sequences
        blanks: (bool) = False - Include all gaped indices, i.e. outside the reference sequence or missing characters
            in the sequence
        remove_termini: (bool) = True - Remove indices that are outside the reference sequence boundaries
        remove_query_gaps: (bool) = True - Remove indices where there are gaps present in the query sequence
        only_gaps: (bool) = False - Only include reference indices that are missing query residues.
            All "to" values will be a gap "-"
        zero_index: (bool) = False - Whether to return the indices zero-indexed (like python Sequence) or one-indexed
        return_all: (bool) = False - Whether to return all the indices and there corresponding mutational data
        return_to: (bool) = False - Whether to return only the "to" amino acid type
        return_from: (bool) = False - Whether to return only the "from" amino acid type
    Returns:
        {alias: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...} unless return_to or return_from is True, then
            {alias: {mutation_index: 'K', ...}, ...}
    """
    # offset_value = kwargs.get('offset')
    kwargs['offset'] = (kwargs.get('offset') or False)  # Default to False if there was no argument passed
    mutations = {alias: generate_mutations(reference, sequence, **kwargs)  # offset=False,
                 for alias, sequence in sequences.items()}

    # Add the reference sequence to mutation data
    if kwargs.get('return_to') or kwargs.get('return_from'):
        mutations[putils.reference_name] = dict(enumerate(reference, 0 if kwargs.get('zero_index') else 1))
    else:
        mutations[putils.reference_name] = \
            {sequence_idx: {'from': aa, 'to': aa}
             for sequence_idx, aa in enumerate(reference, 0 if kwargs.get('zero_index') else 1)}

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


# @sdutils.handle_errors(errors=(FileNotFoundError,))
def read_fasta_file(file_name: AnyStr, **kwargs) -> Iterable[SeqRecord]:
    """Opens a fasta file and return a parser object to load the sequences to SeqRecords

    Args:
        file_name: The location of the file on disk
    Returns:
        An iterator of the sequences in the file [record1, record2, ...]
    """
    return SeqIO.parse(file_name, 'fasta')


# @sdutils.handle_errors(errors=(FileNotFoundError,))
def read_sequence_file(file_name: AnyStr, **kwargs) -> Iterable[SeqRecord]:
    """Opens a fasta file and return a parser object to load the sequences to SeqRecords

    Args:
        file_name: The location of the file on disk
    Returns:
        An iterator of the sequences in the file [record1, record2, ...]
    """
    raise NotImplementedError()
    return SeqIO.parse(file_name, 'csv')


@sdutils.handle_errors(errors=(FileNotFoundError,))
def read_alignment(file_name: AnyStr, alignment_type: str = 'fasta', **kwargs) -> MultipleSeqAlignment:
    """Open an alignment file and parse the alignment to a Biopython MultipleSeqAlignment

    Args:
        file_name: The location of the file on disk
        alignment_type: The type of file that the alignment is stored in. Used for parsing
    Returns:
        The parsed alignment
    """
    return AlignIO.read(file_name, alignment_type)


def write_fasta(sequence_records: Iterable[SeqRecord], file_name: AnyStr = None, name: str = None,
                out_dir: AnyStr = os.getcwd()) -> AnyStr:
    """Write an iterator of SeqRecords to a .fasta file with fasta format. '.fasta' is appended if not specified in name

    Args:
        sequence_records: The sequences to write. Should be Biopython SeqRecord format
        file_name: The explicit name of the file
        name: The name of the file to output
        out_dir: The location on disk to output file
    Returns:
        The name of the output file
    """
    if file_name is None:
        file_name = os.path.join(out_dir, name)
        if not file_name.endswith('.fasta'):
            file_name = f'{file_name}.fasta'

    SeqIO.write(sequence_records, file_name, 'fasta')

    return file_name


def write_sequence_to_fasta(sequence: str, file_name: AnyStr, name: str, out_dir: AnyStr = os.getcwd()) -> AnyStr:
    """Write an iterator of SeqRecords to a .fasta file with fasta format. '.fasta' is appended if not specified in name

    Args:
        sequence: The sequence to write
        name: The name of the sequence. Will be used as the default file_name base name if file_name not provided
        file_name: The explicit name of the file
        out_dir: The location on disk to output the file. Only used if file_name not explicitly provided
    Returns:
        The name of the output file
    """
    if file_name is None:
        file_name = os.path.join(out_dir, name)
        if not file_name.endswith('.fasta'):
            file_name = f'{file_name}.fasta'

    with open(file_name, 'w') as outfile:
        outfile.write(f'>{name}\n{sequence}\n')

    return file_name


def concatenate_fasta_files(file_names: Iterable[AnyStr], out_path: str = 'concatenated_fasta') -> AnyStr:
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
    return write_fasta(seq_records, out_dir=out_path)


def write_sequences(sequences: Sequence | dict[str, Sequence], names: Sequence = None,
                    out_path: AnyStr = os.getcwd(), file_name: AnyStr = None, csv: bool = False) -> AnyStr:
    """Write a fasta file from sequence(s)

    Args:
        sequences: If a list, can be list of tuples(name, sequence), or list[sequence] where names contain the
            corresponding sequence names. If dict, uses key as name, value as sequence. If str, treats as the sequence
        names: The name or names of the sequence record(s). If a single name, will be used as the default file_name
            base name if file_name not provided. Otherwise, will be used iteratively
        out_path: The location on disk to output file
        file_name: The explicit name of the file
        csv: Whether the file should be written as a .csv. Default is .fasta
    Returns:
        The name of the output file
    """
    if file_name is None:
        if isinstance(names, str):  # Not an iterable
            file_name = os.path.join(out_path, names)
        else:
            raise ValueError(f'Must provide argument "file_name" or "names" as a str to {write_sequences.__name__}')

    if csv:
        start, sep = '', ','
        extension = '.csv'
    else:
        start, sep = '>', '\n'
        extension = '.fasta'

    provided_extension = os.path.splitext(file_name)[-1]
    if not file_name.endswith(extension) or provided_extension != extension:
        file_name = f'{os.path.splitext(file_name)[0]}{extension}'

    def data_dump():
        return f'names={names}, sequences={sequences}, extension={extension}, out_path={out_path}, ' \
               f'file_name={file_name}'

    with open(file_name, 'a') as outfile:
        if isinstance(sequences, list):
            if isinstance(sequences[0], tuple):  # where seq[0] is name, seq[1] is seq
                formatted_sequence_gen = (f'{start}{name}{sep}{seq}' for name, seq, *_ in sequences)
            elif isinstance(names, Sequence):
                if isinstance(sequences[0], str):
                    formatted_sequence_gen = (f'{start}{name}{sep}{seq}' for name, seq in zip(names, sequences))
                elif isinstance(sequences[0], Sequence):
                    formatted_sequence_gen = (f'{start}{name}{sep}{"".join(seq)}' for name, seq in zip(names, sequences))
                else:
                    raise TypeError(f"{write_sequences.__name__} Can't parse data to make fasta\n{data_dump()}")
            # elif isinstance(sequences[0], list):  # where interior list is alphabet (AA or DNA)
            #     for idx, seq in enumerate(sequences):
            #         outfile.write(f'>{name}_{idx}\n')  # write header
            #         # Check if alphabet is 3 letter protein
            #         outfile.write(f'{" ".join(seq)}\n' if len(seq[0]) == 3 else f'{"".join(seq)}\n')
            # elif isinstance(sequences[0], str):  # likely 3 aa format...
            #     outfile.write(f'>{name}\n{" ".join(sequences)}\n')
            else:
                raise TypeError(f"{write_sequences.__name__} Can't parse data to make fasta\n{data_dump()}")
        elif isinstance(sequences, dict):
            formatted_sequence_gen = (f'{start}{name}{sep}{"".join(seq)}' for name, seq in sequences.items())
        elif isinstance(names, str):  # assume sequences is a str or tuple
            formatted_sequence_gen = (f'{start}{names}{sep}{"".join(sequences)}\n',)
        else:
            raise TypeError(f"{write_sequences.__name__} Can't parse data to make fasta\n{data_dump()}")
        outfile.write('%s\n' % '\n'.join(formatted_sequence_gen))

    return file_name
