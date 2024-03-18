from __future__ import annotations

import abc
import logging
import os
import subprocess
import time
from abc import ABC
from collections import UserList
from collections.abc import Iterable, Sequence
from copy import copy
from itertools import repeat, count
from logging import Logger
from math import floor
from pathlib import Path
from typing import Any, AnyStr, get_args, Literal

import numpy as np
from Bio.Align import substitution_matrices

from symdesign import metrics, utils
from symdesign.sequence import alignment_programs_literal, alignment_programs, hhblits, \
    MultipleSequenceAlignment, mutation_dictionary, numerical_translation_alph1_bytes, \
    numerical_translation_alph1_gaped_bytes, parse_hhblits_pssm, ProfileDict, ProfileEntry, protein_letters_alph1, \
    protein_letters_alph3, write_sequence_to_fasta, write_sequences, \
    get_equivalent_indices, generate_mutations, protein_letters_literal, AminoAcidDistribution
from symdesign.structure.utils import StructureException

# import dependencies.bmdca as bmdca
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
ZERO_OFFSET = 1
sequence_type_literal = Literal['reference_sequence', 'sequence']
sequence_types: tuple[sequence_type_literal, ...] = get_args(sequence_type_literal)
# aa_counts = dict(zip(utils.protein_letters_alph1, repeat(0)))
aa_counts_alph3 = dict(zip(protein_letters_alph3, repeat(0)))
blank_profile_entry = aa_counts_alph3.copy()
"""{utils.profile_keys, repeat(0))}"""
blank_profile_entry.update({'lod': aa_counts_alph3.copy(), 'info': 0., 'weight': 0.})
aa_nan_counts_alph3 = dict(zip(protein_letters_alph3, repeat(np.nan)))
"""{protein_letters_alph3, repeat(numpy.nan))}"""
nan_profile_entry = aa_nan_counts_alph3.copy()
"""{utils.profile_keys, repeat(numpy.nan))}"""
nan_profile_entry.update({'lod': aa_nan_counts_alph3.copy(), 'info': 0., 'weight': 0.})  # 'type': residue_type,

"""The shape should be (number of residues, number of characters in the alphabet"""
# 'uniclust30_2018_08'

# protein_letters_literal: tuple[str, ...] = get_args(utils.protein_letters_alph1_literal)
# numerical_translation = dict(zip(utils.protein_letters_alph1, range(len(utils.protein_letters_alph1))))
# protein_letters_alph1_extended: tuple[str, ...] = get_args(utils.protein_letters_alph1_extended_literal)


def sequence_to_one_hot(sequence: Sequence[str], translation_table: dict[str, int] = None,
                        alphabet_order: int = 1) -> np.ndarray:
    """Convert a sequence into a numeric array

    Args:
        sequence: The sequence to encode
        translation_table: If a translation table (in bytes) is provided, it will be used. If not, use alphabet_order
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3

    Returns:
        The one-hot encoded sequence with shape (sequence length, translation_table length)
    """
    numeric_sequence = sequence_to_numeric(sequence, translation_table, alphabet_order=alphabet_order)
    if translation_table is None:
        # Assumes that alphabet_order is used and there aren't missing letters...
        num_entries = 20
    else:
        num_entries = len(translation_table)
    one_hot = np.zeros((len(sequence), num_entries), dtype=np.int32)
    try:
        one_hot[:, numeric_sequence] = 1
    except IndexError:  # Our assumption above was wrong
        from symdesign import sequence as _seq
        embedding = getattr(_seq, f'numerical_translation_alph{alphabet_order}_bytes') \
            if translation_table is None else translation_table
        raise ValueError(
            f"Couldn't produce a proper one-hot encoding for the provided sequence embedding: {embedding}")
    return one_hot


def sequence_to_numeric(sequence: Sequence[str], translation_table: dict[str, int] = None,
                        alphabet_order: int = 1) -> np.ndarray:
    """Convert a sequence into a numeric array

    Args:
        sequence: The sequence to encode
        translation_table: If a translation table (in bytes) is provided, it will be used. If not, use alphabet_order
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3

    Returns:
        The numerically encoded sequence where each entry along axis=0 is the indexed amino acid. Indices are according
            to the 1 letter alphabetical amino acid
    """
    _array = np.array(list(sequence), np.string_)
    if translation_table is not None:
        return np.vectorize(translation_table.__getitem__)(_array)
    else:
        if alphabet_order == 1:
            return np.vectorize(numerical_translation_alph1_bytes.__getitem__)(_array)
        elif alphabet_order == 3:
            raise NotImplementedError('Need to make the "numerical_translation_alph3_bytes" table')
            return np.vectorize(numerical_translation_alph3_bytes.__getitem__)(_array)
        else:
            raise ValueError(
                f"The 'alphabet_order' {alphabet_order} isn't valid. Choose from either 1 or 3")


def sequences_to_numeric(sequences: Iterable[Sequence[str]], translation_table: dict[str, int] = None,
                         alphabet_order: int = 1) -> np.ndarray:
    """Convert sequences into a numeric array

    Args:
        sequences: The sequences to encode
        translation_table: If a translation table (in bytes) is provided, it will be used. If not, use alphabet_order
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3

    Returns:
        The numerically encoded sequence where each entry along axis=0 is the indexed amino acid. Indices are according
            to the 1 letter alphabetical amino acid
    """
    _array = np.array([list(sequence) for sequence in sequences], np.string_)
    if translation_table is not None:
        return np.vectorize(translation_table.__getitem__)(_array)
    else:
        if alphabet_order == 1:
            return np.vectorize(numerical_translation_alph1_bytes.__getitem__)(_array)
        elif alphabet_order == 3:
            raise NotImplementedError('Need to make the "numerical_translation_alph3_bytes" table')
            return np.vectorize(numerical_translation_alph3_bytes.__getitem__)(_array)
        else:
            raise ValueError(
                f"The 'alphabet_order' {alphabet_order} isn't valid. Choose from either 1 or 3")


def pssm_as_array(pssm: ProfileDict, alphabet: str = protein_letters_alph1, lod: bool = False) \
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


def write_pssm_file(pssm: ProfileDict, file_name: AnyStr = None, name: str = None,
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
        f.write(f'\n\n{" " * 12}{separation1.join(protein_letters_alph3)}'
                f'{separation1}{(" " * 3).join(protein_letters_alph3)}\n')
        for residue_number, profile in pssm.items():
            if isinstance(profile['lod']['A'], float):  # lod_freq:  # relevant for favor_fragment
                lod_string = ' '.join(f'{profile["lod"][aa]:>4.2f}' for aa in protein_letters_alph3) \
                    + ' '
            else:
                lod_string = ' '.join(f'{profile["lod"][aa]:>3d}' for aa in protein_letters_alph3) \
                    + ' '

            if isinstance(profile['A'], float):  # counts_freq: # relevant for freq calculations
                counts_string = ' '.join(f'{floor(profile[aa] * 100):>3.0f}' for aa in
                                         protein_letters_alph3) \
                    + ' '
            else:
                counts_string = ' '.join(f'{profile[aa]:>3d}' for aa in protein_letters_alph3) \
                    + ' '
            f.write(f'{residue_number:>5d} {profile["type"]:1s}   {lod_string:80s} {counts_string:80s} '
                    f'{round(profile.get("info", 0.), 4):4.2f} {round(profile.get("weight", 0.), 4):4.2f}\n')

    return file_name


class Profile(UserList[ProfileEntry]):
    data: list[ProfileEntry]
    """[{'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...},
         'type': 'W', 'info': 3.20, 'weight': 0.73},
        {}, ...]
    """
    def __init__(self, entries: Iterable[ProfileEntry], dtype: str = 'profile', **kwargs):
        """Construct the instance

        Args:
            entries: The per-residue entries to create the instance
            dtype: The datatype of the profile.
            **kwargs:
        """
        super().__init__(initlist=entries, **kwargs)  # initlist sets UserList.data
        self.dtype = dtype

    @property
    def available_keys(self) -> tuple[Any, ...]:
        """Returns the available ProfileEntry keys that are present in the Profile"""
        return tuple(self.data[0].keys())

    @property
    def lods(self) -> list[AminoAcidDistribution]:
        """The log of odds values, given for each amino acid type, for each entry in the Profile"""
        if 'lod' not in self.available_keys:
            raise ValueError(
                f"Couldn't find the {self.__class__.__name__}.lods' as the underlying data is missing key 'lod'"
            )
        return [position_info['lod'] for position_info in self]

    @property
    def types(self) -> list[protein_letters_literal]:
        """The amino acid type, for each entry in the Profile"""
        if 'type' not in self.available_keys:
            raise ValueError(
                f"Couldn't find the {self.__class__.__name__}.types' as the underlying data is missing key 'type'"
            )
        return [position_info['type'] for position_info in self]

    @property
    def weights(self) -> list[float]:
        """The weight assigned to each entry in the Profile"""
        if 'weight' not in self.available_keys:
            raise ValueError(
                f"Couldn't find the {self.__class__.__name__}.weights' as the underlying data is missing key 'weight'"
            )
        return [position_info['weight'] for position_info in self]

    @property
    def info(self) -> list[float]:
        """The information present for each entry in the Profile"""
        if 'info' not in self.available_keys:
            raise ValueError(
                f"Couldn't find the {self.__class__.__name__}.info' as the underlying data is missing key 'info'"
            )
        return [position_info['info'] for position_info in self]

    def as_array(self, alphabet: str = protein_letters_alph1, lod: bool = False) -> np.ndarray:
        """Convert the Profile into a numeric array

        Args:
            alphabet: The amino acid alphabet to use. Array values will be returned in this order
            lod: Whether to return the array for the log of odds values

        Returns:
            The numerically encoded pssm where each entry along axis 0 is the position, and the entries on axis 1 are
                the frequency data at every indexed amino acid. Indices are according to the specified amino acid
                alphabet, i.e. array([[0.1, 0.01, 0.12, ...], ...])
        """
        if lod:
            if self.lods:
                data_source = self.lods
            else:
                raise ValueError(
                    f"There aren't any values available for {self.__class__.__name__}.lods")
        else:
            data_source = self

        return np.array([[position_info[aa] for aa in alphabet]
                         for position_info in data_source], dtype=np.float32)

    def write(self, file_name: AnyStr = None, name: str = None, out_dir: AnyStr = os.getcwd()) -> AnyStr:
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
        data: list[list[float]] = [[position_info[aa] for aa in protein_letters_alph3] for position_info in self]
        if self.dtype == 'fragment':
            # Need to convert np.nan to zeros
            data = [[0 if np.nan else num for num in entry] for entry in data]
            logger.warning(f'Converting {self.dtype} type Profile np.nan values to 0.0')

        # Find out if the pssm has values expressed as frequencies (percentages) or as counts and modify accordingly
        lods = self.lods
        if isinstance(lods[0]['A'], float):
            separation1 = " " * 4
        else:
            separation1 = " " * 3
            # lod_freq = True
        # if type(pssm[first_key]['A']) == float:
        #     counts_freq = True

        if file_name is None:
            if name is None:
                raise ValueError(
                    f"Must provide argument 'file_name' or 'name' as a str to {self.write.__name__}")
            else:
                file_name = os.path.join(out_dir, name)

        if os.path.splitext(file_name)[-1] == '':  # No extension
            file_name = f'{file_name}.pssm'

        with open(file_name, 'w') as f:
            f.write(f'\n\n{" " * 12}{separation1.join(protein_letters_alph3)}'
                    f'{separation1}{(" " * 3).join(protein_letters_alph3)}\n')
            for residue_number, (entry, lod, _type, info, weight) in enumerate(
                    zip(data, lods, self.types, self.info, self.weights), 1):
                if isinstance(lod['A'], float):  # relevant for favor_fragment
                    lod_string = \
                        ' '.join(f'{lod[aa]:>4.2f}' for aa in protein_letters_alph3) + ' '
                else:
                    lod_string = \
                        ' '.join(f'{lod[aa]:>3d}' for aa in protein_letters_alph3) + ' '

                if isinstance(entry[0], float):  # relevant for freq calculations
                    counts_string = ' '.join(f'{floor(value * 100):>3.0f}' for value in entry) + ' '
                else:
                    counts_string = ' '.join(f'{value:>3d}' for value in entry) + ' '

                f.write(f'{residue_number:>5d} {_type:1s}   {lod_string:80s} {counts_string:80s} '
                        f'{round(info, 4):4.2f} {round(weight, 4):4.2f}\n')

        return file_name


class GeneEntity(ABC):
    """Contains the sequence information for a ContainsResidues."""
    _alpha: float
    _collapse_profile: np.ndarray  # pd.DataFrame
    _evolutionary_profile: dict | ProfileDict
    # _fragment_profile: list[list[set[dict]]] | list[ProfileEntry] | None
    _hydrophobic_collapse: np.ndarray
    _msa: MultipleSequenceAlignment | None
    _sequence_array: np.ndarray
    _sequence_numeric: np.ndarray
    alpha: list[float]
    fragment_profile: Profile | None
    h_fields: np.ndarray | None
    j_couplings: np.ndarray | None
    name: str
    profile: ProfileDict | dict

    def __init__(self, **kwargs):
        """Construct the instance

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)  # GeneEntity
        self._evolutionary_profile = {}  # position specific scoring matrix
        self.h_fields = None
        self.j_couplings = None
        self.profile = {}  # design/structure specific scoring matrix

    @property
    @abc.abstractmethod
    def log(self) -> Logger:
        """"""

    @property
    @abc.abstractmethod
    def sequence(self) -> str:
        """"""

    @property
    @abc.abstractmethod
    def reference_sequence(self) -> str:
        """"""

    @property
    def number_of_residues(self) -> int:
        """"""
        return len(self.sequence)

    @property
    def evolutionary_profile(self) -> dict:
        """Access the evolutionary_profile"""
        return self._evolutionary_profile

    @evolutionary_profile.setter
    def evolutionary_profile(self, evolutionary_profile: dict[int, profile]):
        """Set the evolutionary_profile"""
        self._evolutionary_profile = evolutionary_profile
        if not self._verify_evolutionary_profile():
            self._fit_evolutionary_profile_to_structure()

    @property
    def msa(self) -> MultipleSequenceAlignment | None:
        """The MultipleSequenceAlignment object for the instance"""
        try:
            return self._msa
        except AttributeError:
            return None

    @msa.setter
    def msa(self, msa: MultipleSequenceAlignment):
        """Set the GeneEntity MultipleSequenceAlignment object using a file path or an initialized instance"""
        if isinstance(msa, MultipleSequenceAlignment):
            self._msa = copy(msa)
            self._fit_msa_to_structure()
        else:
            self.log.warning(f"The passed msa (type: {msa.__class__.__name__}) isn't of the required type "
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
                np.vectorize(numerical_translation_alph1_gaped_bytes.__getitem__, otypes='l')(self._sequence_array)
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


    def _verify_evolutionary_profile(self) -> bool:
        """Returns True if the evolutionary_profile and ContainsResidues sequences are equivalent"""
        evolutionary_profile_len = len(self._evolutionary_profile)
        if self.number_of_residues != evolutionary_profile_len:
            self.log.debug(f'{self.name}: Profile and {self.__class__.__name__} are different lengths. Profile='
                           f'{evolutionary_profile_len}, Pose={self.number_of_residues}')
            return False

        # Check sequence from Pose and _evolutionary_profile to compare identity before proceeding
        warn = False
        mismatch_warnings = []
        for idx, (aa, position_data) in enumerate(zip(self.sequence, self._evolutionary_profile.values())):
            profile_res_type = position_data['type']
            # pose_res_type = protein_letters_3to1[residue.type]
            if profile_res_type != aa:
                # This may not be the worst thing in the world... If the profile was made off of an entity
                # that is not the exact structure, there should be some reality to it. I think the issue would
                # be with Rosetta loading of the Sequence Profile and not matching. I am trying to mutate the
                # offending residue type in the evolutionary profile to the Pose residue type. The frequencies
                # will reflect the actual values desired, however the surface level will be different.
                # Otherwise, generating evolutionary profiles from individual files will be required which
                # don't contain a reference sequence and therefore have their own caveats. Warning the user
                # will allow the user to understand what is happening at least
                mismatch_warnings.append(f'Sequence {idx=}: .evolutionary_profile={profile_res_type}, '
                                         f'.sequence={aa}')
                if position_data[aa] > 0:  # The occurrence data indicates this AA is also possible
                    warn = True
                    position_data['type'] = aa
                else:
                    self.log.critical(f'The {repr(self)}.evolutionary_profile was likely created from a different file,'
                                      " and the information contained ISN'T viable")
                    break
        else:
            if warn:
                self.log.warning(f'The {repr(self)}.evolutionary_profile was likely created from a different file, '
                                 f'however, the information is still viable. Each of {len(mismatch_warnings)} '
                                 'mismatched residues in the profile will be substituted for the corresponding '
                                 '.sequence residue')
                mismatch_str = "\n\t".join(mismatch_warnings)
                self.log.info(f'{self.__class__.__name__}.evolutionary_profile and .sequence mismatched:'
                              f'{mismatch_str}')
                if len(mismatch_warnings) > 3:
                    self.log.warning(f'This error occurred {len(mismatch_warnings)} times. Your modeling accuracy may '
                                     'suffer')
            return True

        return False

    def add_evolutionary_profile(self, file: AnyStr = None, out_dir: AnyStr = os.getcwd(),
                                 profile_source: alignment_programs_literal = putils.hhblits, force: bool = False,
                                 **kwargs):
        """Add the evolutionary profile to the GeneEntity. If the profile isn't provided, it is generated through search
        of homologous protein sequences using the profile_source argument

        Args:
            file: Location where profile file should be loaded from
            out_dir: Location where sequence files should be written
            profile_source: One of 'hhblits' or 'psiblast'
            force: Whether to force generation of a new profile

        Sets:
            self.evolutionary_profile (ProfileDict)
        """
        if profile_source not in alignment_programs:  # [putils.hhblits, 'psiblast']:
            raise ValueError(
                f'{self.add_evolutionary_profile.__name__}: Profile generation only possible from '
                f'{", ".join(alignment_programs)}, not {profile_source}')

        if file is not None:
            pssm_file = file
        else:  # Check to see if the files of interest already exist
            # Extract/Format Sequence Information. SEQRES is prioritized if available
            name = self.name
            sequence_file = self.write_sequence_to_fasta(out_dir=out_dir)
            temp_file = Path(out_dir, f'{name}.hold')
            pssm_file = os.path.join(out_dir, f'{name}.hmm')
            if not os.path.exists(pssm_file) or force:
                if not os.path.exists(temp_file):  # No work on this pssm file has been initiated
                    # Create blocking file to prevent excess work
                    with open(temp_file, 'w') as f:
                        self.log.info(f"Fetching '{name}' sequence data")

                    self.log.info(f'Generating Evolutionary Profile for {name}')
                    getattr(self, profile_source)(sequence_file=sequence_file, out_dir=out_dir)
                    temp_file.unlink(missing_ok=True)
                    # if os.path.exists(temp_file):
                    #     os.remove(temp_file)
                else:  # Block is in place, another process is working
                    self.log.info(f"Waiting for '{name}' profile generation...")
                    while not os.path.exists(pssm_file):
                        if int(time.time()) - int(os.path.getmtime(temp_file)) > 5400:  # > 1 hr 30 minutes have passed
                            # os.remove(temp_file)
                            temp_file.unlink(missing_ok=True)
                            raise StructureException(
                                f'{self.add_evolutionary_profile.__name__}: Generation of the profile for {name} '
                                'took longer than the time limit\nKilled')
                        time.sleep(20)

        # Set self.evolutionary_profile
        self.evolutionary_profile = parse_hhblits_pssm(pssm_file)

    def create_null_profile(self, nan: bool = False, zero_index: bool = False, **kwargs) -> ProfileDict:
        """Make a blank profile

        Args:
            nan: Whether to fill the null profile with np.nan
            zero_index: bool = False - If True, return the dictionary with zero indexing

        Returns:
            Dictionary containing profile information with keys as the index (zero or one-indexed), values as PSSM
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
        """
        offset = 0 if zero_index else ZERO_OFFSET

        if nan:
            _profile_entry = nan_profile_entry
        else:
            _profile_entry = blank_profile_entry

        profile = {residue: _profile_entry.copy()
                   for residue in range(offset, offset + self.number_of_residues)}

        for residue_data, residue_type in zip(profile.values(), self.sequence):
            residue_data['type'] = residue_type

        return profile

    @staticmethod
    def create_null_entries(entry_numbers: Iterable[int], nan: bool = False, **kwargs) -> ProfileDict:
        """Make a blank profile

        Args:
            entry_numbers: The numbers to generate null entries for
            nan: Whether to fill the null profile with np.nan

        Returns:
            Dictionary containing profile information with the specified entries as the index, values as PSSM
            Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'info': 3.20, 'weight': 0.73},
                 2: {}, ...}
            Importantly, there is no 'type' key. This must be added
        """
        # offset = 0 if zero_index else ZERO_OFFSET

        if nan:
            _profile_entry = nan_profile_entry
        else:
            _profile_entry = blank_profile_entry

        return {entry: _profile_entry.copy() for entry in entry_numbers}

    def _fit_evolutionary_profile_to_structure(self):
        """From an evolutionary profile generated according to a reference sequence, align the profile to the Residue
        sequence, removing profile information for Residue instances that are absent

        Sets:
            self.evolutionary_profile (ProfileDict)
        """
        # Generate the disordered indices which are positions that are present in evolutionary_profile_sequence,
        # but are missing in structure or structure missing in evolutionary_profile_sequence
        # Removal of unmodeled positions from self.evolutionary_profile and addition of unaligned structure regions
        # will produce a properly indexed profile
        evolutionary_profile_sequence = ''.join(data['type'] for data in self._evolutionary_profile.values())
        evolutionary_gaps = \
            generate_mutations(evolutionary_profile_sequence, self.sequence, only_gaps=True, return_to=True)
        self.log.debug(f'evolutionary_gaps: {evolutionary_gaps}')

        # Insert c-terminal structure residues
        first_index = ZERO_OFFSET
        last_profile_number = len(evolutionary_profile_sequence)
        # entry_number < first_index removes any structure gaps as these are all >= first_index
        nterm_extra_structure_sequence = [entry_number for entry_number in evolutionary_gaps
                                          if entry_number < first_index]
        number_of_nterm_entries = len(nterm_extra_structure_sequence)
        cterm_extra_profile_entries = self.create_null_entries(
            [entry_number for entry_number, residue_type in evolutionary_gaps.items()
             if entry_number > last_profile_number and residue_type != '-'])
        if cterm_extra_profile_entries:
            for entry_number, residue_data in cterm_extra_profile_entries.items():
                residue_data['type'] = evolutionary_gaps.pop(entry_number)
            # Offset any remaining gaps by the number of n-termini added
            cterm_extra_profile_entries = {entry_number + number_of_nterm_entries: residue_data
                                           for entry_number, residue_data in cterm_extra_profile_entries.items()}
            self.log.debug(f'structure_evolutionary_profile c-term entries: {cterm_extra_profile_entries.keys()}')

        # Insert n-terminal residues
        if nterm_extra_structure_sequence:
            structure_evolutionary_profile = self.create_null_entries(
                range(first_index, first_index + number_of_nterm_entries))
            for entry_number, residue_data in zip(nterm_extra_structure_sequence, structure_evolutionary_profile.values()):
                residue_data['type'] = evolutionary_gaps.pop(entry_number)

            self.log.debug(f'structure_evolutionary_profile n-term entries: {structure_evolutionary_profile.keys()}')
            # Offset any remaining gaps by the number added
            evolutionary_gaps = {entry_number + number_of_nterm_entries: residue_data
                                 for entry_number, residue_data in evolutionary_gaps.items()}
        else:
            structure_evolutionary_profile = {}

        # Renumber the structure_evolutionary_profile to offset all to 1
        new_entry_number = count(number_of_nterm_entries + 1)
        structure_evolutionary_profile.update({next(new_entry_number): residue_data
                                               for entry_number, residue_data in self._evolutionary_profile.items()})
        structure_evolutionary_profile.update(cterm_extra_profile_entries)

        internal_sequence_characters = set(evolutionary_gaps.values()).difference(('-',))
        if internal_sequence_characters:  # There are internal insertions
            evolutionary_gaps = {entry_number: residue_type for entry_number, residue_type in evolutionary_gaps.items()
                                 if residue_type != '-'}
            logger.debug("There are internal regions which aren't accounted for in the evolutionary_profile, but are "
                         f'present in the structure: {evolutionary_gaps}')
            existing_structure_profile_values = list(structure_evolutionary_profile.values())
            # existing_structure_profile_keys = list(structure_evolutionary_profile.keys())
            null_insertion_profiles = self.create_null_entries(evolutionary_gaps.keys())
            # Insert these in reverse order to keep numbering correct, one at a time...
            # for mutation_entry_number, residue_type in reversed(evolutionary_gaps.items()):
            for mutation_entry_number, residue_type in evolutionary_gaps.items():
                # for entry_number in reversed(existing_structure_profile_keys[mutation_entry_number - 1:]):
                #     structure_evolutionary_profile[entry_number + 1] = structure_evolutionary_profile.pop(entry_number)
                insertion_profile_entry = null_insertion_profiles[mutation_entry_number]
                insertion_profile_entry['type'] = residue_type
                existing_structure_profile_values.insert(mutation_entry_number - 1, insertion_profile_entry)

            # structure_evolutionary_profile = {entry_number: structure_evolutionary_profile[entry_number]
            #                                   for entry_number in sorted(structure_evolutionary_profile.keys())}
            structure_evolutionary_profile = {
                entry_number: entry for entry_number, entry in enumerate(existing_structure_profile_values, 1)}

        self.log.debug(f'structure_evolutionary_profile.keys(): {structure_evolutionary_profile.keys()}')

        evolutionary_profile_sequence = ''.join(data['type'] for data in structure_evolutionary_profile.values())

        query_align_indices, reference_align_indices = \
            get_equivalent_indices(evolutionary_profile_sequence, self.sequence, mutation_allowed=True)

        for idx, entry in enumerate(list(structure_evolutionary_profile.keys())):
            if idx not in query_align_indices:
                structure_evolutionary_profile.pop(entry)

        self.log.debug(f'{self._fit_evolutionary_profile_to_structure.__name__}:\n\tOld:\n'
                       # f'{"".join(res["type"] for res in self._evolutionary_profile.values())}\n\tNew:\n'
                       f'{evolutionary_profile_sequence}\n\tNew:\n'
                       f'{"".join(res["type"] for res in structure_evolutionary_profile.values())}')
        self._evolutionary_profile = structure_evolutionary_profile

    def _fit_msa_to_structure(self):
        """From a multiple sequence alignment to the reference sequence, align the profile to the Residue sequence.

        Removes the view of all data not present in the structure

        Sets:
            self.msa.sequence_indices (np.ndarray)
        """
        msa = self.msa
        sequence = self.sequence
        # Similar routine in fit_evolutionary_profile_to_structure()
        # See if there are any insertions in the self.sequence that are not in the MSA
        # return_to will give the self.sequence values at the mutation site
        mutations_structure_missing_from_msa = \
            generate_mutations(msa.query, sequence, only_gaps=True, return_to=True, zero_index=True)
        if mutations_structure_missing_from_msa:  # Two sequences don't align
            self.log.debug(f'mutations_structure_missing_from_msa: {mutations_structure_missing_from_msa}')

            # Get any mutations that are present in the structure but not the msa
            # Perform insertions in (almost) reverse order
            last_msa_number = msa.query_length
            cterm_extra_structure_indices = [
                index for index, residue_type in mutations_structure_missing_from_msa.items()
                if index >= last_msa_number and residue_type != '-']
            if cterm_extra_structure_indices:
                self.log.debug(f'c-term msa insertion indices: {cterm_extra_structure_indices}')
                cterm_sequence = ''.join(mutations_structure_missing_from_msa.pop(idx)
                                         for idx in cterm_extra_structure_indices)
                self.log.debug(f'c-term insertion sequence: {cterm_sequence}')
                msa.insert(last_msa_number, cterm_sequence)

            msa_start_index = 0
            nterm_extra_structure_indices = [index for index in mutations_structure_missing_from_msa
                                             if index < msa_start_index]
            if nterm_extra_structure_indices:
                self.log.debug(f'n-term msa insertion indices: {nterm_extra_structure_indices}')
                nterm_sequence = ''.join(mutations_structure_missing_from_msa.pop(idx)
                                         for idx in nterm_extra_structure_indices)
                insert_length = len(nterm_sequence)
                self.log.debug(f'Inserting {insert_length} residues on the n-term with sequence: {nterm_sequence}')
                msa.insert(msa_start_index, nterm_sequence)
                mutations_structure_missing_from_msa = {
                    index + insert_length: mutation for index, mutation in mutations_structure_missing_from_msa.items()}

            internal_sequence_characters = set(mutations_structure_missing_from_msa.values()).difference(('-',))
            if internal_sequence_characters:  # There are internal insertions
                mutations_structure_missing_from_msa = {
                    entry_number: residue_type
                    for entry_number, residue_type in mutations_structure_missing_from_msa.items()
                    if residue_type != '-'}
                logger.debug("There are internal regions which aren't accounted for in the MSA, but are present in the "
                             f'structure: {mutations_structure_missing_from_msa}')
                # Insert these in reverse order to keep numbering correct, one at a time...
                # for mutation_idx in reversed(mutations_structure_missing_from_msa.keys()):
                for mutation_idx in mutations_structure_missing_from_msa.keys():
                    msa.insert(mutation_idx, mutations_structure_missing_from_msa[mutation_idx])

        # Get the sequence_indices now that insertions are handled
        sequence_indices = msa.sequence_indices
        # self.log.critical(sequence_indices.shape)
        # Get all non-zero/False, numerical indices for the query
        msa_query_indices = np.flatnonzero(msa.query_indices)
        # if len(self.reference_sequence) != msa.query_length:
        # if self.number_of_residues != msa.query_length:
        #     # Todo this check isn't sufficient as the residue types might not be the same
        #     #  see verify_evolutionary_profile and make into -> self._verify_msa_profile()
        #     self.log.info(f'The {self.name} .sequence length, {self.number_of_residues} != '
        #                   f'{msa.query_length}, the MultipleSequenceAlignment query length')

        # query_align_indices, reference_align_indices = get_equivalent_indices(msa.query, self.reference_sequence)
        # Todo consolidate the alignment and generate_mutations call here with the generate_mutations below
        #  mutations = generate_mutations(target, query, keep_gaps=True, return_all=True)
        query_align_indices, reference_align_indices = \
            get_equivalent_indices(msa.query, sequence, mutation_allowed=True)
        self.log.debug('For MSA alignment to the .sequence, found the corresponding MSA query indices:'
                       f' {query_align_indices}')
        # Set all indices that align to True, all others are False
        aligned_query_indices = msa_query_indices[query_align_indices]
        sequence_indices[0] = False
        sequence_indices[0, aligned_query_indices] = True

        # Set the updated indices
        msa.sequence_indices = sequence_indices

    # def psiblast(self, out_dir: AnyStr = os.getcwd(), remote: bool = False):
    #     """Generate a position specific scoring matrix using PSI-BLAST subprocess
    #
    #     Args:
    #         out_dir: Disk location where generated file should be written
    #         remote: Whether to perform the search through the web. If False, need blast installed locally
    #
    #     Sets:
    #         self.pssm_file (str): Name of the file generated by psiblast
    #     """
    #     pssm_file = os.path.join(out_dir, f'{self.name}.pssm')
    #     sequence_file = self.write_sequence_to_fasta(out_dir=out_dir)
    #     cmd = ['psiblast', '-db', putils.alignmentdb, '-query', sequence_file, '-out_ascii_pssm',
    #            pssm_file, '-save_pssm_after_last_round', '-evalue', '1e-6', '-num_iterations', '0']  # Todo # iters
    #     if remote:
    #         cmd.append('-remote')
    #     else:
    #         cmd.extend(['-num_threads', '8'])  # Todo
    #
    #     p = subprocess.Popen(cmd)
    #     p.communicate()

    # # @handle_errors(errors=(FileNotFoundError,))
    # def parse_psiblast_pssm(self, **kwargs):
    #     """Take the contents of a pssm file, parse, and input into a sequence dictionary.
    #     # Todo
    #     #  Currently impossible to use in calculate_profile. Change psiblast lod score parsing
    #     Sets:
    #         self.evolutionary_profile (ProfileDict): Dictionary containing residue indexed profile information
    #         Ex: {1: {'A': 0, 'R': 0, ..., 'lod': {'A': -5, 'R': -5, ...}, 'type': 'W', 'info': 3.20, 'weight': 0.73},
    #              2: {}, ...}
    #     """
    #     with open(pssm_file, 'r') as f:
    #         lines = f.readlines()
    #
    #     evolutionary_profile = {}
    #     for line in lines:
    #         line_data = line.strip().split()
    #         if len(line_data) == 44:
    #             residue_number = int(line_data[0])
    #             evolutionary_profile[residue_number] = copy(aa_counts_alph3)
    #             for i, aa in enumerate(protein_letters_alph3, 22):  # pose_dict[residue_number], 22):
    #                 # Get normalized counts for pose_dict
    #                 evolutionary_profile[residue_number][aa] = (int(line_data[i]) / 100.0)
    #             evolutionary_profile[residue_number]['lod'] = {}
    #             for i, aa in enumerate(protein_letters_alph3, 2):
    #                 evolutionary_profile[residue_number]['lod'][aa] = line_data[i]
    #             evolutionary_profile[residue_number]['type'] = line_data[1]
    #             evolutionary_profile[residue_number]['info'] = float(line_data[42])
    #             evolutionary_profile[residue_number]['weight'] = float(line_data[43])
    #
    #     self.evolutionary_profile = evolutionary_profile

    def hhblits(self, out_dir: AnyStr = os.getcwd(), **kwargs) -> list[str] | None:
        """Generate a position specific scoring matrix from hhblits using Hidden Markov Models

        Args:
            out_dir: Disk location where generated file should be written

        Keyword Args:
            sequence_file: AnyStr = None - The file containing the sequence to use
            threads: Number of cpu's to use for the process
            return_command: Whether to simply return the hhblits command

        Returns:
            The command if return_command is True, otherwise None
        """
        result = hhblits(self.name, out_dir=out_dir, **kwargs)

        if result:  # return_command is True
            return result
        # Otherwise, make alignment file(s)
        name = self.name
        # Set file attributes according to logic of hhblits()
        a3m_file = os.path.join(out_dir, f'{name}.a3m')
        msa_file = os.path.join(out_dir, f'{name}.sto')
        fasta_msa = os.path.join(out_dir, f'{name}.fasta')
        # Preferred alignment type
        if os.access(putils.reformat_msa_exe_path, os.X_OK):
            p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, msa_file, '-num', '-uc'])
            p.communicate()
            p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, fasta_msa, '-M', 'first', '-r'])
            p.communicate()
        else:
            logger.error(f"Couldn't execute multiple sequence alignment reformatting script")

    def add_msa_from_file(self, msa_file: AnyStr, file_format: msa_supported_types_literal = 'stockholm'):
        """Add a multiple sequence alignment to the profile. Handles correct sizing of the MSA

        Args:
            msa_file: The multiple sequence alignment file to add to the Entity
            file_format: The file type to read the multiple sequence alignment
        """
        if file_format == 'stockholm':
            constructor = MultipleSequenceAlignment.from_stockholm
        elif file_format == 'fasta':
            constructor = MultipleSequenceAlignment.from_fasta
        else:
            raise ValueError(
                f"The file format '{file_format}' isn't an available format. Available formats include "
                f"{msa_supported_types}")

        self.msa = constructor(msa_file)

    def collapse_profile(self, msa_file: AnyStr = None, **kwargs) -> np.ndarray:
        """Make a profile out of the hydrophobic collapse index (HCI) for each sequence in a multiple sequence alignment

        Takes ~5-10 seconds depending on the size of the msa

        Calculate HCI for each sequence in the MSA (which are different lengths). This is the Hydro Collapse array. For
        each sequence, make a Gap mask, with full shape (length, number_of_residues) to account for gaps in
        each sequence. Apply the mask using a map between the Gap mask and the Hydro Collapse array. Finally, drop the
        columns from the array that are gaps in the reference sequence.

        iter array   -   Gap mask      -       Hydro Collapse array     -     Aligned HCI     - -     Final HCI

        ------------

        iter - - - - - - 0 is gap    - -     compute for each     -     account for gaps   -  (drop idx 2)

        it 1 2 3 4  - - 0 | 1 | 2 - - - - - - - 0 | 1 | 2 - - - - - - 0 | 1 | 2 - - - - - - - 0 | 1 | 3 | ... N

        0 0 1 2 2  - - 1 | 1 | 0 - - -   - - - 0.5 0.2 0.5 - -  = - - 0.5 0.2 0.0 -  ->   - - 0.5 0.2 0.4 ... 0.3

        1 0 0 1 2  - - 0 | 1 | 1 - - -   - - - 0.4 0.7 0.4 - -  = - - 0.0 0.4 0.7 -  ->   - - 0.0 0.4 0.4 ... 0.1

        2 0 0 1 2  - - 0 | 1 | 1 - - -   - - - 0.3 0.6 0.3 - -  = - - 0.0 0.3 0.6 -  ->   - - 0.0 0.3 0.4 ... 0.0

        Where index 0 is the MSA query sequence

        After iteration cumulative summation, the iterator is multiplied by the gap mask. Next the Hydro Collapse array
        value is accessed by the gaped iterator. This places the Hydro Collapse array or np.nan (if there is a 0 index,
        i.e. a gap). After calculation, the element at index 2 is dropped from the array when the aligned sequence gaps
        are removed. Finally, only the indices of the query sequence are left in the profile, essentially giving the HCI
        for each sequence in the native context, adjusted to the specific context of the protein sequence at hand

        Args:
            msa_file: The multiple sequence alignment file to use for collapse. Will use .msa attribute if not provided

        Keyword Args:
            file_format: msa_supported_types_literal = 'stockholm' - The file type to read the multiple sequence
                alignment
            hydrophobicity: int = 'standard' – The hydrophobicity scale to consider. Either 'standard' (FILV),
                'expanded' (FMILYVW), or provide one with 'custom' keyword argument
            custom: mapping[str, float | int] = None – A user defined mapping of amino acid type, hydrophobicity value
                pairs
            alphabet_type: alphabet_types = None – The amino acid alphabet if the sequence consists of integer
                characters
            lower_window: int = 3 – The smallest window used to measure
            upper_window: int = 9 – The largest window used to measure

        Returns:
            Array with shape (length, number_of_residues) containing the hydrophobic collapse values for
                per-residue, per-sequence in the profile. The "query" sequence from the MultipleSequenceAlignment.query
                is located at index 0 on axis=0
        """
        try:
            return self._collapse_profile
        except AttributeError:
            msa = self.msa
            if not msa:
                self.add_msa_from_file(msa_file)
                msa = self.msa

            # Make the output array. Use one additional length to add np.nan value at the 0 index for gaps
            evolutionary_collapse_np = np.zeros((msa.length, msa.number_of_positions + 1))
            evolutionary_collapse_np[:, 0] = np.nan  # np.nan for all missing indices
            for idx, sequence in enumerate(msa.sequences):
                non_gaped_sequence = str(sequence).replace('-', '')
                evolutionary_collapse_np[idx, 1:len(non_gaped_sequence) + 1] = \
                    metrics.hydrophobic_collapse_index(non_gaped_sequence, **kwargs)
            # Todo this should be possible now metrics.hydrophobic_collapse_index(self.msa.array)

            msa_sequence_indices = msa.sequence_indices
            iterator_np = np.cumsum(msa_sequence_indices, axis=1) * msa_sequence_indices
            aligned_hci_np = np.take_along_axis(evolutionary_collapse_np, iterator_np, axis=1)
            # Select only the query sequence indices
            # sequence_hci_np = aligned_hci_np[:, self.msa.query_indices]
            # print('aligned_hci_np', aligned_hci_np.shape, aligned_hci_np)
            # print('self.msa.query_indices', self.msa.query_indices.shape, self.msa.query_indices)
            self._collapse_profile = aligned_hci_np[:, msa.query_indices]
            # self._collapse_profile = pd.DataFrame(aligned_hci_np[:, self.msa.query_indices],
            #                                       columns=list(range(1, self.msa.query_length + 1)))  # One-indexed
            # summary = pd.concat([sequence_hci_df, pd.concat([sequence_hci_df.mean(), sequence_hci_df.std()], axis=1,
            #                                                 keys=['mean', 'std']).T])

            return self._collapse_profile

    def direct_coupling_analysis(self, msa_file: AnyStr = None, **kwargs) -> np.ndarray:
        """Using boltzmann machine direct coupling analysis (bmDCA), score each sequence in an alignment based on the
         statistical energy compared to the learned DCA model

        Args:
            msa_file: The multiple sequence alignment file to use for collapse. Will use .msa attribute if not provided

        Keyword Args:
            file_format: msa_supported_types_literal = 'stockholm' - The file type to read the multiple sequence
                alignment

        Returns:
            Array with shape (length, number_of_residues) where the values are the energy for each residue/sequence
                based on direct coupling analysis parameters
        """
        # Check if required attributes are present
        _raise = False
        missing_attrs = []
        msa = self.msa
        if not msa:
            if msa_file:
                self.add_msa_from_file(msa_file)
                msa = self.msa
            else:
                missing_attrs.append('.msa')
        h_fields = self.h_fields
        if not h_fields:
            missing_attrs.append('.h_fields')
        j_couplings = self.j_couplings
        if not j_couplings:
            missing_attrs.append('.j_couplings')

        if missing_attrs:
            raise AttributeError(
                f"The required attribute(s) {', '.join(missing_attrs)} aren't available. Add to the "
                f'Entity before {self.direct_coupling_analysis.__name__}')

        analysis_length = msa.query_length
        idx_range = np.arange(analysis_length)
        # h_fields = bmdca.load_fields(os.path.join(data_dir, '%s_bmDCA' % self.name, 'parameters_h_final.bin'))
        # h_fields = h_fields.T  # this isn't required when coming in Fortran order, i.e. (21, analysis_length)
        # sum the h_fields values for each sequence position in every sequence
        h_values = h_fields[msa.numerical_alignment, idx_range[None, :]].sum(axis=1)
        h_sum = h_values.sum(axis=1)

        # coming in as a 4 dimension (analysis_length, analysis_length, alphabet_number, alphabet_number) ndarray
        # j_couplings = bmdca.load_couplings(os.path.join(data_dir, '%s_bmDCA' % self.name, 'parameters_J_final.bin'))
        i_idx = np.repeat(idx_range, analysis_length)
        j_idx = np.tile(idx_range, analysis_length)
        i_aa = np.repeat(msa.numerical_alignment, analysis_length)
        j_aa = np.tile(msa.numerical_alignment, msa.query_length)
        j_values = np.zeros((msa.length, len(i_idx)))
        for idx in range(msa.length):
            j_values[idx] = j_couplings[i_idx, j_idx, i_aa, j_aa]
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
        # j_sum = np.zeros((self.msa.length, len(couplings_idx)))
        # for idx in range(self.msa.length):
        #     j_sum[idx] = j_couplings[couplings_idx[idx]]
        # return -h_sum - j_sum
        return -h_values - j_values

    def write_sequence_to_fasta(self, dtype: sequence_type_literal = 'reference_sequence', file_name: AnyStr = None,
                                name: str = None, out_dir: AnyStr = os.getcwd()) -> AnyStr:
        """Write a sequence to a .fasta file with fasta format and return the file location.
        '.fasta' is appended if not specified in the name argument

        Args:
            dtype: The type of sequence to write. Can be the the keywords 'reference_sequence' or 'sequence'
            file_name: The explicit name of the file
            name: The name of the sequence record. If not provided, the instance name will be used.
                Will be used as the default file_name base name if file_name not provided
            out_dir: The location on disk to output the file. Only used if file_name not explicitly provided
        Returns:
            The path to the output file
        """
        if dtype in sequence_types:
            # Get the attribute from the instance
            sequence = getattr(self, dtype)
        else:
            raise ValueError(
                f"Couldn't find a sequence matching the {dtype=}"
            )

        if name is None:
            name = self.name
        if file_name is None:
            file_name = os.path.join(out_dir, name)
            if not file_name.endswith('.fasta'):
                file_name = f'{file_name}.fasta'

        return write_sequence_to_fasta(sequence=sequence, name=name, file_name=file_name)


dtype_literals = Literal['list', 'set', 'tuple', 'float', 'int']


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


def overlap_consensus(issm, aa_set):  # UNUSED
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


def get_cluster_dicts(db: str = putils.biological_interfaces, id_list: list[str] = None) -> dict[str, dict]:
    """Generate an interface specific scoring matrix from the fragment library

    Args:
        db: The source of the fragment information
        id_list: [1_2_24, ...]

    Returns:
         cluster_dict: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    """
    info_db = putils.frag_directory[db]
    if id_list is None:
        directory_list = utils.get_base_root_paths_recursively(info_db)
    else:
        directory_list = []
        for _id in id_list:
            c_id = _id.split('_')
            _dir = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(_dir)

    cluster_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        cluster_dict[os.path.basename(cluster)] = utils.unpickle(filename)

    return cluster_dict


def return_cluster_id_string(cluster_rep: str, index_number: int = 3) -> str:
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


def offset_index(dictionary: dict[int, Any], to_zero: bool = False) -> dict[int, dict]:
    """Modify the index of a sequence dictionary. Default is to one-indexed. to_zero=True gives zero-indexed"""
    if to_zero:
        return {residue - ZERO_OFFSET: dictionary[residue] for residue in dictionary}
    else:
        return {residue + ZERO_OFFSET: dictionary[residue] for residue in dictionary}


def residue_object_to_number(
    residue_dict: dict[str, Iterable['structure.base.Residue']]
) -> dict[str, list[tuple[int, ...]]]:
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
                residue_num_set.append(resi_number)
            pairs.append(tuple(residue_num_set))
        residue_dict[entry] = pairs

    return residue_dict


# def consensus_sequence(pssm):
#     """Return the consensus sequence from a PSSM
#
#     Args:
#         pssm (dict): pssm dictionary
#     Return:
#         consensus_identities (dict): {1: 'M', 2: 'H', ...} One-indexed
#     """
#     consensus_identities = {}
#     for residue in pssm:
#         max_lod = 0
#         max_res = pssm[residue]['type']
#         for aa in protein_letters_alph3:
#             if pssm[residue]['lod'][aa] > max_lod:
#                 max_lod = pssm[residue]['lod'][aa]
#                 max_res = aa
#         consensus_identities[residue + zero_offset] = max_res
#
#     return consensus_identities


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
                residue_num = residue_atom.residue_number - ZERO_OFFSET  # zero index
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
#
#     Keyword Args:
#         outpath=None (str): Disk location where generated file should be written
#         remote=False (bool): Whether to perform the serach locally (need blast installed locally) or perform search through web
#
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


# @handle_errors(errors=(FileNotFoundError,))
# def parse_stockholm_to_msa(file, **kwargs):
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
#     return MultipleSequenceAlignment((read_stockholm_file(file)), **kwargs)


# def make_pssm_file(pssm_dict: ProfileDict, name: str, out_dir: AnyStr = os.getcwd()):
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
#             line = '{:>5d} {:1s}   {:80s} {:80s} {:4.2f} {:4.2f}''\n'.format(res + ZERO_OFFSET, aa_type, lod_string,
#                                                                              counts_string, round(info, 4),
#                                                                              round(weight, 4))
#             f.write(line)
#         f.write(footer)
#
#     return out_file


def sequence_difference(seq1: Sequence, seq2: Sequence, d: dict = None, matrix: str = 'BLOSUM62') -> float:
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
    scores = [matrix_.get((letter1, letter2), matrix_.get((letter2, letter1))) for letter1, letter2 in pairs]

    return sum(scores)


msa_supported_types_literal = Literal['fasta', 'stockholm']
msa_supported_types: tuple[msa_supported_types_literal, ...] = get_args(msa_supported_types_literal)
msa_format_extension = dict(zip(msa_supported_types, ('.fasta', '.sto')))
