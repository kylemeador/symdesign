from __future__ import annotations

import logging
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import count
from math import log2
from pathlib import Path
from typing import Any, AnyStr, get_args, Literal, TypedDict

import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Align import Alignment, MultipleSeqAlignment, PairwiseAligner, PairwiseAlignments, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from symdesign import utils
putils = utils.path


# Globals
zero_offset = 1
logger = logging.getLogger(__name__)
# Types
protein_letters3_alph1_literal = Literal[
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
    'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
protein_letters3_alph1_extended_literal = Literal[
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
    'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ASX', 'XAA', 'GLX', 'XLE', 'SEC', 'PYL']
protein_letters3_alph1: tuple[str, ...] = get_args(protein_letters3_alph1_literal)
protein_letters3_alph1_extended: tuple[str, ...] = get_args(protein_letters3_alph1_extended_literal)
protein_letters_literal = Literal[
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
protein_letters_alph1_literal = Literal[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
protein_letters_alph1_gaped_literal = Literal[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
protein_letters_alph1_extended_literal = Literal[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'B', 'X', 'Z', 'J', 'U', 'O']
protein_letters_alph1_extended_and_gap_literal = Literal[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'B', 'X', 'Z', 'J', 'U', 'O', '-']
protein_letters_alph3_unknown_gaped_literal = Literal[
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-']
protein_letters_alph1: str = ''.join(get_args(protein_letters_alph1_literal))
"""ACDEFGHIKLMNPQRSTVWY"""
protein_letters_alph1_extended: str = ''.join(get_args(protein_letters_alph1_extended_literal))
"""ACDEFGHIKLMNPQRSTVWYBXZJUO"""
protein_letters_extended_and_gap: str = ''.join(get_args(protein_letters_alph1_extended_and_gap_literal))
"""ACDEFGHIKLMNPQRSTVWYBXZJUO-"""
protein_letters_3to1: dict[str, str] = dict(zip(protein_letters3_alph1, protein_letters_alph1))
protein_letters_1to3: dict[str, str] = dict(zip(protein_letters_alph1, protein_letters3_alph1))
protein_letters_3to1_extended: dict[str, str] = dict(zip(protein_letters3_alph1_extended, protein_letters_alph1_extended))
protein_letters_1to3_extended: dict[str, str] = dict(zip(protein_letters_alph1_extended, protein_letters3_alph1_extended))
protein_letters_alph3 = ''.join(get_args(protein_letters_literal))
protein_letters_alph1_unknown = protein_letters_alph1 + 'X'
protein_letters_alph3_unknown = protein_letters_alph3 + 'X'
protein_letters_alph1_gaped = protein_letters_alph1 + '-'
protein_letters_alph3_gaped = protein_letters_alph3 + '-'
protein_letters_alph1_unknown_gaped = protein_letters_alph1 + 'X-'
protein_letters_alph3_unknown_gaped = protein_letters_alph3 + 'X-'
# Todo the default value for many of these might have conflicting use cases
#  For instance, a value of 20 for protein_letters_alph1_unknown_gaped would but a missing in the unknown position
#  which seems good, but protein_letters_alph1 puts in a value that is not expected to be possible in an array of only
#  these letters
numerical_translation_alph1 = defaultdict(lambda: 20, zip(protein_letters_alph1, count()))
numerical_translation_alph3 = defaultdict(lambda: 20, zip(protein_letters_alph3, count()))
numerical_translation_alph1_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1), count()))
numerical_translation_alph3_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3), count()))
sequence_translation_alph1 = defaultdict(lambda: '-', zip(count(), protein_letters_alph1))
sequence_translation_alph3 = defaultdict(lambda: '-', zip(count(), protein_letters_alph3))
numerical_translation_alph1_gaped = defaultdict(lambda: 20, zip(protein_letters_alph1_gaped, count()))
numerical_translation_alph3_gaped = defaultdict(lambda: 20, zip(protein_letters_alph3_gaped, count()))
numerical_translation_alph1_gaped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_gaped), count()))
numerical_translation_alph3_gaped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_gaped), count()))
numerical_translation_alph1_unknown_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_unknown), count()))
numerical_translation_alph3_unknown_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_unknown), count()))
numerical_translation_alph1_unknown_gaped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_unknown_gaped), count()))
numerical_translation_alph3_unknown_gaped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_unknown_gaped), count()))
alphabet_types_literal = Literal[
    'protein_letters_alph1', 'protein_letters_alph3', 'protein_letters_alph1_gaped',
    'protein_letters_alph3_gaped', 'protein_letters_alph1_unknown', 'protein_letters_alph3_unknown',
    'protein_letters_alph1_unknown_gaped', 'protein_letters_alph3_unknown_gaped']
alphabet_types: tuple[str, ...] = get_args(alphabet_types_literal)
alphabets_literal = Literal[
    'ACDEFGHIKLMNPQRSTVWY', 'ARNDCQEGHILKMFPSTWYV', 'ACDEFGHIKLMNPQRSTVWY-', 'ARNDCQEGHILKMFPSTWYV-',
    'ACDEFGHIKLMNPQRSTVWYX', 'ARNDCQEGHILKMFPSTWYVX', 'ACDEFGHIKLMNPQRSTVWYX-', 'ARNDCQEGHILKMFPSTWYVX-'
]
alphabets: tuple[str, ...] = get_args(alphabets_literal)
alphabet_to_alphabet_type = dict(zip(alphabets, alphabet_types))
alphabet_type_to_alphabet = dict(zip(alphabet_types, alphabets))
alignment_programs_literal = Literal['hhblits', 'psiblast']
alignment_programs: tuple[str, ...] = get_args(alignment_programs_literal)
profile_types = Literal['evolutionary', 'fragment', '']
optimization_species_literal = Literal[
    'b_subtilis', 'c_elegans', 'd_melanogaster', 'e_coli', 'g_gallus', 'h_sapiens', 'm_musculus',
    'm_musculus_domesticus', 's_cerevisiae']
# optimization_species = get_args(optimization_species_literal)


class AminoAcidDistribution(TypedDict):
    A: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    I: float
    K: float
    L: float
    M: float
    N: float
    P: float
    Q: float
    R: float
    S: float
    T: float
    V: float
    W: float
    Y: float
    # B: float
    # J: float
    # O: float
    # U: float
    # X: float
    # Z: float


profile_keys = Literal[protein_letters_literal, 'lod', 'type', 'info', 'weight']
ProfileEntry = TypedDict(
    'ProfileEntry', {**AminoAcidDistribution.__dict__['__annotations__'], 'lod': AminoAcidDistribution,
                     'type': protein_letters_literal, 'info': float, 'weight': float},
    total=False
)
ProfileDict = dict[int, ProfileEntry]
"""{1: {'A': 0.04, 'C': 0.12, ..., 'lod': {'A': -5, 'C': -9, ...},
        'type': 'W', 'info': 0.00, 'weight': 0.00}, {...}}
"""


def create_numeric_translation_table(alphabet: Sequence[str], bytes_: bool = True) -> dict[bytes | str, int]:
    """Return the numeric translation from an alphabet to the integer position in that alphabet

    Args:
        alphabet: The alphabet to use. Example 'ARNDCQEGHILKMFPSTWYVBZX*'
        bytes_: Whether to map from byte characters
    Returns:
        The mapping from the character to the positional integer
    """
    if bytes_:
        alphabet = (char.encode() for char in alphabet)

    return dict(zip(alphabet, count()))


def get_numeric_translation_table(alphabet_type: alphabet_types_literal) -> defaultdict[str, int] | dict[str, int]:
    """Given an amino acid alphabet type, return the corresponding numerical translation table.
    If a table is passed, just return it

    Returns:
        The integer mapping to the sequence of the requested alphabet
    """
    # try:
    #     match alphabet_type:
    #         case 'protein_letters_alph1':
    #             numeric_translation_table = numerical_translation_alph1_bytes
    #         case 'protein_letters_alph3':
    #             numeric_translation_table = numerical_translation_alph3_bytes
    #         case 'protein_letters_alph1_gaped':
    #             numeric_translation_table = numerical_translation_alph1_gaped_bytes
    #         case 'protein_letters_alph3_gaped':
    #             numeric_translation_table = numerical_translation_alph3_gaped_bytes
    #         case 'protein_letters_alph1_unknown':
    #             numeric_translation_table = numerical_translation_alph1_unknown_bytes
    #         case 'protein_letters_alph3_unknown':
    #             numeric_translation_table = numerical_translation_alph3_unknown_bytes
    #         case 'protein_letters_alph1_unknown_gaped':
    #             numeric_translation_table = numerical_translation_alph1_unknown_gaped_bytes
    #         case 'protein_letters_alph3_unknown_gaped':
    #             numeric_translation_table = numerical_translation_alph3_unknown_gaped_bytes
    #         case _:
    #             try:  # To see if we already have the alphabet, and just return defaultdict
    #                 numeric_translation_table = alphabet_to_type[alphabet_type]
    #             except KeyError:
    #                 # raise ValueError(wrong_alphabet_type)
    #                 logger.warning(wrong_alphabet_type)
    #                 numeric_translation_table = create_numeric_translation_table(alphabet_type)
    # except SyntaxError:  # python version not 3.10
    if alphabet_type == 'protein_letters_alph1':
        numeric_translation_table = numerical_translation_alph1_bytes
    elif alphabet_type == 'protein_letters_alph3':
        numeric_translation_table = numerical_translation_alph3_bytes
    elif alphabet_type == 'protein_letters_alph1_gaped':
        numeric_translation_table = numerical_translation_alph1_gaped_bytes
    elif alphabet_type == 'protein_letters_alph3_gaped':
        numeric_translation_table = numerical_translation_alph3_gaped_bytes
    elif alphabet_type == 'protein_letters_alph1_unknown':
        numeric_translation_table = numerical_translation_alph1_unknown_bytes
    elif alphabet_type == 'protein_letters_alph3_unknown':
        numeric_translation_table = numerical_translation_alph3_unknown_bytes
    elif alphabet_type == 'protein_letters_alph1_unknown_gaped':
        numeric_translation_table = numerical_translation_alph1_unknown_gaped_bytes
    elif alphabet_type == 'protein_letters_alph3_unknown_gaped':
        numeric_translation_table = numerical_translation_alph3_unknown_gaped_bytes
    else:
        try:  # To see if we already have the alphabet, and return the defaultdict
            _type = alphabet_to_alphabet_type[alphabet_type]
        except KeyError:
            raise KeyError(
                f"The alphabet '{alphabet_type}' isn't an allowed alphabet_type."
                f" See {', '.join(alphabet_types)}")
            # raise ValueError(wrong_alphabet_type)
        logger.warning(f"{get_numeric_translation_table.__name__}: The alphabet_type '{alphabet_type}' "
                       "isn't viable. Attempting to create it")
        numeric_translation_table = create_numeric_translation_table(alphabet_type)

    return numeric_translation_table


default_substitution_matrix_name = 'BLOSUM62'
_substitution_matrices_cache = \
    {default_substitution_matrix_name: substitution_matrices.load(default_substitution_matrix_name)}
default_substitution_matrix_ = _substitution_matrices_cache.get(default_substitution_matrix_name)
default_substitution_matrix_translation_table = create_numeric_translation_table(default_substitution_matrix_.alphabet)
default_substitution_matrix_array = np.array(default_substitution_matrix_)
# Alignment = namedtuple('Alignment', 'seqA, seqB, score, start, end')


# def generate_alignment(seq1: Sequence[str], seq2: Sequence[str], matrix: str = 'BLOSUM62', local: bool = False,
#                        top_alignment: bool = True) -> Alignment | list[Alignment]:
#     """Use Biopython's pairwise2 to generate a sequence alignment
#
#     Args:
#         seq1: The first sequence to align
#         seq2: The second sequence to align
#         matrix: The matrix used to compare character similarities
#         local: Whether to run a local alignment. Only use for generally similar sequences!
#         top_alignment: Only include the highest scoring alignment
#     Returns:
#         The resulting alignment
#     """
#     if local:
#         _type = 'local'
#     else:
#         _type = 'global'
#     _matrix = _substitution_matrices_cache.get(matrix, substitution_matrices.load(matrix))
#     gap_penalty = -10
#     gap_ext_penalty = -1
#     # logger.debug(f'Generating sequence alignment between:\n{seq1}\n\tAND:\n{seq2}')
#     # Create sequence alignment
#     align = getattr(pairwise2.align, f'{_type}ds')(seq1, seq2, _matrix, gap_penalty, gap_ext_penalty,
#                                                    one_alignment_only=top_alignment)
#     logger.debug(f'Generated alignment:\n{pairwise2.format_alignment(*align[0])}')
#
#     return align[0] if top_alignment else align

# Set these the default from blastp
blastp_open_gap_score = -12.  # gap_penalty
blastp_extend_gap_score = -1.  # gap_ext_penalty
protein_alignment_variables = dict(
    query_left_open_gap_score=0,
    query_left_extend_gap_score=0,
    target_left_open_gap_score=0,
    target_left_extend_gap_score=0,
    query_internal_open_gap_score=blastp_open_gap_score,
    query_internal_extend_gap_score=blastp_extend_gap_score,
    target_internal_open_gap_score=blastp_open_gap_score,
    target_internal_extend_gap_score=blastp_extend_gap_score,
    query_right_open_gap_score=0,
    query_right_extend_gap_score=0,
    target_right_open_gap_score=0,
    target_right_extend_gap_score=0,
)


def generate_alignment(seq1: Sequence[str], seq2: Sequence[str], matrix: str = default_substitution_matrix_name,
                       local: bool = False, top_alignment: bool = True, **alignment_kwargs) \
        -> Alignment | PairwiseAlignments:
    """Use Biopython's pairwise2 to generate a sequence alignment

    Args:
        seq1: The first sequence to align
        seq2: The second sequence to align
        matrix: The matrix used to compare character similarities
        local: Whether to run a local alignment. Only use for generally similar sequences!
        top_alignment: Only include the highest scoring alignment
    Keyword Args:
        query_left_open_gap_score: int = 0 - The score used for opening a gap in the alignment procedure
        query_left_extend_gap_score: int = 0 - The score used for extending a gap in the alignment procedure
        target_left_open_gap_score: int = 0 - The score used for opening a gap in the alignment procedure
        target_left_extend_gap_score: int = 0 - The score used for extending a gap in the alignment procedure
        query_internal_open_gap_score: int = -12 - The score used for opening a gap in the alignment procedure
        query_internal_extend_gap_score: int = -1 - The score used for extending a gap in the alignment procedure
        target_internal_open_gap_score: int = -12 - The score used for opening a gap in the alignment procedure
        target_internal_extend_gap_score: int = -1 - The score used for extending a gap in the alignment procedure
        query_right_open_gap_score: int = 0 - The score used for opening a gap in the alignment procedure
        query_right_extend_gap_score: int = 0 - The score used for extending a gap in the alignment procedure
        target_right_open_gap_score: int = 0 - The score used for opening a gap in the alignment procedure
        target_right_extend_gap_score: int = 0 - The score used for extending a gap in the alignment procedure
    Returns:
        The resulting alignment(s). Will be an Alignment object if top_alignment is True else PairwiseAlignments object
    """
    matrix_ = _substitution_matrices_cache.get(matrix)
    if matrix_ is None:
        try:  # To get the new matrix and store for future ops
            matrix_ = _substitution_matrices_cache[matrix] = substitution_matrices.load(matrix)
        except FileNotFoundError:  # Missing this
            raise KeyError(
                f"Couldn't find the substitution matrix '{matrix}' ")

    if local:
        mode = 'local'
    else:
        mode = 'global'

    if alignment_kwargs:
        protein_alignment_variables_ = protein_alignment_variables.copy()
        protein_alignment_variables_.update(**alignment_kwargs)
    else:
        protein_alignment_variables_ = protein_alignment_variables
    # logger.debug(f'Generating sequence alignment between:\n{seq1}\n\tAND:\n{seq2}')
    # Create sequence alignment
    aligner = PairwiseAligner(mode=mode, substitution_matrix=matrix_, **protein_alignment_variables_)
    try:
        alignments = aligner.align(seq1, seq2)
    except ValueError:  # sequence contains letters not in the alphabet
        print(f'Sequence1: {seq1}')
        print(f'Sequence2: {seq2}')
        raise
    first_alignment = alignments[0]
    logger.debug(f'Found alignment with score: {alignments.score}\n{first_alignment}')
    # print("Number of alignments: %d" % len(alignments))
    #   Number of alignments: 1
    # alignment = alignments[0]
    # print("Score = %.1f" % alignment.score)
    #   Score = 13.0
    # print(alignment)
    #   target            0 KEVLA 5
    #                     0 -|||- 5
    #   query             0 -EVL- 3
    # print(alignment.target)
    # print(alignment.query)
    # print(alignment.indices)  # returns array([[ 0,  1,  2,  3,  4], [ -1,  0,  1,  2, -1]]) where -1 are gaps
    # This would be useful in checking for gaps during generate_mutations()
    # print(alignment.inverse_indices)  # returns [array([ 0,  1,  2,  3,  4], [ 1,  2,  3]])
    # where -1 are outside array and each index is the position in the alignment. This would be useful for instance with
    # get_equivalent_indices() which is precalculated and now does this routine twice during Entity.__init__()
    # logger.debug(f'Generated alignment:\n{pairwise2.format_alignment(*align[0])}')

    # return align[0] if top_alignment else align
    return first_alignment if top_alignment else alignments


def read_fasta_file(file_name: AnyStr, **kwargs) -> Iterable[SeqRecord]:
    """Opens a fasta file and return a parser object to load the sequences to SeqRecords

    Args:
        file_name: The location of the file on disk
    Returns:
        An iterator of the sequences in the file [record1, record2, ...]
    """
    return SeqIO.parse(file_name, 'fasta')


def read_sequence_file(file_name: AnyStr, **kwargs) -> Iterable[SeqRecord]:
    """Opens a fasta file and return a parser object to load the sequences to SeqRecords

    Args:
        file_name: The location of the file on disk
    Returns:
        An iterator of the sequences in the file [record1, record2, ...]
    """
    raise NotImplementedError()
    return SeqIO.parse(file_name, 'csv')


@utils.handle_errors(errors=(FileNotFoundError,))
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
    """Write a fasta file from sequence(s). If a single sequence is provided, pass as a string

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
            raise ValueError(
                f"Must provide argument 'file_name' or 'names' as a str to {write_sequences.__name__}()")

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
        return f"{write_sequences.__name__} can't parse data to make fasta\n" \
               f'names={names}, sequences={sequences}, extension={extension}, out_path={out_path}, ' \
               f'file_name={file_name}'

    with open(file_name, 'a') as outfile:
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()

        if not sequences:
            raise ValueError(f"No sequences provided, couldn't write anything")

        if isinstance(sequences, list):
            if isinstance(sequences[0], tuple):  # Where seq[0] is name, seq[1] is seq
                formatted_sequence_gen = (f'{start}{name}{sep}{seq}' for name, seq, *_ in sequences)
            elif isinstance(names, Sequence):
                if isinstance(sequences[0], str):
                    formatted_sequence_gen = (f'{start}{name}{sep}{seq}' for name, seq in zip(names, sequences))
                elif isinstance(sequences[0], Sequence):
                    formatted_sequence_gen = \
                        (f'{start}{name}{sep}{"".join(seq)}' for name, seq in zip(names, sequences))
                else:
                    raise TypeError(data_dump())
            # elif isinstance(sequences[0], list):  # Where interior list is alphabet (AA or DNA)
            #     for idx, seq in enumerate(sequences):
            #         outfile.write(f'>{name}_{idx}\n')  # Write header
            #         # Check if alphabet is 3 letter protein
            #         outfile.write(f'{" ".join(seq)}\n' if len(seq[0]) == 3 else f'{"".join(seq)}\n')
            # elif isinstance(sequences[0], str):  # Likely 3 aa format...
            #     outfile.write(f'>{name}\n{" ".join(sequences)}\n')
            else:
                raise TypeError(data_dump())
        elif isinstance(sequences, dict):
            formatted_sequence_gen = (f'{start}{name}{sep}{"".join(seq)}' for name, seq in sequences.items())
        elif isinstance(sequences, tuple):  # Where seq[0] is name, seq[1] is seq
            try:
                name, seq, *_ = sequences
            except ValueError:
                raise ValueError(f"When using a tuple, expected that the tuple contain (name, sequence) pairs")
            formatted_sequence_gen = (f'{start}{name}{sep}{seq}',)
        elif isinstance(names, str):  # Assume sequences is a str or tuple
            formatted_sequence_gen = (f'{start}{names}{sep}{"".join(sequences)}',)
        else:
            raise TypeError(data_dump())
        outfile.write('%s\n' % '\n'.join(formatted_sequence_gen))

    return file_name


hhblits_threads = 2


def hhblits(name: str, sequence_file: Sequence[str] = None, sequence: Sequence[str] = None,
            out_dir: AnyStr = os.getcwd(), threads: int = hhblits_threads,
            return_command: bool = False, **kwargs) -> list[str] | None:
    """Generate a position specific scoring matrix from HHblits using Hidden Markov Models

    When the command is run, it is possible to create six files in out_dir with the pattern '/outdir/name.*'
    where the '*' extensions are:
     hhblits profile - .hmm

     hhblits resulting cluster description - .hrr

     sequence alignment in a3m format - .a3m

     sequence file - .seq (if sequence_file)

     sequence alignment in stockholm format - .sto (if return_command is False)

     sequence alignment in fasta format - .fasta (if return_command is False)

    Args:
        name: The name associated with the sequence
        sequence_file: The file containing the sequence to use
        sequence: The sequence to use. Used in place of sequence_file
        out_dir: Disk location where generated files should be written
        threads: Number of cpu's to use for the process
        return_command: Whether to simply return the hhblits command
    Raises:
        RuntimeError if hhblits command is run and returns a non-zero exit code
    Returns:
        The command if return_command is True, otherwise None
    """
    if sequence_file is None:
        if sequence is None:
            raise ValueError(
                f"{hhblits.__name__}: Can't proceed without argument 'sequence_file' or 'sequence'")
        else:
            sequence_file = write_sequences((name, sequence), file_name=os.path.join(out_dir, f'{name}.seq'))
    pssm_file = os.path.join(out_dir, f'{name}.hmm')
    a3m_file = os.path.join(out_dir, f'{name}.a3m')
    # Todo for higher performance set up https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
    #  Create a ramdisk to store a database chunk to make hhblits/Jackhmmer run fast.
    #  sudo mkdir -m 777 --parents /tmp/ramdisk
    #  sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk
    cmd = [putils.hhblits_exe, '-d', putils.uniclust_db, '-i', sequence_file, '-ohhm', pssm_file, '-oa3m', a3m_file,
           '-hide_cons', '-hide_pred', '-hide_dssp', '-E', '1E-06', '-v', '1', '-cpu', str(threads)]

    if return_command:
        return cmd

    logger.debug(f'{name} Profile Command: {subprocess.list2cmdline(cmd)}')
    p = subprocess.Popen(cmd)
    p.communicate()
    if p.returncode != 0:
        # temp_file = os.path.join(out_path, f'{self.name}.hold')
        temp_file = Path(out_dir, f'{name}.hold')
        temp_file.unlink(missing_ok=True)
        # if os.path.exists(temp_file):  # remove hold file blocking progress
        #     os.remove(temp_file)
        raise utils.SymDesignException(
            f'Profile generation for {name} got stuck. Found return code {p.returncode}')  #

    # Preferred alignment type
    msa_file = os.path.join(out_dir, f'{name}.sto')
    p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, msa_file, '-num', '-uc'])
    p.communicate()
    fasta_msa = os.path.join(out_dir, f'{name}.fasta')
    p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, fasta_msa, '-M', 'first', '-r'])
    p.communicate()

    return None


def optimize_protein_sequence(sequence: str, species: optimization_species_literal = 'e_coli') -> str:
    """Optimize a sequence for expression in a desired organism

    Args:
        sequence: The sequence of interest
        species: The species context to optimize nucleotide sequence usage
    Returns:
        The input sequence optimized to nucleotides for expression considerations
    """
    seq_length = len(sequence)
    species = species.lower()
    try:
        from symdesign.third_party.DnaChisel.dnachisel import reverse_translate, DnaOptimizationProblem, \
            CodonOptimize, EnforceGCContent, AvoidHairpins, AvoidPattern, UniquifyAllKmers, AvoidRareCodons, \
            EnforceTranslation
    except ModuleNotFoundError:
        raise RuntimeError(
            f"Can't {optimize_protein_sequence.__name__} as the dependency DnaChisel is not available")

    try:
        dna_sequence = reverse_translate(sequence)
    except KeyError as error:
        raise KeyError(
            f'Warning an invalid character was found in the protein sequence: {error}')

    problem = DnaOptimizationProblem(
        sequence=dna_sequence, logger=None,  # max_random_iters=20000,
        objectives=[CodonOptimize(species=species)],
        # method='harmonize_rca')] <- Useful for folding speed when original organism known
        constraints=[EnforceGCContent(mini=0.25, maxi=0.65),  # Twist required
                     EnforceGCContent(mini=0.35, maxi=0.65, window=50),  # Twist required
                     AvoidHairpins(stem_size=20, hairpin_window=48),  # Efficient translate
                     AvoidPattern('GGAGG', location=(1, seq_length, 1)),  # Ribosome bind
                     AvoidPattern('TAAGGAG', location=(1, seq_length, 1)),  # Ribosome bind
                     AvoidPattern('AAAAA', location=(1, seq_length, 0)),  # Terminator
                     # AvoidPattern('TTTTT', location=(1, seq_length, 1)),  # Terminator
                     AvoidPattern('GGGGGGGGGG', location=(1, seq_length, 0)),  # Homopoly
                     # AvoidPattern('CCCCCCCCCC', location=(1, seq_length)),  # Homopoly
                     UniquifyAllKmers(20),  # Twist required
                     AvoidRareCodons(0.08, species=species),
                     EnforceTranslation(),
                     # EnforceMeltingTemperature(mini=10,maxi=62,location=(1, seq_length)),
                     ])

    # Solve constraints and solve in regard to the objective
    problem.max_random_iters = 20000
    problem.resolve_constraints()
    problem.optimize()

    # Display summaries of constraints that pass
    # print(problem.constraints_text_summary())
    # print(problem.objectives_text_summary())

    # Get final sequence as string
    final_sequence = problem.sequence
    # Get final sequene as BioPython record
    # final_record = problem.to_record(with_sequence_edits=True)

    return final_sequence


def make_mutations(sequence: Sequence, mutations: dict[int, dict[str, str]], find_orf: bool = True) -> str:
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Assumes a zero-index sequence and zero-index mutations
    Args:
        sequence: 'Wild-type' sequence to mutate
        mutations: {mutation_index: {'from': AA, 'to': AA}, ...}
        find_orf: Whether to find the correct ORF for the mutations and the seq
    Returns:
        seq: The mutated sequence
    """
    # Seq can be either list or string
    if find_orf:
        offset = find_orf_offset(sequence, mutations)
        logger.info(f'Found ORF. Offset = {offset}')
    else:
        offset = 0  # zero_offset

    seq = sequence
    index_errors = []
    for key, mutation in mutations.items():
        index = key + offset
        try:
            if seq[index] == mutation['from']:  # Adjust key for zero index slicing
                seq = seq[:index] + mutation['to'] + seq[index + 1:]
            else:  # Find correct offset, or mark mutation source as doomed
                index_errors.append(key)
        except IndexError:
            logger.error(key + offset)
    if index_errors:
        logger.warning(f'{make_mutations.__name__} index errors: {", ".join(map(str, index_errors))}')

    return seq


def find_orf_offset(sequence: Sequence, mutations: mutation_dictionary) -> int:
    """Using a sequence and mutation data, find the open reading frame that matches mutations closest

    Args:
        sequence: Sequence to search for ORF in 1 letter format
        mutations: {mutation_index: {'from': AA, 'to': AA}, ...} zero-indexed sequence dictionary
    Returns:
        The zero-indexed integer to offset the provided sequence to best match the provided mutations
    """
    def gen_offset_repr():
        return f'Found the orf_offsets: {",".join(f"{k}={v}" for k, v in orf_offsets.items())}'

    unsolvable = False
    orf_start_idx = 0
    orf_offsets = {idx: 0 for idx, aa in enumerate(sequence) if aa == 'M'}
    methionine_positions = list(orf_offsets.keys())
    while True:
        if not orf_offsets:  # MET is missing for the sequnce/we haven't found the ORF start and need to scan a range
            orf_offsets = {start_idx: 0 for start_idx in range(-30, 50)}

        # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
        for test_orf_index in orf_offsets:
            for mutation_index, mutation in mutations.items():
                try:
                    if sequence[test_orf_index + mutation_index] == mutation['from']:
                        orf_offsets[test_orf_index] += 1
                except IndexError:  # We have reached the end of the sequence
                    break

        # logger.debug(gen_offset_repr())
        max_count = max(list(orf_offsets.values()))
        # Check if likely ORF has been identified (count < number mutations/2). If not, MET is missing/not the ORF start
        if max_count < len(mutations) / 2:
            if unsolvable:
                raise RuntimeError(f"Couldn't find a orf_offset max_count {max_count} < {len(mutations) / 2} (half the "
                                   f"mutations). The orf_start_idx={orf_start_idx} still\n\t{gen_offset_repr()}")
            orf_offsets = {}
            unsolvable = True  # if we reach this spot again, the problem is deemed unsolvable
        else:  # Find the index of the max_count
            for idx, count_ in orf_offsets.items():
                if max_count == count_:  # orf_offsets[offset]:
                    orf_start_idx = idx  # Select the first occurrence of the max count
                    break
            else:
                raise RuntimeError(f"Couldn't find a orf_offset count == {max_count} (the max_count). "
                                   f"The orf_start_idx={orf_start_idx} still\n\t{gen_offset_repr()}")

            # For cases where the orf doesn't begin on Met, try to find a prior Met. Otherwise, selects the id'd Met
            closest_met = None
            for met_index in methionine_positions:
                if met_index <= orf_start_idx:
                    closest_met = met_index
                else:  # We have passed the identified orf_start_idx
                    if closest_met is not None:
                        orf_start_idx = closest_met  # + zero_offset # change to one-index
                    break
            break

    return orf_start_idx


# class MutationEntry(TypedDict):
#     to: protein_letters_alph3_gaped_literal
#     from: protein_letters_alph3_gaped_literal
MutationEntry = TypedDict('MutationEntry', {'to': protein_letters_literal,
                                            'from': protein_letters_literal})
# mutation_entry = dict[Literal['to', 'from'], protein_letters_alph3_gaped_literal]
# mutation_entry = Type[dict[Literal['to', 'from'], protein_letters_alph3_gaped_literal]]
"""Mapping of a reference sequence amino acid type, 'to', and the resulting sequence amino acid type, 'from'"""
mutation_dictionary = dict[int, MutationEntry]
"""The mapping of a residue number to a mutation entry containing the reference, 'to', and sequence, 'from', amino acid 
type
"""
sequence_dictionary = dict[int, protein_letters_literal]
"""The mapping of a residue number to the corresponding amino acid type"""


def generate_mutations(reference: Sequence, query: Sequence, offset: bool = True, keep_gaps: bool = False,
                       remove_termini: bool = True, remove_query_gaps: bool = True, only_gaps: bool = False,
                       zero_index: bool = False,
                       return_all: bool = False, return_to: bool = False, return_from: bool = False) \
        -> mutation_dictionary | sequence_dictionary:
    """Create mutation data in a typical A5K format. Integer indexed dictionary keys with the index matching reference
    sequence. Sequence mutations accessed by "from" and "to" keys. By default, only mutated positions are returned and
    all gaped sequences are excluded

    For PDB comparison, reference should be expression sequence (SEQRES), query should be atomic sequence (ATOM)

    Args:
        reference: Reference sequence to align mutations against. Character values are returned to the "from" key
        query: Query sequence. Character values are returned to the "to" key
        offset: Whether sequences are different lengths. Will create an alignment of the two sequences
        keep_gaps: Return gaped indices, i.e. outside the aligned sequences or missing internal characters
        remove_termini: Remove indices that are outside the reference sequence boundaries
        remove_query_gaps: Remove indices where there are gaps present in the query sequence
        only_gaps: Only include reference indices that are missing query residues. All "to" values will be a gap "-"
        zero_index: Whether to return the indices zero-indexed (like python) or one-indexed
        return_all: Whether to return all the indices and there corresponding mutational data
        return_to: Whether to return only the 'to' amino acid type
        return_from: Whether to return only the 'from' amino acid type
    Returns:
        Mutation index to mutations with format
            {1: {'from': 'A', 'to': 'K'}, ...}
            unless return_to or return_from is True, then
            {1: 'K', ...} or {1: 'A', ...}, respectively
    """
    if offset:
        alignment = generate_alignment(reference, query)
        # # numeric_to_sequence()
        # seq_indices1, seq_indices2 = alignment.indices
        # seq_gaps1 = seq_indices1 == -1
        # seq_gaps2 = seq_indices2 == -1
        # align_seq_1 = alignment.target
        # align_seq_2 = alignment.query
        align_seq_1, align_seq_2 = alignment
        # # align_seq_1, align_seq_2 = alignment.sequences
        # # align_seq_1, align_seq_2, *_ = generate_alignment(reference, query)
    else:
        align_seq_1, align_seq_2 = reference, query

    idx_offset = 0 if zero_index else zero_offset

    # Get the first matching index of the reference sequence
    starting_idx_of_seq1 = align_seq_1.find(reference[0])
    # Ensure iteration sequence1/reference starts at idx 0       v
    sequence_iterator = enumerate(zip(align_seq_1, align_seq_2), -starting_idx_of_seq1 + idx_offset)
    # Extract differences from the alignment
    if return_all:
        mutations = {idx: {'from': char1, 'to': char2} for idx, (char1, char2) in sequence_iterator}
    else:
        mutations = {idx: {'from': char1, 'to': char2} for idx, (char1, char2) in sequence_iterator if char1 != char2}

    # Find last index of reference (including internal gaps)
    starting_key_of_seq1 = idx_offset
    ending_key_of_seq1 = starting_key_of_seq1 + align_seq_1.rfind(reference[-1])
    remove_mutation_list = []
    if only_gaps:  # Remove the actual mutations, keep internal and external gap indices and the reference sequence
        keep_gaps = True
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if mutation['from'] != '-' and mutation['to'] != '-'])
    if keep_gaps:  # Leave all types of keep_gaps, otherwise check for each requested type
        remove_termini = remove_query_gaps = False

    if remove_termini:  # Remove indices outside of sequence 1
        remove_mutation_list.extend([entry for entry in mutations
                                     if entry < starting_key_of_seq1 or entry > ending_key_of_seq1])

    if remove_query_gaps:  # Remove indices where sequence 2 is gaped
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if starting_key_of_seq1 < entry <= ending_key_of_seq1 and mutation['to'] == '-'])
    for entry in remove_mutation_list:
        mutations.pop(entry, None)

    if return_to:
        mutations = {idx: _mutation_dictionary['to'] for idx, _mutation_dictionary in mutations.items()}
    elif return_from:
        mutations = {idx: _mutation_dictionary['from'] for idx, _mutation_dictionary in mutations.items()}

    return mutations


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
        offset: bool = True - Whether sequences are different lengths. Will create an alignment of the two sequences
        keep_gaps: bool = False - Return gaped indices, i.e. outside the aligned sequences or missing internal
            characters
        remove_termini: bool = True - Remove indices that are outside the reference sequence boundaries
        remove_query_gaps: bool = True - Remove indices where there are gaps present in the query sequence
        only_gaps: bool = False - Only include reference indices that are missing query residues.
            All "to" values will be a gap "-"
        zero_index: bool = False - Whether to return the indices zero-indexed (like python Sequence) or one-indexed
        return_all: bool = False - Whether to return all the indices and there corresponding mutational data
        return_to: bool = False - Whether to return only the "to" amino acid type
        return_from: bool = False - Whether to return only the "from" amino acid type
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


def numeric_to_sequence(numeric_sequence: np.ndarray, translation_table: dict[str, int] = None,
                        alphabet_order: int = 1) -> np.ndarray:
    """Convert a numeric sequence array into a sequence array

    Args:
        numeric_sequence: The sequence to convert
        translation_table: If a translation table is provided, it will be used. If not, use alphabet_order
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3
    Returns:
        The alphabetically encoded sequence where each entry along axis=-1 is the one letter amino acid
    """
    if translation_table is not None:
        return np.vectorize(translation_table.__getitem__)(numeric_sequence)
    else:
        if alphabet_order == 1:
            return np.vectorize(sequence_translation_alph1.__getitem__)(numeric_sequence)
        elif alphabet_order == 3:
            return np.vectorize(sequence_translation_alph3.__getitem__)(numeric_sequence)
        else:
            raise ValueError(f"The 'alphabet_order' {alphabet_order} isn't valid. Choose from either 1 or 3")


def get_equivalent_indices(target: Sequence = None, query: Sequence = None, mutation_allowed: bool = False) \
        -> tuple[list[int], list[int]]:
    """From two sequences, find the indices where both sequences are equal

    Args:
        target: The first sequence to compare
        query: The second sequence to compare
        mutation_allowed: Whether equivalent indices can exist at mutation sites
    Returns:
        The pair of indices where the sequences align.
            Ex: sequence1 = A B C D E F ...
                sequence2 = A B - D E F ...
            returns        [0,1,  3,4,5, ...],
                           [0,1,  2,3,4, ...]
    """
    # alignment: Sequence = None
    #     alignment: An existing Bio.Align.Alignment object
    # if alignment is None:
    if target is not None and query is not None:
        # # Get all mutations from the alignment of sequence1 and sequence2
        mutations = generate_mutations(target, query, keep_gaps=True, return_all=True)
        # alignment = generate_alignment(target, query)
        # alignment.inverse_indices
        # return
    else:
        raise ValueError(f"Can't {get_equivalent_indices.__name__} without passing either 'alignment' or "
                         f"'target' and 'query'")
    # else:  # Todo this may not be ever useful since the alignment needs to go into the generate_mutations()
    #     raise NotImplementedError(f"Set {get_equivalent_indices.__name__} up with an Alignment object from Bio.Align")

    target_mutations = ''.join([mutation['to'] for mutation in mutations.values()])
    query_mutations = ''.join([mutation['from'] for mutation in mutations.values()])
    logger.debug(f"Sequence info:\ntarget :{target_mutations}\nquery  :{query_mutations}")
    # Get only those indices where there is an aligned aa on the opposite chain
    sequence1_indices, sequence2_indices = [], []
    # to_idx = from_idx = 0
    to_idx, from_idx = count(), count()
    # sequence1 'from' is fixed, sequence2 'to' is moving
    for mutation_idx, mutation in enumerate(mutations.values()):
        if mutation['to'] == mutation['from']:  # They are equal
            sequence1_indices.append(next(from_idx))  # from_idx)
            sequence2_indices.append(next(to_idx))  # to_idx)
        elif mutation['from'] == '-':  # increment to_idx/fixed_idx
            # to_idx += 1
            next(to_idx)
        elif mutation['to'] == '-':  # increment from_idx/moving_idx
            # from_idx += 1
            next(from_idx)
        elif mutation['to'] != mutation['from']:
            if mutation_allowed:
                sequence1_indices.append(next(from_idx))
                sequence2_indices.append(next(to_idx))
            else:
                next(from_idx)
                next(to_idx)
            # to_idx += 1
            # from_idx += 1
        else:  # What else is there
            target_mutations = ''.join([mutation['to'] for mutation in mutations.values()])
            query_mutations = ''.join([mutation['from'] for mutation in mutations.values()])
            raise RuntimeError(f"This should never be reached. Ran into error at index {mutation_idx}:\n"
                               f"{mutation}\nSequence info:\ntarget :{target_mutations}\nquery  :{query_mutations}")
        # else:  # They are equal
        #     sequence1_indices.append(next(from_idx))  # from_idx)
        #     sequence2_indices.append(next(to_idx))  # to_idx)
        #     # to_idx += 1
        #     # from_idx += 1

    return sequence1_indices, sequence2_indices


latest_uniclust_background_frequencies = \
    {'A': 0.0835, 'C': 0.0157, 'D': 0.0542, 'E': 0.0611, 'F': 0.0385, 'G': 0.0669, 'H': 0.0228, 'I': 0.0534,
     'K': 0.0521, 'L': 0.0926, 'M': 0.0219, 'N': 0.0429, 'P': 0.0523, 'Q': 0.0401, 'R': 0.0599, 'S': 0.0791,
     'T': 0.0584, 'V': 0.0632, 'W': 0.0127, 'Y': 0.0287}


def get_lod(frequencies: dict[protein_letters_literal, float],
            background: dict[protein_letters_literal, float], as_int: bool = True) -> dict[str, int | float]:
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
        try:  # Todo why is this 2. * the log2? I believe this is a heuristic of BLOSUM62. This should be removed...
            lods[aa] = float(2. * log2(freq / background[aa]))  # + 0.0
        except ValueError:  # math domain error
            lods[aa] = -9
        except KeyError:
            if aa in protein_letters_alph1:
                raise KeyError(f'{aa} was not in the background frequencies: {", ".join(background)}')
            else:  # We shouldn't worry about a missing value if it's not an amino acid
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
                dict(zip(protein_letters_alph3,
                         [x / 100. for x in map(int, line_data[22:len(
                             protein_letters_alph3) + 22])]))
            # pose_dict[residue_number] = aa_counts_alph3.copy()
            # for i, aa in enumerate(protein_letters_alph3, 22):
            #     # Get normalized counts for pose_dict
            #     pose_dict[residue_number][aa] = int(line_data[i]) / 100.

            # for i, aa in enumerate(protein_letters_alph3, 2):
            #     pose_dict[residue_number]['lod'][aa] = line_data[i]
            pose_dict[residue_number]['lod'] = \
                dict(zip(protein_letters_alph3, line_data[2:len(
                    protein_letters_alph3) + 2]))
            pose_dict[residue_number]['type'] = line_data[1]
            pose_dict[residue_number]['info'] = float(line_data[42])
            pose_dict[residue_number]['weight'] = float(line_data[43])

    return pose_dict


def parse_hhblits_pssm(file: AnyStr, null_background: bool = True, **kwargs) -> ProfileDict:
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
                    null_bg = {aa: to_freq(value) for value, aa in zip(background_values,
                                                                       protein_letters_alph3)}

            if len(line.split()) == 23:
                residue_type, residue_number, *position_values = line.strip().split()
                aa_freqs = {aa: to_freq(value) for value, aa in zip(position_values,
                                                                    protein_letters_alph1)}

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
    #                 null_bg = {i: {} for i in protein_letters_alph3}
    #                 for i, aa in enumerate(protein_letters_alph3, 1):
    #                     null_bg[aa] = to_freq(background[i])
    #
    #         if len(line.split()) == 23:
    #             items = line.strip().split()
    #             residue_number = int(items[1])
    #             pose_dict[residue_number] = {}
    #             for i, aa in enumerate(protein_letters_alph1, 2):
    #                 pose_dict[residue_number][aa] = to_freq(items[i])
    #             pose_dict[residue_number]['lod'] = get_lod(pose_dict[residue_number], null_bg)
    #             pose_dict[residue_number]['type'] = items[0]
    #             pose_dict[residue_number]['info'] = dummy
    #             pose_dict[residue_number]['weight'] = dummy
    #
    # return pose_dict


# def weight_sequences(alignment_counts: Sequence[Sequence[int]], alignment: MultipleSeqAlignment,
#                      column_counts: Sequence[int]) -> list[float]:  # UNUSED
#     """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment
#
#     Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count)) as was described by Heinkoff and Heinkoff, 1994
#     Args:
#         alignment_counts: The counts of each AA in each column [{'A': 31, 'C': 0, ...}, 2: {}, ...]
#         alignment:
#         column_counts: The indexed counts for each column in the msa that are not gaped
#     Returns:
#         Weight of each sequence in the MSA - [2.390, 2.90, 5.33, 1.123, ...]
#     """
#     sequence_weights = []
#     for record in alignment:
#         s = 0  # "diversity/surprise"
#         for j, aa in enumerate(record.seq):
#             s += (1 / (column_counts[j] * alignment_counts[j][aa]))
#         sequence_weights.append(s)
#
#     return sequence_weights


numerical_profile = np.ndarray  # Type[np.ndarray]


class MultipleSequenceAlignment:
    _alphabet_type: alphabet_types_literal
    _array: np.ndarray
    _counts_by_position: list[list[int]] | np.ndarray
    _deletion_matrix: np.ndarray
    _frequencies: np.ndarray
    _gap_index: int
    _gaps_per_position: np.ndarray
    _numeric_translation_type: dict[str, int]
    """Given an amino acid alphabet type, return the corresponding numerical translation table"""
    _numerical_alignment: np.ndarray
    _observations_by_position: np.ndarray
    _query_aligned: str
    _sequence_identifiers: list[str]
    _sequence_indices: np.ndarray
    _sequence_weights: list[float]
    alignment: MultipleSeqAlignment
    # counts: list[dict[extended_protein_letters_and_gap, int]]
    query: str
    """The sequence used to perform the MultipleSequenceAlignment search"""
    query_length: int
    """The length of the query sequence. No gaps"""
    # query_aligned: str
    # """The sequence used to perform the MultipleSequenceAlignment search. May contain gaps from alignment"""

    @classmethod
    def from_stockholm(cls, file, **kwargs) -> MultipleSequenceAlignment:
        try:
            return cls(alignment=read_alignment(file, alignment_type='stockholm'), **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The multiple sequence alignment file '{file}' doesn't exist")

    @classmethod
    def from_fasta(cls, file) -> MultipleSequenceAlignment:
        try:
            return cls(alignment=read_alignment(file))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The multiple sequence alignment file '{file}' doesn't exist")

    @classmethod
    def from_dictionary(cls, named_sequences: dict[str, str], **kwargs) -> MultipleSequenceAlignment:
        """Create a MultipleSequenceAlignment from a dictionary of named sequences

        Args:
            named_sequences: Where name and sequence must be a string, i.e. {'1': 'MNTEELQVAAFEI...', ...}
        Returns:
            The MultipleSequenceAlignment object for the provided sequences
        """
        return cls(alignment=MultipleSeqAlignment(
            [SeqRecord(Seq(sequence), id=name)  # annotations={'molecule_type': 'Protein'},
             for name, sequence in named_sequences.items()]), **kwargs)

    @classmethod
    def from_seq_records(cls, seq_records: Iterable[SeqRecord], **kwargs) -> MultipleSequenceAlignment:
        """Create a MultipleSequenceAlignment from a SeqRecord Iterable

        Args:
            seq_records: {name: sequence, ...} ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
        """
        return cls(alignment=MultipleSeqAlignment(seq_records), **kwargs)

    def __init__(self, alignment: MultipleSeqAlignment, aligned_sequence: str = None,
                 alphabet: str = protein_letters_alph1_gaped, **kwargs):
        """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed

        gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
        probability of residue i in the specific column more accurately.

        Args:
            alignment: "Array" of SeqRecords
            aligned_sequence: Provide the sequence on which the alignment is based, otherwise the first
                sequence will be used
            alphabet: 'ACDEFGHIKLMNPQRSTVWY-'
        Sets:
            alignment - (Bio.Align.MultipleSeqAlignment)
            number_of_sequences - 214
            query - 'MGSTHLVLK...' from aligned_sequence argument OR alignment argument, index 0
            query_with_gaps - 'MGS--THLVLK...'
            counts - {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...}
            frequencies - {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...}
            observations - {1: 210, 2:211, ...}
        """
        # count_gaps: bool = False
        # count_gaps: Whether gaps (-) should be counted in column weights
        # if alignment is None:
        #     raise NotImplementedError(
        #         f"Can't create a {MultipleSequenceAlignment.__name__} with alignment=None")

        self.alignment = alignment
        self.alphabet = alphabet
        if aligned_sequence is None:
            self.query_aligned = str(alignment[0].seq)
        else:
            self.query_aligned = aligned_sequence

        # Add Info to 'meta' record as needed and populate an amino acid count dict (one-indexed)

        # self.observations = find_column_observations(self.counts, **kwargs)

    @property
    def sequence_weights(self) -> list[float]:
        """Weights for each sequence in the alignment. Default is based on the sequence "surprise", however provided
        weights can also be set
        """
        try:
            return self._sequence_weights
        except AttributeError:
            self._sequence_weights = self.weight_alignment_by_sequence()

    @sequence_weights.setter
    def sequence_weights(self, sequence_weights: list[float]):
        """If the alignment should be weighted, and weights are available, the weights for each sequence"""
        self._sequence_weights = sequence_weights
        #
        # counts_ = [[0 for _ in self.alphabet] for _ in range(self.number_of_positions)]  # list[list]
        # for sequence in self.sequences:
        #     for i, (_count, aa) in enumerate(zip(counts_, sequence)):
        #         _count[numerical_translation_alph1_gaped[aa]] += sequence_weights_[i]
        #         # self.counts[i][aa] += sequence_weights[i]
        #
        # self._counts = counts_
        # logger.critical('OLD sequence_weight self._counts', self._counts)

    def weight_alignment_by_sequence(self) -> list[float]:
        """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment

        Default means for weighting sequences. Important for creating representative sequence populations in the MSA as
        was described by Heinkoff and Heinkoff, 1994 (PMID: 7966282)

        Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count))

        Returns:
            Weight of each sequence in the MSA - [2.390, 2.90, 5.33, 1.123, ...]
        """
        # create a 1/ obs * counts = positional_weights
        #  alignment.number_of_positions - 0   1   2  ...
        #      / obs 0 [[32]   count seq 0 '-' -  2   0   0  ...   [[ 64   0   0 ...]  \
        # 1 / |  obs 1  [33] * count seq 1 'A' - 10  10   0  ... =  [330 330   0 ...]   |
        #      \ obs 2  [33]   count seq 2 'C' -  8   8   1  ...    [270 270  33 ...]] /
        #   ...   ...]               ...  ... ... ...
        position_weights = 1 / (self.observations_by_position[None, :] * self.counts_by_position)
        # take_along_axis from this with the transposed numerical_alignment (na) where each successive na idx
        # is the sequence position at the na and therefore is grabbing the position_weights by that index
        # finally sum along each sequence
        # The position_weights count seq idx must be taken by a sequence index. This happens to be on NA axis 1
        # at the moment so specified with .T and take using axis=0. Keeping both as axis=0 doen't index
        # correctly. Maybe this is a case where 'F' array ordering is needed?
        sequence_weights = np.take_along_axis(position_weights, self.numerical_alignment.T, axis=0).sum(axis=0)
        logger.critical('New sequence_weights_', sequence_weights)

        # # Old calculation
        # counts_ = [[0 for _ in self.alphabet] for _ in range(self.number_of_positions)]  # list[list]
        # for sequence in self.sequences:
        #     for _count, aa in zip(counts_, sequence):
        #         _count[numerical_translation_alph1_gaped[aa]] += 1
        #         # self.counts[i][aa] += 1
        #
        # self._counts = counts_
        # logger.critical('OLD self._counts', self._counts)
        # self._observations = [sum(_aa_counts[:self.gap_index]) for _aa_counts in self._counts]  # list[list]

        observations_by_position = self.observations_by_position
        counts_by_position = self.counts_by_position
        numerical_alignment = self.numerical_alignment
        # sequence_weights_ = weight_sequences(self._counts, self.alignment, self.observations_by_position)
        # sequence_weights_ = weight_sequences(counts_by_position, self.alignment, observations_by_position)
        sequence_weights_ = []
        for sequence_idx in range(self.length):
            sequence_weights_.append(
                (1 / (observations_by_position * counts_by_position[numerical_alignment[sequence_idx]])).sum())

        logger.critical('OLD sequence_weights_', sequence_weights_)

        return sequence_weights

    def update_counts_by_position_with_sequence_weights(self):  # UNUSED
        """Overwrite the current counts with weighted counts"""
        # Add each sequence weight to the indices indicated by the numerical_alignment
        self._counts_by_position = np.zeros((self.number_of_positions, self.alphabet_length))
        numerical_alignment = self.numerical_alignment
        sequence_weights = self.sequence_weights
        try:
            for sequence_idx in range(self.length):
                self._counts_by_position[:, numerical_alignment[sequence_idx]] += sequence_weights[sequence_idx]
        except IndexError:  # sequence_weights is the wrong length
            raise IndexError(
                f"Couldn't index the provided 'sequence_weights' with length {len(sequence_weights)}")
        logger.info('sequence_weight self.counts', self.counts_by_position)
        logger.info('May need to refactor weight sequences() to MultipleSequenceAlignment. Take particular care in '
                    'putting the alignment back together after .insert()/.delete() <- if written')

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
    def alphabet_length(self):
        """The number of sequence characters in the character alphabet"""
        return len(self.alphabet)

    @property
    def counts_by_position(self) -> np.ndarray:
        """The counts of each alphabet character for each residue position in the alignment with shape
        (number of residues, alphabet size)
        """
        try:
            return self._counts_by_position
        except AttributeError:
            # Set up the counts of each position in the alignment
            number_of_positions = self.number_of_positions
            alphabet_length = self.alphabet_length
            numerical_alignment = self.numerical_alignment
            self._counts_by_position = np.zeros((number_of_positions, alphabet_length))

            # Invert the "typical" format to length of the alignment in axis 0, and the numerical letters in axis 1
            for position_idx in range(number_of_positions):
                self._counts_by_position[position_idx, :] = \
                    np.bincount(numerical_alignment[:, position_idx], minlength=alphabet_length)

    @property
    def gap_index(self) -> int:
        """The index in the alphabet where the gap character resides"""
        try:
            return self._gap_index
        except AttributeError:
            self._gap_index = 0
            # Find where gaps and unknown start. They are always at the end
            if 'gaped' in self.alphabet_type:
                self._gap_index -= 1
            if 'unknown' in self.alphabet_type:
                self._gap_index -= 1

    @property
    def query_aligned(self) -> str:
        """The sequence used to create the MultipleSequenceAlignment potentially containing gaps from alignment"""
        return self._query_aligned

    @query_aligned.setter
    def query_aligned(self, sequence: str):
        """Set the aligned sequence used to query sequence databases for this Alignment"""
        self._query_aligned = sequence
        self.query = sequence.replace('-', '')
        # self.query_length = len(self.query)

    @property
    def query_length(self) -> int:
        """The number of residues in the MultipleSequenceAlignment query"""
        return len(self.query)

    @property
    def length(self) -> int:
        """The number of sequences in the MultipleSequenceAlignment"""
        return len(self.alignment)

    @property
    def number_of_positions(self) -> int:
        """The number of residues plus gaps found in each sequence of the MultipleSequenceAlignment"""
        return self.alignment.get_alignment_length()

    @property
    def sequences(self) -> Iterator[str]:
        """Iterate over the sequences present in the alignment"""
        for record in self.alignment:
            yield record.seq

    @property
    def sequence_identifiers(self) -> list[str]:
        """Return the identifiers associated with each sequence in the alignment"""
        try:
            return self._sequence_identifiers
        except AttributeError:
            self._sequence_identifiers = [sequence.id for sequence in self.alignment]
            return self._sequence_identifiers

    @property
    def alphabet_type(self) -> alphabet_types_literal:
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
                self._alphabet_type += '_gaped'

            return self._alphabet_type

    @alphabet_type.setter
    def alphabet_type(self, alphabet_type: alphabet_types_literal):
        """Set the alphabet_type to allow interpretation of .numeric_sequence to the correct encoding"""
        if alphabet_type in alphabet_types:
            self._alphabet_type = alphabet_type
        else:  # We got the alphabet, not its name
            self._alphabet_type = alphabet_to_alphabet_type[alphabet_type]

        alphabet_type_dependent_attrs = ['_numeric_sequence', '_numeric_translation_type', '_gap_index']
        for attr in alphabet_type_dependent_attrs:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue

    @property
    def query_indices(self) -> np.ndarray:
        """View the query as a boolean array (1, sequence_length) where gap positions, "-", are False"""
        try:
            return self._sequence_indices[0]
        except AttributeError:
            self._sequence_indices = self.array != b'-'
            return self._sequence_indices[0]

    @property
    def sequence_indices(self) -> np.ndarray:
        """View the alignment as a boolean array (length, number_of_positions) where gap positions, "-", are
        False
        """
        try:
            return self._sequence_indices
        except AttributeError:
            self._sequence_indices = self.array != b'-'
            return self._sequence_indices

    @sequence_indices.setter
    def sequence_indices(self, sequence_indices: np.ndarray):
        """Set the indices that should be included in the sequence alignment"""
        if sequence_indices.shape != (self.length, self.number_of_positions):
            raise ValueError(
                f"The shape of the sequence_indices {sequence_indices.shape}, isn't equal to the alignment shape "
                f"{(self.length, self.number_of_positions)}")
        self._sequence_indices = sequence_indices

    @property
    def numerical_alignment(self) -> np.ndarray:
        """Return the alignment as an integer array (length, number_of_positions) of the amino acid characters

        Maps the instance .alphabet characters to their resulting sequence index
        """
        try:
            return self._numerical_alignment  # [:, self.query_indices]
        except AttributeError:
            try:
                translation_type = self._numeric_translation_type
            except AttributeError:
                translation_type = self._numeric_translation_type = \
                    get_numeric_translation_table(self.alphabet_type)

            self._numerical_alignment = np.vectorize(translation_type.__getitem__)(self.array)
            return self._numerical_alignment  # [:, self.query_indices]

    @property
    def array(self) -> np.ndarray:
        """Return the alignment as a character array (length, number_of_positions) with numpy.string_ dtype"""
        try:
            return self._array
        except AttributeError:
            self._array = np.array([list(sequence) for sequence in self.sequences], np.string_)
            return self._array

    @property
    def deletion_matrix(self) -> np.ndarray:
        """Return the number of deletions at every query aligned sequence index for each sequence in the alignment"""
        try:
            return self._deletion_matrix
        except AttributeError:
            self._init_deletion_matrix()
            return self._deletion_matrix

    def _init_deletion_matrix(self):
        # def debug_array(name, array_):
        #     # input(f'{name}\n{array_[1].astype(int).tolist()[317:350]}')
        #     _, ar_len = array_.shape
        #     input(f'{name}\n{array_[1].astype(int).tolist()[ar_len - 30:]}')
        #
        # def debug_array1d(name, array_):
        #     # input(f'{name}\n{array_.astype(int).tolist()[317:350]}')
        #     ar_len = len(array_)
        #     input(f'{name}\n{array_.astype(int).tolist()[ar_len - 30:]}')
        # Create the deletion_matrix_int by using the gaped sequence_indices (inverse of sequence_indices)
        # and taking the cumulative sum of them. Finally, after selecting for only the sequence_indices, perform
        # a subtraction of position idx+1 by position idx
        query_indices = self.query_indices
        # gaped_query_indices = ~query_indices
        # gaped_query_indices = ~self.query_indices
        # debug_array1d('gaped_query_indices', gaped_query_indices)
        # sequence_indices = self.sequence_indices
        # debug_array('sequence_indices', sequence_indices)
        # Find where there is sequence information, while gaped (i.e. ~) query information
        sequence_deletion_indices = self.sequence_indices * ~query_indices
        # debug_array('sequence_deletion_indices', sequence_deletion_indices)
        # Perform a cumulative sum of the "deletion" indices,
        # logger.critical(f"Created sequence_deletion_indices: {np.nonzero(sequence_deletion_indices[:2])}")
        sequence_deletion_indices_sum = np.cumsum(sequence_deletion_indices, axis=1)
        # debug_array('sequence_deletion_indices_sum', sequence_deletion_indices_sum)
        # logger.critical(f"Created sequence_deletion_indices_sum: "
        #                 f"{sequence_deletion_indices_sum[:2, -100:].tolist()}")
        # # Then remove any summation that is in gaped query
        # deletion_matrix = sequence_deletion_indices_sum - sequence_deletion_indices  # <- subtract indices for offset !!
        # # debug_array('deletion_matrix', deletion_matrix)
        # # ONLY THING LEFT TO DO IS TO REMOVE THE NON-DELETION PROXIMAL CUMSUM, i.e: 0, 8, *8, *8,
        # # Which is accomplished by the subtraction of position idx+1 by position idx
        # # logger.critical(f"Created deletion_matrix: {deletion_matrix[:2].tolist()}")
        # # deletion_matrix[:, 1:] = deletion_matrix[:, 1:] - deletion_matrix[:, :-1]
        # deletion_matrix = np.diff(deletion_matrix, prepend=0)
        # # debug_array('deletion_matrix', deletion_matrix)
        # # self._deletion_matrix = deletion_matrix[:, query_indices]
        # # Finally, clear any information in the gaped_query_indices by multiplying by query_indices mask
        # self._deletion_matrix = deletion_matrix * query_indices
        # # logger.critical(f"Created subtracted, indexed, deletion_matrix: {self._deletion_matrix[-2:].tolist()}")
        # # debug_array('self._deletion_matrix', self._deletion_matrix)
        # # logger.debug(f"Created subtracted, indexed, deletion_matrix: {self._deletion_matrix[1, query_indices]}")

        # Remove any indices that aren't query indices. Solving this problem in the context of the entire MSA doesn't
        # appear possible unless using a sequence specific implementation like from Alphafold
        repeat_deletion_matrix = sequence_deletion_indices_sum[:, query_indices]
        # debug_array('repeat_deletion_matrix', repeat_deletion_matrix)
        self._deletion_matrix = np.diff(repeat_deletion_matrix, prepend=0)  # <- repeat subtraction!!
        # debug_array('diff repeat_deletion_matrix', repeat_deletion_matrix_diff)
        logger.debug(f"Created subtracted, indexed, deletion_matrix: {self._deletion_matrix[1]}")

        # # Alphafold implementation
        # # Count the number of deletions w.r.t. query.
        # _deletion_matrix = []
        # query_aligned = self.query_aligned
        # for sequence in self.sequences:
        #     deletion_vec = []
        #     deletion_count = 0
        #     for seq_res, query_res in zip(sequence, query_aligned):
        #         if seq_res != '-' or query_res != '-':
        #             if query_res == '-':
        #                 deletion_count += 1
        #             else:
        #                 deletion_vec.append(deletion_count)
        #                 deletion_count = 0
        #     _deletion_matrix.append(deletion_vec)
        # logger.critical(f"Created AF _deletion_matrix: {_deletion_matrix[1]}")
        # # End AF implementation

    @property
    def frequencies(self) -> np.ndarray:
        """Access the per-residue, alphabet frequencies with shape (number of residues, alphabet characters). Bounded
        between 0 and 1
        """
        # self._frequencies = [[count/observation for count in amino_acid_counts[:self.gap_index]]  # don't use the gap
        #                      for amino_acid_counts, observation in zip(self._counts, self._observations)]
        # logger.critical('OLD self._frequencies', self._frequencies)

        # self.frequencies = np.zeros((self.number_of_positions, self.alphabet_length))  # self.counts.shape)
        # gap_index = self.gap_index
        # for position_idx in range(self.number_of_positions):
        #     self.frequencies[position_idx, :] = self.counts[:, :gap_index] / self.observations
        try:
            return self._frequencies
        except AttributeError:  # Don't use gaped indices
            self._frequencies = self.counts_by_position[:, :self.gap_index] / self.observations_by_position[:, None]
        return self._frequencies

    @property
    def observations_by_position(self) -> np.ndarray:
        """The number of sequences with observations at each residue position in the alignment"""
        try:
            return self._observations_by_position
        except AttributeError:
            # if count_gaps:
            #     self._observations_by_position = np.array([self.length for _ in range(self.number_of_positions)])
            # else:
            # gap_observations = [_aa_counts['-'] for _aa_counts in self.counts]  # list[dict]
            # gap_observations = [_aa_counts[0] for _aa_counts in self.counts]  # list[list]
            # self.observations = [counts - gap for counts, gap in zip(self.observations, gap_observations)]
            self._observations_by_position = self.counts_by_position[:, :self.gap_index].sum(axis=1)
            if not np.any(self._observations_by_position):  # Check if an observation is 0
                raise ValueError(
                    "Can't have a MSA column (sequence index) with 0 observations. Found at ("
                    f'{",".join(map(str, np.flatnonzero(self.observations_by_position)))}')
                #     f'{",".join(str(idx) for idx, pos in enumerate(self.observations) if not pos)}')

        return self._observations_by_position

    @property
    def gaps_per_position(self) -> np.ndarray:
        """The number of gaped letters at each position in the sequence with shape (number of residues,)"""
        try:
            return self._gaps_per_position
        except AttributeError:
            self._gaps_per_position = self.length - self.observations_by_position
        return self._gaps_per_position

    def get_probabilities_from_profile(self, profile: numerical_profile) -> np.ndarray:
        """For each sequence in the alignment, extract the values from a profile corresponding to the amino acid type
        of each residue in each sequence

        Args:
            profile: A profile of values with shape (length, alphabet_length) where length is the number_of_positions
        Returns:
            The array with shape (length, number_of_positions) with the value for each amino acid index in profile
        """
        # transposed_alignment = self.numerical_alignment.T
        return np.take_along_axis(profile, self.numerical_alignment.T, axis=1).T
        # observed = {profile: np.take_along_axis(background, transposed_alignment, axis=1).T
        #             for profile, background in backgrounds.items()}
        # observed = {profile: np.where(np.take_along_axis(background, transposed_alignment, axis=1) > 0, 1, 0).T
        #             for profile, background in backgrounds.items()}

    def insert(self, at: int, sequence: str, msa_index: bool = False):
        """Insert new sequence in the MultipleSequenceAlignment where the added sequence is added to all columns

        Args:
            at: The index to insert the sequence at. By default, the index is in reference to where self.query_indices
                are True, i.e the query sequence
            sequence: The sequence to insert. Will be inserted for every sequence of the alignment
            msa_index: Whether the insertion index is in the frame of the entire multiple sequence alignment.
                Default, False, indicates the index is in the frame of the query sequence index, i.e. no gaps
        Sets:
            self.alignment: The existing alignment updated with the new sequence in alignment form
        """
        if msa_index:
            at = at
        else:
            try:  # To get the index 'at' for those indices that are present in the query
                at = np.flatnonzero(self.query_indices)[at]
            except IndexError:  # This index is outside of query
                if at >= self.query_length:
                    # Treat as append
                    at = self.number_of_positions
                else:
                    raise NotImplementedError(f"Couldn't index with a negative index...")
        begin_slice = slice(at)
        end_slice = slice(at, None)

        logger.debug(f'Insertion is occurring at {self.__class__.__name__} index {at}')

        new_sequence = Seq(sequence)
        new_alignment = MultipleSeqAlignment(
            [SeqRecord(new_sequence, id=id_)  # annotations={'molecule_type': 'Protein'},
             for id_ in self.sequence_identifiers])

        logger.debug(f'number of sequences in new_alignment: {len(new_alignment)}')
        start_alignment_slice = self.alignment[:, begin_slice]
        start_alignment_len = len(start_alignment_slice)
        if start_alignment_len:
            logger.debug(f'number of sequences in start_alignment_slice: {start_alignment_len}')
            new_alignment = start_alignment_slice + new_alignment

        end_alignment_slice = self.alignment[:, end_slice]
        end_alignment_len = len(end_alignment_slice)
        if end_alignment_len:
            logger.debug(f'number of sequences in end_alignment_slice: {end_alignment_len}')
            new_alignment = new_alignment + end_alignment_slice

        # Set the alignment
        self.alignment = new_alignment
        self.query_aligned = str(new_alignment[0].seq)

        # Update alignment dependent features
        self.reset_state()
        logger.debug(f'Inserted alignment has shape ({self.length}, {self.number_of_positions})')

    def reset_state(self):
        """Remove any state attributes"""
        for attr in ['_array', '_deletion_matrix', '_numerical_alignment', '_sequence_indices',
                     '_sequence_identifiers', '_observations_by_position', '_counts_by_position', '_gaps_per_position',
                     '_frequencies']:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue

    def pad_alignment(self, length, axis: int = 0):
        """Extend the alignment by a set length

        Args:
            length: The length to pad the alignment
            axis: The axis to pad. 0 pads the sequences, 1 pads the residues
        Sets:
            self.alignment with the specified padding
        """
        if axis == 0:
            dummy_record = SeqRecord(Seq('-' * self.number_of_positions), id='dummy')
            self.alignment.extend([dummy_record for _ in range(length)])
            self.reset_state()
        else:  # axis == 1
            self.insert(self.number_of_positions, '-' * length, msa_index=True)

        logger.debug(f'padded alignment has shape ({self.length}, {self.number_of_positions})')
