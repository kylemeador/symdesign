from __future__ import annotations

import csv
import logging
import os
import subprocess
from collections import defaultdict
from itertools import count
from math import log2
from pathlib import Path
from typing import Sequence, AnyStr, Iterable, Literal, Any, TypedDict, get_args, Generator

import numpy as np
from Bio import AlignIO, SeqIO
# Todo
# BiopythonDeprecationWarning: Bio.pairwise2 has been deprecated, and we intend to remove it in a future release of
# Biopython. As an alternative, please consider using Bio.Align.PairwiseAligner as a replacement, and contact the
# Biopython developers if you still need the Bio.pairwise2 module.
from Bio.Align import Alignment, MultipleSeqAlignment, PairwiseAligner, PairwiseAlignments, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from symdesign import utils
putils = utils.path


# Globals
zero_offset = 1
logger = logging.getLogger(__name__)
# Types
protein_letters3_alph1: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR')
protein_letters3_extended: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ASX', 'XAA', 'GLX', 'XLE', 'SEC', 'PYL')
protein_letters_alph1: str = 'ACDEFGHIKLMNPQRSTVWY'
protein_letters_alph1_extended: str = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
protein_letters_3to1: dict[str, str] = dict(zip(protein_letters3_alph1, protein_letters_alph1))
protein_letters_1to3: dict[str, str] = dict(zip(protein_letters_alph1, protein_letters3_alph1))
protein_letters_3to1_extended: dict[str, str] = dict(zip(protein_letters3_extended, protein_letters_alph1_extended))
protein_letters_1to3_extended: dict[str, str] = dict(zip(protein_letters_alph1_extended, protein_letters3_extended))
protein_letters_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
_alph_3_aa: tuple[protein_letters_literal, ...] = get_args(protein_letters_literal)
protein_letters_alph3 = ''.join(_alph_3_aa)
protein_letters_alph1_literal = Literal[tuple(protein_letters_alph1)]
protein_letters_alph1_unknown = protein_letters_alph1 + 'X'
protein_letters_alph3_unknown = protein_letters_alph3 + 'X'
protein_letters_alph1_gapped = protein_letters_alph1 + '-'
protein_letters_alph3_gapped = protein_letters_alph3 + '-'
protein_letters_alph1_unknown_gapped = protein_letters_alph1 + 'X-'
protein_letters_alph3_unknown_gapped = protein_letters_alph3 + 'X-'
protein_letters_alph1_extended_literal = Literal[tuple(protein_letters_alph1_extended)]
protein_letters_alph3_unknown_gapped_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-']
# Todo the default value for many of these might have conflicting use cases
#  For instance, a value of 20 for protein_letters_alph1_unknown_gapped would but a missing in the unknown position
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
numerical_translation_alph1_gapped = defaultdict(lambda: 20, zip(protein_letters_alph1_gapped, count()))
numerical_translation_alph3_gapped = defaultdict(lambda: 20, zip(protein_letters_alph3_gapped, count()))
numerical_translation_alph1_gapped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_gapped), count()))
numerical_translation_alph3_gapped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_gapped), count()))
numerical_translation_alph1_unknown_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_unknown), count()))
numerical_translation_alph3_unknown_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_unknown), count()))
numerical_translation_alph1_unknown_gapped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph1_unknown_gapped), count()))
numerical_translation_alph3_unknown_gapped_bytes = \
    defaultdict(lambda: 20, zip((char.encode() for char in protein_letters_alph3_unknown_gapped), count()))
extended_protein_letters_and_gap_literal = Literal[get_args(protein_letters_alph1_extended_literal), '-']
extended_protein_letters_and_gap: tuple[str, ...] = get_args(extended_protein_letters_and_gap_literal)
alphabet_types_literal = Literal[
    'protein_letters_alph1', 'protein_letters_alph3', 'protein_letters_alph1_gapped',
    'protein_letters_alph3_gapped', 'protein_letters_alph1_unknown', 'protein_letters_alph3_unknown',
    'protein_letters_alph1_unknown_gapped', 'protein_letters_alph3_unknown_gapped']
alphabet_to_type = {'ACDEFGHIKLMNPQRSTVWY': protein_letters_alph1,
                    'ARNDCQEGHILKMFPSTWYV': protein_letters_alph3,
                    'ACDEFGHIKLMNPQRSTVWY-': protein_letters_alph1_gapped,
                    'ARNDCQEGHILKMFPSTWYV-': protein_letters_alph3_gapped,
                    'ACDEFGHIKLMNPQRSTVWYX': protein_letters_alph1_unknown,
                    'ARNDCQEGHILKMFPSTWYVX': protein_letters_alph3_unknown,
                    'ACDEFGHIKLMNPQRSTVWYX-': protein_letters_alph1_unknown_gapped,
                    'ARNDCQEGHILKMFPSTWYVX-': protein_letters_alph3_unknown_gapped}
alignment_programs_literal = Literal['hhblits', 'psiblast']
alignment_programs: tuple[str, ...] = get_args(alignment_programs_literal)
profile_types = Literal['evolutionary', 'fragment', '']


class LodDict(TypedDict):
    A: int
    R: int
    N: int
    D: int
    C: int
    Q: int
    E: int
    G: int
    H: int
    I: int
    L: int
    K: int
    M: int
    F: int
    P: int
    S: int
    T: int
    W: int
    Y: int
    V: int
    # B: int
    # J: int
    # O: int
    # U: int
    # X: int
    # Z: int


profile_keys = Literal[protein_letters_literal, 'lod', 'type', 'info', 'weight']
ProfileEntry = TypedDict('ProfileEntry', {**LodDict.__dict__['__annotations__'],
                                          'lod': LodDict,
                                          'type': protein_letters_literal,
                                          'info': float,
                                          'weight': float})
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


def get_sequence_to_numeric_translation_table(alphabet_type: alphabet_types_literal) -> defaultdict[str, int] | dict[str, int]:
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
    #         case 'protein_letters_alph1_gapped':
    #             numeric_translation_table = numerical_translation_alph1_gapped_bytes
    #         case 'protein_letters_alph3_gapped':
    #             numeric_translation_table = numerical_translation_alph3_gapped_bytes
    #         case 'protein_letters_alph1_unknown':
    #             numeric_translation_table = numerical_translation_alph1_unknown_bytes
    #         case 'protein_letters_alph3_unknown':
    #             numeric_translation_table = numerical_translation_alph3_unknown_bytes
    #         case 'protein_letters_alph1_unknown_gapped':
    #             numeric_translation_table = numerical_translation_alph1_unknown_gapped_bytes
    #         case 'protein_letters_alph3_unknown_gapped':
    #             numeric_translation_table = numerical_translation_alph3_unknown_gapped_bytes
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
    elif alphabet_type == 'protein_letters_alph1_gapped':
        numeric_translation_table = numerical_translation_alph1_gapped_bytes
    elif alphabet_type == 'protein_letters_alph3_gapped':
        numeric_translation_table = numerical_translation_alph3_gapped_bytes
    elif alphabet_type == 'protein_letters_alph1_unknown':
        numeric_translation_table = numerical_translation_alph1_unknown_bytes
    elif alphabet_type == 'protein_letters_alph3_unknown':
        numeric_translation_table = numerical_translation_alph3_unknown_bytes
    elif alphabet_type == 'protein_letters_alph1_unknown_gapped':
        numeric_translation_table = numerical_translation_alph1_unknown_gapped_bytes
    elif alphabet_type == 'protein_letters_alph3_unknown_gapped':
        numeric_translation_table = numerical_translation_alph3_unknown_gapped_bytes
    else:
        try:  # To see if we already have the alphabet, and return the defaultdict
            numeric_translation_table = alphabet_to_type[alphabet_type]
        except KeyError:
            # raise ValueError(wrong_alphabet_type)
            logger.warning(
                f"Parameter alphabet_type option '{alphabet_type}' isn't viable. Attempting to create it")
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


def generate_alignment(seq1: Sequence[str], seq2: Sequence[str], matrix: str = default_substitution_matrix_name,
                       local: bool = False, top_alignment: bool = True) -> Alignment | PairwiseAlignments:
    """Use Biopython's pairwise2 to generate a sequence alignment

    Args:
        seq1: The first sequence to align
        seq2: The second sequence to align
        matrix: The matrix used to compare character similarities
        local: Whether to run a local alignment. Only use for generally similar sequences!
        top_alignment: Only include the highest scoring alignment
    Returns:
        The resulting alignment(s). Will be an Alignment object if top_alignment is True else PairwiseAlignments object
    """
    matrix_ = _substitution_matrices_cache.get(matrix)
    if matrix_ is None:
        try:  # To get the new matrix and store for future ops
            matrix_ = _substitution_matrices_cache[matrix] = substitution_matrices.load(matrix)
        except FileNotFoundError:  # Missing this
            raise KeyError(f"Couldn't find the substitution matrix '{matrix}' ")

    if local:
        mode = 'local'
    else:
        mode = 'global'

    # logger.debug(f'Generating sequence alignment between:\n{seq1}\n\tAND:\n{seq2}')
    # Set these the default from blastp
    open_gap_score = -12.  # gap_penalty
    extend_gap_score = -1.  # gap_ext_penalty
    # Create sequence alignment
    aligner = PairwiseAligner(mode=mode, substitution_matrix=matrix_,  # scoring='blastp')
                              open_gap_score=open_gap_score, extend_gap_score=extend_gap_score)
    alignments = aligner.align(seq1, seq2)
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
            raise ValueError(f"Can't perform {hhblits.__name__} without a 'sequence_file' or a 'sequence'")
        else:
            sequence_file = write_sequences((name, sequence), file_name=os.path.join(out_dir, f'{name}.seq'))
    pssm_file = os.path.join(out_dir, f'{name}.hmm')
    a3m_file = os.path.join(out_dir, f'{name}.a3m')
    # Todo for higher performance set up https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
    #  Create a ramdisk to store a database chunk to make hhblits/Jackhmmer run fast.
    #  sudo mkdir -m 777 --parents /tmp/ramdisk
    #  sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk
    cmd = [putils.hhblits_exe, '-d', putils.uniclust_db, '-i', sequence_file,
           '-ohhm', pssm_file, '-oa3m', a3m_file,  # '-Ofas', self.msa_file,
           '-hide_cons', '-hide_pred', '-hide_dssp', '-E', '1E-06',
           '-v', '1', '-cpu', str(threads)]

    if return_command:
        return cmd  # subprocess.list2cmdline(cmd)

    logger.debug(f'{name} Profile Command: {subprocess.list2cmdline(cmd)}')
    p = subprocess.Popen(cmd)
    p.communicate()
    if p.returncode != 0:
        # temp_file = os.path.join(out_path, f'{self.name}.hold')
        temp_file = Path(out_dir, f'{name}.hold')
        temp_file.unlink(missing_ok=True)
        # if os.path.exists(temp_file):  # remove hold file blocking progress
        #     os.remove(temp_file)
        raise RuntimeError(f'Profile generation for {name} got stuck')  #
        # raise DesignError(f'Profile generation for {self.name} got stuck. See the error for details -> {p.stderr} '
        #                   f'output -> {p.stdout}')  #

    # Preferred alignment type
    msa_file = os.path.join(out_dir, f'{name}.sto')
    p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, msa_file, '-num', '-uc'])
    p.communicate()
    fasta_msa = os.path.join(out_dir, f'{name}.fasta')
    p = subprocess.Popen([putils.reformat_msa_exe_path, a3m_file, fasta_msa, '-M', 'first', '-r'])
    p.communicate()
    # os.system('rm %s' % self.a3m_file)
    return None


def optimize_protein_sequence(sequence: str, species: str = 'e_coli') -> str:
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
            CodonOptimize,EnforceGCContent, AvoidHairpins, AvoidPattern, UniquifyAllKmers, AvoidRareCodons, \
            EnforceTranslation
    except ModuleNotFoundError:
        raise RuntimeError(f"Can't {optimize_protein_sequence.__name__} as the dependency DnaChisel is not available")

    try:
        dna_sequence = reverse_translate(sequence)
    except KeyError as error:
        raise KeyError(f'Warning an invalid character was found in your protein sequence: {error}')

    problem = DnaOptimizationProblem(sequence=dna_sequence,  # max_random_iters=20000,
                                     objectives=[CodonOptimize(species=species)], logger=None,
                                     constraints=[EnforceGCContent(mini=0.25, maxi=0.65),  # twist required
                                                  EnforceGCContent(mini=0.35, maxi=0.65, window=50),  # twist required
                                                  AvoidHairpins(stem_size=20, hairpin_window=48),  # efficient translate
                                                  AvoidPattern('GGAGG', location=(1, seq_length, 1)),  # ribosome bind
                                                  AvoidPattern('TAAGGAG', location=(1, seq_length, 1)),  # ribosome bind
                                                  AvoidPattern('AAAAA', location=(1, seq_length, 0)),  # terminator
                                                  # AvoidPattern('TTTTT', location=(1, seq_length, 1)),  # terminator
                                                  AvoidPattern('GGGGGGGGGG', location=(1, seq_length, 0)),  # homopoly
                                                  # AvoidPattern('CCCCCCCCCC', location=(1, seq_length)),  # homopoly
                                                  UniquifyAllKmers(20),  # twist required
                                                  AvoidRareCodons(0.08, species=species),
                                                  EnforceTranslation(),
    #                                             EnforceMeltingTemperature(mini=10, maxi=62, location=(1, seq_length)),
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


def create_mulitcistronic_sequences(args):
    # if not args.multicistronic_intergenic_sequence:
    #     args.multicistronic_intergenic_sequence = expression.ncoI_multicistronic_sequence
    raise NotImplementedError('Please refactor to a protocols/tools module so that JobResources can be used.')
    job = job_resources_factory()
    file = args.file[0]  # since args.file is collected with nargs='*', select the first
    if file.endswith('.csv'):
        with open(file) as f:
            protein_sequences = [SeqRecord(Seq(sequence), annotations={'molecule_type': 'Protein'}, id=name)
                                 for name, sequence in csv.reader(f)]
    elif file.endswith('.fasta'):
        protein_sequences = list(read_fasta_file(file))
    else:
        raise NotImplementedError(f'Sequence file with extension {os.path.splitext(file)[-1]} is not supported!')

    # Convert the SeqRecord to a plain sequence
    # design_sequences = [str(seq_record.seq) for seq_record in design_sequences]
    nucleotide_sequences = {}
    for idx, group_start_idx in enumerate(list(range(len(protein_sequences)))[::args.number_of_genes], 1):
        # Call attribute .seq to get the sequence
        cistronic_sequence = optimize_protein_sequence(protein_sequences[group_start_idx].seq,
                                                       species=args.optimize_species)
        for protein_sequence in protein_sequences[group_start_idx + 1: group_start_idx + args.number_of_genes]:
            cistronic_sequence += args.multicistronic_intergenic_sequence
            cistronic_sequence += optimize_protein_sequence(protein_sequence.seq,
                                                            species=args.optimize_species)
        new_name = f'{protein_sequences[group_start_idx].id}_cistronic'
        nucleotide_sequences[new_name] = cistronic_sequence
        logger.info(f'Finished sequence {idx} - {new_name}')

    location = file
    if not args.prefix:
        args.prefix = f'{os.path.basename(os.path.splitext(location)[0])}_'
    else:
        args.prefix = f'{args.prefix}_'

    nucleotide_sequence_file = write_sequences(nucleotide_sequences, csv=args.csv,
                                               file_name=os.path.join(job.output_directory,
                                                                      'MulticistronicNucleotideSequences'))
    logger.info(f'Multicistronic nucleotide sequences written to: {nucleotide_sequence_file}')


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


protein_letters_alph3_gapped_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']

# class MutationEntry(TypedDict):
#     to: protein_letters_alph3_gapped_literal
#     from: protein_letters_alph3_gapped_literal
MutationEntry = TypedDict('MutationEntry', {'to': protein_letters_alph3_gapped_literal,
                                            'from': protein_letters_alph3_gapped_literal})
# mutation_entry = dict[Literal['to', 'from'], protein_letters_alph3_gapped_literal]
# mutation_entry = Type[dict[Literal['to', 'from'], protein_letters_alph3_gapped_literal]]
"""Mapping of a reference sequence amino acid type, 'to', and the resulting sequence amino acid type, 'from'"""
mutation_dictionary = dict[int, MutationEntry]
"""The mapping of a residue number to a mutation entry containing the reference, 'to', and sequence, 'from', amino acid 
type
"""
sequence_dictionary = dict[int, protein_letters_alph3_gapped_literal]
"""The mapping of a residue number to the corresponding amino acid type"""


def generate_mutations(reference: Sequence, query: Sequence, offset: bool = True, blanks: bool = False,
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
        blanks: Include all gaped indices, i.e. outside the reference sequence or missing characters in the sequence
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
        mutations = {idx: {'from': seq1, 'to': seq2} for idx, (seq1, seq2) in sequence_iterator}
    else:
        mutations = {idx: {'from': seq1, 'to': seq2} for idx, (seq1, seq2) in sequence_iterator if seq1 != seq2}

    # Find last index of reference
    ending_index_of_seq1 = starting_idx_of_seq1 + align_seq_1.rfind(reference[-1])
    remove_mutation_list = []
    if only_gaps:  # Remove the actual mutations, keep internal and external gap indices and the reference sequence
        blanks = True
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if idx_offset < entry <= ending_index_of_seq1 and mutation['to'] != '-'])
    if blanks:  # Leave all types of blanks, otherwise check for each requested type
        remove_termini, remove_query_gaps = False, False

    if remove_termini:  # Remove indices outside of sequence 1
        remove_mutation_list.extend([entry for entry in mutations
                                     if entry < idx_offset or ending_index_of_seq1 < entry])

    if remove_query_gaps:  # Remove indices where sequence 2 is gaped
        remove_mutation_list.extend([entry for entry, mutation in mutations.items()
                                     if 0 < entry <= ending_index_of_seq1 and mutation['to'] == '-'])
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
        blanks: bool = False - Include all gaped indices, i.e. outside the reference sequence or missing characters
            in the sequence
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


def get_equivalent_indices(target: Sequence = None, query: Sequence = None, alignment: Sequence = None) \
        -> tuple[list[int], list[int]]:
    """From two sequences, find the indices where both sequences are equal

    Args:
        target: The first sequence to compare
        query: The second sequence to compare
        alignment: An existing Bio.Align.Alignment object
    Returns:
        The pair of indices where the sequences align.
            Ex: sequence1 = A B C D E F ...
                sequence2 = A B - D E F ...
            returns        [0,1,  3,4,5, ...],
                           [0,1,  2,3,4, ...]
    """
    if alignment is None:
        if target is not None and query is not None:
            # # Get all mutations from the alignment of sequence1 and sequence2
            mutations = generate_mutations(target, query, blanks=True, return_all=True)
            # alignment = generate_alignment(target, query)
            # alignment.inverse_indices
            # return
        else:
            raise ValueError(f"Can't {get_equivalent_indices.__name__} without passing either 'alignment' or "
                             f"'target' and 'query'")
    else:  # Todo this may not be ever useful since the alignment needs to go into the generate_mutations()
        raise NotImplementedError(f"Set {get_equivalent_indices.__name__} up with an Alignment object from Bio.Align")

    target_mutations = ''.join([mutation['to'] for mutation in mutations.values()])
    query_mutations = ''.join([mutation['from'] for mutation in mutations.values()])
    logger.debug(f"Sequence info:\ntarget :{target_mutations}\nquery  :{query_mutations}")
    # Get only those indices where there is an aligned aa on the opposite chain
    sequence1_indices, sequence2_indices = [], []
    # to_idx = from_idx = 0
    to_idx, from_idx = count(0), count(0)
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


numerical_profile = np.ndarray  # Type[np.ndarray]


class MultipleSequenceAlignment:
    _alphabet_type: alphabet_types_literal
    _array: np.ndarray
    _deletion_matrix: np.ndarray
    _frequencies: np.ndarray
    _gaps_per_position: np.ndarray
    _numerical_alignment: np.ndarray
    _sequence_identifiers: list[str]
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
                 alphabet: str = protein_letters_alph1_gapped,
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
        if alignment is None:
            raise NotImplementedError(f"Can't create a {MultipleSequenceAlignment.__name__} with alignment=None")

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
            logger.critical('sequence_weights', sequence_weights)
            self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
            for record in self.alignment:
                for i, aa in enumerate(record.seq):
                    self._counts[i][numerical_translation_alph1_gapped[aa]] += 1
                    # self.counts[i][aa] += 1
            logger.critical('OLD self._counts', self._counts)
            self._observations = [sum(_aa_counts[:self._gap_index]) for _aa_counts in self._counts]  # list[list]

            sequence_weights_ = weight_sequences(self._counts, self.alignment, self._observations)
            logger.critical('OLD sequence_weights_', sequence_weights_)

        if sequence_weights is not None:  # overwrite the current counts with weighted counts
            self.sequence_weights = sequence_weights
            # Todo update this as well
            self._counts = [[0 for letter in alphabet] for _ in range(self.length)]  # list[list]
            for record in self.alignment:
                for i, aa in enumerate(record.seq):
                    self._counts[i][numerical_translation_alph1_gapped[aa]] += sequence_weights_[i]
                    # self.counts[i][aa] += sequence_weights[i]
            logger.critical('OLD sequence_weight self._counts', self._counts)

            # add each sequence weight to the indices indicated by the numerical_alignment
            self.counts = np.zeros((self.length, len(protein_letters_alph1_gapped)))
            for idx in range(self.number_of_sequences):
                self.counts[:, numerical_alignment[idx]] += sequence_weights[idx]
            logger.critical('sequence_weight self.counts', self.counts)
        else:
            self.sequence_weights = []

        # Set up the deletion matrix
        # Create the deletion_matrix_int by using the gaped sequence_indices (inverse of sequence_indices)
        # and taking the cumulative sum of them. Finally, after selecting for only the sequence_indices, perform
        # a subtraction of position idx+1 by position idx
        sequence_indices = self.sequence_indices
        query_indices = self.query_indices
        # Find where there is some sequence information
        # sequence_or_query_indices = (sequence_indices + query_indices) > 0
        gaped_query_indices = ~query_indices
        # gaped_query_indices = ~self.query_indices
        # Find where there is sequence information but not query information
        # sequence_deletion_indices = sequence_or_query_indices * gaped_query_indices
        sequence_deletion_indices = sequence_indices * gaped_query_indices
        # Perform a cumulative sum of the "deletion" indices,
        # logger.critical(f"Created sequence_deletion_indices: {np.nonzero(sequence_deletion_indices[:2])}")
        sequence_deletion_indices_sum = np.cumsum(sequence_deletion_indices, axis=1)
        logger.critical(f"Created sequence_deletion_indices_sum: "
                        f"{sequence_deletion_indices_sum[:2, -100:].tolist()}")
        # Then remove any summation that is in gaped query
        deletion_matrix = sequence_deletion_indices_sum * gaped_query_indices
        # ONLY THING LEFT TO DO IS TO REMOVE THE NON-DELETION PROXIMAL CUMSUM, i.e: 0, 8, *8, *8,
        # Which is accomplished by the subtraction of position idx+1 by position idx
        # logger.critical(f"Created deletion_matrix: {deletion_matrix[:2].tolist()}")
        deletion_matrix[:, 1:] = deletion_matrix[:, 1:] - deletion_matrix[:, :-1]
        # self._deletion_matrix = deletion_matrix[:, query_indices]
        # Finally, clear any information in the gaped_query_indices by multiplying by query_indices mask
        self._deletion_matrix = deletion_matrix * query_indices
        logger.critical(f"Created subtracted, indexed, deletion_matrix: {self._deletion_matrix[-2:].tolist()}")

        # msa_gap_indices = ~sequence_indices
        # # iterator_np = np.cumsum(msa_gap_indices, axis=1) * msa_gap_indices
        # # gap_sum = np.cumsum(msa_gap_indices, axis=1)[sequence_indices]
        # gap_sum = np.cumsum(msa_gap_indices, axis=1) * sequence_indices
        # deletion_matrix = np.zeros_like(gap_sum)
        # deletion_matrix[:, 1:] = gap_sum[:, 1:] - gap_sum[:, :-1]
        # logger.critical(f"Created deletion_matrix: {deletion_matrix[:2].tolist()}")
        # Alphafold implementation
        # Count the number of deletions w.r.t. query.
        _deletion_matrix = []
        query = self.query_with_gaps
        for sequence in self.sequences:
            deletion_vec = []
            deletion_count = 0
            for seq_res, query_res in zip(sequence, query):
                if seq_res != '-' or query_res != '-':
                    if query_res == '-':
                        deletion_count += 1
                    else:
                        deletion_vec.append(deletion_count)
                        deletion_count = 0
            _deletion_matrix.append(deletion_vec)
        logger.critical(f"Created AF _deletion_matrix: {_deletion_matrix[-2:]}")
        # End AF implementation

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
            named_sequences: Where name and sequence must be a string, i.e. {'1': 'MNTEELQVAAFEI...', ...}
        Returns:
            The MultipleSequenceAlignment object for the provided sequences
        """
        return cls(alignment=MultipleSeqAlignment(
            [SeqRecord(Seq(sequence), annotations={'molecule_type': 'Protein'}, id=name)
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
    def sequences(self) -> Generator[str, None, None]:
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
                self._alphabet_type += '_gapped'

            return self._alphabet_type

    @alphabet_type.setter
    def alphabet_type(self, alphabet_type: alphabet_types_literal):
        """Set the alphabet_type to allow interpretation of .numeric_sequence to the correct encoding"""
        self._alphabet_type = alphabet_type

        alphabet_type_dependent_attrs = ['_numeric_sequence', '_numeric_translation_type']
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
        """View the alignment as a boolean array (number_of_sequences, sequence_length) where gap positions, "-", are
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
        if sequence_indices.shape != (self.number_of_sequences, self.length):
            raise ValueError(f"The shape of the sequence_indices {sequence_indices.shape}, isn't equal to the alignment"
                             f" {(self.number_of_sequences, self.length)}")
        self._sequence_indices = sequence_indices
        # logger.critical(f'981 Found {len(np.flatnonzero(self.sequence_indices[0]))} indices utilized in design')

    @property
    def numerical_alignment(self) -> np.ndarray:
        """Return the alignment as an integer array (number_of_sequences, length) of the amino acid characters

        Maps the instance .alphabet characters to their resulting sequence index
        """
        try:
            return self._numerical_alignment  # [:, self.query_indices]
        except AttributeError:
            try:
                translation_type = self._numeric_translation_type
            except AttributeError:
                self._numeric_translation_type = get_sequence_to_numeric_translation_table(self.alphabet_type)
                translation_type = self._numeric_translation_type

            self._numerical_alignment = np.vectorize(translation_type.__getitem__)(self.array)
            return self._numerical_alignment  # [:, self.query_indices]

    @property
    def array(self) -> np.ndarray:
        """Return the alignment as a character array (number_of_sequences, length) with numpy.string_ dtype"""
        try:
            return self._array
        except AttributeError:
            self._array = np.array([list(record) for record in self.alignment], np.string_)
            return self._array

    @property
    def deletion_matrix(self) -> np.ndarray:
        """Return the number of deletions at every query aligned sequence index for each sequence in the alignment"""
        return self._deletion_matrix

    @property
    def frequencies(self) -> np.ndarray:
        """Access the per residue (axis=0) amino acid frequencies (axis=1) bounded between 0 and 1"""
        # self._frequencies = [[count/observation for count in amino_acid_counts[:self._gap_index]]  # don't use the gap
        #                      for amino_acid_counts, observation in zip(self._counts, self._observations)]
        # logger.critical('OLD self._frequencies', self._frequencies)

        # self.frequencies = np.zeros((self.length, len(protein_letters_alph1)))  # self.counts.shape)
        # for residue_idx in range(self.length):
        #     self.frequencies[residue_idx, :] = self.counts[:, :self._gap_index] / self.observations
        try:
            return self._frequencies
        except AttributeError:  # Don't use gapped indices
            self._frequencies = self.counts[:, :self._gap_index] / self.observations[:, None]
        return self._frequencies

    @property
    def gaps_per_postion(self) -> np.ndarray:
        """This represents the number of gaped letters at each position in the sequence"""
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
