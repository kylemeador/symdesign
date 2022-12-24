from __future__ import annotations

import csv
import logging
import os
from collections import namedtuple, defaultdict
from typing import Sequence, AnyStr, Iterable, Type, Literal, Any, get_args

import numpy as np
from Bio import pairwise2, SeqIO, AlignIO
from Bio.Align import substitution_matrices, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from symdesign import utils as sdutils
from symdesign.third_party.DnaChisel.dnachisel import reverse_translate, DnaOptimizationProblem, CodonOptimize, \
    EnforceGCContent, AvoidHairpins, AvoidPattern, UniquifyAllKmers, AvoidRareCodons, EnforceTranslation
# from symdesign.utils import path as putils
putils = sdutils.path

# Globals
zero_offset = 1
logger = logging.getLogger(__name__)
subs_matrices = {'BLOSUM62': substitution_matrices.load('BLOSUM62')}
Alignment = namedtuple('Alignment', 'seqA, seqB, score, start, end')

# Types
protein_letters3: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR')
protein_letters3_extended: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ASX', 'XAA', 'GLX', 'XLE', 'SEC', 'PYL')
protein_letters_alph1: str = 'ACDEFGHIKLMNPQRSTVWY'
protein_letters_alph1_extended: str = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
protein_letters_3to1: dict[str, str] = dict(zip(protein_letters3, protein_letters_alph1))
protein_letters_1to3: dict[str, str] = dict(zip(protein_letters_alph1, protein_letters3))
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
numerical_translation_alph1 = defaultdict(lambda: 20, zip(protein_letters_alph1,
                                                          range(len(protein_letters_alph1))))
numerical_translation_alph3 = defaultdict(lambda: 20, zip(protein_letters_alph3,
                                                          range(len(protein_letters_alph3))))
numerical_translation_alph1_bytes = defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph1],
                                                                range(len(protein_letters_alph1))))
numerical_translation_alph3_bytes = defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph3],
                                                                range(len(protein_letters_alph3))))
sequence_translation_alph1 = defaultdict(lambda: '-', zip(range(len(protein_letters_alph1)), protein_letters_alph1))
sequence_translation_alph3 = defaultdict(lambda: '-', zip(range(len(protein_letters_alph3)), protein_letters_alph3))
numerical_translation_alph1_gapped = defaultdict(lambda: 20, zip(protein_letters_alph1_gapped,
                                                                 range(len(protein_letters_alph1_gapped))))
numerical_translation_alph3_gapped = defaultdict(lambda: 20, zip(protein_letters_alph3_gapped,
                                                                 range(len(protein_letters_alph3_gapped))))
numerical_translation_alph1_gapped_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph1_gapped],
                                range(len(protein_letters_alph1_gapped))))
numerical_translation_alph3_gapped_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph3_gapped],
                                range(len(protein_letters_alph3_gapped))))
numerical_translation_alph1_unknown_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph1_unknown],
                                range(len(protein_letters_alph1_unknown))))
numerical_translation_alph3_unknown_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph3_unknown],
                                range(len(protein_letters_alph1_unknown))))
numerical_translation_alph1_unknown_gapped_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph1_unknown_gapped],
                                range(len(protein_letters_alph1_unknown_gapped))))
numerical_translation_alph3_unknown_gapped_bytes = \
    defaultdict(lambda: 20, zip([item.encode() for item in protein_letters_alph3_unknown_gapped],
                                range(len(protein_letters_alph1_unknown_gapped))))
extended_protein_letters_and_gap_literal = Literal[get_args(protein_letters_alph1_extended_literal), '-']
extended_protein_letters_and_gap: tuple[str, ...] = get_args(extended_protein_letters_and_gap_literal)
alphabet_types = Literal['protein_letters_alph1', 'protein_letters_alph3', 'protein_letters_alph1_gapped',
                         'protein_letters_alph3_gapped', 'protein_letters_alph1_unknown',
                         'protein_letters_alph3_unknown', 'protein_letters_alph1_unknown_gapped',
                         'protein_letters_alph3_unknown_gapped']
alphabet_to_type = {'ACDEFGHIKLMNPQRSTVWY': protein_letters_alph1,
                    'ARNDCQEGHILKMFPSTWYV': protein_letters_alph3,
                    'ACDEFGHIKLMNPQRSTVWY-': protein_letters_alph1_gapped,
                    'ARNDCQEGHILKMFPSTWYV-': protein_letters_alph3_gapped,
                    'ACDEFGHIKLMNPQRSTVWYX': protein_letters_alph1_unknown,
                    'ARNDCQEGHILKMFPSTWYVX': protein_letters_alph3_unknown,
                    'ACDEFGHIKLMNPQRSTVWYX-': protein_letters_alph1_unknown_gapped,
                    'ARNDCQEGHILKMFPSTWYVX-': protein_letters_alph3_unknown_gapped}


def create_translation_tables(alphabet_type: alphabet_types) -> defaultdict:
    """Given an amino acid alphabet type, return the corresponding numerical translation table.
    If a table is passed, just return it

    Returns:
        The integer mapping to the sequence of the requested alphabet
    """
    wrong_alphabet_type = ValueError(f"alphabet_type '{alphabet_type}' isn't viable")
    try:
        match alphabet_type:
            case 'protein_letters_alph1':
                numeric_translation_type = numerical_translation_alph1_bytes
            case 'protein_letters_alph3':
                numeric_translation_type = numerical_translation_alph3_bytes
            case 'protein_letters_alph1_gapped':
                numeric_translation_type = numerical_translation_alph1_gapped_bytes
            case 'protein_letters_alph3_gapped':
                numeric_translation_type = numerical_translation_alph3_gapped_bytes
            case 'protein_letters_alph1_unknown':
                numeric_translation_type = numerical_translation_alph1_unknown_bytes
            case 'protein_letters_alph3_unknown':
                numeric_translation_type = numerical_translation_alph3_unknown_bytes
            case 'protein_letters_alph1_unknown_gapped':
                numeric_translation_type = numerical_translation_alph1_unknown_gapped_bytes
            case 'protein_letters_alph3_unknown_gapped':
                numeric_translation_type = numerical_translation_alph3_unknown_gapped_bytes
            case _:
                try:  # To see if we already have the alphabet, and just return defaultdict
                    numeric_translation_type = alphabet_to_type[alphabet_type]
                except KeyError:
                    raise wrong_alphabet_type
    except SyntaxError:  # python version not 3.10
        if alphabet_type == 'protein_letters_alph1':
            numeric_translation_type = numerical_translation_alph1_bytes
        elif alphabet_type == 'protein_letters_alph3':
            numeric_translation_type = numerical_translation_alph3_bytes
        elif alphabet_type == 'protein_letters_alph1_gapped':
            numeric_translation_type = numerical_translation_alph1_gapped_bytes
        elif alphabet_type == 'protein_letters_alph3_gapped':
            numeric_translation_type = numerical_translation_alph3_gapped_bytes
        elif alphabet_type == 'protein_letters_alph1_unknown':
            numeric_translation_type = numerical_translation_alph1_unknown_bytes
        elif alphabet_type == 'protein_letters_alph3_unknown':
            numeric_translation_type = numerical_translation_alph3_unknown_bytes
        elif alphabet_type == 'protein_letters_alph1_unknown_gapped':
            numeric_translation_type = numerical_translation_alph1_unknown_gapped_bytes
        elif alphabet_type == 'protein_letters_alph3_unknown_gapped':
            numeric_translation_type = numerical_translation_alph3_unknown_gapped_bytes
        else:
            try:  # To see if we already have the alphabet, and return the defaultdict
                numeric_translation_type = alphabet_to_type[alphabet_type]
            except KeyError:
                raise wrong_alphabet_type

    return numeric_translation_type


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
                    formatted_sequence_gen = (f'{start}{name}{sep}{"".join(seq)}' for name, seq in zip(names, sequences))
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
            name, seq, *_ = sequences
            formatted_sequence_gen = (f'{start}{name}{sep}{seq}',)
        elif isinstance(names, str):  # Assume sequences is a str or tuple
            formatted_sequence_gen = (f'{start}{names}{sep}{"".join(sequences)}\n',)
        else:
            raise TypeError(data_dump())
        outfile.write('%s\n' % '\n'.join(formatted_sequence_gen))

    return file_name


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
    #     args.multicistronic_intergenic_sequence = expression.default_multicistronic_sequence

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
    if args.suffix:
        args.suffix = f'_{args.suffix}'

    nucleotide_sequence_file = write_sequences(nucleotide_sequences, csv=args.csv,
                                               file_name=os.path.join(os.getcwd(),
                                                                      f'{args.prefix}MulticistronicNucleotideSequences'
                                                                      f'{args.suffix}'))
    logger.info(f'Multicistronic nucleotide sequences written to: {nucleotide_sequence_file}')


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


protein_letters_alph3_gapped_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
mutation_entry = Type[dict[Literal['to', 'from'], protein_letters_alph3_gapped_literal]]
"""Mapping of a reference sequence amino acid type, 'to', and the resulting sequence amino acid type, 'from'"""
mutation_dictionary = dict[int, mutation_entry]
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


def numeric_to_sequence(numeric_sequence: np.ndarray, alphabet_order: int = 1) -> np.ndarray:
    """Convert a numeric sequence array into a sequence array

    Args:
        numeric_sequence: The sequence to convert
        alphabet_order: The alphabetical order of the amino acid alphabet. Can be either 1 or 3
    Returns:
        The alphabetically encoded sequence where each entry along axis=-1 is the one letter amino acid
    """
    if alphabet_order == 1:
        return np.vectorize(sequence_translation_alph1.__getitem__)(numeric_sequence)
    elif alphabet_order == 3:
        return np.vectorize(sequence_translation_alph3.__getitem__)(numeric_sequence)
    else:
        raise ValueError(f"The alphabet_order {alphabet_order} isn't valid. Choose from either 1 or 3")


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
