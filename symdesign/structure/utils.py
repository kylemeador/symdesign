from __future__ import annotations

import _pickle
import logging
from collections import defaultdict
from collections.abc import Generator
from typing import get_args, Literal

from symdesign import utils

logger = logging.getLogger(__name__)
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
profile_keys = Literal[protein_letters_literal, 'lod', 'type', 'info', 'weight']
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
protein_letters_alph3_gapped_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
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
            case other:
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


available_letters: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # '0123456789~!@#$%^&*()-+={}[]|:;<>?'


def chain_id_generator() -> Generator[str, None, None]:
    """Provide a generator which produces all combinations of chain ID strings

    Returns
        The generator producing a maximum 2 character string where single characters are exhausted,
            first in uppercase, then in lowercase
    """
    return (first + second for modification in ['upper', 'lower']
            for first in [''] + list(getattr(available_letters, modification)())
            for second in list(getattr(available_letters, 'upper')()) +
            list(getattr(available_letters, 'lower')()))


class DesignError(Exception):
    pass


class ConstructionError(DesignError):
    pass


class ClashError(DesignError):
    pass


class SymmetryError(DesignError):
    pass


# 0 indexed, 1 letter aa, alphabetically sorted at the origin
try:
    reference_residues = utils.unpickle(utils.path.reference_residues_pkl)
except _pickle.UnpicklingError:
    logger.error(f'The reference_residues ran into an error upon load. You need to regenerate the serialized version!')
    reference_residues = None
