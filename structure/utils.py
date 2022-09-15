from __future__ import annotations

from typing import get_args, Literal

protein_letters3: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR')
extended_protein_letters3: tuple[str, ...] = \
    ('ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN',
     'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'ASX', 'XAA', 'GLX', 'XLE', 'SEC', 'PYL')
protein_letters: str = 'ACDEFGHIKLMNPQRSTVWY'
extended_protein_letters: str = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'
protein_letters_3to1: dict[str, str] = dict(zip(protein_letters3, protein_letters))
protein_letters_1to3: dict[str, str] = dict(zip(protein_letters, protein_letters3))
protein_letters_3to1_extended: dict[str, str] = dict(zip(extended_protein_letters3, extended_protein_letters))
protein_letters_1to3_extended: dict[str, str] = dict(zip(extended_protein_letters, extended_protein_letters3))
protein_letters3_literal = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
_alph_3_aa: tuple[protein_letters3_literal, ...] = get_args(protein_letters3_literal)
alph_3_aa = ''.join(_alph_3_aa)
protein_letter_plus_literals = Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                       'T', 'W', 'Y', 'V', 'lod', 'type', 'info', 'weight']
protein_letters_1aa_literal = Literal[tuple(protein_letters)]
protein_letters_unknown = protein_letters + 'X'
protein_letters3_unknown = alph_3_aa + 'X'
protein_letters_gapped = protein_letters + '-'
protein_letters3_gapped = alph_3_aa + '-'
protein_letters_unknown_gapped = protein_letters + 'X-'
protein_letters_unknown_gapped3 = alph_3_aa + 'X-'
extended_protein_letters_literal = Literal[tuple(extended_protein_letters)]
protein_letters3_literal_gapped = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
protein_letters3_literal_unknown_gapped = \
    Literal['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
            'X', '-']
