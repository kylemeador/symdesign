from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Literal, get_args

logger = logging.getLogger(__name__)
available_letters: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # '0123456789~!@#$%^&*()-+={}[]|:;<>?'
coords_type_literal = Literal['all', 'backbone', 'backbone_and_cb', 'ca', 'cb', 'heavy']
coords_types: tuple[coords_type_literal, ...] = get_args(coords_type_literal)
default_clash_criteria = 'backbone_and_cb'
default_clash_distance = 2.1
design_programs_literal = Literal['consensus', 'proteinmpnn', 'rosetta']
termini_literal = Literal['n', 'c']


def chain_id_generator() -> Generator[str, None, None]:
    """Provide a generator which produces all combinations of chain ID strings

    Returns
        The generator producing a maximum 2 character string where single characters are exhausted,
            first in uppercase, then in lowercase
    """
    # Todo
    #  Some PDB file use numeric chains... so chain 1 for instance couldn't be parsed correctly
    return (first + second for modification in ['upper', 'lower']
            for first in [''] + list(getattr(available_letters, modification)())
            for second in list(getattr(available_letters, 'upper')()) +
            list(getattr(available_letters, 'lower')()))


class StructureException(Exception):
    pass


class DesignError(StructureException):
    pass


class ConstructionError(StructureException):
    pass


class ClashError(StructureException):
    pass


class SymmetryError(StructureException):
    pass


def parse_stride(stride_file, **kwargs):
    """From a Stride file, parse information for residue level secondary structure assignment

    Sets:
        self.secondary_structure
    """
    with open(stride_file, 'r') as f:
        stride_output = f.readlines()

    return ''.join(line[24:25] for line in stride_output if line[0:3] == 'ASG')
