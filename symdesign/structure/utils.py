from __future__ import annotations

import _pickle
import logging
from collections.abc import Generator

from symdesign import utils

logger = logging.getLogger(__name__)
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
except (_pickle.UnpicklingError, ImportError):
    logger.error(f'The reference_residues ran into an error upon load. You need to regenerate the serialized version!')
    reference_residues = None
