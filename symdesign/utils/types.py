from __future__ import annotations

from typing import Type, TypedDict

import numpy as np


class TransformationMapping(TypedDict):
    # Todo ?
    #  transformation: tuple[rotation, translation]
    rotation: Type[list[list[float]] | np.ndarray]
    rotation2: Type[list[list[float]] | np.ndarray]
    translation: Type[list[float] | np.ndarray]
    translation2: Type[list[float] | np.ndarray]
