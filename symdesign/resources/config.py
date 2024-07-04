from __future__ import annotations

from typing import Literal, get_args

MAXIMUM_SEQUENCE = 10_000
MAXIMUM_ENTITIES = 4
default_weight_parameter: dict[str, str] = {
    'rosetta': 'interface_energy',
    'proteinmpnn': 'proteinmpnn_score_complex',
    'nanohedra': 'nanohedra_score_center',
    'nanohedra+proteinmpnn': 'proteinmpnn_dock_cross_entropy_per_residue',
}
default_pca_variance = 0.8
