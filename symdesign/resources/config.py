from __future__ import annotations

from typing import Literal, get_args

from symdesign.utils import path as putils

MAXIMUM_SEQUENCE = 10000
MAXIMUM_ENTITIES = 4
MAXIMUM_INTERFACES = 2
relax_options_literal = Literal['all', 'best', 'none']
relax_options: tuple[str, ...] = get_args(relax_options_literal)
default_weight_parameter: dict[str, str] = {
    putils.rosetta: 'interface_energy',
    putils.proteinmpnn: 'proteinmpnn_score_complex',
    putils.nanohedra: 'nanohedra_score_center',  # 'nanohedra_score_center_normalized',
    f'{putils.nanohedra}+{putils.proteinmpnn}': 'proteinmpnn_dock_cross_entropy_per_residue',
}
default_pca_variance = 0.8
