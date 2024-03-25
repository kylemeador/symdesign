from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

fragment_resources = Path('structure') / 'fragment' / 'resources'
biological_interfaces = 'biological_interfaces'
fragment_database_source_literal = Literal['biological_interfaces']  # , 'bio', 'xtal', 'bio_xtal']
fragment_database_sources = get_args(fragment_database_source_literal)
biological_fragment_db_pickle = fragment_resources / f'{biological_interfaces}.pkl'
frag_directory = {source: fragment_resources / source for source in fragment_database_sources}
monofrag_cluster_rep_dirpath = fragment_resources / 'fragment_representatives'
intfrag_cluster_rep_dirpath = fragment_resources / 'paired_representatives'
binary_lookup_table_path = fragment_resources / 'euler_lookup_40.npz'
