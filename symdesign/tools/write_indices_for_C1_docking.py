import os
import sys
from itertools import product

from symdesign.protocols.fragdock import fragment_dock
# from utils.path import biological_fragment_db_pickle
# from utils import unpickle
# from resources.EulerLookup import euler_factory
from symdesign.utils.path import program_command

raise NotImplementedError(f'This tool has been depreciated. '
                          f'Use "{program_command} nanohedra --only-write-frag-info" instead')
print('USAGE:\nNavigate to a directory with the ".pdb" files of interest to generate fragment indices for and execute:'
      f'\npython {os.path.abspath(__file__)} file_name_for_ghost_fragments.pdb directory_for_output')

sym_entry = utils.SymEntry.symmetry_factory.get(161)
root_out_dir = sys.argv[2]  # os.getcwd()
entities1, entities2 = [], []
for file in os.listdir(root_out_dir):
    if '.pdb' not in file:
        continue
    # if file.startswith('1nu4'):  # U1a
    if file == sys.argv[1]:
        entities1.append(file)
    else:
        entities2.append(file)

if not entities1:
    raise ValueError(f'No files found for {sys.argv[1]}')

for pdb1, pdb2 in list(product(entities1, entities2)):
    try:
        nanohedra_dock(sym_entry, root_out_dir, pdb1, pdb2, write_frags_only=True)
    except RuntimeError as error:
        print(error)
        continue
