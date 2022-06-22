import os
import sys
from itertools import product

from FragDock import nanohedra_dock
from PathUtils import biological_fragment_db_pickle
from SymDesignUtils import unpickle
from classes.EulerLookup import euler_factory
from classes.SymEntry import symmetry_factory


print('USAGE:\nNavigate to a directory with the ".pdb" files of interest to generate fragment indices for and execute:'
      f'\npython {os.path.abspath(__file__)} file_name_for_ghost_fragments.pdb')
# Create fragment database for all ijk cluster representatives
ijk_frag_db = unpickle(biological_fragment_db_pickle)
# Load Euler Lookup table for each instance
euler_lookup = euler_factory()
sym_entry = symmetry_factory.get(261)
master_outdir = os.getcwd()
entities1, entities2 = [], []
for file in os.listdir(master_outdir):
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
        nanohedra_dock(sym_entry, ijk_frag_db, euler_lookup, master_outdir, pdb1, pdb2, write_frags=True)
    except RuntimeError as error:
        print(error)
        continue
