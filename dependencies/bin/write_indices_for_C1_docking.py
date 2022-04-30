import os
from itertools import product

from FragDock import nanohedra_dock
from PathUtils import biological_fragment_db_pickle
from SymDesignUtils import unpickle
from classes.EulerLookup import EulerLookup
from classes.SymEntry import SymEntry


# Create fragment database for all ijk cluster representatives
ijk_frag_db = unpickle(biological_fragment_db_pickle)
# Load Euler Lookup table for each instance
euler_lookup = EulerLookup()
sym_entry = SymEntry(261)
master_outdir = os.getcwd()
entities1, entities2 = [], []

for file in os.listdir(master_outdir):
    if file.startswith('1nu4'):
        entities1.append(file)
    else:
        entities2.append(file)

for pdb1, pdb2 in list(product(entities1, entities2)):
    try:
        nanohedra_dock(sym_entry, ijk_frag_db, euler_lookup, master_outdir, pdb1, pdb2)
    except RuntimeError as error:
        print(error)
        continue
