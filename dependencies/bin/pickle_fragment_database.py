import time
from shutil import copy

from PathUtils import biological_fragment_db_pickle
from SymDesignUtils import pickle_object
from JobResources import FragmentDatabase

# Create fragment database for all ijk cluster representatives
# ijk_frag_db = unpickle(biological_fragment_db_pickle)
input('Before executing this, ensure you modify the required_types in Structure._create_residues() to accept residues '
      'without "C" atoms. Ctrl-C to terminate. Enter to continue')
ijk_frag_db = FragmentDatabase()

copy(biological_fragment_db_pickle, '%s.bak-%s' % (biological_fragment_db_pickle, time.strftime('%y-%m-%d-%H%M%S')))
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')
