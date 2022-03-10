import time
from shutil import copy

from PathUtils import biological_fragment_db_pickle
from SymDesignUtils import pickle_object
from Database import FragmentDatabase

# Create fragment database for all ijk cluster representatives
# ijk_frag_db = unpickle(biological_fragment_db_pickle)

ijk_frag_db = FragmentDatabase(init_db=True)
#
# # Get complete IJK fragment representatives database dictionaries
# ijk_frag_db.get_monofrag_cluster_rep_dict()
# ijk_frag_db.get_intfrag_cluster_rep_dict()
# ijk_frag_db.get_intfrag_cluster_info_dict()
copy(biological_fragment_db_pickle, '%s.bak-%s' % (biological_fragment_db_pickle, time.strftime('%y-%m-%d-%H%M%S')))
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')
