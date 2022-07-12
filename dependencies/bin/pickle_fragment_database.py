import time
from shutil import copy, move

from PathUtils import biological_fragment_db_pickle, reference_aa_file, reference_residues_pkl
from Structure import Structure
from SymDesignUtils import pickle_object
from JobResources import FragmentDatabase

# Create fragment database for all ijk cluster representatives
# ijk_frag_db = unpickle(biological_fragment_db_pickle)
input('Before executing this, ensure you modify the required_types in Structure._create_residues() to accept residues '
      'without "C" atoms. Ctrl-C to terminate. Enter to continue')
ijk_frag_db = FragmentDatabase()

copy(biological_fragment_db_pickle, f'{biological_fragment_db_pickle}.bak-{time.strftime("%y-%m-%d-%H%M%S")}')
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')

# make the Residue reference Structure from scratch
ref_aa = Structure.from_file(reference_aa_file, log=None, entities=False)
move(reference_residues_pkl, f'{reference_residues_pkl}.bak')
pickle_object(ref_aa.residues, name=reference_residues_pkl, out_path='')
