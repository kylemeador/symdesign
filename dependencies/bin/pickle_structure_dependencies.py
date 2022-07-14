import time
from shutil import copy, move

from PathUtils import biological_fragment_db_pickle, reference_aa_file, reference_residues_pkl
from Structure import Structure
from SymDesignUtils import pickle_object, unpickle
from fragment import FragmentDatabase

# Create fragment database for all ijk cluster representatives
input('Before executing this, ensure you modify the required_types in Structure._create_residues() to accept residues '
      'without "C" atoms. Ctrl-C to terminate. Enter to continue')
ijk_frag_db = FragmentDatabase()

copy(biological_fragment_db_pickle, f'{biological_fragment_db_pickle}.bak-{time.strftime("%y-%m-%d-%H%M%S")}')
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')

try:
    reference_residues = unpickle(reference_residues_pkl)  # 0 indexed, 1 letter aa, alphabetically sorted at the origin
    reference_aa = Structure.from_residues(residues=reference_residues)
except Exception:  # If something goes wrong, we should remake this too
    # Create and save the new reference_residues_pkl from scratch
    Structure.protein_backbone_atom_types = {'N', 'CA', 'O'}  # 'C', Removing 'C' for fragment library guide atoms...
    ref_aa = Structure.from_file(reference_aa_file)
    move(reference_residues_pkl, f'{reference_residues_pkl}.bak')
    pickle_object(ref_aa.residues, name=reference_residues_pkl, out_path='')
