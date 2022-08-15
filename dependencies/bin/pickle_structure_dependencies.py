from shutil import copy, move

from resources.fragment import FragmentDatabase
from utils.path import biological_fragment_db_pickle, reference_aa_file, reference_residues_pkl
from utils import timestamp, pickle_object

try:
    from structure import base
except Exception:  # If something goes wrong, we should remake this too
    # Create and save the new reference_residues_pkl from scratch
    ref_aa = base.Structure.from_file(reference_aa_file)
    move(reference_residues_pkl, f'{reference_residues_pkl}.bak')
    pickle_object(ref_aa.residues, name=reference_residues_pkl, out_path='')

# Create fragment database for all ijk cluster representatives
# This should work now
from structure import base
base.protein_backbone_atom_types = {'N', 'CA', 'O'}  # 'C', Removing 'C' for fragment library guide atoms...
ijk_frag_db = FragmentDatabase()

copy(biological_fragment_db_pickle, f'{biological_fragment_db_pickle}.bak-{timestamp}')
pickle_object(ijk_frag_db, name=biological_fragment_db_pickle, out_path='')
