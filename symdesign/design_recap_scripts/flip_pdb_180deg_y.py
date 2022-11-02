import os
import sys

from structure.model import Model
from symdesign import utils

pdb_filepath = sys.argv[1]

pdb = Model.from_file(pdb_filepath)
flipped_pdb = pdb.get_transformed_copy(rotation=utils.symmetry.flip_y_matrix)
flipped_pdb.write('%s_flipped_180y.pdb' % os.path.splitext(pdb_filepath)[0])
