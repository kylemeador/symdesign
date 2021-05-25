import os
import sys

from PDB import PDB
from classes.SymEntry import flip_y_matrix

pdb_filepath = sys.argv[1]

pdb = PDB.from_file(pdb_filepath)
flipped_pdb = pdb.return_transformed_copy(rotation=flip_y_matrix)
flipped_pdb.write('%s_flipped_180y.pdb' % os.path.splitext(pdb_filepath)[0])
