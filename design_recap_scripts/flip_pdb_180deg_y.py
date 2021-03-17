import os
import sys

from PDB import PDB
# from utils.PDBUtils import rot_txint_set_txext_pdb

# September 29th 2020
# Joshua Laniado

pdb_filepath = sys.argv[1]
flipped_pdb_out_path = os.path.splitext(pdb_filepath)[0] + "_flipped_180y.pdb"

rot_mat1 = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]  # rot 180y

pdb = PDB()
pdb.readfile(pdb_filepath)

# flipped_pdb = rot_txint_set_txext_pdb(pdb, rot_mat=rot_mat1)
flipped_pdb = pdb.return_transformed_copy(rotation=rot_mat1)
flipped_pdb.write(flipped_pdb_out_path)
