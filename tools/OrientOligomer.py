import os
import sys

from PDB import PDB
from SymDesignUtils import get_all_file_paths


def orient_oligomer(pdb_path, sym, out_dir=os.getcwd()):
    pdb = PDB.from_file(pdb_path)
    pdb.orient(symmetry=sym)
    return pdb.write(out_path=os.path.join(out_dir, os.path.basename(os.path.splitext(pdb_path)[0]) + '_oriented.pdb'))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('USAGE: python OrientOligomer.py symmetry_type pdb_or_pdb_directory_path absolute_output_path')
    else:
        sym = sys.argv[1].upper()
        pdb_paths = sys.argv[2]

        out_dir = sys.argv[3]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if '.pdb' in pdb_paths and type(pdb_paths) != list:
            all_pdb_paths = [pdb_paths, ]
        else:
            all_pdb_paths = get_all_file_paths(pdb_paths, extension='.pdb')

        all_oriented_files = [orient_oligomer(file_path, sym, out_dir=out_dir) for file_path in all_pdb_paths]
        print('All files were \'attempted\' to be oriented.\nReturned filenames include:\n%s'
              % '\n'.join(all_oriented_files))
