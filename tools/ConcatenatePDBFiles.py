import argparse
import os

from PDB import PDB
from SymDesignUtils import get_all_file_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate a list of PDB files provided in a directory')
    input_mutex = parser.add_mutually_exclusive_group()
    input_mutex.add_argument('-f', '--files', nargs='+', help='PDB files to be concatenated', default=[])
    input_mutex.add_argument('-d', '--directory', help='A directory with the PDB files to be concatenated')
    parser.add_argument('-Od', '--output_directory',
                        help='The directory where the concatenated PDB files should be written')

    args, additional_args = parser.parse_known_args()

    if args.directory:
        args.files = get_all_file_paths(args.directory, extension='.pdb')
    pdbs = [PDB.from_file(pdb_file, log=None) for pdb_file in args.files]
    pdb = PDB.from_chains([pdb.chain for pdb in pdbs], name=''.join(pdb.name for pdb in pdbs), log=None, entities=False)
    pdb.write(out_path=os.path.join(args.output_directory, ), increment_chains=True)
