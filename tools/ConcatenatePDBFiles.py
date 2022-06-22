import argparse
import os

from PDB import PDB
from SymDesignUtils import get_directory_file_paths, start_log

# globals
logger = start_log(name=os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate a group of structure files. The name of the resulting '
                                                 'file will be the individual structures concatenated in the same order'
                                                 ' as provided. For a directory, this is alphabetically')
    input_mutex = parser.add_mutually_exclusive_group(required=True)
    input_mutex.add_argument('-f', '--files', nargs='+', help='PDB files to be concatenated', default=[])
    input_mutex.add_argument('-d', '--directory', help='A directory with the PDB files to be concatenated')
    parser.add_argument('-Od', '--output_directory', type=os.path.abspath, required=True,
                        help='The directory where the concatenated PDB files should be written')
    args, additional_args = parser.parse_known_args()

    # input files
    if args.directory:
        args.files = get_directory_file_paths(args.directory, extension='.pdb')
    # initialize PDB Objects
    pdbs = [PDB.from_file(pdb_file, log=logger) for pdb_file in args.files]
    pdb = PDB.from_chains([chain for pdb in pdbs for chain in pdb.chains], name='-'.join(pdb.name for pdb in pdbs),
                          log=logger, entities=False)
    # output file
    os.makedirs(args.output_directory, exist_ok=True)
    pdb.write(out_path=os.path.join(args.output_directory, f'{pdb.name}.pdb'))
