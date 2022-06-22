import os
from glob import glob
import argparse

from SymDesignUtils import get_directory_file_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files the belong in a directory')
    parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where models are located',
                        required=True)
    parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for.', default='')
    parser.add_argument('-e', '--extension', type=str, help='An extension in the files to search for.')
    parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
                        required=True)

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------
    with open(args.out_path, 'w') as f:
        f.write('%s\n' % '\n'.join(get_directory_file_paths(args.directory, suffix=args.suffix,
                                                            extension=args.extension)))
