import os
from glob import glob
import argparse


def get_all_pdb_file_paths(pdb_dir, suffix='', extension='.pdb'):
    return [os.path.join(os.path.abspath(pdb_dir), file)
            for file in glob(os.path.join(pdb_dir, '*%s*%s' % (suffix, extension)))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files the belong in a directory')
    parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where designed poses are located.',
                        required=True)
    parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for.')
    parser.add_argument('-e', '--extension', type=str, help='An extension in the files to search for.')
    parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
                        required=True)

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------

    with open(args.out_path, 'w') as f:
        f.write('\n'.join(pdb for pdb in get_all_pdb_file_paths(args.directory, suffix=args.suffix)))  # Todo extension
        f.write('\n')
