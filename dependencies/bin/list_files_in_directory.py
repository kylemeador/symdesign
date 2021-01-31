import os
import argparse


def get_all_pdb_file_paths(pdb_dir):
    return [os.path.join(os.path.abspath(root), file) for root, dirs, files in os.walk(pdb_dir) for file in files
            if '.pdb' in file]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files the belong in a directory')
    parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where designed poses are located.',
                        required=True)
    parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
                        required=True)

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------

    with open(args.out_path, 'w') as f:
        f.write('\n'.join(pdb for pdb in get_all_pdb_file_paths(args.directory)))
