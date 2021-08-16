import os
import argparse
from csv import reader

from SymDesignUtils import io_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files the belong in a directory')
    # parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where models are located',
    #                     required=True)
    parser.add_argument('-f', '--file', type=os.path.abspath, help='File where pose IDs are located', required=True)
    parser.add_argument('-k', '--keep_design_id', action='store_true', help='An extension in the files to search for.')
    # parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for.', default='')
    # parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
    #                     required=True)

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------

    with open(args.file) as file:
        pose_ids, *_ = zip(*reader(file))

    if args.keep_design_id:
        pose_ids = [pose_id for pose_id in map(str.strip, pose_ids) if pose_id != '']
    else:  # assumes each design will have _clean_asu suffix appended
        pose_ids = [pose_id.split('_clean_asu')[0] for pose_id in map(str.strip, pose_ids) if pose_id != '']

    io_save(pose_ids)
