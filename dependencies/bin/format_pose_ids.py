import os
import argparse
from csv import reader

from PathUtils import submodule_help
from SymDesignUtils import io_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files the belong in a directory')
    # parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where models are located',
    #                     required=True)
    parser.add_argument('-f', '--file', type=os.path.abspath, help='File where pose IDs are located', required=True)
    # parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for.', default='')
    # parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
    #                     required=True)
    subparsers = parser.add_subparsers(title='Modules', dest='module',
                                       description='These are the different modes that Pose IDs can be processed',
                                       help='Chose a Module followed by Module specific flags. To get help with a '
                                            'Module flags enter:\t%s\n.' % submodule_help)
    # ---------------------------------------------------
    extract_parser = subparsers.add_parser('extract', help='Extract Pose IDs from file for poses of interest.')
    extract_parser.add_argument('-k', '--keep_design_id', action='store_true',
                                help='Whether to keep a Design ID appended to the Pose ID')
    extract_parser.add_argument('-s', '--split_design_id', action='store_true',
                                help='Whether an additional design name should be split into pose, design pairs')
    extract_parser.add_argument('-p', '--project', type=str, help='Add a project directory to the Pose IDs')

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------

    with open(args.file) as file:
        pose_ids, *_ = zip(*reader(file))

    if args.project:
        # if os.path.exists():
        # convert path to Pose ID format
        project = args.project.replace(os.sep, '-')
        if not project.endswith('-'):
            project += '-'
    else:
        project = ''

    if args.keep_design_id:
        pose_ids = [project + pose_id for pose_id in map(str.strip, pose_ids) if pose_id != '']
    elif args.split_design_id:  # assumes each design will have _clean_asu suffix appended
        pose_ids = ['%s, clean_asu%s' % (project + pose_id.split('_clean_asu')[0], pose_id.split('_clean_asu')[0])
                    for pose_id in map(str.strip, pose_ids) if pose_id != '']
    else:  # assumes each design will have _clean_asu suffix appended
        pose_ids = [project + pose_id.split('_clean_asu')[0] for pose_id in map(str.strip, pose_ids) if pose_id != '']

    io_save(pose_ids)
