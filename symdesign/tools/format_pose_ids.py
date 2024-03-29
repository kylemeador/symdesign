import os
import argparse
import re
from csv import reader
from itertools import repeat

from symdesign import flags
from symdesign import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of files that should be grouped into a continuous job')
    # parser.add_argument('-d', '--directory', type=os.path.abspath, help='Directory where models are located',
    #                     required=True)
    parser.add_argument(*flags.file_args, type=os.path.abspath, help='File where pose IDs are located', required=True)
    parser.add_argument(*flags.output_file_args, type=os.path.abspath,
                        help='File where pose IDs should be written')
    # parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for.', default='')
    # parser.add_argument('-o', '--out_path', type=os.path.abspath, help='Directory where list should be written.',
    #                     required=True)
    parser.add_argument('-k', '--keep-design-id', action='store_true',
                        help='Whether to keep a Design ID appended to the Pose ID')
    parser.add_argument('-s', '--split-design-id', action='store_true',
                        help='Whether an additional design name should be split into pose, design pairs')
    parser.add_argument('-p', '--project', type=str, help='Add a project directory to the Pose IDs')
    parser.add_argument('-E', '--include-extra-info', action='store_true',
                        help='Whether to include extra file info in the returned pose IDs')
    subparsers = parser.add_subparsers(title='Modules', dest='module',
                                       description='These are the different modes that Pose IDs can be processed',
                                       help='Chose a Module followed by Module specific flags. To get help with a '
                                            f'Module flags enter:\t{utils.path.submodule_help}.\n')
    # ---------------------------------------------------
    filter_parser = subparsers.add_parser('filter', help='Filter Pose IDs from file by a selection file to retrieve '
                                                         'poses of interest.')
    filter_parser.add_argument('-sf', '--selection-id-file', type=str,
                               help='The file used to filter (select) IDs of interest')
    # ---------------------------------------------------
    extract_parser = subparsers.add_parser('extract', help='Extract Pose IDs from file for poses of interest.')
    # extract_parser.add_argument('-k', '--keep_design_id', action='store_true',
    #                             help='Whether to keep a Design ID appended to the Pose ID')
    # extract_parser.add_argument('-s', '--split_design_id', action='store_true',
    #                             help='Whether an additional design name should be split into pose, design pairs')
    # extract_parser.add_argument('-p', '--project', type=str, help='Add a project directory to the Pose IDs')

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------

    if args.file.endswith('.csv'):  # assumes from a metrics.csv or a sequences.csv
        with open(args.file) as file:
            pose_id_lines, *extra_info = zip(*reader(file))
    else:
        with open(args.file) as file:
            # pose_id_lines = list(map(str.strip, file.readlines()))
            pose_id_lines, *extra_info = zip(*map(str.split, map(str.strip, file.readlines()), repeat(',')))

    if args.project:
        # if os.path.exists():
        # convert path to Pose ID format
        project = args.project.replace(os.sep, '-')
        if not project.endswith('-'):
            project += '-'
    else:
        project = ''

    # # extract only pose_ids from the input pose_id_lines
    # pose_ids = list(filter(re.compile(r'.*(\w{4})_(\w{4})[/-]'
    #                                   '.*_?[0-9]_[0-9][/-]'
    #                                   '.*_?([0-9]+)_([0-9]+)[/-]'
    #                                   '[tT][xX]_([0-9]+).*').match, pose_id_lines))
    # # print('POSE IDS', pose_ids[:5])
    pose_ids = pose_id_lines
    if not pose_ids:  # Don't have names generated from Nanohedra Docking
        pose_ids = pose_id_lines
        print(pose_ids[:5])
        # raise NotImplementedError('The format_pose_ids tool needs to be adapted for non-Nanohedra docking entries')

    # if args.keep_design_id:
    #     final_pose_ids = [project + pose_id for pose_id in map(str.strip, pose_ids) if pose_id != '']
    # elif args.split_design_id:  # assumes each design will have _clean_asu suffix appended
    #     final_pose_ids = ['{}{},clean_asu{}'.format(project, *pose_id.split('_clean_asu'))
    #                       for pose_id in map(str.strip, pose_ids) if pose_id != '']
    # else:  # assumes each design will have _clean_asu suffix appended
    #     final_pose_ids = \
    #         [project + pose_id.split('_clean_asu')[0] for pose_id in map(str.strip, pose_ids) if pose_id != '']

    if args.design_id:
        final_pose_ids = [project + pose_id for pose_id in map(str.strip, pose_ids) if pose_id != '']
    elif args.split_design_id:  # assumes each design will have _clean_asu suffix appended
        final_pose_ids = ['{}{},clean_asu{}'.format(project, *pose_id.split('_clean_asu'))
                          for pose_id in map(str.strip, pose_ids) if pose_id != '']
    else:  # assumes each design will have _clean_asu suffix appended
        final_pose_ids = \
            [project + pose_id.split('_clean_asu')[0] for pose_id in map(str.strip, pose_ids) if pose_id != '']

    if args.module == 'filter':
        with open(args.selection_id_file) as file:
            lines = file.readlines()
            selection_id_lines = list(map(str.strip, lines))

        selection_ids = list(filter(re.compile(r'.*(\w{4})_(\w{4})[/-]'
                                               '.*_?[0-9]_[0-9][/-]'
                                               '.*_?([0-9]+)_([0-9]+)[/-]'
                                               '[tT][xX]_([0-9]+).*').match, selection_id_lines))
        # print('Selection IDS', selection_ids[:5])
        selected_pose_ids = []
        for id1 in final_pose_ids:
            for id2 in selection_ids:
                if id1 in id2:
                    if id1 not in selected_pose_ids:
                        selected_pose_ids.append(id1)
                        # break
        final_pose_ids = selected_pose_ids
        # print('INTO include_extra_info POSE IDS', final_pose_ids[:5])

    else:  # not really sure how extract should work differently...
        pass

    if args.include_extra_info:
        # must format data to how it was retrieved (only works with .csv for now Todo
        final_groups = []
        for group in zip(pose_id_lines, *extra_info):
            for final_id in final_pose_ids:
                if final_id in group[0]:
                    final_groups.append(group)
        final_pose_ids = list(map(str.join, repeat(','), final_groups))
        # print('GROUPS', final_groups[:5])
        # print('FINAL POSE IDS', final_pose_ids[:5])

    utils.io_save(final_pose_ids, file_name=args.output_file)
