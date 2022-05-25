import argparse
import os
from json import loads, dumps

import PoseDirectory
import PathUtils as PUtils
import SymDesignUtils as SDUtils


def merge_scores(dirs1, dirs2):
    merged = []
    failed = []
    for dir1 in dirs1:
        for dir2 in dirs2:
            if str(dir1) == str(dir2):
                with open(os.path.join(dir1.scores, PUtils.scores_file), 'a') as f1:
                    with open(os.path.join(dir2.scores, PUtils.scores_file), 'a') as f2:
                        lines = [loads(line)for line in f2.readlines()]
                    f1.write('\n'.join(dumps(line) for line in lines))
                merged.append(dir1.path)
                break
        failed.append(dir1.path)

    return merged, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge the score files from two different design directories'
                                                 % PUtils.program_name)
    parser.add_argument('-d1', '--directory1', type=str, help='Directory 1 where %s poses are located. Default=CWD'
                                                              % PUtils.program_name, default=os.getcwd())
    parser.add_argument('-d2', '--directory2', type=str, help='Directory 2 where %s poses are located. Default=CWD'
                                                              % PUtils.program_name, default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of %s poses' % PUtils.program_name
                        , default=None)
    parser.add_argument('-m', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    args = parser.parse_args()

    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=1)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), propagate=True)

    logger.info('Starting %s with options:\n%s' %
                (os.path.basename(__file__),
                 '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Grab all poses (directories) to be processed from either directory name or file
    all_poses1, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
    assert all_poses1 != list(), 'No %s directories found within \'%s\' input! Please ensure correct location' \
                                 % (PUtils.nano, location)
    all_design_directories1 = PoseDirectory.set_up_directory_objects(all_poses1)
    logger.info('%d Poses found in \'%s\'' % (len(all_poses1), location))

    # Grab all poses (directories) to be processed from either directory name or file
    all_poses2, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
    assert all_poses2 != list(), 'No %s directories found within \'%s\' input! Please ensure correct location' \
                                 % (PUtils.nano, location)
    all_design_directories2 = PoseDirectory.set_up_directory_objects(all_poses2)
    logger.info('%d Poses found in \'%s\'' % (len(all_poses2), location))

    success, failures = merge_scores(all_design_directories1, all_design_directories2)
    if len(success) == len(all_design_directories1):
        logger.info('Success! All score files merged')
    else:
        logger.warning('The following score files failed to merged...')
        for fail in failures:
            logger.error(fail)
