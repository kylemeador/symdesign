import argparse
import os
from itertools import repeat

import pandas as pd

from old import AnalyzeMutatedSequences as Ams
import DesignMetrics as AOut
import DesignDirectory
import PathUtils as PUtils
# import PDB
import SymDesignUtils as SDUtils

# import CmdUtils as CUtils
logger = SDUtils.start_log(name=__name__)
per_res_keys = ['jsd', 'int_jsd']


def pose_jsd(des_dir, debug=False):
    other_pose_metrics = {}
    all_design_files = des_dir.get_designs()
    pose_res_dict = Ams.analyze_mutations(des_dir,
                                          AnalyzeMutatedSequences.mutate_wildtype_sequences(all_design_files, DesignDirectory.get_wildtype_file(des_dir)))
    for key in per_res_keys:
        other_pose_metrics[key + '_per_res'] = AOut.per_res_metric(pose_res_dict, key=key)

    jsd_s = pd.Series(other_pose_metrics)
    jsd_s = pd.concat([jsd_s], keys=['dock'])
    jsd_s = pd.concat([jsd_s], keys=['pose'])
    jsd_s.name = str(des_dir)

    return jsd_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nPose JSD. Requires a '
                                                 'directory with wild-type PDB/PSSM/flags_design files, a folder of '
                                                 'design PDB files, and a score.sc file with json objects as input.'
                                                 % __name__)
    parser.add_argument('-d', '--directory', type=str, help='Directory where Nanohedra output is located. Default=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of Nanohedra output.', default=None)
    parser.add_argument('-m', '--multi_processing', action='store_true', help='Should job be run with multiprocessing? '
                                                                              'Default=False')
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out? Default=False')

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

    # Collect all designs to be processed
    all_designs, location = SDUtils.collect_designs(files=args.file, directory=args.directory)
    assert all_designs != list(), 'No %s directories found within \'%s\' input! Please ensure correct location'\
                                  % (PUtils.nano, location)
    logger.info('%d Poses found in \'%s\'' % (len(all_designs), location))
    # logger.info('All pose specific logs are located in their corresponding directories.\nEx: \'%s\'' %
    #             os.path.join(all_designs[0].path, os.path.basename(all_designs[0].path) + '.log'))

    # Start Pose processing
    if args.multi_processing:
        # Calculate the number of cores to use depending on computer resources
        cores = SDUtils.calculate_mp_cores()
        logger.info('Starting multiprocessing using %s cores' % str(cores))
        zipped_args = zip(all_designs, repeat(args.debug))
        pose_results, exceptions = SDUtils.old_mp_starmap(pose_jsd, zipped_args, cores)
        print(type(pose_results[0]))
        # design_df = pd.concat([result for result in pose_results])
    else:
        logger.info('Starting processing. If single process is taking awhile, use -m during submission')
        pose_results, exceptions = [], []
        for des_directory in all_designs:
            result = pose_jsd(des_directory, debug=args.debug)  # ,error
            pose_results.append(result)
            # exceptions.append(error)
        # design_df = pd.DataFrame(pose_results)

    # print(pose_results)
    #
    # failures = [i for i in range(len(exceptions)) if exceptions[i]]
    # for index in reversed(failures):
    #     del pose_results[index]
    #
    # exceptions = list(set(exceptions))
    # if len(exceptions) == 1 and exceptions[0]:
    #     logger.warning('\nThe following exceptions were thrown. Design for these directories is inaccurate\n')
    #     for exception in exceptions:
    #         logger.warning(exception)

    if len(all_designs) >= 1:
        # print(type(pose_results[0]))
        design_df = pd.DataFrame(pose_results)
        print(design_df)
        # Save Design dataframe
        out_path = os.path.join(args.directory, 'jsd_addition.csv')  # args.directory
        if os.path.exists(out_path):
            design_df.to_csv(out_path, mode='a', header=False)
        else:
            design_df.to_csv(out_path)
        logger.info('All Design Pose Analysis written to %s' % out_path)
