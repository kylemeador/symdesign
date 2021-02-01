"""
Module for distribution of SymDesign commands. Includes pose initialization, distribution of Rosetta commands to
SLURM/PBS computational clusters, analysis of designed poses, and renaming of completed structures.

"""
import argparse
import datetime
import os
import shutil
import subprocess
import time
from csv import reader
from glob import glob
from itertools import repeat, product, combinations
from json import loads, dumps

import pandas as pd

import AnalyzeMutatedSequences
import AnalyzeMutatedSequences as Ams
import CmdUtils as CUtils

from DesignDirectory import DesignDirectory, set_up_directory_objects
import PathUtils as PUtils
import Pose
import SequenceProfile
import SymDesignUtils as SDUtils
from AnalyzeOutput import analyze_output_s, analyze_output_mp
from CommandDistributer import distribute
from NanohedraWrap import nanohedra_command_mp, nanohedra_command_s, nanohedra_recap_mp, nanohedra_recap_s
from PDB import PDB, generate_sequence_template
from PathUtils import interface_design_command
from PoseProcessing import pose_rmsd_s, pose_rmsd_mp, cluster_poses
# from AnalyzeMutatedSequences import filter_pose, get_pdb_sequences, select_sequences_s, select_sequences_mp, write_fasta_file
from ProteinExpression import find_all_matching_pdb_expression_tags, add_expression_tag, find_expression_tags
from Query.Flags import query_user_for_flags, return_default_flags
from classes.SymEntry import SymEntry
from utils.CmdLineArgParseUtils import query_mode


def rename(des_dir, increment=PUtils.nstruct):
    """Rename the decoy numbers in a DesignDirectory by a specified increment

    Args:
        des_dir (DesignDirectory): A DesignDirectory object
    Keyword Args:
        increment=PUtils.nstruct (int): The number to increment by
    """
    for pdb in os.listdir(des_dir.design_pdbs):
        if os.path.splitext(pdb)[0][-1].isdigit():
            SDUtils.change_filename(os.path.join(des_dir.design_pdbs, pdb), increment=increment)
    SDUtils.modify_decoys(os.path.join(des_dir.scores, PUtils.scores_file), increment=increment)


def pair_directories(dirs2, dirs1):
    """Pair directories with the same pose name, returns source (dirs2) first, destination (dirs1) second
    Args:
        dirs2 (list): List of DesignDirectory objects
        dirs1 (list): List of DesignDirectory objects
    Returns:
        (list), (list): [(source, destination), ...], [directories missing a pair, ...]
    """
    success, pairs = [], []
    for dir1 in dirs1:
        for dir2 in dirs2:
            if str(dir1) == str(dir2):
                pairs.append((dir2, dir1))
                success.append(dir1)
                dirs2.remove(dir2)
                break

    return pairs, list(set(dirs1) - set(success))


def pair_dock_directories(dirs):
    """Specific function for the DEGEN merging in the design recapitulation experiments.
    Can this be used in the future?
    Assumes that dir.path is equal to /Nanohedra_input/NanohedraEntry*DockedPoses/1abc_2xyz
    """
    merge_pairs = []
    flipped_sufffix = '_flipped_180y'
    destination_string = 'DEGEN_1_2'
    origin_string = 'DEGEN_1_1'
    for _dir in dirs:
        # remove the flipped suffix from the dirname
        new_dir = os.path.dirname(_dir.path)[:-len(flipped_sufffix)]
        # where the link will live
        destination = os.path.join(os.path.abspath(new_dir), os.path.basename(_dir.path), destination_string)
        # where the link will attach too. Adding the flipped suffix to the building_blocks name
        original_dir = os.path.join(os.path.abspath(_dir.path) + flipped_sufffix, origin_string)
        merge_pairs.append((original_dir, destination))

    return merge_pairs, list()


def merge_docking_pair(pair, force=False):
    """Combine docking files of one docking combination with another

    Args:
        pair (tuple): source directory, destination directory (link)
    Keyword Args:
        force=False (bool): Whether to remove links before creation
    """
    if force:
        os.remove(pair[1])
    os.symlink(pair[0], pair[1], target_is_directory=True)
    # else:
    #     exit('Functionality does not yet exist')


def merge_design_pair(pair):
    """Combine Rosetta design files of one pose with the files of a second pose

    Args:
        pair (tuple): source directory, destination directory
    """
    def merge_scores():
        with open(os.path.join(pair[1].scores, PUtils.scores_file), 'a') as f1:
            f1.write('\n')  # first ensure a new line at the end of first file
            with open(os.path.join(pair[0].scores, PUtils.scores_file), 'r') as f2:
                lines = [loads(line) for line in f2.readlines()]
            f1.write('\n'.join(dumps(line) for line in lines))
            f1.write('\n')  # first a new line at the end of the combined file

    def merge_designs():
        for pdb in os.listdir(pair[0].designs):
            shutil.copy(os.path.join(pair[0].designs, pdb), os.path.join(pair[1].designs, pdb))
    merge_scores()
    merge_designs()


def rsync_dir(des_dir):
    """Takes a DEGEN_1_1 specific formatted list of directories and finds the DEGEN_1_2 directories to condense down to
    a single DEGEEN_1_2 directory making hard links for every file in a DEGEN_1_2 and higher directory to DEGEN_1_1"""

    for s, sym_dir in enumerate(des_dir.program_root):
        if '_flipped_180y' in sym_dir:
            for bb_dir in des_dir.building_blocks[s]:
                building_block_dir = os.path.join(des_dir.get_building_block_dir(bb_dir))
                destination = os.path.join(building_block_dir, 'DEGEN_1_1%s' % os.sep)

                p = {}  # make a dict for all the processes
                for k, entry in enumerate(os.scandir(building_block_dir)):
                    if entry.is_dir() and 'DEGEN_1_' in entry.name and entry.name != "DEGEN_1_1":
                        abs_entry_path = os.path.join(building_block_dir, entry.name)
                        cmd = ['rsync', '-a', '--link-dest=%s%s' % (abs_entry_path, os.sep), #  '--remove-source-files',
                               '%s%s' % (abs_entry_path, os.sep), destination]
                        #          ^ requires '/' - os.sep
                        logger.debug('Performing transfer: %s' % subprocess.list2cmdline(cmd))
                        p[abs_entry_path] = subprocess.Popen(cmd)
                # Check to see if all processes are done, then move on.
                for entry in p:
                    p[entry].communicate()

                # # Remove all empty subdirectories from the now empty DEGEN_1_2 and higher directory
                # p2 = {}
                # for l, entry in enumerate(p):
                #     find_cmd = ['find', '%s%s' % (entry, os.sep), '-type', 'd', '-empty', '-delete']
                #     # rm_cmd = ['rm', '-r', entry]
                #     logger.debug('Removing empty directories: %s' % subprocess.list2cmdline(find_cmd))
                #     p2[l] = subprocess.Popen(find_cmd)
                # # Check for the last command, then move on
                # for m, process in enumerate(p2):
                #     p2[m].communicate()

                print('%s has been consolidated' % building_block_dir)

    return des_dir.path


def status(all_design_directories, _stage, number=None, active=True, inactive_time=30 * 60):  # 30 minutes
    complete, running, incomplete = [], [], []
    if _stage == 'rmsd':
        # Relies on the assumption that each docked dir has only one set of building blocks making up its constituent files
        start = datetime.datetime(2020, 11, 3, 0, 0)  # when the most recent set of results was started
        all_to_all_start = datetime.datetime(2020, 11, 12, 0, 0)

        def _rmsd_dir(des_dir_symmetry):
            return os.path.join(des_dir_symmetry, 'rmsd_calculation')
        outcome_strings_d = {0: 'rmsd_to_cluster.sh', 1: 'all_to_cluster.sh', 2: 'rmsd_clustering.sh'}
        rmsd, all_to_all, clustering_files = [], [], []
        for des_dir in all_design_directories:
            rmsd_file, all_to_all_file, final_clustering_file = None, None, None
            rmsd_dir = _rmsd_dir(des_dir.program_root[0])
            try:
                rmsd_file = glob(os.path.join(rmsd_dir, 'crystal_vs_docked_irmsd.txt'))[0]
                # ensure that RMSD files were created for the most recent set of results using 'start'
                if int(time.time()) - int(os.path.getmtime(rmsd_file)) > start.now().timestamp() - start.timestamp():
                    rmsd_file = None
                all_to_all_file = glob(os.path.join(rmsd_dir, 'top*_all_to_all_docked_poses_irmsd.txt'))[0]
                final_clustering_file = glob(os.path.join(rmsd_dir, '*_clustered.txt'))[0]
                if int(time.time()) - int(os.path.getmtime(final_clustering_file)) > \
                        all_to_all_start.now().timestamp() - all_to_all_start.timestamp():
                    final_clustering_file = None
            except IndexError:
                incomplete.append(rmsd_dir)
            rmsd.append(rmsd_file)
            all_to_all.append(all_to_all_file)
            clustering_files.append(final_clustering_file)

        # report all designs which are done, return all commands for designs which need to be completed.
        for k, results in enumerate(zip(rmsd, all_to_all, clustering_files)):
            _status = True
            # _status = results[2]
            for r, stage in enumerate(results):
                if not stage:
                    _status = False
                    # _status = os.path.join(_rmsd_dir(design_directories[k], outcome_strings_d[r]))
                    running.append(os.path.join(_rmsd_dir(design_directories[k].symmetry[0]), outcome_strings_d[r]))
                    break
            if _status:
                complete.append(glob(os.path.join(_rmsd_dir(all_design_directories[k].program_root[0]), '*_clustered.txt'))[0])

    elif _stage == 'nanohedra':
        from classes import get_last_sampling_state
        # observed_building_blocks = []
        for des_dir in all_design_directories:
            # if os.path.basename(des_dir.building_blocks) in observed_building_blocks:
            #     continue
            # else:
            #     observed_building_blocks.append(os.path.basename(des_dir.building_blocks))
            # f_degen1, f_degen2, f_rot1, f_rot2 = get_last_sampling_state('%s_log.txt' % des_dir.building_blocks)
            # degens, rotations = \
            # SDUtils.degen_and_rotation_parameters(SDUtils.gather_docking_metrics(des_dir.program_root))
            #
            # dock_dir = DesignDirectory(path, auto_structure=False)
            # dock_dir.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses'))
            # dock_dir.building_blocks = [next(os.walk(dir))[1] for dir in dock_dir.program_root]
            # dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.program_root]
            # dock_dir.building_block_logs = [os.path.join(_sym, bb_dir, 'bb_dir_log.txt') for sym in dock_dir.building_blocks
            #                                 for bb_dir in sym] # TODO change to PUtils

            # docking_file = glob(
            #     os.path.join(des_dir + '_flipped_180y', '%s*_log.txt' % os.path.basename(des_dir)))
            # if len(docking_file) != 1:
            #     incomplete.append(des_dir)
            #     continue
            # else:
            #     log_file = docking_file[0]
            for sym_idx, building_blocks in enumerate(des_dir.building_block_logs):  # Added from dock_dir patch
                for bb_idx, log_file in enumerate(building_blocks):  # Added from dock_dir patch
                    f_degen1, f_degen2, f_rot1, f_rot2 = get_last_sampling_state(log_file, zero=False)
                    # degens, rotations = Pose.degen_and_rotation_parameters(
                    #     Pose.gather_docking_metrics(des_dir.log[sym_idx]))
                    des_dir.gather_docking_metrics()  # log[sym_idx]))
                    degens, rotations = des_dir.degen_and_rotation_parameters()
                    degen1, degen2 = tuple(degens)
                    last_rot1, last_rot2 = des_dir.compute_last_rotation_state()
                    # last_rot1, last_rot2 = Pose.compute_last_rotation_state(*rotations)
                    # REMOVE after increment gone
                    f_degen1, f_degen2 = 1, 1
                    if f_rot2 > last_rot2:
                        f_rot2 = int(f_rot2 % last_rot2)
                        if f_rot2 == 0:
                            f_rot2 = last_rot2
                    # REMOVE
                    print('Last State:', f_degen1, f_degen2, f_rot1, f_rot2)
                    print('Expected:', degen1, degen2, last_rot1, last_rot2)
                    print('From log: %s' % log_file)
                    if f_degen1 == degen1 and f_degen2 == degen2 and f_rot1 == last_rot1 and f_rot2 == last_rot2:
                        complete.append(os.path.join(des_dir.program_root[sym_idx], des_dir.building_blocks[sym_idx][bb_idx]))
                        # complete.append(des_dir)
                        # complete.append(des_dir.program_root)
                    else:
                        if active:
                            if int(time.time()) - int(os.path.getmtime(log_file)) < inactive_time:
                                running.append(os.path.join(des_dir.program_root[sym_idx],
                                                            des_dir.building_blocks[sym_idx][bb_idx]))
                        incomplete.append(os.path.join(des_dir.program_root[sym_idx],
                                                       des_dir.building_blocks[sym_idx][bb_idx]))
                        # incomplete.append(des_dir)
                        # incomplete.append(des_dir.program_root)
        complete = map(os.path.dirname, complete)  # can remove if building_block name is removed
        running = list(map(os.path.dirname, running))  # can remove if building_block name is removed
        incomplete = map(os.path.dirname, incomplete)  # can remove if building_block name is removed
    else:
        if not number:
            number = PUtils.stage_f[_stage]['len']
            if not number:
                return False

        for des_dir in design_directories:
            files = des_dir.get_designs(design_type=PUtils.stage_f[_stage]['path'])
            if number >= len(files):
                complete.append(des_dir.path)
            else:
                incomplete.append(des_dir.path)

    if active:
        active_path = os.path.join(args.directory, 'running_%s_pose_status' % _stage)
        with open(active_path, 'w') as f_act:
            f_act.write('\n'.join(r for r in running))
        incomplete = list(set(incomplete) - set(running))

    complete_path = os.path.join(args.directory, 'complete_%s_pose_status' % _stage)
    with open(complete_path, 'w') as f_com:
        f_com.write('\n'.join(c for c in complete))

    incomplete_path = os.path.join(args.directory, 'incomplete_%s_pose_status' % _stage)
    with open(incomplete_path, 'w') as f_in:
        f_in.write('\n'.join(n for n in incomplete))

    return True


def fix_files_mp(des_dir):
    with open(os.path.join(des_dir.scores, PUtils.scores_file), 'r+') as f1:
        lines = f1.readlines()

        # Remove extra newlines from file
        # clean_lines = []
        # for line in lines:
        #     if line == '\n':
        #         continue
        #     clean_lines.append(line.strip())

        # Take multi-entry '{}{}' json record and make multi-line
        # new_line = {}
        # for z, line in enumerate(lines):
        #     if len(line.split('}{')) == 2:
        #         sep = line.find('}{') + 1
        #         new_line[z] = line[sep:]
        #         lines[z] = line[:sep]
        # for error_idx in new_line:
        #     lines.insert(error_idx, new_line[error_idx])

        # f1.seek(0)
        # f1.write('\n'.join(clean_lines))
        # f1.write('\n')
        # f1.truncate()

        if lines[-1].startswith('{"decoy":"clean_asu_for_consenus"'):
            j = True
        else:
            j = False

    return j, None


def orient_pdb_file(pdb_path, log_path, sym=None, out_dir=None):
    pdb_filename = os.path.basename(pdb_path)
    oriented_file_path = os.path.join(out_dir, pdb_filename)
    if not os.path.exists(oriented_file_path):
        pdb = PDB.from_file(pdb_path)
        with open(log_path, 'a+') as f:
            try:
                pdb.orient(sym=sym, out_dir=out_dir, generate_oriented_pdb=True)
                f.write("oriented: %s\n" % pdb_filename)
                return oriented_file_path
            except ValueError as val_err:
                f.write(str(val_err))
            except RuntimeError as rt_err:
                f.write(str(rt_err))
            return None
    else:
        return oriented_file_path


def terminate(all_exceptions):
    any_exceptions = [exception for exception in all_exceptions if exception]
    # any_exceptions = list(filter(bool, exceptions))
    # any_exceptions = list(set(exceptions))
    # if len(any_exceptions) > 1 or any_exceptions and any_exceptions[0]:
    if len(any_exceptions) > 1:
        logger.warning('\nThe following exceptions were thrown. Design for %d directories is inaccurate!!!\n' %
                       len(any_exceptions))
        all_exception_poses = []
        for exception in any_exceptions:
            # if exception:
            des_dir, exception_msg = exception
            try:
                logger.warning('%s: %s' % (des_dir.path, exception_msg))
            except AttributeError:
                logger.warning('%s: %s' % (des_dir, exception_msg))

            all_exception_poses.append(des_dir)
        print('\n')
        # except_file = os.path.join(args.location, 'EXCEPTIONS.log')
        except_file = SDUtils.write_list_to_file(all_exception_poses, name='EXCEPTIONS.log',
                                                 location=args.directory)
        logger.info('All poses with exceptions written to file: %s' % except_file)

    exit('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='\nControl all input/output of %s including:\n1. Pose initialization\n'
                                                 '2. Command distribution to computational nodes\n'
                                                 '3. Analysis of designs' % PUtils.program_name)
    # ---------------------------------------------------
    parser.add_argument('-c', '--command_only', action='store_true',
                        help='Should commands be written but not executed?\nDefault=False')
    parser.add_argument('-d', '--directory', type=os.path.abspath,
                        help='Master directory where %s design poses are located. This may be the output directory '
                             'from %s.py, or a directory with poses requiring interface design'
                             % (PUtils.program_name, PUtils.nano.title()))
    # parser.add_argument('-@', '--design_flags', type=str)
    parser.add_argument('-f', '--file', type=os.path.abspath,
                        help='File with location(s) of %s design poses' % PUtils.program_name, default=None)
    parser.add_argument('-m', '--mode', type=str, choices=['design', 'dock'], help='Which mode to operate in?') \
        # , required=True)
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    parser.add_argument('-s', '--design_string', type=str, help='If pose names are specified by design string instead '
                                                                'of directories, which directory path to '
                                                                'prefix with?\nDefault=None', default=None)
    subparsers = parser.add_subparsers(title='SubModules', dest='sub_module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter: \n%s\n\nAny'
                                            'SubModule help can be accessed in this way'  # Todo
                                            % PUtils.submodule_help)
    # ---------------------------------------------------
    parser_guide = subparsers.add_parser('guide', help='Access the %s guide! Start here if your a first time user'
                                                       % PUtils.program_name)
    # ---------------------------------------------------
    parser_query = subparsers.add_parser('query', help='Query %s.py docking entries' % PUtils.nano.title())
    # ---------------------------------------------------
    parser_flag = subparsers.add_parser('flags', help='Generate a flags file for %s' % PUtils.program_name)
    parser_flag.add_argument('-t', '--template', action='store_true',
                             help='Generate a flags template to edit on your own.')
    # ---------------------------------------------------
    parser_mask = subparsers.add_parser('mask', help='Generate a residue mask for %s' % PUtils.program_name)
    # ---------------------------------------------------
    parser_selection = subparsers.add_parser('design_selection',
                                             help='Generate a residue selection for %s' % PUtils.program_name)
    # ---------------------------------------------------
    parser_filter = subparsers.add_parser('filter', help='Filter designs based on design specific metrics.')
    parser_filter.add_argument('-m', '--metric', type=str, help='What metric would you like to filter Designs by?',
                               choices=['score', 'fragments_matched'], required=True)
    # ---------------------------------------------------
    parser_dock = subparsers.add_parser('dock', help='Submit jobs to %s.py\nIf a docking directory structure is set up,'
                                                     ' provide the overall directory location with program argument '
                                                     '-d/-f, otherwise, use the -d1 -d2 \'pose\' module arguments to '
                                                     'specify lists of oligomers to dock' % PUtils.nano.title())
    parser_dock.add_argument('-d1', '--pdb_path1', type=os.path.abspath, required=True,
                             help='Disk location where the first oligomers are located\nREQUIRED', default=None)
    parser_dock.add_argument('-d2', '--pdb_path2', type=os.path.abspath,
                             help='Disk location where the second oligomers are located\nDefault=None', default=None)
    parser_dock.add_argument('-e', '--entry', type=int, help='The entry number of %s.py docking combinations to use' %
                                                             PUtils.nano.title(), default=None)
    # parser_dock.add_argument('-f', '--additional_flags', nargs='*', default=None,
    #                          help='Any additional flags to pass to %s.py. Should be passed WITHOUT leading \'-\'!'
    #                               % PUtils.nano.title())
    parser_dock.add_argument('-o', '--outdir', type=str, default=None,
                             help='Where should the output from commands be written?\n'
                                  'Default=CWD/NanohedraEntry(entry)DockedPoses')
    # ---------------------------------------------------
    parser_fragments = subparsers.add_parser('generate_fragments',
                                             help='Generate fragment overlap for poses of interest.')
    # ---------------------------------------------------
    parser_design = subparsers.add_parser('design', help='Gather poses of interest and format for design using sequence'
                                                         'constraints in Rosetta. Constrain using evolutionary profiles'
                                                         ' of homologous sequences and/or fragment profiles extracted '
                                                         'from the PDB or neither.')
    parser_design.add_argument('-i', '--fragment_database', type=str,
                               help='Database to match fragments for interface specific scoring matrices. One of %s'
                                    '\nDefault=%s' % (','.join(list(PUtils.frag_directory.keys())),
                                                      list(PUtils.frag_directory.keys())[0]),
                               default=list(PUtils.frag_directory.keys())[0])
    # parser_design.add_argument('-x', '--suspend', action='store_true',
    #                            help='Should Rosetta design trajectory be suspended?\nDefault=False')
    parser_design.add_argument('-p', '--mpi', action='store_true',
                               help='Should job be set up for cluster submission?\nDefault=False')
    # ---------------------------------------------------
    parser_analysis = subparsers.add_parser('analysis', help='Run analysis on all poses specified and their designs.')
    parser_analysis.add_argument('-o', '--output', type=str, default=PUtils.analysis_file,
                                 help='Name to output .csv files.\nDefault=%s' % PUtils.analysis_file)
    parser_analysis.add_argument('-n', '--no_save', action='store_true',
                                 help='Don\'t save trajectory information.\nDefault=False')
    parser_analysis.add_argument('-f', '--figures', action='store_true',
                                 help='Create and save figures for all poses?\nDefault=False')
    parser_analysis.add_argument('-j', '--join', action='store_true',
                                 help='Join Trajectory and Residue Dataframes?\nDefault=False')
    parser_analysis.add_argument('-g', '--delta_g', action='store_true',
                                 help='Compute deltaG versus Refine structure?\nDefault=False')
    # ---------------------------------------------------
    parser_sequence = subparsers.add_parser('sequence', help='Generate protein sequences for selected designs. '
                                                             'Either -d or -p is required. If both are provided, '
                                                             '-p will be prioritized')
    sequence_required = parser_sequence.add_mutually_exclusive_group(required=True)
    sequence_required.add_argument('-d', '--dataframe', type=os.path.abspath,  # required=True,
                                   help='Dataframe.csv from analysis containing pose info.')
    sequence_required.add_argument('-p', '--pose_design_file', type=str,
                                   help='Name of pose, design .csv file to serve as sequence selector')
    parser_sequence.add_argument('-c', '--consensus', action='store_true', help='Whether to grab the consensus sequence'
                                                                                '\nDefault=False')
    parser_sequence.add_argument('-f', '--filters', action='store_true', help='Metrics with which to filter on\n'
                                                                              'Default=None', default=None)
    parser_sequence.add_argument('-n', '--number', type=int, help='Number of top sequences to return per design',
                                 default=1)
    parser_sequence.add_argument('-s', '--selection_string', type=str, help='Output identifier for sequence selection')
    parser_sequence.add_argument('-w', '--weights', type=str, help='Weights of various metrics to final poses\n'
                                                                   'Default=1/number of --filters')
    # ---------------------------------------------------
    parser_dist = subparsers.add_parser('distribute',
                                        help='Distribute specific design step commands to computational resources. '
                                             'In distribution mode, the --file or --directory argument specifies which '
                                             'pose commands should be distributed.')
    parser_dist.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                             help='The stage of design to be prepared. One of %s' %
                                  ', '.join(list(v for v in PUtils.stage_f.keys())), required=True)
    parser_dist.add_argument('-y', '--success_file', help='The name/location of file containing successful commands\n'
                                                          'Default={--stage}_stage_pose_successes', default=None)
    parser_dist.add_argument('-n', '--failure_file', help='The name/location of file containing failed commands\n'
                                                          'Default={--stage}_stage_pose_failures', default=None)
    parser_dist.add_argument('-m', '--max_jobs', type=int, help='How many jobs to run at once?\nDefault=80',
                             default=80)
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Get design status for selected designs')
    parser_status.add_argument('-n', '--number_designs', type=int, help='Number of trajectories per design',
                               default=None)
    parser_status.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                               help='The stage of design to check status of. One of %s'
                                    % ', '.join(list(v for v in PUtils.stage_f.keys())), default=None)
    # ---------------------------------------------------
    parser_merge = subparsers.add_parser('merge', help='Merge all completed designs from location 2 (-f2/-d2) to '
                                                       'location 1(-f/-d). Includes renaming. Highly suggested you copy'
                                                       ' original data!!!')
    parser_merge.add_argument('-d2', '--directory2', type=os.path.abspath, default=None,
                              help='Directory 2 where poses should be copied from and appended to location 1 poses')
    parser_merge.add_argument('-f2', '--file2', type=str, help='File 2 where poses should be copied from and appended '
                                                               'to location 1 poses', default=None)
    parser_merge.add_argument('-F', '--force', action='store_true', help='Overwrite merge paths?\nDefault=False')
    parser_merge.add_argument('-i', '--increment', type=int, help='How many to increment each design by?\nDefault=%d'
                                                                  % PUtils.nstruct)
    # ---------------------------------------------------
    parser_modify = subparsers.add_parser('modify', help='Modify something for testing')
    parser_modify.add_argument('-m', '--mod', type=str, help='Which type of modification?\nChoose from '
                                                             'consolidate_degen or pose_map')
    # ---------------------------------------------------
    parser_rename_scores = subparsers.add_parser('rename_scores', help='Rename Protocol names according to dictionary')

    args, additional_flags = parser.parse_known_args()
    # -----------------------------------------------------------------------------------------------------------------
    # Start Logging
    # -----------------------------------------------------------------------------------------------------------------
    if args.debug:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name=os.path.basename(__file__), level=2)

    if args.sub_module not in ['query', 'guide', 'flags', 'design_selection']:
        logger.info('Starting %s with options:\n\t%s' %
                    (os.path.basename(__file__),
                     '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))
    # -----------------------------------------------------------------------------------------------------------------
    # Process additional flags
    # -----------------------------------------------------------------------------------------------------------------
    if additional_flags:  # args.additional_flags
        logger.debug('Additional: %s' % additional_flags)

        def format_flags(flags):
            formatted_flags = []
            for flag in flags:
                if flag[0] == '-' and flag[1] != '-':
                    formatted_flags.append(flag)
                elif flag[0] == '-' and flag[1] == '-':
                    formatted_flags.append(flag[1:])
                elif flag[0] != '-':
                    formatted_flags.append('-%s' % flag)

            return formatted_flags
        extra_flags = format_flags(additional_flags)
        # extra_flags.extend(format_flags(args.additional_flags))

        # parses ['-nanohedra_output True', ...]
        design_flags = {}
        for flag_arg in extra_flags:
            if ' ' in flag_arg:
                flag = flag_arg.split()[0].lstrip('-')
                design_flags[flag] = flag_arg.split()[1]
                if design_flags[flag].title() in ['None', 'True', 'False']:
                    design_flags[flag] = eval(design_flags[flag].title())
            else:
                design_flags[flag_arg[1:0]] = None

    else:
        design_flags = return_default_flags(args.sub_module)
        extra_flags = None  # Todo remove
    design_flags.update(args.__dict__)
    logger.debug('All flags: %s' % design_flags)
    # -----------------------------------------------------------------------------------------------------------------
    all_docked_poses, all_dock_directories, pdb_pairs, design_directories, location = None, None, None, None, None
    # Grab all Designs (DesignDirectory) to be processed from either directory name or file
    initial_iter = None
    # -----------------------------------------------------------------------------------------------------------------
    if args.sub_module in ['distribute', 'query', 'guide', 'flags', 'design_selection']:
        pass
    # Todo depreciate args.mode here
    elif args.sub_module in ['design', 'filter', 'generate_fragments'] or args.mode == 'design':
        mode = 'design'
        if args.directory or args.file:
            # Pull nanohedra_output and mask_design_using_sequence out of flags
            # design_flags = load_flags(args.design_flags)
            # Todo move to a verify flags function
            pdb_mask, entity_mask, chain_mask, residue_mask, atom_mask = None, None, None, set(), None
            if 'select_designable_residues_by_sequence' in design_flags \
                    and design_flags['select_designable_residues_by_sequence']:
                residue_mask = residue_mask.union(
                    SequenceProfile.generate_sequence_mask(design_flags['select_designable_residues_by_sequence']))
            if 'select_designable_residues_by_pose_number' in design_flags \
                    and design_flags['select_designable_residues_by_pose_number']:
                residue_mask = residue_mask.union(
                    SequenceProfile.generate_residue_mask(design_flags['select_designable_residues_by_pose_number']))

            if 'select_designable_chains' in design_flags and design_flags['select_designable_chains']:
                chain_mask = SequenceProfile.generate_chain_mask(design_flags['select_designable_chains'])

            design_flags.update({'design_selection': {'pdbs': pdb_mask, 'entities': entity_mask, 'chains': chain_mask,
                                                      'residues': residue_mask, 'atoms': atom_mask}})
            logger.debug('Design flags after masking: %s' % design_flags)

            # Add additional program flags to design_flags
            # if args.mpi:  # Todo
            #     # extras = ' mpi %d' % CUtils.mpi
            #     logger.info(
            #         'Setting job up for submission to MPI capable computer. Pose trajectories will run in parallel,'
            #         ' %s at a time. This will speed up pose processing ~%f-fold.' %
            #         (CUtils.mpi - 1, PUtils.nstruct / (CUtils.mpi - 1)))
            #     design_flags.update({'mpi': True, 'script': True})

            # Set up DesignDirectories  # Todo args.design_String?!?
            inputs_moved = False
            if 'nanohedra_output' in design_flags and design_flags['nanohedra_output']:
                all_poses, location = SDUtils.collect_directories(args.directory, file=args.file, dir_type=PUtils.nano)
                design_directories = [DesignDirectory.from_nanohedra(pose, mode=mode, project=args.design_string,
                                                                     **design_flags)
                                      for pose in all_poses]
            else:
                # We have to ensure that if the user has provided it, the symmetry is correct
                if 'symmetry' in design_flags and design_flags['symmetry']:
                    if design_flags['symmetry'] in SDUtils.possible_symmetries:
                        sym_entry = SDUtils.parse_symmetry_to_nanohedra_entry(design_flags['symmetry'])
                        design_flags['sym_entry_number'] = sym_entry
                    elif design_flags['symmetry'].lower()[:5] == 'cryst':
                        do_something = True
                        # the symmetry information should be in the pdb headers
                    else:
                        raise SDUtils.DesignError('The symmetry %s is not supported! Supported symmetries include:\n%s'
                                                  '\nCorrect your flags file and try again'
                                                  % (design_flags['symmetry'],
                                                     ', '.join(SDUtils.possible_symmetries.keys())))

                all_poses, location = SDUtils.collect_directories(args.directory, file=args.file)
                design_directories = [DesignDirectory.from_file(pose, project=args.design_string, **design_flags)
                                      for pose in all_poses]
                all_poses = [design.asu for design in design_directories]
                inputs_moved = True
            if not design_directories:
                raise SDUtils.DesignError('No SymDesign directories found within \'%s\'! Please ensure correct '
                                          'location. Are you sure you want to run with -%s %s'
                                          % (location, 'nanohedra_output', design_flags['nanohedra_output']))
            if not args.debug:
                logger.info('All design specific logs are located in their corresponding directories.\n\tEx: %s'
                            % design_directories[0].log.handlers[0].baseFilename)

            if 'generate_fragments' in design_flags and design_flags['generate_fragments']:
                interface_type = 'biological_interfaces'  # Todo parameterize
                logger.info('Initializing FragmentDatabase from %s\n' % interface_type)
                fragment_db = SequenceProfile.FragmentDatabase(source='directory', location=interface_type)
                for design in design_directories:
                    design.connect_db(frag_db=fragment_db)

            if not args.file or inputs_moved:
                # Make single file with names of each directory where all_docked_poses can be found
                project_string = design_directories[0].project_designs
                args.file = os.path.join(os.getcwd(), '%s_pose.paths' % project_string)
                with open(args.file, 'w') as design_f:
                    design_f.write('\n'.join(pose for pose in all_poses))
                logger.critical('The file \'%s\' contains all the designs in your current project. Utilize this file to'
                                ' interact with %s designs in future commands for this project such as \'%s --file %s '
                                'analysis\''
                                % (args.file, PUtils.program_name, PUtils.program_command, args.file))
        else:
            raise SDUtils.DesignError('No design directories/files were specified!\n'
                                      'Please specify --directory or --file and run your command again')
        logger.info('%d unique poses found in \'%s\'' % (len(design_directories), location))

    elif args.sub_module == 'dock' or args.mode == 'dock':  # Todo depreciate args.mode here
        args.mode = 'dock'
        # Getting PDB1 and PDB2 File paths
        if args.pdb_path1:
            if not args.entry:
                exit('If using --pdb_path1 (-d1) and/or --pdb_path2 (-d2), please specify --entry as well. '
                     '--entry can be found using SubModule \'%s query\'' % PUtils.program_command)

            else:
                sym_entry = SymEntry(args.entry)
                oligomer_symmetry_1 = sym_entry.get_group1_sym()
                oligomer_symmetry_2 = sym_entry.get_group2_sym()

            # Orient Input Oligomers to Canonical Orientation
            logger.info('Orienting PDB\'s for Nanohedra Docking')
            oriented_pdb1_out_dir = os.path.join(os.path.dirname(args.pdb_path1), '%s_oriented_with_%s_symmetry'
                                                 % (os.path.basename(args.pdb_path1), oligomer_symmetry_1))
            if not os.path.exists(oriented_pdb1_out_dir):
                os.makedirs(oriented_pdb1_out_dir)

            if '.pdb' in args.pdb_path1:
                pdb1_filepaths = [args.pdb_path1]
            else:
                pdb1_filepaths = SDUtils.get_all_pdb_file_paths(args.pdb_path1)
            pdb1_oriented_filepaths = [orient_pdb_file(pdb_path,
                                                       os.path.join(oriented_pdb1_out_dir, PUtils.orient_log_file),
                                                       sym=oligomer_symmetry_1, out_dir=oriented_pdb1_out_dir)
                                       for pdb_path in pdb1_filepaths]
            logger.info('%d filepaths found' % len(pdb1_oriented_filepaths))
            # pdb1_oriented_filepaths = filter(None, pdb1_oriented_filepaths)

            if args.pdb_path2:
                if args.pdb_path1 != args.pdb_path2:
                    oriented_pdb2_out_dir = os.path.join(os.path.dirname(args.pdb_path2), '%s_oriented_with_%s_symmetry'
                                                         % (os.path.basename(args.pdb_path2), oligomer_symmetry_2))
                    if not os.path.exists(oriented_pdb2_out_dir):
                        os.makedirs(oriented_pdb2_out_dir)

                    if '.pdb' in args.pdb_path2:
                        pdb2_filepaths = [args.pdb_path2]
                    else:
                        pdb2_filepaths = SDUtils.get_all_pdb_file_paths(args.pdb_path2)
                    pdb2_oriented_filepaths = [orient_pdb_file(pdb_path,
                                                               os.path.join(oriented_pdb1_out_dir,
                                                                            PUtils.orient_log_file),
                                                               sym=oligomer_symmetry_2, out_dir=oriented_pdb2_out_dir)
                                               for pdb_path in pdb2_filepaths]

                    pdb_pairs = list(product(filter(None, pdb1_oriented_filepaths), filter(None, pdb2_oriented_filepaths)))
                    # pdb_pairs = list(product(pdb1_oriented_filepaths, pdb2_oriented_filepaths))
                    # pdb_pairs = list(product(SDUtils.get_all_pdb_file_paths(oriented_pdb1_out_dir),
                    #                          SDUtils.get_all_pdb_file_paths(oriented_pdb2_out_dir)))
                    location = '%s & %s' % (args.pdb_path1, args.pdb_path2)
            else:  # Todo add initial to the first some how
                pdb_pairs = list(combinations(filter(None, pdb1_oriented_filepaths), 2))
                # pdb_pairs = list(combinations(pdb1_oriented_filepaths, 2))
                # pdb_pairs = list(combinations(SDUtils.get_all_pdb_file_paths(oriented_pdb1_out_dir), 2))
                location = args.pdb_path1
            initial_iter = [False for i in range(len(pdb_pairs))]
            initial_iter[0] = True
            design_directories = pdb_pairs  # for logging purposes below Todo combine this with pdb_pairs variable
        elif args.directory or args.file:
            all_dock_directories, location = SDUtils.collect_directories(args.directory, file=args.file,
                                                                         dir_type=args.mode)
            design_directories = [set_up_directory_objects(dock_dir, mode=args.mode, project=args.design_string)
                                  for dock_dir in all_dock_directories]
            if len(design_directories) == 0:
                raise SDUtils.DesignError('No docking directories/files were found!\n'
                                          'Please specify --directory1, and/or --directory2 or --directory or --file. '
                                          'See %s' % PUtils.help(args.sub_module))

        logger.info('%d unique building block docking combinations found in \'%s\'' % (len(design_directories),
                                                                                       location))
    else:
        raise SDUtils.DesignError('Error: --mode flag must be passed since the module is %s!' % args.sub_module)

    args.suspend = False
    if args.command_only:
        args.suspend = True
        logger.info('Writing modelling commands out to file only, no modelling will occur until commands are executed')

    # -----------------------------------------------------------------------------------------------------------------

    # Parse SubModule specific commands
    results, exceptions = [], []
    # -----------------------------------------------------------------------------------------------------------------
    if args.sub_module == 'guide':
        with open(PUtils.readme, 'r') as f:
            print(f.read(), end='')
    # ---------------------------------------------------
    elif args.sub_module == 'query':
        query_flags = [__file__, '-query'] + additional_flags
        logger.debug('Query %s.py with: %s' % (PUtils.nano.title(), ', '.join(query_flags)))
        query_mode(query_flags)
    # ---------------------------------------------------
    elif args.sub_module == 'flags':
        if args.template:
            query_user_for_flags(template=True)
        elif args.mode:
            query_user_for_flags(mode=args.mode)
        else:
            query_user_for_flags()
    # ---------------------------------------------------
    elif args.sub_module == 'design_selection':
        fasta_file = generate_sequence_template(args.additional_flags[0])
        logger.info('The design_selection template was written to %s. Please edit this file so that the design_selection can be generated for '
                    'protein design. Mask should be formatted so a \'-\' replaces all sequence of interest to be '
                    'overlooked during design. Example:\n>pdb_template_sequence\nMAGHALKMLV...\n>design_selection\nMAGH----LV\n'
                    % fasta_file)
    # ---------------------------------------------------
    elif args.sub_module == 'filter':
        if args.metric == 'score':
            designpath_metric_tup_list = [(des_dir.asu, des_dir.score) for des_dir in design_directories]
        elif args.metric == 'fragments_matched':
            designpath_metric_tup_list = [(des_dir.asu, des_dir.number_of_fragments) for des_dir in design_directories]
        else:
            raise SDUtils.DesignError('The filter metric \'%s\' is not supported!' % args.metric)

        logger.info('Sorting designs according to \'%s\'' % args.metric)
        designpath_metric_tup_list_sorted = sorted(designpath_metric_tup_list, key=lambda tup: (tup[1] or 0),
                                                   reverse=True)
    # ---------------------------------------------------
    elif args.sub_module == 'dock':  # -d1 pdb_path1, -d2 pdb_path2, -e entry, -o outdir, -f additional_flags
        # Initialize docking procedure
        if args.multi_processing:
            # Calculate the number of threads to use depending on computer resources
            threads = SDUtils.calculate_mp_threads(maximum=True, no_model=args.suspend)
            logger.info('Starting multiprocessing using %s threads' % str(threads))
            if args.command_only:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    zipped_args = zip(repeat(args.entry), *zip(*pdb_pairs), repeat(args.outdir), repeat(extra_flags),
                                      repeat(args.design_string), initial_iter)
                    results, exceptions = zip(*SDUtils.mp_starmap(nanohedra_command_mp, zipped_args, threads))
                else:  # args.directory or args.file set up docking directories
                    zipped_args = zip(design_directories, repeat(args.design_string))
                    results, exceptions = zip(*SDUtils.mp_starmap(nanohedra_recap_mp, zipped_args, threads))
                results = list(results)
            else:  # TODO implementation where SymDesignControl calls Nanohedra.py
                logger.info('Can\'t run %s.py docking from here yet. Must pass python %s -c for execution'
                            % (PUtils.nano, __file__))
                exit()
        else:
            logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
            if args.command_only:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    for initial, (path1, path2) in zip(initial_iter, pdb_pairs):
                        result, error = nanohedra_command_s(args.entry, path1, path2, args.outdir, extra_flags,
                                                            args.design_string, initial)
                        results.append(result)
                        exceptions.append(error)
                else:  # single directory docking (already made directories)
                    for dock_directory in design_directories:
                        result, error = nanohedra_recap_s(dock_directory, args.design_string)
                        results.append(result)
                        exceptions.append(error)
            else:  # TODO implementation where SymDesignControl calls Nanohedra.py
                logger.info('Can\'t run %s.py docking from here yet. Must pass python %s -c for execution'
                            % (PUtils.nano, __file__))
                exit()

        # Make single file with names of each directory. Specific for docking due to no established directory
        args.file = os.path.join(args.directory, 'all_docked_directories.paths')  # Todo Parameterized
        with open(args.file, 'w') as design_f:
            command_directories = map(os.path.dirname, results)  # get only the directory of the command
            design_f.write('\n'.join(docking_pair for docking_pair in command_directories if docking_pair))

        all_commands = [result for result in results if result]
        if len(all_commands) > 0:
            command_file = SDUtils.write_commands(all_commands, name=PUtils.nano, out_path=args.directory)
            logger.info('All \'%s\' commands were written to \'%s\'' % (PUtils.nano, command_file))

            args.success_file = None
            args.failure_file = None
            args.max_jobs = 80
            distribute(logger, stage=PUtils.nano, directory=args.directory, file=command_file,
                       success_file=args.success_file, failure_file=args.success_file, max_jobs=args.max_jobs)
            # logger.info('\nTo process commands, execute:\n%s' %  # cd %s\n%s' % (args.directory,
            #             'python %s -f %s distribute -s %s' % (__file__, command_file, PUtils.nano))  # args.file,
        else:
            logger.error('No \'%s\' commands were written!' % PUtils.nano)
    # ---------------------------------------------------
    elif args.sub_module == 'generate_fragments':  # -c command_only, -i fragment_library, -p mpi, -x suspend
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # Calculate the number of threads to use depending on computer resources
            threads = SDUtils.calculate_mp_threads(maximum=True, no_model=args.suspend)  # mpi=args.mpi, Todo
            logger.info('Starting multiprocessing using %s threads' % str(threads))
            results, exceptions = zip(*SDUtils.mp_map(DesignDirectory.generate_interface_fragments, design_directories,
                                                      threads))
            results = list(results)
        else:
            logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
            for design in design_directories:
                design.generate_interface_fragments()
    # ---------------------------------------------------
    elif args.sub_module == 'design':  # -c command_only, -i fragment_library, -p mpi, -x suspend
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # Calculate the number of threads to use depending on computer resources
            threads = SDUtils.calculate_mp_threads(maximum=True, no_model=args.suspend)  # mpi=args.mpi, Todo
            logger.info('Starting multiprocessing using %s threads' % str(threads))
            results, exceptions = zip(*SDUtils.mp_map(DesignDirectory.interface_design, design_directories, threads))
            results = list(results)
        else:
            logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
            for design in design_directories:
                result, error = design.interface_design()
                results.append(result)
                exceptions.append(error)

        # # Start pose processing and preparation for Rosetta
        # if args.multi_processing:
        #     # Calculate the number of threads to use depending on computer resources
        #     threads = SDUtils.calculate_mp_threads(mpi=args.mpi, maximum=True, no_model=args.suspend)
        #     logger.info('Starting multiprocessing using %s threads' % str(threads))
        #     zipped_args = zip(design_directories, repeat(args.fragment_database), repeat(args.symmetry_group),
        #                       repeat(args.command_only), repeat(args.mpi), repeat(args.suspend),
        #                       repeat(args.debug))
        #     results, exceptions = zip(*SDUtils.mp_starmap(initialization_mp, zipped_args, threads))
        #     results = list(results)
        # else:
        #     logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
        #     for des_directory in design_directories:
        #         result, error = initialization_s(des_directory, args.fragment_database, args.symmetry_group,
        #                                          script=args.command_only, mpi=args.mpi, suspend=args.suspend,
        #                                          debug=args.debug)
        #         results.append(result)
        #         exceptions.append(error)
        #
        if args.command_only:
            all_commands = []
            # all_commands = [[] for s in PUtils.stage_f]
            # command_files = [[] for s in PUtils.stage_f]
            for des_directory in design_directories:
                # for i, stage in enumerate(PUtils.stage_f):
                all_commands.append(os.path.join(des_directory.scripts, '%s.sh' % interface_design_command))
            # for i, stage in enumerate(PUtils.stage_f):
            #     if i > 1:  # 1/31/21 only have one option for commands now  # > 3:  # No consensus
            #         break
            command_file = SDUtils.write_commands(all_commands, name=interface_design_command, out_path=args.directory)
            logger.info('All \'%s\' commands were written to \'%s\'' % (interface_design_command, command_file))
            args.success_file = None
            args.failure_file = None
            args.max_jobs = 80
            distribute(logger, stage=PUtils.nano, directory=args.directory, file=command_file,
                       success_file=args.success_file, failure_file=args.success_file, max_jobs=args.max_jobs)
            # logger.info('\nTo process all commands in correct order, execute:\ncd %s\n%s' %
            #             (args.directory, '\n'.join(['python %s -f %s distribute -s %s' %
            #                                         (__file__, command_files[i], stage)
            #                                         for i, stage in enumerate(list(PUtils.stage_f.keys()))[:3]])))
            #                                                                                        args.file,
            # TODO make args.file equal to command_files[i] and change distribution to no stage prefix...
    # ---------------------------------------------------
    elif args.sub_module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
        distribute(args, logger)
    # ---------------------------------------------------
    elif args.sub_module == 'analysis':  # -o output, -f figures, -n no_save, -j join, -g delta_g
        save = True
        if args.no_save:
            save = False
        # Start pose analysis of all designed files
        if args.multi_processing:
            # Calculate the number of threads to use depending on computer resources
            threads = SDUtils.calculate_mp_threads(maximum=True)
            logger.info('Starting multiprocessing using %s threads' % str(threads))
            zipped_args = zip(design_directories, repeat(args.delta_g), repeat(args.join), repeat(args.debug),
                              repeat(save), repeat(args.figures))
            # results, exceptions = SDUtils.mp_try_starmap(analyze_output_mp, zipped_args, threads)
            results, exceptions = zip(*SDUtils.mp_starmap(analyze_output_mp, zipped_args, threads))
            results = list(results)
        else:
            logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
            for des_directory in design_directories:
                result, error = analyze_output_s(des_directory, delta_refine=args.delta_g, merge_residue_data=args.join,
                                                 debug=args.debug, save_trajectories=save, figures=args.figures)
                results.append(result)
                exceptions.append(error)

        failures = [index for index, exception in enumerate(exceptions) if exception]
        for index in reversed(failures):
            del results[index]

        if len(design_directories) >= 1:
            # Save Design DataFrame
            design_df = pd.DataFrame(results)
            out_path = os.path.join(args.directory, args.output)
            design_df.to_csv(out_path)
            logger.info('Analysis of all poses written to %s' % out_path)
            if save:
                logger.info('Analysis of all Trajectories and Residues written to %s'
                            % design_directories[0].all_scores)
    # ---------------------------------------------------
    elif args.sub_module == 'merge':  # -d2 directory2, -f2 file2, -i increment, -F force
        directory_pairs, failures = None, None
        if args.directory2 or args.file2:
            # Grab all poses (directories) to be processed from either directory name or file
            all_poses2, location2 = SDUtils.collect_directories(args.directory2, file=args.file2)
            assert all_poses2 != list(), logger.critical(
                'No %s.py directories found within \'%s\'! Please ensure correct location' % (PUtils.nano.title(),
                                                                                              location2))
            all_design_directories2 = set_up_directory_objects(all_poses2)
            logger.info('%d Poses found in \'%s\'' % (len(all_poses2), location2))
            if args.mode == 'design':
                directory_pairs, failures = pair_directories(all_design_directories2, design_directories)
            else:
                logger.warning('Source location was specified, but the --mode isn\'t design. Destination directory '
                               'will be ignored')
        else:
            if args.mode == 'design':
                exit('No source location was specified! Use -d2 or -f2 to specify the source of poses when merging '
                     'design directories')
            elif args.mode == 'dock':
                directory_pairs, failures = pair_dock_directories(design_directories)  #  all_dock_directories)
                for pair in directory_pairs:
                    if args.force:
                        merge_docking_pair(pair, force=True)
                    else:
                        try:
                            merge_docking_pair(pair)
                        except FileExistsError:
                            logger.info('%s directory already exits, moving on. Use --force to overwrite.' % pair[1])
                terminate(exceptions)

        if failures:
            logger.warning('The following directories have no partner:\n%s' % '\n'.join(fail.path
                                                                                        for fail in failures))
        if args.multi_processing:
            # Calculate the number of threads to use depending on computer resources
            threads = SDUtils.calculate_mp_threads(maximum=True)
            logger.info('Starting multiprocessing using %s threads' % str(threads))
            zipped_args = zip(design_directories, repeat(args.increment))
            results = SDUtils.mp_starmap(rename, zipped_args, threads)
            results2 = SDUtils.mp_map(merge_design_pair, directory_pairs, threads)
        else:
            logger.info('Starting processing. If single process is taking awhile, use -mp during submission')
            for des_directory in design_directories:
                rename(des_directory, increment=args.increment)
            for directory_pair in directory_pairs:
                merge_design_pair(directory_pair)
    # ---------------------------------------------------
    elif args.sub_module == 'rename_scores':
        rename = {'combo_profile_switch': 'limit_to_profile', 'favor_profile_switch': 'favor_frag_limit_to_profile'}
        for des_directory in design_directories:
            SDUtils.rename_decoy_protocols(des_directory, rename)
    # ---------------------------------------------------
    elif args.sub_module == 'modify':  # -m mod
        if args.multi_processing:
            if args.mod == 'consolidate_degen':
                exit('Operation impossible')
            elif args.mod == 'pose_map':
                threads = SDUtils.calculate_mp_threads(maximum=True)
                # results, exceptions = zip(*SDUtils.mp_map(fix_files_mp, design_directories, threads=threads))
                pose_map = pose_rmsd_mp(design_directories, threads=threads)
        else:
            if args.mod == 'consolidate_degen':
                logger.info('Consolidating DEGEN directories')
                accessed_dirs = [rsync_dir(design_directory) for design_directory in design_directories]
            elif args.mod == 'pose_map':
                pose_map = pose_rmsd_s(design_directories)

                pose_cluster_map = cluster_poses(pose_map)
                pose_cluster_file = SDUtils.pickle_object(pose_cluster_map, PUtils.clustered_poses,
                                                          out_path=design_directories[0].protein_data)

        # for protein_pair in pose_map:
        #     if os.path.basename(protein_pair) == '4f47_4grd':
        #     logger.info('\n'.join(['%s\n%s' % (pose1, '\n'.join(['%s\t%f' %
        #                                                          (pose2, pose_map[protein_pair][pose1][pose2])
        #                                                          for pose2 in pose_map[protein_pair][pose1]]))
        #                            for pose1 in pose_map[protein_pair]]))

        # errors = []
        # for i, result in enumerate(results):
        #     if not result:
        #         errors.append(design_directories[i].path)
        #
        # logger.error('%d directories missing consensus metrics!' % len(errors))
        # with open('missing_consensus_metrics', 'w') as f:
        #     f.write('\n'.join(errors))
    # ---------------------------------------------------
    elif args.sub_module == 'status':  # -n number, -s stage
        if args.number_designs:
            logger.info('Checking for %d files. If no stage is specified, results will be incorrect for all but design '
                        'stage' % args.number_designs)
        if args.stage:
            status(design_directories, args.stage, number=args.number_designs)
        else:
            for stage in PUtils.stage_f:
                s = status(design_directories, stage, number=args.number_designs)
                if s:
                    logger.info('For \'%s\' stage, default settings should generate %d files'
                                % (stage, PUtils.stage_f[stage]['len']))
    # --------------------------------------------------- # TODO v move to AnalyzeMutatedSequence.py
    elif args.sub_module == 'sequence':  # -c consensus, -d dataframe, -f filters, -n number, -p pose_design_file,
        if args.pose_design_file:        # -s selection_string, -w weights
            # Grab all poses (directories) to be processed from either directory name or file
            with open(args.pose_design_file) as csv_file:
                csv_lines = [line for line in reader(csv_file)]
            all_poses, pose_design_numbers = zip(*csv_lines)
            # pose_design_numbers = list(map(list, pose_design_numbers))
            # all_poses, pose_design_numbers = zip(*reader(args.pose_design_file))
            design_directories = set_up_directory_objects(all_poses, project=args.design_string)
            results.append(zip(design_directories, pose_design_numbers))
            location = args.pose_design_file
        else:
            output = False
            if args.dataframe:  # Figure out poses from a dataframe, filters, and weights
                # TODO parameterize
                if args.filters:
                    exit('Vy made this and I am going to put in here!')
                design_requirements = {'percent_int_area_polar': 0.2, 'buns_per_ang': 0.002}
                crystal_means1 = {'int_area_total': 570, 'shape_complementarity': 0.63, 'number_hbonds': 5}
                crystal_means2 = {'shape_complementarity': 0.63, 'number_hbonds': 5}  # 'int_area_total': 570,
                symmetry_requirements = crystal_means1
                filters = {}
                filters.update(design_requirements)
                filters.update(symmetry_requirements)
                if args.consensus:
                    consensus_weights1 = {'interaction_energy_complex': 0.5, 'percent_fragment': 0.5}
                    consensus_weights2 = {'interaction_energy_complex': 0.33, 'percent_fragment': 0.33,
                                          'shape_complementarity': 0.33}
                    filters = {'percent_int_area_polar': 0.2}
                    weights = consensus_weights2
                else:
                    weights1 = {'protocol_energy_distance_sum': 0.25, 'shape_complementarity': 0.25,
                                'observed_evolution': 0.25, 'int_composition_diff': 0.25}
                    # Used without the interface area filter
                    weights2 = {'protocol_energy_distance_sum': 0.20, 'shape_complementarity': 0.20,
                                'observed_evolution': 0.20, 'int_composition_diff': 0.20, 'int_area_total': 0.20}
                    weights = weights1
                # if not args.weights:
                #     sys.exit('VY made this and I am going to put in here!')

                selected_poses = Ams.filter_pose(args.dataframe, filters, weights,  # num_designs=args.number,
                                                 consensus=args.consensus, filter_file=PUtils.filter_and_sort)

                # Sort results according to clustered poses
                pose_cluster_map = SDUtils.unpickle(
                    os.path.join(design_directories[0].protein_data, '%s.pkl' % PUtils.clustered_poses))
                # {building_blocks: {design_string: cluster_representative}, ...}
                pose_clusters_found, final_poses = [], []
                # for des_dir in design_directories:
                for pose in selected_poses:
                    if pose_cluster_map[pose.split('-')[0]][pose] not in pose_clusters_found:
                        pose_clusters_found.append(pose_cluster_map[pose.split('-')[0]][pose])
                        final_poses.append(pose)

                if len(final_poses) > args.number:
                    final_poses = final_poses[:args.number]
                logger.info('Final poses after clustering:\n%s' % '\n'.join(pose for pose in final_poses))

                design_directories = set_up_directory_objects(final_poses, project=args.design_string)
                if args.consensus:
                    results.append(zip(design_directories, repeat('consensus')))
                    output = True

            if output:
                pass
            elif args.multi_processing:
                # Calculate the number of threads to use depending on computer resources
                threads = SDUtils.calculate_mp_threads(maximum=True)
                logger.info('Starting multiprocessing using %s threads' % str(threads))
                sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
                                    'int_energy_res_summary_delta': 0.25}
                # sequence_weights = None  # Remove once calculated
                zipped_args = zip(design_directories, repeat(sequence_weights))
                results, exceptions = zip(*SDUtils.mp_starmap(Ams.select_sequences_mp, zipped_args, threads))
                # results, exceptions = zip(*SDUtils.mp_map(Ams.select_sequences_mp, design_directories, threads))
                # results - contains tuple of (DesignDirectory, design index) for each sequence
                # could simply return the design index then zip with the directory
            else:
                results, exceptions = zip(*list(Ams.select_sequences_s(des_directory, number=args.number)
                                                for des_directory in design_directories))
                # for des_directory in design_directories:
                #     result, error = select_sequences_s(des_directory, number=args.number)
                #     results.append(result)
                #     exceptions.append(error)

        results = list(results)
        failures = [index for index, exception in enumerate(exceptions) if exception]
        for index in reversed(failures):
            del results[index]

        if not args.selection_string:
            args.selection_string = '%s_' % os.path.basename(os.path.splitext(location)[0])
        else:
            args.selection_string += '_'
        outdir = os.path.join(os.path.dirname(location), '%sSelected_PDBs' % args.selection_string)
        outdir_traj = os.path.join(outdir, 'Trajectories')
        outdir_res = os.path.join(outdir, 'Residues')

        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(outdir_traj)
            os.makedirs(outdir_res)

        # Create symbolic links to the output PDB's
        for pose in results:
            pose_des_dirs, design = zip(*pose)
            for i, pose_des_dir in enumerate(pose_des_dirs):
                file = glob('%s*%s*' % (pose_des_dir.designs, design[i]))
                if file == list():  # If no file found, skip and add to exceptions
                    exceptions.append((pose_des_dir.path, 'No file found for %s*%s*' %
                                       (pose_des_dir.designs, design[i])))
                    continue
                try:
                    os.symlink(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(pose_des_dir), design[i])))
                    os.symlink(pose_des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(pose_des_dir.trajectories)))
                    os.symlink(pose_des_dir.residues, os.path.join(outdir_res, os.path.basename(pose_des_dir.residues)))
                except FileExistsError:
                    pass

        # Format sequences for expression
        chains = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        final_sequences, inserted_sequences = {}, {}
        for pose in results:
            pose_des_dirs, design = zip(*pose)
            for i, pose_des_dir in enumerate(pose_des_dirs):
                # coming in as (chain: seq}
                design_pose = PDB.from_file(glob('%s*%s*' % (pose_des_dir.designs, design[i]))[0])
                # v {chain: sequence, ...}
                design_sequences = AnalyzeMutatedSequences.get_pdb_sequences(design_pose)
                pose_pdbs = os.path.basename(pose_des_dir.building_blocks).split('_')
                # need the original pose chain identity
                # source_pose = PDB(file=pose_des_dir.asu)  # Why can't I use design_sequences? localds quality!
                source_pose = PDB.from_file(pose_des_dir.source)  # Think this is the best, need to modify chains
                source_pose.reorder_chains()
                source_pose.atom_sequences = AnalyzeMutatedSequences.get_pdb_sequences(source_pose)
                # if not source_pose.sequences:
                oligomers = [PDB.from_file(Pose.retrieve_pdb_file_path(pdb)) for pdb in pose_pdbs]
                # oligomers = [SDUtils.read_pdb(SDUtils.retrieve_pdb_file_path(pdb)) for pdb in pose_pdbs]
                oligomer_chain_database_chain_map = {chain: oligomers[k].chain_id_list[0]
                                                     for k, chain in enumerate(source_pose.chain_id_list)}
                # print('SEQRES:\n%s' % '\n'.join(['%s - %s' % (chain, oligomer.sequences[chain])
                #                                  for oligomer in oligomers for chain in oligomer.chain_id_list]))
                source_seqres = {chain: oligomers[k].reference_sequence[oligomers[k].chain_id_list[0]]
                                 for k, chain in enumerate(chains[:len(oligomers)])}

                # seqres_pose = PDB.PDB()
                # for oligomer, _chain in zip(oligomers, reversed(chains)):
                #     # print('Before', oligomer.chain_id_list)
                #     oligomer.rename_chain(oligomer.chain_id_list[0], _chain)
                #     # print('After', oligomer.chain_id_list)
                #     seqres_pose.read_atom_list(oligomer.chain(_chain))
                #     # print('In', seqres_pose.chain_id_list)
                # # print('Out', seqres_pose.chain_id_list)
                # seqres_pose.renumber_residues()  # Why is this necessary
                # seqres_pose.seqres_sequences = source_seqres  # Ams.get_pdb_sequences(seqres_pose,source='seqres')
                # print('Reorder', seqres_pose.chain_id_list)
                # Insert loops identified by comparison of SEQRES and ATOM

                # missing_termini_d = {chain: Ams.generate_mutations_from_seq(pdb_atom_seq[chain],
                #                                                         template_pdb.sequences[chain],
                #                                                         offset=True,
                #                                                         termini=True)
                #                      for chain in template_pdb.chain_id_list}
                # print('Source ATOM Sequences:\n%s' % '\n'.join(['%s - %s' % (chain, source_pose.atom_sequences[chain])
                #                                                 for chain in design_sequences]))
                # print('Source SEQRES Sequences:\n%s' % '\n'.join(['%s - %s' % (chain, source_seqres[chain])
                #                                                   for chain in source_seqres]))
                pose_offset_d = AnalyzeMutatedSequences.pdb_to_pose_num(source_seqres)
                # all_missing_residues_d = {chain: Ams.generate_mutations_from_seq(design_sequences[chain],
                #                                                                  seqres_pose.seqres_sequences[chain],
                #                                                                  offset=True, only_gaps=True)
                #                           for chain in design_sequences}

                # Find all gaps between the SEQRES and ATOM record
                all_missing_residues_d = {chain: SequenceProfile.generate_mutations(source_pose.atom_sequences[chain],
                                                                                    source_seqres[chain], offset=True,
                                                                                    only_gaps=True)
                                          for chain in source_seqres}
                # pose_insert_offset_d = Ams.pdb_to_pose_num(all_missing_residues_d)

                # print('Pre-pose numbering:\n%s' %
                #       '\n'.join(['%s - %s' % (chain, ', '.join([str(res) for res in all_missing_residues_d[chain]]))
                #                  for chain in all_missing_residues_d]))

                # Modify residue indices to pose numbering
                all_missing_residues_d = {chain: {residue + pose_offset_d[chain]: all_missing_residues_d[chain][residue]
                                                  for residue in all_missing_residues_d[chain]}
                                          for chain in all_missing_residues_d}
                # Modify residue indices to include prior pose inserts pose numbering
                # all_missing_residues_d = {chain: {residue + pose_insert_offset_d[chain]: all_missing_residues_d[chain][residue]
                #                                   for residue in all_missing_residues_d[chain]}
                #                           for chain in all_missing_residues_d}
                # print('Post-pose numbering:\n%s' %
                #       '\n'.join(['%s - %s' % (chain, ', '.join([str(res) for res in all_missing_residues_d[chain]]))
                #                  for chain in all_missing_residues_d]))

                # print('Design Sequences:\n%s' % '\n'.join(['%s - %s' % (chain, design_sequences[chain])
                #                                            for chain in design_sequences]))
                # Insert residues into design PDB object
                for chain in all_missing_residues_d:
                    # design_pose.renumber_residues()  TODO for correct pdb_number considering insert_residues function
                    for residue in all_missing_residues_d[chain]:
                        design_pose.insert_residue(chain, residue, all_missing_residues_d[chain][residue]['from'])
                        # if chain == 'B':
                        #     print('%s\tLocation %d' % (design_pose.get_structure_sequence(chain), residue))

                # Get modified sequence
                design_sequences_disordered = AnalyzeMutatedSequences.get_pdb_sequences(design_pose)
                # print('Disordered Insertions:\n%s' %
                #       '\n'.join(['%s - %s' % (chain, design_sequences_disordered[chain])
                #                  for chain in design_sequences_disordered]))

                # I need the source sequence as mutations to get the mutation index on the design sequence
                # grabs the mutated residues from design in the index of the seqres sequence
                mutations = {chain: SequenceProfile.generate_mutations(source_seqres[chain], design_sequences[chain])
                             for chain in design_sequences}
                # print('Mutations:\n%s' %
                #       '\n'.join(['%s - %s' % (chain, mutations[chain]) for chain in mutations]))

                # Next find the correct start MET using the modified (residue inserted) design sequence
                coding_offset = {chain: SequenceProfile.find_orf_offset(design_sequences_disordered[chain],
                                                                        mutations[chain])
                                 for chain in design_sequences_disordered}
                # print('Coding Offset:\n%s'
                #       % '\n'.join(['%s: %s' % (chain, coding_offset[chain]) for chain in coding_offset]))

                # Apply the ORF sequence start to the inserted design sequence, removing all residues prior
                pretag_sequences = {chain: design_sequences_disordered[chain][coding_offset[chain]:]
                                    for chain in coding_offset}
                # print('Pre-tag Sequences:\n%s' % '\n'.join([pretag_sequences[chain] for chain in pretag_sequences]))

                # for residue in all_missing_residues_d:
                #     if all_missing_residues_d[residue]['from'] == 'M':

                # for chain in gapped_residues_d:
                #     for residue in gapped_residues_d[chain]:

                # Check for expression tag addition to the designed sequences
                tag_sequences = {}
                for j, (pdb_code, chain) in enumerate(zip(pose_pdbs, pretag_sequences)):
                    # if sequence doesn't have a tag find all compatible tags
                    if not find_expression_tags(pretag_sequences[chain]):  # == dict():
                        tag_sequences[pdb_code] = \
                            find_all_matching_pdb_expression_tags(pdb_code, oligomer_chain_database_chain_map[chain])
                        # seq = add_expression_tag(tag_with_some_overlap, ORF adjusted design mutation sequence)
                        seq = add_expression_tag(tag_sequences[pdb_code]['seq'], pretag_sequences[chains[j]])
                    else:  # re-use existing
                        # tag_sequences[pdb_code] = None
                        seq = pretag_sequences[chains[j]]

                    # tag_sequences = {pdb: find_all_matching_pdb_expression_tags(pdb,
                    #                                                             oligomer_chain_database_chain_map[chain])
                    #                  for pdb, chain in zip(pose_pdbs, source_pose.chain_id_list)}

                    # for j, pdb_code in enumerate(tag_sequences):
                    #     if tag_sequences[pdb_code]:
                    #         seq = add_expression_tag(tag_sequences[pdb_code]['seq'], pretag_sequences[chains[j]])
                    #     else:
                    #         seq = pretag_sequences[chains[j]]
                    # If no MET start site, include one
                    if seq[0] != 'M':
                        seq = 'M%s' % seq

                    design_string = '%s_design_%s_%s' % (pose_des_dir, design[i], pdb_code)
                    final_sequences[design_string] = seq

                    # For final manual check of the process, find sequence additions compared to the design and output
                    # concatenated to see where additions lie on sequence. Cross these addition with pose pdb to check
                    # if insertion is compatible
                    full_insertions = {residue: {'to': aa}
                                       for residue, aa in enumerate(final_sequences[design_string], 1)}
                    full_insertions.update(
                        SequenceProfile.generate_mutations(design_sequences[chains[j]], final_sequences[design_string],
                                                           blanks=True))
                    # Reduce to sequence only
                    inserted_sequences[design_string] = '%s\n%s' % (''.join([full_insertions[idx]['to']
                                                                             for idx in full_insertions]),
                                                                    final_sequences[design_string])

                # full_insertions = {pdb: Ams.generate_mutations_from_seq(design_sequences[chains[j]],
                #                                                         final_sequences['%s_design_%s_%s' %
                #                                             (pose_des_dir, design[i], pdb)], offset=True, blanks=True)
                #                    for j, pdb in enumerate(tag_sequences)}
                # for pdb in full_insertions:
                #     inserted_sequences['%s_design_%s_%s' % (pose_des_dir, design[i], pdb)] = '%s\n%s' % \
                #         (''.join([full_insertions[pdb][idx]['to'] for idx in full_insertions[pdb]]),
                #          final_sequences['%s_design_%s_%s' % (pose_des_dir, design[i], pdb)])

                # final_sequences[design] = {pdb: add_expression_tag(tag_sequences[pdb]['seq'],
                #                                                    design_sequences[chains[j]])
                #                            for j, pdb in enumerate(tag_sequences)}

        # Write output sequences to fasta file
        seq_file = SequenceProfile.write_fasta_file(final_sequences, '%sSelected_Sequences' % args.selection_string,
                                                    outpath=outdir)
        seq_comparison_file = SequenceProfile.write_fasta_file(inserted_sequences,
                                                               '%sSelected_Sequences_Expression_Additions' %
                                                               args.selection_string, outpath=outdir)
    # -----------------------------------------------------------------------------------------------------------------
    # Report any program exceptions
    terminate(exceptions)
