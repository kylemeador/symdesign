"""
Module for distribution of SymDesign commands. Includes pose initialization, distribution of Rosetta commands to
SLURM computational clusters, analysis of designed poses, and sequence selection of completed structures.

"""
import copy
import datetime
import os
import shutil
from argparse import _SubParsersAction
from subprocess import Popen, list2cmdline
import sys
import time
from glob import glob
from itertools import repeat, product, combinations, chain
from json import loads, dumps
from csv import reader
from typing import List, Any, Union, Dict

import pandas as pd
import psutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data.IUPACData import protein_letters
from DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError

from FragDock import nanohedra_dock
import PathUtils as PUtils
import SymDesignUtils as SDUtils
from Query.PDB import retrieve_pdb_entries_by_advanced_query
from Query.utils import input_string, bool_d, validate_input, boolean_choice, invalid_string
from utils.CmdLineArgParseUtils import query_mode
from classes.SymEntry import SymEntry, parse_symmetry_to_sym_entry
from classes.EulerLookup import EulerLookup
from CommandDistributer import distribute, hhblits_memory_threshold, update_status
from PoseDirectory import PoseDirectory
from JobResources import JobResources, fragment_factory
from PDB import PDB, orient_pdb_file
from ClusterUtils import cluster_designs, invert_cluster_map, group_compositions, ialign  # pose_rmsd, cluster_poses
from ProteinExpression import find_expression_tags, find_matching_expression_tags, add_expression_tag, \
    select_tags_for_sequence, remove_expression_tags, expression_tags, optimize_protein_sequence, \
    default_multicistronic_sequence
from DesignMetrics import prioritize_design_indices, query_user_for_metrics
from SequenceProfile import generate_mutations, find_orf_offset, write_fasta, read_fasta_file  # , pdb_to_pose_offset
from utils.Flags import argparsers, parser_entire, parser_options, parser_module, parser_input, parser_guide_module, \
    process_residue_selector_flags, return_default_flags
from utils.GeneralUtils import write_docking_parameters
from utils.SetUp import set_up_instructions
from utils.guide import interface_design_guide, analysis_guide, interface_metrics_guide, select_poses_guide, \
    select_sequences_guide, cluster_poses_guide, refine_guide, optimize_designs_guide


def rename(des_dir, increment=PUtils.nstruct):
    """Rename the decoy numbers in a PoseDirectory by a specified increment

    Args:
        des_dir (PoseDirectory): A PoseDirectory object
    Keyword Args:
        increment=PUtils.nstruct (int): The number to increment by
    """
    for pdb in os.listdir(des_dir.designs):
        if os.path.splitext(pdb)[0][-1].isdigit():
            SDUtils.change_filename(os.path.join(des_dir.designs, pdb), increment=increment)
    SDUtils.modify_decoys(os.path.join(des_dir.scores, PUtils.scores_file), increment=increment)


def pair_directories(dirs2, dirs1):
    """Pair directories with the same pose name, returns source (dirs2) first, destination (dirs1) second
    Args:
        dirs2 (list): List of PoseDirectory objects
        dirs1 (list): List of PoseDirectory objects
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
        # where the link will attach too. Adding the flipped suffix to the composition name
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

    raise RuntimeError('The function %s is no longer operational as of 5/13/22' % 'get_building_block_dir')
    for s, sym_dir in enumerate(des_dir.program_root):
        if '_flipped_180y' in sym_dir:
            for bb_dir in des_dir.composition[s]:
                building_block_dir = os.path.join(des_dir.get_building_block_dir(bb_dir))
                destination = os.path.join(building_block_dir, 'DEGEN_1_1%s' % os.sep)

                p = {}  # make a dict for all the processes
                for k, entry in enumerate(os.scandir(building_block_dir)):
                    if entry.is_dir() and 'DEGEN_1_' in entry.name and entry.name != "DEGEN_1_1":
                        abs_entry_path = os.path.join(building_block_dir, entry.name)
                        cmd = ['rsync', '-a', '--link-dest=%s%s' % (abs_entry_path, os.sep), #  '--remove-source-files',
                               '%s%s' % (abs_entry_path, os.sep), destination]
                        #          ^ requires '/' - os.sep
                        logger.debug('Performing transfer: %s' % list2cmdline(cmd))
                        p[abs_entry_path] = Popen(cmd)
                # Check to see if all processes are done, then move on.
                for entry in p:
                    p[entry].communicate()

                # # Remove all empty subdirectories from the now empty DEGEN_1_2 and higher directory
                # p2 = {}
                # for l, entry in enumerate(p):
                #     find_cmd = ['find', '%s%s' % (entry, os.sep), '-type', 'd', '-empty', '-delete']
                #     # rm_cmd = ['rm', '-r', entry]
                #     logger.debug('Removing empty directories: %s' % list2cmdline(find_cmd))
                #     p2[l] = Popen(find_cmd)
                # # Check for the last command, then move on
                # for m, process in enumerate(p2):
                #     p2[m].communicate()

                logger.info('%s has been consolidated' % building_block_dir)

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
                rmsd_file = sorted(glob(os.path.join(rmsd_dir, 'crystal_vs_docked_irmsd.txt')))[0]
                # ensure that RMSD files were created for the most recent set of results using 'start'
                if int(time.time()) - int(os.path.getmtime(rmsd_file)) > start.now().timestamp() - start.timestamp():
                    rmsd_file = None
                all_to_all_file = sorted(glob(os.path.join(rmsd_dir, 'top*_all_to_all_docked_poses_irmsd.txt')))[0]
                final_clustering_file = sorted(glob(os.path.join(rmsd_dir, '*_clustered.txt')))[0]
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
                    # _status = os.path.join(_rmsd_dir(pose_directories[k], outcome_strings_d[r]))
                    running.append(os.path.join(_rmsd_dir(pose_directories[k].symmetry[0]), outcome_strings_d[r]))
                    break
            if _status:
                complete.append(sorted(glob(os.path.join(_rmsd_dir(all_design_directories[k].program_root[0]),
                                                         '*_clustered.txt')))[0])

    elif _stage == PUtils.nano:
        from classes import get_last_sampling_state
        # observed_building_blocks = []
        for des_dir in all_design_directories:
            # if os.path.basename(des_dir.composition) in observed_building_blocks:
            #     continue
            # else:
            #     observed_building_blocks.append(os.path.basename(des_dir.composition))
            # f_degen1, f_degen2, f_rot1, f_rot2 = get_last_sampling_state('%s_log.txt' % des_dir.composition)
            # degens, rotations = \
            # SDUtils.degen_and_rotation_parameters(SDUtils.gather_docking_metrics(des_dir.program_root))
            #
            # dock_dir = PoseDirectory(path, auto_structure=False)
            # dock_dir.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses'))
            # dock_dir.composition = [next(os.walk(dir))[1] for dir in dock_dir.program_root]
            # dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.program_root]
            # dock_dir.building_block_logs = [os.path.join(_sym, bb_dir, 'bb_dir_log.txt') for sym in dock_dir.composition
            #                                 for bb_dir in sym]

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
                    raise DesignError('This functionality has been removed "des_dir.gather_docking_metrics()"')
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
                    logger.info('Last State:', f_degen1, f_degen2, f_rot1, f_rot2)
                    logger.info('Expected:', degen1, degen2, last_rot1, last_rot2)
                    logger.info('From log: %s' % log_file)
                    if f_degen1 == degen1 and f_degen2 == degen2 and f_rot1 == last_rot1 and f_rot2 == last_rot2:
                        complete.append(os.path.join(des_dir.program_root[sym_idx], des_dir.composition[sym_idx][bb_idx]))
                        # complete.append(des_dir)
                        # complete.append(des_dir.program_root)
                    else:
                        if active:
                            if int(time.time()) - int(os.path.getmtime(log_file)) < inactive_time:
                                running.append(os.path.join(des_dir.program_root[sym_idx],
                                                            des_dir.composition[sym_idx][bb_idx]))
                        incomplete.append(os.path.join(des_dir.program_root[sym_idx],
                                                       des_dir.composition[sym_idx][bb_idx]))
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

        for des_dir in pose_directories:
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


def format_additional_flags(flags):
    """Takes non-argparse specified flags and returns them into a dictionary compatible with argparse style.
    This is useful for handling general program flags that apply to many modules, would be tedious to hard code and
    request from the user

    Returns:
        (dict)
    """
    # combines ['-symmetry', 'O', '-nanohedra_output', True', ...]
    combined_extra_flags = []
    for idx, flag in enumerate(flags):
        if flag[0] == '-' and flag[1] == '-':  # format flags by removing extra '-'. Issue with PyMol command in future?
            flags[idx] = flag[1:]

        if flag.startswith('-'):  # this is a real flag
            extra_arguments = ''
            # iterate over arguments after the flag until a flag with "-" is reached. This is a new flag
            increment = 1
            while (idx + increment) != len(flags) and not flags[idx + increment].startswith('-'):  # we have an argument
                extra_arguments += ' %s' % flags[idx + increment]
                increment += 1
            # remove - from the front and add all arguments to single flag argument list item
            combined_extra_flags.append('%s%s' % (flag.lstrip('-'), extra_arguments))  # extra_flags[idx + 1]))
    # logger.debug('Combined flags: %s' % combined_extra_flags)

    # parse the combined flags ['-nanohedra_output True', ...]
    final_flags = {}
    for flag_arg in combined_extra_flags:
        if ' ' in flag_arg:
            flag, *args = flag_arg.split()  #[0]
            # flag = flag.lstrip('-')
            # final_flags[flag] = flag_arg.split()[1]
            if len(args) > 1:  # we have multiple arguments, set all to the flag
                final_flags[flag] = args

            # check for specific strings and set to corresponding python values
            elif args[0].lower() == 'true':
                # final_flags[flag] = eval(final_flags[flag].title())
                final_flags[flag] = True
            elif args[0].lower() == 'false':
                final_flags[flag] = False
            elif args[0].lower() == 'none':
                final_flags[flag] = None
            else:
                final_flags[flag] = args[0]
        else:  # add to the dictionary with default argument of True
            final_flags[flag_arg] = True

    return final_flags


sbatch_warning = 'Ensure the SBATCH script(s) below are correct. Specifically, check that the job array and any '\
                 'node specifications are accurate. You can look at the SBATCH manual (man sbatch or sbatch --help) to'\
                 ' understand the variables or ask for help if you are still unsure.'


def terminate(results: Union[List[Any], Dict] = None, output: bool = True):
    """Format designs passing output parameters and report program exceptions

    Args:
        results: The returned results from the module run. By convention contains results and exceptions
        output: Whether the module used requires a file to be output
    """
    global out_path
    global design_source
    # save any information found during the design command to it's serialized state
    for design in pose_directories:
        design.pickle_info()

    if results:
        if pose_directories:  # pose_directories is empty list when nano
            success = \
                [pose_directories[idx] for idx, result in enumerate(results) if not isinstance(result, BaseException)]
            exceptions = \
                [(pose_directories[idx], result) for idx, result in enumerate(results)
                 if isinstance(result, BaseException)]
        else:
            success, exceptions = results, []
    else:
        success, exceptions = [], []

    exit_code = 0
    if exceptions:
        print('\n')
        logger.warning('Exceptions were thrown for %d designs. Check their logs for further details\n\t%s' %
                       (len(exceptions), '\n\t'.join('%s: %s' % (str(design.path), _error)
                                                     for (design, _error) in exceptions)))
        print('\n')
        # exit_code = 1

    if success and output:
        job_paths = job.job_paths
        job.make_path(job_paths)
        if low and high:
            design_source = '%s-%.2f-%.2f' % (design_source, low, high)
        # Make single file with names of each directory where all_docked_poses can be found
        # project_string = os.path.basename(pose_directories[0].project_designs)
        default_output_tuple = (SDUtils.starttime, args.module, design_source)
        if args.output_file and args.module not in [PUtils.analysis, PUtils.cluster_poses]:
            designs_file = args.output_file
        else:
            designs_file = os.path.join(job_paths, PUtils.default_path_file % default_output_tuple)

        if pose_directories:  # pose_directories is empty list when nano
            with open(designs_file, 'w') as f:
                f.write('%s\n' % '\n'.join(design.path for design in success))
            logger.critical('The file "%s" contains the locations of all designs in your current project that passed '
                            'internal checks/filtering. Utilize this file to interact with %s designs in future '
                            'commands for this project such as:\n\t%s --file %s MODULE\n'
                            % (designs_file, PUtils.program_name, PUtils.program_command, designs_file))

        if args.module == PUtils.analysis:
            all_scores = job.all_scores
            # Save Design DataFrame
            design_df = pd.DataFrame([result for result in results if not isinstance(result, BaseException)])
            args.output_file = args.output_file if args.output_file.endswith('.csv') else '%s.csv' % args.output_file
            design_df.to_csv(args.output_file)
            logger.info('Analysis of all poses written to %s' % args.output_file)
            if args.save:
                logger.info('Analysis of all Trajectories and Residues written to %s' % all_scores)
        elif args.module == PUtils.cluster_poses:
            logger.info('Clustering analysis results in the following similar poses:\nRepresentatives\n\tMembers\n')
            for representative, members, in results.items():
                print('%s\n\t%s' % (representative, '\n\t'.join(map(str, members))))
            logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                        % (len(pose_cluster_map), len(pose_directories), pose_cluster_file))
            logger.info('Each cluster above has one representative which identifies with each of the members. If '
                        'clustering was performed by transformation or interface_residues, then the representative is '
                        'the most similar to all members. If clustering was performed by ialign, then the '
                        'representative is randomly chosen.')
            logger.info('To utilize the above clustering, during %s, using the option --cluster_map, will apply '
                        'clustering to poses to select a cluster representative based on the most favorable cluster '
                        'member' % PUtils.select_poses)

        # Set up sbatch scripts for processed Poses
        design_stage = PUtils.scout if getattr(args, PUtils.scout, None) \
            else (PUtils.interface_design if getattr(args, PUtils.no_hbnet, None)
                  else (PUtils.structure_background if getattr(args, PUtils.structure_background, None)
                        else PUtils.hbnet_design_profile))
        module_files = {PUtils.interface_design: design_stage, PUtils.nano: PUtils.nano, PUtils.refine: PUtils.refine,
                        PUtils.interface_metrics: PUtils.interface_metrics,
                        # 'custom_script': os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0],
                        PUtils.optimize_designs: PUtils.optimize_designs}
        stage = module_files.get(args.module)
        if stage and not args.run_in_shell:
            if len(success) == 0:
                exit_code = 1
                exit(exit_code)
            if pose_directories:
                command_file = SDUtils.write_commands([os.path.join(des.scripts, '%s.sh' % stage) for des in success],
                                                      out_path=job_paths, name='_'.join(default_output_tuple))
                sbatch_file = distribute(file=command_file, out_path=job.sbatch_scripts, scale=args.module)
                #                                                                        ^ for sbatch template
            else:  # pose_directories is empty list when nano, use success as the commands holder
                command_file = SDUtils.write_commands([list2cmdline(cmd) for cmd in success], out_path=job_paths,
                                                      name='_'.join(default_output_tuple))
                sbatch_file = distribute(file=command_file, out_path=job.sbatch_scripts, scale=args.module,
                                         number_of_commands=len(success))
            logger.critical(sbatch_warning)
            global pre_refine
            if args.module == PUtils.interface_design and not pre_refine:  # False, so should refine before design
                refine_file = SDUtils.write_commands([os.path.join(design.scripts, f'{PUtils.refine}.sh')
                                                      for design in success], out_path=job_paths,
                                                     name='_'.join((SDUtils.starttime, PUtils.refine, design_source)))
                sbatch_refine_file = distribute(file=refine_file, out_path=job.sbatch_scripts, scale=PUtils.refine)
                logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch {sbatch_refine_file}'
                            f'\nTHEN:\n\tsbatch {sbatch_file}')
            else:
                logger.info(f'Once you are satisfied, enter the following to distribute:\n\tsbatch {sbatch_file}')

    # test for the size of each of the designdirectories
    if pose_directories:
        print('Before fragments\naverage_design_directory_size equals %f' %
              (float(psutil.virtual_memory().used) / len(pose_directories)))
        global fragment_db
        try:
            del fragment_db
            del job.fragment_db
        except (NameError, AttributeError) as error:
            print(error)
        time.sleep(10)  # let the garbage collector get rid of any ties
        print('After fragments\naverage_design_directory_size equals %f' %
              (float(psutil.virtual_memory().used) / len(pose_directories)))

    print('\n')
    exit(exit_code)


def load_total_dataframe(pose: bool = False) -> pd.DataFrame:
    """Return a pandas DataFrame with the trajectories of every pose_directory loaded and formatted according to the
    design directory and design on the index"""
    global pose_directories
    # global results
    all_dfs = []  # None for design in pose_directories]
    for idx, design in enumerate(pose_directories):
        try:
            all_dfs.append(pd.read_csv(design.trajectories, index_col=0, header=[0]))
        except FileNotFoundError as error:
            # results[idx] = error
            logger.warning('%s: No trajectory analysis found. Skipping' % str(design))

    if pose:
        for idx, df in enumerate(all_dfs):
            # get rid of all individual trajectories and std, not mean
            design_name = pose_directories[idx].name
            df.fillna(0., inplace=True)  # shouldn't be necessary if saved files were formatted correctly
            # try:
            df.drop([index for index in df.index.to_list() if isinstance(index, float)], inplace=True)
            df.drop([index for index in df.index.to_list() if design_name in index or 'std' in index], inplace=True)
            # except TypeError:
            #     for index in df.index.to_list():
            #         print(index, type(index))
    else:  # designs
        for idx, df in enumerate(all_dfs):
            # get rid of all statistic entries, mean, std, etc.
            design_name = pose_directories[idx].name
            df.drop([index for index in df.index.to_list() if design_name not in index], inplace=True)
    df = pd.concat(all_dfs, keys=pose_directories)  # must add the design directory string to each index
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


def generate_sequence_template(pdb_file):
    pdb = PDB.from_file(pdb_file, entities=False)
    sequence = SeqRecord(Seq(''.join(chain.sequence for chain in pdb.chains), 'Protein'), id=pdb.filepath)
    sequence_mask = copy.copy(sequence)
    sequence_mask.id = 'residue_selector'
    sequences = [sequence, sequence_mask]
    return write_fasta(sequences, file_name='%s_residue_selector_sequence' % os.path.splitext(pdb.filepath)[0])


def get_sym_entry_from_nanohedra_directory(nanohedra_dir):
    try:
        with open(os.path.join(nanohedra_dir, PUtils.master_log), 'r') as f:
            for line in f.readlines():
                if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                    return SymEntry(int(line.split(':')[-1]))  # sym_map inclusion?
    except FileNotFoundError:
        raise FileNotFoundError('The Nanohedra Output Directory is malformed. Missing required docking file %s'
                                % os.path.join(nanohedra_dir, PUtils.master_log))
    raise SDUtils.DesignError('The Nanohedra Output docking file %s is malformed. Missing required info Nanohedra Entry'
                              ' Number' % os.path.join(nanohedra_dir, PUtils.master_log))


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # Process optional program flags
    # -----------------------------------------------------------------------------------------------------------------
    args, additional_args = argparsers[parser_options].parse_known_args()  # additional_args, args)
    # -----------------------------------------------------------------------------------------------------------------
    # Start Logging - Root logs to stream with level warning
    # -----------------------------------------------------------------------------------------------------------------
    if args.debug:
        # Root logs to stream with level debug
        logger = SDUtils.start_log(level=1)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        # Root logger logs to stream with level 'warning'
        SDUtils.start_log(level=3, set_handler_level=True)
        # Root logger logs all emissions to a single file with level 'info'. Stream above still emits at 'warning'
        SDUtils.start_log(handler=2, location=os.path.join(os.getcwd(), args.output_directory if args.output_directory else '', PUtils.program_name))
        # SymDesign main logs to stream with level info
        logger = SDUtils.start_log(name=PUtils.program_name, propagate=False)
        # All Designs will log to specific file with level info unless -skip_logging is passed
    # -----------------------------------------------------------------------------------------------------------------
    # Display the program guide if requested
    # -----------------------------------------------------------------------------------------------------------------
    if args.guide:  # or not args.module:
        args, additional_args = argparsers[parser_guide_module].parse_known_args(additional_args, args)
        if args.module == PUtils.analysis:
            logger.info(analysis_guide)
        elif args.module == PUtils.cluster_poses:
            logger.info(cluster_poses_guide)
        elif args.module == PUtils.interface_design:
            logger.info(interface_design_guide)
        elif args.module == PUtils.interface_metrics:
            logger.info(interface_metrics_guide)
        # elif args.module == 'custom_script':
        #     logger.info()
        elif args.module == PUtils.optimize_designs:
            logger.info(optimize_designs_guide)
        elif args.module == PUtils.refine:
            logger.info(refine_guide)
        elif args.module == PUtils.nano:
            logger.info()
        elif args.module == PUtils.select_poses:
            logger.info(select_poses_guide)
        elif args.module == PUtils.select_sequences:
            logger.info(select_sequences_guide)
        elif args.module == 'expand_asu':
            logger.info()
        elif args.module == 'check_clashes':
            logger.info()
        elif args.module == 'residue_selector':
            logger.info()
        # elif args.module == 'visualize':
        #     logger.info('Usage: %s -r %s -- [-d %s, -df %s, -f %s] visualize --range 0-10'
        #                 % (SDUtils.ex_path('pymol'), PUtils.program_command.replace('python ', ''),
        #                    SDUtils.ex_path('pose_directory'), SDUtils.ex_path('DataFrame.csv'),
        #                    SDUtils.ex_path('design.paths')))
        else:  # print the full program readme
            with open(PUtils.readme, 'r') as f:
                print(f.read(), end='')
        exit()
    elif args.set_up:
        set_up_instructions()
        exit()
    # ---------------------------------------------------
    # elif args.flags:  # Todo
    #     if args.template:
    #         Flags.query_user_for_flags(template=True)
    #     else:
    #         Flags.query_user_for_flags(mode=args.flags_module)
    # ---------------------------------------------------
    # elif args.module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
    #     distribute(**vars(args))
    # ---------------------------------------------------
    # elif args.residue_selector:  # Todo
    #     if not args.single:
    #         raise SDUtils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
    #                                   'residue_selector' % (PUtils.program_name, PUtils.program_command))
    #     fasta_file = generate_sequence_template(args.single)
    #     logger.info('The residue_selector template was written to %s. Please edit this file so that the '
    #                 'residue_selector can be generated for protein design. Selection should be formatted as a "*" '
    #                 'replaces all sequence of interest to be considered in design, while a Mask should be formatted as '
    #                 'a "-". Ex:\n>pdb_template_sequence\nMAGHALKMLV...\n>residue_selector\nMAGH**KMLV\n\nor'
    #                 '\n>pdb_template_sequence\nMAGHALKMLV...\n>design_mask\nMAGH----LV\n'
    #                 % fasta_file)
    # -----------------------------------------------------------------------------------------------------------------
    # Process remaining flags and arguments for program initialization
    # -----------------------------------------------------------------------------------------------------------------
    # ensure module specific arguments are collected and argument help is printed in full
    entire_parser = argparsers[parser_entire]
    entire_parser.parse_known_args()
    # parse arguments for the actual runtime which accounts for differential argument ordering from standard argparse
    argparser_order = [parser_input, parser_module]  # , parser_options]
    for argparser in argparser_order:
        args, additional_args = argparsers[argparser].parse_known_args(additional_args, args)
    # args, additional_args = parser_module.parse_known_args()
    # args, additional_args = parser_input.parse_known_args(additional_args, args)
    if additional_args:
        exit('\nSuspending run. Found flag(s) that are not recognized program wide: %s\nPlease correct (try adding '
             '--help if unsure), and resubmit your command\n' % ', '.join(additional_args))
    default_flags = return_default_flags()
    formatted_flags = format_additional_flags(additional_args)
    default_flags.update(formatted_flags)

    # Add additional program flags to queried_flags
    queried_flags = vars(args)
    queried_flags.update(default_flags)
    queried_flags.update(process_residue_selector_flags(queried_flags))
    # We have to ensure that if the user has provided it, the symmetry is correct
    if queried_flags['symmetry'] and queried_flags.get(PUtils.sym_entry):
        queried_flags[PUtils.sym_entry] = \
            parse_symmetry_to_sym_entry(sym_entry=queried_flags[PUtils.sym_entry], symmetry=queried_flags['symmetry'])
    elif queried_flags.get(PUtils.sym_entry):
        queried_flags[PUtils.sym_entry] = SymEntry(int(queried_flags[PUtils.sym_entry]))
    elif queried_flags['symmetry']:
        if queried_flags['symmetry'].lower()[:5] == 'cryst':
            # the symmetry information is in the pdb header
            queried_flags['symmetry'] = 'cryst'
        else:  # queried_flags['symmetry'] in possible_symmetries:
            queried_flags[PUtils.sym_entry] = parse_symmetry_to_sym_entry(symmetry=queried_flags['symmetry'])
    sym_entry = queried_flags[PUtils.sym_entry]

    initialize_modules = [PUtils.nano, PUtils.interface_design, PUtils.interface_metrics, PUtils.optimize_designs]
    #                       PUtils.analysis, 'custom_script', PUtils.refine]
    # TODO consolidate these checks
    if args.module in [PUtils.interface_design, PUtils.generate_fragments, 'orient', 'expand_asu',
                       PUtils.interface_metrics, PUtils.refine, PUtils.optimize_designs, 'rename_chains',
                       'check_clashes']:  # , 'custom_script', 'find_asu', 'status', 'visualize'
        initialize, queried_flags['construct_pose'] = True, True  # set up design directories
    elif args.module in [PUtils.analysis, PUtils.cluster_poses, PUtils.select_poses, PUtils.select_sequences]:
        # analysis types can be run from nanohedra_output, so we ensure that we don't construct new
        initialize, queried_flags['construct_pose'] = True, False
        if args.module == PUtils.select_poses:
            if args.dataframe:  # when selecting by a dataframe, don't initialize, proper input is handled in module
                initialize = False
        if args.module == PUtils.select_sequences and args.select_number == sys.maxsize and not args.total:
            # change default number to a single sequence/pose when not doing a total selection
            args.select_number = 1
    else:  # [PUtils.nano, 'flags', 'residue_selector', 'multicistronic']
        initialize = False
        if getattr(args, 'query', None):  # run nanohedra query mode
            query_flags = [__file__, '-query'] + additional_args
            logger.debug('Query %s.py with: %s' % (PUtils.nano.title(), ', '.join(query_flags)))
            query_mode(query_flags)
            terminate(output=False)

    if args.module not in ['multicistronic']:  # Todo why is multicistronic here? 'flags', 'residue_selector',
        formatted_queried_flags = queried_flags.copy()
        # where input values should be reported instead of processed version, or the argument is not important
        for flag in ['design_selector', 'construct_pose']:
            formatted_queried_flags.pop(flag, None)
            # get all the default program args and compare them to the provided values
        reported_args = {}
        for group in entire_parser._action_groups:
            for arg in group._group_actions:
                if isinstance(arg, _SubParsersAction):
                    for name, sub_parser in arg.choices.items():
                        for sub_group in sub_parser._action_groups:
                            for arg in sub_group._group_actions:
                                value = formatted_queried_flags.pop(arg.dest, None)  # get the parsed flag value
                                if value is not None and value != arg.default:  # compare it to the default
                                    reported_args[arg.dest] = value  # add it to reported args if not the default
                else:
                    value = formatted_queried_flags.pop(arg.dest, None)  # get the parsed flag value
                    if value is not None and value != arg.default:  # compare it to the default
                        reported_args[arg.dest] = value  # add it to reported args if not the default

        # custom removal/formatting for all remaining
        for custom_arg in list(formatted_queried_flags.keys()):
            value = formatted_queried_flags.pop(custom_arg, None)
            if value is not None:
                reported_args[custom_arg] = value

        flags_sym_entry = reported_args.pop(PUtils.sym_entry, None)
        if flags_sym_entry:
            reported_args[PUtils.sym_entry] = flags_sym_entry.entry_number
        logger.info('Starting with options:\n\t%s' % '\n\t'.join(SDUtils.pretty_format_table(reported_args.items())))
    # -----------------------------------------------------------------------------------------------------------------
    # Initialize common resources
    # -----------------------------------------------------------------------------------------------------------------
    # Check if output already exists  # or provide --overwrite
    if args.output_file and os.path.exists(args.output_file):
        logger.critical(f'The specified output file "{args.output_file}" already exists, this will overwrite your old '
                        f'data! Please specify a new name with with -Of/--output_file')
        exit(1)
    # elif args.output_directory and os.path.exists(args.output_directory):  # Todo is this necessary?
    #     logger.critical(f'The specified output directory "{args.output_file}" already exists, this will overwrite
    #                     f'your old data! Please specify a new one with with -Od/--output_directory')
    #     exit(1)

    # Set up JobResources which handles flags and shared program objects necessary for processing and i/o
    symdesign_directory = SDUtils.get_base_symdesign_dir(args.directory)
    if not symdesign_directory:  # Not from SymDesignOutput
        if args.output_directory:  # use a user specified directory
            # if args.output_directory == '':
            #     symdesign_directory = '%s_%s_%s_poses' % default_output_tuple
            # else:
            queried_flags['output_directory'] = True
            symdesign_directory = args.output_directory
        else:
            symdesign_directory = os.path.join(os.getcwd(), PUtils.program_output)
        os.makedirs(symdesign_directory, exist_ok=True)

    job = JobResources(symdesign_directory, **queried_flags)
    logger.info(f'Using JobResources from Database located at "{job.protein_data}"')
    queried_flags['job_resources'] = job

    # Set up Databases
    if args.module in [PUtils.nano, PUtils.generate_fragments, PUtils.interface_design] or \
            queried_flags.get(PUtils.generate_fragments, None):  # Todo remove, only in interface_design_analysis
        if queried_flags.get('no_term_constraint', None):
            fragment_db, euler_lookup = None, None
        else:
            queried_flags[PUtils.generate_fragments] = True
            fragment_db = fragment_factory(source=args.fragment_database)
            euler_lookup = EulerLookup()
    else:
        fragment_db, euler_lookup = None, None

    job.fragment_db = fragment_db
    job.euler_lookup = euler_lookup

    # -----------------------------------------------------------------------------------------------------------------
    # Grab all Designs (PoseDirectory) to be processed from either database, directory, project name, or file
    # -----------------------------------------------------------------------------------------------------------------
    all_poses: List[Union[str, bytes]] | None = None
    pose_directories: List[PoseDirectory] = []
    location: str | None = None
    all_dock_directories, entity_pairs = None, None
    low, high, low_range, high_range = None, None, None, None
    pre_refine, pre_loop_model = None, None  # set below if needed
    if initialize:
        if args.range:
            low, high = map(float, args.range.split('-'))
            low_range, high_range = int((low / 100) * len(all_poses)), int((high / 100) * len(all_poses))
            if low_range < 0 or high_range > len(all_poses):
                raise SDUtils.DesignError('The input --range is outside of the acceptable bounds [0-100]')
            logger.info(f'Selecting poses within range: {low_range if low_range else 1}-{high_range}')

        logger.info(f'Setting up input files for {args.module}')
        if args.nanohedra_output:  # Nanohedra directory
            all_poses, location = SDUtils.collect_nanohedra_designs(files=args.file, directory=args.directory)
            if all_poses:
                if all_poses[0].count(os.sep) == 0:
                    job.nanohedra_root = args.directory
                else:
                    job.nanohedra_root = f'{os.sep}{os.path.join(*all_poses[0].split(os.sep)[:-4])}'
                if not sym_entry:
                    queried_flags[PUtils.sym_entry] = get_sym_entry_from_nanohedra_directory(job.nanohedra_root)
                pose_directories = [PoseDirectory.from_nanohedra(pose, **queried_flags)
                                    for pose in all_poses[low_range:high_range]]
        elif args.specification_file:  # Todo, combine this with collect_designs
            if not args.directory:
                raise SDUtils.DesignError('A --directory must be provided when using --specification_file')
            design_specification = SDUtils.PoseSpecification(args.specification_file)
            # Todo this works for file locations as well! should I have a separate mechanism for each?
            pose_directories = [PoseDirectory.from_pose_id(pose, root=args.directory, specific_design=design,
                                                           directives=directives, **queried_flags)
                                for pose, design, directives in design_specification.return_directives()]
            location = f'{os.path.basename(args.directory)}&{args.specification_file}'
        else:
            all_poses, location = SDUtils.collect_designs(files=args.file, directory=args.directory,
                                                          projects=args.project, singles=args.single)
            if all_poses:
                if all_poses[0].count(os.sep) == 0:
                    # TODO in the case the list of files is in the current directory where SymDesign was run...
                    # assume that we have received pose-IDs and process accordingly
                    pose_directories = [PoseDirectory.from_pose_id(pose, root=args.directory, **queried_flags)
                                        for pose in all_poses[low_range:high_range]]
                else:
                    pose_directories = [PoseDirectory.from_file(pose, **queried_flags)
                                        for pose in all_poses[low_range:high_range]]
        if not pose_directories:
            raise SDUtils.DesignError(f'No {PUtils.program_name} directories found within "{location}"! Please ensure '
                                      f'correct location')
        # Todo could make after collect_designs? Pass to all pose_directories
        #  for file, take all_poses first file. I think prohibits multiple dirs, projects, single...
        example_directory = next(iter(pose_directories))
        # example_directory = JobResources(pose_directories[0].program_root)
        # if not location:
        #     design_source = os.path.basename(example_directory.project_designs)
        # else:
        design_source = os.path.splitext(os.path.basename(location))[0]
        default_output_tuple = (SDUtils.starttime, args.module, design_source)

        # Todo logic error when initialization occurs with module that doesn't call this, subsequent runs are missing
        #  directories/resources that haven't been made
        # check to see that proper files have been created if doing design
        # including orientation, refinement, loop modeling, hhblits, bmdca?
        initialized = example_directory.initialized
        if not initialized and args.module in initialize_modules or args.nanohedra_output or args.load_database:
            # if args.load_database:  # Todo why is this set_up_design_directory here?
            #     for design in pose_directories:
            #         design.set_up_design_directory()
            # args.orient, args.refine = True, True  # Todo make part of argparse? Could be variables in NanohedraDB
            # for each pose_directory, ensure that the pdb files used as source are present in the self.orient_dir
            orient_dir = job.orient_dir
            orient_asu_dir = job.orient_asu_dir
            stride_dir = job.stride_dir
            load_resources = False
            if args.preprocessed:
                all_entities, found_entity_names = [], []
                for entity in [entity for design in pose_directories for entity in design.init_pdb.entities]:
                    if entity.name not in found_entity_names:
                        all_entities.append(entity)
                        found_entity_names.append(entity.name)
            else:
                logger.critical('The requested poses require preprocessing before design modules should be used')
                # Collect all entities required for processing the given commands
                required_entities = list(map(set, list(zip(*[design.entity_names for design in pose_directories]))))
                all_entities = []
                # Select entities, orient them, then load each entity to all_entities for further database processing
                symmetry_map = sym_entry.groups if sym_entry else repeat(None)
                for symmetry, entities in zip(symmetry_map, required_entities):
                    if not entities:  # useful in a case where symmetry groups are the same
                        continue
                    elif not symmetry:
                        logger.info('PDB files are being processed without consideration for symmetry: %s'
                                    % ', '.join(entities))
                        raise RuntimeError('This is not implemented!')
                        all_entities.extend()
                        continue
                    elif symmetry == 'C1':
                        logger.info('PDB files are being processed with C1 symmetry: %s'
                                    % ', '.join(entities))
                    else:
                        logger.info('Ensuring PDB files are oriented with %s symmetry (stored at %s): %s'
                                    % (symmetry, orient_dir, ', '.join(entities)))
                    all_entities.extend(job.resources.orient_entities(entities, symmetry=symmetry))

            info_messages = []
            # set up the hhblits and profile bmdca for each input entity
            profile_dir = job.profiles
            job.make_path(profile_dir)
            sequences_dir = job.sequences
            job.make_path(sequences_dir)
            hhblits_cmds, bmdca_cmds = [], []
            for entity in all_entities:
                entity.sequence_file = job.resources.sequences.retrieve_file(name=entity.name)
                if not entity.sequence_file:  # Todo reference_sequence source accuracy throughout protocol
                    entity.write_fasta_file(entity.reference_sequence, name=entity.name, out_path=sequences_dir)
                    # entity.add_evolutionary_profile(out_path=job.resources.hhblits_profiles.location)
                else:
                    entity.evolutionary_profile = job.resources.hhblits_profiles.retrieve_data(name=entity.name)
                    # entity.h_fields = job.resources.bmdca_fields.retrieve_data(name=entity.name)
                    # TODO reinstate entity.j_couplings = job.resources.bmdca_couplings.retrieve_data(name=entity.name)
                if not entity.evolutionary_profile:
                    # to generate in current runtime
                    # entity.add_evolutionary_profile(out_path=job.resources.hhblits_profiles.location)
                    # to generate in a sbatch script
                    # profile_cmds.append(entity.hhblits(out_path=profile_dir, return_command=True))
                    hhblits_cmds.append(entity.hhblits(out_path=profile_dir, return_command=True))
                # if not entity.j_couplings:  # TODO reinstate
                #     bmdca_cmds.append([PUtils.bmdca_exe_path, '-i', os.path.join(profile_dir, '%s.fasta' % entity.name),
                #                        '-d', os.path.join(profile_dir, '%s_bmDCA' % entity.name)])
            if hhblits_cmds:
                if not os.access(PUtils.hhblits_exe, os.X_OK):
                    print(f'Couldn\'t locate the {PUtils.hhblits} executable. Ensure the executable file '
                          f'{PUtils.hhblits_exe} exists then try your job again.')
                    exit()
                # prepare files for running hhblits commands
                instructions = 'Please follow the instructions below to generate sequence profiles for input proteins'
                info_messages.append(instructions)
                # hhblits_cmds, reformat_msa_cmds = zip(*profile_cmds)
                # hhblits_cmds, _ = zip(*hhblits_cmds)
                reformat_msa_cmd1 = [PUtils.reformat_msa_exe_path, 'a3m', 'sto',
                                     '\'%s\'' % os.path.join(profile_dir, '*.a3m'), '.sto', '-num', '-uc']
                reformat_msa_cmd2 = [PUtils.reformat_msa_exe_path, 'a3m', 'fas',
                                     '\'%s\'' % os.path.join(profile_dir, '*.a3m'), '.fasta', '-M', 'first', '-r']
                hhblits_cmd_file = \
                    SDUtils.write_commands(hhblits_cmds, name='%s-hhblits' % SDUtils.starttime, out_path=profile_dir)
                hhblits_sbatch = distribute(file=hhblits_cmd_file, out_path=job.sbatch_scripts, scale='hhblits',
                                            max_jobs=len(hhblits_cmds), number_of_commands=len(hhblits_cmds),
                                            log_file=os.path.join(profile_dir, 'generate_profiles.log'),
                                            finishing_commands=[list2cmdline(reformat_msa_cmd1),
                                                                list2cmdline(reformat_msa_cmd2)])
                hhblits_sbatch_message = \
                    f'Enter the following to distribute {PUtils.hhblits} jobs:\n\tsbatch %s' % hhblits_sbatch
                info_messages.append(hhblits_sbatch_message)
                load_resources = True
            else:
                hhblits_sbatch = None

            if bmdca_cmds:
                # bmdca_cmds = \
                #     [list2cmdline([PUtils.bmdca_exe_path, '-i', os.path.join(profile_dir, '%s.fasta' % entity.name),
                #                   '-d', os.path.join(profile_dir, '%s_bmDCA' % entity.name)])
                #      for entity in all_entities.values()]
                bmdca_cmd_file = \
                    SDUtils.write_commands(bmdca_cmds, name='%s-bmDCA' % SDUtils.starttime, out_path=profile_dir)
                bmdca_sbatch = distribute(file=bmdca_cmd_file, out_path=job.sbatch_scripts, scale='bmdca',
                                          max_jobs=len(bmdca_cmds), number_of_commands=len(bmdca_cmds),
                                          log_file=os.path.join(profile_dir, 'generate_couplings.log'))
                # reformat_msa_cmd_file = \
                #     SDUtils.write_commands(reformat_msa_cmds, name='%s-reformat_msa' % SDUtils.starttime,
                #                            out_path=profile_dir)
                # reformat_sbatch = distribute(file=reformat_msa_cmd_file, out_path=job.program_root,
                #                              scale='script', max_jobs=len(reformat_msa_cmds),
                #                              log_file=os.path.join(profile_dir, 'generate_profiles.log'),
                #                              number_of_commands=len(reformat_msa_cmds))
                print('\n' * 2)
                # Todo add bmdca_sbatch to hhblits_cmds finishing_commands kwarg
                bmdca_sbatch_message = \
                    'Once you are satisfied, enter the following to distribute jobs:\n\tsbatch %s' \
                    % bmdca_sbatch if not load_resources else 'ONCE this job is finished, to calculate evolutionary ' \
                                                              'couplings i,j for each amino acid in the multiple ' \
                                                              'sequence alignment, enter:\n\tsbatch %s' % bmdca_sbatch
                info_messages.append(bmdca_sbatch_message)
                load_resources = True
            else:
                bmdca_sbatch, reformat_sbatch = None, None

            if not args.preprocessed:
                preprocess_instructions, pre_refine, pre_loop_model = \
                    job.resources.preprocess_entities_for_design(all_entities, load_resources=load_resources,
                                                                 script_out_path=job.sbatch_scripts,
                                                                 batch_commands=not args.run_in_shell)
                info_messages += preprocess_instructions

            if load_resources or pre_refine or pre_loop_model:  # entity processing commands are needed
                if info_messages:
                    logger.critical(sbatch_warning)
                    for message in info_messages:
                        logger.info(message)
                    print('\n')
                    logger.info(f'After completion of sbatch script(s), re-run your {PUtils.program_name} command:\n\t'
                                f'python {" ".join(sys.argv)}')
                    terminate(output=False)
                    # After completion of sbatch, the next time initialized, there will be no refine files left allowing
                    # initialization to proceed
                else:
                    raise SDUtils.DesignError('This shouldn\'t have happened!')

            if args.preprocessed:  # ensure we report to PoseDirectory the results after skiping set up
                pre_refine = True
                pre_loop_model = True

        if args.multi_processing:  # and not args.skip_master_db:
            logger.info('Loading Database for multiprocessing fork')
            # Todo set up a job based data acquisition as it takes some time and isn't always necessary!
            job.resources.load_all_data()
            # Todo tweak behavior of these two parameters. Need Queue based PoseDirectory
            # SDUtils.mp_map(PoseDirectory.set_up_design_directory, pose_directories, processes=cores)
            # SDUtils.mp_map(PoseDirectory.link_master_database, pose_directories, processes=cores)
        # set up in series
        for design in pose_directories:
            design.set_up_design_directory(pre_refine=pre_refine, pre_loop_model=pre_loop_model)

        logger.info(f'{len(pose_directories)} unique poses found in "{location}"')
        if not job.debug and not job.skip_logging:
            example_log = getattr(example_directory.log.handlers[0], 'baseFilename', None)
            if example_log:
                logger.info(f'All design specific logs are located in their corresponding directories\n\tEx: '
                            f'{example_log}')

    elif args.module == PUtils.nano:
        logger.critical('Setting up inputs for %s Docking' % PUtils.nano)
        # Todo make current with sql ambitions
        # make master output directory.           sym_entry is required, so this won't fail v
        job.docking_master_dir = os.path.join(job.projects, 'NanohedraEntry%dDockedPoses' % sym_entry.entry_number)
        os.makedirs(job.docking_master_dir, exist_ok=True)
        # Transform input oligomers to canonical orientation and return their ASU
        symmetry_map = sym_entry.groups
        all_entities = []
        load_resources = False
        orient_log = SDUtils.start_log(name='orient', handler=2,
                                       location=os.path.join(job.resources.oriented.location, PUtils.orient_log_file),
                                       propagate=True)
        if args.query_codes:
            if validate_input('Do you want to save the PDB query?', {'y': True, 'n': False}):
                args.save_query = True
            else:
                args.save_query = False
            entities1 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
            entities2 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
        else:
            if args.pdb_codes1:
                entities1 = set(SDUtils.to_iterable(args.pdb_codes1, ensure_file=True))
                # all_entities.extend(job.resources.orient_entities(entities1, symmetry=symmetry_map[0]))
            else:  # args.oligomer1:
                logger.critical('Ensuring provided file(s) at %s are oriented for Nanohedra Docking'
                                % args.oligomer1)
                if '.pdb' in args.oligomer1:
                    pdb1_filepaths = [args.oligomer1]
                else:
                    pdb1_filepaths = SDUtils.get_all_file_paths(args.oligomer1, extension='.pdb')
                # Todo this mechanism conflicts with the one in job.resources. The use of both causes their varioius
                #  nuances to require extensive checks. Ex C1 symmetry, stride file production, asu/oligomer production
                #  fix the divergence of these mechanisms to one single mechanism relying on entities and filepaths
                pdb1_oriented_filepaths = [orient_pdb_file(file, log=orient_log, symmetry=symmetry_map[0],
                                                           out_dir=job.resources.oriented.location)
                                           for file in pdb1_filepaths]
                # pull out the entity names and use job.resources.orient_entities to retrieve the entity alone
                entities1 = list(map(os.path.basename,
                                     [os.path.splitext(file)[0] for file in filter(None, pdb1_oriented_filepaths)]))
                # logger.info('%d filepaths found' % len(pdb1_oriented_filepaths))
                # pdb1_oriented_filepaths = filter(None, pdb1_oriented_filepaths)
        all_entities.extend(job.resources.orient_entities(entities1, symmetry=symmetry_map[0]))

        single_component_design = False
        if args.oligomer2:
            if args.oligomer1 != args.oligomer2:  # see if they are the same input
                logger.critical('Ensuring provided file(s) at %s are oriented for Nanohedra Docking'
                                % args.oligomer1)
                if '.pdb' in args.oligomer2:
                    pdb2_filepaths = [args.oligomer2]
                else:
                    pdb2_filepaths = SDUtils.get_all_file_paths(args.oligomer2, extension='.pdb')
                pdb2_oriented_filepaths = \
                    [orient_pdb_file(file, log=orient_log, symmetry=symmetry_map[1],
                                     out_dir=job.resources.oriented.location)
                     for file in pdb2_filepaths]
                # pull out the entity names and use job.resources.orient_entities to retrieve the entity alone
                entities2 = list(map(os.path.basename,
                                     [os.path.splitext(file)[0] for file in filter(None, pdb2_oriented_filepaths)]))
            else:  # the entities are the same symmetry, or we have single component and bad input
                entities2 = []
        elif args.pdb_codes2:
            # Collect all entities required for processing the given commands
            entities2 = set(SDUtils.to_iterable(args.pdb_codes2, ensure_file=True))
        elif args.query_codes:
            pass
        else:
            entities2 = []
            # if not entities2:
            logger.info('No additional entities requested for docking, treating as single component')
            single_component_design = True
        # Select entities, orient them, then load each entity to all_entities for further database processing
        all_entities.extend(job.resources.orient_entities(entities2, symmetry=symmetry_map[1]))

        info_messages = []
        preprocess_instructions, pre_refine, pre_loop_model = \
            job.resources.preprocess_entities_for_design(all_entities, load_resources=load_resources,
                                                         script_out_path=job.sbatch_scripts,
                                                         batch_commands=not args.run_in_shell)
        if load_resources or pre_refine or pre_loop_model:  # entity processing commands are needed
            logger.critical(sbatch_warning)
            for message in info_messages + preprocess_instructions:
                logger.info(message)
            print('\n')
            logger.info(f'After completion of sbatch script(s), re-run your {PUtils.program_name} command:\n\tpython '
                        f'{" ".join(sys.argv)}')
            terminate(output=False)
            # After completion of sbatch, the next time command is entered docking will proceed

        # make all possible entity_pairs given input entities
        for entity in all_entities:
            entity.make_oligomer(symmetry=entity.symmetry)
        entities1 = [entity for entity in all_entities if entity.name in entities1]
        entities2 = [entity for entity in all_entities if entity.name in entities2]
        entity_pairs = list(product(entities1, entities2))
        location = args.oligomer1
        design_source = os.path.splitext(os.path.basename(location))[0]
    else:
        # this logic is possible with args.module in
        # PUtils.nano, 'multicistronic', or select_poses with --metric or --dataframe
        # job.resources = None
        # design_source = os.path.basename(example_directory.project_designs)
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # Set up Job specific details and resources
    # -----------------------------------------------------------------------------------------------------------------
    # Format computational requirements
    if args.module in [PUtils.nano, PUtils.interface_design]:  # Todo make possible for all
        if args.run_in_shell:
            logger.info('Modeling will occur in this process, ensure you don\'t lose connection to the shell!')
        else:
            logger.info('Writing modeling commands out to file, no modeling will occur until commands are executed')
    if args.multi_processing:
        # Calculate the number of cores to use depending on computer resources
        cores = SDUtils.calculate_mp_cores(cores=args.cores)  # mpi=args.mpi, Todo
        logger.info('Starting multiprocessing using %d cores' % cores)
    else:
        cores = 1
        logger.info('Starting processing. If single process is taking awhile, use --multi_processing during submission')

    # Format memory requirements with module dependencies
    if args.module == PUtils.nano:  # Todo
        required_memory = PUtils.baseline_program_memory + PUtils.nanohedra_memory  # 30 GB ?
    elif args.module == PUtils.analysis:
        required_memory = (PUtils.baseline_program_memory +
                           len(pose_directories) * PUtils.approx_ave_design_directory_memory_w_assembly) * 1.2
    else:
        required_memory = (PUtils.baseline_program_memory +
                           len(pose_directories) * PUtils.approx_ave_design_directory_memory_w_pose) * 1.2

    job.reduce_memory = True if psutil.virtual_memory().available < required_memory else False
    # logger.info('Available: %f' % psutil.virtual_memory().available)
    # logger.info('Requried: %f' % required_memory)
    # logger.info('Reduce Memory?: %s', job.reduce_memory)

    # Run specific checks
    if args.module == PUtils.interface_design and not queried_flags[PUtils.no_evolution_constraint]:  # hhblits to run
        if psutil.virtual_memory().available <= required_memory + hhblits_memory_threshold:
            logger.critical('The amount of memory for the computer is insufficient to run hhblits (required for '
                            'designing with evolution)! Please allocate the job to a computer with more memory or the '
                            'process will fail. Otherwise, select --%s' % PUtils.no_evolution_constraint)
            exit(1)
    # -----------------------------------------------------------------------------------------------------------------
    # Parse SubModule specific commands
    # -----------------------------------------------------------------------------------------------------------------
    results, success, exceptions = [], [], []
    # ---------------------------------------------------
    if args.module == 'orient':
        args.to_design_directory = True  # default to True when using this module
        if args.multi_processing:
            zipped_args = zip(pose_directories, repeat(args.to_design_directory))
            results = SDUtils.mp_starmap(PoseDirectory.orient, zipped_args, processes=cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.orient(to_design_directory=args.to_design_directory))

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'find_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.find_asu, pose_directories, processes=cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.find_asu())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'expand_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.expand_asu, pose_directories, processes=cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.expand_asu())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'rename_chains':
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.rename_chains, pose_directories, processes=cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.rename_chains())

        terminate(results=results)
    # ---------------------------------------------------
    # elif args.module == 'check_unmodelled_clashes':  # Todo
    #     if args.multi_processing:
    #         results = SDUtils.mp_map(PoseDirectory.check_unmodelled_clashes, pose_directories, processes=cores)
    #     else:
    #         for design_dir in pose_directories:
    #             results.append(design_dir.check_unmodelled_clashes())
    #
    #     terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'check_clashes':
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.check_clashes, pose_directories, processes=cores)
        else:
            for design_dir in pose_directories:
                results.append(design_dir.check_clashes())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.generate_fragments:
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.generate_interface_fragments, pose_directories, processes=cores)
        else:
            for design in pose_directories:
                results.append(design.generate_interface_fragments())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.nano:  # -o1 oligomer1, -o2 oligomer2, -e entry, -o outdir
        # Initialize docking procedure
        if args.run_in_shell:
            if args.debug:
                # Root logs to stream with level debug according to prior logging initialization
                master_logger, bb_logger = logger, logger
            else:
                master_log_filepath = os.path.join(args.output_directory, PUtils.master_log)
                master_logger = SDUtils.start_log(name=PUtils.nano.title(), handler=2, location=master_log_filepath,
                                                  propagate=True, no_log_name=True)
                bb_logger = None  # have to include this incase started as debug
            master_logger.info('Nanohedra\nMODE: DOCK\n\n')
            write_docking_parameters(args.oligomer1, args.oligomer2, args.rotation_step1, args.rotation_step2,
                                     sym_entry, job.docking_master_dir, log=master_logger)
            if args.multi_processing:
                zipped_args = zip(repeat(sym_entry), repeat(fragment_db), repeat(euler_lookup),
                                  repeat(job.docking_master_dir), *zip(*entity_pairs),
                                  repeat(args.rotation_step1), repeat(args.rotation_step2), repeat(args.min_matched),
                                  repeat(args.high_quality_match_value), repeat(args.initial_z_value),
                                  repeat(args.output_assembly), repeat(args.output_surrounding_uc), repeat(bb_logger))
                results = SDUtils.mp_starmap(nanohedra_dock, zipped_args, processes=cores)
            else:  # using combinations of directories with .pdb files
                for entity1, entity2 in entity_pairs:
                    master_logger.info('Docking %s / %s' % (entity1.name, entity1.name))
                    # result = nanohedra_dock(sym_entry, fragment_db, euler_lookup, job.docking_master_dir, pdb1, pdb2,
                    # result = None
                    nanohedra_dock(sym_entry, fragment_db, euler_lookup, job.docking_master_dir, entity1, entity2,
                                   rotation_step1=args.rotation_step1, rotation_step2=args.rotation_step2,
                                   min_matched=args.min_matched, high_quality_match_value=args.high_quality_match_value,
                                   initial_z_value=args.initial_z_value, output_assembly=args.output_assembly,
                                   output_surrounding_uc=args.output_surrounding_uc, log=bb_logger)
                    # results.append(result)  # DONT need. Results uses pose_directories. There are none and no output
            terminate(results=results, output=False)
        else:  # write all commands to a file and use sbatch
            design_source = 'Entry%d' % sym_entry.entry_numbe  # used for terminate()
            # script_out_dir = os.path.join(job.docking_master_dir, PUtils.scripts)
            # os.makedirs(script_out_dir, exist_ok=True)
            cmd = ['python', PUtils.nanohedra_dock_file, '-dock']
            kwargs = dict(outdir=job.docking_master_dir, entry=sym_entry.entry_number, rot_step1=args.rotation_step1,
                          rot_step2=args.rotation_step2, min_matcher=args.min_matched,
                          high_quality_match_value=args.high_quality_match_value,
                          initial_z_value=args.initial_z_value, output_assembly=args.output_assembly,
                          output_surrounding_uc=args.output_surrounding_uc)
            cmd.extend(chain.from_iterable([['-%s' % key, str(value)] for key, value in kwargs.items()]))
            commands = [cmd + [PUtils.nano_entity_flag1, entity1.name, PUtils.nano_entity_flag2, entity2.name] +
                        (['-initial'] if idx == 0 else []) for idx, (entity1, entity2) in enumerate(entity_pairs)]
            terminate(results=commands)
        # # Make single file with names of each directory. Specific for docking due to no established directory
        # args.file = os.path.join(args.directory, 'all_docked_directories.paths')
        # with open(args.file, 'w') as design_f:
        #     command_directories = map(os.path.dirname, results)  # get only the directory of the command
        #     design_f.write('\n'.join(docking_pair for docking_pair in command_directories if docking_pair))

        # all_commands = [result for result in results if result]
        # if len(all_commands) > 0:
        #     command_file = SDUtils.write_commands(all_commands, name=PUtils.nano, out_path=args.directory)
        #     args.success_file = None
        #     args.failure_file = None
        #     args.max_jobs = 80
        #     distribute(stage=PUtils.nano, directory=args.directory, file=command_file,
        #                success_file=args.success_file, failure_file=args.success_file, max_jobs=args.max_jobs)
        #     logger.info('All "%s" commands were written to "%s"' % (PUtils.nano, command_file))
        # else:
        #     logger.error('No "%s" commands were written!' % PUtils.nano)
    # ---------------------------------------------------
    elif args.module == PUtils.interface_metrics:
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # zipped_args = zip(pose_directories, repeat(args.force_flags), repeat(queried_flags.get('development')))
            results = SDUtils.mp_map(PoseDirectory.interface_metrics, pose_directories, processes=cores)
        else:
            for design in pose_directories:
                # if design.sym_entry is None:
                #     continue
                results.append(design.interface_metrics())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.optimize_designs:
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # zipped_args = zip(pose_directories, repeat(args.force_flags), repeat(queried_flags.get('development')))
            results = SDUtils.mp_map(PoseDirectory.optimize_designs, pose_directories, processes=cores)
        else:
            for design in pose_directories:
                results.append(design.optimize_designs())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'custom_script':
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            zipped_args = zip(pose_directories, repeat(args.script), repeat(args.force_flags),
                              repeat(args.file_list), repeat(args.native), repeat(args.suffix), repeat(args.score_only),
                              repeat(args.variables))
            results = SDUtils.mp_starmap(PoseDirectory.custom_rosetta_script, zipped_args, processes=cores)
        else:
            for design in pose_directories:
                results.append(design.custom_rosetta_script(args.script, force_flags=args.force_flags,
                                                            file_list=args.file_list, native=args.native,
                                                            suffix=args.suffix, score_only=args.score_only,
                                                            variables=args.variables))

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.refine:  # -i fragment_library, -s scout
        args.to_design_directory = True  # always the case when using this module
        if args.multi_processing:
            zipped_args = zip(pose_directories, repeat(args.to_design_directory), repeat(args.interface_to_alanine),
                              repeat(args.gather_metrics))
            results = SDUtils.mp_starmap(PoseDirectory.refine, zipped_args, processes=cores)
        else:
            for design in pose_directories:
                results.append(design.refine(to_design_directory=args.to_design_directory,
                                             interface_to_alanine=args.interface_to_alanine,
                                             gather_metrics=args.gather_metrics))

        terminate(results=results, output=True)
    # ---------------------------------------------------
    elif args.module == PUtils.interface_design:  # -i fragment_library, -s scout
        # if args.mpi:  # Todo implement
        #     # extras = ' mpi %d' % CommmandDistributer.mpi
        #     logger.info(
        #         'Setting job up for submission to MPI capable computer. Pose trajectories run in parallel,'
        #         ' %s at a time. This will speed up pose processing ~%f-fold.' %
        #         (CommmandDistributer.mpi - 1, PUtils.nstruct / (CommmandDistributer.mpi - 1)))
        #     queried_flags.update({'mpi': True, 'script': True})
        if not queried_flags[PUtils.no_evolution_constraint]:  # hhblits to run
            job.make_path(job.sequences)
            job.make_path(job.profiles)
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = SDUtils.mp_map(PoseDirectory.interface_design, pose_directories, processes=cores)
        else:
            for design in pose_directories:
                results.append(design.interface_design())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.analysis:  # -o output, -f figures, -n no_save, -j join
        if args.no_save:
            args.save = False
        else:
            args.save = True
        # job = next(iter(pose_directories))
        # ensure analysis write directory exists
        job.make_path(job.all_scores)
        # Start pose analysis of all designed files
        if args.output_file == PUtils.analysis_file:
            args.output_file = os.path.join(job.all_scores, args.output_file % (SDUtils.starttime, design_source))
        elif len(args.output_file.split(os.sep)) <= 1:  # the path isn't an absolute or relative path, prepend location
            args.output_file = os.path.join(job.all_scores, args.output_file)

        if args.multi_processing:
            zipped_args = zip(pose_directories, repeat(args.join), repeat(args.save), repeat(args.figures))
            results = SDUtils.mp_starmap(PoseDirectory.interface_design_analysis, zipped_args, processes=cores)
        else:
            # @profile  # memory_profiler
            # def run_single_analysis():
            for design in pose_directories:
                results.append(design.interface_design_analysis(merge_residue_data=args.join,
                                                                save_trajectories=args.save, figures=args.figures))
            # run_single_analysis()
        terminate(results=results)
    # ---------------------------------------------------
    # elif args.module == 'merge':  # -d2 directory2, -f2 file2, -i increment, -F force
    #     directory_pairs, failures = None, None
    #     if args.directory2 or args.file2:
    #         # Grab all poses (directories) to be processed from either directory name or file
    #         all_poses2, location2 = SDUtils.collect_designs(files=args.file2, directory=args.directory2)
    #         assert all_poses2 != list(), logger.critical(
    #             'No %s.py directories found within "%s"! Please ensure correct location' % (PUtils.nano.title(),
    #                                                                                           location2))
    #         all_design_directories2 = set_up_directory_objects(all_poses2)
    #         logger.info('%d Poses found in "%s"' % (len(all_poses2), location2))
    #         if args.merge_mode == PUtils.interface_design:
    #             directory_pairs, failures = pair_directories(all_design_directories2, pose_directories)
    #         else:
    #             logger.warning('Source location was specified, but the --directory_type isn\'t design. Destination '
    #                            'directory will be ignored')
    #     else:
    #         if args.merge_mode == PUtils.interface_design:
    #             exit('No source location was specified! Use -d2 or -f2 to specify the source of poses when merging '
    #                  'design directories')
    #         elif args.merge_mode == PUtils.nano:
    #             directory_pairs, failures = pair_dock_directories(pose_directories)  #  all_dock_directories)
    #             for pair in directory_pairs:
    #                 if args.force:
    #                     merge_docking_pair(pair, force=True)
    #                 else:
    #                     try:
    #                         merge_docking_pair(pair)
    #                     except FileExistsError:
    #                         logger.info('%s directory already exits, moving on. Use --force to overwrite.' % pair[1])
    #
    #     if failures:
    #         logger.warning('The following directories have no partner:\n\t%s' % '\n\t'.join(fail.path
    #                                                                                         for fail in failures))
    #     if args.multi_processing:
    #         zipped_args = zip(pose_directories, repeat(args.increment))
    #         results = SDUtils.mp_starmap(rename, zipped_args, processes=cores)
    #         results2 = SDUtils.mp_map(merge_design_pair, directory_pairs, processes=cores)
    #     else:
    #         for des_directory in pose_directories:
    #             rename(des_directory, increment=args.increment)
    #         for directory_pair in directory_pairs:
    #             merge_design_pair(directory_pair)
    # ---------------------------------------------------
    elif args.module == PUtils.select_poses:
        if args.total:  # Figure out poses from directory specification, filters, and weights
            df = load_total_dataframe(pose=True)
            if args.protocol:  # Todo adapt to protocol column not in Trajectories right now...
                group_df = df.groupby('protocol')
                df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                               keys=list(zip(group_df.groups, repeat('mean'))))
            else:
                df = pd.concat([df], axis=1, keys=['pose', 'metric'])
            # Figure out designs from dataframe, filters, and weights
            selected_poses_df = prioritize_design_indices(df, filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            # remove excess pose instances
            number_chosen = 0
            selected_indices, selected_poses = [], set()
            for design_directory, design in selected_poses_df.index.to_list():
                if design_directory not in selected_poses:
                    selected_poses.add(design_directory)
                    selected_indices.append((design_directory, design))
                    number_chosen += 1
                    if number_chosen == args.select_number:
                        break

            # drop the specific design for the dataframe. If they want the design, they should run select designs
            save_poses_df = \
                selected_poses_df.loc[selected_indices, :].droplevel(-1).droplevel(0, axis=1).droplevel(0, axis=1)
        elif args.specification_file:  # Figure out poses from a specification file, filters, and weights
            indices = [(design_directory, design_directory.specific_design) for design_directory in pose_directories]
            df = load_total_dataframe(pose=True)
            selected_poses_df = prioritize_design_indices(df.loc[indices, :], filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            # remove excess pose instances
            number_chosen = 0
            selected_indices, selected_poses = [], set()
            for design_directory, design in selected_poses_df.index.to_list():
                if design_directory not in selected_poses:
                    selected_poses.add(design_directory)
                    selected_indices.append((design_directory, design))
                    number_chosen += 1
                    if number_chosen == args.select_number:
                        break

            # specify the result order according to any filtering and weighting
            # drop the specific design for the dataframe. If they want the design, they should run select designs
            save_poses_df = \
                selected_poses_df.loc[selected_indices, :].droplevel(-1).droplevel(0, axis=1).droplevel(0, axis=1)
        elif args.dataframe:  # Figure out poses from a pose dataframe, filters, and weights
            # program_root = next(iter(pose_directories)).program_root
            if args.dataframe and not pose_directories:  # not args.directory:
                logger.critical('If using a --dataframe for selection, you must include the directory where the designs'
                                ' are located in order to properly select designs. Please specify -d/--directory on the'
                                ' command line')
                exit(1)
            program_root = job.program_root
            df = pd.read_csv(args.dataframe, index_col=0, header=[0, 1, 2])
            df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)
            selected_poses_df = prioritize_design_indices(df, filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            # only drop excess columns as there is no MultiIndex, so no design in the index
            save_poses_df = selected_poses_df.droplevel(0, axis=1).droplevel(0, axis=1)
            selected_poses = [PoseDirectory.from_pose_id(pose, root=program_root, **queried_flags)
                              for pose in save_poses_df.index.to_list()]
        else:  # generate design metrics on the spot
            selected_poses, selected_poses_df, df = [], pd.DataFrame(), pd.DataFrame()
            logger.debug('Collecting designs to sort')
            if args.metric == 'score':
                metric_design_dir_pairs = [(des_dir.score, des_dir.path) for des_dir in pose_directories]
            elif args.metric == 'fragments_matched':
                metric_design_dir_pairs = [(des_dir.number_of_fragments, des_dir.path)
                                           for des_dir in pose_directories]
            # else:
            #     raise SDUtils.DesignError('The metric "%s" is not supported!' % args.metric)

            logger.debug('Sorting designs according to "%s"' % args.metric)
            metric_design_dir_pairs = [(score, path) for score, path in metric_design_dir_pairs if score]
            sorted_metric_design_dir_pairs = sorted(metric_design_dir_pairs, key=lambda pair: (pair[0] or 0),
                                                    reverse=True)
            top_designs_string = 'Top ranked Designs according to %s:\n\t%s\tDesign\n\t%s'\
                                 % (args.metric, args.metric.title(), '%s')
            results_strings = ['%.2f\t%s' % tup for tup in sorted_metric_design_dir_pairs]
            logger.info(top_designs_string % '\n\t'.join(results_strings[:500]))
            if len(pose_directories) > 500:
                design_source = 'top_%s' % args.metric
                default_output_tuple = (SDUtils.starttime, args.module, design_source)
                designs_file = os.path.join(job.job_paths, '%s_%s_%s_pose.scores' % default_output_tuple)
                with open(designs_file, 'w') as f:
                    f.write(top_designs_string % '\n\t'.join(results_strings))
                logger.info(f'Stdout performed a cutoff of ranked Designs at ranking 500. See the output design file '
                            f'"{designs_file}" for the remainder')

            terminate(output=False)
        # else:
        #     logger.critical('Missing a required method to provide or find metrics from %s. If you meant to gather '
        #                     'metrics from every pose in your input specification, ensure you include the --global '
        #                     'argument' % PUtils.program_output)
        #     exit()

        if args.filter or args.weight:
            new_dataframe = os.path.join(program_root, '%s-%s%sDesignPoseMetrics.csv'
                                         % (SDUtils.starttime, 'Filtered' if args.filter else '',
                                            'Weighted' if args.weight else ''))
        else:
            new_dataframe = os.path.join(program_root, '%s-DesignPoseMetrics.csv' % SDUtils.starttime)

        logger.info('%d poses were selected' % len(selected_poses_df))  # :\n\t%s , '\n\t'.join(selected_poses)))
        if len(selected_poses_df) != len(df):
            selected_poses_df.to_csv(new_dataframe)
            logger.info('Newly selected DataFrame was written to %s' % new_dataframe)

        # Sort results according to clustered poses if clustering exists
        if args.cluster_map:
            cluster_map = args.cluster_map
        else:  # Todo PUtils.clustered_poses is not right...
            cluster_map = os.path.join(job.protein_data, '%s.pkl' % PUtils.clustered_poses)

        if os.path.exists(cluster_map):
            pose_cluster_map = SDUtils.unpickle(cluster_map)
        else:  # try to generate the cluster_map
            logger.info('No cluster pose map was found at %s. Clustering similar poses may eliminate redundancy '
                        'from the final design selection. To cluster poses broadly, run "%s %s"'
                        % (cluster_map, PUtils.program_command, PUtils.cluster_poses))
            while True:
                confirm = input('Would you like to %s on the subset of designs (%d) located so far? [y/n]%s'
                                % (PUtils.cluster_poses, len(selected_poses), input_string))
                if confirm.lower() in bool_d:
                    break
                else:
                    print('%s %s is not a valid choice!' % (invalid_string, confirm))
            if bool_d[confirm.lower()] or confirm.isspace():  # the user wants to separate poses
                compositions = group_compositions(selected_poses)
                if args.multi_processing:
                    mp_results = SDUtils.mp_map(cluster_designs, compositions.values(), processes=cores)
                    pose_cluster_map = {}
                    for result in mp_results:
                        pose_cluster_map.update(result.items())
                else:
                    pose_cluster_map = {}
                    for composition_group in compositions.values():
                        pose_cluster_map.update(cluster_designs(composition_group))

                pose_cluster_file = SDUtils.pickle_object(pose_cluster_map,
                                                          PUtils.clustered_poses % (location, SDUtils.starttime),
                                                          out_path=next(iter(pose_directories)).protein_data)
                logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                            % (len(pose_cluster_map), len(pose_directories), pose_cluster_file))
            else:
                pose_cluster_map = {}

        if pose_cluster_map:
            # {design_string: [design_string, ...]} where key is representative, values are matching designs
            # OLD -> {composition: {design_string: cluster_representative}, ...}
            pose_cluster_membership_map = invert_cluster_map(pose_cluster_map)
            pose_clusters_found, pose_not_found = {}, []
            # convert all of the selected poses to their string representation
            for idx, pose_directory in enumerate(map(str, selected_poses)):
                cluster_membership = pose_cluster_membership_map.get(pose_directory, None)
                if cluster_membership:
                    if cluster_membership not in pose_clusters_found:  # include as this pose hasn't been identified
                        pose_clusters_found[cluster_membership] = [pose_directory]
                    else:  # This cluster has already been found and it was identified again. Report and only
                        # include the highest ranked pose in the output as it provides info on all occurrences
                        pose_clusters_found[cluster_membership].append(pose_directory)
                else:
                    pose_not_found.append(pose_directory)

            # Todo report the clusters and the number of instances
            final_poses = [members[0] for members in pose_clusters_found.values()]
            if pose_not_found:
                logger.warning('Couldn\'t locate the following poses:\n\t%s\nWas %s only run on a subset of the '
                               'poses that were selected? Adding all of these to your final poses...'
                               % ('\n\t'.join(pose_not_found), PUtils.cluster_poses))
                final_poses.extend(pose_not_found)
            logger.info('Found %d poses after clustering' % len(final_poses))
        else:
            logger.info('Grabbing all selected poses')
            final_poses = selected_poses

        if len(final_poses) > args.select_number:
            final_poses = final_poses[:args.select_number]
            logger.info('Found %d poses after applying your select_number selection criteria' % len(final_poses))

        # Need to initialize pose_directories to terminate()
        pose_directories = final_poses
        design_source = program_root  # for terminate()
        # write out the chosen poses to a pose.paths file
        terminate(results=pose_directories)
    # ---------------------------------------------------
    elif args.module == PUtils.cluster_poses:
        pose_cluster_map = {}
        if args.mode == 'ialign':  # interface_residues, tranformation
            is_threshold = 0.4  # 0.5  # TODO
            # measure the alignment of all selected pose_directories
            # all_files = [design.source_file for design in pose_directories]

            # need to change directories to prevent issues with the path length being passed to ialign
            prior_directory = os.getcwd()
            os.chdir(job.protein_data)  # os.path.join(job.protein_data, 'ialign_output'))
            temp_file_dir = os.path.join(os.getcwd(), 'temp')
            if not os.path.exists(temp_file_dir):
                os.makedirs(temp_file_dir)

            # save the interface for each design to the temp directory
            design_interfaces = []
            for design in pose_directories:
                design.identify_interface()  # calls design.load_pose()
                interface = design.pose.return_interface()
                design_interfaces.append(
                    # interface.write(out_path=os.path.join(temp_file_dir, '%s_interface.pdb' % design.name)))  # Todo reinstate
                    interface.write(out_path=os.path.join(temp_file_dir, '%s.pdb' % design.name)))

            design_directory_pairs = list(combinations(pose_directories, 2))
            design_pairs = []
            if args.multi_processing:
                # zipped_args = zip(combinations(design_interfaces, 2))
                design_scores = SDUtils.mp_starmap(ialign, combinations(design_interfaces, 2), processes=cores)

                for idx, is_score in enumerate(design_scores):
                    if is_score > is_threshold:
                        design_pairs.append(set(design_directory_pairs[idx]))
            else:
                # for design1, design2 in combinations(pose_directories, 2):  # all_files
                for idx, (interface_file1, interface_file2) in enumerate(combinations(design_interfaces, 2)):  # all_files
                    # is_score = ialign(design1.source, design2.source, out_path='ialign')
                    is_score = ialign(interface_file1, interface_file2)
                    #                   out_path=os.path.join(job.protein_data, 'ialign_output'))
                    if is_score > is_threshold:
                        design_pairs.append(set(design_directory_pairs[idx]))
                        # design_pairs.append({design1, design2})
            # now return to prior directory
            os.chdir(prior_directory)

            # cluster all those designs together that are in alignment
            if design_pairs:
                for design1, design2 in design_pairs:
                    cluster1, cluster2 = pose_cluster_map.get(design1), pose_cluster_map.get(design2)
                    # if cluster1:
                    #     cluster1.append(design2)
                    # else:
                    #     design_clusters[design1] = [design2]
                    try:
                        cluster1.append(design2)
                    except AttributeError:
                        pose_cluster_map[design1] = [design2]
                    try:
                        cluster2.append(design1)
                    except AttributeError:
                        pose_cluster_map[design2] = [design1]
        elif args.mode == 'transform':
            # First, identify the same compositions
            compositions = group_compositions(pose_directories)
            if args.multi_processing:
                results = SDUtils.mp_map(cluster_designs, compositions.values(), processes=cores)
                for result in results:
                    pose_cluster_map.update(result.items())
            else:
                # pose_map = pose_rmsd_s(pose_directories)
                # pose_cluster_map = cluster_poses(pose_map)
                for composition_group in compositions.values():
                    pose_cluster_map.update(cluster_designs(composition_group))
        else:
            exit('%s is not a viable mode!' % args.mode)

        if pose_cluster_map:
            if args.output_file:
                pose_cluster_file = SDUtils.pickle_object(pose_cluster_map, args.output_file, out_path='')
            else:
                pose_cluster_file = SDUtils.pickle_object(pose_cluster_map,
                                                          PUtils.clustered_poses % (location, SDUtils.starttime),
                                                          out_path=job.clustered_poses)
            logger.info('Cluster map written to %s' % pose_cluster_file)
        else:
            logger.info('No significant clusters were located! Clustering ended')

        terminate(results=pose_cluster_map)
    # --------------------------------------------------- # TODO v move to AnalyzeMutatedSequence.py
    elif args.module == PUtils.select_sequences:  # -p protocol, -f filters, -w weights, -ns number_sequences
        program_root = job.program_root
        if args.total:
            df = load_total_dataframe()
            if args.protocol:
                group_df = df.groupby('protocol')
                df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                               keys=list(zip(group_df.groups, repeat('mean'))))
            else:
                df = pd.concat([df], axis=1, keys=['pose', 'metric'])
            # Figure out designs from dataframe, filters, and weights
            selected_poses_df = prioritize_design_indices(df, filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            selected_designs = selected_poses_df.index.to_list()
            if args.allow_multiple_poses:
                logger.info('Choosing maximum %d designs as specified, from the top ranked designs regardless of pose'
                            % args.select_number)
                results = selected_designs[:args.select_number]
            else:
                logger.info('Choosing maximum %d designs as specified, with only one design allowed per pose'
                            % args.select_number)
                number_chosen = 0
                results, selected_poses = [], set()
                for design_directory, design in selected_designs:
                    if design_directory not in selected_poses:
                        selected_poses.add(design_directory)
                        results.append((design_directory, design))
                        number_chosen += 1
                        if number_chosen == args.select_number:
                            break
            logger.info('%d designs were selected' % len(results))

            if args.filter or args.weight:
                new_dataframe = os.path.join(program_root, '%s-%s%sDesignPoseMetrics.csv'
                                             % (SDUtils.starttime, 'Filtered' if args.filter else '',
                                                'Weighted' if args.weight else ''))
            else:
                new_dataframe = os.path.join(program_root, '%s-DesignPoseMetrics.csv' % SDUtils.starttime)
            # include only the found index names to the saved dataframe
            save_poses_df = selected_poses_df.loc[results, :].droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        elif args.specification_file:
            results = [(design_directory, design_directory.specific_design) for design_directory in pose_directories]
            df = load_total_dataframe()
            selected_poses_df = prioritize_design_indices(df.loc[results, :], filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            # specify the result order according to any filtering and weighting
            # results = selected_poses_df.index.to_list()  TODO reinstate
            save_poses_df = selected_poses_df.droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        else:  # select designed sequences from each pose provided (PoseDirectory)
            trajectory_df = None  # currently used to get the column headers
            if args.filter:
                trajectory_df = pd.read_csv(job.trajectories, index_col=0, header=[0])
                sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
                sequence_filters = query_user_for_metrics(sequence_metrics, mode='filter', level='sequence')
            else:
                sequence_filters = None

            if args.weight:
                if not trajectory_df:
                    trajectory_df = pd.read_csv(job.trajectories, index_col=0, header=[0])
                    sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
                sequence_weights = query_user_for_metrics(sequence_metrics, mode='weight', level='sequence')
            else:
                sequence_weights = None

            if args.multi_processing:
                # sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
                #                     'int_energy_res_summary_delta': 0.25}
                zipped_args = zip(pose_directories, repeat(sequence_filters), repeat(sequence_weights),
                                  repeat(args.select_number), repeat(args.protocol))
                # result_mp = zip(*SDUtils.mp_starmap(Ams.select_sequences, zipped_args, processes=cores))
                result_mp = SDUtils.mp_starmap(PoseDirectory.select_sequences, zipped_args, processes=cores)
                # results - contains tuple of (PoseDirectory, design index) for each sequence
                # could simply return the design index then zip with the directory
                results = {design: results for design, result in zip(pose_directories, result_mp)}
            else:
                results = {design: design.select_sequences(filters=sequence_filters, weights=sequence_weights,
                                                           number=args.select_number, protocols=args.protocol)
                           for design in pose_directories}
            save_poses_df = None  # Todo make possible!

        # Format selected sequences for output
        if not args.selection_string:
            args.selection_string = '%s_' % os.path.basename(os.path.splitext(location)[0])
        else:
            args.selection_string += '_'
        outdir = os.path.join(os.path.dirname(program_root), '%sSelectedDesigns' % args.selection_string)
        # outdir_traj, outdir_res = os.path.join(outdir, 'Trajectories'), os.path.join(outdir, 'Residues')
        os.makedirs(outdir, exist_ok=True)  # , os.makedirs(outdir_traj), os.makedirs(outdir_res)

        if save_poses_df is not None:
            selection_trajectory_df_file = os.path.join(outdir, 'TrajectoryMetrics.csv')
            save_poses_df.to_csv(selection_trajectory_df_file)
            logger.info('New DataFrame with selected designs was written to %s' % selection_trajectory_df_file)

        logger.info('Relevant design files are being copied to the new directory: %s' % outdir)
        # Create new output of designed PDB's  # TODO attach the state to these files somehow for further SymDesign use
        for des_dir, design in results:
            file_path = os.path.join(des_dir.designs, '*%s*' % design)
            file = sorted(glob(file_path))
            if not file:  # add to exceptions
                exceptions.append((des_dir.path, 'No file found for "%s"' % file_path))
                continue
            if not os.path.exists(os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design))):
                shutil.copy(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
                # shutil.copy(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
                # shutil.copy(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # try:
            #     # Create symbolic links to the output PDB's
            #     os.symlink(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
            #     os.symlink(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
            #     os.symlink(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # except FileExistsError:
            #     pass

        # Format sequences for expression
        args.output_file = os.path.join(outdir, '%sSelectedDesigns.paths' % args.selection_string)
        pose_directories = [des_dir for des_dir, design in results]
        if args.skip_sequence_generation:
            terminate(output=False)
        else:
            with open(args.output_file, 'w') as f:
                f.write('%s\n' % '\n'.join(des_dir.path for des_dir in pose_directories))

        # use one directory as indication of entity specification for them all. Todo modify for different length inputs
        example_directory.load_pose()
        if args.tag_entities:
            if args.tag_entities == 'all':
                tag_index = [True for _ in example_directory.pose.entities]
                number_of_tags = len(example_directory.pose.entities)
            elif args.tag_entities == 'single':
                tag_index = [True for _ in example_directory.pose.entities]
                number_of_tags = 1
            elif args.tag_entities == 'none':
                tag_index = [False for _ in example_directory.pose.entities]
                number_of_tags = None
            else:
                tag_specified_list = list(map(str.translate, set(args.entity_specification.split(',')).difference(['']),
                                              repeat(SDUtils.digit_translate_table)))
                for idx, item in enumerate(tag_specified_list):
                    try:
                        tag_specified_list[idx] = int(item)
                    except ValueError:
                        continue

                for _ in range(len(example_directory.pose.entities) - len(tag_specified_list)):
                    tag_specified_list.append(0)
                tag_index = [True if is_tag else False for is_tag in tag_specified_list]
                number_of_tags = sum(tag_specified_list)
        else:
            tag_index = [False for _ in example_directory.pose.entities]
            number_of_tags = None

        if args.multicistronic or args.multicistronic_intergenic_sequence:
            args.multicistronic = True
            if args.multicistronic_intergenic_sequence:
                intergenic_sequence = args.multicistronic_intergenic_sequence
            else:
                intergenic_sequence = default_multicistronic_sequence
        else:
            intergenic_sequence = ''

        missing_tags = {}  # result: [True, True] for result in results
        tag_sequences, final_sequences, inserted_sequences, nucleotide_sequences = {}, {}, {}, {}
        codon_optimization_errors = {}
        # for des_dir, design in results:
        for des_dir, designs in results.items():
            des_dir.load_pose(source=des_dir.asu_path)
            des_dir.pose.pdb.reorder_chains()  # Do I need to modify chains?
            for design in designs:
                file_glob = '%s%s*%s*' % (des_dir.designs, os.sep, design)
                file = sorted(glob(file_glob))
                if not file:
                    logger.error('No file found for %s' % file_glob)
                    continue
                design_pose = PDB.from_file(file[0], log=des_dir.log, entity_names=des_dir.entity_names)
                designed_atom_sequences = [entity.structure_sequence for entity in design_pose.entities]

                missing_tags[(des_dir, design)] = [1 for _ in des_dir.pose.entities]
                prior_offset = 0
                # all_missing_residues = {}
                # mutations = []
                # referenced_design_sequences = {}
                sequences_and_tags = {}
                entity_termini_availability, entity_helical_termini = {}, {}
                for idx, (source_entity, design_entity) in enumerate(zip(des_dir.pose.entities, design_pose.entities)):
                    # source_entity.retrieve_info_from_api()
                    # source_entity.reference_sequence
                    sequence_id = '%s_%s' % (des_dir, source_entity.name)
                    # design_string = '%s_design_%s_%s' % (des_dir, design, source_entity.name)  # [i])), pdb_code)
                    design_string = '%s_%s' % (design, source_entity.name)
                    uniprot_id = source_entity.uniprot_id
                    termini_availability = des_dir.return_termini_accessibility(source_entity)
                    logger.debug('Design %s has the following termini accessible for tags: %s'
                                 % (sequence_id, termini_availability))
                    if args.avoid_tagging_helices:
                        termini_helix_availability = des_dir.return_termini_accessibility(source_entity, report_if_helix=True)
                        logger.debug('Design %s has the following helical termini available: %s'
                                     % (sequence_id, termini_helix_availability))
                        termini_availability = {'n': termini_availability['n'] and not termini_helix_availability['n'],
                                                'c': termini_availability['c'] and not termini_helix_availability['c']}
                        entity_helical_termini[design_string] = termini_helix_availability
                    logger.debug('The termini %s are available for tagging' % termini_availability)
                    entity_termini_availability[design_string] = termini_availability
                    true_termini = [term for term, is_true in termini_availability.items() if is_true]

                    # Find sequence specified attributes required for expression formatting
                    # disorder = generate_mutations(source_entity.structure_sequence, source_entity.reference_sequence,
                    #                               only_gaps=True)
                    # disorder = source_entity.disorder
                    source_offset = source_entity.offset
                    indexed_disordered_residues = {res_number + source_offset + prior_offset: mutation
                                                   for res_number, mutation in source_entity.disorder.items()}
                    # Todo, moved below indexed_disordered_residues on 7/26, ensure correct!
                    prior_offset += len(indexed_disordered_residues)
                    # generate the source TO design mutations before any disorder handling
                    mutations = generate_mutations(source_entity.structure_sequence, design_entity.structure_sequence,
                                                   offset=False)
                    # Insert the disordered residues into the design pose
                    for residue_number, mutation in indexed_disordered_residues.items():
                        logger.debug('Inserting %s into position %d on chain %s'
                                     % (mutation['from'], residue_number, source_entity.chain_id))
                        design_pose.insert_residue_type(mutation['from'], at=residue_number, chain=source_entity.chain_id)
                        # adjust mutations to account for insertion
                        for mutation_index in sorted(mutations.keys(), reverse=True):
                            if mutation_index < residue_number:
                                break
                            else:  # mutation should be incremented by one
                                mutations[mutation_index + 1] = mutations.pop(mutation_index)

                    # Check for expression tag addition to the designed sequences
                    inserted_design_sequence = design_entity.structure_sequence
                    selected_tag = {}
                    available_tags = find_expression_tags(inserted_design_sequence)
                    if available_tags:  # look for existing tag to remove from sequence and save identity
                        tag_names, tag_termini, existing_tag_sequences = \
                            zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags])
                        try:
                            preferred_tag_index = tag_names.index(args.preferred_tag)
                            if tag_termini[preferred_tag_index] in true_termini:
                                selected_tag = available_tags[preferred_tag_index]
                        except ValueError:
                            pass
                        pretag_sequence = remove_expression_tags(inserted_design_sequence, existing_tag_sequences)
                    else:
                        pretag_sequence = inserted_design_sequence
                    logger.debug('The pretag sequence is:\n%s' % pretag_sequence)

                    # Find the open reading frame offset using the structure sequence after insertion
                    offset = find_orf_offset(pretag_sequence, mutations)
                    formatted_design_sequence = pretag_sequence[offset:]
                    logger.debug('The open reading frame offset is %d' % offset)
                    logger.debug('The formatted_design sequence is:\n%s' % formatted_design_sequence)

                    if number_of_tags is None:  # don't solve tags
                        sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': {}}
                        continue

                    if not selected_tag:  # find compatible tags from matching PDB observations
                        uniprot_id_matching_tags = tag_sequences.get(uniprot_id, None)
                        if not uniprot_id_matching_tags:
                            uniprot_id_matching_tags = find_matching_expression_tags(uniprot_id=uniprot_id)
                            tag_sequences[uniprot_id] = uniprot_id_matching_tags

                        if uniprot_id_matching_tags:
                            tag_names, tag_termini, _ = \
                                zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in uniprot_id_matching_tags])
                        else:
                            tag_names, tag_termini, _ = [], [], []

                        iteration = 0
                        while iteration < len(tag_names):
                            try:
                                preferred_tag_index_2 = tag_names[iteration:].index(args.preferred_tag)
                                if tag_termini[preferred_tag_index_2] in true_termini:
                                    selected_tag = uniprot_id_matching_tags[preferred_tag_index_2]
                                    break
                            except ValueError:
                                selected_tag = \
                                    select_tags_for_sequence(sequence_id, uniprot_id_matching_tags,
                                                             preferred=args.preferred_tag, **termini_availability)
                                break
                            iteration += 1

                    if selected_tag.get('name'):
                        missing_tags[(des_dir, design)][idx] = 0
                        logger.debug('The pre-existing, identified tag is:\n%s' % selected_tag)
                    sequences_and_tags[design_string] = {'sequence': formatted_design_sequence, 'tag': selected_tag}

                # after selecting all tags, consider tagging the design as a whole
                if number_of_tags is not None:
                    number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])
                    if number_of_tags > number_of_found_tags:
                        print('There were %d requested tags for design %s and %d were found'
                              % (number_of_tags, des_dir, number_of_found_tags))
                        current_tag_options = \
                            '\n\t'.join(['%d - %s\n\tAvailable Termini: %s\n\t\t   TAGS: %s'
                                         % (i, entity_name, entity_termini_availability[entity_name], tag_options['tag'])
                                         for i, (entity_name, tag_options) in enumerate(sequences_and_tags.items(), 1)])
                        print('Current Tag Options:\n\t%s' % current_tag_options)
                        if args.avoid_tagging_helices:
                            print('Helical Termini:\n\t%s'
                                  % '\n\t'.join('%s\t%s' % item for item in entity_helical_termini.items()))
                        satisfied = input('If this is acceptable, enter "continue", otherwise, '
                                          'you can modify the tagging options with any other input.%s' % input_string)
                        if satisfied == 'continue':
                            number_of_found_tags = number_of_tags

                        iteration_idx = 0
                        while number_of_tags != number_of_found_tags:
                            if iteration_idx == len(missing_tags[(des_dir, design)]):
                                print('You have seen all options, but the number of requested tags (%d) doesn\'t equal the '
                                      'number selected (%d)' % (number_of_tags, number_of_found_tags))
                                satisfied = input('If you are satisfied with this, enter "continue", otherwise enter '
                                                  'anything and you can view all remaining options starting from the first '
                                                  'entity%s' % input_string)
                                if satisfied == 'continue':
                                    break
                                else:
                                    iteration_idx = 0
                            for idx, entity_missing_tag in enumerate(missing_tags[(des_dir, design)][iteration_idx:]):
                                sequence_id = '%s_%s' % (des_dir, des_dir.pose.entities[idx].name)
                                if entity_missing_tag and tag_index[idx]:  # isn't tagged but could be
                                    print('Entity %s is missing a tag. Would you like to tag this entity?' % sequence_id)
                                    if not boolean_choice():
                                        continue
                                else:
                                    continue
                                if args.preferred_tag:
                                    tag = args.preferred_tag
                                    while True:
                                        termini = input('Your preferred tag will be added to one of the termini. Which '
                                                        'termini would you prefer? [n/c]%s' % input_string)
                                        if termini.lower() in ['n', 'c']:
                                            break
                                        else:
                                            print('"%s" is an invalid input, one of "n" or "c" is required')
                                else:
                                    while True:
                                        tag_input = input('What tag would you like to use? Enter the number of the below '
                                                          'options.\n\t%s\n%s' %
                                                          ('\n\t'.join(['%d - %s' % (i, tag)
                                                                        for i, tag in enumerate(expression_tags, 1)]),
                                                           input_string))
                                        if tag_input.isdigit():
                                            tag_input = int(tag_input)
                                            if tag_input <= len(expression_tags):
                                                tag = list(expression_tags.keys())[tag_input - 1]
                                                break
                                        print('Input doesn\'t match available options. Please try again')
                                    while True:
                                        termini = input('Your tag will be added to one of the termini. Which termini would '
                                                        'you prefer? [n/c]%s' % input_string)
                                        if termini.lower() in ['n', 'c']:
                                            break
                                        else:
                                            print('"%s" is an invalid input. One of "n" or "c" is required' % termini)

                                selected_entity = list(sequences_and_tags.keys())[idx]
                                if termini == 'n':
                                    new_tag_sequence = \
                                        expression_tags[tag] + 'SG' + sequences_and_tags[selected_entity]['sequence'][:12]
                                else:  # termini == 'c'
                                    new_tag_sequence = \
                                        sequences_and_tags[selected_entity]['sequence'][-12:] + 'GS' + expression_tags[tag]
                                sequences_and_tags[selected_entity]['tag'] = {'name': tag, 'sequence': new_tag_sequence}
                                missing_tags[(des_dir, design)][idx] = 0
                                break

                            iteration_idx += 1
                            number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

                    elif number_of_tags < number_of_found_tags:  # when more than the requested number of tags were id'd
                        print('There were only %d requested tags for design %s and %d were found'
                              % (number_of_tags, des_dir, number_of_found_tags))
                        while number_of_tags != number_of_found_tags:
                            tag_input = input('Which tag would you like to remove? Enter the number of the currently '
                                              'configured tag option that you would like to remove. If you would like to '
                                              'keep all, specify "keep" \n\t%s\n%s'
                                              % ('\n\t'.join(['%d - %s\n\t\t%s' % (i, entity_name, tag_options['tag'])
                                                              for i, (entity_name, tag_options)
                                                              in enumerate(sequences_and_tags.items(), 1)]), input_string))
                            if tag_input == 'keep':
                                break
                            elif tag_input.isdigit():
                                tag_input = int(tag_input)
                                if tag_input <= len(sequences_and_tags):
                                    missing_tags[(des_dir, design)][tag_input - 1] = 1
                                    selected_entity = list(sequences_and_tags.keys())[tag_input - 1]
                                    sequences_and_tags[selected_entity]['tag'] = \
                                        {'name': None, 'termini': None, 'sequence': None}
                                    # tag = list(expression_tags.keys())[tag_input - 1]
                                    break
                                else:
                                    print('Input doesn\'t match an integer from the available options. Please try again')
                            else:
                                print('"%s" is an invalid input. Try again'
                                      % tag_input)
                            number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

                # apply all tags to the sequences
                cistronic_sequence = ''
                for idx, (design_string, sequence_tag) in enumerate(sequences_and_tags.items()):
                    tag, sequence = sequence_tag['tag'], sequence_tag['sequence']
                    # print('TAG:\n', tag.get('sequence'), '\nSEQUENCE:\n', sequence)
                    design_sequence = add_expression_tag(tag.get('sequence'), sequence)
                    if tag.get('sequence') and design_sequence == sequence:  # tag exists and no tag added
                        tag_sequence = expression_tags[tag.get('name')]
                        if tag.get('termini') == 'n':
                            if design_sequence[0] == 'M':  # remove existing Met to append tag to n-term
                                design_sequence = design_sequence[1:]
                            design_sequence = tag_sequence + 'SG' + design_sequence
                        else:  # termini == 'c'
                            design_sequence = design_sequence + 'GS' + tag_sequence

                    # If no MET start site, include one
                    if design_sequence[0] != 'M':
                        design_sequence = 'M%s' % design_sequence
                    if 'X' in design_sequence:
                        logger.critical('An unrecognized amino acid was specified in the sequence %s. '
                                        'This requires manual intervention!' % design_string)
                        # idx = 0
                        seq_length = len(design_sequence)
                        while True:
                            idx = design_sequence.find('X')
                            if idx == -1:  # Todo clean
                                break
                            idx_range = (idx - 6 if idx - 6 > 0 else 0, idx + 6 if idx + 6 < seq_length else seq_length)
                            while True:
                                new_amino_acid = input('What amino acid should be swapped for "X" in this sequence '
                                                       'context?\n\t%s\n\t%s%s'
                                                       % ('%d%s%d' % (idx_range[0] + 1, ' ' *
                                                                      (len(range(*idx_range)) -
                                                                       (len(str(idx_range[0])) + 1)), idx_range[1] + 1),
                                                          design_sequence[idx_range[0]:idx_range[1]], input_string)).upper()
                                if new_amino_acid in protein_letters:
                                    design_sequence = design_sequence[:idx] + new_amino_acid + design_sequence[idx + 1:]
                                    break
                                else:
                                    print('Input doesn\'t match a single letter canonical amino acid. Please try again')

                    # For a final manual check of sequence generation, find sequence additions compared to the design model
                    # and save to view where additions lie on sequence. Cross these additions with design structure to check
                    # if insertions are compatible
                    all_insertions = {residue: {'to': aa} for residue, aa in enumerate(design_sequence, 1)}
                    all_insertions.update(generate_mutations(design_sequence, designed_atom_sequences[idx], blanks=True))
                    # Reduce to sequence only
                    inserted_sequences[design_string] = '%s\n%s' % (''.join([res['to'] for res in all_insertions.values()]),
                                                                    design_sequence)
                    logger.info('Formatted sequence comparison:\n%s' % inserted_sequences[design_string])
                    final_sequences[design_string] = design_sequence
                    if args.nucleotide:
                        try:
                            nucleotide_sequence = \
                                optimize_protein_sequence(design_sequence, species=args.optimize_species)
                        except NoSolutionError:  # add the protein sequence?
                            logger.warning('Optimization of %s was not successful!' % design_string)
                            codon_optimization_errors[design_string] = design_sequence
                            break

                        if args.multicistronic:
                            if idx > 0:
                                cistronic_sequence += intergenic_sequence
                            cistronic_sequence += nucleotide_sequence
                        else:
                            nucleotide_sequences[design_string] = nucleotide_sequence
                if args.multicistronic:
                    nucleotide_sequences[str(des_dir)] = cistronic_sequence

        # Report Errors
        if codon_optimization_errors:
            error_file = SDUtils.write_fasta_file(codon_optimization_errors,
                                                  '%sOptimizationErrorProteinSequences' % args.selection_string,
                                                  out_path=outdir, csv=args.csv)
        # Write output sequences to fasta file
        seq_file = SDUtils.write_fasta_file(final_sequences, '%sSelectedSequences' % args.selection_string,
                                            out_path=outdir, csv=args.csv)
        logger.info('Final Design protein sequences written to %s' % seq_file)
        seq_comparison_file = SDUtils.write_fasta_file(inserted_sequences, '%sSelectedSequencesExpressionAdditions'
                                                       % args.selection_string, out_path=outdir, csv=args.csv)
        logger.info('Final Expression sequence comparison to Design sequence written to %s' % seq_comparison_file)
        # check for protein or nucleotide output
        if args.nucleotide:
            nucleotide_sequence_file = SDUtils.write_fasta_file(nucleotide_sequences, '%sSelectedSequencesNucleotide'
                                                                % args.selection_string, out_path=outdir, csv=args.csv)
            logger.info('Final Design nucleotide sequences written to %s' % nucleotide_sequence_file)
    # ---------------------------------------------------
    elif args.module == 'multicistronic':
        if args.multicistronic_intergenic_sequence:
            intergenic_sequence = args.multicistronic_intergenic_sequence
        else:
            intergenic_sequence = default_multicistronic_sequence

        file = args.file[0]
        if file.endswith('.csv'):
            with open(file) as f:
                design_sequences = [SeqRecord(Seq(sequence), annotations={'molecule_type': 'Protein'}, id=name)
                                    for name, sequence in reader(f)]
                #                    for name, sequence in zip(*reader(f))]
        else:
            design_sequences = list(read_fasta_file(file))

        nucleotide_sequences = {}
        for idx, group_start_idx in enumerate(list(range(len(design_sequences)))[::args.number_of_genes], 1):
            cistronic_sequence = \
                optimize_protein_sequence(design_sequences[group_start_idx], species=args.optimize_species)
            for protein_sequence in design_sequences[group_start_idx + 1: group_start_idx + args.number_of_genes]:
                cistronic_sequence += intergenic_sequence
                cistronic_sequence += optimize_protein_sequence(protein_sequence, species=args.optimize_species)
            new_name = '%s_cistronic' % design_sequences[group_start_idx].id
            nucleotide_sequences[new_name] = cistronic_sequence
            logger.info('Finished sequence %d - %s' % (idx, new_name))

        location = file
        if not args.selection_string:
            args.selection_string = '%s_' % os.path.basename(os.path.splitext(location)[0])
        else:
            args.selection_string += '_'
        outdir = os.path.join(os.getcwd(), '%sSelectedDesigns' % args.selection_string)

        nucleotide_sequence_file = SDUtils.write_fasta_file(nucleotide_sequences, '%sSelectedSequencesNucleotide'
                                                            % args.selection_string, out_path=os.getcwd(), csv=args.csv)
        logger.info('Final Design nucleotide sequences written to %s' % nucleotide_sequence_file)
    # ---------------------------------------------------
    # elif args.module == 'rename_scores':
    #     rename = {'combo_profile_switch': 'limit_to_profile', 'favor_profile_switch': 'favor_frag_limit_to_profile'}
    #     for des_directory in pose_directories:
    #         SDUtils.rename_decoy_protocols(des_directory, rename)
    # ---------------------------------------------------
    # elif args.module == 'modify':  # -m mod
    #     if args.multi_processing:
    #         if args.mod == 'consolidate_degen':
    #             exit('Operation impossible with flag "-mp"')
    #
    #     else:
    #         if args.mod == 'consolidate_degen':
    #             logger.info('Consolidating DEGEN directories')
    #             accessed_dirs = [rsync_dir(pose_directory) for pose_directory in pose_directories]
    # ---------------------------------------------------
    elif args.module == 'status':  # -n number, -s stage, -u update
        if args.update:
            for design in pose_directories:
                update_status(design.serialized_info, args.stage, mode=args.update)
        else:
            if args.number_of_trajectories:
                logger.info('Checking for %d files based on --number_of_trajectories flag' % args.number_of_trajectories)
            if args.stage:
                status(pose_directories, args.stage, number=args.number_of_trajectories)
            else:
                for stage in PUtils.stage_f:
                    s = status(pose_directories, stage, number=args.number_of_trajectories)
                    if s:
                        logger.info('For "%s" stage, default settings should generate %d files'
                                    % (stage, PUtils.stage_f[stage]['len']))
    # ---------------------------------------------------
    elif args.module == 'visualize':
        import visualization.VisualizeUtils as VSUtils
        from pymol import cmd

        # if 'escher' in sys.argv[1]:
        if not args.directory:
            exit('A directory with the desired designs must be specified using -d/--directory!')

        if ':' in args.directory:  # args.file  Todo location
            print('Starting the data transfer from remote source now...')
            os.system('scp -r %s .' % args.directory)
            file_dir = os.path.basename(args.directory)
        else:  # assume the files are local
            file_dir = args.directory
        # files = VSUtils.get_all_file_paths(file_dir, extension='.pdb', sort=not args.order)

        if args.order == 'alphabetical':
            files = VSUtils.get_all_file_paths(file_dir, extension='.pdb')  # sort=True)
        else:  # if args.order == 'none':
            files = VSUtils.get_all_file_paths(file_dir, extension='.pdb', sort=False)

        print('FILES:\n %s' % files[:4])
        if args.order == 'paths':  # TODO FIX janky paths handling below
            # for design in pose_directories:
            with open(args.file[0], 'r') as f:
                paths = \
                    map(str.replace, map(str.strip, f.readlines()),
                        repeat('/yeates1/kmeador/Nanohedra_T33/SymDesignOutput/Projects/'
                               'NanohedraEntry54DockedPoses_Designs/'), repeat(''))
                paths = list(paths)
            ordered_files = []
            for path in paths:
                for file in files:
                    if path in file:
                        ordered_files.append(file)
                        break
            files = ordered_files
            # raise NotImplementedError('--order choice "paths" hasn\'t been set up quite yet... Use another method')
            # ordered_files = []
            # for index in df.index:
            #     for file in files:
            #         if index in file:
            #             ordered_files.append(file)
            #             break
            # files = ordered_files
        elif args.order == 'dataframe':
            if not args.dataframe:
                df_glob = sorted(glob(os.path.join(file_dir, 'TrajectoryMetrics.csv')))
                try:
                    args.dataframe = df_glob[0]
                except IndexError:
                    raise IndexError('There was no --dataframe specified and one couldn\'t be located in %s. Initialize'
                                     ' again with the path to the relevant dataframe' % location)

            df = pd.read_csv(args.dataframe, index_col=0, header=[0])
            print('INDICES:\n %s' % df.index.to_list()[:4])
            ordered_files = []
            for index in df.index:
                for file in files:
                    # if index in file:
                    if os.path.splitext(os.path.basename(file))[0] in index:
                        ordered_files.append(file)
                        break
            # print('ORDERED FILES (%d):\n %s' % (len(ordered_files), ordered_files))
            files = ordered_files

        if not files:
            exit('No .pdb files found in %s. Are you sure this is correct?' % location)

        # if len(sys.argv) > 2:
        #     low, high = map(float, sys.argv[2].split('-'))
        #     low_range, high_range = int((low / 100) * len(files)), int((high / 100) * len(files))
        #     if low_range < 0 or high_range > len(files):
        #         raise ValueError('The input range is outside of the acceptable bounds [0-100]')
        #     print('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))
        # else:
        print(low_range, high_range)
        print(all_poses)
        for idx, file in enumerate(files[low_range:high_range], low_range + 1):
            if args.name == 'original':
                cmd.load(file)
            else:  # if args.name == 'numerical':
                cmd.load(file, object=idx)

        print('\nTo expand all designs to the proper symmetry, issue:\nPyMOL> expand name=all, symmetry=T'
              '\nYou should replace "T" with whatever symmetry your design is in\n')
    # ---------------------------------------------------
    # else:
    #     exit('No module was selected! Did you include one? To get started, checkout the %s' % PUtils.guide_string)
    # -----------------------------------------------------------------------------------------------------------------
    # Finally run terminate(). Formats designs passing output parameters and report program exceptions
    # -----------------------------------------------------------------------------------------------------------------
