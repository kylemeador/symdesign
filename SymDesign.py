"""
Module for distribution of SymDesign commands. Includes pose initialization, distribution of Rosetta commands to
SLURM computational clusters, analysis of designed poses, and sequence selection of completed structures.

"""
import argparse
import copy
import datetime
import os
import shutil
import subprocess
import sys
import time
from csv import reader
from glob import glob
from itertools import repeat, product, combinations
from json import loads, dumps

import pandas as pd
import psutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import PathUtils as PUtils
import SymDesignUtils as SDUtils
from Query.PDB import input_string, bool_d, invalid_string, verify_choice, boolean_choice
from utils.CmdLineArgParseUtils import query_mode
from utils.PDBUtils import orient_pdb_file
from Query import Flags
from classes.SymEntry import SymEntry
from classes.EulerLookup import EulerLookup
from interface_analysis.Database import FragmentDatabase, Database
from CommandDistributer import distribute, hhblits_memory_threshold, update_status, script_cmd, rosetta_flags
from DesignDirectory import DesignDirectory, get_sym_entry_from_nanohedra_directory, relax_flags
from NanohedraWrap import nanohedra_command, nanohedra_design_recap
from PDB import PDB
from Pose import fetch_pdb_file
from ClusterUtils import pose_rmsd_mp, pose_rmsd_s, cluster_poses, cluster_designs, invert_cluster_map, \
    group_compositions
from ProteinExpression import find_expression_tags, find_matching_expression_tags, add_expression_tag, \
    select_tags_for_sequence, remove_expression_tags, expression_tags
from DesignMetrics import filter_pose, master_metrics, query_user_for_metrics, rank_dataframe_by_metric_weights
from SequenceProfile import generate_mutations, find_orf_offset  # , pdb_to_pose_offset


def rename(des_dir, increment=PUtils.nstruct):
    """Rename the decoy numbers in a DesignDirectory by a specified increment

    Args:
        des_dir (DesignDirectory): A DesignDirectory object
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
            # if os.path.basename(des_dir.composition) in observed_building_blocks:
            #     continue
            # else:
            #     observed_building_blocks.append(os.path.basename(des_dir.composition))
            # f_degen1, f_degen2, f_rot1, f_rot2 = get_last_sampling_state('%s_log.txt' % des_dir.composition)
            # degens, rotations = \
            # SDUtils.degen_and_rotation_parameters(SDUtils.gather_docking_metrics(des_dir.program_root))
            #
            # dock_dir = DesignDirectory(path, auto_structure=False)
            # dock_dir.program_root = glob(os.path.join(path, 'NanohedraEntry*DockedPoses'))
            # dock_dir.composition = [next(os.walk(dir))[1] for dir in dock_dir.program_root]
            # dock_dir.log = [os.path.join(_sym, 'master_log.txt') for _sym in dock_dir.program_root]
            # dock_dir.building_block_logs = [os.path.join(_sym, bb_dir, 'bb_dir_log.txt') for sym in dock_dir.composition
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
                    raise DesignError('This functionality has been removed \'des_dir.gather_docking_metrics()\'')
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


def format_additional_flags(flags):
    """Takes non-argparse specified flags and returns them into a dictionary compatible with argparse style.
    This is useful for handling general program flags that apply to many modules, would be tedious to hard code and
    request from the user

    Returns:
        (dict)
    """
    formatted_flags = []
    for flag in flags:
        if flag[0] == '-' and flag[1] != '-':
            formatted_flags.append(flag)
        elif flag[0] == '-' and flag[1] == '-':
            formatted_flags.append(flag[1:])
        elif flag[0] != '-':
            formatted_flags.append(flag)  # '-%s' % flag)

    # combines ['-symmetry', 'O', '-nanohedra_output', True', ...]
    combined_extra_flags = []
    for idx, flag in enumerate(formatted_flags):
        if flag.startswith('-'):  # this is a real flag
            extra_arguments = ''
            increment = 1
            while (idx + increment) != len(formatted_flags) and not formatted_flags[idx + 1].startswith('-'):  # an argument
                extra_arguments += ' %s' % formatted_flags[idx + increment]
                increment += 1
            combined_extra_flags.append('%s%s' % (flag, extra_arguments))  # extra_flags[idx + 1]))
    # logger.debug('Combined flags: %s' % combined_extra_flags)

    # parses ['-nanohedra_output True', ...]
    final_flags = {}
    for flag_arg in combined_extra_flags:
        if ' ' in flag_arg:
            flag = flag_arg.split()[0].lstrip('-')
            final_flags[flag] = flag_arg.split()[1]
            if final_flags[flag].title() in ['None', 'True', 'False']:
                final_flags[flag] = eval(final_flags[flag].title())
        else:  # remove - from the front and add to the dictionary
            final_flags[flag_arg[1:]] = None

    return final_flags


def catch_and_clean_exceptions():
    try:
        symdesign()
    except Exception as ex:
        master_directory = next(iter(designs))
        raise ex


def terminate(module, designs, location=None, results=None, output=True):
    """Format designs passing output parameters and report program exceptions

    Args:
        module (str): The module used
        designs (list(DesignDirectory)): The designs processed
    Keyword Args:
        location=None (str): Where the designs were retrieved from
        results=None (list): The returned results from the module run. By convention contains results and exceptions
        output=False (bool): Whether the module used requires a file to be output
    Returns:
        (None)
    """
    global out_path, timestamp
    # save any information found during the design command to it's serialized state
    for design in designs:
        design.pickle_info()

    if results:
        success = [designs[idx] for idx, result in enumerate(results) if not isinstance(result, BaseException)]
        exceptions = [(designs[idx], result) for idx, result in enumerate(results) if isinstance(result, BaseException)]
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

    if success and output:  # and (all_poses and design_directories and not args.file):  # Todo
        master_directory = next(iter(designs))
        job_paths = master_directory.job_paths
        if not location:
            design_source = os.path.basename(master_directory.project_designs)
        else:
            design_source = os.path.splitext(os.path.basename(location))[0]
        if low and high:
            timestamp = '%s-%.2f-%.2f' % (timestamp, low, high)
        # Make single file with names of each directory where all_docked_poses can be found
        # project_string = os.path.basename(design_directories[0].project_designs)
        # program_root = design_directories[0].program_root
        if args.output_design_file:
            designs_file = args.output_design_file
        else:
            designs_file = os.path.join(job_paths, '%s_%s_%s_pose.paths' % (module, design_source, timestamp))

        with open(designs_file, 'w') as f:
            f.write('%s\n' % '\n'.join(design.path for design in success))
        logger.critical('The file \'%s\' contains the locations of all designs in your current project that passed '
                        'internal checks/filtering. Utilize this file to interact with %s designs in future commands '
                        'for this project such as \'%s --file %s MODULE\'\n'
                        % (designs_file, PUtils.program_name, PUtils.program_command, designs_file))

        if module == PUtils.analysis:
            # failures = [idx for idx, result in enumerate(results) if isinstance(result, BaseException)]
            # for index in reversed(failures):
            #     del results[index]
            successes = [result for result in results if not isinstance(result, BaseException)]

            if len(success) > 0:
                all_scores = master_directory.all_scores
                # Save Design DataFrame
                design_df = pd.DataFrame(successes)
                if args.output == PUtils.analysis_file:
                    out_path = os.path.join(all_scores, args.output % (os.path.splitext(location)[0], timestamp))
                else:  # user provided the output path
                    local_dummy = True  # the global out_path should be used
                    # out_path = os.path.join(all_scores, args.output)
                out_path = out_path if out_path.endswith('.csv') else '%s.csv' % out_path
                design_df.to_csv(out_path)
                logger.info('Analysis of all poses written to %s' % out_path)
                if save:
                    logger.info('Analysis of all Trajectories and Residues written to %s' % all_scores)

        design_stage = PUtils.stage[12] if getattr(args, 'scout', None) \
            else (PUtils.stage[2] if getattr(args, 'legacy', None)
                  else (PUtils.stage[14] if getattr(args, 'structure_background', None)
                        else PUtils.stage[13]))  # hbnet_design_profile
        module_files = {PUtils.interface_design: design_stage, PUtils.nano: PUtils.nano,
                        'interface_metrics': 'interface_metrics',
                        'custom_script': os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0]}
        stage = module_files.get(module)
        if stage:
            if len(success) == 0:
                exit(exit_code)
            # sbatch_scripts = master_directory.sbatch_scripts
            command_file = SDUtils.write_commands([os.path.join(design.scripts, '%s.sh' % stage) for design in success],
                                                  out_path=job_paths, name='_'.join((module, design_source, timestamp)))
            sbatch_file = distribute(file=command_file, out_path=master_directory.sbatch_scripts, scale=module)
            #                                                                    ^ for sbatch template
            logger.critical('Ensure the created SBATCH script(s) are correct. Specifically, check that the job array '
                            'and any node specifications are accurate. You can look at the SBATCH manual (man sbatch or'
                            ' sbatch --help) to understand the variables or ask for help if you are still unsure.')
            logger.info('Once you are satisfied, enter the following to distribute:\n\tsbatch %s' % sbatch_file)
    print('\n')
    exit(exit_code)


def generate_sequence_template(pdb_file):
    pdb = PDB.from_file(pdb_file)
    sequence = SeqRecord(Seq(''.join(pdb.atom_sequences.values()), 'Protein'), id=pdb.filepath)
    sequence_mask = copy.copy(sequence)
    sequence_mask.id = 'residue_selector'
    sequences = [sequence, sequence_mask]
    return SDUtils.write_fasta(sequences, file_name='%s_residue_selector_sequence' % os.path.splitext(pdb.filepath)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description=
                                     '\nControl all input/output of the various %s operations including: '
                                     '\n\t1. %s docking '
                                     '\n\t2. Pose set up, sampling, assembly generation, fragment decoration'
                                     '\n\t3. Interface design using constrained residue profiles and Rosetta'
                                     '\n\t4. Analysis of all designs using interface metrics '
                                     '\n\t5. Design selection and sequence formatting by combinatorial linear weighting'
                                     ' of interface metrics.\n\n'
                                     'If your a first time user, try \'%s --guide\''
                                     '\nAll jobs have built in features for command monitoring & distribution to '
                                     'computational clusters for parallel processing.\n'
                                     % (PUtils.program_name, PUtils.nano.title(), PUtils.program_command),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---------------------------------------------------
    # parser.add_argument('-symmetry', '--symmetry', type=str, help='The design symmetry to use. Possible symmetries '
    #                                                             'include %s' % ', '.join(SDUtils.possible_symmetries))
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to stdout?\nDefault=False')
    parser.add_argument('-C', '--cluster_map', type=os.path.abspath,
                        help='The location of a serialized file containing spatially or interfacially clustered poses')
    parser.add_argument('-c', '--cores', type=int,
                        help='Number of cores to use with --multiprocessing. If -mp is run in a cluster environment, '
                             'the number of cores will reflect the allocation provided by the cluster, otherwise, '
                             'specify the number of cores\nDefault=#ofCores-1')
    parser.add_argument('-d', '--directory', type=os.path.abspath, metavar='/path/to_your_pdb_files/',
                        help='Master directory where poses to be designed with %s are located. This may be the output '
                             'directory from %s.py, a random directory with poses requiring interface design, or the '
                             'output from %s. If the directory lives in a %sOutput directory, all projects within the '
                             'directory will be selected. For finer control over which poses to manipulate, use --file,'
                             ' --project, or --single flags.'
                             % (PUtils.program_name, PUtils.nano, PUtils.program_name, PUtils.program_name))
    parser.add_argument('-f', '--file', type=os.path.abspath, metavar='/path/to/file_with_directory_names.txt',
                        help='File with location(s) of %s designs. For each run of %s, a file will be created '
                             'specifying the specific directories to use in subsequent %s commands of the same designs.'
                             ' If pose-IDs are specified in a file, say as the result of %s or %s, in addition to the '
                             'pose-ID file, provide your %s working directory to locate the pose-Ids of interest.'
                             % (PUtils.program_name, PUtils.program_name, PUtils.program_name, PUtils.analysis,
                                PUtils.select_designs, PUtils.program_name),
                        default=None, nargs=1)  # , nargs='*')  # TODO make list of unknown length
    parser.add_argument('-g', '--guide', action='store_true',
                        help='Access the %s guide! Display the program or module specific guide. Ex: \'%s --guide\' '
                             'or \'%s\'' % (PUtils.program_name, PUtils.program_command, PUtils.submodule_guide))
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-of', '--output_design_file', type=str,
                        help='If provided, the name of the output designs file. If blank, one will be automatically '
                             'generated based off input_location, module, and the time.')
    parser.add_argument('-p', '--project', type=os.path.abspath,
                        metavar='/path/to/SymDesignOutput/Projects/your_project',
                        help='If pose names are specified by project instead of directories, which project to use?')
    parser.add_argument('-r', '--run_in_shell', action='store_true',
                        help='Should commands be executed through %s command? Doesn\'t maximize cassini\'s '
                             'computational resources and can cause long trajectories to fail on a single mistake.'
                             '\nDefault=False' % PUtils.program_name)
    parser.add_argument('-s', '--single', type=os.path.abspath,
                        metavar='/path/to/SymDesignOutput/Projects/your_project/single_design[.pdb]',
                        help='If design name is specified by a single path instead')
    subparsers = parser.add_subparsers(title='Modules', dest='module',
                                       description='These are the different modes that designs can be processed',
                                       help='Chose a Module followed by Module specific flags. To get help with a '
                                            'Module such as Module algorithmic specifics or to see example commands '
                                            'enter:\t%s\n\nTo get help with Module flags enter:\t%s\n, Any Module '
                                            '--guide or --help can be accessed in this way.'
                                            % (PUtils.submodule_guide, PUtils.submodule_help))
    parser.add_argument('-F', '--force_flags', action='store_true',
                        help='Force generation of a new flags file to update script parameters')
    # ---------------------------------------------------
    parser_query = subparsers.add_parser('query', help='Query %s.py docking entries' % PUtils.nano.title())
    # ---------------------------------------------------
    parser_flag = subparsers.add_parser('flags', help='Generate a flags file for %s' % PUtils.program_name)
    parser_flag.add_argument('-t', '--template', action='store_true',
                             help='Generate a flags template to edit on your own.')
    parser_flag.add_argument('-m', '--module', dest='flags_module', action='store_true',
                             help='Generate a flags template to edit on your own.')
    # ---------------------------------------------------
    parser_selection = subparsers.add_parser('residue_selector',
                                             help='Generate a residue selection for %s' % PUtils.program_name)
    # ---------------------------------------------------
    parser_orient = subparsers.add_parser('orient',
                                          help='Orient a symmetric assembly in a cannonical orientation at the origin')
    # ---------------------------------------------------
    parser_asu = subparsers.add_parser('find_asu', help='From a symmetric assembly, locate an ASU and save the result.')
    # ---------------------------------------------------
    parser_expand = subparsers.add_parser('expand_asu',
                                          help='For given poses, expand the asymmetric unit to a symmetric assembly and'
                                               ' write the result to the design directory.')
    # ---------------------------------------------------
    parser_rename = subparsers.add_parser('rename_chains',
                                          help='For given poses, rename the chains in the source PDB to the alphabetic '
                                               'order. Useful for writing a multi-model as distinct chains or fixing '
                                               'common PDB formatting errors as well. Writes to design directory')
    # ---------------------------------------------------
    parser_dock = subparsers.add_parser(PUtils.nano,
                                        help='Submit jobs to %s.py\nIf a docking directory structure is set up, provide'
                                             ' the overall directory location with program argument -d/-f, otherwise, '
                                             'use the Module arguments -d1/-d2 to specify directories with lists of '
                                             'oligomers to dock' % PUtils.nano.title())
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
    parser_fragments = subparsers.add_parser(PUtils.generate_fragments,
                                             help='Generate fragment overlap for poses of interest.')
    # parser_design.add_argument('-i', '--fragment_database', type=str,
    #                            help='Database to match fragments for interface specific scoring matrices. One of %s'
    #                                 '\nDefault=%s' % (','.join(list(PUtils.frag_directory.keys())),
    #                                                   list(PUtils.frag_directory.keys())[0]),
    #                            default=list(PUtils.frag_directory.keys())[0])
    # ---------------------------------------------------
    parser_cluster = subparsers.add_parser(PUtils.cluster_poses,
                                           help='Cluster all designs by their spatial similarity. This can remove '
                                                'redundancy or be useful in identifying conformationally flexible '
                                                'docked configurations.')
    parser_cluster.add_argument('-o', '--output', type=str, default=PUtils.clustered_poses,
                                help='Name of the output .pkl file containing design clusters Will be saved to the %s/'
                                     ' folder of the output.\nDefault=%s'
                                     % (PUtils.data.title(), PUtils.clustered_poses % ('LOCATION', 'TIMESTAMP')))
    # ---------------------------------------------------
    parser_design = subparsers.add_parser(PUtils.interface_design,
                                          help='Gather poses of interest and format for design using sequence '
                                               'constraints in Rosetta. Constrain using evolutionary profiles of '
                                               'homologous sequences and/or fragment profiles extracted from the PDB or'
                                               ' neither.')
    parser_design.add_argument('-s', '--scout', action='store_true',
                               help='Whether to set up a low resolution scouting protocol to survey designability.')
    # parser_design.add_argument('-i', '--fragment_database', type=str,
    #                            help='Database to match fragments for interface specific scoring matrices. One of %s'
    #                                 '\nDefault=%s' % (','.join(list(PUtils.frag_directory.keys())),
    #                                                   list(PUtils.frag_directory.keys())[0]),
    #                            default=list(PUtils.frag_directory.keys())[0])
    # ---------------------------------------------------
    parser_interface_metrics = \
        subparsers.add_parser('interface_metrics',
                              help='Set up RosettaScript to analyze interface metrics from an interface design job. '
                                   'If the specific flags should be generated fresh use --force_flags')
    # parser_interface_metrics.add_argument('-F', '--force_flags', action='store_true',
    #                                       help='Force generation of a new flags file to update script parameters')
    # ---------------------------------------------------
    parser_custom_script = \
        subparsers.add_parser('custom_script',
                              help='Set up a custom RosettaScripts.xml for designs. The custom_script will be provided '
                                   'to every directory specified and can be run with a number of options specified '
                                   'below. Additionally, If the script should be run multiple times, include the flag '
                                   '--number_of_trajectories INT. If the specific flags should be generated fresh use '
                                   '--force_flags')
    # parser_interface_metrics.add_argument('-F', '--force_flags', action='store_true',
    #                                       help='Force generation of a new flags file to update script parameters')
    parser_custom_script.add_argument('-l', '--file_list', action='store_true',
                                      help='Whether to use already produced designs in the designs/ directory')
    parser_custom_script.add_argument('-n', '--native', type=str,
                                      help='What structure to use as a \'native\' structure for Rosetta reference '
                                           'calculations. Default=refined_pdb',
                                      choices=['source', 'asu', 'assembly', 'refine_pdb', 'refined_pdb',
                                               'consensus_pdb', 'consensus_design_pdb'])
    parser_custom_script.add_argument('--score_only', action='store_true', help='Whether to only score the design(s)')
    parser_custom_script.add_argument('script', type=os.path.abspath, help='The location of the custom script')
    parser_custom_script.add_argument('--suffix', type=str, metavar='SUFFIX',
                                      help='Append to each output file (decoy in .sc and .pdb) the script name (i.e. '
                                           '\'decoy_SUFFIX\') to identify this protocol. No extension will be included')
    parser_custom_script.add_argument('-v', '--variables', type=str, nargs='*',
                                      help='Additional variables that should be populated in the script. Provide a list'
                                           ' of such variables with the format \'variable1=value variable2=value\'. '
                                           'Where variable1 is a RosettaScripts %%%%variable1%%%% and value is a' 
                                           # ' either a'  # Todo
                                           ' known value'
                                           # ' or an attribute available to the Pose object'
                                           '. For variables that must'
                                           ' be calculated on the fly for each design, please modify the Pose.py class '
                                           'to produce a method that can generate an attribute with the specified name')
    # ---------------------------------------------------
    parser_analysis = subparsers.add_parser(PUtils.analysis,
                                            help='Analyze all designs specified. %s --guide %s will inform you about '
                                                 'the various metrics available to analyze.'
                                                 % (PUtils.program_command, PUtils.analysis))
    parser_analysis.add_argument('-o', '--output', type=str, default=PUtils.analysis_file,
                                 help='Name of the output .csv file containing design metrics. Will be saved to the %s/'
                                      ' folder of the output.\nDefault=%s'
                                      % (PUtils.all_scores, PUtils.analysis_file % ('LOCATION', 'TIMESTAMP')))
    parser_analysis.add_argument('-N', '--no_save', action='store_true',
                                 help='Don\'t save trajectory information.\nDefault=False')
    parser_analysis.add_argument('-f', '--figures', action='store_true',
                                 help='Create and save figures for all poses?\nDefault=False')
    parser_analysis.add_argument('-j', '--join', action='store_true',
                                 help='Join Trajectory and Residue Dataframes?\nDefault=False')
    # ---------------------------------------------------
    parser_filter = subparsers.add_parser(PUtils.select_designs,
                                          help='Select designs based on design specific metrics. Can be one of a '
                                               'handful of --metrics or from a DesignAnalysis.csv file generated by %s'
                                               % PUtils.analysis)
    filter_required = parser_filter.add_mutually_exclusive_group(required=True)
    filter_required.add_argument('-df', '--dataframe', type=os.path.abspath,
                                 metavar='/path/to/AllPoseDesignMetrics.csv',
                                 help='Dataframe.csv from analysis containing pose info.')
    filter_required.add_argument('-m', '--metric', type=str,
                                 help='If a simple metric filter is required, what metric would you like to sort '
                                      'Designs by?', choices=['score', 'fragments_matched'])
    filter_required.add_argument('-p', '--pose_design_file', type=str, metavar='/path/to/pose_design.csv',
                                 help='Name of .csv file with (pose, design pairs to serve as sequence selector')
    parser_filter.add_argument('-f', '--filter', action='store_true',
                               help='Whether to filter sequence selection using metrics from DataFrame')
    parser_filter.add_argument('-np', '--number_poses', type=int, default=0, metavar='INT',
                               help='Number of top poses to return per pool of designs.\nDefault=All')
    parser_filter.add_argument('-s', '--selection_string', type=str, metavar='string',
                               help='String to prepend to output for custom design selection name')
    parser_filter.add_argument('-w', '--weight', action='store_true',
                               help='Whether to weight sequence selection using metrics from DataFrame')
    # ---------------------------------------------------
    parser_sequence = subparsers.add_parser('sequence_selection',
                                            help='Generate protein sequences for selected designs. Either -df or -p is '
                                                 'required. If both are provided, -p will be prioritized')
    parser_sequence.add_argument('-a', '--avoid_tagging_helices', action='store_true',
                                 help='Should tags be avoided at termini with helices?')
    parser_sequence.add_argument('-g', '--global_sequences', action='store_true',
                                 help='Should sequences be selected based on their ranking in the total design pool. '
                                      'This will search for the top sequences from all poses and then choose only one '
                                      'sequence per pose')
    parser_sequence.add_argument('-e', '--entity_specification', type=str,
                                 help='If there are specific entities in the designs you want to tag, indicate how '
                                      'tagging should occur. Viable options include \'single\'- a single entity, '
                                      '\'all\'- all antities, \'none\'- no entities, or provide a comma separated list '
                                      'such as \'1,0,1\' where \'1\' indicates a tag requirement and \'0\' indicates no'
                                      ' tag is required.')
    parser_sequence.add_argument('-f', '--filter', action='store_true',
                                 help='Whether to filter sequence selection using metrics from DataFrame')
    parser_sequence.add_argument('-ns', '--number_sequences', type=int, default=sys.maxsize, metavar='INT',
                                 help='Number of top sequences to return. If global_sequences is True, returns the '
                                      'specified number_sequences sequences (Default=No Limit).\nOtherwise the '
                                      'specified number will be found from each pose (Default=1)')
    parser_sequence.add_argument('-p', '--protocol', type=str,
                                 help='Is there a specific protocol to grab sequences from?')
    parser_sequence.add_argument('-s', '--selection_string', type=str, metavar='string',
                                 help='String to prepend to output for custom sequence selection name')
    parser_sequence.add_argument('-t', '--preferred_tag', type=str,
                                 help='The name of your preferred expression tag. Default=his_tag',
                                 choices=expression_tags.keys(), default='his_tag')
    parser_sequence.add_argument('-w', '--weight', action='store_true',
                                 help='Whether to weight sequence selection using metrics from DataFrame')
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Get design status for selected designs')
    parser_status.add_argument('-n', '--number_designs', type=int, help='Number of trajectories per design',
                               default=None)
    parser_status.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                               help='The stage of design to check status of. One of %s'
                                    % ', '.join(list(v for v in PUtils.stage_f.keys())), default=None)
    parser_status.add_argument('-u', '--update', type=str, choices=('check', 'set', 'remove'),
                               help='Provide an update to the serialized state of the specified stage', default=None)
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
    # parser_merge = subparsers.add_parser('merge',
    #                                      help='Merge all completed designs from location 2 (-f2/-d2) to location '
    #                                           '1(-f/-d). Includes renaming. Highly suggested you copy original data,'
    #                                           ' very untested!!!')
    # parser_merge.add_argument('-d2', '--directory2', type=os.path.abspath, default=None,
    #                           help='Directory 2 where poses should be copied from and appended to location 1 poses')
    # parser_merge.add_argument('-f2', '--file2', type=str, default=None,
    #                           help='File 2 where poses should be copied from and appended to location 1 poses')
    # parser_merge.add_argument('-F', '--force', action='store_true', help='Overwrite merge paths?\nDefault=False')
    # parser_merge.add_argument('-i', '--increment', type=int,
    #                           help='How many to increment each design by?\nDefault=%d' % PUtils.nstruct)
    # parser_merge.add_argument('-m', '--merge_mode', type=str, help='Whether to operate merge in design or dock mode?')
    # ---------------------------------------------------
    # parser_modify = subparsers.add_parser('modify', help='Modify something for program testing')
    # parser_modify.add_argument('-m', '--mod', type=str,
    #                            help='Which type of modification?\nChoose from consolidate_degen or pose_map')
    # ---------------------------------------------------
    parser_rename_scores = subparsers.add_parser('rename_scores', help='Rename Protocol names according to dictionary')
    # these might be helpful for intermixing arguments before/after subparsers... (Modules)
    # parser.parse_intermixed_args(args=None, namespace=None)
    # parser.parse_known_intermixed_args
    unknown_args = None
    args, additional_flags = parser.parse_known_args()
    # TODO work this into the flags parsing to grab module if included first and program flags if included after
    # while len(additional_flags) and additional_flags != unknown_args:
    #     args, additional_flags = parser.parse_known_args(additional_flags, args)
    #     unknown_args = additional_flags
    # args, additional_flags = parser.parse_known_args(additional_flags, args)
    # -----------------------------------------------------------------------------------------------------------------
    # Start Logging - Root logs to stream with level warning
    # -----------------------------------------------------------------------------------------------------------------
    timestamp = time.strftime('%y-%m-%d-%H%M%S')
    if args.debug:
        # Root logs to stream with level debug
        logger = SDUtils.start_log(level=1, set_logger_level=True)
        logger.debug('Debug mode. Verbose output')
    else:
        # Root logger logs to stream with level 'warning'
        SDUtils.start_log(level=3, set_logger_level=True)
        # Root logger logs all emissions to a single file with level 'info'. Stream above still emits at 'warning'
        SDUtils.start_log(handler=2, set_logger_level=True,
                          location=os.path.join(os.getcwd(), os.path.basename(__file__).split('.')[0].lower()))
        # SymDesign main logs to stream with level info
        logger = SDUtils.start_log(name=os.path.basename(__file__).split('.')[0], propagate=False)
        # All Designs will log to specific file with level info unless -skip_logging is passed
    # -----------------------------------------------------------------------------------------------------------------
    # Display the program guide
    # -----------------------------------------------------------------------------------------------------------------
    if args.guide or not args.module:
        if not args.module:
            with open(PUtils.readme, 'r') as f:
                print(f.read(), end='')
        elif args.module == PUtils.analysis:
            metrics_description = [(metric, attributes['description'])
                                   for metric, attributes in sorted(master_metrics.items())]
            formatted_metrics = SDUtils.pretty_format_table(metrics_description)
            logger.info('After running \'%s analysis\', the following metrics will be available for each pose '
                        '(unique design configuration) selected for analysis:\n\t%s\n\nAdditionally, you can view the '
                        'pose specific files [pose-id]_Trajectory.csv for comparison of different design trials for an '
                        'individual pose, and [pose-id]_Residues.csv for residue specific information over the various '
                        'trajectories. Usage of overall pose metrics '
                        'across all poses should facilitate selection of the best configurations to move forward'
                        ' with, while Trajectory and Residue information can inform your choice of sequence selection '
                        'parameters. Selection of the resulting poses'
                        ' can be accomplished through the %s module \'sequence_selection\'.\n\t'
                        '\'%s sequence_selection -h\' will get you started.'
                        % (PUtils.program_command, '\n\t'.join(formatted_metrics), PUtils.program_name,
                           PUtils.program_command))
        elif args.module == PUtils.interface_design:
            logger.info()
        elif args.module == PUtils.nano:
            logger.info()
        elif args.module == 'expand_asu':
            logger.info()
        elif args.module == PUtils.select_designs:
            logger.info()
        elif args.module == 'sequence_selection':
            logger.info()
        exit()
    # -----------------------------------------------------------------------------------------------------------------
    # Process additional flags
    # -----------------------------------------------------------------------------------------------------------------
    default_flags = Flags.return_default_flags(args.module)
    if additional_flags:
        formatted_flags = format_additional_flags(additional_flags)
    else:
        formatted_flags = dict()
        extra_flags = None
        # Todo remove/modify
        #  This serves to pass additional arguments to NanohedraWrap. it does so through a list of args. Not very
        #  compatible with the above parsing
    default_flags.update(formatted_flags)

    # Add additional program flags to queried_flags
    queried_flags = vars(args)
    queried_flags.update(default_flags)
    queried_flags.update(Flags.process_residue_selector_flags(queried_flags))
    # We have to ensure that if the user has provided it, the symmetry is correct
    if queried_flags.get('sym_entry'):
        queried_flags['sym_entry'] = SymEntry(int(queried_flags['sym_entry']))
    if queried_flags['symmetry']:
        if queried_flags['symmetry'] in SDUtils.possible_symmetries:
            queried_flags['sym_entry'] = SDUtils.parse_symmetry_to_sym_entry(queried_flags['symmetry'])
        elif queried_flags['symmetry'].lower()[:5] == 'cryst':
            queried_flags['symmetry'] = 'cryst'
            # the symmetry information should be in the pdb headers
        else:
            raise SDUtils.DesignError('The symmetry \'%s\' is not supported! Supported symmetries include:'
                                      '\n\t%s\nCorrect your flags and try again'
                                      % (queried_flags['symmetry'], ', '.join(SDUtils.possible_symmetries)))

    # TODO consolidate this check
    if args.module in [PUtils.interface_design, PUtils.generate_fragments, 'orient', 'find_asu', 'expand_asu',
                       'interface_metrics', 'custom_script', 'rename_chains', 'status']:
        initialize, construct_pose = True, True  # set up design directories
        if args.module in ['orient', 'expand_asu']:
            if queried_flags['nanohedra_output'] or queried_flags['symmetry']:
                queried_flags['output_assembly'] = True
            else:
                logger.critical('Cannot %s without providing symmetry! Provide symmetry with \'--symmetry\''
                                % args.module)
                exit(1)
    elif args.module in [PUtils.nano, PUtils.select_designs, PUtils.analysis, PUtils.cluster_poses,
                         'sequence_selection']:
        queried_flags[args.module] = True  # Todo what is this for? Analysis (No more) in DesignDirectory and ?
        initialize, construct_pose = True, False
        if args.module == PUtils.select_designs:
            if not args.debug:
                queried_flags['skip_logging'] = True  # automatically skip logging if opening a large number of files
            if not args.metric:
                initialize = False
        elif args.module == 'sequence_selection':
            if not args.global_sequences and args.number_sequences == sys.maxsize:
                args.number_sequences = 1
    else:  # ['distribute', 'query', 'guide', 'flags', 'residue_selector']
        initialize, construct_pose = False, False
        if args.module == 'query':
            args.directory = True

    if not args.guide and args.module not in ['distribute', 'query', 'guide', 'flags', 'residue_selector']:
        options_table = SDUtils.pretty_format_table(queried_flags.items())
        logger.info('Starting with options:\n\t%s' % '\n\t'.join(options_table))
    # -----------------------------------------------------------------------------------------------------------------
    # Grab all Designs (DesignDirectory) to be processed from either database, directory, project name, or file
    # -----------------------------------------------------------------------------------------------------------------
    all_poses, all_dock_directories, pdb_pairs, design_directories, location = None, None, None, None, None
    initial_iter = None
    low, high, low_range, high_range = None, None, None, None
    # nanohedra_initialization = False
    if not args.directory and not args.file and not args.project and not args.single:
        raise SDUtils.DesignError('No designs were specified!\nPlease specify --directory, --file, '
                                  '--project, or --single to locate designs of interest and run your command again')

    if args.multi_processing:
        # Calculate the number of threads to use depending on computer resources
        threads = SDUtils.calculate_mp_threads(cores=args.cores)  # mpi=args.mpi, Todo
        logger.info('Starting multiprocessing using %d threads' % threads)
    else:
        threads = 1
        logger.info('Starting processing. If single process is taking awhile, use -mp during submission')

    if initialize:  # Set up DesignDirectories
        nano = queried_flags.get('nanohedra_output', None)
        if nano:
            all_poses, location = SDUtils.collect_nanohedra_designs(files=args.file, directory=args.directory)
        else:
            all_poses, location = SDUtils.collect_designs(files=args.file, directory=args.directory,
                                                          project=args.project, single=args.single)
        if queried_flags['design_range']:
            low, high = map(float, queried_flags['design_range'].split('-'))
            low_range, high_range = int((low / 100) * len(all_poses)), int((high / 100) * len(all_poses))
            if low_range < 0 or high_range > len(all_poses):
                raise SDUtils.DesignError('The input --design_range is outside of the acceptable bounds [0-%d]'
                                          % len(all_poses))
            logger.info('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))

        if all_poses:  # TODO fetch a state from files that have already been SymDesigned...
            if all_poses[0].count('/') == 0:  # assume that we have received pose-IDs and process accordingly
                if nano:
                    queried_flags['sym_entry'] = get_sym_entry_from_nanohedra_directory(args.directory)
                design_directories = [DesignDirectory.from_pose_id(pose, nano=nano, root=args.directory,
                                                                   construct_pose=construct_pose, **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
            elif nano:
                base_directory = '/%s' % os.path.join(*all_poses[0].split(os.sep)[:-4])
                queried_flags['sym_entry'] = get_sym_entry_from_nanohedra_directory(base_directory)
                design_directories = [DesignDirectory.from_nanohedra(pose, construct_pose=construct_pose,
                                                                     **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
            else:
                design_directories = [DesignDirectory.from_file(pose, **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
        if not design_directories:
            raise SDUtils.DesignError('No SymDesign directories found within \'%s\'! Please ensure correct '
                                      'location. Are you sure you want to run with -%s %s?'
                                      % (location, 'nanohedra_output', queried_flags['nanohedra_output']))
        # Todo could make after collect_designs? Pass to all design_directories
        #  for file, take all_poses first file. I think prohibits multiple dirs, projects, single...
        master_directory = next(iter(design_directories))
        logger.info('Loading all resources in the current Database found in \'%s\'' % master_directory.protein_data)
        master_db = Database(master_directory.orient_dir, master_directory.orient_asu_dir, master_directory.refine_dir,
                             master_directory.stride_dir, master_directory.sequences, master_directory.profiles,
                             sql=None, log=logger)

        master_directory.make_path(master_directory.protein_data)
        master_directory.make_path(master_directory.pdbs)
        master_directory.make_path(master_directory.sequence_info)
        master_directory.make_path(master_directory.sequences)
        master_directory.make_path(master_directory.profiles)
        master_directory.make_path(master_directory.job_paths)
        master_directory.make_path(master_directory.sbatch_scripts)
        if queried_flags['nanohedra_output']:
            # for each design_directory, ensure that the pdb files used as source are present in the self.orient_dir
            stride_dir = master_directory.stride_dir
            orient_dir = master_directory.orient_dir
            orient_asu_dir = master_directory.orient_asu_dir
            refine_dir = master_directory.refine_dir
            # args.orient, args.refine = True, True  # Todo make part of argparse? Could be variables in NanohedraDB
            # if args.orient:
            master_directory.make_path(orient_dir)
            master_directory.make_path(orient_asu_dir)
            master_directory.make_path(stride_dir)
            logger.info('The required files for %s designs are being collected and oriented if necessary' % PUtils.nano)
            required_oligomers1 = set(design.oligomer_names[0] for design in design_directories)
            required_oligomers2 = set(design.oligomer_names[1] for design in design_directories)
            orient_files = [os.path.splitext(file)[0] for file in os.listdir(orient_dir)]
            qsbio_confirmed = SDUtils.unpickle(PUtils.qs_bio)
            orient_log = SDUtils.start_log(name='orient', handler=1)
            SDUtils.start_log(name='orient', handler=2, location=os.path.join(orient_dir, PUtils.orient_log_file))
            for idx, required_oligomers in enumerate([required_oligomers1, required_oligomers2], 1):
                symmetry = getattr(master_directory.sym_entry, 'group%d' % idx)
                logger.info('Ensuring PDB files are oriented with %s symmetry (stored at %s): %s'
                            % (symmetry, orient_dir, ', '.join(required_oligomers)))
                for oligomer in required_oligomers:
                    if oligomer in orient_files:
                        continue
                    biological_assemblies = qsbio_confirmed.get(oligomer)
                    if biological_assemblies:  # v first assembly in list
                        assembly = biological_assemblies[0]
                    else:
                        logger.warning('No confirmed biological assembly was found for %s. Using the first assembly'
                                       ' listed in the PDB' % oligomer)
                        assembly = 1
                    logger.debug('Fetching oligomer %s from PDB' % oligomer)
                    pdb_path = fetch_pdb_file('%s_%d' % (oligomer, assembly), out_dir=master_directory.pdbs, asu=False)
                    if pdb_path:
                        orient_file = orient_pdb_file(pdb_path, log=orient_log, sym=symmetry, out_dir=orient_dir)
                        if not orient_file:
                            # logger.error('Couln\'t orient the file %s' % pdb_path)
                            continue
                        # extract the asu from the oriented file for symmetric refinement
                        oriented_pdb = PDB.from_file(orient_file, log=None)
                        oriented_asu = oriented_pdb.entities[0]
                        oriented_asu.stride(to_file=os.path.join(stride_dir, '%s.stride' % oriented_asu.name))
                        oriented_asu.write(out_path=os.path.join(orient_asu_dir, '%s.pdb' % oriented_asu.name))
                    else:
                        logger.warning('Couldn\'t locate the .pdb file %s, there may have been an issue '
                                       'downloading it from the PDB. Attempting to copy from %s job data source'
                                       % (pdb_path, PUtils.nano))
                        raise SDUtils.DesignError('This functionality hasn\'t been written yet. Use the '
                                                  'canonical_pdb1/2 attribute of DesignDirectory to pull the'
                                                  'pdb file source.')
            # if args.refine:
            # later if sequence design is attempted, ensure all of these are present in the self.refine_dir
            # required_oligomers1 = set(design.oligomer_names[0] for design in design_directories)
            # required_oligomers2 = set(design.oligomer_names[1] for design in design_directories)
            oriented_asu_files = os.listdir(orient_asu_dir)
            master_directory.make_path(refine_dir)
            refine_files = os.listdir(refine_dir)
            oligomers_to_refine, sym_def_files = [], {}
            for idx, required_oligomers in enumerate([required_oligomers1, required_oligomers2], 1):
                symmetry = getattr(master_directory.sym_entry, 'group%d' % idx)
                sym_def_files[symmetry] = SDUtils.sdf_lookup(symmetry)
                for orient_asu_file in oriented_asu_files:  # iterating this way to forgo missing "missed orient"
                    base_pdb_code = os.path.splitext(orient_asu_file)[0]  # os.path.basename()
                    if base_pdb_code in required_oligomers and orient_asu_file not in refine_files:
                        oligomers_to_refine.append((os.path.join(orient_asu_dir, orient_asu_file), symmetry))
            set_oligomers_to_refine = set(oligomers_to_refine)
            while set_oligomers_to_refine:  # If no files found unrefined, we should proceed
                logger.info('The following oriented oligomers are not yet refined and are being set up for refinement'
                            ' into the Rosetta Scorefunction for optimized sequence design:\n\t%s'
                            % ', '.join([os.path.splitext(os.path.basename(file))[0]
                                         for file, sym in set_oligomers_to_refine]))
                refine_input = input('Would you like to refine them now? If you plan on performing sequence design '
                                     'on designs containing them, it is highly recommended you perform refinement. '
                                     'Indicate [y/n].%s' % input_string)
                if not bool_d[refine_input.lower()]:  # Todo make input crash proof
                    confirm = input('To confirm, asymmetric units are going to be generated with unrefined '
                                    'coordinates. Confirm \'y\' one more time to proceed regardless. '
                                    'Indicate [y/n].%s' % input_string)
                    if bool_d[confirm.lower()]:  # Todo make input crash proof
                        break
                else:
                    # generate sbatch refine command
                    # orient_files, symmetries = zip(*oligomers_to_refine)
                    flags_file = os.path.join(refine_dir, 'refine_flags')
                    if not os.path.exists(flags_file):
                        flags = copy.copy(rosetta_flags) + relax_flags
                        flags.extend(['-out:path:pdb %s' % refine_dir, '-no_scorefile true'])  # Todo test
                        flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
                        with open(flags_file, 'w') as f:
                            f.write('%s\n' % '\n'.join(flags))

                    refine_cmd = ['@%s' % flags_file, '-parser:protocol',
                                  os.path.join(PUtils.rosetta_scripts, '%s_oligomer.xml' % PUtils.stage[1]),
                                  '-parser:script_vars']
                    refine_cmds = \
                        [script_cmd + refine_cmd + ['sdf=%s' % sym_def_files[sym], '-in:file:s', orient_asu_file]
                         for orient_asu_file, sym in set_oligomers_to_refine]
                    commands_file = \
                        SDUtils.write_commands([subprocess.list2cmdline(cmd) for cmd in refine_cmds],
                                               name='refine_oligomers_%s' % timestamp, out_path=refine_dir)
                    refine_sbatch = \
                        distribute(file=commands_file, out_path=master_directory.program_root, scale='refine',
                                   max_jobs=int(len(refine_cmds) / 2 + 0.5), number_of_commands=len(refine_cmds))
                    print('\n' * 3)
                    logger.info('The located designs require preprocessing before design related modules can be'
                                ' used. Please follow the instructions below to refine your input files')
                    logger.critical('Ensure the below created SBATCH script is correct. Specifically, check that the '
                                    'job array and any node specifications are accurate. You can look at the SBATCH '
                                    'manual (man sbatch or sbatch --help) to understand the variables or ask for help '
                                    'if you are still unsure.')
                    logger.info('Once you are satisfied, enter the following to distribute jobs:\n\t%s'
                                % 'sbatch %s' % refine_sbatch)
                    logger.info('After completion of the refinement sbatch script, re-run your %s command:'
                                '\n\t%s\nto finish set up of the designs of interest.'
                                % (PUtils.program_name, ' '.join(sys.argv)))
                    terminate(args.module, design_directories, output=False)
                    # break
                    # The next time this directory is initialized, there will be no refine files left hopefully
                    # then this while loop won't be triggered and DesignDirectory initialization will proceed
            # ensure nanohedra_initialization
            # args.module = 'nanohedra_initialization'
            # nanohedra_initialization = False
            # nanohedra_initialization = True

        if args.multi_processing:  # Todo tweak behavior of these two parameters. Need Queue based DesignDirectory
            master_db.load_all_data()
            # SDUtils.mp_map(DesignDirectory.set_up_design_directory, design_directories, threads=threads)
            # SDUtils.mp_map(DesignDirectory.link_master_database, design_directories, threads=threads)
        # else:  # for now just do in series
        for design in design_directories:
            design.link_master_database(master_db)
            design.set_up_design_directory()

        logger.info('%d unique poses found in \'%s\'' % (len(design_directories), location))
        if not args.debug and not queried_flags['skip_logging']:
            example_log = getattr(design_directories[0].log.handlers[0], 'baseFilename', None)
            if example_log:
                logger.info('All design specific logs are located in their corresponding directories.\n\tEx: %s'
                            % example_log)

    elif args.module == PUtils.nano:  # Todo consolidate this operation with above and nano orient
        # Getting PDB1 and PDB2 File paths
        if args.pdb_path1:
            if not args.entry:
                logger.critical('If using --pdb_path1 (-d1) and/or --pdb_path2 (-d2), please specify --entry as '
                                'well. --entry can be found using the module \'%s query\'' % PUtils.program_command)
                exit()
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
            orient_log = SDUtils.start_log(name='orient', handler=2,
                                           location=os.path.join(os.path.dirname(args.pdb_path1),
                                                                 PUtils.orient_log_file))
            pdb1_oriented_filepaths = [orient_pdb_file(pdb_path, log=orient_log, sym=oligomer_symmetry_1,
                                                       out_dir=oriented_pdb1_out_dir)
                                       for pdb_path in pdb1_filepaths]
            logger.info('%d filepaths found' % len(pdb1_oriented_filepaths))
            # pdb1_oriented_filepaths = filter(None, pdb1_oriented_filepaths)

            if args.pdb_path2:
                if args.pdb_path1 != args.pdb_path2:
                    oriented_pdb2_out_dir = os.path.join(os.path.dirname(args.pdb_path2),
                                                         '%s_oriented_with_%s_symmetry'
                                                         % (os.path.basename(args.pdb_path2), oligomer_symmetry_2))
                    if not os.path.exists(oriented_pdb2_out_dir):
                        os.makedirs(oriented_pdb2_out_dir)

                    if '.pdb' in args.pdb_path2:
                        pdb2_filepaths = [args.pdb_path2]
                    else:
                        pdb2_filepaths = SDUtils.get_all_pdb_file_paths(args.pdb_path2)
                    pdb2_oriented_filepaths = [orient_pdb_file(pdb_path, log=orient_log, sym=oligomer_symmetry_2,
                                                               out_dir=oriented_pdb2_out_dir)
                                               for pdb_path in pdb2_filepaths]

                    pdb_pairs = list(product(filter(None, pdb1_oriented_filepaths),
                                             filter(None, pdb2_oriented_filepaths)))
                    # pdb_pairs = list(product(pdb1_oriented_filepaths, pdb2_oriented_filepaths))
                    # pdb_pairs = list(product(SDUtils.get_all_pdb_file_paths(oriented_pdb1_out_dir),
                    #                          SDUtils.get_all_pdb_file_paths(oriented_pdb2_out_dir)))
                    location = '%s & %s' % (args.pdb_path1, args.pdb_path2)
            else:
                pdb_pairs = list(combinations(filter(None, pdb1_oriented_filepaths), 2))
                # pdb_pairs = list(combinations(pdb1_oriented_filepaths, 2))
                # pdb_pairs = list(combinations(SDUtils.get_all_pdb_file_paths(oriented_pdb1_out_dir), 2))
                location = args.pdb_path1
            initial_iter = [False for _ in range(len(pdb_pairs))]
            initial_iter[0] = True
            design_directories = pdb_pairs  # for logging purposes below Todo combine this with pdb_pairs variable
        elif args.directory or args.file:
            all_dock_directories, location = SDUtils.collect_nanohedra_designs(files=args.file,
                                                                               directory=args.directory, dock=True)
            design_directories = [DesignDirectory.from_nanohedra(dock_dir, mode=args.directory_type,
                                                                 project=args.project, **queried_flags)
                                  for dock_dir in all_dock_directories]
            if len(design_directories) == 0:
                raise SDUtils.DesignError('No docking directories/files were found!\n'
                                          'Please specify --directory1, and/or --directory2 or --directory or '
                                          '--file. See %s' % PUtils.help(args.module))
        master_directory = next(iter(design_directories))
        logger.info('%d unique building block docking combinations found in \'%s\''
                    % (len(design_directories), location))
    else:
        # raise SDUtils.DesignError('This logic is impossible?!')
        pass  # this logic is possible with select_designs without --metric

    if args.module in [PUtils.nano, PUtils.interface_design]:
        if args.run_in_shell:
            logger.info('Modelling will occur in this process, ensure you don\'t lose connection to the shell!')
        else:
            logger.info('Writing modelling commands out to file, no modelling will occur until commands are executed.')

    if queried_flags.get(Flags.generate_frags, None) or args.module == PUtils.generate_fragments \
            or queried_flags.get('design_with_fragments', None):
        interface_type = 'biological_interfaces'  # Todo parameterize
        logger.info('Initializing %s FragmentDatabase\n' % interface_type)
        fragment_db = SDUtils.unpickle(PUtils.biological_fragment_db_pickle)
        # fragment_db = FragmentDatabase(source=interface_type, init_db=True)  # Todo sql=args.frag_db
        euler_lookup = EulerLookup()
        for design in design_directories:
            design.connect_db(frag_db=fragment_db)
            design.euler_lookup = euler_lookup

    # -----------------------------------------------------------------------------------------------------------------
    # Ensure all Nanohedra Directories are set up by performing required transformation, then saving the pose
    # if nanohedra_initialization:
    #     if args.multi_processing:
    #         results = SDUtils.mp_map(DesignDirectory.load_pose, design_directories, threads=threads)
    #     else:
    #         for design in design_directories:
    #             design.load_pose()
    # -----------------------------------------------------------------------------------------------------------------
    # Parse SubModule specific commands
    # -----------------------------------------------------------------------------------------------------------------
    results, success, exceptions = [], [], []
    # ---------------------------------------------------
    if args.module == 'query':
        query_flags = [__file__, '-query'] + additional_flags
        logger.debug('Query %s.py with: %s' % (PUtils.nano.title(), ', '.join(query_flags)))
        query_mode(query_flags)
    # ---------------------------------------------------
    elif args.module == 'flags':
        if args.template:
            Flags.query_user_for_flags(template=True)
        else:
            Flags.query_user_for_flags(mode=args.flags_module)
    # ---------------------------------------------------
    elif args.module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
        distribute(**vars(args))
    # ---------------------------------------------------
    elif args.module == 'residue_selector':  # Todo
        if not args.single:
            raise SDUtils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
                                      'residue_selector' % (PUtils.program_name, PUtils.program_command))
        fasta_file = generate_sequence_template(args.single)
        logger.info('The residue_selector template was written to %s. Please edit this file so that the '
                    'residue_selector can be generated for protein design. Selection should be formatted as a \'*\' '
                    'replaces all sequence of interest to be considered in design, while a Mask should be formatted as '
                    'a\'-\'. Ex:\n>pdb_template_sequence\nMAGHALKMLV...\n>residue_selector\nMAGH**KMLV\n\nor'
                    '\n>pdb_template_sequence\nMAGHALKMLV...\n>design_mask\nMAGH----LV\n'
                    % fasta_file)
    # ---------------------------------------------------
    elif args.module == 'orient':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.orient, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.orient())

        terminate(args.module, design_directories, location=location, results=results)
    # ---------------------------------------------------
    elif args.module == 'find_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.find_asu, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.find_asu())

        terminate(args.module, design_directories, location=location, results=results)
    # ---------------------------------------------------
    elif args.module == 'expand_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.expand_asu, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.expand_asu())

        terminate(args.module, design_directories, location=location, results=results)
    # ---------------------------------------------------
    elif args.module == 'rename_chains':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.rename_chains, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.rename_chains())

        terminate(args.module, design_directories, location=location, results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.nano:  # -d1 pdb_path1, -d2 pdb_path2, -e entry, -o outdir
        # Initialize docking procedure
        if args.multi_processing:
            if args.run_in_shell:
                # TODO implementation where SymDesignControl calls Nanohedra.py
                logger.error('Can\'t run %s.py docking from here yet. Must pass python %s -c for execution'
                             % (PUtils.nano, __file__))
                exit(1)
            else:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    zipped_args = zip(repeat(args.entry), *zip(*pdb_pairs), repeat(args.outdir), repeat(extra_flags),
                                      repeat(args.project), initial_iter)
                    results = SDUtils.mp_starmap(nanohedra_command, zipped_args, threads=threads)
                else:  # args.directory or args.file set up docking directories
                    zipped_args = zip(design_directories, repeat(args.project))
                    results = SDUtils.mp_starmap(nanohedra_design_recap, zipped_args, threads=threads)
        else:
            if args.run_in_shell:
                logger.error('Can\'t run %s.py docking from here yet. Must pass python %s -c for execution'
                             % (PUtils.nano, __file__))
                exit(1)
            else:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    for initial, (path1, path2) in zip(initial_iter, pdb_pairs):
                        result = nanohedra_command(args.entry, path1, path2, args.outdir, extra_flags, args.project,
                                                   initial)
                        results.append(result)
                else:  # single directory docking (already made directories)
                    for dock_directory in design_directories:
                        result = nanohedra_design_recap(dock_directory, args.project)
                        results.append(result)

        terminate(args.module, design_directories, location=args.directory, results=results, output=False)
        #                                          location=location,
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
        #     logger.info('All \'%s\' commands were written to \'%s\'' % (PUtils.nano, command_file))
        # else:
        #     logger.error('No \'%s\' commands were written!' % PUtils.nano)
    # ---------------------------------------------------
    elif args.module == PUtils.generate_fragments:  # -i fragment_library
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.generate_interface_fragments, design_directories, threads=threads)
        else:
            for design in design_directories:
                results.append(design.generate_interface_fragments())

        terminate(args.module, design_directories, location=location, results=results)

    # ---------------------------------------------------
    elif args.module == 'interface_metrics':
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.force_flags), repeat(queried_flags.get('development')))
            results = SDUtils.mp_starmap(DesignDirectory.rosetta_interface_metrics, zipped_args, threads=threads)
        else:
            for design in design_directories:
                results.append(design.rosetta_interface_metrics(force_flags=args.force_flags,
                                                                development=queried_flags.get('development')))

        terminate(args.module, design_directories, location=location, results=results)

    # ---------------------------------------------------
    elif args.module == 'custom_script':
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.script), repeat(args.force_flags),
                              repeat(args.file_list), repeat(args.native), repeat(args.suffix), repeat(args.score_only),
                              repeat(args.variables))
            results = SDUtils.mp_starmap(DesignDirectory.custom_rosetta_script, zipped_args, threads=threads)
        else:
            for design in design_directories:
                results.append(design.custom_rosetta_script(args.script, force_flags=args.force_flags,
                                                            file_list=args.file_list, native=args.native,
                                                            suffix=args.suffix, score_only=args.score_only,
                                                            variables=args.variables))

        terminate(args.module, design_directories, location=location, results=results)

    # ---------------------------------------------------
    elif args.module == PUtils.interface_design:  # -i fragment_library, -s scout
        # if args.mpi:  # Todo implement
        #     # extras = ' mpi %d' % CommmandDistributer.mpi
        #     logger.info(
        #         'Setting job up for submission to MPI capable computer. Pose trajectories run in parallel,'
        #         ' %s at a time. This will speed up pose processing ~%f-fold.' %
        #         (CommmandDistributer.mpi - 1, PUtils.nstruct / (CommmandDistributer.mpi - 1)))
        #     queried_flags.update({'mpi': True, 'script': True})
        if queried_flags['design_with_evolution']:
            if psutil.virtual_memory().available <= hhblits_memory_threshold:
                logger.critical('The amount of virtual memory for the computer is insufficient to run hhblits '
                                '(the backbone of -design_with_evolution)! Please allocate the job to a computer with'
                                'more memory or the process will fail. Otherwise, select -design_with_evolution False')
            master_directory.make_path(master_directory.sequences)
            master_directory.make_path(master_directory.profiles)
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.interface_design, design_directories, threads=threads)
        else:
            for design in design_directories:
                results.append(design.interface_design())

        terminate(args.module, design_directories, location=location, results=results)

        # if not args.run_in_shell and len(success) > 0:  # any(success): ALL success are None type
        #     design_name = os.path.basename(next(iter(design_directories)).project_designs)
        #     program_root = next(iter(design_directories)).program_root
        #     all_commands = [[] for s in PUtils.stage_f]
        #     command_files = [[] for s in PUtils.stage_f]
        #     sbatch = [[] for s in PUtils.stage_f]
        #     for des_directory in design_directories:
        #         for idx, stage in enumerate(PUtils.stage_f, 1):
        #             if idx > 3:  # No analysis or higher
        #                 break
        #             all_commands[idx].append(os.path.join(des_directory.scripts, '%s.sh' % stage))
        #     for idx, stage in enumerate(PUtils.stage_f, 1):  # 1 - refine, 2 - design, 3 - metrics
        #         if idx > 3:  # No analysis or higher
        #             break
        #         command_files[idx] = SDUtils.write_commands(all_commands[idx], name='%s_%s' % (stage, design_name)
        #                                                     , out_path=program_root)
        #         sbatch[idx] = distribute(stage=stage, directory=program_root, file=command_files[idx])
        #         # logger.info('All \'%s\' commands were written to \'%s\'' % (stage, sbatch[idx]))
        #
        #     logger.info('\nTo process all commands in correct order, execute:\n\t%s' %
        #                 ('\n\t'.join('sbatch %s' % sbatch[idx]
        #                              for idx, stage in enumerate(list(PUtils.stage_f.keys())[:3], 1))))
        #     print('\n' * 5)
        # WHEN ONE FILE RUNS ALL THREE MODES
        # all_commands = []
        # for des_directory in design_directories:
        #     all_commands.append(os.path.join(des_directory.scripts, '%s.sh' % PUtils.interface_design))
        # command_file = SDUtils.write_commands(all_commands, name=PUtils.interface_design, out_path=args.directory)
        # args.success_file = None
        # args.failure_file = None
        # args.max_jobs = 80
        # TODO add interface_design to PUtils.stage_f
        # distribute(stage=PUtils.interface_design, directory=args.directory, file=command_file,
        #            success_file=args.success_file, failure_file=args.success_file, max_jobs=args.max_jobs)
        # logger.info('All \'%s\' commands were written to \'%s\'' % (PUtils.interface_design, command_file))
    # ---------------------------------------------------
    elif args.module == PUtils.analysis:  # -o output, -f figures, -n no_save, -j join
        if args.no_save:
            save = False
        else:
            save = True
        master_directory = next(iter(design_directories))
        # ensure analysis write directory exists
        master_directory.make_path(master_directory.all_scores)
        # Start pose analysis of all designed files
        if len(args.output.split('/')) > 1:  # the path is a full or relative path, we should use it
            out_path = args.output
        else:
            out_path = os.path.join(master_directory.program_root, args.output)

        if os.path.exists(out_path):
            logger.critical('The specified output file \'%s\' already exists, this will overwrite your old analysis '
                            'data! Please modify that file or specify a new output name with -o/--output'
                            % out_path)
            exit(1)
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.join), repeat(save), repeat(args.figures))
            results = SDUtils.mp_starmap(DesignDirectory.design_analysis, zipped_args, threads=threads)
        else:
            for design in design_directories:
                results.append(design.design_analysis(merge_residue_data=args.join, save_trajectories=save,
                                                      figures=args.figures))

        terminate(args.module, design_directories, location=location, results=results)
    # ---------------------------------------------------
    # elif args.module == 'merge':  # -d2 directory2, -f2 file2, -i increment, -F force
    #     directory_pairs, failures = None, None
    #     if args.directory2 or args.file2:
    #         # Grab all poses (directories) to be processed from either directory name or file
    #         all_poses2, location2 = SDUtils.collect_designs(file=args.file2, directory=args.directory2)
    #         assert all_poses2 != list(), logger.critical(
    #             'No %s.py directories found within \'%s\'! Please ensure correct location' % (PUtils.nano.title(),
    #                                                                                           location2))
    #         all_design_directories2 = set_up_directory_objects(all_poses2)
    #         logger.info('%d Poses found in \'%s\'' % (len(all_poses2), location2))
    #         if args.merge_mode == PUtils.interface_design:
    #             directory_pairs, failures = pair_directories(all_design_directories2, design_directories)
    #         else:
    #             logger.warning('Source location was specified, but the --directory_type isn\'t design. Destination '
    #                            'directory will be ignored')
    #     else:
    #         if args.merge_mode == PUtils.interface_design:
    #             exit('No source location was specified! Use -d2 or -f2 to specify the source of poses when merging '
    #                  'design directories')
    #         elif args.merge_mode == PUtils.nano:
    #             directory_pairs, failures = pair_dock_directories(design_directories)  #  all_dock_directories)
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
    #         zipped_args = zip(design_directories, repeat(args.increment))
    #         results = SDUtils.mp_starmap(rename, zipped_args, threads=threads)
    #         results2 = SDUtils.mp_map(merge_design_pair, directory_pairs, threads)
    #     else:
    #         for des_directory in design_directories:
    #             rename(des_directory, increment=args.increment)
    #         for directory_pair in directory_pairs:
    #             merge_design_pair(directory_pair)
    # ---------------------------------------------------
    elif args.module == PUtils.select_designs:
        # -df dataframe, -f filter, -m metric, -p pose_design_file, -s selection_string, -w weight
        # program_root = next(iter(design_directories)).program_root
        # if not args.directory or not args.project or not args.single:  Todo
        if not args.directory:
            logger.critical('If using a --dataframe for selection, you must include the directory where the designs are'
                            'located in order to properly select designs. Please specify -d/--directory on the command '
                            'line')
            # logger.critical('If using a --dataframe for selection, you must include the directory where the designs are'
            #                 'located in order to properly select designs. Please specify -d/--directory, -p/--project, '
            #                 'or -s/--single on the command line') TODO
            exit(1)
            program_root = None
            # Todo change this mechanism so not reliant on args.directory and outputs pose IDs/ Alternatives fix csv
            #  to output paths
        else:
            program_root = args.directory

        if args.pose_design_file:
            # Grab all poses (directories) to be processed from either directory name or file
            with open(args.pose_design_file) as csv_file:
                csv_lines = [line for line in reader(csv_file)]
            all_poses, pose_design_numbers = zip(*csv_lines)

            design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
                                  for pose in all_poses]
            # design_directories = set_up_directory_objects(all_poses, project=args.project)  # **queried_flags
            results.append(zip(design_directories, pose_design_numbers))
            location = args.pose_design_file
        elif args.dataframe:
            # Figure out poses from a dataframe, filters, and weights. Returns pose id's
            selected_poses_df = filter_pose(args.dataframe, filter=args.filter, weight=args.weight)
            selected_poses = selected_poses_df.index.to_list()
            logger.info('%d poses were selected' % len(selected_poses_df))  # :\n\t%s , '\n\t'.join(selected_poses)))
            if args.filter or args.weight:
                new_dataframe = os.path.join(args.directory, '%s%sDesignPoseMetrics-%s.csv'
                                             % ('Filtered' if args.weight else '', 'Weighted' if args.weight else '',
                                                timestamp))
                selected_poses_df.to_csv(new_dataframe)
                logger.info('New DataFrame was written to %s' % new_dataframe)

            # Sort results according to clustered poses if clustering exists  # Todo parameterize name
            if args.cluster_map:
                cluster_map = args.cluster_map
            else:
                # cluster_map = os.path.join(next(iter(design_directories)).protein_data, '%s.pkl' % PUtils.clustered_poses)
                cluster_map = os.path.join(program_root, PUtils.data.title(), '%s.pkl' % PUtils.clustered_poses)
            if os.path.exists(cluster_map):
                cluster_representative_pose_member_map = SDUtils.unpickle(cluster_map)
            else:
                logger.info('No cluster pose map was found at %s. Clustering similar poses may eliminate redundancy '
                            'from the final design selection. To cluster poses broadly, run \'%s %s\''
                            % (cluster_map, PUtils.program_command, PUtils.cluster_poses))
                while True:
                    confirm = input('Would you like to %s on the subset of designs (%d) located so far? [y/n]%s'
                                    % (PUtils.cluster_poses, len(selected_poses), input_string))
                    if confirm.lower() in bool_d:
                        break
                    else:
                        print('%s %s is not a valid choice!' % (invalid_string, confirm))
                if bool_d[confirm.lower()] or confirm.isspace():  # the user wants to separate poses
                    if len(selected_poses) > 1000:
                        queried_flags['skip_logging'] = True
                    design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
                                          for pose in selected_poses]
                    compositions = group_compositions(design_directories)
                    if args.multi_processing:
                        results = SDUtils.mp_map(cluster_designs, compositions.values(), threads=threads)
                        cluster_representative_pose_member_map = {}
                        for result in results:
                            cluster_representative_pose_member_map.update(result.items())
                    else:
                        cluster_representative_pose_member_map = {}
                        for composition_group in compositions.values():
                            cluster_representative_pose_member_map.update(cluster_designs(composition_group))
                    # cluster_representative_pose_member_string_map = \
                    #     {str(representative): str(member)
                    #      for representative, members in cluster_representative_pose_member_map.items()
                    #      for member in members}
                    pose_cluster_file = SDUtils.pickle_object(cluster_representative_pose_member_map,
                                                              PUtils.clustered_poses % (location, timestamp),
                                                              out_path=next(iter(design_directories)).protein_data)
                    logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                                % (len(cluster_representative_pose_member_map), len(design_directories),
                                   pose_cluster_file))
                else:
                    cluster_representative_pose_member_map = {}

            if cluster_representative_pose_member_map:
                # {design_string: [design_string, ...]} where key is representative, values are matching designs
                # OLD -> {composition: {design_string: cluster_representative}, ...}
                pose_cluster_membership_map = invert_cluster_map(cluster_representative_pose_member_map)
                pose_clusters_found, pose_not_found = {}, []
                for idx, pose in enumerate(selected_poses):
                    cluster_membership = pose_cluster_membership_map.get(pose, None)
                    if cluster_membership:
                        if cluster_membership not in pose_clusters_found:  # include as this pose hasn't been identified
                            pose_clusters_found[cluster_membership] = [pose]
                        else:  # This cluster has already been found and it was identified again. Report and only
                            # include the highest ranked pose in the output as it provides info on all occurrences
                            pose_clusters_found[cluster_membership].append(pose)
                    else:
                        pose_not_found.append(pose)

                # Todo report the clusters and the number of instances
                final_poses = [members[0] for members in pose_clusters_found.values()]
                if pose_not_found:
                    logger.warning('Couldn\'t locate the following poses:\n\t%s\nWas %s only run on a subset of the '
                                   'poses that were selected in %s? Adding all of these to your final poses...'
                                   % ('\n\t'.join(pose_not_found), PUtils.cluster_poses, args.dataframe))
                    final_poses.extend(pose_not_found)
                logger.info('Found %d poses after clustering' % len(final_poses))
            else:
                logger.info('Grabbing all selected poses.')
                final_poses = selected_poses

            if args.number_poses and len(final_poses) > args.number_poses:
                final_poses = final_poses[:args.number_poses]
                logger.info('Found %d poses after applying your number_of_poses selection criteria' % len(final_poses))

            if len(final_poses) > 1000:
                queried_flags['skip_logging'] = True
            design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
                                  for pose in final_poses]

            for design in design_directories:
                design.set_up_design_directory()
            location = program_root
            # write out the chosen poses to a pose.paths file
            terminate(args.module, design_directories, location=location, results=design_directories)
        else:
            logger.debug('Collecting designs to sort')
            if args.metric == 'score':
                metric_design_dir_pairs = [(des_dir.score, des_dir.path) for des_dir in design_directories]
            elif args.metric == 'fragments_matched':
                metric_design_dir_pairs = [(des_dir.number_of_fragments, des_dir.path)
                                           for des_dir in design_directories]
            else:
                raise SDUtils.DesignError('The metric \'%s\' is not supported!' % args.metric)

            logger.debug('Sorting designs according to \'%s\'' % args.metric)
            metric_design_dir_pairs = [(score, path) for score, path in metric_design_dir_pairs if score]
            sorted_metric_design_dir_pairs = sorted(metric_design_dir_pairs, key=lambda pair: (pair[0] or 0),
                                                    reverse=True)
            logger.info('Top ranked Designs according to %s:\n\t%s\tDesign\n\t%s'
                        % (args.metric, args.metric.title(),
                           '\n\t'.join('%.2f\t%s' % tup for tup in sorted_metric_design_dir_pairs[:7995])))
            # Todo write all to file
            if len(design_directories) > 7995:
                logger.info('Top ranked Designs cutoff at 7995')
    # ---------------------------------------------------
    elif args.module == PUtils.cluster_poses:
        # First, identify the same compositions
        compositions = group_compositions(design_directories)

        if args.multi_processing:
            results = SDUtils.mp_map(cluster_designs, compositions.values(), threads=threads)
            cluster_representative_pose_member_map = {}
            for result in results:
                cluster_representative_pose_member_map.update(result.items())
        else:
            # pose_map = pose_rmsd_s(design_directories)
            # cluster_representative_pose_member_map = cluster_poses(pose_map)
            cluster_representative_pose_member_map = {}
            for composition_group in compositions.values():
                cluster_representative_pose_member_map.update(cluster_designs(composition_group))

        if args.output:
            pose_cluster_file = SDUtils.pickle_object(cluster_representative_pose_member_map, args.output, out_path='')
        else:
            pose_cluster_file = SDUtils.pickle_object(cluster_representative_pose_member_map,
                                                      PUtils.clustered_poses % (location, timestamp),
                                                      out_path=next(iter(design_directories)).protein_data)
        logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                    % (len(cluster_representative_pose_member_map), len(design_directories), pose_cluster_file))
        logger.info('To utilize the clustering, perform %s and cluster analysis will be applied to the poses to select '
                    'the cluster representative.' % PUtils.select_designs)
        # for protein_pair in pose_map:
        #     if os.path.basename(protein_pair) == '4f47_4grd':
        #     logger.info('\n'.join(['%s\n%s' % (pose1, '\n'.join(['%s\t%f' %
        #                                                          (pose2, pose_map[protein_pair][pose1][pose2])
        #                                                          for pose2 in pose_map[protein_pair][pose1]]))
        #                            for pose1 in pose_map[protein_pair]]))
    # --------------------------------------------------- # TODO v move to AnalyzeMutatedSequence.py
    elif args.module == 'sequence_selection':  # -c consensus, -f filters, -n number
        # master_directory = next(iter(design_directories))
        program_root = master_directory.program_root
        # if args.pose_design_file:        # -s selection_string, -w weights
        #     # Grab all poses (directories) to be processed from either directory name or file
        #     with open(args.pose_design_file) as csv_file:
        #         csv_lines = [line for line in reader(csv_file)]
        #     all_poses, pose_design_numbers = zip(*csv_lines)
        #
        #     design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
        #                           for pose in all_poses]
        #     # design_directories = set_up_directory_objects(all_poses, project=args.project)  # **queried_flags
        #     results.append(zip(design_directories, pose_design_numbers))
        #     location = args.pose_design_file
        # else:
        #     # sequence_weights = None
        #     # # TODO moved all of this to 'filter_designs' module
        #     # if args.dataframe:  # Figure out poses from a dataframe, filters, and weights
        #     #     # TODO parameterize
        #     #     # if args.filters:
        #     #     #     exit('Vy made this and I am going to put in here!')
        #     #     # design_requirements = {'percent_int_area_polar': 0.2, 'buns_per_ang': 0.002}
        #     #     # crystal_means1 = {'int_area_total': 570, 'shape_complementarity': 0.63, 'number_hbonds': 5}
        #     #     # crystal_means2 = {'shape_complementarity': 0.63, 'number_hbonds': 5}
        #     #     # symmetry_requirements = crystal_means1
        #     #     # filters = {}
        #     #     # filters.update(design_requirements)
        #     #     # filters.update(symmetry_requirements)
        #     #     # if args.consensus:
        #     #     #     consensus_weights1 = {'interaction_energy_complex': 0.5, 'percent_fragment': 0.5}
        #     #     #     consensus_weights2 = {'interaction_energy_complex': 0.33, 'percent_fragment': 0.33,
        #     #     #                           'shape_complementarity': 0.33}
        #     #     #     filters = {'percent_int_area_polar': 0.2}
        #     #     #     weights = consensus_weights2
        #     #     # else:
        #     #     #     weights1 = {'protocol_energy_distance_sum': 0.25, 'shape_complementarity': 0.25,
        #     #     #                 'observed_evolution': 0.25, 'int_composition_diff': 0.25}
        #     #     #     # Used without the interface area filter
        #     #     #     weights2 = {'protocol_energy_distance_sum': 0.20, 'shape_complementarity': 0.20,
        #     #     #                 'observed_evolution': 0.20, 'int_composition_diff': 0.20, 'int_area_total': 0.20}
        #     #     #     weights = weights1
        #     #
        #     #     selected_poses = Ams.filter_pose(args.dataframe, filter=args.filter, weight=args.weight,
        #     #                                      consensus=args.consensus)
        #     #
        #     #     # Sort results according to clustered poses
        #     #     cluster_map = os.path.join(next(iter(design_directories)).protein_data, '%s.pkl' % PUtils.clustered_poses)
        #     #     if os.path.exists(cluster_map):
        #     #         pose_cluster_map = SDUtils.unpickle(cluster_map)
        #     #         # {composition: {design_string: cluster_representative}, ...}
        #     #         pose_clusters_found, final_poses = [], []
        #     #         # for des_dir in design_directories:
        #     #         for pose in selected_poses:
        #     #             if pose_cluster_map[pose.split('-')[0]][pose] not in pose_clusters_found:
        #     #                 pose_clusters_found.append(pose_cluster_map[pose.split('-')[0]][pose])
        #     #                 final_poses.append(pose)
        #     #         logger.info('Final poses after clustering:\n\t%s' % '\n\t'.join(final_poses))
        #     #     else:
        #     #         final_poses = selected_poses
        #     #
        #     #     if len(final_poses) > args.number_poses:
        #     #         final_poses = final_poses[:args.number_poses]
        #     #
        #     #     design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
        #     #                           for pose in final_poses]

        if args.weight:
            trajectory_df = pd.read_csv(master_directory.trajectories, index_col=0, header=[0])
            sequence_metrics = set(trajectory_df.columns.get_level_values(-1).to_list())
            sequence_weights = query_user_for_metrics(sequence_metrics, mode='weight', level='sequence')
        else:
            sequence_weights = None

        if args.global_sequences:
            all_dfs = [pd.read_csv(design.trajectories, index_col=0, header=[0]) for design in design_directories]
            # logger.info([design for idx, df in enumerate(all_dfs[:3]) for design in df.index.to_list() if design_directories[idx].name in design])
            for idx, df in enumerate(all_dfs):
                # all_dfs[idx] = df.drop([des for des in df.index.to_list() if design_directories[idx].name in des])
                df.drop([design for design in df.index.to_list() if design_directories[idx].name not in design],
                        inplace=True)
            # logger.info([df.index for df in all_dfs[:3]])
            df = pd.concat(all_dfs, keys=design_directories)  # must add the design directory string to each index
            # df.index = [' '.join(col).strip() for col in df.index.values]

            # logger.info(df.index[:5])
            design_list = rank_dataframe_by_metric_weights(df, weights=sequence_weights)
            number_chosen = 0
            results, selected_designs = [], []
            for design_directory, design in design_list:
                if design_directory not in selected_designs:
                    selected_designs.append(design_directory)
                    results.append((design_directory, design))
                    number_chosen += 1
                    if number_chosen == args.number_sequences:
                        break
        else:
            if args.multi_processing:
                # sequence_weights = {'buns_per_ang': 0.2, 'observed_evolution': 0.3, 'shape_complementarity': 0.25,
                #                     'int_energy_res_summary_delta': 0.25}
                zipped_args = zip(design_directories, repeat(sequence_weights), repeat(args.number_sequences),
                                  repeat(args.protocol))
                # result_mp = zip(*SDUtils.mp_starmap(Ams.select_sequences, zipped_args, threads))
                # returns [[], [], ...]
                result_mp = SDUtils.mp_starmap(DesignDirectory.select_sequences, zipped_args, threads)
                results = []
                for result in result_mp:
                    results.extend(result)
                # results - contains tuple of (DesignDirectory, design index) for each sequence
                # could simply return the design index then zip with the directory
            else:
                results = []
                for design in design_directories:
                    results.extend(design.select_sequences(weights=sequence_weights, number=args.number_sequences,
                                                           protocol=args.protocol))

        if not args.selection_string:
            args.selection_string = '%s_' % os.path.basename(os.path.splitext(location)[0])
        else:
            args.selection_string += '_'
        outdir = os.path.join(os.path.dirname(program_root), '%sSelectedDesigns' % args.selection_string)
        outdir_traj = os.path.join(outdir, 'Trajectories')
        outdir_res = os.path.join(outdir, 'Residues')
        logger.info('Relevant design files are being copied to the new directory: %s' % outdir)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(outdir_traj)
            os.makedirs(outdir_res)

        # Create new output of designed PDB's  # TODO attach the state to these files somehow for further SymDesign use
        for des_dir, design in results:
            # pose_des_dirs, design = zip(*pose)
            # for i, des_dir in enumerate(pose_des_dirs):
            file = glob('%s/*%s*' % (des_dir.designs, design))  # [i]))
            if not file:
                # add to exceptions
                exceptions.append((des_dir.path, 'No file found for \'%s/*%s*\'' % (des_dir.designs, design)))  # [i])))
                continue
            if not os.path.exists(os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design))):
                shutil.copy(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
                shutil.copy(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
                shutil.copy(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # try:
            #     # Create symbolic links to the output PDB's
            #     os.symlink(file[0], os.path.join(outdir, '%s_design_%s.pdb' % (str(des_dir), design)))  # [i])))
            #     os.symlink(des_dir.trajectories, os.path.join(outdir_traj, os.path.basename(des_dir.trajectories)))
            #     os.symlink(des_dir.residues, os.path.join(outdir_res, os.path.basename(des_dir.residues)))
            # except FileExistsError:
            #     pass

        # Format sequences for expression
        tag_specified_list = \
            list(map(int, map(str.translate, set(args.entity_specification.split(',')).difference(''),
                              SDUtils.digit_translate_table)))
        for _ in range(len(des_dir.pose.entities) - len(tag_specified_list)):
            tag_specified_list.append(0)
        if args.entity_specification == 'all':
            tag_index = [True for _ in des_dir.pose.entities]
            number_of_tags = len(des_dir.pose.entities)
        elif args.entity_specification == 'single':
            tag_index = [True for _ in des_dir.pose.entities]
            number_of_tags = 1
        elif args.entity_specification == 'none':
            tag_index = [False for _ in des_dir.pose.entities]
            number_of_tags = 0
        elif tag_specified_list:
            tag_index = [True if is_tag else False for is_tag in args.tag_specification]
            number_of_tags = sum(tag_specified_list)
        else:
            tag_index = [False for _ in des_dir.pose.entities]
            number_of_tags = None

        missing_tags = {}  # result: [True, True] for result in results
        tag_sequences, final_sequences, inserted_sequences = {}, {}, {}
        for des_dir, design in results:
            # pose_des_dirs, design = zip(*pose)
            # for i, des_dir in enumerate(pose_des_dirs):
            # coming in as (chain: seq}
            file = glob('%s/*%s*' % (des_dir.designs, design))  # [i]))
            if not file:
                logger.error('No file found for %s' % '%s/*%s*' % (des_dir.designs, design))  # [i]))
                continue
            design_pose = PDB.from_file(file[0])
            designed_atom_sequences = [entity.structure_sequence for entity in design_pose.entities]
            # v {chain: sequence, ...}
            # design_sequences = design_pose.atom_sequences

            # need the original pose chain identity
            # source_pose = PDB(file=des_dir.asu)  # Why can't I use design_sequences? localds quality!
            des_dir.load_pose()  # des_dir.source)
            des_dir.pose.pdb.reorder_chains()  # Do I need to modify chains?
            # source_pose = PDB.from_file(des_dir.source)  # Think this works the best
            # source_pose.reorder_chains()  # Do I need to modify chains?
            # source_pose.atom_sequences = AnalyzeMutatedSequences.get_pdb_sequences(source_pose)
            # source_seqres = {}
            # for source_entity in des_dir.pose.entities:
            #     source_entity.retrieve_info_from_api()
            #     source_entity.retrieve_sequence_from_api(entity_id=source_entity.name)
                # source_seqres[entity.chain_id] = entity.reference_sequence

            # for entity in source_pose.entities:
            #     entity.retrieve_sequence_from_api(entity_id=entity.name)
            #     entity.retrieve_info_from_api()
            #     source_seqres[entity.chain_id] = entity.reference_sequence
            # if not source_pose.sequences:
            # oligomers = [PDB.from_file(Pose.retrieve_pdb_file_path(pdb)) for pdb in pose_entities]
            # oligomers = [SDUtils.read_pdb(SDUtils.retrieve_pdb_file_path(pdb)) for pdb in pose_entities]
            # entity_chain_database_chain_mapping = {entity.chain_id: next(iter(entity.api_entry))
            #                                        for entity in des_dir.pose.entities}
            # entity_chain_database_chain_mapping = {entity.chain_id: next(iter(entity.api_entry))
            #                                      for entity in source_pose.entities}
            # print('SEQRES:\n%s' % '\n'.join(['%s - %s' % (chain, oligomer.sequences[chain])
            #                                  for oligomer in oligomers for chain in oligomer.chain_id_list]))

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
            # all_missing_residues_d = {chain: Ams.generate_mutations_from_seq(design_sequences[chain],
            #                                                                  seqres_pose.seqres_sequences[chain],
            #                                                                  offset=True, only_gaps=True)
            #                           for chain in design_sequences}

            # Find all gaps between the ATOM record and SEQRES (reference sequence)
            # all_missing_residues_d = \
            #     {source_entity: generate_mutations(source_entity.structure_sequence, source_entity.reference_sequence,
            #                                        only_gaps=True)
            #      for source_entity in des_dir.pose.entities}
            # all_missing_residues_d = {chain: generate_mutations(source_pose.atom_sequences[chain],
            #                                                     source_seqres[chain], offset=True, only_gaps=True)
            #                           for chain in source_seqres}
            # pose_insert_offset_d = Ams.pdb_to_pose_num(all_missing_residues_d)

            # print('Pre-pose numbering:\n%s' %
            #       '\n'.join(['%s - %s' % (chain, ', '.join([str(res) for res in all_missing_residues_d[chain]]))
            #                  for chain in all_missing_residues_d]))

            # pose_offset = pdb_to_pose_offset(source_seqres)
            # # Modify residue indices to pose numbering for every Entity after the first
            # all_missing_residues = \
            #     {source_entity: {residue + source_entity.offset: mutation for residue, mutation in mutations.items()}
            #      for source_entity, mutations in all_missing_residues_d.items()}
            # Have to modify residue indices to include prior inserts length into pose numbering
        # prior_offset = 0
        # all_missing_residues = {}
        # mutations = []
        # referenced_design_sequences = {}
        # # for source_entity, mutations in all_missing_residues_d.items():
        # # for source_entity in des_dir.pose.entities:
        # for idx, (source_entity, design_entity) in enumerate(zip(des_dir.pose.entities, design_pose.entities)):
        #     source_entity.retrieve_info_from_api()
        #     source_entity.retrieve_sequence_from_api(entity_id=source_entity.name)
        #     disorder = generate_mutations(source_entity.structure_sequence, source_entity.reference_sequence,
        #                                   only_gaps=True)
        #     # all_missing_residues[source_entity] = \
        #     #     {residue + source_entity.offset + prior_offset: mutation for residue, mutation in disorder.items()}
        #     indexed_disordered_residues = \
        #         {residue + source_entity.offset + prior_offset: mutation for residue, mutation in disorder.items()}
        #     # for residue, mutation in mutations.items():
        #     #     all_missing_residues[source_entity][residue + source_entity.offset + prior_offset] = mutation
        #     prior_offset += len(disorder)
        #     mutations.append(generate_mutations(source_entity.structure_sequence, design_entity.structure_sequence,
        #                                         offset=False))
        #     # insert the disordered residues into the design pose
        #     for residue_number, mutation in indexed_disordered_residues.items():
        #         logger.debug('Inserting %s into position %d on chain %s'
        #                      % (mutation['from'], residue_number, source_entity.chain_id))
        #         design_pose.insert_residue_type(mutation['from'], at=residue_number,
        #                                         chain=source_entity.chain_id)
        #     # find the offset using the structure sequence after insertion
        #     offset = find_orf_offset(design_entity.structure_sequence, mutations[idx])
        #     referenced_design_sequences[source_entity] = design_entity.structure_sequence[offset:]
            # Modify residue indices to include prior inserts length into pose numbering
            # all_missing_residues_d = {chain: {residue + pose_insert_offset_d[chain]: all_missing_residues_d[chain][residue]
            #                                   for residue in all_missing_residues_d[chain]}
            #                           for chain in all_missing_residues_d}
            # print('Post-pose numbering:\n%s' %
            #       '\n'.join(['%s - %s' % (chain, ', '.join([str(res) for res in all_missing_residues_d[chain]]))
            #                  for chain in all_missing_residues_d]))

            # designed_atom_sequences = [entity.structure_sequence for entity in design_pose.entities]
            # # Compare the source entity, reference sequence as reference to the designed sequence to get mutations
            # # returns the designed residue identities in the index of the reference sequence
            # # MAYBE THIS SHOULD BE PERFORMED ON DESIGN SEQUENCE AFTER INSERTION TO BYPASS DIFFICULTY WITH GAPPED ALIGNS
            # # mutations = \
            # #     {idx: generate_mutations(source_entity.reference_sequence, design_seq)
            # #      for idx, (source_entity, design_seq) in enumerate(zip(des_dir.pose.entities, designed_atom_sequences))}
            # mutations = [generate_mutations(source_entity.structure_sequence, design_seq, offset=False)
            #              for source_entity, design_seq in zip(des_dir.pose.entities, designed_atom_sequences)]

            # print('Mutations:\n%s' %
            #       '\n'.join(['%s - %s' % (chain, mutations[chain]) for chain in mutations]))
            # print('Design Sequences:\n%s' % '\n'.join(['%s - %s' % (chain, design_sequences[chain])
            #                                            for chain in design_sequences]))

            # Insert missing reference sequence records (residue in mutations 'from' key) into design Structure
            # for source_entity, missing_mutations in reversed(all_missing_residues.items()):
            # for source_entity, missing_mutations in all_missing_residues.items():
            #     # for residue_number, mutation in reversed(missing_mutations.items()):
            #     for residue_number, mutation in missing_mutations.items():
            #         logger.debug('Inserting %s into position %d on chain %s'
            #                      % (mutation['from'], residue_number, source_entity.chain_id))
            #         design_pose.insert_residue_type(mutation['from'], at=residue_number, chain=source_entity.chain_id)

            # Get modified sequence
            # design_pose.get_chain_sequences()
            # design_sequences_disordered = design_pose.atom_sequences
            # print('Disordered Insertions:\n%s' %
            #       '\n'.join(['%s - %s' % (chain, design_sequences_disordered[chain])
            #                  for chain in design_sequences_disordered]))

            # Next find the correct start MET using the modified (residue inserted) design sequence
            # coding_offset = {design_entity: find_orf_offset(design_entity.structure_sequence, mutations[idx])
            #                  for idx, design_entity in enumerate(design_pose.entities)}
            # print('Coding Offset:\n%s'
            #       % '\n'.join(['%s: %s' % (chain, coding_offset[chain]) for chain in coding_offset]))

            # Remove all residues preceding ORF start from the inserted design sequence
            # referenced_design_sequences = {design_entity: design_entity.structure_sequence[offset:]
            #                                for design_entity, offset in coding_offset.items()}
            # Next, find the correct ORF using the modified (residue inserted) design sequence and
            # remove residues preceding ORF start
            # referenced_design_sequences = {}
            # for idx, design_entity in enumerate(design_pose.entities):
            #     offset = find_orf_offset(design_entity.structure_sequence, mutations[idx])
            #     referenced_design_sequences[design_entity] = design_entity.structure_sequence[offset:]

            # referenced_design_sequences = {design_entity: design_entity.structure_sequence
            #                     for design_entity, offset in coding_offset.items()}
            # print('Pre-tag Sequences:\n%s' % '\n'.join([referenced_design_sequences[chain] for chain in referenced_design_sequences]))

            # for residue in all_missing_residues_d:
            #     if all_missing_residues_d[residue]['from'] == 'M':

            # for chain in gapped_residues_d:
            #     for residue in gapped_residues_d[chain]:
            missing_tags[(des_dir, design)] = [1 for _ in des_dir.pose.entities]
            prior_offset = 0
            # all_missing_residues = {}
            # mutations = []
            # referenced_design_sequences = {}
            sequences_and_tags = {}
            # for source_entity, mutations in all_missing_residues_d.items():
            # for source_entity in des_dir.pose.entities:
            for idx, (source_entity, design_entity) in enumerate(zip(des_dir.pose.entities, design_pose.entities)):
                source_entity.retrieve_info_from_api()
                source_entity.retrieve_sequence_from_api(entity_id=source_entity.name)
                sequence_id = '%s_%s' % (des_dir, source_entity.name)
                design_string = '%s_design_%s_%s' % (des_dir, design, source_entity.name)  # [i])), pdb_code)
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
                true_termini = [term for term, is_true in termini_availability.items() if is_true]
                logger.debug('The termini %s are available for tagging' % termini_availability)
                # Find sequence specifid attributes required for expression formatting
                disorder = generate_mutations(source_entity.structure_sequence, source_entity.reference_sequence,
                                              only_gaps=True)
                # all_missing_residues[source_entity] = \
                #     {residue + source_entity.offset + prior_offset: mutation for residue, mutation in disorder.items()}
                indexed_disordered_residues = \
                    {residue + source_entity.offset + prior_offset: mutation for residue, mutation in disorder.items()}
                # for residue, mutation in mutations.items():
                #     all_missing_residues[source_entity][residue + source_entity.offset + prior_offset] = mutation
                prior_offset += len(disorder)
                # mutations.append(generate_mutations(source_entity.structure_sequence, design_entity.structure_sequence,
                #                                     offset=False))
                mutations = \
                    generate_mutations(source_entity.structure_sequence, design_entity.structure_sequence, offset=False)
                # insert the disordered residues into the design pose
                for residue_number, mutation in indexed_disordered_residues.items():
                    logger.debug('Inserting %s into position %d on chain %s'
                                 % (mutation['from'], residue_number, source_entity.chain_id))
                    design_pose.insert_residue_type(mutation['from'], at=residue_number,
                                                    chain=source_entity.chain_id)
                # find the offset using the structure sequence after insertion
                # offset = find_orf_offset(design_entity.structure_sequence, mutations[idx])
                offset = find_orf_offset(design_entity.structure_sequence, mutations)
                # referenced_design_sequences[source_entity] = design_entity.structure_sequence[offset:]
                formatted_design_sequence = design_entity.structure_sequence[offset:]
            # Check for expression tag addition to the designed sequences
            # for idx, (entity, formatted_design_sequence) in enumerate(zip(des_dir.pose.entities, referenced_design_sequences.values())):
            # for idx, (source_entity, formatted_design_sequence) in enumerate(referenced_design_sequences.items()):
                # pdb_code = entity.name[:4]
                selected_tag = {}
                available_tags = find_expression_tags(formatted_design_sequence)
                if available_tags:  # try to use existing tag
                    tag_namea = list(zip(*[(tag['name'], tag['termini'], tag['sequence']) for tag in available_tags]))
                    print(tag_namea)

                    tag_names, tag_termini, ind_tag_sequences = zip(*[(tag['name'], tag['termini'], tag['sequence'])
                                                                    for tag in available_tags])
                    try:
                        preferred_tag_index = tag_names.index(args.preferred_tag)
                        if tag_termini[preferred_tag_index] in true_termini:
                            selected_tag = available_tags[preferred_tag_index]
                    except ValueError:
                        pass
                    # if preferred_tag_index != -1:
                    #     if tag_termini[preferred_tag_index] in true_termini:
                    #         selected_tag = available_tags[preferred_tag_index]
                    pretag_sequence = remove_expression_tags(formatted_design_sequence, ind_tag_sequences)
                    # design_sequence = formatted_design_sequence
                else:
                    pretag_sequence = formatted_design_sequence

                if not selected_tag:  # find compatible tags from matching PDB observations
                    matching_tags_by_unp_id = tag_sequences.get(uniprot_id, None)
                    if not matching_tags_by_unp_id:
                        matching_tags_by_unp_id = find_matching_expression_tags(uniprot_id=uniprot_id)
                        tag_sequences[uniprot_id] = matching_tags_by_unp_id
                    tag_names, tag_termini, ind_tag_sequences = zip(*[(tag['name'], tag['termini'], tag['sequence'])
                                                                  for tag in matching_tags_by_unp_id['matching_tags']])
                    tag_names, tag_termini, ind_tag_sequences = list(tag_names), list(tag_termini), list(ind_tag_sequences)
                    iteration = 0
                    while iteration < len(tag_names):
                        try:
                            preferred_tag_index_2 = tag_names[iteration:].index(args.preferred_tag)
                            if tag_termini[preferred_tag_index_2] in true_termini:
                                selected_tag = matching_tags_by_unp_id[preferred_tag_index_2]
                                break
                        except ValueError:
                        # if preferred_tag_index == -1:
                            break
                        # elif tag_termini[preferred_tag_index] in true_termini:
                        #     selected_tag = available_tags[preferred_tag_index_2]
                        #     break
                        iteration += 1

                    selected_tag = select_tags_for_sequence(sequence_id, matching_tags_by_unp_id, **termini_availability)

                if selected_tag.get('name'):
                    missing_tags[(des_dir, design)][idx] = 0
                sequences_and_tags[design_string] = {'sequence': pretag_sequence, 'tag': selected_tag}

            # after selecting all tags, consider tagging the design as a whole
            number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])
            if number_of_tags > number_of_found_tags:
                print('There were %d requested tags for design %s and only %d were found'
                      % (number_of_tags, des_dir, number_of_found_tags))
                current_tag_options = \
                    '\n\t'.join(['%d - %s\n\t\t%s' % (i, entity_name, tag_options['tag'])
                                 for i, (entity_name, tag_options) in enumerate(sequences_and_tags.items(), 1)])
                print('Current Tag Options:%s' % current_tag_options)
                satisfied = input('If this is acceptable, enter \'continue\' to move on to the next design. Otherwise, '
                                  'you can modify the tagging options with any other input.%s' % input_string)
                if satisfied == 'continue':
                    number_of_found_tags = number_of_tags

                iteration_idx = 0
                while number_of_tags != number_of_found_tags:
                    for idx, entity_missing_tag in enumerate(missing_tags[(des_dir, design)][iteration_idx:]):
                        sequence_id = '%s_%s' % (des_dir, des_dir.pose.entities[idx].name)
                        if entity_missing_tag and tag_index[idx]:  # isn't tagged but could be
                            print('The entity %s is missing a tag. Would you like to tag this entity?' % sequence_id)
                            if not boolean_choice():
                                continue
                        else:
                            continue
                        if args.preferred_tag:
                            tag = args.preferred_tag
                            while True:
                                termini = input('Your preferred tag will be added to one of the termini. Which '
                                                'termini would you prefer? [n/c]%s'
                                                % (sequence_id, input_string))
                                if termini in ['n', 'c']:
                                    break
                                else:
                                    print('\'%s\' is an invalid input, one of \'n\' or \'c\' is required.')
                        else:
                            while True:
                                tag_input = int(
                                    input('What tag would you like to use? Enter the number of the below options.'
                                          '\n\t%s\n%s'
                                          % ('\n\t'.join(['%d - %s' % (i, tag)
                                                          for i, tag in enumerate(expression_tags, 1)]),
                                             input_string)))
                                if tag_input <= len(expression_tags):
                                    tag = list(expression_tags.keys())[tag_input - 1]
                                    break
                                else:
                                    print('Input doesn\'t match available options. Please try again')
                            while True:
                                termini = input('Your tag will be added to one of the termini. Which termini would '
                                                'you prefer? [n/c]%s' % input_string)
                                if termini in ['n', 'c']:
                                    break
                                else:
                                    print('\'%s\' is an invalid input, one of \'n\' or \'c\' is required.')

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
                    if iteration_idx == len(missing_tags[(des_dir, design)]):
                        print('You have seen all options, but the number of requested tags (%d) doesn\t equal the '
                              'number selected (%d)' % (number_of_tags, number_of_found_tags))
                        satisfied = input('If you are satisfied with this, enter \'continue\', otherwise enter anything'
                                          ' and you can view all remaining options starting from the first entity%s'
                                          % input_string)
                        if satisfied == 'continue':
                            break
                        else:
                            iteration_idx = 0
                    number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

            elif number_of_tags < number_of_found_tags:
                print('There were only %d requested tags for design %s and %d were found'
                      % (number_of_tags, des_dir, number_of_found_tags))
                while number_of_tags != number_of_found_tags:
                    tag_input = input('Which tag would you like to remove? Enter the number of the currently '
                                      'configured tag option that you would like to remove. If you would like to '
                                      'keep all, specify \'keep\' \n\t%s\n%s'
                                      % ('\n\t'.join(['%d - %s\n\t\t%s' % (i, entity_name, tag_options['tag'])
                                                      for i, (entity_name, tag_options)
                                                      in enumerate(sequences_and_tags.items(), 1)]), input_string))
                    if tag_input == 'keep':
                        break
                    else:
                        tag_input = int(tag_input)
                        if tag_input <= len(sequences_and_tags):
                            missing_tags[(des_dir, design)][tag_input - 1] = 1
                            selected_entity = list(sequences_and_tags.keys())[tag_input - 1]
                            sequences_and_tags[selected_entity]['tag'] = None
                            # tag = list(expression_tags.keys())[tag_input - 1]
                            break
                        else:
                            print('Input doesn\'t match available options. Please try again')
                    number_of_found_tags = len(des_dir.pose.entities) - sum(missing_tags[(des_dir, design)])

            # apply all tags to the sequences
            for idx, (design_string, sequence_tag) in enumerate(sequences_and_tags.items()):
                design_sequence = add_expression_tag(sequence_tag['tag'].get('sequence'), sequence_tag['sequence'])

                # If no MET start site, include one
                if design_sequence[0] != 'M':
                    design_sequence = 'M%s' % design_sequence
                if 'X' in design_sequence:
                    logger.critical('An unrecognized amino acid was specified in the sequence %s. '
                                    'This will require manual intervention!' % sequence_id)

                # For a final manual check of sequence generation, find sequence additions compared to the design model
                # and save to view where additions lie on sequence. Cross these additions with design structure to check
                # if insertions are compatible
                all_insertions = {residue: {'to': aa} for residue, aa in enumerate(design_sequence, 1)}
                all_insertions.update(generate_mutations(designed_atom_sequences[idx], design_sequence, blanks=True))
                # Reduce to sequence only
                inserted_sequences[design_string] = '%s\n%s' % (''.join([res['to'] for res in all_insertions.values()]),
                                                                design_sequence)
                final_sequences[design_string] = design_sequence

        # Write output sequences to fasta file
        seq_comparison_file = SDUtils.write_fasta_file(inserted_sequences, '%sSelectedSequencesExpressionAdditions'
                                                       % args.selection_string, out_path=outdir)
        logger.info('Design insertions for expression comparison written to %s' % seq_comparison_file)
        seq_file = SDUtils.write_fasta_file(final_sequences, '%sSelectedSequences' % args.selection_string,
                                            out_path=outdir)
        logger.info('Final Design sequences written to %s' % seq_file)
    # ---------------------------------------------------
    elif args.module == 'rename_scores':
        rename = {'combo_profile_switch': 'limit_to_profile', 'favor_profile_switch': 'favor_frag_limit_to_profile'}
        for des_directory in design_directories:
            SDUtils.rename_decoy_protocols(des_directory, rename)
    # ---------------------------------------------------
    # elif args.module == 'modify':  # -m mod
    #     if args.multi_processing:
    #         if args.mod == 'consolidate_degen':
    #             exit('Operation impossible with flag \'-mp\'')
    #
    #     else:
    #         if args.mod == 'consolidate_degen':
    #             logger.info('Consolidating DEGEN directories')
    #             accessed_dirs = [rsync_dir(design_directory) for design_directory in design_directories]
    # ---------------------------------------------------
    elif args.module == 'status':  # -n number, -s stage, -u update
        if args.update:
            for design in design_directories:
                update_status(design.serialized_info, args.stage, mode=args.update)
        else:
            if args.number_designs:
                logger.info('Checking for %d files. If no stage is specified, results will be incorrect for all but '
                            'design stage' % args.number_designs)
            if args.stage:
                status(design_directories, args.stage, number=args.number_designs)
            else:
                for stage in PUtils.stage_f:
                    s = status(design_directories, stage, number=args.number_designs)
                    if s:
                        logger.info('For \'%s\' stage, default settings should generate %d files'
                                    % (stage, PUtils.stage_f[stage]['len']))
    # else:
    #     exit('No module was selected! Did you include one? To get started, checkout the %s' % PUtils.guide_string)
    # -----------------------------------------------------------------------------------------------------------------
    # Finally run terminate(). Formats designs passing output parameters and report program exceptions
    # -----------------------------------------------------------------------------------------------------------------
