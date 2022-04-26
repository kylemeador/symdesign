"""
Module for distribution of SymDesign commands. Includes pose initialization, distribution of Rosetta commands to
SLURM computational clusters, analysis of designed poses, and sequence selection of completed structures.

"""
import argparse
import copy
import datetime
import os
import shutil
from subprocess import Popen, list2cmdline
import sys
import time
from glob import glob
from itertools import repeat, product, combinations
from json import loads, dumps
from csv import reader

import pandas as pd
import psutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data.IUPACData import protein_letters

from FragDock import nanohedra_dock
from SymDesignUtils import ex_path
from dependencies.DnaChisel.dnachisel.DnaOptimizationProblem.NoSolutionError import NoSolutionError

import PathUtils as PUtils
import SymDesignUtils as SDUtils
from Query.PDB import retrieve_pdb_entries_by_advanced_query
from Query.utils import input_string, bool_d, validate_input, boolean_choice, invalid_string
from utils.CmdLineArgParseUtils import query_mode
from Query import Flags
from classes.SymEntry import SymEntry, parse_symmetry_to_sym_entry
from classes.EulerLookup import EulerLookup
from Database import Database  # FragmentDatabase,
from CommandDistributer import distribute, hhblits_memory_threshold, update_status
from DesignDirectory import DesignDirectory, get_sym_entry_from_nanohedra_directory, JobResources
from NanohedraWrap import nanohedra_command, nanohedra_design_recap
from PDB import PDB, orient_pdb_file
from ClusterUtils import cluster_designs, invert_cluster_map, group_compositions, ialign  # pose_rmsd, cluster_poses
from ProteinExpression import find_expression_tags, find_matching_expression_tags, add_expression_tag, \
    select_tags_for_sequence, remove_expression_tags, expression_tags, optimize_protein_sequence, \
    default_multicistronic_sequence
from DesignMetrics import prioritize_design_indices, master_metrics, query_user_for_metrics
from SequenceProfile import generate_mutations, find_orf_offset, write_fasta, read_fasta_file  # , pdb_to_pose_offset
from utils.GeneralUtils import write_docking_parameters
from utils.guide import interface_design_guide, analysis_guide


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


def terminate(results=None, output=True):
    """Format designs passing output parameters and report program exceptions

    Keyword Args:
        results=None (list): The returned results from the module run. By convention contains results and exceptions
        output=False (bool): Whether the module used requires a file to be output
    Returns:
        (None)
    """
    global out_path
    # save any information found during the design command to it's serialized state
    for design in design_directories:
        design.pickle_info()

    if results:
        success = \
            [design_directories[idx] for idx, result in enumerate(results) if not isinstance(result, BaseException)]
        exceptions = \
            [(design_directories[idx], result) for idx, result in enumerate(results) if isinstance(result, BaseException)]
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
        # job = next(iter(design_directories))
        job_paths = job.job_paths
        job.make_path(job_paths)
        if low and high:
            timestamp = '%s-%.2f-%.2f' % (SDUtils.starttime, low, high)
        # Make single file with names of each directory where all_docked_poses can be found
        # project_string = os.path.basename(design_directories[0].project_designs)
        default_output_tuple = (SDUtils.starttime, args.module, design_source)
        if args.output_file and args.module not in [PUtils.analysis, PUtils.cluster_poses]:
            designs_file = args.output_file
        else:
            designs_file = os.path.join(job_paths, '%s_%s_%s_pose.paths' % default_output_tuple)

        with open(designs_file, 'w') as f:
            f.write('%s\n' % '\n'.join(design.path for design in success))
        logger.critical('The file "%s" contains the locations of all designs in your current project that passed '
                        'internal checks/filtering. Utilize this file to interact with %s designs in future commands '
                        'for this project such as:\n\t%s --file %s MODULE\n'
                        % (designs_file, PUtils.program_name, PUtils.program_command, designs_file))

        if args.module == PUtils.analysis:
            all_scores = job.all_scores
            # Save Design DataFrame
            design_df = pd.DataFrame([result for result in results if not isinstance(result, BaseException)])
            args.output_file = args.output_file if args.output_file.endswith('.csv') else '%s.csv' % args.output_file
            design_df.to_csv(args.output_file)
            logger.info('Analysis of all poses written to %s' % args.output_file)
            if save:
                logger.info('Analysis of all Trajectories and Residues written to %s' % all_scores)
        elif args.module == PUtils.cluster_poses:
            logger.info('Clustering analysis results in the following similar poses:\nRepresentatives\n\tMembers\n')
            for representative, members, in results.items():
                print('%s\n\t%s' % (representative, '\n\t'.join(map(str, members))))
            logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                        % (len(pose_cluster_map), len(design_directories), pose_cluster_file))
            logger.info('Each cluster above has one representative which identifies with each of the members. If '
                        'clustering was performed by transformation or interface_residues, then the representative is '
                        'the most similar to all members. If clustering was performed by ialign, then the '
                        'representative is randomly chosen.')
            logger.info('To utilize the above clustering, during %s, using the option --cluster_map, will apply '
                        'clustering to poses to select a cluster representative based on the most favorable cluster '
                        'member' % PUtils.select_designs)

        design_stage = PUtils.stage[12] if getattr(args, PUtils.scout, None) \
            else (PUtils.stage[2] if getattr(args, PUtils.no_hbnet, None)
                  else (PUtils.stage[14] if getattr(args, PUtils.structure_background, None)
                        else PUtils.stage[13]))  # hbnet_design_profile
        module_files = {PUtils.interface_design: design_stage, PUtils.nano: PUtils.nano, PUtils.refine: PUtils.refine,
                        'interface_metrics': 'interface_metrics',
                        'custom_script': os.path.splitext(os.path.basename(getattr(args, 'script', 'c/custom')))[0],
                        'optimize_designs': 'optimize_design'}
        stage = module_files.get(args.module)
        if stage:
            if len(success) == 0:
                exit_code = 1
                exit(exit_code)
            # sbatch_scripts = job.sbatch_scripts
            command_file = SDUtils.write_commands([os.path.join(design.scripts, '%s.sh' % stage) for design in success],
                                                  out_path=job_paths, name='_'.join(default_output_tuple))
            sbatch_file = distribute(file=command_file, out_path=job.sbatch_scripts, scale=args.module)
            #                                                                        ^ for sbatch template
            logger.critical(sbatch_warning)
            if args.module == PUtils.interface_design and not job.pre_refine:  # must refine before design
                refine_file = SDUtils.write_commands([os.path.join(design.scripts, '%s.sh' % PUtils.refine)
                                                      for design in success], out_path=job_paths,
                                                     name='_'.join((SDUtils.starttime, PUtils.refine, design_source)))
                sbatch_refine_file = distribute(file=refine_file, out_path=job.sbatch_scripts, scale=PUtils.refine)
                logger.info('Once you are satisfied, enter the following to distribute:\n\tsbatch %s\nTHEN:\n\tsbatch '
                            '%s' % (sbatch_refine_file, sbatch_file))
            else:
                logger.info('Once you are satisfied, enter the following to distribute:\n\tsbatch %s' % sbatch_file)
    print('\n')
    exit(exit_code)


def load_global_dataframe():
    """Return a pandas DataFrame with the trajectories of every design_directory loaded and formatted according to the
    design directory and design on the index

    Returns:
        (pandas.DataFrame)
    """
    all_dfs = [pd.read_csv(design.trajectories, index_col=0, header=[0]) for design in design_directories]
    for idx, df in enumerate(all_dfs):
        # get rid of all statistic entries, mean, std, etc.
        df.drop([index for index in df.index.to_list() if design_directories[idx].name not in index],
                inplace=True)
    df = pd.concat(all_dfs, keys=design_directories)  # must add the design directory string to each index
    df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    return df


def generate_sequence_template(pdb_file):
    pdb = PDB.from_file(pdb_file)
    sequence = SeqRecord(Seq(''.join(pdb.atom_sequences.values()), 'Protein'), id=pdb.filepath)
    sequence_mask = copy.copy(sequence)
    sequence_mask.id = 'residue_selector'
    sequences = [sequence, sequence_mask]
    return write_fasta(sequences, file_name='%s_residue_selector_sequence' % os.path.splitext(pdb.filepath)[0])


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
                                     'If your a first time user, try "%s --guide"'
                                     '\nAll jobs have built in features for command monitoring & distribution to '
                                     'computational clusters for parallel processing.\n'
                                     % (PUtils.program_name, PUtils.nano.title(), PUtils.program_command),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---------------------------------------------------
    # parser.add_argument('-symmetry', '--symmetry', type=str, help='The design symmetry to use. Possible symmetries '
    #                                                             'include %s' % ', '.join(possible_symmetries))
    parser.add_argument('-a', '--output_assembly', action='store_true',
                        help='Should the assembly be output for docked poses? 2- and 3-D materials will be output '
                             'with a single unit cell.')
    parser.add_argument('-b', '--debug', action='store_true',
                        help='Whether to log debugging messages to stdout\nDefault=False')
    parser.add_argument('-C', '--cluster_map', type=os.path.abspath,
                        help='The location of a serialized file containing spatially or interfacially clustered poses')
    parser.add_argument('-c', '--cores', type=int,
                        help='Number of cores to use with --multiprocessing. If -mp is run in a cluster environment, '
                             'the number of cores will reflect the allocation provided by the cluster, otherwise, '
                             'specify the number of cores\nDefault=#ofCores - 1')
    parser.add_argument('-d', '--directory', type=os.path.abspath, metavar=ex_path('your_pdb_files'),
                        help='Master directory where poses to be designed with %s are located. This may be the output '
                             'directory from %s.py, a random directory with poses requiring interface design, or the '
                             'output from %s. If the directory lives in a %sOutput directory, all projects within the '
                             'directory will be selected. For finer control over which poses to manipulate, use --file,'
                             ' --project, or --single flags.'
                             % (PUtils.program_name, PUtils.nano, PUtils.program_name, PUtils.program_name))
    parser.add_argument('-dr', '--design_range', type=float,
                        help='The range of designs to consider from a larger chunk of work to complete. The argument '
                             'should specify a percentage of work from 0-100 and should separate two numbers by a '
                             'single "-". Ex: 25-50', default=None)
    parser.add_argument('-df', '--dataframe', type=os.path.abspath, metavar=ex_path('Metrics.csv'),
                        help='A DataFrame created by %s analysis containing pose info. File is .csv, named such as '
                             'Metrics.csv' % PUtils.program_name)
    parser.add_argument('-fc', '--fuse_chains', type=str, nargs='*', default=[],
                        help='The name of a pair of chains to fuse during design. Pairs should be separated by a colon,'
                             ' new instances by a space. Ex --fuse_chains A:B C:D')
    parser.add_argument('-f', '--file', type=os.path.abspath, metavar=ex_path('file_with_directory_names.txt'),
                        help='File with location(s) of %s designs. For each run of %s, a file will be created '
                             'specifying the specific directories to use in subsequent %s commands of the same designs.'
                             ' If pose-IDs are specified in a file, say as the result of %s or %s, in addition to the '
                             'pose-ID file, provide your %s working directory to locate the pose-Ids of interest.'
                             % (PUtils.program_name, PUtils.program_name, PUtils.program_name, PUtils.analysis,
                                PUtils.select_designs, PUtils.program_name),
                        default=None, nargs='*')
    parser.add_argument('-sf', '--specification_file', type=str, metavar=ex_path('pose_design_specifications.csv'),
                        help='Name of comma separated file with each line formatted:\n'
                             'poseID, [designID], [residue_number:design_directive '
                             'residue_number2-residue_number9:directive ...]')
    parser.add_argument('-F', '--%s' % PUtils.force_flags, action='store_true',
                        help='Force generation of a new flags file to update script parameters')
    parser.add_argument('-g', '--guide', action='store_true',
                        help='Access the %s guide! Display the program or module specific guide. Ex: "%s --guide" '
                             'or "%s"' % (PUtils.program_name, PUtils.program_command, PUtils.submodule_guide))
    parser.add_argument('-gf', '--%s' % PUtils.generate_fragments, action='store_true',
                        help='Generate fragment overlap for poses of interest.')
    parser.add_argument('-ic', '--ignore_clashes', action='store_true',
                        help='Whether errors raised from identified clashes should be ignored and allowed to process')
    parser.add_argument('-l', '--load_database', action='store_true',
                        help='Whether to fetch and store resources for each Structure in the sequence/structure '
                             'database')
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-no', '--nanohedra_output', action='store_true',
                        help='Whether the directory in question is a Nanohedra Docking output')
    parser.add_argument('-od', '--output_directory', type=os.path.abspath, default=None,
                        help='If provided, the name of the directory to output all created files. If blank, one will be'
                             ' automatically generated based off input_location, module, and the time.')
    parser.add_argument('-of', '--output_file', type=str,
                        help='If provided, the name of the output designs file. If blank, one will be automatically '
                             'generated based off input_location, module, and the time.')
    parser.add_argument('-p', '--project', type=os.path.abspath, nargs='*',
                        metavar=ex_path('SymDesignOutput', 'Projects', 'your_project(s)'),
                        help='If you wish to operate on designs specified by a whole project, which project(s) to use?')
    parser.add_argument('-r', '--run_in_shell', action='store_true',
                        help='Should commands be executed at %s runtime? In most cases, it won\'t maximize cassini\'s '
                             'computational resources. Additionally, all computation may fail on a single trajectory '
                             'mistake.\nDefault=False' % PUtils.program_name)
    parser.add_argument('-s', '--single', type=os.path.abspath, nargs='*',
                        metavar=ex_path('SymDesignOutput', 'Projects', 'your_project', 'single_design(s)[.pdb]'),
                        help='If you wish to operate on designs specified by a single pose, which pose(s) to use?')
    parser.add_argument('-se', '--%s' % PUtils.sym_entry, type=int, default=None,
                        help='The entry number of %s.py docking combinations to use' % PUtils.nano.title())
    parser.add_argument('-S', '--symmetry', type=str, default=None,
                        help='The specific symmetry of the designs of interest. Preferably in a composition formula'
                             'such as T:{C3}{C3}...')
    parser.add_argument('-wf', '--write_fragments', action='store_true',  # Todo invert the default
                        help='For any fragments generated, write them along with the Pose')
    parser.add_argument('-wo', '--write_oligomers', action='store_true',
                        help='For any oligomers generated, write them along with the Pose')
    subparsers = parser.add_subparsers(title='Modules', dest='module',
                                       description='These are the different modes that designs can be processed',
                                       help='Chose a Module followed by Module specific flags. To get help with a '
                                            'Module such as Module algorithmic specifics or to see example commands '
                                            'enter:\t%s\n\nTo get help with Module flags enter:\t%s\n, Any Module '
                                            '--guide or --help can be accessed in this way.'
                                            % (PUtils.submodule_guide, PUtils.submodule_help))
    # ---------------------------------------------------
    # Set Up SubModule Parsers
    # ---------------------------------------------------
    # Set Up SubModule Parsers
    # ---------------------------------------------------
    parser_query = subparsers.add_parser('nanohedra_query', help='Query %s.py docking entries' % PUtils.nano.title())
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
                                          help='Orient a symmetric assembly in a canonical orientation at the origin')
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
    parser_clashes = subparsers.add_parser('check_clashes',
                                           help='Check for clashes between full models. Useful for understanding '
                                                'if loops are missing, whether their modelled density is '
                                                'compatible with the pose')
    # ---------------------------------------------------
    parser_dock = subparsers.add_parser(PUtils.nano,
                                        help='Run or submit jobs to %s.py.\nUse the Module arguments -c1/-c2, -o1/-o2, '
                                             'or -q to specify PDB Entity codes, building block directories, or query '
                                             'the PDB for building blocks to dock' % PUtils.nano.title())
    parser_dock.add_argument('-e', '--entry', type=int, default=None, dest='sym_entry', required=True,
                             help='The entry number of %s.py docking combinations to use' % PUtils.nano.title())
    parser_dock.add_argument('-mv', '--match_value', type=float, default=0.5, dest='high_quality_match_value',
                             help='What is the minimum match score required for a high quality fragment?')
    parser_dock.add_argument('-m', '--min_matched', type=int, default=3,
                             help='How many high quality fragment pairs should be present before a pose is identified?')
    parser_dock.add_argument('-o', '--outdir', type=str, dest='output_directory', default=None,
                             # default=os.path.join(os.getcwd(), PUtils.program_output, PUtils.data.title()),
                             help='Where should the output from commands be written?\n'
                                  'Default=%s' % ex_path(PUtils.program_output, PUtils.data.title(),
                                                         'NanohedraEntry[ENTRYNUMBER]DockedPoses'))
    dock1_group = parser_dock.add_mutually_exclusive_group(required=True)
    dock2_group = parser_dock.add_mutually_exclusive_group()
    dock1_group.add_argument('-c1', '--pdb_codes1', type=os.path.abspath,
                             help='File with list of PDB_entity codes for component 1\n', default=None)
    dock2_group.add_argument('-c2', '--pdb_codes2', type=os.path.abspath,
                             help='File with list of PDB_entity codes for component 2\n', default=None)
    dock1_group.add_argument('-o1', '--oligomer1', type=os.path.abspath,
                             help='Disk location where the first oligomer(s) are located\n', default=None)
    dock2_group.add_argument('-o2', '--oligomer2', type=os.path.abspath,
                             help='Disk location where the second oligomer(s) are located\n', default=None)
    dock1_group.add_argument('-qc', '--query_codes', action='store_true',
                             help='Search the PDB API for corresponding codes\n')
    parser_dock.add_argument('-q', '--query', action='store_true',
                             help='Run %s in query mode\n' % PUtils.nano)
    parser_dock.add_argument('-r1', '--rot_step1', type=os.path.abspath, default=3,
                             help='Disk location where the first oligomers are located\nREQUIRED')
    parser_dock.add_argument('-r2', '--rot_step2', type=os.path.abspath, default=3,
                             help='Disk location where the second oligomers are located\nDefault=None')
    parser_dock.add_argument('-suc', '--output_surrounding_uc', action='store_true',
                             help='Should the surrounding unit cells be output for docked poses? Only viable for 2D and'
                                  ' 3D materials')
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
    parser_cluster.add_argument('-m', '--mode', type=str, choices=['transform', 'ialign', 'interface_residues'],
                                default='transform')
    parser_cluster.add_argument('-of', '--output_file', type=str, default=PUtils.clustered_poses,
                                help='Name of the output .pkl file containing design clusters Will be saved to the %s'
                                     ' folder of the output.\nDefault=%s'
                                     % (PUtils.data.title(), PUtils.clustered_poses % ('LOCATION', 'TIMESTAMP')))
    # ---------------------------------------------------
    parser_refine = subparsers.add_parser(PUtils.refine, help='')
    parser_refine.add_argument('-ala', '--interface_to_alanine', action='store_true',
                               help='Whether to mutate all interface residues to alanine before refinement')
    parser_refine.add_argument('-met', '--gather_metrics', action='store_true',
                               help='Whether to gather interface metrics for contained interfaces after refinement')
    # ---------------------------------------------------
    parser_design = subparsers.add_parser(PUtils.interface_design,
                                          help='Gather poses of interest and format for design using sequence '
                                               'constraints in Rosetta. Constrain using evolutionary profiles of '
                                               'homologous sequences and/or fragment profiles extracted from the PDB or'
                                               ' neither.')
    parser_design.add_argument('-nec', '--%s' % PUtils.no_evolution_constraint, action='store_true',
                               help='Whether to skip evolutionary constraints during design')
    parser_design.add_argument('-nhb', '--%s' % PUtils.no_hbnet, action='store_true',
                               help='Whether to skip hydrogen bond networks in the design')
    parser_design.add_argument('-ntc', '--%s' % PUtils.no_term_constraint, action='store_true',
                               help='Whether to skip tertiary motif constraints during design')
    parser_design.add_argument('-n', '--%s' % PUtils.number_of_trajectories, type=int, default=PUtils.nstruct,
                               help='How many unique sequences should be generated for each input?')
    parser_design.add_argument('-sb', '--%s' % PUtils.structure_background, action='store_true',
                               help='Whether to skip all constraints and measure the structure in an optimal context')
    parser_design.add_argument('-sc', '--%s' % PUtils.scout, action='store_true',
                               help='Whether to set up a low resolution scouting protocol to survey designability')
    # parser_design.add_argument('-i', '--fragment_database', type=str,
    #                            help='Database to match fragments for interface specific scoring matrices. One of %s'
    #                                 '\nDefault=%s' % (','.join(list(PUtils.frag_directory.keys())),
    #                                                   list(PUtils.frag_directory.keys())[0]),
    #                            default=list(PUtils.frag_directory.keys())[0])
    # ---------------------------------------------------
    parser_metrics = subparsers.add_parser('interface_metrics',
                                           help='Set up RosettaScript to analyze interface metrics from a pose')
    parser_metrics.add_argument('-sp', '--specific_protocol', type=str,
                                help='The specific protocol to perform interface_metrics on')
    # ---------------------------------------------------
    parser_optimize_designs = subparsers.add_parser('optimize_designs',
                                                    help='Optimize and touch up designs after running interface design.'
                                                         ' Useful for reverting excess mutations to wild-type, or '
                                                         'directing targeted exploration of specific troublesome areas')
    # ---------------------------------------------------
    parser_custom = subparsers.add_parser('custom_script',
                                          help='Set up a custom RosettaScripts.xml for designs. The custom '
                                               'script will be run in every pose specified using specified options')
    parser_custom.add_argument('-l', '--file_list', action='store_true',
                               help='Whether to use already produced designs in the designs/ directory')
    parser_custom.add_argument('-n', '--native', type=str, help='What structure to use as a "native" structure for '
                                                                'Rosetta reference calculations. Default=refined_pdb',
                               choices=['source', 'asu', 'assembly', 'refine_pdb', 'refined_pdb', 'consensus_pdb',
                                        'consensus_design_pdb'])
    parser_custom.add_argument('--score_only', action='store_true', help='Whether to only score the design(s)')
    parser_custom.add_argument('script', type=os.path.abspath, help='The location of the custom script')
    parser_custom.add_argument('--suffix', type=str, metavar='SUFFIX',
                               help='Append to each output file (decoy in .sc and .pdb) the script name (i.e. '
                                    '"decoy_SUFFIX") to identify this protocol. No extension will be included')
    parser_custom.add_argument('-v', '--variables', type=str, nargs='*',
                               help='Additional variables that should be populated in the script. Provide a list'
                                    ' of such variables with the format "variable1=value variable2=value". '
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
    parser_analysis.add_argument('-of', '--output_file', type=str, default=PUtils.analysis_file,
                                 help='Name of the output .csv file containing design metrics. Will be saved to the %s'
                                      ' folder of the output.\nDefault=%s'
                                      % (PUtils.all_scores, PUtils.analysis_file % ('TIMESTAMP', 'LOCATION')))
    parser_analysis.add_argument('-N', '--no_save', action='store_true',
                                 help='Don\'t save trajectory information.\nDefault=False')
    parser_analysis.add_argument('-fig', '--figures', action='store_true',
                                 help='Create and save figures for all poses?\nDefault=False')
    parser_analysis.add_argument('-j', '--join', action='store_true',
                                 help='Join Trajectory and Residue Dataframes?\nDefault=False')
    # ---------------------------------------------------
    parser_filter = subparsers.add_parser(PUtils.select_designs,
                                          help='Select designs based on design specific metrics. Can be one of a '
                                               'handful of --metrics or from a DesignAnalysis.csv file generated by %s.'
                                               ' Either -df or -pf is required. If both are provided, -pf will be '
                                               'prioritized' % PUtils.analysis)
    filter_required = parser_filter.add_mutually_exclusive_group(required=True)
    # filter_required.add_argument('-df', '--dataframe', type=os.path.abspath,
    #                              metavar=ex_path('AllPoseDesignMetrics.csv'),
    #                              help='Dataframe.csv from analysis containing pose info.')
    filter_required.add_argument('-m', '--metric', type=str,
                                 help='If a simple metric filter is required, what metric would you like to sort '
                                      'Designs by?', choices=['score', 'fragments_matched'])
    filter_required.add_argument('-pf', '--pose_design_file', type=str, metavar=ex_path('pose_design.csv'),
                                 help='Name of .csv file with (pose, design pairs to serve as sequence selector')
    parser_filter.add_argument('--filter', action='store_true',
                               help='Whether to filter sequence selection using metrics from DataFrame')
    parser_filter.add_argument('-np', '--number_poses', type=int, default=0, metavar='INT',
                               help='Number of top poses to return per pool of designs.\nDefault=All')
    parser_filter.add_argument('--protocol', type=str, help='Use specific protocol(s) to grab designs from?',
                               default=None, nargs='*')
    parser_filter.add_argument('-s', '--selection_string', type=str, metavar='string',
                               help='String to prepend to output for custom design selection name')
    parser_filter.add_argument('--weight', action='store_true',
                               help='Whether to weight sequence selection using metrics from DataFrame')
    metric_weight_functions = ['rank', 'normalize']
    parser_filter.add_argument('-wf', '--weight_function', choices=metric_weight_functions,
                               help='How to standardize metrics during sequence selection weighting')
    # ---------------------------------------------------
    parser_sequence = subparsers.add_parser(PUtils.select_sequences,
                                            help='From the provided Design Poses, generate nucleotide/protein sequences'
                                                 ' based on specified selection criteria and prioritized metrics. '
                                                 'Generation of output sequences can take multiple forms depending on '
                                                 'downstream needs. By default, disordered region insertion, tagging '
                                                 'for expression, and codon optimization (--nucleotide) are performed')
    parser_sequence.add_argument('-amp', '--allow_multiple_poses', action='store_true',
                                 help='Allow multiple sequences to be selected from the same Pose when using '
                                      '--global_sequences. By default, --global_sequences filters the selected '
                                      'sequences by a single sequence/Pose')
    parser_sequence.add_argument('-ath', '--avoid_tagging_helices', action='store_true',
                                 help='Should tags be avoided at termini with helices?')
    parser_sequence.add_argument('--csv', action='store_true',
                                 help='Write the sequences file as a .csv instead of the default .fasta')
    parser_sequence.add_argument('-e', '--entity_specification', type=str,
                                 # choices=['single', 'all', 'none'], Todo make work with list...
                                 help='If there are specific entities in the designs you want to tag, indicate how '
                                      'tagging should occur. Viable options include "single" - a single entity, '
                                      '"all" - all entities, "none" - no entities, or provide a comma separated '
                                      'list such as "1,0,1" where "1" indicates a tag requirement and "0" '
                                      'indicates no tag is required.')
    parser_sequence.add_argument('--filter', action='store_true',
                                 help='Whether to filter sequence selection using metrics from DataFrame')
    parser_sequence.add_argument('-g', '--global_sequences', action='store_true',
                                 help='Should sequences be selected based on their ranking in the total design pool. '
                                      'This will search for the top sequences from all poses and then choose only one '
                                      'sequence per pose')
    parser_sequence.add_argument('-m', '--multicistronic', action='store_true',
                                 help='Whether to output nucleotide sequences in multicistronic format. '
                                      'By default, use without --multicistronic_intergenic_sequence uses the pET-Duet '
                                      'intergeneic sequence containing a T7 promoter, LacO, and RBS')
    parser_sequence.add_argument('-ms', '--multicistronic_intergenic_sequence', type=str,
                                 help='The sequence to use in the intergenic region of a multicistronic expression '
                                      'output')
    parser_sequence.add_argument('-n', '--nucleotide', action='store_true',
                                 help='Whether to output codon optimized nucleotide sequences')
    parser_sequence.add_argument('-ns', '--number_sequences', type=int, default=sys.maxsize, metavar='INT',
                                 help='Number of top sequences to return. If global_sequences is True, returns the '
                                      'specified number_sequences sequences (Default=No Limit).\nOtherwise the '
                                      'specified number will be found from each pose (Default=1)')
    parser_sequence.add_argument('-opt', '--optimize_species', type=str, default='e_coli',
                                 help='The organism where expression will occur and nucleotide usage should be '
                                      'optimized')
    parser_sequence.add_argument('--protocol', type=str, help='Use specific protocol(s) to grab designs from?',
                                 default=None, nargs='*')
    parser_sequence.add_argument('-ssg', '--skip_sequence_generation', action='store_true',
                                 help='Should sequence generation be skipped? Only selected structure files will be '
                                      'collected')
    parser_sequence.add_argument('-ss', '--selection_string', type=str, metavar='string',
                                 help='String to prepend to output for custom sequence selection name')
    parser_sequence.add_argument('-t', '--preferred_tag', type=str,
                                 help='The name of your preferred expression tag. Default=his_tag',
                                 choices=expression_tags.keys(), default='his_tag')
    parser_sequence.add_argument('--weight', action='store_true',
                                 help='Whether to weight sequence selection using metrics from DataFrame')
    parser_sequence.add_argument('-wf', '--weight_function', choices=metric_weight_functions,
                                 help='How to standardize metrics during sequence selection weighting')
    # ---------------------------------------------------
    parser_multicistron = subparsers.add_parser('multicistronic',
                                                help='Generate nucleotide sequences for selected designs by codon '
                                                     'optimizing protein sequences, then concatenating nucleotide '
                                                     'sequences. REQUIRES an input .fasta file specified as -f/--file')
    parser_multicistron.add_argument('-c', '--csv', action='store_true',
                                     help='Write the sequences file as a .csv instead of the default .fasta')
    parser_multicistron.add_argument('-ms', '--multicistronic_intergenic_sequence', type=str,
                                     help='The sequence to use in the intergenic region of a multicistronic expression '
                                          'output')
    parser_multicistron.add_argument('-n', '--number_of_genes', type=int,
                                     help='The number of protein sequences to concatenate into a multicistronic '
                                          'expression output')
    parser_multicistron.add_argument('-opt', '--optimize_species', type=str, default='e_coli',
                                     help='The organism where expression will occur and nucleotide usage should be '
                                          'optimized')
    parser_multicistron.add_argument('-ss', '--selection_string', type=str, metavar='string',
                                     help='String to prepend to output for custom sequence selection name')
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Get design status for selected designs')
    parser_status.add_argument('-n', '--%s' % PUtils.number_of_trajectories, type=int, default=PUtils.nstruct,
                               help='How many unique sequences (trajectories) were generated per design')
    parser_status.add_argument('--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                               help='The stage of design to check status of. One of %s'
                                    % ', '.join(list(v for v in PUtils.stage_f.keys())), default=None)
    parser_status.add_argument('-u', '--update', type=str, choices=('check', 'set', 'remove'),
                               help='Provide an update to the serialized state of the specified stage', default=None)
    # ---------------------------------------------------
    parser_visualize = subparsers.add_parser('visualize', help='Get design status for selected designs')
    visualize_names = ['original', 'numerical']
    parser_visualize.add_argument('-n', '--name', choices=visualize_names, default='original', type=str,
                                  help='Number of trajectories per design. Options include "%s"'
                                       % ', '.join(visualize_names))
    visualize_order = ['alphabetical', 'none', 'dataframe', 'paths']
    parser_visualize.add_argument('-o', '--order', choices=visualize_order, default='alphabetical',
                                  help='The order in which designs will be loaded. Options include "%s".\nIf the '
                                       'order is other than alphabetical, provide the required datasource through '
                                       'one of the %s flags such as --dataframe or --file'
                                       % (', '.join(visualize_order), PUtils.program_name))
    # ---------------------------------------------------
    # parser_dist = subparsers.add_parser('distribute',
    #                                     help='Distribute specific design step commands to computational resources. '
    #                                        'In distribution mode, the --file or --directory argument specifies which '
    #                                          'pose commands should be distributed.')
    # parser_dist.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
    #                          help='The stage of design to be prepared. One of %s' %
    #                               ', '.join(list(v for v in PUtils.stage_f.keys())), required=True)
    # parser_dist.add_argument('-y', '--success_file', help='The name/location of file containing successful commands\n'
    #                                                       'Default={--stage}_stage_pose_successes', default=None)
    # parser_dist.add_argument('-n', '--failure_file', help='The name/location of file containing failed commands\n'
    #                                                       'Default={--stage}_stage_pose_failures', default=None)
    # parser_dist.add_argument('-m', '--max_jobs', type=int, help='How many jobs to run at once?\nDefault=80',
    #                          default=80)
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
    # parser_rename_scores = subparsers.add_parser('rename_scores', help='Rename Protocol names according to dictionary')
    # -----------------------------------------------------------------------------------------------------------------
    # Process flags
    # -----------------------------------------------------------------------------------------------------------------
    # these might be helpful for intermixing arguments before/after subparsers... (Modules)
    # parser.parse_intermixed_args(args=None, namespace=None)
    # parser.parse_known_intermixed_args
    unknown_args = None
    args, additional_args = parser.parse_known_args()
    # TODO work this into the flags parsing to grab module if included first and program flags if included after
    while len(additional_args) and additional_args != unknown_args:
        args, additional_args = parser.parse_known_args(additional_args, args)
        unknown_args = additional_args
    args, additional_args = parser.parse_known_args(additional_args, args)
    # print(args)
    if additional_args:
        print('Found additional arguments that are not recognized program wide:', additional_args)
    default_flags = Flags.return_default_flags(args.module)
    formatted_flags = format_additional_flags(additional_args)
    default_flags.update(formatted_flags)

    # Add additional program flags to queried_flags
    queried_flags = vars(args)
    queried_flags.update(default_flags)
    queried_flags.update(Flags.process_residue_selector_flags(queried_flags))
    # -----------------------------------------------------------------------------------------------------------------
    # Start Logging - Root logs to stream with level warning
    # -----------------------------------------------------------------------------------------------------------------
    if args.debug:
        # Root logs to stream with level debug
        logger = SDUtils.start_log(level=1, set_logger_level=True)
        SDUtils.set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        # Root logger logs to stream with level 'warning'
        SDUtils.start_log(level=3, set_logger_level=True)
        # Root logger logs all emissions to a single file with level 'info'. Stream above still emits at 'warning'
        SDUtils.start_log(handler=2, set_logger_level=True,
                          location=os.path.join(os.getcwd(), args.output_directory if args.output_directory else '',
                                                PUtils.program_name))
        # SymDesign main logs to stream with level info
        logger = SDUtils.start_log(name=PUtils.program_name, propagate=False)
        # All Designs will log to specific file with level info unless -skip_logging is passed
    # -----------------------------------------------------------------------------------------------------------------
    # Display the program guide if requested
    # -----------------------------------------------------------------------------------------------------------------
    if args.guide or not args.module:
        if not args.module:
            with open(PUtils.readme, 'r') as f:
                print(f.read(), end='')
        elif args.module == PUtils.analysis:
            logger.info(analysis_guide)
        elif args.module == PUtils.interface_design:
            logger.info(interface_design_guide)
        elif args.module == PUtils.nano:
            logger.info()
        elif args.module == 'expand_asu':
            logger.info()
        elif args.module == PUtils.select_designs:
            logger.info()
        elif args.module == PUtils.select_sequences:
            logger.info()
        elif args.module == 'visualize':
            logger.info('Usage: %s -r %s -- [-d %s, -df %s, -f %s] visualize --design_range 0-10'
                        % (ex_path('pymol'), PUtils.program_command.replace('python ', ''), ex_path('design_directory'),
                           ex_path('DataFrame.csv'), ex_path('design.paths')))
        exit()
    # -----------------------------------------------------------------------------------------------------------------
    # Process arguments for program initialization
    # -----------------------------------------------------------------------------------------------------------------
    # We have to ensure that if the user has provided it, the symmetry is correct
    if queried_flags['symmetry'] and queried_flags.get('sym_entry'):
        queried_flags['sym_entry'] = \
            parse_symmetry_to_sym_entry(queried_flags['symmetry'], sym_entry=queried_flags['sym_entry'])
    elif queried_flags.get('sym_entry'):
        queried_flags['sym_entry'] = SymEntry(int(queried_flags['sym_entry']))
    elif queried_flags['symmetry']:
        if queried_flags['symmetry'].lower()[:5] == 'cryst':
            # the symmetry information is in the pdb header
            queried_flags['symmetry'] = 'cryst'
        else:  # queried_flags['symmetry'] in possible_symmetries:
            queried_flags['sym_entry'] = parse_symmetry_to_sym_entry(queried_flags['symmetry'])
    sym_entry = queried_flags['sym_entry']

    initialize_modules = [PUtils.nano, PUtils.interface_design, PUtils.interface_metrics,  # PUtils.refine,
                          'optimize_designs', 'custom_script']  # PUtils.analysis,
    # TODO consolidate these checks
    if args.module in [PUtils.interface_design, PUtils.generate_fragments, 'orient', 'find_asu', 'expand_asu', 'status',
                       PUtils.interface_metrics, PUtils.refine, 'optimize_designs', 'custom_script', 'rename_chains',
                       'check_clashes', 'visualize']:
        initialize, queried_flags['construct_pose'] = True, True  # set up design directories
        # if args.module in ['orient', 'expand_asu']:
        #     if queried_flags['nanohedra_output'] or queried_flags['symmetry']:
        #         queried_flags['output_assembly'] = True
        #     else:
        #         logger.critical('Cannot %s without providing symmetry! Provide symmetry with "--symmetry"'
        #                         % args.module)
        #         exit(1)
    elif args.module in [PUtils.analysis, PUtils.cluster_poses]:
        # analysis could be run from Nanohedra docking, so we ensure that we don't construct new
        initialize, queried_flags['construct_pose'] = True, False
    elif args.module in [PUtils.select_designs, PUtils.select_sequences]:
        initialize, queried_flags['construct_pose'] = True, False
        # automatically skip logging as we are going to open a large number of files
        if not args.debug:
            queried_flags['skip_logging'] = True
        if args.module == PUtils.select_designs:
            if not args.metric:  # when not selecting by a metric, but by a dataframe, save time, don't initialize
                initialize = False
        elif args.module == PUtils.select_sequences:
            if not args.global_sequences and args.number_sequences == sys.maxsize:
                args.number_sequences = 1
    else:  # [PUtils.nano, 'guide', 'flags', 'residue_selector', 'multicistronic']
        initialize = False
        if getattr(args, 'query', None):  # run nanohedra query mode
            # if args.module == 'nanohedra_query':
            query_flags = [__file__, '-query'] + additional_args
            logger.debug('Query %s.py with: %s' % (PUtils.nano.title(), ', '.join(query_flags)))
            query_mode(query_flags)
            terminate(output=False)

    if not args.guide and args.module not in ['guide', 'flags', 'residue_selector', 'multicistronic']:
        formatted_queried_flags = queried_flags.copy()
        # where input values should be reported instead of processed version, or the argument is not important
        for flag in ['design_selector', 'construct_pose']:
            formatted_queried_flags.pop(flag, None)
            # get all the default program args and compare them to the provided values
        reported_args = {}
        for group in parser._action_groups:
            for arg in group._group_actions:
                # value = getattr(queried_flags, arg.dest, None)  # get the parsed flag value
                value = formatted_queried_flags.pop(arg.dest, None)  # get the parsed flag value
                if value and value != arg.default:  # compare it to the default
                    reported_args[arg.dest] = value  # add it to reported args if not the default
        # custom removal/formatting for all remaining
        for custom_arg in list(formatted_queried_flags.keys()):
            value = formatted_queried_flags.pop(custom_arg)
            if value:
                reported_args[custom_arg] = value

        sym_entry = reported_args.pop('sym_entry', None)
        if sym_entry:
            reported_args['sym_entry'] = sym_entry.entry_number
        logger.info('Starting with options:\n\t%s' % '\n\t'.join(SDUtils.pretty_format_table(reported_args.items())))
    # -----------------------------------------------------------------------------------------------------------------
    # Grab all Designs (DesignDirectory) to be processed from either database, directory, project name, or file
    # -----------------------------------------------------------------------------------------------------------------
    all_poses, design_directories, location = None, [], None
    all_dock_directories, pdb_pairs = None, None
    low, high, low_range, high_range = None, None, None, None

    if args.multi_processing:
        # Calculate the number of threads to use depending on computer resources
        threads = SDUtils.calculate_mp_threads(cores=args.cores)  # mpi=args.mpi, Todo
        logger.info('Starting multiprocessing using %d threads' % threads)
    else:
        threads = 1
        logger.info('Starting processing. If single process is taking awhile, use -mp during submission')

    # Set up JobResources, DesignDirectories input and outputs or Nanohedra inputs
    symdesign_directory = SDUtils.get_base_symdesign_dir(args.directory)
    if symdesign_directory:  # SymDesignOutput
        job = JobResources(symdesign_directory)
    elif args.output_directory:
        job = JobResources(queried_flags['output_directory'])
    else:
        symdesign_directory = os.path.join(os.getcwd(), PUtils.program_output)
        SDUtils.make_path(symdesign_directory)
        job = JobResources(symdesign_directory)
    queried_flags['job_resources'] = job

    initial_iter = None
    if initialize:
        if not args.directory and not args.file and not args.project and not args.single and not args.specification_file:
            raise SDUtils.DesignError(
                'No designs were specified! Please specify --directory, --file, --specification_file,'
                ' --project, or --single to locate designs of interest and run your command again')
        if args.nanohedra_output:
            all_poses, location = SDUtils.collect_nanohedra_designs(files=args.file, directory=args.directory)
        elif args.specification_file:  # Todo, combine this with collect_designs
            # # Grab all poses (directories) to be processed from either directory name or file
            # with open(args.specification_file) as csv_file:
            #     design_specification_dialect = Dialect()
            #     # csv_lines = [line for line in reader(csv_file)]
            #     all_poses, pose_design_numbers = zip(*reader(csv_file, dialect=))
            # # all_poses, pose_design_numbers = zip(*csv_lines)
            if not args.directory:
                raise SDUtils.DesignError('A --directory must be provided when using --specification_file')
            design_specification = SDUtils.DesignSpecification(args.specification_file)
            # Todo this works for file locations as well! should I have a separate mechanism for each?
            design_directories = [DesignDirectory.from_pose_id(pose, root=args.directory, specific_design=design,
                                                               directives=directives, **queried_flags)
                                  for pose, design, directives in design_specification.return_directives()]
            location = args.specification_file
        else:
            all_poses, location = SDUtils.collect_designs(files=args.file, directory=args.directory,
                                                          projects=args.project, singles=args.single)
        if args.design_range:
            low, high = map(float, args.design_range.split('-'))
            low_range, high_range = int((low / 100) * len(all_poses)), int((high / 100) * len(all_poses))
            if low_range < 0 or high_range > len(all_poses):
                raise SDUtils.DesignError('The input --design_range is outside of the acceptable bounds [0-%d]'
                                          % len(all_poses))
            logger.info('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))

        if all_poses:  # TODO fetch a state from files that have already been SymDesigned...
            if all_poses[0].count(os.sep) == 0:
                # assume that we have received pose-IDs and process accordingly
                # TODO another case, the list of files could be in the current directory that SymDesign was run in...
                if args.nanohedra_output:
                    queried_flags['sym_entry'] = get_sym_entry_from_nanohedra_directory(args.directory)
                design_directories = [DesignDirectory.from_pose_id(pose, root=args.directory, **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
            elif args.nanohedra_output:
                base_directory = '%s%s' % (os.sep, os.path.join(*all_poses[0].split(os.sep)[:-4]))
                queried_flags['sym_entry'] = get_sym_entry_from_nanohedra_directory(base_directory)
                design_directories = [DesignDirectory.from_nanohedra(pose, **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
            else:
                design_directories = [DesignDirectory.from_file(pose, **queried_flags)
                                      for pose in all_poses[low_range:high_range]]
        if not design_directories:
            raise SDUtils.DesignError('No %s directories found within "%s"! Please ensure correct '
                                      'location. Are you sure you want to run with -%s %s?'
                                      % (PUtils.program_name, location, 'nanohedra_output', args.nanohedra_output))
        # Todo could make after collect_designs? Pass to all design_directories
        #  for file, take all_poses first file. I think prohibits multiple dirs, projects, single...
        example_directory = next(iter(design_directories))
        # example_directory = JobResources(design_directories[0].program_root)
        if not location:
            design_source = os.path.basename(example_directory.project_designs)
        else:
            design_source = os.path.splitext(os.path.basename(location))[0]
        default_output_tuple = (SDUtils.starttime, args.module, design_source)

        # must make the output_directory before set_up_design_directory is called
        if args.output_directory is not None:
            if args.output_directory == '':
                designs_directory = '%s_%s_%s_poses' % default_output_tuple
            else:
                designs_directory = args.output_directory
            os.makedirs(designs_directory, exist_ok=True)
        master_db = Database(job.orient_dir, job.orient_asu_dir, job.refine_dir, job.full_model_dir, job.stride_dir,
                             job.sequences, job.profiles, sql=None)  # , log=logger)
        logger.info('Using design resources from Database located at "%s"' % job.protein_data)

        # Todo logic error when initialization occurs with module that doens't call this, subsequent runs are missing
        #  directories/resources that haven't been made
        # check to see that proper files have been created if doing design
        # including orientation, refinement, loop modeling, hhblits, bmdca?
        if not example_directory.initialized and args.module in initialize_modules \
                or args.nanohedra_output or args.load_database:  # or args.module == PUtils.nano
            # job.make_path(job.protein_data)
            # job.make_path(job.pdbs)
            # job.make_path(job.sequence_info)
            # job.make_path(job.sequences)
            # job.make_path(job.profiles)
            # job.make_path(job.job_paths)
            # job.make_path(job.sbatch_scripts)
            if args.load_database:  # Todo why is this set_up_design_directory here?
                for design in design_directories:
                    design.set_up_design_directory()
            # args.orient, args.refine = True, True  # Todo make part of argparse? Could be variables in NanohedraDB
            # for each design_directory, ensure that the pdb files used as source are present in the self.orient_dir
            orient_dir = job.orient_dir
            # job.make_path(orient_dir)
            orient_asu_dir = job.orient_asu_dir
            # job.make_path(orient_asu_dir)
            stride_dir = job.stride_dir
            # job.make_path(stride_dir)
            logger.critical('The requested poses require preprocessing before design modules should be used')
            # logger.info('The required files for %s designs are being collected and oriented if necessary' % PUtils.nano)
            # for design in design_directories:
            #     print(design.info.keys())
            # required_entities1 = set(design.entity_names[0] for design in design_directories)
            # required_entities2 = set(design.entity_names[1] for design in design_directories)

            # Collect all entities required for processing the given commands
            required_entities = list(map(set, list(zip(*[design.entity_names for design in design_directories]))))
            # all_entity_names = required_entities1.union(required_entities2)
            # all_entity_names = []
            # for entity_group in required_entities:
            #     all_entity_names.extend(entity_group)
            # all_entity_names = set(all_entity_names)
            all_entities = []
            load_resources = False
            # orient_files = [os.path.splitext(file)[0] for file in os.listdir(orient_dir)]
            # qsbio_confirmed = SDUtils.unpickle(PUtils.qs_bio)
            # Select entities, orient them, then load each entity to all_entities for further database processing
            symmetry_map = example_directory.sym_entry.sym_map.values() if example_directory.sym_entry else repeat(None)
            for symmetry, entities in zip(symmetry_map, required_entities):
                if not entities:
                    continue
                elif not symmetry or symmetry == 'C1':
                    logger.info('PDB files are being processed without consideration for symmetry: %s'
                                % ', '.join(entities))
                    continue
                    # example_directory.transform_d[idx]['translation'] = -center_of_mass
                    # example_directory.transform_d[idx]['rotation'] = some_guide_coord_based_rotation
                else:
                    logger.info('Ensuring PDB files are oriented with %s symmetry (stored at %s): %s'
                                % (symmetry, orient_dir, ', '.join(entities)))
                all_entities.extend(master_db.orient_entities(entities, symmetry=symmetry))

            info_messages = []
            # set up the hhblits and profile bmdca for each input entity
            profile_dir = job.profiles
            job.make_path(profile_dir)
            sequences_dir = job.sequences
            job.make_path(sequences_dir)
            hhblits_cmds, bmdca_cmds = [], []
            for entity in all_entities:
                entity.sequence_file = master_db.sequences.retrieve_file(name=entity.name)
                if not entity.sequence_file:  # Todo reference_sequence source accuracy throughout protocol
                    entity.write_fasta_file(entity.reference_sequence, name=entity.name, out_path=sequences_dir)
                    # entity.add_evolutionary_profile(out_path=master_db.hhblits_profiles.location)
                else:
                    entity.evolutionary_profile = master_db.hhblits_profiles.retrieve_data(name=entity.name)
                    # entity.h_fields = master_db.bmdca_fields.retrieve_data(name=entity.name)
                    # TODO reinstate entity.j_couplings = master_db.bmdca_couplings.retrieve_data(name=entity.name)
                if not entity.evolutionary_profile:
                    # to generate in current runtime
                    # entity.add_evolutionary_profile(out_path=master_db.hhblits_profiles.location)
                    # to generate in a sbatch script
                    # profile_cmds.append(entity.hhblits(out_path=profile_dir, return_command=True))
                    hhblits_cmds.append(entity.hhblits(out_path=profile_dir, return_command=True))
                # if not entity.j_couplings:  # TODO reinstate
                #     bmdca_cmds.append([PUtils.bmdca_exe_path, '-i', os.path.join(profile_dir, '%s.fasta' % entity.name),
                #                        '-d', os.path.join(profile_dir, '%s_bmDCA' % entity.name)])
            if hhblits_cmds:
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
                    'Once you are satisfied, enter the following to distribute hhblits jobs:\n\tsbatch %s' \
                    % hhblits_sbatch
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

            refine_loop_model_instructions, pre_refine, pre_loop_model = \
                master_db.preprocess_entities_for_design(all_entities, script_outpath=job.sbatch_scripts,
                                                         load_resources=load_resources)
            if load_resources or pre_refine or pre_loop_model:
                logger.critical(sbatch_warning)
                for message in info_messages + refine_loop_model_instructions:
                    logger.info(message)
                print('\n')
                logger.info('After completion of sbatch script(s), re-run your %s command:\n\tpython %s\n'
                            % (PUtils.program_name, ' '.join(sys.argv)))
                terminate(output=False)
                # After completion of sbatch, the next time initialized, there will be no refine files left allowing
                # initialization to proceed
        else:
            pre_refine = None  # False
            pre_loop_model = None  # False

        if args.multi_processing:  # and not args.skip_master_db:
            # Todo tweak behavior of these two parameters. Need Queue based DesignDirectory
            master_db.load_all_data()
            # SDUtils.mp_map(DesignDirectory.set_up_design_directory, design_directories, threads=threads)
            # SDUtils.mp_map(DesignDirectory.link_master_database, design_directories, threads=threads)
        # else:  # for now just do in series
        for design in design_directories:
            # design.link_database(resource_db=master_db)
            design.set_up_design_directory(pre_refine=pre_refine, pre_loop_model=pre_loop_model)

        logger.info('%d unique poses found in "%s"' % (len(design_directories), location))
        if not args.debug and not queried_flags['skip_logging']:
            example_log = getattr(example_directory.log.handlers[0], 'baseFilename', None)
            if example_log:
                logger.info('All design specific logs are located in their corresponding directories.\n\tEx: %s'
                            % example_log)

    elif args.module == PUtils.nano:
        master_db = Database(job.orient_dir, job.orient_asu_dir, job.refine_dir, job.full_model_dir, job.stride_dir,
                             job.sequences, job.profiles, sql=None)  # , log=logger)
        logger.info('Using design resources from Database located at "%s"' % job.protein_data)
        if args.directory or args.file:
            all_dock_directories, location = SDUtils.collect_nanohedra_designs(files=args.file,
                                                                               directory=args.directory, dock=True)
            design_directories = [DesignDirectory.from_nanohedra(dock_dir, dock=True,  # mode=args.directory_type,
                                                                 project=args.project, **queried_flags)
                                  for dock_dir in all_dock_directories]
            if not design_directories:
                raise SDUtils.DesignError('No docking directories/files were found!\n'
                                          'Please specify --directory1, and/or --directory2 or --directory or '
                                          '--file. See %s' % PUtils.help(args.module))
            # master_directory = next(iter(design_directories))
            logger.info('%d unique building block docking combinations found in "%s"'
                        % (len(design_directories), location))
        else:
            # if args.output_directory:
            #     master_directory = JobResources(queried_flags['output_directory'])
            # else:
            #     master_directory = JobResources(os.path.join(os.getcwd(), PUtils.program_output))
            # Todo make current with sql ambitions
            job.docking_master_dir = os.path.join(job.projects, 'NanohedraEntry%dDockedPoses' % sym_entry.entry_number)
            # sym_entry is required so this won't fail ^

            # Getting PDB1 and PDB2 File paths
            symmetry_map = sym_entry.groups
            all_entities = []
            load_resources = False
            orient_log = SDUtils.start_log(name='orient', handler=2, propagate=True,
                                           location=os.path.join(os.path.dirname(args.oligomer1),
                                                                 PUtils.orient_log_file))
            if args.query_codes:
                # raise SDUtils.DesignError('This functionality is not yet available. Just connect Query.PDB.__main__')
                if validate_input('Do you want to save the PDB query?', {'y': True, 'n': False}):
                    args.save_query = True
                else:
                    args.save_query = False
                entities1 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
                entities2 = retrieve_pdb_entries_by_advanced_query(save=args.save_query, entity=True)
            else:
                if args.pdb_codes1:
                    entities1 = set(SDUtils.to_iterable(args.pdb_codes1))
                    # all_entities.extend(master_db.orient_entities(entities1, symmetry=symmetry_map[0]))
                else:
                    # args.oligomer1:
                    # Orient Input Oligomers to Canonical Orientation
                    # oriented_pdb1_out_dir = os.path.join(os.path.dirname(args.oligomer1), '%s_oriented_with_%s_symmetry'
                    #                                      % (os.path.basename(args.oligomer1), oligomer_symmetry_1))
                    # os.makedirs(oriented_pdb1_out_dir, exist_ok=True)

                    if '.pdb' in args.oligomer1:
                        pdb1_filepaths = [args.oligomer1]
                    else:
                        pdb1_filepaths = SDUtils.get_all_pdb_file_paths(args.oligomer1)
                    pdb1_oriented_filepaths = \
                        [orient_pdb_file(file, log=orient_log, symmetry=symmetry_map[0],
                                         out_dir=master_db.oriented.location)
                         for file in pdb1_filepaths]
                    entities1 = list(map(os.path.basename,
                                         [os.path.splitext(file)[0] for file in filter(None, pdb1_oriented_filepaths)]))
                    # logger.info('%d filepaths found' % len(pdb1_oriented_filepaths))
                    # pdb1_oriented_filepaths = filter(None, pdb1_oriented_filepaths)
            # logger.info('Orienting PDB\'s for Nanohedra Docking')
            all_entities.extend(master_db.orient_entities(entities1, symmetry=symmetry_map[0]))

            single_component_design = False
            if args.oligomer2:
                if args.oligomer1 != args.oligomer2:  # see if they are the same input
                    # oriented_pdb2_out_dir = os.path.join(os.path.dirname(args.oligomer2),
                    #                                      '%s_oriented_with_%s_symmetry'
                    #                                      % (os.path.basename(args.oligomer2), oligomer_symmetry_2))
                    # if not os.path.exists(oriented_pdb2_out_dir):
                    #     os.makedirs(oriented_pdb2_out_dir)
                    if '.pdb' in args.oligomer2:
                        pdb2_filepaths = [args.oligomer2]
                    else:
                        pdb2_filepaths = SDUtils.get_all_pdb_file_paths(args.oligomer2)
                    pdb2_oriented_filepaths = \
                        [orient_pdb_file(file, log=orient_log, symmetry=symmetry_map[1],
                                         out_dir=master_db.oriented.location)
                         for file in pdb2_filepaths]
                    entities2 = list(map(os.path.basename,
                                         [os.path.splitext(file)[0] for file in filter(None, pdb2_oriented_filepaths)]))
                else:  # the entities are the same symmetry or we have single component and bad input
                    entities2 = []
            elif args.pdb_codes2:
                # Collect all entities required for processing the given commands
                entities2 = set(SDUtils.to_iterable(args.pdb_codes2))
                # Select entities, orient them, then load each entity to all_entities for further database processing
                # for symmetry, entities in zip(symmetry_map, required_entities):
                #     if not entities:
                #         continue
                #     elif not symmetry or symmetry == 'C1':
                #         logger.info('PDB files are being processed without consideration for symmetry: %s'
                #                     % ', '.join(entities))
                #         continue
                #         # example_directory.transform_d[idx]['translation'] = -center_of_mass
                #         # example_directory.transform_d[idx]['rotation'] = some_guide_coord_based_rotation
                #     else:
                #         logger.info('Ensuring PDB files are oriented with %s symmetry (stored at %s): %s'
                #                     % (symmetry, job.orient_dir, ', '.join(entities)))
            else:
                entities2 = []
                # if not entities2:
                logger.info('No additional entities requested for docking, treating as single component')
                single_component_design = True
            all_entities.extend(master_db.orient_entities(entities2, symmetry=symmetry_map[1]))

            info_messages = []
            refine_loop_model_instructions, pre_refine, pre_loop_model = \
                master_db.preprocess_entities_for_design(all_entities, script_outpath=job.sbatch_scripts,
                                                         load_resources=load_resources)
            if load_resources or pre_refine or pre_loop_model:
                logger.critical(sbatch_warning)
                for message in info_messages + refine_loop_model_instructions:
                    logger.info(message)
                print('\n')
                logger.info('After completion of sbatch script(s), re-run your %s command:\n\tpython %s\n'
                            % (PUtils.program_name, ' '.join(sys.argv)))
                terminate(output=False)
                # After completion of sbatch, the next time command is entered docking will proceed

            # make all possible pdb_pairs given input entities
            entities1 = [entity for entity in all_entities if entity.name in entities1]
            entities2 = [entity for entity in all_entities if entity.name in entities2]
            pdb_pairs = list(product(entities1, entities2))
            # pdb_pairs = list(combinations(entities1, 2))
            # pdb_pairs = list(combinations(pdb1_oriented_filepaths, 2))
            # pdb_pairs = list(combinations(SDUtils.get_all_pdb_file_paths(oriented_pdb1_out_dir), 2))
            location = args.oligomer1
            initial_iter = [False for _ in range(len(pdb_pairs))]
            initial_iter[0] = True
        design_source = os.path.splitext(os.path.basename(location))[0]
    else:  # this logic is possible with select_designs without --metric
        master_db = None
        # design_source = os.path.basename(example_directory.project_designs)
        # job = JobResources(queried_flags['output_directory'])
        pass

    if args.module in [PUtils.nano, PUtils.interface_design]:
        if args.run_in_shell:
            logger.info('Modeling will occur in this process, ensure you don\'t lose connection to the shell!')
        else:
            logger.info('Writing modeling commands out to file, no modeling will occur until commands are executed')

    if queried_flags.get(PUtils.generate_fragments, None) or not queried_flags.get('no_term_constraint', None) \
            or args.module in [PUtils.nano, PUtils.generate_fragments]:
        interface_type = 'biological_interfaces'  # Todo parameterize
        logger.info('Initializing %s FragmentDatabase\n' % interface_type)
        fragment_db = SDUtils.unpickle(PUtils.biological_fragment_db_pickle)
        # fragment_db.location = PUtils.frag_directory.get(fragment_db.source, None)  # has since been depreciated
        # fragment_db = FragmentDatabase(source=interface_type, init_db=True)  # Todo sql=args.frag_db
        euler_lookup = EulerLookup()
    else:
        fragment_db, euler_lookup = None, None

    # for design in design_directories:
    #     design.link_database(frag_db=fragment_db, resource_db=master_db)
    #     design.euler_lookup = euler_lookup
    job.fragment_db = fragment_db
    job.resources = master_db
    job.euler_lookup = euler_lookup
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
    if args.module == 'flags':
        if args.template:
            Flags.query_user_for_flags(template=True)
        else:
            Flags.query_user_for_flags(mode=args.flags_module)
    # ---------------------------------------------------
    # elif args.module == 'distribute':  # -s stage, -y success_file, -n failure_file, -m max_jobs
    #     distribute(**vars(args))
    # ---------------------------------------------------
    elif args.module == 'residue_selector':  # Todo
        if not args.single:
            raise SDUtils.DesignError('You must pass a single pdb file to %s. Ex:\n\t%s --single my_pdb_file.pdb '
                                      'residue_selector' % (PUtils.program_name, PUtils.program_command))
        fasta_file = generate_sequence_template(args.single)
        logger.info('The residue_selector template was written to %s. Please edit this file so that the '
                    'residue_selector can be generated for protein design. Selection should be formatted as a "*" '
                    'replaces all sequence of interest to be considered in design, while a Mask should be formatted as '
                    'a "-". Ex:\n>pdb_template_sequence\nMAGHALKMLV...\n>residue_selector\nMAGH**KMLV\n\nor'
                    '\n>pdb_template_sequence\nMAGHALKMLV...\n>design_mask\nMAGH----LV\n'
                    % fasta_file)
    # ---------------------------------------------------
    elif args.module == 'orient':
        args.to_design_directory = True  # default to True when using this module
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.to_design_directory))
            results = SDUtils.mp_starmap(DesignDirectory.orient, zipped_args, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.orient(to_design_directory=args.to_design_directory))

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'find_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.find_asu, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.find_asu())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'expand_asu':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.expand_asu, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.expand_asu())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'rename_chains':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.rename_chains, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.rename_chains())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'check_clashes':
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.check_clashes, design_directories, threads=threads)
        else:
            for design_dir in design_directories:
                results.append(design_dir.check_clashes())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.generate_fragments or queried_flags.get(PUtils.generate_fragments):
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.generate_interface_fragments, design_directories, threads=threads)
        else:
            for design in design_directories:
                results.append(design.generate_interface_fragments())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.nano:  # -o1 oligomer1, -o2 oligomer2, -e entry, -o outdir
        # Initialize docking procedure
        if args.run_in_shell:
            # make master output directory
            os.makedirs(job.docking_master_dir, exist_ok=True)
            if args.debug:
                # Root logs to stream with level debug according to prior logging initialization
                master_logger, bb_logger = logger, logger
            else:
                master_log_filepath = os.path.join(args.output_directory, PUtils.master_log)
                master_logger = SDUtils.start_log(name=PUtils.nano.title(), propagate=True,
                                                  handler=2, location=master_log_filepath)
            master_logger.info('Nanohedra\nMODE: DOCK\n\n')
            write_docking_parameters(args.oligomer1, args.oligomer2, args.rot_step_deg1, args.rot_step_deg2, sym_entry,
                                     job.docking_master_dir, log=master_logger)
        if args.multi_processing:
            if args.run_in_shell:
                zipped_args = zip(repeat(sym_entry), repeat(fragment_db), repeat(euler_lookup),
                                  repeat(job.docking_master_dir), *zip(*pdb_pairs),
                                  repeat(args.rot_step_deg1), repeat(args.rot_step_deg2),
                                  repeat(args.min_matched), repeat(args.high_quality_match_value),
                                  repeat(args.output_assembly), repeat(args.output_surrounding_uc))
                SDUtils.mp_starmap(nanohedra_dock, zipped_args, threads=threads)
            else:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    zipped_args = zip(repeat(args.entry), *zip(*pdb_pairs), repeat(args.outdir), repeat(args.project),
                                      initial_iter)
                    results = SDUtils.mp_starmap(nanohedra_command, zipped_args, threads=threads)
                else:  # args.directory or args.file set up docking directories
                    zipped_args = zip(design_directories, repeat(args.project))
                    results = SDUtils.mp_starmap(nanohedra_design_recap, zipped_args, threads=threads)
        else:
            if args.run_in_shell:
                for pdb_pair in pdb_pairs:
                    pdb1, pdb2 = pdb_pair
                    master_logger.info('Docking %s / %s \n' % (pdb1.name, pdb2.name))
                    nanohedra_dock(sym_entry, fragment_db, euler_lookup, job.docking_master_dir,
                                   pdb1, pdb2, rot_step_deg1=args.rot_step_deg1, rot_step_deg2=args.rot_step_deg2,
                                   min_matched=args.min_matched, high_quality_match_value=args.high_quality_match_value,
                                   output_assembly=args.output_assembly,
                                   output_surrounding_uc=args.output_surrounding_uc)
                    # clash_dist=2.2, init_max_z_val=1., subseq_max_z_val=2.
            else:
                if pdb_pairs and initial_iter:  # using combinations of directories with .pdb files
                    for initial, (path1, path2) in zip(initial_iter, pdb_pairs):
                        result = nanohedra_command(args.entry, path1, path2, args.outdir, args.project,
                                                   initial)
                        results.append(result)
                else:  # single directory docking (already made directories)
                    for dock_directory in design_directories:
                        result = nanohedra_design_recap(dock_directory, args.project)
                        results.append(result)
        design_source = args.directory  # for terminate()
        terminate(results=results, output=False)
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
        #     logger.info('All "%s" commands were written to "%s"' % (PUtils.nano, command_file))
        # else:
        #     logger.error('No "%s" commands were written!' % PUtils.nano)
    # ---------------------------------------------------
    elif args.module == 'interface_metrics':
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # zipped_args = zip(design_directories, repeat(args.force_flags), repeat(queried_flags.get('development')))
            results = SDUtils.mp_map(DesignDirectory.rosetta_interface_metrics, design_directories, threads=threads)
        else:
            for design in design_directories:
                # if design.sym_entry is None:
                #     continue
                results.append(design.rosetta_interface_metrics())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == 'optimize_designs':
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            # zipped_args = zip(design_directories, repeat(args.force_flags), repeat(queried_flags.get('development')))
            results = SDUtils.mp_map(DesignDirectory.optimize_designs, design_directories, threads=threads)
        else:
            for design in design_directories:
                results.append(design.optimize_designs())

        terminate(results=results)
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

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.refine:  # -i fragment_library, -s scout
        args.to_design_directory = True  # always the case when using this module
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.to_design_directory), repeat(args.interface_to_alanine),
                              repeat(args.gather_metrics))
            results = SDUtils.mp_starmap(DesignDirectory.refine, zipped_args, threads=threads)
        else:
            for design in design_directories:
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
        if queried_flags['design_with_evolution']:
            if psutil.virtual_memory().available <= hhblits_memory_threshold:
                logger.critical('The amount of virtual memory for the computer is insufficient to run hhblits '
                                '(the backbone of --design_with_evolution)! Please allocate the job to a computer with '
                                'more memory or the process will fail. Otherwise, select --design_with_evolution False')
            job.make_path(job.sequences)
            job.make_path(job.profiles)
        # Start pose processing and preparation for Rosetta
        if args.multi_processing:
            results = SDUtils.mp_map(DesignDirectory.interface_design, design_directories, threads=threads)
        else:
            for design in design_directories:
                results.append(design.interface_design())

        terminate(results=results)
    # ---------------------------------------------------
    elif args.module == PUtils.analysis:  # -o output, -f figures, -n no_save, -j join
        if args.no_save:
            save = False
        else:
            save = True
        # job = next(iter(design_directories))
        # ensure analysis write directory exists
        job.make_path(job.all_scores)
        # Start pose analysis of all designed files
        if args.output_file == PUtils.analysis_file:
            args.output_file = os.path.join(job.all_scores, args.output_file % (SDUtils.starttime, design_source))
        elif len(args.output_file.split(os.sep)) <= 1:  # the path isn't an absolute or relative path, prepend location
            args.output_file = os.path.join(job.all_scores, args.output_file)

        if os.path.exists(args.output_file):
            logger.critical('The specified output file "%s" already exists, this will overwrite your old analysis '
                            'data! Please modify that file or specify a new one with with -of/--output_file'
                            % args.output_file)
            exit(1)
        if args.multi_processing:
            zipped_args = zip(design_directories, repeat(args.join), repeat(save), repeat(args.figures))
            results = SDUtils.mp_starmap(DesignDirectory.design_analysis, zipped_args, threads=threads)
        else:
            for design in design_directories:
                results.append(design.design_analysis(merge_residue_data=args.join, save_trajectories=save,
                                                      figures=args.figures))
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
        if args.dataframe and not args.directory:
            logger.critical('If using a --dataframe for selection, you must include the directory where the designs are'
                            ' located in order to properly select designs. Please specify -d/--directory on the command'
                            ' line')
            # logger.critical('If using a --dataframe for selection, you must include the directory where the designs are'
            #                 'located in order to properly select designs. Please specify -d/--directory, -p/--project, '
            #                 'or -s/--single on the command line') TODO
            exit(1)
            program_root = None
            # Todo change this mechanism so not reliant on args.directory and outputs pose IDs/ Alternatives fix csv
            #  to output paths
        # elif not job:
        #     program_root = args.directory
        else:
            program_root = job.program_root

        if args.dataframe:
            # Figure out poses from a dataframe, filters, and weights
            selected_poses_df = prioritize_design_indices(args.dataframe, filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            selected_poses = selected_poses_df.index.to_list()
            logger.info('%d poses were selected' % len(selected_poses_df))  # :\n\t%s , '\n\t'.join(selected_poses)))
            if args.filter or args.weight:
                new_dataframe = os.path.join(program_root, '%s-%s%sDesignPoseMetrics.csv'
                                             % (SDUtils.starttime, 'Filtered' if args.filter else '',
                                                'Weighted' if args.weight else '',))
                selected_poses_df.to_csv(new_dataframe)
                logger.info('New DataFrame was written to %s' % new_dataframe)

            # Sort results according to clustered poses if clustering exists  # Todo parameterize name
            if args.cluster_map:
                cluster_map = args.cluster_map
            else:
                # cluster_map = os.path.join(next(iter(design_directories)).protein_data, '%s.pkl' % PUtils.clustered_poses)
                cluster_map = os.path.join(program_root, PUtils.data.title(), '%s.pkl' % PUtils.clustered_poses)
            if os.path.exists(cluster_map):
                pose_cluster_map = SDUtils.unpickle(cluster_map)
            else:
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
                    if len(selected_poses) > 1000:
                        queried_flags['skip_logging'] = True
                    design_directories = [DesignDirectory.from_pose_id(pose, root=program_root, **queried_flags)
                                          for pose in selected_poses]
                    compositions = group_compositions(design_directories)
                    if args.multi_processing:
                        results = SDUtils.mp_map(cluster_designs, compositions.values(), threads=threads)
                        pose_cluster_map = {}
                        for result in results:
                            pose_cluster_map.update(result.items())
                    else:
                        pose_cluster_map = {}
                        for composition_group in compositions.values():
                            pose_cluster_map.update(cluster_designs(composition_group))
                    # cluster_representative_pose_member_string_map = \
                    #     {str(representative): str(member)
                    #      for representative, members in pose_cluster_map.items()
                    #      for member in members}
                    pose_cluster_file = SDUtils.pickle_object(pose_cluster_map,
                                                              PUtils.clustered_poses % (location, SDUtils.starttime),
                                                              out_path=next(iter(design_directories)).protein_data)
                    logger.info('Found %d unique clusters from %d pose inputs. All clusters stored in %s'
                                % (len(pose_cluster_map), len(design_directories),
                                   pose_cluster_file))
                else:
                    pose_cluster_map = {}

            if pose_cluster_map:
                # {design_string: [design_string, ...]} where key is representative, values are matching designs
                # OLD -> {composition: {design_string: cluster_representative}, ...}
                pose_cluster_membership_map = invert_cluster_map(pose_cluster_map)
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
            design_source = program_root  # for terminate()
            # write out the chosen poses to a pose.paths file
            terminate(results=design_directories)
        else:  # generate design metrics on the spot
            logger.debug('Collecting designs to sort')
            if args.metric == 'score':
                metric_design_dir_pairs = [(des_dir.score, des_dir.path) for des_dir in design_directories]
            elif args.metric == 'fragments_matched':
                metric_design_dir_pairs = [(des_dir.number_of_fragments, des_dir.path)
                                           for des_dir in design_directories]
            else:
                raise SDUtils.DesignError('The metric "%s" is not supported!' % args.metric)

            logger.debug('Sorting designs according to "%s"' % args.metric)
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
        pose_cluster_map = {}
        if args.mode == 'ialign':  # interface_residues, tranformation
            is_threshold = 0.4  # 0.5  # TODO
            # measure the alignment of all selected design_directories
            # all_files = [design.source_file for design in design_directories]

            # need to change directories to prevent issues with the path length being passed to ialign
            prior_directory = os.getcwd()
            os.chdir(job.protein_data)  # os.path.join(job.protein_data, 'ialign_output'))
            temp_file_dir = os.path.join(os.getcwd(), 'temp')
            if not os.path.exists(temp_file_dir):
                os.makedirs(temp_file_dir)

            # save the interface for each design to the temp directory
            design_interfaces = []
            for design in design_directories:
                design.identify_interface()  # calls design.load_pose()
                interface = design.pose.return_interface()
                design_interfaces.append(
                    # interface.write(out_path=os.path.join(temp_file_dir, '%s_interface.pdb' % design.name)))  # Todo reinstate
                    interface.write(out_path=os.path.join(temp_file_dir, '%s.pdb' % design.name)))

            design_directory_pairs = list(combinations(design_directories, 2))
            design_pairs = []
            if args.multi_processing:
                # zipped_args = zip(combinations(design_interfaces, 2))
                design_scores = SDUtils.mp_starmap(ialign, combinations(design_interfaces, 2), threads=threads)

                for idx, is_score in enumerate(design_scores):
                    if is_score > is_threshold:
                        design_pairs.append(set(design_directory_pairs[idx]))
            else:
                # for design1, design2 in combinations(design_directories, 2):  # all_files
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
            compositions = group_compositions(design_directories)
            if args.multi_processing:
                results = SDUtils.mp_map(cluster_designs, compositions.values(), threads=threads)
                for result in results:
                    pose_cluster_map.update(result.items())
            else:
                # pose_map = pose_rmsd_s(design_directories)
                # pose_cluster_map = cluster_poses(pose_map)
                for composition_group in compositions.values():
                    pose_cluster_map.update(cluster_designs(composition_group))
        # elif args.mode == 'interface_residues':
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
        if args.global_sequences:
            df = load_global_dataframe()
            if args.protocol:
                group_df = df.groupby('protocol')
                df = pd.concat([group_df.get_group(x) for x in group_df.groups], axis=1,
                               keys=list(zip(group_df.groups, repeat('mean'))))
            else:
                df = pd.concat([df], axis=1, keys=['pose', 'metric'])
            # Figure out designs from dataframe, filters, and weights
            selected_poses_df = prioritize_design_indices(df, filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            design_indices = selected_poses_df.index.to_list()
            if args.allow_multiple_poses:
                logger.info('Choosing maximum %d designs as specified, from the top ranked designs regardless of pose'
                            % args.number_sequences)
                results = design_indices[:args.number_sequences]
            else:
                logger.info('Choosing maximum %d designs as specified, with only one design allowed per pose'
                            % args.number_sequences)
                number_chosen = 0
                results, selected_designs = [], set()
                for design_directory, design in design_indices:
                    if design_directory not in selected_designs:
                        selected_designs.add(design_directory)
                        results.append((design_directory, design))
                        number_chosen += 1
                        if number_chosen == args.number_sequences:
                            break
            logger.info('%d designs were selected' % len(results))

            if args.filter or args.weight:
                new_dataframe = \
                    os.path.join(program_root, '%s-%s%sDesignPoseMetrics.csv'
                                 % (SDUtils.starttime, 'Filtered' if args.filter else '', 'Weighted' if args.weight else ''))
            else:
                new_dataframe = os.path.join(program_root, '%s-DesignPoseMetrics.csv' % SDUtils.starttime)
            # include only the found index names to the saved dataframe
            save_poses_df = selected_poses_df.loc[results, :].droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        elif args.specification_file:
            results = [(design_directory, design_directory.specific_design) for design_directory in design_directories]
            df = load_global_dataframe()
            selected_poses_df = prioritize_design_indices(df.loc[results, :], filter=args.filter, weight=args.weight,
                                                          protocol=args.protocol, function=args.weight_function)
            # specify the result order according to any filtering and weighting
            # results = selected_poses_df.index.to_list()  TODO reinstate
            save_poses_df = selected_poses_df.droplevel(0).droplevel(0, axis=1).droplevel(0, axis=1)
        else:  # select designed sequences from each pose provided (DesignDirectory)
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
                zipped_args = zip(design_directories, repeat(sequence_filters), repeat(sequence_weights),
                                  repeat(args.number_sequences), repeat(args.protocol))
                # result_mp = zip(*SDUtils.mp_starmap(Ams.select_sequences, zipped_args, threads))
                result_mp = SDUtils.mp_starmap(DesignDirectory.select_sequences, zipped_args, threads)
                # results - contains tuple of (DesignDirectory, design index) for each sequence
                # could simply return the design index then zip with the directory
                results = []
                for result in result_mp:
                    results.extend(result)
            else:
                results = []
                for design in design_directories:
                    results.extend(design.select_sequences(filters=sequence_filters, weights=sequence_weights,
                                                           number=args.number_sequences, protocols=args.protocol))
            save_poses_df = None  # Todo make possible!

        if not args.selection_string:
            args.selection_string = '%s_' % os.path.basename(os.path.splitext(location)[0])
        else:
            args.selection_string += '_'
        outdir = os.path.join(os.path.dirname(program_root), '%sSelectedDesigns' % args.selection_string)
        # outdir_traj, outdir_res = os.path.join(outdir, 'Trajectories'), os.path.join(outdir, 'Residues')
        if not os.path.exists(outdir):
            os.makedirs(outdir)  # , os.makedirs(outdir_traj), os.makedirs(outdir_res)

        if save_poses_df is not None:
            selection_trajectory_df_file = os.path.join(outdir, 'TrajectoryMetrics.csv')
            save_poses_df.to_csv(selection_trajectory_df_file)
            logger.info('New DataFrame with selected designs was written to %s' % selection_trajectory_df_file)

        logger.info('Relevant design files are being copied to the new directory: %s' % outdir)
        # Create new output of designed PDB's  # TODO attach the state to these files somehow for further SymDesign use
        for des_dir, design in results:
            file_path = os.path.join(des_dir.designs, '*%s*' % design)
            file = glob(file_path)
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
        design_directories = [des_dir for des_dir, design in results]
        if args.skip_sequence_generation:
            terminate(output=False)
        else:
            with open(args.output_file, 'w') as f:
                f.write('%s\n' % '\n'.join(des_dir.path for des_dir in design_directories))

        # use one directory as indication of entity specification for them all. Todo modify for different length inputs
        example_directory.load_pose()
        if args.entity_specification:
            if args.entity_specification == 'all':
                tag_index = [True for _ in example_directory.pose.entities]
                number_of_tags = len(example_directory.pose.entities)
            elif args.entity_specification == 'single':
                tag_index = [True for _ in example_directory.pose.entities]
                number_of_tags = 1
            elif args.entity_specification == 'none':
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
        for des_dir, design in results:
            file = glob('%s/*%s*' % (des_dir.designs, design))
            if not file:
                logger.error('No file found for %s' % '%s/*%s*' % (des_dir.designs, design))
                continue
            design_pose = PDB.from_file(file[0], log=des_dir.log, entity_names=des_dir.entity_names)
            designed_atom_sequences = [entity.structure_sequence for entity in design_pose.entities]

            des_dir.load_pose(source=des_dir.asu)
            des_dir.pose.pdb.reorder_chains()  # Do I need to modify chains?
            missing_tags[(des_dir, design)] = [1 for _ in des_dir.pose.entities]
            prior_offset = 0
            # all_missing_residues = {}
            # mutations = []
            # referenced_design_sequences = {}
            sequences_and_tags = {}
            entity_termini_availability, entity_helical_termini = {}, {}
            for idx, (source_entity, design_entity) in enumerate(zip(des_dir.pose.entities, design_pose.entities)):
                # source_entity.retrieve_info_from_api()
                source_entity.retrieve_sequence_from_api(entity_id=source_entity.name)
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
                true_termini = [term for term, is_true in termini_availability.items() if is_true]
                logger.debug('The termini %s are available for tagging' % termini_availability)
                entity_termini_availability[design_string] = termini_availability
                # Find sequence specified attributes required for expression formatting
                # disorder = generate_mutations(source_entity.structure_sequence, source_entity.reference_sequence,
                #                               only_gaps=True)
                disorder = source_entity.disorder
                indexed_disordered_residues = \
                    {residue + source_entity.offset + prior_offset: mutation for residue, mutation in disorder.items()}
                prior_offset += len(disorder)  # Todo, moved below indexed_disordered_residues on 7/26, ensure correct!
                # generate the source TO design mutations before any disorder handling
                mutations = \
                    generate_mutations(source_entity.structure_sequence, design_entity.structure_sequence, offset=False)
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
                all_insertions.update(generate_mutations(designed_atom_sequences[idx], design_sequence, blanks=True))
                # Reduce to sequence only
                inserted_sequences[design_string] = '%s\n%s' % (''.join([res['to'] for res in all_insertions.values()]),
                                                                design_sequence)
                logger.info('Formatted sequence comparison:\n%s' % inserted_sequences[design_string])
                final_sequences[design_string] = design_sequence
                if args.nucleotide:
                    try:
                        nucleotide_sequence = optimize_protein_sequence(design_sequence, species=args.optimize_species)
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
                                    # for name, sequence in zip(*reader(f))]
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
    #     for des_directory in design_directories:
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
    #             accessed_dirs = [rsync_dir(design_directory) for design_directory in design_directories]
    # ---------------------------------------------------
    elif args.module == 'status':  # -n number, -s stage, -u update
        if args.update:
            for design in design_directories:
                update_status(design.serialized_info, args.stage, mode=args.update)
        else:
            if args.number_of_trajectories:
                logger.info('Checking for %d files based on --number_of_trajectories flag' % args.number_of_trajectories)
            if args.stage:
                status(design_directories, args.stage, number=args.number_of_trajectories)
            else:
                for stage in PUtils.stage_f:
                    s = status(design_directories, stage, number=args.number_of_trajectories)
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
            # for design in design_directories:
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
                df_glob = glob(os.path.join(file_dir, 'TrajectoryMetrics.csv'))
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
                    if os.path.splitext(os.path.basename(file))[0] in index:
                    # if index in file:
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
