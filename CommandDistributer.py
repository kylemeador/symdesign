"""
Module for Distribution of Rosetta commands found for individual poses to SLURM/PBS computational cluster
Finds commands within received directories (poses)
"""

import argparse
# import SymDesignUtils as SDUtils
import multiprocessing as mp
import os
import signal
import subprocess
from random import random

import CmdUtils as CUtils
import PathUtils as PUtils


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        with open(args.failure_file, 'a') as f:
            for i, pose in enumerate(poses):
                f.write('%s\n' % pose)

        # Append SLURM output to log_file(s)
        job_id = int(os.environ.get('SLURM_JOB_ID'))
        file = 'output/%s_%s.out' % (job_id, array_task)
        for i, task_id in enumerate(range(cmd_slice, final_cmd_slice)):
            # file = '%s_%s.out' % (job_id, task_id)
            run(file, log_files[i], program='cat')
            # run(file, '/dev/null', program='rm')
        return None


def exit_gracefully(signum, frame):
    with open(args.failure_file, 'a') as f:
        for i, pose in enumerate(poses):
            f.write('%s\n' % pose)

    # Append SLURM output to log_file(s)
    job_id = int(os.environ.get('SLURM_JOB_ID'))
    file = 'output/%s_%s.out' % (job_id, array_task)
    for i, task_id in enumerate(range(cmd_slice, final_cmd_slice)):
        # file = '%s_%s.out' % (job_id, task_id)
        run(file, log_files[i], program='cat')
        # run(file, '/dev/null', program='rm')


def create_file(file):
    if not os.path.exists(file):
        with open(file, 'w') as new_file:
            dummy = True


# 2.7 compatible #


def merge_names(a, b):
    return '{} & {}'.format(a, b)


def merge_names_unpack(args):
    return merge_names(*args)


def unpack_mp_args(args):
    return run(*args)


# 2.7 compatible #


def run(cmd, log_file, program='bash'):  # , log_file=None):
    """Executes specified command and appends command results to log file

    Args:
        cmd (str): The name of a command file which should be executed by the system
        log_file (str): Location on disk of log file
    Keyword Args:
        program='bash' (str): The interpreter for said command
    Returns:
        (bool): Whether or not command executed successfully
    """
    # des_dir = SDUtils.DesignDirectory(os.path.dirname(cmd))
    # if not log_file:
    #     log_file = os.path.join(des_dir.path, os.path.basename(des_dir.path) + '.log')
    with open(log_file, 'a') as log_f:
        p = subprocess.Popen([program, cmd], stdout=log_f, stderr=log_f)
        p.communicate()

    if p.returncode == 0:
        return True
    else:
        return False


def distribute(args, logger):
    if args.file:
        # _commands, location = SDUtils.collect_directories(args.directory, file=args.file)
        pass
    else:
        logger.error('Error: You must pass a file containing a list of commands to process. This is typically output to'
                     ' a \'stage.cmd\' file. Ensure that this file exists and resubmit with -f \'stage.cmd\', replacing'
                     ' stage with the desired stage.')

    # Automatically detect if the commands file has executable scripts
    script_present = '-C'
    for _command in _commands:
        if not os.path.exists(_command):
            script_present = False
            break

    # Create success and failures files
    ran_num = int(100 * random())
    if not args.success_file:
        args.success_file = os.path.join(args.directory, '%s_sbatch-%d_success.log' % (args.stage, ran_num))
    if not args.failure_file:
        args.failure_file = os.path.join(args.directory, '%s_sbatch-%d_failures.log' % (args.stage, ran_num))
    logger.info('\nSuccessful poses will be listed in \'%s\'\nFailed poses will be listed in \'%s\''
                % (args.success_file, args.failure_file))

    # Grab sbatch template and stage cpu divisor to facilitate array set up and command distribution
    with open(PUtils.sbatch_templates[args.stage]) as template_f:
        template_sbatch = template_f.readlines()

    # Make sbatch file from template, array details, and command distribution script
    filename = os.path.join(args.directory, '%s_%s-%d.sh' % (args.stage, PUtils.sbatch, ran_num))
    output = os.path.join(args.directory, 'output')
    if not os.path.exists(output):
        os.mkdir(output)

    command_divisor = CUtils.process_scale[args.stage]
    with open(filename, 'w') as new_f:
        for template_line in template_sbatch:
            new_f.write(template_line)
        out = 'output=%s/%s' % (output, '%A_%a.out')
        new_f.write(PUtils.sb_flag + out + '\n')
        array = 'array=1-%d%%%d' % (int(len(_commands) / command_divisor + 0.5), args.max_jobs)
        new_f.write(PUtils.sb_flag + array + '\n')
        new_f.write('\npython %s --stage %s --success_file %s --failure_file %s --command_file %s %s\n' %
                    (PUtils.cmd_dist, args.stage, args.success_file, args.failure_file, args.file,
                     (script_present or '')))

    logger.info('To distribute commands enter the following:\ncd %s\nsbatch %s'
                % (args.directory, os.path.basename(filename)))


def mp_starmap(function, process_args, threads=1, context='spawn'):
    """Maps iterable to a function using multiprocessing Pool

    Args:
        function (function): Which function should be executed
        process_args (list(tuple)): Arguments to be unpacked in the defined function, order specific
    Keyword Args:
        threads=1 (int): How many workers/threads should be spawned to handle function(arguments)?
        context='spawn' (str): One of 'spawn', 'fork', or 'forkserver'
    Returns:
        results (list): The results produced from the function and process_args
    """
    with mp.get_context(context).Pool(processes=threads, maxtasksperchild=100) as p:
        results = p.starmap(function, process_args)  # , chunksize=1
    p.join()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=os.path.basename(__file__)
                                     + '\nGather commands set up by %s and distribute to computational nodes for '
                                     'Rosetta processing.' % PUtils.program_name)
    parser.add_argument('-s', '--stage', choices=tuple(CUtils.process_scale.keys()),
                        help='The stage of design to be distributed. Each stage has optimal computing requirements to'
                             ' maximally utilize computers . One of %s' % ', '.join(list(CUtils.process_scale.keys())))
    # TODO combine with command file as 1 arg?
    parser.add_argument('-C', '--command_present', action='store_true',
                        help='Whether command file has commands already')
    parser.add_argument('-c', '--command_file', help='File with command(s) to be distributed. Required', required=True)  # TODO -f
    parser.add_argument('-y', '--success_file', help='The disk location of output file containing successful commands')
    parser.add_argument('-n', '--failure_file', help='The disk location of output file containing failed commands')
    args = parser.parse_args()

    # Grab all possible poses
    with open(args.command_file, 'r') as cmd_f:
        all_commands = cmd_f.readlines()

    # Select exact poses to be handled according to array_ID and design stage
    array_task = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    cmd_slice = (array_task - 1) * CUtils.process_scale[args.stage]  # adjust from SLURM one index
    # cmd_slice = (array_task - SDUtils.index_offset) * CUtils.process_scale[args.stage]  # adjust from SLURM one index
    if cmd_slice + CUtils.process_scale[args.stage] > len(all_commands):  # check to ensure list index isn't missing
        final_cmd_slice = None
    else:
        final_cmd_slice = cmd_slice + CUtils.process_scale[args.stage]
    specific_commands = list(map(str.strip, all_commands[cmd_slice:final_cmd_slice]))

    # Prepare Commands
    # command_name = args.stage + '.sh'
    # python2.7 compatibility
    def path_maker(path_name):
        return os.path.join(path_name, '%s.sh' % args.stage)

    if args.command_present:
        command_paths = specific_commands
        # log_files = [os.path.join(os.path.dirname(log_dir), '%s.log' % os.path.splitext(os.path.basename(log_dir))[0] for log_dir in command_paths)]
    else:
        command_paths = list(map(path_maker, specific_commands))

    log_files = [os.path.join(os.path.dirname(log_dir), '%s.log' % os.path.splitext(os.path.basename(log_dir))[0])
                 for log_dir in command_paths]
    #
    # if args.stage == 'nanohedra':
    #     log_dirs = [SDUtils.set_up_pseudo_design_dir(cmd, os.getcwd(), os.getcwd()) for cmd in specific_commands]
    #     # des_dirs = [SDUtils.set_up_pseudo_design_dir(cmd, os.getcwd(), os.getcwd()) for cmd in specific_commands]
    # else:
    #
    #     # des_dirs = SDUtils.set_up_pseudo_design_dir(poses)
    #     # des_dirs = SDUtils.set_up_directory_objects(poses)
    # log_files = [os.path.join(log_dir, os.path.basename(log_dir) + '.log') for log_dir in log_dirs]
    commands = zip(command_paths, log_files)

    # Ensure all log files exist
    for log_file in log_files:
        create_file(log_file)
    create_file(args.success_file)
    create_file(args.failure_file)

    # Run commands in parallel
    # monitor = GracefulKiller()  # TODO solution to SIGTERM. TEST shows this doesn't appear to be possible...
    signal.signal(signal.SIGINT, exit_gracefully)
    # signal.signal(signal.SIGKILL, exit_gracefully)  # Doesn't work, not possible
    signal.signal(signal.SIGTERM, exit_gracefully)
    # while not monitor.kill_now:

    # # python 2.7 compatibility NO MP here
    # results = []
    # for command, log_file in commands:
    #     results.append(run(command, log_file))

    # python 3.7 compatible
    if len(command_paths) > 1:  # set by CUtils.process_scale
        results = mp_starmap(run, commands, threads=len(command_paths))
        # results = SDUtils.mp_starmap(run, commands, threads=len(command_paths))
    else:
        results = [run(command, log_file) for command, log_file in commands]

    # Write out successful and failed commands TODO ensure write is only possible one at a time
    with open(args.success_file, 'a') as f:
        for i, result in enumerate(results):
            if result:
                f.write('%s\n' % command_paths[i])

    with open(args.failure_file, 'a') as f:
        for i, result in enumerate(results):
            if not result:
                f.write('%s\n' % command_paths[i])

    # # Append SLURM output to log_file(s)
    # job_id = int(os.environ.get('SLURM_JOB_ID'))
    # for i, task_id in enumerate(range(cmd_slice, final_cmd_slice)):
    #     file = '%s_%s.out' % (job_id, array_task)
    #     # file = '%s_%s.out' % (job_id, task_id)
    #     run(file, log_files[i], program='cat')
    #     run(file, '/dev/null', program='rm')
