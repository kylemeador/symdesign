"""
Module for Distribution of Rosetta commands found for individual poses to SLURM/PBS computational cluster
Finds commands within received directories (poses)
"""

import argparse
import os
import signal
import subprocess

import CmdUtils as CUtils
import PathUtils as PUtils
import SymDesignUtils as SDUtils


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
    cmd_slice = (array_task - SDUtils.index_offset) * CUtils.process_scale[args.stage]  # adjust from SLURM one index
    if cmd_slice + CUtils.process_scale[args.stage] > len(all_commands):  # check to ensure list index isn't missing
        final_cmd_slice = None
    else:
        final_cmd_slice = cmd_slice + CUtils.process_scale[args.stage]
    poses = list(map(str.strip, all_commands[cmd_slice:final_cmd_slice]))

    # Prepare Commands
    # command_name = args.stage + '.sh'
    # python2.7 compatibility
    def path_maker(path_name):
        return os.path.join(path_name, '%s.sh' % args.stage)

    if not args.command_present:
        commands_of_interest = list(map(path_maker, poses))
    else:
        commands_of_interest = poses

    if args.stage == 'nanohedra':
        des_dirs = [SDUtils.set_up_pseudo_design_dir(pose, os.getcwd(), os.getcwd()) for pose in poses]
    else:
        des_dirs = SDUtils.set_up_directory_objects(poses)
    log_files = [os.path.join(des_dir.path, os.path.basename(des_dir.path) + '.log') for des_dir in des_dirs]
    commands = zip(commands_of_interest, log_files)

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

    # python 2.7 compatibility NO MP here
    results = []
    for command, log_file in commands:
        results.append(run(command, log_file))

    # python 3.7 compatible
    # results = SDUtils.mp_starmap(run, commands, threads=len(commands_of_interest))  # TODO reinstate
    #
    #     # Write out successful and failed commands TODO ensure write is only possible one at a time
    #     with open(args.success_file, 'a') as f:
    #         for i, pose in enumerate(poses):
    #             if results[i]:
    #                 f.write('%s\n' % pose)
    #                 # f.write('%d %s\n' % (cmd_slice, pose.rstrip(command_name)))
    #             cmd_slice += i
    #     with open(args.failure_file, 'a') as f:
    #         for i, pose in enumerate(poses):
    #             if not results[i]:
    #                 f.write('%s\n' % pose)
    #     break

    # if monitor.kill_now:
    #
    # else:

    # Write out successful and failed commands TODO ensure write is only possible one at a time
    with open(args.success_file, 'a') as f:
        for i, pose in enumerate(poses):
            if results[i]:
                f.write('%s\n' % pose)

    with open(args.failure_file, 'a') as f:
        for i, pose in enumerate(poses):
            if not results[i]:
                f.write('%s\n' % pose)

    # # Append SLURM output to log_file(s)
    # job_id = int(os.environ.get('SLURM_JOB_ID'))
    # for i, task_id in enumerate(range(cmd_slice, final_cmd_slice)):
    #     file = '%s_%s.out' % (job_id, array_task)
    #     # file = '%s_%s.out' % (job_id, task_id)
    #     run(file, log_files[i], program='cat')
    #     run(file, '/dev/null', program='rm')
