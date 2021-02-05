"""
Module for Distribution of Rosetta commands found for individual poses to SLURM/PBS computational cluster
Finds commands within received directories (poses)
"""

import argparse
import os
import signal
import subprocess
from random import random
from itertools import repeat

import CmdUtils as CUtils
import PathUtils as PUtils
from SymDesignUtils import index_offset, start_log, DesignError, collect_directories, mp_starmap, unpickle, \
    pickle_object

start_log(__name__)


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        with open(args.failure_file, 'a') as f:
            for i, pose in enumerate(command_paths):
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
        for i, pose in enumerate(command_paths):
            f.write('%s\n' % pose)

    # Append SLURM output to log_file(s)
    job_id = int(os.environ.get('SLURM_JOB_ID'))
    file = 'output/%s_%s.out' % (job_id, array_task)
    for i, task_id in enumerate(range(cmd_slice, final_cmd_slice)):
        # file = '%s_%s.out' % (job_id, task_id)
        run(file, log_files[i], program='cat')
        # run(file, '/dev/null', program='rm')


def create_file(file):
    """If file doesn't exist, create a blank one"""
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


def run(cmd, log_file, srun=None, program='bash'):  # , log_file=None):
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
    cluster_prefix = srun if srun else []
    with open(log_file, 'a') as log_f:
        p = subprocess.Popen(cluster_prefix + [program, cmd], stdout=log_f, stderr=log_f)
        p.wait()

    if p.returncode == 0:
        return True
    else:
        return False


def distribute(logger, stage=None, directory=None, file=None, success_file=None, failure_file=None, max_jobs=80):
    # Todo back out logger
    if not stage:
        raise DesignError('No --stage specified. Required!!!')
    if file:  # or directory: Todo
        # here using collect directories get the commands from the provided file
        _commands, location = collect_directories(directory, file=file)
    else:
        raise DesignError('Error: You must pass a file containing a list of commands to process. This is '
                                  'typically output to a \'[stage].cmds\' file. Ensure that this file exists and '
                                  'resubmit with -f \'[stage].cmds\'\n')
        #             ', replacing stage with the desired stage.')

    # Automatically detect if the commands file has executable scripts or errors
    script_present = None
    for idx, _command in enumerate(_commands):
        if not os.path.exists(_command):  # check for any missing commands and report
            raise DesignError('%s is malformed at line %d! The command at location (%s) doesn\'t exist!\n'
                                      % (file, idx + 1, _command))
        if not _command.endswith('.sh'):  # if the command string is not a shell script (end with .sh)
            if idx != 0 and script_present:  # There was a change from script files to non-script files
                raise DesignError('%s is malformed at line %d! All commands must either have a file extension '
                                          'or not. Cannot mix!\n' % (file, idx + 1))
            # break
        else:  # the command string is a shell script
            if idx != 0 and not script_present:  # There was a change from non-script files to script files
                raise DesignError('%s is malformed at line %d! All commands must either have a file extension '
                                          'or not. Cannot mix!\n' % (file, idx + 1))
            script_present = '-c'

    # Create success and failures files
    ran_num = int(100 * random())
    if not success_file:
        success_file = os.path.join(directory, '%s_sbatch-%d_success.log' % (stage, ran_num))
    if not failure_file:
        failure_file = os.path.join(directory, '%s_sbatch-%d_failures.log' % (stage, ran_num))
    logger.info('\nSuccessful poses will be listed in \'%s\'\nFailed poses will be listed in \'%s\''
                % (success_file, failure_file))

    # Grab sbatch template and stage cpu divisor to facilitate array set up and command distribution
    with open(PUtils.sbatch_templates[stage]) as template_f:
        template_sbatch = template_f.readlines()

    # Make sbatch file from template, array details, and command distribution script
    filename = os.path.join(directory, '%s_%s-%d.sh' % (stage, PUtils.sbatch, ran_num))
    output = os.path.join(directory, 'output')
    if not os.path.exists(output):
        os.mkdir(output)

    command_divisor = CUtils.process_scale[stage]
    with open(filename, 'w') as new_f:
        for template_line in template_sbatch:
            new_f.write(template_line)
        out = 'output=%s/%s' % (output, '%A_%a.out')
        new_f.write(PUtils.sb_flag + out + '\n')
        array = 'array=1-%d%%%d' % (int(len(_commands) / command_divisor + 0.5), max_jobs)
        new_f.write(PUtils.sb_flag + array + '\n')
        new_f.write('\npython %s --stage %s --success_file %s --failure_file %s --command_file %s %s\n' %
                    (PUtils.cmd_dist, stage, success_file, failure_file, file,
                     (script_present or '')))

    logger.info('To distribute commands, ensure the sbatch script (%s) is correct, then enter the following:'
                '\nsbatch %s' % (filename, os.path.basename(filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nGather commands set up by %s and distribute to computational '
                                                 'nodes for Rosetta processing.'
                                                 % (os.path.basename(__file__), PUtils.program_name))
    parser.add_argument('--stage', choices=tuple(CUtils.process_scale.keys()),
                        help='The stage of design to be distributed. Each stage has optimal computing requirements to'
                             ' maximally utilize computers . One of %s' % ', '.join(list(CUtils.process_scale.keys())))
    subparsers = parser.add_subparsers(title='SubModules', dest='module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter: \n%s\n\nAny'
                                            'SubModule help can be accessed in this way' % PUtils.submodule_help)
    # ---------------------------------------------------
    parser_distirbute = subparsers.add_parser('distribute', help='Access the %s guide! Start here if your a first time user'
                                                       % PUtils.program_name)
    parser_distirbute.add_argument('-c', '--command_present', action='store_true',
                                   help='Whether command file has commands already')  # TODO combine with command file as 1 arg
    parser_distirbute.add_argument('-f', '--command_file',
                                   help='File with command(s) to be distributed. Required', required=True)
    parser_distirbute.add_argument('-y', '--success_file',
                                   help='The disk location of output file containing successful commands')
    parser_distirbute.add_argument('-n', '--failure_file',
                                   help='The disk location of output file containing failed commands')
    parser_distirbute.add_argument('-S', '--srun', action='store_true',
                                   help='Utilize srun to allocate resources, launch the job and communicate with SLURM')
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Check the status of the command')
    parser_status.add_argument('-c', '--check', action='store_true', help='Check the status of the command')
    parser_status.add_argument('-i', '--info', type=str, help='The location of the state file')
    parser_status.add_argument('-s', '--set', action='store_true', help='Set the status as True')

    args = parser.parse_args()

    if args.module == 'status':
        info = unpickle(args.info)
        if args.check:
            if info['status'][args.stage]:  # if the status of the stage is True
                exit(1)
        elif args.set:
            info['status'][args.stage] = True

            pickle_object(info, name=args.info, out_path='')
            exit(0)
    elif args.module == 'distribute':
        # Grab all possible poses
        with open(args.command_file, 'r') as cmd_f:
            all_commands = cmd_f.readlines()

        # Select exact poses to be handled according to array_ID and design stage
        array_task = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
        cmd_slice = (array_task - index_offset) * CUtils.process_scale[args.stage]  # adjust from SLURM one index
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
        else:  # Todo, depreciate this mechanism
            command_paths = list(map(path_maker, specific_commands))

        log_files = ['%s.log' % os.path.splitext(log_dir)[0] for log_dir in command_paths]
        iteration = 0
        complete = False
        while not complete:
            # allocation = ['srun', '-c', 1, '-p', 'long', '--mem-per-cpu', CUtils.memory_scale[args.stage]]
            allocation = None
            zipped_commands = zip(command_paths, log_files, repeat(allocation))

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
                results = mp_starmap(run, zipped_commands, threads=len(command_paths))  # Todo change the command paths
            else:
                results = [run(*command_pair) for command_pair in zipped_commands]
            iteration += 1

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
