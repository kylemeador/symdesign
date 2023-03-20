from __future__ import annotations

import argparse
import os
import signal
import subprocess
from itertools import repeat

from symdesign.resources.distribute import create_file, default_shell, process_scale, run
from symdesign.utils import calculate_mp_cores, mp_starmap, path as putils

index_offset = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}\nGather commands set up by '
                                                 f'{putils.program_name} and distribute to computational nodes for '
                                                 f'processing')
    parser.add_argument('--stage', choices=tuple(process_scale.keys()),
                        help='The stage of design to be distributed. Each stage has optimal computing requirements to '
                             f'maximally utilize computers. One of {", ".join(list(process_scale.keys()))}')
    subparsers = parser.add_subparsers(title='SubModules', dest='module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter: \n'
                                            f"{putils.submodule_help}\n\nAny module's help can be accessed in this way")
    # ---------------------------------------------------
    # parser_distribute = subparsers.add_parser('distribute', help='Submit a job to SLURM for processing')
    parser.add_argument('-f', '--command-file', '--command_file',
                        help='File with command(s) to be distributed. Required', required=True)
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='The number of jobs to be executed at once')
    parser.add_argument('-l', '--log-file', '--log_file', type=str, default=None,
                        help='The name of the log file to append command stdout and stderr')
    parser.add_argument('-n', '--failure-file', '--failure_file',
                        help='The disk location of output file containing failed commands')
    parser.add_argument('-S', '--srun', action='store_true',
                        help='Utilize srun to allocate resources, launch the job and communicate with SLURM')
    parser.add_argument('-y', '--success-file', '--success_file',
                        help='The disk location of output file containing successful commands')
    # # ---------------------------------------------------
    # parser_status = subparsers.add_parser('status', help='Check the status of the command')
    # parser_status.add_argument('-c', '--check', action='store_true', help='Check the status of the command')
    # parser_status.add_argument('-i', '--info', type=str, help='The location of the state file')
    # parser_status.add_argument('-s', '--set', action='store_true', help='Set the status as True')
    # parser_status.add_argument('-r', '--remove', action='store_true', help='Set the status as False')

    args, additional_args = parser.parse_known_args()

    # if args.module == 'status':
    #     mode = 'check' if args.check else 'set' if args.set else 'remove'
    #     update_status(args.info, args.stage, mode=mode)
    # elif args.module == 'distribute':
    # Grab all possible poses
    with open(args.command_file, 'r') as cmd_f:
        all_commands = cmd_f.readlines()

    # Select exact poses to be handled according to array_ID and design stage
    # Todo change to args.number_of_processes instead of args.stage
    number_of_processes = process_scale.get(args.stage, 1)
    array_number = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_number:
        array_task_number = int(array_number)
        # Adjust from SLURM one index and figure out how many commands to grab from command pool
        cmd_start_slice = (array_task_number - index_offset) * number_of_processes
        if cmd_start_slice > len(all_commands):
            exit()
        cmd_end_slice = cmd_start_slice + number_of_processes
    else:  # Not in SLURM, use multiprocessing
        cmd_start_slice = cmd_end_slice = None
    # Set the type for below if the specific command can be split further
    specific_commands: list[str] | list[list[str]] = \
        list(map(str.strip, all_commands[cmd_start_slice:cmd_end_slice]))

    # Prepare Commands
    # Check if the commands have a program followed by a space
    if len(specific_commands[0].split()) > 1:
        program = None
        specific_commands = [cmd.split() for cmd in specific_commands]
    else:  # A single file is present which is probably a shell script. Use bash
        program = default_shell

    if args.log_file:
        print(f'Writing job log to "{args.log_file}"')
        log_files = [args.log_file for _ in specific_commands]
    elif program == 'bash':
        # v this overlaps with len(specific_commands[0].split()) > 1 as only shell scripts really satisfy this
        log_files = [f'{os.path.splitext(shell_path)[0]}.log' for shell_path in specific_commands]
        for idx, log_file in enumerate(log_files):
            print(f'Writing job {idx} log to "{args.log_file}"')
    else:
        log_files = [None for _ in specific_commands]

    # iteration = 0
    # complete = False
    # Todo implementing an srun prefix to any command allows for multiple job steps to be controlled. This is useful
    #  when a prior step gets hung up and needs to be cancelled, but the remaining job steps should be executed
    #  downside to all this is that the allocation is done by inherently neglecting the hyperthreading. The srun
    #  would respect the one cpu, one task logic.
    # while not complete:
    #     allocation = ['srun', '-c', 1, '-p', 'long', '--mem-per-cpu', CUtils.memory_scale[args.stage]]
    #     allocation = None
    #     zipped_commands = zip(specific_commands, log_files, repeat(allocation))
    # print('Running command:\n', subprocess.list2cmdline(specific_commands[0]))
    zipped_commands = zip(specific_commands, log_files, repeat(program))

    # Ensure all log and reporting files exist
    for log_file in log_files:
        create_file(log_file)
    create_file(args.success_file)
    create_file(args.failure_file)

    def exit_gracefully(signum, frame):
        with open(args.failure_file, 'a') as f:
            # Todo only report those that are still running
            f.write('%s\n' % '\n'.join(specific_commands))

        # Handle SLURM output
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            file = f'output{os.sep}{job_id}_{array_task_number}.out'
            # for idx, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
            for log_file in log_files:
                # Append SLURM output to log_file(s)
                run(file, log_file, program='cat')
                # # Remove SLURM output
                # run(file, '/dev/null', program='rm')
        else:
            return

    # Run commands in parallel
    # monitor = GracefulKiller()  # TODO solution to SIGTERM. TEST shows this doesn't appear to be possible...
    signal.signal(signal.SIGINT, exit_gracefully)
    # signal.signal(signal.SIGKILL, exit_gracefully)  # Doesn't work, not possible
    signal.signal(signal.SIGTERM, exit_gracefully)
    # while not monitor.kill_now:

    # number_of_commands is different from process scale and could reflect edge cases
    number_of_commands = len(specific_commands)
    if number_of_commands > 1:
        # The args.jobs was set by the process_scale dictionary
        results = mp_starmap(run, zipped_commands,
                             processes=calculate_mp_cores(cores=number_of_commands, jobs=args.jobs))
    else:
        results = [run(*command) for command in zipped_commands]
    #    iteration += 1

    # Write out successful and failed commands
    with open(args.success_file, 'a') as f_success, open(args.failure_file, 'a') as f_failure:
        for result, specific_command in zip(results, specific_commands):
            if program:
                command_out = specific_command
            else:
                command_out = subprocess.list2cmdline(specific_command)

            if result:
                f_success.write(f'{command_out}\n')
            else:  # if not result:
                f_failure.write(f'{command_out}\n')

    # # Append SLURM output to log_file(s)
    # job_id = int(os.environ.get('SLURM_JOB_ID'))
    # for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
    #     file = os.path.join(sbatch_output, '%s_%s.out' % (job_id, array_task_number))  # Todo set sbatch_output
    #     # file = '%s_%s.out' % (job_id, task_id)
    #     run(file, log_files[i], program='cat')
    #     # run(file, '/dev/null', program='rm')
