"""
Module for Distribution of Rosetta commands found for individual poses to SLURM/PBS computational cluster
Finds commands within received directories (poses)
"""
import argparse
import os
import signal
import subprocess
from itertools import repeat

from PathUtils import stage, sbatch_template_dir, nano, rosetta, rosetta_extras, dalphaball, submodule_help, cmd_dist, \
    program_name, interface_design
from SymDesignUtils import start_log, DesignError, collect_designs, mp_starmap, unpickle, pickle_object, handle_errors

# Globals
logger = start_log(name=__name__)
index_offset = 1
min_cores_per_job = 1  # currently one for the MPI node, and 5 workers
mpi = 4
num_thread_per_process = 2
hhblits_threads = 1
hhblits_memory_threshold = 10000000000
reference_average_residue_weight = 3  # for REF2015
sbatch = 'sbatch'
sb_flag = '#SBATCH --'
run_cmds = {'default': '',
            'python': '',
            'cxx11thread': '',
            'mpi': ['mpiexec', '--oversubscribe', '-np'],
            'cxx11threadmpi': ['mpiexec', '--oversubscribe', '-np', str(int(min_cores_per_job / num_thread_per_process))]}  # TODO Optimize
extras_flags = {'default': [],
                'python': [],
                'cxx11thread': ['-multithreading:total_threads ' + str(num_thread_per_process),
                                '-multithreading:interaction_graph_threads ' + str(num_thread_per_process)],
                'mpi': [],
                'cxx11threadmpi': ['-multithreading:total_threads ' + str(num_thread_per_process)]}
# Todo modify .linuxgccrelease depending on os
script_cmd = [os.path.join(rosetta, 'source/bin/rosetta_scripts.%s.linuxgccrelease' % rosetta_extras),
              '-database', os.path.join(rosetta, 'database')]
rosetta_flags = extras_flags[rosetta_extras] + \
    ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res', '-ignore_zero_occupancy false',
     # '-overwrite',
     '-linmem_ig 10', '-out:file:scorefile_format json', '-output_only_asymmetric_unit true', '-no_chainend_ter true',
     '-write_seqres_records true', '-output_pose_energies_table false', '-output_pose_cache_data false',
     '-holes:dalphaball %s' % dalphaball if os.path.exists(dalphaball) else '',
     '-use_occurrence_data',  # Todo integrate into xml with Rosetta Source update
     '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer VirtualBB ShoveBB VirtualNTerm '
     'VirtualDNAPhosphate CTermConnect sc_orbitals pro_hydroxylated_case1 N_acetylated C_methylamidated cys_acetylated'
     'pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_diiodinated tyr_sulfated'
     'lys_dimethylated lys_monomethylated lys_trimethylated lys_acetylated glu_carboxylated MethylatedProteinCterm',
     '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol protocols.jd2.JobDistributor']

# Those jobs having a scale of 2 utilize two threads. Therefore two commands are selected from a supplied commands list
# and are launched inside a python environment once the SLURM controller starts a SBATCH array job
process_scale = {stage[1]: 2, interface_design: 2, stage[2]: 2, stage[3]: 2, stage[5]: 2, nano: 2,
                 stage[6]: 1, stage[7]: 1, stage[8]: 1, stage[9]: 1, stage[10]: 1,
                 stage[11]: 1, stage[12]: 2, stage[13]: 2,
                 'metrics_bound': 2, 'interface_metrics': 2
                 }
# Cluster Dependencies and Multiprocessing
sbatch_templates = {stage[1]: os.path.join(sbatch_template_dir, stage[1]),
                    interface_design: os.path.join(sbatch_template_dir, stage[2]),
                    stage[2]: os.path.join(sbatch_template_dir, stage[2]),
                    stage[12]: os.path.join(sbatch_template_dir, stage[2]),
                    stage[3]: os.path.join(sbatch_template_dir, stage[2]),
                    stage[4]: os.path.join(sbatch_template_dir, stage[1]),
                    stage[5]: os.path.join(sbatch_template_dir, stage[1]),
                    nano: os.path.join(sbatch_template_dir, nano),
                    stage[6]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[7]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[8]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[9]: os.path.join(sbatch_template_dir, stage[6]),
                    'metrics_bound': os.path.join(sbatch_template_dir, stage[2]),
                    'interface_metrics': os.path.join(sbatch_template_dir, stage[2])
                    }


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
        for pose in command_paths:
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


def run(cmd, log_file_name, program=None, srun=None):
    """Executes specified command and appends command results to log file

    Args:
        cmd (str): The name of a command file which should be executed by the system
        log_file_name (str): Location on disk of log file
    Keyword Args:
        program=None (str): The interpreter for said command
    Returns:
        (bool): Whether or not command executed successfully
    """
    # des_dir = SDUtils.DesignDirectory(os.path.dirname(cmd))
    # if not log_file:
    #     log_file = os.path.join(des_dir.path, os.path.basename(des_dir.path) + '.log')
    cluster_prefix = srun if srun else []
    program = [program] if program else []
    command = [cmd] if isinstance(cmd, str) else cmd
    with open(log_file_name, 'a') as log_f:
        p = subprocess.Popen(cluster_prefix + program + command, stdout=log_f, stderr=log_f)
        p.wait()

    if p.returncode == 0:
        return True
    else:
        return False


def distribute(file=None, out_path=os.getcwd(), scale=None, success_file=None, failure_file=None, max_jobs=80,
               number_of_commands=None, mpi=None, **kwargs):
    """Take a file of commands formatted for execution in the SLURM environment and process into a sbatch script

    Keyword Args:
        file=None (str): The location of the file which contains your commands to distribute through an sbatch array
        out_path=os.getcwd() (str): Where to write out the sbatch script
        scale=None (str): The stage of design to distribute. Works with CommandUtils and PathUtils to allocate jobs
        success_file=None (str): What file to write the successful jobs to for job organization
        failure_file=None (str): What file to write the failed jobs to for job organization
        max_jobs=80 (int): The size of the job array limiter. This caps the number of commands executed at once
        number_of_commands=None (int): The size of the job array
        mpi=None (int): The number of processes to run concurrently with MPI
    Returns:
        (str): The name of the sbatch script that was written
    """
    if not scale:
        # elif process_scale: Todo in order to make stage unnecessary, would need to provide scale and template
        #                      Could add a hyperthreading=True parameter to remove process scale
        #     command_divisor = process_scale
        # else:
        raise DesignError('No --stage specified. Required!!!')

    if number_of_commands:
        _commands = [0 for _ in range(number_of_commands)]
        script_present = '-c'
    elif file:  # use collect directories get the commands from the provided file and verify content
        _commands, location = collect_designs(files=[file], directory=out_path)
        # Automatically detect if the commands file has executable scripts or errors
        script_present = None
        for idx, _command in enumerate(_commands):
            if not _command.endswith('.sh'):  # if the command string is not a shell script (doesn't end with .sh)
                if idx != 0 and script_present:  # There was a change from script files to non-script files
                    raise DesignError('%s is malformed at line %d! All commands must either have a file extension '
                                      'or not. Cannot mix!\n' % (file, idx + 1))
                # break
            else:  # the command string is a shell script
                if not os.path.exists(_command):  # check for any missing commands and report
                    raise DesignError('%s is malformed at line %d! The command at location (%s) doesn\'t exist!\n'
                                      % (file, idx + 1, _command))
                if idx != 0 and not script_present:  # There was a change from non-script files to script files
                    raise DesignError('%s is malformed at line %d! All commands must either have a file extension '
                                      'or not. Cannot mix!\n' % (file, idx + 1))
                script_present = '-c'
    else:
        raise DesignError('You must pass number_of_commands or file which contains a list of commands to process')
        # 'A file is typically output as a \'STAGE.cmds\' file. Ensure that this file exists and resubmit with
        # -f \'STAGE.cmds\'\n')

    # Create success and failures files
    name = os.path.basename(os.path.splitext(file)[0])
    if not success_file:
        success_file = os.path.join(out_path, '%s_%s_success.log' % (name, sbatch))
    if not failure_file:
        failure_file = os.path.join(out_path, '%s_%s_failures.log' % (name, sbatch))
    output = os.path.join(out_path, 'sbatch_output')
    if not os.path.exists(output):
        os.mkdir(output)

    # Make sbatch file from template, array details, and command distribution script
    filename = os.path.join(out_path, '%s_%s.sh' % (name, sbatch))
    with open(filename, 'w') as new_f:
        # Todo set up sbatch accordingly. Include a multiplier for the number of CPU's. Actually, might be passed
        # if mpi:
        #     do_mpi_stuff = True
        # grab and write sbatch template
        with open(sbatch_templates[scale]) as template_f:
            new_f.write(''.join(template_f.readlines()))
        out = 'output=%s/%s' % (output, '%A_%a.out')
        new_f.write('%s%s\n' % (sb_flag, out))
        array = 'array=1-%d%%%d' % (int(len(_commands) / process_scale[scale] + 0.5), max_jobs)
        new_f.write('%s%s\n' % (sb_flag, array))
        new_f.write('\npython %s --stage %s distribute --success_file %s --failure_file %s --command_file %s %s\n' %
                    (cmd_dist, scale, success_file, failure_file, file, (script_present or '')))

    return filename


@handle_errors(errors=(FileNotFoundError,))
def update_status(serialized_info, stage, mode='check'):
    """Update the serialized info for a designs commands such as checking or removing status, and marking completed"""
    info = unpickle(serialized_info)
    if mode == 'check':
        if info['status'][stage]:  # if the status of the stage is True
            exit(1)
    elif mode == 'set':
        info['status'][stage] = True
        pickle_object(info, name=serialized_info, out_path='')
        # exit()
    elif mode == 'remove':
        info['status'][stage] = False
        pickle_object(info, name=serialized_info, out_path='')
        # exit()
    else:
        exit(127)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nGather commands set up by %s and distribute to computational '
                                                 'nodes for Rosetta processing.'
                                                 % (os.path.basename(__file__), program_name))
    parser.add_argument('--stage', choices=tuple(process_scale.keys()),
                        help='The stage of design to be distributed. Each stage has optimal computing requirements to'
                             ' maximally utilize computers . One of %s' % ', '.join(list(process_scale.keys())))
    subparsers = parser.add_subparsers(title='SubModules', dest='module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter: \n%s\n\nAny'
                                            'SubModule help can be accessed in this way' % submodule_help)
    # ---------------------------------------------------
    parser_distirbute = subparsers.add_parser('distribute', help='Submit a job to SLURM for processing')
    # TODO combine with command file as 1 arg
    parser_distirbute.add_argument('-c', '--command_present', action='store_true',
                                   help='Whether command file has commands already')
    parser_distirbute.add_argument('-f', '--command_file',
                                   help='File with command(s) to be distributed. Required', required=True)
    parser_distirbute.add_argument('-l', '--log_file', type=str, default=None,
                                   help='The name of the log file to append command stdout and stderr')
    parser_distirbute.add_argument('-n', '--failure_file',
                                   help='The disk location of output file containing failed commands')
    parser_distirbute.add_argument('-S', '--srun', action='store_true',
                                   help='Utilize srun to allocate resources, launch the job and communicate with SLURM')
    parser_distirbute.add_argument('-y', '--success_file',
                                   help='The disk location of output file containing successful commands')
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Check the status of the command')
    parser_status.add_argument('-c', '--check', action='store_true', help='Check the status of the command')
    parser_status.add_argument('-i', '--info', type=str, help='The location of the state file')
    parser_status.add_argument('-s', '--set', action='store_true', help='Set the status as True')
    parser_status.add_argument('-r', '--remove', action='store_true', help='Set the status as False')

    args = parser.parse_args()

    if args.module == 'status':
        mode = 'check' if args.check else 'set' if args.set else 'remove'
        update_status(args.info, args.stage, mode=mode)
    elif args.module == 'distribute':
        # Grab all possible poses
        with open(args.command_file, 'r') as cmd_f:
            all_commands = cmd_f.readlines()

        # Select exact poses to be handled according to array_ID and design stage
        array_task = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
        # adjust from SLURM one index and figure out how many commands to grab from command pool
        cmd_slice = (array_task - index_offset) * process_scale[args.stage]
        if cmd_slice + process_scale[args.stage] > len(all_commands):  # check to ensure list index isn't missing
            final_cmd_slice = None
            if cmd_slice > len(all_commands):
                exit()
        else:
            final_cmd_slice = cmd_slice + process_scale[args.stage]
        specific_commands = list(map(str.strip, all_commands[cmd_slice:final_cmd_slice]))

        # Prepare Commands
        if len(specific_commands[0].split()) > 1:
            # the command provided probably has an attached program type. Set to None, then split to a list
            program = None
            specific_commands = [cmd.split() for cmd in specific_commands]
        else:
            program = 'bash'

        # command_name = args.stage + '.sh'
        # python2.7 compatibility
        def path_maker(path_name):  # Todo depreciate
            return os.path.join(path_name, '%s.sh' % args.stage)

        if args.command_present:
            command_paths = specific_commands
        else:  # Todo, depreciate this mechanism
            command_paths = list(map(path_maker, specific_commands))

        if args.log_file:
            log_files = [args.log_file for cmd in command_paths]
        else:
            log_files = ['%s.log' % os.path.splitext(design_directory)[0] for design_directory in command_paths]

        iteration = 0
        complete = False
        # while not complete:
        #     allocation = ['srun', '-c', 1, '-p', 'long', '--mem-per-cpu', CUtils.memory_scale[args.stage]]
        #     allocation = None
        #     zipped_commands = zip(command_paths, log_files, repeat(allocation))
        zipped_commands = zip(command_paths, log_files, repeat(program))

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

        number_of_commands = len(specific_commands)  # different from process scale as this could reflect edge cases
        if number_of_commands > 1:  # set by CUtils.process_scale
            results = mp_starmap(run, zipped_commands, threads=number_of_commands)
        else:
            results = [run(*command) for command in zipped_commands]
        #    iteration += 1

        # Write out successful and failed commands
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
        #     file = os.path.join(sbatch_output, '%s_%s.out' % (job_id, array_task))  # Todo set sbatch_output
        #     # file = '%s_%s.out' % (job_id, task_id)
        #     run(file, log_files[i], program='cat')
        #     # run(file, '/dev/null', program='rm')
