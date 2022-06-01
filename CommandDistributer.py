"""
Module for Distribution of Rosetta commands found for individual poses to SLURM/PBS computational cluster
Finds commands within received directories (poses)
"""
import argparse
import os
import signal
import subprocess
from itertools import repeat, chain
from typing import Union, List

from PathUtils import stage, sbatch_template_dir, nano, rosetta_main, rosetta_extras, dalphaball, submodule_help, \
    cmd_dist, program_name, interface_design, interface_metrics, optimize_designs, refine, rosetta_scripts, \
    sym_weights, solvent_weights_sym, solvent_weights, scout, consensus
from SymDesignUtils import start_log, DesignError, collect_designs, mp_starmap, unpickle, pickle_object, handle_errors, \
    calculate_mp_cores

# Globals
logger = start_log(name=__name__)
index_offset = 1
min_cores_per_job = 1  # currently one for the MPI node, and 5 workers
mpi = 4
num_thread_per_process = 2
hhblits_threads = 2
hhblits_memory_threshold = 30000000000  # 30GB
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
script_cmd = [os.path.join(rosetta_main, 'source', 'bin', 'rosetta_scripts.%s.linuxgccrelease' % rosetta_extras),
              '-database', os.path.join(rosetta_main, 'database')]
rosetta_flags = extras_flags[rosetta_extras] + \
    ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res', '-ignore_zero_occupancy false',
     # '-overwrite',
     '-linmem_ig 10', '-out:file:scorefile_format json', '-output_only_asymmetric_unit true', '-no_chainend_ter true',
     '-write_seqres_records true', '-output_pose_energies_table false', '-output_pose_cache_data false',
     '-holes:dalphaball %s' % dalphaball if os.path.exists(dalphaball) else '',  # This creates a new line if not used
     '-use_occurrence_data',  # Todo integrate into xml with Rosetta Source update
     '-preserve_header true', '-write_pdb_title_section_records true',
     '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer VirtualBB ShoveBB VirtualNTerm '
     'VirtualDNAPhosphate CTermConnect sc_orbitals pro_hydroxylated_case1 N_acetylated C_methylamidated cys_acetylated'
     'pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_diiodinated tyr_sulfated'
     'lys_dimethylated lys_monomethylated lys_trimethylated lys_acetylated glu_carboxylated MethylatedProteinCterm',
     '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol protocols.jd2.JobDistributor']

relax_pairs = ['-relax:ramp_constraints false', '-no_optH false', '-relax:coord_cst_stdev 0.5',
               '-nblist_autoupdate true', '-no_nstruct_label true', '-relax:bb_move false']  # Todo remove this one?
relax_singles = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:coord_constrain_sidechains', '-flip_HNQ',
                 '-no_his_his_pairE']
relax_flags = relax_singles + relax_pairs
relax_flags_cmdline = relax_singles + list(chain.from_iterable(map(str.split, relax_pairs)))
# relax_flags_cmdline = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints', 'false',
#                        '-no_optH', 'false', '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev', '0.5',
#                        '-no_his_his_pairE', '-flip_HNQ', '-nblist_autoupdate', 'true', '-no_nstruct_label', 'true',
#                        '-relax:bb_move', 'false']
rosetta_variables = [('scripts', rosetta_scripts), ('sym_score_patch', sym_weights),
                     ('solvent_sym_score_patch', solvent_weights_sym),
                     ('solvent_score_patch', solvent_weights)]
# Those jobs having a scale of 2 utilize two threads. Therefore, two commands are selected from a supplied commands list
# and are launched inside a python environment once the SLURM controller starts a SBATCH array job
process_scale = {refine: 2, interface_design: 2, stage[3]: 2, consensus: 2, nano: 2,
                 stage[6]: 1, stage[7]: 1, stage[8]: 1, stage[9]: 1, stage[10]: 1,
                 stage[11]: 1, stage[12]: 2, stage[13]: 2, optimize_designs: 2,
                 'metrics_bound': 2, interface_metrics: 2, 'hhblits': 1, 'bmdca': 2}
# Cluster Dependencies and Multiprocessing
sbatch_templates = {refine: os.path.join(sbatch_template_dir, refine),
                    interface_design: os.path.join(sbatch_template_dir, interface_design),
                    scout: os.path.join(sbatch_template_dir, interface_design),
                    stage[3]: os.path.join(sbatch_template_dir, interface_design),
                    stage[4]: os.path.join(sbatch_template_dir, refine),
                    consensus: os.path.join(sbatch_template_dir, refine),
                    nano: os.path.join(sbatch_template_dir, nano),
                    stage[6]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[7]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[8]: os.path.join(sbatch_template_dir, stage[6]),
                    stage[9]: os.path.join(sbatch_template_dir, stage[6]),
                    'metrics_bound': os.path.join(sbatch_template_dir, interface_design),
                    interface_metrics: os.path.join(sbatch_template_dir, interface_design),
                    optimize_designs: os.path.join(sbatch_template_dir, 'hhblits'),
                    'hhblits': os.path.join(sbatch_template_dir, 'hhblits'),
                    'bmdca': os.path.join(sbatch_template_dir, 'bmdca')
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
        file = 'output%s%s_%s.out' % (os.sep, job_id, array_task_number)
        for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
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
    file = 'output%s%s_%s.out' % (os.sep, job_id, array_task_number)
    for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
        # file = '%s_%s.out' % (job_id, task_id)
        run(file, log_files[i], program='cat')
        # run(file, '/dev/null', program='rm')


def create_file(file):
    """If file doesn't exist, create a blank one"""
    if not os.path.exists(file):
        with open(file, 'w') as new_file:
            dummy = True


def run(cmd: str, log_file_name: str, program: str = None, srun: str = None) -> bool:  #
    """Executes specified command and appends command results to log file

    Args:
        cmd: The name of a command file which should be executed by the system
        log_file_name: Location on disk of log file
        program: The interpreter for said command
        srun: Whether to utilize a job step prefix during command issuance
    Returns:
        Whether the command executed successfully
    """
    cluster_prefix = srun if srun else []
    program = [program] if program else []
    command = [cmd] if isinstance(cmd, str) else cmd
    with open(log_file_name, 'a') as log_f:
        command = cluster_prefix + program + command
        log_f.write('Command: %s\n' % subprocess.list2cmdline(command))
        p = subprocess.Popen(command, stdout=log_f, stderr=log_f)
        p.communicate()

    return p.returncode == 0


def distribute(file: Union[str, bytes] = None, out_path: Union[str, bytes] = os.getcwd(), scale: str = None,
               success_file: Union[str, bytes] = None, failure_file: Union[str, bytes] = None, max_jobs: int = 80,
               number_of_commands: int = None, mpi: int = None, log_file: Union[str, bytes] = None,
               finishing_commands: List[str] = None, **kwargs) -> str:
    """Take a file of commands formatted for execution in the SLURM environment and process into a sbatch script

    Args:
        file: The location of the file which contains your commands to distribute through a sbatch array
        out_path: Where to write out the sbatch script
        scale: The stage of design to distribute. Works with CommandUtils and PathUtils to allocate jobs
        success_file: What file to write the successful jobs to for job organization
        failure_file: What file to write the failed jobs to for job organization
        max_jobs: The size of the job array limiter. This caps the number of commands executed at once
        number_of_commands: The size of the job array. Inclusion circumvents automatic detection of corrupted commands
        mpi: The number of processes to run concurrently with MPI
        log_file: The name of a log file to write command results to
        finishing_commands: Commands to run once all sbatch processes are completed
    Returns:
        The name of the sbatch script that was written
    """
    # Should this be included in docstring?
    # If the commands are provided as a list of raw commands and not a command living in a PoseDirectory, the argument
    #     number_of_commands should be used! It will skip checking for the presence of commands in the corresponding
    #     PoseDirectory
    if not scale:
        # elif process_scale: Todo in order to make stage unnecessary, would need to provide scale and template
        #                      Could add a hyperthreading=True parameter to remove process scale
        #     command_divisor = process_scale
        # else:
        raise DesignError('No --stage specified. Required!!!')

    script_or_command = '%s is malformed at line %d!\n%s\nAll commands must either have a file extension or not. Cannot mix!'
    if number_of_commands:
        _commands = [0 for _ in range(number_of_commands)]
    elif file:  # Automatically detect if the commands file has executable scripts or errors
        # use collect_designs to get commands from the provided file
        _commands, location = collect_designs(files=[file])  # , directory=out_path)
        script_present, error = False, None
        for idx, _command in enumerate(_commands, 1):
            if _command.endswith('.sh'):  # the command string is a shell script, ex: "refine.sh"
                if not os.path.exists(_command):  # check for any missing commands and report
                    error = '%s is malformed at line %d! The command at location (%s) doesn\'t exist!'
                if idx != 1 and not script_present:  # There was a change from non-script files to script files
                    error = script_or_command
                script_present = True
            else:  # the command string is not a shell script
                if idx != 1 and script_present:  # There was a change from script files to non-script files
                    error = script_or_command
        if error:
            raise DesignError(error % (file, idx, _command))
    else:
        raise DesignError('You must pass number_of_commands or file to %s' % distribute.__name__)

    # Create success and failures files
    name = os.path.basename(os.path.splitext(file)[0])
    if not success_file:
        success_file = os.path.join(out_path, '%s_%s_success.log' % (name, sbatch))
    if not failure_file:
        failure_file = os.path.join(out_path, '%s_%s_failures.log' % (name, sbatch))
    output = os.path.join(out_path, 'sbatch_output')
    os.makedirs(output, exist_ok=True)

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
        new_f.write('%s%s\n\n' % (sb_flag, array))
        new_f.write('python %s --stage %s distribute %s--success_file %s --failure_file %s --command_file %s\n' %
                    (cmd_dist, scale, '--log_file %s ' % log_file if log_file else '', success_file, failure_file, file,
                     ))
        if finishing_commands:
            new_f.write('# Wait for all to complete\nwait\n\n# Then execute\n%s\n' % '\n'.join(finishing_commands))

    return filename


# @handle_errors(errors=(FileNotFoundError,))
def update_status(serialized_info: Union[str, bytes], stage: str, mode: str = 'check'):
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
    parser_distribute = subparsers.add_parser('distribute', help='Submit a job to SLURM for processing')
    parser_distribute.add_argument('-f', '--command_file',
                                   help='File with command(s) to be distributed. Required', required=True)
    parser_distribute.add_argument('-j', '--jobs', type=int, default=None,
                                   help='The number of jobs to be executed at once')
    parser_distribute.add_argument('-l', '--log_file', type=str, default=None,
                                   help='The name of the log file to append command stdout and stderr')
    parser_distribute.add_argument('-n', '--failure_file',
                                   help='The disk location of output file containing failed commands')
    parser_distribute.add_argument('-S', '--srun', action='store_true',
                                   help='Utilize srun to allocate resources, launch the job and communicate with SLURM')
    parser_distribute.add_argument('-y', '--success_file',
                                   help='The disk location of output file containing successful commands')
    # ---------------------------------------------------
    parser_status = subparsers.add_parser('status', help='Check the status of the command')
    parser_status.add_argument('-c', '--check', action='store_true', help='Check the status of the command')
    parser_status.add_argument('-i', '--info', type=str, help='The location of the state file')
    parser_status.add_argument('-s', '--set', action='store_true', help='Set the status as True')
    parser_status.add_argument('-r', '--remove', action='store_true', help='Set the status as False')

    args, additional_args = parser.parse_known_args()

    if args.module == 'status':
        mode = 'check' if args.check else 'set' if args.set else 'remove'
        update_status(args.info, args.stage, mode=mode)
    elif args.module == 'distribute':
        # Grab all possible poses
        with open(args.command_file, 'r') as cmd_f:
            all_commands = cmd_f.readlines()

        # Select exact poses to be handled according to array_ID and design stage
        # Todo change to args.number_of_processes instead of args.stage
        number_of_processes = process_scale.get(args.stage, 1)
        array_number = os.environ.get('SLURM_ARRAY_TASK_ID')
        if array_number:
            array_task_number = int(array_number)
            # adjust from SLURM one index and figure out how many commands to grab from command pool
            cmd_start_slice = (array_task_number - index_offset) * number_of_processes
            if cmd_start_slice > len(all_commands):
                exit()
            cmd_end_slice = cmd_start_slice + number_of_processes
        else:  # not in SLURM, use multiprocessing
            cmd_start_slice, cmd_end_slice = None, None
        specific_commands = list(map(str.strip, all_commands[cmd_start_slice:cmd_end_slice]))

        # Prepare Commands
        if len(specific_commands[0].split()) > 1:  # the commands probably have a program preceding the command
            program = None
            specific_commands = [cmd.split() for cmd in specific_commands]
        else:
            program = 'bash'

        if args.log_file:
            log_files = [args.log_file for _ in specific_commands]
        else:  # todo overlaps with len(specific_commands[0].split()) > 1 as only shell scripts really satisfy this
            log_files = ['%s.log' % os.path.splitext(shell_path)[0] for shell_path in specific_commands]

        # iteration = 0
        # complete = False
        # Todo implementing an srun prefix to any command allows for multiple job steps to be controlled. This is useful
        #  when a prior step gets hung up and needs to be cancelled, but the remaining job steps should be executed
        #  downside to all this is that the allocation is done by inherently neglecting the hyperthreading. The srun
        #  would respect the one cpu, one task logic.
        # while not complete:
        #     allocation = ['srun', '-c', 1, '-p', 'long', '--mem-per-cpu', CUtils.memory_scale[args.stage]]
        #     allocation = None
        #     zipped_commands = zip(command_paths, log_files, repeat(allocation))
        # print('Running command:\n', subprocess.list2cmdline(specific_commands[0]))
        zipped_commands = zip(specific_commands, log_files, repeat(program))

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
        if number_of_commands > 1:  # set by process_scale
            processes = calculate_mp_cores(cores=number_of_commands, jobs=args.jobs)
            results = mp_starmap(run, zipped_commands, processes=processes)
        else:
            results = [run(*command) for command in zipped_commands]
        #    iteration += 1

        # Write out successful and failed commands
        with open(args.success_file, 'a') as f:
            for i, result in enumerate(results):
                if result:
                    f.write('%s\n' % specific_commands[i])

        with open(args.failure_file, 'a') as f:
            for i, result in enumerate(results):
                if not result:
                    f.write('%s\n' % specific_commands[i])

        # # Append SLURM output to log_file(s)
        # job_id = int(os.environ.get('SLURM_JOB_ID'))
        # for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
        #     file = os.path.join(sbatch_output, '%s_%s.out' % (job_id, array_task_number))  # Todo set sbatch_output
        #     # file = '%s_%s.out' % (job_id, task_id)
        #     run(file, log_files[i], program='cat')
        #     # run(file, '/dev/null', program='rm')
