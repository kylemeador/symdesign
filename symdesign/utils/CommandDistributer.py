"""Module for Distribution of commands found for individual poses to SLURM/PBS computational cluster
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
from itertools import repeat, chain
from typing import AnyStr

from symdesign import flags, utils
from symdesign.utils.path import sbatch_template_dir, rosetta_main, rosetta_extras, dalphaball, submodule_help, \
    program_name, rosetta_scripts_dir, sym_weights, solvent_weights_sym, solvent_weights, hbnet_design_profile, \
    hhblits

# Globals
cmd_dist = os.path.abspath(__file__)
logger = logging.getLogger(__name__)
index_offset = 1
min_cores_per_job = 1  # currently one for the MPI node, and 5 workers
mpi = 4
num_thread_per_process = 2
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
                'cxx11thread': [f'-multithreading:total_threads {num_thread_per_process}',
                                f'-multithreading:interaction_graph_threads {num_thread_per_process}'],
                'mpi': [],
                'cxx11threadmpi': [f'-multithreading:total_threads {num_thread_per_process}']}
# Todo modify .linuxgccrelease depending on os
script_cmd = [os.path.join(rosetta_main, 'source', 'bin', f'rosetta_scripts.{rosetta_extras}.linuxgccrelease'),
              '-database', os.path.join(rosetta_main, 'database')]
rosetta_flags = extras_flags[rosetta_extras] + \
    ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res', '-ignore_zero_occupancy false',
     # '-overwrite',
     '-linmem_ig 10', '-out:file:scorefile_format json', '-output_only_asymmetric_unit true', '-no_chainend_ter true',
     '-write_seqres_records true', '-output_pose_energies_table false', '-output_pose_cache_data false',
     f'-holes:dalphaball {dalphaball}' if os.path.exists(dalphaball) else '',  # This creates a new line if not used
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
rosetta_variables = [('scripts', rosetta_scripts_dir), ('sym_score_patch', sym_weights),
                     ('solvent_sym_score_patch', solvent_weights_sym),
                     ('solvent_score_patch', solvent_weights)]
# Those jobs having a scale of 2 utilize two threads. Therefore, two commands are selected from a supplied commands list
# and are launched inside a python environment once the SLURM controller starts a SBATCH array job
process_scale = {
    flags.refine: 2, flags.interface_design: 2, 'metrics': 2, flags.consensus: 2, flags.nanohedra: 2,
    'rmsd_calculation': 1, 'all_to_all': 1, 'rmsd_clustering': 1, 'rmsd_to_cluster': 1, 'rmsd': 1, 'all_to_cluster': 1,
    flags.scout: 2, hbnet_design_profile: 2, flags.optimize_designs: 2, 'metrics_bound': 2, flags.interface_metrics: 2,
    hhblits: 1, 'bmdca': 2}
# Cluster Dependencies and Multiprocessing
sbatch_templates = {flags.refine: os.path.join(sbatch_template_dir, flags.refine),
                    flags.interface_design: os.path.join(sbatch_template_dir, flags.interface_design),
                    flags.scout: os.path.join(sbatch_template_dir, flags.interface_design),
                    'metrics': os.path.join(sbatch_template_dir, flags.interface_design),
                    flags.analysis: os.path.join(sbatch_template_dir, flags.refine),
                    flags.consensus: os.path.join(sbatch_template_dir, flags.refine),
                    flags.nanohedra: os.path.join(sbatch_template_dir, flags.nanohedra),
                    'rmsd_calculation': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
                    'all_to_all': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
                    'rmsd_clustering': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
                    'rmsd_to_cluster': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
                    'metrics_bound': os.path.join(sbatch_template_dir, flags.interface_design),
                    flags.interface_metrics: os.path.join(sbatch_template_dir, flags.interface_design),
                    flags.optimize_designs: os.path.join(sbatch_template_dir, hhblits),
                    hhblits: os.path.join(sbatch_template_dir, hhblits),
                    'bmdca': os.path.join(sbatch_template_dir, 'bmdca')
                    }


# class GracefulKiller:
#     kill_now = False
#
#     def __init__(self):
#         signal.signal(signal.SIGINT, self.exit_gracefully)
#         signal.signal(signal.SIGTERM, self.exit_gracefully)
#
#     def exit_gracefully(self, signum, frame):
#         self.kill_now = True
#         with open(args.failure_file, 'a') as f:
#             for i, pose in enumerate(command_paths):
#                 f.write('%s\n' % pose)
#
#         # Append SLURM output to log_file(s)
#         job_id = int(os.environ.get('SLURM_JOB_ID'))
#         file = 'output%s%s_%s.out' % (os.sep, job_id, array_task_number)
#         for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
#             # file = '%s_%s.out' % (job_id, task_id)
#             run(file, log_files[i], program='cat')
#             # run(file, '/dev/null', program='rm')
#         return None


def create_file(file: AnyStr = None):
    """If file doesn't exist, create a blank one"""
    if file and not os.path.exists(file):
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
    if log_file_name:
        with open(log_file_name, 'a') as log_f:
            command = cluster_prefix + program + command
            log_f.write('Command: %s\n' % subprocess.list2cmdline(command))
            p = subprocess.Popen(command, stdout=log_f, stderr=log_f)
            p.communicate()
    else:
        p = subprocess.Popen(command)
        p.communicate()

    return p.returncode == 0


def distribute(file: AnyStr = None, out_path: AnyStr = os.getcwd(), scale: str = None,
               success_file: AnyStr = None, failure_file: AnyStr = None, max_jobs: int = 80,
               number_of_commands: int = None, mpi: int = None, log_file: AnyStr = None,
               finishing_commands: list[str] = None, **kwargs) -> str:
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
    if scale is None:
        # elif process_scale: Todo in order to make stage unnecessary, would need to provide scale and template
        #                      Could add a hyperthreading=True parameter to remove process scale
        #     command_divisor = process_scale
        # else:
        raise utils.InputError('Required argument "stage" not specified')

    script_or_command = \
        '{} is malformed at line {}. All commands should match.\n* * *\n{}\n* * *' \
        '\nEither a file extension OR a command requried. Cannot mix'
    if number_of_commands:
        directives = [0 for _ in range(number_of_commands)]
    elif file:  # Automatically detect if the commands file has executable scripts or errors
        # use collect_designs to get commands from the provided file
        directives, location = utils.collect_designs(files=[file])  # , directory=out_path)
        # Check if the file lines (directives) contain a script or a command
        scripts = True if directives[0].endswith('.sh') else False
        # command_present = not scripts
        start_idx = 1
        for idx, directive in enumerate(directives[start_idx:], start_idx):
            # Check if the command string is a shell script type file string. Ex: "refine.sh"
            if directive.endswith('.sh'):  # This is a file
                if not os.path.exists(directive):  # Check if file is missing
                    raise utils.InputError(f"{file} is malformed at line {idx}. "
                                           f"The command at location '{directive}' doesn't exist")
                if not scripts:  # There was a change from non-script files
                    raise utils.InputError(script_or_command.format(file, idx, directive))
            else:  # directive is a command
                # Check if there was a change from script files to non-script files
                if scripts:
                    raise utils.InputError(script_or_command.format(file, idx, directive))
                else:
                    scripts = False

    else:
        raise utils.InputError(f'Must pass number_of_commands or file to {distribute.__name__}')

    # Create success and failures files
    name = os.path.basename(os.path.splitext(file)[0])
    if success_file is None:
        success_file = os.path.join(out_path, f'{name}-{sbatch}.success')
    if failure_file is None:
        failure_file = os.path.join(out_path, f'{name}-{sbatch}.failures')
    output = os.path.join(out_path, 'sbatch_output')
    os.makedirs(output, exist_ok=True)

    # Make sbatch file from template, array details, and command distribution script
    filename = os.path.join(out_path, f'{name}_{sbatch}.sh')
    with open(filename, 'w') as new_f:
        # Todo set up sbatch accordingly. Include a multiplier for the number of CPU's. Actually, might be passed
        # if mpi:
        #     do_mpi_stuff = True
        # grab and write sbatch template
        with open(sbatch_templates[scale]) as template_f:
            new_f.write(''.join(template_f.readlines()))
        out = f'output={output}/%A_%a.out'
        new_f.write(f'{sb_flag}{out}\n')
        array = f'array=1-{int(len(directives) / process_scale[scale] + 0.5)}%{max_jobs}'
        new_f.write(f'{sb_flag}{array}\n\n')
        new_f.write(f'python {cmd_dist} --stage {scale} distribute %s'
                    f'--success_file {success_file} --failure_file {failure_file} --command_file {file}\n'
                    % f'--log_file {log_file} ' if log_file else '')
        if finishing_commands:
            new_f.write('# Wait for all to complete\nwait\n\n# Then execute\n%s\n' % '\n'.join(finishing_commands))

    return filename


# @handle_errors(errors=(FileNotFoundError,))
def update_status(serialized_info: AnyStr, stage: str, mode: str = 'check'):
    """Update the serialized info for a designs commands such as checking or removing status, and marking completed"""
    info = utils.unpickle(serialized_info)
    if mode == 'check':
        if info['status'][stage]:  # if the status of the stage is True
            exit(1)
    elif mode == 'set':
        info['status'][stage] = True
        utils.pickle_object(info, name=serialized_info, out_path='')
        # exit()
    elif mode == 'remove':
        info['status'][stage] = False
        utils.pickle_object(info, name=serialized_info, out_path='')
        # exit()
    else:
        exit(127)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}\nGather commands set up by '
                                                 f'{program_name} and distribute to computational nodes for processing')
    parser.add_argument('--stage', choices=tuple(process_scale.keys()),
                        help='The stage of design to be distributed. Each stage has optimal computing requirements to '
                             f'maximally utilize computers. One of {", ".join(list(process_scale.keys()))}')
    subparsers = parser.add_subparsers(title='SubModules', dest='module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter: \n'
                                            f"{submodule_help}\n\nAny module's help can be accessed in this way")
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
        # set the type for below if the specific command can be split further
        specific_commands: list[str] | list[list[str]] = list(map(str.strip, all_commands[cmd_start_slice:cmd_end_slice]))

        # Prepare Commands
        if len(specific_commands[0].split()) > 1:  # the commands probably have a program preceding the command
            program = None
            specific_commands = [cmd.split() for cmd in specific_commands]
        else:
            program = 'bash'

        if args.log_file:
            log_files = [args.log_file for _ in specific_commands]
        elif program == 'bash':
            # v this overlaps with len(specific_commands[0].split()) > 1 as only shell scripts really satisfy this
            log_files = [f'{os.path.splitext(shell_path)[0]}.log' for shell_path in specific_commands]
        else:
            log_files = [None for shell_path in specific_commands]

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
                f.write('%s\n' % '\n'.join(specific_commands))

            # Handle SLURM output
            job_id = os.environ.get('SLURM_JOB_ID')
            file = f'output{os.sep}{job_id}_{array_task_number}.out'
            # for idx, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
            for log_file in log_files:
                # Append SLURM output to log_file(s)
                run(file, log_file, program='cat')
                # # Remove SLURM output
                # run(file, '/dev/null', program='rm')

        # Run commands in parallel
        # monitor = GracefulKiller()  # TODO solution to SIGTERM. TEST shows this doesn't appear to be possible...
        signal.signal(signal.SIGINT, exit_gracefully)
        # signal.signal(signal.SIGKILL, exit_gracefully)  # Doesn't work, not possible
        signal.signal(signal.SIGTERM, exit_gracefully)
        # while not monitor.kill_now:

        number_of_commands = len(specific_commands)  # different from process scale as this could reflect edge cases
        if number_of_commands > 1:  # set by process_scale
            processes = utils.calculate_mp_cores(cores=number_of_commands, jobs=args.jobs)
            results = utils.mp_starmap(run, zipped_commands, processes=processes)
        else:
            results = [run(*command) for command in zipped_commands]
        #    iteration += 1

        # Write out successful and failed commands
        with open(args.success_file, 'a') as f:
            for i, result in enumerate(results):
                if result:
                    f.write('%s\n' % specific_commands[i] if program else subprocess.list2cmdline(specific_commands[i]))

        with open(args.failure_file, 'a') as f:
            for i, result in enumerate(results):
                if not result:
                    f.write('%s\n' % specific_commands[i] if program else subprocess.list2cmdline(specific_commands[i]))

        # # Append SLURM output to log_file(s)
        # job_id = int(os.environ.get('SLURM_JOB_ID'))
        # for i, task_id in enumerate(range(cmd_start_slice, cmd_end_slice)):
        #     file = os.path.join(sbatch_output, '%s_%s.out' % (job_id, array_task_number))  # Todo set sbatch_output
        #     # file = '%s_%s.out' % (job_id, task_id)
        #     run(file, log_files[i], program='cat')
        #     # run(file, '/dev/null', program='rm')
