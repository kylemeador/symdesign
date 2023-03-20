"""Module for Distribution of commands found for individual poses to SLURM/PBS computational cluster
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import AnyStr, Iterable, Literal, Sequence, get_args

from symdesign import flags
from symdesign.utils import collect_designs, InputError, path as putils, pickle_object, unpickle

# Globals
logger = logging.getLogger(__name__)
logger.setLevel(20)
logger.addHandler(logging.StreamHandler())
index_offset = 1
mpi = 4
hhblits_memory_threshold = 20e9  # 20 GB
default_shell = 'bash'
sbatch = 'sbatch'
sb_flag = '#SBATCH --'
sbatch_template_dir = os.path.join(putils.dependency_dir, sbatch)
sbatch_exe = ''


# def get_sbatch_exe() -> AnyStr | None:
#     """Locate where in the $PATH the executable 'sbatch' can be found"""
#     return shutil.which(sbatch)
#     # p = subprocess.Popen(['which', sbatch], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#     # sbatch_exe_out, err = p.communicate()
#     # return sbatch_exe_out.decode('utf-8').strip()


def is_sbatch_available() -> bool:
    """Ensure the sbatch executable is available and executable"""
    global sbatch_exe
    sbatch_exe = shutil.which(sbatch)  # get_sbatch_exe()
    # if sbatch_exe is not None:
    try:
        return os.path.exists(sbatch_exe) and os.access(sbatch_exe, os.X_OK)
    except TypeError:  # NoneType
        return False


# Todo modify .linuxgccrelease depending on os

# relax_flags_cmdline = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints', 'false',
#                        '-no_optH', 'false', '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev', '0.5',
#                        '-no_his_his_pairE', '-flip_HNQ', '-nblist_autoupdate', 'true', '-no_nstruct_label', 'true',
#                        '-relax:bb_move', 'false']
# Those jobs having a scale of 2 utilize two threads. Therefore, two commands are selected from a supplied commands list
# and are launched inside a python environment once the SLURM controller starts a SBATCH array job
protocols_literal = Literal[
    'refine',
    'interface-design',
    'design',
    'consensus',
    'nanohedra',
    'rmsd_calculation',
    'all_to_all',
    'rmsd_clustering',
    'rmsd_to_cluster',
    'scout',
    'hbnet_design_profile',
    'optimize-designs',
    'interface-metrics',
    'hhblits',
    'bmdca',
    'predict-structure',
]
protocols: tuple[str, ...] = get_args(protocols_literal)
# Cluster Dependencies and Multiprocessing
processes = (2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1)
process_scale = dict(zip(protocols, processes))

# process_scale = {
#     flags.refine: 2,
#     flags.interface_design: 2,
#     flags.consensus: 2,
#     flags.nanohedra: 1,
#     'rmsd_calculation': 1,
#     'all_to_all': 1,
#     'rmsd_clustering': 1,
#     'rmsd_to_cluster': 1,
#     flags.scout: 2,
#     putils.hbnet_design_profile: 2,
#     flags.optimize_designs: 2,
#     flags.interface_metrics: 2,
#     putils.hhblits: 1,
#     'bmdca': 2}

sbatch_templates_tuple = (
    os.path.join(sbatch_template_dir, flags.refine),
    os.path.join(sbatch_template_dir, flags.interface_design),
    os.path.join(sbatch_template_dir, flags.design),
    os.path.join(sbatch_template_dir, flags.refine),
    os.path.join(sbatch_template_dir, flags.nanohedra),
    os.path.join(sbatch_template_dir, 'rmsd_calculation'),
    os.path.join(sbatch_template_dir, 'rmsd_calculation'),
    os.path.join(sbatch_template_dir, 'rmsd_calculation'),
    os.path.join(sbatch_template_dir, 'rmsd_calculation'),
    os.path.join(sbatch_template_dir, flags.interface_design),
    os.path.join(sbatch_template_dir, flags.interface_design),
    os.path.join(sbatch_template_dir, putils.hhblits),
    os.path.join(sbatch_template_dir, flags.interface_design),
    os.path.join(sbatch_template_dir, putils.hhblits),
    os.path.join(sbatch_template_dir, 'bmdca'),
    os.path.join(sbatch_template_dir, flags.predict_structure)
)
sbatch_templates = dict(zip(protocols, sbatch_templates_tuple))
# sbatch_templates = {
#     flags.refine: os.path.join(sbatch_template_dir, flags.refine),
#     flags.interface_design: os.path.join(sbatch_template_dir, flags.interface_design),
#     flags.consensus: os.path.join(sbatch_template_dir, flags.refine),
#     flags.nanohedra: os.path.join(sbatch_template_dir, flags.nanohedra),
#     'rmsd_calculation': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
#     'all_to_all': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
#     'rmsd_clustering': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
#     'rmsd_to_cluster': os.path.join(sbatch_template_dir, 'rmsd_calculation'),
#     flags.scout: os.path.join(sbatch_template_dir, flags.interface_design),
#     putils.hbnet_design_profile: os.path.join(sbatch_template_dir, flags.interface_design),
#     flags.optimize_designs: os.path.join(sbatch_template_dir, putils.hhblits),
#     flags.interface_metrics: os.path.join(sbatch_template_dir, flags.interface_design),
#     putils.hhblits: os.path.join(sbatch_template_dir, putils.hhblits),
#     'bmdca': os.path.join(sbatch_template_dir, 'bmdca')
# }
# # 'metrics': os.path.join(sbatch_template_dir, flags.interface_design),
# # 'metrics_bound': os.path.join(sbatch_template_dir, flags.interface_design),
# # flags.analysis: os.path.join(sbatch_template_dir, flags.refine),


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


def run(cmd: list[str] | AnyStr, log_file_name: str, program: str = None, srun: str = None) -> bool:
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
            log_f.write(f'Command: {subprocess.list2cmdline(command)}\n')
            p = subprocess.Popen(command, stdout=log_f, stderr=log_f)
            p.communicate()
    else:
        logger.info(f'Command: {subprocess.list2cmdline(command)}\n')
        p = subprocess.Popen(command)
        p.communicate()

    return p.returncode == 0


def check_scripts_exist(directives: Iterable[str] = None, file: AnyStr = None):
    """Check for the existence of scripts provided by an Iterable or present in a file

    Args:
        directives: The locations of scripts which should be executed
        file: The location of a file containing the location(s) of scripts/commands
    Raises:
        InputError: When the scripts/commands passed are malformed or do not exist
    Returns:
        None
    """
    script_or_command = \
        '{} is malformed at line {}. All commands should match.\n* * *\n{}\n* * *' \
        '\nEither a file extension OR a command requried. Cannot mix'
    # Automatically detect if the commands file has executable scripts or errors
    if directives is None:
        if file is None:
            raise ValueError(f"Must pass either 'directives' or 'file'. Neither were passed")
        else:
            # Use collect_designs to get commands from the provided file
            scripts, _ = collect_designs(files=[file])

    # Check if the file lines (commands) contain a script or a command
    first_directive, *remaining_directives = directives
    contains_scripts = True if first_directive.endswith('.sh') else False
    for idx, directive in enumerate(remaining_directives, 1):
        # Check if the directive string is a shell script type file string. Ex: "refine.sh"
        if directive[-3:] == '.sh':  # This is a file
            if not os.path.exists(directive):  # Check if file is missing
                raise InputError(
                    f"{file} is malformed at line {idx}. The command at location '{directive}' doesn't exist")
            if not contains_scripts:  # There was a change from non-script files
                raise InputError(script_or_command.format(file, idx, directive))
        else:  # directive is a command
            # Check if there was a change from script files to non-script files
            if contains_scripts:
                raise InputError(script_or_command.format(file, idx, directive))
            else:
                contains_scripts = False


def distribute(file: AnyStr, scale: protocols_literal, number_of_commands: int, out_path: AnyStr = os.getcwd(),
               success_file: AnyStr = None, failure_file: AnyStr = None, log_file: AnyStr = None, max_jobs: int = 80,
               mpi: int = None,
               finishing_commands: Iterable[str] = None, batch: bool = is_sbatch_available(), **kwargs) -> str:
    """Take a file of commands formatted for execution in the SLURM environment and process into a sbatch script

    Args:
        file: The location of the file which contains your commands to distribute through a sbatch array
        scale: The stage of design to distribute. Works with CommandUtils and PathUtils to allocate jobs
        number_of_commands: The size of the job array
        out_path: Where to write out the sbatch script
        success_file: What file to write the successful jobs to for job organization
        failure_file: What file to write the failed jobs to for job organization
        log_file: The name of a log file to write command results to
        max_jobs: The size of the job array limiter. This caps the number of commands executed at once
        mpi: The number of processes to run concurrently with MPI
        finishing_commands: Commands to run once all sbatch processes are completed
        batch: Whether the distribution file should be formatted as a SLURM sbatch script
    Returns:
        The name of the script that was written
    """
    # Create success and failures files
    name, ext = os.path.splitext(os.path.basename(file))
    if success_file is None:
        success_file = os.path.join(out_path, f'{name}-{sbatch}.success')
    if failure_file is None:
        failure_file = os.path.join(out_path, f'{name}-{sbatch}.failures')
    output = os.path.join(out_path, 'sbatch_output')
    putils.make_path(output)

    # Make sbatch file from template, array details, and command distribution script
    if batch:
        filename = os.path.join(out_path, f'{name}_{sbatch}.sh')
    else:
        filename = os.path.join(out_path, f'{name}.sh')

    with open(filename, 'w') as new_f:
        # Todo set up sbatch accordingly. Include a multiplier for the number of CPU's. Actually, might be passed
        #  if mpi:
        #      do_mpi_stuff = True
        if batch:
            # grab and write sbatch template
            with open(sbatch_templates[scale]) as template_f:
                new_f.write(''.join(template_f.readlines()))
            out = f'output={output}/%A_%a.out'
            new_f.write(f'{sb_flag}{out}\n')
            array = f'array=1-{int(number_of_commands / process_scale[scale] + 0.5)}%{max_jobs}'
            new_f.write(f'{sb_flag}{array}\n\n')
        new_f.write(f'python {putils.distributer_tool} --stage {scale} {f"--log-file {log_file} " if log_file else ""}'
                    f'--success-file {success_file} --failure-file {failure_file} --command-file {file}')
        if finishing_commands:
            if batch:
                new_f.write('\n# Wait for all to complete\n'
                            'wait\n'
                            '\n'
                            '# Then execute\n'
                            '%s\n' % '\n'.join(finishing_commands))
            else:
                new_f.write(' &&\n# Wait for all to complete, then execute\n'
                            '%s\n' % '\n'.join(finishing_commands))
        else:
            new_f.write('\n')

    return filename


sbatch_warning = 'Ensure the SBATCH script(s) below are correct. Specifically, check that the job array and any '\
                 'node specifications are accurate. You can look at the SBATCH manual (man sbatch or sbatch --help) to'\
                 ' understand the variables or ask for help if you are still unsure'
script_warning = 'Ensure the script(s) below are correct'


def commands(commands: Sequence[str], name: str, protocol: protocols_literal,
             out_path: AnyStr = os.getcwd(), commands_out_path: AnyStr = None, **kwargs) -> str:
    """Given a batch of commands, write them to a file and distribute that work for completion using specified
    computational resources

    Args:
        commands: The commands which should be written to a file and then formatted for distribution
        name: The name of the collection of commands. Will be applied to commands file and distribution file(s)
        protocol: The type of protocol to distribute
        out_path: Where should the distributed script be written?
        commands_out_path: Where should the commands file be written? If not specified, is written to out_path
    Keyword Args:
        success_file: AnyStr = None - What file to write the successful jobs to for job organization
        failure_file: AnyStr = None - What file to write the failed jobs to for job organization
        log_file: AnyStr = None - The name of a log file to write command results to
        max_jobs: int = 80 - The size of the job array limiter. This caps the number of commands executed at once
        mpi: bool = False - The number of processes to run concurrently with MPI
        finishing_commands: Iterable[str] = None - Commands to run once all sbatch processes are completed
        batch: bool = is_sbatch_available() - Whether the distribution file should be formatted as a SLURM sbatch script
    Returns:
        The name of the distribution script that was written
    """
    if is_sbatch_available():
        shell = sbatch
        logger.info(sbatch_warning)
    else:
        shell = default_shell
        logger.info(script_warning)

    putils.make_path(out_path)
    if commands_out_path is None:
        commands_out_path = out_path
    else:
        putils.make_path(commands_out_path)
    command_file = write_commands(commands, name=name, out_path=commands_out_path)
    script_file = distribute(command_file, protocol, len(commands), out_path=out_path, **kwargs)

    logger.info(f'Once you are satisfied, enter the following to distribute:\n\t{shell} {script_file}')

    return script_file


# @handle_errors(errors=(FileNotFoundError,))
def update_status(serialized_info: AnyStr, stage: str, mode: str = 'check'):
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


def write_script(command: str, name: str = 'script', out_path: AnyStr = os.getcwd(),
                 additional: list = None, shell: str = 'bash', status_wrap: str = None) -> AnyStr:
    """Take a command and write to a name.sh script. By default, bash is used as the shell interpreter

    Args:
        command: The command formatted using subprocess.list2cmdline(list())
        name: The name of the output shell script
        out_path: The location where the script will be written
        additional: Additional commands also formatted using subprocess.list2cmdline()
        shell: The shell which should interpret the script
        status_wrap: The name of a file in which to check and set the status of the command in the shell
    Returns:
        The name of the file
    """
    if status_wrap:
        modifier = '&&'
        _base_cmd = ['python', putils.distributer_tool, '--stage', name, 'status', '--info', status_wrap]
        check = subprocess.list2cmdline(_base_cmd + ['--check', modifier, '\n'])
        _set = subprocess.list2cmdline(_base_cmd + ['--set'])
    else:
        check = _set = modifier = ''

    file_name = os.path.join(out_path, name if name.endswith('.sh') else f'{name}.sh')
    with open(file_name, 'w') as f:
        f.write(f'#!/bin/{shell}\n\n{check}{command} {modifier}\n\n')
        if additional:
            f.write('%s\n\n' % ('\n\n'.join(f'{command} {modifier}' for command in additional)))
        f.write(f'{_set}\n')

    return file_name


def write_commands(commands: Iterable[str], name: str = 'all_commands', out_path: AnyStr = os.getcwd()) -> AnyStr:
    """Write a list of commands out to a file

    Args:
        commands: An iterable with the commands as values
        name: The name of the file. Will be appended with '.cmd(s)'
        out_path: The directory where the file will be written
    Returns:
        The filename of the new file
    """
    file = os.path.join(out_path, f'{name}.cmds' if len(commands) > 1 else f'{name}.cmd')
    with open(file, 'w') as f:
        f.write('%s\n' % '\n'.join(command for command in commands))

    return file
