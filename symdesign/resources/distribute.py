"""Module for Distribution of commands found for individual poses to SLURM/PBS computational cluster
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import AnyStr, Literal, get_args

from symdesign import flags
from symdesign.utils import collect_designs, InputError, path as putils, pickle_object, unpickle

# Globals
cmd_dist = os.path.abspath(__file__)
logger = logging.getLogger(__name__)
index_offset = 1
mpi = 4
hhblits_memory_threshold = 30000000000  # 30GB
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
]
protocols: tuple[str, ...] = get_args(protocols_literal)
# Cluster Dependencies and Multiprocessing
processes = (2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2)
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
    os.path.join(sbatch_template_dir, 'bmdca')
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


def distribute(file: AnyStr, scale: protocols_literal, out_path: AnyStr = os.getcwd(),
               success_file: AnyStr = None, failure_file: AnyStr = None, max_jobs: int = 80,
               number_of_commands: int = None, mpi: int = None, log_file: AnyStr = None,
               finishing_commands: list[str] = None, batch: bool = is_sbatch_available(), **kwargs) -> str:
    """Take a file of commands formatted for execution in the SLURM environment and process into a sbatch script

    Args:
        file: The location of the file which contains your commands to distribute through a sbatch array
        scale: The stage of design to distribute. Works with CommandUtils and PathUtils to allocate jobs
        out_path: Where to write out the sbatch script
        success_file: What file to write the successful jobs to for job organization
        failure_file: What file to write the failed jobs to for job organization
        max_jobs: The size of the job array limiter. This caps the number of commands executed at once
        number_of_commands: The size of the job array. Inclusion circumvents automatic detection of corrupted commands
        mpi: The number of processes to run concurrently with MPI
        log_file: The name of a log file to write command results to
        finishing_commands: Commands to run once all sbatch processes are completed
        batch: Whether the distribution file should take the form of a slurm sbatch script
    Returns:
        The name of the script that was written
    """
    # Should this be included in docstring?
    # If the commands are provided as a list of raw commands and not a command living in a PoseJob, the argument
    #     number_of_commands should be used! It will skip checking for the presence of commands in the corresponding
    #     PoseJob
    # if scale is None:
    #     # elif process_scale: Todo in order to make stage unnecessary, would need to provide scale and template
    #     #                      Could add a hyperthreading=True parameter to remove process scale
    #     #     command_divisor = process_scale
    #     # else:
    #     raise InputError('Required argument "scale" not specified')

    script_or_command = \
        '{} is malformed at line {}. All commands should match.\n* * *\n{}\n* * *' \
        '\nEither a file extension OR a command requried. Cannot mix'
    if number_of_commands is None:
        # Automatically detect if the commands file has executable scripts or errors
        # Use collect_designs to get commands from the provided file
        commands, _ = collect_designs(files=[file])
        # Check if the file lines (commands) contain a script or a command
        scripts = True if commands[0].endswith('.sh') else False
        start_idx = 1
        for idx, directive in enumerate(commands[start_idx:], start_idx):
            # Check if the command string is a shell script type file string. Ex: "refine.sh"
            if directive.endswith('.sh'):  # This is a file
                if not os.path.exists(directive):  # Check if file is missing
                    raise InputError(f"{file} is malformed at line {idx}. "
                                     f"The command at location '{directive}' doesn't exist")
                if not scripts:  # There was a change from non-script files
                    raise InputError(script_or_command.format(file, idx, directive))
            else:  # directive is a command
                # Check if there was a change from script files to non-script files
                if scripts:
                    raise InputError(script_or_command.format(file, idx, directive))
                else:
                    scripts = False

        number_of_commands = len(commands)
    # else:
    #     # commands = [0 for _ in range(number_of_commands)]
    #     pass

    # else:
    #     raise InputError(f'Must pass "number_of_commands" or "file" to {distribute.__name__}')

    # Create success and failures files
    name = os.path.basename(os.path.splitext(file)[0])
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
        new_f.write(f'python {cmd_dist} --stage {scale} distribute {f"--log_file {log_file} " if log_file else ""}'
                    f'--success_file {success_file} --failure_file {failure_file} --command_file {file}')
        if finishing_commands:
            if batch:
                new_f.write('\n# Wait for all to complete\n'
                            'wait\n'
                            '\n'
                            '# Then execute\n'
                            '%s\n' % '\n'.join(finishing_commands))
            else:
                new_f.write('&&\n# Wait for all to complete, then execute\n'
                            '%s\n' % '\n'.join(finishing_commands))
        else:
            new_f.write('\n')

    return filename


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
