from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import AnyStr

import symdesign

logger = logging.getLogger(__name__)
# Project strings and file names
utils_dir = Path(__file__).absolute().parent  # Reveals utils subdirectory
python_source = utils_dir.parent  # Reveals the root python code directory
git_source = python_source.parent  # Reveals the root git directory
try:
    p = subprocess.Popen(['git', '--git-dir', str(git_source / '.git'), 'rev-parse', 'HEAD'],
                         stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    commit = stdout.decode().strip()
    commit_short = commit[:9]
except subprocess.CalledProcessError:
    commit = commit_short = symdesign.version  # 'unknown'

program_name = 'SymDesign'
program_exe = python_source.name  # Exported at setup, essentially where __main__.py is located
# Project install and setup paths
documentation_dir = git_source / 'docs'
readme_file = documentation_dir / 'README.md'
conda_env_path = git_source / 'env.yml'
tools = python_source / 'tools'
setup_exe = tools / 'setup.py'
hhsuite_setup_command = f'python {setup_exe} --hhsuite-database'
program_install = f'pip install {program_name.lower()}'
# Program operations strings
program_help = f'{program_exe} --help'
program_guide = f'{program_exe} --guide'
submodule_guide = f'{program_exe} MODULE --guide'
submodule_help = f'{program_exe} MODULE --help'
guide_string = f"{program_name} guide. Enter '{program_guide}'"
program_output = f'{program_name}Output'
projects = 'Projects'
data = 'data'
default_path_file = '{}_{}_{}.poses'
default_execption_file = '{}_{}_{}_exceptions.poses'
default_specification_file = '{}_{}_{}.specification'
default_analysis_file = '{}-{}PoseMetrics.csv'
default_clustered_pose_file = '{}ClusteredPoses-{}.pkl'
nanohedra = 'nanohedra'
nano_publication = 'Laniado, Meador, & Yeates, PEDS. 2021'
fragment_docking = 'fragment_docking'
fragment_docking_publication = 'Agdanowski, Meador, & Yeates, ACS Nano. 2024 In Preparation'
output = 'output'
orient = 'orient'
protocol = 'protocol'
design_parent = 'design_parent'
hhblits = 'hhblits'
reference = 'reference'
# Memory Requirements
# 10.5MB was measured 5/13/22 with self.pose,
# after deleting self.pose: 450 poses -> 6.585339. 900 poses -> 3.346759
approx_ave_design_directory_memory_w_pose = 4_000_000  # 4MB
approx_ave_design_directory_memory_w_o_pose = 150_000  # with 44279 poses get 147299.66 bytes/pose
approx_ave_design_directory_memory_w_assembly = 500_000_000  # 500MB
baseline_program_memory = 3_000_000_000  # 3GB
nanohedra_memory = 30_000_000_000  # 30Gb
# Package paths
sym_op_location = utils_dir / 'symmetry_operators'
point_group_symmetry_operator_location = sym_op_location / 'point_group_operators.pkl'
point_group_symmetry_operatorT_location = sym_op_location / 'point_group_operators_transposed.pkl'
space_group_symmetry_operator_location = sym_op_location / 'space_group_operators.pkl'
space_group_symmetry_operatorT_location = sym_op_location / 'space_group_operators_transposed.pkl'
config_file = python_source / 'cfg.json'
models_to_multimodel_exe = tools / 'models_to_multimodel.py'
list_pdb_files = tools / 'list_files_in_directory.py'
distributer_tool = tools / 'distribute.py'
data_dir = python_source / data
databases = data_dir / 'databases'
pdb_db = os.path.join(databases, 'pdbDB')
"""Local copy of the PDB database"""
qs_bio_dir = data_dir / 'qsbio'
qs_bio = qs_bio_dir / 'QSbioHighConfidenceAssemblies.pkl'
qs_bio_monomers_file = qs_bio_dir / 'QSbio_Monomers.txt'
# External dependencies
dependency_dir = python_source / 'dependencies'
binary_dir = dependency_dir / 'bin'
hbnet_sort = binary_dir / 'sort_hbnet_silent_file_results.sh'
bmdca_exe_path = dependency_dir / 'bmDCA' / 'src' / 'bmdca'
ialign_exe_path = dependency_dir / 'ialign' / 'bin' / 'ialign.pl'
# Alignment programs
uniclust_hhsuite_file_identifier = '_hhsuite.tar.gz'
hhsuite_dir = databases / 'hhsuite'
hhsuite_db_dir = hhsuite_dir / 'databases'
# alignment_db_path = dependency_dir / 'databases' / 'uniref90'

# Below matches internals of utils.read_json()
try:
    with config_file.open('r') as f_save:
        config = json.load(f_save)
except FileNotFoundError:  # May be running setup.py or this wasn't installed properly
    config = {'hhblits_env': '', 'hhblits_db': '',
              'rosetta_env': '', 'rosetta_make': 'default',
              'af_params': ''}
    logger.debug(f"Couldn't find the config file '{config_file}'. Setting default config:\n"
                 f"{', '.join(f'{k}={v}' for k, v in config.items())}")


def get_uniclust_db() -> str:
    """Get the newest UniClust file by sorting alphanumerically"""
    try:  # To get database files and remove any extra characters from filename
        md5sums = '.md5sums'
        matching_files = sorted(glob(str(hhsuite_db_dir / f'*{md5sums}')), reverse=True)
        return matching_files[0].replace(md5sums, '')
    except IndexError:
        return ''


hhblits_exe: str | None = os.environ.get(config.get('hhblits_env'), shutil.which(hhblits))
if hhblits_exe is None:
    # Provide a placeholder
    hhsuite_source_root_directory = Path("HHSUITE_DIR")
else:
    # Find the reformat script by backing out two directories. This is fine for conda and from source
    hhsuite_source_root_directory = Path(hhblits_exe).parent.parent

reformat_msa_exe_path = str(hhsuite_source_root_directory / 'scripts' / 'reformat.pl')
if hhsuite_db_dir.exists():
    uniclust_db = hhsuite_db_dir / config.get('uniclust_db', get_uniclust_db())
else:
    uniclust_db = Path('')
hhsuite_git = 'https://github.com/soedinglab/hh-suite'
# Alphafold
alphafold_db_dir = dependency_dir / 'alphafold'
alphafold_params_dir = alphafold_db_dir / 'params'
alphafold_params_name = config.get('af_params')
"""The version of the AlphaFold parameters downloaded"""
alphafold_git = 'https://github.com/deepmind/alphafold.git'
alphafold_install_instructions = f'Ensure that after {program_install}, you perform pip install git+{alphafold_git}'
# openmm_path = shutil.which('openmm')
symmetry_utils_path = utils_dir / 'symmetry.py'
# Help and warnings
documentation_url = 'https://kylemeador.github.io/symdesign/'
git_url = 'https://github.com/kylemeador/symdesign'
git_issue_url = 'https://github.com/kylemeador/symdesign/issues'
issue_submit_warning = f' If problems still persist, please submit an issue at {git_issue_url}'
symmetry_documentation_url = 'https://kylemeador.github.io/symdesign/#symmetry'
see_symmetry_documentation = f' See the documentation on symmetry: {symmetry_documentation_url}'
report_issue = f' Please report this at {git_issue_url}'


def ex_path(*directories: Sequence[str]) -> str:
    """Create an example path prepended with '/path/to/'

    Args:
        directories: Example: ('provided', 'directories')

    Returns:
        '/path/to/provided/directories'
    """
    return os.path.join('path', 'to', *directories)


def get_path_to_target_directory_from(
    search_path: AnyStr | Path, target_directory: os.PathLike | Path
) -> Path | None:
    """Search for the target_directory name in the specified search path and return the path to it

    Args:
        search_path: The path to search
        target_directory: The directory to search for

    Returns:
        The absolute path to the target_directory
    """
    search_path = Path(search_path).absolute()
    if search_path.exists():
        try:
            target_index = search_path.parts.index(str(target_directory))
            # if target_directory in search_path.parts:  # directory1/'target_directory'/directory2/directory3
        except ValueError:  # Couldn't find the `target_directory` in the path
            # See if 'target_directory' is a child of the provided search_path
            if search_path.is_dir() and target_directory in search_path.iterdir():  # directory_provided/'target_directory'
                directory = search_path / target_directory
            else:
                directory = None
        else:  # Return the path to that directory
            directory = Path(*search_path.parts[:target_index + 1])
    else:
        directory = None

    return directory


def collect_program_input_paths(
    files: Sequence[AnyStr] = None, directory: AnyStr = None, projects: Sequence[AnyStr] = None,
    singles: Sequence[AnyStr] = None
) -> tuple[list[Path], str]:
    """Grab all paths to program inputs from various input sources.

    Args:
        files: Iterable with disk location of files containing design directories
        directory: Disk location of the program directory
        projects: Disk location of a project directory
        singles: Disk location of a single design directory

    Raises:
        ValueError if no arguments were passed
        IsADirectoryError if a file was passed, but it is actually a directory

    Returns:
        All file/directory paths located and the location where they were located
    """
    if files:
        all_paths = []
        for path in map(Path, files):
            if not path.exists():
                logger.warning(f"No {path=} found. Please ensure correct location/name")
                continue

            if '.pdb' in path.suffix:  # Single .pdb file passed as input
                all_paths.append(path)
            elif '.cif' in path.suffix:  # Single .cif file passed as input
                all_paths.append(path)
            # Todo
            # elif '.fasta' in path.suffix:  # Single .fasta file passed as input
            #     all_paths.append(path)
            else:  # Assume a file that specifies other files was passed and load all paths in that file
                if path.is_dir():
                    raise IsADirectoryError(
                        f"'{path}' is a directory not a file. Did you mean to pass 'files'?'")

                with open(path, 'r') as f:
                    # Only strip the trailing 'os.sep' in case file names are passed
                    paths = [Path(loc.rstrip(os.sep)) for loc in f.read().splitlines() if loc != '']

                all_paths.extend(paths)
    else:
        if directory is not None:
            base_directory = get_path_to_target_directory_from(directory, program_output)
        else:
            base_directory = directory

        if base_directory or projects or singles:
            # Get all directory paths within:
            #  base directory -> /base/Projects/project1, ... /base/Projects/projectN
            #  specified projects -> /base/Projects/project1, /base/Projects/project2, ...
            #  specified singles -> /base/Projects/project/design1, /base/Projects/project/design2, ...
            all_paths = []
            if base_directory:  # base/Projects/*/*/
                all_paths.extend((p for p in base_directory.glob(f'{projects}/*/*') if p.is_dir()))
                # all_paths.extend(base_directory.glob(f'{putils.projects}/*/*/')))  # Todo Python3.11
            if projects:  # base/Projects/project/*/
                for project in map(Path, projects):
                    all_paths.extend((p for p in project.glob('*') if p.is_dir()))
                    # all_paths.extend(project.glob('*/'))  # Todo Python3.11
            if singles:  # base/Projects/project/single/
                for single in map(Path, singles):
                    all_paths.extend(single.glob('*'))
                    # all_paths.extend(glob(f'{single}{os.sep}'))

            all_paths = [p.absolute() for p in all_paths]
        elif directory:  # This is probably an uninitialized project. Grab all .pdb files
            directory_path = Path(directory)
            all_paths = list(directory_path.glob('*.pdb*')) + list(directory_path.glob('*.cif'))
            # Todo
            #     + list(directory_path.glob('*.fasta'))
            all_paths = sorted(all_paths, key=lambda p: p.stem)
            directory = directory_path.name  # This is for the location variable return
        else:  # Function was called with all set to None. This shouldn't happen
            raise ValueError(
                f"Can't {collect_program_input_paths.__name__}() with no arguments passed")

    location = (files or directory or projects or singles)

    return sorted(set(all_paths)), location
