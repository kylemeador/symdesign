from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from glob import glob
from pathlib import Path

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
refine = 'refine'
structure_background = 'structure_background'
hbnet_design_profile = 'hbnet_design_profile'
protocol = 'protocol'
design_parent = 'design_parent'
scout = 'scout'
consensus = 'consensus'
hhblits = 'hhblits'
rosetta = 'rosetta'
proteinmpnn = 'proteinmpnn'
reference = 'reference'
frag_dir = 'matching_fragments'
frag_text_file = 'frag_match_info_file.txt'
frag_file = os.path.join(frag_dir, frag_text_file)
pose_file = 'docked_pose_info_file.txt'
design_profile = 'design_profile'
evolutionary_profile = 'evolutionary_profile'
fragment_profile = 'fragment_profile'
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
# Alphfold
alphafold_db_dir = dependency_dir / 'alphafold'
alphafold_params_dir = alphafold_db_dir / 'params'
alphafold_params_name = config.get('af_params')
"""The version of the AlphaFold parameters downloaded"""
# openmm_path = shutil.which('openmm')
# Rosetta
rosetta_extras = config.get('rosetta_make')
rosetta_main = os.environ.get(config.get('rosetta_env'))
rosetta_main = rosetta_main if rosetta_main else 'main'  # Ensure not None
rosetta_source = Path(rosetta_main, 'source')
rosetta_default_bin_path = rosetta_source / 'bin'
make_sdf_path = rosetta_source / 'src' / 'apps' / 'public' / 'symmetry' / 'make_symmdef_file.pl'
# Rosetta Scripts and Misc Files'
rosetta_scripts_dir = os.path.join(dependency_dir, 'rosetta')
symmetry_def_files = os.path.join(rosetta_scripts_dir, 'sdf')
scout_symmdef = os.path.join(symmetry_def_files, 'scout_symmdef_file.pl')
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
