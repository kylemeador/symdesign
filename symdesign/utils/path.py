from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from glob import glob
from typing import AnyStr, Sequence

logger = logging.getLogger(__name__)
# Project strings and file names
utils_dir = os.path.dirname(os.path.realpath(__file__))  # Reveals utils subdirectory
python_source = os.path.dirname(utils_dir)  # Reveals the root python code directory
git_source = os.path.dirname(python_source)  # Reveals the root git directory
try:
    p = subprocess.Popen(['git', '--git-dir', os.path.join(git_source, '.git'), 'rev-parse', 'HEAD'],
                         stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    commit = stdout.decode().strip()
    commit_short = commit[:9]
except subprocess.CalledProcessError:
    commit = commit_short = 'unknown'

program_name = 'SymDesign'
program_exe = git_source  # Where __main__.py is located
conda_environment = os.path.join(git_source, 'conda_env.yml')
logging_cfg_file = os.path.join(python_source, 'logging.cfg')
config_file = os.path.join(python_source, 'cfg.json')
third_party_dir = os.path.join(python_source, 'third_party')
program_command = f'{sys.executable} {program_exe}'
program_command_tuple = (sys.executable, program_exe)
program_help = f'{program_command} --help'
submodule_guide = f'{program_command} MODULE --guide'
submodule_help = f'{program_command} MODULE --help'
guide_string = f'{program_name} guide. Enter "{program_command} --guide"'
program_output = f'{program_name}Output'
projects = 'Projects'
data = 'data'
pose_metrics_file = 'pose_scores.sc'  # UNUSED
# default_path_file = '{}_{}_{}_pose.paths'
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
alphafold = 'alphafold'
setup_exe = os.path.join(git_source, 'setup.py')
hhsuite_setup_command = f'python {setup_exe} --hhsuite-database'
rosetta = 'rosetta'
proteinmpnn = 'proteinmpnn'
protein_mpnn_dir = 'ProteinMPNN'
protein_mpnn_ca_weights_dir = os.path.join(third_party_dir, protein_mpnn_dir, 'ca_model_weights')
protein_mpnn_weights_dir = os.path.join(third_party_dir, protein_mpnn_dir, 'vanilla_model_weights')
master_log = 'nanohedra_master_logfile.txt'  # v1
reference_name = 'reference'
frag_dir = 'matching_fragments'  # was 'matching_fragments_representatives' in v0
frag_text_file = 'frag_match_info_file.txt'
frag_file = os.path.join(frag_dir, frag_text_file)
pose_file = 'docked_pose_info_file.txt'
design_profile = 'design_profile'
evolutionary_profile = 'evolutionary_profile'
fragment_profile = 'fragment_profile'
# Memory Requirements
# 10.5MB was measured 5/13/22 with self.pose,
# after deleting self.pose: 450 poses -> 6.585339. 900 poses -> 3.346759
approx_ave_design_directory_memory_w_pose = 4000000  # 4MB
approx_ave_design_directory_memory_w_o_pose = 150000  # with 44279 poses get 147299.66 bytes/pose
approx_ave_design_directory_memory_w_assembly = 500000000  # 500MB
baseline_program_memory = 3000000000  # 3GB
nanohedra_memory = 30000000000  # 30Gb

# Project paths
documentation_dir = os.path.join(git_source, 'docs')
readme_file = os.path.join(documentation_dir, 'README.md')
# dependency_dir = os.path.join(source, 'dependencies')
dependency_dir = os.path.join(git_source, 'dependencies')
tools = os.path.join(python_source, 'tools')
sym_op_location = os.path.join(dependency_dir, 'symmetry_operators')
sasa_debug_dir = os.path.join(dependency_dir, 'sasa')
point_group_symmetry_operator_location = os.path.join(sym_op_location, 'point_group_operators.pkl')
point_group_symmetry_operatorT_location = os.path.join(sym_op_location, 'point_group_operators_transposed.pkl')
space_group_symmetry_operator_location = os.path.join(sym_op_location, 'space_group_operators.pkl')
space_group_symmetry_operatorT_location = os.path.join(sym_op_location, 'space_group_operators_transposed.pkl')
protocols_dir = os.path.join(python_source, f'protocols')
binaries = os.path.join(dependency_dir, 'bin')
models_to_multimodel_exe = os.path.join(tools, 'models_to_multimodel.py')
list_pdb_files = os.path.join(tools, 'list_files_in_directory.py')
distributer_tool = os.path.join(tools, 'distribute.py')
hbnet_sort = os.path.join(binaries, 'sort_hbnet_silent_file_results.sh')
data_dir = os.path.join(python_source, data)
pickle_program_requirements = os.path.join(data_dir, 'pickle_structure_dependencies.py')
pickle_program_requirements_cmd = f'python {pickle_program_requirements}'
binary_lookup_table_path = os.path.join(dependency_dir, 'euler_lookup', 'euler_lookup_40.npz')
reference_aa_file = os.path.join(data_dir, 'AAreference.pdb')
# reference_aa_pickle = os.path.join(data_dir, 'AAreference.pkl')
reference_residues_pkl = os.path.join(data_dir, 'AAreferenceResidues.pkl')
uniprot_pdb_map = os.path.join(data_dir, '200121_UniProtPDBMasterDict.pkl')
# filter_and_sort = os.path.join(data_dir, 'filter_and_sort_df.csv')
affinity_tags = os.path.join(data_dir, 'affinity-tags.csv')
database = os.path.join(data_dir, 'databases')
pdb_db = os.path.join(database, 'pdbDB')
"""Local copy of the PDB database"""
pisa_db = os.path.join(database, 'pisaDB')
"""Local copy of the PISA database"""
# Todo
#  qsbio = os.path.join(data_dir, 'QSbioAssemblies.pkl')  # 200121_QSbio_GreaterThanHigh_Assemblies.pkl
# qs_bio = os.path.join(data_dir, 'QSbio_GreaterThanHigh_Assemblies.pkl')
qs_bio = os.path.join(data_dir, 'QSbioHighConfidenceAssemblies.pkl')
qs_bio_monomers_file = os.path.join(data_dir, 'QSbio_Monomers.csv')
# Fragment Database
fragment_db = os.path.join(database, 'fragment_db')
biological_interfaces = 'biological_interfaces'
bio = 'bio'
xtal = 'xtal'
bio_xtal = 'bio_xtal'
fragment_dbs = [biological_interfaces,
                # bio, xtal, bio_xtal
                ]
biological_fragment_db = os.path.join(fragment_db, biological_interfaces)  # TODO change this directory style
biological_fragment_db_pickle = os.path.join(fragment_db, f'{biological_interfaces}.pkl')
bio_fragment_db = os.path.join(fragment_db, bio)
xtal_fragment_db = os.path.join(fragment_db, xtal)
full_fragment_db = os.path.join(fragment_db, bio+xtal)
frag_directory = {biological_interfaces: biological_fragment_db, bio: bio_fragment_db, xtal: xtal_fragment_db,
                  bio+xtal: full_fragment_db}
# Nanohedra Specific
monofrag_cluster_rep_dirpath = os.path.join(fragment_db, 'Top5MonoFragClustersRepresentativeCentered')
intfrag_cluster_rep_dirpath = os.path.join(fragment_db, 'Top75percent_IJK_ClusterRepresentatives_1A')
intfrag_cluster_info_dirpath = os.path.join(fragment_db, 'IJK_ClusteredInterfaceFragmentDBInfo_1A')

# External Program Dependencies
# Free SASA Executable Path
freesasa_dir = os.path.join(third_party_dir, 'freesasa')
freesasa_exe_path = os.path.join(third_party_dir, 'freesasa', 'src', 'freesasa')
freesasa_config_path = os.path.join(dependency_dir, 'freesasa-2.0.config')

orient_exe_dir = os.path.join(dependency_dir, 'orient')
# orient_exe = 'orient_oligomer.f'  # Non_compiled
orient_exe = 'orient_oligomer'
orient_exe_path = os.path.join(orient_exe_dir, orient_exe)
orient_log_file = 'orient_oligomer_log.txt'
errat_exe_path = os.path.join(dependency_dir, 'errat', 'errat')
errat_residue_source = os.path.join(dependency_dir, 'errat', 'errat_every_residue.cpp')
stride_dir = os.path.join(dependency_dir, 'stride')
stride_exe_path = os.path.join(dependency_dir, 'stride', 'stride')
bmdca_exe_path = os.path.join(dependency_dir, 'bmDCA', 'src', 'bmdca')
ialign_exe_path = os.path.join(dependency_dir, 'ialign', 'bin', 'ialign.pl')
# Set up for alignment programs
uniclust_hhsuite_file_identifier = '_hhsuite.tar.gz'
hhsuite_dir = os.path.join(dependency_dir, 'hhsuite')
hhsuite_db_dir = os.path.join(hhsuite_dir, 'databases')
alignmentdb = os.path.join(dependency_dir, 'ncbi_databases', 'uniref90')
# alignment_db = os.path.join(dependency_dir, 'databases/uniref90')  # TODO

# Below matches internals of utils.read_json()
try:
    with open(config_file, 'r') as f_save:
        config = json.load(f_save)
except FileNotFoundError:  # May be running setup.py or this wasn't installed properly
    config = {'hhblits_env': '', 'hhblits_db': '',
              'rosetta_env': '', 'rosetta_make': 'default',
              'af_params': ''}
    logger.debug(f"Couldn't find the config file '{config_file}'. Setting default config:\n"
                 f"{', '.join(f'{k}={v}' for k, v in config.items())}")
    # pass


# def get_hhblits_exe():
#     # hhblits_exe_out = shutil.which(hhblits)
#     return shutil.which(hhblits)
#     # p = subprocess.Popen(['which', 'hhblits'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#     # hhblits_exe_out, err = p.communicate()
#     # return hhblits_exe_out.decode('utf-8').strip()


def get_uniclust_db() -> str:
    """Get the newest UniClust file by sorting alphanumerically"""
    try:  # To get database files and remove any extra characters from filename
        md5sums = '.md5sums'
        return sorted(glob(os.path.join(hhsuite_db_dir, f'*{md5sums}')), reverse=True)[0].replace(md5sums, '')
        # return sorted(os.listdir(hhsuite_db_dir), reverse=True)[0].replace(uniclust_hhsuite_file_identifier, '')
    except IndexError:
        return ''


hhblits_exe = os.environ.get(config.get('hhblits_env'), shutil.which(hhblits))
# Find the reformat script by backing out two directories. This is fine for conda and from source
hhsuite_source_root_directory = os.path.dirname(os.path.dirname(hhblits_exe))
# reformat_msa_exe_path = os.path.join(hhsuite_dir, 'scripts', 'reformat.pl')
reformat_msa_exe_path = os.path.join(hhsuite_source_root_directory, 'scripts', 'reformat.pl')
# hhblits_exe = hhblits_exe if hhblits_exe else 'hhblits'  # ensure not None
# uniclustdb = os.path.join(dependency_dir, 'hh-suite', 'databases', 'UniRef30_2020_02')
if os.path.exists(hhsuite_db_dir):
    uniclust_db = os.path.join(hhsuite_db_dir, config.get('uniclust_db', get_uniclust_db()))
else:
    uniclust_db = ''
install_hhsuite_exe = os.path.join(binaries, 'install_hhsuite.sh')
hhsuite_git = 'https://github.com/soedinglab/hh-suite'
# Alphfold
alphafold_db_dir = os.path.join(dependency_dir, alphafold)
alphafold_params_dir = os.path.join(alphafold_db_dir, 'params')
alphafold_params_name = config.get('af_params')
alphafold_source_dir = os.path.join(third_party_dir, alphafold)
alphafold_common_dir = os.path.join(alphafold_source_dir, alphafold, 'common')
alphafold_openmm_patch = os.path.join(alphafold_source_dir, 'docker', 'openmm.patch')
openmm_path = shutil.which('openmm')
# Rosetta
rosetta_extras = config.get('rosetta_make')
rosetta_main = os.environ.get(config.get('rosetta_env'))
rosetta_main = rosetta_main if rosetta_main else 'main'  # Ensure not None
rosetta_source = os.path.join(rosetta_main, 'source')
rosetta_default_bin = os.path.join(rosetta_source, 'bin')
make_symmdef = os.path.join(rosetta_source, 'src', 'apps', 'public', 'symmetry', 'make_symmdef_file.pl')
# Rosetta Scripts and Misc Files'
rosetta_scripts_dir = os.path.join(dependency_dir, 'rosetta')
symmetry_def_file_dir = 'rosetta_symmetry_definition_files'
symmetry_def_files = os.path.join(rosetta_scripts_dir, 'sdf')
scout_symmdef = os.path.join(symmetry_def_files, 'scout_symmdef_file.pl')

sym_utils_file = 'symmetry.py'
path_to_sym_utils = os.path.join(os.path.dirname(__file__), sym_utils_file)
# help and warnings
git_url = 'https://github.com/kylemeador/symdesign'
git_issue_url = 'https://github.com/kylemeador/symdesign/issues'
issue_submit_warning = f' If problems still persist, please submit an issue at {git_issue_url}'
symmetry_docs_url = 'https://github.com/kylemeador/symdesign/tree/main#symmetry'
see_symmetry_documentation = f' See the documentation on symmetry: {symmetry_docs_url}'
report_issue = f' Please report this at {git_issue_url}'

logging_cfg = {
    'version': 1,
    'formatters': {
        'standard': {
            'class': 'logging.Formatter',
            'format': '\033[38;5;93m{name}\033[0;0m-\033[38;5;208m{levelname}\033[0;0m: {message}',
            'style': '{'
        },
        'file_standard': {
            'class': 'logging.Formatter',
            'format': '{name}-{levelname}: {message}',
            'style': '{'
        },
        'none': {
            'class': 'logging.Formatter',
            'format': '{message}',
            'style': '{'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': sys.stdout,
        },
        'main_file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'mode': 'a',
            'formatter': 'file_standard',
            'filename': f'{program_name.upper()}.log',
        },
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'loggers': {
        program_name.lower(): {
            'level': 'INFO',  # 'WARNING',
            'handlers': ['console'],  # , 'main_file'],
            'propagate': 'no'
        },
        'orient': {
            'level': 'INFO',  # 'WARNING',
            'handlers': ['console'],  # , 'main_file'],
            'propagate': 'no'
        },
        'null': {
            'level': 'WARNING',
            'handlers': ['null'],
            'propagate': 'no'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['null'],
        # Can't include any stream or file handlers from above as the handlers get added to configuration twice
    },
}
default_logging_level = 20


def make_path(path: AnyStr, condition: bool = True):
    """Make all required directories in specified path if it doesn't exist, and optional condition is True

    Args:
        path: The path to create
        condition: A condition to check before the path production is executed
    """
    if condition:
        os.makedirs(path, exist_ok=True)


def ex_path(*directories: Sequence[str]) -> AnyStr:
    """Create an example path prepended with /path/to/provided/directories

    Args:
        directories: Example: ('provided', 'directories')
    """
    return os.path.join('path', 'to', *directories)
