from __future__ import annotations

import json
import os
import subprocess

# Project strings and file names
utils_dir = os.path.dirname(os.path.realpath(__file__))  # reveals utils subdirectory
python_source = os.path.dirname(utils_dir)  # reveals master symdesign directory
git_source = os.path.dirname(python_source)  # reveals master symdesign directory
try:
    p = subprocess.Popen(['git', '--git-dir', os.path.join(git_source, '.git'), 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    program_version = stdout.decode().strip()
except subprocess.CalledProcessError:
    program_version = 'unknown'

program_name = 'SymDesign'
program_exe = os.path.join(python_source, f'{program_name}.py')
logging_cfg_file = os.path.join(python_source, 'logging.cfg')
config_file = os.path.join(python_source, 'cfg.json')
third_party_dir = os.path.join(python_source, 'third_party')
program_output = f'{program_name}Output'
projects = 'Projects'
program_command = f'python {program_exe}'
submodule_guide = f'{program_command} MODULE --guide'
submodule_help = f'{program_command} MODULE --help'
guide_string = f'{program_name} guide. Enter "{program_command} --guide"'
sym_entry = 'sym_entry'
output_fragments = 'output_fragments'
output_oligomers = 'output_oligomers'
output_structures = 'output_structures'
output_trajectory = 'output_trajectory'
nanohedra = 'nanohedra'
nano_publication = 'Laniado, Meador, & Yeates, PEDS. 2021'
nano_entity_flag1 = 'oligomer1'  # Todo make entity1
nano_entity_flag2 = 'oligomer2'
options = 'options'
skip_logging = 'skip_logging'
multi_processing = 'multi_processing'
distribute_work = 'distribute_work'
output_directory = 'output_directory'
output_file = 'output_file'
output_assembly = 'output_assembly'
output_surrounding_uc = 'output_surrounding_uc'
input_ = 'input'
output = 'output'
orient = 'orient'
refine = 'refine'
residue_selector = 'residue_selector'
multicistronic = 'multicistronic'
interface_design = 'interface_design'
interface_metrics = 'interface_metrics'
optimize_designs = 'optimize_designs'
generate_fragments = 'generate_fragments'
check_clashes = 'check_clashes'
rename_chains = 'rename_chains'
expand_asu = 'expand_asu'
analysis = 'analysis'
cluster_poses = 'cluster_poses'
select_poses = 'select_poses'
select_designs = 'select_designs'
select_sequences = 'select_sequences'
evolution_constraint = 'evolution_constraint'
term_constraint = 'term_constraint'
structure_background = 'structure_background'
hbnet_design_profile = 'hbnet_design_profile'
ca_only = 'ca_only'
development = 'development'
ignore_clashes = 'ignore_clashes'
ignore_pose_clashes = 'ignore_pose_clashes'
ignore_symmetric_clashes = 'ignore_symmetric_clashes'
sequences = 'sequences'
structures = 'structures'
temperatures = 'temperatures'
protocol = 'protocol'
groups = protocol
number_of_trajectories = 'number_of_trajectories'
force_flags = 'force_flags'
current_energy_function = 'REF2015'
hbnet = 'hbnet'
scout = 'scout'
consensus = 'consensus'
hhblits = 'hhblits'
rosetta_str = 'rosetta'
proteinmpnn = 'proteinmpnn'
protein_mpnn_dir = 'ProteinMPNN'
protein_mpnn_weights_dir = os.path.join(third_party_dir, protein_mpnn_dir, 'vanilla_model_weights')

temp = 'temp.hold'
pose_prefix = 'tx_'
# master_log = 'master_log.txt'  # v0
master_log = 'nanohedra_master_logfile.txt'  # v1
asu_file_name = 'asu.pdb'
# asu = 'central_asu.pdb'
clean_asu = 'clean_asu.pdb'
state_file = 'info.pkl'
reference_name = 'reference'
pose_source = 'pose_source'
pssm = 'evolutionary.pssm'  # was 'asu_pose.pssm' 1/25/21
fssm = 'fragment.pssm'
dssm = 'design.pssm'
assembly = 'assembly.pdb'
docked_pose_file = 'docked_pose_info_file.txt'
frag_dir = 'matching_fragments'  # was 'matching_fragments_representatives' in v0
frag_text_file = 'frag_match_info_file.txt'
frag_file = os.path.join(frag_dir, frag_text_file)
pose_file = 'docked_pose_info_file.txt'
design_profile = 'design_profile'
evolutionary_profile = 'evolutionary_profile'
fragment_profile = 'fragment_profile'
sequence_info = 'SequenceInfo'  # was Sequence_Info 1/25/21
structure_info = 'StructureInfo'
# profiles = 'profiles'
pose_directory = 'Poses'

data = 'data'
designs = 'designs'  # was rosetta_pdbs/ 1/25/21
all_scores = 'AllScores'
scores_outdir = 'scores'
scripts = 'scripts'
scores_file = 'design_scores.sc'  # was all_scores.sc 1/25/21
pose_metrics_file = 'pose_scores.sc'  # UNUSED
default_path_file = '%s_%s_%s_pose.paths'
default_analysis_file = '%s-%sPoseMetrics.csv'
default_clustered_pose_file = '%sClusteredPoses-%s.pkl'
pdb_source = 'db'  # 'fetch_pdb_file'  # TODO set up
# nanohedra_directory_structure = './design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C\n' \
#                                 'Ex:P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2'\
#                                 '\nIn design directory \'tx_c/\', output is located in \'%s\' and \'%s\'.' \
#                                 '\nTotal design_symmetry_pg score are located in ./design_symmetry_pg/building_blocks/%s' \
#                                 % (designs, scores_outdir, scores_outdir)
# Memory Requirements
# 10.5MB was measured 5/13/22 with self.pose,
# after deleting self.pose: 450 poses -> 6.585339. 900 poses -> 3.346759
approx_ave_design_directory_memory_w_pose = 4000000  # 4MB
approx_ave_design_directory_memory_w_o_pose = 150000  # with 44279 poses get 147299.66 bytes/pose
approx_ave_design_directory_memory_w_assembly = 500000000  # 500MB
baseline_program_memory = 3000000000  # 3GB
nanohedra_memory = 30000000000  # 30Gb

# Project paths
readme = os.path.join(git_source, 'README.md')
# dependency_dir = os.path.join(source, 'dependencies')
dependency_dir = os.path.join(git_source, 'dependencies')
tools = os.path.join(python_source, 'tools')
sym_op_location = os.path.join(dependency_dir, 'symmetry_operators')
sasa_debug_dir = os.path.join(dependency_dir, 'sasa')
point_group_symmetry_operator_location = os.path.join(sym_op_location, 'point_group_operators.pkl')
space_group_symmetry_operator_location = os.path.join(sym_op_location, 'space_group_operators.pkl')
nanohedra_main = os.path.join(python_source, f'{nanohedra.title()}.py')
nanohedra_dock_file = os.path.join(python_source, 'fragdock.py')
binaries = os.path.join(dependency_dir, 'bin')
models_to_multimodel_exe = os.path.join(tools, 'models_to_multimodel.py')
list_pdb_files = os.path.join(tools, 'list_files_in_directory.py')
command_distributer = os.path.join(utils_dir, 'CommandDistributer.py')
hbnet_sort = os.path.join(binaries, 'sort_hbnet_silent_file_results.sh')
sbatch_template_dir = os.path.join(dependency_dir, 'sbatch')
disbatch = os.path.join(binaries, 'diSbatch.sh')  # DEPRECIATED
data_dir = os.path.join(python_source, data)
binary_lookup_table_path = os.path.join(dependency_dir, 'euler_lookup', 'euler_lookup_40.npz')
reference_aa_file = os.path.join(data_dir, 'AAreference.pdb')
# reference_aa_pickle = os.path.join(data_dir, 'AAreference.pkl')
reference_residues_pkl = os.path.join(data_dir, 'AAreferenceResidues.pkl')
uniprot_pdb_map = os.path.join(data_dir, '200121_UniProtPDBMasterDict.pkl')
# filter_and_sort = os.path.join(data_dir, 'filter_and_sort_df.csv')
pdb_uniprot_map = os.path.join(data_dir, 'pdb_uniprot_map')  # TODO
# uniprot_pdb_map = os.path.join(data_dir, 'uniprot_pdb_map')  # TODO
affinity_tags = os.path.join(data_dir, 'modified-affinity-tags.csv')
database = os.path.join(data_dir, 'databases')
pdb_db = os.path.join(database, 'pdbDB')  # pointer to pdb database
pisa_db = os.path.join(database, 'pisaDB')  # pointer to pisa database
# Todo
#  qsbio = os.path.join(data_dir, 'QSbioAssemblies.pkl')  # 200121_QSbio_GreaterThanHigh_Assemblies.pkl
# qs_bio = os.path.join(data_dir, 'QSbio_GreaterThanHigh_Assemblies.pkl')
qs_bio = os.path.join(data_dir, 'QSbioHighConfidenceAssemblies.pkl')
qs_bio_monomers_file = os.path.join(data_dir, 'QSbio_Monomers.csv')

# TODO script this file creation ?
#  qsbio_data_url = 'https://www.weizmann.ac.il/sb/faculty_pages/ELevy/downloads/QSbio.xlsx'
#  response = requests.get(qsbio_data_url)
#  with open(qsbio_file, 'wb') as output
#      output.write(response.content)
#  qsbio_df = pd.DataFrame(qsbio_file)
#  or
#  qsbio_df = pd.DataFrame(response.content)
#  qsbio_df.groupby('QSBio Confidence', inplace=True)
#  greater_than_high_df = qsbio_df[qsbio_df['QSBio Confidence'] in ['Very high', 'High']
#  oligomeric_assemblies = greater_than_high_df.drop(qsbio_monomers, errors='ignore')
#  for pdb_code in qs_bio:
#      qs_bio[pdb_code] = set(int(ass_id) for ass_id in qs_bio[pdb_code])

# Fragment Database
fragment_db = os.path.join(database, 'fragment_db')
# fragment_db = os.path.join(database, 'fragment_DB')  # TODO when full MySQL DB is operational
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
freesasa_exe_path = os.path.join(third_party_dir, 'freesasa', 'src', 'freesasa')
freesasa_config_path = os.path.join(dependency_dir, 'freesasa-2.0.config')

orient_dir = os.path.join(dependency_dir, 'orient')
# orient_exe = 'orient_oligomer.f'  # Non_compiled
orient_exe = 'orient_oligomer'
orient_exe_path = os.path.join(orient_dir, orient_exe)
orient_log_file = 'orient_oligomer_log.txt'
errat_exe_path = os.path.join(dependency_dir, 'errat', 'errat')
stride_exe_path = os.path.join(dependency_dir, 'stride', 'stride')
bmdca_exe_path = os.path.join(dependency_dir, 'bmDCA', 'src', 'bmdca')
ialign_exe_path = os.path.join(dependency_dir, 'ialign', 'bin', 'ialign.pl')
# Set up for alignment programs
reformat_msa_exe_path = os.path.join(dependency_dir, 'hh-suite', 'scripts', 'reformat.pl')
alignmentdb = os.path.join(dependency_dir, 'ncbi_databases', 'uniref90')
# alignment_db = os.path.join(dependency_dir, 'databases/uniref90')  # TODO

# Below matches internals of utils.read_json()
with open(config_file, 'r') as f_save:
    config = json.load(f_save)

hhblits_exe = os.environ.get(config.get('hhblits_env'))
hhblits_exe = hhblits_exe if hhblits_exe else 'hhblits'  # ensure not None
uniclustdb = os.path.join(dependency_dir, 'hh-suite', 'databases', 'UniRef30_2020_02')  # TODO make db dynamic at config
# uniclust_db = os.path.join(database, 'hh-suite', 'databases', 'UniRef30_2020_02')  # TODO
install_hhsuite = os.path.join(binaries, 'install_hhsuite.sh')
# Rosetta
rosetta_extras = config.get('rosetta_make')
rosetta_main = os.environ.get(config.get('rosetta_env'))
rosetta_main = rosetta_main if rosetta_main else 'main'  # ensure not None
make_symmdef = os.path.join(rosetta_main, 'source', 'src', 'apps', 'public', 'symmetry', 'make_symmdef_file.pl')
# Todo v dependent on external compile. cd to the directory, then type "make" to compile the executable
dalphaball = os.path.join(rosetta_main, 'source', 'external', 'DAlpahBall', 'DAlphaBall.gcc')
# Rosetta Scripts and Misc Files'
rosetta_scripts_dir = os.path.join(dependency_dir, 'rosetta')
symmetry_def_file_dir = 'rosetta_symmetry_definition_files'
symmetry_def_files = os.path.join(rosetta_scripts_dir, 'sdf')
sym_weights = 'ref2015_sym.wts_patch'
solvent_weights = 'ref2015_solvent.wts_patch'
solvent_weights_sym = 'ref2015_sym_solvent.wts_patch'
scout_symmdef = os.path.join(symmetry_def_files, 'scout_symmdef_file.pl')

sym_utils_file = 'symmetry.py'
path_to_sym_utils = os.path.join(os.path.dirname(__file__), sym_utils_file)
# help and warnings
git_issue_url = 'https://github.com/kylemeador/symdesign/issues'
issue_submit_warning = f'If problems still persist please submit an issue {git_issue_url}'

# Todo place this is a config file or something similar
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
        },
        'main_file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'mode': 'a',
            'formatter': 'file_standard',
            'filename': f'{program_name.upper()}.log',
        },
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'loggers': {
        'symdesign': {
            'level': 'INFO',  # 'WARNING',
            'handlers': ['console', 'main_file'],
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
        # 'handlers': ['console'],  # Can't include this and any above as the handlers get added twice
    },
}
