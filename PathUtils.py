from __future__ import annotations

from os import path, environ

# Project strings and file names
program_name = 'SymDesign'
program_exe = path.join(path.dirname(__file__), f'{program_name}.py')
program_output = f'{program_name}Output'
projects = 'Projects'
program_command = f'python {program_exe}'
submodule_guide = f'{program_command} MODULE --guide'
submodule_help = f'{program_command} MODULE --help'
guide_string = f'{program_name} guide. Enter "{program_command} --guide"'
sym_entry = 'sym_entry'
output_oligomers = 'output_oligomers'
output_fragments = 'output_fragments'
nano = 'nanohedra'
nano_entity_flag1 = '-oligomer1'  # Todo make -entity1
nano_entity_flag2 = '-oligomer2'
refine = 'refine'
interface_design = 'interface_design'
interface_metrics = 'interface_metrics'
optimize_designs = 'optimize_designs'
generate_fragments = 'generate_fragments'
analysis = 'analysis'
cluster_poses = 'cluster_poses'
select_poses = 'select_poses'
select_designs = 'select_designs'
select_sequences = 'select_sequences'
no_evolution_constraint = 'no_evolution_constraint'
no_term_constraint = 'no_term_constraint'
structure_background = 'structure_background'
hbnet_design_profile = 'hbnet_design_profile'
ignore_clashes = 'ignore_clashes'
ignore_pose_clashes = 'ignore_pose_clashes'
ignore_symmetric_clashes = 'ignore_symmetric_clashes'
protocol = 'protocol'
groups = protocol
number_of_trajectories = 'number_of_trajectories'
force_flags = 'force_flags'
current_energy_function = 'REF2015'
no_hbnet = 'no_hbnet'
scout = 'scout'
consensus = 'consensus'
hhblits = 'hhblits'
rosetta_str = 'rosetta'
string_ops = [str.upper, str.lower, str.title]


def search_env_for_variable(search_variable: str) -> str | None:
    env_variable = None
    i = 0
    search_strings = []
    try:
        string_op_it = iter(string_ops)
        while not env_variable:
            string = next(string_op_it)(search_variable)
            search_strings.append(string)
            env_variable = environ.get(string)
    except StopIteration:
        pass

    if not env_variable:
        print(f'No environmental variable specifying {search_variable} software location at {", ".join(search_strings)}'
              f'. {search_variable} inaccessible.')

    return env_variable


nstruct = 20
stage = {1: refine, 2: interface_design, 3: 'metrics', 4: 'analysis', 5: consensus,
         6: 'rmsd_calculation', 7: 'all_to_all', 8: 'rmsd_clustering', 9: 'rmsd_to_cluster', 10: 'rmsd',
         11: 'all_to_cluster', 12: scout, 13: hbnet_design_profile, 14: structure_background}
stage_f = {refine: {'path': '*_refine.pdb', 'len': 1}, interface_design: {'path': '*_design_*.pdb', 'len': nstruct},
           stage[3]: {'path': '', 'len': None}, stage[4]: {'path': '', 'len': None},
           consensus: {'path': '*_consensus.pdb', 'len': 1}, nano: {'path': '', 'len': None},
           stage[6]: {'path': '', 'len': None}, stage[7]: {'path': '', 'len': None},
           stage[8]: {'path': '', 'len': None}, stage[9]: {'path': '', 'len': None},
           stage[10]: {'path': '', 'len': None}, stage[11]: {'path': '', 'len': None}}
rosetta_extras = 'mpi'  # 'cxx11threadmpi' TODO make dynamic at config
temp = 'temp.hold'
pose_prefix = 'tx_'
# master_log = 'master_log.txt'  # v0
master_log = 'nanohedra_master_logfile.txt'  # v1
asu = 'asu.pdb'
# asu = 'central_asu.pdb'
clean_asu = 'clean_asu.pdb'
reference_name = 'reference'
pssm = 'evolutionary.pssm'  # was 'asu_pose.pssm' 1/25/21
fssm = 'fragment.pssm'
dssm = 'design.pssm'
assembly = 'assembly.pdb'
frag_dir = 'matching_fragments'  # was 'matching_fragments_representatives' in v0
frag_text_file = 'frag_match_info_file.txt'
docked_pose_file = 'docked_pose_info_file.txt'
frag_file = path.join(frag_dir, frag_text_file)
pose_file = 'docked_pose_info_file.txt'
design_profile = 'design_profile'
evolutionary_profile = 'evolutionary_profile'
fragment_profile = 'fragment_profile'
protein_data = 'ProteinData'
sequence_info = 'SequenceInfo'  # was Sequence_Info 1/25/21
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
analysis_file = '%s-%sPoseMetrics.csv'
clustered_poses = '%sClusteredPoses-%s.pkl'
pdb_source = 'db'  # 'fetch_pdb'  # TODO set up
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
source = path.dirname(path.realpath(__file__))  # reveals master symdesign folder
readme = path.join(source, 'README.md')
command = path.join(source, 'SymDesign.py')
cmd_dist = path.join(source, 'CommandDistributer.py')
dependency_dir = path.join(source, 'dependencies')
sym_op_location = path.join(dependency_dir, 'symmetry_operators')
point_group_symmetry_operator_location = path.join(sym_op_location, 'point_group_operators.pkl')
space_group_symmetry_operator_location = path.join(sym_op_location, 'space_group_operators.pkl')
nanohedra_main = path.join(source, f'{nano.title()}.py')
nanohedra_dock_file = path.join(source, 'FragDock.py')
binaries = path.join(dependency_dir, 'bin')
models_to_multimodel_exe = path.join(binaries, 'models_to_multimodel.py')
list_pdb_files = path.join(binaries, 'list_files_in_directory.py')
hbnet_sort = path.join(binaries, 'sort_hbnet_silent_file_results.sh')
sbatch_template_dir = path.join(dependency_dir, 'sbatch')
disbatch = path.join(binaries, 'diSbatch.sh')  # DEPRECIATED
data_dir = path.join(source, data)
binary_lookup_table_path = path.join(dependency_dir, 'euler_lookup', 'euler_lookup_40.npz')
reference_aa_file = path.join(data_dir, 'AAreference.pdb')
# reference_aa_pickle = path.join(data_dir, 'AAreference.pkl')
reference_residues_pkl = path.join(data_dir, 'AAreferenceResidues.pkl')
uniprot_pdb_map = path.join(data_dir, '200121_UniProtPDBMasterDict.pkl')
# filter_and_sort = path.join(data_dir, 'filter_and_sort_df.csv')
pdb_uniprot_map = path.join(data_dir, 'pdb_uniprot_map')  # TODO
# uniprot_pdb_map = path.join(data_dir, 'uniprot_pdb_map')  # TODO
affinity_tags = path.join(data_dir, 'modified-affinity-tags.csv')
database = path.join(data_dir, 'databases')
pdb_db = path.join(database, 'pdbDB')  # pointer to pdb database
pisa_db = path.join(database, 'pisaDB')  # pointer to pisa database
# Todo
#  qsbio = path.join(data_dir, 'QSbioAssemblies.pkl')  # 200121_QSbio_GreaterThanHigh_Assemblies.pkl
# qs_bio = path.join(data_dir, 'QSbio_GreaterThanHigh_Assemblies.pkl')
qs_bio = path.join(data_dir, 'QSbioHighConfidenceAssemblies.pkl')
qs_bio_monomers_file = path.join(data_dir, 'QSbio_Monomers.csv')

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
fragment_db = path.join(database, 'fragment_db')
# fragment_db = path.join(database, 'fragment_DB')  # TODO when full MySQL DB is operational
biological_interfaces = 'biological_interfaces'
bio = 'bio'
xtal = 'xtal'
bio_xtal = 'bio_xtal'
fragment_dbs = [biological_interfaces,
                # bio, xtal, bio_xtal
                ]
biological_fragment_db = path.join(fragment_db, biological_interfaces)  # TODO change this directory style
biological_fragment_db_pickle = path.join(fragment_db, f'{biological_interfaces}.pkl')
bio_fragment_db = path.join(fragment_db, bio)
xtal_fragment_db = path.join(fragment_db, xtal)
full_fragment_db = path.join(fragment_db, bio+xtal)
frag_directory = {biological_interfaces: biological_fragment_db, bio: bio_fragment_db, xtal: xtal_fragment_db,
                  bio+xtal: full_fragment_db}
# Nanohedra Specific
monofrag_cluster_rep_dirpath = path.join(fragment_db, 'Top5MonoFragClustersRepresentativeCentered')
intfrag_cluster_rep_dirpath = path.join(fragment_db, 'Top75percent_IJK_ClusterRepresentatives_1A')
intfrag_cluster_info_dirpath = path.join(fragment_db, 'IJK_ClusteredInterfaceFragmentDBInfo_1A')

# External Program Dependencies
# Free SASA Executable Path
free_sasa_exe_path = path.join(dependency_dir, 'sasa', 'freesasa-2.0', 'src', 'freesasa')
free_sasa_configuration_path = path.join(dependency_dir, 'sasa', 'freesasa-2.0.config')
# free_sasa_exe_path = path.join(nanohedra_source, "sasa", "freesasa-2.0", "src", "freesasa")

orient_dir = path.join(dependency_dir, 'orient')
# orient_exe = 'orient_oligomer.f'  # Non_compiled
orient_exe = 'orient_oligomer'
orient_exe_path = path.join(orient_dir, orient_exe)
orient_log_file = 'orient_oligomer_log.txt'
errat_exe_path = path.join(dependency_dir, 'errat', 'errat')
stride_exe_path = path.join(dependency_dir, 'stride', 'stride')
bmdca_exe_path = path.join(dependency_dir, 'bmDCA', 'src', 'bmdca')
ialign_exe_path = path.join(dependency_dir, 'ialign', 'bin', 'ialign.pl')
# Set up for alignment programs
reformat_msa_exe_path = path.join(dependency_dir, 'hh-suite', 'scripts', 'reformat.pl')
alignmentdb = path.join(dependency_dir, 'ncbi_databases', 'uniref90')
# alignment_db = path.join(dependency_dir, 'databases/uniref90')  # TODO
# TODO set up hh-suite in source or elsewhere on system and dynamically modify config file
hhblits_exe = search_env_for_variable(hhblits)
hhblits_exe = hhblits_exe if hhblits_exe else 'hhblits'  # ensure not None
uniclustdb = path.join(dependency_dir, 'hh-suite', 'databases', 'UniRef30_2020_02')  # TODO make db dynamic at config
# uniclust_db = path.join(database, 'hh-suite', 'databases', 'UniRef30_2020_02')  # TODO
install_hhsuite = path.join(binaries, 'install_hhsuite.sh')
# Rosetta
rosetta_main = search_env_for_variable(rosetta_str)
rosetta_main = rosetta_main if rosetta_main else 'main'  # ensure not None
make_symmdef = path.join(rosetta_main, 'source', 'src', 'apps', 'public', 'symmetry', 'make_symmdef_file.pl')
# Todo v dependent on external compile. cd to the directory, then type "make" to compile the executable
dalphaball = path.join(rosetta_main, 'source', 'external', 'DAlpahBall', 'DAlphaBall.gcc')
# Rosetta Scripts and Misc Files'
rosetta_scripts = path.join(dependency_dir, 'rosetta')
symmetry_def_file_dir = 'rosetta_symmetry_definition_files'
symmetry_def_files = path.join(rosetta_scripts, 'sdf')
sym_weights = 'ref2015_sym.wts_patch'
solvent_weights = 'ref2015_solvent.wts_patch'
solvent_weights_sym = 'ref2015_sym_solvent.wts_patch'
scout_symmdef = path.join(symmetry_def_files, 'scout_symmdef_file.pl')
symmetry_protocol = {0: 'make_point_group', 2: 'make_layer', 3: 'make_lattice'}  # -1: 'asymmetric',


# help and warnings
warn_missing_symmetry = f'Cannot %s without providing symmetry! Provide symmetry with "--symmetry" or "--{sym_entry}"'


def help(module):  # command is SymDesign.py
    return '"%s %s -h" for help' % (command, module)
