import os
# if nanohedra is outside of symdesign source folder
# import sys
# sys.path.append('..\\dependency_dir')


# Project strings and file names
program_name = 'SymDesign'
program_output = '%sOutput' % program_name
projects = 'Projects'
program_command = 'python SymDesign.py'
submodule_guide = '%s --guide MODULE' % program_command
submodule_help = '%s MODULE --help' % program_command
guide_string = '%s guide. Enter \'%s --guide\'' % (program_name, program_command)
nano = 'nanohedra'
interface_design = 'interface_design'
generate_fragments = 'generate_fragments'
analysis = 'analysis'
cluster_poses = 'cluster_poses'
select_designs = 'select_designs'
# orient_exe = 'orient_oligomer.f'  # Non_compiled
orient_exe = 'orient_oligomer'
hhblits = 'hhblits'
rosetta = os.environ.get('ROSETTA')
nstruct = 20  # Todo back to 50?
stage = {1: 'refine', 2: 'design', 3: 'metrics', 4: 'analysis', 5: 'consensus',
         6: 'rmsd_calculation', 7: 'all_to_all', 8: 'rmsd_clustering', 9: 'rmsd_to_cluster', 10: 'rmsd',
         11: 'all_to_cluster', 12: 'scout', 13: 'hbnet_design_profile'}
stage_f = {stage[1]: {'path': '*_refine.pdb', 'len': 1}, stage[2]: {'path': '*_design_*.pdb', 'len': nstruct},
           stage[3]: {'path': '', 'len': None}, stage[4]: {'path': '', 'len': None},
           stage[5]: {'path': '*_consensus.pdb', 'len': 1}, 'nanohedra': {'path': '', 'len': None},
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
pssm = 'evolutionary.pssm'  # was 'asu_pose.pssm' 1/25/21
fssm = 'fragment.pssm'
dssm = 'design.pssm'
assembly = 'assembly.pdb'
frag_dir = 'matching_fragments'  # was 'matching_fragments_representatives' in v0
frag_text_file = 'frag_match_info_file.txt'
docked_pose_file = 'docked_pose_info_file.txt'
frag_file = os.path.join(frag_dir, frag_text_file)
pose_file = 'docked_pose_info_file.txt'
frag_profile = '_fragment_profile'
protein_data = 'ProteinData'
sequence_info = 'SequenceInfo'  # was Sequence_Info 1/25/21
# profiles = 'profiles'
design_directory = 'Designs'

data = 'data'
pdbs_outdir = 'designs'  # was rosetta_pdbs/ 1/25/21
all_scores = 'AllScores'
scores_outdir = 'scores'
scripts = 'scripts'
scores_file = 'design_scores.sc'  # was all_scores.sc 1/25/21
pose_metrics_file = 'pose_scores.sc'  # UNUSED
analysis_file = '%sPoseMetrics-%s.csv'
directory_structure = './design_symmetry_pg/building_blocks/DEGEN_A_B/ROT_A_B/tx_C\n' \
                      'Ex:P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2'\
                      '\nIn design directory \'tx_c/\', output is located in \'%s\' and \'%s\'.' \
                      '\nTotal design_symmetry_pg score are located in ./design_symmetry_pg/building_blocks/%s' \
                      % (pdbs_outdir, scores_outdir, scores_outdir)
variance = 0.8
clustered_poses = '%sClusteredPoses-%s.pkl'
pdb_source = 'db'  # 'fetch_pdb'  # TODO set up

# Project paths
source = os.path.dirname(os.path.realpath(__file__))  # reveals master symdesign folder
readme = os.path.join(source, 'README.md')
command = os.path.join(source, 'SymDesign.py')
# filter_designs = os.path.join(source, 'DesignMetrics.py')
cmd_dist = os.path.join(source, 'CommandDistributer.py')
dependency_dir = os.path.join(source, 'dependencies')
sym_op_location = os.path.join(dependency_dir, 'symmetry_operators')
# nanohedra_source = os.path.join(dependency_dir, nano)
nanohedra_main = os.path.join(source, '%s.py' % nano.title())
nanohedra_dock = os.path.join(source, 'FragDock.py')
# nanohedra_main = os.path.join(nanohedra_source, '%s.py' % nano)
# Nanohedra inheritance
# Free SASA Executable Path
free_sasa_exe_path = os.path.join(dependency_dir, 'sasa', 'freesasa-2.0', 'src', 'freesasa')
# free_sasa_exe_path = os.path.join(nanohedra_source, "sasa", "freesasa-2.0", "src", "freesasa")
binary_lookup_table_path = os.path.join(dependency_dir, 'euler_lookup', 'euler_lookup_40.npz')
# Stop Inheritance ####
orient_dir = os.path.join(dependency_dir, 'orient')
orient_exe_path = os.path.join(orient_dir, orient_exe)
orient_log_file = 'orient_oligomer_log.txt'
stride_exe_path = os.path.join(dependency_dir, 'stride', 'stride')
binaries = os.path.join(dependency_dir, 'bin')
list_pdb_files = os.path.join(binaries, 'list_files_in_directory.py')
hbnet_sort = os.path.join(binaries, 'sort_hbnet_silent_file_results.sh')
sbatch_template_dir = os.path.join(binaries, 'sbatch')
disbatch = os.path.join(binaries, 'diSbatch.sh')  # DEPRECIATED
install_hhsuite = os.path.join(binaries, 'install_hhsuite.sh')
data_dir = os.path.join(source, data)
reference_aa_file = os.path.join(data_dir, 'AAreference.pdb')
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
biological_fragmentDB = os.path.join(fragment_db, 'biological_interfaces')  # TODO change this directory style
biological_fragment_db_pickle = os.path.join(fragment_db, 'biological_interfaces.pkl')
bio_fragmentDB = os.path.join(fragment_db, 'bio')
# bio_frag_db = os.path.join(fragment_db, 'bio')  # TODO
xtal_fragmentDB = os.path.join(fragment_db, 'xtal')
# xtal_frag_db = os.path.join(fragment_db, 'xtal')  # TODO
full_fragmentDB = os.path.join(fragment_db, 'bio+xtal')
# full_frag_db = os.path.join(fragment_db, 'bio+xtal')  # TODO
frag_directory = {'biological_interfaces': biological_fragmentDB, 'bio': bio_fragmentDB, 'xtal': xtal_fragmentDB,
                  'bio+xtal': full_fragmentDB}
# Nanohedra Specific
frag_db = frag_directory['biological_interfaces']  # was fragment_db on 1/25/21
monofrag_cluster_rep_dirpath = os.path.join(fragment_db, "Top5MonoFragClustersRepresentativeCentered")
intfrag_cluster_rep_dirpath = os.path.join(fragment_db, "Top75percent_IJK_ClusterRepresentatives_1A")
intfrag_cluster_info_dirpath = os.path.join(fragment_db, "IJK_ClusteredInterfaceFragmentDBInfo_1A")

# frag_directory = {'biological_interfaces': biological_frag_db, 'bio': bio_frag_db, 'xtal': xtal_frag_db,
#                   'bio+xtal': full_frag_db}  # TODO

# External Program Dependencies
make_symmdef = os.path.join(rosetta, 'source/src/apps/public/symmetry/make_symmdef_file.pl')
# Todo v dependent on external compile. cd to the directory, then type "make" to compile the executable
dalphaball = os.path.join(rosetta, 'source/external/DAlpahBall/DAlphaBall.gcc')
alignmentdb = os.path.join(dependency_dir, 'ncbi_databases/uniref90')
# alignment_db = os.path.join(dependency_dir, 'databases/uniref90')  # TODO
# TODO set up hh-suite in source or elsewhere on system and dynamically modify config file
uniclustdb = os.path.join(dependency_dir, 'hh-suite/databases', 'UniRef30_2020_02')  # TODO make db dynamic at config
# uniclust_db = os.path.join(database, 'hh-suite/databases', 'UniRef30_2020_02')  # TODO
# Rosetta Scripts and Misc Files
rosetta_scripts = os.path.join(dependency_dir, 'rosetta')
symmetry_def_file_dir = 'rosetta_symmetry_definition_files'
symmetry_def_files = os.path.join(rosetta_scripts, 'sdf')
sym_weights = 'ref2015_sym.wts_patch'
solvent_weights = 'ref2015_sym_solvent.wts_patch'
scout_symmdef = os.path.join(symmetry_def_files, 'scout_symmdef_file.pl')
protocol = {0: 'make_point_group', 2: 'make_layer', 3: 'make_lattice'}  # -1: 'asymmetric',


def help(module):  # command is SymDesign.py
    return '\'%s %s -h\' for help' % (command, module)
