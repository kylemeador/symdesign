import os


# Project strings and file names
nano = 'Nanohedra.py'
program_name = 'SymDesign'
hhblits = 'hhblits'
nstruct = 25  # back to 50?
stage = {1: 'refine', 2: 'design', 3: 'metrics', 4: 'analysis', 5: 'consensus'}  # ,
stage_f = {'refine': {'path': '*_refine.pdb', 'len': 1}, 'design': {'path': '*_design_*.pdb', 'len': nstruct},
           'metrics': {'path': '', 'len': None}, 'analysis': {'path': '', 'len': None},
           'consensus': {'path': '*_consensus.pdb', 'len': 1}}
rosetta_extras = 'mpi'  # 'cxx11threadmpi' TODO make dynamic at config
sb_flag = '#SBATCH --'
sbatch = '_sbatch.sh'
temp = 'temp.hold'
pose_prefix = 'tx_'
asu = 'central_asu.pdb'
clean = 'clean_asu.pdb'
msa_pssm = 'asu_pose.pssm'
dssm = 'pose.dssm'
frag_dir = 'matching_fragment_representatives'
frag_file = os.path.join(frag_dir, 'frag_match_info_file.txt')
frag_type = '_fragment_profile'
data = 'data'
sequence_info = 'Sequence_Info'
pdbs_outdir = 'rosetta_pdbs/'  # TODO change to designs/
scores_outdir = 'scores/'
scores_file = 'all_scores.sc'
analysis_file = 'AllDesignPoseMetrics.csv'
directory_structure = './design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C\nEx:P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2'\
                      '\nIn design directory \'tx_c/\', output is located in \'%s\' and \'%s\'.' \
                      '\nTotal design_symmetry score are located in ./design_symmetry/building_blocks/%s' \
                      % (pdbs_outdir, scores_outdir, scores_outdir)
variance = 0.8
clustered_poses = 'ClusteredPoses'

# Project paths
# command = 'SymDesign.py -h'
command = 'SymDesignControl -h'
source = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2])
# source = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])  # TODO
dependencies = os.path.join(source, 'dependencies')  # TODO remove
pdb_uniprot_map = os.path.join(source, 'pdb_uniprot_map')  # TODO
# uniprot_pdb_map = os.path.join(source, 'uniprot_pdb_map')  # TODO
database = os.path.join(source, 'database')
pdb_db = os.path.join(database, 'PDB.db')  # TODO pointer to pdb database or to pdb website?
pdb_source = 'db'  # 'download_pdb'  # TODO set up
binaries = os.path.join(dependencies, 'bin')
# binaries = os.path.join(source, 'bin')  # TODO
process_commands = os.path.join(binaries, 'ProcessDesignCommands.sh `pwd`')
disbatch = os.path.join(binaries, 'diSbatch.sh')
fragment_database = os.path.join(dependencies, 'fragment_database')
# fragment_db = os.path.join(database, 'fragment_db')  # TODO
python_scripts = os.path.join(dependencies, 'python')
uniprot_pdb_map = os.path.join(python_scripts, '200121_UniProtPDBMasterDict.pkl')  # TODO move to source
# python_scripts = os.path.join(source, 'python')  # TODO
filter_and_sort = os.path.join(python_scripts, 'filter_and_sort_df.csv')
# filter_and_sort = os.path.join(source, 'filter_and_sort_df.csv')  # TODO
rosetta_scripts = os.path.join(dependencies, 'rosetta')
# rosetta_scripts = os.path.join(source, 'rosetta')  # TODO
symmetry_def_files = os.path.join(rosetta_scripts, 'sdf')
scout_symmdef = os.path.join(symmetry_def_files, 'scout_symmdef_file.pl')
install_hhsuite = os.path.join(binaries, 'install_hhsuite.sh')

# External Program Dependencies
orient = os.path.join(source, 'orient_oligomer')  # TODO
affinity_tags = os.path.join(database, 'modified-affinity-tags.csv')
alignmentdb = os.path.join(dependencies, 'ncbi_databases/uniref90')
# alignment_db = os.path.join(dependencies, 'databases/uniref90')  # TODO
# TODO set up hh-suite in source or elsewhere on system and dynamically modify config file
uniclustdb = os.path.join(dependencies, 'hh-suite/databases', 'UniRef30_2020_02')  # TODO make db dynamic at config
# uniclust_db = os.path.join(database, 'hh-suite/databases', 'UniRef30_2020_02')  # TODO
rosetta = str(os.environ.get('ROSETTA'))
make_symmdef = os.path.join(rosetta, 'source/src/apps/public/symmetry/make_symmdef_file.pl')

# Python Scripts
filter_designs = os.path.join(python_scripts, 'AnalyzeOutput.py')
cmd_dist = os.path.join(python_scripts, 'CommandDistributer.py')

# Fragment Database
biological_fragmentDB = os.path.join(fragment_database, 'biological_interfaces')
bio_fragmentDB = os.path.join(fragment_database, 'bio')
# bio_frag_db = os.path.join(fragment_db, 'bio')  # TODO
xtal_fragmentDB = os.path.join(fragment_database, 'xtal')
# xtal_frag_db = os.path.join(fragment_db, 'xtal')  # TODO
full_fragmentDB = os.path.join(fragment_database, 'bio+xtal')
# full_frag_db = os.path.join(fragment_db, 'bio+xtal')  # TODO
frag_directory = {'biological_interfaces': biological_fragmentDB, 'bio': bio_fragmentDB, 'xtal': xtal_fragmentDB,
                  'bio+xtal': full_fragmentDB}
# frag_directory = {'biological_interfaces': biological_frag_db, 'bio': bio_frag_db, 'xtal': xtal_frag_db,
#                   'bio+xtal': full_frag_db}  # TODO

# Rosetta Scripts and Files
sym_weights = (os.path.join(rosetta_scripts, 'ref2015_sym.wts_patch'))
protocol = {1: 'make_point_group', 2: 'make_layer', 3: 'make_lattice'}

# Cluster Dependencies and Multiprocessing
# stage = {1: 'refine', 2: 'design', 3: 'metrics', 4: 'analysis', 5: 'consensus'}
sbatch_templates = {stage[1]: os.path.join(binaries, sbatch[1:7], stage[1]),
                    stage[2]: os.path.join(binaries, sbatch[1:7], stage[2]),
                    stage[3]: os.path.join(binaries, sbatch[1:7], stage[2]),
                    stage[4]: os.path.join(binaries, sbatch[1:7], stage[1]),
                    stage[5]: os.path.join(binaries, sbatch[1:7], stage[1])}
