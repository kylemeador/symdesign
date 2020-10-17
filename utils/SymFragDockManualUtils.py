def print_usage():
    print ''
    print '\033[32m' + '\033[1m' + "Nanohedra, 2019" + '\033[0m'
    print '\033[32m' + '\033[1m' + "Joshua Laniado & Todd O Yeates" + '\033[0m'
    print ''
    print '\033[1m' + '\033[95m' + "Nanohedra Manual" + '\033[95m' + '\033[0m'
    print ''
    print '\033[1m' + "QUERY MODE" + '\033[0m'
    print "REQUIRED FLAG"
    print "-query: used to enter query mode"
    print ''
    print "SELECT FROM ONE OF THE FOLLOWING QUERY OPTIONS"
    print "-all_entries"
    print "-query_combination"
    print "-query_result"
    print "-query_counterpart"
    print "-dimension"
    print ''
    print '\033[1m' + "DOCKING MODE" + '\033[0m'
    print "REQUIRED FLAGS"
    print "-dock: used to enter multiprocessing docking mode"
    print "-entry: used to specify symmetry entry number"
    print "-pdb_dir1_path: path to directory containing lower symmetry PDB files"
    print "-pdb_dir2_path: path to directory containing higher symmetry PDB files"
    print "-cores: number of cpu cores used for multiprocessing"
    print "-outdir: path to output directory"
    print ''
    print "OPTIONAL FLAGS"
    print "-rot_step1: PDB1 rotation sampling step in degrees"
    print "-rot_step2: PDB2 rotation sampling step in degrees"
    print "-output_uc: output design's central unit cell"
    print "-output_surrounding_uc: output design's surrounding unit cells"
    print ''
    print '\033[1m' + "POST PROCESSING MODE" + '\033[0m'
    print "REQUIRED FLAGS"
    print "-postprocess: used to enter post processing mode"
    print "-design_dir: path to design directory"
    print "-outdir: path to output directory"
    print ''
    print "SELECT ONE OR BOTH OF THE FOLLOWING POST PROCESSING FILTERS"
    print "-min_matched: used to specify a minimum number of matched fragments"
    print "-min_score: used to specify a minimum score"
    print ''
    print '\033[1m' + "EXAMPLES" + '\033[0m'
    print 'python Nanohedra.py -query -all_entries'
    print 'python Nanohedra.py -query -combination C3 D4'
    print 'python Nanohedra.py -query -result F432'
    print 'python Nanohedra.py -query -counterpart C5'
    print 'python Nanohedra.py -dock_mp -entry 65 -pdb_dir1_path /home/user/SymBuildBlocks/C3 -pdb_dir2_path /home/user/SymBuildBlocks/D4 -outdir /home/user/SymDockProject'
    print 'python Nanohedra.py -postprocess -design_dir /home/user/SymDockProject -min_matched 7 -outdir /home/user/SymDockProject/PostProcess'
    print 'python Nanohedra.py -postprocess -design_dir /home/user/SymDockProject -min_score 5.0 -outdir /home/user/SymDockProject/PostProcess'
    print 'python Nanohedra.py -postprocess -design_dir /home/user/SymDockProject -min_matched 7 -min_score 5.0 -outdir /home/user/SymDockProject/PostProcess'
    print ''
