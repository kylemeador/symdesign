def print_usage():
    print '\033[32m' + '\033[1m' + "NANOHEDRA\n" + '\033[0m'
    print '\033[32m' + '\033[1m' + "Copyright 2020 Joshua Laniado and Todd O. Yeates\n" + '\033[0m'
    print ''
    print '\033[1m' + '\033[95m' + "USER MANUAL" + '\033[95m' + '\033[0m'
    print ''
    print '\033[1m' + "QUERY MODE" + '\033[0m'
    print "REQUIRED FLAG"
    print "-query: used to enter query mode"
    print ''
    print "SELECT FROM ONE OF THE FOLLOWING QUERY OPTIONS"
    print "-all_entries: show all symmetry combination materials (SCMs)"
    print "-query_combination: show all SCMs that can be constructed by combining the two specified point groups"
    print "-query_result: show all SCMs that display the point group, layer group or space group symmetry specified"
    print "-query_counterpart: show all SCMs that can be constructed with the specified point group"
    print "-dimension: show all zero-dimensional, two-dimensional or three-dimensional SCMs"
    print ''
    print '\033[1m' + "DOCKING MODE" + '\033[0m'
    print "REQUIRED FLAGS"
    print "-dock: used to enter docking mode"
    print "-entry: specify symmetry combination material entry number"
    print "-oligomer1: specify path to directory containing the input PDB file(s) for the LOWER symmetry oligomer"
    print "-oligomer2: specify path to directory containing the input PDB file(s) for the HIGHER symmetry oligomer"
    print "            this flag is only used when both oligomeric components do not obey the SAME point group symmetry"
    print "            for SCMs where both oligomeric components obey the SAME point group symmetry only the -oligomer1"
    print "            flag is used to specify a path to a single directory containing the input PDB file(s)"
    print "-outdir: specify project output directory"
    print ''
    print "OPTIONAL FLAGS"
    print "-rot_step1: PDB1 rotation sampling step in degrees [default value is 3 degrees]"
    print "-rot_step2: PDB2 rotation sampling step in degrees [default value is 3 degrees]"
    print "-output_uc: output central unit cell for 2D and 3D symmetry combination materials"
    print "-output_surrounding_uc: output surrounding unit cells for 2D and 3D symmetry combination materials"
    print "-output_exp_assembly: output expanded cage assembly"
    print "-min_matched: specify a minimum amount of unique high quality surface fragment matches [default value is 3]"
    print "-init_match_type: Specify type i_j fragment pair type used for initial fragment matching."
    print "                  Default is helix_helix: 1_1. Other options are:"
    print "                  helix_strand, strand_helix and strand_strand: 1_2, 2_1 and 2_2 respectively."
    print ''
    print '\033[1m' + "POST PROCESSING MODE" + '\033[0m'
    print "REQUIRED FLAGS"
    print "-postprocess: used to enter post processing mode"
    print "-design_dir: specify path to project directory (i.e. path to output directory specified in DOCKING MODE)"
    print "-outdir: specify output directory for post processing output file(s)"
    print ''
    print "1) USE ONE OR BOTH OF THE FOLLOWING POST PROCESSING FILTERS"
    print "-min_matched: specify a minimum number of matched fragments"
    print "-min_score: specify a minimum score"
    print "2) OR USE THE FOLLOWING FLAG TO RANK DOCKED POSES"
    print "-rank: followed by 'score' to rank by score or 'matched' to rank by the number of unique surface fragment matches"
    print ''
    print '\033[1m' + "EXAMPLES" + '\033[0m'
    print 'python nanohedra.py -query -all_entries'
    print 'python nanohedra.py -query -combination C3 D4'
    print 'python nanohedra.py -query -result F432'
    print 'python nanohedra.py -query -counterpart C5'
    print 'python nanohedra.py -query -dimension 3'
    print 'python nanohedra.py -dock -entry 54 -oligomer1 /home/user/C3oligomers -outdir /home/user/T33Project'
    print 'python nanohedra.py -dock -entry 67 -oligomer1 /home/user/C3oligomers -oligomer2 /home/user/D4oligomers -outdir /home/user/P432Project'
    print 'python nanohedra.py -postprocess -design_dir /home/user/P432Project -min_score 20.0 -outdir /home/user/P432Project/PostProcess'
    print 'python nanohedra.py -postprocess -design_dir /home/user/P432Project -min_matched 7 -min_score 20.0 -outdir /home/user/P432Project/PostProcess'
    print 'python nanohedra.py -postprocess -design_dir /home/user/P432Project -rank score -outdir /home/user/P432Project/PostProcess'
    print ''
