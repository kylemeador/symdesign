def pose_rmsd(all_des_dirs):
    from Bio.PDB import PDBParser
    from Bio.PDB.Selection import unfold_entities
    from itertools import combinations
    threshold = 1.0  # TODO test

    pose_map = {}
    for pair in combinations(all_des_dirs, 2):
        if pair[0].building_blocks == pair[1].building_blocks:
            # returns a list with all ca atoms from a structure
            # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].asu, pair[1].asu], SDUtils.get_biopdb_ca)
            pdb_parser = PDBParser()
            # pdb = parser.get_structure(pdb_name, filepath)
            pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
            # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].path, pair[1].path], SDUtils.get_biopdb_ca)
            # grabs stats['des_resides'] from the design_directory
            des_residue_list = [pair[n].stats['des_residues'] for n in pair]
            # des_residues = SDUtils.unpickle(os.path.join(des_dir.data, PUtils.des_residues))
            # # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/data
            # pair should be a structure...
            rmsd_residue_list = [residue for n, structure in enumerate(pair_structures)
                                 for residue in structure.get_residues() if residue in des_residue_list[n]]
            pair_atom_list = SDUtils.get_rmsd_atoms(rmsd_residue_list, SDUtils.get_biopdb_ca)
            # pair_rmsd = SDUtils.superimpose(pair_atoms, threshold)
            pair_rmsd = SDUtils.superimpose(pair_atom_list, threshold)
            if not pair_rmsd:
                continue
            if pair[0].building_blocks in pose_map:
                # {building_blocks: {(pair1, pair2): rmsd, ...}, ...}
                if str(pair[0]) in pose_map[pair[0].building_blocks]:
                    pose_map[pair[0].building_blocks][str(pair[0])][str(pair[1])] = pair_rmsd[2]  # 2 is the rmsd value
                else:
                    pose_map[pair[0].building_blocks][str(pair[0])] = {str(pair[1]): pair_rmsd[2]}
                # pose_map[pair[0].building_blocks][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
            else:
                pose_map[pair[0].building_blocks] = {str(pair[0]): {str(pair[1]): pair_rmsd[2]}}
                # pose_map[pair[0].building_blocks] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}

    return pose_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nControl all input/output of %s including:\n1. Pose initialization\n'
                    '2. Command distribution to computational nodes\n'
                    '3. Analysis of designs' % PUtils.program_name)
    parser.add_argument('-d', '--directory', type=str, help='Directory where %s poses are located. Default=CWD'
                                                            % PUtils.program_name, default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of %s poses' % PUtils.program_name,
                        default=None)
    parser.add_argument('-m', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-b', '--debug', action='store_true',
                        help='Debug all steps to standard out?\nDefault=False')
    parser.add_argument('-s', '--design_string', type=str,
                        help='If pose names are specified by design string instead '
                             'of directories, which directory path to '
                             'prefix with?\nDefault=None', default=None)

    subparsers = parser.add_subparsers(title='SubModules', dest='sub_module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags')

    parser_pose = subparsers.add_parser('pose', help='Gather output from %s and format for input into Rosetta. '
                                                     'Sets up interface design constrained evolutionary profiles '
                                                     'of homologous sequences and by fragment profiles extracted '
                                                     'from the PDB' % PUtils.nano)
    parser_pose.add_argument('-i', '--fragment_database', type=str,
                             help='Database to match fragments for interface specific scoring matrices. One of %s'
                                  '\nDefault=%s' %
                                  (','.join(list(PUtils.frag_directory.keys())),
                                   list(PUtils.frag_directory.keys())[0]),
                             default=list(PUtils.frag_directory.keys())[0])
    parser_pose.add_argument('symmetry_group', type=int,
                             help='What type of symmetry group does your design belong too? One of 0-Point Group, '
                                  '2-Plane Group, or 3-Space Group')  # TODO remove from input, make automatic
    parser_pose.add_argument('-c', '--command_only', action='store_true',
                             help='Should commands be written but not executed?\nDefault=False')
    parser_pose.add_argument('-x', '--suspend', action='store_true',
                             help='Should Rosetta design trajectory be suspended?\nDefault=False')
    parser_pose.add_argument('-p', '--mpi', action='store_true',
                             help='Should job be set up for cluster submission?\nDefault=False')

    parser_dist = subparsers.add_parser('distribute',
                                        help='Distribute specific design step commands to computational resources. '
                                             'In distribution mode, the --file or --directory argument specifies which '
                                             'pose commands should be distributed.')
    parser_dist.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                             help='The stage of design to be prepared. One of %s' %
                                  ', '.join(list(v for v in PUtils.stage_f.keys())), required=True)
    parser_dist.add_argument('-y', '--success_file',
                             help='The name/location of file containing successful commands\n'
                                  'Default={--stage}_stage_pose_successes', default=None)
    parser_dist.add_argument('-n', '--failure_file', help='The name/location of file containing failed commands\n'
                                                          'Default={--stage}_stage_pose_failures', default=None)
    parser_dist.add_argument('-m', '--max_jobs', type=int, help='How many jobs to run at once?\nDefault=80',
                             default=80)

    parser_analysis = subparsers.add_parser('analysis', help='Run analysis on all poses specified')
    parser_analysis.add_argument('-o', '--output', type=str,
                                 help='Name to output comma delimitted files.\nDefault=%s' % PUtils.analysis_file,
                                 default=PUtils.analysis_file)
    parser_analysis.add_argument('-n', '--no_save', action='store_true',
                                 help='Don\'t save trajectory information.\nDefault=False')
    parser_analysis.add_argument('-f', '--figures', action='store_true',
                                 help='Create and save all pose figures?\nDefault=False')
    parser_analysis.add_argument('-j', '--join', action='store_true',
                                 help='Join Trajectory and Residue Dataframes?\nDefault=False')
    parser_analysis.add_argument('-g', '--delta_g', action='store_true',
                                 help='Compute deltaG versus Refine structure?\nDefault=False')

    parser_merge = subparsers.add_parser('merge', help='Merge all completed designs from location 2 (-f2/-d2) to '
                                                       'location 1(-f/-d). Includes renaming. Highly suggested you copy'
                                                       ' original data!!!')
    parser_merge.add_argument('-d2', '--directory2', type=str, help='Directory 2 where poses should be copied '
                                                                    'from and appended to location 1 poses',
                              default=None)
    parser_merge.add_argument('-f2', '--file2', type=str,
                              help='File 2 where poses should be copied from and appended '
                                   'to location 1 poses', default=None)
    parser_merge.add_argument('-i', '--increment', type=int,
                              help='How many to increment each design by?\nDefault=%d'
                                   % PUtils.nstruct)

    parser_modify = subparsers.add_parser('modify', help='Modify something for testing')

    parser_status = subparsers.add_parser('status', help='Get design status for selected designs')
    parser_status.add_argument('-n', '--number_designs', type=int, help='Number of trajectories per design',
                               default=None)
    parser_status.add_argument('-s', '--stage', choices=tuple(v for v in PUtils.stage_f.keys()),
                               help='The stage of design to check status of. One of %s'
                                    % ', '.join(list(v for v in PUtils.stage_f.keys())), default=None)

    parser_sequence = subparsers.add_parser('sequence', help='Generate protein sequences for selected designs')
    parser_sequence.add_argument('-c', '--consensus', action='store_true',
                                 help='Whether to grab the consensus sequence'
                                      '\nDefault=False')
    parser_sequence.add_argument('-d', '--dataframe', type=str,
                                 help='Dataframe.csv from analysis containing pose info')
    # TODO ^ require this or pose_design_file
    parser_sequence.add_argument('-f', '--filters', type=dict, help='Metrics with which to filter on\nDefault=None',
                                 default=None)
    parser_sequence.add_argument('-n', '--number', type=int, help='Number of top sequences to return per design',
                                 default=1)
    parser_sequence.add_argument('-p', '--pose_design_file', type=str,
                                 help='Name of pose, design .csv file to serve as'
                                      ' sequence selector')
    parser_sequence.add_argument('-s', '--selection_string', type=str,
                                 help='Output identifier for sequence selection')
    parser_sequence.add_argument('-w', '--weights', type=str, help='Weights of various metrics to final poses\n'
                                                                   'Default=1/number of --filters')

    parser_rename_scores = subparsers.add_parser('rename_scores',
                                                 help='Rename Protocol names according to dictionary')

    args = parser.parse_args()

    # Grab all poses (directories) to be processed from either directory name or file
    all_poses, location = SDUtils.collect_designs(args.directory, file=args.file)
    assert all_poses != list(), logger.critical('No %s directories found within \'%s\'! Please ensure correct location'
                                                % (PUtils.nano, location))

    all_design_directories = SDUtils.set_up_directory_objects(all_poses, symmetry=args.design_string)
    logger.info('%d Poses found in \'%s\'' % (len(all_poses), location))
    logger.info('All pose specific logs are located in corresponding directories, ex:\n%s' %
                os.path.join(all_design_directories[0].path, os.path.basename(all_design_directories[0].path) + '.log'))