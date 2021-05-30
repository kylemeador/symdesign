import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from sklearn.neighbors import BallTree

from SymDesignUtils import start_log, collect_designs
from Structure import MonoFragment
from interface_analysis.Database import FragmentDB
from PDB import PDB


def decorate_with_fragments(pdb_path, out_path=os.getcwd()):
    init_dir = 'init_fragments'
    complete_dir = 'complete_fragments'
    if not os.path.exists(os.path.join(out_path, init_dir)):
        os.makedirs(os.path.join(out_path, init_dir))

    if not os.path.exists(os.path.join(out_path, complete_dir)):
        os.makedirs(os.path.join(out_path, complete_dir))

    # Get PDB1 Symmetric Building Block
    # pdb1 = PDB()
    # pdb1.readfile(pdb_path)
    pdb1 = PDB.from_file(pdb_path)
    pdb1.renumber_residues()

    # Get Oligomer 1 Ghost Fragments With Guide Coordinates Using Initial Match Fragment Database
    kdtree_oligomer1_backbone = BallTree(np.array(pdb1.get_backbone_coords()))
    surface_residue_numbers = pdb1.get_surface_residues()
    surf_frags_1 = pdb1.get_fragments(residue_numbers=surface_residue_numbers)

    ghost_frag_list = []
    # ghost_frag_guide_coords_list = []
    for frag1 in surf_frags_1:
        monofrag1 = MonoFragment(frag1, {'1': ijk_monofrag_cluster_rep_pdb_dict['1']})
        monofrag_ghostfrag_list = monofrag1.get_ghost_fragments({'1': {'1': ijk_intfrag_cluster_rep_dict['1']['1']}},
                                                                kdtree_oligomer1_backbone,
                                                                {'1': {'1': ijk_intfrag_cluster_info_dict['1']['1']}})
        if monofrag_ghostfrag_list is not None:
            ghost_frag_list.extend(monofrag_ghostfrag_list)
            # ghost_frag_guide_coords_list.extend(map(GhostFragment.get_guide_coords, monofrag_ghostfrag_list))

    # Get Oligomer1 Ghost Fragments With Guide Coordinates Using COMPLETE Fragment Database
    complete_ghost_frag_list = []
    for frag1 in surf_frags_1:
        complete_monofrag1 = MonoFragment(frag1, ijk_monofrag_cluster_rep_pdb_dict)
        complete_monofrag1_ghostfrag_list = complete_monofrag1.get_ghost_fragments(
            ijk_intfrag_cluster_rep_dict, kdtree_oligomer1_backbone, ijk_intfrag_cluster_info_dict)
        if complete_monofrag1_ghostfrag_list:
            complete_ghost_frag_list.extend(complete_monofrag1_ghostfrag_list)

    for fragment in ghost_frag_list:
        fragment.pdb.write(out_path=os.path.join(out_path, init_dir, 'frag%s_chain%s_res%s.pdb'
                                                 % ('%d_%d_%d' % fragment.get_ijk(),
                                                    *fragment.get_aligned_chain_and_residue())))

    for fragment in complete_ghost_frag_list:
        fragment.pdb.write(out_path=os.path.join(out_path, complete_dir, 'frag%s_chain%s_res%s.pdb'
                                                 % ('%d_%d_%d' % fragment.get_ijk(),
                                                    *fragment.get_aligned_chain_and_residue())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nTurn file(s) from a full PDB biological assembly into an ASU containing one copy of all entities'
                    ' in contact with the chain specified by chain')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    parser.add_argument('-d', '--directory', type=str, help='Directory where \'.pdb\' files to set up ASU extraction'
                                                            'are located.\n')
    parser.add_argument('-f', '--file', type=str, help='File with list of pdb files of interest\n')
    parser.add_argument('-s', '--single', type=str, help='PDB file of interest\n')
    parser.add_argument('-o', '--out_path', type=str, help='Where should new files be saved?\nDefault=CWD')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name=os.path.basename(__file__), level=2)

    logger.info('Starting %s with options:\n\t%s' %
                (os.path.basename(__file__),
                 '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    if args.directory or args.file:
        file_paths, location = collect_designs(file=args.file, directory=args.directory)
    elif args.single:
        file_paths = [args.single]
    else:
        exit('Specify either a file or a directory to locate the files!')

    logger.info('Getting Fragment Information')
    ijk_frag_db = FragmentDB()
    # Get complete IJK fragment representatives database dictionaries
    ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
    ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
    ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()

    decorate = [decorate_with_fragments(file, out_path=args.out_path) for file in file_paths]
