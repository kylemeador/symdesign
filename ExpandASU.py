import argparse
import os

from SymDesignUtils import read_pdb, start_log
from nanohedra.utils.ExpandAssemblyUtils import get_ptgrp_sym_op, get_expanded_ptgrp_pdb, write_expanded_ptgrp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nExpand a designed ASU to the full assembly')
    # ---------------------------------------------------
    parser.add_argument('-b', '--debug', action='store_true', help='Whether to run in debug mode')
    parser.add_argument('-f', '--file', type=str, help='.pdb file you wish to expand', default=None)
    parser.add_argument('-o', '--output', type=str, help='The output pathname of the expanded pdb?')
    parser.add_argument('-s', '--symmetry', type=str, help='What symmetry to be expanded?', default=None)

    args, additional_flags = parser.parse_known_args()

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name=os.path.basename(__file__), level=2)

    logger.info('Starting %s with options:\n\t%s' %
                (os.path.basename(__file__),
                 '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))
    if args.file and args.symmetry:
        asu_pdb = read_pdb(args.file)
        expand_matrices = get_ptgrp_sym_op(args.symmetry.upper())  # only works for T, O, I
        expanded_pdb = get_expanded_ptgrp_pdb(asu_pdb, expand_matrices)
        if args.output:
            path_name = args.output
        else:
            path_name = os.path.join(os.getcwd(), '%s_%s.pdb' % (os.path.splitext(args.file)[0], 'Expanded'))
        write_expanded_ptgrp(expanded_pdb, args.outpath)
        logger.info('Expanded assembly saved as %s' % path_name)
