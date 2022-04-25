import argparse
import os

from PDB import PDB
from SymDesignUtils import start_log, to_iterable, set_logging_to_debug
from utils.SymmetryUtils import get_ptgrp_sym_op, get_expanded_ptgrp_pdb  #, write_expanded_ptgrp


def expand_asu(file, symmetry, out_path=None):
    if out_path:
        path_name = os.path.join(out_path, '%s_%s.pdb' % (os.path.basename(os.path.splitext(file)[0]), 'expanded'))
    else:
        path_name = '%s_%s.pdb' % (os.path.splitext(file)[0], 'expanded')

    asu_pdb = PDB(file=file)
    expand_matrices = get_ptgrp_sym_op(symmetry.upper())  # currently only works for T, O, I
    expanded_pdb = get_expanded_ptgrp_pdb(asu_pdb, expand_matrices)
    write_expanded_ptgrp(expanded_pdb, path_name)

    return path_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nExpand target ASU(s) to their full assembly. Assumes that the ASU '
                                                 'is symmetrically related to the origin (i.e. center of mass symm = '
                                                 '0, 0, 0) and if multiple files are specified, all have the same '
                                                 'symmetry. Garbage in, garbage out,')
    # ---------------------------------------------------
    parser.add_argument('-b', '--debug', action='store_true', help='Whether to run in debug mode')
    parser.add_argument('-f', '--file', type=str, help='A single .pdb file or file with list of .pdb files you wish to '
                                                       'expand', required=True)
    parser.add_argument('-o', '--out_path', type=str, help='The output path of the expanded pdb')
    parser.add_argument('-s', '--symmetry', type=str, choices=['T', 'O', 'I'], help='What symmetry to be expanded?',
                        required=True)

    args, additional_flags = parser.parse_known_args()

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        set_logging_to_debug()
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name=os.path.basename(__file__), propagate=True)

    logger.info('Starting %s with options:\n\t%s' %
                (os.path.basename(__file__),
                 '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Get all possible input files
    files = to_iterable(args.file)
    logger.info('Found %d files in input.' % len(files))
    # Symmetrize the input asu(s)
    path_names = [expand_asu(file, args.symmetry, out_path=args.out_path) for file in files]
    logger.info('\nExpanded assembl%s saved as:\n%s\n\n' % ('y' if len(files) == 1 else 'ies', '\n'.join(path_names)))
