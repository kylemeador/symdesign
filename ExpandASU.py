import argparse
import os
from typing import Union

from PDB import PDB
from SymDesignUtils import start_log, to_iterable, set_logging_to_debug
from utils.SymmetryUtils import cubic_point_groups, point_group_symmetry_operators


def get_expanded_ptgrp_pdb(pdb_asu, expand_matrices):
    """Returns a list of PDB objects from the symmetry mates of the input expansion matrices"""
    asu_symm_mates = []
    # asu_coords = pdb_asu.extract_coords()
    # asu_coords = pdb_asu.extract_all_coords()
    for r in expand_matrices:
        asu_sym_mate_pdb = pdb_asu.return_transformed_copy(rotation=r.T)
        asu_symm_mates.append(asu_sym_mate_pdb)

    return asu_symm_mates


def write_expanded_ptgrp(expanded_ptgrp_pdbs, outfile_path):
    outfile = open(outfile_path, "w")
    model_count = 1
    for pdb in expanded_ptgrp_pdbs:
        outfile.write("MODEL     {:>4s}\n".format(str(model_count)))
        model_count += 1
        for atom in pdb.all_atoms:
            outfile.write(str(atom))
        outfile.write("ENDMDL\n")
    outfile.close()


def expand_asu(file, symmetry, out_path=None) -> Union[str, bytes]:
    if out_path:
        path_name = os.path.join(out_path, '%s_%s.pdb' % (os.path.basename(os.path.splitext(file)[0]), 'expanded'))
    else:
        path_name = '%s_%s.pdb' % (os.path.splitext(file)[0], 'expanded')

    asu_pdb = PDB(file=file)
    expand_matrices = point_group_symmetry_operators[symmetry.upper()]
    expanded_pdb = get_expanded_ptgrp_pdb(asu_pdb, expand_matrices)
    write_expanded_ptgrp(expanded_pdb, path_name)

    return path_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nExpand target ASU(s) to their full assembly. Assumes that the ASU '
                                                 'is symmetrically related to the origin (i.e. center of mass symm = '
                                                 '0, 0, 0) and if multiple files are specified, all have the same '
                                                 'symmetry. Garbage in, garbage out,')
    # ---------------------------------------------------
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode')
    parser.add_argument('-f', '--file', type=str, help='A single .pdb file or file with list of .pdb files you wish to '
                                                       'expand', required=True)
    parser.add_argument('-o', '--out_path', type=str, help='The output path of the expanded pdb')
    parser.add_argument('-s', '--symmetry', type=str, choices=cubic_point_groups, help='What symmetry to be expanded?',
                        required=True)

    args, additional_flags = parser.parse_known_args()

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = start_log(name=os.path.basename(__file__), propagate=True)

    logger.info('Starting %s with options:\n\t%s' %
                (os.path.basename(__file__),
                 '\n\t'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Get all possible input files
    files = to_iterable(args.file, ensure_file=True)
    logger.info('Found %d files in input.' % len(files))
    # Symmetrize the input asu(s)
    path_names = [expand_asu(file, args.symmetry, out_path=args.out_path) for file in files]
    logger.info('\nExpanded assembl%s saved as:\n%s\n\n' % ('y' if len(files) == 1 else 'ies', '\n'.join(path_names)))
