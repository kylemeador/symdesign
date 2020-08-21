import os
import argparse
import SymDesignUtils as SDUtils


def make_asu(files, chain, destination=os.getcwd):
    return [SDUtils.extract_asu(file, chain=chain, outpath=destination) for file in files]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='\nTurn file(s) from a full PDB biological assembly into an ASU containing one copy of all entities'
                    ' in contact with the chain specified by chain')
    parser.add_argument('-d', '--directory', type=str, help='Directory where \'.pdb\' files are located.\nDefault=None',
                        default=None)
    parser.add_argument('-f', '--file', type=str, help='File with list of pdb files of interest\nDefault=None',
                        default=None)
    parser.add_argument('-c', '--chain', type=str, help='What chain would you like to leave?\nDefault=A', default='A')
    parser.add_argument('-o', '--output_destination', type=str, help='Where should new files be saved?\nDefault=CWD')

    args = parser.parse_args()
    logger = SDUtils.start_log()
    if args.directory and args.file:
        logger.error('Can only specify one of -d or -f not both. If you want to use a file list from a different '
                     'directory than the one you are in, you will have to navigate there!')
        exit()
    elif not args.directory and not args.file:
        logger.error('No file list specified. Please specify one of -d or -f to collect the list of files')
        exit()
    elif args.directory:
        file_list = SDUtils.get_all_pdb_file_paths(args.directory)
        # location = SDUtils.get_all_pdb_file_paths(args.directory)
    else:  # args.file
        file_list = SDUtils.to_iterable(args.file)
        # location = args.file

    # file_list = SDUtils.to_iterable(location)
    new_files = make_asu(file_list, args.chain, destination=args.output_destination)
    logger.info('ASU files were written to:\n%s' % '\n'.join(new_files))
