import argparse
import os
from itertools import repeat

import pandas as pd

from Pose import Pose
import PathUtils as PUtils
from SymDesignUtils import start_log, unpickle, get_all_file_paths, to_iterable, mp_starmap, set_logging_to_debug
from classes.EulerLookup import EulerLookup

# Globals
# Create fragment database for all ijk cluster representatives
ijk_frag_db = unpickle(PUtils.biological_fragment_db_pickle)
# ijk_frag_db = FragmentDB()
# # Get complete IJK fragment representatives database dictionaries
# ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
# ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
# ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()

# Initialize Euler Lookup Class
eul_lookup = EulerLookup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nScore selected interfaces using Nanohedra Score')
    # ---------------------------------------------------
    parser.add_argument('-d', '--directory', type=str, help='Directory where interface files are located. Default=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='A serialized dictionary with selected PDB code: [interface ID] '
                                                       'pairs', required=True)
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-t', '--threads', type=int, help='How many threads should be utilized?\nDefault=1', default=1)
    parser.add_argument('--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')

    args, additional_flags = parser.parse_known_args()
    # Program input
    print('USAGE: python ScoreNative.py interface_type_pickled_dict interface_filepath_location number_of_threads')

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        set_logging_to_debug()
        logger.debug('Debug mode. Produces verbose output and not written to any .log files')
    else:
        logger = start_log(name=os.path.basename(__file__), level=2, propagate=True)

    interface_pdbs = []
    if args.directory:
        interface_reference_d = unpickle(args.file)
        bio_reference_l = interface_reference_d['bio']

        print('Total of %d PDB\'s to score' % len(bio_reference_l))
        # try:
        #     print('1:', '1AB0-1' in bio_reference_l)
        #     print('2:', '1AB0-2' in bio_reference_l)
        #     print('no dash:', '1AB0' in bio_reference_l)
        # except KeyError as e:
        #     print(e)

        if args.debug:
            first_5 = ['2OPI-1', '4JVT-1', '3GZD-1', '4IRG-1', '2IN5-1']
            next_5 = ['3ILK-1', '3G64-1', '3G64-4', '3G64-6', '3G64-23']
            next_next_5 = ['3AQT-2', '2Q24-2', '1LDF-1', '1LDF-11', '1QCZ-1']
            paths = next_next_5
            root = '/home/kmeador/yeates/fragment_database/all/all_interfaces/%s.pdb'
            # paths = ['op/2OPI-1.pdb', 'jv/4JVT-1.pdb', 'gz/3GZD-1.pdb', 'ir/4IRG-1.pdb', 'in/2IN5-1.pdb']
            interface_filepaths = [root % '%s/%s' % (path[1:3].lower(), path) for path in paths]
            # interface_filepaths = list(map(os.path.join, root, paths))
        else:
            interface_filepaths = get_all_file_paths(args.directory, extension='.pdb')

        # # Used for all biological interface scoring
        # missing_index = [i for i, file_path in enumerate(interface_filepaths)
        #                  if os.path.splitext(os.path.basename(file_path))[0] not in bio_reference_l]
        #
        # for i in reversed(missing_index):
        #     del interface_filepaths[i]
        #
        # # for interface_path in interface_filepaths:
        # #     pdb = PDB(file=interface_path)
        # #     # pdb = read_pdb(interface_path)
        # #     pdb.name = os.path.splitext(os.path.basename(interface_path))[0]

        # Viable for the design recap test where files are I32-01.pdb
        interface_poses = [Pose.from_asu_file(interface_path, symmetry=os.path.basename(interface_path[0]))
                           for interface_path in interface_filepaths]

    elif args.file:
        # # Used to write all pdb interfaces to an output location
        # # pdb_codes = to_iterable(args.file)
        # pdb_interface_d = unpickle(args.file)
        # # for pdb_code in pdb_codes:
        # #     for interface_id in pdb_codes[pdb_code]:
        # interface_pdbs = [return_pdb_interface(pdb_code, interface_id) for pdb_code in pdb_interface_d
        #                   for interface_id in pdb_interface_d[pdb_code]]
        # if args.output:
        #     out_path = args.output_dir
        #     pdb_code_id_tuples = [(pdb_code, interface_id) for pdb_code in pdb_interface_d
        #                           for interface_id in pdb_interface_d[pdb_code]]
        #     for interface_pdb, pdb_code_id_tuple in zip(interface_pdbs, pdb_code_id_tuples):
        #         interface_pdb.write(os.path.join(args.output_dir, '%s-%d.pdb' % pdb_code_id_tuple))

        interface_filepaths = to_iterable(args.file, ensure_file=True)
        interface_poses = [Pose.from_asu_file(interface_path, symmetry=os.path.basename(interface_path[0]))
                           for interface_path in interface_filepaths]
    else:
        interface_poses = False
    #     logger.critical('Either --file or --directory must be specified')
    #     exit()

    if args.multi_processing:
        # # used without Pose
        # results = mp_map(calculate_interface_score, interface_pdbs, threads=args.threads)
        # interface_d = {result for result in results}
        # # interface_d = {key: result[key] for result in results for key in result}

        zipped_args = zip(interface_poses, repeat(1), repeat(2))
        # Todo change zipped_args, args to (entity1, entity2) self.entities?
        #  score_interface(entity1=None, entity2=None)
        results = mp_starmap(Pose.score_interface, zipped_args, threads=args.threads)
        interface_d = {pose.name: result.values() for pose, result in zip(interface_poses, results)}
    else:
        raise RuntimeError('This functionality is currently broken')
        interface_d = {calculate_interface_score(interface_pdb) for interface_pdb in interface_pdbs}

    interface_df = pd.DataFrame(interface_d)
    # dataframe_name = 'BiologicalInterfaceNanohedraScores.csv'
    dataframe_name = 'DesignedCagesInterfaceNanohedraScores.csv'
    interface_df.to_csv(dataframe_name)
