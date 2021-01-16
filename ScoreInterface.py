import argparse
import os

import numpy as np
import pandas as pd

from PDB import PDB
from Pose import calculate_interface_score
from SymDesignUtils import start_log, unpickle, get_all_pdb_file_paths, mp_map
# from symdesign.interface_analysis.InterfaceSorting import return_pdb_interface
from classes import EulerLookup
from classes.Fragment import FragmentDB

# Globals
# Nanohedra.py Path
main_script_path = os.path.dirname(os.path.realpath(__file__))

# Fragment Database Directory Paths
frag_db = os.path.join(main_script_path, 'data', 'databases', 'fragment_db', 'biological_interfaces')
monofrag_cluster_rep_dirpath = os.path.join(frag_db, "Top5MonoFragClustersRepresentativeCentered")
ijk_intfrag_cluster_rep_dirpath = os.path.join(frag_db, "Top75percent_IJK_ClusterRepresentatives_1A")
intfrag_cluster_info_dirpath = os.path.join(frag_db, "IJK_ClusteredInterfaceFragmentDBInfo_1A")

# Free SASA Executable Path
free_sasa_exe_path = os.path.join(main_script_path, 'nanohedra', "sasa", "freesasa-2.0", "src", "freesasa")

# Create fragment database for all ijk cluster representatives
ijk_frag_db = FragmentDB(monofrag_cluster_rep_dirpath, ijk_intfrag_cluster_rep_dirpath,
                         intfrag_cluster_info_dirpath)
# Get complete IJK fragment representatives database dictionaries
ijk_monofrag_cluster_rep_pdb_dict = ijk_frag_db.get_monofrag_cluster_rep_dict()
ijk_intfrag_cluster_rep_dict = ijk_frag_db.get_intfrag_cluster_rep_dict()
ijk_intfrag_cluster_info_dict = ijk_frag_db.get_intfrag_cluster_info_dict()
if not ijk_intfrag_cluster_rep_dict:
    print('No reps found!')

# Initialize Euler Lookup Class
eul_lookup = EulerLookup()


def get_interface_fragment_chain_residue_numbers(pdb1, pdb2, cb_distance=8):
    """Given two PDBs, return the unique chain and interacting residue lists"""
    # Get the interface residues
    pdb1_cb_coords, pdb1_cb_indices = pdb1.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)
    pdb2_cb_coords, pdb2_cb_indices = pdb2.get_CB_coords(ReturnWithCBIndices=True, InclGlyCA=True)

    pdb1_cb_kdtree = sklearn.neighbors.BallTree(np.array(pdb1_cb_coords))

    # Query PDB1 CB Tree for all PDB2 CB Atoms within "cb_distance" in A of a PDB1 CB Atom
    query = pdb1_cb_kdtree.query_radius(pdb2_cb_coords, cb_distance)

    # Get ResidueNumber, ChainID for all Interacting PDB1 CB, PDB2 CB Pairs
    interacting_pairs = []
    for pdb2_query_index in range(len(query)):
        if query[pdb2_query_index].tolist() != list():
            pdb2_cb_res_num = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].residue_number
            pdb2_cb_chain_id = pdb2.all_atoms[pdb2_cb_indices[pdb2_query_index]].chain
            for pdb1_query_index in query[pdb2_query_index]:
                pdb1_cb_res_num = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].residue_number
                pdb1_cb_chain_id = pdb1.all_atoms[pdb1_cb_indices[pdb1_query_index]].chain
                interacting_pairs.append(((pdb1_cb_res_num, pdb1_cb_chain_id), (pdb2_cb_res_num, pdb2_cb_chain_id)))

    # Get interface fragment information
    pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list = [], []
    for pair in interacting_pairs:

        pdb1_central_res_num = pair[0][0]
        pdb1_central_chain_id = pair[0][1]
        pdb2_central_res_num = pair[1][0]
        pdb2_central_chain_id = pair[1][1]

        pdb1_res_num_list = [pdb1_central_res_num - 2, pdb1_central_res_num - 1, pdb1_central_res_num,
                             pdb1_central_res_num + 1, pdb1_central_res_num + 2]
        pdb2_res_num_list = [pdb2_central_res_num - 2, pdb2_central_res_num - 1, pdb2_central_res_num,
                             pdb2_central_res_num + 1, pdb2_central_res_num + 2]

        frag1_ca_count = 0
        for atom in pdb1.all_atoms:
            if atom.chain == pdb1_central_chain_id:
                if atom.residue_number in pdb1_res_num_list:
                    if atom.is_CA():
                        frag1_ca_count += 1

        frag2_ca_count = 0
        for atom in pdb2.all_atoms:
            if atom.chain == pdb2_central_chain_id:
                if atom.residue_number in pdb2_res_num_list:
                    if atom.is_CA():
                        frag2_ca_count += 1

        if frag1_ca_count == 5 and frag2_ca_count == 5:
            if (pdb1_central_chain_id, pdb1_central_res_num) not in pdb1_central_chainid_resnum_unique_list:
                pdb1_central_chainid_resnum_unique_list.append((pdb1_central_chain_id, pdb1_central_res_num))

            if (pdb2_central_chain_id, pdb2_central_res_num) not in pdb2_central_chainid_resnum_unique_list:
                pdb2_central_chainid_resnum_unique_list.append((pdb2_central_chain_id, pdb2_central_res_num))

    return pdb1_central_chainid_resnum_unique_list, pdb2_central_chainid_resnum_unique_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nScore selected interfaces using Nanohedra Score')
    # ---------------------------------------------------
    parser.add_argument('-d', '--directory', type=str, help='Directory where interface files are located. Default=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='A serialized dictionary with selected PDB code: [interface ID] '
                                                       'pairs', required=True)
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')

    args, additional_flags = parser.parse_known_args()
    # Program input
    print('USAGE: python ScoreNative.py interface_type_pickled_dict interface_filepath_location number_of_threads')

    if args.debug:
        logger = start_log(name=os.path.basename(__file__), level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name=os.path.basename(__file__), level=2)

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
            interface_filepaths = get_all_pdb_file_paths(args.directory)

        missing_index = [i for i, file_path in enumerate(interface_filepaths)
                         if os.path.splitext(os.path.basename(file_path))[0] not in bio_reference_l]

        for i in reversed(missing_index):
            del interface_filepaths[i]

        for interface_path in interface_filepaths:
            pdb = PDB(file=interface_path)
            # pdb = read_pdb(interface_path)
            pdb.name = os.path.splitext(os.path.basename(interface_path))[0]

    elif args.file:
        # pdb_codes = to_iterable(args.file)
        pdb_interface_d = unpickle(args.file)
        # for pdb_code in pdb_codes:
        #     for interface_id in pdb_codes[pdb_code]:
        interface_pdbs = [return_pdb_interface(pdb_code, interface_id) for pdb_code in pdb_interface_d
                          for interface_id in pdb_interface_d[pdb_code]]
        if args.output:
            out_path = args.output_dir
            pdb_code_id_tuples = [(pdb_code, interface_id) for pdb_code in pdb_interface_d
                                  for interface_id in pdb_interface_d[pdb_code]]
            for interface_pdb, pdb_code_id_tuple in zip(interface_pdbs, pdb_code_id_tuples):
                interface_pdb.write(os.path.join(args.output_dir, '%s-%d.pdb' % pdb_code_id_tuple))
    # else:
    #     logger.critical('Either --file or --directory must be specified')
    #     exit()

    if args.multi_processing:
        results = mp_map(calculate_interface_score, interface_pdbs, threads=int(sys.argv[3]))
        interface_d = {result for result in results}
        # interface_d = {key: result[key] for result in results for key in result}
    else:
        interface_d = {calculate_interface_score(interface_pdb) for interface_pdb in interface_pdbs}

    interface_df = pd.DataFrame(interface_d)
    interface_df.to_csv('BiologicalInterfaceNanohedraScores.csv')
