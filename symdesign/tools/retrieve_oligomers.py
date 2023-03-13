import argparse
import logging
import os

from symdesign import flags, utils
from symdesign.resources.query.pdb import QueryParams, nanohedra_building_blocks_query, parse_pdb_response_for_ids, \
    solve_confirmed_assemblies
from symdesign.utils import path as putils

logger = logging.getLogger(__name__)
logger.setLevel(20)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for symmetric oligomers\n')
    parser.add_argument(*flags.symmetry_args, required=True, type=str,
                        help='What is the schoenflies symbol of the desired oligomeric symmetry?')
    #                   **flags.symmetry_kwargs)
    parser.add_argument('--lower-length', default=80, type=int, help='How short is the shortest allowed protein?')
    parser.add_argument('--upper-length', default=300, type=int, help='How long is the longest allowed protein?')
    # parser.add_argument('-f', '--file', nargs='*', type=os.path.abspath, metavar=f'{putils.ex_path("pdblist.file")}',
    #                     help='File(s) containing EntryID codes. Can be newline or comma separated')
    # parser.add_argument('-d', '--download', action='store_true',
    #                     help='Whether files should be downloaded. Default=False')
    # parser.add_argument('-p', '--input-pdb-directory', type=os.path.abspath,
    #                     help='Where should reference PDB files be found? Default=CWD', default=os.getcwd())
    # parser.add_argument('-i', '--input-pisa-directory', type=os.path.abspath,
    #                     help='Where should reference PISA files be found? Default=CWD', default=os.getcwd())
    # parser.add_argument('-o', '--output-directory', type=os.path.abspath,
    #                     help='Where should files be saved?')
    # parser.add_argument('-q', '--query-web', action='store_true',
    #                     help='Should information be retrieved from the web?')
    # parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    # Testing retrieval of ids
    symmetry = 'C3'
    lower_length = 80
    upper_length = 300
    query_params = QueryParams(symmetry=args.symmetry, lower_length=args.lower_length, upper_length=args.upper_length)

    # Get thermophilic representative group ids and representatives
    # Grab the groups
    thermophilic_groups = \
        nanohedra_building_blocks_query(query_params.symmetry, query_params.lower_length, query_params.upper_length,
                                        thermophile=True, groups=True)
    # group_ids = [group['identifier'] for group in query_json["group_set"]]
    thermophilic_group_ids = parse_pdb_response_for_ids(thermophilic_groups, groups=True)
    grouped_thermophilic_entity_ids = [parse_pdb_response_for_ids(group) for group in thermophilic_groups['group_set']]
    logger.debug(f'Found the thermophilic return ids: {grouped_thermophilic_entity_ids}')
    # top_thermophilic_entity_ids = [ids_[0] for ids_ in grouped_thermophilic_entity_ids]

    thermophilic_group_to_members = dict(zip(thermophilic_group_ids, grouped_thermophilic_entity_ids))
    logger.info('Starting thermo assembly limiting')
    top_thermophilic_entity_ids, remove_thermo_groups = \
        solve_confirmed_assemblies(query_params, thermophilic_group_to_members)
    top_thermophilic_entity_ids = sorted(top_thermophilic_entity_ids)
    logger.info(f'Found the top thermophilic return ids: {top_thermophilic_entity_ids}')
    logger.info(f'Found the thermophilic group ids which have an unsolvable assembly: {remove_thermo_groups}')
    # Remove any that weren't solved
    for group_id in remove_thermo_groups:
        thermophilic_group_to_members.pop(group_id)
    logger.info('Starting other assembly limiting')
    # Then use these to put back into the search
    other_symmetry_groups = \
        nanohedra_building_blocks_query(query_params.symmetry, query_params.lower_length, query_params.upper_length,
                                        groups=True, limit_by_groups=thermophilic_group_to_members.keys())
    other_symmetry_group_ids = parse_pdb_response_for_ids(other_symmetry_groups, groups=True)
    grouped_other_symmetry_entity_ids = \
        [parse_pdb_response_for_ids(group) for group in other_symmetry_groups['group_set']]

    other_symmetry_group_to_members = dict(zip(other_symmetry_group_ids, grouped_other_symmetry_entity_ids))
    top_other_symmetry_entity_ids, remove_other_groups = \
        solve_confirmed_assemblies(query_params, other_symmetry_group_to_members)
    top_other_symmetry_entity_ids = sorted(top_other_symmetry_entity_ids)
    logger.info(f'Found the top other return ids: {top_other_symmetry_entity_ids}')
    logger.info(f'Found other group ids which have an unsolvable assembly: {remove_other_groups}')
    # Remove any that weren't solved
    for group_id in remove_other_groups:
        other_symmetry_group_to_members.pop(group_id)
    # other_symmetry_returns = \
    #     nanohedra_building_blocks_query(query_params.symmetry, query_params.lower_length, query_params.upper_length,
    #                                     limit_by_groups=thermophilic_group_ids)
    # logger.debug(f'other_symmetry_returns: {other_symmetry_returns}')
    # other_symmetry_return_ids = parse_pdb_response_for_ids(other_symmetry_returns)
    # logger.info(f'Found all other return_ids: {other_symmetry_return_ids}')
    final_building_blocks = top_thermophilic_entity_ids + top_other_symmetry_entity_ids
    final_building_blocks = sorted(final_building_blocks)
    logger.info(f'Found all EntityIDs: {final_building_blocks}')

    utils.io_save(final_building_blocks)

# Cleaned ids:
# '1aa0', '1p9h', '1qu1', '1sg2', '1wt6', '1yq8', '2ed6', '2edm', '2fkk', '2ibl', '2qih', '2vmd', '2vnl',
# '2vrs', '2wn3', '2xdj', '2xqh', '3gud', '3gw6', '3emf', '3emi', '3pqi', '3qr7', '3qr8', '3wit', '3wpr', '3wqa',
# '4cjd', '4jdo', '4jj2', '4lgo', '4n23', '4nkj', '4ru3', '4ufq', '4ypc', '5apy', '5apz', '5cj9', '5m9f', '6eup',
# '6m1v', '6mic', '6nz3', '6qp4', '6wko', '6wxa', '7o97', '7oah', '7qrj', '1cun', '2pls', '2wl7', '2xdh', '3l0a',
# '3lyb', '3px1', '3sb7', '5n2c', '5z25', '6ozb', '6wp2', '7kgc', '4kmb', '4n3v'
