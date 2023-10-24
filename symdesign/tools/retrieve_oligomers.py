import argparse
import logging

from symdesign import flags, utils
from symdesign.resources.query.pdb import QueryParams, nanohedra_building_blocks_query, parse_pdb_response_for_ids, \
    solve_confirmed_assemblies

logger = logging.getLogger(__name__)
logger.setLevel(20)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for symmetric oligomers\n')
    parser.add_argument(*flags.symmetry_args, required=True, type=str.upper,
                        help='What is the schoenflies symbol of the desired oligomeric\n'
                             "symmetry? For asymmetric, provide the argument as 'C1'")
    #                   **flags.symmetry_kwargs)
    parser.add_argument('--lower-length', default=80, type=int,
                        help='How many amino acids is the shortest allowed protein?')
    parser.add_argument('--upper-length', default=300, type=int,
                        help='How many amino acids is the longest allowed protein?')
    args = parser.parse_args()

    # Generate parameters to perform retrieval of ids
    query_params = QueryParams(symmetry=args.symmetry, lower_length=args.lower_length, upper_length=args.upper_length)

    # Get thermophilic representative group ids and representatives
    # Grab the groups
    thermophilic_groups = \
        nanohedra_building_blocks_query(query_params.symmetry, query_params.lower_length, query_params.upper_length,
                                        thermophile=True, groups=True)
    if thermophilic_groups:
        thermophilic_group_ids = parse_pdb_response_for_ids(thermophilic_groups, groups=True)
        grouped_thermophilic_entity_ids = [parse_pdb_response_for_ids(group)
                                           for group in thermophilic_groups['group_set']]
        logger.debug(f'Found the thermophilic return ids: {grouped_thermophilic_entity_ids}')
    else:
        thermophilic_group_ids = []
        grouped_thermophilic_entity_ids = []

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
    # Then use the identified groups to limit an additional search
    other_symmetry_groups = \
        nanohedra_building_blocks_query(query_params.symmetry, query_params.lower_length, query_params.upper_length,
                                        groups=True, limit_by_groups=thermophilic_group_to_members.keys())
    if other_symmetry_groups:
        other_symmetry_group_ids = parse_pdb_response_for_ids(other_symmetry_groups, groups=True)
        grouped_other_symmetry_entity_ids = \
            [parse_pdb_response_for_ids(group) for group in other_symmetry_groups['group_set']]
    else:
        other_symmetry_group_ids = []
        grouped_other_symmetry_entity_ids = []

    other_symmetry_group_to_members = dict(zip(other_symmetry_group_ids, grouped_other_symmetry_entity_ids))
    top_other_symmetry_entity_ids, remove_other_groups = \
        solve_confirmed_assemblies(query_params, other_symmetry_group_to_members)
    top_other_symmetry_entity_ids = sorted(top_other_symmetry_entity_ids)
    logger.info(f'Found the top other return ids: {top_other_symmetry_entity_ids}')
    logger.info(f'Found other group ids which have an unsolvable assembly: {remove_other_groups}')
    # Remove any that weren't solved
    for group_id in remove_other_groups:
        other_symmetry_group_to_members.pop(group_id)

    final_building_blocks = sorted(top_thermophilic_entity_ids + top_other_symmetry_entity_ids)
    logger.info(f'Found all EntityIDs: {final_building_blocks}')

    utils.io_save(final_building_blocks)
