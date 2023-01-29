from __future__ import annotations

import argparse
import logging
import os
import time
from copy import deepcopy
from json import dumps, load
from typing import Any, Iterable

import requests

from symdesign import utils
from .utils import input_string, confirmation_string, bool_d, validate_input, invalid_string, \
    header_string, format_string, connection_exception_handler, UKB, GB
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
# General Formatting
user_input_format = f'\n{format_string.format("Option", "Description")}\n%s'
additional_input_string = f'\nWould you like to add another%s? [y/n]{input_string}'
instance_d = {'string': str, 'integer': int, 'number': float, 'date': str}

# Websites
# pdb_query_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
pdb_query_url = 'https://search.rcsb.org/rcsbsearch/v2/query'  # Use started 8/25/22
# v TODO use inspect webpage to pull the entire dictionary of possibilities
pdb_advanced_search_url = 'http://www.rcsb.org/search/advanced'
pdb_rest_url = 'http://data.rcsb.org/rest/v1/core/'  # uniprot/  # 1AB3/1'
attribute_url = 'https://search.rcsb.org/search-attributes.html'
attribute_metadata_schema_json = 'https://search.rcsb.org/rcsbsearch/v1/metadata/schema'
# additional resources for formatting the schema may be found here - https://data.rcsb.org/#data-schema

# PDB Query formatting
attribute_prefix = 'rcsb_'
# uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'
request_types = {'group': 'Results must satisfy a group of requirements', 'terminal': 'A specific requirement'}

# Query formatting requirements
request_type_examples = {'group': '"type": "group", "logical_operator": "and", "nodes": [%s]',
                         'terminal': '"type": "terminal", "service": "text", "parameters":{%s}'}
# replacing the group %s with a request_type_examples['terminal'], or the terminal %s with a parameter_query
return_types = {'entry': "PDB ID's",
                'polymer_entity': "PDB ID's appended with entityID - '[pdb_id]_[entity_id]' for macromolecules",
                'non_polymer_entity': "PDB ID's appended with entityID - '[pdb_id]_[entity_id]' for "
                                      'non-polymers',
                'polymer_instance': "PDB ID's appended with asymID's (chains) - '[pdb_id]_[asym_id]'",
                'assembly': "PDB ID's appended with biological assemblyID's - '[pdb_id]-[assembly_id]'"}
services = {'text': 'linguistic searches against textual annotations associated with PDB structures',
            'sequence': 'employs the MMseqs2 software and performs fast sequence matching searches (BLAST-like)'
                        ' based on a user-provided FASTA sequence',
            'seqmotif': 'Performs short motif searches against nucleotide or protein sequences',
            'structure': 'Search global 3D shape of assemblies or chains of a given entry, using a BioZernike '
                         'descriptor strategy',
            'strucmotif': 'Performs structural motif searches on all available PDB structures',
            'chemical': 'Queries small-molecule constituents of PDB structures based on chemical formula and '
                        'chemical structure'}
operators = {'exact_match', 'greater', 'less', 'greater_or_equal', 'less_or_equal', 'equals', 'contains_words',
             'contains_phrase', 'range', 'exists', 'in', 'range', 'range_closed'}
group_operators = {'and', 'or'}
attributes = {'symmetry': 'rcsb_struct_symmetry.symbol', 'experimental_method': 'exptl.method',
              'resolution': 'rcsb_entry_info.resolution_combined',
              'accession_id': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.'
                              'database_accession',
              'accession_db': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.'
                              'database_name',
              'organism': 'rcsb_entity_source_organism.taxonomy_lineage.name'}
search_term = 'ELECTRON MICROSCOPY'  # For example

parameter_query = {'attribute': attributes, 'operator': operators, 'value': search_term}
query = {"query": {"type": "terminal", "service": "text", 'parameters': parameter_query}, "return_type": "entry"}
# Example Query in JSON format:
example_uniprot = {"query": {"type": "group", "logical_operator": "and", "nodes": [
                                {"type": "terminal", "service": "text", "parameters":
                                    {"operator": "exact_match", "value": "P69905",
                                     "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence"
                                                  "_identifiers.database_accession"}},
                                {"type": "terminal", "service": "text", "parameters":
                                    {"operator": "exact_match", "value": "UniProt",
                                     "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence"
                                                  "_identifiers.database_name"}}
                                ]},
                   "return_type": "polymer_entity"}
# The useful information comes out of the ['result_set'] for result['identifier'] in result_set
example_uniprot_return = {"query_id": "057be33f-e4a1-4912-8d30-673dd0326984", "result_type": "polymer_entity",
                          "total_count": 289, "explain_meta_data": {"total_timing": 8, "terminal_node_timings": {"5238":
                                                                                                                 4}},
                          "result_set": [
                              # the entry identifier specified, rank on the results list
                              {"identifier": "1XZU_1", "score": 1.0, "services": [
                                  {"service_type" : "text", "nodes": [{"node_id": 5238,
                                                                       "original_score": 137.36669921875,
                                                                       "norm_score": 1.0}]
                                   }]},
                              {"identifier": "6NBC_1", "score": 1.0, "services": [
                                  {"service_type": "text", "nodes": [{"node_id": 5238,
                                                                      "original_score": 137.36669921875,
                                                                      "norm_score": 1.0}]
                                   }]}]
                          }


def retrieve_entity_id_by_sequence(sequence: str) -> str | None:
    """From a given sequence, retrieve the top matching Entity ID from the PDB API

    Args:
        sequence: The sequence used to query for the EntityID
    Returns:
        '1ABC_1'
    """
    matching_entities = find_matching_entities_by_sequence(sequence, all_matching=False)
    if matching_entities:
        logger.debug(f'Sequence search found the matching EntityIDs: {", ".join(matching_entities)}')
        return matching_entities[0]
    else:
        return None


def find_matching_entities_by_sequence(sequence: str = None, return_id: str = 'polymer_entity', **kwargs)\
        -> list[str] | None:
    """Search the PDB for matching IDs given a sequence and a return_type. Pass all_matching=False to retrieve the top
    10 IDs, otherwise return all IDs

    Args:
        sequence: The sequence used to query for EntityID's
        return_id: The type of value to return where the acceptable values are in return_types
    Returns:
        The EntityID's matching the sequence
    """
    if return_id not in return_types:
        raise KeyError(f"The specified return type '{return_id}' isn't supported. Viable types include "
                       f"{', '.join(return_types)}")
    logger.debug(f'Using the default sequence similarity parameters: '
                 f'{", ".join(f"{k}: {v}" for k, v in default_sequence_values.items())}')
    sequence_query = generate_terminal_group(service='sequence', sequence=sequence)
    sequence_query_results = query_pdb(generate_query(sequence_query, return_id=return_id, sequence=sequence, **kwargs))
    if sequence_query_results:
        return parse_pdb_response_for_ids(sequence_query_results)
    else:
        logger.warning(f"Sequence wasn't found by the PDB API:\n{sequence}")
        return None  # [None]


def parse_pdb_response_for_ids(response: dict[str, dict[str, str]]) -> list[str]:
    # logger.debug(f'Response contains the results: {response["result_set"]}')
    return [result['identifier'] for result in response['result_set']]


def parse_pdb_response_for_score(response: dict[str, dict[str, str]]) -> list[float] | list:
    if response:
        return list(map(float, (result['score'] for result in response['result_set'])))
    else:
        return []


# Todo move to GraphQL
#  For target queries of all the information needed from a single Entity
# https://data.rcsb.org/graphql?query=%7B%0A%20%20entries(entry_ids:%20%5B%224HHB%22,%20%2212CA%22,%20%223PQR%22%5D)%20%7B%0A%20%20%20%20exptl%20%7B%0A%20%20%20%20%20%20method%0A%20%20%20%20%7D%0A%20%20%7D%0A%7D
# Potential route of incorporating into query_pdb()
# pdb_graphql_url = https://data.rcsb.org/graphql
# params = {query:{
#             entries(entry_ids:["4HHB", "12CA", "3PQR"]) {
#               exptl {
#                 method
#               }
#             }
#           }
# query_response = requests.get(pdb_graphql_url, params=params)


def query_pdb(query_) -> dict[str, Any] | None:
    """Take a JSON formatted PDB API query and return the results

    PDB response can look like:
    {'query_id': 'ecc736b3-f19c-4a54-a5d6-3db58ce6520b',
     'result_type': 'entry',
    'total_count': 104,
    'result_set': [{'identifier': '4A73', 'score': 1.0,
                    'services': [{'service_type': 'text', 'nodes': [{'node_id': 11198,
                                                                     'original_score': 222.23667907714844,
                                                                     'norm_score': 1.0}]}]},
                   {'identifier': '5UCQ', 'score': 1.0,
                    'services': [{'service_type': 'text', 'nodes': [{'node_id': 11198,
                                                                     'original_score': 222.23667907714844,
                                                                     'norm_score': 1.0}]}]},
                   {'identifier': '6P3L', 'score': 1.0,
                    'services': [{'service_type': 'text', 'nodes': [{'node_id': 11198,
                                                                     'original_score': 222.23667907714844,
                                                                     'norm_score': 1.0}]}]},
                    ...
                  ]
    }
    """
    query_response = None
    iteration = 0
    while True:
        try:
            query_response = requests.get(pdb_query_url, params={'json': dumps(query_)})
            # logger.debug(f'Found the PDB query with url: {query_response.url}')
            if query_response.status_code == 200:
                return query_response.json()
            elif query_response.status_code == 204:
                logger.warning('No response was returned. Your query likely found no matches!')
                break
            elif query_response.status_code == 429:
                logger.debug('Too many requests, pausing momentarily')
                time.sleep(2)
            else:
                logger.debug(f'Your query returned an unrecognized status code ({query_response.status_code})')
                time.sleep(1)
                iteration += 1
        except requests.exceptions.ConnectionError:
            logger.debug('Requests ran into a connection error')
            time.sleep(1)
            iteration += 1

        if iteration > 5:
            logger.error('The maximum number of resource fetch attempts was made with no resolution. '
                         f'Offending request {getattr(query_response, "url", pdb_query_url)}')  # Todo format url
            break
            # raise DesignError('The maximum number of resource fetch attempts was made with no resolution. '
            #                   'Offending request %s' % getattr(query_response, 'url', pdb_query_url))
    return None


default_sequence_values = {'evalue_cutoff': 0.0001, 'identity_cutoff': 0.5}


# Todo set up by kwargs
def generate_parameters(attribute=None, operator=None, negation=None, value=None, sequence=None, **kwargs):
    if sequence:  # scaled identity_cutoff to 50% due to scoring function and E-value usage
        return {**default_sequence_values, 'sequence_type': 'protein', 'value': sequence}
        # 'target': 'pdb_protein_sequence',
    else:
        return {'attribute': attribute, 'operator': operator, 'negation': negation, 'value': value}


def pdb_id_matching_uniprot_id(uniprot_id, return_id: str = 'polymer_entity') -> list[str]:
    """Find all matching PDB entries from a specified UniProt ID and specific return ID

    Args:
        uniprot_id: The UniProt ID of interest
        return_id: The type of value to return where the acceptable values are in return_types
    Returns:
        The list of matching IDs
    """
    if return_id not in return_types:
        raise KeyError('The specified return type "%s" is not supported. Viable types include %s'
                       % (return_id, ', '.join(return_types)))
    database = {'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name',
                'negation': False, 'operator': 'exact_match', 'value': 'UniProt'}
    accession = \
        {'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',
         'negation': False, 'operator': 'in', 'value': [uniprot_id]}

    uniprot_query = [generate_terminal_group('text', **database), generate_terminal_group('text', **accession)]
    final_query = generate_group('and', uniprot_query)
    search_query = generate_query(final_query, return_id=return_id)
    response_d = query_pdb(search_query)

    return parse_pdb_response_for_ids(response_d)


def generate_group(operation, child_groups):
    return {'type': 'group', 'logical_operator': operation, 'nodes': list(child_groups)}


def generate_terminal_group(service, *parameter_args, **kwargs):
    return {'type': 'terminal', 'service': service, 'parameters': generate_parameters(**kwargs)}


# Used to return the uniprot_accession number
sequence_request_options = {
    'group_by_return_type': 'representatives',
    'group_by': {
        'aggregation_method': 'matching_uniprot_accession',  # 'sequence_identity'
        'ranking_criteria_type': {
            'sort_by': 'rcsb_entry_info.resolution_combined',
            'direction': 'asc'
        }
        # "similarity_cutoff": 95  # for 'sequence_identity'
    }
}


def generate_query(search: dict, return_id: str = 'entry', sequence: bool = False, all_matching: bool = True) \
        -> dict[str, dict | str]:
    """Format a PDB query with the specific return type and parameters affecting search results

    Args:
        search: Contains the key, value pairs in accordance with groups and terminal groups
        return_id: The type of ID that should be returned
        sequence: Whether the query generated is a sequence type query
        all_matching: Whether to get all matching IDs
    Returns:
        The formatted query to be sent via HTTP GET
    """
    if return_id not in return_types:
        raise KeyError(f"The specified return type '{return_id}' isn't supported. Viable types include "
                       f"{', '.join(return_types)}")

    query_d = {'query': search, 'return_type': return_id}
    request_options = {'results_content_type': ['experimental'],  # "computational" for Alphafold
                       'sort': [{
                           'sort_by': 'score',
                           'direction': 'desc'}],
                       'scoring_strategy': 'combined'
                       }
    if sequence:
        request_options.update(sequence_request_options)

    if all_matching:
        request_options.update({'return_all_hits': True})

    query_d.update({'request_options': request_options})

    return query_d


def retrieve_pdb_entries_by_advanced_query(save: bool = True, return_results: bool = True,
                                           force_schema_update: bool = False, entity: bool = False,
                                           assembly: bool = False, chain: bool = False, entry: bool = False, **kwargs) \
        -> str | list | None:
    """

    Args:
        save:
        return_results:
        force_schema_update:
        entity:
        assembly:
        chain:
        entry:
    Returns:

    """
    # {attribute: {'dtype': 'string', 'description': 'XYZ', 'operators': {'equals',}, 'choices': []}, ...}

    def search_schema(term):
        return [(key, schema[key]['description']) for key in schema if schema[key]['description'] and
                term.lower() in schema[key]['description'].lower()]

    def make_groups(*args, recursive_depth=0):
        # Todo remove ^ * expression?
        # on initialization have [{}, {}, ...]
        #  was [(), (), ...]
        # on recursion get (terminal_queries, grouping,
        terminal_queries = args[0]
        work_on_group = args[recursive_depth]
        all_grouping_indices = {i for i in range(1, len(work_on_group) + 1)}

        group_introduction = '\n%s\n' \
                             'Because you have %d search queries, you need to combine these to a total search strategy'\
                             '. This is accomplished by grouping your search queries together using the operations %s.'\
                             ' You must eventually group all queries into a single logical operation. ' \
                             '\nIf you have multiple groups, you will need to group those groups, so on and so forth.' \
                             '\nIndicate your group selections with a space separated list! You will choose the group '\
                             'operation to combine this list afterwards.\nFollow prior prompts if you need a reminder '\
                             'of how group#\'s relate to query#\'s' \
                             % (header_string % 'Grouping Instructions', len(terminal_queries), group_operators)
        group_grouping_intro = '\nGroups remain, you must group groups as before.'
        group_inquiry_string = '\nWhich of these (identified by #) would you like to combine into a group?%s' % \
                               input_string
        group_specification_string = 'You specified "%s" as a single group.'
        group_logic_string = '\nWhat group operator %s would you like for this group?%s' % (group_operators,
                                                                                            input_string)

        available_query_string = '\nYour available queries are:\n%s\n' % \
                                 '\n'.join(query_display_string % (query_num, service.upper(), attribute,
                                                                   'NOT ' if negate else '', operator.upper(), value)
                                           for query_num, (service, attribute, operator, negate, value)
                                           in enumerate(list(terminal_queries.values()), 1))

        if recursive_depth == 0:
            intro_string = group_introduction
            available_entity_string = available_query_string
        else:
            intro_string = group_grouping_intro
            available_entity_string = '\nYour available groups are:\n%s\n' % \
                                      '\n'.join(f'\tGroup Group #{i}{format_string.format(*group)}'
                                                for i, group in enumerate(list(work_on_group.values()), 1))

        print(intro_string)  # provide an introduction
        print(available_entity_string)  # display available entities which switch between guery and group...

        selected_grouping_indices = deepcopy(all_grouping_indices)
        groupings = []
        while len(selected_grouping_indices) > 1:  # check if more work needs to be done
            while True:  # ensure grouping input is viable
                while True:
                    grouping = set(map(int, input(group_inquiry_string).split()))  # get new grouping
                    # error on isdigit() ^
                    if len(grouping) > 1:
                        break
                    else:
                        print('More than one group is required. Your group "%s" is invalid' % grouping)
                while True:
                    confirm = input('%s\n%s' % (group_specification_string % grouping, confirmation_string))
                    if confirm.lower() in bool_d:
                        break
                    else:
                        print('%s %s is not a valid choice!' % (invalid_string, confirm))

                if bool_d[confirmation.lower()] or confirmation.isspace():  # confirm that grouping is as specified
                    while True:  # check if logic input is viable
                        group_logic = input(group_logic_string).lower()
                        if group_logic in group_operators:
                            break
                        else:
                            print(invalid_string)
                    groupings.append((grouping, group_logic))
                    break

            # remove specified from the pool of available until all are gone
            selected_grouping_indices = selected_grouping_indices.difference(grouping)

        if len(selected_grouping_indices) > 0:
            groupings.append((selected_grouping_indices, 'and'))  # When only 1 remains, automatically add 'and'
            # Todo test logic of and with one group?

        args.extend((groupings,))  # now [{} {}, ..., ([(grouping, group_logic), (), ...])
        # once all groupings are grouped, recurse
        if len(groupings) > 1:
            # todo without the return call, the stack never comes back to update args?
            make_groups(*args, recursive_depth=recursive_depth + 1)

        return list(args)  # list() may be unnecessary

    # Start the user input routine -------------------------------------------------------------------------------------
    schema = get_rcsb_metadata_schema(force_update=force_schema_update)
    print(f'\n{header_string % "PDB API Advanced Query"}\n'
          f'This prompt will walk you through generating an advanced search query and retrieving the matching '
          "set of entry ID's from the PDB. This automatically parses the ID's of interest for downstream use, which "
          'can save you some headache. If you want to take advantage of the PDB webpage GUI to perform the advanced '
          f'search, visit:\n\t{pdb_advanced_search_url}\nThen enter "json" in the prompt below and follow those '
          'instructions.\n\n'
          'Otherwise, this command line prompt takes advantage of the same GUI functionality. If you have a '
          'search specified from a prior query that you want to process again, using "json" will be useful as well. '
          'To proceed with the command line search just hit "Enter"')
    program_start = input(input_string)
    if program_start.lower() == 'json':
        if entity:
            return_type = 'Polymer Entities'  # 'polymer_entity'
        elif assembly:
            return_type = 'Assemblies'  # 'assembly'
        elif chain:
            return_type = 'Polymer Entities'  # This isn't available on web GUI -> 'polymer_instance'
        elif entry:
            return_type = 'Structures'  # 'entry'
        else:
            return_type = 'Structures'  # 'entry'

        return_type_prompt = f'At the bottom left of the dialog, there is a drop down menu next to "Return". ' \
                             f'Choose {return_type}'
        print('DETAILS: To save time formatting and immediately move to your design pipeline, build your Query with the'
              ' PDB webpage GUI, then save the resulting JSON text to a file. To do this, first build your full query '
              f'on the advanced search page, {return_type_prompt} then click the Search button (magnifying glass icon).'
              ' After the page loads, a new section of the search page should appear above the Advanced Search Query '
              'Builder dialog. There, click the JSON|->| button to open a new page with an automatically built JSON '
              'representation of your query. Save the entirety of this JSON formatted query to a file to return your '
              "chosen ID's\n")
        # ('Paste your JSON object below. IMPORTANT select from the opening \'{\' to '
        #  '\'"return_type": "entry"\' and paste. Before hitting enter, add a closing \'}\'. This hack '
        #  'ensures ALL results are retrieved with no sorting or pagination applied\n\n%s' %
        #  input_string)
        prior_query = input(f'Please specify the path where the JSON query file is located{input_string}')
        while not os.path.exists(prior_query):
            prior_query = input(f"The specified path '{prior_query}' doesn't exist! Please try again{input_string}")

        with open(prior_query, 'r') as f:
            json_input = load(f)

        # remove any paginate instructions from the json_input
        json_input['request_options'].pop('paginate', None)
        # if all_matching:
        # Ensure we get all matching
        json_input['request_options'].update({'return_all_hits': True})
        response_d = query_pdb(json_input)
    # elif program_start.lower() == 'previous':
    #     while True:
    #         prior_query = input('Please specify the path where the search file is located%s' % input_string)
    #         if os.path.exists(prior_query):
    #             with open(prior_query, 'r') as f:
    #                 search_query = loads(f.readlines())
    #         else:
    #             print('The specified path \'%s\' doesn\'t exist! Please try again.' % prior_query)
    else:
        if entity:
            return_type = 'polymer_entity'
        elif assembly:
            return_type = 'assembly'
        elif chain:
            return_type = 'polymer_instance'
        elif entry:
            return_type = 'entry'
        else:
            return_identifier_string = '\nFor each set of options, choose the option from the first column for the ' \
                                       'description in the second.\nWhat type of identifier do you want to search the '\
                                       f'PDB for?%s{input_string}' % user_input_format % \
                                       '\n'.join(format_string.format(*item) for item in return_types.items())
            return_type = validate_input(return_identifier_string, return_types)

        terminal_group_queries = []
        # terminal_group_queries = {}
        increment = 1
        while True:
            # TODO only text search is available now
            # query_builder_service_string = '\nWhat type of search method would you like to use?%s%s' % \
            #                                (user_input_format % '\n'.join(format_string % item
            #                                                               for item in services.items()), input_string)
            query_builder_attribute_string = \
                '\nWhat type of attribute would you like to use? Examples include:\n\t%s\n\n' \
                f'For a more thorough list indicate "s" for search.\nAlternatively, you can browse {attribute_url}\n' \
                f'Ensure that your spelling is exact if you want your query to succeed!{input_string}' % \
                '\n\t'.join(utils.pretty_format_table(attributes.items(), header=('Option', 'Description')))
            query_builder_operator_string = '\nWhat operator would you like to use?\nPossible operators include:' \
                                            '\n\t%s\nIf you would like to negate the operator, on input type "not" ' \
                                            f'after your selection. Ex: equals not{input_string}'
            query_builder_value_string = '\nWhat value should be %s? Required type is: %s.%s%s'
            query_display_string = 'Query #%d: Search the PDB by "%s" for "%s" attributes "%s%s" "%s".'

            while True:  # start the query builder routine
                while True:
                    # service = input(query_builder_service_string)
                    service = 'text'  # TODO
                    if service in services:
                        break
                    else:
                        print(invalid_string)

                # {attribute: {'dtype': 'string', 'description': 'XYZ', 'operators': {'equals',}, 'choices': []}, ...}
                while True:
                    attribute = input(query_builder_attribute_string)
                    while attribute.lower() == 's':  # If the user would like to search all possible
                        search_term = input('What term would you like to search?%s' % input_string)
                        attribute = input(f'Found the following instances of "{search_term.upper()}":\n%s\nWhich option'
                                          f' are you interested in? Enter "s" to repeat search.{input_string}' %
                                          user_input_format %
                                          '\n'.join(format_string.format(*key_description_pair) for key_description_pair
                                                    in search_schema(search_term)))
                        if attribute != 's':
                            break
                    if attribute in schema:  # Confirm the user wants to go forward with this
                        break
                    else:
                        print(f'***ERROR: {attribute} was not found in PDB schema***')
                        # while True:  # confirm that the confirmation input is valid
                        #     confirmation = input('ERROR: %s was not found in PDB schema! If you proceed, your search is'
                        #                          ' almost certain to fail.\nProceed anyway? [y/n]%s' %
                        #                          (attribute, input_string))
                        #     if confirmation.lower() in bool_d:
                        #         break
                        #     else:
                        #         print('%s %s is not a valid choice!' % invalid_string, confirmation)
                        # if bool_d[confirmation.lower()] or confirmation.isspace():  # break the attribute routine on y or ''
                        #     break

                while True:  # Retrieve the operator for the search
                    while True:  # Check if the operator should be negated
                        operator = input(query_builder_operator_string % ', '.join(schema[attribute]['operators']))
                        if len(operator.split()) > 1:
                            negation = operator.split()[1]
                            operator = operator.split()[0]
                            if negation.lower() == 'not':  # Can negate any search
                                negate = True
                                break
                            else:
                                print(f"{invalid_string} {negation} is not a recognized negation!\n "
                                      f"Try '{operator} not' instead or remove extra input")
                        else:
                            negate = False
                            break
                    if operator in schema[attribute]['operators']:
                        break
                    else:
                        print(f"{invalid_string} {operator} isn't a valid operator")

                op_in = True
                while op_in:  # Check if operator is 'in'
                    if operator == 'in':
                        print("\nThe 'in' operator can take multiple values. If you want multiple values, specify "
                              'each as a separate input')
                    else:
                        op_in = False

                    while True:  # Retrieve the value for the search
                        value = input(query_builder_value_string % (operator.upper(), instance_d[schema[attribute]['dtype']]
                                                                    , ('\nPossible choices:\n\t%s' %
                                                                       ', '.join(schema[attribute]['choices'])
                                                                       if schema[attribute]['choices'] else ''),
                                                                    input_string))
                        if isinstance(value, instance_d[schema[attribute]['dtype']]):  # check if the right data type
                            break
                        else:
                            try:  # try to convert the input value to the specified type
                                value = instance_d[schema[attribute]['dtype']](value)
                                if schema[attribute]['choices']:  # if there is a choice
                                    if value in schema[attribute]['choices']:  # check if the value is in the possible choices
                                        break
                                    else:  # if not, confirm the users desire to do this
                                        while True:  # confirm that the confirmation input is valid
                                            confirmation = input('%s was not found in the possible choices: %s\nProceed'
                                                                 ' anyway? [y/n]%s' %
                                                                 (value, ', '.join(schema[attribute]['choices']),
                                                                  input_string))
                                            if confirmation.lower() in bool_d:
                                                break
                                            else:
                                                print(f"{invalid_string} {confirmation} isn't a valid choice")
                                        if bool_d[confirmation.lower()] or confirmation.isspace():  # break the value routine on y or ''
                                            break

                                else:
                                    break
                            except ValueError:  # catch any conversion issue like float('A')
                                print(f"{invalid_string} {value} isn't a valid {instance_d[schema[attribute]['dtype']]}"
                                      " value!")

                    while op_in:
                        # TODO ensure that the in parameters are spit out as a list
                        additional = input(additional_input_string % " value to your 'in' operator")
                        if additional.lower() in bool_d:
                            if bool_d[additional.lower()] or additional.isspace():
                                break  # Stop the inner 'in' check loop
                            else:
                                op_in = False  # Stop the inner and outer 'in' while loops
                        else:
                            print(f"{invalid_string} {additional} isn't a valid choice")

                while True:
                    confirmation = input('\n%s\n%s' % (query_display_string %
                                                       (increment, service.upper(), attribute,
                                                        'NOT ' if negate else '', operator.upper(), value),
                                                       confirmation_string))
                    if confirmation.lower() in bool_d:
                        break
                    else:
                        print(f"{invalid_string} {confirmation} isn't a valid choice")
                if bool_d[confirmation.lower()] or confirmation.isspace():
                    break

            # terminal_group_queries[increment] = (service, attribute, operator, negate, value)
            terminal_group_queries.append(dict(service=service, attribute=attribute, operator=operator, negate=negate,
                                               value=value))
            increment += 1
            while True:
                additional = input(additional_input_string % ' query')
                if additional.lower() in bool_d:
                    break
                else:
                    print(f"{invalid_string} {confirmation} isn't a valid choice")
            if not bool_d[additional.lower()]:  # or confirmation.isspace():
                break

        # Group terminal queries into groups if there are more than 1
        if len(terminal_group_queries) > 1:
            recursive_query_tree = make_groups(terminal_group_queries)
            # expecting return of [terminal_group_queries, bottom group hierarchy, second group hierarchy, ..., top]
        else:
            recursive_query_tree = [terminal_group_queries]
            # recursive_query_tree = (terminal_group_queries, )
        # recursive_query_tree = (queries, grouping1, grouping2, etc.)
        for i, node in enumerate(recursive_query_tree):
            if i == 0:
                recursive_query_tree[i] = {j: generate_terminal_group(**leaf) for j, leaf in enumerate(node, 1)}
                # recursive_query_tree[i] = {j: generate_terminal_group(*node[leaf]) for j, leaf in enumerate(node, 1)}

                # terminal_group_queries = {j: generate_terminal_group(*leaf) for j, leaf in enumerate(node)}
                # generate_terminal_group(service, parameter_args)
                # terminal_group_queries[increment] = generate_terminal_group(service, attribute, operator, value)
            else:
                # if i == 1:
                #     child_groups = terminal_group_queries
                #     # child_groups = [terminal_group_queries[j] for j in child_nodes]
                # else:
                #     child_groups = recursive_query_tree[i]
                # operation, child_nodes = node
                # groups = {j: generate_group(operation, child_groups) for j, leaf in enumerate(node)}

                # NOPE Subtract the k indices to ensure that the user input numbers match with python zero indexing
                # i - 1 gives the index of the previous index of the recursive_query_tree to operate on
                # k pulls the groups specified in the input out to make a list with the corresponding terminai groups
                recursive_query_tree[i] = {j: generate_group(operation, [recursive_query_tree[i - 1][k]
                                                                         for k in child_group_nums])
                                           for j, (child_group_nums, operation) in enumerate(node, 1)}
                # for k in child_group_nums}
        final_query = recursive_query_tree[-1][1]  #

        search_query = generate_query(final_query, return_id=return_type)
        response_d = query_pdb(search_query)
    logger.debug(f'The server returned:\n{response_d}')

    retrieved_ids = parse_pdb_response_for_ids(response_d)

    if save:
        utils.io_save(retrieved_ids)

    if return_results:
        return retrieved_ids


def query_pdb_by(entry: str = None, assembly_id: str = None, assembly_integer: int | str = None, entity_id: str = None,
                 entity_integer: int | str = None, chain: str = None, **kwargs) -> dict | list[list[str]]:
    """Retrieve information from the PDB API by EntryID, AssemblyID, or EntityID

    Args:
        entry: The 4 character PDB EntryID of interest
        assembly_id: The AssemblyID to query with format (1ABC-1)
        assembly_integer: The particular assembly integer to query. Must include entry as well
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entity_integer: The entity integer from the EntryID of interest
        chain: The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of interest
    Returns:
        The query result
    """
    if entry is not None:
        if len(entry) == 4:
            if entity_integer is not None:
                logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                return _get_entity_info(entry=entry, entity_integer=entity_integer)
            elif assembly_integer is not None:
                logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                return _get_assembly_info(entry=entry, assembly_integer=assembly_integer)
            else:
                logger.debug(f'Querying PDB API with {entry}')
                data = _get_entry_info(entry)
                if chain:
                    integer = None
                    for entity_idx, chains in data.get('entity').items():
                        if chain in chains:
                            integer = entity_idx
                            break
                    if integer:
                        logger.debug(f'Querying PDB API with {entry}_{integer}')
                        return _get_entity_info(entry=entry, entity_integer=integer)
                    else:
                        raise KeyError(f'No chain "{chain}" found in PDB ID {entry}. Possible chains '
                                       f'{", ".join(ch for chns in data.get("entity", {}).items() for ch in chns)}')
                else:
                    return data
        else:
            logger.debug(f"EntryID '{entry}' isn't the required format and will not be found with the PDB API")
    elif assembly_id is not None:
        entry, assembly_integer, *extra = assembly_id.split('-')
        if not extra and len(entry) == 4:
            logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
            return _get_assembly_info(entry=entry, assembly_integer=assembly_integer)

        logger.debug(f"AssemblyID '{assembly_id}' isn't the required format and will not be found with the PDB API")

    elif entity_id is not None:
        entry, entity_integer, *extra = entity_id.split('_')
        if not extra and len(entry) == 4:
            logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
            return _get_entity_info(entry=entry, entity_integer=entity_integer)

        logger.debug(f"EntityID '{entity_id}' isn't the required format and will not be found with the PDB API")
    else:
        raise RuntimeError(f'No valid arguments passed to {query_pdb_by.__name__}. Valid arguments include: '
                           f'entry, assembly_id, assembly_integer, entity_id, entity_integer, chain')


# def _query_*_id(*_id: str = None, entry: str = None, *_integer: str | int = None) -> dict[str, Any] | None:
# Ex. entry = '4atz'
# ex. assembly = 1
# url = https://data.rcsb.org/rest/v1/core/assembly/4atz/1
# ex. entity
# url = https://data.rcsb.org/rest/v1/core/polymer_entity/4atz/1
# ex. chain
# url = https://data.rcsb.org/rest/v1/core/polymer_entity_instance/4atz/A


def query_assembly_id(assembly_id: str = None, entry: str = None, assembly_integer: str | int = None) -> \
        requests.Response | None:
    """Retrieve PDB AssemblyID information from the PDB API. More info at http://data.rcsb.org/#data-api

    For all method types the following keys are available:
    {'rcsb_polymer_entity_annotation', 'entity_poly', 'rcsb_polymer_entity', 'entity_src_gen',
     'rcsb_polymer_entity_feature_summary', 'rcsb_polymer_entity_align', 'rcsb_id', 'rcsb_cluster_membership',
     'rcsb_polymer_entity_container_identifiers', 'rcsb_entity_host_organism', 'rcsb_latest_revision',
     'rcsb_entity_source_organism'}
    NMR only - {'rcsb_polymer_entity_feature'}
    EM only - set()
    X-ray_only_keys - {'rcsb_cluster_flexibility'}

    Args:
        assembly_id: The AssemblyID to query with format (1ABC-1)
        entry: The 4 character PDB EntryID of interest
        assembly_integer: The particular assembly integer to query. Must include entry as well
    Returns:
        The assembly information according to the PDB
    """
    if assembly_id:
        entry, assembly_integer, *_ = assembly_id.split('-')  # assume that this was passed correctly

    if entry and assembly_integer:
        return connection_exception_handler(f'http://data.rcsb.org/rest/v1/core/assembly/{entry}/{assembly_integer}')


def _get_assembly_info(entry: str = None, assembly_integer: int = None, assembly_id: str = None) -> \
        list[list[str]] | list:
    """Retrieve information on the assembly for a particular entry from the PDB API

    Args:
        entry: The pdb code of interest
        assembly_integer: The particular assembly number to query
        assembly_id: The AssemblyID to query with format (1ABC-1)
    Returns:
        The mapped entity number to the chain ID's in the assembly. Ex: {1: ['A', 'A', 'A', ...]}
    """
    assembly_request = query_assembly_id(assembly_id=assembly_id, entry=entry, assembly_integer=assembly_integer)
    if not assembly_request:
        return []

    return parse_assembly_json(assembly_request.json())


def parse_assembly_json(assembly_json: dict[str, Any]) -> list[list[str]]:
    """For a PDB API AssemblyID, parse the associated 'clustered' chains

    Args:
        assembly_json: The json type dictionary returned from requests.Response.json()
    Returns:
        The chain ID's which cluster in the assembly -
        Ex: [['A', 'A', 'A', ...], ...]
    """
    entity_clustered_chains = []
    if not assembly_json:
        return entity_clustered_chains

    for symmetry in assembly_json['rcsb_struct_symmetry']:
        # for symmetry in symmetries:  # [{}, ...]
        # symmetry contains:
        # {symbol: "O", type: 'Octahedral, stoichiometry: [], oligomeric_state: "Homo 24-mer", clusters: [],
        #  rotation_axes: [], kind: "Global Symmetry"}
        for cluster in symmetry['clusters']:  # [{}, ...]
            # CLUSTER_IDX is not a mapping to entity index...
            # cluster contains:
            # {members: [], avg_rmsd: 5.219512137974998e-14} which indicates how similar each member in the cluster is
            # cluster_members = []
            # for member in cluster['members']:  # [{}, ...]
            #     # member contains:
            #     # {asym_id: "A", pdbx_struct_oper_list_ids: []}
            #     cluster_members.append(member.get('asym_id'))
            entity_clustered_chains.append([member.get('asym_id') for member in cluster['members']])

    return entity_clustered_chains


def query_entry_id(entry: str = None) -> requests.Response | None:
    """Fetches the JSON object for the EntryID from the PDB API

    The following information is returned:
    All methods (SOLUTION NMR, ELECTRON MICROSCOPY, X-RAY DIFFRACTION) have the following keys:
    {'rcsb_primary_citation', 'pdbx_vrpt_summary', 'pdbx_audit_revision_history', 'audit_author',
     'pdbx_database_status', 'rcsb_id', 'pdbx_audit_revision_details', 'struct_keywords',
     'rcsb_entry_container_identifiers', 'entry', 'rcsb_entry_info', 'struct', 'citation', 'exptl',
     'rcsb_accession_info'}
    EM only keys:
    {'em3d_fitting', 'em3d_fitting_list', 'em_image_recording', 'em_specimen', 'em_software', 'em_entity_assembly',
     'em_vitrification', 'em_single_particle_entity', 'em3d_reconstruction', 'em_experiment', 'pdbx_audit_support',
     'em_imaging', 'em_ctf_correction'}
    Xray only keys:
    {'diffrn_radiation', 'cell', 'reflns', 'diffrn', 'software', 'refine_hist', 'diffrn_source', 'exptl_crystal',
     'symmetry', 'diffrn_detector', 'refine', 'reflns_shell', 'exptl_crystal_grow'}
    NMR only keys:
    {'pdbx_nmr_exptl', 'pdbx_audit_revision_item', 'pdbx_audit_revision_category', 'pdbx_nmr_spectrometer',
     'pdbx_nmr_refine', 'pdbx_nmr_representative', 'pdbx_nmr_software', 'pdbx_nmr_exptl_sample_conditions',
     'pdbx_nmr_ensemble'}

    entry_json['rcsb_entry_info'] = \
        {'assembly_count': 1, 'branched_entity_count': 0, 'cis_peptide_count': 3, 'deposited_atom_count': 8492,
        'deposited_model_count': 1, 'deposited_modeled_polymer_monomer_count': 989,
        'deposited_nonpolymer_entity_instance_count': 0, 'deposited_polymer_entity_instance_count': 6,
        'deposited_polymer_monomer_count': 1065, 'deposited_solvent_atom_count': 735,
        'deposited_unmodeled_polymer_monomer_count': 76, 'diffrn_radiation_wavelength_maximum': 0.9797,
        'diffrn_radiation_wavelength_minimum': 0.9797, 'disulfide_bond_count': 0, 'entity_count': 3,
        'experimental_method': 'X-ray', 'experimental_method_count': 1, 'inter_mol_covalent_bond_count': 0,
        'inter_mol_metalic_bond_count': 0, 'molecular_weight': 115.09, 'na_polymer_entity_types': 'Other',
        'nonpolymer_entity_count': 0, 'polymer_composition': 'heteromeric protein', 'polymer_entity_count': 2,
        'polymer_entity_count_dna': 0, 'polymer_entity_count_rna': 0, 'polymer_entity_count_nucleic_acid': 0,
        'polymer_entity_count_nucleic_acid_hybrid': 0, 'polymer_entity_count_protein': 2,
        'polymer_entity_taxonomy_count': 2, 'polymer_molecular_weight_maximum': 21.89,
        'polymer_molecular_weight_minimum': 16.47, 'polymer_monomer_count_maximum': 201,
        'polymer_monomer_count_minimum': 154, 'resolution_combined': [1.95],
        'selected_polymer_entity_types': 'Protein (only)',
        'software_programs_combined': ['PHASER', 'REFMAC', 'XDS', 'XSCALE'], 'solvent_entity_count': 1,
        'diffrn_resolution_high': {'provenance_source': 'Depositor assigned', 'value': 1.95}}

    Args:
        entry: The PDB code to search for
    Returns:
        The entry information according to the PDB
    """
    if entry:
        return connection_exception_handler(f'http://data.rcsb.org/rest/v1/core/entry/{entry}')


def _get_entry_info(entry: str = None, **kwargs) -> dict[str, Any] | None:
    """Retrieve PDB EntryID information from the PDB API. More info at http://data.rcsb.org/#data-api

    Makes 1 + num_of_entities calls to the PDB API for information

    Args:
        entry: The PDB code to search for
    Returns:
        The entry dictionary with format -
        {'entity':
            {'EntityID':
                {'chains': ['A', 'B', ...],
                 'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
                 'reference_sequence': 'MSLEHHHHHH...',
                 'thermophilicity': 1.0},
             ...}
         'method': xray,
         'res': resolution,
         'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
         }
    """
    entry_request = query_entry_id(entry)
    if not entry_request:
        return {}
    else:
        json = entry_request.json()
        return dict(entity=parse_entities_json([query_entity_id(entry=entry, entity_integer=integer).json()
                                                for integer in range(1, int(json['rcsb_entry_info']
                                                                            ['polymer_entity_count']) + 1)]),
                    **parse_entry_json(json))


def parse_entry_json(entry_json: dict[str, Any]) -> dict[str, dict]:
    """For a PDB API EntryID, parse the associated entity ID's and chains

    Args:
        entry_json: The json type dictionary returned from requests.Response.json()
    Returns:
        The structural information present in the PDB EntryID with format -
        {'method': xray,
         'res': resolution,
         'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
         }
    """
    # entry_json = entry_request.json()
    # if 'method' in entry_json['exptl'][0]:
    experimental_method = entry_json['rcsb_entry_info'].get('experimental_method')
    if experimental_method:
        # Todo make ray, diffraction
        if 'ray' in experimental_method.lower() and 'cell' in entry_json and 'symmetry' in entry_json:
            cell = entry_json['cell']
            ang_a, ang_b, ang_c = cell['angle_alpha'], cell['angle_beta'], cell['angle_gamma']
            a, b, c = cell['length_a'], cell['length_b'], cell['length_c']
            space_group = entry_json['symmetry']['space_group_name_hm']
            struct_d = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
            resolution = entry_json['rcsb_entry_info']['resolution_combined'][0]
        else:  # Todo NMR and EM
            logger.warning(f"Didn't add any useful information with the experimental method {experimental_method} as "
                           "this method hasn't been written yet")
            struct_d = {}
            resolution = None
    else:
        logger.warning('Entry has no "experimental_method" keyword')
        struct_d = {}
        resolution = None

    return {'res': resolution, 'struct': struct_d, 'method': experimental_method.lower()}


# These Taxonomic IDs are pulled from Thermobase, from the database version: ThermoBase_ver_1.0_2022
thermophilic_taxonomy_ids = [
    2320, 1927912, 2505679005, 2687453501, 35616, 2309, 638154507, 39456, 54256, 640069314, 650716080, 2262, 638154513,
    74610, 638154515, 46540, 2278, 639633053, 53953, 638154514, 52001, 650716098, 650716079, 640069326, 640427135,
    2687453162, 2636415503, 46539, 638154520, 641522657, 648028035, 2511231117, 640069332, 640753029, 54259, 114376,
    985052, 638154511, 2512047039, 94694, 1227648, 1273541, 2181, 2554235461, 2275, 54257, 70771, 12915, 2511231093,
    649633067, 1806371, 644736411, 2513237398, 187879, 2758568021, 2565956587, 639633064, 2714, 637000010, 207809,
    650716096, 2521172705, 2758568023, 2758568025, 2016361, 2511231053, 73007, 1128108, 2554235458, 650716001, 87734,
    110163, 2518645553, 277988, 2636415462, 49900, 2264, 649633040, 1702097, 2711768646, 71280, 2758568024, 477696,
    644281, 105851, 69656, 2758568026, 999894, 113653, 650716056, 646564547, 638154505, 2558860254, 84597, 2681812990,
    648028062, 227597, 646311906, 646564534, 646564573, 646564583, 139207, 155321, 90909, 2639762927, 641228483, 867916,
    867917, 2287, 2506520012, 648028003, 2049879, 1301915, 638154518, 1069083, 2287, 282676, 2588253738, 643348580,
    2758568027, 2554235387, 643348584, 643692002, 2016373, 310769, 2687453428, 2558309119, 498846, 164452, 411357,
    411358, 2513020047, 2512875013, 138903, 646311944, 41675, 646564582, 83868, 638154502, 640427150, 1008305,
    644736412, 240167, 2287, 643348540, 2283, 2524023197, 2283, 213231, 2286, 2286, 644736385, 646311964, 48383,
    1577684, 2634166507, 1302351, 2728369534, 646862347, 111955, 207243, 227598, 92642, 648028013, 54076, 2504136002,
    381751, 240166, 345592, 638154517, 1408159, 274, 649633102, 2513237177, 1170224, 282676, 2728369541, 2588253768,
    640427120, 2503508007, 373054, 213185, 312539, 56956, 2586643, 649633023, 444093, 444091, 62609, 649633024,
    642487181, 448141, 496831, 638154519, 643348542, 649633101, 936138, 649633104, 171869, 2522125074, 380763, 47304,
    2540341142, 649633025, 643692030, 276, 2510065009, 223788, 240169, 633813, 90323, 90323, 348842, 694059, 911092,
    644736370, 175696, 64160, 640753057, 680118, 79601, 1123386, 532059, 456163, 206045, 520764, 640427106, 2505119042,
    2588253728, 2507149014, 2082189, 29344, 153832, 640753026, 44754, 2704, 133270, 1123380, 238991, 651457, 1045764,
    1123388, 649633022, 2423, 548918, 641228485, 554949, 271, 1123389, 646311936, 637000060, 646564579, 106578, 412593,
    317015, 159723, 432331, 1330330, 1198630, 2506485007, 266825, 74714, 301953, 33938, 637000118, 933949, 1757, 246264,
    643348583, 1653476, 48382, 551788, 646311949, 482827, 646311904, 646564501, 643348582, 79269, 1167092, 648028061,
    291989, 159731, 1422, 1495650, 937334, 646564580, 2547829, 403219, 883812, 660294, 117503, 660294, 117503, 182900,
    407998, 1532350, 41673, 146920, 39100, 447594, 380685, 380685, 643348530, 646311962, 38322, 649633047, 649633048,
    81408, 699281, 1914305, 1462921, 1462921, 501340, 83982, 650716051, 47303, 671077, 874266, 2509362, 337094, 1936021,
    641228511, 2045011, 872965, 1430331, 33957, 314583, 650716055, 42471, 33950, 500465, 1279027, 1108801, 1016997,
    576944, 498375, 1566, 2058097, 54265, 732235, 732234, 29350, 2728369540, 161154, 2524614646, 460873, 643692050,
    171864, 648028053, 869279, 1125966, 198703, 2504643006, 160792, 44256, 2627853919, 120958, 33070, 348840,
    643348508, 944479, 447593, 1323375, 135517, 1508420, 2512564091, 1324935, 1330700, 1401547, 641522655, 1670455,
    34092, 213419, 2513237181, 693075, 648028041, 638154510, 2186, 2508501120, 1469144, 1214928, 661089, 870242,
    441616, 676965, 202604, 1266720, 1226730, 520767, 643348581, 2508501109, 1874277, 1382356, 54294, 644736325,
    646564525, 206249, 171121, 108007, 1449080, 89784, 1156395, 2519103102, 650716095, 2511231121, 153151, 1173302,
    301148, 2718218154, 659425, 515264, 1323370, 285110, 152526, 570855, 185139, 873078, 2554235408, 1930763, 86166,
    1573172, 744873, 281456, 182411, 81408, 169283, 33940, 198467, 1424653, 389480, 411923, 515440, 1560, 639633001,
    575178, 651866, 674972, 1548890, 1096341, 2224, 291048, 1565, 142584, 1339250, 1938353, 79270, 227476, 2078954,
    265546, 1245522, 409997, 454194, 1294262, 1294262, 392844, 1426, 650716039, 771855, 1560, 1750597, 1246530,
    2660238162, 643348557, 261685, 649633077, 178899, 2505119043, 53342, 278994, 1742923, 242698, 128945, 715028,
    267990, 1392640, 1392641, 1003997, 404937, 85993, 201973, 456330, 196180, 265470, 638154522, 1123352, 317575,
    311460, 265948, 150248, 47490, 581036, 482825, 42448, 1608465, 116090, 2186, 186116, 186116, 933063, 294699,
    1081799, 1868734, 247480, 1490051, 296745, 650508, 568790, 213806, 53573, 1677857, 688066, 2224, 498371, 1229250,
    190967, 16785, 301141, 665105, 46632, 638154512, 637000248, 35837, 2026184, 227924, 55505, 235932, 641228500,
    201975, 1495039, 49896, 2521172702, 55779, 94009, 2507262051, 166485, 1473580, 697197, 1236215, 58136, 2509601011,
    604332, 307124, 1571185, 637000167, 637000320, 1006576, 192387, 252246, 646564511, 360411, 129339, 203540, 1245521,
    1204385, 133453, 2540341085, 1259795, 649633026, 213224, 630405, 223786, 545536, 1325125, 1679721, 2200, 639633038,
    217041, 646273, 640753037, 927786, 640427128, 106262, 204046, 28237, 1674942, 471799, 649746, 646564546, 526227,
    573074, 1382306, 990712, 2504136000, 2504756006, 1304284, 95159, 394967, 1707527, 108328, 1248146, 1242148, 1702242,
    2493090, 114249, 40991, 1281767, 1750598, 1644118, 1644115, 1173111, 1471761, 996765, 646311963, 115335, 58112,
    2020, 1169913, 1519374, 507754, 637000306, 646564581, 51173, 109258, 112729, 564288, 436000, 230470, 353224,
    638154521, 2627854140, 76633, 79271, 1936991, 312540, 312540, 1289106, 2506783062, 610254, 1564487, 73780, 29376,
    1324956, 696763, 279810, 1035120, 285106, 29372, 5087, 2508501069, 1247960, 203471, 370323, 209537, 122960, 288966,
    232270, 649633005, 1325335, 1471, 337097, 1393122, 643348527, 908809, 391358, 307483, 1856807, 380392, 218205,
    224436, 1441386, 2512564055, 431057, 1293048, 1033264, 412895, 156203, 367742, 1245527, 644736404, 608505, 557451,
    1120986, 869805, 646564577, 65551, 1329516, 340095, 981385, 176598, 1852309, 1101373, 307486, 2517093030, 392012,
    392015, 392016, 28034, 1450156, 879970, 876478, 364032, 340097, 1411133, 159343, 335431, 2601677, 1348429, 209285,
    225584, 646564586, 225988, 1245523, 996558, 1274353, 1215031, 251051, 1245525, 1242744, 1378066, 420412, 376175,
    392842, 392421, 121872, 1927129, 58137, 1216062, 214905, 2016530, 884107, 1469162, 1088767, 52023, 52023, 1274356,
    1490296, 1445971, 213588, 2503508009, 478105, 204470, 392014, 136160, 470864, 627231, 1076937, 910412, 224999,
    28565, 5541, 2588941, 1490292, 1485588, 2078952, 1178786, 206665, 424696, 706427, 1494606, 1383816, 637000319,
    380174, 1383067, 28235, 1848, 2707016, 1181891, 1530170, 2171751, 2565931, 1776164, 5088, 415230, 1707952,
    646311961, 1215385, 105483, 2512047033, 667306, 1806172, 286152, 644383, 1765737, 78554, 110101, 1073253, 368624,
    33942, 191027, 301967, 1285191, 85995, 1189325, 1775544, 269259, 62320, 768533, 884155, 1050, 929101, 229731,
    642555142, 641522632, 2718217640, 759415, 64713, 2248, 295419, 1546155, 1110, 157784, 45157, 2771, 370324, 156202,
    403191, 1470181, 759412, 182773, 182455, 229919, 105229, 227865, 1147123, 655338, 520762, 196588, 1416969, 146826,
    35745, 177400, 385682, 263906, 2636415612, 123783, 1833852, 886292, 28234, 640753047, 1933902, 571898, 264951,
    1627895, 403181, 1236180, 78579, 392010, 1450, 1962263, 403181, 1042953, 1555112, 438856, 267746, 1658519, 747076,
    863370, 1490290, 2184083, 641228488, 332524, 388259, 1452450, 1830138, 1715347, 1586233, 489913, 355548, 2257,
    61858, 63128, 2654588040, 699433, 1017351, 591197, 2163, 1175445, 419665, 204050, 1715806, 1679091, 68911, 1452487,
    672459, 227138, 671094, 1478153, 1411117, 1411118, 547461, 2707049, 1341696, 1273429, 360241, 2588942, 321763,
    302484, 47959, 392011, 488549, 1073081, 1125967, 278990, 1642702, 68825, 72144, 390806, 767519, 2675099, 368454,
    1917811, 2054627, 2054628, 516051, 553469, 1758255, 128, 278991, 1553448, 488547, 996115, 1394850, 643692029,
    159849, 2017, 342666, 253107, 182779, 1457, 2505679008, 265429, 455373, 1679083, 1679076, 88724, 68910, 1004304,
    1186196, 1562698, 1475481, 637000073, 5149, 650716003, 929, 2587410, 405, 396317, 368623, 444162, 265477, 381852,
    1109010, 97393, 97393, 641522631, 1031594, 74386, 2041189, 33906, 368812, 1325932, 1325931, 1187080, 43776, 1193105,
    489913, 2675903501, 63129, 29283, 1334262, 1236046, 2208, 536044, 1586232, 348826, 1642646, 185696, 714067, 1972068,
    1348092, 2681813425, 60454, 1134406, 1592317, 2004704, 1526162, 1076695, 441119, 225406, 408753, 36850, 489911,
    85450, 44470, 768534, 869890, 503834, 1678448, 2583990, 515814, 503170, 2665495, 1502, 1485585, 521158, 1197874,
    413810, 413811, 558529, 267368, 1202768, 413812, 1052014, 1306154, 1485586, 1082276, 35719, 1475485, 1027249,
    393087, 411959, 1812810, 1798155, 1798157, 1798158, 114628, 2562278, 1312905, 700500, 2593339207, 42859, 51589,
    1367881, 1198301, 1547897, 890420, 42419, 36738, 60847, 555874, 2175149, 869891, 1267564, 1048396, 376415, 35743,
    119434, 63740, 72612, 1463631, 1487865, 1395169, 1655277, 1851175, 135740, 1082479, 1850348, 1308, 1571184, 1485204,
    2075419, 165813, 42879, 1765684, 2246, 307643, 237684, 36842, 47854, 2052941, 1380675, 1670458, 392413, 627618,
    1333998, 915180, 516123, 39664, 293256, 644790, 1232219, 73781, 1034015, 160404, 1680761, 546108, 1384639, 175632,
    341036, 425254, 1675686, 732237, 333146, 5763, 349931, 253108, 1630166, 1897628, 200616, 35825, 204285, 5041, 29282,
    293091, 268735, 256826, 642589, 1179467, 393657, 545502, 54121, 869886, 869889, 2707018, 110505, 1397668, 47055,
    76489, 116085, 410072, 33043, 2162, 864069, 29313, 364901, 1661150, 129338, 568899, 930124, 571932, 1133106,
    1417629, 1424294, 1211322, 376490, 218206, 1721088, 1721087, 57732, 28442, 367189, 1046043, 957042, 2033802,
    1897629, 2029849, 944671, 1615586, 710191, 483896, 83983, 2205, 1682650, 1930624, 1263547, 1460447, 1940533, 1087,
    1086, 407019, 424188, 1482737, 1709001, 42879, 2188, 81417, 1534102, 481366, 2201, 1732, 489912, 425468, 36812,
    425942, 69540, 299308, 230356, 933068, 54120, 2315, 60695, 135083, 332175, 1562603, 56636, 2627853727, 643692045,
    2684622511, 2684622610, 1200300, 2636415484, 2636415580, 2689387, 2627853791, 646311959, 643692044, 643692046,
    643692047, 650377982, 643692048, 643348543, 2508501045, 2540341047, 1111107, 649633105, 648028059, 641522656, 49339,
    640427139, 643692049, 375896, 650716010, 2627854017, 49340, 49341, 646564578, 646311953, 1402, 223182, 648028039,
    2571042483, 644736322, 1131473, 2761201727, 2506520015, 1053, 39669, 129311, 2504756058, 1563981, 420335, 477278,
    2506210035, 2506210034, 2505679071, 2507262044, 642555132, 35835, 646311930, 373981, 642555138, 195750, 646862346,
    272, 116039, 42735]
_formatted_theromphilic_taxonomy_ids = ', '.join([f'"{taxonomy_id}"' for taxonomy_id in thermophilic_taxonomy_ids])
# Use this search operator to limit searches to thermophilic organism IDs
termophilic_json_terminal_operator = \
    '{"type": "terminal", "service": "text", "parameters": ' \
    '{"attribute": "rcsb_entity_source_organism.taxonomy_lineage.id","operator": "in","negation": false, ' \
    '"value": [%s]}}' % _formatted_theromphilic_taxonomy_ids


def entity_thermophilicity(entry: str = None, entity_integer: int | str = None, entity_id: str = None) -> float | None:
    """Query the PDB API for an EntityID and return the associated chains and reference dictionary

    Args:
        entry: The 4 character PDB EntryID of interest
        entity_integer: The entity integer from the EntryID of interest
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
    Returns:
        Value ranging from 0-1 where 1 is completely thermophilic according to taxonomic classification
    """
    entity_request = query_entity_id(entry=entry, entity_integer=entity_integer, entity_id=entity_id)
    if not entity_request:
        return None

    return thermophilicity_from_entity_json(entity_request.json())


# def thermophilicity_from_entity_json(entity_json: dict[str, Any]) -> bool:
#     lineage_keywords = [line.get('name', '').lower()
#                         for organism in entity_json.get('rcsb_entity_source_organism', {})
#                         for line in organism.get('taxonomy_lineage', [])]
#     return True if 'thermo' in lineage_keywords else False


def thermophilicity_from_entity_json(entity_json: dict[str, Any]) -> float:
    """Return the extent to which the entity json entry in question is thermophilic

    Args:
        entity_json: The return json from PDB API query
    Returns:
        Value ranging from 0-1 where 1 is completely thermophilic
    """
    thermophilic_source = []
    for organism in entity_json.get('rcsb_entity_source_organism', {}):
        taxonomy_id = int(organism.get('ncbi_taxonomy_id', '-1'))
        if taxonomy_id in thermophilic_taxonomy_ids:
            thermophilic_source.append(1)
        else:
            thermophilic_source.append(0)

    return sum(thermophilic_source) / len(thermophilic_source)


def parse_entities_json(entity_jsons: Iterable[dict[str, Any]]) -> dict[str, dict]:
    """

    Args:
        entity_jsons: An Iterable of json like objects containing EntityID information as retrieved from the PDB API
    Returns:
        The entity dictionary with format -
        {'EntityID':
            {'chains': ['A', 'B', ...],
             'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
             'reference_sequence': 'MSLEHHHHHH...',
             'thermophilicity': 1.0},
         ...}
    """
    def extract_dbref(entity_ids_json: dict[str, Any]) -> dict[str, dict]:
        """For a PDB API EntityID, parse the associated chains and database reference identifiers

        Args:
            entity_ids_json: The json type dictionary returned from requests.Response.json()
        Returns:
            Ex: {'db': DATABASE, 'accession': 'Q96DC8'} where DATABASE can be one of 'GenBank', 'Norine', 'UniProt'
        """
        database_keys = ['db', 'accession']
        try:
            uniprot_ids = entity_ids_json['uniprot_ids']
            # Todo choose the most accurate if more than 2...
            #  'rcsb_polymer_entity_align' indicates how the model from the PDB aligns to UniprotKB through SIFTS
            #  [{provenance_source: "SIFTS",
            #    reference_database_accession: "P12528",
            #    reference_database_name: "UniProt",
            #    aligned_regions: [{entity_beg_seq_id: 1,
            #                       length: 124,
            #                       ref_beg_seq_id: 2}]
            #   },
            #   {}, ...
            #  ]
            if len(uniprot_ids) > 1:
                logger.warning(f'For Entity {entity_ids_json["rcsb_id"]}, found multiple UniProt Entries: '
                               f'{", ".join(uniprot_ids)}')  # . Using the first')
            db_d = dict(zip(database_keys, (UKB, tuple(uniprot_ids))))  # uniprot_ids[0])))
        except KeyError:  # No 'uniprot_ids'
            # GenBank = GB, which is mostly RNA or DNA structures or antibody complexes
            # Norine = NOR, which is small peptide structures, sometimes bound to proteins...
            try:
                identifiers = [dict(db=ident['database_name'], accession=(ident['database_accession'],))
                               for ident in entity_ids_json.get('reference_sequence_identifiers', [])]
            except KeyError:  # There are really no identifiers of use
                return {}
            if identifiers:
                if len(identifiers) == 1:  # Only one solution
                    # db_d = dict(zip(database_keys, identifiers[0]))
                    db_d = identifiers[0]
                else:  # find the most ideal accession_database UniProt > GenBank > Norine > ???
                    whatever_else = 0
                    priority_l = [[] for _ in range(len(identifiers))]
                    for idx, (database, accession) in enumerate(identifiers):
                        if database == UKB:
                            priority_l[0].append(idx)
                            # identifiers[idx][0] = uniprot  # rename the database_name
                        # elif database == NOR:
                        #     priority_l[2].append(idx)
                        elif database == GB:
                            priority_l[1].append(idx)  # Two elements are required from above len check, never IndexError
                            # identifiers[idx][0] = 'GB'  # rename the database_name
                        elif not whatever_else:  # only sets the first time an unknown identifier is seen
                            whatever_else = idx
                    # Loop through the list of prioritized identifiers
                    for identifier_idx in priority_l:
                        if identifier_idx:  # we have found a priority database, choose the corresponding identifier idx
                            # make the db_d with the db name as first arg and all the identifiers as the second arg
                            db_d = dict(zip(database_keys,
                                            (identifiers[identifier_idx[0]]['db'], [identifiers[idx]['accession']
                                                                                    for idx in identifier_idx])))
                            break
                    else:  # if no solution from priority but something else, choose the other
                        # db_d = dict(zip(database_keys, identifiers[whatever_else]))
                        db_d = identifiers[whatever_else]
            else:
                db_d = {}

        return db_d

    # entity_chain_d, ref_d = {}, {}
    entity_info = {}
    # I can use 'polymer_entity_count_protein' to further identify the entities in a protein, which gives me the chains
    # for entity_idx in range(1, int(entry_json['rcsb_entry_info']['polymer_entity_count_protein']) + 1):
    for entity_idx, entity_json in enumerate(entity_jsons, 1):
        # entity_ref_d = _get_entity_info(entry=entry, entity_integer=entity_idx)
        # entity_id = entity_json['rcsb_polymer_entity_container_identifiers']['rcsb_id']
        # entity_ref_d = parse_entity_json(entity_json)
        # ref_d.update(entity_ref_d)
        # entity_chain_d[entity_idx] = list(entity_ref_d.keys())  # these are the chains
        if entity_json is None:
            continue
        entity_json_ids = entity_json.get('rcsb_polymer_entity_container_identifiers')
        if entity_json_ids:
            entity_info[entity_json_ids['rcsb_id'].lower()] = dict(
                chains=entity_json_ids['asym_ids'],
                dbref=extract_dbref(entity_json_ids),
                reference_sequence=entity_json['entity_poly']['pdbx_seq_one_letter_code_can'],
                thermophilicity=thermophilicity_from_entity_json(entity_json),
            )
        # dbref = {chain: {'db': db, 'accession': db_accession_id}}
    # OR dbref = {entity: {'db': db, 'accession': db_accession_id}}
    # cryst = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

    # return {'entity': entity_chain_d, 'dbref': ref_d, 'reference_sequence': ref_seq}
    return entity_info


def _get_entity_info(entry: str = None, entity_integer: int | str = None, entity_id: str = None) -> \
        dict[str, dict[str, str]] | dict:
    """Query the PDB API for an EntityID and return the associated chains and reference dictionary

    Args:
        entry: The 4 character PDB EntryID of interest
        entity_integer: The entity integer from the EntryID of interest
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
    Returns:
        The entity dictionary with format -
        {'EntityID':
            {'chains': ['A', 'B', ...],
             'dbref': {'accession': ('Q96DC8',), 'db': 'UniProt'},
             'reference_sequence': 'MSLEHHHHHH...',
             'thermophilicity': 1.0},
         ...}
    """
    entity_request = query_entity_id(entry=entry, entity_integer=entity_integer, entity_id=entity_id)
    if not entity_request:
        return {}

    return parse_entities_json([entity_request.json()])


def query_entity_id(entry: str = None, entity_integer: str | int = None, entity_id: str = None) -> \
        requests.Response | None:
    """Retrieve PDB EntityID information from the PDB API. More info at http://data.rcsb.org/#data-api
     
    For all method types the following keys are available:
    {'rcsb_polymer_entity_annotation', 'entity_poly', 'rcsb_polymer_entity', 'entity_src_gen',
     'rcsb_polymer_entity_feature_summary', 'rcsb_polymer_entity_align', 'rcsb_id', 'rcsb_cluster_membership',
     'rcsb_polymer_entity_container_identifiers', 'rcsb_entity_host_organism', 'rcsb_latest_revision',
     'rcsb_entity_source_organism'}
    NMR only - {'rcsb_polymer_entity_feature'}
    EM only - set()
    X-ray_only_keys - {'rcsb_cluster_flexibility'}
    
    Args:
        entry: The 4 character PDB EntryID of interest
        entity_integer: The integer of the entity_id
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
    Returns:
        The entity information according to the PDB
    """
    if entity_id:
        entry, entity_integer, *_ = entity_id.split('_')  # Assume that this was passed correctly

    if entry and entity_integer:
        return connection_exception_handler(
            f'http://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity_integer}')


# Todo not completely useful in this module
def get_entity_id(entry: str = None, entity_integer: int | str = None, entity_id: str = None, chain: str = None) -> \
        tuple[str, str] | tuple[None]:
    """Retrieve a UniProtID from the PDB API by passing various PDB identifiers or combinations thereof

    Args:
        entry: The 4 character PDB EntryID of interest
        entity_integer: The entity integer from the EntryID of interest
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        chain: The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of interest
    Returns:
        The Entity_ID
    """
    if entry is not None:
        if len(entry) != 4:
            logger.warning(f'EntryID "{entry}" is not of the required format and will not be found with the PDB API')
        elif entity_integer is not None:
            return entry, entity_integer
            # entity_id = f'{entry}_{entity_integer}'
        else:
            info = _get_entry_info(entry)
            chain_entity = {chain: entity_idx for entity_idx, chains in info.get('entity', {}).items() for chain in chains}
            if chain is not None:
                try:
                    return entry, chain_entity[chain]
                    # entity_id = f'{entry}_{chain_entity[chain]}'
                except KeyError:
                    raise KeyError(f'No chain "{chain}" found in PDB ID {entry}. '
                                   f'Possible chains {", ".join(chain_entity)}')
            else:
                entity_integer = next(iter(chain_entity.values()))
                logger.warning('Using the argument "entry" without either "entity_integer" or "chain" is not '
                               f'recommended. Choosing the first EntityID "{entry}_{entity_integer}"')
                return entry, entity_integer
                # entity_id = f'{entry}_{entity_integer}'

    elif entity_id is not None:
        entry, entity_integer, *extra = entity_id.split('_')
        if not extra and len(entry) == 4:
            return entry, entity_integer

        logger.debug(f"EntityID '{entity_id}' isn't the required format and will not be found with the PDB API")

    return None,


# Todo refactor to internal ResourceDB retrieval from existing entity_json
def get_entity_uniprot_id(**kwargs) -> str | None:
    """Retrieve a UniProtID from the PDB API by passing various PDB identifiers or combinations thereof

    Keyword Args:
        entry=None (str): The 4 character PDB EntryID of interest
        entity_integer=None (str): The entity integer from the EntryID of interest
        entity_id=None (str): The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        chain=None (str): The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of
            interest
    Returns:
        The UniProt ID
    """
    entity_request = query_entity_id(*get_entity_id(**kwargs))
    if entity_request:
        # return the first uniprot entry
        return entity_request.json().get('rcsb_polymer_entity_container_identifiers')['uniprot_id'][0]


# Todo refactor to internal ResourceDB retrieval from existing entity_json
def get_entity_reference_sequence(**kwargs) -> str | None:
    """Query the PDB API for the reference amino acid sequence for a specified entity ID (PDB EntryID_Entity_ID)

    Keyword Args:
        entry=None (str): The 4 character PDB EntryID of interest
        entity_integer=None (str): The entity integer from the EntryID of interest
        entity_id=None (str): The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        chain=None (str): The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of
            interest
    Returns:
        One letter amino acid sequence
    """
    entity_request = query_entity_id(*get_entity_id(**kwargs))
    if entity_request:
        return entity_request.json().get('entity_poly')['pdbx_seq_one_letter_code_can']  # returns non-cannonical as 'X'
        # return entity_json.get('entity_poly')['pdbx_seq_one_letter_code']  # returns non-cannonical amino acids


def get_rcsb_metadata_schema(file=os.path.join(current_dir, 'rcsb_schema.pkl'), search_only=True, force_update=False):
    """Parse the rcsb metadata schema for useful information from the format
         {"properties" : {"assignment_version" : {"type" : "string", "examples" : [ "V4_0_2" ],
                                             "description" : "Identifies the version of the feature assignment.",
                                             "rcsb_description" : [
                                              {"text" : "Identifies the version of the feature assignment.",
                                               "context" : "dictionary"},
                                              {"text" : "Feature Version", "context" : "brief"} ]
                                            },
                          ...
                          "symmetry_type" : {"type" : "string",     <-- provide data type
               provide options     -->       "enum" : [ "2D CRYSTAL", "3D CRYSTAL", "HELICAL", "POINT" ],
               provide description -->       "description" : "The type of symmetry applied to the reconstruction",
               provide operators   -->       "rcsb_search_context" : [ "exact-match" ],
                                             "rcsb_full_text_priority" : 10,
                                             "rcsb_description" : [
                                                {"text" : "The type of symmetry applied to the reconstruction",
                                                 "context" : "dictionary"},
                                                {"text" : "Symmetry Type (Em 3d Reconstruction)", "context" : "brief"} ]
                                            },
                          ... },
          "title" : "Core Metadata", "additionalProperties" : false, "$comment" : "Schema version: 1.14.0"
          "required" : ["rcsb_id", "rcsb_entry_container_identifiers", "rcsb_entry_info",
                        "rcsb_pubmed_container_identifiers", "rcsb_polymer_entity_container_identifiers",
                        "rcsb_assembly_container_identifiers", "rcsb_uniprot_container_identifiers" ],
          "$schema" : "http://json-schema.org/draft-07/schema#",
          "description" : "Collective JSON schema that includes definitions for all indexed cores with RCSB metadata extensions.",
         }
    Returns:
        (dict): {attribute: {'dtype': 'string', 'description': 'XYZ', 'operators': {'equals'}, 'choices': []}, ...}
    """
    schema_pairs = {'dtype': 'type', 'description': 'description', 'operators': 'rcsb_search_context',
                    'choices': 'enum'}
    operator_d = {'full-text': 'contains_words, contains_phrase, exists', 'exact-match': 'in, exact_match, exists',
                  'default-match': 'equals, greater, less, greater_or_equal, less_or_equal, range, range_closed, '
                                   'exists', 'suggest': None}
    # Types of rcsb_search_context: (can be multiple)
    # full-text - contains_words, contains_phrase, exists
    # exact-match - in, exact-match, exists
    # default-match - equals, greater, less, greater_or_equal, less_or_equal, range, range_closed, exists
    # suggests - provides an example to the user in the GUI
    data_types = ['string', 'integer', 'number']

    def recurse_metadata(metadata_d, stack=tuple()):  # this puts the yield inside a local iter so we don't return
        for attribute in metadata_d:
            if metadata_d[attribute]['type'] == 'array':  # 'items' must be a keyword in dictionary
                # stack += (attribute, 'a')
                if metadata_d[attribute]['items']['type'] in data_types:  # array is the final attribute of the branch
                    yield stack + (attribute, 'a')
                elif metadata_d[attribute]['items']['type'] == 'object':  # type must be object, therefore contain 'properties' key and then have more attributes as leafs
                    yield from recurse_metadata(metadata_d[attribute]['items']['properties'], stack=stack + ((attribute, 'a', 'o',)))
                else:
                    logger.debug('Array with type %s found in %s' % (metadata_d[attribute], stack))
            elif metadata_d[attribute]['type'] == 'object':  # This should never be reachable?
                # print('%s object found %s' % (attribute, stack))
                if 'properties' in metadata_d[attribute]:  # check may be unnecessary
                    yield from recurse_metadata(metadata_d[attribute]['properties'], stack=stack + (attribute, 'o',))
                else:
                    logger.debug('Object with no properties found %s in %s' % (metadata_d[attribute], stack))
                    # yield stack + ('o', attribute,)
            elif metadata_d[attribute]['type'] in data_types:
                yield stack + (attribute,)  # + ('o', attribute,) add 'o' as the parent had properties from the object type
            else:
                logger.debug('other type = %s' % metadata_d[attribute]['type'])

    if not os.path.exists(file) or force_update:  # Todo and date.datetime - date.current is not greater than a month...
        logger.info('Gathering the most current PDB metadata. This may take a couple minutes...')
        metadata_json = requests.get(attribute_metadata_schema_json).json()
        metadata_properties_d = metadata_json['properties']
        gen_schema = recurse_metadata(metadata_properties_d)
        schema_header_tuples = [yield_schema for yield_schema in gen_schema]

        schema_dictionary_strings_d = {'a': "['items']", 'o': "['properties']"}  # 'a': "['items']['properties']"
        schema_d = {}
        for i, attribute_tuple in enumerate(schema_header_tuples):
            attribute_full = '.'.join(attribute for attribute in attribute_tuple
                                      if attribute not in schema_dictionary_strings_d)
            if i < 5:
                logger.debug(attribute_full)
            schema_d[attribute_full] = {}
            d_search_string = ''.join(f"['{attribute}']" if attribute not in schema_dictionary_strings_d
                                      else schema_dictionary_strings_d[attribute] for attribute in attribute_tuple)
            evaluation_d = eval(f'{metadata_properties_d}{d_search_string}')
            for key, value in schema_pairs.items():
                if value in evaluation_d:
                    schema_d[attribute_full][key] = evaluation_d[value]
                else:
                    schema_d[attribute_full][key] = None

            if 'format' in evaluation_d:
                schema_d[attribute_full]['dtype'] = 'date'

            if schema_d[attribute_full]['description']:  # convert the description to a simplified descriptor
                schema_d[attribute_full]['description'] = schema_d[attribute_full]['description'].split('\n')[0]

            if schema_d[attribute_full]['operators']:  # convert the rcsb_search_context to valid operator(s)
                schema_d[attribute_full]['operators'] = set(', '.join(
                    operator_d[search_context] for search_context in schema_d[attribute_full]['operators']
                    if operator_d[search_context]).split(', '))
            else:
                if search_only:  # remove entries that don't have a corresponding operator as these aren't searchable
                    schema_d.pop(attribute_full)

        pickled_schema_file = utils.pickle_object(schema_d, file, out_path='')
    else:
        return utils.unpickle(file)

    return schema_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for entries\n')
    parser.add_argument('-f', '--file_list', type=os.path.abspath,
                        help=f'{putils.ex_path("pdblist.file")}. Can be newline or comma separated')
    parser.add_argument('-d', '--download', type=bool, default=False,
                        help='Whether files should be downloaded. Default=False')
    parser.add_argument('-p', '--input_pdb_directory', type=os.path.abspath,
                        help='Where should reference PDB files be found? Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input_pisa_directory', type=os.path.abspath,
                        help='Where should reference PISA files be found? Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output_directory', type=os.path.abspath,
                        help='Where should interface files be saved?')
    parser.add_argument('-q', '--query_web', action='store_true',
                        help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    retrieve_pdb_entries_by_advanced_query()
