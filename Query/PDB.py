from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from json import dumps, load
from typing import Dict, Optional, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import requests

from SymDesignUtils import start_log, io_save, unpickle, pickle_object, ex_path
from Query.utils import input_string, confirmation_string, bool_d, validate_input, invalid_string, header_string, \
    format_string, connection_exception_handler

# Globals
logger = start_log(name=__name__)
# General Formatting
user_input_format = '\n%s\n%s' % (format_string % ('Option', 'Description'), '%s')
additional_input_string = '\nWould you like to add another%s? [y/n]%s' % ('%s', input_string)
instance_d = {'string': str, 'integer': int, 'number': float, 'date': str}

# Websites
pdb_query_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
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
return_types = {'entry': 'PDB ID\'s',
                'polymer_entity': 'PDB ID\'s appened with entityID - \'[pdb_id]_[entity_id] for macromolecules',
                'non_polymer_entity': 'PDB ID\'s appened with entityID - \'[pdb_id]_[entity_id] for '
                                      'non-polymers',
                'polymer_instance': 'PDB ID\'s appened with asymID\'s (chains) - \'[pdb_id]_[asym_id]',
                'assembly': 'PDB ID\'s appened with biological assemblyID\'s - \'[pdb_id]-[assembly_id]'}
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


def retrieve_entity_id_by_sequence(sequence):
    """From a given sequence, retrieve the top matching Entity ID from the PDB API

    Returns:
        (str): '1ABC_1'
    """
    return find_matching_entities_by_sequence(sequence=sequence, all_matching=False)[0]


def find_matching_entities_by_sequence(sequence=None, return_id='polymer_entity', **kwargs):
    """Search the PDB for matching IDs given a sequence and a return_type. Pass all_matching=False to retrieve the top
    10 IDs, otherwise return all IDs

    Keyword Args:
        return_id='polymer_entity' (str): The type of value to return where the acceptable values are in return_types
    Returns:
        (list[str]): The entities matching the sequence
    """
    if return_id not in return_types:
        raise KeyError('The specified return type "%s" is not supported. Viable types include %s'
                       % (return_id, ', '.join(return_types)))
    sequence_query = generate_terminal_group(service='sequence', sequence=sequence)
    sequence_query_results = query_pdb(generate_query(sequence_query, return_id=return_id, **kwargs))
    if sequence_query_results:
        return parse_pdb_response_for_ids(sequence_query_results)
    else:
        logger.warning(f'Sequence not found in PDB API!:\n{sequence}')
        return [None]


def parse_pdb_response_for_ids(response) -> list[str]:
    return [result['identifier'] for result in response['result_set']]


def parse_pdb_response_for_score(response) -> list[float] | list:
    if response:
        return list(map(float, (result['score'] for result in response['result_set'])))
    else:
        return []


def query_pdb(_query) -> dict[str, Any]:
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
            query_response = requests.get(pdb_query_url, params={'json': dumps(_query)})
            if query_response.status_code == 200:
                return query_response.json()
            elif query_response.status_code == 204:
                logger.warning('No response was returned. Your query likely found no matches!')
                break
            elif query_response.status_code == 429:
                logger.debug('Too many requests, pausing momentarily')
                time.sleep(2)
            else:
                logger.debug('Your query returned an unrecognized status code (%d)' % query_response.status_code)
                time.sleep(1)
                iteration += 1
        except requests.exceptions.ConnectionError:
            logger.debug('Requests ran into a connection error')
            time.sleep(1)
            iteration += 1

        if iteration > 5:
            logger.error('The maximum number of resource fetch attempts was made with no resolution. '
                         'Offending request %s' % getattr(query_response, 'url', pdb_query_url))  # todo format url
            break
            # raise DesignError('The maximum number of resource fetch attempts was made with no resolution. '
            #                   'Offending request %s' % getattr(query_response, 'url', pdb_query_url))
    return


def generate_parameters(attribute=None, operator=None, negation=None, value=None, sequence=None, **kwargs):  # Todo set up by kwargs
    if sequence:  # scaled identity_cutoff to 50% due to scoring function and E-value usage
        return {'evalue_cutoff': 0.0001, 'identity_cutoff': 0.5, 'target': 'pdb_protein_sequence', 'value': sequence}
    else:
        return {'attribute': attribute, 'operator': operator, 'negation': negation, 'value': value}


def pdb_id_matching_uniprot_id(uniprot_id, return_id='polymer_entity'):
    """Find all matching PDB entries from a specified UniProt ID and specific return ID

    Args:
        uniprot_id (str): The UniProt ID of interest
    Keyword Args:
        return_id='polymer_entity' (str): The type of value to return where the acceptable values are in return_types
    Returns:
        (list[str]): The list of matching IDs
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


def generate_query(search, return_id='entry', all_matching=True):
    """Format a PDB query with the specific return type and parameters affecting search results

    Args:
        search (dict): Contains the key, value pairs in accordance with groups and terminal groups
    Keyword Args:
        return_id='entry' (str): The type of value to return where the acceptable values are in return_types
    Returns:
        (dict): The formatted query to be sent via HTTP GET
    """
    if return_id not in return_types:
        raise KeyError('The specified return type "%s" is not supported. Viable types include %s'
                       % (return_id, ', '.join(return_types)))

    query_d = {'query': search, 'return_type': return_id}
    if all_matching:
        query_d.update({'request_options': {'return_all_hits': True}})

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
                                      '\n'.join('\tGroup Group #%d%s' % (i, format_string % group)
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
          f'This prompt will walk you through generating an advanced search query and retrieving the matching'
          ' set of entry ID\'s from the PDB. This automatically parses the ID\'s of interest for downstream use, which'
          ' can save you some headache. If you want to take advantage of the PDB webpage GUI to perform the advanced '
          f'search, visit:\n{pdb_advanced_search_url}\tThen enter "json" in the prompt below and follow those '
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
                             f'Choose {return_type} '
        print('DETAILS: To save time formatting and immediately move to your design pipeline, build your Query with the'
              ' PDB webpage GUI, then save the resulting JSON text to a file. To do this, first build your full query '
              f'on the advanced search page,{return_type_prompt}then click the Search button (magnifying glass icon). '
              f'After the page loads, a new section of the search page should appear above the Advanced Search Query '
              f'Builder dialog. There, click the JSON|->| button to open a new page with an automatically built JSON '
              f'representation of your query. Save the entirety of this JSON formatted query to a file to return your '
              f'chosen ID\'s\n')
        # ('Paste your JSON object below. IMPORTANT select from the opening \'{\' to '
        #  '\'"return_type": "entry"\' and paste. Before hitting enter, add a closing \'}\'. This hack '
        #  'ensures ALL results are retrieved with no sorting or pagination applied\n\n%s' %
        #  input_string)
        prior_query = input('Please specify the path where the JSON query file is located%s' % input_string)
        while not os.path.exists(prior_query):
            prior_query = input(f'The specified path "{prior_query}" doesn\'t exist! Please try again{input_string}')

        with open(prior_query, 'r') as f:
            json_input = load(f)

        # remove any paginate instructions from the json_input
        json_input['request_options'].pop('paginate', None)
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
                                       'PDB for?%s%s' % (user_input_format % '\n'.join(format_string % item
                                                                                       for item in return_types.items())
                                                         , input_string)
            return_type = validate_input(return_identifier_string, return_types)

        terminal_group_queries = []
        # terminal_group_queries = {}
        increment = 1
        while True:
            # TODO only text search is available now
            # query_builder_service_string = '\nWhat type of search method would you like to use?%s%s' % \
            #                                (user_input_format % '\n'.join(format_string % item
            #                                                               for item in services.items()), input_string)
            query_builder_attribute_string = '\nWhat type of attribute would you like to use? Examples include:%s' \
                                             '\n\nFor a more thorough list indicate "s" for search.\n' \
                                             'Alternatively, you can browse %s\nEnsure that your spelling' \
                                             ' is exact if you want your query to succeed!%s' % \
                                             (user_input_format % '\n'.join(format_string % (value, key)
                                                                            for key, value in attributes.items()),
                                              attribute_url, input_string)
            query_builder_operator_string = '\nWhat operator would you like to use?\n' \
                                            'Possible operators include:\n\t%s\nIf you would like to negate the ' \
                                            'operator, on input type "not" after your selection. Ex: equals not%s' % \
                                            ('%s', input_string)
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
                        attribute = input('Found the following instances of "%s":\n%s\nWhich option are you interested '
                                          'in? Enter "s" to repeat search.%s' %
                                          (search_term.upper(), user_input_format %
                                           '\n'.join(format_string % key_description_pair for key_description_pair
                                                     in search_schema(search_term)), input_string))
                        if attribute != 's':
                            break
                    if attribute in schema:  # confirm the user wants to go forward with this
                        break
                    else:
                        print('ERROR: %s was not found in PDB schema!' % attribute)
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

                while True:  # retrieve the operator for the search
                    while True:  # check if the operator should be negated
                        operator = input(query_builder_operator_string % ', '.join(schema[attribute]['operators']))
                        if len(operator.split()) > 1:
                            negation = operator.split()[1]
                            operator = operator.split()[0]
                            if negation.lower() == 'not':  # can negate any search
                                negate = True
                                break
                            else:
                                print('%s %s is not a recognized negation!\n Try \'%s not\' instead or remove extra '
                                      'input' % (invalid_string, negation, operator))
                        else:
                            negate = False
                            break
                    if operator in schema[attribute]['operators']:
                        break
                    else:
                        print('%s %s is not a valid operator!' % invalid_string, operator)

                op_in = True
                while op_in:  # check if operator is 'in'
                    if operator == 'in':
                        print('\nThe \'in\' operator can take multiple values. If you want multiple values, specify '
                              'each as a separate input.')
                    else:
                        op_in = False

                    while True:  # retrieve the value for the search
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
                                                print('%s %s is not a valid choice!' % (invalid_string, confirmation))
                                        if bool_d[confirmation.lower()] or confirmation.isspace():  # break the value routine on y or ''
                                            break

                                else:
                                    break
                            except ValueError:  # catch any conversion issue like float('A')
                                print('%s %s is not a valid %s value!' % (invalid_string,
                                                                          value, instance_d[schema[attribute]['dtype']]))

                    while op_in:
                        # TODO ensure that the in parameters are spit out as a list
                        additional = input(additional_input_string % ' value to your \'in\' operator')
                        if additional.lower() in bool_d:
                            if bool_d[additional.lower()] or additional.isspace():
                                break  # Stop the inner 'in' check loop
                            else:
                                op_in = False  # Stop the inner and outer 'in' while loops
                        else:
                            print('%s %s is not a valid choice!' % (invalid_string, confirmation))

                while True:
                    confirmation = input('\n%s\n%s' % (query_display_string %
                                                       (increment, service.upper(), attribute,
                                                        'NOT ' if negate else '', operator.upper(), value),
                                                       confirmation_string))
                    if confirmation.lower() in bool_d:
                        break
                    else:
                        print('%s %s is not a valid choice!' % (invalid_string, confirmation))
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
                    print('%s %s is not a valid choice!' % invalid_string, confirmation)
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
        io_save(retrieved_ids)

    if return_results:
        return retrieved_ids


def query_pdb_by(entry: str = None, assembly_id: str = None, assembly_integer: int | str = None, entity_id: str = None,
                 entity_integer: int | str = None, chain: str = None, **kwargs) -> dict | None:
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
    if entry:
        if len(entry) == 4:
            if entity_integer:
                logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
                return _get_entity_info(entry=entry, entity_integer=entity_integer)
            elif assembly_integer:
                logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
                return _get_assembly_info(entry=entry, assembly_integer=assembly_integer)
            else:
                logger.debug(f'Querying PDB API with {entry}')
                info = _get_entry_info(entry)
                if chain:
                    chain_entity = \
                        {chain: entity_idx for entity_idx, chains in info.get('entity').items() for chain in chains}
                    try:
                        logger.debug(f'Querying PDB API with {entry}_{chain_entity[chain]}')
                        return _get_entity_info(entry=entry, entity_integer=chain_entity[chain])
                    except KeyError:
                        raise KeyError(f'No chain "{chain}" found in PDB ID {entry}. '
                                       f'Possible chains {", ".join(chain_entity)}')
                else:
                    return info
        else:
            logger.warning(f'EntryID "{entry}" is not of the required format and will not be found with the PDB API')
    elif assembly_id:
        entry, assembly_integer, *extra = assembly_id.split('_')
        if not extra and len(entry) == 4:
            logger.debug(f'Querying PDB API with {entry}-{assembly_integer}')
            return _get_assembly_info(entry=entry, assembly_integer=assembly_integer)

        logger.warning(f'AssemblyID "{entry}-{assembly_integer}" is not of the required format and will not be found '
                       f'with the PDB API')

    elif entity_id:
        entry, entity_integer, *extra = entity_id.split('_')
        if not extra and len(entry) == 4:
            logger.debug(f'Querying PDB API with {entry}_{entity_integer}')
            return _get_entity_info(entry=entry, entity_integer=entity_integer)

        logger.warning(f'EntityID "{entry}_{entity_integer}" is not of the required format and will not be found with '
                       f'the PDB API')


# def _query_*_id(*_id: str = None, entry: str = None, *_integer: str | int = None) -> dict[str, Any] | None:
# Ex. entry = '4atz'
# ex. assembly = 1
# url = https://data.rcsb.org/rest/v1/core/assembly/4atz/1
# ex. chain
# url = https://data.rcsb.org/rest/v1/core/polymer_entity_instance/4atz/A


def _query_assembly_id(assembly_id: str = None, entry: str = None, assembly_integer: str | int = None) -> \
        requests.Response | None:
    """Fetches the JSON object for the AssemblyID from the PDB API

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
        entry, assembly_integer, *_ = assembly_id.split('_')  # assume that this was passed correctly

    if entry and assembly_integer:
        return connection_exception_handler(f'http://data.rcsb.org/rest/v1/core/assembly/{entry}/{assembly_integer}')


def _get_assembly_info(assembly_id: str = None, entry: str = None, assembly_integer: int = 1) -> dict[int, list[str]] | None:
    """Retrieve information on the assembly for a particular entry from the PDB API

    Args:
        assembly_id: The AssemblyID to query with format (1ABC-1)
        entry: The pdb code of interest
        assembly_integer: The particular assembly number to query
    Returns:
        The mapped entity number to the chain ID's in the assembly. Ex: {1: ['A', 'A', 'A', ...]}
    """
    assembly_request = _query_assembly_id(assembly_id=assembly_id, entry=entry, assembly_integer=assembly_integer)
    if not assembly_request:
        return
    if not assembly_id:
        assembly_id = f'{entry}_{assembly_integer}'

    entity_clustered_chains = {}
    for entity_idx, symmetry in enumerate(assembly_request.json()['rcsb_struct_symmetry'], 1):
        # for symmetry in symmetries:  # [{}, ...]
        # symmetry contains:
        # {symbol: "O", type: 'Octahedral, stoichiometry: [], oligomeric_state: "Homo 24-mer", clusters: [],
        #  rotation_axes: [], kind: "Global Symmetry"}
        for cluster_idx, cluster in enumerate(symmetry['clusters'], 1):  # [{}, ...]
            # CLUSTER_IDX is not a mapping to entity index...
            # cluster contains:
            # {members: [], avg_rmsd: 5.219512137974998e-14}
            # for cluster in clusters:
            cluster_members = []
            for member in cluster['members']:  # [{}, ...]
                # member contains:
                # {asym_id: "A", pdbx_struct_oper_list_ids: []}
                cluster_members.append(member.get('asym_id'))
            entity_clustered_chains[cluster_idx] = cluster_members

    return entity_clustered_chains


def _query_entry_id(entry: str = None) -> requests.Response | None:
    """Fetches the JSON object for the AssemblyID from the PDB API

    Args:
        entry: The PDB code to search for
    Returns:
        The entry information according to the PDB
    """
    #     The following information is returned:
    #     All methods (SOLUTION NMR, ELECTRON MICROSCOPY, X-RAY DIFFRACTION) have the following keys:
    #     {'rcsb_primary_citation', 'pdbx_vrpt_summary', 'pdbx_audit_revision_history', 'audit_author',
    #      'pdbx_database_status', 'rcsb_id', 'pdbx_audit_revision_details', 'struct_keywords',
    #      'rcsb_entry_container_identifiers', 'entry', 'rcsb_entry_info', 'struct', 'citation', 'exptl',
    #      'rcsb_accession_info'}
    #     EM only keys:
    #     {'em3d_fitting', 'em3d_fitting_list', 'em_image_recording', 'em_specimen', 'em_software', 'em_entity_assembly',
    #      'em_vitrification', 'em_single_particle_entity', 'em3d_reconstruction', 'em_experiment', 'pdbx_audit_support',
    #      'em_imaging', 'em_ctf_correction'}
    #     Xray only keys:
    #     {'diffrn_radiation', 'cell', 'reflns', 'diffrn', 'software', 'refine_hist', 'diffrn_source', 'exptl_crystal',
    #      'symmetry', 'diffrn_detector', 'refine', 'reflns_shell', 'exptl_crystal_grow'}
    #     NMR only keys:
    #     {'pdbx_nmr_exptl', 'pdbx_audit_revision_item', 'pdbx_audit_revision_category', 'pdbx_nmr_spectrometer',
    #      'pdbx_nmr_refine', 'pdbx_nmr_representative', 'pdbx_nmr_software', 'pdbx_nmr_exptl_sample_conditions',
    #      'pdbx_nmr_ensemble'}

    # entry_json['rcsb_entry_info'] = \
    #     {'assembly_count': 1, 'branched_entity_count': 0, 'cis_peptide_count': 3, 'deposited_atom_count': 8492,
    #     'deposited_model_count': 1, 'deposited_modeled_polymer_monomer_count': 989,
    #     'deposited_nonpolymer_entity_instance_count': 0, 'deposited_polymer_entity_instance_count': 6,
    #     'deposited_polymer_monomer_count': 1065, 'deposited_solvent_atom_count': 735,
    #     'deposited_unmodeled_polymer_monomer_count': 76, 'diffrn_radiation_wavelength_maximum': 0.9797,
    #     'diffrn_radiation_wavelength_minimum': 0.9797, 'disulfide_bond_count': 0, 'entity_count': 3,
    #     'experimental_method': 'X-ray', 'experimental_method_count': 1, 'inter_mol_covalent_bond_count': 0,
    #     'inter_mol_metalic_bond_count': 0, 'molecular_weight': 115.09, 'na_polymer_entity_types': 'Other',
    #     'nonpolymer_entity_count': 0, 'polymer_composition': 'heteromeric protein', 'polymer_entity_count': 2,
    #     'polymer_entity_count_dna': 0, 'polymer_entity_count_rna': 0, 'polymer_entity_count_nucleic_acid': 0,
    #     'polymer_entity_count_nucleic_acid_hybrid': 0, 'polymer_entity_count_protein': 2,
    #     'polymer_entity_taxonomy_count': 2, 'polymer_molecular_weight_maximum': 21.89,
    #     'polymer_molecular_weight_minimum': 16.47, 'polymer_monomer_count_maximum': 201,
    #     'polymer_monomer_count_minimum': 154, 'resolution_combined': [1.95],
    #     'selected_polymer_entity_types': 'Protein (only)',
    #     'software_programs_combined': ['PHASER', 'REFMAC', 'XDS', 'XSCALE'], 'solvent_entity_count': 1,
    #     'diffrn_resolution_high': {'provenance_source': 'Depositor assigned', 'value': 1.95}}

    if entry:
        return connection_exception_handler(f'http://data.rcsb.org/rest/v1/core/entry/{entry}')


def _get_entry_info(entry: str = None, **kwargs) -> dict[str, Any] | None:
    """Retrieve PDB information from the RCSB API. More info at http://data.rcsb.org/#data-api

    Makes 1 + num_of_entities calls to the PDB API for information

    Args:
        entry: The PDB code to search for
    Returns:
        {'entity': {1: ['A', 'B'], ...}, 'res': resolution, 'dbref': {chain: {'accession': ID, 'db': UNP}, ...},
         'struct': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}}
    """
    entry_request = _query_entry_id(entry)
    if not entry_request:
        return

    entry_json = entry_request.json()
    # if 'method' in entry_json['exptl'][0]:
    if 'experimental_method' in entry_json['rcsb_entry_info']:
        # exptl_method = entry_json['exptl'][0]['method']
        exptl_method = entry_json['rcsb_entry_info']['experimental_method'].lower()
        if 'ray' in exptl_method and 'cell' in entry_json and 'symmetry' in entry_json:  # Todo make ray, diffraction
            ang_a, ang_b, ang_c = entry_json['cell']['angle_alpha'], entry_json['cell']['angle_beta'], entry_json['cell']['angle_gamma']
            a, b, c = entry_json['cell']['length_a'], entry_json['cell']['length_b'], entry_json['cell']['length_c']
            space_group = entry_json['symmetry']['space_group_name_hm']
            struct_d = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
            resolution = entry_json['rcsb_entry_info']['resolution_combined'][0]
        else:  # Todo NMR and EM
            struct_d = {}
            resolution = None
    else:
        logger.warning('Entry has no "experimental_method" keyword')
        return

    entity_chain_d, ref_d, db_d = {}, {}, {}
    # I can use 'polymer_entity_count_protein' to further identify the entities in a protein, which gives me the chains
    # for entity_idx in range(1, int(entry_json['rcsb_entry_info']['polymer_entity_count_protein']) + 1):
    for entity_idx in range(1, int(entry_json['rcsb_entry_info']['polymer_entity_count']) + 1):
        entity_ref_d = _get_entity_info(entry=entry, entity_integer=entity_idx)
        ref_d.update(entity_ref_d)
        entity_chain_d[entity_idx] = list(entity_ref_d.keys())  # these are the chains
        # dbref = {chain: {'db': db, 'accession': db_accession_id}}
    # OR dbref = {entity: {'db': db, 'accession': db_accession_id}}
    # cryst = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

    return {'entity': entity_chain_d, 'res': resolution, 'dbref': ref_d, 'struct': struct_d, 'method': exptl_method}


def _get_entity_info(entity_id: str = None, entry: str = None, entity_integer: int | str = None) -> \
        dict[str, dict[str, str]] | dict:
    """Query the PDB API for an EntityID and return the associated chains and reference dictionary

    Args:
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entry: The 4 character PDB EntryID of interest
        entity_integer: The entity integer from the EntryID of interest
    Returns:
        {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
    """
    entity_request = _query_entity_id(entry=entry, entity_integer=entity_integer, entity_id=entity_id)
    if not entity_request:
        return {}

    if not entity_id:
        entity_id = f'{entry}_{entity_integer}'

    entity_json = entity_request.json()
    chains = entity_json['rcsb_polymer_entity_container_identifiers']['asym_ids']  # = ['A', 'B', 'C']
    database_keys = ['db', 'accession']
    db_d = {}
    try:
        try:
            uniprot_id = entity_json['rcsb_polymer_entity_container_identifiers']['uniprot_ids']
            database = 'UNP'
            if len(uniprot_id) > 1:  # Todo choose the most accurate if more than 2...
                logger.warning('For Entity %s, found multiple UniProt Entries %s. Selecting the first'
                               % (entity_id, uniprot_id))
            db_d = dict(zip(database_keys, ('UNP', uniprot_id[0])))  # may be an issue where there is more than one
        except KeyError:  # if no uniprot_id
            # GenBank = GB, which is mostly RNA or DNA structures or antibody complexes
            # Norine = NOR, which is small peptide structures, sometimes bound to proteins...
            identifiers = [(ident['database_name'], ident['database_accession'])
                           for ident in entity_json[
                               'rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers']]
            if len(identifiers) > 1:  # we find the most ideal accession_database UniProt > GenBank > Norine > ???
                whatever_else = None
                priority_l = [[] for _ in range(len(identifiers))]
                for idx, (database, accession) in enumerate(identifiers):
                    if database == 'UniProt':
                        priority_l[0].append(idx)
                        identifiers[idx][0] = 'UNP'
                    elif database == 'GenBank':
                        priority_l[1].append(idx)  # two elements are required from above len check, never IndexError
                        identifiers[idx][0] = 'GB'
                    elif not whatever_else:
                        whatever_else = idx
                for identifier_idx in priority_l:
                    if identifier_idx:  # we have found a priority database, choose the corresponding identifier idx
                        # Todo choose most accurate if more than 2...
                        db_d = dict(zip(database_keys, identifiers[identifier_idx[0]]))
                        break
                # finally if no solution from priority but something else, choose the other
                if not db_d and whatever_else:
                    db_d = dict(zip(database_keys, identifiers[whatever_else]))
            else:
                db_d = dict(zip(database_keys, identifiers[0]))

        ref_d = {chain: db_d for chain in chains}
    except KeyError:  # there are no known identifiers found
        ref_d = {chain: db_d for chain in chains}
    return ref_d


def _query_entity_id(entry: str = None, entity_integer: str | int = None, entity_id: str = None) -> \
        requests.Response | None:
    """Fetches the JSON object for the entity_id from the PDB API
     
    For all method types the following keys are available:
    {'rcsb_polymer_entity_annotation', 'entity_poly', 'rcsb_polymer_entity', 'entity_src_gen',
     'rcsb_polymer_entity_feature_summary', 'rcsb_polymer_entity_align', 'rcsb_id', 'rcsb_cluster_membership',
     'rcsb_polymer_entity_container_identifiers', 'rcsb_entity_host_organism', 'rcsb_latest_revision',
     'rcsb_entity_source_organism'}
    NMR only - {'rcsb_polymer_entity_feature'}
    EM only - set()
    X-ray_only_keys - {'rcsb_cluster_flexibility'}
    
    Args:
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entry: The 4 character PDB EntryID of interest
        entity_integer: The integer of the entity_id
    Returns:
        The entity information according to the PDB
    """
    if entity_id:
        entry, entity_integer, *_ = entity_id.split('_')  # assume that this was passed correctly

    if entry and entity_integer:
        return connection_exception_handler(
            f'http://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity_integer}')


# Todo not completely useful in this module
def get_entity_id(entity_id: str = None, entry: str = None, entity_integer: int | str = None, chain: str = None) -> \
        tuple[str, str] | None:
    """Retrieve a UniProtID from the PDB API by passing various PDB identifiers or combinations thereof

    Args:
        entity_id: The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entry: The 4 character PDB EntryID of interest
        entity_integer: The entity integer from the EntryID of interest
        chain: The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of interest
    Returns:
        The Entity_ID
    """
    if entry:
        if len(entry) != 4:
            logger.warning(f'EntryID "{entry}" is not of the required format and will not be found with the PDB API')
        elif entity_integer:
            return entry, entity_integer
            # entity_id = f'{entry}_{entity_integer}'
        else:
            info = _get_entry_info(entry)
            chain_entity = {chain: entity_idx for entity_idx, chains in info.get('entity').items() for chain in chains}
            if chain:
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

    elif entity_id:
        entry, entity_integer, *extra = entity_id.split('_')
        if not extra and len(entry) == 4:
            return entry, entity_integer

        logger.warning(f'EntityID "{entry}_{entity_integer}" is not of the required format and will not be found with '
                       f'the PDB API')


# Todo refactor to internal ResourceDB retrieval from existing entity_json
def get_entity_uniprot_id(**kwargs) -> str | None:
    """Retrieve a UniProtID from the PDB API by passing various PDB identifiers or combinations thereof

    Keyword Args:
        entity_id=None (str): The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entry=None (str): The 4 character PDB EntryID of interest
        entity_integer=None (str): The entity integer from the EntryID of interest
        chain=None (str): The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of
            interest
    Returns:
        The UniProt ID
    """
    entity_request = _query_entity_id(*get_entity_id(**kwargs))
    if entity_request:
        # return the first uniprot entry
        return entity_request.json().get('rcsb_polymer_entity_container_identifiers')['uniprot_id'][0]


# Todo refactor to internal ResourceDB retrieval from existing entity_json
def get_entity_reference_sequence(**kwargs) -> str | None:
    """Query the PDB API for the reference amino acid sequence for a specified entity ID (PDB EntryID_Entity_ID)

    Keyword Args:
        entity_id=None (str): The PDB formatted EntityID. Has the format EntryID_Integer (1ABC_1)
        entry=None (str): The 4 character PDB EntryID of interest
        entity_integer=None (str): The entity integer from the EntryID of interest
        chain=None (str): The polymer "chain" identifier otherwise known as the "asym_id" from the PDB EntryID of
            interest
    Returns:
        One letter amino acid sequence
    """
    entity_request = _query_entity_id(*get_entity_id(**kwargs))
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

        schema_dictionary_strings_d = {'a': '[\'items\']', 'o': '[\'properties\']'}  # 'a': '[\'items\'][\'properties\']'
        schema_d = {}
        for i, attribute_tuple in enumerate(schema_header_tuples):
            attribute_full = '.'.join(attribute for attribute in attribute_tuple if attribute not in schema_dictionary_strings_d)
            if i < 5:
                logger.debug(attribute_full)
            schema_d[attribute_full] = {}
            d_search_string = ''.join('[\'%s\']' % attribute if attribute not in schema_dictionary_strings_d
                                      else schema_dictionary_strings_d[attribute] for attribute in attribute_tuple)
            evaluation_d = eval('%s%s' % (metadata_properties_d, d_search_string))
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

        pickled_schema_file = pickle_object(schema_d, file, out_path='')
    else:
        return unpickle(file)

    return schema_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for entries\n')
    parser.add_argument('-f', '--file_list', type=str,
                        help='%s. Can be newline or comma separated.' % ex_path('pdblist.file'))
    parser.add_argument('-d', '--download', type=bool, default=False,
                        help='Whether files should be downloaded. Default=False')
    parser.add_argument('-p', '--input_pdb_directory', type=str, help='Where should reference PDB files be found? '
                                                                      'Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input_pisa_directory', type=str, help='Where should reference PISA files be found? '
                                                                       'Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output_directory', type=str, help='Where should interface files be saved?')
    parser.add_argument('-q', '--query_web', action='store_true',
                        help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    retrieve_pdb_entries_by_advanced_query()
