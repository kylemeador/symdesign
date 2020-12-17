import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from copy import deepcopy
from json import dumps

import requests
from SymDesignUtils import start_log, io_save

# Globals
pdb_query_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
# v TODO use inspect webpage to pull the entire dictionary of possibilities
pdb_advanced_search_url = 'http://www.rcsb.org/search/advanced'
pdb_rest_url = 'http://data.rcsb.org/rest/v1/core/'  # uniprot/  # 1AB3/1'
attribute_url = 'https://search.rcsb.org/search-attributes.html'
attribute_metadata_schema_json = 'https://search.rcsb.org/rcsbsearch/v1/metadata/schema'

# Types of rcsb_search_context: (can be multiple)
# full-text - contains_words, contains_phrase, exists
# exact-match - in, exact-match, exists
# default-match - equals, greater, less, greater_or_equal, less_or_equal, range, range_closed, exists
# suggests - provides an example to the user in the GUI

# can negate any search
# in - operators can have multiple enum's


def get_rcsb_metadata_schema():
    schema_pairs = {'dtype': 'type', 'description': 'description', 'operators': 'rcsb_search_context',
                    'choices': 'enum'}
    operator_d = {'full-text': 'contains_words, contains_phrase, exists', 'exact-match': 'in, exact-match, exists',
                  'default-match': 'equals, greater, less, greater_or_equal, less_or_equal, range, range_closed, exists',
                  'suggests': ''}

    def recurse_metadata_iter(properties, stack=tuple()):  # this puts the yield inside a local iter so we don't return
        # print('Stack: %s' % (str(stack) or ''))
        for _property in properties:
            if 'items' in properties[_property] and 'properties' in properties[_property]['items']:
                yield from recurse_metadata_iter(properties[_property]['items']['properties'], stack=stack + (_property,))
        # print('Stack: %s' % (str(stack) or ''))
            else:
                # print('Yielding stack: %s' % str(stack + (_property,)))
                yield stack + (_property,)  # zip(property)  # , list(properties.keys()))

    def recurse_metadata(properties, stack=tuple()):  # property, properties):  # GOLD
        # print(properties)
        # for property in properties:
        #     if 'items' in properties[property]:
        #         if 'properties' in properties[property]['items']:
        print('Stack: %s' % (str(stack) or ''))
        if 'items' in properties:
            if 'properties' in properties['items']:
                for child_property in properties['items']['properties']:
                    print('Child: %s' % child_property)
                    yield from recurse_metadata(properties['items']['properties'][child_property], stack=stack + (child_property,))
                    # for recursed_property in recurse_metadata(child_property, properties['items']['properties'][child_property]):
                        # yield (property, recursed_property)
                    # yield child_property, recurse_metadata(properties['items']['properties'][child_property])  # Never tried
                    # yield from property, recurse_metadata(child_property, properties['items']['properties'])
                    # for recursed_property in recurse_metadata(child_property, properties['items']['properties']):
                    #     yield property, recursed_property
                    # yield zip(repeat(property), zip(*recurse_metadata(child_property, properties['items']['properties'])))  # GOLD
                    # yield property, recurse_metadata(child_property, properties['items']['properties'])
                    # return zip(property, zip(*recurse_metadata(child_property, properties['items']['properties'])))
            else:
                print('Yielding stack: %s' % str(stack))
                yield stack  # zip(property)  # , list(properties.keys()))
        else:
            print('Yielding stack: %s' % str(stack))
            yield stack  # zip(property)  # , list(properties.keys()))
        # return None  # zip(property)  # , list(properties.keys()))
        # return property  # zip(property)  # , list(properties.keys()))  # GOLD

    # returns don't work because I am iterating through the child_properties. I could build a list and return the list
    # but that seems messy when there are so many pythonic tools

    # if I remove passing the property/child property and forgo it's return (return none), I can handle the return in
    # one section...

    metadata_json = requests.get(attribute_metadata_schema_json).json()
    metadata_properties_d = metadata_json['properties']
    # gen_schema = [recurse_metadata(metadata_properties_d[schema_property], stack=(schema_property,))
    #               for schema_property in metadata_properties_d]  # VERY CLOSE
    # print(gen_schema, len(gen_schema), id(gen_schema[0]))  #
    # ready = input('Ready?')
    # schema_headers = [schema_tuple.__next__() for schema_tuple in gen_schema]

    # schema_properties = ((m_property, recurse_metadata(metadata_properties_d[m_property])) for m_property in metadata_properties_d)
    # schema_properties = (recurse_metadata(property, metadata_properties_d[property]) for property in metadata_properties_d)
    # print(schema_properties[:5])

    # print(metadata_properties_d['pdbx_struct_special_symmetry'])

    gen_schema = recurse_metadata_iter(metadata_properties_d)
    # ready = input('Ready?')
    schema_headers = [schema_tuple for schema_tuple in gen_schema]

    # for i, property_tuple in enumerate(schema_headers):
    #     print('property_tuple: %s' % str(property_tuple))
    #     for _property in property_tuple:
    #         print('property5: %s' % _property)
    #     if i == 5:
    #         break

    clean_schema_d = {}
    for i, property_tuple in enumerate(schema_headers):
        dict_string = '[\'items\'][\'properties\']'.join('[\'%s\']' % _property for _property in property_tuple)
        evaluation_d = eval('%s%s' % (metadata_properties_d, dict_string))
        attribute = '.'.join(_property for _property in property_tuple)
        clean_schema_d[attribute] = {}
        for key, value in schema_pairs.items():
            if value in evaluation_d:
                clean_schema_d[attribute][key] = evaluation_d[value]
            else:
                clean_schema_d[attribute][key] = None

        if clean_schema_d[attribute]['operators']:  # convert the rcsb_search_context to valid operator(s)
            clean_schema_d[attribute]['operators'] = set(', '.join(
                operator_d[search_context] for search_context in clean_schema_d[attribute]['operators']).split(', '))
        if i % 10 == 0:
            print(attribute, ':', clean_schema_d[attribute])

    return clean_schema_d

    # metadata_subheader = metadata_header['items']['properties']
    # for metadata_subhead in metadata_subheader:
    #     if 'items' in metadata_subhead:
    #         True
    #     else:
    #         metadata_subhead['items']['properties']

    # "properties" : {"assignment_version" : {"type" : "string", "examples" : [ "V4_0_2" ],
    #                                         "description" : "Identifies the version of the feature assignment.",
    #                                         "rcsb_description" : [
    #                                          {"text" : "Identifies the version of the feature assignment.",
    #                                           "context" : "dictionary"},
    #                                          {"text" : "Feature Version", "context" : "brief"} ]
    #                                        },
    # ...
    #                 "symmetry_type" : {"type" : "string",     <-- provide data type
    #      provide options     -->       "enum" : [ "2D CRYSTAL", "3D CRYSTAL", "HELICAL", "POINT" ],
    #      provide description -->       "description" : "The type of symmetry applied to the reconstruction",
    #      provide operators   -->       "rcsb_search_context" : [ "exact-match" ],
    #                                    "rcsb_full_text_priority" : 10,
    #                                    "rcsb_description" : [
    #                                       {"text" : "The type of symmetry applied to the reconstruction",
    #                                        "context" : "dictionary"},
    #                                       {"text" : "Symmetry Type (Em 3d Reconstruction)", "context" : "brief"} ]
    #                                   },



# uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'
attribute_prefix = 'rcsb_'

# def get_uniprot_protein_name(uniprot_id):
#     uniprot_response = requests.get(uniprot_url.format(uniprot_id)).text
#     return fromstring(uniprot_response).find('.//{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName').text

# Query formatting requirements
request_types = {'group': 'Results must satisfy a group of requirements', 'terminal': 'A specific requirement'}
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
attributes = {'symmetry': '%sstruct_symmetry.symbol', 'experimental_method': '%sexptl.method',
              'resolution': '%sentry_info.resolution_combined',
              'accession_id': '%spolymer_entity_container_identifiers.reference_sequence_identifiers.'
                              'database_accession',
              'accession_db': '%spolymer_entity_container_identifiers.reference_sequence_identifiers.'
                              'database_name',
              'organism': '%sentity_source_organism.taxonomy_lineage.name'}

search_term = 'ELECTRON MICROSCOPY'  # For example. of 'C2'
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


def parse_pdb_response_for_ids(response):
    return [result['identifier'] for result in response['result_set']]


def parse_pdb_response_for_score(response):
    return [result['score'] for result in response['result_set']]


def query_pdb(query):
    return requests.get(pdb_query_url, params={'json': dumps(query)}).json()


def retrieve_pdb_entries_by_advanced_query(save=True, return_results=True):
    format_string = '\t%s\t\t%s'
    input_string = '\nInput:'
    invalid_string = 'Invalid choice, please try again'
    confirmation_string = 'If this is correct, indicate \'y\', if not \'n\', and you can re-input.%s' % input_string
    bool_d = {'y': True, 'n': False}
    user_input_format = '\n%s\n%s' % (format_string % ('Option', 'Description'), '%s')
    additional_input_string = '\nWould you like to add another?%s' % input_string
    schema = get_rcsb_metadata_schema()

    def search_schema(term):
        return {key: schema[key]['description'] for key in schema if term.lower() in schema[key]['description'].lower()}

    def generate_parameters(attribute, operator, value):
        return {"operator": operator, "value": value, "attribute": attribute}

    def generate_terminal_group(service, *parameter_args):
        return {'type': 'terminal', 'service': service, 'parameters': generate_parameters(*parameter_args)}

    def generate_group(operation, child_groups):
        return {'type': 'group', 'operation': operation, 'nodes': [child_groups]}

    def generate_query(search, return_type, return_all=True):
        query_d = {'query': search, 'return_type': return_type}
        if return_all:  # Todo test this as far as true being capitalized or not (example wasn't)
            query_d.update({'request_options': {'return_all_hits': True}})

        return query_d

    def make_groups(*args, recursive_depth=0):  # (terminal_queries, grouping,
        terminal_queries = args[0]
        work_on_group = args[recursive_depth]
        all_grouping_indices = {i for i in range(1, len(work_on_group) + 1)}

        group_introduction = '\nGROUPING INSTRUCTIONS:\n' \
                             'Because you have %d search queries, you need to combine these to a total search strategy'\
                             '. This is accomplished by grouping your search queries together using the operations %s.'\
                             ' You must eventually group all queries into a single logical operation. ' \
                             '\nIf you have multiple groups, you will need to group those groups, so on and so forth.' \
                             '\nIndicate your group selections with a space separated list! You will choose the group '\
                             'operation to combine this list afterwards.\nFollow prior prompts if you need a reminder '\
                             'of how group # relate to query #' % (len(terminal_queries), group_operators)
        group_grouping_intro = '\nGroups remain, you must group groups as before.'
        group_inquiry_string = '\nWhich of these (identified by #) would you like to combine into a group?%s' % \
                               input_string
        group_specification_string = 'You specified \'%s\' as a single group.'
        group_logic_string = '\nWhat group operator %s would you like for this group?%s' % (group_operators,
                                                                                            input_string)

        available_query_string = '\nYour available queries are:\n%s\n' % \
                                 '\n'.join(query_display_string % (query_num, *_query)
                                           for query_num, _query in enumerate(terminal_queries, 1))  # , terminal_query in enumerate(terminal_group_queries))
        # available_grouping_string = 'Your available groups are:\n%s\n\n' % \
        #                             '\n'.join('\tGroup Group #%d%s' %
        #                                       (i, format_string % group) for i, group in enumerate(work_on_group, 1))
        #  %s % available_query_string)
        if recursive_depth == 0:
            intro_string = group_introduction
            available_entity_string = available_query_string
        else:
            intro_string = group_grouping_intro
            available_entity_string = '\nYour available groups are:\n%s\n' % \
                                      '\n'.join('\tGroup Group #%d%s' % (i, format_string % group)
                                                for i, group in enumerate(work_on_group, 1))

        print(intro_string)  # provide an introduction
        print(available_entity_string)  # display available entities which switch between guery and group...

        selected_grouping_indices = deepcopy(all_grouping_indices)
        groupings = []
        while len(selected_grouping_indices) > 1:  # check if more work needs to be done
            while True:  # ensure grouping input is viable
                while True:
                    grouping = set(map(int, input(group_inquiry_string).split()))  # get new grouping
                    if len(grouping) > 1:
                        break
                    else:
                        print('More than one group is required. Your group %s is invalid' % grouping)
                while True:
                    confirm = input('%s\n%s' % (group_specification_string % grouping, confirmation_string))
                    if confirm.lower() in bool_d:
                        break
                    else:
                        print(invalid_string)

                if bool_d[confirmation.lower()]:  # confirm that grouping is as specified
                    while True:  # check if logic input is viable
                        group_logic = input(group_logic_string).lower()
                        if group_logic in group_operators:
                            break
                        else:
                            print(invalid_string)
                    groupings.append((grouping, group_logic))
                    break
            selected_grouping_indices -= grouping  # remove specified from the pool of available until all are gone

        if len(selected_grouping_indices) > 0:
            groupings.append((selected_grouping_indices, 'and'))  # When only 1 remains, automatically add and

        args += (groupings,)
        # once all groupings are grouped, recurse
        if len(groupings) > 1:
            make_groups(*args, recursive_depth=recursive_depth + 1)

        return list(args)

    # Start the user input routine -------------------------------------------------------------------------------------
    print('\n\nThis function will walk you through generating a PDB advanced search query and retrieving the matching '
          'set of PDB ID\'s. If you want to take advantage of a GUI to do this, you can visit:\n%s\n\n'
          'This function takes advantage of the same functionality, but automatically parses the returned ID\'s for '
          'downstream use. If you require > 25,000 ID\'s, this will save you some headache. You can also use the GUI '
          'and this tool in combination, as detailed below. Type \'JSON\' into the next prompt to do so, otherwise hit '
          '\'Enter\'\n\n'
          'DETAILS: If you want to use this function to save time formatting and/or pipeline interruption,'
          ' a unique solution is to build your Query with the GUI on the PDB then bring the resulting JSON back here to'
          ' submit. To do this, first build your full query, then hit \'Enter\' or the Search icon button '
          '(magnifying glass icon). A new section of the search page should appear above the Query builder. '
          'Clicking the JSON|->| button will open a new page with an automatically built JSON representation of your '
          'query. You can copy and paste this JSON object into the prompt to return your chosen ID\'s' %
          pdb_advanced_search_url)
    program_start = input(input_string)
    if program_start.upper() == 'JSON':
        # TODO get a method for taking the pasted JSON and formatting accordingly. Pasting now is causing enter on input
        json_input = input('Paste your JSON object below. IMPORTANT select from the opening \'{\' to '
                           '\'"return_type": "entry"\' and paste. Before hitting enter, add a closing \'}\'. This hack '
                           'ensure ALL results are retrieved with no sorting or pagination applied\n\n%s' %
                           input_string)
        response_d = query_pdb(json_input)
    else:
        return_identifier_string = '\nFor each set of options, choose the option from the first column for the ' \
                                   'description in the second.\nWhat type of identifier do you want to search the PDB '\
                                   'for?%s%s' % (user_input_format % '\n'.join(format_string % item
                                                                               for item in return_types.items()),
                                                 input_string)
        while True:
            return_type = input(return_identifier_string)
            if return_type in return_types:
                break
            else:
                print(invalid_string)

        terminal_group_queries = []
        # terminal_group_queries = {}
        increment = 1
        while True:
            query_builder_service_string = '\nWhat type of search method would you like to use?%s%s' % \
                                           (user_input_format % '\n'.join(format_string % item
                                                                          for item in services.items()), input_string)
            query_builder_attribute_string = '\nWhat type of attribute would you like to use? Examples include:%s' \
                                             '\n\nFor a more thorough list please search %s\nEnsure that your spelling'\
                                             ' is exact if you want your query to succeed!%s' % \
                                             (user_input_format % '\n'.join(format_string %
                                                                            (value % attribute_prefix, key)
                                                                            for key, value in attributes.items()),
                                              attribute_url, input_string)
            query_builder_operator_string = '\nWhat operator would you like to use?\n' \
                                            'Common operators include:\n\t%s%s' % (', '.join(operators), input_string)
            query_builder_value_string = '\nWhat value should be %s?%s'
            query_display_string = 'Query #%d: Search the PDB\'s \'%s\' service for \'%s\' attributes, \'%s\' \'%s\'.'

            while True:
                while True:
                    service = input(query_builder_service_string)
                    if service in services:
                        break
                    else:
                        print(invalid_string)

                # while True:  # Implement upon correct formatting check
                attribute = input(query_builder_attribute_string)
                while attribute.lower() == 's':
                    search_term = input('What term would you like to search?%s' % input_string)
                    attribute = input('Found the following instances of \'%s\':\n%s\nWhich option are you interested '
                                      'in?%s' % (search_term, user_input_format %
                                                 '\n'.join(format_string % item for item in search_schema(search_term)),
                                                 input_string))
                # TODO giant attribute dictionary with valid operators and value sets...
                operator = input(query_builder_operator_string)
                # TODO ensure for the attribute that the operator is valid!
                value = input(query_builder_value_string % (operator.upper(), input_string))
                if value.isdigit():
                    value = float(value)
                # TODO check if is.digit then conver to int()/float(). JSON can dumps() int/float

                while True:
                    confirmation = input('\n%s\n%s' % (query_display_string %
                                                       (increment, service.upper(), attribute, operator.upper(), value),
                                                       confirmation_string))
                    if confirmation.lower() in bool_d:
                        break
                    else:
                        print(invalid_string)
                if bool_d[confirmation.lower()] or confirmation.isspace():
                    break

            # terminal_group_queries[increment] = (service, attribute, operator, value)
            terminal_group_queries.append((service, attribute, operator, value))
            increment += 1
            while True:
                additional = input(additional_input_string)
                if additional.lower() in bool_d:
                    break
                else:
                    print(invalid_string)
            if not bool_d[additional.lower()]:
                break

        if len(terminal_group_queries) > 1:
            recursive_query_tree = make_groups(terminal_group_queries)
        else:
            recursive_query_tree = [terminal_group_queries]
            # recursive_query_tree = (terminal_group_queries, )
        # recursive_query_tree = (queries, grouping1, grouping2, etc.)
        for i, node in enumerate(recursive_query_tree):
            if i == 0:
                recursive_query_tree[i] = {j: generate_terminal_group(*leaf) for j, leaf in enumerate(node, 1)}
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
                recursive_query_tree[i] = {j: generate_group(operation, [recursive_query_tree[i - 1][k]
                                                                         for k in child_group_nums])
                                           for j, (child_group_nums, operation) in enumerate(node, 1)}
                                           # for k in child_group_nums}
        final_query = recursive_query_tree[-1][1]
        search_query = generate_query(final_query, return_type)
        response_d = query_pdb(search_query)
        print('The server returned:\n%s' % response_d)

    retrieved_ids = parse_pdb_response_for_ids(response_d)

    if save:
        io_save(retrieved_ids)
    if return_results:
        return retrieved_ids
    else:
        return None


def get_pdb_info_by_entry(entry):
    """Retrieve PDB information from the RCSB API

    Returns:
        (dict): {'entity': {1: ['A', 'B'], ...}, 'res': resolution, 'ref': {chain: {'accession': ID, 'db': UNP}, ...},
            'cryst': {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}}
    """
    # Connects the chain to the entity ID
    # entry = '4atz'
    entry_json = requests.get('http://data.rcsb.org/rest/v1/core/entry/%s' % entry).json()
    # The following information is returned
    # return_entry['rcsb_entry_info'] = \
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
    ang_a, ang_b, ang_c = entry_json['cell']['angle_alpha'], entry_json['cell']['angle_beta'], entry_json['cell']['angle_gamma']
    a, b, c = entry_json['cell']['length_a'], entry_json['cell']['length_b'], entry_json['cell']['length_c']
    space_group = entry_json['symmetry']['space_group_name_hm']
    cryst_d = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}
    resolution = entry_json['rcsb_entry_info']['resolution_combined'][0]

    entity_chain_d, ref_d, db_d = {}, {}, {}
    # I can use 'polymer_entity_count_protein' to further Identify the entities in a protein, which gives me the chains
    for i in range(1, len(entry_json['rcsb_entry_info']['polymer_entity_count_protein']) + 1):
        entity_json = requests.get('http://data.rcsb.org/rest/v1/core/polymer_entity/%s/%s' % (entry.upper(), i)).json()
        chains = entity_json["rcsb_polymer_entity_container_identifiers"]['asym_ids']  # = ['A', 'B', 'C']
        entity_chain_d[i] = chains

        try:
            uniprot_id = entity_json["rcsb_polymer_entity_container_identifiers"]['uniprot_ids']
            database = 'UNP'
            db_d = {'db': database, 'accession': uniprot_id}
        except KeyError:
            # GenBank = GB, which is mostly RNA or DNA structures or antibody complexes
            # Norine = NOR, which is small peptide structures, sometimes bound to proteins...
            identifiers = [[ident['database_accession'], ident['database_name']] for ident in
                           entity_json['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers']]

            if len(identifiers) > 1:  # we find the most ideal accession_database UniProt > GenBank > Norine > ???
                whatever_else = None
                priority_l = [None for i in range(len(identifiers))]
                for i, tup in enumerate(identifiers, 1):
                    if tup[1] == 'UniProt':
                        priority_l[0] = i
                        identifiers[i - 1][1] = 'UNP'
                    elif tup[1] == 'GenBank':
                        priority_l[1] = i  # two elements are required from above len check, never IndexError
                        identifiers[i - 1][1] = 'GB'
                    elif not whatever_else:
                        whatever_else = i
                for idx in priority_l:
                    if idx:  # we have found a database from the priority list, choose the corresponding identifier idx
                        db_d = {'accession': identifiers[idx - 1][0], 'db': identifiers[idx - 1][1]}
                        break
                    else:
                        db_d = {'accession': identifiers[whatever_else - 1][0], 'db': identifiers[whatever_else - 1][1]}
            else:
                db_d = {'accession': identifiers[0], 'db': identifiers[1]}

        ref_d = {chain: db_d for chain in chains}
    # dbref = {chain: {'db': db, 'accession': db_accession_id}}
    # OR dbref = {entity: {'db': db, 'accession': db_accession_id}}
    # cryst = {'space': space_group, 'a_b_c': (a, b, c), 'ang_a_b_c': (ang_a, ang_b, ang_c)}

    return {'entity': entity_chain_d, 'res': resolution, 'ref': ref_d, 'cryst': cryst_d}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for entries\n')
    parser.add_argument('-f', '--file_list', type=str, help='path/to/pdblist.file. Can be newline or comma separated.')
    parser.add_argument('-d', '--download', type=bool, help='Whether files should be downloaded. Default=False',
                        default=False)
    parser.add_argument('-p', '--input_pdb_directory', type=str, help='Where should reference PDB files be found? '
                                                                      'Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input_pisa_directory', type=str, help='Where should reference PISA files be found? '
                                                                       'Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output_directory', type=str, help='Where should interface files be saved?')
    parser.add_argument('-q', '--query_web', action='store_true',
                        help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    logger = start_log(name=os.path.basename(__file__), level=2)

    # Input data/files
    pdb_directory = '/databases/pdb'  # ends with .ent not sub_directoried
    pisa_directory = '/home/kmeador/yeates/fragment_database/all/pisa_files'
    current_interface_file_path = '/yeates1/kmeador/fragment_database/current_pdb_lists/'  # Todo parameterize
    # pdb_directory = '/yeates1/kmeador/fragment_database/all_pdb'  # sub directoried
    # pdb_directory = 'all_pdbs'

    retrieve_pdb_entries_by_advanced_query()
