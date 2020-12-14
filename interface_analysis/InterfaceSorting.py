import argparse
import os
import time
from copy import deepcopy
from itertools import chain as iter_chain
from json import dumps
from xml.etree.ElementTree import fromstring

import ParsePisa as pp
import requests
from symdesign.SymDesignUtils import start_log, pickle_object, unpickle, get_all_pdb_file_paths, to_iterable, read_pdb, \
    fill_pdb, retrieve_pdb_file_path, download_pisa, io_save

# Globals
pisa_type_extensions = {'multimers': '.xml', 'interfaces': '.xml', 'multimer': '.pdb', 'pisa': '.pkl'}
pdb_query_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
pdb_rest_url = 'http://data.rcsb.org/rest/v1/core/'  # uniprot/  # 1AB3/1'
attribute_url = 'https://search.rcsb.org/search-attributes.html'
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
             'contains_phrase', 'range', 'exists', }
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


def query_pdb(query):
    return requests.get(pdb_query_url, params={'json': dumps(query)}).json()


def retrieve_pdb_entries_by_advanced_query(save=True, return_results=True):
    pdb_advanced_search_url = 'http://www.rcsb.org/search/advanced'
    format_string = '\t%s\t\t%s'
    input_string = '\nInput:'
    invalid_string = 'Invalid choice, please try again'
    confirmation_string = 'If this is correct, indicate \'y\', if not \'n\', and you can re-input.%s' % input_string
    bool_d = {'y': True, 'n': False}
    user_input_format = '\n%s\n%s' % (format_string % ('Option', 'Description'), '%s')
    return_identifier_string = 'What type of identifier do you want to search the PDB for?\n%s\nInput:' % \
                               ('\n'.join(format_string % item for item in return_types.items()))
    additional_input_string = 'Would you like to add another?'

    def generate_parameters(attribute, operator, value):
        return {"operator": operator, "value": value, "attribute": attribute}

    def generate_terminal_group(service, parameter_args):
        return {'type': 'terminal', 'service': service, 'parameters': generate_parameters(*parameter_args)}

    def generate_group(operation, child_groups):
        return {'type': 'group', 'operation': operation, 'nodes': [child_groups]}

    def generate_query(search, return_type, return_all=True):
        query_d = {'query': search, 'return_type': return_type}
        if return_all:  # Todo test this as far as true being capitalized or not (example wasn't)
            query_d.update({'request_options': {'return_all_hits': True}})

        return query_d

    def make_groups(args, recursive_depth=0):  # (terminal_queries, grouping,
        terminal_queries = args[0]
        work_on_group = args[recursive_depth]
        all_grouping_indices = {i for i in range(1, len(work_on_group) + 1)}

        group_introduction = 'GROUPING INSTRUCTIONS:\n' \
                             'Because you have %d search queries, you need to combine these to a total search strategy.' \
                             ' This is accomplished by grouping your search queries together using the operations (%s) ' \
                             '\nYou must eventually group all queries into a single logical operation. ' \
                             '\nIf you have multiple groups, you will need to group those groups, so on and so forth.' \
                             '\nIndicate your group selections with a space separated list! You choose the group ' \
                             'operation to combine this list afterwards' \
                             '\nFollow the prior prompts if you need a reminder of how group numbers relate to query #' % \
                             (len(terminal_queries), group_operators)
        group_grouping_intro = 'Groups remain, you must group groups as before.'
        group_inquiry_string = 'Which of these (identified by #) would you like to combine into a group?%s' % input_string
        group_specification_string = 'You specified \'%s\' as a single group.'
        group_logic_string = 'What group operator (out of %s) would you like for this group?%s' % \
                             (','.join(group_operators), input_string)

        available_query_string ='Your available queries are:\n%s\n\n' % \
                                '\n'.join(query_display_string % (query_num, terminal_queries[query_num])
                                          for query_num in terminal_queries)  # , terminal_query in enumerate(terminal_group_queries))
        available_grouping_string = 'Your available groups are:\n%s\n\n' % \
                                    '\n'.join('\tGroup Group #%d%s' %
                                              (i + 1, format_string % group) for i, group in enumerate(work_on_group))
        #  %s % available_query_string)
        if recursive_depth == 0:
            intro_string = group_introduction
            available_entity_string = available_query_string
        else:
            intro_string = group_grouping_intro
            available_entity_string = available_grouping_string

        print(intro_string)  # provide an introduction
        print(available_entity_string)  # display available entities which switch between guery and group...

        selected_grouping_indices = deepcopy(all_grouping_indices)
        groupings = []
        while selected_grouping_indices:  # check if more work needs to be done
            while True:  # ensure grouping input is viable
                grouping = set(map(int, input(group_inquiry_string).split()))  # get new grouping
                confirmation = input('%s\n%s' % (group_specification_string % grouping, confirmation_string))
                if bool_d[confirmation.lower()]:  # confirm that grouping is as specified
                    while True:  # check if logic input is viable
                        group_logic = input(group_logic_string).lower()
                        if group_logic in group_operators:
                            break
                        else:
                            print(invalid_string)
                    groupings.append((group_logic, grouping))
                    break
                else:
                    print(invalid_string)
            selected_grouping_indices -= grouping  # remove specified from the pool of available until all are gone

        args += (groupings,)
        # once all groupings are grouped, recurse
        if len(groupings) > 1:
            make_groups(*args, recursive_depth=recursive_depth + 1)

        return args

    # Start the user input routine -------------------------------------------------------------------------------------
    print('This function will walk you through generating an advanced search query and retrieving the matching set of '
          'PDB ID\'s. If you want to take advantage of a GUI to do this, you can visit:\n%s\n\n'
          'This function takes advantage of the same functionality, but automatically parses the returned ID\'s for '
          'downstream use. If you require > 25,000 ID\'s, this could save you some headache. You can also use the GUI '
          'and this tool in combination, as detailed below. Just type \'JSON\' into the next prompt.\n\n'
          'DETAILS: If you want to use this function to save time formatting and/or pipeline interruption,'
          ' a unique solution is to build your Query with the GUI on the PDB then bring the resulting JSON back here to'
          ' submit. To do this, first build your full query, then hit \'enter\' or the search button (magnifying glass '
          'icon). A new section of the search page should appear above the Query builder. Clicking the JSON|->| button '
          'will open a new page with an automatically built JSON representation of your query. You can copy and paste '
          'this JSON object into the prompt to return your chosen ID\'s' % pdb_advanced_search_url)
    program_start = input(input_string)
    if program_start.upper() == 'JSON':
        json_input = input('Paste your JSON object below. IMPORTANT select from the opening \'{\' to '
                           '\'"return_type": "entry"\' and paste. Before hitting enter, add a closing \'}\'. This hack '
                           'ensure ALL results are retrieved with no sorting or pagination applied\n\n%s' % input_string)
        response_d = query_pdb(json_input)
    else:
        print('For each set of options, choose the option from the first column for the description in the second')
        while True:
            return_type = input(return_identifier_string)
            if return_type in return_types:
                break
            else:
                print(invalid_string)

        terminal_group_queries = []
        increment = 1
        while True:
            query_builder_service_string = 'What type of search method would you like to use?%s%s' % \
                                           (user_input_format % '\n'.join(format_string % item
                                                                          for item in services.items()), input_string)
            query_builder_attribute_string = 'What type of attribute would you like to use? Examples include:%s' \
                                             '\n\nFor a more thorough list please search %s\n ' \
                                             'Ensure that your spelling is exact if you want your query to succeed!%s' % \
                                             (user_input_format % '\n'.join(format_string %
                                                                            (key, value % attribute_prefix)
                                                                            for key, value in attributes.items()),
                                              attribute_url, input_string)
            # query_builder_attribute_string = 'What type of attribute would you like to use? Examples include\n%s' \
            #                                  '\n\nFor a more thorough list please search %s\n ' \
            #                                  'Ensure that your spelling is exact if you want your query to succeed!' \
            #                                  '\nInput:' % \
            #                                 ('\n'.join(format_string % (key, value % attribute_prefix)
            #                                            for key, value in attributes.items()), attribute_url)
            query_builder_operator_string = 'What is the operator that you would like to use?\n' \
                                            'Common operators include:\n%s%s' % (','.join(operators), input_string)
            query_builder_value_string = 'What value are should be %s?%s'
            query_display_string = 'Query #%d: Search the PDB\'s \'%s\' service for \'%s\' attributes, \'%s\' \'%s\'.\n'

            while True:
                while True:
                    service = input(query_builder_service_string)
                    if service in services:
                        break
                    else:
                        print(invalid_string)

                # while True:  # Implement upon correct formatting check
                attribute = input(query_builder_attribute_string)
                operator = input(query_builder_operator_string)
                value = input(query_builder_value_string % (operator.upper(), input_string))

                # while True:
                confirmation = input('%s\n%s' % (query_display_string %
                                                 (increment, service.upper(), attribute, operator.upper(), value),
                                                 confirmation_string))
                if bool_d[confirmation.lower()]:
                    break
                else:
                    print(invalid_string)

            terminal_group_queries.append((service, attribute, operator, value))
            increment += 1
            additional = input(additional_input_string)
            if not bool_d[additional.lower()]:
                break
            else:
                print(invalid_string)

        if len(terminal_group_queries) > 1:
            recursive_query_tree = make_groups(terminal_group_queries)
            # available_query_indices = {i for i in range(1, len(terminal_group_queries) + 1)}
            # group_introduction = 'Because you have %d search queries, you need to combine these to a total search strategy. ' \
            #                      'This is accomplished by grouping your search queries together using the operations (%s) ' \
            #                      'as you did previously.\nYou can repeat this process until all queries have been grouped. ' \
            #                      'If you have multiple groups, you will need to group the groups.' % \
            #                      (len(terminal_group_queries), operators),
            #
            # available_query_string ='Your available queries are::\n%s\n\n' % \
            #                         '\n'.join(query_display_string % (query_num, terminal_group_queries[query_num])
            #                                   for query_num in available_query_indices)  # , terminal_query in enumerate(terminal_group_queries))
            # group_inquiry_string = 'Which of these queries (identified by Query #) would you like to combine into a group?' \
            #                        'Indicate your selection with a space separated list! You will be able to choose the ' \
            #                        'group logical operation next%s' % input_string
            # group_specification_string = 'You specified queries \'%s\' as a single group.'
            # print(group_introduction)  # provide an introduction
            #
            # selected_query_indices = available_query_indices
            # groupings = []
            # while selected_query_indices:  # check if more work needs to be done
            #     while True:  # ensure grouping input is viable
            #         print(available_query_string)  # display available queries
            #         grouping = set(map(int, input(group_inquiry_string).split()))  # get new grouping
            #
            #         confirmation = input('%s\n%s' % (group_specification_string % grouping, confirmation_string))
            #         if bool_d[confirmation.lower()]:  # confirm that grouping is as specified
            #             while True:  # check if logic input is viable
            #                 group_logic_string = 'What group operator (out of %s) would you like for this group?%' % \
            #                                      (','.join(group_operators), input_string)
            #                 group_logic = input(group_logic_string).lower()
            #                 if group_logic in group_operators:
            #                     break
            #                 else:
            #                     print(invalid_string)
            #             groupings.append((group_logic, grouping))
            #             break
            #         else:
            #             print(invalid_string)
            #     selected_query_indices -= grouping  # remove specified from the pool of available until all are gone
            #
            # # once all groupings are grouped
            # if len(groupings) > 1:
            #     available_grouping_indices = {i for i in range(1, len(groupings) + 1)}
            #     group_grouping_string = 'Groups remain, you must group groups. Your groups are:\n%s\n\n%s' % \
            #                             ('\n'.join('\tGroup Group #%d%s' %
            #                                        (i + 1, format_string % group) for i, group in enumerate(groupings)),
            #                              available_query_string)
            #     print(group_grouping_string)  # provide an introduction
            #     selected_grouping_indices = available_grouping_indices
            #     while selected_grouping_indices:  # check if more work needs to be done
            #         group_grouping = input()
        else:
            recursive_query_tree = (terminal_group_queries, )
        # recursive_query_tree = (queries, grouping1, grouping2, etc.)
        for i, node in enumerate(recursive_query_tree):
            if i == 0:
                recursive_query_tree[i] = {j: generate_terminal_group(*leaf) for j, leaf in enumerate(node, 1)}
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
                recursive_query_tree[i] = {j: generate_group(operation, recursive_query_tree[i - 1][k])
                                           for j, (operation, child_group_nums) in enumerate(node, 1)
                                           for k in child_group_nums}

        search_query = generate_query(recursive_query_tree[-1], return_type)
        response_d = query_pdb(search_query)

    retrieved_ids = parse_pdb_response_for_ids(response_d)

    if save:
        io_save(retrieved_ids)
    if return_results:
        return retrieved_ids
    else:
        return None


def get_uniprot_accession_id(response_xml):  # DEPRECIATED
    root = fromstring(response_xml)
    #     for el in list(root.getchildren()[0]):
    #         if 'dbSource' in el.attrib:
    #             if el.attrib['dbSource'] == 'UniProt':
    #                 return el.attrib['dbAccessionId']

    #     return 0

    return next((item.attrib['dbAccessionId'] for item in list(root.getchildren()[0]) if 'dbSource' in item.attrib if
                 item.attrib['dbSource'] == 'UniProt'), None)


def map_pdb_to_uniprot(pdb_chain):  # DEPRECIATED
    uniprot_id = 0
    retry = False
    while uniprot_id == 0:
        pdb_mapping_response = requests.get(pdb_query_url, params={'json': dumps(query)}).text
        # pdb_mapping_response = requests.get(pdb_query_url, params={'query': pdb_chain}).text

        if pdb_mapping_response[:3] == 'Bad':
            return False
        else:
            uniprot_id = get_uniprot_accession_id(pdb_mapping_response)
            if uniprot_id == 0:
                if retry:
                    print(pdb_chain, pdb_mapping_response)
                    print('Failed again. Sleep for a minute...')
                    time.sleep(60)
                time.sleep(10)
                print('Retry Request')
                retry = True

    return uniprot_id  # {'pdb_id': pdb_chain, 'uniprot_id': uniprot_id, 'uniprot_name': uniprot_name}



def return_pdb_interface(pdb_code, interface_id, full_chain=True, db=False):
    try:
        # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
        # This makes flexible with MySQL
        if not db:
            pdb_file_path = retrieve_pdb_file_path(pdb_code, directory=pdb_directory)
            pisa_file_path = retrieve_pisa_file_path(pdb_code, directory=pisa_directory)
            source_pdb = read_pdb(pdb_file_path)
            pisa_data = unpickle(pisa_file_path)  # Get PISA data
        else:
            print("Connection to MySQL DB not yet supported")
            exit()

        interface_data = pisa_data['interfaces']
        interface_chain_data = pisa_data['interfaces'][interface_id]['chain_data']
        interface = extract_interface(pdb, interface_chain_data, full_chain=full_chain)

        return interface

    except Exception as e:
        print(e.__doc__)
        print(e, pdb_code)

        return pdb_code


def extract_interface(pdb, chain_data_d, full_chain=True):
    """
    'interfaces': {interface_ID: {interface stats, {chain data}}, ...}
        Ex: {1: {'occ': 2, 'area': 998.23727478, 'solv_en': -11.928783903, 'stab_en': -15.481081211,
             'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104, 'int_res': {'87': 23.89, '89': 45.01, ...},
                            2: ...}},
             2: {'occ': ..., },
             'all_ids': {interface_type: [interface_id1, matching_id2], ...}
            } interface_type and id connect the interfaces that are the same, but present in multiple PISA complexes

    """
    # if one were trying to map the interface fragments created in a fragment extraction back to the pdb, they would
    # want to use the interface in interface_data and the chain id (renamed from letter to ID #) to find the chain and
    # the translation from the pisa.xml file
    # pdb_code, subdirectory = return_and_make_pdb_code_and_subdirectory(pdb_file_path)
    # out_path = os.path.join(os.getcwd(), subdirectory)
    # try:
    #     # If the location of the PDB data and the PISA data is known the pdb_code would suffice.
    #     # This makes flexible with MySQL
    #     source_pdb = read_pdb(pdb_file_path)
    #     pisa_data = unpickle(pisa_file_path)  # Get PISA data
    #     interface_data = pisa_data['interfaces']
    #     # interface_data, chain_data = pp.parse_pisa_interfaces_xml(pisa_file_path)
    #     for interface_id in interface_data:
    #         if not interface_id.is_digit():  # == 'all_ids':
    #             continue
    # interface_pdb = PDB.PDB()
    temp_names = ('.', ',')
    interface_chain_pdbs = []
    temp_chain_d = {}
    for temp_name_idx, chain_id in enumerate(chain_data_d):
        # chain_pdb = PDB.PDB()
        chain = chain_data_d[chain_id]['chain']
        if not chain:  # for instances of ligands, stop process, this is not a protein-protein interface
            break
        else:
            if full_chain:  # get the entire chain
                interface_atoms = deepcopy(pdb.chain(chain))
            else:  # get only the specific residues at the interface
                residues = chain_data_d[chain_id]['int_res']
                interface_atoms = []
                for residue_number in residues:
                    residue_atoms = pdb.getResidueAtoms(chain, residue_number)
                    interface_atoms.extend(deepcopy(residue_atoms))
                # interface_atoms = list(iter_chain.from_iterable(interface_atoms))
            chain_pdb = fill_pdb(interface_atoms)
            # chain_pdb.read_atom_list(interface_atoms)

            rot = chain_data_d[chain_id]['r_mat']
            trans = chain_data_d[chain_id]['t_vec']
            chain_pdb.apply(rot, trans)
            chain_pdb.rename_chain(chain, temp_names[temp_name_idx])  # ensure that chain names are not the same
            temp_chain_d[temp_names[temp_name_idx]] = str(chain_id)
            interface_chain_pdbs.append(chain_pdb)
            # interface_pdb.read_atom_list(chain_pdb.all_atoms)

    interface_pdb = fill_pdb(iter_chain.from_iterable([chain_pdb.all_atoms for chain_pdb in interface_chain_pdbs]))
    if len(interface_pdb.chain_id_list) == 2:
        for temp_name in temp_chain_d:
            interface_pdb.rename_chain(temp_name, temp_chain_d[temp_name])

    return interface_pdb


# Todo PISA path
def retrieve_pisa_file_path(pdb_code, directory='/home/kmeador/yeates/fragment_database/all/pisa_files', file_type='pisa'):
    """Returns the PISA path that corresponds to the PDB code from the local PISA Database.
    Attempts a download if the files are not found
    """
    if file_type in pisa_type_extensions:
        pdb_code = pdb_code.upper()
        sub_dir = pdb_code[1:3].lower()
        root_path = os.path.join(directory, sub_dir)
        specific_file = ' %s_%s%s' % (pdb_code, file_type, pisa_type_extensions[file_type])
        if os.path.exists(os.path.join(root_path, specific_file)):
            return os.path.join(root_path, specific_file)
        else:  # attempt to make the file if not pickled or download if requisite files don't exist
            if file_type == 'pisa':
                downloaded = False
                while True:
                    status = extract_pisa_files_and_pickle(root_path, pdb_code)
                    if status:
                        return os.path.join(root_path, specific_file)
                    else:  # try to download required files
                        if not downloaded:
                            logger.info('Attempting to download PISA files')
                            for pisa_type in pisa_type_extensions:
                                download_pisa(pdb, pisa_type, out_path=directory)
                            downloaded = True
                        else:
                            break

    return None


def extract_pisa_files_and_pickle(root, pdb_code):
    """Take a set of pisa files acquired from pisa server and parse for saving as a complete pisa dictionary"""
    individual_d = {}
    try:
        interface, chain = pp.parse_pisa_interfaces_xml(os.path.join(root, '%s_interfaces.xml' % pdb_code))
        individual_d['interfaces'] = interface
        individual_d['chains'] = chain
        individual_d['multimers'] = pp.parse_pisa_multimers_xml(os.path.join(root, '%s_multimers.xml' % pdb_code))
    except OSError:
        return False

    pickle_object(individual_d, '%s_pisa' % pdb_code, out_path=root)

    return True


def set_up_interface_dict(pdb_interface_codes):
    """From all PDB interfaces, make a dictionary of the PDB: [interface number] pairs

    Returns:
        {pdb_code: [1, 3, 4], ...}
    """
    int_dict = {}
    for code in pdb_interface_codes:
        pdb_code = code.split('-')
        # ['1ABC', '3']
        if pdb_code[0] in int_dict:
            int_dict[pdb_code[0].upper()].add(int(pdb_code[1]))
        else:
            int_dict[pdb_code[0].upper()] = set(int(pdb_code[1]))

    return int_dict


def sort_pdb_interfaces_by_contact_type(pisa_d, interface_number_set, assembly_confirmed):
    """From a directory of interface pdbs with structure interface_pdbs/ab/1ABC-3.pdb
        grab the multimeric pisa xml file and search for all biological assemblies

    """
    pisa_multimer = pisa_d['multimers']
    pisa_interfaces = pisa_d['interfaces']

    bio_ass = {}
    if pisa_multimer[(1, 1)] != 'MONOMER':  # if the pdb is not a monomer
        for set_complex in pisa_multimer:
            biological_assembly = pisa_multimer[set_complex]['pdb_BA']
            if biological_assembly != 0:  # if set_complex pair is a pdb_BiologicalAssembly
                if biological_assembly in bio_ass:  # is it already in BA dict?
                    bio_ass[biological_assembly].add(set_complex)
                else:  # add to the BA dict
                    bio_ass[biological_assembly] = {set_complex}

        # Grab all unique interface numbers from PDB assigned BA invoked in multimer entries
        possible_bio_int = []
        for ba in bio_ass:
            for set_complex in bio_ass[ba]:
                possible_bio_int += pisa_multimer[set_complex]['interfaces'].keys()

        other_candidates = []
        for candidate_int in possible_bio_int:
            for int_type in pisa_interfaces['all_ids']:
                if candidate_int in pisa_interfaces['all_ids'][int_type]:  # check if the interface has matching interfaces
                    other_candidates += pisa_interfaces['all_ids'][int_type]
        all_possible_bio_int = set(possible_bio_int) | set(other_candidates)

        # Finally, check if the possible biological_interfaces have confirmed assemblies
        if assembly_confirmed != set():  # there are assemblies confirmed
            # Only use ba if they are in QSBio
            not_bio_int = []
            for ba in bio_ass:
                # Todo make ba an int in qsbio_confrimed
                if str(ba) not in assembly_confirmed:  # ensure biological assembly is QSBio confirmed assembly
                    for set_complex in bio_ass[ba]:
                        not_bio_int += pisa_multimer[set_complex]['interfaces'].keys()
                        # for interface in pisa_multimer[set_complex]['interfaces'].keys():
                        #     all_possible_bio_int.remove(interface)

            final_unknown_bio_int = set(not_bio_int)
            # take the intersection (&) of interface_number_set and bio_int
            final_bio_int = interface_number_set & (all_possible_bio_int - final_unknown_bio_int)
            final_xtal_int = interface_number_set - final_bio_int
        else:  # the pdb does not have a confirmed assembly
            # take the difference (-) of interface_number_set and all_possible_bio_int
            final_bio_int = set()
            final_unknown_bio_int = all_possible_bio_int
            final_xtal_int = interface_number_set - final_unknown_bio_int  # all interface identified, minus possible BA

    return {'bio': final_bio_int, 'xtal': final_xtal_int, 'unknown_bio': final_unknown_bio_int}


def sort_pdbs_to_uniprot_d(pdbs, pdb_uniprot_d):
    unp_master = {}
    no_unpid = []
    for pdb in pdbs:
        for chain in pdb_uniprot_d[pdb]['ref']:
            if pdb_uniprot_d[pdb]['ref'][chain]['db'] == 'UNP':  # only add the pdb and chain if corresponding UniProtID
                unpid = pdb_uniprot_d[pdb]['ref'][chain]['accession']
                if unpid in unp_master:
                    unp_master[unpid]['all'].append(pdb + '.' + chain)
                    unp_master[unpid]['unique_pdb'].add(pdb)
                    space_group = pdb_uniprot_d[pdb]['cryst']['space']
                    if space_group in unp_master[unpid]['space_groups']:
                        unp_master[unpid]['space_groups'][space_group]['all'].append(pdb)
                    else:
                        unp_master[unpid]['space_groups'][space_group] = {'all': [pdb], 'top': None}
                else:
                    unp_master[unpid] = {'all': ['%s.%s' % (pdb, chain)], 'unique_pdb': {pdb}, 'top': None,
                                         'bio': set(), 'partners': {}, 'space_groups': {
                            pdb_uniprot_d[pdb]['cryst']['space']: {'all': [pdb], 'top': None}}}
            else:
                no_unpid.append('%s.%s' % (pdb, chain))

    return unp_master, no_unpid


def process_uniprot_entry(uniprot_id, unp_d, pdb_uniprot_info, min_resolution_threshold=3.0):
    # for unpid in unp_master:
    # unp_master[unpid]['unique_pdb'] = list(set(unp_master[unpid]['unique_pdb']))
    min_res = min_resolution_threshold
    top_pdb = None
    for pdb in unp_d['unique_pdb']:
        # Find Uniprot ID with top Resolution
        if pdb_uniprot_info[pdb]['res'] <= min_res:  # structural resolution is less than threshold
            top_pdb = pdb
            min_res = pdb_uniprot_info[pdb]['res']

        # Check for pdbs which comprise Self and other Uniprot IDs. These represent distinct heteromeric interfaces
        if len(pdb_uniprot_info[pdb]['ref']) > 1:  # Check if there is more than one chain in the pdb
            for chain in pdb_uniprot_info[pdb]['ref']:
                if pdb_uniprot_info[pdb]['ref'][chain]['db'] == 'UNP':  # only add if there is corresponding UniProtID
                    partner_unpid = pdb_uniprot_info[pdb]['ref'][chain]['accession']
                    if partner_unpid != uniprot_id:  # only get other Uniprot IDs
                        if partner_unpid in unp_d['partners']:
                            unp_d['partners'][partner_unpid]['all'].append(pdb + '.' + chain)
                        else:
                            unp_d['partners'][partner_unpid] = {'all': [pdb + '.' + chain], 'top': None}

    # Save highest Resolution PDB Code and add to Biological Assembly list
    if top_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
        unp_d['top'] = top_pdb
    # unp_master[unpid]['bio'].add(top_pdb)  # Todo why? Could be a monomer

    # Add the highest resolution UniProt Partners to the Biological Assemblies set
    for partner in unp_d['partners']:
        min_partner_res = min_resolution_threshold
        top_partner = None
        for pdb in unp_d['partners'][partner]['all']:
            partner_res = pdb_uniprot_info[pdb.split('.')[0]]['res']
            if partner_res <= min_partner_res:
                top_partner = pdb
                min_partner_res = partner_res
        if top_partner:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['partners'][partner]['top'] = top_partner
            unp_d['bio'].add(top_partner.split('.')[0])

    # Save highest Resolution PDB Codes for each Space Group
    for cryst in unp_d['space_groups']:
        min_xtal_res = min_resolution_threshold
        top_xtal_pdb = None
        for pdb in unp_d['space_groups'][cryst]['all']:
            if pdb_uniprot_info[pdb]['res'] <= min_xtal_res:
                top_xtal_pdb = pdb
                min_xtal_res = pdb_uniprot_info[pdb]['res']
        if top_xtal_pdb:  # isn't necessary if filtered by min_resolution_threshold due to iterator
            unp_d['space_groups'][cryst]['top'] = top_xtal_pdb

    return unp_d


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Chain/Chain Interfaces from a PDB or PDB Library\n')
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


    # Variables
    pdb_resolution_threshold = 3.0
    write_to_file = True  # TODO MySQL DB option would make this false

    all_pdbs_of_interest = 'under3A_protein_no_multimodel_no_mmcif_no_bad_pisa.txt'  # Todo parameterize
    all_protein_file = os.path.join(current_interface_file_path, all_pdbs_of_interest)
    if query:
        # TODO Figure out how to query PDB using under3A no multimodel no mmcif, no bad pisa file reliably.
        retrieve_pdb_entries_by_advanced_query()
    else:
        pdbs_of_interest = to_iterable(all_protein_file)

    # Current
    interfaces_dir = '/yeates1/kmeador/fragment_database/all_interfaces'
    all_interface_pdb_paths = get_all_pdb_file_paths(interfaces_dir)
    pdb_interface_codes = list(file_ext_split[0]
                               for file_ext_split in map(os.path.splitext, map(os.path.basename, all_interface_pdb_paths)))
    pdb_interface_d = set_up_interface_dict(pdb_interface_codes)
    # Optimal
    pdb_interface_file_name = 'AllPDBInterfaces'
    pdb_interface_file = os.path.join(current_interface_file_path, pdb_interface_file_name)
    if os.path.exists(pdb_interface_file):  # retrieve the pdb, [interfaces] dictionary
        pdb_interface_d = unpickle(pdb_interface_file)
    else:  # make the dictionary
        if args.query_web:
            dummy = None  # TODO
        else:
            pdb_interface_d = {}
            for pdb_code in pdbs_of_interest:
                pisa_d = unpickle(retrieve_pisa_file_path(pdb_code, directory=pisa_directory))
                interface_data = pisa_d['interfaces']
                for interface_id in interface_data:
                    if interface_id.is_digit():
                        if pdb_code in pdb_interface_d:
                            pdb_interface_d[pdb_code].add(interface_id)
                        else:
                            pdb_interface_d[pdb_code] = {interface_id}

        pickle_object(pdb_interface_d, pdb_interface_file, out_path=None)

    # TODO script this file creation
    #  See if there is an update to QSBio verified assemblies
    qsbio_file_name = 'QSbio_GreaterThanHigh_Assemblies'  # .pkl'  # Todo parameterize
    qsbio_file = os.path.join(current_interface_file_path, qsbio_file_name)
    qsbio_confirmed_d = unpickle(qsbio_file)

    # TODO script this file creation ?
    qsbio_monomers_file_name = 'QSbio_Monomers.csv'  # Todo parameterize
    qsbio_monomers_file = os.path.join(current_interface_file_path, qsbio_monomers_file_name)
    # with open(qsbio_monomers_file, 'r') as f:
    #     all_lines = f.readlines()
    #     monomers = []
    #     for line in all_lines:
    #         line = line.strip()
    #         monomers.append(line)
    monomers = to_iterable(qsbio_monomers_file)
    print('The number of monomers found in %s: %d' % (qsbio_monomers_file, len(set(monomers))))

    # representatives_90_file_name = 'under3A_pdb_90_sequence_cluster_reps.txt'
    # representatives_90_file = os.path.join(current_interface_file_path, representatives_90_file_name)
    # rep_90 = to_iterable(representatives_90_file)

    all_pdb_uniprot_file_name = '200206_MostCompleteAllPDBSpaceGroupUNPResInfo'  # .pkl'  # Todo parameterize
    all_pdb_uniprot_file = os.path.join(current_interface_file_path, all_pdb_uniprot_file_name)
    # Dictionary Structure
    #  {'entity': {1: ['A']}, 'res': 2.6, 'ref': {'A': {'db': 'UNP', 'accession': 'P35755'})
    #   'cryst': {'space': 'P 21 21 2', 'a_b_c': (106.464, 75.822, 34.109), 'ang_a_b_c': (90.0, 90.0, 90.0)}}
    # 200121 has 'path': '/home/kmeador/yeates/fragment_database/all/all_pdbs/kn/3KN7.pdb', not entity
    if os.path.exists(all_pdb_uniprot_file):  # retrieve the pdb, DBreference, resolution, and crystal dictionary
        pdb_uniprot_info = unpickle(all_pdb_uniprot_file)
    else:
        if args.query_web:  # Retrieve from the PDB web
            pdb_uniprot_info = {pdb_code: get_pdb_info_by_entry(pdb_code) for pdb_code in all_pdbs_of_interest}
        else:  # From the database of files
            pdb_uniprot_info = {}
            for pdb_code in all_pdbs_of_interest:
                pdb = read_pdb(retrieve_pdb_file_path(pdb_code, directory=pdb_directory), coordinates_only=False)
                pdb_uniprot_info[pdb_code] = {'entity': pdb.entities, 'cryst': pdb.cryst, 'ref': pdb.dbref,
                                              'res': pdb.res}

        pickle_object(pdb_uniprot_info, all_pdb_uniprot_file, out_path=None)

    # Output data/files
    sorted_file_name = 'PDBInterfacesSorted'  # Todo parameterize
    uniprot_heterooligomer_interface_file_name = 'UniquePDBHeteroOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_homooligomer_interface_file_name = 'UniquePDBHomoOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_unknown_bio_interface_file_name = 'UniquePDBUnknownOligomerInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    uniprot_xtal_interface_file_name = 'UniquePDBXtalInterfaces'  # was 200121_FinalInterfaceDict  # Todo parameterize
    unique_chains_file_name = 'UniquePDBChains'  # was 200121_FinalIntraDict  # Todo parameterize

    # TODO Back-end: Figure out how to convert these sets into MySQL tables so that the PDB files present are
    #  easily identified based on their assembly. Write load PDB from MySQL function. Also ensure I have a standard
    #  query format to grab different table schemas depending on the step of the fragment processing.

    interface_sort_d = {}
    missing_pisa_paths = []
    for pdb_code in pdb_interface_d:
        pisa_path = retrieve_pisa_file_path(pdb_code)
        if pisa_path:
            pisa_d = unpickle(pisa_path)
            interface_sort_d[pdb_code] = sort_pdb_interfaces_by_contact_type(pisa_d, pdb_interface_d[pdb_code],
                                                                             set(qsbio_confirmed_d[pdb_code]))
        else:
            missing_pisa_paths.append(pdb_code)

    all_missing_biological = []
    all_biological = []
    for pdb in interface_sort_d:
        if interface_sort_d[pdb]['bio'] == set():
            all_missing_biological.append(pdb)
        else:
            all_biological.append(pdb)

    proposed_qsbio_monomer_set = []
    for pdb in all_missing_biological:
        if pdb in qsbio_confirmed_d:
            proposed_qsbio_monomer_set.append(pdb)

    confirmed_count = 0
    final_monomers = []
    starting_monomer_pdbs = []
    missing_from_confirmed = []
    missing_from_final = []
    for entry in monomers:
        pdb = entry.split('_')[0].upper()
        starting_monomer_pdbs.append(pdb)
        if pdb in qsbio_confirmed_d:
            confirmed_count += 1
        else:
            missing_from_confirmed.append(pdb)
        try:
            if interface_sort_d[pdb]['bio'] == set():
                final_monomers.append(pdb)
        except KeyError:
            missing_from_final.append(pdb)
    actual_monomers = set(final_monomers) & set(starting_monomer_pdbs)
    missing_monomers = set(final_monomers) - set(starting_monomer_pdbs)
    incorrect_assembly_assignment = list(set(proposed_qsbio_monomer_set) - set(final_monomers))
    print('Length of QSbio monomers (%d) should equal %d (all monomers confirmed in QSbio) and '
          'maybe monomers missing from final (%d). If not, then those missing from final (%d)'
          ' + final set monomers (%d) = %d should equal %d'
          % (len(monomers), confirmed_count, len(final_monomers), len(missing_from_final),
             len(final_monomers), len(missing_from_final) + len(final_monomers), len(monomers)))
    # '. It doesn\'t, so this is the difference', len(missing_from_confirmed), 'Some examples are:', missing_from_confirmed[:5])
    print('Number missing from the interface_dictionary = %d, with set = %d.' %
          (len(missing_from_final), len(set(missing_from_final))))
    print('There are %d PDB\'s missing Bio Assemblies and %d with Bio Assemblies.' %
          (len(all_missing_biological), len(all_biological)))
    print('Of those without Bio Assemblies, %d should be monomers per QSBio' %
          len(set(proposed_qsbio_monomer_set) & set(final_monomers)))
    print('And therefore %d are failures of the above processing.' % len(set(qsbio_monomer_set) - set(final_monomers)))

    # print('Unique monomeric PDB\'s in final_sorted_dict:', len(actual_monomers), '. There are', len(missing_monomers), 'failures in monomeric identification.')
    # print('The number of flawed monomer assignments, and therefore errors, in PISA parsing is', len(incorrect_assembly_assignment))
    print('Some examples of which include:', incorrect_assembly_assignment[:5])

    i = 0
    j = 0
    for entry in incorrect_assembly_assignment:
        if interface_sort_d[entry]['unknown_bio'] == set():
            i += 1
        if interface_sort_d[entry]['xtal'] != set():
            j += 1
    print(i, j)

    for entry in incorrect_assembly_assignment:
        interface_sort_d[entry]['unknown_bio'] = interface_sort_d[entry]['xtal']
        interface_sort_d[entry]['xtal'] = set()

    j = 0
    for entry in missing_from_final:
        try:
            i = interface_sort_d[entry]
        except KeyError:
            j += 1
    print('There are actually %d missing from the interface_dictionary that are in the monomers file. '
          'These should be outside of the interface selection criteria')

    k = 0
    for entry in missing_from_final:
        try:
            i = pdbs_of_interest.index()
        except:
            k += 1
    print('Infact, there are', k, 'missing from proteins of interest in the first place')

    sorted_interfaces_file = os.path.join(current_interface_file_path, sorted_file_name)
    # with open(final_file, 'wb') as f:
    #     pickle.dump(final_int_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle_object(interface_sort_d, sorted_interfaces_file, out_path=current_interface_file_path)

    # not_qsbio_file = '200121_pdb_lists/0204_not_qs_bio'  # .pkl'
    # with open(not_qsbio_file, 'wb') as f:
    #     pickle.dump(not_in_qsbio, f, pickle.HIGHEST_PROTOCOL)
    # pickle_object(not_in_qsbio, not_qsbio_file, out_path=current_interface_file_path)

    # NEXT Sorting the interfaces by UniProtID
    # Gather all UniProtIDs from the Representative file specified
    uniprot_sorted, no_unp_code = sort_pdbs_to_uniprot_d(pdbs_of_interest, pdb_uniprot_info)
    print('Total UniProtID\'s added: %d\nTotal without UniProtID: %s' % (len(uniprot_sorted), len(no_unp_code)))  # \nThose missing chain info: %d no_chains,

    uniprot_master = {uniprot_id: process_uniprot_entry(uniprot_id, uniprot_sorted[uniprot_id], pdb_uniprot_info,
                                                        min_resolution_threshold=pdb_resolution_threshold)
                      for uniprot_id in uniprot_sorted}

    # Find the unique UniProtIDs which are representative of the homo-oligomer and unknown PDB interfaces
    all_unknown_bio_interfaces, all_homo_bio_interfaces = {}, {}
    for pdb in interface_sort_d:
        if interface_sort_d[pdb]['unknown_bio'] != set():
            all_unknown_bio_interfaces[pdb] = interface_sort_d[pdb]['unknown_bio']
        if interface_sort_d[pdb]['bio'] != set():
            all_homo_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']

    # all_homo_uniprot_ids, all_unk_uniprot_ids = set(), set()
    all_homo_uniprot_ids, all_unk_uniprot_ids = {}, {}
    uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, unique_pdb_chains = {}, {}, {}
    for uniprot_id in uniprot_master:
        # Get the highest resolution heteromeric biological interfaces
        for pdb in uniprot_master[uniprot_id]['bio']:
            uniprot_hetero_bio_interfaces[pdb] = interface_sort_d[pdb]['bio']
        # Get the highest resolution pdb for each spaces group. These are assumed to comprise separate crystal contacts
        for space_group in uniprot_master[uniprot_id]['space_groups']:
            pdb = uniprot_master[uniprot_id]['space_groups'][space_group]['top']
            uniprot_filtered_xtal_interfaces[pdb] = interface_sort_d[pdb]['xtal']
        # Get all the highest resolution unique pdb and chain
        for pdb_chain in uniprot_master[uniprot_id]['all']:
            pdb = pdb_chain.split('.')[0]
            if pdb == uniprot_master[uniprot_id]['top']:
                unique_pdb_chains[pdb] = pdb_chain.split('.')[1]
                # unique_pdb_chains[uniprot_id] = pdb_chain  # Was this, but the format should match the others...
                break
        # From all unique_pdbs, see if they are in set of homo_ or unknown_bio interfaces
        for pdb in uniprot_master[uniprot_id]['unique_pdb']:
            if pdb in all_homo_bio_interfaces:
                if uniprot_id in all_homo_uniprot_ids:
                    all_homo_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_homo_uniprot_ids[uniprot_id] = {pdb}
                # all_homo_uniprot_ids.add(uniprot_id)
            if pdb in all_unknown_bio_interfaces:
                # all_unk_uniprot_ids.add(uniprot_id)
                if uniprot_id in all_unk_uniprot_ids:
                    all_unk_uniprot_ids[uniprot_id].add(pdb)
                else:
                    all_unk_uniprot_ids[uniprot_id] = {pdb}
                
    uniprot_unknown_bio_interfaces, uniprot_homo_bio_interfaces = {}, {}
    for uniprot_id in all_homo_uniprot_ids:
        # for pdb in all_homo_uniprot_ids[uniprot_id]:
        #     if pdb == uniprot_master[uniprot_id]['top']:
        #         uniprot_homo_bio_interfaces[pdb] = all_homo_bio_interfaces[pdb]
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_homo_uniprot_ids[uniprot_id]:
            uniprot_homo_bio_interfaces[top_pdb] = all_homo_bio_interfaces[top_pdb]
        else:
            highest_res = pdb_resolution_threshold
            highest_res_pdb = None
            for pdb in all_homo_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_homo_bio_interfaces[highest_res_pdb] = all_homo_bio_interfaces[highest_res_pdb]

    for uniprot_id in all_unk_uniprot_ids:
        top_pdb = uniprot_master[uniprot_id]['top']
        if top_pdb in all_unk_uniprot_ids[uniprot_id]:
            uniprot_unknown_bio_interfaces[top_pdb] = all_unknown_bio_interfaces[top_pdb]
        else:
            highest_res = pdb_resolution_threshold
            highest_res_pdb = None
            for pdb in all_unk_uniprot_ids[uniprot_id]:
                if pdb_uniprot_info[pdb]['res'] < highest_res:
                    highest_res = pdb_uniprot_info[pdb]['res']
                    highest_res_pdb = pdb
            uniprot_unknown_bio_interfaces[highest_res_pdb] = all_unknown_bio_interfaces[highest_res_pdb]

    print(len(uniprot_hetero_bio_interfaces), len(uniprot_filtered_xtal_interfaces), len(uniprot_unknown_bio_interfaces), len(unique_pdb_chains))
    print(len(set(unique_pdb_chains)))

    if write_to_file:
        # All have format {pdb_code: [1, 3], ...} where the list is interface numbers
        pickle_object(uniprot_hetero_bio_interfaces, uniprot_heterooligomer_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(uniprot_filtered_xtal_interfaces, uniprot_xtal_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(unique_pdb_chains, unique_chains_file_name, out_path=current_interface_file_path)

        pickle_object(uniprot_homo_bio_interfaces, uniprot_homooligomer_interface_file_name,
                      out_path=current_interface_file_path)
        pickle_object(uniprot_unknown_bio_interfaces, uniprot_unknown_bio_interface_file_name,
                      out_path=current_interface_file_path)

    # Old method
    interface_list = [uniprot_hetero_bio_interfaces, uniprot_filtered_xtal_interfaces, uniprot_unknown_bio_interfaces]
    # final_interface_sort = [[] for i in range(len(interface_list))]
    finalInterfaceDict = {'bio': [], 'xtal': [], 'unknown_bio': []}
    for i in range(len(interface_list)):
        for entry in interface_list[i]:
            for interface in interface_list[i][entry]:
                if i == 0:
                    finalInterfaceDict['bio'].append(entry + '-' + interface)
                elif i == 1:
                    finalInterfaceDict['xtal'].append(entry + '-' + interface)
                elif i == 2:
                    finalInterfaceDict['unknown_bio'].append(entry + '-' + interface)
    #             final_interface_sort[i].append(entry + '-' + interface)

    inverse_final_interface = {}
    i = 0
    total = 0
    for entry in finalInterfaceDict:
        total += len(set(finalInterfaceDict[entry]))
        for pdb in finalInterfaceDict[entry]:
            inverse_final_interface[pdb] = entry
            i += 1
    print('Length of Inverse Final: %d, Actual operations: %d, Total number considered: %d' % (
        len(inverse_final_interface), i, total))
