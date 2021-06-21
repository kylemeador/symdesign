"""Add expression tags onto the termini of specific designs"""
import csv
from itertools import chain as iter_chain  # combinations,

from Bio.SeqUtils import IUPACData

import PathUtils as PUtils
import SymDesignUtils as SDUtils
# from PDB import PDB
# import Pose
import SequenceProfile
from Query.PDB import input_string, get_entity_reference_sequence, pdb_id_matching_uniprot_id
from dependencies.DnaChisel.dnachisel import DnaOptimizationProblem, CodonOptimize, reverse_translate

# Globals
logger = SDUtils.start_log(name=__name__)
uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
with open(PUtils.affinity_tags, 'r') as f:
    expression_tags = {'_'.join(map(str.lower, row[0].split())): row[1] for row in csv.reader(f)}
default_multicistronic_sequence = \
    'taatgcttaagtcgaacagaaagtaatcgtattgtacacggccgcataatcgaaat' \
    'taatacgactcactataggggaattgtgagcggataacaattccccatcttagtatattagttaagtataagaaggagatatacat'
#    ^ Start of T7 promoter
#                       ^ start of LacO       ^ last nucleotid of LacO


def find_matching_expression_tags(uniprot_id=None, pdb_code=None, chain=None):
    """Take a pose and find expression tags from each PDB reference asking user for input on tag choice

    Args:
        uniprot_id=None (str): The uniprot_id to query tags from
        pdb_code=None (str): The pdb to query tags from. Requires chain argument as well
        chain=None (str): The chain to query tags from. Requires pdb argument as well
    Returns:
        (dict): {'n': {His Tag: 2}, 'c': {Spy Catcher: 1},
                 'matching_tags': [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]}
    """
    # uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
    if not uniprot_id:
        if not pdb_code or not chain:
            # raise AttributeError('One of uniprot_id or pdb_code AND chain is required')
            logger.error('One of uniprot_id or pdb_code AND chain is required')
            return {'n': {}, 'c': {}, 'matching_tags': []}
        uniprot_id = pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=chain)

    # from PDB API
    partner_sequences = [get_entity_reference_sequence(entity_id=entity_id)
                         for entity_id in pdb_id_matching_uniprot_id(uniprot_id=uniprot_id)]
    # # from internal data storage
    # if uniprot_id not in uniprot_pdb_d:
    #     return {'name': None, 'seq': None}
    #     # return AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(pdb_code), chain=chain,
    #     #                                                  source='seqres')
    #
    # # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    # # pdb_chain_d = {}
    # partner_sequences = []  # v in this dictionary 'all' gives PDB.Chain, 'unique' gives only PDB handle
    # for matching_pdb_chain in uniprot_pdb_d[uniprot_id]['all']:
    #     matching_pdb, chain = matching_pdb_chain.split('.')
    #     # pdb_chain_d[matching_pdb] = chain  # This is essentially a set as duplicates are overwritten
    # # # for matching_pdb, chain in pdb_chain_d.items():
    # #     partner_pdb = PDB.from_file(Pose.fetch_pdb_file(matching_pdb), log=None, lazy=True, entities=False)
    # #     # partner_d = AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(matching_pdb),
    # #     #                                                       chain=pdb_chain_d[matching_pdb], source='seqres')
    # #     partner_sequences.append(partner_pdb.reference_sequence[chain])
    #     # api_info = get_pdb_info_by_entry(matching_pdb)
    #     # chain_entity = {chain: entity_idx for entity_idx, chains in api_info.get('entity').items() for ch in chains}
    #     partner_sequences.append(get_entity_reference_sequence(pdb=matching_pdb, chain=chain))

    # matching_pdb_tags = {idx: find_expression_tags(seq) for idx, seq in enumerate(partner_sequences)}
    # [[{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], ...]
    # matching_pdb_tags = list(iter_chain.from_iterable(find_expression_tags(sequence) for sequence in partner_sequences))
    # reduce the iter of iterables for missing values. ^ can return empty lists
    matching_pdb_tags = []
    for sequence in partner_sequences:
        matching_pdb_tags.extend(find_expression_tags(sequence))

    # Next, align all the tags to the reference sequence and tally the tag location and type
    pdb_tag_tally = {'n': {}, 'c': {}, 'matching_tags': matching_pdb_tags}
    for partner_tag in matching_pdb_tags:
        # if partner_pdb_tags:
        #     for partner_tag in partner_pdb_tags:
        if partner_tag['name'] in pdb_tag_tally[partner_tag['termini']]:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] += 1
        else:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] = 1

    return pdb_tag_tally


def select_tags_for_sequence(sequence_id, pdb_tag_tally, preferred=None, n=True, c=True):
    """From a list of possible tags, solve for the tag with the most observations in the PDB. If there are
    discrepancies, query the user for a solution

    Args:
        sequence_id (str): The sequence identifier
        pdb_tag_tally (dict): {'n': {His Tag: 2}, 'c': {Spy Catcher: 1},
                               'matching_tags': [[{'name': His Tag, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'},
                                                  ...], ...]}
    Keyword Args:
        preferred=None (str): The name of a preferred tag provided by the user
        n=True (bool): Whether the n-termini can be tagged
        c=True (bool): Whether the c-termini can be tagged
    Returns:
        (dict): {'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}
    """
    final_tag_sequence = {'name': None, 'termini': None, 'sequence': None}
    # n_term, c_term = 0, 0
    # if pdb_tag_tally['n']:
    n_term = sum([pdb_tag_tally['n'][tag_name] for tag_name in pdb_tag_tally.get('n', {})])
    # if pdb_tag_tally['c']:
    c_term = sum([pdb_tag_tally['c'][tag_name] for tag_name in pdb_tag_tally.get('c', {})])
    if n_term == 0 and c_term == 0:  # No tags found
        return final_tag_sequence
    if n_term > c_term and n or (n_term < c_term and n and not c):
        termini = 'n'
    elif n_term < c_term and c or (n_term > c_term and c and not n):
        termini = 'c'
    elif not c and not n:
        while True:
            termini = \
                input('For sequence target %s, NEITHER termini are available for tagging.\n\n'
                      'You can set up tags anyway and modify this sequence later, or skip tagging.\nThe tag options, '
                      'formatted as, termini: {tag name: count}}, are as follows:\n%s\nWhich termini would you prefer '
                      '[n/c]? To skip, input \'skip\'%s' %
                      (sequence_id, '\n'.join('\t%s: %s' % item for item in pdb_tag_tally.items()
                                              if item[0] != 'matching_tags'),
                       input_string)).lower()
            if termini in ['n', 'c']:
                break
            elif termini == 'skip':
                return final_tag_sequence
            else:
                print('\'%s\' is an invalid input, one of \'n\', \'c\', or \'skip\' is required.')
    else:  # termini = 'Both'
        if c and not n:
            termini = 'c'
        elif not c and n:
            termini = 'n'
        else:
            while True:
                termini = \
                    input('For sequence target %s, BOTH termini are available and have the same number of matched tags.'
                          '\nThe tag options formatted as, termini: {tag name: count}}, are as follows:\n%s'
                          '\nWhich termini would you prefer [n/c]?%s' %
                          (sequence_id, '\n'.join('\t%s: %s' % item for item in pdb_tag_tally.items()),
                           input_string)).lower()
                if termini in ['n', 'c']:
                    break
                else:
                    print('\'%s\' is an invalid input, one of \'n\' or \'c\' is required.')

    # Find the most common tag at the specific termini
    all_tags = []
    max_type, max_count = None, 0
    for tag_name in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][tag_name] > max_count:
            max_count = pdb_tag_tally[termini][tag_name]
            max_type = tag_name
    if max_type:
        all_tags.append(max_type)

    # Check if there are equally represented tags
    for tag_name in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][tag_name] == max_count and tag_name != max_type:
            all_tags.append(tag_name)

    final_tags = {'termini': termini, 'name': all_tags}
    # Finally report results to the user and solve ambiguous tags
    custom = False
    final_choice = {}
    if not final_tags['name']:  # ensure list has at least one element
        return final_tag_sequence
    while True:
        if preferred and preferred == final_tags['name'][0]:
            default = 'y'
        else:
            default = \
                input('For %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf the Termini or Type is '
                      'undesired, you can see the underlying options by specifying \'options\'. Otherwise, \'%s\' will '
                      'be chosen.\nIf you would like to proceed with the RECOMMENDED options, enter \'y\'.%s'
                      % (sequence_id, final_tags['termini'], final_tags['name'][0], final_tags['name'][0],
                         input_string)).lower()
        if default == 'y':
            # if len(final_tags['name']) > 1:
            #     if 'His Tag' in final_tags:
            #         final_choice['name'] = 'His Tag'
            #     # else choose the first choice
            final_choice['name'] = final_tags['name'][0]
            final_choice['termini'] = final_tags['termini']
            break
        elif default == 'options':
            print('\nFor %s, all tag options are:\n\tTermini Tag:\tCount\n%s\nAll tags:\n%s\n'
                  % (sequence_id, '\n'.join('\t%s:\t%s' % item for item in pdb_tag_tally.items()
                                            if item[0] != 'matching_tags'), pdb_tag_tally['matching_tags']))
            # Todo pretty_table_format on the .values() from each item in above list() ('name', 'termini', 'sequence')
            while True:
                termini_input = input('What termini would you like to use [n/c]? If no tag option is appealing, '
                                      'enter \'none\' or specify the termini and select \'custom\' at the next step %s'
                                      % input_string).lower()
                if termini in ['n', 'c']:
                    final_choice['termini'] = termini_input
                    break
                elif termini == 'none':
                    return final_tag_sequence
                else:
                    print('Input doesn\'t match available options. Please try again')
            while True:
                tag_input = int(input('What tag would you like to use? Enter the number of the below options.\n\t%s%s'
                                      '\n%s'
                                      % ('\n\t'.join(['%d - %s' % (i, tag)
                                                     for i, tag in enumerate(pdb_tag_tally[termini_input], 1)]),
                                         '\n\t%d - %s' % (len(pdb_tag_tally[termini_input]) + 1, 'CUSTOM'),
                                         input_string)))
                if tag_input <= len(pdb_tag_tally[termini_input]):
                    final_choice['name'] = list(pdb_tag_tally[termini_input].keys())[tag_input - 1]
                    break
                elif tag_input == len(pdb_tag_tally[termini_input]) + 1:
                    custom = True
                    while True:
                        tag_input = int(input('What tag would you like to use? Enter the number of the below options.'
                                              '\n\t%s\n%s'
                                              % ('\n\t'.join(['%d - %s' % (i, tag)
                                                              for i, tag in enumerate(expression_tags, 1)]),
                                                 input_string)))
                        if tag_input <= len(expression_tags):
                            final_choice['name'] = list(expression_tags.keys())[tag_input - 1]
                            break
                        else:
                            print('Input doesn\'t match available options. Please try again')
                else:
                    print('Input doesn\'t match available options. Please try again')
            break
        else:
            print('Input doesn\'t match. Please try again')

    final_tag_sequence['name'] = final_choice['name']
    final_tag_sequence['termini'] = final_choice['termini']
    all_matching_tags = []
    # [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    for tag in pdb_tag_tally['matching_tags']:
        # for tag in pdb_match:
        if final_choice['name'] == tag['name'] and final_choice['termini'] == tag['termini']:
            all_matching_tags.append(tag['sequence'])

    # TODO align multiple and choose the consensus
    # all_alignments = []
    # max_tag_idx, max_len = None, []  # 0
    # for idx, (tag1, tag2) in enumerate(combinations(all_matching_tags, 2)):
    #     alignment = SequenceProfile.generate_alignment(tag1, tag2)
    #     all_alignments.append(alignment)
    #     # if max_len < alignment[0][4]:  # the length of alignment
    #     max_len.append(alignment[0][4])
    #     # have to find the alignment with the max length, then find which one of the sequences has the max length for
    #     # multiple alignments, then need to select all alignments to this sequence to generate the MSA
    #
    # total_alignment = SequenceProfile.create_bio_msa({idx: tag for idx, tag in enumerate(all_matching_tags)})
    # tag_msa = SequenceProfile.process_alignment(total_alignment)
    if custom:
        final_tag_sequence['sequence'] = expression_tags[final_choice['name']]
    else:
        final_tag_sequence['sequence'] = all_matching_tags[0]  # for now grab the first

    return final_tag_sequence


def add_expression_tag(tag, sequence):
    """Take a raw sequence and add expression tag by aligning a specified tag by PDB reference

    Args:
        tag (dict):
        sequence (str):
    Returns:
        tagged_sequence (str): The final sequence with the tag added
    """
    if not tag:
        return sequence
    alignment = SequenceProfile.generate_alignment(tag, sequence)
    tag_seq = alignment[0][0]
    seq = alignment[0][1]
    # print(alignment[0])
    # print(tag_seq)
    # print(seq)
    # starting_index_of_seq2 = seq.find(sequence[0])
    # i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    final_seq = ''
    for i, (seq1_aa, seq2_aa) in enumerate(zip(tag_seq, seq)):
        if seq2_aa == '-':
            if seq1_aa in IUPACData.protein_letters:
                final_seq += seq1_aa
        else:
            final_seq += seq2_aa

    return final_seq


def pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=None):
    # uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
    source = 'unique_pdb'
    pdb_code = pdb_code.upper()
    if chain:
        # pdb_code = '%s.%s' % (pdb_code, chain)
        # source = 'all'
        dummy = 'TODO ensure that this works once the database is integrated'  # TODO

    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id
    return None


def find_expression_tags(sequence, alignment_length=12):
    """Find all expression_tags on an input sequence from a reference set of expression_tags. Returns the matching tag
    sequence with additional protein sequence context equal to the passed alignment_length

    Args:
        sequence (str): 'MSGHHHHHHGKLKPNDLRI...'
    Keyword Args:
        # tag_file=PathUtils.affinity_tags (list): List of tuples where tuple[0] is the name and tuple[1] is the string
        alignment_length=12 (int): length to perform the clipping of the native sequence in addition to found tag
    Returns:
        (list[dict]): [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], [] if none are found
    """
    matching_tags = []
    for tag, tag_sequence in expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # no match was found
            continue
        # save the tag name, the termini of the sequence it is closest to, and the source sequence context
        tag_length = len(tag_sequence)
        alignment_index = tag_length + alignment_length
        if tag_index == 0 or tag_index < len(sequence)/2:
            termini = 'n'
            matching_sequence = sequence[tag_index:tag_index + alignment_index]
        else:
            termini = 'c'
            matching_sequence = sequence[tag_index - alignment_index:tag_index + tag_length]
        matching_tags.append({'name': tag, 'termini': termini, 'sequence': matching_sequence})

    return matching_tags


def remove_expression_tags(sequence, tags):
    """Remove all specified tags from an input sequence

    Args:
        sequence (str): 'MSGHHHHHHGKLKPNDLRI...'
        tags (list): A list with the sequences of found tags
    Returns:
        (str): The modified sequence without the tag
    """
    for tag in tags:
        tag_index = sequence.find(tag)
        if tag_index == -1:  # no match was found
            continue
        sequence = sequence[:tag_index] + sequence[:tag_index + len(tag)]

    return sequence


def optimize_protein_sequence(sequence, species='e_coli'):
    """Optimize a sequence for expression in a desired organism

    Args:
        sequence (str): The sequence of interest
    Keyword Args:
        species='e_coli' (str): The species context to optimize nucleotide sequence usage
    Returns:
        (str): The input sequence optimized to nucleotides for expression considerations
    """
    problem = DnaOptimizationProblem(sequence=reverse_translate(sequence), objectives=[CodonOptimize(species=species)])
    # constraints=[
    #     AvoidPattern("BsaI_site"),
    #     EnforceGCContent(mini=0.3, maxi=0.7, window=50),
    #     EnforceTranslation(location=(500, 1400))
    # ],

    # SOLVE THE CONSTRAINTS, OPTIMIZE WITH RESPECT TO THE OBJECTIVE
    problem.resolve_constraints()
    problem.optimize()

    # PRINT SUMMARIES TO CHECK THAT CONSTRAINTS PASS
    print(problem.constraints_text_summary())
    print(problem.objectives_text_summary())

    # GET THE FINAL SEQUENCE (AS STRING OR ANNOTATED BIOPYTHON RECORDS)
    final_sequence = problem.sequence  # string
    # final_record = problem.to_record(with_sequence_edits=True)
    return final_sequence
