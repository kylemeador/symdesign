"""Add expression tags onto the termini of specific designs"""
from __future__ import annotations

import logging
# from itertools import chain as iter_chain  # combinations,

from . import generate_alignment, protein_letters_alph1
from .constants import h2o_mass, aa_polymer_molecular_weights, instability_order, instability_array
from symdesign.resources import config, query
from symdesign import utils
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
uniprot_pdb_d = utils.unpickle(putils.uniprot_pdb_map)


def calculate_protein_molecular_weight(sequence: str) -> float:
    sequence = sequence.upper()


    # Find the molecular mass for each aa in the sequence
    seq_index = [instability_order.index(aa) for aa in sequence]
    # Add h2o_mass as the n- and c-term are free
    return aa_polymer_molecular_weights[seq_index].sum() + h2o_mass


def calculate_instability_index(sequence):
    sequence = sequence.upper()
    # dipeptide_stability_sum = 0.
    # for idx, aa in enumerate(sequence[:-2], 1):  # only want to iterate until the second to last amino acid
    #     # get the current amino acid and the next amino acid
    #     dipeptide_stability_sum += instability_array[instability_order.index(aa)][instability_order.index(sequence[idx])]
    #
    # return dipeptide_stability_sum

    # index1, index2 = [], []
    # for idx, aa in enumerate(sequence[:-2], 1):  # only want to iterate until the second to last amino acid
    #     index1.append(instability_order.index(aa))
    #     index2.append(instability_order.index(sequence[idx]))
    # return instability_array[index1, index2].sum()

    # index_pairs = list(zip(*((instability_order.index(aa), instability_order.index(sequence[idx])) for idx, aa in
    #                          enumerate(sequence[:-1], 1))))
    # return instability_array[index_pairs].sum()

    # use this to input all sequences to SequenceProfile. This will form the basis for all sequence handling by array
    seq_index = [instability_order.index(aa) for aa in sequence]
    return instability_array[seq_index[:-1], seq_index[1:]].sum()


# E(Prot) = Numb(Tyr) * Ext(Tyr) + Numb(Trp) * Ext(Trp) + Numb(Cystine) * Ext(Cystine)
# where(for proteins in water measured at 280 nm): Ext(Tyr) = 1490, Ext(Trp) = 5500, Ext(Cystine) = 125
def pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=None):
    # uniprot_pdb_d = SDUtils.unpickle(putils.uniprot_pdb_map)
    source = 'unique_pdb'
    pdb_code = pdb_code.upper()
    if chain:
        # pdb_code = '%s.%s' % (pdb_code, chain)
        # source = 'all'
        # Todo ensure that this works once the database is integrated
        dummy = ''

    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id
    return None


def find_matching_expression_tags(uniprot_id=None, pdb_code=None, chain=None):
    """Take a pose and find expression tags from each PDB reference asking user for input on tag choice

    Args:
        uniprot_id=None (str): The uniprot_id to query tags from
        pdb_code=None (str): The pdb to query tags from. Requires chain argument as well
        chain=None (str): The chain to query tags from. Requires pdb argument as well
    Returns:
        (list[dict]): [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    """
    #         (dict): {'n': {His Tag: 2}, 'c': {Spy Catcher: 1},
    #                  'matching_tags': [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]}
    matching_pdb_tags = []
    if not uniprot_id:
        if not pdb_code or not chain:
            # raise AttributeError('One of uniprot_id or pdb_code AND chain is required')
            logger.error('One of uniprot_id OR pdb_code AND chain is required')
            return matching_pdb_tags
        uniprot_id = pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=chain)

    # from PDB API
    partner_sequences = [query.pdb.get_entity_reference_sequence(entity_id=entity_id)
                         for entity_id in query.pdb.pdb_id_matching_uniprot_id(uniprot_id=uniprot_id)]
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
    # #     partner_pdb = Model.from_file(Pose.fetch_pdb_file(matching_pdb), log=None, entities=False)
    # #     # partner_d = AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(matching_pdb),
    # #     #                                                       chain=pdb_chain_d[matching_pdb], source='seqres')
    # #     partner_sequences.append(partner_pdb.reference_sequence[chain])
    #     # api_info = _get_entry_info(matching_pdb)
    #     # chain_entity = {chain: entity_idx for entity_idx, chains in api_info.get('entity').items() for ch in chains}
    #     partner_sequences.append(query.pdb.get_entity_reference_sequence(entry=matching_pdb, chain=chain))

    # matching_pdb_tags = {idx: find_expression_tags(seq) for idx, seq in enumerate(partner_sequences)}
    # [[{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], ...]
    # matching_pdb_tags = list(iter_chain.from_iterable(find_expression_tags(sequence) for sequence in partner_sequences))
    # reduce the iter of iterables for missing values. ^ can return empty lists
    for sequence in partner_sequences:
        matching_pdb_tags.extend(find_expression_tags(sequence))

    return matching_pdb_tags
    # # Next, align all the tags to the reference sequence and tally the tag location and type
    # pdb_tag_tally = {'n': {}, 'c': {}, 'matching_tags': matching_pdb_tags}
    # for partner_tag in matching_pdb_tags:
    #     # if partner_pdb_tags:
    #     #     for partner_tag in partner_pdb_tags:
    #     if partner_tag['name'] in pdb_tag_tally[partner_tag['termini']]:
    #         pdb_tag_tally[partner_tag['termini']][partner_tag['name']] += 1
    #     else:
    #         pdb_tag_tally[partner_tag['termini']][partner_tag['name']] = 1
    #
    # return pdb_tag_tally


def select_tags_for_sequence(sequence_id, matching_pdb_tags, preferred=None, n=True, c=True):
    """From a list of possible tags, solve for the tag with the most observations in the PDB. If there are
    discrepancies, query the user for a solution

    Args:
        sequence_id (str): The sequence identifier
        matching_pdb_tags (list[dict]): [{'name': His Tag, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    Keyword Args:
        preferred=None (str): The name of a preferred tag provided by the user
        n=True (bool): Whether the n-termini can be tagged
        c=True (bool): Whether the c-termini can be tagged
    Returns:
        (dict): {'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}
    """
    # Next, align all the tags to the reference sequence and tally the tag location and type
    # {'n': {His Tag: 2}, 'c': {Spy Catcher: 1}}
    pdb_tag_tally = {'n': {}, 'c': {}}
    for partner_tag in matching_pdb_tags:
        # if partner_pdb_tags:
        #     for partner_tag in partner_pdb_tags:
        if partner_tag['name'] in pdb_tag_tally[partner_tag['termini']]:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] += 1
        else:
            pdb_tag_tally[partner_tag['termini']][partner_tag['name']] = 1

    final_tag_sequence = {'name': None, 'termini': None, 'sequence': None}
    # n_term, c_term = 0, 0
    n_term = sum([pdb_tag_tally['n'][tag_name] for tag_name in pdb_tag_tally.get('n', {})])
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
                      'are as follows:\n  Termini: {tag name: count}}\n\t%s\nWhich termini would you prefer '
                      '[n/c]? To skip, input "skip"%s' %
                      (sequence_id, '\n\t'.join('%s: %s' % it for it in pdb_tag_tally.items()), query.utils.input_string)).lower()
            if termini in ['n', 'c']:
                break
            elif termini == 'skip':
                return final_tag_sequence
            else:
                print('"%s" is an invalid input, one of "n", "c", or "skip" is required.' % termini)
    else:  # termini = 'Both'
        if c and not n:
            termini = 'c'
        elif not c and n:
            termini = 'n'
        else:
            while True:
                termini = \
                    input('For sequence target %s, BOTH termini are available and have the same number of matched tags.'
                          '\nThe tag options, are as follows:\n  Termini: {tag name: count}}\n\t%s\nWhich termini would'
                          ' you prefer [n/c]? To skip, input "skip"%s' %
                          (sequence_id, '\n\t'.join('%s: %s' % it for it in pdb_tag_tally.items()),
                           query.utils.input_string)).lower()
                if termini in ['n', 'c']:
                    break
                elif termini == 'skip':
                    return final_tag_sequence
                else:
                    print('"%s" is an invalid input, one of "n", "c", or "skip" is required.' % termini)

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

    # Finally report results to the user and solve ambiguous tags
    final_tags = {'termini': termini, 'name': all_tags}
    if not final_tags['name']:  # ensure list has at least one element
        return final_tag_sequence
    custom = False
    final_choice = {}
    while True:
        if preferred and preferred == final_tags['name'][0]:
            default = 'y'
        else:
            default = \
                input('For %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf the Termini or Type is '
                      'undesired, you can see the underlying options by specifying "options". Otherwise, "%s" will '
                      'be chosen.\nIf you would like to proceed with the RECOMMENDED options, enter "y".%s'
                      % (sequence_id, final_tags['termini'], final_tags['name'][0], final_tags['name'][0],
                         query.utils.input_string)).lower()
        if default == 'y':
            final_choice['name'] = final_tags['name'][0]
            final_choice['termini'] = final_tags['termini']
            break
        elif default == 'options':
            print('\nFor %s, all tag options are:\n\tTermini Tag:\tCount\n%s\nAll tags:\n%s\n'
                  % (sequence_id, '\n'.join('\t%s:\t%s' % item for item in pdb_tag_tally.items()), matching_pdb_tags))
            # Todo pretty_table_format on the .values() from each item in above list() ('name', 'termini', 'sequence')
            while True:
                termini_input = input('What termini would you like to use [n/c]? If no tag option is appealing, '
                                      'enter "none" or specify the termini and select "custom" at the next step %s'
                                      % query.utils.input_string).lower()
                if termini in ['n', 'c']:
                    final_choice['termini'] = termini_input
                    break
                elif termini == 'none':
                    return final_tag_sequence
                else:
                    print(f"Input '{termini_input}' doesn't match available options. Please try again")
            while True:
                tag_input = input('What tag would you like to use? Enter the number of the below options.\n\t%s%s\n%s'
                                  % ('\n\t'.join(['%d - %s' % (i, tag)
                                                  for i, tag in enumerate(pdb_tag_tally[termini_input], 1)]),
                                     '\n\t%d - %s' % (len(pdb_tag_tally[termini_input]) + 1, 'CUSTOM'), query.utils.input_string))
                if tag_input.isdigit():
                    tag_input = int(tag_input)
                    if tag_input <= len(pdb_tag_tally[termini_input]):
                        final_choice['name'] = list(pdb_tag_tally[termini_input].keys())[tag_input - 1]
                        break
                    elif tag_input == len(pdb_tag_tally[termini_input]) + 1:
                        custom = True
                        while True:
                            tag_input = input('What tag would you like to use? Enter the number of the below options.'
                                              '\n\t%s\n%s'
                                              % ('\n\t'.join(['%d - %s' % (i, tag)
                                                              for i, tag in enumerate(config.expression_tags, 1)]),
                                                 query.utils.input_string))
                            if tag_input.isdigit():
                                tag_input = int(tag_input)
                            if tag_input <= len(config.expression_tags):
                                final_choice['name'] = list(config.expression_tags.keys())[tag_input - 1]
                                break
                            print(f"Input '{tag_input}' doesn't match available options. Please try again")
                        break
                print(f"Input '{tag_input}' doesn't match available options. Please try again")
            break
        print(f"Input '{default}' doesn't match. Please try again")

    final_tag_sequence['name'] = final_choice['name']
    final_tag_sequence['termini'] = final_choice['termini']
    all_matching_tags = []
    # [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
    for tag in matching_pdb_tags:
        # for tag in pdb_match:
        if final_choice['name'] == tag['name'] and final_choice['termini'] == tag['termini']:
            all_matching_tags.append(tag['sequence'])

    # TODO align multiple and choose the consensus
    # all_alignments = []
    # max_tag_idx, max_len = None, []  # 0
    # for idx, (tag1, tag2) in enumerate(combinations(all_matching_tags, 2)):
    #     alignment = SequenceProfile.generate_alignment(tag1, tag2)
    #     all_alignments.append(alignment)
    #     # if max_len < alignment[4]:  # the length of alignment
    #     max_len.append(alignment[4])
    #     # have to find the alignment with the max length, then find which one of the sequences has the max length for
    #     # multiple alignments, then need to select all alignments to this sequence to generate the MSA
    #
    # total_alignment = SequenceProfile.create_bio_msa({idx: tag for idx, tag in enumerate(all_matching_tags)})
    # tag_msa = SequenceProfile.generate_msa_dictionary(total_alignment)
    if custom:
        final_tag_sequence['sequence'] = config.expression_tags[final_choice['name']]
    else:
        final_tag_sequence['sequence'] = all_matching_tags[0]  # for now grab the first

    return final_tag_sequence


def add_expression_tag(tag: str, sequence: str) -> str:
    """Add an expression tag to a sequence by aligning a tag specified with PDB reference sequence to the sequence

    Args:
        tag: The tag with additional PDB reference sequence appended
        sequence: The sequence of interest
    Returns:
        The final sequence with the tag added
    """
    if not tag:
        return sequence
    alignment = generate_alignment(tag, sequence)
    # print('Expression TAG alignment:', alignment[0])
    tag_seq, seq = alignment.sequences
    # score = alignment.score
    # # tag_seq, seq, score, *_ = alignment
    # # score = alignment[2]  # first alignment, grab score value
    # # print('Expression TAG alignment score:', score)
    # # if score == 0:  # TODO find the correct score for a bad alignment to indicate there was no productive alignment?
    # #     # alignment[0][4]:  # the length of alignment
    # #     # match_score = score / len(sequence)  # could also use which ever sequence is greater
    # #     return sequence
    # # print(alignment[0])
    # # print(tag_seq)
    # # print(seq)
    # # starting_index_of_seq2 = seq.find(sequence[0])
    # # i = -starting_index_of_seq2 + zero_offset  # make 1 index so residue value starts at 1
    final_seq = ''
    for i, (seq1_aa, seq2_aa) in enumerate(zip(tag_seq, seq)):
        if seq2_aa == '-':
            if seq1_aa in protein_letters_alph1:
                final_seq += seq1_aa
        else:
            final_seq += seq2_aa

    return final_seq


def find_expression_tags(sequence: str, alignment_length: int = 12) -> list | list[dict[str, str]]:
    """Find all expression_tags on an input sequence from a reference set of expression_tags. Returns the matching tag
    sequence with additional protein sequence context equal to the passed alignment_length

    Args:
        sequence: 'MSGHHHHHHGKLKPNDLRI...'
        alignment_length: length to perform the clipping of the native sequence in addition to found tag
    Keyword Args:
        # tag_file=PathUtils.affinity_tags (list): List of tuples where tuple[0] is the name and tuple[1] is the string
    Returns:
        (list[dict]): [{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], [] if none are found
    """
    matching_tags = []
    for tag, tag_sequence in config.expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # no match was found
            continue
        # save the tag name, the termini of the sequence it is closest to, and the source sequence context
        tag_length = len(tag_sequence)
        alignment_size = tag_length + alignment_length
        if tag_index == 0 or tag_index < len(sequence) / 2:
            termini = 'n'
            matching_sequence = sequence[tag_index:tag_index + alignment_size]
        else:
            termini = 'c'
            matching_sequence = sequence[tag_index - alignment_size:tag_index + tag_length]
        matching_tags.append({'name': tag, 'termini': termini, 'sequence': matching_sequence})

    return matching_tags


def remove_expression_tags(sequence, tags):
    """Remove all specified tags from an input sequence

    Args:
        sequence (str): 'MSGHHHHHHGKLKPNDLRI...'
        tags (list): A list with the sequences of found tags
    Returns:
        (str): 'MSGGKLKPNDLRI...' The modified sequence without the tag
    """
    for tag in tags:
        tag_index = sequence.find(tag)
        if tag_index == -1:  # no match was found
            continue
        sequence = sequence[:tag_index] + sequence[tag_index + len(tag):]

    return sequence
