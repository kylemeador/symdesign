"""Add expression tags onto the termini of specific designs"""
from __future__ import annotations

import logging
from itertools import count
from typing import Sequence

import numpy as np

from . import generate_alignment, protein_letters_alph1
from .constants import expression_tags, h2o_mass, aa_polymer_molecular_weights, instability_array
from symdesign.resources import query
from symdesign import utils

putils = utils.path

# Globals
tags = expression_tags
logger = logging.getLogger(__name__)
uniprot_pdb_d = utils.unpickle(putils.uniprot_pdb_map)


def format_sequence_to_numeric(sequence: Sequence[str | int]) -> Sequence[int]:
    try:
        seq0 = sequence[0]
    except IndexError:
        raise IndexError('The passed value of sequence must be a Sequence of str or int. Found an empty Sequence')
    if isinstance(seq0, str):
        seq_index = [protein_letters_alph1.index(aa) for aa in sequence]
    elif isinstance(seq0, int):
        seq_index = sequence
    else:
        raise ValueError(
            f"Couldn't {format_sequence_to_numeric.__name__} with the input sequence with type {seq0}")
    return seq_index


def get_sequence_features(sequence: Sequence[str | int]) -> dict[str, float]:
    """For the specified amino acid sequence perform biochemical calculations related to solution properties

    Args:
        sequence: The sequence to measure
    Returns:
        A feature dictionary with each feature from the set below mapped to a float.
        {'extinction_coefficient_reduced',
         'extinction_coefficient_oxidized',
         'instability_index',
         'molecular_weight',
         'isoelectric_point',
         'absorbance_0.1_red',
         'absorbance_0.1_ox',
         }
    """
    sequence = format_sequence_to_numeric(sequence)
    ox_coef, red_coef = molecular_extinction_coefficient(sequence)
    mw = calculate_protein_molecular_weight(sequence)
    abs_01_ox = ox_coef / mw
    abs_01_red = red_coef / mw
    return {
        'number_of_residues': len(sequence),
        'extinction_coefficient_reduced': red_coef,
        'extinction_coefficient_oxidized': ox_coef,
        'instability_index': calculate_instability_index(sequence),
        'molecular_weight': mw,
        'isoelectric_point': calculate_protein_isoelectric_point(sequence),
        'absorbance_0.1_red': abs_01_red,
        'absorbance_0.1_ox': abs_01_ox
    }


def calculate_protein_molecular_weight(sequence: Sequence[str | int]) -> float:
    """Find the total molecular mass for the amino acids present in a sequence

    Args:
        sequence: The sequence to measure
    Returns:
        The molecular weight in daltons/atomic mass units
    """
    seq_index = format_sequence_to_numeric(sequence)

    # Add h2o_mass as the n- and c-term are free
    return aa_polymer_molecular_weights[seq_index].sum() + h2o_mass


# 1-amino acid alphabetical order
cys_index = 1
asp_index = 2
glu_index = 3
his_index = 6
lys_index = 8
arg_index = 14
trp_index = 18
tyr_index = 19
# These amino acid side chain values are from protein polymers derived from the fit by the publication PMID:27769290
# [9.094,  # N-term
#  7.555,  # C
#  3.872,  # D
#  4.412,  # E
#  5.637,  # H
#  9.052,  # K
#  11.84,  # R
#  10.85,  # Y
#  2.869,  # C-term
# ]
positive_charged_pka = np.array([9.094, 5.637, 9.052, 11.84])
negative_charged_pka = np.array([7.555, 3.872, 4.412, 10.85, 2.869])


def calculate_protein_isoelectric_point(sequence: Sequence[str | int], threshold: float = 0.01) -> float:
    """Find the total isoelectric point (pI) for the amino acids present in a sequence

    Args:
        sequence: The sequence to measure
        threshold: The pH unit value to consider the found pI passing
    Returns:
        The molecular weight in daltons/atomic mass units
    """
    seq_index = format_sequence_to_numeric(sequence)
    # # Take the mean of every individual pI
    # return aa_isoelectric_point[seq_index].mean()

    n_cys = seq_index.count(cys_index)
    n_asp = seq_index.count(asp_index)
    n_glu = seq_index.count(glu_index)
    n_his = seq_index.count(his_index)
    n_lys = seq_index.count(lys_index)
    n_arg = seq_index.count(arg_index)
    n_tyr = seq_index.count(tyr_index)

    # def calculate_aa_charge_contribution(aa: str, pka_sol: float = 7.5):
    #     negative_charged = {'C': 7.555, 'D': 3.872, 'E': 4.412, 'Y': 10.85}
    #     positive_charged = {'H': 5.637, 'K': 9.052, 'R': 11.84}
    #     pka_aa = positive_charged.get(aa)
    #     if pka_aa:
    #         pka_aa = 1 / (1 + 10**(pka_sol-pka_aa))  # positive
    #     else:
    #         pka_aa = -1 / (1 + 10**(negative_charged[aa]-pka_sol))  # negative

    # Array version
    # Make an array of the counts padding respective ends with n- and c-termini counts (i.e. 1)
    positive_counts = np.array([1, n_his, n_lys, n_arg])
    negative_counts = np.array([n_cys, n_asp, n_glu, n_tyr, 1])

    iterative_multiplier = 1
    delta_magnitude = 4
    test_pi = 7.  # - delta_magnitude  # <- ensures that the while loop search starts at 7.
    # pka_aa_pos = 1 / (1 + 10 ** (test_pi - positive_charged_pka))  # positive
    # pka_aa_neg = -1 / (1 + 10 ** (negative_charged_pka - test_pi))  # negative
    # pos_charge = pka_aa_pos * positive_counts
    # neg_charge = pka_aa_neg * negative_counts
    # total_charge = neg_charge + pos_charge
    # remaining_charge = abs(total_charge)
    # negative_last = abs(neg_charge) > abs(pos_charge)
    # neg_charge = pos_charge = negative_last = 0
    if threshold < 0:
        raise ValueError(f"The argument 'threshold' can't be lower than 0. Got {threshold}")
    remaining_charge = threshold + 1
    direction = 0
    count_ = count()
    tested_pis = {}
    while remaining_charge >= threshold:
        if next(count_) > 0:
            # Check which type of charge has a larger magnitude
            if abs(neg_charge) > abs(pos_charge):
                # Estimation is more negative
                direction = -1
                if last_direction + direction == 0:
                    # pass  # This was also negative last iteration, haven't gone far enough
                # else:  # Searched this direction far enough, turn around and decrease step
                    iterative_multiplier /= 2
            else:  # Positive larger
                direction = 1
                if last_direction + direction == 0:
                    # Searched this direction far enough, turn around and decrease step
                    iterative_multiplier /= 2
                # else:
                #     pass  # This was positive last iteration, haven't gone far enough
        last_direction = direction

        # Calculate the error
        pi_modifier = direction * delta_magnitude * iterative_multiplier
        test_pi_ = pi_modifier + test_pi
        if test_pi_ in tested_pis:
            # This has already been checked. Cut the space in half again
            iterative_multiplier /= 2
            pi_modifier = direction * delta_magnitude * iterative_multiplier

        test_pi += pi_modifier

        pka_aa_pos = 1 / (1 + 10**(test_pi-positive_charged_pka))  # positive
        pka_aa_neg = -1 / (1 + 10**(negative_charged_pka-test_pi))  # negative
        # Multiply the individual contributions by the number of observations
        pos_charge = (pka_aa_pos * positive_counts).sum()
        neg_charge = (pka_aa_neg * negative_counts).sum()
        total_charge = neg_charge + pos_charge
        remaining_charge = abs(total_charge)
        logger.debug(f'pI = {test_pi}, remainder = {remaining_charge}')  # Iteration {next(count_)}:
        tested_pis[test_pi] = remaining_charge

    return test_pi + (total_charge/2)  # Approximately the real pi


def calculate_instability_index(sequence: Sequence[str | int]) -> float:
    """Find the total instability index for the amino acids present in a sequence. See PMID:2075190

    Args:
        sequence: The sequence to measure
    Returns:
        The value of the stability index where a value less than 40 indicates stability
    """

    seq_index = format_sequence_to_numeric(sequence)
    return (instability_array[seq_index[:-1], seq_index[1:]].sum() * 10) / len(sequence)


# For proteins in water measured at 280 nm absorbance
ext_coef_cys_red = 0
ext_coef_cys_ox = 125
ext_coef_trp = 5500
ext_coef_tyr = 1490


def molecular_extinction_coefficient(sequence: Sequence[str | int]) -> tuple[float, float]:
    """Calculate the molecular extinction coefficient for an amino acid sequence using the formula
    E(Prot) = Numb(Tyr) * Ext(Tyr) + Numb(Trp) * Ext(Trp) + Numb(Cystine) * Ext(Cystine)

    Args:
        sequence: The sequence to measure
    Returns:
        The pair of molecular extinction coefficients, first with all Cystine oxidized, then reduced
    """
    seq_index = format_sequence_to_numeric(sequence)
    n_tyr = seq_index.count(tyr_index)
    n_trp = seq_index.count(trp_index)
    n_cys = seq_index.count(cys_index)
    coef_ox = n_tyr*ext_coef_tyr + n_trp*ext_coef_trp + n_cys*ext_coef_cys_ox
    coef_red = n_tyr*ext_coef_tyr + n_trp*ext_coef_trp  # + n_cys*ext_coef_cys_red
    return coef_ox, coef_red


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


def find_matching_expression_tags(uniprot_id: str = None, pdb_code: str = None, chain: str = None) \
        -> list | list[dict[str, str]]:
    """Find matching expression tags by PDB ID reference

    Args:
        uniprot_id: The uniprot_id to query tags from
        pdb_code: The pdb to query tags from. Requires chain argument as well
        chain: The chain to query tags from. Requires pdb argument as well
    Returns:
        [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], or [] if none found
    """
    #         (dict): {'n': {His Tag: 2}, 'c': {Spy Catcher: 1},
    #                  'matching_tags': [{'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]}
    matching_pdb_tags = []
    if uniprot_id is None:
        if pdb_code is None or chain is None:
            # raise AttributeError('One of uniprot_id or pdb_code AND chain is required')
            logger.error('One of "uniprot_id" OR "pdb_code" AND "chain" is required')
            return matching_pdb_tags
        uniprot_id = pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=chain)

    # From PDB API
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


def select_tags_for_sequence(sequence_id: str, matching_pdb_tags: list[dict[str, str]], preferred: str = None,
                             n: bool = True, c: bool = True) -> dict[str, str | None]:
    """From a list of possible tags, solve for the tag with the most observations in the PDB. If there are
    discrepancies, query the user for a solution

    Args:
        sequence_id: The sequence identifier
        matching_pdb_tags: [{'name': His Tag, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...]
        preferred: The name of a preferred tag provided by the user
        n: Whether the n-termini can be tagged
        c: Whether the c-termini can be tagged
    Returns:
        {'name': 'his_tag', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}
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

    formatted_tags = [(termini, tag, counts) for termini, tags_counts in pdb_tag_tally.items()
                      for tag, counts in tags_counts.items()]
    custom = False
    final_choice = {}
    while True:
        tag_type = final_tags['name'][0]
        recommended_termini = final_tags['termini']
        if preferred:
            if preferred == tag_type:
                default = 'y'
            else:
                return final_tag_sequence
                # default = 'y'
                # logger.info(
                #     f"The preferred tag '{preferred}' wasn't found from observations in the PDB. Using it anyway")
                # # default = input(f'If you would like to proceed with it anyway, enter "y"'
                # #                 f'.{query.utils.input_string}').lower()
                # tag_type = preferred
                # print(f'For {sequence_id}, the tag options are:\n\t%s'
                #       % '\n\t'.join(utils.pretty_format_table([tag.values() for tag in matching_pdb_tags],
                #                                               header=matching_pdb_tags[0].keys())))
                # # Solve for the termini
                # print(f'For {sequence_id}, the termini tag options are:\n\t%s'
                #       % '\n\t'.join(utils.pretty_format_table(formatted_tags, header=('Termini', 'Tag', 'Count'))))
                # while True:
                #     termini_input = input(f'What termini would you like to use [n/c]?{query.utils.input_string}') \
                #         .lower()
                #     if termini_input in ['n', 'c']:
                #         recommended_termini = termini_input
                #         break
                #     elif termini_input == 'none':
                #         return final_tag_sequence
                #     else:
                #         print(f"Input '{termini_input}' doesn't match available options. Please try again")
        else:
            logger.info(f'For {sequence_id}, the RECOMMENDED tag options are:\n'
                        f'\tTermini-{recommended_termini} Type-{tag_type}\n'
                        'If the Termini or Type is undesired, you can see the underlying options by specifying '
                        "'options'. Otherwise, '{tag_type}' will be chosen")
            default = input(f'If you would like to proceed with the \033[38;5;208mrecommended\033[0;0m: options, '
                            f'enter "y".{query.utils.input_string}').lower()
        if default == 'y':
            final_choice['name'] = tag_type
            final_choice['termini'] = recommended_termini
            break
        elif default == 'options':
            print(f'\nFor {sequence_id}, all tag options are:\n\tTermini Tag:\tCount\n%s\nAll tags:\n%s\n'
                  % ('\n'.join('\t%s:\t%s' % item for item in pdb_tag_tally.items()), matching_pdb_tags))
            # Todo pretty_table_format on the .values() from each item in above list() ('name', 'termini', 'sequence')
            while True:
                termini_input = input('What termini would you like to use [n/c]? If no tag option is appealing, '
                                      'enter "none" or specify the termini and select "custom" at the next step '
                                      f'{query.utils.input_string}').lower()
                if termini_input in ['n', 'c']:
                    final_choice['termini'] = termini_input
                    break
                elif termini_input == 'none':
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
                                                              for i, tag in enumerate(expression_tags, 1)]),
                                                 query.utils.input_string))
                            if tag_input.isdigit():
                                tag_input = int(tag_input)
                            if tag_input <= len(expression_tags):
                                final_choice['name'] = list(expression_tags.keys())[tag_input - 1]
                                break
                            print(f"Input '{tag_input}' doesn't match available options. Please try again")
                        break
                print(f"Input '{tag_input}' doesn't match available options. Please try again")
            break
        print(f"Input '{default}' doesn't match. Please try again")

    final_tag_sequence['name'] = final_choice['name']
    final_tag_sequence['termini'] = final_choice['termini']
    all_matching_tags = []
    tag: dict[str, str]
    """{'name': tag_name, 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}"""
    for tag in matching_pdb_tags:
        # for tag in pdb_match:
        if final_choice['name'] == tag['name'] and final_choice['termini'] == tag['termini']:
            all_matching_tags.append(tag['sequence'])

    # Todo align multiple and choose the consensus
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
        final_tag_sequence['sequence'] = expression_tags[final_choice['name']]
    else:
        logger.debug(f'Grabbing the first matching tag out of {len(all_matching_tags)} possible')
        final_tag_sequence['sequence'] = all_matching_tags[0]  # For now grab the first

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
    tag_seq, seq = alignment
    # print('Expression TAG alignment:', alignment[0])
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
    for seq1_aa, seq2_aa in zip(tag_seq, seq):
        if seq2_aa == '-':
            # if seq1_aa in protein_letters_alph1:
            final_seq += seq1_aa
        else:
            final_seq += seq2_aa

    return final_seq


def find_expression_tags(sequence: str, alignment_length: int = 12) -> list | list[dict[str, str]]:
    """Find all expression_tags on an input sequence from a reference set of expression_tags. Returns the matching tag
    sequence with additional protein sequence context equal to the passed alignment_length

    Args:
        sequence: The sequence of interest i.e. 'MSGHHHHHHGKLKPNDLRI...'
        alignment_length: length to perform the clipping of the native sequence in addition to found tag
    Returns:
        [{'name': 'tag_name', 'termini': 'n', 'sequence': 'MSGHHHHHHGKLKPNDLRI'}, ...], [] if none are found
    """
    half_sequence_length = len(sequence) / 2
    matching_tags = []
    for name, tag_sequence in expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # No match was found
            continue
        # Save the tag name, the termini of the sequence it is closest to, and the source sequence context
        tag_length = len(tag_sequence)
        alignment_size = tag_length + alignment_length
        if tag_index < half_sequence_length:
            termini = 'n'
            matching_sequence = sequence[tag_index:tag_index + alignment_size]
        else:
            termini = 'c'
            final_tag_index = tag_index + tag_length
            matching_sequence = sequence[final_tag_index - alignment_size:final_tag_index]
        matching_tags.append({'name': name, 'termini': termini, 'sequence': matching_sequence})

    return matching_tags


# This variant only removes the tag, not the entire termini
def remove_internal_tags(sequence: str, tag_names: list[str]) -> str:
    """Remove matching tag sequences only, from the specified sequence

    Defaults to known tags in constants.expression_tags

    Args:
        sequence: The sequence of interest i.e. 'MSGHHHHHHGKLKPNDLRI...'
        tag_names: If only certain tags should be removed, a list with the names of known tags
    Returns:
        'MSGGKLKPNDLRI...' The modified sequence without the tag
    """
    if tag_names:
        _expression_tags = {tag_name: expression_tags[tag_name] for tag_name in tag_names}
    else:
        _expression_tags = expression_tags

    half_sequence_length = len(sequence) / 2
    for name, tag_sequence in _expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # No match was found
            continue

        # Remove the tag from the source sequence
        tag_length = len(tag_sequence)
        sequence = sequence[:tag_index] + sequence[tag_index + tag_length:]

    return sequence


def remove_terminal_tags(sequence: str, tag_names: list[str] = None) -> str:
    """Remove matching tag sequences and any remaining termini from the specified sequence

    Defaults to known tags in constants.expression_tags

    Args:
        sequence: The sequence of interest i.e. 'MSGHHHHHHGKLKPNDLRI...'
        tag_names: If only certain tags should be removed, a list with the names of known tags
    Returns:
        'GGKLKPNDLRI...' The modified sequence without the tagged termini
    """
    if tag_names:
        _expression_tags = {tag_name: expression_tags[tag_name] for tag_name in tag_names}
    else:
        _expression_tags = expression_tags

    half_sequence_length = len(sequence) / 2
    for name, tag_sequence in _expression_tags.items():
        tag_index = sequence.find(tag_sequence)
        if tag_index == -1:  # No match was found
            continue

        # Remove the tag from the source sequence
        tag_length = len(tag_sequence)

        # Remove from one end based on termini proximity
        if tag_index < half_sequence_length:  # termini = 'n'
            sequence = sequence[tag_index + tag_length:]
        else:  # termini = 'c'
            sequence = sequence[:tag_index]

    return sequence
