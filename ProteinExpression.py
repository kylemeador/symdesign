"""Add expression tags onto the termini of specific designs"""
import csv

from Bio.SeqUtils import IUPACData

import PathUtils as PUtils
import SymDesignUtils as SDUtils
from PDB import PDB
import Pose
import SequenceProfile


# Globals
SDUtils.start_log(name=__name__)
uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
with open(PUtils.affinity_tags, 'r') as f:
    expression_tags = {row[0]: row[1] for row in csv.reader(f)}


def find_all_matching_pdb_expression_tags(pdb_code, chain):  # Todo separate find and user input functionality
    """Take a pose and find expression tags from each PDB reference asking user for input on tag choice

    Args:
        pdb_code (str): The pdb to query tags from
        chain (str): The chain to query tags from
    Returns:
        (dict): {pdb: {'name': 'His Tag', 'seq': 'MSGHHHHHHGKLKPNDLRI'}, ...}
    """
    uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)
    # 'all' gives PDB.Chain, 'unique' gives only PDB handle
    uniprot_id = pull_uniprot_id_by_pdb(uniprot_pdb_d, pdb_code, chain=chain)
    if uniprot_id not in uniprot_pdb_d:
        return {'name': None, 'seq': None}
        # return AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(pdb_code), chain=chain,
        #                                                  source='seqres')
    else:
        all_matching_pdb_chain = uniprot_pdb_d[uniprot_id]['all']

    # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    pdb_chain_d = {}
    for matching_pdb_chain in all_matching_pdb_chain:
        matching_pdb, chain = matching_pdb_chain.split('.')
        pdb_chain_d[matching_pdb] = chain  # This is essentially a set as duplicates are overwritten

    partner_sequences = []
    for matching_pdb in pdb_chain_d:
        partner_pdb = PDB.from_file(Pose.fetch_pdb_file(matching_pdb), log=None, )
        # partner_d = AnalyzeMutatedSequences.get_pdb_sequences(Pose.retrieve_pdb_file_path(matching_pdb),
        #                                                       chain=pdb_chain_d[matching_pdb], source='seqres')
        partner_sequences.append(partner_pdb.reference_sequence[pdb_chain_d[matching_pdb]])
        # TODO chain can be not found... Should this be available based on Uniprot-PDB Map creation? Need to extend this

    # {0: {1: {'name': tag_name, 'termini': 'N', 'seq': 'MSGHHHHHHGKLKPNDLRI'}}, ...}
    matching_pdb_tags = {idx: find_expression_tags(seq) for idx, seq in enumerate(partner_sequences)}
    # can return an empty dict

    # Next, align all the tags to the reference sequence and tally the tag location and type
    pdb_tag_tally = {'N': {}, 'C': {}}
    for partner in matching_pdb_tags:
        if matching_pdb_tags[partner] != dict():
            for partner_tag in matching_pdb_tags[partner]:
                if matching_pdb_tags[partner][partner_tag]['name'] \
                        in pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']]:
                    pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']][
                        matching_pdb_tags[partner][partner_tag]['name']] += 1
                else:
                    pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']][
                        matching_pdb_tags[partner][partner_tag]['name']] = 1

    final_tags = {}
    n_term, c_term = 0, 0
    if pdb_tag_tally['N'] != dict():
        n_term = [pdb_tag_tally['N'][_type] for _type in pdb_tag_tally['N']]
        n_term = sum(n_term)
    if pdb_tag_tally['C'] != dict():
        c_term = [pdb_tag_tally['C'][_type] for _type in pdb_tag_tally['C']]
        c_term = sum(c_term)
    if n_term == 0 and c_term == 0:  # No tags found
        return {'name': None, 'seq': None}
    if n_term > c_term:
        termini = 'N'
    elif n_term < c_term:
        termini = 'C'
    else:  # termini = 'Both'
        while True:
            termini = input('For %s, BOTH termini have the same number of matched tags.\n'
                            'The tag options are as follows {terminus:{tag name: count}}:\n%s\n'
                            'Which termini would you prefer?\n[n/c]:' % (pdb_code, pdb_tag_tally))
            termini = termini.upper()
            if termini == 'N' or termini == 'C':
                break

    # Find the most common tag at the specific termini
    all_tags = []
    max_type = None
    max_count = 0
    for _type in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][_type] > max_count:
            max_count = pdb_tag_tally[termini][_type]
            max_type = _type
    all_tags.append(max_type)

    # Check if there are equally represented tags
    for _type in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][_type] == max_count and _type != max_type:
            all_tags.append(_type)
    final_tags['name'] = all_tags
    final_tags['termini'] = termini

    # Finally report results to the user and solve ambiguous tags
    final_choice = {}
    while True:
        default = input('For %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf the Termini or Type is '
                        'undesired, you can see the underlying options by specifying \'o\'. Otherwise, \'%s\' will be '
                        'chosen.\nIf you would like to proceed with the RECOMMENDED options, enter \'y\'.\nInput [o/y]:'
                        % (pdb_code, final_tags['termini'], final_tags['name'], final_tags['name'][0]))
        if default.lower() == 'y':
            if len(final_tags['name']) > 1:
                if 'His Tag' in final_tags:
                    final_choice['name'] = 'His Tag'
                # else choose the first choice
            final_choice['name'] = final_tags['name'][0]
            final_choice['termini'] = final_tags['termini']
            break
        elif default.lower() == 'o':
            _input = input('For %s, the FULL tag options are: %s\nIf none of these are appealing, enter \'n\', '
                           'otherwise hit enter.' % (pdb_code, pdb_tag_tally))
            if _input.upper() == 'N':
                return {'name': None, 'seq': None}
            else:
                while True:
                    termini_input = input('What termini would you like to use?\nInput [n/c]:')
                    termini_input = termini_input.upper()
                    if termini_input == 'N' or termini_input == 'C':
                        final_choice['termini'] = termini_input
                        break
                    else:
                        print('Input doesn\'t match. Please try again')
                while True:
                    tag_input = input('What tag would you like to use? Enter the number of the below options.\n%s' %
                                      '\n'.join(['%d - %s' % (i, tag)
                                                 for i, tag in enumerate(pdb_tag_tally[termini_input])]))
                    tag_input = int(tag_input)
                    if tag_input < len(pdb_tag_tally[termini_input]):
                        final_choice['name'] = pdb_tag_tally[termini_input][tag_input]
                        break
                    else:
                        print('Input doesn\'t match. Please try again')
            break
        else:
            print('Input doesn\'t match. Please try again')

    final_tag_sequence = {'name': final_choice['name'], 'seq': None}
    for partner_idx in matching_pdb_tags:
        for partner_tag in matching_pdb_tags[partner_idx]:
            if final_choice['name'] == matching_pdb_tags[partner_idx][partner_tag]['name']:
                final_tag_sequence['seq'] = matching_pdb_tags[partner_idx][partner_tag]['seq']
                # TODO align multiple and choose the consensus?

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
        (list[dict]): [{'name': tag_name, 'termini': 'n', 'seq': 'MSGHHHHHHGKLKPNDLRI'}, ...], [] if none are found
    """
    matching_tags = []
    for tag in expression_tags:
        tag_index = sequence.find(expression_tags[tag])
        if tag_index == -1:  # no match was found
            continue
        # save the tag name, the termini of the sequence it is closest to, and the source sequence context
        found_tag = {'name': tag}
        # matching_tags[count]['name'] = tag_name
        alignment_index = len(expression_tags[tag]) + alignment_length
        if tag_index == 0 or tag_index < len(sequence)/2:
            found_tag['termini'] = 'n'
            found_tag['seq'] = sequence[tag_index:tag_index + alignment_index]
        else:
            found_tag['termini'] = 'c'
            found_tag['seq'] = sequence[tag_index - alignment_index:tag_index + len(expression_tags[tag])]
        matching_tags.append(found_tag)

    return matching_tags
