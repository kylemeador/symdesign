"""Add expression tags onto the termini of specific designs"""
import os
import sys
import csv
from numpy import array
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import AnalyzeMutatedSequences as Ams

# Globals
uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)


def find_expression_tags(pdb_code, chain):
    """Take a pose and find expression tags from each PDB reference

    Args:
        pdb_code (str): The pdb to query tags from
        chain (str): The chain to query tags from
        des_dir (DesignDirectory): Object containing the pose information
    Returns:
        (dict): {pdb: {'name': 'His Tag', 'seq': 'MSGHHHHHHGKLKPNDLRI'}, ...}
    """
    # {'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...}
    # pdbs = os.path.basename(des_dir.building_blocks).split('_')
    # pose = SDUtils.read_pdb(des_dir.source)
    # design_seq_d = Ams.get_pdb_sequences(des_dir.source, chain=None)
    # {pdb: [1XYZ.A, 2ABC.B], ...}
    # reference_seq_d =
    # all_matching_pdb_chain = []
    # for pdb_code, chain in zip(pdbs, pose.chain_id_list):
    all_matching_pdb_chain = uniprot_pdb_d[pull_uniprot_id_by_pdb(pdb_code, chain=chain)]['all']
    # reference_seq_d = reference_seq_d[chain]

    # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    partner_sequences = []
    # for pdb in pdbs:
    #     partner_sequences[pdb] = []
    for matching_pdb_chain in all_matching_pdb_chain:
        matching_pdb, chain = matching_pdb_chain.split('.')
        partner_d = Ams.get_pdb_sequences(SDUtils.fetch_pdb(matching_pdb), chain=chain, source='seqres')
        partner_sequences.append(partner_d[chain])
        # TODO chain can be not found!

    # {0: {1: {'name': tag_name, 'termini': 'N', 'seq': 'MSGHHHHHHGKLKPNDLRI'}}, ...}
    matching_pdb_tags = {}
    # for pdb in pdbs:
    #     matching_pdb_tags[pdb] = {}
    for idx, seq in enumerate(partner_sequences):
        # tags[pdb][idx] = find_tags(partner_sequences[pdb][idx])
        matching_pdb_tags[idx] = find_tags(seq)  # can return an empty dict

    # next, align all the tags to the reference sequence and tally the tag location and type
    # pdb_tag_tally = {}
    # for pdb in pdbs:
        # pdb_tag_tally[pdb] = {'N': 0, 'C': 0, 'types': {}}
        # pdb_tag_tally[pdb] = {'N': {}, 'C': {}}
    pdb_tag_tally = {'N': {}, 'C': {}}
    for partner in matching_pdb_tags:
        if matching_pdb_tags[partner] != dict():
            for partner_tag in matching_pdb_tags[partner]:
                # for tag_idx in matching_pdb_tags[partner][partner_tag]:
                    # count the number of termini
                if matching_pdb_tags[partner][partner_tag]['name'] \
                        in pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']]:
                    # pdb_tag_tally[pdb]['N'][tag_name] +=1
                    pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']][
                        matching_pdb_tags[partner][partner_tag]['name']] += 1
                else:
                    pdb_tag_tally[matching_pdb_tags[partner][partner_tag]['termini']][
                        matching_pdb_tags[partner][partner_tag]['name']] = 0
                    # pdb_tag_tally[partner_tag[tag_idx]['termini']][partner_tag[tag_idx]['name']] = 0
                    # # pdb_tag_tally[pdb][tags[tag]['termini']] += 1
                    # if tags[tag]['name'] in pdb_tag_tally[pdb]['types']:
                    #     # pdb_tag_tally[pdb]['types'][tag_name]
                    #     pdb_tag_tally[pdb]['types'][tags[tag]['name']] += 1
                    # else:
                    #     pdb_tag_tally[pdb]['types'][tags[tag]['name']] = 0

    final_tags = {}
    # for pdb in pdbs:
    #     final_tags[pdb] = {}
        # final_tags[pdb] = {'termini': None, 'type': None}
    n_term, c_term = 0, 0
    if pdb_tag_tally['N'] != dict():
        n_term = [pdb_tag_tally['N'][_type] for _type in pdb_tag_tally['N']]
        n_term = array(n_term).sum()
    if pdb_tag_tally['C'] != dict():
        c_term = [pdb_tag_tally['C'][_type] for _type in pdb_tag_tally['C']]
        c_term = array(c_term).sum()
    if n_term == 0 and c_term == 0:
        return {'name': None, 'seq': None}
        # if len(pdb_tag_tally[pdb]['N']) > len(pdb_tag_tally[pdb]['C']):
    if n_term > c_term:
        # final_tags[pdb]['termini'] = 'N'
        termini = 'N'
    # elif len(pdb_tag_tally[pdb]['N']) < len(pdb_tag_tally[pdb]['C']):
    elif n_term < c_term:
        # final_tags[pdb]['termini'] = 'C'
        termini = 'C'
    else:  # termini = 'Both'
        while True:
            termini = input('For %s, BOTH termini have the same number of matched tags.\n'
                            'The tag options are as follows {terminus:{tag name: count}}:\n%s\n'
                            'Which termini would you prefer?\n[n/c]:' % (pdb_code, pdb_tag_tally))
            termini = termini.upper()
            if termini == 'N' or termini == 'C':
                break

    # find the most common tag at the specific termini
    all_tags = []
    max_type = None
    max_count = 0
    # for _type in pdb_tag_tally[pdb][termini]:
    for _type in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][_type] > max_count:
            max_count = pdb_tag_tally[termini][_type]
            max_type = _type
    all_tags.append(max_type)
    # check if there are equally represented tags
    for _type in pdb_tag_tally[termini]:
        if pdb_tag_tally[termini][_type] == max_count and _type != max_type:
            all_tags.append(_type)
    final_tags['name'] = all_tags
    final_tags['termini'] = termini

    # Finally report results to the user and solve ambiguous tags
    final_choice = {}
    # for pdb in pdbs:
    while True:
        default = input('For %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf the Termini or Type is undesired, '
                        'you can see the underlying options by specifying \'o\'. Otherwise, \'%s\' will be chosen.\n'
                        'If you would like to proceed with the RECOMMENDED options, enter \'y\'.\nInput [o/y]:'
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

    # final_tag_sequence = {}
    # for pdb in pdbs:
    final_tag_sequence = {'name': final_choice['name'], 'seq': None}
    for partner_idx in matching_pdb_tags:
        for partner_tag in matching_pdb_tags[partner_idx]:
            if final_choice['name'] == matching_pdb_tags[partner_idx][partner_tag]['name']:
                final_tag_sequence['seq'] = matching_pdb_tags[partner_idx][partner_tag]['seq']
                # TODO align multiple and choose the consensus

    return final_tag_sequence


def find_expression_tags_multi(des_dir):
    """Take a pose and find expression tags from each PDB reference

    Args:
        des_dir (DesignDirectory): Object containing the pose information
    Returns:
        (dict): {pdb: {'name': 'His Tag', 'seq': 'MSGHHHHHHGKLKPNDLRI'}, ...}
    """
    # {'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...}
    pdbs = os.path.basename(des_dir.building_blocks).split('_')
    pose = SDUtils.read_pdb(des_dir.source)
    # design_seq_d = Ams.get_pdb_sequences(des_dir.source, chain=None)
    # {pdb: [1XYZ.A, 2ABC.B], ...}
    reference_seq_d, all_matching_pdb_chain = {}, {}
    for pdb_code, chain in zip(pdbs, pose.chain_id_list):
        all_matching_pdb_chain[pdb_code] = uniprot_pdb_d[pull_uniprot_ib_by_pdb(pdb_code, chain=chain)]['all']
        reference_seq_d[pdb_code] = reference_seq_d[chain]

    # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    partner_sequences = {}
    for pdb in pdbs:
        partner_sequences[pdb] = []
        for matching_pdb_chain in all_matching_pdb_chain[pdb]:
            matching_pdb, chain = matching_pdb_chain.split('.')
            partner_sequences[pdb].append(Ams.get_pdb_sequences(SDUtils.fetch_pdb(matching_pdb), chain=chain,
                                                                source='seqres'))

    # {pdb: {0: {1: {'name': tag_name, 'termini': 'N', 'seq': 'MSGHHHHHHGKLKPNDLRI'}}, ...}, ...}
    pdb_tags = {}
    for pdb in pdbs:
        pdb_tags[pdb] = {}
        for idx, seq in enumerate(partner_sequences[pdb]):
            # tags[pdb][idx] = find_tags(partner_sequences[pdb][idx])
            pdb_tags[pdb][idx] = find_tags(seq)

    # next, align all the tags to the reference sequence and tally the tag location and type
    pdb_tag_tally = {}
    for pdb in pdbs:
        # pdb_tag_tally[pdb] = {'N': 0, 'C': 0, 'types': {}}
        pdb_tag_tally[pdb] = {'N': {}, 'C': {}}
        for partner in pdb_tags[pdb]:
            for tags in pdb_tags[pdb][partner]:
                if tags != dict():
                    for tag in tags:
                        # count the number of termini
                        if tags[tag]['name'] in pdb_tag_tally[pdb][tags[tag]['termini']]:
                            # pdb_tag_tally[pdb]['N'][tag_name] +=1
                            pdb_tag_tally[pdb][tags[tag]['termini']][tags[tag]['name']] += 1
                        else:
                            pdb_tag_tally[pdb][tags[tag]['termini']][tags[tag]['name']] = 0
                        # # pdb_tag_tally[pdb][tags[tag]['termini']] += 1
                        # if tags[tag]['name'] in pdb_tag_tally[pdb]['types']:
                        #     # pdb_tag_tally[pdb]['types'][tag_name]
                        #     pdb_tag_tally[pdb]['types'][tags[tag]['name']] += 1
                        # else:
                        #     pdb_tag_tally[pdb]['types'][tags[tag]['name']] = 0

    final_tags = {}
    for pdb in pdbs:
        final_tags[pdb] = {}
        # final_tags[pdb] = {'termini': None, 'type': None}
        n_term, c_term = 0, 0
        if pdb_tag_tally[pdb]['N'] != dict():
            n_term = [pdb_tag_tally[pdb]['N'][_type] for _type in pdb_tag_tally[pdb]['N']]
            n_term = array(n_term).sum()
        if pdb_tag_tally[pdb]['C'] != dict():
            c_term = [pdb_tag_tally[pdb]['C'][_type] for _type in pdb_tag_tally[pdb]['C']]
            c_term = array(c_term).sum()
        if n_term == 0 and c_term == 0:
            # if len(pdb_tag_tally[pdb]['N']) > len(pdb_tag_tally[pdb]['C']):
            if n_term > c_term:
                # final_tags[pdb]['termini'] = 'N'
                termini = 'N'
            # elif len(pdb_tag_tally[pdb]['N']) < len(pdb_tag_tally[pdb]['C']):
            elif n_term < c_term:
                # final_tags[pdb]['termini'] = 'C'
                termini = 'C'
            else:  # termini = 'Both'
                while True:
                    termini = input('For %s, PDB %s, BOTH termini have the same number of matched tags.\n'
                                    'The tag options are as follows {terminus:{tag name: count}}:\n%s\n'
                                    'Which termini would you prefer?\n[n/c]:' % (des_dir, pdb, pdb_tag_tally[pdb]))
                    termini.upper()
                    if termini == 'N' or termini == 'C':
                        break

            # find the most common tag at the specific termini
            all_tags = []
            max_type = None
            max_count = 0
            # for _type in pdb_tag_tally[pdb][termini]:
            for _type in pdb_tag_tally[pdb][termini]:
                if pdb_tag_tally[pdb][termini][_type] > max_count:
                    max_count = pdb_tag_tally[pdb][termini][_type]
                    max_type = _type
            all_tags.append(max_type)
            # check if there are equally represented tags
            for _type in pdb_tag_tally[pdb][termini]:
                if pdb_tag_tally[pdb][termini][_type] == max_count and _type != max_type:
                    all_tags.append(_type)
            final_tags[pdb]['name'] = all_tags
            final_tags[pdb]['termini'] = termini

    # Finally report results to the user and solve ambiguous tags
    final_choice = {}
    for pdb in pdbs:
        default = input('For %s, PDB %s, the RECOMMENDED tag options are: Termini-%s Type-%s\nIf Termini or Type is '
                        'ambiguous, or undesired, you can see the underlying options and specify. Otherwise, one will '
                        'be randomly chosen.\nIf you '
                        'would like to proceed with the RECOMMENDED options, enter \'y\'. To choose from other options, specify '
                        '\'o\'.' % (des_dir, pdb, final_tags[pdb]['termini'], final_tags[pdb]['name']))
        if default.lower() == 'y':
            if len(final_tags[pdb]['name']) > 1:
                if 'His Tag' in final_tags:
                    final_choice[pdb]['name'] = 'His Tag'
                    # else choose the first choice
            final_choice[pdb]['name'] = final_tags[pdb]['name'][0]
            final_choice[pdb]['termini'] = final_tags[pdb]['termini']
        elif default.lower() == 'o':
            print('For %s, PDB %s, the FULL tag options are: %s\n' % (des_dir, pdb, pdb_tag_tally[pdb]))
            while True:
                termini_input = input('What termini would you like to use?\nInput [n/c]:')
                termini_input = termini_input.upper()
                if termini_input == 'N' or termini_input == 'C':
                    final_choice[pdb]['termini'] = termini_input
                    break
                else:
                    print('Input doesn\'t match. Please try again')
            while True:
                tag_input = input('What tag would you like to use? Enter the number of the below options.\n%s' %
                                  '\n'.join(['%d - %s' % (i, tag) for i, tag in enumerate(pdb_tag_tally[pdb][termini_input])]))
                if tag_input < len(pdb_tag_tally[pdb][termini_input]):
                    final_choice[pdb]['name'] = pdb_tag_tally[pdb][termini_input][tag_input]
                    break
                else:
                    print('Input doesn\'t match. Please try again')

    final_tag_sequence = {}
    for pdb in pdbs:
        final_tag_sequence[pdb] = {'name': final_choice['name']}
        for partner_idx in pdb_tags[pdb]:
            for tag_idx in pdb_tags[pdb][partner_idx]:
                if final_choice['name'] == pdb_tags[pdb][partner_idx][tag_idx]['name']:
                    final_tag_sequence[pdb]['seq'] = pdb_tags[pdb][partner_idx][tag_idx]['seq']
                    # TODO align multiple and choose the consensus

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
    alignment = Ams.generate_alignment(tag, sequence)
    tag_seq = alignment[0][0]
    seq = alignment[0][1]
    print(alignment[0])
    print(tag_seq)
    print(seq)
    # starting_index_of_seq2 = seq.find(sequence[0])
    # i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    final_seq = ''
    for i, (seq1_aa, seq2_aa) in enumerate(zip(tag_seq, seq)):
        if seq1_aa != seq2_aa:
            final_seq += seq1_aa[i]
        else:
            final_seq += seq2_aa[i]

    return final_seq


def pull_uniprot_id_by_pdb(pdb_code, chain=None):
    # uniprot_pdb_d = pickle.load(unidictf)
    source = 'unique_pdb'
    pdb_code = pdb_code.upper()
    if chain:
        # pdb_code = '%s.%s' % (pdb_code, chain)
        # source = 'all'
        dummy = 'TODO ensure that this works once the database is integrated'  # TODO

    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id


def find_tags(seq, tag_file=PUtils.affinity_tags, alignment_length=12):
    """Find all strings (tags) on an input string (sequence) from a reference set of strings

    Args:
        chain_seq_d (dict): {'A': 'MSGHHHHHHGKLKPNDLRI...', ...}
    Keyword Args:
        tags=affinity_tags (list): List of tuples where tuple[0] is the name and tuple[1] is the string
    Returns:
        tag_dict (dict): {1: {'name': tag_name, 'termini': 'N', 'seq': 'MSGHHHHHHGKLKPNDLRI'}}, ...}, ...}
    """
    with open(tag_file, 'r') as f:
        reader = csv.reader(f)
        tags = {row[0]: row[1] for row in reader}

    tag_dict = {}
    count = 1
    # for j, tag in enumerate(tags):
    for tag in tags:
        # tag_name = tag[0]
        # tag_seq = tag[1]
        if seq.find(tags[tag]) > -1:
            # if tag is found save the tag name, the termini it is closer to, and the source sequence concatenation
            tag_dict[count] = {}
            tag_index = seq.find(tags[tag])
            tag_dict[count]['name'] = tag
            # tag_dict[count]['name'] = tag_name
            alignment_index = len(tags[tag]) + alignment_length

            if tag_index == 0 or tag_index < len(seq)/2:
                # print('Tag is at the N-term. \n')
                tag_dict[count]['termini'] = 'N'
                # tag_dict[chain][tag_name]['termini'] = 'N'
                # while tag_index - alignment_index > len(seq) or tag_index - alignment_index < 0:
                #     alignment_index -= 1
                tag_dict[count]['seq'] = seq[tag_index:tag_index + alignment_index]
                # tag_dict[chain][tag_name]['seq'] = seq[tag_index:alignment_index]
            else:  # tag_index + len(tag_seq) == len(test_seq):
                # print('Tag is at the C-term.\n')
                tag_dict[count]['termini'] = 'C'
                # op = operator.sub()
                # while tag_index - alignment_index > len(seq) or tag_index - alignment_index < 0:
                #     alignment_index -= 1
                tag_dict[count]['seq'] = seq[tag_index - alignment_index:tag_index + len(tags[tag])]
            print('Original Seq: %s' % seq)
            print('Alignment index are from %d to %d' % (tag_index, alignment_index))
            print('Final Seq: %s' % tag_dict[count]['seq'])

    return tag_dict
