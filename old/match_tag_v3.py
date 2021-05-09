"""Add expression tags onto the termini of specific designs"""
import csv
import os

from old import AnalyzeMutatedSequences
import PathUtils as PUtils
import Pose
import SymDesignUtils as SDUtils


# def convert_aa3to1(seqlist):
#     newlist = []
#     for aa in seqlist:
#         newlist.append(aa_dict_to_1letter[aa])
#     newlist = ''.join(newlist)
#     return newlist

# aa_dict_to_1letter from Kyle's module
# aa_dict_to_1letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'MSE': 'M'}  # , 'KCX': 'K', 'LLP': 'K'}

# with open(sys.argv[1], 'r') as pdb_file:
#     lines_list = []
#
#     # Make a list of all the lines that start with SEQRES
#     for row in pdb_file:
#         if row.startswith('SEQRES'):
#             thisline_list = row[19:70].split(' ')
#             thisline_list = list(filter(None, thisline_list))
#             thisline_list.insert(0, row[11])
#             lines_list.append(thisline_list)
#
#     # string id list:
#     chainid = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     i = 0
#     thischain = []
#
#     sequences = []
#     for item in lines_list:
#         if item[0] == chainid[i]:
#             thischain += item[1:]
#         elif item[0] == chainid[i+1]:
#             thischain = convert_aa3to1(thischain)
#             sequences.append('Chain ' + chainid[i] + ' ' + thischain)
#             thischain = []
#             thischain += item[1:]
#             i += 1
#     else:
#         thischain = convert_aa3to1(thischain)
#         sequences.append('Chain ' + chainid[i] + ' ' + thischain)

# Now we have a list of chain id + chain sequence

##################


def find_expression_tags(des_dir):
    """Take a pose and find expression tags from each PDB reference

    Args:
        des_dir (DesignDirectory): Object containing the pose information
    Returns:
        (dict): {design_chain: sequence, ...}
    """
    # KM {chain: seq}
    # need to have reference_files mapped from design_chain to pdb_code
    # I can accomplish this with a deconstruction of the design.pdb into a DesignDirectory object and pull out the
    # oligomer names from .composition assigning A to composition.split('_')[0] and B to [1]<- need chain
    # ambivilance though probably isn't hard. Or could throw all of these into the /data directory to grab in future.
    # I don't need to map the reference to the design because of the source information.

    # {'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...}
    pdbs = os.path.basename(des_dir.building_blocks).split('_')
    desirgn_seq_d = AnalyzeMutatedSequences.get_pdb_sequences(des_dir.source, chain=None)
    # {pdb: [1XYZ.A, 2ABC.B], ...}
    reference_seq_d, all_matching_pdb_chain = {}, {}
    for pdb_code, chain in zip(pdbs, desirgn_seq_d):
        all_matching_pdb_chain[pdb_code] = uniprot_pdb_d[pull_uniprot_ib_by_pdb(pdb_code, chain=chain)]['all']
        reference_seq_d[pdb_code] = reference_seq_d[chain]

    # {pdb: [{'A': 'MSGHHHHHHGKLKPNDLRI...'}, ...], ...}
    partner_sequences = {}
    for pdb in pdbs:
        partner_sequences[pdb] = []
        for matching_pdb_chain in all_matching_pdb_chain[pdb]:
            matching_pdb, chain = matching_pdb_chain.split('.')
            partner_sequences[pdb].append(
                AnalyzeMutatedSequences.get_pdb_sequences(Pose.fetch_pdb_file(matching_pdb), chain=chain,
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
            n_term = np.array(n_term).sum()
        if pdb_tag_tally[pdb]['C'] != dict():
            c_term = [pdb_tag_tally[pdb]['C'][_type] for _type in pdb_tag_tally[pdb]['C']]
            c_term = np.array(c_term).sum()
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
                        '\'o\'.' % (des_dir, pdb, final_tags[pdb]['termini'], final_tags[pdb]['type']))
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
        for partner_idx in pdb_tags[pdb]:
            for tag_idx in pdb_tags[pdb][partner_idx]:
                if final_choice['name'] == pdb_tags[pdb][partner_idx][tag_idx]['name']:
                    final_tag_sequence[pdb] = pdb_tags[pdb][partner_idx][tag_idx]['seq']
                    # TODO align multiple and choose the consensus

    return final_tag_sequence


def add_expression_tags(sequence, tag):
    """Take a raw sequence and add expression tag by aligning a specified tag by PDB reference

    Args:
        sequence (str):
        tag (dict):
    Returns:
        tagged_sequence (str): The final sequence with the tag added
    """

uniprot_pdb_d = SDUtils.unpickle(PUtils.uniprot_pdb_map)


def pull_uniprot_ib_by_pdb(pdb_code, chain=False):
    # uniprot_pdb_d = pickle.load(unidictf)
    source = 'unique_pdb'
    if chain:
        pdb_code = '%s.%s' % (pdb_code, chain)
        source = 'all'

    # pdb_chain = pdb_code + '.' + chain
    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id, uniprot_pdb_d[uniprot_id]


with open(PUtils.affinity_tags, 'r') as f:
    affinity_tags = csv.reader(f)


def find_tags(seq, tags=affinity_tags, alignment_length=12):  # TODO get rid of chains
    """Find all strings (tags) on an input string (sequence) from a reference set of strings

    Args:
        chain_seq_d (dict): {'A': 'MSGHHHHHHGKLKPNDLRI...', ...}
    Keyword Args:
        tags=affinity_tags (list): List of tuples where tuple[0] is the name and tuple[1] is the string
    Returns:
        tag_dict (dict): {1: {'name': tag_name, 'termini': 'N', 'seq': 'MSGHHHHHHGKLKPNDLRI'}}, ...}, ...}
    """

    # TODO, grab multiple tags
    tag_dict = {}
    # for chain in chain_seq_d:
    #     test_seq = chain_seq_d[chain]
    #     tag_dict[chain] = {}
    count = 1
    for j, tag in enumerate(tags):
        tag_name = tag[0]
        tag_seq = tag[1]
        # tag_len = len(tag_seq)
        # if j == 28:
        #     print(chain_name + ' does not contain any tag.')
        #     break
        if seq.find(tag_seq) > -1:
            # if tag is found save the tag name, the termini it is closer to, and the source sequence concatenation
            tag_dict[count] = {}
            tag_index = seq.find(tag_seq)
            tag_dict[count]['name'] = tag_name
            alignment_index = len(tag_seq) + alignment_length
            # subsection = test_seq[tag_index:]
            # if subsection > len(tag_seq) +

            # fromC = len(test_seq) - tag_len - tag_index
            # print(chain_name + ' contains ' + tag_name)
            # print('Tag starts at index ' + str(tag_index))
            # print('Length of tag sequence: ' + str(tag_len))
            # print('Length of input sequence: ' + str(len(test_seq)))

            if tag_index == 0 or tag_index < len(seq)/2:
                # print('Tag is at the N-term. \n')
                tag_dict[count]['termini'] = 'N'
                # tag_dict[chain][tag_name]['termini'] = 'N'
                while tag_index - alignment_index > len(seq) or tag_index - alignment_index < 0:
                    alignment_index -= 1
                tag_dict[count]['seq'] = seq[tag_index:alignment_index]
                # tag_dict[chain][tag_name]['seq'] = seq[tag_index:alignment_index]
            else:  # tag_index + len(tag_seq) == len(test_seq):
                # print('Tag is at the C-term.\n')
                tag_dict[count]['termini'] = 'C'
                # op = operator.sub()
                while tag_index - alignment_index > len(seq) or tag_index - alignment_index < 0:
                    alignment_index -= 1
                tag_dict[count]['seq'] = seq[alignment_index:tag_index + len(tag_seq)]

            # while op(tag_index, alignment_index) > len(test_seq) or op(tag_index, alignment_index) < 0:
            #     alignment_index -= 1

            # elif tag_index < fromC:
            #     print('Tag is closer to the N-term, with '
            #           + str(tag_index) + ' residues from the N-term.\n')
            # elif tag_index > fromC:
            #     print('Tag is closer to the C term, with '
            #           + str(fromC) + ' residues from the C-term.\n')

    return tag_dict
        # else:
            # j += 1
    # else:
        # f.close()
# else:
#     print('--END--')
