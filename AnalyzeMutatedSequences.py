import sys
import os
import argparse
import math
from glob import glob
from itertools import combinations, repeat
import PDB
from Bio.SeqUtils import IUPACData
from Bio.Alphabet import generic_protein
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, pdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import PathUtils as PUtils
import SymDesignUtils as SDUtils


# Globals
logger = SDUtils.start_log(__name__)
db = PUtils.biological_fragmentDB
index_offset = SDUtils.index_offset


def remove_non_mutations(frequency_msa, residue_list):
    """Keep residues which are present in provided list

    Args:
        frequency_msa (dict): {0: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 1: {}, ...}
        residue_list (list): [15, 16, 18, 20, 34, 35, 67, 108, 119]
    Returns:
        mutation_dict (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
    """
    mutation_dict = {}
    for residue in frequency_msa:
        if residue in residue_list:
            mutation_dict[residue] = frequency_msa[residue]

    return mutation_dict


def return_consensus_design(frequency_sorted_msa):
    for residue in frequency_sorted_msa:
        if residue == 0:
            pass
        else:
            if len(frequency_sorted_msa[residue]) > 2:
                for alternative in frequency_sorted_msa[residue]:
                    # Prepare for Letter sorting SchemA
                    sequence_logo = None
            else:
                # DROP from analysis...
                frequency_sorted_msa[residue] = None


def pos_specific_jsd(msa, background):
    """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency

    Both msa_dictionary and background must be the same index
    Args:
        msa (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        background (dict): {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
            Must contain residue index with inner dictionary of single amino acid types
    Returns:
        divergence_dict (dict): {15: 0.732, 16: 0.552, ...}
    """
    return {residue: res_divergence(msa[residue], background[residue]) for residue in msa if residue in background}


def res_divergence(position_freq, bgd_freq, jsd_lambda=0.5):
    """Calculate residue specific Jensen-Shannon Divergence value

    Args:
        position_freq (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        bgd_freq (dict): {15: {'A': 0, 'R': 0, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        divergence (float): 0.732, Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    sum_prob1, sum_prob2 = 0, 0
    for aa in position_freq:
        p = position_freq[aa]
        q = bgd_freq[aa]
        r = (jsd_lambda * p) + ((1 - jsd_lambda) * q)
        if r == 0:
            continue
        if q != 0:
            prob2 = (q * math.log(q / r, 2))
            sum_prob2 += prob2
        if p != 0:
            prob1 = (p * math.log(p / r, 2))
            sum_prob1 += prob1
    divergence = round(jsd_lambda * sum_prob1 + (1 - jsd_lambda) * sum_prob2, 3)

    return divergence


def create_bio_msa(sequence_dict):
    """
    Args:
        sequence_dict (dict): {name: sequence, ...}
            ex: {'clean_asu': 'MNTEELQVAAFEI...', ...}
    Returns:
        new_alignment (MultipleSeqAlignment): [SeqRecord(Seq("ACTGCTAGCTAG", generic_dna), id="Alpha"),
                                               SeqRecord(Seq("ACT-CTAGCTAG", generic_dna), id="Beta"), ...]
    """
    sequences = [SeqRecord(Seq(sequence_dict[name], generic_protein), id=name) for name in sequence_dict]
    # for name in sequence_dict:
    #     sequences.append(SeqRecord(Seq(sequence_dict[name], generic_protein), id=name))
    new_alignment = MultipleSeqAlignment(sequences)

    return new_alignment


def generate_mutations(all_design_files, wild_type_file, pose_num=False):
    """From a list of PDB's and a wild-type PDB, generate a list of 'A5K' style mutations

    Args:
        all_design_files (list): PDB files on disk to extract sequence info and compare
        wild_type_file (str): PDB file on disk which contains a reference sequence
    Returns:
        mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    """
    pdb_dict = {'ref': SDUtils.read_pdb(wild_type_file)}
    for file_name in all_design_files:
        pdb = SDUtils.read_pdb(file_name)
        pdb.AddName(os.path.splitext(os.path.basename(file_name))[0])
        pdb_dict[pdb.name] = pdb

    return extract_sequence_from_pdb(pdb_dict, mutation=True, pose_num=pose_num)  # , offset=False)


#####################
# Sequence handling
#####################


def write_fasta_file(sequence, name, outpath=os.getcwd()):
    """Write a fasta file from sequence(s)

    Args:
        sequence (iterable): One of either list, dict, or string. If list, can be list of tuples(name, sequence),
            list of lists, etc. Smart solver using object type
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
    Returns:
        (str): The name of the output file
    """
    file_name = os.path.join(outpath, name + '.fasta')
    with open(file_name, 'w') as outfile:
        if type(sequence) is list:
            if type(sequence[0]) is list:  # where inside list is of alphabet (AA or DNA)
                for idx, seq in enumerate(sequence):
                    outfile.write('>%s_%d\n' % (name, idx))  # header
                    if len(seq[0]) == 3:  # Check if alphabet is 3 letter protein
                        outfile.write(' '.join(aa for aa in seq))
                    else:
                        outfile.write(''.join(aa for aa in seq))
            elif isinstance(sequence[0], str):
                outfile.write('>%s\n%s\n' % name, ' '.join(aa for aa in sequence))
            elif type(sequence[0]) is tuple:  # where seq[0] is header, seq[1] is seq
                outfile.write('\n'.join('>%s\n%s' % seq for seq in sequence))
            else:
                raise SDUtils.DesignError('Cannot parse data to make fasta')
        elif isinstance(sequence, dict):
            outfile.write('\n'.join('>%s\n%s' % (seq_name, sequence[seq_name]) for seq_name in sequence))
        elif isinstance(sequence, str):
            outfile.write('>%s\n%s\n' % (name, sequence))
        else:
            raise SDUtils.DesignError('Cannot parse data to make fasta')

    return file_name


def write_multi_line_fasta_file(sequences, name, path=os.getcwd()):  # REDUNDANT DEPRECIATED
    """Write a multi-line fasta file from a dictionary where the keys are >headers and values are sequences

    Args:
        sequences (dict): {'my_protein': 'MSGFGHKLGNLIGV...', ...}
        name (str): The name of the file to output
    Keyword Args:
        path=os.getcwd() (str): The location on disk to output file
    Returns:
        (str): The name of the output file
    """
    file_name = os.path.join(path, name)
    with open(file_name, 'r') as f:
        # f.write('>%s\n' % seq)
        f.write('\n'.join('>%s\n%s' % (seq_name, sequences[seq_name]) for seq_name in sequences))

    return file_name


def extract_aa_seq(pdb, aa_code=1, source='atom', chain=0):
    # Extracts amino acid sequence from either ATOM or SEQRES record of PDB object
    if type(chain) == int:
        chain = pdb.chain_id_list[chain]
    final_sequence = None
    sequence_list = []
    failures = []
    aa_code = int(aa_code)

    if source == 'atom':
        # Extracts sequence from ATOM records
        if aa_code == 1:
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and (atom.alt_location == '' or atom.alt_location == 'A'):
                    try:
                        sequence_list.append(IUPACData.protein_letters_3to1[atom.residue_type.title()])
                    except KeyError:
                        sequence_list.append('X')
                        failures.append((atom.residue_number, atom.residue_type))
            final_sequence = ''.join(sequence_list)
        elif aa_code == 3:
            for atom in pdb.all_atoms:
                if atom.chain == chain and atom.type == 'N' and atom.alt_location == '' or atom.alt_location == 'A':
                    sequence_list.append(atom.residue_type)
            final_sequence = sequence_list
        else:
            logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))

    elif source == 'seqres':
        # Extract sequence from the SEQRES record
        fail = False
        while True:
            if chain in pdb.sequence_dictionary:
                sequence = pdb.sequence_dictionary[chain]
                break
            else:
                if not fail:
                    temp_pdb = PDB.PDB()
                    temp_pdb.readfile(pdb.filepath, coordinates_only=False)
                    fail = True
                else:
                    raise SDUtils.DesignError('Invalid PDB input, no SEQRES record found')
        if aa_code == 1:
            final_sequence = sequence
            for i in range(len(sequence)):
                if sequence[i] == 'X':
                    failures.append((i, sequence[i]))
        elif aa_code == 3:
            for i, residue in enumerate(sequence):
                sequence_list.append(IUPACData.protein_letters_1to3[residue])
                if residue == 'X':
                    failures.append((i, residue))
            final_sequence = sequence_list
        else:
            logger.critical('In %s, incorrect argument \'%s\' for \'aa_code\'' % (aa_code, extract_aa_seq.__name__))
    else:
        raise SDUtils.DesignError('Invalid sequence input')

    return final_sequence, failures


def pdb_to_pose_num(reference_dict):
    """Take a dictionary with chain name as keys and return the length of values as reference length"""
    offset_dict = {}
    prior_chain, prior_chains_len = None, 0
    for i, chain in enumerate(reference_dict):
        if i > 0:
            prior_chains_len += len(reference_dict[prior_chain])
        offset_dict[chain] = prior_chains_len
        # insert function here? Make this a decorator!?
        prior_chain = chain

    return offset_dict


def extract_sequence_from_pdb(pdb_class_dict, aa_code=1, seq_source='atom', mutation=False, pose_num=True,
                              outpath=None):
    """Extract the sequence from PDB objects

    Args:
        pdb_class_dict (dict): {pdb_code: PDB object, ...}
    Keyword Args:
        aa_code=1 (int): Whether to return sequence with one-letter or three-letter code [1,3]
        seq_source='atom' (str): Whether to return the ATOM or SEQRES record ['atom','seqres','compare']
        mutation=False (bool): Whether to return mutations in sequences compared to a reference.
            Specified by pdb_code='ref'
        outpath=None (str): Where to save the results to disk
    Returns:
        mutation_dict (dict): IF mutation=True {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...},
            ...}
        or
        sequence_dict (dict): ELSE {pdb: {chain_id: 'Sequence', ...}, ...}
    """
    reference_seq_dict = None
    if mutation:
        # If looking for mutations, the reference PDB object should be given as 'ref' in the dictionary
        if 'ref' not in pdb_class_dict:
            sys.exit('No reference sequence specified, but mutations requested. Include a key \'ref\' in PDB dict!')
        reference_seq_dict = {}
        fail_ref = []
        reference = pdb_class_dict['ref']
        for chain in pdb_class_dict['ref'].chain_id_list:
            reference_seq_dict[chain], fail = extract_aa_seq(reference, aa_code, seq_source, chain)
            if fail != list():
                fail_ref.append((reference, chain, fail))

        if fail_ref:
            logger.error('Ran into following errors generating mutational analysis reference:\n%s' % str(fail_ref))

    if seq_source == 'compare':
        mutation = True

    def handle_extraction(pdb_code, _pdb, _aa, _source, _chain):
        if _source == 'compare':
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, 'atom', _chain)
            sequence2, failures2 = extract_aa_seq(_pdb, _aa, 'seqres', _chain)
        else:
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, _source, _chain)
            sequence2 = reference_seq_dict[_chain]
            sequence_dict[pdb_code][_chain] = sequence1
        if mutation:
            seq_mutations = generate_mutations_from_seq(sequence1, sequence2, offset=False, remove_blanks=False)
            mutation_dict[pdb_code][_chain] = seq_mutations
        if failures1:
            error_list.append((_pdb, _chain, failures1))

    error_list = []
    sequence_dict = {}
    mutation_dict = {}
    for pdb in pdb_class_dict:
        sequence_dict[pdb] = {}
        mutation_dict[pdb] = {}
        # if pdb == 'ref':
        # if len(chain_dict[pdb]) > 1:
        #     for chain in chain_dict[pdb]:
        for chain in pdb_class_dict[pdb].chain_id_list:
            handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain)
        # else:
        #     handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain_dict[pdb])

    if outpath:
        sequences = {}
        for pdb in sequence_dict:
            for chain in sequence_dict[pdb]:
                sequences[pdb + '_' + chain] = sequence_dict[pdb][chain]
        filepath = write_multi_line_fasta_file(sequences, 'sequence_extraction.fasta', path=outpath)
        logger.info('The following file was written:\n%s' % filepath)

    if error_list:
        logger.error('The following residues were not extracted:\n%s' % str(error_list))

    if mutation:
        for chain in reference_seq_dict:
            for i, aa in enumerate(reference_seq_dict[chain]):
                mutation_dict['ref'][chain][i + index_offset] = {'from': reference_seq_dict[chain][i],
                                                                 'to': reference_seq_dict[chain][i]}
        if pose_num:
            new_mutation_dict = {}
            offset_dict = pdb_to_pose_num(reference_seq_dict)
            for chain in offset_dict:
                for pdb in mutation_dict:
                    if pdb not in new_mutation_dict:
                        new_mutation_dict[pdb] = {}
                    new_mutation_dict[pdb][chain] = {}
                    for mutation in mutation_dict[pdb][chain]:
                        new_mutation_dict[pdb][chain][mutation+offset_dict[chain]] = mutation_dict[pdb][chain][mutation]
            mutation_dict = new_mutation_dict

        return mutation_dict
    else:
        return sequence_dict


def make_sequences_from_mutations(wild_type, mutation_dict, aligned=False):
    """Takes a list of sequence mutations and returns the mutated form on wildtype

    Args:
        wild_type (str): Sequence to mutate
        mutation_dict (dict): {name: {mutation_index: {'from': AA, 'to': AA}, ...}, ...}, ...}
    Keyword Args:
        aligned=False (bool): Whether the input sequences are already aligned
        output=False (bool): Whether to make a .fasta file of the sequence
    Returns:
        all_sequences (dict): {name: sequence, ...}
    """
    return {pdb: make_mutations(wild_type, mutation_dict[pdb], find_orf=not aligned) for pdb in mutation_dict}


def make_mutations(seq, mutations, find_orf=True):
    """Modify a sequence to contain mutations specified by a mutation dictionary

    Args:
        seq (str): 'Wild-type' sequence to mutate
        mutations (dict): {mutation_index: {'from': AA, 'to': AA}, ...}
    Keyword Args:
        find_orf=True (bool): Whether or not to find the correct ORF for the mutations and the seq
    Returns:
        seq (str): The mutated sequence
    """
    # Seq can be either list or string
    if find_orf:
        offset = -find_orf_offset(seq, mutations)
        logger.info('Found ORF. Offset = %d' % -offset)
    else:
        offset = index_offset

    # zero index seq and 1 indexed mutation_dict
    index_errors = []
    for key in mutations:
        try:
            if seq[key - offset] == mutations[key]['from']:  # adjust seq for zero index slicing
                seq = seq[:key - offset] + mutations[key]['to'] + seq[key - offset + 1:]
            else:  # find correct offset, or mark mutation source as doomed
                index_errors.append(key)
        except IndexError:
            print(key - offset)
    if index_errors:
        logger.warning('Index errors:\n%s' % str(index_errors))

    return seq


def find_orf_offset(seq, mutations):
    """Using one sequence and mutation data, find the sequence offset which matches mutations closest

    Args:
        seq (str): 'Wild-type' sequence to mutate
        mutations (dict): {mutation_index: {'from': AA, 'to': AA}, ...}
    Returns:
        orf_offset (int): The index to offset the sequence by in order to match the mutations the best
    """
    met_offset_list = []
    for i, aa in enumerate(seq):
        if aa == 'M':
            met_offset_list.append(i)
    if met_offset_list:
        # Weight potential MET offsets by finding the one which gives the highest number correct mutation sites
        which_met_offset_counts = []
        for index in met_offset_list:
            index -= index_offset
            s = 0
            for mut in mutations:
                try:
                    if seq[mut + index] == mutations[mut][0]:
                        s += 1
                except IndexError:
                    break
            which_met_offset_counts.append(s)
        max_count = np.max(which_met_offset_counts)
    else:
        max_count = 0

    # Check if likely ORF has been identified (count < number mutations/2). If not, MET is missing/not the ORF start
    if max_count < len(mutations) / 2:
        upper_range = 50  # This corresponds to how far away the max seq start is from the ORF MET start site
        offset_list = []
        for i in range(0, upper_range):
            s = 0
            for mut in mutations:
                if seq[mut + i] == mutations[mut]['from']:
                    s += 1
            offset_list.append(s)
        max_count = np.max(offset_list)
        # find likely orf offset index
        orf_offset = offset_list.index(max_count)  # + lower_range  # + mut_index_correct
    else:
        orf_offset = met_offset_list[which_met_offset_counts.index(max_count)] - index_offset

    return orf_offset


def parse_mutations(mutation_list):  # UNUSED
    if isinstance(mutation_list, str):
        mutation_list = mutation_list.split(', ')

    # Takes a list of mutations in the form A37K and parses the index (37), the FROM aa (A), and the TO aa (K)
    # output looks like {37: ('A', 'K'), 440: ('K', 'Y'), ...}
    mutation_dict = {}
    for mutation in mutation_list:
        to_letter = mutation[-1]
        from_letter = mutation[0]
        index = int(mutation[1:-1])
        mutation_dict[index] = (from_letter, to_letter)

    return mutation_dict


def generate_mutations_from_seq(seq1, seq2, offset=True, remove_blanks=True):
    """Create mutations with format A5K, one-indexed

    Index so residue value starts at 1. For PDB file comparison, seq1 should be crystal sequence (ATOM), seq2 should be
     expression sequence (SEQRES)
    Args:
        seq1 (str): Mutant sequence
        seq2 (str): Wild-type sequence
    Keyword Args:
        offset=True (bool): Whether to calculate alignment offset
        remove_blanks=True (bool): Whether to remove all sequence that has zero index or missing residues
    Returns:
        mutations (dict): {index: {'from': 'A', 'to': 'K'}, ...}
    """
    if offset:
        alignment = SDUtils.generate_alignment(seq1, seq2)
        align_seq_1 = alignment[0][0]
        align_seq_2 = alignment[0][1]
    else:
        align_seq_1 = seq1
        align_seq_2 = seq2

    # Extract differences from the alignment
    starting_index_of_seq2 = align_seq_2.find(seq2[0])
    i = -starting_index_of_seq2 + index_offset  # make 1 index so residue value starts at 1
    mutations = {}
    for seq1_aa, seq2_aa in zip(align_seq_1, align_seq_2):
        if seq1_aa != seq2_aa:
            mutations[i] = {'from': seq2_aa, 'to': seq1_aa}
            # mutation_list.append(str(seq2_aa) + str(i) + str(seq1_aa))
        i += 1

    if remove_blanks:
        # Remove any blank mutations and negative/zero indices
        remove_mutation_list = []
        for entry in mutations:
            if entry > 0:
                # if mutations[entry].find('-') == -1:
                for index in mutations[entry]:
                    if mutations[entry][index] == '-':
                    # if mutations[entry][index] == '0':
                        remove_mutation_list.append(entry)

        for entry in remove_mutation_list:
            mutations.pop(entry)

    return mutations


def make_mutations_chain_agnostic(mutation_dict):
    """Remove chain identifier from mutation dictionary

    Args:
        mutation_dict (dict): {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Returns:
        flattened_dict (dict): {pdb: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}
    """
    flattened_dict = {}
    for pdb in mutation_dict:
        flattened_dict[pdb] = {}
        for chain in mutation_dict[pdb]:
            flattened_dict[pdb].update(mutation_dict[pdb][chain])

    return flattened_dict


def simplify_mutation_dict(mutation_dict, to=True):
    """Simplify mutation dictionary to 'to'/'from' AA key

    Args:
        mutation_dict (dict): {pdb: {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    Keyword Args:
        to=True (bool): Whether to use 'to' AA (True) or 'from' AA (False)
    Returns:
        mutation_dict (dict): {pdb: {mutation_index: 'K', ...}, ...}
    """
    simplification = get_mutation_to
    if not to:
        simplification = get_mutation_from

    for pdb in mutation_dict:
        for index in mutation_dict[pdb]:
            mutation_dict[pdb][index] = simplification(mutation_dict[pdb][index])

    return mutation_dict


def get_mutation_from(mutation_dict):
    """Remove 'to' identifier from mutation dictionary

    Args:
        mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
    Returns:
        mutation_dict (str): 'A'
    """
    return mutation_dict['from']


def get_mutation_to(mutation_dict):
    """Remove 'from' identifier from mutation dictionary
    Args:
        mutation_dict (dict): {mutation_index: {'from': 'A', 'to': 'K'}, ...},
    Returns:
        mutation_dict (str): 'K'
    """
    return mutation_dict['to']


def generate_sequences(wild_type_seq_dict, all_design_mutations):
    """Separate chains from mutation dictionary and generate mutated sequences

    Args:
        wild_type_seq_dict (dict): {chain: sequence, ...}
        all_design_mutations (dict): {'name': {chain: {mutation_index: {'from': AA, 'to': AA}, ...}, ...}, ...}
            Index so mutation_index starts at 1
    Returns:
        mutated_sequences (dict): {chain: {name: sequence, ...}
    """
    mutated_sequences = {}
    for chain in wild_type_seq_dict:
        chain_mutation_dict = {}
        for pdb in all_design_mutations:
            if chain in all_design_mutations[pdb]:
                chain_mutation_dict[pdb] = all_design_mutations[pdb][chain]
        mutated_sequences[chain] = make_sequences_from_mutations(wild_type_seq_dict[chain], chain_mutation_dict,
                                                                 aligned=True)

    return mutated_sequences


def get_wildtype_file(des_directory):
    """Retrieve the wild-type file name from Design Directory"""
    # wt_file = glob(os.path.join(des_directory.building_blocks, PUtils.clean))
    wt_file = glob(os.path.join(des_directory.path, PUtils.clean))
    assert len(wt_file) == 1, '%s: More than one matching file found with %s' % (des_directory.path, PUtils.asu)
    return wt_file[0]
    # for file in os.listdir(des_directory.building_blocks):
    #     if file.endswith(PUtils.asu):
    #         return os.path.join(des_directory.building_blocks, file)


def get_pdb_sequences(pdb_file, chain=None):
    """Return all sequences or those specified by a chain from a PDB file

    Args:
        pdb_file (str): Location on disk of a reference .pdb file
    Keyword Args:
        chain=None (str): If a particular chain is desired, specify it
    Returns:
        wt_seq_dict (dict): {chain: sequence, ...}
    """
    wt_pdb = SDUtils.read_pdb(pdb_file)
    wt_seq_dict = {}
    for _chain in wt_pdb.chain_id_list:
        wt_seq_dict[_chain], fail = extract_aa_seq(wt_pdb, chain=_chain)
    if chain:
        wt_seq_dict = SDUtils.clean_dictionary(wt_seq_dict, chain, remove=False)

    return wt_seq_dict


def mutate_wildtype_sequences(sequence_dir_files, wild_type_file):
    """Take a directory with PDB files and compare to a Wild-type PDB"""
    wt_seq_dict = get_pdb_sequences(wild_type_file)
    return generate_sequences(wt_seq_dict, generate_mutations(sequence_dir_files, wild_type_file))


def weave_mutation_dict(sorted_freq, mut_prob, resi_divergence, int_divergence, des_divergence):
    """Make final dictionary, index to sequence

    Args:
        sorted_freq (dict): {15: ['S', 'A', 'T'], ... }
        mut_prob (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        resi_divergence (dict): {15: 0.732, 16: 0.552, ...}
        int_divergence (dict): {15: 0.732, 16: 0.552, ...}
        des_divergence (dict): {15: 0.732, 16: 0.552, ...}
    Returns:
        weaved_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    """
    weaved_dict = {}
    for residue in sorted_freq:
        final_resi = residue + SDUtils.index_offset
        weaved_dict[final_resi] = {}
        for aa in sorted_freq[residue]:
            weaved_dict[final_resi][aa] = round(mut_prob[residue][aa], 3)
        weaved_dict[final_resi]['jsd'] = resi_divergence[residue]
        weaved_dict[final_resi]['int_jsd'] = int_divergence[residue]
        weaved_dict[final_resi]['des_jsd'] = des_divergence[residue]

    return weaved_dict


def weave_sequence_dict(base_dict=None, **kwargs):  # *args, # sorted_freq, mut_prob, resi_divergence, int_divergence):
    """Make final dictionary indexed to sequence, from same-indexed, residue numbered, sequence dictionaries

    Args:
        *args (dict)
    Keyword Args:
        base=None (dict): Original dictionary
        **kwargs (dict): key=dictionary pairs to include in the final dictionary
            sorted_freq={15: ['S', 'A', 'T'], ... }, mut_prob={15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...},
                divergence (dict): {15: 0.732, 16: 0.552, ...}
    Returns:
        weaved_dict (dict): {16: {'freq': {'S': 0.134, 'A': 0.050, ...,} 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    """
    if base_dict:
        weaved_dict = base_dict
    else:
        weaved_dict = {}

    # print('kwargs', kwargs)
    for seq_dict in kwargs:
        # print('seq_dict', seq_dict)
        for residue in kwargs[seq_dict]:
            if residue not in weaved_dict:
                weaved_dict[residue] = {}
            # else:
            #     weaved_dict[residue][seq_dict] = {}
            if isinstance(kwargs[seq_dict][residue], dict):  # TODO make endlessly recursive?
                weaved_dict[residue][seq_dict] = {}
                for sub_key in kwargs[seq_dict][residue]:  # kwargs[seq_dict][residue]
                    weaved_dict[residue][seq_dict][sub_key] = kwargs[seq_dict][residue][sub_key]
            else:
                weaved_dict[residue][seq_dict] = kwargs[seq_dict][residue]

    # ensure all residues in weaved_dict have every keyword
    # missing_keys = {}
    # for residue in weaved_dict:
    #     missing_set = set(kwargs.keys()) - set(weaved_dict[residue].keys())
    #     if missing_set:
    #         for missing in missing_set:
    #             weaved_dict[residue][missing] = None
        # missing_keys[residue] = set(kwargs.keys()) - set(weaved_dict[residue].keys())
    # for residue in missing_keys:

    return weaved_dict


def multi_chain_alignment(mutated_sequences):
    """Combines different chain's Multiple Sequence Alignments into a single MSA

    Args:
        mutated_sequences (dict): {chain: {name: sequence, ...}
    Returns:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...,
                                'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {0: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 1: {}, ...},
                                'rep': {0: 210, 1: 211, 2:211, ...}}
            Zero-indexed counts and rep dictionary elements
    """
    alignment = {chain: create_bio_msa(mutated_sequences[chain]) for chain in mutated_sequences}

    # Combine alignments for all chains from design file Ex: A: 1-102, B: 130. Alignment: 1-232
    first = True
    total_alignment = None
    for chain in alignment:
        if first:
            total_alignment = alignment[chain][:, :]
            first = False
        else:
            total_alignment += alignment[chain][:, :]

    if total_alignment:
        return SDUtils.process_alignment(total_alignment)
    else:
        logger.error('%s - No sequences were found!' % multi_chain_alignment.__name__)
        raise SDUtils.DesignError('%s - No sequences were found!' % multi_chain_alignment.__name__)


def analyze_mutations(des_dir, mutated_sequences, residues=None, print_results=False):  # DEPRECIATED
    """Use the JSD to look at the mutation probabilities of a design. Combines chains after Multiple Sequence Alignment

    Args:
        des_dir (DesignDirectory): DesignDirectory Object
        mutated_sequences (dict): {chain: {name: sequence, ...}
    Keyword Args:
        residues=None (list): [13, 16, 40, 88, 129, 130, 131, 190, 300] - A list of residue numbers
        print_results=False (bool): Whether to print the results to standard out
    Returns:
        final_mutation_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    """
    alignment = {chain: create_bio_msa(mutated_sequences[chain]) for chain in mutated_sequences}

    # Combine alignments for all chains from design file Ex: A: 1-102, B: 130. Alignment: 1-232
    first = True
    total_alignment = None
    for chain in alignment:
        if first:
            total_alignment = alignment[chain][:, :]
            first = False
        else:
            total_alignment += alignment[chain][:, :]

    if total_alignment:
        alignment_dict = SDUtils.process_alignment(total_alignment)
    else:
        logger.error('%s: No sequences were found!' % des_dir.path)
        raise SDUtils.DesignError('No sequences were found in %s' % des_dir.path)

    # Retrieve design information
    if residues:
        keep_residues = residues
    else:
        design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
        keep_residues = SDUtils.get_interface_residues(design_flags, zero=True)

    mutation_frequencies = remove_non_mutations(alignment_dict['counts'], keep_residues)
    ranked_frequencies = SDUtils.rank_possibilities(mutation_frequencies)

    # Calculate Jensen Shannon Divergence from DSSM using the occurrence data in col 2 and design Mutations
    dssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    design_divergence = pos_specific_jsd(mutation_frequencies, dssm)

    interface_bkgd = SDUtils.get_db_aa_frequencies(db)
    interface_divergence = SDUtils.compute_jsd(mutation_frequencies, interface_bkgd)

    if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):  # TODO Wrap into DesignDirectory object
        pssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.msa_pssm))
    else:
        pssm = SDUtils.parse_pssm(os.path.join(des_dir.building_blocks, PUtils.msa_pssm))
    evolution_divergence = pos_specific_jsd(mutation_frequencies, pssm)

    final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_frequencies, evolution_divergence,
                                              interface_divergence, design_divergence)

    if print_results:
        logger('Mutation Frequencies:', mutation_frequencies)
        logger('Ranked Frequencies:', ranked_frequencies)
        logger('Design Divergence values:', design_divergence)
        logger('Evolution Divergence values:', evolution_divergence)

    return final_mutation_dict


def df_filter_index_by_value(df, **kwargs):
    """Take a df and retrieve the indices which have column values greater_equal/less_equal to a value depending
    on whether the column should be sorted max/min
    Args:
        df (pandas.DataFrame): DataFrame to filter indices on
    Keyword Args:
        kwargs (dict): {column: {'direction': 'min', 'value': 0.3, 'idx': ['0001', '0002', ...]}, ...}
    """
    for idx in kwargs:
        if kwargs[idx]['direction'] == 'max':
            kwargs[idx]['idx'] = df[df[idx] >= kwargs[idx]['value']].index.to_list()
        if kwargs[idx]['direction'] == 'min':
            kwargs[idx]['idx'] = df[df[idx] <= kwargs[idx]['value']].index.to_list()

    return kwargs


def filter_pose(df_file, filters, weights, num_designs=1, consensus=False, filter_file=PUtils.filter_and_sort):
    idx = pd.IndexSlice
    df = pd.read_csv(df_file, index_col=0, header=[0, 1, 2])
    filter_df = pd.read_csv(filter_file, index_col=0)
    logger.info('Number of starting designs = %d' % len(df))
    logger.info('Using filter parameters: %s' % str(filters))

    # design_requirements = {'percent_int_area_polar': 0.4, 'buns_per_ang': 0.002}
    # crystal_means = {'int_area_total': 570, 'shape_complementarity': 0.63, 'number_hbonds': 5}
    # sort = {'protocol_energy_distance_sum': 0.25, 'shape_complementarity': 0.25, 'observed_evolution': 0.25,
    #         'int_composition_diff': 0.25}
    weights_s = pd.Series(weights)

    # When df is not ranked by percentage
    _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': filters[metric]}
                for metric in filters}

    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    _df = df.loc[:, idx['pose', df.columns.get_level_values(1) != 'std', :]].droplevel(1, axis=1).droplevel(0, axis=1)
    # Filter the DataFrame to include only those values which are lower or higher than the specified filter
    filters_with_idx = df_filter_index_by_value(_df, **_filters)
    filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
    logger.info('\n%s' % '\n'.join('Number of designs passing \'%s\' filter = %d' %
                                   (metric, len(filtered_indices[metric])) for metric in filtered_indices))
    final_indices = SDUtils.index_intersection(filtered_indices)
    # When df IS ranked by percentage
    # bottom_percent = (num_designs / len(df))
    # top_percent = 1 - bottom_percent
    # min_max_to_top_bottom = {'min': bottom_percent, 'max': top_percent}
    # _filters = {metric: {'direction': filter_df.loc['direction', metric],
    #                      'value': min_max_to_top_bottom[filter_df.loc['direction', metric]]} for metric in filters}

    # _sort = {metric: {'direction': filter_df.loc['direction', metric],
    #                   'value': min_max_to_top_bottom[filter_df.loc['direction', metric]]} for metric in sort_s.index}
    # filters_with_idx = df_filter_index_by_value(ranked_df, **_sort)

    if consensus:
        protocol_df = df.loc[:, idx['consensus', ['mean', 'stats'], :]].droplevel(1, axis=1)
        #     df.loc[:, idx[df.columns.get_level_values(0) != 'pose', ['mean', 'stats'], :]].droplevel(1, axis=1)
        # stats_protocol_df = \
        #     df.loc[:, idx[df.columns.get_level_values(0) != 'pose', df.columns.get_level_values(1) == 'stats',
        #     :]].droplevel(1, axis=1)
        # design_protocols_df = pd.merge(protocol_df, stats_protocol_df, left_index=True, right_index=True)
        _df = pd.merge(protocol_df.loc[:, idx['consensus', :]],
                       df.droplevel(0, axis=1).loc[:, idx[:, 'percent_fragment']],
                       left_index=True, right_index=True).droplevel(0, axis=1)
    # filtered_indices = {}
    logger.info('Using weighting parameters: %s' % str(weights))
    # for metric in filters:
    #     filtered_indices[metric] = set(df[df.droplevel(0, axis=1)[metric] >= filters[metric]].index.to_list())
    #     logger.info('Number of designs passing %s = %d' % (metric, len(filtered_indices[metric])))

    logger.info('Final set of designs passing all metric filters has %d members' % len(final_indices))
    _df = _df.loc[final_indices, :]
    ranked_df = _df.rank(method='min', pct=True, )
    # need {column: {'direction': 'max', 'value': 0.5, 'idx': []}, ...}

    # only used to check out the number of designs in each filter
    # for _filter in crystal_filters_with_idx:
    #     print('%s designs = %d' % (_filter, len(crystal_filters_with_idx[_filter]['idx'])))

    # {column: {'direction': 'min', 'value': 0.3, 'idx': ['0001', '0002', ...]}, ...}

    # display(ranked_df[weights_s.index.to_list()] * weights_s)
    design_scores_s = (ranked_df[weights_s.index.to_list()] * weights_s).sum(axis=1).sort_values(ascending=False)
    design_list = design_scores_s.index.to_list()[:num_designs]
    logger.info('%d poses were selected:\n%s' % (num_designs, '\n'.join(design_list)))

    return design_list


@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def select_sequences_s(des_dir, number=1, debug=False):
    return select_sequences(des_dir, number=number, debug=debug)


def select_sequences_mp(des_dir, number=1, debug=False):
    try:
        pose = select_sequences(des_dir, number=number, debug=debug)
        return pose, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, (des_dir.path, e)


def select_sequences(des_dir, number=1, debug=False):
    """From a design directory find the sequences with the most neighbors to select for further characterization

    Args:
        des_dir (DesignDirectory)
    Keyword Args:
        number=1 (int): The number of sequences to consider for each design
        debug=False (bool): Whether or not to debug
    Returns:
        (list): Containing tuples with (DesignDirectory.path, design index) for each sequence found
    """
    desired_protocol = 'combo_profile'
    # Log output
    if debug:
        global logger
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))

    # Load relevant data from the design directory
    trajectory_file = glob(os.path.join(des_dir.all_scores, '%s_Trajectories.csv' % str(des_dir)))
    assert len(trajectory_file) == 1, 'Couldn\'t find files for %s' % \
                                      os.path.join(des_dir.all_scores, '%s_Trajectories.csv' % str(des_dir))
    trajectory_df = pd.read_csv(trajectory_file[0], index_col=0, header=[0])  # , 1, 2]

    sequences_pickle = glob(os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir)))
    assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
                                       os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))

    # {chain: {name: sequence, ...}, ...}
    all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
    # all_design_sequences.pop(PUtils.stage[1])  # Remove refine from sequences, not in trajectory_df so unnecessary
    chains = list(all_design_sequences.keys())
    # designs = trajectory_df.index.to_list()  # can't use with the mean and std statistics
    # designs = list(all_design_sequences[chains[0]].keys())
    designs = trajectory_df[trajectory_df['protocol'] == desired_protocol].index.to_list()
    concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains]) for design in designs]
    logger.debug(chains)
    logger.debug(concatenated_sequences)

    # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, SDUtils.sequence_difference)
    # Using concatenated sequences makes the values incredibly similar and inflated as most residues are the same
    # doing min/max normalization to see variation
    pairwise_sequence_diff_l = [SDUtils.sequence_difference(*seq_pair)
                                for seq_pair in combinations(concatenated_sequences, 2)]
    pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
    _min = min(pairwise_sequence_diff_l)
    # _max = max(pairwise_sequence_diff_l)
    pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
    # logger.info(pairwise_sequence_diff_l)

    # PCA analysis of distances
    pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
    for k, dist in enumerate(pairwise_sequence_diff_np):
        i, j = SDUtils.condensed_to_square(k, len(designs))
        pairwise_sequence_diff_mat[i, j] = dist
    pairwise_sequence_diff_mat = SDUtils.sym(pairwise_sequence_diff_mat)

    pairwise_sequence_diff_mat = StandardScaler().fit_transform(pairwise_sequence_diff_mat)
    seq_pca = PCA(PUtils.variance)
    seq_pc_np = seq_pca.fit_transform(pairwise_sequence_diff_mat)
    seq_pca_distance_vector = pdist(seq_pc_np)
    # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
    epsilon = seq_pca_distance_vector.mean() * 0.5
    logger.info('Finding maximum neighbors within distance of %f' % epsilon)

    # logger.info(pairwise_sequence_diff_np)
    # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
    # epsilon = math.sqrt(seq_pc_np.myean()) * 0.5
    # epsilon = math.sqrt(pairwise_sequence_diff_np.mean()) * 0.5

    # Find the nearest neighbors for the pairwise distance matrix using the X*X^T (PCA) matrix, linear transform
    seq_neighbors = BallTree(seq_pc_np)
    seq_neighbor_counts = seq_neighbors.query_radius(seq_pc_np, epsilon, count_only=True)  # , sort_results=True)
    top_count, top_idx = 0, None
    for count in seq_neighbor_counts:  # idx, enumerate()
        if count > top_count:
            top_count = count

    sorted_seqs = sorted(seq_neighbor_counts, reverse=True)
    top_neighbor_counts = sorted(set(sorted_seqs[:number]), reverse=True)

    # Find only the designs which match the top x (number) of neighbor counts
    final_designs = {designs[idx]: num_neighbors for num_neighbors in top_neighbor_counts
                     for idx, count in enumerate(seq_neighbor_counts) if count == num_neighbors}
    logger.info('The final sequence(s) and file(s):\nNeighbors\tDesign\n%s'
                # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.index_offset,
                % '\n'.join('\t%d\t%s' % (neighbors, os.path.join(des_dir.design_pdbs, des))
                            for des, neighbors in final_designs.items()))

    # logger.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(des_dir.design_pdbs, seq))
    #                                                         for i, seq in enumerate(final_designs, 1)))

    # Compute the highest density cluster using DBSCAN algorithm
    # seq_cluster = DBSCAN(eps=epsilon)
    # seq_cluster.fit(pairwise_sequence_diff_np)
    #
    # seq_pc_df = pd.DataFrame(seq_pc, index=designs,
    #                          columns=['pc' + str(x + SDUtils.index_offset) for x in range(len(seq_pca.components_))])
    # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

    # If final designs contains more sequences than specified, find the one with the lowest energy
    if len(final_designs) > number:
        energy_s = trajectory_df.loc[final_designs, 'int_energy_res_summary_delta']  # includes solvation energy
        try:
            energy_s = pd.Series(energy_s)
        except ValueError:
            raise SDUtils.DesignError('no dataframe')
        energy_s.sort_values(inplace=True)
        final_seqs = zip(repeat(des_dir), energy_s.iloc[:number].index.to_list())  # , :].index.to_list()) - index_offset
    else:
        final_seqs = zip(repeat(des_dir), final_designs.keys())

    return final_seqs


def calculate_sequence_metrics(des_dir, alignment_dict, residues=None):
    if residues:
        keep_residues = residues
        mutation_probabilities = remove_non_mutations(alignment_dict['counts'], keep_residues)
    else:
        mutation_probabilities = alignment_dict['counts']
    #     design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
    #     keep_residues = SDUtils.get_interface_residues(design_flags, zero=True)

    ranked_frequencies = SDUtils.rank_possibilities(mutation_probabilities)

    # Calculate Jensen Shannon Divergence from DSSM using the occurrence data in col 2 and design Mutations
    dssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    residue_divergence_values = pos_specific_jsd(mutation_probabilities, dssm)

    interface_bkgd = SDUtils.get_db_aa_frequencies(db)
    interface_divergence_values = SDUtils.compute_jsd(mutation_probabilities, interface_bkgd)

    if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):  # TODO Wrap into DesignDirectory object
        pssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.msa_pssm))
    else:
        pssm = SDUtils.parse_pssm(os.path.join(des_dir.building_blocks, PUtils.msa_pssm))
    evolution_divergence_values = pos_specific_jsd(mutation_probabilities, pssm)

    final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_probabilities, evolution_divergence_values,
                                              interface_divergence_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nAnalyze mutations compared to a wild_type protein. Requires a '
                                                 'directory with \'mutated\' PDB files and a wild-type PDB reference.'
                                                 % __name__)
    parser.add_argument('-d', '--directory', type=str, help='Where is the design PDB directory located?',
                        default=os.getcwd())
    parser.add_argument('-w', '--wildtype', type=str, help='Where is the wild-type PDB located?', default=None)
    parser.add_argument('-p', '--print', action='store_true', help='Print the output the the console? Default=False')
    parser.add_argument('-s', '--score', type=str, help='Where is the score file located?', default=None)
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default=False')

    args = parser.parse_args()
    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name='main', level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name='main', level=2)

    logger.info('Starting %s with options:\n%s' %
                (__name__, '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    design_directory = SDUtils.DesignDirectory(args.directory)

    logger.warning('If you are running into issues with locating files, the problem is not you, it is me. '
                   'I have limited capacity to locate specific files given the scope of my creation.')
    if os.path.basename(args.directory).startswith('tx_'):
        logger.info('Design directory specified, using standard method and disregarding additional inputs '
                    '(-s, -score) and (-w, --wildtype).')
        analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                          print_results=args.print)
    else:
        if args.directory and args.wildtype and args.score:
            path_object = SDUtils.set_up_pseudo_design_dir(args.wildtype, args.directory, args.score)
            analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                              print_results=args.print)
        else:
            logger.critical('Must pass all three, wildtype, directory, and score if using non-standard %s '
                            'directory structure' % PUtils.program_name)
            sys.exit()
