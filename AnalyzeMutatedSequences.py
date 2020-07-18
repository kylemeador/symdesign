import sys
import os
import argparse
import math
from glob import glob
from itertools import combinations
import PDB
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

    return SDUtils.extract_sequence_from_pdb(pdb_dict, mutation=True, pose_num=pose_num)  # , offset=False)


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
        mutated_sequences[chain] = SDUtils.make_sequences_from_mutations(wild_type_seq_dict[chain], chain_mutation_dict,
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
        wt_seq_dict[chain], fail = SDUtils.extract_aa_seq(wt_pdb, chain=_chain)
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


@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def select_sequences_s(des_dir, number=1):
    return select_sequences(des_dir, number=number)


def select_sequences_mp(des_dir, number=1):
    try:
        pose = select_sequences(des_dir, number=number)
        return pose, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, (des_dir.path, e)


def select_sequences(des_dir, number=1, debug=False):
    # Log output
    if debug:
        global logger
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))

    # Load relevant data from the design directory
    trajectory_file = glob(os.path.join(des_dir.all_scores, '%s_Trajectories.csv' % str(des_dir)))
    assert len(trajectory_file) == 1, 'Multiples files found for %s' % \
                                      os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))
    trajectory_df = pd.read_csv(trajectory_file[0], index_col=0, header=[0,1,2])

    sequences_pickle = glob(os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir)))
    assert len(sequences_pickle) == 1, 'Multiples files found for %s' % \
                                       os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))

    # {chain: {name: sequence, ...}, ...}
    all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
    # all_design_sequences.pop(PUtils.stage[1])  # Remove refine from sequences, not in trajectory_df so its unnecessary
    chains = list(all_design_sequences.keys())
    designs = trajectory_df.index.to_list()
    # designs = list(all_design_sequences[chains[0]].keys())
    concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains]) for design in designs]
    # concatenated_sequences = {design: ''.join([all_design_sequences[chain][design] for chain in chains])
    #                           for design in designs}
    logger.debug(chains)
    logger.debug(concatenated_sequences)

    # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, SDUtils.sequence_difference)
    # Using concatenated sequences makes the values incredibly similar and inflated as most residues are the same
    # doing min/max normalization to see variation
    pairwise_sequence_diff_l = [SDUtils.sequence_difference(*seq_pair)
                                for seq_pair in combinations(concatenated_sequences, 2)]  # for design in concatenated_sequences], 2))]
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
    epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
    logger.info('Finding maximum neighbors within distance of %f' % epsilon)

    # logger.info(pairwise_sequence_diff_np)
    # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
    # epsilon = math.sqrt(seq_pc_np.mean()) * 0.5
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
    logger.info('The final sequence(s) and file(s):\n%s'
                % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.index_offset,
                                       os.path.join(des_dir.design_pdbs, des))
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
        energy_s = pd.Series()
        # try:
        #     final_designs.pop('refine')
        # except KeyError:
        #     pass
        for design in final_designs:
            energy_s[design] = trajectory_df.loc[design, 'int_energy_res_summary_delta']
        energy_s.sort_values(inplace=True)
        final_seqs = energy_s.index.to_list()[:number]
    else:
        final_seqs = list(final_designs.keys())

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
