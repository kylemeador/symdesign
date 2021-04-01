import argparse
import math
import os
import sys
from itertools import combinations, repeat

import numpy as np
import pandas as pd
# try:
#     from Bio.Alphabet import generic_protein  # , IUPAC
# except ImportError:
# from Bio.Align import substitution_matrices
# generic_protein = None
from Bio.SeqUtils import IUPACData
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

import PathUtils as PUtils
from DesignMetrics import master_metrics
from SymDesignUtils import DesignError, handle_design_errors, index_offset, start_log, unpickle, clean_dictionary,\
    index_intersection, condensed_to_square, sym
from PDB import PDB
from Query.Flags import query_user_for_metrics
from SequenceProfile import remove_non_mutations, pos_specific_jsd, weave_mutation_dict, \
    generate_mutations, make_mutations, SequenceProfile, create_bio_msa, sequence_difference

# Globals
logger = start_log(name=__name__)  # was from SDUtils logger, but moved here per standard suggestion
db = PUtils.biological_fragmentDB


#####################
# Sequence handling
#####################


# def write_multi_line_fasta_file(sequences, name, path=os.getcwd()):  # REDUNDANT DEPRECIATED
#     """Write a multi-line fasta file from a dictionary where the keys are >headers and values are sequences
#
#     Args:
#         sequences (dict): {'my_protein': 'MSGFGHKLGNLIGV...', ...}
#         name (str): The name of the file to output
#     Keyword Args:
#         path=os.getcwd() (str): The location on disk to output file
#     Returns:
#         (str): The name of the output file
#     """
#     file_name = os.path.join(path, name)
#     with open(file_name, 'r') as f:
#         # f.write('>%s\n' % seq)
#         f.write('\n'.join('>%s\n%s' % (seq_name, sequences[seq_name]) for seq_name in sequences))
#
#     return file_name


# def parse_mutations(mutation_list):  # UNUSED
#     if isinstance(mutation_list, str):
#         mutation_list = mutation_list.split(', ')
#
#     # Takes a list of mutations in the form A37K and parses the index (37), the FROM aa (A), and the TO aa (K)
#     # output looks like {37: ('A', 'K'), 440: ('K', 'Y'), ...}
#     mutation_dict = {}
#     for mutation in mutation_list:
#         to_letter = mutation[-1]
#         from_letter = mutation[0]
#         index = int(mutation[1:-1])
#         mutation_dict[index] = (from_letter, to_letter)
#
#     return mutation_dict


# def analyze_mutations(des_dir, mutated_sequences, residues=None, print_results=False):  # DEPRECIATED
#     """Use the JSD to look at the mutation probabilities of a design. Combines chains after Multiple Sequence Alignment
#
#     Args:
#         des_dir (DesignDirectory): DesignDirectory Object
#         mutated_sequences (dict): {chain: {name: sequence, ...}
#     Keyword Args:
#         residues=None (list): [13, 16, 40, 88, 129, 130, 131, 190, 300] - A list of residue numbers
#         print_results=False (bool): Whether to print the results to standard out
#     Returns:
#         final_mutation_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
#     """
#     alignment = {chain: create_bio_msa(mutated_sequences[chain]) for chain in mutated_sequences}
#
#     # Combine alignments for all chains from design file Ex: A: 1-102, B: 130. Alignment: 1-232
#     first = True
#     total_alignment = None
#     for chain in alignment:
#         if first:
#             total_alignment = alignment[chain][:, :]
#             first = False
#         else:
#             total_alignment += alignment[chain][:, :]
#
#     if total_alignment:
#         alignment_dict = SDUtils.process_alignment(total_alignment)
#     else:
#         logger.error('%s: No sequences were found!' % des_dir.path)
#         raise SDUtils.DesignError('No sequences were found in %s' % des_dir.path)
#
#     # Retrieve design information
#     if residues:
#         keep_residues = residues
#     else:
#         design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
#         keep_residues = SDUtils.get_interface_residues(design_flags, zero=True)
#
#     mutation_frequencies = remove_non_mutations(alignment_dict['counts'], keep_residues)
#     ranked_frequencies = SDUtils.rank_possibilities(mutation_frequencies)
#
#     # Calculate Jensen Shannon Divergence from DSSM using the occurrence data in col 2 and design Mutations
#     dssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
#     design_divergence = pos_specific_jsd(mutation_frequencies, dssm)
#
#     interface_bkgd = SequenceProfile.get_db_aa_frequencies(db)
#     interface_divergence = SDUtils.compute_jsd(mutation_frequencies, interface_bkgd)
#
#     if os.path.exists(os.path.join(des_dir.path, PUtils.pssm)):
#         pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.pssm))
#     else:
#         pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.composition, PUtils.pssm))
#     evolution_divergence = pos_specific_jsd(mutation_frequencies, pssm)
#
#     final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_frequencies, evolution_divergence,
#                                               interface_divergence, design_divergence)
#
#     if print_results:
#         logger('Mutation Frequencies:', mutation_frequencies)
#         logger('Ranked Frequencies:', ranked_frequencies)
#         logger('Design Divergence values:', design_divergence)
#         logger('Evolution Divergence values:', evolution_divergence)
#
#     return final_mutation_dict


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


def filter_pose(df_file, filter=None, weight=None, consensus=False):
    """Return the indices from a dataframe that pass an set of filters (optional) and are ranked according to weight as
    specified by user values.

    Args:
        df_file (str): DataFrame to filter/weight indices
    Keyword Args:
        filter=False (bool): Whether filters are going to remove viable candidates
        weight=False (bool): Whether weights are going to select the poses
        consensus=False (bool): Whether consensus designs should be chosen
    Returns:
        (pandas.DataFrame): The dataframe of selected designs based on the provided filters and weights
    """
    idx_slice = pd.IndexSlice
    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    df = pd.read_csv(df_file, index_col=0, header=[0, 1, 2])
    logger.info('Number of starting designs = %d' % len(df))
    _df = df.loc[:, idx_slice['pose',
                              df.columns.get_level_values(1) != 'std', :]].droplevel(1, axis=1).droplevel(0, axis=1)

    filter_df = pd.read_csv(master_metrics, index_col=0)
    if filter:
        available_filters = _df.columns.to_list()
        filters = query_user_for_metrics(available_filters, mode='filter', level='design')
        logger.info('Using filter parameters: %s' % str(filters))

        # When df is not ranked by percentage
        _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
                    for metric, value in filters.items()}

        # Filter the DataFrame to include only those values which are le/ge the specified filter
        filters_with_idx = df_filter_index_by_value(_df, **_filters)
        filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
        logger.info('Number of designs passing filters:\n\t%s'
                    % '\n\t'.join('%6d - %s' % (len(indices), metric) for metric, indices in filtered_indices.items()))
        final_indices = index_intersection(filtered_indices)
        logger.info('Final set of designs passing all filters has %d members' % len(final_indices))
        if len(final_indices) == 0:
            raise DesignError('There are no poses left after filtering! Try choosing less stringent values or make '
                              'better designs!')
        _df = _df.loc[final_indices, :]

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
        protocol_df = df.loc[:, idx_slice['consensus', ['mean', 'stats'], :]].droplevel(1, axis=1)
        #     df.loc[:, idx_slice[df.columns.get_level_values(0) != 'pose', ['mean', 'stats'], :]].droplevel(1, axis=1)
        # stats_protocol_df = \
        #     df.loc[:, idx_slice[df.columns.get_level_values(0) != 'pose', df.columns.get_level_values(1) == 'stats',
        #     :]].droplevel(1, axis=1)
        # design_protocols_df = pd.merge(protocol_df, stats_protocol_df, left_index=True, right_index=True)
        # TODO make more robust sampling from specific protocol
        _df = pd.merge(protocol_df.loc[:, idx_slice['consensus', :]],
                       df.droplevel(0, axis=1).loc[:, idx_slice[:, 'percent_fragment']],
                       left_index=True, right_index=True).droplevel(0, axis=1)
    # filtered_indices = {}

    # for metric in filters:
    #     filtered_indices[metric] = set(df[df.droplevel(0, axis=1)[metric] >= filters[metric]].index.to_list())
    #     logger.info('Number of designs passing %s = %d' % (metric, len(filtered_indices[metric])))
    # ranked_df = _df.rank(method='min', pct=True, )  # default is to rank lower values as closer to 1
    # need {column: {'direction': 'max', 'value': 0.5, 'idx_slice': []}, ...}

    # only used to check out the number of designs in each filter
    # for _filter in crystal_filters_with_idx:
    #     print('%s designs = %d' % (_filter, len(crystal_filters_with_idx[_filter]['idx_slice'])))

    # {column: {'direction': 'min', 'value': 0.3, 'idx_slice': ['0001', '0002', ...]}, ...}
    if weight:
        # display(ranked_df[weights_s.index.to_list()] * weights_s)
        available_metrics = _df.columns.to_list()
        weights = query_user_for_metrics(available_metrics, mode='weight', level='design')
        logger.info('Using weighting parameters: %s' % str(weights))
        _weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
                    for metric, value in weights.items()}
        weight_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
        # weights_s = pd.Series(weights)
        weight_score_s_d = {}
        for metric in _weights:
            weight_score_s_d[metric] = \
                _df[metric].rank(ascending=weight_direction[_weights[metric]['direction']],
                                 method=_weights[metric]['direction'], pct=True) * _weights[metric]['value']

        design_score_df = pd.concat([weight_score_s_d[weight] for weight in weights], axis=1)
        weighted_s = design_score_df.sum(axis=1).sort_values(ascending=False)
        weighted_df = pd.concat([weighted_s], keys=[('pose', 'sum', 'selection_weight')])
        final_df = pd.merge(weighted_df, df, left_index=True, right_index=True)
        # designs = weighted_s.index.to_list()
    else:
        final_df = df.loc[_df.index.to_list(), :]
        # designs = _df.index.to_list()
    # these will be sorted by the largest value to the smallest
    # design_scores_s = (ranked_df[weights_s.index.to_list()] * weights_s).sum(axis=1).sort_values(ascending=False)
    # designs = design_scores_s.index.to_list()
    # designs = design_scores_s.index.to_list()[:num_designs]
    # return designs
    return final_df


# @handle_design_errors(errors=(DesignError, AssertionError))
# def select_sequences_s(des_dir, weights=None, number=1, desired_protocol=None, debug=False):
#     return select_sequences(des_dir, weights=weights, number=number, desired_protocol=desired_protocol, debug=debug)
#
#
# def select_sequences_mp(des_dir, weights=None, number=1, desired_protocol=None, debug=False):
#     try:
#         pose = select_sequences(des_dir, weights=weights, number=number, desired_protocol=desired_protocol, debug=debug)
#         return pose
#     except (DesignError, AssertionError) as e:
#         return e


@handle_design_errors(errors=(DesignError, AssertionError))
def select_sequences(des_dir, weights=None, number=1, desired_protocol=None):
    """From a single design, select sequences for further characterization. If weights, then using weights the user can
     prioritize sequences, otherwise the sequence with the most neighbors will be selected

    Args:
        des_dir (DesignDirectory)
    Keyword Args:
        weights=None (iter): The weights to use in sequence selection
        number=1 (int): The number of sequences to consider for each design
        debug=False (bool): Whether or not to debug
    Returns:
        (list[tuple[DesignDirectory, str]]): Containing the selected sequences found
    """
    # Load relevant data from the design directory
    trajectory_df = pd.read_csv(des_dir.trajectories, index_col=0, header=[0])
    trajectory_df.dropna(inplace=True)
    # trajectory_df.dropna('protocol', inplace=True)
    # designs = trajectory_df.index.to_list()  # can't use with the mean and std statistics
    # designs = list(all_design_sequences[chains[0]].keys())
    logger.info('Number of starting trajectories = %d' % len(trajectory_df))

    if weights:
        filter_df = pd.read_csv(master_metrics, index_col=0)
        # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
        # _df = trajectory_df.loc[final_indices, :]
        _df = trajectory_df
        logger.info('Using weighting parameters: %s' % str(weights))
        _weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': weights[metric]}
                    for metric in weights}
        weight_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
        # weights_s = pd.Series(weights)
        weight_score_s_d = {}
        for metric in _weights:
            weight_score_s_d[metric] = _df[metric].rank(ascending=weight_direction[_weights[metric]['direction']],
                                                        method=_weights[metric]['direction'], pct=True) \
                                       * _weights[metric]['value']

        design_score_df = pd.concat([weight_score_s_d[weight] for weight in weights], axis=1)
        design_list = design_score_df.sum(axis=1).sort_values(ascending=False).index.to_list()
        logger.info('Final ranking of trajectories:\n%s' % ', '.join(pose for pose in design_list))

        return zip(repeat(des_dir), design_list[:number])
    else:
        if desired_protocol:
            unique_protocols = trajectory_df['protocol'].unique().tolist()
            while True:
                desired_protocol = input('Do you have a protocol that you prefer to pull your designs from? Possible '
                                         'protocols include:\n%s' % ', '.join(unique_protocols))
                if desired_protocol in unique_protocols:
                    break
                else:
                    print('%s is not a valid protocol, try again.' % desired_protocol)

            designs = trajectory_df[trajectory_df['protocol'] == desired_protocol].index.to_list()
        else:
            designs = trajectory_df.index.to_list()
        # sequences_pickle = glob(os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir)))
        # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
        #                                    os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))
        #
        # all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
        # {chain: {name: sequence, ...}, ...}
        all_design_sequences = unpickle(des_dir.design_sequences)
        # all_design_sequences.pop(PUtils.stage[1])  # Remove refine from sequences, not in trajectory_df so unnecessary
        chains = list(all_design_sequences.keys())
        concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains])
                                  for design in designs]
        logger.debug(chains)
        logger.debug(concatenated_sequences)

        # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, sequence_difference)
        # Using concatenated sequences makes the values incredibly similar and inflated as most residues are the same
        # doing min/max normalization to see variation
        pairwise_sequence_diff_l = [sequence_difference(*seq_pair)
                                    for seq_pair in combinations(concatenated_sequences, 2)]
        pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
        _min = min(pairwise_sequence_diff_l)
        # _max = max(pairwise_sequence_diff_l)
        pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
        # logger.info(pairwise_sequence_diff_l)

        # PCA analysis of distances
        pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
        for k, dist in enumerate(pairwise_sequence_diff_np):
            i, j = condensed_to_square(k, len(designs))
            pairwise_sequence_diff_mat[i, j] = dist
        pairwise_sequence_diff_mat = sym(pairwise_sequence_diff_mat)

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
        seq_neighbors = BallTree(seq_pc_np)  # Todo make brute force or automatic, not BallTree
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
                    % '\n'.join('\t%d\t%s' % (neighbors, os.path.join(des_dir.designs, design))
                                for design, neighbors in final_designs.items()))

        # logger.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(des_dir.designs, seq))
        #                                                         for i, seq in enumerate(final_designs, 1)))

        # Compute the highest density cluster using DBSCAN algorithm
        # seq_cluster = DBSCAN(eps=epsilon)
        # seq_cluster.fit(pairwise_sequence_diff_np)
        #
        # seq_pc_df = pd.DataFrame(seq_pc, index=designs,
        #                       columns=['pc' + str(x + SDUtils.index_offset) for x in range(len(seq_pca.components_))])
        # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

        # If final designs contains more sequences than specified, find the one with the lowest energy
        if len(final_designs) > number:
            energy_s = trajectory_df.loc[final_designs, 'int_energy_res_summary_delta']  # includes solvation energy
            try:
                energy_s = pd.Series(energy_s)
            except ValueError:
                raise DesignError('no dataframe')
            energy_s.sort_values(inplace=True)
            final_seqs = zip(repeat(des_dir), energy_s.iloc[:number].index.to_list())
        else:
            final_seqs = zip(repeat(des_dir), final_designs.keys())

        return list(final_seqs)


def calculate_sequence_metrics(des_dir, alignment_dict, residues=None):  # Unused Todo SequenceProfile.py
    if residues:
        keep_residues = residues
        mutation_probabilities = remove_non_mutations(alignment_dict['counts'], keep_residues)
    else:
        mutation_probabilities = alignment_dict['counts']
    #     design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
    #     keep_residues = SDUtils.get_interface_residues(design_flags, zero=True)

    ranked_frequencies = rank_possibilities(mutation_probabilities)

    # Calculate Jensen Shannon Divergence from DSSM using the occurrence data in col 2 and design Mutations
    dssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    residue_divergence_values = pos_specific_jsd(mutation_probabilities, dssm)

    interface_bkgd = SequenceProfile.get_db_aa_frequencies(db)
    interface_divergence_values = compute_jsd(mutation_probabilities, interface_bkgd)

    if os.path.exists(os.path.join(des_dir.path, PUtils.pssm)):
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.pssm))
    else:
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.composition, PUtils.pssm))
    evolution_divergence_values = pos_specific_jsd(mutation_probabilities, pssm)

    final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_probabilities, evolution_divergence_values,
                                              interface_divergence_values)


def mutate_wildtype_sequences(sequence_dir_files, wild_type_file):
    """Take a directory with PDB files and compare to a Wild-type PDB"""
    wt_seq_dict = get_pdb_sequences(wild_type_file)
    return generate_sequences(wt_seq_dict, generate_all_design_mutations(sequence_dir_files, wild_type_file))


def generate_all_design_mutations(all_design_files, wild_type_file, pose_num=False):  # Todo DEPRECIATE
    """From a list of PDB's and a wild-type PDB, generate a list of 'A5K' style mutations

    Args:
        all_design_files (list): PDB files on disk to extract sequence info and compare
        wild_type_file (str): PDB file on disk which contains a reference sequence
    Returns:
        mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    """
    pdb_dict = {'ref': PDB.from_file(wild_type_file)}
    for file_name in all_design_files:
        pdb = PDB.from_file(file_name, log=start_log(handler=3), entities=False)
        pdb.name = os.path.splitext(os.path.basename(file_name))[0]
        pdb_dict[pdb.name] = pdb

    return extract_sequence_from_pdb(pdb_dict, mutation=True, pose_num=pose_num)  # , offset=False)


def pdb_to_pose_num(reference):
    """Take a dictionary with chain name as keys and return the length of Pose numbering offset

    Args:
        reference (dict(iter)): {'A': 'MSGKLDA...', ...} or {'A': {1: 'A', 2: 'S', ...}, ...}
    Order of dictionary must maintain chain order, so 'A', 'B', 'C'. Python 3.6+ should be used
    Returns:
        (dict): {'A': 0, 'B': 123, ...}
    """
    offset = {}
    # prior_chain = None
    prior_chains_len = 0
    for i, chain in enumerate(reference):
        if i > 0:
            prior_chains_len += len(reference[prior_chain])
        offset[chain] = prior_chains_len
        # insert function here? Make this a decorator!?
        prior_chain = chain

    return offset


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
            reference_seq_dict[chain] = reference.atom_sequences[chain]
            fail = None
            # reference_seq_dict[chain], fail = extract_aa_seq(reference, aa_code, seq_source, chain)
            if fail:
                fail_ref.append((reference, chain, fail))

        if fail_ref:
            logger.error('Ran into following errors generating mutational analysis reference:\n%s' % str(fail_ref))

    if seq_source == 'compare':
        mutation = True

    def handle_extraction(pdb_code, _pdb, _aa, _source, _chain):
        if _source == 'compare':
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, 'atom', _chain)
            sequence2, failures2 = extract_aa_seq(_pdb, _aa, 'seqres', _chain)
            _offset = True
        else:
            sequence1 = _pdb.atom_sequences[_chain]
            failures1 = None
            # sequence1, failures1 = extract_aa_seq(_pdb, _aa, _source, _chain)
            sequence2 = reference_seq_dict[_chain]
            sequence_dict[pdb_code][_chain] = sequence1
            _offset = False
        if mutation:
            mutation_dict[pdb_code][_chain] = generate_mutations(sequence1, sequence2, offset=_offset)
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
            # name, PDB obj, 1 or 3 letter code, ATOM or SEQRES, chain_id
            handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain)
        # else:
        #     handle_extraction(pdb, pdb_class_dict[pdb], aa_code, seq_source, chain_dict[pdb])

    if outpath:
        sequences = {}
        for pdb in sequence_dict:
            for chain in sequence_dict[pdb]:
                sequences[pdb + '_' + chain] = sequence_dict[pdb][chain]
        # filepath = write_multi_line_fasta_file(sequences, 'sequence_extraction.fasta', path=outpath)
        # logger.info('The following file was written:\n%s' % filepath)
        # logger.info('No file was written:\n%s' % filepath)

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
    Returns:
        all_sequences (dict): {name: sequence, ...}
    """
    return {pdb: make_mutations(wild_type, mutation_dict[pdb], find_orf=not aligned) for pdb in mutation_dict}


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


def get_pdb_sequences(pdb, chain=None, source='atom'):
    """Return all sequences or those specified by a chain from a PDB file

    Args:
        pdb (str or PDB): Location on disk of a reference .pdb file or PDB object
    Keyword Args:
        chain=None (str): If a particular chain is desired, specify it
        source='atom' (str): One of 'atom' or 'seqres'
    Returns:
        (dict): {chain: sequence, ...}
    """
    print('get_pdb_sequences is using pdb parameter: %s' % pdb)
    if not isinstance(pdb, PDB):
        pdb = PDB.from_file(pdb, log=start_log(handler=3), entities=False)

    if source == 'atom':
        seq_dict = pdb.atom_sequences
    else:
        seq_dict = pdb.reference_sequence
    # for _chain in pdb.chain_id_list:
    #     seq_dict[_chain] =
    if chain:
        seq_dict = clean_dictionary(seq_dict, chain, remove=False)

    return seq_dict


def find_gapped_columns(alignment_dict):  # UNUSED
    target_seq_index = []
    n = 1
    for aa in alignment_dict['meta']['query']:
        if aa != '-':
            target_seq_index.append(n)
        n += 1

    return target_seq_index


def update_alignment_meta(alignment_dict):  # UNUSED UNFINISHED
    all_meta = []
    for alignment in alignment_dict:
        all_meta.append(alignment_dict[alignment]['meta'])

    meta_strings = ['' for i in range(len(next(all_meta)))]
    for meta in all_meta:
        j = 0
        for data in meta:
            meta_strings[j] += meta[data]

    return alignment_dict


def modify_index(count_dict, index_start=0):  # UNUSED NOT Working
    return {i + index_start: count_dict[i] for i in count_dict}


def modify_alignment_dict_index(alignment_dict, index=0):  # UNUSED UNFINISHED
    alignment_dict['counts'] = modify_index(alignment_dict['counts'], index_start=index)
    alignment_dict['rep'] = modify_index(alignment_dict['rep'], index_start=index)

    return alignment_dict


def merge_alignment_dicts(alignment_merge):  # UNUSED UNFINISHED
    length = [0]
    for i, alignment in enumerate(alignment_merge):
        alignment_dict = modify_alignment_dict_index(alignment_merge[alignment], index=length[i])
        length.append(len(alignment_merge[alignment]['meta']['query']))
        merged_alignment_dict = {'meta': update_alignment_meta(alignment)} # alignment_dict
    for alignment in alignment_merge:
        merged_alignment_dict.update(alignment_merge[alignment])

    return merged_alignment_dict


def clean_gapped_columns(alignment_dict, correct_index):  # UNUSED
    """Cleans an alignment dictionary by revising key list with correctly indexed positions. 0 indexed"""
    return {i: alignment_dict[index] for i, index in enumerate(correct_index)}


def weight_sequences(msa_dict, alignment):  # UNUSED
    """Measure diversity/surprise when comparing a single alignment entry to the rest of the alignment

    Operation is: SUM(1 / (column_j_aa_representation * aa_ij_count)) as was described by Heinkoff and Heinkoff, 1994
    Args:
        msa_dict (dict): { 1: {'A': 31, 'C': 0, ...}, 2: {}, ...}
        alignment (biopython.MSA):
    Returns:
        seq_weight_dict (dict): { 1: 2.390, 2: 2.90, 3:5.33, 4: 1.123, ...} - sequence_in_MSA: sequence_weight_factor
    """
    col_tot_aa_count_dict = {}
    for i in range(len(msa_dict)):
        s = 0  # column amino acid representation
        for aa in msa_dict[i]:
            if aa == '-':
                continue
            elif msa_dict[i][aa] > 0:
                s += 1
        col_tot_aa_count_dict[i] = s

    seq_weight_dict = {}
    for k, record in enumerate(alignment):
        s = 0  # "diversity/surprise"
        for j, aa in enumerate(record.seq):
            s += (1 / (col_tot_aa_count_dict[j] * msa_dict[j][aa]))
        seq_weight_dict[k] = s

    return seq_weight_dict


def generate_msa_dictionary(alignment, alphabet=IUPACData.protein_letters, weighted_dict=None, weight=False):
    """Generate an alignment dictinary from a Biopython MultipleSeqAlignment object

    Args:
        alignment (MultipleSeqAlignment): List of SeqRecords
    Keyword Args:
        alphabet=IUPACData.protein_letters (str): 'ACDEFGHIKLMNPQRSTVWY'
        weighted_dict=None (dict): A weighted sequence dictionary with weights for each alignment sequence
        weight=False (bool): If weights should be used to weight the alignment
    Returns:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {0: {'A': 13, 'C': 1, 'D': 23, ...}, 1: {}, ...})
    """
    aligned_seq = str(alignment[0].seq)
    # Add Info to 'meta' record as needed
    alignment_dict = {'meta': {'num_sequences': len(alignment), 'query': aligned_seq.replace('-', ''),
                               'query_with_gaps': aligned_seq}}
    # Populate Counts Dictionary (one-indexed)
    alignment_counts_dict = SequenceProfile.populate_design_dictionary(alignment.get_alignment_length(), alphabet,
                                                                       dtype=int)
    if weight:
        for record in alignment:
            for i, aa in enumerate(record.seq, 1):
                alignment_counts_dict[i][aa] += weighted_dict[i]
    else:
        for record in alignment:
            for i, aa in enumerate(record.seq, 1):
                alignment_counts_dict[i][aa] += 1
    alignment_dict['counts'] = alignment_counts_dict

    return alignment_dict


def add_column_weight(counts_dict, gaps=False):
    """Find total representation for each column in the alignment

    Args:
        counts_dict (dict): {'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...}
    Keyword Args:
        gaps=False (bool): Whether the alignment contains gaps
    Returns:
        counts_dict (dict): {1: 210, 2:211, ...}
    """
    return {idx: sum_column_weight(aa_counts, gaps=gaps) for idx, aa_counts in counts_dict.items()}


def sum_column_weight(column, gaps=False):
    """Sum the column weight for a single alignment dict column

    Args:
        column (dict): {'A': 13, 'C': 1, 'D': 23, ...}
    Keyword Args:
        gaps=False (bool): Whether to count gaps or not
    Returns:
        s (int): Total counts in the alignment
    """
    s = 0
    if gaps:
        for key in column:
            s += column[key]
    else:
        for key in column:
            if key == '-':
                continue
            else:
                s += column[key]

    return s


def msa_to_prob_distribution(alignment_dict):
    """Turn Alignment dictionary into a probability distribution

    Args:
        alignment_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                         'query_with_gaps': 'MGS---THLVLK...'}}
                                'counts': {1: {'A': 13, 'C': 1, 'D': 23, ...}, 2: {}, ...},
                                'rep': {1: 210, 2:211, ...}}
    Returns:
        (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                          'query_with_gaps': 'MGS---THLVLK...'}}
                 'counts': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
                 'rep': {1: 210, 2:211, ...}}
    """
    for residue in alignment_dict['counts']:
        total_weight_in_column = alignment_dict['rep'][residue]
        assert total_weight_in_column != 0, '%s: Processing error... Downstream cannot divide by 0. Position = %s' % \
                                            (msa_to_prob_distribution.__name__, residue)  # Todo correct?
        for aa in alignment_dict['counts'][residue]:
            alignment_dict['counts'][residue][aa] /= total_weight_in_column

    return alignment_dict


def compute_jsd(msa, bgd_freq, jsd_lambda=0.5):
    """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict

    Args:
        msa (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
        bgd_freq (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
    Keyword Args:
        jsd_lambda=0.5 (float): Value bounded between 0 and 1
    Returns:
        divergence (float): 0.732, Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    divergence_dict = {}
    for residue in msa:
        sum_prob1, sum_prob2 = 0, 0
        for aa in IUPACData.protein_letters:
            p = msa[residue][aa]
            q = bgd_freq[aa]
            r = (jsd_lambda * p) + ((1 - jsd_lambda) * q)
            if r == 0:
                continue
            if q != 0:
                prob2 = (q * math.log2(q / r))
                sum_prob2 += prob2
            if p != 0:
                prob1 = (p * math.log2(p / r))
                sum_prob1 += prob1
        divergence = jsd_lambda * sum_prob1 + (1 - jsd_lambda) * sum_prob2
        divergence_dict[residue] = round(divergence, 3)

    return divergence_dict


def weight_gaps(divergence, representation, alignment_length):  # UNUSED
    for i in range(len(divergence)):
        divergence[i] = divergence[i] * representation[i] / alignment_length

    return divergence


def window_score(score_dict, window_len, score_lambda=0.5):  # UNUSED
    """Takes a MSA score dict and transforms so that each position is a weighted average of the surrounding positions.
    Positions with scores less than zero are not changed and are ignored calculation

    Modified from Capra and Singh 2007 code
    Args:
        score_dict (dict):
        window_len (int): Number of residues on either side of the current residue
    Keyword Args:
        lamda=0.5 (float): Float between 0 and 1
    Returns:
        (dict):
    """
    if window_len == 0:
        return score_dict
    else:
        window_scores = {}
        for i in range(len(score_dict) + index_offset):
            s, number_terms = 0, 0
            if i <= window_len:
                for j in range(1, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            elif i + window_len > len(score_dict):
                for j in range(i - window_len, len(score_dict) + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            else:
                for j in range(i - window_len, i + window_len + index_offset):
                    if i != j:
                        number_terms += 1
                        s += score_dict[j]
            window_scores[i] = (1 - score_lambda) * (s / number_terms) + score_lambda * score_dict[i]

        return window_scores


def rank_possibilities(probability_dict):
    """Gather alternative residues and sort them by probability.

    Args:
        probability_dict (dict): {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
    Returns:
         sorted_alternates_dict (dict): {15: ['S', 'A', 'T'], ... }
    """
    sorted_alternates_dict = {}
    for residue in probability_dict:
        residue_probability_list = []
        for aa in probability_dict[residue]:
            if probability_dict[residue][aa] > 0:
                residue_probability_list.append((aa, round(probability_dict[residue][aa], 5)))  # tuple instead of list
        residue_probability_list.sort(key=lambda tup: tup[1], reverse=True)
        # [('S', 0.13190), ('A', 0.0500), ...]
        sorted_alternates_dict[residue] = [aa[0] for aa in residue_probability_list]

    return sorted_alternates_dict


def process_alignment(bio_alignment_object, gaps=False):
    """Take a Biopython MultipleSeqAlignment object and process for residue specific information. One-indexed

    gaps=True treats all column weights the same. This is fairly inaccurate for scoring, so False reflects the
    probability of residue i in the specific column more accurately.
    Args:
        bio_alignment_object (MultipleSeqAlignment): List of SeqRecords
    Keyword Args:
        gaps=False (bool): Whether gaps (-) should be counted in column weights
    Returns:
        probability_dict (dict): {'meta': {'num_sequences': 214, 'query': 'MGSTHLVLK...'
                                  'query_with_gaps': 'MGS---THLVLK...'}}
                                  'counts': {1: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 2: {}, ...},
                                  'rep': {1: 210, 2:211, ...}}
    """
    alignment_dict = generate_msa_dictionary(bio_alignment_object)
    alignment_dict['rep'] = add_column_weight(alignment_dict['counts'], gaps=gaps)
    probability_dict = msa_to_prob_distribution(alignment_dict)

    return probability_dict


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
        return process_alignment(total_alignment)
    else:
        logger.error('%s - No sequences were found!' % multi_chain_alignment.__name__)
        raise DesignError('%s - No sequences were found!' % multi_chain_alignment.__name__)


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
        logger = start_log(name='main', level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name='main', level=2)

    logger.info('Starting %s with options:\n%s' %
                (__name__, '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    design_directory = DesignDirectory.DesignDirectory(args.directory)

    logger.warning('If you are running into issues with locating files, the problem is not you, it is me. '
                   'I have limited capacity to locate specific files given the scope of my creation.')
    if os.path.basename(args.directory).startswith('tx_'):
        logger.info('Design directory specified, using standard method and disregarding additional inputs '
                    '(-s, -score) and (-w, --wildtype).')
        analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                          print_results=True)  # args.print)
    else:
        if args.directory and args.wildtype and args.score:
            path_object = DesignDirectory.set_up_pseudo_design_dir(args.wildtype, args.directory, args.score)
            analyze_mutations(design_directory, mutate_wildtype_sequences(args.directory, args.wildtype),
                              print_results=True)  # args.print)
        else:
            logger.critical('Must pass all three, wildtype, directory, and score if using non-standard %s '
                            'directory structure' % PUtils.program_name)
            exit()


def extract_aa_seq(pdb, aa_code=1, source='atom', chain=0):
    """Extracts amino acid sequence from either ATOM or SEQRES record of PDB object
    Returns:
        (str): Sequence of PDB
    """
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
        while True:  # TODO WTF is this used for
            if chain in pdb.seqres_sequences:
                sequence = pdb.seqres_sequences[chain]
                break
            else:
                if not fail:
                    temp_pdb = PDB.from_file(file=pdb.filepath)
                    fail = True
                else:
                    raise DesignError('Invalid PDB input, no SEQRES record found')
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
        raise DesignError('Invalid sequence input')

    return final_sequence, failures
