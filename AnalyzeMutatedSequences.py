import argparse
import os
import sys
from itertools import combinations, repeat

import numpy as np
import pandas as pd

import DesignDirectory
import SequenceProfile
from PDB import PDB
from PoseProcessing import extract_aa_seq
from SequenceProfile import remove_non_mutations, pos_specific_jsd, weave_mutation_dict, \
    generate_mutations, index_offset, make_mutations
from SymDesignUtils import logger

try:
    from Bio.SubsMat import MatrixInfo as matlist
    from Bio.Alphabet import generic_protein
except ImportError:
    from Bio.Align.substitution_matrices import MatrixInfo as matlist
    generic_protein = None
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

import PathUtils as PUtils
import SymDesignUtils as SDUtils

# Globals
# logger = SDUtils.start_log(__name__)
db = PUtils.biological_fragmentDB
index_offset = SDUtils.index_offset


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
#     if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):  # TODO Wrap into DesignDirectory object
#         pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.msa_pssm))
#     else:
#         pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.building_blocks, PUtils.msa_pssm))
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


def filter_pose(df_file, filters, weights, consensus=False, filter_file=PUtils.filter_and_sort):

    # if debug:
    #     global logger
    # else:
    logger = SDUtils.start_log(name=__name__, handler=1, level=2)
                                   # location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))

    idx = pd.IndexSlice
    df = pd.read_csv(df_file, index_col=0, header=[0, 1, 2])
    filter_df = pd.read_csv(filter_file, index_col=0)
    logger.info('Number of starting designs = %d' % len(df))
    logger.info('Using filter parameters: %s' % str(filters))

    # design_requirements = {'percent_int_area_polar': 0.4, 'buns_per_ang': 0.002}
    # crystal_means = {'int_area_total': 570, 'shape_complementarity': 0.63, 'number_hbonds': 5}
    # sort = {'protocol_energy_distance_sum': 0.25, 'shape_complementarity': 0.25, 'observed_evolution': 0.25,
    #         'int_composition_diff': 0.25}

    # When df is not ranked by percentage
    _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': filters[metric]}
                for metric in filters}

    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    _df = df.loc[:, idx['pose', df.columns.get_level_values(1) != 'std', :]].droplevel(1, axis=1).droplevel(0, axis=1)

    # Filter the DataFrame to include only those values which are le/ge the specified filter
    filters_with_idx = df_filter_index_by_value(_df, **_filters)
    filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
    logger.info('\n%s' % '\n'.join('Number of designs passing \'%s\' filter = %d' %
                                   (metric, len(filtered_indices[metric])) for metric in filtered_indices))
    final_indices = SDUtils.index_intersection(filtered_indices)
    logger.info('Final set of designs passing all filters has %d members' % len(final_indices))
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
        # TODO make more robust sampling from specific protocol
        _df = pd.merge(protocol_df.loc[:, idx['consensus', :]],
                       df.droplevel(0, axis=1).loc[:, idx[:, 'percent_fragment']],
                       left_index=True, right_index=True).droplevel(0, axis=1)
    # filtered_indices = {}

    # for metric in filters:
    #     filtered_indices[metric] = set(df[df.droplevel(0, axis=1)[metric] >= filters[metric]].index.to_list())
    #     logger.info('Number of designs passing %s = %d' % (metric, len(filtered_indices[metric])))
    _df = _df.loc[final_indices, :]
    # ranked_df = _df.rank(method='min', pct=True, )  # default is to rank lower values as closer to 1
    # need {column: {'direction': 'max', 'value': 0.5, 'idx': []}, ...}

    # only used to check out the number of designs in each filter
    # for _filter in crystal_filters_with_idx:
    #     print('%s designs = %d' % (_filter, len(crystal_filters_with_idx[_filter]['idx'])))

    # {column: {'direction': 'min', 'value': 0.3, 'idx': ['0001', '0002', ...]}, ...}

    # display(ranked_df[weights_s.index.to_list()] * weights_s)
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
    # these will be sorted by the largest value to the smallest
    # design_scores_s = (ranked_df[weights_s.index.to_list()] * weights_s).sum(axis=1).sort_values(ascending=False)
    # design_list = design_scores_s.index.to_list()
    # design_list = design_scores_s.index.to_list()[:num_designs]
    logger.info('%d poses were selected:\n%s' % (len(design_list), '\n'.join(design_list)))

    return design_list


@SDUtils.handle_errors(errors=(DesignDirectory.DesignError, AssertionError))
def select_sequences_s(des_dir, weights=None, filter_file=PUtils.filter_and_sort, number=1, debug=False):
    return select_sequences(des_dir, weights=weights, filter_file=filter_file, number=number, debug=debug)


def select_sequences_mp(des_dir, weights=None, filter_file=PUtils.filter_and_sort, number=1, debug=False):
    try:
        pose = select_sequences(des_dir, weights=weights, filter_file=filter_file, number=number, debug=debug)
        return pose, None
    except (DesignDirectory.DesignError, AssertionError) as e:
        return None, (des_dir.path, e)


def select_sequences(des_dir, weights=None, filter_file=PUtils.filter_and_sort, number=1, debug=False):
    """From a design directory find the sequences with the most neighbors to select for further characterization

    Args:
        des_dir (DesignDirectory)
    Keyword Args:
        weights=None (iter): The weights to use in sequence selection
        number=1 (int): The number of sequences to consider for each design
        debug=False (bool): Whether or not to debug
    Returns:
        (list): Containing tuples with (DesignDirectory, design index) for each sequence found
    """
    desired_protocol = 'combo_profile'
    # Log output
    if debug:
        global logger
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))

    # Load relevant data from the design directory
    # trajectory_file = glob(os.path.join(des_dir.all_scores, '%s_Trajectories.csv' % str(des_dir)))
    # assert len(trajectory_file) == 1, 'Couldn\'t find files for %s' % \
    #                                   os.path.join(des_dir.all_scores, '%s_Trajectories.csv' % str(des_dir))
    # trajectory_df = pd.read_csv(trajectory_file[0], index_col=0, header=[0])  # , 1, 2]
    trajectory_df = pd.read_csv(des_dir.trajectories, index_col=0, header=[0])  # , 1, 2]
    trajectory_df.dropna(inplace=True)
    # trajectory_df.dropna('protocol', inplace=True)
    # designs = trajectory_df.index.to_list()  # can't use with the mean and std statistics
    # designs = list(all_design_sequences[chains[0]].keys())
    logger.info('Number of starting trajectories = %d' % len(trajectory_df))
    designs = trajectory_df[trajectory_df['protocol'] == desired_protocol].index.to_list()

    if weights:
        filter_df = pd.read_csv(filter_file, index_col=0)
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

    # sequences_pickle = glob(os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir)))
    # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
    #                                    os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))
    #
    # all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
    # {chain: {name: sequence, ...}, ...}
    all_design_sequences = SDUtils.unpickle(des_dir.design_sequences)
    # all_design_sequences.pop(PUtils.stage[1])  # Remove refine from sequences, not in trajectory_df so unnecessary
    chains = list(all_design_sequences.keys())
    concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains]) for design in designs]
    logger.debug(chains)
    logger.debug(concatenated_sequences)

    # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, SDUtils.sequence_difference)
    # Using concatenated sequences makes the values incredibly similar and inflated as most residues are the same
    # doing min/max normalization to see variation
    pairwise_sequence_diff_l = [SequenceProfile.sequence_difference(*seq_pair)
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
            raise DesignDirectory.DesignError('no dataframe')
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
    dssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    residue_divergence_values = pos_specific_jsd(mutation_probabilities, dssm)

    interface_bkgd = SequenceProfile.get_db_aa_frequencies(db)
    interface_divergence_values = SDUtils.compute_jsd(mutation_probabilities, interface_bkgd)

    if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):  # TODO Wrap into DesignDirectory object
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.path, PUtils.msa_pssm))
    else:
        pssm = SequenceProfile.parse_pssm(os.path.join(des_dir.building_blocks, PUtils.msa_pssm))
    evolution_divergence_values = pos_specific_jsd(mutation_probabilities, pssm)

    final_mutation_dict = weave_mutation_dict(ranked_frequencies, mutation_probabilities, evolution_divergence_values,
                                              interface_divergence_values)


def mutate_wildtype_sequences(sequence_dir_files, wild_type_file):
    """Take a directory with PDB files and compare to a Wild-type PDB"""
    wt_seq_dict = get_pdb_sequences(wild_type_file)
    return generate_sequences(wt_seq_dict, generate_all_design_mutations(sequence_dir_files, wild_type_file))


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
            sys.exit()


def generate_all_design_mutations(all_design_files, wild_type_file, pose_num=False):  # Todo DEPRECIATE
    """From a list of PDB's and a wild-type PDB, generate a list of 'A5K' style mutations

    Args:
        all_design_files (list): PDB files on disk to extract sequence info and compare
        wild_type_file (str): PDB file on disk which contains a reference sequence
    Returns:
        mutations (dict): {'file_name': {chain_id: {mutation_index: {'from': 'A', 'to': 'K'}, ...}, ...}, ...}
    """
    pdb_dict = {'ref': PDB(file=wild_type_file)}
    for file_name in all_design_files:
        pdb = PDB(file=file_name)
        pdb.set_name(os.path.splitext(os.path.basename(file_name))[0])
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
            _offset = True
        else:
            sequence1, failures1 = extract_aa_seq(_pdb, _aa, _source, _chain)
            sequence2 = reference_seq_dict[_chain]
            sequence_dict[pdb_code][_chain] = sequence1
            _offset = False
        if mutation:
            seq_mutations = generate_mutations(sequence1, sequence2, offset=_offset)
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
        logger.info('No file was written:\n%s' % filepath)

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
    if not isinstance(pdb, PDB):
        pdb = PDB(file=pdb)

    seq_dict = {}
    for _chain in pdb.chain_id_list:
        seq_dict[_chain], fail = extract_aa_seq(pdb, source=source, chain=_chain)
    if chain:
        seq_dict = SDUtils.clean_dictionary(seq_dict, chain, remove=False)

    return seq_dict