from __future__ import annotations

import os
import subprocess
from itertools import combinations
from typing import Iterable, Any
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import sklearn

#     handle_design_errors, DesignError
from DesignMetrics import prioritize_design_indices, nanohedra_metrics  # query_user_for_metrics,
from path import ialign_exe_path
from PoseDirectory import PoseDirectory
from Structure import superposition3d, transform_coordinate_sets
from utils import start_log, digit_translate_table, sym

# globals
logger = start_log(name=__name__)


# def pose_rmsd_mp(pose_directories: list[PoseDirectory], cores: int = 1):
#     """Map the RMSD for a Nanohedra output based on building block directory (ex 1abc_2xyz)
#
#     Args:
#         pose_directories: List of relevant design directories
#         cores: Number of multiprocessing cores to run
#     Returns:
#         (dict): {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
#     """
#     pose_map = {}
#     pairs_to_process = []
#     singlets = {}
#     for pair1, pair2 in combinations(pose_directories, 2):
#         if pair1.composition == pair2.composition:
#             singlets.pop(pair1.composition, None)
#             pairs_to_process.append((pair1, pair2))
#         else:
#             # add all individual poses to a singles pool. pair2 is included in pair1, no need to add additional
#             singlets[pair1.composition] = pair1
#     compositions: dict[tuple[str, ...], list[PoseDirectory]] = group_compositions(pose_directories)
#     pairs_to_process = [grouping for entity_tuple, pose_directories in compositions.items()
#                         for grouping in combinations(pose_directories, 2)]
#     # find the rmsd between a pair of poses.  multiprocessing to increase throughput
#     _results = mp_map(pose_pair_rmsd, pairs_to_process, processes=cores)
#
#     # Make dictionary with all pairs
#     for pair, pair_rmsd in zip(pairs_to_process, _results):
#         protein_pair_path = os.path.basename(pair[0].building_blocks)
#         # protein_pair_path = pair[0].composition
#         # pose_map[result[0]] = result[1]
#         if protein_pair_path in pose_map:
#             # # {composition: {(pair1, pair2): rmsd, ...}, ...}
#             # {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
#             if str(pair[0]) in pose_map[protein_pair_path]:
#                 pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                 if str(pair[1]) not in pose_map[protein_pair_path]:
#                     pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself
#         else:
#             pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}  # add the pair with itself
#             pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#             pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself
#
#     # Add all singlets (poses that are missing partners) to the map
#     for protein_pair in singlets:
#         protein_path = os.path.basename(protein_pair)
#         if protein_path in pose_map:
#             # This logic is impossible??
#             pose_map[protein_path][str(singlets[protein_pair])] = {str(singlets[protein_pair]): 0.0}
#         else:
#             pose_map[protein_path] = {str(singlets[protein_pair]): {str(singlets[protein_pair]): 0.0}}
#
#     return pose_map


def pose_pair_rmsd(pair: tuple[PoseDirectory, PoseDirectory]) -> float:
    """Calculate the rmsd between pairs of Poses using CB coordinates. Must be the same length pose

    Args:
        pair: Paired PoseDirectory objects from pose processing directories
    Returns:
        RMSD value
    """
    #  using the intersecting residues at the interface of each pose

    # # protein_pair_path = pair[0].composition
    # # Grab designed resides from the pose_directory
    # design_residues = [set(pose.interface_design_residues) for pose in pair]
    #
    # # Set up the list of residues undergoing design (interface) on each pair. Return the intersection
    # # could use the union as well...?
    # des_residue_set = index_intersection(design_residues)
    # if not des_residue_set:  # when the two structures are not overlapped
    #     return np.nan
    # else:
    #     # pdb_parser = PDBParser(QUIET=True)
    #     # pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
    #     # rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
    #     #                       if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
    #     # pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
    #     #                   for entity_list in rmsd_residue_list]
    #     #
    #     # return superimpose(pair_atom_list)
    return superposition3d(*[pose.pose.cb_coords for pose in pair])[0]


# def pose_rmsd_s(all_des_dirs):
#     pose_map = {}
#     for pair in combinations(all_des_dirs, 2):
#         if pair[0].composition == pair[1].composition:
#             protein_pair_path = pair[0].composition
#             # Grab designed resides from the pose_directory
#             pair_rmsd = pose_pair_rmsd(pair)
#             # des_residue_list = [pose.info['des_residues'] for pose in pair]
#             # # could use the union as well...
#             # des_residue_set = index_intersection({pair[n]: set(pose_residues)
#             #                                               for n, pose_residues in enumerate(des_residue_list)})
#             # if des_residue_set == list():  # when the two structures are not significantly overlapped
#             #     pair_rmsd = np.nan
#             # else:
#             #     pdb_parser = PDBParser(QUIET=True)
#             #     # pdb = parser.get_structure(pdb_name, filepath)
#             #     pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
#             #     # returns a list with all ca atoms from a structure
#             #     # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].asu, pair[1].asu], SDUtils.get_biopdb_ca)
#             #     # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].path, pair[1].path], SDUtils.get_biopdb_ca)
#             #
#             #     # pair should be a structure...
#             #     # for structure in pair_structures:
#             #     #     for residue in structure.residues:
#             #     #         print(residue)
#             #     #         print(residue[0])
#             #     rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
#             #                           if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
#             #
#             #     # rmsd_residue_list = [[residue for residue in structure.residues
#             #     #                       if residue.get_id()[1] in des_residue_list[n]]
#             #     #                      for n, structure in enumerate(pair_structures)]
#             #
#             #     # print(rmsd_residue_list)
#             #     pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
#             #                       for entity_list in rmsd_residue_list]
#             #     # [atom for atom in structure.get_atoms if atom.get_id() == 'CA']
#             #     # pair_atom_list = SDUtils.get_rmsd_atoms(rmsd_residue_list, SDUtils.get_biopdb_ca)
#             #     # pair_rmsd = SDUtils.superimpose(pair_atoms, threshold)
#             #
#             #     pair_rmsd = SDUtils.superimpose(pair_atom_list)  # , threshold)
#             # if not pair_rmsd:
#             #     continue
#             if protein_pair_path in pose_map:
#                 # {composition: {(pair1, pair2): rmsd, ...}, ...}
#                 if str(pair[0]) in pose_map[protein_pair_path]:
#                     pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                     if str(pair[1]) not in pose_map[protein_pair_path]:
#                         pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
#                     # else:
#                     #     print('\n' * 6 + 'NEVER ACCESSED' + '\n' * 6)
#                     #     pose_map[pair[0].composition][str(pair[1])][str(pair[1])] = 0.0
#                 # else:
#                 #     print('\n' * 6 + 'ACCESSED' + '\n' * 6)
#                 #     pose_map[pair[0].composition][str(pair[0])] = {str(pair[1]): pair_rmsd}
#                 #     pose_map[pair[0].composition][str(pair[0])][str(pair[0])] = 0.0
#                 # pose_map[pair[0].composition][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
#             else:
#                 pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}
#                 pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
#                 pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
#                 # pose_map[pair[0].composition] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}
#
#     return pose_map


def ialign(pdb_file1, pdb_file2, chain1=None, chain2=None, out_path=os.path.join(os.getcwd(), 'ialign')):
    """Run non-sequential iAlign on two .pdb files

    Returns:
        (float): The IS score from Mu & Skolnic 2010
    """
    # example command
    # perl ../bin/ialign.pl -w output -s -a 0 1lyl.pdb AC 12as.pdb AB | grep "IS-score = "
    # output
    # IS-score = 0.38840, P-value = 0.3808E-003, Z-score =  7.873
    # chains = []
    # if chain1:
    #     chains += ['-c1', chain1]
    # if chain2:
    #     chains += ['-c2', chain2]

    if not chain1:
        chain1 = 'AB'
    if not chain2:
        chain2 = 'AB'
    chains = ['-c1', chain1, '-c2', chain2]

    temp_pdb_file1 = os.path.join(os.getcwd(), 'temp', os.path.basename(pdb_file1.translate(digit_translate_table)))
    temp_pdb_file2 = os.path.join(os.getcwd(), 'temp', os.path.basename(pdb_file2.translate(digit_translate_table)))
    os.system('scp %s %s' % (pdb_file1, temp_pdb_file1))
    os.system('scp %s %s' % (pdb_file2, temp_pdb_file2))
    # cmd = ['perl', ialign_exe_path, '-s', '-w', out_path, '-p1', pdb_file1, '-p2', pdb_file2] + chains
    cmd = ['perl', ialign_exe_path, '-s', '-w', out_path, '-p1', temp_pdb_file1, '-p2', temp_pdb_file2] + chains
    logger.debug('iAlign command: %s' % subprocess.list2cmdline(cmd))
    ialign_p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ialign_out, ialign_err = ialign_p.communicate()
    grep_p = subprocess.Popen(['grep', 'IS-score = '], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ialign_is_score, err = grep_p.communicate(input=ialign_out)

    ialign_is_score = ialign_is_score.decode()
    try:
        is_score, pvalue, z_score = [score.split('=')[-1].strip() for score in ialign_is_score.split(',')]
        logger.info('iAlign interface alignment: %s' % ialign_is_score.strip())
        return float(is_score)
        # return float(is_score), \
        #     os.path.splitext(os.path.basename(pdb_file1))[0], os.path.splitext(os.path.basename(pdb_file2))[0]
    except ValueError:
        logger.info('No significiant interface found')
        pass
    return 0.0
    # return 0.0, \
    #     os.path.splitext(os.path.basename(pdb_file1))[0], os.path.splitext(os.path.basename(pdb_file2))[0]


def cluster_poses(identifier_pairs: Iterable[tuple[Any, Any]], values: Iterable[float], epsilon: float = 1.) -> \
        dict[str | PoseDirectory, list[str | PoseDirectory]]:
    """Take pairs of identifiers and a computed value (such as RMSD) and cluster using DBSCAN algorithm

    Args:
        identifier_pairs: The identifiers for each pair measurement
        values: The corresponding measurement values for each pair of identifiers
        epsilon: The parameter for DBSCAN to influence the spread of clusters, needs to be tuned for measurement values
    Returns:
        {PoseDirectory representative: [PoseDirectory members], ... }
    """
    # BELOW IS THE INPUT FORMAT I WANT FOR cluster_poses()
    # index = list(combinations(pose_directories, 2)) + list(zip(pose_directories, pose_directories))
    # values = values + tuple(repeat(0, len(pose_directories)))
    # pd.Series(values, index=pd.MultiIndex.from_tuples(index)).unstack()

    pair_df = pd.Series(values, index=pd.MultiIndex.from_tuples(identifier_pairs)).fillna(0.).unstack()
    # symmetric_pair_values = sym(pair_df.values)

    # PCA analysis of distances
    # building_block_rmsd_matrix = sklearn.preprocessing.StandardScaler().fit_transform(symmetric_pair_values)
    # pca = PCA(PUtils.variance)
    # building_block_rmsd_pc_np = pca.fit_transform(building_block_rmsd_matrix)
    # pca_distance_vector = pdist(building_block_rmsd_pc_np)
    # epsilon = pca_distance_vector.mean() * 0.5
    # Compute pose clusters using DBSCAN algorithm
    # precomputed specifies that a precomputed distance matrix is being passed
    dbscan = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=2, metric='precomputed')
    dbscan.fit(sym(pair_df.to_numpy()))
    # find the cluster representative by minimizing the cluster mean
    cluster_ids = set(dbscan.labels_)
    # print(dbscan.labels_)
    # use of dbscan.core_sample_indices_ returns all core_samples which is not a nearest neighbors mean index
    # print(dbscan.core_sample_indices_)
    try:
        cluster_ids.remove(-1)  # remove outlier label, will add all these later
    except KeyError:
        pass

    # Find the cluster representative and members
    clustered_poses = {}
    for cluster_id in cluster_ids:
        # loc_indices = pair_df.index[np.where(cluster_id == dbscan.labels_)]
        # cluster_representative = pair_df.loc[loc_indices, loc_indices].mean().argmax()
        iloc_indices = np.where(dbscan.labels_ == cluster_id)
        # take mean (doesn't matter which axis) and find the minimum (most similar to others) as representative
        cluster_representative_idx = pair_df.iloc[iloc_indices, iloc_indices].mean().argmin()
        # set all the cluster members belonging to the cluster representative
        # pose_cluster_members = pair_df.index[iloc_indices].to_list()
        clustered_poses[pair_df.index[cluster_representative_idx]] = pair_df.index[iloc_indices].to_list()

    # Add all outliers to the clustered poses as a representative
    outlier_poses = pair_df.index[np.where(dbscan.labels_ == -1)]
    clustered_poses.update(dict(zip(outlier_poses, outlier_poses)))

    return clustered_poses


def predict_best_pose_from_transformation_cluster(train_trajectories_file, training_clusters):
    """From full training Nanohedra, Rosetta Sequecnce Design analyzed trajectories, train a linear model to select the
    best trajectory from a group of clustered poses given only the Nanohedra Metrics

    Args:
        train_trajectories_file (str): Location of a Cluster Trajectory Analysis .csv with complete metrics for cluster
        training_clusters (dict): Mapping of cluster representative to cluster members

    Returns:
        (sklearn.linear_model)
    """
    possible_lin_reg = {'MultiTaskLassoCV': sklearn.linear_model.MultiTaskLassoCV,
                        'LassoCV': sklearn.linear_model.LassoCV,
                        'MultiTaskElasticNetCV': sklearn.linear_model.MultiTaskElasticNetCV,
                        'ElasticNetCV': sklearn.linear_model.ElasticNetCV}
    idx_slice = pd.IndexSlice
    trajectory_df = pd.read_csv(train_trajectories_file, index_col=0, header=[0, 1, 2])
    # 'dock' category is synonymous with nanohedra metrics
    trajectory_df = trajectory_df.loc[:, idx_slice[['pose', 'no_constraint'],
                                                   ['mean', 'dock', 'seq_design'], :]].droplevel(1, axis=1)
    # scale the data to a standard gaussian distribution for each trajectory independently
    # Todo ensure this mechanism of scaling is correct for each cluster individually
    scaler = sklearn.preprocessing.StandardScaler()
    train_traj_df = pd.concat([scaler.fit_transform(trajectory_df.loc[cluster_members, :])
                               for cluster_members in training_clusters.values()], keys=list(training_clusters.keys()),
                              axis=0)

    # standard_scale_traj_df[train_traj_df.columns] = standard_scale.transform(train_traj_df)

    # select the metrics which the linear model should be trained on
    nano_traj = train_traj_df.loc[:, nanohedra_metrics]

    # select the Rosetta metrics to train model on
    # potential_training_metrics = set(train_traj_df.columns).difference(nanohedra_metrics)
    # rosetta_select_metrics = query_user_for_metrics(potential_training_metrics, mode='design', level='pose')
    rosetta_metrics = {'shape_complementarity': sklearn.preprocessing.StandardScaler(),  # I think a gaussian dist is preferable to MixMax
                       # 'protocol_energy_distance_sum': 0.25,  This will select poses by evolution
                       'int_composition_similarity': sklearn.preprocessing.StandardScaler(),  # gaussian preferable to MixMax
                       'interface_energy': sklearn.preprocessing.StandardScaler(),  # gaussian preferable to MaxAbsScaler,
                       # 'observed_evolution': 0.25}  # also selects by evolution
                       }
    # assign each metric a weight proportional to it's share of the total weight
    rosetta_select_metrics = {item: 1 / len(rosetta_metrics) for item in rosetta_metrics}
    # weighting scheme inherently standardizes the weights between [0, 1] by taking a linear combination of the metrics
    targets = prioritize_design_indices(train_trajectories_file, weight=rosetta_select_metrics)  # weight=True)

    # for proper MultiTask model training, must scale the selected metrics. This is performed on trajectory_df above
    # targets2d = train_traj_df.loc[:, rosetta_select_metrics.keys()]
    pose_traj_df = train_traj_df.loc[:, idx_slice['pose', 'int_composition_similarity']]
    no_constraint_traj_df = \
        train_traj_df.loc[:, idx_slice['no_constraint',
                                       set(rosetta_metrics.keys()).difference('int_composition_similarity')]]
    targets2d = pd.concat([pose_traj_df, no_constraint_traj_df])

    # split training and test dataset
    trajectory_train, trajectory_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(nano_traj, targets, random_state=42)
    trajectory_train2d, trajectory_test2d, target_train2d, target_test2d = \
        sklearn.model_selection.train_test_split(nano_traj, targets2d, random_state=42)
    # calculate model performance with cross-validation, alpha tuning
    alphas = np.logspace(-10, 10, 21)  # Todo why log space here?
    # then compare between models based on various model scoring parameters
    reg_scores, mae_scores = [], []
    for lin_reg, model in possible_lin_reg.items():
        if lin_reg.startswith('MultiTask'):
            trajectory_train, trajectory_test = trajectory_train2d, trajectory_test2d
            target_train, target_test = target_train2d, target_test2d
        # else:
        #     target = target_train
        test_reg = model(alphas=alphas).fit(trajectory_train, target_train)
        reg_scores.append(test_reg.score(trajectory_train, target_train))
        target_test_prediction = test_reg.predict(trajectory_test, target_test)
        mae_scores.append(sklearn.metrics.median_absolute_error(target_test, target_test_prediction))


def chose_top_pose_from_model(test_trajectories_file, clustered_poses, model):
    """
    Args:
        test_trajectories_file (str): Location of a Nanohedra Trajectory Analysis .csv with Nanohedra metrics
        clustered_poses (dict): A set of clustered poses that share similar transformational parameters
    Returns:

    """
    test_docking_df = pd.read_csv(test_trajectories_file, index_col=0, header=[0, 1, 2])

    for cluster_representative, designs in clustered_poses.items():
        trajectory_df = test_docking_df.loc[designs, nanohedra_metrics]
        trajectory_df['model_predict'] = model.predict(trajectory_df)
        trajectory_df.sort_values('model_predict')


def return_transform_pair_as_guide_coordinate_pair(transform1, transform2):
    # make a blank set of guide coordinates for each incoming transformation
    guide_coords = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), (len(transform1['rotation']), 1, 1))
    transformed_guide_coords1 = transform_coordinate_sets(guide_coords, **transform1)
    transformed_guide_coords2 = transform_coordinate_sets(guide_coords, **transform2)

    return np.concatenate([transformed_guide_coords1.reshape(-1, 9), transformed_guide_coords2.reshape(-1, 9)], axis=1)


def cluster_transformation_pairs(transform1: dict[str, np.ndarray], transform2: dict[str, np.ndarray],
                                 distance: float = 1., minimum_members: int = 2) -> \
        tuple[sklearn.neighbors._unsupervised.NearestNeighbors, sklearn.cluster._dbscan.DBSCAN]:
    #                              return_representatives=True):
    """Cluster Pose conformations according to their specific transformation parameters to find Poses which occupy
    essentially the same space

    Args:
        transform1: First set of rotations/translations to be clustered
            {'rotation': rot_array, 'translation': tx_array, 'rotation2': rot2_array, 'translation2': tx2_array}
        transform2: Second set of rotations/translations to be clustered
            {'rotation': rot_array, 'translation': tx_array, 'rotation2': rot2_array, 'translation2': tx2_array}
        distance: The distance to query neighbors in transformational space
        minimum_members: The minimum number of members in each cluster
    Returns:
        The sklearn tree with the calculated nearest neighbors, the DBSCAN clustering object
        Representative indices, DBSCAN cluster membership indices
    """
    # Todo tune DBSCAN distance (epsilon) to be reflective of the data, should be related to radius in NearestNeighbors
    #  but smaller by some amount. Ideal amount would be the distance between two transformed guide coordinate sets of
    #  a similar tx and a 3 degree step of rotation.
    transformed_guide_coords = return_transform_pair_as_guide_coordinate_pair(transform1, transform2)

    # create a tree structure describing the distances of all transformed points relative to one another
    nearest_neightbors_ball_tree = sklearn.neighbors.NearestNeighbors(algorithm='ball_tree', radius=distance)
    nearest_neightbors_ball_tree.fit(transformed_guide_coords)
    #                                sort_results returns only non-zero entries and provides the smallest distance first
    distance_graph = nearest_neightbors_ball_tree.radius_neighbors_graph(mode='distance', sort_results=True)  # <- sort doesn't work?
    #                                                      X=transformed_guide_coords is implied
    # because this doesn't work to sort_results and pull out indices, I have to do another step 'radius_neighbors'
    dbscan_cluster: sklearn.cluster.DBSCAN = \
        sklearn.cluster.DBSCAN(eps=distance, min_samples=minimum_members, metric='precomputed').fit(distance_graph)
    #                                         sample_weight=A WEIGHT?

    # if return_representatives:
    #     return find_cluster_representatives(nearest_neightbors_ball_tree, dbscan_cluster)
    # else:  # return data structure
    return nearest_neightbors_ball_tree, dbscan_cluster  # .labels_


def find_cluster_representatives(transform_tree: sklearn.neighbors._unsupervised.NearestNeighbors,
                                 cluster: sklearn.cluster._dbscan.DBSCAN) \
        -> tuple[list[int], np.ndarray]:
    """Return the cluster representative indices and the cluster membership identity for all member data

    Args:
        transform_tree: The sklearn tree with the calculated nearest neighbors
        cluster: The DBSCAN clustering object
    Returns:
        The list of representative indices, array of all indices membership
    """
    # Get the neighbors for each point in the tree according to the fit distance
    tree_distances, tree_indices = transform_tree.radius_neighbors(sort_results=True)
    # Find mean distance to all neighbors for each index
    with catch_warnings():
        # Empty slices can't compute mean, so catch warning if cluster is an outlier
        simplefilter('ignore', category=RuntimeWarning)
        mean_cluster_dist = np.empty(tree_distances.shape[0])
        for idx in range(tree_distances.shape[0]):
            mean_cluster_dist[idx] = tree_distances[idx].mean()

    # For each label (cluster), add the minimal mean (representative) the representative transformation indices
    outlier = -1  # -1 are outliers in DBSCAN
    representative_transformation_indices = []
    for label in set(cluster.labels_) - {outlier}:  # labels live here
        cluster_indices = np.flatnonzero(cluster.labels_ == label)
        # Get the minimal argument from the mean distances for each index in the cluster
        # This index is the cluster representative
        representative_transformation_indices.append(cluster_indices[mean_cluster_dist[cluster_indices].argmin()])
    # Add all outliers to representatives
    representative_transformation_indices.extend(np.flatnonzero(cluster.labels_ == outlier).tolist())

    return representative_transformation_indices, cluster.labels_


# @handle_design_errors(errors=(DesignError, AssertionError))
# @handle_errors(errors=(DesignError, ))
def cluster_designs(compositions: list[PoseDirectory]) -> dict[str | PoseDirectory, list[str | PoseDirectory]]:
    """From a group of poses with matching protein composition, cluster the designs according to transformational
    parameters to identify the unique poses in each composition

    Args:
        compositions: The group of PoseDirectory objects to pull transformation data from
    Returns:
        Cluster with representative pose as the key and matching poses as the values
    """
    # format all transforms for the selected compositions
    stacked_transforms = [pose_directory.pose_transformation for pose_directory in compositions]
    trans1_rot1, trans1_tx1, trans1_rot2, trans1_tx2 = zip(*[transform[0].values()
                                                             for transform in stacked_transforms])
    trans2_rot1, trans2_tx1, trans2_rot2, trans2_tx2 = zip(*[transform[1].values()
                                                             for transform in stacked_transforms])

    # must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
    transformation1 = {'rotation': np.array(trans1_rot1), 'translation': np.array(trans1_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans1_rot2), 'translation2': np.array(trans1_tx2)[:, np.newaxis, :]}
    transformation2 = {'rotation': np.array(trans2_rot1), 'translation': np.array(trans2_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans2_rot2), 'translation2': np.array(trans2_tx2)[:, np.newaxis, :]}

    # This section could be added to the Nanohedra docking routine
    cluster_representative_indices, cluster_labels = \
        find_cluster_representatives(*cluster_transformation_pairs(transformation1, transformation2))

    representative_labels = cluster_labels[cluster_representative_indices]
    # pull out pose's from the input composition_designs groups (PoseDirectory)
    # if return_pose_id:  # convert all DesignDirectories to pose-id's
    #     # don't add the outliers now (-1 labels)
    #     composition_map = \
    #         {str(compositions[rep_idx]):
    #             [str(compositions[idx]) for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
    #          for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels) if rep_label != -1}
    #     # add the outliers as separate occurrences
    #     composition_map.update({str(compositions[idx]): []
    #                             for idx in np.flatnonzero(cluster_labels == -1).tolist()})
    # else:  # return the PoseDirectory object
    composition_map = \
        {compositions[rep_idx]: [compositions[idx] for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
         for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels) if rep_label != -1}
    composition_map.update({compositions[idx]: [] for idx in np.flatnonzero(cluster_labels == -1).tolist()})

    return composition_map


def group_compositions(pose_directories: list[PoseDirectory]) -> dict[tuple[str, ...], list[PoseDirectory]]:
    """From a set of DesignDirectories, find all the compositions and group together

    Args:
        pose_directories: The PoseDirectory to group according to composition
    Returns:
        List of similarly named PoseDirectory mapped to their name
    """
    compositions = {}
    for pose in pose_directories:
        entity_names = tuple(pose.entity_names)
        found_composition = None
        for permutation in combinations(entity_names, len(entity_names)):
            found_composition = compositions.get(permutation, None)
            if found_composition:
                break

        if found_composition:
            compositions[entity_names].append(pose)
        else:
            compositions[entity_names] = [pose]

    return compositions


def invert_cluster_map(cluster_map: dict[str | PoseDirectory, list[str | PoseDirectory]]):
    """Return an inverted cluster map where the cluster members map to the representative

    Args:
        cluster_map: The standard pose_cluster_map format
    Returns:
        An invert pose_cluster_map where the members are keys and the representative is the value
    """
    inverted_map = {pose: cluster_rep for cluster_rep, poses in cluster_map.items() for pose in poses}
    inverted_map.update({cluster_rep: cluster_rep for cluster_rep in cluster_map})  # to add all outliers

    return inverted_map
