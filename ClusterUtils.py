import os
import subprocess
from itertools import combinations
from typing import List, Tuple, Dict, Union
from warnings import catch_warnings, simplefilter

import numpy
import numpy as np
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import euclidean, pdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLassoCV, LassoCV, MultiTaskElasticNetCV, ElasticNetCV
from sklearn.metrics import median_absolute_error  # r2_score,
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from DesignDirectory import DesignDirectory
from PathUtils import ialign_exe_path
from SymDesignUtils import index_intersection, mp_map, sym, rmsd_threshold, digit_translate_table, start_log
#     handle_design_errors, DesignError
from DesignMetrics import prioritize_design_indices, nanohedra_metrics  # query_user_for_metrics,
from Structure import superposition3d
from utils.GeneralUtils import transform_coordinate_sets


# globals
logger = start_log(name=__name__)


def pose_rmsd_mp(all_des_dirs, threads=1):
    """Map the RMSD for a Nanohedra output based on building block directory (ex 1abc_2xyz)

    Args:
        all_des_dirs (list[DesignDirectory]): List of relevant design directories
    Keyword Args:
        threads: Number of multiprocessing threads to run
    Returns:
        (dict): {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
    """
    pose_map = {}
    pairs_to_process = []
    singlets = {}
    for pair1, pair2 in combinations(all_des_dirs, 2):
        if pair1.composition == pair2.composition:
            singlets.pop(pair1.composition, None)
            pairs_to_process.append((pair1, pair2))
        else:
            # add all individual poses to a singles pool. pair2 is included in pair1, no need to add additional
            singlets[pair1.composition] = pair1
    # find the rmsd between a pair of poses.  multiprocessing to increase throughput
    _results = mp_map(pose_pair_rmsd, pairs_to_process, threads=threads)

    # Make dictionary with all pairs
    for pair, pair_rmsd in zip(pairs_to_process, _results):
        protein_pair_path = os.path.basename(pair[0].building_blocks)
        # protein_pair_path = pair[0].composition
        # pose_map[result[0]] = result[1]
        if protein_pair_path in pose_map:
            # # {composition: {(pair1, pair2): rmsd, ...}, ...}
            # {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
            if str(pair[0]) in pose_map[protein_pair_path]:
                pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
                if str(pair[1]) not in pose_map[protein_pair_path]:
                    pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself
        else:
            pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}  # add the pair with itself
            pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
            pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}  # add the pair with itself

    # Add all singlets (poses that are missing partners) to the map
    for protein_pair in singlets:
        protein_path = os.path.basename(protein_pair)
        if protein_path in pose_map:
            # This logic is impossible??
            pose_map[protein_path][str(singlets[protein_pair])] = {str(singlets[protein_pair]): 0.0}
        else:
            pose_map[protein_path] = {str(singlets[protein_pair]): {str(singlets[protein_pair]): 0.0}}

    return pose_map


def pose_pair_rmsd(pair):
    """Calculate the rmsd between Nanohedra pose pairs using the intersecting residues at the interface of each pose

    Args:
        pair (tuple[DesignDirectory.DesignDirectory, DesignDirectory.DesignDirectory]):
            Paired DesignDirectory objects from pose processing directories
    Returns:
        (float): RMSD value
    """
    # protein_pair_path = pair[0].composition
    # Grab designed resides from the design_directory
    design_residues = [set(pose.design_residues) for pose in pair]

    # Set up the list of residues undergoing design (interface) on each pair. Return the intersection
    # could use the union as well...?
    des_residue_set = index_intersection(design_residues)
    if not des_residue_set:  # when the two structures are not overlapped
        return np.nan
    else:
        # pdb_parser = PDBParser(QUIET=True)
        # pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
        # rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
        #                       if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
        # pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
        #                   for entity_list in rmsd_residue_list]
        #
        # return superimpose(pair_atom_list)
        return superposition3d(*[pose.pose.coords for pose in pair])[0]


def pose_rmsd_s(all_des_dirs):
    pose_map = {}
    for pair in combinations(all_des_dirs, 2):
        if pair[0].composition == pair[1].composition:
            protein_pair_path = pair[0].composition
            # Grab designed resides from the design_directory
            pair_rmsd = pose_pair_rmsd(pair)
            # des_residue_list = [pose.info['des_residues'] for pose in pair]
            # # could use the union as well...
            # des_residue_set = index_intersection({pair[n]: set(pose_residues)
            #                                               for n, pose_residues in enumerate(des_residue_list)})
            # if des_residue_set == list():  # when the two structures are not significantly overlapped
            #     pair_rmsd = np.nan
            # else:
            #     pdb_parser = PDBParser(QUIET=True)
            #     # pdb = parser.get_structure(pdb_name, filepath)
            #     pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
            #     # returns a list with all ca atoms from a structure
            #     # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].asu, pair[1].asu], SDUtils.get_biopdb_ca)
            #     # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].path, pair[1].path], SDUtils.get_biopdb_ca)
            #
            #     # pair should be a structure...
            #     # for structure in pair_structures:
            #     #     for residue in structure.residues:
            #     #         print(residue)
            #     #         print(residue[0])
            #     rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
            #                           if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
            #
            #     # rmsd_residue_list = [[residue for residue in structure.residues
            #     #                       if residue.get_id()[1] in des_residue_list[n]]
            #     #                      for n, structure in enumerate(pair_structures)]
            #
            #     # print(rmsd_residue_list)
            #     pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
            #                       for entity_list in rmsd_residue_list]
            #     # [atom for atom in structure.get_atoms if atom.get_id() == 'CA']
            #     # pair_atom_list = SDUtils.get_rmsd_atoms(rmsd_residue_list, SDUtils.get_biopdb_ca)
            #     # pair_rmsd = SDUtils.superimpose(pair_atoms, threshold)
            #
            #     pair_rmsd = SDUtils.superimpose(pair_atom_list)  # , threshold)
            # if not pair_rmsd:
            #     continue
            if protein_pair_path in pose_map:
                # {composition: {(pair1, pair2): rmsd, ...}, ...}
                if str(pair[0]) in pose_map[protein_pair_path]:
                    pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
                    if str(pair[1]) not in pose_map[protein_pair_path]:
                        pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
                    # else:
                    #     print('\n' * 6 + 'NEVER ACCESSED' + '\n' * 6)
                    #     pose_map[pair[0].composition][str(pair[1])][str(pair[1])] = 0.0
                # else:
                #     print('\n' * 6 + 'ACCESSED' + '\n' * 6)
                #     pose_map[pair[0].composition][str(pair[0])] = {str(pair[1]): pair_rmsd}
                #     pose_map[pair[0].composition][str(pair[0])][str(pair[0])] = 0.0
                # pose_map[pair[0].composition][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
            else:
                pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}
                pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
                pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
                # pose_map[pair[0].composition] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}

    return pose_map


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


def cluster_poses(pose_map):
    """Take a pose map calculated by pose_rmsd (_mp or _s) and cluster using DBSCAN algorithm

    Args:
        pose_map (dict): {composition: {pair1: {pair2: rmsd, ...}, ...}, ...}
    Returns:
        (dict): {composition: {'poses clustered'}, ... }
    """
    pose_cluster_map = {}
    for building_blocks in pose_map:
        building_block_rmsd_df = pd.DataFrame(pose_map[building_blocks]).fillna(0.0)

        # PCA analysis of distances
        # pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
        # for k, dist in enumerate(pairwise_sequence_diff_np):
        #     i, j = SDUtils.condensed_to_square(k, len(designs))
        #     pairwise_sequence_diff_mat[i, j] = dist
        building_block_rmsd_matrix = sym(building_block_rmsd_df.values)
        # print(building_block_rmsd_df.values)
        # print(building_block_rmsd_matrix)
        # building_block_rmsd_matrix = StandardScaler().fit_transform(building_block_rmsd_matrix)
        # pca = PCA(PUtils.variance)
        # building_block_rmsd_pc_np = pca.fit_transform(building_block_rmsd_matrix)
        # pca_distance_vector = pdist(building_block_rmsd_pc_np)
        # epsilon = pca_distance_vector.mean() * 0.5

        # Compute pose clusters using DBSCAN algorithm
        # logger.info('Finding pose clusters within RMSD of %f' % rmsd_threshold) # TODO
        dbscan = DBSCAN(eps=rmsd_threshold, min_samples=2, metric='precomputed')
        dbscan.fit(building_block_rmsd_matrix)

        # find the cluster representative by minimizing the cluster mean
        cluster_ids = set(dbscan.labels_)
        # print(dbscan.labels_)
        # print(dbscan.core_sample_indices_)
        if -1 in cluster_ids:
            cluster_ids.remove(-1)  # remove outlier label, will add all these later
        pose_indices = building_block_rmsd_df.index.to_list()
        cluster_members_map = {cluster_id: [pose_indices[n] for n, cluster in enumerate(dbscan.labels_)
                                            if cluster == cluster_id]
                               for cluster_id in cluster_ids}

        # Find the cluster representative
        # use of dbscan.core_sample_indices_ returns all core_samples which is not a nearest neighbors mean index
        clustered_poses = {}
        for cluster in cluster_members_map:
            cluster_df = building_block_rmsd_df.loc[cluster_members_map[cluster], cluster_members_map[cluster]]
            cluster_representative = cluster_df.mean().sort_values().index[0]
            for member in cluster_members_map[cluster]:
                clustered_poses[member] = cluster_representative  # includes representative
            # cluster_representative_map[cluster] = cluster_representative
            # cluster_representative_map[cluster_representative] = cluster_members_map[cluster]

        # make dictionary with the core representative as the label and the matches as a list
        # clustered_poses = {cluster_representative_map[cluster]: cluster_members_map[cluster]
        #                    for cluster in cluster_representative_map}

        # clustered_poses = {building_block_rmsd_df.iloc[idx, :].index:
        #                    building_block_rmsd_df.iloc[idx, [n
        #                                                      for n, cluster in enumerate(dbscan.labels_)
        #                                                      if cluster == dbscan.labels_[idx]]].index.to_list()
        #                    for idx in dbscan.core_sample_indices_}

        # Add all outliers to the clustered poses as a representative
        clustered_poses.update({building_block_rmsd_df.index[idx]: building_block_rmsd_df.index[idx]
                                for idx, cluster in enumerate(dbscan.labels_) if cluster == -1})
        pose_cluster_map[building_blocks] = clustered_poses

    return pose_cluster_map


def predict_best_pose_from_transformation_cluster(train_trajectories_file, training_clusters):
    """From full training Nanohedra, Rosetta Sequecnce Design analyzed trajectories, train a linear model to select the
    best trajectory from a group of clustered poses given only the Nanohedra Metrics

    Args:
        train_trajectories_file (str): Location of a Cluster Trajectory Analysis .csv with complete metrics for cluster
        training_clusters (dict): Mapping of cluster representative to cluster members

    Returns:
        (sklearn.linear_model)
    """
    possible_lin_reg = {'MultiTaskLassoCV': MultiTaskLassoCV,
                        'LassoCV': LassoCV,
                        'MultiTaskElasticNetCV': MultiTaskElasticNetCV,
                        'ElasticNetCV': ElasticNetCV}
    idx_slice = pd.IndexSlice
    trajectory_df = pd.read_csv(train_trajectories_file, index_col=0, header=[0, 1, 2])
    # 'dock' category is synonymous with nanohedra metrics
    trajectory_df = trajectory_df.loc[:, idx_slice[['pose', 'no_constraint'],
                                                   ['mean', 'dock', 'seq_design'], :]].droplevel(1, axis=1)
    # scale the data to a standard gaussian distribution for each trajectory independently
    # Todo ensure this mechanism of scaling is correct for each cluster individually
    scaler = StandardScaler()
    train_traj_df = pd.concat([scaler.fit_transform(trajectory_df.loc[cluster_members, :])
                               for cluster_members in training_clusters.values()], keys=list(training_clusters.keys()),
                              axis=0)

    # standard_scale_traj_df[train_traj_df.columns] = standard_scale.transform(train_traj_df)

    # select the metrics which the linear model should be trained on
    nano_traj = train_traj_df.loc[:, nanohedra_metrics]

    # select the Rosetta metrics to train model on
    # potential_training_metrics = set(train_traj_df.columns).difference(nanohedra_metrics)
    # rosetta_select_metrics = query_user_for_metrics(potential_training_metrics, mode='design', level='pose')
    rosetta_metrics = {'shape_complementarity': StandardScaler(),  # I think a gaussian dist is preferable to MixMax
                       # 'protocol_energy_distance_sum': 0.25,  This will select poses by evolution
                       'int_composition_similarity': StandardScaler(),  # gaussian preferable to MixMax
                       'interface_energy': StandardScaler(),  # gaussian preferable to MaxAbsScaler,
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
    trajectory_train, trajectory_test, target_train, target_test = train_test_split(nano_traj, targets, random_state=42)
    trajectory_train2d, trajectory_test2d, target_train2d, target_test2d = train_test_split(nano_traj, targets2d,
                                                                                            random_state=42)
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
        mae_scores.append(median_absolute_error(target_test, target_test_prediction))


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


def cluster_transformation_pairs(transform1, transform2, distance=1.0, minimum_members=2):  # , return_representatives=True):
    """Cluster Pose conformations according to their specific transformation parameters to find Poses which occupy
    essentially the same space

    Args:
        transform1 (dict[mapping[str, numpy.ndarray]]): First set of rotations/translations to be clustered
            {'rotation': rot_array, 'translation': tx_array, 'rotation2': rot2_array, 'translation2': tx2_array}
        transform2 (dict[mapping[str, numpy.ndarray]]): Second set of rotations/translations to be clustered
            {'rotation': rot_array, 'translation': tx_array, 'rotation2': rot2_array, 'translation2': tx2_array}
    Keyword Args:
        distance=1.0 (float): The distance to query neighbors in transformational space
        minimum_members (int): The minimum number of members in each cluster
    Returns:
        (tuple[sklearn.neighbors.NearestNeighbors, sklearn.dbscan_cluster.DBSCAN]): Representative indices DBSCAN cluster membership indices
    """
    # Todo tune DBSCAN distance (epsilon) to be reflective of the data, should be related to radius in NearestNeighbors
    #  but smaller by some amount. Ideal amount would be the distance between two transformed guide coordinate sets of
    #  a similar tx and a 3 degree step of rotation.
    transformed_guide_coords = return_transform_pair_as_guide_coordinate_pair(transform1, transform2)

    # create a tree structure describing the distances of all transformed points relative to one another
    nearest_neightbors_ball_tree = NearestNeighbors(algorithm='ball_tree', radius=distance)
    nearest_neightbors_ball_tree.fit(transformed_guide_coords)
    #                                sort_results returns only non-zero entries and provides the smallest distance first
    distance_graph = nearest_neightbors_ball_tree.radius_neighbors_graph(mode='distance', sort_results=True)  # <- sort doesn't work?
    #                                                      X=transformed_guide_coords is implied
    # because this doesn't work to sort_results and pull out indices, I have to do another step 'radius_neighbors'
    dbscan_cluster = DBSCAN(eps=distance, min_samples=minimum_members, metric='precomputed').fit(distance_graph)  # , sample_weight=A WEIGHT?

    # if return_representatives:
    #     return find_cluster_representatives(nearest_neightbors_ball_tree, dbscan_cluster)
    # else:  # return data structure
    return nearest_neightbors_ball_tree, dbscan_cluster  # .labels_


def find_cluster_representatives(transform_tree, cluster) -> Tuple[List, numpy.ndarray]:
    """Return the cluster representative indices and the cluster membership identity for all member data

    Args:
        transform_tree (sklearn.neighbors.NearestNeighbors):
        cluster (sklearn.cluster.DBSCAN):
    Returns:
        (tuple[list, numpy.ndarray]) The list of representative indices and the array of all indices membership
    """
    outlier = -1  # -1 are outliers in DBSCAN
    tree_distances, tree_indices = transform_tree.radius_neighbors(sort_results=True)
    # find cluster mean for each index
    with catch_warnings():
        # empty slices can't have mean, so catch warning if cluster is an outlier
        simplefilter('ignore', category=RuntimeWarning)
        mean_cluster_dist = np.empty(tree_distances.shape[0])
        for idx, array in enumerate(tree_distances.tolist()):
            mean_cluster_dist[idx] = array.mean()

    # for each label (cluster), add the minimal mean (representative) the representative transformation indices
    representative_transformation_indices = []
    for label in set(cluster.labels_) - {outlier}:  # labels live here
        cluster_indices = np.flatnonzero(cluster.labels_ == label)
        representative_transformation_indices.append(cluster_indices[mean_cluster_dist[cluster_indices].argmin()])
    # add all outliers to representatives
    representative_transformation_indices.extend(np.flatnonzero(cluster.labels_ == outlier).tolist())

    return representative_transformation_indices, cluster.labels_


# @handle_design_errors(errors=(DesignError, AssertionError))
# @handle_errors(errors=(DesignError, ))
def cluster_designs(composition_designs: List[DesignDirectory], return_pose_id: bool = True) -> \
        Dict[Union[str, DesignDirectory], List[Union[str, DesignDirectory]]]:
    """From a group of poses with matching protein composition, cluster the designs according to transformational
    parameters to identify the unique poses in each composition

    Args:
        composition_designs: The group of DesignDirectory objects to pull transformation data from
        return_pose_id: Whether the DesignDirectory object should be returned instead of its name
    Returns:
        Cluster with representative pose as the key and matching poses as the values
    """
    # format all transforms for the selected compositions
    stacked_transforms = [design_directory.pose_transformation for design_directory in composition_designs]
    trans1_rot1, trans1_tx1, trans1_rot2, trans1_tx2 = zip(*[transform[1].values()
                                                             for transform in stacked_transforms])
    trans2_rot1, trans2_tx1, trans2_rot2, trans2_tx2 = zip(*[transform[2].values()
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
    # pull out pose's from the input composition_designs groups (DesignDirectory)
    if return_pose_id:  # convert all DesignDirectories to pose-id's
        # don't add the outliers now (-1 labels)
        composition_map = \
            {str(composition_designs[rep_idx]):
                [str(composition_designs[idx]) for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
             for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels) if rep_label != -1}
        # add the outliers as separate occurrences
        composition_map.update({str(composition_designs[idx]): []
                                for idx in np.flatnonzero(cluster_labels == -1).tolist()})
    else:  # return the DesignDirectory object
        composition_map = \
            {composition_designs[rep_idx]: [composition_designs[idx]
                                            for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
             for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels) if rep_label != -1}
        composition_map.update({composition_designs[idx]: [] for idx in np.flatnonzero(cluster_labels == -1).tolist()})

    return composition_map


def group_compositions(design_directories):
    """From a set of DesignDirectories, find all the compositions and group together"""
    compositions = {}
    for design in design_directories:
        entity_names = tuple(design.entity_names)
        if compositions.get(entity_names, None):
            compositions[entity_names].append(design)
        else:
            compositions[entity_names] = [design]

    return compositions


def invert_cluster_map(cluster_map):
    """Return an inverted cluster map where the cluster members map to the representative

    Args:
        cluster_map (dict[mapping[representative DesignDirectoryID, list[member DesignDirectoryID]])
    Returns:
        (dict[mapping[member DesignDirectoryID, representative DesignDirectoryID]])
    """
    inverted_map = {pose: cluster_rep for cluster_rep, poses in cluster_map.items() for pose in poses}
    inverted_map.update({cluster_rep: cluster_rep for cluster_rep in cluster_map})  # to add all outliers

    return inverted_map
