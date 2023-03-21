from __future__ import annotations

import logging
import os
import subprocess
import sys
from itertools import combinations, repeat
from typing import Iterable, AnyStr, Any, Sequence
import warnings

import numpy as np
import pandas as pd
import sklearn

from .pose import PoseJob
from symdesign.resources.job import job_resources_factory
from symdesign.structure.coords import superposition3d, transform_coordinate_sets
from symdesign import utils, flags
putils = utils.path

logger = logging.getLogger(__name__)


def cluster_poses(pose_jobs: list[PoseJob]):
    # -> dict[str | PoseJob, list[str | PoseJob]] | None
    job = job_resources_factory.get()
    pose_cluster_map: dict[str | PoseJob, list[str | PoseJob]] = {}
    """Mapping which takes the format:
    {pose_string: [pose_string, ...]} where keys are representatives, values are matching designs
    """
    results = []
    mode_map_description = 'the representative is the most similar to all members of the cluster'
    if job.cluster.mode == 'ialign':
        mode_map_description = 'the representative is randomly chosen amongst the members'
        # Measure the alignment of all selected pose_jobs
        # all_files = [pose_job.source_file for pose_job in pose_jobs]

        # Need to change directories to prevent issues with the path length being passed to ialign
        prior_directory = os.getcwd()
        os.chdir(job.data_path)  # os.path.join(job.data_path, 'ialign_output'))
        temp_file_dir = os.path.join(os.getcwd(), 'temp')
        putils.make_path(temp_file_dir)

        # Save the interface for each design to the temp directory
        design_interfaces = []
        for pose_job in pose_jobs:
            pose_job.identify_interface()  # calls design.load_pose()
            # Todo this doesn't work for asymmetric Poses
            interface = pose_job.pose.get_interface()
            design_interfaces.append(
                # interface.write(out_path=os.path.join(temp_file_dir, f'{pose_job.name}_interface.pdb')))  # Todo reinstate
                interface.write(out_path=os.path.join(temp_file_dir, f'{pose_job.name}.pdb')))

        design_directory_pairs = list(combinations(pose_jobs, 2))
        if job.multi_processing:
            results = utils.mp_starmap(ialign, combinations(design_interfaces, 2), processes=job.cores)
        else:
            for idx, interface_files in enumerate(combinations(design_interfaces, 2)):
                results.append(ialign(*interface_files))
                #                                     out_path=os.path.join(job.data_path, 'ialign_output'))

        if results:
            # Separate interfaces which fall below a threshold
            is_threshold = 0.4  # 0.5  # TODO
            pose_pairs = []
            for idx, is_score in enumerate(results):
                if is_score > is_threshold:
                    pose_pairs.append(set(design_directory_pairs[idx]))
                    # pose_pairs.append({design1, design2})

            # Cluster all those designs together that are in alignment
            # Add both orientations to the pose_cluster_map
            for pose1, pose2 in pose_pairs:
                cluster1 = pose_cluster_map.get(pose1)
                try:
                    cluster1.append(pose2)
                except AttributeError:
                    pose_cluster_map[pose1] = [pose2]

                cluster2 = pose_cluster_map.get(pose2)
                try:
                    cluster2.append(pose1)
                except AttributeError:
                    pose_cluster_map[pose2] = [pose1]

        # Return to prior directory
        os.chdir(prior_directory)
    elif job.cluster.mode == 'transform':
        # First, identify the same compositions
        compositions: dict[tuple[str, ...], list[PoseJob]] = \
            group_compositions(pose_jobs)
        if job.multi_processing:
            results = utils.mp_map(cluster_pose_by_transformations, compositions.values(), processes=job.cores)
        else:
            for composition_group in compositions.values():
                results.append(cluster_pose_by_transformations(composition_group))

        # Add all clusters to the pose_cluster_map
        for result in results:
            pose_cluster_map.update(result.items())
    elif job.cluster.mode == 'rmsd':
        logger.critical(f"The mode {job.mode} hasn't been thoroughly debugged")
        # First, identify the same compositions
        compositions: dict[tuple[str, ...], list[PoseJob]] = \
            group_compositions(pose_jobs)
        # pairs_to_process = [grouping for entity_tuple, pose_jobs in compositions.items()
        #                     for grouping in combinations(pose_jobs, 2)]
        # composition_pairings = [combinations(pose_jobs, 2) for entity_tuple, pose_jobs in compositions.items()]
        # Find the rmsd between a pair of poses
        if job.multi_processing:
            results = utils.mp_map(pose_pair_by_rmsd, compositions.items(), processes=job.cores)
        else:
            for entity_tuple, pose_jobs in compositions.items():
                results.append(pose_pair_by_rmsd(pose_jobs))

        # Add all clusters to the pose_cluster_map
        for result in results:
            pose_cluster_map.update(result.items())
    # else:
    #     print(f"{job.cluster.mode} isn't a viable mode")
    #     sys.exit()

    if pose_cluster_map:
        if job.cluster.as_objects:
            pass  # They are by default objects
        else:
            for representative in list(pose_cluster_map.keys()):
                # Remove old entry and convert all arguments to pose_id strings, saving as pose_id strings
                pose_cluster_map[str(representative)] = \
                    [str(member) for member in pose_cluster_map.pop(representative)]

        if not job.output_file:
            job.output_file = putils.default_clustered_pose_file.format(utils.starttime, job.input_source)
        #     if len(job.output_file.split(os.sep)) <= 1:
        #         # The path isn't an absolute or relative path, so prepend the job.clustered_poses location
        #         job.output_file = os.path.join(job.clustered_poses, job.output_file)
        # else:
        # Prepend the job.clustered_poses location
        job.output_file = os.path.join(job.clustered_poses, job.output_file)

        job.cluster.map = utils.pickle_object(pose_cluster_map, name=job.output_file, out_path='')
        # elif job.module == flags.cluster_poses:
        logger.info('Clustering analysis results in the following similar poses:\nRepresentatives\n\tMembers\n')
        for representative, members, in pose_cluster_map.items():
            print(f'{representative}\n\t%s' % '\n\t'.join(map(str, members)))
        logger.info(f'Found {len(pose_cluster_map)} unique clusters from {len(pose_jobs)} pose inputs. '
                    f'All clusters wrote to: {job.cluster.map}')
        logger.info('Each cluster above has one representative which identifies with each of the members. For '
                    f'{job.cluster.mode}, {mode_map_description}.')
        logger.info(f'To utilize the clustering during {flags.select_poses}, provide the option --{flags.cluster_map}. '
                    'This will apply clustering to poses to select a cluster representative based on the most favorable'
                    ' cluster member')

        return
        # return pose_cluster_map

    logger.warning('No significant clusters were located. Clustering ended')
    sys.exit()
    # return None


# Used with single argment for mp_map
# def pose_pair_rmsd(pair: tuple[PoseJob, PoseJob]) -> float:
#     """Calculate the rmsd between pairs of Poses using CB coordinates. Must be the same length pose
#
#     Args:
#         pair: Paired PoseJob objects from pose processing directories
#     Returns:
#         RMSD value
#     """
#     # This focuses on all residues, not any particular set of residues
#     return superposition3d(*[pose.pose.cb_coords for pose in pair])[0]


def pose_pair_rmsd(pose1: PoseJob, pose2: PoseJob) -> float:
    """Calculate the rmsd between pairs of Poses using CB coordinates. Must be the same length pose

    Args:
        pose1: First PoseJob object
        pose2: Second PoseJob object
    Returns:
        RMSD value
    """
    # This focuses on all residues, not any particular set of residues
    rmsd, rot, tx = superposition3d(pose1.pose.cb_coords, pose2.pose.cb_coords)
    return rmsd


def pose_pair_by_rmsd(compositions: Iterable[Sequence[PoseJob]]) -> dict[str | PoseJob, list[str | PoseJob]]:
    """Perform rmsd comparison for all compositions of PoseJob instances

    Args:
        compositions: Groups of PoseJob instances that should be measured against one another pairwise
    Returns:
        {PoseJob representative: [PoseJob members], ... }
    """
    for pose_jobs in compositions:
        # Make all PoseJob combinations for this pair
        pose_job_pairs = list(combinations(pose_jobs, 2))
        results = [pose_pair_rmsd(*pair) for pair in pose_job_pairs]
        # Add all identical comparison results (all rmsd are 0 as they are with themselves
        results.extend(list(repeat(0, len(pose_jobs))))
        # Add all identical PoseJob combinations to pose_job_pairs
        pose_job_pairs.extend(list(zip(pose_jobs, pose_jobs)))

        return cluster_poses_by_value(pose_job_pairs, results)


# def ialign(pdb_file1: AnyStr, pdb_file2: AnyStr, chain1: str = None, chain2: str = None,
def ialign(*pdb_files: AnyStr, chain1: str = None, chain2: str = None,
           out_path: AnyStr = os.path.join(os.getcwd(), 'ialign')) -> float:
    """Run non-sequential iAlign on two .pdb files

    Args:
        pdb_files:
        # pdb_file1:
        # pdb_file2:
        chain1:
        chain2:
        out_path: The path to write iAlign results to
    Returns:
        The IS score from Mu & Skolnic 2010
    """
    if chain1 is None:
        chain1 = 'AB'
    if chain2 is None:
        chain2 = 'AB'
    chains = ['-c1', chain1, '-c2', chain2]

    pdb_file1, pdb_file2, *_ = pdb_files
    temp_pdb_file1 = os.path.join(os.getcwd(), 'temp',
                                  os.path.basename(pdb_file1.translate(utils.keep_digit_table)))
    temp_pdb_file2 = os.path.join(os.getcwd(), 'temp',
                                  os.path.basename(pdb_file2.translate(utils.keep_digit_table)))
    # Move the desired files to a temporary file location
    os.system(f'scp {pdb_file1} {temp_pdb_file1}')
    os.system(f'scp {pdb_file2} {temp_pdb_file2}')
    # Perform the iAlign process
    # Example: perl ../bin/ialign.pl -w output -s -a 0 1lyl.pdb AC 12as.pdb AB | grep "IS-score = "
    cmd = ['perl', putils.ialign_exe_path, '-s', '-w', out_path, '-p1', temp_pdb_file1, '-p2', temp_pdb_file2] + chains
    logger.debug(f'iAlign command: {subprocess.list2cmdline(cmd)}')
    ialign_p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ialign_out, ialign_err = ialign_p.communicate()
    # Format the output
    # Example: IS-score = 0.38840, P-value = 0.3808E-003, Z-score =  7.873
    grep_p = subprocess.Popen(['grep', 'IS-score = '], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ialign_is_score, err = grep_p.communicate(input=ialign_out)
    ialign_is_score = ialign_is_score.decode()
    logger.debug(f'iAlign interface alignment: {ialign_is_score.strip()}')
    is_score, pvalue, z_score = [score.split('=')[-1].strip() for score in ialign_is_score.split(',')]
    try:
        is_score = float(is_score)
    except ValueError:  # is_score isn't a number
        logger.debug('No significant interface found')
        is_score = 0.

    return is_score
    # return 0., \
    #     os.path.splitext(os.path.basename(pdb_file1))[0], os.path.splitext(os.path.basename(pdb_file2))[0]


def cluster_poses_by_value(identifier_pairs: Iterable[tuple[Any, Any]], values: Iterable[float], epsilon: float = 1.) \
        -> dict[str | PoseJob, list[str | PoseJob]]:
    """Take pairs of identifiers and a precomputed distance metric (such as RMSD) and cluster using DBSCAN algorithm

    Args:
        identifier_pairs: The identifiers for each pair measurement
        values: The corresponding measurement values for each pair of identifiers
        epsilon: The parameter for DBSCAN to influence the spread of clusters, needs to be tuned for measurement values
    Returns:
        {PoseJob representative: [PoseJob members], ... }
    """
    # BELOW IS THE INPUT FORMAT I WANT FOR cluster_poses_by_value()
    # index = list(combinations(pose_jobs, 2)) + list(zip(pose_jobs, pose_jobs))
    # values = values + tuple(repeat(0, len(pose_jobs)))
    # pd.Series(values, index=pd.MultiIndex.from_tuples(index)).unstack()

    pair_df = pd.Series(values, index=pd.MultiIndex.from_tuples(identifier_pairs)).fillna(0.).unstack()
    # symmetric_pair_values = sym(pair_df.values)

    # PCA analysis of distances
    # building_block_rmsd_matrix = sklearn.preprocessing.StandardScaler().fit_transform(symmetric_pair_values)
    # pca = PCA(putils.default_pca_variance)
    # building_block_rmsd_pc_np = pca.fit_transform(building_block_rmsd_matrix)
    # pca_distance_vector = pdist(building_block_rmsd_pc_np)
    # epsilon = pca_distance_vector.mean() * 0.5
    # Compute pose clusters using DBSCAN algorithm
    # precomputed specifies that a precomputed distance matrix is being passed
    dbscan = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=2, metric='precomputed')
    dbscan.fit(utils.sym(pair_df.to_numpy()))
    # find the cluster representative by minimizing the cluster mean
    cluster_ids = set(dbscan.labels_)
    # print(dbscan.labels_)
    # Use of dbscan.core_sample_indices_ returns all core_samples which is not a nearest neighbors mean index
    # print(dbscan.core_sample_indices_)
    outlier = -1
    try:
        cluster_ids.remove(outlier)  # Remove outlier label, will add all these later
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
        # pose_cluster_members = pair_df.index[iloc_indices].tolist()
        clustered_poses[pair_df.index[cluster_representative_idx]] = pair_df.index[iloc_indices].tolist()

    # Add all outliers to the clustered poses as a representative
    outlier_poses = pair_df.index[np.where(dbscan.labels_ == outlier)]
    clustered_poses.update(dict(zip(outlier_poses, outlier_poses)))

    return clustered_poses


number_of_coordinate_values = 9


def apply_transform_groups_to_guide_coordinates(*transforms: tuple[dict[str: np.ndarray]]) -> list[np.ndarray]:
    """For each incoming transformation, transform guide coordinates according to the specified transformations

    Args:
        transforms: The individual transformation groups that should be applied to a guide coordinate
    Returns:
        Guide coordinates transformed for each passed transform in each passed transform group
    """
    # Make a blank set of guide coordinates for each incoming transformation
    # number_of_coordinate_values = 9
    guide_coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    try:
        allowed_keys = ['rotation', 'translation']
        operation_lengths = []
        for key in allowed_keys:
            operation = transforms[0].get(key)
            if operation is not None:
                operation_lengths.append(len(operation))
        try:
            tiled_length = max(operation_lengths)
        except ValueError:  # operation_lengths is empty
            raise KeyError(
                f'{apply_transform_groups_to_guide_coordinates.__name__}: Must pass one of the values '
                f'{" or ".join(allowed_keys)}')

        tiled_guide_coords = np.tile(guide_coords, (tiled_length, 1, 1))
    except IndexError:  # transforms[0] failed
        raise IndexError(
            f'{apply_transform_groups_to_guide_coordinates.__name__}: No arguments passed for transforms')

    transformed_guide_coords_sets = \
        [transform_coordinate_sets(tiled_guide_coords, **transform) for transform in transforms]

    return transformed_guide_coords_sets
    # return np.concatenate(transformed_guide_coords_sets, axis=1)


def cluster_transformation_pairs(*transforms: tuple[dict[str, np.ndarray]], distance: float = 1.,
                                 minimum_members: int = 2) \
        -> tuple[sklearn.neighbors._unsupervised.NearestNeighbors, sklearn.cluster._dbscan.DBSCAN]:
    """Cluster a group of transformation parameters sets to find those which occupy essentially the same space

    Args:
        transforms: Group containing multiple sets of transformation operations where each transformation operation set
            takes the form {'rotation': rot_array, 'translation': tx_array,
                            'rotation2': rot2_array, 'translation2': tx2_array}
        distance: The distance to query neighbors in transformational space
        minimum_members: The minimum number of members in each cluster
    Returns:
        The sklearn tree with the calculated nearest neighbors, the DBSCAN clustering object
        Representative indices, DBSCAN cluster membership indices
    """
    transformed_guide_coord_pairs = apply_transform_groups_to_guide_coordinates(*transforms)
    transformed_guide_coords = np.concatenate(
        [coords.reshape(-1, number_of_coordinate_values) for coords in transformed_guide_coord_pairs], axis=1)

    # Create a tree structure describing the distances of all transformed points relative to one another
    nearest_neightbors_ball_tree = sklearn.neighbors.NearestNeighbors(algorithm='ball_tree', radius=distance)
    nearest_neightbors_ball_tree.fit(transformed_guide_coords)
    # sort_results only returns non-zero entries with the smallest distance first, however it doesn't seem to work...?
    distance_graph = nearest_neightbors_ball_tree.radius_neighbors_graph(mode='distance', sort_results=True)
    #                                                                    X=transformed_guide_coords is implied
    # Because this doesn't work to sort_results and pull out indices, I have to do another step 'radius_neighbors'
    # Todo Why is this happening? Perhaps when the precomputed data is too small?
    # Caution /home/kylemeador/miniconda3/envs/dev/lib/python3.10/site-packages/sklearn/neighbors/_base.py:206:
    # EfficiencyWarning: Precomputed sparse input was not sorted by data.
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
    with warnings.catch_warnings():
        # Empty slices can't compute mean, so catch warning if cluster is an outlier
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_cluster_dist = np.array([tree_distance.mean() for tree_distance in list(tree_distances)])

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


def cluster_pose_by_transformations(compositions: list[PoseJob], **kwargs) -> dict[str | PoseJob, list[str | PoseJob]]:
    """From a group of poses with matching protein composition, cluster the designs according to transformational
    parameters to identify the unique poses in each composition

    Args:
        compositions: The group of PoseJob objects to pull transformation data from
    Keyword Args:
        distance: float = 1. - The distance to query neighbors in transformational space
        minimum_members: int = 2 - The minimum number of members in each cluster
    Returns:
        Cluster with representative pose as the key and matching poses as the values
    """
    # Format transforms for the selected compositions
    stacked_transforms1, stacked_transforms2 = list(zip(pose_jobs.pose_transformation
                                                        for pose_jobs in compositions))
    trans1_rot1, trans1_tx1, trans1_rot2, trans1_tx2 = \
        zip(*[transform.values() for transform in stacked_transforms1])
    trans2_rot1, trans2_tx1, trans2_rot2, trans2_tx2 = \
        zip(*[transform.values() for transform in stacked_transforms2])

    # Must add a new axis to translations so the operations are broadcast together in transform_coordinate_sets()
    transformation1 = {'rotation': np.array(trans1_rot1), 'translation': np.array(trans1_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans1_rot2), 'translation2': np.array(trans1_tx2)[:, np.newaxis, :]}
    transformation2 = {'rotation': np.array(trans2_rot1), 'translation': np.array(trans2_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans2_rot2), 'translation2': np.array(trans2_tx2)[:, np.newaxis, :]}

    # Find the representatives of the cluster based on minimal distance of each point to its nearest neighbors
    return cluster_by_transformations(compositions, transformation1, transformation2, **kwargs)
    # cluster_representative_indices, cluster_labels = \
    #     find_cluster_representatives(*cluster_transformation_pairs(transformation1, transformation2, **kwargs))
    #
    # representative_labels = cluster_labels[cluster_representative_indices]
    #
    # # Pull out pose's from the input compositions group
    # outlier = -1
    # composition_map = \
    #     {compositions[rep_idx]: [compositions[idx] for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
    #      for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels) if rep_label != outlier}
    # # Add all outliers
    # composition_map.update({compositions[idx]: [compositions[idx]]
    #                         for idx in np.flatnonzero(cluster_labels == outlier).tolist()})
    #
    # return composition_map


def cluster_by_transformations(*transforms: tuple[dict[str, np.ndarray]], values: list[Any] = None, **kwargs) \
        -> dict[Any, list[Any]]:
    """From a set of objects with associated transformational parameters, identify and cluster the unique objects by
    representatives and members

    Args:
        transforms: Group containing multiple sets of transformation operations where each transformation operation set
            takes the form {'rotation': rot_array, 'translation': tx_array,
                            'rotation2': rot2_array, 'translation2': tx2_array}
        values: The group of objects to cluster
    Keyword Args:
        distance: float = 1. - The distance to query neighbors in transformational space
        minimum_members: int = 2 - The minimum number of members in each cluster
    Returns:
        Clustered objects with representative as the key and members as the values
    """
    # Find the representatives of the cluster based on minimal distance of each point to its nearest neighbors
    # This section could be added to the Nanohedra docking routine
    cluster_representative_indices, cluster_labels = \
        find_cluster_representatives(*cluster_transformation_pairs(*transforms, **kwargs))

    representative_labels = cluster_labels[cluster_representative_indices]

    # Sort out clustered transform_values from the input transform_values
    outlier = -1
    cluster_map = \
        {values[rep_idx]: [values[idx] for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
         for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels)
         if rep_label != outlier}
    # Add all outliers
    cluster_map.update({values[idx]: [values[idx]] for idx in np.flatnonzero(cluster_labels == outlier).tolist()})

    return cluster_map


def group_compositions(pose_jobs: list[PoseJob]) -> dict[tuple[str, ...], list[PoseJob]]:
    """From a set of DesignDirectories, find all the compositions and group together

    Args:
        pose_jobs: The PoseJob to group according to composition
    Returns:
        List of similarly named PoseJob mapped to their name
    """
    compositions = {}
    for pose_job in pose_jobs:
        entity_names = tuple(pose_job.entity_names)
        found_composition = None
        for permutation in combinations(entity_names, len(entity_names)):
            found_composition = compositions.get(permutation, None)
            if found_composition:
                break

        if found_composition:
            compositions[entity_names].append(pose_job)
        else:
            compositions[entity_names] = [pose_job]

    return compositions


def invert_cluster_map(cluster_map: dict[Any, list[Any]]):
    """Return an inverted cluster map where the cluster members map to the representative

    Args:
        cluster_map: The standard pose_cluster_map format
    Returns:
        An inverted cluster_map where the members are keys and the representative is the value
    """
    inverted_map = {member: cluster_rep for cluster_rep, members in cluster_map.items() for member in members}
    # Add all representatives
    inverted_map.update({cluster_rep: cluster_rep for cluster_rep in cluster_map})

    return inverted_map
