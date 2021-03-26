import os
from itertools import combinations

import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import euclidean, pdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities

import SymDesignUtils as SDUtils
from utils.GeneralUtils import transform_coordinate_sets


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
    _results = SDUtils.mp_map(pose_pair_rmsd, pairs_to_process, threads=threads)

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
        pair (tuple[DesignDirectory, DesignDirectory]): Two DesignDirectory objects from pose processing directories
    Returns:
        (float): RMSD value
    """
    # protein_pair_path = pair[0].composition
    # Grab designed resides from the design_directory
    des_residue_list = [pose.info['design_residues'] for pose in pair]

    # Set up the list of residues undergoing design (interface) on each pair. Return the intersection
    # could use the union as well...?
    des_residue_set = SDUtils.index_intersection({pair[n]: set(pose_residues)
                                                  for n, pose_residues in enumerate(des_residue_list)})
    if not des_residue_set:  # when the two structures are not significantly overlapped
        return np.nan
    else:
        pdb_parser = PDBParser(QUIET=True)
        pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
        rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
                              if residue.get_id()[1] in des_residue_set] for structure in pair_structures]
        pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
                          for entity_list in rmsd_residue_list]

        return SDUtils.superimpose(pair_atom_list)

    # return pair_rmsd
    # return {protein_pair_path: {str(pair[0]): {str(pair[0]): pair_rmsd}}}


def pose_rmsd_s(all_des_dirs):
    pose_map = {}
    for pair in combinations(all_des_dirs, 2):
        if pair[0].composition == pair[1].composition:
            protein_pair_path = pair[0].composition
            # Grab designed resides from the design_directory
            des_residue_list = [pose.info['des_residues'] for pose in pair]
            # could use the union as well...
            des_residue_set = SDUtils.index_intersection({pair[n]: set(pose_residues)
                                                          for n, pose_residues in enumerate(des_residue_list)})
            if des_residue_set == list():  # when the two structures are not significantly overlapped
                pair_rmsd = np.nan
            else:
                pdb_parser = PDBParser(QUIET=True)
                # pdb = parser.get_structure(pdb_name, filepath)
                pair_structures = [pdb_parser.get_structure(str(pose), pose.asu) for pose in pair]
                # returns a list with all ca atoms from a structure
                # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].asu, pair[1].asu], SDUtils.get_biopdb_ca)
                # pair_atoms = SDUtils.get_rmsd_atoms([pair[0].path, pair[1].path], SDUtils.get_biopdb_ca)

                # pair should be a structure...
                # for structure in pair_structures:
                #     for residue in structure.residues:
                #         print(residue)
                #         print(residue[0])
                rmsd_residue_list = [[residue for residue in structure.residues  # residue.get_id()[1] is res number
                                      if residue.get_id()[1] in des_residue_set] for structure in pair_structures]

                # rmsd_residue_list = [[residue for residue in structure.residues
                #                       if residue.get_id()[1] in des_residue_list[n]]
                #                      for n, structure in enumerate(pair_structures)]

                # print(rmsd_residue_list)
                pair_atom_list = [[atom for atom in unfold_entities(entity_list, 'A') if atom.get_id() == 'CA']
                                  for entity_list in rmsd_residue_list]
                # [atom for atom in structure.get_atoms if atom.get_id() == 'CA']
                # pair_atom_list = SDUtils.get_rmsd_atoms(rmsd_residue_list, SDUtils.get_biopdb_ca)
                # pair_rmsd = SDUtils.superimpose(pair_atoms, threshold)

                pair_rmsd = SDUtils.superimpose(pair_atom_list)  # , threshold)
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
        building_block_rmsd_matrix = SDUtils.sym(building_block_rmsd_df.values)
        # print(building_block_rmsd_df.values)
        # print(building_block_rmsd_matrix)
        # building_block_rmsd_matrix = StandardScaler().fit_transform(building_block_rmsd_matrix)
        # pca = PCA(PUtils.variance)
        # building_block_rmsd_pc_np = pca.fit_transform(building_block_rmsd_matrix)
        # pca_distance_vector = pdist(building_block_rmsd_pc_np)
        # epsilon = pca_distance_vector.mean() * 0.5

        # Compute pose clusters using DBSCAN algorithm
        # logger.info('Finding pose clusters within RMSD of %f' % SDUtils.rmsd_threshold) # TODO
        dbscan = DBSCAN(eps=SDUtils.rmsd_threshold, min_samples=2, metric='precomputed')
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


def cluster_transformations(transform1, transform2, distance=1.0):
    """Cluster Pose conformations according to their specific transformation parameters to find Poses which occupy
    essentially the same space

    Args:
        transform1 (dict[mapping[str, numpy.ndarray]]): A set of rotations/translations to be clustered
        transform2 (dict[mapping[str, numpy.ndarray]]): A set of rotations/translations to be clustered
    Keyword Args:
        distance=1.0 (float): The distance to query neighbors in transformational space
    Returns:
        (tuple[numpy.ndarray(int), numpy.ndarray(int)]): Representative indices, cluster membership indices
    """
    # Todo tune DBSCAN distance (epsilon) to be reflective of the data, should be related to radius in NearestNeighbors
    #  but smaller by some amount. Ideal amount would be the distance between two transformed guide coordinate sets of
    #  a similar tx and a 3 degree step of rotation.
    # make a blank set of guide coordinates for each incoming transformation
    guide_coords = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), (len(transform1['rotation']), 1, 1))
    # transforms = {'rotation': rotation_array, 'translation': translation_array,
    #               'rotation2': rot2_array, 'translation2': tx2_array}
    # print(transform1['translation'])
    transformed_guide_coords1 = transform_coordinate_sets(guide_coords, **transform1)
    # print(transformed_guide_coords1)
    transformed_guide_coords_reshape1 = transformed_guide_coords1.reshape(-1, 9)

    # print(transformed_guide_coords_reshape1)
    transformed_guide_coords2 = transform_coordinate_sets(guide_coords, **transform2)
    # print(transformed_guide_coords2)
    # transformed_guide_coords = np.concatenate([transformed_guide_coords1, transformed_guide_coords2], axis=2)
    transformed_guide_coords = np.concatenate([transformed_guide_coords1.reshape(-1, 9),
                                               transformed_guide_coords2.reshape(-1, 9)], axis=1)
    # print(transformed_guide_coords)
    # create a tree structure describing the distances of all transformed points relative to one another
    transform_tree = NearestNeighbors(algorithm='ball_tree', radius=distance)
    transform_tree.fit(transformed_guide_coords)
    #                                sort_results returns only non-zero entries and provides the smallest distance first
    distance_graph = transform_tree.radius_neighbors_graph(mode='distance', sort_results=True)  # <- sort doesn't work?
    #                                                      X=transformed_guide_coords is implied
    # because this doesn't work to sort_results and pull out indices, I have to do another step 'radius_neighbors'
    cluster = DBSCAN(eps=distance, min_samples=2, metric='precomputed').fit(distance_graph)  # , sample_weight=SOME WEIGHTS?
    outlier = -1  # -1 are outliers in DBSCAN
    labels = set(cluster.labels_) - {outlier}  # labels live here
    tree_distances, tree_indices = transform_tree.radius_neighbors(sort_results=True)

    # find cluster mean for each index
    # print(tree_distances, tree_indices)
    # print(type(tree_distances), type(tree_indices))
    # mean_cluster_dist = tree_distances.mean()
    mean_cluster_dist = np.empty(tree_distances.shape[0])
    for idx, array in enumerate(tree_distances.tolist()):
        mean_cluster_dist[idx] = array.mean()  # empty slices can't have mean, so return np.nan if cluster is an outlier

    # for each label (cluster), add the minimal mean (representative) the representative transformation indices
    representative_transformation_indices = []
    for label in labels:
        cluster_indices = np.flatnonzero(cluster.labels_ == label)
        # print(mean_cluster_dist[np.flatnonzero(cluster.labels_ == label)])
        # print(cluster_indices[mean_cluster_dist[cluster_indices].argmin()])
        representative_transformation_indices.append(cluster_indices[mean_cluster_dist[cluster_indices].argmin()])
        # representative_transformation_indices.append(mean_cluster_dist[np.where(cluster.labels_ == label)].argmin())
    # add all outliers to representatives
    representative_transformation_indices.extend(np.flatnonzero(cluster.labels_ == outlier).tolist())

    return representative_transformation_indices, cluster.labels_


def cluster_designs(composition_group):
    """From a group of poses with matching protein composition, cluster the designs according to transformational
    parameters

    Args:
        (iterable[DesignDirectory]): The group of DesignDirectory objects to pull transformation data from
    Returns:
        (dict[mapping[DesignDirectoryID, list[DesignDirectoryID]]): Cluster with the representative as the key and
        matching poses as the values
    """
    # # First, identify the same compositions
    # compositions = {}
    # for design in design_directories:
    #     if compositions.get(design.composition, None):
    #         compositions[design.composition].append(design)
    #     else:
    #         compositions[design.composition] = [design]

    # Next, identify the unique poses in each composition
    # pose_cluster_map = {}
    # for composition_group in compositions.values():
    # format all transforms for the selected compositions
    stacked_transforms = [design_directory.pose_transformation() for design_directory in composition_group]
    # transformations1 = [transform[1].values() for transform in stacked_transforms]
    trans1_rot1, trans1_tx1, trans1_rot2, trans1_tx2 = zip(*[transform[1].values()
                                                             for transform in stacked_transforms])
    # must add a new axis to translations so the operations broadcast together
    transformation1 = {'rotation': np.array(trans1_rot1), 'translation': np.array(trans1_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans1_rot2), 'translation2': np.array(trans1_tx2)[:, np.newaxis, :]}

    trans2_rot1, trans2_tx1, trans2_rot2, trans2_tx2 = zip(*[transform[2].values()
                                                             for transform in stacked_transforms])
    transformation2 = {'rotation': np.array(trans2_rot1), 'translation': np.array(trans2_tx1)[:, np.newaxis, :],
                       'rotation2': np.array(trans2_rot2), 'translation2': np.array(trans2_tx2)[:, np.newaxis, :]}

    # rotation1, translation1, rotation2, translation2 = zip(*[design_directory.pose_transformation()
    #                                                          for design_directory in design_directories])
    # rotation1, translation1, rotation2, translation2 = stacked_transforms

    # This section could be added to the Nanohedra docking routine
    cluster_representative_indices, cluster_labels = cluster_transformations(transformation1, transformation2)
    representative_labels = cluster_labels[cluster_representative_indices]
    print(representative_labels)

    # pull out the pose-id from the input composition groups (DesignDirectory)
    composition_map = {str(composition_group[rep_idx]): [str(composition_group[idx])
                                                         for idx in np.flatnonzero(cluster_labels == rep_label).tolist()]
                       for rep_idx, rep_label in zip(cluster_representative_indices, representative_labels)
                       if rep_label != -1}  # don't add outliers (-1 labels) now
    # add the outliers as separate occurrences
    composition_map.update({str(composition_group[idx]): [] for idx in np.flatnonzero(cluster_labels == -1).tolist()})
    print(composition_map)
    return composition_map
    # pose_cluster_map.update(composition_map)

    # return pose_cluster_map


def invert_cluster_map(cluster_map):
    """Return an inverted cluster map where the cluster members map to the representative

    Args:
        cluster_map (dict[mapping[representative DesignDirectoryID, list[member DesignDirectoryID]])
    Returns:
        (dict[mapping[member DesignDirectoryID, representative DesignDirectoryID]])
    """
    return {pose: cluster_rep for cluster_rep, poses in cluster_map.items() for pose in poses}
