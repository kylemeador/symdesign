import os
from itertools import combinations

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities

import SymDesignUtils as SDUtils


def pose_rmsd_mp(all_des_dirs, threads=1):
    """Map the RMSD for a Nanohedra output based on building block directory (ex 1abc_2xyz)

    Args:
        all_des_dirs (list[DesignDirectory]): List of relevant design directories
    Keyword Args:
        threads: Number of multiprocessing threads to run
    Returns:
        (dict): {building_blocks: {pair1: {pair2: rmsd, ...}, ...}, ...}
    """
    pose_map = {}
    pairs_to_process = []
    singlets = {}
    for pair in combinations(all_des_dirs, 2):
        # add all individual poses to a singles pool
        singlets[pair[0].building_blocks] = pair[0]
        if pair[0].building_blocks == pair[1].building_blocks:
            singlets.pop(pair[0].building_blocks)
            pairs_to_process.append(pair)
    # find the rmsd between a pair of poses.  multiprocessing to increase throughput
    _results = SDUtils.mp_map(pose_pair_rmsd, pairs_to_process, threads=threads)

    # Make dictionary with all pairs
    for pair, pair_rmsd in zip(pairs_to_process, _results):
        protein_pair_path = os.path.basename(pair[0].building_blocks)
        # protein_pair_path = pair[0].building_blocks
        # pose_map[result[0]] = result[1]
        if protein_pair_path in pose_map:
            # # {building_blocks: {(pair1, pair2): rmsd, ...}, ...}
            # {building_blocks: {pair1: {pair2: rmsd, ...}, ...}, ...}
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
    # protein_pair_path = pair[0].building_blocks
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
        if pair[0].building_blocks == pair[1].building_blocks:
            protein_pair_path = pair[0].building_blocks
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
                # {building_blocks: {(pair1, pair2): rmsd, ...}, ...}
                if str(pair[0]) in pose_map[protein_pair_path]:
                    pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
                    if str(pair[1]) not in pose_map[protein_pair_path]:
                        pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
                    # else:
                    #     print('\n' * 6 + 'NEVER ACCESSED' + '\n' * 6)
                    #     pose_map[pair[0].building_blocks][str(pair[1])][str(pair[1])] = 0.0
                # else:
                #     print('\n' * 6 + 'ACCESSED' + '\n' * 6)
                #     pose_map[pair[0].building_blocks][str(pair[0])] = {str(pair[1]): pair_rmsd}
                #     pose_map[pair[0].building_blocks][str(pair[0])][str(pair[0])] = 0.0
                # pose_map[pair[0].building_blocks][(str(pair[0]), str(pair[1]))] = pair_rmsd[2]
            else:
                pose_map[protein_pair_path] = {str(pair[0]): {str(pair[0]): 0.0}}
                pose_map[protein_pair_path][str(pair[0])][str(pair[1])] = pair_rmsd
                pose_map[protein_pair_path][str(pair[1])] = {str(pair[1]): 0.0}
                # pose_map[pair[0].building_blocks] = {(str(pair[0]), str(pair[1])): pair_rmsd[2]}

    return pose_map


def cluster_poses(pose_map):
    """Take a pose map calculated by pose_rmsd (_mp or _s) and cluster using DBSCAN algorithm

    Args:
        pose_map (dict): {building_blocks: {pair1: {pair2: rmsd, ...}, ...}, ...}
    Returns:
        (dict): {building_block: {'poses clustered'}, ... }
    """
    pose_cluster_map = {}
    for building_block in pose_map:
        building_block_rmsd_df = pd.DataFrame(pose_map[building_block]).fillna(0.0)

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

        # Compute the highest density cluster using DBSCAN algorithm
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
                                            if cluster == cluster_id] for cluster_id in cluster_ids}
        # print(cluster_members_map)
        # cluster_representative_map = {}
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
        pose_cluster_map[building_block] = clustered_poses

    return pose_cluster_map