import os
import sys
from itertools import chain


def list_diff(li1, li2):
    return list(list(set(li1)-set(li2)) + list(set(li2)-set(li1)))


def main():
    clust_rmsd_thresh = 1.0

    # Get All to All IRMSD Text File Path
    rmsd_file_path = sys.argv[1]
    rmsd_file = open(rmsd_file_path, "r")
    rmsd_file_lines = rmsd_file.readlines()
    rmsd_file.close()

    # Reference Structure (i.e. Crystal, EM or Rosetta Prediction) VS Docked Poses
    # IRMSD / Residue Level Summation Score Ranking Text File Path
    ref_vs_dockedposes_file_path = sys.argv[2]

    # Clustering Output Text File Path
    design_string = sys.argv[3]  # KM added
    clusters_output_file_path = os.path.dirname(rmsd_file_path) + "/%s_clustered.txt" % design_string

    # Create Python Dictionary Containing all Structure Names (Keys)
    # And a List of Neighbors within RMSD Threshold (Values)
    rmsd_dict = {}
    for line in rmsd_file_lines:
        line = line.rstrip()
        line = line.split()

        if line[0] in rmsd_dict:
            if float(line[2]) <= clust_rmsd_thresh:
                rmsd_dict[line[0]].append(line[1])
        else:
            if float(line[2]) <= clust_rmsd_thresh:
                rmsd_dict[line[0]] = [line[1]]
            else:
                rmsd_dict[line[0]] = []

        if line[1] in rmsd_dict:
            if float(line[2]) <= clust_rmsd_thresh:
                rmsd_dict[line[1]].append(line[0])
        else:
            if float(line[2]) <= clust_rmsd_thresh:
                rmsd_dict[line[1]] = [line[0]]
            else:
                rmsd_dict[line[1]] = []

    # Cluster
    all_claimed_structures = []
    return_clusters = []
    flattened_query = list(chain.from_iterable(rmsd_dict.values()))
    # flattened_query = np.concatenate(rmsd_dict.values()).tolist()
    while flattened_query != list():

        # Find Structure With Most Neighbors within RMSD Threshold
        max_neighbor_structure = None
        max_neighbor_count = 0
        for query_structure in rmsd_dict:
            neighbor_count = len(rmsd_dict[query_structure])
            if neighbor_count > max_neighbor_count:
                max_neighbor_structure = query_structure
                max_neighbor_count = neighbor_count

        # Create Cluster Containing Max Neighbor Structure (Cluster Representative) and its Neighbors
        cluster = rmsd_dict[max_neighbor_structure]
        return_clusters.append((max_neighbor_structure, cluster))

        # Remove Claimed Structures from rmsd_dict
        claimed_structures = [max_neighbor_structure] + cluster

        # Keep Track of all Claimed Structures
        all_claimed_structures.extend(claimed_structures)

        updated_dict = {}
        for query_structure in rmsd_dict:
            if query_structure not in claimed_structures:
                tmp_list = []
                for idx in rmsd_dict[query_structure]:
                    if idx not in claimed_structures:
                        tmp_list.append(idx)
                updated_dict[query_structure] = tmp_list
            else:
                updated_dict[query_structure] = []

        rmsd_dict = updated_dict

        flattened_query = list(chain.from_iterable(rmsd_dict.values()))
        # flattened_query = np.concatenate(rmsd_dict.values()).tolist()

    # Single Member Clusters
    unclaimed_structure_names = list_diff(all_claimed_structures, rmsd_dict.keys())
    solo_clusters = [(structure_name, []) for structure_name in unclaimed_structure_names]

    # Add Single Member Clusters to return_clusters
    return_clusters.extend(solo_clusters)

    # Read in Reference Structure VS Docked Poses IRMSD / Residue Level Summation Score Ranking Text File
    with open(ref_vs_dockedposes_file_path, "r") as ref_vs_dockedposes_file:
        ref_vs_dock_dict = {}
        for ref_vs_dock_line in ref_vs_dockedposes_file.readlines():
            ref_vs_dock_line = ref_vs_dock_line.rstrip().split()

            pose_id = str(ref_vs_dock_line[0])
            ref_vs_pose_irmsd = float(ref_vs_dock_line[1])
            pose_score = float(ref_vs_dock_line[2])
            pose_score_rank = int(ref_vs_dock_line[3])

            ref_vs_dock_dict[pose_id] = (ref_vs_pose_irmsd, pose_score, pose_score_rank)

    # Write Out Clustering Results to a Text File

    clusters_with_refdock_info_dict = {}
    for cluster in return_clusters:
        cluster_representative_id = cluster[0]
        clust_rep_ref_vs_pose_irmsd, clust_rep_pose_score, clust_rep_pose_score_rank = ref_vs_dock_dict[cluster_representative_id]
        representative_tuple = (cluster_representative_id, clust_rep_ref_vs_pose_irmsd, clust_rep_pose_score, clust_rep_pose_score_rank)
        clusters_with_refdock_info_dict[representative_tuple] = []

        for cluster_member_id in cluster[1]:
            clust_memb_ref_vs_pose_irmsd, clust_memb_pose_score, clust_memb_pose_score_rank = ref_vs_dock_dict[cluster_member_id]
            clusters_with_refdock_info_dict[representative_tuple].append((cluster_member_id, clust_memb_ref_vs_pose_irmsd, clust_memb_pose_score, clust_memb_pose_score_rank))

    clusters_sorted_by_clust_rep_pose_score = sorted(clusters_with_refdock_info_dict.items(), key=lambda kv: kv[0][2], reverse=True)

    clusters_output_file = open(clusters_output_file_path, "w")
    cluster_score_rank = 0
    for cluster in clusters_sorted_by_clust_rep_pose_score:
        cluster_score_rank += 1
        cluster_rep_info = cluster[0]
        # Column headers are cluster_rep_id, ref_vs_pose_irmsd, pose_score, pose_score_rank, cluster rank
        clust_rep_str = "REP {:35s} {:8.3f} {:8.3f} {:10d} {:10d}\n".format(cluster_rep_info[0], cluster_rep_info[1], cluster_rep_info[2], cluster_rep_info[3], cluster_score_rank)
        clusters_output_file.write(clust_rep_str)

        for cluster_member_info in cluster[1]:
            clust_memb_str = "### {:35s} {:8.3f} {:8.3f} {:10d}\n".format(cluster_member_info[0], cluster_member_info[1], cluster_member_info[2], cluster_member_info[3])
            clusters_output_file.write(clust_memb_str)

        clusters_output_file.write("\n")

    clusters_output_file.close()


if __name__ == '__main__':
    main()
