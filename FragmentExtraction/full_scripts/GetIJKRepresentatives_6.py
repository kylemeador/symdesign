import os
import shutil
from itertools import chain
import FragUtils as Frag

# Globals
module = 'Get IJK Cluster Representatives:'


# def cluster_fragment_rmsds(rmsd_file_path):
#     # Get All to All RMSD File
#     with open(rmsd_file_path, 'r') as rmsd_file:
#         rmsd_file_lines = rmsd_file.readlines()
#
#     # Create Dictionary Containing Structure Name as Key and a List of Neighbors within RMSD Threshold as Values
#     rmsd_dict = {}
#     for line in rmsd_file_lines:
#         line = line.rstrip()
#         line = line.split()
#
#         if line[0] in rmsd_dict:
#             rmsd_dict[line[0]].append(line[1])
#         else:
#             rmsd_dict[line[0]] = [line[1]]
#
#         if line[1] in rmsd_dict:
#             rmsd_dict[line[1]].append(line[0])
#         else:
#             rmsd_dict[line[1]] = [line[0]]
#
#     print(module, 'Finished Creating RMSD Dictionary with a total of', len(rmsd_dict), 'Clusters')
#
#     # Cluster
#     return_clusters = []
#     flattened_query = list(chain.from_iterable(rmsd_dict.values()))
#     while flattened_query != list():
#         # Find Structure With Most Neighbors within RMSD Threshold
#         max_neighbor_structure = None
#         max_neighbor_count = 0
#         for query_structure in rmsd_dict:
#             neighbor_count = len(rmsd_dict[query_structure])
#             if neighbor_count > max_neighbor_count:
#                 max_neighbor_structure = query_structure
#                 max_neighbor_count = neighbor_count
#
#         # Create Cluster Containing Max Neighbor Structure (Cluster Representative) and its Neighbors
#         cluster = rmsd_dict[max_neighbor_structure]
#         return_clusters.append((max_neighbor_structure, cluster))
#
#         # Remove Claimed Structures from rmsd_dict
#         claimed_structures = [max_neighbor_structure] + cluster
#         updated_dict = {}
#         for query_structure in rmsd_dict:
#             if query_structure not in claimed_structures:
#                 tmp_list = []
#                 for idx in rmsd_dict[query_structure]:
#                     if idx not in claimed_structures:
#                         tmp_list.append(idx)
#                 updated_dict[query_structure] = tmp_list
#             else:
#                 updated_dict[query_structure] = []
#
#         rmsd_dict = updated_dict
#         flattened_query = list(chain.from_iterable(rmsd_dict.values()))
#
#     return return_clusters


def main():
    print(module, 'Beginning')
    minimum_clust_size = 4
    rmsd_thresh = 1

    # IJ Clustered Directory
    ij_mapped_dir = os.path.join(os.getcwd(), 'ij_mapped_paired_frags')

    # Output Directories
    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    db_outdir = os.path.join(outdir, 'db_' + str(rmsd_thresh))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(db_outdir):
        os.makedirs(db_outdir)

    # Cluster
    for root, dirs, files in os.walk(ij_mapped_dir):
        if not dirs:
            rmsd_file_path = os.path.join(root, 'all_to_all_guide_atom_rmsd.txt')
            i_dir = os.path.basename(root).split('_')[0]
            ij_dir = os.path.basename(root)
            ij_outdir = os.path.join(db_outdir, i_dir, ij_dir)

            if not os.path.exists(os.path.join(db_outdir, i_dir)):
                os.makedirs(os.path.join(db_outdir, i_dir))
            if not os.path.exists(ij_outdir):
                os.makedirs(ij_outdir)

            print(module, 'Starting on', os.path.basename(root))
            return_clusters = Frag.cluster_fragment_rmsds(rmsd_file_path)
            print(module, 'Clustering Finished, Creating Representatives')

            # Get Cluster Representatives
            cluster_count = 1
            for cluster in return_clusters:
                if len(cluster[1]) >= minimum_clust_size:
                    cluster_rep = cluster[0] + '.pdb'
                    cluster_rep_pdb_path = os.path.join(root, cluster_rep)

                    k_cluster_dirname = os.path.basename(ij_outdir) + '_' + str(cluster_count)
                    cluster_outdir = os.path.join(ij_outdir, k_cluster_dirname)
                    if not os.path.exists(cluster_outdir):
                        os.makedirs(cluster_outdir)

                    cluster_rep_name = cluster[0] + '_orientation_' + str(cluster_count) + '_representative.pdb'
                    shutil.copyfile(cluster_rep_pdb_path, os.path.join(cluster_outdir, cluster_rep_name))
                    cluster_count += 1

    print(module, 'Finished')


if __name__ == '__main__':
    main()
