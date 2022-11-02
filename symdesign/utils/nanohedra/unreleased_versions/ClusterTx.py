import numpy as np
from scipy.spatial.distance import euclidean


class ClusterTx:
    def __init__(self, tx_param_list, tx_param_dist_thresh=0.5, min_cluster_size=2):
        self.tx_param_list = tx_param_list
        self.tx_param_dist_thresh = tx_param_dist_thresh
        self.min_cluster_size = min_cluster_size

        self.clustered_tx_param_list = []
        self.top_clusters_tx_param_index_list = []
        self.top_representatives_tx_params = []
        self.top_representatives_tx_params_indices = []
        self.cluster_tx()

    def cluster_tx(self):
        # Calculate Euclidean distance for all pairs of translational parameters
        all_to_all_distances = []
        for i in range(len(self.tx_param_list) - 1):
            tx_param_i = self.tx_param_list[i]
            for j in range(i + 1, len(self.tx_param_list)):
                tx_param_j = self.tx_param_list[j]
                d_euclidean = euclidean(tx_param_i, tx_param_j)
                all_to_all_distances.append((i, j, d_euclidean))

        # Create python dictionary containing all indices of translational parameters from tx_param_list (Keys)
        # And a list of neighbors within a defined distance threshold (tx_param_dist_thresh) (Values)
        tx_param_dict = {}
        for dist in all_to_all_distances:
            if dist[0] in tx_param_dict:
                if dist[2] <= self.tx_param_dist_thresh:
                    tx_param_dict[dist[0]].append(dist[1])
            else:
                if dist[2] <= self.tx_param_dist_thresh:
                    tx_param_dict[dist[0]] = [dist[1]]
                else:
                    tx_param_dict[dist[0]] = []

            if dist[1] in tx_param_dict:
                if dist[2] <= self.tx_param_dist_thresh:
                    tx_param_dict[dist[1]].append(dist[0])
            else:
                if dist[2] <= self.tx_param_dist_thresh:
                    tx_param_dict[dist[1]] = [dist[0]]
                else:
                    tx_param_dict[dist[1]] = []

        # Cluster
        return_clusters = []
        flattened_query = np.concatenate(tx_param_dict.values()).tolist()
        while flattened_query != list():

            # Find structure with most neighbors within distance threshold
            max_neighbor_structure = None
            max_neighbor_count = 0
            for query_structure in tx_param_dict:
                neighbor_count = len(tx_param_dict[query_structure])
                if neighbor_count > max_neighbor_count:
                    max_neighbor_structure = query_structure
                    max_neighbor_count = neighbor_count

            # Create cluster with the maximum neighboring tx parameters (cluster representative) and its neighbors
            cluster = tx_param_dict[max_neighbor_structure]
            return_clusters.append((max_neighbor_structure, cluster))

            # Remove claimed translational parameters from tx_param_dict
            claimed_structures = [max_neighbor_structure] + cluster
            updated_dict = {}
            for query_structure in tx_param_dict:
                if query_structure not in claimed_structures:
                    tmp_list = []
                    for idx in tx_param_dict[query_structure]:
                        if idx not in claimed_structures:
                            tmp_list.append(idx)
                    updated_dict[query_structure] = tmp_list
                else:
                    updated_dict[query_structure] = []

            tx_param_dict = updated_dict

            flattened_query = np.concatenate(tx_param_dict.values()).tolist()

        clustered_tx_param_list = []
        top_clusters_tx_param_index_list = []
        top_representatives_tx_params = []
        top_representatives_tx_params_indices = []
        for return_cluster in return_clusters:
            clustered_tx_param_list.append((self.tx_param_list[return_cluster[0]],
                                            [self.tx_param_list[return_cluster[1][i]] for i in
                                             range(len(return_cluster[1]))]))
            if len(return_cluster[1]) >= self.min_cluster_size:
                top_clusters_tx_param_index_list.append(return_cluster)
                top_representatives_tx_params.append(self.tx_param_list[return_cluster[0]])
                top_representatives_tx_params_indices.append(return_cluster[0])

        self.clustered_tx_param_list = clustered_tx_param_list
        self.top_representatives_tx_params = top_representatives_tx_params
        self.top_representatives_tx_params_indices = top_representatives_tx_params_indices
        self.top_clusters_tx_param_index_list = top_clusters_tx_param_index_list

    def get_clustered_tx_param_list(self):
        return self.clustered_tx_param_list

    def get_top_representatives_tx_params(self):
        return self.top_representatives_tx_params

    def get_top_representatives_tx_params_indices(self):
        return self.top_representatives_tx_params_indices

    def get_top_clusters_tx_param_index_list(self):
        return self.top_clusters_tx_param_index_list

# TEST
# with open('/Users/jlaniado/Desktop/TxParams/tx_params.pkl', 'rb') as f:
#     param_list = pickle.load(f)
#
#
# cluster_tx = ClusterTx(param_list)
# l = cluster_tx.get_top_clusters_tx_param_index_list()
# for c in l:
#     print c
