import os
import math
import pickle
import copy

# GLOBALS
bio_fragmentDB = '/home/kmeador/yeates/fragment_database/bio'


def get_all_base_root_paths(directory):
    dir_paths = []
    for root, dirs, files in os.walk(directory):
        if not dirs:
            dir_paths.append(root)

    return dir_paths


def parameterize_frag_length(length):
    divide_2 = math.floor(length / 2)
    modulus = length % 2
    if modulus == 1:
        upper_bound = 0 + divide_2
        lower_bound = 0 - divide_2
    else:
        upper_bound = 0 + divide_2
        lower_bound = upper_bound - length + 1
    offset_to_one = -lower_bound + 1

    return lower_bound, upper_bound, offset_to_one


def return_cluster_id_string(cluster_rep, index_number=3):
    while len(cluster_rep) < 3:
        cluster_rep += '0'
    if len(cluster_rep.split('_')) != 3:
        index = [cluster_rep[:1], cluster_rep[1:2], cluster_rep[2:]]
    else:
        index = cluster_rep.split('_')

    info = []
    n = 0
    for i in range(index_number):
        info.append(index[i])
        n += 1
    while n < 3:
        info.append('0')
        n += 1

    return '_'.join(info)


def get_cluster_dicts(info_db=bio_fragmentDB, id_list=None):
    # generate an interface specific scoring matrix from the fragment library
    # assuming residue_cluster_id_list has form [(1_2_24, [78, 87]), ...]
    if id_list is None:
        directory_list = get_all_base_root_paths(info_db)
    else:
        directory_list = []
        for c_id in id_list:
            directory = os.path.join(info_db, c_id[0], c_id[0] + '_' + c_id[1], c_id[0] + '_' + c_id[1] + '_' + c_id[2])
            directory_list.append(directory)

    cluster_dict_dict = {}
    for cluster in directory_list:
        filename = os.path.join(cluster, os.path.basename(cluster) + '.pkl')
        with open(filename, 'rb') as f:
            cluster_dict_dict[os.path.basename(cluster)] = pickle.load(f)

    # OUTPUT format: {'1_2_45': {'size': ..., 'rmsd': ..., 'rep': ..., 'mapped': ..., 'paired': ...}, ...}
    return cluster_dict_dict


def get_cluster_info(cluster_dict, frag_size=5):
    # Format: cluster_dict = [{'size': cluster_member_count, 'rmsd': mean_cluster_rmsd, 'rep': str(cluster_rep),
    #                          'mapped': mapped_freq_dict, 'paired': partner_freq_dict}
    # 'mapped'/'paired' format = {-2: {'A': 0.23, 'C': 0.01, ..., 'stats': [12, 0.37]}, -1:...}
    # 'stats'[0] is # of fragments in cluster, 'stats'[1] is weight of fragment index
    def get_cluster(cluster_rep):
        strings = cluster_rep.split('_')
        info = []
        grab = False
        for string in strings:
            if grab:
                info.append(string.strip('.pdb'))
            if string == 'fragtype':
                grab = True
            elif string == 'orientationtype':
                grab = True
            else:
                grab = False

        return '_'.join(info)

    m = 'mapped'
    p = 'paired'
    cluster_stats = {}
    zero = False
    max_index = None
    # zeros = {'mapped': 0, 'paired': 0}
    for i in [m, p]:
        rep = cluster_dict['rep']
        cluster = get_cluster(rep)
        weight, max_weight, count = 0.0, 0.0, 0
        for frag_index in cluster_dict[i]:
            if cluster_dict[i][frag_index]['stats'][1] != 0.0:
                count += 1
                if cluster_dict[i][frag_index]['stats'][1] > max_weight:
                    max_weight = cluster_dict[i][frag_index]['stats'][1]
                    max_index = frag_index
            weight += cluster_dict[i][frag_index]['stats'][1]

        if count == 0:
            zero = True
            # zeros[i] += 1
            # print('No weights:', cluster, i)
            present = 0.0
        else:
            present = weight/count
        cluster_stats[i] = [present, max_index]

    return cluster_stats, zero


def recurse_weights(_dict):
    start_key = next(iter(_dict))
    frag_d = {k: 0 for k in range(lower, upper + 1)}
    info = [0.0, 0.0, frag_d, copy.copy(frag_d)]  # , None: 0
    prior_list = ['', [], []]
    last = [start_key.split('_')[0], start_key.split('_')[1], '']
    count = [0, 0, 0]
    count_dict = {}
    start_index = 2

    def func(key, key_index, double=False):
        key_index -= 1
        if key.split('_')[key_index] == last[key_index]:
            if key_index > 0:
                func(key, key_index)
            count[key_index] += 1
        else:
            if key_index > 0:
                func(key, key_index, double=True)
            new_key = last[key_index - 1] + last[key_index]
            count_dict[new_key] = [copy.deepcopy(info), count[key_index]]
            if prior_list[len(new_key)] != list():
                # Remove old list counts
                # for index in range(len(count_dict[new_key][0])):
                for index in range(2):
                    for prior in prior_list[len(new_key)]:
                        count_dict[new_key][0][index] -= count_dict[prior][0][index]
                # Remove old dict counts
                for index in range(2, 4):
                    for prior in prior_list[len(new_key)]:
                        # Index in info dictionary
                        for frag_index in range(-2, 3):
                            count_dict[new_key][0][index][frag_index] -= count_dict[prior][0][index][frag_index]

            prior_list[len(new_key)].append(new_key)
            if not double:
                last[key_index] = key.split('_')[key_index]
                last[key_index - 1] = key.split('_')[key_index - 1]
            count[key_index] = 1

    for i in _dict:
        func(i, start_index)
        info[0] += _dict[i]['mapped'][0]
        info[1] += _dict[i]['paired'][0]
        info[2][_dict[i]['mapped'][1]] += 1
        info[3][_dict[i]['paired'][1]] += 1
        count[2] += 1
    func(start_key, start_index)
    count_dict['all'] = [info, count[2]]

    return count_dict


def main(database):
    all_cluster_dicts = get_cluster_dicts(info_db=database)
    all_cluster_zeros = []
    no_weights = []
    for cluster in all_cluster_dicts:
        all_cluster_dicts[cluster], check = get_cluster_info(all_cluster_dicts[cluster])
        if check:
            all_cluster_zeros.append(cluster)
            no_weights.append(cluster)

    all_dict_keys_sorted = sorted(all_cluster_dicts)
    total = len(all_dict_keys_sorted)
    for key in all_cluster_zeros:
        all_dict_keys_sorted.remove(key)
    print(total, '-', len(all_cluster_zeros), '=', len(all_dict_keys_sorted))
    print('Clusters removed with zero weight:', sorted(no_weights))

    sorted_cluster_dicts = {}
    for key in all_dict_keys_sorted:
        sorted_cluster_dicts[key] = all_cluster_dicts[key]

    all_cluster_total_weight_counts = recurse_weights(sorted_cluster_dicts)
    for cluster_id in all_cluster_total_weight_counts:
        for index in range(2):
            all_cluster_total_weight_counts[cluster_id][0][index] = round(
                all_cluster_total_weight_counts[cluster_id][0][index] / all_cluster_total_weight_counts[cluster_id][1],
                3)

    final_weight_counts = {return_cluster_id_string(key): all_cluster_total_weight_counts[key]
                           for key in sorted(all_cluster_total_weight_counts)}
    print(final_weight_counts)

    with open(os.path.join(database, 'statistics.pkl'), 'wb') as f:
        pickle.dump(final_weight_counts, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fragment_length = 5
    rmsd_thresh = 1
    outdir = os.path.join(os.getcwd(), 'ijk_clusters')
    info_db = os.path.join(outdir, 'info_' + str(rmsd_thresh))
    lower, upper, offset = parameterize_frag_length(fragment_length)
    main(info_db)
