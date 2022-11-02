import os
from operator import methodcaller
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


# INPUT
success_cluster_files_dirpath = "success_recap_clustered_files"
fail_cluster_files_dirpath = "failed_recap_clustered_files"
irmsd_cut = 1.0
top_ranked_reps_considered = int(sys.argv[1])
if 'rpx' in sys.argv:
    rpx = True
else:
    rpx = False
fig_outfile_path = "RecapitulationPlot%f_%s_considered.png" % (irmsd_cut, top_ranked_reps_considered)


def get_clustered_files(_dir):
    target_cluster_filepaths = []
    for root1, dirs1, files1 in os.walk(_dir):
        for file1 in files1:
            if "clustered.txt" in file1:
                target_cluster_filepaths.append(root1 + "/" + file1)

    return target_cluster_filepaths


def get_rpx_status(design):
    entry_d = {'I': {('C2', 'C3'): 8, ('C2', 'C5'): 14, ('C3', 'C5'): 56}, 'T': {('C2', 'C3'): 4, ('C3', 'C3'): 52}}
    design_sym = design[:1]
    design_components = design[1:3]
    entry = entry_d[design_sym][('C%s' % design_components[1], 'C%s' % design_components[0])]
    if design[:3] in ['I32', 'I52']:
        return True
    else:
        return False


def find_recovered_counts(filepaths):
    all_recov_counts = []
    for target_cluster_filepath in filepaths:
        recov_counts = [0 for x in range(top_ranked_reps_considered)]

        target_cluster_file = open(target_cluster_filepath, "r")
        cluster_rep_lines = [line.rstrip() for line in target_cluster_file.readlines() if line.startswith("REP ")]
        target_cluster_file.close()

        for cluster_rep_line in cluster_rep_lines:
            cluster_rep_irmsd = float(cluster_rep_line.split()[2])
            cluster_rep_rank = int(cluster_rep_line.split()[5])

            if (cluster_rep_rank <= top_ranked_reps_considered) and (cluster_rep_irmsd <= irmsd_cut):
                for i in range(len(recov_counts)):
                    if i + 1 >= cluster_rep_rank:
                        recov_counts[i] += 1
                        #     break NEED to implement this or we could have double counting in the graph...
        all_recov_counts.append(recov_counts)

    return all_recov_counts


def recover_targets(all_recov_counts):
    total_target_number = len(all_recov_counts)
    recovered_targets = [0 for x in range(top_ranked_reps_considered)]
    for j in range(total_target_number):
        for k in range(top_ranked_reps_considered):
            # for k in range(top_ranked_reps_considered):
            #     for j in range(total_target_number):
            if all_recov_counts[j][k] > 0:
                recovered_targets[k] += 1
                # break  # Added here to prvent double counting...

    return recovered_targets


success_files = get_clustered_files(success_cluster_files_dirpath)
fail_files = get_clustered_files(fail_cluster_files_dirpath)
success_rpx = {'4NWN': False, '4NWO': False, '4NWP': False, '4NWR': False, '4ZK7': False, '5CY5': False, '5IM4': True,
               '5IM5': False, '5IM6': True, '6P6F': False, '6VFH': True, '6VFI': True, '6VFJ': True, '6VL6': True}
success_rpx_np = np.array(
    list(success_rpx[i[:4].upper()] for i in map(os.path.basename, map(os.path.dirname, success_files))))
rpx_designs_np = np.array(list(map(get_rpx_status, map(os.path.basename, fail_files))))
# not_rpx_designs = [~_bool for _bool in rpx_designs]
# not_rpx_designs_np = not_rpx_designs * fail_recov_counts

fail_recov_counts = find_recovered_counts(fail_files)
suc_recov_counts = find_recovered_counts(success_files)

fail_recovered_targets = recover_targets(fail_recov_counts)
suc_recovered_targets = recover_targets(suc_recov_counts)

# print(rpx_designs, 'rpx_mask\n' * 5, fail_recov_counts, 'Fail_counts\n' * 5)
# rpx_designs_int_np = np.array(rpx_designs)[:, None] * np.ones(top_ranked_reps_considered)[None, :]  # fail_recov_counts)
# rpx_designs_np = np.array(rpx_designs_int_np, dtype=bool)
# fail_no_rpx_mask_np = np.ma.array(fail_recov_counts, mask=~rpx_designs_np)  # returns values for non RPX designs

success_recov_counts_np = np.array(suc_recov_counts)
suc_non_rpx_recov_counts = success_recov_counts_np[~success_rpx_np, :]
suc_non_rpx_recovered_targets = recover_targets(suc_non_rpx_recov_counts)

suc_rpx_recov_counts = success_recov_counts_np[success_rpx_np, :]
suc_rpx_recovered_targets = recover_targets(suc_rpx_recov_counts)

fail_recov_counts_np = np.array(fail_recov_counts)
fail_non_rpx_recov_counts = fail_recov_counts_np[~rpx_designs_np, :]
# non_rpx_recov_counts = np.ma.compressed(fail_no_rpx_mask_np)
# print(rpx_designs_np, 'rpx*Fail_np\n' * 5, fail_no_rpx_mask_np, 'masked_fail\n' * 5, non_rpx_recov_counts)
fail_non_rpx_recovered_targets = recover_targets(fail_non_rpx_recov_counts)

# not_rpx_designs_np = np.array(not_rpx_designs)[:, None] * np.ones(top_ranked_reps_considered)[None, :]
# fail_rpx_mask_np = np.ma.array(fail_recov_counts, mask=rpx_designs_np)  # returns values for only RPX designs
# rpx_recov_counts = np.ma.compressed(fail_rpx_mask_np)
fail_rpx_recov_counts = fail_recov_counts_np[rpx_designs_np, :]
fail_rpx_recovered_targets = recover_targets(fail_rpx_recov_counts)

suc_recovered_targets_percentage = [(x * 100.0) / float(len(suc_recov_counts)) for x in suc_recovered_targets]
suc_rpx_recovered_targets_percentage = [(x * 100.0) / float(len(suc_rpx_recov_counts)) for x in
                                        suc_rpx_recovered_targets]
suc_non_rpx_recovered_targets_percentage = [(x * 100.0) / float(len(suc_non_rpx_recov_counts)) for x in
                                            suc_non_rpx_recovered_targets]
fail_recovered_targets_percentage = [(x * 100.0) / float(len(fail_recov_counts)) for x in fail_recovered_targets]
fail_rpx_recovered_targets_percentage = [(x * 100.0) / float(len(fail_rpx_recov_counts)) for x in
                                         fail_rpx_recovered_targets]
fail_non_rpx_recovered_targets_percentage = [(x * 100.0) / float(len(fail_non_rpx_recov_counts)) for x in
                                             fail_non_rpx_recovered_targets]

recov_target_percent = [fail_recovered_targets_percentage, suc_recovered_targets_percentage]
type_d = {1: 'Success', 0: 'Failed'}
target_rank = 70
for t, _type in enumerate(recov_target_percent):
    announced, announced_2 = False, False
    for i, rank in enumerate(_type):
        if i + 1 == 20:
            print('%s: At %d poses, %f%% of designs could be recovered' % (type_d[t], i + 1, rank))
        if rank >= target_rank and not announced:
            print('%s: 70%% at %d' % (type_d[t], i + 1))
            announced = True
        if rank >= 100 and not announced_2:
            print('%s: 100%% at %d' % (type_d[t], i + 1))
            announced_2 = True

print(fail_recovered_targets_percentage[-1])
print('Number of successful designs: %d' % len(suc_recov_counts))
print('Number of failed designs: %d' % len(fail_recov_counts))
plt.plot(range(top_ranked_reps_considered + 1), [0.0] + suc_recovered_targets_percentage, 'k', linewidth=2,
         label="Success")
if rpx:
    print('Number of successful RPX designs: %d' % len(suc_rpx_recov_counts))
    print('Number of failed RPX designs: %d' % len(fail_rpx_recov_counts))
    plt.plot(range(top_ranked_reps_considered + 1), [0.0] + suc_rpx_recovered_targets_percentage, 'k', linewidth=2,
             label="Success RPX", color='blue')  # , linestyle='dashed')
    plt.plot(range(top_ranked_reps_considered + 1), [0.0] + fail_rpx_recovered_targets_percentage, 'k', linewidth=2,
             label="Failed RPX", color='firebrick')  # , linestyle='dashed')
plt.plot(range(top_ranked_reps_considered + 1), [0.0] + fail_recovered_targets_percentage, 'k', linewidth=2,
         color='red', label="Failed")  # , linestyle='dashdot')
if rpx:
    print('Number of successful Non-RPX designs: %d' % len(suc_non_rpx_recov_counts))
    print('Number of failed Non-RPX designs: %d' % len(fail_non_rpx_recov_counts))
    plt.plot(range(top_ranked_reps_considered + 1), [0.0] + suc_non_rpx_recovered_targets_percentage, 'k', linewidth=2,
             color='lightblue', label="Success Non-RPX")  # , linestyle='dotted')
    plt.plot(range(top_ranked_reps_considered + 1), [0.0] + fail_non_rpx_recovered_targets_percentage, 'k', linewidth=2,
             color='lightcoral', label="Failed Non-RPX")  # , linestyle='dotted')
plt.legend(loc="lower right")
# plt.xlabel('# Ranked Poses Considered')
# plt.ylabel('% Targets Recovered')

plt.gca().set_aspect(aspect=2, adjustable='box')
# plt.gca().set_aspect('equal', adjustable='box')

font = {'family': 'normal', 'weight': 'bold', 'size': 18}
plt.rc('font', **font)

plt.savefig(fig_outfile_path, dpi=300)

# plt.show()

with open('Supplemental_design_recap_lists', 'w') as f:
    f.write('Success:\n')
    f.write(', '.join(sorted(s[:4] for s in map(os.path.basename, map(os.path.dirname, success_files)))))
    f.write('\n')
    f.write('Failed:\n')
    f.write(', '.join(sorted(l[0] for l in map(methodcaller('split', '_'), map(os.path.basename, fail_files)))))
    f.write('\n')
    f.write('RPX designs were those utilizing the motif library in Bale, J. et al. 2016 which includs I32 and I52 '
            'designs. All others are Non-RPX designs.\n')
