import os
import shutil
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
sys.path.extend([parent_dir])
# from PoseDirectory import set_up_directory_objects
# from SymDesignUtils import collect_designs
from PDB import PDB


def frag_match_count_filter(master_design_dirpath, min_frag_match_count, master_design_outdir_path):
    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "docked_pose_info_file.txt" in file1:
                info_file_filepath = root1 + os.sep + file1

                tx_filepath = os.path.dirname(root1)
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split(os.sep)[-1]
                rot_filename = rot_filepath.split(os.sep)[-1]
                degen_filename = degen_filepath.split(os.sep)[-1]
                design_filename = design_filepath.split(os.sep)[-1]

                outdir = master_design_outdir_path + os.sep + design_filename + os.sep + degen_filename + os.sep + rot_filename + os.sep + tx_filename

                info_file = open(info_file_filepath, 'r')
                for line in info_file.readlines():
                    if "Unique Mono Fragments Matched:" in line:
                        frag_match_count = int(line[30:])
                        if frag_match_count >= min_frag_match_count:
                            shutil.copytree(tx_filepath, outdir)
                info_file.close()


def score_filter(master_design_dirpath, min_score, master_design_outdir_path):
    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "docked_pose_info_file.txt" in file1:
                info_file_filepath = root1 + os.sep + file1

                tx_filepath = os.path.dirname(root1)
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split(os.sep)[-1]
                rot_filename = rot_filepath.split(os.sep)[-1]
                degen_filename = degen_filepath.split(os.sep)[-1]
                design_filename = design_filepath.split(os.sep)[-1]

                outdir = master_design_outdir_path + os.sep + design_filename + os.sep + degen_filename + os.sep + rot_filename + os.sep + tx_filename

                info_file = open(info_file_filepath, 'r')
                for line in info_file.readlines():
                    if "Nanohedra Score:" in line:
                        score = float(line[17:])
                        if score >= min_score:
                            shutil.copytree(tx_filepath, outdir)
                info_file.close()


def score_and_frag_match_count_filter(master_design_dirpath, min_score, min_frag_match_count,
                                      master_design_outdir_path):
    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "docked_pose_info_file.txt" in file1:
                info_file_filepath = root1 + os.sep + file1

                tx_filepath = root1
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split(os.sep)[-1]
                rot_filename = rot_filepath.split(os.sep)[-1]
                degen_filename = degen_filepath.split(os.sep)[-1]
                design_filename = design_filepath.split(os.sep)[-1]

                outdir = master_design_outdir_path + os.sep + design_filename + os.sep + degen_filename + os.sep + rot_filename + os.sep + tx_filename

                score = None
                frag_match_count = None
                info_file = open(info_file_filepath, 'r')
                for line in info_file.readlines():
                    if "Nanohedra Score:" in line:
                        score = float(line[17:])
                    if "Unique Mono Fragments Matched:" in line:
                        frag_match_count = int(line[30:])
                info_file.close()

                if score is not None and frag_match_count is not None:
                    if score >= min_score and frag_match_count >= min_frag_match_count:
                        shutil.copytree(tx_filepath, outdir)


def rank(master_design_dirpath, metric, outdir):

    if metric == 'score':
        metric_str = "Nanohedra Score:"
    elif metric == 'matched':
        metric_str = "Unique Mono Fragments Matched:"
    else:
        raise ValueError('\n%s is not a recognized ranking metric. '
                         'Recognized ranking metrics are: score and matched.\n' %str(metric))

    designpath_metric_tup_list = []

    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "docked_pose_info_file.txt" in file1:
                info_file_filepath = root1 + "/" + file1

                tx_filepath = root1
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split(os.sep)[-1]
                rot_filename = rot_filepath.split(os.sep)[-1]
                degen_filename = degen_filepath.split(os.sep)[-1]
                design_filename = design_filepath.split(os.sep)[-1]

                design_path = os.sep + design_filename + os.sep + degen_filename + os.sep + rot_filename + os.sep + tx_filename

            if metric == 'score':
                info_file = open(info_file_filepath, 'r')
                for line in info_file.readlines():
                    if metric_str in line:
                        score = float(line[17:])
                        designpath_metric_tup_list.append((design_path, score))
                info_file.close()

            elif metric == 'matched':
                info_file = open(info_file_filepath, 'r')
                for line in info_file.readlines():
                    if metric_str in line:
                        frag_match_count = int(line[30:])
                        designpath_metric_tup_list.append((design_path, frag_match_count))
                info_file.close()

    designpath_metric_tup_list_sorted = sorted(designpath_metric_tup_list, key=lambda tup: tup[1], reverse=True)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = open(outdir + "/ranked_designs_%s.txt" % metric, 'w')
    for p, m in designpath_metric_tup_list_sorted:
        outfile.write("%s\t%s\n" % (str(p), str(m)))
    outfile.close()


# unreleased rank
# def rank(master_design_dirpath, metric, outdir):
#     if metric == 'score':
#         metric_str = "Nanohedra Score:"
#     elif metric == 'matched':
#         metric_str = "Unique Mono Fragments Matched:"
#     else:
#         raise ValueError('\n%s is not a recognized ranking metric. Recognized ranking metrics are: score and matched.\n'
#                          % str(metric))
#
#     # designpath_metric_tup_list = []
#     # print('Finding all Nanohedra directories')
#     all_design_directories, location = collect_designs(master_design_dirpath, dir_type='nanohedra')
#     # print('Setting up directory objects')
#     all_design_directories = set_up_directory_objects(all_design_directories)
#     # for root1, dirs1, files1 in os.walk(master_design_dirpath):
#     #     for file1 in files1:
#     #         if "frag_match_info_file.txt" in file1:
#     #             info_file_filepath = root1 + "/" + file1
#     #
#     #             tx_filepath = os.path.dirname(root1)
#     #             rot_filepath = os.path.dirname(tx_filepath)
#     #             degen_filepath = os.path.dirname(rot_filepath)
#     #             design_filepath = os.path.dirname(degen_filepath)
#     #
#     #             tx_filename = tx_filepath.split("/")[-1]
#     #             rot_filename = rot_filepath.split("/")[-1]
#     #             degen_filename = degen_filepath.split("/")[-1]
#     #             design_filename = design_filepath.split("/")[-1]
#     #
#     #             design_path = "/" + design_filename + "/" + degen_filename + "/" + rot_filename + "/" + tx_filename
#     # print('Gathering scores')
#     designpath_metric_tup_list = [(des_dir.path, des_dir.pose_score())
#                                   for des_dir in all_design_directories]
#     # print('Sorting')
#     # if metric == 'score':
#     #     info_file = open(info_file_filepath, 'r')
#     #     for line in info_file.readlines():
#     #         if metric_str in line:
#     #             # score = float(line[17:])
#     #             score = float(line[30:])
#     #             designpath_metric_tup_list.append((design_path, score))
#     #     info_file.close()
#     #
#     # elif metric == 'matched':
#     #     info_file = open(info_file_filepath, 'r')
#     #     for line in info_file.readlines():
#     #         if metric_str in line:
#     #             frag_match_count = int(line[38:])
#     #             designpath_metric_tup_list.append((design_path, frag_match_count))
#     #     info_file.close()
#
#     designpath_metric_tup_list_sorted = sorted(designpath_metric_tup_list, key=lambda tup: (tup[1] or 0), reverse=True)
#
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     with open(outdir + "/ranked_designs_%s.txt" % metric, 'w') as outfile:
#         for p, m in designpath_metric_tup_list_sorted:
#             outfile.write("%s\t%s\n" % (str(p), str(m)))

# unreleased rank
def ss_match_count_filter(master_design_dirpath, min_ss_match_count, master_design_outdir_path):
    # get original oligomer 1 and oligomer 2 PDB file paths
    original_oligomer_1_pdb_path = ""
    original_oligomer_2_pdb_path = ""
    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "frag_match_info_file.txt" in file1:
                info_file_filepath = root1 + "/" + file1
                info_file = open(info_file_filepath, 'r')
                info_file_lines = info_file.readlines()
                info_file.close()
                for line in info_file_lines:
                    if line.startswith("Original PDB 1 Path:"):
                        original_oligomer_1_pdb_path = line[21:].rstrip()
                    elif line.startswith("Original PDB 2 Path:"):
                        original_oligomer_2_pdb_path = line[21:].rstrip()
                break
        if original_oligomer_1_pdb_path != "" and original_oligomer_2_pdb_path != "":
            break

    # for each residue in the first chain of oligomer 1
    # get the residue secondary structure type
    # and the number of the secondary structure element it belongs to
    ss_res_info_dict_1 = {}
    pdb_oligomer_1 = PDB.from_file(original_oligomer_1_pdb_path)
    ch_id_oligomer_1 = pdb_oligomer_1.chain_ids[0]
    ss_asg_oligomer_1 = pdb_oligomer_1.chain(ch_id_oligomer_1).get_secondary_structure()
    ss_num_1 = 0
    prev_ss_type_1 = None
    for res, ss_type_1 in zip(pdb_oligomer_1.residues, ss_asg_oligomer_1):
        if prev_ss_type_1 != ss_type_1:
            ss_num_1 += 1
        ss_res_info_dict_1[res.number_pdb] = (ss_type_1, ss_num_1)
        prev_ss_type_1 = ss_type_1

    # for each residue in the first chain of oligomer 2
    # get the residue secondary structure type and
    # the number of the secondary structure element it belongs to
    ss_res_info_dict_2 = {}
    pdb_oligomer_2 = PDB.from_file(original_oligomer_2_pdb_path)
    ch_id_oligomer_2 = pdb_oligomer_2.chain_ids[0]
    ss_asg_oligomer_2 = pdb_oligomer_2.chain(ch_id_oligomer_2).get_secondary_structure()
    ss_num_2 = 0
    prev_ss_type_2 = None
    for res, ss_type_2 in zip(pdb_oligomer_2.residues, ss_asg_oligomer_2):
        if prev_ss_type_2 != ss_type_2:
            ss_num_2 += 1
        ss_res_info_dict_2[res.number_pdb] = (ss_type_2, ss_num_2)
        prev_ss_type_2 = ss_type_2

    # for all docked poses get central residue numbers of matched surface fragments from frag_match_info_file.txt files
    # then count the total number of distinct secondary structure elements for which fragments were found
    # copy files to new output directory for docked poses that have a secondary structure match count larger than
    # or equal to the minimum specified threshold "min_ss_match_count"
    for root1, dirs1, files1 in os.walk(master_design_dirpath):
        for file1 in files1:
            if "frag_match_info_file.txt" in file1:
                info_file_filepath = root1 + "/" + file1
                f = open(info_file_filepath, 'r')
                f_lines = f.readlines()
                f.close()

                tx_filepath = os.path.dirname(root1)
                rot_filepath = os.path.dirname(tx_filepath)
                degen_filepath = os.path.dirname(rot_filepath)
                design_filepath = os.path.dirname(degen_filepath)

                tx_filename = tx_filepath.split("/")[-1]
                rot_filename = rot_filepath.split("/")[-1]
                degen_filename = degen_filepath.split("/")[-1]
                design_filename = design_filepath.split("/")[-1]

                outdir = master_design_outdir_path + "/" + design_filename + "/" + degen_filename + "/" + rot_filename + "/" + tx_filename

                # design_id = degen_filename + "_" + rot_filename + "_" + tx_filename

                surffrags_central_resnums_1 = []
                surffrags_central_resnums_2 = []
                for line in f_lines:
                    if line.startswith("Surface Fragment Oligomer1 Residue Number:"):
                        oligomer_1_res_num = int(line[42:])
                        surffrags_central_resnums_1.append(oligomer_1_res_num)
                    if line.startswith("Surface Fragment Oligomer2 Residue Number:"):
                        oligomer_2_res_num = int(line[42:])
                        surffrags_central_resnums_2.append(oligomer_2_res_num)

                distinct_ss_elem_nums_1 = []
                for resnum in surffrags_central_resnums_1:
                    ss_elem_num = ss_res_info_dict_1[resnum][1]
                    if ss_elem_num not in distinct_ss_elem_nums_1:
                        distinct_ss_elem_nums_1.append(ss_elem_num)

                distinct_ss_elem_nums_2 = []
                for resnum in surffrags_central_resnums_2:
                    ss_elem_num = ss_res_info_dict_2[resnum][1]
                    if ss_elem_num not in distinct_ss_elem_nums_2:
                        distinct_ss_elem_nums_2.append(ss_elem_num)

                total_distinct_ss_elem_count = len(distinct_ss_elem_nums_1) + len(distinct_ss_elem_nums_2)

                if total_distinct_ss_elem_count >= min_ss_match_count:
                    # print design_id, total_distinct_ss_elem_count
                    shutil.copytree(tx_filepath, outdir)
