from classes.SymEntry import sym_comb_dict

header_format_string = "{:5s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}  {:10s}  {:9s}  {:^20s}  {:6s}"
query_output_format_string = "{:>5s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}"


def print_query_header():
    print(header_format_string.format("ENTRY", "GROUP1", "IntDofRot1", "IntDofTx1", "ReferenceFrameDof1", "GROUP2",
                                      "IntDofRot2", "IntDofTx2", "ReferenceFrameDof2", "RESULT"))


def query_combination(combination_list):
    if type(combination_list) == list and len(combination_list) == 2:
        matching_entries = []
        for entry_number, entry in sym_comb_dict.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if combination_list == [group1, group2] or combination_list == [group2, group1]:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                          str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                          str(int_rot2), str(int_tx2),
                                                                          ref_frame_tx_dof_group2, result))
        if matching_entries == list():
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s + %s" % (combination_list[0], combination_list[1]) +
                  '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def query_result(desired_result):
    if type(desired_result) == str:
        matching_entries = []
        for entry_number, entry in sym_comb_dict.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if desired_result == result:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                          str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                          str(int_rot2), str(int_tx2),
                                                                          ref_frame_tx_dof_group2, result))
        if matching_entries == list():
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s" % desired_result + '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def query_counterpart(query_group):
    if type(query_group) == str:
        matching_entries = []
        for entry_number, entry in sym_comb_dict.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _ = entry
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if query_group in [group1, group2]:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries.append(
                    "{:>5s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}  {:>10s}  {:>9s}  {:^20s}  {:>6s}".format(
                        str(entry_number), group1, str(int_rot1), str(int_tx1), ref_frame_tx_dof_group1, group2,
                        str(int_rot2), str(int_tx2), ref_frame_tx_dof_group2, result))
        if matching_entries == []:
            print('\033[1m' + "NO MATCHING ENTRY FOUND" + '\033[0m')
            print('')
        else:
            print('\033[1m' + "POSSIBLE COMBINATION(S) FOR: %s" % query_group + '\033[0m')
            print_query_header()
            for match in matching_entries:
                print(match)
    else:
        print("INVALID ENTRY")


def all_entries():
    all_entries_list = []
    for entry_number, entry in sym_comb_dict.items():
        group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
        ref_frame_tx_dof_group2, _, result, dimension, _ = entry
        # group2 = entry[6]
        # int_dof_group1 = entry[3]
        # int_dof_group2 = entry[8]
        # ref_frame_tx_dof_group1 = entry[5]
        # ref_frame_tx_dof_group2 = entry[10]
        # result = entry[12]
        int_rot1 = 0
        int_tx1 = 0
        int_rot2 = 0
        int_tx2 = 0
        for int_dof in int_dof_group1:
            if int_dof.startswith('r'):
                int_rot1 = 1
            if int_dof.startswith('t'):
                int_tx1 = 1
        for int_dof in int_dof_group2:
            if int_dof.startswith('r'):
                int_rot2 = 1
            if int_dof.startswith('t'):
                int_tx2 = 1
        all_entries_list.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                  str(int_tx1), ref_frame_tx_dof_group1, group2,
                                                                  str(int_rot2), str(int_tx2), ref_frame_tx_dof_group2,
                                                                  result))
    print('\033[1m' + "ALL ENTRIES" + '\033[0m')
    print_query_header()
    for entry in all_entries_list:
        print(entry)


def dimension(dim):
    if dim in [0, 2, 3]:
        matching_entries_list = []
        for entry_number, entry in sym_comb_dict.items():
            group1, int_dof_group1, _, ref_frame_tx_dof_group1, group2, int_dof_group2, _, \
                ref_frame_tx_dof_group2, _, result, dimension, _ = entry
            # group1 = entry[1]
            # group2 = entry[6]
            # int_dof_group1 = entry[3]
            # int_dof_group2 = entry[8]
            # ref_frame_tx_dof_group1 = entry[5]
            # ref_frame_tx_dof_group2 = entry[10]
            # result = entry[12]
            if dimension == dim:
                int_rot1 = 0
                int_tx1 = 0
                int_rot2 = 0
                int_tx2 = 0
                for int_dof in int_dof_group1:
                    if int_dof.startswith('r'):
                        int_rot1 = 1
                    if int_dof.startswith('t'):
                        int_tx1 = 1
                for int_dof in int_dof_group2:
                    if int_dof.startswith('r'):
                        int_rot2 = 1
                    if int_dof.startswith('t'):
                        int_tx2 = 1
                matching_entries_list.append(query_output_format_string.format(str(entry_number), group1, str(int_rot1),
                                                                               str(int_tx1), ref_frame_tx_dof_group1,
                                                                               group2, str(int_rot2), str(int_tx2),
                                                                               ref_frame_tx_dof_group2, result))
        print('\033[1m' + "ALL ENTRIES FOUND WITH DIMENSION " + str(dim) + ": " + '\033[0m')
        print_query_header()
        for entry in matching_entries_list:
            print(entry)
    else:
        print("DIMENSION NOT SUPPORTED, VALID DIMENSIONS ARE: 0, 2 or 3 ")
