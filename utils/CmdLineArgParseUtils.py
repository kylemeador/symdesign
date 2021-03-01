import os
import sys

from PathUtils import master_log
from utils import PostProcessUtils
from utils import SymQueryUtils


def query_mode(arg_list):
    valid_query_flags = ["-all_entries", "-combination", "-result", "-counterpart", "-dimension"]
    if len(arg_list) >= 3 and arg_list[1] == "-query" and arg_list[2] in valid_query_flags:
        print('\033[32m' + '\033[1m' + "Nanohedra" + '\033[0m')
        print('\033[1m' + '\033[95m' + "MODE: QUERY" + '\033[95m' + '\033[0m' + '\n')
        if arg_list[2] == "-all_entries":
            if len(arg_list) == 3:
                SymQueryUtils.all_entries()
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY" + '\033[0m')

        elif arg_list[2] == "-combination":
            if len(arg_list) == 5:
                query = [arg_list[3], arg_list[4]]
                SymQueryUtils.query_combination(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID COMBINATION QUERY" + '\033[0m')

        elif arg_list[2] == "-result":
            if len(arg_list) == 4:
                query = arg_list[3]
                SymQueryUtils.query_result(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID RESULT QUERY" + '\033[0m')

        elif arg_list[2] == "-counterpart":
            if len(arg_list) == 4:
                query = arg_list[3]
                SymQueryUtils.query_counterpart(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID COUNTERPART QUERY" + '\033[0m')

        elif arg_list[2] == "-dimension":
            if len(arg_list) == 4 and arg_list[3].isdigit():
                query = int(arg_list[3])
                SymQueryUtils.dimension(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY" + '\033[0m')

    else:
        sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY, CHOOSE ONE OF THE FOLLOWING QUERY FLAGS: -all_entries,"
                                          " -combination, -result, -counterpart, -dimension" + '\033[0m')


def get_docking_parameters(arg_list):
    if "-outdir" in arg_list:
        outdir_index = arg_list.index('-outdir') + 1
        if outdir_index < len(arg_list):
            outdir = arg_list[outdir_index]
        else:
            log_filepath = os.getcwd() + "/Nanohedra_log.txt"
            logfile = open(log_filepath, "a+")
            logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED" + '\n')
            logfile.close()
            sys.exit()
    else:
        log_filepath = os.getcwd() + "/Nanohedra_log.txt"
        logfile = open(log_filepath, "a+")
        logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED" + '\n')
        logfile.close()
        sys.exit()


    if not os.path.exists(outdir):
        os.makedirs(outdir)

    master_log_filepath = os.path.join(outdir, master_log)
    # with open(master_log_filepath, "w") as master_logfile:
    master_logfile = open(master_log_filepath, "w")
    master_logfile.write('Nanohedra\nMODE: DOCK\n\n')

    valid_flags = ["-dock", "-entry", "-pdb_dir1_path", "-pdb_dir2_path", "-rot_step1", "-rot_step2", "-outdir",
                   "-output_uc", "-output_surrounding_uc", "-min_matched", "-output_exp_assembly",
                   '-no_time', '-initial']  # "-cores", '-resume', "-init_match_type"

    # CHECK INPUT FLAGS
    for sys_input in arg_list:
        if sys_input.startswith('-') and sys_input not in valid_flags:
            master_logfile.write("ERROR: " + sys_input + " IS AN INVALID FLAG" + "\n")
            master_logfile.write("VALID FLAGS FOR DOCKING ARE:" + "\n")
            for flag in valid_flags:
                master_logfile.write(flag + "\n")
            master_logfile.close()
            sys.exit()

    # SymEntry PARAMETER
    if "-entry" in arg_list:
        entry_index = arg_list.index('-entry') + 1
        if entry_index < len(arg_list):
            if arg_list[entry_index].isdigit() and (int(arg_list[entry_index]) in range(1, 126)):
                entry = int(arg_list[arg_list.index('-entry') + 1])
            else:
                master_logfile.write("ERROR: INVALID SYMMETRY ENTRY. SUPPORTED VALUES ARE: 1 to 125" + "\n")
                master_logfile.close()
                sys.exit()
        else:
            master_logfile.write("ERROR: SYMMETRY ENTRY NOT SPECIFIED" + "\n")
            master_logfile.close()
            sys.exit()
    else:
        master_logfile.write("ERROR: SYMMETRY ENTRY NOT SPECIFIED" + "\n")
        master_logfile.close()
        sys.exit()

    # General INPUT PARAMETERS
    if ("-pdb_dir1_path" in arg_list) and ("-pdb_dir2_path" in arg_list):
        path1_index = arg_list.index('-pdb_dir1_path') + 1
        path2_index = arg_list.index('-pdb_dir2_path') + 1

        if (path1_index < len(arg_list)) and (path2_index < len(arg_list)):
            path1 = arg_list[arg_list.index('-pdb_dir1_path') + 1]
            path2 = arg_list[arg_list.index('-pdb_dir2_path') + 1]
            if os.path.exists(path1) and os.path.exists(path2):
                pdb_dir1_path = path1
                pdb_dir2_path = path2
            else:
                master_logfile.write("ERROR: SPECIFIED PDB DIRECTORY PATH(S) DO(ES) NOT EXIST" + "\n")
                master_logfile.close()
                sys.exit()
        else:
            master_logfile.write("ERROR: PDB DIRECTORY PATH(S) NOT SPECIFIED" + "\n")
            master_logfile.close()
            sys.exit()
    else:
        master_logfile.write("ERROR: PDB DIRECTORY PATH(S) NOT SPECIFIED" + "\n")
        master_logfile.close()
        sys.exit()

    # if "-cores" in arg_list:
    #     cores_index = arg_list.index('-cores') + 1
    #
    #     if cores_index < len(arg_list):
    #         if arg_list[cores_index].isdigit():
    #             cores = int(arg_list[cores_index])
    #         else:
    #             master_logfile.write("ERROR: VALUE SPECIFIED FOR NUMBER OF CPU CORES IS NOT AN INTEGER" + "\n")
    #             master_logfile.close()
    #             sys.exit()
    #     else:
    #         master_logfile.write("ERROR: NUMBER OF CPU CORES NOT SPECIFIED" + "\n")
    #         master_logfile.close()
    #         sys.exit()
    # else:
    #     cores = 1
    #
    # if "-init_match_type" in arg_list:
    #     init_match_type_index = arg_list.index('-init_match_type') + 1
    #
    #     if init_match_type_index < len(arg_list):
    #         if arg_list[init_match_type_index] in ["1_1", "1_2", "2_1", "2_2"]:
    #             init_match_type = str(arg_list[init_match_type_index])
    #         else:
    #             master_logfile.write("ERROR: INITIAL FRAGMENT MATCH TYPE SPECIFIED NOT RECOGNIZED" + "\n")
    #             master_logfile.close()
    #             sys.exit()
    #     else:
    #         master_logfile.write("ERROR: INITIAL FRAGMENT MATCH TYPE NOT SPECIFIED" + "\n")
    #         master_logfile.close()
    #         sys.exit()
    # else:
    #     init_match_type = "1_1"  # default initial fragment match type is set to helix-helix interactions ==> "1_1"

    # FragDock PARAMETERS
    if "-rot_step1" in arg_list:
        rot_step_index1 = arg_list.index('-rot_step1') + 1
        if rot_step_index1 < len(arg_list):
            if arg_list[rot_step_index1].isdigit():
                rot_step_deg1 = int(arg_list[rot_step_index1])
            else:
                master_logfile.write("ERROR: ROTATION STEP SPECIFIED IS NOT AN INTEGER" + "\n")
                master_logfile.close()
                sys.exit()
        else:
            master_logfile.write("ERROR: ROTATION STEP NOT SPECIFIED" + "\n")
            master_logfile.close()
            sys.exit()
    else:
        rot_step_deg1 = None

    if "-rot_step2" in arg_list:
        rot_step_index2 = arg_list.index('-rot_step2') + 1
        if rot_step_index2 < len(arg_list):
            if arg_list[rot_step_index2].isdigit():
                rot_step_deg2 = int(arg_list[rot_step_index2])
            else:
                master_logfile.write("ERROR: ROTATION STEP SPECIFIED IS NOT AN INTEGER" + "\n")
                master_logfile.close()
                sys.exit()
        else:
            master_logfile.write("ERROR: ROTATION STEP NOT SPECIFIED" + "\n")
            master_logfile.close()
            sys.exit()
    else:
        rot_step_deg2 = None

    if "-output_exp_assembly" in arg_list:
        output_exp_assembly = True
    else:
        output_exp_assembly = False

    if "-output_uc" in arg_list:
        output_uc = True
    else:
        output_uc = False

    if "-output_surrounding_uc" in arg_list:
        output_surrounding_uc = True
    else:
        output_surrounding_uc = False

    if "-min_matched" in arg_list:
        min_matched_index = arg_list.index('-min_matched') + 1
        if min_matched_index < len(arg_list):
            if arg_list[min_matched_index].isdigit():
                min_matched = int(arg_list[min_matched_index])
            else:
                master_logfile.write(
                    "ERROR: MINIMUM NUMBER OF REQUIRED MATCHED FRAGMENT(S) SPECIFIED IS NOT AN INTEGER" + "\n")
                master_logfile.close()
                sys.exit()
        else:
            master_logfile.write("ERROR: MINIMUM NUMBER OF REQUIRED MATCHED FRAGMENT(S) NOT SPECIFIED" + "\n")
            master_logfile.close()
            sys.exit()
    else:
        min_matched = 3

    # if '-resume' in arg_list:
    #     resume = True
    # else:
    #     resume = False

    if '-no_time' in arg_list:
        keep_time = False
    else:
        keep_time = True

    master_logfile.close()

    if '-initial' in arg_list:
        initial = True
    else:
        initial = False

    return entry, pdb_dir1_path, pdb_dir2_path, rot_step_deg1, rot_step_deg2, outdir, output_exp_assembly, \
        output_uc, output_surrounding_uc, min_matched, keep_time, initial  # init_match_type, cores, resume


def postprocess_mode(arg_list):
    valid_flags = ["-outdir", "-design_dir", "-min_score", "-min_matched", "-postprocess", "-rank", "-min_matched_ss"]
    for arg in arg_list:
        if arg[0] == '-' and arg not in valid_flags:
            log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
            logfile = open(log_filepath, "a+")
            logfile.write("ERROR: %s IS AN INVALID FLAG\n" % arg)
            logfile.write("VALID FLAGS ARE: -outdir, -design_dir, -min_score, -min_matched, -postprocess\n")
            logfile.close()
            sys.exit()

    if "-outdir" in arg_list:
        outdir_index = arg_list.index('-outdir') + 1
        if outdir_index < len(arg_list):
            outdir = arg_list[outdir_index]
        else:
            log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
            logfile = open(log_filepath, "a+")
            logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED\n")
            logfile.close()
            sys.exit()
    else:
        log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
        logfile = open(log_filepath, "a+")
        logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED\n")
        logfile.close()
        sys.exit()

    log_filepath = outdir + "/Nanohedra_PostProcess_log.txt"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    try:
        design_dir_path_index = arg_list.index("-design_dir") + 1
        try:
            design_dir_path = arg_list[design_dir_path_index]
            if not os.path.exists(design_dir_path):
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: DESIGN DIRECTORY PATH SPECIFIED DOES NOT EXIST\n")
                logfile.close()
                sys.exit()
        except IndexError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -design_dir FLAG FOLLOWED BY DESIGN DIRECTORY PATH IS REQUIRED\n")
            logfile.close()
            sys.exit()
    except ValueError:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR: -design_dir FLAG FOLLOWED BY DESIGN DIRECTORY PATH IS REQUIRED\n")
        logfile.close()
        sys.exit()

    if "-min_score" in arg_list and "-min_matched" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
        try:
            min_score_index = arg_list.index("-min_score") + 1
            try:
                min_score_str = arg_list[min_score_index]
                try:
                    min_score = float(min_score_str)
                    PostProcessUtils.score_filter(design_dir_path, min_score, outdir)
                except ValueError:
                    logfile = open(log_filepath, "w")
                    logfile.write("ERROR: MINIMUM SCORE SPECIFIED IS NOT A FLOAT\n")
                    logfile.close()
                    sys.exit()
            except IndexError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
                logfile.close()
                sys.exit()
        except ValueError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
            logfile.close()
            sys.exit()

    if "-min_matched" in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
        try:
            min_matched_index = arg_list.index("-min_matched") + 1
            try:
                min_matched_str = arg_list[min_matched_index]
                try:
                    min_matched = int(min_matched_str)
                    PostProcessUtils.frag_match_count_filter(design_dir_path, min_matched, outdir)
                except ValueError:
                    logfile = open(log_filepath, "w")
                    logfile.write("ERROR: MINIMUM MATCHED FRAGMENT COUNT SPECIFIED IS NOT AN INTEGER\n")
                    logfile.close()
                    sys.exit()
            except IndexError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
                logfile.close()
                sys.exit()
        except ValueError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
            logfile.close()
            sys.exit()

    if "-min_matched" in arg_list and "-min_score" in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
        min_matched_index = arg_list.index("-min_matched") + 1
        min_score_index = arg_list.index("-min_score") + 1
        try:
            min_matched_str = arg_list[min_matched_index]
            min_score_str = arg_list[min_score_index]

            try:
                min_matched = int(min_matched_str)
            except ValueError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: MINIMUM MATCHED FRAGMENT COUNT SPECIFIED IS NOT AN INTEGER\n")
                logfile.close()
                sys.exit()

            try:
                min_score = float(min_score_str)
            except ValueError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: MINIMUM SCORE SPECIFIED IS NOT A FLOAT\n")
                logfile.close()
                sys.exit()

            PostProcessUtils.score_and_frag_match_count_filter(design_dir_path, min_score, min_matched, outdir)

        except IndexError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
            logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
            logfile.close()
            sys.exit()

    if "-rank" in arg_list and "-min_matched" not in arg_list and "-min_score" not in arg_list and "-min_matched_ss" not in arg_list:
        # try:
        rank_index = arg_list.index("-rank") + 1
        # try:
        metric = arg_list[rank_index]
        # try:
        metric_str = str(metric)
        if metric_str in ["score", "matched"]:
            PostProcessUtils.rank(design_dir_path, metric_str, outdir)
        else:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT RECOGNIZED\n")
            logfile.close()
            sys.exit()
            # except ValueError:
            #     logfile = open(log_filepath, "w")
            #     logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT A STRING\n")
            #     logfile.close()
            #     sys.exit()
        #     except IndexError:
        #         logfile = open(log_filepath, "w")
        #         logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
        #         logfile.close()
        #         sys.exit()
        # except ValueError:
        #     logfile = open(log_filepath, "w")
        #     logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
        #     logfile.close()
        #     sys.exit()

    if "-min_matched_ss" in arg_list and "-min_matched" not in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list:
        try:
            min_matched_ss_index = arg_list.index("-min_matched_ss") + 1
            try:
                min_matched_ss_str = arg_list[min_matched_ss_index]
                try:
                    min_matched_ss = int(min_matched_ss_str)
                    PostProcessUtils.ss_match_count_filter(design_dir_path, min_matched_ss, outdir)
                except ValueError:
                    logfile = open(log_filepath, "w")
                    logfile.write(
                        "ERROR: VALUE SPECIFIED FOR MINIMUM SECONDARY STRUCTURE ELEMENT MATCHES IS NOT AN INTEGER\n")
                    logfile.close()
                    sys.exit()
            except IndexError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: -min_matched_ss FLAG IS NOT FOLLOWED BY A MINIMUM VALUE\n")
                logfile.close()
                sys.exit()
        except ValueError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -min_matched_ss FLAG IS NOT FOLLOWED BY A MINIMUM VALUE\n")
            logfile.close()
            sys.exit()

    if ("-min_matched" in arg_list or "-min_score" in arg_list) and "-rank" in arg_list:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR:\n")
        logfile.write("EITHER: FILTER BY SCORE AND/OR BY MINIMUM FRAGMENT(S) MATCHED\n")
        logfile.write("OR: PERFORM RANKING BY SCORE\n")
        logfile.write("OR: PERFORM RANKING BY NUMBER OF FRAGMENT(S) MATCHED\n")
        logfile.close()
        sys.exit()

    if ("-min_matched" in arg_list or "-min_score" in arg_list) and "-min_matched_ss" in arg_list:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR:\n")
        logfile.write("EITHER: FILTER BY SCORE AND/OR BY MINIMUM FRAGMENT(S) MATCHED\n")
        logfile.write("OR: FILTER BY MINIMUM SECONDARY STRUCTURE ELEMENT(S) MATCHED\n")
        logfile.close()
        sys.exit()

    if "-rank" in arg_list and "-min_matched_ss" in arg_list:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR:\n")
        logfile.write("EITHER: FILTER BY MINIMUM SECONDARY STRUCTURE ELEMENT(S) MATCHED\n")
        logfile.write("OR: PERFORM RANKING BY SCORE\n")
        logfile.write("OR: PERFORM RANKING BY NUMBER OF FRAGMENT(S) MATCHED\n")
        logfile.close()
        sys.exit()

    if "-min_matched" not in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR: POST PROCESSING FLAG REQUIRED\n")
        logfile.close()
        sys.exit()
