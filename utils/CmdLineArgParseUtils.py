import os
import sys

from classes.SymEntry import all_entries, query_combination, query_result, query_counterpart, dimension
from classes.SymEntry import nanohedra_symmetry_combinations
from utils import PostProcessUtils
from PathUtils import nano_entity_flag1, nano_entity_flag2
from SymDesignUtils import start_log


# Globals
logger = start_log(name=__name__)


def query_mode(arg_list):
    valid_query_flags = ["-all_entries", "-combination", "-result", "-counterpart", "-dimension"]
    if len(arg_list) >= 3 and arg_list[1] == "-query" and arg_list[2] in valid_query_flags:
        print('\033[32m' + '\033[1m' + "Nanohedra" + '\033[0m')
        print('\033[1m' + '\033[95m' + "MODE: QUERY" + '\033[95m' + '\033[0m' + '\n')
        if arg_list[2] == "-all_entries":
            if len(arg_list) == 3:
                all_entries()
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY" + '\033[0m')
        elif arg_list[2] == "-combination":
            if len(arg_list) == 5:
                query = [arg_list[3], arg_list[4]]
                query_combination(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID COMBINATION QUERY" + '\033[0m')
        elif arg_list[2] == "-result":
            if len(arg_list) == 4:
                query = arg_list[3]
                query_result(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID RESULT QUERY" + '\033[0m')
        elif arg_list[2] == "-counterpart":
            if len(arg_list) == 4:
                query = arg_list[3]
                query_counterpart(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID COUNTERPART QUERY" + '\033[0m')
        elif arg_list[2] == "-dimension":
            if len(arg_list) == 4 and arg_list[3].isdigit():
                query = int(arg_list[3])
                dimension(query)
            else:
                sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY" + '\033[0m')
    else:
        sys.exit('\033[91m' + '\033[1m' + "ERROR: INVALID QUERY, CHOOSE ONE OF THE FOLLOWING QUERY FLAGS: -all_entries,"
                                          " -combination, -result, -counterpart, -dimension" + '\033[0m')


def get_docking_parameters(arg_list):
    valid_flags = ['-dock', '-entry', '-oligomer1', '-oligomer2', '-rot_step1', '-rot_step2', '-outdir',
                   '-output_uc', '-output_surrounding_uc', '-min_matched', '-output_exp_assembly', '-output_assembly',
                   '-no_time', '-initial', '-debug']
    if '-outdir' in arg_list:
        outdir_index = arg_list.index('-outdir') + 1
        if outdir_index < len(arg_list):
            outdir = arg_list[outdir_index]
        else:
            logger.error('OUTPUT DIRECTORY NOT SPECIFIED\n')
            exit(1)
    else:
        logger.error('OUTPUT DIRECTORY NOT SPECIFIED\n')
        exit(1)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # CHECK INPUT FLAGS
    for sys_input in arg_list:
        if sys_input.startswith('-') and sys_input not in valid_flags:
            logger.error('%s IS AN INVALID FLAG\nVALID FLAGS FOR DOCKING ARE:\n%s'
                         % (sys_input, '\n'.join(valid_flags)))
            exit(1)

    # SymEntry PARAMETER
    if '-entry' in arg_list:
        entry_index = arg_list.index('-entry') + 1
        if entry_index < len(arg_list):
            if arg_list[entry_index].isdigit() and \
                    (int(arg_list[entry_index]) in range(1, len(nanohedra_symmetry_combinations))):
                entry = int(arg_list[arg_list.index('-entry') + 1])
            else:
                logger.error('INVALID SYMMETRY ENTRY. SUPPORTED VALUES ARE: %d to %d' %
                             (1, len(nanohedra_symmetry_combinations)))
                exit(1)
        else:
            logger.error('SYMMETRY ENTRY NOT SPECIFIED')
            exit(1)
    else:
        logger.error('SYMMETRY ENTRY NOT SPECIFIED')
        exit(1)

    # General INPUT PARAMETERS
    if nano_entity_flag1 in arg_list and nano_entity_flag2 in arg_list:
        path1_index = arg_list.index(nano_entity_flag1) + 1
        path2_index = arg_list.index(nano_entity_flag2) + 1

        if (path1_index < len(arg_list)) and (path2_index < len(arg_list)):
            path1 = arg_list[arg_list.index(nano_entity_flag1) + 1]
            path2 = arg_list[arg_list.index(nano_entity_flag2) + 1]
            if os.path.exists(path1) and os.path.exists(path2):
                pdb_dir1_path = path1
                pdb_dir2_path = path2
            else:
                logger.error('SPECIFIED PDB DIRECTORY PATH(S) DO(ES) NOT EXIST')
                exit(1)
        else:
            logger.error('PDB DIRECTORY PATH(S) NOT SPECIFIED')
            exit(1)
    else:
        logger.error('PDB DIRECTORY PATH(S) NOT SPECIFIED')
        exit(1)

    # FragDock PARAMETERS
    if '-rot_step1' in arg_list:
        rot_step_index1 = arg_list.index('-rot_step1') + 1
        if rot_step_index1 < len(arg_list):
            if arg_list[rot_step_index1].isdigit():
                rot_step_deg1 = int(arg_list[rot_step_index1])
            else:
                logger.error("ROTATION STEP SPECIFIED IS NOT AN INTEGER\n")
                exit(1)
        else:
            logger.error("ROTATION STEP NOT SPECIFIED\n")
            exit(1)
    else:
        rot_step_deg1 = None

    if "-rot_step2" in arg_list:
        rot_step_index2 = arg_list.index('-rot_step2') + 1
        if rot_step_index2 < len(arg_list):
            if arg_list[rot_step_index2].isdigit():
                rot_step_deg2 = int(arg_list[rot_step_index2])
            else:
                logger.error("ROTATION STEP SPECIFIED IS NOT AN INTEGER\n")
                exit(1)
        else:
            logger.error("ROTATION STEP NOT SPECIFIED\n")
            exit(1)
    else:
        rot_step_deg2 = None

    output_assembly = False
    if "-output_exp_assembly" in arg_list or "-output_assembly" in arg_list or "-output_uc" in arg_list:
        output_assembly = True

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
                logger.error("MINIMUM NUMBER OF REQUIRED MATCHED FRAGMENT(S) SPECIFIED IS NOT AN INTEGER\n")
                exit(1)
        else:
            logger.error("ERROR: MINIMUM NUMBER OF REQUIRED MATCHED FRAGMENT(S) NOT SPECIFIED\n")
            exit(1)
    else:
        min_matched = 3

    if '-no_time' in arg_list:
        keep_time = False
    else:
        keep_time = True

    if '-initial' in arg_list:
        initial = True
    else:
        initial = False

    if '-debug' in arg_list:
        debug = True
    else:
        debug = False

    return entry, pdb_dir1_path, pdb_dir2_path, rot_step_deg1, rot_step_deg2, outdir, output_assembly, \
        output_surrounding_uc, min_matched, keep_time, initial, debug


def postprocess_mode(arg_list):
    valid_flags = ["-outdir", "-design_dir", "-min_score", "-min_matched", "-postprocess", "-rank"]
    for arg in arg_list:
        if arg[0] == '-' and arg not in valid_flags:
            log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
            logfile = open(log_filepath, "a+")
            logfile.write("ERROR: %s IS AN INVALID FLAG\n" %arg)
            logfile.write("VALID FLAGS ARE: -outdir, -design_dir, -min_score, -min_matched, -rank, -postprocess\n")
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

    if "-min_score" in arg_list and "-min_matched" not in arg_list and "-rank" not in arg_list:
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

    if "-min_matched" in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list:
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

    if "-min_matched" in arg_list and "-min_score" in arg_list and "-rank" not in arg_list:
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

    if "-rank" in arg_list and "-min_matched" not in arg_list and "-min_score" not in arg_list:
        try:
            rank_index = arg_list.index("-rank") + 1
            try:
                metric = arg_list[rank_index]
                try:
                    metric_str = str(metric)
                    if metric_str in ["score", "matched"]:
                        PostProcessUtils.rank(design_dir_path, metric_str, outdir)
                    else:
                        logfile = open(log_filepath, "w")
                        logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT RECOGNIZED\n")
                        logfile.close()
                        sys.exit()
                except ValueError:
                    logfile = open(log_filepath, "w")
                    logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT A STRING\n")
                    logfile.close()
                    sys.exit()
            except IndexError:
                logfile = open(log_filepath, "w")
                logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
                logfile.close()
                sys.exit()
        except ValueError:
            logfile = open(log_filepath, "w")
            logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
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

    if "-min_matched" not in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list:
        logfile = open(log_filepath, "w")
        logfile.write("ERROR: POST PROCESSING FLAG REQUIRED\n")
        logfile.close()
        sys.exit()


# unreleased_postprocess_mode
# def postprocess_mode(arg_list):
#     valid_flags = ["-outdir", "-design_dir", "-min_score", "-min_matched", "-postprocess", "-rank", "-min_matched_ss"]
#     for arg in arg_list:
#         if arg[0] == '-' and arg not in valid_flags:
#             log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
#             logfile = open(log_filepath, "a+")
#             logfile.write("ERROR: %s IS AN INVALID FLAG\n" % arg)
#             logfile.write("VALID FLAGS ARE: -outdir, -design_dir, -min_score, -min_matched, -postprocess\n")
#             logfile.close()
#             sys.exit()
#
#     if "-outdir" in arg_list:
#         outdir_index = arg_list.index('-outdir') + 1
#         if outdir_index < len(arg_list):
#             outdir = arg_list[outdir_index]
#         else:
#             log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
#             logfile = open(log_filepath, "a+")
#             logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED\n")
#             logfile.close()
#             sys.exit()
#     else:
#         log_filepath = os.getcwd() + "/Nanohedra_PostProcess_log.txt"
#         logfile = open(log_filepath, "a+")
#         logfile.write("ERROR: OUTPUT DIRECTORY NOT SPECIFIED\n")
#         logfile.close()
#         sys.exit()
#
#     log_filepath = outdir + "/Nanohedra_PostProcess_log.txt"
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     try:
#         design_dir_path_index = arg_list.index("-design_dir") + 1
#         try:
#             design_dir_path = arg_list[design_dir_path_index]
#             if not os.path.exists(design_dir_path):
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: DESIGN DIRECTORY PATH SPECIFIED DOES NOT EXIST\n")
#                 logfile.close()
#                 sys.exit()
#         except IndexError:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: -design_dir FLAG FOLLOWED BY DESIGN DIRECTORY PATH IS REQUIRED\n")
#             logfile.close()
#             sys.exit()
#     except ValueError:
#         logfile = open(log_filepath, "w")
#         logfile.write("ERROR: -design_dir FLAG FOLLOWED BY DESIGN DIRECTORY PATH IS REQUIRED\n")
#         logfile.close()
#         sys.exit()
#
#     if "-min_score" in arg_list and "-min_matched" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
#         try:
#             min_score_index = arg_list.index("-min_score") + 1
#             try:
#                 min_score_str = arg_list[min_score_index]
#                 try:
#                     min_score = float(min_score_str)
#                     PostProcessUtils.score_filter(design_dir_path, min_score, outdir)
#                 except ValueError:
#                     logfile = open(log_filepath, "w")
#                     logfile.write("ERROR: MINIMUM SCORE SPECIFIED IS NOT A FLOAT\n")
#                     logfile.close()
#                     sys.exit()
#             except IndexError:
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
#                 logfile.close()
#                 sys.exit()
#         except ValueError:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
#             logfile.close()
#             sys.exit()
#
#     if "-min_matched" in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
#         try:
#             min_matched_index = arg_list.index("-min_matched") + 1
#             try:
#                 min_matched_str = arg_list[min_matched_index]
#                 try:
#                     min_matched = int(min_matched_str)
#                     PostProcessUtils.frag_match_count_filter(design_dir_path, min_matched, outdir)
#                 except ValueError:
#                     logfile = open(log_filepath, "w")
#                     logfile.write("ERROR: MINIMUM MATCHED FRAGMENT COUNT SPECIFIED IS NOT AN INTEGER\n")
#                     logfile.close()
#                     sys.exit()
#             except IndexError:
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
#                 logfile.close()
#                 sys.exit()
#         except ValueError:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
#             logfile.close()
#             sys.exit()
#
#     if "-min_matched" in arg_list and "-min_score" in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
#         min_matched_index = arg_list.index("-min_matched") + 1
#         min_score_index = arg_list.index("-min_score") + 1
#         try:
#             min_matched_str = arg_list[min_matched_index]
#             min_score_str = arg_list[min_score_index]
#
#             try:
#                 min_matched = int(min_matched_str)
#             except ValueError:
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: MINIMUM MATCHED FRAGMENT COUNT SPECIFIED IS NOT AN INTEGER\n")
#                 logfile.close()
#                 sys.exit()
#
#             try:
#                 min_score = float(min_score_str)
#             except ValueError:
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: MINIMUM SCORE SPECIFIED IS NOT A FLOAT\n")
#                 logfile.close()
#                 sys.exit()
#
#             PostProcessUtils.score_and_frag_match_count_filter(design_dir_path, min_score, min_matched, outdir)
#
#         except IndexError:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: -min_matched FLAG FOLLOWED BY A MINIMUM MATCHED FRAGMENT COUNT IS REQUIRED\n")
#             logfile.write("ERROR: -min_score FLAG FOLLOWED BY A MINIMUM SCORE IS REQUIRED\n")
#             logfile.close()
#             sys.exit()
#
#     if "-rank" in arg_list and "-min_matched" not in arg_list and "-min_score" not in arg_list and "-min_matched_ss" not in arg_list:
#         # try:
#         rank_index = arg_list.index("-rank") + 1
#         # try:
#         metric = arg_list[rank_index]
#         # try:
#         metric_str = str(metric)
#         if metric_str in ["score", "matched"]:
#             PostProcessUtils.rank(design_dir_path, metric_str, outdir)
#         else:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT RECOGNIZED\n")
#             logfile.close()
#             sys.exit()
#             # except ValueError:
#             #     logfile = open(log_filepath, "w")
#             #     logfile.write("ERROR: RANKING METRIC SPECIFIED IS NOT A STRING\n")
#             #     logfile.close()
#             #     sys.exit()
#         #     except IndexError:
#         #         logfile = open(log_filepath, "w")
#         #         logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
#         #         logfile.close()
#         #         sys.exit()
#         # except ValueError:
#         #     logfile = open(log_filepath, "w")
#         #     logfile.write("ERROR: -rank FLAG FOLLOWED BY A RANKING METRIC IS REQUIRED\n")
#         #     logfile.close()
#         #     sys.exit()
#
#     if "-min_matched_ss" in arg_list and "-min_matched" not in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list:
#         try:
#             min_matched_ss_index = arg_list.index("-min_matched_ss") + 1
#             try:
#                 min_matched_ss_str = arg_list[min_matched_ss_index]
#                 try:
#                     min_matched_ss = int(min_matched_ss_str)
#                     PostProcessUtils.ss_match_count_filter(design_dir_path, min_matched_ss, outdir)
#                 except ValueError:
#                     logfile = open(log_filepath, "w")
#                     logfile.write(
#                         "ERROR: VALUE SPECIFIED FOR MINIMUM SECONDARY STRUCTURE ELEMENT MATCHES IS NOT AN INTEGER\n")
#                     logfile.close()
#                     sys.exit()
#             except IndexError:
#                 logfile = open(log_filepath, "w")
#                 logfile.write("ERROR: -min_matched_ss FLAG IS NOT FOLLOWED BY A MINIMUM VALUE\n")
#                 logfile.close()
#                 sys.exit()
#         except ValueError:
#             logfile = open(log_filepath, "w")
#             logfile.write("ERROR: -min_matched_ss FLAG IS NOT FOLLOWED BY A MINIMUM VALUE\n")
#             logfile.close()
#             sys.exit()
#
#     if ("-min_matched" in arg_list or "-min_score" in arg_list) and "-rank" in arg_list:
#         logfile = open(log_filepath, "w")
#         logfile.write("ERROR:\n")
#         logfile.write("EITHER: FILTER BY SCORE AND/OR BY MINIMUM FRAGMENT(S) MATCHED\n")
#         logfile.write("OR: PERFORM RANKING BY SCORE\n")
#         logfile.write("OR: PERFORM RANKING BY NUMBER OF FRAGMENT(S) MATCHED\n")
#         logfile.close()
#         sys.exit()
#
#     if ("-min_matched" in arg_list or "-min_score" in arg_list) and "-min_matched_ss" in arg_list:
#         logfile = open(log_filepath, "w")
#         logfile.write("ERROR:\n")
#         logfile.write("EITHER: FILTER BY SCORE AND/OR BY MINIMUM FRAGMENT(S) MATCHED\n")
#         logfile.write("OR: FILTER BY MINIMUM SECONDARY STRUCTURE ELEMENT(S) MATCHED\n")
#         logfile.close()
#         sys.exit()
#
#     if "-rank" in arg_list and "-min_matched_ss" in arg_list:
#         logfile = open(log_filepath, "w")
#         logfile.write("ERROR:\n")
#         logfile.write("EITHER: FILTER BY MINIMUM SECONDARY STRUCTURE ELEMENT(S) MATCHED\n")
#         logfile.write("OR: PERFORM RANKING BY SCORE\n")
#         logfile.write("OR: PERFORM RANKING BY NUMBER OF FRAGMENT(S) MATCHED\n")
#         logfile.close()
#         sys.exit()
#
#     if "-min_matched" not in arg_list and "-min_score" not in arg_list and "-rank" not in arg_list and "-min_matched_ss" not in arg_list:
#         logfile = open(log_filepath, "w")
#         logfile.write("ERROR: POST PROCESSING FLAG REQUIRED\n")
#         logfile.close()
#         sys.exit()
