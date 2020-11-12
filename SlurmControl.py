import argparse
import os
import subprocess
import sys
from glob import glob
from itertools import repeat

import SymDesignUtils as SDUtils


def find_list_indices(reference_cmds, query_ids):
    """Search for ID's present in supplied list in a reference list and return the indices of the reference list where
    they are found"""
    # full_lines_set = set(full_lines)
    # cmd_lines_set = set(cmd_lines)
    query_ids_sort = sorted(query_ids)
    # cmd_lines_sort = sorted(cmd_lines)

    idxs = []
    for i, cmd_id in enumerate(reference_cmds):  # _sort
        for id in query_ids_sort:
            if id in cmd_id:
                idxs.append(i)
                break
    idxs_sorted = sorted(idxs)
    print(','.join(str(i + 1) for i in idxs_sorted))

    return idxs_sorted


def filter_by_indices(array, _iterator, zero=True):
    offset = 1
    if zero:
        offset = 0
    return [_iterator[idx - offset] for idx in array]


def array_map(array, cont=False):
    if cont:
        sorted_input_cmd_lines = sorted(cmd_lines)
        continue_commands_interest = filter_by_indices(array, sorted_input_cmd_lines)
        # continue_commands_interest = array_filter(input_indices, sorted_input_cmd_lines)
        with open(sys.argv[1] + '_continue_cmds', 'w') as f:
            f.write('\n'.join(i for i in continue_commands_interest))
    else:
        commands_interest = filter_by_indices(array, cmd_lines)
        sorted_commands = sorted(commands_interest)
        input_indices = []
        for s_command in sorted_commands:
            input_indices.append(cmd_lines.index(s_command))

        with open(sys.argv[1] + '_array', 'w') as f:
            f.write('\n'.join(idx for idx in input_indices))


def concat_job_to_array(job_id, array):
    return ['%s_%s' % (str(job_id), str(idx)) for idx in array]


def scancel(job_id):
    _cmd = ['scancel', job_id]
    p = subprocess.Popen(_cmd)

    # get the results here
    # return status


def error_type(job_file):
    if job_file:
        fail_p = subprocess.Popen(['grep', '\"DUE TO NODE\"', job_file],  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fail = fail_p.communicate()
        mem_p = subprocess.Popen(['grep', '\"slurmstepd: error: Exceeded job memory limit\"', job_file],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        mem = mem_p.communicate()

        if mem != '':
            return 'memory'
        elif fail != '':
            return 'failure'
        else:
            return 'other'
    return None


def link_pair(pair, force=False):
    """Combine docking files of one docking combination with another

    Args:
        pair (tuple): source file, destination file (link)
    Keyword Args:
        force=False (bool): Whether to remove links before creation
    """
    if force:
        os.remove(pair[1])
    os.symlink(*pair)  # , target_is_directory=True)


def job_array_failed(job_id, output_dir=os.path.join(os.getcwd(), 'output')):
    """Returns an array for each of the errors encountered. All=True returns the set"""
    matching_jobs = glob('%s%s*%s*' % (output_dir, os.sep, job_id))
    potential_errors = [job if os.path.getsize(job) > 0 else None for i, job in enumerate(matching_jobs)]
    print(','.join(str(i) for i, potential_error in enumerate(potential_errors) if potential_error))
    parsed_errors = list(map(error_type, potential_errors))
    sorted_array_jobs = {job_file: int(os.path.splitext(job_file.split('_')[-1])[0]) for job_file in matching_jobs}
    # generate a dictionary of the job_file to array_id
    # for job in matching_jobs:
    #     array_id = os.path.splitext(job.split('_')[-1])[0]
    #     sorted_array_jobs[array] = job
    memory_array = [sorted_array_jobs[matching_jobs[i]] for i, error in enumerate(parsed_errors) if error == 'memory']
    failure_array = [sorted_array_jobs[matching_jobs[i]] for i, error in enumerate(parsed_errors) if error == 'failure']
    other_array = [sorted_array_jobs[matching_jobs[i]] for i, error in enumerate(parsed_errors) if error == 'other']
    print(memory_array)

    return memory_array, failure_array, other_array


def parse_script(script_file):
    with open(script_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '--command_file' in line:
                command_distribution_cmd_l = line.split()
                command_file_idx = command_distribution_cmd_l.index('--command_file') + 1
                command_file = command_distribution_cmd_l[command_file_idx]

    return command_file


def change_script_array(script_file, array):
    """Take a script file and replace the array line with a new array"""
    with open(script_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '#SBATCH -a' in line or '#SBATCH --array' in line:
                lines[i] = '#SBATCH --array=%s' % ','.join(str(a) for a in array)

    new_script = '%s_%s' % (os.path.splitext(script_file)[0], 're-do_SLURM_failures.sh')
    with open(new_script, 'w') as f:
        f.write('\n'.join(line for line in map(str.strip, lines)))

    return new_script


def job_failed():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nControl all SLURM input/output including:\n')
    # ---------------------------------------------------
    parser.add_argument('-d', '--directory', type=str, help='Directory where Job output is located. Default=CWD')
    parser.add_argument('-f', '--file', type=str, help='File where the commands for the array were kept',
                        default=None, required=True)
    parser.add_argument('-mp', '--multi_processing', action='store_true',
                        help='Should job be run with multiprocessing?\nDefault=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out?\nDefault=False')
    subparsers = parser.add_subparsers(title='SubModules', dest='sub_module',
                                       description='These are the different modes that designs are processed',
                                       help='Chose one of the SubModules followed by SubModule specific flags. To get '
                                            'help on a SubModule such as specific commands and flags enter:\n%s\n\nAny'
                                            'SubModule help can be accessed in this way' % 'SPACE FILLER')
    # ---------------------------------------------------
    parser_fail = subparsers.add_parser('fail', help='Find job failures')
    parser_fail.add_argument('-a', '--array', action='store_true', help='Whether the failures should be returned as an '
                                                                        'array')
    # parser_fail.add_argument('-f', '--file', type=str, help='File where the commands for the array were kept',
    #                          default=None, required=True)
    parser_fail.add_argument('-j', '--job_id', type=str, help='What is the JobID provided by SLURM upon execution?',
                             required=True)
    parser_fail.add_argument('-m', '--mode', type=str, choices=['memory', 'other'],
                             help='What type of failure should be located')
    parser_fail.add_argument('-s', '--script', type=str, help='What is the Script used to creat the job_id?')
    # parser_fail.add_argument('-r', '--return_array', action='store_true', help='Whether the failures should be returned'
    #                                                                            ' as an array')

    # ---------------------------------------------------
    parser_fail = subparsers.add_parser('filter', help='Find job failures')
    # parser_fail.add_argument('-f', '--file', type=str, help='File where the commands for the array were kept',
    #                          default=None, required=True)
    parser_fail.add_argument('-e', '--exclude', action='store_true')
    parser_fail.add_argument('-r', '--running', action='store_true')
    parser_fail.add_argument('-q', '--query',  type=str, help='File with the query ID\'s for reference commands',
                             required=True)
    # ---------------------------------------------------
    parser_link = subparsers.add_parser('link', help='Find job failures')
    parser_link.add_argument('-F', '--force', action='store_true')

    args, additional_flags = parser.parse_known_args()
    if args.sub_module == 'fail':  # -a array, -j job_id, -m mode, -s script
        # do array
        memory, failure, other = job_array_failed(args.job_id)  # , output_dir=args.directory)
        print('Memory error size:', len(memory))
        print('Failure error size:', len(failure))
        print('Other error size:', len(other))
        all_array = sorted(set(memory + failure + other))
        if args.script:
            # commands = SDUtils.to_iterable(args.file)
            args.file = parse_script(args.script)
            script_with_new_array = change_script_array(args.script, all_array)
            print('\n\nRun new script with:\nsbatch %s' % script_with_new_array)
            if len(memory) > 0:
                print('Memory failures may require you to rerun with a higher memory. It is suggested to edit the above'
                      ' script to include ~10-20% more memory')
        else:
            commands = SDUtils.to_iterable(args.file)
            print('There are a total of commmands:', len(commands))
        # if args.array:

        # else:
            restart_memory = [commands[idx] for idx in memory]
            restart_failure = [commands[idx] for idx in failure]
            restart_other = [commands[idx] for idx in other]
            SDUtils.io_save(restart_memory, filename='%s_%s' % (args.file, 'memory_failures'))
            SDUtils.io_save(restart_failure, filename='%s_%s' % (args.file, 'other_failures'))
            SDUtils.io_save(restart_other, filename='%s_%s' % (args.file, 'other_output'))

    elif args.sub_module == 'scancel':
        array_ids = SDUtils.to_iterable(args.file)
        job_array = concat_job_to_array(args.job_id, array_ids)
        status_array = [scancel(job) for job in job_array]
        # mode = sys.argv[1]
        # job_id = sys.argv[2]
        # array_file = sys.argv[3]
        # array_ids = file_to_iterable(array_file)
        # job_array = concat_job_to_array(job_id, array_ids)
        # status_array = [scancel(job) for job in job_array]
    elif args.sub_module == 'filter':  # -e exclude, -r running, -q query
        reference_l = SDUtils.to_iterable(args.file)
        query_ids = SDUtils.to_iterable(args.query)

        array = find_list_indices(reference_l, query_ids)
        filtered_reference_l = filter_by_indices(array, reference_l)
        if args.exclude:
            modified_reference = list(set(reference_l) - set(filtered_reference_l))
            SDUtils.io_save(modified_reference, filename=args.file + '_excluded_%s' % os.path.basename(args.query))
        else:
            SDUtils.io_save(filtered_reference_l, filename=args.file + '_filtered_%s' % os.path.basename(args.query))
    elif args.sub_module == 'link':
        output_dir = os.path.join(os.getcwd(), args.directory)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        reference_l = SDUtils.to_iterable(args.file)
        link_names = map(os.path.basename, reference_l)
        link_name_dirs = list(map(os.path.join, repeat(output_dir), link_names))

        for pair in zip(reference_l, link_name_dirs):
            link_pair(pair, force=args.force)

            # if args.running:
        #     cont = True
        # else:
        #     cont = False
        # array_map(array, cont=cont)
        # cmds = sys.argv[1]  # 'all_rmsd.cmd'
        # full = sys.argv[2]  # 'fully_sampled_designs_201104_pm'
        # # Grab the fully finished designs
        # with open(full, 'r') as f:
        #     lines = f.readlines()
        #     lines = list(map(os.path.basename, lines))
        #     lines = list(map(str.strip, lines))
        #     full_lines = lines
        #
        # # Grab the design ID containing directory paths
        # with open(cmds, 'r') as f:
        #     lines = f.readlines()
        #     #     lines = map(os.path.dirname, lines)
        #     #     lines = map(os.path.dirname, lines)
        #     #     lines = list(map(os.path.basename, lines))
        #     lines = list(map(str.strip, lines))
        #     cmd_lines = lines
