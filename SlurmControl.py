import argparse
import os
import subprocess
import sys
from glob import glob

import SymDesignUtils as SDUtils


def main():
    full_lines_set = set(full_lines)
    # cmd_lines_set = set(cmd_lines)
    full_lines_sort = sorted(full_lines)
    # cmd_lines_sort = sorted(cmd_lines)

    idxs = []
    for i, cmd_id in enumerate(cmd_lines):  # _sort
        for id in full_lines_sort:
            if id in cmd_id:
                idxs.append(i)
                break
    idxs_sorted = sorted(idxs)
    print(','.join(str(i + 1) for i in idxs_sorted))

    return idxs_sorted


def array_filter(array, _iterator, zero=True):
    offset = 1
    if zero:
        offset = 0
    commands_interest = []
    for number in array:
        commands_interest.append(_iterator[number - offset])
    return commands_interest


def array_map(array, cont=False):
    if cont:
        sorted_input_cmd_lines = sorted(cmd_lines)
        continue_commands_interest = array_filter(array, sorted_input_cmd_lines)
        # continue_commands_interest = array_filter(input_indices, sorted_input_cmd_lines)
        with open(sys.argv[1] + '_continue_cmds', 'w') as f:
            f.write('\n'.join(i for i in continue_commands_interest))
    else:
        commands_interest = array_filter(array, cmd_lines)
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
        return None


def job_array_failed(job_id, output_dir=os.path.join(os.getcwd(), 'output')):
    matching_jobs = glob('%s%s*%s*' % (output_dir, os.sep, job_id))
    print('Potential jobs:', len(matching_jobs))
    potential_errors = [job if os.path.getsize(job) > 0 else None for job in matching_jobs]
    print('Potential errors:', len(potential_errors))
    parsed_errors = list(map(error_type, potential_errors))
    memory_array = [i for i, error in enumerate(parsed_errors) if error == 'memory']
    other_array = [i for i, error in enumerate(parsed_errors) if error == 'other']
    print('Memory error size:', len(memory_array))
    print('Other error size:', len(other_array))

    return memory_array, other_array


def job_failed():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nControl all SLURM input/output including:\n')
    # ---------------------------------------------------
    parser.add_argument('-d', '--directory', type=str, help='Directory where Job output is located. Default=CWD')
    parser.add_argument('-f', '--file', type=str, help='File with command(s)', default=None)
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
    parser_fail.add_argument('-a', '--array', action='store_true')
    parser_fail.add_argument('-f', '--file', type=str, help='File where the commands for the array were kept',
                             default=None, required=True)
    parser_fail.add_argument('-j', '--job_id', type=str, help='What is the JobID provided by SLURM upon execution?')
    parser_fail.add_argument('-m', '--mode', type=str, choices=['memory', 'other'],
                             help='What type of failure should be located')

    args, additional_flags = parser.parse_known_args()

    if args.sub_module == 'fail':
        if args.array:
            # do array
            memory, other = job_array_failed(args.job_id)  # , output_dir=args.directory)
            commands = SDUtils.to_iterable(args.file)
            print('There are a total of commmands:', len(commands))
            restart_memory = [commands[idx] for idx in memory]
            restart_other = [commands[idx] for idx in other]
            SDUtils.io_save(restart_memory, filename='%s_%s' % (args.file, 'memory_failures'))
            SDUtils.io_save(restart_other, filename='%s_%s' % (args.file, 'other_failures'))
        else:
            job_failed()

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
    elif args.sub_module == 'file_to_array':
        if 'running' in sys.argv:
            cont = True
        else:
            cont = False

        array = main()
        array_map(array, cont=cont)

        cmds = sys.argv[1]  # 'all_rmsd.cmd'
        full = sys.argv[2]  # 'fully_sampled_designs_201104_pm'
        # Grab the fully finished designs
        with open(full, 'r') as f:
            lines = f.readlines()
            lines = list(map(os.path.basename, lines))
            lines = list(map(str.strip, lines))
            full_lines = lines

        # Grab the design ID containing directory paths
        with open(cmds, 'r') as f:
            lines = f.readlines()
            #     lines = map(os.path.dirname, lines)
            #     lines = map(os.path.dirname, lines)
            #     lines = list(map(os.path.basename, lines))
            lines = list(map(str.strip, lines))
            cmd_lines = lines
