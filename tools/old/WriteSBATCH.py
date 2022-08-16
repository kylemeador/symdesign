import argparse
import os
import subprocess

# Globals
shebang = '#!/bin/bash\n'
sb = '#SBATCH --'
user = str(os.environ.get('USER'))
user_string = '%u'
job_string = '%x'
# nodes = 1
ntasks = 1
mem = 4000
max_jobs = '800'
_cmd = '0'
output = 'output'
process_commands_script = '$SymDesign/dependencies/bin/diSbatch.sh'


def prepare_options(_options, descriptor=False, cpu=False, mail=False, array=False):
    """Take input options and format a list of integers to query those arguments from the user

    Args:
        _options (int): The number of options to query
    Keyword Args:
        descriptor=False (bool): Whether to query descriptor options (job_name, output, error)
        cpu=False (bool): Whether to query descriptor options (partition, ntasks, nodes, mem, mem_per_cpu)
        mail=False (bool): Whether to query descriptor options (mail_user, mail_type)
        array=False (bool): Whether to query descriptor options (array)
    Returns:
        option_numbers (list): [1, 2, 3]
    """
    option_numbers = []
    if descriptor and cpu and mail and array:
        option_numbers = [i for i in range(_options)]
    else:
        if not descriptor:
            for i in [0, 1, 2]:
                option_numbers.append(i)
        if not cpu:
            for i in [3, 4, 5]:    
                option_numbers.append(i)
        if not mail:
            for i in [6, 7]:
                option_numbers.append(i)
        if array:
            option_numbers.append(8)

    return option_numbers


def write_args(option_nums):
    """Take a list of options as integers and if an integer is present, query the user for their value for that option

    Args:
        option_nums (list): [1, 4, 5]
    Returns:
        file_string (str): A string of options to write ot the SBATCH file taken from the options dictionary
    """
    file_string = shebang
    if option_nums:
        print('If any option should be excluded, input False for \'Your Value=\'')
        for i, option in enumerate(options):
            if i in option_nums:
                variable = input('\nValue for \'%s\'?\nDefault=\'%s\'\nDescription=\'%s\'. Press ENTER for Default\n'
                                 'Your Value=' % (option, options[option]['default'], options[option]['description']))
                if variable != 'False':
                    file_string += sb + option + options[option]['delimiter'] + variable + '\n'
    else:
        for option in options:
            file_string += sb + option + options[option]['delimiter'] + options[option]['default'] + '\n'

    return file_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBATCH script writer.\nUse for updating the parameters in a SLURM Job'
                                                 ' submission.\nIf no specific options are included, all options will '
                                                 'be queried, else, only the options for --descriptors, --cpu, and '
                                                 '--mail will be queried. Supply --default for all default options.\n'
                                                 'If there is a list of commands to be executed, put these is a file '
                                                 'separated by newlines and specify -f path/to/file')
    parser.add_argument('filename', type=str, help='What should the file be named? Should be appended with \'.sh\'')
    parser.add_argument('-f', '--filepath_for_command', type=str, help='Is there a file with commands to be executed? '
                                                                       'What is this file? Default=None', default=None)
    parser.add_argument('-d', '--descriptors', action='store_false', help='Modify descriptive options? (job-title, '
                                                                          'error, output). Default=True')
    parser.add_argument('-c', '--cpu', action='store_false', help='Modifiy cpu options? (partition, nodes, mem). '
                                                                  'Default=True')
    parser.add_argument('-m', '--mail', action='store_false', help='Modifiy mail options? (mail-user, mail-type). '
                                                                   'Default=True')
    parser.add_argument('--ntasks', type=int, help='Number of tasks for each process? Default=1', default=ntasks)
    # parser.add_argument('--nodes', type=int, help='How many nodes for each process? Default=1', default=nodes)
    parser.add_argument('--mem', type=int, help='How much memory for the job? (In KB). Default=4000', default=mem)
    # parser.add_argument('--mem_per_cpu', type=int, help='How much memory/cpu? (In MB)', default=4000)
    parser.add_argument('--default', action='store_true', help='Write all options as default?')
    args = parser.parse_args()

    if args.filepath_for_command:
        # Calculate the size of the command array based on number of lines
        wc_command = subprocess.run(['wc', '-l', args.filepath_for_command], capture_output=True)
        _cmd = '1-' + wc_command.stdout.split()[0].decode('utf-8') + '%' + max_jobs
        jobname = user_string + '_' + os.path.basename(args.filepath_for_command)
    else:
        args.filepath_for_command = False
        jobname = user_string + '_job'

    outpath = os.path.join(os.getcwd(), output)
    if not os.path.exists(os.path.join(os.getcwd(), output)):
        os.makedirs(os.path.join(os.getcwd(), output))

    options = {'job-name': {'default': jobname, 'description': 'A title for your job, where %u is $USER',
                            'delimiter': '='},
               'output': {'default': outpath + '%x_%A_%a.out', 'description': ('Standard Output path, where %s is the '
                                                                               'Job Name, %A is the SLURM-JOB-ID and %a'
                                                                               ' is the SLURM-ARRAY-ID') % job_string,
                          'delimiter': '='},
               'error': {'default': outpath + '%A_%a.er', 'description': 'Standard Error path, where %A is the '
                                                                         'SLURM-JOB-ID and %a is the SLURM-ARRAY-ID',
                         'delimiter': '='},
               'partition': {'default': 'long', 'description': 'Where or how long to allocate JOB. See command `snodes`'
                                                               ' for current cluster availability', 'delimiter': '='},
               # 'nodes': {'default': str(args.nodes), 'description': 'How many nodes should be used per process?',
               #           'delimiter': '='},
               'ntasks': {'default': str(args.ntasks), 'description': 'How many tasks should be used per process?',
                          'delimiter': '='},
               'mem': {'default': str(args.mem), 'description': 'How much memory for each process?', 'delimiter': '='},
               # 'mem-per-cpu': {'default': str(args.mem_per_cpu), 'description': 'How much memory for each CPU?',
               #                 'delimiter': '='},
               'mail-user': {'default': user_string, 'description': 'Where to send job status updates via email? Where'
                                                                    '%s is $USER.' % user_string,
                             'delimiter': '='},
               'mail-type': {'default': 'ALL', 'description': 'What status should be sent, where status options are '
                                                              'BEGIN,END,ERROR,ALL', 'delimiter': '='},
               'array': {'default': str(_cmd), 'description': 'How big should the JOB ARRAY be? (how many separate '
                                                              'times should the input command be repeated)',
                         'delimiter': '='},
               'no-requeue': {'default': True, 'description': 'Whether to prevent failures from requeueing',
                              'delimiter': ' '}}

    with open(args.filename, 'w') as f:
        if args.default:  # TODO remove confusing double negatives in prepare_options and args.default option
            f.write(write_args(list()))
        else:
            f.write(write_args(prepare_options(len(options), descriptor=args.descriptors, cpu=args.cpu, mail=args.mail,
                                               array=args.filepath_for_command)))
        f.write('\n')
        if args.filepath_for_command:
            f.write('bash %s %s\n' % (process_commands_script, args.filepath_for_command))
        else:
            f.write(input('What command do you want to run?\n Command:'))

    print('SBATCH Script written to %s' % args.filename)
