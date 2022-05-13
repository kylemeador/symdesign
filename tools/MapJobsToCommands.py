import os
import sys
import argparse
import logging

import SymDesignUtils as SDUtils

logging.getLogger().setLevel(logging.DEBUG)


def map_commands_started(output_directory, _job_id):
    command_map = {}
    array_map = {}
    for file in os.listdir(output_directory):
        if file.startswith(_job_id):
            if file.endswith('.out'):
                with open(os.path.join(output_directory, file), 'r') as f:
                    array_num = f.readline()
                    _command = f.readline()
                    command_map[_command] = array_num
                    array_map[array_num] = _command

    return command_map, array_map


def map_commands(command_file):
    command_map = {}
    array_map = {}
    with open(command_file, 'r') as f:
        # all_commands = f.readlines()
        for i, _command in enumerate(f.readlines(), 1):
            command_map[_command] = i
            array_map[i] = _command

    return command_map, array_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Map design commands to array number')
    parser.add_argument('-p', '--processed', action='store_true', help='Map processed commands? Default = False. '
                                                                       'If true, provide directory where output logs '
                                                                       'are written as -c /path/to/output and jobID as'
                                                                       '-j user_designs.cmd4789392')
    parser.add_argument('-c', '--commands', type=str, help='Path to command file.', default=None)
    parser.add_argument('-j', '--job_id', type=str, help='Job name + jobID number of sbatch submission', default='')
    parser.add_argument('-o', '--no_output', action='store_true', help='Output map objects to current working '
                                                                       'directory? Default = False')
    args = parser.parse_args()
    if not args.commands:
        sys.exit('Must specify a command file!')

    if args.processed:
        command, array = map_commands_started(args.commands, args.job_id)
    else:
        command, array = map_commands(args.commands)
    if not args.no_output:
        SDUtils.pickle_object(command, args.commands + '_CommandToArray')
        SDUtils.pickle_object(array, args.commands + '_ArrayToCommand')
