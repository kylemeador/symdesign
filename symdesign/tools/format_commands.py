import argparse
import os
import sys

from symdesign import flags, utils

if __name__ == '__main__':
    # distribute = 'distribute'
    # parser_distribute = {distribute: dict()}
    parser_distribute = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}\nFormat commands for slurm')
    distribute_arguments = {
        (f'--command',): dict(help='The command to distribute over the input specification'),  # required=True,
        (f'--{flags.number_of_commands}',): dict(type=int, required=True, help='The number of commands to spawn'),
        flags.output_file_args: dict(required=True, help='The location to write the commands')
    }
    for args, kwargs in distribute_arguments.items():
        parser_distribute.add_argument(*args, **kwargs)
    args, additional_args = parser_distribute.parse_known_args()

    formatted_command = sys.argv[1:]
    print(formatted_command)
    n_cmds_index = formatted_command.index(f'--{flags.number_of_commands}')
    formatted_command.pop(n_cmds_index + 1)
    formatted_command.pop(n_cmds_index)
    of_index = formatted_command.index(f'--{flags.output_file}')
    formatted_command.pop(of_index + 1)
    formatted_command.pop(of_index)
    formatted_command = 'python ' + utils.path.program_exe + ' ' + ' '.join(formatted_command)
    print(formatted_command)
    with open(args.output_file, 'w') as f:
        for i in range(args.number_of_commands):
            f.write(f'{formatted_command} --range {100 * i / args.number_of_commands:.4f}'
                    f'-{100 * (1 + i) / args.number_of_commands:.4f}\n')

    print(f"Distribution file written to '{args.output_file}'")
