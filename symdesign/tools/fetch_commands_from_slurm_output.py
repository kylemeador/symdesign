import argparse
import os

from symdesign import flags
from symdesign.utils import to_iterable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}: Compare the items in two list files '
                                                 f'through multiple set operators')
    parser.add_argument(*flags.file_args, nargs='*', required=True,
                        help='A file with the output of SLURM sjobs or simply one containing SLURM jobIDs')
    parser.add_argument(*flags.directory_args, required=True,
                        help='The directory with SLURM output for the job in question')
    parser.add_argument(*flags.output_file_args, required=True,
                        help='The file where the selected commands should be written')

    args, additional_args = parser.parse_known_args()

    commands = []
    for file in to_iterable(args.file):
        for job_id in to_iterable(file):
            # Ensure if the whole sjobs output, the only thing parsed is the job_id
            job_id, *other = job_id.split()
            job_output = os.path.join(args.directory, f'{job_id}.out')
            print(job_output)
            if os.path.exists(job_output):
                with open(job_output, 'r') as f:
                    command = f.readline()
                commands.append(command)
            else:
                print(f"Couldn't find the file {job_output}")

    if commands:
        with open(args.output_file, 'w') as f:
            f.write('%s\n' % '\n'.join(commands))
