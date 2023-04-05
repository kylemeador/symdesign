import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}: Compare the identifiers in a file '
                                                 f'with those of a job allocation and return the array numbers')
    parser.add_argument('--job-file', help='File with the job identifiers', required=True)
    parser.add_argument('--identifier-file', help='File with the identifiers of interest', required=True)
    parser.add_argument('-sc', '--skip-commas', action='store_true',
                        help='Whether commas in files should be split or maintained')
    # parser.add_argument(*flags.output_file_args, help='The file where the found overlap should be written')

    args, additional_args = parser.parse_known_args()

    with open(args.job_file, 'r') as f:
        identifiers = f.readlines()

    with open(args.identifier_file, 'r') as f:
        remaining_ids = f.readlines()

    identifier_indices = []
    not_found = []
    for remaining_id in remaining_ids:
        try:
            identifier_index = identifiers.index(remaining_id)
        except ValueError:  # Not found
            not_found.append(remaining_id)
        else:
            identifier_indices.append(identifier_index)

    print(f'Found the identifier array indices:\n{",".join(map(str, identifier_indices))}')
    print(f'Found the missing identifiers:\n{",".join(not_found)}')
