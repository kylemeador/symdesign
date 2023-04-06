import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}: Compare the identifiers in a file '
                                                 f'with those of a job, returning the numbers of the '
                                                 f'identifiers in the --identifier-file from the job array')
    parser.add_argument('-j', '--job-file', help='File containing all job identifiers', required=True)
    parser.add_argument('-i', '--identifier-file', help='File with the identifiers of interest', required=True)
    parser.add_argument('-n', '--not-present', action='store_true',
                        help='Whether the indices displayed are the inverse of the ones passed')
    parser.add_argument('-z', '--zero-index', action='store_true',
                        help='Should identifier array be output as a zero indexed array?')
    # parser.add_argument('-sc', '--skip-commas', action='store_true',
    #                     help='Whether commas in files should be split or maintained')
    # parser.add_argument(*flags.output_file_args, help='The file where the found overlap should be written')

    args, additional_args = parser.parse_known_args()

    with open(args.job_file, 'r') as f:
        job_identifiers = f.readlines()

    with open(args.identifier_file, 'r') as f:
        interest_ids = f.readlines()

    identifier_indices = set()
    not_found = []
    for interest in interest_ids:
        try:
            identifier_index = job_identifiers.index(interest)
        except ValueError:  # Not found
            not_found.append(interest)
        else:
            identifier_indices.add(identifier_index)

    if args.not_present:
        print('Taking the inverse of the --identifier-file ids')
        identifier_indices = set(range(len(job_identifiers))).difference(identifier_indices)

    identifier_indices = sorted(identifier_indices)

    if args.zero_index:
        print(f'Found the identifier array indices (zero-indexed):\n{",".join(map(str, identifier_indices))}')
    else:
        print(f'Found the identifier array indices (one-indexed):\n'
              f'{",".join(map(str, [i + 1 for i in identifier_indices]))}')
    print(f'Found the missing identifiers:\n{",".join(not_found)}')
