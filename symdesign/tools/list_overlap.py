import argparse
import os
import sys

import numpy as np

from symdesign import flags, utils


def set_overlap(sets_of_interest):
    # ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
    num_sets = len(sets_of_interest)
    set_ops = {'|': set.union, '&': set.intersection, '-': set.difference, '^': set.symmetric_difference}
    set_operator_description = {'|': 'Union', '&': 'Intersection', '-': 'Difference', '^': 'Symmetric Difference'}

    comparison_set = [[[] for x in range(num_sets)] for y in range(num_sets)]
    # comparison_set_descriptor = [[[] for x in range(num_lists)] for y in range(num_lists)]
    # comparison_set = np.zeros((num_lists, num_lists))
    # comparison_set = np.zeros((num_sets, num_sets))
    comparison_set_descriptor = np.zeros((num_sets, num_sets))
    # print(str('\n'.join(list(set_operator_description.items()))))
    op_char = input('Enter the set operand from one of:\n\t%s\nThen press Enter\n' %
                    '\n\t'.join(f'{tup1} = {tup2}' for tup1, tup2 in set_operator_description.items()))
    try:
        set_op_func = set_ops[op_char]
    except KeyError as e:
        print(f'{e.args} not available!')
        sys.exit()

    for i in range(num_sets):
        for j in range(num_sets):
            comparison_set[i][j] = set_op_func(sets_of_interest[i], sets_of_interest[j])
            comparison_set_descriptor[i][j] = len(comparison_set[i][j])

    print(f'{set_operator_description[op_char]} operation produced final set matrix with lengths:\n'
          f'{comparison_set_descriptor}\n')

    try:
        row = int(input('What comparison are you interested in? Hit enter to terminate\nRow #:'))
        column = int(input('Column #:'))
    except ValueError:
        sys.exit()
    try:
        set_of_interest = comparison_set[row - 1][column - 1]
    except IndexError as e:
        print(f'{e.args} is out of range')
        sys.exit()

    return sorted(set_of_interest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)}: Compare the items in two list files '
                                                 f'through multiple set operators')
    parser.add_argument(*flags.file_args, nargs='+', help='Files with items to be compared. Two required',
                        required=True)
    parser.add_argument('-sc', '--skip-commas', action='store_true',
                        help='Whether commas in files should be split or maintained')
    parser.add_argument(*flags.output_file_args, help='The file where the found overlap should be written')

    args, additional_args = parser.parse_known_args()

    num_lists = len(args.file)
    if num_lists < 2:
        raise utils.InputError(f'2 or more files required. Only got\n{args.file}')

    sets = []
    for file in args.file:
        file_list = utils.to_iterable(file, ensure_file=True, skip_comma=args.skip_commas)
        # Uncomment below for use with ugly prefixes or suffixes
        # if i == 1:
        #     for j, _item in enumerate(_list[i]):
        #         _list[i][j] = _item[:-10]  # [9:] if removing /mntpoint, [:-10] if removing design.sh
        file_set = set(file_list)
        sets.append(file_set)

        print(f'List of {file} has {len(file_list)} values. Set has {len(file_set)} values')

    list_of_interest = set_overlap(sets)
    utils.io_save(list_of_interest, file_name=args.output_file)
