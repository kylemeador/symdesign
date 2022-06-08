import argparse
import os

import numpy as np

from SymDesignUtils import to_iterable, io_save


def set_overlap(_set):
    # ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
    num_sets = len(_set)
    set_ops = {'|': set.union, '&': set.intersection, '-': set.difference, '^': set.symmetric_difference}
    set_operator_description = {'|': 'Union', '&': 'Intersection', '-': 'Difference', '^': 'Symmetric Difference'}

    comparison_set = [[[] for x in range(num_sets)] for y in range(num_sets)]
    # comparison_set_descriptor = [[[] for x in range(num_lists)] for y in range(num_lists)]
    # comparison_set = np.zeros((num_lists, num_lists))
    # comparison_set = np.zeros((num_sets, num_sets))
    comparison_set_descriptor = np.zeros((num_sets, num_sets))
    # print(str('\n'.join(list(set_operator_description.items()))))
    op_char = input('Enter the set operand from one of:\n\t%s\nThen press Enter\n' %
                    '\n\t'.join('%s = %s' % tup for tup in set_operator_description.items()))
    try:
        set_op_func = set_ops[op_char]
    except KeyError as e:
        exit('%s not available!' % e.args)

    for i in range(num_sets):
        for j in range(num_sets):
            comparison_set[i][j] = set_op_func(_set[i], _set[j])
            comparison_set_descriptor[i][j] = len(comparison_set[i][j])

    print('%s operation produced final set matrix with lengths:\n%s\n' %
          (set_operator_description[op_char], comparison_set_descriptor))

    try:
        row = int(input('What comparison are you interested in? Hit enter to terminate\nRow #:'))
        column = int(input('Column #:'))
    except ValueError:
        exit()
    try:
        set_of_interest = comparison_set[row - 1][column - 1]
    except IndexError as e:
        exit('%s is out of range' % e.args)

    list_of_interest = sorted(set_of_interest)
    io_save(list_of_interest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s: Compare the items in two list files through multiple set '
                                                 'operators.' % os.path.basename(__file__))
    parser.add_argument('-f', '--files', nargs='+', help='Files with items to be compared. Two required',
                        required=True, default=None)
    parser.add_argument('-sc', '--skip_commas', action='store_true',
                        help='Whether commas in files should be split or maintained')
    args = parser.parse_args()

    num_lists = len(args.files)
    file = [[] for x in range(num_lists)]
    _list = [[] for x in range(num_lists)]
    sets = [[] for x in range(num_lists)]

    if num_lists < 1:
        exit('No list files found: %' % args.files)
    for i in range(num_lists):
        file[i] = args.files[i]
        _list[i] = to_iterable(file[i], ensure_file=True, skip_comma=args.skip_commas)
        # Uncomment below for use with ugly prefixes or suffixes
        # if i == 1:
        #     for j, _item in enumerate(_list[i]):
        #         _list[i][j] = _item[:-10]  # [9:] if removing /mntpoint, [:-10] if removing design.sh
        sets[i] = set(_list[i])
        # file[i] = sys.argv[i + 1]
        print('List of %s has %d values. Set has %d values' % (file[i], len(_list[i]), len(sets[i])))

    set_overlap(sets)
