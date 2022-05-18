"""
To profile CPU usage and execution times, run your script with the command format:
python -m cProfile -o [output_file.name] [SCRIPT] [ARGS]
Then run this with:
python ProfileScripts.py -f [output_file.name] [-s] [-t]

To profile memory (RAM) usage, run your script with the command format:
python -m memory_profiler [SCRIPT] [ARGS] > output_file.name
"""
import os
import argparse
import pstats
from pstats import SortKey

import SymDesignUtils as SDUtils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sort_by', type=str, default='CUMULATIVE',
                        choices=list(key for key in SortKey.__dict__.keys() if not key.startswith('_')),
                        help='The specific key to sort statistics by. Choices are defined by pstats.SortKey. '
                             'Default=CUMULATIVE')
    parser.add_argument('-f', '--file', type=os.path.abspath, metavar=SDUtils.ex_path('file_with_stats.txt'),
                        help='File containing the stats from a cProfile run', default=None)
    parser.add_argument('-t', '--top', type=int, help='The number of results to view', default=10)

    args, additional_args = parser.parse_known_args()

    p = pstats.Stats(args.file)
    p.strip_dirs().sort_stats(getattr(SortKey, args.sort_by)).print_stats(args.top)
