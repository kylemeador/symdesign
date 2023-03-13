import argparse
import logging
import os

from symdesign.resources.query.pdb import retrieve_pdb_entries_by_advanced_query
from symdesign.utils import path as putils

logger = logging.getLogger(__name__)
logger.setLevel(20)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query the PDB for entries\n')
    parser.add_argument('-f', '--file', nargs='*', type=os.path.abspath, metavar=f'{putils.ex_path("pdblist.file")}',
                        help='File(s) containing EntryID codes. Can be newline or comma separated')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Whether files should be downloaded. Default=False')
    parser.add_argument('-p', '--input-pdb-directory', type=os.path.abspath,
                        help='Where should reference PDB files be found? Default=CWD', default=os.getcwd())
    parser.add_argument('-i', '--input-pisa-directory', type=os.path.abspath,
                        help='Where should reference PISA files be found? Default=CWD', default=os.getcwd())
    parser.add_argument('-o', '--output-directory', type=os.path.abspath,
                        help='Where should interface files be saved?')
    parser.add_argument('-q', '--query-web', action='store_true',
                        help='Should information be retrieved from the web?')
    parser.add_argument('-db', '--database', type=str, help='Should a database be connected?')
    args = parser.parse_args()

    retrieve_pdb_entries_by_advanced_query()
