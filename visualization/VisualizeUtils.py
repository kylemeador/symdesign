from __future__ import annotations
import os
import re
import sys
from pickle import load
from glob import glob

# import pandas as pd
# import numpy as np
from pymol import finish_launching, cmd, stored
finish_launching(['pymol', '-q'])
# finish_launching()
# from Pose import Pose
# from SymDesignUtils import possible_symmetries

# TODO integration of the capabilities of SymDesign with pymol plugins to manipulate and design
#  https://raw.githubusercontent.com/Pymol-Scripts/Pymol-script-repo/master/plugins/SuperSymPlugin.py
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T', 'O32': 'O', 'O42': 'O', 'O43': 'O',
                       'I23': 'I', 'I25': 'I', 'I35': 'I', 'T23': 'T', 'O23': 'O', 'O24': 'O', 'O34': 'O',
                       'T': 'T', 'T:{C2}': 'T', 'T:{C3}': 'T',
                       'T:{C2}{C3}': 'T', 'T:{C3}{C2}': 'T', 'T:{C3}{C3}': 'T',
                       'O': 'O', 'O:{C2}': 'O', 'O:{C3}': 'O', 'O:{C4}': 'O',
                       'O:{C2}{C3}': 'O', 'O:{C2}{C4}': 'O', 'O:{C3}{C4}': 'O',
                       # 'O:234': 'O', 'O:324': 'O', 'O:342': 'O', 'O:432': 'O', 'O:423': 'O', 'O:243': 'O',
                       # 'O:{C2}{C3}{C4}': 'O', 'O:{C3}{C2}{C4}': 'O', 'O:{C3}{C4}{C2}': 'O', 'O:{C4}{C3}{C2}': 'O',
                       # 'O:{C4}{C2}{C3}': 'O', 'O:{C2}{C4}{C3}': 'O',
                       'O:{C3}{C2}': 'O', 'O:{C4}{C2}': 'O', 'O:{C4}{C3}': 'O',
                       'I': 'I', 'I:{C2}': 'I', 'I:{C3}': 'I', 'I:{C5}': 'I',
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I',
                       # 'I:235': 'I', 'I:325': 'I', 'I:352': 'I', 'I:532': 'I', 'I:253': 'I', 'I:523': 'I',
                       # 'I:{C2}{C3}{C5}': 'I', 'I:{C3}{C2}{C5}': 'I', 'I:{C3}{C5}{C2}': 'I', 'I:{C5}{C3}{C2}': 'I',
                       # 'I:{C2}{C5}{C3}': 'I', 'I:{C5}{C2}{C3}': 'I',
                       'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'C5': 'C5', 'C6': 'C6',
                       'D2': 'D2', 'D3': 'D3', 'D4': 'D4', 'D5': 'D5', 'D6': 'C6',
                       # layer groups
                       # 'p6', 'p4', 'p3', 'p312', 'p4121', 'p622',
                       # space groups  # Todo
                       # 'cryst': 'cryst'
                       }


def unpickle(file_name: str | bytes):  # , protocol=pickle.HIGHEST_PROTOCOL):
    """Unpickle (deserialize) and return a python object located at filename"""
    if '.pkl' not in file_name and '.pickle' not in file_name:
        file_name = '%s.pkl' % file_name
    try:
        with open(file_name, 'rb') as serial_f:
            new_object = load(serial_f)
    except EOFError as ex:
        raise ValueError('The object serialized at location %s couldn\'t be accessed. No data present!' % file_name)

    return new_object


source = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # reveals master symdesign folder
dependency_dir = os.path.join(source, 'dependencies')
sym_op_location = os.path.join(dependency_dir, 'symmetry_operators')
point_group_symmetry_operator_location = os.path.join(sym_op_location, 'point_group_operators.pkl4')
point_group_symmetry_operators = unpickle(point_group_symmetry_operator_location)


def get_all_file_paths(dir, suffix='', extension=None, sort=True):
    """Return all files in a directory with specified extensions and suffixes

    Keyword Args:
        sorted=True (bool): Whether to return the files in alphanumerically sorted order
    """
    if not extension:
        extension = '.pdb'
    if sort:
        return [os.path.join(os.path.abspath(dir), file)
                for file in sorted(glob(os.path.join(os.path.abspath(dir), '*%s*%s' % (suffix, extension))))]
    else:
        return [os.path.join(os.path.abspath(dir), file)
                for file in glob(os.path.join(os.path.abspath(dir), '*%s*%s' % (suffix, extension)))]


def generate_symmetry_mates_pymol(name, expand_matrices):  # name: str, expand_matrices: list[list[float]]):
    prefix = name
    chains = cmd.get_chains(name)
    tx = [0, 0, 0]
    for sym_idx, rotation in enumerate(expand_matrices, 1):
        matrix = []
        for idx, rot_vec in enumerate(rotation):
            matrix.extend(rot_vec + [tx[idx]])  # incase we want to use these for real one day
        matrix.extend([0, 0, 0, 1])
        copy = f'{prefix}_{sym_idx}'
        cmd.create(copy, f'/{name}//%s' % '+'.join(chains))
        cmd.alter(copy, f'segi="{sym_idx}"')
        cmd.transform_object(copy, matrix)
    cmd.disable(name)
    cmd.group(f'{prefix}_expanded', f'{prefix}_*')


def expand(name=None, symmetry=None):
    symmetry_result = possible_symmetries.get(symmetry)
    if symmetry_result:
        expand_matrices = point_group_symmetry_operators[symmetry_result].tolist()
        # Todo
        #  expand_matrices = space_group_symmetry_operators[symmetry_result]
    else:
        print(f'No symmetry "{symmetry}" was found in the possible symmetries. Must be one of the following:\n%s'
              % ', '.join(sorted(set(possible_symmetries.keys()))))
        return

    if name == 'all':
        for object_name in cmd.get_object_list():
            generate_symmetry_mates_pymol(object_name, expand_matrices)
        return
    elif not name:
        name = cmd.get_object_list()[0]

    generate_symmetry_mates_pymol(name, expand_matrices)


def save_group(group='all', one_file=True, out_dir=os.getcwd()):
    """Save all files inside a group to either one file or a list of files.
    Assumes the groups were generated using 'expand'

    Keyword Args:
        group='all' (str): name of the group to save
        one_file=True (bool): Saves each protein in the group to one file, if False, then saves all members individually
        out_dir=os.get_cwd() (str): The directory location to save the files to

    Returns:
        (None)
    """
    if group == 'all':
        # remove appended symmetry mate number '_#' from the group instances and take the set of structures
        expanded_group = set('_'.join(name.split('_')[:-1]) for name in cmd.get_names('group_objects', enabled_only=1))
        groups = ['%s_expanded' % group for group in expanded_group]
    elif cmd.get_type(group) == 'object:group':
        groups = [group]
    else:
        print('Error: please provide a group name to save.')
        return None

    if one_file:
        for group in groups:
            cmd.save(os.path.join(out_dir, '%s.pdb' % group), group)
    else:
        for group in groups:
            stored.models = set()
            cmd.iterate(group, 'stored.models.add(model)')
            for model in stored.models:
                cmd.save(os.path.join(out_dir, '%s.pdb' % model), model)


# TODO
def select_residues(obj=None, chains=None, residues=None):
    print('Received object', obj)
    print('Received chains', chains)
    print('Received residues', residues)
    cmd.select('new_selection', '%s & %s & %s'
               % (('byobj %s' % obj) if obj else '',
                  ' '.join('chain %s' % chain for chain in chains) if chains else '',
                  ' '.join('resi %s' % residue for residue in residues) if residues else ''))


cmd.extend('expand', expand)
cmd.extend('select_residues', select_residues)
cmd.extend('save_group', save_group)


# if __name__ == 'pymol':
if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('Usage:     pymol -r VisualizeUtils.py -- path/to/designs 0-10 [original_name original_order ranked_order]'
             '\n'
             'pymol executable       this script               design range      flags for naming/sorting designs')

    original_name, original_order = False, False
    ranked_order = False
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            if arg == 'original_name':  # design_name:
                original_name = True
            elif arg == 'original_order':
                original_order = True
            elif arg == 'ranked_order':
                ranked_order = True

    if 'escher' in sys.argv[1]:
        print('Starting the data transfer now...')
        os.system('scp -r %s .' % sys.argv[1])
        file_dir = os.path.basename(sys.argv[1])
    else:  # assume the files are local
        file_dir = sys.argv[1]
    files = get_all_file_paths(file_dir, extension='.pdb', sort=not original_order)

    df_glob = sorted(glob(os.path.join(file_dir, 'TrajectoryMetrics.csv')))
    # Todo reinstate when not "lightweight"
    # if ranked_order and df_glob:
    #     df = pd.read_csv(df_glob[0], index_col=0, header=[0])
    #     ordered_files = []
    #     df_indices_pose_ids = list(filter(re.compile('(\w{4})_(\w{4})[/-]'
    #                                                  '.*_?[0-9]_[0-9][/-]'
    #                                                  '.*_?([0-9]+)_([0-9]+)[/-]'
    #                                                  '[tT][xX]_([0-9]+)').match, df.index))
    #
    #     for index in df.index:
    #         index = re.compile('(\w{4})_(\w{4})[/-].*_?[0-9]_[0-9][/-].*_?([0-9]+)_([0-9]+)[/-][tT][xX]_([0-9]+)')\
    #             .match(index)
    #         if not index:
    #             continue
    #     # for index in df_indices_pose_ids:
    #         for file in files:
    #             if index.group(0) in file:
    #                 ordered_files.append(file)
    #                 break
    #     files = ordered_files

    if not files:
        exit('No .pdb files found in directory %s. Are you sure this is correct?' % sys.argv[1])

    if len(sys.argv) > 2:
        low, high = map(float, sys.argv[2].split('-'))
        low_range, high_range = int((low / 100) * len(files)), int((high / 100) * len(files))
        if low_range < 0 or high_range > len(files):
            raise ValueError('The input range is outside of the acceptable bounds [0-100]')
        print('Selecting Designs within range: %d-%d' % (low_range if low_range else 1, high_range))
    else:
        low_range, high_range = None, None

    for idx, file in enumerate(files[low_range:high_range], low_range + 1):
        if original_name:
            cmd.load(file)
        else:
            cmd.load(file, object=idx)

    print('\nTo expand all designs to the proper symmetry, issue:\n\texpand name=all, symmetry=T')
          # '\nReplace \'T\' with whatever symmetry your design is in\n')
