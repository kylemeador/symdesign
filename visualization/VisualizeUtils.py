import os
import re
import sys
from glob import glob

import pandas as pd
from pymol import finish_launching, cmd, stored
finish_launching(['pymol', '-q'])
# finish_launching()
# from Pose import Pose
# from SymDesignUtils import possible_symmetries

# TODO integration of the capabilities of SymDesign with pymol plugins to manipulate and design
#  https://raw.githubusercontent.com/Pymol-Scripts/Pymol-script-repo/master/plugins/SuperSymPlugin.py


expand_matrices = \
    {'T': [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.0, 1.0]],
           [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
           [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.0, -1.0]],
           [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
           [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
           [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
           [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
           [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
           [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -0.0, 0.0]],
           [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
           [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
           [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -0.0, 0.0]]]}
# REMARK 350 formatting
# REMARK 350 BIOMOLECULE: 1
# REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B
# REMARK 350   BIOMT1   1  1.0 0.0 0.0 0
# REMARK 350   BIOMT2   1  0.0 1.0 0.0 0
# REMARK 350   BIOMT3   1  0.0 -0.0 1.0 0
# REMARK 350   BIOMT1   2  -1.0 0.0 0.0 0
# REMARK 350   BIOMT2   2  0.0 -1.0 0.0 0
# REMARK 350   BIOMT3   2  0.0 0.0 1.0 0
# REMARK 350   BIOMT1   3  1.0 0.0 0.0 0
# REMARK 350   BIOMT2   3  0.0 -1.0 0.0 0
# REMARK 350   BIOMT3   3  0.0 -0.0 -1.0 0
# REMARK 350   BIOMT1   4  -1.0 0.0 0.0 0
# REMARK 350   BIOMT2   4  0.0 1.0 0.0 0
# REMARK 350   BIOMT3   4  0.0 0.0 -1.0 0
# REMARK 350   BIOMT1   5  0.0 0.0 1.0 0
# REMARK 350   BIOMT2   5  1.0 0.0 0.0 0
# REMARK 350   BIOMT3   5  0.0 1.0 0.0 0
# REMARK 350   BIOMT1   6  0.0 0.0 -1.0 0
# REMARK 350   BIOMT2   6  -1.0 0.0 0.0 0
# REMARK 350   BIOMT3   6  0.0 1.0 0.0 0
# REMARK 350   BIOMT1   7  0.0 0.0 1.0 0
# REMARK 350   BIOMT2   7  -1.0 0.0 0.0 0
# REMARK 350   BIOMT3   7  0.0 -1.0 0.0 0
# REMARK 350   BIOMT1   8  0.0 0.0 -1.0 0
# REMARK 350   BIOMT2   8  1.0 0.0 0.0 0
# REMARK 350   BIOMT3   8  0.0 -1.0 0.0 0
# REMARK 350   BIOMT1   9  0.0 1.0 0.0 0
# REMARK 350   BIOMT2   9  0.0 0.0 1.0 0
# REMARK 350   BIOMT3   9  1.0 -0.0 0.0 0
# REMARK 350   BIOMT1  10  0.0 -1.0 0.0 0
# REMARK 350   BIOMT2  10  0.0 0.0 -1.0 0
# REMARK 350   BIOMT3  10  1.0 0.0 0.0 0
# REMARK 350   BIOMT1  11  0.0 1.0 0.0 0
# REMARK 350   BIOMT2  11  0.0 0.0 -1.0 0
# REMARK 350   BIOMT3  11  -1.0 0.0 0.0 0
# REMARK 350   BIOMT1  12  0.0 -1.0 0.0 0
# REMARK 350   BIOMT2  12  0.0 0.0 1.0 0
# REMARK 350   BIOMT3  12  -1.0 -0.0 0.0 0
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T',
                       'I23': 'I', 'I25': 'I', 'I35': 'I', 'T23': 'T',
                       'T:{C2}{C3}': 'T', 'T:{C3}{C2}': 'T', 'T:{C3}{C3}': 'T',
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I',
                       'T': 'T', 'O': 'O', 'I': 'I',
                       }


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


def generate_symmetry_mates_pymol(name, expand_matrices):
    prefix = name
    chains = cmd.get_chains(name)
    tx = [0, 0, 0]
    for sym_idx, rot in enumerate(expand_matrices, 1):
        matrix = []
        for idx, rot_vec in enumerate(rot):
            matrix.extend(rot_vec + [tx[idx]])
        matrix.extend([0, 0, 0, 1])
        copy = '%s_%d' % (prefix, sym_idx)
        cmd.create(copy, '/%s//%s' % (name, '+'.join(chains)))
        cmd.alter(copy, 'segi="%d"' % sym_idx)
        cmd.transform_object(copy, matrix)
    cmd.disable(name)
    cmd.group('%s_expanded' % prefix, '%s_*' % prefix)


def expand(name=None, symmetry=None):
    symmetry_ = possible_symmetries.get(symmetry)
    if symmetry_:  # ['T', 'O', 'I']:
        # expand_matrices = Pose.get_ptgrp_sym_op(symmetry_)
        _expand_matrices = expand_matrices[symmetry_]
        # _expand_matrices = point_group_symmetry_operators[self.symmetry]
        # _expand_matrices = space_group_symmetry_operators[self.symmetry]
    else:
        print('No symmetry \'%s\' was found in the possible symmetries. Must be one of the following:\n%s'
              % (symmetry, ', '.join(possible_symmetries)))
        return

    if name == 'all':
        for object_name in cmd.get_object_list():
            generate_symmetry_mates_pymol(object_name, _expand_matrices)
        return
    elif not name:
        name = cmd.get_object_list()[0]

    generate_symmetry_mates_pymol(name, _expand_matrices)


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
    if ranked_order and df_glob:
        df = pd.read_csv(df_glob[0], index_col=0, header=[0])
        ordered_files = []
        df_indices_pose_ids = list(filter(re.compile('(\w{4})_(\w{4})[/-]'
                                                     '.*_?[0-9]_[0-9][/-]'
                                                     '.*_?([0-9]+)_([0-9]+)[/-]'
                                                     '[tT][xX]_([0-9]+)').match, df.index))

        for index in df.index:
            index = re.compile('(\w{4})_(\w{4})[/-].*_?[0-9]_[0-9][/-].*_?([0-9]+)_([0-9]+)[/-][tT][xX]_([0-9]+)')\
                .match(index)
            if not index:
                continue
        # for index in df_indices_pose_ids:
            for file in files:
                if index.group(0) in file:
                    ordered_files.append(file)
                    break
        files = ordered_files

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
