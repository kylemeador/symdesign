import os
import sys
from glob import glob

from pymol import finish_launching, cmd, stored
finish_launching(['pymol', '-q'])
# finish_launching()
# from Pose import Pose
# from SymDesignUtils import possible_symmetries

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
possible_symmetries = {'I32': 'I', 'I52': 'I', 'I53': 'I', 'T32': 'T', 'T33': 'T',
                       'I23': 'I', 'I25': 'I', 'I35': 'I', 'T23': 'T',
                       'T:{C2}{C3}': 'T', 'T:{C3}{C2}': 'T', 'T:{C3}{C3}': 'T',
                       'I:{C2}{C3}': 'I', 'I:{C2}{C5}': 'I', 'I:{C3}{C5}': 'I',
                       'I:{C3}{C2}': 'I', 'I:{C5}{C2}': 'I', 'I:{C5}{C3}': 'I',
                       'T': 'T', 'O': 'O', 'I': 'I',
                       }


def get_all_file_paths(dir, suffix='', extension=None):
    if not extension:
        extension = '.pdb'
    return [os.path.join(os.path.abspath(dir), file)
            for file in glob(os.path.join(dir, '*%s*%s' % (suffix, extension)))]


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
        # if self.dimension == 0 else self.get_sg_sym_op(self.symmetry)  # ensure symmetry is Hermann–Mauguin notation
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


cmd.extend('expand', expand)
cmd.extend('save_group', save_group)


if __name__ == '__main__':
# if __name__ == 'pymol':
    if len(sys.argv) != 2:
        exit('Usage: python VisualizeDesigns.py path/to/designs')
    # os.system('scp -r escher:%s .' % sys.argv[0])
    if sys.argv[1].startswith('escher:'):
        os.system('scp -r %s .' % sys.argv[1])
        files = get_all_file_paths(os.getcwd(), extension='.pdb')
    else:  # assume the files are local
        files = get_all_file_paths(sys.argv[1], extension='.pdb')

    if not files:
        exit('No .pdb files found in directory %s. Are you sure this is correct?' % sys.argv[1])

    for file in files:
        cmd.load(file)

    print('To expand all designs to the proper symmetry, issue:\n\texpand name=\'all\', symmetry=\'T\''
          '\nReplace \'T\' with whatever symmetry your design is in')

# print(__name__)
