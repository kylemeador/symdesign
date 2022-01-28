import os
from pymol import cmd, stored

# from Pose import Pose
from classes.SymEntry import possible_symmetries
from utils.SymmetryUtils import get_ptgrp_sym_op


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
        expand_matrices = get_ptgrp_sym_op(symmetry_)
        # if self.dimension == 0 else self.get_sg_sym_op(self.symmetry)  # ensure symmetry is Hermann–Mauguin notation
    else:
        print('No symmetry \'%s\' was found in the possible symmetries. Must be one of the following:\n%s'
              % (symmetry, ', '.join(possible_symmetries)))
        return None

    if name == 'all':
        for object_name in cmd.get_object_list():
            generate_symmetry_mates_pymol(object_name, expand_matrices)
        return None
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


cmd.extend('expand', expand)
cmd.extend('save_group', save_group)
