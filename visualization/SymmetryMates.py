import os
from pymol import cmd

from Pose import Pose
from SymDesignUtils import possible_symmetries


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
        expand_matrices = Pose.get_ptgrp_sym_op(symmetry_)
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


cmd.extend('expand', expand)
