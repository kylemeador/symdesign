import os

from pymol import cmd, stored

import symdesign.utils.symmetry as symutils


def generate_symmetry_mates_pymol(name, rotation_matrices):  # name: str, rotation_matrices: list[list[float]]):
    prefix = name
    chains = cmd.get_chains(name)
    tx = [0, 0, 0]
    for sym_idx, rotation in enumerate(rotation_matrices, 1):
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


def expand(name: str = None, symmetry: str = None):
    symmetry_result = symutils.possible_symmetries.get(symmetry)
    if symmetry_result:
        rotation_matrices = symutils.point_group_symmetry_operatorsT[symmetry_result].tolist()
        # Todo
        #  rotation_matrices = space_group_symmetry_operators[symmetry_result]
    else:
        print(f'No symmetry "{symmetry}" was found in the possible symmetries. Must be one of the following:\n%s'
              % ', '.join(sorted(set(symutils.possible_symmetries.keys()))))
        return

    if name == 'all':
        for object_name in cmd.get_object_list():
            generate_symmetry_mates_pymol(object_name, rotation_matrices)
        return
    elif not name:
        name = cmd.get_object_list()[0]

    generate_symmetry_mates_pymol(name, rotation_matrices)


def save_group(group: str = 'all', one_file: bool = True, out_dir: str = os.getcwd()) -> None:
    """Save all files inside a group to either one file or a list of files.
    Assumes the groups were generated using 'expand'

    Args:
        group: name of the group to save
        one_file: Saves each protein in the group to one file, if False, then saves all members individually
        out_dir: The directory location to save the files to

    Returns:
        None
    """
    if group == 'all':
        # remove appended symmetry mate number '_#' from the group instances and take the set of structures
        expanded_group = set('_'.join(name.split('_')[:-1]) for name in cmd.get_names('group_objects', enabled_only=1))
        groups = [f'{group}_expanded' for group in expanded_group]
    elif cmd.get_type(group) == 'object:group':
        groups = [group]
    else:
        print('Error: please provide a group name to save.')
        return None

    if one_file:
        for group in groups:
            cmd.save(os.path.join(out_dir, f'{group}.pdb'), group)
    else:
        for group in groups:
            stored.models = set()
            cmd.iterate(group, 'stored.models.add(model)')
            for model in stored.models:
                cmd.save(os.path.join(out_dir, f'{model}.pdb'), model)


cmd.extend('expand', expand)
cmd.extend('save_group', save_group)
