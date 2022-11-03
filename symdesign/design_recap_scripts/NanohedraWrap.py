"""For wrapping the distribution of Nanohedra jobs to Cassini Cluster"""
import os
import subprocess
from itertools import chain
from typing import Union

from symdesign import utils
from symdesign.utils import path as putils


@utils.handle_errors(errors=(utils.DesignError, ))
def nanohedra_design_recap(dock_dir, suffix=None):
    """From a directory set up for docking, a '_dock.pkl' file specifies the arguments passed to nanohedra commands"""

    entry_d = {'I': {('C2', 'C3'): 8, ('C2', 'C5'): 14, ('C3', 'C5'): 56}, 'T': {('C2', 'C3'): 4, ('C3', 'C3'): 52}}
    symmetries = ['C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'T', 'O', 'I']
    sym_hierarchy = {sym: i for i, sym in enumerate(symmetries, 1)}

    des_dir_d = utils.unpickle(os.path.join(dock_dir, '%s_vflip_dock.pkl' % os.path.basename(dock_dir)))  # 9/29/20 removed .pkl added _vflip
    # {0_Sym: PDB1, 1_Sym2: PDB2, 'final_symmetry': I}

    syms = list(set(des_dir_d.keys()) - {'final_symmetry'})  # ex: [0_C2, 1_C3]
    symmetry_rank = 0
    sym_d = {'higher': None, 'higher_path': None, 'lower': None, 'lower_path': None}
    for i, sym in enumerate(syms):
        sym_l = sym.split('_')
        sym_l[0] = str(int(sym_l[0]) + 1)
        _sym = sym_l[1]
        new_sym = '_'.join(sym_l)
        # Used with the flipped_180y pdb's
        if sym.split('_')[0] == '1':  # The higher symmetry
            if not os.path.exists(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())):
                raise utils.DesignError('Missing symmetry %s PDB file %s!' % (new_sym, des_dir_d[sym].lower()))
            else:
                path2 = os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())
        else:
            if not os.path.exists(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())):
                raise utils.DesignError('Missing symmetry %s PDB file %s!' % (new_sym, des_dir_d[sym].lower()))
            else:
                path1 = os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())

        # for pdb in des_dir_d[sym]:
            # if not os.path.exists(os.path.join(dock_dir, sym, '%s.pdb' % pdb.lower())):
            #    raise SDUtils.utils.DesignError(['Missing symmetry %s PDB file %s!' % (sym, pdb.lower())])
        # print(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower()))

        # check if .pdb exists
        if not os.path.exists(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())):
            raise utils.DesignError('Missing symmetry %s PDB file %s!' % (new_sym, des_dir_d[sym].lower()))

        # This protocol should be obsolete with DesignRecapFileController.py fixed symmetry order TODO, remove when old pickles are unnecessary
        new_symmetry_rank = sym_hierarchy[_sym]
        if new_symmetry_rank >= symmetry_rank:  # the case where sym2 is greater than sym1 or equal to sym1
            symmetry_rank = new_symmetry_rank
            sym_d['lower'] = sym_d['higher']
            sym_d['lower_path'] = sym_d['higher_path']
            sym_d['higher'] = _sym
            sym_d['higher_path'] = new_sym
        else:  # The case where 1 is greater than 2
            sym_d['lower'] = _sym
            sym_d['lower_path'] = new_sym
    if len(des_dir_d) == 1:
        sym_d['lower'] = sym_d['higher']

    # sym_tuple = (lower_sym, higher_sym)
    sym_tuple = (sym_d['lower'], sym_d['higher'])
    entry_num = entry_d[des_dir_d['final_symmetry']][sym_tuple]
    # out_dir = os.path.join(dock_dir, 'NanohedraEntry%sDockedPoses' % entry_num)
    # out_dir = '/gscratch/kmeador/Nanohedra_design_recap_test/Nanohedra_output'
    # out_dir = os.path.join(os.path.dirname(dock_dir).split(os.sep)[-2])

    # used with second docking set up (degeneracies)
    return nanohedra_command(str(entry_num), path1, path2, out_dir=dock_dir, suffix=suffix, default=False)

    # Used in first docking set up
    # return nanohedra_command(str(entry_num), os.path.join(dock_dir, '%s' % sym_d['lower_path']),
    #                          os.path.join(dock_dir, '%s' % sym_d['higher_path']), out_dir=dock_dir, suffix=suffix,
    #                          default=False)


def nanohedra_command_concise(path1: Union[str, bytes], path2: Union[str, bytes], entry: int = 0,
                              out_dir: Union[str, bytes] = os.getcwd(), initial: bool = False, **kwargs) -> str:
    """Write out Nanohedra commands to shell scripts for processing by computational clusters

    Return:
        The name of the file containing the Nanohedra command
    """
    nano_out_dir = os.path.join(out_dir, 'NanohedraEntry%sDockedPoses%s' % entry)
    script_out_dir = os.path.join(nano_out_dir, putils.scripts)
    os.makedirs(nano_out_dir, exist_ok=True)
    os.makedirs(script_out_dir, exist_ok=True)

    cmd = ['python', putils.nanohedra_dock_file, '-dock', '-entry', entry, putils.nano_entity_flag1, path1,
           putils.nano_entity_flag2, path2, '-outdir', nano_out_dir]
    #        ['rot_step1', rotation1, '-rot_step2', rotation2, '-min_matched']
    cmd.extend(chain.from_iterable([['-%s' % key, str(value)] for key, value in kwargs]))
    if initial:
        cmd.extend(['-initial'])

    return utils.write_shell_script(subprocess.list2cmdline(cmd), out_path=script_out_dir,
                                    name=f'nanohedra_{os.path.basename(os.path.splitext(path1)[0])}_'
                                         f'{os.path.basename(os.path.splitext(path2)[0])}')


def nanohedra_command(entry, path1, path2, out_dir=None, suffix=None, initial=False):
    """Write out Nanohedra commands to shell scripts for processing by computational clusters

    Return:
        (str): The name of the file containing the Nanohedra command
    """

    if not out_dir:
        nano_out_dir = os.path.join(os.getcwd(), 'NanohedraEntry%sDockedPoses%s' % (entry, str(suffix or '')))
    else:
        nano_out_dir = os.path.join(out_dir, 'NanohedraEntry%sDockedPoses%s' % (entry, str(suffix or '')))
    if not os.path.exists(nano_out_dir):
        try:
            os.makedirs(nano_out_dir)
        except FileExistsError:
            pass

    script_out_dir = os.path.join(nano_out_dir, 'job_scripts')
    program = putils.nanohedra_main
    if os.path.splitext(path1)[1] == '':  # check if path1 is directory
        if os.path.splitext(path2)[1] == '':  # check if path2 is directory
            # both are directories
            name = 'nanohedra_%s_%s' % (os.path.basename(path1), os.path.basename(path2))
        else:  # path2 isn't directory
            name = 'nanohedra_%s_%s' % (os.path.basename(path1), os.path.splitext(os.path.basename(path2))[0])
    else:  # path1 isn't directory
        if os.path.splitext(path2)[1] == '':  # check if path2 is directory
            name = 'nanohedra_%s_%s' % (os.path.splitext(os.path.basename(path1))[0], os.path.basename(path2))
        else:  # neither are directories
            building_block_string = '%s_%s' % (os.path.splitext(os.path.basename(path1))[0],
                                               os.path.splitext(os.path.basename(path2))[0])
            name = 'nanohedra_%s' % building_block_string
            script_out_dir = os.path.join(nano_out_dir, building_block_string)
            program = putils.nanohedra_dock_file

    if not os.path.exists(script_out_dir):
        os.makedirs(script_out_dir)

    # if default:
    #     step_1, step_2 = '3', '3'
    # else:
    #     step_1, step_2 = '2', '2'

    _cmd = ['python', program, '-dock', '-entry', str(entry), '-pdb_dir1_path', path1, '-pdb_dir2_path', path2,
            # '-rot_step1', step_1, '-rot_step2', step_2,
            '-outdir', nano_out_dir]
    # if flags:
    #     _cmd.extend(flags)

    if initial:
        _cmd.extend(['-initial'])

    return utils.write_shell_script(subprocess.list2cmdline(_cmd), name=name, out_path=script_out_dir)
