"""For wrapping the distribution of Nanohedra jobs to Cassini Cluster
"""

import os
import subprocess
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import CmdUtils as CUtils


# TODO multiprocessing compliant (picklable) error decorator
@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def nanohedra_s(dock_dir):
    return nanohedra(dock_dir)


def nanohedra_mp(dock_dir):
    try:
        file = nanohedra(dock_dir)
        return file, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, (dock_dir, e)


def nanohedra(dock_dir):
    # des_dir_d = {design: {Sym: PDB1, Sym2: PDB2, Final_Sym:I}}
    # {1_Sym: PDB1, 1_Sym2: PDB2, 'final_symmetry': I}
    entry_d = {'I': {('C2', 'C3'): 8, ('C2', 'C5'): 14, ('C3', 'C5'): 56}, 'T': {('C2', 'C3'): 4, ('C3', 'C3'): 52}}
    symmetries = ['C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'T', 'O', 'I']
    sym_hierarchy = {sym: i for i, sym in enumerate(symmetries, 1)}

    des_dir_d = SDUtils.unpickle(os.path.join(dock_dir, '%s_dock.pkl.pkl' % os.path.basename(dock_dir)))  # TODO remove .pkl
    syms = list(set(des_dir_d.keys()) - {'final_symmetry'})  # ex: [0_C2, 1_C3]
    symmetry_rank = 0
    sym_d = {'higher': None, 'higher_path': None, 'lower': None, 'lower_path': None}
    for i, sym in enumerate(syms):
        sym_l = sym.split('_')
        sym_l[0] = str(int(sym_l[0]) + 1)
        new_sym = '_'.join(sym_l)
        # for pdb in des_dir_d[sym]:
            # if not os.path.exists(os.path.join(dock_dir, sym, '%s.pdb' % pdb.lower())):
            #    raise SDUtils.DesignError(['Missing symmetry %s PDB file %s!' % (sym, pdb.lower())])
        # print(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower()))

        # check if .pdb exists
        if not os.path.exists(os.path.join(dock_dir, new_sym, '%s.pdb' % des_dir_d[sym].lower())):
            raise SDUtils.DesignError(['Missing symmetry %s PDB file %s!' % (new_sym, des_dir_d[sym].lower())])

        _sym = sym_l[1]
        new_symmetry_rank = sym_hierarchy[_sym]
        if new_symmetry_rank >= symmetry_rank:  # the case where sym2 is greater than sym1 or equal to sym1
            symmetry_rank = new_symmetry_rank
            # lower_sym = higher_sym
            sym_d['lower'] = sym_d['higher']
            sym_d['lower_path'] = sym_d['higher_path']
            # higher_sym = _sym
            sym_d['higher'] = _sym
            sym_d['higher_path'] = new_sym
        else:  # The case where 1 is greater than 2
            # lower_sym = _sym
            sym_d['lower'] = _sym
            sym_d['lower_path'] = new_sym
    if len(des_dir_d) == 1:
        sym_d['lower'] = sym_d['higher']
        # lower_sym = higher_sym

    # {Sym: PDB1, Sym2: PDB2, 'final_symmetry': I}
    # des_dir_d = {}
    # dock_file = os.path.join(dock_dir, '%s_components.dock' % os.path.basename(dock_dir))  # TODO '.dock'
    # with open(dock_file, 'r') as f:
    #     parameters = f.readlines()
    #     # parameters = map(str.split(), parameters)
    #     for line in parameters:
    #         info = line.split()
    #         # if line.find('final_symmetry', 6) != -1:
    #         #     final_sym = line.split('final_symmetry ')[1]
    #         #     info[1] = info[1][:2]
    #         # 4G41 C2
    #         # 2CHC C3final_symmetry I
    #         if info[0] == 'final_symmetry':
    #             des_dir_d[info[0]] = [info[1]]
    #             continue
    #         if info[1] not in des_dir_d:
    #             des_dir_d[info[1]] = [info[0]]
    #         else:
    #             des_dir_d[info[1]].append(info[0])
    #     # 4G41 C2
    #     # 2CHC C3
    #     # final_symmetry I

    # for sym in des_dir_d:
    #     sym_l = sym.split('_')
    #     if sym_l[0] == '1':
    #         lower_sym = sym_l[1]
    #     elif sym_l[0] == '2':
    #         higher_sym = sym_l[1]

    # sym_tuple = (lower_sym, higher_sym)
    sym_tuple = (sym_d['lower'], sym_d['higher'])
    entry_num = entry_d[des_dir_d['final_symmetry']][sym_tuple]
    out_dir = os.path.join(dock_dir, 'NanohedraEntry%sDockedPoses' % entry_num)
    # out_dir = '/gscratch/kmeador/Nanohedra_design_recap_test/Nanohedra_output'
    # out_dir = os.path.join(os.path.dirname(dock_dir).split(os.sep)[-2])

    return nanohedra_command(str(entry_num), os.path.join(dock_dir, '%s' % sym_d['lower_path']),
                             os.path.join(dock_dir, '%s' % sym_d['higher_path']), out_dir=out_dir, default=False)

    # _cmd = ['python', PUtils.nanohedra_main, '-dock', '-entry', str(entry_num), '-pdb_dir1_path',
    #         os.path.join(dock_dir, '%s' % sym_d['lower_path']),
    #         '-pdb_dir2_path', os.path.join(dock_dir, '%s' % sym_d['higher_path']),
    #         '-rot_step1', '2', '-rot_step2', '2', '-outdir', out_dir]
    #
    # return SDUtils.write_shell_script(subprocess.list2cmdline(_cmd), name='nanohedra', outpath=dock_dir)


# TODO multiprocessing compliant (picklable) error decorator
@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def nanohedra_command_s(entry, path1, path2, out_dir):
    return nanohedra_command(entry, path1, path2, out_dir)


def nanohedra_command_mp(entry, path1, path2, out_dir):
    try:
        file = nanohedra_command(entry, path1, path2, out_dir)
        return file, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, ((path1, path2), e)


def nanohedra_command(entry, path1, path2, out_dir=None, default=True):
    """Write out Nanohedra commands to shell scripts for processing by computational clusters"""

    if not out_dir:
        if not os.path.exists(os.path.join(os.getcwd(), 'NanohedraEntry%sDockedPoses' % entry)):
            os.makedirs(os.path.join(os.getcwd(), 'NanohedraEntry%sDockedPoses' % entry))
        out_dir = os.path.join(os.getcwd(), 'NanohedraEntry%sDockedPoses' % entry, '%s_%s' %
                               (os.path.splitext(os.path.basename(path1))[0], os.path.splitext(os.path.basename(path2))[0]))
    else:
        if not os.path.exists(os.path.join(out_dir, 'NanohedraEntry%sDockedPoses' % entry)):
            os.makedirs(os.path.join(out_dir, 'NanohedraEntry%sDockedPoses' % entry))
        out_dir = os.path.join(out_dir, 'NanohedraEntry%sDockedPoses' % entry, '%s_%s' %
                               (os.path.splitext(os.path.basename(path1))[0], os.path.splitext(os.path.basename(path2))[0]))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if default:
        step_1, step_2 = '3', '3'
    else:
        step_1, step_2 = '2', '2'
    _cmd = ['python', PUtils.nanohedra_s_main, '-dock', '-entry', str(entry), '-pdb_dir1_path',
            path1, '-pdb_dir2_path', path2, '-rot_step1', step_1, '-rot_step2', step_2, '-outdir', out_dir]

    return SDUtils.write_shell_script(subprocess.list2cmdline(_cmd), name='nanohedra', outpath=out_dir)
