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
    # des_dir_d = {Sym: PDB1, Sym2: PDB2}
    des_dir_d = {}
    with open(dock_dir, 'r') as f:
        parameters = f.readlines()
        # parameters = map(str.split(), parameters)
        for line in parameters:
            info = line.split()
            if line.find('final_symmetry', 6) != -1:
                final_sym = line.split('final_symmetry ')[1]
                info[1] = info[1][:2]
            des_dir_d[info[1]] = info[0]
        # 4G41 C2
        # 2CHC C3final_symmetry I

    entry_d = {'I': {('C2', 'C3'): 8, ('C2', 'C5'): 14, ('C3', 'C5'): 56}, 'T': {('C2', 'C3'): 4, ('C3', 'C3'): 52}}
    # des_dir_d = {}
    # for root, dirs, files in os.walk(des_dir):
    #     for dir in dirs:
    #         des_dir_d[]
    symmetries = ['C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'T', 'O', 'I']
    sym_hierarchy = {sym: i for i, sym in enumerate(symmetries, 1)}
    symmetry_rank, higher_sym = 0, None
    for sym in des_dir_d:
        new_symmetry_rank = sym_hierarchy[sym]
        if new_symmetry_rank >= symmetry_rank:  # the case where sym2 is greater than sym1 or equal to sym1
            symmetry_rank = new_symmetry_rank
            lower_sym = higher_sym
            higher_sym = sym
        else:  # The case where 1 is greater than 2
            lower_sym = sym

    sym_tuple = (lower_sym, higher_sym)
    entry_num = entry_d[final_sym][sym_tuple]
    out_dir = '/gscratch/kmeador/Nanohedra_design_recap_test/Nanohedra_output'
    # out_dir = os.path.join(os.path.dirname(dock_dir).split(os.sep)[-2])
    for sym in sym_tuple:
        if not os.path.exists(os.path.join(dock_dir, sym, '%s.pdb' % des_dir_d[sym])):
            raise SDUtils.DesignError(['Missing symmetry %s PDB file!' % sym])

    _cmd = ['python', PUtils.nanohedra_main, '-dock', '-entry', str(entry_num), '-pdb_dir1_path',
            os.path.join(dock_dir, lower_sym, '%s.pdb' % des_dir_d[lower_sym]), '-pdb_dir2_path',
            os.path.join(dock_dir, higher_sym, '%s.pdb' % des_dir_d[higher_sym]),
            '-rot_step1', '2', '-rot_step2', '2', '-outdir', out_dir]
    command_file = SDUtils.write_shell_script(subprocess.list2cmdline(_cmd), name='nanohedra',
                                              outpath=os.path.dirname(dock_dir))

    return command_file
