import os

import cPickle as pickle

from PathUtils import sym_op_location


def get_sg_sym_op(sym_type, sg_op_filepath):
    expand_uc_matrices = []

    sg_op_file = open(sg_op_filepath, "r")
    sg_op_lines = sg_op_file.readlines()
    sg_op_file.close()

    is_sg = False
    rot_mat = []
    tx_mat = []
    line_count = 0
    for line in sg_op_lines:
        if "'" in line:
            is_sg = False
        if "'%s'" % sym_type in line:
            is_sg = True
        if is_sg and "'" not in line and ":" not in line and not line[0].isdigit():
            line_split = line.split()
            line_float = [float(s) for s in line_split]
            rot_mat.append(line_float[0:3])
            tx_mat.append(line_float[-1])
            line_count += 1
            if line_count % 3 == 0:
                expand_uc_matrices.append((rot_mat, tx_mat))
                tx_mat = []
                rot_mat = []

    return expand_uc_matrices


def generate_sym_op_txtfiles(sg_op_filepath):
    sg_lst = ['P23', 'P4222', 'P321', 'P6322', 'P312', 'P622', 'F23', 'F222', 'P6222', 'I422', 'I213', 'R32', 'P4212',
              'I432', 'P4132', 'I4132', 'P3', 'P6', 'I4122', 'P4', 'C222', 'P222', 'P432', 'F4132', 'P422', 'P213',
              'F432', 'P4232']
    for sg in sg_lst:
        sym_op_outfile_path = "/Users/jlaniado/Desktop/SG_SYM_OP/" + sg + ".txt"
        sym_op_outfile = open(sym_op_outfile_path, "w")
        sym_op = get_sg_sym_op(sg, sg_op_filepath)
        for op in sym_op:
            sym_op_outfile.write(str(op) + "\n")
        sym_op_outfile.close()


def generate_sym_op_pickles(sg_op_filepath):
    sg_lst = ['P23', 'P4222', 'P321', 'P6322', 'P312', 'P622', 'F23', 'F222', 'P6222', 'I422', 'I213', 'R32', 'P4212',
              'I432', 'P4132', 'I4132', 'P3', 'P6', 'I4122', 'P4', 'C222', 'P222', 'P432', 'F4132', 'P422', 'P213',
              'F432', 'P4232']
    for sg in sg_lst:
        sym_op_outfile_path = "/Users/jlaniado/Desktop/SG_SYM_OP/" + sg + ".pickle"
        sym_op_outfile = open(sym_op_outfile_path, "w")
        sym_op = get_sg_sym_op(sg, sg_op_filepath)
        pickle.dump(sym_op, sym_op_outfile)
        sym_op_outfile.close()


generate_sym_op_pickles(os.path.join(sym_op_location, "spacegroups_op.txt"))
