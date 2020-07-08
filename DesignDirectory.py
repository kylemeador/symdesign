import os
import sys
import math
import subprocess
import logging
import pickle
import copy
import numpy as np
import multiprocessing as mp
import sklearn.neighbors
from itertools import repeat
import PDB
from Bio.SeqUtils import IUPACData
from Bio.SubsMat import MatrixInfo
from Bio import pairwise2

import PathUtils as PUtils
import CmdUtils as CUtils
import AnalyzeOutput as AOut
logging.getLogger().setLevel(logging.INFO)


class DesignDirectory:

    def __init__(self, directory, auto_structure=True):
        self.symmetry = None         # design_symmetry (P432)
        self.sequences = None        # design_symmetry/sequences (P432/sequences)
        self.building_blocks = None  # design_symmetry/building_blocks (P432/4ftd_5tch)
        self.all_scores = None       # design_symmetry/building_blocks/scores (P432/4ftd_5tch/scores)
        self.path = directory        # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2
        self.scores = None           # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/scores (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/scores)
        self.design_pdbs = None      # design_symmetry/building_blocks/DEGEN_A_B/ROT_A_B/tx_C/rosetta_pdbs (P432/4ftd_5tch/DEGEN1_2/ROT_1/tx_2/rosetta_pdbs)
        self.log = None
        self.pose = None
        self.components = []  # for each oligomer, [name, name] name:{chain:self.pose.get_chain_index(, position: 0}
        self.pssm_files = {}  # {name: file, }
        self.pose_seq = {}  # {name: seq, }
        self.errors = {}  # {name: seq_errors, }
        self.pdb_seq_file = {}  # {name: seq_file, }
        self.pssm_process = {}  # {name: subprocess object, }
        if auto_structure:
            self.make_directory_structure()
            self.pose = self.read_pdb(os.path.join(self.path, PUtils.asu)

    def __str__(self):
        if self.symmetry:
            return self.path.replace(self.symmetry + '/', '').replace('/', '-')
        else:
            # When is this relevant?
            return self.path.replace('/', '-')[1:]

    def make_directory_structure(self):
        # Prepare Output Directory/Files. path always has format:
        self.symmetry = self.path[:self.path.find(self.path.split('/')[-4]) - 1]
        self.sequences = os.path.join(self.symmetry, PUtils.sequence_info)
        self.building_blocks = self.path[:self.path.find(self.path.split('/')[-3]) - 1]
        self.all_scores = os.path.join(self.building_blocks, PUtils.scores_outdir)
        self.scores = os.path.join(self.path, PUtils.scores_outdir)
        self.design_pdbs = os.path.join(self.path, PUtils.pdbs_outdir)

        if not os.path.exists(self.sequences):
            os.makedirs(self.sequences)
        if not os.path.exists(self.all_scores):
            os.makedirs(self.all_scores)
        if not os.path.exists(self.scores):
            os.makedirs(self.scores)
        if not os.path.exists(self.design_pdbs):
            os.makedirs(self.design_pdbs)

    def start_log(self, name=None, level=2):
        _name = __name__ + '.' + str(self)
        if name:
            _name = name
        self.log = start_log(_name, handler=2, level=level, location=os.path.join(self.path, os.path.basename(self.path)))

    def read_pdb(file):
        """Wrapper on the PDB __init__ and readfile functions
        Args:
            file (str): disk location of pdb file
        Returns:
            pdb (PDB): Initialized PDB object
        """
        pdb = PDB.PDB()
        pdb.readfile(file)

        return pdb


def get_design_directories(directory):
    all_design_directories = []
    for design_root, design_dirs, design_files in os.walk(directory):
        if os.path.basename(design_root).startswith('tx_'):
            all_design_directories.append(design_root)
        for directory in design_dirs:
            if directory.startswith('tx_'):
                all_design_directories.append(os.path.join(design_root, directory))
    all_design_directories = set(all_design_directories)
    all_des_dirs = []
    for directory in all_design_directories:
        all_des_dirs.append(DesignDirectory(directory))

    return all_des_dirs


class DesignError(Exception):
    # pass

    def __init__(self, message):  # expression,
        # self.expression = expression
        self.message = message


def set_up_pseudo_design_dir(wildtype, directory, score):
    pseudo_dir = DesignDirectory(wildtype, auto_structure=False)
    pseudo_dir.path = os.path.dirname(wildtype)
    pseudo_dir.building_blocks = os.path.dirname(wildtype)
    pseudo_dir.design_pdbs = directory
    pseudo_dir.scores = os.path.dirname(score)
    pseudo_dir.all_scores = os.getcwd()

    return pseudo_dir