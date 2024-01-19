from __future__ import annotations

import _pickle
import abc
import logging
import os
import subprocess
import sys
import time
import traceback
from abc import ABC
from collections import UserList, defaultdict
from collections.abc import Callable, Container, Iterable, Sequence, Collection, Generator
from copy import copy
from itertools import count, repeat
from logging import Logger
from pathlib import Path
from random import random
from typing import Any, AnyStr, Generic, get_args, IO, Literal, Type, TypeVar, Union

import numpy as np
from sklearn.neighbors import BallTree

from .coordinates import Coordinates, superposition3d
from . import fragment, utils as stutils
from symdesign.sequence import protein_letters3_alph1, protein_letters_alph1, protein_letters_1to3, \
    protein_letters_3to1_extended, protein_letters3_alph1_literal
from symdesign import utils
from symdesign.metrics import default_sasa_burial_threshold
from symdesign.third_party.pdbecif.src.pdbecif.mmcif_io import CifFileReader
putils = utils.path

# Globals
cif_reader = CifFileReader()
logger = logging.getLogger(__name__)
protein_letters_3to1_extended_mse = protein_letters_3to1_extended.copy()
protein_letters_3to1_extended_mse['MSE'] = 'M'
DEFAULT_SS_PROGRAM = 'stride'
DEFAULT_SS_COIL_IDENTIFIER = 'C'
"""Secondary structure identifier mapping
Stride
B/b:Isolated bridge
E:Strand/Extended conformation 
G:3-10 helix
H:Alpha helix
I:PI-helix
T:Turn
C:Coil (none of the above)
SS_TURN_IDENTIFIERS = 'T'
SS_DISORDER_IDENTIFIERS = 'C'

DSSP
B:beta bridge
E:strand/beta bulge
G:310 helix
H:α helix
I:π helix
T:turns
S:high curvature (where the angle between i-2, i, and i+2 is at least 70°)
" "(space):loop
SS_DISORDER_IDENTIFIERS = ' '
SS_TURN_IDENTIFIERS = 'TS'
"""
SS_HELIX_IDENTIFIERS = 'H'  # Todo is 310 helix desired?
SS_TURN_IDENTIFIERS = 'T'
SS_DISORDER_IDENTIFIERS = 'C'
directives = Literal[
    'special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding', 'branched']
mutation_directives: tuple[directives, ...] = get_args(directives)
atom_or_residue_literal = Literal['atom', 'residue']
structure_container_types = Literal['atoms', 'residues', 'chains', 'entities']
protein_backbone_atom_types = {'N', 'CA', 'C', 'O'}
protein_backbone_and_cb_atom_types = {'N', 'CA', 'C', 'O', 'CB'}
phosphate_backbone_atom_types = {'P', 'OP1', 'OP2'}
# P, OP1, OP2,
dna_sugar_atom_types = {"O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
# O5', C5', C4', O4', C3', O3', C2', O2', C1'  # RNA only
rna_sugar_atom_types = dna_sugar_atom_types | {"O2'"}
# O5', C5', C4', O4', C3', O3', C2', C1'  # DNA only
# For A, i.e. adenosine
# N9, C8, N7, C5, C6, N6, N1, C2, N3, C4
# For DA, i.e. deoxyadenosine
# N9, C8, N7, C5, C6, N6, N1, C2, N3, C4
# For C, i.e. cytosine
# N1, C2, O2, N3, C4, N4, C5, C6
# For DC, i.e. deoxycytosine
# N1, C2, O2, N3, C4, N4, C5, C6
# For G i.e. guanosine
# N9, C8, N7, C5, C6, O6, N1, C2, N2, N3, C4
# For DG i.e. deoxyguanosine
# N9, C8, N7, C5, C6, O6, N1, C2, N2, N3, C4
# For U i.e. urosil
# N1, C2, O2, N3, C4, O4, C5, C6
# For DT, i.e. deoxythymidine
# N1, C2, O2, N3, C4, O4, C5, C7, C6
residue_properties = {
    'ALA': {'hydrophobic', 'apolar'},
    'CYS': {'special', 'hydrophobic', 'apolar', 'polar', 'hbonding'},
    'ASP': {'charged', 'polar', 'hbonding'},
    'GLU': {'charged', 'polar', 'hbonding'},
    'PHE': {'hydrophobic', 'apolar', 'aromatic'},
    'GLY': {'special'},
    'HIS': {'charged', 'polar', 'aromatic', 'hbonding'},
    'ILE': {'hydrophobic', 'apolar', 'branched'},
    'LYS': {'charged', 'polar', 'hbonding'},
    'LEU': {'hydrophobic', 'apolar', 'branched'},
    'MET': {'hydrophobic', 'apolar'},
    'ASN': {'polar', 'hbonding'},
    'PRO': {'special', 'hydrophobic', 'apolar'},
    'GLN': {'polar', 'hbonding'},
    'ARG': {'charged', 'polar', 'hbonding'},
    'SER': {'polar', 'hbonding'},
    'THR': {'polar', 'hbonding', 'branched'},
    'VAL': {'hydrophobic', 'apolar', 'branched'},
    'TRP': {'hydrophobic', 'apolar', 'aromatic', 'hbonding'},
    'TYR': {'hydrophobic', 'apolar', 'aromatic', 'hbonding'}
}
# useful in generating aa_by_property from mutation_directives and residue_properties
# aa_by_property = {}
# for type_ in mutation_directives:
#     aa_by_property[type_] = set()
#     for res in residue_properties:
#         if type_ in residue_properties[res]:
#             aa_by_property[type_].append(res)
#     aa_by_property[type_] = list(aa_by_property[type_])
aa_by_property = \
    {'special': {'CYS', 'GLY', 'PRO'},
     'charged': {'ARG', 'GLU', 'ASP', 'HIS', 'LYS'},
     'polar': {'CYS', 'ASP', 'GLU', 'HIS', 'LYS', 'ASN', 'GLN', 'ARG', 'SER', 'THR'},
     'apolar': {'ALA', 'CYS', 'PHE', 'ILE', 'LEU', 'MET', 'PRO', 'VAL', 'TRP', 'TYR'},
     'hydrophobic': {'ALA', 'CYS', 'PHE', 'ILE', 'LEU', 'MET', 'PRO', 'VAL', 'TRP', 'TYR'},  # same as apolar
     'aromatic': {'PHE', 'HIS', 'TRP', 'TYR'},
     'hbonding': {'CYS', 'ASP', 'GLU', 'HIS', 'LYS', 'ASN', 'GLN', 'ARG', 'SER', 'THR', 'TRP', 'TYR'},
     'branched': {'ILE', 'LEU', 'THR', 'VAL'}}
gxg_sasa = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
            'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
            'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167, 'GLU': 223, 'GLN': 225, 'GLY': 104, 'HIS': 224,
            'ILE': 197, 'LEU': 201, 'LYS': 236, 'MET': 224, 'PHE': 240, 'PRO': 159, 'SER': 155, 'THR': 172, 'TRP': 285,
            'TYR': 263, 'VAL': 174}  # From table 1, theoretical values of Tien et al. 2013, PMID:24278298
# This publication claims to have normalized tripeptides of all the standard AA which can be used to calculate non-polar
# and polar area: https://doi.org/10.1016/j.compbiolchem.2014.11.007
# They are not actually available, however. Should email authors to acquire these...
# Set up hydrophobicity values for various calculations
black_and_mould = [
    0.702, 0.987, -1.935, -1.868, 2.423, 0.184, -1.321, 2.167, -0.790, 2.167, 1.246, -1.003, 1.128, -0.936, -2.061,
    -0.453, -0.042, 1.640, 1.878, 1.887
]
hydrophobicity_values = \
    dict(black_and_mould=dict(zip(protein_letters3_alph1, black_and_mould)))
glycine_val = black_and_mould[5]
# This is used for the SAP calculation. See PMID:19571001
hydrophobicity_values_glycine_centered = {value_name: {aa: value - glycine_val for aa, value in values.items()}
                                          for value_name, values in hydrophobicity_values.items()}


def unknown_index():
    return -1


polarity_types_literal = Literal['apolar', 'polar']
sasa_types_literal = Literal['total', polarity_types_literal]
sasa_types: tuple[polarity_types_literal, ...] = get_args(sasa_types_literal)
polarity_types: tuple[polarity_types_literal, ...] = get_args(polarity_types_literal)
# Todo add nucleotide polarities to this table. The atom types are located above. Polarities in freesasa-2.0.config
atomic_polarity_table = {  # apolar = 0, polar = 1
    'ALA': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0}),
    'ARG': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'NE': 1, 'CZ': 0,
                                       'NH1': 1, 'NH2': 1}),
    'ASN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'ND2': 1}),
    'ASP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'OD2': 1}),
    'CYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'SG': 1}),
    'GLN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'NE2': 1}),
    'GLU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'OE2': 1}),
    'GLY': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1}),
    'HIS': defaultdict(unknown_index,
                       {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'ND1': 1, 'CD2': 0, 'CE1': 0, 'NE2': 1}),
    'ILE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0, 'CD1': 0}),
    'LEU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0}),
    'LYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'CE': 0, 'NZ': 1}),
    'MET': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'SD': 1, 'CE': 0}),
    'PHE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0,
                                       'CE2': 0, 'CZ': 0}),
    'PRO': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0}),
    'SER': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG': 1}),
    'THR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG1': 1, 'CG2': 0}),
    'TRP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'NE1': 1,
                                       'CE2': 0, 'CE3': 0, 'CZ2': 0, 'CZ3': 0, 'CH2': 0}),
    'TYR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0,
                                       'CE2': 0, 'CZ': 0, 'OH': 1}),
    'VAL': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0})}
hydrogens = {   # The doubled numbers (and single number second) are from PDB version of hydrogen inclusion
    'ALA': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '3HB': 0, 'HB1': 0, 'HB2': 0, 'HB3': 0},
    'ARG': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, 'HE': 1, '1HH1': 1, '2HH1': 1,
            '1HH2': 1, '2HH2': 1,
            'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0, 'HD1': 0, 'HD2': 0, 'HH11': 1, 'HH12': 1, 'HH21': 1, 'HH22': 1},
    'ASN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HD2': 1, '2HD2': 1, 'HB1': 0, 'HB2': 0, 'HD21': 1, 'HD22': 1,
            '1HD1': 1, '2HD1': 1, 'HD11': 1, 'HD12': 1},  # These are the alternative specification
    'ASP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HB1': 0, 'HB2': 0},
    'CYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HB1': 0, 'HB2': 0, 'HG': 1},
    'GLN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE2': 1, '2HE2': 1, 'HB1': 0, 'HB2': 0, 'HG1': 0,
            'HG2': 0, 'HE21': 1, 'HE22': 1,
            '1HE1': 1, '2HE1': 1, 'HE11': 1, 'HE12': 1},  # These are the alternative specification
    'GLU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, 'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0},
    'GLY': {'H': 1, '1HA': 0, 'HA1': 0, '2HA': 0, 'HA2': 0, '3HA': 0, 'HA3': 0},
    'HIS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 1, 'HD2': 0, 'HE1': 0, 'HE2': 1, 'HB1': 0, 'HB2': 0, '1HD': 1,
            '2HD': 0, '1HE': 0, '2HE': 1},  # This assumes HD1 is on ND1, HE2 is on NE2
    'ILE': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0, '1HD1': 0, '2HD1': 0,
            '3HD1': 0, 'HG11': 0, 'HG12': 0, 'HG21': 0, 'HG22': 0, 'HG23': 0, 'HD11': 0, 'HD12': 0, 'HD13': 0,
            'HG13': 0, '3HG1': 0},  # this is the alternative specification
    'LEU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 0, '1HD1': 0, '2HD1': 0, '3HD1': 0, '1HD2': 0, '2HD2': 0,
            '3HD2': 0, 'HB1': 0, 'HB2': 0, 'HD11': 0, 'HD12': 0, 'HD13': 0, 'HD21': 0, 'HD22': 0, 'HD23': 0},
    'LYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, '1HE': 0, '2HE': 0, '1HZ': 1,
            '2HZ': 1, '3HZ': 1, 'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0,
            'HZ1': 1, 'HZ2': 1, 'HZ3': 1},
    'MET': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE': 0, '2HE': 0, '3HE': 0, 'HB1': 0, 'HB2': 0,
            'HG1': 0, 'HG2': 0, 'HE1': 0, 'HE2': 0, 'HE3': 0},
    'PHE': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HB1': 0, 'HB2': 0, '1HD': 0,
            '2HD': 0, '1HE': 0, '2HE': 0, 'HZ': 0},
    'PRO': {'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 1, 'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0,
            'HD1': 0, 'HD2': 1},
    # Yes, 3HB is not in existence on this residue, but 5upp.pdb has it...
    'SER': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '3HB': 0, 'HB1': 0, 'HB2': 0, 'HB3': 0, 'HG': 1},
    'THR': {'HA': 0, 'HB': 0, 'H': 1, 'HG1': 1, '1HG2': 0, '2HG2': 0, '3HG2': 0, '1HG': 1, 'HG21': 0, 'HG22': 0,
            # these are the alternative specification
            'HG23': 0, 'HG2': 1, '1HG1': 0, '2HG1': 0, '3HG1': 0, '2HG': 1, 'HG11': 0, 'HG12': 0, 'HG13': 0},
    'TRP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HE1': 1, 'HE3': 0, 'HZ2': 0, 'HZ3': 0, 'HH2': 0, 'HB1': 0,
            'HB2': 0, '1HD': 0, '1HE': 1, '3HE': 0, '2HZ': 0, '3HZ': 0, '2HH': 0,  # assumes HE1 is on NE1
            'HE2': 0, 'HZ1': 0, 'HH1': 0, 'HH3': 0, '2HE': 0, '1HZ': 0, '1HH': 0, '3HH': 0},  # none of these should be possible given standard nomenclature, but including incase
    'TYR': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HB1': 0, 'HB2': 0, '1HD': 0,
            '2HD': 0, '1HE': 0, '2HE': 0, 'HH': 1},
    'VAL': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '3HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0, 'HG11': 0,
            'HG12': 0, 'HG13': 0, 'HG21': 0, 'HG22': 0, 'HG23': 0}}
termini_polarity = {'1H': 1, '2H': 1, '3H': 1, 'H': 1, 'H1': 1, 'H2': 1, 'H3': 1, 'OXT': 1}
for res_type, residue_atoms in atomic_polarity_table.items():
    residue_atoms.update(termini_polarity)
    residue_atoms.update(hydrogens[res_type])


def parse_seqres(seqres_lines: list[str]) -> dict[str, str]:  # list[str]:
    """Convert SEQRES information to single amino acid dictionary format

    Args:
        seqres_lines: The list of lines containing SEQRES information
    Returns:
        The mapping of each chain to its reference sequence
    """
    # SEQRES   1 A  182  THR THR ALA SER THR SER GLN VAL ARG GLN ASN TYR HIS
    # SEQRES   2 A  182  GLN ASP SER GLU ALA ALA ILE ASN ARG GLN ILE ASN LEU
    # SEQRES   3 A  182  GLU LEU TYR ALA SER TYR VAL TYR LEU SER MET SER TYR
    # SEQRES ...
    # SEQRES  16 C  201  SER TYR ILE ALA GLN GLU
    # In order to account for MultiModel files where the chain names are all the same, using the parsed order
    # instead of a dictionary as later entries would overwrite earlier ones making them inaccurate
    # If the file is screwed up in that it has chains in a different order than the seqres, then this wouldn't work
    # I am opting for the standard .pdb file format and if it is messed up this is the users problem
    reference_sequence = {}
    for line in seqres_lines:
        chain, length, *sequence = line.split()
        if chain in reference_sequence:
            reference_sequence[chain].extend(list(sequence))
        else:
            reference_sequence[chain] = list(sequence)

    # Format the sequences as a one AA letter list
    reference_sequences = {}  # []
    for chain, sequence in reference_sequence.items():
        # Ensure we parse selenomethionine correctly
        one_letter_sequence = [protein_letters_3to1_extended_mse.get(aa, '-')
                               for aa in sequence]
        reference_sequences[chain] = ''.join(one_letter_sequence)
        # reference_sequences.append(''.join(one_letter_sequence))

    return reference_sequences


atom_index_slice = slice(-5, None)
slice_remark, slice_number, slice_atom_type, slice_alt_location, slice_residue_type, slice_chain, \
    slice_residue_number, slice_code_for_insertion, slice_x, slice_y, slice_z, slice_occ, slice_temp_fact, \
    slice_element, slice_charge = slice(0, 6), slice(6, 11), slice(12, 16), slice(16, 17), slice(17, 20), \
    slice(21, 22), slice(22, 26), slice(26, 27), slice(30, 38), slice(38, 46), slice(46, 54), slice(54, 60), \
    slice(60, 66), slice(76, 78), slice(78, 80)


def read_pdb_file(file: AnyStr = None, pdb_lines: Iterable[str] = None, separate_coords: bool = True, **kwargs) -> \
        dict[str, Any]:
    """Reads .pdb file and returns structural information pertaining to parsed file

    By default, returns the coordinates as a separate numpy.ndarray which is parsed directly by StructureBase. This will
    be associated with each Atom however, separate parsing is done for efficiency. To include coordinate info with the
    individual Atom instances, pass separate_coords=False. (Not recommended)

    Args:
        file: The path to the file to parse
        pdb_lines: If lines are already read, provide the lines instead
        separate_coords: Whether to separate parsed coordinates from Atom instances. Will be returned as two separate
            entries in the parsed dictionary, otherwise returned with coords=None
    Returns:
        The dictionary containing all the parsed structural information
    """
    if pdb_lines:
        # path, extension = None, None
        assembly: str | None = None
        name = None
    elif file is not None:
        with open(file, 'r') as f:
            pdb_lines = f.readlines()
        path, extension = os.path.splitext(file)
        name = os.path.basename(path)

        if extension[-1].isdigit():
            # If last character is not a letter, then the file is an assembly, or the extension was provided weird
            assembly: str | None = extension.translate(utils.keep_digit_table)
        else:
            assembly = None
    else:
        raise ValueError(
            f"{read_pdb_file.__name__}: Must provide the argument 'file' or 'pdb_lines'")

    # type to info index:   1    2    3    4    5    6    7     11     12   13   14
    # eventual type:      int, str, str, str, str, int, str, float, float, str, str]] = []
    temp_info: list[tuple[str, str, str, str, str, str, str, str, str, str, str]] = []
    # type to info index:   1    2    3    4    5    6    7 8,9,10     11     12   13   14
    # fields w/ coords   [int, str, str, str, str, int, str, float, float, float, str, str]
    coords: list[list[float]] = []
    cryst_record: str = None
    dbref: dict[str, dict[str, str]] = {}
    entity_info: dict[str, dict[str, dict | list | str]] = {}
    header: list = []
    resolution: float | None = None
    seq_res_lines: list[str] = []
    biomt = []

    entity = None
    current_operation = -1
    alt_loc_str = ' '
    # for line_tokens in map(str.split, pdb_lines):
    #     # 0       1       2          3             4             5      6               7                   8  9
    #     # remark, number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, x, y,
    #     #     10 11   12         13       14
    #     #     z, occ, temp_fact, element, charge = \
    #     #     line[6:11].strip(), int(line[6:11]), line[12:16].strip(), line[16:17].strip(), line[17:20].strip(),
    #     #     line[21:22], int(line[22:26]), line[26:27].strip(), float(line[30:38]), float(line[38:46]), \
    #     #     float(line[46:54]), float(line[54:60]), float(line[60:66]), line[76:78].strip(), line[78:80].strip()
    for line in pdb_lines:
        remark = line[slice_remark]
        if remark == 'ATOM  ' or line[slice_residue_type] == 'MSE' and remark == 'HETATM':
            # if remove_alt_location and alt_location not in ['', 'A']:
            if line[slice_alt_location] not in [alt_loc_str, 'A']:
                continue
            # number = int(line[slice_number])
            residue_type = line[slice_residue_type].strip()
            if residue_type == 'MSE':
                residue_type = 'MET'
                atom_type = line[slice_atom_type].strip()
                if atom_type == 'SE':
                    atom_type = 'SD'  # change type from Selenium to Sulfur delta
            else:
                atom_type = line[slice_atom_type].strip()
            # prepare line information for population of Atom objects
            temp_info.append((line[slice_number], atom_type, alt_loc_str, residue_type, line[slice_chain],
                              line[slice_residue_number], line[slice_code_for_insertion].strip(),
                              line[slice_occ], line[slice_temp_fact],
                              line[slice_element].strip(), line[slice_charge].strip()))
            # temp_info.append((int(line[slice_number]), atom_type, alt_loc_str, residue_type, line[slice_chain],
            #                   int(line[slice_residue_number]), line[slice_code_for_insertion].strip(),
            #                   float(line[slice_occ]), float(line[slice_temp_fact]),
            #                   line[slice_element].strip(), line[slice_charge].strip()))
            # Prepare the atomic coordinates for addition to numpy array
            coords.append([float(line[slice_x]), float(line[slice_y]), float(line[slice_z])])
        elif remark == 'SEQRES':
            seq_res_lines.append(line[11:])
        elif remark == 'REMARK':
            header.append(line.strip())
            remark_number = line[slice_number]
            # elif line[:18] == 'REMARK 350   BIOMT':
            if remark_number == ' 350 ':  # 6:11  '   BIOMT'
                # integration of the REMARK 350 BIOMT
                # REMARK 350
                # REMARK 350 BIOMOLECULE: 1
                # REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: TRIMERIC
                # REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: TRIMERIC
                # REMARK 350 SOFTWARE USED: PISA
                # REMARK 350 TOTAL BURIED SURFACE AREA: 6220 ANGSTROM**2
                # REMARK 350 SURFACE AREA OF THE COMPLEX: 28790 ANGSTROM**2
                # REMARK 350 CHANGE IN SOLVENT FREE ENERGY: -42.0 KCAL/MOL
                # REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B, C
                # REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
                # REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
                # REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
                try:
                    _, _, biomt_indicator, operation_number, x, y, z, tx = line.split()
                except ValueError:  # Not enough values to unpack
                    continue
                if biomt_indicator == 'BIOMT':
                    if operation_number != current_operation:  # Reached a new transformation matrix
                        current_operation = operation_number
                        biomt.append([])
                    # Add the transformation to the current matrix
                    biomt[-1].append(list(map(float, (x, y, z, tx))))
            elif remark_number == '   2 ':  # 6:11 ' RESOLUTION'
                try:
                    resolution = float(line[22:30].strip().split()[0])
                except (IndexError, ValueError):
                    resolution = None
        elif 'DBREF' in remark:
            header.append(line.strip())
            chain = line[12:14].strip().upper()
            if line[5:6] == '2':
                db_accession_id = line[18:40].strip()
            else:
                db = line[26:33].strip()
                if line[5:6] == '1':  # skip grabbing db_accession_id until DBREF2
                    continue
                db_accession_id = line[33:42].strip()
            dbref[chain] = {'db': db, 'accession': db_accession_id}  # implies each chain has only one id
        elif remark == 'COMPND' and 'MOL_ID' in line:
            header.append(line.strip())
            entity = line[line.rfind(':') + 1: line.rfind(';')].strip()
        elif remark == 'COMPND' and 'CHAIN' in line and entity:  # retrieve from standard .pdb file notation
            header.append(line.strip())
            # entity number (starting from 1) = {'chains' : {A, B, C}}
            entity_info[f'{name}_{entity}'] = \
                {'chains': list(map(str.strip, line[line.rfind(':') + 1:].strip().rstrip(';').split(',')))}
            entity = None
        elif remark == 'SCALE ':
            header.append(line.strip())
        elif remark == 'CRYST1':
            header.append(line.strip())
            cryst_record = line  # Don't .strip() so '\n' is attached for output
            # uc_dimensions, space_group = parse_cryst_record(cryst_record)
            # cryst = {'space': space_group, 'a_b_c': tuple(uc_dimensions[:3]), 'ang_a_b_c': tuple(uc_dimensions[3:])}

    if not temp_info:
        if file:
            raise ValueError(
                f'The file {file} has no ATOM records')
        else:
            raise ValueError("The provided 'pdb_lines' have no ATOM records")

    # Combine entity_info with the reference_sequence info and dbref info
    if seq_res_lines:
        reference_sequence = parse_seqres(seq_res_lines)
    else:
        reference_sequence = None

    for entity_name, info in entity_info.items():
        # Grab the first chain from the identified chains, and use it to grab the reference sequence
        chain = info['chains'][0]
        try:
            info['reference_sequence'] = reference_sequence[chain]  # Used when parse_seqres returns dict[str, str]
        except TypeError:  # This is None
            pass
        try:
            info['dbref'] = dbref[chain]
        except KeyError:  # Keys are missing
            pass

    # # Convert the incrementing reference sequence to a list of the sequences
    # reference_sequence = list(reference_sequence.values())

    if biomt:
        biomt = np.array(biomt, dtype=float)
        rotation_matrices = biomt[:, :, :3]
        translation_matrices = biomt[:, :, 3:].squeeze()
    else:
        rotation_matrices = translation_matrices = None

    parsed_info = \
        dict(atoms=[Atom.without_coordinates(idx, *info) for idx, info in enumerate(temp_info)]
             if separate_coords else
             # Initialize with individual coords. Not sure why anyone would do this, but include for compatibility
             [Atom(number=int(number), atom_type=atom_type, alt_location=alt_location, residue_type=residue_type,
                   chain_id=chain_id, residue_number=int(residue_number), code_for_insertion=code_for_insertion,
                   coords=coords[idx], occupancy=float(occupancy), b_factor=float(b_factor), element=element,
                   charge=charge)
              for idx, (number, atom_type, alt_location, residue_type, chain_id, residue_number, code_for_insertion,
                        occupancy, b_factor, element, charge)
              in enumerate(temp_info)],
             biological_assembly=assembly,
             rotation_matrices=rotation_matrices,
             translation_matrices=translation_matrices,
             coords=coords if separate_coords else None,
             cryst_record=cryst_record,
             entity_info=entity_info,
             name=name,
             resolution=resolution,
             reference_sequence=reference_sequence,
             )
    # Explicitly overwrite any parsing if argument was passed to caller
    parsed_info.update(**kwargs)
    return parsed_info


def read_mmcif_file(file: AnyStr = None, **kwargs) -> dict[str, Any]:
    """Reads .cif file and returns structural information pertaining to parsed file

    By default, returns the coordinates as a separate numpy.ndarray which is parsed directly by StructureBase. This will
    be associated with each Atom however, separate parsing is done for efficiency. To include coordinate info with the
    individual Atom instances, pass separate_coords=False. (Not recommended)

    Args:
        file: The path to the file to parse
    Returns:
        The dictionary containing all the parsed structural information
    """
    # if lines:
    #     # path, extension = None, None
    #     assembly: str | None = None
    #     name = None
    # el
    if file is not None:
        path, extension = os.path.splitext(file)
        name = os.path.basename(path)
        # Todo
        #  Add all fields that are not:
        #  '_atom_site', '_cell', '_symmetry', '_reflns', '_entity_poly', '_struct_ref', '_pdbx_struct_oper_list'
        ignore_fields = []
        data: dict[str, dict[str, Any]] = cif_reader.read(file, ignore=ignore_fields)
        if extension[-1].isdigit():
            # If last character is not a letter, then the file is an assembly, or the extension was provided weird
            assembly: str | None = extension.translate(utils.keep_digit_table)
        # Todo debug reinstate v
        elif 'assembly' in name:
            assembly = name[name.find('assembly'):].translate(utils.keep_digit_table)
        else:
            assembly = None
    else:
        raise ValueError(
            f"{read_mmcif_file.__name__}: Must provide the argument 'file'"
            # f" or 'lines'")
        )

    # name = kwargs.pop('name', None)
    # if not name:
    #     name = os.path.basename(os.path.splitext(file)[0])

    # input(data.keys())
    # for k, v_ in data.items():
    #     # input(f'{list(v_.keys())}')
    #     # ['_entry', '_audit_conform', '_database_2', '_pdbx_database_PDB_obs_spr', '_pdbx_database_related',
    #     #  '_pdbx_database_status', '_audit_author', '_citation', '_citation_author', '_cell', '_symmetry',
    #     #  '_entity', '_entity_poly', '_entity_poly_seq', '_entity_src_gen', '_struct_ref', '_struct_ref_seq',
    #     #  '_struct_ref_seq_dif', '_chem_comp', '_exptl', '_exptl_crystal', '_exptl_crystal_grow', '_diffrn',
    #     #  '_diffrn_detector', '_diffrn_radiation', '_diffrn_radiation_wavelength', '_diffrn_source', '_reflns',
    #     #  '_reflns_shell', '_refine', '_refine_hist', '_refine_ls_restr', '_refine_ls_shell', '_pdbx_refine',
    #     #  '_struct', '_struct_keywords', '_struct_asym', '_struct_biol', '_struct_conf', '_struct_conf_type',
    #     #  '_struct_mon_prot_cis', '_struct_sheet', '_struct_sheet_order', '_struct_sheet_range',
    #     #  '_pdbx_struct_sheet_hbond', '_atom_sites', '_atom_type', '_atom_site', '_atom_site_anisotrop',
    #     #  '_pdbx_poly_seq_scheme', '_pdbx_struct_assembly', '_pdbx_struct_assembly_gen',
    #     #  '_pdbx_struct_assembly_prop', '_pdbx_struct_oper_list', '_pdbx_audit_revision_history',
    #     #  '_pdbx_audit_revision_details', '_pdbx_audit_revision_group', '_pdbx_refine_tls',
    #     #  '_pdbx_refine_tls_group', '_pdbx_phasing_MR', '_phasing', '_software', '_pdbx_validate_torsion',
    #     #  '_pdbx_unobs_or_zero_occ_atoms', '_pdbx_unobs_or_zero_occ_residues', '_space_group_symop']
    #     for idx, (k, v) in enumerate(v_.items()):
    #         # if k == '_atom_sites':
    #         #     input(v.keys())
    #         # if k in ['_database_2', '_pdbx_database_PDB_obs_spr', '_pdbx_database_related', '_pdbx_database_status']:
    #         #     print('Database key', k)
    #         #     input(v)
    #         # if k in ['_entity', '_entity_poly', '_entity_poly_seq', '_struct_ref']:  # '_entity_src_gen', '_struct_ref_seq'
    #         #     print('Sequence key', k)
    #         #     input(v)
    #         # if k in ['_cell', '_symmetry',
    #         #          # '_space_group_symop'
    #         #          ]:
    #         #     print('Symmetry key', k)
    #         #     input(v)
    #         # if k in ['_reflns', '_reflns_shell', '_refine', '_refine_hist', '_refine_ls_restr', '_refine_ls_shell', '_pdbx_refine',]:
    #         #     print('Diffraction key', k)
    #         #     input(v)
    #         # if k in [
    #         #     # '_struct', '_struct_keywords', '_struct_asym', '_struct_biol',
    #         #     '_pdbx_struct_assembly', '_pdbx_struct_assembly_gen',
    #         #     # '_pdbx_struct_assembly_prop',
    #         #     '_pdbx_struct_oper_list',
    #         # ]:
    #         #     print('struct key', k)
    #         #     input(v)
    #         # if k == '_atom_site':
    #         #     """
    #         #     key group_PDB
    #         #     values ['ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM', 'ATOM']
    #         #     key id
    #         #     values ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    #         #     key type_symbol
    #         #     values ['N', 'C', 'C', 'O', 'C', 'O', 'N', 'C', 'C', 'O']
    #         #     key label_atom_id
    #         #     values ['N', 'CA', 'C', 'O', 'CB', 'OG', 'N', 'CA', 'C', 'O']
    #         #     key label_alt_id
    #         #     values ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
    #         #     key label_comp_id
    #         #     values ['SER', 'SER', 'SER', 'SER', 'SER', 'SER', 'VAL', 'VAL', 'VAL', 'VAL']
    #         #     key label_asym_id
    #         #     values ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    #         #     key label_entity_id
    #         #     values ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
    #         #     key label_seq_id
    #         #     values ['3', '3', '3', '3', '3', '3', '4', '4', '4', '4']
    #         #     key pdbx_PDB_ins_code
    #         #     values ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?']
    #         #     key Cartn_x
    #         #     values ['25.947', '25.499', '24.208', '23.310', '26.585', '27.819', '24.126', '22.943', '22.353', '23.081']
    #         #     key Cartn_y
    #         #     values ['8.892', '10.149', '9.959', '10.800', '10.734', '10.839', '8.851', '8.533', '7.200', '6.224']
    #         #     key Cartn_z
    #         #     values ['43.416', '42.828', '42.038', '42.084', '41.925', '42.615', '41.310', '40.519', '40.967', '41.152']
    #         #     key occupancy
    #         #     values ['1.00', '1.00', '1.00', '1.00', '1.00', '1.00', '1.00', '1.00', '1.00', '1.00']
    #         #     key B_iso_or_equiv
    #         #     values ['67.99', '79.33', '61.67', '60.15', '81.39', '86.58', '62.97', '56.54', '56.17', '85.78']
    #         #     key pdbx_formal_charge
    #         #     values ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?']
    #         #     key auth_seq_id
    #         #     values ['3', '3', '3', '3', '3', '3', '4', '4', '4', '4']
    #         #     key auth_comp_id
    #         #     values ['SER', 'SER', 'SER', 'SER', 'SER', 'SER', 'VAL', 'VAL', 'VAL', 'VAL']
    #         #     key auth_asym_id
    #         #     values ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    #         #     key auth_atom_id
    #         #     values ['N', 'CA', 'C', 'O', 'CB', 'OG', 'N', 'CA', 'C', 'O']
    #         #     key pdbx_PDB_model_num
    #         #     values ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
    #         #     """
    #         #     for k__, v__ in v.items():
    #         #         # ['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'pdbx_PDB_model_num']
    #         #         print('key', k__)
    #         #         input(f'values {v__[:10]}')
    #         #     input('DONE')
    #         pass

    provided_dataname, *_ = data.keys()
    if _:
        raise ValueError(
            f"Found multiple values for the cif 'provided_dataname'={_}")

    # Extract the data of interest as saved by the provided_dataname
    data = data[provided_dataname]

    # def format_mmcif_dict(data: dict[str, Any], name: str = None, **kwargs) -> dict:
    atom_data = data.get('_atom_site')
    atom_numbers = atom_data.get('id')
    number_of_atoms = len(atom_numbers)
    atom_types = atom_data.get('label_atom_id')
    alt_locations = atom_data.get('label_alt_id', repeat(' ', number_of_atoms))
    residue_types = atom_data.get('label_comp_id')
    chains = atom_data.get('label_asym_id')
    residue_numbers = atom_data.get('label_seq_id')
    occupancies = atom_data.get('occupancy')
    code_for_insertion = atom_data.get('', repeat(' ', number_of_atoms))
    b_factors = atom_data.get('B_iso_or_equiv')
    element = atom_data.get('type_symbol', repeat(None, number_of_atoms))
    charge = atom_data.get('pdbx_formal_charge', repeat(None, number_of_atoms))
    alt_locations = ''.join(alt_locations).replace('.', ' ')
    code_for_insertion = ''.join(code_for_insertion).replace('.', ' ')
    charge = ''.join(charge).replace('?', ' ')

    atoms = [Atom.without_coordinates(idx, *info) for idx, info in enumerate(
        zip(atom_numbers, atom_types, alt_locations, residue_types, chains, residue_numbers, code_for_insertion,
            occupancies, b_factors, element, charge))]

    coords = np.array([atom_data.get('Cartn_x'),
                       atom_data.get('Cartn_y'),
                       atom_data.get('Cartn_z')], dtype=float).T
    cell_data = data.get('_cell')
    if cell_data:
        # 'length_a': '124.910', 'length_b': '189.250', 'length_c': '376.830',
        # 'angle_alpha': '90.00', 'angle_beta': '90.02', 'angle_gamma': '90.00'
        # Formatted in Hermann-Mauguin notation
        space_group = data.get('_symmetry', {}).get('space_group_name_H-M')
        cryst_record = utils.symmetry.generate_cryst1_record(
            list(map(float, (cell_data['length_a'], cell_data['length_b'], cell_data['length_c'],
                             cell_data['angle_alpha'], cell_data['angle_beta'], cell_data['angle_gamma']))),
            space_group)
    else:
        cryst_record = None

    reflections_data = data.get('_reflns')
    if reflections_data:
        resolution = reflections_data['d_resolution_high']
    else:
        resolution = None

    # _struct_ref
    # Get the cannonical sequence
    entity_data = data.get('_entity_poly')
    db_data = data.get('_struct_ref')
    if db_data:
        # 'db_name': ['UNP', 'UNP'], 'db_code': ['B0BGB0_9BACT', 'Q4Q413_LEIMA'],
        # 'pdbx_db_accession': ['B0BGB0', 'Q4Q413'], 'entity_id': ['1', '2'],
        # 'pdbx_seq_one_letter_code': ['MESVNTSFLSPSLVTIRDFDNGQFAVLRIGRTGFPADKGDIDLCLDKMKGVRDAQQSIGDDTEFGFKGPHIRIRCVDIDD\nKHTYNAMVYVDLIVGTGASEVERETAEELAKEKLRAALQVDIADEHSCVTQFEMKLREELLSSDSFHPDKDEYYKDFL', 'MPVIQTFVSTPLDHHKRENLAQVYRAVTRDVLGKPEDLVMMTFHDSTPMHFFGSTDPVACVRVEALGGYGPSEPEKVTSI\nVTAAITKECGIVADRIFVLYFSPLHCGWNGTNF']
        entity_info = {f'{name}_{entity_id}': {'chains': [],
                                               'reference_sequence': reference_sequence.replace('\n', ''),
                                               'dbref': {'db': db_name, 'accession': db_accession}}
                       for entity_id, reference_sequence, db_name, db_accession in zip(
                db_data.get('entity_id'),
                entity_data.get('pdbx_seq_one_letter_code_can'),  # db_data.get('pdbx_seq_one_letter_code'),
                db_data.get('db_name'), db_data.get('pdbx_db_accession'))
                       }
    else:
        entity_info = {}
    # Todo
    #  'rcsb_polymer_entity_container_identifiers' 'asym_id'
    # struct key _struct_asym
    # {'id': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'BA', 'CA', 'DA', 'EA', 'FA', 'GA', 'HA', 'IA', 'JA', 'KA', 'LA', 'MA', 'NA', 'OA', 'PA', 'QA', 'RA', 'SA', 'TA', 'UA', 'VA', 'WA', 'XA', 'YA', 'ZA', 'AB', 'BB', 'CB', 'DB', 'EB', 'FB', 'GB', 'HB', 'IB', 'JB', 'KB', 'LB', 'MB', 'NB', 'OB', 'PB', 'QB', 'RB', 'SB', 'TB', 'UB', 'VB', 'WB', 'XB', 'YB', 'ZB', 'AC', 'BC', 'CC', 'DC', 'EC', 'FC', 'GC', 'HC', 'IC', 'JC', 'KC', 'LC', 'MC', 'NC', 'OC', 'PC', 'QC', 'RC'],
    # struct key _struct_biol
    # {'id': '1', 'details': 'The biological assembly is a protein cage with tetrahedral point group symmetry that comprises twenty-four subunits, where twelve each are a total of four homotrimers.'}
    # struct key _pdbx_struct_assembly
    # {'id': ['1', '2', '3', '4'], 'details': ['author_and_software_defined_assembly', 'author_and_software_defined_assembly', 'author_and_software_defined_assembly', 'author_and_software_defined_assembly'], 'method_details': ['PISA', 'PISA', 'PISA', 'PISA'], 'oligomeric_details': ['24-meric', '24-meric', '24-meric', '24-meric'], 'oligomeric_count': ['24', '24', '24', '24']}
    # struct key _pdbx_struct_assembly_gen
    # {'assembly_id': ['1', '2', '3', '4'], 'oper_expression': ['1', '1', '1', '1'], 'asym_id_list': ['A,B,C,D,E,F,G,H,I,J,K,L,M,XA,YA,ZA,AB,BB,CB,DB,EB,FB,GB,HB', 'N,O,P,Q,R,S,T,U,V,W,X,Y,IB,JB,KB,LB,MB,NB,OB,PB,QB,RB,SB,TB', 'Z,AA,BA,CA,DA,EA,FA,GA,HA,IA,JA,KA,UB,VB,WB,XB,YB,ZB,AC,BC,CC,DC,EC,FC', 'LA,MA,NA,OA,PA,QA,RA,SA,TA,UA,VA,WA,GC,HC,IC,JC,KC,LC,MC,NC,OC,PC,QC,RC']}
    operations = data.get('_pdbx_struct_oper_list')
    if operations:
        biomt = np.array(
            [operations['matrix[1][1]'], operations['matrix[1][2]'], operations['matrix[1][3]'],
             operations['vector[1]'],
             operations['matrix[2][1]'], operations['matrix[2][2]'], operations['matrix[2][3]'],
             operations['vector[2]'],
             operations['matrix[3][1]'], operations['matrix[3][2]'], operations['matrix[3][3]'],
             operations['vector[3]']], dtype=float)
        # print('biomt', biomt)

        if isinstance(operations['id'], list):
            biomt = biomt.T
        biomt = biomt.reshape(-1, 3, 4).tolist()
        # print('biomt', biomt)
    else:
        biomt = None

    # # Separated...
    # rotations = np.array(
    #     [operations['matrix[1][1]'], operations['matrix[1][2]'], operations['matrix[1][3]'],
    #      operations['matrix[2][1]'], operations['matrix[2][2]'], operations['matrix[2][3]'],
    #      operations['matrix[3][1]'], operations['matrix[3][2]'], operations['matrix[3][3]']], dtype=float)
    # translations = np.array([operations['vector[1]'], operations['vector[2]'], operations['vector[3]']],
    #                         dtype=float)
    # print('rotations', rotations)
    # print('translations', translations)
    # if isinstance(operations['id'], list):
    #     # number_of_operations = len(operations['id'])
    #     rotations = rotations.T  # .reshape(-1, 3, 3)
    #     # rotations = rotations.reshape(-1, number_of_operations)
    #     translations = translations.T
    # # else:
    # #     pass
    # translations = translations.reshape(-1, 3)
    # rotations = rotations.reshape(-1, 3, 3)
    #
    # print('rotations', rotations)
    # print('translations', translations)
    # # biomt = rotations, translations

    # reference_sequence = {chain: reference_sequence}
    formatted_info = dict(
        atoms=atoms,
        biological_assembly=assembly,
        biomt=biomt,
        coords=coords,
        # coords=coords if separate_coords else None,
        cryst_record=cryst_record,
        entity_info=entity_info,
        # header=header,  # Todo
        name=name,
        resolution=resolution,
        # reference_sequence=reference_sequence,
    )
    # Explicitly overwrite any parsing if argument was passed to caller
    formatted_info.update(**kwargs)

    return formatted_info
    # return format_mmcif_dict(data[provided_dataname], name=name, **kwargs)


class Log:
    """Responsible for StructureBase logging operations"""

    def __init__(self, log: Logger | None = logging.getLogger('null')):
        """
        Args:
            log: The logging.Logger to handle StructureBase logging. If None is passed a Logger with NullHandler is used
        """
        self.log = log

    def __copy__(self) -> Log:  # Todo -> Self: in python 3.11
        cls = self.__class__
        other = cls.__new__(cls)
        other.log = self.log
        return other

    copy = __copy__  # Overwrites to use this instance __copy__


null_struct_log = Log()


class SymmetryBase(ABC):
    """Adds functionality for symmetric manipulation of Structure instances"""
    _symmetry: str | None
    _symmetric_dependents: str | None
    """The Structure container where dependent symmetric Structures are contained"""
    symmetry_state_attributes: set[str] = set()

    def __init__(self, rotation_matrices: np.ndarray = None, translation_matrices: np.ndarray = None, **kwargs):
        self._symmetry = self._symmetric_dependents = None
        try:
            super().__init__(**kwargs)  # SymmetryBase
        except TypeError:
            raise TypeError(
                f"The argument(s) passed to the {self.__class__.__name__} instance weren't recognized and aren't "
                f"accepted by the object class: {', '.join(kwargs.keys())}\n\nIt's likely that your class MRO is "
                "insufficient for your passed arguments or you have passed invalid arguments")

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the Structure described by its Schönflies notation"""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, symmetry: str | None):
        try:
            self._symmetry = symmetry.upper()
        except AttributeError:  # Not a string
            if symmetry is None:
                self.reset_symmetry_state()
            else:
                raise ValueError(
                    f"Can't set '.symmetry' with {type(symmetry).__name__}. Must be 'str' or NoneType")
        else:
            if self._symmetry == 'C1':
                self._symmetry = None

    def reset_symmetry_state(self) -> None:
        """Remove any state variable associated with the instance"""
        # self.log.debug(f"Removing symmetric attributes from {repr(self)}")
        for attribute in self.symmetry_state_attributes:
            try:
                delattr(self, attribute)
            except AttributeError:
                continue

    def is_symmetric(self) -> bool:
        """Query whether the Structure is symmetric. Returns True if self.symmetry is not None"""
        return self._symmetry is not None

    def has_symmetric_dependents(self) -> bool:
        """Evaluates to True if the Structure has symmetrically dependent children"""
        return True if self._symmetric_dependents else False

    @property
    def symmetric_dependents(self) -> list[StructureBase] | list:
        """Access the symmetrically dependent Structure instances"""
        return getattr(self, self._symmetric_dependents, [])

    @symmetric_dependents.setter
    def symmetric_dependents(self, symmetric_dependents: str | None):
        """Set the attribute name where dependent Structure instances occupy"""
        try:
            # set() COULD BE USED WHEN MULTIPLE self._symmetric_dependents.add(symmetric_dependents.lower())
            self._symmetric_dependents = symmetric_dependents.lower()
        except AttributeError:  # Not a string
            if symmetric_dependents is None:
                self.reset_symmetry_state()
            else:
                raise ValueError(
                    f"Can't set '.symmetric_dependents' with {type(symmetric_dependents).__name__}. "
                    f"Must be class 'str'")


class StructureMetadata:
    biological_assembly: str | None
    cryst_record: str | None
    entity_info: dict[str, dict[dict | list | str]] | dict
    file_path: AnyStr | None
    header: list
    reference_sequence: str | dict[str, str] = None
    resolution: float | None

    def __init__(self, biological_assembly: str | int = None, cryst_record: str = None,
                 entity_info: dict[str, dict[dict | list | str]] = None, file_path: AnyStr = None,
                 reference_sequence: str | dict[str, str] = None, resolution: float = None, **kwargs):
        """
        Args:
            biological_assembly: The integer of the biological assembly (as indicated by PDB AssemblyID format)
            cryst_record: The string specifying how the molecule is situated in a lattice
            entity_info: A mapping of the metadata to their distinct molecular identifiers
            file_path: The location on disk where the file was accessed
            reference_sequence: The reference sequence (according to expression sequence or reference database)
            resolution: The level of detail available from an experimental dataset contributing to the sharpness with
                which structural data can contribute towards building a model
        """
        if biological_assembly is None:
            self.biological_assembly = biological_assembly
        else:
            self.biological_assembly = str(biological_assembly)
        self.cryst_record = cryst_record
        self.entity_info = entity_info
        self.file_path = file_path
        self.reference_sequence = reference_sequence
        self.resolution = resolution
        # super().__init__(**kwargs)  # StructureMetadata


# parent Structure controls these attributes
parent_variable = '_parent_'
new_parent_attributes = ('_coords', '_log', '_atoms', '_residues')
parent_attributes = (parent_variable, *new_parent_attributes)
"""Holds all the attributes which the parent StructureBase controls including _parent, _coords, _log, _atoms, 
and _residues
"""


class CoordinateOpsMixin(abc.ABC):
    _transforming: bool = False
    """Whether the StructureBase.coords are being updated internally."""

    @property
    @abc.abstractmethod
    def coords(self):
        """"""
    @coords.setter
    @abc.abstractmethod
    def coords(self, value):
        """"""

    def distance_from_reference(
        self, reference: np.ndarray = utils.symmetry.origin, measure: str = 'mean', **kwargs
    ) -> float:
        """From a Structure, find the furthest coordinate from the origin (default) or from a reference.

        Args:
            reference: The reference where the point should be measured from. Default is origin
            measure: The measurement to take with respect to the reference. Could be 'mean', 'min', 'max', or any
                numpy function to describe computed distance scalars
        Returns:
            The distance from the reference point to the furthest point
        """
        if reference is None:
            reference = utils.symmetry.origin

        return getattr(np, measure)(np.linalg.norm(self.coords - reference, axis=1))

    def translate(self, translation: list[float] | np.ndarray, **kwargs) -> None:
        """Perform a translation to the Structure ensuring only the Structure container of interest is translated
        ensuring the underlying coords are not modified

        Args:
            translation: The first translation to apply, expected array shape (3,)
        """
        self._transforming = True
        self.coords += translation
        self._transforming = False

    def rotate(self, rotation: list[list[float]] | np.ndarray, **kwargs) -> None:
        """Perform a rotation to the Structure ensuring only the Structure container of interest is rotated ensuring the
        underlying coords are not modified

        Args:
            rotation: The first rotation to apply, expected array shape (3, 3)
        """
        self._transforming = True
        self.coords = np.matmul(self.coords, np.transpose(rotation))  # Allows a list to be passed
        # self.coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))  # Essentially a transpose
        self._transforming = False

    def transform(
        self, rotation: list[list[float]] | np.ndarray = None, translation: list[float] | np.ndarray = None,
        rotation2: list[list[float]] | np.ndarray = None, translation2: list[float] | np.ndarray = None, **kwargs
    ) -> None:
        """Perform a specific transformation to the Structure ensuring only the Structure container of interest is
        transformed ensuring the underlying coords are not modified

        Transformation proceeds by matrix multiplication and vector addition with the order of operations as:
        rotation, translation, rotation2, translation2

        Args:
            rotation: The first rotation to apply, expected array shape (3, 3)
            translation: The first translation to apply, expected array shape (3,)
            rotation2: The second rotation to apply, expected array shape (3, 3)
            translation2: The second translation to apply, expected array shape (3,)
        """
        if rotation is not None:  # Required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))  # Allows list to be passed...
            # new_coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))  # Essentially a transpose
        else:
            new_coords = self.coords  # No need to copy as this is a view

        if translation is not None:  # Required for np.ndarray or None checks
            new_coords += translation

        if rotation2 is not None:  # Required for np.ndarray or None checks
            np.matmul(new_coords, np.transpose(rotation2), out=new_coords)  # Allows list to be passed...
            # np.matmul(new_coords, rotation2.swapaxes(-2, -1), out=new_coords)  # Essentially a transpose

        if translation2 is not None:  # Required for np.ndarray or None checks
            new_coords += translation2

        self._transforming = True
        self.coords = new_coords
        self._transforming = False

    @abc.abstractmethod
    def copy(self):
        """"""

    def get_transformed_copy(
        self, rotation: list[list[float]] | np.ndarray = None, translation: list[float] | np.ndarray = None,
        rotation2: list[list[float]] | np.ndarray = None, translation2: list[float] | np.ndarray = None
    ) -> StructureBase:  # Todo -> Self in python 3.11
        """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space

        Transformation proceeds by matrix multiplication and vector addition with the order of operations as:
        rotation, translation, rotation2, translation2

        Args:
            rotation: The first rotation to apply, expected array shape (3, 3)
            translation: The first translation to apply, expected array shape (3,)
            rotation2: The second rotation to apply, expected array shape (3, 3)
            translation2: The second translation to apply, expected array shape (3,)
        Returns:
            A transformed copy of the original object
        """
        if rotation is not None:  # Required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))  # Allows list to be passed...
            # new_coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))  # Essentially a transpose
        else:
            new_coords = self.coords  # No need to copy as this is a view

        if translation is not None:  # Required for np.ndarray or None checks
            new_coords += translation

        if rotation2 is not None:  # Required for np.ndarray or None checks
            np.matmul(new_coords, np.transpose(rotation2), out=new_coords)  # Allows list to be passed...
            # np.matmul(new_coords, rotation2.swapaxes(-2, -1), out=new_coords)  # Essentially a transpose

        if translation2 is not None:  # Required for np.ndarray or None checks
            new_coords += translation2

        new_structure = self.copy()
        new_structure._transforming = True
        new_structure.coords = new_coords
        new_structure._transforming = False

        return new_structure


class StructureBase(SymmetryBase, CoordinateOpsMixin, ABC):
    """StructureBase manipulates the Coords and Log instances as well as the atom_indices for a StructureBase.
    Additionally. sorts through parent Structure and dependent Structure hierarchies during Structure subclass creation.
    Collects known keyword arguments for all derived classes calls to protect `object`. Should always be the last class
    in the method resolution order of derived classes.
    """
    _atom_indices: ArrayIndexer = Ellipsis  # slice(None)
    _coords: Coordinates
    _copier: bool = False
    """Whether the StructureBase is being copied by a Container object. If so, cut corners"""
    _dependent_is_updating: bool = False
    """Whether the StructureBase.coords are being updated by a dependent. If so, cut corners"""
    _log: Log
    _parent_is_updating: bool = False
    """Whether the StructureBase.coords are being updated by a parent. If so, cut corners"""
    _parent_: StructureBase | None = None
    state_attributes: set[str] = set()
    ignore_copy_attrs: set[str] = set()

    def __init__(self, parent: StructureBase = None, log: Log | Logger | bool = True,
                 coords: np.ndarray | Coordinates | list[list[float]] = None, name: str = None,
                 # metadata: StructureMetadata = None,
                 file_path: AnyStr = None, biological_assembly: str | int = None,
                 cryst_record: str = None, resolution: float = None,
                 entity_info: dict[str, dict[dict | list | str]] = None,
                 reference_sequence: str | dict[str, str] = None,
                 # entity_info=None,
                 **kwargs):
        # These shouldn't be passed as they should be stripped by prior constructors...
        # entity_names=None, rotation_matrices=None, translation_matrices=None,
        # metadata=None, pose_format=None, query_by_sequence=True, rename_chains=None
        """
        Args:
            parent: If another Structure object created this Structure instance, pass the 'parent' instance. Will take
                ownership over Structure containers (coords, atoms, residues) for dependent Structures
            log: The Log or Logger instance, or the name for the logger to handle parent Structure logging.
                None or False prevents logging while any True assignment enables it
            coords: When setting up a parent Structure instance, the coordinates of that Structure
            name: The identifier for the Structure instance
        """
        self.name = name if name not in [None, False] else f'Unnamed_{self.__class__.__name__}'
        if parent is not None:  # Initialize StructureBase from parent
            self._parent = parent
        else:  # This is the parent
            self._parent_ = None
            self.metadata = StructureMetadata(
                biological_assembly=biological_assembly, cryst_record=cryst_record, entity_info=entity_info,
                file_path=file_path, reference_sequence=reference_sequence, resolution=resolution
            )
            # Initialize Log
            if log:
                if isinstance(log, Log):  # Initialized Log
                    self._log = log
                elif isinstance(log, Logger):  # logging.Logger object
                    self._log = Log(log)
                elif isinstance(log, str):
                    self._log = Log(utils.start_log(name=f'{__name__}.{self.name}', level=log))
                else:  # log is True or some other type:  # Use the module logger
                    self._log = Log(logger)
            else:  # When explicitly passed as None or False, uses the null logger
                self._log = null_struct_log  # Log()

            # Initialize Coordinates
            if coords is None:  # Check this first
                # Most init occurs from Atom instances that are their own parent until another StructureBase adopts them
                self._coords = Coordinates()
            elif isinstance(coords, Coordinates):
                self._coords = coords
            else:  # Create a Coordinates instance. This assumes the dimensions are correct. Coordinates() handles if not
                self._coords = Coordinates(coords)

        # try:
        super().__init__(**kwargs)  # StructureBase
        # except TypeError:
        #     raise TypeError(
        #         f"The argument(s) passed to the {self.__class__.__name__} instance weren't recognized and aren't "
        #         f"accepted by the object class: {', '.join(kwargs.keys())}\n\nIt's likely that your class MRO is "
        #         "insufficient for your passed arguments or you have passed invalid arguments")

    @property
    def parent(self) -> StructureBase | None:
        """Return the instance's 'parent' StructureBase which is responsible for manipulation of Structure containers"""
        return self._parent_

    # This getter is a placeholder so that _parent set automatically handles _log and _coords set in derived classes
    @property
    def _parent(self):
        """The 'parent' StructureBase of this instance. _parent should only be set"""
        raise NotImplementedError('_parent should only be set')

    @_parent.setter
    def _parent(self, parent: StructureBase):
        """Set the 'parent' of this instance"""
        self._parent_ = parent
        self._log = parent._log
        self._coords = parent._coords
        # self._atoms = parent._atoms
        # self._residues = parent._residues

    def is_dependent(self) -> bool:
        """Is this instance a dependent on a parent StructureBase?"""
        return self._parent_ is not None

    def is_parent(self) -> bool:
        """Is this instance a parent?"""
        return self._parent_ is None

    @property
    def log(self) -> Logger:
        """The StructureBase Logger"""
        return self._log.log

    @log.setter
    def log(self, log: Logger | Log):
        """Set the StructureBase to a logging.Logger object"""
        if isinstance(log, Logger):  # Prefer this protection method versus Log.log property overhead?
            self._log.log = log
        elif isinstance(log, Log):
            self._log.log = log.log
        else:
            raise TypeError(
                f"Can't set {Log.__class__.__name__} to {type(log).__name__}. Must be type logging.Logger")

    # @property
    # @abc.abstractmethod
    # def atom_indices(self) -> list[int]:
    #     """The Atoms/Coords indices which the StructureBase has access to"""

    @property
    def coords(self) -> np.ndarray:
        """The coordinates for the Atoms in the StructureBase object"""
        # returns self.Coords.coords(a np.array)[sliced by the instance's atom_indices]
        return self._coords.coords[self._atom_indices]

    @coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        # self.log.debug(f'Setting {self.name} coords')
        # # NOT Setting this first to ensure proper size of later manipulations
        # # Update the whole Coords.coords as symmetry is not everywhere
        # self._coords.replace(self._atom_indices, coords)

        # Check for coordinate update requirements
        if self.is_parent() and self.has_symmetric_dependents() and not self._dependent_is_updating:
            # This Structure is a parent with symmetric dependents, update dependent coords to update the parent
            self.log.debug(f'Updating symmetric dependent coords')
            for dependent in self.symmetric_dependents:
                dependent._parent_is_updating = True
                # self.log.debug(f'Setting {dependent.name} _symmetric_dependent coords')
                dependent.coords = coords[dependent.atom_indices]
                dependent._parent_is_updating = False

        # Setting this after the dependent.coords set, because symmetric dependents use .coords
        # and setting before removes dependent reference
        self._coords.replace(self._atom_indices, coords)

    def make_parent(self):
        """Remove this instance from its parent, making it a parent in the process"""
        # Set parent explicitly as None
        self.__setattr__(parent_variable, None)
        # Create a new, Coords instance detached from the parent
        self._coords = Coordinates(self.coords)

    def reset_state(self):
        """Remove attributes that are valid for the current state

        This is useful for transfer of ownership, or changes in the state that should be overwritten
        """
        for attr in self.state_attributes:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue

        if self.is_symmetric():
            self.reset_symmetry_state()

    # @property
    # @abc.abstractmethod
    # def center_of_mass(self) -> np.ndarray:
    #     """"""

    # @property
    # @abc.abstractmethod
    # def radius(self) -> float:
    #     """"""


    def __copy__(self) -> StructureBase:  # Todo -> Self: in python 3.11
        cls = self.__class__
        other = cls.__new__(cls)
        other__dict__ = other.__dict__
        other__dict__.update(self.__dict__)

        ignore_attrs = (*parent_attributes, *cls.ignore_copy_attrs)
        for attr, value in other__dict__.items():
            if attr not in ignore_attrs:
                other__dict__[attr] = copy(value)
        if self.is_parent():  # This Structure is the parent, it's copy should be too
            # Set the copying Structure attribute .spawn to indicate to dependents their "copies" parent is "other"
            self.spawn = other
            other__dict__[parent_variable] = other  # None
            try:
                for attr in new_parent_attributes:
                    other__dict__[attr] = self.__dict__[attr].copy()
            except KeyError:  # '_atoms', '_residues' may not be present and come after _log, _coords
                pass
            # Remove the attribute spawn after other Structure containers are copied
            del self.spawn
        else:  # This Structure is a dependent
            try:  # If initiated by the parent, this Structure's copy should be a dependent too
                other._parent = self.parent.spawn
            except AttributeError:  # This copy was not initiated by the parent
                if self._copier:  # Copy initiated by StructureBaseContainer
                    pass
                else:
                    other.make_parent()
                    other.log.debug(f"The dependent {repr(other)}'s copy is now a parent")

        return other

    copy = __copy__  # Overwrites to use this instance __copy__

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'


# class Atom(StructureBase):
class Atom(CoordinateOpsMixin):
    """An Atom container with the full Structure coordinates and the Atom unique data"""
    _coords_: list[float]
    _coords: Coordinates
    _copier: bool = False
    """Whether the StructureBase is being copied by a Container object. If so, cut corners"""
    # _sasa: float
    _type_str: str
    index: int | None
    number: int | None
    # type: str | None
    alt_location: str | None
    residue_type: str | None
    chain_id: str | None
    residue_number: int | None
    code_for_insertion: str | None
    occupancy: float | None
    b_factor: float | None
    element: str | None
    charge: str | None
    sasa: float | None

    @classmethod
    def without_coordinates(cls, idx, number, atom_type, alt_location, residue_type, chain_id, residue_number,
                            code_for_insertion, occupancy, b_factor, element, charge):
        """Initialize Atom record data without coordinates. Performs all type casting"""
        return cls(index=idx, number=int(number), atom_type=atom_type, alt_location=alt_location,
                   residue_type=residue_type, chain_id=chain_id, residue_number=int(residue_number),
                   code_for_insertion=code_for_insertion, occupancy=float(occupancy), b_factor=float(b_factor),
                   element=element, charge=charge, coords=[])  # Use a list for speed

    def __init__(self, index: int = None, number: int = None, atom_type: str = None, alt_location: str = ' ',
                 residue_type: str = None, chain_id: str = None, residue_number: int = None,
                 code_for_insertion: str = ' ', x: float = None, y: float = None, z: float = None,
                 occupancy: float = None, b_factor: float = None, element: str = None, charge: str = None,
                 coords: list[float] = None, **kwargs):
        """

        Args:
            index: The zero-indexed number to describe this Atom instance's position in a StructureBaseContainer
            number: The integer number describing this Atom instances position in comparison to others instances
            atom_type: The characters describing the classification of atom via membership to a molecule
            alt_location: Whether the observation of the Atom has alternative evidences
            residue_type: The characters describing the molecular unit to which this Atom belongs
            chain_id: The identifier which signifies this Atom instances membership to a larger polymer
            residue_number: The integer of the polymer that this Atom belongs too
            code_for_insertion: Whether the Atom is included in a residue which is inserted according to a common
                numbering scheme
            x: The x axis value of the instances coordinate position, i.e., it's x coordinate
            y: The y axis value of the instances coordinate position, i.e., it's y coordinate
            z: The z axis value of the instances coordinate position, i.e., it's z coordinate
            occupancy: The fraction of the time this Atom is observed in the experiment
            b_factor: The thermal fluctuation for the Atom OR a value for which structural representation should account
            element: The atomic symbol for the Atom
            charge: The numeric value for the electronic charge of the Atom
            coords: The set of values of the x, y, and z coordinate position
            **kwargs:
        """
        # super().__init__(**kwargs)  # Atom
        self.index = index
        # self._atom_indices = [index]
        self.number = number
        self._type = atom_type
        # Comply with special .pdb formatting syntax by padding type with a space if len(atom_type) == 4
        self._type_str = f'{"" if atom_type[3:] else " "}{atom_type:<3s}'
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.code_for_insertion = code_for_insertion
        if coords is not None:
            self._coords_ = coords
        elif x is not None and y is not None and z is not None:
            self._coords_ = [x, y, z]
        else:
            self._coords_ = []
        self.occupancy = occupancy
        self.b_factor = b_factor
        self.element = element
        self.charge = charge
        self.sasa = None

    @property
    def type(self) -> str:
        """This can't currently be set"""
        # If setting is desired, the self._type_str needs to be changed upon new type
        return self._type

    @property
    def atom_indices(self) -> list[int]:
        """The index of the Atom in the Atoms/Coords container"""
        return [self.index]

    # This getter is a placeholder so that _parent set automatically handles _log and _coords set in derived classes
    @property
    def _parent(self):
        """The 'parent' StructureBase of this instance. _parent should only be set"""
        raise NotImplementedError('_parent should only be set')

    @_parent.setter
    def _parent(self, parent: StructureBase):
        """Set the 'parent' of this instance"""
        self._parent_ = None  # parent
        # self._log = parent._log
        self._coords = parent._coords

    # @staticmethod
    def is_dependent(self) -> bool:
        """Is this instance a dependent on a parent StructureBase?"""
        return True

    # @staticmethod
    def is_parent(self) -> bool:
        """Is this instance a parent?"""
        return False

    @property
    def coords(self) -> np.ndarray:
        """The coordinates for the Atom. Array is 1 dimensional in contrast to other .coords properties"""
        # returns self.Coords.coords(a np.array)[sliced by the instance's atom_indices]
        try:
            return self._coords.coords[self.index]
        except (AttributeError, IndexError):
            # Possibly the Atom was set with keyword argument coords instead of Structure Coords
            # This shouldn't be used often as it will be quite slow... give warning?
            return np.array(self._coords_)

    @coords.setter
    def coords(self, coords: np.ndarray | list[float]):
        try:
            self._coords.replace([self.index], coords)
        except AttributeError:
            if not isinstance(coords, (list, np.ndarray)):
                raise ValueError(
                    f"Can't pass {coords=}. Must be either list[float] or numpy.ndarray"
                )
            self._coords_ = coords

    def make_parent(self):
        """Remove this instance from its parent, making it a parent in the process"""
        # super().make_parent()  # When subclassing StructureBase
        # Create a new, Coords instance detached from the parent
        self._coords = Coordinates(self.coords)
        self.index = 0
        self.reset_state()

    def reset_state(self):
        """Remove attributes that are valid for the current state

        This is useful for transfer of ownership, or changes in the state that should be overwritten
        """
        for attr in self.state_attributes:
            try:
                self.__delattr__(attr)
            except AttributeError:
                continue

    # Below properties are considered part of the Atom state
    # state_attributes = StructureBase.state_attributes | {'sasa'}
    state_attributes = {'sasa'}

    # @property
    # def sasa(self) -> float:
    #     """The Solvent accessible surface area for the Atom. Raises AttributeError if .sasa isn't set"""
    #     # try:  # Let the Residue owner handle errors
    #     return self._sasa
    #     # except AttributeError:
    #     #     raise AttributeError
    #
    # @sasa.setter
    # def sasa(self, sasa: float):
    #     self._sasa = sasa

    # End state properties

    # @property
    # def next_atom(self) -> Atom | None:
    #     """The next Atom in the Structure if this Atom is part of a polymer"""
    #     try:
    #         return self._next_atom
    #     except AttributeError:
    #         return None
    #
    # @next_atom.setter
    # def next_atom(self, other: Atom):
    #     """Set the next_atom for this Atom and the prev_atom for the other Atom"""
    #     self._next_atom = other
    #     other._next_atom = self
    #
    # @property
    # def prev_atom(self) -> Atom | None:
    #     """The next Atom in the Structure if this Atom is part of a polymer"""
    #     try:
    #         return self._prev_atom
    #     except AttributeError:
    #         return None
    #
    # @prev_atom.setter
    # def prev_atom(self, other: Atom):
    #     """Set the prev_atom for this Atom and the next_atom for the other Atom"""
    #     self._prev_atom = other
    #     other._prev_atom = self

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass (the Atom coordinates). Provided for compatibility with StructureBase API"""
        return self.coords

    @property
    def radius(self) -> float:
        """The width of the Atom"""
        raise NotImplementedError("This isn't finished")

    @property
    def x(self) -> float:
        """Access the value for the x coordinate"""
        return self.coords[0]

    @x.setter
    def x(self, x: float):
        """Set the value for the x coordinate"""
        _, y, z = self.coords
        try:
            self._coords.replace([self.index], [x, y, z])
        except AttributeError:  # When _coords not used
            self._coords_ = [x, y, z]

    @property
    def y(self) -> float:
        """Access the value for the y coordinate"""
        return self.coords[1]

    @y.setter
    def y(self, y: float):
        """Set the value for the y coordinate"""
        x, _, z = self.coords
        try:
            self._coords.replace([self.index], [x, y, z])
        except AttributeError:  # When _coords not used
            self._coords_ = [x, y, z]

    @property
    def z(self) -> float:
        """Access the value for the z coordinate"""
        return self.coords[2]

    @z.setter
    def z(self, z: float):
        """Set the value for the z coordinate"""
        x, y, _ = self.coords
        try:
            self._coords.replace([self.index], [x, y, z])
        except AttributeError:  # When _coords not used
            self._coords_ = [x, y, z]

    def is_backbone_and_cb(self) -> bool:
        """Is the Atom is a backbone or CB Atom? Includes N, CA, C, O, and CB"""
        return self._type in protein_backbone_and_cb_atom_types

    def is_backbone(self) -> bool:
        """Is the Atom is a backbone Atom? These include N, CA, C, and O"""
        return self._type in protein_backbone_atom_types

    def is_cb(self, gly_ca: bool = True) -> bool:
        """Is the Atom a CB atom? Default returns True if Glycine and Atom is CA

        Args:
            gly_ca: Whether to include Glycine CA in the boolean evaluation
        """
        if gly_ca:
            return self._type == 'CB' or (self.residue_type == 'GLY' and self._type == 'CA')
        else:
            #                                         When Rosetta assigns, it is this   v, but PDB assigns this v
            return self._type == 'CB' or (self.residue_type == 'GLY' and (self._type == '2HA' or self._type == 'HA3'))

    def is_ca(self) -> bool:
        """Is the Atom a CA atom?"""
        return self._type == 'CA'

    def is_heavy(self) -> bool:
        """Is the Atom a heavy atom?"""
        return 'H' not in self._type

    @property
    def _key(self) -> tuple[int, str, str, float]:
        return self.index, self._type, self.residue_type, self.b_factor

    def get_atom_record(self) -> str:
        """Provide the Atom as an Atom record string

        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        x, y, z = list(self.coords)
        # Add 1 to the self.index since this is 0 indexed
        return f'ATOM  {self.index + 1:5d} {self._type_str}{self.alt_location:1s}{self.residue_type:3s}' \
               f'{self.chain_id:>2s}{self.residue_number:4d}{self.code_for_insertion:1s}   ' \
               f'{x:8.3f}{y:8.3f}{z:8.3f}{self.occupancy:6.2f}{self.b_factor:6.2f}          ' \
               f'{self.element:>2s}{self.charge:2s}'

    def __str__(self) -> str:
        """Represent Atom in PDB format"""
        # Use self.type_str to comply with the PDB format specifications because of the atom type field
        # ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- PDB format
        # Checks if len(atom.type)=4 with slice v. If not insert a space
        return f'ATOM  {"{}"} {self._type_str}{self.alt_location:1s}{"{}"}{"{}"}{"{}"}' \
               f'{self.code_for_insertion:1s}   {"{}"}{self.occupancy:6.2f}{self.b_factor:6.2f}          ' \
               f'{self.element:>2s}{self.charge:2s}'
        # Todo if parent:  # return full ATOM record
        # return 'ATOM  {:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   '\
        #     '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'\
        #     .format(self.index, self.type, self.alt_location, self.residue_type, self.chain_id, self.residue_number,
        #             self.code_for_insertion, *list(self.coords), self.occupancy, self.b_factor, self.element,
        #             self.charge)

    def __eq__(self, other: Atom) -> bool:
        if isinstance(other, Atom):
            return self._key == other._key
        raise NotImplementedError(
            f"Can't compare {self.__class__.__name__} instance to {type(other).__name__} instance")

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return f'{Atom.__name__}({self._type} at {self.coords.tolist()})'

    def __copy__(self) -> Atom:  # Todo -> Self: in python 3.11
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        if self._copier:  # Copy initiated by Atoms container
            pass
        else:
            other.make_parent()

        return other

    copy = __copy__  # Overwrites to use this instance __copy__

_StructType = TypeVar('_StructType')


class StructureBaseContainer(Generic[_StructType]):

    def __init__(self, structs: Sequence[StructureBase] | np.ndarray = None):
        """

        Args:
            structs: The StructureBase instances to store. Should be a homogeneous Sequence
        """
        if structs is None:
            structs = []
        elif not isinstance(structs, (np.ndarray, list)):
            structs = list(structs)
            # raise TypeError(
            #     f"Can't initialize {self.__class__.__name__} with {type(structs).__name__}. Type must be a "
            #     f'numpy.ndarray or list[{StructureBase.__name__}]')

        self.structs = np.array(structs, dtype=np.object_)

    def are_dependents(self) -> bool:
        """Check if any of the StructureBase instances are dependents of another StructureBase"""
        for struct in self:
            if struct.is_dependent():
                return True
        return False

    def append(self, new_structures: list[_StructType] | np.ndarray):
        """Append additional StructureBase instances to the StructureBaseContainer

        Args:
            new_structures: The Structure instances to append
        Sets:
            self.structs = numpy.concatenate((self.residues, new_structures))
        """
        self.structs = np.concatenate((self.structs, new_structures))

    def delete(self, indices: Sequence[int]):
        """Delete StructureBase instances from the StructureBaseContainer

        Args:
            indices: The indices to delete
        Sets:
            self.structs = numpy.delete(self.structs, indices)
        """
        self.structs = np.delete(self.structs, indices)

    def insert(self, at: int, new_structures: list[_StructType] | np.ndarray):
        """Insert StructureBase instances into the StructureBaseContainer

        Args:
            at: The index to perform the insert at
            new_structures: The Structure instances to insert
        """
        self.structs = np.concatenate((
            self.structs[:at],
            new_structures if isinstance(new_structures, Iterable) else [new_structures],
            self.structs[at:]
        ))

    def set(self, new_structures: list[_StructType] | np.ndarray):
        """Set the StructureBaseContainer with new StructureBase instances

        Args:
            new_structures: The new instances which should make up the container
        Sets:
            self.structs = numpy.array(new_instances)
        """
        self.structs = np.array(new_structures)
        self.reindex()

    @property
    @abc.abstractmethod
    def reindex(self):
        """"""

    def reset_state(self):
        """Remove any attributes from the Structure instances that are part of the current state

        This is useful for transfer of ownership, or changes in the state that need to be overwritten
        """
        for struct in self:
            struct.reset_state()

    def set_attributes(self, **kwargs):
        """Set Structure attributes passed by keyword to their corresponding value"""
        for struct in self:
            for key, value in kwargs.items():
                setattr(struct, key, value)

    # def set_attribute_from_array(self, **kwargs):  # UNUSED
    #     """For each Residue, set the attribute passed by keyword to the attribute corresponding to the Residue index in
    #     a provided array
    #
    #     Ex: residues.attribute_from_array(mutation_rate=residue_mutation_rate_array)
    #     """
    #     for idx, residue in enumerate(self):
    #         for key, value in kwargs.items():
    #             setattr(residue, key, value[idx])

    def __copy__(self) -> StructureBaseContainer:  # Todo -> Self: in python 3.11
        cls = self.__class__
        other = cls.__new__(cls)

        other_structs = [None for _ in range(len(self))]
        for idx, struct in enumerate(self):
            # Set an attribute to indicate the struct shouldn't be "detached"
            # since a Structure owns this Structures instance
            struct._copier = True
            other_structs[idx] = new_struct = struct.copy()
            new_struct._copier = struct._copier = False

        other.structs = np.array(other_structs, dtype=np.object_)

        return other

    copy = __copy__  # Overwrites to use this instance __copy__

    def __getitem__(self, item: ArrayIndexer) -> _StructType | list[_StructType]:  # StructureBaseContainer[_StructType]:
        items = self.structs[item]
        if isinstance(item, int):
            return items
        else:  # Convert numpy to list
            return items.tolist()

    def __len__(self) -> int:
        return len(self.structs)

    def __iter__(self) -> Generator[_StructType, None, None]:
        yield from self.structs.tolist()


class Atoms(StructureBaseContainer):

    def reindex(self, start_at: int = 0):
        """Set each Atom instance index according to incremental Atoms/Coords index

        Args:
            start_at: The index to start reindexing at. Must be [0, 'inf']
        """
        if start_at == 0:
            _start_at = start_at
        else:
            _start_at = start_at - 1
            if start_at < 0:
                _start_at += len(self)
        try:
            prior_struct, *other_structs = self.structs[_start_at:]
        except ValueError:  # Not enough values to unpack as the index didn't slice anything
            raise IndexError(
                f'{self.reindex.__name__}: {start_at=} is outside of the {self.__class__.__name__} indices with '
                f'size {len(self)}')
        else:
            if start_at == 0:
                prior_struct.index = 0

            for idx, struct in enumerate(other_structs, prior_struct.index + 1):
                struct.index = idx

    def __copy__(self) -> Atoms:  # Todo -> Self: in python 3.11
        return super().__copy__()

    copy = __copy__  # Overwrites to use this instance __copy__


# Define a ideal, 15 residue alpha helix
alpha_helix_15_atoms = Atoms([
    Atom(0, 1, 'N', ' ', 'ALA', 'A', 1, ' ', 27.128, 20.897, 37.943, 1.00, 0.00, 'N', ''),
    Atom(1, 2, 'CA', ' ', 'ALA', 'A', 1, ' ', 27.933, 21.940, 38.546, 1.00, 0.00, 'C', ''),
    Atom(2, 3, 'C', ' ', 'ALA', 'A', 1, ' ', 28.402, 22.920, 37.481, 1.00, 0.00, 'C', ''),
    Atom(3, 4, 'O', ' ', 'ALA', 'A', 1, ' ', 28.303, 24.132, 37.663, 1.00, 0.00, 'O', ''),
    Atom(4, 5, 'CB', ' ', 'ALA', 'A', 1, ' ', 29.162, 21.356, 39.234, 1.00, 0.00, 'C', ''),
    Atom(5, 6, 'N', ' ', 'ALA', 'A', 2, ' ', 28.914, 22.392, 36.367, 1.00, 0.00, 'N', ''),
    Atom(6, 7, 'CA', ' ', 'ALA', 'A', 2, ' ', 29.395, 23.219, 35.278, 1.00, 0.00, 'C', ''),
    Atom(7, 8, 'C', ' ', 'ALA', 'A', 2, ' ', 28.286, 24.142, 34.793, 1.00, 0.00, 'C', ''),
    Atom(8, 9, 'O', ' ', 'ALA', 'A', 2, ' ', 28.508, 25.337, 34.610, 1.00, 0.00, 'O', ''),
    Atom(9, 10, 'CB', ' ', 'ALA', 'A', 2, ' ', 29.857, 22.365, 34.102, 1.00, 0.00, 'C', ''),
    Atom(10, 11, 'N', ' ', 'ALA', 'A', 3, ' ', 27.092, 23.583, 34.584, 1.00, 0.00, 'N', ''),
    Atom(11, 12, 'CA', ' ', 'ALA', 'A', 3, ' ', 25.956, 24.355, 34.121, 1.00, 0.00, 'C', ''),
    Atom(12, 13, 'C', ' ', 'ALA', 'A', 3, ' ', 25.681, 25.505, 35.079, 1.00, 0.00, 'C', ''),
    Atom(13, 14, 'O', ' ', 'ALA', 'A', 3, ' ', 25.488, 26.639, 34.648, 1.00, 0.00, 'O', ''),
    Atom(14, 15, 'CB', ' ', 'ALA', 'A', 3, ' ', 24.703, 23.490, 34.038, 1.00, 0.00, 'C', ''),
    Atom(15, 16, 'N', ' ', 'ALA', 'A', 4, ' ', 25.662, 25.208, 36.380, 1.00, 0.00, 'N', ''),
    Atom(16, 17, 'CA', ' ', 'ALA', 'A', 4, ' ', 25.411, 26.214, 37.393, 1.00, 0.00, 'C', ''),
    Atom(17, 18, 'C', ' ', 'ALA', 'A', 4, ' ', 26.424, 27.344, 37.270, 1.00, 0.00, 'C', ''),
    Atom(18, 19, 'O', ' ', 'ALA', 'A', 4, ' ', 26.055, 28.516, 37.290, 1.00, 0.00, 'O', ''),
    Atom(19, 20, 'CB', ' ', 'ALA', 'A', 4, ' ', 25.519, 25.624, 38.794, 1.00, 0.00, 'C', ''),
    Atom(20, 21, 'N', ' ', 'ALA', 'A', 5, ' ', 27.704, 26.987, 37.142, 1.00, 0.00, 'N', ''),
    Atom(21, 22, 'CA', ' ', 'ALA', 'A', 5, ' ', 28.764, 27.968, 37.016, 1.00, 0.00, 'C', ''),
    Atom(22, 23, 'C', ' ', 'ALA', 'A', 5, ' ', 28.497, 28.876, 35.825, 1.00, 0.00, 'C', ''),
    Atom(23, 24, 'O', ' ', 'ALA', 'A', 5, ' ', 28.602, 30.096, 35.937, 1.00, 0.00, 'O', ''),
    Atom(24, 25, 'CB', ' ', 'ALA', 'A', 5, ' ', 30.115, 27.292, 36.812, 1.00, 0.00, 'C', ''),
    Atom(25, 26, 'N', ' ', 'ALA', 'A', 6, ' ', 28.151, 28.278, 34.682, 1.00, 0.00, 'N', ''),
    Atom(26, 27, 'CA', ' ', 'ALA', 'A', 6, ' ', 27.871, 29.032, 33.478, 1.00, 0.00, 'C', ''),
    Atom(27, 28, 'C', ' ', 'ALA', 'A', 6, ' ', 26.759, 30.040, 33.737, 1.00, 0.00, 'C', ''),
    Atom(28, 29, 'O', ' ', 'ALA', 'A', 6, ' ', 26.876, 31.205, 33.367, 1.00, 0.00, 'O', ''),
    Atom(29, 30, 'CB', ' ', 'ALA', 'A', 6, ' ', 27.429, 28.113, 32.344, 1.00, 0.00, 'C', ''),
    Atom(30, 31, 'N', ' ', 'ALA', 'A', 7, ' ', 25.678, 29.586, 34.376, 1.00, 0.00, 'N', ''),
    Atom(31, 32, 'CA', ' ', 'ALA', 'A', 7, ' ', 24.552, 30.444, 34.682, 1.00, 0.00, 'C', ''),
    Atom(32, 33, 'C', ' ', 'ALA', 'A', 7, ' ', 25.013, 31.637, 35.507, 1.00, 0.00, 'C', ''),
    Atom(33, 34, 'O', ' ', 'ALA', 'A', 7, ' ', 24.652, 32.773, 35.212, 1.00, 0.00, 'O', ''),
    Atom(34, 35, 'CB', ' ', 'ALA', 'A', 7, ' ', 23.489, 29.693, 35.478, 1.00, 0.00, 'C', ''),
    Atom(35, 36, 'N', ' ', 'ALA', 'A', 8, ' ', 25.814, 31.374, 36.543, 1.00, 0.00, 'N', ''),
    Atom(36, 37, 'CA', ' ', 'ALA', 'A', 8, ' ', 26.321, 32.423, 37.405, 1.00, 0.00, 'C', ''),
    Atom(37, 38, 'C', ' ', 'ALA', 'A', 8, ' ', 27.081, 33.454, 36.583, 1.00, 0.00, 'C', ''),
    Atom(38, 39, 'O', ' ', 'ALA', 'A', 8, ' ', 26.874, 34.654, 36.745, 1.00, 0.00, 'O', ''),
    Atom(39, 40, 'CB', ' ', 'ALA', 'A', 8, ' ', 27.193, 31.773, 38.487, 1.00, 0.00, 'C', ''),
    Atom(40, 41, 'N', ' ', 'ALA', 'A', 9, ' ', 27.963, 32.980, 35.700, 1.00, 0.00, 'N', ''),
    Atom(41, 42, 'CA', ' ', 'ALA', 'A', 9, ' ', 28.750, 33.859, 34.858, 1.00, 0.00, 'C', ''),
    Atom(42, 43, 'C', ' ', 'ALA', 'A', 9, ' ', 27.834, 34.759, 34.042, 1.00, 0.00, 'C', ''),
    Atom(43, 44, 'O', ' ', 'ALA', 'A', 9, ' ', 28.052, 35.967, 33.969, 1.00, 0.00, 'O', ''),
    Atom(44, 45, 'CB', ' ', 'ALA', 'A', 9, ' ', 29.621, 33.061, 33.894, 1.00, 0.00, 'C', ''),
    Atom(45, 46, 'N', ' ', 'ALA', 'A', 10, ' ', 26.807, 34.168, 33.427, 1.00, 0.00, 'N', ''),
    Atom(46, 47, 'CA', ' ', 'ALA', 'A', 10, ' ', 25.864, 34.915, 32.620, 1.00, 0.00, 'C', ''),
    Atom(47, 48, 'C', ' ', 'ALA', 'A', 10, ' ', 25.230, 36.024, 33.448, 1.00, 0.00, 'C', ''),
    Atom(48, 49, 'O', ' ', 'ALA', 'A', 10, ' ', 25.146, 37.165, 33.001, 1.00, 0.00, 'O', ''),
    Atom(49, 50, 'CB', ' ', 'ALA', 'A', 10, ' ', 24.752, 34.012, 32.097, 1.00, 0.00, 'C', ''),
    Atom(50, 51, 'N', ' ', 'ALA', 'A', 11, ' ', 24.783, 35.683, 34.660, 1.00, 0.00, 'N', ''),
    Atom(51, 52, 'CA', ' ', 'ALA', 'A', 11, ' ', 24.160, 36.646, 35.544, 1.00, 0.00, 'C', ''),
    Atom(52, 53, 'C', ' ', 'ALA', 'A', 11, ' ', 25.104, 37.812, 35.797, 1.00, 0.00, 'C', ''),
    Atom(53, 54, 'O', ' ', 'ALA', 'A', 11, ' ', 24.699, 38.970, 35.714, 1.00, 0.00, 'O', ''),
    Atom(54, 55, 'CB', ' ', 'ALA', 'A', 11, ' ', 23.810, 36.012, 36.887, 1.00, 0.00, 'C', ''),
    Atom(55, 56, 'N', ' ', 'ALA', 'A', 12, ' ', 26.365, 37.503, 36.107, 1.00, 0.00, 'N', ''),
    Atom(56, 57, 'CA', ' ', 'ALA', 'A', 12, ' ', 27.361, 38.522, 36.370, 1.00, 0.00, 'C', ''),
    Atom(57, 58, 'C', ' ', 'ALA', 'A', 12, ' ', 27.477, 39.461, 35.177, 1.00, 0.00, 'C', ''),
    Atom(58, 59, 'O', ' ', 'ALA', 'A', 12, ' ', 27.485, 40.679, 35.342, 1.00, 0.00, 'O', ''),
    Atom(59, 60, 'CB', ' ', 'ALA', 'A', 12, ' ', 28.730, 37.900, 36.625, 1.00, 0.00, 'C', ''),
    Atom(60, 61, 'N', ' ', 'ALA', 'A', 13, ' ', 27.566, 38.890, 33.974, 1.00, 0.00, 'N', ''),
    Atom(61, 62, 'CA', ' ', 'ALA', 'A', 13, ' ', 27.680, 39.674, 32.761, 1.00, 0.00, 'C', ''),
    Atom(62, 63, 'C', ' ', 'ALA', 'A', 13, ' ', 26.504, 40.634, 32.645, 1.00, 0.00, 'C', ''),
    Atom(63, 64, 'O', ' ', 'ALA', 'A', 13, ' ', 26.690, 41.815, 32.360, 1.00, 0.00, 'O', ''),
    Atom(64, 65, 'CB', ' ', 'ALA', 'A', 13, ' ', 27.690, 38.779, 31.527, 1.00, 0.00, 'C', ''),
    Atom(65, 66, 'N', ' ', 'ALA', 'A', 14, ' ', 25.291, 40.121, 32.868, 1.00, 0.00, 'N', ''),
    Atom(66, 67, 'CA', ' ', 'ALA', 'A', 14, ' ', 24.093, 40.932, 32.789, 1.00, 0.00, 'C', ''),
    Atom(67, 68, 'C', ' ', 'ALA', 'A', 14, ' ', 24.193, 42.112, 33.745, 1.00, 0.00, 'C', ''),
    Atom(68, 69, 'O', ' ', 'ALA', 'A', 14, ' ', 23.905, 43.245, 33.367, 1.00, 0.00, 'O', ''),
    Atom(69, 70, 'CB', ' ', 'ALA', 'A', 14, ' ', 22.856, 40.120, 33.158, 1.00, 0.00, 'C', ''),
    Atom(70, 71, 'N', ' ', 'ALA', 'A', 15, ' ', 24.604, 41.841, 34.986, 1.00, 0.00, 'N', ''),
    Atom(71, 72, 'CA', ' ', 'ALA', 'A', 15, ' ', 24.742, 42.878, 35.989, 1.00, 0.00, 'C', ''),
    Atom(72, 73, 'C', ' ', 'ALA', 'A', 15, ' ', 25.691, 43.960, 35.497, 1.00, 0.00, 'C', ''),
    Atom(73, 74, 'O', ' ', 'ALA', 'A', 15, ' ', 25.390, 45.147, 35.602, 1.00, 0.00, 'O', ''),
    Atom(74, 75, 'CB', ' ', 'ALA', 'A', 15, ' ', 24.418, 41.969, 34.808, 1.00, 0.00, 'C', '')])
"""
Caution. After the first time this helix is created as a Structure, the alpha_helix_15_atoms Atom instances will have 
their .coords set to the alpha_helix_15 'parent' Structure instance. Transformation of this Structure causes the 
starting coordinates in each Atom instance in alpha_helix_15_atoms to be transformed as well. Making new structures 
from this Atoms container is sufficient to detach them, but the coordinates are moved from program start after usage the
first time. This behavior isn't ideal, but a consequence of construction of Atom instances from the singular 
attribute basis with the intention to be reused
"""
# These were pulled directly from alphafold and violates DRY
# This is at the expense of working through import issues (they may not exist)
# This mapping is used to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_order)
# END DRY violation


class ContainsAtoms(StructureBase, ABC):
    _atoms: Atoms
    _inverse_number_atoms: np.ndarray
    _indices_attributes: set[str] = {
        '_backbone_and_cb_indices', '_backbone_indices', '_ca_indices', '_cb_indices', '_heavy_indices',
        '_side_chain_indices'}
    # These state_attributes are used by all subclasses
    state_attributes = StructureBase.state_attributes | _indices_attributes | {'_inverse_number_atoms'}

    @classmethod
    def from_atoms(cls, atoms: list[Atom] | Atoms = None, **kwargs):
        return cls(atoms=atoms, **kwargs)

    def __init__(self, atoms: list[Atom] | Atoms = None, atom_indices: list[int] = None, **kwargs):
        """
        Args:
            atoms: Atom instances to initialize the instance
        """
        super().__init__(**kwargs)  # ContainsAtoms
        if self.is_parent():
            if atoms is not None:
                self._assign_atoms(atoms)
            else:  # Create an empty container
                # try:
                #     self._atoms
                # except AttributeError:
                self._atoms = Atoms()
        elif atom_indices is not None:
            try:
                atom_indices[0]
            except (TypeError, IndexError):
                raise ValueError(
                    f"The {self.__class__.__name__} wasn't passed 'atom_indices' which are required for initialization")

            if not isinstance(atom_indices, list):
                atom_indices = list(atom_indices)
            self._atom_indices = atom_indices

    @StructureBase._parent.setter
    def _parent(self, parent: StructureBase):
        """Set the 'parent' StructureBase of this instance"""
        # Had to call _StructureBase__parent to access this variable given issue with inheritance
        super(ContainsAtoms, ContainsAtoms)._parent.fset(self, parent)
        self._atoms = parent._atoms

    def make_parent(self):
        """Remove this instance from its parent, making it a parent in the process"""
        super().make_parent()
        # Populate the Structure with its existing instances removed of any indexing
        self._assign_atoms(self.atoms)
        self.reset_state()

    def get_base_containers(self) -> dict[str, Any]:
        """Returns the instance structural containers as a dictionary with attribute as key and container as value"""
        return dict(coords=self._coords, atoms=self._atoms)

    def _assign_atoms(self, atoms: Atoms | list[Atom], atoms_only: bool = True, **kwargs):
        """Assign Atom instances to the StructureBase, create Atoms object

        Args:
            atoms: The Atom instances to assign to the StructureBase
            atoms_only: Whether Atom instances are being assigned on their own.
                If False, atoms won't become dependents of this instance until specifically called using
                Atoms.set_attributes(_parent=self)
        Keyword Args:
            coords: numpy.ndarray = None - The coordinates to assign to the StructureBase.
                Optional, will use a .coords attribute from Atoms container if not specified
        Sets:
            self._atom_indices (list[int])

            self._atoms (Atoms)
        """
        # Set proper atoms attributes
        self._atom_indices = list(range(len(atoms)))
        if not isinstance(atoms, Atoms):  # Must create the Atoms object
            atoms = Atoms(atoms)

        if atoms.are_dependents():  # Copy Atoms object to set new attributes on each member Atom
            atoms = atoms.copy()
            atoms.reset_state()  # Clear runtime attributes
        self._atoms = atoms
        self.renumber_atoms()

        if atoms_only:
            self._populate_coords(**kwargs)  # Coords may be passed
            # self._create_residues()
            # Ensure that coordinate lengths match atoms
            self._validate_coords()
            # Update Atom instance attributes to ensure they are dependants of this instance
            # Must do this after _populate_coords to ensure that coordinate info isn't overwritten
            self._atoms.set_attributes(_parent=self)
            self._atoms.reindex()

    def _populate_coords(self, coords: np.ndarray = None, from_source: structure_container_types = 'atoms'):
        """Set up the coordinates, initializing them from_source coords if none are set

        Only useful if the calling StructureBase is a parent, and coordinate initialization has yet to occur

        Args:
            coords: The coordinates to assign to the StructureBase. Will use from_source.coords if not specified
            from_source: The source to set the coordinates from if they are missing
        """
        if coords is not None:
            # Try to set the provided coords. This will handle issue where empty
            # Coords class should be set. Setting .coords through normal mechanism
            # preserves subclasses requirement to handle symmetric coordinates.
            self.coords = np.concatenate(coords)

        # Check if _coords (Coords) has been populated
        if len(self._coords.coords) == 0:
            # If it hasn't, then coords weren't passed.
            # Try to set from self.from_source and catch missing 'from_source'.
            try:  # Probably missing from_source. .coords is available in all structure_container_types...
                source_coordinate_container = getattr(self, from_source)
            except AttributeError:
                raise AttributeError(
                    f"'{from_source}' aren't available for the {repr(self)} instance")

            try:
                coords = np.concatenate([container.coords for container in source_coordinate_container])
            except AttributeError:
                raise AttributeError(
                    f"Missing '.coords' attribute for the {repr(self)} container attribute '{from_source}'. "
                    f"This isn't supposed to happen. Is '{from_source}' a Structure container?")
            else:
                if from_source == 'atoms':
                    coords = coords.reshape(-1, 3)

            self._coords.set(coords)

    def _validate_coords(self):
        """Ensure that the StructureBase coordinates are formatted correctly"""
        # This is a crucial functionality for when making a valid new Structure
        if self.number_of_atoms != len(self.coords):
            # .number_of_atoms was typically just set by self._atom_indices
            raise ValueError(
                f'The number of Atoms, {self.number_of_atoms} != {len(self.coords)}, the number of Coords. Consider '
                f"initializing {self.__class__.__name__} without explicitly passing coords if this wasn't expected")

    @property
    def atoms(self) -> list[Atom] | None:
        """Return the Atom instances in the StructureBase"""
        try:
            return self._atoms[self._atom_indices]
        except AttributeError:  # When self._atoms isn't set or is None and doesn't have .atoms
            return None

    @property
    def atom_indices(self) -> list[int]:
        """The Atoms/Coords indices which the StructureBase has access to"""
        return self._atom_indices

    @property
    def number_of_atoms(self) -> int:
        """The number of atoms/coordinates in the StructureBase"""
        try:
            return len(self._atom_indices)
        except TypeError:
            return 0

    def neighboring_atom_indices(self, distance: float = 8., **kwargs) -> list[int]:  # np.ndarray:
        """Returns the Atom instances in the Structure

        Args:
            distance: The distance to measure neighbors by
        Returns:

        """
        parent_coords = self._coords.coords
        atom_indices = self.atom_indices

        # Create a "self.coords" and modify only coordinates in the ContainsAtoms instance
        modified_self_coords = parent_coords.copy()
        # Translate each self coordinate
        modified_self_coords[atom_indices] = parent_coords.max(axis=0) + [1000, 1000, 1000]
        modified_coords_balltree = BallTree(modified_self_coords)

        # Query for neighbors of the self coordinates but excluding the self indices
        coords = parent_coords[atom_indices]
        query = modified_coords_balltree.query_radius(coords, distance)

        return sorted({idx for contacts in query.tolist() for idx in contacts.tolist()})
        # return np.unique(np.concatenate(query)).tolist()

    # @atoms.setter
    # def atoms(self, atoms: Atoms | list[Atom]):
    #     """Set the Structure atoms to an Atoms object"""
    #     # Todo make this setter function in the same way as self._coords.replace?
    #     if isinstance(atoms, Atoms):
    #         self._atoms = atoms
    #     else:
    #         self._atoms = Atoms(atoms)

    # # Todo enable this type of functionality
    # @atoms.setter
    # def atoms(self, atoms: Atoms):
    #     self._atoms.replace(self._atom_indices, atoms)

    # # Todo create add_atoms that is like list append
    # def add_atoms(self, atom_list):
    #     """Add Atoms in atom_list to the Structure instance"""
    #     raise NotImplementedError('This function (add_atoms) is currently broken')
    #     atoms = self.atoms.tolist()
    #     atoms.extend(atom_list)
    #     self.atoms = atoms
    #     # Todo need to update all referrers
    #     # Todo need to add the atoms to coords

    @property
    @abc.abstractmethod
    def backbone_and_cb_indices(self) -> list[int]:
        """The indices that index the StructureBase backbone and CB Atoms/Coords"""

    @property
    def number_of_backbone_and_cb_atoms(self) -> int:
        return len(self._backbone_and_cb_indices)

    @property
    @abc.abstractmethod
    def backbone_indices(self) -> list[int]:
        """The indices that index the StructureBase backbone and CB Atoms/Coords"""

    @property
    def number_of_backbone_atoms(self) -> int:
        return len(self._backbone_indices)

    @property
    @abc.abstractmethod
    def ca_indices(self) -> list[int]:
        """The indices that index the StructureBase CA Atoms/Coords"""

    @property
    def number_of_ca_atoms(self) -> int:
        return len(self._ca_indices)

    @property
    @abc.abstractmethod
    def cb_indices(self) -> list[int]:
        """The indices that index the StructureBase CB Atoms/Coords"""

    @property
    def number_of_cb_atoms(self) -> int:
        return len(self._cb_indices)

    @property
    @abc.abstractmethod
    def heavy_indices(self) -> list[int]:
        """The indices that index the StructureBase heavy (non-hydrogen) Atoms/Coords"""

    @property
    def number_of_heavy_atoms(self) -> int:
        return len(self._heavy_indices)

    @property
    @abc.abstractmethod
    def side_chain_indices(self) -> list[int]:
        """The indices that index the StructureBase side-chain Atoms/Coords"""

    @property
    def number_of_side_chain_atoms(self) -> int:
        return len(self._side_chain_indices)

    @property
    def backbone_atoms(self) -> list[Atom]:
        """Returns backbone Atom instances from the StructureBase"""
        return self._atoms[self.backbone_indices]

    @property
    def backbone_and_cb_atoms(self) -> list[Atom]:
        """Returns backbone and CB Atom instances from the StructureBase"""
        return self._atoms[self.backbone_and_cb_indices]

    @property
    def ca_atoms(self) -> list[Atom]:
        """Returns CA Atom instances from the StructureBase"""
        return self._atoms[self.ca_indices]

    @property
    def cb_atoms(self) -> list[Atom]:
        """Returns CB Atom instances from the StructureBase"""
        return self._atoms[self.cb_indices]

    @property
    def heavy_atoms(self) -> list[Atom]:
        """Returns heavy Atom instances from the StructureBase"""
        return self._atoms[self.heavy_indices]

    @property
    def side_chain_atoms(self) -> list[Atom]:
        """Returns side chain Atom instances from the StructureBase"""
        return self._atoms[self.side_chain_indices]

    def atom(self, atom_number: int) -> Atom | None:
        """Returns the Atom specified by atom number if a matching Atom is found, otherwise None"""
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass for the Atom coordinates"""
        try:
            return np.matmul(self._inverse_number_atoms, self.coords)
        except AttributeError:
            number_of_atoms = self.number_of_atoms
            self._inverse_number_atoms = np.full(number_of_atoms, 1 / number_of_atoms)
            return np.matmul(self._inverse_number_atoms, self.coords)

    @property
    def radius(self) -> float:
        """The furthest point from the center of mass of the StructureBase"""
        return np.max(np.linalg.norm(self.coords - self.center_of_mass, axis=1))

    @property
    def radius_of_gyration(self) -> float:
        """The measurement of the implied radius (Angstroms) affecting how the StructureBase diffuses through solution

        Satisfies the equation:
            Rg = SQRT(SUM|i->N(Ri**2)/N)
        Where:
            - Ri is the radius of the point i from the center of mass point
            - N is the total number of points
        """
        return np.sqrt(np.mean(np.linalg.norm(self.coords - self.center_of_mass, axis=1) ** 2))

    @property
    def backbone_coords(self) -> np.ndarray:
        """Return a view of the Coords from the StructureBase with backbone atom coordinates"""
        return self._coords.coords[self.backbone_indices]

    @property
    def backbone_and_cb_coords(self) -> np.ndarray:
        """Return a view of the Coords from the StructureBase with backbone and CB atom coordinates. Includes glycine CA
        """
        return self._coords.coords[self.backbone_and_cb_indices]

    @property
    def ca_coords(self) -> np.ndarray:
        """Return a view of the Coords from the Structure with CA atom coordinates"""
        return self._coords.coords[self.ca_indices]

    @property
    def cb_coords(self) -> np.ndarray:
        """Return a view of the Coords from the Structure with CB atom coordinates"""
        return self._coords.coords[self.cb_indices]

    @property
    def heavy_coords(self) -> np.ndarray:
        """Return a view of the Coords from the StructureBase with heavy atom coordinates"""
        return self._coords.coords[self.heavy_indices]

    @property
    def side_chain_coords(self) -> np.ndarray:
        """Return a view of the Coords from the StructureBase with side chain atom coordinates"""
        return self._coords.coords[self.side_chain_indices]

    def renumber_atoms(self, at: int = 1):
        """Renumber all Atom objects sequentially starting with 1

        Args:
            at: The number to start renumbering at
        """
        for idx, atom in enumerate(self.atoms, at):
            atom.number = idx

    @property
    def start_index(self) -> int:
        """The first atomic index of the StructureBase"""
        return self._atom_indices[0]

    @property
    def end_index(self) -> int:
        """The last atomic index of the StructureBase"""
        return self._atom_indices[-1]

    def reset_indices(self):
        """Reset the indices attached to the instance"""
        for attr in self._indices_attributes:
            try:
                delattr(self, attr)
            except AttributeError:
                continue

    def format_header(self, **kwargs) -> str:
        """Returns the base .pdb formatted header

        Returns:
            The .pdb file header string
        """
        # XXXX should be substituted for the PDB code. Ex:
        # f'HEADER    VIRAL PROTEIN                           28-MAY-21   7OP2              \n'
        # f'HEADER    VIRAL PROTEIN/DE NOVO PROTEIN           11-MAY-12   4ATZ              \n'
        return \
            f'HEADER    {self.name[:40]:<40s}{utils.short_start_date.upper():<12s}{"XXXX":<18s}\n' \
            'EXPDTA    THEORETICAL MODEL                                                     \n' \
            'REMARK 220                                                                      \n' \
            'REMARK 220 EXPERIMENTAL DETAILS                                                 \n' \
            'REMARK 220  EXPERIMENT TYPE                : THEORETICAL MODELLING              \n' \
            f'REMARK 220  DATE OF DATA COLLECTION        : {utils.long_start_date:<35s}\n' \
            f'REMARK 220 REMARK: MODEL GENERATED BY {putils.program_name.upper():<50s}\n' \
            f'REMARK 220         VERSION {putils.commit_short:<61s}\n'

    @abc.abstractmethod
    def get_atom_record(self, **kwargs) -> str:
        """Provide the Structure Atom instances as a .pdb file string

        Keyword Args:
            chain_id: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """

    def write(
        self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None, **kwargs
    ) -> AnyStr | None:
        """Write Atom instances to a file specified by out_path or with a passed file_handle

        If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        Keyword Args
            chain_id: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom index by. Default uses one-indexed atom numbers
        Returns:
            The name of the written file if out_path is used
        """
        if file_handle:
            file_handle.write(f'{self.get_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.format_header()
            if header is not None:
                if not isinstance(header, str):
                    header = str(header)
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.get_atom_record(**kwargs)}\n')
            return out_path

    def get_atoms(self, numbers: Container = None, **kwargs) -> list[Atom]:
        """Retrieves Atom instances. Returns all by default unless a list of numbers is specified

        Args:
            numbers: The Atom numbers of interest
        Returns:
            The requested Atom objects
        """
        if numbers is not None:
            if isinstance(numbers, Container):
                return [atom for atom in self.atoms if atom.number in numbers]
            else:
                self.log.error(f'The passed numbers type "{type(numbers).__name__}" must be a Container. Returning'
                               f' all Atom instances instead')
        return self.atoms

    def set_atoms_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Atoms in the Structure

        Keyword Args:
            numbers: Container[int] = None - The Atom numbers of interest
            pdb: bool = False - Whether to search for numbers as they were parsed (if True)
        """
        for atom in self.get_atoms(**kwargs):
            for kwarg, value in kwargs.items():
                setattr(atom, kwarg, value)

    # Todo
    #  self.atom_indices isn't long term sustainable...
    @property
    def _key(self) -> tuple[str, int, ...]:
        return self.name, *self.atom_indices

    def __eq__(self, other: StructureBase) -> bool:
        if isinstance(other, StructureBase):
            return self._key == other._key
        raise NotImplementedError(
            f"Can't compare {self.__class__.__name__} instance to {type(other).__name__} instance")

    # Must define __hash__ in all subclasses that define an __eq__
    def __hash__(self) -> int:
        return hash(self._key)


residue_attributes_literal = Literal[
    'contact_order',
    'local_density',
    'spatial_aggregation_propensity',
    'sasa',
    'sasa_apolar',
    'sasa_polar',
    'secondary_structure',
]


class Residue(ContainsAtoms, fragment.ResidueFragment):
    _ca_indices: list[int]
    _cb_indices: list[int]
    _bb_indices: list[int]
    _bb_and_cb_indices: list[int]
    _heavy_atom_indices: list[int]
    _index: int
    _sc_indices: list[int]
    _backbone_indices: list[int]
    _backbone_and_cb_indices: list[int]
    _heavy_indices: list[int]
    _side_chain_indices: list[int]
    _c_index: int
    _ca_index: int
    _cb_index: int
    _h_index: int
    _n_index: int
    _o_index: int
    _contact_order: float
    _local_density: float
    _sap: float
    _sasa: float
    _sasa_apolar: float
    _sasa_polar: float
    _secondary_structure: str
    chain_id: str = ''  # Used for abstract class with chain_id as plain attribute
    number: int = None  # Used for abstract class with number as plain attribute
    state_attributes = ContainsAtoms.state_attributes \
        | {'_contact_order', '_local_density',
           '_sasa', '_sasa_apolar', '_sasa_polar', '_secondary_structure'}
    # ignore_copy_attrs: set[str] =
    # ContainsAtoms.ignore_copy_attrs | {'_next_residue', '_prev_residue'}
    type: str

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)  # Residue
        if self.is_parent():
            # Setting up a parent (independent) Residue
            self._ensure_valid_residue()
            self.start_index = 0

        self.delegate_atoms()

    @property
    def index(self):
        """The Residue index in a Residues container"""
        try:
            return self._index
        except AttributeError:
            raise TypeError(
                f"{self.__class__.__name__} isn't a member of a Residues container and has no index")

    # @index.setter
    # def index(self, index):
    #     self._index = index

    def _ensure_valid_residue(self) -> bool:
        """Returns True if the Residue is constructed properly, otherwise raises an error

        Raises:
            ValueError: If the Residue is set improperly
        """
        try:
            first_atom, *other_atoms = self.atoms
        except ValueError:  # There are no Atom instances...
            self.log.warning(f'Constructing {self.__class__.__name__} with no Atom instances')
            return False

        first_residue_number = first_atom.residue_number
        first_residue_type = first_atom.residue_type
        found_types = set()
        for idx, atom in enumerate(other_atoms, 1):
            if atom.residue_number == first_residue_number and atom.residue_type == first_residue_type:
                atom_type = atom.type
                if atom_type not in found_types:
                    found_types.add(atom_type)
                else:
                    raise ValueError(
                        f'Invalid {self.__class__.__name__}. The Atom.type={atom_type} at index {idx} '
                        'was already observed')
            else:
                raise ValueError(
                    f"Invalid {self.__class__.__name__}. The {repr(atom)} at index {idx} doesn't have the same "
                    f'properties as prior Atom instances, such as {repr(first_atom)}')

        if protein_backbone_atom_types.difference(found_types):  # Todo Modify if building NucleotideResidue
            raise ValueError(
                f"Invalid {self.__class__.__name__}. The provided Atom instances don't contain the required "
                f"types, i.e. '{', '.join(protein_backbone_atom_types)}' for construction")

        return True

    @ContainsAtoms.start_index.setter
    def start_index(self, index: int):
        """Set Residue atom_indices starting with atom_indices[0] as start_index. Creates remainder incrementally and
        updates individual Atom instance .index accordingly
        """
        self._atom_indices = list(range(index, index + self.number_of_atoms))
        for atom, index in zip(self._atoms[self._atom_indices], self._atom_indices):
            atom.index = index
        # Clear all the indices attributes for this Residue
        self.reset_indices()

    @property
    def range(self) -> list[int]:
        """The range of indices corresponding to the Residue atoms"""
        return list(range(self.number_of_atoms))

    def delegate_atoms(self):
        """Set the Residue atoms from a parent StructureBase"""
        side_chain_indices, heavy_indices = [], []
        # try:
        #     for idx, atom in enumerate(self.atoms):
        #         match atom.type:  # Todo python 3.10
        #             case 'N':
        #                 self._n_index = idx
        #                 if self.number is None:
        #                     self.chain_id = atom.chain_id
        #                     self.number = atom.residue_number
        #                     self.type = atom.residue_type
        #                else:
        #                    raise stutils.ConstructionError(
        #                        f"Couldn't create a {self.__class__.__name__} with multiple 'N' Atom instances"
        #                    )
        #             case 'CA':
        #                 self._ca_index = idx
        #             case 'CB':
        #                 self._cb_index = idx
        #             case 'C':
        #                 self._c_index = idx
        #             case 'O':
        #                 self._o_index = idx
        #             case 'H':
        #                 self._h_index = idx
        #             case _:
        #                 side_chain_indices.append(idx)
        #                 if 'H' not in atom.type:
        #                     heavy_indices.append(idx)
        # except SyntaxError:  # python version not 3.10
        for idx, atom in enumerate(self.atoms):
            atom_type = atom.type
            if atom_type == 'N':
                self._n_index = idx
                if self.number is None:
                    self.chain_id = atom.chain_id
                    self.number = atom.residue_number
                    self.type = atom.residue_type
                else:
                    raise stutils.ConstructionError(
                        f"Couldn't create a {self.__class__.__name__} with multiple 'N' Atom instances"
                    )
            elif atom_type == 'CA':
                self._ca_index = idx
            elif atom_type == 'CB':
                self._cb_index = idx
            elif atom_type == 'C':
                self._c_index = idx
            elif atom_type == 'O':
                self._o_index = idx
            elif atom_type == 'H':  # 1H or H1
                self._h_index = idx
            # elif atom_type == 'OXT':
            #     self._oxt_index = idx
            # elif atom_type == 'H2':
            #     self._h2_index = idx
            # elif atom_type == 'H3':
            #     self._h3_index = idx
            else:
                side_chain_indices.append(idx)
                if 'H' not in atom_type:
                    heavy_indices.append(idx)

        # Construction ensures proper order for _bb_indices even if out of order
        # Important this order is correct for ProteinMPNN
        self.backbone_indices = [getattr(self, f'_{index}_index', None) for index in ['n', 'ca', 'c', 'o']]
        cb_index = getattr(self, '_cb_index', None)
        if cb_index:
            cb_indices = [cb_index]
        else:
            if self.type == 'GLY':  # Set _cb_index, but don't include in backbone_and_cb_indices
                self._cb_index = getattr(self, '_ca_index')
            cb_indices = []

        # By using private backbone_indices variable v, None is removed
        self.backbone_and_cb_indices = self._bb_indices + cb_indices
        self.heavy_indices = self._bb_and_cb_indices + heavy_indices
        self.side_chain_indices = side_chain_indices
        # if not self.ca_index:  # This is likely a NH or a C=O so there isn't a full residue
        #     self.log.error(f'{repr(self)} has no CA atom')
        #     # Todo this residue should be built out, but as of 6/28/22 it can only be deleted
        #     self.ca_index = idx  # Use the last found index as a rough guess
        #     self.secondary_structure = 'C'  # Just a placeholder since stride shouldn't work

    @property
    def type1(self) -> str:
        """Access the one character representation of the amino acid type"""
        return protein_letters_3to1_extended[self.type]

    @property
    def backbone_indices(self) -> list[int]:
        """The indices that index the Residue backbone Atoms/Coords"""
        try:
            return self._backbone_indices
        except AttributeError:
            self._backbone_indices = [self._atom_indices[idx] for idx in self._bb_indices]
            return self._backbone_indices

    @backbone_indices.setter
    def backbone_indices(self, indices: Iterable[int]):
        self._bb_indices = [idx for idx in indices if idx is not None]  # Check as some will be None if not provided

    @property
    def backbone_and_cb_indices(self) -> list[int]:
        """The indices that index the Residue backbone and CB Atoms/Coords"""
        try:
            return self._backbone_and_cb_indices
        except AttributeError:
            self._backbone_and_cb_indices = [self._atom_indices[idx] for idx in self._bb_and_cb_indices]
            return self._backbone_and_cb_indices

    @backbone_and_cb_indices.setter
    def backbone_and_cb_indices(self, indices: list[int]):
        self._bb_and_cb_indices = indices

    @property
    def ca_indices(self) -> list[int]:  # This is for compatibility with ContainsAtoms
        """Return the index of the CA Atom as a list in the Residue Atoms/Coords"""
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = [self._atom_indices[self._ca_index]]
            return self._ca_indices

    @property
    def cb_indices(self) -> list[int]:  # This is for compatibility with ContainsAtoms
        """Return the index of the CB Atom as a list in the Residue Atoms/Coords. Will return CA index if Glycine"""
        try:
            return self._cb_indices
        except AttributeError:
            self._cb_indices = [self._atom_indices[self._cb_index]]
            return self._cb_indices

    @property
    def side_chain_indices(self) -> list[int]:
        """The indices that index the Residue side chain Atoms/Coords"""
        try:
            return self._side_chain_indices
        except AttributeError:
            self._side_chain_indices = [self._atom_indices[idx] for idx in self._sc_indices]
        return self._side_chain_indices

    @side_chain_indices.setter
    def side_chain_indices(self, indices: list[int]):
        self._sc_indices = indices

    @property
    def heavy_indices(self) -> list[int]:
        """The indices that index the Residue heavy (non-hydrogen) Atoms/Coords"""
        try:
            return self._heavy_indices
        except AttributeError:
            self._heavy_indices = [self._atom_indices[idx] for idx in self._heavy_atom_indices]
            return self._heavy_indices

    @heavy_indices.setter
    def heavy_indices(self, indices: list[int]):
        self._heavy_atom_indices = indices

    def contains_hydrogen(self) -> bool:  # in Structure too
        """Returns whether the Residue contains hydrogen atoms"""
        return self.heavy_indices != self._atom_indices

    @property
    def n(self) -> Atom | None:
        """Return the amide N Atom object"""
        try:
            return self._atoms[self._atom_indices[self._n_index]]
        except AttributeError:
            return None

    @property
    def n_coords(self) -> np.ndarray | None:
        """Return the amide N Atom coordinate"""
        try:
            return self._coords.coords[self._atom_indices[self._n_index]]
        except AttributeError:
            return None

    @property
    def n_atom_index(self) -> int | None:
        """Return the index of the amide N Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._n_index]
        except AttributeError:
            return None

    @property
    def n_index(self) -> int | None:
        """Return the index of the amide N Atom in the Residue Atoms/Coords"""
        try:
            return self._n_index
        except AttributeError:
            return None

    # @n_index.setter
    # def n_index(self, index: int):
    #     self._n_index = index

    @property
    def h(self) -> Atom | None:
        """Return the amide H Atom object"""
        try:
            return self._atoms[self._atom_indices[self._h_index]]
        except AttributeError:
            return None

    @property
    def h_coords(self) -> np.ndarray | None:
        """Return the amide H Atom coordinate"""
        try:
            return self._coords.coords[self._atom_indices[self._h_index]]
        except AttributeError:
            return None

    @property
    def h_atom_index(self) -> int | None:
        """Return the index of the amide H Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._h_index]
        except AttributeError:
            return None

    @property
    def h_index(self) -> int | None:
        """Return the index of the amide H Atom in the Residue Atoms/Coords"""
        try:
            return self._h_index
        except AttributeError:
            return None

    # @h_index.setter
    # def h_index(self, index: int):
    #     self._h_index = index

    @property
    def ca(self) -> Atom | None:
        """Return the CA Atom object"""
        try:
            return self._atoms[self._atom_indices[self._ca_index]]
        except AttributeError:
            return None

    @property
    def ca_coords(self) -> np.ndarray | None:
        """Return the CA Atom coordinate"""
        try:
            return self._coords.coords[self._atom_indices[self._ca_index]]
        except AttributeError:
            return None  # np.ndarray([])

    @property
    def ca_atom_index(self) -> int | None:
        """Return the index of the CA Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._ca_index]
        except AttributeError:
            return None

    @property
    def ca_index(self) -> int | None:
        """Return the index of the CA Atom in the Residue Atoms/Coords"""
        try:
            return self._ca_index
        except AttributeError:
            return None

    # @ca_index.setter
    # def ca_index(self, index: int):
    #     self._ca_index = index

    @property
    def cb(self) -> Atom | None:
        """Return the CB Atom object"""
        try:
            return self._atoms[self._atom_indices[self._cb_index]]
        except AttributeError:
            return None

    @property
    def cb_coords(self) -> np.ndarray | None:
        """Return the CB Atom coordinate"""
        try:
            return self._coords.coords[self._atom_indices[self._cb_index]]
        except AttributeError:
            return None

    @property
    def cb_atom_index(self) -> int | None:
        """Return the index of the CB Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._cb_index]
        except AttributeError:
            return None

    @property
    def cb_index(self) -> int | None:
        """Return the index of the CB Atom in the Residue Atoms/Coords"""
        try:
            return self._cb_index
        except AttributeError:
            return None

    # @cb_index.setter
    # def cb_index(self, index: int):
    #     self._cb_index = index

    @property
    def c(self) -> Atom | None:
        """Return the carbonyl C Atom object"""
        try:
            return self._atoms[self._atom_indices[self._c_index]]
        except AttributeError:
            return None

    @property
    def c_coords(self) -> np.ndarray | None:
        """Return the carbonyl C Atom coordinate"""
        try:
            return self._coords.coords[self._atom_indices[self._c_index]]
        except AttributeError:
            return None

    @property
    def c_atom_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._c_index]
        except AttributeError:
            return None

    @property
    def c_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Residue Atoms/Coords"""
        try:
            return self._c_index
        except AttributeError:
            return None

    # @c_index.setter
    # def c_index(self, index: int):
    #     self._c_index = index

    @property
    def o(self) -> Atom | None:
        """Return the carbonyl O Atom object"""
        try:
            return self._atoms[self._atom_indices[self._o_index]]
        except AttributeError:
            return None

    @property
    def o_coords(self) -> np.ndarray | None:
        """Return the carbonyl O Atom coords"""
        try:
            return self._coords.coords[self._atom_indices[self._o_index]]
        except AttributeError:
            return None

    @property
    def o_atom_index(self) -> int | None:
        """Return the index of the carbonyl C Atom in the Structure Atoms/Coords"""
        try:
            return self._atom_indices[self._o_index]
        except AttributeError:
            return None

    @property
    def o_index(self) -> int | None:
        """Return the index of the carbonyl O Atom in the Residue Atoms/Coords"""
        try:
            return self._o_index
        except AttributeError:
            return None

    # @o_index.setter
    # def o_index(self, index: int):
    #     self._o_index = index

    @property
    def _residues(self) -> Residues:
        try:
            return self.parent.residues
        except AttributeError as exc:
            raise stutils.DesignError(
                f"{repr(self)} isn't a parent and doesn't have have any siblings..."
            ) from exc

    @property
    def prev_residue(self) -> Residue | None:
        """The previous Residue in the Structure if this Residue is part of a polymer"""
        try:
            return self._residues[self.index - 1]
        except IndexError:
            return None

    def is_n_termini(self) -> bool:
        """Returns whether the Residue is the n-termini of the parent Structure"""
        return self.prev_residue is None

    @property
    def next_residue(self) -> Residue | None:
        """The next Residue in the Structure if this Residue is part of a polymer"""
        try:
            return self._residues[self.index + 1]
        except IndexError:
            return None

    def is_c_termini(self) -> bool:
        """Returns whether the Residue is the c-termini of the parent Structure"""
        return self.next_residue is None

    def get_upstream(self, number: int = None) -> list[Residue]:
        """Get the Residues upstream of (n-terminal to) the current Residue

        Args:
            number: The number of residues to retrieve. If not provided gets all
        Returns:
            The Residue instances in n- to c-terminal order
        """
        if number is None:
            number = sys.maxsize
        elif number == 0:
            raise ValueError("Can't get 0 upstream residues. 1 or more must be specified")

        last_prev_residue = self.prev_residue
        prev_residues = [last_prev_residue]
        idx = 0
        try:
            for idx in range(abs(number) - 1):
                prev_residue = last_prev_residue.prev_residue
                prev_residues.append(prev_residue)
                last_prev_residue = prev_residue
        except AttributeError:  # Hit a termini, where prev_residue is None
            # logger.debug(f'Stopped at {idx=} with {repr(prev_residues)}. Popping the last')
            prev_residues.pop(idx)
        else:  # For the edge case where the last added residue is a termini, strip from results
            if last_prev_residue is None:
                prev_residues.pop()

        return prev_residues[::-1]

    def get_downstream(self, number: int = None) -> list[Residue]:
        """Get the Residues downstream of (c-terminal to) the current Residue

        Args:
            number: The number of residues to retrieve. If not provided gets all
        Returns:
            The Residue instances in n- to c-terminal order
        """
        if number is None:
            number = sys.maxsize
        elif number == 0:
            raise ValueError("Can't get 0 downstream residues. 1 or more must be specified")

        last_next_residue = self.next_residue
        next_residues = [last_next_residue]
        idx = 0
        try:
            for idx in range(abs(number) - 1):
                next_residue = last_next_residue.next_residue
                next_residues.append(next_residue)
                last_next_residue = next_residue
        except AttributeError:  # Hit a termini, where next_residue is None
            # logger.debug(f'Stopped at {idx=} with {repr(next_residues)}. Popping the last')
            next_residues.pop(idx)
        else:  # For the edge case where the last added residue is a termini, strip from results
            if last_next_residue is None:
                next_residues.pop()

        return next_residues

    def get_neighbors(self, distance: float = 8., **kwargs) -> list[Residue]:
        """If this Residue instance is part of a polymer, find neighboring Residue instances defined by a distance

        Args:
            distance: The distance to measure neighbors by
        Returns:
            The Residue instances that are within the distance to this Residue
        """
        try:
            return self.parent.get_residues_by_atom_indices(self.neighboring_atom_indices(distance=distance, **kwargs))
        except AttributeError:  # This Residue is the parent
            return [self]

    def _warn_missing_attribute(self, attribute_name: str, func: str = None) -> str:
        return f"{repr(self)} has no '.{attribute_name}' attribute. Ensure you call " \
               f'{func} before you request Residue {" ".join(attribute_name.split("_"))} information'

    # Below properties are considered part of the Residue state
    @property
    def local_density(self) -> float:
        """Describes how many heavy Atoms are within a distance (default = 12 Angstroms) of Residue heavy Atoms"""
        try:
            return self._local_density
        except AttributeError:
            self._local_density = 0.  # Set to 0 so summation can occur
            try:
                self.parent.local_density()
                return self._local_density
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.local_density.fget.__name__, 'local_density'))

    @local_density.setter
    def local_density(self, local_density: float):
        self._local_density = local_density

    @property
    def secondary_structure(self) -> str:
        """Return the secondary structure designation as defined by a secondary structure calculation"""
        try:
            return self._secondary_structure
        except AttributeError:
            # self.parent._generate_secondary_structure()  # Sets this ._secondary_structure
            # try:
            #     return self._secondary_structure
            # except AttributeError:
            raise AttributeError(
                self._warn_missing_attribute(Residue.secondary_structure.fget.__name__,'secondary_structure'))

    @secondary_structure.setter
    def secondary_structure(self, ss_code: str):
        """Set the secondary_structure for the Residue"""
        self._secondary_structure = ss_code

    def _segregate_sasa(self):
        """Separate sasa into apolar and polar according to Atoms. If not available, try parent.get_sasa()"""
        residue_atom_polarity = atomic_polarity_table[self.type]
        polarity_list = [[], [], []]  # apolar = 0, polar = 1, unknown = 2 (-1)
        try:
            # Careful. residue_atom_polarity.get() doesn't work properly with defaultdict return -1
            #  this was causing TypeError on 8/15/22
            for atom in self.atoms:
                polarity_list[residue_atom_polarity[atom.type]].append(atom.sasa)
        except AttributeError:  # Missing atom.sasa
            self.parent.get_sasa()
            for atom in self.atoms:
                polarity_list[residue_atom_polarity[atom.type]].append(atom.sasa)
        # except TypeError:
        #     print(residue_atom_polarity, atom.type, self.type, self.number)

        self._sasa_apolar, self._sasa_polar, _ = map(sum, polarity_list)
        # if _ > 0:
        #     print('Found %f unknown surface area' % _)

    @property
    def sasa(self) -> float:
        """Return the solvent accessible surface area as calculated by a solvent accessible surface area calculator"""
        try:
            return self._sasa
        except AttributeError:
            try:  # let _segregate_sasa() call get_sasa() from the .parent if sasa is missing
                self._sasa = self.sasa_apolar + self.sasa_polar
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.sasa.fget.__name__, 'get_sasa'))

            return self._sasa

    @sasa.setter
    def sasa(self, sasa: float):
        """Set the total solvent accessible surface area for the Residue"""
        self._sasa = sasa

    @property
    def sasa_apolar(self) -> float:
        """Return apolar solvent accessible surface area as calculated by a solvent accessible surface area calculator
        """
        try:
            return self._sasa_apolar
        except AttributeError:
            try:
                self._segregate_sasa()
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.sasa_apolar.fget.__name__, 'get_sasa'))

            return self._sasa_apolar

    @sasa_apolar.setter
    def sasa_apolar(self, sasa: float | int):
        """Set the apolar solvent accessible surface area for the Residue"""
        self._sasa_apolar = sasa

    @property
    def sasa_polar(self) -> float:
        """Return polar solvent accessible surface area as calculated by a solvent accessible surface area calculator
        """
        try:
            return self._sasa_polar
        except AttributeError:
            try:
                self._segregate_sasa()
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.sasa_polar.fget.__name__, 'get_sasa'))

            return self._sasa_polar

    @sasa_polar.setter
    def sasa_polar(self, sasa: float | int):
        """Set the polar solvent accessible surface area for the Residue"""
        self._sasa_polar = sasa

    @property
    def relative_sasa(self) -> float:
        """The solvent accessible surface area relative to the standard surface accessibility of the Residue type"""
        return self.sasa / gxg_sasa[self.type]  # May cause problems if self.type attribute can be non-cannonical AA

    @property
    def spatial_aggregation_propensity(self) -> float:
        """The Residue contact order, which describes how far away each Residue makes contacts in the polymer chain"""
        try:
            return self._sap
        except AttributeError:
            try:
                self.parent.spatial_aggregation_propensity_per_residue()
                return self._sap
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.spatial_aggregation_propensity.fget.__name__,
                                                 'spatial_aggregation_propensity'))

    @spatial_aggregation_propensity.setter
    def spatial_aggregation_propensity(self, sap: float):
        self._sap = sap

    @property
    def contact_order(self) -> float:
        """The Residue contact order, which describes how far away each Residue makes contacts in the polymer chain"""
        try:
            return self._contact_order
        except AttributeError:
            try:
                self.parent.contact_order_per_residue()
                return self._contact_order
            except AttributeError:
                raise AttributeError(
                    self._warn_missing_attribute(Residue.contact_order.fget.__name__, 'contact_order'))

    @contact_order.setter
    def contact_order(self, contact_order: float):
        self._contact_order = contact_order

    # End state properties
    @property
    def b_factor(self) -> float:
        try:
            return sum(atom.b_factor for atom in self.atoms) / self.number_of_atoms
        except ZeroDivisionError:
            return 0.

    @b_factor.setter
    def b_factor(self, dtype: str | float = None, **kwargs):
        """Set the temperature factor for the Atoms in the Residue

        Args:
            dtype: The data type that should fill the temperature_factor from Residue attributes
                or an iterable containing the explicit b_factor float values
        """
        try:
            for atom in self.atoms:
                atom.b_factor = self.__getattribute__(dtype)
        except AttributeError:
            raise AttributeError(
                f"The attribute {dtype} wasn't found in the {repr(self)}. Are you sure this is the attribute you want?")
        except TypeError:  # dtype isn't a string
            # raise TypeError(f'{type(dtype)} is not a string. To set b_factor, you must provide the dtype as a string')
            # try:
            for atom in self.atoms:
                atom.b_factor = dtype
            # except TypeError:
            #     raise TypeError(f'{type(dtype)} is not a string nor an Iterable. To set b_factor, you must provide the '
            #                     f'dtype as a string specifying a Residue attribute OR an integer with length = '
            #                     f'Residue.number_of_atoms')

    def mutation_possibilities_from_directive(
        self, directive: directives = None, background: set[str] = None, special: bool = False, **kwargs
    ) -> set[protein_letters3_alph1_literal] | set:
        """Select mutational possibilities for each Residue based on the Residue and a directive

        Args:
            directive: Where the choice is one of 'special', 'same', 'different', 'charged', 'polar', 'apolar',
                'hydrophobic', 'aromatic', 'hbonding', 'branched'
            background: The background amino acids to compare possibilities against
            special: Whether to include special residues

        Returns:
            The possible amino acid types available given the mutational directive
        """
        if not directive or directive not in mutation_directives:
            self.log.debug(f'{self.mutation_possibilities_from_directive.__name__}: The mutation directive {directive} '
                           f'is not a valid directive yet. Possible directives are: {", ".join(mutation_directives)}')
            return set()
            # raise TypeError('%s: The mutation directive %s is not a valid directive yet. Possible directives are: %s'
            #                 % (self.mutation_possibilities_from_directive.__name__, directive,
            #                    ', '.join(mutation_directives)))

        current_properties = residue_properties[self.type]
        if directive == 'same':
            properties = current_properties
        elif directive == 'different':  # hmm not right -> .difference({hbonding, branched}) <- for ex. polar if apolar
            properties = set(aa_by_property.keys()).difference(current_properties)
        else:
            properties = [directive]
        available_aas = set(aa for prop in properties for aa in aa_by_property[prop])

        if directive != 'special' and not special:
            available_aas = available_aas.difference(aa_by_property['special'])
        if background:
            available_aas = background.intersection(available_aas)

        return available_aas

    def distance(self, other: Residue, dtype: str = 'ca') -> float:
        """Return the distance from this Residue to another specified by atom type "dtype"

        Args:
            other: The other Residue to measure against
            dtype: The Atom type to perform the measurement with
        Returns:
            The Euclidean distance between the specified Atom type
        """
        return np.linalg.norm(getattr(self, f'.{dtype}_coords') - getattr(other, f'.{dtype}_coords'))

    # def residue_string(self, pdb: bool = False, chain_id: str = None, **kwargs) -> tuple[str, str, str]:
    #     """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
    #     formatting
    #
    #     Args:
    #         pdb: Whether the Residue representation should use the pdb number at file parsing
    #         chain_id: The ID of the chain_id to use
    #     Returns:
    #         Tuple of formatted Residue attributes
    #     """
    #     return format(self.type, '3s'), (chain_id or self.chain_id), \
    #         format(getattr(self, f'number{"_pdb" if pdb else ""}'), '4d')

    def __getitem__(self, idx) -> Atom:
        return self.atoms[idx]

    @property
    def _key(self) -> tuple[int, int, str]:
        return self._index, self.start_index, self.type  # self.entity_id

    def __eq__(self, other: Residue) -> bool:
        if isinstance(other, Residue):
            return self._key == other._key
        raise NotImplementedError(
            f"Can't compare {self.__class__.__name__} instance to {type(other).__name__} instance")

    def get_atom_record(self, chain_id: str = None, atom_offset: int = 0, **kwargs) -> str:
        """Provide the Structure Atoms as a PDB file string

        Args:
            chain_id: The chain ID to use
            atom_offset: How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        return f'{self.__str__(chain_id=chain_id, atom_offset=atom_offset, **kwargs)}\n'

    def __str__(self, chain_id: str = None, atom_offset: int = 0, **kwargs) -> str:
        #         type=None, number=None
        """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
        formatting

        Args:
            chain_id: The chain ID to use
            atom_offset: How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Residue
        """
        #     pdb: Whether the Residue representation should use the number at file parsing
        # format the string returned from each Atom, such as
        #  'ATOM  %s  CG2 %s %s%s    %s  1.00 17.48           C  0'
        #       AtomIdx  TypeChNumberCoords
        # To
        #  'ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  0'
        # self.type, self.alt_location, self.code_for_insertion, self.occupancy, self.b_factor,
        #                     self.element, self.charge)
        # res_str = self.residue_string(**kwargs)
        res_str = format(self.type, '3s'), format(chain_id or self.chain_id, '>2s'), format(self.number, '4d')
        # Add 1 to make index one-indexed
        offset = 1 + atom_offset
        # Limit atom_index with atom_index_slice to keep ATOM record correct length v
        return '\n'.join(atom.__str__().format(format(atom_idx + offset, '5d')[atom_index_slice],
                                               *res_str, '{:8.3f}{:8.3f}{:8.3f}'.format(*coord))
                         for atom, atom_idx, coord in zip(self.atoms, self._atom_indices, self.coords.tolist()))

    def __repr__(self) -> str:
        return f'{Residue.__name__}({self.number}{self.chain_id})'

    def __hash__(self) -> int:
        return hash(self._key)

    def __copy__(self) -> Residue:  # Todo -> Self: in python 3.11
        other: Residue = super().__copy__()


        return other

    copy = __copy__  # Overwrites to use this instance __copy__


class Residues(StructureBaseContainer):

    # def find_prev_and_next(self):
    #     """Set prev_residue and next_residue attributes for each Residue. One inherently sets the other in Residue"""
    #     structs = self.structs
    #     for struct, next_struct in zip(structs[:-1], structs[1:]):
    #         struct.next_residue = next_struct

    def reindex(self, start_at: int = 0):
        """Index the Residue instances and their corresponding Atom/Coords indices according to their position

        Args:
            start_at: The index to start reindexing at. Must be [0, 'inf']
        """
        self.set_index(start_at=start_at)
        self.reindex_atoms(start_at=start_at)

    def set_index(self, start_at: int = 0):
        """Index the Residue instances according to their position in the Residues container

        Args:
            start_at: The Residue index to start reindexing at
        """
        for idx, struct in enumerate(self.structs[start_at:].tolist(), start_at):
            struct._index = idx

    def reindex_atoms(self, start_at: int = 0):
        """Index the Residue instances Atoms/Coords indices according to incremental Atoms/Coords index. Responsible
        for updating member Atom.index attributes as well

        Args:
            start_at: The index to start reindexing at. Must be [0, 'inf']
        """
        if start_at == 0:
            _start_at = start_at
        else:
            _start_at = start_at - 1
            if start_at < 0:
                _start_at += len(self)

        struct: Residue
        prior_struct: Residue
        try:
            # prior_struct, *other_structs = self.structs[start_at - 1:]
            prior_struct, *other_structs = self[_start_at:]
        except ValueError:  # Not enough values to unpack as the index didn't slice anything
            raise IndexError(
                f'{self.reindex_atoms.__name__}: {start_at=} is outside of the allowed {self.__class__.__name__} '
                f'indices with size {len(self)}')
        else:
            if start_at == 0:
                prior_struct.start_index = start_at

            for struct in other_structs:
                struct.start_index = prior_struct.end_index + 1
                prior_struct = struct

    def __copy__(self) -> Residues:  # Todo -> Self: in python 3.11
        other: Residues = super().__copy__()

        return other

    copy = __copy__  # Overwrites to use this instance __copy__


chain_assignment_error = "Can't solve for the Residue chainID association automatically. If the new " \
                         'Residue is at a Structure termini in a multi-Structure, Structure container, ' \
                         "you must specify which Structure it belongs to by passing chain_id='ID'"
# from types import EllipsisType
ArrayIndexer = Type[Union[Sequence[int], Sequence[bool], slice, None]]  # EllipsisType python 3.10
"""Where integers, slices `:`, ellipsis `...`, None and integer or boolean arrays are valid indices"""


class StructureIndexMixin(ABC):
    """"""

    @abc.abstractmethod
    def is_parent(self) -> bool:
        """"""

    def _start_indices(self, at: int = 0, dtype: atom_or_residue_literal = None):
        """Modify Structure container indices by a set integer amount

        Args:
            at: The index to insert indices at
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        try:  # To get the indices through the public property
            indices = self.__getattribute__(f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(
                f"The {dtype=} wasn't found from the {self.__class__.__name__}.dtype_indices. Possible values "
                "of dtype are 'atom' or 'residue'")
        offset = at - indices[0]
        # Set the indices through the private attribute
        self.__setattr__(f'_{dtype}_indices', [prior_idx + offset for prior_idx in indices])

    def _insert_indices(self, at: int, new_indices: list[int], dtype: atom_or_residue_literal = None):
        """Modify Structure container indices by a set integer amount

        Args:
            at: The index to insert indices at
            new_indices: The indices to insert
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        if new_indices is None:
            return
        try:
            indices = self.__getattribute__(f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(
                f"The {dtype=} wasn't found from the {self.__class__.__name__}.dtype_indices. Possible values "
                "of dtype are 'atom' or 'residue'")
        number_new = len(new_indices)
        self.__setattr__(f'_{dtype}_indices',
                         indices[:at] + new_indices + [idx + number_new for idx in indices[at:]])

    def _offset_indices(self, start_at: int = None, offset: int = None, dtype: atom_or_residue_literal = 'atom'):
        """Reindex Structure 'dtype'_indices by an integer offset, starting with the 'start_at' index

        Args:
            start_at: The index to start reindexing at. Must be [0, 'inf']
            offset: The integer to offset the index by. For negative offset, pass a negative value
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        try:
            indices = self.__getattribute__(f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(
                f"The {dtype=} wasn't found from the {self.__class__.__name__}.dtype_indices. Possible values "
                f"of dtype are 'atom' or 'residue'")
        if start_at is not None:
            try:
                self.__setattr__(f'_{dtype}_indices',
                                 indices[:start_at] + [idx + offset for idx in indices[start_at:]])
                # self._atom_indices = \
                #     self._atom_indices[:start_at] + [idx + offset for idx in self._atom_indices[start_at:]]
            except TypeError:  # None is not valid
                raise ValueError(
                    f"offset value {offset} isn't valid. Must provide an integer when offsetting {dtype} indices using "
                    "the argument 'start_at'")
        elif self.is_parent():  # Just reset all the indices without regard for gaps
            self.__setattr__(f'_{dtype}_indices', list(range(self.__getattribute__(f'number_of_{dtype}'))))
            # self._atom_indices = list(range(self.number_of_atoms))
        # This shouldn't be used for a Structure object who is dependent on another Structure
        else:
            raise ValueError(
                f"{self._offset_indices.__name__}: Must include 'start_at' when offsetting {dtype} indices from a "
                'dependent structure')

    def _delete_indices(self, delete_indices: Iterable[int], dtype: atom_or_residue_literal = 'atom'):
        """Remove the delete_indices from the Structure dtype_indices. Currently pops the indices len(delete_indices)
        times which reindexes the remaining indices

        Args:
            delete_indices: The indices to delete
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        if delete_indices is None:
            return
        try:
            indices = self.__getattribute__(f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(
                f"The {dtype=} wasn't found from the {self.__class__.__name__}.dtype_indices. Possible values "
                "of dtype are 'atom' or 'residue'")
        try:
            for _ in delete_indices:  # reversed(delete_indices):
                indices.pop()  # idx)
        except IndexError:  # Reached the last index with None remaining
            self.log.debug(f'{self._delete_indices.__name__}: Reached the end of the {dtype}_indices, but requested '
                           'more be deleted')
            return
            # raise IndexError(f"The index, {idx}, wasn't found in the {self.name}.{dtype}_indices")
        except AttributeError:
            raise AttributeError(
                f"The {self.__class__.__name__} doesn't have any {dtype}_indices")


class ContainsResidues(ContainsAtoms, StructureIndexMixin):
    """Structure object handles Atom/Residue/Coords manipulation of all Structure containers.
    Must pass parent and residue_indices, atoms and coords, or residues to initialize

    Polymer/Chain designation. This designation essentially means it contains Residue instances in a Residues object
    """
    ss_sequence_indices: list[int]
    """Index which indicates the Residue membership to the secondary structure type element sequence"""
    ss_type_sequence: list[str]
    """The ordered secondary structure type sequence which contains one character/secondary structure element"""
    _backbone_and_cb_indices: list[int]
    _backbone_indices: list[int]
    _ca_indices: list[int]
    _cb_indices: list[int]
    _contains_hydrogen: bool
    _fragment_db: fragment.db.FragmentDatabase | None
    _heavy_indices: list[int]
    _helix_cb_indices: list[int]
    _side_chain_indices: list[int]
    _contact_order: np.ndarray
    _residues: Residues | None
    _residue_indices: list[int] | None
    _sap: np.ndarray
    _secondary_structure: str
    _sequence: str
    nucleotides_present: bool = False
    secondary_structure: str | None
    sasa: float | None
    _coords_indexed_residues_: np.ndarray  # Residues # list[Residue]
    # _coords_indexed_residue_atoms: np.ndarray  # list[int]
    """Specifies which containers of Structure instances are utilized by this class to aid state changes like copy()"""
    state_attributes = ContainsAtoms.state_attributes \
        | {'_sequence', '_helix_cb_indices', '_secondary_structure', '_coords_indexed_residues_'}
    ignore_copy_attrs: set[str] = ContainsAtoms.ignore_copy_attrs | {'_coords_indexed_residues_'}

    @classmethod
    def from_residues(cls, residues: list[Residue] | Residues, **kwargs):
        """Initialize from existing Residue instances"""
        return cls(residues=residues, **kwargs)

    @classmethod
    def from_structure(cls, structure: ContainsResidues, **kwargs):
        """Initialize from an existing Structure"""
        return cls(structure=structure, **kwargs)

    def __init__(self, structure: ContainsResidues = None,
                 residues: list[Residue] | Residues = None, residue_indices: list[int] = None,
                 pose_format: bool = False, fragment_db: fragment.db.FragmentDatabase = None, **kwargs):
        """
        Args:
            structure: Create the instance based on an existing Structure instance
            residues: The Residue instances which should constitute a new Structure instance
            residue_indices: The indices which specify the particular Residue instances to make this Structure instance.
                Used with a parent to specify a subdivision of a larger Structure
            pose_format: Whether to initialize with continuous Residue numbering from 1 to N
        Keyword Args:
            atoms: list[Atom] | Atoms = None - The Atom instances which should constitute a new Structure instance
            parent: StructureBase = None - If another Structure object created this Structure instance, pass the
                'parent' instance. Will take ownership over Structure containers (coords, atoms, residues) for
                dependent Structures
            log: Log | Logger | bool = True - The Log or Logger instance, or the name for the logger to handle parent
                Structure logging. None or False prevents logging while any True assignment enables it
            coords: Coords | np.ndarray | list[list[float]] = None - When setting up a parent Structure instance, the
                coordinates of that Structure
            name: str = None - The identifier for the Structure instance
        """
        if structure:
            if isinstance(structure, ContainsResidues):
                model_kwargs = structure.get_base_containers()
                for key, value in model_kwargs.items():
                    if key in kwargs:
                        logger.warning(f"Passing an argument for '{key}' while providing the 'model' argument "
                                       f"overwrites the '{key}' argument from the 'model'")
                new_model_kwargs = {**model_kwargs, **kwargs}
                super().__init__(**new_model_kwargs)
            else:
                raise NotImplementedError(
                    f"Setting {self.__class__.__name__} with model={type(structure).__name__} isn't supported")
        else:
            super().__init__(**kwargs)  # ContainsResidues

        # self._coords_indexed_residues_ = None
        # self._residue_indices = None
        # self.secondary_structure = None
        # self.nucleotides_present = False
        self._fragment_db = fragment_db
        self.sasa = None
        self.ss_sequence_indices = []
        self.ss_type_sequence = []

        if self.is_dependent():
            try:
                residue_indices[0]
            except TypeError:
                if isinstance(self, Structures):  # Structures handles this itself
                    return
                raise stutils.ConstructionError(
                    f"The {self.__class__.__name__} wasn't passed 'residue_indices' which are required for "
                    f"initialization"
                )

            if not isinstance(residue_indices, list):
                residue_indices = list(residue_indices)
            # Must set this before setting _atom_indices
            self._residue_indices = residue_indices
            # Get the atom_indices from the provided residues
            self._atom_indices = [idx for residue in self.residues for idx in residue.atom_indices]
        # Setting up a parent Structure
        elif residues:  # Assume the passed residues aren't bound to an existing Structure
            self._assign_residues(residues)
        elif self.atoms:
            # Assume ContainsAtoms initialized .atoms. Make Residue instances, Residues
            self._create_residues()
        else:  # Set up an empty Structure or let subclass handle population
            return

        if pose_format:
            self.pose_numbering()

    def assign_residues_from_structures(self, structures: Iterable[ContainsResidues]):
        """Initialize the instance from existing structure instance attributes, .coords, and .atoms, and .residues"""
        atoms, residues, coords = [], [], []
        for structure in structures:
            atoms.extend(structure.atoms)
            residues.extend(structure.residues)
            coords.append(structure.coords)

        self._assign_residues(residues, atoms=atoms, coords=coords)

    @StructureBase._parent.setter
    def _parent(self, parent: StructureBase):
        """Set the 'parent' StructureBase of this instance"""
        super(ContainsAtoms, ContainsAtoms)._parent.fset(self, parent)
        self._atoms = parent._atoms
        self._residues = parent._residues

    def make_parent(self):
        """Remove this instance from its parent, making it a parent in the process"""
        super(ContainsAtoms, ContainsAtoms).make_parent(self)
        # Populate the Structure with its existing instances removed of any indexing
        self._assign_residues(self.residues, atoms=self.atoms)
        self.reset_state()

    def get_base_containers(self) -> dict[str, Any]:
        """Returns the instance structural containers as a dictionary with attribute as key and container as value"""
        return dict(coords=self._coords, atoms=self._atoms, residues=self._residues)

    @property
    def fragment_db(self) -> fragment.db.FragmentDatabase:
        """The FragmentDatabase used to create Fragment instances"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: fragment.db.FragmentDatabase):
        """Set the Structure FragmentDatabase to assist with Fragment creation, manipulation, and profiles"""
        if not isinstance(fragment_db, fragment.db.FragmentDatabase):
            self.log.debug(f"'fragment_db was set to the default since a {type(fragment_db).__name__} was passed which "
                           f"isn't the required type {fragment.db.FragmentDatabase.__name__}")
            fragment_db = fragment.db.fragment_factory.get(source=putils.biological_interfaces, token=fragment_db)

        self._fragment_db = fragment_db

    # Below properties are considered part of the Structure state
    def contains_hydrogen(self) -> bool:  # in Residue too
        """Returns whether the Structure contains hydrogen atoms"""
        try:
            return self._contains_hydrogen
        except AttributeError:
            # Sample 20 residues from various "locations".
            # If H is still present but not found, there is an anomaly in the Structure
            slice_amount = max(int(self.number_of_residues / 20), 1)
            for residue in self.residues[::slice_amount]:
                if residue.contains_hydrogen():
                    self._contains_hydrogen = True
                    break
            else:
                self._contains_hydrogen = False

        return self._contains_hydrogen

    @property
    def sequence(self) -> str:
        """Holds the Structure amino acid sequence"""
        try:
            return self._sequence
        except AttributeError:
            self._sequence = ''.join(residue.type1 for residue in self.residues)
            return self._sequence

    @sequence.setter
    def sequence(self, sequence: str):
        if len(sequence) != self.number_of_residues:
            raise ValueError(
                f"Couldn't set the {repr(self)}.sequence to the provided sequence with length "
                f"{len(sequence)} != {self.number_of_residues}, the number of residues"
            )
        self._sequence = sequence

    @property
    def residue_indices(self) -> list[int] | None:
        """Return the residue indices which belong to the Structure"""
        try:
            return self._residue_indices
        except AttributeError:
            return

    @property
    def residues(self) -> list[Residue] | None:
        """Return the Residue instances in the Structure"""
        try:
            return self._residues[self._residue_indices]
        except AttributeError:  # When self._residues isn't set
            return None

    # @residues.setter
    # def residues(self, residues: Residues | list[Residue]):
    #     """Set the Structure atoms to a Residues object"""
    #     # Todo make this setter function in the same way as self._coords.replace?
    #     if isinstance(residues, Residues):
    #         self._residues = residues
    #     else:
    #         self._residues = Residues(residues)

    def _assign_residues(self, residues: Residues | list[Residue], atoms: Atoms | list[Atom] = None, **kwargs):
        """Assign Residue instances, create Residues instances

        This will make all Residue instances (and their Atom instances) dependents of this instance

        Args:
            residues: The Residue instances to assign to the Structure
            atoms: The Atom instances or Atoms to assign. Optional, will use Residues.atoms if not specified
        Keyword Args:
            coords: numpy.ndarray = None - The coordinates to assign to the StructureBase.
                Optional, will use Residue.coords if not specified
        Sets:
            self._atom_indices (list[int])

            self._atoms (Atoms)

            self._residue_indices (list[int])

            self._residues (Residues)
        """
        if not atoms:
            atoms = []
            for residue in residues:
                atoms.extend(residue.atoms)
        self._assign_atoms(atoms, atoms_only=False)  # Not passing kwargs as _populate_coords() handles
        # Done below with _residues.reindex(), not necessary here
        # self._atoms.reindex()

        # Set proper residues attributes
        self._residue_indices = list(range(len(residues)))
        if not isinstance(residues, Residues):  # Must create the residues object
            residues = Residues(residues)

        if residues.are_dependents():  # Copy Residues object to set new attributes on each member Residue
            # This may be an additional copy with the Residues(residues) construction above
            residues = residues.copy()
            residues.reset_state()  # Clear runtime attributes
        else:
            raise RuntimeError(
                f'{self.__class__.__name__} {self.name} received Residue instances that are not dependents of a parent.'
                'This check was put in place to inspect program runtime. How did this situation occur that residues '
                'are not dependents?')
        self._residues = residues

        self._populate_coords(from_source='residues', **kwargs)  # coords may be passed in kwargs
        # Ensure that coordinates lengths match
        self._validate_coords()
        # Update Atom instance attributes to ensure they are dependants of this instance
        self._atoms.set_attributes(_parent=self)
        # Update Residue instance attributes to ensure they are dependants of this instance
        # Perform after _populate_coords as .coords may be set and then 'residues' .coords are overwritten
        self._residues.set_attributes(_parent=self)
        self._residues.reindex()

    # @property
    # def residue_indexed_atom_indices(self) -> list[list[int]]:
    #     """For every Residue in the Structure provide the Residue instance indexed, Structure Atom indices
    #
    #     Returns:
    #         Residue objects indexed by the Residue position in the corresponding .coords attribute
    #     """
    #     try:
    #         return self._residue_indexed_atom_indices  # [self._atom_indices]
    #     except (AttributeError, TypeError):  # Todo self.is_parent()
    #         raise AttributeError(f'The Structure "{self.name}" doesn\'t "own" it\'s coordinates. The attribute '
    #                              f'{self.residue_indexed_atom_indices.__name__} can only be accessed by the Structure '
    #                              f'object that owns these coordinates and therefore owns this Structure')

    @property
    def alphafold_atom_mask(self) -> np.ndarray:
        """Return an Alphafold mask describing which Atom positions have Coord data"""
        # Todo Fix naming errors in arginine residues where NH2 is incorrectly
        #  assigned to be closer to CD than NH1...
        # This works except for the off case that we sum to 0, which may be hard given float precision
        return (self.alphafold_coords.sum(axis=-1) != 0).astype(dtype=np.int32)

    @property
    def alphafold_coords(self) -> np.ndarray:
        """Return a view of the Coords from the StructureBase in Alphafold coordinate format"""
        try:
            return self._af_coords
        except AttributeError:
            # Todo Fix naming errors in arginine residues where NH2 is incorrectly
            #  assigned to be closer to CD than NH1...
            # af_coords = np.zeros((self.number_of_residues, atom_type_num, 3), dtype=np.float32)
            # residue_template_array = np.zeros((atom_type_num, 3), dtype=np.float32)
            structure_array = [[] for _ in range(self.number_of_residues)]
            origin_coord = [0., 0., 0.]
            residue_template_array = [origin_coord for _ in range(atom_type_num)]
            for idx, residue in enumerate(self.residues):
                residue_array = residue_template_array.copy()
                # Don't include any hydrogens
                for atom, xyz_coord in zip(residue.heavy_atoms, residue.heavy_coords.tolist()):
                    residue_array[atom_order[atom.type]] = xyz_coord
                structure_array[idx] = residue_array
                # af_coords[idx] = residue_array
            # self._af_coords = af_coords
            self._af_coords = np.array(structure_array, dtype=np.float32)

        return self._af_coords

    @property
    def _coords_indexed_residues(self) -> np.ndarray:  # Residues:
        try:
            return self._coords_indexed_residues_
        except AttributeError:
            # Index the coordinates to the Residue they belong to and their associated atom_index"""
            if self.is_dependent():
                raise stutils.DesignError(
                    f"Couldn't access '_coords_indexed_residues' for a dependent {repr(self)}"
                )
            # residues_atom_idx = [(residue, atom_idx) for residue in self.residues for atom_idx in residue.range]
            residues_by_atom = [residue for residue in self.residues for atom_idx in residue.range]
            if len(residues_by_atom) != len(self._atom_indices):
                raise ValueError(
                    f'The length of _coords_indexed_residues {len(residues_by_atom)} '
                    f'!= {len(self._atom_indices)}, the length of _atom_indices')
            # self._coords_indexed_residues_ = Residues(residues_by_atom)
            self._coords_indexed_residues_ = np.array(residues_by_atom)
            # self._coords_indexed_residue_atoms_ = np.array(residues_atom_idx)

            return self._coords_indexed_residues_

    @property
    def coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        return _struct._coords_indexed_residues[self._atom_indices].tolist()

    @property
    def backbone_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each backbone Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        return _struct._coords_indexed_residues[self.backbone_indices].tolist()

    @property
    def backbone_and_cb_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each backbone and CB Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        return _struct._coords_indexed_residues[self.backbone_and_cb_indices].tolist()

    @property
    def heavy_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each heavy Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        return _struct._coords_indexed_residues[self.heavy_indices].tolist()

    @property
    def side_chain_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each side chain Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        return _struct._coords_indexed_residues[self.side_chain_indices].tolist()

    # @property
    # def coords_indexed_residue_atoms(self) -> list[int]:
    #     """Returns a map of the Residue atom_indices for each Coord in the Structure
    #
    #     Returns:
    #         Index of the Atom position in the Residue for the index of the .coords attribute
    #     """
    #     if self.is_parent():
    #         return self._coords_indexed_residue_atoms[self._atom_indices].tolist()
    #     else:
    #         return self.parent._coords_indexed_residue_atoms[self._atom_indices].tolist()

    @property
    def number_of_residues(self) -> int:
        """Access the number of Residues in the Structure"""
        return len(self._residue_indices)

    def get_coords_subset(self, residue_numbers: Container[int] = None, indices: Iterable[int] = None,
                          start: int = None, end: int = None, dtype: stutils.coords_type_literal = 'ca') -> np.ndarray:
        """Return a view of a subset of the Coords from the Structure specified by a range of Residue numbers
        
        Args:
            residue_numbers: The Residue numbers to return
            indices: Residue indices of interest
            start: The Residue number to start at. Inclusive
            end: The Residue number to end at. Inclusive
            dtype: The type of coordinates to get
        Returns:
            The specific coordinates from the Residue instances with the specified dtype
        """
        if indices is None:
            if residue_numbers is None:
                if start is not None and end is not None:
                    residue_numbers = list(range(start, end + 1))
                else:
                    raise ValueError(
                        f"{self.get_coords_subset.__name__}: Must provide either 'indices', 'residue_numbers' or "
                        f"'start' and 'end'")
            residues = self.get_residues(residue_numbers)
        else:
            residues = self.get_residues(indices=indices)

        coords_type = 'coords' if dtype == 'all' else f'{dtype}_coords'
        return np.concatenate([getattr(residue, coords_type) for residue in residues])

    def set_residues_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Residues in the Structure

        Keyword Args:
            numbers: Container[int] = None - The Atom numbers of interest
            pdb: bool = False - Whether to search for numbers as they were parsed (if True)
        """
        for residue in self.get_residues(**kwargs):
            for kwarg, value in kwargs.items():
                setattr(residue, kwarg, value)

    # def set_residues_attributes_from_array(self, **kwargs):
    #     """Set attributes specified by key, value pairs for all Residues in the Structure"""
    #     # self._residues.set_attribute_from_array(**kwargs)
    #     for idx, residue in enumerate(self.residues):
    #         for key, value in kwargs.items():
    #             setattr(residue, key, value[idx])

    def get_residue_atom_indices(self, **kwargs) -> list[int]:
        """Returns Atom indices for Residue instances. Returns all by default unless numbers are specified, returns only
        those Residue instance selected by number.

        Keyword Args:
            numbers: Container[int] = None - The Residue numbers of interest
        """
        atom_indices = []
        for residue in self.get_residues(**kwargs):
            atom_indices.extend(residue.atom_indices)
        return atom_indices

    def get_residues_by_atom_indices(self, atom_indices: ArrayIndexer) -> list[Residue]:
        """Retrieve Residues in the Structure specified by Atom indices

        Args:
            atom_indices: The atom indices to retrieve Residue objects by
        Returns:
            The sorted, unique Residue instances corresponding to the provided atom_indices
        """
        if self.is_parent():
            _struct = self
        else:
            _struct = self.parent

        all_residues = _struct._coords_indexed_residues[atom_indices].tolist()
        return sorted(set(all_residues), key=lambda residue: residue.index)

    @property
    def backbone_indices(self) -> list[int]:
        """The indices that index the Structure backbone Atoms/Coords"""
        try:
            return self._backbone_indices
        except AttributeError:
            self._backbone_indices = []
            for residue in self.residues:
                self._backbone_indices.extend(residue.backbone_indices)
            return self._backbone_indices

    @property
    def backbone_and_cb_indices(self) -> list[int]:
        """The indices that index the Structure backbone and CB Atoms. Inherently gets CA of Residue instances missing
        CB
        """
        try:
            return self._backbone_and_cb_indices
        except AttributeError:
            self._backbone_and_cb_indices = []
            for residue in self.residues:
                self._backbone_and_cb_indices.extend(residue.backbone_and_cb_indices)
            return self._backbone_and_cb_indices

    @property
    def cb_indices(self) -> list[int]:
        """The indices that index the Structure CB Atoms. Inherently gets CA of Residue instances missing CB"""
        try:
            return self._cb_indices
        except AttributeError:
            self._cb_indices = [residue.cb_atom_index for residue in self.residues if residue.cb_atom_index]
            # self._cb_indices = [residue.cb_atom_index for residue in self.residues]
            return self._cb_indices

    @property
    def ca_indices(self) -> list[int]:
        """The indices that index the Structure CA Atoms/Coords"""
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = [residue.ca_atom_index for residue in self.residues]
            return self._ca_indices

    @property
    def heavy_indices(self) -> list[int]:
        """The indices that index the Structure heavy Atoms/Coords"""
        try:
            return self._heavy_indices
        except AttributeError:
            self._heavy_indices = []
            for residue in self.residues:
                self._heavy_indices.extend(residue.heavy_indices)
            return self._heavy_indices

    @property
    def side_chain_indices(self) -> list[int]:
        """The indices that index the Structure side chain Atoms/Coords"""
        try:
            return self._side_chain_indices
        except AttributeError:
            self._side_chain_indices = []
            for residue in self.residues:
                self._side_chain_indices.extend(residue.side_chain_indices)
            return self._side_chain_indices

    @property
    def helix_cb_indices(self) -> list[int]:
        """The indices that index the Structure helical CB Atoms/Coords"""
        try:
            return self._helix_cb_indices
        except AttributeError:
            h_cb_indices = []
            for residue in self.residues:
                if residue.secondary_structure == 'H':
                    h_cb_indices.append(residue.cb_atom_index)
            self._helix_cb_indices = h_cb_indices
            return self._helix_cb_indices

    def renumber(self):
        """Change the Atom and Residue numbering. Access the readtime Residue number in Residue.pdb_number attribute"""
        self.renumber_atoms()
        self.pose_numbering()

    def pose_numbering(self):
        """Change the Residue numbering to start at 1. Access the readtime Residue number in .pdb_number attribute"""
        for idx, residue in enumerate(self.residues, 1):
            residue.number = idx
        self.log.debug(f'{self.name} was formatted in Pose numbering (residues now 1 to {self.number_of_residues})')

    def renumber_residues(self, index: int = 0, at: int = 1):
        """Renumber Residue objects sequentially starting with "at"

        Args:
            index: The index to start the renumbering process
            at: The number to start renumbering at
        """
        for idx, residue in enumerate(self.residues[index:], at):
            residue.number = idx

    def get_residues(self, numbers: Container[int] = None, indices: Iterable[int] = None, **kwargs) -> list[Residue]:
        """Returns Residue instances as specified. Returns all by default

        Args:
            numbers: Residue numbers of interest
            indices: Residue indices of interest for the Structure
        Returns:
            The requested Residue instances, sorted in the order they appear in the Structure
        """
        residues = self.residues
        if numbers is not None:
            if isinstance(numbers, Container):
                residues = [residue for residue in residues if residue.number in numbers]
            else:
                self.log.warning(f'The passed numbers type "{type(numbers).__name__}" must be a Container. Returning '
                                 f'all Residue instances instead')
        elif indices is not None:
            try:
                residues = [residues[idx] for idx in indices]
            except IndexError:
                number_of_residues = self.number_of_residues
                for idx in indices:
                    if idx < 0 or idx >= number_of_residues:
                        raise IndexError(
                            f'The residue index {idx} is out of bounds for the {self.__class__.__name__} {self.name} '
                            f'with size of {number_of_residues} residues')
        return residues

    def _create_residues(self):
        """For the Structure, create Residue instances/Residues object. Doesn't allow for alternative atom locations

        Sets:
            self._atom_indices (list[int])
            self._residue_indices (list[int])
            self._residues (Residues)
        """
        new_residues, remove_atom_indices, found_types = [], [], set()
        atoms = self.atoms
        current_residue_number = atoms[0].residue_number
        start_atom_index = idx = 0
        for idx, atom in enumerate(atoms):
            # If the current residue number is the same as the prior number and the atom.type is not already present
            # We get rid of alternate conformations upon PDB load, so must be a new residue with bad numbering
            if atom.residue_number == current_residue_number and atom.type not in found_types:
                # atom_indices.append(idx)
                found_types.add(atom.type)
            else:
                if protein_backbone_atom_types.difference(found_types):  # Not an empty set, remove [start_idx:idx]
                    # Todo remove this check when nucleotides can be parsed
                    if dna_sugar_atom_types.intersection(found_types):
                        self.nucleotides_present = True
                    remove_atom_indices.extend(range(start_atom_index, idx))
                else:  # Proper format
                    new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx)), parent=self))
                start_atom_index = idx
                found_types = {atom.type}  # atom_indices = [idx]
                current_residue_number = atom.residue_number

        # ensure last residue is added after stop iteration
        if protein_backbone_atom_types.difference(found_types):  # Not an empty set, remove indices [start_idx:idx]
            remove_atom_indices.extend(range(start_atom_index, idx + 1))
        else:  # Proper format. For each need to increment one higher than the last v
            new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx + 1)), parent=self))

        self._residue_indices = list(range(len(new_residues)))
        self._residues = Residues(new_residues)
        # Set container dependent attributes since this is a fresh Residues instance
        self._residues.set_index()

        # Remove bad atom_indices
        atom_indices = self._atom_indices
        for index in remove_atom_indices[::-1]:  # Ensure popping happens in reverse
            atom_indices.pop(index)
        self._atom_indices = atom_indices
        # self._atoms.remove()

    # When alt_location parsing performed, there may be some use to below implementation
    # def _create_residues(self):
    #     """For the Structure, create all possible Residue instances. Doesn't allow for alternative atom locations"""
    #     start_indices, residue_ranges = [], []
    #     remove_atom_indices = []
    #     remove_indices = []
    #     new_residues = []
    #     atom_indices, found_types = [], set()
    #     atoms = self.atoms
    #     current_residue_number = atoms[0].residue_number
    #     start_atom_index = idx = 0
    #     for idx, atom in enumerate(atoms):
    #         # if the current residue number is the same as the prior number and the atom.type is not already present
    #         # We get rid of alternate conformations upon PDB load, so must be a new residue with bad numbering
    #         if atom.residue_number == current_residue_number and atom.type not in found_types:
    #             atom_indices.append(idx)
    #             found_types.add(atom.type)
    #         # if atom.residue_number == current_residue_number:  # current residue number the same as the prior number
    #         #     if atom.type not in found_types:  # the atom.type is not already present
    #         #         # atom_indices.append(idx)
    #         #         found_types.add(atom.type)
    #         #     else:  # atom is already present. We got rid of alternate conformations upon PDB load, so new residu
    #         #         remove_indices.append(idx)
    #         else:  # we are starting a new residue
    #             if protein_backbone_atom_types.difference(found_types):  # not an empty set, remove start idx to idx indice
    #                 remove_atom_indices.append(list(range(start_atom_index, idx)))  # remove_indices
    #             else:  # proper format
    #                 start_indices.append(start_atom_index)
    #                 residue_ranges.append(len(found_types))
    #                 # only add those indices that are duplicates was used without alternative conformations
    #                 remove_atom_indices.append(remove_indices)  # <- empty list
    #             # remove_indices = []
    #             start_atom_index = idx
    #             found_types = {atom.type}  # atom_indices = [idx]
    #             current_residue_number = atom.residue_number
    #
    #     # ensure last residue is added after stop iteration
    #     if protein_backbone_atom_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
    #         remove_atom_indices.append(atom_indices)
    #     else:  # proper format
    #         start_indices.append(start_atom_index)
    #         residue_ranges.append(len(found_types))
    #         # only add those indices that are duplicates was used without alternative conformations
    #         remove_atom_indices.append(remove_indices)  # <- empty list
    #
    #     # remove bad atoms and correct atom_indices
    #     # atom_indices = self._atom_indices
    #     atoms = self.atoms
    #     for indices in remove_atom_indices[::-1]:  # ensure popping happens in reverse
    #         for index in indices[::-1]:  # ensure popping happens in reverse
    #             atoms.pop(index)  # , atom_indices.pop(index)
    #
    #     self._atom_indices = list(range(len(atoms)))  # atom_indices
    #     self._atoms = atoms
    #
    #     for start_index, residue_range in zip(start_indices, residue_ranges):
    #         new_residues.append(Residue(atom_indices=list(range(start_atom_index, start_atom_index + residue_range)),
    #                                     atoms=self._atoms, coords=self._coords, log=self._log))
    #     self._residue_indices = list(range(len(new_residues)))
    #     self._residues = Residues(new_residues)

    def residue(self, residue_number: int) -> Residue | None:
        """Retrieve the specified Residue

        Args:
            residue_number: The number of the Residue to search for
        """
        for residue in self.residues:
            if residue.number == residue_number:
                return residue
        return None

    @property
    def n_terminal_residue(self) -> Residue:
        """Retrieve the Residue from the n-termini"""
        return self.residues[0]

    @property
    def c_terminal_residue(self) -> Residue:
        """Retrieve the Residue from the c-termini"""
        return self.residues[-1]

    def add_ideal_helix(self, termini: stutils.termini_literal = 'n', length: int = 5, alignment_length: int = 5):
        """Add an ideal helix to a termini given by a certain length
        
        Args:
            termini: The termini to add the ideal helix to
            length: The length of the addition, where viable values are [1-10] residues additions
            alignment_length: The number of residues used to calculation overlap of the target to the ideal helix
        """
        maximum_extension_length = 15 - alignment_length
        if length > maximum_extension_length:
            number_of_iterations, length = divmod(length, maximum_extension_length)
            # First add the remainder 'length' with the specified alignment length,
            self.add_ideal_helix(termini=termini, length=length, alignment_length=alignment_length)
            # Then perform 10 residue extensions until complete
            # Using the default 10 and 5 alignment to prevent any errors from alignment of shorter amount from
            # propagating to the ideal addition and creating ideal "kinks"
            addition_count = count()
            while next(addition_count) != number_of_iterations:
                self.add_ideal_helix(termini=termini, length=10)

            return
        elif length < 1:
            return
        else:
            self.log.debug(f'Adding {length} residue ideal helix to {termini}-terminus of {self.name}')

        alpha_helix_15_struct = ContainsResidues.from_atoms(alpha_helix_15_atoms)

        if termini == 'n':
            residue = self.n_terminal_residue
            residue_index = residue.index
            fixed_coords = self.get_coords_subset(
                indices=list(range(residue_index, residue_index + alignment_length)), dtype='backbone')

            ideal_end_index = alpha_helix_15_struct.c_terminal_residue.index + 1
            ideal_start_index = ideal_end_index - alignment_length
            moving_coords = alpha_helix_15_struct.get_coords_subset(
                indices=list(range(ideal_start_index, ideal_end_index)), dtype='backbone')
            rmsd, rot, tx = superposition3d(fixed_coords, moving_coords)
            alpha_helix_15_struct.transform(rotation=rot, translation=tx)

            # Add residues
            helix_add_start = ideal_start_index - length
            # Exclude ideal_start_index from residue selection
            add_residues = alpha_helix_15_struct.get_residues(indices=list(range(helix_add_start, ideal_start_index)))
        elif termini == 'c':
            residue = self.c_terminal_residue
            residue_index = residue.index + 1
            fixed_coords = self.get_coords_subset(
                indices=list(range(residue_index - alignment_length, residue_index)), dtype='backbone')

            ideal_start_index = alpha_helix_15_struct.n_terminal_residue.index
            ideal_end_index = ideal_start_index + alignment_length
            moving_coords = alpha_helix_15_struct.get_coords_subset(
                indices=list(range(ideal_start_index, ideal_end_index)), dtype='backbone')
            rmsd, rot, tx = superposition3d(fixed_coords, moving_coords)
            alpha_helix_15_struct.transform(rotation=rot, translation=tx)

            # Add residues
            helix_add_end = ideal_end_index + length
            # Leave out the residue with ideal_end_index, and include the helix_add_end number
            add_residues = alpha_helix_15_struct.get_residues(indices=list(range(ideal_end_index, helix_add_end)))
        else:
            raise ValueError(
                f"'termini' must be either 'n' or 'c', not {termini}")

        self.insert_residues(residue_index, add_residues, chain_id=residue.chain_id)

    def get_residue_atoms(self, **kwargs) -> list[Atom]:
        """Return the Atoms contained in the Residue objects matching a set of residue numbers

        Keyword Args:
            numbers: Container[int] = None – Residue numbers of interest
            indices: Iterable[int] = None – Residue indices of interest for the Structure
        Returns:
            The Atom instances belonging to the Residue instances
        """
        atoms = []
        for residue in self.get_residues(**kwargs):
            atoms.extend(residue.atoms)
        return atoms

    def mutate_residue(self, residue: Residue = None, index: int = None, number: int = None, to: str = 'A', **kwargs) \
            -> list[int] | list:
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Args:
            residue: A Residue instance to mutate
            index: A Residue index to select the Residue instance of interest
            number: A Residue number to select the Residue instance of interest
            to: The type of amino acid to mutate to
        Returns:
            The indices of the Atoms being removed from the Structure
        """
        if index is not None:
            try:
                residue = self.residues[index]
            except IndexError:
                raise IndexError(
                    f'The residue index {index} is out of bounds for the {self.__class__.__name__} '
                    f'{self.name} with size of {self.number_of_residues} residues')
        elif number is not None:
            residue = self.residue(number)

        if residue is None:
            raise ValueError(
                f"Can't {self.mutate_residue.__name__} without Residue instance, index, or number")
        elif self.is_dependent():
            _parent = self.parent
            self.log.debug(f"{self.mutate_residue.__name__} can't be performed on a dependent StructureBase. "
                           f"Calling on the {self.__class__.__name__}.parent {repr(_parent)}")
            # Ensure the deletion is done by the Structure parent to account for everything correctly
            return _parent.mutate_residue(residue, to=to)

        to = protein_letters_1to3.get(to.upper(), to.upper())
        if residue.type == to:  # No mutation necessary
            return []
        else:
            try:
                protein_letters_3to1_extended[to]
            except KeyError:
                raise KeyError(
                    f"The mutation type '{to}' isn't a viable Residue type")

        # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
        self.log.debug(f'Mutating {residue.type}{residue.number}{to}')
        residue.type = to
        # Todo is the Atom mutation necessary? Put in Residue
        for atom in residue.atoms:
            atom.residue_type = to

        # Todo Currently, deleting side-chain indices and letting Rosetta handle building
        # Find the corresponding Residue Atom indices to delete
        delete_indices = residue.side_chain_indices
        if not delete_indices:  # There are no indices
            return []
        else:
            # Clear all state variables for all Residue instances
            # Todo create mutate_residues() and only call this once...
            #  It is redundant with @Residue.start_index.setter in _residues.reindex_atoms()
            self._residues.reset_state()
            # residue.side_chain_indices = []

        # Remove indices from the Residue, and Structure atom_indices
        residue_atom_indices = residue.atom_indices
        residue_atom_delete_index = residue_atom_indices.index(delete_indices[0])
        _atom_indices = self._atom_indices
        structure_atom_delete_index = _atom_indices.index(delete_indices[0])
        for _ in iter(delete_indices):
            residue_atom_indices.pop(residue_atom_delete_index)
            _atom_indices.pop(structure_atom_delete_index)

        self._offset_indices(start_at=structure_atom_delete_index, offset=-len(delete_indices), dtype='atom')

        # Re-index all succeeding Atom and Residue instance indices
        self._coords.delete(delete_indices)
        self._atoms.delete(delete_indices)
        # self._atoms.reindex(start_at=structure_atom_delete_index)
        self._residues.reindex_atoms(start_at=residue.index)

        # Reissue the atom assignments for the Residue
        residue.delegate_atoms()
        self.reset_state()

        return delete_indices

    def delete_residues(self, residues: Iterable[Residue] = None, indices: Iterable[int] = None,
                        numbers: Container[int] = None, **kwargs) -> list[Residue] | list:
        """Deletes Residue instances from the Structure

        Args:
            residues: Residue instances to delete
            indices: Residue indices to select the Residue instances of interest
            numbers: Residue numbers to select the Residue instances of interest
        Returns:
            Each deleted Residue
        """
        if indices is not None:
            residues = self.get_residues(indices=indices)
        elif numbers is not None:
            residues = self.get_residues(numbers=numbers)

        if residues is None:
            raise ValueError(
                f"Can't {self.delete_residues.__name__} without Residue instances. Provide with indices, numbers, or "
                "residues")
        elif not residues:
            self.log.debug(f'{self.delete_residues.__name__}: No residues found')
            return []
        elif self.is_dependent():  # Call on the parent
            _parent = self.parent
            self.log.debug(f"{self.delete_residues.__name__} can't be performed on a dependent StructureBase. "
                           f"Calling on the {self.__class__.__name__}.parent {repr(_parent)}")
            # Ensure the deletion is done by the Structure parent to account for everything correctly
            return _parent.delete_residues(residues=residues)

        # Find the Residue, Atom indices to delete
        atom_indices = []
        for residue in residues:
            self.log.debug(f'Deleting {residue.type}{residue.number}')
            atom_indices.extend(residue.atom_indices)

        if not atom_indices:
            return []  # There are no indices for the Residue instances
        else:  # Find the Residue indices to delete
            residue_indices = []
            for residue in residues:
                residue_indices.append(residue.index)

        # Remove indices from the Residue, and Structure atom_indices
        self._delete_indices(atom_indices, dtype='atom')
        self._delete_indices(residue_indices, dtype='residue')
        # Re-index all succeeding Atom and Residue instance indices
        self._coords.delete(atom_indices)
        self._atoms.delete(atom_indices)
        self._residues.delete(residue_indices)
        self._residues.reindex(start_at=residue_indices[0])

        # Clear state variables for remaining Residue instances. Residue deletion affected remaining attrs and indices
        self._residues.reset_state()
        # Reindex the coords/residues map
        self.reset_state()

        return residues

    def insert_residue_type(self, index: int, residue_type: str, chain_id: str = None) -> Residue:
        """Insert a standard Residue type into the Structure at the origin. No structural alignment is performed!

        Args:
            index: The residue index where a new Residue should be inserted into the Structure
            residue_type: Either the 1 or 3 letter amino acid code for the residue in question
            chain_id: The chain identifier to associate the new Residue with
        Returns:
            The newly inserted Residue object
        """
        if self.is_dependent():  # Call on the parent
            _parent = self.parent
            self.log.debug(f"{self.insert_residue_type.__name__} can't be performed on a dependent StructureBase. "
                           f"Calling on the {self.__class__.__name__}.parent {repr(_parent)}")
            # Ensure the deletion is done by the Structure parent to account for everything correctly
            if chain_id is None:
                chain_id = getattr(self, 'chain_id', None)
            return _parent.insert_residue_type(index, residue_type, chain_id=chain_id)

        self.log.debug(f'Inserting {residue_type} into index {index} of {self.name}')

        # Convert incoming amino acid to an index to select the stutils.reference_residue.
        # protein_letters_alph1 has a matching index
        reference_index = \
            protein_letters_alph1.find(protein_letters_3to1_extended.get(residue_type, residue_type.upper()))
        if reference_index == -1:
            raise IndexError(
                f"{self.insert_residue_type.__name__} of residue_type '{residue_type}' isn't allowed")
        if index < 0:
            raise IndexError(
                f"{self.insert_residue_type.__name__} at index {index} < 0 isn't allowed")

        # Grab the reference atom coordinates and push into the atom list
        new_residue = reference_residues[reference_index].copy()

        # Find the prior and next Residue, atom_start_index (starting atom in new Residue index)
        residues = self.residues
        if index == 0:  # n-termini = True
            prev_residue = None
            atom_start_index = 0
        else:
            prev_residue = residues[index - 1]
            atom_start_index = prev_residue.end_index + 1

        try:
            next_residue = residues[index]
        except IndexError:  # c_termini = True
            next_residue = None
            if not prev_residue:  # Insertion on an empty Structure? block for now to simplify chain identification
                raise stutils.DesignError(
                    f"Can't {self.insert_residue_type.__name__} for an empty {self.__class__.__name__} class")

        # Set found attributes
        new_residue._insert = True

        # Insert the new_residue coords, atoms, and residues into the Structure
        self._coords.insert(atom_start_index, new_residue.coords)
        self._atoms.insert(atom_start_index, new_residue.atoms)
        self._residues.insert(index, [new_residue])
        # After coords, atoms, residues insertion into "_" containers, set parent to self
        new_residue._parent = self

        # Reformat indices
        # new_residue.start_index = atom_start_index
        # self._atoms.reindex(start_at=atom_start_index)  # Called in self._residues.reindex
        self._residues.reindex(start_at=index)  # .set_index()
        # Insert new_residue index and atom_indices into Structure indices
        self._insert_indices(index, [index], dtype='residue')
        self._insert_indices(atom_start_index, new_residue.atom_indices, dtype='atom')

        # Set the new chain_id. Must occur after self._residue_indices update if chain isn't provided
        if chain_id is None:  # Try to solve without it...
            if prev_residue and next_residue:
                if prev_residue.chain_id == next_residue.chain_id:
                    chain_id = prev_residue.chain_id
                else:  # There is a discrepancy which means this is an internal termini
                    raise stutils.DesignError(chain_assignment_error)
            # This can be solved as it represents an absolute termini case
            elif prev_residue:
                chain_id = prev_residue.chain_id
            else:
                chain_id = next_residue.chain_id
        new_residue.chain_id = chain_id

        # Solve the Residue number and renumber Structure if there is overlap
        if prev_residue and next_residue:
            tentative_number = prev_residue.number + 1
            if tentative_number == next_residue.number:
                # There is a conflicting insertion
                # The prev_residue could also be inserted and needed to be numbered lower
                try:  # Check residue._insert
                    prev_residue._insert
                except AttributeError:  # Not inserted. Renumber all subsequent
                    # self.renumber_residues()
                    residues_renumber = residues[index:]
                    for residue in residues_renumber:
                        residue.number = residue.number + 1
                else:
                    for residue in prev_residue.get_upstream() + [prev_residue]:
                        residue.number = residue.number - 1

            new_residue.number = tentative_number
        elif prev_residue:
            new_residue.number = prev_residue.number + 1
        else:  # next_residue
            # This cautionary note may not apply anymore
            #  Subtracting one may not be enough if this insert_residue_type() is part of a set of inserts and all
            #  n-terminal insertions are being conducted before this next_residue. Clean this in the first check of
            #  this logic block
            new_residue.number = next_residue.number - 1

        try:
            secondary_structure = self._secondary_structure
        except AttributeError:  # When not set yet
            self.calculate_secondary_structure()  # The new_residue will be included
        else:  # Insert the new ss with a coiled assumption
            # ASSUME the insertion is disordered and coiled segment
            new_residue.secondary_structure = DEFAULT_SS_COIL_IDENTIFIER
            self._secondary_structure = \
                secondary_structure[:index] + DEFAULT_SS_COIL_IDENTIFIER + secondary_structure[index:]

        # Reindex the coords/residues map
        self.reset_state()

        return new_residue

    def insert_residues(self, index: int, new_residues: Iterable[Residue], chain_id: str = None) -> list[Residue]:
        """Insert Residue instances into the Structure at the origin. No structural alignment is performed!

        Args:
            index: The index to perform the insertion at
            new_residues: The Residue instances to insert
            chain_id: The chain identifier to associate the new Residue instances with
        Returns:
            The inserted Residue instances
        """
        if not new_residues:
            return []

        if self.is_dependent():  # Call on the parent
            _parent = self.parent
            self.log.debug(f"{self.insert_residues.__name__} can't be performed on a dependent StructureBase. "
                           f"Calling on the {self.__class__.__name__}.parent {repr(_parent)}")
            # Ensure the deletion is done by the Structure parent to account for everything correctly
            if chain_id is None:
                chain_id = getattr(self, 'chain_id', None)
            return _parent.insert_residues(index, new_residues, chain_id=chain_id)

        # Make a copy of the Residue instances
        new_residues = [residue.copy() for residue in new_residues]
        number_new_residues = len(new_residues)

        # Find the prior and next Residue, atom_start_index (starting atom in new Residue index)
        residues = self.residues
        if index == 0:  # n-termini = True
            prev_residue = _prev_residue = None
            atom_start_index = 0
        else:
            prev_residue = _prev_residue = residues[index - 1]
            atom_start_index = prev_residue.end_index + 1

        try:
            next_residue = residues[index]
        except IndexError:  # c_termini = True
            next_residue = None
            if not prev_residue:  # Insertion on an empty Structure? block for now to simplify chain identification
                raise stutils.DesignError(
                    f"Can't {self.insert_residue_type.__name__} for an empty {self.__class__.__name__} class")

        # Set found attributes
        # prev_residue, *other_residues = new_residues
        for residue in new_residues:
            residue._insert = True

        # Insert the new_residue coords, atoms, and residues into the Structure
        self._coords.insert(atom_start_index, np.concatenate([residue.coords for residue in new_residues]))
        self._atoms.insert(atom_start_index, [atom for residue in new_residues for atom in residue.atoms])
        self._residues.insert(index, new_residues)
        # After coords, atoms, residues insertion into "_" containers, set parent to self
        for residue in new_residues:
            residue._parent = self

        # Reformat indices
        # new_residue.start_index = atom_start_index
        # self._atoms.reindex(start_at=atom_start_index)  # Called in self._residues.reindex
        self._residues.reindex(start_at=index)  # .set_index()
        # Insert new_residue index and atom_indices into Structure indices
        new_residue_atom_indices = list(range(atom_start_index, new_residues[-1].end_index + 1))
        self._insert_indices(atom_start_index, new_residue_atom_indices, dtype='atom')
        new_residue_indices = list(range(index, index + number_new_residues))
        self._insert_indices(index, new_residue_indices, dtype='residue')

        # Set the new chain_id. Must occur after self._residue_indices update if chain isn't provided
        if chain_id is None:  # Try to solve without it...
            if prev_residue and next_residue:
                if prev_residue.chain_id == next_residue.chain_id:
                    chain_id = prev_residue.chain_id
                else:  # There is a discrepancy which means this is an internal termini
                    raise stutils.DesignError(chain_assignment_error)
            # This can be solved as it represents an absolute termini case
            elif prev_residue:
                chain_id = prev_residue.chain_id
            else:
                chain_id = next_residue.chain_id
        for residue in new_residues:
            residue.chain_id = chain_id

        # Solve the Residue number and renumber Structure if there is overlap
        if prev_residue and next_residue:
            first_number = prev_residue.number + 1
            tentative_residue_numbers = list(range(first_number, first_number + number_new_residues))
            if next_residue.number in tentative_residue_numbers:
                # There is a conflicting insertion. Correct existing residue numbers
                # The prev_residue could also be inserted and needed to be numbered lower
                try:  # Check residue._insert
                    prev_residue._insert
                except AttributeError:  # Not inserted. Renumber all subsequent
                    # self.renumber_residues()
                    residues_renumber = residues[index:]
                    for residue in residues_renumber:
                        residue.number = residue.number + number_new_residues
                else:
                    for residue in prev_residue.get_upstream() + [prev_residue]:
                        residue.number = residue.number - number_new_residues
            # Set the new numbers
            for residue, new_number in zip(new_residues, tentative_residue_numbers):
                residue.number = new_number
        elif prev_residue:
            _prev_residue = prev_residue
            for residue in new_residues:
                residue.number = _prev_residue.number + 1
                _prev_residue = residue
        else:  # next_residue
            # This cautionary note may not apply to insert_residues(
            #  Subtracting one may not be enough if insert_residues() is part of a set of inserts and all
            #  n-terminal insertions are being conducted before this next_residue. Clean this in the first check of
            #  this logic block
            _prev_residue = next_residue
            for residue in reversed(new_residues):
                residue.number = _prev_residue.number - 1
                _prev_residue = residue

        # Reindex the coords/residues map
        self.reset_state()

        return new_residues

    def delete_termini(self, how: str = 'unstructured', termini: stutils.termini_literal = None):
        """Remove Residue instances from the Structure termini that are not helices

        Uses the default secondary structure prediction program's SS_HELIX_IDENTIFIERS (typically 'H') to search for
        non-conforming secondary structure

        Args:
            how: How should termini be trimmed? Either 'unstructured' (default) or 'to_helices' can be used.
                If 'unstructured',
                    will use 'SS_DISORDER_IDENTIFIERS' (typically coil) to detect disorder. Function will then remove any
                    disordered segments, as well as turns ('T') that exist between disordered segments
                If 'to_helices',
                    will use 'SS_HELIX_IDENTIFIERS' (typically 'H') to remove any non-conforming secondary structure
                    elements until a helix is reached
            termini: If a specific termini should be targeted, which one?
        """
        if termini is None or termini.lower() == 'nc':
            termini_ = 'nc'
        elif termini in 'NnCc':  # Only one of the two
            termini_ = termini.lower()
        else:
            raise ValueError(
                f"'termini' must be one of 'n' or 'c', not '{termini}")

        secondary_structure = working_secondary_structure = self.secondary_structure
        number_of_residues = self.number_of_residues
        # no_nterm_disorder_ss = secondary_structure.lstrip(SS_DISORDER_IDENTIFIERS)
        # n_removed_nterm_res = number_of_residues - len(no_nterm_disorder_ss)
        # no_cterm_disorder_ss = secondary_structure.rstrip(SS_DISORDER_IDENTIFIERS)
        # n_removed_cterm_res = number_of_residues - len(no_cterm_disorder_ss)
        # Todo
        #  Could remove disorder by a relative_sasa threshold. A brief investigation shows that ~0.6 could be a
        #  reasonable threshold when combined with other ss indicators
        # sasa = self.relative_sasa
        # self.log.debug(f'Found n-term relative sasa {sasa[:n_removed_nterm_res + 10]}')
        # self.log.debug(f'Found c-term relative sasa {sasa[-(n_removed_cterm_res + 10):]}')

        n_removed_nterm_res = 0
        # Remove coils. Find the next coil. If only ss present is (T)urn, then remove that as well and start again
        for idx, termini in enumerate('NC'):
            if idx == 0:  # n-termini
                self.log.debug(f'Starting N-term is: {working_secondary_structure[:15]}')
                possible_secondary_structure = working_secondary_structure
            else:  # c-termini
                self.log.debug(f'N-term is: {working_secondary_structure[:15]}')
                # Get the number of n-termini removed
                n_removed_nterm_res = number_of_residues - len(working_secondary_structure)
                # Reverse the sequence to get the c-termini first
                possible_secondary_structure = working_secondary_structure[::-1]
                self.log.debug(f'Starting C-term (reversed) is: {possible_secondary_structure[:15]}')

            if how == 'to_helices':
                ss_helix_index = possible_secondary_structure.find(SS_HELIX_IDENTIFIERS)
                working_secondary_structure = working_secondary_structure[ss_helix_index:]
            else:  # how == 'unstructured'
                ss_disorder_index = possible_secondary_structure.find(SS_DISORDER_IDENTIFIERS)
                while ss_disorder_index == 0:  # Go again
                    # Remove DISORDER ss
                    working_secondary_structure = possible_secondary_structure.lstrip(SS_DISORDER_IDENTIFIERS)
                    # Next try to remove TURN ss. Only remove if it is between DISORDER segments
                    possible_secondary_structure = working_secondary_structure.lstrip(SS_TURN_IDENTIFIERS)
                    ss_disorder_index = possible_secondary_structure.find(SS_DISORDER_IDENTIFIERS)

        self.log.debug(f'C-term (reversed) is: {working_secondary_structure[:15]}')

        residues = self.residues
        _delete_residues = []
        context_length = 10
        if 'n' in termini_ and n_removed_nterm_res:
            old_ss = secondary_structure[:n_removed_nterm_res + context_length]
            self.log.debug(f'Found N-term secondary_structure {secondary_structure[:n_removed_nterm_res + 5]}')
            self.log.info(f"Removing {n_removed_nterm_res} N-terminal residues. Resulting secondary structure:\n"
                          f"\told : {old_ss}...\n"
                          f"\tnew : {'-' * n_removed_nterm_res}{old_ss[n_removed_nterm_res:]}...")
            _delete_residues += residues[:n_removed_nterm_res]

        # Get the number of c-termini removed
        n_removed_cterm_res = number_of_residues - len(working_secondary_structure) - n_removed_nterm_res
        if 'c' in termini_ and n_removed_cterm_res:
            c_term_index = number_of_residues - n_removed_cterm_res
            old_ss = secondary_structure[c_term_index - context_length:]
            self.log.debug(f'Found C-term secondary_structure {secondary_structure[-(n_removed_cterm_res + 5):]}')
            self.log.info(f"Removing {n_removed_cterm_res} C-terminal residues. Resulting secondary structure:\n"
                          f"\told :...{old_ss}\n"
                          f"\tnew :...{old_ss[:-n_removed_cterm_res]}{'-' * n_removed_cterm_res}")
            _delete_residues += residues[c_term_index:]

        self.delete_residues(_delete_residues)

    def local_density(self, residues: list[Residue] = None, residue_numbers: list[int] = None, distance: float = 12.) \
            -> list[float]:
        """Return the number of Atoms within 'distance' Angstroms of each Atom in the requested Residues

        Args:
            residues: The Residues to include in the calculation
            residue_numbers: The numbers of the Residues to include in the calculation
            distance: The cutoff distance with which Atoms should be included in local density
        Returns:
            An array like containing the local density around each requested Residue
        """
        if residues:
            coords = []
            for residue in residues:
                coords.extend(residue.heavy_coords)
            coords_indexed_residues = [residue for residue in residues for _ in residue.heavy_indices]
        elif residue_numbers:
            coords = []
            residues = self.get_residues(numbers=residue_numbers)
            for residue in residues:
                coords.extend(residue.heavy_coords)
            coords_indexed_residues = [residue for residue in residues for _ in residue.heavy_indices]
        else:  # use all Residue instances
            residues = self.residues
            coords = self.heavy_coords
            coords_indexed_residues = self.heavy_coords_indexed_residues

        # in case this was already called, we should set all to 0.
        if residues[0].local_density > 0:
            for residue in residues:
                residue.local_density = 0.

        all_atom_tree = BallTree(coords)
        all_atom_counts_query = all_atom_tree.query_radius(coords, distance, count_only=True)
        # residue_neighbor_counts, current_residue = 0, coords_indexed_residues[0]
        current_residue = coords_indexed_residues[0]
        for residue, atom_neighbor_counts in zip(coords_indexed_residues, all_atom_counts_query.tolist()):
            if residue == current_residue:
                current_residue.local_density += atom_neighbor_counts
            else:  # We have a new residue, find the average
                current_residue.local_density /= current_residue.number_of_heavy_atoms
                current_residue = residue
                current_residue.local_density += atom_neighbor_counts
        # Ensure the last residue is calculated
        current_residue.local_density /= current_residue.number_of_heavy_atoms  # Find the average

        return [residue.local_density for residue in self.residues]

    def is_clash(self, measure: stutils.coords_type_literal = stutils.default_clash_criteria,
                 distance: float = stutils.default_clash_distance,
                 warn: bool = False, silence_exceptions: bool = False,
                 report_hydrogen: bool = False) -> bool:
        """Check if the Structure contains any self clashes. If clashes occur with the Backbone, return True. Reports
        the Residue where the clash occurred and the clashing Atoms

        Args:
            measure: The atom type to measure clashing by
            distance: The distance which clashes should be checked
            warn: Whether to emit warnings about identified clashes. Output grouped into measure vs non-measure
            silence_exceptions: Whether to silence the raised ClashError and Return True instead
            report_hydrogen: Whether to report clashing hydrogen atoms
        Returns:
            True if the Structure clashes, False if not
        Raises:
            ClashError if the Structure has an identified clash
        """
        if measure == 'backbone_and_cb':
            other = 'non-cb sidechain'
        elif measure == 'heavy':
            other = 'hydrogen'
            report_hydrogen = True
        elif measure == 'backbone':
            other = 'sidechain'
        elif measure == 'cb':
            other = 'non-cb'
        elif measure == 'ca':
            other = 'non-ca'
        else:  # measure == 'all'
            other = 'solvent'  # this should never appear unless someone added solvent parsing

        coords_type = 'coords' if measure == 'all' else f'{measure}_coords'
        # cant use heavy_coords as the Residue.atom_indices aren't offset for the BallTree made from them...
        # another option is to only care about heavy atoms on the residues...
        # if self.contains_hydrogen():
        #     atom_tree = BallTree(self.heavy_coords)
        #     coords_indexed_residues = self.heavy_coords_indexed_residues
        #     atoms = self.heavy_atoms
        # else:

        # Set up the query indices. BallTree is faster upon timeit with 131 msec/loop
        atom_tree = BallTree(self.coords)
        atoms = self.atoms
        measured_clashes, other_clashes = [], []

        def return_true(): return True

        clashes = False
        _any_clashes: Callable[[Iterable[int]], bool]
        """Local helper to separate clash reporting from clash generation"""
        clash_msg = f'{self.name} contains Residue {measure} atom clashes at a {distance} A distance'
        if warn:

            def _any_clashes(_clash_indices: Iterable[int]) -> bool:
                new_clashes = any(_clash_indices)
                if new_clashes:
                    for clashing_idx in _clash_indices:
                        atom = atoms[clashing_idx]
                        if getattr(atom, f'is_{measure}', return_true)():
                            measured_clashes.append((residue, atom))
                        elif report_hydrogen:  # Report all clashes, no need to check
                            other_clashes.append((residue, atom))
                        elif atom.is_heavy():  # Check if atom is a heavy atom then report if it is
                            other_clashes.append((residue, atom))

                return clashes or new_clashes

            # Using _any_clashes to set the global clashes
            #     clashes = _any_clashes(): ... return clashes
            # While checking global clashes against new_clashes
        else:  # Raise a ClashError to defer to caller
            def _any_clashes(_clash_indices: Iterable[int]) -> bool:
                for clashing_idx in _clash_indices:
                    if getattr(atoms[clashing_idx], f'is_{measure}', return_true)():
                        raise stutils.ClashError(clash_msg)

                return clashes

        residues = self.residues
        # Check first and last residue with different considerations given covalent bonding
        try:
            residue, next_residue = residues[:2]
            # residue, next_residue, *other_residues = self.residues
        except ValueError:  # Can't unpack
            # residues < 2. Insufficient to check clashing
            return False

        # Query each residue with requested coords_type against the atom_tree
        residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
        # residue_atom_contacts returns as ragged nested array, (array of different sized array)
        # Reduce the dimensions to all contacts
        all_contacts = {atom_contact for residue_contacts in residue_atom_contacts.tolist()
                        for atom_contact in residue_contacts.tolist()}
        try:
            # Subtract the N and C atoms from the adjacent residues for each residue as these are within a bond
            clashes = _any_clashes(
                all_contacts.difference(residue.atom_indices + [next_residue.n_atom_index]))
            prev_residue = residue
            residue = next_residue

            # Perform routine for all middle residues
            for next_residue in residues[2:]:
                residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
                all_contacts = {atom_contact for residue_contacts in residue_atom_contacts.tolist()
                                for atom_contact in residue_contacts.tolist()}
                clashes = _any_clashes(
                    all_contacts.difference(
                        [prev_residue.o_atom_index, prev_residue.c_atom_index, next_residue.n_atom_index]
                        + residue.atom_indices
                    ))
                prev_residue = residue
                residue = next_residue

            residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
            all_contacts = {atom_contact for residue_contacts in residue_atom_contacts.tolist()
                            for atom_contact in residue_contacts.tolist()}
            clashes = _any_clashes(
                all_contacts.difference([prev_residue.o_atom_index, prev_residue.c_atom_index]
                                        + residue.atom_indices)
            )
        except stutils.ClashError as error:  # Raised by _any_clashes()
            if silence_exceptions:
                return True
            else:
                raise error
        else:
            if clashes:
                if measured_clashes:
                    bb_info = '\n\t'.join(f'Chain {residue.chain_id} {residue.number:5d}: {atom.get_atom_record()}'
                                          for residue, atom in measured_clashes)
                    self.log.error(f'{self.name} contains {len(measured_clashes)} {measure} clashes from the following '
                                   f'Residues to the corresponding Atom:\n\t{bb_info}')
                    raise stutils.ClashError(clash_msg)
                if other_clashes:
                    sc_info = '\n\t'.join(f'Chain {residue.chain_id} {residue.number:5d}: {atom.get_atom_record()}'
                                          for residue, atom in other_clashes)
                    self.log.warning(f'{self.name} contains {len(other_clashes)} {other} clashes between the '
                                     f'following Residues:\n\t{sc_info}')
        return False

    def get_sasa(self, probe_radius: float = 1.4, atom: bool = True, **kwargs):  # Todo to Residue too? ContainsAtomsMix
        """Use FreeSASA to calculate the surface area of residues in the Structure object.

        Args:
            probe_radius: The radius which surface area should be generated
            atom: Whether the output should be generated for each atom. If False, will be generated for each Residue
        Sets:
            self.sasa, self.residue(s).sasa
        """
        if atom:
            out_format = 'pdb'
        # --format=pdb --depth=atom
        # REMARK 999 This PDB file was generated by FreeSASA 2.0.
        # REMARK 999 In the ATOM records temperature factors have been
        # REMARK 999 replaced by the SASA of the atom, and the occupancy
        # REMARK 999 by the radius used in the calculation.
        # MODEL        1                                        [radii][sasa]
        # ATOM   2557  C   PHE C 113      -2.627 -17.654  13.108  1.61  1.39
        # ATOM   2558  O   PHE C 113      -2.767 -18.772  13.648  1.42 39.95
        # ATOM   2559  CB  PHE C 113      -1.255 -16.970  11.143  1.88 13.46
        # ATOM   2560  CG  PHE C 113      -0.886 -17.270   9.721  1.61  1.98
        # ATOM   2563 CE1  PHE C 113      -0.041 -18.799   8.042  1.76 28.76
        # ATOM   2564 CE2  PHE C 113      -0.694 -16.569   7.413  1.76  2.92
        # ATOM   2565  CZ  PHE C 113      -0.196 -17.820   7.063  1.76  4.24
        # ATOM   2566 OXT  PHE C 113      -2.515 -16.590  13.750  1.46 15.09
        # ...
        # TER    7913      GLU A 264
        # ENDMDL EOF
        # if residue:
        else:
            out_format = 'seq'
        # --format=seq
        # Residues in ...
        # SEQ A    1 MET :   74.46
        # SEQ A    2 LYS :   96.30
        # SEQ A    3 VAL :    0.00
        # SEQ A    4 VAL :    0.00
        # SEQ A    5 VAL :    0.00
        # SEQ A    6 GLN :    0.00
        # SEQ A    7 ILE :    0.00
        # SEQ A    8 LYS :    0.87
        # SEQ A    9 ASP :    1.30
        # SEQ A   10 PHE :   64.55
        # ...
        # \n EOF
        if self.contains_hydrogen():
            include_hydrogen = ['--hydrogen']  # the addition of hydrogen changes results quite a bit
        else:
            include_hydrogen = []
        cmd = [putils.freesasa_exe_path, f'--format={out_format}', '--probe-radius', str(probe_radius),
               '-c', putils.freesasa_config_path, '--n-threads=2'] + include_hydrogen
        self.log.debug(f'FreeSASA:\n{subprocess.list2cmdline(cmd)}')
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=self.get_atom_record().encode('utf-8'))
        # if err:  # usually results from Hydrogen atoms, silencing
        #     self.log.warning('\n%s' % err.decode('utf-8'))
        sasa_output = out.decode('utf-8').split('\n')
        if_idx = 0
        if atom:
            # slice removes first REMARK, MODEL and final TER, MODEL regardless of # of chains, TER inclusion
            # since return_atom_record doesn't have models, these won't be present and no option to freesasa about model
            # would be provided with above subprocess call
            atoms = self.atoms
            for line_split in map(str.split, sasa_output[5:-2]):  # Does slice remove need for if line[0] == 'ATOM'?
                if line_split[0] == 'ATOM':  # This line appears necessary as MODEL can be added if MODEL is written
                    atoms[if_idx].sasa = float(line_split[-1])
                    if_idx += 1
        else:
            seq_slice = slice(3)
            sasa_slice = slice(16, None)
            residues = self.residues
            for idx, line in enumerate(sasa_output[1:-1]):  # Does slice remove the need for if line[:3] == 'SEQ'?
                if line[seq_slice] == 'SEQ':  # Doesn't seem that this equality is sufficient ^
                    residues[if_idx].sasa = float(line[sasa_slice])
                    if_idx += 1
        try:
            self.sasa = sum([residue.sasa for residue in self.residues])
        except RecursionError:
            self.log.error('RecursionError measuring SASA')
            os.makedirs(putils.sasa_debug_dir, exist_ok=True)
            self.write(out_path=os.path.join(putils.sasa_debug_dir, f'SASA-INPUT-{self.name}.pdb'))
            with open(os.path.join(putils.sasa_debug_dir, f'SASA-OUTPUT-{self.name}.pdb'), 'w') as f:
                f.write('%s\n' % '\n'.join(sasa_output))

            raise stutils.DesignError(
                "Measurement of SASA isn't working, probably due to a missing Atom. Debug files written to "
                f'{putils.sasa_debug_dir}')

    @property
    def relative_sasa(self) -> list[float]:
        """Return a per-residue array of the relative solvent accessible area for the Structure"""
        return [residue.relative_sasa for residue in self.residues]

    @property
    def sasa_apolar(self) -> list[float]:
        """Return a per-residue array of the polar (hydrophilic) solvent accessible area for the Structure"""
        return [residue.sasa_apolar for residue in self.residues]

    @property
    def sasa_polar(self) -> list[float]:
        """Return a per-residue array of the apolar (hydrophobic) solvent accessible area for the Structure"""
        return [residue.sasa_polar for residue in self.residues]

    @property
    def surface_residues(self, relative_sasa_thresh: float = default_sasa_burial_threshold, **kwargs) -> list[Residue]:
        """Get the Residue instances that reside on the surface of the molecule

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should have before it is considered
                'surface'. Default cutoff is based on Levy, E. 2010
        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated
        Returns:
            The surface Residue instances
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        return [residue for residue in self.residues if residue.relative_sasa >= relative_sasa_thresh]

    @property
    def interior_residues(self, relative_sasa_thresh: float = default_sasa_burial_threshold, **kwargs) -> list[Residue]:
        """Get the Residue instances that reside in the interior of the molecule

        Args:
            relative_sasa_thresh: The relative area threshold that the Residue should fall below before it is considered
                'interior'. Default cutoff is based on Levy, E. 2010
        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated
        Returns:
            The interior Residue instances
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        return [residue for residue in self.residues if residue.relative_sasa < relative_sasa_thresh]

    def get_surface_area_residues(
        self, residues: list[Residue] = None, dtype: sasa_types_literal = 'polar', **kwargs
    ) -> float:
        """Get the surface area for specified residues

        Args:
            residues: The Residues to sum. If not provided, will be retrieved by `.get_residues()`
            dtype: The type of area classification to query.
        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated
            numbers: Container[int] = None – Residue numbers of interest
            indices: Iterable[int] = None – Residue indices of interest for the Structure
        Returns:
            Angstrom^2 of surface area
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        if not residues:
            residues = self.get_residues(**kwargs)

        sasa_dtype = f'sasa_{dtype}'
        try:
            return sum([getattr(residue, sasa_dtype) for residue in residues])
        except AttributeError:
            raise ValueError(
                f" {dtype=} is an invalid 'sasa_dtype'. Viable types are {', '.join(sasa_types)}"
            )

    def errat(self, out_path: AnyStr = os.getcwd()) -> tuple[float, np.ndarray]:
        """Find the overall and per residue Errat accuracy for the given Structure

        Args:
            out_path: The path where Errat files should be written
        Returns:
            Overall Errat score, Errat value/residue array
        """
        # name = 'errat_input-%s-%d.pdb' % (self.name, random() * 100000)
        # current_struc_file = self.write(out_path=os.path.join(out_path, name))
        # errat_cmd = [putils.errat_exe_path, os.path.splitext(name)[0], out_path]  # for writing file first
        # os.system('rm %s' % current_struc_file)
        out_path = out_path if out_path[-1] == os.sep else out_path + os.sep  # errat needs trailing "/"
        errat_cmd = [putils.errat_exe_path, out_path]  # for passing atoms by stdin
        # p = subprocess.Popen(errat_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # out, err = p.communicate(input=self.get_atom_record().encode('utf-8'))
        # logger.info(self.get_atom_record()[:120])
        iteration = 1
        all_residue_scores = []
        number_of_residues = self.number_of_residues
        while iteration < 5:
            p = subprocess.run(errat_cmd, input=self.get_atom_record(), encoding='utf-8', capture_output=True)
            all_residue_scores = p.stdout.strip().split('\n')
            # Subtract one due to the addition of overall score
            if len(all_residue_scores) - 1 == number_of_residues:
                break
            iteration += 1

        if iteration == 5:
            error = p.stderr.strip().split('\n')
            self.log.debug(f"{self.errat.__name__} couldn't generate the correct output length. "
                           f'({len(all_residue_scores) - 1}) != number_of_residues ({number_of_residues}). Got stderr:'
                           f'\n{error}')
        # errat_output_file = os.path.join(out_path, '%s.ps' % name)
        # errat_output_file = os.path.join(out_path, 'errat.ps')
        # else:
        # print(subprocess.list2cmdline(['grep', 'Overall quality factor**: ', errat_output_file]))
        # p = subprocess.Popen(['grep', 'Overall quality factor', errat_output_file],
        #                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # errat_out, errat_err = p.communicate()
        try:
            # overall_score = set(errat_out.decode().split('\n'))
            # all_residue_scores = list(map(str.strip, errat_out.split('\n'), 'Residue '))
            # all_residue_scores = errat_out.split('\n')
            overall_score = all_residue_scores.pop(-1)
            return float(overall_score.split()[-1]), \
                np.array([float(score[-1]) for score in map(str.split, all_residue_scores)])
        except (IndexError, AttributeError, ValueError):  # ValueError when returning text instead of float
            self.log.warning(f'{self.name}: Failed to generate ERRAT measurement. Errat returned: {all_residue_scores}')
            return 0., np.array([0. for _ in range(number_of_residues)])

    def stride(self, to_file: AnyStr = None, **kwargs):
        """Use Stride to calculate the secondary structure of a PDB.

        Args
            to_file: The location of a file to save the Stride output
        Sets:
            Residue.secondary_structure
        """
        # REM  -------------------- Secondary structure summary -------------------  XXXX
        # REM                .         .         .         .         .               XXXX
        # SEQ  1    IVQQQNNLLRAIEAQQHLLQLTVWGIKQLQAGGWMEWDREINNYTSLIHS   50          XXXX
        # STR       HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  HHHHHHHHHHHHHHHHH               XXXX
        # REM                                                                        XXXX
        # SEQ  51   LIEESQN                                              57          XXXX
        # STR       HHHHHH                                                           XXXX
        # REM                                                                        XXXX
        # LOC  AlphaHelix   ILE     3 A      ALA     33 A                            XXXX
        # LOC  AlphaHelix   TRP    41 A      GLN     63 A                            XXXX
        # REM                                                                        XXXX
        # REM  --------------- Detailed secondary structure assignment-------------  XXXX
        # REM                                                                        XXXX
        # REM  |---Residue---|    |--Structure--|   |-Phi-|   |-Psi-|  |-Area-|      XXXX
        # ASG  ILE A    3    1    H    AlphaHelix    360.00    -29.07     180.4      XXXX
        # ASG  VAL A    4    2    H    AlphaHelix    -64.02    -45.93      99.8      XXXX
        # ASG  GLN A    5    3    H    AlphaHelix    -61.99    -39.37      82.2      XXXX

        # ASG    Detailed secondary structure assignment
        # Format:
        #  5-8  Residue type
        #  9-10 Protein chain identifier
        #  11-15 PDB residue number
        #  16-20 Ordinal residue number
        #  24-25 One letter secondary structure code **)
        #  26-39 Full secondary structure name
        #  42-49 Phi angle
        #  52-59 Psi angle
        #  61-69 Residue solvent accessible area
        #
        # -rId1Id2..  Read only Chains Id1, Id2 ...
        # -cId1Id2..  Process only Chains Id1, Id2 ...

        # The Stride based secondary structure names of each unique element where possible values are
        #  H:Alpha helix,
        #  G:3-10 helix,
        #  I:PI-helix,
        #  E:Extended conformation,
        #  B/b:Isolated bridge,
        #  T:Turn,
        #  C:Coil (none of the above)'
        current_struc_file = self.write(out_path=f'stride_input-{self.name}-{random() * 100000}.pdb')
        p = subprocess.Popen([putils.stride_exe_path, current_struc_file],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        struct_file = Path(current_struc_file)
        struct_file.unlink(missing_ok=True)
        # self.log.debug(f'Stride file is at: {current_struc_file}')

        if out:
            if to_file:
                with open(to_file, 'wb') as f:
                    f.write(out)
            stride_output = out.decode('utf-8').split('\n')
        else:
            self.log.warning(f'{self.name}: No secondary structure assignment found with Stride')
            return

        residue_idx = count()
        residues = self.residues
        for line in stride_output:
            # residue_idx = int(line[10:15])
            if line[0:3] == 'ASG':
                # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
                # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
                # self.chain(line[9:10]).residue(int(line[10:15].strip())).secondary_structure = line[24:25]
                residues[next(residue_idx)].secondary_structure = line[24:25]

        self.secondary_structure = ''.join(residue.secondary_structure for residue in residues)

    def is_termini_helical(self, termini: stutils.termini_literal = 'n', window: int = 5) -> bool:
        """Using assigned secondary structure, probe for helical termini using a segment of 'window' residues. Will
        remove any disordered residues from the specified termini before checking, with the assumption that the
        disordered terminal residues are not integral to the structure

        Args:
            termini: Either 'n' or 'c' should be specified
            window: The segment size to search
        Returns:
            True if the specified terminus has a stretch of helical residues the length of the window
        """
        # Strip "disorder" from the termini, then use the window to compare against the secondary structure
        search_window = window * 2
        if termini in 'Nn':
            term_window = self.secondary_structure.lstrip(SS_DISORDER_IDENTIFIERS)[:search_window]
        elif termini in 'Cc':
            term_window = self.secondary_structure.rstrip(SS_DISORDER_IDENTIFIERS)[-search_window:]
        else:
            raise ValueError(
                f"The termini value {termini} isn't allowed. Must indicate one of {get_args(stutils.termini_literal)}")

        if 'H' * window in term_window:
            return True
        else:
            return False

    def calculate_secondary_structure(self, **kwargs):
        """Perform the secondary structure calculation for the Structure using the DEFAULT_SS_PROGRAM

        Keyword Args:
            to_file: AnyStr = None - The location of a file to save secondary structure calculations
        """
        self.__getattribute__(DEFAULT_SS_PROGRAM)(**kwargs)  # self.stride()

    @property
    def secondary_structure(self) -> str:
        """The Structure secondary structure assignment as provided by the DEFAULT_SS_PROGRAM"""
        try:
            return self._secondary_structure
        except AttributeError:
            try:
                self._secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)
            except AttributeError:  # When residue.secondary_structure not set
                self.calculate_secondary_structure()
            # self._secondary_structure = self.fill_secondary_structure()
            return self._secondary_structure

    @secondary_structure.setter
    def secondary_structure(self, secondary_structure: Sequence[str]):
        if secondary_structure:
            if len(secondary_structure) == self.number_of_residues:
                self._secondary_structure = ''.join(secondary_structure)
                for residue, ss in zip(self.residues, secondary_structure):
                    residue.secondary_structure = ss
            else:
                self.log.warning(f"The passed secondary_structure length, {len(secondary_structure)} != "
                                 f'{self.number_of_residues}, the number of residues')  # . Recalculating...')

    def termini_proximity_from_reference(self, termini: stutils.termini_literal = 'n',
                                         reference: np.ndarray = utils.symmetry.origin, **kwargs) -> float:
        """Finds the orientation of the termini from the origin (default) or from a reference point

        Args:
            termini: Either 'n' or 'c' should be specified
            reference: The reference where the point should be measured from
        Returns:
            When compared to the reference, 1 if the termini is more than halfway from the center of the Structure and
                -1 if the termini is less than halfway from the center of the Structure
        """
        if termini == 'n':
            residue_coords = self.residues[0].n_coords
        elif termini == 'c':
            residue_coords = self.residues[-1].c_coords
        else:
            raise ValueError(
                f"'termini' must be either 'n' or 'c', not {termini}")

        if reference is None:
            reference = utils.symmetry.origin

        max_distance = self.distance_from_reference(reference=reference, measure='max')
        min_distance = self.distance_from_reference(reference=reference, measure='min')
        coord_distance = np.linalg.norm(residue_coords - reference)
        if abs(coord_distance - max_distance) < abs(coord_distance - min_distance):
            return 1  # Termini further from the reference
        else:
            return -1  # Termini closer to the reference

    def get_atom_record(self, **kwargs) -> str:
        """Provides the Structure as a 'PDB' formatted string of Atom records

        Keyword Args:
            chain_id: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        return '\n'.join(residue.__str__(**kwargs) for residue in self.residues)

    def get_fragments(self, residues: list[Residue] = None, residue_numbers: list[int] = None,
                      fragment_db: fragment.db.FragmentDatabase = None, **kwargs) -> list[fragment.MonoFragment]:
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Args:
            residues: The specific Residues to search for
            residue_numbers: The specific residue numbers to search for
            fragment_db: The FragmentDatabase with representative fragment types  to query against
        Returns:
            The MonoFragments found on the Structure
        """
        if not residues and not residue_numbers:
            return []

        if fragment_db is None:
            fragment_db = self.fragment_db
            if fragment_db is None:
                raise ValueError("Can't assign fragments without passing 'fragment_db' or setting .fragment_db")
            self.log.warning(f"Without passing 'fragment_db', using the existing .fragment_db={repr(fragment_db)}")

        try:
            fragment_db.representatives
        except AttributeError:
            raise TypeError(
                f"The passed fragment_db is not of the required type "
                f"'{fragment.db.FragmentDatabase.__class__.__name__}'")

        fragment_length = fragment_db.fragment_length
        fragment_range = range(*fragment_db.fragment_range)
        fragments = []
        for residue_number in residue_numbers:
            frag_residues = self.get_residues(numbers=[residue_number + i for i in fragment_range])

            if len(frag_residues) == fragment_length:
                new_fragment = fragment.MonoFragment(residues=frag_residues, fragment_db=fragment_db, **kwargs)
                if new_fragment.i_type:
                    fragments.append(new_fragment)

        return fragments

    # Preferred method using Residue fragments
    def get_fragment_residues(self, residues: list[Residue] = None, residue_numbers: list[int] = None,
                              fragment_db: fragment.db.FragmentDatabase = None,
                              rmsd_thresh: float = fragment.Fragment.rmsd_thresh, **kwargs) -> list | list[Residue]:
        """Assigns a Fragment type to Residue instances identified from a FragmentDatabase, and returns them

        Args:
            residues: The specific Residues to search for
            residue_numbers: The specific residue numbers to search for
            fragment_db: The FragmentDatabase with representative fragment types to query the Residue against
            rmsd_thresh: The threshold for which a rmsd should fail to produce a fragment match
        Sets:
            Each Fragment Residue instance self.guide_coords, self.i_type
        Returns:
            The Residue instances that match Fragment representatives from the Structure
        """
        if fragment_db is None:
            fragment_db = self.fragment_db
            if fragment_db is None:
                raise ValueError("Can't assign fragments without passing 'fragment_db' or setting .fragment_db")
            self.log.warning(f"Without passing 'fragment_db', using the existing .fragment_db={repr(fragment_db)}")

        try:
            fragment_db.representatives
        except AttributeError:
            raise TypeError(
                f"The passed fragment_db is not of the required type "
                f"'{fragment.db.FragmentDatabase.__class__.__name__}'")

        if residue_numbers is not None:
            residues = self.get_residues(numbers=residue_numbers)

        # Get iterable of residues
        residues = self.residues if residues is None else residues

        # Get neighboring ca coords on each side by retrieving flanking residues. If not fragment_length, remove
        fragment_length = fragment_db.fragment_length
        frag_lower_range, frag_upper_range = fragment_db.fragment_range

        # Iterate over the residues in reverse to remove any indices that are missing and convert to coordinates
        viable_residues = []
        residues_ca_coords = []
        for residue in residues:
            residue_set = \
                residue.get_upstream(frag_lower_range) + [residue] + residue.get_downstream(frag_upper_range-1)
            if len(residue_set) == fragment_length:
                residues_ca_coords.append([residue.ca_coords for residue in residue_set])
                viable_residues.append(residue)

        residue_ca_coords = np.array(residues_ca_coords)

        # Solve for fragment type (secondary structure classification could be used too)
        found_fragments = []
        for idx, residue in enumerate(viable_residues):
            min_rmsd = float('inf')
            residue_ca_coord_set = residue_ca_coords[idx]
            for fragment_type, representative in fragment_db.representatives.items():
                rmsd, rot, tx = superposition3d(residue_ca_coord_set, representative.ca_coords)
                if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                    residue.frag_type = fragment_type
                    min_rmsd = rmsd

            if residue.frag_type:
                residue.fragment_db = fragment_db
                residue._fragment_coords = fragment_db.representatives[residue.frag_type].backbone_coords
                found_fragments.append(residue)

        return found_fragments

    def find_fragments(self, fragment_db: fragment.db.FragmentDatabase = None, **kwargs) \
            -> list[tuple[fragment.GhostFragment, fragment.Fragment, float]]:
        """Search Residue instances to find Fragment instances that are neighbors, returning all Fragment pairs.
        By default, returns all Residue instances neighboring FragmentResidue instances

        Args:
            fragment_db: The FragmentDatabase with representative fragment types to query the Residue against
        Keyword Args:
            residues: list[Residue] = None - The specific Residues to search for
            residue_numbers: list[int] = None - The specific residue numbers to search for
            rmsd_thresh: float = fragment.Fragment.rmsd_thresh - The threshold for which a rmsd should fail to produce
                a fragment match
            distance: float = 8.0 - The distance to query for neighboring fragments
            min_match_value: float = 2 - The minimum value which constitutes an acceptable fragment z_score
            clash_coords: np.ndarray = None – The coordinates to use for checking for GhostFragment clashes
        Returns:
            The GhostFragment, Fragment pairs, along with their match score
        """
        if fragment_db is None:
            fragment_db = self.fragment_db
            if fragment_db is None:
                raise ValueError("Can't assign fragments without passing 'fragment_db' or setting .fragment_db")
            self.log.warning(f"Without passing 'fragment_db', using the existing .fragment_db={repr(fragment_db)}")

        fragment_time_start = time.time()
        frag_residues = self.get_fragment_residues(fragment_db=fragment_db, **kwargs)
        self.log.info(f'Found {len(frag_residues)} fragments on {self.name}')

        frag_residue_indices = [residue.index for residue in frag_residues]
        all_fragment_pairs = []
        for frag_residue in frag_residues:
            # frag_neighbors = frag_residue.get_residue_neighbors(**kwargs)
            neighbors = self.get_residues_by_atom_indices(frag_residue.neighboring_atom_indices(**kwargs))
            # THIS GETS NON-SELF FRAGMENTS TOO
            # frag_neighbors = [residue for residue in neighbors if residue.frag_type]
            # if not frag_neighbors:
            #     continue
            frag_neighbors = [residue for residue in neighbors if residue.index in frag_residue_indices]
            if not frag_neighbors:
                continue
            all_fragment_pairs.extend(fragment.find_fragment_overlap([frag_residue], frag_neighbors, **kwargs))

        self.log.debug(f'Took {time.time() - fragment_time_start:.8f}s')
        return all_fragment_pairs

    @property
    def spatial_aggregation_propensity(self) -> np.ndarray:
        """Return the spatial aggregation propensity on a per-residue basis

        Returns:
            The array of floats representing the spatial aggregation propensity for each Residue in the Structure
        """
        try:
            return self._sap
        except AttributeError:
            self._sap = np.array(self.spatial_aggregation_propensity_per_residue())
            return self._sap

    def spatial_aggregation_propensity_per_residue(self, distance: float = 5., **kwargs) -> list[float]:
        """Calculate the spatial aggregation propensity on a per-residue basis using calculated heavy atom contacts to
        define which Residue instances are in contact.

        Caution: Contrasts with published method due to use of relative sasa for each Residue instance instead of
        relative sasa for each Atom instance

        Args:
            distance: The distance in angstroms to measure Atom instances in contact
        Keyword Args:
            probe_radius: float = 1.4 - The radius which surface area should be generated
        Returns:
            The floats representing the spatial aggregation propensity for each Residue in the Structure
        """
        # SASA Keyword args that are not reported as available
        # atom: bool = True - Whether the output should be generated for each atom.
        #     If False, will be generated for each Residue
        if not self.sasa:
            self.get_sasa(**kwargs)
        # Set up the hydrophobicity parameters
        hydrophobicity_ = hydrophobicity_values_glycine_centered['black_and_mould']
        # Get heavy Atom coordinates
        heavy_coords = self.heavy_coords
        # Make and query a tree
        tree = BallTree(heavy_coords)
        query = tree.query_radius(heavy_coords, distance)

        residues = self.residues
        # In case this was already called, all should be set to 0.0
        for residue in residues:
            residue.spatial_aggregation_propensity = 0.
            # Set the hydrophobicity_ attribute in a first pass to reduce repetitive lookups
            residue.hydrophobicity_ = hydrophobicity_[residue.type]

        heavy_atom_coords_indexed_residues = self.heavy_coords_indexed_residues
        contacting_pairs = set((heavy_atom_coords_indexed_residues[idx1], heavy_atom_coords_indexed_residues[idx2])
                               for idx2, contacts in enumerate(query.tolist()) for idx1 in contacts.tolist())
        # Residue.spatial_aggregation_propensity starts as 0., so we are adding any observation to that attribute
        for residue1, residue2 in contacting_pairs:
            # Multiply suggested hydrophobicity value by the Residue.relative_sasa
            # Only set on residue1 as this is the "center" of the calculation
            residue1.spatial_aggregation_propensity = residue2.hydrophobicity_ * residue2.relative_sasa

        return [residue.spatial_aggregation_propensity for residue in residues]

    @property
    def contact_order(self) -> np.ndarray:
        """Return the contact order on a per-Residue basis

        Returns:
            The array of floats representing the contact order for each Residue in the Structure
        """
        try:
            return self._contact_order
        except AttributeError:
            self._contact_order = np.array(self.contact_order_per_residue())
            return self._contact_order

    @contact_order.setter
    def contact_order(self, contact_order: Sequence[float]):
        """Set the contact order for each Residue from an array

        Args:
            contact_order: A zero-indexed per residue measure of the contact order
        """
        residues = self.residues
        if len(residues) != len(contact_order):
            raise ValueError(
                f"Can't set {self.contact_order.__name__} with a sequence length ({len(contact_order)})"
                f'!= to the number of residues, {self.number_of_residues}')
        for residue, contact_order in zip(residues, contact_order):
            residue.contact_order = contact_order

    # Distance of 6 angstroms between heavy atoms was used for 1998 contact order work,
    # subsequent residue wise contact order has focused on the Cb-Cb heuristic of 12 A
    # KM thinks that an atom-atom based measure is more accurate, see below for alternative method.
    # The BallTree creation is the biggest time cost regardless.
    def contact_order_per_residue(self, sequence_distance_cutoff: int = 2, distance: float = 6.) -> list[float]:
        """Calculate the contact order on a per-residue basis using calculated heavy atom contacts

        Args:
            sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
            distance: The distance in angstroms to measure Atom instances in contact
        Returns:
            The floats representing the contact order for each Residue in the Structure
        """
        # Get heavy Atom coordinates
        heavy_coords = self.heavy_coords
        # Make and query a tree
        tree = BallTree(heavy_coords)
        query = tree.query_radius(heavy_coords, distance)

        residues = self.residues
        # In case this was already called, we should set all to 0.0
        for residue in residues:
            residue.contact_order = 0.

        heavy_atom_coords_indexed_residues = self.heavy_coords_indexed_residues
        contacting_pairs = set((heavy_atom_coords_indexed_residues[idx1], heavy_atom_coords_indexed_residues[idx2])
                               for idx2, contacts in enumerate(query.tolist()) for idx1 in contacts.tolist())
        # Residue.contact_order starts as 0., so we are adding any observation to that attribute
        for residue1, residue2 in contacting_pairs:
            # Calculate using number since index might not actually specify the intended distance
            residue_sequence_distance = abs(residue1.number - residue2.number)
            if residue_sequence_distance >= sequence_distance_cutoff:
                # Only set on residue1 so that we don't overcount
                residue1.contact_order += residue_sequence_distance

        number_residues = len(residues)
        for residue in residues:
            residue.contact_order /= number_residues

        return [residue.contact_order for residue in residues]

    # # This method uses 12 A cb - cb heuristic
    # def contact_order_per_residue(self, sequence_distance_cutoff: int = 2, distance: float = 12.) -> list[float]:
    #     """Calculate the contact order on a per-residue basis using CB - CB contacts
    #
    #     Args:
    #         sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
    #         distance: The distance in angstroms to measure Atom instances in contact
    #     Returns:
    #         The floats representing the contact order for each Residue in the Structure
    #     """
    #     # Get CB coordinates
    #     coords = self.cb_coords
    #     # make and query a tree
    #     tree = BallTree(coords)
    #     query = tree.query_radius(coords, distance)
    #
    #     residues = self.residues
    #     # in case this was already called, we should set all to 0.
    #     if residues[0].contact_order > 0:
    #         for residue in residues:
    #             residue.contact_order = 0.
    #
    #     contacting_pairs = \
    #         [(residues[idx1], residues[idx2]) for idx2, contacts in enumerate(query.tolist())
    #          for idx1 in contacts.tolist()]
    #
    #     # Residue.contact_order starts as 0., so we are adding any observation to that attribute
    #     for residue1, residue2 in contacting_pairs:
    #         residue_sequence_distance = abs(residue1.number - residue2.number)
    #         if residue_sequence_distance >= sequence_distance_cutoff:
    #             residue1.contact_order += residue_sequence_distance
    #
    #     number_residues = len(residues)
    #     for residue in residues:
    #         residue.contact_order /= number_residues
    #
    #     return [residue.contact_order for residue in residues]

    def format_resfile_from_directives(self, residue_directives: dict[int | Residue, str],
                                       include: dict[int | Residue, set[str]] = None,
                                       background: dict[int | Residue, set[str]] = None, **kwargs) -> list[str]:
        """Format Residue mutational potentials given Residues/residue numbers and corresponding mutation directive.
        Optionally, include specific amino acids and limit to a specific background. Both dictionaries accessed by same
        keys as residue_directives

        Args:
            residue_directives: {Residue object: 'mutational_directive', ...}
            include: Include a set of specific amino acids for each residue
            background: The background amino acids to compare possibilities against
        Keyword Args:
            special: bool = False - Whether to include special residues
        Returns:
            For each Residue, returns the string formatted for a resfile with a 'PIKAA' and amino acid type string
        """
        if background is None:
            background = {}
        if include is None:
            include = {}

        res_file_lines = []
        residues = self.residues
        for residue_index, directive in residue_directives.items():
            if isinstance(residue_index, Residue):
                residue = residue_index
                residue_index = residue.index
            else:
                residue = residues[residue_index]

            allowed_aas = residue.mutation_possibilities_from_directive(
                directive, background=background.get(residue_index), **kwargs)
            allowed_aas = {protein_letters_3to1_extended[aa] for aa in allowed_aas}
            allowed_aas = allowed_aas.union(include.get(residue_index, {}))
            res_file_lines.append(f'{residue.number} {residue.chain_id} PIKAA {"".join(sorted(allowed_aas))}')

        return res_file_lines

    def make_resfile(self, residue_directives: dict[Residue | int, str], out_path: AnyStr = os.getcwd(),
                     header: list[str] = None, **kwargs) -> AnyStr:
        """Format a resfile for the Rosetta Packer from Residue mutational directives

        Args:
            residue_directives: {Residue/int: 'mutational_directive', ...}
            out_path: Directory to write the file
            header: A header to constrain all Residues for packing
        Keyword Args:
            include: dict[Residue | int, set[str]] = None - Include a set of specific amino acids for each residue
            background: dict[Residue | int, set[str]] = None - The background amino acids to compare possibilities
            special: bool = False - Whether to include special residues
        Returns:
            The path to the resfile
        """
        residue_lines = self.format_resfile_from_directives(residue_directives, **kwargs)
        res_file = os.path.join(out_path, f'{self.name}.resfile')
        with open(res_file, 'w') as f:
            # Format the header
            f.write('%s\n' % ('\n'.join(header + ['start']) if header else 'start'))
            # Start the body
            f.write('%s\n' % '\n'.join(residue_lines))

        return res_file

    def set_b_factor_by_attribute(self, dtype: residue_attributes_literal):
        """Set the b-factor entry for every Residue to a Residue attribute

        Args:
            dtype: The attribute of interest
        """
        if isinstance(dtype, str):
            # self.set_residues_attributes(b_factor=dtype)
            for residue in self.residues:
                residue.b_factor = getattr(residue, dtype)
        else:
            raise TypeError(
                f"The type '{dtype.__class__.__name__}' isn't a string. To {self.set_b_factor_by_attribute.__name__}, "
                "you must provide 'dtype' as a string specifying a Residue attribute")

    def set_b_factor_data(self, values: Iterable[float]):
        """Set the b-factor entry for every Residue to a value from an array-like

        Args:
            values: Array-like of integer types to set each Residue instance 'b_factor' attribute to
        """
        if isinstance(values, Iterable):
            values = list(values)
            if len(values) != self.number_of_residues:
                raise ValueError(
                    f"Can't provide a array-like of values with length {len(values)} != {self.number_of_residues}, the "
                    "number of residues")
            for residue, value in zip(self.residues, values):
                residue.b_factor = value
        else:
            raise TypeError(
                f"The type '{values.__class__.__name__}' isn't an Iterable. To {self.set_b_factor_data.__name__}, you "
                "must provide the 'values' as an Iterable of integer type with length = number_of_residues")

    def __copy__(self) -> ContainsResidues:  # Todo -> Self: in python 3.11
        # self.log.debug(f'In ContainsResidues __copy__ of {repr(self)}')
        other: ContainsResidues = super().__copy__()
        for attr in self.ignore_copy_attrs:
            other.__dict__.pop(attr, None)

        return other

    copy = __copy__  # Overwrites to use this instance __copy__

    @property
    def _key(self) -> tuple[str, int, ...]:
        return self.name, *self._residue_indices



class Structures(ContainsResidues, UserList):
    # Todo
    #   mesh inheritance of both Structure and UserClass...
    #   FROM set_residues_attributes in Structure, check all Structure attributes and methods that could be in conflict
    #   are all concatenated Structure methods and attributes accounted for?
    #   ensure UserList .append(), .extend() etc. are allowed and work as intended or overwrite them
    """A view of a set of Structure instances"""
    data: list[ContainsResidues]
    dtype: str
    """The type of Structure in instance"""

    def __init__(self, structures: Iterable[ContainsResidues], dtype: str = None, **kwargs):
        """Pass the parent Structure with parent= to initialize .log, .coords, .atoms, and .residues

        Args:
            structures: The Iterable of Structure to set the Structures with
            dtype: If an empty Structures, the specific subclass of Structure that Structures contains
        """
        super().__init__(initlist=structures, **kwargs)  # initlist sets UserList.data to Iterable[Structure]

        if self.is_parent():
            raise stutils.ConstructionError(
                f"Couldn't create {Structures.__name__} without passing 'parent' argument"
            )

        if not self.data:  # Set up an empty Structures
            self.dtype = dtype if dtype else 'Structure'
        elif all([True if isinstance(structure, ContainsResidues) else False for structure in self]):
            # self.data = [structure for structure in structures]
            self._atom_indices = []
            for structure in self:
                self._atom_indices.extend(structure.atom_indices)
            self._residue_indices = []
            for structure in self:
                self._residue_indices.extend(structure.residue_indices)

            self.dtype = dtype if dtype else type(self.data[0]).__name__
        else:
            raise ValueError(
                f"Can't set {self.__class__.__name__} by passing '{', '.join(type(structure) for structure in self)}, "
                f'must set with type [Structure, ...] or an empty constructor. Ex: Structures()')

        # Overwrite attributes in Structure
        self.name = f'{self.parent.name}-{dtype}Selection'

    # this setter would work in a world where Structures has it's own .coords, .atoms, and .residues
    # @StructureBase._parent.setter
    # def _parent(self, parent: StructureBase):
    #     """Set the 'parent' StructureBase of this instance"""
    #     # peel parent off to protect StructureBase from .coords, .atoms, and .residues as these don't index parent
    #     self._log = parent._log
    #     self._coords = Coords(np.concatenate([structure.coords for structure in self.data]))
    #
    #     atoms = []
    #     for structure in self.data:
    #         atoms.extend(structure.atoms)
    #     self._atoms = Atoms(atoms)
    #
    #     residues = []
    #     for structure in self.data:
    #         residues.extend(structure.residues)
    #     self._residues = Residues(residues)

    @property
    def structures(self) -> list[ContainsResidues]:
        """Returns the underlying data in Structures"""
        return self.data

    # @property
    # def name(self) -> str:
    #     """The id of the Structures container"""
    #     try:
    #         return self._name
    #     except AttributeError:
    #         try:
    #             self._name = f'{self.parent.name}-{self.dtype}_{Structures.__name__}'
    #         except AttributeError:  # if not .parent.name
    #             self._name = f'{"-".join(structure.name for structure in self)}_{Structures.__name__}'
    #         return self._name

    # @name.setter
    # def name(self, name: str):
    #     self._name = name

    @property
    def number_of_structures(self):
        return len(self.data)

    # @property
    # def model_coords(self):
    #     """Return a view of the modeled Coords. These may be symmetric if a SymmetricModel"""
    #     return self._model_coords.coords
    #
    # @model_coords.setter
    # def model_coords(self, coords):
    #     if isinstance(coords, Coords):
    #         self._model_coords = coords
    #     else:
    #         raise AttributeError(
    #             'The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    #             'view. To pass the Coords object for a Structure, use the private attribute _coords')

    # @property
    # def coords(self) -> np.ndarray:
    #     """Return a view of the Coords from the Structures"""
    #     try:
    #         return self._coords.coords
    #     except AttributeError:  # if not set by parent, try to set from each individual structure
    #         self._coords = Coords(np.concatenate([structure.coords for structure in self]))
    #         return self._coords.coords
    #
    # @property
    # def atoms(self):
    #     """Return a view of the Atoms from the Structures"""
    #     try:
    #         return self._atoms
    #     except AttributeError:  # if not set by parent, try to set from each individual structure
    #         atoms = []
    #         # for structure in self.structures:
    #         for structure in self:
    #             atoms.extend(structure.atoms)
    #         self._atoms = Atoms(atoms)
    #         return self._atoms
    #
    # @property
    # def residues(self):
    #     try:
    #         return self._residues
    #     except AttributeError:  # if not set by parent, try to set from each individual structure
    #         residues = []
    #         for structure in self:
    #             residues.extend(structure.residues)
    #         self._residues = Residues(residues)
    #         return self._residues

    # the use of Structure methods for coords_indexed_residue* should work well
    # @property
    # def coords_indexed_residues(self) -> np.ndarray:
    #     try:
    #         return self._coords_indexed_residues
    #     except AttributeError:
    #         self._coords_indexed_residues = np.array([residue for residue in self.residues for _ in residue.range])
    #         return self._coords_indexed_residues
    #
    # @property
    # def coords_indexed_residue_atoms(self) -> np.ndarray:
    #     try:
    #         return self._coords_indexed_residue_atoms
    #     except AttributeError:
    #         self._coords_indexed_residue_atoms = \
    #             np.array([res_atom_idx for residue in self.residues for res_atom_idx in residue.range])
    #         return self._coords_indexed_residue_atoms
    #
    # @property
    # def residue_indexed_atom_indices(self) -> list[list[int]]:
    #     """For every Residue in the Structure provide the Residue instance indexed, Structures Atom indices
    #
    #     Returns:
    #         Residue objects indexed by the Residue position in the corresponding .coords attribute
    #     """
    #     try:
    #         return self._residue_indexed_atom_indices
    #     except AttributeError:
    #         range_idx = prior_range_idx = 0
    #         self._residue_indexed_atom_indices = []
    #         for residue in self.residues:
    #             range_idx += residue.number_of_atoms
    #             self._residue_indexed_atom_indices.append(list(range(prior_range_idx, range_idx)))
    #             prior_range_idx = range_idx
    #         return self._residue_indexed_atom_indices

    @property
    def backbone_indices(self):
        try:
            return self._backbone_indices
        except AttributeError:
            self._backbone_indices = []
            for structure in self:
                self._backbone_indices.extend(structure.backbone_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._backbone_indices.extend([idx - start_index for idx in structure.backbone_indices])
            return self._backbone_indices

    @property
    def backbone_and_cb_indices(self):
        try:
            return self._backbone_and_cb_indices
        except AttributeError:
            self._backbone_and_cb_indices = []
            for structure in self:
                self._backbone_and_cb_indices.extend(structure.backbone_and_cb_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._backbone_and_cb_indices.extend([idx - start_index for idx in structure.backbone_and_cb_indices])
            return self._backbone_and_cb_indices

    @property
    def cb_indices(self):
        try:
            return self._cb_indices
        except AttributeError:
            self._cb_indices = []
            for structure in self:
                self._cb_indices.extend(structure.cb_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._cb_indices.extend([idx - start_index for idx in structure.cb_indices])
            return self._cb_indices

    @property
    def ca_indices(self):
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = []
            for structure in self:
                self._ca_indices.extend(structure.ca_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._ca_indices.extend([idx - start_index for idx in structure.ca_indices])
            return self._ca_indices

    @property
    def heavy_indices(self):
        try:
            return self._heavy_indices
        except AttributeError:
            self._heavy_indices = []
            for structure in self:
                self._heavy_indices.extend(structure.heavy_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._heavy_indices.extend([idx - start_index for idx in structure.heavy_indices])
            return self._heavy_indices

    @property
    def side_chain_indices(self):
        try:
            return self._side_chain_indices
        except AttributeError:
            self._side_chain_indices = []
            for structure in self:
                self._side_chain_indices.extend(structure.side_chain_indices)
                # this way would be according to indexing operations performed on this structure.coords/.atoms
                # start_index = structure.start_index
                # self._heavy_indices.extend([idx - start_index for idx in structure.side_chain_indices])
            return self._side_chain_indices

    # These below should work when performed using the Structure superclass methods
    # def translate(self, **kwargs):
    #     """Perform a translation to the Structures ensuring only the Structure containers of interest are translated
    #     ensuring the underlying coords are not modified
    #
    #     Keyword Args:
    #         translation=None (numpy.ndarray | list[float]): The first translation to apply, expected array shape (3,)
    #     """
    #     for structure in self:
    #         structure.translate(**kwargs)
    #
    # def rotate(self, **kwargs):
    #     """Perform a rotation to the Structures ensuring only the Structure containers of interest are rotated ensuring
    #     the underlying coords are not modified
    #
    #     Keyword Args:
    #         rotation=None (numpy.ndarray | list[list[float]]): The first rotation to apply, expected array shape (3, 3)
    #     """
    #     for structure in self:
    #         structure.rotate(**kwargs)
    #
    # def transform(self, **kwargs):
    #     """Perform a specific transformation to the Structures ensuring only the Structure containers of interest are
    #     transformed ensuring the underlying coords are not modified
    #
    #     Transformation proceeds by matrix multiplication and vector addition with the order of operations as:
    #     rotation, translation, rotation2, translation2
    #
    #     Keyword Args:
    #         rotation=None (numpy.ndarray | list[list[float]]): The first rotation to apply, expected array shape (3, 3)
    #         translation=None (numpy.ndarray | list[float]): The first translation to apply, expected array shape (3,)
    #         rotation2=None (numpy.ndarray | list[list[float]]): The second rotation to apply, expected array shape (3,
    #             3)
    #         translation2=None (numpy.ndarray | list[float]): The second translation to apply, expected array shape (3,)
    #     """
    #     for structure in self:
    #         structure.transform(**kwargs)
    #
    # def get_transformed_copy(self, **kwargs):  # rotation=None, translation=None, rotation2=None, translation2=None):
    #     """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space
    #
    #     Transformation proceeds by matrix multiplication and vector addition with the order of operations as:
    #     rotation, translation, rotation2, translation2
    #
    #     Keyword Args:
    #         rotation=None (numpy.ndarray | list[list[float]]): The first rotation to apply, expected array shape (3, 3)
    #         translation=None (numpy.ndarray | list[float]): The first translation to apply, expected array shape (3,)
    #         rotation2=None (numpy.ndarray | list[list[float]]): The second rotation to apply, expected array shape (3,
    #             3)
    #         translation2=None (numpy.ndarray | list[float]): The second translation to apply, expected array shape (3,)
    #     """
    #     new_structures = self.__new__(self.__class__)
    #     # print('Transformed Structure type (__new__) %s' % type(new_structures))
    #     # print('self.__dict__ is %s' % self.__dict__)
    #     new_structures.__init__([structure.get_transformed_copy(**kwargs) for structure in self])
    #     # print('Transformed Structures, structures %s' % [structure for structure in new_structures.structures])
    #     # print('Transformed Structures, models %s' % [structure for structure in new_structures.models])
    #     return new_structures
    #     # return Structures(structures=[structure.get_transformed_copy(**kwargs) for structure in self.structures])

    # def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, increment_chains: bool = True,
    #           header: str = None, **kwargs) -> str | None:
    #     """Write Structures to a file specified by out_path or with a passed file_handle
    #
    #     Args:
    #         out_path: The location where the Structure object should be written to disk
    #         file_handle: Used to write Structure details to an open FileObject
    #         increment_chains: Whether to write each Structure with a new chain name, otherwise write as a new Model
    #         header: If there is header information that should be included. Pass new lines with a "\n"
    #     Returns:
    #         The name of the written file if out_path is used
    #     """
    #     if file_handle:  # _Todo increment_chains compatibility
    #         file_handle.write('%s\n' % self.get_atom_record(**kwargs))
    #         return
    #
    #     with open(out_path, 'w') as f:
    #         if header:
    #             if isinstance(header, str):
    #                 f.write(header)
    #             # if isinstance(header, Iterable):
    #
    #         if increment_chains:
    #             available_chain_ids = chain_id_generator()
    #             for structure in self.structures:
    #                 chain_id = next(available_chain_ids)
    #                 structure.write(file_handle=f, chain_id=chain_id)
    #                 c_term_residue = structure.c_terminal_residue
    #                 f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
    #                                                                       c_term_residue.type, chain_id,
    #                                                                       c_term_residue.number))
    #         else:
    #             for model_number, structure in enumerate(self.structures, 1):
    #                 f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
    #                 structure.write(file_handle=f)
    #                 c_term_residue = structure.c_terminal_residue
    #                 f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
    #                                                                       c_term_residue.type, structure.chain_id,
    #                                                                       c_term_residue.number))
    #                 f.write('ENDMDL\n')
    #
    #     return out_path

    # def __repr__(self) -> str:
    #     return f'{self.__class__.__name__}({self.name})'

    # def __str__(self):
    #     return self.name

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> ContainsResidues:
        yield from iter(self.data)

    def __getitem__(self, idx: int) -> ContainsResidues:
        return self.data[idx]


# 0 indexed, 1 letter aa, alphabetically sorted at the origin
try:
    reference_residues: list[Residue] = utils.unpickle(putils.reference_residues_pkl)
except (_pickle.UnpicklingError, ImportError, FileNotFoundError, AttributeError, utils.InputError) as error:
    logger.error(''.join(traceback.format_exc()))
    logger.error(f'The reference_residues ran into an error upon load. You need to regenerate the serialized version '
                 f'using {putils.pickle_program_requirements_cmd}')
    print('\n')
    reference_residues = None
