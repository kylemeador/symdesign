from __future__ import annotations

import os
import subprocess
from abc import ABC
from collections import UserList, defaultdict
from copy import copy
from logging import Logger
from random import random
from typing import IO, Sequence, Container, Literal, get_args, Callable, Any, AnyStr, Iterable

import numpy as np
from sklearn.neighbors import BallTree  # , KDTree, NearestNeighbors

from structure.coords import Coords, superposition3d
from structure.fragment import Fragment, MonoFragment, ResidueFragment
from structure.fragment.db import FragmentDatabase, fragment_factory
from structure.utils import protein_letters_alph1, protein_letters_1to3, protein_letters_3to1_extended
from utils.path import freesasa_exe_path, stride_exe_path, errat_exe_path, freesasa_config_path, \
    reference_residues_pkl, program_name, program_version, biological_interfaces
from utils import start_log, null_log, unpickle, digit_translate_table, DesignError, ClashError, startdate
from utils.symmetry import origin

# globals
logger = start_log(name=__name__)
protein_letters_3to1_extended_mse = protein_letters_3to1_extended.copy()
protein_letters_3to1_extended_mse['MSE'] = 'M'
coords_type_literal = Literal['all', 'backbone', 'backbone_and_cb', 'ca', 'cb', 'heavy']
directives = Literal['special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding',
                     'branched']
mutation_directives: tuple[directives, ...] = get_args(directives)
atom_or_residue = Literal['atom', 'residue']
structure_container_types = Literal['atoms', 'residues', 'chains', 'entities']
termini_literal = Literal['n', 'c']
# protein_backbone_atom_types = {'N', 'CA', 'O'}  # 'C', Removing 'C' for fragment library guide atoms...
protein_backbone_atom_types = {'N', 'CA', 'C', 'O'}
protein_backbone_and_cb_atom_types = {'N', 'CA', 'C', 'O', 'CB'}
# mutation_directives = \
#     ['special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding', 'branched']
residue_properties = {'ALA': {'hydrophobic', 'apolar'},
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
                      'TYR': {'hydrophobic', 'apolar', 'aromatic', 'hbonding'}}
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
            'TYR': 263, 'VAL': 174}  # from table 1, theoretical values of Tien et al. 2013


def unknown_index():
    return -1


polarity_types_literal = Literal['apolar', 'polar']
polarity_types: tuple[polarity_types_literal, ...] = get_args(polarity_types_literal)
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
    'HIS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'ND1': 1, 'CD2': 0, 'CE1': 0,
                                       'NE2': 1}),
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
hydrogens = {   # the doubled up numbers (and single number second) are from PDB version of hydrogen inclusion
    'ALA': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '3HB': 0, 'HB1': 0, 'HB2': 0, 'HB3': 0},
    'ARG': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, 'HE': 1, '1HH1': 1, '2HH1': 1,
            '1HH2': 1, '2HH2': 1,
            'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0, 'HD1': 0, 'HD2': 0, 'HH11': 1, 'HH12': 1, 'HH21': 1, 'HH22': 1},
    'ASN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HD2': 1, '2HD2': 1, 'HB1': 0, 'HB2': 0, 'HD21': 1, 'HD22': 1,
            '1HD1': 1, '2HD1': 1, 'HD11': 1, 'HD12': 1},  # these are the alternative specification
    'ASP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HB1': 0, 'HB2': 0},
    'CYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HB1': 0, 'HB2': 0, 'HG': 1},
    'GLN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE2': 1, '2HE2': 1, 'HB1': 0, 'HB2': 0, 'HG1': 0,
            'HG2': 0, 'HE21': 1, 'HE22': 1,
            '1HE1': 1, '2HE1': 1, 'HE11': 1, 'HE12': 1},  # these are the alternative specification
    'GLU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, 'HB1': 0, 'HB2': 0, 'HG1': 0, 'HG2': 0},
    'GLY': {'H': 1, '1HA': 0, 'HA1': 0, '2HA': 0, 'HA2': 0, '3HA': 0, 'HA3': 0},
    'HIS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 1, 'HD2': 0, 'HE1': 0, 'HE2': 1, 'HB1': 0, 'HB2': 0, '1HD': 1,
            '2HD': 0, '1HE': 0, '2HE': 1},  # this assumes HD1 is on ND1, HE2 is on NE2
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
termini_polarity = {'1H': 1, '2H': 1, '3H': 1, 'OXT': 1}
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


slice_remark, slice_number, slice_atom_type, slice_alt_location, slice_residue_type, slice_chain, \
    slice_residue_number, slice_code_for_insertion, slice_x, slice_y, slice_z, slice_occ, slice_temp_fact, \
    slice_element, slice_charge = slice(0, 6), slice(6, 11), slice(12, 16), slice(16, 17), slice(17, 20), \
    slice(21, 22), slice(22, 26), slice(26, 27), slice(30, 38), slice(38, 46), slice(46, 54), slice(54, 60), \
    slice(60, 66), slice(76, 78), slice(78, 80)


def read_pdb_file(file: AnyStr, pdb_lines: list[str] = None, separate_coords: bool = True, **kwargs) -> \
        dict[str, Any]:
    """Reads .pdb file and returns structural information pertaining to parsed file

    By default, returns the coordinates as a separate numpy.ndarray which is parsed directly by Structure. This will be
    associated with each Atom however, separate parsing is done for efficiency. To include coordinate info with the
    individual Atom instances, pass separate_coords=False. (Not recommended)

    Args:
        file: The path to the file to parse
        pdb_lines: If lines are already read, provide the lines instead
        separate_coords: Whether to separate parsed coordinates from Atom instances. Will be returned as two separate
            entries in the parsed dictionary, otherwise returned with coords=None
    Returns:
        The dictionary containing all the parsed structural information
    """
    # Todo figure out handling of header lines. Best way I see is to take out any seqres line indices (we add later)
    #  and only like the head from any line before an ATOM record. The modification of any of the coordinates or Atoms
    #  would cause the header information to be invalidated as it would then become a "SymDesign Model"
    if pdb_lines:
        path, extension = None, None
    else:
        with open(file, 'r') as f:
            pdb_lines = f.readlines()
        path, extension = os.path.splitext(file)

    # PDB
    assembly: str | None = None
    # type to info index:   1    2    3    4    5    6    7     11     12   13   14
    temp_info: list[tuple[int, str, str, str, str, int, str, float, float, str, str]] = []
    # if separate_coords:
    #     # type to info index:   1    2    3    4    5    6    7 8,9,10    11     12   13   14
    #     atom_info: list[tuple[int, str, str, str, str, int, str, None, float, float, str, str]] = []
    # else:
    #     # type to info index:   1    2    3    4    5    6    7      8,9,10      11     12   13   14
    #     atom_info: list[tuple[int, str, str, str, str, int, str, list[float], float, float, str, str]] = []
    #     # atom_info: dict[int | str | list[float]] = {}

    # atom_info: dict[int | str | list[float]] = {}
    coords: list[list[float]] = []
    # cryst: dict[str, str | tuple[float]] = {}
    cryst_record: str = ''
    dbref: dict[str, dict[str, str]] = {}
    entity_info: dict[str, dict[dict | list | str]] = {}
    name = os.path.basename(path) if path else None  # .replace('pdb', '')
    header: list = []
    multimodel: bool = False
    resolution: float | None = None
    # space_group: str | None = None
    seq_res_lines: list[str] = []
    # uc_dimensions: list[float] = []
    # Structure
    biomt: list = []
    biomt_header: str = ''

    if extension[-1].isdigit():
        # If last character is not a letter, then the file is an assembly, or the extension was provided weird
        assembly = extension.translate(digit_translate_table)

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
    #     if line_tokens[0] == 'ATOM' or line_tokens[4] == 'MSE' and line_tokens[0] == 'HETATM':
    #         if line_tokens[3] not in ['', 'A']:
    #             continue
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
            temp_info.append((int(line[slice_number]), atom_type, alt_loc_str, residue_type, line[slice_chain],
                              int(line[slice_residue_number]), line[slice_code_for_insertion].strip(),
                              float(line[slice_occ]), float(line[slice_temp_fact]),
                              line[slice_element].strip(), line[slice_charge].strip()))
            # atom_info.append((int(line[slice_number]), atom_type, alt_loc_str, residue_type, line[slice_chain],
            #                   int(line[slice_residue_number]), line[slice_code_for_insertion].strip(), None,
            #                   float(line[slice_occ]), float(line[slice_temp_fact]),
            #                   line[slice_element].strip(), line[slice_charge].strip()))
            # atom_info.append(dict(number=int(line[slice_number]), type=atom_type, alt_location=alt_loc_str,
            #                       residue_type=residue_type, chain=line[slice_chain],
            #                       residue_number=int(line[slice_residue_number]),
            #                       code_for_insertion=line[slice_code_for_insertion].strip(),
            #                       coords=[float(line[slice_x]), float(line[slice_y]), float(line[slice_z])]
            #                       occupancy=float(line[slice_occ]), b_factor=float(line[slice_temp_fact]),
            #                       element_symbol=line[slice_element].strip(),
            #                       charge=line[slice_charge].strip()))
            # prepare the atomic coordinates for addition to numpy array
            coords.append([float(line[slice_x]), float(line[slice_y]), float(line[slice_z])])
        elif remark == 'MODEL ':
            multimodel = True
        elif remark == 'SEQRES':
            seq_res_lines.append(line[11:])
        elif remark == 'REMARK':
            header.append(line.strip())
            remark_number = line[slice_number]
            # elif line[:18] == 'REMARK 350   BIOMT':
            if remark_number == ' 350 ':  # 6:11  '   BIOMT'
                biomt_header += line
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
                except ValueError:  # not enough values to unpack
                    continue
                if biomt_indicator == 'BIOMT':
                    if operation_number != current_operation:  # we reached a new transformation matrix
                        current_operation = operation_number
                        biomt.append([])
                    # add the transformation to the current matrix
                    biomt[-1].append(list(map(float, [x, y, z, tx])))
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
            # self.entity_info[entity] = \
            # {'chains': list(map(str.strip, line[line.rfind(':') + 1:].strip().rstrip(';').split(',')))}
            entity_info[f'{name}_{entity}'] = \
                {'chains': list(map(str.strip, line[line.rfind(':') + 1:].strip().rstrip(';').split(',')))}
            entity = None
        elif remark == 'SCALE ':
            header.append(line.strip())
        elif remark == 'CRYST1':
            header.append(line.strip())
            cryst_record = line  # don't .strip() so we can keep \n attached for output
            # uc_dimensions, space_group = parse_cryst_record(cryst_record)
            # cryst = {'space': space_group, 'a_b_c': tuple(uc_dimensions[:3]), 'ang_a_b_c': tuple(uc_dimensions[3:])}

    if not temp_info:
        raise ValueError(f'The file {file} has no ATOM records!')

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

    parsed_info = \
        dict(biological_assembly=assembly,
             atoms=[Atom.without_coordinates(idx, *info) for idx, info in enumerate(temp_info)] if separate_coords
             else
             # initialize with individual coords. Not sure why anyone would do this, but include for compatibility
             [Atom(number=number, atom_type=atom_type, alt_location=alt_location, residue_type=residue_type,
                   chain=chain, residue_number=residue_number, code_for_insertion=code_for_insertion,
                   coords=coords[idx], occupancy=occupancy, b_factor=b_factor, element=element, charge=charge)
              for idx, (number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion,
                        occupancy, b_factor, element, charge)
              in enumerate(temp_info)],
             biomt=biomt,  # go to Structure
             biomt_header=biomt_header,  # go to Structure
             coords=coords if separate_coords else None,
             # cryst=cryst,
             cryst_record=cryst_record,
             # dbref=dbref,
             entity_info=entity_info,
             header=header,
             multimodel=multimodel,
             name=name,
             resolution=resolution,
             reference_sequence=reference_sequence,
             # space_group=space_group,
             # uc_dimensions=uc_dimensions,
             )
    parsed_info.update(**kwargs)  # explictly overwrites any parsing if argument was passed
    return parsed_info


mmcif_error = f'This type of parsing is not available yet, but YOU can make it happen! Modify ' \
              f'{read_pdb_file.__name__}() slightly to parse a .cif file and create read_mmcif_file()'


class Log:
    """Responsible for StructureBase logging operations

    Args:
        log: The logging.Logger to handle StructureBase logging. If None is passed a Logger with NullHandler is used
    """
    def __init__(self, log: Logger | None = null_log):
        self.log = log


null_struct_log = Log()


class Symmetry:
    # _has_symmetry: bool | None
    symmetry: str | None
    _symmetric_dependents: list[Structure]

    def __init__(self, **kwargs):
        # self._has_symmetry = None
        self._symmetric_dependents = []
        super().__init__(**kwargs)

    # @property
    def is_symmetric(self) -> bool:
        """Query whether the Structure is symmetric. Returns True if self.symmetry is not None"""
        try:
            return self.symmetry is not None
        except AttributeError:
            return False


# parent Structure controls these attributes
parent_variable = '_StructureBase__parent'
new_parent_attributes = ('_coords', '_log', '_atoms', '_residues')
parent_attributes = (parent_variable,) + new_parent_attributes
"""Holds all the attributes which the parent StructureBase controls including _parent, _coords, _log, _atoms, 
and _residues
"""


class StructureBase(Symmetry, ABC):
    """Structure object sets up and handles Coords and Log objects as well as maintaining atom_indices and the history
    of Structure subclass creation and subdivision from parent Structure to dependent Structure's. Collects known
    keyword arguments for all derived class construction calls to protect base object. Should always be the last class
    in the method resolution order of derived classes

    Args:
        parent: If a Structure object created this Structure instance, that objects instance. Will share ownership of
            the log and coords to and dependent Structures
        log: If this is a parent Structure instance, the object that handles Structure object logging
        coords: If this is a parent Structure instance, the Coords of that Structure
    """
    _atom_indices: list[int] | None  # np.ndarray
    _coords: Coords
    _copier: bool = False
    """Whether the StructureBase is being copied by a Container object. If so cut corners"""
    _log: Log
    __parent: StructureBase | None
    state_attributes: set[str] = set()

    def __init__(self, parent: StructureBase = None, log: Log | Logger | bool = True, coords: np.ndarray | Coords = None
                 , biological_assembly=None, cryst_record=None, entity_info=None, file_path=None, header=None,
                 multimodel=None, resolution=None, reference_sequence=None, sequence=None, entities=None,
                 pose_format=None, query_by_sequence=True, entity_names=None, rename_chains=None, **kwargs):
        if parent:  # initialize StructureBase from parent
            self._parent = parent
        else:  # this is the parent
            self.__parent = None  # requries use of _StructureBase__parent attribute checks
            # initialize Log
            if log:
                if log is True:  # use the module logger
                    self._log = Log(logger)
                elif isinstance(log, Log):  # initialized Log
                    self._log = log
                elif isinstance(log, Logger):  # logging.Logger object
                    self._log = Log(log)
                else:
                    raise TypeError(f"Can't set Log to {type(log).__name__}. Must be type logging.Logger")
            else:  # when explicitly passed as None or False, uses the null logger
                self._log = null_struct_log  # Log()

            # initialize Coords
            if coords is None:  # check this first
                # most init occurs from Atom instances that are their own parent until another StructureBase adopts them
                self._coords = Coords()  # null_coords
            elif isinstance(coords, Coords):
                self._coords = coords
            else:  # sets as None if coords wasn't passed and update later
                self._coords = Coords(coords)

        try:
            super().__init__(**kwargs)
        except TypeError:
            raise TypeError(f"The argument(s) passed to the StructureBase object were not recognized and aren't "
                            f'accepted by the object class: {", ".join(kwargs.keys())}')

    @property
    def parent(self) -> StructureBase | None:
        """Return the instance's "parent" StructureBase"""
        # try:
        return self.__parent
        # except AttributeError:
        #     self.__parent = None
        #     return self.__parent

    # Placeholder getter for _parent setter so that derived classes automatically set _log and _coords from _parent set
    @property
    def _parent(self) -> StructureBase | None:
        """Return the instance's "parent" StructureBase"""
        return self.parent

    @_parent.setter
    def _parent(self, parent: StructureBase):
        """Return the instance's "parent" StructureBase"""
        # print('setting __parent')
        self.__parent = parent
        # print('type(self)', type(self))
        # print('id(self)', id(self))
        # print('self.__parent is', self.__parent)
        self._log = parent._log
        self._coords = parent._coords
        # self._atoms = parent._atoms  # Todo make empty Atoms for StructureBase objects?
        # self._residues = parent._residues  # Todo make empty Residues for StructureBase objects?

    def is_dependent(self) -> bool:
        """Is the StructureBase a dependent?"""
        return self.__parent is not None

    def is_parent(self) -> bool:
        """Is the StructureBase a parent?"""
        return self.__parent is None

    @property
    def log(self) -> Logger:
        """Access to the StructureBase Logger"""
        return self._log.log

    @log.setter
    def log(self, log: Logger | Log):
        """Set the StructureBase to a logging.Logger object"""
        if isinstance(log, Logger):  # prefer this protection method versus Log.log property overhead?
            self._log.log = log
        elif isinstance(log, Log):
            self._log.log = log.log
        else:
            raise TypeError(f"Can't set Log to {type(log).__name__}. Must be type logging.Logger")

    @property
    def atom_indices(self) -> list[int] | None:
        """The Atoms/Coords indices which the StructureBase has access to"""
        try:
            return self._atom_indices
        except AttributeError:
            return None

    @property
    def number_of_atoms(self) -> int:
        """The number of atoms/coordinates in the StructureBase"""
        try:
            return len(self._atom_indices)
        except TypeError:
            return 0

    @property
    def coords(self) -> np.ndarray:
        """The coordinates for the Atoms in the StructureBase object"""
        # returns self.Coords.coords(a np.array)[sliced by the instance's atom_indices]
        return self._coords.coords[self._atom_indices]

    @coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        # self.log.critical(f'Setting {self.name} coords')
        if self.is_parent() and self.is_symmetric() and self._symmetric_dependents:
            # This Structure is a symmetric parent, update dependent coords to update the parent
            # self.log.debug(f'self._symmetric_dependents: {self._symmetric_dependents}')
            for dependent in self._symmetric_dependents:
                if dependent.is_symmetric():
                    dependent._parent_is_updating = True
                    # self.log.debug(f'Setting {dependent.name} _symmetric_dependent coords')
                    dependent.coords = coords[dependent.atom_indices]
                    del dependent._parent_is_updating
            # Update the whole Coords.coords as symmetry is not everywhere
            self._coords.replace(self._atom_indices, coords)
        else:  # Simply update these Coords.coords
            self._coords.replace(self._atom_indices, coords)

    def reset_state(self):
        """Remove StructureBase attributes that are valid for the current state but not for a new state

        This is useful for transfer of ownership, or changes in the StructureBase state that should be overwritten
        """
        for attr in self.state_attributes:
            try:
                delattr(self, attr)
            except AttributeError:
                continue


class Atom(StructureBase):
    """An Atom container with the full Structure coordinates and the Atom unique data"""
    # . Pass a reference to the full Structure coordinates for Keyword Arg coords=self.coords
    __coords: list[float]
    # _next_atom: Atom
    # _prev_atom: Atom
    _sasa: float
    _type_str: str
    index: int | None
    number: int | None
    # type: str | None
    alt_location: str | None
    residue_type: str | None
    chain: str | None
    pdb_residue_number: int | None
    residue_number: int | None
    code_for_insertion: str | None
    occupancy: float | None
    b_factor: float | None
    element: str | None
    charge: str | None
    state_attributes: set[str] = StructureBase.state_attributes | {'_sasa'}

    def __init__(self, index: int = None, number: int = None, atom_type: str = None, alt_location: str = ' ',
                 residue_type: str = None, chain: str = None, residue_number: int = None, code_for_insertion: str = ' ',
                 x: float = None, y: float = None, z: float = None, occupancy: float = None, b_factor: float = None,
                 element: str = None, charge: str = None, coords: list[float] = None, **kwargs):
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        super().__init__(**kwargs)
        self.index = index
        self.number = number
        self._type = atom_type
        self._type_str = f'{"" if atom_type[3:] else " "}{atom_type:<3s}'  # pad with space if atom_type is len()=4
        self.alt_location = alt_location
        self.residue_type = residue_type
        self.chain = chain
        self.pdb_residue_number = residue_number
        self.residue_number = residue_number  # originally set the same as parsing
        self.code_for_insertion = code_for_insertion
        if coords is not None:
            self.__coords = coords
        elif x is not None and y is not None and z is not None:
            self.__coords = [x, y, z]
        else:
            self.__coords = []
        self.occupancy = occupancy
        self.b_factor = b_factor
        self.element = element
        self.charge = charge
        # self.sasa = sasa
        # # Set Atom from parent attributes. By default parent is None
        # parent = self.parent
        # if parent:
        #     self._atoms = parent._atoms  # Todo make empty Atoms for Structure objects?
        #     self._residues = parent._residues  # Todo make empty Residues for Structure objects?

    @classmethod
    def without_coordinates(cls, idx, number, atom_type, alt_location, residue_type, chain, residue_number,
                            code_for_insertion, occupancy, b_factor, element, charge):
        """Initialize without coordinates"""
        return cls(index=idx, number=number, atom_type=atom_type, alt_location=alt_location, residue_type=residue_type,
                   chain=chain, residue_number=residue_number, code_for_insertion=code_for_insertion,
                   occupancy=occupancy, b_factor=b_factor, element=element, charge=charge, coords=[])  # list for speed

    def detach_from_parent(self):
        """Remove the current instance from the parent that created it"""
        setattr(self, parent_variable, None)  # set parent explicitly as None
        # # Extract the coordinates
        # coords = self.coords
        # create a new, empty Coords instance
        self._coords = Coords(self.coords)
        self.index = 0
        self.reset_state()

    @property
    def type(self) -> str:  # This can't be set since the self._type_str needs to be changed then
        return self._type

    @property
    def _atom_indices(self) -> list[int]:
        """The index of the Atom in the Atoms/Coords container"""
        return [self.index]

    # Below properties are considered part of the Atom state
    @property
    def sasa(self) -> float:
        """The Solvent accessible surface area for the Atom. Raises AttributeError if .sasa isn't set"""
        # try:  # let the Residue owner handle errors
        return self._sasa
        # except AttributeError:
        #     raise AttributeError

    @sasa.setter
    def sasa(self, sasa: float):
        self._sasa = sasa

    # End state properties
    @property
    def coords(self) -> np.ndarray:
        """The coordinates for the Atoms in the StructureBase object"""
        # returns self.Coords.coords(a np.array)[sliced by the instance's atom_indices]
        try:
            # return self._coords.coords[self.index]
            # ^ this method is what is needed, but not in line with API. v call flatten() to return correct shape
            return self._coords.coords[self._atom_indices].flatten()
        except AttributeError:  # possibly the Atom was set with keyword argument coords instead of Structure Coords
            # this shouldn't be used often as it will be quite slow... give warning?
            # Todo try this
            #  self.parent._collect_coords()  # this should grab all Atom coords and make them _coords (Coords)
            return self.__coords

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass (the Atom coordinates) which is just for compatibility with StructureBase API"""
        return self.coords

    @property
    def x(self) -> float:
        """Access the value for the x coordinate"""
        return self.coords[0]

    @x.setter
    def x(self, x: float):
        """Set the value for the x coordinate"""
        try:
            self._coords.replace(self._atom_indices, [x, self.coords[1], self.coords[2]])
        except AttributeError:  # when _coords not used
            self.__coords = [x, self.coords[1], self.coords[2]]

    @property
    def y(self) -> float:
        """Access the value for the y coordinate"""
        return self.coords[1]

    @y.setter
    def y(self, y: float):
        """Set the value for the y coordinate"""
        try:
            self._coords.replace(self._atom_indices, [self.coords[0], y, self.coords[2]])
        except AttributeError:  # when _coords not used
            self.__coords = [self.coords[0], y, self.coords[2]]

    @property
    def z(self) -> float:
        """Access the value for the z coordinate"""
        return self.coords[2]

    @z.setter
    def z(self, z: float):
        """Set the value for the z coordinate"""
        try:
            self._coords.replace(self._atom_indices, [self.coords[0], self.coords[1], z])
        except AttributeError:  # when _coords not used
            self.__coords = [self.coords[0], self.coords[1], z]

    def is_backbone_and_cb(self) -> bool:
        """Is the Atom is a backbone or CB Atom? Includes N, CA, C, O, and CB"""
        return self.type in protein_backbone_and_cb_atom_types

    def is_backbone(self) -> bool:
        """Is the Atom is a backbone Atom? These include N, CA, C, and O"""
        return self.type in protein_backbone_atom_types

    def is_cb(self, gly_ca: bool = True) -> bool:
        """Is the Atom a CB atom? Default returns True if Glycine and Atom is CA

        Args:
            gly_ca: Whether to include Glycine CA in the boolean evaluation
        """
        if gly_ca:
            return self.type == 'CB' or (self.residue_type == 'GLY' and self.type == 'CA')
        else:
            #                                    When Rosetta assigns, it is this  v  but PDB assigns as this  v
            return self.type == 'CB' or (self.residue_type == 'GLY' and (self.type == '2HA' or self.type == 'HA3'))

    def is_ca(self) -> bool:
        """Is the Atom a CA atom?"""
        return self.type == 'CA'

    def is_heavy(self) -> bool:
        """Is the Atom a heavy atom?"""
        return 'H' not in self.type

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

    def __key(self) -> tuple[int, str, str, float]:
        return self.index, self.type, self.residue_type, self.b_factor

    def get_atom_record(self) -> str:
        """Provide the Atom as an Atom record string

        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        x, y, z = list(self.coords)
        # Add 1 to the self.index since this is 0 indexed
        return f'ATOM  {self.index + 1:5d} {self._type_str}{self.alt_location:1s}{self.residue_type:3s}' \
               f'{self.chain:>2s}{self.residue_number:4d}{self.code_for_insertion:1s}   '\
               f'{x:8.3f}{y:8.3f}{z:8.3f}{self.occupancy:6.2f}{self.b_factor:6.2f}          ' \
               f'{self.element:>2s}{self.charge:2s}'

    def __str__(self) -> str:  # type=None, number=None, pdb=False, chain=None, **kwargs
        """Represent Atom in PDB format"""
        # Use self.type_str to comply with the PDB format specifications because of the atom type field
        # ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  <-- PDB format
        # Checks if len(atom.type)=4 with slice v. If not insert a space
        return f'ATOM  %s {self._type_str}{self.alt_location:1s}%s%s%s' \
               f'{self.code_for_insertion:1s}   %s{self.occupancy:6.2f}{self.b_factor:6.2f}          ' \
               f'{self.element:>2s}{self.charge:2s}'
        # Todo if parent:  # return full ATOM record
        # return 'ATOM  {:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   '\
        #     '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'\
        #     .format(self.index, self.type, self.alt_location, self.residue_type, self.chain, self.residue_number,
        #             self.code_for_insertion, *list(self.coords), self.occupancy, self.b_factor, self.element,
        #             self.charge)

    def __eq__(self, other: Atom) -> bool:
        if isinstance(other, Atom):
            return self.__key() == other.__key()
        raise NotImplementedError(f'Can\' compare {type(self).__name__} instance to {type(other).__name__} instance')

    def __hash__(self) -> int:
        return hash(self.__key())

    # def __copy__(self):  # Todo this is ready, but isn't needed anywhere
    #     other = self.__class__.__new__(self.__class__)
    #     other.__dict__ = copy(self.__dict__)
    #
    #     if self.is_parent():  # this Structure is the parent, it's copy should be too
    #         # set the copying Structure attribute ".spawn" to indicate to dependents the "other" of this copy
    #         self.spawn = other
    #         try:
    #             for attr in parent_attributes:
    #                 other.__dict__[attr] = copy(self.__dict__[attr])
    #         except KeyError:  # '_atoms' is not present and will be after _log, _coords
    #             pass
    #         # remove the attribute spawn after other Structure containers are copied
    #         del self.spawn
    #     else:  # this Structure is a dependent, it's copy should be too
    #         try:
    #             other._parent = self.parent.spawn
    #         except AttributeError:  # this copy was initiated by a Structure that is not the parent
    #             # self.log.debug(f'The copied {type(self).__name__} is being set as a parent. It was a dependent '
    #             #                f'previously')
    #             if self._copier:  # Copy initiated by Atoms container
    #                 pass
    #             else:
    #                 other.detach_from_parent()
    #
    #     return other


alpha_helix_15 = [Atom(0, 1, 'N', ' ', 'ALA', 'A', 1, ' ', 27.128, 20.897, 37.943, 1.00, 0.00, 'N', ''),
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
                  Atom(39, 40, 'CB', ' ', 'ALA', 'A', 8, ' ', 25.581, 31.506, 36.435, 1.00, 0.00, 'C', ''),
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
                  Atom(74, 75, 'CB', ' ', 'ALA', 'A', 15, ' ', 24.418, 41.969, 34.808, 1.00, 0.00, 'C', '')]


class Atoms:
    atoms: np.ndarray

    def __init__(self, atoms: list[Atom] | np.ndarray = None):
        if atoms is None:
            self.atoms = np.array([])
        elif not isinstance(atoms, (np.ndarray, list)):
            raise TypeError(f'Can\'t initialize {type(self).__name__} with {type(atoms).__name__}. Type must be a '
                            f'numpy.ndarray or list of {Atom.__name__} instances')
        else:
            self.atoms = np.array(atoms, dtype=np.object_)
    #     self.find_prev_and_next()
    #
    # def find_prev_and_next(self):
    #     """Set prev_atom and next_atom attributes for each Atom. One inherently sets the other in Atom"""
    #     for next_idx, atom in enumerate(self.atoms[:-1], 1):
    #         atom.next_atom = self.atoms[next_idx]

    def are_dependents(self) -> bool:
        """Check if any of the Atom instance are dependents on another Structure"""
        for atom in self:
            if atom.is_dependent():
                return True
        return False

    def reindex(self, start_at: int = 0):
        """Set each Atom instance index according to incremental Atoms/Coords index

        Args:
            start_at: The integer to start renumbering at
        """
        if start_at > 0:
            if start_at < self.atoms.shape[0]:  # if in the Atoms index range
                prior_atom = self.atoms[start_at - 1]
                for idx, atom in enumerate(self.atoms[start_at:].tolist(), prior_atom.idx + 1):
                    atom.index = idx
        else:
            for idx, atom in enumerate(self, start_at):
                atom.index = idx

    def delete(self, indices: Sequence[int]):
        """Delete Atom instances from the Atoms container

        Args:
            indices: The indices to delete from the Coords array
        """
        self.atoms = np.delete(self.atoms, indices)

    def insert(self, at: int, new_atoms: list[Atom] | np.ndarray):
        """Insert Atom objects into the Atoms container

        Args:
            at: The index to perform the insert at
            new_atoms: The residues to include into Residues
        """
        self.atoms = np.concatenate((self.atoms[:at],
                                     new_atoms if isinstance(new_atoms, Iterable) else [new_atoms],
                                     self.atoms[at:]))

    def append(self, new_atoms: list[Atom] | np.ndarray):
        """Append additional Atom instances into the Atoms container

        Args:
            new_atoms: The Atom instances to include into Atoms
        Sets:
            self.atoms = numpy.concatenate((self.atoms, new_atoms))
        """
        self.atoms = np.concatenate((self.atoms, new_atoms))

    def reset_state(self):
        """Remove any attributes from the Atom instances that are part of the current Structure state

        This is useful for transfer of ownership, or changes in the Atom state that need to be overwritten
        """
        for atom in self:
            atom.reset_state()

    def set_attributes(self, **kwargs):
        """Set Atom attributes passed by keyword to their corresponding value"""
        for atom in self:
            for key, value in kwargs.items():
                setattr(atom, key, value)

    def __copy__(self) -> Atoms:  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.atoms = self.atoms.copy()
        # copy all Atom
        for idx, atom in enumerate(other.atoms):
            # Set an attribute to indicate the atom shouldn't be "detached"
            # since a Structure owns this Atoms instance
            atom._copier = True
            other.atoms[idx] = copy(atom)
            atom._copier = False

        # other.find_prev_and_next()

        return other

    def __len__(self) -> int:
        return self.atoms.shape[0]

    def __iter__(self) -> Atom:
        yield from self.atoms.tolist()


class ContainsAtomsMixin(StructureBase):
    # _atom_indices: list[int]
    _atoms: Atoms
    # _coords: Coords
    backbone_and_cb_indices: list[int]
    backbone_indices: list[int]
    ca_indices: list[int]
    cb_indices: list[int]
    heavy_indices: list[int]
    number_of_atoms: int
    side_chain_indices: list[int]
    # These state_attributes are used by all subclasses despite no usage in this class
    state_attributes: set[str] = StructureBase.state_attributes | \
        {'_backbone_and_cb_indices', '_backbone_indices', '_ca_indices', '_cb_indices', '_heavy_indices',
         '_side_chain_indices'}

    def __init__(self, atoms: list[Atom] | Atoms = None, **kwargs):
        super().__init__(**kwargs)
        if atoms is not None:
            self._assign_atoms(atoms)

    @classmethod
    def from_atoms(cls, atoms: list[Atom] | Atoms = None, **kwargs):
        return cls(atoms=atoms, **kwargs)

    @property
    def atoms(self) -> list[Atom] | None:
        """Return the Atom instances in the Structure"""
        try:
            return self._atoms.atoms[self._atom_indices].tolist()
        except AttributeError:  # when self._atoms isn't set or is None and doesn't have .atoms
            return

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
    def ca_atoms(self) -> list[Atom]:
        """Return CA Atom instances from the StructureBase"""
        return self._atoms.atoms[self.ca_indices].tolist()

    @property
    def cb_atoms(self) -> list[Atom]:
        """Return CB Atom instances from the StructureBase"""
        return self._atoms.atoms[self.cb_indices].tolist()

    @property
    def backbone_atoms(self) -> list[Atom]:
        """Return backbone Atom instances from the StructureBase"""
        return self._atoms.atoms[self.backbone_indices].tolist()

    @property
    def backbone_and_cb_atoms(self) -> list[Atom]:
        """Return backbone and CB Atom instances from the StructureBase"""
        return self._atoms.atoms[self.backbone_and_cb_indices].tolist()

    @property
    def heavy_atoms(self) -> list[Atom]:
        """Return heavy Atom instances from the StructureBase"""
        return self._atoms.atoms[self.heavy_indices].tolist()

    @property
    def side_chain_atoms(self) -> list[Atom]:
        """Return side chain Atom instances from the StructureBase"""
        return self._atoms.atoms[self.side_chain_indices]

    def atom(self, atom_number: int) -> Atom | None:
        """Return the Atom specified by atom number if a matching Atom is found, otherwise None"""
        for atom in self.atoms:
            if atom.number == atom_number:
                return atom
        return None

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass for the StructureBase coordinates"""
        number_of_atoms = self.number_of_atoms
        return np.matmul(np.full(number_of_atoms, 1 / number_of_atoms), self.coords)
        # try:
        #     return self._center_of_mass
        # except AttributeError:
        #     self.find_center_of_mass()
        #     return self._center_of_mass

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

    def _assign_atoms(self, atoms: Atoms | list[Atom], atoms_only: bool = True, **kwargs):
        """Assign Atom instances to the Structure, create Atoms object

        Args:
            atoms: The Atom instances to assign to the Structure
            atoms_only: Whether Atom instances are being assigned on their own. Residues will be created if so.
                If not, indicate False and use other Structure information such as Residue instances to complete set up.
                When False, atoms won't become dependents of this instance until specifically called using
                Atoms.set_attributes(_parent=self)
        Keyword Args:
            coords: (numpy.ndarray) = None - The coordinates to assign to the Structure.
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
            atoms = copy(atoms)
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
            # if not self.file_path:  # Assume this instance wasn't parsed and Atom indices are incorrect
            self._atoms.reindex()
            # self._set_coords_indexed()

    def _populate_coords(self, coords: np.ndarray = None, from_source: structure_container_types = 'atoms'):
        """Set up the coordinates, initializing them from_source coords if none are set

        Only useful if the calling Structure is a parent, and coordinate initialization has yet to occur

        Args:
            coords: The coordinates to assign to the Structure. Will use from_source.coords if not specified
            from_source: The source to set the coordinates from if they are missing
        """
        if coords is not None:
            # Try to set the provided coords. This will handle issue where empty Coords class should be set
            # Setting .coords through normal mechanism preserves subclasses requirement to handle symmetric coordinates
            self.coords = np.concatenate(coords)
        if self._coords.coords.shape[0] == 0:  # Check if Coords (_coords) hasn't been populated
            # If it hasn't, then coords weren't passed. Try to set from self.from_source. catch missing from_source
            try:
                self._coords.set(np.concatenate([s.coords for s in getattr(self, from_source)]))
            except AttributeError:
                try:  # Probably missing from_source. .coords is available in all structure_container_types...
                    getattr(self, from_source)
                except AttributeError:
                    raise AttributeError(f'{from_source} is not set on the current {type(self).__name__} instance!')
                raise AttributeError(f'Missing .coords attribute on the current {type(self).__name__} '
                                     f'instance.{from_source} attribute. This is really not supposed to happen! '
                                     f'Congrats you broke a core feature! 0.15 bitcoin have been added to your wallet')

    def _validate_coords(self):
        """Ensure that the StructureBase coordinates are formatted correctly"""
        # This is the functionality we car about most of the time when making a new Structure
        if self.number_of_atoms != len(self.coords):  # .number_of_atoms typically just set by self._atom_indices
            raise ValueError(f'The number of Atoms ({self.number_of_atoms}) != number of Coords ({len(self.coords)}). '
                             f'Consider initializing {type(self).__name__} without explicitly passing coords if this '
                             f"isn't expected")

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

    def format_header(self, **kwargs) -> str:
        """Format any records desired in the Structure header

        Returns:
            The header with PDB file formatting
        """
        return \
            f'HEADER                                          {startdate}   XXXX              \n' \
            f'EXPDTA    THEORETICAL MODEL                                                     \n' \
            f'REMARK 220                                                                      \n' \
            f'REMARK 220 EXPERIMENTAL DETAILS                                                 \n' \
            f'REMARK 220  EXPERIMENT TYPE                : THEORETICAL MODELLING              \n' \
            f'REMARK 220  DATE OF DATA COLLECTION        : {startdate}                        \n' \
            f'REMARK 220                                                                      \n' \
            f'REMARK 220 MODEL GENERATED BY {program_name.upper():<50s}\n' \
            f'REMARK 220 VERSION {program_version:<61s}\n'

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None, **kwargs) -> \
            str | None:
        """Write Structure Atoms to a file specified by out_path or with a passed file_handle

        If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        Keyword Args
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
            chain: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The name of the written file if out_path is used
        """
        if file_handle:
            file_handle.write(f'{self.get_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.format_header(**kwargs)
            if header is not None and isinstance(header, str):
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.get_atom_record(**kwargs)}\n')
            return out_path

    def get_atoms(self, numbers: Container = None, pdb: bool = False, **kwargs) -> list[Atom]:
        """Retrieve Atom objects in Structure. Returns all by default. If a list of numbers is provided, the selected
        Atom numbers are returned

        Args:
            numbers: The Atom numbers of interest
            pdb: Whether to search for numbers as they were parsed
        Returns:
            The requested Atom objects
        """
        if numbers is not None:
            if isinstance(numbers, Container):
                number_source = 'number_pdb' if pdb else 'number'
                return [atom for atom in self.atoms if getattr(atom, number_source) in numbers]
            else:
                self.log.error(f'The passed numbers type "{type(numbers).__name__}" must be a Container. Returning'
                               f' all Atom instances instead')
        return self.atoms

    def set_atoms_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Atoms in the Structure

        Keyword Args:
            numbers: (Container[int]) = None - The Atom numbers of interest
            pdb: (bool) = False - Whether to search for numbers as they were parsed (if True)
        """
        for atom in self.get_atoms(**kwargs):
            for kwarg, value in kwargs.items():
                setattr(atom, kwarg, value)


class Residue(ResidueFragment, ContainsAtomsMixin):
    _ca_indices: list[int]
    _cb_indices: list[int]
    _bb_indices: list[int]
    _bb_and_cb_indices: list[int]
    _heavy_atom_indices: list[int]
    _index: int
    _sc_indices: list[int]
    _atoms: Atoms
    _backbone_indices: list[int]
    _backbone_and_cb_indices: list[int]
    _heavy_indices: list[int]
    _side_chain_indices: list[int]
    _start_index: int
    _c_index: int
    _ca_index: int
    _cb_index: int
    _h_index: int
    _n_index: int
    _o_index: int
    _contact_order: float
    _local_density: float
    _next_residue: Residue
    _prev_residue: Residue
    _sasa: float
    _sasa_apolar: float
    _sasa_polar: float
    _secondary_structure: str
    chain: str
    # coords: Coords
    number: int
    number_pdb: int
    state_attributes: set[str] = ContainsAtomsMixin.state_attributes | \
        {'_contact_order', '_local_density',
         '_next_residue', '_prev_residue',
         '_sasa', '_sasa_apolar', '_sasa_polar', '_secondary_structure'}
    type: str

    def __init__(self, atoms: list[Atoms] | Atoms = None, atom_indices: list[int] = None, **kwargs):
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        # kwargs passed to ResidueFragment -> Fragment
        #          fragment_type: int = None, guide_coords: np.ndarray = None, fragment_length: int = 5,
        super().__init__(**kwargs)

        parent = self.parent
        if parent:  # we are setting up a dependent Residue
            self._atom_indices = atom_indices
            try:
                self._start_index = atom_indices[0]
            except (TypeError, IndexError):
                raise ValueError("The Residue wasn't passed atom_indices which are required for initialization")
        # we are setting up a parent (independent) Residue
        elif atoms:  # is not None  # no parent passed, construct from atoms
            self._assign_atoms(atoms)
            self.is_residue_valid()
            # Structure._populate_coords(self)
            # Structure._validate_coords(self)
            self._start_index = 0
            # update Atom instance attributes to ensure they are dependants of this instance
            # must do this after (potential) coords setting to ensure that coordinate info isn't overwritten
            # self._atoms.set_attributes(_parent=self)
            # self._atoms.reindex()
            # self.renumber_atoms()
        else:  # create an empty Residue
            self._atoms = Atoms()
        self.delegate_atoms()

    @property
    def index(self):
        """The Residue index in a Residues container"""
        try:
            return self._index
        except AttributeError:
            raise TypeError(f"{type(self).__name__} is not a member of a Residues container and has no index!")

    # @index.setter
    # def index(self, index):
    #     self._index = index

    @StructureBase._parent.setter
    def _parent(self, parent: StructureBase):
        """Set the Coords object while propagating changes to symmetry "mate" chains"""
        # all of the below comments get at the issue of calling self.__parent here
        # I think I have to call _StructureBase__parent to access this variable for some reason having to do with class
        # inheritance

        # # StructureBase._parent.fset(self, parent)
        # print('type(parent)', type(parent))
        # print('super(Residue, Residue)._parent', super(Residue, Residue)._parent)
        # print('super(Residue, Residue)._parent.fset', super(Residue, Residue)._parent.fset)
        # try:
        #     print('self._log is', self._log)
        # except AttributeError:
        #     print('No log yet')
        super(Residue, Residue)._parent.fset(self, parent)
        # super()._parent.fset(self, parent)
        # print('id(self)', id(self))
        # print('self._log is', self._log)
        # print('self._parent is', self._parent)
        # print('self.is_dependent()', self.is_dependent())
        # # print('self.__parent is', self.__parent)  # WTF doesn't this work?
        # print('self.__dict__', self.__dict__)
        self._atoms = parent._atoms
        # self._residues = parent._residues  # Todo make empty Residues for Structure objects?

    def detach_from_parent(self):
        """Remove the current instance from the parent that created it"""
        setattr(self, parent_variable, None)  # set parent explicitly as None
        # # Extract the coordinates
        # coords = self.coords
        # create a new, empty Coords instance
        self._coords = Coords(self.coords)
        # populate the Structure with its existing instances removed of any indexing
        self._assign_atoms(self.atoms)  # , coords=coords)
        self.reset_state()

    # @StructureBase.coords.setter
    # def coords(self, coords: np.ndarray | list[list[float]]):
    #     """Set the Residue coords according to a new coordinate system. Transforms .guide_coords to the new reference"""
    #     # if self.i_type:  # Fragment has been assigned. Transform the guide_coords according to the new coords
    #     #     _, self.rotation, self.translation = superposition3d(coords, self.coords)
    #     #     # self.guide_coords = np.matmul(self.guide_coords, np.transpose(self.rotation)) + self.translation
    #     super(Residue, Residue).coords.fset(self, coords)  # prefer this over below, as this mechanism could change
    #     # self._coords.replace(self._atom_indices, coords)

    def is_residue_valid(self) -> bool:
        """Returns True if the Residue is constructed properly otherwise raises an error

        Raises:
            ValueError: If the Residue is set improperly
        """
        remove_atom_indices, found_types = [], set()
        atoms = self.atoms
        current_residue_number, current_residue_type = atoms[0].residue_number, atoms[0].residue_type
        for idx, atom in enumerate(atoms[1:], 1):
            if atom.residue_number == current_residue_number and atom.residue_type == current_residue_type:
                if atom.type not in found_types:
                    found_types.add(atom.type)
                else:
                    raise ValueError(f'{self.is_residue_valid.__name__}: The Atom type at index {idx} was already'
                                     f'observed')
            else:
                raise ValueError(f'{self.is_residue_valid.__name__}: The Atom at index {idx} doesn\'t have the '
                                 f'same properties as all previous Atoms')

        if protein_backbone_atom_types.difference(found_types):  # modify if building NucleotideResidue
            raise ValueError(f'{self.is_residue_valid.__name__}: The provided Atoms don\'t contain the required '
                             f'types ({", ".join(protein_backbone_atom_types)}) to build a {type(self).__name__}')

        return True

    # @property
    # def start_index(self) -> int:
    #     """The first atomic index of the Residue"""
    #     return self._start_index

    @ContainsAtomsMixin.start_index.setter
    def start_index(self, index: int):
        """Set Residue atom_indices starting with atom_indices[0] as start_index. Creates remainder incrementally and
        updates individual Atom instance .index accordingly
        """
        self._start_index = index
        self._atom_indices = list(range(index, index + self.number_of_atoms))
        for atom, index in zip(self._atoms.atoms[self._atom_indices].tolist(), self._atom_indices):
            atom.index = index

    @property
    def range(self) -> list[int]:
        """The range of indices corresponding to the Residue atoms"""
        return list(range(self.number_of_atoms))

    # @property
    # def atom_indices(self) -> list[int] | None:
    #     """The indices which belong to the Residue Atoms in the parent Atoms/Coords container"""  # Todo separate __doc?
    #     try:
    #         return self._atom_indices
    #     except AttributeError:
    #         return

    # @atom_indices.setter
    # def atom_indices(self, indices: list[int]):
    #     """Set the Structure atom indices to a list of integers"""
    #     self._atom_indices = indices
    #     try:
    #         self._start_index = indices[0]
    #     except (TypeError, IndexError):
    #         raise IndexError('The Residue wasn\'t passed any atom_indices which are required for initialization')

    def delegate_atoms(self):
        """Set the Residue atoms from a parent StructureBase"""
        side_chain_indices, heavy_indices = [], []
        try:
            for idx, atom in enumerate(self.atoms):
                match atom.type:
                    case 'N':
                        self._n_index = idx
                        self.chain = atom.chain
                        self.number = atom.residue_number
                        self.number_pdb = atom.pdb_residue_number
                        self.type = atom.residue_type
                    case 'CA':
                        self._ca_index = idx
                    case 'CB':
                        self._cb_index = idx
                    case 'C':
                        self._c_index = idx
                    case 'O':
                        self._o_index = idx
                    case 'H':
                        self._h_index = idx
                    case other:
                        side_chain_indices.append(idx)
                        if 'H' not in atom.type:
                            heavy_indices.append(idx)
        except SyntaxError:  # python version not 3.10
            for idx, atom in enumerate(self.atoms):
                if atom.type == 'N':
                    self._n_index = idx
                    self.chain = atom.chain
                    self.number = atom.residue_number
                    self.number_pdb = atom.pdb_residue_number
                    self.type = atom.residue_type
                elif atom.type == 'CA':
                    self._ca_index = idx
                elif atom.type == 'CB':
                    self._cb_index = idx
                elif atom.type == 'C':
                    self._c_index = idx
                elif atom.type == 'O':
                    self._o_index = idx
                elif atom.type == 'H':
                    self._h_index = idx
                else:
                    side_chain_indices.append(idx)
                    if 'H' not in atom.type:
                        heavy_indices.append(idx)

        self.backbone_indices = [getattr(self, f'_{index}_index', None) for index in ['n', 'ca', 'c', 'o']]
        cb_index = getattr(self, '_cb_index', None)
        if cb_index:
            cb_indices = [cb_index]
        else:
            if self.type == 'GLY':
                self._cb_index = getattr(self, '_ca_index')
            cb_indices = []

        # By using private variables, None is removed v
        self.backbone_and_cb_indices = self._bb_indices + cb_indices
        self.heavy_indices = self._bb_and_cb_indices + heavy_indices
        self.side_chain_indices = side_chain_indices
        # if not self.ca_index:  # this is likely a NH or a C=O so we don't have a full residue
        #     self.log.error('Residue %d has no CA atom!' % self.number)
        #     # Todo this residue should be built out, but as of 6/28/22 it can only be deleted
        #     self.ca_index = idx  # use the last found index as a rough guess
        #     self.secondary_structure = 'C'  # just a placeholder since stride shouldn't work

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
        self._bb_indices = [idx for idx in indices if idx is not None]  # check as some will be None if not be provided

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
    def ca_indices(self) -> list[int]:  # This is for compatibility with ContainsAtomsMixin
        """Return the index of the CA Atom as a list in the Residue Atoms/Coords"""
        try:
            return self._ca_indices
        except AttributeError:
            self._ca_indices = [self._atom_indices[self._ca_index]]
            return self._ca_indices

    @property
    def cb_indices(self) -> list[int]:  # This is for compatibility with ContainsAtomsMixin
        """Return the index of the CB Atom as a list in the Residue Atoms/Coords"""
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
            return self._atoms.atoms[self._atom_indices[self._n_index]]
        except AttributeError:
            return None

    @property
    def n_coords(self) -> np.ndarry | None:
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
            return self._atoms.atoms[self._atom_indices[self._h_index]]
        except AttributeError:
            return None

    @property
    def h_coords(self) -> np.ndarry | None:
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
            return self._atoms.atoms[self._atom_indices[self._ca_index]]
        except AttributeError:
            return None

    @property
    def ca_coords(self) -> np.ndarry | None:
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
            return self._atoms.atoms[self._atom_indices[self._cb_index]]
        except AttributeError:
            return None

    @property
    def cb_coords(self) -> np.ndarry | None:
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
            return self._atoms.atoms[self._atom_indices[self._c_index]]
        except AttributeError:
            return None

    @property
    def c_coords(self) -> np.ndarry | None:
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
            return self._atoms.atoms[self._atom_indices[self._o_index]]
        except AttributeError:
            return None

    @property
    def o_coords(self) -> np.ndarry | None:
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
    def next_residue(self) -> Residue | None:
        """The next Residue in the Structure if this Residue is part of a polymer"""
        try:
            return self._next_residue
        except AttributeError:
            return None

    @next_residue.setter
    def next_residue(self, other: Residue):
        """Set the next_residue for this Residue and the prev_residue for the other Residue"""
        self._next_residue = other
        other._prev_residue = self

    @property
    def prev_residue(self) -> Residue | None:
        """The previous Residue in the Structure if this Residue is part of a polymer"""
        try:
            return self._prev_residue
        except AttributeError:
            return None

    @prev_residue.setter
    def prev_residue(self, other: Residue):
        """Set the prev_residue for this Residue and the next_residue for the other Residue"""
        self._prev_residue = other
        other._next_residue = self

    def get_upstream(self, number: int) -> list[Residue]:
        """Get the Residues upstream of (n-terminal to) the current Residue

        Args:
            number: The number of residues to retrieve
        Returns:
            The Residue instances in n- to c-terminal order
        """
        if number == 0:
            raise ValueError("Can't get 0 upstream residues. 1 or more must be specified")

        prior_residues = [self.prev_residue]
        for idx in range(abs(number) - 1):
            try:
                prior_residues.append(prior_residues[idx].prev_residue)
            except AttributeError:  # we hit a termini
                break

        return [residue for residue in prior_residues[::-1] if residue]

    def get_downstream(self, number: int) -> list[Residue]:
        """Get the Residues downstream of (c-terminal to) the current Residue

        Args:
            number: The number of residues to retrieve
        Returns:
            The Residue instances in n- to c-terminal order
        """
        if number == 0:
            raise ValueError("Can't get 0 downstream residues. 1 or more must be specified")

        next_residues = [self.next_residue]
        for idx in range(abs(number) - 1):
            try:
                next_residues.append(next_residues[idx].next_residue)
            except AttributeError:  # we hit a termini
                break

        return [residue for residue in next_residues if residue]

    # Below properties are considered part of the Residue state
    @property
    def local_density(self) -> float:
        """Describes how many heavy Atoms are within a distance (default = 12 Angstroms) of Residue heavy Atoms"""
        try:
            return self._local_density
        except AttributeError:
            self._local_density = 0.  # set to 0 so summation can occur
            try:
                self.parent.local_density()
                return self._local_density
            except AttributeError:
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.local_density.__name__}" '
                                     f'attribute! Ensure you call {Structure.local_density.__name__} before you request'
                                     f' Residue local density information')

    @local_density.setter
    def local_density(self, local_density: float):
        self._local_density = local_density

    @property
    def secondary_structure(self) -> str:
        """Return the secondary structure designation as defined by a secondary structure calculation"""
        try:
            return self._secondary_structure
        except AttributeError:
            raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.secondary_structure.__name__}" '
                                 f'attribute! Ensure you call {Structure.get_secondary_structure.__name__} before you '
                                 f'request Residue secondary structure information')

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
        except AttributeError:  # missing atom.sasa
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
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.sasa.__name__}" attribute! '
                                     f'Ensure you call {Structure.get_sasa.__name__} before you request Residue SASA '
                                     f'information')
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
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.sasa_apolar.__name__}" '
                                     f'attribute! Ensure you call {Structure.get_sasa.__name__} before you request '
                                     f'Residue SASA information')
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
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.sasa_polar.__name__}" '
                                     f'attribute! Ensure you call {Structure.get_sasa.__name__} before you request '
                                     f'Residue SASA information')
            return self._sasa_polar

    @sasa_polar.setter
    def sasa_polar(self, sasa: float | int):
        """Set the polar solvent accessible surface area for the Residue"""
        self._sasa_polar = sasa

    @property
    def relative_sasa(self) -> float:
        """The solvent accessible surface area relative to the standard surface accessibility of the Residue type"""
        return self.sasa / gxg_sasa[self.type]  # may cause problems if self.type attribute can be non-cannonical AA

    @property
    def contact_order(self) -> float:
        """The Residue contact order, which describes how far away each Residue makes contacts in the polymer chain"""
        try:
            return self._contact_order
        except AttributeError:
            # self._contact_order = 0.  # set to 0 so summation can occur
            try:
                self.parent.contact_order_per_residue()
                return self._contact_order
            except AttributeError:
                raise AttributeError(f'Residue {self.number}{self.chain} has no ".{self.contact_order.__name__}" '
                                     f'attribute! Ensure you call {Structure.contact_order.__name__} before you request'
                                     f' Residue contact order information')

    @contact_order.setter
    def contact_order(self, contact_order: float):
        self._contact_order = contact_order

    # End state properties

    # @property
    # def number_of_atoms(self) -> int:
    #     """The number of atoms in the Structure"""
    #     return len(self._atom_indices)

    @property
    def number_of_heavy_atoms(self) -> int:
        return len(self._heavy_indices)

    @property
    def b_factor(self) -> float:
        try:
            return sum(atom.b_factor for atom in self.atoms) / self.number_of_atoms
        except ZeroDivisionError:
            return 0.

    @b_factor.setter
    def b_factor(self, dtype: str | Iterable[float] = None, **kwargs):
        """Set the temperature factor for the Atoms in the Residue

        Args:
            dtype: The data type that should fill the temperature_factor from Residue attributes
                or an iterable containing the explicit b_factor float values
        """
        try:
            for atom in self.atoms:
                atom.b_factor = getattr(self, dtype)
        except AttributeError:
            raise AttributeError(f'The attribute {dtype} was not found in the Residue {self.number}{self.chain}. Are '
                                 f'you sure this is the attribute you want?')
        except TypeError:
            # raise TypeError(f'{type(dtype)} is not a string. To set b_factor, you must provide the dtype as a string')
            try:
                for atom, b_fact in zip(self.atoms, dtype):
                    atom.b_factor = b_fact
            except TypeError:
                raise TypeError(f'{type(dtype)} is not a string nor an Iterable. To set b_factor, you must provide the '
                                f'dtype as a string specifying a Residue attribute OR an Iterable with length = '
                                f'Residue.number_of_atoms')

    def mutation_possibilities_from_directive(self, directive: directives = None, background: set[str] = None,
                                              special: bool = False, **kwargs) -> set[str]:
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
            The euclidean distance between the specified Atom type
        """
        return np.linalg.norm(getattr(self, f'.{dtype}_coords') - getattr(other, f'.{dtype}_coords'))

    # def residue_string(self, pdb: bool = False, chain: str = None, **kwargs) -> tuple[str, str, str]:
    #     """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
    #     formatting
    #
    #     Args:
    #         pdb: Whether the Residue representation should use the pdb number at file parsing
    #         chain: The ID of the chain to use
    #     Returns:
    #         Tuple of formatted Residue attributes
    #     """
    #     return format(self.type, '3s'), (chain or self.chain), \
    #         format(getattr(self, f'number{"_pdb" if pdb else ""}'), '4d')

    def __getitem__(self, idx) -> Atom:
        return self.atoms[idx]

    def __key(self) -> tuple[int, int, str]:
        return self._start_index, self.number_of_atoms, self.type

    def __eq__(self, other: Residue) -> bool:
        if isinstance(other, Residue):
            return self.__key() == other.__key()
        raise NotImplementedError(f'Can\' compare {type(self).__name__} instance to {type(other).__name__} instance')

    def get_atom_record(self, **kwargs) -> str:
        """Provide the Structure Atoms as a PDB file string

        Keyword Args:
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
            chain: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        return f'{self.__str__(**kwargs)}\n'

    def __str__(self, pdb: bool = False, chain: str = None, atom_offset: int = 0, **kwargs) -> str:
        #         type=None, number=None
        """Format the Residue into the contained Atoms. The Atom number is truncated at 5 digits for PDB compliant
        formatting

        Args:
            pdb: Whether the Residue representation should use the number at file parsing
            chain: The chain ID to use
            atom_offset: How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Residue
        """
        # format the string returned from each Atom, such as
        #  'ATOM  %s  CG2 %s %s%s    %s  1.00 17.48           C  0'
        #       AtomIdx  TypeChNumberCoords
        # To
        #  'ATOM     32  CG2 VAL A 132       9.902  -5.550   0.695  1.00 17.48           C  0'
        # self.type, self.alt_location, self.code_for_insertion, self.occupancy, self.b_factor,
        #                     self.element, self.charge)
        # res_str = self.residue_string(**kwargs)
        res_str = format(self.type, '3s'), format(chain or self.chain, '>2s'), \
            format(getattr(self, f'number{"_pdb" if pdb else ""}'), '4d')
        offset = 1 + atom_offset  # add 1 to make index one-indexed
        # limit idx + offset with [-5:] to keep pdb string to a minimum v
        return '\n'.join(atom.__str__() % (format(idx + offset, '5d')[-5:], *res_str,
                                           '{:8.3f}{:8.3f}{:8.3f}'.format(*coord))
                         for atom, idx, coord in zip(self.atoms, self._atom_indices, self.coords.tolist()))

    def __hash__(self) -> int:
        return hash(self.__key())

    def __copy__(self) -> Residue:  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        # Todo only copy mutable objects like
        #  _atom_indices
        #  _bb_indices
        #  _bb_and_cb_indices
        #  _heavy_atom_indices
        #  _sc_indices
        other.__dict__ = copy(self.__dict__)

        if self.is_parent():  # This Structure is the parent, it's copy should be too
            # Set the copying Structure attribute ".spawn" to indicate to dependents the "other" of this copy
            self.spawn = other
            try:
                for attr in parent_attributes:
                    other.__dict__[attr] = copy(self.__dict__[attr])
            except KeyError:  # '_residues' is not present and will be last
                pass
            other._atoms.set_attributes(_parent=other)  # Todo comment out when Atom.__copy__ needed somewhere
            # Remove the attribute spawn after other Structure containers are copied
            del self.spawn
        else:  # This Structure is a dependent, it's copy should be too
            try:
                other._parent = self.parent.spawn
            except AttributeError:  # This copy was initiated by a Structure that is not the parent
                if self._copier:  # Copy initiated by Residues container
                    pass
                else:
                    other.detach_from_parent()
                    # self.log.debug(f'The copied {type(self).__name__} is being set as a parent. It was a dependent '
                    #                f'previously')

        return other


class Residues:
    residues: np.ndarray

    def __init__(self, residues: list[Residue] | np.ndarray = None):
        if residues is None:
            self.residues = np.array([])
        elif not isinstance(residues, (np.ndarray, list)):
            raise TypeError(f"Can't initialize {type(self).__name__} with {type(residues).__name__}. Type must be a "
                            f'numpy.ndarray or list of {Residue.__name__} instances')
        else:
            self.residues = np.array(residues, dtype=np.object_)

        # can't set these here since we don't always make Residue copies
        # self.set_index()
        # self.find_prev_and_next()

    def find_prev_and_next(self):
        """Set prev_residue and next_residue attributes for each Residue. One inherently sets the other in Residue"""
        for next_idx, residue in enumerate(self.residues[:-1], 1):
            residue.next_residue = self.residues[next_idx]

    def are_dependents(self) -> bool:
        """Check if any of the Residue instance are dependents on another Structure"""
        for residue in self:
            if residue.is_dependent():
                return True
        return False

    def reindex(self, start_at: int = 0):
        """Index the Residue instances and their corresponding Atom/Coords indices according to their position

        Args:
            start_at: The Residue index to start reindexing at
        """
        self.set_index(start_at=start_at)
        self.reindex_atoms(start_at=start_at)

    def set_index(self, start_at: int = 0):
        """Index the Residue instances according to their position in the Residues container

        Args:
            start_at: The Residue index to start reindexing at
        """
        for idx, residue in enumerate(self.residues[start_at:], start_at):
            residue._index = idx

    def reindex_atoms(self, start_at: int = 0):
        """Index the Residue instances Atoms/Coords indices according to incremental Atoms/Coords index

        Args:
            start_at: The Residue index to start reindexing at
        """
        residue: Residue
        if start_at > 0:
            if start_at < self.residues.shape[0]:  # if in the Residues index range
                prior_residue = self.residues[start_at-1]
                # prior_residue.start_index = start_at
                for residue in self.residues[start_at:].tolist():
                    residue.start_index = prior_residue.atom_indices[-1]+1
                    prior_residue = residue
            else:
                # self.residues[-1].start_index = self.residues[-2].atom_indices[-1]+1
                raise IndexError(f'{Residues.reindex_atoms.__name__}: Starting index is outside of the '
                                 f'allowable indices in the Residues object!')
        else:  # when start_at is 0 or less
            prior_residue = self.residues[0]
            prior_residue.start_index = start_at
            for residue in self.residues[1:].tolist():
                residue.start_index = prior_residue.atom_indices[-1]+1
                prior_residue = residue

    def delete(self, indices: Sequence[int]):
        """Delete Residue instances from the Residues container

        Args:
            indices: The indices to delete from the Residues array
        """
        self.residues = np.delete(self.residues, indices)

    def insert(self, at: int, new_residues: list[Residue] | np.ndarray):
        """Insert Residue instances into the Residues object

        Args:
            at: The index to perform the insert at
            new_residues: The residues to include into Residues
        """
        self.residues = np.concatenate((self.residues[:at],
                                        new_residues if isinstance(new_residues, Iterable) else [new_residues],
                                        self.residues[at:]))

    def append(self, new_residues: list[Residue] | np.ndarray):
        """Append additional Residue instances into the Residues container

        Args:
            new_residues: The Residue instances to include into Residues
        Sets:
            self.residues = numpy.concatenate((self.residues, new_residues))
        """
        self.residues = np.concatenate((self.residues, new_residues))

    def reset_state(self):
        """Remove any attributes from the Residue instances that are part of the current Structure state

        This is useful for transfer of ownership, or changes in the Atom state that need to be overwritten
        """
        for residue in self:
            residue.reset_state()

    def set_attributes(self, **kwargs):
        """Set Residue attributes passed by keyword to their corresponding value"""
        for residue in self:
            for key, value in kwargs.items():
                setattr(residue, key, value)

    def set_attribute_from_array(self, **kwargs):  # UNUSED
        """For each Residue, set the attribute passed by keyword to the attribute corresponding to the Residue index in
        a provided array

        Ex: residues.attribute_from_array(mutation_rate=residue_mutation_rate_array)
        """
        for idx, residue in enumerate(self):
            for key, value in kwargs.items():
                setattr(residue, key, value[idx])

    def __copy__(self) -> Residues:  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        other.residues = self.residues.copy()
        for idx, residue in enumerate(other.residues):
            # Set an attribute to indicate the atom shouldn't be "detached"
            # since a Structure owns this Atoms instance
            residue._copier = True
            other.residues[idx] = copy(residue)
            residue._copier = False

        # Todo, these were removed as current caller of Residues.__copy__ typically calls both of them
        # other.find_prev_and_next()
        # other.set_index()

        return other

    def __len__(self) -> int:
        return self.residues.shape[0]

    def __iter__(self) -> Residue:
        yield from self.residues.tolist()


class Structure(ContainsAtomsMixin):  # Todo Polymer?
    """Structure object handles Atom/Residue/Coords manipulation of all Structure containers.
    Must pass parent and residue_indices, atoms and coords, or residues to initialize

    Polymer/Chain designation. This designation essentially means it contains Residue instances in a Residues object

    Args:
        atoms: The Atom instances which should constitute a new Structure instance
        name:
        residues: The Residue instances which should constitute a new Structure instance
        residue_indices: The indices which specify the particular Residue instances that make this Structure instance.
            Used with a parent to specify a subdivision of a larger Structure
        parent: If a Structure is creating this Structure as a division of itself, pass the parent instance
    """
    _atoms: Atoms | None
    _backbone_and_cb_indices: list[int]
    _backbone_indices: list[int]
    _ca_indices: list[int]
    _cb_indices: list[int]
    _fragment_db: FragmentDatabase
    _heavy_indices: list[int]
    _helix_cb_indices: list[int]
    _side_chain_indices: list[int]
    _contact_order: np.ndarry
    _coords_indexed_residues: np.ndarray  # list[Residue]
    _coords_indexed_residue_atoms: np.ndarray  # list[int]
    _residues: Residues | None
    _residue_indices: list[int] | None
    _sequence: str
    biomt: list
    biomt_header: str
    file_path: AnyStr | None
    name: str
    secondary_structure: str | None
    sasa: float | None
    structure_containers: list | list[str]
    state_attributes: set[str] = ContainsAtomsMixin.state_attributes | {'_sequence', '_helix_cb_indices'}

    def __init__(self, atoms: list[Atom] | Atoms = None, residues: list[Residue] | Residues = None,
                 residue_indices: list[int] = None, name: str = None,
                 file_path: AnyStr = None,
                 biomt: list = None, biomt_header: str = None,
                 **kwargs):
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        super().__init__(atoms=atoms, **kwargs)
        # self._atoms = None
        # self._atom_indices = None
        # self._coords = None
        # self._coords_indexed_residues = None
        # self._residues = None
        # self._residue_indices = None
        self.biomt = biomt if biomt else []  # list of vectors to format
        self.biomt_header = biomt_header if biomt_header else ''  # str with already formatted header
        self.file_path = file_path
        self.name = name if name not in [None, False] else f'nameless_{type(self).__name__}'
        self.secondary_structure = None
        self.sasa = None
        self.structure_containers = []

        # if log is False:  # when explicitly passed as False, use the module logger
        #     self._log = Log(logger)
        # elif isinstance(log, Log):
        #     self._log = log
        # else:
        #     self._log = Log(log)

        parent = self.parent
        if parent:  # we are setting up a dependent Structure
            # self._atoms = parent._atoms
            # self._residues = parent._residues
            try:
                residue_indices[0]
            except TypeError:
                if isinstance(self, Structures):  # Structures handles this itself
                    return
                raise ValueError(f'Argument residue_indices must be provided when constructing {type(self).__name__} '
                                 f' class from a parent')
            # must set this before setting _atom_indices
            self._residue_indices = residue_indices  # None
            # set the atom_indices from the provided residues
            self._atom_indices = [idx for residue in self.residues for idx in residue.atom_indices]
        # we are setting up a parent (independent) Structure
        elif residues:  # is not None  # assume the passed residues aren't bound to an existing Structure
            self._assign_residues(residues, atoms=atoms)
        elif self.atoms:  # assume ContainsAtomsMixin initialized .atoms, continue making Residues
            # self._assign_atoms(atoms)
            self._create_residues()
            self._set_coords_indexed()
        else:  # set up an empty Structure or let subclass handle population
            pass

    @classmethod
    def from_file(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a file with Atom records"""
        if '.pdb' in file:
            return cls.from_pdb(file, **kwargs)
        elif '.cif' in file:
            return cls.from_mmcif(file, **kwargs)
        else:
            raise NotImplementedError(f'{type(cls).__name__}: The file type {os.path.splitext(file)[-1]} is not '
                                      f'supported for parsing. Please use the supported types ".pdb" or ".cif". '
                                      f'Alternatively use those constructors instead (ex: from_pdb(), from_mmcif()) if '
                                      f'the file extension is nonsense, but the file format is respected')

    @classmethod
    def from_pdb(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a .pdb formatted file"""
        return cls(file_path=file, **read_pdb_file(file, **kwargs))

    @classmethod
    def from_mmcif(cls, file: AnyStr, **kwargs):
        """Create a new Structure from a .cif formatted file"""
        raise NotImplementedError(mmcif_error)
        return cls(file_path=file, **read_mmcif_file(file, **kwargs))

    @classmethod
    def from_residues(cls, residues: list[Residue] | Residues = None, **kwargs):
        return cls(residues=residues, **kwargs)

    @StructureBase._parent.setter
    def _parent(self, parent: StructureBase):
        """Set the Coords object while propagating changes to symmetry "mate" chains"""
        super(Structure, Structure)._parent.fset(self, parent)
        self._atoms = parent._atoms
        self._residues = parent._residues

    def detach_from_parent(self):
        """Remove the current instance from the parent that created it"""
        setattr(self, parent_variable, None)  # set parent explicitly as None
        # # Extract the coordinates
        # coords = self.coords
        # create a new, empty Coords instance
        self._coords = Coords(self.coords)
        # populate the Structure with its existing instances removed of any indexing
        self._assign_residues(self.residues, atoms=self.atoms)  # , coords=coords)
        self.reset_state()

    def get_structure_containers(self) -> dict[str, Any]:
        """Return the instance structural containers as a dictionary with attribute as key and container as value"""
        return dict(coords=self._coords, atoms=self._atoms, residues=self._residues)  # log=self._log,

    @property
    def fragment_db(self) -> FragmentDatabase:
        """The FragmentDatabase that the Fragment was created from"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: FragmentDatabase):
        # self.log.critical(f'Found fragment_db {type(fragment_db)}. '
        #                   f'isinstance(fragment_db, FragmentDatabase) = {isinstance(fragment_db, FragmentDatabase)}')
        if not isinstance(fragment_db, FragmentDatabase):
            # Todo add fragment_length, sql kwargs
            self.log.debug(f'fragment_db was set to the default since a {type(fragment_db).__name__} was passed which '
                           f'is not of the required type {FragmentDatabase.__name__}')
            fragment_db = fragment_factory.get(source=biological_interfaces, token=fragment_db)

        self._fragment_db = fragment_db

    # @property
    # def log(self) -> Logger:
    #     """Returns the log object holding the Logger"""
    #     return self._log.log

    # @log.setter
    # def log(self, log: Logger | Log):
    #     """Set the Structure, Atom, and Residue log with specified Log Object"""
    #     # try:
    #     #     log_object.log
    #     # except AttributeError:
    #     #     log_object = Log(log_object)
    #     # self._log = log_object
    #     if isinstance(log, Logger):  # prefer this protection method versus Log.log property overhead?
    #         self._log.log = log
    #     elif isinstance(log, Log):
    #         self._log = log
    #     else:
    #         raise TypeError(f'The log type ({type(log)}) is not of the specified type logging.Logger')

    def contains_hydrogen(self) -> bool:  # in Residue too
        """Returns whether the Structure contains hydrogen atoms"""
        return self.residues[0].contains_hydrogen()

    # Below properties are considered part of the Structure state
    # Todo refactor properties to below here for accounting
    @property
    def sequence(self) -> str:
        """Holds the Structure amino acid sequence"""
        # Todo if the Structure is mutated, this mechanism will cause errors, must re-extract sequence
        try:
            return self._sequence
        except AttributeError:
            self._sequence = \
                ''.join([protein_letters_3to1_extended.get(residue.type, '-') for residue in self.residues])
            return self._sequence

    @sequence.setter
    def sequence(self, sequence: str):
        self._sequence = sequence

    @property
    def structure_sequence(self) -> str:
        """Holds the Structure amino acid sequence"""
        return self.sequence

    @structure_sequence.setter
    def structure_sequence(self, sequence: str):
        self._sequence = sequence

    def _start_indices(self, at: int = 0, dtype: atom_or_residue = None):
        """Modify Structure container indices by a set integer amount

        Args:
            at: The index to insert indices at
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        try:  # To get the indices through the public property
            indices = getattr(self, f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(f'The dtype {dtype}_indices was not found the in {type(self).__name__} object. '
                                 f'Possible values of dtype are "atom" or "residue"')
        offset = at - indices[0]
        # Set the indices through the private attribute
        setattr(self, f'_{dtype}_indices', [prior_idx + offset for prior_idx in indices])

    def _insert_indices(self, at: int = 0, new_indices: list[int] = None, dtype: atom_or_residue = None):
        """Modify Structure container indices by a set integer amount

        Args:
            at: The index to insert indices at
            new_indices: The indices to insert
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        if new_indices is None:
            return  # new_indices = []
        try:
            indices = getattr(self, f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(f'The dtype {dtype}_indices was not found the Structure object. Possible values of '
                                 f'dtype are atom or residue')
        number_new = len(new_indices)
        setattr(self, f'_{dtype}_indices', indices[:at] + new_indices + [idx + number_new for idx in indices[at:]])

    def _offset_indices(self, start_at: int = 0, offset: int = None):
        """Reindex the Structure atom_indices by an offset, starting with the start_at index

        Args:
            start_at: The integer to start reindexing atom_indices at
            offset: The integer to offset the index by. For negative offset, pass a negative value
        """
        if start_at:
            try:
            # if offset:
                self._atom_indices = \
                    self._atom_indices[:start_at] + [idx + offset for idx in self._atom_indices[start_at:]]
            # else:
            except TypeError:  # None is not valide
                raise ValueError(f'{offset} is a not a valid value. Must provide an integer when re-indexing atoms '
                                 f'using the argument "start_at"')
        elif self.is_parent():
            # this shouldn't be used for a Structure object who is dependent on another Structure!
            self._atom_indices = list(range(self.number_of_atoms))
        else:
            raise ValueError(f'{self.name}: Must include start_at when re-indexing atoms from a child structure!')

    @property
    def residue_indices(self) -> list[int] | None:
        """Return the residue indices which belong to the Structure"""
        try:
            return self._residue_indices
        except AttributeError:
            return

    # @residue_indices.setter
    # def residue_indices(self, indices: list[int]):
    #     self._residue_indices = indices  # np.array(indices)

    @property
    def residues(self) -> list[Residue] | None:
        """Return the Residue instances in the Structure"""
        try:
            return self._residues.residues[self._residue_indices].tolist()
        except AttributeError:  # when self._residues isn't set or is None and doesn't have .residues
            return

    # @residues.setter
    # def residues(self, residues: Residues | list[Residue]):
    #     """Set the Structure atoms to a Residues object"""
    #     # Todo make this setter function in the same way as self._coords.replace?
    #     if isinstance(residues, Residues):
    #         self._residues = residues
    #     else:
    #         self._residues = Residues(residues)

    # def add_residues(self, residue_list):
    #     """Add Residue objects in a list to the Structure instance"""
    #     raise NotImplementedError('This function is broken')  # TODO BROKEN
    #     residues = self.residues
    #     residues.extend(residue_list)
    #     self.set_residues(residues)
    #     # Todo need to add the residue coords to coords

    def _assign_residues(self, residues: Residues | list[Residue], atoms: Atoms | list[Atom] = None, **kwargs):
        """Assign Residue instances to the Structure, create Residues object

        This will make all Residue instances (and their Atom instances) dependents of this Structure instance

        Args:
            residues: The Residue instances to assign to the Structure
            atoms: The Atom instances to assign to the Structure. Optional, will use Residues.atoms if not specified
        Keyword Args:
            coords: numpy.ndarray = None - The coordinates to assign to the Structure.
                Optional, will use Residues.coords if not specified
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
        self._assign_atoms(atoms, atoms_only=False)  # No passing of kwargs as below _populate_coords() handles
        # Done below with _residues.reindex_atoms(), not necessary here
        # if not self.file_path:  # assume this instance wasn't parsed and Atom indices are incorrect
        #     self._atoms.reindex()

        # Set proper residues attributes
        self._residue_indices = list(range(len(residues)))
        if not isinstance(residues, Residues):  # Must create the residues object
            residues = Residues(residues)

        if residues.are_dependents():  # Copy Residues object to set new attributes on each member Residue
            # This may be an additional copy with the Residues(residues) construction above
            residues = copy(residues)
            residues.reset_state()  # Clear runtime attributes
        else:
            raise RuntimeError(f'{type(self).__name__} {self.name} '
                               f'received Residue instances that are not dependents of a parent.'
                               f'This check was put in place to inspect program runtime. '
                               f'How did this situation occur that residues are not dependets?')
        self._residues = residues

        self._populate_coords(from_source='residues', **kwargs)  # coords may be passed in kwargs
        # Ensure that coordinates lengths match
        self._validate_coords()
        # Update Atom instance attributes to ensure they are dependants of this instance
        self._atoms.set_attributes(_parent=self)
        # Update Residue instance attributes to ensure they are dependants of this instance
        # Perform after _populate_coords as .coords may be set and then 'residues' .coords are overwritten
        self._residues.set_attributes(_parent=self)
        self._residues.find_prev_and_next()  # Duplicate call with "residues = copy(residues)"
        self._residues.reindex()  # Duplicates call .set_index with "residues = copy(residues)"
        self._set_coords_indexed()

    # def store_coordinate_index_residue_map(self):
    #     self.coords_indexed_residues = [(residue, res_idx) for residue in self.residues for res_idx in residue.range]

    # @property
    # def coords_indexed_residues(self):
    #     """Returns a map of the Residues and Residue atom_indices for each Coord in the Structure
    #
    #     Returns:
    #         (list[tuple[Residue, int]]): Indexed by the by the Residue position in the corresponding .coords attribute
    #     """
    #     try:
    #         return [(self._residues.residues[res_idx], res_atom_idx)
    #                 for res_idx, res_atom_idx in self._coords_indexed_residues[self._atom_indices].tolist()]
    #     except (AttributeError, TypeError):
    #         raise AttributeError('The current Structure object "%s" doesn\'t "own" it\'s coordinates. The attribute '
    #                              '.coords_indexed_residues can only be accessed by the Structure object that owns these'
    #                              ' coordinates and therefore owns this Structure' % self.name)
    #
    # @coords_indexed_residues.setter
    # def coords_indexed_residues(self, index_pairs):
    #     """Create a map of the coordinate indices to the Residue and Residue atom index"""
    #     self._coords_indexed_residues = np.array(index_pairs)

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

    # @residue_indexed_atom_indices.setter
    # def residue_indexed_atom_indices(self, indices: list[list[int]]):
    #     self._residue_indexed_atom_indices = indices

    def _set_coords_indexed(self):
        """Index the coordinates to the Residue they belong to and their associated atom_index"""
        residues_atom_idx = [(residue, res_atom_idx) for residue in self.residues for res_atom_idx in residue.range]
        self._coords_indexed_residues, self._coords_indexed_residue_atoms = map(np.array, zip(*residues_atom_idx))
        if len(self._coords_indexed_residues) != len(self._atom_indices):
            raise ValueError(f'The length of _coords_indexed_residues {len(self._coords_indexed_residues)} '
                             f'!= _atom_indices {len(self._atom_indices)}')

    @property
    def coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            return self._coords_indexed_residues[self._atom_indices].tolist()
        else:
            return self.parent._coords_indexed_residues[self._atom_indices].tolist()

    # @coords_indexed_residues.setter
    # def coords_indexed_residues(self, residues: list[Residue]):
    #     """Create a map of the coordinate indices to the Residue"""
    #     self._coords_indexed_residues = np.array(residues)

    @property
    def backbone_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each backbone Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            return self._coords_indexed_residues[self.backbone_indices].tolist()
        else:
            return self.parent._coords_indexed_residues[self.backbone_indices].tolist()

    @property
    def backbone_and_cb_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each backbone and CB Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            return self._coords_indexed_residues[self.backbone_and_cb_indices].tolist()
        else:
            return self.parent._coords_indexed_residues[self.backbone_and_cb_indices].tolist()

    @property
    def heavy_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each heavy Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            return self._coords_indexed_residues[self.heavy_indices].tolist()
        else:
            return self.parent._coords_indexed_residues[self.heavy_indices].tolist()

    @property
    def side_chain_coords_indexed_residues(self) -> list[Residue]:
        """Returns the Residue associated with each side chain Atom/Coord in the Structure

        Returns:
            Each Residue which owns the corresponding index in the .atoms/.coords attribute
        """
        if self.is_parent():
            return self._coords_indexed_residues[self.side_chain_indices].tolist()
        else:
            return self.parent._coords_indexed_residues[self.side_chain_indices].tolist()

    @property
    def coords_indexed_residue_atoms(self) -> list[int]:
        """Returns a map of the Residue atom_indices for each Coord in the Structure

        Returns:
            Index of the Atom position in the Residue for the index of the .coords attribute
        """
        # try:
        if self.is_parent():
            return self._coords_indexed_residue_atoms[self._atom_indices].tolist()
        else:
            return self.parent._coords_indexed_residue_atoms[self._atom_indices].tolist()
        # except (AttributeError, TypeError):
        #     raise AttributeError(f'The Structure "{self.name}" doesn\'t "own" it\'s coordinates. The attribute '
        #                          f'{self.coords_indexed_residue_atoms.__name__} can only be accessed by the Structure '
        #                          f'object that owns these coordinates and therefore owns this Structure')

    # @coords_indexed_residue_atoms.setter
    # def coords_indexed_residue_atoms(self, indices: list[int]):
    #     """Create a map of the coordinate indices to the Residue and Residue atom index"""
    #     self._coords_indexed_residue_atoms = np.array(indices)

    @property
    def number_of_residues(self) -> int:
        """Access the number of Residues in the Structure"""
        return len(self._residue_indices)

    def get_coords_subset(self, residue_numbers: Container[int] = None, start: int = None, end: int = None, 
                          dtype: coords_type_literal = 'ca') -> np.ndarray:
        """Return a view of a subset of the Coords from the Structure specified by a range of Residue numbers
        
        Args:
            residue_numbers: The Residue numbers to return
            start: The Residue number to start at. Inclusive
            end: The Residue number to end at. Inclusive
            dtype: The type of coordinates to get
        Returns:
            The specifiec coordinates 
        """
        coords_type = 'coords' if dtype == 'all' else f'{dtype}_coords'

        if residue_numbers is None:
            if start is not None and end is not None:
                residue_numbers = list(range(start, end+1))
            else:
                raise ValueError(f'{self.get_coords_subset.__name__}:'
                                 f' Must provide either residue_numbers or start and end')

        out_coords = []
        for residue in self.get_residues(residue_numbers):
            out_coords.append(getattr(residue, coords_type))

        return np.concatenate(out_coords)

    def _update_structure_container_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all Structure container members"""
        for structure_type in self.structure_containers:
            for structure in getattr(self, structure_type):
                for kwarg, value in kwargs.items():
                    setattr(structure, kwarg, value)

    def set_residues_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Residues in the Structure

        Keyword Args:
            numbers: (Container[int]) = None - The Atom numbers of interest
            pdb: (bool) = False - Whether to search for numbers as they were parsed (if True)
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
        """Retrieve Atom indices for Residues in the Structure. Returns all by default. If residue numbers are provided
         the selected Residues are returned

        Keyword Args:
            numbers=None (Container[int]): The Residue numbers of interest
        """
        # return [atom.index for atom in self.get_residue_atoms(numbers=numbers, **kwargs)]
        atom_indices = []
        for residue in self.get_residues(**kwargs):
            atom_indices.extend(residue.atom_indices)
        return atom_indices

    def get_residues_by_atom_indices(self, atom_indices: Iterable[int]) -> list[Residue]:
        """Retrieve Residues in the Structure specified by Atom indices. Must be the coords_owner

        Args:
            atom_indices: The atom indices to retrieve Residue objects from
        Returns:
            The Residues corresponding to the provided atom_indices
        """
        if self.is_parent():
            all_residues = self._coords_indexed_residues[atom_indices].tolist()
        else:
            all_residues = self.parent._coords_indexed_residues[atom_indices].tolist()

        return sorted(set(all_residues), key=lambda residue: residue.number)

    # Todo
    #  The functions below don't really serve a purpose... but this pseudocode would apply to similar code patterns
    #  each of the below properties could be part of same __getitem__ function
    #  ex:
    #   def __getitem__(self, value):
    #       THE CRUX OF PROBLEM IS HOW TO SEPARATE THESE GET FROM OTHER STRUCTURE GET
    #       if value.startswith('coords_indexed_')
    #       try:
    #           return getattr(self, f'_coords_indexed{value}_indices')
    #       except AttributeError:
    #           test_indice = getattr(self, f'_coords_indexed{value}_indices')
    #           setattr(self, f'_coords_indexed{value}_indices', [idx for idx, atom_idx in enumerate(self._atom_indices)
    #                                                             if atom_idx in test_indices])
    #           return getattr(self, f'_coords_indexed{value}_indices')
    #
    # @property
    # def coords_indexed_backbone_indices(self) -> list[int]:
    #     """Return backbone Atom indices from the Structure indexed to the Coords view"""
    #     try:
    #         return self._coords_indexed_backbone_indices
    #     except AttributeError:
    #         # for idx, (atom_idx, bb_idx) in enumerate(zip(self._atom_indices, self.backbone_indices)):
    #         # backbone_indices = []
    #         # for residue, res_atom_idx in self.coords_indexed_residues:
    #         #     backbone_indices.extend(residue.backbone_indices)
    #         test_indices = self.backbone_indices
    #         self._coords_indexed_backbone_indices = \
    #             [idx for idx, atom_idx in enumerate(self._atom_indices) if atom_idx in test_indices]
    #     return self._coords_indexed_backbone_indices
    #
    # @property
    # def coords_indexed_backbone_and_cb_indices(self) -> list[int]:
    #     """Return backbone and CB Atom indices from the Structure indexed to the Coords view"""
    #     try:
    #         return self._coords_indexed_backbone_and_cb_indices
    #     except AttributeError:
    #         test_indices = self.backbone_and_cb_indices
    #         self._coords_indexed_backbone_and_cb_indices = \
    #             [idx for idx, atom_idx in enumerate(self._atom_indices) if atom_idx in test_indices]
    #     return self._coords_indexed_backbone_and_cb_indices
    #
    # @property
    # def coords_indexed_cb_indices(self) -> list[int]:
    #     """Return CA Atom indices from the Structure indexed to the Coords view"""
    #     try:
    #         return self._coords_indexed_cb_indices
    #     except AttributeError:
    #         test_indices = self.cb_indices
    #         self._coords_indexed_cb_indices = \
    #             [idx for idx, atom_idx in enumerate(self._atom_indices) if atom_idx in test_indices]
    #     return self._coords_indexed_cb_indices
    #
    # @property
    # def coords_indexed_ca_indices(self) -> list[int]:
    #     """Return CB Atom indices from the Structure indexed to the Coords view"""
    #     try:
    #         return self._coords_indexed_ca_indices
    #     except AttributeError:
    #         test_indices = self.ca_indices
    #         self._coords_indexed_ca_indices = \
    #             [idx for idx, atom_idx in enumerate(self._atom_indices) if atom_idx in test_indices]
    #     return self._coords_indexed_ca_indices

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
        """Change the Atom and Residue numbering. Access the readtime Residue number in .pdb_number attribute"""
        self.renumber_atoms()
        self.pose_numbering()
        # self.renumber_residues()
        # self.log.debug(f'{self.name} was formatted in Pose numbering (residues now 1 to {self.number_of_residues})')

    def pose_numbering(self):
        """Change the Residue numbering to start at 1. Access the readtime Residue number in .pdb_number attribute"""
        for idx, residue in enumerate(self.residues, 1):
            residue.number = idx
        self.log.debug(f'{self.name} was formatted in Pose numbering (residues now 1 to {self.number_of_residues})')

    def renumber_residues(self, at: int = 1):
        """Renumber Residue objects sequentially starting with "at"

        Args:
            at: The number to start renumbering at
        """
        for idx, residue in enumerate(self.residues, at):
            residue.number = idx

    def get_residues(self, numbers: Container[int] = None, pdb: bool = False, **kwargs) -> list[Residue]:
        """Retrieve Residue objects in Structure. Returns all by default. If a list of numbers is provided, the selected
        Residues numbers are returned

        Args:
            numbers: The Residue numbers of interest
            pdb: Whether to search for numbers as they were parsed
        Returns:
            The requested Residue objects
        """
        if numbers is not None:
            if isinstance(numbers, Container):
                number_source = 'number_pdb' if pdb else 'number'
                return [residue for residue in self.residues if getattr(residue, number_source) in numbers]
            else:
                self.log.error(f'The passed numbers type "{type(numbers).__name__}" must be a Container. Returning'
                               f' all Residue instances instead')
        return self.residues

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
            # if the current residue number is the same as the prior number and the atom.type is not already present
            # We get rid of alternate conformations upon PDB load, so must be a new residue with bad numbering
            if atom.residue_number == current_residue_number and atom.type not in found_types:
                # atom_indices.append(idx)
                found_types.add(atom.type)
            else:
                if protein_backbone_atom_types.difference(found_types):  # Not an empty set, remove [start_idx:idx]
                    remove_atom_indices.extend(range(start_atom_index, idx))
                else:  # proper format
                    new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx)), parent=self))
                start_atom_index = idx
                found_types = {atom.type}  # atom_indices = [idx]
                current_residue_number = atom.residue_number

        # ensure last residue is added after stop iteration
        if protein_backbone_atom_types.difference(found_types):  # Not an empty set, remove indices [start_idx:idx]
            remove_atom_indices.extend(range(start_atom_index, idx + 1))
        else:  # proper format. For each need to increment one higher than the last v
            new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx + 1)), parent=self))

        self._residue_indices = list(range(len(new_residues)))
        self._residues = Residues(new_residues)
        # Set these attributes since this is a fresh Residues
        self._residues.find_prev_and_next()
        self._residues.set_index()

        # remove bad atom_indices
        atom_indices = self._atom_indices
        for index in remove_atom_indices[::-1]:  # ensure popping happens in reverse
            atom_indices.pop(index)
        self._atom_indices = atom_indices
        # Todo remove bad atoms
        # self._atoms.remove()

    # when alt_location parsing allowed, there may be some use to this, however above works great without alt location
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

    def residue(self, residue_number: int, pdb: bool = False) -> Residue | None:
        """Retrieve the specified Residue

        Args:
            residue_number: The number of the Residue to search for
            pdb: Whether the numbering is the parsed residue numbering or current
        """
        number_source = 'number_pdb' if pdb else 'number'
        for residue in self.residues:
            if getattr(residue, number_source) == residue_number:
                return residue

    @property
    def n_terminal_residue(self) -> Residue:
        """Retrieve the Residue from the n-termini"""
        return self.residues[0]

    @property
    def c_terminal_residue(self) -> Residue:
        """Retrieve the Residue from the c-termini"""
        return self.residues[-1]

    def add_ideal_helix(self, termini: termini_literal = 'n', length: int = 5):
        """Add an ideal helix to a termini given by a certain length
        
        Args:
            termini: The termini to add the ideal helix to
            length: The length of the addition
        """
        self.log.debug(f'Adding ideal helix to {termini}-terminus of {self.name}')

        if length > 10:
            raise ValueError(f'{self.add_ideal_helix.__name__}: length can not be greater than 10')

        alpha_helix_15_struct = Structure(atoms=alpha_helix_15)

        align_length = 4  # 4 as the get_coords_subset is inclusive
        if termini == 'n':
            first_residue = self.n_terminal_residue
            first_residue_number = first_residue.number
            last_residue_number = first_residue_number+align_length
            fixed_coords = self.get_coords_subset(start=first_residue_number, end=last_residue_number, 
                                                  dtype='backbone')
            helix_align_start = 11
            helix_align_end = 15
            helix_start = helix_align_start - length
            # helix_end = helix_align_start-1
            moving_coords = alpha_helix_15_struct.get_coords_subset(start=helix_align_start, end=helix_align_end, dtype='backbone')
            rmsd, rot, tx = superposition3d(fixed_coords, moving_coords)
            alpha_helix_15_struct.transform(rotation=rot, translation=tx)

            # Add residues then renumber
            self._residues.insert(0, alpha_helix_15_struct.get_residues(
                list(range(helix_start, helix_align_start))))  # helix_end+1
            self._residues.reindex()  # .set_index()
            # Rename new residues to self.chain
            self.set_residues_attributes(chain=first_residue.chain)

        elif termini == 'c':
            last_residue = self.c_terminal_residue
            last_residue_number = last_residue.number
            first_residue_number = last_residue_number-align_length
            fixed_coords = self.get_coords_subset(start=first_residue_number, end=last_residue_number, 
                                                  dtype='backbone')
            helix_align_start = 1
            helix_align_end = 5
            # helix_start = helix_align_end+1
            helix_end = helix_align_end + length
            moving_coords = alpha_helix_15_struct.get_coords_subset(start=helix_align_start, end=helix_align_end,
                                                                    dtype='backbone')
            rmsd, rot, tx = superposition3d(fixed_coords, moving_coords)
            alpha_helix_15_struct.transform(rotation=rot, translation=tx)

            # Add residues then renumber
            self._residues.append(
                alpha_helix_15_struct.get_residues(list(range(helix_align_end, helix_end+1))))  # helix_start
            self._residues.reindex()  # .set_index()
            # Rename new residues to self.chain
            self.set_residues_attributes(chain=last_residue.chain)
        else:
            raise ValueError('termini must be wither "n" or "c"')

    @property
    def radius(self) -> float:
        """The furthest point from the center of mass of the Structure"""
        return np.max(np.linalg.norm(self.coords - self.center_of_mass, axis=1))

    def get_residue_atoms(self, numbers: Container[int] = None, **kwargs) -> list[Atom]:
        """Return the Atoms contained in the Residue objects matching a set of residue numbers

        Args:
            numbers: The residue numbers to search for
        Returns:
            The Atom instances belonging to the Residue instances
        """
        atoms = []
        for residue in self.get_residues(numbers=numbers, **kwargs):
            atoms.extend(residue.atoms)
        return atoms

    def residue_from_pdb_numbering(self, residue_number: int) -> Residue | None:
        """Returns the Residue object from the Structure according to PDB residue number

        Args:
            residue_number: The number of the Residue to search for
        """
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue

    def residue_number_from_pdb(self, residue_number: int) -> int | None:
        """Returns the Residue 'pose number' from the parsed number

        Args:
            residue_number: The number of the Residue to search for
        """
        for residue in self.residues:
            if residue.number_pdb == residue_number:
                return residue.number

    def residue_number_to_pdb(self, residue_number: int) -> int | None:
        """Returns the Residue parsed number from the 'pose number'

        Args:
            residue_number: The number of the Residue to search for
        """
        for residue in self.residues:
            if residue.number == residue_number:
                return residue.number_pdb

    def mutate_residue(self, residue: Residue = None, index: int = None, number: int = None, to: str = 'ALA', **kwargs)\
            -> list[int] | list:
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format
        Args:
            residue: A Residue object to mutate
            index: A Residue index to select the Residue instance of interest by
            number: A Residue number to select the Residue instance of interest by
            to: The type of amino acid to mutate to
        Keyword Args:
            pdb: bool = False - Whether to pull the Residue by PDB number
        Returns:
            The indices of the Atoms being removed from the Structure
        """
        # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
        to = protein_letters_1to3.get(to.upper(), to.upper())

        if index is not None:
            try:
                residue = self.residues[index]
            except IndexError:
                raise IndexError(f'The residue index {index} is out of bounds for the {type(self).__name__} '
                                 f'{self.name} with {self.number_of_residues} residues')
        elif number is not None:
            residue = self.residue(number, **kwargs)

        if residue is None:
            raise DesignError(f"Can't {self.mutate_residue.__name__} without Residue instance, index, or number")

        residue.type = to
        for atom in residue.atoms:
            atom.residue_type = to

        # Find the corresponding Residue Atom indices to delete. Currently, using side-chain and letting Rosetta handle
        delete_indices = residue.side_chain_indices
        if not delete_indices:  # There are no indices
            return []

        # Remove indices from the Residue, and Structure atom_indices
        residue_delete_index = residue.atom_indices.index(delete_indices[0])
        for _ in iter(delete_indices):
            residue.atom_indices.pop(residue_delete_index)

        # If this Structure isn't parent, then parent Structure must update the atom_indices
        atom_delete_index = self._atom_indices.index(delete_indices[0])
        self._offset_indices(start_at=atom_delete_index, offset=-len(delete_indices))

        # Re-index all succeeding Atom and Residue instance indices
        self._coords.delete(delete_indices)
        self._atoms.delete(delete_indices)
        self._atoms.reindex(start_at=atom_delete_index)
        self._residues.reindex_atoms(start_at=residue.index)

        return delete_indices

    def insert_residue_type(self, residue_type: str, at: int = None, chain: str = None) -> Residue:
        """Insert a standard Residue type into the Structure based on Pose numbering (1 to N) at the origin.
        No structural alignment is performed!

        Args:
            residue_type: Either the 1 or 3 letter amino acid code for the residue in question
            at: The pose numbered location which a new Residue should be inserted into the Structure
            chain: The chain identifier to associate the new Residue with
        Returns:
            The newly inserted Residue object
        """
        # Todo solve this issue for self.is_dependents()
        #  this check and error really isn't True with the Residues object shared. It can be overcome...
        if self.is_dependent():
            raise DesignError(f"This Structure '{self.name}' is not the owner of it's attributes and therefore cannot "
                              'handle residue insertion!')
        # Convert incoming aa to residue index so that AAReference can fetch the correct amino acid
        reference_index = \
            protein_letters_alph1.find(protein_letters_3to1_extended.get(residue_type, residue_type.upper()))
        if reference_index == -1:
            raise IndexError(f'{self.insert_residue_type.__name__} of residue_type "{residue_type}" is not allowed')
        if at < 1:
            raise IndexError(f'{self.insert_residue_type.__name__} at index {at} < 1 is not allowed')

        # Grab the reference atom coordinates and push into the atom list
        new_residue = copy(reference_aa.residue(reference_index))
        new_residue.number = at
        residue_index = at - 1  # since at is one-indexed integer, take from pose numbering to zero-indexed
        # insert the new_residue coords and atoms into the Structure Atoms
        self._coords.insert(new_residue.start_index, new_residue.coords)
        self._atoms.insert(new_residue.start_index, new_residue.atoms)
        self._atoms.reindex(start_at=new_residue.start_index)
        # insert the new_residue into the Structure Residues
        self._residues.insert(residue_index, [new_residue])
        self._residues.reindex(start_at=residue_index)  # .set_index()
        # after coords, atoms, residues insertion into "_" containers, set parent to self
        new_residue.parent = self
        # set new residue_indices and atom_indices
        self._insert_indices(at=residue_index, new_indices=[residue_index], dtype='residue')
        self._insert_indices(at=new_residue.start_index, new_indices=new_residue.atom_indices, dtype='atom')
        # self._atom_indices = self._atom_indices.insert(new_residue.start_index, idx + new_residue.start_index)
        self.renumber()

        # find the prior and next residues and add attributes
        if residue_index:  # not 0
            prior_residue = self.residues[residue_index - 1]
            new_residue.prev_residue = prior_residue
        else:  # n-termini = True
            prior_residue = None

        try:
            next_residue = self.residues[residue_index + 1]
            new_residue.next_residue = next_residue
        except IndexError:  # c_termini = True
            if not prior_residue:  # insertion on an empty Structure? block for now to simplify chain identification
                raise DesignError(f"Can't insert_residue_type for an empty {type(self).__name__} class")
            next_residue = None

        # set the new chain_id, number_pdb. Must occur after self._residue_indices update if chain isn't provided
        chain_assignment_error = "Can't solve for the new Residue polymer association automatically! If the new " \
                                 'Residue is at a Structure termini in a multi-Structure Structure container, you must'\
                                 ' specify which Structure it belongs to by passing chain='
        if chain is not None:
            new_residue.chain = chain
        else:  # try to solve without it...
            if prior_residue and next_residue:
                if prior_residue.chain == next_residue.chain:
                    res_with_info = prior_residue
                else:  # we have a discrepancy which means this is an internal termini
                    raise DesignError(chain_assignment_error)
            else:  # we can solve as this represents an absolute termini case
                res_with_info = prior_residue if prior_residue else next_residue
            new_residue.chain = res_with_info.chain
            new_residue.number_pdb = prior_residue.number_pdb + 1 if prior_residue else next_residue.number_pdb - 1

        if self.secondary_structure:
            # ASSUME the insertion is disordered and coiled segment
            self.secondary_structure = \
                self.secondary_structure[:residue_index] + 'C' + self.secondary_structure[residue_index:]

        # Todo solve this v for self.is_dependents()
        # re-index the coords and residues map
        residues_atom_idx = [(residue, res_atom_idx) for residue in self.residues for res_atom_idx in residue.range]
        self._coords_indexed_residues, self._coords_indexed_residue_atoms = map(np.array, zip(*residues_atom_idx))
        # range_idx = prior_range_idx = 0
        # residue_indexed_ranges = []
        # for residue in self.residues:
        #     range_idx += residue.number_of_atoms
        #     residue_indexed_ranges.append(list(range(prior_range_idx, range_idx)))
        #     prior_range_idx = range_idx
        # self.residue_indexed_atom_indices = residue_indexed_ranges

        return new_residue

    # def get_structure_sequence(self):
    #     """Returns the single AA sequence of Residues found in the Structure. Handles odd residues by marking with '-'
    #
    #     Returns:
    #         (str): The amino acid sequence of the Structure Residues
    #     """
    #     return ''.join([protein_letters_3to1_extended.get(res.type, '-') for res in self.residues])

    def translate(self, translation: list[float] | np.ndarray, **kwargs):
        """Perform a translation to the Structure ensuring only the Structure container of interest is translated
        ensuring the underlying coords are not modified

        Args:
            translation: The first translation to apply, expected array shape (3,)
        """
        self.coords = self.coords + translation

    def rotate(self, rotation: list[list[float]] | np.ndarray, **kwargs):
        """Perform a rotation to the Structure ensuring only the Structure container of interest is rotated ensuring the
        underlying coords are not modified

        Args:
            rotation: The first rotation to apply, expected array shape (3, 3)
        """
        self.coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))  # Essentially a transpose

    def transform(self, rotation: list[list[float]] | np.ndarray = None, translation: list[float] | np.ndarray = None,
                  rotation2: list[list[float]] | np.ndarray = None, translation2: list[float] | np.ndarray = None,
                  **kwargs):
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
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))  # Essentially a transpose
        else:
            new_coords = self.coords  # No need to copy as this is a view

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += translation

        if rotation2 is not None:  # required for np.ndarray or None checks
            np.matmul(new_coords, rotation2.swapaxes(-2, -1), out=new_coords)  # Essentially a transpose

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += translation2

        self.coords = new_coords

    def get_transformed_copy(self, rotation: list[list[float]] | np.ndarray = None,
                             translation: list[float] | np.ndarray = None,
                             rotation2: list[list[float]] | np.ndarray = None,
                             translation2: list[float] | np.ndarray = None) -> Structure:
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
        if rotation is not None:  # required for np.ndarray or None checks
            new_coords = np.matmul(self.coords, np.transpose(rotation))  # This allows list to be passed...
            # new_coords = np.matmul(self.coords, rotation.swapaxes(-2, -1))
        else:
            new_coords = self.coords  # No need to copy as this is a view

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            np.matmul(new_coords, np.transpose(rotation2), out=new_coords)  # This allows list to be passed...
            # np.matmul(new_coords, rotation2.swapaxes(-2, -1), out=new_coords)

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        new_structure = copy(self)
        new_structure.coords = new_coords

        return new_structure

    # Todo this should be a property keeping in line with Residue.local_density, however the arguments are important at
    #  this level... Need to reconcile the API for this
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
        for residue, atom_neighbor_counts in zip(coords_indexed_residues, all_atom_counts_query):  # should be same len
            if residue == current_residue:
                current_residue.local_density += atom_neighbor_counts
            else:  # we have a new residue
                current_residue.local_density /= current_residue.number_of_heavy_atoms  # find the average
                current_residue = residue
                current_residue.local_density += atom_neighbor_counts
        # ensure the last residue is calculated
        current_residue.local_density /= current_residue.number_of_heavy_atoms  # find the average

        return [residue.local_density for residue in self.residues]

    def is_clash(self, measure: coords_type_literal = 'backbone_and_cb', distance: float = 2.1,
                 warn: bool = True, silence_exceptions: bool = False,
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
        # Todo switch measure:
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
        # Todo make self.atom_tree a property?
        atom_tree = BallTree(self.coords)
        residues = self.residues
        atoms = self.atoms
        measured_clashes, other_clashes = [], []
        clashes = False

        def return_true(): return True

        any_clashes: Callable[[Iterable[int]], bool]
        """Local helper to separate clash reporting from clash generation"""

        clash_msg = f'{self.name} contains Residue {measure} atom clashes at a {distance}A distance'
        if warn:
            def any_clashes(_clash_indices: Iterable[int]):
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

                # Set the global clashes (clashes = return ...) while checking global clashes against new_clashes
                return clashes or new_clashes
        else:  # Raise a ClashError as we can immediately stop execution if this is the case
            def any_clashes(_clash_indices: Iterable[int]):
                for clashing_idx in _clash_indices:
                    if getattr(atoms[clashing_idx], f'is_{measure}', return_true)():
                        raise ClashError(clash_msg)

                return clashes

        try:
            # check first and last residue with different considerations given covalent bonds
            residue = residues[0]
            # query the first residue with chosen coords type against the atom_tree
            residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
            # reduce the dimensions and format as a single array
            all_contacts = {atom_contact for residue_contacts in residue_atom_contacts
                            for atom_contact in residue_contacts}
            # We must subtract the N and C atoms from the adjacent residues for each residue as these are within a bond
            clashes = any_clashes(
                all_contacts.difference(residue.atom_indices
                                        + [residue.next_residue.n_atom_index]))

            # perform routine for all middle residues
            for residue in residues[1:-1]:  # avoid first and last since no prev_ or next_residue
                residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
                all_contacts = {atom_contact for residue_contacts in residue_atom_contacts
                                for atom_contact in residue_contacts}
                prior_residue = residue.prev_residue
                clashes = any_clashes(
                    all_contacts.difference([prior_residue.o_atom_index,
                                             prior_residue.c_atom_index,
                                             residue.next_residue.n_atom_index]
                                            + residue.atom_indices)
                    )

            residue = residues[-1]
            residue_atom_contacts = atom_tree.query_radius(getattr(residue, coords_type), distance)
            all_contacts = {atom_contact for residue_contacts in residue_atom_contacts
                            for atom_contact in residue_contacts}
            prior_residue = residue.prev_residue
            clashes = any_clashes(
                all_contacts.difference([prior_residue.o_atom_index, prior_residue.c_atom_index]
                                        + residue.atom_indices))

            if clashes:
                if measured_clashes:
                    bb_info = '\n\t'.join(f'Residue {residue.number:5d}: {atom.get_atom_record()}'
                                          for residue, atom in measured_clashes)
                    self.log.error(f'{self.name} contains {len(measured_clashes)} {measure} clashes from the following '
                                   f'Residues to the corresponding Atom:\n\t{bb_info}')
                    raise ClashError(clash_msg)
            # else:
                if other_clashes:
                    sc_info = '\n\t'.join(f'Residue {residue.number:5d}: {atom.get_atom_record()}'
                                          for residue, atom in other_clashes)
                    self.log.warning(f'{self.name} contains {len(other_clashes)} {other} clashes between the '
                                     f'following Residues:\n\t{sc_info}')
        except ClashError as error:  # This was raised from any_clashes()
            if silence_exceptions:
                return True
            else:
                raise error

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
        cmd = [freesasa_exe_path, f'--format={out_format}', '--probe-radius', str(probe_radius),
               '-c', freesasa_config_path, '--n-threads=2'] + include_hydrogen
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
            for line_split in map(str.split, sasa_output[5:-2]):  # slice could remove need for if ATOM
                if line_split[0] == 'ATOM':  # this seems necessary as MODEL can be added if MODEL is written
                    atoms[if_idx].sasa = float(line_split[-1])
                    if_idx += 1
        else:
            residues = self.residues
            for idx, line in enumerate(sasa_output[1:-1]):  # slice removes need for if == 'SEQ'
                if line[:3] == 'SEQ':  # doesn't seem to be the case that we can do this ^
                    residues[if_idx].sasa = float(line[16:])
                    if_idx += 1
        # Todo change to sasa property to call this automatically if AttributeError?
        self.sasa = sum([residue.sasa for residue in self.residues])

    @property
    def surface_residues(self, relative_sasa_thresh: float = 0.25, **kwargs) -> list[Residue]:
        """Get the Residue instances that reside on the surface of the molecule

        Args:
            relative_sasa_thresh: The area threshold that the Residue should have before it is considered "surface"
                Default cutoff percent is based on Levy, E. 2010
        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated
        Returns:
            The surface Residue instances
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        # return [residue.number for residue in self.residues if residue.sasa > sasa_thresh]
        return [residue for residue in self.residues if residue.relative_sasa >= relative_sasa_thresh]

    @property
    def core_residues(self, relative_sasa_thresh: float = 0.25, **kwargs) -> list[Residue]:
        """Get the Residue instances that reside in the core of the molecule

        Args:
            relative_sasa_thresh: The area threshold that the Residue should fall below before it is considered "core"
                Default cutoff percent is based on Levy, E. 2010
        Keyword Args:
            atom: bool = True - Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius: float = 1.4 - The radius which surface area should be generated
        Returns:
            The core Residue instances
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        # return [residue.number for residue in self.residues if residue.sasa > sasa_thresh]
        return [residue for residue in self.residues if residue.relative_sasa < relative_sasa_thresh]

    # def get_residue_surface_area(self, residue_number, probe_radius=2.2):
    #     """Get the surface area for specified residues
    #
    #     Returns:
    #         (float): Angstrom^2 of surface area
    #     """
    #     if not self.sasa:
    #         self.get_sasa(probe_radius=probe_radius)
    #
    #     # return self.sasa[self.residues.index(residue_number)]
    #     return self.sasa[self.residues.index(residue_number)]

    def get_surface_area_residues(self, residues: list[Residue] = None, numbers: list[int] = None,
                                  dtype: polarity_types_literal = 'polar', **kwargs) -> float:
        """Get the surface area for specified residues

        Args:
            residues: The Residues to sum
            numbers: The Residue numbers to sum. Only used if residues is not provided
            dtype: The type of area to query. Can be 'polar' or 'apolar'
        Keyword Args:
            pdb=False (bool): Whether to search for numbers as they were parsed
            atom=True (bool): Whether the output should be generated for each atom.
                If False, will be generated for each Residue
            probe_radius=1.4 (float): The radius which surface area should be generated
        Returns:
            Angstrom^2 of surface area
        """
        if not self.sasa:
            self.get_sasa(**kwargs)

        if not residues:
            residues = self.get_residues(numbers=numbers, **kwargs)
        # return sum([sasa for residue_number, sasa in zip(self.sasa_residues, self.sasa) if residue_number in numbers])
        return sum([getattr(residue, dtype) for residue in residues if residue.number in numbers])

    def errat(self, out_path: AnyStr = os.getcwd()) -> tuple[float, np.ndarray]:
        """Find the overall and per residue Errat accuracy for the given Structure

        Args:
            out_path: The path where Errat files should be written
        Returns:
            Overall Errat score, Errat value/residue array
        """
        # name = 'errat_input-%s-%d.pdb' % (self.name, random() * 100000)
        # current_struc_file = self.write(out_path=os.path.join(out_path, name))
        # errat_cmd = [errat_exe_path, os.path.splitext(name)[0], out_path]  # for writing file first
        # os.system('rm %s' % current_struc_file)
        out_path = out_path if out_path[-1] == os.sep else out_path + os.sep  # errat needs trailing "/"
        errat_cmd = [errat_exe_path, out_path]  # for passing atoms by stdin
        # p = subprocess.Popen(errat_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # out, err = p.communicate(input=self.get_atom_record().encode('utf-8'))
        # logger.info(self.get_atom_record()[:120])
        iteration = 1
        all_residue_scores = []
        while iteration < 5:
            p = subprocess.run(errat_cmd, input=self.get_atom_record(), encoding='utf-8', capture_output=True)
            all_residue_scores = p.stdout.strip().split('\n')
            # Subtract one due to the addition of overall score
            if len(all_residue_scores)-1 == self.number_of_residues:
                break
            iteration += 1

        if iteration == 5:
            self.log.error(f"{self.errat.__name__} couldn't generate the correct output length. "
                           f'({len(all_residue_scores)-1}) != number_of_residues ({self.number_of_residues})')
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
            return 0., np.array([0. for _ in range(self.number_of_residues)])

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
        p = subprocess.Popen([stride_exe_path, current_struc_file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()
        os.system(f'rm {current_struc_file}')

        if out:
            if to_file:
                with open(to_file, 'wb') as f:
                    f.write(out)
            stride_output = out.decode('utf-8').split('\n')
        else:
            self.log.warning(f'{self.name}: No secondary structure assignment found with Stride')
            return
        # except:
        #     stride_out = None

        # if stride_out is not None:
        #     lines = stride_out.split('\n')
        residue_idx = 0
        residues = self.residues
        for line in stride_output:
            # residue_idx = int(line[10:15])
            if line[0:3] == 'ASG':
                # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
                # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
                # self.chain(line[9:10]).residue(int(line[10:15].strip())).secondary_structure = line[24:25]
                residues[residue_idx].secondary_structure = line[24:25]
                residue_idx += 1
        self.secondary_structure = ''.join(residue.secondary_structure for residue in residues)

    def parse_stride(self, stride_file: AnyStr, **kwargs):
        """From a Stride file, parse information for residue level secondary structure assignment

        Sets:
            self.secondary_structure
        """
        with open(stride_file, 'r') as f:
            stride_output = f.readlines()

        # residue_idx = 0
        # residues = self.residues
        for line in stride_output:
            # residue_idx = int(line[10:15])
            if line[0:3] == 'ASG':
                # residue_idx = int(line[15:20])  # one-indexed, use in Structure version...
                # line[10:15].strip().isdigit():  # residue number -> line[10:15].strip().isdigit():
                self.residue(int(line[10:15].strip()), pdb=True).secondary_structure = line[24:25]
                # residues[residue_idx].secondary_structure = line[24:25]
                # residue_idx += 1
        self.secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)

    def is_termini_helical(self, termini: termini_literal = 'n', window: int = 5) -> int:
        """Using assigned secondary structure, probe for a helical C-termini using a segment of 'window' residues

        Args:
            termini: Either 'n' or 'c' should be specified
            window: The segment size to search
        Returns:
            Whether the termini has a stretch of helical residues with length of the window (1) or not (0)
        """
        residues = list(reversed(self.residues)) if termini.lower() == 'c' else self.residues
        # if not residues[0].secondary_structure:
        #     raise DesignError(f'You must call {self.get_secondary_structure.__name__} on {self.name} before querying '
        #                       f'for helical termini')
        term_window = ''.join(residue.secondary_structure for residue in residues[:window*2])
        if 'H'*window in term_window:
            return 1  # True
        else:
            return 0  # False

    def get_secondary_structure(self):
        if self.secondary_structure:
            return self.secondary_structure
        else:
            self.fill_secondary_structure()
            if self.secondary_structure:  # check if there is at least 1 secondary struc assignment
                return self.secondary_structure
            else:
                return

    def fill_secondary_structure(self, secondary_structure=None):
        if secondary_structure:
            self.secondary_structure = secondary_structure
            if len(self.secondary_structure) == self.number_of_residues:
                for idx, residue in enumerate(self.residues):
                    residue.secondary_structure = secondary_structure[idx]
            else:
                self.log.warning(f'The passed secondary_structure length ({len(self.secondary_structure)}) is not equal'
                                 f' to the number of residues ({self.number_of_residues}). Recalculating...')
                self.stride()  # we tried for efficiency, but its inaccurate, recalculate
        else:
            if self.residues[0].secondary_structure:
                self.secondary_structure = ''.join(residue.secondary_structure for residue in self.residues)
            else:
                self.stride()

    def termini_proximity_from_reference(self, termini: termini_literal = 'n', reference: np.ndarray = origin) -> float:
        """From an Entity, find the orientation of the termini from the origin (default) or from a reference point

        Args:
            termini: Either 'n' or 'c' should be specified
            reference: The reference where the point should be measured from
        Returns:
            1 if the termini is further from the reference, -1 if the termini is closer to the reference
        """
        # Todo, this is pretty coarse logic. Calculate from N number of residues up or downstream? That has issues
        if termini.lower() == 'n':
            residue_coords = self.residues[0].n_coords
        elif termini.lower() == 'c':
            residue_coords = self.residues[-1].c_coords
        else:
            raise ValueError(f'Termini must be either "n" or "c", not "{termini}"')

        max_distance = self.distance_from_reference(reference=reference, measure='max')
        min_distance = self.distance_from_reference(reference=reference, measure='min')
        coord_distance = np.linalg.norm(residue_coords - reference)
        if abs(coord_distance - max_distance) < abs(coord_distance - min_distance):
            return 1  # termini further from the reference
        else:
            return -1  # termini closer to the reference

    def distance_from_reference(self, reference: np.ndarray = origin, measure: str = 'mean') -> float:
        """From a Structure, find the furthest coordinate from the origin (default) or from a reference.

        Args:
            reference: The reference where the point should be measured from. Default is origin
            measure: The measurement to take with respect to the reference. Could be 'mean', 'min', 'max', or any
                numpy function to describe computed distance scalars
        Returns:
            The distance from the reference point to the furthest point
        """
        return getattr(np, measure)(np.linalg.norm(self.coords - reference, axis=1))

    def get_atom_record(self, **kwargs) -> str:
        """Provide the Structure Atoms as a PDB file string

        Keyword Args:
            pdb: bool = False - Whether the Residue representation should use the number at file parsing
            chain: str = None - The chain ID to use
            atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
        Returns:
            The archived .pdb formatted ATOM records for the Structure
        """
        return '\n'.join(residue.__str__(**kwargs) for residue in self.residues)

    def format_header(self, **kwargs) -> str:  # Todo move to PDB/Model (parsed) and Entity (oligomer)?
        """Return the BIOMT record based on the Structure

        Returns:
            The header with PDB file formatting
        """
        return super().format_header(**kwargs) + self.format_biomt(**kwargs)

    def format_biomt(self, **kwargs) -> str:  # Todo move to PDB/Model (parsed) and Entity (oligomer)?
        """Return the BIOMT record for the Structure if there was one parsed

        Returns:
            The BIOMT REMARK 350 with PDB file formatting
        """
        if self.biomt_header != '':
            return self.biomt_header
        elif self.biomt:
            return '%s\n' \
                % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}            '
                            .format(v_idx, m_idx, *vec)
                            for m_idx, matrix in enumerate(self.biomt, 1) for v_idx, vec in enumerate(matrix, 1))
        else:
            return self.biomt_header  # This is '' if not set

    # def write_header(self, file_handle: IO, header: str = None, **kwargs):
    #     """Handle writing of Structure header information to the file
    #
    #     Args:
    #         file_handle: An open file object where the header should be written
    #         header: A string that is desired at the top of the file
    #     """
    #     _header = self.format_header(**kwargs)  # biomt and seqres
    #     if header and isinstance(header, Iterable):
    #         if isinstance(header, str):  # used for cryst_record now...
    #             _header += (header if header[-2:] == '\n' else f'{header}\n')
    #         # else:  # TODO
    #         #     location.write('\n'.join(header))
    #     if _header != '':
    #         file_handle.write(_header)

    # def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, **kwargs) -> str | None:
    #     #     header: str = None, increment_chains: bool = False,
    #     """Write Structure Atoms to a file specified by out_path or with a passed file_handle
    #
    #     If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
    #     Args:
    #         out_path: The location where the Structure object should be written to disk
    #         file_handle: Used to write Structure details to an open FileObject
    #     Keyword Args
    #         header: None | str - A string that is desired at the top of the .pdb file
    #         pdb: bool = False - Whether the Residue representation should use the number at file parsing
    #         chain: str = None - The chain ID to use
    #         atom_offset: int = 0 - How much to offset the atom number by. Default returns one-indexed
    #     Returns:
    #         The name of the written file if out_path is used
    #     """
    #     if file_handle:
    #         file_handle.write(f'{self.get_atom_record(**kwargs)}\n')
    #         return None
    #     else:  # out_path always has default argument current working directory
    #         with open(out_path, 'w') as outfile:
    #             self.write_header(outfile, **kwargs)
    #             outfile.write(f'{self.get_atom_record(**kwargs)}\n')
    #         return out_path

    def get_fragments(self, residues: list[Residue] = None, residue_numbers: list[int] = None, fragment_db: int = None,
                      **kwargs) -> list[MonoFragment]:
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

        # residues = self.residues
        # ca_stretches = [[residues[idx + i].ca for i in range(-2, 3)] for idx, residue in enumerate(residues)]
        # compare ca_stretches versus monofrag ca_stretches
        # monofrag_array = repeat([ca_stretch_frag_index1, ca_stretch_frag_index2, ...]
        # monofrag_indices = filter_euler_lookup_by_zvalue(ca_stretches, monofrag_array, z_value_func=fragment_overlap,
        #                                                  max_z_value=rmsd_threshold)
        if fragment_db is None:
            raise ValueError(f"Can't assign fragments without passing fragment_db")
        else:
            try:
                fragment_db.representatives
            except AttributeError:
                raise TypeError(f'The passed fragment_db is not of the required type "FragmentDatabase"')

        fragment_length = fragment_db.fragment_length
        fragment_range = range(*fragment_db.fragment_range)
        fragments = []
        for residue_number in residue_numbers:
            frag_residues = self.get_residues(numbers=[residue_number + i for i in fragment_range])

            if len(frag_residues) == fragment_length:
                fragment = MonoFragment(residues=frag_residues, fragment_db=fragment_db, **kwargs)
                if fragment.i_type:
                    fragments.append(fragment)

        return fragments

    # Preferred method using Residue fragments
    def get_fragment_residues(self, residues: list[Residue] = None, residue_numbers: list[int] = None,
                              # fragment_length: int = 5,
                              fragment_db: object = None,
                              # fragment_db: FragmentDatabase = None, # Todo typing with FragmentDatabase
                              rmsd_thresh: float = Fragment.rmsd_thresh, **kwargs) -> list | list[Residue]:
        """Assign a Fragment type to Residues in the Structure, as identified from a FragmentDatabase, then return them

        Args:
            residues: The specific Residues to search for
            residue_numbers: The specific residue numbers to search for
            fragment_db: The FragmentDatabase with representative fragment types to query the Residue against
            rmsd_thresh: The threshold for which a rmsd should fail to produce a fragment match
        Sets:
            Each Fragment Residue instance self.guide_coords, self.i_type, self.fragment_length
        Returns:
            The Residue instances that match Fragment representatives from the Structure
        """
        #            fragment_length: The length of the fragment observations used
        if fragment_db is None:
            raise ValueError(f"Can't assign fragments without passing fragment_db")
        else:
            try:
                fragment_db.representatives
            except AttributeError:
                raise TypeError(f'The passed fragment_db is not of the required type "FragmentDatabase"')

        if residue_numbers is not None:
            residues = self.get_residues(numbers=residue_numbers)

        # get iterable of residues
        residues = self.residues if residues is None else residues

        # Get neighboring ca coords on each side by retrieving flanking residues. If not fragment_length, we remove
        fragment_length = fragment_db.fragment_length
        frag_lower_range, frag_upper_range = fragment_db.fragment_range

        # Iterate over the residues in reverse to remove any indices that are missing and convert to coordinates
        viable_residues, residues_ca_coords = [], []
        # for idx, residue in zip(range(len(residues)-1, -1, -1), residues[::-1]):
        for residue in residues:
            residue_set = \
                residue.get_upstream(frag_lower_range) + [residue] + residue.get_downstream(frag_upper_range-1)
            if len(residue_set) == fragment_length:
                residues_ca_coords.append([residue.ca_coords for residue in residue_set])
                viable_residues.append(residue)
            # else:
            #     popped = residues.pop(idx)
            #     print('Popping', idx, 'which was:', popped.number)

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
                    # min_rmsd, residue.rotation, residue.translation = rmsd, rot, tx

            if residue.frag_type:
                # residue.guide_coords = \
                #     np.matmul(Fragment.template_coords, np.transpose(residue.rotation)) + residue.translation
                residue.fragment_db = fragment_db
                # residue._fragment_coords = residue_ca_coord_set
                residue._fragment_coords = fragment_db.representatives[residue.frag_type].backbone_coords
                found_fragments.append(residue)

        return found_fragments

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
            raise ValueError(f"Can't set {self.contact_order.__name__} with a sequence length ({len(contact_order)})"
                             f'!= to the number of residues, {self.number_of_residues}')
        for residue, contact_order in zip(residues, contact_order):
            residue.contact_order = contact_order

    # distance of 6 angstroms between heavy atoms was used for 1998 contact order work,
    # subsequent residue wise contact order has focused on the Cb-Cb heuristic of 12 A
    # KM thinks that an atom-atom based measure is more accurate, see below for alternative method
    # The BallTree creation is the biggest time cost regardless
    def contact_order_per_residue(self, sequence_distance_cutoff: int = 2, distance: float = 6.) -> list[float]:
        """Calculate the contact order on a per-residue basis using calculated heavy atom contacts

        Args:
            sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
            distance: The distance in angstroms to measure atomic contact distances in contact
        Returns:
            The floats representing the contact order for each Residue in the Structure
        """
        # Get heavy Atom coordinates
        coords = self.heavy_coords
        # Make and query a tree
        tree = BallTree(coords)
        query = tree.query_radius(coords, distance)

        residues = self.residues
        # In case this was already called, we should set all to 0.
        for residue in residues:
            residue.contact_order = 0.

        heavy_atom_coords_indexed_residues = self.heavy_coords_indexed_residues
        contacting_pairs = set((heavy_atom_coords_indexed_residues[idx1], heavy_atom_coords_indexed_residues[idx2])
                               for idx2, contacts in enumerate(query) for idx1 in contacts)
        # Residue.contact_order starts as 0., so we are adding any observation to that attribute
        for residue1, residue2 in contacting_pairs:
            residue_sequence_distance = abs(residue1.number - residue2.number)
            if residue_sequence_distance >= sequence_distance_cutoff:
                residue1.contact_order += residue_sequence_distance

        number_residues = len(residues)
        for residue in residues:
            residue.contact_order /= number_residues

        return [residue.contact_order for residue in residues]

    # # this method uses 12 A cb - cb heuristic
    # def contact_order_per_residue(self, sequence_distance_cutoff: int = 2, distance: float = 12.) -> list[float]:
    #     """Calculate the contact order on a per-residue basis using CB - CB contacts
    #
    #     Args:
    #         sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
    #         distance: The distance in angstroms to measure atomic contact distances in contact
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
    #         [(residues[idx1], residues[idx2]) for idx2, contacts in enumerate(query) for idx1 in contacts]
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

    def format_resfile_from_directives(self, residue_directives, include=None, background=None, **kwargs):
        """Format Residue mutational potentials given Residues/residue numbers and corresponding mutation directive.
        Optionally, include specific amino acids and limit to a specific background. Both dictionaries accessed by same
        keys as residue_directives

        Args:
            residue_directives (dict[mapping[Residue | int],str]): {Residue object: 'mutational_directive', ...}
        Keyword Args:
            include=None (dict[mapping[Residue | int],set[str]]):
                Include a set of specific amino acids for each residue
            background=None (dict[mapping[Residue | int],set[str]]):
                The background amino acids to compare possibilities against
            special=False (bool): Whether to include special residues
        Returns:
            (list[str]): Formatted resfile lines for each Residue with a PIKAA and amino acid type string
        """
        if not background:
            background = {}
        if not include:
            include = {}

        res_file_lines = []
        if isinstance(next(iter(residue_directives)), int):  # this isn't a residue object, instead residue numbers
            for residue_number, directive in residue_directives.items():
                residue = self.residue(residue_number)
                allowed_aas = residue. \
                    mutation_possibilities_from_directive(directive, background=background.get(residue_number),
                                                          **kwargs)
                allowed_aas = {protein_letters_3to1_extended[aa] for aa in allowed_aas}
                allowed_aas = allowed_aas.union(include.get(residue_number, {}))
                res_file_lines.append('%d %s PIKAA %s' % (residue.number, residue.chain, ''.join(sorted(allowed_aas))))
                # res_file_lines.append('%d %s %s' % (residue.number, residue.chain,
                #                                     'PIKAA %s' % ''.join(sorted(allowed_aas)) if len(allowed_aas) > 1
                #                                     else 'NATAA'))
        else:
            for residue, directive in residue_directives.items():
                allowed_aas = residue. \
                    mutation_possibilities_from_directive(directive, background=background.get(residue), **kwargs)
                allowed_aas = {protein_letters_3to1_extended[aa] for aa in allowed_aas}
                allowed_aas = allowed_aas.union(include.get(residue, {}))
                res_file_lines.append('%d %s PIKAA %s' % (residue.number, residue.chain, ''.join(sorted(allowed_aas))))
                # res_file_lines.append('%d %s %s' % (residue.number, residue.chain,
                #                                     'PIKAA %s' % ''.join(sorted(allowed_aas)) if len(allowed_aas) > 1
                #                                     else 'NATAA'))

        return res_file_lines

    def make_resfile(self, residue_directives, out_path=os.getcwd(), header=None, **kwargs):
        """Format a resfile for the Rosetta Packer from Residue mutational directives

        Args:
            residue_directives (dict[Residue | int, str]): {Residue/int: 'mutational_directive', ...}
        Keyword Args:
            out_path=os.getcwd() (str): Directory to write the file
            header=None (list[str]): A header to constrain all Residues for packing
            include=None (dict[Residue | int, set[str]]):
                Include a set of specific amino acids for each residue
            background=None (dict[Residue | int, set[str]]):
                The background amino acids to compare possibilities against
            special=False (bool): Whether to include special residues
        Returns:
            (str): The path to the resfile
        """
        residue_lines = self.format_resfile_from_directives(residue_directives, **kwargs)
        res_file = os.path.join(out_path, '%s.resfile' % self.name)
        with open(res_file, 'w') as f:
            # format the header
            f.write('%s\n' % ('\n'.join(header + ['start']) if header else 'start'))
            # start the body
            f.write('%s\n' % '\n'.join(residue_lines))

        return res_file

    # def read_secondary_structure(self, filename=None, source='stride'):
    #     if source == 'stride':
    #         secondary_structure = self.parse_stride(filename)
    #     elif source == 'dssp':
    #         secondary_structure = None
    #     else:
    #         raise DesignError('Must pass a source to %s' % Structure.read_secondary_structure.__name__)
    #
    #     return secondary_structure
    def set_b_factor_data(self, dtype=None):
        """Set the b-factor entry for every Residue to a Residue attribute

        Keyword Args:
            dtype=None (str): The attribute of interest
        """
        # kwargs = dict(b_factor=dtype)
        # self._residues.set_attributes(b_factor=dtype)  # , **kwargs)
        self.set_residues_attributes(b_factor=dtype)  # , **kwargs)

    def _copy_structure_containers(self):  # Todo what about Structures() use. change mechanism
        """Copy all member Structures that reside in Structure containers"""
        # self.log.debug('In Structure copy_structure_containers()')
        for structure_type in self.structure_containers:
            structures = getattr(self, structure_type)
            for idx, structure in enumerate(structures):
                structures[idx] = copy(structure)

    def __copy__(self) -> Structure:  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        # Copy each of the key value pairs to the new, other dictionary
        for attr, obj in self.__dict__.items():
            if attr in parent_attributes:
                # Perform shallow copy on these attributes. They will be handled correctly below
                other.__dict__[attr] = obj
            else:  # Perform a deeper copy
                other.__dict__[attr] = copy(obj)

        if self.is_parent():  # This Structure is the parent, it's copy should be too
            # Set the copying Structure attribute ".spawn" to indicate to dependents the "other" of this copy
            self.spawn = other
            for attr in parent_attributes:
                other.__dict__[attr] = copy(self.__dict__[attr])
            # Todo comment out when Atom.__copy__ needed somewhere. Maybe used in Atoms.__copy__
            other._atoms.set_attributes(_parent=other)
            # other._residues.set_attributes(_parent=other)
            other._copy_structure_containers()
            other._update_structure_container_attributes(_parent=other)
            # Remove the attribute spawn after other Structure containers are copied
            del self.spawn
        else:  # This Structure is a dependent
            try:  # If initiated by the parent, this Structure's copy should be a dependent too
                other._parent = self.parent.spawn
            except AttributeError:  # Copy was not initiated by the parent, set this Structure as parent
                self.log.debug(f'The copied {type(self).__name__} {self.name} is being set as a parent. '
                               f'It was a dependent previously')
                other.detach_from_parent()
                other._copy_structure_containers()
                other._update_structure_container_attributes(_parent=other)

        return other

    # Todo this isn't long term sustainable. Perhaps a better case would be the ._sequence
    def __key(self) -> tuple[str, int, ...]:
        return self.name, *self._residue_indices

    def __eq__(self, other: Structure) -> bool:
        if isinstance(other, Structure):
            return self.__key() == other.__key()
        raise NotImplementedError(f'Can\' compare {type(self).__name__} instance to {type(other).__name__} instance')

    # Must define __hash__ in all subclasses that define an __eq__
    def __hash__(self) -> int:
        return hash(self.__key())

    def __str__(self) -> str:
        return self.name


try:
    reference_residues = unpickle(reference_residues_pkl)  # 0 indexed, 1 letter aa, alphabetically sorted at the origin
    reference_aa = Structure.from_residues(residues=reference_residues)
except Exception as error:  # If something goes wrong, we should remake this
    # logger.critical('The reference residues are out of date and need to be regenerated. '
    #                 'Please execute pickle_structure_dependencies.py')
    raise error  # catching in pickle_structure_dependencies.py


class Structures(Structure, UserList):
    # Todo mesh inheritance of both  Structure and UserClass...
    #  FROM set_residues_attributes in Structure, check all Structure attributes and methods that could be in conflict
    #  are all concatenated Structure methods and attributes accounted for?
    #  ensure UserList .append(), .extend() etc. are allowed and work as intended or overwrite them
    """Keep track of groups of Structure objects

    Pass the parent Structure with parent= to initialize .log, .coords, .atoms, and .residues

    Args:
        structures: The Iterable of Structure to set the Structures with
        dtype: If an empty Structures, tee specific subclass of Structure that Structures contains
    """
    data: list[Structure]
    dtype: str  # the type of Structure in instance

    def __init__(self, structures: Iterable[Structure], dtype: str = None, **kwargs):
        super().__init__(initlist=structures, **kwargs)  # initlist sets UserList.data to Iterable[Structure]

        # Todo should Structures be allowed to be a parent...
        if not self.data:  # set up an empty Structures
            self.dtype = dtype if dtype else 'Structure'
        elif all([True if isinstance(structure, Structure) else False for structure in self]):
            # self.data = [structure for structure in structures]
            self._atom_indices = []
            for structure in self:
                self._atom_indices.extend(structure.atom_indices)
            self._residue_indices = []
            for structure in self:
                self._residue_indices.extend(structure.residue_indices)

            self.dtype = dtype if dtype else type(self.data[0]).__name__
        else:
            raise ValueError(f'Can\'t set {type(self).__name__} by passing '
                             f'{", ".join(type(structure) for structure in self)}, must set with type [Structure, ...]'
                             f'or an empty constructor. Ex: Structures()')

        # overwrite attributes in Structure
        try:
            self.name = f'{self.parent.name}-{self.dtype}_{Structures.__name__}'
        except AttributeError:  # if not .parent.name
            self._name = f'{"-".join(structure.name for structure in self)}_{Structures.__name__}'

    # this setter would work in a world where Structures has it's own .coords, .atoms, and .residues
    # @StructureBase._parent.setter
    # def _parent(self, parent: StructureBase):
    #     """Set the Coords object while propagating changes to symmetry "mate" chains"""
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
    def structures(self) -> list[Structure]:
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
    # def model_coords(self):  # TODO RECONCILE with coords, SymmetricModel, and State variation
    #     """Return a view of the modelled Coords. These may be symmetric if a SymmetricModel"""
    #     return self._model_coords.coords
    #
    # @model_coords.setter
    # def model_coords(self, coords):
    #     if isinstance(coords, Coords):
    #         self._model_coords = coords
    #     else:
    #         raise AttributeError(
    #             'The supplied coordinates are not of class Coords!, pass a Coords object not a Coords '
    #             'view. To pass the Coords object for a Strucutre, use the private attribute _coords')

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
    #         return self._atoms.atoms.tolist()
    #     except AttributeError:  # if not set by parent, try to set from each individual structure
    #         atoms = []
    #         # for structure in self.structures:
    #         for structure in self:
    #             atoms.extend(structure.atoms)
    #         self._atoms = Atoms(atoms)
    #         return self._atoms.atoms.tolist()
    #
    # @property
    # def residues(self):
    #     try:
    #         return self._residues.residues.tolist()
    #     except AttributeError:  # if not set by parent, try to set from each individual structure
    #         residues = []
    #         for structure in self:
    #             residues.extend(structure.residues)
    #         self._residues = Residues(residues)
    #         return self._residues.residues.tolist()

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
    #                 chain = next(available_chain_ids)
    #                 structure.write(file_handle=f, chain=chain)
    #                 c_term_residue = structure.c_terminal_residue
    #                 f.write('{:6s}{:>5d}      {:3s} {:1s}{:>4d}\n'.format('TER', c_term_residue.atoms[-1].number + 1,
    #                                                                       c_term_residue.type, chain,
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

    def __repr__(self) -> str:
        return f'<Structure.Structures object at {id(self)}>'

    # def __str__(self):
    #     return self.name

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Structure:
        yield from iter(self.data)

    def __getitem__(self, idx: int) -> Structure:
        return self.data[idx]


def parse_stride(stride_file, **kwargs):
    """From a Stride file, parse information for residue level secondary structure assignment

    Sets:
        self.secondary_structure
    """
    with open(stride_file, 'r') as f:
        stride_output = f.readlines()

    return ''.join(line[24:25] for line in stride_output if line[0:3] == 'ASG')
